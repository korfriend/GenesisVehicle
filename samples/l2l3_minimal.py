"""l2l3_minimal.py — the shortest L2 × L3 program.

K interacting vehicles share ONE Genesis world (they can collide — L2),
and that whole world is replicated across N parallel scenarios (L3). One
``MultiVehiclePhysics(scene, vehicles, n_envs=N)`` advances all K·N at once.

This is the "interacting traffic × N rollouts" pattern, stripped to the
essentials — for the full visual demo see ``city_traffic_ego.py``; for the
scaling benchmark see ``perf_l2_l3_combined.py``.

When to reach for this vs plain ``VehiclePhysics``:
  - 1 vehicle (optionally × N RL envs)         → VehiclePhysics(n_envs=N)
  - N identical, NON-interacting vehicles      → VehiclePhysics(n_envs=N)   (L3)
  - K vehicles that INTERACT in one world      → MultiVehiclePhysics        (L2)
  - K interacting × N parallel scenarios       → MultiVehiclePhysics(n_envs=N)  (L2 × L3)  ← this file
See ../docs/batching.md for the full decision matrix.

Run
---
    python -m genesis_vehicle.samples.l2l3_minimal                 # GPU, K=4, N=8
    python -m genesis_vehicle.samples.l2l3_minimal --k 6 --n_envs 16
    python -m genesis_vehicle.samples.l2l3_minimal --cpu --k 3 --n_envs 2
"""

from __future__ import annotations

import sys
import pathlib
_SDK_PARENT = str(pathlib.Path(__file__).resolve().parents[2])
if _SDK_PARENT not in sys.path:
    sys.path.insert(0, _SDK_PARENT)

import argparse
import tempfile
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4, help="interacting vehicles per scenario (L2)")
    ap.add_argument("--n_envs", type=int, default=8, help="parallel scenarios (L3)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import genesis as gs
    import torch
    from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann
    from genesis_vehicle.samples.road_loop import _car_urdf, _save_urdf

    K, N = args.k, args.n_envs
    DT = 0.02

    # VehicleScene owns gs.init / the scene / build / step. solver="batched" (the
    # default) groups the K same-kind vehicles into ONE batched compute (this IS
    # MultiVehiclePhysics under the hood); n_envs=N replicates the world N× (L3).
    vs = VehicleScene(
        n_envs=N, backend="cpu" if args.cpu else "gpu", raycast_mode="single_scene",
        dt=DT, substeps=10,
        rigid_options=gs.options.RigidOptions(dt=DT, enable_collision=True,
                                              enable_self_collision=False),
    )
    vs.add_ground_plane(friction=1.0)

    tmpdir = tempfile.mkdtemp(prefix="gv_l2l3_")
    urdf = _save_urdf(_car_urdf("car", (0.2, 0.4, 0.85, 1.0),
                                body_size=(4.5, 1.8, 0.7),
                                cabin_size=(2.2, 1.6, 0.4)), tmpdir, "car")

    # K vehicles lined up in +X, 12 m apart, in ONE world so they CAN collide (L2).
    # Same preset → one batched kind (auto-grouped). n_envs=N replicates them (L3).
    vehs = [vs.add_vehicle(urdf, preset=car_4w_rwd_ackermann,
                           morph=gs.morphs.URDF(file=urdf, pos=(k * 12.0, 0.0, 0.6),
                                                fixed=False, align=False))
            for k in range(K)]
    vs.build()

    # Per-(scenario, vehicle) control: every car constant-throttle; the lead car
    # (k=0) brakes in scenario 0 ONLY — a shape-(N,) tensor = a value per scenario.
    # set_inputs persists, so we set it once before the loops.
    dev = gs.device
    for k, veh in enumerate(vehs):
        thr = torch.full((N,), 0.3, device=dev)
        brk = torch.zeros(N, device=dev)
        if k == 0:
            brk[0] = 1.0          # lead car brakes — only in scenario 0
            thr[0] = 0.0
        veh.set_inputs(throttle=thr, brake=brk, steer=0.0)

    def sync():
        if not args.cpu and torch.cuda.is_available():
            torch.cuda.synchronize()

    for _ in range(args.warmup):
        vs.step()

    sync(); t0 = time.perf_counter()
    for _ in range(args.steps):
        vs.step()
    sync(); wall = time.perf_counter() - t0

    total = K * N
    ms = wall / args.steps * 1000.0
    print(f"\nL2 × L3 minimal  |  K={K} interacting × N={N} scenarios = {total} vehicles")
    print(f"  {ms:.2f} ms/step   ({total} vehicles/step,  "
          f"{total * args.steps / wall:,.0f} vehicle-steps/s)")
    print(f"  per-vehicle: {ms * 1000.0 / total:.1f} µs/step   "
          f"(solver={vs.solver}, {vs.physics.n_kinds} kind)")

    # scenario 0 (lead braked) diverges from the others.
    pos = vehs[0].get_pos()
    if hasattr(pos, "cpu"):
        pos = pos.cpu().numpy()
    print(f"  lead-car x: scenario0(braked)={pos[0, 0]:6.2f} m  "
          f"scenario1(rolling)={pos[1, 0]:6.2f} m   "
          f"→ per-scenario control works")


if __name__ == "__main__":
    main()
