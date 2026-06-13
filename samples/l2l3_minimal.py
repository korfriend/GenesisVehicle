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
    from genesis_vehicle import (
        MultiVehiclePhysics, VehicleInputs, add_vehicle, car_4w_rwd_ackermann,
    )
    from genesis_vehicle.samples.road_loop import _car_urdf, _save_urdf

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")

    DT = 0.02
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=DT, substeps=10),
        rigid_options=gs.options.RigidOptions(dt=DT, enable_collision=True,
                                              enable_self_collision=False),
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=1.0))

    tmpdir = tempfile.mkdtemp(prefix="gv_l2l3_")
    urdf = _save_urdf(_car_urdf("car", (0.2, 0.4, 0.85, 1.0),
                                body_size=(4.5, 1.8, 0.7),
                                cabin_size=(2.2, 1.6, 0.4)), tmpdir, "car")

    K, N = args.k, args.n_envs

    # SAME cfg instance for all K → MultiVehiclePhysics groups them as one
    # "kind" and batches their compute together. (Different cfg objects would
    # become separate kinds — that's how you mix vehicle types in one scene.)
    cfg = car_4w_rwd_ackermann(urdf)

    # K vehicles lined up in +X, 12 m apart, in ONE world. They share the world
    # so they CAN collide (L2). scene.build(n_envs=N) then replicates the whole
    # world N times (L3).
    vehicles = []
    for k in range(K):
        ent, sensor, _ = add_vehicle(scene, urdf, car_4w_rwd_ackermann,
                                     pos=(k * 12.0, 0.0, 0.6))
        vehicles.append((ent, sensor, cfg))

    scene.build(n_envs=N)
    mphys = MultiVehiclePhysics(scene, vehicles, n_envs=N)

    # Per-vehicle inputs. Each of the K inputs may carry a shape-(N,) tensor
    # (a different value per scenario) OR a scalar (broadcast across all N).
    # Here: every car constant-throttle; lead car (k=0) brakes in scenario 0
    # only, to show per-(env,vehicle) control + interaction (rear cars close in).
    dev = gs.device
    drive = []
    for k in range(K):
        thr = torch.full((N,), 0.3, device=dev)
        brk = torch.zeros(N, device=dev)
        if k == 0:
            brk[0] = 1.0          # lead car brakes — only in scenario 0
            thr[0] = 0.0
        drive.append(VehicleInputs(throttle=thr, brake=brk, steer=0.0))

    def sync():
        if not args.cpu and torch.cuda.is_available():
            torch.cuda.synchronize()

    for _ in range(args.warmup):
        mphys.step(drive)
        scene.step()

    sync(); t0 = time.perf_counter()
    for _ in range(args.steps):
        mphys.step(drive)
        scene.step()
    sync(); wall = time.perf_counter() - t0

    total = K * N
    ms = wall / args.steps * 1000.0
    print(f"\nL2 × L3 minimal  |  K={K} interacting × N={N} scenarios = {total} vehicles")
    print(f"  {ms:.2f} ms/step   ({total} vehicles/step,  "
          f"{total * args.steps / wall:,.0f} vehicle-steps/s)")
    print(f"  per-vehicle: {ms * 1000.0 / total:.1f} µs/step")

    # Show that scenario 0 (lead braked) diverged from the others.
    x0 = mphys.kinds[0]._proto  # noqa: F841  (proto holds batched omega etc.)
    lead_ent = vehicles[0][0]
    pos = lead_ent.get_pos()
    if hasattr(pos, "cpu"):
        pos = pos.cpu().numpy()
    print(f"  lead-car x: scenario0(braked)={pos[0, 0]:6.2f} m  "
          f"scenario1(rolling)={pos[1, 0]:6.2f} m   "
          f"→ per-scenario control works")


if __name__ == "__main__":
    main()
