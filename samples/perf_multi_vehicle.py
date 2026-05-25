"""perf_multi_vehicle.py — per_vehicle vs multi_batched solver benchmark.

Sweeps both ``MultiVehiclePhysics`` (L2 batched) and the default
per-vehicle loop (one ``VehiclePhysics`` per entity) on the SAME
multi-vehicle scene (4 kinds: FWD / RWD / AWD / Truck), at increasing
``--n_per_kind`` values. Prints a scaling table.

Each measurement runs in a fresh subprocess (clean GPU state) — the same
pattern as ``perf_vectorization.py``.

NO ``--viewer`` flag — this is a benchmark. Camera rendering adds per-step
overhead that distorts the throughput numbers. For a visual demo of a
multi-vehicle fleet on a track, see ``samples/road_loop.py --viewer``
(which exercises the same `MultiVehiclePhysics` solver via
``--solver multi_batched``).

Default sweep: n_per_kind ∈ [1, 2, 4, 8]. With 4 kinds that's 4, 8, 16,
32 total vehicles. The per_vehicle solver is N Python `step()` calls;
the multi_batched solver groups by kind so it's 4 calls regardless of K
(each call batched over K_kind via `MultiVehicleKindPhysics`).

Expected reading
----------------
- At K = 1 (4 vehicles) the two solvers are similar — only 4 step calls
  either way, no batching headroom.
- As K grows the multi_batched advantage grows: the compute is shared
  across K vehicles per kind, while the per_vehicle loop scales linearly.
- Total speedup is bounded by Genesis's per-entity ``scene.step()`` cost,
  which is the same for both solvers. So you won't see 10× — typically
  5-15% in absolute terms, growing with K.

For pure throughput (no shared scene), use ``perf_vectorization.py``'s
n_envs scaling (L3) instead — 44× at n_envs=64.

Run
---
    python -m genesis_vehicle.samples.perf_multi_vehicle
    python -m genesis_vehicle.samples.perf_multi_vehicle --n_per_kind_list 2,4,8,16
    python -m genesis_vehicle.samples.perf_multi_vehicle --internal --solver multi_batched --n_per_kind 4
        (internal mode: one measurement, prints machine-parseable RESULT line)
"""

from __future__ import annotations

# Bootstrap: allow `python path/to/this_file.py` (and `python -m ...`) both.
import sys, pathlib
_SDK_PARENT = str(pathlib.Path(__file__).resolve().parents[2])
if _SDK_PARENT not in sys.path:
    sys.path.insert(0, _SDK_PARENT)

import argparse
import math
import os
import re
import subprocess
import tempfile
import time


# ---------------------------------------------------------------------------
# Internal mode — one measurement, prints `RESULT ...`
# ---------------------------------------------------------------------------

def _internal_run(solver: str, n_per_kind: int,
                  warmup: int, steps: int) -> None:
    import genesis as gs
    import torch
    from genesis_vehicle import (
        VehiclePhysics, MultiVehiclePhysics, VehicleInputs,
        car_4w_fwd_ackermann, car_4w_rwd_ackermann, car_4w_awd_ackermann,
        truck_6w_partial_ackermann,
    )
    from genesis_vehicle.scene_helpers import make_wheel_raycaster

    # Pull the same parametric URDF generators that road_loop uses, so the
    # benchmark exercises an identical fleet shape.
    from genesis_vehicle.samples.road_loop import (
        _car_urdf, _truck_urdf, _save_urdf,
    )

    KINDS = [
        ("FWD",   lambda: _car_urdf("car_fwd", (0.85, 0.15, 0.15, 1.0),
                                     body_size=(3.8, 1.6, 1.0),
                                     cabin_size=(2.0, 1.5, 0.5)),
                  car_4w_fwd_ackermann, 2.7, 4),
        ("RWD",   lambda: _car_urdf("car_rwd", (0.15, 0.35, 0.85, 1.0),
                                     body_size=(4.5, 1.8, 0.7),
                                     cabin_size=(2.2, 1.6, 0.4)),
                  car_4w_rwd_ackermann, 2.7, 4),
        ("AWD",   lambda: _car_urdf("car_awd", (0.10, 0.70, 0.20, 1.0),
                                     body_size=(4.3, 2.0, 1.4),
                                     cabin_size=(2.6, 1.8, 0.6)),
                  car_4w_awd_ackermann, 2.7, 4),
        ("Truck", lambda: _truck_urdf((0.95, 0.80, 0.15, 1.0)),
                  truck_6w_partial_ackermann, 3.6, 6),
    ]
    K_per_kind = n_per_kind
    N_TOTAL = K_per_kind * len(KINDS)

    tmpdir = tempfile.mkdtemp(prefix="gv_perf_multi_")
    urdf_paths = [_save_urdf(uf(), tmpdir, name.lower())
                  for (name, uf, _p, _wb, _nw) in KINDS]

    gs.init(backend=gs.gpu, logging_level="warning")
    DT = 0.02
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=10),
        rigid_options=gs.options.RigidOptions(
            dt=DT, enable_collision=True,
            enable_self_collision=False, enable_joint_limit=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(
        gs.morphs.Plane(pos=(0, 0, 0)),
        material=gs.materials.Rigid(friction=1.0),
    )

    # Vehicles spread on a wide grid so they don't collide.
    cfg_per_kind = [pf(urdf_paths[k], stability="control")
                    for k, (_n, _u, pf, _wb, _nw) in enumerate(KINDS)]
    vehicles = []
    for global_idx in range(N_TOTAL):
        kind_idx = global_idx % len(KINDS)
        row = global_idx // len(KINDS)
        col = kind_idx
        pos = (col * 8.0 - 12.0, row * 8.0 - K_per_kind * 4.0, 1.0)
        morph = gs.morphs.URDF(file=urdf_paths[kind_idx], pos=pos)
        ent = scene.add_entity(morph, material=gs.materials.Rigid(friction=1.0))
        sens = make_wheel_raycaster(scene, ent, urdf_paths[kind_idx])
        vehicles.append((ent, sens, cfg_per_kind[kind_idx]))

    scene.build(n_envs=1)

    if solver == "per_vehicle":
        physics_objs = [VehiclePhysics(scene, e, s, c, n_envs=1)
                        for (e, s, c) in vehicles]
        def step_all(inputs_list):
            for ph, inp in zip(physics_objs, inputs_list):
                ph.step(inp)
    else:  # multi_batched
        mphys = MultiVehiclePhysics(scene, vehicles)
        def step_all(inputs_list):
            mphys.step(inputs_list)

    # Inputs: gentle constant throttle, mild constant steer (just to exercise
    # the steering strategy code path).
    drive_in = [VehicleInputs(throttle=0.3, brake=0.0, steer=0.0)
                for _ in range(N_TOTAL)]

    # Settle on brake.
    settle_in = [VehicleInputs(throttle=0.0, brake=1.0, steer=0.0)
                 for _ in range(N_TOTAL)]
    for _ in range(int(1.0 / DT)):
        step_all(settle_in)
        scene.step()

    # Warmup with drive inputs (kernel compile + visual sync init).
    for _ in range(warmup):
        step_all(drive_in)
        scene.step()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step_all(drive_in)
        scene.step()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    ms_per_step = wall / steps * 1000.0
    veh_steps_per_s = N_TOTAL * steps / wall
    print(f"RESULT solver={solver} n_per_kind={K_per_kind} "
          f"n_total={N_TOTAL} ms_per_step={ms_per_step:.4f} "
          f"vehicle_steps_per_s={veh_steps_per_s:.2f}")


# ---------------------------------------------------------------------------
# Parent — spawn one subprocess per (solver, n_per_kind), aggregate
# ---------------------------------------------------------------------------

_RESULT_RE = re.compile(
    r"RESULT solver=(\w+) n_per_kind=(\d+) n_total=(\d+) "
    r"ms_per_step=([\d.]+) vehicle_steps_per_s=([\d.]+)"
)


def _run_one(solver: str, n_per_kind: int,
             warmup: int, steps: int) -> tuple[float, float] | None:
    cmd = [
        sys.executable, "-m", "genesis_vehicle.samples.perf_multi_vehicle",
        "--internal", "--solver", solver, "--n_per_kind", str(n_per_kind),
        "--warmup", str(warmup), "--steps", str(steps),
    ]
    print(f"  [solver={solver:>13}  n_per_kind={n_per_kind:>3}]  "
          f"spawning subprocess...", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = _SDK_PARENT + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if out.returncode != 0:
        print(f"    FAILED (exit {out.returncode})")
        for line in out.stderr.strip().splitlines()[-5:]:
            print(f"    stderr: {line}")
        return None
    m = _RESULT_RE.search(out.stdout)
    if not m:
        print(f"    NO RESULT line — last stdout lines:")
        for line in out.stdout.strip().splitlines()[-5:]:
            print(f"    stdout: {line}")
        return None
    ms = float(m.group(4))
    vps = float(m.group(5))
    print(f"    {ms:8.2f} ms/step   |   {vps:>10,.0f} vehicle-steps/s")
    return ms, vps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_kind_list", default="1,2,4,8",
                    help="Comma-separated K values (per kind). 4 kinds → total = 4·K.")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--steps", type=int, default=120)
    # internal-mode args
    ap.add_argument("--internal", action="store_true")
    ap.add_argument("--solver", default="per_vehicle",
                    choices=["per_vehicle", "multi_batched"])
    ap.add_argument("--n_per_kind", type=int, default=4)
    args = ap.parse_args()

    if args.internal:
        _internal_run(args.solver, args.n_per_kind, args.warmup, args.steps)
        return

    n_list = [int(x) for x in args.n_per_kind_list.split(",") if x.strip()]
    print(f"genesis_vehicle perf_multi_vehicle — solver comparison")
    print(f"  n_per_kind values : {n_list}   (4 kinds — total = 4·K)")
    print(f"  warmup            : {args.warmup}   measured: {args.steps}")
    print(f"  scene             : 4 kinds × K vehicles on flat ground, "
          f"constant throttle 0.3\n")

    rows = []   # (n_per_kind, n_total, per_vehicle_ms, multi_batched_ms)
    for K in n_list:
        per_v = _run_one("per_vehicle",   K, args.warmup, args.steps)
        mul_b = _run_one("multi_batched", K, args.warmup, args.steps)
        if per_v is None or mul_b is None:
            print(f"  [skipping K={K} due to subprocess failure]")
            continue
        rows.append((K, 4 * K, per_v[0], mul_b[0]))

    if not rows:
        print("No successful measurements.")
        return

    print()
    print("=" * 88)
    print(f"=== SCALING COMPARISON  (4 kinds: FWD/RWD/AWD/Truck) ===")
    print(f"  {'n_per_kind':>10}  {'total_veh':>9}  "
          f"{'per_vehicle (ms)':>17}  {'multi_batched (ms)':>19}  {'speedup':>9}")
    print(f"  {'-'*10}  {'-'*9}  {'-'*17}  {'-'*19}  {'-'*9}")
    for K, n_tot, ms_pv, ms_mb in rows:
        speedup = ms_pv / ms_mb
        print(f"  {K:>10}  {n_tot:>9}  "
              f"{ms_pv:>17.2f}  {ms_mb:>19.2f}  {speedup:>8.2f}×")
    print("=" * 88)
    print()
    print("Reading:")
    print("  per_vehicle    — N independent VehiclePhysics, Python loop over step()")
    print("  multi_batched  — MultiVehiclePhysics groups by URDF/cfg, one batched compute per kind")
    print("  speedup        — per_vehicle / multi_batched (higher = batching helps more)")
    print()
    print("Both solvers share scene.step() (the dominant cost as N grows). The")
    print("speedup is bounded by per-vehicle Python + compute overhead only. For")
    print("pure throughput on one vehicle kind, n_envs > 1 (L3) beats this — see")
    print("`samples/perf_vectorization.py`.")


if __name__ == "__main__":
    main()
