"""perf_vectorization.py — measure n_envs batching speedup.

Times the SDK at increasing ``n_envs`` (parallel-universe batching) and
prints a scaling table. Genesis builds one scene per ``gs.init`` call,
so this script spawns a fresh subprocess per measurement to keep GPU
state clean across configurations.

Default sweep
-------------
    n_envs ∈ [1, 4, 16, 64, 256, 1024]

A linear scaling (each row 4× the previous) would mean a single env-step
costs the same regardless of batch size — i.e., GPU is throughput-bound
and batching is free. In practice the per-env cost drops sharply from
``n_envs = 1`` (kernel-launch-dominated) to ``n_envs = 64+`` (compute-
dominated), then plateaus. The ``per env`` column shows this drop.

Run
---
    python -m genesis_vehicle.samples.perf_vectorization
    python -m genesis_vehicle.samples.perf_vectorization --n_envs_list 1,16,64
    python -m genesis_vehicle.samples.perf_vectorization --internal --n_envs 64
        (internal mode: single measurement, machine-parseable stdout —
         used by the parent process during a sweep)
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# Internal mode — one measurement, machine-parseable output
# ---------------------------------------------------------------------------

def _internal_run(n_envs: int, warmup: int, steps: int) -> None:
    """Run the SDK in this process at the given n_envs, print one line:

        RESULT n_envs=<N> ms_per_step=<X> env_steps_per_s=<Y>
    """
    import torch
    import genesis as gs

    from genesis_vehicle import (
        VehiclePhysics, VehicleInputs,
        car_4w_rwd_ackermann, add_vehicle,
    )

    URDF = os.path.join(os.path.dirname(__file__), "urdf", "car_4w.urdf")

    gs.init(backend=gs.gpu, logging_level="warning")
    DT = 0.02
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=20),
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
    car, sensor, _ = add_vehicle(
        scene, URDF, preset_fn=None, pos=(0.0, 0.0, 1.0),
        material=gs.materials.Rigid(friction=1.0),
    )
    cfg = car_4w_rwd_ackermann(URDF, stability="control")
    scene.build(n_envs=n_envs)
    physics = VehiclePhysics(scene, car, sensor, cfg, n_envs=n_envs)
    device = car.get_pos().device

    # Per-env random throttle / steer (modest values, no extreme behavior).
    g = torch.Generator(device=device).manual_seed(0)
    throttle = 0.4 * torch.rand(n_envs, generator=g, device=device)
    steer    = 0.4 * (torch.rand(n_envs, generator=g, device=device) - 0.5)
    brake    = torch.zeros(n_envs, device=device)
    inputs   = VehicleInputs(throttle=throttle, brake=brake, steer=steer)

    # Settle on brake (uniform across envs).
    settle = VehicleInputs(throttle=0.0, brake=1.0, steer=0.0)
    for _ in range(int(1.0 / DT)):
        physics.step(settle)
        scene.step()

    # Warmup at the random per-env inputs (compiles kernels for current n_envs).
    for _ in range(warmup):
        physics.step(inputs)
        scene.step()

    # Measurement.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        physics.step(inputs)
        scene.step()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    ms_per_step      = wall / steps * 1000.0
    env_steps_per_s  = n_envs * steps / wall

    # Result line picked up by the parent process.
    print(f"RESULT n_envs={n_envs} ms_per_step={ms_per_step:.4f} "
          f"env_steps_per_s={env_steps_per_s:.2f}")


# ---------------------------------------------------------------------------
# Sweep mode — spawn one subprocess per n_envs value, aggregate
# ---------------------------------------------------------------------------

_RESULT_RE = re.compile(
    r"RESULT n_envs=(\d+) ms_per_step=([\d.]+) env_steps_per_s=([\d.]+)"
)


def _run_one(n_envs: int, warmup: int, steps: int) -> tuple[float, float] | None:
    """Spawn this script in --internal mode at the given n_envs.
    Returns (ms_per_step, env_steps_per_s) or None on failure."""
    cmd = [
        sys.executable, "-m", "genesis_vehicle.samples.perf_vectorization",
        "--internal", "--n_envs", str(n_envs),
        "--warmup", str(warmup), "--steps", str(steps),
    ]
    print(f"  [n_envs={n_envs:>5}]  spawning subprocess...", flush=True)
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        print(f"    FAILED (exit {out.returncode})")
        if out.stderr.strip():
            for line in out.stderr.strip().splitlines()[-5:]:
                print(f"    stderr: {line}")
        return None
    m = _RESULT_RE.search(out.stdout)
    if not m:
        print(f"    NO RESULT line in stdout — output:")
        for line in out.stdout.strip().splitlines()[-5:]:
            print(f"    stdout: {line}")
        return None
    ms = float(m.group(2))
    vps = float(m.group(3))
    print(f"    {ms:7.2f} ms/step   |   {vps:>10,.0f} env-steps/s")
    return ms, vps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_envs_list", default="1,4,16,64,256,1024",
                    help="Comma-separated n_envs values to sweep "
                         "(default: 1,4,16,64,256,1024).")
    ap.add_argument("--warmup", type=int, default=30,
                    help="Warmup steps per measurement (default 30).")
    ap.add_argument("--steps",  type=int, default=150,
                    help="Measured steps per measurement (default 150).")
    # internal-mode args (only meaningful with --internal)
    ap.add_argument("--internal", action="store_true",
                    help="Internal one-shot mode invoked by the parent process.")
    ap.add_argument("--n_envs",   type=int, default=1,
                    help="(internal) n_envs for this measurement.")
    args = ap.parse_args()

    if args.internal:
        _internal_run(args.n_envs, args.warmup, args.steps)
        return

    n_envs_list = [int(x) for x in args.n_envs_list.split(",") if x.strip()]
    print(f"genesis_vehicle perf_vectorization — n_envs sweep")
    print(f"  values  : {n_envs_list}")
    print(f"  warmup  : {args.warmup}  measured: {args.steps}")
    print(f"  scene   : 1 car (car_4w_rwd_ackermann), random per-env throttle/steer\n")

    results = {}
    for n in n_envs_list:
        r = _run_one(n, args.warmup, args.steps)
        if r is not None:
            results[n] = r

    if not results:
        print("\nNo successful measurements.")
        return

    base_ms = results[min(results)][0]
    base_n  = min(results)

    print()
    print("=" * 74)
    print(f"=== SCALING TABLE  (baseline = n_envs={base_n}) ===")
    print(f"  {'n_envs':>7}  {'ms/step':>10}  "
          f"{'env-steps/s':>14}  {'per env (μs)':>13}  {'speedup':>10}")
    print(f"  {'-'*7}  {'-'*10}  {'-'*14}  {'-'*13}  {'-'*10}")
    for n in sorted(results):
        ms, vps = results[n]
        per_env_us = ms / n * 1000.0
        speedup = vps / results[base_n][1] * (base_n / n)   # vs baseline per-env throughput
        # Simpler reading: how many envs we get for the cost of `base_n` envs at baseline.
        gain = base_ms / ms * n / base_n
        print(f"  {n:>7}  {ms:10.2f}  {vps:>14,.0f}  "
              f"{per_env_us:13.1f}  {gain:9.1f}×")
    print("=" * 74)
    print()
    print("Reading:")
    print("  ms/step       wall-clock cost of ONE simulation step at that n_envs.")
    print("                (one step = one physics + one scene.step call.)")
    print("  env-steps/s   throughput — N parallel envs × steps per wall-second.")
    print("  per env (μs)  amortized cost per env. Smaller = better batching.")
    print(f"  speedup       env-throughput vs n_envs={base_n} baseline, normalized")
    print(f"                so 1× means 'no batching benefit, linear in n_envs'.")
    print("                Values > 1 are the actual batching speedup.")


if __name__ == "__main__":
    main()
