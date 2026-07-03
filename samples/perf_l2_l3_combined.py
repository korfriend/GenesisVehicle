"""perf_l2_l3_combined.py — L2 (cross-vehicle) × L3 (cross-env) scaling.

The SDK has three orthogonal batching axes:

  L1 — wheels of ONE vehicle (batched since v0.5.0; transparent)
  L2 — K vehicles of the same kind in ONE Genesis env (MultiVehiclePhysics)
  L3 — N parallel Genesis envs (scene.build(n_envs=N))

v0.5.14 made L2 and L3 stack: ``MultiVehicleKindPhysics(..., n_envs=N)``
runs ``N × K`` total vehicles in a single batched compute call (the
"parallel-scenarios-with-traffic" use case: each MPPI scenario has K
vehicles, you want N parallel scenarios for sample-efficient rollouts).

This script sweeps a 2D grid of ``(N, K)`` and times each cell in a
fresh subprocess (clean GPU state per row).

NO ``--viewer`` flag — this is a benchmark. Camera rendering adds per-step
overhead that distorts the throughput numbers. For a VISUAL L2 × L3 demo
(ego + traffic across N parallel scenarios), see
``samples/city_traffic_ego.py --viewer --n_envs N``.

The headline result:

- L1 alone: K=1, N=1 → 1 vehicle, baseline ms/step
- L2 only:  K=4, N=1 → 4 vehicles same scene, batched compute
- L3 only:  K=1, N=4 → 1 vehicle, 4 parallel envs (existing pattern)
- L2+L3:    K=4, N=4 → 16 total in one batched call
- Push:     K=4, N=64 → 256 total in one batched call

Run
---
    python -m genesis_vehicle.samples.perf_l2_l3_combined
    python -m genesis_vehicle.samples.perf_l2_l3_combined --grid 1,4:1,4,16,64
        (--grid <K_list>:<N_list>, comma-separated)
    python -m genesis_vehicle.samples.perf_l2_l3_combined --internal --K 4 --N 16
        (internal: one cell, machine-parseable stdout)
"""

from __future__ import annotations

# Bootstrap: allow `python path/to/this_file.py` (and `python -m ...`) both.
import sys, pathlib
_SDK_PARENT = str(pathlib.Path(__file__).resolve().parents[2])
if _SDK_PARENT not in sys.path:
    sys.path.insert(0, _SDK_PARENT)

import argparse
import os
import re
import subprocess
import time


# ---------------------------------------------------------------------------
# Internal mode — one (K, N) measurement
# ---------------------------------------------------------------------------

def _internal_run(K: int, N: int, warmup: int, steps: int, gpu: bool = False) -> None:
    import torch, genesis as gs
    from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann

    URDF = os.path.join(os.path.dirname(__file__), "urdf", "car_4w.urdf")

    DT = 0.02
    # VehicleScene owns gs.init / scene / build / step. Default solver="batched"
    # groups the K same-kind vehicles into ONE batched compute (L2); n_envs=N
    # replicates the world N× (L3) — so this measures the combined L2 × L3 path.
    VehicleScene.init_backend("gpu" if gpu else "cpu")
    vs = VehicleScene(
        n_envs=N, raycast_mode="single_scene", dt=DT, substeps=10,
        rigid_options=gs.options.RigidOptions(
            dt=DT, enable_collision=True,
            enable_self_collision=False, enable_joint_limit=True,
        ),
    )
    vs.add_ground_plane(friction=1.0)

    # K vehicles spread along X (same preset → one batched kind); n_envs=N replicates.
    vehs = [vs.add_vehicle(URDF, preset=car_4w_rwd_ackermann, stability="control",
                           pos=(k * 6.0 - (K - 1) * 3.0, 0.0, 1.0),
                           material=gs.materials.Rigid(friction=1.0))
            for k in range(K)]
    vs.build()
    device = torch.device("cuda" if gpu else "cpu")

    # Per-vehicle, per-env random inputs (each env_i different to keep the compute
    # honest — no broadcasting freebies). Generated once, applied after settle.
    g = torch.Generator(device=device).manual_seed(0)
    drive = []
    for _ in range(K):
        throttle = 0.4 * torch.rand(N, generator=g, device=device)
        steer    = 0.3 * (torch.rand(N, generator=g, device=device) - 0.5)
        brake    = torch.zeros(N, device=device)
        drive.append((throttle, brake, steer))

    # Settle on brake (uniform).
    for veh in vehs:
        veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
    for _ in range(int(1.0 / DT)):
        vs.step()

    # Apply the per-(vehicle, env) drive inputs (persist across steps).
    for veh, (thr, brk, st) in zip(vehs, drive):
        veh.set_inputs(throttle=thr, brake=brk, steer=st)
    for _ in range(warmup):
        vs.step()

    if gpu and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        vs.step()
    if gpu and torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    ms_per_step  = wall / steps * 1000.0
    total_batch  = N * K
    veh_steps_ps = total_batch * steps / wall

    print(f"RESULT K={K} N={N} total={total_batch} "
          f"ms_per_step={ms_per_step:.4f} veh_steps_per_s={veh_steps_ps:.2f}")


# ---------------------------------------------------------------------------
# Sweep mode
# ---------------------------------------------------------------------------

_RESULT_RE = re.compile(
    r"RESULT K=(\d+) N=(\d+) total=(\d+) "
    r"ms_per_step=([\d.]+) veh_steps_per_s=([\d.]+)"
)


def _run_cell(K: int, N: int, warmup: int, steps: int,
              gpu: bool = False) -> tuple[float, float] | None:
    cmd = [sys.executable, "-m", "genesis_vehicle.samples.perf_l2_l3_combined",
           "--internal", "--K", str(K), "--N", str(N),
           "--warmup", str(warmup), "--steps", str(steps)]
    if gpu:
        cmd.append("--gpu")
    print(f"  [K={K:>3} N={N:>4}  total={K*N:>5}]  spawning...", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = _SDK_PARENT + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if out.returncode != 0:
        print(f"    FAILED (exit {out.returncode})")
        for line in out.stderr.strip().splitlines()[-3:]:
            print(f"    stderr: {line}")
        return None
    m = _RESULT_RE.search(out.stdout)
    if not m:
        print(f"    NO RESULT — stdout:")
        for line in out.stdout.strip().splitlines()[-3:]:
            print(f"    stdout: {line}")
        return None
    ms = float(m.group(4))
    vps = float(m.group(5))
    print(f"    {ms:7.2f} ms/step   |   {vps:>10,.0f} veh-steps/s")
    return ms, vps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="1,4:1,4,16,64",
                    help="Sweep grid as '<K_list>:<N_list>' (comma-separated). "
                         "Default: K∈[1,4]  N∈[1,4,16,64].")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--steps",  type=int, default=120)
    # internal-mode
    ap.add_argument("--internal", action="store_true")
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--N", type=int, default=1)
    ap.add_argument("--gpu", action="store_true",
                    help="Opt into the GPU backend (default: CPU). GPU only wins "
                         "at large K×N (hundreds of batched vehicles).")
    args = ap.parse_args()

    if args.internal:
        _internal_run(args.K, args.N, args.warmup, args.steps, gpu=args.gpu)
        return

    K_part, N_part = args.grid.split(":")
    Ks = [int(x) for x in K_part.split(",") if x.strip()]
    Ns = [int(x) for x in N_part.split(",") if x.strip()]
    print(f"genesis_vehicle perf_l2_l3_combined  —  K ∈ {Ks},  N ∈ {Ns}")
    print(f"  warmup={args.warmup}  steps={args.steps}   "
          f"({len(Ks)} × {len(Ns)} = {len(Ks)*len(Ns)} cells)\n")

    results = {}      # (K, N) → (ms, vps)
    for K in Ks:
        for N in Ns:
            r = _run_cell(K, N, args.warmup, args.steps, gpu=args.gpu)
            if r is not None:
                results[(K, N)] = r

    if not results:
        print("\nNo successful cells.")
        return

    # Find baseline (smallest K, smallest N).
    base = min(results)
    base_ms = results[base][0]

    print()
    print("=" * 86)
    print(f"=== L2 × L3 SCALING TABLE  (baseline cell K={base[0]} N={base[1]}) ===")
    print(f"  {'K':>4}  {'N':>5}  {'total':>6}  {'ms/step':>10}  "
          f"{'per veh (μs)':>13}  {'veh-steps/s':>14}  {'gain':>9}")
    print(f"  {'-'*4}  {'-'*5}  {'-'*6}  {'-'*10}  {'-'*13}  {'-'*14}  {'-'*9}")
    for (K, N) in sorted(results):
        ms, vps = results[(K, N)]
        per_veh_us = ms / (K * N) * 1000.0
        # "gain": batching speedup. 1× = linear scaling (no batching benefit
        # — ms grows proportional to total batch). > 1× = real batching gain
        # (ms grows slower than batch).
        gain = (base_ms / ms) * ((K * N) / (base[0] * base[1]))
        print(f"  {K:>4}  {N:>5}  {K*N:>6}  {ms:>10.2f}  "
              f"{per_veh_us:>13.1f}  {vps:>14,.0f}  {gain:>8.1f}×")
    print("=" * 86)
    print()
    print("Reading:")
    print("  total       N × K — vehicles processed per simulation step.")
    print("  per veh     ms / total*1000 — amortized cost per vehicle. Smaller = more batching benefit.")
    print("  gain        Throughput vs baseline, normalized to 1× = linear scaling")
    print("              (i.e., 'no batching benefit'). Numbers > 1× are the actual speedup.")
    print()
    print("With K=1, N>1: pure L3 (parallel-env batching) — same as perf_vectorization.")
    print("With K>1, N=1: pure L2 (cross-vehicle in one env) — same as perf_multi_vehicle.")
    print("With K>1, N>1: L2 × L3 combined — for parallel scenarios that each contain")
    print("              multiple vehicles (autonomous-driving ego + traffic, multi-")
    print("              agent RL, MPPI candidate fleets in different ego states).")


if __name__ == "__main__":
    main()
