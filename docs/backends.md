# Choosing the physics backend — CPU vs GPU

| abbr | meaning |
|---|---|
| L2 / L3 | K interacting entities × 1 env / 1 entity × n_envs (see [`batching.md`](batching.md)) |
| launch-bound | fixed per-kernel-call latency dominates over the compute itself |
| DtoH / HtoD | GPU→CPU / CPU→GPU transfer |
| crossover | the scale at which GPU becomes faster than CPU |
| K / N | interacting vehicles per env (L2 axis) / parallel envs (`n_envs`, L3 axis) |
| veh-steps/s | vehicle-steps per second of wall time (K × N × steps / s) |

## 1. Two independent "backends"

- **Physics (compute) backend** — where the solver + SDK tensor math run.
  **CPU is the default everywhere** (SDK, samples, both server modes) since
  v1.0.14; GPU is an explicit opt-in. Process-global and set ONCE:

  ```python
  VehicleScene.init_backend("gpu")   # BEFORE constructing any scene; default "cpu"
  ```
  ```bash
  python -m genesis_vehicle.server --multi-env --gpu   # server opt-in (both modes accept it)
  ```
  Requesting a *different* backend once one is up warns and is ignored
  (the backend cannot change within a process); requesting `"gpu"` without
  a working CUDA stack is a REAL init failure and propagates.

- **Renderer** — independent of the physics backend: the viewer/cameras
  rasterize on the GPU graphics stack regardless of where physics runs
  (falling back to slow software rendering only when no GPU is present).
  CPU-physics + GPU-rendering is the normal interactive combo.

## 2. Why CPU is the default

GPU kernel launches carry a fixed per-call latency (particularly high under
WSL2). A physics step issues dozens of kernels per substep regardless of
batch size, so at small batch widths the GPU pays that fixed cost without
enough parallel work to amortize it. The SDK's torch pipeline follows the
same backend, so small `(n_envs, n_wheels)` tensors on GPU are
launch-bound too.

## 3. Measured crossovers

### L2 and L3 separately (v1.1.6 — 10-wheel tanks, simple terrain, OSC server, ms/step)

| tanks | L2 CPU | L2 GPU | L3 CPU | L3 GPU |
|---|---|---|---|---|
| 30 | 15.7 | 109.5 | 12.1 | 14.5 |
| 100 | 31.1 | **684.0** | 17.4 | 15.6 |
| 200 | 73.2 | (impractical) | 27.4 | 31.5 |
| 400 | — | — | 46.2 | **27.8** |

- **L3 crosses over at ~250–300 envs**: CPU grows ~linearly with n_envs
  while GPU stays launch-bound-flat, so at 400 envs GPU wins 1.66×.
- **L2 has NO practical crossover — it anti-scales on GPU** (7× slower at
  30 tanks, 22× at 100). Growing K grows ONE env's system: every
  per-vehicle sensor read and solver stage pays launch/sync latency, and
  there is no env-axis batch width to amortize it. Link-level parallelism
  does not rescue it.

### L2 × L3 combined (v1.1.8 — `car_4w`, SDK-direct via [`perf_l2_l3_combined.py`](../samples/perf_l2_l3_combined.py), ms/step, **bold = winner**)

| CPU / GPU | N=1 | N=10 | N=50 | N=100 | N=300 |
|---|---|---|---|---|---|
| K=1 | **13.1** / 37.8 | **13.8** / 42.8 | **17.3** / 47.7 | **21.8** / 42.8 | **43.7** / 46.4 |
| K=2 | **15.9** / 40.3 | **18.6** / 46.3 | **29.0** / 48.1 | **40.6** / 49.8 | 105.0 / **49.6** |
| K=5 | **17.6** / 51.3 | **22.8** / 52.8 | **49.3** / 58.0 | 83.1 / **63.6** | 250.2 / **72.7** |
| K=10 | **18.3** / 59.2 | **29.3** / 69.5 | 85.6 / **76.2** | 165.7 / **94.3** | 509.7 / **123.7** |

- **The crossover invariant is TOTAL batched vehicles: K×N ≈ 300–400.**
  The crossover N shifts left as K grows (K=1 → N~310, K=2 → ~150,
  K=5 → ~65, K=10 → ~40) because CPU cost is ~linear in K·N while GPU
  stays launch-bound-flat in N; the K×N product at the crossover stays
  roughly constant. (Consistent with the L3 tank figure above — 10-wheel
  tanks do more work per vehicle, so they cross slightly earlier.)
- The GPU floor still rises mildly with K (N=1: 37.8 → 59.2 ms as K
  1 → 10 — the residual L2 anti-scaling term), so tiny-N configs stay CPU
  regardless of K.
- **Throughput ceilings differ 4×**: CPU saturates at ~6,000 veh-steps/s
  for ANY K/N mix, while GPU reaches 24,200 veh-steps/s at K=10×N=300
  (3,000 vehicles, 4.1×) and is still climbing — for MPPI/RL rollout
  volume, the GPU ceiling is what matters.

**Rules of thumb**

| workload | backend |
|---|---|
| Interacting vehicles (L2, `n_envs=1`), any count | **CPU, always** |
| Non-interacting fleet (L3) ≤ ~250 envs | **CPU** |
| Non-interacting fleet (L3) ~300+ envs | **GPU** (`--gpu` / `init_backend("gpu")`) |
| L2 × L3 (K interacting × N envs, e.g. MPPI/RL with traffic) | **GPU once K×N ≳ 300–400 total vehicles** (measured, table above); CPU below |

## 4. GPU serving architecture (OSC server specifics)

When the server runs on GPU, the serving loop is engineered so that **only
pure physics runs on the GPU; serving math runs on the CPU** (v1.1.1–1.1.3):

- capture downloads its raw reads (chassis pose, steer/dist/spin, obstacle
  poses) in ONE on-device concat + a single `.cpu()` DtoH sync
  (`_to_host_batched`), and computes the closed-form wheel poses **on the
  CPU** (`wheel_visual_transforms_host`) — no capture-side kernel launches;
- the L3 loop synchronizes INSIDE the physics timing, so `[STATS]` per-step
  reflects real GPU execution time (not launch time).

Measured steady state (L3×100 simple): CPU physics 16.2 / serving ~20.3;
GPU physics 21.7 / serving ~25.6 ms — the residual gap is GPU physics
itself plus per-step HtoD input uploads, not data volume. Full analysis in
[`server.md`](server.md) §2.1.

## 5. WSL2 notes

- In non-interactive WSL shells, GPU runs may need
  `export LD_LIBRARY_PATH=/usr/lib/wsl/lib` so libcuda resolves.
- WSL2 kernel-launch and DtoH-sync latencies are noticeably higher than
  native Linux — the crossover shifts LEFT (GPU earlier) on native/beefier
  hardware; re-run `python -m genesis_vehicle.server.benchmark --gpu` to
  locate it on yours.

## See also

- [`server.md`](server.md) §2.1 (official benchmark + full CPU/GPU tables) and §3 (mode selection)
- [`batching.md`](batching.md) (L1/L2/L3 axes and what each batches)
