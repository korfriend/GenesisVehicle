# `genesis_vehicle/samples/`

Three minimal, self-contained sample scripts that exercise the SDK's core
API surface. All three depend only on the SDK itself and the bundled
`urdf/car_4w.urdf` — no external mesh files, no umbrella-level helpers.

| # | Script | Demonstrates | viewer |
|---|---|---|---|
| 1 | [`quickstart.py`](quickstart.py) | The minimum-viable SDK call pattern: preset → `add_vehicle` → `scene.build` → step loop. Open-loop forward drive for 5 s on flat ground. | ✓ `--viewer` |
| 2 | [`slope_hold.py`](slope_hold.py) | `StaticFrictionLock` side-slope hold regression check. Brake-locked car on a tilted ground should not creep laterally. Used to verify the v0.5.7 stick-slip fix. | ✓ `--viewer` |
| 3 | [`batched_rollout.py`](batched_rollout.py) | The batched `n_envs > 1` API for RL / MPPI. Spawn N cars in parallel, per-env random controls, measure per-step throughput. | ✓ `--viewer` (grid) |
| 4 | [`road_loop.py`](road_loop.py) | **Multi-vehicle visual demo** — 4 vehicle kinds (FWD red sedan, RWD blue coupe, AWD green SUV, yellow 6-wheel truck), `--n_per_kind` each, all driving a circular track under constant Ackermann steering. Top-down camera frames the whole fleet. `--solver multi_batched` switches from N independent `VehiclePhysics` calls to one `MultiVehiclePhysics` that batches the compute per kind (~10% faster on a 16-vehicle scene). | ✓ `--viewer` |
| 5 | [`perf_vectorization.py`](perf_vectorization.py) | **n_envs batching speedup benchmark.** Sweeps `n_envs ∈ [1, 4, 16, 64, 256, 1024]` (one fresh subprocess per measurement) and prints a scaling table showing per-env cost dropping from ~26 ms (single env) to < 1 ms (64+ envs). Use to gauge RL / MPPI throughput on your machine. | ✗ headless by design |
| 6 | [`multi_env_render.py`](multi_env_render.py) | **Render every parallel env in one grid view.** Uses Genesis's `env_separate_rigid=True` + `env_spacing` so `n_envs > 1` parallel rollouts are laid out in a `√n × √n` grid (physics still overlapping, only visualization offset). Each env gets a different random throttle/steer — eyeball RL/MPPI diversity at a glance. | always renders |
| 7 | [`perf_multi_vehicle.py`](perf_multi_vehicle.py) | **L2 batching benchmark.** Compares `MultiVehiclePhysics` (groups by kind, batched compute per kind) against the default per-vehicle loop on the same 4-kind fleet, sweeping `--n_per_kind ∈ [1, 2, 4, 8]`. Subprocess per measurement. At K=1 multi_batched is slightly slower (batching overhead with nothing to batch); from K=2+ the gap widens — typically 1.1-1.2× faster. Use to decide which solver fits your multi-vehicle scene. | ✗ headless by design |
| 8 | [`perf_l2_l3_combined.py`](perf_l2_l3_combined.py) | **L2 × L3 combined scaling.** 2D grid sweep `(K, N)` showing that L2 (cross-vehicle in one env) and L3 (parallel envs) stack multiplicatively. e.g. K=2 alone gives 1.5×, N=4 alone gives 2.8×, K=2×N=4 combined gives **4.6×** vs single-vehicle baseline — close to the theoretical product. Subprocess per cell. The headline batching pattern for autonomous-driving ego + traffic in MPPI / multi-agent RL. | ✗ headless by design |
| 9 | [`city_traffic_ego.py`](city_traffic_ego.py) | **Autonomous-driving scenario — ego + traffic on a 4-lane highway.** 1 red AWD ego + 7 traffic agents (3 RWD blue coupes, 3 FWD small dark-red sedans, 1 yellow truck) under a simple P lane-keeper. Top-down camera frames the road. `--n_envs N` enables the L2 × L3 combined batching pattern (one MPPI candidate per env, all 8×N vehicles batched). The reference setup for "MPPI ego in surrounding traffic" workflows. | ✓ `--viewer` |

The three perf benchmarks (5, 7, 8) are intentionally headless — camera
rendering adds per-step overhead that distorts the throughput numbers
they're trying to measure. The visual equivalents are in their
respective docstring pointers (multi_env_render for L3, road_loop for
L2 with `--solver multi_batched`, city_traffic_ego for L2 × L3).

## Bundled asset

- [`urdf/car_4w.urdf`](urdf/car_4w.urdf) — 1.2-ton 4-wheel RWD sedan,
  primitive geometry only (no mesh files), follows project URDF
  conventions (steer axis `(0, 0, -1)`, FL/FR/RL/RR wheel order,
  chassis collision box bottom above wheel ray origin).

## Running

From any directory after `pip install genesis-vehicle` (or after
cloning):

```bash
python -m genesis_vehicle.samples.quickstart
python -m genesis_vehicle.samples.slope_hold
python -m genesis_vehicle.samples.slope_hold --slope 30
python -m genesis_vehicle.samples.batched_rollout --n_envs 256 --steps 200
python -m genesis_vehicle.samples.road_loop
python -m genesis_vehicle.samples.road_loop --n_per_kind 8 --duration 30
python -m genesis_vehicle.samples.perf_vectorization
python -m genesis_vehicle.samples.perf_vectorization --n_envs_list 1,4,16,64
python -m genesis_vehicle.samples.multi_env_render --n_envs 16
python -m genesis_vehicle.samples.multi_env_render --n_envs 64 --spacing 8
python -m genesis_vehicle.samples.perf_multi_vehicle
python -m genesis_vehicle.samples.perf_multi_vehicle --n_per_kind_list 2,4,8,16
python -m genesis_vehicle.samples.perf_l2_l3_combined
python -m genesis_vehicle.samples.perf_l2_l3_combined --grid 1,4:1,4,16,64
python -m genesis_vehicle.samples.city_traffic_ego --viewer
python -m genesis_vehicle.samples.city_traffic_ego --n_envs 16 --bench
```

All three are headless (no viewer, no `cv2`, no `pynput`). The chase-cam
HUD / keyboard-driven demos live one level up at the umbrella project
(`GeneVehicle_*/demo_drive.py`) and are not bundled with the SDK on
purpose — they have opencv-python + pynput dependencies and are more
demo than reference. Use these `samples/` scripts as the reference for
what the SDK looks like from a caller's perspective.

## Cross-references in docs

| Doc | Sample it references |
|---|---|
| [`../docs/quickstart.md`](../docs/quickstart.md) | `quickstart.py` |
| [`../docs/stability-profiles.md`](../docs/stability-profiles.md) | `slope_hold.py` |
| [`../docs/api-reference.md`](../docs/api-reference.md) | `quickstart.py`, `slope_hold.py`, `batched_rollout.py`, `road_loop.py`, `perf_vectorization.py` |
