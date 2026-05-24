# `genesis_vehicle/samples/`

Three minimal, self-contained sample scripts that exercise the SDK's core
API surface. All three depend only on the SDK itself and the bundled
`urdf/car_4w.urdf` — no external mesh files, no umbrella-level helpers.

| # | Script | Demonstrates |
|---|---|---|
| 1 | [`quickstart.py`](quickstart.py) | The minimum-viable SDK call pattern: preset → `add_vehicle` → `scene.build` → step loop. Open-loop forward drive for 5 s on flat ground. |
| 2 | [`slope_hold.py`](slope_hold.py) | `StaticFrictionLock` side-slope hold regression check. Brake-locked car on a tilted ground should not creep laterally. Used to verify the v0.5.7 stick-slip fix. |
| 3 | [`batched_rollout.py`](batched_rollout.py) | The batched `n_envs > 1` API for RL / MPPI. Spawn N cars in parallel, per-env random controls, measure per-step throughput. |
| 4 | [`road_loop.py`](road_loop.py) | **Multi-vehicle visual demo** — 4 vehicle kinds (FWD red sedan, RWD blue coupe, AWD green SUV, yellow 6-wheel truck), `--n_per_kind` each, all driving a circular track under constant Ackermann steering. Top-down camera frames the whole fleet. |
| 5 | [`perf_vectorization.py`](perf_vectorization.py) | **n_envs batching speedup benchmark.** Sweeps `n_envs ∈ [1, 4, 16, 64, 256, 1024]` (one fresh subprocess per measurement) and prints a scaling table showing per-env cost dropping from ~26 ms (single env) to < 1 ms (64+ envs). Use to gauge RL / MPPI throughput on your machine. |

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
