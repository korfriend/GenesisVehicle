# `genesis_vehicle/samples/`

A set of minimal, self-contained sample scripts that exercise the SDK's core
API surface. They depend only on the SDK itself and the bundled assets
(`urdf/car_4w.urdf`, `urdf/tank_ray.urdf`, `data/tank_sweep_signed.csv` —
all primitive-geometry / plain CSV) — no external mesh files, no
umbrella-level helpers.

| # | Script | Demonstrates | viewer |
|---|---|---|---|
| 1 | [`quickstart.py`](quickstart.py) | The minimum-viable SDK call pattern through the unified **`VehicleScene`** API (no manual `gs.init`/`build`/`step`): `add_ground_plane` → `add_vehicle` → `build` → loop of `veh.set_inputs(...) + vs.step()`. Open-loop forward drive for 5 s on flat ground. | ✓ `--viewer` |
| 2 | [`slope_hold.py`](slope_hold.py) | `StaticFrictionLock` side-slope hold regression check (on **`VehicleScene`**; the tilted ground is an `add_static` body). Brake-locked car on a tilted ground should not creep laterally. Used to verify the v0.5.7 stick-slip fix. | ✓ `--viewer` |
| 3 | [`batched_rollout.py`](batched_rollout.py) | The batched `n_envs > 1` API for RL / MPPI. Spawn N cars in parallel, per-env random controls, measure per-step throughput. | ✓ `--viewer` (grid) |
| 4 | [`road_loop.py`](road_loop.py) | **Multi-vehicle visual demo** — 4 vehicle kinds (FWD red sedan, RWD blue coupe, AWD green SUV, yellow 6-wheel truck), `--n_per_kind` each, all driving a circular track under constant Ackermann steering. Top-down camera frames the whole fleet. `--solver multi_batched` switches from N independent `VehiclePhysics` calls to one `MultiVehiclePhysics` that batches the compute per kind (~10% faster on a 16-vehicle scene). | ✓ `--viewer` |
| 5 | [`perf_vectorization.py`](perf_vectorization.py) | **n_envs batching speedup benchmark.** Sweeps `n_envs ∈ [1, 4, 16, 64, 256, 1024]` (one fresh subprocess per measurement) and prints a scaling table showing per-env cost dropping from ~26 ms (single env) to < 1 ms (64+ envs). Use to gauge RL / MPPI throughput on your machine. | ✗ headless by design |
| 6 | [`multi_env_render.py`](multi_env_render.py) | **Render every parallel env in one grid view.** Uses Genesis's `env_separate_rigid=True` + `env_spacing` so `n_envs > 1` parallel rollouts are laid out in a `√n × √n` grid (physics still overlapping, only visualization offset). Each env gets a different random throttle/steer — eyeball RL/MPPI diversity at a glance. | always renders |
| 7 | [`perf_multi_vehicle.py`](perf_multi_vehicle.py) | **L2 batching benchmark.** Compares `MultiVehiclePhysics` (groups by kind, batched compute per kind) against the default per-vehicle loop on the same 4-kind fleet, sweeping `--n_per_kind ∈ [1, 2, 4, 8]`. Subprocess per measurement. At K=1 multi_batched is slightly slower (batching overhead with nothing to batch); from K=2+ the gap widens — typically 1.1-1.2× faster. Use to decide which solver fits your multi-vehicle scene. | ✗ headless by design |
| 8 | [`perf_l2_l3_combined.py`](perf_l2_l3_combined.py) | **L2 × L3 combined scaling.** 2D grid sweep `(K, N)` showing that L2 (cross-vehicle in one env) and L3 (parallel envs) stack multiplicatively. e.g. K=2 alone gives 1.5×, N=4 alone gives 2.8×, K=2×N=4 combined gives **4.6×** vs single-vehicle baseline — close to the theoretical product. Subprocess per cell. The headline batching pattern for autonomous-driving ego + traffic in MPPI / multi-agent RL. | ✗ headless by design |
| 9 | [`city_traffic_ego.py`](city_traffic_ego.py) | **Autonomous-driving scenario — ego + traffic on a 4-lane highway.** 1 red AWD ego + 7 traffic agents (3 RWD blue coupes, 3 FWD small dark-red sedans, 1 yellow truck) under a simple P lane-keeper. Top-down camera frames the road. `--n_envs N` enables the L2 × L3 combined batching pattern (one MPPI candidate per env, all 8×N vehicles batched). The reference setup for "MPPI ego in surrounding traffic" workflows. | ✓ `--viewer` |
| 10 | [`l2l3_minimal.py`](l2l3_minimal.py) | **Shortest L2 × L3 program (~90 lines).** K interacting vehicles share one world (collide — L2) × N parallel scenarios (L3), advanced by one `MultiVehiclePhysics(scene, vehicles, n_envs=N)`. Demonstrates per-(scenario, vehicle) control: the lead car brakes in scenario 0 only and diverges from the rolling copies. The clean reference for "how do I use L2 × L3" before reading the full `city_traffic_ego` demo. `--k`, `--n_envs`, `--cpu`. | ✗ headless |
| 11 | [`dual_scene_terrain.py`](dual_scene_terrain.py) | **`VehicleScene` unified API + wheel-raycast dual/single scene modes.** Drives a car over a heightfield terrain with the high-level `VehicleScene` (no manual `gs.init`/`build`/`step`). `--compare` times `dual_scene` (default; terrain raycast in a separate static-BVH scene) vs `single_scene` (classic one scene); `--n-envs N` shows the dual_scene win growing with L3 batch size (the static BVH is shared across envs). `--horizontal-scale`, `--cpu`. | ✗ headless |
| 12 | [`obstacles_and_ramp.py`](obstacles_and_ramp.py) | **The encapsulated obstacle API + parameter-behavior matrix.** Builds a course entirely with `VehicleScene` — `add_ground_plane`, `add_static` (a wheel-raycast platform/block, with the `collision_morph`/`wheel_raycast_morph` split), and `add_dynamic` (a collide-only box the car pushes; a `wheel_raycast=True` ramp the wheels sense). Prints the body registry (each body's main/raycast entities — `docs/api-reference.md` §0.2 made concrete), then drives through and reports. `--mode single_scene`, `--cpu`. | ✗ headless |
| 13 | [`path_follow_demo.py`](path_follow_demo.py) | **Closed-loop path following (`genesis_vehicle.control`).** The bundled 10-wheel tank follows a waypoint path around a central wall: `PathFollower` inverts the bundled reference sweep table (`data/tank_sweep_signed.csv`) each step into (throttle, steer, brake). PASS = final position within 3 m of the goal. The end-to-end reference for [`docs/path-following.md`](../docs/path-following.md). `--mp4 [PATH]` records the run headless (bird's-eye camera + HUD; needs opencv-python). | ✓ `--viewer` / `--mp4` |
| 14 | [`path_follow_osc_demo.py`](path_follow_osc_demo.py) | **Trajectory following THROUGH the OSC server** — the physics runs in a separate `genesis_vehicle.server` process (as for UE/Unity) and this script plays the game-client role: TargetBulk state in (UE wire frame → Genesis, window-FD velocity), `PathFollower` on the client side, `(steer, throttle, brake)` back over `/Genesis/Vehicle/Control`. Server-side tuning is matched to the bundled sweep table via the `Vehicle/Init` json (`omegaMaxDrive`, `maxBrake`, `wheelOverrides`). PASS = final position within 3 m of the goal. `--viewer` opens the SERVER's Genesis viewer; closing that window ends the server and the client shuts down with it. | ✓ `--viewer` (server-side) |
| 15 | [`path_follow_reverse_demo.py`](path_follow_reverse_demo.py) | **Reverse maneuver with explicit waypoint yaw (v1.1.14 5-tuples).** "Back into a parking bay": drive past the bay, cusp (auto stop-and-reverse), then back along a bezier arc whose waypoints carry the PLANNED chassis heading (`(x, y, z, speed, yaw)`) so the tank arrives facing bay-north, front-out. PASS = within 2 m of the bay AND within 0.4 rad of the arrival heading. Waypoint markers + polyline always drawn (forward cyan / reverse orange); `--mp4 [PATH]` records the run headless. | ✓ `--viewer` / `--mp4` |

The three perf benchmarks (5, 7, 8) are intentionally headless — camera
rendering adds per-step overhead that distorts the throughput numbers
they're trying to measure. The visual equivalents are in their
respective docstring pointers (multi_env_render for L3, road_loop for
L2 with `--solver multi_batched`, city_traffic_ego for L2 × L3).

### Wheel animation is auto-managed by `VehicleScene`

When the main scene renders (native viewer or any camera), `VehicleScene`
draws the wheels via the solver-free **instanced renderer** (v1.1.17):
closed-form `wheel_visual_transforms` poses are streamed into instanced
render nodes each step. The wheels are **NOT updated through the Genesis
rigid solver** — the nodes live in Genesis's external render channel (the
same machinery as the debug-draw overlays) — so physics is identical to a
headless run; the only cost is the pose streaming itself (a slight
per-step overhead, ~2–3 ms at 30 vehicles on CPU). Headless runs draw
nothing and pay nothing. Whenever a viewer or camera is present, wheel
visuals are always active — there is no off switch.
It is not a user-facing knob (`VehicleScene(wheel_render_mode=...)`
exists only to force the legacy joint-sync fallback, which is also used
automatically for `n_envs > 1`).

For an **external** renderer (UE / Unity), keep the run headless and read
wheel poses from `wheel_visual_transforms()` /
`visual_parts_transforms()` (closed-form, ~µs) — the same data source
the instanced renderer uses.

## Bundled assets

- [`urdf/car_4w.urdf`](urdf/car_4w.urdf) — 1.2-ton 4-wheel RWD sedan,
  primitive geometry only (no mesh files), follows project URDF
  conventions (steer axis `(0, 0, -1)`, FL/FR/RL/RR wheel order,
  chassis collision box bottom above wheel ray origin).
- [`urdf/tank_ray.urdf`](urdf/tank_ray.urdf) — 10-wheel skid-steer tank
  (primitive geometry, turret + barrel joints), the reference vehicle of
  `path_follow_demo`.
- [`data/tank_sweep_signed.csv`](data/tank_sweep_signed.csv) — reference
  sweep table for `tank_ray.urdf` + `tank_10w_skid_belt` +
  [`tank_tuning.py`](tank_tuning.py) overrides at dt 0.025 × substeps 10
  (the SDK-recommended timing), measured with the body-frame-corrected CLI
  (see [`docs/path-following.md`](../docs/path-following.md); re-measure
  if any of the four changes — URDF / preset / overrides / dt).
- [`tank_tuning.py`](tank_tuning.py) — those overrides, importable
  (`TankTuning`) and directly usable as `sweep_measure --config`.

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
python -m genesis_vehicle.samples.dual_scene_terrain --compare
python -m genesis_vehicle.samples.dual_scene_terrain --compare --n-envs 64
python -m genesis_vehicle.samples.obstacles_and_ramp
python -m genesis_vehicle.samples.obstacles_and_ramp --mode single_scene --cpu
```

The perf benchmarks are headless (no viewer, no `cv2`, no `pynput`). The chase-cam
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
