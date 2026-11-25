# `genesis_vehicle/samples/`

A set of minimal, self-contained sample scripts that exercise the SDK's core
API surface. They depend only on the SDK itself and the bundled assets
(`urdf/car_4w.urdf`, `urdf/tank_ray.urdf`, `data/tank_sweep_signed.csv` —
all primitive-geometry / plain CSV) — no external mesh files, no
umbrella-level helpers.

| abbr | meaning |
|---|---|
| MPC / NMPC / LMPC | Model Predictive Control / its nonlinear (do_mpc + IPOPT) and linear (OSQP) implementations in `PathFollower` (v2.0.0) |
| PD | the 1.x `PathFollower` law (`mpc_model="off"`) |
| CTE | cross-track error: distance to the course polyline (m) |
| tot>25 | share of control steps whose control + simulation time exceeds 25 ms |

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
| 10 | [`l2l3_minimal.py`](l2l3_minimal.py) | **Shortest L2 × L3 program (~90 lines).** K interacting vehicles share one world (collide — L2) × N parallel scenarios (L3), advanced by one `MultiVehiclePhysics(scene, vehicles, n_envs=N)`. Demonstrates per-(scenario, vehicle) control: the lead car brakes in scenario 0 only and diverges from the rolling copies. The clean reference for "how do I use L2 × L3" before reading the full `city_traffic_ego` demo. `--k`, `--n_envs`, `--gpu`. | ✗ headless |
| 11 | [`dual_scene_terrain.py`](dual_scene_terrain.py) | **`VehicleScene` unified API + wheel-raycast dual/single scene modes.** Drives a car over a heightfield terrain with the high-level `VehicleScene` (no manual `gs.init`/`build`/`step`). `--compare` runs both modes and checks they agree; `--n-envs N` batches over L3 (the static BVH is shared across envs). The terrain morph is explicitly re-centred with `pos=(-20, -20, 0)` — `gs.morphs.Terrain`'s origin is a CORNER, and without that shift the car spawned at `(0,0)` half off the mesh and fell to z=-84 while the sample still exited 0 (fixed with a FINAL block: all wheel rays hit, ride height stable, drove forward, stayed in bounds; **exits non-zero on FAIL**). `--compare` also prints each mode's ms/step — the absolute per-step cost on your machine — but **no ratio and no speedup verdict**: both modes run back to back in one process in a fixed order (single_scene first), and the two slots are not equivalent (single_scene measures slower in the second slot than in the first, and swapping the order moves the comparison), so the two numbers must not be compared to each other. `--horizontal-scale`, `--drive-s`, `--gpu`. Since v1.6.1 `run(terrain_size=...)` builds a bigger plate (the (d) bound is derived from the half-extent the run actually used) and `run()` returns its own conditions block (dt, substeps, terrain_size/half, settle/drive seconds, throttle, steer, window speeds, engine and SDK versions) — that dict is what `bench_raycast_mode.py` (19) consumes, so treat its keys as a contract. | ✗ headless |
| 12 | [`obstacles_and_ramp.py`](obstacles_and_ramp.py) | **The encapsulated obstacle API + what a wheel contact model does at a vertical edge.** Builds a course entirely with `VehicleScene` — `add_ground_plane`, `add_static` (a wheel-raycast block, with the `collision_morph`/`wheel_raycast_morph` split), `add_dynamic(physics=True, wheel_raycast=False)` (a collide-only box the CHASSIS knocks away) and `add_dynamic(physics=False, wheel_raycast=True)` (a 0.12 m ramp the wheels sense). Prints the body registry (`docs/api-reference.md` §0.2 made concrete), drives through, and ends in a FINAL block that **exits non-zero on FAIL** (it used to print `RESULT: PASS`/`CHECK` and exit 0 either way). `--wheel-contact point|swept_envelope` (v1.6.0) shows the contact model AT the lip: how much of the 0.12 m step arrives in ONE `dt` — measured 0.120 m (`point`, i.e. the whole obstacle) vs 0.072 m (`swept_envelope`, M=9). The lift (~0.8 m for a 0.12 m obstacle) and peak-force criteria are **regression bands, not physical acceptance** — read the module docstring before quoting any number, and note no ratio between the two modes is printed (the magnitudes are phase-sensitive). The knock box had to be raised to 1.0 m in v1.6.0: at 0.5 m it passed clean under the chassis collider and was never touched. `--mode single_scene`, `--viewer`, `--gpu`, `--drive-s`, `--bench` (reporting only, no ratio). | ✓ `--viewer` |
| 13 | [`path_follow_demo.py`](path_follow_demo.py) | **Closed-loop path following (`genesis_vehicle.control`).** The bundled 10-wheel tank follows a waypoint path around a central wall: `PathFollower` inverts the bundled reference sweep table (`data/tank_sweep_signed.csv`) each step into (throttle, steer, brake). PASS = final position within 3 m of the goal and not `GOAL-MISSED` / `MPC-ABORT`. The end-to-end reference for [`docs/path-following.md`](../docs/path-following.md). Since v2.0.0 its follower is `mpc_model="off"` (the 1.x PD follower; `PathFollower(path, plant)` alone raises on 2.0.0). Since v2.0.1 the run ends on any terminal mode (`DONE`, `GOAL-MISSED`, `MPC-ABORT`; `PathFollower.TERMINAL_MODES`) and prints it with `stop_reason`; `GOAL-MISSED` / `MPC-ABORT` is a FAIL (through 2.0.0 the run ended early only on `DONE`). Since v2.0.2 it applies the terminal `(0, 0, 1)` command and steps until the tank is still (\|v\| < 0.01 m/s) or 2 s have passed before leaving; the follower stays `command_mode="signed"` (tanks refuse gear mode). `--mp4 [PATH]` records the run headless (bird's-eye camera + HUD; needs opencv-python). | ✓ `--viewer` / `--mp4` |
| 14 | [`path_follow_osc_demo.py`](path_follow_osc_demo.py) | **Trajectory following THROUGH the OSC server** — the physics runs in a separate `genesis_vehicle.server` process (as for UE/Unity) and this script plays the game-client role: TargetBulk state in (UE wire frame → Genesis, window-FD velocity), `PathFollower` on the client side, `(steer, throttle, brake)` back over `/Genesis/Vehicle/Control`. Server-side tuning is matched to the bundled sweep table via the `Vehicle/Init` json (`omegaMaxDrive`, `maxBrake`, `wheelOverrides`). PASS = final position within 3 m of the goal and not `GOAL-MISSED` / `MPC-ABORT`. `--viewer` opens the SERVER's Genesis viewer; closing that window ends the server and the client shuts down with it. Since v2.0.0 its follower is `mpc_model="off"` (the 1.x PD follower; `PathFollower(path, plant)` alone raises on 2.0.0). Since v2.0.1 the run ends on any terminal mode (`DONE`, `GOAL-MISSED`, `MPC-ABORT`; `PathFollower.TERMINAL_MODES`) and prints it with `stop_reason`; `GOAL-MISSED` / `MPC-ABORT` is a FAIL (through 2.0.0 the run ended early only on `DONE`). Since v2.0.2 it keeps SENDING the terminal command, one per received state, until the windowed speed is below 0.01 m/s or 2 s of server time have passed. The OSC wire has no gear field, so the follower stays signed. | ✓ `--viewer` (server-side) |
| 15 | [`path_follow_reverse_demo.py`](path_follow_reverse_demo.py) | **Reverse maneuver with explicit waypoint yaw (v1.1.14 5-tuples).** "Back into a parking bay": drive past the bay, cusp (auto stop-and-reverse), then back along a bezier arc whose waypoints carry the PLANNED chassis heading (`(x, y, z, speed, yaw)`) so the tank arrives facing bay-north, front-out. PASS = within 2 m of the bay AND within 0.4 rad of the arrival heading AND not `GOAL-MISSED` / `MPC-ABORT`. Waypoint markers + polyline always drawn (forward cyan / reverse orange); `--mp4 [PATH]` records the run headless. Since v2.0.0 its follower is `mpc_model="off"` (the 1.x PD follower; `PathFollower(path, plant)` alone raises on 2.0.0). Since v2.0.1 the run ends on any terminal mode (`DONE`, `GOAL-MISSED`, `MPC-ABORT`; `PathFollower.TERMINAL_MODES`) and prints it with `stop_reason`; `GOAL-MISSED` / `MPC-ABORT` is a FAIL (through 2.0.0 the run ended early only on `DONE`). Since v2.0.2 it applies the terminal command until the tank is still or 2 s have passed before leaving. | ✓ `--viewer` / `--mp4` |
| 16 | [`sweep_and_drive.py`](sweep_and_drive.py) | **Measure a sweep table for a URDF, then drive it — the full offline→online loop in one script (v1.2.2+).** `--vehicle car\|tank` picks the bundled URDF + preset; phase 1 measures a sweep table (cached to `data/<vehicle>_sweep_generated.csv`, one-time; `--remeasure` refreshes), phase 2 drives a dogleg path with `PathFollower` inverting it. Measure + drive live together so the table and the plant it was measured against stay in sync. PASS = final position within 3 m of the goal and not `GOAL-MISSED` / `MPC-ABORT`. `--gpu`, `--quick` (coarse smoke table). Since v2.0.0 its follower is `mpc_model="off"` (the 1.x PD follower; `PathFollower(path, plant)` alone raises on 2.0.0). Since v2.0.1 the drive ends on any terminal mode (`PathFollower.TERMINAL_MODES`) and prints the end mode with `stop_reason`; through 2.0.0 only `DONE` ended it early. | ✓ `--viewer` |
| 17 | [`aero_drag_playground.py`](aero_drag_playground.py) | **Tune aerodynamic drag LIVE while a car drives (v1.2.4).** Shows `ChassisConfig.drag_area` / `air_density` are runtime parameters — the pipeline reads `resolved.chassis` every step, so mutating it mid-drive changes physics next step. A Tk slider panel gives live Cd·A / air-density controls while the car holds full throttle; you watch top speed rise/fall as drag changes (top speed = drag + omega cap together). `--sweep` runs a scripted Cd·A sequence headless when Tk / a display is unavailable. | ✓ `--viewer` + Tk GUI |
| 18 | [`fleet_follow_demo.py`](fleet_follow_demo.py) | **A whole fleet path-following at once — autodiff plant vs measured table (v1.4.0).** K tanks on K lanes, each with its own phase-shifted S-curve so no two ask the inverse plant for the same thing. `--plant jacobian` drives every vehicle from ONE batched `DifferentiablePlant` through `FleetFollower` (plan per follower → one batched solve → finish per follower); `--plant sweep` shares ONE stateless `SweepTable` across K independent `PathFollower`s. `--plant both` (default) runs each on the same fleet and prints arrivals, cross-track (point-to-polyline, not to the nearest waypoint) and control cost per vehicle. The reference for how the two plants' costs SCALE differently, and for the course-dependence of their tracking — read the module docstring before concluding either is better. `--k N`, `--horizon`, `--newton-iters`, `--k-w`, `--mp4 [PATH]`. Since v2.0.0 every follower is `mpc_model="off"` — `FleetFollower` rejects MPC followers. Since v2.0.1 a member ends on any terminal mode (`DONE` or `GOAL-MISSED`), only `DONE` counts as arrived, and members that stopped without arriving are printed with their `stop_reason`; through 2.0.0 the run waited for every member's `DONE`. | ✓ `--viewer` / `--mp4` |
| 19 | [`bench_raycast_mode.py`](bench_raycast_mode.py) | **Order-independent, paired, fail-closed benchmark of the two `raycast_mode` values (v1.6.1).** The harness the v1.5.1 retraction asked for: ONE fresh process per measurement (it drives `dual_scene_terrain.run(..., measure=True)` in a child), an alternating slot order, and a paired ratio `r_k = ms(single_scene) / ms(dual_scene)` inside each repeat. It ALWAYS prints every `r_k`, the pooled median with `[min .. max]` and n, and both slot-order groups; it prints a RATIO line only if (i) each group's median lies inside the other's `[min, max]` and (ii) 1.0 lies outside the pooled range — a heuristic disclosure rule, not a statistical test, and it is allowed to answer "not measurable here" (it did, in 2 of its first 4 runs on the dev machine). Every worker, including the discarded warm-up, must pass `dual_scene_terrain.verdicts()` plus an x-margin gate; any failure ABORTS with no aggregate, because a run that fell off the plate is a run that is faster. Defaults: 640x640 m plate, `hs 4.0` (51,212 faces), 3 s settle, 20 s timed window, 6 repeats + 1 warm-up = 13 processes. `--repeats`, `--drive-s`, `--terrain-size`, `--horizontal-scale`, `--n-envs`, `--json`, `--gpu` (UNVERIFIED — every number in it was measured on CPU). | ✗ headless by design |
| 20 | [`terrain_drive.py`](terrain_drive.py) | **Driving an UNDULATING mesh terrain through `VehicleScene`.** Generates a wavy ground mesh and registers it with `add_static(collision=False)`, so in the default `dual_scene` mode it lives ONLY in the raycast scene as a Kinematic target (the rays follow every bump; that scene's BVH is static). `--rigid-terrain` runs the other path (`single_scene` with a rigid mesh) for an A/B look. The terrain is x-periodic and the car is wrapped back one period per crossing, so with a viewer it drives forever until ESC. `--amp`, `--duration`, `--native` (Genesis OpenGL viewer; may not surface a window under WSLg), `--gpu`. | ✓ `--viewer` / `--native` |
| 21 | [`mpc_follow_demo.py`](mpc_follow_demo.py) | **The 2.0.0 default path follower: sweep table + MPC layer (v2.0.0).** The reference car follows the field demo course (straight 20 m, a 90° left turn of 30 m radius, straight 15 m; from rest, ramped at 1 m/s² to 8 m/s) with `PathFollower` running the MPC layer on the forward block. On its first run it measures and caches, in `data/mpc_cache/` (git-ignored; `--remeasure` refreshes), a sweep table with the grid flags (the default grid stops at 4 m/s) and the identify params JSON — a cached params file keeps the `sdk_version` it was measured with. Needs `requirements-mpc.txt`. `--mpc-impl linear` (LMPC), `--mpc-model off` (the PD follower, same course), `--plant diff` (`DifferentiablePlant` in stage 2), `--course cusp` (forward MPC, cusp, reverse PD), `--course misaligned` (v2.0.1: the documented 2.0.0 stall start, spawned on the first waypoint; `--start-yaw` rad; needs `--t-max 60` — measured `DONE` at 41.775 s with the table and params the stall was documented with, so the default `--t-max` 40.0 prints FAIL), `--course vertex --vertex-deg D` (v2.0.1: straight 20 m, a vertex with heading change D, straight 20 m, 4 m/s), `--radius` (v2.0.1: demo turn radius, default 30 m), `--align-gate off` / `--stall-timeout off` (v2.0.1: `mpc_align_gate=None` / `mpc_stall_timeout=None`; both off gives the 2.0.0 commands only on runs with no `GOAL-MISSED`, no block-end sign step and no `fail_budget` abort — not an exact 2.0.0 restore, `docs/path-following.md` §5.8), `--log PATH` (v2.0.1: one JSON line per control step), `--on-solve-failure hold`, `--table` / `--params` (existing files), `--command-mode gear\|signed` (v2.0.2, default `gear`, also with `--mpc-model off`, where the brake gain is derived from the params file; `signed` is the 2.0.1 loop), `--gear-feedback on\|off` (v2.0.2, default `on`: the engaged gear is passed to `PathFollower.step(gear=)`), `--kturn off\|on` (v2.0.3, default `off`: `on` passes `mpc_kturn=KTurnParams()`, the opt-in K-turn of `docs/path-following.md` §5.10; needs the MPC layer in gear mode with the heading gate — not `--mpc-model off`, `--command-mode signed` or `--align-gate off` — otherwise exit 2), `--lateral-limit L` / `--max-reverse M` (v2.0.3, the `KTurnParams` fields; imply `--kturn on`), `--course behind --behind-len N` (v2.0.3: a straight path N m behind the spawned car, default 6 — the reference K-turn case; `reverse_entry_step` counts only `DRV-1`, not K-turn reverse legs). Since v2.0.2 the car is driven in gear mode by default (decelerations by brake, reverse only on the cusp course's reverse block — and, with `--kturn on`, on K-turn reverse legs), the log carries the commanded and the engaged gear, `MPC_RESULT` gains `command_mode`, `gear_feedback`, `post_stop_v`, `post_stop_s`, and on a terminal mode the terminal command is applied until the car is still or 2 s have passed (the metrics stop at the terminal step). Prints CTE mean / max and the share of control steps over 25 ms; last line `MPC_RESULT {json}`. PASS = DONE within `--t-max` and max CTE < 1.0 m (other courses: DONE). Since v2.0.1 the run ends on any terminal mode and the `MPC_RESULT` record carries `end_mode` and `stop_reason`; `GOAL-MISSED` / `MPC-ABORT` is never a PASS. Measured (CPU/WSL2, genesis-world 1.4.5, signed, 2.0.0): NMPC CTE 0.0670 / 0.682 m, PD 0.1068 / 0.297 m, LMPC 0.112 / 0.242 m (`docs/path-following.md` §5.7); 2.0.2 gear mode with feedback: NMPC 0.0678 / 0.6835 m, `DONE` 13.45 s, \|v\| 0.0074 m/s after 0.93 s of the terminal command (CHANGELOG 2.0.2 §8.1, §9). 2.0.3: the default runs are unchanged (the `--course vertex --vertex-deg 150` log is byte-identical to 2.0.2's); `--course behind --behind-len 6 --kturn on --t-max 60` exit 0, PASS, `DONE` 29.30 s, CTE max 10.536 m (CHANGELOG 2.0.3 §5.7). | ✓ `--viewer` |
| 22 | [`bench_path_follower.py`](bench_path_follower.py) | **How often does a `PathFollower` control step overrun 25 ms on THIS machine? (v2.0.0)** Runs `mpc_follow_demo` in one fresh process per measurement, one at a time, alternating the arm order between rounds; arms `nonlinear` / `linear` / `pd`. Always prints every run's tot>25 (control + simulation over 25 ms), control-only share, control mean and solve percentiles; prints a per-arm median only when the two order groups agree — a fail-closed disclosure heuristic, not a statistical test. On the dev machine it declined to publish the NMPC median in both sessions it ran (5.2-6.4 % and 7.5-9.5 % per run; `docs/path-following.md` §5.6). `--arms`, `--repeats`, `--table`, `--params`, `--t-max`, `--json`, `--gpu` (unverified). Needs `requirements-mpc.txt` for the MPC arms. Since v2.0.2 every arm is pinned to `command_mode="signed"`, so its figures stay comparable with the published (signed) ones; it does not measure gear mode. | ✗ headless by design |

**Path-following loops and the terminal command (v2.0.2).** Through v2.0.1
`path_follow_demo`, `path_follow_reverse_demo`, `path_follow_osc_demo` and
`mpc_follow_demo` left their loop BEFORE applying the returned `(0, 0, 1)`
command; since v2.0.2 they apply it and keep stepping (the OSC demo keeps
sending it) until the vehicle is still (|v| < 0.01 m/s) or 2 s have passed —
measured by the implementation (the samples were not edited after): |v|
0.0031-0.0074 m/s at exit (CHANGELOG 2.0.2 §9). `sweep_and_drive` applies it first; `fleet_follow_demo` keeps applying it
to members that have ended while the others drive, and still breaks before
applying it on the step the last member ends — a loop copied from it into a
longer-running program must apply the command before leaving, as the
[`docs/path-following.md`](../docs/path-following.md) quick start does.

The five benchmarks (5, 7, 8, 19, 22) are intentionally headless — camera
rendering adds per-step overhead that distorts the numbers they're trying
to measure. The visual equivalents are in their
respective docstring pointers (multi_env_render for L3, road_loop for
L2 with `--solver multi_batched`, city_traffic_ego for L2 × L3).

### Wheel animation is auto-managed by `VehicleScene`

When the main scene renders (native viewer or any visualizer camera),
`VehicleScene` draws the wheels by overriding the poses of the ENGINE's own
wheel render nodes with the closed-form `wheel_visual_transforms` pose on every
sync of the main scene's rasterizer context, at every `n_envs` (v2.0.6; a wheel
the engine draws no node for, e.g. under `vis_mode="collision"`, gets an SDK
fallback node). The wheels are **NOT updated through the Genesis rigid
solver**: no SDK code writes solver state for rendering, and rendered runs were
measured bit-identical to headless on the rows in CHANGELOG 2.0.6 §6. Wheel
pixels carry their wheel link's segmentation id. Headless runs draw nothing and
pay nothing. Whenever a viewer or camera is present, wheel visuals are always
active — there is no off switch, and `VehicleScene(wheel_render_mode=...)` has
no other mode (`"internal_sync"` raises since v2.0.6). Camera SENSORS
(`scene.add_sensor(...)`) are not hooked and show the URDF wheel at the solver
pose. (Through v2.0.5 this paragraph described SDK-instanced nodes at `n_envs
== 1` "in Genesis's external render channel" — a channel genesis-world 1.5.0
removed — and a legacy joint-sync fallback "used automatically for `n_envs >
1`"; that fallback moved the chassis trajectory and was removed in v2.0.6.)

**Native-viewer camera follow:** use the viewer's own
`vs.viewer.follow_entity(veh.entity_main)` (see
[`terrain_drive.py`](terrain_drive.py) for the pattern incl. a fallback for
older Genesis), NOT a per-step `set_camera_pose` from the drive loop — the
viewer draws on its own thread, and a camera set after `vs.step()` returns
lags the vehicle within each drawn frame, which reads as the vehicle
"trembling". (Through v2.0.5 this paragraph said that since v1.1.25
`VehicleScene.build()` makes the follow camera, wheel buffers and node poses
commit atomically per frame — true on genesis-world <= 1.3.3 only; the patch
was removed in v2.0.5, physics-contracts.md §7.8. A followed vehicle can
tremble in the native viewer at speed; a draw artefact.)

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
  `path_follow_demo` and `fleet_follow_demo`.
- [`data/tank_sweep_signed.csv`](data/tank_sweep_signed.csv) — reference
  sweep table for `tank_ray.urdf` + `tank_skid_belt` +
  [`tank_tuning.py`](tank_tuning.py) overrides at **dt 0.025 × substeps 4**
  (the SDK default, shared by `VehicleScene`, `sweep_measure` and the OSC
  server since v1.2.1 — before that the tool measured at substeps 10 while
  `path_follow_osc_demo` drove the table through a substeps-4 server),
  measured with the body-frame-corrected CLI
  (see [`docs/path-following.md`](../docs/path-following.md); re-measure
  if any of the four changes — URDF / preset / overrides / dt).
  Re-measured at v1.2.1 after `tank_skid_belt` switched to mass-derived
  suspension; at matched timing the plant barely moved (mean |Δa| 0.0013 m/s²,
  mean |Δω_z| 0.0002 rad/s), which is the check that the derivation reproduces
  the literals it replaced.
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
python -m genesis_vehicle.samples.dual_scene_terrain
python -m genesis_vehicle.samples.dual_scene_terrain --mode single_scene
python -m genesis_vehicle.samples.dual_scene_terrain --compare
python -m genesis_vehicle.samples.dual_scene_terrain --compare --n-envs 64
python -m genesis_vehicle.samples.obstacles_and_ramp
python -m genesis_vehicle.samples.obstacles_and_ramp --mode single_scene
python -m genesis_vehicle.samples.obstacles_and_ramp --wheel-contact swept_envelope
python -m genesis_vehicle.samples.bench_raycast_mode
python -m genesis_vehicle.samples.bench_raycast_mode --json bench.json
python -m genesis_vehicle.samples.bench_raycast_mode --repeats 10
python -m genesis_vehicle.samples.mpc_follow_demo                      # needs requirements-mpc.txt
python -m genesis_vehicle.samples.mpc_follow_demo --mpc-impl linear
python -m genesis_vehicle.samples.mpc_follow_demo --mpc-model off
python -m genesis_vehicle.samples.bench_path_follower --arms nonlinear,linear,pd --repeats 5
```

The perf benchmarks are headless (no viewer, no `cv2`, no `pynput`). The chase-cam
HUD / keyboard-driven demos live one level up at the umbrella project
(the per-vehicle `demo_drive.py` scripts) and are not bundled with the SDK on
purpose — they have opencv-python + pynput dependencies and are more
demo than reference. Use these `samples/` scripts as the reference for
what the SDK looks like from a caller's perspective.

## Cross-references in docs

| Doc | Sample it references |
|---|---|
| [`../docs/quickstart.md`](../docs/quickstart.md) | `quickstart.py` |
| [`../docs/stability-profiles.md`](../docs/stability-profiles.md) | `slope_hold.py` |
| [`../docs/api-reference.md`](../docs/api-reference.md) | `quickstart.py`, `slope_hold.py`, `batched_rollout.py`, `road_loop.py`, `perf_vectorization.py` |
| [`../docs/dual-scene-raycast.md`](../docs/dual-scene-raycast.md) | `dual_scene_terrain.py` (API + pose equivalence, `--compare`), `bench_raycast_mode.py` (the order-free, fail-closed benchmark of the two modes) |
