# genesis_vehicle

A ray-cast wheel + Pacejka-tire vehicle physics SDK on top of the
[Genesis](https://genesis-embodied-ai.github.io/) physics engine.

genesis_vehicle is a Python package that wraps Genesis with a batched
(`n_envs ≥ 1`), composable vehicle physics layer. The wheel-count of your
URDF and the choice of steering / drivetrain / coupling / tire model are
independent — wire any combination through `VehicleConfig` and let the SDK
run the same 5-step pipeline (raycast → suspension → slip → tire → ω) for
every wheel, every step, every env. Designed for RL / MPPI control loops
and Real2Sim parameter fitting.

## More About genesis_vehicle

At a glance, the SDK provides:

- **A single `VehiclePhysics` driver** that orchestrates the 5-step
  ray-wheel + Pacejka pipeline batched over `n_envs`. One `step()` call
  advances every env, every wheel, in lock-step.
- **Composable building blocks** for steering, drivetrain, wheel coupling,
  tire model, and stability hooks (see the table below). Each block is a
  small class you can subclass.
- **URDF-as-default, API-as-truth merge.** Wheel positions, radii, masses,
  inertia, and joint names come from your URDF; anything you set explicitly
  on `WheelConfig` wins. After `resolve()`, the resulting `ResolvedConfig`
  is the only source of truth at runtime.
- **ISO 8855 sign convention** end-to-end: `+X` forward, `+Y` left, `+Z`
  up, `+steer` = right turn, `+throttle` = forward. Internal sign flips
  for Genesis RHS / URDF axis quirks are absorbed inside the strategies.
- **Stability profiles** (`"control"` / `"raw"` / `"research"`) that bundle
  the right numerical stabilizers for your use case so MPPI / RL / Real2Sim
  users don't have to assemble hook lists by hand.
- **Two wheel-contact models.** `wheel_contact="point"` (the default) is one
  downward ray per wheel — light and stable, but a zero-radius probe that reads
  ground height as a step function. `wheel_contact="swept_envelope"` (v1.6.0,
  opt-in) casts M rays per wheel and reduces them with the swept-circle lower
  envelope, so a wheel starts climbing a kerb about `sqrt(2*r*h)` before it
  instead of taking the whole obstacle in one `dt`. See
  [`docs/tire-and-contact.md`](docs/tire-and-contact.md).
- **Pure-Python tests** (no Genesis runtime needed) that you can run in
  any CI.

The full mental model is laid out in
[`docs/concepts.md`](docs/concepts.md); the API surface lives in
[`docs/api-reference.md`](docs/api-reference.md).

## Three-axis vectorization (L1 / L2 / L3)

Vehicle simulation has three orthogonal axes you can batch over. The SDK
hits all three with the same lightweight Python surface:

| Axis | Meaning | API | Batched by | Status |
|---|---|---|---|---|
| **L1** | Wheels of ONE vehicle (4-10) | `VehiclePhysics` (built-in) | Compute pipeline runs on `(*, n_wheels)` tensors — no per-wheel Python loop | ✓ since v0.5.0 (always on) |
| **L2** | K vehicles of the same kind sharing ONE Genesis env | `MultiVehiclePhysics(scene, vehicles)` | One batched compute per kind; per-vehicle I/O via batched solver calls | ✓ since v0.5.11 |
| **L3** | N parallel Genesis envs (parallel-universe rollouts) | `scene.build(n_envs=N)` + `VehiclePhysics(n_envs=N)` | All env work collapses into one CUDA dispatch chain | ✓ since v0.1 |
| **L2 × L3** | K vehicles × N parallel envs (`N·K` total) | `MultiVehiclePhysics(scene, vehicles, n_envs=N)` | Combined: compute on `(N·K, n_wheels)` tensors, one batched call per kind | ✓ since v0.5.14 |

**Rule of thumb**:
- 1 vehicle, many parallel rollouts → **L3** alone (`n_envs > 1`). 44× at N=64.
- K vehicles visible in one scene (traffic, multi-agent demo) → **L2** alone.
- K vehicles per scenario × N parallel scenarios (ego + traffic for MPPI) → **L2 + L3 combined**, ~multiplicative speedup.

Measured (RTX 5070 Laptop, see [`docs/batching.md`](docs/batching.md)):

| K | N | total | gain vs (K=1, N=1) |
|--:|--:|------:|-------------------:|
| 1 | 4 |     4 | **2.8×** (L3 only) |
| 2 | 1 |     2 | **1.5×** (L2 only) |
| 2 | 4 |     8 | **4.6×** (combined ≈ 1.5 × 2.8) |
| 1 | 64 |  64 | **44×** (L3 push) |

Full design + decision matrix in [`docs/batching.md`](docs/batching.md).
Runnable benchmarks (headless):
[`samples/perf_vectorization.py`](samples/perf_vectorization.py) (L3),
[`samples/perf_multi_vehicle.py`](samples/perf_multi_vehicle.py) (L2),
[`samples/perf_l2_l3_combined.py`](samples/perf_l2_l3_combined.py) (combined).
Visual demos (with `--viewer`):
[`samples/multi_env_render.py`](samples/multi_env_render.py) (L3 grid),
[`samples/road_loop.py`](samples/road_loop.py) (L2 multi-kind track),
[`samples/city_traffic_ego.py`](samples/city_traffic_ego.py) (L2 × L3 ego + traffic highway).

## Installation

Requires Python 3.12+ and
[Genesis](https://genesis-embodied-ai.github.io/) — **`genesis-world` 1.5.0 is
REQUIRED since v2.0.5** ([`requirements.txt`](requirements.txt): `genesis-world>=1.5.0,<1.6`),
together with the dependency set the 1.5.0 wheel pulls: **`quadrants` 1.4.0,
`coacd` >= 1.0.14, `PyGEL3D` >= 0.8.0** (the measured install: coacd 1.0.14,
PyGEL3D 0.8.2). A clone that updates by `git pull` (the UE deployment) or a
frozen field build must upgrade those packages itself, BEFORE pulling 2.0.5 —
see the field note in CHANGELOG 2.0.5 §12. For genesis-world 1.4.5 stay on SDK
v2.0.4 (commit `9e5c4d5`; there is no tag). 1.5.0 no longer pulls `moviepy`,
`vtk` or `z3-solver` on a fresh install; the SDK and its demos do not use them.

```bash
pip install -r requirements.txt     # from the repo root: genesis-world>=1.5.0,<1.6
```

**What enforces it — a refusal at import.** Through v2.0.4 this section
described a warn-only floor (`_gs_compat.check_genesis_floor`, 1.4.5, never
raised); it was removed in v2.0.5. Now `_gs_compat.require_genesis` runs right
after `import genesis` in `core`, `vehicle_scene`, `multi_vehicle`, `raycast`,
`scene_helpers`, `visual` and `server`, and raises `GenesisVersionError` (an
`ImportError`, exported from the package) below 1.5.0 or on a 1.5.0
pre-release — so `VehicleScene`, `VehiclePhysics`, `MultiVehiclePhysics` and
the server fail at import, before any scene exists, `VehicleScene(init_genesis=False)`
included. The message is also logged once at ERROR on the `genesis_vehicle`
logger, because an `except ImportError:` in caller code would otherwise hide
it. It does NOT cover every module that touches genesis: `kinematics`, `urdf`
and `control.sweep_measure` import without it, and `import genesis_vehicle`,
`PathFollower`, `SweepTable`, `DifferentiablePlant` and `control.identify`
import on an older engine (genesis itself must still be importable). A
genesis-world >= 1.6 logs one "not validated" WARNING and continues; **a 1.5.x
patch release (1.5.1, ...) is admitted silently and is NOT validated** — the 20
engine-keyed identity locks skip on it, and it needs its own capture / parity
round. The 2.0.5, 2.0.6 and 2.1.0 suites run on 1.5.0; **measurements published before v2.0.5
were taken on 1.4.5 (v1.6.9-v2.0.4) or 1.4.0 (before) and keep their labels**;
a figure in these docs without an engine label is not a 1.5.0 figure. Every
`<= 1.3.3` / 1.4.x compatibility branch was removed in v2.0.5 (through v2.0.4
this section called them "dead weight to be removed in a later release"); the
legacy `RigidOptions.dt` normalisation (`vehicle_scene._normalize_rigid_dt`)
remains. See
the SDK v2.0.5 CHANGELOG entry for what 1.5.0 changed and the parity evidence;
SDK v1.6.9 for the engine 1.4.0 → 1.4.5 bump; SDK v1.5.0 for the engine 1.3.3 →
1.4.0 bump. Each cross-version parity
table is a record of a measurement taken once at the bump, not an ongoing
guarantee.

Ask for the solver rate with `VehicleScene(substeps=...)` and do **not** pass
`dt=` to `RigidOptions` — since genesis-world 1.4.0 that field means the
substep interval. See [`CHANGELOG.md`](CHANGELOG.md) for the per-release
backend-compatibility notes.

**Default materials and memory.** Since 1.4.5 genesis-world computes every
collision geom's SDF (signed distance field) at `build()`. On 1.4.5 the default
`gs.materials.Rigid()` cost a car +0.8 .. +1.2 s of build and +490 .. +508 MB of
peak resident memory versus 1.4.0 (two default-material cars, CHANGELOG 1.6.9
§3) — a 1.4.5 fact. On 1.5.0 that penalty was not seen: reference car, default
material 4.43 s / 1132 MB vs the SDK's coarse kwargs 4.34 s / 1133 MB
(implementation, CPU/WSL2, cold asset cache, n=3 medians, CHANGELOG 2.0.5 §9).
The SDK does NOT change a material you pass or leave as `None`. For a body whose
geoms are all convex, in a scene with no non-convex, non-terrain collision geom
(so no contact reads any SDF), `gs.materials.Rigid(**_gs_compat.COARSE_SDF_MATERIAL_KW)`
(`sdf_cell_size=1e4, sdf_min_res=16, sdf_max_res=16` since v2.0.5) gives every
geom the 16-per-axis minimum; a large `sdf_cell_size` alone does not switch the
SDF off.

The SDK is a single Python package directory; there is no `pip install`
yet. Clone and add the parent directory to `PYTHONPATH`:

```bash
git clone https://github.com/korfriend/GenesisVehicle.git
# either:
export PYTHONPATH="$(pwd):$PYTHONPATH"
# or inside your entry script:
#   import sys; sys.path.insert(0, "<parent of cloned dir>")
```

Then `import genesis_vehicle` from anywhere.

Dev dependency for running the test suite: `pip install pytest`.

### Optional: the path follower's MPC layer (v2.0.0)

`PathFollower`'s MPC layer — the 2.0.0 default stage 1, see
[`docs/path-following.md`](docs/path-following.md) §5 — needs packages the
rest of the SDK does not use, pinned in
[`requirements-mpc.txt`](requirements-mpc.txt): `casadi==3.8.2` + `do-mpc==5.1.2`
for the nonlinear MPC (the default), `osqp==1.1.3` + `threadpoolctl>=3` (+ scipy,
already present) for the linear one. They pull in more (matplotlib, pandas,
jinja2, joblib, setuptools) and pip may change packages you already have, so
look first and fall back to `--no-deps` if anything would move:

```bash
pip install --dry-run --report - -r requirements-mpc.txt
pip install --no-deps -r requirements-mpc.txt && pip check   # if an installed package would change
pip install -r requirements-mpc.txt                          # otherwise
```

`import genesis_vehicle` does not import any of them, and
`PathFollower(..., mpc_model="off")` (the 1.x follower) never needs them.
Licences, as facts: casadi LGPL-3.0-or-later (bundles Ipopt EPL-2.0, MUMPS
CeCILL-C, METIS glue EPL-1.0, OpenBLAS BSD-style), do-mpc LGPL-3.0, osqp
Apache-2.0, threadpoolctl BSD-3-Clause; they are installed by you, not
redistributed. The MPC code derives from NVlabs AlpaSim (Apache-2.0) — see
[`NOTICE`](NOTICE).

## Getting Started

```python
from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann

URDF = "<path to your 4-wheel-car URDF>"

# Physics backend — process-global, set ONCE, before any scene. Default is
# "cpu" (faster below ~100 envs; GPU is kernel-launch bound at small batch).
# The renderer is separate (always GPU). Use "gpu" only for large-n_envs L3.
VehicleScene.init_backend("cpu")

# VehicleScene is the single entry point: it owns gs.init / the scene(s) / build / step.
# raycast_mode defaults to "dual_scene" — keep it. "single_scene" works on a
# plane but cannot host a detailed road mesh at all (docs/dual-scene-raycast.md).
# substeps refines the CHASSIS contact integration, NOT the ray-wheel
# suspension, and changing it changes the trajectory (docs/server.md §2.3).
vs = VehicleScene(dt=0.025, substeps=4)      # dt: the preset's recommended_dt
                                             # (VehicleScene's own default is
                                             # 1/200 s); substeps=4 IS the default
vs.add_ground_plane(friction=1.0)
veh = vs.add_vehicle(URDF, preset=car_4w_rwd_ackermann, pos=(0, 0, 1.0))
vs.build()

for step in range(480):                       # 12 s @ 40 Hz
    veh.set_inputs(throttle=0.5, brake=0.0, steer=0.0)   # scalars or (n_envs,) tensors
    vs.step()

print(veh.get_pos()[0].cpu().numpy())
print(veh.all_wheels_grounded)   # (n_envs,) bool — is it actually on the terrain?
```

`veh.wheels_grounded` / `veh.all_wheels_grounded` (v1.5.2) answer "did each
wheel's ray find ground on the last step?" against the raycaster's own
miss sentinel. Use them rather than thresholding `veh.distances` yourself — a
miss reads back as the sensor's `max_range`, which is whatever you configured.

`VehicleScene` is the recommended entry point — it registers vehicles / static /
dynamic bodies (`add_vehicle` / `add_static` / `add_dynamic`), cameras
(`add_camera`), L2/L3 batching (`solver="batched"` default, `n_envs=`), and the
per-step loop, and owns the physics backend (`VehicleScene.init_backend`). To drop
*below* it to the raw `gs.Scene` + `VehiclePhysics` / `MultiVehiclePhysics` layer
(for control it doesn't expose), see [`docs/api-reference.md`](docs/api-reference.md)
§1 and the two-API-layers note in [`docs/concepts.md`](docs/concepts.md).

On first `VehiclePhysics` construction, the SDK prints a one-line banner:

```
[genesis_vehicle v0.6.0] Initialized: 4 wheels, Ackermann, RWD, Independent, n_envs=1, hooks=[RollingResistance, LowSpeedRegularizer]
```

Telling you, at a glance: version, topology, which strategies are wired
up, batch size, and which stability hooks are active.

More examples and the worked-out 10 s SETTLE / ACCEL / TURN / BRAKE scenario
walkthrough are in [`docs/quickstart.md`](docs/quickstart.md).

## Building Blocks

The five composition slots a `VehicleConfig` exposes. All are independent
of wheel count — pick any combination as long as the soft per-strategy
constraints hold (Ackermann needs an L + R wheel pair on each steered
axle; SkidSteer and SameSideBelt need every wheel to declare `side='L'`
or `side='R'`).

| Slot | Concrete options shipped |
|---|---|
| **Steering** | `Ackermann`, `PartialAckermann`, `SkidSteer`, `NoSteer` |
| **Drivetrain** | `FWD`, `RWD`, `AWD`, `PerSide` |
| **Coupling** | `Independent`, `SameSideBelt` |
| **Tire model** | `PacejkaAnisotropic`, `CoulombIsotropic` |
| **Wheel contact** (via `add_vehicle(wheel_contact=)`, v1.6.0) | `"point"` (default, one ray per wheel), `"swept_envelope"` (M rays reduced by the swept-circle lower envelope) |
| **Stability hooks** (via `stability=` profile) | `RollingResistance`, `LowSpeedRegularizer`, `StaticFrictionLock` |

Subclass any of the ABCs (`SteeringStrategy`, `DrivetrainStrategy`,
`CouplingStrategy`, `TireModel`, `StabilityHook`) to add new behaviors.

## Reading per-link transforms (telemetry / animation / attach)

Genesis only gives each link's **world** pose. For telemetry → animation
retargeting, attaching external sensors/effects, or placing ghost copies you
usually want each component relative to its **URDF parent**. `get_link_transforms`
(or `physics.link_transforms(...)`) composes that for you:

```python
from genesis_vehicle import get_link_transforms

lt = get_link_transforms(car, frame="parent")   # "world" | "base" | "parent"
lt.names          # ['base_link', 'front_left_axle_carrier', ...]
lt.parent_local   # parent link's local index per link, -1 for the root
lt.pos            # (n_envs, n_links, 3)
lt.quat           # (n_envs, n_links, 4)  wxyz
lt.matrices()     # (n_envs, n_links, 4, 4)
```

| `frame` | each link relative to |
|---|---|
| `"world"` | world (raw Genesis output) |
| `"base"` | the entity base/root link |
| `"parent"` (default) | its immediate URDF parent link (hierarchy-local; root → world) |

Vectorised over `n_envs` and `n_links` — no per-link Python loop.

## Bundled Presets

Four ready-to-use `VehicleConfig` builders:

| Function | Wheels | Steering | Drivetrain | Coupling |
|---|---|---|---|---|
| `car_4w_fwd_ackermann` | 4 | Ackermann front | FWD | Independent |
| `car_4w_rwd_ackermann` | 4 | Ackermann front | RWD | Independent |
| `car_4w_awd_ackermann` | 4 | Ackermann front | AWD | Independent |
| `truck_6w_partial_ackermann` | 6 | Ackermann on axle 0 | Drive on axles 1+2 (mid + rear) | Independent |
| `tank_skid_belt` | any (validated 10 / 14) | SkidSteer | PerSide (gear cap 0.3) | SameSideBelt |

These cover the reference patterns the SDK was built from; copy and tweak
for your own topology. Every preset takes a keyword-only
`stability="control" | "raw" | "research"` argument (default `"control"`
— see [`docs/stability-profiles.md`](docs/stability-profiles.md)).

## Samples

**22 runnable sample programs** under [`samples/`](samples/) cover the most
common call patterns (counting basis: every `samples/*.py` with a `__main__`
entry point, excluding the `_hud.py` / `tank_tuning.py` helpers and the
deprecated `two_scene_terrain` alias). Full catalog with descriptions + viewer
support in [`samples/README.md`](samples/README.md). All run as Python modules
right after `git pull` — they depend only on the SDK and the bundled
[`samples/urdf/car_4w.urdf`](samples/urdf/car_4w.urdf); the ten below are the
most common entry points:

```bash
python -m genesis_vehicle.samples.quickstart            # 1. minimum-viable API
python -m genesis_vehicle.samples.slope_hold            # 2. StaticFrictionLock regression
python -m genesis_vehicle.samples.batched_rollout       # 3. n_envs > 1 API (L3)
python -m genesis_vehicle.samples.road_loop             # 4. multi-vehicle loop (L2)
python -m genesis_vehicle.samples.perf_vectorization    # 5. L3 scaling bench (headless)
python -m genesis_vehicle.samples.multi_env_render      # 6. all parallel envs in a grid
python -m genesis_vehicle.samples.perf_multi_vehicle    # 7. L2 scaling bench (headless)
python -m genesis_vehicle.samples.perf_l2_l3_combined   # 8. L2 × L3 scaling bench (headless)
python -m genesis_vehicle.samples.city_traffic_ego      # 9. ego + traffic highway (L2 × L3)
python -m genesis_vehicle.samples.bench_raycast_mode    # 10. raycast_mode benchmark (fail-closed)
```

Most scripts accept `--viewer` for an offscreen camera render; the five
benchmarks (`perf_vectorization`, `perf_multi_vehicle`, `perf_l2_l3_combined`,
`bench_raycast_mode`, `bench_path_follower`) are intentionally headless (rendering would distort the
numbers). With `--viewer`, `VehicleScene` draws the wheels by overriding the
engine's own wheel render nodes from the closed form (v2.0.6, at every
`n_envs`; no solver write). (Through v2.0.5 this paragraph said the samples set
`cfg.enable_wheel_joint_internal_sync = args.viewer` — no sample has done so
for several releases, and since v2.0.6 `True` raises `ConfigError`.) The
catalog [`samples/README.md`](samples/README.md) has the per-sample viewer
status, the wheel-rendering note, and pointers to which sample is the right
visual counterpart for each benchmark.

## Documentation

Detailed docs live under [`docs/`](docs/):

| Page | What's in it |
|---|---|
| [`docs/index.md`](docs/index.md) | Documentation home — full TOC |
| [`samples/README.md`](samples/README.md) | Catalog of all 22 runnable samples + viewer support matrix |
| [`docs/quickstart.md`](docs/quickstart.md) | Minimal example, runnable end-to-end |
| [`docs/concepts.md`](docs/concepts.md) | Mental model: 5-step pipeline, ISO 8855, hook intuition, batched-by-default |
| [`docs/batching.md`](docs/batching.md) | The L1 / L2 / L3 vectorization axes — when to use which, measured speedups, the L2 × L3 combined pattern |
| [`docs/path-following.md`](docs/path-following.md) | `PathFollower`: the MPC layer (v2.0.0 default) and the 1.x PD law, the two inverse plants, sweep measurement, identify, the v2.0.2 gear command mode and its stage-2 brake model (§5.9), the v2.0.3 opt-in K-turn (§5.10), the v2.1.0 opt-in end-gated brake bound (§5.11), migrating from 1.x and between 2.x releases |
| [`docs/tire-and-contact.md`](docs/tire-and-contact.md) | Tire model (Pacejka vs Coulomb) & ground-contact mechanism (raycast wheel etc.) — the two axes, standard names, recommendation |
| [`docs/server.md`](docs/server.md) | OSC physics server for external clients (Unreal / Unity) — purpose, modes, full wire-schema reference |
| [`docs/api-reference.md`](docs/api-reference.md) | Full public API surface (every class + function + default) |
| [`docs/pipeline-and-hooks.md`](docs/pipeline-and-hooks.md) | Hook insertion points in the 5-step pipeline |
| [`docs/stability-profiles.md`](docs/stability-profiles.md) | `control` / `raw` / `research` profiles + the "one rule" for MPPI / Real2Sim |
| [`docs/physics-contracts.md`](docs/physics-contracts.md) | Brake sign, non-negative N, `i_wheel` truth policy, steering / coupling order, URDF contracts (§7.9), the v2.0.4 `StaticFrictionLock` engage rule and spring cap (§7.22) |
| [`docs/testing.md`](docs/testing.md) | Test inventory + module map |

## Repository Layout

```
genesis_vehicle/
├── README.md                       <-- you are here
├── LICENSE                         Apache 2.0
├── NOTICE                          AlpaSim (Apache-2.0) attribution for control/mpc/
├── CHANGELOG.md                    per-version release notes
├── requirements-mpc.txt            optional packages of the PathFollower MPC layer
├── __init__.py                     single-import public surface
├── _version.py                     __version__ / VERSION_INFO source of truth
├── core.py                         VehiclePhysics — 5-step pipeline orchestrator
├── config.py                       WheelConfig / VehicleConfig / resolve()
├── inputs.py                       VehicleInputs + typed inputs (v2.0.2: keyword-only gear)
├── gear.py                         v2.0.2 opt-in gear mode: shift rule, drive-torque clamp, skid-steer refusal
├── urdf.py                         parse_urdf() + URDFParsedConfig
├── raycast.py                      WheelRayPattern
├── dynamics.py                     brake_projection / deadbeat_scale / suspension_normal_force
├── visual.py                       wheel renderer: engine-node override (v2.0.6; the joint-writing WheelJointInternalSync was removed)
├── tire_models/                    TireModel ABC + Pacejka + Coulomb
├── strategies/                     steering / drivetrain / coupling / stability
├── presets.py                      5 ready-to-use VehicleConfig builders
├── control/                        PathFollower (MPC layer + PD; gear mode v2.0.2; opt-in K-turn v2.0.3; opt-in end-gated brake bound v2.1.0), inverse plants, brake model, sweep_measure, identify
├── docs/                           the wiki
└── tests/                          pure-Python unit tests (no Genesis needed)
```

## Tests

From the repo root:

```bash
python -m pytest tests/ -v
```

1556 tests collected (1553 passed, 3 skipped) covering URDF parsing, config resolve, suspension sizing, strategy
math, dynamics primitives, version reporting, stability-profile semantics,
multi-vehicle grouping/input routing, the ray-MISS sentinel and grounded
predicates, the swept-envelope wheel contact (including a no-axle-torque
witness bit-identical to the v2.0.4 tree on genesis-world 1.5.0 since v2.0.5,
and a driving-rollout lock re-captured from the v2.0.6 tree in v2.0.6 because
the lock's ground gate changed the braked spawn drop — through v2.0.4 the
witness reached back to the v1.5.0 tree on 1.4.5, a statement 2.0.5 no longer
checks), the config-rebuild state carry and the reset
semantics (v1.6.2), the raycast-mode benchmark harness (schedule,
gates, pairing and publication rule, driven through an injected runner), the
build-time config derivations staying live under a post-`build()` write
(v1.6.4), simulation time being fixed at `build()` and the server's
`sim_options` write guard (v1.6.5), the L2 serving counters and the
post-override capture skip gate (v1.6.6), the L2 `--send-hz` state-send gate
(v1.6.7 — the rate limiter's realised rate, the frozen gate body and the new
`[SERVE]` keys), the wall-clock `[STATS]`/`[SERVE]` window and the benchmark's
pooled weighting (v1.6.8 — the trigger in both modes, a byte-identity freeze of
both `[STATS]` strings, the bare `[SERVE] [L3]` tag, the shared
`--stats-interval` resolver and its loud clamp, and the re-weighting driven on
windows of deliberately unequal length), the genesis-world
supported-floor warning and the coarse-SDF material for SDK-owned collision
bodies (v1.6.9 — version parsing, both call sites, the post-build non-convex
check, a real-engine 16-node grid, and every server obstacle branch building
only convex geoms), rolling resistance as an axle torque applied before the tyre
and the server wheel overrides that reach the physics (v1.7.0 — closed-form
coast-down, no wheel-spin alternation on a free coast, the rolling-resistance coefficient guard, the
`DifferentiablePlant` saturated-warm-start restart, the per-step input copy),
the axle torques applied before the tyre, the Coulomb brake, the friction
ellipse at the transmitted force, the low-speed deadbeats and the
`StaticFrictionLock` rest anchor (v1.8.0 — no ω alternation under throttle or
brake, both-pedal and downhill stops that hold, the post-stop back-travel
matrix, the lateral 2-cycle gone on the reference car and a low-yaw-inertia
test car, the tyre's closed form against a fine grid, a plant that never
touches the driver), the path follower's MPC layer (v2.0.0 — the 1.x PD
follower against a golden captured on 1.8.0, the constructor contract and its
check order, the solve classification on fake and real solvers, a field-parity
replay that runs only on its capture platform, the optional-package gate),
the follower's heading gate, watchdog, per-block budgets and goal-passed stop
(v2.0.1 — every budget and the run-of-entries rule on a fake solver, the 2.0.0
stall reproduced on the real NMPC, the PD golden up to its four `GOAL-MISSED`
steps, a pre-registered run-away property test on a kinematic plant),
the opt-in gear mode and the gear-mode follower (v2.0.2 — the shift and
drive-torque rules on both solvers, every skid-steer refusal site, a
a platform-gated byte-identity check of signed inputs, since v2.0.6 against a
capture of the 2.0.6 tree on genesis-world 1.5.0 (in v2.0.5 of the 2.0.4 tree;
the 2.0.1, 2.0.4 and 2.0.5 fixtures are kept, not asserted),
the stage-2 brake model and its low-speed coast, `SHIFT` / `GUARD` /
`GEAR-ABORT` against an independent oracle),
the opt-in K-turn (v2.0.3 — the trigger and planner, mirror symmetry, the leg
budgets across re-plans, every K-turn limit on exactly the next call, and the
default follower replayed against a 2.0.2 capture),
the pedal-free `StaticFrictionLock` and its spring cap (v2.0.4 — `brake_rem`,
the engage window, holds at any pedal where the brake can hold, the drive
overpowering the brake, the per-wheel and rigid-body caps over every wheel
subset and steer angle, the cap report, a light vehicle and a low-yaw-inertia
car at rest, the server's generic hook set), the required engine and where its
check is called, `--no-target-collision` and its refusals, and per-env
variants (v2.0.5), the `StaticFrictionLock` without its reversal re-anchor
and with its ground gate, the 2.0.6 identity-lock capture, and the engine-node
wheel renderer (v2.0.6 — rendered vs headless bit-identical in one process, the
wheel pose at every render flush, wheel segmentation, the removed joint-writing
path and its raising settings), the opt-in end-gated brake bound (v2.1.0 — the
gate's truth table, the warm-start transfer, the refusals and their placement,
the default's solver parameters pinned to a 2.0.6 capture, the sample's 2.1.0
flags), and the server subpackage surface.
Almost all pure-Python; a handful build a real `VehicleScene` on the CPU
backend, so `genesis-world` 1.5.0 must be importable. No GPU needed — measured
on the v2.1.0 tree with the optional MPC packages installed, `1553 passed,
3 skipped`, 0 failed, 1556 node ids (implementation, 9 chunks (two moved to the background by the tool timeout,
nothing else running); impl-review the same); no timing quoted; CPU/WSL2, genesis-world 1.5.0,
`test_fusion_probe.py` excluded;
the 3 skips are the env-gated live server tests,
run them with `GENESIS_VEHICLE_LIVE_SERVER_TEST=1`; without casadi + do_mpc the
solver-dependent MPC tests skip, `docs/testing.md`). Earlier: `1470 passed,
3 skipped`, 1473 ids at v2.0.6 (implementation, impl-review and a full re-run
on the final tree), `1388 passed,
3 skipped`, 1391 ids at v2.0.5 (genesis-world 1.5.0), `1369 passed,
3 skipped` at v2.0.4 (genesis-world 1.4.5), `1285 passed,
3 skipped` in five parts, 527.3 s summed, at v2.0.3, `1225 passed,
3 skipped` in 433.5 s at v2.0.2, `1127 passed,
3 skipped` in 347.1 s at v2.0.1, `1010 passed,
3 skipped` in 226.1 s at v2.0.0, `824 passed,
3 skipped in 278.5s` at v1.8.0, `712 passed, 3 skipped in
89.31s` at v1.7.0,
`634 passed, 3 skipped in 88.17s` at v1.6.10, 633 in 58.69 s at v1.6.9,
`605 passed, 3 skipped in 135.07s` at v1.6.8 (genesis-world 1.4.0),
`578 passed, 1 skipped in 137.19s` at v1.6.7, `544 passed in 144.45s` at
v1.6.6, `519 passed in 144.33s` at v1.6.5,
`507 passed in 170.53s` at v1.6.4, 116 s / 450 tests at
v1.6.2-v1.6.3, 88 s at v1.6.1 — the wall time swings more between runs than
between releases, so treat it as an order of magnitude, not a metric.

## Releases and Versioning

[Semantic Versioning](https://semver.org/), with one project rule on top: a
MINOR bump may change observable public behaviour, and the CHANGELOG names
every such change (1.8.0 lists eight items strict SemVer would call breaking
in its "Why a minor version" section). 2.0.0 is a major bump: the default
`PathFollower(path, plant)` call stops working. 2.0.1 is a patch that departs
from the minor-bump rule (two new kwargs, new modes, changed default behaviour
including `mpc_model="off"`), by user decision; its CHANGELOG section says why.
2.0.2 is a patch that carries a breaking change — the default MPC
`PathFollower` returns a 4-field `GearCommand` — plus new public surface (the
opt-in gear mode), again by user decision; CHANGELOG 2.0.2 §1 lists every
departure. 2.0.3 is a patch that adds an opt-in kwarg (`mpc_kturn`) and new
modes without changing any default; by the minor-bump rule it would be 2.1.0,
and its number follows the 2.0.2 roadmap (CHANGELOG 2.0.3 §1.3). 2.0.4 is a
patch that changes braking physics with no opt-out and REMOVES
`StaticFrictionLock`'s `brake_thr` / `REST_SPEED` (breaking), by user decision
(CHANGELOG 2.0.4 §1). 2.0.5 is a patch that REQUIRES a new engine
(genesis-world 1.5.0; 1.4.5 dropped) and adds public surface
(`GenesisVersionError`, `require_genesis`, per-env variants), by user decision
(CHANGELOG 2.0.5 §1). 2.0.6 is a patch that REMOVES the joint-writing wheel
visuals (their settings raise), removes `StaticFrictionLock._s_eng` /
`_rested`, and changes braked low-speed physics, by user decision (CHANGELOG
2.0.6 §1-§2). 2.1.0 is an additive minor release: one opt-in keyword
(`PathFollower(mpc_decel_limit=)`), one new `mpc_stats` key, one new
`solver_params` keyword; the default value stays `"throttle"` (its commands
and per-step logs identical to 2.0.6 on the measured runs; CHANGELOG 2.1.0 §1,
§4).
(This line said "pre-1.0, so minor bumps
may break" — stale since 1.0.) Current version: `genesis_vehicle.__version__`. Per-release notes
in [`CHANGELOG.md`](CHANGELOG.md).

**Upgrading to 2.1.0** (additive; genesis-world 1.5.0 as for 2.0.5): nothing to
do for the default — its commands and per-step logs are identical to 2.0.6 on
16 + 3 closed-loop runs compared per step, a parameter fixture and the round-1
acceptance rows (CHANGELOG 2.1.0 §4); `mpc_stats` gains `decel_gate: None`, so
code comparing its exact key set sees one more key. New opt-in:
`PathFollower(..., mpc_decel_limit="brake")` (gear mode only) bounds the MPC by
the params record's brake-1 row on END-GATED solves only — the active forward
block's reference end time within N·DT_MPC (2.0 s) of the solve time in
REFERENCE time (N = horizon nodes, DT_MPC = 0.1 s node spacing), and the rig
(rear-axle) reference's end point ahead of the rear axle — and by the throttle
row otherwise: mid-path decelerations get nothing, and a vehicle behind its
schedule gets the brake row earlier and farther from the end; equality with the
default before the first open solve was measured, not guaranteed. Construction
takes about 0.15–0.22 s longer with it (constructor medians at tyre friction
μ 1.0 / 0.6;
`setup_ms` +121 / +145 ms; implementation). Use a params record measured at the
vehicle's grip (`--mu-lat` / `--mu-long`; ground-plane friction does not reach
the tyre): a μ 1.0 record (μ = tyre friction coefficient) on a μ 0.6 vehicle
locked the front wheels and turned one `DONE` into `GOAL-MISSED` — the verdict
difference is about 0.12 m around the 1.5 m arrival circle and the default's
`DONE` is itself marginal; on a second course the order reversed (impl-review)
— and nothing on the `SweepTable` path detects it. No real-time figure exists for it.
`mpc_model="pacejka"` still raises: the Pacejka prediction model planned for
2.1.0 failed its non-harm gates and was removed (CHANGELOG 2.1.0 §8). Read
CHANGELOG 2.1.0 (§10 is the field-team note) first.

**Upgrading to 2.0.6** (breaking in a patch release, by user decision;
genesis-world 1.5.0 as for 2.0.5): remove `wheel_render_mode="internal_sync"`
(`ValueError`) and `enable_wheel_joint_internal_sync=True` (`ConfigError`);
imports of `WheelJointInternalSync` / `KindVisualBatch` raise `ImportError`
(read `physics.wheel_spin_angle` instead of `physics.visual.wheel_visual_angle`;
`physics.visual is None` keeps working); `visual_susp_mode` has no effect (one
WARNING for a value other than `"auto"`); code that copies
`StaticFrictionLock` state by name must use `STATE_ATTRS` (5 names; `_s_eng` /
`_rested` raise). A raw `VehiclePhysics` in your own `gs.Scene` no longer gets
animated wheels from the SDK. No SDK path writes the wheel joints any more, so
rendered `n_envs > 1` trajectories change from 2.0.5 (2.0.5 moved the reference
car +0.142 m over 10.7 m against headless); rendered runs were bit-identical to
headless on every row measured (CHANGELOG 2.0.6 §6, including an L2-server
`RigidOptions` row and an L3-style reset row) — not on every scene, and the live
L3 server with its viewer was not run. Wheel segmentation
works at every `n_envs` (integers at `n_envs == 1` renumber — read them through
`scene.segmentation_idx_dict`). Braked low-speed physics changed (the lock's
reversal re-anchor removed, a ground gate added; 18 identity locks
re-captured): re-run identify where the `[genesis_vehicle:sfl-cap]` line
reports; sweep tables are unaffected. Run the render tests on every engine
bump. One acceptance gate failed and was waived by the user (a light tank's
50-s rest). Read CHANGELOG 2.0.6 (§11 is the field-team note) first.

**Upgrading to 2.0.5:** upgrade the ENGINE first, in this order — Python
>= 3.12, then `genesis-world==1.5.0 quadrants==1.4.0 coacd==1.0.14
PyGEL3D==0.8.2`, then `git pull`. Pulling 2.0.5 onto a 1.4.5 venv fails at
import with `GenesisVersionError`; upgrading the engine under 2.0.4 breaks 2.0.4
(`skip_forward` is gone from 1.5.0). The first run recompiles kernels (a first
build took about 60 s cold on the measured machine). Physics on the default
headless path is the 2.0.4 tree's on 1.5.0 bit for bit (46 items); across the
engine change only tolerances hold (≤ 5.63e-5 m in x and ≤ 5.93e-5 m in z on 16
parity scenarios; one
forced tank scene 0.126 m apart through a round-off-decided touchdown sign that
exists on 2.0.4, fixed in 2.0.6). Changed: `--no-target-collision`
now works for primitive targets and EXITS (rc 2) with a URDF vehicle or
`--multi-env` (it was a silent no-op); on 1.5.0 a URDF `effort="0"` made the
`visual_susp_mode="control"` PD live (through 2.0.5; no SDK code reads
`visual_susp_mode` since 2.0.6). The rendered `WheelJointInternalSync`
path moves the chassis trajectory on both engines (a defect that exists on
2.0.4; fixed in 2.0.6 by removing the path). Rollback = 2.0.4 + genesis-world
1.4.5. Read CHANGELOG 2.0.5 (§12 is the field-team note) first.

**Upgrading to 2.0.4:** `StaticFrictionLock` (in every `"control"` preset and,
new, on the OSC server's generic-mapping vehicles) engages on every wheel the
brake holds, at ANY pedal. In the measured rows (reference car, tank, truck
demo URDF, a light car; plane μ 1.0, CPU) a braked vehicle on a slope came to
rest where its brakes could hold it (reference car, 5°, brake 0.3: 4e-6 m/s;
2.0.3 crept at 0.025 m/s) and rolled where they could not (2.0.3 slid on
locked wheels above brake 0.3) — measured, not enforced
(`docs/physics-contracts.md` §7.22.3). The brake no longer overrides the drive:
a wheel whose drive torque exceeds its brake torque turns — `tank_skid_belt`
brakes each of its n wheels with `30000·brake/n` N·m but drives each with the
whole side torque, so the 10-wheel reference tank at full brake pivots at
standstill under steer above 0.1 (in general 1/n; steer 1.0: −409° in 5 s;
2.0.3 −0.09°). The
lock's spring is capped per wheel from the static load, the planar rigid-body
factor and `dt` (one `[genesis_vehicle:sfl-cap]` WARNING where it binds — the
reference car above dt 0.0257 s); the cap bounds a planar stiffness eigenvalue
only and is not a stability guarantee. `StaticFrictionLock(brake_thr=...)`, a
positional call, and `brake_thr` / `REST_SPEED` attributes raise. Brake-free
rollouts and sweep tables are byte-identical to 2.0.3 (measured on the scenes
listed in CHANGELOG 2.0.4 §8; sweeps by construction, brake 0); braked scenes are not
byte-identical to 2.0.1 any more (the legacy fixture was re-captured). Read
CHANGELOG 2.0.4 (§10 is the field-team note) and `docs/physics-contracts.md`
§7.22 first.

**Upgrading to 2.0.3:** nothing changes unless you opt in — a follower that
does not pass `mpc_kturn` issues the 2.0.2 commands (measured per step on 51
runs); `FollowerPlan` gains a trailing `steer_cmd` field and `mpc_stats` a
`kturn` dict. `PathFollower(..., mpc_kturn=KTurnParams())` (gear mode, MPC layer)
lets the follower stop and turn with reverse legs when a forward-only turn is
predicted not to fit. It does not know about obstacles, and one acceptance
criterion FAILED: on a 150° vertex course with a 5 m final leg, without gear
feedback, the run ends `DONE` with a cross-track error of 16.007 m against an
11.5 m limit (re-plans flip on sub-metre pose differences, because
legs planned at 1.2 × the minimum turn radius are driven at full lock). Read CHANGELOG 2.0.3 and
`docs/path-following.md` §5.10 first.

**Upgrading to 2.0.2:** an MPC `PathFollower` constructed without
`command_mode=` now runs gear mode and returns `GearCommand(throttle, steer,
brake, gear)`: `thr, st, br = cmd` raises, `cmd[0]` raises, and `cmd == (0, 0,
1)` / `np.asarray(cmd)` silently do something else. Apply it with
`veh.set_inputs(throttle=cmd.throttle, brake=cmd.brake, steer=cmd.steer,
gear=cmd.gear)` (pass `gear=int(veh.gear[0])` to `step` as feedback), or
construct with `command_mode="signed"` for the 2.0.1 3-tuple. Dropping the gear
from the command drives reverse blocks forward until the follower's `GUARD`
brakes. Loop on `PathFollower.TERMINAL_MODES` (it gained `GEAR-ABORT`). A custom
stage-2 plant needs `accel_at` and a brake-gain source. The brake model is in
the command conversion only (the MPC prediction is unchanged), validated on the
reference car, flat ground, μ 1.0, 2-12 m/s. Vehicle inputs without a gear,
`mpc_model="off"`, `FleetFollower` and the OSC server are unchanged. On a slope
a brake ≤ 0.3 does not bring a vehicle to rest (pre-existing, identical on
2.0.1; fixed in 2.0.4). Read CHANGELOG 2.0.2 first.

**Upgrading to 2.0.1:** a follower can now end on `GOAL-MISSED` (every mode)
or `MPC-ABORT` (MPC layer) as well as `DONE` — loop on
`last_mode in PathFollower.TERMINAL_MODES`, or an uncapped loop waiting for
`DONE` never ends. The 2.0.0 misaligned-start stall is fixed by a heading gate
and a no-progress watchdog, on by default (also under `on_solve_failure="hold"`;
a vehicle you hold still while stepping the follower hands over to the PD law
after 2.0 s); `mpc_align_gate=None, mpc_stall_timeout=None` comes closest to
2.0.0 but no kwarg combination restores it exactly. `mpc_model="off"` changed
in two places (the goal-passed stop and the block-end speed sign). The
follower is not guaranteed not to run away. Read CHANGELOG 2.0.1 first.

**Upgrading to 2.0.0:** `PathFollower(path, plant)` raises `ValueError` for
every plant type — add `mpc_model="off"` for the 1.x follower (bit-identical to
1.8.0 on every scenario measured for 2.0.0; since 2.0.1 only before a
`GOAL-MISSED` step and away from block-end sign steps), or install `requirements-mpc.txt`, measure
the vehicle (`sweep_measure` with the grid flags + `control.identify`) and pass
`mpc_params=`. `FleetFollower` members must be `"off"`; `extract_state` gains
`yaw_rate` / `v_lat`; `dynamics.brake_torque_signed` is removed. The nonlinear
MPC is not guaranteed to fit a 25 ms control step, and an MPC follower started
facing away from its path can stall (never move) with no warning — a limitation
inherited from the field code (this said "root cause and fix planned for
2.1.0"; superseded — both shipped in 2.0.1, above). Read the 2.0.0 "Breaking"
list and `docs/path-following.md` §6 first.

**Upgrading to 1.8.0:** the brake is now a Coulomb projection that holds a
stopped wheel, the drive and brake torques act on the wheel spin before the
tyre, and new low-speed bounds apply — the default trajectory moves, every sweep
table or identified parameter set measured before 1.8.0 is stale, and
`RAY_MISS_THRESHOLD` is gone (a `VehiclePhysics(sensor=None)` fed
`step(distances=)` needs `set_ray_miss_value(miss)`). Read the 1.8.0
"Why a minor version" list in the CHANGELOG before upgrading.

## License

genesis_vehicle is released under the [Apache License 2.0](LICENSE),
matching the upstream Genesis physics engine.
