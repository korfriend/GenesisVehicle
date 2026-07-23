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
[Genesis](https://genesis-embodied-ai.github.io/) (`genesis-world ≥ 1.0.0`)
with a CUDA-enabled `torch`. The SDK is validated against the Genesis
`1.0.0` backend; see [`CHANGELOG.md`](CHANGELOG.md) for the per-release
backend-compatibility notes.

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

## Getting Started

```python
from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann

URDF = "<path to your 4-wheel-car URDF>"

# Physics backend — process-global, set ONCE, before any scene. Default is
# "cpu" (faster below ~100 envs; GPU is kernel-launch bound at small batch).
# The renderer is separate (always GPU). Use "gpu" only for large-n_envs L3.
VehicleScene.init_backend("cpu")

# VehicleScene is the single entry point: it owns gs.init / the scene(s) / build / step.
vs = VehicleScene(raycast_mode="single_scene", dt=0.025, substeps=4)  # substeps=4 is the default
vs.add_ground_plane(friction=1.0)
veh = vs.add_vehicle(URDF, preset=car_4w_rwd_ackermann, pos=(0, 0, 1.0))
vs.build()

for step in range(480):                       # 12 s @ 40 Hz
    veh.set_inputs(throttle=0.5, brake=0.0, steer=0.0)   # scalars or (n_envs,) tensors
    vs.step()

print(veh.get_pos()[0].cpu().numpy())
```

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

Nine runnable scripts under [`samples/`](samples/) cover the most
common call patterns. Full catalog with descriptions + viewer support
in [`samples/README.md`](samples/README.md). All run as Python modules
right after `git pull` — they depend only on the SDK and the bundled
[`samples/urdf/car_4w.urdf`](samples/urdf/car_4w.urdf):

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
```

Most scripts accept `--viewer` for an offscreen camera render; the three
perf benches are intentionally headless (rendering would distort
throughput numbers). Since v0.7.14 the samples set
`cfg.enable_wheel_joint_internal_sync = args.viewer` (it defaults to `False` now), so
headless runs skip the per-step viewer-joint FK and report pure-physics
timing. The catalog [`samples/README.md`](samples/README.md) has the
per-sample viewer status, the `--viewer` ↔ `enable_wheel_joint_internal_sync` note, and
pointers to which sample is the right visual counterpart for each benchmark.

## Documentation

Detailed docs live under [`docs/`](docs/):

| Page | What's in it |
|---|---|
| [`docs/index.md`](docs/index.md) | Documentation home — full TOC |
| [`samples/README.md`](samples/README.md) | Catalog of all 10 runnable samples + viewer support matrix |
| [`docs/quickstart.md`](docs/quickstart.md) | Minimal example, runnable end-to-end |
| [`docs/concepts.md`](docs/concepts.md) | Mental model: 5-step pipeline, ISO 8855, hook intuition, batched-by-default |
| [`docs/batching.md`](docs/batching.md) | The L1 / L2 / L3 vectorization axes — when to use which, measured speedups, the L2 × L3 combined pattern |
| [`docs/tire-and-contact.md`](docs/tire-and-contact.md) | Tire model (Pacejka vs Coulomb) & ground-contact mechanism (raycast wheel etc.) — the two axes, standard names, recommendation |
| [`docs/server.md`](docs/server.md) | OSC physics server for external clients (Unreal / Unity) — purpose, modes, full wire-schema reference |
| [`docs/api-reference.md`](docs/api-reference.md) | Full public API surface (every class + function + default) |
| [`docs/pipeline-and-hooks.md`](docs/pipeline-and-hooks.md) | Hook insertion points in the 5-step pipeline |
| [`docs/stability-profiles.md`](docs/stability-profiles.md) | `control` / `raw` / `research` profiles + the "one rule" for MPPI / Real2Sim |
| [`docs/physics-contracts.md`](docs/physics-contracts.md) | Brake sign, non-negative N, `i_wheel` truth policy, steering / coupling order, URDF contracts (§7.9) |
| [`docs/testing.md`](docs/testing.md) | Test inventory + module map |

## Repository Layout

```
genesis_vehicle/
├── README.md                       <-- you are here
├── LICENSE                         Apache 2.0
├── CHANGELOG.md                    per-version release notes
├── __init__.py                     single-import public surface
├── _version.py                     __version__ / VERSION_INFO source of truth
├── core.py                         VehiclePhysics — 5-step pipeline orchestrator
├── config.py                       WheelConfig / VehicleConfig / resolve()
├── inputs.py                       VehicleInputs + typed inputs
├── urdf.py                         parse_urdf() + URDFParsedConfig
├── raycast.py                      WheelRayPattern
├── dynamics.py                     brake_torque_signed / suspension_normal_force
├── visual.py                       WheelJointInternalSync (viewer wheel-joint driver)
├── tire_models/                    TireModel ABC + Pacejka + Coulomb
├── strategies/                     steering / drivetrain / coupling / stability
├── presets.py                      4 ready-to-use VehicleConfig builders
├── docs/                           the wiki
└── tests/                          pure-Python unit tests (no Genesis needed)
```

## Tests

From the repo root:

```bash
python -m pytest tests/ -v
```

212 pure-Python tests covering URDF parsing, config resolve, suspension
sizing, strategy math, dynamics primitives, version reporting,
stability-profile semantics, multi-vehicle grouping/input routing, and the
server subpackage surface. No Genesis runtime needed — they run on CPU in ~40 s.

## Releases and Versioning

[Semantic Versioning](https://semver.org/) (pre-1.0, so minor bumps may
break). Current version: `genesis_vehicle.__version__`. Per-release notes
in [`CHANGELOG.md`](CHANGELOG.md).

## License

genesis_vehicle is released under the [Apache License 2.0](LICENSE),
matching the upstream Genesis physics engine.
