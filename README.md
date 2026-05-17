# genesis_vehicle

A ray-cast wheel + Pacejka-tire vehicle physics SDK on top of the
[Genesis](https://genesis-embodied-ai.github.io/) physics engine. Batched
(`n_envs ≥ 1`) by default, designed for RL / MPPI control loops and
Real2Sim parameter fitting.

## Strategy axes (mix and match — independent of wheel count)

| Axis | Concrete options |
|---|---|
| **Steering** | `Ackermann`, `PartialAckermann`, `SkidSteer`, `NoSteer` |
| **Drivetrain** | `FWD`, `RWD`, `AWD`, `PerSide` |
| **Coupling** | `Independent`, `SameSideBelt` |
| **Tire model** | `PacejkaAnisotropic`, `CoulombIsotropic` |
| **Stability hooks** (via `stability=` profile) | `RollingResistance`, `LowSpeedRegularizer`, `StaticFrictionLock` |

Wheel count is whatever your URDF declares — Ackermann on a 6-wheel truck,
skid-steer on a 4-wheel rover, AWD on a 10-wheel tank — all valid as long
as the soft per-strategy constraints hold:

- Ackermann / PartialAckermann: each steered axle has an L + R pair
- SkidSteer / SameSideBelt: every wheel has `side='L'` or `side='R'`

Subclass any strategy ABC (`SteeringStrategy`, `DrivetrainStrategy`,
`CouplingStrategy`, `TireModel`, `StabilityHook`) to add new behaviors.

## Bundled presets

Four ready-to-use `VehicleConfig` builders:

| Function | Wheels | Steering | Drivetrain | Coupling |
|---|---|---|---|---|
| `car_4w_rwd_ackermann` | 4 | Ackermann front | RWD | Independent |
| `car_4w_awd_ackermann` | 4 | Ackermann front | AWD | Independent |
| `truck_6w_partial_ackermann` | 6 | Ackermann on axle 0 | AWD (uniform) | Independent |
| `tank_10w_skid_belt` | 10 | SkidSteer | PerSide (gear cap 0.3) | SameSideBelt |

These cover the reference patterns the SDK was built from; copy and tweak
for your own topology.

## Installation

Requires Python 3.12+, [Genesis](https://genesis-embodied-ai.github.io/)
with a CUDA-enabled `torch`, plus `numpy`.

The SDK is a directory; there is no `pip install` yet. After cloning:

```bash
git clone https://github.com/korfriend/GenesisVehicle.git
cd GenesisVehicle
```

Either add the parent of the cloned directory to `PYTHONPATH` or use
`sys.path.insert(0, '<parent>')` in your entry script so that
`import genesis_vehicle` resolves.

Dev dependency (for the test suite): `pip install pytest`.

## 1-minute quickstart

```python
import genesis as gs
from genesis_vehicle import (
    VehiclePhysics, VehicleInputs, WheelRayPattern,
    car_4w_rwd_ackermann, parse_urdf,
)

URDF = "<path to your car_raywheel.urdf>"

gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/48, substeps=50))
scene.add_entity(gs.morphs.Plane())
car = scene.add_entity(gs.morphs.URDF(file=URDF, pos=(0, 0, 1.5)))

parsed = parse_urdf(URDF)
sensor = scene.add_sensor(gs.sensors.Raycaster(
    pattern=WheelRayPattern([w.position for w in parsed.wheels]),
    entity_idx=car.idx, max_range=20.0, return_world_frame=True,
))
scene.build(n_envs=1)

cfg = car_4w_rwd_ackermann(URDF)
physics = VehiclePhysics(scene, car, sensor, cfg, n_envs=1)

for step in range(480):                                       # 10 s @ 48 Hz
    physics.step(VehicleInputs(throttle=0.5, brake=0.0, steer=0.0))
    scene.step()

print(car.get_pos()[0].cpu().numpy())
```

On first construction, `VehiclePhysics` prints a one-line banner:

```
[genesis_vehicle v0.4.0] Initialized: 4 wheels, Ackermann, RWD, Independent, n_envs=1, hooks=[RollingResistance, LowSpeedRegularizer]
```

## Documentation

| Page | What's in it |
|---|---|
| [`docs/index.md`](docs/index.md) | Documentation home — full TOC |
| [`docs/quickstart.md`](docs/quickstart.md) | Minimal example, runnable end-to-end |
| [`docs/concepts.md`](docs/concepts.md) | Mental model: 5-step pipeline, ISO 8855 signs, hooks intuition, batched-by-default |
| [`docs/api-reference.md`](docs/api-reference.md) | Full public API surface (every class + function + default value) |
| [`docs/pipeline-and-hooks.md`](docs/pipeline-and-hooks.md) | Hook insertion points in the 5-step pipeline |
| [`docs/stability-profiles.md`](docs/stability-profiles.md) | `control` / `raw` / `research` profiles + the "one rule" for MPPI / Real2Sim |
| [`docs/physics-contracts.md`](docs/physics-contracts.md) | Brake-sign, non-negative N, `i_wheel` truth policy, steering / coupling order |
| [`docs/migration.md`](docs/migration.md) | Mapping legacy 3-variant style code onto the SDK |
| [`docs/testing.md`](docs/testing.md) | Test inventory + module map |

Version history: [`CHANGELOG.md`](CHANGELOG.md).

## Repository layout

```
genesis_vehicle/
├── README.md                       <-- you are here
├── CHANGELOG.md                    per-version release notes
├── __init__.py                     single-import public surface
├── _version.py                     __version__ / VERSION_INFO source of truth
├── core.py                         VehiclePhysics — 5-step pipeline orchestrator
├── config.py                       WheelConfig / VehicleConfig / resolve()
├── inputs.py                       VehicleInputs + typed inputs
├── urdf.py                         parse_urdf() + URDFParsedConfig
├── raycast.py                      WheelRayPattern
├── dynamics.py                     brake_torque_signed / suspension_normal_force
├── visual.py                       VisualSync
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

58 pure-Python tests covering URDF parsing, config resolve, strategy math,
dynamics primitives, version reporting, and stability-profile semantics. No
Genesis runtime needed — they run on CPU in ~3 s.

## Versioning

[Semantic Versioning](https://semver.org/) (pre-1.0, so minor bumps may
break). Current version: `genesis_vehicle.__version__`. See
[`CHANGELOG.md`](CHANGELOG.md) for release notes.

## License

(Add license terms here.)
