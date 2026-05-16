# genesis_vehicle — API Reference

Ray-cast wheel + Pacejka-tire vehicle SDK on top of Genesis. One package, one
import path, batched (`n_envs >= 1`) by default.

---

## 1. Quick start

```python
import genesis as gs
from genesis_vehicle import (
    VehiclePhysics, VehicleInputs, WheelRayPattern,
    car_4w_rwd_ackermann, parse_urdf,
)

# 1. Build the Genesis scene as usual (vehicle URDF + ground plane).
gs.init(backend=gs.gpu)
scene = gs.Scene(...)
scene.add_entity(gs.morphs.Plane())
car = scene.add_entity(gs.morphs.URDF(file=URDF_PATH, pos=(0, 0, 1.5)))

# 2. Build the wheel raycaster from URDF wheel positions.
parsed = parse_urdf(URDF_PATH)
sensor = scene.add_sensor(gs.sensors.Raycaster(
    pattern=WheelRayPattern([w.position for w in parsed.wheels]),
    entity_idx=car.idx, max_range=20.0, return_world_frame=True,
))
scene.build(n_envs=1)

# 3. Choose a preset (or build VehicleConfig manually) and step the vehicle.
cfg = car_4w_rwd_ackermann(URDF_PATH)
physics = VehiclePhysics(scene, car, sensor, cfg, n_envs=1)
for step in range(480):
    physics.step(VehicleInputs(throttle=0.5, brake=0.0, steer=0.0))
    scene.step()
```

That's the whole API surface for a basic demo. The rest of this document covers
the pieces you reach for when you want custom topology, RL inputs, or to
write your own strategy.

---

## 2. Concepts

### 2.1 The 5-step pipeline

Every step, for every wheel:

1. **Raycast** — distance `d` from chassis-local wheel point straight down.
2. **Suspension** — compression `c = max(rest_d - d, 0)`; `N = K_susp * c + C(c_dot) * c_dot` with **per-wheel asymmetric damper** (`c_compression` for compression, `c_extension` for rebound).
3. **Slip** — `kappa = (v_roll - v_long) / max(|v_long|, eps_v)`, `alpha = atan2(v_lat, |v_long|)`.
4. **Tire force** — `(F_long, F_lat) = TireModel(v_long, v_lat, v_roll, N, wheel_params)`. Pacejka does friction-ellipse clamping internally.
5. **omega update + chassis force-at-point** — `domega = (T_drive - T_brake - R*F_long) / I_wheel`; chassis gets `N*up + F_long*fwd + F_lat*lat` and `r x F` torque at `r = wheel - chassis_center`.

`VehiclePhysics.step()` runs this loop with strategy hooks plugged at fixed seams (see §3.5).

### 2.2 Coordinate / sign conventions (ISO 8855)

| Axis | Direction |
|---|---|
| `+X` | forward |
| `+Y` | left |
| `+Z` | up |

| Input | Range | Sign |
|---|---|---|
| `throttle` | `[-1, +1]` | `+` accelerate forward, `-` reverse |
| `brake` | `[0, 1]` | always positive magnitude |
| `steer` | `[-1, +1]` | **`+steer` = right turn** |

Implementations hide internal flips (Genesis RHS, URDF `<axis 0 0 -1>`, KDU's legacy `+steer=LEFT`) inside the strategy / visual layer. The user-facing API is unambiguous.

### 2.3 Strategies (composition over inheritance)

A `VehicleConfig` is built from four orthogonal strategies plus optional stability hooks:

| Strategy axis | Concrete options |
|---|---|
| `SteeringStrategy` | `Ackermann`, `PartialAckermann`, `SkidSteer`, `NoSteer` |
| `DrivetrainStrategy` | `FWD`, `RWD`, `AWD`, `PerSide` |
| `CouplingStrategy` | `Independent`, `SameSideBelt` |
| `TireModel` | `PacejkaAnisotropic`, `CoulombIsotropic` |
| `StabilityHook` (opt-in list) | `RollingResistance`, `LowSpeedRegularizer`, `StaticFrictionLock` |

Each strategy is a small class you can subclass to add a new vehicle behaviour.

### 2.4 URDF as default, API as override

For every URDF-derivable field (positions, radii, masses, inertia, joint names),
the API contract is:

1. URDF parsing produces a default `WheelConfig` per wheel.
2. User-provided `WheelConfig` fields with non-`None` values **win**.
3. Anything still `None` falls back to module-level defaults (`config.DEFAULT_*`).

This applies whether you use `VehicleConfig.from_urdf(...)` or build a `VehicleConfig`
explicitly. Either way, `resolve()` performs the merge.

### 2.5 Batched by default

Every state tensor is `(n_envs, n_wheels)` or `(n_envs, 3/4)`. Single-env case
(`n_envs=1`) is just a special case. Scalar OR `(n_envs,)` tensor inputs are
both accepted. Use `physics.reset(env_ids=...)` for partial reset (RL / MPPI).

---

## 3. Public API reference

Imported as `from genesis_vehicle import <name>` unless otherwise noted.

### 3.1 `VehiclePhysics` — the driver

```python
class VehiclePhysics:
    def __init__(scene, entity, sensor, config: VehicleConfig, n_envs: int = 1)
    def step(inputs: VehicleInputs | <typed>) -> None
    def reset(env_ids: torch.Tensor | None = None) -> None
```

State (read-only, all `(n_envs, n_wheels)`):
- `omega` — wheel angular velocities
- `last_distances`, `last_compression`, `last_N`
- `last_F_long`, `last_F_lat`, `last_kappa`, `last_alpha`
- `last_T_drive`, `last_T_brake`

Resolved internals:
- `resolved: ResolvedConfig` — post-merge config
- `wheel_meta: WheelMeta` — cached per-wheel tensors (positions, radius, side masks, axle indices, etc.)

`step()` accepts either the strategy's `InputType` (e.g. `AckermannInputs`) or a
unified `VehicleInputs`; the latter is auto-converted via `from_unified()`.

### 3.2 Inputs

```python
@dataclass
class VehicleInputs:
    throttle: float | torch.Tensor    # [-1, +1] signed
    brake:    float | torch.Tensor    # [0, 1]
    steer:    float | torch.Tensor    # [-1, +1], + = right turn

@dataclass
class AckermannInputs:           # for Ackermann / PartialAckermann
    throttle, brake, steer
@dataclass
class SkidSteerInputs:           # for SkidSteer + PerSide
    throttle, brake, steer_diff
@dataclass
class NoSteerInputs:             # for NoSteer + any drivetrain
    throttle, brake
```

Each typed input has a `from_unified(VehicleInputs) -> Self` classmethod for
RL pipelines that prefer the unified `VehicleInputs` schema.

### 3.3 Config

```python
@dataclass
class WheelConfig:
    # All fields Optional. None = derive from URDF or module default.
    name, position, side, axle_index
    radius, mass, i_wheel
    rest_stroke, k_susp, c_compression, c_extension, comp_rate_clamp
    mu_long, mu_lat, rolling_resistance_cr
    pb_x, pc_x, pe_x, pb_y, pc_y, pe_y          # Pacejka coefficients
    susp_joint_name, steer_joint_name, spin_joint_name

@dataclass
class ChassisConfig:
    base_link_name: str = "base_link"
    mass: float | None = None              # None = read from URDF
    frame: str = "ISO8855"
    omega_max: float = 100.0
    eps_v: float = 0.5

@dataclass
class VehicleConfig:
    urdf_path: str
    wheels: list[WheelConfig]
    steering, drivetrain, coupling, tire    # strategies / model
    chassis: ChassisConfig = ChassisConfig()
    stability_hooks: list[StabilityHook] = []
    dt: float = 1.0/48.0
    enable_visual_sync: bool = True

    @classmethod
    def from_urdf(urdf_path, *, steering, drivetrain, coupling, tire,
                  wheel_overrides: dict[str, WheelConfig] = None,
                  chassis: ChassisConfig = None, **kwargs) -> VehicleConfig
```

`from_urdf()` is the easiest path: pass strategies + a dict of per-wheel
overrides keyed by URDF wheel link name; the wheel list is auto-populated.

`resolve(config) -> ResolvedConfig` runs the URDF→config→default merge and
each strategy's `validate()`. Called automatically by `VehiclePhysics.__init__`.

`ConfigError` — raised on bad config (missing required fields, wheel count
mismatch, sides missing for skid-steer, etc.).

### 3.4 URDF utilities

```python
parse_urdf(urdf_path: str) -> URDFParsedConfig
parse_inertia_max_principal_genesis(entity, link_name: str) -> float
```

`parse_urdf()` discovers wheels by walking the URDF joint tree:
- Wheels = prismatic joints whose name ends with `_susp` or `_suspension_joint`
- Spin joint = first descendant `continuous` joint with axis (0, ±1, 0)
- Steer joint = `revolute` joint with axis (0, 0, ±1) along the chain
- Side = `l_*`/`_left_*` → 'L', `r_*`/`_right_*` → 'R', else `None`
- Axle index = clusters by x-coordinate, sorted descending so axle 0 = front-most

`URDFParsedConfig` fields:
- `base_link_name`, `chassis_mass`, `wheels: list[WheelConfig]`
- `steer_axis_signs: dict[str, int]` — used by visual layer for `<axis 0 0 -1>` flip
- `susp_has_dynamics: dict[str, bool]` — picks `set_dofs_position` vs `control_dofs_position`

`parse_inertia_max_principal_genesis()` reads `link.inertial_i` from a built
Genesis entity and returns the max diagonal — robust against Genesis rotating
the inertial frame (which makes URDF `iyy` unreliable as the spin MOI).

### 3.5 Strategies — full surface

#### Steering

```python
class Ackermann(SteeringStrategy):
    def __init__(max_steer_rad=0.7, front_axle=0,
                 wheelbase=None, track_width=None)
    # Geometry derived from wheel positions if not supplied.
    InputType = AckermannInputs

class PartialAckermann(Ackermann):
    def __init__(max_steer_rad=0.7, steered_axles=(0,),
                 wheelbase=None, track_width=None)
    # For multi-axle vehicles where only some axles steer.
    InputType = PartialAckermannInputs

class SkidSteer(SteeringStrategy):
    # No steer angle; turning happens via PerSide drivetrain.
    InputType = SkidSteerInputs
    def affects_drive(self) -> bool: return True

class NoSteer(SteeringStrategy):
    InputType = NoSteerInputs
```

#### Drivetrain

```python
class RWD(DrivetrainStrategy):
    def __init__(t_drive_max, t_brake_max,
                 driven_axles=None,    # default: rear-most axle
                 brake_bias=None)      # default: 60/40 front/rear (2-axle); uniform otherwise

class FWD(RWD):                       # default driven_axles = front-most axle
class AWD(DrivetrainStrategy):
    def __init__(t_drive_max, t_brake_max,
                 drive_weights=None,   # per-wheel weights, default uniform
                 brake_bias=None)

class PerSide(DrivetrainStrategy):
    """For tank-style skid steer."""
    def __init__(t_drive_max, t_brake_max,
                 steer_gain=1.0,
                 omega_max_drive=100.0,
                 throttle_gear_cap=1.0,
                 use_per_side_taper=True)
    # Per-side ω taper (KDU pattern): T_max(omega) = T_drive_max*(1 - omega/omega_max_drive)
    # only on the same direction as omega; opposite direction (decel) keeps full T_max.
```

#### Coupling

```python
class Independent(CouplingStrategy):  # passthrough; the default
class SameSideBelt(CouplingStrategy):
    # Tank track: omega[L] := mean(omega[L]) per env; same for R.
```

#### Stability hooks (opt-in via `VehicleConfig.stability_hooks`)

```python
class RollingResistance(StabilityHook):
    def __init__(tanh_scale=0.5)
    # F_long -= cr * N * tanh(v_long / tanh_scale)
    # cr is per-wheel via WheelConfig.rolling_resistance_cr.

class LowSpeedRegularizer(StabilityHook):
    def __init__(v_kin_com=0.5, ang_kin=0.5,
                 disable_when_control_active=True,
                 control_threshold=0.01)
    # PRE_LOOP: compute moving = clamp(max(|v|/v_kin, |omega|/ang_kin), 0, 1)
    # POST_TIRE: scale F by `moving`; pull omega toward v_long/radius for grounded wheels.
    # When user has throttle/brake pressed: moving := 1 (so the car can pull away).

class StaticFrictionLock(StabilityHook):
    def __init__(brake_thr=0.3, v_thr=0.5, hold_k=200_000.0)
    # When brake > brake_thr AND |v_long| < v_thr:
    #   F_long := clamp(-hold_k * v_long, +- mu_long * N)
    #   omega := 0  (forced via ctx.omega_override)
```

### 3.6 Tire models

```python
class PacejkaAnisotropic(TireModel):
    def __init__(eps_v=0.5)
    # Per-wheel coefficients (B, C, E for long + lat axes), mu_long, mu_lat
    # come from wheel_params — one model instance serves the whole vehicle.
    # Returns (F_long, F_lat, kappa, alpha) with friction-ellipse clamp.

class CoulombIsotropic(TireModel):
    def __init__(eps_v=0.5)
    # F = -mu * N * v_slip / |v_slip|, isotropic.
```

### 3.7 Sensor / utilities

```python
class WheelRayPattern(genesis.options.sensors.raycaster.RaycastPattern):
    def __init__(positions: list[tuple[float, float, float]])
    @classmethod
    def from_config(resolved: ResolvedConfig) -> WheelRayPattern
    # One -z body-frame ray per wheel. Pass to gs.sensors.Raycaster(pattern=...).

read_distances(sensor, n_envs: int) -> torch.Tensor
    # Returns (n_envs, n_wheels). Handles the n_envs=1 sensor shape quirk.
```

### 3.8 Presets

Each preset returns a `VehicleConfig` ready for `VehiclePhysics(...)`:

| Function | Topology | Steering | Drive | Coupling |
|---|---|---|---|---|
| `car_4w_rwd_ackermann(urdf_path)` | 4 wheels | Ackermann front | RWD | Independent |
| `car_4w_awd_ackermann(urdf_path)` | 4 wheels | Ackermann front | AWD | Independent |
| `truck_6w_partial_ackermann(urdf_path)` | 6 wheels | Ackermann on axle 0 | AWD (uniform) | Independent |
| `tank_10w_skid_belt(urdf_path)` | 10 wheels | SkidSteer | PerSide (gear cap 0.3) | SameSideBelt |

Tune them by editing the returned config (`cfg.dt = ...`, replace a strategy,
add a stability hook) before passing to `VehiclePhysics`.

---

## 4. Common patterns

### 4.1 Custom 4-wheel car (no preset)

```python
from genesis_vehicle import (
    VehicleConfig, ChassisConfig, WheelConfig,
    Ackermann, RWD, Independent, PacejkaAnisotropic,
    RollingResistance, LowSpeedRegularizer,
)

cfg = VehicleConfig.from_urdf(
    URDF_PATH,
    steering=Ackermann(max_steer_rad=0.6),
    drivetrain=RWD(t_drive_max=1500.0, t_brake_max=3000.0),
    coupling=Independent(),
    tire=PacejkaAnisotropic(),
    wheel_overrides={
        "front_left_wheel":  WheelConfig(c_compression=20_000, c_extension=6_000),
        "front_right_wheel": WheelConfig(c_compression=20_000, c_extension=6_000),
    },
    chassis=ChassisConfig(omega_max=120.0),
    stability_hooks=[RollingResistance(), LowSpeedRegularizer()],
    dt=1/60,
)
```

### 4.2 RL rollout (batched, partial reset)

```python
N_ENVS = 256
physics = VehiclePhysics(scene, car, sensor, cfg, n_envs=N_ENVS)
throttle = torch.zeros(N_ENVS, device='cuda')
brake = torch.zeros(N_ENVS, device='cuda')
steer = torch.zeros(N_ENVS, device='cuda')

for t in range(T):
    # Each input is a (N_ENVS,) tensor:
    physics.step(VehicleInputs(throttle, brake, steer))
    scene.step()
    if t % 100 == 0:
        done = some_termination_mask              # (N_ENVS,) bool
        physics.reset(env_ids=done)
```

### 4.3 Custom strategy

Subclass any strategy ABC. Example: per-wheel torque vectoring on top of AWD.

```python
import torch
from genesis_vehicle import AWD

class TorqueVectoringAWD(AWD):
    def distribute_torque(self, inputs, omega, wheel_meta, device, dtype):
        T_drive, T_brake = super().distribute_torque(inputs, omega, wheel_meta, device, dtype)
        # Add a yaw-rate based per-wheel bias.
        ...
        return T_drive, T_brake
```

Same pattern works for `SteeringStrategy.per_wheel_steer`,
`CouplingStrategy.apply`, `TireModel.__call__`, and stability hooks.

### 4.4 Stability hook ordering

`stability_hooks` is a list; hooks run in list order. PRE_LOOP hooks run once
per step; POST_TIRE hooks run once per wheel after the tire model. If both
`RollingResistance` and `StaticFrictionLock` are active, place `RollingResistance`
**first** so the lock gets the last word on `F_long`.

```python
cfg.stability_hooks = [RollingResistance(), StaticFrictionLock()]
```

---

## 5. Migration cheatsheet

### From HJW (`HJW/car_raywheel.py`)

| HJW symbol | SDK |
|---|---|
| `CarRayWheelPhysics(scene, car, sensor)` | `VehiclePhysics(scene, car, sensor, cfg)` where `cfg = car_4w_rwd_ackermann(URDF_PATH)` |
| `physics.step(throttle, brake, steer)` | `physics.step(VehicleInputs(throttle, brake, steer))` |
| Module-level constants (`K_SUSP`, `C_COMP`, `MU`, `PB_X`, …) | `WheelConfig` fields, set via `wheel_overrides` |
| `WHEEL_POSITIONS`, `N_WHEELS`, `BRAKE_BIAS_PER_WHEEL` | URDF + `RWD(brake_bias=[…])` |
| Steer sign flip at `car_raywheel.py:298` | hidden inside `Ackermann` (no manual flip) |

### From JMK (`JMK/real2sim/gt_ray/0_test_ray_step{11,12}.py`)

The inlined `apply_forces_gpu` body becomes:

```python
cfg = car_4w_rwd_ackermann(URDF_PATH)
cfg.dt = 0.02   # JMK runs at 50 Hz
physics = VehiclePhysics(scene, car, sensor, cfg, n_envs=1)
# Step 11 (straight line):
physics.step(VehicleInputs(throttle=THROTTLE, brake=0, steer=0))
# Step 12 (Ackermann circle):
physics.step(VehicleInputs(throttle=0.5, brake=0, steer=+0.4))   # ISO + = right
```

JMK's hard-coded chassis mass (1330) and track width (1.48) **are dropped**;
URDF (1200, 1.32) wins per the SDK's URDF-as-truth rule.

### From KDU (`KDU/physics.py`)

| KDU symbol | SDK |
|---|---|
| `parse_tank_urdf()` | `parse_urdf()` (generalised; same naming convention) |
| `TankRayPhysics(tank, sensor, urdf_path)` | `VehiclePhysics(scene, tank, sensor, cfg)` where `cfg = tank_10w_skid_belt(urdf_path)` |
| `physics.step(throttle, brake, steer)` with `+steer = LEFT` | `physics.step(VehicleInputs(throttle, brake, steer=-old_steer))` — note the **sign flip** to match ISO 8855 |
| `omega[LEFT_IDX] = mean()` belt constraint | `SameSideBelt()` coupling |
| `_cap_torque` per-side taper | `PerSide(use_per_side_taper=True)` |
| Static friction lock | `StaticFrictionLock(brake_thr=0.3, v_thr=0.5, hold_k=200_000.0)` |
| `THROTTLE_GEAR_CAP = 0.3` | `PerSide(throttle_gear_cap=0.3)` |
| Turret / barrel DoFs | NOT in SDK — keep in user demo code |

---

## 6. Module map

| File | Purpose |
|---|---|
| `core.py` | `VehiclePhysics` — 5-step pipeline orchestrator |
| `config.py` | `WheelConfig`, `ChassisConfig`, `VehicleConfig`, `ResolvedConfig`, `resolve()`, `ConfigError`, `DEFAULT_*` |
| `inputs.py` | `VehicleInputs` + typed inputs |
| `urdf.py` | `parse_urdf()`, `URDFParsedConfig`, runtime inertia helper |
| `raycast.py` | `WheelRayPattern`, `read_distances()` |
| `visual.py` | `VisualSync` (auto-invoked by core; flips URDF axis quirks) |
| `tire_models/` | `TireModel` ABC + `PacejkaAnisotropic`, `CoulombIsotropic` |
| `strategies/steering.py` | `SteeringStrategy` + 4 concrete |
| `strategies/drivetrain.py` | `DrivetrainStrategy` + 4 concrete |
| `strategies/coupling.py` | `CouplingStrategy` + 2 concrete |
| `strategies/stability.py` | `StabilityHook` + 3 concrete |
| `presets.py` | 4 ready-to-use `VehicleConfig` builders |
| `tests/` | Pure-Python unit tests (no Genesis runtime needed) |

---

## 7. Defaults reference (from `config.py`)

| Constant | Value | Notes |
|---|---|---|
| `DEFAULT_K_SUSP` | 70_000 N/m | spring stiffness |
| `DEFAULT_C_COMPRESSION` | 14_000 N·s/m | compression damping (HJW direction) |
| `DEFAULT_C_EXTENSION` | 4_000 N·s/m | rebound damping |
| `DEFAULT_REST_STROKE` | 0.10 m | suspension rest length |
| `DEFAULT_MU` | 1.0 | friction (long & lat) |
| `DEFAULT_ROLLING_RESISTANCE` | 0.015 | `Cr` for `RollingResistance` hook |
| `DEFAULT_COMP_RATE_CLAMP` | 30 m/s | numerical clamp on `c_dot` |
| `DEFAULT_RADIUS` | 0.35 m | wheel radius (used only if URDF has no `<cylinder>`) |
| `DEFAULT_MASS` | 20 kg | wheel mass fallback |
| `DEFAULT_I_WHEEL` | 1.5 kg·m² | wheel spin MOI fallback |
| `DEFAULT_PACEJKA` | `PB_X=10, PC_X=1.65, PE_X=0.4, PB_Y=8, PC_Y=1.30, PE_Y=0.4` | mid-grip car defaults |

These are last-resort fallbacks. URDF + `WheelConfig` overrides take precedence.
