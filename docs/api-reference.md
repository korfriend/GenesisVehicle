# API Reference

Full public surface. Imported as `from genesis_vehicle import <name>` unless
otherwise noted. For the mental model behind these names, read
[`concepts.md`](concepts.md) first.

## 1. `VehiclePhysics` — the driver

```python
class VehiclePhysics:
    def __init__(scene, entity, sensor, config: VehicleConfig, n_envs: int = 1)
    def step(inputs: VehicleStepInputs) -> None
    def reset(env_ids: torch.Tensor | None = None) -> None
```

State (read-only, all `(n_envs, n_wheels)`):

- `omega` — wheel angular velocities
- `last_distances`, `last_compression`, `last_N`
- `last_F_long`, `last_F_lat`, `last_kappa`, `last_alpha`
- `last_T_drive`, `last_T_brake`

Resolved internals:

- `resolved: ResolvedConfig` — post-merge config; the only source of truth at runtime
- `wheel_meta: WheelMeta` — cached per-wheel tensors (positions, radius, side masks, axle indices, etc.)

`step()` accepts either the strategy's `InputType` (e.g. `AckermannInputs`)
or a unified `VehicleInputs`. The latter is auto-converted via
`from_unified()`.

`reset(env_ids=None)` zeroes `omega`, `prev_compression`, and the visual
angle accumulators. Pass a `(K,)` long tensor or a `(n_envs,)` bool mask to
reset only specific environments (RL / MPPI partial reset).

`VehiclePhysics.__init__` prints a one-line version banner the first time
it's called per process. There is no global "quiet" flag yet.

## 2. Inputs

```python
@dataclass
class VehicleInputs:
    throttle: float | torch.Tensor    # [-1, +1] signed
    brake:    float | torch.Tensor    # [0, 1] positive magnitude
    steer:    float | torch.Tensor    # [-1, +1], + = right turn

@dataclass
class AckermannInputs:           # for Ackermann / PartialAckermann
    throttle, brake, steer
@dataclass
class SkidSteerInputs:           # for SkidSteer + PerSide
    throttle, brake, steer_diff  # + = right turn
@dataclass
class NoSteerInputs:             # for NoSteer + any drivetrain
    throttle, brake

# Convenience alias for VehiclePhysics.step() type hints.
VehicleStepInputs = Union[
    VehicleInputs, AckermannInputs, PartialAckermannInputs,
    SkidSteerInputs, NoSteerInputs,
]
```

Each typed input has a `from_unified(VehicleInputs) -> Self` classmethod for
pipelines that prefer the unified schema.

## 3. Config

```python
from dataclasses import field

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
    steering, drivetrain, coupling, tire
    chassis: ChassisConfig = field(default_factory=ChassisConfig)
    stability_hooks: list[StabilityHook] = field(default_factory=list)
    dt: float = 1.0/48.0
    enable_visual_sync: bool = True

    @classmethod
    def from_urdf(urdf_path, *, steering, drivetrain, coupling, tire,
                  wheel_overrides: dict[str, WheelConfig] = None,
                  chassis: ChassisConfig = None, **kwargs) -> VehicleConfig
```

> Mutable defaults on dataclass fields (`chassis: ChassisConfig = ChassisConfig()` or `list = []`) are unsafe — they are shared across instances. The SDK uses `field(default_factory=...)` throughout. When constructing your own dataclasses, follow the same pattern.

`from_urdf()` is the easiest path: pass strategies + a dict of per-wheel
overrides keyed by URDF wheel link name; the wheel list is auto-populated.

`resolve(config) -> ResolvedConfig` runs the URDF → user → default merge and
each strategy's `validate()`. Called automatically by `VehiclePhysics.__init__`.

`ConfigError` — raised on bad config (missing required fields, wheel count
mismatch, sides missing for skid-steer, etc.).

## 4. URDF utilities

```python
parse_urdf(urdf_path: str) -> URDFParsedConfig

estimate_spin_inertia_from_genesis(
    entity, link_name: str,
    spin_axis_local: tuple[float, float, float] | None = None,
) -> float
```

`parse_urdf()` is a **convention-based helper** that discovers wheels by
walking the URDF joint tree:

- Wheels = prismatic joints whose name ends with `_susp` or `_suspension_joint`
- Spin joint = first descendant `continuous` joint with axis `(0, ±1, 0)`
- Steer joint = `revolute` joint with axis `(0, 0, ±1)` along the chain
- Side: `l_*` / `_left_*` → `'L'`; `r_*` / `_right_*` → `'R'`; else `None`
- Axle index: clusters by x-coordinate, sorted descending so axle 0 = front-most

> **For non-conforming URDFs**, construct `VehicleConfig` explicitly or supply
> `wheel_overrides` with the relevant joint names. `parse_urdf()` is a
> best-effort convenience for our supported vehicle naming conventions, not a
> general URDF interpreter.

`URDFParsedConfig` fields:

- `base_link_name`, `chassis_mass`, `wheels: list[WheelConfig]`
- `steer_axis_signs: dict[str, int]` — used by the visual layer for `<axis 0 0 -1>` flip
- `susp_has_dynamics: dict[str, bool]` — picks `set_dofs_position` vs `control_dofs_position`

`estimate_spin_inertia_from_genesis()` is a **fallback estimate** consulted
only when `WheelConfig.i_wheel` is not set (by the user OR by the URDF).
When `spin_axis_local` is provided, the helper projects the inertia tensor
onto that axis (`a^T diag(I) a`); otherwise it returns `max(diag(inertial_i))`,
which is the spin MOI for cylindrical wheels but a heuristic for general
shapes. **`WheelConfig.i_wheel` (when supplied) is always authoritative.**

## 5. Strategies

### 5.1 Steering

```python
class Ackermann(SteeringStrategy):
    def __init__(max_steer_rad=0.7, front_axle=0,
                 wheelbase=None, track_width=None)
    # Geometry derived from wheel positions if not supplied.
    InputType = AckermannInputs

class PartialAckermann(Ackermann):
    def __init__(max_steer_rad=0.7, steered_axles=(0,),
                 wheelbase=None, track_width=None)
    InputType = PartialAckermannInputs

class SkidSteer(SteeringStrategy):
    # No steer angle; turning happens via PerSide drivetrain.
    InputType = SkidSteerInputs

class NoSteer(SteeringStrategy):
    InputType = NoSteerInputs
```

### 5.2 Drivetrain

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
    """For tank-style skid steer.
    Unified-input mapping (ISO +steer = right turn):
        left_cmd  = throttle + steer_gain * steer
        right_cmd = throttle - steer_gain * steer
    So +steer makes the LEFT side faster → right turn."""
    def __init__(t_drive_max, t_brake_max,
                 steer_gain=1.0, omega_max_drive=100.0,
                 throttle_gear_cap=1.0, use_per_side_taper=True)
```

### 5.3 Coupling

```python
class Independent(CouplingStrategy):  # passthrough; the default
class SameSideBelt(CouplingStrategy):
    # Tank track: omega[L] := mean(omega[L]) per env; same for R.
```

`CouplingStrategy.apply()` runs once per step after the per-wheel omega
integration and before visual sync. See
[`physics-contracts.md`](physics-contracts.md#75-coupling-order) for the
one-step lag detail.

### 5.4 Stability hooks (via `stability=` profile)

```python
class RollingResistance(StabilityHook):
    def __init__(tanh_scale=0.5)
    # F_long -= cr * N * tanh(v_long / tanh_scale)
    # cr is per-wheel via WheelConfig.rolling_resistance_cr.

class LowSpeedRegularizer(StabilityHook):
    def __init__(v_kin_com=0.5, ang_kin=0.5,
                 disable_when_control_active=True, control_threshold=0.01)

class StaticFrictionLock(StabilityHook):
    def __init__(brake_thr=0.3, v_thr=0.5, k_spring=500_000.0, k_damp=20_000.0)
```

Hooks are selected via `stability=` profile in the presets (see
[`stability-profiles.md`](stability-profiles.md)), not assembled by hand
in the common path.

## 6. Tire models

```python
class PacejkaAnisotropic(TireModel):
    def __init__(eps_v=0.5)
    # Per-wheel (B, C, E) for long + lat, mu_long, mu_lat come from wheel_params.
    # Returns (F_long, F_lat, kappa, alpha) with friction-ellipse clamp.
class CoulombIsotropic(TireModel):
    def __init__(eps_v=0.5)
    # F = -mu * N * v_slip / |v_slip|, isotropic.
```

## 7. Sensor / dynamics utilities

```python
class WheelRayPattern(genesis.options.sensors.raycaster.RaycastPattern):
    def __init__(positions: list[tuple[float, float, float]])
    @classmethod
    def from_config(resolved: ResolvedConfig) -> WheelRayPattern

read_distances(sensor, n_envs: int) -> torch.Tensor
    # Returns (n_envs, n_wheels). Handles the n_envs=1 sensor shape quirk.

# --- Pure-Python dynamics primitives (testable without Genesis) ---
brake_torque_signed(
    t_brake: torch.Tensor, omega: torch.Tensor, smoothing_scale: float = 0.5,
) -> torch.Tensor
suspension_normal_force(
    compression, comp_rate, k_susp, c_compression, c_extension, air_mask,
) -> torch.Tensor

# --- Version ---
genesis_vehicle.__version__               # str, e.g. "0.4.0"
genesis_vehicle.VERSION_INFO              # tuple, e.g. (0, 4, 0)
genesis_vehicle.version() -> str          # same as __version__
genesis_vehicle.version_info() -> tuple   # same as VERSION_INFO
```

## 8. Presets

Every preset takes a keyword-only `stability` argument that picks the
stability profile (see [`stability-profiles.md`](stability-profiles.md)).

```python
def car_4w_fwd_ackermann(
    urdf_path: str, n_envs: int = 1, *, stability: str = "control",
) -> VehicleConfig
def car_4w_rwd_ackermann(
    urdf_path: str, n_envs: int = 1, *, stability: str = "control",
) -> VehicleConfig
def car_4w_awd_ackermann(
    urdf_path: str, n_envs: int = 1, *, stability: str = "control",
) -> VehicleConfig
def truck_6w_partial_ackermann(
    urdf_path: str, n_envs: int = 1, *, stability: str = "control",
) -> VehicleConfig
def tank_10w_skid_belt(
    urdf_path: str, n_envs: int = 1, *, stability: str = "control",
) -> VehicleConfig

stability_hooks_for_profile(
    profile: str = "control", vehicle_kind: str = "car",
) -> list[StabilityHook]
```

| Function | Topology | Steering | Drive | Coupling |
|---|---|---|---|---|
| `car_4w_fwd_ackermann(urdf_path)` | 4 wheels | Ackermann front | FWD | Independent |
| `car_4w_rwd_ackermann(urdf_path)` | 4 wheels | Ackermann front | RWD | Independent |
| `car_4w_awd_ackermann(urdf_path)` | 4 wheels | Ackermann front | AWD | Independent |
| `truck_6w_partial_ackermann(urdf_path)` | 6 wheels | Ackermann on axle 0 | Drive on mid + rear axles | Independent |
| `tank_10w_skid_belt(urdf_path)` | 10 wheels | SkidSteer | PerSide (gear cap 0.3) | SameSideBelt |

Tune by editing the returned config (`cfg.dt = ...`, replace a strategy,
override `cfg.stability_hooks`) before passing it to `VehiclePhysics`.

## 9. Common patterns

### 9.1 Custom 4-wheel car (no preset)

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

### 9.2 RL rollout (batched, partial reset)

```python
N_ENVS = 256
physics = VehiclePhysics(scene, car, sensor, cfg, n_envs=N_ENVS)
throttle = torch.zeros(N_ENVS, device='cuda')
brake    = torch.zeros(N_ENVS, device='cuda')
steer    = torch.zeros(N_ENVS, device='cuda')

for t in range(T):
    physics.step(VehicleInputs(throttle, brake, steer))
    scene.step()
    if t % 100 == 0:
        done = some_termination_mask              # (N_ENVS,) bool
        physics.reset(env_ids=done)
```

### 9.3 Custom strategy

```python
import torch
from genesis_vehicle import AWD

class TorqueVectoringAWD(AWD):
    def distribute_torque(self, inputs, omega, wheel_meta, device, dtype):
        T_drive, T_brake = super().distribute_torque(
            inputs, omega, wheel_meta, device, dtype)
        # Per-wheel bias from yaw rate, etc.
        return T_drive, T_brake
```

Same pattern works for `SteeringStrategy.per_wheel_steer`,
`CouplingStrategy.apply`, `TireModel.__call__`, and stability hooks.

## 10. Module-level defaults (from `config.py`)

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
| `DEFAULT_I_WHEEL` | 1.5 kg·m² | wheel spin MOI fallback (last-resort) |
| `DEFAULT_PACEJKA` | `PB_X=10, PC_X=1.65, PE_X=0.4, PB_Y=8, PC_Y=1.30, PE_Y=0.4` | mid-grip car defaults |

These are last-resort fallbacks. URDF + `WheelConfig` overrides take
precedence per [`concepts.md`](concepts.md#4-urdf-is-the-default-source-explicit-api-overrides-are-the-final-truth).
