# genesis_vehicle — API Reference

Ray-cast wheel + Pacejka-tire vehicle SDK on top of Genesis. One package, one
import path, batched (`n_envs >= 1`) by default.

> **Version:** see `genesis_vehicle.__version__`. Release history in
> [`CHANGELOG.md`](CHANGELOG.md). `VehiclePhysics.__init__` prints a one-line
> banner the first time it is constructed in a process:
>
> ```
> [genesis_vehicle v0.4.0] Initialized: 4 wheels, Ackermann, RWD, Independent, n_envs=1, hooks=[RollingResistance, LowSpeedRegularizer]
> ```

---

## 1. Quick start

```python
import genesis as gs
from genesis_vehicle import (
    VehiclePhysics, VehicleInputs, WheelRayPattern,
    car_4w_rwd_ackermann, parse_urdf,
)

gs.init(backend=gs.gpu)
scene = gs.Scene(...)
scene.add_entity(gs.morphs.Plane())
car = scene.add_entity(gs.morphs.URDF(file=URDF_PATH, pos=(0, 0, 1.5)))

# Wheel raycaster — positions come from URDF.
parsed = parse_urdf(URDF_PATH)
sensor = scene.add_sensor(gs.sensors.Raycaster(
    pattern=WheelRayPattern([w.position for w in parsed.wheels]),
    entity_idx=car.idx, max_range=20.0, return_world_frame=True,
))
scene.build(n_envs=1)

cfg = car_4w_rwd_ackermann(URDF_PATH)
physics = VehiclePhysics(scene, car, sensor, cfg, n_envs=1)
for step in range(480):
    physics.step(VehicleInputs(throttle=0.5, brake=0.0, steer=0.0))
    scene.step()
```

---

## 2. Concepts

### 2.1 The 5-step pipeline

Every step, for every wheel:

1. **Raycast** — distance `d` from chassis-local wheel point straight down.
2. **Suspension** — compression `c = max(rest_d - d, 0)`, then `N` from a per-wheel asymmetric damper (see §7.2 for the exact contract).
3. **Slip** — `kappa = (v_roll - v_long) / max(|v_long|, eps_v)`, `alpha = atan2(v_lat, |v_long|)`.
4. **Tire force** — `(F_long, F_lat) = TireModel(v_long, v_lat, v_roll, N, wheel_params)`. Pacejka does friction-ellipse clamping internally.
5. **omega + chassis force-at-point** — `domega = (T_drive - T_brake_eff - R*F_long) / I_wheel`; chassis gets `N*up + F_long*fwd + F_lat*lat` and `r x F` torque. `T_brake_eff` is the signed form (§7.1).

`VehiclePhysics.step()` runs this loop with strategy hooks plugged at fixed seams (§5).

### 2.2 Coordinate / sign conventions (ISO 8855)

| Axis | Direction |
|---|---|
| `+X` | forward |
| `+Y` | left |
| `+Z` | up |

| Input | Range | Meaning |
|---|---|---|
| `throttle` | `[-1, +1]` | signed; `+` accelerates forward, `-` reverses |
| `brake` | `[0, 1]` | positive magnitude — internally converted to a signed torque opposing wheel rotation (§7.1) |
| `steer` | `[-1, +1]` | **`+steer` = right turn** |

Internal sign flips (Genesis RHS, URDF `<axis 0 0 -1>`, KDU's legacy `+steer = LEFT`) are absorbed inside the strategy / visual layer. The user-facing API is unambiguous.

### 2.3 Strategies (composition over inheritance)

A `VehicleConfig` is built from four orthogonal strategies plus optional stability hooks:

| Axis | Concrete options |
|---|---|
| `SteeringStrategy` | `Ackermann`, `PartialAckermann`, `SkidSteer`, `NoSteer` |
| `DrivetrainStrategy` | `FWD`, `RWD`, `AWD`, `PerSide` |
| `CouplingStrategy` | `Independent`, `SameSideBelt` |
| `TireModel` | `PacejkaAnisotropic`, `CoulombIsotropic` |
| `StabilityHook` (opt-in list, §6) | `RollingResistance`, `LowSpeedRegularizer`, `StaticFrictionLock` |

Each is a small class you can subclass to add a new behaviour.

### 2.4 URDF is the default source; explicit API overrides are the final truth

For every URDF-derivable field (positions, radii, masses, inertia, joint names),
the precedence is:

```
module default  <  URDF-derived default  <  explicit API override
```

In words: URDF provides defaults; `WheelConfig` / `VehicleConfig` values that
the user explicitly set always win. After `resolve()`, the `ResolvedConfig` is
the only source of truth used by `VehiclePhysics`.

`i_wheel` follows the same precedence with one addition: if both the user and
the URDF leave it unset, the SDK consults Genesis runtime metadata as a
last-resort estimate (§3.4, §7.3).

### 2.5 What is a "stability hook"?

A **hook** is a post-processing callback that runs at a fixed point in the
5-step pipeline. The main dynamics (raycast → suspension → slip → tire → ω)
are not modified; the hook just *adds one more term* at a designated seam,
the same way `git pre-commit`, a web-framework middleware, or a Java
interceptor works. Examples in this SDK:

- `RollingResistance` runs after the tire model produces `F_long`, and
  subtracts `cr * N * tanh(v_long / scale)`.
- `LowSpeedRegularizer` runs once pre-loop (computes a `moving∈[0,1]`
  factor) and once per wheel post-tire (scales `F_long`/`F_lat` by `moving`
  and pulls `omega` toward rolling-without-slip).
- `StaticFrictionLock` runs post-tire when `brake > thr` and
  `|v_long| < thr`; overrides `F_long` with `-K * v_long` and forces
  `omega = 0`.

In v0.3.0 the SDK selects hooks via a **stability profile** rather than a
free-form list (§6). Advanced users can still pass `stability="research"`
and assemble `cfg.stability_hooks` manually.

### 2.6 Batched by default

Every state tensor is `(n_envs, n_wheels)` or `(n_envs, 3/4)`. Single-env
(`n_envs=1`) is a special case. Scalar OR `(n_envs,)` tensor inputs are both
accepted. Use `physics.reset(env_ids=...)` for partial reset (RL / MPPI).

---

## 3. Public API reference

Imported as `from genesis_vehicle import <name>` unless otherwise noted.

### 3.1 `VehiclePhysics` — the driver

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
- `resolved: ResolvedConfig` — post-merge config; only source of truth at runtime
- `wheel_meta: WheelMeta` — cached per-wheel tensors (positions, radius, side masks, axle indices, etc.)

`step()` accepts either the strategy's `InputType` (e.g. `AckermannInputs`) or
a unified `VehicleInputs`. The latter is auto-converted via `from_unified()`.

### 3.2 Inputs

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

Every typed input has a `from_unified(VehicleInputs) -> Self` classmethod for
RL pipelines that prefer the unified `VehicleInputs` schema.

### 3.3 Config

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

> **Implementation note.** Mutable defaults on dataclass fields (`chassis: ChassisConfig = ChassisConfig()` or `list = []`) are unsafe — they are shared across instances. The SDK uses `field(default_factory=...)` throughout. When constructing your own dataclasses, follow the same pattern.

`from_urdf()` is the easiest path: pass strategies + a dict of per-wheel
overrides keyed by URDF wheel link name; the wheel list is auto-populated.

`resolve(config) -> ResolvedConfig` runs the URDF → user → default merge and
each strategy's `validate()`. Called automatically by `VehiclePhysics.__init__`.

`ConfigError` — raised on bad config (missing required fields, wheel count
mismatch, sides missing for skid-steer, etc.).

### 3.4 URDF utilities

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
- `steer_axis_signs: dict[str, int]` — used by visual layer for `<axis 0 0 -1>` flip
- `susp_has_dynamics: dict[str, bool]` — picks `set_dofs_position` vs `control_dofs_position`

`estimate_spin_inertia_from_genesis()` is a **fallback estimate** consulted
only when `WheelConfig.i_wheel` is not set (by the user OR by the URDF). When
`spin_axis_local` is provided, the helper projects the inertia tensor onto
that axis (`a^T diag(I) a`); otherwise it returns `max(diag(inertial_i))`,
which is the spin MOI for cylindrical wheels but a heuristic for general
shapes. **`WheelConfig.i_wheel` (when supplied) is always authoritative.**

The old name `parse_inertia_max_principal_genesis` is kept as a deprecated
alias and will be removed in a future revision.

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
    InputType = PartialAckermannInputs

class SkidSteer(SteeringStrategy):
    # No steer angle; turning happens via PerSide drivetrain.
    InputType = SkidSteerInputs

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
    """For tank-style skid steer.
    Unified-input mapping (ISO +steer = right turn):
        left_cmd  = throttle + steer_gain * steer
        right_cmd = throttle - steer_gain * steer
    So +steer makes the LEFT side faster -> right turn."""
    def __init__(t_drive_max, t_brake_max,
                 steer_gain=1.0, omega_max_drive=100.0,
                 throttle_gear_cap=1.0, use_per_side_taper=True)
```

#### Coupling

```python
class Independent(CouplingStrategy):  # passthrough; the default
class SameSideBelt(CouplingStrategy):
    # Tank track: omega[L] := mean(omega[L]) per env; same for R.
```

**`CouplingStrategy.apply()` contract.** Coupling fires once per step **after**
the per-wheel omega integration in the wheel loop and **before** visual sync.
The chassis force/torque applied in the same step still uses the pre-coupling
omegas (since they were computed during the wheel loop); the coupled omega is
the state observed by the next simulation step. This matches the KDU reference
behaviour and is documented as the SDK contract — strategies must not assume
they run inside the per-wheel loop.

#### Stability hooks (selected via `stability=` profile; §6)

Hook ordering matters — put `RollingResistance` before `StaticFrictionLock`
so the lock has the last word on `F_long`. The `stability_hooks_for_profile`
helper enforces a sensible order.

```python
class RollingResistance(StabilityHook):
    def __init__(tanh_scale=0.5)
    # F_long -= cr * N * tanh(v_long / tanh_scale)
    # cr is per-wheel via WheelConfig.rolling_resistance_cr.

class LowSpeedRegularizer(StabilityHook):
    def __init__(v_kin_com=0.5, ang_kin=0.5,
                 disable_when_control_active=True, control_threshold=0.01)
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
    # Per-wheel (B, C, E) for long + lat, mu_long, mu_lat come from wheel_params.
    # Returns (F_long, F_lat, kappa, alpha) with friction-ellipse clamp.
class CoulombIsotropic(TireModel):
    def __init__(eps_v=0.5)
    # F = -mu * N * v_slip / |v_slip|, isotropic.
```

### 3.7 Sensor / dynamics utilities

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
    # See §7.1.

suspension_normal_force(
    compression, comp_rate,
    k_susp, c_compression, c_extension, air_mask,
) -> torch.Tensor
    # See §7.2.

# --- Version API ---
genesis_vehicle.__version__               # str, e.g. "0.3.0"
genesis_vehicle.VERSION_INFO              # tuple, e.g. (0, 3, 0)
genesis_vehicle.version() -> str          # same as __version__
genesis_vehicle.version_info() -> tuple   # same as VERSION_INFO
# VehiclePhysics.__init__ prints a banner with the version + a config summary
# on first construction per process. There is no global "quiet" flag yet.
```

### 3.8 Presets

Every preset takes a keyword-only `stability` argument that picks the
stability profile (see §6).

```python
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
| `car_4w_rwd_ackermann(urdf_path)` | 4 wheels | Ackermann front | RWD | Independent |
| `car_4w_awd_ackermann(urdf_path)` | 4 wheels | Ackermann front | AWD | Independent |
| `truck_6w_partial_ackermann(urdf_path)` | 6 wheels | Ackermann on axle 0 | AWD (uniform) | Independent |
| `tank_10w_skid_belt(urdf_path)` | 10 wheels | SkidSteer | PerSide (gear cap 0.3) | SameSideBelt |

Tune by editing the returned config (`cfg.dt = ...`, replace a strategy,
override `cfg.stability_hooks`) before passing it to `VehiclePhysics`.

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
brake    = torch.zeros(N_ENVS, device='cuda')
steer    = torch.zeros(N_ENVS, device='cuda')

for t in range(T):
    physics.step(VehicleInputs(throttle, brake, steer))
    scene.step()
    if t % 100 == 0:
        done = some_termination_mask              # (N_ENVS,) bool
        physics.reset(env_ids=done)
```

### 4.3 Custom strategy

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

### 4.4 Stability hook ordering

```python
cfg.stability_hooks = [RollingResistance(), StaticFrictionLock()]
```

Hooks run in list order. If both are active, place `RollingResistance` first
so the lock overrides any rolling-resistance contribution at rest.

---

## 5. Pipeline hook insertion points

```
VehiclePhysics.step(inputs)

[0] Input adaptation
    typed = inputs if isinstance(inputs, steering.InputType)
            else steering.InputType.from_unified(inputs)

[1] PRE-LOOP
    steer_per_wheel = SteeringStrategy.per_wheel_steer(typed)
    T_drive_pw, T_brake_pw = DrivetrainStrategy.distribute_torque(typed, omega, meta)
    for hook in stability_hooks if PRE_LOOP in hook.slots:
        hook.apply_pre_loop(ctx)                     # e.g. LowSpeedRegularizer

[2] Raycast (read_distances) + first-step protection

[3] Chassis state read

[per-wheel loop i = 0 .. n_wheels-1, all batched over n_envs]
    (A) compression, comp_rate, asymmetric damper -> N      (§7.2)
    (B) wheel-frame fwd/lat using steer_per_wheel[:, i]
    (C) F_long, F_lat = TireModel(...)
        for hook in stability_hooks if POST_TIRE in hook.slots:
            hook.apply_post_tire(ctx, i)             # RollingResistance, ...
    (D) T_brake_eff = brake_torque_signed(T_brake_pw[:, i], omega[:, i])  (§7.1)
        domega = (T_drive - T_brake_eff - R * F_long) / I_wheel
        omega[:, i] = clamp(omega[:, i] + DT*domega, ±OMEGA_MAX)
    (E) accumulate F_world, torque into total_F, total_T

[4] CouplingStrategy.apply(omega, meta)              # post-loop, pre-force-apply
[5] solver.apply_links_external_force/torque (chassis)
[6] VisualSync (if enabled)
```

---

## 6. Stability profiles (`stability="control" | "raw" | "research"`)

Low-speed stabilization of a ray-wheel + Pacejka model is a fundamental
numerical concern, not an optional add-on. Every preset and the
`stability_hooks_for_profile` helper select a **profile** that materializes
the appropriate set of hooks. This mirrors the convention in other vehicle
SDKs (PhysX `eSTICKY_TIRE_FRICTION` mode, Jolt anti-rollback, Chaos low-speed
threshold, Unity WheelCollider stick-slip): the low-speed stabilization is
*part of the model*, not a plug-in the user has to remember to enable.

| Profile | Hooks materialized (car) | Hooks materialized (tank) | Use case |
|---|---|---|---|
| `"control"` (default) | `RollingResistance`, `LowSpeedRegularizer(disable_when_control_active=False)` | + `StaticFrictionLock(0.3, 0.5, 200_000)` | **Recommended for almost everyone**: RL / MPPI / general control / Real2Sim. The regularizer is active even under throttle (the v0.3.0 footgun fix). |
| `"raw"` | _empty_ | _empty_ | Raw ray-wheel + Pacejka with NO numerical stabilization. For debugging the bare dynamics or running hook-ablation studies. Not a general-purpose default. |
| `"research"` | _empty_ | _empty_ | Caller assembles `cfg.stability_hooks` manually (advanced). |

```python
# RL / MPPI / Real2Sim — default profile is correct.
cfg = car_4w_rwd_ackermann(URDF)

# Raw dynamics — debugging or ablation study.
cfg = car_4w_rwd_ackermann(URDF, stability="raw")

# Custom hook stack (advanced).
cfg = car_4w_rwd_ackermann(URDF, stability="research")
cfg.stability_hooks = [RollingResistance(tanh_scale=0.3),
                       StaticFrictionLock(brake_thr=0.2)]
```

### The one rule (covers MPPI, RL, and Real2Sim)

**Fitting forward model = deployment forward model.** Hooks are part of the
model. Whatever stability profile you deploy with, fit with the same. For our
SDK that almost always means `"control"` in both fitting and deployment.

- **MPPI / RL** — train against `"control"`, deploy against `"control"`.
- **Real2Sim** — fit (μ, K, B, …) against `"control"` so the fitted values
  reflect "how this vehicle behaves under the model you'll actually use".
  Fitted parameters are model-coupled: Pacejka `B` is not "rubber's true
  stiffness" but "the coefficient that, inside Pacejka inside our SDK with
  these hooks, best matches real data". That coupling is fine — desired,
  even — as long as deployment uses the same model.

### When `"raw"` is actually useful

Not for general Real2Sim. Use `"raw"` only when:

1. You want to study the bare ray-wheel + Pacejka dynamics (e.g. debug a
   stiff-ODE oscillation).
2. You want to measure the contribution of the hooks themselves (run the
   same scenario in `"control"` and `"raw"` and diff).
3. You are exporting parameter values to a different simulator that
   guarantees no equivalent hooks — but note: parameter values are not
   portable in any rigorous sense; refit in the destination simulator.

> 핵심: hook 도 forward model 의 일부. fitting 과 deployment 가 같은 hook config 라면 그게 일관된 모델. 우리 SDK 로 fitting + 우리 SDK 로 배포면 `"control"` 그대로 쓰면 됨. `"raw"` 는 hook 의 기여도를 분리해서 보고 싶을 때만 (debugging / ablation).

---

## 7. Physics contracts

These contracts are enforced by `genesis_vehicle.dynamics` (pure-Python,
unit-tested in `tests/test_dynamics.py`) and consumed by `core.py`. They are
the SDK's promises about ambiguous physical conventions.

### 7.1 Brake torque is a positive command magnitude

`brake` (user input) is always in `[0, 1]`. Internally, the SDK converts it
to a signed torque opposing wheel rotation:

```
T_brake_eff = T_brake * tanh(omega / smoothing_scale)
domega     = (T_drive - T_brake_eff - R * F_long) / I_wheel
```

- For `omega > 0`, `T_brake_eff > 0`, so `-T_brake_eff` decelerates the wheel.
- For `omega < 0` (reverse spin), `T_brake_eff < 0`, so `-T_brake_eff > 0`
  again decelerates.
- For `omega ≈ 0`, `T_brake_eff ≈ 0` — the smooth brake cannot pin the wheel.
  Pair with `StaticFrictionLock` for hard hold-at-rest behaviour.

### 7.2 Normal force is non-negative; air-mask wheels contribute nothing

Per-wheel suspension force uses the asymmetric damper (different coefficient
on compression vs extension) and is clamped non-negative — the ground cannot
pull a wheel down:

```
c_damp = c_compression if c_dot > 0 else c_extension
N_raw  = K_susp * compression + c_damp * c_dot
N      = max(N_raw, 0)
N      = 0 if the ray missed the ground (air_mask)
```

When `N = 0`, the per-wheel `F_long`, `F_lat` are also zero (no contact),
though `T_drive` and `T_brake` still update `omega` (the wheel spins freely
in the air).

### 7.3 `WheelConfig.i_wheel` truth policy

```
1. WheelConfig.i_wheel set by the user            -> AUTHORITATIVE (used as-is)
2. URDF inertia (via parse_urdf, max diagonal)    -> default / estimate
3. Genesis-runtime metadata                       -> fallback estimate
4. DEFAULT_I_WHEEL                                -> last-resort fallback
```

For ray-wheel dynamics, the wheel spin inertia is often different from the
URDF hinge inertia (e.g. URDF wheel hinge is visual-only while real ray-wheel
inertia comes from a coarser estimate). In Real2Sim / parameter fitting,
**always set `WheelConfig.i_wheel` explicitly** to take this out of the
estimation pipeline.

### 7.4 Steering sign convention (ISO 8855)

`+steer` is right turn under all strategies (`Ackermann`, `PartialAckermann`,
`SkidSteer`). Unit-tested:

- **Ackermann right turn**: both front wheels turn positive; the right wheel
  (the inner wheel for a right turn) has the larger angle.
- **SkidSteer right turn**: the left side commands more torque than the right
  side (`left_cmd > right_cmd` via `left_cmd = throttle + steer_gain * steer`).

KDU's legacy `+steer = LEFT` is **not** carried forward; KDU demo callsites
must flip the sign in migration.

### 7.5 Coupling order

`CouplingStrategy.apply(omega)` runs after the per-wheel omega integration in
the current step and before the next step. Drive torque distribution in the
same step uses **pre-coupling** omegas (one-step lag), matching the KDU
reference implementation.

---

## 8. Migration cheatsheet

### From HJW (`HJW/car_raywheel.py`)

| HJW symbol | SDK |
|---|---|
| `CarRayWheelPhysics(scene, car, sensor)` | `VehiclePhysics(scene, car, sensor, cfg)` where `cfg = car_4w_rwd_ackermann(URDF_PATH)` |
| `physics.step(throttle, brake, steer)` | `physics.step(VehicleInputs(throttle, brake, steer))` |
| Module-level constants (`K_SUSP`, `C_COMP`, `MU`, `PB_X`, ...) | `WheelConfig` fields, set via `wheel_overrides` |
| `WHEEL_POSITIONS`, `N_WHEELS`, `BRAKE_BIAS_PER_WHEEL` | URDF + `RWD(brake_bias=[...])` |
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

JMK's hard-coded chassis mass (1330) and track width (1.48) **are dropped**:
URDF (1200, 1.32) provides the defaults; only an explicit user override would
restore them. (Per the precedence in §2.4, the SDK never invents these values.)

### From KDU (`KDU/physics.py`)

| KDU symbol | SDK |
|---|---|
| `parse_tank_urdf()` | `parse_urdf()` (generalised; same naming convention) |
| `TankRayPhysics(tank, sensor, urdf_path)` | `VehiclePhysics(scene, tank, sensor, cfg)` where `cfg = tank_10w_skid_belt(urdf_path)` |
| `physics.step(throttle, brake, steer)` with `+steer = LEFT` | `physics.step(VehicleInputs(throttle, brake, steer=-old_steer))` — **sign flip** to ISO 8855 |
| `omega[LEFT_IDX] = mean()` belt constraint | `SameSideBelt()` coupling (post-loop, pre-force-apply — see §7.5) |
| `_cap_torque` per-side taper | `PerSide(use_per_side_taper=True)` |
| Static friction lock | `StaticFrictionLock(brake_thr=0.3, v_thr=0.5, hold_k=200_000.0)` |
| `THROTTLE_GEAR_CAP = 0.3` | `PerSide(throttle_gear_cap=0.3)` |
| Turret / barrel DoFs | NOT in SDK — keep in user demo code |

---

## 9. Recommended unit tests (kept in `genesis_vehicle/tests/`)

Pure-Python, runnable without Genesis (`pytest genesis_vehicle/tests/`).

| Coverage area | Test file | Notes |
|---|---|---|
| Config resolve (URDF default, user override, defaults fill) | `test_config_resolve.py` | Includes `test_user_explicit_i_wheel_wins_over_urdf` |
| URDF parsing (HJW + KDU naming) | `test_urdf_parse.py` | |
| Ackermann sign + inner/outer | `test_strategies_unit.py` | `+steer` → both wheels positive, FR > FL |
| SkidSteer sign (left faster on +steer) | `test_strategies_unit.py` | `test_perside_iso_right_turn_left_faster` |
| SameSideBelt averages each side | `test_strategies_unit.py` | |
| RWD front-drive-zero invariant | `test_strategies_unit.py` | |
| `brake_torque_signed` reverses with omega | `test_dynamics.py` | `omega < 0` → `T_brake_eff < 0` |
| Suspension N clamped non-negative | `test_dynamics.py` | Strong rebound → `N = 0`, not negative |
| Suspension air-mask → `N = 0` | `test_dynamics.py` | |
| Asymmetric damper (compression vs extension) | `test_dynamics.py` | |

---

## 10. Module map

| File | Purpose |
|---|---|
| `core.py` | `VehiclePhysics` — 5-step pipeline orchestrator |
| `config.py` | `WheelConfig`, `ChassisConfig`, `VehicleConfig`, `ResolvedConfig`, `resolve()`, `ConfigError`, `DEFAULT_*` |
| `inputs.py` | `VehicleInputs`, `VehicleStepInputs`, typed inputs |
| `urdf.py` | `parse_urdf()`, `URDFParsedConfig`, `estimate_spin_inertia_from_genesis` |
| `dynamics.py` | `brake_torque_signed`, `suspension_normal_force` — pure helpers |
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

## 11. Defaults reference (from `config.py`)

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
| `DEFAULT_I_WHEEL` | 1.5 kg·m² | wheel spin MOI fallback (last-resort; see §7.3) |
| `DEFAULT_PACEJKA` | `PB_X=10, PC_X=1.65, PE_X=0.4, PB_Y=8, PC_Y=1.30, PE_Y=0.4` | mid-grip car defaults |

These are last-resort fallbacks. URDF + `WheelConfig` overrides take precedence
per §2.4.
