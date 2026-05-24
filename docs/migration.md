# Migration

Mapping legacy 3-variant style code onto the SDK.

`HJW`, `JMK`, and `KDU` below refer to three independent ray-wheel +
Pacejka vehicle implementations that pre-date this SDK (4-wheel car ×2 and
10-wheel skid-steer tank). They are NOT part of this repository; they are
illustrative examples of the patterns the SDK consolidates. If your codebase
follows a similar layout — module-level physics constants, an inlined
per-step function, vehicle-class-specific physics classes — the symbol
mappings below transfer directly.

Migrate by writing new entry-point scripts that import the SDK and re-use
your URDF + assets.

## From HJW (`HJW/car_raywheel.py`)

| HJW symbol | SDK |
|---|---|
| `CarRayWheelPhysics(scene, car, sensor)` | `VehiclePhysics(scene, car, sensor, cfg)` where `cfg = car_4w_rwd_ackermann(URDF_PATH)` |
| `physics.step(throttle, brake, steer)` | `physics.step(VehicleInputs(throttle, brake, steer))` |
| Module-level constants (`K_SUSP`, `C_COMP`, `MU`, `PB_X`, ...) | `WheelConfig` fields, set via `wheel_overrides` |
| `WHEEL_POSITIONS`, `N_WHEELS`, `BRAKE_BIAS_PER_WHEEL` | URDF + `RWD(brake_bias=[...])` |
| Steer sign flip at `car_raywheel.py:298` | hidden inside `Ackermann` (no manual flip) |

A worked-out migration of an HJW-style 4-phase scenario test follows the
pattern below — preset selection, `VehicleInputs` per-step, telemetry
print — and is the recommended template for porting your own code.

## From JMK (`JMK/real2sim/gt_ray/0_test_ray_step{11,12}.py`)

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
URDF (1200, 1.32) provides the defaults; only an explicit user override
would restore them. (Per the precedence in
[`concepts.md`](concepts.md#4-urdf-is-the-default-source-explicit-api-overrides-are-the-final-truth),
the SDK never invents these values.)

## From KDU (`KDU/physics.py`)

| KDU symbol | SDK |
|---|---|
| `parse_tank_urdf()` | `parse_urdf()` (generalised; same naming convention) |
| `TankRayPhysics(tank, sensor, urdf_path)` | `VehiclePhysics(scene, tank, sensor, cfg)` where `cfg = tank_10w_skid_belt(urdf_path)` |
| `physics.step(throttle, brake, steer)` with `+steer = LEFT` | `physics.step(VehicleInputs(throttle, brake, steer=-old_steer))` — **sign flip** to ISO 8855 |
| `omega[LEFT_IDX] = mean()` belt constraint | `SameSideBelt()` coupling (post-loop, pre-force-apply — see [`physics-contracts.md`](physics-contracts.md#75-coupling-order)) |
| `_cap_torque` per-side taper | `PerSide(use_per_side_taper=True)` |
| Static friction lock | `StaticFrictionLock(brake_thr=0.3, v_thr=0.5)` (already part of `"control"` profile for tanks; v0.5.7 position-anchored stick-slip — no `hold_k` proportional gain anymore) |
| `THROTTLE_GEAR_CAP = 0.3` | `PerSide(throttle_gear_cap=0.3)` |
| Turret / barrel DoFs | NOT in SDK — keep in user demo code |

## Sanity-check pattern (for every migration)

Run the original variant test and the SDK-based equivalent side by side
under the same scenario. The final state numbers (x, y, yaw, omega per
wheel) should be within tight tolerance for the same throttle/steer
sequence. Acceptable drift sources:

- Visual-layer numerical noise (different `set_dofs_position` timing)
- `LowSpeedRegularizer` is now active under throttle (was a no-op in the
  legacy variants) → smoother low-speed behavior, slight forward-distance
  differences during pull-away
- `StaticFrictionLock` may engage on KDU stops that previously oscillated
  → cleaner brake-to-rest

For exact-match comparisons, use `stability="raw"` on the SDK side to
disable all hooks.
