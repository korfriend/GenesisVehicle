# Physics contracts

The SDK's promises about ambiguous physical conventions. These contracts are
enforced by `genesis_vehicle.dynamics` (pure-Python, unit-tested in
`tests/test_dynamics.py`) and consumed by `core.py`.

## 7.1 Brake torque is a positive command magnitude

`brake` (user input) is always in `[0, 1]`. Internally, the SDK converts it
to a signed torque opposing wheel rotation:

```
T_brake_eff = T_brake * tanh(omega / smoothing_scale)
domega      = (T_drive - T_brake_eff - R * F_long) / I_wheel
```

- For `omega > 0`, `T_brake_eff > 0`, so `-T_brake_eff` decelerates the wheel.
- For `omega < 0` (reverse spin), `T_brake_eff < 0`, so `-T_brake_eff > 0`
  again decelerates.
- For `omega ≈ 0`, `T_brake_eff ≈ 0` — the smooth brake cannot pin the
  wheel. Pair with `StaticFrictionLock` for hard hold-at-rest behaviour.

Implementation: `genesis_vehicle.dynamics.brake_torque_signed`.

## 7.2 Normal force is non-negative; air-mask wheels contribute nothing

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

Implementation: `genesis_vehicle.dynamics.suspension_normal_force`.

## 7.3 `WheelConfig.i_wheel` truth policy

```
1. WheelConfig.i_wheel set by the user            -> AUTHORITATIVE (used as-is)
2. URDF inertia (via parse_urdf, max diagonal)    -> default / estimate
3. Genesis-runtime metadata                       -> fallback estimate
4. DEFAULT_I_WHEEL                                -> last-resort fallback
```

For ray-wheel dynamics, the wheel spin inertia is often different from the
URDF hinge inertia (e.g. URDF wheel hinge is visual-only while real
ray-wheel inertia comes from a coarser estimate). In Real2Sim / parameter
fitting, **always set `WheelConfig.i_wheel` explicitly** to take this out of
the estimation pipeline.

## 7.4 Steering sign convention (ISO 8855)

`+steer` is right turn under all strategies (`Ackermann`,
`PartialAckermann`, `SkidSteer`). Unit-tested:

- **Ackermann right turn**: both front wheels turn positive; the right wheel
  (the inner wheel for a right turn) has the larger angle.
- **SkidSteer right turn**: the left side commands more torque than the
  right side (`left_cmd > right_cmd` via
  `left_cmd = throttle + steer_gain * steer`).

KDU's legacy `+steer = LEFT` is **not** carried forward; skid-steer here is
`+steer = right` (ISO 8855), so any code ported from a `+steer = LEFT`
convention must flip the steer sign.

### URDF authoring recommendation — steer joint axis = `(0, 0, -1)`

For new URDFs, declare steer joint axes as `(0, 0, -1)` so the joint angle
follows the same sign convention as the user-facing `+steer = right`:

```xml
<joint name="front_left_steer_joint" type="revolute">
  ...
  <axis xyz="0 0 -1"/>   <!-- ISO 8855: +angle = CW from above = RIGHT turn -->
  ...
</joint>
```

Vehicle frame z is up, so +joint_angle around `(0, 0, -1)` is CW viewed
from above = right turn — matches `+steer`. Using `(0, 0, 1)` instead makes
+joint_angle = CCW = left turn, which is **opposite-handed** to the SDK's
user-facing convention.

`WheelJointInternalSync` does compensate either way (`visual_cmd = -phys * sign`), so
existing URDFs with `(0, 0, 1)` still render correctly. The recommendation
is only for NEW URDFs: declaring `(0, 0, -1)` keeps URDF joint values and
user-facing steer values in the same sign domain, which makes URDF-side
inspection / debugging less surprising.

Examples in this repo:
- `HJW/urdf/car_raywheel.urdf` — `(0, 0, -1)` ✓ (matches recommendation)
- `GeneVehicle_Truck6w/urdf/truck_6w.urdf` — `(0, 0, -1)` ✓ (fixed in v0.5.4)
- `JMK/URDF/test_v1_raywheel.urdf` — `(0, 0, 1)` (external author; SDK handles via WheelJointInternalSync sign flip)

## 7.5 Coupling order

`CouplingStrategy.apply(omega)` runs after the per-wheel omega integration
in the current step and before the next step. Drive torque distribution in
the same step uses **pre-coupling** omegas (one-step lag), matching the KDU
reference implementation. Strategies must not assume they run inside the
per-wheel loop.

## 7.6 First-step protection

The wheel raycaster is not populated until the first `scene.step()`. To
avoid a NaN cascade, `VehiclePhysics.step()` skips force application on the
first call when all distances are zero, sets `_prev_init = True`, and runs
normally from the second step onward.

## 7.7 Longitudinal friction-force overshoot clamp (v0.6.0)

The tire-friction analogue of §7.1. Explicit-Euler integration of the
slip-dependent friction torque `T_fric = R·F_long` is stiff near rolling
(its relaxation rate `R²·C_kappa/(I·|v_long|) → ∞` as `v_long → 0`); below
the stability limit the wheel oscillates across the rolling point
(forward force → reverse slip → backward force → …), seen as wheel
"trembling" and a stuck `kappa ≈ −1` drag on undriven wheels at launch.

`VehiclePhysics.step()` caps `F_long` so the friction torque cannot carry
the wheel **past the rolling speed** `omega_target = v_long/R` in one step:

```
omega_nofric = omega + DT·(T_drive - T_brake_eff)/I_wheel
F_long_limit = (omega_nofric - omega_target)·I_wheel / (DT·R)
omega_nofric > omega_target → F_long ∈ [0, F_long_limit]
omega_nofric < omega_target → F_long ∈ [F_long_limit, 0]
```

The clamp binds **only near rolling** (small `omega_nofric − omega_target`),
so it removes the oscillation while leaving the high-slip saturated regime —
driven-wheel launch slip — untouched (`quickstart` launch is preserved). The
**clamped** `F_long` is what is applied to both the wheel-ω update and the
chassis force, so a custom `TireModel` or parameter-fit sees a force that may
be reduced from its raw output near rolling. Implementation: inline in
`core.py` step (D); cf. `brake_torque_signed` (§7.1).

## 7.8 High-cast rays and over-compression (v1.1.16)

Wheel rays start `RAY_UP_OFFSET` (default 1.0 m) ABOVE the wheel
attachment point and the read layer (`read_distances`) subtracts the
offset from hits, so every consumer still sees attachment-relative
distances. Contract points:

- A ray MISS keeps its sentinel value (>= 19.9); the offset is NOT
  subtracted from misses.
- A hit may therefore report a **negative** distance: the ground is above
  the attachment point (the chassis has sunk past its wheels). That is a
  VALID reading — compression maxes out and `N` pushes the vehicle back
  up. Do not "sanitize" negative distances back to air.
- Why: with origins AT the attachment points, a hard landing could sink
  the origins below the ground; the rays then missed, the air mask killed
  `N`, and the vehicle rested on its chassis collision box forever (a
  stable buried equilibrium — the v1.1.16 field report).
- Vehicles are never raycast targets (`use_visual_raycasting` defaults
  to False), so the elevated origin cannot self-hit in either raycast
  mode. Keep overhead raycast terrain (tunnel ceilings) more than
  `RAY_UP_OFFSET` above the wheel attachment points.

Related: `WheelJointInternalSync` is intended to be cosmetic but is not
perfectly physics-neutral (the control path's PD applies real joint
forces; the set path teleports wheel mass). Its suspension targets are
stroke-clamped and slew-rate-limited (`_SUSP_VIS_MAX_RATE`), bounding
the measured disturbance to < 1 cm extra compression on hard landings
(was 2-3 cm unclamped — enough to flip a marginal landing into the
buried state before the high-cast fix removed that failure mode).

Since v1.1.17 `VehicleScene` no longer uses VJS by default: rendered
scenes with `n_envs == 1` draw wheels via `InstancedWheelRenderer` —
closed-form poses streamed into instanced render nodes (Genesis's
external render-node channel — the debug-draw machinery — NOT the rigid
solver), verified physics-identical to headless to 1e-6 m at a slight
pose-streaming cost (~2–3 ms/step at 30 vehicles, CPU)
(`wheel_render_mode="internal_sync"` restores the old behavior; multi-env
and raw-`VehiclePhysics` use still fall back to VJS).


## 7.9 URDF contracts for ray-wheels (auto-corrected since v1.1.22)

Ray-wheel physics makes three demands on a vehicle URDF. Every
SDK-authored vehicle satisfies them; an arbitrary URDF (e.g. one exported
for a normal rigid-body sim) usually does not, so
`VehicleScene.add_vehicle` ALWAYS runs `genesis_vehicle.urdf_prep` on
it (not a knob — the prepared file feeds the entity, the parse AND the
ray pattern, which must agree), writing a corrected temp copy next to the
original — the original file is never modified, and a URDF that already
complies is used as-is (no copy).

| contract | why | auto-correction |
|---|---|---|
| **Wheels must not collide** | ground contact IS the raycast + suspension force model; a colliding wheel is a SECOND support that fights it (vehicle jitters in place, or rides on its colliders while the suspension pushes several times its weight) | wheel colliders are removed. A collider that is the wheel's ONLY geometry is first promoted to a `<visual>`, so the wheel still RENDERS — the instanced wheel renderer draws visual geoms, and physics never touches them |
| **The suspension attach point IS the wheel centre** | the ray is cast down from `WheelConfig.position` (= the prismatic joint origin) and `rest_d = radius + rest_stroke` measures from there | a chain that hangs the wheel off a carrier (`body --susp--> carrier --spin(z=+h)--> wheel`) puts the attach `h` below the wheel centre — the hull then settles `h` too high with the wheels visibly floating (measured on an M1A2 model: h = 0.433 m). The spin-joint offset is folded into the suspension origin, leaving every link's rest pose unchanged |
| **Moving links need a valid inertial** | links with no `<inertial>` have zero mass AND inertia; Genesis warns ("mass and inertia of moving bodies must be larger than mjMINVAL"), falls back to its legacy URDF parser, and the articulated chain goes degenerate — the hull stops responding to force properly | a small inertial is injected where one is missing |

Wheel RENDERING is independent of all of this: the instanced renderer
harvests the wheel link's visual geoms (falling back to its colliders, and
finally to a cylinder synthesized from `radius`) and draws them at the
closed-form ray-wheel pose. The wheel link's own solver pose is never used
— its visuals are hidden and its body only contributes mass.
