# Pipeline and hook insertion points

The 5-step pipeline with every strategy and stability-hook seam marked.

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
    (A) compression, comp_rate, asymmetric damper -> N     (physics-contracts.md S7.2)
    (B) wheel-frame fwd/lat using steer_per_wheel[:, i]
    (C) F_long, F_lat = TireModel(...)
        for hook in stability_hooks if POST_TIRE in hook.slots:
            hook.apply_post_tire(ctx, i)             # RollingResistance, ...
    (D) T_brake_eff = brake_torque_signed(T_brake_pw[:, i], omega[:, i])  (S7.1)
        # F_long overshoot clamp (v0.6.0, S7.7): cap so friction cannot carry
        # the wheel past rolling (omega_target = v_long/R) in one step.
        omega_nofric = omega[:, i] + DT*(T_drive - T_brake_eff)/I_wheel
        F_long_limit = (omega_nofric - v_long/R) * I_wheel / (DT * R)
        F_long = clamp_toward_rolling(F_long, F_long_limit)  # see physics-contracts.md S7.7
        domega = (T_drive - T_brake_eff - R * F_long) / I_wheel
        omega[:, i] = clamp(omega[:, i] + DT*domega, ±OMEGA_MAX)
    (E) accumulate F_world, torque into total_F, total_T

[4] CouplingStrategy.apply(omega, meta)              # post-loop, pre-force-apply
[5] solver.apply_links_external_force/torque (chassis)
[6] wheel visuals (rendered scenes only): VehicleScene streams closed-form
    wheel poses into instanced render nodes AFTER the physics advance — NOT
    via the rigid solver; with the native viewer the stream happens inside
    viewer.update()'s render-lock hold, atomic with the camera + node poses
    (v1.1.25; physics-contracts.md §7.8). WheelJointInternalSync, the legacy
    solver-joint fallback, runs here instead for n_envs > 1 /
    wheel_render_mode="internal_sync"; ext. renderers use
    visual_parts_transforms.
```

## Hook slots

A stability hook declares which slots it implements via the `slots` class
attribute (a tuple containing some of `"PRE_LOOP"`, `"POST_TIRE"`).

| Hook | Slots | What it does |
|---|---|---|
| `RollingResistance` | `("POST_TIRE",)` | Subtracts `cr * N * tanh(v_long / scale)` from `F_long` per wheel. |
| `LowSpeedRegularizer` | `("PRE_LOOP", "POST_TIRE")` | Pre-loop: compute `moving ∈ [0,1]` from chassis speed. Post-tire: scale `F_long`/`F_lat` by `moving`; record an omega pull target so core blends ω toward `v_long / radius`. |
| `StaticFrictionLock` | `("POST_TIRE",)` | When `brake > thr` and planar speed `sqrt(v_long² + v_lat²) < thr`: 2D stick-slip lock with per-wheel position anchor. Stores anchor at lock engagement; computes `F = -K_spring·displacement - K_damp·velocity` in both axes; projects onto the per-wheel friction ellipse (same form as `pacejka.py`); requests `omega = 0` via `ctx.omega_override`. v0.5.7 — replaced v0.5.6's tanh velocity damper with proper position-anchored stick-slip. Vehicle holds with **zero drift** on any slope up to `μ ≈ tan(slope_angle)` (truck preset with μ=1.0: rock-solid up to ~30°, vehicle rolls over physically beyond ~35°). |

Hook order is the list order (see
[`stability-profiles.md`](stability-profiles.md#hook-ordering-inside-a-profile)
for the recommended ordering).

## Coupling timing detail

`CouplingStrategy.apply()` runs at step [4], after the per-wheel loop
completes and after `omega` has been integrated for every wheel, but
**before** `solver.apply_links_external_force/torque` is called. The chassis
force applied in the same step still reflects the pre-coupling per-wheel
F_long (because F_long was computed inside the loop using pre-coupling
omegas). The coupled omega becomes visible to the next step. This is the
reference-tank behavior and is documented as the SDK contract.

## What hooks CANNOT do

- Mutate `T_drive_per_wheel` / `T_brake_per_wheel` (those are owned by the
  drivetrain; if you need custom torque routing, subclass the strategy).
- Insert at custom seams; only `PRE_LOOP` and `POST_TIRE` exist.
- Run during the air-mask branch with non-zero `N` (the air mask is applied
  before the hook sees the wheel).

For anything else, subclass the relevant strategy ABC instead.
