# Stability profiles — `control` / `raw` / `research`

Low-speed stabilization of a ray-wheel + Pacejka model is a fundamental
numerical concern, not an optional add-on. Every preset and the
`stability_hooks_for_profile` helper select a **profile** that materializes
the appropriate set of hooks.

This mirrors the convention in other vehicle SDKs (PhysX
`eSTICKY_TIRE_FRICTION` mode, Jolt anti-rollback, Chaos low-speed threshold,
Unity WheelCollider stick-slip): the low-speed stabilization is *part of the
model*, not a plug-in the user has to remember to enable.

## The three profiles

| abbr | meaning |
|---|---|
| B_rem | brake torque left on a wheel after the brake projection (N·m; `ctx.brake_rem`) |
| D | the held-wheel one-step deadbeat on `F_long` (`physics-contracts.md` §7.16) |
| θ | slope angle |
| s | the deadbeat's rigid-body scale `min(1, I_cg / Σ (N_i/g)ρ_i²)` |
| ρ | brake torque over the slope's torque demand (> 1: the brake can hold) |

| Profile | Hooks materialized (car) | Hooks materialized (tank) | Use case |
|---|---|---|---|
| `"control"` (default) | `RollingResistance`, `LowSpeedRegularizer(disable_when_control_active=True)`, `StaticFrictionLock(v_thr=0.5, k_spring=500_000, k_damp=20_000)` (through 2.0.3 also `brake_thr=0.3`, removed in v2.0.4) | same as car (v0.5.8 unified — was tank-only pre-v0.5.8) | **Recommended for almost everyone**: RL / MPPI / general control / Real2Sim. The regularizer is OFF while the user throttles or brakes (`disable_when_control_active=True`, since v0.5.1, so a vehicle can pull away from rest). **This row said `disable_when_control_active=False` and "active even under throttle (the v0.3.0 footgun fix)" — wrong since v0.5.1, which reverted v0.3.0 (`presets.py`).** `StaticFrictionLock` (v0.5.7; rest anchor since v1.8.0, [`physics-contracts.md` §7.17](physics-contracts.md)) is a position-anchored stick-slip model that holds the point where the contact came to rest, limited by friction AND by the brake capacity. **Since v2.0.4 it engages on every wheel the brake holds (`B_rem > 0`) below `min(v_thr, 0.5)` m/s at ANY pedal, never forces ω, and its spring is capped per wheel from the vehicle's static load, planar rigid-body factor and `dt`** ([`physics-contracts.md` §7.22](physics-contracts.md)): reference car, 5° downhill, brake 0.3 rests (4e-6 m/s; 2.0.3 crept at 0.02497 m/s; measured, not enforced, `physics-contracts.md` §7.22.3), and a wheel whose drive torque exceeds its brake torque turns (`tank_skid_belt` brakes each of its n wheels with `30000·brake/n` N·m but drives each with the whole side torque, so the 10-wheel reference tank at full brake pivots at rest under steer above 0.1, in general 1/n). Through 2.0.3 it engaged only at brake > `brake_thr` (0.3), so at brake ≤ 0.3 a braked vehicle crept down a slope at the held-wheel deadbeat speed `g·tan θ·dt/s` (D acting as a damper); this row described that as the chassis sliding "on tyre slip" — **the mechanism was wrong** (the deadbeat D is a damper, `physics-contracts.md` §7.16), and v2.0.2 had already called the row incomplete for omitting the `brake_thr` condition. **This row said "zero drift on any slope up to the physical friction limit (μ ≈ tan(slope_angle))" — wrong: the hold is brake-limited (the `tank_skid_belt` preset's 75 kN of brake does not hold 10°, a 99 kN slope load), and on 1.7.0 the car crept down 10° under full brake at 0.907 m/s (CHANGELOG 1.8.0 §2.2).** Since v1.8.0 the reference car, braking at 1.0 on a 10° or 20° downhill, stops and then holds (≤ 7e-6 m/s after the stop); the stop itself is not drift-free — at 20° the base link reverses at up to -0.0208 m/s (from 3 m/s, 0 mm back-travel) / -0.0258 m/s with 0.81 mm back-travel (from rest), CHANGELOG 1.8.0 §6.2. Through v2.0.3 this row added "a light brake on a shallow slope still creeps (car 3°, brake 0.1: 0.0173 m/s)"; since v2.0.4 that row rests (reported, not gated: ρ ≈ 1.02 there — derived in the 2.0.4 release plan, not measured). See [`../samples/slope_hold.py`](../samples/slope_hold.py) for a regression check. `RollingResistance` takes effect only since v1.7.0 (through v1.6.10 the `F_long` overshoot clamp erased it); a coast now decelerates at `sum(cr*N)/M_eff` (cr = rolling-resistance coefficient, N = wheel load, M_eff = mass plus wheel-inertia equivalent) when rolling speed is well above 0.5 m/s (smaller below), and holding a speed needs throttle. |
| `"raw"` | _empty_ | _empty_ | Raw ray-wheel + Pacejka with NO numerical stabilization. For debugging the bare dynamics or running hook-ablation studies. Not a general-purpose default. |
| `"research"` | _empty_ | _empty_ | Caller assembles `cfg.stability_hooks` manually (advanced). |

```python
from genesis_vehicle import car_4w_rwd_ackermann, RollingResistance, StaticFrictionLock

# RL / MPPI / Real2Sim — default profile is correct.
cfg = car_4w_rwd_ackermann(URDF)

# Raw dynamics — debugging or ablation study.
cfg = car_4w_rwd_ackermann(URDF, stability="raw")

# Custom hook stack (advanced).
cfg = car_4w_rwd_ackermann(URDF, stability="research")
cfg.stability_hooks = [RollingResistance(tanh_scale=0.3),
                       StaticFrictionLock(k_spring=300_000.0)]   # keyword-only since 2.0.4
```

Through v2.0.3 this example passed `StaticFrictionLock(brake_thr=0.2)`; since
v2.0.4 that raises `TypeError` (the argument was removed, and the constructor
takes keyword arguments only). A `k_spring` you pass is still capped per wheel
by the lock (`physics-contracts.md` §7.22).

## The one rule (covers MPPI, RL, and Real2Sim)

> **Fitting forward model = deployment forward model.** Hooks are part of
> the model. Whatever stability profile you deploy with, fit with the same.
> For our SDK that almost always means `"control"` in both fitting and
> deployment.

- **MPPI / RL** — train against `"control"`, deploy against `"control"`.
- **Real2Sim** — fit (μ, K, B, …) against `"control"` so the fitted values
  reflect "how this vehicle behaves under the model you'll actually use".
  Fitted parameters are model-coupled: Pacejka `B` is not "rubber's true
  stiffness" but "the coefficient that, inside Pacejka inside our SDK with
  these hooks, best matches real data". That coupling is fine — desired,
  even — as long as deployment uses the same model.

## When `"raw"` is actually useful

Not for general Real2Sim. Use `"raw"` only when:

1. You want to study the bare ray-wheel + Pacejka dynamics (e.g. debug a
   stiff-ODE oscillation).
2. You want to measure the contribution of the hooks themselves (run the
   same scenario in `"control"` and `"raw"` and diff).
3. You are exporting parameter values to a different simulator that
   guarantees no equivalent hooks — but note: parameter values are not
   portable in any rigorous sense; refit in the destination simulator.

> Key point: the hooks are part of the forward model. Fitting and deployment with the same hook config is one consistent model. Fitting with this SDK and deploying with this SDK means using `"control"` as-is. Use `"raw"` only to isolate the hooks' contribution (debugging / ablation).

## Hook ordering inside a profile

Hooks run in the list order returned by `stability_hooks_for_profile`,
once per phase. Since v1.7.0 a `POST_TIRE` hook gets two calls per step:
every hook's `apply_pre_tire` runs (in list order) before the tyre, then every
hook's `apply_post_tire` runs (in list order) after it
([`pipeline-and-hooks.md`](pipeline-and-hooks.md)). The `"control"` profile
keeps `StaticFrictionLock` last so the lock has the final word on `F_long` at
rest; that is still what the order buys.

**Correction (v1.7.0).** This section used to say the profile keeps
`RollingResistance` first as part of the same rationale, implying its position
decided what happened to `F_long`. That was wrong: through v1.6.10 the hook's
`F_long` edit was erased by the overshoot clamp regardless of order, so its
position had no effect. Since v1.7.0 `RollingResistance` does all its work in
`apply_pre_tire` (it adds to `ctx.T_resist`, an axle torque applied before the
tyre), which runs before every hook's `apply_post_tire` whatever the list
order. Its list position now matters only relative to other pre-tire hooks,
and `T_resist` contributions add. If you build a custom hook stack via
`"research"`, keep `StaticFrictionLock` last.
