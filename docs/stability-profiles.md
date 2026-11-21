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

| Profile | Hooks materialized (car) | Hooks materialized (tank) | Use case |
|---|---|---|---|
| `"control"` (default) | `RollingResistance`, `LowSpeedRegularizer(disable_when_control_active=True)`, `StaticFrictionLock(brake_thr=0.3, v_thr=0.5, k_spring=500_000, k_damp=20_000)` | same as car (v0.5.8 unified — was tank-only pre-v0.5.8) | **Recommended for almost everyone**: RL / MPPI / general control / Real2Sim. The regularizer is OFF while the user throttles or brakes (`disable_when_control_active=True`, since v0.5.1, so a vehicle can pull away from rest). **This row said `disable_when_control_active=False` and "active even under throttle (the v0.3.0 footgun fix)" — wrong since v0.5.1, which reverted v0.3.0 (`presets.py`).** `StaticFrictionLock` (v0.5.7; rest anchor since v1.8.0, [`physics-contracts.md` §7.17](physics-contracts.md)) is a position-anchored stick-slip model that holds the point where the contact came to rest, limited by friction AND by the brake capacity. **This row said "zero drift on any slope up to the physical friction limit (μ ≈ tan(slope_angle))" — wrong: the hold is brake-limited (the `tank_skid_belt` preset's 75 kN of brake does not hold 10°, a 99 kN slope load), and on 1.7.0 the car crept down 10° under full brake at 0.907 m/s (CHANGELOG 1.8.0 §2.2).** Since v1.8.0 the reference car, braking at 1.0 on a 10° or 20° downhill, stops and then holds (≤ 7e-6 m/s after the stop); the stop itself is not drift-free — at 20° the base link reverses at up to -0.0208 m/s (from 3 m/s, 0 mm back-travel) / -0.0258 m/s with 0.81 mm back-travel (from rest), CHANGELOG 1.8.0 §6.2; a light brake on a shallow slope still creeps (car 3°, brake 0.1: 0.0173 m/s). See [`../samples/slope_hold.py`](../samples/slope_hold.py) for a regression check. `RollingResistance` takes effect only since v1.7.0 (through v1.6.10 the `F_long` overshoot clamp erased it); a coast now decelerates at `sum(cr*N)/M_eff` (cr = rolling-resistance coefficient, N = wheel load, M_eff = mass plus wheel-inertia equivalent) when rolling speed is well above 0.5 m/s (smaller below), and holding a speed needs throttle. |
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
                       StaticFrictionLock(brake_thr=0.2)]
```

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
