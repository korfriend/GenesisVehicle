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
| `"control"` (default) | `RollingResistance`, `LowSpeedRegularizer(disable_when_control_active=False)`, `StaticFrictionLock(brake_thr=0.3, v_thr=0.5, k_spring=500_000, k_damp=20_000)` | same as car (v0.5.8 unified — was tank-only pre-v0.5.8) | **Recommended for almost everyone**: RL / MPPI / general control / Real2Sim. The regularizer is active even under throttle (the v0.3.0 footgun fix). `StaticFrictionLock` (v0.5.7) is a position-anchored stick-slip model — zero drift on any slope up to the physical friction limit (μ ≈ tan(slope_angle)). See [`../samples/slope_hold.py`](../samples/slope_hold.py) for a regression check. |
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

> 핵심: hook 도 forward model 의 일부. fitting 과 deployment 가 같은 hook config 라면 그게 일관된 모델. 우리 SDK 로 fitting + 우리 SDK 로 배포면 `"control"` 그대로 쓰면 됨. `"raw"` 는 hook 의 기여도를 분리해서 보고 싶을 때만 (debugging / ablation).

## Hook ordering inside a profile

Hooks run in the list order returned by `stability_hooks_for_profile`. The
`"control"` profile keeps `RollingResistance` first and
`StaticFrictionLock` last so the lock has the final word on `F_long` at
rest. If you build a custom hook stack via `"research"`, follow the same
convention.
