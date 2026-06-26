# Concepts

The mental model. First the two API layers, then five things to know before
reading the API reference.

## Start here: the two API layers

The SDK has two layers. **Reach for the high-level one first.**

- **High-level — `VehicleScene`** (the recommended entry point). It owns the
  Genesis scene(s), `gs.init` / `build` / `step` / the wheel raycast (including
  the two-scene optimization), and the registered vehicles / static / dynamic
  bodies. You write `add_vehicle` / `add_static` / `add_dynamic` → `build` → a
  loop of `veh.set_inputs(...) + vs.step()`, with no manual Genesis boilerplate.

- **Low-level — `VehiclePhysics`** (one vehicle, batched over `n_envs`) **/
  `MultiVehiclePhysics`** (K vehicles in one scene). This is the actual driver
  `VehicleScene` wraps internally. You own the `gs.Scene`, `scene.build`, the
  per-step loop, and the wheel raycaster; you call
  `physics.step(VehicleInputs(...))` then `scene.step()` yourself.

| | **`VehicleScene`** (high-level) | **`VehiclePhysics` / `MultiVehiclePhysics`** (low-level) |
|---|---|---|
| Owns | `gs.init` · scene(s) · `build` · `step` · raycast · two-scene | you own the scene + step loop |
| Use it for | the default — drop a car and drive, terrain / obstacles, L3 (`n_envs`) | fine control: the L2 batched solver (`MultiVehiclePhysics`), a custom step loop, solver / throughput benchmarks |
| Relationship | **wraps** `VehiclePhysics` | the **engine inside** `VehicleScene` |

Rule of thumb: **start with `VehicleScene`; drop to `VehiclePhysics` /
`MultiVehiclePhysics` only when you need control it doesn't expose** — e.g. the
L2 cross-vehicle batched solver, or a hand-written step loop. The
[`samples/`](../samples/README.md) show both: `quickstart` / `slope_hold` /
`obstacles_and_ramp` / `two_scene_terrain` use `VehicleScene`; the L2 / L3
batching and perf samples (`road_loop`, `l2l3_minimal`, `perf_*`,
`city_traffic_ego`) use the low-level API directly. See
[`api-reference.md`](api-reference.md) §0 (`VehicleScene`) and §1
(`VehiclePhysics`), and [`batching.md`](batching.md) for L1 / L2 / L3.

## 1. The 5-step pipeline

Every simulation step, for every wheel:

1. **Raycast** — distance `d` from the chassis-local wheel point straight down.
2. **Suspension** — compression `c = max(rest_d - d, 0)`, then a per-wheel
   asymmetric damper produces normal force `N`. See
   [`physics-contracts.md`](physics-contracts.md#72-normal-force-is-non-negative-air-mask-wheels-contribute-nothing)
   for the exact formula.
3. **Slip** — `kappa = (v_roll - v_long) / max(|v_long|, eps_v)`,
   `alpha = atan2(v_lat, |v_long|)`.
4. **Tire force** — `(F_long, F_lat) = TireModel(v_long, v_lat, v_roll, N, wheel_params)`.
   Pacejka does friction-ellipse clamping internally.
5. **omega + chassis force-at-point** —
   `domega = (T_drive - T_brake_eff - R * F_long) / I_wheel`; chassis gets
   `N*up + F_long*fwd + F_lat*lat` and `r × F` torque at
   `r = wheel - chassis_center`. `T_brake_eff` is the signed form
   (see [contracts](physics-contracts.md#71-brake-torque-is-a-positive-command-magnitude)).

`VehiclePhysics.step()` runs this loop with strategy hooks plugged at fixed
seams. The full diagram lives in
[`pipeline-and-hooks.md`](pipeline-and-hooks.md).

## 2. Coordinate / sign conventions (ISO 8855)

| Axis | Direction |
|---|---|
| `+X` | forward |
| `+Y` | left |
| `+Z` | up |

| Input | Range | Meaning |
|---|---|---|
| `throttle` | `[-1, +1]` | signed; `+` accelerates forward, `-` reverses |
| `brake` | `[0, 1]` | positive magnitude — internally converted to a signed torque opposing wheel rotation |
| `steer` | `[-1, +1]` | **`+steer` = right turn** |

Internal sign flips (Genesis RHS, URDF `<axis 0 0 -1>`, KDU's legacy
`+steer = LEFT`) are absorbed inside the strategy / visual layer. The
user-facing API is unambiguous.

## 3. Strategies — composition over inheritance

A `VehicleConfig` is built from four orthogonal strategies plus optional
stability hooks:

| Axis | Concrete options |
|---|---|
| `SteeringStrategy` | `Ackermann`, `PartialAckermann`, `SkidSteer`, `NoSteer` |
| `DrivetrainStrategy` | `FWD`, `RWD`, `AWD`, `PerSide` |
| `CouplingStrategy` | `Independent`, `SameSideBelt` |
| `TireModel` | `PacejkaAnisotropic`, `CoulombIsotropic` |
| `StabilityHook` (via `stability=` profile) | `RollingResistance`, `LowSpeedRegularizer`, `StaticFrictionLock` |

Each is a small class you can subclass to add a new behavior. See
[`api-reference.md`](api-reference.md) for signatures.

## 4. URDF is the default source; explicit API overrides are the final truth

For every URDF-derivable field (positions, radii, masses, inertia, joint
names) the precedence is:

```
module default  <  URDF-derived default  <  explicit API override
```

URDF provides defaults; `WheelConfig` / `VehicleConfig` values that the user
explicitly set always win. After `resolve()`, the `ResolvedConfig` is the only
source of truth used by `VehiclePhysics`.

`i_wheel` follows the same precedence with one addition: if both the user
and the URDF leave it unset, the SDK consults Genesis runtime metadata as a
last-resort estimate. See
[`physics-contracts.md`](physics-contracts.md#73-wheelconfigi_wheel-truth-policy).

## 5. What is a "stability hook"?

A **hook** is a post-processing callback that runs at a fixed point in the
5-step pipeline. The main dynamics (raycast → suspension → slip → tire → ω)
are not modified; the hook just *adds one more term* at a designated seam,
the same way `git pre-commit`, a web-framework middleware, or a Java
interceptor works. Examples:

- `RollingResistance` runs after the tire model produces `F_long`, and
  subtracts `cr * N * tanh(v_long / scale)`.
- `LowSpeedRegularizer` runs once pre-loop (computes a `moving ∈ [0, 1]`
  factor) and once per wheel post-tire (scales `F_long`/`F_lat` by `moving`
  and pulls `omega` toward rolling-without-slip).
- `StaticFrictionLock` runs post-tire when `brake > thr` and planar
  wheel speed `< thr`; engages a per-wheel position anchor and applies
  a 2D spring-damper `F = -K_spring·displacement - K_damp·velocity`,
  projected onto the per-wheel friction ellipse, then forces `omega = 0`.
  Stuck = vehicle is truly stationary; slipping = anchor advances along
  the friction limit (kinetic mode).

Selecting which hooks are active is done via a **stability profile**
(`"control"` / `"raw"` / `"research"`), not by assembling a free-form list.
See [`stability-profiles.md`](stability-profiles.md) for the why and the how.

## 6. Batched by default — three axes

Every state tensor is `(n_envs, n_wheels)` or `(n_envs, 3/4)`. Single-env
(`n_envs=1`) is just a special case. Scalar OR `(n_envs,)` tensor inputs are
both accepted. Use `physics.reset(env_ids=...)` for partial reset (RL / MPPI).

There are three orthogonal batching axes:

- **L1** — wheels of ONE vehicle, batched inside `VehiclePhysics` (always on).
- **L2** — K vehicles of the SAME URDF/cfg in one Genesis env, batched via
  `MultiVehiclePhysics` (since v0.5.11) — or just `VehicleScene(solver="batched")`,
  the default, which groups same-kind vehicles for you.
- **L3** — N parallel envs, batched via `scene.build(n_envs=N)` (since v0.1), or
  `VehicleScene(n_envs=N)`.

L2 and L3 compose into `MultiVehiclePhysics(scene, vehicles, n_envs=N)`
for `N·K` total vehicles in one batched compute call (since v0.5.14) — which is
exactly what `VehicleScene(solver="batched", n_envs=N)` drives under the hood.

See [`batching.md`](batching.md) for the decision matrix, measured
speedups, and the L2 × L3 combined pattern (the headline workflow for
autonomous-driving ego + traffic in MPPI / RL).
