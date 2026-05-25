# Changelog — genesis_vehicle

All notable changes are documented here. Versions follow
[Semantic Versioning](https://semver.org/) (major.minor.patch).

`__version__` lives in `genesis_vehicle/_version.py` and is re-exported as
`genesis_vehicle.__version__` plus the `version()` / `version_info()`
functions. `VehiclePhysics.__init__` prints a one-line banner with the
running version the first time it is instantiated in a process.

---

## [0.5.13] — 2026-05-25

### Added — `perf_multi_vehicle` sample (L2 solver comparison)

[`samples/perf_multi_vehicle.py`](samples/perf_multi_vehicle.py) — the
L2 counterpart to `perf_vectorization.py`. Sweeps `--n_per_kind` and
runs the same 4-kind fleet under both solvers (`per_vehicle` and
`multi_batched`) in fresh subprocesses, prints a scaling table:

| n_per_kind | total | per_vehicle (ms) | multi_batched (ms) | speedup |
|-----------:|------:|-----------------:|-------------------:|--------:|
|          1 |     4 |            75.27 |              78.84 |   0.95× |
|          2 |     8 |           208.20 |             183.25 |   1.14× |

At K=1 (4 vehicles, 1 per kind) `multi_batched` is slightly SLOWER —
the batching machinery has setup overhead with nothing to batch (each
"kind" group is just 1 vehicle). From K=2 upward the gap widens; in
practice expect 1.1-1.2× speedup for realistic multi-vehicle scenes.

Use this script to:
- Decide which solver to use for your specific fleet size + kind mix.
- Catch regressions if either solver's perf drifts.

The output makes it concrete that L2 batching is helpful but bounded
(see also `MultiVehiclePhysics` docstring) — for pure throughput on
one vehicle kind, n_envs > 1 (L3) remains the right tool.

### Docs

- `samples/README.md` — added row #7.

---

## [0.5.12] — 2026-05-25

### Fixed — `MultiVehicleKindPhysics` now supports VisualSync

v0.5.11 disabled the internal VisualSync entirely because the proto
`VehiclePhysics(n_envs=K)` it inherits from would issue a
`set_dofs_position` with shape `(K, n_dofs)` into a scene actually
built with `n_envs=1`, which Genesis rejects (the K dim is "vehicles
in this env", not "parallel envs").

Fix: build K independent `VisualSync` instances (one per entity, each
`n_envs=1`). After the batched compute pipeline produces (K, n_wheels)
spin / steer / suspension targets, the visual step slices them into K
per-entity (1, n_wheels) chunks and dispatches K small
`set_dofs_position` calls — a Python loop, but each call is tiny.

Measured overhead: ~34 ms added to the 16-vehicle road_loop step
(760 → 794 ms / step). Net L2 speedup vs per_vehicle (848 → 794)
shrinks slightly from 10% to ~6%, but every vehicle now gets correct
wheel-spin and steering visuals — making `multi_batched` a drop-in
replacement for `per_vehicle` in visual demos.

### Migration

None — same `MultiVehiclePhysics` constructor signature. Existing code
written against v0.5.11 keeps working and gets visuals for free.

---

## [0.5.11] — 2026-05-25

### Added — `MultiVehiclePhysics` (L2 cross-vehicle batching)

New SDK class for the case where K vehicles share ONE Genesis scene at
DIFFERENT positions (traffic, multi-agent, MPPI candidate fleets). The
default ``VehiclePhysics`` is one-per-entity, so K vehicles in the same
scene meant a Python loop of K ``step()`` calls per simulation step.

``MultiVehiclePhysics`` groups vehicles by URDF / cfg identity and runs
ONE batched compute pipeline per kind:

- Pacejka tire model: one call over ``(K_kind, n_wheels)`` instead of
  K_kind separate calls.
- Stability hooks: one batched invocation per hook per kind.
- Force + torque application: one ``apply_links_external_force`` call
  with K_kind link indices instead of K_kind separate calls.

The compute batching uses the rigid solver's existing batched APIs
(``get_links_pos``, ``get_links_quat``, ``get_links_vel``,
``get_links_ang``) so cross-vehicle reads are also a single call per
quantity per kind.

```python
from genesis_vehicle import MultiVehiclePhysics, VehicleInputs

# Build the scene as usual: K different entities, each spawned at its own
# pose, each with its own sensor. SHARE the cfg instance across vehicles
# of the same kind — MultiVehiclePhysics groups by ``id(cfg)``.
cfg_per_kind = [preset_fn(urdf) for ...]
vehicles = []
for k, (urdf, preset_fn) in enumerate(kinds):
    for _ in range(K):
        ent, sens, _ = add_vehicle(scene, urdf, preset_fn=None, pos=(...))
        vehicles.append((ent, sens, cfg_per_kind[k]))

scene.build(n_envs=1)
mphys = MultiVehiclePhysics(scene, vehicles)
for step in range(N):
    mphys.step([VehicleInputs(...) for _ in vehicles])
    scene.step()
```

### Performance

L2 batching helps but is bounded by Genesis's per-entity ``scene.step()``
cost (which dominates as the number of in-scene entities grows). For a
16-vehicle ``road_loop`` scenario the measured speedup is **~10%**
(848 → 760 ms/step on an RTX 5070 Laptop).

For dramatic batching speedups, ``n_envs > 1`` (L3) is still the right
tool — see ``samples/perf_vectorization.py`` (44× at n_envs=64). L2's
value is enabling multi-vehicle scenes (visualization, multi-agent
interaction, mixed-kind comparisons) at a modest extra cost vs the
per-vehicle loop, NOT replacing L3 for pure-throughput RL/MPPI.

### Sample integration

``samples/road_loop.py`` gains ``--solver {per_vehicle, multi_batched}``
(default ``per_vehicle``) and ``--bench`` flags so users can compare the
two on the same 4-kind, K-per-kind fleet:

```bash
python -m genesis_vehicle.samples.road_loop --n_per_kind 4 --bench --solver per_vehicle
python -m genesis_vehicle.samples.road_loop --n_per_kind 4 --bench --solver multi_batched
```

### Known limitations

- `MultiVehicleKindPhysics` disables its internal `VisualSync` (the
  visual joint updates would need K separate VisualSync objects, not
  in v0.5.11). Wheels won't visibly spin in the chase-cam view; chassis
  motion is correct.
- Vehicles of the SAME kind must share the SAME cfg INSTANCE (not just
  the same URDF) — group by passing `cfg_per_kind[k]` instead of calling
  the preset fresh per vehicle. `MultiVehiclePhysics` groups by
  `id(cfg)`.

All 60 SDK tests pass; no changes to `VehiclePhysics` or other existing
APIs.

---

## [0.5.10] — 2026-05-25

### Added — `multi_env_render` sample

[`samples/multi_env_render.py`](samples/multi_env_render.py) closes the
loop on the v0.5.9 batching story: ``perf_vectorization`` shows you the
44× speedup at ``n_envs=64`` as numbers, but until now there was no
way to actually SEE what those 64 parallel rollouts were doing —
Genesis simulates them at the same world coordinates (the env axis is
a "parallel universe" axis, not a spatial one).

The fix is a Genesis feature pair we hadn't exposed in samples yet:

```python
scene = gs.Scene(
    vis_options=gs.options.VisOptions(env_separate_rigid=True, ...),
    ...,
)
scene.build(
    n_envs=N,
    env_spacing=(dx, dy),     # visualization-only offset
    n_envs_per_row=K,         # √N by default
)
```

The renderer then offsets each env's rigid entities to its grid cell.
Physics is unchanged (still parallel universes at the same coords);
only the visualization adds the per-env transform. The overhead
camera frames the entire grid, so MPPI candidate trajectories or RL
rollout diversity is visible at a glance.

Per-env throttle / steer are randomized so every cell looks different —
useful as a sanity check that batched simulation isn't accidentally
making all envs identical.

### Docs

- `samples/README.md` updated with row #6.

---

## [0.5.9] — 2026-05-24

### Added — two more samples (`road_loop`, `perf_vectorization`)

- [`samples/road_loop.py`](samples/road_loop.py) — multi-vehicle visual
  demo. 4 distinct kinds (FWD red sedan, RWD blue coupe, AWD green SUV,
  yellow 6-wheel truck), `--n_per_kind` each, all driving a circular
  track under constant Ackermann steering. Top-down camera frames the
  whole fleet. URDFs are generated parametrically at runtime to a
  tempdir so the demo stays self-contained.

- [`samples/perf_vectorization.py`](samples/perf_vectorization.py) —
  `n_envs` batching speedup benchmark. Sweeps
  `n_envs ∈ [1, 4, 16, 64, 256, 1024]` (one fresh subprocess per
  measurement to keep GPU state clean) and prints a scaling table.
  Typical result on an RTX 5070 Laptop:

  | n_envs | ms / step | env-steps / s | per env (μs) | speedup |
  |-------:|----------:|--------------:|-------------:|--------:|
  |      1 |     26.28 |            38 |       26 277 |    1.0× |
  |      4 |     36.78 |           109 |        9 196 |    2.9× |
  |     16 |     37.28 |           429 |        2 330 |   11.3× |
  |     64 |     37.86 |         1 691 |          592 |  44.4× |

  `ms / step` stays roughly constant from n_envs=4 upward — Genesis +
  the SDK saturate the GPU, so each additional parallel env is nearly
  free. This is the headline batching benefit that RL / MPPI workloads
  exploit. The sweep is honest (each row a separate GPU process) rather
  than relying on simulated kernel-launch overhead.

### Docs

- `docs/api-reference.md` runnable-references callout extended to list
  all five samples.
- `samples/README.md` updated.

---

## [0.5.8] — 2026-05-24

### Added — bundled `samples/` directory

Three minimal, self-contained reference scripts ship with the SDK so
that `pip install genesis-vehicle` is immediately runnable without
hunting for assets or umbrella-level helpers:

- [`samples/quickstart.py`](samples/quickstart.py) — minimum-viable
  call pattern (preset → `add_vehicle` → step loop). Drives a car
  forward for 5 s on flat ground.
- [`samples/slope_hold.py`](samples/slope_hold.py) — `StaticFrictionLock`
  side-slope hold regression check. Reports OK / REGRESSION based on
  lateral slip over 10 s under `brake=1.0`.
- [`samples/batched_rollout.py`](samples/batched_rollout.py) — batched
  `n_envs > 1` API for RL / MPPI. Demonstrates per-env random controls
  and per-step throughput.

Bundled asset:
- [`samples/urdf/car_4w.urdf`](samples/urdf/car_4w.urdf) — minimal
  1.2 t 4-wheel RWD sedan, primitive geometry only (no mesh files),
  follows project URDF conventions.

Run as Python modules:

```bash
python -m genesis_vehicle.samples.quickstart
python -m genesis_vehicle.samples.slope_hold --slope 20
python -m genesis_vehicle.samples.batched_rollout --n_envs 256
```

Docs (`quickstart.md`, `stability-profiles.md`, `api-reference.md`)
now cross-reference the samples as runnable examples.

### Changed — `"control"` profile is now uniform across vehicle kinds

`stability_hooks_for_profile("control", vehicle_kind=...)` now returns
the same hook set (`[RollingResistance, LowSpeedRegularizer,
StaticFrictionLock]`) regardless of `vehicle_kind`. Previously only
`vehicle_kind="tank"` got the lock — `vehicle_kind="car"` was hook[0:2]
only. That was a real footgun: any preset car on a slope (or after an
impact-induced lateral velocity) would creep indefinitely under brake.

The new stick-slip lock (v0.5.7) is cheap enough that there's no
reason to leave cars without it. The truck preset's manual `append` of
the lock is removed since the base profile now includes it.

`vehicle_kind` is kept as a parameter for forward compatibility with
future vehicle-specific tweaks, but currently has no effect on the
profile hook set.

Test update: `test_profile_control_car_includes_static_friction_lock`
replaces the pre-v0.5.8 `..._returns_two_hooks` test.

---

## [0.5.7] — 2026-05-24

### Fixed — `StaticFrictionLock` is now true static friction (stick-slip)

User physics critique on v0.5.6 (the tanh velocity-damper version): a
real asphalt-tire contact at μ=1.0 should hold a vehicle stationary on
any slope where μ > tan(slope_angle) — for a 20° slope that's μ > 0.36,
well within μ=1.0. v0.5.6 still showed ~5 cm drift over 10 s at 20°
because `F = -μN·tanh(v/scale)` is fundamentally a *kinetic* friction
model: any non-zero `v` produces an opposing force, but `v=0` produces
`F=0`. Equilibrium on a slope therefore requires non-zero v_lat — visible
creep, even though physics says the vehicle should be motionless.

### Changes

- `genesis_vehicle/strategies/stability.py` — rewrote `StaticFrictionLock`
  as a **position-anchored stick-slip** model:
  - When the lock first engages on a wheel, the contact position becomes
    its anchor (displacement `d = 0`).
  - Each step while active: integrate `d += v · dt`; compute force as a
    spring + damper: `F = -K_spring·d - K_damp·v`.
  - Project `(F_long, F_lat)` onto the per-wheel friction ellipse (same
    form as `tire_models/pacejka.py`).
  - If the projection clamps (i.e., wheel is slipping), advance the anchor
    so the next-step spring force matches the friction limit — kinetic-
    friction behaviour without spring runaway.
  - Otherwise (stuck), the spring quietly absorbs any external lateral
    force up to μN. Vehicle is **truly stationary**.
  - New tuning parameters: `k_spring` (default 500_000 N/m per wheel,
    around the explicit-Euler stability limit at dt=0.02 for a 5 t
    chassis) and `k_damp` (default 20_000 N·s/m per wheel, ~half-critical
    at the natural frequency).
  - Legacy `hold_k`, `hold_k_lat`, `slip_scale` constructor kwargs are
    REMOVED (see BREAKING section below).
- `genesis_vehicle/core.py` — added `ctx.dt: float` to PipelineContext
  so hooks can integrate per-step state (the stick-slip lock needs it).
- `genesis_vehicle/presets.py` — tank and truck preset call sites updated
  to the new signature (`StaticFrictionLock(brake_thr=0.3, v_thr=0.5)`,
  no `hold_k` arg).
- `genesis_vehicle/urdf.py` + `__init__.py` — removed deprecated alias
  `parse_inertia_max_principal_genesis` (was a back-compat shim from
  v0.4.x → v0.5.0 rename; no longer needed in pre-1.0).

### Verified — `samples/slope_lateral_slip.py` truck preset

| slope    | v0.5.5 (1D lock)  | v0.5.6 (2D tanh) | v0.5.7 (stick-slip) |
|---|---|---|---|
| flat 0°  | -72 mm            | 0 mm             | **0.0 mm**          |
| 20°      | -114 mm           | -54 mm           | **2.9 mm**          |
| 30°      | (not tested)      | (not tested)     | **0.4 mm**          |
| 40°      | (rolls over)      | (rolls over)     | (rolls over)        |

The 40° case is a real physical limit: μ=1.0 → max-hold-able slope ≈
arctan(1.0) = 45°, but a high-COG 5 t truck tips over at ~35-40° well
before reaching the friction limit. The lock cannot prevent tip-over
(would need a separate "rollover restraint" hook, out of scope).

### BREAKING — Migration required

`StaticFrictionLock` no longer accepts `hold_k`, `hold_k_lat`, or
`slip_scale`. They were proportional / tanh gains in the prior velocity-
damper models (v0.5.5, v0.5.6) and have no meaning in the new stick-slip
model. Replace with `k_spring` / `k_damp` (or drop them and use the
defaults, which are tuned for typical 1-5 t vehicles):

```python
# Before:
StaticFrictionLock(brake_thr=0.3, v_thr=0.5, hold_k=400_000.0)
# After:
StaticFrictionLock(brake_thr=0.3, v_thr=0.5)
# or explicitly:
StaticFrictionLock(brake_thr=0.3, v_thr=0.5,
                   k_spring=500_000.0, k_damp=20_000.0)
```

Pre-1.0 SDK — no deprecation shim. Bundled presets (tank, truck) have
been updated.

---

## [0.5.6] — 2026-05-24

### Fixed — `StaticFrictionLock` 2D extension (lateral slip on slopes)

Student bug report from MPPI work: a truck holding `brake=1.0` on a 20°
side slope continued to creep downhill ~11 cm over 10 s, despite the
control-profile stability hooks all being active. Root cause:
``StaticFrictionLock`` only wrote ``ctx.F_long``; ``ctx.F_lat`` came
straight from Pacejka, which is ill-conditioned near rest because
``alpha = atan2(v_lat, max(|v_long|, eps_v=0.5))`` artificially
compresses the slip angle (so ``F_lat`` from the magic formula was
much smaller than what a real tire would produce at rest).

### Changes

- `genesis_vehicle/strategies/stability.py` — `StaticFrictionLock` now:
  - Activates on planar wheel speed ``sqrt(v_long² + v_lat²) < v_thr``
    (was: ``|v_long| < v_thr`` only).
  - Overrides both ``F_long`` and ``F_lat`` with a smooth saturated
    hold ``-mu·N · tanh(v / slip_scale)`` per axis, then projects onto
    the per-wheel friction ellipse — same form as the Pacejka clamp.
  - Switched from `clamp(-hold_k · v, ±mu·N)` (proportional + hard
    clamp) to `tanh(v / slip_scale)`: the original form was a
    bang-bang controller whose effective gain in the saturated regime
    exceeded the explicit-Euler stability bound (`K_total · dt / mass
    = 9.6` for the truck preset, well above the ~2 stability limit) —
    fine in 1D where ``v_long`` was always tiny, but on a side slope
    the sustained lateral gravity excited the discretization into a
    growing oscillation. The tanh form keeps `|F| ≤ mu·N` smoothly,
    eliminating overshoot.
  - New `hold_k_lat` parameter (defaults to `hold_k`) and `slip_scale`
    (defaults to `v_thr / 5` — the stable upper limit on the truck
    preset at dt=0.02).
- `_version.py`: 0.5.5 → 0.5.6.
- `docs/pipeline-and-hooks.md`, `docs/stability-profiles.md`: updated
  hook descriptions.

### Verified — `samples/slope_lateral_slip.py`

| slope | pre-patch slip | post-patch slip | verdict |
|---|---|---|---|
| flat (0°)     | -72 mm  | **0 mm**   | OK |
| 10° side      | n/a     | **32 mm**  | NO BUG OBSERVED (under 80 mm threshold) |
| 20° side      | -114 mm | **54 mm**  | NO BUG OBSERVED |

### Known limitation

The tanh saturated form is a pure velocity damper: any sustained external
lateral force (gravity component on a slope) produces an equilibrium drift
velocity proportional to (force) / (gain near v=0). At the maximum stable
gain, this is ~5-10 mm/s on the truck preset at 20°. Truly zero-drift
hold on steep slopes (> ~25°) would require a position-tracked stick-slip
model (store a per-wheel anchor at lock-on time, spring + damper to it,
release when force exceeds the static-friction circle). Filed as future
work; current behavior is the standard fix for the Real2Sim / MPPI use
cases that motivated the bug report.

### Migration

`StaticFrictionLock(brake_thr=..., v_thr=..., hold_k=...)` continues to
work — `hold_k_lat` defaults to `hold_k`, `slip_scale` defaults from
`v_thr`. No call-site changes required in `presets.py` or external code.

---

## [0.5.5] — 2026-05-18

### Changed — `truck_6w_partial_ackermann` brake behavior

User report from the Truck6w demo: pressing SPACE (brake) while the
scenario kept driving throttle on produced visibly slow deceleration —
the brake was working mechanically (~1.8 m/s² decel for a 5-ton truck
with throttle still applied), but not enough to feel responsive.

Two preset tweaks:

- `t_brake_max` 8000 → 15_000 N·m. Realistic emergency-brake decel for a
  5-ton chassis is ~4-5 m/s²; the old value capped at ~2 m/s² (and that
  was while fighting throttle). Now ~4 m/s² with throttle off.
- The `"control"` stability profile for the truck preset now also includes
  `StaticFrictionLock(brake_thr=0.3, v_thr=0.5, hold_k=400_000)` (same
  pattern as the tank preset). The truck used to creep at < 0.5 m/s after
  the brake's `tanh(omega/0.5)` smoothing weakened — now it holds at rest.

The `"raw"` and `"research"` profiles get no StaticFrictionLock (consistent
with their "no hooks" semantics).

### Demo behavior — `GeneVehicle_Truck6w/demo_drive.py`

SPACE is now an **emergency brake**: it forces throttle to 0 in addition
to setting brake = 1. Mirrors real-car panic-brake semantics (brake pedal
overrides the accelerator). The scenario throttle resumes when SPACE is
released. Reason for the change: the previous behavior (SPACE adds brake
on top of scenario throttle) made `T_drive` and `T_brake` fight each other
and decelerated very gently — confusing to the human.

For raw "brake-while-throttling" research, hold SPACE and observe — but be
aware that the scenario throttle is now zeroed under SPACE.

---

## [0.5.4] — 2026-05-18

### Documentation — URDF steer joint axis recommendation

Added a recommendation in `docs/physics-contracts.md` §7.4: new URDFs
should declare steer joint axes as `<axis xyz="0 0 -1"/>` so that
`+joint_angle` and `+steer` (user-facing ISO 8855) share the same sign
domain. The SDK's `VisualSync` still compensates for either axis convention
via `visual_cmd = -phys * sign`, so existing URDFs with `(0, 0, 1)` (e.g.
JMK) keep working — the recommendation is only for new URDFs.

### Fixed in repo
- `GeneVehicle_Truck6w/urdf/truck_6w.urdf` — steer joint axes changed from
  `(0, 0, 1)` to `(0, 0, -1)` to match the recommendation. Visual behavior
  was already correct (v0.5.3 fix), but the URDF now follows the
  recommended convention internally.

---

## [0.5.3] — 2026-05-18

### Fixed — steer wheel visual rotated the wrong direction

`VisualSync` was driving the steer joints **opposite to the physics-side
steering direction** for both URDF axis conventions:

- URDF axis `(0, 0, 1)` (e.g. truck preset, JMK URDF) — visual wheels rotated LEFT when physics rotated RIGHT, and vice versa.
- URDF axis `(0, 0, -1)` (HJW URDF) — same inversion.

The old formula `visual_cmd = phys * sign` assumed the only conversion
needed was the URDF axis flip captured in `sign`. It missed that the
physics-side `steer_per_wheel` is **opposite-handed** from the URDF joint
convention to begin with (physics +θ = right turn = CW from above; URDF
axis `(0,0,1)` +joint = CCW from above). The correct formula is
`visual_cmd = -phys * sign`:

- axis `(0, 0, 1)`: `sign=+1` → `visual_cmd = -phys` → joint goes CW for +phys → right turn visual ✓
- axis `(0, 0, -1)`: `sign=-1` → `visual_cmd = +phys` → joint goes CW for +phys → right turn visual ✓

Demos affected: HJW (4-wheel car, axis -1) and Truck6w (axis +1). The
4-wheel HJW demo never had the wheel direction verified visually; the
truck demo just hit it (front wheels turned left when the truck arced
right). KDU (skid-steer, no steer joints) and JMK demos are unaffected.

---

## [0.5.2] — 2026-05-18

### Changed — `truck_6w_partial_ackermann` preset

- **Drive layout fixed to match real trucks.** Was `AWD` with uniform 1/6 split across all 6 wheels (front-axle steered wheels were also driven). Now `RWD(driven_axles=(1, 2))` — only the middle and rear axles drive, the front axle is steered-only. This matches conventional truck drivetrain layouts (front steer, rear/mid drive) and stops the visible weirdness of the steering wheels also producing forward torque.
- **Torque sized for the truck's mass.** `t_drive_max` 1500 → 10_000 N·m (was producing ~0.13 m/s² acceleration on a 5-ton truck after rolling resistance — barely visible). `t_brake_max` 3500 → 8000 N·m for symmetry. At full throttle the truck now hits ~2-3 m/s² acceleration without wheelspin (Pacejka peak per driven wheel at equilibrium ≈ 8800 N vs ~2500 N required).

---

## [0.5.1] — 2026-05-18

### Fixed — preset vehicles could not start from rest under throttle

The `"control"` stability profile built `LowSpeedRegularizer` with
`disable_when_control_active=False` (set in v0.3.0). At `v=0, omega=0`,
the regularizer:

1. Scaled `F_long` and `F_lat` by `moving = 0` → no propulsion force.
2. Set `omega_pull_factor = 1`, `omega_pull_target = v_long / radius = 0`
   → forced `omega` back to 0 every step.

Combined: any preset vehicle (`car_4w_rwd_ackermann`, `tank_10w_skid_belt`,
`truck_6w_partial_ackermann`, …) issuing `throttle > 0` at rest would
spin its wheels for one step and then have `omega` snapped back to 0.
**Vehicle stuck at rest indefinitely.** Discovered while running the
6-wheel truck demo where the truck wouldn't accelerate.

### Reverted — `disable_when_control_active=True` is back in `"control"` profile

This restores the original HJW behavior: the regularizer is off when the
user is actively throttling or braking. The vehicle can accelerate from
rest as expected. The regularizer still fires when the chassis is at
rest with no input (suppressing drift jitter), which is its original
intent.

### Trade-off note

The v0.3.0 change was made on the hypothesis that an always-on
regularizer would help MPPI low-speed uphill oscillation. That
hypothesis was never validated and the resulting behavior breaks every
demo. MPPI users who want an always-on regularizer can opt in
explicitly via `stability="research"` + a custom hook list:

```python
cfg.stability_hooks = [
    RollingResistance(),
    LowSpeedRegularizer(disable_when_control_active=False),
]
```

### Test update

`test_profile_control_uses_mppi_friendly_lowspeed_default` →
`test_profile_control_disables_regularizer_under_throttle`. Same assertion
flipped (`False` → `True`).

---

## [0.5.0] — 2026-05-18

### Performance — vectorized per-wheel pipeline

The Python `for i in range(n_wheels)` loop inside `VehiclePhysics.step()` is gone. All per-wheel work — suspension N, wheel-frame fwd/lat transforms, slip, tire force, stability hooks, omega update, force accumulation — is now a single batched tensor op set operating on `(n_envs, n_wheels)` tensors.

Measured on the KDU 10-wheel tank interactive demo:
- Before (v0.4.4): physics ≈ 50 ms / step (chase-cam interactive, `--profile` mode)
- 250+ CUDA kernel launches per step (10 wheels × ~25 ops per wheel)
- At `n_envs=1`, launch overhead (~10-30 µs each) dominated the actual GPU work, costing 5-10 ms in pure dispatch overhead alone.

After v0.5.0: ~25-30 kernel launches per step regardless of `n_wheels`. Helps interactive single-env use (where overhead was the bottleneck) AND batched RL/MPPI (cleaner code, same speed or better).

### BREAKING — stability hook API

- `StabilityHook.apply_post_tire(ctx, wheel_idx)` → `StabilityHook.apply_post_tire(ctx)`. No more per-wheel index. All `ctx.F_long`, `ctx.F_lat`, `ctx.N`, `ctx.v_long`, `ctx.v_lat` are now `(n_envs, n_wheels)` tensors. Hooks operate on the whole batch at once.
- `PipelineContext.wheel_params` (list of WheelConfig) → `PipelineContext.wheel_meta` (the cached `WheelMeta` with batched coefficient tensors). Read per-wheel coefficients via `ctx.wheel_meta.mu_long`, `ctx.wheel_meta.pb_x`, etc.
- All three bundled hooks (`RollingResistance`, `LowSpeedRegularizer`, `StaticFrictionLock`) updated. Custom hooks need to be ported.

### BREAKING — TireModel API

- `TireModel.__call__(v_long, v_lat, v_roll, N, wheel_params)` now expects `(n_envs, n_wheels)` tensors and a `WheelMeta` (not a single `WheelConfig`). Both `PacejkaAnisotropic` and `CoulombIsotropic` read per-wheel coefficients from the meta. Custom tire models need to be ported.

### Added — `WheelMeta` batched coefficient tensors

`WheelMeta` now exposes per-wheel `(n_wheels,)` tensors for: `mu_long`, `mu_lat`, `rolling_resistance_cr`, `pb_x`, `pc_x`, `pe_x`, `pb_y`, `pc_y`, `pe_y`. Built once at `VehiclePhysics.__init__` from the resolved `WheelConfig` list. Cheap to broadcast against `(n_envs, n_wheels)` state.

### Removed — `dynamics.suspension_normal_force` (still importable)

The pure-Python helper is still exported and unit-tested, but `core.py` no longer calls it — the suspension N calculation is inlined as a batched op in `step()` for one less function-call layer. Helper remains for external use.

### Other

- `StaticFrictionLock.apply_post_tire` no longer does the `if not active.any(): return` short-circuit (forced a CPU↔GPU sync per step). The branchless batched path is always taken.

### Migration

External hooks / tire models with the v0.4.x per-wheel API will break — update to the batched signature. No change needed for code that only uses the public `VehiclePhysics` / `VehicleInputs` / presets / strategy APIs.

---

## [0.4.4] — 2026-05-18

### Performance
- **Tank chase-cam fps fix** — interactive demo on the KDU tank ran at ~15 fps vs ~25 fps for the original `KDU/example_interactive.py`. Two contributions:
  - `VisualSync` was syncing wheel spin angle (`set_dofs_position` for the 10 continuous spin joints) every step. The original KDU intentionally skipped this since the cylinder primitive wheels are rotationally symmetric and a spinning visual is invisible.
  - The chassis `up_world` reference tensor was being re-allocated every step inside `VehiclePhysics.step()` instead of being cached at init time.

### Added
- `VehicleConfig.visual_spin_enabled: bool = True` (also on `ResolvedConfig`). Set `False` to skip the per-step spin-angle `set_dofs_position` call. Saves ~3-5 ms / step in interactive mode (one fewer Genesis call). Cars keep the default `True` so mesh wheels visibly roll; tanks (`tank_10w_skid_belt` preset) now default to `False`.

### Changed
- `VehiclePhysics.__init__` caches `_up_world` once instead of re-creating it each `step()`.
- `VisualSync` integrates the visual spin angle in place (`add_`) when enabled.

---

## [0.4.3] — 2026-05-18

### Fixed
- **Heavy-wheel suspension visual fix (two-part)** — on the KDU 10-wheel tank (each wheel ≈ 500 kg):
  - *Part 1 (free-fall)*: the wheel meshes were sinking below the ground while the chassis stayed up. `VisualSync` chose `set_dofs_position` vs `control_dofs_position` by detecting `<dynamics>` declarations in the URDF, which KDU's URDF omits entirely. With no PD control (`kp`/`kv` = 0), Genesis lets the heavy wheel free-fall between substeps and the kinematic `set_dofs_position` cannot snap it back fast enough.
  - *Part 2 (spinning in air)*: after the PD fix above, wheels stopped falling but were spinning in air ~5 cm above ground. The `control_dofs_position` path used `target = max(0, rest_d - d)` (a non-negative compression), which can only push the wheel UP from rest, never DOWN to reach the ground. This is fine when the carrier rest position sits *above* ground level (HJW URDF: susp origin z = 0.34), but breaks when the carrier rest sits *at* base_link z (KDU URDF: susp origin z = 0). The `control_dofs_position` path now uses the same `joint_pos = mesh_radius - d` formula as the `set_dofs_position` path so the wheel mesh lands on the ground in both cases.

### Added
- `VehicleConfig.visual_susp_mode: str = "auto"` (also on `ResolvedConfig`). Three values:
  - `"auto"` (default) — per-joint decision based on URDF `<dynamics>` presence; preserves the v0.4.2 behavior for HJW-style URDFs.
  - `"kinematic"` — force `set_dofs_position` everywhere (light wheels, HJW-style).
  - `"control"` — force `control_dofs_position` with PD kp=1e7 / kv=1e5 everywhere (heavy wheels, KDU-style).
  - `resolve()` validates the value and raises `ConfigError` on unknown strings.
- `tank_10w_skid_belt` preset now sets `visual_susp_mode="control"` so the wheels stay attached on the KDU tank without the user having to know.

### Migration
- HJW / JMK / car presets unchanged — `"auto"` keeps their existing behavior.
- If you constructed a `VehicleConfig` manually for a heavy-wheel vehicle and saw the wheel-sinking bug, set `visual_susp_mode="control"` explicitly.

---

## [0.4.2] — 2026-05-18

### Added
- `genesis_vehicle.scene_helpers` module with two thin Genesis-side
  convenience helpers:
  - `make_wheel_raycaster(scene, entity, urdf_path, ...)` — replaces the
    five-line `parse_urdf` + `WheelRayPattern` + `gs.sensors.Raycaster` +
    `scene.add_sensor` boilerplate with one call.
  - `add_vehicle(scene, urdf_path, preset_fn=None, ...)` — adds the URDF
    entity + raycaster (and optionally builds the preset `VehicleConfig`)
    in one call. Returns `(car, sensor, cfg)`. Caller still owns
    `scene.build()` and `VehiclePhysics(...)` construction so other
    entities / sensors can be added between.
- Both are lazy-imported through the top-level package, so
  `from genesis_vehicle import add_vehicle` works without Genesis
  installed at import time.

### Design note
The SDK intentionally does NOT fully encapsulate Genesis. Helpers cut the
canonical boilerplate for the common case; full-encapsulation facades for
CARLA-style or Gym-style use will live under `genesis_vehicle.adapters/`
when needed. This keeps the core API research-friendly (viewer, terrain,
custom materials, multi-agent scenes all stay accessible) while still
offering a one-liner for the common path.

---

## [0.4.1] — 2026-05-17

### Added
- `car_4w_fwd_ackermann` preset — 4-wheel front-wheel-drive car with front Ackermann steering. Fills the obvious gap between the existing `car_4w_rwd_ackermann` and `car_4w_awd_ackermann` presets (real-world passenger cars are mostly FWD). Mirrors the RWD preset's parameters with `FWD(driven_axles=(0,))`.

---

## [0.4.0] — 2026-05-17

### Changed (BREAKING)
- **Stability profile renamed**: `"physical"` → `"raw"`. No deprecated alias kept.
  - **Reason for break.** The original `"physical"` name implied "the physically correct config" or "the right profile for parameter fitting", which is incorrect. Fitted parameters in this SDK are always model-coupled (Pacejka `B` is the coefficient that fits real inside our SDK; it is not "rubber's true stiffness"). Hooks are part of the model. The right Real2Sim rule is *"fit with the same profile you'll deploy with"* — for our SDK that is almost always `"control"`. The renamed `"raw"` profile is now correctly framed: hook-free raw dynamics for **debugging / ablation**, not a general fitting recommendation.
  - **No alias kept.** Pre-1.0 versioning + essentially no external users; keeping the misleading name even as a deprecated alias would perpetuate the confusion the rename is meant to resolve.
- API.md S6 rewritten around "one rule: fitting forward model = deployment forward model", with `"raw"` reframed as an analysis tool rather than a Real2Sim recommendation.

### Migration
- Replace every `stability="physical"` with `stability="raw"` (or, more likely, with the default `"control"`, since most users were on `"physical"` because the doc misled them).
- If you were on `"physical"` for Real2Sim fitting and plan to deploy with `"control"`, **change fitting to `"control"`** so the forward models match.

---

## [0.3.0] — 2026-05-17

### Added
- **Stability profile** on every preset: `stability="control" | "physical" | "research"` (default `"control"`).
  - `"control"` — RL/MPPI-friendly hooks: `RollingResistance` + `LowSpeedRegularizer(disable_when_control_active=False)` (+ `StaticFrictionLock` for tank). Default for `car_4w_rwd_ackermann`, `car_4w_awd_ackermann`, `truck_6w_partial_ackermann`, `tank_10w_skid_belt`.
  - `"physical"` — empty hook list. Use for Real2Sim parameter identification so numerical stabilizers don't leak into fitted parameters.
  - `"research"` — empty hook list. Caller assembles hooks manually.
- `stability_hooks_for_profile(profile, vehicle_kind)` public helper to materialize the same hook lists used by the presets.
- Version API: `__version__`, `VERSION_INFO`, `version()`, `version_info()` exported from `genesis_vehicle`.
- One-time version banner printed on first `VehiclePhysics` initialization:
  ```
  [genesis_vehicle v0.3.0] Initialized: 4 wheels, Ackermann, RWD, Independent, n_envs=1, hooks=[RollingResistance, LowSpeedRegularizer]
  ```
- `CHANGELOG.md` (this file) tracking version history.

### Changed
- **Footgun fix** — `"control"` profile builds `LowSpeedRegularizer` with `disable_when_control_active=False`. The legacy default (`True`) made the regularizer a no-op exactly when MPPI was applying throttle, so the documented "low-speed stabilization" never fired under control. Direct construction (`LowSpeedRegularizer()`) keeps the legacy default for backwards compatibility.

### Notes
- The hook system is now treated as a stability profile rather than a free-form list assembled by the user. This aligns with industry vehicle SDKs (PhysX sticky tire mode, Jolt anti-rollback, Chaos low-speed threshold, Unity WheelCollider stick-slip) where low-speed numerical stabilization is part of the model, not an opt-in plug-in. Advanced users can still pass `stability="research"` and assemble `cfg.stability_hooks` manually.

---

## [0.2.0]

Reviewer-driven revisions of the v0.1.0 reference implementation. The SDK's
public contracts and helper boundaries were tightened ahead of the first
variant migration.

### Added
- `genesis_vehicle/dynamics.py` — pure-Python primitives:
  - `brake_torque_signed(t_brake, omega, smoothing_scale=0.5)` — positive brake magnitude → signed torque opposing wheel rotation.
  - `suspension_normal_force(compression, comp_rate, k_susp, c_compression, c_extension, air_mask)` — asymmetric damper + non-negative clamp + air-mask zero in one call.
- `VehicleStepInputs` typing alias on `VehiclePhysics.step()` (Union of `VehicleInputs` + every typed input).
- `estimate_spin_inertia_from_genesis(entity, link_name, spin_axis_local=None)` with optional axis projection (`a^T diag(I) a`).
- New tests:
  - `tests/test_dynamics.py` (8 tests) for `brake_torque_signed` and `suspension_normal_force`.
  - `tests/test_config_resolve.py::test_user_explicit_i_wheel_wins_over_urdf`, `test_user_explicit_radius_wins_over_urdf`.

### Changed
- **`i_wheel` truth policy** — `WheelConfig.i_wheel` set by the user (or populated by URDF) is now authoritative; the Genesis-runtime spin-inertia refinement only fires for wheels where neither the user nor the URDF supplied a value. Previously the Genesis estimate always overrode.
- **Inertia helper renamed**: `parse_inertia_max_principal_genesis` → `estimate_spin_inertia_from_genesis`. The old name is kept as a deprecated alias and will be removed in a later release.
- `core.py` refactored to call the new `dynamics.py` helpers instead of inlining the brake/suspension formulas.
- `__init__.py` exposes the full public surface (strategies, tire models, presets) as a single import path: `from genesis_vehicle import ...`.

### Documentation
- API.md restructured into 11 sections; new "Physics contracts" section codifies brake-sign, non-negative N, `i_wheel` policy, ISO 8855 steer convention, and coupling order. Migration cheatsheet now uses the "URDF default, API override wins" wording instead of the misleading "URDF as truth".

---

## [0.1.0]

Initial Phase 3 reference implementation of the genesis_vehicle SDK.

### Added
- Top-level `VehiclePhysics` driver running the shared 5-step ray-wheel pipeline batched over `n_envs ≥ 1`.
- Strategy abstractions: `SteeringStrategy` (`Ackermann`, `PartialAckermann`, `SkidSteer`, `NoSteer`), `DrivetrainStrategy` (`FWD`, `RWD`, `AWD`, `PerSide`), `CouplingStrategy` (`Independent`, `SameSideBelt`), `StabilityHook` (`RollingResistance`, `LowSpeedRegularizer`, `StaticFrictionLock`).
- Tire models: `PacejkaAnisotropic`, `CoulombIsotropic`.
- Config & merge: `WheelConfig`, `ChassisConfig`, `VehicleConfig`, `ResolvedConfig`, `resolve()`, `ConfigError`. URDF defaults + per-wheel overrides.
- Inputs: unified `VehicleInputs` + typed (`AckermannInputs`, `SkidSteerInputs`, etc.) with `from_unified()` adapters.
- `parse_urdf()` convention-based parser supporting HJW (`*_suspension_joint`) and KDU (`*_susp`) naming.
- Visual layer (`VisualSync`) absorbing URDF axis-sign quirks.
- Presets for four reference vehicles: `car_4w_rwd_ackermann`, `car_4w_awd_ackermann`, `truck_6w_partial_ackermann`, `tank_10w_skid_belt`.
- Pure-Python tests for URDF parsing, config resolve, strategy math.
