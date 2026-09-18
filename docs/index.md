# genesis_vehicle — Documentation

Landing page. Pick the document that matches what you're doing.

| If you want to... | Start here |
|---|---|
| Run something in 1 minute | [`quickstart.md`](quickstart.md) |
| See all 20 runnable samples (+ which have `--viewer`) | [`../samples/README.md`](../samples/README.md) |
| Build a mental model before reading code | [`concepts.md`](concepts.md) |
| Choose the high-level (`VehicleScene`) vs low-level (`VehiclePhysics`) API | [`concepts.md`](concepts.md#start-here-the-two-api-layers) |
| Pick the right batching axis (L1 / L2 / L3) for your workflow | [`batching.md`](batching.md) |
| Decide CPU vs GPU physics (measured crossover, `init_backend`, `--gpu`) | [`backends.md`](backends.md) |
| Choose a tire model (Pacejka vs Coulomb) and understand the ground-contact mechanism | [`tire-and-contact.md`](tire-and-contact.md) |
| Stop a wheel taking a whole kerb in one step (`wheel_contact="swept_envelope"`, v1.6.0) | [`tire-and-contact.md`](tire-and-contact.md#axis-a-upgrade--the-swept-envelope-wheel_contactswept_envelope-v160) |
| Drive from one unified `VehicleScene` object, and cut the wheel-raycast cost on heavy static terrain | [`dual-scene-raycast.md`](dual-scene-raycast.md) |
| Make a vehicle follow a waypoint path (path → Steer/Throttle) | [`path-following.md`](path-following.md) |
| Run the OSC physics server for an external client (Unreal / Unity), or look up the wire schema | [`server.md`](server.md) |
| Look up a specific class / function / default | [`api-reference.md`](api-reference.md) |
| Understand WHERE in the simulation step your hook runs | [`pipeline-and-hooks.md`](pipeline-and-hooks.md) |
| Pick a stability profile (or understand why it matters for MPPI / Real2Sim) | [`stability-profiles.md`](stability-profiles.md) |
| Verify what the SDK promises about brake sign / N clamp / `i_wheel` truth / steering convention | [`physics-contracts.md`](physics-contracts.md) |
| Author a vehicle URDF by type (naming/axes the parser needs, per-preset topology, where suspension comes from when the URDF is silent, `<dynamics>` overrides) | [`urdf-guide.md`](urdf-guide.md) |
| Author a URDF the ray-wheel model accepts (no wheel colliders, suspension origin = wheel centre, every link inertial) | [`physics-contracts.md` §7.9](physics-contracts.md#79-urdf-contracts-for-ray-wheels-auto-corrected-since-v1122) |
| Change a cfg AFTER `build()`, or reset a vehicle mid-rollout, without restarting its dynamics | [`physics-contracts.md` §7.12](physics-contracts.md) |
| See what's tested and where things live | [`testing.md`](testing.md) |
| Track what changed between versions | [`../CHANGELOG.md`](../CHANGELOG.md) |

## Single import surface

The SDK exposes everything through one import path:

```python
from genesis_vehicle import (
    # driver
    VehiclePhysics,
    # inputs (unified + typed)
    VehicleInputs, AckermannInputs, SkidSteerInputs, NoSteerInputs,
    # config
    VehicleConfig, WheelConfig, ChassisConfig,
    # strategies
    Ackermann, PartialAckermann, SkidSteer, NoSteer,
    FWD, RWD, AWD, PerSide,
    Independent, SameSideBelt,
    RollingResistance, LowSpeedRegularizer, StaticFrictionLock,
    # tire models
    PacejkaAnisotropic, CoulombIsotropic,
    # presets
    car_4w_rwd_ackermann, car_4w_awd_ackermann,
    truck_6w_partial_ackermann, tank_skid_belt,
    # utilities
    WheelRayPattern, parse_urdf, stability_hooks_for_profile,
    # path following (docs/path-following.md)
    PathFollower, FleetFollower, DifferentiablePlant, SweepTable,
    # per-link transforms (telemetry / animation / attach)
    get_link_transforms, LinkTransforms,
    # version
    __version__, version, version_info,
)
```

See [`api-reference.md`](api-reference.md) for the full surface.

## Built-in utilities

What ships with the SDK beyond the core physics drivers, one line each.
The right-hand column is where the full story lives.

**Control & workflow**

| Utility | What it does | Entry point | Details |
|---|---|---|---|
| Path following | waypoints + signed target speeds → per-step `(throttle, steer, brake)` by inverting the vehicle's response map; cusps (speed-sign flips) handled as stop-and-reverse | `PathFollower` | [`path-following.md`](path-following.md) |
| Inverse plant (default) | autodiff through the SDK's own ray-wheel force model → 2×2 Jacobian → Newton solve, every step. No measurement, no CSV, tracks plant edits automatically | `DifferentiablePlant` | [`path-following.md`](path-following.md) §1 |
| Fleet control | one plant inverts K vehicles × N envs in ONE unrolled batch (26.5× faster than a plant each, at 30 tanks); `plan` → batched `solve` → `finish` | `FleetFollower` | [`path-following.md`](path-following.md) §1 |
| Inverse plant (measured) | a pre-measured (v, throttle, steer, pitch, roll) → (a, ω_z) grid; numpy-only, so a simulator-free process can still path-follow | `SweepTable` | [`path-following.md`](path-following.md) §2 |
| Sweep measurement | measures that grid in one batched L3 run (build-once, body-frame accurate) — only needed for `SweepTable` | `python -m genesis_vehicle.control.sweep_measure` | [`path-following.md`](path-following.md) §2 |
| Stability profiles | maps a use-case profile (`"control"` / `"raw"` / `"research"`) to the right stability-hook stack | `stability_hooks_for_profile` | [`stability-profiles.md`](stability-profiles.md) |
| OSC physics server | runs the SDK as a standalone physics process for an external engine (UE / Unity), L2 and L3 modes | `python -m genesis_vehicle.server` | [`server.md`](server.md) |
| Server benchmark | official server perf matrix (mock UE client, tanks × terrain × mode × backend) | `python -m genesis_vehicle.server.benchmark` | [`server.md`](server.md) §2.1 |
| Server serving counters | L2-only per-window `[SERVE]` line (`recv_loops` / `nonskip_loops` / `skipped_captures` / `post_step_captures` / `post_step_captures_ref`, plus `sends` / `send_skips` / `send_flushes` / `window_ms` since v1.6.7) plus the post-override `capture_state` skip gate and its `--legacy-override-capture` rollback. No speed figure is published for the skip (v1.6.6) | `python -m genesis_vehicle.server --serve-timers`, `python -m genesis_vehicle.server.benchmark --kinds N --input-hz H` | [`server.md`](server.md) §2.5 |
| Server state-send downsampling | `--send-hz H` gates the L2 state send to H Hz on a wall clock while physics keeps running at `dt`; default OFF and bit-identical when off. A reset is flushed with the exact reset pose, a teleport is not (delayed ≤ `1/H` + one loop interval, ≤ 75 ms at H=20 / dt=25 ms). The cost is LAG, quoted by regime; the measured arrival rate drops 136.4 → 20.0 Hz, and NO speed figure is published (v1.6.7). L2 only — `--multi-env` WARNs | `python -m genesis_vehicle.server --send-hz 20`, `python -m genesis_vehicle.server.benchmark --send-hz 20` | [`server.md`](server.md) §2.6 |
| Raycast-mode benchmark | order-independent, paired, fail-closed comparison of `dual_scene` vs `single_scene` (fresh process per measurement; prints NO ratio when the machine's noise cannot carry one) | `python -m genesis_vehicle.samples.bench_raycast_mode` | [`dual-scene-raycast.md`](dual-scene-raycast.md) |
| Samples | 20 runnable, self-contained example programs with bundled assets (every `samples/*.py` with a `__main__` entry, minus the `_hud.py` / `tank_tuning.py` helpers and the deprecated `two_scene_terrain` alias) | `python -m genesis_vehicle.samples.<name>` | [`../samples/README.md`](../samples/README.md) |

**Model preparation & scene assembly**

| Utility | What it does | Entry point | Details |
|---|---|---|---|
| URDF parsing | discovers wheels/axle order, steer-axis sign, and per-wheel geometry from any URDF; the default source of `WheelConfig` | `parse_urdf`, `estimate_spin_inertia_from_genesis` | [`api-reference.md`](api-reference.md) §4 |
| URDF auto-repair | makes an arbitrary URDF ray-wheel ready: wheel colliders become render-only, a suspension origin sitting off the wheel centre is folded back onto it, missing `<inertial>`s injected. Runs automatically inside `add_vehicle` and the OSC server; the original file is never modified | `prepare_vehicle_urdf` | [`physics-contracts.md` §7.9](physics-contracts.md#79-urdf-contracts-for-ray-wheels-auto-corrected-since-v1122) |
| Scene helpers | one-call vehicle + wheel-raycaster insertion for a raw `gs.Scene` (the low-level path; prefer `VehicleScene`) | `add_vehicle`, `make_wheel_raycaster` | [`api-reference.md`](api-reference.md) §0/§9 |
| Raycast helpers | wheel ray pattern definition and shape-normalized sensor reads — `read_distances` returns `(n_envs, n_wheels)` for 1 and N envs AND for a swept-envelope fan, whose raw read is `(n_envs, n_wheels, M)` (v1.6.0) | `WheelRayPattern`, `read_distances`, `fan_height_offsets` | [`tire-and-contact.md`](tire-and-contact.md) |
| Wheel contact model | `wheel_contact="point"` (DEFAULT, one ray per wheel) or `"swept_envelope"` (M rays reduced by the swept-circle lower envelope, so a wheel climbs an edge instead of teleporting onto it). Opt-in; the default is bit-identical to earlier releases | `VehicleScene.add_vehicle(wheel_contact=, contact_samples=)`, `WheelRayPattern(fan_samples=)` | [`tire-and-contact.md`](tire-and-contact.md), [`physics-contracts.md` §7.11](physics-contracts.md) |
| Ray-MISS / grounded | "did this wheel's ray find ground?", against the raycaster's OWN miss sentinel rather than a fixed threshold | `Vehicle.wheels_grounded`, `all_wheels_grounded`, `MultiVehiclePhysics.grounded_list`, `is_ray_hit`, `is_ray_hit_corrected` | [`physics-contracts.md`](physics-contracts.md) §7.8, §7.10 |
| Live vs derived config | which post-`build()` config writes take effect on the next step (steering geometry, driven axles, brake bias, AWD weights, the drive-omega cap, `eps_v`, a hook's `v_thr`) and which need a rebuild (anything per-wheel, read off `WheelMeta`). Derived once at build, re-derived when a source moves (v1.6.4) | `_hotset.HOT_DEPENDENTS`, `VehicleScene.mark_config_dirty` | [`physics-contracts.md` §7.13](physics-contracts.md) |
| Simulation time / gravity | `dt` and `substeps` are FIXED at `build()` on genesis 1.4.0 (the supported backend) and on 1.3.3 alike — post-build `sim_options` writes are inert (and raise on genesis 1.4.0, where `Scene.sim_options` is gone). READ the engine, do not echo the ctor args; `set_gravity` is the one live knob (v1.6.5) | `VehicleScene.effective_dt`, `VehicleScene.substeps`, `VehicleScene.set_gravity` | [`physics-contracts.md` §7.14](physics-contracts.md) |
| Config rebuild / reset | re-resolve a cfg after `build()` while the vehicle keeps its runtime state, and reset per vehicle rather than per flat row. A wheel-count or wheel-order change raises; a failed rebuild keeps the previous driver and re-raises on the next step (v1.6.2) | `VehicleScene.mark_config_dirty`, `VehicleScene.reset`, `MultiVehiclePhysics.reset(vehicle_ids=)`, `MultiVehicleKindPhysics.reset(rows=)` | [`physics-contracts.md` §7.12](physics-contracts.md) |

**Telemetry & rendering feed**

| Utility | What it does | Entry point | Details |
|---|---|---|---|
| Per-link transforms | batched world pose of every link (telemetry, attachments, external animation) | `get_link_transforms`, `LinkTransforms` | [`api-reference.md`](api-reference.md) §7 |
| Render feed | closed-form wheel / visual-part poses (~µs, no engine FK) for an external renderer; what the OSC server streams | `wheel_visual_transforms()`, `VisualPartsTransforms` | [`api-reference.md`](api-reference.md) §7, [`server.md`](server.md) |
| Viewer wheel visuals | instanced wheel rendering in the Genesis viewer/cameras — auto-managed by `VehicleScene` at `build()`; always on when rendering. Wheels are NOT driven through the rigid solver (external render nodes, closed-form poses) — physics-identical to headless, slight pose-streaming cost (~2–3 ms @ 30 vehicles) | `VehicleScene(wheel_render_mode=...)` | [`../samples/README.md`](../samples/README.md) |

## Version

Current SDK version is `genesis_vehicle.__version__`. `VehiclePhysics.__init__`
prints a one-line banner with the version on first construction in a process:

```
[genesis_vehicle v0.6.0] Initialized: 4 wheels, Ackermann, RWD, Independent, n_envs=1, hooks=[RollingResistance, LowSpeedRegularizer]
```

Release history lives in
[`../CHANGELOG.md`](../CHANGELOG.md).

**Backend compatibility:** the supported Genesis physics backend is
`genesis-world` **1.4.0** — the floor and the only version the tests and every
published measurement run on (v1.6.6; the `≥ 1.0.0` claim printed here through
v1.6.4 was never enforced, and v1.6.5's "1.3.3 and 1.4.0" is now narrowed).
**Nothing in the code enforces that floor**: the `<= 1.3.3` branches in
`genesis_vehicle/_gs_compat.py` (force/torque -> wrench, link inertial
accessors) and in `vehicle_scene.py` (`RigidOptions.dt`, terrain raycast
mirror) still run if 1.3.3 is installed, and there is no version check. They
are dead weight slated for removal, not a compatibility promise — untested,
unmeasured, and closed to additions. The instanced-wheel renderer likewise
branches on the 1.3.x buffer API — see the 1.2.7 CHANGELOG entry. Per-release
backend notes are in [`../CHANGELOG.md`](../CHANGELOG.md) (the 1.5.0 entry
records the 1.3.3 → 1.4.0 bump and a cross-version parity table measured ONCE
at that bump — a historical record, not a standing guarantee — and 0.5.33 the
0.4.6 → 1.0.0 one).

**On `genesis-world >= 1.4.0`, never pass `dt=` to `RigidOptions`** — that
field became the solver SUBSTEP interval, so restating `SimOptions.dt` there
now contradicts `substeps` and `Scene.build` raises. Ask for the rate with
`VehicleScene(substeps=...)` instead; `VehicleScene` strips the legacy
spelling and warns once.
