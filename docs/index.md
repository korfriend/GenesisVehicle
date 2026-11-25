# genesis_vehicle — Documentation

Landing page. Pick the document that matches what you're doing.

| If you want to... | Start here |
|---|---|
| Run something in 1 minute | [`quickstart.md`](quickstart.md) |
| See all 22 runnable samples (+ which have `--viewer`) | [`../samples/README.md`](../samples/README.md) |
| Build a mental model before reading code | [`concepts.md`](concepts.md) |
| Choose the high-level (`VehicleScene`) vs low-level (`VehiclePhysics`) API | [`concepts.md`](concepts.md#start-here-the-two-api-layers) |
| Pick the right batching axis (L1 / L2 / L3) for your workflow; give obstacles / ray surfaces a different variant per env (v2.0.5) | [`batching.md`](batching.md) |
| Decide CPU vs GPU physics (measured crossover, `init_backend`, `--gpu`) | [`backends.md`](backends.md) |
| Choose a tire model (Pacejka vs Coulomb) and understand the ground-contact mechanism | [`tire-and-contact.md`](tire-and-contact.md) |
| Stop a wheel taking a whole kerb in one step (`wheel_contact="swept_envelope"`, v1.6.0) | [`tire-and-contact.md`](tire-and-contact.md#axis-a-upgrade--the-swept-envelope-wheel_contactswept_envelope-v160) |
| Drive from one unified `VehicleScene` object, and keep wheel rays off the vehicle itself (`dual_scene`; through v2.0.4 this row promised to "cut the wheel-raycast cost on heavy static terrain" — a pre-1.5.0 mechanism, false on genesis-world 1.5.0) | [`dual-scene-raycast.md`](dual-scene-raycast.md) |
| Make a vehicle follow a waypoint path (path → Steer/Throttle) | [`path-following.md`](path-following.md) — **2.0.0: `PathFollower(path, plant)` raises unless you pass `mpc_model="off"` (the 1.x follower) or `mpc_params=` (the MPC layer); see "Which follower" and §6 "Migrating from 1.x". 2.0.2: the MPC follower returns a `GearCommand` by default (§5.9; `command_mode="signed"` for the 2.0.1 3-tuple). 2.0.3: opt-in K-turn, `mpc_kturn=KTurnParams()` (§5.10; off by default — the default follower issues the 2.0.2 commands). 2.1.0: opt-in end-gated brake bound, `mpc_decel_limit="brake"` (§5.11; the brake row on end-gated solves only — reference time, the last 2 s of a forward block's reference schedule, with the rig reference end ahead of the rear axle — so a vehicle behind its schedule gets it earlier and farther from the end; equality with the default before the first open solve measured, not guaranteed; the default's commands and per-step logs identical to 2.0.6 on the measured runs (I-1, I-2, I-6 of CHANGELOG 2.1.0 §4), `mpc_stats` gains `decel_gate: None`)** |
| Let the MPC follower brake harder for the last 2 s of a forward block's reference schedule (opt-in `mpc_decel_limit="brake"`, v2.1.0; gated in reference time, so a vehicle behind its schedule gets it earlier and farther from the end; equality with the default before the first open solve measured, not guaranteed; at low grip it can lock the front wheels, and a record that over-states the grip is not detected) | [`path-following.md` §5.11](path-following.md), [`physics-contracts.md` §7.24](physics-contracts.md) |
| Drive a vehicle with a forward/reverse gear and brake-only deceleration (opt-in gear mode, v2.0.2) | [`physics-contracts.md` §7.20](physics-contracts.md), [`path-following.md` §5.9](path-following.md) |
| Let the path follower reverse by itself when a forward-only turn will not fit (opt-in K-turn, v2.0.3; one documented acceptance FAIL) | [`path-following.md` §5.10](path-following.md), [`physics-contracts.md` §7.21](physics-contracts.md) |
| Know when a braked vehicle holds on a slope, what `StaticFrictionLock`'s spring cap enforces, and why a tank at full brake pivots under steer (v2.0.4) | [`physics-contracts.md` §7.22](physics-contracts.md), CHANGELOG 2.0.4 |
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
    car_4w_rwd_ackermann, car_4w_fwd_ackermann, car_4w_awd_ackermann,
    truck_6w_partial_ackermann, tank_skid_belt,
    # utilities
    WheelRayPattern, parse_urdf, stability_hooks_for_profile,
    # path following (docs/path-following.md)
    PathFollower, FleetFollower, DifferentiablePlant, SweepTable,
    # gear mode (v2.0.2; physics-contracts.md §7.20)
    GearCommand, GEAR_FORWARD, GEAR_REVERSE, GEAR_SHIFT_MAX_SPEED,
    # (the MPC layer's MPCVehicleParams / MPCGains / load_mpc_params /
    #  identify_mpc_params / FollowerPlan / extract_state and the v2.0.3
    #  KTurnParams live in genesis_vehicle.control, not the root)
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
| Path following | waypoints + signed target speeds → per-step `(throttle, steer, brake)` by inverting the vehicle's response map; cusps (speed-sign flips) handled as stop-and-reverse. Since 2.0.0 stage 1 is an MPC layer by default (needs `mpc_params`); `mpc_model="off"` is the 1.x follower. Since 2.0.2 the MPC follower defaults to `command_mode="gear"` and returns a `GearCommand` (decelerates with the brake; `SHIFT` / `GUARD` / terminal `GEAR-ABORT`) — a breaking return-type change in a patch release, by user decision. Since 2.0.3, opt-in (`mpc_kturn=KTurnParams()`), a gear-mode MPC follower can stop and drive a multi-point turn with reverse legs when its trigger predicts the forward turn will not fit (collision-unaware; one measured acceptance FAIL, CTE 16.007 m vs 11.5 m on one course without gear feedback; legs planned at 1.2 · R_min but driven at full lock). Since 2.1.0, opt-in (`mpc_decel_limit="brake"`), the MPC is bounded by the record's brake-1 row instead of its throttle row on end-gated solves only (reference time: within the last 2.0 s of a forward block's reference SCHEDULE, with the rig reference end ahead of the rear axle; see `PathFollower.mpc_decel_limit`) — mid-path decelerations get nothing, a vehicle behind its schedule gets it earlier and farther from the end, and equality with the default before the first open solve is measured, not guaranteed; no gate sees front-wheel lock or understeer under it | `PathFollower`, `GearCommand`, `control.KTurnParams` | [`path-following.md`](path-following.md) §5.9, §5.10, §5.11 |
| Gear mode (v2.0.2, opt-in) | `set_inputs(..., gear=±1)`: the throttle is a magnitude in [0, 1] applied in the ENGAGED gear's direction; every wheel's drive torque is clamped to that gear after the shift decision; the gear changes only at throttle 0 and planar speed < 0.5 m/s (deferred otherwise); skid-steer vehicles refuse it; `gear=None` is the 2.0.1 signed input (bit-identical to 2.0.1 on the measured scenes through 2.0.3; since 2.0.4 the braking physics changed for every input, so braked scenes no longer reproduce 2.0.1 numbers); not on the OSC wire | `Vehicle.set_inputs(gear=)`, `Vehicle.gear`, `MultiVehiclePhysics.gear_list()`, `GEAR_SHIFT_MAX_SPEED` | [`physics-contracts.md` §7.20](physics-contracts.md) |
| MPC layer (v2.0.0; gate, watchdog, stops v2.0.1; end-gated brake bound v2.1.0) | Model Predictive Control: the field `PathFollower_MPC` controller (AlpaSim-derived bicycle model — the only prediction model; a Pacejka model was tried for 2.1.0 and removed after it failed its gates; do_mpc + IPOPT, or OSQP) on forward blocks of a path, the 1.x proportional path law (PD) on reverse blocks and after a failed solve; two-axle Ackermann only; optional packages (`requirements-mpc.txt`). The nonlinear solve is NOT guaranteed to fit a 25 ms step (5-24 % of steps over on the measured machine, session-dependent); on 2.0.0 a follower started facing away from its path could stall (never move) with no warning — this row said "fix planned for 2.1.0"; superseded: since v2.0.1 a heading gate and a no-progress watchdog hand such steps to the PD law (on by default), per-forward-block budgets end the PD hand-overs with `MPC-ABORT`, and a goal-passed stop (`GOAL-MISSED`, every mode) stops a vehicle on the final block whose projection is on the last segment, that is past the goal's normal plane, at least `arrival_goal + 0.1` m away, and whose distance to the goal grew by more than 1e-3 m on each of the last two `plan()` calls (a call count, not a distance or time bound); loop on `PathFollower.TERMINAL_MODES`. The follower is NOT guaranteed not to run away (§5.3, §5.8) | `PathFollower(mpc_params=...)`, `python -m genesis_vehicle.control.identify`, `python -m genesis_vehicle.samples.bench_path_follower` | [`path-following.md`](path-following.md) §5 |
| Inverse plant (no measurement) | autodiff through the SDK's own ray-wheel force model → 2×2 Jacobian → Newton solve, every step. No measurement, no CSV, tracks plant edits automatically | `DifferentiablePlant` | [`path-following.md`](path-following.md) §1 |
| Fleet control | one plant inverts K vehicles × N envs in ONE unrolled batch (26.5× faster than a plant each, at 30 tanks); `plan` → batched `solve` → `finish`; PD followers only (`mpc_model="off"`, v2.0.0) | `FleetFollower` | [`path-following.md`](path-following.md) §1 |
| Inverse plant (measured) | a pre-measured (v, throttle, steer, pitch, roll) → (a, ω_z) grid; numpy-only, so a simulator-free process can still path-follow | `SweepTable` | [`path-following.md`](path-following.md) §2 |
| Sweep measurement | measures that grid in one batched L3 run (build-once, body-frame accurate) — only needed for `SweepTable`; v2.0.0 grid flags (`--v-list=...`) for MPC speeds above the default 4 m/s | `python -m genesis_vehicle.control.sweep_measure` | [`path-following.md`](path-following.md) §2 |
| Stability profiles | maps a use-case profile (`"control"` / `"raw"` / `"research"`) to the right stability-hook stack | `stability_hooks_for_profile` | [`stability-profiles.md`](stability-profiles.md) |
| Rolling resistance | an axle resisting torque `cr*N*r*tanh(ω r / tanh_scale)` (cr = rolling-resistance coefficient, N = wheel load, r = wheel radius) applied to wheel spin ω before the tyre, through a hook's `apply_pre_tire` and `PipelineContext.T_resist`; part of the default `"control"` profile, and in effect only since v1.7.0 (the overshoot clamp erased it before) | `RollingResistance`, `StabilityHook.apply_pre_tire` | [`physics-contracts.md` §7.15](physics-contracts.md) |
| Brake, axle-torque order, low-speed bounds | since v1.8.0 the drive and brake torques act on the wheel spin before the tyre; the brake is a Coulomb projection that holds a stopped WHEEL; downhill and both-pedal stops hold the VEHICLE where `StaticFrictionLock` takes over — **since v2.0.4 on every brake-held wheel at any pedal** (reference car, 5°, brake 0.3: rests at 4e-6 m/s; 2.0.3 crept at 0.02497 m/s), with its spring capped per wheel; a wheel whose drive torque exceeds its brake torque turns (brake does not override drive). Through 2.0.3 the lock engaged only at brake > `brake_thr` (0.3); this row then said the chassis "slides on tyre slip" below it — wrong mechanism (the held-wheel deadbeat is a damper, so the vehicle crept at `g·tan θ·dt/s` — g gravity, θ slope angle, s the deadbeat's rigid-body scale, `physics-contracts.md` §7.16); and through v2.0.1 it said "(downhill and both-pedal stops hold)" without the condition — incomplete. `F_lat` is taken at the transmitted `F_long`; held-wheel and lateral one-step deadbeats, scaled by the vehicle's yaw inertia, remove the low-speed lateral 2-cycle; `StaticFrictionLock` holds the takeover point (through v2.0.5 "the point where the contact came to rest", via a reversal re-anchor removed in v2.0.6 because its branch was decided by round-off) and, since v2.0.6, engages only on a wheel with `ctx.N > 0` (ground gate) | `dynamics.brake_projection`, `dynamics.deadbeat_scale`, `TireModel.at_long_force`, `StaticFrictionLock.STATE_ATTRS`, `PipelineContext.brake_rem`, `dynamics.static_mass_shares` / `rigid_share_factor` | [`physics-contracts.md`](physics-contracts.md) §7.1, §7.16, §7.17, §7.22 |
| OSC physics server | runs the SDK as a standalone physics process for an external engine (UE / Unity), L2 and L3 modes. `--no-target-collision` works for primitive targets and is refused (rc 2) with URDF vehicles or `--multi-env` since v2.0.5 (§2.8) | `python -m genesis_vehicle.server` | [`server.md`](server.md) |
| Server benchmark | official server perf matrix (mock UE client, tanks × terrain × mode × backend) | `python -m genesis_vehicle.server.benchmark` | [`server.md`](server.md) §2.1 |
| Server serving counters | per-window `[SERVE]` line: L2 carries `recv_loops` / `nonskip_loops` / `skipped_captures` / `post_step_captures` / `post_step_captures_ref` (v1.6.6) + `sends` / `send_skips` / `send_flushes` (v1.6.7) + `loops` / `steps` / `zero_step_loops` / `serve_ms` / `phys_ms_sum` (v1.6.8), and `window_ms`; **L3 gained a bare `[SERVE] [L3]` line with the six window-shape keys in v1.6.8** (the L2-only counters are ABSENT from it, not zero). Plus the post-override `capture_state` skip gate and its `--legacy-override-capture` rollback. No speed figure is published for the skip (v1.6.6) | `python -m genesis_vehicle.server --serve-timers`, `python -m genesis_vehicle.server.benchmark --kinds N --input-hz H` | [`server.md`](server.md) §2.5 |
| Server state-send downsampling | `--send-hz H` gates the L2 state send to H Hz on a wall clock while physics keeps running at `dt`; default OFF and bit-identical when off. A reset is flushed with the exact reset pose, a teleport is not (delayed ≤ `1/H` + one loop interval, ≤ 75 ms at H=20 / dt=25 ms). The cost is LAG, quoted by regime; the measured arrival rate drops 136.4 → 20.0 Hz, and NO speed figure is published (v1.6.7). L2 only — `--multi-env` WARNs | `python -m genesis_vehicle.server --send-hz 20`, `python -m genesis_vehicle.server.benchmark --send-hz 20` | [`server.md`](server.md) §2.6 |
| Server reporting window | the `[STATS]`/`[SERVE]` window is `--stats-interval` seconds of WALL CLOCK in both modes (default 1.0; a fixed 50 LOOPS through v1.6.7, which collapsed to a median 0.0–0.1 ms on a server with headroom). The publishable metric is the duty cycle `D = Σserve_ms / Σwindow_ms`, plus `S' = Σserve_ms / Σsteps`; `Loop Avg` / `steps/loop` are still diluted by the busy-wait loops and that is NOT fixed. The benchmark's headline columns are now pooled over `loops`, so pre-v1.6.8 tables are not directly comparable (v1.6.8) | `python -m genesis_vehicle.server --stats-interval 1.0`, `python -m genesis_vehicle.server.benchmark --stats-interval 2.0` | [`server.md`](server.md) §2.5 |
| Raycast-mode benchmark | order-independent, paired, fail-closed comparison of `dual_scene` vs `single_scene` (fresh process per measurement; prints NO ratio when the machine's noise cannot carry one). On genesis-world 1.5.0 one run published single/dual 0.753 [0.648 .. 0.788] n=6 (`single_scene` faster) with caveats; no speed advantage is claimed, and the `dual_scene` default rests on its ray scene holding no vehicle geometry, not on speed (v2.0.5) | `python -m genesis_vehicle.samples.bench_raycast_mode` | [`dual-scene-raycast.md`](dual-scene-raycast.md) |
| Samples | 22 runnable, self-contained example programs with bundled assets (every `samples/*.py` with a `__main__` entry, minus the `_hud.py` / `tank_tuning.py` helpers and the deprecated `two_scene_terrain` alias) | `python -m genesis_vehicle.samples.<name>` | [`../samples/README.md`](../samples/README.md) |

**Model preparation & scene assembly**

| Utility | What it does | Entry point | Details |
|---|---|---|---|
| URDF parsing | discovers wheels/axle order, steer-axis sign, and per-wheel geometry from any URDF; the default source of `WheelConfig` | `parse_urdf`, `estimate_spin_inertia_from_genesis` | [`api-reference.md`](api-reference.md) §4 |
| URDF auto-repair | makes an arbitrary URDF ray-wheel ready: wheel colliders become render-only, a suspension origin sitting off the wheel centre is folded back onto it, missing `<inertial>`s injected. Runs automatically inside `add_vehicle` and the OSC server; the original file is never modified | `prepare_vehicle_urdf` | [`physics-contracts.md` §7.9](physics-contracts.md#79-urdf-contracts-for-ray-wheels-auto-corrected-since-v1122) |
| Scene helpers | one-call vehicle + wheel-raycaster insertion for a raw `gs.Scene` (the low-level path; prefer `VehicleScene`) | `add_vehicle`, `make_wheel_raycaster` | [`api-reference.md`](api-reference.md) §0/§9 |
| Raycast helpers | wheel ray pattern definition and shape-normalized sensor reads — `read_distances` returns `(n_envs, n_wheels)` for 1 and N envs AND for a swept-envelope fan, whose raw read is `(n_envs, n_wheels, M)` (v1.6.0) | `WheelRayPattern`, `read_distances`, `fan_height_offsets` | [`tire-and-contact.md`](tire-and-contact.md) |
| Wheel contact model | `wheel_contact="point"` (DEFAULT, one ray per wheel) or `"swept_envelope"` (M rays reduced by the swept-circle lower envelope, so a wheel climbs an edge instead of teleporting onto it). Opt-in; the default's contact is bit-identical to earlier releases only where no axle torque, brake or low-speed bound acts (a v1.8.0 witness rollout against the v1.5.0 tree); v1.7.0 moved the default vehicle (rolling resistance) and v1.8.0 the driving rollout itself | `VehicleScene.add_vehicle(wheel_contact=, contact_samples=)`, `WheelRayPattern(fan_samples=)` | [`tire-and-contact.md`](tire-and-contact.md), [`physics-contracts.md` §7.11](physics-contracts.md) |
| Ray-MISS / grounded | "did this wheel's ray find ground?", against the raycaster's OWN miss sentinel rather than a fixed threshold | `Vehicle.wheels_grounded`, `all_wheels_grounded`, `MultiVehiclePhysics.grounded_list`, `is_ray_hit`, `is_ray_hit_corrected` | [`physics-contracts.md`](physics-contracts.md) §7.8, §7.10 |
| Live vs derived config | which post-`build()` config writes take effect on the next step (steering geometry, driven axles, brake bias, AWD weights, the drive-omega cap, `eps_v`, a hook's `v_thr`, and since v2.0.4 `StaticFrictionLock`'s capped `k_spring`) and which need a rebuild (anything per-wheel, read off `WheelMeta`). Derived once at build, re-derived when a source moves (v1.6.4) | `_hotset.HOT_DEPENDENTS`, `VehicleScene.mark_config_dirty` | [`physics-contracts.md` §7.13](physics-contracts.md) |
| Simulation time / gravity | `dt` and `substeps` are FIXED at `build()` (measured on genesis 1.4.0 at v1.6.5; the test pinning it passes on 1.5.0) — post-build `sim_options` writes are inert (and `Scene.sim_options` is gone since genesis 1.4.0). READ the engine, do not echo the ctor args; `set_gravity` is the one live knob (v1.6.5) | `VehicleScene.effective_dt`, `VehicleScene.substeps`, `VehicleScene.set_gravity` | [`physics-contracts.md` §7.14](physics-contracts.md) |
| Config rebuild / reset | re-resolve a cfg after `build()` while the vehicle keeps its runtime state, and reset per vehicle rather than per flat row. A wheel-count or wheel-order change raises; a failed rebuild keeps the previous driver and re-raises on the next step (v1.6.2) | `VehicleScene.mark_config_dirty`, `VehicleScene.reset`, `MultiVehiclePhysics.reset(vehicle_ids=)`, `MultiVehicleKindPhysics.reset(rows=)` | [`physics-contracts.md` §7.12](physics-contracts.md) |

**Telemetry & rendering feed**

| Utility | What it does | Entry point | Details |
|---|---|---|---|
| Per-link transforms | batched world pose of every link (telemetry, attachments, external animation) | `get_link_transforms`, `LinkTransforms` | [`api-reference.md`](api-reference.md) §7 |
| Render feed | closed-form wheel / visual-part poses (~µs, no engine FK) for an external renderer; what the OSC server streams | `wheel_visual_transforms()`, `VisualPartsTransforms` | [`api-reference.md`](api-reference.md) §7, [`server.md`](server.md) |
| Viewer wheel visuals | since v2.0.6, at every `n_envs`: `VehicleScene` overrides the poses of the engine's own wheel render nodes with the closed-form pose on every sync of the main scene's rasterizer context (SDK fallback nodes where the engine draws none, e.g. `vis_mode="collision"`); no SDK code writes solver state for rendering, and rendered runs were measured bit-identical to headless on the rows in CHANGELOG 2.0.6 §6; wheel pixels carry their wheel link's segmentation id. Not hooked: camera sensors, visual raycasting / picking, viewer frames during `build()` (URDF wheel at the solver pose, no warning). Through v2.0.5 the `n_envs > 1` / `internal_sync` fallback `WheelJointInternalSync` moved the chassis trajectory (+0.142 m over 10.7 m on the reference car); it was removed in v2.0.6 | `VehicleScene(wheel_render_mode=...)` | [`physics-contracts.md` §7.8](physics-contracts.md), [`../samples/README.md`](../samples/README.md) |

## Version

Current SDK version is `genesis_vehicle.__version__`. `VehiclePhysics.__init__`
prints a one-line banner with the version on first construction in a process:

```
[genesis_vehicle v0.6.0] Initialized: 4 wheels, Ackermann, RWD, Independent, n_envs=1, hooks=[RollingResistance, LowSpeedRegularizer]
```

Release history lives in
[`../CHANGELOG.md`](../CHANGELOG.md).

**Backend compatibility:** since v2.0.5 the SDK REQUIRES `genesis-world`
**1.5.0** (`requirements.txt`: `>=1.5.0,<1.6`), with `quadrants` 1.4.0,
`coacd` >= 1.0.14, `PyGEL3D` >= 0.8.0 and Python >= 3.12; for 1.4.5 stay on
v2.0.4. The v2.0.5, v2.0.6 and v2.1.0 tests run on 1.5.0; **measurements published before v2.0.5
were taken on 1.4.5 (v1.6.9-v2.0.4) or 1.4.0 (earlier) and keep those
labels** — they were not re-run unless the 2.0.5 CHANGELOG says so.
**The floor is a refusal at import** (through v2.0.4 this paragraph described
the warn-only `check_genesis_floor`, 1.4.5, removed in v2.0.5):
`_gs_compat.require_genesis` runs right after `import genesis` in `core`,
`vehicle_scene`, `multi_vehicle`, `raycast`, `scene_helpers`, `visual` and
`server`, and raises `GenesisVersionError` (an `ImportError`) below 1.5.0,
logging the message once at ERROR first. Both names are exported from the
package root since v2.0.5 (`from genesis_vehicle import GenesisVersionError,
require_genesis`; signatures in [`api-reference.md`](api-reference.md) §7,
"Required engine"). `kinematics`, `urdf` and
`control.sweep_measure` import without it, and `import genesis_vehicle`,
`PathFollower` / `SweepTable` do not need it — so it is not "every module that
touches genesis". >= 1.6 warns once and continues; a 1.5.x patch release is
silent and NOT validated (the 20 engine-keyed identity locks skip on it). Every
`<= 1.3.3` and 1.4.x compatibility branch (force/torque -> wrench, link
inertial accessors, `Scene.sim_options`, the terrain-mirror probe, the
renderer buffer API) was removed in v2.0.5; the legacy `RigidOptions.dt`
normalisation (`vehicle_scene._normalize_rigid_dt`) remains. `_gs_compat.py`
keeps one 1.5.0 code path per engine-private read. Per-release backend notes are in
[`../CHANGELOG.md`](../CHANGELOG.md) (SDK v2.0.5 records the engine 1.4.5 → 1.5.0 bump —
what the engine changed, the parity evidence and its limits, the field-team
upgrade order and the rollback; SDK v1.6.9 the engine 1.4.0 → 1.4.5 bump — the eager
signed-distance-field (SDF) build, the coarse-SDF material on SDK-owned bodies,
the measured hibernation delay, also [`server.md`](server.md) §2.7; SDK v1.5.0
for the engine 1.3.3 → 1.4.0 bump; SDK v0.5.33 for the engine 0.4.6 → 1.0.0
one). Each parity table was
measured ONCE at its bump — a historical record, not a standing guarantee.

**Never pass `dt=` to `RigidOptions`** (since genesis-world 1.4.0) — that
field became the solver SUBSTEP interval, so restating `SimOptions.dt` there
now contradicts `substeps` and `Scene.build` raises. Ask for the rate with
`VehicleScene(substeps=...)` instead; `VehicleScene` strips the legacy
spelling and warns once.
