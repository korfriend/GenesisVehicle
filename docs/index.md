# genesis_vehicle — Documentation

Landing page. Pick the document that matches what you're doing.

| If you want to... | Start here |
|---|---|
| Run something in 1 minute | [`quickstart.md`](quickstart.md) |
| See all 15 runnable samples (+ which have `--viewer`) | [`../samples/README.md`](../samples/README.md) |
| Build a mental model before reading code | [`concepts.md`](concepts.md) |
| Choose the high-level (`VehicleScene`) vs low-level (`VehiclePhysics`) API | [`concepts.md`](concepts.md#start-here-the-two-api-layers) |
| Pick the right batching axis (L1 / L2 / L3) for your workflow | [`batching.md`](batching.md) |
| Decide CPU vs GPU physics (measured crossover, `init_backend`, `--gpu`) | [`backends.md`](backends.md) |
| Choose a tire model (Pacejka vs Coulomb) and understand the ground-contact mechanism | [`tire-and-contact.md`](tire-and-contact.md) |
| Drive from one unified `VehicleScene` object, and cut the wheel-raycast cost on heavy static terrain | [`dual-scene-raycast.md`](dual-scene-raycast.md) |
| Make a vehicle follow a waypoint path (path → Steer/Throttle, sweep-table based) | [`path-following.md`](path-following.md) |
| Run the OSC physics server for an external client (Unreal / Unity), or look up the wire schema | [`server.md`](server.md) |
| Look up a specific class / function / default | [`api-reference.md`](api-reference.md) |
| Understand WHERE in the simulation step your hook runs | [`pipeline-and-hooks.md`](pipeline-and-hooks.md) |
| Pick a stability profile (or understand why it matters for MPPI / Real2Sim) | [`stability-profiles.md`](stability-profiles.md) |
| Verify what the SDK promises about brake sign / N clamp / `i_wheel` truth / steering convention | [`physics-contracts.md`](physics-contracts.md) |
| Author a URDF the ray-wheel model accepts (no wheel colliders, suspension origin = wheel centre, every link inertial) | [`physics-contracts.md` §7.9](physics-contracts.md#79-urdf-contracts-for-ray-wheels-auto-corrected-since-v1122) |
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
    truck_6w_partial_ackermann, tank_10w_skid_belt,
    # utilities
    WheelRayPattern, parse_urdf, stability_hooks_for_profile,
    # path following (docs/path-following.md)
    PathFollower, SweepTable,
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
| Path following | waypoints + signed target speeds → per-step `(throttle, steer, brake)` by inverting a measured sweep table; cusps (speed-sign flips) handled as stop-and-reverse; numpy-only at control time | `PathFollower`, `SweepTable` | [`path-following.md`](path-following.md) |
| Sweep measurement | measures a vehicle's (v, throttle, steer, pitch, roll) → (a, ω_z) response grid in one batched L3 run (build-once, body-frame accurate) | `python -m genesis_vehicle.control.sweep_measure` | [`path-following.md`](path-following.md) §1 |
| Stability profiles | maps a use-case profile (`"control"` / `"raw"` / `"research"`) to the right stability-hook stack | `stability_hooks_for_profile` | [`stability-profiles.md`](stability-profiles.md) |
| OSC physics server | runs the SDK as a standalone physics process for an external engine (UE / Unity), L2 and L3 modes | `python -m genesis_vehicle.server` | [`server.md`](server.md) |
| Server benchmark | official server perf matrix (mock UE client, tanks × terrain × mode × backend) | `python -m genesis_vehicle.server.benchmark` | [`server.md`](server.md) §2.1 |
| Samples | 15 runnable, self-contained examples with bundled assets | `python -m genesis_vehicle.samples.<name>` | [`../samples/README.md`](../samples/README.md) |

**Model preparation & scene assembly**

| Utility | What it does | Entry point | Details |
|---|---|---|---|
| URDF parsing | discovers wheels/axle order, steer-axis sign, and per-wheel geometry from any URDF; the default source of `WheelConfig` | `parse_urdf`, `estimate_spin_inertia_from_genesis` | [`api-reference.md`](api-reference.md) §4 |
| URDF auto-repair | makes an arbitrary URDF ray-wheel ready: wheel colliders become render-only, a suspension origin sitting off the wheel centre is folded back onto it, missing `<inertial>`s injected. Runs automatically inside `add_vehicle` and the OSC server; the original file is never modified | `prepare_vehicle_urdf` | [`physics-contracts.md` §7.9](physics-contracts.md#79-urdf-contracts-for-ray-wheels-auto-corrected-since-v1122) |
| Scene helpers | one-call vehicle + wheel-raycaster insertion for a raw `gs.Scene` (the low-level path; prefer `VehicleScene`) | `add_vehicle`, `make_wheel_raycaster` | [`api-reference.md`](api-reference.md) §0/§9 |
| Raycast helpers | wheel ray pattern definition and shape-normalized sensor reads (`(n_envs, N_WHEELS)` both for 1 and N envs) | `WheelRayPattern`, `read_distances` | [`tire-and-contact.md`](tire-and-contact.md) |

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

**Backend compatibility:** validated against the Genesis physics backend
`genesis-world ≥ 1.0.0`. Per-release backend notes are in
[`../CHANGELOG.md`](../CHANGELOG.md) (see the 0.5.33 entry for the
0.4.6 → 1.0.0 bump).
