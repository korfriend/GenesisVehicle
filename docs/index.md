# genesis_vehicle — Documentation

Landing page. Pick the document that matches what you're doing.

| If you want to... | Start here |
|---|---|
| Run something in 1 minute | [`quickstart.md`](quickstart.md) |
| See all 12 runnable samples (+ which have `--viewer`) | [`../samples/README.md`](../samples/README.md) |
| Build a mental model before reading code | [`concepts.md`](concepts.md) |
| Pick the right batching axis (L1 / L2 / L3) for your workflow | [`batching.md`](batching.md) |
| Choose a tire model (Pacejka vs Coulomb) and understand the ground-contact mechanism | [`tire-and-contact.md`](tire-and-contact.md) |
| Drive from one unified `VehicleScene` object, and cut the wheel-raycast cost on heavy static terrain | [`two-scene-raycast.md`](two-scene-raycast.md) |
| Run the OSC physics server for an external client (Unreal / Unity), or look up the wire schema | [`server.md`](server.md) |
| Look up a specific class / function / default | [`api-reference.md`](api-reference.md) |
| Understand WHERE in the simulation step your hook runs | [`pipeline-and-hooks.md`](pipeline-and-hooks.md) |
| Pick a stability profile (or understand why it matters for MPPI / Real2Sim) | [`stability-profiles.md`](stability-profiles.md) |
| Verify what the SDK promises about brake sign / N clamp / `i_wheel` truth / steering convention | [`physics-contracts.md`](physics-contracts.md) |
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
    # per-link transforms (telemetry / animation / attach)
    get_link_transforms, LinkTransforms,
    # version
    __version__, version, version_info,
)
```

See [`api-reference.md`](api-reference.md) for the full surface.

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
