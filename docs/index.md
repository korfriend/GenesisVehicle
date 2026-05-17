# genesis_vehicle — Documentation

Landing page. Pick the document that matches what you're doing.

| If you want to... | Start here |
|---|---|
| Run something in 1 minute | [`quickstart.md`](quickstart.md) |
| Build a mental model before reading code | [`concepts.md`](concepts.md) |
| Look up a specific class / function / default | [`api-reference.md`](api-reference.md) |
| Understand WHERE in the simulation step your hook runs | [`pipeline-and-hooks.md`](pipeline-and-hooks.md) |
| Pick a stability profile (or understand why it matters for MPPI / Real2Sim) | [`stability-profiles.md`](stability-profiles.md) |
| Verify what the SDK promises about brake sign / N clamp / `i_wheel` truth / steering convention | [`physics-contracts.md`](physics-contracts.md) |
| Port your HJW / JMK / KDU code onto the SDK | [`migration.md`](migration.md) |
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
    # version
    __version__, version, version_info,
)
```

See [`api-reference.md`](api-reference.md) for the full surface.

## Version

Current SDK version is `genesis_vehicle.__version__`. `VehiclePhysics.__init__`
prints a one-line banner with the version on first construction in a process:

```
[genesis_vehicle v0.4.0] Initialized: 4 wheels, Ackermann, RWD, Independent, n_envs=1, hooks=[RollingResistance, LowSpeedRegularizer]
```

Release history lives in
[`../CHANGELOG.md`](../CHANGELOG.md).
