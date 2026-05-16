"""One-shot check that the public surface imports cleanly without Genesis runtime.
Run via:  python -m genesis_vehicle.tests._check_import
"""
from genesis_vehicle import (
    VehicleInputs, AckermannInputs, PartialAckermannInputs,
    SkidSteerInputs, NoSteerInputs,
    VehicleConfig, WheelConfig, ChassisConfig, ResolvedConfig,
    ConfigError, resolve,
    URDFParsedConfig, parse_urdf, estimate_spin_inertia_from_genesis,
    SteeringStrategy, DrivetrainStrategy, CouplingStrategy, StabilityHook,
    Ackermann, PartialAckermann, SkidSteer, NoSteer,
    FWD, RWD, AWD, PerSide,
    Independent, SameSideBelt,
    RollingResistance, LowSpeedRegularizer, StaticFrictionLock,
    TireModel, PacejkaAnisotropic, CoulombIsotropic,
    car_4w_rwd_ackermann, car_4w_awd_ackermann,
    truck_6w_partial_ackermann, tank_10w_skid_belt,
)
print("eager surface imports OK")
import genesis_vehicle as gv
lazy_names = ["VehiclePhysics", "WheelMeta", "PipelineContext",
              "WheelRayPattern", "read_distances", "VisualSync"]
for name in lazy_names:
    assert name in gv.__all__, f"{name} missing from __all__"
print(f"lazy names registered (deferred until first access): {lazy_names}")
