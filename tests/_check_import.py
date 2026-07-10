"""One-shot check that the public surface imports cleanly without Genesis runtime.
Run via:  python -m genesis_vehicle.tests._check_import
"""
import genesis_vehicle
from genesis_vehicle import (
    __version__, VERSION_INFO, version, version_info,
    VehicleInputs, VehicleStepInputs,
    AckermannInputs, PartialAckermannInputs,
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
    brake_torque_signed, suspension_normal_force,
    car_4w_rwd_ackermann, car_4w_awd_ackermann,
    truck_6w_partial_ackermann, tank_10w_skid_belt,
    stability_hooks_for_profile,
)

print(f"genesis_vehicle v{__version__} {VERSION_INFO}")
print(f"  version() = {version()}")
print(f"  version_info() = {version_info()}")
print("eager surface imports OK")

lazy_names = ["VehiclePhysics", "WheelMeta", "PipelineContext",
              "WheelRayPattern", "read_distances",
              "VisualPartsTransforms",
              "add_vehicle", "make_wheel_raycaster"]
for name in lazy_names:
    assert name in genesis_vehicle.__all__, f"{name} missing from __all__"
print(f"lazy names registered (deferred until first access): {lazy_names}")
