"""genesis_vehicle: ray-cast wheel + Pacejka-tire vehicle SDK on top of Genesis.

A single import gives you everything:

    from genesis_vehicle import (
        # Top-level driver
        VehiclePhysics,
        # Inputs (unified + typed)
        VehicleInputs, VehicleStepInputs,
        AckermannInputs, PartialAckermannInputs, SkidSteerInputs, NoSteerInputs,
        # Config
        VehicleConfig, WheelConfig, ChassisConfig,
        # Strategies
        Ackermann, PartialAckermann, SkidSteer, NoSteer,
        FWD, RWD, AWD, PerSide,
        Independent, SameSideBelt,
        RollingResistance, LowSpeedRegularizer, StaticFrictionLock,
        # Tire models
        PacejkaAnisotropic, CoulombIsotropic,
        # Sensors / utilities
        WheelRayPattern,
        # Dynamics primitives (testable, pure-Python)
        brake_torque_signed, suspension_normal_force,
        # Presets
        car_4w_rwd_ackermann, car_4w_awd_ackermann,
        truck_6w_partial_ackermann, tank_10w_skid_belt,
    )

Lazy imports: modules that touch the Genesis C extension (core, raycast,
visual) are imported on first attribute access so that pure-Python unit tests
on URDF parsing, config merging, and strategy logic can run without GPU.
"""

# --- Eager (no Genesis runtime needed) ---------------------------------------

from .config import (
    ChassisConfig,
    ConfigError,
    ResolvedConfig,
    VehicleConfig,
    WheelConfig,
    resolve,
)
from .inputs import (
    AckermannInputs,
    NoSteerInputs,
    PartialAckermannInputs,
    SkidSteerInputs,
    VehicleInputs,
    VehicleStepInputs,
)
from .urdf import (
    URDFParsedConfig,
    estimate_spin_inertia_from_genesis,
    parse_inertia_max_principal_genesis,   # deprecated alias
    parse_urdf,
)
from .dynamics import (
    brake_torque_signed,
    suspension_normal_force,
)
from .strategies import (
    # Steering
    SteeringStrategy,
    Ackermann, PartialAckermann, SkidSteer, NoSteer,
    # Drivetrain
    DrivetrainStrategy,
    FWD, RWD, AWD, PerSide,
    # Coupling
    CouplingStrategy,
    Independent, SameSideBelt,
    # Stability
    StabilityHook,
    RollingResistance, LowSpeedRegularizer, StaticFrictionLock,
)
from .tire_models import (
    TireModel,
    PacejkaAnisotropic,
    CoulombIsotropic,
)
from .presets import (
    car_4w_rwd_ackermann,
    car_4w_awd_ackermann,
    truck_6w_partial_ackermann,
    tank_10w_skid_belt,
)

# --- Lazy (require Genesis at import time) -----------------------------------

_GENESIS_LAZY = {
    "VehiclePhysics": ("genesis_vehicle.core", "VehiclePhysics"),
    "WheelMeta": ("genesis_vehicle.core", "WheelMeta"),
    "PipelineContext": ("genesis_vehicle.core", "PipelineContext"),
    "WheelRayPattern": ("genesis_vehicle.raycast", "WheelRayPattern"),
    "read_distances": ("genesis_vehicle.raycast", "read_distances"),
    "VisualSync": ("genesis_vehicle.visual", "VisualSync"),
}


def __getattr__(name: str):
    if name in _GENESIS_LAZY:
        import importlib
        module_name, attr = _GENESIS_LAZY[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'genesis_vehicle' has no attribute {name!r}")


__all__ = [
    # Driver (lazy)
    "VehiclePhysics", "WheelMeta", "PipelineContext",
    # Sensor / raycast (lazy)
    "WheelRayPattern", "read_distances",
    # Visual (lazy)
    "VisualSync",
    # Config
    "VehicleConfig", "WheelConfig", "ChassisConfig", "ResolvedConfig", "ConfigError",
    "resolve",
    # Inputs
    "VehicleInputs", "VehicleStepInputs",
    "AckermannInputs", "PartialAckermannInputs", "SkidSteerInputs", "NoSteerInputs",
    # URDF
    "URDFParsedConfig", "parse_urdf",
    "estimate_spin_inertia_from_genesis",
    "parse_inertia_max_principal_genesis",   # deprecated alias
    # Dynamics primitives
    "brake_torque_signed", "suspension_normal_force",
    # Strategy ABCs
    "SteeringStrategy", "DrivetrainStrategy", "CouplingStrategy", "StabilityHook",
    # Steering
    "Ackermann", "PartialAckermann", "SkidSteer", "NoSteer",
    # Drivetrain
    "FWD", "RWD", "AWD", "PerSide",
    # Coupling
    "Independent", "SameSideBelt",
    # Stability hooks
    "RollingResistance", "LowSpeedRegularizer", "StaticFrictionLock",
    # Tire models
    "TireModel", "PacejkaAnisotropic", "CoulombIsotropic",
    # Presets
    "car_4w_rwd_ackermann", "car_4w_awd_ackermann",
    "truck_6w_partial_ackermann", "tank_10w_skid_belt",
]
