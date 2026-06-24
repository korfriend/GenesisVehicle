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

# --- Version --------------------------------------------------------------

from ._version import __version__, VERSION_INFO


def version() -> str:
    """Return the SDK version as a string (e.g. ``'0.3.0'``)."""
    return __version__


def version_info() -> tuple[int, int, int]:
    """Return the SDK version as a ``(major, minor, patch)`` tuple."""
    return VERSION_INFO




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
    car_4w_fwd_ackermann,
    car_4w_awd_ackermann,
    truck_6w_partial_ackermann,
    tank_10w_skid_belt,
    stability_hooks_for_profile,
)

# --- Lazy (require Genesis at import time) -----------------------------------

_GENESIS_LAZY = {
    "VehiclePhysics": ("genesis_vehicle.core", "VehiclePhysics"),
    "WheelMeta": ("genesis_vehicle.core", "WheelMeta"),
    "PipelineContext": ("genesis_vehicle.core", "PipelineContext"),
    "MultiVehiclePhysics": ("genesis_vehicle.multi_vehicle", "MultiVehiclePhysics"),
    "MultiVehicleKindPhysics": ("genesis_vehicle.multi_vehicle", "MultiVehicleKindPhysics"),
    "WheelRayPattern": ("genesis_vehicle.raycast", "WheelRayPattern"),
    "read_distances": ("genesis_vehicle.raycast", "read_distances"),
    "get_link_transforms": ("genesis_vehicle.kinematics", "get_link_transforms"),
    "LinkTransforms": ("genesis_vehicle.kinematics", "LinkTransforms"),
    "VisualJointSync": ("genesis_vehicle.visual", "VisualJointSync"),
    "VisualPartsTransforms": ("genesis_vehicle.core", "VisualPartsTransforms"),
    "add_vehicle": ("genesis_vehicle.scene_helpers", "add_vehicle"),
    "make_wheel_raycaster": ("genesis_vehicle.scene_helpers", "make_wheel_raycaster"),
    "VehicleScene": ("genesis_vehicle.vehicle_scene", "VehicleScene"),
    "Vehicle": ("genesis_vehicle.vehicle_scene", "Vehicle"),
    "StaticBody": ("genesis_vehicle.vehicle_scene", "StaticBody"),
    "Obstacle": ("genesis_vehicle.vehicle_scene", "Obstacle"),
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
    # Version
    "__version__", "VERSION_INFO", "version", "version_info",
    # Driver (lazy)
    "VehiclePhysics", "WheelMeta", "PipelineContext",
    "MultiVehiclePhysics", "MultiVehicleKindPhysics",
    # Sensor / raycast (lazy)
    "WheelRayPattern", "read_distances",
    # Kinematics — per-link transforms (lazy)
    "get_link_transforms", "LinkTransforms",
    # Render feed for external engines (lazy)
    "VisualPartsTransforms",
    # Scene helpers (lazy — thin wrappers around Genesis boilerplate)
    "add_vehicle", "make_wheel_raycaster",
    # Unified high-level API (lazy) — VehicleScene is the center of the SDK
    "VehicleScene", "Vehicle", "StaticBody", "Obstacle",
    # Visual joint sync for the Genesis viewer (lazy)
    "VisualJointSync",
    # Config
    "VehicleConfig", "WheelConfig", "ChassisConfig", "ResolvedConfig", "ConfigError",
    "resolve",
    # Inputs
    "VehicleInputs", "VehicleStepInputs",
    "AckermannInputs", "PartialAckermannInputs", "SkidSteerInputs", "NoSteerInputs",
    # URDF
    "URDFParsedConfig", "parse_urdf",
    "estimate_spin_inertia_from_genesis",
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
    "car_4w_rwd_ackermann", "car_4w_fwd_ackermann", "car_4w_awd_ackermann",
    "truck_6w_partial_ackermann", "tank_10w_skid_belt",
    "stability_hooks_for_profile",
]
