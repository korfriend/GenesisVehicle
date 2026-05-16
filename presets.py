"""Preset VehicleConfig builders for the four reference topologies."""

from __future__ import annotations

from typing import Any, Optional

from .config import (
    DEFAULT_C_COMPRESSION,
    DEFAULT_C_EXTENSION,
    DEFAULT_REST_STROKE,
    ChassisConfig,
    VehicleConfig,
    WheelConfig,
)
from .strategies import (
    Ackermann,
    AWD,
    LowSpeedRegularizer,
    PartialAckermann,
    PerSide,
    RWD,
    RollingResistance,
    SameSideBelt,
    SkidSteer,
    StaticFrictionLock,
    Independent,
)
from .tire_models import PacejkaAnisotropic


# ---------------------------------------------------------------------------
# Cars (HJW-style 4-wheel Ackermann)
# ---------------------------------------------------------------------------


def _hjw_brake_bias() -> list[float]:
    """Front-biased 60/40 split, equal within axle. Matches HJW BRAKE_BIAS_PER_WHEEL."""
    return [0.30, 0.30, 0.20, 0.20]


def _hjw_wheel_overrides() -> dict[str, WheelConfig]:
    """Per-wheel suspension/tire overrides matching HJW car_raywheel.py constants."""
    common = dict(
        rest_stroke=0.10,
        k_susp=70_000.0,
        c_compression=14_000.0,
        c_extension=4_000.0,
        comp_rate_clamp=30.0,
        mu_long=1.0,
        mu_lat=1.0,
        rolling_resistance_cr=0.015,
        pb_x=10.0, pc_x=1.65, pe_x=0.4,
        pb_y=8.0, pc_y=1.30, pe_y=0.4,
        radius=0.358,
    )
    return {
        "front_left_wheel":  WheelConfig(**common),
        "front_right_wheel": WheelConfig(**common),
        "rear_left_wheel":   WheelConfig(**common),
        "rear_right_wheel":  WheelConfig(**common),
    }


def car_4w_rwd_ackermann(urdf_path: str, n_envs: int = 1) -> VehicleConfig:
    """4-wheel RWD car with front Ackermann steering. Matches HJW reference."""
    return VehicleConfig.from_urdf(
        urdf_path,
        steering=Ackermann(max_steer_rad=0.7, front_axle=0),
        drivetrain=RWD(
            t_drive_max=1000.0,
            t_brake_max=2500.0,
            driven_axles=(1,),
            brake_bias=_hjw_brake_bias(),
        ),
        coupling=Independent(),
        tire=PacejkaAnisotropic(eps_v=0.5),
        wheel_overrides=_hjw_wheel_overrides(),
        chassis=ChassisConfig(omega_max=100.0, eps_v=0.5),
        stability_hooks=[
            RollingResistance(),
            LowSpeedRegularizer(v_kin_com=0.5, ang_kin=0.5),
        ],
        dt=1.0 / 48.0,
    )


def car_4w_awd_ackermann(urdf_path: str, n_envs: int = 1) -> VehicleConfig:
    """4-wheel AWD car with front Ackermann steering."""
    return VehicleConfig.from_urdf(
        urdf_path,
        steering=Ackermann(max_steer_rad=0.7, front_axle=0),
        drivetrain=AWD(
            t_drive_max=1000.0,
            t_brake_max=2500.0,
            drive_weights=[0.25, 0.25, 0.25, 0.25],
            brake_bias=_hjw_brake_bias(),
        ),
        coupling=Independent(),
        tire=PacejkaAnisotropic(eps_v=0.5),
        wheel_overrides=_hjw_wheel_overrides(),
        chassis=ChassisConfig(omega_max=100.0, eps_v=0.5),
        stability_hooks=[
            RollingResistance(),
            LowSpeedRegularizer(v_kin_com=0.5, ang_kin=0.5),
        ],
        dt=1.0 / 48.0,
    )


def truck_6w_partial_ackermann(urdf_path: str, n_envs: int = 1) -> VehicleConfig:
    """6-wheel truck: front Ackermann, middle + rear axles driven."""
    return VehicleConfig.from_urdf(
        urdf_path,
        steering=PartialAckermann(max_steer_rad=0.55, steered_axles=(0,)),
        drivetrain=AWD(
            t_drive_max=1500.0,
            t_brake_max=3500.0,
            # Driven on middle (axle 1) + rear (axle 2) wheels; weighted equally.
            # drive_weights are aligned to wheel index, so the user supplies them
            # explicitly when the URDF wheel order is known. We default to None
            # here, which uses uniform 1/6 across all wheels — adequate as a
            # baseline preset and easily overridden.
            brake_bias=None,
        ),
        coupling=Independent(),
        tire=PacejkaAnisotropic(eps_v=0.5),
        chassis=ChassisConfig(omega_max=100.0, eps_v=0.5),
        stability_hooks=[
            RollingResistance(),
            LowSpeedRegularizer(v_kin_com=0.5, ang_kin=0.5),
        ],
        dt=1.0 / 48.0,
    )


# ---------------------------------------------------------------------------
# Tank (KDU-style 10-wheel skid-steer)
# ---------------------------------------------------------------------------


def _kdu_wheel_overrides() -> dict[str, WheelConfig]:
    """Per-wheel parameters matching KDU/physics.py constants. KDU uses a
    SYMMETRIC damper (C_SUSP applied both directions), expressed here as
    c_compression == c_extension."""
    common = dict(
        rest_stroke=0.05,
        k_susp=1_000_000.0,
        c_compression=120_000.0,
        c_extension=120_000.0,
        comp_rate_clamp=30.0,
        mu_long=0.9,
        mu_lat=0.9 * 0.7,        # KDU LAT_SCALE = 0.7
        rolling_resistance_cr=0.05,
        pb_x=5.0, pc_x=1.6, pe_x=0.4,
        pb_y=4.0, pc_y=1.4, pe_y=0.4,
        radius=0.4,
    )
    names = [
        "l_sprocket", "l_road1", "l_road2", "l_road3", "l_idler",
        "r_sprocket", "r_road1", "r_road2", "r_road3", "r_idler",
    ]
    return {n: WheelConfig(**common) for n in names}


def tank_10w_skid_belt(urdf_path: str, n_envs: int = 1) -> VehicleConfig:
    """10-wheel skid-steer tank with same-side belt coupling. Matches KDU reference."""
    return VehicleConfig.from_urdf(
        urdf_path,
        steering=SkidSteer(),
        drivetrain=PerSide(
            t_drive_max=30_000.0,
            t_brake_max=30_000.0,
            steer_gain=1.0,
            omega_max_drive=100.0,
            throttle_gear_cap=0.3,
            use_per_side_taper=True,
        ),
        coupling=SameSideBelt(),
        tire=PacejkaAnisotropic(eps_v=0.5),
        wheel_overrides=_kdu_wheel_overrides(),
        chassis=ChassisConfig(omega_max=100.0, eps_v=0.5),
        stability_hooks=[
            RollingResistance(tanh_scale=0.5),
            StaticFrictionLock(brake_thr=0.3, v_thr=0.5, hold_k=200_000.0),
        ],
        dt=0.005,
    )
