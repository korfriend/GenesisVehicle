"""Preset VehicleConfig builders for the four reference topologies.

Every preset takes a ``stability`` keyword (``"control" | "raw" | "research"``)
that selects which numerical stabilization hooks the resulting config carries.

  - ``"control"`` (default) — RL/MPPI-friendly. Bundles RollingResistance +
    LowSpeedRegularizer (active under throttle/brake, see footgun note in
    CHANGELOG v0.3.0) + StaticFrictionLock for tanks. **For Real2Sim too**:
    fitting and deployment must share the same forward model, and that
    forward model includes the hooks.
  - ``"raw"`` — empty hook list (raw ray-wheel + Pacejka, no numerical
    stabilization). For debugging the bare dynamics or running hook-ablation
    studies, NOT a general Real2Sim recommendation. (See API.md S6.)
  - ``"research"`` — empty hook list; caller is expected to assemble hooks
    themselves via ``cfg.stability_hooks``.
"""

from __future__ import annotations

from typing import Any, Optional

from .config import (
    DEFAULT_C_COMPRESSION,
    DEFAULT_C_EXTENSION,
    DEFAULT_REST_STROKE,
    ChassisConfig,
    ConfigError,
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
    StabilityHook,
    StaticFrictionLock,
    Independent,
)
from .tire_models import PacejkaAnisotropic


# ---------------------------------------------------------------------------
# Stability profile
# ---------------------------------------------------------------------------


_VALID_PROFILES = ("control", "raw", "research")


def stability_hooks_for_profile(
    profile: str = "control",
    vehicle_kind: str = "car",
) -> list[StabilityHook]:
    """Materialize the stability hooks for a given profile + vehicle kind.

    Used by the presets; exposed publicly so users assembling a VehicleConfig
    from scratch can opt in to the same defaults.

    Args:
        profile: ``"control"``, ``"raw"``, or ``"research"``.
        vehicle_kind: ``"car"`` (no static lock) or ``"tank"`` (adds
            StaticFrictionLock under the ``"control"`` profile).
    """
    if profile not in _VALID_PROFILES:
        raise ConfigError(
            f"Unknown stability profile {profile!r}; "
            f"expected one of {_VALID_PROFILES}."
        )
    if profile in ("raw", "research"):
        return []
    # profile == "control"
    hooks: list[StabilityHook] = [
        RollingResistance(),
        LowSpeedRegularizer(
            v_kin_com=0.5,
            ang_kin=0.5,
            disable_when_control_active=False,   # active under MPPI throttle
        ),
    ]
    if vehicle_kind == "tank":
        hooks.append(
            StaticFrictionLock(brake_thr=0.3, v_thr=0.5, hold_k=200_000.0)
        )
    return hooks


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


def car_4w_rwd_ackermann(
    urdf_path: str,
    n_envs: int = 1,
    *,
    stability: str = "control",
) -> VehicleConfig:
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
        stability_hooks=stability_hooks_for_profile(stability, vehicle_kind="car"),
        dt=1.0 / 48.0,
    )


def car_4w_awd_ackermann(
    urdf_path: str,
    n_envs: int = 1,
    *,
    stability: str = "control",
) -> VehicleConfig:
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
        stability_hooks=stability_hooks_for_profile(stability, vehicle_kind="car"),
        dt=1.0 / 48.0,
    )


def truck_6w_partial_ackermann(
    urdf_path: str,
    n_envs: int = 1,
    *,
    stability: str = "control",
) -> VehicleConfig:
    """6-wheel truck: front Ackermann, middle + rear axles driven."""
    return VehicleConfig.from_urdf(
        urdf_path,
        steering=PartialAckermann(max_steer_rad=0.55, steered_axles=(0,)),
        drivetrain=AWD(
            t_drive_max=1500.0,
            t_brake_max=3500.0,
            brake_bias=None,
        ),
        coupling=Independent(),
        tire=PacejkaAnisotropic(eps_v=0.5),
        chassis=ChassisConfig(omega_max=100.0, eps_v=0.5),
        stability_hooks=stability_hooks_for_profile(stability, vehicle_kind="car"),
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


def tank_10w_skid_belt(
    urdf_path: str,
    n_envs: int = 1,
    *,
    stability: str = "control",
) -> VehicleConfig:
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
        stability_hooks=stability_hooks_for_profile(stability, vehicle_kind="tank"),
        dt=0.005,
    )
