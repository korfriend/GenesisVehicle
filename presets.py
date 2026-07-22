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
    suspension_from_mass,
)
from .strategies import (
    Ackermann,
    AWD,
    FWD,
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
        vehicle_kind: ``"car"`` or ``"tank"``. Both get the same hook set
            in v0.5.8+; the parameter is kept for forward compatibility
            with future vehicle-specific tweaks.
    """
    if profile not in _VALID_PROFILES:
        raise ConfigError(
            f"Unknown stability profile {profile!r}; "
            f"expected one of {_VALID_PROFILES}."
        )
    if profile in ("raw", "research"):
        return []
    # profile == "control" — applied to every vehicle_kind. Prior to v0.5.8
    # StaticFrictionLock was tank-only; the new stick-slip lock is cheap
    # enough (no work when not engaged) that there's no reason to leave
    # cars without it. Removing it caused a real footgun where any car
    # preset on a slope or after impact creeped indefinitely under brake.
    hooks: list[StabilityHook] = [
        RollingResistance(),
        LowSpeedRegularizer(
            v_kin_com=0.5,
            ang_kin=0.5,
            # IMPORTANT: regularizer is OFF when the user is actively throttling
            # or braking. Otherwise the omega-pull / F-scale at v=0 would force
            # omega back to 0 every step → vehicle cannot start from rest under
            # throttle. (v0.3.0 set this to False thinking it would help MPPI
            # low-speed oscillation; v0.5.1 reverted it because the startup
            # footgun is far worse. MPPI users who need an always-on
            # regularizer can opt into `disable_when_control_active=False`
            # via `stability="research"` + a custom hook list.)
            disable_when_control_active=True,
        ),
        StaticFrictionLock(brake_thr=0.3, v_thr=0.5),
    ]
    return hooks


# ---------------------------------------------------------------------------
# Cars (4-wheel Ackermann; reference-car tuning)
# ---------------------------------------------------------------------------


def _car_brake_bias() -> list[float]:
    """Front-biased 60/40 split, equal within axle (reference-car tuning)."""
    return [0.30, 0.30, 0.20, 0.20]


def _car_wheel_overrides() -> dict[str, WheelConfig]:
    """Per-wheel suspension/tire overrides (reference-car tuning constants)."""
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
    """4-wheel RWD car with front Ackermann steering (reference-car tuning)."""
    return VehicleConfig.from_urdf(
        urdf_path,
        steering=Ackermann(max_steer_rad=0.7, front_axle=0),
        drivetrain=RWD(
            t_drive_max=1000.0,
            t_brake_max=2500.0,
            driven_axles=(1,),
            brake_bias=_car_brake_bias(),
        ),
        coupling=Independent(),
        tire=PacejkaAnisotropic(eps_v=0.5),
        wheel_overrides=_car_wheel_overrides(),
        chassis=ChassisConfig(omega_max=100.0, eps_v=0.5),
        stability_hooks=stability_hooks_for_profile(stability, vehicle_kind="car"),
        recommended_dt=0.025,   # 40 Hz SDK default (v1.0.19)
    )


def car_4w_fwd_ackermann(
    urdf_path: str,
    n_envs: int = 1,
    *,
    stability: str = "control",
) -> VehicleConfig:
    """4-wheel FWD car with front Ackermann steering (typical passenger car)."""
    return VehicleConfig.from_urdf(
        urdf_path,
        steering=Ackermann(max_steer_rad=0.7, front_axle=0),
        drivetrain=FWD(
            t_drive_max=1000.0,
            t_brake_max=2500.0,
            driven_axles=(0,),
            brake_bias=_car_brake_bias(),
        ),
        coupling=Independent(),
        tire=PacejkaAnisotropic(eps_v=0.5),
        wheel_overrides=_car_wheel_overrides(),
        chassis=ChassisConfig(omega_max=100.0, eps_v=0.5),
        stability_hooks=stability_hooks_for_profile(stability, vehicle_kind="car"),
        recommended_dt=0.025,   # 40 Hz SDK default (v1.0.19)
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
            brake_bias=_car_brake_bias(),
        ),
        coupling=Independent(),
        tire=PacejkaAnisotropic(eps_v=0.5),
        wheel_overrides=_car_wheel_overrides(),
        chassis=ChassisConfig(omega_max=100.0, eps_v=0.5),
        stability_hooks=stability_hooks_for_profile(stability, vehicle_kind="car"),
        recommended_dt=0.025,   # 40 Hz SDK default (v1.0.19)
    )


def truck_6w_partial_ackermann(
    urdf_path: str,
    n_envs: int = 1,
    *,
    stability: str = "control",
) -> VehicleConfig:
    """6-wheel truck: front-axle Ackermann steering, middle + rear axles driven.

    Drive layout mirrors a real truck — front axle steers (no drive), mid and
    rear axles split the drive torque equally. Uses RWD strategy with
    ``driven_axles=(1, 2)`` (RWD's name is slightly misleading here; it just
    means "drive uniformly on listed axles").

    Torque sized for a ~5-ton truck (URDF chassis mass 5000 kg, 6 wheels @
    40 kg each):
      - ``t_drive_max = 10_000 N·m`` total → 2500 N·m / driven wheel at full
        throttle. ~2-3 m/s² acceleration on flat ground without wheelspin.
      - ``t_brake_max = 15_000 N·m`` total → realistic emergency-brake decel
        of ~4 m/s² when throttle is off (lower brake torques felt weak on a
        5-ton chassis).

    The ``"control"`` stability profile (v0.5.8+) includes a
    ``StaticFrictionLock`` by default for every vehicle kind, so the truck
    holds at rest under brake instead of creeping.
    """
    hooks = stability_hooks_for_profile(stability, vehicle_kind="car")
    return VehicleConfig.from_urdf(
        urdf_path,
        steering=PartialAckermann(max_steer_rad=0.55, steered_axles=(0,)),
        drivetrain=RWD(
            t_drive_max=10_000.0,
            t_brake_max=15_000.0,
            driven_axles=(1, 2),    # mid + rear (NOT front, which steers)
            brake_bias=None,        # default uniform within axle
        ),
        coupling=Independent(),
        tire=PacejkaAnisotropic(eps_v=0.5),
        chassis=ChassisConfig(omega_max=100.0, eps_v=0.5),
        stability_hooks=hooks,
        recommended_dt=0.025,   # 40 Hz SDK default (v1.0.19)
    )


# ---------------------------------------------------------------------------
# Tank (skid-steer, same-side belt coupling)
# ---------------------------------------------------------------------------


# Tracked-vehicle suspension sizing. `TANK_REST_STROKE` is the ray budget and
# `TANK_TARGET_SAG` the static deflection we size the spring for; the damper is
# symmetric (compression == extension), which is what the reference tracked
# vehicle used. These reproduce the constants this preset carried as literals
# until v1.2.1 (k = 1.00e6, c = 120,000 for the 53.2 t / 10-wheel reference) to
# within 4% — the literals were never wrong, they were just never *applied* to
# any URDF whose wheel links happened to be named differently.
TANK_REST_STROKE = 0.05
TANK_TARGET_SAG = 0.05
TANK_ZETA = 0.80


def _tank_wheel_overrides(
    parsed: "Any",
    target_sag: float = TANK_TARGET_SAG,
) -> dict[str, WheelConfig]:
    """Per-wheel tracked-vehicle parameters, keyed by THIS URDF's wheel names.

    Suspension is derived from the URDF's sprung mass (v1.2.1) rather than
    hard-coded, so the preset holds its ride frequency across tracked vehicles
    of different mass and wheel count instead of only fitting the one vehicle
    the literals came from. Tire constants stay fixed — they describe rubber /
    track behaviour, which does not scale with hull mass.
    """
    n = len(parsed.wheels)
    sprung = parsed.sprung_mass or parsed.chassis_mass
    if sprung and n:
        k, c_comp, c_ext = suspension_from_mass(
            sprung, n, target_sag=target_sag,
            zeta_compression=TANK_ZETA, zeta_extension=TANK_ZETA,
        )
    else:
        # URDF declares no masses at all — nothing to size against.
        k, c_comp, c_ext = 1_000_000.0, 120_000.0, 120_000.0

    common = dict(
        comp_rate_clamp=30.0,
        mu_long=0.9,
        mu_lat=0.9 * 0.7,        # reference LAT_SCALE = 0.7
        rolling_resistance_cr=0.05,
        pb_x=5.0, pc_x=1.6, pe_x=0.4,
        pb_y=4.0, pc_y=1.4, pe_y=0.4,
    )
    susp = dict(rest_stroke=TANK_REST_STROKE, k_susp=k,
                c_compression=c_comp, c_extension=c_ext)

    # NB: no `radius` override. The URDF's own wheel geometry is authoritative;
    # a hard-coded radius silently rescaled every other tracked vehicle.
    out: dict[str, WheelConfig] = {}
    for w in parsed.wheels:
        # A URDF that declares its own suspension (non-standard `<dynamics
        # stiffness=...>`) outranks the preset's derived value — the author said
        # something specific and we should not overwrite it. Anything it left
        # unset still gets the derived value. (v1.2.1)
        fields = dict(common)
        for key, val in susp.items():
            if getattr(w, key, None) is None:
                fields[key] = val
        out[w.name] = WheelConfig(**fields)
    return out


def tank_skid_belt(
    urdf_path: str,
    n_envs: int = 1,
    *,
    stability: str = "control",
    target_sag: float = TANK_TARGET_SAG,
) -> VehicleConfig:
    """Skid-steer tank with same-side belt coupling (any wheel count).

    Wheel-count-generic in BOTH senses since v1.2.1: ``VehicleConfig.from_urdf``
    discovers the wheels and ``SkidSteer`` / ``PerSide`` / ``SameSideBelt``
    scale to however many the URDF has, AND the suspension is now sized from the
    URDF's own sprung mass and attached to the URDF's own wheel names. Before
    v1.2.1 the tuning constants were hard-coded literals keyed by the reference
    vehicle's wheel-link names, so any other tracked URDF silently kept
    car-sized module defaults (a 40 t hull on a 70 kN/m spring — four times its
    travel in static sag, wallowing at 0.8 Hz).

    ``target_sag`` sets that sizing: static deflection in metres, which fixes
    the ride frequency (0.05 m -> ~2.2 Hz). Raise it for a softer, longer-travel
    vehicle; lower it for a tighter one.

    Uses ``visual_susp_mode="control"`` because the tank's wheels are heavy
    (500 kg each) and a kinematic set_dofs_position cannot prevent them from
    falling under gravity between substeps. The control path applies a stiff
    PD (kp=1e7, kv=1e5) on each suspension prismatic joint, matching the
    reference-tank behavior.
    """
    from .urdf import parse_urdf

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
        wheel_overrides=_tank_wheel_overrides(parse_urdf(urdf_path), target_sag),
        chassis=ChassisConfig(omega_max=100.0, eps_v=0.5),
        stability_hooks=stability_hooks_for_profile(stability, vehicle_kind="tank"),
        recommended_dt=0.025,   # 40 Hz SDK default (v1.0.19; legacy 0.005 verified unnecessary)
        visual_susp_mode="control",
        # Tank wheels are cylinder primitives — visible spin doesn't add anything
        # but the extra set_dofs_position call costs ~3-5ms per step (noticeable
        # in interactive chase-cam mode).
        visual_spin_enabled=False,
    )
