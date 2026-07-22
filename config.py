"""Configuration dataclasses and the URDF -> config -> default merge algorithm.

WheelConfig fields are all Optional; resolve() fills None values from the
parsed URDF, then from module-level defaults.
"""

import logging
import os
from dataclasses import dataclass, field, fields, replace
from typing import Any, Optional

_logger = logging.getLogger("genesis_vehicle")


class ConfigError(ValueError):
    """Raised when a VehicleConfig fails validation (missing fields, bad combinations)."""


# Module-level defaults used as final fallback when neither user nor URDF supplies a value.
DEFAULT_PACEJKA = dict(
    pb_x=10.0, pc_x=1.65, pe_x=0.4,
    pb_y=8.0, pc_y=1.30, pe_y=0.4,
)
DEFAULT_K_SUSP = 70_000.0
DEFAULT_C_COMPRESSION = 14_000.0
DEFAULT_C_EXTENSION = 4_000.0
DEFAULT_REST_STROKE = 0.10
DEFAULT_MU = 1.0
DEFAULT_ROLLING_RESISTANCE = 0.015
DEFAULT_COMP_RATE_CLAMP = 30.0
DEFAULT_RADIUS = 0.35
DEFAULT_MASS = 20.0
DEFAULT_I_WHEEL = 1.5


def suspension_from_mass(
    sprung_mass: float,
    n_wheels: int,
    *,
    target_sag: float = 0.05,
    zeta_compression: float = 0.70,
    zeta_extension: float = 0.45,
    gravity: float = 9.81,
) -> tuple[float, float, float]:
    """Size a suspension from the mass it actually carries. (v1.2.1)

    Returns ``(k_susp, c_compression, c_extension)`` for ONE wheel, assuming the
    sprung mass is shared evenly::

        k      = (m_sprung * g / n) / target_sag
        c_crit = 2 * sqrt(k * m_sprung / n)
        c_comp = zeta_compression * c_crit,  c_ext = zeta_extension * c_crit

    ``sprung_mass`` must be the mass the SPRINGS carry — use
    ``parse_urdf(...).sprung_mass``, not ``chassis_mass``: the latter is the base
    link alone and omits sprung children (a turret, a cargo body), which
    undersizes the spring by however much those weigh.

    A fixed spring rate cannot serve both a 2 t car and a 40 t tank: at the
    car-sized default (70 kN/m) a 40 t hull sags four times its own suspension
    travel and wallows at 0.8 Hz. Deriving from mass keeps the sag — and hence
    the ride frequency — constant across vehicle scales.
    """
    if n_wheels <= 0:
        raise ConfigError(f"n_wheels must be positive, got {n_wheels}")
    if sprung_mass <= 0.0:
        raise ConfigError(f"sprung_mass must be positive, got {sprung_mass}")
    if target_sag <= 0.0:
        raise ConfigError(f"target_sag must be positive, got {target_sag}")
    import math

    m_share = sprung_mass / n_wheels
    k = (m_share * gravity) / target_sag
    c_crit = 2.0 * math.sqrt(k * m_share)
    return (k, zeta_compression * c_crit, zeta_extension * c_crit)


@dataclass
class WheelConfig:
    """Per-wheel parameters. Any field left None will be filled by resolve()
    from the parsed URDF or from module-level defaults."""

    # Identity / placement
    name: Optional[str] = None
    position: Optional[tuple[float, float, float]] = None  # chassis-local ray origin
    side: Optional[str] = None                              # 'L' | 'R' | None
    axle_index: Optional[int] = None                        # 0 = front, +1 = rearward

    # Wheel dynamics
    radius: Optional[float] = None
    mass: Optional[float] = None
    i_wheel: Optional[float] = None

    # Suspension (per-wheel asymmetric damper)
    rest_stroke: Optional[float] = None
    k_susp: Optional[float] = None
    c_compression: Optional[float] = None
    c_extension: Optional[float] = None
    comp_rate_clamp: Optional[float] = None

    # Friction / tire
    mu_long: Optional[float] = None
    mu_lat: Optional[float] = None
    rolling_resistance_cr: Optional[float] = None

    # Pacejka coefficients
    pb_x: Optional[float] = None
    pc_x: Optional[float] = None
    pe_x: Optional[float] = None
    pb_y: Optional[float] = None
    pc_y: Optional[float] = None
    pe_y: Optional[float] = None

    # URDF joint names (visual layer + DoF lookup)
    susp_joint_name: Optional[str] = None
    steer_joint_name: Optional[str] = None
    spin_joint_name: Optional[str] = None


@dataclass
class ChassisConfig:
    """Chassis-level (vehicle-wide) parameters."""
    base_link_name: str = "base_link"
    mass: Optional[float] = None
    frame: str = "ISO8855"
    omega_max: float = 100.0
    eps_v: float = 0.5


@dataclass
class VehicleConfig:
    """User-facing vehicle configuration. Wheels may be partially specified;
    resolve() fills missing fields from URDF + defaults."""

    urdf_path: str
    wheels: list[WheelConfig]
    steering: Any           # SteeringStrategy
    drivetrain: Any         # DrivetrainStrategy
    coupling: Any           # CouplingStrategy
    tire: Any               # TireModel
    chassis: ChassisConfig = field(default_factory=ChassisConfig)
    stability_hooks: list[Any] = field(default_factory=list)
    # Advisory only — the recommended simulation step size for this preset.
    # ``VehiclePhysics`` pulls its actual ``self.dt`` from ``scene.sim.dt``
    # (Genesis owns physical time); ``recommended_dt`` is what the sample
    # is expected to feed into ``SimOptions``:
    #   cfg = car_4w_rwd_ackermann(...)            # preset declares dt
    #   gs.Scene(sim_options=SimOptions(dt=cfg.recommended_dt, ...))
    # If the user picks a different ``SimOptions.dt``, VehiclePhysics
    # emits a one-time warning (per process) on the first mismatch but
    # does not refuse — Genesis's value wins.
    # v1.0.19: SDK-wide default is 40 Hz (0.025) — verified indistinguishable
    # from 1/48 (and from the tank's legacy 0.005) on bumpy terrain: cruise
    # speed / z-oscillation / yaw-rate within noise (see CHANGELOG 1.0.17).
    # Users can always pick their own dt (server: --override_dt).
    recommended_dt: float = 0.025
    # Drive the URDF wheel visual joints (spin/steer/suspension) through the
    # engine each step so the **Genesis native viewer** shows wheels moving.
    # Default False: the dominant path (headless / external UE·Unity renderer)
    # does NOT need it — read poses from wheel_visual_transforms() (closed-form,
    # ~µs) instead, which works regardless of this flag. Set True ONLY when you
    # actually open the Genesis viewer (`gs.Scene(show_viewer=True)` or a sample
    # `--viewer`). Leaving it on headless is a silent ~ms/step perf trap (it is
    # the dominant SDK cost at scale). Internal/legacy since v1.1.18 —
    # VehicleScene manages wheel visuals itself (instanced renderer);
    # this flag only drives the joint-sync fallback. See docs/server.md.
    enable_wheel_joint_internal_sync: bool = False

    # Visual suspension joint mode:
    #   "auto"       — per-joint decision based on URDF <dynamics> presence (legacy default)
    #   "kinematic"  — force set_dofs_position (light wheels < ~50 kg)
    #   "control"    — force control_dofs_position with high kp/kv (heavy wheels)
    # Heavy wheels (e.g. a 500 kg tank wheel) need PD control or they fall under
    # gravity between substeps even though set_dofs_position is re-applied each
    # step. Tank preset opts into "control" automatically.
    visual_susp_mode: str = "auto"

    # Whether to sync wheel spin angle (set_dofs_position on each wheel's
    # continuous joint). Costs one Genesis call per step; safe to disable for
    # vehicles where wheel rotation is invisible (e.g. tank cylinder wheels are
    # rotationally symmetric in mesh). Tank preset sets this to False for ~3-5ms
    # /step savings in interactive mode. Cars keep it True so mesh wheels visibly
    # roll.
    visual_spin_enabled: bool = True

    # Visual suspension-offset clamp (metres). Bounds how far the wheel mesh may
    # visually travel from rest in wheel_visual_transforms / WheelJointInternalSync — a
    # safety bound against raycast spikes, NOT a physics limit.
    #   None  -> auto: per-wheel = that wheel's rest_stroke (min 0.02 m). Scales
    #            with the vehicle, so large-travel rigs aren't muted.
    #   float -> uniform clamp applied to every wheel (e.g. 0.30).
    # (The old hardcoded 0.19 m is what muted trucks/tanks; None restores travel.)
    susp_visual_clamp: Optional[float] = None

    # Deprecated alias kept so existing user code (cfg.dt, cfg.dt = ...)
    # keeps working through v0.5.x. Will be removed in v0.6.
    @property
    def dt(self) -> float:
        import warnings
        warnings.warn(
            "VehicleConfig.dt is deprecated; use VehicleConfig.recommended_dt. "
            "VehiclePhysics now pulls its actual dt from scene.sim.dt; "
            "recommended_dt is advisory only (see v0.5.31 CHANGELOG).",
            DeprecationWarning, stacklevel=2,
        )
        return self.recommended_dt

    @dt.setter
    def dt(self, value: float) -> None:
        import warnings
        warnings.warn(
            "VehicleConfig.dt is deprecated; assign VehicleConfig.recommended_dt "
            "instead.",
            DeprecationWarning, stacklevel=2,
        )
        self.recommended_dt = float(value)

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str,
        *,
        steering: Any,
        drivetrain: Any,
        coupling: Any,
        tire: Any,
        wheel_overrides: Optional[dict[str, "WheelConfig"]] = None,
        chassis: Optional[ChassisConfig] = None,
        **kwargs: Any,
    ) -> "VehicleConfig":
        """Build a VehicleConfig where wheels[] is auto-populated from the URDF.

        wheel_overrides: dict keyed by wheel name -> partial WheelConfig whose
        non-None fields override the URDF-derived values.

        Back-compat: ``dt=`` is accepted as a kwarg and forwarded to
        ``recommended_dt=`` (emits a DeprecationWarning).
        """
        if "dt" in kwargs and "recommended_dt" not in kwargs:
            import warnings
            warnings.warn(
                "from_urdf(dt=...) is deprecated; pass recommended_dt=... "
                "instead. (See v0.5.31 CHANGELOG.)",
                DeprecationWarning, stacklevel=2,
            )
            kwargs["recommended_dt"] = kwargs.pop("dt")
        # Local import to avoid a circular import (urdf imports from config).
        from .urdf import parse_urdf

        parsed = parse_urdf(urdf_path)
        wheels: list[WheelConfig] = []
        overrides = wheel_overrides or {}
        for w in parsed.wheels:
            override = overrides.get(w.name)
            if override is None:
                wheels.append(w)
            else:
                wheels.append(_merge_wheel(w, override))

        # An override key that matches no wheel is silently dropped, which used
        # to let a whole tuning block (spring rate, friction, Pacejka shape)
        # disappear and leave the vehicle on generic module defaults. Say so.
        # (v1.2.1 - same class of guard as the server's wheelOverrides warning.)
        wheel_names = {w.name for w in parsed.wheels}
        unmatched = [k for k in overrides if k not in wheel_names]
        if unmatched:
            _logger.warning(
                "wheel_overrides: %d of %d key(s) matched no wheel in %s and "
                "were DROPPED (%s). Those wheels keep URDF/module defaults. "
                "This URDF's wheels are: %s",
                len(unmatched), len(overrides), os.path.basename(urdf_path),
                ", ".join(sorted(unmatched)[:8])
                + (", ..." if len(unmatched) > 8 else ""),
                ", ".join(sorted(wheel_names)[:8])
                + (", ..." if len(wheel_names) > 8 else ""),
            )

        ch = chassis if chassis is not None else ChassisConfig()
        if ch.mass is None and parsed.chassis_mass is not None:
            ch = replace(ch, mass=parsed.chassis_mass)
        if parsed.base_link_name and ch.base_link_name == "base_link":
            # Honour parsed base link name only if user didn't customise it.
            ch = replace(ch, base_link_name=parsed.base_link_name)

        return cls(
            urdf_path=urdf_path,
            wheels=wheels,
            steering=steering,
            drivetrain=drivetrain,
            coupling=coupling,
            tire=tire,
            chassis=ch,
            **kwargs,
        )


@dataclass
class ResolvedConfig:
    """Post-merge config. All wheel fields are non-None; consumed by VehiclePhysics."""
    wheels: list[WheelConfig]
    chassis: ChassisConfig
    steering: Any
    drivetrain: Any
    coupling: Any
    tire: Any
    stability_hooks: list[Any]
    recommended_dt: float
    enable_wheel_joint_internal_sync: bool
    urdf: Any   # URDFParsedConfig — used by visual layer for joint axis-sign lookup
    visual_susp_mode: str = "auto"
    visual_spin_enabled: bool = True
    susp_visual_clamp: Optional[float] = None


def _merge_wheel(base: WheelConfig, override: WheelConfig) -> WheelConfig:
    """Return a WheelConfig where each field is override's value if non-None,
    else base's value."""
    out = WheelConfig()
    for f in fields(WheelConfig):
        ov = getattr(override, f.name)
        if ov is not None:
            setattr(out, f.name, ov)
        else:
            setattr(out, f.name, getattr(base, f.name))
    return out


def _fill_defaults(w: WheelConfig) -> WheelConfig:
    """Fill remaining None fields on a WheelConfig with module-level defaults."""
    defaults = {
        "radius": DEFAULT_RADIUS,
        "mass": DEFAULT_MASS,
        "i_wheel": DEFAULT_I_WHEEL,
        "rest_stroke": DEFAULT_REST_STROKE,
        "k_susp": DEFAULT_K_SUSP,
        "c_compression": DEFAULT_C_COMPRESSION,
        "c_extension": DEFAULT_C_EXTENSION,
        "comp_rate_clamp": DEFAULT_COMP_RATE_CLAMP,
        "mu_long": DEFAULT_MU,
        "mu_lat": DEFAULT_MU,
        "rolling_resistance_cr": DEFAULT_ROLLING_RESISTANCE,
        **DEFAULT_PACEJKA,
    }
    out = WheelConfig()
    for f in fields(WheelConfig):
        cur = getattr(w, f.name)
        if cur is None and f.name in defaults:
            setattr(out, f.name, defaults[f.name])
        else:
            setattr(out, f.name, cur)
    return out


def _warn_if_undersprung(
    parsed: Any, wheels: list[WheelConfig], urdf_path: str, gravity: float = 9.81
) -> None:
    """Warn when the resolved springs cannot hold the vehicle up. (v1.2.1)

    Static sag is ``load / k``. Once it runs past ``rest_stroke`` the vehicle
    rests beyond its own suspension travel: ride height collapses, the wheels
    have no droop left, and the hull wallows at a fraction of the intended
    frequency. That is a silent failure today — nothing in the pipeline
    complains, the vehicle just feels soft — so check it once at resolve time.

    The threshold is 1.25x, not 1.0x: ``rest_stroke`` is a ray budget rather
    than a mechanical bump stop, and the tracked preset deliberately sits at
    ~1.0 (its belt coupling, not droop travel, keeps the wheels loaded). Past
    ~1.25x there is no reading under which the spring is merely tight.
    """
    RATIO_LIMIT = 1.25
    sprung = getattr(parsed, "sprung_mass", None)
    if not sprung or not wheels:
        return
    load = sprung * gravity / len(wheels)
    worst_ratio, worst = 0.0, None
    for w in wheels:
        if not w.k_susp or not w.rest_stroke:
            continue
        ratio = (load / w.k_susp) / w.rest_stroke
        if ratio > worst_ratio:
            worst_ratio, worst = ratio, w
    if worst is None or worst_ratio <= RATIO_LIMIT:
        return
    sag = load / worst.k_susp
    k_needed = load / worst.rest_stroke
    _logger.warning(
        "%s: suspension is undersprung - %.0f kg sprung mass over %d wheel(s) "
        "is %.1f kN/wheel, which sags %.0f mm on k=%.0f N/m against a %.0f mm "
        "rest_stroke (%.1fx). The vehicle will sit below its own travel and "
        "wallow. Need k >= %.0f N/m (see suspension_from_mass()).",
        os.path.basename(urdf_path), sprung, len(wheels), load / 1000.0,
        sag * 1000.0, worst.k_susp, worst.rest_stroke * 1000.0, worst_ratio,
        k_needed,
    )


def resolve(config: VehicleConfig) -> ResolvedConfig:
    """URDF -> user-config -> module-default merge.

    1. Parse URDF to extract per-wheel positions, joint names, geometry, inertia.
    2. For each wheel in config.wheels, merge with the URDF wheel of the same name
       (user value wins if non-None; URDF fills the rest).
    3. Fill any still-None fields with module-level defaults.
    4. Run each strategy's validate() on the resolved wheel list.
    """
    from .urdf import parse_urdf

    parsed = parse_urdf(config.urdf_path)
    urdf_by_name = {w.name: w for w in parsed.wheels}

    merged: list[WheelConfig] = []
    for user_w in config.wheels:
        if user_w.name is not None and user_w.name in urdf_by_name:
            base = urdf_by_name[user_w.name]
        else:
            # No URDF match — start from a blank wheel; user-provided fields win.
            base = WheelConfig(name=user_w.name)
        m = _merge_wheel(base, user_w)
        m = _fill_defaults(m)
        if m.position is None:
            raise ConfigError(
                f"WheelConfig {m.name!r} has no position and URDF lookup failed.")
        merged.append(m)

    chassis = config.chassis
    if chassis.mass is None and parsed.chassis_mass is not None:
        chassis = replace(chassis, mass=parsed.chassis_mass)

    _warn_if_undersprung(parsed, merged, config.urdf_path)

    # Strategy validation hooks (can raise ConfigError).
    for strat in (config.steering, config.drivetrain, config.coupling):
        validator = getattr(strat, "validate", None)
        if validator is not None:
            validator(merged)

    if config.visual_susp_mode not in ("auto", "kinematic", "control"):
        raise ConfigError(
            f"visual_susp_mode must be 'auto', 'kinematic', or 'control', "
            f"got {config.visual_susp_mode!r}"
        )

    if config.susp_visual_clamp is not None and config.susp_visual_clamp <= 0:
        raise ConfigError(
            f"susp_visual_clamp must be > 0 (metres) or None (auto), "
            f"got {config.susp_visual_clamp!r}"
        )

    return ResolvedConfig(
        wheels=merged,
        chassis=chassis,
        steering=config.steering,
        drivetrain=config.drivetrain,
        coupling=config.coupling,
        tire=config.tire,
        stability_hooks=list(config.stability_hooks),
        recommended_dt=config.recommended_dt,
        enable_wheel_joint_internal_sync=config.enable_wheel_joint_internal_sync,
        urdf=parsed,
        visual_susp_mode=config.visual_susp_mode,
        visual_spin_enabled=config.visual_spin_enabled,
        susp_visual_clamp=config.susp_visual_clamp,
    )
