"""Configuration dataclasses and the URDF -> config -> default merge algorithm.

WheelConfig fields are all Optional; resolve() fills None values from the
parsed URDF, then from module-level defaults.
"""

from dataclasses import dataclass, field, fields, replace
from typing import Any, Optional


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
    # does not refuse — Genesis's value wins. Tanks generally need
    # >=200 Hz (recommended_dt=0.005); cars are fine at 100 Hz.
    recommended_dt: float = 1.0 / 48.0
    # Drive the URDF wheel visual joints (spin/steer/suspension) through the
    # engine each step so the **Genesis native viewer** shows wheels moving.
    # Default False: the dominant path (headless / external UE·Unity renderer)
    # does NOT need it — read poses from wheel_visual_transforms() (closed-form,
    # ~µs) instead, which works regardless of this flag. Set True ONLY when you
    # actually open the Genesis viewer (`gs.Scene(show_viewer=True)` or a sample
    # `--viewer`). Leaving it on headless is a silent ~ms/step perf trap (it is
    # the dominant SDK cost at scale). See VisualJointSync / docs/server.md.
    enable_visual_joint_sync: bool = False

    # Visual suspension joint mode:
    #   "auto"       — per-joint decision based on URDF <dynamics> presence (legacy default)
    #   "kinematic"  — force set_dofs_position (HJW-style, light wheels < ~50kg)
    #   "control"    — force control_dofs_position with high kp/kv (KDU-style, heavy wheels)
    # Heavy wheels (e.g. KDU tank's 500 kg) need PD control or they fall under
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
    # visually travel from rest in wheel_visual_transforms / VisualJointSync — a
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
    enable_visual_joint_sync: bool
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
        enable_visual_joint_sync=config.enable_visual_joint_sync,
        urdf=parsed,
        visual_susp_mode=config.visual_susp_mode,
        visual_spin_enabled=config.visual_spin_enabled,
        susp_visual_clamp=config.susp_visual_clamp,
    )
