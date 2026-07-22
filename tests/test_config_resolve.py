"""Pure-Python tests for the URDF -> user-config -> default merge algorithm."""

from __future__ import annotations

import logging
import math
import os

import pytest

from genesis_vehicle.config import (
    DEFAULT_C_COMPRESSION,
    DEFAULT_C_EXTENSION,
    DEFAULT_K_SUSP,
    DEFAULT_PACEJKA,
    ChassisConfig,
    ConfigError,
    VehicleConfig,
    WheelConfig,
    resolve,
    suspension_from_mass,
)
from genesis_vehicle.strategies import (
    Ackermann,
    Independent,
    LowSpeedRegularizer,
    RWD,
    SameSideBelt,
    SkidSteer,
    PerSide,
)
from genesis_vehicle.tire_models import PacejkaAnisotropic


_DATA = os.path.join(os.path.dirname(__file__), "data")
CAR_URDF = os.path.join(_DATA, "car_ref.urdf")
TANK_URDF = os.path.join(_DATA, "tank_ref.urdf")


def _car_basic_config(**kwargs) -> VehicleConfig:
    return VehicleConfig.from_urdf(
        CAR_URDF,
        steering=Ackermann(max_steer_rad=0.7, front_axle=0),
        drivetrain=RWD(t_drive_max=1000.0, t_brake_max=2500.0, driven_axles=(1,)),
        coupling=Independent(),
        tire=PacejkaAnisotropic(),
        **kwargs,
    )


def test_from_urdf_populates_wheels():
    cfg = _car_basic_config()
    assert len(cfg.wheels) == 4
    names = {w.name for w in cfg.wheels}
    assert names == {
        "front_left_wheel", "front_right_wheel",
        "rear_left_wheel", "rear_right_wheel",
    }


def test_resolve_fills_defaults_for_missing_fields():
    cfg = _car_basic_config()
    resolved = resolve(cfg)
    assert len(resolved.wheels) == 4
    for w in resolved.wheels:
        # All fields that resolve() promises to fill should be non-None.
        assert w.position is not None
        assert w.radius is not None
        assert w.k_susp == pytest.approx(DEFAULT_K_SUSP)
        assert w.c_compression == pytest.approx(DEFAULT_C_COMPRESSION)
        assert w.c_extension == pytest.approx(DEFAULT_C_EXTENSION)
        for k, v in DEFAULT_PACEJKA.items():
            assert getattr(w, k) == pytest.approx(v)


def test_per_wheel_override_wins_over_urdf():
    overrides = {
        "front_left_wheel": WheelConfig(k_susp=12345.0, mu_long=0.42),
    }
    cfg = _car_basic_config(wheel_overrides=overrides)
    resolved = resolve(cfg)
    by_name = {w.name: w for w in resolved.wheels}
    assert by_name["front_left_wheel"].k_susp == pytest.approx(12345.0)
    assert by_name["front_left_wheel"].mu_long == pytest.approx(0.42)
    # Other wheels still use the default.
    assert by_name["front_right_wheel"].k_susp == pytest.approx(DEFAULT_K_SUSP)


def test_chassis_mass_from_urdf_when_not_supplied():
    cfg = _car_basic_config()
    assert cfg.chassis.mass == pytest.approx(2200.0)


def test_chassis_mass_user_override_wins():
    cfg = _car_basic_config(chassis=ChassisConfig(mass=9999.0))
    resolved = resolve(cfg)
    assert resolved.chassis.mass == pytest.approx(9999.0)


def test_skidsteer_strategy_validate_passes_for_tank():
    cfg = VehicleConfig.from_urdf(
        TANK_URDF,
        steering=SkidSteer(),
        drivetrain=PerSide(t_drive_max=30000.0, t_brake_max=30000.0),
        coupling=SameSideBelt(),
        tire=PacejkaAnisotropic(),
    )
    # Should not raise.
    resolve(cfg)


def test_skidsteer_strategy_validate_fails_for_car():
    # The reference car has front steer joints — SkidSteer must reject it.
    cfg = VehicleConfig.from_urdf(
        CAR_URDF,
        steering=SkidSteer(),
        drivetrain=PerSide(t_drive_max=1000.0, t_brake_max=2500.0),
        coupling=SameSideBelt(),
        tire=PacejkaAnisotropic(),
    )
    with pytest.raises(ConfigError):
        resolve(cfg)


def test_resolve_preserves_stability_hooks_and_dt():
    hooks = [LowSpeedRegularizer()]
    cfg = _car_basic_config(stability_hooks=hooks, recommended_dt=0.01)
    resolved = resolve(cfg)
    assert resolved.recommended_dt == pytest.approx(0.01)
    assert len(resolved.stability_hooks) == 1
    assert isinstance(resolved.stability_hooks[0], LowSpeedRegularizer)


def test_user_explicit_i_wheel_wins_over_urdf():
    """WheelConfig.i_wheel set by the user must survive resolve() unchanged.
    URDF iyy is only a fallback estimate (contract section 4 of API.md)."""
    overrides = {
        "front_left_wheel": WheelConfig(i_wheel=42.0),
    }
    cfg = _car_basic_config(wheel_overrides=overrides)
    resolved = resolve(cfg)
    by_name = {w.name: w for w in resolved.wheels}
    # User-set wheel keeps 42.0
    assert by_name["front_left_wheel"].i_wheel == pytest.approx(42.0)
    # Other wheels fall back to the URDF-derived value (iyy = 2.348).
    assert by_name["rear_right_wheel"].i_wheel == pytest.approx(2.348)


def test_user_explicit_radius_wins_over_urdf():
    """Same contract for radius: explicit user value wins over URDF cylinder."""
    overrides = {
        "front_left_wheel": WheelConfig(radius=0.50),
    }
    cfg = _car_basic_config(wheel_overrides=overrides)
    resolved = resolve(cfg)
    by_name = {w.name: w for w in resolved.wheels}
    assert by_name["front_left_wheel"].radius == pytest.approx(0.50)
    # The car URDF wheel mesh has no cylinder element -> URDF radius = None ->
    # default fallback. Other wheels fall back to the module default.
    other = by_name["rear_right_wheel"].radius
    assert other is not None


# ---------------------------------------------------------------------------
# v1.2.1 — mass-derived suspension + silent-failure guards
# ---------------------------------------------------------------------------


@pytest.fixture
def gv_caplog(caplog):
    """caplog wired to the SDK logger.

    `genesis_vehicle/__init__.py` installs its own handler and sets
    `propagate = False`, so records never reach the root logger caplog attaches
    to. Attach caplog's handler directly for the duration of the test.
    """
    logger = logging.getLogger("genesis_vehicle")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="genesis_vehicle"):
            yield caplog
    finally:
        logger.removeHandler(caplog.handler)


def test_suspension_from_mass_matches_the_documented_formula():
    k, cc, ce = suspension_from_mass(
        4000.0, 4, target_sag=0.05, zeta_compression=0.7, zeta_extension=0.45)
    assert k == pytest.approx(1000.0 * 9.81 / 0.05)
    c_crit = 2.0 * math.sqrt(k * 1000.0)
    assert cc == pytest.approx(0.70 * c_crit)
    assert ce == pytest.approx(0.45 * c_crit)


def test_suspension_from_mass_holds_sag_across_vehicle_scale():
    """The whole point: 20x the mass gets 20x the spring, so sag is unchanged."""
    k_light, _, _ = suspension_from_mass(2000.0, 4)
    k_heavy, _, _ = suspension_from_mass(40000.0, 4)
    assert k_heavy == pytest.approx(20.0 * k_light)


def test_suspension_from_mass_rejects_nonsense():
    with pytest.raises(ConfigError):
        suspension_from_mass(0.0, 4)
    with pytest.raises(ConfigError):
        suspension_from_mass(1000.0, 0)
    with pytest.raises(ConfigError):
        suspension_from_mass(1000.0, 4, target_sag=0.0)


def test_unmatched_wheel_override_key_warns(gv_caplog):
    """A key matching no wheel used to vanish silently, taking a whole tuning
    block with it (the v1.2.1 tracked-preset bug)."""
    _car_basic_config(wheel_overrides={"no_such_wheel": WheelConfig(k_susp=1.0)})
    assert "matched no wheel" in gv_caplog.text
    assert "no_such_wheel" in gv_caplog.text
    # The message must name the wheels the URDF actually has, so the fix is
    # obvious from the log alone.
    assert "front_left_wheel" in gv_caplog.text


def test_matched_wheel_override_key_does_not_warn(gv_caplog):
    _car_basic_config(wheel_overrides={"front_left_wheel": WheelConfig(k_susp=1.0)})
    assert "matched no wheel" not in gv_caplog.text


def test_undersprung_suspension_warns(gv_caplog):
    """Static sag past 1.25x rest_stroke = the vehicle rests below its own
    travel. Warn instead of merely feeling soft."""
    overrides = {n: WheelConfig(k_susp=100.0, rest_stroke=0.1)
                 for n in ("front_left_wheel", "front_right_wheel",
                           "rear_left_wheel", "rear_right_wheel")}
    resolve(_car_basic_config(wheel_overrides=overrides))
    assert "undersprung" in gv_caplog.text


def test_adequately_sprung_suspension_does_not_warn(gv_caplog):
    resolve(_car_basic_config())
    assert "undersprung" not in gv_caplog.text


def test_tank_preset_derives_suspension_from_urdf_mass(gv_caplog):
    """tank_skid_belt must key its overrides by THIS URDF's wheel names and
    size the spring from THIS URDF's sprung mass."""
    from genesis_vehicle.presets import TANK_REST_STROKE, tank_skid_belt
    from genesis_vehicle.urdf import parse_urdf

    resolved = resolve(tank_skid_belt(TANK_URDF))
    assert "matched no wheel" not in gv_caplog.text      # every key must land
    assert "undersprung" not in gv_caplog.text

    parsed = parse_urdf(TANK_URDF)
    k_expected, _, _ = suspension_from_mass(
        parsed.sprung_mass, len(parsed.wheels), target_sag=0.05)
    for w in resolved.wheels:
        assert w.k_susp == pytest.approx(k_expected)
        assert w.rest_stroke == pytest.approx(TANK_REST_STROKE)
        # Symmetric damper — the tracked-vehicle convention.
        assert w.c_compression == pytest.approx(w.c_extension)
    # Reproduces the literal it replaced (1.0e6) to within 5%.
    assert resolved.wheels[0].k_susp == pytest.approx(1_000_000.0, rel=0.05)


def test_urdf_declared_suspension_outranks_the_preset(tmp_path):
    """Priority chain: caller override > URDF <dynamics> > mass-derived.
    A URDF that declares its own spring must not be overwritten by the
    preset's derived value. (v1.2.1)"""
    from genesis_vehicle.presets import tank_skid_belt

    src = open(TANK_URDF).read().replace(
        '<axis xyz="0 0 1"/>',
        '<axis xyz="0 0 1"/><dynamics stiffness="250000" damping="9000"/>')
    path = tmp_path / "declared_tank.urdf"
    path.write_text(src)

    resolved = resolve(tank_skid_belt(str(path)))
    for w in resolved.wheels:
        assert w.k_susp == pytest.approx(250_000.0)     # URDF wins
        assert w.c_compression == pytest.approx(9_000.0)
    # ...but a caller override still beats the URDF.
    cfg = tank_skid_belt(str(path))
    for w in cfg.wheels:
        w.k_susp = 111_000.0
    assert resolve(cfg).wheels[0].k_susp == pytest.approx(111_000.0)


def test_tank_preset_target_sag_scales_the_spring():
    from genesis_vehicle.presets import tank_skid_belt

    soft = resolve(tank_skid_belt(TANK_URDF, target_sag=0.10))
    stiff = resolve(tank_skid_belt(TANK_URDF, target_sag=0.05))
    assert stiff.wheels[0].k_susp == pytest.approx(2.0 * soft.wheels[0].k_susp)


def test_tank_preset_keeps_urdf_wheel_radius():
    """The hard-coded radius=0.4 override is gone; URDF geometry wins."""
    from genesis_vehicle.presets import tank_skid_belt
    from genesis_vehicle.urdf import parse_urdf

    resolved = resolve(tank_skid_belt(TANK_URDF))
    urdf_radius = {w.name: w.radius for w in parse_urdf(TANK_URDF).wheels}
    for w in resolved.wheels:
        assert w.radius == pytest.approx(urdf_radius[w.name])
