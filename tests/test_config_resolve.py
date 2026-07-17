"""Pure-Python tests for the URDF -> user-config -> default merge algorithm."""

from __future__ import annotations

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
