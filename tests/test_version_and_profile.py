"""Tests for version reporting + stability profile semantics (v0.3.0)."""

from __future__ import annotations

import os

import pytest

import genesis_vehicle
from genesis_vehicle import (
    FWD,
    LowSpeedRegularizer,
    RollingResistance,
    StaticFrictionLock,
    car_4w_fwd_ackermann,
    car_4w_rwd_ackermann,
    stability_hooks_for_profile,
    tank_10w_skid_belt,
)
from genesis_vehicle.config import ConfigError


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HJW_URDF = os.path.join(REPO_ROOT, "HJW", "urdf", "car_raywheel.urdf")
KDU_URDF = os.path.join(REPO_ROOT, "KDU", "tank_ray.urdf")


# ---------------------------------------------------------------------------
# Version API
# ---------------------------------------------------------------------------


def test_version_string_format():
    v = genesis_vehicle.__version__
    parts = v.split(".")
    assert len(parts) == 3, f"expected major.minor.patch, got {v!r}"
    for p in parts:
        int(p)   # must be integers


def test_version_function_matches_attribute():
    assert genesis_vehicle.version() == genesis_vehicle.__version__


def test_version_info_matches_string():
    v_tuple = genesis_vehicle.version_info()
    v_string = genesis_vehicle.__version__
    assert v_string == ".".join(str(x) for x in v_tuple)


# ---------------------------------------------------------------------------
# stability_hooks_for_profile
# ---------------------------------------------------------------------------


def test_profile_control_car_returns_two_hooks():
    hooks = stability_hooks_for_profile("control", vehicle_kind="car")
    types = [type(h).__name__ for h in hooks]
    assert types == ["RollingResistance", "LowSpeedRegularizer"]


def test_profile_control_tank_adds_static_friction_lock():
    hooks = stability_hooks_for_profile("control", vehicle_kind="tank")
    types = [type(h).__name__ for h in hooks]
    assert types == ["RollingResistance", "LowSpeedRegularizer", "StaticFrictionLock"]


def test_profile_control_disables_regularizer_under_throttle():
    """The 'control' profile builds LowSpeedRegularizer with
    disable_when_control_active=True so the vehicle can start from rest under
    throttle (see CHANGELOG v0.5.1 — reverts the v0.3.0 setting that froze
    every preset vehicle at rest)."""
    hooks = stability_hooks_for_profile("control", vehicle_kind="car")
    lsr = next(h for h in hooks if isinstance(h, LowSpeedRegularizer))
    assert lsr.disable_when_control_active is True


def test_profile_raw_returns_empty():
    assert stability_hooks_for_profile("raw", vehicle_kind="car") == []
    assert stability_hooks_for_profile("raw", vehicle_kind="tank") == []


def test_profile_research_returns_empty():
    assert stability_hooks_for_profile("research", vehicle_kind="car") == []


def test_profile_unknown_raises():
    with pytest.raises(ConfigError):
        stability_hooks_for_profile("bad", vehicle_kind="car")


# ---------------------------------------------------------------------------
# Preset integration
# ---------------------------------------------------------------------------


def test_preset_car_default_profile_is_control():
    cfg = car_4w_rwd_ackermann(HJW_URDF)
    types = [type(h).__name__ for h in cfg.stability_hooks]
    assert "RollingResistance" in types
    assert "LowSpeedRegularizer" in types


def test_preset_car_raw_profile_has_no_hooks():
    cfg = car_4w_rwd_ackermann(HJW_URDF, stability="raw")
    assert cfg.stability_hooks == []


def test_preset_tank_default_includes_static_friction_lock():
    cfg = tank_10w_skid_belt(KDU_URDF)
    types = [type(h).__name__ for h in cfg.stability_hooks]
    assert "StaticFrictionLock" in types


def test_preset_tank_raw_profile_has_no_hooks():
    cfg = tank_10w_skid_belt(KDU_URDF, stability="raw")
    assert cfg.stability_hooks == []


def test_preset_car_fwd_uses_fwd_drivetrain():
    """FWD preset must wire up an FWD drivetrain (not RWD/AWD)."""
    cfg = car_4w_fwd_ackermann(HJW_URDF)
    assert isinstance(cfg.drivetrain, FWD), \
        f"expected FWD, got {type(cfg.drivetrain).__name__}"


def test_preset_car_fwd_default_profile_has_hooks():
    cfg = car_4w_fwd_ackermann(HJW_URDF)
    types = [type(h).__name__ for h in cfg.stability_hooks]
    assert "RollingResistance" in types
    assert "LowSpeedRegularizer" in types
