"""CLI plant-override plumbing for sweep_measure — numpy/parse only, no Genesis.

`apply_plant_overrides` lets a user parameterise the sweep plant from the
command line instead of a --config file (v1.2.2). It runs on a preset-built
VehicleConfig before build, so it is testable without a Genesis runtime.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control.sweep_measure import apply_plant_overrides
from genesis_vehicle.presets import tank_skid_belt

_DATA = os.path.join(os.path.dirname(__file__), "data")
TANK_URDF = os.path.join(_DATA, "tank_ref.urdf")  # wheels radius 0.4 m


def _cfg():
    return tank_skid_belt(TANK_URDF)


def test_top_speed_sets_omega_from_mean_radius():
    cfg = _cfg()
    apply_plant_overrides(cfg, TANK_URDF, top_speed=20.0, log=lambda *_: None)
    # tank_ref wheels are radius 0.4 -> omega = 20 / 0.4 = 50 rad/s
    assert cfg.drivetrain.omega_max_drive == pytest.approx(50.0, rel=1e-3)


def test_omega_max_drive_sets_cap_directly():
    cfg = _cfg()
    apply_plant_overrides(cfg, TANK_URDF, omega_max_drive=42.0, log=lambda *_: None)
    assert cfg.drivetrain.omega_max_drive == pytest.approx(42.0)


def test_top_speed_wins_over_omega_max_drive():
    cfg = _cfg()
    apply_plant_overrides(cfg, TANK_URDF, top_speed=20.0, omega_max_drive=999.0,
                          log=lambda *_: None)
    assert cfg.drivetrain.omega_max_drive == pytest.approx(50.0, rel=1e-3)


def test_only_passed_flags_are_applied():
    """Suspension must stay whatever the preset produced (None on the cfg ->
    mass-derived at resolve) unless --k-susp is explicitly given."""
    cfg = _cfg()
    before_k = [w.k_susp for w in cfg.wheels]
    before_i = [w.i_wheel for w in cfg.wheels]
    apply_plant_overrides(cfg, TANK_URDF, mu_lat=0.5, log=lambda *_: None)
    assert all(w.mu_lat == pytest.approx(0.5) for w in cfg.wheels)
    assert [w.k_susp for w in cfg.wheels] == before_k     # untouched
    assert [w.i_wheel for w in cfg.wheels] == before_i     # untouched


def test_per_wheel_and_drivetrain_overrides_apply():
    cfg = _cfg()
    apply_plant_overrides(
        cfg, TANK_URDF, i_wheel=100.0, mu_long=1.0, mu_lat=0.5,
        k_susp=500000.0, rest_stroke=0.06, brake_max=200000.0,
        log=lambda *_: None)
    for w in cfg.wheels:
        assert w.i_wheel == pytest.approx(100.0)
        assert w.mu_long == pytest.approx(1.0)
        assert w.mu_lat == pytest.approx(0.5)
        assert w.k_susp == pytest.approx(500000.0)
        assert w.rest_stroke == pytest.approx(0.06)
    assert cfg.drivetrain.t_brake_max == pytest.approx(200000.0)


def test_no_overrides_is_a_noop():
    cfg = _cfg()
    before = (cfg.drivetrain.omega_max_drive, cfg.drivetrain.t_brake_max,
              [w.mu_lat for w in cfg.wheels])
    apply_plant_overrides(cfg, TANK_URDF, log=lambda *_: None)
    after = (cfg.drivetrain.omega_max_drive, cfg.drivetrain.t_brake_max,
             [w.mu_lat for w in cfg.wheels])
    assert before == after
