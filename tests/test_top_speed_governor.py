"""Top-speed governor + unit helpers — pure Python / torch, no Genesis.

The drive-omega cap is the de-facto top-speed governor (v1.2.3): with
negligible drag a driven wheel spins to the cap, so top speed = cap x radius.
Presets set it from a target top speed (m/s) so the speed is radius-independent.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.units import kmh_to_mps, mps_to_kmh, omega_from_top_speed
from genesis_vehicle.strategies import RWD, AWD
from genesis_vehicle.presets import (
    CAR_TOP_SPEED, TANK_TOP_SPEED,
    car_4w_rwd_ackermann, car_4w_awd_ackermann, tank_skid_belt,
)

_DATA = os.path.join(os.path.dirname(__file__), "data")
CAR_URDF = os.path.join(_DATA, "car_ref.urdf")
TANK_URDF = os.path.join(_DATA, "tank_ref.urdf")   # wheels radius 0.4 m


# --- unit helpers -------------------------------------------------------------

def test_kmh_mps_roundtrip():
    assert kmh_to_mps(200.0) == pytest.approx(55.5556, rel=1e-4)
    assert mps_to_kmh(18.6) == pytest.approx(66.96, rel=1e-4)
    assert mps_to_kmh(kmh_to_mps(72.0)) == pytest.approx(72.0)


def test_omega_from_top_speed():
    assert omega_from_top_speed(18.6, 0.33) == pytest.approx(18.6 / 0.33)
    with pytest.raises(ValueError):
        omega_from_top_speed(10.0, 0.0)


# --- rev-limiter taper --------------------------------------------------------

def _rwd():
    return RWD(t_drive_max=1000.0, t_brake_max=1000.0, omega_max_drive=50.0)


def test_rev_limit_untouched_below_cap():
    d = _rwd()
    T = torch.tensor([[100.0, 100.0]])
    omega = torch.zeros(1, 2)                       # at rest -> full torque
    out = d._rev_limit(T, omega)
    assert torch.allclose(out, T)


def test_rev_limit_tapers_to_zero_at_cap():
    d = _rwd()
    T = torch.tensor([[100.0, 100.0]])
    omega = torch.tensor([[50.0, 50.0]])            # exactly at the cap
    out = d._rev_limit(T, omega)
    assert torch.allclose(out, torch.zeros_like(T), atol=1e-5)


def test_rev_limit_half_at_half_cap():
    d = _rwd()
    T = torch.tensor([[100.0]])
    omega = torch.tensor([[25.0]])                  # half the cap -> half torque
    out = d._rev_limit(T, omega)
    assert out.item() == pytest.approx(50.0, rel=1e-4)


def test_rev_limit_keeps_engine_braking():
    """Drive torque opposing rotation (engine braking) is NOT tapered."""
    d = _rwd()
    T = torch.tensor([[-100.0]])                    # torque backward
    omega = torch.tensor([[50.0]])                  # wheel spinning forward at cap
    out = d._rev_limit(T, omega)
    assert out.item() == pytest.approx(-100.0)


def test_rev_limit_noop_when_uncapped():
    d = RWD(t_drive_max=1.0, t_brake_max=1.0)       # omega_max_drive None
    T = torch.tensor([[100.0]])
    omega = torch.tensor([[9999.0]])
    assert torch.allclose(d._rev_limit(T, omega), T)


# --- preset governor (radius-independent) -------------------------------------

def _mean_radius(urdf):
    from genesis_vehicle.urdf import parse_urdf
    from genesis_vehicle.config import DEFAULT_RADIUS
    rr = [w.radius for w in parse_urdf(urdf).wheels if w.radius]
    return sum(rr) / len(rr) if rr else DEFAULT_RADIUS


def test_car_preset_caps_at_200_kmh():
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    v_top = cfg.drivetrain.omega_max_drive * _mean_radius(CAR_URDF)
    assert v_top == pytest.approx(CAR_TOP_SPEED, rel=1e-6)
    assert mps_to_kmh(v_top) == pytest.approx(200.0, rel=0.01)


def test_awd_preset_also_caps():
    cfg = car_4w_awd_ackermann(CAR_URDF)
    assert cfg.drivetrain.omega_max_drive is not None


def test_tank_preset_caps_at_realistic_speed():
    cfg = tank_skid_belt(TANK_URDF)
    v_top = cfg.drivetrain.omega_max_drive * _mean_radius(TANK_URDF)
    assert v_top == pytest.approx(TANK_TOP_SPEED, rel=1e-6)


def test_top_speed_is_radius_independent():
    """Same target speed on two different wheel radii -> same top speed, even
    though the omega caps differ."""
    from genesis_vehicle.urdf import parse_urdf
    car = car_4w_rwd_ackermann(CAR_URDF, top_speed=30.0)
    tank = tank_skid_belt(TANK_URDF, top_speed=30.0)
    v_car = car.drivetrain.omega_max_drive * _mean_radius(CAR_URDF)
    v_tank = tank.drivetrain.omega_max_drive * _mean_radius(TANK_URDF)
    assert v_car == pytest.approx(30.0, rel=1e-6)
    assert v_tank == pytest.approx(30.0, rel=1e-6)
    # the omega caps themselves differ because the radii differ
    assert car.drivetrain.omega_max_drive != pytest.approx(
        tank.drivetrain.omega_max_drive)


def test_top_speed_kwarg_overrides_default():
    slow = tank_skid_belt(TANK_URDF, top_speed=10.0)
    fast = tank_skid_belt(TANK_URDF, top_speed=20.0)
    assert fast.drivetrain.omega_max_drive == pytest.approx(
        2.0 * slow.drivetrain.omega_max_drive)
