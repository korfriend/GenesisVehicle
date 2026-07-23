"""Vehicle runtime parameter setters — pure Python, no Genesis.

`Vehicle.set_aero_drag` / `set_top_speed` / `set_omega_max_drive` are the
explicit, supported way to change plant parameters mid-drive (v1.2.5). They
mutate the live resolved config; here we drive them through a stub whose
``.resolved`` is a real resolved car config, so no Genesis build is needed.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.vehicle_scene import Vehicle
from genesis_vehicle.config import resolve
from genesis_vehicle.presets import car_4w_rwd_ackermann
from genesis_vehicle.units import omega_from_top_speed

_DATA = os.path.join(os.path.dirname(__file__), "data")
CAR_URDF = os.path.join(_DATA, "car_ref.urdf")


def _vehicle_with_resolved():
    """A Vehicle handle whose `.resolved` returns a real resolved car config,
    via a stub `physics` object (the property returns physics.resolved)."""
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    resolved = resolve(cfg)
    veh = Vehicle(name="car", urdf_path=CAR_URDF, cfg=cfg, wheel_positions=[],
                  pos=(0, 0, 0), quat=(1, 0, 0, 0), material=None)
    veh.physics = types.SimpleNamespace(resolved=resolved)
    return veh, resolved


def test_set_aero_drag_updates_live_config():
    veh, resolved = _vehicle_with_resolved()
    veh.set_aero_drag(drag_area=2.5, air_density=1.0)
    assert resolved.chassis.drag_area == pytest.approx(2.5)
    assert resolved.chassis.air_density == pytest.approx(1.0)


def test_set_aero_drag_only_touches_given_args():
    veh, resolved = _vehicle_with_resolved()
    rho0 = resolved.chassis.air_density
    veh.set_aero_drag(drag_area=1.0)          # air_density omitted
    assert resolved.chassis.drag_area == pytest.approx(1.0)
    assert resolved.chassis.air_density == pytest.approx(rho0)


def test_set_aero_drag_clamps_negative():
    veh, resolved = _vehicle_with_resolved()
    veh.set_aero_drag(drag_area=-5.0)
    assert resolved.chassis.drag_area == 0.0


def test_set_omega_max_drive():
    veh, resolved = _vehicle_with_resolved()
    veh.set_omega_max_drive(42.0)
    assert resolved.drivetrain.omega_max_drive == pytest.approx(42.0)
    veh.set_omega_max_drive(None)
    assert resolved.drivetrain.omega_max_drive is None


def test_set_top_speed_radius_independent():
    veh, resolved = _vehicle_with_resolved()
    veh.set_top_speed(30.0)
    r = veh._mean_wheel_radius()
    assert resolved.drivetrain.omega_max_drive == pytest.approx(
        omega_from_top_speed(30.0, r))


def test_setters_chain():
    veh, resolved = _vehicle_with_resolved()
    out = veh.set_aero_drag(drag_area=1.0).set_top_speed(20.0)
    assert out is veh
    assert resolved.chassis.drag_area == pytest.approx(1.0)
