"""Aerodynamic drag — pure Python / torch, no Genesis.

`aero_drag_force` is the chassis drag folded into the per-step chassis force
(v1.2.4). Runtime physics (the force actually slowing the vehicle) is exercised
by the samples; here we pin the formula, the config plumbing and the preset
defaults.
"""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.dynamics import aero_drag_force
from genesis_vehicle.config import ChassisConfig, resolve
from genesis_vehicle.presets import (
    CAR_DRAG_AREA, TANK_DRAG_AREA,
    car_4w_rwd_ackermann, tank_skid_belt,
)

_DATA = os.path.join(os.path.dirname(__file__), "data")
CAR_URDF = os.path.join(_DATA, "car_ref.urdf")
TANK_URDF = os.path.join(_DATA, "tank_ref.urdf")


# --- the force formula --------------------------------------------------------

def test_drag_matches_half_rho_cda_v_squared():
    vel = torch.tensor([[10.0, 0.0, 0.0]])
    F = aero_drag_force(vel, drag_area=2.0, air_density=1.2)
    # magnitude = 0.5 * 1.2 * 2.0 * 10^2 = 120, opposing +x
    assert F[0, 0].item() == pytest.approx(-120.0)
    assert F[0, 1].item() == pytest.approx(0.0)
    assert F[0, 2].item() == pytest.approx(0.0)


def test_drag_scales_with_velocity_squared():
    d = dict(drag_area=1.0, air_density=1.0)
    f1 = aero_drag_force(torch.tensor([[5.0, 0.0, 0.0]]), **d)[0, 0].item()
    f2 = aero_drag_force(torch.tensor([[10.0, 0.0, 0.0]]), **d)[0, 0].item()
    assert f2 == pytest.approx(4.0 * f1)   # 2x speed -> 4x drag


def test_drag_opposes_velocity_direction():
    F = aero_drag_force(torch.tensor([[3.0, 4.0, 0.0]]), drag_area=1.0)
    # antiparallel to (3,4): F direction = -(3,4)/5
    v = torch.tensor([3.0, 4.0, 0.0])
    cos = torch.dot(F[0], v) / (F[0].norm() * v.norm())
    assert cos.item() == pytest.approx(-1.0, abs=1e-5)


def test_drag_ignores_vertical_velocity():
    """Vertical velocity must not produce drag (never fight the suspension)."""
    F = aero_drag_force(torch.tensor([[0.0, 0.0, 9.0]]), drag_area=5.0)
    assert torch.allclose(F, torch.zeros_like(F))


def test_drag_horizontal_only_component():
    F = aero_drag_force(torch.tensor([[6.0, 0.0, 8.0]]), drag_area=1.0, air_density=1.0)
    # only the horizontal speed (6) drives drag; z is dropped
    assert F[0, 0].item() == pytest.approx(-0.5 * 1.0 * 1.0 * 6.0 * 6.0)
    assert F[0, 2].item() == pytest.approx(0.0)


def test_zero_drag_area_is_no_force():
    F = aero_drag_force(torch.tensor([[20.0, 0.0, 0.0]]), drag_area=0.0)
    assert torch.allclose(F, torch.zeros_like(F))


def test_drag_batched():
    vel = torch.tensor([[10.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    F = aero_drag_force(vel, drag_area=1.0, air_density=1.0)
    assert F[0, 0].item() == pytest.approx(-50.0)
    assert F[1, 1].item() == pytest.approx(-12.5)


# --- config + preset plumbing -------------------------------------------------

def test_chassis_config_drag_defaults_off():
    ch = ChassisConfig()
    assert ch.drag_area == 0.0            # back-compat: no drag by default
    assert ch.air_density == pytest.approx(1.225)


def test_car_preset_has_drag_tank_does_not():
    car = resolve(car_4w_rwd_ackermann(CAR_URDF))
    tank = resolve(tank_skid_belt(TANK_URDF))
    assert car.chassis.drag_area == pytest.approx(CAR_DRAG_AREA)
    assert car.chassis.drag_area > 0.0
    assert tank.chassis.drag_area == pytest.approx(TANK_DRAG_AREA)
    assert tank.chassis.drag_area == 0.0


def test_drag_area_kwarg_overrides_and_survives_resolve():
    cfg = car_4w_rwd_ackermann(CAR_URDF, drag_area=1.5)
    assert cfg.chassis.drag_area == pytest.approx(1.5)
    assert resolve(cfg).chassis.drag_area == pytest.approx(1.5)


def test_chassis_is_mutable_for_runtime_tuning():
    """The sample tunes drag live; the pipeline reads resolved.chassis each
    step, so the dataclass must stay mutable (not frozen)."""
    ch = resolve(car_4w_rwd_ackermann(CAR_URDF)).chassis
    ch.drag_area = 3.0
    assert ch.drag_area == pytest.approx(3.0)
