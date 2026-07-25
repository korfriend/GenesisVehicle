"""Bump-stop spring — pure Python / torch, no Genesis.

`bump_stop_force` bounds the transient over-compression a penalty contact needs
on a slope hit / landing (v1.2.6). Slope-entry effectiveness and the
dt-stability bound are exercised by measurement (see CHANGELOG); here we pin
the formula and the config/preset plumbing.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.dynamics import bump_stop_force
from genesis_vehicle.config import DEFAULT_K_BUMP, resolve
from genesis_vehicle.presets import (
    TANK_BUMP_FACTOR, car_4w_rwd_ackermann, tank_skid_belt,
)

_DATA = os.path.join(os.path.dirname(__file__), "data")
CAR_URDF = os.path.join(_DATA, "car_ref.urdf")
TANK_URDF = os.path.join(_DATA, "tank_ref.urdf")


# --- the force term -----------------------------------------------------------

def test_zero_inside_the_stroke():
    c = torch.tensor([[0.0, 0.02, 0.05]])          # at/below stroke
    rs = torch.full((1, 3), 0.05)
    kb = torch.full((1, 3), 1.0e6)
    assert torch.allclose(bump_stop_force(c, rs, kb), torch.zeros(1, 3))


def test_linear_beyond_the_stroke():
    c = torch.tensor([[0.07]])                     # 20 mm past a 50 mm stroke
    rs = torch.tensor([[0.05]])
    kb = torch.tensor([[1.0e6]])
    F = bump_stop_force(c, rs, kb)
    assert F.item() == pytest.approx(1.0e6 * 0.02)


def test_zero_when_disabled():
    c = torch.tensor([[0.30]])                     # deep over-compression
    F = bump_stop_force(c, torch.tensor([[0.05]]), torch.tensor([[0.0]]))
    assert F.item() == 0.0


def test_batched_broadcast():
    c = torch.tensor([[0.04, 0.06], [0.10, 0.05]])
    rs = torch.full((1, 2), 0.05)
    kb = torch.full((1, 2), 2.0e6)
    F = bump_stop_force(c, rs, kb)
    assert F[0, 0].item() == 0.0
    assert F[0, 1].item() == pytest.approx(2.0e6 * 0.01)
    assert F[1, 0].item() == pytest.approx(2.0e6 * 0.05)
    assert F[1, 1].item() == 0.0


# --- config / preset plumbing ---------------------------------------------------

def test_default_is_off():
    assert DEFAULT_K_BUMP == 0.0
    resolved = resolve(car_4w_rwd_ackermann(CAR_URDF))
    for w in resolved.wheels:
        assert w.k_bump == 0.0                     # cars: no bump-stop yet


def test_tank_preset_enables_bump_stop_at_factor_x_spring():
    resolved = resolve(tank_skid_belt(TANK_URDF))
    for w in resolved.wheels:
        assert w.k_bump == pytest.approx(TANK_BUMP_FACTOR * w.k_susp)
        assert w.k_bump > 0.0


def test_tank_bump_factor_is_within_the_dt_stability_bound():
    """(k_susp + k_bump) * dt^2 / m_share must stay well under the measured
    divergence threshold (~0.86 diverges, 0.61 chatters) at the recommended dt.
    Guards against someone raising TANK_BUMP_FACTOR without re-checking."""
    from genesis_vehicle.urdf import parse_urdf
    resolved = resolve(tank_skid_belt(TANK_URDF))
    parsed = parse_urdf(TANK_URDF)
    m_share = parsed.sprung_mass / len(resolved.wheels)
    dt = resolved.recommended_dt
    w = resolved.wheels[0]
    ratio = (w.k_susp + w.k_bump) * dt * dt / m_share
    assert ratio < 0.5, f"stability ratio {ratio:.2f} too close to divergence"


def test_sweep_cli_k_bump_override():
    from genesis_vehicle.control.sweep_measure import apply_plant_overrides
    cfg = tank_skid_belt(TANK_URDF)
    apply_plant_overrides(cfg, TANK_URDF, k_bump=0.0, log=lambda *_: None)
    assert all(w.k_bump == 0.0 for w in cfg.wheels)   # explicit off wins
