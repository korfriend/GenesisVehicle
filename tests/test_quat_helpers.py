"""Pure-Python tests for the quaternion helpers used by
``VehiclePhysics.wheel_visual_transforms`` (closed-form wheel visual pose).

These don't need Genesis — they test the math (axis-angle build, Hamilton
product, suspension-offset formula) directly with torch.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from genesis_vehicle.core import (
    _quat_axis_angle, _quat_mul, _susp_visual_offset,
)


def _q(w, x, y, z):
    return torch.tensor([w, x, y, z], dtype=torch.float64)


def test_axis_angle_identity_at_zero():
    for ax in ("x", "y", "z"):
        q = _quat_axis_angle(ax, torch.zeros(1, dtype=torch.float64))[0]
        assert torch.allclose(q, _q(1, 0, 0, 0), atol=1e-7)


def test_axis_angle_z_90deg():
    q = _quat_axis_angle("z", torch.tensor([math.pi / 2], dtype=torch.float64))[0]
    c = math.cos(math.pi / 4)
    assert torch.allclose(q, _q(c, 0, 0, c), atol=1e-7)


def test_axis_angle_y_180deg():
    q = _quat_axis_angle("y", torch.tensor([math.pi], dtype=torch.float64))[0]
    assert torch.allclose(q, _q(0, 0, 1, 0), atol=1e-6)


def test_quat_mul_identity():
    a = _q(0.5, 0.5, 0.5, 0.5).reshape(1, 4)
    ident = _q(1, 0, 0, 0).reshape(1, 4)
    assert torch.allclose(_quat_mul(a, ident), a, atol=1e-7)
    assert torch.allclose(_quat_mul(ident, a), a, atol=1e-7)


def test_quat_mul_z_then_z_adds_angles():
    # R_z(30°) ∘ R_z(60°) == R_z(90°)
    qa = _quat_axis_angle("z", torch.tensor([math.radians(30)], dtype=torch.float64))
    qb = _quat_axis_angle("z", torch.tensor([math.radians(60)], dtype=torch.float64))
    q90 = _quat_axis_angle("z", torch.tensor([math.radians(90)], dtype=torch.float64))
    assert torch.allclose(_quat_mul(qa, qb), q90, atol=1e-7)


def test_quat_mul_noncommutative():
    qz = _quat_axis_angle("z", torch.tensor([1.0], dtype=torch.float64))
    qy = _quat_axis_angle("y", torch.tensor([1.0], dtype=torch.float64))
    assert not torch.allclose(_quat_mul(qz, qy), _quat_mul(qy, qz), atol=1e-3)


def test_susp_offset_grounded_and_air():
    mesh_r, l_susp = 0.4, 0.1
    d = torch.tensor([0.4, 0.5, 0.0, 20.0], dtype=torch.float64)  # touch, droop, air(0), air(far)
    off = _susp_visual_offset(d, mesh_r, l_susp)
    # grounded at d=mesh_r → 0 ; d=0.5 → -0.1 ; air → -l_susp
    assert abs(float(off[0]) - 0.0) < 1e-7
    assert abs(float(off[1]) - (-0.1)) < 1e-7
    assert abs(float(off[2]) - (-l_susp)) < 1e-7
    assert abs(float(off[3]) - (-l_susp)) < 1e-7


def test_susp_offset_clamped():
    # very close hit → large positive offset clamped to +0.19
    off = _susp_visual_offset(torch.tensor([0.05], dtype=torch.float64), 0.4, 0.1)
    assert abs(float(off[0]) - 0.19) < 1e-7


def test_susp_offset_negative_distance_is_overcompression_not_air():
    # v1.1.16 high-cast rays: a NEGATIVE distance means the ground is above
    # the wheel attachment point (deep over-compression) — a VALID reading.
    # It must produce the clamped compression offset, NOT the air pose
    # (-l_susp): the old d <= 1e-6 air test misclassified it.
    off = _susp_visual_offset(torch.tensor([-0.3], dtype=torch.float64), 0.4, 0.1)
    assert abs(float(off[0]) - 0.19) < 1e-7      # clamped up, not -l_susp
    # exact 0.0 (unpopulated sensor) is still air:
    off0 = _susp_visual_offset(torch.tensor([0.0], dtype=torch.float64), 0.4, 0.1)
    assert abs(float(off0[0]) - (-0.1)) < 1e-7
