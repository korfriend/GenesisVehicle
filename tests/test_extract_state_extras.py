"""extract_state / extract_state_from_arrays: the two keys added in 2.0.0.

``yaw_rate`` = world z angular velocity; ``v_lat`` = ``-vx*sin(yaw) +
vy*cos(yaw)`` (+ = left). The five 1.x keys are unchanged bit for bit, and an
object without ``get_ang()`` still works (the two keys are then absent).

Mutants this file must fail on (plan section 11): the ``v_lat`` sign flipped;
``get_ang()`` called unconditionally.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control import extract_state, extract_state_from_arrays  # noqa: E402


def _state_1_8_0(pos, q, vel):
    """The 1.8.0 extract_state_from_arrays body, copied verbatim."""
    w = float(q[0]); x = float(q[1]); y = float(q[2]); z = float(q[3])
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    cy, sy = math.cos(yaw), math.sin(yaw)
    v_long = float(vel[0]) * cy + float(vel[1]) * sy
    return {"pos_xy": (float(pos[0]), float(pos[1])), "yaw": yaw, "v_long": v_long,
            "pitch": math.degrees(pitch), "roll": math.degrees(roll)}


def _quat(yaw, pitch=0.0, roll=0.0):
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def test_five_keys_unchanged_and_two_added():
    rng = np.random.default_rng(7)
    for _ in range(200):
        pos, vel, ang = rng.normal(size=3), rng.normal(size=3) * 5, rng.normal(size=3)
        q = _quat(*rng.uniform(-1, 1, 3))
        old = _state_1_8_0(pos, q, vel)
        new = extract_state_from_arrays(pos, q, vel, ang)
        assert {k: new[k] for k in old} == old
        assert set(new) == set(old) | {"yaw_rate", "v_lat"}
        assert extract_state_from_arrays(pos, q, vel) == old


def test_yaw_rate_and_v_lat_definitions():
    yaw = math.pi / 2                                 # facing +Y
    st = extract_state_from_arrays((0, 0, 0), _quat(yaw), (-1.5, 2.0, 0.0), (0.1, 0.2, 0.3))
    assert st["yaw_rate"] == 0.3
    assert st["v_long"] == pytest.approx(2.0)
    assert st["v_lat"] == pytest.approx(1.5)          # moving -X while facing +Y = to the left


class _Duck:
    def __init__(self, with_ang):
        self.with_ang = with_ang

    def get_pos(self):
        return np.array([1.0, 2.0, 0.0])

    def get_quat(self):
        return np.array(_quat(0.0))

    def get_vel(self):
        return np.array([0.0, 1.0, 0.0])

    def __getattr__(self, name):
        if name == "get_ang" and self.with_ang:
            return lambda: np.array([0.0, 0.0, 0.25])
        raise AttributeError(name)


def test_duck_typed_vehicle_without_get_ang():
    st = extract_state(_Duck(False))
    assert set(st) == {"pos_xy", "yaw", "v_long", "pitch", "roll"}
    st = extract_state(_Duck(True))
    assert st["yaw_rate"] == 0.25 and st["v_lat"] == 1.0
