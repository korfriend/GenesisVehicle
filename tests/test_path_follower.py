"""PathFollower unit + closed-loop tests — numpy-only, no Genesis needed.

The closed-loop tests drive a kinematic unicycle whose dynamics MATCH the
synthetic sweep table (a = 3*thr - 0.5*v, omega = 2*steer), so the follower
+ table inversion must steer it along the path and reach DONE.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control import PathFollower, extract_state_from_arrays
from genesis_vehicle.tests.test_sweep_table import make_table


# --- extract_state_from_arrays ------------------------------------------------

def test_extract_state_yaw_and_vlong():
    # Facing +Y (yaw 90 deg), moving +Y at 2 m/s -> v_long = +2.
    hp = math.pi / 4.0
    quat = (math.cos(hp), 0.0, 0.0, math.sin(hp))    # wxyz, yaw = pi/2
    st = extract_state_from_arrays((1.0, 2.0, 0.0), quat, (0.0, 2.0, 0.0))
    assert st["yaw"] == pytest.approx(math.pi / 2)
    assert st["v_long"] == pytest.approx(2.0)
    assert st["pos_xy"] == (1.0, 2.0)
    assert st["pitch"] == pytest.approx(0.0)
    assert st["roll"] == pytest.approx(0.0)


def test_extract_state_reverse_motion_negative_vlong():
    st = extract_state_from_arrays(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (-1.5, 0.0, 0.0))
    assert st["v_long"] == pytest.approx(-1.5)


def test_extract_state_pitch_in_degrees():
    hp = math.radians(10.0) / 2.0
    quat = (math.cos(hp), 0.0, math.sin(hp), 0.0)    # pitch +10 deg about Y
    st = extract_state_from_arrays((0, 0, 0), quat, (0, 0, 0))
    assert st["pitch"] == pytest.approx(10.0, abs=1e-6)


# --- Construction --------------------------------------------------------------

def test_min_two_waypoints():
    with pytest.raises(ValueError, match="at least 2"):
        PathFollower([(0, 0, 0, 1.0)], make_table())


def test_backward_waypoint_yaw_flipped():
    f = PathFollower([(0, 0, 0, -1.0), (10, 0, 0, -1.0)], make_table())
    # Travel direction is +X but driven backward -> chassis yaw = pi.
    assert abs(abs(f.yaws[0]) - math.pi) < 1e-9


# --- Explicit per-waypoint yaw (optional 5th element) ---------------------------

def test_explicit_yaw_used_verbatim_no_flip():
    # Backward waypoints with an explicit yaw: used as-is (no +pi flip).
    f = PathFollower([(0, 0, 0, -1.0, 0.3), (10, 0, 0, -1.0, 0.3)],
                     make_table())
    assert f.yaws[0] == pytest.approx(0.3)


def test_explicit_yaw_none_falls_back_to_tangential():
    f = PathFollower([(0, 0, 0, 2.0, None), (0, 10, 0, 2.0, None)],
                     make_table())
    assert f.yaws[0] == pytest.approx(math.pi / 2)     # tangential +Y


def test_explicit_yaw_normalized_and_mixed_tuples():
    # 4-tuples and 5-tuples mix; yaw outside [-pi, pi) is normalized.
    f = PathFollower([(0, 0, 0, 2.0, 2 * math.pi + 0.5), (10, 0, 0, 2.0)],
                     make_table())
    assert f.yaws[0] == pytest.approx(0.5)


def test_explicit_yaw_at_cusp_wins_block_end_yaw():
    fwd = _densify([(0.0, 0.0, 0.0, 1.5), (10.0, 0.0, 0.0, 1.5)])
    bwd = _densify([(10.0, 0.0, 0.0, -1.5), (5.0, 0.0, 0.0, -1.5)])
    path = fwd + bwd
    cusp_idx = len(fwd)                      # first waypoint of the bwd leg
    p = list(path[cusp_idx]); path[cusp_idx] = (p[0], p[1], p[2], p[3], -0.25)
    f = PathFollower(path, make_table())
    assert f.blocks[0][1] == cusp_idx
    assert f._block_end_yaw[0] == pytest.approx(-0.25)


def test_bad_waypoint_length_raises():
    with pytest.raises(ValueError, match="length 3"):
        PathFollower([(0, 0, 0), (10, 0, 0)], make_table())


def test_explicit_yaw_straight_reverse_closed_loop():
    # Same scenario as test_cusp_straight_line_reverse, but the backward
    # leg pins the chassis heading east (yaw=0) EXPLICITLY instead of
    # relying on the tangential+flip default — must behave identically.
    fwd = _densify([(0.0, 0.0, 0.0, 2.0), (10.0, 0.0, 0.0, 2.0)])
    bwd = [(x, y, z, s, 0.0) for (x, y, z, s) in
           _densify([(10.0, 0.0, 0.0, -1.0), (5.0, 0.0, 0.0, -1.0)])]
    fwd2 = _densify([(5.0, 0.0, 0.0, 2.0), (15.0, 0.0, 0.0, 2.0)])
    f = PathFollower(fwd + bwd + fwd2, make_table())
    x, y, v, modes, done = _run_unicycle(f, 0.0, 0.0, 0.0, max_steps=12000)
    assert done
    assert {"DRV+1", "DRV-1", "BRAKE_TRANS"} <= modes
    assert abs(y) < 1.0
    assert abs(x - 15.0) < f.arrival_goal + 0.5


# --- Closed loop over a matched kinematic unicycle ------------------------------

def _densify(corners, spacing=0.5):
    """(x, y, z, target_speed) corners -> densified waypoint list — the
    same corner format as the demo's ``densify`` (z interpolated; the
    follower ignores it)."""
    path = []
    for (x0, y0, z0, s0), (x1, y1, z1, _s1) in zip(corners[:-1], corners[1:]):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / spacing))
        for k in range(n):
            t = k / n
            path.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0),
                         z0 + t * (z1 - z0), s0))
    path.append(corners[-1])
    return path


def _run_unicycle(follower, x, y, yaw, dt=0.05, max_steps=4000):
    """Integrate the matched unicycle under follower control."""
    v = 0.0
    modes = set()
    for _ in range(max_steps):
        thr, steer, brake = follower.step((x, y), yaw, v, 0.0, 0.0)
        modes.add(follower.last_mode)
        if follower.last_mode == "DONE":
            return x, y, v, modes, True
        if brake > 0.5:
            v *= 0.5                        # strong brake toward rest
            if abs(v) < 1e-3:
                v = 0.0
        else:
            v += (3.0 * thr - 0.5 * v) * dt   # matches the synthetic table
        yaw += (2.0 * steer) * dt
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
    return x, y, v, modes, False


def test_follows_straight_path_to_done():
    path = _densify([(0.0, 0.0, 0.0, 2.0), (20.0, 0.0, 0.0, 2.0)])
    f = PathFollower(path, make_table())
    x, y, v, modes, done = _run_unicycle(f, 0.0, 0.0, 0.0)
    assert done, "never reached DONE"
    assert math.hypot(x - 20.0, y) < f.arrival_goal + 0.5
    assert abs(y) < 1.0                     # stayed on the line
    assert "DRV+1" in modes


def test_follows_l_shaped_path():
    path = _densify([(0.0, 0.0, 0.0, 2.0), (12.0, 0.0, 0.0, 2.0), (12.0, 12.0, 0.0, 2.0)])
    f = PathFollower(path, make_table())
    x, y, v, modes, done = _run_unicycle(f, 0.0, 0.0, 0.0)
    assert done
    assert math.hypot(x - 12.0, y - 12.0) < f.arrival_goal + 0.5


def test_cusp_stops_then_reverses():
    # Forward to x=10, then back up diagonally to (6, -4). The goal sits
    # 4 m off the forward leg, so DONE (proximity to the FINAL waypoint)
    # cannot trigger while still driving the forward leg.
    fwd = _densify([(0.0, 0.0, 0.0, 1.5), (10.0, 0.0, 0.0, 1.5)])
    bwd = _densify([(10.0, 0.0, 0.0, -1.5), (6.0, -4.0, 0.0, -1.5)])
    f = PathFollower(fwd + bwd, make_table())
    x, y, v, modes, done = _run_unicycle(f, 0.0, 0.0, 0.0, max_steps=8000)
    assert "BRAKE_TRANS" in modes, "cusp transition never triggered"
    assert "DRV-1" in modes, "never drove backward"
    assert done
    assert math.hypot(x - 6.0, y + 4.0) < f.arrival_goal + 0.5


def test_cusp_straight_line_reverse():
    # The original deliverable's verified scenario: forward east, reverse
    # west along the SAME line (chassis faces east throughout), forward
    # again. Regression for the backward-yaw flip fix.
    fwd = _densify([(0.0, 0.0, 0.0, 2.0), (10.0, 0.0, 0.0, 2.0)])
    bwd = _densify([(10.0, 0.0, 0.0, -1.0), (5.0, 0.0, 0.0, -1.0)])
    fwd2 = _densify([(5.0, 0.0, 0.0, 2.0), (15.0, 0.0, 0.0, 2.0)])
    f = PathFollower(fwd + bwd + fwd2, make_table())
    x, y, v, modes, done = _run_unicycle(f, 0.0, 0.0, 0.0, max_steps=12000)
    assert done
    assert {"DRV+1", "DRV-1", "BRAKE_TRANS"} <= modes
    assert abs(y) < 1.0, "left the straight line while reversing"
    assert abs(x - 15.0) < f.arrival_goal + 0.5


def test_done_immediately_when_at_goal():
    path = _densify([(0.0, 0.0, 0.0, 2.0), (10.0, 0.0, 0.0, 2.0)])
    f = PathFollower(path, make_table())
    thr, steer, brake = f.step((10.0, 0.0), 0.0, 0.0, 0.0, 0.0)
    assert f.last_mode == "DONE"
    assert (thr, steer, brake) == (0.0, 0.0, 1.0)


def test_block_end_yaw_ignores_a_backwards_cusp_hop():
    """A doubling-back path can place the reverse leg's first waypoint BEHIND
    the forward leg's last one. That trailing hop must not become the forward
    block's arrival heading — taking it verbatim asks the vehicle to spin 180
    degrees at the end of a straight leg, after which the projection can no
    longer advance and the cusp never fires. (v1.2.1 regression)"""
    corners = [(0.0, 0.0, 0.0, 2.0),
               (0.0, 10.0, 0.0, 2.0),     # drive +y
               (0.0, 9.7, 0.0, -2.0),     # reverse leg starts 0.3 m BEHIND
               (0.0, 0.0, 0.0, -2.0)]
    f = PathFollower(_densify(corners), make_table())
    assert len(f.blocks) == 2
    # Forward block must still arrive heading +y (+90 deg), not -90.
    assert f._block_end_yaw[0] == pytest.approx(math.pi / 2, abs=0.05)


def test_block_end_yaw_follows_a_genuine_turn():
    """The guard must not flatten a real corner: an L-shaped forward leg still
    ends heading along its last segment."""
    corners = [(0.0, 0.0, 0.0, 2.0),
               (10.0, 0.0, 0.0, 2.0),     # +x
               (10.0, 8.0, 0.0, 2.0),     # then +y
               (10.0, 7.7, 0.0, -2.0),
               (10.0, 0.0, 0.0, -2.0)]
    f = PathFollower(_densify(corners), make_table())
    assert f._block_end_yaw[0] == pytest.approx(math.pi / 2, abs=0.05)


def test_explicit_waypoint_yaw_still_wins_at_a_cusp():
    """The guard only supplies a fallback heading — an explicit yaw on the
    boundary waypoint still takes precedence."""
    f = PathFollower(
        [(0.0, 0.0, 0.0, 2.0), (0.0, 5.0, 0.0, 2.0), (0.0, 10.0, 0.0, 2.0),
         (0.0, 9.7, 0.0, -2.0, 0.25), (0.0, 5.0, 0.0, -2.0),
         (0.0, 0.0, 0.0, -2.0)],
        make_table())
    assert f._block_end_yaw[0] == pytest.approx(0.25)


def test_param_overrides_respected():
    path = _densify([(0.0, 0.0, 0.0, 2.0), (10.0, 0.0, 0.0, 2.0)])
    f = PathFollower(path, make_table(), lookahead=2.0, arrival_goal=3.0,
                     steer_cap=0.25)
    assert f.lookahead == 2.0
    # arrival_goal=3.0 -> DONE already at x=7.5.
    f.step((7.5, 0.0), 0.0, 0.0, 0.0, 0.0)
    assert f.last_mode == "DONE"
    # steer_cap honored even with a big heading error.
    f2 = PathFollower(path, make_table(), steer_cap=0.25)
    _thr, steer, _brk = f2.step((0.0, 3.0), math.pi / 2, 1.0, 0.0, 0.0)
    assert abs(steer) <= 0.25 + 1e-9
