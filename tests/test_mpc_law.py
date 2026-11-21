"""``_MPCLaw`` (the MPC layer's stage 1) and the MPC finish, with a fake solver.

No casadi / do_mpc / Genesis. What is pinned: the 0.1 s solve cadence and the
held command between solves; the initial state handed to the solver (rig
frame, measured acceleration over the last 0.1 s, the AlpaSim d0 rule); the
clip of the acceleration target to the per-speed limits and the field
``omega = max(v, 0) * tan(delta) / L`` feedforward; the forward-only reference;
the finish rules (caps, brake at rest, MPC-HOLD on ``a_ff <= 0``).

Mutants this file must fail on (plan section 11): ``a_ff <= 0.0`` -> ``< 0.0``
in ``finish_command``; the 0.1 s cadence check loosened.
(The PathFollower-level parts of plan section 11 — DONE and "no solver call in
a reverse block" — arrive with the PathFollower integration.)
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control.mpc.law import (  # noqa: E402
    DT_MPC, _MPCLaw, finish_command, rig_reference)
from genesis_vehicle.control.mpc.params import accel_limits  # noqa: E402
from genesis_vehicle.tests._mpc_fakes import FakeSolver, ok  # noqa: E402

DT = 0.025
PATH = [(float(x), 0.0, 0.0, 4.0) for x in np.arange(0.0, 40.0, 0.5)]


def _law(results=None, **kw):
    s = FakeSolver(results)
    law = _MPCLaw(s, rig_offset=1.35, **kw)
    law.start_block(PATH, t0=0.0)
    return law, s


def test_solves_every_0_1_s_and_holds_between():
    law, s = _law()
    outs = []
    for k in range(13):
        outs.append(law.step(k * DT, (0.1 * k, 0.0), 0.0, 3.0, 0.0, 0.0))
    solved = [o.solved for o in outs]
    assert solved == [True, False, False, False] * 3 + [True]
    assert round(4 * DT, 12) == DT_MPC
    assert len(s.calls) == 4
    # between solves the target is the held command
    assert all(o.a_t == outs[0].a_t for o in outs[:4])


def test_cadence_tolerates_float_time():
    # t accumulated by repeated addition still solves at the 4th step
    law, s = _law()
    t = 0.0
    n = 0
    for _ in range(4):
        n += law.step(t, (0.0, 0.0), 0.0, 1.0, 0.0, 0.0).solved
        t += DT
    n += law.step(t, (0.0, 0.0), 0.0, 1.0, 0.0, 0.0).solved
    assert n == 2


def test_cadence_is_not_faster_than_0_1_s():
    law, s = _law()
    for k in range(40):
        law.step(k * DT * 0.999, (0.0, 0.0), 0.0, 1.0, 0.0, 0.0)
    # at 0.999*dt spacing 4 steps are 0.0999 s < 0.1 s - 1e-6: one solve per 5 steps
    assert len(s.calls) == 8


def test_initial_state_and_d0_rule():
    law, s = _law()
    law.step(0.0, (2.0, 1.0), 0.1, 6.0, 0.3, -0.05)
    x0 = s.calls[0][0]
    L = 2.7
    assert x0[:3].tolist() == [0.0, 0.0, 0.0]
    assert x0[3] == 6.0 and x0[4] == -0.05 and x0[5] == 0.3
    assert x0[6] == pytest.approx(math.atan(0.3 * L / 6.0), rel=1e-15)
    assert x0[7] == 0.0                              # empty history -> a0 = 0
    law2, s2 = _law()
    law2.step(0.0, (0.0, 0.0), 0.0, 0.2, 0.3, 0.0)   # v <= 0.25 -> d0 = 0
    assert s2.calls[0][0][6] == 0.0
    law3, s3 = _law()
    law3.step(0.0, (0.0, 0.0), 0.0, -1.0, 0.0, 0.0)  # backwards -> v state 0
    assert s3.calls[0][0][3] == 0.0


def test_a0_is_speed_change_over_the_last_0_1_s():
    law, s = _law()
    for k, v in enumerate([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]):
        law.step(k * DT, (0.0, 0.0), 0.0, v, 0.0, 0.0)
    # solves at k = 0, 4, 8; at k = 8 the history spans t = 0.1 .. 0.2
    assert s.calls[1][0][7] == pytest.approx((1.4 - 1.0) / 0.1)
    assert s.calls[2][0][7] == pytest.approx((1.8 - 1.4) / 0.1)


def test_reference_is_in_the_rig_frame():
    law, s = _law()
    law.step(0.0, (1.35, 0.0), 0.0, 0.0, 0.0, 0.0)   # base_link at x=1.35 -> rig at 0
    ref = s.calls[0][1]
    assert ref.shape == (21, 3)
    # first node: the path start's rig point (-1.35, 0) seen from the rig at (0, 0)
    assert ref[0, 0] == pytest.approx(-1.35)
    assert np.all(np.diff(ref[:, 0]) >= 0) and np.allclose(ref[:, 1:], 0.0)


def test_targets_clip_and_feedforward():
    law, s = _law([ok(delta=0.1, accel=5.0)])
    o = law.step(0.0, (0.0, 0.0), 0.0, 4.0, 0.0, 0.0)
    lo, hi = accel_limits(s.p, 4.0)
    assert o.a_t == hi                                # 5.0 clipped to the 4 m/s limit
    assert o.w_t == 4.0 * math.tan(0.1) / 2.7
    o2 = law.step(DT, (0.0, 0.0), 0.0, -0.5, 0.0, 0.0)
    assert o2.w_t == 0.0                              # max(v, 0)


def test_reverse_waypoints_are_rejected_by_the_reference():
    with pytest.raises(ValueError, match="forward"):
        rig_reference([(0, 0, 0, 1.0), (1, 0, 0, -1.0)], 1.35, 0.5)


def test_reference_timing_uses_v_floor_and_t0():
    t, X, Y, H = rig_reference([(0, 0, 0, 0.0), (1, 0, 0, 0.0), (2, 0, 0, 0.0)], 1.0, 0.5, t0=3.0)
    assert t.tolist() == [3.0, 5.0, 7.0]              # 1 m at the 0.5 m/s floor = 2 s
    assert X.tolist() == [-1.0, 0.0, 1.0]


def test_explicit_first_yaw_is_used():
    t, X, Y, H = rig_reference([(0, 0, 0, 1.0, 0.5), (1, 0, 0, 1.0, 0.5)], 1.0, 0.5)
    assert H.tolist() == [0.5, 0.5]


@pytest.mark.parametrize("v,a_ff,thr,want_mode,want", [
    (0.05, 0.0, -0.3, "MPC-HOLD", (0.0, 1.0)),       # at rest, plan wants a_ff = 0 -> hold
    (0.05, -0.2, -0.3, "MPC-HOLD", (0.0, 1.0)),
    (-0.2, 0.5, 0.4, "MPC-HOLD", (0.0, 1.0)),        # rolling back
    (0.05, 0.3, 0.01, "MPC", (0.01, 1.0)),           # at rest, small throttle -> brake at rest
    (2.0, -0.5, -0.3, "MPC", (-0.3, 0.0)),           # moving: negative throttle decelerates
])
def test_finish_rules(v, a_ff, thr, want_mode, want):
    t, s, b, mode = finish_command(v, a_ff, thr, 0.9, 0.5)
    assert mode == want_mode
    assert (t, b) == want
    assert s == 0.5                                   # steer cap


def test_start_block_restarts_the_command():
    law, s = _law([ok(delta=0.2, accel=1.0)])
    law.step(0.0, (0.0, 0.0), 0.0, 2.0, 0.0, 0.0)
    assert law.cmd_delta == 0.2
    law.start_block(PATH, t0=5.0)
    assert law.cmd_delta == 0.0 and law.cmd_a == 0.0 and law.vhist == []
    assert law.due(5.0)
    assert law.ref[0][0] == 5.0
