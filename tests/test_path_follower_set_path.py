"""PathFollower.set_path (v2.0.0): atomic path replacement.

PD: the recomputed path model equals a fresh constructor's; ``pos_xy`` snaps
the projection to the globally nearest segment (beyond the 5-segment advance
cap); a failing call leaves everything unchanged. MPC: the reference is
re-timed so the snapped point is due at ``t``; the NLP is NOT rebuilt (same
solver object, same do_mpc problem); do_mpc's ``u_prev`` is reset to the
measured ``(d0, a0)`` before the next solve (plan section 3.7).

Mutants this file must fail on (plan section 11): set_path rebuilding the NLP
(a new solver); ``u0`` kept from the previous solve instead of reset.
"""
import copy
import math
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control import PathFollower  # noqa: E402
from genesis_vehicle.control.mpc import MPCVehicleParams  # noqa: E402
from genesis_vehicle.tests.data import make_mpc_field_replay as rp  # noqa: E402
from genesis_vehicle.tests.test_sweep_table import make_table  # noqa: E402

A = [(float(x), 0.0, 0.0, 2.0) for x in np.arange(0.0, 20.0, 0.5)]
B = ([(5.0, float(y), 0.0, 1.5) for y in np.arange(0.0, 10.0, 0.5)]
     + [(5.0 - float(d), 10.0, 0.0, -1.0) for d in np.arange(0.0, 5.0, 0.5)])

PD_ATTRS = ("path", "wp_yaws", "n", "blocks", "yaws", "_block_end_yaw")


def test_pd_model_equals_a_fresh_constructor():
    f = PathFollower(A, make_table(), mpc_model="off")
    for k in range(30):
        f.step((0.3 * k, 0.1), 0.0, 1.0, 0.0, 0.0)
    f.transitioning = True
    f.set_path(B)
    g = PathFollower(B, make_table(), mpc_model="off")
    for a in PD_ATTRS + ("block_i", "current_idx", "projection_t", "projection_xy", "transitioning"):
        assert getattr(f, a) == getattr(g, a), a
    # and drives identically from there
    for k in range(40):
        st = ((5.0, 0.25 * k), math.pi / 2, 1.0, 0.0, 0.0)
        assert f.step(*st) == g.step(*st)


def test_pos_xy_snaps_beyond_the_advance_cap():
    f = PathFollower(A, make_table(), mpc_model="off")
    f.set_path(A, pos_xy=(12.2, 0.3))
    assert f.current_idx == 24 and f.projection_t == pytest.approx(0.4)
    assert f.projection_xy == pytest.approx((12.2, 0.0))


def test_failure_leaves_the_follower_unchanged():
    f = PathFollower(A, make_table(), mpc_model="off")
    for k in range(10):
        f.step((0.3 * k, 0.0), 0.0, 1.0, 0.0, 0.0)
    before = {a: copy.deepcopy(getattr(f, a)) for a in PD_ATTRS + ("current_idx", "projection_t")}
    with pytest.raises(ValueError, match="length 3"):
        f.set_path([(0, 0, 0), (1, 1, 1)])
    with pytest.raises(ValueError, match="at least 2"):
        f.set_path([(0, 0, 0, 1.0)])
    assert {a: getattr(f, a) for a in before} == before


@pytest.fixture
def mpc_follower():
    pytest.importorskip("casadi")
    pytest.importorskip("do_mpc")
    p = MPCVehicleParams.from_dict(copy.deepcopy(rp.PARAMS_SDK))
    return PathFollower(A, rp.make_table(), mpc_params=p)


def test_mpc_set_path_keeps_the_nlp_and_resets_u_prev(mpc_follower):
    f = mpc_follower
    solver, problem = f._law.solver, f._law.solver._mpc
    for k in range(12):
        f.step((0.3 * k, 0.0), 0.0, 1.0 + 0.1 * k, 0.0, 0.0, yaw_rate=0.02, v_lat=0.0, t=k * 0.025)
    assert np.any(np.array(problem.opt_p_num["_u_prev"]).ravel() != 0.0)
    C = [(float(x), 2.0, 0.0, 3.0) for x in np.arange(0.0, 30.0, 0.5)]
    f.set_path(C, pos_xy=(6.1, 2.0), t=1.0)
    assert f._law.solver is solver and f._law.solver._mpc is problem
    v, r = 3.0, 0.2
    f.step((6.1, 2.0), 0.0, v, 0.0, 0.0, yaw_rate=r, v_lat=0.0, t=1.0)
    used = np.array(problem.opt_p_num["_u_prev"]).ravel()
    assert used[0] == pytest.approx(math.atan(r * 2.7 / v), rel=1e-12)
    assert used[1] == 0.0                              # history cleared


def test_mpc_set_path_retimes_the_reference(mpc_follower):
    f = mpc_follower
    C = [(float(x), 0.0, 0.0, 2.0) for x in np.arange(0.0, 30.0, 0.5)]
    f.set_path(C, pos_xy=(10.0, 0.0), t=4.0)
    t, X, Y, H = f._law.ref
    # the snapped point (x = 10 m, waypoint 20) is due at t = 4 s
    assert np.interp(10.0 - 1.35, X, t) == pytest.approx(4.0, abs=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f.set_path([(float(x), 0.0, 0.0, -1.0) for x in range(5)])
    assert f._mpc_block is None
