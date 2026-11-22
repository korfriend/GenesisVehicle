"""impl-review round 1 fixes (2.0.0), each pinned by a test.

Mutants this file must fail on:
* ``NonlinearMPC.max_violation`` returning 0.0 (or treating NaN as feasible);
* ``reset_history`` default ``False`` (api-review C6);
* the yaw_rate / v_lat TypeError raised AFTER the clock / MPC reference moved;
* ``LinearMPC`` keeping an unapplied (infeasible) solution as ``u_prev``;
* ``_cross_check_live`` skipping a check without a warning.
"""
import copy
import math
import os
import sys
import types

import numpy as np
import pytest

pytest.importorskip("casadi")
pytest.importorskip("do_mpc")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control import PathFollower  # noqa: E402
from genesis_vehicle.control.mpc import MPCVehicleParams  # noqa: E402
from genesis_vehicle.control.mpc.alpasim_nonlinear import NonlinearMPC, SolveResult  # noqa: E402
from genesis_vehicle.control.mpc.law import FEAS_TOL, classify_solve  # noqa: E402
from genesis_vehicle.tests._mpc_fakes import solver_params  # noqa: E402
from genesis_vehicle.tests.data import make_mpc_field_replay as rp  # noqa: E402


def _left(v=6.0, R=15.0):
    s = np.arange(21) * 0.1 * v
    th = s / R
    return np.column_stack([R * np.sin(th), R - R * np.cos(th), th])


def test_real_capped_solve_is_classified_infeasible():
    s = NonlinearMPC(solver_params(), yaw_bound=math.pi / 2, nlpsol_opts={"ipopt.max_iter": 1})
    r = s.solve(np.array([0, 0, 0, 6.0, 0, 0, 0, 0]), _left())
    assert r.return_status == "Maximum_Iterations_Exceeded"
    viol = s.max_violation()
    assert viol > FEAS_TOL, viol
    cls, key, feas = classify_solve(r, s)
    assert (cls, key, feas) == ("CAP", "capped_infeasible", False)


def test_converged_solve_is_feasible():
    s = NonlinearMPC(solver_params(), yaw_bound=math.pi / 2)
    s.solve(np.array([0, 0, 0, 6.0, 0, 0, 0, 0]), _left())
    assert s.max_violation() <= FEAS_TOL


def _stub(x, ubx, g=(), lbg=(), ubg=()):
    m = types.SimpleNamespace(
        opt_x_num=types.SimpleNamespace(cat=np.asarray(x, float)),
        _lb_opt_x=types.SimpleNamespace(cat=np.full(len(x), -np.inf)),
        _ub_opt_x=types.SimpleNamespace(cat=np.asarray(ubx, float)),
        opt_g_num=np.asarray(g, float), nlp_cons_lb=np.asarray(lbg, float),
        nlp_cons_ub=np.asarray(ubg, float))
    return types.SimpleNamespace(_mpc=m)


def test_nan_is_never_feasible():
    mv = NonlinearMPC.max_violation
    assert mv(_stub([0.0, 0.2], [0.5, 0.5])) == 0.0
    assert mv(_stub([0.0, 1.0], [0.5, 0.5])) == pytest.approx(0.5)
    assert mv(_stub([0.0, np.nan, 1.0], [0.5, 0.5, 0.5])) == float("inf")
    assert mv(_stub([0.0], [0.5], g=[np.nan], lbg=[-1.0], ubg=[1.0])) == float("inf")


def test_reset_history_is_on_by_default():
    s = NonlinearMPC(solver_params(), yaw_bound=math.pi / 2)
    for k in range(5):
        s.solve(np.array([0, 0, 0, 3.0 + 0.1 * k, 0, 0, 0, 0]), _left(3.0))
    assert s.reset_history is True
    assert np.asarray(s._mpc.data["_x"]).shape[0] == 0


def _mpc_follower():
    p = MPCVehicleParams.from_dict(copy.deepcopy(rp.PARAMS_SDK))
    return PathFollower([(float(x), 0.0, 0.0, 3.0) for x in np.arange(0.0, 30.0, 0.5)],
                        rp.make_table(), mpc_params=p, command_mode="signed")


def test_missing_yaw_rate_raises_before_anything_moves():
    f, g = _mpc_follower(), _mpc_follower()
    st = ((0.4, 0.0), 0.0, 1.0, 0.0, 0.0)
    for _ in range(2):
        with pytest.raises(TypeError, match="extract_state"):
            f.step(*st)
        assert f._t_next == 0.0 and f._mpc_block is None and f._law.ref is None
        assert (f.block_i, f.current_idx, f.projection_t) == (0, 0, 0.0)
    assert f.step(*st, yaw_rate=0.0, v_lat=0.0) == g.step(*st, yaw_rate=0.0, v_lat=0.0)


def test_linear_u_prev_keeps_only_applied_solutions():
    from genesis_vehicle.control.mpc.alpasim_linear import LinearMPC
    lm = LinearMPC(solver_params())
    assert lm.rterm == "abs"                         # what PathFollower constructs
    lm.set_u_prev(0.1, 0.2)
    real = lm._solve_qp
    lm._solve_qp = lambda *a: SolveResult(u=np.array([2e9, 2e9]), return_status="primal infeasible",
                                          extra={"status_val": 3})
    lm.solve(np.zeros(8), np.zeros((21, 3)))
    assert lm.u_prev.tolist() == [0.1, 0.2]
    lm._solve_qp = real
    r = lm.solve(np.zeros(8), np.column_stack([np.linspace(0, 5, 21), np.zeros(21), np.zeros(21)]))
    assert lm.u_prev.tolist() == r.u.tolist()


def test_cross_check_names_what_it_could_not_check():
    from genesis_vehicle.control.path_follower import _cross_check_live
    p = MPCVehicleParams.from_dict(copy.deepcopy(rp.PARAMS_SDK))
    plant = types.SimpleNamespace(mass_props=types.SimpleNamespace(mass=1330.0),
                                  src=types.SimpleNamespace(wheel_meta=None, resolved=None))
    with pytest.warns(UserWarning, match="could not be cross-checked.*wheelbase"):
        _cross_check_live(plant, p)
    plant.src.wheel_meta = types.SimpleNamespace(positions=None, axle_index=None)
    with pytest.warns(UserWarning, match="wheelbase \\(AttributeError"):
        _cross_check_live(plant, p)


def test_mpc_v_floor_default_is_visible():
    import inspect
    assert repr(inspect.signature(PathFollower).parameters["mpc_v_floor"].default) == "0.5"
