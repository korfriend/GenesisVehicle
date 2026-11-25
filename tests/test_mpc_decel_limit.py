"""2.1.0 opt-in ``mpc_decel_limit``: the MPC's deceleration bound from the record's brake row, END-GATED.

numpy-level: ``solver_params(decel_limit=)`` tables (including the 0 m/s rule), validation (A4) and the
"narrower" warning — ``solver_params`` is a TABLE selector without a gate. Follower level (needs casadi /
do_mpc), rewritten in 2.1.0 round 2 (A26) for the end gate: ``PathFollower(mpc_decel_limit="brake")``
builds the throttle-row and the brake-row problems (C10 as corrected); a held ``cmd_a = -8`` is clipped to
the THROTTLE row while the last solve was closed and to the BRAKE row after a solve that opened the gate
(U4); the gear-mode finish at the open brake-row target brakes >= 0.99 (U5). API: allowed values,
``"off"`` refusal (even ``"throttle"``), signed + brake, empty / invalid brake row, and no side effect
before each raise (A3: a plant's ``steer_range`` is unchanged). The gate itself (predicate, transfer,
stats, construction, LMPC) is tested in ``test_mpc_decel_gate.py``.

Mutants this file must fail on (plan_r21 section 7.2 / C10, plan_r21_round2 section 3.3): ``decel_limit``
ignored by ``solver_params``; the 0 m/s entry taken from the throttle row under ``"brake"``; the default
switched to ``"brake"``; brake accepted with signed; ``mpc_decel_limit`` accepted with ``"off"``; the
follower not passing ``decel_limit="brake"`` on for the brake-row problem; the gate never opening (U4,
U5); the clip read from the brake table while the gate is closed (U4).
"""
import copy
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control import PathFollower  # noqa: E402
from genesis_vehicle.control.mpc import MPCVehicleParams  # noqa: E402
from genesis_vehicle.tests.data.make_mpc_field_replay import PARAMS_SDK  # noqa: E402
from genesis_vehicle.tests.test_sweep_table import make_table  # noqa: E402

PATH = [(float(x), 0.0, 0.0, 3.0) for x in np.arange(0.0, 30.0, 0.5)]
BRAKE_TAB = ([0.0, 3.0, 6.0, 10.0, 14.0], [-5.17, -5.17, -5.17, -5.19, -5.21])
THROTTLE_TAB = ([0.0, 3.0, 6.0, 10.0, 14.0], [-2.15, -2.15, -2.16, -2.18, -2.2])


def _d(**over):
    d = copy.deepcopy(PARAMS_SDK)
    d.update(over)
    return d


def _rec(**over):
    return MPCVehicleParams.from_dict(_d(**over))


# --- solver_params ------------------------------------------------------------------------------------

def test_brake_bound_tables_and_the_0_ms_rule():
    sp = _rec().solver_params("bicycle", steer_cap=0.5, dt_ctrl=0.025, decel_limit="brake")
    assert sp.accel_min_tab == BRAKE_TAB
    assert sp.accel_min == -5.17
    th = _rec().solver_params("bicycle", steer_cap=0.5, dt_ctrl=0.025)
    assert th.accel_min_tab == THROTTLE_TAB and th.accel_min == -2.15
    # everything else is the same parameter set
    for k in ("mass", "inertia", "accel_max_tab", "accel_max", "steer_max", "steering_time_constant"):
        assert getattr(sp, k) == getattr(th, k)


def test_brake_0_ms_entry_is_the_lowest_brake_speed_not_the_throttle_row():
    sp = _rec(decel_brake_by_speed={"6": -5.3, "3": -5.0, "10": -5.4}).solver_params(decel_limit="brake")
    assert sp.accel_min_tab == ([0.0, 3.0, 6.0, 10.0], [-5.0, -5.0, -5.3, -5.4])
    assert sp.accel_min == -5.0


def test_default_decel_limit_is_throttle():
    import inspect
    assert inspect.signature(MPCVehicleParams.solver_params).parameters["decel_limit"].default == "throttle"
    assert _rec().solver_params().accel_min_tab == THROTTLE_TAB


@pytest.mark.parametrize("bad", ["Brake", None, "", "pd"])
def test_unknown_decel_limit_raises(bad):
    with pytest.raises(ValueError, match="mpc_decel_limit must be one of"):
        _rec().solver_params(decel_limit=bad)


@pytest.mark.parametrize("row", [{}, {"3": 0.0}, {"3": 1.2}, {"3": float("nan")}, {"3": -5.0, "6": float("inf")},
                                 {"x": -5.0}, {"3": "fast"}])
def test_brake_row_must_be_finite_and_negative(row):
    with pytest.raises(ValueError, match="decel_brake_by_speed"):
        _rec(decel_brake_by_speed=row).solver_params(decel_limit="brake")
    _rec(decel_brake_by_speed=row).solver_params()          # the throttle bound does not read it


def test_brake_row_weaker_than_throttle_row_warns():
    with pytest.warns(UserWarning, match=r"NARROWER there") as w:
        _rec(decel_brake_by_speed={"3": -2.0, "6": -5.0, "10": -5.0, "14": -5.0}).solver_params(decel_limit="brake")
    assert "[3.0]" in str(w[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _rec().solver_params(decel_limit="brake")


# --- follower API -------------------------------------------------------------------------------------

def _table_with_range():
    t = make_table()
    t.steer_range = (-1.0, 1.0)
    return t


@pytest.mark.parametrize("val", ["throttle", "brake"])
def test_off_refuses_mpc_decel_limit_even_at_its_default(val):
    with pytest.raises(ValueError, match="mpc_decel_limit"):
        PathFollower(PATH, make_table(), mpc_model="off", mpc_decel_limit=val)


def test_off_follower_resolves_none():
    assert PathFollower(PATH, make_table(), mpc_model="off").mpc_decel_limit is None


@pytest.mark.parametrize("bad", ["Brake", None, 1])
def test_bad_value_raises_at_step_7_before_any_side_effect(bad):
    t = _table_with_range()
    with pytest.raises(ValueError, match="mpc_decel_limit must be one of"):
        PathFollower(PATH, t, mpc_params=_rec(), mpc_decel_limit=bad)
    assert t.steer_range == (-1.0, 1.0)


def test_value_is_checked_before_the_dependencies(monkeypatch):
    from genesis_vehicle.control.mpc import _deps
    monkeypatch.setattr(_deps, "require_mpc_deps", lambda impl: (_ for _ in ()).throw(ImportError("deps")))
    with pytest.raises(ValueError, match="mpc_decel_limit"):
        PathFollower(PATH, make_table(), mpc_params=_rec(), mpc_decel_limit="x")


@pytest.fixture(scope="module")
def deps():
    pytest.importorskip("casadi")
    pytest.importorskip("do_mpc")


def test_brake_with_signed_raises_before_any_side_effect(deps):
    t = _table_with_range()
    with pytest.raises(ValueError, match="needs command_mode=\"gear\""):
        PathFollower(PATH, t, mpc_params=_rec(), mpc_decel_limit="brake", command_mode="signed")
    assert t.steer_range == (-1.0, 1.0)
    PathFollower(PATH, t, mpc_params=_rec(), mpc_decel_limit="throttle", command_mode="signed")
    assert t.steer_range == (-0.5, 0.5)             # positive control: construction writes it


@pytest.mark.parametrize("row", [{}, {"3": 0.5}])
def test_invalid_brake_row_raises_at_step_9_before_any_side_effect(deps, row):
    t = _table_with_range()
    with pytest.raises(ValueError, match="decel_brake_by_speed"):
        PathFollower(PATH, t, mpc_params=_rec(decel_brake_by_speed=row), mpc_decel_limit="brake")
    assert t.steer_range == (-1.0, 1.0)


SHORT = [(float(x), 0.0, 0.0, 2.0) for x in np.arange(0.0, 3.01, 0.5)]   # reference end within 2 s


def test_c10_follower_builds_the_throttle_and_the_brake_problem(deps):
    from genesis_vehicle.control.mpc.alpasim_linear import _DecelGatedLMPC
    from genesis_vehicle.control.mpc.alpasim_nonlinear import _DecelGatedNMPC
    f = PathFollower(PATH, make_table(), mpc_params=_rec(), mpc_decel_limit="brake")
    assert f.mpc_decel_limit == "brake" and f._law.decel_gate
    w = f._law.solver
    assert type(w) is _DecelGatedNMPC
    assert w._thr.p.accel_min_tab == THROTTLE_TAB and w._thr.p.accel_min == -2.15
    assert w._brk.p.accel_min_tab == BRAKE_TAB and w._brk.p.accel_min == -5.17
    assert w.p is w._thr.p                          # closed until a solve opens it
    g = PathFollower(PATH, make_table(), mpc_params=_rec(), mpc_impl="linear", mpc_decel_limit="brake")
    assert type(g._law.solver) is _DecelGatedLMPC
    assert g._law.solver._p_thr.accel_min_tab == THROTTLE_TAB
    assert g._law.solver._p_brk.accel_min_tab == BRAKE_TAB
    assert g._law.solver.p is g._law.solver._p_thr


def _held_clip_and_finish(fol, path_end_open):
    """One real solve at the path start (opens the gate on SHORT, not on PATH), then a held step with
    cmd_a = -8 at 10 m/s: the stage-1 clip and the gear-mode finish."""
    from genesis_vehicle.control import FollowerPlan
    law = fol._law
    law.start_block(SHORT if path_end_open else PATH, t0=0.0)
    law.step(0.0, (0.0, 0.0), 0.0, 2.0, 0.0, 0.0)                     # solves; the gate decides
    law.cmd_t, law.cmd_a, law.cmd_delta, law.mode = 0.0, -8.0, 0.0, "MPC"
    lo = law.step(0.05, (0.0, 0.0), 0.0, 10.0, 0.0, 0.0)              # not due: no solve
    assert not lo.solved
    plan = FollowerPlan("MPC", lo.a_t, 0.0, 10.0, 10.0, 0.0, 0.0, None, regime="R-MPC", a_ff=lo.a_t, gear=1)
    thr = fol.plant.throttle_for(10.0, lo.a_t, 0.0, 0.0)
    return law, lo.a_t, fol._finish_gear(plan, thr, 0.0)


def test_u4_held_clip_follows_the_last_solves_row(deps):
    # k_b = coast(10) - brake row(10) on this synthetic table (-2.0 + 5.19): the record's brake row and
    # its gain describe the same vehicle, as identify's do (reference car: brake 1.0011 at 10 m/s)
    f = PathFollower(PATH, make_table(), mpc_params=_rec(brake_gain=3.19), mpc_decel_limit="brake")
    law, a_closed, _ = _held_clip_and_finish(f, path_end_open=False)
    assert law.decel_gate_stats()["open"] is False
    assert a_closed == pytest.approx(np.interp(10.0, *THROTTLE_TAB))     # closed: the throttle row
    g = PathFollower(PATH, make_table(), mpc_params=_rec(brake_gain=3.19), mpc_decel_limit="brake")
    law, a_open, _ = _held_clip_and_finish(g, path_end_open=True)
    assert law.decel_gate_stats()["open"] is True
    assert a_open == pytest.approx(np.interp(10.0, *BRAKE_TAB))          # open: the brake row
    d = PathFollower(PATH, make_table(), mpc_params=_rec(brake_gain=3.19))   # the default, same SHORT solve
    _, a_def, _ = _held_clip_and_finish(d, path_end_open=True)
    assert a_def == pytest.approx(np.interp(10.0, *THROTTLE_TAB))


def test_u5_gear_finish_brakes_fully_at_an_open_brake_row_target(deps):
    f = PathFollower(PATH, make_table(), mpc_params=_rec(brake_gain=3.19), mpc_decel_limit="brake")
    _, _, cmd = _held_clip_and_finish(f, path_end_open=True)
    assert cmd.brake >= 0.99 and cmd.throttle == 0.0
    g = PathFollower(PATH, make_table(), mpc_params=_rec(brake_gain=3.19), mpc_decel_limit="brake")
    _, _, closed = _held_clip_and_finish(g, path_end_open=False)
    assert closed.brake < 0.5                        # closed: the throttle row brakes far less (control)
