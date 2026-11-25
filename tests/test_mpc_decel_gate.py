"""2.1.0 ``PathFollower(mpc_decel_limit="brake")``: the END GATE (plan_r21_round2 section 1.4, C29-C35,
A16-A28).

The brake-row problem is used on a solve iff the active forward block's reference end time is within
``N * DT_MPC`` of the solve time in REFERENCE time (``t + DT_MPC * N >= t_end - 1e-9``) AND the rig
reference's end point has body ``x > 0`` in the solve frame; otherwise the throttle-row problem (the
default's). Tests (all Genesis-free):

* U1 the predicate's truth table at ``t_end - N * DT_MPC +- 1e-6``, end ahead / behind, node 0 behind with
  the end ahead (G4 geometry), ``N != 20``; ``select_decel`` is called before ``solve``.
* U2 (real NLPs) a switch copies ``opt_x_num``, ``opt_x_num_unscaled`` and ``u0`` bitwise and by value
  (``is not``), in both directions, from the previously active solver; the duals only when it has them.
* U3 (real NLPs) a ``"brake"`` and a default follower fed the same state stream give bitwise-equal
  commands and law outputs on every step before the first open solve, which is the predicted one and is
  solved by the brake-row NLP.
* U6 / A17 ``mpc_stats["decel_gate"]`` is ``None`` under ``"throttle"`` (NMPC and LMPC) and the counter
  dict under ``"brake"`` (``open`` None before the first solve; counts kept by ``set_path``; failed solves
  counted in ``open_solves``).
* U7 / C29 (real NLP) a first solve that opens (source never solved) raises nothing and gives the u of a
  cold brake-row solver bitwise; A19 on that solve (switch + restart): the solving NLP's ``_u_prev`` is
  ``(d0, a0)``.
* A18 the explicit seam: ``decel_gate=True`` needs a callable ``select_decel`` (``TypeError``); with False
  the law never looks it up.
* A20 the wrapper's explicit members; A21 the constructor's layout checks and the transfer's shape check
  (``ValueError``, nothing assigned); A23 construction (spy); A24 the LMPC swap; A25 a transfer error
  propagates out of ``plan`` (no PD fall-back, status unchanged).

Mutants this file must fail on (plan_r21_round2 section 3.3, C29, C34, A17): gate always open; ahead sign
flipped; horizon hard-coded to 2.0; ahead tested on ``ref_l[0]``; no warm-start transfer; transfer from the
wrong solver; unconditional dual copy; gate evaluated after the solve; a ``decel_gate`` dict under
``"throttle"``.
"""
import copy
import math
import os
import sys
import types
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control import PathFollower  # noqa: E402
from genesis_vehicle.control.mpc import MPCGains, MPCVehicleParams  # noqa: E402
from genesis_vehicle.control.mpc.law import DT_MPC, _MPCLaw  # noqa: E402
from genesis_vehicle.tests._mpc_fakes import FakeSolver, exc, ok  # noqa: E402
from genesis_vehicle.tests.data.make_mpc_field_replay import PARAMS_SDK  # noqa: E402
from genesis_vehicle.tests.test_sweep_table import make_table  # noqa: E402

DT = 0.025
RIG = float(PARAMS_SDK["rig_offset"])
LONG = [(float(x), 0.0, 0.0, 4.0) for x in np.arange(0.0, 30.0, 0.5)]          # end ~7.4 s away
SHORT = [(float(x), 0.0, 0.0, 2.0) for x in np.arange(0.0, 3.01, 0.5)]         # end within 2 s
BRAKE_TAB = ([0.0, 3.0, 6.0, 10.0, 14.0], [-5.17, -5.17, -5.17, -5.19, -5.21])
THROTTLE_TAB = ([0.0, 3.0, 6.0, 10.0, 14.0], [-2.15, -2.15, -2.16, -2.18, -2.2])


def _rec(**over):
    d = copy.deepcopy(PARAMS_SDK)
    d.update(over)
    return MPCVehicleParams.from_dict(d)


def _sp(decel_limit="throttle"):
    return _rec().solver_params("bicycle", steer_cap=0.5, dt_ctrl=0.025, decel_limit=decel_limit)


@pytest.fixture(scope="module")
def deps():
    pytest.importorskip("casadi")
    pytest.importorskip("do_mpc")


@pytest.fixture(scope="module")
def lmpc_deps():
    pytest.importorskip("osqp")


# --- U1: the predicate (fake solver, so N is free) -------------------------------------------------

class _GatedFake(FakeSolver):
    def __init__(self, N=20):
        super().__init__(N=N)
        self.events = []

    def select_decel(self, open_):
        self.events.append(("select", bool(open_)))

    def solve(self, x0, ref):
        self.events.append(("solve",))
        return super().solve(x0, ref)


def _gate_once(N, dt_before_end, car_x):
    """A fresh gated law on LONG; one solve at t = t_end - N * DT_MPC + dt_before_end with the vehicle
    (base_link) at (car_x, 0), heading +x. Returns (open passed to select_decel, events, law)."""
    s = _GatedFake(N=N)
    law = _MPCLaw(s, rig_offset=RIG, decel_gate=True)
    law.start_block(LONG, t0=0.0)
    t_end = float(law.ref[0][-1])
    t = t_end - N * DT_MPC + dt_before_end
    law.step(t, (car_x, 0.0), 0.0, 4.0, 0.0, 0.0)
    return s.events[0][1], s.events, law


@pytest.mark.parametrize("N", [20, 10])
def test_u1_reference_time_boundary(N):
    end_x = LONG[-1][0]
    assert _gate_once(N, +1e-6, end_x - 6.0)[0] is True
    assert _gate_once(N, -1e-6, end_x - 6.0)[0] is False
    assert _gate_once(N, -0.5, end_x - 6.0)[0] is False      # 0.5 s outside an N-node horizon
    assert _gate_once(N, +0.5, end_x - 6.0)[0] is True


@pytest.mark.parametrize("N", [20, 10])
def test_u1_end_ahead_vs_behind(N):
    end_x = LONG[-1][0]
    assert _gate_once(N, +0.5, end_x - 1.0)[0] is True        # rig end (end_x - RIG) ahead of rig (end_x - 1 - RIG)
    assert _gate_once(N, +0.5, end_x + 1.0)[0] is False       # past it: behind the rear axle
    assert _gate_once(N, +0.5, end_x)[0] is False             # exactly at it: x == 0 is not ahead


def test_u1_node_zero_behind_end_ahead_opens():
    # G4 geometry: the reference's first nodes lie behind the rear axle, its end ahead.
    open_, _, law = _gate_once(20, +0.1, LONG[-1][0] - 4.0)
    t, X, Y, H = law.ref
    assert open_ is True
    first = float(np.interp(law.cmd_t, t, X)) - (LONG[-1][0] - 4.0 - RIG)
    assert first < 0.0                                         # node 0 is behind the car


def test_u1_select_runs_before_the_solve_on_every_solve():
    s = _GatedFake()
    law = _MPCLaw(s, rig_offset=RIG, decel_gate=True)
    law.start_block(LONG, t0=0.0)
    for k in range(9):
        law.step(k * DT, (4.0 * k * DT, 0.0), 0.0, 4.0, 0.0, 0.0)
    assert s.events == [("select", False), ("solve",)] * 3


# --- A18: the explicit seam ---------------------------------------------------------------------------

class _NoGate(FakeSolver):
    def __init__(self):
        super().__init__()
        object.__setattr__(self, "looked_up", [])

    def __getattr__(self, name):
        self.__dict__["looked_up"].append(name)
        raise AttributeError(name)


def test_a18_decel_gate_true_needs_a_callable_select_decel():
    with pytest.raises(TypeError, match="select_decel"):
        _MPCLaw(FakeSolver(), rig_offset=RIG, decel_gate=True)
    s = FakeSolver()
    s.select_decel = 1                                         # present but not callable
    with pytest.raises(TypeError, match="select_decel"):
        _MPCLaw(s, rig_offset=RIG, decel_gate=True)


def test_a18_without_the_gate_select_decel_is_never_looked_up():
    s = _NoGate()
    law = _MPCLaw(s, rig_offset=RIG)
    law.start_block(SHORT, t0=0.0)
    for k in range(9):
        law.step(k * DT, (2.0 * k * DT, 0.0), 0.0, 2.0, 0.0, 0.0)
    assert "select_decel" not in s.looked_up
    assert law.decel_gate_stats() is None


def test_a17_counters_with_a_fake_pair():
    s = _GatedFake()
    s.results = [ok(), exc(), ok(), ok()]
    law = _MPCLaw(s, rig_offset=RIG, decel_gate=True)
    assert law.decel_gate_stats() == {"open": None, "open_solves": 0, "switches": 0}
    law.start_block(LONG, t0=0.0)
    t_end = float(law.ref[0][-1])
    law.step(0.0, (0.0, 0.0), 0.0, 4.0, 0.0, 0.0)                              # closed
    law.step(t_end - 1.0, (LONG[-1][0] - 4.0, 0.0), 0.0, 4.0, 0.0, 0.0)        # open, the solve raises
    law.step(t_end - 0.9, (LONG[-1][0] - 4.0, 0.0), 0.0, 4.0, 0.0, 0.0)        # open
    law.step(t_end - 0.8, (LONG[-1][0] + 2.0, 0.0), 0.0, 4.0, 0.0, 0.0)        # end behind: closed
    assert law.decel_gate_stats() == {"open": False, "open_solves": 2, "switches": 2}
    law.start_block(LONG, t0=10.0)                                             # set_path re-enters here
    assert law.decel_gate_stats() == {"open": False, "open_solves": 2, "switches": 2}


# --- real NLPs ------------------------------------------------------------------------------------

def _pair():
    from genesis_vehicle.control.mpc.alpasim_nonlinear import NonlinearMPC, _DecelGatedNMPC
    thr = NonlinearMPC(_sp("throttle"), None, yaw_bound=math.pi / 2)
    brk = NonlinearMPC(_sp("brake"), None, yaw_bound=math.pi / 2)
    return _DecelGatedNMPC(thr, brk), thr, brk


def _ref(v=4.0, y_end=0.0, N=20):
    s = v * DT_MPC * np.arange(N + 1)
    return np.column_stack([s, np.linspace(0.0, y_end, N + 1), np.zeros(N + 1)])


def _state(s):
    m = s._mpc
    return (np.array(m.opt_x_num.master), np.array(m.opt_x_num_unscaled.master),
            np.array(m.u0.cat).ravel(), m.opt_x_num.master, m.opt_x_num_unscaled.master)


def test_u2_transfer_is_bitwise_by_value_both_directions(deps):
    w, thr, brk = _pair()
    x0 = np.array([0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    assert thr.solve(x0, _ref(y_end=0.5)).exception is None             # THR now has duals
    ox, oxu, u0, ox_obj, oxu_obj = _state(thr)
    w.select_decel(True)
    bx, bxu, bu0, bx_obj, bxu_obj = _state(brk)
    assert np.array_equal(bx, ox) and np.array_equal(bxu, oxu) and np.array_equal(bu0, u0)
    assert bx_obj is not ox_obj and bxu_obj is not oxu_obj                  # by value
    assert np.array_equal(np.array(brk._mpc.lam_x_num), np.array(thr._mpc.lam_x_num))
    assert brk._mpc.lam_x_num is not thr._mpc.lam_x_num
    assert brk._mpc.flags["initial_run"] is True
    assert w.p is brk.p
    assert w.solve(np.array([0.3, 0.0, 0.0, 3.5, 0.0, 0.0, 0.0, -1.0]), _ref(v=3.5, y_end=-0.4)).exception is None
    ox, oxu, u0, ox_obj, _ = _state(brk)
    assert not np.array_equal(ox, _state(thr)[0])                        # the two really differ now
    w.select_decel(False)                                                # back: BRK -> THR
    tx, txu, tu0, tx_obj, _ = _state(thr)
    assert np.array_equal(tx, ox) and np.array_equal(txu, oxu) and np.array_equal(tu0, u0)
    assert tx_obj is not ox_obj
    assert w.p is thr.p
    w.select_decel(False)                                                # no change: nothing copied
    assert _state(thr)[3] is tx_obj


def test_u2_source_that_never_solved_copies_no_duals(deps):
    w, thr, brk = _pair()
    assert not hasattr(thr._mpc, "lam_x_num")
    brk._mpc.flags["initial_run"] = True                                 # would pass stale duals
    w.select_decel(True)
    assert brk._mpc.flags["initial_run"] is False
    assert np.array_equal(np.array(brk._mpc.opt_x_num.master), np.array(thr._mpc.opt_x_num.master))


def test_a20_explicit_members_only(deps):
    w, thr, brk = _pair()
    assert not hasattr(w, "_mpc")
    assert "__getattr__" not in type(w).__dict__
    assert w.setup_ms == thr.setup_ms + brk.setup_ms
    assert (w.N, w.dt) == (thr.N, thr.dt)
    for name in ("p", "N", "dt", "u_prev", "setup_ms", "solve", "max_violation", "plan_vector",
                 "set_u_prev", "select_decel"):
        assert hasattr(w, name), name
    w.set_u_prev(0.1, -0.2)
    assert np.array_equal(w.u_prev, thr.u_prev) and np.array_equal(thr.u_prev, [0.1, -0.2])


class _Twin:
    """A stand-in for a NonlinearMPC whose do_mpc layout pieces can be altered (A21)."""

    def __init__(self, real, **over):
        self.N, self.dt, self.p, self.setup_ms = real.N, real.dt, real.p, real.setup_ms
        m = real._mpc
        self._mpc = types.SimpleNamespace(opt_x=m.opt_x, nlp_cons_lb=m.nlp_cons_lb,
                                          opt_x_scaling=m.opt_x_scaling)
        for k, v in over.items():
            setattr(self._mpc, k, v)


def test_a21_constructor_layout_checks(deps):
    import casadi
    from genesis_vehicle.control.mpc.alpasim_nonlinear import NonlinearMPC, _DecelGatedNMPC
    from genesis_vehicle.control.mpc.params import _AlpaSimParams
    _, thr, brk = _pair()
    _DecelGatedNMPC(thr, _Twin(brk))                                      # positive control
    with pytest.raises(ValueError, match="N / dt"):
        _DecelGatedNMPC(thr, NonlinearMPC(_sp("brake"), None, n_horizon=10, yaw_bound=math.pi / 2))
    with pytest.raises(ValueError, match="N / dt"):
        _DecelGatedNMPC(thr, NonlinearMPC(_sp("brake"), None, dt_mpc=0.05, yaw_bound=math.pi / 2))

    class Sub(_AlpaSimParams):
        pass
    t = _Twin(brk)
    t.p = Sub(**{k: getattr(brk.p, k) for k in brk.p.__dataclass_fields__})
    with pytest.raises(ValueError, match="types differ"):
        _DecelGatedNMPC(thr, t)
    t = _Twin(brk)
    t.p = copy.copy(brk.p)
    t.p.mass = brk.p.mass + 1.0
    with pytest.raises(ValueError, match="'mass' differs"):
        _DecelGatedNMPC(thr, t)
    with pytest.raises(ValueError, match="labels"):
        _DecelGatedNMPC(thr, _Twin(brk, opt_x=types.SimpleNamespace(labels=lambda: ["x"])))
    with pytest.raises(ValueError, match="nlp_cons_lb"):
        _DecelGatedNMPC(thr, _Twin(brk, nlp_cons_lb=casadi.DM.zeros(3)))
    sc = np.array(brk._mpc.opt_x_scaling.cat, dtype=float).ravel()
    sc[0] = 2.0
    with pytest.raises(ValueError, match="opt_x_scaling"):
        _DecelGatedNMPC(thr, _Twin(brk, opt_x_scaling=types.SimpleNamespace(cat=sc)))
    with pytest.raises(ValueError, match="same object"):
        _DecelGatedNMPC(thr, thr)


def test_a21_transfer_checks_shapes_before_assigning(deps):
    import casadi
    w, thr, brk = _pair()
    bad = casadi.DM.zeros(5)
    fake = types.SimpleNamespace(_mpc=types.SimpleNamespace(
        opt_x_num=types.SimpleNamespace(master=bad), opt_x_num_unscaled=types.SimpleNamespace(master=bad),
        u0=types.SimpleNamespace(cat=casadi.DM.zeros(2)), nlp_cons_lb=brk._mpc.nlp_cons_lb, flags={}))
    w._brk = fake
    before = np.array(thr._mpc.opt_x_num.master)
    with pytest.raises(ValueError, match="opt_x_num shape"):
        w.select_decel(True)
    assert fake._mpc.opt_x_num.master is bad and fake._mpc.flags == {}   # nothing assigned
    assert w.p is thr.p and np.array_equal(np.array(thr._mpc.opt_x_num.master), before)


# --- follower level (real NLPs) ---------------------------------------------------------------------

def _drive(f, n, v=4.0, x0=0.0, start_t=0.0):
    """n control steps along +x at constant speed (a synthetic state stream, the same for every follower);
    returns per-step (command, law output snapshot, gate stats)."""
    out = []
    for k in range(n):
        t = start_t + k * DT
        cmd = f.step((x0 + v * k * DT, 0.0), 0.0, v, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=t, gear=1)
        law = f._law
        out.append((tuple(cmd), (law.cmd_a, law.cmd_delta, law.mode, law.cmd_t, law.last_status),
                    f.mpc_stats["decel_gate"]))
    return out


def test_u3_brake_follower_equals_the_default_until_the_predicted_open_solve(deps):
    a = PathFollower(LONG, make_table(), mpc_params=_rec())
    b = PathFollower(LONG, make_table(), mpc_params=_rec(), mpc_decel_limit="brake")
    w = b._law.solver
    by = []
    for name in ("_thr", "_brk"):
        s = getattr(w, name)
        orig = s.solve

        def spy(x0, ref, _o=orig, _n=name):
            by.append(_n)
            return _o(x0, ref)
        s.solve = spy
    n = 300
    ra, rb = _drive(a, n), _drive(b, n)
    t_end = float(b._law.ref[0][-1])
    end_rig = LONG[-1][0] - RIG
    # predicted first open solve, from the definition (solves at every 4th step from k = 0)
    pred = next(k for k in range(0, n, 4)
                if k * DT + DT_MPC * 20 >= t_end - 1e-9 and end_rig - (4.0 * k * DT - RIG) > 0.0)
    first_open = next(k for k in range(n) if rb[k][2]["open"])
    assert first_open == pred
    for k in range(first_open):
        assert ra[k][0] == rb[k][0] and ra[k][1] == rb[k][1], k                 # bitwise, every step before
        assert rb[k][2]["open"] in (False, None)
    assert by[pred // 4] == "_brk" and set(by[:pred // 4]) == {"_thr"}         # the open solve used BRK
    assert ra[first_open][2] is None
    assert any(ra[k][0] != rb[k][0] for k in range(first_open, n))           # it does act after opening


def test_u7_first_solve_opens_with_a_never_solved_source(deps):
    from genesis_vehicle.control.mpc.alpasim_nonlinear import NonlinearMPC
    f = PathFollower(SHORT, make_table(), mpc_params=_rec(), mpc_decel_limit="brake")
    w = f._law.solver
    assert not hasattr(w._thr._mpc, "lam_x_num")
    seen = {}
    orig_solve, orig_init = w._brk.solve, f._law.initial_state

    def spy(x0, ref):
        seen["x0"], seen["ref"] = np.array(x0), np.array(ref)
        seen["u_prev"] = np.array(w._brk._mpc.u0.cat).ravel()
        r = orig_solve(x0, ref)
        seen["u_prev_used"] = np.array(w._brk._mpc.opt_p_num["_u_prev"]).ravel()
        return r

    def spy_init(*a):
        out = orig_init(*a)
        seen["d0a0"] = (out[1], out[2])
        return out
    w._brk.solve, f._law.initial_state = spy, spy_init
    f.step((0.0, 0.0), 0.0, 2.0, 0.0, 0.0, yaw_rate=0.1, v_lat=0.0, t=0.0, gear=1)
    st = f.mpc_stats
    assert st["decel_gate"] == {"open": True, "open_solves": 1, "switches": 1}
    assert st["status"].get("exception", 0) == 0 and f._law.last_result.exception is None
    # A19: this solve both switched and restarted; the NLP that solved used u_prev = (d0, a0)
    assert np.array_equal(seen["u_prev_used"], np.array(seen["d0a0"], dtype=float))
    cold = NonlinearMPC(_sp("brake"), None, yaw_bound=math.pi / 2)
    cold.set_u_prev(*seen["d0a0"])
    r = cold.solve(seen["x0"], seen["ref"])
    assert np.array_equal(r.u, f._law.last_result.u)                     # bitwise


def test_u6_a17_stats_present_for_every_mpc_follower(deps, lmpc_deps):
    for kw in ({}, {"mpc_impl": "linear"}, {"command_mode": "signed"}):
        f = PathFollower(LONG, make_table(), mpc_params=_rec(), **kw)
        assert "decel_gate" in f.mpc_stats and f.mpc_stats["decel_gate"] is None
        _drive(f, 5) if kw.get("command_mode") != "signed" else None
        assert f.mpc_stats["decel_gate"] is None
    for kw in ({}, {"mpc_impl": "linear"}):
        f = PathFollower(SHORT, make_table(), mpc_params=_rec(), mpc_decel_limit="brake", **kw)
        assert f.mpc_stats["decel_gate"] == {"open": None, "open_solves": 0, "switches": 0}
        _drive(f, 5, v=2.0)
        assert f.mpc_stats["decel_gate"] == {"open": True, "open_solves": 2, "switches": 1}
        f.set_path(LONG, pos_xy=(0.0, 0.0), t=1.0)
        assert f.mpc_stats["decel_gate"] == {"open": True, "open_solves": 2, "switches": 1}
        _drive(f, 1, start_t=1.0)
        assert f.mpc_stats["decel_gate"] == {"open": False, "open_solves": 2, "switches": 2}
        st = f.mpc_stats
        st["decel_gate"]["open"] = "x"
        assert f.mpc_stats["decel_gate"]["open"] is False                    # a copy


def test_a25_transfer_error_propagates_out_of_plan(deps, monkeypatch):
    from genesis_vehicle.control.mpc import alpasim_nonlinear as an

    def boom(src, dst):
        raise RuntimeError("injected transfer failure")
    monkeypatch.setattr(an, "_transfer_warm_start", boom)
    f = PathFollower(SHORT, make_table(), mpc_params=_rec(), mpc_decel_limit="brake")
    with pytest.raises(RuntimeError, match="injected transfer failure"):
        f.plan((0.0, 0.0), 0.0, 2.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=0.0, gear=1)
    st = f.mpc_stats
    assert st["status"] == {} and st["n_solve"] == 0
    assert f._law.mode != "PD-FALLBACK" and f.last_mode != "PD-FALLBACK"
    assert st["decel_gate"] == {"open": None, "open_solves": 0, "switches": 0}


# --- A23: construction ------------------------------------------------------------------------------

def _spy_ctor(monkeypatch, cls):
    calls = []
    orig = cls.__init__

    def spy(self, *a, **k):
        calls.append((a, dict(k)))
        orig(self, *a, **k)
    monkeypatch.setattr(cls, "__init__", spy)
    return calls


def test_a23_throttle_builds_exactly_the_2_0_6_solver(deps, monkeypatch):
    from genesis_vehicle.control.mpc import alpasim_nonlinear as an
    calls = _spy_ctor(monkeypatch, an.NonlinearMPC)
    f = PathFollower(LONG, make_table(), mpc_params=_rec())
    assert len(calls) == 1 and type(f._law.solver) is an.NonlinearMPC and not f._law.decel_gate
    (sp, gains), kw = calls[0]
    assert gains is None and kw == {"yaw_bound": math.pi / 2}
    assert sp.accel_min_tab == THROTTLE_TAB


def test_a23_brake_builds_two_by_the_same_call(deps, monkeypatch):
    from genesis_vehicle.control.mpc import alpasim_nonlinear as an
    calls = _spy_ctor(monkeypatch, an.NonlinearMPC)
    g = MPCGains()
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                   # no NARROWER warning here
        PathFollower(LONG, make_table(), mpc_params=_rec(), mpc_decel_limit="brake", mpc_gains=g)
    assert len(calls) == 2
    (s1, g1), k1 = calls[0]
    (s2, g2), k2 = calls[1]
    assert g1 is g and g2 is g and k1 == k2 == {"yaw_bound": math.pi / 2}
    assert s1.accel_min_tab == THROTTLE_TAB and s2.accel_min_tab == BRAKE_TAB


def test_a23_r_min_check_uses_the_throttle_row_params(deps, monkeypatch):
    orig = MPCVehicleParams.solver_params

    def sp(self, *a, decel_limit="throttle", **k):
        p = orig(self, *a, decel_limit=decel_limit, **k)
        if decel_limit == "brake":
            p.steer_max = 2.0                  # an R_min check on this would raise "steering limit"
        return p
    monkeypatch.setattr(MPCVehicleParams, "solver_params", sp)
    with pytest.raises(ValueError, match="parameter 'steer_max' differs"):
        PathFollower(LONG, make_table(), mpc_params=_rec(), mpc_decel_limit="brake")


# --- A24: the LMPC variant ----------------------------------------------------------------------------

def test_a24_lmpc_swaps_p_on_one_qp(deps, lmpc_deps, monkeypatch):
    from genesis_vehicle.control.mpc import alpasim_linear as al
    from genesis_vehicle.control.mpc.params import accel_limits
    calls = _spy_ctor(monkeypatch, al.LinearMPC)
    f = PathFollower(SHORT, make_table(), mpc_params=_rec(), mpc_impl="linear", mpc_decel_limit="brake")
    w = f._law.solver
    assert len(calls) == 1 and type(w) is al._DecelGatedLMPC
    assert not hasattr(w, "plan_vector") and not hasattr(w, "_mpc") and "__getattr__" not in type(w).__dict__
    inner, p_thr, p_brk = w._lmpc, w._p_thr, w._p_brk
    thr_tab, brk_tab = copy.deepcopy(p_thr.accel_min_tab), copy.deepcopy(p_brk.accel_min_tab)
    f.step((0.0, 0.0), 0.0, 2.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=0.0, gear=1)          # opens
    assert f.mpc_stats["decel_gate"]["open"] is True and inner.p is p_brk and w.p is p_brk
    lo = accel_limits(p_brk, 2.0)[0]
    assert inner._xmin[3] == lo and inner._umin[1] == lo and lo == -5.17
    assert f.last_plan is None
    w.select_decel(False)
    assert inner.p is p_thr
    inner.solve(np.array([0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0]), _ref(v=6.0))
    assert inner._xmin[3] == inner._umin[1] == accel_limits(p_thr, 6.0)[0] == -2.16
    assert p_thr.accel_min_tab == thr_tab and p_brk.accel_min_tab == brk_tab      # never mutated


def test_a24_lmpc_constructor_checks(deps, lmpc_deps):
    from genesis_vehicle.control.mpc.alpasim_linear import LinearMPC, _DecelGatedLMPC
    p_thr, p_brk = _sp("throttle"), _sp("brake")
    with pytest.raises(ValueError, match="throttle-row"):
        _DecelGatedLMPC(LinearMPC(p_brk), p_thr, p_brk)
    q = copy.copy(p_brk)
    q.wheelbase = 3.0
    with pytest.raises(ValueError, match="'wheelbase' differs"):
        _DecelGatedLMPC(LinearMPC(p_thr), p_thr, q)
