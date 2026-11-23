"""PathFollower K-turn (v2.0.3, opt-in ``mpc_kturn=KTurnParams()``): the follower side.

Fake solver (``tests/_mpc_fakes.py``), synthetic params and table (``make_mpc_field_replay``), gear mode, and
a small kinematic bicycle with the vehicle's shift rule (gear engages only at throttle 0 below 0.5 m/s) for
the closed-loop cases. Covers plan r203 section 5.1 / Amendment 1:

* construction: default ``mpc_kturn=None`` (off, ``mpc_stats["kturn"]`` zeros), explicit KTurnParams with
  signed / gate None -> ValueError, ``mpc_model="off"`` rejects it even as None, wrong types;
* the default follower never enters R-KTURN on the straight-6 trigger state and its commands equal an
  explicit ``mpc_kturn=None`` follower; the heading-error sign of the decision (api condition 1);
* the straight-6 episode: KTURN-STOP (2 slow calls) -> KTURN-PLAN over several calls -> legs with the
  leg gear stamped, steer sign, yaw-rate sign, signed leg end -> release (re-seed, anchor, gate);
* SHIFT / GUARD inside the episode keep it and count on the leg counter (D1); BRAKE_TRANS, DONE and a
  terminal mode cancel it; second R-ALIGN entry / after FALLBACK: no trigger (D8);
* GOAL-MISSED not evaluated on R-KTURN calls, three calls needed after the release;
* the caps at exactly the next call: KT-STOP 121st call, leg call cap, no-progress (81st driving call);
  ``lateral_limit`` too small -> ``kturn_infeasible`` without moving;
* re-plan (i) keep / (ii) new plan within the C1 budgets / (iii) continue (and N6 abort with a limit);
  set_path resets; no feedback (two slow SHIFT calls); bend domain on the stall course (C2);
* Amendment 2 Fix 1 (kept): no leg throttle before a call at |v_long| <= v_stop (KTURN-SETTLE), settle calls
  on the hand-over stop counter, a SHIFT/GUARD inside a leg requires a new standstill (A2-C1), the leg starts
  at its settling call (A2-C3), a re-settle keeps yaw0 / distance (A3-4);
* Amendment 3: legs at full lock (A3-2; steering to r_plan was measured in round 2 and withdrawn).
"""
import copy
import math
import os
import sys

import numpy as np
import pytest

pytest.importorskip("casadi")
pytest.importorskip("do_mpc")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control import PathFollower, KTurnParams  # noqa: E402
from genesis_vehicle.control import kturn as K  # noqa: E402
from genesis_vehicle.control import path_follower as pfm  # noqa: E402
from genesis_vehicle.control.mpc import MPCVehicleParams  # noqa: E402
from genesis_vehicle.tests._mpc_fakes import FakeSolver, ok  # noqa: E402
from genesis_vehicle.tests.data import make_mpc_field_replay as rp  # noqa: E402

DT = 0.025
RMIN = 2.7 / math.tan(0.35)


def densify(c, sp=0.5, v=2.0):
    out = []
    for (x0, y0), (x1, y1) in zip(c[:-1], c[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / sp))
        out += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, 0.0, v) for i in range(n)]
    out.append((c[-1][0], c[-1][1], 0.0, v))
    return out


S6 = densify([(0.0, 0.0), (-6.0, 0.0)])
S30 = densify([(0.0, 0.0), (-30.0, 0.0)])
STALL = [(30.0, 0.0, 0.0, 2.0), (10.0, -8.0, 0.0, 2.0), (-10.0, -8.0, 0.0, 2.0), (-30.0, 0.0, 0.0, 2.0)]


def _params():
    return MPCVehicleParams.from_dict(copy.deepcopy(rp.PARAMS_SDK))


def _f(path, kt="on", **kw):
    if kt == "on":
        kw["mpc_kturn"] = KTurnParams()
    elif kt == "none":
        kw["mpc_kturn"] = None
    elif kt is not None and kt != "default":
        kw["mpc_kturn"] = kt
    f = PathFollower(path, rp.make_table(), mpc_params=_params(), **kw)
    f._law.solver = FakeSolver(None, default=ok(0.0, 0.5))
    return f


class Car:
    """Kinematic bicycle (wheelbase 2.7, steer = steer_input_sign * s * 0.7 = -0.7 s), throttle in the
    engaged gear, brake decelerates toward 0 without crossing it; the gear engages only at throttle 0
    below 0.5 m/s (the vehicle's rule)."""

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, gear=1, b=1.0):
        self.x, self.y, self.yaw, self.v, self.gear, self.b = x, y, yaw, v, gear, b

    def apply(self, cmd):
        thr, st, br, g = cmd
        if g != self.gear and thr == 0.0 and abs(self.v) < 0.5:
            self.gear = g
        a = self.gear * thr * 3.0
        v = self.v + a * DT
        if br > 0.0:
            dv = 5.0 * br * DT
            v = 0.0 if abs(v) <= dv else v - math.copysign(dv, v)
        self.v = v
        delta = -0.7 * st
        self.yaw = K.wrap(self.yaw + self.b * self.v * math.tan(delta) / 2.7 * DT)
        self.x += self.v * math.cos(self.yaw) * DT
        self.y += self.v * math.sin(self.yaw) * DT

    def step(self, f, k, fb=True):
        kw = {"gear": self.gear} if fb else {}
        cmd = f.step((self.x, self.y), self.yaw, self.v, 0.0, 0.0,
                     yaw_rate=0.0, v_lat=0.0, t=k * DT, **kw)
        return cmd


def _drive(f, car, n, fb=True, stop=None):
    log = []
    for k in range(n):
        y0 = car.yaw
        cmd = car.step(f, k, fb)
        plan = f._kt_dbg_last if hasattr(f, "_kt_dbg_last") else None
        log.append(dict(k=k, mode=f.last_mode, regime=f.regime, cmd=tuple(cmd), yaw=y0,
                        kt=None if f._kt is None else dict(phase=f._kt["phase"], gear=f._kt["gear"],
                                                          leg=None if f._kt["leg"] is None else f._kt["leg"]["gdp"])))
        if f.last_mode in PathFollower.TERMINAL_MODES:
            break
        car.apply(cmd)
        if stop is not None and stop(f, log):
            break
    return log


# --- construction -------------------------------------------------------------------------------

def test_default_is_off_and_stats_present():
    f = _f(S6, kt="default")
    assert f._kturn is None and f._kt is None
    assert f.mpc_stats["kturn"] == dict(episodes=0, legs=0, reverse_m=0.0, plan_nodes=0, plan_calls=0, plan_ms=0.0)
    g = _f(S6, kt="default", command_mode="signed")
    assert g.mpc_stats["kturn"]["episodes"] == 0
    assert PathFollower(S6, rp.make_table(), mpc_model="off").mpc_stats is None


def test_kwarg_errors():
    with pytest.raises(ValueError, match="mpc_kturn"):
        PathFollower(S6, rp.make_table(), mpc_params=_params(), command_mode="signed", mpc_kturn=KTurnParams())
    with pytest.raises(ValueError, match="mpc_kturn"):
        PathFollower(S6, rp.make_table(), mpc_params=_params(), mpc_align_gate=None, mpc_kturn=KTurnParams())
    for v in (None, KTurnParams()):
        with pytest.raises(ValueError, match="mpc_kturn"):
            PathFollower(S6, rp.make_table(), mpc_model="off", mpc_kturn=v)
    for bad in ({}, "on", True, 4):
        with pytest.raises(ValueError, match="mpc_kturn"):
            PathFollower(S6, rp.make_table(), mpc_params=_params(), mpc_kturn=bad)
    # signed / gate None with the default or an explicit None: fine
    PathFollower(S6, rp.make_table(), mpc_params=_params(), command_mode="signed", mpc_kturn=None)
    PathFollower(S6, rp.make_table(), mpc_params=_params(), mpc_align_gate=None)
    # U1 is still reported first
    with pytest.raises(ValueError, match="mpc_params"):
        PathFollower(S6, rp.make_table(), mpc_kturn={})


def test_default_never_kturns_and_matches_explicit_none():
    a, b = _f(S6, kt="default"), _f(S6, kt="none")
    ca, cb = Car(), Car()
    la, lb = _drive(a, ca, 400), _drive(b, cb, 400)
    assert [r["cmd"] for r in la] == [r["cmd"] for r in lb]
    assert not any(r["regime"] == "R-KTURN" for r in la)
    assert la[0]["regime"] == "R-ALIGN"


def test_decision_heading_error_sign():
    f = _f(densify([(0.0, 0.0), (40.0, 0.0)]))
    f._kt_call = (0.0, 0.0, 0.3, 0.0)
    e, ylat, s_rem, b, bend, py = f._kt_geometry(0.0, 0.5, 0.3)
    assert e == pytest.approx(0.3) and ylat == pytest.approx(0.5) and py == 0.0
    assert e == pytest.approx(-f._heading_error(0.3))


def test_stall_course_bend_inside_domain_C2():
    f = _f(STALL)
    e, ylat, s_rem, b, bend, py = f._kt_geometry(30.0, 0.0, 0.0)
    assert math.degrees(bend) == pytest.approx(43.6, abs=0.1)
    assert s_rem == pytest.approx(63.08, abs=0.01) and math.degrees(e) == pytest.approx(158.2, abs=0.05)


# --- the episode --------------------------------------------------------------------------------

def _until_release(f, log):
    return f._kt is None and any(r["regime"] == "R-KTURN" for r in log)


def test_straight6_episode():
    f = _f(S6)
    starts = []
    real = f._law.start_block
    f._law.start_block = lambda w, t0=0.0: (starts.append(t0), real(w, t0))[1]
    car = Car()
    log = _drive(f, car, 3000, stop=_until_release)
    modes = [r["mode"] for r in log]
    assert modes[0] == "KTURN-STOP" and log[0]["regime"] == "R-KTURN"
    assert f.mpc_stats["handover"]["align"] == 0
    i_plan = modes.index("KTURN-PLAN")
    assert i_plan <= 2
    n_plan = sum(1 for m in modes if m == "KTURN-PLAN")
    assert n_plan >= 2                                # the search spans several calls
    legs = [r for r in log if r["mode"] in ("KTURN-REV", "KTURN-FWD")]
    assert legs and any(r["mode"] == "KTURN-REV" for r in legs)
    for r in legs:
        g, d, phi = r["kt"]["leg"]
        assert r["cmd"][3] == g                      # stamped leg gear
        assert r["cmd"][0] >= 0.0
        if r["cmd"][0] > 0:
            assert r["cmd"][1] == pytest.approx(d * g * 0.5 * -1.0)   # full lock, steer_input_sign -1
    st = f.mpc_stats["kturn"]
    assert st["episodes"] == 1 and 1 <= st["legs"] <= 4 and st["reverse_m"] > 0
    assert st["plan_calls"] >= n_plan and st["plan_nodes"] > 1000
    # the release step re-seeded the MPC reference (start_block twice: t0=0, then t_now - t_ref) and
    # armed the watchdog; nothing started the law before the release
    assert len(starts) == 2 and starts[0] == 0.0       # _reseed_block (a plain law start is one call)
    assert f._anchor is not None and f._anchor[0] == log[-1]["k"] * DT
    assert f._kt is None and f._kt_block == 0
    assert log[-1]["regime"] != "R-KTURN"


def test_leg_yaw_turns_with_d_and_ends_on_signed_rotation():
    f = _f(S6)
    car = Car()
    seen = []                                         # [d, phi, accumulated yaw change] per leg
    cur = None
    for k in range(3000):
        y0 = car.yaw
        cmd = car.step(f, k)
        leg = f._kt["leg"] if f._kt is not None else None
        if f.last_mode in ("KTURN-REV", "KTURN-FWD"):
            if cur is None or cur[3] is not leg:
                cur = [leg["gdp"][1], leg["gdp"][2], 0.0, leg]
                seen.append(cur)
        car.apply(cmd)
        if f.last_mode in ("KTURN-REV", "KTURN-FWD"):
            cur[2] += K.wrap(car.yaw - y0)
        if f._kt is None and k > 5:
            break
    assert len(seen) >= 2
    for d, phi, rot, _ in seen:
        assert d * rot > 0                            # yaw turns with the planned sign d
        assert d * rot >= phi - K.KTURN_LEG_TOL - 0.05   # and the leg ended on its rotation


def test_unsigned_rotation_does_not_end_a_leg():
    f = _f(S6)
    car = Car()
    k = _to_leg_accept(f, car)
    leg = f._kt["leg"]
    # the settling call (v 0) starts the leg at the car's yaw (A2-C3)
    f.step((car.x, car.y), car.yaw, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=998.975, gear=car.gear)
    assert leg["settled"] and leg["started"] and leg["yaw0"] == car.yaw
    g, d, phi = leg["gdp"]
    # turned the WRONG way by more than phi: not an end
    yaw = K.wrap(leg["yaw0"] - d * (phi + 0.1)) if phi + 0.1 < math.pi else K.wrap(leg["yaw0"] - d * 0.5)
    f.step((car.x, car.y), yaw, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=999.0, gear=car.gear)
    assert f._kt is not None and f._kt["leg"] is leg
    yaw2 = K.wrap(leg["yaw0"] + d * phi)
    f.step((car.x, car.y), yaw2, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=999.025, gear=car.gear)
    assert f._kt is None or f._kt["leg"] is not leg


def test_shift_and_guard_keep_the_episode_and_count_D1():
    f = _f(S6)
    car = Car()
    for k in range(2000):
        car.apply(car.step(f, k))
        if f._kt is not None and f._kt["phase"] == "LEG" and f._kt["gear"] == -1:
            break
    assert f._kt["gear"] == -1
    leg = f._kt["leg"]
    c0 = leg["calls"]
    cmd = f.step((car.x, car.y), car.yaw, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=50.0, gear=1)
    assert f.last_mode == "SHIFT" and f.regime == "R-KTURN" and f._kt is not None
    assert tuple(cmd) == (0.0, 0.0, 1.0, -1) and leg["calls"] == c0 + 1 and leg["drive"] == 0
    # no feedback: GUARD while rolling forward against the reverse leg gear
    g = _f(S6)
    car = Car()
    for k in range(2000):
        car.apply(car.step(g, k, fb=False))
        if g._kt is not None and g._kt["phase"] == "LEG" and g._kt["gear"] == -1 and g.last_mode == "KTURN-REV":
            break
    assert g.last_mode == "KTURN-REV"
    g.step((car.x, car.y), car.yaw, 1.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=60.0)
    assert g.last_mode == "GUARD" and g.regime == "R-KTURN" and g._kt is not None


def test_no_feedback_two_slow_shift_calls():
    f = _f(S6)
    car = Car()
    log = _drive(f, car, 2000, fb=False, stop=lambda f, log: log[-1]["mode"] == "KTURN-REV")
    i = [r["mode"] for r in log].index("KTURN-REV")
    assert [r["mode"] for r in log[i - 2:i]] == ["SHIFT", "SHIFT"]
    assert all(r["regime"] == "R-KTURN" for r in log[i - 2:i + 1])


def _to_leg(f, car, gear=None, fb=True):
    for k in range(2000):
        car.apply(car.step(f, k, fb))
        if f._kt is not None and f._kt["phase"] == "LEG" and f.last_mode.startswith("KTURN-") and (
                gear is None or f._kt["gear"] == gear):
            return k
    raise AssertionError("no leg")


def test_done_cancels():
    f = _f(S6)
    car = Car()
    k = _to_leg(f, car)
    f.step((-5.5, 0.0), car.yaw, -0.5, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + 1) * DT, gear=car.gear)
    assert f.last_mode == "DONE" and f._kt is None


def test_brake_trans_cancels():
    path = densify([(0.0, 0.0), (-6.0, 0.0)]) + densify([(-6.0, 0.0), (0.0, 0.0)], v=-2.0)
    f = _f(path)
    car = Car()
    k = _to_leg(f, car)
    f.step((-5.9, 0.0), car.yaw, 0.5, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + 1) * DT, gear=car.gear)
    assert f.last_mode == "BRAKE_TRANS" and f._kt is None and f._kt_block == 0


def test_terminal_gear_abort_cancels():
    f = _f(S6)
    car = Car()
    k = _to_leg(f, car, gear=-1)
    f._kt["leg"]["callcap"] = 10 ** 6
    for j in range(pfm.SHIFT_MAX_CALLS + 1):
        f.step((car.x, car.y), car.yaw, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + 1 + j) * DT, gear=1)
    assert f.last_mode == "GEAR-ABORT" and f._kt is None


def test_second_entry_and_fallback_do_not_trigger_D8():
    f = _f(S6)
    f._kt_block = 0                                  # the block was decided (fired or not)
    Car().step(f, 0)
    assert f.regime == "R-ALIGN" and f._kt is None
    # FALLBACK (no plan, no lateral limit): planner that finds nothing -> 2.0.2 alignment
    g = _f(S6)
    car = Car()
    car.apply(car.step(g, 0))
    assert g._kt is not None
    g._kt["planner"].s_rem0 = -100.0                 # nothing reaches the goal region
    log = _drive(g, car, 40)
    assert g._kt is None and log[-1]["regime"] == "R-ALIGN"
    assert g.mpc_stats["handover"]["align"] >= 1


def test_goal_missed_not_evaluated_on_kturn_calls():
    f = _f(S6)
    car = Car()
    k = _to_leg(f, car)
    # receding from the goal past its plane on the last segment, during a leg
    for j in range(6):
        f.step((-8.0 - 0.1 * j, 0.0), car.yaw, -1.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0,
               t=(k + 1 + j) * DT, gear=car.gear)
        assert f.last_mode != "GOAL-MISSED"
    assert f._kt is not None and f._gp_hist == []    # nothing appended during the episode
    # after a release, three calls are needed again
    f._kt_release(1.0, -8.5, 0.0)
    assert f._gp_hist == []


def test_stop_cap_exact():
    f = _f(S6)
    f.step((0.0, 0.0), 0.0, 2.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=0.0, gear=1)   # trigger at 2 m/s
    assert f.last_mode == "KTURN-STOP"
    for j in range(1, K_STOP := pfm.KTURN_STOP_MAX_CALLS):
        f.step((0.0, 0.0), 0.0, 2.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=j * DT, gear=1)
        assert f.last_mode == "KTURN-STOP", j
    f.step((0.0, 0.0), 0.0, 2.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=K_STOP * DT, gear=1)
    assert f.last_mode == "MPC-ABORT" and f.stop_reason == "kturn_budget" and f._kt is None


def test_leg_call_cap_formula_and_exact():
    f = _f(S6)
    car = Car()
    k = _to_leg(f, car)
    leg = f._kt["leg"]
    g, d, phi = leg["gdp"]
    cap = 1.25 * f._kt_r_plan * phi + 0.5
    if g < 0:
        cap = min(cap, f._kt_max_rev)
    assert leg["cap"] == pytest.approx(cap)
    assert leg["callcap"] == math.ceil(2 * cap / (K.KTURN_V * f.dt_ctrl)) + pfm.SHIFT_MAX_CALLS
    n0 = leg["calls"]
    leg["callcap"] = n0 + 10
    x = car.x
    for j in range(10):
        x += 0.3 * (-1) ** j                          # moves (no no-progress abort), no rotation
        f.step((x, car.y), leg["yaw0"], 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=10 + j * DT, gear=g)
        assert f._kt is not None and f.last_mode.startswith("KTURN-"), j
    f.step((x + 0.3, car.y), leg["yaw0"], 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=11.0, gear=g)
    assert f.last_mode == "MPC-ABORT" and f.stop_reason == "kturn_budget"


def _to_leg_accept(f, car):
    """Run until the call that accepted a plan (KTURN-PLAN, phase now LEG, no driving call yet)."""
    for k in range(2000):
        car.apply(car.step(f, k))
        if f._kt is not None and f._kt["phase"] == "LEG" and f.last_mode == "KTURN-PLAN":
            return k
    raise AssertionError("no plan accepted")


def test_no_progress_abort_exact():
    f = _f(S6)
    car = Car()
    k = _to_leg_accept(f, car)
    leg = f._kt["leg"]
    g = leg["gdp"][0]
    assert leg["drive"] == 0 and f._kt_noprog == 80
    for n in range(1, 81):                           # driving call 1 anchors; 2..80 stay inside 0.25 m
        cmd = f.step((car.x + 0.001 * n, car.y), leg["yaw0"], 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0,
                     t=(k + n) * DT, gear=g)
        assert f.last_mode.startswith("KTURN-") and cmd.throttle > 0, n
    f.step((car.x + 0.081, car.y), leg["yaw0"], 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + 81) * DT, gear=g)
    assert f.last_mode == "MPC-ABORT" and f.stop_reason == "kturn_blocked" and f._kt is None
    # 0.25 m of displacement re-anchors
    h = _f(S6)
    car = Car()
    k = _to_leg_accept(h, car)
    leg = h._kt["leg"]
    for n in range(1, 200):
        x = car.x + (0.26 if n % 60 == 0 else 0.0) * (n // 60)
        h.step((x, car.y), leg["yaw0"], 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + n) * DT, gear=leg["gdp"][0])
        assert h.last_mode != "MPC-ABORT", n


def test_lateral_limit_too_small_is_infeasible():
    f = _f(S6, kt=KTurnParams(lateral_limit=2.0))
    car = Car()
    log = _drive(f, car, 200)
    assert log[-1]["mode"] == "MPC-ABORT" and f.stop_reason == "kturn_infeasible"
    assert math.hypot(car.x, car.y) <= 0.5
    assert all(r["mode"] in ("KTURN-STOP", "KTURN-PLAN", "MPC-ABORT") for r in log)


def test_lateral_limit_also_triggers():
    f = _f(S30, kt=KTurnParams(lateral_limit=8.0))
    Car().step(f, 0)
    assert f._kt is not None and f._kt_last_decision["margin"] < 0
    g = _f(S30)
    Car().step(g, 0)
    assert g._kt is None and g.regime == "R-ALIGN"


def _stopped_after_first_leg(kt_params=None):
    f = _f(S6, kt=kt_params or "on")
    car = Car()
    for k in range(3000):
        car.apply(car.step(f, k))
        if f._kt is not None and f._kt["executed"] == 1 and f._kt["phase"] == "STOP":
            return f, car, k
    raise AssertionError("no second stop")


def _hook_starts(f):
    calls = []
    pl = f._kt["planner"]
    real = pl.start
    pl.start = lambda *a, **kw: (calls.append(kw), real(*a, **kw))[1]
    return calls


def _run_to_leg(f, car, k):
    while f._kt is not None and f._kt["phase"] != "LEG":
        k += 1
        f.step((car.x, car.y), car.yaw, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=k * DT, gear=car.gear)
    return k


def test_replan_keep_new_continue_and_budgets_C1():
    # (i) keep: the remaining plan still valid -> kept, no search
    f, car, k = _stopped_after_first_leg()
    rem = list(f._kt["legs"])
    f._kt["planner"].remaining_ok = lambda *a: True
    calls = _hook_starts(f)
    _run_to_leg(f, car, k)
    assert f._kt["legs"] == rem and calls == []
    # (ii) a new plan within the budgets CARRIED across re-plans: after two reverse legs (R,R) no third
    # reverse leg (R,R,R,F refused) and at most max_legs - executed legs
    g, car2, k2 = _stopped_after_first_leg()
    g._kt["rev_executed"] = 2
    g._kt["executed"] = 2
    g._kt["legs"] = [(1, -1, math.radians(45))]         # consistent with the budgets, invalid from here
    g._kt["planner"].remaining_ok = lambda *a: False
    calls = _hook_starts(g)
    _run_to_leg(g, car2, k2)
    assert calls == [dict(legs_left=2, rev_left=0)]
    assert g._kt is None or all(l[0] > 0 for l in g._kt["legs"])
    h, car3, k3 = _stopped_after_first_leg()
    calls = _hook_starts(h)
    h._kt["planner"].remaining_ok = lambda *a: False
    _run_to_leg(h, car3, k3)
    ex = h._kt["executed"] if h._kt is not None else 1
    assert calls == [dict(legs_left=4 - 1, rev_left=2 - 0)]


def test_replan_iii_continue_and_N6():
    f, car, k = _stopped_after_first_leg()
    old = [(1, 1, math.radians(15))]
    f._kt["legs"] = list(old)
    f._kt["planner"].goal = lambda *a: False          # neither keep nor a new plan
    while f._kt is not None and f._kt["phase"] != "LEG":
        k += 1
        f.step((car.x, car.y), car.yaw, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=k * DT, gear=car.gear)
    assert f._kt is not None and f._kt["legs"] == old            # (iii) continued
    g, car2, k2 = _stopped_after_first_leg(KTurnParams(lateral_limit=40.0))
    g._kt["legs"] = list(old)
    g._kt["planner"].goal = lambda *a: False
    g._kt["planner"].lat = 1e-3                                  # the old plan leaves the limit
    while g._kt is not None and g.last_mode != "MPC-ABORT":
        k2 += 1
        g.step((car2.x, car2.y), car2.yaw, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=k2 * DT, gear=car2.gear)
    assert g.stop_reason == "kturn_infeasible"


def test_set_path_resets_and_cancels():
    f = _f(S6)
    car = Car()
    _to_leg(f, car)
    f.set_path(S6, pos_xy=(car.x, car.y))
    assert f._kt is None and f._kt_block is None
    assert f.mpc_stats["kturn"]["episodes"] == 1                  # stats survive set_path


def test_plan_steer_cmd_only_on_leg_driving_calls():
    f = _f(S6)
    car = Car()
    seen = set()
    for k in range(600):
        p = f.plan((car.x, car.y), car.yaw, car.v, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=k * DT, gear=car.gear)
        if p.steer_cmd is not None:
            assert p.regime == "R-KTURN" and p.mode in ("KTURN-REV", "KTURN-FWD") and p.command is None
            assert p.omega_target == pytest.approx((1 if p.steer_cmd * p.gear < 0 else -1) * 1.0 / f._kt_r_plan)
            seen.add(p.mode)
        car.apply(f.finish(p, f.plant.throttle_for(car.v, p.a_target, 0.0, 0.0),
                           p.steer_cmd if p.steer_cmd is not None else 0.0))
    assert seen


# --- AC-6: the default follower's commands are the 2.0.2 commands (fixture captured on 76a6b24) -----

_FIX = os.path.join(os.path.dirname(__file__), "data", "kturn_off_2_0_2.json")


def _replay(run, mode, **kw):
    f = PathFollower([tuple(p) for p in run["path"]], rp.make_table(), mpc_params=_params(),
                     command_mode=mode, **kw)
    f._law.solver = FakeSolver(None, default=ok(0.0, 0.5))
    out = []
    for k, (x, y, yaw, vl, w, vlat, g) in enumerate(run["states"]):
        gk = {"gear": g} if mode == "gear" else {}
        cmd = list(f.step((x, y), yaw, vl, 0.0, 0.0, yaw_rate=w, v_lat=vlat, t=k * 0.025, **gk))
        out.append(dict(cmd=[float(a) for a in cmd[:3]] + ([int(cmd[3])] if mode == "gear" else []),
                        mode=f.last_mode, regime=f.regime, sr=f.stop_reason))
    return out


@pytest.mark.parametrize("key", ["straight6/gear", "straight6/signed", "v150/gear", "v150/signed",
                                 "stall/gear", "stall/signed"])
def test_default_and_explicit_none_match_the_2_0_2_capture(key):
    import json
    fx = json.load(open(_FIX))
    assert fx["sdk"] == "2.0.2"
    run = fx["runs"][key]
    mode = key.split("/")[1]
    assert _replay(run, mode) == run["out"]
    assert _replay(run, mode, mpc_kturn=None) == run["out"]
    if key == "straight6/gear":
        assert run["out"][0]["regime"] == "R-ALIGN"           # the fixture holds a trigger state
        on = _replay(run, mode, mpc_kturn=KTurnParams())
        assert on[0]["regime"] == "R-KTURN"                    # ...on which an enabled follower differs


# --- Amendment 2: standstill before every leg (Fix 1) and the planned-radius steer (Fix 2) -------

def _episode(f, car, n=3000, fb=True):
    """Drive until the episode is released; returns the per-call log (mode, cmd, v_long before, engaged
    gear before, leg id)."""
    log = []
    started = False
    for k in range(n):
        v0, g0 = car.v, car.gear
        cmd = car.step(f, k, fb)
        leg = f._kt["leg"] if f._kt is not None else None
        log.append(dict(mode=f.last_mode, regime=f.regime, cmd=tuple(cmd), v=v0, eng=g0,
                        leg=id(leg) if leg is not None else None,
                        n_stop=f._kt["n_stop"] if f._kt is not None else None))
        started = started or f._kt is not None
        if f.last_mode in PathFollower.TERMINAL_MODES or (started and f._kt is None):
            break
        car.apply(cmd)
    return log


@pytest.mark.parametrize("fb", [True, False])
def test_no_throttle_before_standstill_in_any_handover(fb):
    f = _f(S6)
    car = Car()
    log = _episode(f, car, fb=fb)
    assert f.mpc_stats["kturn"]["legs"] >= 3
    seen_still = set()
    for r in log:
        if r["leg"] is not None and abs(r["v"]) <= f.v_stop:
            seen_still.add(r["leg"])
        if r["regime"] == "R-KTURN" and r["cmd"][0] > 0:
            assert r["leg"] in seen_still, r            # throttle only after a call at |v| <= v_stop
        # the AC-3 rule (with and without feedback)
        assert not (r["cmd"][0] > 0 and r["eng"] * r["v"] < -0.1), r
    if fb:      # without feedback the two slow SHIFT calls already brake this kinematic car to rest
        assert any(r["mode"] == "KTURN-SETTLE" for r in log)
    for r in log:
        if r["mode"] == "KTURN-SETTLE":
            assert r["cmd"][:3] == (0.0, 0.0, 1.0) and r["regime"] == "R-KTURN"


def test_settle_stamped_with_leg_gear_counted_on_n_stop_and_settling_call_drives():
    f = _f(S6)
    car = Car()
    k = _to_leg_accept(f, car)
    leg = f._kt["leg"]
    g = leg["gdp"][0]
    n0 = f._kt["n_stop"]
    cmd = f.step((car.x, car.y), car.yaw, 0.3 * g, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + 1) * DT, gear=g)
    assert f.last_mode == "KTURN-SETTLE" and cmd.gear == g and cmd.throttle == 0.0 and cmd.brake == 1.0
    assert f._kt["n_stop"] == n0 + 1 and not leg["settled"] and leg["calls"] == 0
    cmd = f.step((car.x + 0.01, car.y), car.yaw + 0.01, 0.04 * g, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0,
                 t=(k + 2) * DT, gear=g)
    assert f.last_mode in ("KTURN-REV", "KTURN-FWD") and cmd.throttle > 0     # same-call proceed
    assert leg["settled"] and leg["yaw0"] == car.yaw + 0.01 and leg["dist"] == 0.0   # A2-C3 capture
    assert leg["px"] == car.x + 0.01 and leg["drive"] == 1


def test_never_settling_aborts_on_the_121st_counted_call_of_the_handover():
    f = _f(S6)
    car = Car()
    k = _to_leg_accept(f, car)
    g = f._kt["leg"]["gdp"][0]
    n0 = f._kt["n_stop"]                     # KT-STOP calls of this hand-over already counted
    assert n0 >= 1
    j = 0
    while True:
        j += 1
        f.step((car.x, car.y), car.yaw, 0.2 * g, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + j) * DT, gear=g)
        if f.last_mode != "KTURN-SETTLE":
            break
        assert j < 200
    assert f.last_mode == "MPC-ABORT" and f.stop_reason == "kturn_budget"
    assert n0 + j == pfm.KTURN_STOP_MAX_CALLS + 1


def test_shift_or_guard_inside_a_leg_requires_a_new_standstill_A2_C1():
    f = _f(S6)
    car = Car()
    k = _to_leg_accept(f, car)
    leg = f._kt["leg"]
    g = leg["gdp"][0]
    f.step((car.x, car.y), car.yaw, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + 1) * DT, gear=g)
    assert leg["settled"]
    yaw0, d0 = leg["yaw0"], leg["dist"]
    # GUARD (no feedback, rolling against the leg gear) then slow calls rolling 0.2 m/s against it
    f.step((car.x, car.y), car.yaw, -0.5 * g, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + 2) * DT)
    assert f.last_mode == "GUARD" and not leg["settled"]
    modes = []
    for j in range(3, 12):
        cmd = f.step((car.x, car.y), car.yaw, -0.2 * g, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + j) * DT)
        modes.append(f.last_mode)
        assert cmd.throttle == 0.0, (j, f.last_mode)
    assert "KTURN-SETTLE" in modes                      # GUARD released, still not at standstill
    cmd = f.step((car.x, car.y), car.yaw, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + 12) * DT)
    assert cmd.throttle > 0 and leg["settled"] and leg["yaw0"] == yaw0     # re-settle keeps the leg start


def test_leg_end_not_tested_during_settle_A2_C3():
    f = _f(S6)
    car = Car()
    k = _to_leg_accept(f, car)
    leg = f._kt["leg"]
    g, d, phi = leg["gdp"]
    # rolling (not settled) with a yaw / position that would end the leg by rotation and by the distance cap
    f.step((car.x + leg["cap"] + 1.0, car.y), K.wrap(car.yaw + d * phi), 0.3 * g, 0.0, 0.0,
           yaw_rate=0.0, v_lat=0.0, t=(k + 1) * DT, gear=g)
    assert f.last_mode == "KTURN-SETTLE" and f._kt["leg"] is leg and f._kt["executed"] == 0


def test_leg_steer_is_full_lock_A3_2():
    """Amendment 3: legs at full lock, steer_cmd = d*g*steer_cap*steer_input_sign on every driving call
    (steering open-loop to r_plan was measured in round 2 and withdrawn)."""
    f = _f(S6)
    assert not hasattr(f, "_kt_steer_ff")
    car = Car()
    log = _episode(f, car)
    n = 0
    for r in log:
        if r["mode"] in ("KTURN-REV", "KTURN-FWD"):
            n += 1
            assert abs(r["cmd"][1]) == f.steer_cap
    assert n > 100


def test_guard_resettle_keeps_leg_start_heading_and_distance_A3_4():
    """Amendment 3 / A3-4 (deviation (i)): a mid-leg SHIFT/GUARD re-settle re-captures only the position;
    the leg's start heading yaw0 and its distance stay, so the leg distance cap is not reset; the re-settle
    calls are charged to the hand-over's stop counter."""
    f = _f(S6)
    car = Car()
    k = _to_leg_accept(f, car)
    leg = f._kt["leg"]
    g = leg["gdp"][0]
    x0, y0, yaw = car.x, car.y, car.yaw
    f.step((x0, y0), yaw, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + 1) * DT, gear=g)   # settles
    yaw_start = leg["yaw0"]
    for j in range(2, 7):                                    # drive 0.2 m per call, slight heading change
        f.step((x0 + 0.2 * j * math.cos(yaw), y0 + 0.2 * j * math.sin(yaw)), yaw + 0.001 * j, 0.8 * g,
               0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + j) * DT, gear=g)
    d_before = leg["dist"]
    assert d_before > 0.8 and leg["yaw0"] == yaw_start
    px, py = x0 + 1.2 * math.cos(yaw), y0 + 1.2 * math.sin(yaw)
    n0 = f._kt["n_stop"]
    f.step((px, py), yaw + 0.006, -0.5 * g, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + 7) * DT)     # GUARD
    assert f.last_mode == "GUARD" and not leg["settled"]
    for j in range(8, 30):
        f.step((px, py), yaw + 0.006, -0.2 * g, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + j) * DT)
        if f.last_mode == "KTURN-SETTLE":
            break
    assert f.last_mode == "KTURN-SETTLE" and f._kt["n_stop"] > n0
    f.step((px, py), yaw + 0.006, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + 31) * DT)       # re-settles
    assert leg["settled"] and leg["yaw0"] == yaw_start
    assert leg["dist"] >= d_before                          # not reset to 0
    assert (leg["px"], leg["py"]) == (px, py)               # position re-captured


def test_leg_distance_cap_ends_a_leg_on_the_exact_call():
    """impl-review S2: a leg that never reaches its rotation ends on the call at which its distance reaches
    the cap (forward 1.25 r_plan phi + 0.5; reverse also <= max_reverse)."""
    f = _f(S6)
    car = Car()
    k = _to_leg_accept(f, car)
    leg = f._kt["leg"]
    g = leg["gdp"][0]
    yaw = car.yaw
    f.step((car.x, car.y), yaw, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + 1) * DT, gear=g)  # settles
    assert leg["settled"] and leg["dist"] == 0.0
    step_m = 0.3
    n_end = math.ceil(leg["cap"] / step_m - 1e-9)
    ex0 = f._kt["executed"]
    for j in range(1, n_end + 1):
        x = car.x + step_m * j * math.cos(yaw) * g
        y = car.y + step_m * j * math.sin(yaw) * g
        f.step((x, y), yaw, 0.5 * g, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + 1 + j) * DT, gear=g)
        if j < n_end:
            assert f._kt is not None and f._kt["leg"] is leg and f._kt["executed"] == ex0, j
    assert f._kt is None or (f._kt["leg"] is not leg and f._kt["executed"] == ex0 + 1)
    assert leg["dist"] >= leg["cap"] and leg["dist"] - leg["cap"] < step_m + 1e-9   # one call of overrun


def test_resync_gear_during_a_leg_reshifts_and_resettles():
    """impl-review note 7: resync_gear() inside a leg (no feedback) -> SHIFT calls on the leg counter
    (episode kept), then a new standstill before throttle; the leg start is kept."""
    f = _f(S6)
    car = Car()
    k = _to_leg_accept(f, car)
    leg = f._kt["leg"]
    g = leg["gdp"][0]
    f.step((car.x, car.y), car.yaw, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + 1) * DT)        # settles, drives
    assert leg["settled"]
    yaw0, c0 = leg["yaw0"], leg["calls"]
    f.resync_gear()
    modes = []
    for j in range(2, 6):
        cmd = f.step((car.x, car.y), car.yaw, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + j) * DT)
        modes.append(f.last_mode)
    assert modes[:2] == ["SHIFT", "SHIFT"] and f.regime == "R-KTURN" and f._kt is not None
    assert modes[2] in ("KTURN-REV", "KTURN-FWD") and cmd.throttle > 0    # v 0 -> the re-settle call drives
    assert f._kt["leg"] is leg and leg["yaw0"] == yaw0 and leg["calls"] > c0


def test_resync_gear_with_a_vehicle_that_never_slows_is_bounded():
    """resync_gear() inside a leg while the vehicle keeps moving at 1 m/s (never slow): the SHIFT never
    completes and the run ends on GEAR-ABORT after SHIFT_MAX_CALLS (bounded, no hang)."""
    f = _f(S6)
    car = Car()
    k = _to_leg_accept(f, car)
    g = f._kt["leg"]["gdp"][0]
    f.step((car.x, car.y), car.yaw, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + 1) * DT)
    f.resync_gear()
    n = 0
    while f.last_mode not in PathFollower.TERMINAL_MODES:
        n += 1
        f.step((car.x, car.y), car.yaw, 1.0 * g, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=(k + 1 + n) * DT)
        assert n <= pfm.SHIFT_MAX_CALLS + 1
    assert f.last_mode == "GEAR-ABORT" and f._kt is None and n == pfm.SHIFT_MAX_CALLS + 1
