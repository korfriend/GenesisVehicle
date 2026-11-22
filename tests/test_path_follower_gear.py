"""Gear-mode ``PathFollower`` (v2.0.2, ``command_mode="gear"``), Genesis-free.

States are fed directly; plants are the synthetic sweep table of the PD golden
(``tests/data/make_pd_golden.make_table``) or fakes. Covers plan_r202 section 3.3:

(i)   golden replay ORACLE: the table scenarios of ``pd_golden_1x.json`` replayed (the
      recorded states fed open loop, feedback = the gear requested on the previous
      call) through a gear follower; every step compared with ``==`` against an
      INDEPENDENT re-implementation of the section 3.3 rules written here (it imports
      neither ``control.brake.brake_for`` nor ``PathFollower._finish_gear``), SHIFT
      steps included. Cross-check with the 1.x golden where the two must agree: on
      steps where ``gear * golden_throttle >= 0.02`` the gear command's (throttle,
      brake) equals (gear * golden_throttle, golden_brake); on the brake-at-rest steps
      both brake 1.
(ii)  feedback SHIFT, its timeout -> GEAR-ABORT; without feedback the two-slow-call
      rule, GUARD latch / release, the KICK guard (a state with u_g < 1, C5), the
      GUARD position after GOAL-MISSED.
(iii) set_path at 8 m/s without feedback: no SHIFT / brake; resync_gear() always
      SHIFTs (C1); set_path resets the counters and the GUARD latch, keeps the gear,
      clears GEAR-ABORT.
(iv)  SHIFT calls on the final block do not feed GOAL-MISSED (m3).
(v)   default command_mode, GearCommand unpacking / indexing.
(vi)  FleetFollower: one command_mode, one batched coast prediction.
(vii) construction errors: no accel_at, no brake gain source, a non-callable
      ``brake_gain`` attribute; the MPC gear finish (brake instead of a negative
      throttle, MPC-HOLD kept).

Mutants this file must fail on: the KICK guard removed ((ii) C5 case); GUARD removed
((ii) latch); the two-slow-call rule reduced to one ((ii)); the ``_gp_hist``
restore removed ((iv)); ``brake_for``'s sign flipped ((i) oracle); the construction
motion rule re-applied after resync_gear ((iii) C1).

| abbr | meaning |
|---|---|
| u / u_g | the plant's signed throttle / ``gear * u`` |
| k_b | brake gain (m/s^2 per unit brake) |
| C1 / C5 / m3 | plan-review conditions / item (resync -> SHIFT; KICK case with u_g < 1; GOAL-MISSED history) |
"""
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control import FleetFollower, GearCommand, PathFollower  # noqa: E402
from genesis_vehicle.control.path_follower import (SHIFT_MAX_CALLS,  # noqa: E402
                                                   SHIFT_SLOW_SPEED, V_AGAINST)
from genesis_vehicle.tests.data import make_pd_golden as gold  # noqa: E402

K_B = 4.0
_GOLDEN = os.path.join(os.path.dirname(__file__), "data", "pd_golden_1x.json")
FWD = gold.densify([(0.0, 0.0, 0.0, 2.0), (20.0, 0.0, 0.0, 2.0)])
REV = gold.densify([(0.0, 0.0, 0.0, -1.5), (-15.0, 0.0, 0.0, -1.5)])


def _gf(path, **kw):
    kw.setdefault("brake_gain", K_B)
    return PathFollower(path, gold.make_table(), mpc_model="off", command_mode="gear", **kw)


# --- (i) golden replay oracle ------------------------------------------------------

def _oracle_coast(table, v, p, r, g):
    """Amendment 1 Rev 1 (R1.1), written independently of control.brake: the table's
    flat coast below its first non-zero speed faded by tanh(|v|/0.5), the slope term
    read at w = max(|v|, v_h), the flat part mirrored on a side with no grid speed."""
    import numpy as np
    f0 = table._throttle_lu.forward
    ax = np.unique(table.columns["v_init"][table.columns["steer"] == 0.0])
    sides = {1.0: min([a for a in ax if a > 0], default=None),
             -1.0: max([a for a in ax if a < 0], default=None)}
    s = (1.0 if v > 0 else -1.0) if abs(v) > 1e-3 else float(g)
    sg = s if sides[s] is not None else -s
    vh, u = abs(sides[sg]), abs(v)
    if u >= vh and sg == s:
        return f0(v, 0.0, p, r)
    w = sg * max(u, vh)
    G = f0(w, 0.0, p, r) - f0(w, 0.0, 0.0, 0.0)
    flat = f0(w, 0.0, 0.0, 0.0) if u >= vh else \
        f0(sg * vh, 0.0, 0.0, 0.0) * math.tanh(u / 0.5) / math.tanh(vh / 0.5)
    return G + (flat if sg == s else -flat)


def _oracle_finish(table, plan, u, st, g, steer_cap):
    """Section 3.3 gear finish, written independently of the production code."""
    if plan.command is not None:
        return (0.0, 0.0, 1.0, g)
    v, vt = plan.v_long, plan.v_target
    ug = g * u
    kick = (abs(vt) > 0.1 and (1.0 if vt > 0 else -1.0) == g
            and v * g < 0.4 * abs(vt) and v * g >= -0.05)
    if kick:
        thr, brk = 1.0, 0.0
    elif ug >= 0:
        thr, brk = min(ug, 1.0), 0.0
    else:
        a_coast = _oracle_coast(table, v, plan.pitch_deg, plan.roll_deg, g)
        s = (1.0 if v > 0 else -1.0) if abs(v) > 1e-3 else float(g)
        thr, brk = 0.0, min(1.0, max(0.0, s * (a_coast - plan.a_target) / K_B))
    stc = max(-steer_cap, min(steer_cap, st))
    if thr < 0.02 and abs(v) < 0.1:
        brk = 1.0
    return (thr, stc, brk, g)


def _golden_runs():
    with open(_GOLDEN) as fh:
        return json.load(fh)["runs"]


K_STAR = {"square/table": 665, "square_tuned/table": 684}
TABLE_KEYS = sorted(k for k in _golden_runs() if k.endswith("/table"))


@pytest.mark.parametrize("key", TABLE_KEYS)
def test_gear_finish_matches_an_independent_oracle_on_the_golden_states(key):
    name = key.split("/")[0]
    path, kw, _, _ = gold.scenarios()[name]
    steps = _golden_runs()[key]["steps"]
    f = _gf(path, **kw)
    table = gold.make_table()
    fb = 1                                   # the vehicle starts in forward gear
    n_shift = n_cross = n_brake_steps = 0
    for k, (x, y, yaw, v, pitch, gthr, gst, gbrk, gmode, _bi, _ci) in enumerate(steps):
        desired_before = f._desired_gear()
        plan = f.plan((x, y), yaw, v, pitch, 0.0, gear=fb)
        desired = f._desired_gear()
        if plan.command is None:
            u = table.throttle_for(v, plan.a_target, pitch, 0.0)
            st = table.steer_for(v, plan.omega_target, pitch, 0.0)
        else:
            u = st = 0.0
        cmd = f.finish(plan, u, st)
        assert isinstance(cmd, GearCommand)
        # SHIFT exactly where the feedback differs from the wanted gear (and the
        # step is not already decided by DONE / a terminal / the cusp brake).
        decided = plan.mode in ("DONE", "BRAKE_TRANS", "GOAL-MISSED")
        assert (plan.mode == "SHIFT") == (not decided and fb != desired), (k, plan.mode, fb)
        assert plan.mode != "GUARD"          # feedback always confirms -> never GUARD
        want = _oracle_finish(table, plan, u, st, desired if plan.mode != "DONE"
                              else fb, f.steer_cap)
        assert (cmd.throttle, cmd.steer, cmd.brake, cmd.gear) == want, (k, plan.mode, cmd, want)
        n_shift += plan.mode == "SHIFT"
        n_brake_steps += plan.command is None and desired * u < 0
        # cross-check with the 1.x golden where both must agree
        if (k < K_STAR.get(key, len(steps)) and plan.mode == gmode
                and desired == desired_before):
            if desired * gthr >= 0.02:
                assert (cmd.throttle, cmd.brake) == (desired * gthr, gbrk), (k, cmd, gthr)
                n_cross += 1
            elif abs(gthr) < 0.02 and abs(v) < 0.1:
                assert cmd.brake == 1.0 and gbrk == 1.0
        fb = cmd.gear
        if f.last_mode == "DONE":
            break
    if "cusp" in name:
        assert n_shift >= 1                   # one per gear-changing block switch
    if name != "at_rest":
        assert n_cross > 0


def _speed_drop_path():
    """Forward 3 m/s then 0.5 m/s, cusp, reverse 1.5 m/s then 0.3 m/s: the PD asks for
    decelerations in both gears (the golden states never brake in gear terms)."""
    return (gold.densify([(0.0, 0.0, 0.0, 3.0), (12.0, 0.0, 0.0, 3.0)])
            + gold.densify([(12.0, 0.0, 0.0, 0.5), (18.0, 0.0, 0.0, 0.5)])[1:]
            + gold.densify([(18.0, 0.0, 0.0, -1.5), (8.0, 0.0, 0.0, -1.5)])
            + gold.densify([(8.0, 0.0, 0.0, -0.3), (5.0, 0.0, 0.0, -0.3)])[1:])


def test_gear_finish_matches_the_oracle_while_braking_in_both_gears():
    """Closed loop on a unicycle (throttle in the gear's direction, brake = a
    deceleration of 3 m/s^2 per unit against the motion), the same independent
    oracle as above on every step; braking steps (u_g < 0 with |v| > 0) must occur
    in both gears, so a flipped brake sign cannot pass."""
    f = _gf(_speed_drop_path())
    table = gold.make_table()
    x = y = yaw = v = 0.0
    fb, dt = 1, 0.05
    n_brake = {1: 0, -1: 0}
    for k in range(4000):
        plan = f.plan((x, y), yaw, v, 0.0, 0.0, gear=fb)
        if plan.command is None:
            u = table.throttle_for(v, plan.a_target, 0.0, 0.0)
            st = table.steer_for(v, plan.omega_target, 0.0, 0.0)
        else:
            u = st = 0.0
        c = f.finish(plan, u, st)
        g = f._desired_gear() if plan.mode != "DONE" else fb
        want = _oracle_finish(table, plan, u, st, g, f.steer_cap)
        assert tuple(c) == want, (k, plan.mode, tuple(c), want)
        if plan.command is None and g * u < 0 and abs(v) > 1e-3 and c.brake > 0:
            n_brake[g] += 1
        if f.last_mode == "DONE":
            break
        fb = c.gear                                        # the vehicle engages at once
        a = 3.0 * c.gear * c.throttle - 0.5 * v
        if abs(v) > 1e-9:
            a -= math.copysign(3.0 * c.brake, v)
        v_new = v + a * dt
        if c.brake > 0 and v * v_new < 0:
            v_new = 0.0                                    # the brake cannot reverse
        v = v_new
        yaw += 2.0 * st * dt
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
    assert f.last_mode == "DONE"
    assert n_brake[1] > 0 and n_brake[-1] > 0, n_brake


# --- (ii) SHIFT / GUARD / KICK guard -------------------------------------------------

def test_feedback_shift_repeats_then_gear_abort():
    f = _gf(FWD)
    for k in range(SHIFT_MAX_CALLS):
        c = f.step((0.0, 0.0), 0.0, 0.0, 0.0, 0.0, gear=-1)
        assert f.last_mode == "SHIFT" and tuple(c) == (0.0, 0.0, 1.0, 1), (k, c)
    c = f.step((0.0, 0.0), 0.0, 0.0, 0.0, 0.0, gear=-1)
    assert f.last_mode == "GEAR-ABORT" and f.stop_reason == "shift_timeout"
    # a terminal step keeps the gear the follower believes engaged (-1, the
    # feedback), not the wanted one (+1)
    assert c.gear == -1 and f._desired_gear() == 1
    assert "GEAR-ABORT" in PathFollower.TERMINAL_MODES
    f.step((0.0, 0.0), 0.0, 0.0, 0.0, 0.0, gear=1)                 # sticky
    assert f.last_mode == "GEAR-ABORT"
    f.set_path(FWD)                                               # set_path resumes
    f.step((0.0, 0.0), 0.0, 0.0, 0.0, 0.0, gear=1)
    assert f.last_mode == "DRV+1" and f.stop_reason is None


def test_a_driving_call_resets_the_shift_count():
    f = _gf(FWD)
    for _ in range(SHIFT_MAX_CALLS):
        f.step((0.0, 0.0), 0.0, 0.0, 0.0, 0.0, gear=-1)
    f.step((0.0, 0.0), 0.0, 0.0, 0.0, 0.0, gear=1)
    assert f.last_mode == "DRV+1"
    for _ in range(SHIFT_MAX_CALLS):
        f.step((0.0, 0.0), 0.0, 0.0, 0.0, 0.0, gear=-1)
        assert f.last_mode == "SHIFT"


def test_without_feedback_two_slow_shift_calls_then_assumed():
    f = _gf(REV)
    modes = []
    for _ in range(3):
        f.step((0.0, 0.0), 0.0, 0.0, 0.0, 0.0)
        modes.append(f.last_mode)
    assert modes == ["SHIFT", "SHIFT", "DRV-1"]
    assert f._gear_assumed == -1


def test_without_feedback_a_fast_call_restarts_the_slow_count():
    f = _gf(REV)
    seq = [0.0, 0.6, 0.0, 0.2, 0.0]          # forward (against reverse) at 0.6 m/s once
    modes = []
    for v in seq:
        f.step((0.0, 0.0), 0.0, v, 0.0, 0.0, v_lat=0.0)
        modes.append(f.last_mode)
    assert modes == ["SHIFT", "SHIFT", "SHIFT", "SHIFT", "DRV-1"]
    # v_lat counts toward the planar speed: 0.3 long + 0.4 lat = 0.5 >= 0.45
    g = _gf(REV)
    for _ in range(4):
        g.step((0.0, 0.0), 0.0, 0.3, 0.0, 0.0, v_lat=0.4)
        assert g.last_mode == "SHIFT"
    assert SHIFT_SLOW_SPEED == pytest.approx(0.45)


def test_guard_latches_against_the_gear_and_releases_after_two_slow_calls():
    f = _gf(REV)
    for _ in range(3):                                  # sync: SHIFT, SHIFT, DRV-1
        f.step((0.0, 0.0), 0.0, 0.0, 0.0, 0.0)
    out = []
    for v in (0.5, 0.35, 0.2, 0.1, 0.05):              # moving forward = against reverse
        c = f.step((0.0, 0.0), 0.0, v, 0.0, 0.0, v_lat=0.0)
        out.append((f.last_mode, tuple(c)))
    assert [m for m, _ in out] == ["GUARD", "GUARD", "GUARD", "DRV-1", "DRV-1"]
    assert out[0][1] == (0.0, 0.0, 1.0, -1)
    assert V_AGAINST == 0.3
    g = _gf(REV)                                        # feedback confirming: no GUARD
    for v in (0.5, 0.4):
        g.step((0.0, 0.0), 0.0, v, 0.0, 0.0, gear=-1)
        assert g.last_mode == "DRV-1"


def test_kick_guard_with_u_g_below_one():
    """C5: v_along in (-0.3, -0.05), the 1.x KICK condition true, u_g < 1: the gear
    command is u_g, not the KICK's 1 (the GUARD does not fire at -0.2 m/s)."""
    path = gold.densify([(0.0, 0.0, 0.0, 0.15), (20.0, 0.0, 0.0, 0.15)])
    f = _gf(path)
    table = gold.make_table()
    plan = f.plan((0.0, 0.0), 0.0, -0.2, 0.0, 0.0, gear=1)
    assert plan.mode == "DRV+1" and abs(plan.v_target) > 0.1
    assert -0.2 < 0.4 * abs(plan.v_target)                     # the KICK condition
    u = table.throttle_for(-0.2, plan.a_target, 0.0, 0.0)
    assert 0.02 < u < 1.0
    c = f.finish(plan, u, 0.0)
    assert c.throttle == u and c.brake == 0.0
    s = PathFollower(path, gold.make_table(), mpc_model="off")  # signed: KICK
    p2 = s.plan((0.0, 0.0), 0.0, -0.2, 0.0, 0.0)
    assert s.finish(p2, u, 0.0)[0] == 1.0


# --- (iii) set_path / resync --------------------------------------------------------

def test_set_path_at_speed_keeps_the_assumed_gear_and_resync_always_shifts():
    FWD8 = gold.densify([(0.0, 0.0, 0.0, 8.0), (60.0, 0.0, 0.0, 8.0)])
    f = _gf(FWD8)
    for k in range(5):
        c = f.step((1.0 + 0.2 * k, 0.0), 0.0, 8.0, 0.0, 0.0, v_lat=0.0)
        assert f.last_mode == "DRV+1" and c.brake == 0.0       # construction rule: in gear
    f._shift_ok, f._shift_n, f._guard, f._guard_slow = 1, 7, True, 1
    f.set_path(FWD8[5:], pos_xy=(2.0, 0.0))
    assert (f._shift_ok, f._shift_n, f._guard, f._guard_slow) == (0, 0, False, 0)
    assert f._gear_assumed == 1
    c = f.step((2.2, 0.0), 0.0, 8.0, 0.0, 0.0, v_lat=0.0)
    assert f.last_mode == "DRV+1" and c.brake == 0.0
    f.resync_gear()                       # C1: moving at 8 m/s forward, still SHIFT
    f.step((2.4, 0.0), 0.0, 8.0, 0.0, 0.0, v_lat=0.0)
    assert f.last_mode == "SHIFT"


def test_construction_rule_needs_motion_in_the_wanted_direction():
    f = _gf(FWD)
    f.step((0.0, 0.0), 0.0, 0.4, 0.0, 0.0, v_lat=0.0)
    assert f.last_mode == "SHIFT"
    g = _gf(REV)
    g.step((0.0, 0.0), 0.0, -0.6, 0.0, 0.0, v_lat=0.0)
    assert g.last_mode == "DRV-1"


# --- (iv) GOAL-MISSED history ----------------------------------------------------------

def test_shift_calls_do_not_feed_goal_missed():
    path = gold.densify([(0.0, 0.0, 0.0, 2.0), (10.0, 0.0, 0.0, 2.0)])
    f = _gf(path)
    for x in (0.0, 2.0, 4.0, 6.0, 7.0, 8.0):            # advance the projection
        f.step((x, 0.0), 0.0, 2.0, 0.0, 0.0, gear=1)
    f.step((12.0, 0.0), 0.0, 2.0, 0.0, 0.0, gear=1)      # past the goal, d = 2.0
    f.step((12.0, 0.0), 0.0, 2.0, 0.0, 0.0, gear=1)      # d = 2.0 again: no growth
    for x in (13.0, 13.5, 14.0):                         # receding, but SHIFT calls
        f.step((x, 0.0), 0.0, 0.0, 0.0, 0.0, gear=-1)
        assert f.last_mode == "SHIFT"
    f.step((14.5, 0.0), 0.0, 0.0, 0.0, 0.0, gear=1)
    assert f.last_mode != "GOAL-MISSED"                  # history [2.0, 2.0, 4.5]
    f.step((15.0, 0.0), 0.0, 0.0, 0.0, 0.0, gear=1)
    assert f.last_mode == "GOAL-MISSED"


# --- (v) defaults and GearCommand ------------------------------------------------------

def test_default_command_mode_and_gear_command_shape():
    assert PathFollower(FWD, gold.make_table(), mpc_model="off").command_mode == "signed"
    s = PathFollower(FWD, gold.make_table(), mpc_model="off")
    assert len(s.step((0.0, 0.0), 0.0, 0.0, 0.0, 0.0)) == 3
    assert s.plan((0.0, 0.0), 0.0, 0.0, 0.0, 0.0).gear is None          # R4
    c = GearCommand(0.3, 0.1, 0.0, -1)
    thr, st, br, g = c
    assert (thr, st, br, g) == (0.3, 0.1, 0.0, -1) and c.signed() == (-0.3, 0.1, 0.0)
    with pytest.raises(ValueError):
        thr, st, br = c
    with pytest.raises(TypeError):
        c[0]
    assert c != (0.3, 0.1, 0.0, -1)
    with pytest.raises(ValueError, match="command_mode"):
        PathFollower(FWD, gold.make_table(), mpc_model="off", command_mode="auto")
    with pytest.raises(ValueError, match="gear-mode argument"):
        PathFollower(FWD, gold.make_table(), mpc_model="off", brake_gain=4.0)
    with pytest.raises(ValueError, match="feedback"):
        _gf(FWD).step((0.0, 0.0), 0.0, 0.0, 0.0, 0.0, gear=0)


def test_mpc_default_is_gear_mode():
    pytest.importorskip("casadi")
    pytest.importorskip("do_mpc")
    import copy
    from genesis_vehicle.control.mpc import MPCVehicleParams
    from genesis_vehicle.tests.data import make_mpc_field_replay as rp
    p = MPCVehicleParams.from_dict(copy.deepcopy(rp.PARAMS_SDK))
    f = PathFollower(FWD, rp.make_table(), mpc_params=p)
    assert f.command_mode == "gear"
    # k_b derived from the record's decel_brake_by_speed (no brake_gain key)
    t = rp.make_table()
    dec = p.data["decel_brake_by_speed"]
    want = sum(t.accel_at(float(v), 0.0) - a for v, a in dec.items()) / len(dec)
    assert f._k_b == pytest.approx(want)


# --- (vi) FleetFollower -----------------------------------------------------------------

class _FleetPlant(gold.FakeCoupledPlant):
    n_members = 2

    def __init__(self):
        super().__init__()
        self.brake = 0.3
        self.predicts = []

    def solve(self, v, a, w, pitch_deg=0.0, roll_deg=0.0):
        return [-0.5, 0.4], [0.0, 0.0]

    def accel_at(self, v, throttle, pitch_deg=0.0, roll_deg=0.0):
        raise AssertionError("the fleet fills a_coast from one predict")

    def predict(self, throttle, steer):
        self.predicts.append((list(throttle), list(steer), self.brake))
        return [-0.2, -0.2], [0.0, 0.0]

    def set_applied_member(self, i, throttle, steer):
        pass


def test_fleet_one_mode_and_one_batched_coast_prediction():
    plant = _FleetPlant()
    a = _gf(FWD)
    b = PathFollower(FWD, gold.make_table(), mpc_model="off")
    with pytest.raises(ValueError, match="command_mode"):
        FleetFollower([a, b], plant)
    fl = FleetFollower([_gf(FWD), _gf(FWD)], plant)
    st = dict(pos_xy=(0.0, 0.0), yaw=0.0, v_long=3.0, pitch=0.0, roll=0.0, gear=1)
    cmds = fl.step([st, dict(st)])
    assert all(isinstance(c, GearCommand) for c in cmds)
    assert len(plant.predicts) == 1 and plant.predicts[0][2] == 0.0     # at brake 0
    assert plant.brake == 0.3                                          # restored
    assert cmds[0].throttle == 0.0 and cmds[0].brake > 0.0
    assert cmds[1].throttle == pytest.approx(0.4) and cmds[1].brake == 0.0


# --- (vii) construction errors and the MPC finish ----------------------------------------

def test_construction_errors():
    class NoAccel(gold.FakeCoupledPlant):
        pass

    class FloatGain(gold.FakeCoupledPlant):
        brake_gain = 5.0

        def accel_at(self, *a):
            return 0.0

    class NoGain(gold.FakeCoupledPlant):
        def accel_at(self, *a):
            return 0.0
    with pytest.raises(TypeError, match="accel_at"):
        PathFollower(FWD, NoAccel(), mpc_model="off", command_mode="gear")
    with pytest.raises(TypeError, match="brake_gain"):
        PathFollower(FWD, FloatGain(), mpc_model="off", command_mode="gear")
    with pytest.raises(ValueError, match="brake gain"):
        PathFollower(FWD, NoGain(), mpc_model="off", command_mode="gear")
    assert PathFollower(FWD, NoGain(), mpc_model="off", command_mode="gear",
                        brake_gain=3.0)._k_b == 3.0
    with pytest.raises(ValueError, match="brake_gain"):
        PathFollower(FWD, NoGain(), mpc_model="off", command_mode="gear", brake_gain=-1.0)


def test_mpc_gear_finish_brakes_instead_of_a_negative_throttle():
    pytest.importorskip("casadi")
    pytest.importorskip("do_mpc")
    import copy
    from genesis_vehicle.control.mpc import MPCVehicleParams
    from genesis_vehicle.tests._mpc_fakes import FakeSolver, ok
    from genesis_vehicle.tests.data import make_mpc_field_replay as rp
    p = MPCVehicleParams.from_dict(copy.deepcopy(rp.PARAMS_SDK))
    path = gold.densify([(0.0, 0.0, 0.0, 3.0), (40.0, 0.0, 0.0, 3.0)])
    f = PathFollower(path, rp.make_table(), mpc_params=p, brake_gain=K_B)
    f._law.solver = FakeSolver(default=ok(0.0, -2.0))
    c = f.step((0.0, 0.0), 0.0, 3.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=0.0, gear=1)
    assert f.regime == "R-MPC" and f.last_mode == "MPC"
    t = rp.make_table()
    u = t.throttle_for(3.0, -2.0, 0.0, 0.0)
    assert u < 0
    want = min(1.0, max(0.0, (t.accel_at(3.0, 0.0) - (-2.0)) / K_B))
    assert (c.throttle, c.gear) == (0.0, 1) and c.brake == pytest.approx(want) and c.brake > 0
    g = PathFollower(path, rp.make_table(), mpc_params=p, brake_gain=K_B)
    g._law.solver = FakeSolver(default=ok(0.0, -2.0))
    c = g.step((0.0, 0.0), 0.0, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=0.0, gear=1)
    assert g.last_mode == "MPC-HOLD" and tuple(c)[::2] == (0.0, 1.0)


# --- round 2: Amendment 1 call site, impl-review F3 / F4 --------------------------------

def test_braking_below_the_hold_speed_uses_the_faded_coast():
    """At 1 m/s (below the golden table's first speed 1.5) a braking PD step commands
    brake_for(a_t, coast per R1.1); at 4 m/s the rev-2 value (raw accel_at)."""
    table = gold.make_table()
    for v, below in ((1.0, True), (4.0, False)):
        path = gold.densify([(0.0, 0.0, 0.0, 0.2), (20.0, 0.0, 0.0, 0.2)])
        f = _gf(path)
        plan = f.plan((0.0, 0.0), 0.0, v, 0.0, 0.0, gear=1)
        u = table.throttle_for(v, plan.a_target, 0.0, 0.0)
        assert u < 0
        c = f.finish(plan, u, 0.0)
        coast = _oracle_coast(table, v, 0.0, 0.0, 1)
        want = min(1.0, max(0.0, (coast - plan.a_target) / K_B))
        assert c.brake == want and c.throttle == 0.0
        raw = table._throttle_lu.forward(v, 0.0, 0.0, 0.0)
        assert (coast != raw) == below


def test_guard_calls_do_not_feed_goal_missed():
    """F3 (m-N3): a GUARD call takes its goal-distance entry back out. Final reverse
    block, the vehicle past the goal and moving away from it AGAINST the reverse
    gear (+0.5 m/s, no feedback): GUARD, not GOAL-MISSED."""
    path = gold.densify([(0.0, 0.0, 0.0, -1.5), (-10.0, 0.0, 0.0, -1.5)])
    f = _gf(path)
    for x in (0.0, 0.0, -2.0, -4.0, -6.0, -7.0, -8.0):      # sync (2 SHIFT) + advance
        f.step((x, 0.0), 0.0, 0.0, 0.0, 0.0, v_lat=0.0)
    assert f.last_mode == "DRV-1"
    f.step((-12.0, 0.0), 0.0, 0.0, 0.0, 0.0, v_lat=0.0)     # past the goal, d = 2.0
    f.step((-12.0, 0.0), 0.0, 0.0, 0.0, 0.0, v_lat=0.0)     # no growth
    for x in (-12.5, -13.0, -13.5, -14.0):
        f.step((x, 0.0), 0.0, 0.5, 0.0, 0.0, v_lat=0.0)
        assert f.last_mode == "GUARD", (x, f.last_mode)


def test_resync_clears_the_guard_latch():
    f = _gf(REV)
    for _ in range(3):
        f.step((0.0, 0.0), 0.0, 0.0, 0.0, 0.0)
    f.step((0.0, 0.0), 0.0, 0.5, 0.0, 0.0, v_lat=0.0)
    f.step((0.0, 0.0), 0.0, 0.4, 0.0, 0.0, v_lat=0.0)
    assert f.last_mode == "GUARD" and f._guard and f._guard_slow == 1
    f.resync_gear()
    assert (f._guard, f._guard_slow, f._shift_ok, f._shift_n, f._gear_assumed) == (False, 0, 0, 0, None)


def test_mpc_gear_finish_keeps_the_brake_at_rest_on_a_negative_u():
    """The R-MPC gear finish brakes with max(brake_for, the field finish's brake at
    rest): |thr| < 0.02 at |v| < 0.1 with a positive a_ff (no MPC-HOLD) -> brake 1."""
    from genesis_vehicle.control.path_follower import FollowerPlan
    f = _gf(FWD)
    plan = FollowerPlan(mode="MPC", a_target=0.05, v_long=0.05, regime="R-MPC",
                        a_ff=0.05, gear=1)
    c = f.finish(plan, -0.01, 0.0)
    assert f.last_mode == "MPC" and c.throttle == 0.0 and c.brake == 1.0
