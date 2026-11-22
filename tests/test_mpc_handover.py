"""PathFollower heading gate (R-ALIGN) and no-progress watchdog (R-STALL), v2.0.1.

Fake solver (``tests/_mpc_fakes.py``), synthetic params, synthetic table; the
states are fed directly, no plant. Covers the state machine of plan_r210
section 3.3-3.4:

* R-ALIGN: hysteresis (enter above ``theta_on``, hold while >= ``theta_off``,
  release below it); fires mid-block at a sharp vertex, not only on block
  entry; on release the reference is re-timed (``start_block`` with
  ``t0 == t_now - t_ref``) and the next solve gets ``u_prev = (d0, a0)``; on a
  forward block after a cusp (block 1) the re-seed uses THAT block.
* R-STALL: fires at exactly ``t - t_anchor >= 2.0`` of follower time for
  ``t`` spacing 0.025 and 0.05; fires under +-0.1 m jitter; a 0.3 m move moves
  the anchor; does not fire within ``arrival_goal`` of the last waypoint;
  keeps the anchor across R-MPC / R-FAIL alternation; fires from ``plan()``
  alone (no ``finish``); after 1.0 m of travel re-seeds and returns to R-MPC.
* release events: CMD, R-REV, block change, ``set_path`` (also during
  R-ALIGN / R-STALL) clear the hold and the anchor and keep the counts.
* ``None`` disables each feature independently; ``mpc_model="off"`` rejects
  both kwargs (at ``None`` and at their defaults); range / type checks.

Mutants this file must fail on (plan_r210 section 6): a step-count timer
instead of follower time (the 0.05 case); a brake-continuity predicate
recorded in ``finish`` (the ``plan()``-only case); the anchor reset on an
R-FAIL step; the gate evaluated on block entry only (the mid-block vertex);
the re-seed pinned to block 0 (the block-1 case); ``set_path`` keeping the
hold. Since the 2.0.1 amendment (plan_r210 A5 / A5R1.M), also: the F5 progress
reset removed (P0), its distance lowered to 2.9 m (P1) or to any progress
(P1z), ground distance instead of path progress (P2), comparison with the last
run start (P3) or with the previous entry's raw arc length (P4) instead of the
furthest earlier entry, the progress mark kept across a block change (P5) or
``set_path`` (P6 — behaviourally equivalent: the first plan() after set_path
takes the block-change branch; killed only by the state inspection in
``test_set_path_resets_budgets_and_terminal_state``), a run start setting the
count to 1 (P7) or applied after the increment (P8); ``set_path`` keeping the
budgets (R1); a DONE step not clearing the terminal state (R2); the deleted
F4 no-room refusal restored on either entry path (F4d / F4x).
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

from genesis_vehicle.control import PathFollower  # noqa: E402
from genesis_vehicle.control.mpc import MPCVehicleParams  # noqa: E402
from genesis_vehicle.control.mpc.law import rig_reference  # noqa: E402
from genesis_vehicle.tests._mpc_fakes import FakeSolver, exc, ok  # noqa: E402
from genesis_vehicle.tests.data import make_mpc_field_replay as rp  # noqa: E402
from genesis_vehicle.tests.test_path_follower import _densify  # noqa: E402
from genesis_vehicle.tests.test_sweep_table import make_table  # noqa: E402

LONG = _densify([(0.0, 0.0, 0.0, 2.0), (40.0, 0.0, 0.0, 2.0)])
# Second leg 40 m: lengthened in the 2.0.1 draft for the since-deleted F4
# no-room test; kept.
_C150 = (20.0 + 40.0 * math.cos(math.radians(150)), 40.0 * math.sin(math.radians(150)))
V150 = _densify([(0.0, 0.0, 0.0, 2.0), (20.0, 0.0, 0.0, 2.0), (*_C150, 0.0, 2.0)])
# reverse block (chassis yaw 0, travelling -x), cusp at (-5, 0), then a forward
# block heading atan2(5, -10) = 2.678 rad: the gate fires on entering block 1.
# Block 1 is 33.5 m long: lengthened in the 2.0.1 draft for the since-deleted
# F4 no-room test; kept.
REV_FWD = (_densify([(0.0, 0.0, 0.0, -1.5), (-5.0, 0.0, 0.0, -1.5)])
           + _densify([(-5.0, 0.0, 0.0, 1.5), (-35.0, 15.0, 0.0, 1.5)]))
H1 = math.atan2(5.0, -10.0)


def _params():
    return MPCVehicleParams.from_dict(copy.deepcopy(rp.PARAMS_SDK))


def _f(path, results=None, policy="pd", **kw):
    f = PathFollower(path, make_table(), mpc_params=_params(), on_solve_failure=policy,
                     command_mode="signed", **kw)
    fake = FakeSolver(results, default=ok(0.0, 0.5))
    f._law.solver = fake
    return f, fake


def _step(f, pos, yaw, t, v=0.0, r=0.0):
    return f.step(pos, yaw, v, 0.0, 0.0, yaw_rate=r, v_lat=0.0, t=t)


def _record_starts(f):
    starts = []
    real = f._law.start_block

    def rec(wps, t0=0.0):
        starts.append((list(wps), t0))
        return real(wps, t0)
    f._law.start_block = rec
    return starts


def _t_ref(f, block):
    """Reference time of the current projection, from the block's waypoints."""
    bs = f.blocks[block][0]
    tt = rig_reference(f._block_waypoints(block), f._law.rig, f._law.v_floor, 0.0)[0]
    k = f.current_idx - bs
    k1 = min(k + 1, len(tt) - 1)
    return float(tt[k] + f.projection_t * (tt[k1] - tt[k]))


# --- R-ALIGN ------------------------------------------------------------------------------

def test_align_hysteresis_three_zones_and_pd_command():
    f, fake = _f(LONG)
    g = PathFollower(LONG, make_table(), mpc_model="off")
    pos = (5.0, 0.0)
    out = _step(f, pos, -2.0, 0.0)                      # |e| = 2.0 > pi/2: enter
    assert f.last_mode == "PD-ALIGN" and f.regime == "R-ALIGN" and fake.calls == []
    assert out == g.step(pos, -2.0, 0.0, 0.0, 0.0)      # the PD law, KICK included
    assert out[0] == 1.0
    _step(f, pos, -1.0, 0.025)                          # pi/4 <= 1.0: hold
    assert f.last_mode == "PD-ALIGN" and fake.calls == []
    _step(f, pos, -0.5, 0.05)                           # 0.5 < pi/4: release, solve now
    assert f.regime == "R-MPC" and f.last_mode == "MPC" and len(fake.calls) == 1
    _step(f, pos, -1.0, 0.075)                          # 1.0 is not > pi/2: no re-entry
    assert f.regime == "R-MPC"
    assert f.mpc_stats["handover"] == {"align": 1, "stall": 0}


def test_align_fresh_follower_between_thresholds_is_not_gated():
    f, fake = _f(LONG)
    _step(f, (5.0, 0.0), -1.0, 0.0)
    assert f.regime == "R-MPC" and len(fake.calls) == 1


def _drive_to_vertex(f):
    for k, x in enumerate(np.arange(0.0, 20.0, 0.5)):
        _step(f, (float(x), 0.0), 0.0, k * 0.025, v=2.0)
        assert f.regime == "R-MPC"
    return 40 * 0.025


def test_align_fires_mid_block_at_a_sharp_vertex():
    f, fake = _f(V150)
    t = _drive_to_vertex(f)
    n = len(fake.calls)
    c, s = math.cos(math.radians(150)), math.sin(math.radians(150))
    _step(f, (20.0 + c, s), 0.0, t, v=2.0)
    assert f._mpc_block == 0                            # mid-block, reference already built
    assert f.regime == "R-ALIGN" and f.last_mode == "PD-ALIGN" and len(fake.calls) == n
    assert f.mpc_stats["handover"]["align"] == 1


def test_align_release_reseeds_reference_and_u_prev():
    f, fake = _f(V150)
    t = _drive_to_vertex(f)
    c, s = math.cos(math.radians(150)), math.sin(math.radians(150))
    _step(f, (20.0 + c, s), 0.0, t, v=2.0)
    assert f.regime == "R-ALIGN"
    starts = _record_starts(f)
    t2, v, r = t + 0.025, 1.0, 0.1
    _step(f, (20.0 + 1.2 * c, 1.2 * s), math.radians(150) - 0.3, t2, v=v, r=r)
    assert f.regime == "R-MPC"
    assert len(starts) == 2 and starts[0][1] == 0.0
    assert starts[1][0] == f._block_waypoints(0)
    assert starts[1][1] == pytest.approx(t2 - _t_ref(f, 0), abs=1e-12)
    d0 = math.atan(r * f._law.L / v)
    assert fake.u_prev_sets[-1] == pytest.approx((d0, 0.0), abs=1e-15)
    assert tuple(fake.calls[-1][2]) == pytest.approx((d0, 0.0), abs=1e-15)


def _enter_block1(f):
    _step(f, (-4.5, 0.0), 0.0, 0.0)                     # near the cusp: BRAKE_TRANS
    assert f.last_mode == "BRAKE_TRANS"
    _step(f, (-4.5, 0.0), 0.0, 0.025)                   # stopped: block 1, gate fires
    assert f.block_i == 1


def test_align_on_block1_entry_then_reseed_uses_block1():
    f, fake = _f(REV_FWD)
    _enter_block1(f)
    assert f.regime == "R-ALIGN" and f._mpc_block is None and f._law.ref is None
    starts = _record_starts(f)
    _step(f, (-4.6, 0.05), H1 - 0.2, 0.05)
    assert f.regime == "R-MPC" and f._mpc_block == 1
    assert len(starts) == 2
    assert starts[1][0] == f._block_waypoints(1) and starts[1][0] != f._block_waypoints(0)
    assert starts[1][1] == pytest.approx(0.05 - _t_ref(f, 1), abs=1e-12)


# --- R-STALL ------------------------------------------------------------------------------

def _first(f, states, mode="PD-STALL", plan_only=False):
    for (pos, t) in states:
        if plan_only:
            p = f.plan(pos, 0.0, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=t)
            if p.mode == mode:
                return t
        else:
            _step(f, pos, 0.0, t)
            if f.last_mode == mode:
                return t
    return None


@pytest.mark.parametrize("dt", [0.025, 0.05])
def test_stall_fires_at_two_seconds_of_follower_time(dt):
    f, _ = _f(LONG)
    t_fire = _first(f, [((5.0, 0.0), k * dt) for k in range(int(5.0 / dt))])
    assert t_fire == pytest.approx(2.0, abs=1e-12)
    assert f.regime == "R-STALL" and f.mpc_stats["handover"] == {"align": 0, "stall": 1}


def test_stall_not_yet_at_1_975():
    f, _ = _f(LONG)
    assert _first(f, [((5.0, 0.0), k * 0.025) for k in range(80)]) is None   # up to 1.975


def test_stall_fires_under_jitter():
    f, _ = _f(LONG)
    st = [((5.0 + 0.1 * (-1) ** k, 0.0), k * 0.025) for k in range(200)]
    assert _first(f, st) == pytest.approx(2.0, abs=1e-12)


def test_stall_anchor_moves_on_a_0_3_m_step():
    f, _ = _f(LONG)
    st = [((5.0 if k < 60 else 5.3, 0.0), k * 0.025) for k in range(200)]
    assert _first(f, st) == pytest.approx(3.5, abs=1e-12)


def test_stall_not_within_arrival_goal_of_the_last_waypoint():
    path = (_densify([(0.0, 0.0, 0.0, 2.0), (10.0, 0.0, 0.0, 2.0)])
            + _densify([(10.0, 0.0, 0.0, -1.5), (0.5, 0.0, 0.0, -1.5)]))
    f, _ = _f(path)                                     # stands 0.3 m from the goal, block 0
    assert _first(f, [((0.2, 0.0), k * 0.025) for k in range(200)]) is None
    assert f.regime == "R-MPC"
    f2, _ = _f(path)                                    # control: 2.5 m from the goal
    assert _first(f2, [((3.0, 0.0), k * 0.025) for k in range(200)]) == pytest.approx(2.0)


def test_stall_anchor_kept_across_rmpc_rfail_alternation():
    f, fake = _f(LONG, results=[exc(), ok(0.0, 0.5)] * 40)
    regimes = set()
    t_fire = None
    for k in range(200):
        _step(f, (5.0, 0.0), 0.0, k * 0.025)
        regimes.add(f.regime)
        if f.last_mode == "PD-STALL":
            t_fire = k * 0.025
            break
    assert {"R-MPC", "R-FAIL"} <= regimes
    assert t_fire == pytest.approx(2.0, abs=1e-12)


def test_stall_fires_from_plan_alone():
    f, _ = _f(LONG)
    t_fire = _first(f, [((5.0, 0.0), k * 0.025) for k in range(200)], plan_only=True)
    assert t_fire == pytest.approx(2.0, abs=1e-12) and f.regime == "R-STALL"


def test_stall_exit_after_one_metre_reseeds_and_resumes_mpc():
    f, fake = _f(LONG)
    assert _first(f, [((5.0, 0.0), k * 0.025) for k in range(100)]) == pytest.approx(2.0)
    n = len(fake.calls)
    _step(f, (5.5, 0.0), 0.0, 2.025, v=0.5)
    assert f.last_mode == "PD-STALL" and len(fake.calls) == n
    starts = _record_starts(f)
    _step(f, (6.0, 0.0), 0.0, 2.05, v=0.5)
    assert f.regime == "R-MPC" and len(fake.calls) == n + 1
    assert len(starts) == 2 and starts[1][1] == pytest.approx(2.05 - _t_ref(f, 0), abs=1e-12)
    assert f._anchor == (2.05, 6.0, 0.0)
    assert f.mpc_stats["handover"] == {"align": 0, "stall": 1}


def test_stall_exit_into_align_when_misaligned():
    f, _ = _f(LONG)
    _first(f, [((5.0, 0.0), k * 0.025) for k in range(100)])
    _step(f, (6.0, 0.0), -2.0, 2.025)
    assert f.regime == "R-ALIGN" and f.mpc_stats["handover"] == {"align": 1, "stall": 1}
    assert f._anchor is None


# --- release events -------------------------------------------------------------------------

def _released(f):
    return f._ho_regime is None and f._anchor is None and f._stall_entry is None


def test_cmd_releases_hold_and_keeps_counts():
    f, _ = _f(LONG)
    _first(f, [((5.0, 0.0), k * 0.025) for k in range(100)])
    assert f.regime == "R-STALL"
    _step(f, (39.5, 0.0), 0.0, 2.1)
    assert f.last_mode == "DONE" and _released(f)
    assert f.mpc_stats["handover"] == {"align": 0, "stall": 1}


def test_rrev_releases_hold():
    f, _ = _f(REV_FWD)
    f._ho_regime, f._ho_block, f._stall_entry, f._anchor = "R-STALL", 0, (-1.0, 0.0), (0.0, -1.0, 0.0)
    _step(f, (-1.0, 0.0), 0.0, 0.0)
    assert f.regime == "R-REV" and _released(f)


def test_block_change_releases_hold():
    f, _ = _f(REV_FWD)
    _enter_block1(f)
    _step(f, (-4.6, 0.05), H1 - 0.2, 0.05)
    assert f.regime == "R-MPC"
    f._ho_regime, f._ho_block, f._stall_entry = "R-STALL", 0, (-4.6, 0.05)   # stale block
    _step(f, (-4.6, 0.05), H1 - 0.2, 0.075)
    assert f.regime == "R-MPC"


def test_set_path_releases_align_and_reevaluates():
    f, _ = _f(LONG)
    _step(f, (5.0, 0.0), -2.0, 0.0)
    assert f.regime == "R-ALIGN"
    c = _densify([(5.0, 0.0, 0.0, 2.0), (5.0 + 20 * math.cos(-1.0), 20 * math.sin(-1.0), 0.0, 2.0)])
    f.set_path(c, pos_xy=(5.0, 0.0), t=0.025)
    assert _released(f)
    _step(f, (5.0, 0.0), -2.0, 0.025)                   # e = 1.0 on the new path: no hold
    assert f.regime == "R-MPC"
    f.set_path(LONG, pos_xy=(5.0, 0.0), t=0.05)
    _step(f, (5.0, 0.0), -2.0, 0.05)                    # still misaligned: R-ALIGN at once
    assert f.regime == "R-ALIGN"
    assert f.mpc_stats["handover"] == {"align": 2, "stall": 0}


def test_set_path_releases_stall():
    f, _ = _f(LONG)
    _first(f, [((5.0, 0.0), k * 0.025) for k in range(100)])
    assert f.regime == "R-STALL"
    f.set_path(LONG, pos_xy=(5.0, 0.0), t=2.025)
    assert _released(f)
    _step(f, (5.0, 0.0), 0.0, 2.025)
    assert f.regime == "R-MPC" and f.mpc_stats["handover"]["stall"] == 1


# --- None, construction ---------------------------------------------------------------------

def test_gate_none_disables_only_the_gate():
    f, fake = _f(LONG, mpc_align_gate=None)
    _step(f, (5.0, 0.0), -2.0, 0.0)
    assert f.regime == "R-MPC" and len(fake.calls) == 1
    assert _first(f, [((5.0, 0.0), k * 0.025) for k in range(1, 200)]) == pytest.approx(2.0)


def test_stall_none_disables_only_the_watchdog():
    f, _ = _f(LONG, mpc_stall_timeout=None)
    assert _first(f, [((5.0, 0.0), k * 0.025) for k in range(400)]) is None
    _step(f, (5.0, 0.0), -2.0, 10.0)
    assert f.regime == "R-ALIGN"


def test_both_none_never_hand_over():
    f, _ = _f(LONG, mpc_align_gate=None, mpc_stall_timeout=None)
    for k in range(200):
        _step(f, (5.0, 0.0), -2.0 if k % 2 else 0.0, k * 0.025)
        assert f.regime == "R-MPC"
    assert f.mpc_stats["handover"] == {"align": 0, "stall": 0}


@pytest.mark.parametrize("kw", [dict(mpc_align_gate=None), dict(mpc_align_gate=(math.pi / 2, math.pi / 4)),
                                dict(mpc_stall_timeout=None), dict(mpc_stall_timeout=2.0)])
def test_off_rejects_the_new_kwargs_even_at_none_and_default(kw):
    with pytest.raises(ValueError, match=list(kw)[0]):
        PathFollower(LONG, make_table(), mpc_model="off", **kw)


@pytest.mark.parametrize("gate", [(math.pi / 4, math.pi / 2), (1.0, 0.0), (4.0, 1.0), (1.0, -0.5),
                                  (float("nan"), 0.5), (1.0, float("nan")), (float("inf"), 0.5),
                                  (True, 0.5), (1.0,), (1.5, 0.5, 0.1), "ab", 1.0, (1.0, "x")])
def test_bad_gate_raises(gate):
    with pytest.raises(ValueError, match="mpc_align_gate"):
        PathFollower(LONG, make_table(), mpc_params=_params(), mpc_align_gate=gate)


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan"), True, "2", (2.0,)])
def test_bad_stall_timeout_raises(timeout):
    with pytest.raises(ValueError, match="mpc_stall_timeout"):
        PathFollower(LONG, make_table(), mpc_params=_params(), mpc_stall_timeout=timeout)


def test_good_values_are_normalised_to_float():
    f = PathFollower(LONG, make_table(), mpc_params=_params(),
                     mpc_align_gate=[np.float64(math.pi), 1], mpc_stall_timeout=np.float32(3))
    assert f._align_gate == (math.pi, 1.0) and type(f._align_gate[1]) is float
    assert f._stall_timeout == 3.0 and type(f._stall_timeout) is float


def test_u1_reported_before_a_bad_gate():
    with pytest.raises(ValueError, match="mpc_params="):
        PathFollower(LONG, make_table(), mpc_align_gate=(0.1, 0.2))


def test_handover_is_merged_at_the_follower_level_and_copied():
    f, _ = _f(LONG)
    _step(f, (5.0, 0.0), -2.0, 0.0)
    s = f.mpc_stats
    assert s["handover"] == {"align": 1, "stall": 0}
    assert "handover" not in f._law.mpc_stats(0.0)
    s["handover"]["align"] = 99
    assert f.mpc_stats["handover"]["align"] == 1


# --- v2.0.1 delta: budgets, t jump, terminal modes, F5 runs (plan_r210 3B, A5, A5R1) ----------
#
# Mutants these fail on: the budgets per entry instead of per block (F3 / F8);
# ``_reseed_block`` setting ``_t_next = t_now`` (A-C11); the F8 block reset tied
# to ``_handover_step`` (B-C8); the TypeError snapshot missing GP-AWAY2's
# history (B-C9); the deleted F4 restored on the direct entry (F4d) or on the
# entry out of R-STALL (F4x); the F5 / set_path / DONE mutants P0-P8, R1, R2 of
# the module docstring.

LONG100 = _densify([(0.0, 0.0, 0.0, 2.0), (100.0, 0.0, 0.0, 2.0)])


def test_r_min_and_budget_come_from_the_solver_steer_limit():
    f, _ = _f(LONG)
    assert f._r_min == pytest.approx(2.7 / math.tan(0.5 * 0.7), rel=1e-12)
    assert f._pd_budget == pytest.approx(math.pi * f._r_min, rel=1e-12)


@pytest.mark.parametrize("over", [dict(max_steer_rad=4.0), dict(max_steer_rad=0.0),
                                  dict(wheelbase=0.0), dict(wheelbase=float("nan"))])
def test_bad_steer_limit_or_wheelbase_raises_at_construction(over):
    d = copy.deepcopy(rp.PARAMS_SDK)
    d.update(over)
    with pytest.raises(ValueError, match="steering limit|wheelbase"):
        PathFollower(LONG, make_table(), mpc_params=MPCVehicleParams.from_dict(d))


def test_zero_steer_cap_raises_with_mpc_but_not_off():
    with pytest.raises(ValueError, match="steering limit"):
        PathFollower(LONG, make_table(), mpc_params=_params(), steer_cap=0.0)
    PathFollower(LONG, make_table(), mpc_model="off", steer_cap=0.0)


def test_terminal_modes_constant_and_stop_reason_default():
    assert PathFollower.TERMINAL_MODES == ("DONE", "GOAL-MISSED", "MPC-ABORT", "GEAR-ABORT")
    f, _ = _f(LONG)
    assert f.stop_reason is None


def test_f3_align_budget_is_cumulative_per_block_across_episodes():
    f, fake = _f(LONG100)
    budget = f._pd_budget                                   # 23.24 m
    x, t = 2.0, 0.0
    _step(f, (x, 0.0), -2.0, t)
    assert f.regime == "R-ALIGN"
    for _ in range(15):                                      # 15 m in R-ALIGN
        x += 1.0; t += 0.025
        _step(f, (x, 0.0), -2.0, t)
    assert f.regime == "R-ALIGN" and f._d_align == pytest.approx(15.0)
    x += 1.0; t += 0.025
    _step(f, (x, 0.0), -0.5, t)                              # release (+1 m charged)
    assert f.regime == "R-MPC"
    for _ in range(3):                                       # MPC travel is not charged
        x += 1.0; t += 0.025
        _step(f, (x, 0.0), 0.0, t)
    assert f._d_align == pytest.approx(16.0)
    x += 1.0; t += 0.025
    _step(f, (x, 0.0), -2.0, t)                              # second entry: allowed
    assert f.regime == "R-ALIGN" and f.mpc_stats["handover"]["align"] == 2
    n = 0
    while f.last_mode != "MPC-ABORT":
        x += 1.0; t += 0.025; n += 1
        out = _step(f, (x, 0.0), -2.0, t)
        assert n < 20
    assert out == (0.0, 0.0, 1.0) and f.stop_reason == "align_budget"
    assert f._d_align > budget and f._d_align - 1.0 <= budget
    _step(f, (x, 0.0), 0.0, t + 0.025)                       # sticky
    assert f.last_mode == "MPC-ABORT" and f.regime == "CMD"


def test_f3_budget_carries_into_an_entry_out_of_r_stall():
    f, _ = _f(LONG100)
    x, t = 2.0, 0.0
    _step(f, (x, 0.0), -2.0, t)
    for _ in range(20):                                      # 20 m in R-ALIGN
        x += 1.0; t += 0.025
        _step(f, (x, 0.0), -2.0, t)
    x += 1.0; t += 0.025
    _step(f, (x, 0.0), 0.0, t)                               # release (21 m charged)
    t0 = t
    while f.regime != "R-STALL":                             # stand: the watchdog fires
        t += 0.025
        _step(f, (x, 0.0), 0.0, t)
        assert t - t0 < 3.0
    t += 0.025
    _step(f, (x + 1.0, 0.0), -2.0, t)                        # 1 m exit into R-ALIGN
    assert f.regime == "R-ALIGN" and f._d_align == pytest.approx(21.0)
    for k in range(4):
        t += 0.025
        _step(f, (x + 2.0 + k, 0.0), -2.0, t)
    assert f.stop_reason == "align_budget"


def test_short_final_block_misaligned_start_hands_over():
    """T2: F4 (no-room refusal) is deleted — a 6 m final block, car facing back,
    hands over to R-ALIGN."""
    short = _densify([(0.0, 0.0, 0.0, 2.0), (6.0, 0.0, 0.0, 2.0)])
    f, fake = _f(short)
    _step(f, (0.0, 0.0), math.pi, 0.0)
    assert f.regime == "R-ALIGN" and f.last_mode == "PD-ALIGN"
    assert f.mpc_stats["handover"]["align"] == 1 and f.stop_reason is None
    assert fake.calls == []


def test_entry_out_of_r_stall_on_a_short_final_block_hands_over():
    """T2b: the R-STALL exit into a misaligned pose on a 12 m final block
    (about 9 m left) enters R-ALIGN, not an abort."""
    path = _densify([(0.0, 0.0, 0.0, 2.0), (12.0, 0.0, 0.0, 2.0)])
    f, _ = _f(path)
    t_fire = _first(f, [((2.0, 0.0), k * 0.025) for k in range(100)])
    assert t_fire == pytest.approx(2.0)
    _step(f, (3.0, 0.0), -2.0, t_fire + 0.025)
    assert f.regime == "R-ALIGN" and f.stop_reason is None
    assert f.mpc_stats["handover"] == {"align": 1, "stall": 1}


def test_f5_fourth_stall_aborts():
    f, _ = _f(LONG100)
    x, t = 5.0, 0.0
    for n in range(1, 4):
        t0 = t
        while f.regime != "R-STALL":
            _step(f, (x, 0.0), 0.0, t)
            t += 0.025
            assert t - t0 < 3.0
        x += 1.0
        _step(f, (x, 0.0), 0.0, t)                           # exit
        t += 0.025
        assert f.regime == "R-MPC" and f.mpc_stats["handover"]["stall"] == n
    t0 = t
    while f.last_mode != "MPC-ABORT":
        _step(f, (x, 0.0), 0.0, t)
        t += 0.025
        assert t - t0 < 3.0
    assert f.stop_reason == "stall_budget" and f.mpc_stats["handover"]["stall"] == 3


def test_f6_forward_t_jump_reanchors():
    f, _ = _f(LONG)
    for k in range(41):                                      # t 0 .. 1.0 standing
        _step(f, (5.0, 0.0), 0.0, k * 0.025)
    _step(f, (5.0, 0.0), 0.0, 3.0)                           # +2.0 s jump > timeout / 2
    assert f.regime == "R-MPC" and f._anchor[0] == 3.0
    assert _first(f, [((5.0, 0.0), 3.0 + k * 0.025) for k in range(1, 100)]) == pytest.approx(5.0)


def test_f8_fail_budget_is_cumulative_across_r_fail_episodes():
    f, fake = _f(LONG100, results=[exc(), ok(0.0, 0.5)] * 40)
    x, t, n_fail_steps = 1.0, 0.0, 0
    while f.last_mode != "MPC-ABORT":
        _step(f, (x, 0.0), 0.0, t)
        n_fail_steps += f.regime == "R-FAIL"
        x += 1.0; t += 0.025
        assert t < 10.0
    assert f.stop_reason == "fail_budget"
    assert f._d_fail > f._pd_budget and n_fail_steps > 8          # several episodes
    assert f._d_fail - 1.0 <= f._pd_budget                        # overshoot <= one call's 1 m


def test_f8_block_reset_does_not_need_the_gate_or_watchdog():
    """B-C8: None, None, "pd", R-FAIL travel 16 m in each of two forward blocks
    (32 m > budget 23.2 m in total) — no abort: D_fail restarts per block."""
    path = (_densify([(0.0, 0.0, 0.0, 2.0), (20.0, 0.0, 0.0, 2.0)])
            + _densify([(20.0, 0.0, 0.0, -1.5), (15.0, 0.0, 0.0, -1.5)])
            + _densify([(15.0, 0.0, 0.0, 2.0), (45.0, 0.0, 0.0, 2.0)]))
    f, _ = _f(path, results=[exc()] * 400, mpc_align_gate=None, mpc_stall_timeout=None)
    t = 0.0

    def go(pos, yaw=0.0, v=1.0):
        nonlocal t
        _step(f, pos, yaw, t, v=v)
        t += 0.025
        return f.regime
    for x in range(17):
        assert go((float(x), 0.0)) == "R-FAIL"
    assert f._d_fail == pytest.approx(16.0)
    go((19.6, 0.0))                                          # BRAKE_TRANS
    go((19.6, 0.0), v=0.0)                                   # block 1 (reverse)
    assert f.block_i == 1
    for x in (19.0, 18.0, 17.0, 16.5):
        go((x, 0.0), v=-1.0)
    go((15.6, 0.0), v=-1.0)
    go((15.6, 0.0), v=0.0)
    assert f.block_i == 2
    for x in range(16, 33):
        assert go((float(x), 0.0)) == "R-FAIL", f.stop_reason
    # block 2 started at the transition call at x = 15.6 (an R-FAIL step), so
    # 0.4 + 16 m is charged to block 2 alone
    assert f.stop_reason is None and f._d_fail == pytest.approx(16.4)


def test_reseed_keeps_the_internal_clock():
    """A-C11: with t=None the call after a release sees t = release + dt_ctrl."""
    f, _ = _f(LONG)
    f.step((5.0, 0.0), -2.0, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0)
    f.step((5.0, 0.0), -2.0, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0)
    assert f.regime == "R-ALIGN"
    f.step((5.0, 0.0), 0.0, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0)       # release
    assert f.regime == "R-MPC"
    t_rel = f._t_now
    f.step((5.0, 0.0), 0.0, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0)
    assert f._t_now == pytest.approx(t_rel + f.dt_ctrl, abs=1e-12)


def test_typeerror_restores_the_goal_passed_history():
    """B-C9: a raised call leaves GP-AWAY2's history as it was."""
    f, _ = _f(LONG)
    for k in range(5):
        _step(f, (0.5 * k, 0.0), 0.0, k * 0.025, v=1.0)
    before = list(f._gp_hist)
    with pytest.raises(TypeError):
        f.plan((3.0, 0.0), 0.0, 1.0, 0.0, 0.0, t=0.2)
    assert f._gp_hist == before


def _align_budget_abort(f):
    """Drive LONG100 in R-ALIGN (yaw -2.0) 1 m per call until align_budget."""
    x, t, out = 2.0, 0.0, None
    while f.last_mode != "MPC-ABORT":
        out = _step(f, (x, 0.0), -2.0, t)
        x += 1.0; t += 0.025
        assert x < 40.0
    return x, t, out


def test_abort_is_sticky_until_set_path_and_done_takes_precedence():
    """T3: an align_budget abort stays set while aligned; a DONE step inside
    arrival_goal clears stop_reason; after it an aligned step outside the
    circle is not terminal; set_path clears an abort."""
    f, _ = _f(LONG100)
    x, t, _ = _align_budget_abort(f)
    assert f.stop_reason == "align_budget"
    _step(f, (x, 0.0), 0.0, t)                               # aligned now: still aborted
    assert f.last_mode == "MPC-ABORT"
    _step(f, (99.0, 0.0), 0.0, t + 0.025)                    # inside arrival_goal: DONE
    assert f.last_mode == "DONE" and f.stop_reason is None
    _step(f, (97.0, 0.0), 0.0, t + 0.05)                     # outside, aligned: drives again
    assert f.last_mode not in PathFollower.TERMINAL_MODES
    f2, _ = _f(LONG100)
    x, t, _ = _align_budget_abort(f2)
    f2.set_path(LONG100, pos_xy=(x, 0.0), t=t)
    assert f2.stop_reason is None
    _step(f2, (x, 0.0), 0.0, t)
    assert f2.last_mode == "MPC"


def test_abort_goes_through_finish_and_reports_to_the_plant():
    """T3b: the abort step returns (0, 0, 1) through finish() and reports (0, 0)."""
    f, _ = _f(LONG100)
    seen = []
    f._report_applied = lambda thr, st: seen.append((thr, st))
    _, _, out = _align_budget_abort(f)
    assert out == (0.0, 0.0, 1.0) and seen[-1] == (0.0, 0.0)


# --- F5: runs of R-STALL entries with a 3.0 m path-progress reset (A5.3, A5R1) ----------------

def _fires(f, positions, yaw=0.0, t=0.0, max_stand=3.0):
    """Stand at each position until the watchdog fires (an R-STALL entry) or the
    follower aborts. Each next position is >= 1.0 m from the previous one, so it
    is also the R-STALL exit. Returns ([(n_stall, s_hw) per entry], stop_reason
    or None, t)."""
    out = []
    yawfn = yaw if callable(yaw) else (lambda f_, p: yaw)
    for p in positions:
        t0 = t
        n0 = f.mpc_stats["handover"]["stall"]
        while True:
            _step(f, p, yawfn(f, p), t)
            t += 0.025
            if f.last_mode == "MPC-ABORT":
                return out, f.stop_reason, t
            if f.regime == "R-STALL" and f.mpc_stats["handover"]["stall"] > n0:
                out.append((f._n_stall, f._stall_s_hw))
                break
            assert t - t0 < max_stand, (p, f.regime, f.last_mode)
    return out, None, t


def _xs(*xs):
    return [(x, 0.0) for x in xs]


def test_f5_progress_resets_the_count():
    """T4: each entry >= 3.05 m beyond the previous one starts a new run."""
    f, _ = _f(LONG100)
    fires, end, _ = _fires(f, _xs(5.0, 8.05, 11.1, 14.15, 17.2))
    assert end is None and [n for n, _ in fires] == [1, 1, 1, 1, 1]
    assert f.mpc_stats["handover"]["stall"] == 5


def test_f5_progress_below_3_m_does_not_reset():
    """T5: 2.95 m between entries: counts 1, 2, 3, the 4th entry aborts."""
    f, _ = _f(LONG100)
    fires, end, _ = _fires(f, _xs(5.0, 7.95, 10.9, 13.85))
    assert end == "stall_budget" and [n for n, _ in fires] == [1, 2, 3]


def test_f5_sideways_or_backward_motion_is_not_progress():
    """T6: 4 m of lateral displacement, projection unchanged: the 4th aborts."""
    f, _ = _f(LONG100)
    fires, end, _ = _fires(f, [(5.0, 0.0), (5.0, 4.0), (5.0, 0.0), (5.0, 4.0)])
    assert end == "stall_budget" and len(fires) == 3


def test_f5_compares_with_the_furthest_earlier_entry():
    """T7: 1.6 m between entries (4.8 m from the first): no new run, the 4th aborts."""
    f, _ = _f(LONG100)
    fires, end, _ = _fires(f, _xs(5.0, 6.6, 8.2, 9.8))
    assert end == "stall_budget" and [n for n, _ in fires] == [1, 2, 3]


def test_f5_progress_uses_the_high_water_mark():
    """T8: one 100 m segment, entries at 10, 7.5, 11, 9.5: the third is 3.5 m
    beyond the previous entry but only 1.0 m beyond the furthest: counts
    1, 2, 3, abort on the 4th."""
    f, _ = _f([(0.0, 0.0, 0.0, 2.0), (100.0, 0.0, 0.0, 2.0)])
    fires, end, _ = _fires(f, _xs(10.0, 7.5, 11.0, 9.5))
    assert end == "stall_budget" and [n for n, _ in fires] == [1, 2, 3]


def test_f5_progress_mark_resets_with_set_path():
    """T8b: an entry at x = 30 sets the mark; after set_path, entries 3.5 m apart
    from x = 5 each start a new run."""
    f, _ = _f(LONG100)
    fires, end, t = _fires(f, _xs(30.0))
    assert f._stall_s_hw == pytest.approx(30.0, abs=0.01)
    f.set_path(LONG100, pos_xy=(5.0, 0.0), t=t)
    fires, end, _ = _fires(f, _xs(5.0, 8.5, 12.0, 15.5), t=t)
    assert end is None and [n for n, _ in fires] == [1, 1, 1, 1]


def test_f5_progress_mark_resets_with_the_block():
    """T8c: forward 40 m / reverse 5 m / forward 40 m; an entry at s ~ 30 in
    block 0, then entries 3.5 m apart in block 2 each start a new run."""
    path = (_densify([(0.0, 0.0, 0.0, 2.0), (40.0, 0.0, 0.0, 2.0)])
            + _densify([(40.0, 0.0, 0.0, -1.5), (35.0, 0.0, 0.0, -1.5)])
            + _densify([(35.0, 0.0, 0.0, 2.0), (75.0, 0.0, 0.0, 2.0)]))
    f, _ = _f(path)
    fires, end, t = _fires(f, _xs(30.0))
    assert f._stall_s_hw == pytest.approx(30.0, abs=0.01)
    for x in [31.0 + k for k in range(10)] + [40.0, 40.0]:       # to the cusp, brake, switch
        _step(f, (x, 0.0), 0.0, t)
        t += 0.025
    for x in [39.5 - 0.5 * k for k in range(10)] + [35.0, 35.0]:  # reverse block, second cusp
        _step(f, (x, 0.0), 0.0, t)
        t += 0.025
    assert f.block_i == 2
    fires, end, _ = _fires(f, _xs(36.0, 39.5, 43.0, 46.5), t=t)
    assert end is None and [n for n, _ in fires] == [1, 1, 1, 1]


def test_f5_count_right_after_a_run_start():
    """T13: entries 5.0, 8.2 (new run), 9.4, 8.2, 9.4: counts 1, 1, 2, 3, the
    5th entry aborts."""
    f, _ = _f(LONG100)
    fires, end, _ = _fires(f, _xs(5.0, 8.2, 9.4, 8.2, 9.4))
    assert [n for n, _ in fires] == [1, 1, 2, 3] and end == "stall_budget"


def test_f5_curve_inside_starts_runs_without_ground_progress():
    """T14, disclosure pin (plan review's construction): hairpin of radius
    7.4 m, car 1.9 m from the centre, entries 1.03 m apart on the ground —
    every entry starts a new run (progress is measured on the path projection)."""
    rm = 7.4
    c = [(-30.0, -rm, 0.0, 2.0)]
    c += [(rm * math.cos(-math.pi / 2 + math.pi * k / 36),
           rm * math.sin(-math.pi / 2 + math.pi * k / 36), 0.0, 2.0) for k in range(37)]
    c.append((-30.0, rm, 0.0, 2.0))
    f, _ = _f(_densify(c, spacing=0.5))
    pos = [(1.9 * math.cos(a), 1.9 * math.sin(a)) for a in [-math.pi / 2 + 0.55 * k for k in range(6)]]
    fires, end, _ = _fires(f, pos, yaw=lambda f_, p: f_.yaws[f_.current_idx])
    assert end is None and [n for n, _ in fires] == [1] * 6
    assert [round(s_, 1) for _, s_ in fires] == [30.2, 34.2, 38.1, 42.0, 46.4, 50.3]


def test_set_path_resets_budgets_and_terminal_state():
    """T9 (also the only kill of P6, by state inspection)."""
    f, _ = _f(LONG100)
    x, t = 2.0, 0.0
    for _ in range(4):
        _step(f, (x, 0.0), -2.0, t)
        x += 1.0; t += 0.025
    _step(f, (x, 0.0), 0.0, t)                               # release
    t += 0.025
    fires, end, t = _fires(f, _xs(x), t=t)
    assert f._d_align > 0 and f._n_stall == 1 and f._stall_s_hw is not None
    f.set_path(LONG100, pos_xy=(x, 0.0), t=t)
    assert (f._d_align, f._d_fail, f._n_stall, f._stall_s_hw) == (0.0, 0.0, 0, None)
    assert f.stop_reason is None and f._gp_hist == []
    assert f.mpc_stats["handover"] == {"align": 1, "stall": 1}


def test_align_budget_fills_from_position_jitter():
    """T10: a blocked car whose reported position alternates +-0.01 m fills the
    R-ALIGN budget (0.02 m per call): align_budget after budget / 0.02 calls."""
    f, _ = _f(LONG100)
    t, k = 0.0, 0
    while f.last_mode != "MPC-ABORT":
        _step(f, (5.0, 0.01 if k % 2 else -0.01), -2.0, t)
        t += 0.025; k += 1
        assert k < 2000
    assert f.stop_reason == "align_budget"
    assert abs(k - f._pd_budget / 0.02) <= 3


def test_f5_progress_mark_survives_a_done_step():
    """C-C4: the progress mark is NOT reset by the hand-over release that a
    command step (here DONE) performs. Entries at x = 5.0 / 6.1 / 7.2 (counts
    1-3), a DONE step, then an entry at x = 16.0: 16.0 >= 7.2 + 3.0, so a new
    run starts (count 1), no abort.

    Mutant this fails on: ``_stall_s_hw`` reset inside ``_clear_handover``
    (then the 16.0 entry is the 4th of the run and aborts ``stall_budget``)."""
    path = _densify([(0.0, 0.0, 0.0, 2.0), (20.0, 0.0, 0.0, 2.0)])
    f, _ = _f(path)
    fires, end, t = _fires(f, _xs(5.0, 6.1, 7.2))
    assert end is None and [n for n, _ in fires] == [1, 2, 3]
    _step(f, (19.6, 0.0), 0.0, t)
    assert f.last_mode == "DONE"
    fires, end, _ = _fires(f, _xs(16.0), t=t + 0.025)
    assert end is None and [n for n, _ in fires] == [1]
    assert f.stop_reason is None
