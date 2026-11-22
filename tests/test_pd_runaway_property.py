"""S-8: outcome-based run-away property of the 2.0.1 follower (plan_r210 section 6).

Kinematic bicycle (NOT the ray-wheel vehicle): wheelbase 2.7 m, |delta| <= 0.35,
a in [-4, 2] m/s^2, dt 0.025, the follower's stage-1 targets applied directly
(``plan()`` only: no KICK, no table inversion); a command step brakes at
``-4 * sign(v)`` (``-v/dt`` below 0.05 m/s). 120 s horizon.

``"off"`` part — seed 21, 120 ``std`` + 120 ``cusp`` random paths, clean and
with noise (pose 5 mm / 0.002 rad, speed 0.02 m/s fed to the follower). The
path generator ``gen`` is copied VERBATIM from the planner's
``.pm_r2/r210_probe/replan2/lb_validate_gen.py`` (its ``densify`` /
``make_table`` from ``tests/data/make_pd_golden``). The expected outcome counts
and the expected-escape list were produced BEFORE the implementation by an
emulation probe (``.pm_r2/r210_probe/replan3/expected_escapes_seed21.py``,
condition R4-C1); here GP-AWAY2 and W2 come from the implemented code. An
"escape" is a run that at 120 s is neither ``DONE`` nor stopped in
``GOAL-MISSED``; the listed escapes are the 1.x run-away classes 2.0.1 does
not stop (plan section 3B.4 (d)) — strict: an escape not in the list fails,
and a listed run that finishes or stops also fails.

MPC part — default kwargs with two fake solvers (always converged with
``u = 0``, always raising), 100 ``std`` paths each (single forward block):
the PD travel on the forward block (after R-ALIGN / R-STALL / R-FAIL steps,
plus the braking after an ``MPC-ABORT``) stays within
2 * (pi * R_min + v_max * dt) + n_stall * (1.0 m + v_max * dt) +
v_max^2 / (2 * 4 m/s^2), n_stall = the run's R-STALL entries, and n_stall <=
3 * (floor(L_block / 3.0 m) + 1) (plan_r210 A5.5 item 5 / A5R1.10).
With the always-raising solver every run ends within 120 s in a terminal
mode. With the always-converged ``u = 0`` solver a run whose PD hand-over left
the car moving coasts on (this plant has no drag, and ``a = 0`` under R-MPC):
measured 39 of 100 runs still in R-MPC at 120 s on the final 2.0.1 tree (the
others: 55 GOAL-MISSED, 5 DONE, 1 MPC-ABORT align_budget), by a survey script
that reuses this test's helpers, not by this test. That is the documented
exclusion — distance driven by R-MPC is not budgeted — so the test asserts
that every run that does not end is in R-MPC, not that every run ends
(deviation from the plan's wording, reported).

Mutants this file fails on (measured; plan_r210 A5R1.M with R5-C3): GP-AWAY2
removed; W2 removed; F8 removed; F3 removed; the F8 budget reset at each R-FAIL
episode. It does NOT kill (measured): F5 removed, the F3 budget reset at each
R-ALIGN entry — those are killed by ``test_mpc_handover.py``.
"""
import copy
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control import PathFollower  # noqa: E402
from genesis_vehicle.tests.data import make_pd_golden as G  # noqa: E402

L_WB, DMAX, DT, SEED, N_RUNS, T_END = 2.7, 0.35, 0.025, 21, 120, 120.0


# --- copied verbatim from .pm_r2/r210_probe/replan2/lb_validate_gen.py -------------------
def gen(rnd, kind):
    if kind == "cusp":
        v1, v2 = rnd.choice((1.0, 2.0, 4.0)), -rnd.choice((1.0, 2.0))
        l1, l2 = rnd.uniform(5, 20), rnd.uniform(4, 12); th = math.radians(rnd.uniform(-60, 60))
        p1 = G.densify([(0, 0, 0, v1), (l1, 0, 0, v1)])
        p2 = G.densify([(l1, 0, 0, v2), (l1 - l2 * math.cos(th), -l2 * math.sin(th), 0, v2)])
        path = p1 + p2[1:] if rnd.random() < 0.5 else p1 + p2
        if rnd.random() < 0.5:
            e = path[-1]; v3 = rnd.choice((1.0, 2.0)); path = path + G.densify([(e[0], e[1], 0, v3), (e[0] - 8, e[1] + 4, 0, v3)])[1:]
        return path
    sp = (1.0, 2.0, 4.0) if kind == "std" else (4.0, 6.0, 8.0)
    leg1, leg2 = rnd.uniform(3, 20), rnd.uniform(3, 20); th = math.radians(rnd.uniform(0, 170)) * rnd.choice((-1, 1)); v0 = rnd.choice(sp)
    return G.densify([(0, 0, 0, v0), (leg1, 0, 0, v0), (leg1 + leg2 * math.cos(th), leg2 * math.sin(th), 0, v0)])
# -------------------------------------------------------------------------------------------


#: Expected outcomes, seed 21 (from expected_escapes_seed21.json; not edited).
EXPECTED = {
    "std_clean": dict(counts={"DONE": 50, "GOAL-MISSED": 68, "stopped": 0, "escape": 2},
                      escapes={13: "other", 108: "lastseg_near_side"}),
    "std_noisy": dict(counts={"DONE": 50, "GOAL-MISSED": 68, "stopped": 0, "escape": 2},
                      escapes={13: "other", 108: "lastseg_near_side"}),
    "cusp_clean": dict(counts={"DONE": 15, "GOAL-MISSED": 102, "stopped": 0, "escape": 3},
                       escapes={8: "nonfinal_block", 32: "other", 34: "other"}),
    "cusp_noisy": dict(counts={"DONE": 14, "GOAL-MISSED": 104, "stopped": 0, "escape": 2},
                       escapes={8: "nonfinal_block", 34: "other"}),
}
# Trace notes of the "other" escapes (from the expectation probe, R4-C1):
#   std i=13: pinned at current_idx 48 of a 55-waypoint single block (mid-path),
#             2.00 m/s, 221.6 m from the goal and receding.
#   cusp i=32: final forward block [49, 65], current_idx 64, 1.00 m/s, 39.2 m from
#              the goal, beyond the goal plane and approaching slowly (-1.8 m over
#              the last 10 s).
#   cusp i=34: final forward block [30, 46], current_idx 45, 2.00 m/s, 63.4 m from
#              the goal, beyond the goal plane, moving at 2 m/s at a nearly constant
#              distance (+0.4 m over the last 10 s): the distance does not grow on two
#              consecutive calls, so GP-AWAY2 does not fire.


def _beyond_plane(f, x, y):
    bs, be, _ = f.blocks[f.block_i]
    g = f.path[be]
    j = be - 1
    while j > bs and math.hypot(g[0] - f.path[j][0], g[1] - f.path[j][1]) < 1e-9:
        j -= 1
    return (x - g[0]) * (g[0] - f.path[j][0]) + (y - g[1]) * (g[1] - f.path[j][1]) > 0


def _brake(v):
    return -4.0 * math.copysign(1, v) if abs(v) > 0.05 else -v / DT


def sim_off(path, x, y, yaw, nseed, noisy, T=T_END, no_stop=False):
    nz = random.Random(nseed)
    f = PathFollower(path, G.make_table(), mpc_model="off")
    if no_stop:                                     # 1.x + W2: the stop disabled
        f._goal_passed = lambda pos_xy, d_goal: False
    v = 0.0
    sp, sy, sv = (0.005, 0.002, 0.02) if noisy else (0.0, 0.0, 0.0)
    x10 = y10 = None
    for k in range(int(T / DT)):
        mx, my, myaw, mv = x + nz.gauss(0, sp), y + nz.gauss(0, sp), yaw + nz.gauss(0, sy), v + nz.gauss(0, sv)
        pl = f.plan((mx, my), myaw, mv, 0.0, 0.0)
        if pl.mode == "DONE":
            return "DONE", None
        if pl.command is not None:
            a, w = _brake(v), 0.0
        else:
            a, w = pl.a_target, pl.omega_target
        a = max(-4, min(2, a)); wm = abs(v) * math.tan(DMAX) / L_WB; w = max(-wm, min(wm, w))
        v += a * DT; yaw += w * DT; x += v * math.cos(yaw) * DT; y += v * math.sin(yaw) * DT
        if pl.mode == "GOAL-MISSED" and abs(v) < 1e-6:
            return "GOAL-MISSED", None
        if k == int((T - 10) / DT):
            x10, y10 = x, y
    g = path[-1]
    d = math.hypot(x - g[0], y - g[1])
    bs, be, sgn = f.blocks[f.block_i]
    if abs(v) < 0.05:
        return "stopped", None
    if f.block_i != len(f.blocks) - 1:
        c = "nonfinal_block"
    elif f.current_idx >= be - 1 and not _beyond_plane(f, x, y) and d >= 10:
        c = "lastseg_near_side"
    elif f.block_i == 0 and f.current_idx <= 5 and d >= 10:
        c = "start_pinned"
    elif d < 10:
        c = "near_goal"
    else:
        c = "other"
    return "escape", c


@pytest.fixture(scope="module")
def off_outcomes():
    out = {}
    for kind in ("std", "cusp"):
        for noisy in (False, True):
            rnd = random.Random(SEED)
            res = []
            for i in range(N_RUNS):
                path = gen(rnd, kind)
                x0, y0, yaw0 = rnd.uniform(-2, 2), rnd.uniform(-2, 2), rnd.uniform(-math.pi, math.pi)
                res.append(sim_off(path, x0, y0, yaw0, SEED * 7919 + i, noisy))
            out[f"{kind}_{'noisy' if noisy else 'clean'}"] = res
    return out


@pytest.mark.parametrize("config", sorted(EXPECTED))
def test_off_goal_missed_is_never_a_false_stop(off_outcomes, config):
    """Every GOAL-MISSED run: the same run without the stop (1.x + W2) does
    not reach DONE within 150 s (expected false stops: 0)."""
    kind, mode = config.split("_")
    rnd = random.Random(SEED)
    false = []
    for i in range(N_RUNS):
        path = gen(rnd, kind)
        x0, y0, yaw0 = rnd.uniform(-2, 2), rnd.uniform(-2, 2), rnd.uniform(-math.pi, math.pi)
        if off_outcomes[config][i][0] != "GOAL-MISSED":
            continue
        if sim_off(path, x0, y0, yaw0, SEED * 7919 + i, mode == "noisy", T=150.0,
                   no_stop=True)[0] == "DONE":
            false.append(i)
    assert false == []


@pytest.mark.parametrize("config", sorted(EXPECTED))
def test_off_outcome_counts_match_the_pre_registered_expectation(off_outcomes, config):
    res = off_outcomes[config]
    counts = {"DONE": 0, "GOAL-MISSED": 0, "stopped": 0, "escape": 0}
    for outcome, _ in res:
        counts[outcome] += 1
    assert counts == EXPECTED[config]["counts"]


@pytest.mark.parametrize("config", sorted(EXPECTED))
def test_off_escapes_are_exactly_the_listed_ones(off_outcomes, config):
    """Strict: an unlisted escape fails; a listed run that ends fails."""
    got = {i: cls for i, (outcome, cls) in enumerate(off_outcomes[config]) if outcome == "escape"}
    assert got == EXPECTED[config]["escapes"]


# --- MPC mode, fake solvers ---------------------------------------------------------------

def _mpc_follower(path, fake_kind):
    pytest.importorskip("casadi")
    pytest.importorskip("do_mpc")
    from genesis_vehicle.control.mpc import MPCVehicleParams
    from genesis_vehicle.tests._mpc_fakes import FakeSolver, exc, ok
    from genesis_vehicle.tests.data import make_mpc_field_replay as rp
    f = PathFollower(path, G.make_table(),
                     mpc_params=MPCVehicleParams.from_dict(copy.deepcopy(rp.PARAMS_SDK)))
    f._law.solver = FakeSolver(default=ok(0.0, 0.0) if fake_kind == "converged" else exc())
    return f


@pytest.mark.parametrize("fake_kind", ["converged", "failing"])
def test_mpc_default_runs_end_and_pd_travel_is_bounded(fake_kind):
    rnd = random.Random(SEED)
    worst = 0.0
    n_coast = 0
    for i in range(100):
        path = gen(rnd, "std")
        x, y, yaw = rnd.uniform(-2, 2), rnd.uniform(-2, 2), rnd.uniform(-math.pi, math.pi)
        f = _mpc_follower(path, fake_kind)
        v_max = max(abs(p[3]) for p in path)
        # one forward block and no set_path in this generator, so the per-block
        # entry cap applies to the whole run (C-C7: handover["stall"] is
        # cumulative since construction, not per block)
        l_block = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:]))
        v, pd_travel, ended, prev_pd = 0.0, 0.0, None, False
        for k in range(int(T_END / DT)):
            pl = f.plan((x, y), yaw, v, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=k * DT)
            if pl.mode == "DONE":
                ended = "DONE"
                break
            braking_abort = pl.mode == "MPC-ABORT"
            if pl.command is not None:
                a, w = _brake(v), 0.0
            else:
                a, w = pl.a_target, pl.omega_target
            a = max(-4, min(2, a)); wm = abs(v) * math.tan(DMAX) / L_WB; w = max(-wm, min(wm, w))
            x1, y1 = x, y
            v += a * DT; yaw += w * DT; x += v * math.cos(yaw) * DT; y += v * math.sin(yaw) * DT
            if f.regime in ("R-ALIGN", "R-STALL", "R-FAIL") or braking_abort:
                pd_travel += math.hypot(x - x1, y - y1)
            if pl.mode in PathFollower.TERMINAL_MODES and abs(v) < 1e-6:
                ended = pl.mode
                break
        if fake_kind == "failing":
            assert ended is not None, f"run {i}: no terminal mode in 120 s ({f.regime})"
        elif ended is None:
            assert f.regime == "R-MPC", f"run {i}: not ended and not in R-MPC ({f.regime})"
            n_coast += 1
        n_stall = f.mpc_stats["handover"]["stall"]
        assert n_stall <= 3 * (math.floor(l_block / 3.0) + 1)
        bound = (2 * (math.pi * f._r_min + v_max * DT) + n_stall * (1.0 + v_max * DT)
                 + v_max ** 2 / (2 * 4.0))
        assert pd_travel <= bound, f"run {i}: PD travel {pd_travel:.2f} m > bound {bound:.2f} m"
        worst = max(worst, pd_travel / bound)
    assert worst > 0.0                                  # the PD regimes were exercised
    if fake_kind == "converged":
        assert n_coast < 100                            # some runs do end (abort / GOAL-MISSED)
