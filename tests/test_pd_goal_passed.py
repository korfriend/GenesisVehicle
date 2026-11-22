"""GP-AWAY2, the goal-passed stop (v2.0.1, every mode, cannot be switched off).

Predicate (``PathFollower._goal_passed``): final block, projection on its last
segment, beyond the goal's segment-normal plane, distance to the goal at least
``arrival_goal + 0.1`` m, and grown by more than 1e-3 m on each of the last
two ``plan()`` calls. Then ``GOAL-MISSED``, command ``(0, 0, 1)``,
``stop_reason == "goal_passed"``, sticky until ``set_path``; ``DONE`` (inside
``arrival_goal``) takes precedence and clears it.

Covered: the 1.x fixed point behind the goal (|yaw_err| -> 0.3*pi/0.7 = 1.346
rad, the car drives straight away) is reproduced with the stop disabled and
stopped with it; DONE first; ``arrival_goal + 0.05`` does not fire; one growth
step does not fire; the start of a loop path does not fire; a reverse final
block stops; ``set_path`` clears; the result counts ``plan()`` calls (same
call index at ``t`` spacing 0.025 and 0.05); and three literal runs that the
1.x follower completes (generator of ``.pm_r2/r210_probe/replan/gp_variants.py``,
seed 1, i = 31 / 52 / 142) still reach DONE.

Kinematic bicycle where a closed loop is used (wheelbase 2.7 m, |delta| <=
0.35, a in [-4, 2] m/s^2, dt 0.025), not the ray-wheel vehicle.

Mutants this file must fail on (plan_r210 section 6): pure GP (beyond the
plane, no growth / margin condition — the three literal runs then stop); the
0.1 m margin removed; the last-segment condition removed; the margin met by
an extrapolated next distance (G2b).
"""
import copy
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control import PathFollower  # noqa: E402
from genesis_vehicle.tests.data import make_pd_golden as G  # noqa: E402

L_WB, DMAX, DT = 2.7, 0.35, 0.025
STRAIGHT = G.densify([(0.0, 0.0, 0.0, 2.0), (10.0, 0.0, 0.0, 2.0)])


def _off(path=STRAIGHT):
    return PathFollower(path, G.make_table(), mpc_model="off")


def _walk(f, pts, yaw=0.0, v=1.0):
    modes = []
    for p in pts:
        modes.append(f.plan(p, yaw, v, 0.0, 0.0).mode)
    return modes


def _past_goal(y=2.0, x_end=14.0, dx=0.25):
    """States along +x at lateral offset y (never inside the 1.5 m circle at
    y=2): the projection walks to the last segment, then the car recedes."""
    n = int(x_end / dx)
    return [(dx * k, y) for k in range(n + 1)]


def _sim(path, x, y, yaw, T=60.0, stop=True):
    f = _off(path)
    if not stop:
        f._goal_passed = lambda pos_xy, d_goal: False
    v = 0.0
    for k in range(int(T / DT)):
        pl = f.plan((x, y), yaw, v, 0.0, 0.0)
        if pl.mode == "DONE":
            return f, "DONE", k, (x, y, yaw, v)
        if pl.command is not None:
            a, w = (-4.0 * math.copysign(1, v) if abs(v) > 0.05 else -v / DT), 0.0
        else:
            a, w = pl.a_target, pl.omega_target
        a = max(-4, min(2, a)); wm = abs(v) * math.tan(DMAX) / L_WB; w = max(-wm, min(wm, w))
        v += a * DT; yaw += w * DT; x += v * math.cos(yaw) * DT; y += v * math.sin(yaw) * DT
        if pl.mode == "GOAL-MISSED" and abs(v) < 1e-6:
            return f, "GOAL-MISSED", k, (x, y, yaw, v)
    return f, "timeout", k, (x, y, yaw, v)


def test_predicate_fires_past_the_goal_and_is_sticky():
    f = _off()
    modes = _walk(f, _past_goal())
    k = modes.index("GOAL-MISSED")
    assert all(m.startswith("DRV") for m in modes[:k])
    assert all(m == "GOAL-MISSED" for m in modes[k:])
    assert f.stop_reason == "goal_passed"
    x = _past_goal()[k][0]
    assert x > 10.0 and math.hypot(x - 10.0, 2.0) >= 1.6
    out = f.step((20.0, 2.0), 0.0, 1.0, 0.0, 0.0)
    assert out == (0.0, 0.0, 1.0) and f.last_mode == "GOAL-MISSED"


def test_fixed_point_behind_the_goal_is_reproduced_and_stopped():
    start = (4.0, 3.5, 0.0)
    g, out_g, _, (x, y, yaw, v) = _sim(STRAIGHT, *start, stop=False)
    # 1.x: no DONE within the 60 s run, drives straight away at the path speed; the heading law
    # sits at its fixed point |0.7 yaw_err + 0.3 pos_err| = 0, |yaw_err| ~ 1.346
    assert out_g == "timeout" and math.hypot(x - 10.0, y) > 50.0 and abs(v) > 1.9
    yaw_err = (g.yaws[g.current_idx] - yaw + math.pi) % (2 * math.pi) - math.pi
    assert abs(abs(yaw_err) - 0.3 * math.pi / 0.7) < 0.05
    f, out_f, k, (x, y, _, _) = _sim(STRAIGHT, *start)
    assert out_f == "GOAL-MISSED" and f.stop_reason == "goal_passed"
    assert math.hypot(x - 10.0, y) < 6.0


def test_done_takes_precedence_and_clears_the_reason():
    f = _off()
    _walk(f, _past_goal())
    assert f.last_mode in ("INIT",) and f.stop_reason == "goal_passed"
    f.step((10.5, 0.5), 0.0, 0.0, 0.0, 0.0)          # back inside the circle
    assert f.last_mode == "DONE" and f.stop_reason is None


def test_margin_holds_at_arrival_goal_plus_0_05():
    f = _off()
    _walk(f, [(0.25 * k, 0.0 if k < 30 else 1.0) for k in range(38)])   # projection to the end
    base = f._gp_hist[:]
    # beyond the plane, distance 1.51 -> 1.55 growing by > 1e-3 per call: no fire
    for d in (1.51, 1.52, 1.53, 1.54, 1.55):
        assert f.plan((10.0 + math.sqrt(d * d - 1.0), 1.0), 0.0, 1.0, 0.0, 0.0).mode != "GOAL-MISSED"
    assert base != f._gp_hist
    # the first growing call at >= arrival_goal + 0.1 fires
    assert f.plan((10.0 + math.sqrt(1.62 ** 2 - 1.0), 1.0), 0.0, 1.0, 0.0, 0.0).mode == "GOAL-MISSED"


def test_one_growth_step_does_not_fire():
    f = _off()
    _walk(f, [(0.25 * k, 2.0) for k in range(41)])     # x up to 10.0 (d = 2.0, minimum)
    assert f.stop_reason is None
    # equal, grow, equal, grow, grow: only the second consecutive growth fires
    seq = [(10.0, 2.0), (10.3, 2.0), (10.3, 2.0), (10.6, 2.0), (10.9, 2.0)]
    modes = _walk(f, seq)
    assert "GOAL-MISSED" not in modes[:4]
    assert modes[4] == "GOAL-MISSED"


def test_loop_path_start_does_not_fire():
    loop = G.densify([(0.0, 0.0, 0.0, 2.0), (10.0, 0.0, 0.0, 2.0), (10.0, 10.0, 0.0, 2.0),
                      (0.0, 10.0, 0.0, 2.0), (0.0, 2.0, 0.0, 2.0)])
    f = _off(loop)
    modes = _walk(f, [(-0.2 * k, -0.1 * k) for k in range(1, 30)], yaw=math.pi)
    assert "GOAL-MISSED" not in modes and f.current_idx < 3


def test_reverse_final_block_stops():
    path = (G.densify([(0.0, 0.0, 0.0, 2.0), (10.0, 0.0, 0.0, 2.0)])
            + G.densify([(10.0, 0.0, 0.0, -1.5), (0.0, 0.0, 0.0, -1.5)]))
    f = _off(path)
    assert f.plan((9.6, 0.0), 0.0, 0.0, 0.0, 0.0).mode == "BRAKE_TRANS"
    assert f.plan((9.6, 0.0), 0.0, 0.0, 0.0, 0.0).mode == "DRV-1" and f.block_i == 1
    modes = _walk(f, [(9.5 - 0.25 * k, 2.0) for k in range(60)], v=-1.0)
    k = modes.index("GOAL-MISSED")
    assert 9.5 - 0.25 * k < 0.0 and f.stop_reason == "goal_passed"


def test_set_path_clears_the_stop():
    f = _off()
    _walk(f, _past_goal())
    assert f.stop_reason == "goal_passed"
    f.set_path(STRAIGHT)
    assert f.stop_reason is None and f._gp_hist == []
    assert f.plan((0.5, 0.0), 0.0, 1.0, 0.0, 0.0).mode == "DRV+1"


def test_counts_plan_calls_not_time():
    pytest.importorskip("casadi")
    pytest.importorskip("do_mpc")
    from genesis_vehicle.control.mpc import MPCVehicleParams
    from genesis_vehicle.tests._mpc_fakes import FakeSolver, ok
    from genesis_vehicle.tests.data import make_mpc_field_replay as rp
    idx = []
    for h in (0.025, 0.05):
        f = PathFollower(STRAIGHT, G.make_table(),
                         mpc_params=MPCVehicleParams.from_dict(copy.deepcopy(rp.PARAMS_SDK)))
        f._law.solver = FakeSolver(default=ok(0.0, 0.5))
        for k, p in enumerate(_past_goal()):
            f.plan(p, 0.0, 1.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=k * h)
            if f.stop_reason == "goal_passed":
                idx.append(k)
                break
    assert len(idx) == 2 and idx[0] == idx[1]


# seed 1 of the gp_variants generator (leg1, leg2, theta, v; start pose), extracted
# by .pm_r2/r201b/extract_literals.py; 1.x reaches DONE at 43.2 / 5.5 / 10.4 s.
LITERAL_DONE = {
    31: (9.912181744309425, 9.46391407195376, -2.9410486345410773, 1.0,
         -0.6364101435336664, 0.46074413023614635, 1.771252567819932),
    52: (15.118383884490909, 8.728175472668754, 1.8137242578681603, 4.0,
         -1.1221232697904906, 1.6864140353069121, 1.653698584119696),
    142: (8.145414548628718, 5.263106762744217, -1.7805070751083165, 1.0,
          -1.0376740331940888, 1.5889274309460002, -1.4168967575373657),
}


@pytest.mark.parametrize("i", sorted(LITERAL_DONE))
def test_runs_that_1x_completes_still_complete(i):
    leg1, leg2, th, v0, x0, y0, yaw0 = LITERAL_DONE[i]
    path = G.densify([(0, 0, 0, v0), (leg1, 0, 0, v0),
                      (leg1 + leg2 * math.cos(th), leg2 * math.sin(th), 0, v0)])
    f, out, k, _ = _sim(path, x0, y0, yaw0, T=60.0)
    assert out == "DONE", (out, k * DT, f.stop_reason)


def test_margin_is_checked_on_this_calls_distance():
    """T11: distances 1.52 / 1.56 / 1.59 (growing) do not fire; 1.62 does.

    Mutant this fails on: the margin met by an extrapolated next distance (G2b)."""
    f = _off()
    _walk(f, [(0.25 * k, 0.0 if k < 30 else 1.0) for k in range(38)])
    modes = [f.plan((10.0 + math.sqrt(d * d - 1.0), 1.0), 0.0, 1.0, 0.0, 0.0).mode
             for d in (1.52, 1.56, 1.59)]
    assert "GOAL-MISSED" not in modes
    assert f.plan((10.0 + math.sqrt(1.62 ** 2 - 1.0), 1.0), 0.0, 1.0, 0.0, 0.0).mode == "GOAL-MISSED"
