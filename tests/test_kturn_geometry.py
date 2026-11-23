"""K-turn geometry, trigger rule and planner (v2.0.3, ``control/kturn.py``). Genesis-free, no MPC deps.

Covers plan r203 section 5.1: the forward-swing lower bound (closed values, sign of y' irrelevant); the
trigger's boundaries (strict ``>``, M_FWD, G, b, the lateral rule) and its domain (|v| <= 4.5 m/s, bend <=
45 deg, both inclusive); the remaining-bend formula (C2: stall course 43.6 deg); planner properties on 200
random poses (last leg forward, goal reached, legs within the budgets, reverse distance <= max_reverse,
phi <= pi, every region sample inside); the C1 budget (no reverse leg when ``rev_left == 0``); AC-7
mirror symmetry (plan and node count); a capped level discarded whole; at most ``per_call`` nodes per
call and ``ceil(cap / per_call)`` calls; determinism; ``KTurnParams`` validation.

Mutants this file must fail on: ``>`` -> ``>=``; ``-|y'|`` -> ``+|y'|``; G or b dropped; M_FWD changed; the
domain checks removed; the bend skipping the first (partial) segment; the level discard removed (one
direction first + cap); the per-call node budget removed; the reverse budget ignored.
"""
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control import kturn as K  # noqa: E402
from genesis_vehicle.control.kturn import KTurnParams, KTurnPlanner  # noqa: E402

RMIN = 2.7 / math.tan(0.35)
RP = K.KTURN_KAPPA * RMIN
MAXREV = 2.0 * RMIN


def _planner(s_rem0=6.0, lat=None, poly=None, cap=K.KTURN_NODE_CAP, per_call=K.KTURN_NODES_PER_CALL,
             max_rev=MAXREV):
    return KTurnPlanner(s_rem0, RP, RMIN, max_rev, math.pi / 4, 3.5, lateral_limit=lat, poly=poly,
                        node_cap=cap, per_call=per_call)


def _run(pl, x, y, p, legs_left=4, rev_left=2):
    s = pl.start(x, y, p, legs_left, rev_left)
    calls = 0
    while not s.advance():
        calls += 1
    return s.result, calls + 1


# --- trigger ------------------------------------------------------------------------------------

def test_forward_swing_lower_closed_values_and_sign_of_y():
    assert K.forward_swing_lower(math.pi, 0.0, RMIN) == pytest.approx(2 * RMIN)
    assert K.forward_swing_lower(math.pi / 2, 0.5, RMIN) == pytest.approx(RMIN - 0.5)
    assert K.forward_swing_lower(1.0, 0.3, RMIN) == K.forward_swing_lower(1.0, -0.3, RMIN)
    assert K.forward_swing_lower(-1.0, 0.3, RMIN) == K.forward_swing_lower(1.0, 0.3, RMIN)
    assert K.forward_swing(1.0, -0.3, RMIN) == pytest.approx(RMIN * (1 - math.cos(1.0)) + 0.3)


def _margin(e, y, s_rem, b, G):
    return RMIN * (1 - math.cos(e)) - abs(y) - (s_rem + b + G + 1.5)


@pytest.mark.parametrize("e,y,s_rem,b,G", [(math.pi, 0.0, 6.0, 0.0, 1.5), (math.radians(-128.2), 0.209, 3.971, 0.0, 1.5),
                                         (2.5, -0.4, 3.0, 0.7, 1.0)])
def test_trigger_margin_formula(e, y, s_rem, b, G):
    fire, m, dom = K.trigger(e, y, s_rem, b, G, 0.0, 1.0, RMIN)
    assert dom and m == pytest.approx(_margin(e, y, s_rem, b, G))
    assert fire == (m > 0)


def test_trigger_reference_courses():
    # straight N at the start (e = pi, y' = 0, s_rem = N, b = 0): plan section 3.2 / step 0
    for N, G, want in ((6, 1.5, 5.794), (10, 1.5, 1.794), (12, 1.5, -0.206), (12, 1.0, 0.294), (15, 1.5, -3.206)):
        fire, m, _ = K.trigger(math.pi, 0.0, float(N), 0.0, G, 0.0, 0.0, RMIN)
        assert m == pytest.approx(want, abs=1e-3) and fire == (want > 0)


def test_trigger_is_strict_and_uses_every_term():
    e = math.pi
    s_rem = 2 * RMIN - 1.5 - 1.5            # margin exactly 0
    fire, m, _ = K.trigger(e, 0.0, s_rem, 0.0, 1.5, 0.0, 0.0, RMIN)
    assert m == pytest.approx(0.0, abs=1e-12) and not fire
    assert K.trigger(e, 0.0, s_rem - 1e-6, 0.0, 1.5, 0.0, 0.0, RMIN)[0]
    # b and G count as available length; |y'| reduces the need on either side
    assert not K.trigger(e, 0.0, s_rem - 1e-6, 1e-5, 1.5, 0.0, 0.0, RMIN)[0]
    assert not K.trigger(e, 0.0, s_rem - 1e-6, 0.0, 1.5 + 1e-5, 0.0, 0.0, RMIN)[0]
    assert not K.trigger(e, 1e-5, s_rem - 1e-6, 0.0, 1.5, 0.0, 0.0, RMIN)[0]
    assert not K.trigger(e, -1e-5, s_rem - 1e-6, 0.0, 1.5, 0.0, 0.0, RMIN)[0]
    assert K.KTURN_M_FWD == 1.5


def test_trigger_domain_inclusive_and_lateral_rule():
    args = (math.pi, 0.0, 6.0, 0.0, 1.5)
    assert K.trigger(*args, 0.0, 4.5, RMIN)[0]
    assert K.trigger(*args, 0.0, -4.5, RMIN)[0]
    assert not K.trigger(*args, 0.0, 4.5 + 1e-9, RMIN)[0]
    assert K.trigger(*args, math.radians(45), 0.0, RMIN)[0]
    f, _, dom = K.trigger(*args, math.radians(45) + 1e-9, 0.0, RMIN)
    assert not f and not dom
    # lateral rule: margin negative (long block), swing 14.79 > 8 -> fires; > 15 -> not
    assert not K.trigger(math.pi, 0.0, 60.0, 0.0, 1.5, 0.0, 0.0, RMIN)[0]
    assert K.trigger(math.pi, 0.0, 60.0, 0.0, 1.5, 0.0, 0.0, RMIN, lateral_limit=8.0)[0]
    assert not K.trigger(math.pi, 0.0, 60.0, 0.0, 1.5, 0.0, 0.0, RMIN, lateral_limit=15.0)[0]
    assert not K.trigger(math.pi, 0.0, 60.0, 0.0, 1.5, 0.0, 5.0, RMIN, lateral_limit=8.0)[0]


def test_remaining_bend_stall_course():
    P = [(30.0, 0.0), (10.0, -8.0), (-10.0, -8.0), (-30.0, 0.0)]
    yaws = [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(P[:-1], P[1:])]
    assert math.degrees(K.remaining_bend(yaws)) == pytest.approx(43.6, abs=0.1)
    assert math.degrees(K.remaining_bend(yaws[1:])) == pytest.approx(21.8, abs=0.1)
    # jitter accumulates (documented: can disable the K-turn on a noisy path)
    jit = [0.01 * (-1) ** k for k in range(100)]
    assert K.remaining_bend(jit) == pytest.approx(99 * 0.02)


def test_s_join():
    assert K.s_join(RP, 0.0, 0.0) == 0.0
    assert K.s_join(RP, 0.3, 0.2) == K.s_join(RP, -0.3, -0.2)
    assert K.s_join(RP, math.pi / 4, 0.0) > 0


# --- params -------------------------------------------------------------------------------------

def test_kturn_params_validation():
    p = KTurnParams()
    assert (p.lateral_limit, p.max_reverse, p.max_legs) == (None, None, 4)
    import numpy as np
    q = KTurnParams(lateral_limit=np.float64(8), max_reverse=10, max_legs=np.int64(2))
    assert q.lateral_limit == 8.0 and isinstance(q.max_reverse, float) and q.max_legs == 2
    for kw in (dict(lateral_limit=0.0), dict(lateral_limit=-1.0), dict(lateral_limit=float("nan")),
               dict(lateral_limit=True), dict(max_reverse=float("inf")), dict(max_reverse="3"),
               dict(max_legs=5), dict(max_legs=1), dict(max_legs=True), dict(max_legs=3.0)):
        name = next(iter(kw))
        with pytest.raises(ValueError, match=name):
            KTurnParams(**kw)
    with pytest.raises(Exception):
        p.max_legs = 3                       # frozen


# --- planner ------------------------------------------------------------------------------------

def _check_plan(pl, x, y, p, legs, legs_left, rev_left):
    assert legs, (x, y, p)
    assert len(legs) <= legs_left
    assert legs[-1][0] == 1
    assert sum(1 for l in legs if l[0] < 0) <= rev_left
    for g, d, phi in legs:
        assert g in (1, -1) and d in (1, -1) and 0 < phi <= math.pi + 1e-12
        if g < 0:
            assert RP * phi <= MAXREV + 1e-9
        assert pl.region(x, y, p, g, d, phi)
        for f in K._REGION_F:
            assert K.arc(x, y, p, g, d, phi * f, RP)[0] <= pl.s_rem0
        x, y, p = K.arc(x, y, p, g, d, phi, RP)
    e = K.wrap(p)
    assert abs(e) <= math.pi / 4 and abs(y) <= K.KTURN_Y_TOL
    assert K.s_join(RP, e, y) + 3.5 <= pl.s_rem0 - x


def _poses(n=200, seed=7):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        out.append((rnd.uniform(-2.0, 1.0), rnd.uniform(-1.5, 1.5),
                    rnd.choice((1, -1)) * rnd.uniform(math.pi / 2, math.pi), rnd.uniform(3.0, 12.0)))
    return out


def test_planner_properties_random():
    found = 0
    for x, y, p, s in _poses():
        pl = _planner(s)
        legs, _ = _run(pl, x, y, p)
        assert pl.nodes <= K.KTURN_NODE_CAP
        if legs is None:
            continue
        found += 1
        _check_plan(pl, x, y, p, legs, 4, 2)
    assert found >= 150


def test_reverse_budget_carried_across_replans_C1():
    # a pose whose best plan uses reverse legs; with rev_left == 0 the planner must not use any (R,R,R,F
    # refused), and with legs_left == 1 it plans one forward leg or nothing.
    x, y, p, s = 0.0, 0.0, math.pi, 6.0
    full, _ = _run(_planner(s), x, y, p, 4, 2)
    assert any(l[0] < 0 for l in full)
    none_rev, _ = _run(_planner(s), x, y, p, 2, 0)
    assert none_rev is None or all(l[0] > 0 for l in none_rev)
    one, _ = _run(_planner(s), x, y, p, 1, 1)
    assert one is None or (len(one) == 1 and one[0][0] == 1)
    for legs_left in (1, 2, 3, 4):
        for rev_left in (0, 1, 2):
            r, _ = _run(_planner(s), x, y, p, legs_left, rev_left)
            if r is not None:
                assert len(r) <= legs_left and sum(1 for l in r if l[0] < 0) <= rev_left


def test_mirror_symmetry_AC7():
    n = 0
    for x, y, p, s in _poses(200, seed=11):
        if p == 0.0 and y == 0.0:
            continue
        a, b = _planner(s), _planner(s)
        la, _ = _run(a, x, y, p)
        lb, _ = _run(b, x, -y, -p)
        assert a.nodes == b.nodes
        if la is None:
            assert lb is None
            continue
        n += 1
        assert [(g, -d, phi) for g, d, phi in la] == lb
    assert n >= 150


def test_mirror_symmetry_with_lateral_limit():
    poly = [(x, 0.0) for x in (-40.0, -20.0, 0.0, 6.0)]
    mpoly = [(x, -y) for x, y in poly]
    for x, y, p, s in _poses(40, seed=3):
        a = _planner(min(s, 6.0), lat=8.0, poly=poly)
        b = _planner(min(s, 6.0), lat=8.0, poly=mpoly)
        la, _ = _run(a, x, y, p)
        lb, _ = _run(b, x, -y, -p)
        assert a.nodes == b.nodes
        assert (la is None and lb is None) or [(g, -d, phi) for g, d, phi in la] == lb


def test_lateral_limit_region_and_infeasible():
    poly = [(-40.0, 0.0), (6.0, 0.0)]
    pl = _planner(6.0, lat=8.0, poly=poly)
    legs, _ = _run(pl, 0.0, 0.0, math.pi)
    assert legs is not None
    x, y, p = 0.0, 0.0, math.pi
    for g, d, phi in legs:
        for f in K._REGION_F:
            assert abs(K.arc(x, y, p, g, d, phi * f, RP)[1]) <= 8.0
        x, y, p = K.arc(x, y, p, g, d, phi, RP)
    small, _ = _run(_planner(6.0, lat=2.0, poly=poly), 0.0, 0.0, math.pi)
    assert small is None
    assert _planner(6.0, lat=2.0, poly=poly).lateral_violated(0.0, 0.0, math.pi, legs)


def test_capped_level_is_discarded_whole():
    x, y, p, s = 0.0, 0.3, 2.8, 8.0
    full = _planner(s)
    _run(full, x, y, p)
    # node counts at the end of each level
    per_level = []
    for n in range(1, 5):
        pl = _planner(s)
        _run(pl, x, y, p, legs_left=n)
        per_level.append(pl.nodes)
    for n in range(2, 5):
        if per_level[n - 1] == per_level[n - 2]:
            continue
        cap = per_level[n - 2] + (per_level[n - 1] - per_level[n - 2]) // 2   # inside level n
        capped, _ = _run(_planner(s, cap=cap), x, y, p)
        upto, _ = _run(_planner(s), x, y, p, legs_left=n - 1)
        assert capped == upto


def test_per_call_budget_and_call_count():
    pl = _planner(6.0, per_call=1000)
    s = pl.start(0.0, 0.0, math.pi, 4, 2)
    last, calls = 0, 0
    while True:
        done = s.advance()
        calls += 1
        assert pl.nodes - last <= 1000
        last = pl.nodes
        if done:
            break
    assert calls == max(1, math.ceil(pl.nodes / 1000)) or calls == math.ceil(pl.nodes / 1000) + 1
    assert calls >= 2
    capped = _planner(6.0, cap=2500, per_call=1000)
    _, c = _run(capped, 0.0, 0.0, math.pi)
    assert capped.nodes == 2500 and c <= 3


def test_planner_is_deterministic():
    r = [_run(_planner(7.0), -0.4, 0.2, 2.9) for _ in range(3)]
    assert r[0] == r[1] == r[2]


def test_remaining_ok_keeps_a_valid_plan():
    pl = _planner(6.0)
    legs, _ = _run(pl, 0.0, 0.0, math.pi)
    assert pl.remaining_ok(0.0, 0.0, math.pi, legs)
    x, y, p = K.arc(0.0, 0.0, math.pi, *legs[0], RP)
    assert pl.remaining_ok(x, y, p, legs[1:])
    assert not pl.remaining_ok(0.0, 0.0, 0.5 * math.pi, legs)
