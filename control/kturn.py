"""K-turn geometry, trigger and planner for :class:`~genesis_vehicle.control.PathFollower` (v2.0.3, OPT-IN).

A gear-mode MPC follower constructed with ``mpc_kturn=KTurnParams()`` may, at the FIRST heading-gate entry
(R-ALIGN) of a forward block, decide that a forward-only turn will not fit and drive a multi-point turn
instead: stop, plan a sequence of arcs ("legs", planned at r_plan, driven at full lock) in forward and reverse
gear, drive them, and hand back to the MPC. This module holds the parts that do not need the follower: the parameters, the trigger rule,
the remaining-bend measure and the deterministic planner. numpy / math only, no Genesis.

| abbr | meaning |
|---|---|
| e | heading error wrap(chassis yaw - path yaw at the projection), rad (the follower's ``-_heading_error``) |
| y' | signed lateral offset of the chassis from the projection segment, m, left positive |
| s_rem | arc length from the projection point to the active block's end, m |
| b | distance of the chassis behind the block's first waypoint along its first segment, m (>= 0) |
| G | ``arrival_goal`` on the final block, ``cusp_goal`` on an earlier one, m |
| R_min | wheelbase / tan(steer_cap * max_steer_rad), m |
| r_plan | planning radius, ``KTURN_KAPPA * R_min`` |
| leg | one arc at full steering lock in one gear |
| CTE | cross-track error, distance to the path polyline |

**The trigger is a fitted proxy, not a feasibility test.** ``fire`` when
``R_min (1 - cos e) - |y'| > s_rem + b + G + KTURN_M_FWD``, or, with a ``lateral_limit`` L, when
``R_min (1 - cos e) + |y'| > L``. The left side ``R_min (1 - cos e) - |y'|`` is a LOWER BOUND of the
forward swing (exact when the chassis sits on the side opposite to where it points, smaller otherwise), so
the rule leans against firing. It predicts the size of the forward swing, not whether the 2.0.2 forward turn
reaches the goal circle: on the reference car's "straight 10" course (the car 10 m from the goal, facing
away) it fires although the 2.0.2 forward turn reaches ``DONE`` (grazing the 1.5 m circle at 1.471 m). Only
inside the domain ``|v_long| <= KTURN_V_MAX`` and remaining bend ``<= KTURN_BEND_MAX``.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Optional

#: Chosen values (plan r203), not public API.
KTURN_M_FWD = 1.5                      # trigger margin, m
KTURN_V_MAX = 4.5                      # domain: |v_long| at the decision, m/s
KTURN_BEND_MAX = math.radians(45.0)    # domain: remaining bend of the block, rad
KTURN_KAPPA = 1.2                      # r_plan = KTURN_KAPPA * R_min
KTURN_V = 1.0                          # leg speed, m/s
KTURN_GRID_DEG = 15                    # leg angle grid, deg
KTURN_Y_TOL = 1.0                      # goal: |y'| after the last leg, m
KTURN_W_Y = 2.0                        # cost weight of the plan's max |y'|
KTURN_NODE_CAP = 20000                 # nodes per plan
KTURN_NODES_PER_CALL = 1000            # nodes per follower call
KTURN_LEG_TOL = math.radians(2.0)      # a leg ends at phi - KTURN_LEG_TOL of signed rotation
KTURN_NOPROG_S = 2.0                   # no-progress abort window, s (in driving calls: round(2.0 / dt_ctrl))
KTURN_MAX_REVERSE_FACTOR = 2.0         # max_reverse=None -> 2.0 * R_min
KTURN_MAX_LEGS = 4                     # upper limit of KTurnParams.max_legs
KTURN_MIN_LEGS = 2
#: region samples per arc (fractions of the leg angle) and the |y'| samples of the cost
_REGION_F = (1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0)
_YMAX_F = (0.25, 0.5, 0.75, 1.0)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _is_real(x) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


@dataclass(frozen=True)
class KTurnParams:
    """(v2.0.3) Enables the K-turn of a gear-mode MPC :class:`PathFollower` (``mpc_kturn=KTurnParams()``;
    the follower's default ``mpc_kturn=None`` is OFF). Three different ``None`` meanings:

    lateral_limit: ``None`` (default) = no lateral limit on the K-turn's planned arcs (the excursion is then
        bounded only through the leg-distance caps). A number L (m, > 0): every sample of every planned arc
        stays within L of the path polyline (a window of it, see the planner), AND the trigger also fires
        when the forward swing ``R_min (1 - cos e) + |y'|`` exceeds L. With L given, "no plan" before the
        first leg, or continuing an old plan that would leave L, ends on ``MPC-ABORT`` /
        ``"kturn_infeasible"``. Planned arcs only: the executed trajectory is not guaranteed inside L, and
        the 2.0.2 forward alignment (R-ALIGN) is never limited by it.
    max_reverse: ``None`` (default) = ``2.0 * R_min`` (always finite; 14.79 m on the reference car). A number
        (m, > 0): the distance cap of ONE reverse leg (planned and driven; a driven leg can exceed it by the
        travel of one control call plus the uncounted braking travel of mid-leg re-settle calls — at
        most 120 brake-1 calls per hand-over, after a SHIFT / GUARD inside the leg).
    max_legs: 2..4 (default 4): legs per episode across re-plans; reverse legs <= ``max_legs // 2``. An
        episode can contain zero reverse legs (a single forward leg). (5 was refused: it exceeds the node
        cap on measured courses and would silently act as 4.)
    """
    lateral_limit: Optional[float] = None
    max_reverse: Optional[float] = None
    max_legs: int = KTURN_MAX_LEGS

    def __post_init__(self):
        for name in ("lateral_limit", "max_reverse"):
            v = getattr(self, name)
            if v is None:
                continue
            if not _is_real(v) or not (math.isfinite(float(v)) and float(v) > 0.0):
                raise ValueError(f"KTurnParams.{name} must be None or a finite number > 0 (m); got {v!r}")
            object.__setattr__(self, name, float(v))
        m = self.max_legs
        if not (isinstance(m, numbers.Integral) and not isinstance(m, bool)) or not (
                KTURN_MIN_LEGS <= int(m) <= KTURN_MAX_LEGS):
            raise ValueError(f"KTurnParams.max_legs must be an int in [{KTURN_MIN_LEGS}, {KTURN_MAX_LEGS}]; "
                             f"got {m!r}")
        object.__setattr__(self, "max_legs", int(m))


def forward_swing_lower(e: float, y: float, r_min: float) -> float:
    """``R_min (1 - cos e) - |y'|``: a lower bound of the lateral swing of a forward turn back to the path
    heading (the plan's N_need)."""
    return r_min * (1.0 - math.cos(e)) - abs(y)


def forward_swing(e: float, y: float, r_min: float) -> float:
    """``R_min (1 - cos e) + |y'|`` (W_fwd), compared with ``lateral_limit``."""
    return r_min * (1.0 - math.cos(e)) + abs(y)


def remaining_bend(yaws) -> float:
    """Sum of ``|wrap(delta yaw)|`` over consecutive entries of ``yaws`` (the segment yaws from the
    projection segment to the block end; plan r203 C2, p10_rev1.py:110). Heading jitter on a densely or
    noisily sampled path accumulates here and can push it past ``KTURN_BEND_MAX``, which disables the K-turn
    (2.0.2 behaviour)."""
    s, prev = 0.0, None
    for y in yaws:
        if prev is not None:
            s += abs(wrap(float(y) - prev))
        prev = float(y)
    return s


def trigger(e, y, s_rem, behind, goal_r, bend, v_long, r_min, lateral_limit=None):
    """``(fire, margin, in_domain)``; ``margin = forward_swing_lower - (s_rem + b + G + KTURN_M_FWD)``
    (fires iff > 0, or by the lateral rule). Outside the domain ``fire`` is False."""
    margin = forward_swing_lower(e, y, r_min) - (s_rem + behind + goal_r + KTURN_M_FWD)
    dom = abs(v_long) <= KTURN_V_MAX and bend <= KTURN_BEND_MAX
    if not dom:
        return False, margin, False
    fire = margin > 0.0
    if lateral_limit is not None and forward_swing(e, y, r_min) > lateral_limit:
        fire = True
    return fire, margin, True


def s_join(r: float, e: float, y: float) -> float:
    """Path length a forward turn of radius ``r`` needs, from heading error ``e`` and offset ``y`` (local
    frame), to rejoin the path heading on the path: an arc back to heading 0, then an S-curve over the
    remaining offset (p2_kin_kturn.fwd_join, first element)."""
    best = None
    ds = (+1, -1) if abs(e) > math.pi - 0.1 else ((-1,) if e > 0 else (+1,))
    for d in ds:
        cx = -d * r * math.sin(e)
        cy = y + d * r * math.cos(e)
        yf = cy - d * r
        a = abs(yf)
        if a < 1e-9:
            xs = 0.0
        elif a <= 2 * r:
            xs = 2 * r * math.sin(math.acos(1 - a / (2 * r)))
        else:
            xs = 2 * r
        c = cx + xs
        if best is None or c < best:
            best = c
    return best


def arc(x, y, p, g, d, phi, r):
    """End pose of an arc of angle ``phi`` (>= 0) at radius ``r`` in gear ``g`` (+1/-1) with yaw-rate sign
    ``d`` (+1 = counter-clockwise) from ``(x, y, p)``."""
    s = d * g
    cx = x - s * r * math.sin(p)
    cy = y + s * r * math.cos(p)
    p2 = p + d * phi
    return cx + s * r * math.sin(p2), cy - s * r * math.cos(p2), p2


class _Search:
    """One planning run, advanced ``per_call`` nodes at a time (:meth:`advance`)."""

    def __init__(self, gen):
        self._gen = gen
        self.done = False
        self.result = None

    def advance(self) -> bool:
        if not self.done:
            try:
                next(self._gen)
            except StopIteration as s:
                self.done = True
                self.result = s.value
        return self.done


class KTurnPlanner:
    """Deterministic K-turn planner in the episode's local frame (origin = the projection point at the
    K-turn start, x' along the path, y' left; yaw relative to the path yaw there).

    Iterative deepening over the plan's leg count n = 1 .. legs_left, both turn directions COMPLETELY per
    level (a level the node cap interrupts is discarded whole, so a mirrored input gives the mirrored plan);
    within a plan the legs alternate gear and the last leg is forward (n odd starts forward, n even starts in
    reverse); a level whose ``n // 2`` reverse legs exceed ``rev_left`` is skipped — the leg budgets (C1)
    are the caller's, carried across re-plans. Forward legs ``phi`` in 15..180 deg, reverse legs
    ``r_plan * phi <= max_reverse`` (grid 15 deg). Region: every arc sample has x' <= s_rem0, and with a
    lateral limit its distance to the path window polyline is <= the limit. Goal after the last leg:
    ``|e| <= theta_goal``, ``|y'| <= KTURN_Y_TOL``, ``s_join + lookahead <= s_rem0 - x'``. Cost: length +
    R_min per gear change + KTURN_W_Y * max|y'|; ties: first turn direction ``-sign(e)`` (e == 0:
    ``-sign(y')``; both 0: +1), then the phi sequence. At most ``KTURN_NODE_CAP`` nodes per plan and
    ``per_call`` per :meth:`_Search.advance` call."""

    def __init__(self, s_rem0, r_plan, r_min, max_reverse, theta_goal, lookahead,
                 lateral_limit=None, poly=None, node_cap=KTURN_NODE_CAP, per_call=KTURN_NODES_PER_CALL):
        self.s_rem0 = float(s_rem0)
        self.r = float(r_plan)
        self.r_min = float(r_min)
        self.max_reverse = float(max_reverse)
        self.theta_goal = float(theta_goal)
        self.lookahead = float(lookahead)
        self.lat = None if lateral_limit is None else float(lateral_limit)
        self.node_cap = int(node_cap)
        self.per_call = int(per_call)
        g = math.radians(KTURN_GRID_DEG)
        self.fgrid = [math.radians(a) for a in range(KTURN_GRID_DEG, 181, KTURN_GRID_DEG)]
        nrev = int(math.floor(self.max_reverse / self.r / g + 1e-9))
        self.rgrid = [math.radians(KTURN_GRID_DEG * k) for k in range(1, min(nrev, 180 // KTURN_GRID_DEG) + 1)]
        self._poly = None
        if self.lat is not None:
            import numpy as np
            p = np.asarray(poly, float).reshape(-1, 2) if poly is not None else np.zeros((0, 2))
            self._poly = (p[:-1], p[1:] - p[:-1]) if len(p) >= 2 else None
        self.nodes = 0

    # -- geometry -------------------------------------------------------------------------------
    def _cte_max(self, pts):
        """Largest distance of the points ``pts`` to the window polyline (inf without one)."""
        import numpy as np
        if self._poly is None:
            return float("inf")
        a, d = self._poly
        q = np.asarray(pts, float)[:, None, :]
        l2 = (d * d).sum(-1)
        t = np.clip(((q - a) * d).sum(-1) / np.where(l2 > 1e-18, l2, 1.0), 0.0, 1.0)
        c = a + t[..., None] * d
        return float(np.sqrt(((q - c) ** 2).sum(-1)).min(1).max())

    def _samples(self, x, y, p, g, d, phi):
        return [arc(x, y, p, g, d, phi * f, self.r)[:2] for f in _REGION_F]

    def region(self, x, y, p, g, d, phi) -> bool:
        pts = self._samples(x, y, p, g, d, phi)
        for xx, _ in pts:
            if xx > self.s_rem0:
                return False
        if self.lat is not None and self._cte_max(pts) > self.lat:
            return False
        return True

    def goal(self, x, y, p) -> bool:
        e = wrap(p)
        if abs(e) > self.theta_goal or abs(y) > KTURN_Y_TOL:
            return False
        return s_join(self.r, e, y) + self.lookahead <= self.s_rem0 - x

    def remaining_ok(self, x, y, p, legs) -> bool:
        """Re-plan step (i): the remaining legs, from the ACTUAL pose, stay in the region and end in the
        goal."""
        for (g, d, phi) in legs:
            if not self.region(x, y, p, g, d, phi):
                return False
            x, y, p = arc(x, y, p, g, d, phi, self.r)
        return self.goal(x, y, p)

    def lateral_violated(self, x, y, p, legs) -> bool:
        """N6: continuing ``legs`` open loop from the actual pose would leave the lateral limit."""
        if self.lat is None:
            return False
        for (g, d, phi) in legs:
            if self._cte_max(self._samples(x, y, p, g, d, phi)) > self.lat:
                return True
            x, y, p = arc(x, y, p, g, d, phi, self.r)
        return False

    # -- search ---------------------------------------------------------------------------------
    def start(self, x, y, p, legs_left, rev_left) -> _Search:
        self.nodes = 0
        return _Search(self._search(float(x), float(y), float(p), int(legs_left), int(rev_left)))

    def _search(self, x, y, p, legs_left, rev_left):
        e0 = wrap(p)
        self.nodes = 0
        call = 0
        pref = -1 if (e0 > 0 or (e0 == 0 and y > 0)) else 1
        best = None
        capped = False
        for n in range(1, legs_left + 1):
            if n // 2 > rev_left:
                continue
            g0 = 1 if n % 2 == 1 else -1
            lvl = None
            for d in (pref, -pref):
                stack = [(x, y, p, g0, (), 0.0, abs(y))]
                while stack and not capped:
                    cx, cy, cp, g, legs, cost, ymax = stack.pop()
                    if len(legs) == n:
                        if self.goal(cx, cy, cp):
                            key = (round(cost + KTURN_W_Y * ymax, 9), legs[0][1] != pref,
                                   [round(l[2], 9) for l in legs])
                            if lvl is None or key < lvl[0]:
                                lvl = (key, list(legs))
                        continue
                    for phi in (self.fgrid if g > 0 else self.rgrid):
                        if call >= self.per_call and self.nodes < self.node_cap:
                            yield None
                            call = 0
                        if self.nodes >= self.node_cap:
                            capped = True
                            break
                        self.nodes += 1
                        call += 1
                        if not self.region(cx, cy, cp, g, d, phi):
                            continue
                        nx, ny, npp = arc(cx, cy, cp, g, d, phi, self.r)
                        ym = max(ymax, max(abs(arc(cx, cy, cp, g, d, phi * f, self.r)[1]) for f in _YMAX_F))
                        stack.append((nx, ny, npp, -g, legs + ((g, d, phi),),
                                      cost + self.r * phi + (self.r_min if legs else 0.0), ym))
                if capped:
                    break
            if capped:
                break          # the interrupted level is discarded whole
            if lvl is not None and (best is None or lvl[0] < best[0]):
                best = lvl
        return None if best is None else best[1]
