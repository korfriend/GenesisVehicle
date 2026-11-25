# SPDX-License-Identifier: Apache-2.0
# Derived from the field controller PathFollower_MPC/mpc/mpc_follower.py:105-184
# (the follower around the NVlabs/alpasim MPC; its d0 rule is AlpaSim's
# coerce_dynamic_state). Modified by the GenesisVehicle contributors (see NOTICE).
"""The MPC layer's control law (``_MPCLaw``) — independent of Genesis and of
``PathFollower``.

Per forward block of a path it keeps a time-stamped reference of the rig
(rear-axle centre), re-solves the MPC every ``DT_MPC`` (0.1 s), holds the
solution's ``(delta_cmd, a_cmd)`` in between and turns it into the stage-2
target the inverse plant consumes (feedforward ``"field"``)::

    a_t = clip(a_cmd, accel_lo(v), accel_hi(v))
    w_t = max(v, 0) * tan(delta_cmd) / L

all as the field ``MPCFollower`` did (``mpc_follower.py:162-174``);
``accel_lo`` / ``accel_hi`` are read from ``solver.p``, which with the v2.1.0
decel gate (below) is the parameter set of the solver the LAST solve used, so
a held command between solves is clipped by that solve's table. What it adds:

* every solve is classified (plan section 3.6, below) and counted in
  :attr:`mpc_stats`; what is applied depends on ``on_solve_failure``;
* do_mpc's ``_u_prev`` — a cost parameter of the command-change penalty — is
  reset to the vehicle's measured ``(d0, a0)`` before the first solve after a
  restart: a block entry (including the path start), the first solve after a
  failed one under ``"pd"``, and the first solve after :meth:`start_block` is
  called again (``set_path``). Inside a block the solver keeps its own value
  (the field behaviour). At a start from rest ``(d0, a0) == (0, 0)``, do_mpc's
  initial value, so the first solve is unchanged against the field;
* the predicted state at the node times of the last applied solve
  (:attr:`last_plan`; NMPC only — ``None`` with the LMPC);
* (v2.1.0, ``decel_gate=True``, used by ``PathFollower(mpc_decel_limit=
  "brake")``) a per-solve choice of the deceleration bound: the solver's
  BRAKE-row problem is used on a solve when the active forward block's
  reference end time is within ``N * DT_MPC`` (2.0 s at N = 20) of the solve
  time IN REFERENCE TIME — ``t + DT_MPC * N >= t_end - 1e-9``, ``t_end`` =
  the last time of the block's reference, so a vehicle behind its schedule
  gets it earlier and farther from the end — AND the rig reference's end
  point lies ahead of the rear axle (its body-``x`` coordinate in the solve
  frame is ``> 0``); on every other solve the THROTTLE-row problem (the
  default's). Mid-path reference speed drops, a profile that ramps down
  before its last ``N * DT_MPC`` of reference time, and every solve after the
  end point has passed behind the rear axle get the throttle row. Anything
  that re-times the reference (block entry, :meth:`start_block`: the heading
  gate, the watchdog, the K-turn release and ``set_path`` through the
  follower) moves the opening. The choice is re-evaluated on every solve and
  has no other state; counters in :meth:`decel_gate_stats`. Equality with the
  default before the first open solve is measured, not guaranteed.

| abbr | meaning |
|---|---|
| NMPC / LMPC | nonlinear MPC (do_mpc + IPOPT) / linear MPC (OSQP) |
| CONV | the solver converged |
| CAP | stopped by an iteration or time cap |
| INFEAS | the solver reported the problem infeasible or failed numerically |
| OTHER / EXC / NAN | any other status / the solver raised / a non-finite ``u`` |
| d0, a0 | measured front steering angle (AlpaSim rule) and acceleration of the vehicle |

Solve classification (NMPC, IPOPT ``return_status``; order: exception, then
non-finite ``u``, then the status):

======== ================================================ ============== =================
class    statuses                                         ``"pd"``       ``"hold"``
======== ================================================ ============== =================
CONV     Solve_Succeeded, Solved_To_Acceptable_Level,     apply          apply
         Feasible_Point_Found
CAP      Maximum_Iterations_Exceeded, Maximum_CpuTime_    apply if the   apply
         Exceeded, Maximum_WallTime_Exceeded,             iterate is
         Search_Direction_Becomes_Too_Small               feasible, else
                                                          fall back
INFEAS   Infeasible_Problem_Detected, Restoration_Failed, fall back      apply
         Error_In_Step_Computation, Invalid_Number_
         Detected, Diverging_Iterates,
         Not_Enough_Degrees_Of_Freedom
OTHER    any other status                                 fall back      apply
EXC      the solver raised                                fall back      keep previous
NAN      ``u`` not finite (any status)                    fall back      keep previous
======== ================================================ ============== =================

"feasible" = the returned iterate's largest bound/constraint violation
<= 1e-4 (IPOPT's default ``constr_viol_tol``), measured in the NLP's own
(scaled) space. "fall back" = the law returns mode ``"PD-FALLBACK"`` and no
target; the caller runs the PD law until the next applicable solve.

``"hold"`` is the parity mode with the field controller. Two things to know
about it: (1) it APPLIES INFEASIBLE SOLUTIONS — every solution that did not
raise is applied, INFEAS, OTHER and CAP-infeasible included, exactly as the
field did; (2) a non-finite ``u`` KEEPS THE PREVIOUS COMMAND, which is where
it differs from the field (the field applied a non-finite ``u`` whenever the
solver returned a status).

LMPC (OSQP, ``osqp.SolverStatus``): CONV = ``OSQP_SOLVED``; CAP =
``OSQP_SOLVED_INACCURATE``, ``OSQP_MAX_ITER_REACHED``,
``OSQP_TIME_LIMIT_REACHED`` (``"pd"``: apply if feasible, violation of
``l <= A x <= u`` <= 1e-4); INFEAS = the four (in)feasibility statuses and
``OSQP_NON_CVX``; OTHER = anything else. ``"hold"`` applies ``OSQP_SOLVED``
and ``OSQP_SOLVED_INACCURATE`` only and keeps the previous command otherwise
— the field behaviour (``alpasim_mpc.py:215-216`` + ``mpc_follower.py:156``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .params import accel_limits

#: Solve period = horizon node spacing [s] (AlpaSim default; field DT_MPC).
DT_MPC = 0.1
#: Speed treated as standing still [m/s] (field HOLD_V; PathFollower's
#: brake-at-rest threshold).
HOLD_V = 0.1
#: Largest bound/constraint violation still called feasible (IPOPT default
#: ``constr_viol_tol``).
FEAS_TOL = 1e-4
#: Field defaults where the params give no acceleration limit.
ACCEL_LO_DEFAULT, ACCEL_HI_DEFAULT = -9.0, 6.0
#: (v2.1.0) Tolerance of the decel gate's reference-time test [s].
DECEL_GATE_TOL = 1e-9

IPOPT_CONV = frozenset({"Solve_Succeeded", "Solved_To_Acceptable_Level", "Feasible_Point_Found"})
IPOPT_CAP = frozenset({"Maximum_Iterations_Exceeded", "Maximum_CpuTime_Exceeded",
                       "Maximum_WallTime_Exceeded", "Search_Direction_Becomes_Too_Small"})
IPOPT_INFEAS = frozenset({"Infeasible_Problem_Detected", "Restoration_Failed",
                          "Error_In_Step_Computation", "Invalid_Number_Detected",
                          "Diverging_Iterates", "Not_Enough_Degrees_Of_Freedom"})

OSQP_CONV = frozenset({1})             # OSQP_SOLVED
OSQP_CAP = frozenset({2, 7, 8})        # SOLVED_INACCURATE, MAX_ITER_REACHED, TIME_LIMIT_REACHED
OSQP_INFEAS = frozenset({3, 4, 5, 6, 9})  # (DUAL|PRIMAL)_INFEASIBLE(_INACCURATE), NON_CVX
OSQP_HOLD_APPLY = frozenset({1, 2})    # field: "solved", "solved inaccurate"

POLICIES = ("pd", "hold")


@dataclass
class LawOutput:
    """One law step. ``a_t``/``w_t`` are ``None`` in mode ``"PD-FALLBACK"``."""
    a_t: Optional[float]
    w_t: Optional[float]
    mode: str                  # "MPC" | "MPC-HOLDCMD" | "PD-FALLBACK"
    solved: bool               # a solve ran on this step
    status: Optional[str]      # the solve's class key (mpc_stats) when solved


def classify_solve(res, solver, impl: str = "nonlinear"):
    """``(class, key, feasible)`` for a :class:`SolveResult` — class in
    CONV/CAP/INFEAS/OTHER/EXC/NAN, key = the ``mpc_stats["status"]`` key."""
    if res.exception is not None or res.u is None:
        return "EXC", "exception", False
    if not np.all(np.isfinite(np.asarray(res.u, float))):
        return "NAN", "nan", False
    if impl == "linear":
        sv = int(res.extra.get("status_val", -1))
        if sv in OSQP_CONV:
            return "CONV", "converged", True
        if sv in OSQP_CAP:
            feas = solver.max_violation() <= FEAS_TOL
            return "CAP", ("capped" if feas else "capped_infeasible"), feas
        return ("INFEAS" if sv in OSQP_INFEAS else "OTHER"), f"failed:{res.return_status}", False
    st = res.return_status
    if st in IPOPT_CONV:
        return "CONV", "converged", True
    if st in IPOPT_CAP:
        feas = solver.max_violation() <= FEAS_TOL
        return "CAP", ("capped" if feas else "capped_infeasible"), feas
    return ("INFEAS" if st in IPOPT_INFEAS else "OTHER"), f"failed:{st}", False


def applies(cls: str, feasible: bool, policy: str, res=None, impl: str = "nonlinear") -> bool:
    """Whether a classified solve's ``u`` becomes the command (table above)."""
    if policy == "pd":
        return cls == "CONV" or (cls == "CAP" and feasible)
    if cls in ("EXC", "NAN"):
        return False
    if impl == "linear":
        return int(res.extra.get("status_val", -1)) in OSQP_HOLD_APPLY
    return True


def rig_reference(waypoints, rig_offset: float, v_floor: float, t0: float = 0.0):
    """Time-stamped rig reference of a forward block (``mpc_follower.py:105-118``).

    ``waypoints`` = ``(x, y, z, v[, yaw])`` base_link points; speeds below
    ``v_floor`` are timed at ``v_floor``; the heading is the explicit 5th
    element when the FIRST waypoint carries one, else the unwrapped path
    tangent. Returns ``(t, x_rig, y_rig, heading)`` with ``t[0] == t0``.
    """
    P = np.asarray([p[:4] for p in waypoints], float)
    if (P[:, 3] < 0).any():
        raise ValueError("the MPC reference takes a forward block only (target speeds >= 0)")
    x, y, v = P[:, 0], P[:, 1], np.maximum(P[:, 3], v_floor)
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    t = np.r_[0.0, np.cumsum(np.diff(s) / (0.5 * (v[1:] + v[:-1])))]
    if t0 != 0.0:
        t = t + t0
    first = waypoints[0]
    if len(first) > 4 and first[4] is not None:
        h = np.unwrap([p[4] for p in waypoints])
    else:
        h = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    rig = float(rig_offset)
    return t, x - rig * np.cos(h), y - rig * np.sin(h), h


def finish_command(v, a_ff, throttle, steer, steer_cap):
    """The field finish of an MPC step (``mpc_follower.py:176-183``):
    ``(throttle, steer, brake, mode)`` — caps, brake at rest, and MPC-HOLD
    (standing or rolling back while the plan wants to slow: brake instead of
    a negative throttle, which at rest would drive backwards). MPC-HOLD looks
    at ``a_ff``, the MPC's own target before any correction."""
    thr = max(-1.0, min(1.0, float(throttle)))
    st = max(-steer_cap, min(steer_cap, float(steer)))
    brake = 1.0 if abs(thr) < 0.02 and abs(v) < HOLD_V else 0.0
    mode = "MPC"
    if v < -HOLD_V or (v < HOLD_V and a_ff <= 0.0):
        thr, brake = 0.0, 1.0
        mode = "MPC-HOLD"
    return thr, st, brake, mode


class _MPCLaw:
    """Stage 1 of the MPC layer for one vehicle.

    Args:
        solver: a :class:`.alpasim_nonlinear.NonlinearMPC` or
            :class:`.alpasim_linear.LinearMPC` (anything with ``solve``,
            ``set_u_prev``, ``max_violation``, ``N``, ``p``); with
            ``decel_gate=True`` also a callable ``select_decel(open)`` (the
            v2.1.0 ``_DecelGatedNMPC`` / ``_DecelGatedLMPC``), whose ``p`` is
            the active problem's. ``p.wheelbase`` and ``N`` are read ONCE here.
        rig_offset: base_link -> rear-axle centre distance [m, positive].
        on_solve_failure: ``"pd"`` (fall back) or ``"hold"`` (field parity).
        v_floor: speed floor for timing the reference [m/s] (field 0.5).
        impl: ``"nonlinear"`` | ``"linear"`` (selects the status table).
        decel_gate: (v2.1.0) False (default): no attribute of the solver
            beyond the list above is looked up and nothing below runs. True:
            ``TypeError`` here unless ``solver.select_decel`` is callable, and
            every solve calls it with the gate predicate (module docstring)
            AFTER the reference is sampled and BEFORE ``x0`` / ``u_prev`` are
            set; an exception from it propagates (no fall-back).
    """

    def __init__(self, solver, *, rig_offset: float, on_solve_failure: str = "pd",
                 v_floor: float = 0.5, impl: str = "nonlinear", decel_gate: bool = False):
        if on_solve_failure not in POLICIES:
            raise ValueError(f"on_solve_failure must be one of {POLICIES}; got {on_solve_failure!r}")
        if decel_gate and not callable(getattr(solver, "select_decel", None)):
            raise TypeError("decel_gate=True needs a solver with a callable select_decel(open) "
                            f"(a decel-gated solver pair); got {type(solver).__name__}")
        self.solver = solver
        self.policy = on_solve_failure
        self.impl = impl
        self.rig = float(rig_offset)
        self.v_floor = float(v_floor)
        self.L = float(solver.p.wheelbase)
        self.N = int(solver.N)
        self.ref = None
        self.goal = None
        self.n_solve = 0
        self.status = {}
        self.last_status = None
        self.last_solve_ms = None
        self._plan = None
        # v2.1.0 decel gate: counters are cumulative since construction
        # (start_block / set_path keep them); _decel_open is None before the
        # first solve, then the last solve's choice.
        self.decel_gate = bool(decel_gate)
        self._decel_open = None
        self._decel_open_solves = 0
        self._decel_switches = 0
        self._reset_command()

    # -- block lifecycle -----------------------------------------------------------

    def _reset_command(self):
        self.cmd_t = None              # time of the last solve (None: solve now)
        self.cmd_delta = 0.0
        self.cmd_a = 0.0
        self.vhist = []
        self.mode = "MPC"
        self._restart = True           # set u_prev = (d0, a0) before the next solve

    def start_block(self, waypoints, t0: float = 0.0) -> None:
        """(Re)build the reference for a forward block timed from ``t0`` and
        restart: command, acceleration history and ``u_prev`` are reset; the
        solver's primal/dual warm start is kept."""
        self.ref = rig_reference(waypoints, self.rig, self.v_floor, t0)
        last = waypoints[-1]
        self.goal = (float(last[0]), float(last[1]))
        self._reset_command()

    def _ref_at(self, ts):
        t, X, Y, H = self.ref
        tq = np.clip(ts, t[0], t[-1])
        return np.interp(tq, t, X), np.interp(tq, t, Y), np.interp(tq, t, H)

    # -- one control step ------------------------------------------------------------

    def due(self, t: float) -> bool:
        return self.cmd_t is None or t - self.cmd_t >= DT_MPC - 1e-6

    def step(self, t, pos_xy, yaw, v_long, yaw_rate, v_lat) -> LawOutput:
        v = float(v_long)
        solved, key = False, None
        if self.due(t):
            key = self._solve(float(t), pos_xy, float(yaw), v, float(yaw_rate), float(v_lat))
            solved = True
        if self.mode == "PD-FALLBACK":
            return LawOutput(None, None, "PD-FALLBACK", solved, key)
        lo, hi = accel_limits(self.solver.p, max(v, 0.0))
        lo = ACCEL_LO_DEFAULT if lo is None else lo
        hi = ACCEL_HI_DEFAULT if hi is None else hi
        a_t = float(np.clip(self.cmd_a, lo, hi))
        w_t = max(v, 0.0) * math.tan(self.cmd_delta) / self.L
        return LawOutput(a_t, w_t, self.mode, solved, key)

    def initial_state(self, t, v, r, vy):
        """``x0`` of a solve at time ``t`` and the ``(d0, a0)`` it contains;
        appends ``(t, v)`` to the acceleration history (field ``:149-154``)."""
        self.vhist.append((t, v))
        while len(self.vhist) > 2 and self.vhist[0][0] < t - 0.1 - 1e-9:
            self.vhist.pop(0)
        a0 = (self.vhist[-1][1] - self.vhist[0][1]) / max(self.vhist[-1][0] - self.vhist[0][0], 1e-3)
        d0 = math.atan(r * self.L / v) if v > 0.25 else 0.0     # AlpaSim coerce_dynamic_state
        return np.array([0.0, 0.0, 0.0, max(v, 0.0), vy, r, d0, a0]), d0, a0

    def decel_open_at(self, ts, ref_l) -> bool:
        """(v2.1.0) The decel-gate predicate for a solve whose reference was
        sampled at ``ts`` (``t + DT_MPC * arange(N + 1)``, the times
        :meth:`_ref_at` used) giving ``ref_l`` ((N+1, 3), solve frame): the
        horizon's last node reaches the block reference's end time (reference
        time, tolerance :data:`DECEL_GATE_TOL`) AND the reference at that node
        — then the rig reference's end point — has body ``x > 0``."""
        t_end = float(self.ref[0][-1])
        return bool(float(ts[-1]) >= t_end - DECEL_GATE_TOL and float(ref_l[-1, 0]) > 0.0)

    def _select_decel(self, ts, ref_l) -> None:
        open_ = self.decel_open_at(ts, ref_l)
        self.solver.select_decel(open_)          # may raise: propagates (no fall-back)
        if open_ != bool(self._decel_open):      # the pair starts on the throttle row
            self._decel_switches += 1
        self._decel_open = open_
        if open_:
            self._decel_open_solves += 1

    def _solve(self, t, pos_xy, yaw, v, r, vy):
        c, s = math.cos(yaw), math.sin(yaw)
        ox, oy = float(pos_xy[0]) - self.rig * c, float(pos_xy[1]) - self.rig * s
        ts = t + DT_MPC * np.arange(self.N + 1)
        X, Y, H = self._ref_at(ts)
        gx, gy = X - ox, Y - oy
        ref_l = np.column_stack([c * gx + s * gy, -s * gx + c * gy,
                                 np.unwrap((H - yaw + math.pi) % (2 * math.pi) - math.pi)])
        if self.decel_gate:
            self._select_decel(ts, ref_l)
        x0, d0, a0 = self.initial_state(t, v, r, vy)
        if self._restart:
            self.solver.set_u_prev(d0, a0)
            self._restart = False
        res = self.solver.solve(x0, ref_l)
        cls, key, feas = classify_solve(res, self.solver, self.impl)
        ok = applies(cls, feas, self.policy, res, self.impl)
        if ok:
            self.cmd_delta, self.cmd_a = float(res.u[0]), float(res.u[1])
            self.mode = "MPC"
            if hasattr(self.solver, "plan_vector"):
                self._plan = (t, self.solver.plan_vector())
        elif self.policy == "pd":
            self.mode = "PD-FALLBACK"
            self._restart = True       # the next solve restarts from the measured state
        else:
            self.mode = "MPC-HOLDCMD"  # keep the previous command (field)
        self.cmd_t = t
        self.n_solve += 1
        self.status[key] = self.status.get(key, 0) + 1
        self.last_status = key
        self.last_solve_ms = res.solve_ms
        self.last_result = res
        return key

    # -- reporting -------------------------------------------------------------------

    @property
    def last_plan(self):
        """``(times (N+1,), states (N+1, 8))`` of the last APPLIED solve, in the
        rig frame at its solve time, or ``None``. A copy."""
        if self._plan is None:
            return None
        t, X = self._plan
        return t + DT_MPC * np.arange(self.N + 1), X.copy()

    def decel_gate_stats(self) -> Optional[dict]:
        """(v2.1.0) ``None`` without the decel gate; else ``{open, open_solves,
        switches}``: ``open`` = whether the LAST solve used the brake row
        (``None`` before the first solve; held between solves),
        ``open_solves`` = solves run on the brake row, including ones whose
        SOLVER CALL failed or raised (an error from the gate selection or the
        warm-start transfer propagates out of the solve before the counters
        update and is not counted — pinned by
        ``tests/test_mpc_decel_gate.py::test_a25_transfer_error_propagates_out_of_plan``),
        ``switches`` = solves whose row differs from the
        previous solve's (the pair starts on the throttle row, so a first solve
        that opens counts one). Cumulative since construction."""
        if not self.decel_gate:
            return None
        return {"open": self._decel_open, "open_solves": self._decel_open_solves,
                "switches": self._decel_switches}

    def mpc_stats(self, t_now: Optional[float] = None) -> dict:
        age = None
        if t_now is not None and self._plan is not None:
            age = float(t_now) - self._plan[0]
        return {"n_solve": self.n_solve, "status": dict(self.status),
                "last_status": self.last_status, "last_solve_ms": self.last_solve_ms,
                "plan_age_s": age}
