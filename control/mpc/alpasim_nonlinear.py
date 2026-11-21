# SPDX-License-Identifier: Apache-2.0
# Derived from the field port (PathFollower_MPC/mpc/alpasim_nmpc.py) of
# NVlabs/alpasim nonlinear_mpc.py (Copyright (c) 2026 NVIDIA Corporation,
# Apache-2.0); the upstream revision was not recorded by the port.
# Modified by the GenesisVehicle contributors (see NOTICE).
"""Nonlinear MPC (do_mpc + casadi, solved by IPOPT) — the field solver.

Problem (unchanged from the field port, ``alpasim_nmpc.py:116-203``): the
bicycle model of :mod:`.models`, radau collocation degree 2, horizon N = 20 at
0.1 s, ``ipopt.max_iter`` 30; tracking cost (long 2 / lat 1 / heading 1 /
acceleration state 0.1) only from node ``idx_start_penalty`` (10) on, the same
term as terminal cost; do_mpc ``rterm`` (command CHANGE) steering 5.0 /
acceleration 1.0; state/input bounds of the field port including the per-speed
acceleration limits as nonlinear constraints.

Changes from the field port (each one named, none changes the NLP):

* the do_mpc problem is set up in the CONSTRUCTOR, with a placeholder
  reference (the field set it up lazily at the first ``compute``); the first
  :meth:`solve` therefore builds no NLP;
* casadi's NumPy mode ``-1`` is set only around model build, ``setup()`` and
  ``make_step()`` and restored afterwards (the field set it for the whole
  process at import); ``import do_mpc`` silences only its own two feature
  warnings (the field installed process-wide filters);
* the do_mpc history is cleared after every solve (``reset_history``):
  do_mpc appends ~9 arrays per ``make_step`` with a full copy each time. This
  also resets do_mpc's internal clock ``_t0``, which is harmless ONLY because
  ``_tvp_fun`` ignores its time argument and the model has no ``_p``
  parameters — keep both true or drop the reset;
* :meth:`solve` returns a :class:`SolveResult` (the IPOPT ``return_status``,
  ``success``, ``iter_count``, the wall time measured with ``perf_counter``
  around ``make_step``, or the exception) instead of the field's
  ``(u, status-string)``; a ``make_step`` exception is caught, logged at debug
  and reported, the field returned zeros;
* opt-in solve-time options (default off): ``shift_warm_start``,
  ``openblas_threads``, extra ``nlpsol_opts``.

Not thread-safe: casadi's NumPy mode and the OpenBLAS thread count are
process-global.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ._deps import casadi_numpy_mode, casadi_openblas_limit, import_do_mpc, require_mpc_deps
from .models import build_bicycle
from .params import MPCGains, _AlpaSimParams

_log = logging.getLogger(__name__)

#: The field solver options (``alpasim_nmpc.py:126``).
FIELD_NLPSOL_OPTS = {"ipopt.max_iter": 30}


@dataclass
class SolveResult:
    """One solver call. ``u`` is ``None`` when the call raised."""
    u: Optional[np.ndarray]
    return_status: Optional[str] = None
    success: bool = False
    iter_count: Optional[int] = None
    solve_ms: float = 0.0
    exception: Optional[str] = None
    extra: dict = field(default_factory=dict)


class NonlinearMPC:
    """The field ``NonlinearMPC`` with the problem built at construction.

    Args:
        params: solver-side vehicle parameters (``_AlpaSimParams``, from
            :meth:`MPCVehicleParams.solver_params`).
        gains: :class:`MPCGains` (AlpaSim defaults if ``None``).
        n_horizon / dt_mpc: horizon length and node spacing (20, 0.1 s).
        yaw_bound: heading bound in the horizon frame (field default 0.78 rad;
            the follower passes pi/2 like the field ``MPCFollower``).
        nlpsol_opts: extra casadi ``nlpsol`` options merged OVER the field's
            ``{"ipopt.max_iter": 30}``. Unknown IPOPT options raise at setup.
        shift_warm_start: before each solve, shift the previous primal solution
            one node (0.1 s) earlier; do_mpc's default warm start reuses it
            unshifted. Changes results (not the field behaviour).
        openblas_threads: cap casadi's bundled OpenBLAS (used by IPOPT's MUMPS)
            at this many threads during ``make_step`` and restore it after;
            ``None`` = leave it. No effect where the library cannot be reached
            (see :func:`._deps.casadi_openblas`).
        reset_history: clear do_mpc's per-solve history after every solve.
    """

    def __init__(self, params: Optional[_AlpaSimParams] = None,
                 gains: Optional[MPCGains] = None, n_horizon: int = 20,
                 dt_mpc: float = 0.1, yaw_bound: float = 0.78, *,
                 nlpsol_opts: Optional[dict] = None, shift_warm_start: bool = False,
                 openblas_threads: Optional[int] = None, reset_history: bool = True,
                 model_builder=build_bicycle):
        require_mpc_deps("nonlinear")
        self.yaw_bound = float(yaw_bound)
        self.p = params or _AlpaSimParams()
        self.g = gains or MPCGains()
        self.N, self.dt = int(n_horizon), float(dt_mpc)
        self.nlpsol_opts = dict(FIELD_NLPSOL_OPTS)
        self.nlpsol_opts.update(nlpsol_opts or {})
        self.shift_warm_start = bool(shift_warm_start)
        self.openblas_threads = openblas_threads
        self.reset_history = bool(reset_history)
        self._ref = np.zeros((self.N + 1, 3))        # placeholder until solve()
        t0 = time.perf_counter()
        with casadi_numpy_mode(-1):
            self._model = model_builder(self.p)
            self._setup_mpc()
        self.setup_ms = (time.perf_counter() - t0) * 1e3
        self._index_maps()

    # -- problem (verbatim from the field port) ---------------------------------

    def _setup_mpc(self):
        import casadi
        do_mpc = import_do_mpc()
        self._mpc = do_mpc.controller.MPC(self._model)
        self._mpc.settings.n_horizon = self.N
        self._mpc.settings.t_step = self.dt
        self._mpc.settings.n_robust = 0
        self._mpc.settings.open_loop = 0
        self._mpc.settings.state_discretization = "collocation"
        self._mpc.settings.collocation_type = "radau"
        self._mpc.settings.collocation_deg = 2
        self._mpc.settings.collocation_ni = 1
        self._mpc.settings.store_full_solution = False
        self._mpc.settings.nlpsol_opts = dict(self.nlpsol_opts)
        self._mpc.settings.supress_ipopt_output()

        g, m = self.g, self._model
        term = m.tvp["tracking_enabled"] * (
            g.long_position_weight * (m.x["x_rig_inertial"] - m.tvp["x_ref"]) ** 2
            + g.lat_position_weight * (m.x["y_rig_inertial"] - m.tvp["y_ref"]) ** 2
            + g.acceleration_weight * (m.x["acceleration"] ** 2)
            + g.heading_weight * casadi.atan2(casadi.sin(m.x["yaw_angle"] - m.tvp["heading_ref"]),
                                              casadi.cos(m.x["yaw_angle"] - m.tvp["heading_ref"])) ** 2)
        self._mpc.set_objective(mterm=term, lterm=term)
        self._mpc.set_rterm(front_steering_angle_cmd=g.rel_front_steering_angle_weight,
                            acceleration_cmd=g.rel_acceleration_weight)
        self._mpc.set_tvp_fun(self._tvp_fun)

        b = self._mpc.bounds
        b["lower", "_x", "x_rig_inertial"] = -500
        b["upper", "_x", "x_rig_inertial"] = 500
        b["lower", "_x", "y_rig_inertial"] = -20
        b["upper", "_x", "y_rig_inertial"] = 20
        b["lower", "_x", "yaw_angle"] = -self.yaw_bound
        b["upper", "_x", "yaw_angle"] = self.yaw_bound
        b["lower", "_x", "v_cg_x"] = 0.0
        b["upper", "_x", "v_cg_x"] = 35
        b["lower", "_x", "v_cg_y"] = -10
        b["upper", "_x", "v_cg_y"] = 10
        b["lower", "_x", "yaw_rate"] = -3
        b["upper", "_x", "yaw_rate"] = 3
        b["lower", "_x", "front_steering_angle"] = -0.785
        b["upper", "_x", "front_steering_angle"] = 0.785
        b["lower", "_u", "front_steering_angle_cmd"] = -2
        b["upper", "_u", "front_steering_angle_cmd"] = 2
        b["lower", "_u", "acceleration_cmd"] = -9.0
        b["upper", "_u", "acceleration_cmd"] = 6.0
        p = self.p
        if getattr(p, "steer_max", None) is not None:
            b["lower", "_x", "front_steering_angle"] = -p.steer_max
            b["upper", "_x", "front_steering_angle"] = p.steer_max
            b["lower", "_u", "front_steering_angle_cmd"] = -p.steer_max
            b["upper", "_u", "front_steering_angle_cmd"] = p.steer_max
        if getattr(p, "accel_max", None) is not None:
            b["upper", "_x", "acceleration"] = p.accel_max
            b["upper", "_u", "acceleration_cmd"] = p.accel_max
        if getattr(p, "accel_min", None) is not None:
            b["lower", "_x", "acceleration"] = p.accel_min
            b["lower", "_u", "acceleration_cmd"] = p.accel_min
        # Per-speed drive limits as nonlinear constraints on every horizon node:
        # piecewise-linear interpolation, clamped at both ends, as a casadi expression.
        vx, acmd = m.x["v_cg_x"], m.u["acceleration_cmd"]
        for name, tab, sgn in (("accel_hi", getattr(p, "accel_max_tab", None), 1.0),
                               ("accel_lo", getattr(p, "accel_min_tab", None), -1.0)):
            if not tab:
                continue
            vs, ys = [float(v) for v in tab[0]], [float(y) for y in tab[1]]
            lim = ys[0]
            for i in range(1, len(vs)):
                lim = lim + (ys[i] - ys[i - 1]) / (vs[i] - vs[i - 1]) * casadi.fmin(casadi.fmax(vx - vs[i - 1], 0.0), vs[i] - vs[i - 1])
            self._mpc.set_nl_cons(name, sgn * (acmd - lim), ub=0.0)
            if sgn > 0:
                b["upper", "_u", "acceleration_cmd"] = max(ys)
                b["upper", "_x", "acceleration"] = max(ys)
            else:
                b["lower", "_u", "acceleration_cmd"] = min(ys)
                b["lower", "_x", "acceleration"] = min(ys)

        self._mpc.setup()
        self._mpc.x0 = np.zeros((8, 1))
        self._mpc.set_initial_guess()

    def _tvp_fun(self, _t_now):
        # Ignores its time argument (reset_history relies on this).
        tpl = self._mpc.get_tvp_template()
        for k in range(self.N + 1):
            tpl["_tvp", k, "x_ref"] = self._ref[k, 0]
            tpl["_tvp", k, "y_ref"] = self._ref[k, 1]
            tpl["_tvp", k, "heading_ref"] = self._ref[k, 2]
            tpl["_tvp", k, "tracking_enabled"] = 1.0 if k >= self.g.idx_start_penalty else 0.0
        return tpl

    # -- flat indices into the decision vector (computed once) -------------------

    def _index_maps(self):
        f = self._mpc.opt_x.f
        #: index of the state at the END of node k's interval (time k*dt), k = 0..N
        self._plan_idx = np.array([f["_x", k, 0, -1] for k in range(self.N + 1)], dtype=np.int64)
        self._x_blocks = [np.asarray(f["_x", k, 0], dtype=np.int64) for k in range(self.N + 1)]
        self._u_blocks = [np.asarray(f["_u", k, 0], dtype=np.int64) for k in range(self.N)]

    # -- runtime ------------------------------------------------------------------

    @property
    def u_prev(self) -> np.ndarray:
        """do_mpc's previous input ``_u_prev`` — a COST PARAMETER (the rterm
        penalises the change from it), not an initial guess."""
        return np.array(self._mpc.u0.cat).ravel()

    def set_u_prev(self, delta: float, accel: float) -> None:
        """Set do_mpc's ``_u_prev`` (``MPC.u0``); must be an ndarray (a list
        raises AttributeError inside do_mpc)."""
        self._mpc.u0 = np.array([float(delta), float(accel)])

    def _shift(self):
        x = np.array(self._mpc.opt_x_num.cat, dtype=float).ravel()
        n_x = 8
        for k in range(self.N):
            dst, src = self._x_blocks[k], self._x_blocks[k + 1]
            if len(dst) == len(src):
                x[dst] = x[src]
            else:                                    # different point counts: last point
                x[dst] = np.tile(x[src[-n_x:]], len(dst) // n_x)
        for k in range(self.N - 1):
            x[self._u_blocks[k]] = x[self._u_blocks[k + 1]]
        import casadi
        self._mpc.opt_x_num.master = casadi.DM(x)

    def solve(self, x0, ref_xyyaw) -> SolveResult:
        """``x0`` (8,) in the rig frame at solve time, ``ref_xyyaw`` (N+1, 3)
        reference in the same frame at 0.1 s spacing from now."""
        self._ref = np.asarray(ref_xyyaw, float)
        if self.shift_warm_start:
            self._shift()
        t0 = time.perf_counter()
        try:
            with casadi_numpy_mode(-1), casadi_openblas_limit(self.openblas_threads):
                u = self._mpc.make_step(np.asarray(x0, float).reshape(-1, 1))
        except Exception as e:  # noqa: BLE001 — classified EXC by the law (plan 3.6)
            ms = (time.perf_counter() - t0) * 1e3
            _log.debug("NonlinearMPC.make_step raised", exc_info=True)
            return SolveResult(u=None, solve_ms=ms, exception=f"{type(e).__name__}: {e}")
        ms = (time.perf_counter() - t0) * 1e3
        s = self._mpc.solver_stats
        if self.reset_history:
            self._mpc.reset_history()
        return SolveResult(u=np.asarray(u, float).flatten(), return_status=s.get("return_status"),
                           success=bool(s.get("success")), iter_count=s.get("iter_count"),
                           solve_ms=ms)

    def max_violation(self) -> float:
        """Largest bound/constraint violation of the LAST returned iterate, in
        the NLP's own (scaled) space — the space IPOPT's ``constr_viol_tol``
        refers to: ``opt_x_num`` against ``_lb_opt_x/_ub_opt_x`` and
        ``opt_g_num`` against ``nlp_cons_lb/ub`` (do_mpc stores the bounds
        already divided by the scaling).

        Any non-finite entry in the iterate ``x`` or the constraint values
        ``g`` returns ``inf`` (infeasible): a NaN can never read as feasible.
        """
        m = self._mpc
        x = np.array(m.opt_x_num.cat, dtype=float).ravel()
        lbx = np.array(m._lb_opt_x.cat, dtype=float).ravel()
        ubx = np.array(m._ub_opt_x.cat, dtype=float).ravel()
        g = np.array(m.opt_g_num, dtype=float).ravel()
        lbg = np.array(m.nlp_cons_lb, dtype=float).ravel()
        ubg = np.array(m.nlp_cons_ub, dtype=float).ravel()
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(g))):
            return float("inf")
        v = 0.0
        if x.size:
            v = max(v, float(np.max(np.maximum(lbx - x, x - ubx))))
        if g.size:
            v = max(v, float(np.max(np.maximum(lbg - g, g - ubg))))
        return v if math.isfinite(v) else float("inf")

    def plan_vector(self) -> np.ndarray:
        """Copy of the LAST solution's states at the node times (N+1, 8),
        unscaled (``opt_x_num_unscaled``); node 0 is the ``x0`` it was given."""
        v = np.array(self._mpc.opt_x_num_unscaled.cat, dtype=float).ravel()
        return v[self._plan_idx]
