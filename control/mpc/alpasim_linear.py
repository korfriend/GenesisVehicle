# SPDX-License-Identifier: Apache-2.0
# Derived from the field port (PathFollower_MPC/mpc/alpasim_mpc.py:62-217) of
# NVlabs/alpasim linear_mpc.py / mpc_controller.py (Copyright (c) 2026 NVIDIA
# Corporation, Apache-2.0); the upstream revision was not recorded by the port.
# Modified by the GenesisVehicle contributors (see NOTICE).
"""Linear MPC (OSQP) — the field linear solver.

Linearisation (``_linearize_dynamics``: matrix exponential of the bicycle
model frozen at the current state), the condensed QP, its weights and
constraints are the field port's, with ``rterm="abs"`` (command MAGNITUDE
regularisation, the ``linear_mpc.py`` original) as the default.

Changes from the field port:

* the QP build and solve run inside ``threadpool_limits(blas_threads, "blas")``
  (default 1) — numpy/scipy's OpenBLAS only; casadi's own OpenBLAS is not
  involved here and threadpoolctl cannot see it;
* ``OSQP.setup`` and ``solve`` exceptions (``OSQPException`` on invalid data,
  e.g. ``l > u``) are caught and reported, not raised;
* ``solve(raise_error=False)`` is passed explicitly (osqp 1.x warns that its
  default will become ``True``, which raises on every non-solved status);
* the result carries the numeric status (``info.status_val``, compared with
  ``osqp.SolverStatus``) and the returned primal ``x``'s constraint violation,
  instead of the field's ``(u or zeros, status string)``.
* (v2.1.0) ``PathFollower(mpc_decel_limit="brake", mpc_impl="linear")``
  wraps ONE of these in the private :class:`_DecelGatedLMPC`, which swaps its
  ``p`` between the throttle-row and the brake-row parameter sets per solve;
  the brake-row set on end-gated solves only (reference time; rig reference
  end point ahead of the rear axle; see ``PathFollower``'s
  ``mpc_decel_limit``).
"""
from __future__ import annotations

import logging
import math
import time
from typing import Optional

import numpy as np

from ._deps import require_mpc_deps
from .alpasim_nonlinear import SolveResult, _check_decel_pair_params
from .params import MPCGains, _AlpaSimParams, accel_limits

_log = logging.getLogger(__name__)


class LinearMPC:
    IX, IY, IYAW, IVX, IVY, IYAW_RATE, ISTEERING, IACCEL = range(8)
    NX, NU = 8, 2
    OSQP_EPS_ABS = 1e-4
    OSQP_EPS_REL = 1e-4
    OSQP_MAX_ITER = 500
    QP_REGULARIZATION = 1e-6

    def __init__(self, params: Optional[_AlpaSimParams] = None,
                 gains: Optional[MPCGains] = None, n_horizon: int = 20,
                 dt_mpc: float = 0.1, rterm: str = "abs", *, blas_threads: Optional[int] = 1):
        require_mpc_deps("linear")
        self.p = params or _AlpaSimParams()
        self.g = gains or MPCGains()
        self.N, self.dt = int(n_horizon), float(dt_mpc)
        if rterm not in ("abs", "delta"):
            raise ValueError(f"rterm must be 'abs' or 'delta'; got {rterm!r}")
        self.rterm = rterm
        self.blas_threads = blas_threads
        self._Q = np.diag([self.g.long_position_weight, self.g.lat_position_weight, self.g.heading_weight,
                           0.0, 0.0, 0.0, 0.0, self.g.acceleration_weight])
        self._R = np.diag([self.g.rel_front_steering_angle_weight, self.g.rel_acceleration_weight])
        self._Qf = self._Q.copy()
        self._cidx = [self.IYAW, self.IVX, self.ISTEERING, self.IACCEL]
        self._xmin = np.array([-math.pi / 2, 0.0, -math.pi / 4, -8.0])
        self._xmax = np.array([math.pi / 2, 40.0, math.pi / 4, 6.0])
        self._umin = np.array([-2.0, -9.0])
        self._umax = np.array([2.0, 6.0])
        p = self.p
        if p.steer_max is not None:
            self._xmin[2], self._xmax[2] = -p.steer_max, p.steer_max
            self._umin[0], self._umax[0] = -p.steer_max, p.steer_max
        if p.accel_max is not None:
            self._xmax[3] = self._umax[1] = p.accel_max
        if p.accel_min is not None:
            self._xmin[3] = self._umin[1] = p.accel_min
        self._u_prev = np.zeros(self.NU)
        self._last_qp = None

    # -- the field interface's u_prev (only rterm="delta" reads it) ---------------

    @property
    def u_prev(self) -> np.ndarray:
        return self._u_prev.copy()

    def set_u_prev(self, delta: float, accel: float) -> None:
        self._u_prev = np.array([float(delta), float(accel)])

    def solve(self, x0, ref_xyyaw) -> SolveResult:
        x_ref = np.zeros((self.N + 1, self.NX))
        x_ref[:, :3] = ref_xyyaw
        if self.p.accel_max_tab or self.p.accel_min_tab:   # per-speed limits at the current speed
            lo, hi = accel_limits(self.p, float(x0[self.IVX]))
            if lo is not None:
                self._xmin[3] = self._umin[1] = lo
            if hi is not None:
                self._xmax[3] = self._umax[1] = hi
        t0 = time.perf_counter()
        if self.blas_threads is None:
            res = self._solve_qp(np.asarray(x0, float), x_ref, self._u_prev)
        else:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=int(self.blas_threads), user_api="blas"):
                res = self._solve_qp(np.asarray(x0, float), x_ref, self._u_prev)
        res.solve_ms = (time.perf_counter() - t0) * 1e3
        # `_u_prev` only matters for rterm="delta" (PathFollower always uses
        # the default "abs"). Keep it to solutions the field would have
        # applied (solved / solved inaccurate, finite), never an infeasible u.
        if (res.u is not None and np.all(np.isfinite(res.u))
                and res.extra.get("status_val") in (1, 2)):
            self._u_prev = np.asarray(res.u, float).copy()
        return res

    def _linearize_dynamics(self, x_op):
        import scipy.linalg
        p, dt = self.p, self.dt
        yaw, v_cg_x, v_cg_y, yaw_rate = x_op[self.IYAW], x_op[self.IVX], x_op[self.IVY], x_op[self.IYAW_RATE]
        use_kinematic = v_cg_x < p.kinematic_threshold_speed
        A = np.zeros((self.NX, self.NX))
        B = np.zeros((self.NX, self.NU))
        cy, sy = math.cos(yaw), math.sin(yaw)
        if use_kinematic:
            A[self.IX, self.IYAW] = -v_cg_x * sy
            A[self.IX, self.IVX] = cy
            A[self.IY, self.IYAW] = v_cg_x * cy
            A[self.IY, self.IVX] = sy
            A[self.IYAW, self.IYAW_RATE] = 1.0
            A[self.IVX, self.IACCEL] = 1.0
            GAIN = 10.0
            lr, L = p.l_rig_to_cg, p.wheelbase
            A[self.IVY, self.IVY] = -GAIN
            A[self.IVY, self.ISTEERING] = GAIN * v_cg_x * lr / L
            A[self.IYAW_RATE, self.ISTEERING] = GAIN * v_cg_x / L
            A[self.IYAW_RATE, self.IYAW_RATE] = -GAIN
        else:
            km, ki = p.mass * v_cg_x, p.inertia * v_cg_x
            lf, lr = p.wheelbase - p.l_rig_to_cg, p.l_rig_to_cg
            Caf, Car = p.front_cornering_stiffness, p.rear_cornering_stiffness
            lf_caf, lr_car = lf * Caf, lr * Car
            a_00 = -2 * (Caf + Car) / km
            a_01 = -v_cg_x - 2 * (lf_caf - lr_car) / km
            a_10 = -2 * (lf_caf - lr_car) / ki
            a_11 = -2 * (lf * lf_caf + lr * lr_car) / ki
            b_00 = 2 * Caf / p.mass
            b_10 = 2 * lf_caf / p.inertia
            v_rig_y = v_cg_y - lr * yaw_rate
            A[self.IX, self.IYAW] = -v_cg_x * sy - v_rig_y * cy
            A[self.IX, self.IVX] = cy
            A[self.IX, self.IVY] = -sy
            A[self.IX, self.IYAW_RATE] = lr * sy
            A[self.IY, self.IYAW] = v_cg_x * cy - v_rig_y * sy
            A[self.IY, self.IVX] = sy
            A[self.IY, self.IVY] = cy
            A[self.IY, self.IYAW_RATE] = -lr * cy
            A[self.IYAW, self.IYAW_RATE] = 1.0
            A[self.IVX, self.IACCEL] = 1.0
            A[self.IVY, self.IVY] = a_00
            A[self.IVY, self.IYAW_RATE] = a_01
            A[self.IVY, self.ISTEERING] = b_00
            A[self.IYAW_RATE, self.IVY] = a_10
            A[self.IYAW_RATE, self.IYAW_RATE] = a_11
            A[self.IYAW_RATE, self.ISTEERING] = b_10
        A[self.ISTEERING, self.ISTEERING] = -1.0 / p.steering_time_constant
        B[self.ISTEERING, 0] = 1.0 / p.steering_time_constant
        A[self.IACCEL, self.IACCEL] = -1.0 / p.acceleration_time_constant
        B[self.IACCEL, 1] = 1.0 / p.acceleration_time_constant
        nx, nu = self.NX, self.NU
        M = np.zeros((nx + nu, nx + nu))
        M[:nx, :nx] = A * dt
        M[:nx, nx:] = B * dt
        expM = scipy.linalg.expm(M)
        return expM[:nx, :nx], expM[:nx, nx:]

    def _solve_qp(self, x0, x_ref, u_prev=None) -> SolveResult:
        import osqp
        import scipy.sparse as sparse
        N, nx, nu = self.N, self.NX, self.NU
        A_d, B_d = self._linearize_dynamics(x0)
        S_x = np.zeros(((N + 1) * nx, nx))
        S_u = np.zeros(((N + 1) * nx, N * nu))
        A_pow = np.eye(nx)
        for k in range(N + 1):
            S_x[k * nx:(k + 1) * nx, :] = A_pow
            if k < N:
                A_pow = A_d @ A_pow
        for j in range(N):
            Psi = B_d
            for k in range(j, N):
                S_u[(k + 1) * nx:(k + 2) * nx, j * nu:(j + 1) * nu] = Psi
                Psi = A_d @ Psi
        Q_blk = np.zeros(((N + 1) * nx, (N + 1) * nx))
        for k in range(N):
            if k >= self.g.idx_start_penalty:
                Q_blk[k * nx:(k + 1) * nx, k * nx:(k + 1) * nx] = self._Q
        if N >= self.g.idx_start_penalty:
            Q_blk[N * nx:, N * nx:] = self._Qf
        R_blk = np.zeros((N * nu, N * nu))
        for k in range(N):
            R_blk[k * nu:(k + 1) * nu, k * nu:(k + 1) * nu] = self._R
        x_pred_free = S_x @ x0
        g = S_u.T @ Q_blk @ (x_pred_free - x_ref.flatten())
        if self.rterm == "delta":
            D = np.eye(N * nu) - np.eye(N * nu, k=-nu)
            H = S_u.T @ Q_blk @ S_u + D.T @ R_blk @ D
            e = np.zeros(N * nu)
            e[:nu] = np.zeros(nu) if u_prev is None else np.asarray(u_prev, float)
            g = g - D.T @ R_blk @ e
        else:
            H = S_u.T @ Q_blk @ S_u + R_blk
        H = 0.5 * (H + H.T) + self.QP_REGULARIZATION * np.eye(N * nu)
        rows = [k * nx + i for k in range(1, N + 1) for i in self._cidx]
        A_ineq = sparse.vstack([sparse.eye(N * nu, format="csc"), sparse.csc_matrix(S_u[rows, :])], format="csc")
        xp = x_pred_free[rows]
        l_ineq = np.concatenate([np.tile(self._umin, N), np.tile(self._xmin, N) - xp])
        u_ineq = np.concatenate([np.tile(self._umax, N), np.tile(self._xmax, N) - xp])
        try:
            qp = osqp.OSQP()
            qp.setup(P=sparse.csc_matrix(H), q=g, A=A_ineq, l=l_ineq, u=u_ineq, verbose=False,
                     eps_abs=self.OSQP_EPS_ABS, eps_rel=self.OSQP_EPS_REL, max_iter=self.OSQP_MAX_ITER,
                     polish=False)
            r = qp.solve(raise_error=False)
        except Exception as e:  # noqa: BLE001 — OSQPException on invalid data; classified EXC
            _log.debug("LinearMPC: OSQP setup/solve raised", exc_info=True)
            return SolveResult(u=None, exception=f"{type(e).__name__}: {e}")
        x = None if r.x is None else np.asarray(r.x, float)
        viol = float("inf")
        if x is not None and np.all(np.isfinite(x)):
            ax = A_ineq @ x
            viol = max(0.0, float(np.max(np.maximum(l_ineq - ax, ax - u_ineq))))
        self._last_qp = dict(violation=viol)
        u = None if x is None else np.asarray(x[:nu], float)
        return SolveResult(u=u, return_status=str(r.info.status),
                           success=int(r.info.status_val) == int(osqp.SolverStatus.OSQP_SOLVED),
                           iter_count=int(r.info.iter),
                           extra={"status_val": int(r.info.status_val), "violation": viol})

    def max_violation(self) -> float:
        return float("inf") if self._last_qp is None else self._last_qp["violation"]


class _DecelGatedLMPC:
    """(v2.1.0, private) The LMPC side of ``PathFollower(mpc_decel_limit=
    "brake")``: ONE :class:`LinearMPC` built on the throttle-row parameters
    ``p_thr``; :meth:`select_decel` points its ``p`` at ``p_brk`` (open) or
    ``p_thr`` (closed). Neither parameter object is mutated, and nothing is
    transferred: :meth:`LinearMPC.solve` re-reads the deceleration row of its
    current ``p`` at the speed of ``x0`` on every solve (its per-solve box),
    and, while ``p.accel_min_tab`` is set, nothing else in the QP reads the
    bound. That holds only with the table: :meth:`LinearMPC.__init__` caches
    the constant ``accel_min`` box from ``p_thr``, and with neither
    ``accel_min_tab`` nor ``accel_max_tab`` set :meth:`LinearMPC.solve` never
    refreshes it, so the swap would not take effect. ``solver_params`` always
    sets both tables, so that case is not reachable through the SDK's own
    parameter path. The law decides when to open.

    Explicit members only (no ``__getattr__``): ``p`` (the active parameters),
    ``N``, ``dt``, ``u_prev``, ``solve``, ``max_violation``, ``set_u_prev``,
    ``select_decel``. No ``plan_vector`` (the LMPC keeps no state trajectory,
    so ``last_plan`` stays ``None``) and no ``setup_ms`` (:class:`LinearMPC`
    has none). ``ValueError`` unless ``lmpc.p`` is ``p_thr`` and the two
    parameter sets agree on type and every field except the deceleration
    bound."""

    def __init__(self, lmpc: LinearMPC, p_thr, p_brk):
        if lmpc.p is not p_thr:
            raise ValueError("decel-gated LMPC: the LinearMPC must be built on the throttle-row parameters")
        if p_thr is p_brk:
            raise ValueError("decel-gated LMPC: p_thr and p_brk are the same object")
        _check_decel_pair_params(p_thr, p_brk)
        self._lmpc = lmpc
        self._p_thr, self._p_brk = p_thr, p_brk
        self.N, self.dt = int(lmpc.N), float(lmpc.dt)

    @property
    def p(self):
        return self._lmpc.p

    @property
    def u_prev(self) -> np.ndarray:
        return self._lmpc.u_prev

    def select_decel(self, open_: bool) -> None:
        self._lmpc.p = self._p_brk if open_ else self._p_thr

    def set_u_prev(self, delta: float, accel: float) -> None:
        self._lmpc.set_u_prev(delta, accel)

    def solve(self, x0, ref_xyyaw) -> SolveResult:
        return self._lmpc.solve(x0, ref_xyyaw)

    def max_violation(self) -> float:
        return self._lmpc.max_violation()
