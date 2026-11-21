"""A scriptable stand-in for the MPC solvers (no casadi / do_mpc / osqp).

``FakeSolver`` answers ``solve(x0, ref)`` from a queue of results and records
every call (the ``x0`` and reference it was given, and every ``set_u_prev``),
so the law's cadence, initial state, classification and ``u_prev`` handling
can be checked exactly.
"""
import copy

import numpy as np

from genesis_vehicle.control.mpc import MPCVehicleParams
from genesis_vehicle.control.mpc.alpasim_nonlinear import SolveResult
from genesis_vehicle.tests.data.make_mpc_field_replay import PARAMS_SDK


def solver_params(**over):
    d = copy.deepcopy(PARAMS_SDK)
    d.update(over)
    return MPCVehicleParams.from_dict(d).solver_params("bicycle", steer_cap=0.5, dt_ctrl=0.025)


def ok(delta=0.05, accel=0.8, status="Solve_Succeeded"):
    return SolveResult(u=np.array([delta, accel]), return_status=status, success=True,
                       iter_count=5, solve_ms=1.0)


def res(u, status, success=False):
    return SolveResult(u=None if u is None else np.asarray(u, float), return_status=status,
                       success=success, iter_count=30, solve_ms=1.0)


def exc():
    return SolveResult(u=None, exception="RuntimeError: boom", solve_ms=1.0)


class FakeSolver:
    def __init__(self, results=None, violation=0.0, default=None, N=20):
        self.p = solver_params()
        self.N = N
        self.results = list(results or [])
        self.default = default or ok()
        self.violation = violation
        self.calls = []            # (x0, ref, u_prev at call time)
        self.u_prev_sets = []
        self._u_prev = np.zeros(2)

    def set_u_prev(self, delta, accel):
        self._u_prev = np.array([float(delta), float(accel)])
        self.u_prev_sets.append((float(delta), float(accel)))

    def solve(self, x0, ref):
        self.calls.append((np.array(x0, float), np.array(ref, float), self._u_prev.copy()))
        r = self.results.pop(0) if self.results else self.default
        if r.u is not None and r.exception is None:
            self._u_prev = np.asarray(r.u, float).copy()      # do_mpc keeps its own solution
        return r

    def max_violation(self):
        return self.violation
