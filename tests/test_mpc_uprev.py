"""do_mpc's ``_u_prev`` (``MPC.u0``) reset rule (plan section 3.7), real NMPC.

``_u_prev`` is a COST PARAMETER — the rterm penalises the command change
from it — so the law resets it to the vehicle's measured ``(d0, a0)`` before
the first solve after a restart, and leaves do_mpc's own update alone inside
a block. Observed through ``opt_p_num['_u_prev']``, the value do_mpc put into
the NLP for that solve (``controller/_mpc.py:1017``; api-review C1).

Mutants this file must fail on (plan section 11): the reset deleted (a stale
``u_prev`` survives a restart); the reset applied on every solve inside a
block (the field does not — that would break parity).
"""
import math
import os
import sys

import numpy as np
import pytest

pytest.importorskip("casadi")
pytest.importorskip("do_mpc")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control.mpc.alpasim_nonlinear import NonlinearMPC, SolveResult  # noqa: E402
from genesis_vehicle.control.mpc.law import _MPCLaw  # noqa: E402
from genesis_vehicle.tests._mpc_fakes import solver_params  # noqa: E402

DT = 0.025
PATH = [(float(x), 0.0, 0.0, 6.0) for x in np.arange(0.0, 80.0, 0.5)]
L = 2.7


@pytest.fixture(scope="module")
def solver():
    return NonlinearMPC(solver_params(), yaw_bound=math.pi / 2)


def _u_prev_used(solver):
    return np.array(solver._mpc.opt_p_num["_u_prev"]).ravel()


def _drive(law, t0, n, v0, r=0.05, dv=0.02):
    """n control steps with a speed ramp; returns the per-solve u_prev used."""
    used = []
    for k in range(n):
        t = t0 + k * DT
        o = law.step(t, (v0 * (t - t0), 0.0), 0.0, v0 + dv * k, r, 0.01)
        if o.solved:
            used.append(_u_prev_used(law.solver))
    return used


def test_path_start_from_rest_uses_do_mpcs_initial_value(solver):
    law = _MPCLaw(solver, rig_offset=1.35, on_solve_failure="pd")
    law.start_block(PATH, t0=0.0)
    used = _drive(law, 0.0, 1, v0=0.0, r=0.0, dv=0.0)
    assert used[0].tolist() == [0.0, 0.0]


def test_inside_a_block_do_mpc_keeps_its_own_solution(solver):
    law = _MPCLaw(solver, rig_offset=1.35, on_solve_failure="pd")
    law.start_block(PATH, t0=0.0)
    used, sols = [], []
    for k in range(9):
        o = law.step(k * DT, (0.1 * k, 0.0), 0.0, 3.0 + 0.02 * k, 0.05, 0.01)
        if o.solved:
            used.append(_u_prev_used(solver))
            sols.append(np.array(law.last_result.u))
    # the 2nd and 3rd solve see the previous solution, not a measured (d0, a0)
    assert np.array_equal(used[1], sols[0])
    assert np.array_equal(used[2], sols[1])


def test_block_reentry_resets_to_the_measured_state(solver):
    law = _MPCLaw(solver, rig_offset=1.35, on_solve_failure="pd")
    law.start_block(PATH, t0=0.0)
    _drive(law, 0.0, 12, v0=4.0)
    assert np.any(_u_prev_used(solver) != 0.0)
    law.start_block(PATH, t0=10.0)                    # forward block entered again
    v, r = 5.0, 0.2
    law.step(10.0, (0.0, 0.0), 0.0, v, r, 0.0)
    used = _u_prev_used(solver)
    assert used[0] == pytest.approx(math.atan(r * L / v), rel=1e-12)
    assert used[1] == 0.0                             # history cleared: a0 = 0


def test_return_from_a_pd_fallback_resets_to_the_measured_state(solver):
    law = _MPCLaw(solver, rig_offset=1.35, on_solve_failure="pd")
    law.start_block(PATH, t0=0.0)
    _drive(law, 0.0, 8, v0=4.0)                       # solves at k = 0, 4
    real = solver.solve
    solver.solve = lambda x0, ref: SolveResult(u=None, exception="RuntimeError: injected")
    try:
        o = law.step(8 * DT, (1.0, 0.0), 0.0, 4.2, 0.05, 0.01)
        assert o.mode == "PD-FALLBACK"
    finally:
        solver.solve = real
    for k in (9, 10, 11):
        law.step(k * DT, (1.0, 0.0), 0.0, 4.2, 0.05, 0.01)
    v, r = 4.5, 0.1
    o = law.step(12 * DT, (1.2, 0.0), 0.0, v, r, 0.01)
    assert o.solved and o.mode == "MPC"
    used = _u_prev_used(solver)
    a0 = (v - 4.2) / (12 * DT - 8 * DT)               # history: (t=0.2, 4.2) .. (0.3, 4.5)
    assert used[0] == pytest.approx(math.atan(r * L / v), rel=1e-12)
    assert used[1] == pytest.approx(a0, rel=1e-12)
