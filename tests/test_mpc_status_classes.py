"""The solve classification of plan section 3.6 and what each policy applies.

A fake solver returns one representative of every class — CONV
``Solve_Succeeded``; CAP ``Maximum_Iterations_Exceeded`` with a feasible and
with an infeasible iterate; INFEAS ``Infeasible_Problem_Detected``; OTHER
``Internal_Error``; EXC (the solver raised); NAN (``u`` not finite) — and
the law's command, mode and ``mpc_stats`` keys are checked under ``"pd"`` and
``"hold"``. The LMPC (OSQP) rows are checked the same way.

Mutants this file must fail on (plan section 11): ``Infeasible_Problem_Detected``
moved to an applied class; the feasibility check on CAP removed; ``"hold"``
treating an infeasible solve as a fallback (it must apply it, like the field);
``"hold"`` applying a NaN ``u``; the command updated on a failed solve.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control.mpc.alpasim_nonlinear import SolveResult  # noqa: E402
from genesis_vehicle.control.mpc.law import _MPCLaw, classify_solve  # noqa: E402
from genesis_vehicle.tests._mpc_fakes import FakeSolver, exc, ok, res  # noqa: E402

DT = 0.025
PATH = [(float(x), 0.0, 0.0, 4.0) for x in np.arange(0.0, 40.0, 0.5)]
PREV = ok(delta=0.1, accel=0.5)
NEW_U = [0.2, -0.4]

CASES = {
    # name: (result, violation, applied under "pd", applied under "hold", stats key)
    "CONV": (res(NEW_U, "Solve_Succeeded", True), 0.0, True, True, "converged"),
    "CAP-feasible": (res(NEW_U, "Maximum_Iterations_Exceeded"), 1e-6, True, True, "capped"),
    "CAP-infeasible": (res(NEW_U, "Maximum_Iterations_Exceeded"), 4.73, False, True,
                       "capped_infeasible"),
    "INFEAS": (res(NEW_U, "Infeasible_Problem_Detected"), 0.0, False, True,
               "failed:Infeasible_Problem_Detected"),
    "OTHER": (res(NEW_U, "Internal_Error"), 0.0, False, True, "failed:Internal_Error"),
    "EXC": (exc(), 0.0, False, False, "exception"),
    "NAN": (res([float("nan"), 0.1], "Solve_Succeeded", True), 0.0, False, False, "nan"),
}


def _run(policy, case):
    r, viol, *_ = CASES[case]
    s = FakeSolver([PREV, r], violation=viol)
    law = _MPCLaw(s, rig_offset=1.35, on_solve_failure=policy)
    law.start_block(PATH, t0=0.0)
    o1 = law.step(0.0, (0.0, 0.0), 0.0, 2.0, 0.0, 0.0)          # PREV applied
    o2 = law.step(0.1, (0.2, 0.0), 0.0, 2.0, 0.0, 0.0)          # the case
    o3 = law.step(0.125, (0.25, 0.0), 0.0, 2.0, 0.0, 0.0)       # between solves
    return law, s, o1, o2, o3


@pytest.mark.parametrize("case", sorted(CASES))
def test_pd_policy(case):
    law, s, o1, o2, o3 = _run("pd", case)
    _, _, applied, _, key = CASES[case]
    assert o1.mode == "MPC"
    assert o2.status == key and law.last_status == key
    if applied:
        assert o2.mode == "MPC" and (law.cmd_delta, law.cmd_a) == tuple(NEW_U)
        assert o3.mode == "MPC"
    else:
        assert o2.mode == "PD-FALLBACK" and o2.a_t is None and o2.w_t is None
        assert o3.mode == "PD-FALLBACK"                         # until the next solve
        assert (law.cmd_delta, law.cmd_a) == (0.1, 0.5)          # command not updated


@pytest.mark.parametrize("case", sorted(CASES))
def test_hold_policy_is_the_field_behaviour(case):
    law, s, o1, o2, o3 = _run("hold", case)
    _, _, _, applied, key = CASES[case]
    assert o2.status == key
    if applied:
        assert o2.mode == "MPC" and (law.cmd_delta, law.cmd_a) == tuple(NEW_U)
    else:                                                        # EXC, NAN: keep previous
        assert o2.mode == "MPC-HOLDCMD" and o3.mode == "MPC-HOLDCMD"
        assert (law.cmd_delta, law.cmd_a) == (0.1, 0.5)
        assert o2.a_t is not None


def test_pd_recovers_on_the_next_applicable_solve():
    s = FakeSolver([PREV, exc(), ok(0.3, 0.2)])
    law = _MPCLaw(s, rig_offset=1.35, on_solve_failure="pd")
    law.start_block(PATH, t0=0.0)
    law.step(0.0, (0.0, 0.0), 0.0, 2.0, 0.0, 0.0)
    assert law.step(0.1, (0.0, 0.0), 0.0, 2.0, 0.0, 0.0).mode == "PD-FALLBACK"
    o = law.step(0.2, (0.0, 0.0), 0.0, 2.0, 0.0, 0.0)
    assert o.mode == "MPC" and law.cmd_delta == 0.3


def test_mpc_stats_keys_are_the_classes():
    law, *_ = _run("hold", "INFEAS")
    st = law.mpc_stats(t_now=0.15)
    assert set(st) == {"n_solve", "status", "last_status", "last_solve_ms", "plan_age_s"}
    assert st["status"] == {"converged": 1, "failed:Infeasible_Problem_Detected": 1}
    assert st["n_solve"] == 2


def test_unknown_policy():
    with pytest.raises(ValueError, match="hold"):
        _MPCLaw(FakeSolver(), rig_offset=1.0, on_solve_failure="ignore")


# --- LMPC (OSQP) rows ---------------------------------------------------------------

def _osqp(status_val, status, u=(0.1, 0.2)):
    return SolveResult(u=np.asarray(u, float), return_status=status,
                       success=status_val == 1, extra={"status_val": status_val})


@pytest.mark.parametrize("sv,name,viol,pd_ok,hold_ok,key", [
    (1, "solved", 0.0, True, True, "converged"),
    (2, "solved inaccurate", 1e-6, True, True, "capped"),
    (2, "solved inaccurate", 0.5, False, True, "capped_infeasible"),
    (7, "maximum iterations reached", 1e-6, True, False, "capped"),
    (8, "run time limit reached", 0.5, False, False, "capped_infeasible"),
    (3, "primal infeasible", 0.0, False, False, "failed:primal infeasible"),
    (9, "problem non convex", 0.0, False, False, "failed:problem non convex"),
    (11, "unsolved", 0.0, False, False, "failed:unsolved"),
])
def test_lmpc_classes(sv, name, viol, pd_ok, hold_ok, key):
    for policy, want in (("pd", pd_ok), ("hold", hold_ok)):
        s = FakeSolver([PREV, _osqp(sv, name, NEW_U)], violation=viol)
        law = _MPCLaw(s, rig_offset=1.35, on_solve_failure=policy, impl="linear")
        law.start_block(PATH, t0=0.0)
        s.results[0] = _osqp(1, "solved", (0.1, 0.5))
        law.step(0.0, (0.0, 0.0), 0.0, 2.0, 0.0, 0.0)
        o = law.step(0.1, (0.0, 0.0), 0.0, 2.0, 0.0, 0.0)
        assert o.status == key
        assert ((law.cmd_delta, law.cmd_a) == tuple(NEW_U)) == want, (policy, name)


def test_classify_exception_wins_over_status():
    r = SolveResult(u=None, return_status="Solve_Succeeded", exception="X")
    assert classify_solve(r, FakeSolver())[0] == "EXC"
