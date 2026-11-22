"""PathFollower regimes with the MPC layer on (plan section 3.1), fake solver.

* R-REV: in a reverse block the solver is never called and the command is
  the PD law's (equal to an ``mpc_model="off"`` follower fed the same states);
* R-FAIL: a failed solve under ``on_solve_failure="pd"`` gives the PD command
  and ``last_mode == "PD-FALLBACK"``; the next applicable solve returns to MPC;
* DONE: the fixed stop command, no solve;
* re-entering a forward block after a cusp restarts the MPC reference.

Mutants this file must fail on (plan section 11): the MPC also running in a
reverse block; the fallback keeping the MPC command.
"""
import copy
import os
import sys

import numpy as np
import pytest

pytest.importorskip("casadi")
pytest.importorskip("do_mpc")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control import PathFollower  # noqa: E402
from genesis_vehicle.control.mpc import MPCVehicleParams  # noqa: E402
from genesis_vehicle.tests._mpc_fakes import FakeSolver, exc, ok  # noqa: E402
from genesis_vehicle.tests.data import make_mpc_field_replay as rp  # noqa: E402
from genesis_vehicle.tests.test_path_follower import _densify, _run_unicycle  # noqa: E402
from genesis_vehicle.tests.test_sweep_table import make_table  # noqa: E402

FWD = _densify([(0.0, 0.0, 0.0, 1.5), (10.0, 0.0, 0.0, 1.5)])
BWD = _densify([(10.0, 0.0, 0.0, -1.5), (6.0, -4.0, 0.0, -1.5)])
FWD2 = _densify([(6.0, -4.0, 0.0, 1.5), (14.0, -4.0, 0.0, 1.5)])


def _mpc(path, results=None, policy="pd"):
    p = MPCVehicleParams.from_dict(copy.deepcopy(rp.PARAMS_SDK))
    f = PathFollower(path, make_table(), mpc_params=p, on_solve_failure=policy,
                     command_mode="signed")
    fake = FakeSolver(results, default=ok(0.0, 0.5))
    f._law.solver = fake
    return f, fake


def _states():
    """A closed-loop state sequence of the 1.x follower over fwd -> cusp -> bwd -> fwd."""
    off = PathFollower(FWD + BWD + FWD2, make_table(), mpc_model="off")
    seq = []
    orig = off.step

    def rec(pos, yaw, v, pitch, roll, **kw):
        seq.append((tuple(pos), yaw, v))
        return orig(pos, yaw, v, pitch, roll)
    off.step = rec
    _run_unicycle(off, 0.0, 0.0, 0.0, max_steps=12000)
    return seq


def test_reverse_block_is_pd_and_never_solves():
    seq = _states()
    f, fake = _mpc(FWD + BWD + FWD2)
    g = PathFollower(FWD + BWD + FWD2, make_table(), mpc_model="off")
    n_rev = 0
    for k, (pos, yaw, v) in enumerate(seq):
        n_before = len(fake.calls)
        out_f = f.step(pos, yaw, v, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=k * 0.05)
        out_g = g.step(pos, yaw, v, 0.0, 0.0)
        if f.blocks[f.block_i][2] < 0 and f.last_mode == "DRV-1":
            n_rev += 1
            assert len(fake.calls) == n_before, f"solver called in the reverse block at step {k}"
            assert out_f == out_g and f.regime == "R-REV"
    assert n_rev > 50
    assert {"MPC", "DRV-1", "BRAKE_TRANS", "DONE"} <= {f.last_mode} | set(_modes(f, seq))


def _modes(f, seq):
    f2, _ = _mpc(FWD + BWD + FWD2)
    out = []
    for k, (pos, yaw, v) in enumerate(seq):
        f2.step(pos, yaw, v, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=k * 0.05)
        out.append(f2.last_mode)
    return out


def test_reentry_after_the_cusp_restarts_the_reference():
    seq = _states()
    f, fake = _mpc(FWD + BWD + FWD2)
    starts = []
    real = f._law.start_block
    f._law.start_block = lambda wps, t0=0.0: (starts.append((len(wps), t0)), real(wps, t0))[1]
    for k, (pos, yaw, v) in enumerate(seq):
        f.step(pos, yaw, v, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=k * 0.05)
    assert len(starts) == 2 and starts[0][1] == 0.0 and starts[1][1] > 0.0


def test_failed_solve_falls_back_to_pd_then_recovers():
    f, fake = _mpc(FWD, [exc(), ok(0.0, 0.5)])
    g = PathFollower(FWD, make_table(), mpc_model="off")
    st = ((0.5, 0.0), 0.0, 1.0, 0.0, 0.0)
    out_f = f.step(*st, yaw_rate=0.0, v_lat=0.0, t=0.0)
    out_g = g.step(*st)
    assert f.last_mode == "PD-FALLBACK" and f.regime == "R-FAIL" and out_f == out_g
    f.step((0.55, 0.0), 0.0, 1.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=0.025)
    assert f.last_mode == "PD-FALLBACK"                # until the next solve
    f.step((0.8, 0.0), 0.0, 1.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=0.1)
    assert f.last_mode == "MPC" and f.mpc_stats["status"] == {"exception": 1, "converged": 1}


def test_hold_policy_keeps_the_previous_command():
    f, fake = _mpc(FWD, [ok(0.0, 0.7), exc()], policy="hold")
    f.step((0.5, 0.0), 0.0, 1.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=0.0)
    f.step((0.8, 0.0), 0.0, 1.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=0.1)
    assert f.last_mode == "MPC-HOLDCMD" and f._law.cmd_a == 0.7


def test_done_is_the_stop_command_without_a_solve():
    f, fake = _mpc(FWD)
    out = f.step((9.5, 0.0), 0.0, 1.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=0.0)
    assert out == (0.0, 0.0, 1.0) and f.last_mode == "DONE" and fake.calls == []


def test_mpc_hold_at_rest_when_the_plan_wants_to_slow():
    f, fake = _mpc(FWD, [ok(0.0, -0.5)])
    out = f.step((0.5, 0.0), 0.0, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=0.0)
    assert out == (0.0, 0.0, 1.0) and f.last_mode == "MPC-HOLD"
