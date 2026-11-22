"""The 2.0.0 misaligned-start MPC stall, reproduced on the controller side (v2.0.1).

A CONSTANT-STATE reproduction with the real nonlinear MPC (do_mpc + IPOPT):
the vehicle is held at the first waypoint of the documented stall path
(30,0) -> (10,-8) -> (-10,-8) -> (-30,0) at 2 m/s with chassis yaw 0 (path
heading -2.76 rad, |e_psi| = 2.76 rad), and the same state is fed every step.
This is a Genesis-free reproduction of the controller-side mechanism, NOT a
replay of the Genesis run that documented the stall (that run dropped the car
from z = 0.5 and let it settle; its state was not exactly constant). Params
and table are the synthetic in-repo ones (``tests/data/make_mpc_field_replay``,
``tests/test_sweep_table.make_table``), not the documented run's files.

* both features ``None`` (equal to 2.0.0 here: on this constant state no
  ``GOAL-MISSED``, block-end speed sign or ``fail_budget`` step occurs): for 40 solves every solve
  converges to a standstill — ``|a_cmd| < 1e-5``, planned speed below
  0.01 m/s at every node — and every step brakes (brake = 1);
* the default: the first step is ``PD-ALIGN`` with throttle 1 (the PD KICK),
  no solve, and no MPC reference exists during that block-entry R-ALIGN.

Mutants this file must fail on (plan_r210 section 6 + plan-review C3): the
gate comparison ``>`` turned into ``<``; the gate evaluated after
``start_block`` (a reference then exists during R-ALIGN).
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
from genesis_vehicle.tests.data import make_mpc_field_replay as rp  # noqa: E402
from genesis_vehicle.tests.test_sweep_table import make_table  # noqa: E402

STALL_PATH = [(30.0, 0.0, 0.0, 2.0), (10.0, -8.0, 0.0, 2.0), (-10.0, -8.0, 0.0, 2.0),
              (-30.0, 0.0, 0.0, 2.0)]
START = (30.0, 0.0)


def _follower(**kw):
    return PathFollower(STALL_PATH, make_table(),
                        mpc_params=MPCVehicleParams.from_dict(copy.deepcopy(rp.PARAMS_SDK)),
                        command_mode="signed", **kw)


def test_both_none_reproduces_the_standstill_solution():
    f = _follower(mpc_align_gate=None, mpc_stall_timeout=None)
    n_steps = 160                                   # 40 solves at DT_MPC 0.1, dt 0.025
    for k in range(n_steps):
        thr, steer, brake = f.step(START, 0.0, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=k * 0.025)
        assert brake == 1.0, f"step {k}: brake {brake}"
        assert f.regime == "R-MPC"
        if f._law.n_solve and k % 4 == 0:
            assert abs(f._law.cmd_a) < 1e-5, f"step {k}: a_cmd {f._law.cmd_a}"
            plan_v = f.last_plan[1][:, 3]
            assert float(np.abs(plan_v).max()) < 0.01, f"step {k}: planned v {plan_v}"
    st = f.mpc_stats
    assert st["n_solve"] == 40 and st["status"] == {"converged": 40}
    assert st["handover"] == {"align": 0, "stall": 0}


def test_default_hands_the_misaligned_start_to_pd():
    f = _follower()
    thr, steer, brake = f.step(START, 0.0, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=0.0)
    assert f.last_mode == "PD-ALIGN" and f.regime == "R-ALIGN"
    assert thr == 1.0 and brake == 0.0                # the PD KICK
    assert f._law.n_solve == 0
    assert f._mpc_block is None and f._law.ref is None  # block-entry: no reference yet
    assert f.mpc_stats["handover"] == {"align": 1, "stall": 0}
