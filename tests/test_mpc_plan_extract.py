"""Reading the predicted trajectory off a real NMPC solve (plan section 2.2).

With ``store_full_solution=False`` the plan is still in ``opt_x_num``: node k's
state at time ``k * 0.1 s`` is the LAST collocation point of interval k
(``['_x', k, 0][-1]``); index ``[0]`` would be the interval's first point
(radau degree 2: one third of the way in). Read unscaled (api-review C3).

Mutant this file must fail on (plan section 11): ``[-1]`` -> ``[0]``.
"""
import math
import os
import sys

import numpy as np
import pytest

pytest.importorskip("casadi")
pytest.importorskip("do_mpc")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control.mpc.alpasim_nonlinear import NonlinearMPC  # noqa: E402
from genesis_vehicle.control.mpc.law import _MPCLaw  # noqa: E402
from genesis_vehicle.tests._mpc_fakes import solver_params  # noqa: E402


def test_plan_nodes_are_the_states_at_the_node_times():
    s = NonlinearMPC(solver_params(), yaw_bound=math.pi / 2)
    law = _MPCLaw(s, rig_offset=1.35, on_solve_failure="pd")
    law.start_block([(float(x), 0.0, 0.0, 5.0) for x in np.arange(0.0, 60.0, 0.5)], t0=0.0)
    o = law.step(0.0, (1.35, 0.0), 0.0, 5.0, 0.0, 0.0)
    assert o.mode == "MPC"
    times, X = law.last_plan
    assert X.shape == (21, 8)
    assert np.allclose(np.diff(times), 0.1, rtol=0, atol=1e-12) and times[0] == 0.0
    # node 0 is the x0 the solver was given
    assert X[0, 3] == pytest.approx(5.0, rel=1e-9)
    assert np.allclose(X[0, [0, 1, 2]], 0.0, atol=1e-9)
    # node 1 is 0.1 s ahead: ~0.5 m at 5 m/s (the first collocation point is ~0.167 m)
    assert X[1, 0] == pytest.approx(0.5, abs=0.05)
    assert np.all(np.diff(X[:, 0]) > 0)
    # a copy: mutating it does not touch the law's record
    X[:] = 0.0
    assert law.last_plan[1][1, 0] != 0.0
