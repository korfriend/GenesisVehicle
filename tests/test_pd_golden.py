"""The 1.x PD path follower, step for step, against its 1.8.0 capture.

``tests/data/pd_golden_1x.json`` was captured by ``tests/data/make_pd_golden.py``
on the unmodified 1.8.0 tree (``3810c79``), before any 2.0.0 edit to
``control/path_follower.py``. This test replays the same Genesis-free
scenarios (synthetic paths: straight, square, cusp, explicit-yaw cusp; a
kinematic unicycle in closed loop; a synthetic ``SweepTable`` and a fake
coupled plant) and compares every recorded step with ``==`` — state fed in,
``(throttle, steer, brake)`` out, ``last_mode``, block and segment index.

Scope of the guarantee: exactly these scenarios. A Genesis demo's command
sequence is not covered here (it is measured once per release).

Since 2.0.1 (plan_r210 section 3B.7): 34 of the 38 entries are compared on
every step. In the other 4 (``square`` and ``square_tuned``, table and
coupled) the 1.x follower missed the arrival circle and drove away past the
goal; 2.0.1's goal-passed stop (GP-AWAY2) fires there. Those entries are
compared with ``==`` up to the step ``K_STAR`` (literals measured
independently, before the implementation, by
``.pm_r2/r210_probe/replan3/golden_kstar_w2.py``); from ``K_STAR`` on every
step must be ``GOAL-MISSED`` with the stop command ``(0, 0, 1)`` and
``stop_reason == "goal_passed"``. The golden file is NOT re-captured (it is
the 1.8.0 capture). W2 changes no golden step.

Mutants this test must fail on (plan section 11): the GP-AWAY2 stop one step
late or early, the stop not sticky, the growth condition on one call instead
of two (2.0.1); the KICK coefficient
``0.4`` -> ``0.41`` in ``PathFollower.finish``, and the heading blend
``0.7 * yaw_err + 0.3 * pos_err`` rewritten as ``0.7 * (yaw_err + 0.3 / 0.7 *
pos_err)`` (algebraically equal, rounds differently). Swapping the two terms
(``0.3 * pos_err + 0.7 * yaw_err``) is NOT such a mutant: IEEE addition is
commutative, so it is bit-identical and no test can or should catch it.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control import PathFollower  # noqa: E402
from genesis_vehicle.tests.data import make_pd_golden as gold  # noqa: E402

_GOLDEN = os.path.join(os.path.dirname(__file__), "data", "pd_golden_1x.json")

#: Extra constructor arguments that select the 1.x behaviour (the 1.8.0
#: capture ran with none; since 2.0.0 the PD follower is ``mpc_model="off"``).
PD_KWARGS = {"mpc_model": "off"}

#: (2.0.1) first step at which GOAL-MISSED fires, per entry (independent probe).
K_STAR = {"square/table": 665, "square/coupled": 644,
          "square_tuned/table": 684, "square_tuned/coupled": 657}


def _golden():
    with open(_GOLDEN) as fh:
        data = json.load(fh)
    assert data["schema"] == gold.SCHEMA
    return data


_DATA = _golden()


@pytest.mark.parametrize("key", sorted(_DATA["runs"]))
def test_pd_follower_matches_its_1_8_0_capture(key):
    name, plant_kind = key.split("/")
    f, got = gold.run_scenario(name, plant_kind, PD_KWARGS)
    assert isinstance(f, PathFollower)
    want = _DATA["runs"][key]["steps"]
    assert len(got) == len(want), (len(got), len(want))
    k_star = K_STAR.get(key, len(want))
    for k, (g, w) in enumerate(zip(got[:k_star], want[:k_star])):
        assert g == w, f"{key}: first difference at step {k}: got {g}, want {w}"
    for k in range(k_star, len(got)):
        assert got[k][5:9] == [0.0, 0.0, 1.0, "GOAL-MISSED"], f"{key}: step {k}: {got[k]}"
    if key in K_STAR:
        assert want[k_star][8] != "GOAL-MISSED" and got[k_star][:5] == want[k_star][:5]
        assert f.stop_reason == "goal_passed"
    else:
        assert f.stop_reason is None
    if plant_kind == "coupled":
        assert f.plant.n_applied == _DATA["runs"][key]["n_applied"]
        if key in K_STAR:
            assert list(f.plant.applied) == [0.0, 0.0]
        else:
            assert list(f.plant.applied) == _DATA["runs"][key]["applied"]


def test_golden_exercises_every_pd_branch():
    """The capture is not vacuous: KICK saturation, brake-at-rest, the cusp
    state machine and DONE all occur in it."""
    steps = [s for r in _DATA["runs"].values() for s in r["steps"]]
    modes = {s[8] for s in steps}
    assert {"DRV+1", "DRV-1", "BRAKE_TRANS", "DONE"} <= modes
    assert any(abs(s[5]) == 1.0 and s[8].startswith("DRV") for s in steps)
    assert any(s[7] == 1.0 and s[8].startswith("DRV") for s in steps)
