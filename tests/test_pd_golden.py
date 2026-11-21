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

Mutants this test must fail on (plan section 11): the KICK coefficient
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
    for k, (g, w) in enumerate(zip(got, want)):
        assert g == w, f"{key}: first difference at step {k}: got {g}, want {w}"
    if plant_kind == "coupled":
        assert f.plant.n_applied == _DATA["runs"][key]["n_applied"]
        assert list(f.plant.applied) == _DATA["runs"][key]["applied"]


def test_golden_exercises_every_pd_branch():
    """The capture is not vacuous: KICK saturation, brake-at-rest, the cusp
    state machine and DONE all occur in it."""
    steps = [s for r in _DATA["runs"].values() for s in r["steps"]]
    modes = {s[8] for s in steps}
    assert {"DRV+1", "DRV-1", "BRAKE_TRANS", "DONE"} <= modes
    assert any(abs(s[5]) == 1.0 and s[8].startswith("DRV") for s in steps)
    assert any(s[7] == 1.0 and s[8].startswith("DRV") for s in steps)
