"""I-2 (2.1.0): the DEFAULT MPC parameter set is unchanged from 2.0.6, field for field.

Fixture ``tests/data/mpc_default_params_2_0_6.json`` was captured on the 2.0.6 tree (``d68ef4d``) by
``tests/data/make_mpc_default_params.py`` before any 2.1.0 change. The fake-solver tests swap the solver
and build ``p`` with the bicycle / throttle rule themselves (``_mpc_fakes.py``), so they cannot see a
change of the default bound; this file can.

Checks (Genesis-free): every field of the default follower's ``solver.p`` (NMPC gear, NMPC signed, LMPC)
and of ``solver_params()`` / ``solver_params("bicycle", steer_cap=0.5, dt_ctrl=0.025)`` ``==`` the
fixture, with the same class (``_AlpaSimParams``); ``solver_params(..., decel_limit="throttle")`` equals
the call without the keyword; the default ``NonlinearMPC`` is built with ``model_builder=build_bicycle``.

Mutants this file must fail on (plan_r21 section 7.2): the default bound switched to ``"brake"``; the
0 m/s deceleration entry taken from another row; a default follower handed another model builder.
"""
import copy
import dataclasses
import inspect
import json
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control.mpc import MPCVehicleParams  # noqa: E402
from genesis_vehicle.tests.data.make_mpc_default_params import PATH, as_record  # noqa: E402
from genesis_vehicle.tests.data.make_mpc_field_replay import PARAMS_SDK  # noqa: E402
from genesis_vehicle.tests.test_sweep_table import make_table  # noqa: E402

FIX = json.load(open(os.path.join(os.path.dirname(__file__), "data", "mpc_default_params_2_0_6.json")))


def _rec():
    return MPCVehicleParams.from_dict(copy.deepcopy(PARAMS_SDK))


def test_solver_params_defaults_equal_the_2_0_6_capture():
    assert as_record(_rec().solver_params()) == FIX["solver_params_default"]
    assert as_record(_rec().solver_params("bicycle", steer_cap=0.5, dt_ctrl=0.025)) == \
        FIX["solver_params_bicycle_cap05_dt0025"]


def test_explicit_throttle_equals_the_call_without_the_keyword():
    for kw in ({}, {"steer_cap": 0.5, "dt_ctrl": 0.025}):
        a = _rec().solver_params("bicycle", **kw)
        b = _rec().solver_params("bicycle", decel_limit="throttle", **kw)
        assert type(a).__name__ == type(b).__name__ == "_AlpaSimParams"
        assert dataclasses.asdict(a) == dataclasses.asdict(b)


@pytest.fixture(scope="module")
def deps():
    pytest.importorskip("casadi")
    pytest.importorskip("do_mpc")
    pytest.importorskip("osqp")


@pytest.mark.parametrize("name,kw", [("follower_nmpc_gear", {}),
                                     ("follower_nmpc_signed", {"command_mode": "signed"}),
                                     ("follower_lmpc_gear", {"mpc_impl": "linear"})])
def test_default_follower_solver_p_equals_the_2_0_6_capture(deps, name, kw):
    from genesis_vehicle.control import PathFollower
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = PathFollower(PATH, make_table(), mpc_params=_rec(), **kw)
    assert as_record(f._law.solver.p) == FIX[name]
    assert f.mpc_decel_limit == "throttle" and f.mpc_model == "bicycle"


def test_default_nmpc_is_built_with_build_bicycle(deps, monkeypatch):
    from genesis_vehicle.control import PathFollower
    from genesis_vehicle.control.mpc import alpasim_nonlinear as an
    from genesis_vehicle.control.mpc.models import build_bicycle
    seen = []
    orig = an.NonlinearMPC.__init__
    sig = inspect.signature(orig)

    def spy(self, *a, **k):
        b = sig.bind(self, *a, **k)
        b.apply_defaults()
        seen.append(b.arguments["model_builder"])
        orig(self, *a, **k)
    monkeypatch.setattr(an.NonlinearMPC, "__init__", spy)
    PathFollower(PATH, make_table(), mpc_params=_rec())
    assert seen == [build_bicycle]
    np.testing.assert_equal(len(seen), 1)
