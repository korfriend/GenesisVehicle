"""The bicycle prediction model (``models.build_bicycle``) against its numpy
transcription of the field coefficients.

1000 random states/inputs over both branches (kinematic below the threshold,
dynamic above), relative 1e-12; and the branch switches at the params'
``kinematic_threshold_speed``, not at a hard-coded 5.0.

Mutants this file must fail on (plan section 11): the factor 2 (per-wheel
stiffness doubled to an axle) dropped from ``b_00`` in ``build_bicycle``;
the threshold hard-coded to 5.0. (The Pacejka rows of plan section 11 are
2.1.0.)
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("casadi")
pytest.importorskip("do_mpc")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control.mpc._deps import casadi_numpy_mode  # noqa: E402
from genesis_vehicle.control.mpc.models import bicycle_rhs_numpy, build_bicycle  # noqa: E402
from genesis_vehicle.tests._mpc_fakes import solver_params  # noqa: E402


def _rhs_fun(params):
    with casadi_numpy_mode(-1):
        m = build_bicycle(params)

    def f(x, u):
        out = m._rhs_fun(np.asarray(x, float), np.asarray(u, float), np.zeros(m.n_z),
                         np.zeros(m.n_tvp), np.zeros(m.n_p), np.zeros(m.n_w))
        return np.array(out).ravel()
    return f


def test_bicycle_rhs_matches_the_field_transcription():
    p = solver_params()
    f = _rhs_fun(p)
    rng = np.random.default_rng(1234)
    n_kin = 0
    for _ in range(1000):
        x = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-1.5, 1.5),
                      rng.uniform(0.0, 15.0), rng.uniform(-1, 1), rng.uniform(-1, 1),
                      rng.uniform(-0.35, 0.35), rng.uniform(-3, 3)])
        u = np.array([rng.uniform(-0.35, 0.35), rng.uniform(-3, 3)])
        got, want = f(x, u), bicycle_rhs_numpy(x, u, p)
        n_kin += x[3] < p.kinematic_threshold_speed
        assert np.allclose(got, want, rtol=1e-12, atol=1e-12), (x, u, got - want)
    assert 200 < n_kin < 800                          # both branches exercised


def test_branch_threshold_comes_from_the_params():
    p = solver_params()
    p.kinematic_threshold_speed = 3.0
    f = _rhs_fun(p)
    x = np.array([0.0, 0.0, 0.0, 4.0, 0.1, 0.2, 0.05, 0.0])
    u = np.array([0.05, 0.0])
    got = f(x, u)
    assert np.allclose(got, bicycle_rhs_numpy(x, u, p), rtol=1e-12, atol=1e-12)
    # at 4 m/s with a 3 m/s threshold the DYNAMIC branch runs: v_cg_y' differs
    # from the kinematic pull 10 * (v * delta * lr / L - v_y)
    kin = 10.0 * (4.0 * 0.05 * p.l_rig_to_cg / p.wheelbase - 0.1)
    assert abs(got[4] - kin) > 1e-3
