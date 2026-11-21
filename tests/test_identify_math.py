"""The Genesis-free parts of the identify tool (control/identify.py).

Mutants this file must fail on (plan section 11): ``RESIDUAL_WARN`` 3 % ->
30 %; ``s`` dropped from the ``v*`` formula.
"""
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control import identify as idf  # noqa: E402
from genesis_vehicle.strategies.steering import Ackermann, PartialAckermann, SkidSteer  # noqa: E402

M, IZZ, LF, LR, MAXS, SIGN = 1330.0, 2714.4, 1.34, 1.36, 0.7, -1.0
CAF, CAR = 33200.0, 32500.0


def _samples(scale_r=1.0):
    out = []
    for v, s in idf.CORNER:
        d = SIGN * s * MAXS
        vy, r = idf.steady_state_bicycle(v, d, CAF, CAR, M, IZZ, LF, LR)
        out.append([v, vy, r * scale_r, s])
    return out


def test_fit_recovers_the_stiffnesses_of_a_synthetic_bicycle():
    caf, car = idf.fit_cornering_stiffness(_samples(), M, LF, LR, MAXS, SIGN)
    assert caf == pytest.approx(CAF, rel=1e-6) and car == pytest.approx(CAR, rel=1e-6)
    res = idf.cornering_fit_residual(_samples(), caf, car, M, IZZ, LF, LR, MAXS, SIGN)
    assert res["yaw_rate_rel_max"] < 1e-6 and not res["warn"]


@pytest.mark.parametrize("err,warn", [(0.029, False), (0.031, True)])
def test_residual_warning_threshold_is_3_percent(err, warn):
    # measured r = predicted / (1 + err): relative residual exactly err
    s = _samples(scale_r=1.0 / (1.0 + err))
    res = idf.cornering_fit_residual(s, CAF, CAR, M, IZZ, LF, LR, MAXS, SIGN)
    assert res["yaw_rate_rel_max"] == pytest.approx(err, rel=1e-9)
    assert res["warn"] is warn
    assert res["speed_range"] == [4.0, 12.0]


def test_t63():
    dt = 0.025
    y = 1.0 - np.exp(-np.arange(1, 200) * dt / 0.1)
    assert idf.t63(y, dt) == pytest.approx(0.1 * -np.log(0.37), abs=2e-3)
    assert np.isnan(idf.t63(np.zeros(10), dt))


def test_v_star_of_the_reference_car():
    ax = [{"pb_y": 8.0, "pc_y": 1.3, "mu_lat": 1.0, "lat_scale": 1.0}] * 2
    assert idf.lateral_deadbeat_speed(ax, 0.856, 0.025) == pytest.approx(2.98, abs=0.005)
    assert idf.lateral_deadbeat_speed(ax, 1.0, 0.025) == pytest.approx(2.55, abs=0.005)


def test_hold_throttle():
    assert idf.hold_throttle(6.0, 6.0) == pytest.approx(0.08)
    assert idf.hold_throttle(10.0, 0.0) == 1.0


def _resolved(steering, axles):
    return types.SimpleNamespace(steering=steering,
                                 wheels=[types.SimpleNamespace(axle_index=a) for a in axles])


@pytest.mark.parametrize("steering,axles", [(SkidSteer(), [0, 0, 1, 1]),
                                            (Ackermann(max_steer_rad=0.7), [0, 0, 1, 1, 2, 2]),
                                            (Ackermann(max_steer_rad=0.7), [0, None, 1, 1])])
def test_non_identifiable_vehicles_raise_naming_off(steering, axles):
    with pytest.raises(ValueError, match='mpc_model=\\\\"off\\\\"|mpc_model="off"'):
        idf.check_identifiable(_resolved(steering, axles))


def test_identifiable():
    assert idf.check_identifiable(_resolved(Ackermann(max_steer_rad=0.7), [0, 0, 1, 1])) == ("Ackermann", 2)
    assert idf.check_identifiable(_resolved(PartialAckermann(max_steer_rad=0.7, steered_axles=(0,)), [0, 0, 1, 1]))[1] == 2
