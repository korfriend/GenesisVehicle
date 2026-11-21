"""MPCVehicleParams / solver_params / accel_limits — numpy only, no MPC deps.

Mutants this file must fail on (plan section 11): the time-constant floor at
``dt_ctrl`` removed from ``solver_params``; the 0 m/s deceleration rule
changed (e.g. the 0 m/s entry taken from the HIGHEST measured speed).
"""
import copy
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control.mpc import MPCVehicleParams, accel_limits, load_mpc_params  # noqa: E402
from genesis_vehicle.control.mpc.params import SCHEMA, REQUIRED_KEYS  # noqa: E402
from genesis_vehicle.tests.data.make_mpc_field_replay import PARAMS_SDK  # noqa: E402


def _d(**over):
    d = copy.deepcopy(PARAMS_SDK)
    d.update(over)
    return d


def test_json_round_trip(tmp_path):
    p = MPCVehicleParams.from_dict(_d())
    f = tmp_path / "car_mpc_params.json"
    p.to_json(f)
    q = MPCVehicleParams.from_json(f)
    assert q.to_dict() == p.to_dict()
    assert q.mass == 1330.0 and q.schema == SCHEMA
    assert load_mpc_params(str(f)).to_dict() == p.to_dict()
    assert load_mpc_params(p) is p


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        MPCVehicleParams.from_json("/nonexistent/x_mpc_params.json")


@pytest.mark.parametrize("key", ["mass", "steering", "accel_tau", "genesis_version"])
def test_missing_key_is_named(key):
    d = _d()
    del d[key]
    with pytest.raises(ValueError, match=key):
        MPCVehicleParams.from_dict(d)


def test_schema_mismatch_and_field_json_rejected():
    with pytest.raises(ValueError, match="mpc_params/2"):
        MPCVehicleParams.from_dict(_d(schema="genesis_vehicle.mpc_params/2"))
    field = {k: v for k, v in PARAMS_SDK.items() if k != "schema"}
    field["steer_cap"] = 1.0
    with pytest.raises(ValueError, match="identify"):
        MPCVehicleParams.from_dict(field)


@pytest.mark.parametrize("steering,n_axles", [("SkidSteer", 2), ("NoSteer", 2), ("Ackermann", 3)])
def test_non_ackermann_or_three_axles_rejected(steering, n_axles):
    with pytest.raises(ValueError, match='mpc_model="off"'):
        MPCVehicleParams.from_dict(_d(steering=steering, n_axles=n_axles))


def test_optional_keys_default_to_none():
    p = MPCVehicleParams.from_dict(_d())
    assert p.tyre is None and p.v_star is None


def test_solver_params_matches_the_field_rule():
    sp = MPCVehicleParams.from_dict(_d()).solver_params("bicycle", steer_cap=0.5, dt_ctrl=0.025)
    assert sp.mass == 1330.0 and sp.inertia == 2714.4
    assert sp.wheelbase == 2.7 and sp.l_rig_to_cg == 1.36
    assert sp.steer_max == 0.5 * 0.7
    # time constants floored at dt_ctrl: accel_tau 0.015 < 0.025, steer 0.08 kept
    assert sp.acceleration_time_constant == 0.025
    assert sp.steering_time_constant == 0.08
    assert sp.accel_max_tab == ([0.0, 3.0, 6.0, 10.0, 14.0], [1.9, 1.75, 1.63, 1.47, 1.29])
    # 0 m/s decel entry = the LOWEST measured speed's value
    assert sp.accel_min_tab == ([0.0, 3.0, 6.0, 10.0, 14.0], [-2.15, -2.15, -2.16, -2.18, -2.2])
    assert sp.accel_max == 1.29 and sp.accel_min == -2.15
    assert sp.kinematic_threshold_speed == 5.0


def test_tau_floor_applies_to_steering_too():
    sp = MPCVehicleParams.from_dict(_d(steer_yaw_tau=0.01)).solver_params(dt_ctrl=0.05)
    assert sp.steering_time_constant == 0.05


def test_accel_limits_interpolate_and_clamp_at_both_ends():
    sp = MPCVehicleParams.from_dict(_d()).solver_params()
    lo, hi = accel_limits(sp, 4.5)
    assert hi == pytest.approx(1.75 + (1.63 - 1.75) * 0.5)
    assert lo == pytest.approx(-2.15 + (-2.16 + 2.15) * 0.5)
    assert accel_limits(sp, -3.0) == (-2.15, 1.9)
    assert accel_limits(sp, 40.0) == (-2.2, 1.29)


def test_pacejka_is_not_in_2_0_0():
    with pytest.raises(ValueError, match="bicycle"):
        MPCVehicleParams.from_dict(_d()).solver_params("pacejka")


def test_json_file_is_plain_json(tmp_path):
    f = tmp_path / "p.json"
    MPCVehicleParams.from_dict(_d()).to_json(f)
    d = json.loads(f.read_text())
    assert set(REQUIRED_KEYS) <= set(d)
    assert np.isfinite(d["mass"])
