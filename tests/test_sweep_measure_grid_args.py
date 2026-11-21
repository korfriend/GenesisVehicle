"""sweep_measure grid flags (v2.0.0) and the extracted plant-override helpers.

Mutant this file must fail on (plan section 11): the "throttle/steer list must
contain 0" check removed.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control import sweep_measure as sm  # noqa: E402

URDF = os.path.join(os.path.dirname(__file__), "..", "samples", "urdf", "car_4w.urdf")
BASE = ["--urdf", URDF, "--preset", "car_4w_rwd_ackermann", "--output", "/nonexistent/out.csv"]


def _grid(*extra):
    return sm.grid_from_args(sm.build_arg_parser().parse_args(BASE + list(extra)))


def test_default_grid_unchanged():
    g = _grid()
    assert g == (sm.V_INIT_LIST, sm.THROTTLE_LIST, sm.STEER_LIST, sm.PITCH_DEG_LIST, sm.ROLL_DEG_LIST)
    assert len(sm.build_combo_arrays(*g)[0]) == 17325


def test_list_flags_replace_axes():
    g = _grid("--v-list=-2,0,2,4,16", "--throttle-list=-1,0,1", "--steer-list=0,0.5,-0.5",
              "--pitch-list=0", "--roll-list=0")
    assert g == ([-2.0, 0.0, 2.0, 4.0, 16.0], [-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5], [0.0], [0.0])
    assert _grid("--quick", "--v-list=1,2")[0] == [1.0, 2.0]


@pytest.mark.parametrize("flag", ["--throttle-list=-1,1", "--steer-list=0.5,1"])
def test_missing_zero_raises_before_measuring(flag, monkeypatch):
    import genesis_vehicle
    monkeypatch.setattr(genesis_vehicle.VehicleScene, "init_backend",
                        staticmethod(lambda *a, **k: pytest.fail("measurement started")))
    with pytest.raises(ValueError, match="must contain 0"):
        sm.main(BASE + [flag])


def test_bad_number():
    with pytest.raises(ValueError, match="comma-separated"):
        _grid("--v-list=1,x")


def test_signatures_kept():
    import inspect
    assert list(inspect.signature(sm.build_combo_arrays).parameters) == [
        "v_list", "thr_list", "steer_list", "pitch_list", "roll_list"]
    assert list(inspect.signature(sm.run_sweep).parameters)[:10] == [
        "urdf_path", "preset_fn", "config_mod", "n_envs", "v_arr", "t_arr", "s_arr", "p_arr",
        "r_arr", "device"]


def test_plant_override_helpers():
    args = sm.build_arg_parser().parse_args(BASE + ["--mu-lat", "0.6"])
    fn = sm.plant_overrides_fn(args, URDF)
    assert fn.overrides == {"mu_lat": 0.6}
    assert sm.plant_overrides_fn(sm.build_arg_parser().parse_args(BASE), URDF) is None
