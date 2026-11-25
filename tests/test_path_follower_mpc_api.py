"""PathFollower's 2.0.0 constructor surface and its check order (plan section 4.1).

Each check raises before any side effect, in this order: (1) path length,
(2) plant present, (3) ``mpc_model`` value, (4) no MPC-only argument with
``mpc_model="off"``, (5) a live vehicle that is not a 2-axle Ackermann,
(6) U1 — no ``mpc_params``, (7) ``mpc_impl`` / ``on_solve_failure`` values
(v2.1.0 also the ``mpc_decel_limit`` value), (8) optional dependencies, (9) the
params record (v2.1.0 also the brake row under ``"brake"``), then warnings (10) no
forward block, (11) live plant, (12) params vs vehicle mismatch.

Mutants this file must fail on (plan section 11): checks 5 and 6 swapped; the
``"off"`` + MPC-argument check removed; the ``FleetFollower`` MPC check removed
(or moved after the mutation loop); ``"pacejka"`` left in the 2.0.0 allowed
values. (The 2.1.0 ``mpc_decel_limit`` refusals and their placement are tested in
``test_mpc_decel_limit.py``.)
"""
import copy
import os
import sys
import types
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control import FleetFollower, PathFollower  # noqa: E402
from genesis_vehicle.control.mpc import MPCVehicleParams  # noqa: E402
from genesis_vehicle.control.path_follower import U1_MESSAGE  # noqa: E402
from genesis_vehicle.strategies.steering import Ackermann, SkidSteer  # noqa: E402
from genesis_vehicle.tests.data import make_mpc_field_replay as rp  # noqa: E402
from genesis_vehicle.tests.test_sweep_table import make_table  # noqa: E402

PATH = [(float(x), 0.0, 0.0, 3.0) for x in np.arange(0.0, 30.0, 0.5)]
REV = [(float(x), 0.0, 0.0, -1.0) for x in np.arange(0.0, 10.0, 0.5)]


def _params(**over):
    d = copy.deepcopy(rp.PARAMS_SDK)
    d.update(over)
    return MPCVehicleParams.from_dict(d)


class _FakeLivePlant:
    """Looks like a DifferentiablePlant to PathFollower: a coupled solve, a
    ``src`` with ``resolved`` / ``wheel_meta``, ``mass_props``."""
    has_coupled_solve = True

    def __init__(self, steering, axles, mass=1330.0):
        wheels = [types.SimpleNamespace(axle_index=a, pb_y=8.0, pc_y=1.3, pe_y=0.4,
                                        mu_lat=1.0, mu_long=1.0) for a in axles]
        self.src = types.SimpleNamespace(
            resolved=types.SimpleNamespace(steering=steering, wheels=wheels), wheel_meta=None)
        self.mass_props = types.SimpleNamespace(mass=mass)
        self.steer_range = (-1.0, 1.0)

    def solve(self, v, a, w, pitch_deg=0.0, roll_deg=0.0):
        return 0.0, 0.0

    def throttle_for(self, *a):
        return 0.0

    def steer_for(self, *a):
        return 0.0


# --- (1)-(4): order and the "off" path ---------------------------------------------

def test_u1_message_names_both_ways_out():
    with pytest.raises(ValueError) as ei:
        PathFollower(PATH, make_table())
    msg = str(ei.value)
    assert msg == U1_MESSAGE
    assert "mpc_params=" in msg and 'mpc_model="off"' in msg
    assert "python -m genesis_vehicle.control.identify" in msg


@pytest.mark.parametrize("plant", ["no/such/table.csv", "make_table", "live"])
def test_u1_for_every_plant_type(plant):
    p = {"no/such/table.csv": "no/such/table.csv", "make_table": make_table(),
         "live": _FakeLivePlant(Ackermann(max_steer_rad=0.7), [0, 0, 1, 1])}[plant]
    with pytest.raises(ValueError, match="mpc_params="):
        PathFollower(PATH, p)


def test_order_path_and_plant_come_first():
    with pytest.raises(ValueError, match="at least 2"):
        PathFollower([(0, 0, 0, 1.0)], make_table())
    with pytest.raises(ValueError, match="inverse plant"):
        PathFollower(PATH)


def test_error_precedence_change_under_the_default():
    # a bad plant object / malformed waypoints now report U1 first ...
    with pytest.raises(ValueError, match="mpc_params="):
        PathFollower(PATH, object())
    with pytest.raises(ValueError, match="mpc_params="):
        PathFollower([(0, 0, 0), (1, 0, 0)], make_table())
    # ... and keep their 1.x errors with mpc_model="off"
    with pytest.raises(TypeError):
        PathFollower(PATH, object(), mpc_model="off")
    with pytest.raises(ValueError, match="length 3"):
        PathFollower([(0, 0, 0), (1, 0, 0)], make_table(), mpc_model="off")


def test_mpc_model_values_of_2_0_0():
    for bad in ("pacejka", "PD", None):
        with pytest.raises(ValueError, match=r"\['bicycle', 'off'\]"):
            PathFollower(PATH, make_table(), mpc_model=bad)


def test_feedforward_and_correction_are_not_in_2_0_0():
    with pytest.raises(TypeError):
        PathFollower(PATH, make_table(), mpc_model="off", mpc_feedforward="field")
    with pytest.raises(TypeError):
        PathFollower(PATH, make_table(), mpc_model="off", mpc_correction=(0.5, 0.5))


@pytest.mark.parametrize("kw", [dict(mpc_params="x.json"), dict(mpc_impl="nonlinear"),
                                dict(on_solve_failure="pd"), dict(mpc_gains=None),
                                dict(mpc_v_floor=0.5), dict(dt_ctrl=0.025),
                                dict(mpc_decel_limit="throttle"), dict(mpc_decel_limit="brake")])
def test_off_rejects_every_mpc_argument_even_at_its_default(kw):
    with pytest.raises(ValueError, match=list(kw)[0]):
        PathFollower(PATH, make_table(), mpc_model="off", **kw)


def test_signature_shows_effective_defaults():
    import inspect
    sig = inspect.signature(PathFollower)
    assert repr(sig.parameters["mpc_impl"].default) == "'nonlinear'"
    assert repr(sig.parameters["mpc_decel_limit"].default) == "'throttle'"     # v2.1.0
    assert sig.parameters["mpc_model"].default == "bicycle"


# --- (5) before (6): the live-vehicle steering check ---------------------------------

@pytest.mark.parametrize("steering,axles", [(SkidSteer(), [0, 0, 1, 1]),
                                            (Ackermann(max_steer_rad=0.7), [0, 0, 1, 1, 2, 2]),
                                            (Ackermann(max_steer_rad=0.7), [0, None, 1, 1])])
def test_live_non_ackermann_gets_the_ackermann_message_not_u1(steering, axles):
    with pytest.raises(ValueError) as ei:
        PathFollower(PATH, _FakeLivePlant(steering, axles))
    assert "2-axle Ackermann" in str(ei.value) and 'mpc_model="off"' in str(ei.value)
    assert "mpc_params=" not in str(ei.value)


# --- (7) (9): values and the params record -------------------------------------------

def test_bad_impl_and_policy():
    with pytest.raises(ValueError, match="mpc_impl"):
        PathFollower(PATH, make_table(), mpc_params=_params(), mpc_impl="quadratic")
    with pytest.raises(ValueError, match="on_solve_failure"):
        PathFollower(PATH, make_table(), mpc_params=_params(), on_solve_failure="ignore")


def test_params_file_errors(tmp_path):
    pytest.importorskip("casadi")
    pytest.importorskip("do_mpc")
    with pytest.raises(FileNotFoundError):
        PathFollower(PATH, make_table(), mpc_params=str(tmp_path / "missing.json"))
    f = tmp_path / "skid.json"
    d = copy.deepcopy(rp.PARAMS_SDK)
    d["steering"] = "SkidSteer"
    import json
    f.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="2-axle Ackermann"):
        PathFollower(PATH, make_table(), mpc_params=str(f))


# --- an MPC follower (needs the deps) --------------------------------------------------

@pytest.fixture(scope="module")
def deps():
    pytest.importorskip("casadi")
    pytest.importorskip("do_mpc")


def test_no_forward_block_warns(deps):
    with pytest.warns(UserWarning, match="never runs"):
        PathFollower(REV, make_table(), mpc_params=_params())


def test_live_plant_warns_and_cross_checks(deps):
    plant = _FakeLivePlant(Ackermann(max_steer_rad=0.7), [0, 0, 1, 1], mass=1500.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        PathFollower(PATH, plant, mpc_params=_params(), command_mode="signed")
    msgs = [str(x.message) for x in w]
    assert any("live-vehicle plant" in m for m in msgs)
    assert any("mass: params 1330.0 vs vehicle 1500.0" in m for m in msgs)


def test_off_follower_adds_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PathFollower(PATH, _FakeLivePlant(SkidSteer(), [0, 1]), mpc_model="off")


def test_forward_block_needs_yaw_rate_and_v_lat(deps):
    f = PathFollower(PATH, make_table(), mpc_params=_params(), command_mode="signed")
    with pytest.raises(TypeError, match="extract_state"):
        f.step((0.0, 0.0), 0.0, 0.0, 0.0, 0.0)
    out = f.step((0.0, 0.0), 0.0, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0)
    assert len(out) == 3 and f.last_mode in ("MPC", "MPC-HOLD")
    assert f.mpc_stats["n_solve"] == 1


def test_first_step_builds_no_nlp(deps, monkeypatch):
    from genesis_vehicle.control.mpc import alpasim_nonlinear as an
    calls = []
    real = an.NonlinearMPC._setup_mpc
    monkeypatch.setattr(an.NonlinearMPC, "_setup_mpc",
                        lambda self: (calls.append(1), real(self))[1])
    f = PathFollower(PATH, make_table(), mpc_params=_params())
    assert calls == [1]
    f.step((0.0, 0.0), 0.0, 0.0, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0)
    assert calls == [1]


def test_off_follower_attributes():
    f = PathFollower(PATH, make_table(), mpc_model="off")
    assert f.mpc_model == "off" and f.mpc_stats is None and f.last_plan is None
    f.step((0.0, 0.0), 0.0, 0.0, 0.0, 0.0, yaw_rate=None, t=3.0)   # extras ignored
    assert f.regime == "R-OFF"


# --- FleetFollower ---------------------------------------------------------------------

def test_fleet_rejects_mpc_followers_before_touching_any(deps):
    plant = _FakeLivePlant(Ackermann(max_steer_rad=0.7), [0, 0, 1, 1])
    plant.n_members = 2
    off = PathFollower(PATH, make_table(), mpc_model="off")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        on = PathFollower(PATH, make_table(), mpc_params=_params())
    before = (off.plant, off.plant_member, off._coupled_plant)
    with pytest.raises(TypeError, match='mpc_model="off"'):
        FleetFollower([off, on], plant)
    assert (off.plant, off.plant_member, off._coupled_plant) == before


def test_followerplan_new_fields_are_appended():
    from dataclasses import fields
    from genesis_vehicle.control import FollowerPlan
    names = [f.name for f in fields(FollowerPlan)]
    assert names[:8] == ["mode", "a_target", "omega_target", "v_long", "v_target",
                         "pitch_deg", "roll_deg", "command"]
    assert names[8:] == ["regime", "a_ff", "w_ff", "gear", "a_coast", "steer_cmd"]
    p = FollowerPlan("DRV+1", 1.0, 0.5, 2.0, 3.0, 0.0, 0.0, None)    # 1.x positional
    assert p.regime == "R-OFF"


# --- the bicycle's live cross-check warning (moved here in 2.1.0) --

_TYRE_F = {"axle": 0, "pb_y": 8.0, "pc_y": 1.3, "pe_y": 0.4, "mu_lat": 1.0, "mu_long": 1.0,
           "lat_scale": 0.9, "eps_v": 0.7, "n_wheels": 2}
_TYRE_R = {"axle": 1, "pb_y": 9.5, "pc_y": 1.45, "pe_y": -0.2, "mu_lat": 0.95, "mu_long": 1.0,
           "lat_scale": 0.9, "eps_v": 0.7, "n_wheels": 2}


def _tyre_record():
    d = copy.deepcopy(rp.PARAMS_SDK)
    d["tyre"] = {"model": "PacejkaAnisotropic", "axles": [copy.deepcopy(_TYRE_F), copy.deepcopy(_TYRE_R)]}
    d["axle_loads"] = {"settled": {"0": 7400.0, "1": 5650.0}, "front_axle": 0, "rear_axle": 1}
    return MPCVehicleParams.from_dict(d)


class _LivePlant:
    """A DifferentiablePlant look-alike with a resolved config (``tire`` + per-wheel coefficients)."""
    has_coupled_solve = True

    def __init__(self, tire=None, wheel_over=None):
        wheel_over = wheel_over or {}
        from genesis_vehicle.tire_models.pacejka import PacejkaAnisotropic
        wheels = []
        for ax, t in ((0, _TYRE_F), (0, _TYRE_F), (1, _TYRE_R), (1, _TYRE_R)):
            w = {k: t[k] for k in ("pb_y", "pc_y", "pe_y", "mu_lat", "mu_long")}
            w.update(wheel_over.get(ax, {}))
            wheels.append(types.SimpleNamespace(axle_index=ax, **w))
        self.src = types.SimpleNamespace(resolved=types.SimpleNamespace(
            steering=Ackermann(max_steer_rad=0.7), wheels=wheels,
            tire=tire if tire is not None else PacejkaAnisotropic(eps_v=0.7, lat_scale=0.9)), wheel_meta=None)
        self.mass_props = types.SimpleNamespace(mass=1330.0)
        self.steer_range = (-1.0, 1.0)

    def solve(self, *a, **k):
        return 0.0, 0.0

    def throttle_for(self, *a):
        return 0.0

    def steer_for(self, *a):
        return 0.0


def test_bicycle_live_mismatch_warning_text_is_unchanged(deps):
    """The bicycle's live cross-check WARNS with the 2.0.6 text, byte for byte."""
    plant = _LivePlant(wheel_over={1: {"mu_lat": 0.57}})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        PathFollower(PATH, plant, mpc_params=_tyre_record(), command_mode="signed")
    msgs = [str(x.message) for x in w]
    assert "mpc_params do not describe this vehicle: axle 1 mu_lat: params 0.95 vs vehicle 0.57" in msgs
