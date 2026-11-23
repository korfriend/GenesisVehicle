"""2.0.4 Step C (plan r204 D9): the OSC server's generic-mapping vehicles get the
"control" stability hook set, StaticFrictionLock included (through 2.0.3:
RollingResistance + LowSpeedRegularizer only, so they crept on any slope at
every brake). Genesis-free (``build_cfg`` parses the URDF; no scene is built).

| abbr | meaning |
|---|---|
| SFL | `StaticFrictionLock` |
| L2 / L3 | server per-entity mode (`build_vehicle`) / multi-env mode (`l3_runtime`) |
"""
from __future__ import annotations

import inspect
import os

import pytest

pytest.importorskip("genesis")

HERE = os.path.dirname(os.path.abspath(__file__))
CAR = os.path.abspath(os.path.join(HERE, "..", "samples", "urdf", "car_4w.urdf"))
GENERIC = {"driveType": 3, "drivetrainStrategy": 1, "maxTorque": 1000.0, "maxBrake": 2500.0,
           "drivingJoints": [{"jointName": "rear_left_wheel_joint"}, {"jointName": "rear_right_wheel_joint"}],
           "steeringJoints": [{"jointName": "front_left_steer_joint"}, {"jointName": "front_right_steer_joint"}]}


def _fresh():
    from genesis_vehicle.server import vehicle_builder
    vehicle_builder._cfg_cache.clear()
    return vehicle_builder


def test_generic_mapping_gets_the_control_hook_set():
    vb = _fresh()
    from genesis_vehicle.presets import stability_hooks_for_profile
    from genesis_vehicle.strategies.stability import StaticFrictionLock
    cfg = vb.build_cfg(CAR, dict(GENERIC), 1.0, target_id=0)
    got = [type(h).__name__ for h in cfg.stability_hooks]
    want = [type(h).__name__ for h in stability_hooks_for_profile("control", "car")]
    assert got == want == ["RollingResistance", "LowSpeedRegularizer", "StaticFrictionLock"]
    sfl = [h for h in cfg.stability_hooks if isinstance(h, StaticFrictionLock)][0]
    assert (sfl.v_thr, sfl.k_spring, sfl.k_damp) == (0.5, 500_000.0, 20_000.0)


def test_l2_and_l3_share_one_cfg_object():
    """Both server modes go through ``build_cfg``; equal (URDF, mapping,
    friction) give the SAME cfg object, so L2 and L3 get the same hook list."""
    vb = _fresh()
    a = vb.build_cfg(CAR, dict(GENERIC), 1.0, target_id=0)
    b = vb.build_cfg(CAR, dict(GENERIC), 1.0, target_id="L3-shared")
    assert a is b
    from genesis_vehicle.server import l3_runtime
    assert "vehicle_builder.build_cfg(" in inspect.getsource(l3_runtime)
    assert "build_cfg(" in inspect.getsource(vb.build_vehicle)
