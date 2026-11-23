"""2.0.4 Step K: the spring cap's driver side.

``VehiclePhysics._ensure_static_mass_share`` (the ``_ensure_yaw_inertia``
pattern with its own guard, plan r204 condition C2 / api-review condition 6)
on fake drivers, then two real CPU scenes: the reference car at dt 0.025
(uncapped, silent) and at dt 0.04 (capped, one report — not repeated by a
``DifferentiablePlant`` built on it, api-review condition 10).

| abbr | meaning |
|---|---|
| SFL | `StaticFrictionLock` |
| m_static | per-wheel static mass share, kg (`WheelMeta.static_mass_share`) |
"""

from __future__ import annotations

import logging
import os
import types

import pytest
import torch

from genesis_vehicle.config import resolve
from genesis_vehicle.control.plant import PlantMassProperties
from genesis_vehicle.core import VehiclePhysics, build_wheel_meta
from genesis_vehicle.presets import car_4w_rwd_ackermann

HERE = os.path.dirname(__file__)
CAR_URDF = os.path.abspath(os.path.join(HERE, "..", "samples", "urdf", "car_4w.urdf"))
DEV, FDT = torch.device("cpu"), torch.float32
CAR_MP = PlantMassProperties(1330.0, (0.010150376119111714, 0.0, 0.4804511), 1.0)


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, rec):
        msg = rec.getMessage()
        if "[genesis_vehicle:sfl-cap]" in msg:
            self.lines.append(msg)


@pytest.fixture
def caplines():
    h = _Capture()
    lg = logging.getLogger("genesis_vehicle")
    lg.addHandler(h)
    try:
        yield h.lines
    finally:
        lg.removeHandler(h)


def _fake(mass_props=None, entity=None, yaw_preset=True):
    wm = build_wheel_meta(resolve(car_4w_rwd_ackermann(CAR_URDF, stability="control")), DEV, FDT)
    if yaw_preset:
        wm.yaw_inertia = object()        # documented preset path
    return types.SimpleNamespace(wheel_meta=wm, _mass_props=mass_props,
                                 entity=entity if entity is not None else object(),
                                 vehicle_name="fake_car")


def test_static_share_from_carried_mass_props(caplines):
    d = _fake(mass_props=CAR_MP)
    VehiclePhysics._ensure_static_mass_share(d)
    assert [round(float(v), 1) for v in d.wheel_meta.static_mass_share] == [335.0, 335.0, 330.0, 330.0]
    assert d.wheel_meta.static_mass_share.dtype == FDT
    assert caplines == []


def test_static_share_skipped_when_already_set():
    d = _fake(mass_props=CAR_MP)
    pre = torch.tensor([1.0, 2.0, 3.0, 4.0])
    d.wheel_meta.static_mass_share = pre
    VehiclePhysics._ensure_static_mass_share(d)
    assert d.wheel_meta.static_mass_share is pre


def test_no_links_with_preset_yaw_inertia_never_raises(caplines):
    """C2: an entity without links whose yaw inertia was preset gets no static
    share (uncapped) and ONE log line naming the vehicle, not an exception."""
    d = _fake(mass_props=None, entity=types.SimpleNamespace())
    VehiclePhysics._ensure_static_mass_share(d)
    assert d.wheel_meta.static_mass_share is None
    assert len(caplines) == 1 and "fake_car" in caplines[0] and "uncapped" in caplines[0]
    assert d._mass_props is None                  # nothing stored on the driver
    VehiclePhysics._ensure_static_mass_share(d)   # the next step: attempted once only
    assert len(caplines) == 1


def test_massless_mass_props_never_raise(caplines):
    d = _fake(mass_props=PlantMassProperties(float("nan"), (0.0, 0.0, 0.0), 1.0))
    VehiclePhysics._ensure_static_mass_share(d)
    assert d.wheel_meta.static_mass_share is None
    assert len(caplines) == 1
    VehiclePhysics._ensure_static_mass_share(d)
    assert len(caplines) == 1


def test_fallback_logs_once_and_sets_equal_shares(caplines):
    d = _fake(mass_props=PlantMassProperties(1330.0, (3.0, 0.0, 0.5), 1.0))
    VehiclePhysics._ensure_static_mass_share(d)
    assert [float(v) for v in d.wheel_meta.static_mass_share] == [332.5] * 4
    # (Step D) This stub's I_cg = 1.0 - 1330 * 3.0^2 < 0, so the rigid-body
    # factor's own guard logs a second line; the share is still set.
    assert len(caplines) == 2 and "equal shares" in caplines[0]
    assert "rigid-body factor unavailable" in caplines[1]
    assert d.wheel_meta.static_rigid_factor is None
    VehiclePhysics._ensure_static_mass_share(d)   # set now: no further line
    assert len(caplines) == 2


# --- real scenes ---------------------------------------------------------------

def _car_scene(dt):
    from genesis_vehicle import VehicleScene
    import genesis as gs
    VehicleScene.init_backend("cpu")
    vs = VehicleScene(dt=dt, substeps=4, n_envs=1, show_viewer=False, solver="per_vehicle")
    vs.add_ground_plane(friction=1.0)
    veh = vs.add_vehicle(CAR_URDF, car_4w_rwd_ackermann, pos=(0.0, 0.0, 0.5),
                         stability="control", material=gs.materials.Rigid(friction=1.0))
    vs.build()
    return vs, veh


def test_reference_car_dt_0025_is_uncapped_and_silent(caplines):
    vs, veh = _car_scene(0.025)
    for _ in range(10):                            # spawn drop under brake 1.0
        veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        vs.step()
    wm = veh.physics.wheel_meta
    assert [round(float(v), 1) for v in wm.static_mass_share] == [335.0, 335.0, 330.0, 330.0]
    assert caplines == []
    from genesis_vehicle._hotset import derived_value
    from genesis_vehicle.strategies.stability import StaticFrictionLock
    sfl = [h for h in veh.physics.resolved.stability_hooks if isinstance(h, StaticFrictionLock)]
    assert sfl and derived_value(sfl[0], "_k_eff") is sfl[0].k_spring


def test_reference_car_dt_004_reports_once_also_with_a_plant(caplines):
    vs, veh = _car_scene(0.04)
    for _ in range(10):
        veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        vs.step()
    assert len(caplines) == 1
    assert "209375" in caplines[0] and "206250" in caplines[0]
    from genesis_vehicle import DifferentiablePlant
    plant = DifferentiablePlant(veh)
    plant.solve(0.0, 0.5, 0.0)
    plant.predict(0.2, 0.0)
    assert len(caplines) == 1


def test_batched_kind_computes_the_share_on_its_proto(caplines):
    """MultiVehicleKindPhysics.step computes the share on the kind's proto
    (entities[0]); at dt 0.04 the reference car's cap binds and reports once."""
    from genesis_vehicle import VehicleScene
    import genesis as gs
    VehicleScene.init_backend("cpu")
    vs = VehicleScene(dt=0.04, substeps=4, n_envs=1, show_viewer=False, solver="batched")
    vs.add_ground_plane(friction=1.0)
    mat = gs.materials.Rigid(friction=1.0)
    a = vs.add_vehicle(CAR_URDF, car_4w_rwd_ackermann, pos=(0.0, 0.0, 0.5), stability="control",
                       material=mat, name="ba")
    vs.add_vehicle(CAR_URDF, car_4w_rwd_ackermann, pos=(0.0, 6.0, 0.5), stability="control",
                   material=mat, name="bb")
    vs.build()
    assert a.physics is None                       # really the batched driver
    for _ in range(5):
        for v in vs.vehicles if hasattr(vs, "vehicles") else []:
            v.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        vs.step()
    proto = vs._mvp.kinds[0]._proto
    assert [round(float(v), 1) for v in proto.wheel_meta.static_mass_share] == [335.0, 335.0, 330.0, 330.0]
    assert len(caplines) == 1 and "209375" in caplines[0]
