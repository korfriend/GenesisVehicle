"""The 2.0.5 joint-writing wheel visuals are gone, loudly (2.0.6, AC2-M in-suite).

``WheelJointInternalSync`` / ``KindVisualBatch`` wrote the URDF wheel joints into
the rigid solver (``set_dofs_position`` / ``control_dofs_position`` /
``set_dofs_kp`` / ``set_dofs_kv``), which moved the chassis trajectory; the
n_envs 1 instanced renderer hid the URDF wheel visuals through the engine-private
``active_envs_mask`` / ``active_envs_idx``. 2.0.6 removes both. One test per row
of the removal-policy table, plus two spies on a rendered n_envs 2 scene.

| abbr | meaning |
|---|---|
| AC2-M | the 2.0.6 acceptance gate for the removed visual path |
| VJS | the removed joint-writing wheel visuals |
| SDK frame | (R9a) the IMMEDIATE calling frame -- the frame executing an attribute assignment, or the frame that called a spied setter -- whose code file is under the ``genesis_vehicle`` package and not under ``genesis_vehicle/tests/``; no stack walk |
| R9a | the 2026-11-25 plan-review ruling fixing that definition (a stack walk counted the engine's own constructor defaults inside SDK ``add_entity`` calls) |
| K | vehicles in one scene |

Spy rule (R9a): spies are installed (``monkeypatch.setattr``) on the class
``__setattr__`` of ``RigidGeom`` / ``RigidVisGeom`` and on the four setters of
``RigidEntity`` / ``RigidSolver`` before ``VehicleScene`` is constructed; mask
writes are counted over the whole scene lifecycle, setter calls over the 10
rendered steps. Under R9a an engine method's internal call to a spied method
never counts, so an SDK call to ``RigidEntity.set_dofs_position`` counts once.
Validity of the spies on a tree that DOES write lives in the acceptance harness
(2.0.5: 16 mask writes at ``_gs_compat.py:423/425``, 10 solver calls at
``visual.py:527``; 0 / 0 on 2.0.6). In-suite, a positive control runs code
compiled under a filename inside the package (not ``tests/``) that assigns the
mask and calls the solver setter, and a negative control checks that the
engine's own constructor assignment, reached from the test file, counts 0.
**Not covered: the server's instance patch** (``server/vehicle_builder.py``
``apply_monkey_patches``) is outside this scene.

Log assertions reset the once-per-process latch
``config._VISUAL_SUSP_MODE_WARNED`` first and attach to the ``genesis_vehicle``
logger directly (it does not propagate), so they do not depend on test order.

Measured on 2.0.6 (implement agent, 2026-11-25, CPU/WSL2, genesis-world 1.5.0):
0 SDK mask writes against 38 engine constructor writes the spy saw, 0 SDK setter
calls in the window; a mutant that writes one wheel joint through
``RigidSolver.set_dofs_position`` from code compiled under ``vehicle_scene.py``
before every step is counted 10 times.

Wall time: about 17 s on the measuring machine (16.0 s in the suite run); not
platform-gated (no pixel or stored-capture assert).
"""

import logging
import os
import sys

import numpy as np
import pytest

pytest.importorskip("genesis")

import torch
import genesis as gs

import genesis_vehicle
from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann
from genesis_vehicle import config as CFG
from genesis_vehicle.config import ConfigError, JOINT_SYNC_REMOVED_MSG

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.abspath(genesis_vehicle.__file__)) + os.sep
TESTS = os.path.join(PKG, "tests") + os.sep
URDF = os.path.join(HERE, "data", "car_4w_redwheels.urdf")
CAR = os.path.join(PKG, "samples", "urdf", "car_4w.urdf")
TANK = os.path.join(HERE, "data", "tank_ref.urdf")
K = 2
SUSP_TAG = "[genesis_vehicle:visual-susp-mode]"


@pytest.fixture(scope="module")
def cpu_genesis():
    if not getattr(gs, "_initialized", False):
        try:
            VehicleScene.init_backend("cpu")
        except Exception as e:                     # pragma: no cover
            pytest.skip(f"genesis CPU backend unavailable: {e}")
    return gs


@pytest.fixture
def gv_caplog(caplog, monkeypatch):
    """caplog on the SDK logger (``propagate = False``), with the
    visual_susp_mode latch reset (R8c)."""
    monkeypatch.setattr(CFG, "_VISUAL_SUSP_MODE_WARNED", False)
    logger = logging.getLogger("genesis_vehicle")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="genesis_vehicle"):
            yield caplog
    finally:
        logger.removeHandler(caplog.handler)


def _susp_warnings(caplog):
    return [r for r in caplog.records
            if r.levelno == logging.WARNING and SUSP_TAG in r.getMessage()]


def _headless(cfg, *, ne=2, solver="batched"):
    vs = VehicleScene(dt=0.025, substeps=10, n_envs=ne, show_viewer=False,
                      solver=solver, raycast_mode="dual_scene", init_genesis=False)
    vs.add_ground_plane(friction=1.0)
    vehs = [vs.add_vehicle(URDF, car_4w_rwd_ackermann, cfg=cfg,
                           pos=(0.0, -6.0 * i, 1.0), name=f"v{i}",
                           material=gs.materials.Rigid(friction=1.0))
            for i in range(K)]
    vs.build()
    return vs, vehs


def _drive(vs, vehs, steps):
    for _ in range(steps):
        for v in vehs:
            v.set_inputs(throttle=0.3, brake=0.0, steer=0.2)
        vs.step()


def _physics(vs, vehs):
    out = [v.get_pos().cpu().numpy() for v in vehs]
    out += [v.get_quat().cpu().numpy() for v in vehs]
    out += [v.get_vel().cpu().numpy() for v in vehs]
    out += [k.omega.cpu().numpy() for k in vs.physics.kinds]
    out += [k._proto.last_N.cpu().numpy() for k in vs.physics.kinds]
    return out


# --- removal-policy table rows -------------------------------------------------

def test_removed_classes_raise_import_error_in_both_forms():
    from genesis_vehicle import visual as V
    want = {"WheelJointInternalSync": "VehiclePhysics.wheel_visual_transforms()",
            "KindVisualBatch": "MultiVehiclePhysics.wheel_visual_transforms()"}
    for name, repl in want.items():
        with pytest.raises(ImportError) as e1:
            exec(f"from genesis_vehicle.visual import {name}", {})
        with pytest.raises(ImportError) as e2:
            getattr(V, name)
        for e in (e1, e2):
            msg = str(e.value)
            assert msg.startswith(f"{name} was removed in 2.0.6: it wrote the URDF wheel joints")
            assert "moved the chassis trajectory" in msg and repl in msg


def test_joint_sync_flag_true_raises_config_error_in_resolve():
    cfg = car_4w_rwd_ackermann(CAR)
    cfg.enable_wheel_joint_internal_sync = True
    with pytest.raises(ConfigError) as e:
        CFG.resolve(cfg)
    assert str(e.value) == JOINT_SYNC_REMOVED_MSG
    assert JOINT_SYNC_REMOVED_MSG.startswith(
        "enable_wheel_joint_internal_sync=True was removed in 2.0.6:")
    cfg.enable_wheel_joint_internal_sync = False
    CFG.resolve(cfg)                               # False: silent


def test_joint_sync_flag_true_raises_config_error_in_kind_init(monkeypatch):
    """``MultiVehicleKindPhysics.__init__`` checks the VehicleConfig before it
    builds anything (no Genesis object needed past the fan-size check)."""
    from genesis_vehicle import multi_vehicle as MV
    monkeypatch.setattr(MV, "check_fan_uniformity", lambda sensors: 1)
    cfg = car_4w_rwd_ackermann(CAR)
    cfg.enable_wheel_joint_internal_sync = True
    with pytest.raises(ConfigError) as e:
        MV.MultiVehicleKindPhysics(None, [object()], [object()], cfg)
    assert str(e.value) == JOINT_SYNC_REMOVED_MSG
    assert e.traceback[-1].path.name == "multi_vehicle.py"


def test_joint_sync_flag_true_raises_config_error_in_vehicle_physics_init(cpu_genesis, monkeypatch):
    """``VehiclePhysics.__init__`` re-checks the RESOLVED object (covers a
    ResolvedConfig mutated after resolve): ``core.resolve`` is wrapped to set
    the flag on its result, so the raise can only come from core.py."""
    from genesis_vehicle import core
    real = core.resolve

    def resolve_then_mutate(config):
        r = real(config)
        r.enable_wheel_joint_internal_sync = True
        return r
    monkeypatch.setattr(core, "resolve", resolve_then_mutate)
    with pytest.raises(ConfigError) as e:
        _headless(car_4w_rwd_ackermann(URDF, stability="control"), ne=1)
    assert str(e.value) == JOINT_SYNC_REMOVED_MSG
    assert e.traceback[-1].path.name == "core.py"


def test_internal_sync_render_mode_raises_value_error(cpu_genesis):
    with pytest.raises(ValueError) as e:
        VehicleScene(n_envs=2, show_viewer=False, wheel_render_mode="internal_sync",
                     init_genesis=False)
    assert str(e.value).startswith("wheel_render_mode='internal_sync' was removed in 2.0.6:")
    assert "Remove the argument" in str(e.value)
    # "instanced" no longer raises at n_envs > 1 (it means "auto").
    vs = VehicleScene(n_envs=2, show_viewer=False, wheel_render_mode="instanced",
                      init_genesis=False)
    assert vs.wheel_render_mode == "instanced"


def test_visual_susp_mode_is_validated_and_inert(cpu_genesis, gv_caplog):
    """'control' logs exactly one WARNING and the physics is bit-equal to
    'auto' over 60 steps (K 2, n_envs 2, headless); 'auto' logs none; an
    invalid value still raises."""
    bad = car_4w_rwd_ackermann(CAR)
    bad.visual_susp_mode = "foo"
    with pytest.raises(ConfigError, match="visual_susp_mode must be"):
        CFG.resolve(bad)

    vs_a, vehs_a = _headless(car_4w_rwd_ackermann(URDF, stability="control"))
    _drive(vs_a, vehs_a, 60)
    assert _susp_warnings(gv_caplog) == []
    cfg_c = car_4w_rwd_ackermann(URDF, stability="control")
    cfg_c.visual_susp_mode = "control"
    vs_c, vehs_c = _headless(cfg_c)
    _drive(vs_c, vehs_c, 60)
    w = _susp_warnings(gv_caplog)
    assert len(w) == 1, [r.getMessage() for r in w]
    assert "visual_susp_mode='control' has no effect since 2.0.6" in w[0].getMessage()
    for a, c in zip(_physics(vs_a, vehs_a), _physics(vs_c, vehs_c)):
        assert np.array_equal(a, c)


def test_presets_and_server_builder_emit_no_visual_susp_mode_warning(gv_caplog):
    from genesis_vehicle import presets as P
    from genesis_vehicle.server import vehicle_builder as vb
    cfgs = [P.car_4w_rwd_ackermann(CAR), P.car_4w_fwd_ackermann(CAR),
            P.car_4w_awd_ackermann(CAR), P.truck_6w_partial_ackermann(CAR),
            P.tank_skid_belt(TANK)]
    vb._cfg_cache.clear()
    mapping = {"driveType": 3, "drivetrainStrategy": 1, "maxTorque": 1000.0, "maxBrake": 2500.0,
               "drivingJoints": [{"jointName": "rear_left_wheel_joint"},
                                 {"jointName": "rear_right_wheel_joint"}],
               "steeringJoints": [{"jointName": "front_left_steer_joint"},
                                  {"jointName": "front_right_steer_joint"}]}
    cfgs.append(vb.build_cfg(CAR, dict(mapping), 1.0, target_id=0))
    cfgs.append(vb.build_cfg(CAR, dict(mapping, driveType=0), 1.0, target_id=1))
    vb._cfg_cache.clear()
    for c in cfgs:
        assert c.visual_susp_mode == "auto"
        CFG.resolve(c)
    assert _susp_warnings(gv_caplog) == []


def test_visual_attributes_and_slot_snapshot(cpu_genesis):
    """``VehiclePhysics.visual`` is None and read-only; ``kind.visuals == ()``;
    a 2.0.5-style ``export_slot`` dict (R8c(3): a 2.0.6 dict plus the six
    visual keys with non-None tensors of 2.0.5 shape) imports, and the slot's
    physics state equals an import without those keys."""
    from genesis_vehicle.core import VehiclePhysics
    vs, vehs = _headless(car_4w_rwd_ackermann(URDF, stability="control"))
    _drive(vs, vehs, 30)
    kind = vs.physics.kinds[0]
    assert isinstance(kind._proto, VehiclePhysics)
    assert kind._proto.visual is None
    with pytest.raises(AttributeError):
        kind._proto.visual = object()
    assert kind.visuals == ()

    snap = kind.export_slot(0)
    for key in ("visual_angle", "visual_susp_set_prev", "visual_susp_ctrl_prev",
                "batch_angle", "batch_susp_set_prev", "batch_susp_ctrl_prev"):
        assert key not in snap
    ne, nw = kind.n_envs, int(kind._proto.wheel_meta.n_wheels)
    old = dict(snap,
               visual_angle=torch.full((ne, nw), 0.7),
               visual_susp_set_prev=torch.full((ne, nw), 0.05),
               visual_susp_ctrl_prev=torch.full((ne, nw), 0.05),
               batch_angle=torch.full((ne, nw), 0.7),
               batch_susp_set_prev=torch.full((ne, kind.K, nw), 0.05),
               batch_susp_ctrl_prev=torch.full((ne, kind.K, nw), 0.05))
    _drive(vs, vehs, 10)                           # move the slot away from the snapshot
    kind.import_slot(0, snap)
    plain = kind.export_slot(0)
    _drive(vs, vehs, 10)
    kind.import_slot(0, old)
    with_keys = kind.export_slot(0)
    assert set(plain) == set(with_keys)
    for key in plain:
        a, b = plain[key], with_keys[key]
        if torch.is_tensor(a):
            assert torch.equal(a, b), key
        else:
            assert a == b, key


def test_private_visual_shims_are_gone():
    from genesis_vehicle import _gs_compat as C
    from genesis_vehicle import visual as V
    for name in ("hide_vgeoms", "queue_instance_poses"):
        assert not hasattr(C, name), name
    for name in ("_susp_visual_target", "_SUSP_VIS_KP", "_SUSP_VIS_KV",
                 "_SUSP_VIS_MAX_RATE", "_warn_perf_once"):
        assert not hasattr(V, name), name


# --- spies (R9a) -------------------------------------------------------------

def _sdk_frame(f):
    fn = os.path.abspath(f.f_code.co_filename)
    return fn.startswith(PKG) and not fn.startswith(TESTS)


@pytest.fixture
def spies(cpu_genesis, monkeypatch):
    from genesis.engine.entities.rigid_entity import RigidEntity, rigid_geom
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver
    S = dict(window=False, setter=[], setter_raw=0, mask=[], mask_raw=0)
    for cls in (RigidEntity, RigidSolver):
        for name in ("set_dofs_position", "control_dofs_position", "set_dofs_kp", "set_dofs_kv"):
            orig = getattr(cls, name)

            def mk(orig, label):
                def spy(*a, **kw):
                    f = sys._getframe(1)           # the frame that called the spied method
                    if S["window"]:
                        S["setter_raw"] += 1
                        if _sdk_frame(f):
                            S["setter"].append((label, f.f_code.co_filename, f.f_lineno))
                    return orig(*a, **kw)
                return spy
            monkeypatch.setattr(cls, name, mk(orig, f"{cls.__name__}.{name}"))
    for gcls in (rigid_geom.RigidGeom, rigid_geom.RigidVisGeom):
        orig_sa = gcls.__setattr__

        def mk2(orig_sa):
            def sa(self, name, value):
                if name in ("active_envs_mask", "active_envs_idx"):
                    f = sys._getframe(1)           # the frame executing the assignment
                    S["mask_raw"] += 1
                    if _sdk_frame(f):
                        S["mask"].append((name, f.f_code.co_filename, f.f_lineno))
                return orig_sa(self, name, value)
            return sa
        monkeypatch.setattr(gcls, "__setattr__", mk2(orig_sa))
    return S


_POSITIVE_SRC = """
def run(geom, solver, qpos, dofs):
    geom.active_envs_mask = geom.active_envs_mask
    solver.set_dofs_position(qpos, dofs)
"""


@pytest.fixture
def offscreen(cpu_genesis):
    """Skip (with the reason) only when the engine itself cannot render
    offscreen: a bare ``gs.Scene`` with one camera, no SDK code involved.
    Requested BEFORE ``spies`` so the probe scene is not counted."""
    try:
        sc = gs.Scene(show_viewer=False)
        sc.add_entity(gs.morphs.Plane())
        cam = sc.add_camera(res=(32, 24), GUI=False)
        sc.build()
        cam.render(rgb=True)
    except Exception as e:                         # pragma: no cover
        pytest.skip(f"no offscreen rasterizer context: {type(e).__name__}: {e}")
    return True


def test_rendered_n_envs_2_scene_makes_no_solver_or_mask_write_from_sdk_code(offscreen, spies):
    S = spies
    # Negative control: the engine's own constructor assignment, reached from the
    # test file through add_entity, counts 0 (and the spy did see it).
    sc = gs.Scene(show_viewer=False)
    sc.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.5)))
    assert S["mask_raw"] >= 2, "negative control: the engine constructor assignment was not seen"
    assert S["mask"] == [], S["mask"]
    S["mask_raw"] = 0

    # The AC2-M scene: rendered n_envs 2 (both envs), K 2, render every step, 10 steps.
    st = np.random.get_state()
    np.random.seed(0)
    try:
        vis = gs.options.VisOptions(segmentation_level="link", shadow=False,
                                    ambient_light=(1.0, 1.0, 1.0),
                                    background_color=(0.0, 0.0, 0.0), lights=[])
        vs = VehicleScene(dt=0.025, substeps=10, n_envs=2, show_viewer=False,
                          vis_options=vis, raycast_mode="dual_scene", init_genesis=False)
        vs.add_ground_plane(friction=1.0)
        cam = vs.add_camera(pos=(8, 6, 6), lookat=(0, -3, 0.3), GUI=False,
                            res=(200, 150), fov=50.0).entity
        cfg = car_4w_rwd_ackermann(URDF, stability="control")
        vehs = [vs.add_vehicle(URDF, car_4w_rwd_ackermann, cfg=cfg,
                               pos=(0.0, -6.0 * i, 1.0), name=f"v{i}",
                               material=gs.materials.Rigid(friction=1.0))
                for i in range(K)]
        vs.build()
    finally:
        np.random.set_state(st)
    assert vs._wheel_renderer is not None, "premise: the scene draws its wheels"
    assert S["mask_raw"] > 0, "premise: the mask spy saw the engine's constructor writes"
    S["window"] = True
    for _ in range(10):
        for v in vehs:
            v.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        vs.step()
        cam.render(rgb=True, depth=True, segmentation=True)
    S["window"] = False
    assert S["setter"] == [], f"SDK solver setter calls during rendered steps: {S['setter'][:5]}"
    assert S["mask"] == [], f"SDK active_envs_mask/idx writes: {S['mask'][:5]}"

    # Positive control: code compiled under a package (non-tests) filename.
    ns = {}
    exec(compile(_POSITIVE_SRC, os.path.join(PKG, "_ac2m_positive_control.py"), "exec"), ns)
    ent = vehs[0].entity_main
    geom = ent.links[0].geoms[0]
    qpos = ent.get_dofs_position(dofs_idx_local=[0])
    S["window"] = True
    ns["run"](geom, vs.scene.sim.rigid_solver, qpos, [ent.dof_start])
    S["window"] = False
    assert len(S["mask"]) >= 1 and len(S["setter"]) >= 1, S
