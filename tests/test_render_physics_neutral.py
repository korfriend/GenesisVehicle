"""Rendering a VehicleScene must not move the physics (2.0.6, AC2-N in-suite).

Through 2.0.5 the joint-writing wheel visuals (removed in 2.0.6) wrote the URDF
wheel joints into the rigid solver whenever a scene rendered at ``n_envs > 1``
or through an escape-hatch camera, and the chassis trajectory moved (car
+0.142 m). Since 2.0.6 the wheels are drawn by posing the engine's own wheel
nodes from the closed form at every render sync, and no SDK code writes solver
state for it. This file checks that claim the direct way: the same scene,
rendered and headless, in ONE process, compared with ``np.array_equal`` (no
tolerance) at every step on chassis position, quaternion, linear and angular
velocity, wheel spin rate and suspension normal force.

| abbr | meaning |
|---|---|
| AC2-N | the 2.0.6 acceptance gate "rendered == headless, bit for bit" |
| N1 / N2 / N8 / N10 | its rows: per-vehicle solver n_envs 1 / batched n_envs 2 / L3-style teleport reset at step 100 with viewer emulation / escape-hatch camera (``vs.scene.add_camera``) at n_envs 1 and 2 |
| EMUV | viewer emulation: a GEOMETRY-change subscriber forcing a render sync, plus one forced sync after every step (what an open viewer does) |
| K | vehicles in the scene (2 here) |

Scene: the acceptance scene of the 2.0.6 render harness with a 120-step prefix
of its 180-step schedule: ``tests/data/car_4w_redwheels.urdf`` (byte copy of the
harness URDF), ``car_4w_rwd_ackermann(stability="control")``, K = 2 at
y = 0 / -6, spawn z 1.0, dt 0.025 x 10 substeps, plane mu 1, dual_scene; 60
brake steps, then throttle 0.2 with steer 1.0 on the last env only; frames
rendered at 60 / 80 / 100 (200x150, fov 50, ambient light only). numpy's global
RNG is seeded 0 around scene construction and restored afterwards.

Measured once before this file was committed (implement agent, 2026-11-25, CPU/WSL2,
genesis-world 1.5.0): every row's in-process rendered arm is bit-identical to its
headless arm, and both equal the first 120 steps of the acceptance captures
(``.pm_r2/r206/s4r/H_S4_N{1,2,8,10a}``). Not version- or platform-gated: this is
the engine-bump net. It skips only when no offscreen rasterizer context can be
created, with the reason.

Wall time: about 30 s on the measuring machine (27.3 s in the suite run; 9 scene builds).
"""

import os

import numpy as np
import pytest

pytest.importorskip("genesis")

import torch
import genesis as gs

from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann

URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                    "car_4w_redwheels.urdf")
K = 2
STEPS = 120
FRAMES = (60, 80, 100)


@pytest.fixture(scope="module")
def cpu_genesis():
    if not getattr(gs, "_initialized", False):
        try:
            VehicleScene.init_backend("cpu")
        except Exception as e:                     # pragma: no cover
            pytest.skip(f"genesis CPU backend unavailable: {e}")
    return gs


@pytest.fixture(scope="module")
def offscreen(cpu_genesis):
    """Skip (with the reason) only when the engine itself cannot render
    offscreen: a bare ``gs.Scene`` with one camera, no SDK code involved."""
    try:
        sc = gs.Scene(show_viewer=False)
        sc.add_entity(gs.morphs.Plane())
        cam = sc.add_camera(res=(32, 24), GUI=False)
        sc.build()
        cam.render(rgb=True)
    except Exception as e:                         # pragma: no cover
        pytest.skip(f"no offscreen rasterizer context: {type(e).__name__}: {e}")
    return True


def _build(ne, *, render, solver="batched"):
    """``render``: None (headless), "cam" (``vs.add_camera``) or "esc"
    (``vs.scene.add_camera`` before build, the escape hatch)."""
    st = np.random.get_state()
    np.random.seed(0)
    try:
        vis = gs.options.VisOptions(
            segmentation_level="link", shadow=False, ambient_light=(1.0, 1.0, 1.0),
            background_color=(0.0, 0.0, 0.0), lights=[])
        vs = VehicleScene(dt=0.025, substeps=10, n_envs=ne, show_viewer=False,
                          vis_options=vis, solver=solver, raycast_mode="dual_scene",
                          wheel_render_mode="auto", init_genesis=False)
        vs.add_ground_plane(friction=1.0)
        cam = None
        kw = dict(pos=(8, 6, 6), lookat=(0, -3, 0.3), GUI=False, res=(200, 150), fov=50.0)
        if render == "cam":
            cam = vs.add_camera(**kw).entity
        elif render == "esc":
            cam = vs.scene.add_camera(**kw)
        cfg = car_4w_rwd_ackermann(URDF, stability="control")
        vehs = [vs.add_vehicle(URDF, car_4w_rwd_ackermann, cfg=cfg,
                               pos=(0.0, -6.0 * i, 1.0), name=f"v{i}",
                               material=gs.materials.Rigid(friction=1.0))
                for i in range(K)]
        vs.build()
    finally:
        np.random.set_state(st)
    return vs, vehs, cam


def _aim(vs, vehs, cam, ne):
    obs = ne - 1
    eoff = np.asarray(vs.scene.envs_offset) if ne > 1 else np.zeros((1, 3))
    c = np.mean([v.get_pos().reshape(ne, 3)[obs].cpu().numpy() for v in vehs], axis=0)
    c = c + (eoff[obs] if ne > 1 else 0)
    cam.set_pose(pos=c + np.array([7.0, 5.0, 6.0]), lookat=c + np.array([0.0, 0.0, 0.2]),
                 up=(0.0, 0.0, 1.0))


def _set_inputs(vehs, k, ne):
    for v in vehs:
        if k <= 60:
            v.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        else:
            st = 1.0 if k <= 140 else -1.0
            v.set_inputs(throttle=0.2, brake=0.0,
                         steer=torch.tensor([0.0] * (ne - 1) + [st]) if ne > 1 else st)


def _run(ne, *, render, solver="batched", l3reset=None, emuv=False):
    vs, vehs, cam = _build(ne, render=render, solver=solver)
    ctx = vs.scene.visualizer.context if render else None
    if emuv:
        from genesis.engine.state_change import StateChange, Subscriber
        vs.scene.sim.rigid_solver.changelog.subscribe(Subscriber(
            to=frozenset((StateChange.GEOMETRY,)),
            callback=lambda change, **ind: ctx.update(force_render=True)))
    rec = {n: [] for n in ("pos", "quat", "vel", "ang", "omega", "lastN")}
    for k in range(1, STEPS + 1):
        _set_inputs(vehs, k, ne)
        vs.step()
        if emuv:
            ctx.update(force_render=True)
        if k == l3reset:
            ent = vehs[0].entity_main
            p0 = torch.tensor([[0.0, 0.0, 1.0]] * ne)
            q0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * ne)
            ent.set_pos(p0 if ne > 1 else p0[0])
            ent.set_quat(q0 if ne > 1 else q0[0])
            ent.set_dofs_velocity(torch.zeros((ne, ent.n_dofs)) if ne > 1
                                  else torch.zeros(ent.n_dofs))
        for v in vehs:
            rec["pos"].append(v.get_pos().reshape(ne, 3).cpu().numpy())
            rec["quat"].append(v.get_quat().reshape(ne, 4).cpu().numpy())
            rec["vel"].append(v.get_vel().reshape(ne, 3).cpu().numpy())
            rec["ang"].append(v.entity_main.get_ang().reshape(ne, 3).cpu().numpy())
        if solver == "batched":
            om = [kd.omega.reshape(-1).cpu().numpy() for kd in vs.physics.kinds]
            ln = [kd._proto.last_N.reshape(-1).cpu().numpy() for kd in vs.physics.kinds]
        else:
            om = [v.physics.omega.reshape(-1).cpu().numpy() for v in vehs]
            ln = [v.physics.last_N.reshape(-1).cpu().numpy() for v in vehs]
        rec["omega"].append(np.concatenate(om))
        rec["lastN"].append(np.concatenate(ln))
        if render and k in FRAMES:
            _aim(vs, vehs, cam, ne)
            cam.render(rgb=True, depth=True, segmentation=True)
    if render:
        assert vs._wheel_renderer is not None, "premise: the scene draws its wheels"
    return {n: np.stack(x) for n, x in rec.items()}


def _assert_bit_identical(a, b):
    for n in a:
        assert a[n].shape == b[n].shape, n
        if not np.array_equal(a[n], b[n]):
            bad = [i for i in range(len(a[n])) if not np.array_equal(a[n][i], b[n][i])]
            d = float(np.abs(a[n].astype(np.float64) - b[n].astype(np.float64)).max())
            pytest.fail(f"rendered != headless in {n!r}: first differing record {bad[0]}, "
                        f"max |diff| {d:.3e}")


def test_n1_per_vehicle_n_envs_1(offscreen):
    _assert_bit_identical(_run(1, render="cam", solver="per_vehicle"),
                          _run(1, render=None, solver="per_vehicle"))


def test_n2_batched_n_envs_2_and_n10b_escape_hatch(offscreen):
    """N2 and N10 at n_envs 2 share the headless arm (as in the acceptance run)."""
    head = _run(2, render=None)
    _assert_bit_identical(_run(2, render="cam"), head)
    _assert_bit_identical(_run(2, render="esc"), head)


def test_n10a_escape_hatch_n_envs_1(offscreen):
    _assert_bit_identical(_run(1, render="esc"), _run(1, render=None))


def test_n8_teleport_reset_under_viewer_emulation(offscreen):
    _assert_bit_identical(_run(2, render="cam", l3reset=100, emuv=True),
                          _run(2, render=None, l3reset=100))
