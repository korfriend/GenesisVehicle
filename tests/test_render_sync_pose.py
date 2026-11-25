"""Every wheel node the engine draws carries the closed-form pose AT FLUSH (2.0.6,
AC2-S in-suite; the C21a flush gate).

Since 2.0.6 ``VehicleScene`` draws the wheels by overwriting the poses of the
engine's own wheel nodes from the closed form (``wheel_visual_transforms``)
inside every render sync of ``scene.visualizer.context``
(``_gs_compat.on_render_sync``). The SDK's per-sync ordering guard
(``InstancedWheelRenderer._check_last_writes``) sees an engine pose write that
moved after the hook only ONE SYNC LATE, does not check the hooked sync that
follows one that did not complete (checking resumes on the one after), and
does NOT see an engine ``"model"`` buffer queued after the hook without
touching ``primitive.poses``. That case is caught only here: a test-side
wrapper on ``ctx.jit.flush_buffer``
compares what the flush is about to upload (the queued buffer, else
``primitive.poses``) against the closed form, for every wheel node and rendered
env. **Run this file on every genesis-world bump.**
(A non-finite closed-form pose is not written: that wheel instance keeps its
last SDK-written row; not exercised here, see ``tests/test_render_eno_unit.py``.
The flush gate itself fails on a non-finite pose difference.)

| abbr | meaning |
|---|---|
| gate 1 | max-abs entry of the 3x4 difference (drawn pose - closed form o T_local); mixes metres and rotation entries, so it is quoted without a unit |
| gate 5 | every wheel part has exactly one drawn node (engine node or SDK fallback) |
| T_local | a geom's pose in its link frame (``init_pos`` / ``init_quat``) |
| EMUV / REC | viewer emulation (setter subscriber + one forced sync per step) / recording emulation (a camera render inside ``Scene.step`` at listed steps) |
| C1 (i)/(ii)/(iii) | render right after ``build()`` / ``mark_config_dirty()`` rebuild (hook wrap depth stays 1) / env-1 reset mid-run |
| C21a | the test-side gate-1 check at ``ctx.jit.flush_buffer`` |
| L2 | n_envs 2 with ``rendered_envs_idx=[1]`` |

The expected node set comes from the ENGINE model (``ctx.nodes_by_seg_keys``
per wheel geom), not from the renderer; fallback nodes from the renderer. The
closed form is read from the live driver; the 4x4 assembly uses the SDK's
``visual._pose_mats`` / ``visual._local_T`` (the acceptance harness does the
same, so an error inside those two helpers is not caught here).

Scenes: prefixes of the 2.0.6 acceptance scenes (``tests/data/car_4w_redwheels.urdf``,
K = 2, dt 0.025 x 10, 60 brake steps then throttle 0.2 / steer on the last env).
Measured once before commit (implement agent, 2026-11-25, CPU/WSL2,
genesis-world 1.5.0): every gate-1 value logged here equals, entry for entry, the
acceptance harness value at the same step and site (under ``.pm_r2/r206/``:
``s5r/SEG_L2_link``, ``s5r/R2d_TELE``, ``s5r/C1iii_batched``, and
``s8r/insuite/N9_ref``, a 100-step harness capture of the ``s4r/R_S4_N9`` scene,
whose file holds no per-entry log), max 2.98e-8; and two mutants fail these
gates (no pose override: gate-1 error 0.730 on the L2 scene; env-0 rows in every
instance: 0.771 on the env-1 reset scene). Not version- or platform-gated
(engine-bump net); skips only when no offscreen rasterizer context can be
created, with the reason.

Wall time: about 20 s on the measuring machine (18.5 s in the suite run).
"""

import os

import numpy as np
import pytest

pytest.importorskip("genesis")

import torch
import genesis as gs

from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann
from genesis_vehicle import _gs_compat as C
from genesis_vehicle import visual as V

URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                    "car_4w_redwheels.urdf")
K = 2
TOL = 1e-6


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


class Probe:
    """Builds one acceptance scene and logs gate 1 / gate 5 at every flush and
    every sync (``log``: dicts with k, where, err, err_env, g5bad, depth)."""

    def __init__(self, ne, *, ridx=None, frames=(), rec=False, emuv=False,
                 monkeypatch=None):
        self.ne, self.frames, self.k = ne, tuple(frames), 0
        self.log = []
        st = np.random.get_state()
        np.random.seed(0)
        try:
            vk = dict(segmentation_level="link", shadow=False,
                      ambient_light=(1.0, 1.0, 1.0), background_color=(0.0, 0.0, 0.0),
                      lights=[])
            if ridx is not None:
                vk["rendered_envs_idx"] = ridx
            vs = VehicleScene(dt=0.025, substeps=10, n_envs=ne, show_viewer=False,
                              vis_options=gs.options.VisOptions(**vk),
                              raycast_mode="dual_scene", init_genesis=False)
            vs.add_ground_plane(friction=1.0)
            self.cam = vs.add_camera(pos=(8, 6, 6), lookat=(0, -3, 0.3), GUI=False,
                                     res=(200, 150), fov=50.0).entity
            self.cfg = car_4w_rwd_ackermann(URDF, stability="control")
            self.vehs = [vs.add_vehicle(URDF, car_4w_rwd_ackermann, cfg=self.cfg,
                                        pos=(0.0, -6.0 * i, 1.0), name=f"v{i}",
                                        material=gs.materials.Rigid(friction=1.0))
                         for i in range(K)]
            vs.build()
        finally:
            np.random.set_state(st)
        self.vs = vs
        self.r = vs._wheel_renderer
        assert self.r is not None, "premise: the scene draws its wheels"
        self.ctx = ctx = vs.scene.visualizer.context
        self.ridx = list(ctx.rendered_envs_idx)
        flush = ctx.jit.flush_buffer

        def flush_spy(*a, **kw):                   # C21a
            self.gate1("flush")
            return flush(*a, **kw)
        ctx.jit.flush_buffer = flush_spy
        upd = ctx.update

        def upd_spy(force_render=False):
            t0 = ctx._t
            out = upd(force_render=force_render)
            if ctx._t != t0 or force_render:
                self.gate1("sync")
            return out
        ctx.update = upd_spy
        self.gate1("postbuild")
        if emuv:
            from genesis.engine.state_change import StateChange, Subscriber
            vs.scene.sim.rigid_solver.changelog.subscribe(Subscriber(
                to=frozenset((StateChange.GEOMETRY,)),
                callback=lambda change, **ind: ctx.update(force_render=True)))
        self.emuv = emuv
        if rec:
            from genesis.vis.camera import Camera
            probe = self

            def fake_rec(cam_self):
                if cam_self is probe.cam and probe.k in probe.frames:
                    probe.shoot()
            monkeypatch.setattr(Camera, "update_recording", fake_rec)

    # -- expected poses (engine model + live closed form) -----------------
    def _closed(self, vi):
        p, q = self.vehs[vi].wheel_visual_transforms("world")
        n = len(self.cfg.wheels)
        p = p.detach().cpu().double().numpy().reshape(-1, 3)
        q = q.detach().cpu().double().numpy().reshape(-1, 4)
        return V._pose_mats(p, q, np.float64).reshape(-1, n, 4, 4)

    def _expected_nodes(self):
        out, fb = [], {}
        for u in self.r._units:
            for (k, i), node in (u["fallback"] or {}).items():
                fb.setdefault(id(u["entities"][k]), {})[i] = node
        for vi, v in enumerate(self.vehs):
            ent = v.entity_main
            for i, w in enumerate(self.cfg.wheels):
                link = ent.get_link(w.name)
                parts = list(link.vgeoms) if ent.surface.vis_mode is None else list(link.geoms)
                eng = [n for g in list(link.vgeoms) + list(link.geoms)
                       for n in self.ctx.nodes_by_seg_keys.get(g.seg_keys, ())]
                for g in parts:
                    for node in self.ctx.nodes_by_seg_keys.get(g.seg_keys, ()):
                        out.append((vi, i, node, V._local_T(g)))
                f = fb.get(id(ent), {}).get(i)
                if f is not None:
                    out.append((vi, i, f, None))
                if bool(eng) == (f is not None):   # none, or both kinds
                    out.append((vi, i, None, "g5bad"))
        return out

    def gate1(self, where):
        cl = [self._closed(vi)[self.ridx] for vi in range(K)]
        err, g5bad, nn = 0.0, 0, 0
        err_env = [0.0] * len(self.ridx)
        for vi, i, node, Tl in self._expected_nodes():
            if node is None:
                g5bad += 1
                continue
            want = cl[vi][:, i] if Tl is None else cl[vi][:, i] @ Tl
            for prim in node.mesh.primitives:
                qb = self.ctx.jit._buffer.get((prim, "model"))
                got = (np.asarray(qb, dtype=np.float64).transpose(0, 2, 1) if qb is not None
                       else np.asarray(prim.poses, dtype=np.float64))
                if got.shape != want.shape:
                    err = float("inf")
                    continue
                de = np.abs(got[:, :3, :] - want[:, :3, :]).reshape(len(self.ridx), -1).max(axis=1)
                # A NaN on either side must FAIL this gate: Python's max(0.0, nan)
                # is 0.0, which would let a non-finite drawn pose or closed form
                # through (the renderer freezes such a wheel without a log).
                de = np.where(np.isfinite(de), de, np.inf)
                err_env = [max(a, float(b)) for a, b in zip(err_env, de)]
                err = max(err, float(de.max()))
                nn += 1
        self.log.append(dict(k=self.k, where=where, err=err, err_env=err_env, n=nn,
                             g5bad=g5bad, depth=C.render_sync_depth(self.ctx)))

    # -- schedule ------------------------------------------------------------
    def shoot(self):
        ne, obs = self.ne, self.ridx[-1]
        eoff = np.asarray(self.vs.scene.envs_offset) if ne > 1 else np.zeros((1, 3))
        c = np.mean([v.get_pos().reshape(ne, 3)[obs].cpu().numpy() for v in self.vehs], axis=0)
        c = c + (eoff[obs] if ne > 1 else 0)
        self.cam.set_pose(pos=c + np.array([7.0, 5.0, 6.0]),
                          lookat=c + np.array([0.0, 0.0, 0.2]), up=(0.0, 0.0, 1.0))
        rgb, depth, seg, _ = self.cam.render(rgb=True, depth=True, segmentation=True)
        assert np.isfinite(np.asarray(depth, dtype=np.float64)).any()

    def run(self, steps, *, render_frames=True, render_every=False, on_step=None):
        if render_frames and 0 in self.frames:
            self.shoot()                           # C1 (i)
        for k in range(1, steps + 1):
            self.k = k
            for v in self.vehs:
                if k <= 60:
                    v.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
                else:
                    st = 1.0 if k <= 140 else -1.0
                    v.set_inputs(throttle=0.2, brake=0.0,
                                 steer=(torch.tensor([0.0] * (self.ne - 1) + [st])
                                        if self.ne > 1 else st))
            self.vs.step()
            if self.emuv:
                self.ctx.update(force_render=True)
            if on_step is not None:
                on_step(self, k)
            if (render_frames and k in self.frames) or render_every:
                self.shoot()
        return self

    def assert_gates(self):
        assert self.log, "no gate-1 record"
        bad = [x for x in self.log if not x["err"] <= TOL]
        assert not bad, f"gate-1 error > {TOL}: {bad[:3]}"
        assert max(x["g5bad"] for x in self.log) == 0, "a wheel part without exactly one drawn node"
        assert all(x["n"] > 0 for x in self.log), "no wheel node checked"


def test_c1_render_after_build_and_rebuild_keeps_one_hook_l2(offscreen):
    """L2 (n_envs 2, env 1 rendered): frame 0 right after build (C1 (i)),
    renders at 60 / 100, ``mark_config_dirty()`` after step 100 and a render at
    101 after the rebuild (C1 (ii)): gate 1 <= 1e-6 at every flush and sync,
    hook wrap depth 1 throughout."""
    def reb(p, k):
        if k == 100:
            p.vs.mark_config_dirty()
    p = Probe(2, ridx=[1], frames=(0, 60, 100, 101)).run(101, on_step=reb)
    p.assert_gates()
    post = [x for x in p.log if x["where"] == "postbuild"]
    assert len(post) == 1 and post[0]["err"] <= TOL
    assert any(x["k"] == 101 and x["where"] == "flush" for x in p.log), "no draw after the rebuild"
    assert {x["depth"] for x in p.log} == {1}


def test_viewer_and_recording_syncs(offscreen, monkeypatch):
    """n_envs 2, both envs rendered, EMUV (a forced sync after every step) and
    REC (renders inside ``Scene.step`` at 60 / 80 / 100), plus the post-step
    renders at the same steps."""
    p = Probe(2, frames=(60, 80, 100), rec=True, emuv=True, monkeypatch=monkeypatch)
    p.run(100)
    p.assert_gates()
    assert sum(x["where"] == "sync" for x in p.log) >= 100
    assert sum(x["where"] == "flush" and x["k"] in (60, 80, 100) for x in p.log) >= 3


def test_teleport_setter_sync(offscreen):
    """n_envs 2 with EMUV; car 1 teleported +0.5 m (``set_pos`` + zero dof
    velocity, the server's style) after step 100, rendered right after."""
    def tele(p, k):
        if k == 100:
            ent = p.vehs[0].entity_main
            pos = ent.get_pos().clone().reshape(p.ne, 3)
            pos[:, 0] += 0.5
            ent.set_pos(pos)
            ent.set_dofs_velocity(torch.zeros((p.ne, ent.n_dofs)))
            p.shoot()
    p = Probe(2, frames=(60, 80, 100), emuv=True).run(100, on_step=tele)
    p.assert_gates()
    assert any(x["k"] == 100 and x["where"] == "flush" for x in p.log)


def test_c1_env1_reset_batched(offscreen):
    """C1 (iii): ``kind.reset(rows=<env 1's rows>)`` after step 100, render
    every step: env 0 gate 1 <= 1e-6, env 1 finite (measured <= 1e-6 too)."""
    def reset(p, k):
        if k == 100:
            for kind in p.vs.physics.kinds:
                kind.reset(rows=[1 * kind.K + s for s in range(kind.K)])
    p = Probe(2).run(105, render_frames=False, render_every=True, on_step=reset)
    after = [x for x in p.log if x["k"] >= 100 and x["where"] != "postbuild"]
    assert after
    assert max(x["err_env"][0] for x in after) <= TOL
    assert all(np.isfinite(x["err_env"][1]) for x in after)
    assert max(x["g5bad"] for x in p.log) == 0
    assert max(x["err"] for x in p.log if x["k"] < 100) <= TOL
