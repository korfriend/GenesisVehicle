"""Equivalence guard for the v1.0.11 batched proxy sync.

In dual_scene mode every step mirrors each vehicle's chassis base pose onto
its raycast-scene proxy (the wheel-ray origin). Pre-1.0.11 this was a python
loop of per-vehicle ``set_pos``/``set_quat`` — each triggering a whole-scene
FK in the raycast scene (2·K FK passes; measured 29.8 ms at K=30, 80 % of
the L2-vs-L3 dual gap). ``VehicleScene._sync_proxies_batched`` replaces it
with one batched solver write + a single FK. This test pins the batched
path to the loop path: identical proxy poses for every vehicle.
"""

import os

import pytest

pytest.importorskip("genesis")

import torch
import genesis as gs

from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann

URDF = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    "..", "samples", "urdf", "car_4w.urdf"))


@pytest.fixture(scope="module")
def cpu_genesis():
    if not getattr(gs, "_initialized", False):
        try:
            VehicleScene.init_backend("cpu")
        except Exception as e:                     # pragma: no cover
            pytest.skip(f"genesis CPU backend unavailable: {e}")
    return gs


def test_batched_proxy_sync_matches_per_vehicle_loop(cpu_genesis):
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene", init_genesis=False)
    vs.add_ground_plane()
    vehs = [vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(4.0 * i, 0.0, 1.0))
            for i in range(3)]
    vs.build()

    # Move each chassis to a distinct pose so the sync has real work to copy.
    for i, veh in enumerate(vehs):
        veh.entity_main.set_pos(
            torch.tensor([4.0 * i, 0.7 * i, 1.0 + 0.1 * i]), relative=False)
        q = torch.tensor([1.0, 0.0, 0.0, 0.05 * i])
        veh.entity_main.set_quat(q / torch.linalg.norm(q), relative=False)

    # Reference: the per-vehicle loop path.
    for veh in vehs:
        veh._sync_proxy()
    ref = [(veh.proxy.get_pos().clone(), veh.proxy.get_quat().clone())
           for veh in vehs]

    # Batched path must land every proxy on the same pose.
    vs._sync_proxies_batched()
    for (rp, rq), veh in zip(ref, vehs):
        pp = veh.proxy.get_pos()
        qq = veh.proxy.get_quat()
        assert torch.allclose(rp, pp, atol=1e-5), f"pos {rp} != {pp}"
        assert (torch.allclose(rq, qq, atol=1e-5)
                or torch.allclose(rq, -qq, atol=1e-5)), f"quat {rq} != {qq}"


def test_batched_sync_includes_dynamic_obstacle_mirrors(cpu_genesis):
    """1.0.13: dynamic obstacles' raycast mirrors join the SAME batched write
    (+ the single FK) as the vehicle proxies — the per-obstacle _sync_dynamic
    loop cost ~1 ms/obstacle just like the proxies did."""
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene", init_genesis=False)
    vs.add_ground_plane()
    veh = vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(0.0, 0.0, 1.0))
    obs = [vs.add_dynamic(gs.morphs.Box(size=(1.0, 1.0, 1.0), pos=(8.0 + 3.0 * i, 0, 3.0)),
                          wheel_raycast=True, name=f"ramp_{i}")
           for i in range(2)]
    vs.build()

    # Distinct poses so the sync does real work.
    for i, o in enumerate(obs):
        o.entity_main.set_pos(torch.tensor([8.0 + 3.0 * i, 1.0 + i, 2.0]),
                              relative=False)

    # Reference: the per-body loops.
    veh._sync_proxy()
    for o in obs:
        vs._sync_dynamic(o)
    ref = [(o.entity_raycast.get_pos().clone(), o.entity_raycast.get_quat().clone())
           for o in obs]

    vs._proxy_sync_cache = None          # rebuild cache including mirrors
    vs._sync_proxies_batched()
    for (rp, rq), o in zip(ref, obs):
        pp = o.entity_raycast.get_pos()
        qq = o.entity_raycast.get_quat()
        assert torch.allclose(rp, pp, atol=1e-5), f"mirror pos {rp} != {pp}"
        assert (torch.allclose(rq, qq, atol=1e-5)
                or torch.allclose(rq, -qq, atol=1e-5))

    # And the step path must take the batched branch for this scene too.
    vs.step()
    assert vs._proxy_sync_ok is True


def test_batch_pose_reader_matches_entity_get_pos(cpu_genesis):
    """1.0.13: _BatchPoseReader must return exactly what entity.get_pos()/
    get_quat() (env 0) returned per entity — same solver read, one call."""
    from genesis_vehicle.server.physics_server import _BatchPoseReader
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene", init_genesis=False)
    vs.add_ground_plane()
    vehs = [vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(4.0 * i, 0.0, 1.0))
            for i in range(3)]
    vs.build()
    ents = [v.entity_main for v in vehs]
    reader = _BatchPoseReader(ents)
    pos_b, quat_b = reader.read()
    import numpy as np
    for i, e in enumerate(ents):
        p = e.get_pos(); q = e.get_quat()
        p = p[0] if p.dim() > 1 else p
        q = q[0] if q.dim() > 1 else q
        np.testing.assert_allclose(pos_b[i], p.cpu().numpy(), atol=1e-6)
        np.testing.assert_allclose(quat_b[i], q.cpu().numpy(), atol=1e-6)


def test_measure_distances_uses_batched_sync_without_fallback(cpu_genesis):
    """The step path must take the batched branch (health flag True) and not
    silently fall back to the per-vehicle loop."""
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene", init_genesis=False)
    vs.add_ground_plane()
    vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(0.0, 0.0, 1.0))
    vs.build()
    vs.step()
    assert vs._proxy_sync_ok is True
