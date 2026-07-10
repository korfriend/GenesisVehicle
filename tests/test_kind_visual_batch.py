"""Equivalence guards for the v1.0.15 batching of audit items #9/#10.

#10 ``KindVisualBatch``: K same-kind WheelJointInternalSync writers collapse into ONE
solver-level ``set_dofs_position`` (one solver entry + FK instead of K). This
test pins the batched writer to the per-entity loop: identical joint dof
positions for every entity.

#9 cross-kind I/O: a multi-kind ``MultiVehiclePhysics.step`` pre-reads every
kind's chassis state in one batched solver read and applies all kinds' forces
in one combined call pair. Sanity-pinned here (multi-kind scene builds, steps,
drives, no NaN, and the combined index cache exists).
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


def test_kind_visual_batch_matches_per_entity_loop(cpu_genesis):
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene", init_genesis=False,
                      wheel_render_mode="internal_sync")  # this test targets VJS itself
    vs.add_ground_plane()
    for i in range(2):
        vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(5.0 * i, 0.0, 1.0))
    # Force the renders path headless so WheelJointInternalSync (and its batch) build.
    vs.show_viewer = True
    vs.build()
    kind = vs.physics.kinds[0]
    assert kind.visuals, "WheelJointInternalSync not built — renders hack failed"
    assert kind._visual_batch is not None

    N, K = kind.n_envs, kind.K
    n = kind._proto.wheel_meta.n_wheels
    steer = torch.full((N, K, n), 0.1)
    dist = torch.full((N, K, n), 0.30)
    omega = torch.full((N, K, n), 2.0)
    dt = 0.02

    # Same integration start for both paths.
    for v in kind.visuals:
        v.wheel_visual_angle.zero_()
    kind._visual_batch._angle.zero_()

    # Reference: the per-entity loop.
    for k_i, vis in enumerate(kind.visuals):
        vis.step(steer[:, k_i].contiguous(), dist[:, k_i].contiguous(),
                 omega[:, k_i].contiguous(), dt)
    ref = [v.entity.get_dofs_position(dofs_idx_local=v._batch_set_dofs).clone()
           for v in kind.visuals]

    # Batched path with the SAME inputs (its own accumulator integrates from
    # the same zero state) must land every entity on the same dof positions.
    kind._visual_batch.step(steer, dist, omega, dt)
    for v, r in zip(kind.visuals, ref):
        cur = v.entity.get_dofs_position(dofs_idx_local=v._batch_set_dofs)
        assert torch.allclose(cur, r, atol=1e-5), f"{cur} != {r}"


def test_multi_kind_step_batches_io_and_drives(cpu_genesis):
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene", init_genesis=False,
                      wheel_render_mode="internal_sync")  # this test targets VJS itself
    vs.add_ground_plane()
    # Two distinct cfg OBJECTS on the same URDF → two kinds by identity.
    cfg_a = car_4w_rwd_ackermann(URDF, n_envs=1)
    cfg_b = car_4w_rwd_ackermann(URDF, n_envs=1)
    va = vs.add_vehicle(URDF, cfg=cfg_a, pos=(0.0, 0.0, 1.0))
    vb = vs.add_vehicle(URDF, cfg=cfg_b, pos=(6.0, 0.0, 1.0))
    vs.build()
    mvp = vs.physics
    assert mvp.n_kinds == 2
    assert mvp._all_base_idx is not None          # cross-kind I/O cache built

    # settle (the car spawns airborne — no traction until the wheels land)
    for _ in range(200):
        va.set_inputs(brake=1.0)
        vb.set_inputs(brake=1.0)
        vs.step()
    for _ in range(150):
        va.set_inputs(throttle=1.0)
        vb.set_inputs(throttle=1.0)
        vs.step()

    for veh in (va, vb):
        v = veh.get_vel()
        v = v[0] if v.dim() > 1 else v
        assert torch.isfinite(v).all()
        assert float(torch.linalg.norm(v[:2])) > 0.1   # actually driving
