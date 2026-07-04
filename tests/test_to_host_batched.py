"""Equivalence guard for the v1.1.1 single-sync capture download.

``_to_host_batched`` concats every capture read tensor on-device and pays ONE
``.cpu()`` DtoH sync instead of one per read point (GPU-backend serving cost
is per-call sync latency, not data volume). These tests pin the batched
slicing to the per-tensor path (``force_batch=True`` exercises the concat
path on CPU tensors, which CI can run without CUDA).
"""

import numpy as np
import pytest

pytest.importorskip("genesis")

import torch

from genesis_vehicle.server.l3_runtime import _to_host_batched


def _tensors():
    g = torch.Generator().manual_seed(3)
    return [
        torch.randn(3, generator=g),               # (3,) chassis pos, 1 env
        torch.randn(4, generator=g),               # (4,)
        torch.randn(10, 7, 3, generator=g),        # (N, n, 3) wheel pos
        torch.randn(10, 7, 4, generator=g),        # (N, n, 4) wheel quat
        torch.randn(1, 3, generator=g),            # obstacle pos
        torch.randn(1, 4, generator=g),            # obstacle quat
    ]


def test_batched_path_matches_per_tensor_path():
    ts = _tensors()
    ref = [t.numpy() for t in ts]
    out = _to_host_batched(ts, force_batch=True)   # concat path
    assert len(out) == len(ref)
    for r, o in zip(ref, out):
        assert o.shape == r.shape
        np.testing.assert_allclose(o, r)


def test_cpu_default_uses_per_tensor_path_and_matches():
    ts = _tensors()
    out = _to_host_batched(ts)                     # CPU: passthrough path
    for t, o in zip(ts, out):
        np.testing.assert_allclose(o, t.numpy())


def test_mixed_dtypes_fall_back_safely():
    ts = [torch.randn(2, 3), torch.arange(4, dtype=torch.int64)]
    out = _to_host_batched(ts, force_batch=True)   # dtype mix → per-tensor
    np.testing.assert_allclose(out[0], ts[0].numpy())
    np.testing.assert_array_equal(out[1], ts[1].numpy())


def test_empty_and_numpy_inputs():
    assert _to_host_batched([]) == []
    arr = np.ones((2, 3), np.float32)
    out = _to_host_batched([arr])
    np.testing.assert_allclose(out[0], arr)


def test_wheel_visual_transforms_host_matches_device_path():
    """v1.1.3 'GPU 모드 = 물리만 GPU, 캡처 연산은 CPU' 경로: host 계산이
    기존 device 계산과 동일한 휠 월드 포즈를 내야 한다 (CPU 백엔드에서 두
    경로 모두 실행 가능 — 수학은 device 무관이므로 CPU 파리티가 곧 GPU
    파리티다)."""
    import os
    import genesis as gs
    from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann

    if not getattr(gs, "_initialized", False):
        VehicleScene.init_backend("cpu")
    URDF = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        "..", "samples", "urdf", "car_4w.urdf"))
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene", init_genesis=False)
    vs.add_ground_plane()
    veh = vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(0.0, 0.0, 1.0))
    vs.build()
    for _ in range(30):
        veh.set_inputs(throttle=0.6, steer=0.2)
        vs.step()

    kind = vs.physics.kinds[0]
    ref_wp, ref_wq = kind.wheel_visual_transforms("world")     # device path
    reads = kind.wheel_visual_reads()
    assert reads is not None
    hosts = _to_host_batched(list(reads), force_batch=True)
    args = [torch.from_numpy(np.ascontiguousarray(h)) for h in hosts]
    wp, wq = kind.wheel_visual_transforms_host(*args, "world")  # host path
    np.testing.assert_allclose(wp.numpy(), ref_wp.cpu().numpy(), atol=1e-5)
    np.testing.assert_allclose(wq.numpy(), ref_wq.cpu().numpy(), atol=1e-5)
