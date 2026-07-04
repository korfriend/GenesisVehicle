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
