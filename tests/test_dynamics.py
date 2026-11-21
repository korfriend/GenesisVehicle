"""Tests for the pure dynamics primitives in genesis_vehicle.dynamics.

These exercise the suspension-normal-force contracts the
SDK promises (no Genesis runtime needed).
"""

from __future__ import annotations

import pytest
import torch

from genesis_vehicle.dynamics import (
    suspension_normal_force,
)


# ---------------------------------------------------------------------------
# brake_torque_signed — deprecated in 1.8.0, REMOVED in 2.0.0. Its behaviour
# tests went with it; the brake is `brake_projection` (test_axle_torque_order).
# Mutant this test must fail on: the function restored (module or root).
# ---------------------------------------------------------------------------


def test_brake_torque_signed_is_removed():
    import genesis_vehicle
    import genesis_vehicle.dynamics as dyn
    with pytest.raises(ImportError):
        from genesis_vehicle.dynamics import brake_torque_signed  # noqa: F401
    assert not hasattr(dyn, "brake_torque_signed")
    assert "brake_torque_signed" not in genesis_vehicle.__all__
    with pytest.raises(AttributeError):
        genesis_vehicle.brake_torque_signed


# ---------------------------------------------------------------------------
# suspension_normal_force
# ---------------------------------------------------------------------------


def test_suspension_normal_force_clamped_nonnegative_on_rebound():
    """A strong rebound (c_dot << 0, |C*c_dot| > K*c) must NOT produce
    negative normal force; the ground cannot pull a wheel down."""
    compression = torch.tensor([0.05])      # 5 cm compressed
    comp_rate = torch.tensor([-10.0])       # rapid extension
    k_susp = 1_000.0                        # weak spring
    c_compression = 100.0
    c_extension = 1_000.0                   # strong rebound damper
    air_mask = torch.tensor([False])
    N = suspension_normal_force(
        compression, comp_rate, k_susp, c_compression, c_extension, air_mask,
    )
    # raw = 1000*0.05 + 1000*(-10) = 50 - 10000 = -9950 -> clamp to 0
    assert N.item() == pytest.approx(0.0)


def test_suspension_normal_force_zero_on_air_mask():
    """When the ray missed the ground, N is zero regardless of inputs."""
    compression = torch.tensor([0.10, 0.20])
    comp_rate = torch.tensor([1.0, 0.0])
    air_mask = torch.tensor([True, True])
    N = suspension_normal_force(
        compression, comp_rate, 70_000.0, 14_000.0, 4_000.0, air_mask,
    )
    assert torch.all(N == 0.0)


def test_suspension_asymmetric_damper_compression_vs_extension():
    """Same |c_dot| produces different damping on compression vs extension
    when c_compression != c_extension."""
    compression = torch.tensor([0.05, 0.05])
    comp_rate = torch.tensor([+1.0, -1.0])    # compress vs extend
    k_susp = 0.0                              # isolate damper term
    c_compression = 10_000.0
    c_extension = 1_000.0
    air_mask = torch.tensor([False, False])
    N = suspension_normal_force(
        compression, comp_rate, k_susp, c_compression, c_extension, air_mask,
    )
    # compress: 10000*1 = 10000 (positive, kept)
    # extend:   1000*(-1) = -1000 (clamped to 0)
    assert N[0].item() == pytest.approx(10_000.0)
    assert N[1].item() == pytest.approx(0.0)


def test_suspension_normal_force_static_compression_only():
    """At c_dot = 0 (settled), N = K * compression with no damping term."""
    compression = torch.tensor([0.05, 0.10])
    comp_rate = torch.zeros_like(compression)
    air_mask = torch.tensor([False, False])
    N = suspension_normal_force(
        compression, comp_rate, 70_000.0, 14_000.0, 4_000.0, air_mask,
    )
    assert N[0].item() == pytest.approx(3_500.0)
    assert N[1].item() == pytest.approx(7_000.0)
