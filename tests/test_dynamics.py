"""Tests for the pure dynamics primitives in genesis_vehicle.dynamics.

These exercise the brake-sign and suspension-normal-force contracts the
SDK promises (no Genesis runtime needed).
"""

from __future__ import annotations

import pytest
import torch

from genesis_vehicle.dynamics import (
    brake_torque_signed,
    suspension_normal_force,
)


# ---------------------------------------------------------------------------
# brake_torque_signed
# ---------------------------------------------------------------------------


def test_brake_smoothing_opposes_positive_rotation():
    """When omega > 0, T_brake_eff > 0, so -T_brake_eff decelerates the wheel."""
    omega = torch.tensor([0.1, 1.0, 5.0, 10.0])
    t_brake = torch.full_like(omega, 100.0)
    eff = brake_torque_signed(t_brake, omega)
    assert torch.all(eff > 0.0), f"expected positive eff for omega>0, got {eff}"
    # Saturates at large |omega|.
    assert eff[-1].item() == pytest.approx(100.0, abs=1.0)


def test_brake_smoothing_opposes_negative_rotation():
    """When omega < 0 (reverse spin), T_brake_eff < 0 so -T_brake_eff still
    decelerates the wheel (this is the contract section 7 of API.md)."""
    omega = torch.tensor([-0.1, -1.0, -5.0, -10.0])
    t_brake = torch.full_like(omega, 100.0)
    eff = brake_torque_signed(t_brake, omega)
    assert torch.all(eff < 0.0), f"expected negative eff for omega<0, got {eff}"
    assert eff[-1].item() == pytest.approx(-100.0, abs=1.0)


def test_brake_at_rest_is_near_zero():
    """At omega=0 the tanh smoothing yields near-zero brake; pair with
    StaticFrictionLock for a hard hold-at-rest."""
    omega = torch.tensor([0.0])
    t_brake = torch.tensor([100.0])
    eff = brake_torque_signed(t_brake, omega)
    assert abs(eff.item()) < 1e-3


def test_brake_smoothing_scale_changes_response():
    """Smaller smoothing_scale means sharper transition near zero."""
    omega = torch.tensor([0.25])
    t_brake = torch.tensor([100.0])
    eff_sharp = brake_torque_signed(t_brake, omega, smoothing_scale=0.1)
    eff_soft = brake_torque_signed(t_brake, omega, smoothing_scale=1.0)
    assert eff_sharp.item() > eff_soft.item()


def test_brake_clamp_prevents_omega_sign_flip():
    """Regression: with dt + i_wheel passed, brake torque must be capped
    so a single forward-Euler step cannot reverse omega's sign. Without
    the clamp the explicit Euler integrator overshoots zero and the
    wheel oscillates (brake appears to act as propulsion)."""
    # Conditions that previously caused overshoot:
    #   omega = +0.1 rad/s, t_brake = 100 Nm, I = 0.5 kg.m^2, dt = 0.01 s
    omega   = torch.tensor([0.1, -0.1, 5.0, -5.0])
    t_brake = torch.tensor([100.0, 100.0, 100.0, 100.0])
    i_wheel = torch.tensor([0.5, 0.5, 0.5, 0.5])
    DT = 0.01

    eff = brake_torque_signed(t_brake, omega, dt=DT, i_wheel=i_wheel)

    # 1) Sign always matches omega (brake opposes spin).
    assert torch.all((eff * omega) >= 0.0), \
        f"brake torque must have same sign as omega, got eff={eff} omega={omega}"

    # 2) |T_brake_eff| <= |omega| * I / dt for each wheel — i.e. the
    #    torque is exactly the cap when t_brake exceeds it.
    cap = torch.abs(omega) * i_wheel / DT     # max non-reversing torque
    assert torch.all(torch.abs(eff) <= cap + 1e-6), \
        f"brake torque exceeded cap: eff={eff} cap={cap}"

    # 3) Apply one Euler step: new_omega = omega - eff/I * dt. With the
    #    clamp, sign(new_omega) == sign(omega) OR new_omega == 0.
    new_omega = omega - eff / i_wheel * DT
    same_sign_or_zero = (new_omega * omega >= 0.0) | (torch.abs(new_omega) < 1e-6)
    assert torch.all(same_sign_or_zero), \
        f"brake reversed omega: omega={omega} -> new_omega={new_omega}"


def test_brake_clamp_legacy_when_dt_zero():
    """When dt=0 (default), behavior is unchanged from v0.5.31 — only
    the tanh smoothing applies, no magnitude clamp."""
    omega = torch.tensor([0.1])
    t_brake = torch.tensor([100.0])
    eff = brake_torque_signed(t_brake, omega)     # no dt, no i_wheel
    # Same as v0.5.31: t_brake * tanh(omega / 0.5) = 100 * tanh(0.2)
    expected = 100.0 * torch.tanh(torch.tensor(0.2))
    assert eff.item() == pytest.approx(expected.item(), rel=1e-5)


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
