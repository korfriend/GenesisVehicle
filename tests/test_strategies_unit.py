"""Pure-Python tests for steering/drivetrain/coupling strategy logic.

These exercise the math in isolation: no Genesis runtime, no scene, no entity.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from genesis_vehicle.config import ConfigError, WheelConfig
from genesis_vehicle.inputs import (
    AckermannInputs,
    SkidSteerInputs,
)
from genesis_vehicle.strategies import (
    Ackermann,
    Independent,
    PerSide,
    RWD,
    SameSideBelt,
    SkidSteer,
)


@dataclass
class _FakeWheelMeta:
    """Stand-in for core.WheelMeta — only the fields strategies read."""
    n_wheels: int
    positions: torch.Tensor
    axle_index: torch.Tensor
    side_mask_L: torch.Tensor
    side_mask_R: torch.Tensor
    left_idx: torch.Tensor
    right_idx: torch.Tensor


def _hjw_meta() -> _FakeWheelMeta:
    """4-wheel meta matching HJW positions (FL, FR, RL, RR)."""
    positions = torch.tensor([
        [1.38, 0.80, 0.34],
        [1.38, -0.80, 0.34],
        [-1.35, 0.80, 0.34],
        [-1.35, -0.80, 0.34],
    ], dtype=torch.float32)
    axle = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    L = torch.tensor([True, False, True, False])
    R = torch.tensor([False, True, False, True])
    return _FakeWheelMeta(
        n_wheels=4,
        positions=positions,
        axle_index=axle,
        side_mask_L=L,
        side_mask_R=R,
        left_idx=torch.nonzero(L, as_tuple=False).flatten().to(torch.long),
        right_idx=torch.nonzero(R, as_tuple=False).flatten().to(torch.long),
    )


def _kdu_meta() -> _FakeWheelMeta:
    """10-wheel tank meta (5 left, 5 right, 5 axles)."""
    pos = []
    for x in (3.0, 1.5, 0.0, -1.5, -3.0):
        pos.append([x, 1.42, 0.0])
    for x in (3.0, 1.5, 0.0, -1.5, -3.0):
        pos.append([x, -1.42, 0.0])
    positions = torch.tensor(pos, dtype=torch.float32)
    # Axle index by x (0 = front-most x=3.0, 4 = rear-most x=-3.0).
    axle = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=torch.long)
    L = torch.tensor([True] * 5 + [False] * 5)
    R = torch.tensor([False] * 5 + [True] * 5)
    return _FakeWheelMeta(
        n_wheels=10,
        positions=positions,
        axle_index=axle,
        side_mask_L=L,
        side_mask_R=R,
        left_idx=torch.nonzero(L, as_tuple=False).flatten().to(torch.long),
        right_idx=torch.nonzero(R, as_tuple=False).flatten().to(torch.long),
    )


# ---------------------------------------------------------------------------
# Ackermann steering math
# ---------------------------------------------------------------------------


def test_ackermann_zero_steer_returns_zero():
    meta = _hjw_meta()
    strat = Ackermann(max_steer_rad=0.7, front_axle=0)
    inp = AckermannInputs(throttle=0.0, brake=0.0, steer=0.0)
    out = strat.per_wheel_steer(inp, n_envs=2, wheel_meta=meta,
                                 device="cpu", dtype=torch.float32)
    assert out.shape == (2, 4)
    assert torch.allclose(out, torch.zeros_like(out))


def test_ackermann_right_turn_inner_outer_signs():
    """For ISO +steer (right turn), both front wheels turn positive (right).
    Inner wheel (FR, y < 0) should turn MORE than outer (FL, y > 0)."""
    meta = _hjw_meta()
    strat = Ackermann(max_steer_rad=0.7, front_axle=0)
    inp = AckermannInputs(throttle=0.0, brake=0.0, steer=0.3)
    out = strat.per_wheel_steer(inp, n_envs=1, wheel_meta=meta,
                                 device="cpu", dtype=torch.float32)
    # FL = index 0, FR = index 1. Both positive (right turn), FR > FL magnitude.
    fl = float(out[0, 0])
    fr = float(out[0, 1])
    assert fl > 0.0, f"FL angle should be positive for right turn, got {fl}"
    assert fr > 0.0, f"FR angle should be positive for right turn, got {fr}"
    assert fr > fl, f"Inner wheel (FR) should turn more than outer (FL): FR={fr}, FL={fl}"
    # Rear wheels are not steered.
    assert float(out[0, 2]) == pytest.approx(0.0)
    assert float(out[0, 3]) == pytest.approx(0.0)


def test_ackermann_left_turn_signs_inverted():
    """For ISO -steer (left turn), both front wheels turn negative; LEFT wheel
    is now the inner (larger magnitude)."""
    meta = _hjw_meta()
    strat = Ackermann(max_steer_rad=0.7, front_axle=0)
    inp = AckermannInputs(throttle=0.0, brake=0.0, steer=-0.3)
    out = strat.per_wheel_steer(inp, n_envs=1, wheel_meta=meta,
                                 device="cpu", dtype=torch.float32)
    fl = float(out[0, 0])
    fr = float(out[0, 1])
    assert fl < 0.0
    assert fr < 0.0
    assert abs(fl) > abs(fr), \
        f"Inner wheel (FL) should turn more than outer (FR) for left turn: FL={fl}, FR={fr}"


def test_ackermann_geometry_inferred_from_positions():
    """When wheelbase / track_width are None, derive them from the wheel positions.
    HJW: WB = 1.38 - (-1.35) = 2.73, TW = 1.6."""
    meta = _hjw_meta()
    strat = Ackermann(max_steer_rad=0.7, front_axle=0)
    wb, tw = strat._resolve_geometry(meta)
    assert wb == pytest.approx(2.73)
    assert tw == pytest.approx(1.6)


# ---------------------------------------------------------------------------
# SkidSteer
# ---------------------------------------------------------------------------


def test_skidsteer_returns_zero_angles():
    meta = _kdu_meta()
    strat = SkidSteer()
    inp = SkidSteerInputs(throttle=1.0, brake=0.0, steer_diff=1.0)
    out = strat.per_wheel_steer(inp, n_envs=3, wheel_meta=meta,
                                 device="cpu", dtype=torch.float32)
    assert out.shape == (3, 10)
    assert torch.all(out == 0.0)


def test_skidsteer_validate_rejects_unsided():
    strat = SkidSteer()
    wheels = [WheelConfig(name="a", side=None), WheelConfig(name="b", side="L")]
    with pytest.raises(ConfigError):
        strat.validate(wheels)


def test_skidsteer_validate_rejects_steer_joint():
    strat = SkidSteer()
    wheels = [
        WheelConfig(name="a", side="L", steer_joint_name="steer_a"),
        WheelConfig(name="b", side="R"),
    ]
    with pytest.raises(ConfigError):
        strat.validate(wheels)


# ---------------------------------------------------------------------------
# SameSideBelt coupling
# ---------------------------------------------------------------------------


def test_same_side_belt_averages_each_side_per_env():
    meta = _kdu_meta()
    coupling = SameSideBelt()
    omega = torch.zeros(2, 10, dtype=torch.float32)
    # env 0: left = 1..5, right = 10..50
    omega[0, :5] = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    omega[0, 5:] = torch.tensor([10, 20, 30, 40, 50], dtype=torch.float32)
    # env 1: left = 100, right = -100
    omega[1, :5] = 100.0
    omega[1, 5:] = -100.0

    out = coupling.apply(omega, meta)
    # env 0: left mean = 3, right mean = 30
    assert torch.allclose(out[0, :5], torch.full((5,), 3.0))
    assert torch.allclose(out[0, 5:], torch.full((5,), 30.0))
    # env 1: unchanged because all values already equal
    assert torch.allclose(out[1, :5], torch.full((5,), 100.0))
    assert torch.allclose(out[1, 5:], torch.full((5,), -100.0))


def test_independent_coupling_passes_through():
    meta = _hjw_meta()
    omega = torch.randn(2, 4)
    out = Independent().apply(omega, meta)
    assert torch.equal(out, omega)


# ---------------------------------------------------------------------------
# RWD drivetrain split
# ---------------------------------------------------------------------------


def test_rwd_drives_only_rear_axle():
    meta = _hjw_meta()
    drv = RWD(t_drive_max=1000.0, t_brake_max=2500.0, driven_axles=(1,),
              brake_bias=[0.30, 0.30, 0.20, 0.20])
    inp = AckermannInputs(throttle=1.0, brake=0.5, steer=0.0)
    omega = torch.zeros(1, 4)
    Td, Tb = drv.distribute_torque(inp, omega, meta, device="cpu", dtype=torch.float32)
    # Front drive == 0 (HJW invariant)
    assert float(Td[0, 0]) == pytest.approx(0.0)
    assert float(Td[0, 1]) == pytest.approx(0.0)
    # Rear drive = 1000 / 2 each
    assert float(Td[0, 2]) == pytest.approx(500.0)
    assert float(Td[0, 3]) == pytest.approx(500.0)
    # Brake bias check (60/40 front/rear with even split inside axle).
    assert float(Tb[0, 0]) == pytest.approx(0.5 * 2500 * 0.30)
    assert float(Tb[0, 2]) == pytest.approx(0.5 * 2500 * 0.20)


# ---------------------------------------------------------------------------
# PerSide drivetrain
# ---------------------------------------------------------------------------


def test_perside_iso_right_turn_left_faster():
    """ISO 8855: +steer_diff = right turn, so LEFT side commands MORE torque
    than RIGHT side."""
    meta = _kdu_meta()
    drv = PerSide(
        t_drive_max=10_000.0, t_brake_max=10_000.0,
        steer_gain=1.0, omega_max_drive=100.0,
        throttle_gear_cap=1.0, use_per_side_taper=False,
    )
    inp = SkidSteerInputs(throttle=0.0, brake=0.0, steer_diff=0.5)
    omega = torch.zeros(1, 10)
    Td, _ = drv.distribute_torque(inp, omega, meta, device="cpu", dtype=torch.float32)
    # Left wheels (idx 0..4) should be positive; right (5..9) negative.
    left_T = float(Td[0, 0])
    right_T = float(Td[0, 5])
    assert left_T > 0.0
    assert right_T < 0.0
    assert left_T == pytest.approx(0.5 * 10_000)
    assert right_T == pytest.approx(-0.5 * 10_000)


def test_perside_throttle_gear_cap_scales():
    meta = _kdu_meta()
    drv = PerSide(
        t_drive_max=10_000.0, t_brake_max=10_000.0,
        steer_gain=1.0, omega_max_drive=100.0,
        throttle_gear_cap=0.3, use_per_side_taper=False,
    )
    inp = SkidSteerInputs(throttle=1.0, brake=0.0, steer_diff=0.0)
    omega = torch.zeros(1, 10)
    Td, _ = drv.distribute_torque(inp, omega, meta, device="cpu", dtype=torch.float32)
    # Effective throttle = 0.3 -> per-wheel = 3000 N*m.
    assert float(Td[0, 0]) == pytest.approx(3000.0)
    assert float(Td[0, 5]) == pytest.approx(3000.0)
