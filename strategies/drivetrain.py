"""Drivetrain strategies. Distributes (throttle, brake) across wheels.

All strategies return per-wheel drive torque and per-wheel brake torque,
both shape (n_envs, n_wheels), in N*m. Brake is always positive magnitude;
the per-wheel loop in core.py applies it opposite to wheel ω via tanh smoothing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from ..config import ConfigError


class DrivetrainStrategy(ABC):
    """Maps user (throttle, brake) to per-wheel (T_drive, T_brake)."""

    @abstractmethod
    def distribute_torque(
        self,
        inputs: Any,
        omega: torch.Tensor,        # (n_envs, n_wheels)
        wheel_meta: Any,
        device: Any,
        dtype: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (T_drive_per_wheel, T_brake_per_wheel) both (n_envs, n_wheels)."""

    def validate(self, wheels: list[Any]) -> None:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_tensor(x: Any, n_envs: int, device: Any, dtype: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        t = x.to(device=device, dtype=dtype)
        if t.dim() == 0:
            t = t.expand(n_envs)
        return t.contiguous()
    return torch.full((n_envs,), float(x), device=device, dtype=dtype)


def _axle_drive_mask(
    wheel_meta: Any, axles: tuple[int, ...], device: Any, dtype: Any
) -> torch.Tensor:
    """Boolean mask (n_wheels,) marking driven wheels."""
    mask = torch.zeros(wheel_meta.n_wheels, device=device, dtype=torch.bool)
    for a in axles:
        mask = mask | (wheel_meta.axle_index == a)
    return mask


def _normalise_weights(
    weights: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Renormalise so weights[mask] sums to 1; weights[~mask] = 0."""
    w = weights * mask.to(dtype=weights.dtype)
    s = w.sum()
    if float(s) <= 1e-12:
        return w
    return w / s


# ---------------------------------------------------------------------------
# RWD / FWD
# ---------------------------------------------------------------------------


class RWD(DrivetrainStrategy):
    """Rear-wheel drive: torque split equally across rear axle wheels.
    Brake bias defaults to front-biased 60/40 (split equally within each axle)."""

    def __init__(
        self,
        t_drive_max: float,
        t_brake_max: float,
        driven_axles: Optional[tuple[int, ...]] = None,
        brake_bias: Optional[list[float]] = None,
    ):
        self.t_drive_max = float(t_drive_max)
        self.t_brake_max = float(t_brake_max)
        # If driven_axles is None, it's resolved to the rear-most axle at distribute time.
        self.driven_axles = driven_axles
        self.brake_bias = brake_bias

    def _resolve_driven(self, wheel_meta: Any) -> tuple[int, ...]:
        if self.driven_axles is not None:
            return tuple(int(a) for a in self.driven_axles)
        return (int(wheel_meta.axle_index.max()),)

    def _resolve_brake_bias(
        self, wheel_meta: Any, device: Any, dtype: Any
    ) -> torch.Tensor:
        if self.brake_bias is not None:
            bb = torch.tensor(self.brake_bias, device=device, dtype=dtype)
            if bb.numel() != wheel_meta.n_wheels:
                raise ConfigError(
                    f"brake_bias length {bb.numel()} != n_wheels {wheel_meta.n_wheels}")
            return bb
        # Default: 60/40 front/rear, split equally within each axle.
        n = wheel_meta.n_wheels
        axle = wheel_meta.axle_index
        n_axles = int(axle.max()) + 1
        # Assume axle 0 = front, axle n_axles-1 = rear. With > 2 axles, fall back to uniform.
        if n_axles == 2:
            front_mask = (axle == 0).to(dtype=dtype)
            rear_mask = (axle == 1).to(dtype=dtype)
            n_front = float(front_mask.sum().item()) or 1.0
            n_rear = float(rear_mask.sum().item()) or 1.0
            bb = 0.6 * front_mask / n_front + 0.4 * rear_mask / n_rear
        else:
            bb = torch.full((n,), 1.0 / n, device=device, dtype=dtype)
        return bb

    def distribute_torque(
        self,
        inputs: Any,
        omega: torch.Tensor,
        wheel_meta: Any,
        device: Any,
        dtype: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_envs = omega.shape[0]
        throttle = _to_tensor(inputs.throttle, n_envs, device, dtype)
        brake = _to_tensor(inputs.brake, n_envs, device, dtype)

        driven = self._resolve_driven(wheel_meta)
        drive_mask = _axle_drive_mask(wheel_meta, driven, device, dtype).to(dtype=dtype)
        n_driven = float(drive_mask.sum().item()) or 1.0
        # Equal split among driven wheels.
        per_wheel_share = drive_mask / n_driven           # (n_wheels,)
        T_drive = (throttle * self.t_drive_max).unsqueeze(-1) * per_wheel_share.unsqueeze(0)

        bb = self._resolve_brake_bias(wheel_meta, device, dtype)   # (n_wheels,)
        T_brake = (brake * self.t_brake_max).unsqueeze(-1) * bb.unsqueeze(0)
        return T_drive, T_brake


class FWD(RWD):
    """Front-wheel drive: same split logic as RWD but driven axle defaults to front."""

    def _resolve_driven(self, wheel_meta: Any) -> tuple[int, ...]:
        if self.driven_axles is not None:
            return tuple(int(a) for a in self.driven_axles)
        return (int(wheel_meta.axle_index.min()),)


class AWD(DrivetrainStrategy):
    """All-wheel drive: per-wheel drive weights (default uniform across all wheels)."""

    def __init__(
        self,
        t_drive_max: float,
        t_brake_max: float,
        drive_weights: Optional[list[float]] = None,
        brake_bias: Optional[list[float]] = None,
    ):
        self.t_drive_max = float(t_drive_max)
        self.t_brake_max = float(t_brake_max)
        self.drive_weights = drive_weights
        self.brake_bias = brake_bias

    def distribute_torque(
        self,
        inputs: Any,
        omega: torch.Tensor,
        wheel_meta: Any,
        device: Any,
        dtype: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_envs = omega.shape[0]
        throttle = _to_tensor(inputs.throttle, n_envs, device, dtype)
        brake = _to_tensor(inputs.brake, n_envs, device, dtype)
        n = wheel_meta.n_wheels

        if self.drive_weights is not None:
            dw = torch.tensor(self.drive_weights, device=device, dtype=dtype)
            if dw.numel() != n:
                raise ConfigError(f"drive_weights length {dw.numel()} != n_wheels {n}")
            s = dw.sum()
            if float(s) > 1e-12:
                dw = dw / s
        else:
            dw = torch.full((n,), 1.0 / n, device=device, dtype=dtype)

        if self.brake_bias is not None:
            bb = torch.tensor(self.brake_bias, device=device, dtype=dtype)
            if bb.numel() != n:
                raise ConfigError(f"brake_bias length {bb.numel()} != n_wheels {n}")
        else:
            # Default: 60/40 front/rear if 2 axles else uniform.
            axle = wheel_meta.axle_index
            n_axles = int(axle.max()) + 1
            if n_axles == 2:
                front_mask = (axle == 0).to(dtype=dtype)
                rear_mask = (axle == 1).to(dtype=dtype)
                n_front = float(front_mask.sum().item()) or 1.0
                n_rear = float(rear_mask.sum().item()) or 1.0
                bb = 0.6 * front_mask / n_front + 0.4 * rear_mask / n_rear
            else:
                bb = torch.full((n,), 1.0 / n, device=device, dtype=dtype)

        T_drive = (throttle * self.t_drive_max).unsqueeze(-1) * dw.unsqueeze(0)
        T_brake = (brake * self.t_brake_max).unsqueeze(-1) * bb.unsqueeze(0)
        return T_drive, T_brake


# ---------------------------------------------------------------------------
# PerSide (skid-steer / tank)
# ---------------------------------------------------------------------------


class PerSide(DrivetrainStrategy):
    """Per-side drivetrain for skid-steer vehicles (tank tracks).

    Throttle in [-1, 1] is the SDK contract. PerSide treats the forward part as
    forward drive and uses the user's brake input for braking; reverse drive
    is achieved by passing negative throttle.

    Steer differential: ISO 8855 +steer_diff = right turn, so the LEFT side
    spins faster and the RIGHT side slower. Internally this is implemented as
    side_throttle_L = throttle + steer_gain * steer_diff
    side_throttle_R = throttle - steer_gain * steer_diff
    """

    def __init__(
        self,
        t_drive_max: float,
        t_brake_max: float,
        steer_gain: float = 1.0,
        omega_max_drive: float = 100.0,
        throttle_gear_cap: float = 1.0,
        use_per_side_taper: bool = True,
    ):
        self.t_drive_max = float(t_drive_max)
        self.t_brake_max = float(t_brake_max)
        self.steer_gain = float(steer_gain)
        self.omega_max_drive = float(omega_max_drive)
        self.throttle_gear_cap = float(throttle_gear_cap)
        self.use_per_side_taper = bool(use_per_side_taper)

    @staticmethod
    def _cap_torque_batched(
        omega_side: torch.Tensor, T_des: torch.Tensor, omega_max: float, T_max: float
    ) -> torch.Tensor:
        """Batched per-side torque cap. omega_side / T_des shape: (n_envs,).
        Cap drive only in the same direction as omega; opposite direction (decel)
        keeps full T_max."""
        taper_pos = torch.clamp(1.0 - torch.clamp(omega_side, min=0.0) / omega_max, min=0.0)
        taper_neg = torch.clamp(1.0 - torch.clamp(-omega_side, min=0.0) / omega_max, min=0.0)
        T_hi = T_max * taper_pos
        T_lo = -T_max * taper_neg
        return torch.maximum(T_lo, torch.minimum(T_hi, T_des))

    def distribute_torque(
        self,
        inputs: Any,
        omega: torch.Tensor,
        wheel_meta: Any,
        device: Any,
        dtype: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_envs = omega.shape[0]
        n = wheel_meta.n_wheels
        # Accept either SkidSteerInputs (steer_diff) or unified-style (steer).
        throttle = _to_tensor(inputs.throttle, n_envs, device, dtype)
        brake = _to_tensor(inputs.brake, n_envs, device, dtype)
        steer = getattr(inputs, "steer_diff", None)
        if steer is None:
            steer = getattr(inputs, "steer")
        steer = _to_tensor(steer, n_envs, device, dtype)

        # Gear cap: scale throttle before physics (reference-tank tuning).
        throttle = throttle * self.throttle_gear_cap

        # Per-side commanded throttle. ISO: +steer = right turn -> L faster.
        thr_L = throttle + self.steer_gain * steer
        thr_R = throttle - self.steer_gain * steer

        # Per-side mean omega for taper (over the wheels on each side).
        L_idx = wheel_meta.left_idx
        R_idx = wheel_meta.right_idx
        if L_idx.numel() == 0 or R_idx.numel() == 0:
            raise ConfigError("PerSide requires at least one left and one right wheel.")
        omega_L = omega[:, L_idx].mean(dim=-1)   # (n_envs,)
        omega_R = omega[:, R_idx].mean(dim=-1)

        T_des_L = thr_L * self.t_drive_max
        T_des_R = thr_R * self.t_drive_max
        if self.use_per_side_taper:
            T_drive_L = self._cap_torque_batched(omega_L, T_des_L, self.omega_max_drive, self.t_drive_max)
            T_drive_R = self._cap_torque_batched(omega_R, T_des_R, self.omega_max_drive, self.t_drive_max)
        else:
            T_drive_L = T_des_L
            T_drive_R = T_des_R

        T_drive = torch.zeros(n_envs, n, device=device, dtype=dtype)
        T_drive[:, L_idx] = T_drive_L.unsqueeze(-1)
        T_drive[:, R_idx] = T_drive_R.unsqueeze(-1)

        # Brake distributed equally across all wheels.
        T_brake = (brake * self.t_brake_max / float(n)).unsqueeze(-1).expand(n_envs, n).contiguous()
        return T_drive, T_brake

    def validate(self, wheels: list[Any]) -> None:
        if any(w.side is None for w in wheels):
            raise ConfigError("PerSide requires every wheel to have side='L' or 'R'.")
