"""Steering strategies. ISO 8855 sign convention: +steer = right turn.

The user-facing API is unambiguous; URDF/Genesis sign quirks (e.g. <axis 0 0 -1>)
are absorbed inside this layer and the visual layer.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional

import torch

from ..config import ConfigError
from ..inputs import (
    AckermannInputs,
    NoSteerInputs,
    PartialAckermannInputs,
    SkidSteerInputs,
)


class _WheelMetaProto:
    """Subset of WheelMeta the strategies care about. Documented here for clarity."""
    n_wheels: int
    positions: torch.Tensor       # (n_wheels, 3)
    axle_index: torch.Tensor      # (n_wheels,) long
    side_mask_L: torch.Tensor     # (n_wheels,) bool
    side_mask_R: torch.Tensor     # (n_wheels,) bool


class SteeringStrategy(ABC):
    """Maps user steer input to per-wheel steer angles in chassis frame.

    Convention: positive `steer_per_wheel[i]` rotates wheel forward direction
    in chassis frame from +X toward -Y (i.e. right turn in ISO 8855). core.py
    applies `wheel_fwd_local = (cos, -sin, 0)` to honour this.
    """

    InputType: ClassVar[type] = AckermannInputs

    @abstractmethod
    def per_wheel_steer(
        self,
        inputs: Any,
        n_envs: int,
        wheel_meta: Any,
        device: Any,
        dtype: Any,
    ) -> torch.Tensor:
        """Returns (n_envs, n_wheels) steer angles in radians."""

    def affects_drive(self) -> bool:
        """True if this strategy also drives wheels (skid-steer); False otherwise."""
        return False

    def validate(self, wheels: list[Any]) -> None:
        """Optional config validation; raise ConfigError on bad config."""
        return None


# ---------------------------------------------------------------------------
# Ackermann (single steered axle)
# ---------------------------------------------------------------------------


class Ackermann(SteeringStrategy):
    """Standard Ackermann steering on a single axle.

    For +steer (right turn): inner wheel = right (smaller turn radius, larger angle);
    outer wheel = left. Both angles are positive (right-turn) under the ISO convention.
    """

    InputType = AckermannInputs

    def __init__(
        self,
        max_steer_rad: float = 0.7,
        front_axle: int = 0,
        wheelbase: Optional[float] = None,
        track_width: Optional[float] = None,
    ):
        self.max_steer_rad = float(max_steer_rad)
        self.front_axle = int(front_axle)
        self.wheelbase = wheelbase
        self.track_width = track_width

    def _resolve_geometry(self, wheel_meta: Any) -> tuple[float, float]:
        """Compute (wheelbase, track_width) from wheel positions if not supplied."""
        if self.wheelbase is not None and self.track_width is not None:
            return float(self.wheelbase), float(self.track_width)
        positions = wheel_meta.positions   # (n_wheels, 3) torch tensor
        axle_idx = wheel_meta.axle_index   # (n_wheels,) long
        # Front-axle x (mean among wheels with axle_index == front_axle)
        front_mask = (axle_idx == self.front_axle)
        if not bool(front_mask.any()):
            raise ConfigError(
                f"Ackermann.front_axle={self.front_axle} matches no wheel."
            )
        front_x = float(positions[front_mask, 0].mean())
        # Rear-most axle = max axle_index.
        rear_axle = int(axle_idx.max())
        rear_mask = (axle_idx == rear_axle)
        rear_x = float(positions[rear_mask, 0].mean())
        wb = self.wheelbase if self.wheelbase is not None else abs(front_x - rear_x)
        # Track width = max - min y on the front axle.
        front_y = positions[front_mask, 1]
        tw = self.track_width if self.track_width is not None else float(front_y.max() - front_y.min())
        if wb <= 0.0 or tw <= 0.0:
            raise ConfigError(f"Ackermann: derived wheelbase={wb}, track_width={tw}")
        return float(wb), float(tw)

    def per_wheel_steer(
        self,
        inputs: AckermannInputs,
        n_envs: int,
        wheel_meta: Any,
        device: Any,
        dtype: Any,
    ) -> torch.Tensor:
        wb, tw = self._resolve_geometry(wheel_meta)
        steer_in = _to_tensor(inputs.steer, n_envs, device, dtype)   # (n_envs,)
        theta_center = steer_in * self.max_steer_rad                  # (n_envs,)

        out = torch.zeros(n_envs, wheel_meta.n_wheels, device=device, dtype=dtype)
        front_mask = (wheel_meta.axle_index == self.front_axle)
        front_idx = torch.nonzero(front_mask, as_tuple=False).flatten()
        if front_idx.numel() == 0:
            return out

        positions = wheel_meta.positions
        # Bicycle radius from chassis center to ICR projection on x-axis.
        # Use a soft tan to avoid singularity at theta=0; signed.
        small = torch.abs(theta_center) < 1e-6
        tan_theta = torch.tan(theta_center)
        # Avoid div-by-zero; we will mask back to 0 for very small steer.
        safe_tan = torch.where(small, torch.ones_like(tan_theta), tan_theta)
        R_center = wb / safe_tan                                     # (n_envs,) signed

        for wi in front_idx.tolist():
            # y of this wheel in chassis frame (ISO 8855: +Y = left).
            y = float(positions[wi, 1])
            # ISO 8855 right turn (+steer): ICR sits on the right (at chassis-frame
            # y = -R_center). Distance from ICR to a wheel at y_wheel is
            # |y_wheel + R_center|. For +steer, R_center > 0, so the right wheel
            # (y < 0) ends up with a smaller R_w (= inner wheel, larger angle).
            R_w = y + R_center                                       # (n_envs,)
            mag = torch.atan(wb / torch.clamp(torch.abs(R_w), min=1e-6))
            theta_w = torch.sign(theta_center) * mag
            theta_w = torch.where(small, torch.zeros_like(theta_w), theta_w)
            out[:, wi] = theta_w
        return out


class PartialAckermann(Ackermann):
    """Ackermann applied to multiple steered axles (e.g. front + middle).

    Each listed axle is steered with the same Ackermann geometry as the front,
    using its own track width but the chassis wheelbase to the rear-most axle.
    """

    InputType = PartialAckermannInputs

    def __init__(
        self,
        max_steer_rad: float = 0.7,
        steered_axles: tuple[int, ...] = (0,),
        wheelbase: Optional[float] = None,
        track_width: Optional[float] = None,
    ):
        super().__init__(
            max_steer_rad=max_steer_rad,
            front_axle=int(steered_axles[0]),
            wheelbase=wheelbase,
            track_width=track_width,
        )
        self.steered_axles = tuple(int(a) for a in steered_axles)

    def per_wheel_steer(
        self,
        inputs: PartialAckermannInputs,
        n_envs: int,
        wheel_meta: Any,
        device: Any,
        dtype: Any,
    ) -> torch.Tensor:
        wb, _ = self._resolve_geometry(wheel_meta)
        steer_in = _to_tensor(inputs.steer, n_envs, device, dtype)
        theta_center = steer_in * self.max_steer_rad
        out = torch.zeros(n_envs, wheel_meta.n_wheels, device=device, dtype=dtype)
        small = torch.abs(theta_center) < 1e-6
        tan_theta = torch.tan(theta_center)
        safe_tan = torch.where(small, torch.ones_like(tan_theta), tan_theta)
        R_center = wb / safe_tan
        positions = wheel_meta.positions
        for axle in self.steered_axles:
            mask = (wheel_meta.axle_index == axle)
            for wi in torch.nonzero(mask, as_tuple=False).flatten().tolist():
                y = float(positions[wi, 1])
                R_w = y + R_center
                mag = torch.atan(wb / torch.clamp(torch.abs(R_w), min=1e-6))
                theta_w = torch.sign(theta_center) * mag
                theta_w = torch.where(small, torch.zeros_like(theta_w), theta_w)
                out[:, wi] = theta_w
        return out


class SkidSteer(SteeringStrategy):
    """Tank-style: zero steer angle on every wheel; turning happens via per-side torque diff."""

    InputType = SkidSteerInputs

    def affects_drive(self) -> bool:
        return True

    def per_wheel_steer(
        self,
        inputs: SkidSteerInputs,
        n_envs: int,
        wheel_meta: Any,
        device: Any,
        dtype: Any,
    ) -> torch.Tensor:
        return torch.zeros(n_envs, wheel_meta.n_wheels, device=device, dtype=dtype)

    def validate(self, wheels: list[Any]) -> None:
        if any(w.side is None for w in wheels):
            raise ConfigError(
                "SkidSteer requires every wheel to have side='L' or 'R'."
            )
        if any(w.steer_joint_name is not None for w in wheels):
            raise ConfigError(
                "SkidSteer vehicles must not have URDF steer joints; "
                "found one or more wheels with steer_joint_name set."
            )


class NoSteer(SteeringStrategy):
    """Zero steer on every wheel; for vehicles with no steering DoF."""

    InputType = NoSteerInputs

    def per_wheel_steer(
        self,
        inputs: NoSteerInputs,
        n_envs: int,
        wheel_meta: Any,
        device: Any,
        dtype: Any,
    ) -> torch.Tensor:
        return torch.zeros(n_envs, wheel_meta.n_wheels, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _to_tensor(x: Any, n_envs: int, device: Any, dtype: Any) -> torch.Tensor:
    """Broadcast scalar or tensor to (n_envs,)."""
    if torch.is_tensor(x):
        t = x.to(device=device, dtype=dtype)
        if t.dim() == 0:
            t = t.expand(n_envs)
        return t.contiguous()
    return torch.full((n_envs,), float(x), device=device, dtype=dtype)
