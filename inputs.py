"""Input dataclasses for VehiclePhysics.

Defines the unified VehicleInputs adapter and the per-strategy typed inputs
(AckermannInputs, PartialAckermannInputs, SkidSteerInputs, NoSteerInputs).
"""

from dataclasses import dataclass, field
from typing import Optional, Union

import torch

ScalarOrTensor = Union[torch.Tensor, float, int]


@dataclass
class VehicleInputs:
    """Unified adapter. throttle in [-1, 1] signed; brake in [0, 1]; steer in [-1, 1]
    where +steer = right turn (ISO 8855).

    ``gear`` (v2.0.2, keyword-only, default None): None is the signed-throttle
    input above, unchanged. ``+1`` / ``-1`` (or a per-env tensor, mapped by
    sign, 0 -> +1) selects GEAR MODE for this vehicle: the throttle is then a
    magnitude, clamped to [0, 1], applied in the direction of the ENGAGED gear,
    and the engaged gear follows the request only on a step with throttle 0 and
    planar speed below ``gear.GEAR_SHIFT_MAX_SPEED`` (see
    :mod:`genesis_vehicle.gear`). With a gear, a negative Python scalar
    throttle or a scalar gear other than +-1 raises ``ValueError``, and a
    vehicle with a ``PerSide`` drivetrain or ``SkidSteer`` steering refuses it
    (``ValueError``) — checked by ``Vehicle.set_inputs`` and by the drivers'
    ``step`` before any state is written. A tensor throttle is clamped to
    [0, 1], not checked."""
    throttle: ScalarOrTensor
    brake: ScalarOrTensor
    steer: ScalarOrTensor
    gear: Optional[ScalarOrTensor] = field(default=None, kw_only=True)


@dataclass
class AckermannInputs:
    """Steering-by-wheel inputs for a steered front (or rear) axle.
    ``gear``: see :class:`VehicleInputs` (v2.0.2, keyword-only)."""
    throttle: ScalarOrTensor
    brake: ScalarOrTensor
    steer: ScalarOrTensor
    gear: Optional[ScalarOrTensor] = field(default=None, kw_only=True)

    @classmethod
    def from_unified(cls, vi: VehicleInputs) -> "AckermannInputs":
        return cls(vi.throttle, vi.brake, vi.steer, gear=getattr(vi, "gear", None))


@dataclass
class PartialAckermannInputs(AckermannInputs):
    """Same shape as AckermannInputs; carried as a distinct type for strategy dispatch."""
    pass


@dataclass
class SkidSteerInputs:
    """Tank-style differential drive inputs.

    throttle in [-1, 1]; PerSide drivetrain internally clamps the forward part
    to [0, 1] and treats negative throttle as reverse.
    steer_diff in [-1, 1]; +steer_diff = right turn (ISO 8855).

    No ``gear`` field (v2.0.2): skid-steer vehicles refuse gear mode, so
    :meth:`from_unified` raises ``ValueError`` on a ``VehicleInputs`` whose
    ``gear`` is not None instead of dropping it.
    """
    throttle: ScalarOrTensor
    brake: ScalarOrTensor
    steer_diff: ScalarOrTensor

    @classmethod
    def from_unified(cls, vi: VehicleInputs) -> "SkidSteerInputs":
        if getattr(vi, "gear", None) is not None:
            raise ValueError(
                "SkidSteerInputs.from_unified: skid-steer vehicles refuse gear mode; "
                f"got gear={vi.gear!r} — drive with the signed throttle (gear=None)")
        return cls(vi.throttle, vi.brake, vi.steer)


@dataclass
class NoSteerInputs:
    """Inputs for vehicles without steering (single-axle or rail-bound).
    ``gear``: see :class:`VehicleInputs` (v2.0.2, keyword-only)."""
    throttle: ScalarOrTensor
    brake: ScalarOrTensor
    gear: Optional[ScalarOrTensor] = field(default=None, kw_only=True)

    @classmethod
    def from_unified(cls, vi: VehicleInputs) -> "NoSteerInputs":
        return cls(vi.throttle, vi.brake, gear=getattr(vi, "gear", None))


# Convenience alias for VehiclePhysics.step() type hints. Accepts any typed
# input matching the configured strategy's InputType, or the unified
# VehicleInputs (auto-converted via from_unified()).
VehicleStepInputs = Union[
    VehicleInputs,
    AckermannInputs,
    PartialAckermannInputs,
    SkidSteerInputs,
    NoSteerInputs,
]
