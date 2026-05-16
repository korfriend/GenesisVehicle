"""Input dataclasses for VehiclePhysics.

Defines the unified VehicleInputs adapter and the per-strategy typed inputs
(AckermannInputs, PartialAckermannInputs, SkidSteerInputs, NoSteerInputs).
"""

from dataclasses import dataclass
from typing import Union

import torch

ScalarOrTensor = Union[torch.Tensor, float, int]


@dataclass
class VehicleInputs:
    """Unified adapter. throttle in [-1, 1] signed; brake in [0, 1]; steer in [-1, 1]
    where +steer = right turn (ISO 8855)."""
    throttle: ScalarOrTensor
    brake: ScalarOrTensor
    steer: ScalarOrTensor


@dataclass
class AckermannInputs:
    """Steering-by-wheel inputs for a steered front (or rear) axle."""
    throttle: ScalarOrTensor
    brake: ScalarOrTensor
    steer: ScalarOrTensor

    @classmethod
    def from_unified(cls, vi: VehicleInputs) -> "AckermannInputs":
        return cls(vi.throttle, vi.brake, vi.steer)


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
    """
    throttle: ScalarOrTensor
    brake: ScalarOrTensor
    steer_diff: ScalarOrTensor

    @classmethod
    def from_unified(cls, vi: VehicleInputs) -> "SkidSteerInputs":
        return cls(vi.throttle, vi.brake, vi.steer)


@dataclass
class NoSteerInputs:
    """Inputs for vehicles without steering (single-axle or rail-bound)."""
    throttle: ScalarOrTensor
    brake: ScalarOrTensor

    @classmethod
    def from_unified(cls, vi: VehicleInputs) -> "NoSteerInputs":
        return cls(vi.throttle, vi.brake)
