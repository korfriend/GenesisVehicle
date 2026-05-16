"""Tire model abstract base + concrete implementations.

A TireModel maps (v_long, v_lat, v_roll, N, wheel_params) -> (F_long, F_lat,
kappa, alpha). Models are stateless; per-wheel coefficients flow through
wheel_params (a WheelConfig) so a single model instance serves all wheels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class TireModel(ABC):
    """Stateless per-wheel tire force model."""

    @abstractmethod
    def __call__(
        self,
        v_long: torch.Tensor,
        v_lat: torch.Tensor,
        v_roll: torch.Tensor,
        N: torch.Tensor,
        wheel_params: Any,   # WheelConfig with all fields filled
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (F_long, F_lat, kappa, alpha), each shape == v_long.shape."""


from .pacejka import PacejkaAnisotropic    # noqa: E402  (re-export after ABC defn)
from .coulomb import CoulombIsotropic      # noqa: E402

__all__ = ["TireModel", "PacejkaAnisotropic", "CoulombIsotropic"]
