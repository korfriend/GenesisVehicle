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
    """Stateless batched tire force model.

    v0.5.0: inputs are batched ``(n_envs, n_wheels)`` tensors; per-wheel
    coefficients are read from ``wheel_meta`` (a ``WheelMeta`` with
    ``(n_wheels,)`` coefficient tensors). One call covers all wheels.
    """

    @abstractmethod
    def __call__(
        self,
        v_long: torch.Tensor,   # (n_envs, n_wheels)
        v_lat: torch.Tensor,    # (n_envs, n_wheels)
        v_roll: torch.Tensor,   # (n_envs, n_wheels)
        N: torch.Tensor,        # (n_envs, n_wheels)
        wheel_meta: Any,        # WheelMeta with batched per-wheel tensors
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(F_long, F_lat, kappa, alpha)``, each ``(n_envs, n_wheels)``."""


from .pacejka import PacejkaAnisotropic    # noqa: E402  (re-export after ABC defn)
from .coulomb import CoulombIsotropic      # noqa: E402

__all__ = ["TireModel", "PacejkaAnisotropic", "CoulombIsotropic"]
