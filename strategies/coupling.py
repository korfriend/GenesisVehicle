"""Wheel coupling strategies: post-process omega to model mechanical couplings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from ..config import ConfigError


class CouplingStrategy(ABC):
    """Modifies the omega state tensor after wheel-loop integration."""

    @abstractmethod
    def apply(self, omega: torch.Tensor, wheel_meta: Any) -> torch.Tensor:
        """Returns modified omega, shape (n_envs, n_wheels)."""

    def validate(self, wheels: list[Any]) -> None:
        return None


class Independent(CouplingStrategy):
    """No coupling. omega passes through unchanged."""

    def apply(self, omega: torch.Tensor, wheel_meta: Any) -> torch.Tensor:
        return omega


class SameSideBelt(CouplingStrategy):
    """Tank-track coupling: replace each side's omegas with their per-env mean."""

    def apply(self, omega: torch.Tensor, wheel_meta: Any) -> torch.Tensor:
        L = wheel_meta.left_idx
        R = wheel_meta.right_idx
        out = omega.clone()
        if L.numel() > 0:
            out[:, L] = out[:, L].mean(dim=-1, keepdim=True)
        if R.numel() > 0:
            out[:, R] = out[:, R].mean(dim=-1, keepdim=True)
        return out

    def validate(self, wheels: list[Any]) -> None:
        if any(w.side is None for w in wheels):
            raise ConfigError(
                "SameSideBelt requires every wheel to have side='L' or 'R'."
            )
