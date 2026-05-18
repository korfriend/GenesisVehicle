"""Isotropic Coulomb tire model — vectorized.

v0.5.0: accepts batched ``(n_envs, n_wheels)`` inputs and reads ``mu_long``
from ``wheel_meta``. One call covers all wheels in all envs.
"""

from __future__ import annotations

from typing import Any

import torch

from . import TireModel


class CoulombIsotropic(TireModel):
    """Single-mu, slip-vector-opposed friction. ``mu`` per wheel from
    ``wheel_meta.mu_long`` (single mu, isotropic by definition)."""

    def __init__(self, eps_v: float = 0.5):
        self.eps_v = float(eps_v)

    def __call__(
        self,
        v_long: torch.Tensor,   # (n_envs, n_wheels)
        v_lat: torch.Tensor,    # (n_envs, n_wheels)
        v_roll: torch.Tensor,   # (n_envs, n_wheels)
        N: torch.Tensor,        # (n_envs, n_wheels)
        wheel_meta: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        slip_long = v_long - v_roll
        slip_lat = v_lat
        slip_mag = torch.sqrt(slip_long * slip_long + slip_lat * slip_lat)
        denom = torch.clamp(slip_mag, min=self.eps_v)

        mu = wheel_meta.mu_long.unsqueeze(0)           # (1, n_wheels)
        F_long = -mu * N * slip_long / denom
        F_lat = -mu * N * slip_lat / denom

        # Diagnostics (signature parity with PacejkaAnisotropic).
        abs_v_long = torch.clamp(torch.abs(v_long), min=self.eps_v)
        kappa = (v_roll - v_long) / abs_v_long
        alpha = torch.atan2(v_lat, abs_v_long)
        return F_long, F_lat, kappa, alpha
