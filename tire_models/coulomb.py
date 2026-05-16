"""Isotropic Coulomb tire model.

Lifted from HJW/tire_models.py:coulomb_isotropic. F = -mu N v_slip / |v_slip|
with eps_v floor; serves as a baseline comparison against PacejkaAnisotropic.
"""

from __future__ import annotations

from typing import Any

import torch

from . import TireModel


class CoulombIsotropic(TireModel):
    """Single-mu, slip-vector-opposed friction. mu pulled from wheel_params.mu_long."""

    def __init__(self, eps_v: float = 0.5):
        self.eps_v = eps_v

    def __call__(
        self,
        v_long: torch.Tensor,
        v_lat: torch.Tensor,
        v_roll: torch.Tensor,
        N: torch.Tensor,
        wheel_params: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        slip_long = v_long - v_roll
        slip_lat = v_lat
        slip_mag = torch.sqrt(slip_long * slip_long + slip_lat * slip_lat)
        denom = torch.clamp(slip_mag, min=self.eps_v)

        mu = float(wheel_params.mu_long)   # isotropic: only one mu used
        F_long = -mu * N * slip_long / denom
        F_lat = -mu * N * slip_lat / denom

        # Diagnostics (kept for parity with PacejkaAnisotropic.return signature).
        abs_v_long = torch.clamp(torch.abs(v_long), min=self.eps_v)
        kappa = (v_roll - v_long) / abs_v_long
        alpha = torch.atan2(v_lat, abs_v_long)
        return F_long, F_lat, kappa, alpha
