"""Anisotropic Pacejka tire model with friction-circle clamp.

Lifted from HJW/tire_models.py:pacejka_anisotropic. Per-wheel coefficients
are read from wheel_params (a WheelConfig) so one instance serves all wheels.
"""

from __future__ import annotations

from typing import Any

import torch

from . import TireModel


def _pacejka(slip: torch.Tensor, B: float, C: float, D: torch.Tensor, E: float) -> torch.Tensor:
    """Pacejka magic formula. D = peak (= mu * N), tensor-valued; B/C/E scalar."""
    Bs = B * slip
    return D * torch.sin(C * torch.atan(Bs - E * (Bs - torch.atan(Bs))))


class PacejkaAnisotropic(TireModel):
    """Magic formula + friction-circle clamp. Coefficients per-wheel via wheel_params."""

    def __init__(self, eps_v: float = 0.5, lat_scale: float = 1.0):
        # eps_v is a numerical floor on |v_long| in slip-ratio denominator.
        # lat_scale lets a single tire model down/up-scale the lateral mu globally
        # (per-wheel mu_lat already exists; lat_scale is an extra knob for kept-API parity).
        self.eps_v = eps_v
        self.lat_scale = lat_scale

    def __call__(
        self,
        v_long: torch.Tensor,
        v_lat: torch.Tensor,
        v_roll: torch.Tensor,
        N: torch.Tensor,
        wheel_params: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        abs_v_long = torch.clamp(torch.abs(v_long), min=self.eps_v)
        kappa = (v_roll - v_long) / abs_v_long
        alpha = torch.atan2(v_lat, abs_v_long)

        mu_long = float(wheel_params.mu_long)
        mu_lat = float(wheel_params.mu_lat) * self.lat_scale
        mu_N = mu_long * N
        mu_N_lat = mu_lat * N

        F_long = _pacejka(kappa, float(wheel_params.pb_x), float(wheel_params.pc_x),
                          mu_N, float(wheel_params.pe_x))
        F_lat = -_pacejka(alpha, float(wheel_params.pb_y), float(wheel_params.pc_y),
                          mu_N_lat, float(wheel_params.pe_y))

        # Friction circle (combined long + lat <= mu N).
        mu_N_safe = torch.clamp(mu_N, min=1e-6)
        mu_N_lat_safe = torch.clamp(mu_N_lat, min=1e-6)
        norm = torch.sqrt(
            (F_long / mu_N_safe) ** 2 + (F_lat / mu_N_lat_safe) ** 2
        )
        scale = torch.where(
            norm > 1.0,
            1.0 / torch.clamp(norm, min=1e-6),
            torch.ones_like(norm),
        )
        F_long = F_long * scale
        F_lat = F_lat * scale
        return F_long, F_lat, kappa, alpha
