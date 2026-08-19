"""Anisotropic Pacejka tire model with friction-circle clamp — vectorized.

v0.5.0: ``__call__`` accepts batched ``(n_envs, n_wheels)`` inputs and reads
per-wheel coefficients from ``wheel_meta`` (a WheelMeta with batched
tensors). One call covers all wheels in all envs.
"""

from __future__ import annotations

from typing import Any

import torch

from . import TireModel


def _pacejka(
    slip: torch.Tensor,        # (n_envs, n_wheels)
    B: torch.Tensor,           # (1, n_wheels)
    C: torch.Tensor,           # (1, n_wheels)
    D: torch.Tensor,           # (n_envs, n_wheels) — peak (= mu * N)
    E: torch.Tensor,           # (1, n_wheels)
) -> torch.Tensor:
    """Pacejka magic formula, fully batched (broadcasts over (n_envs, n_wheels))."""
    Bs = B * slip
    return D * torch.sin(C * torch.atan(Bs - E * (Bs - torch.atan(Bs))))


class PacejkaAnisotropic(TireModel):
    """Magic formula + friction-circle clamp. Per-wheel B/C/E and mu pulled
    from ``wheel_meta`` (one batched call covers all wheels)."""

    def __init__(self, eps_v: float = 0.5, lat_scale: float = 1.0):
        self.eps_v = float(eps_v)
        self.lat_scale = float(lat_scale)

    def __call__(
        self,
        v_long: torch.Tensor,   # (n_envs, n_wheels)
        v_lat: torch.Tensor,    # (n_envs, n_wheels)
        v_roll: torch.Tensor,   # (n_envs, n_wheels)
        N: torch.Tensor,        # (n_envs, n_wheels)
        wheel_meta: Any,        # WheelMeta with per-wheel coefficient tensors
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        abs_v_long = torch.clamp(torch.abs(v_long), min=self.eps_v)
        kappa = (v_roll - v_long) / abs_v_long
        alpha = torch.atan2(v_lat, abs_v_long)

        mu_long = wheel_meta.mu_long.unsqueeze(0)             # (1, n_wheels)
        mu_lat = wheel_meta.mu_lat.unsqueeze(0) * self.lat_scale
        mu_N = mu_long * N                                     # (n_envs, n_wheels)
        mu_N_lat = mu_lat * N

        pb_x = wheel_meta.pb_x.unsqueeze(0)
        pc_x = wheel_meta.pc_x.unsqueeze(0)
        pe_x = wheel_meta.pe_x.unsqueeze(0)
        pb_y = wheel_meta.pb_y.unsqueeze(0)
        pc_y = wheel_meta.pc_y.unsqueeze(0)
        pe_y = wheel_meta.pe_y.unsqueeze(0)

        F_long = _pacejka(kappa, pb_x, pc_x, mu_N, pe_x)
        F_lat = -_pacejka(alpha, pb_y, pc_y, mu_N_lat, pe_y)

        # Friction-circle clamp (combined long + lat <= unit ellipse).
        #
        # Written as a DOUBLE where so the clamp stays differentiable at zero
        # slip. d/dx sqrt(x) is infinite at x == 0, and `torch.where` evaluates
        # (and backprops through) BOTH branches, so a plain
        # `where(sqrt(sq) > 1, 1/sqrt(sq), 1)` hands back NaN gradients for
        # every wheel that is not sliding — i.e. all of them at a standstill.
        # Pinning the sqrt argument to 1.0 wherever the clamp is inactive keeps
        # it >= 1 on the differentiated path. Forward values are identical to
        # the single-where form (`sq > 1` and `sqrt(sq) > 1` agree for
        # non-negative `sq`).
        mu_N_safe = torch.clamp(mu_N, min=1e-6)
        mu_N_lat_safe = torch.clamp(mu_N_lat, min=1e-6)
        sq = (F_long / mu_N_safe) ** 2 + (F_lat / mu_N_lat_safe) ** 2
        over = sq > 1.0
        ones = torch.ones_like(sq)
        scale = torch.where(over, 1.0 / torch.sqrt(torch.where(over, sq, ones)), ones)
        return F_long * scale, F_lat * scale, kappa, alpha
