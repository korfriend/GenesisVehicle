"""Isotropic Coulomb tire model — vectorized.

v0.5.0: accepts batched ``(n_envs, n_wheels)`` inputs and reads ``mu_long``
from ``wheel_meta``. One call covers all wheels in all envs.
"""

from __future__ import annotations

from typing import Any

import torch

from . import TireModel
from .._hotset import derived, row_tensor, square_f64


class CoulombIsotropic(TireModel):
    """Single-mu, slip-vector-opposed friction. ``mu`` per wheel from
    ``wheel_meta.mu_long`` (single mu, isotropic by definition)."""

    def __init__(self, eps_v: float = 0.5):
        self.eps_v = float(eps_v)

    def _prime_derived(self, wheel_meta: Any, device: Any, dtype: Any) -> None:
        """Build-time hook (``_hotset.prime_derived``)."""
        self._eps_squared(torch.zeros((), device=device, dtype=dtype))

    def _eps_squared(self, ref: torch.Tensor) -> torch.Tensor:
        """``eps_v ** 2`` as a ``(1, 1)`` tensor on ``ref``'s device/dtype.

        Hoisted out of ``__call__`` (it was recomputed on every step). Squared
        in python DOUBLE and only then cast — see ``_hotset.square_f64``. A
        post-build write to ``eps_v`` still lands on the next call (the cache
        key carries it); ``tests/test_tire_coulomb_eps.py`` is the gate."""
        return derived(
            self, "_eps2", None, (self.eps_v, ref.device, ref.dtype),
            lambda: row_tensor(square_f64(self.eps_v), ref),
        )

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
        # Clamp INSIDE the sqrt, not after it: `clamp(sqrt(x), min=eps)` leaves
        # a non-slipping wheel sitting exactly on sqrt's infinite-derivative
        # point, which NaNs the gradient the DifferentiablePlant takes through
        # this model. Substituting eps^2 for the argument below the floor gives
        # the identical forward value with a bounded derivative.
        eps2 = self._eps_squared(slip_long)
        sq = slip_long * slip_long + slip_lat * slip_lat
        # broadcast_to, not full_like: `full_like`'s fill_value must be a
        # Number and raises `TypeError: fill_value must be Number, not Tensor`
        # now that eps2 is a per-row tensor (the same class of site as
        # core.py's `torch.full_like(jp, -l_susp)`).
        denom = torch.sqrt(
            torch.where(sq > eps2, sq, torch.broadcast_to(eps2, sq.shape)))

        mu = wheel_meta.mu_long.unsqueeze(0)           # (1, n_wheels)
        F_long = -mu * N * slip_long / denom
        F_lat = -mu * N * slip_lat / denom

        # Diagnostics (signature parity with PacejkaAnisotropic).
        abs_v_long = torch.clamp(torch.abs(v_long), min=self.eps_v)
        kappa = (v_roll - v_long) / abs_v_long
        alpha = torch.atan2(v_lat, abs_v_long)
        return F_long, F_lat, kappa, alpha
