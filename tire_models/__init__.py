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

    v1.8.0: the pipeline may call a model MORE THAN ONCE per step, with
    different batch sizes — see :meth:`at_long_force`, whose default
    implementation calls ``__call__`` two more times (once with ``K * B``
    rows, once with ``B`` rows; ``K = AT_LONG_FORCE_GRID``). A model must
    therefore be stateless (no per-call side effects) and accept any batch
    size whose per-wheel axis matches ``wheel_meta``.
    """

    #: Grid size of the default :meth:`at_long_force` (v1.8.0).
    AT_LONG_FORCE_GRID: int = 16

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

    def at_long_force(
        self,
        v_long: torch.Tensor,
        v_lat: torch.Tensor,
        v_roll: torch.Tensor,
        N: torch.Tensor,
        wheel_meta: Any,
        *,
        F_long_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(F_lat, kappa)`` of this tyre at the slip where its OWN combined
        longitudinal force equals ``F_long_target`` (v1.8.0).

        Why: ``_pipeline.compute_wheel_step`` clamps the tyre's longitudinal
        force (the overshoot clamp, the held-wheel deadbeat) to what the wheel
        can actually transmit this step. Where the clamp binds, the lateral
        force and the reported slip must belong to the TRANSMITTED force — a
        friction ellipse evaluated against the raw, unclamped force would cut
        the lateral grip for a longitudinal force that never reached the road.

        Contract:

        * ``F_long_target`` lies between 0 and this step's raw ``F_long`` from
          ``__call__`` at ``v_roll`` (guaranteed by the pipeline: the clamp
          interval always contains 0). All tensors are ``(B, n_wheels)``.
        * The pipeline uses the result only where ``F_long_target`` differs
          from the raw ``F_long``; everywhere else the raw ``__call__`` output
          is kept bit for bit, so this method's value there does not matter.
        * The result must be differentiable (``DifferentiablePlant``
          differentiates through it).

        This default is a root search on a grid: ``K = AT_LONG_FORCE_GRID``
        rolling speeds between ``v_long`` (rolling, slip 0) and ``v_roll``
        (quadratically spaced toward ``v_long``), ONE ``__call__`` with
        ``K * B`` rows, linear interpolation of the first crossing of
        ``F_long_target``, then ONE more ``__call__`` with ``B`` rows at the
        interpolated rolling speed. So the tyre is called THREE times per
        step in total; the model must be stateless and accept any batch size.
        The grid assumes the tyre's longitudinal force is monotone in the
        rolling speed on ``[v_long, v_roll]``.

        Accuracy, measured for the Pacejka model against a K = 512 grid on a
        random grid of states (CPU, reference car parameters; bounds asserted in
        ``tests/test_axle_torque_order.py::test_n9_the_generic_k16_grid_stays_within_its_stated_bound``):
        slip error up to 0.23 relative (p99 0.056), lateral force error up to
        2.9% of ``mu * N``. :class:`PacejkaAnisotropic` overrides this with a
        closed form; this grid is for other models, and for a tyre that does
        not subclass :class:`TireModel` at all (a plain callable with the
        ``__call__`` signature): the pipeline calls this method unbound on
        it."""
        # getattr: the pipeline also calls this unbound on a tyre that does not
        # subclass TireModel (a plain callable, accepted as cfg.tire since
        # before 1.8.0), which has no AT_LONG_FORCE_GRID.
        K = int(getattr(self, "AT_LONG_FORCE_GRID", TireModel.AT_LONG_FORCE_GRID))
        Bsz, n = torch.broadcast_shapes(v_long.shape, v_lat.shape, v_roll.shape,
                                        N.shape, F_long_target.shape)
        u = (torch.arange(1, K + 1, dtype=F_long_target.dtype,
                          device=F_long_target.device) / K) ** 2      # (K,)
        u3 = u.view(-1, 1, 1).expand(K, Bsz, n)

        def _rep(t):
            return t.expand(Bsz, n).unsqueeze(0).expand(K, Bsz, n).reshape(K * Bsz, n)

        v_grid = (v_long.unsqueeze(0) + (v_roll - v_long).unsqueeze(0)
                  * u.view(-1, 1, 1)).expand(K, Bsz, n).reshape(K * Bsz, n)
        F_grid, _, _, _ = self(_rep(v_long), _rep(v_lat), v_grid, _rep(N),
                               wheel_meta)
        f = F_grid.reshape(K, Bsz, n) - F_long_target.unsqueeze(0)
        sgn = torch.sign(F_long_target).unsqueeze(0)
        cross = (f * sgn) >= 0
        cross[-1] = True
        k = torch.argmax(cross.to(torch.int8), dim=0, keepdim=True)    # first crossing
        u_k = u3.gather(0, k)[0]
        f_k = f.gather(0, k)[0]
        k_prev = torch.clamp(k - 1, min=0)
        has_prev = k[0] > 0
        u_p = torch.where(has_prev, u3.gather(0, k_prev)[0], torch.zeros_like(u_k))
        f_p = torch.where(has_prev, f.gather(0, k_prev)[0], -F_long_target)
        den = f_k - f_p
        nz = den != 0
        u_star = u_p + (u_k - u_p) * torch.where(
            nz, -f_p / torch.where(nz, den, torch.ones_like(den)),
            torch.zeros_like(den))
        u_star = torch.where(F_long_target == 0, torch.zeros_like(u_star), u_star)
        v_eq = v_long + (v_roll - v_long) * u_star
        _, F_lat, kappa, _ = self(v_long, v_lat, v_eq, N, wheel_meta)
        return F_lat, kappa


from .pacejka import PacejkaAnisotropic    # noqa: E402  (re-export after ABC defn)
from .coulomb import CoulombIsotropic      # noqa: E402

__all__ = ["TireModel", "PacejkaAnisotropic", "CoulombIsotropic"]
