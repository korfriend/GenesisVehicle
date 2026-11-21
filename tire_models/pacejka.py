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
    from ``wheel_meta`` (one batched call covers all wheels).

    ``eps_v`` (m/s) is the low-speed floor on ``|v_long|`` in the slip
    definitions; ``lat_scale`` multiplies the lateral friction coefficient.

    :meth:`at_long_force` is a closed form for this model (v1.8.0), so the
    pipeline calls the tyre ONCE per step; a subclass that overrides
    ``__call__`` falls back to the generic grid of
    :meth:`TireModel.at_long_force` (the closed form is only valid for this
    exact combined law)."""

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
        """Closed form of :meth:`TireModel.at_long_force` for this model (v1.8.0).

        The combined law of ``__call__`` scales BOTH components of the pure
        forces by ``1/sqrt(sq)`` when the point lies outside the friction
        ellipse. On the tyre's own combined curve at the slip whose combined
        ``F_long`` equals ``F_c = F_long_target``:

        * if ``(F_c, F_lat_pure)`` lies inside the ellipse the lateral force is
          the pure lateral force ``F_lat_pure`` (the tyre's own value at this
          slip angle);
        * otherwise ``F_lat = sign(F_lat_pure) * mu_lat*N * sqrt(1 -
          (F_c / (mu_long*N))^2)`` (the ellipse remainder).

        ``kappa`` inverts the pure longitudinal magic formula for the pure
        force ``F_c`` implies (3 Newton steps on ``x(1-E) + E*atan(x) =
        tan(asin(F/D)/C)``), signed like ``F_c``. Measured against a K = 512
        grid of the combined law (bound 1e-3 asserted in
        ``tests/test_axle_torque_order.py::test_n9_pacejka_closed_form_matches_a_k512_grid``): slip relative error up to
        4e-4, lateral force error 4.3e-5 of ``mu*N`` (reference car parameters, ``pe_x``
        0.4). Does not read ``v_roll``. A subclass that overrides ``__call__``
        gets the generic grid instead.

        Newton divisor ``g'(x) = (1-E) + E/(1+x^2)``, ``E = pe_x``: for
        ``E < 1`` it is at least ``min(1, 1-E)`` (in ``(0.6, 1]`` at the
        presets' 0.4); ``E = 1`` keeps it positive. Its magnitude is floored
        at 1e-3 with its sign kept (an exact zero becomes +1e-3). The floor
        can engage only where ``|g'| < 1e-3``: for ``0.999 < pe_x <= 1`` at
        large ``|x|``, and for ``pe_x > 1`` near ``x^2 = 1/(E-1)``. No preset reaches
        it (``tests/test_axle_torque_order.py::
        test_preset_tyres_never_reach_the_newton_divisor_floor``).

        ``pe_x > 1`` (accepted by the config and the OSC server, unused by
        every preset): the curve ``x(1-E) + E*atan(x)`` is then not monotone
        and the divisor is truly NEGATIVE for ``x^2 > 1/(E-1)``; the
        sign-kept floor still takes the true Newton step there. Whenever the
        first iterate lies in that region the target is above the curve's
        reachable maximum (the curve's slope is at most 1, so its value at the
        divisor's zero is below that zero), no root exists and ``kappa`` is
        not meaningful -- it can even take the opposite sign to ``F_c``.
        ``kappa`` and its gradients stay finite, but the slip's ACCURACY at
        ``pe_x >= 1`` is not characterised (``F_lat`` does not depend on it).
        Without the floor an EXACTLY zero divisor gave inf and then NaN
        ``kappa`` and gradients (a constructed float32 state, ``tests/
        test_axle_torque_order.py::
        test_closed_form_kappa_stays_finite_for_pe_x_at_or_above_one``). The
        1.8.0 development floor was a ``clamp(min=1e-3)`` that dropped the
        sign, so it reversed the Newton step wherever the divisor was truly
        negative; at one such state (``pe_x`` 2.0, first divisor -0.33) its
        result lies more than 1.0 away from the true Newton iterate, which
        this floor reproduces
        (``tests/test_axle_torque_order.py::
        test_closed_form_keeps_the_newton_direction_where_the_divisor_is_negative``).
        No sweep over ``pe_x > 1`` is shipped with the repo, so no bound on
        ``|kappa|`` there is stated."""
        if type(self).__call__ is not PacejkaAnisotropic.__call__:
            return TireModel.at_long_force(
                self, v_long, v_lat, v_roll, N, wheel_meta,
                F_long_target=F_long_target)
        wm = wheel_meta
        F_c = F_long_target
        mu_N = torch.clamp(wm.mu_long.unsqueeze(0) * N, min=1e-6)
        mu_N_lat = torch.clamp(wm.mu_lat.unsqueeze(0) * self.lat_scale * N, min=1e-6)
        alpha = torch.atan2(v_lat, torch.clamp(torch.abs(v_long), min=self.eps_v))
        F_lat_pure = -_pacejka(alpha, wm.pb_y.unsqueeze(0), wm.pc_y.unsqueeze(0),
                               wm.mu_lat.unsqueeze(0) * self.lat_scale * N,
                               wm.pe_y.unsqueeze(0))
        # Lateral: pure value inside the ellipse, the ellipse remainder outside.
        q = 1.0 - (F_c / mu_N) ** 2
        rem = mu_N_lat * torch.sqrt(torch.where(q > 0, q, torch.ones_like(q))) * (q > 0)
        F_lat = torch.sign(F_lat_pure) * torch.minimum(torch.abs(F_lat_pure), rem)
        inside = torch.abs(F_lat_pure) <= rem
        # Pure longitudinal force the combined point F_c implies (the combined
        # law scales both components by the same factor).
        den = torch.where(torch.abs(F_lat) > 1e-9, torch.abs(F_lat),
                          torch.ones_like(F_lat))
        F_lp = torch.where(inside, torch.abs(F_c),
                           torch.abs(F_c) * torch.abs(F_lat_pure) / den)
        y = torch.clamp(F_lp / mu_N, max=1.0 - 1e-6)
        C = wm.pc_x.unsqueeze(0)
        Bx = wm.pb_x.unsqueeze(0)
        E = wm.pe_x.unsqueeze(0)
        t = torch.tan(torch.asin(y) / C)
        x = t
        for _ in range(3):
            g = x * (1 - E) + E * torch.atan(x) - t
            # Newton divisor g'(x) = (1-E) + E/(1+x^2), even in x. For E < 1
            # it is >= min(1, 1-E) > 0 (E < 0 gives >= 1, 0 <= E < 1 gives
            # > 1-E); E = 1 gives 1/(1+x^2) > 0. The presets use E = 0.4, so
            # it lies in (0.6, 1] and the floor below never engages (bits
            # unchanged). The floor can engage wherever |g'| < 1e-3: for
            # 0.999 < E <= 1 at large |x|, and for E > 1 near g' = 0 (at
            # x^2 = 1/(E-1)); beyond that zero g' is truly negative, tending
            # to 1-E. An exactly-zero divisor gives inf, then NaN -- in
            # DifferentiablePlant a NaN on the untaken torch.where branch
            # poisons every wheel's gradient. So the MAGNITUDE is floored at
            # 1e-3 and the SIGN kept (a small negative g' becomes -1e-3,
            # keeping the Newton direction; an exact 0, either signed zero,
            # becomes +1e-3).
            gp = (1 - E) + E / (1 + x * x)
            gp = torch.where(gp.abs() < 1e-3,
                             torch.where(gp >= 0, torch.full_like(gp, 1e-3),
                                         torch.full_like(gp, -1e-3)),
                             gp)
            x = x - g / gp
        kappa = torch.sign(F_c) * x / Bx
        return F_lat, kappa
