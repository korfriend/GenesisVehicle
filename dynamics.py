"""Pure dynamics primitives used by core.py. Extracted so they are testable
without Genesis runtime.

The functions encode the physical contracts the SDK promises:

  - `brake_torque_signed`: convert a positive brake magnitude to a signed
    torque opposing the wheel's current rotation (so -T_brake_eff always
    decelerates the wheel).
  - `suspension_normal_force`: per-wheel asymmetric damper + non-negative
    clamp + air-mask zero. Normal force can never pull the wheel down.
  - `aero_drag_force`: chassis aero drag opposing horizontal velocity.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

ScalarOrTensor = Union[torch.Tensor, float, int]


def aero_drag_force(
    vel: torch.Tensor, drag_area: float, air_density: float = 1.225
) -> torch.Tensor:
    """Chassis aerodynamic drag: ``F = -0.5 * rho * (Cd*A) * |v_h| * v_h``.

    Opposes the **horizontal** chassis velocity only (vertical is zeroed so drag
    never fights the suspension). ``drag_area`` is Cd*A in m^2; ``<= 0`` returns
    a zero force (drag disabled). ``vel`` is ``(..., 3)`` world-frame linear
    velocity; the return has the same shape. (v1.2.4)
    """
    if drag_area <= 0.0:
        return torch.zeros_like(vel)
    v_h = vel.clone()
    v_h[..., 2] = 0.0
    speed = torch.linalg.norm(v_h, dim=-1, keepdim=True)
    return -(0.5 * float(air_density) * float(drag_area)) * speed * v_h


def brake_torque_signed(
    t_brake: torch.Tensor,
    omega: torch.Tensor,
    dt: float = 0.0,
    i_wheel: Optional[torch.Tensor] = None,
    smoothing_scale: float = 0.5,
) -> torch.Tensor:
    """Convert a positive-magnitude brake command into a signed torque
    that opposes wheel rotation, **clamped** so a single discrete step
    cannot reverse the wheel.

    Without clamping (the v0.5.31 behavior), explicit Euler integration
    of a sign-dependent friction torque can overshoot zero:

        omega = +0.1 rad/s, t_brake = 100, I = 0.5, dt = 0.01
        -> T_brake_eff = 100 * tanh(0.2) ~ 19.7 Nm
        -> new_omega = 0.1 - 19.7/0.5 * 0.01 = -0.294        (sign flip!)
        next step: brake_eff = -53 Nm -> new_omega = +0.76  (oscillation grows)

    Visible as "brake sometimes acts like propulsion." Classic stiff-
    friction-instability. The tanh smoothing only helps inside
    ``|omega| < smoothing_scale``; beyond that, tanh saturates to ±1 and
    brake re-introduces the overshoot.

    Fix (when ``dt > 0`` and ``i_wheel`` are passed): cap the magnitude
    of the brake-induced ω change to ``|omega|``. Equivalently, cap
    ``|T_brake_eff|`` at ``|omega| * i_wheel / dt`` — the torque that
    exactly zeroes ω this step. Brake can only decelerate to rest, never
    past it. The remainder (true static hold) is the
    ``StaticFrictionLock`` hook's job.

    Args
    ----
    t_brake          : positive brake magnitude, shape ``(n_envs, n_wheels)``.
    omega            : current wheel ang. velocity, same shape.
    dt               : simulation step. ``0.0`` (default) disables the
                       clamp and reproduces the legacy tanh-only behavior
                       — kept for backward compat in existing tests.
    i_wheel          : per-wheel spin inertia, shape ``(n_wheels,)`` or
                       broadcastable. Required to compute the cap.
    smoothing_scale  : tanh transition width (rad/s) for the sub-clamp
                       regime where |omega| is small enough that tanh
                       is the dominant attenuation.

    Returns
    -------
    Signed torque tensor, same shape as ``t_brake``. Sign matches
    ``omega``'s; consumer subtracts it from ``domega`` so the
    acceleration always opposes spin.
    """
    smooth_signed = t_brake * torch.tanh(omega / smoothing_scale)
    if dt > 0.0 and i_wheel is not None:
        # Cap |T_brake_eff| at the torque that exactly zeroes omega this
        # step. Use raw |omega| so the cap doesn't depend on smoothing.
        T_cap = torch.abs(omega) * i_wheel / dt
        smooth_signed = torch.sign(smooth_signed) * torch.minimum(
            torch.abs(smooth_signed), T_cap,
        )
    return smooth_signed


def suspension_normal_force(
    compression: torch.Tensor,
    comp_rate: torch.Tensor,
    k_susp: ScalarOrTensor,
    c_compression: ScalarOrTensor,
    c_extension: ScalarOrTensor,
    air_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-wheel ray-suspension normal force, batched over ``(n_envs,)``.

    Encodes the asymmetric damper (different coefficient on compression vs
    extension), clamps the result non-negative (the ground cannot pull a
    wheel down), and forces ``N = 0`` for wheels whose ray missed the
    ground (``air_mask = True``).

    Inputs
    ------
    compression : (n_envs,) tensor, ``max(rest_d - d, 0)``.
    comp_rate   : (n_envs,) tensor, ``d(compression)/dt`` (clamped upstream).
    k_susp      : per-wheel scalar spring stiffness [N/m].
    c_compression, c_extension : per-wheel scalar damper coefficients [N s/m].
        ``c_compression`` applies when ``comp_rate > 0`` (wheel compressing),
        ``c_extension`` applies when ``comp_rate <= 0`` (rebound).
    air_mask    : (n_envs,) bool tensor; True where the wheel has lost contact.
    """
    c_compression_f = float(c_compression)
    c_extension_f = float(c_extension)
    c_damp = torch.where(
        comp_rate > 0.0,
        torch.full_like(comp_rate, c_compression_f),
        torch.full_like(comp_rate, c_extension_f),
    )
    N = float(k_susp) * compression + c_damp * comp_rate
    N = torch.clamp(N, min=0.0)
    N = torch.where(air_mask, torch.zeros_like(N), N)
    return N
