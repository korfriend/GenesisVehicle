"""Pure dynamics primitives used by core.py. Extracted so they are testable
without Genesis runtime.

The functions encode the physical contracts the SDK promises:

  - `brake_torque_signed`: convert a positive brake magnitude to a signed
    torque opposing the wheel's current rotation (so -T_brake_eff always
    decelerates the wheel).
  - `suspension_normal_force`: per-wheel asymmetric damper + non-negative
    clamp + air-mask zero. Normal force can never pull the wheel down.
"""

from __future__ import annotations

from typing import Union

import torch

ScalarOrTensor = Union[torch.Tensor, float, int]


def brake_torque_signed(
    t_brake: torch.Tensor,
    omega: torch.Tensor,
    smoothing_scale: float = 0.5,
) -> torch.Tensor:
    """Convert a positive-magnitude brake command into a signed torque
    that opposes wheel rotation.

    Returns ``t_brake * tanh(omega / smoothing_scale)``, which has the same
    sign as ``omega``. The pipeline applies it via ``domega -= T_brake_eff``,
    so the resulting acceleration always points against the wheel's spin.

    At ``|omega| < smoothing_scale`` the effective torque is reduced (so brake
    cannot reverse a near-stopped wheel). For a hard hold at rest, use
    ``StaticFrictionLock`` in addition.
    """
    return t_brake * torch.tanh(omega / smoothing_scale)


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
