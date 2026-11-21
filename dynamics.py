"""Pure dynamics primitives used by core.py. Extracted so they are testable
without Genesis runtime.

The functions encode the physical contracts the SDK promises:

  - `brake_projection` (v1.8.0): the Coulomb brake, applied to the wheel spin
    before the tyre as a projection toward zero that cannot cross it; the
    capacity left once the wheel stops (``b_rem``) holds the wheel.
  - `cap_resist_torque`: cap an axle RESISTING torque (rolling resistance) so
    one step cannot reverse the wheel (v1.7.0).
  - `deadbeat_scale` / `YawInertia` / `DEADBEAT_G` (v1.8.0): the rigid-body
    scale ``s`` of the low-speed deadbeat bounds (held wheel, lateral,
    ``StaticFrictionLock`` damper) and its inputs.
  - (`brake_torque_signed`, deprecated in 1.8.0, was removed in 2.0.0; the
    brake is `brake_projection`.)
  - `suspension_normal_force`: per-wheel asymmetric damper + non-negative
    clamp + air-mask zero. Normal force can never pull the wheel down.
  - `aero_drag_force`: chassis aero drag opposing horizontal velocity.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple, Optional, Union

import torch

ScalarOrTensor = Union[torch.Tensor, float, int]


#: Gravitational acceleration (m/s^2) the low-speed deadbeat bounds use to turn
#: a wheel's normal load ``N`` into its mass share ``N / DEADBEAT_G`` (v1.8.0).
#: One constant, used by every one of them: the held-wheel bound (D), the
#: lateral bound (Y), the rigid-body scale (Z, :func:`deadbeat_scale`), the
#: ``StaticFrictionLock`` damper floor (C) and its damper bound. It is NOT read
#: from the scene. On flat ground ``N = m_share * |g|``, so under a gravity of
#: magnitude ``|g| != 9.81`` (set through ``VehicleScene(gravity=...)``,
#: ``VehicleScene.set_gravity`` or the OSC server's gravity message) every one
#: of those bounds is ``|g| / 9.81`` times the same bound computed with the
#: TRUE mass share ``N / |g|`` (``s`` included):
#: TIGHTER under a weaker gravity (D and Y over-limit the force, and the SFL
#: damper floor drops below critical, damping ratio about ``sqrt(|g| / 9.81)``),
#: LOOSER under a stronger one -- the unconservative side. Per mode, on flat
#: ground (derived from the bound, not observed): the one-step TRANSLATIONAL
#: factor of the D / Y / SFL-damper bounds is ``1 - s * |g| / 9.81`` (``1 - s``
#: at 9.81, by design), which reaches -1 -- the step-to-step alternation the
#: bounds exist to remove -- at ``|g| = 2 * 9.81 / s``; the YAW mode stays
#: exactly deadbeat at any ``|g|`` while the computed ``s < 1``, because ``s``
#: is built from the same ``N / 9.81`` and absorbs the ratio (at ``s = 1``
#: both modes reach -1 at ``|g| = 19.62``). Measured, reference car (``car_4w.urdf``,
#: ``car_4w_rwd_ackermann``, s 0.856, CPU/WSL2, genesis-world 1.4.5,
#: per_vehicle, dt 0.025 x 4, plane mu 1.0, gravity set live with
#: ``set_gravity``) at |g| 4.905 / 9.81 / 24.5: the lateral-kick cases
#: (throttle 0.3; 1 m/s steer 0.3) showed 0 v_lat / F_lat / yaw-rate sign
#: flips at every |g| (1.7.0, same cases: up to 214 / 219 / 218), and brake 1.0 from
#: 10 m/s stopped with 0 mm back-travel and vmin -0.0047 / -2.7e-5 / -7.7e-5
#: m/s. At 24.5 (translational ``s * |g| / 9.81`` about 2.14) one omega
#: alternation at 0.01 rad/s appeared in the 1 m/s steer-0.3 case, none
#: elsewhere; the predicted translational 2-cycle past 2 was NOT observed in
#: these cases. On a slope with ``|g| = 9.81``, ``N = m g
#: cos(theta)`` and the bounds are conservative by ``cos(theta)``. (Through the
#: first 1.8.0 implementation this comment said the reverse -- "too loose under
#: a weaker gravity, too tight under a stronger one"; that was wrong.)
#: ``DifferentiablePlant`` also assumes 9.81 for its own gravity vector.
DEADBEAT_G = 9.81


class YawInertia(NamedTuple):
    """Inputs of the rigid-body deadbeat scale :func:`deadbeat_scale` (v1.8.0).

    ``i_cg``: the vehicle's composite yaw inertia about its centre of mass
    (kg*m^2), a python float. ``rho_sq``: ``(n_wheels,)`` (or ``(B,
    n_wheels)``) squared HORIZONTAL distance from that centre of mass to each
    wheel's ray origin (m^2), on the wheel meta's device and dtype.

    ``YawInertia(float("inf"), rho_sq)`` gives ``s == 1`` exactly (the
    deadbeat bounds then use each wheel's own mass share, unscaled)."""
    i_cg: float
    rho_sq: torch.Tensor


def yaw_inertia_from_mass_props(mass: float, com: Any, izz: float,
                                positions: torch.Tensor) -> YawInertia:
    """Build :class:`YawInertia` from planar mass properties.

    ``mass`` (kg), ``com`` = centre-of-mass offset from the base-link origin
    (body frame, m; only x and y are used), ``izz`` = yaw inertia about the
    base-link ORIGIN (kg*m^2) — the convention of
    ``control.plant.PlantMassProperties``. ``positions`` are the wheels' ray
    origins in the body frame, ``(n_wheels, 3)`` (or ``(B, n_wheels, 3)``).

    ``i_cg = izz - mass * (cx^2 + cy^2)`` (parallel-axis theorem, in python
    double), ``rho_sq = (x - cx)^2 + (y - cy)^2`` on ``positions``' device and
    dtype. Raises ``ValueError`` when ``i_cg`` is not a positive number: a
    yaw inertia about the centre of mass that is zero, negative or NaN is not
    a physical rigid body, and the scale built from it would turn every
    deadbeat bound non-positive."""
    cx, cy = float(com[0]), float(com[1])
    i_cg = float(izz) - float(mass) * (cx * cx + cy * cy)
    if math.isnan(i_cg) or not i_cg > 0.0:
        raise ValueError(
            f"yaw inertia about the centre of mass is {i_cg!r} (izz={izz!r} "
            f"about the base-link origin, mass={mass!r}, com=({cx!r}, {cy!r})): "
            f"not a physical rigid body. `izz` is the yaw inertia about the "
            f"base-link ORIGIN; if you have it about the centre of mass, pass "
            f"izz_cg + mass*(cx^2 + cy^2).")
    rho_sq = ((positions[..., 0] - cx) ** 2
              + (positions[..., 1] - cy) ** 2).detach().clone()
    return YawInertia(i_cg, rho_sq)


def deadbeat_scale(N: torch.Tensor, yaw_inertia: YawInertia) -> torch.Tensor:
    """Rigid-body scale of the low-speed deadbeat bounds (v1.8.0, "Z").

    ``s = min(1, i_cg / sum_i (N_i / DEADBEAT_G) * rho_i^2)``, one value per
    batch row, shape ``(B, 1)``. ``N`` is ``(B, n_wheels)``.

    Why: each deadbeat bound (held wheel, lateral, ``StaticFrictionLock``
    damper) caps a wheel's force at what stops that wheel's contact in one step
    on its mass share ``N_i / g``. With every wheel acting at once the contact
    velocities update as ``(I - W*G)`` with ``W = J M^-1 J^T``; for loads in
    moment balance about the centre of mass ``M^-1 J^T G J = s0 * diag(1, 1,
    sum_i (N_i/g) rho_i^2 / i_cg)``, so the yaw mode overshoots when
    ``sum_i (N_i/g) rho_i^2 > i_cg`` (a vehicle whose wheels sit far from its
    centre of mass relative to its yaw inertia). ``s`` scales every bound so
    no planar mode overshoots.

    What it does NOT cover: pitch / roll coupling of the horizontal contact
    forces through the contact height; loads NOT in moment balance (the
    off-diagonal terms of ``M^-1 J^T G J``); and with ONE wheel grounded it
    overestimates that wheel's effective mass ``m*I/(I + m*rho^2)`` by up to
    2x. ``g`` is the constant :data:`DEADBEAT_G`, not the scene's gravity."""
    rho_sq = yaw_inertia.rho_sq
    if rho_sq.dim() == 1:
        rho_sq = rho_sq.unsqueeze(0)
    den = (N / DEADBEAT_G * rho_sq).sum(-1, keepdim=True)
    return torch.clamp(yaw_inertia.i_cg / torch.clamp(den, min=1e-9), max=1.0)


def brake_projection(
    omega_d: torch.Tensor,
    t_brake: torch.Tensor,
    dt: float,
    i_wheel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The Coulomb brake as a projection of the wheel spin toward zero (v1.8.0).

    ``omega_d`` is the spin after the resisting and drive torques of this step
    (``_pipeline.compute_wheel_step``: ω, minus the capped rolling
    resistance, plus ``T_drive * dt / i_wheel``); ``t_brake`` the brake torque
    MAGNITUDE (``>= 0``, N*m, no ``tanh`` smoothing). Returns
    ``(omega_t, b_rem)``:

    * where the brake can stop the wheel this step (``t_brake > 0`` and
      ``t_brake >= |omega_d| * i_wheel / dt``) the wheel is HELD:
      ``omega_t = 0`` and ``b_rem = t_brake - |omega_d| * i_wheel / dt`` is
      the brake capacity left over (N*m), which the pipeline uses to hold the
      wheel against the tyre;
    * elsewhere ``omega_t = omega_d - sign(omega_d) * t_brake * dt / i_wheel``
      (same sign as ``omega_d``, or zero) and ``b_rem = 0``.

    Enforced: ``omega_t`` never has the opposite sign of ``omega_d`` (up to
    float rounding of the subtraction), and a zero brake never counts as held
    (the ``t_brake > 0`` term: ``omega_d == 0`` with no brake is NOT held).
    That term has NO forward effect: at ``t_brake = 0`` and ``omega_d = 0``
    both branches give ``omega_t = 0`` and ``b_rem = 0``. It changes only the
    GRADIENT: on the held branch ``d omega_t / d omega_d = 0``, so without it
    ``DifferentiablePlant``'s throttle authority for a free wheel at
    standstill is REDUCED, not removed: ``d a / d throttle`` at throttle 0
    drops to 0.881 (car) / 2.825 (tank), against 1.141 / 3.758 at throttle
    0.01 (and 1.171 / 3.758 at throttle 0 with the term) -- the plant state of
    the test below, wheels at ω = 0, chassis at 2 m/s, CPU, term removed by
    patching ``_pipeline.brake_projection`` (guarded by
    ``tests/test_axle_torque_order.py::test_n22_plant_throttle_gain_is_continuous_at_zero_throttle``
    and the gradient assertion of ``test_n22_a_zero_brake_never_holds``).
    (The first 1.8.0 implementation said "a zero brake never holds" here as if
    it changed the forward result; it does not.)"""
    zeros = torch.zeros_like(omega_d)
    b_mag = t_brake + zeros
    stop_t = torch.abs(omega_d) * i_wheel / dt
    held = (b_mag > 0) & (b_mag >= stop_t)
    omega_t = torch.where(
        held, zeros, omega_d - torch.sign(omega_d) * b_mag / i_wheel * dt)
    b_rem = torch.where(held, b_mag - stop_t, zeros)
    return omega_t, b_rem


def bump_stop_force(
    compression: torch.Tensor,
    rest_stroke: torch.Tensor,
    k_bump: torch.Tensor,
) -> torch.Tensor:
    """Progressive bump-stop force: ``k_bump * max(compression - rest_stroke, 0)``.

    Zero anywhere the suspension is inside its design stroke; a stiff extra
    spring beyond it — the same mechanism a real vehicle uses to bound travel.
    A penalty contact must over-compress to absorb a slope hit or a landing;
    without this term the linear spring lets that transient run several strokes
    deep and the wheel rim visibly sinks below the ground. All args broadcast
    against ``(n_envs, n_wheels)``; ``k_bump == 0`` returns zeros (off). (v1.2.6)
    """
    over = torch.clamp(compression - rest_stroke, min=0.0)
    return k_bump * over


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


def cap_resist_torque(
    t_signed: torch.Tensor,
    omega: torch.Tensor,
    dt: float,
    i_wheel: torch.Tensor,
) -> torch.Tensor:
    """Cap a signed axle RESISTING torque so one step cannot reverse the wheel.

    ``sign(t) * min(|t|, |omega| * i_wheel / dt)``.

    Since 1.8.0 ``_pipeline.compute_wheel_step`` applies it ONCE, to every
    hook's ``ctx.T_resist`` (rolling resistance) against the PRE-step ω, and
    subtracts the result (``T_rr``) from ω before the drive, the brake and the
    tyre. (v1.7.0 applied it a second time, to brake + ``T_resist``, and fed
    the difference to the overshoot clamp; that second application is gone:
    the brake is now :func:`brake_projection`.) The torque carries the sign of
    ``omega`` only because ``cr >= 0``, which
    ``config.check_rolling_resistance_cr`` enforces at ``resolve`` and
    ``build_wheel_meta``; a hook adding its own ``T_resist`` must keep that
    sign itself. For a torque of ω's sign the result stops ω at zero instead
    of flipping it in one explicit-Euler step (up to float rounding). It does
    not bound the other terms of the ω update (drive torque, tyre friction
    torque, hook ω pulls).

    Args
    ----
    t_signed : signed resisting torque, ``(B, n_wheels)``; sign of ``omega``.
    omega    : wheel angular velocity, same shape.
    dt       : step duration (s), > 0.
    i_wheel  : spin inertia, ``(n_wheels,)`` or broadcastable.
    """
    T_cap = torch.abs(omega) * i_wheel / dt
    return torch.sign(t_signed) * torch.minimum(torch.abs(t_signed), T_cap)


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
