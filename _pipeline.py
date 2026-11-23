"""Shared per-wheel force pipeline (sections A–E of the ray-wheel step).

Single source of truth for the batched suspension → tire → omega → force
math, used by BOTH `VehiclePhysics.step` (batch = n_envs) and
`MultiVehicleKindPhysics.step` (batch = n_envs · K). Previously this math
was mirrored in the two `step()` methods, which had already drifted — the
v0.6.0 `F_long` overshoot clamp was added to `core.py` but not to the
multi-vehicle copy. Extracting it here removes the duplication and makes a
physics fix land in one place.

The function is batch-shape-agnostic: every tensor is `(B, n_wheels)` or
`(B, n_wheels, 3)` where `B` is whatever batch the caller built its proto
with. It performs no I/O (no entity reads, no solver writes, no visual) —
the caller does all of that around it.

Hook phases (v1.7.0). Hooks run in TWO phases per step, both over the
``post_tire_hooks`` list the caller passes (a hook whose class lacks one of the
two methods is skipped for that phase):

1. ``apply_pre_tire(ctx)`` -- BEFORE the tyre. The only phase that may set
   ``ctx.T_resist`` (``(B, n)``, sign of ``omega``; ``RollingResistance``
   does).
2. ``apply_post_tire(ctx)`` -- after the tyre. Writing ``ctx.T_resist`` here
   raises ``RuntimeError`` -- a rebinding always, an in-place write except
   under ``torch.inference_mode()`` (where the ``_version`` read is skipped;
   the physics still uses the pre-tyre snapshot).

Axle torques before the tyre (v1.8.0). Per wheel, in this order, all applied
to the wheel spin BEFORE the tyre is evaluated:

* resisting torque: ``T_rr = cap_resist_torque(T_resist, omega)`` against the
  pre-step ω, ``omega_r = omega - T_rr * dt / i_wheel`` (the v1.7.0 ω′ line);
* drive torque: ``omega_d = omega_r + T_drive * dt / i_wheel``;
* brake (B1): ``dynamics.brake_projection`` -- the full brake MAGNITUDE (no
  ``tanh``) as a projection toward zero that cannot cross it, giving
  ``omega_t``. Where the brake could stop the wheel this step
  (``T_brake > 0`` and ``T_brake >= |omega_d| * i_wheel / dt``) the wheel is
  HELD: ``omega_t = 0`` and ``B_rem`` (the capacity left) widens the clamp
  both ways and resists spin-up in the ω update. A zero brake never holds.

The tyre is evaluated at ``r * clamp(omega_t, +-omega_max)`` (M2: a wheel
speed the ω update can actually reach). The POST_TIRE hooks get
``ctx.omega = omega_t``.

**What was wrong in this docstring through 1.7.0, said here explicitly:** it
stated that the brake is evaluated on the pre-step ω and "capped together with
the resisting torque ... ``cap_resist_torque(T_brake + T_resist, omega) -
T_rr`` is what enters the clamp", and that with no ``T_resist`` set "``omega'``
IS ``omega`` ... and the code path is the pre-1.7.0 one, operation for
operation". Neither is true since 1.8.0: the brake is the projection above,
nothing of the brake or of ``T_resist`` enters the clamp as a torque, and the
tyre is evaluated at ``omega_t`` (which includes this step's drive and brake),
not at the pre-step ω, whether or not ``T_resist`` is set.

The clamp and the bounds (each a guard, with what it does NOT bound):

* Overshoot clamp (v0.6.0, rewritten 1.8.0): the hooked ``F_long`` is limited
  to ``[min(0, L) - B_rem/r, max(0, L) + B_rem/r]`` with
  ``L = (omega_t - v_long/r) * i_wheel / dt / r`` -- the friction torque
  cannot carry the wheel past rolling in one step, and a held wheel may
  transmit its leftover brake capacity. Exposed as ``ctx.F_long_bounds``.
* D, held-wheel deadbeat: on a wheel with ``B_rem > 0``,
  ``|F_long| <= s * (N/g) * |v_long - r * omega_t| / dt`` -- the force that
  stops the contact slip in one step on the wheel's mass share. Exposed as
  ``ctx.F_long_deadbeat``. Not applied on wheels a hook holds
  (``ctx.hold_applied``). ``B_rem`` itself is exposed as ``ctx.brake_rem``
  (v2.0.4, a detached copy, set before the POST_TIRE phase).
* B2, friction ellipse on the transmitted force: where the clamp binds on the
  RAW tyre force, ``F_lat`` and ``kappa`` are taken from
  ``tire.at_long_force`` at the clamped force (a closed form for
  ``PacejkaAnisotropic``, a grid otherwise). The hooks still receive the raw
  ``F_long``; their ``ctx.F_lat`` is already this value where the clamp binds.
* Y, lateral deadbeat: ``|F_lat| <= s * (N/g) * |v_lat| / dt`` on every wheel
  a hook does not hold -- the lateral counterpart of D, against the explicit
  low-speed lateral 2-cycle of the tyre (per-step gain
  ``G = B_y*C_y*mu*g*dt / max(|v_long|, eps_v)``; ``G > 2`` alternates step
  to step; measured gone with Y + Z on the reference car and tank, the HJW demo
  car and ``tests/data/low_yaw_inertia_car.urdf``, other vehicles untested). It
  binds where the tyre's own explicit gain exceeds ``s``: at static load only
  at low speed (below about ``B_y*C_y*mu*g*dt*(N/N_static)/s``: 2.98 m/s for
  the reference car at dt 0.025, mu 1.0, s 0.856), but under a wheel-load
  spike it binds AT SPEED too (a
  landing, a washboard): it then removes the part of the spike force that
  would reverse the contact's lateral velocity within one step. A hook's
  ``F_lat`` is a request Y may cut.
* Z, rigid-body scale: ``s = min(1, I_cg / sum_i (N_i/g) * rho_i^2)``
  (``dynamics.deadbeat_scale``; ``s = 1`` when no inputs exist) makes D, Y
  and ``StaticFrictionLock``'s damper bound deadbeat in every PLANAR mode, the
  yaw mode included. It does NOT cover the pitch / roll coupling of the
  horizontal contact forces through the contact height, nor loads that are
  not in moment balance about the centre of mass.
* ``g`` is the constant ``dynamics.DEADBEAT_G`` = 9.81, not the scene's
  gravity. Off 9.81 every bound is ``|g|/9.81`` times the same bound computed
  with the true mass share ``N/|g|``: TIGHTER under a weaker gravity (D and Y
  over-limit, the SFL damper floor is under-critical), LOOSER under a
  stronger one -- the unconservative side. The one-step TRANSLATIONAL factor
  is ``1 - s*|g|/9.81``, which stops removing the 2-cycle at
  ``|g| >= 2*9.81/s`` (derived); the YAW mode stays deadbeat at any ``|g|``
  while the computed ``s < 1`` (``s`` absorbs the ratio). At |g| 24.5 on the
  reference car (translational ``s*|g|/9.81`` about 2.14) the measured
  lateral-kick and brake cases showed no 2-cycle (one 0.01 rad/s omega
  alternation in one case), see ``dynamics.DEADBEAT_G``. (The first 1.8.0 implementation said "every bound scales by
  ``9.81/|g|``" here, the reverse direction; that was wrong.)

None of these bound the drive torque (it acts within the step), the
``LowSpeedRegularizer`` ω pull or a hook's ``ctx.omega_override``, which act
on the ω update AFTER the clamp, as before. (Through 2.0.3
``StaticFrictionLock`` set ``ctx.omega_override``; since 2.0.4 it does not.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from genesis.utils.geom import transform_by_quat

from .dynamics import (
    DEADBEAT_G, YawInertia, brake_projection, cap_resist_torque, deadbeat_scale,
)
from .tire_models import TireModel

_NO_YAW_INERTIA_WARNED = False


def _warn_no_yaw_inertia() -> None:
    """One WARNING per process when ``compute_wheel_step`` runs with no
    yaw-inertia inputs at all (neither the keyword nor ``wheel_meta.yaw_inertia``),
    i.e. at ``s = 1``. A ``VehiclePhysics`` always has them after its first
    step (or logged its own per-vehicle warning and set ``s = 1`` explicitly),
    and a ``DifferentiablePlant`` always passes its own, so this fires only for
    a direct caller of the pipeline."""
    global _NO_YAW_INERTIA_WARNED
    if _NO_YAW_INERTIA_WARNED:
        return
    _NO_YAW_INERTIA_WARNED = True
    import logging
    logging.getLogger("genesis_vehicle").warning(
        "[genesis_vehicle:deadbeat-scale] compute_wheel_step ran with no "
        "yaw-inertia inputs (no yaw_inertia= keyword and wheel_meta.yaw_inertia "
        "is None): the low-speed deadbeat bounds use s = 1 (no rigid-body "
        "scale). Logged once per process.")


def pw(t: torch.Tensor) -> torch.Tensor:
    """Broadcast a PER-WHEEL array against a ``(B, n)`` batch.

    ``(n,)`` — one value per wheel, shared by every row — becomes ``(1, n)``.
    ``(B, n)`` — one value per row per wheel — is returned unchanged.

    The rank test is what makes it safe to promote a ``WheelMeta`` field to
    per-row later: ``.unsqueeze(0)`` on an already-``(B, n)`` field would
    silently produce ``(1, B, n)`` and broadcast the WRONG axis. At rank 1 this
    is exactly ``t.unsqueeze(0)``, so it is value-identical today."""
    return t.unsqueeze(0) if t.dim() == 1 else t


def pw3(t: torch.Tensor) -> torch.Tensor:
    """:func:`pw` for per-wheel VECTORS: ``(n, k)`` -> ``(1, n, k)``,
    ``(B, n, k)`` unchanged."""
    return t.unsqueeze(0) if t.dim() == 2 else t


def _rotate_by_quat(v: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """``transform_by_quat`` that also works under ``torch.autograd``.

    Genesis's own ``transform_by_quat`` is a TorchScript kernel that writes its
    result into a preallocated buffer with ``copy_``. When its input carries a
    gradient that buffer is a leaf requiring grad, and the in-place write raises
    ``a leaf Variable that requires grad is being used in an in-place
    operation``. That is exactly what happens in
    :class:`genesis_vehicle.control.DifferentiablePlant`, whose unroll
    differentiates the wheel-frame axes with respect to the steer command (a
    skid-steer vehicle never noticed: its per-wheel steer is a constant zero, so
    nothing downstream of it carries a gradient).

    So: keep the fast kernel for the ordinary simulation path, and take the
    functional quaternion rotation only when a gradient is actually flowing.
    The check is one attribute read per call.
    """
    if v.requires_grad or quat.requires_grad:
        w = quat[..., 0:1]
        u = quat[..., 1:4]
        t = 2.0 * torch.cross(u, v, dim=-1)
        return v + w * t + torch.cross(u, t, dim=-1)
    return transform_by_quat(v, quat)


@dataclass
class PipelineResult:
    total_F: torch.Tensor        # (B, 3)  base-link force
    total_T: torch.Tensor        # (B, 3)  base-link torque
    new_omega: torch.Tensor      # (B, n)  wheel speed after update + coupling
    compression: torch.Tensor    # (B, n)  (store as prev_compression)
    N: torch.Tensor              # (B, n)  diagnostics ↓
    F_long: torch.Tensor
    F_lat: torch.Tensor
    kappa: torch.Tensor
    alpha: torch.Tensor


def compute_wheel_step(
    *,
    pos: torch.Tensor,             # (B, 3)
    quat: torch.Tensor,            # (B, 4)
    vel: torch.Tensor,             # (B, 3)
    ang: torch.Tensor,             # (B, 3)
    distances: torch.Tensor,       # (B, n)
    steer_per_wheel: torch.Tensor, # (B, n)
    T_drive_pw: torch.Tensor,      # (B, n)
    T_brake_pw: torch.Tensor,      # (B, n)
    omega: torch.Tensor,           # (B, n)
    prev_compression: torch.Tensor,# (B, n)
    prev_init: bool,
    wheel_meta: Any,               # WheelMeta (rest_d, k_susp, c_*, radius, i_wheel, comp_rate_clamp)
    resolved: Any,                 # ResolvedConfig (tire, coupling, chassis.omega_max)
    wheel_body_b: torch.Tensor,    # (B, n, 3) wheel attach points in body frame
    up_world: torch.Tensor,        # (B, 3) chassis +z in world
    post_tire_hooks: list,
    ctx: Any,                      # PipelineContext (throttle/brake/vel/ang/wheel_meta pre-set)
    dt: float,
    B: int,
    n: int,
    yaw_inertia: Optional[YawInertia] = None,
) -> PipelineResult:
    """Sections A–E of the ray-wheel pipeline. Pure compute; mutates only
    ``ctx`` (for the hooks) and returns a ``PipelineResult``.

    ``ctx.T_resist`` (reset to None here, set only in a hook's
    ``apply_pre_tire``) is the per-wheel axle resisting torque (v1.7.0). It,
    the drive torque and the brake are applied to ω BEFORE the tyre (v1.8.0,
    see the module docstring); ``ctx.omega`` in the POST_TIRE phase is that ω
    (``omega_t``), not the ω passed in. Through 1.7.0 this docstring said
    the brake was "capped as the remainder in section (D)"; since 1.8.0 it is
    not.

    ``yaw_inertia`` (v1.8.0, keyword, default None): the inputs of the
    rigid-body deadbeat scale (``dynamics.YawInertia``). Precedence: this
    keyword, then ``wheel_meta.yaw_inertia`` (the inputs the driver that owns
    the wheel meta computed), else ``s = 1`` (one
    ``[genesis_vehicle:deadbeat-scale]`` warning per process).
    ``DifferentiablePlant`` passes its OWN inputs here (from its mass
    properties) and never writes the wheel meta it shares with the live
    driver, so a plant built with explicit ``mass=`` / ``izz=`` / ``com=`` runs
    this same code with different Z inputs and its prediction can differ from
    the simulator by design.

    The returned ``kappa`` is the tyre's slip at ``omega_t`` (clipped to
    ``+-omega_max``), replaced by ``kappa_eq`` (the slip of the transmitted
    force on the tyre's own curve) where the clamp binds. ``F_long`` /
    ``F_lat`` are the clamped / deadbeat-limited forces actually applied."""
    wm = wheel_meta
    DT = dt

    # World-space wheel positions: transform (B * n, 3) at once.
    quat_b_flat = quat.unsqueeze(1).expand(B, n, 4).reshape(B * n, 4)
    wheel_body_flat = wheel_body_b.reshape(B * n, 3)
    wheel_world = (_rotate_by_quat(wheel_body_flat, quat_b_flat)
                   .reshape(B, n, 3) + pos.unsqueeze(1))

    # (A) Compression / asymmetric damper / N.
    compression = torch.clamp(pw(wm.rest_d) - distances, min=0.0)
    air_mask = compression <= 0
    if prev_init:
        raw_rate = (compression - prev_compression) / DT
        rc = pw(wm.comp_rate_clamp)
        comp_rate = torch.clamp(raw_rate, -rc, rc)
    else:
        comp_rate = torch.zeros_like(compression)

    c_damp = torch.where(
        comp_rate > 0.0,
        pw(wm.c_compression).expand_as(comp_rate),
        pw(wm.c_extension).expand_as(comp_rate),
    )
    N = pw(wm.k_susp) * compression + c_damp * comp_rate
    # Bump-stop (v1.2.6): progressive extra rate beyond rest_stroke, bounding
    # the transient over-compression a penalty contact needs on a slope hit /
    # landing.
    # `has_bump_stop` is a build-time python bool — zero cost when off, and no
    # per-step device->host sync. It is the OR over every row and wheel of
    # `k_bump > 0` (core.py `_build_wheel_meta_impl`), so it is a BRANCH for
    # the whole batch, not a per-row flag: one wheel with a bump stop makes
    # every row take the branch. That is exact, because `bump_stop_force`
    # returns 0 wherever `k_bump == 0` — a row without a bump stop adds zero.
    if getattr(wm, "has_bump_stop", False):
        from .dynamics import bump_stop_force
        N = N + bump_stop_force(
            compression, pw(wm.rest_d - wm.radius), pw(wm.k_bump))
    N = torch.clamp(N, min=0.0)
    N = torch.where(air_mask, torch.zeros_like(N), N)

    # (B) Wheel-frame fwd/lat (ISO 8855: +steer → fwd rotates +X toward -Y).
    cs = torch.cos(steer_per_wheel)
    ss = torch.sin(steer_per_wheel)
    zer = torch.zeros_like(cs)
    wheel_fwd_local = torch.stack([cs, -ss, zer], dim=-1)
    wheel_lat_local = torch.stack([ss,  cs, zer], dim=-1)
    wheel_fwd_world = _rotate_by_quat(
        wheel_fwd_local.reshape(B * n, 3), quat_b_flat
    ).reshape(B, n, 3)
    wheel_lat_world = _rotate_by_quat(
        wheel_lat_local.reshape(B * n, 3), quat_b_flat
    ).reshape(B, n, 3)

    r_vec = wheel_world - pos.unsqueeze(1)
    ang_b = ang.unsqueeze(1).expand(B, n, 3)
    v_hit = vel.unsqueeze(1) + torch.cross(ang_b, r_vec, dim=-1)
    v_long = (v_hit * wheel_fwd_world).sum(dim=-1)
    v_lat = (v_hit * wheel_lat_world).sum(dim=-1)
    i_w = pw(wm.i_wheel)
    radius_b = pw(wm.radius)

    # [Z] Rigid-body scale of the low-speed deadbeat bounds (v1.8.0), per row,
    # from this step's N. Precedence: the keyword, then the wheel meta's own
    # inputs (set by the driver that owns it), else s = 1 (logged once).
    yi = yaw_inertia if yaw_inertia is not None else getattr(wm, "yaw_inertia", None)
    if yi is None:
        _warn_no_yaw_inertia()
        s = 1.0
    else:
        s = deadbeat_scale(N, yi)                        # (B, 1)

    # PRE_TIRE hook phase (v1.7.0). Every per-step hook output is reset here,
    # before EITHER phase, so a pre-tire hook never sees last step's values.
    # F_long / F_lat do not exist yet in this phase.
    ctx.F_long = None; ctx.F_lat = None; ctx.N = N
    ctx.v_long = v_long; ctx.v_lat = v_lat
    ctx.omega = omega; ctx.air_mask = air_mask
    ctx.omega_override = None
    ctx.omega_pull_factor = None; ctx.omega_pull_target = None
    ctx.T_resist = None
    ctx.F_long_bounds = None; ctx.F_long_deadbeat = None; ctx.hold_applied = None
    ctx.brake_rem = None          # (v2.0.4) set before the POST_TIRE phase below
    ctx.deadbeat_scale = s.clone() if torch.is_tensor(s) else s   # hooks get a copy
    ctx.dt = float(DT)
    for hook in post_tire_hooks:
        pre = getattr(hook, "apply_pre_tire", None)
        if pre is not None:
            pre(ctx)
    T_resist_obj = ctx.T_resist
    if T_resist_obj is not None:
        # Snapshot: only this copy is used below, so a later in-place write to
        # the hook's tensor cannot change the physics. Such a write is also
        # detected through the tensor's version counter (a python int, no
        # device sync) and raises below -- except for an inference-mode
        # tensor, which has no version counter: there only rebinding is
        # detected, and the snapshot still keeps the physics unaffected.
        T_resist_ver = (None if T_resist_obj.is_inference()
                        else T_resist_obj._version)
        T_resist = T_resist_obj.clone()
    else:
        T_resist = None
        T_resist_ver = None

    # (P) Axle torques BEFORE the tyre (v1.8.0): the resisting torque (capped
    # against the pre-step omega, exactly the v1.7.0 omega'), then the drive,
    # then the brake as a Coulomb projection toward zero that cannot cross it
    # (dynamics.brake_projection). B_rem = the brake capacity left on a wheel
    # the brake stopped this step; it holds the wheel below.
    if T_resist is not None:
        T_rr = cap_resist_torque(T_resist, omega, DT, i_w)
        omega_r = omega - T_rr * DT / i_w
    else:
        omega_r = omega
    omega_d = omega_r + T_drive_pw / i_w * DT
    omega_t, B_rem = brake_projection(omega_d, T_brake_pw, DT, i_w)
    # [M2] The tyre sees a wheel speed that can exist: omega_t clipped to
    # +-omega_max (the omega update below is clipped the same way); the clamp
    # budget keeps the unclipped omega_t.
    omega_max = float(resolved.chassis.omega_max)
    omega_e = torch.clamp(omega_t, -omega_max, omega_max)

    # (Q) Bounds on the tyre's F_long, fixed BEFORE the tyre and the hooks.
    # Overshoot clamp: friction cannot carry the wheel past rolling in one
    # step; widened both ways by B_rem / r on a held wheel. Always contains 0.
    omega_target = v_long / radius_b
    F_long_limit = (omega_t - omega_target) * i_w / DT / radius_b
    zeros_f = torch.zeros_like(F_long_limit)
    B_rem_f = B_rem / radius_b
    F_lo = torch.minimum(zeros_f, F_long_limit) - B_rem_f
    F_hi = torch.maximum(zeros_f, F_long_limit) + B_rem_f
    # [D] Held-wheel deadbeat: on a wheel the brake holds, |F_long| <= the
    # force that stops its contact slip in one step on its (Z-scaled) mass
    # share.
    F_db = torch.where(
        B_rem > 0,
        N * s / DEADBEAT_G * torch.abs(v_long - radius_b * omega_t) / DT,
        torch.full_like(omega_t, float("inf")))
    F_lo_d = torch.maximum(F_lo, -F_db)
    F_hi_d = torch.minimum(F_hi, F_db)
    # Hook-visible copies (the pipeline keeps its own: a hook writing these
    # cannot move the clamp).
    ctx.F_long_bounds = (F_lo.clone(), F_hi.clone())
    ctx.F_long_deadbeat = F_db.clone()
    ctx.brake_rem = B_rem.detach().clone()   # (v2.0.4) brake left on a held wheel
    v_roll = radius_b * omega_e

    # (C) Tire force — single batched call.
    F_long, F_lat, kappa, alpha = resolved.tire(v_long, v_lat, v_roll, N, wm)

    # [B2] Where the clamp will bind on the RAW tyre force, the lateral force
    # and the slip are taken on the tyre's own combined curve at the
    # TRANSMITTED force (TireModel.at_long_force: a closed form for
    # PacejkaAnisotropic, a grid otherwise). Elsewhere the raw output is kept
    # bit for bit. The hooks still receive the RAW F_long.
    F_c = torch.maximum(torch.minimum(F_long, F_hi_d), F_lo_d)
    bound = F_c != F_long
    # A tyre that does not subclass TireModel (any callable was accepted as
    # cfg.tire through 1.7.0) has no at_long_force: it gets the generic grid.
    at_long_force = getattr(resolved.tire, "at_long_force", None)
    if at_long_force is None:
        F_lat_eq, kappa_eq = TireModel.at_long_force(
            resolved.tire, v_long, v_lat, v_roll, N, wm, F_long_target=F_c)
    else:
        F_lat_eq, kappa_eq = at_long_force(
            v_long, v_lat, v_roll, N, wm, F_long_target=F_c)
    F_lat = torch.where(bound, F_lat_eq, F_lat)
    kappa = torch.where(bound, kappa_eq, kappa)

    # POST_TIRE stability hooks (batched — one call per hook).
    ctx.F_long = F_long; ctx.F_lat = F_lat
    ctx.omega = omega_t
    for hook in post_tire_hooks:
        post = getattr(hook, "apply_post_tire", None)
        if post is not None:
            post(ctx)
    if ctx.T_resist is not T_resist_obj or (
            T_resist_ver is not None and T_resist_obj._version != T_resist_ver):
        raise RuntimeError(
            "ctx.T_resist was written in apply_post_tire; an axle resisting "
            "torque must be set in apply_pre_tire(ctx), which runs before the "
            "tyre (v1.7.0). A post-tyre write would be silently ignored.")
    F_long, F_lat = ctx.F_long, ctx.F_lat
    hold = ctx.hold_applied

    # (D) The clamp: the hooked F_long into [F_lo_d, F_hi_d] (bounds + held-
    # wheel deadbeat); on wheels a hook HOLDS (StaticFrictionLock after rest)
    # into [F_lo, F_hi] only.
    F_hooked = F_long
    F_long = torch.maximum(torch.minimum(F_hooked, F_hi_d), F_lo_d)
    if hold is not None:
        F_hold = torch.maximum(torch.minimum(F_hooked, F_hi), F_lo)
        F_long = torch.where(hold, F_hold, F_long)

    # Omega update: the tyre friction torque on omega_t, then the remaining
    # brake capacity B_rem resists the spin-up (a projection toward zero that
    # cannot cross it). Airborne: omega_t (drive/brake/resistance only).
    T_friction = radius_b * F_long
    domega = -T_friction / i_w
    new_omega = omega_t + domega * DT
    rem = torch.abs(new_omega) - B_rem / i_w * DT
    new_omega = torch.where(
        B_rem > 0, torch.sign(new_omega) * torch.clamp(rem, min=0.0), new_omega)
    new_omega = torch.where(air_mask, omega_t, new_omega)

    # [Y] Lateral deadbeat: |F_lat| <= the force that stops the contact's
    # lateral velocity in one step on the wheel's (Z-scaled) mass share; not
    # on wheels a hook holds.
    F_lat_db = N * s / DEADBEAT_G * torch.abs(v_lat) / DT
    if hold is not None:
        F_lat_db = torch.where(hold, torch.full_like(F_lat_db, float("inf")), F_lat_db)
    F_lat = torch.maximum(torch.minimum(F_lat, F_lat_db), -F_lat_db)

    new_omega = torch.clamp(new_omega, -omega_max, omega_max)

    if ctx.omega_pull_factor is not None and ctx.omega_pull_target is not None:
        pull = ctx.omega_pull_factor
        new_omega = new_omega * (1.0 - pull) + ctx.omega_pull_target * pull
    if ctx.omega_override is not None:
        new_omega = torch.where(
            ctx.omega_override.active, torch.zeros_like(new_omega), new_omega,
        )

    # (E) Force accumulation (batched).
    F_world = (
        N.unsqueeze(-1) * up_world.unsqueeze(1)
        + F_long.unsqueeze(-1) * wheel_fwd_world
        + F_lat.unsqueeze(-1) * wheel_lat_world
    )
    torque = torch.cross(r_vec, F_world, dim=-1)
    total_F = F_world.sum(dim=1)
    total_T = torque.sum(dim=1)

    # [AERO DRAG] chassis-level force opposing HORIZONTAL velocity (v1.2.4).
    # F = -0.5 * rho * (Cd*A) * |v_h| * v_h, folded into the chassis force that
    # is applied once per step — no extra solver call. Read live from
    # resolved.chassis so drag_area / air_density can be tuned AT RUNTIME (mutate
    # physics.resolved.chassis mid-drive). Vertical velocity is excluded so drag
    # never fights the suspension. drag_area == 0 (default) is a no-op.
    ch = resolved.chassis
    drag_area = getattr(ch, "drag_area", 0.0) or 0.0
    if drag_area > 0.0:
        from .dynamics import aero_drag_force
        total_F = total_F + aero_drag_force(vel, drag_area, ch.air_density)

    # [COUPLING] — affects omega for the next step only.
    new_omega = resolved.coupling.apply(new_omega, wm)

    return PipelineResult(
        total_F=total_F, total_T=total_T, new_omega=new_omega,
        compression=compression, N=N, F_long=F_long, F_lat=F_lat,
        kappa=kappa, alpha=alpha,
    )
