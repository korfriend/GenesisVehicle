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

Axle resisting torque (v1.7.0). Hooks run in TWO phases per step, both over
the ``post_tire_hooks`` list the caller passes (a hook whose class lacks one
of the two methods is skipped for that phase):

1. ``apply_pre_tire(ctx)`` -- BEFORE the tyre. The only phase that may set
   ``ctx.T_resist`` (``(B, n)``, sign of ``omega``; ``RollingResistance``
   does). The pipeline caps it against the pre-step ω with
   ``dynamics.cap_resist_torque`` (``T_rr``) and applies it to ω at once:
   ``omega' = omega - T_rr * dt / i_wheel``. The tyre, the POST_TIRE hooks, the
   F_long overshoot clamp, the ω update and the airborne branch all start from
   ``omega'``. So the tyre sees the ω the resisting torque leaves. On a
   freely rolling wheel whose tyre request reaches the overshoot-clamp limit,
   the clamp passes exactly ``-T_rr / r`` to the chassis on the SAME step and
   ω ends at rolling (pinned for the reference car and tank at +-2 and 5 m/s
   by ``test_a_rolling_wheel_stays_rolling_and_the_chassis_gets_the_rr_force``;
   a tyre whose request stays below that limit passes less). (The
   intermediate 1.7.0 design added ``T_resist`` to the brake inside the clamp
   instead; the clamp target is exact rolling, which is not the resistance
   equilibrium, and ω alternated step to step between rolling and
   rolling - ``T_rr*dt/i``.)
2. ``apply_post_tire(ctx)`` -- after the tyre, as before. Writing
   ``ctx.T_resist`` here raises ``RuntimeError`` -- a rebinding always, an
   in-place write except under ``torch.inference_mode()`` (where the
   ``_version`` read is skipped; the physics still uses the pre-tyre snapshot).

The brake is still evaluated on the PRE-step ω and capped together with the
resisting torque exactly as before: ``cap_resist_torque(T_brake +
T_resist, omega) - T_rr`` is what enters the clamp. Brake + resisting torque
therefore cannot reverse ω in one step -- guarded by ``cap_resist_torque`` on
``T_resist`` in phase 1 plus the cap of the sum here (the remainder is at most
``|omega'| * i_wheel / dt`` and has ω's sign, for a ``T_resist`` of ω's sign
and up to float rounding -- a capped step can land a few ulp past zero). That guard does not bound the
drive torque, the tyre friction torque or a hook's ω pull. With no
``T_resist`` set, ``omega'`` IS ``omega`` (the same tensor object) and the
code path is the pre-1.7.0 one, operation for operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from genesis.utils.geom import transform_by_quat

from .dynamics import brake_torque_signed, cap_resist_torque


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
) -> PipelineResult:
    """Sections A–E of the ray-wheel pipeline. Pure compute; mutates only
    ``ctx`` (for the hooks) and returns a ``PipelineResult``.

    ``ctx.T_resist`` (reset to None here, set only in a hook's
    ``apply_pre_tire``) is the per-wheel axle resisting torque (v1.7.0): it is
    capped and applied to ω BEFORE the tyre, and the brake is capped as the
    remainder in section (D) -- see the module docstring. When it is set,
    ``ctx.omega`` in the POST_TIRE phase and the returned ``kappa`` are taken
    at that post-resistance ω, not at the ω passed in."""
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

    # PRE_TIRE hook phase (v1.7.0). Every per-step hook output is reset here,
    # before EITHER phase, so a pre-tire hook never sees last step's values.
    # F_long / F_lat do not exist yet in this phase.
    ctx.F_long = None; ctx.F_lat = None; ctx.N = N
    ctx.v_long = v_long; ctx.v_lat = v_lat
    ctx.omega = omega; ctx.air_mask = air_mask
    ctx.omega_override = None
    ctx.omega_pull_factor = None; ctx.omega_pull_target = None
    ctx.T_resist = None
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
        # Axle resisting torque applied to omega BEFORE the tyre, capped so it
        # alone cannot reverse the wheel.
        T_rr = cap_resist_torque(T_resist, omega, DT, i_w)
        omega_t = omega - T_rr * DT / i_w
    else:
        T_resist = None
        T_resist_ver = None
        omega_t = omega     # same object: the no-resistance path is unchanged
    v_roll = pw(wm.radius) * omega_t

    # (C) Tire force — single batched call.
    F_long, F_lat, kappa, alpha = resolved.tire(v_long, v_lat, v_roll, N, wm)

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

    # (D) Omega update. brake torque clamped against single-step overshoot.
    # The brake is evaluated on the PRE-step omega (unchanged); with a
    # resisting torque present the brake + resistance SUM is capped as before
    # and only the remainder after the part already applied (T_rr) enters
    # the clamp: |remainder| <= |omega_t| * i_wheel / dt, sign of omega.
    T_brake_eff = brake_torque_signed(T_brake_pw, omega, dt=DT, i_wheel=i_w)
    if T_resist is not None:
        T_brake_eff = cap_resist_torque(T_brake_eff + T_resist, omega, DT, i_w) - T_rr

    # [Overshoot Clamp] Cap F_long so the resulting friction torque cannot
    # reverse the tire slip direction in one step.
    omega_target = v_long / radius_b
    domega_nofric = (T_drive_pw - T_brake_eff) / i_w
    omega_nofric = omega_t + domega_nofric * DT
    T_fric_limit = (omega_nofric - omega_target) * i_w / DT
    F_long_limit = T_fric_limit / radius_b
    F_long = torch.where(
        omega_nofric > omega_target,
        torch.maximum(torch.zeros_like(F_long), torch.minimum(F_long, F_long_limit)),
        torch.minimum(torch.zeros_like(F_long), torch.maximum(F_long, F_long_limit)),
    )

    T_friction = radius_b * F_long
    domega = (T_drive_pw - T_brake_eff - T_friction) / i_w
    new_omega = omega_t + domega * DT
    domega_air = (T_drive_pw - T_brake_eff) / i_w
    new_omega_air = omega_t + domega_air * DT
    new_omega = torch.where(air_mask, new_omega_air, new_omega)
    omega_max = float(resolved.chassis.omega_max)
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
