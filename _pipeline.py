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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from genesis.utils.geom import transform_by_quat

from .dynamics import brake_torque_signed


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
    ``ctx`` (for the POST_TIRE hooks) and returns a ``PipelineResult``."""
    wm = wheel_meta
    DT = dt

    # World-space wheel positions: transform (B * n, 3) at once.
    quat_b_flat = quat.unsqueeze(1).expand(B, n, 4).reshape(B * n, 4)
    wheel_body_flat = wheel_body_b.reshape(B * n, 3)
    wheel_world = (transform_by_quat(wheel_body_flat, quat_b_flat)
                   .reshape(B, n, 3) + pos.unsqueeze(1))

    # (A) Compression / asymmetric damper / N.
    compression = torch.clamp(wm.rest_d.unsqueeze(0) - distances, min=0.0)
    air_mask = compression <= 0
    if prev_init:
        raw_rate = (compression - prev_compression) / DT
        rc = wm.comp_rate_clamp.unsqueeze(0)
        comp_rate = torch.clamp(raw_rate, -rc, rc)
    else:
        comp_rate = torch.zeros_like(compression)

    c_damp = torch.where(
        comp_rate > 0.0,
        wm.c_compression.unsqueeze(0).expand_as(comp_rate),
        wm.c_extension.unsqueeze(0).expand_as(comp_rate),
    )
    N = wm.k_susp.unsqueeze(0) * compression + c_damp * comp_rate
    N = torch.clamp(N, min=0.0)
    N = torch.where(air_mask, torch.zeros_like(N), N)

    # (B) Wheel-frame fwd/lat (ISO 8855: +steer → fwd rotates +X toward -Y).
    cs = torch.cos(steer_per_wheel)
    ss = torch.sin(steer_per_wheel)
    zer = torch.zeros_like(cs)
    wheel_fwd_local = torch.stack([cs, -ss, zer], dim=-1)
    wheel_lat_local = torch.stack([ss,  cs, zer], dim=-1)
    wheel_fwd_world = transform_by_quat(
        wheel_fwd_local.reshape(B * n, 3), quat_b_flat
    ).reshape(B, n, 3)
    wheel_lat_world = transform_by_quat(
        wheel_lat_local.reshape(B * n, 3), quat_b_flat
    ).reshape(B, n, 3)

    r_vec = wheel_world - pos.unsqueeze(1)
    ang_b = ang.unsqueeze(1).expand(B, n, 3)
    v_hit = vel.unsqueeze(1) + torch.cross(ang_b, r_vec, dim=-1)
    v_long = (v_hit * wheel_fwd_world).sum(dim=-1)
    v_lat = (v_hit * wheel_lat_world).sum(dim=-1)
    v_roll = wm.radius.unsqueeze(0) * omega

    # (C) Tire force — single batched call.
    F_long, F_lat, kappa, alpha = resolved.tire(v_long, v_lat, v_roll, N, wm)

    # POST_TIRE stability hooks (batched — one call per hook).
    ctx.F_long = F_long; ctx.F_lat = F_lat; ctx.N = N
    ctx.v_long = v_long; ctx.v_lat = v_lat
    ctx.omega = omega; ctx.air_mask = air_mask
    ctx.omega_override = None
    ctx.omega_pull_factor = None; ctx.omega_pull_target = None
    ctx.dt = float(DT)
    for hook in post_tire_hooks:
        hook.apply_post_tire(ctx)
    F_long, F_lat = ctx.F_long, ctx.F_lat

    # (D) Omega update. brake torque clamped against single-step overshoot.
    i_w = wm.i_wheel.unsqueeze(0)
    T_brake_eff = brake_torque_signed(T_brake_pw, omega, dt=DT, i_wheel=i_w)
    radius_b = wm.radius.unsqueeze(0)

    # [Overshoot Clamp] Cap F_long so the resulting friction torque cannot
    # reverse the tire slip direction in one step.
    omega_target = v_long / radius_b
    domega_nofric = (T_drive_pw - T_brake_eff) / i_w
    omega_nofric = omega + domega_nofric * DT
    T_fric_limit = (omega_nofric - omega_target) * i_w / DT
    F_long_limit = T_fric_limit / radius_b
    F_long = torch.where(
        omega_nofric > omega_target,
        torch.maximum(torch.zeros_like(F_long), torch.minimum(F_long, F_long_limit)),
        torch.minimum(torch.zeros_like(F_long), torch.maximum(F_long, F_long_limit)),
    )

    T_friction = radius_b * F_long
    domega = (T_drive_pw - T_brake_eff - T_friction) / i_w
    new_omega = omega + domega * DT
    domega_air = (T_drive_pw - T_brake_eff) / i_w
    new_omega_air = omega + domega_air * DT
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

    # [COUPLING] — affects omega for the next step only.
    new_omega = resolved.coupling.apply(new_omega, wm)

    return PipelineResult(
        total_F=total_F, total_T=total_T, new_omega=new_omega,
        compression=compression, N=N, F_long=F_long, F_lat=F_lat,
        kappa=kappa, alpha=alpha,
    )
