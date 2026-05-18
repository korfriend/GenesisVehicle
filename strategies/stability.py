"""Stability hooks: batched modifiers for tire forces / wheel omega.

v0.5.0 BREAKING: hook ``apply_post_tire(ctx)`` no longer takes a per-wheel
index. All wheels are processed in a single batched tensor op per hook.
``ctx.F_long``/``ctx.F_lat``/``ctx.N``/``ctx.v_long``/``ctx.v_lat`` are now
``(n_envs, n_wheels)``; per-wheel coefficients are read from
``ctx.wheel_meta`` (the cached batched tensors).
"""

from __future__ import annotations

from abc import ABC
from typing import Any

import torch


class StabilityHook(ABC):
    """Base class. Subclasses set ``slots`` to a subset of
    ``{'PRE_LOOP', 'POST_TIRE'}``."""

    slots: tuple[str, ...] = ()

    def apply_pre_loop(self, ctx: Any) -> None:
        """Called once per step before the per-wheel work. Reads/writes
        ``ctx.moving``, ``ctx.throttle``, ``ctx.brake``, ``ctx.vel``,
        ``ctx.ang``."""

    def apply_post_tire(self, ctx: Any) -> None:
        """Called once per step after the tire model. All per-wheel tensors
        on ctx are ``(n_envs, n_wheels)`` — mutate ``ctx.F_long``,
        ``ctx.F_lat``, set ``ctx.omega_override`` /
        ``ctx.omega_pull_factor`` / ``ctx.omega_pull_target`` to influence
        the omega update."""


# ---------------------------------------------------------------------------
# RollingResistance
# ---------------------------------------------------------------------------


class RollingResistance(StabilityHook):
    """Apply rolling resistance: ``F_long -= cr * N * tanh(v_long / scale)``.
    ``cr`` is per-wheel from ``wheel_meta.rolling_resistance_cr``."""

    slots = ("POST_TIRE",)

    def __init__(self, tanh_scale: float = 0.5):
        self.tanh_scale = float(tanh_scale)

    def apply_post_tire(self, ctx: Any) -> None:
        cr = ctx.wheel_meta.rolling_resistance_cr.unsqueeze(0)   # (1, n_wheels)
        sign_v = torch.tanh(ctx.v_long / self.tanh_scale)         # (n_envs, n_wheels)
        ctx.F_long = ctx.F_long - cr * ctx.N * sign_v


# ---------------------------------------------------------------------------
# LowSpeedRegularizer
# ---------------------------------------------------------------------------


class LowSpeedRegularizer(StabilityHook):
    """HJW low-speed regularizer.

    PRE_LOOP: compute a (n_envs,) ``moving`` factor in [0, 1] from chassis
      linear/angular speed. When ``disable_when_control_active`` and the user
      is on throttle/brake, force ``moving := 1`` so the vehicle can pull
      away from rest.
    POST_TIRE: scale ``F_long``/``F_lat`` by ``moving``; set
      ``omega_pull_factor`` so core blends omega toward
      ``v_long / radius`` (rolling-without-slip) for grounded wheels.
    """

    slots = ("PRE_LOOP", "POST_TIRE")

    def __init__(
        self,
        v_kin_com: float = 0.5,
        ang_kin: float = 0.5,
        disable_when_control_active: bool = True,
        control_threshold: float = 0.01,
    ):
        self.v_kin_com = float(v_kin_com)
        self.ang_kin = float(ang_kin)
        self.disable_when_control_active = bool(disable_when_control_active)
        self.control_threshold = float(control_threshold)

    def apply_pre_loop(self, ctx: Any) -> None:
        v_com_speed = torch.linalg.vector_norm(ctx.vel[:, :2], dim=-1)
        ang_speed = torch.linalg.vector_norm(ctx.ang, dim=-1)
        moving = torch.minimum(v_com_speed / self.v_kin_com, torch.ones_like(v_com_speed))
        moving = torch.maximum(
            moving,
            torch.minimum(ang_speed / self.ang_kin, torch.ones_like(ang_speed)),
        )
        moving = torch.clamp(moving, min=0.0, max=1.0)
        if self.disable_when_control_active:
            control_active = (
                (ctx.throttle.abs() > self.control_threshold) |
                (ctx.brake > self.control_threshold)
            )
            moving = torch.where(control_active, torch.ones_like(moving), moving)
        ctx.moving = moving

    def apply_post_tire(self, ctx: Any) -> None:
        if ctx.moving is None:
            return
        moving_b = ctx.moving.unsqueeze(-1)   # (n_envs, 1)
        ctx.F_long = ctx.F_long * moving_b
        ctx.F_lat = ctx.F_lat * moving_b
        pull = torch.clamp(1.0 - moving_b, min=0.0, max=1.0)   # (n_envs, 1)
        ground = (~ctx.air_mask).to(dtype=ctx.F_long.dtype)    # (n_envs, n_wheels)
        ctx.omega_pull_factor = pull * ground                  # (n_envs, n_wheels)
        radius_b = ctx.wheel_meta.radius.unsqueeze(0)          # (1, n_wheels)
        ctx.omega_pull_target = ctx.v_long / radius_b          # (n_envs, n_wheels)


# ---------------------------------------------------------------------------
# StaticFrictionLock
# ---------------------------------------------------------------------------


class _OmegaOverride:
    """Carried on ctx.omega_override when StaticFrictionLock fires. ``active``
    is a (n_envs, n_wheels) bool mask."""
    __slots__ = ("active",)

    def __init__(self, active: torch.Tensor):
        self.active = active


class StaticFrictionLock(StabilityHook):
    """KDU-style static friction lock.

    When ``brake > brake_thr`` AND ``|v_long| < v_thr`` (per env, per wheel):
        ``F_long := clamp(-hold_k * v_long, +-mu_long * N)``
        ``omega := 0``  (signalled via ``ctx.omega_override.active``)
    """

    slots = ("POST_TIRE",)

    def __init__(
        self,
        brake_thr: float = 0.3,
        v_thr: float = 0.5,
        hold_k: float = 200_000.0,
    ):
        self.brake_thr = float(brake_thr)
        self.v_thr = float(v_thr)
        self.hold_k = float(hold_k)

    def apply_post_tire(self, ctx: Any) -> None:
        # active is (n_envs, n_wheels) bool. We deliberately do NOT short-circuit
        # with `if not active.any()` — that forces a CPU<->GPU sync per step.
        active_brake = (ctx.brake > self.brake_thr).unsqueeze(-1)     # (n_envs, 1)
        active_v = ctx.v_long.abs() < self.v_thr                       # (n_envs, n_wheels)
        active = active_brake & active_v                               # (n_envs, n_wheels)

        mu_long_b = ctx.wheel_meta.mu_long.unsqueeze(0)                # (1, n_wheels)
        mu_N = mu_long_b * ctx.N                                       # (n_envs, n_wheels)
        hold = torch.clamp(-self.hold_k * ctx.v_long, min=-mu_N, max=mu_N)
        ctx.F_long = torch.where(active, hold, ctx.F_long)

        prev = ctx.omega_override
        if prev is None:
            ctx.omega_override = _OmegaOverride(active=active)
        else:
            prev.active = prev.active | active
