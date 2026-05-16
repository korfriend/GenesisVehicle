"""Stability hooks: pluggable mid-pipeline modifiers for tire forces / wheel omega.

Two slots: PRE_LOOP (called once before the per-wheel loop, gets chassis state)
and POST_TIRE (called once per wheel, after the tire model has produced F_long
and F_lat for that wheel).
"""

from __future__ import annotations

from abc import ABC
from typing import Any

import torch


class StabilityHook(ABC):
    """Base class for stability hooks. Subclasses set `slots` to indicate which
    of {'PRE_LOOP', 'POST_TIRE'} they implement."""

    slots: tuple[str, ...] = ()

    def apply_pre_loop(self, ctx: Any) -> None:
        """Called once per step before the per-wheel loop. Can read/write ctx
        fields like `moving`, `throttle`, `brake`."""

    def apply_post_tire(self, ctx: Any, wheel_idx: int) -> None:
        """Called once per wheel after the tire model. Mutates ctx.F_long,
        ctx.F_lat, ctx.omega_override (an optional dataclass to force omega)."""


# ---------------------------------------------------------------------------
# RollingResistance
# ---------------------------------------------------------------------------


class RollingResistance(StabilityHook):
    """Apply rolling resistance to F_long (opposes chassis motion).

    cr is read from wheel_params.rolling_resistance_cr (per-wheel).
    """

    slots = ("POST_TIRE",)

    def __init__(self, tanh_scale: float = 0.5):
        self.tanh_scale = float(tanh_scale)

    def apply_post_tire(self, ctx: Any, wheel_idx: int) -> None:
        cr = float(ctx.wheel_params[wheel_idx].rolling_resistance_cr)
        sign_v = torch.tanh(ctx.v_long / self.tanh_scale)
        ctx.F_long = ctx.F_long - cr * ctx.N * sign_v


# ---------------------------------------------------------------------------
# LowSpeedRegularizer
# ---------------------------------------------------------------------------


class LowSpeedRegularizer(StabilityHook):
    """HJW low-speed regularizer.

    PRE_LOOP: compute a `moving` scalar in [0, 1] from chassis linear/angular
              speed; if user controls are active (throttle or brake non-trivial),
              force moving = 1 to allow the vehicle to pull away from rest.
    POST_TIRE: scale F_long/F_lat by `moving`; pull omega toward v_long/radius
               by (1 - moving) for grounded wheels.
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

    def apply_post_tire(self, ctx: Any, wheel_idx: int) -> None:
        if ctx.moving is None:
            return
        ctx.F_long = ctx.F_long * ctx.moving
        ctx.F_lat = ctx.F_lat * ctx.moving
        # omega pull: store target on ctx so core.py can blend after the air-mask branch.
        pull = torch.clamp(1.0 - ctx.moving, min=0.0, max=1.0)
        ground = (~ctx.air_mask).to(dtype=ctx.F_long.dtype)
        ctx.omega_pull_factor = pull * ground
        ctx.omega_pull_target = ctx.v_long / float(ctx.wheel_params[wheel_idx].radius)


# ---------------------------------------------------------------------------
# StaticFrictionLock
# ---------------------------------------------------------------------------


class _OmegaOverride:
    """Helper carried on ctx.omega_override when StaticFrictionLock fires."""
    __slots__ = ("active",)

    def __init__(self, active: torch.Tensor):
        self.active = active


class StaticFrictionLock(StabilityHook):
    """KDU-style static friction lock.

    When brake > brake_thr AND |v_long| < v_thr:
      F_long := clamp(-hold_k * v_long, +-mu_long * N)
      omega[:, i] := 0   (signalled via ctx.omega_override.active)
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

    def apply_post_tire(self, ctx: Any, wheel_idx: int) -> None:
        active = (ctx.brake > self.brake_thr) & (ctx.v_long.abs() < self.v_thr)
        if not bool(active.any()):
            return
        mu_long = float(ctx.wheel_params[wheel_idx].mu_long)
        mu_N = mu_long * ctx.N
        hold = -self.hold_k * ctx.v_long
        hold = torch.clamp(hold, min=-mu_N, max=mu_N)
        ctx.F_long = torch.where(active, hold, ctx.F_long)
        prev = ctx.omega_override
        if prev is None:
            ctx.omega_override = _OmegaOverride(active=active)
        else:
            prev.active = prev.active | active
