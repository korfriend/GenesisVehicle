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
    """Position-anchored stick-slip static friction lock (v0.5.7).

    When ``brake > brake_thr`` AND planar wheel speed
    ``sqrt(v_long^2 + v_lat^2) < v_thr`` (per env, per wheel), the lock
    engages a 2D spring-damper around a per-wheel **anchor** that tracks
    where the contact was when the lock first engaged. Force is

        ``F_long  = -K_spring·d_long  - K_damp·v_long``
        ``F_lat   = -K_spring·d_lat   - K_damp·v_lat``

    where ``d_{long,lat}`` are displacements from the anchor (integrated from
    ``v_{long,lat}·dt`` per step). The force is then projected onto the
    per-wheel friction ellipse:

        ``(F_long / mu_long·N)^2 + (F_lat / mu_lat·N)^2  <=  1``

    If projection clamps (``norm > 1``), the wheel is **slipping** —
    the anchor advances toward the contact (kinetic-friction mode); spring
    cannot pull the contact back past what friction allows. If the projected
    force stays inside the ellipse, the wheel is **stuck** — the spring
    keeps it pinned to within ``F_max / K_spring`` of the anchor.

    Why stick-slip rather than velocity damping (v0.5.7 vs v0.5.6 tanh):
      A pure velocity damper (``F = -K·v`` or ``F = -mu·N·tanh(v/scale)``)
      is fundamentally a *kinetic*-friction model: any non-zero ``v``
      produces an opposing force, but ``v=0`` produces ``F=0``. On a side
      slope where gravity continuously biases ``v_lat``, the damper reaches
      an equilibrium where ``F_damper = F_gravity``, which requires *non-zero*
      ``v_lat`` (~5 mm/s on the truck preset @ 20°). Result: continuous
      creep, even though physically a tire with ``mu = 1.0`` should EASILY
      hold on a 20° slope (``mu > tan(20°) = 0.364``).
      A position-anchored spring + ellipse projection is real *static*
      friction: at ``v = 0`` the force is whatever ``-K_spring·d`` evaluates
      to (which balances any external force up to ``mu·N``). The vehicle
      truly stops when external force ``<= mu·N``.

    Tuning:
      - ``k_spring`` (per wheel): stiffness. Spring's max-allowable
        displacement before friction breaks is ``mu·N / K``; with the truck
        preset (``mu·N ≈ 8000 N`` per wheel) and ``K = 500_000 N/m`` this is
        16 mm. Setting K too high → discretization instability; too low →
        visible creep before stick. Default ``500_000`` is tuned around the
        explicit-Euler bound at ``dt = 0.02`` for a 5-ton chassis.
      - ``k_damp`` (per wheel): velocity damping in addition to the spring.
        Provides settling. Default ``20_000`` is ~half-critical for a 5-ton
        chassis at ``omega_n ≈ 24 rad/s``.
      - ``brake_thr``, ``v_thr``: activation thresholds.

    BREAKING from v0.5.6:
      The old ``hold_k``, ``hold_k_lat``, ``slip_scale`` constructor kwargs
      are GONE — they were proportional / tanh gains in the prior velocity-
      damper model and have no meaning in the stick-slip model. Replace with
      ``k_spring`` / ``k_damp`` (or just drop them and accept the defaults).
      Pre-1.0 SDK; no deprecation shim.
    """

    slots = ("POST_TIRE",)

    def __init__(
        self,
        brake_thr: float = 0.3,
        v_thr: float = 0.5,
        k_spring: float = 500_000.0,
        k_damp: float = 20_000.0,
    ):
        self.brake_thr = float(brake_thr)
        self.v_thr = float(v_thr)
        self.k_spring = float(k_spring)
        self.k_damp = float(k_damp)
        # Per-wheel state, allocated lazily in apply_post_tire when shape is
        # known. (n_envs, n_wheels) — displacement of contact from anchor.
        self._d_long: torch.Tensor | None = None
        self._d_lat: torch.Tensor | None = None
        self._was_active: torch.Tensor | None = None

    def _ensure_state(self, ref: torch.Tensor) -> None:
        if self._d_long is None or self._d_long.shape != ref.shape:
            self._d_long = torch.zeros_like(ref)
            self._d_lat = torch.zeros_like(ref)
            self._was_active = torch.zeros(ref.shape, device=ref.device, dtype=torch.bool)

    def apply_post_tire(self, ctx: Any) -> None:
        # Activation: brake on + low planar speed. We deliberately do NOT
        # short-circuit with `if not active.any()` — that forces a CPU<->GPU
        # sync per step.
        active_brake = (ctx.brake > self.brake_thr).unsqueeze(-1)      # (n_envs, 1)
        v_planar_sq  = ctx.v_long * ctx.v_long + ctx.v_lat * ctx.v_lat  # (n_envs, n_wheels)
        active_v     = v_planar_sq < (self.v_thr * self.v_thr)
        active       = active_brake & active_v                          # (n_envs, n_wheels)

        self._ensure_state(ctx.v_long)
        dt = ctx.dt

        # On rising edge of active (just-engaged), reset anchor → d = 0.
        # On falling edge, d is reset on the next engage; harmless if left.
        just_engaged = active & (~self._was_active)
        self._d_long = torch.where(just_engaged, torch.zeros_like(self._d_long), self._d_long)
        self._d_lat  = torch.where(just_engaged, torch.zeros_like(self._d_lat),  self._d_lat)

        # Integrate displacement (only while active).
        self._d_long = torch.where(
            active, self._d_long + ctx.v_long * dt, self._d_long)
        self._d_lat = torch.where(
            active, self._d_lat  + ctx.v_lat  * dt, self._d_lat)

        # Spring + damper force (will be projected onto friction ellipse below).
        hold_long = -self.k_spring * self._d_long - self.k_damp * ctx.v_long
        hold_lat  = -self.k_spring * self._d_lat  - self.k_damp * ctx.v_lat

        # Friction ellipse — same shape as `tire_models/pacejka.py`.
        mu_long_b = ctx.wheel_meta.mu_long.unsqueeze(0)                # (1, n_wheels)
        mu_lat_b  = ctx.wheel_meta.mu_lat.unsqueeze(0)
        mu_N      = mu_long_b * ctx.N                                  # (n_envs, n_wheels)
        mu_N_lat  = mu_lat_b  * ctx.N
        mu_N_safe     = torch.clamp(mu_N,     min=1e-6)
        mu_N_lat_safe = torch.clamp(mu_N_lat, min=1e-6)
        norm = torch.sqrt(
            (hold_long / mu_N_safe)     ** 2
            + (hold_lat / mu_N_lat_safe) ** 2
        )
        slipping = norm > 1.0                                          # (n_envs, n_wheels)
        scale = torch.where(
            slipping, 1.0 / torch.clamp(norm, min=1e-6), torch.ones_like(norm))
        hold_long_clamped = hold_long * scale
        hold_lat_clamped  = hold_lat  * scale

        # When slipping, advance the anchor so the spring force exactly
        # matches the friction limit at the next step (kinetic-friction
        # behaviour: anchor drifts along with the contact). Without this
        # the anchor would keep growing → spring would build a runaway
        # pre-load that snaps back when the lock re-engages.
        # We solve for the anchor that makes -K_spring·d_new = hold_clamped,
        # ignoring the damper contribution (small at slip moment by
        # construction of the ellipse): d_new = -hold_clamped / K_spring.
        new_d_long = -hold_long_clamped / self.k_spring
        new_d_lat  = -hold_lat_clamped  / self.k_spring
        advance_mask = slipping & active
        self._d_long = torch.where(advance_mask, new_d_long, self._d_long)
        self._d_lat  = torch.where(advance_mask, new_d_lat,  self._d_lat)

        # Apply forces only where active. (When N=0 the friction ellipse
        # has μ·N ≈ 0 so `norm` is huge and `scale ≈ 0` → hold_clamped ≈ 0
        # automatically, so airborne wheels naturally get zero hold force
        # without an explicit ~air_mask gate.)
        ctx.F_long = torch.where(active, hold_long_clamped, ctx.F_long)
        ctx.F_lat  = torch.where(active, hold_lat_clamped,  ctx.F_lat)

        # Omega override (force ω=0 on locked wheels). This is independent
        # of the spring-damper math — brake is engaged, wheel does not spin.
        prev = ctx.omega_override
        if prev is None:
            ctx.omega_override = _OmegaOverride(active=active)
        else:
            prev.active = prev.active | active

        # Persist edge-detect state for next step.
        self._was_active = active
