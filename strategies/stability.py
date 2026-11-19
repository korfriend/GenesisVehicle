"""Stability hooks: batched modifiers for tire forces / wheel omega.

v0.5.0 BREAKING: hook ``apply_post_tire(ctx)`` no longer takes a per-wheel
index. All wheels are processed in a single batched tensor op per hook.
``ctx.F_long``/``ctx.F_lat``/``ctx.N``/``ctx.v_long``/``ctx.v_lat`` are now
``(n_envs, n_wheels)``; per-wheel coefficients are read from
``ctx.wheel_meta`` (the cached batched tensors).

v1.7.0: a hook in the ``'POST_TIRE'`` slot gets TWO calls per step --
``apply_pre_tire(ctx)`` before the tyre (the only phase that may set
``ctx.T_resist``) and ``apply_post_tire(ctx)`` after it. Both default to no-ops
on :class:`StabilityHook`.
"""

from __future__ import annotations

from abc import ABC
from typing import Any

import torch

from .._hotset import derived, row_tensor, square_f64
from .._pipeline import pw


class StabilityHook(ABC):
    """Base class. Subclasses set ``slots`` to a subset of
    ``{'PRE_LOOP', 'POST_TIRE'}``. A ``'POST_TIRE'`` hook receives both
    :meth:`apply_pre_tire` and :meth:`apply_post_tire` every step."""

    slots: tuple[str, ...] = ()

    def apply_pre_loop(self, ctx: Any) -> None:
        """Called once per step before the per-wheel work. Reads/writes
        ``ctx.moving``, ``ctx.throttle``, ``ctx.brake``, ``ctx.vel``,
        ``ctx.ang``."""

    def apply_pre_tire(self, ctx: Any) -> None:
        """Called once per step BEFORE the tire model (v1.7.0), on every hook
        whose ``slots`` contain ``'POST_TIRE'`` (the pipeline calls it on each
        hook of the POST_TIRE list; a duck-typed hook without the method is
        skipped). Valid fields: ``ctx.N``, ``ctx.v_long``, ``ctx.v_lat``,
        ``ctx.omega`` (the pre-step ω), ``ctx.air_mask``, ``ctx.wheel_meta``,
        ``ctx.dt``; ``ctx.F_long`` / ``ctx.F_lat`` are ``None`` here (the tyre
        has not run).

        This is the ONLY phase that may set ``ctx.T_resist``: a per-wheel axle
        resisting torque, ``(n_envs, n_wheels)``, same sign as ``ctx.omega``,
        ``None`` until a hook sets it -- add to it, do not overwrite another
        hook's value. The pipeline caps it with
        ``dynamics.cap_resist_torque`` and subtracts it from ω before the tyre,
        so the tyre (and the F_long overshoot clamp) start from the ω it
        leaves. Default: no-op."""

    def apply_post_tire(self, ctx: Any) -> None:
        """Called once per step after the tire model. All per-wheel tensors
        on ctx are ``(n_envs, n_wheels)`` — mutate ``ctx.F_long``,
        ``ctx.F_lat``, set ``ctx.omega_override`` /
        ``ctx.omega_pull_factor`` / ``ctx.omega_pull_target`` to influence
        the omega update.

        ``ctx.omega`` here is the ω AFTER any pre-tire resisting torque
        (``ctx.T_resist``, e.g. ``RollingResistance``) was applied -- the ω the
        tyre was evaluated at -- and equals the pre-step ω only when no hook set
        ``T_resist``. ``PipelineResult.kappa`` (``last_kappa``) is taken at that
        same ω.

        Writing ``ctx.T_resist`` here -- rebinding it or mutating it in place
        -- raises ``RuntimeError`` from the pipeline (the in-place check
        cannot see an inference-mode tensor; the pipeline uses its own
        snapshot either way); set it in :meth:`apply_pre_tire`.

        NB the ``F_long`` overshoot clamp runs AFTER every hook, so a hook's
        ``F_long`` is a request the clamp may still cut (to zero on a freely
        rolling wheel). A resistance that must survive the clamp belongs in
        ``ctx.T_resist`` (set in :meth:`apply_pre_tire`). Default: no-op."""


# ---------------------------------------------------------------------------
# RollingResistance
# ---------------------------------------------------------------------------


class RollingResistance(StabilityHook):
    """Rolling resistance as an AXLE RESISTING TORQUE (v1.7.0).

    ``T_rr = cr * N * r * tanh(omega * r / tanh_scale)`` per wheel, accumulated
    into ``ctx.T_resist`` in :meth:`apply_pre_tire` (the hook keeps
    ``slots = ("POST_TIRE",)``; every POST_TIRE hook receives both the pre-tire
    and the post-tire call, and this one does all its work in the pre-tire
    one). ``cr`` is ``wheel_meta.rolling_resistance_cr``,
    ``N`` the wheel's own normal load this step (so an airborne wheel, N = 0,
    gets none), ``r`` its radius. Physically this is the moment ``N * e`` of a
    contact pressure centre shifted forward by ``e = cr * r``. The pipeline
    caps it at ``|omega| * i_wheel / dt`` (``dynamics.cap_resist_torque``) and
    subtracts it from ω BEFORE the tyre, ``omega' = omega - T_rr * dt /
    i_wheel``; the tyre, the other hooks, the F_long overshoot clamp and the ω
    update all start from ``omega'``, so on a freely rolling wheel the tyre
    carries the matching ``-T_rr / r`` (``-cr * N`` above ``tanh_scale``) to
    the chassis on the same step, provided its request reaches the overshoot
    clamp's limit (it did on both reference vehicles; see
    ``_pipeline``'s module docstring for the test that pins it). The brake is
    capped together with it (``cap_resist_torque(T_brake + T_rr)``, the part
    already applied subtracted), so brake + rolling resistance together cannot
    reverse ω in one step (the drive torque, the tyre friction torque and the
    other hooks' ω pulls are not bounded by that cap).
    A free coast decelerates at ``sum(cr * N) / M_eff`` with
    ``M_eff = M + sum(i_wheel / r^2)`` ONLY in this regime: the rolling speed
    ``|omega| * r`` is well above ``tanh_scale`` (0.5 m/s by default; below it
    the ``tanh`` is linear and the torque shrinks with speed), and
    ``LowSpeedRegularizer`` is not active (chassis speed at or above its
    ``v_kin_com``, 0.5 m/s by default; below it the tyre force is scaled down
    and ω is pulled toward rolling). Below either threshold the deceleration is
    smaller than the closed form (measured from 1.0 m/s, 20 s: reference car
    0.0063 m/s, reference tank 0.0003 m/s at the end; never reversed). A locked wheel
    (ω = 0) gets no rolling resistance — sliding friction is already acting
    there, so nothing is counted twice.

    ``cr`` must be finite and ``>= 0``: a negative ``cr`` would turn this into
    a torque that PROPELS the wheel. Enforced by
    ``config.check_rolling_resistance_cr``, called from ``config.resolve`` and
    ``core.build_wheel_meta`` (every driver build, ``mark_config_dirty()``
    rebuild and ``DifferentiablePlant``); the OSC server additionally rejects a
    negative ``rollingResistance`` override. NOT guarded: writing the
    ``wheel_meta.rolling_resistance_cr`` tensor directly.

    Explicit-step behaviour. Per step this hook removes ``dw = min(T_rr,
    |omega| * i_wheel / dt) * dt / i_wheel`` from ω before the tyre; in the
    linear region of the ``tanh`` that is ``k * |omega|`` with
    ``k = cr * N * r^2 * dt / (tanh_scale * i_wheel)``. Enforced by the cap
    (for a ``T_resist`` of ω's sign, up to float rounding -- a capped step can
    land a few ulp past zero): ``omega'`` keeps ω's sign or is zero, at any ``k``; at ``k >= 1`` the cap
    engages and ``omega'`` is 0 for that step before the tyre spins the wheel
    back up (not measured in a stable regime: the reference tank reaches
    ``k > 1`` only at dt 0.05, where its suspension is already outside the
    bump-stop stability limit). Not enforced, measured: on a free coast ω
    does not alternate step to step and chassis speed does not rise. CPU/WSL2,
    genesis-world 1.4.5, dual_scene, per_vehicle, flat μ 1.0, dt 0.025 x 4
    substeps, coast from 1.0 m/s for 20 s, default ``"control"`` profile:
    reference tank (``tank_ray.urdf``, ``tank_skid_belt``, k = 0.92 at static
    N) 0 alternations per wheel (consecutive ω steps of opposite sign, both
    > 1e-4 rad/s), max step change 0.032 rad/s, no step where speed rose by
    more than 1e-5 m/s, 0.0003 m/s at 20 s; reference car (``car_4w.urdf``,
    ``car_4w_rwd_ackermann``, k = 0.16) 0 alternations, max 0.012 rad/s, no
    rise, 0.0063 m/s at 20 s. The intermediate 1.7.0 design (``T_resist``
    added to the brake inside the overshoot clamp instead of before the tyre)
    measured, same runs: tank 281 alternations up to 1.24 rad/s and 133 rising
    steps, car 798 alternations up to 0.23 rad/s and 315 rising steps -- the
    tyre was evaluated at an ω the resistance then changed, and the clamp's
    target (exact rolling) is not the resistance equilibrium. The drive and
    brake torques still go through the clamp the old way, and the same kind of
    step-to-step alternation under throttle or brake is pre-1.7.0 behaviour
    this hook does not change (one measured exception: the reference tank
    under brake 0.6, slowing from 9.7 to 8.5 m/s over a 40-step window, shows
    0 alternations per wheel with this hook present and 38 without it).

    ``tanh_scale`` is still in m/s; its argument is now the wheel ROLLING speed
    ``omega * r`` (it was the contact-point speed ``v_long`` through 1.6.10).

    **Through 1.6.10 this hook had no effect, and its docstring
    (``F_long -= cr * N * tanh(v_long / scale)``) described a force that did
    not reach the chassis.** Two mechanisms removed it:
    (i) on a freely rolling wheel the ``F_long`` overshoot clamp (which runs
    after every hook) cut the subtracted force to zero, and (ii) while driving,
    the same ``F_long`` also entered the wheel update as ``r * F_long``, so the
    wheel spun up, the slip re-balanced and the tyre returned ``+cr * N``.
    Measured on the reference car (``samples/urdf/car_4w.urdf``,
    ``car_4w_rwd_ackermann(stability="control")``, CPU, genesis-world 1.4.5,
    dt 0.025 x 4 substeps, flat μ 1.0, per_vehicle solver, dual_scene): a
    5 m/s coast decelerated at 0.0073 m/s^2 both with and without the hook
    (equal to 4 decimals; 2 s window after 0.5 s of coasting), against a
    closed-form ``(cr*M*g + F_aero) / M_eff`` of 0.1478 at the window's mean
    speed. v1.7.0 measures 0.1473 against a closed form of 0.1472 (lower
    than 0.1478 only because the car is now slower in the window, so its aero
    term is smaller); the reference tank (``tank_ray.urdf``,
    ``tank_skid_belt``, no aero) went from -0.0001 to 0.4779 against 0.4778.
    The 200-step brake-then-throttle rollout of
    ``tests/test_fan_default_identity.py`` moved 0.04 mm when the hook was
    removed through 1.6.10, and 0.70 m since v1.7.0 (0.6961 m, both
    solvers; 0.69 m under the intermediate 1.7.0 design).
    """

    slots = ("POST_TIRE",)

    def __init__(self, tanh_scale: float = 0.5):
        self.tanh_scale = float(tanh_scale)

    def apply_pre_tire(self, ctx: Any) -> None:
        # Pre-tire phase: ctx.omega is the pre-step omega; the pipeline caps
        # this torque and applies it to omega before the tyre.
        wm = ctx.wheel_meta
        r = pw(wm.radius)                                         # (1|B, n_wheels)
        # cr >= 0 by config.check_rolling_resistance_cr (resolve / build_wheel_meta)
        crN = pw(wm.rolling_resistance_cr) * ctx.N                # (B, n_wheels), >= 0
        t_rr = crN * r * torch.tanh(ctx.omega * r / self.tanh_scale)
        prev = getattr(ctx, "T_resist", None)
        ctx.T_resist = t_rr if prev is None else prev + t_rr


# ---------------------------------------------------------------------------
# LowSpeedRegularizer
# ---------------------------------------------------------------------------


class LowSpeedRegularizer(StabilityHook):
    """Low-speed regularizer.

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

    def _prime_derived(self, wheel_meta: Any, device: Any, dtype: Any) -> None:
        """Build-time hook (``_hotset.prime_derived``): derive ``_v_thr_sq``
        now, so the threshold is fixed AFTER every documented pre-build write
        (``TankTuning.apply_config`` sets ``v_thr = 5.0`` on the constructed
        hook) and before the first step."""
        self._v_thr_squared(torch.zeros((), device=device, dtype=dtype))

    def _v_thr_squared(self, ref: torch.Tensor) -> torch.Tensor:
        """``v_thr ** 2`` as a ``(1, 1)`` tensor on ``ref``'s device/dtype.

        Hoisted out of the step path (it was ``self.v_thr * self.v_thr`` on
        every call). Squared in python DOUBLE and only then cast — see
        ``_hotset.square_f64``. Per-ROW rather than a python scalar so a fused
        group can carry one threshold per vehicle; a write to ``v_thr`` still
        lands on the next step (the cache key carries it)."""
        return derived(
            self, "_v_thr_sq", None, (self.v_thr, ref.device, ref.dtype),
            lambda: row_tensor(square_f64(self.v_thr), ref),
        )

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
        active_v     = v_planar_sq < self._v_thr_squared(v_planar_sq)
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
