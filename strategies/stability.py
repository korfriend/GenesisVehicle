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

v1.8.0: in ``apply_post_tire`` ``ctx.omega`` is the ω after the resisting,
drive and brake torques of this step (``omega_t``), ``ctx.F_lat`` is already
the friction-ellipse value where the overshoot clamp binds, and the pipeline
exposes ``ctx.F_long_bounds``, ``ctx.F_long_deadbeat``, ``ctx.hold_applied``
and ``ctx.deadbeat_scale`` (see :meth:`StabilityHook.apply_post_tire`).
"""

from __future__ import annotations

from abc import ABC
from typing import Any

import math

import torch

from .._hotset import derived, row_tensor, square_f64
from .._pipeline import pw
from ..dynamics import DEADBEAT_G


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

        ``ctx.omega`` here is ``omega_t`` (v1.8.0): the ω AFTER this step's
        resisting torque (``ctx.T_resist``), drive torque and brake projection
        were applied, before tyre friction -- the ω the tyre was evaluated at
        (clipped to ``+-omega_max`` for the tyre). Through 1.7.0 it was the ω
        after the resisting torque only. The pre-step ω is ``ctx.omega`` in
        :meth:`apply_pre_tire`.

        ``ctx.F_long`` is the RAW tyre force. ``ctx.F_lat`` is the tyre's
        lateral force, ALREADY replaced by the friction-ellipse value at the
        transmitted force (``TireModel.at_long_force``) on wheels where the
        overshoot clamp binds on the raw ``F_long`` (v1.8.0; through 1.7.0 it
        was always the raw lateral force).

        Read-only fields the pipeline sets before this phase (v1.8.0; it
        keeps its own copies, so writing them changes nothing it computes):
        ``ctx.F_long_bounds`` = ``(lo, hi)`` the interval the clamp will hold
        ``F_long`` to (contains 0); ``ctx.F_long_deadbeat`` the held-wheel
        ``|F_long|`` bound (+inf on wheels the brake does not hold);
        ``ctx.deadbeat_scale`` the rigid-body scale ``s``; since v2.0.4
        ``ctx.brake_rem``, the brake torque left on each wheel after the brake
        projection (> 0 where the brake holds the wheel; ``None`` in
        :meth:`apply_pre_tire`; a later hook sees another hook's write).
        Writable:
        ``ctx.hold_applied`` (bool, ``None`` until a hook sets it) — a hook that
        REPLACES a wheel's force with a position-anchored hold ORs that wheel
        into it; those wheels then get the clamp without the held-wheel
        deadbeat and no lateral deadbeat. ``None`` means no wheel is exempt.

        Writing ``ctx.T_resist`` here -- rebinding it or mutating it in place
        -- raises ``RuntimeError`` from the pipeline (the in-place check
        cannot see an inference-mode tensor; the pipeline uses its own
        snapshot either way); set it in :meth:`apply_pre_tire`.

        NB the ``F_long`` clamp runs AFTER every hook, so a hook's ``F_long``
        is a request the clamp may still cut (to zero on a freely rolling
        wheel). Likewise a hook's ``F_lat`` is a request the lateral deadbeat
        may cut (on every wheel not in ``ctx.hold_applied``). A resistance that
        must survive the clamp belongs in ``ctx.T_resist`` (set in
        :meth:`apply_pre_tire`). Default: no-op."""


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
    caps it at ``|omega| * i_wheel / dt`` (``dynamics.cap_resist_torque``)
    against the pre-step ω and subtracts it from ω FIRST, ``omega_r = omega -
    T_rr * dt / i_wheel``; since 1.8.0 the drive torque and the brake
    projection are then applied too (in that order) before the tyre, and the
    tyre, the other hooks, the clamp and the ω update start from the result.
    With no drive and no brake, on a freely rolling wheel the tyre carries the
    matching ``-T_rr / r`` (``-cr * N`` above ``tanh_scale``) to the chassis
    on the same step, provided its request reaches the overshoot clamp's limit
    (it did on both reference vehicles;
    ``tests/test_rolling_resistance.py::test_a_rolling_wheel_stays_rolling_and_the_chassis_gets_the_rr_force``
    pins it). Through 1.7.0 the brake was capped together with this torque
    (``cap_resist_torque(T_brake + T_rr)``, the part already applied
    subtracted) inside the clamp; since 1.8.0 it is a separate projection
    (``dynamics.brake_projection``) that cannot cross zero either, so brake +
    rolling resistance together still cannot reverse ω in one step (the drive
    torque, the tyre friction torque and the other hooks' ω pulls are not
    bounded by either).
    A free coast decelerates at ``sum(cr * N) / M_eff`` with
    ``M_eff = M + sum(i_wheel / r^2)`` ONLY in this regime: the rolling speed
    ``|omega| * r`` is well above ``tanh_scale`` (0.5 m/s by default; below it
    the ``tanh`` is linear and the torque shrinks with speed), and
    ``LowSpeedRegularizer`` is not active (chassis speed at or above its
    ``v_kin_com``, 0.5 m/s by default; below it the tyre force is scaled down
    and ω is pulled toward rolling). Below either threshold the deceleration is
    smaller than the closed form (measured from 1.0 m/s, 20 s, on both the
    1.7.0 and the 1.8.0 tree, harness below: reference car 0.0063 m/s,
    reference tank 0.0003 m/s at the end; never reversed). A locked wheel
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
    does not alternate step to step and chassis speed does not rise.
    Measured with a standalone harness on the scene of
    ``tests/test_rolling_resistance.py::_coast_scene``: CPU/WSL2,
    genesis-world 1.4.5, dual_scene, per_vehicle, flat μ 1.0, dt 0.025 x 4
    substeps, coast from 1.0 m/s for 20 s, default ``"control"`` profile;
    an alternation = consecutive ω steps of opposite sign, both > 1e-4 rad/s.
    On the 1.8.0 tree: reference tank (``tank_ray.urdf``, ``tank_skid_belt``,
    k = 0.92 at static N) 0 alternations per wheel, max step change 0.032
    rad/s, no step where speed rose by more than 1e-5 m/s, 0.0003 m/s at
    20 s; reference car (``car_4w.urdf``, ``car_4w_rwd_ackermann``, k = 0.16)
    0 alternations, max 0.012 rad/s, no rise, 0.0063 m/s at 20 s -- the same
    figures as on the 1.7.0 tree (tank bit for bit; car end speed 0.006262 vs
    0.006256 m/s). The intermediate, unreleased 1.7.0 design (``T_resist``
    added to the brake inside the overshoot clamp instead of before the tyre)
    measured, on the 1.7.0 development tree with the same scenario: tank 281
    alternations up to 1.24 rad/s and 133 rising steps, car 798 alternations
    up to 0.23 rad/s and 315 rising steps -- the tyre was evaluated at an ω
    the resistance then changed, and the clamp's target (exact rolling) is
    not the resistance equilibrium.

    Drive and brake. Through 1.7.0 this paragraph went on: "The drive and
    brake torques still go through the clamp the old way, and the same kind
    of step-to-step alternation under throttle or brake is pre-1.7.0
    behaviour this hook does not change." That described 1.7.0 and is false
    since 1.8.0: the drive torque and the brake (a Coulomb projection,
    ``dynamics.brake_projection``) now act on ω before the tyre, right after
    this hook's torque, and do not enter the clamp as torques. Measured with
    the same harness, reference tank launched at 10 m/s under brake 0.6, the
    40 steps from the first one below 9.7 m/s: on 1.7.0, 0 alternations per
    wheel with this hook and 39 (max step change 1.81 rad/s) without it -- a
    HYPOTHESIS, not backed by an ablation: the hook's pre-tyre torque damped
    the brake's clamp alternation (the 1.7.0 docstring quoted 38 without it; this harness's window start
    may differ from the one that measured 38 -- not reconciled); on 1.8.0,
    0 alternations with the hook (max 0.081 rad/s) AND 0 without it (max
    0.048 rad/s): the brake no longer alternates, with or without rolling
    resistance. Throttle / brake / reverse alternation on 1.8.0 is covered
    by ``tests/test_low_speed_deadbeats.py::test_n3_*``.

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


#: ``id(wheel_meta) -> {"name": str, "reported": set of (k_spring, dt)}`` for
#: the ``[genesis_vehicle:sfl-cap]`` report (v2.0.4). Module-level, so the
#: report is printed ONCE PER PROCESS per (wheel_meta identity, k_spring, dt):
#: a ``DifferentiablePlant``'s deep-copied prediction hooks share the live
#: driver's wheel_meta and therefore do not print it again. An entry is removed
#: by a ``weakref.finalize`` on the wheel_meta, before its id can be reused.
_SFL_CAP_REGISTRY: dict[int, dict] = {}


def _sfl_cap_entry(wheel_meta: Any) -> dict:
    key = id(wheel_meta)
    ent = _SFL_CAP_REGISTRY.get(key)
    if ent is None:
        ent = _SFL_CAP_REGISTRY[key] = {"name": None, "reported": set()}
        try:
            import weakref
            weakref.finalize(wheel_meta, _SFL_CAP_REGISTRY.pop, key, None)
        except TypeError:
            pass     # un-referenceable duck-typed meta: entry kept for the process
    return ent


def register_sfl_cap_name(wheel_meta: Any, name: str) -> None:
    """Name the vehicle a ``wheel_meta`` belongs to, for the
    ``[genesis_vehicle:sfl-cap]`` report (v2.0.4). Called by the driver when it
    sets ``wheel_meta.static_mass_share``."""
    _sfl_cap_entry(wheel_meta)["name"] = str(name)


class _OmegaOverride:
    """A ``ctx.omega_override`` carrier: ``active`` is a (n_envs, n_wheels)
    bool mask of wheels whose spin the pipeline forces to 0. Through 2.0.3
    ``StaticFrictionLock`` set it; since 2.0.4 no SDK hook does (the pipeline
    feature stays for user hooks)."""
    __slots__ = ("active",)

    def __init__(self, active: torch.Tensor):
        self.active = active


_BRAKE_THR_REMOVED = (
    "StaticFrictionLock: brake_thr was removed in 2.0.4 — the lock now engages "
    "where the brake holds the wheel; drop the argument")
_REMOVED_MSG = {
    "brake_thr": _BRAKE_THR_REMOVED,
    "REST_SPEED": ("StaticFrictionLock.REST_SPEED was removed in 2.0.4 — the "
                   "lock takes over only through its creep test; there is no "
                   "rest-speed takeover any more"),
}


def _removed_get(name):
    def _get(self):
        raise AttributeError(_REMOVED_MSG[name])
    return _get


def _removed_set(name):
    def _set(self, *_a):
        raise AttributeError(_REMOVED_MSG[name])
    return _set


class _StaticFrictionLockMeta(type(StabilityHook)):
    """Makes CLASS-level reads / writes / deletes of the removed names raise
    too (``StaticFrictionLock.REST_SPEED``, ``StaticFrictionLock.brake_thr =
    x``); the instance-level properties live in the class body."""
    brake_thr = property(_removed_get("brake_thr"), _removed_set("brake_thr"),
                         _removed_set("brake_thr"))
    REST_SPEED = property(_removed_get("REST_SPEED"), _removed_set("REST_SPEED"),
                          _removed_set("REST_SPEED"))


class StaticFrictionLock(StabilityHook, metaclass=_StaticFrictionLockMeta):
    """Position-anchored stick-slip static friction lock (v0.5.7; rest anchor
    since v1.8.0; pedal-free since v2.0.4).

    ENGAGES per env, per wheel where the BRAKE physically holds the wheel this
    step — ``ctx.brake_rem > 0``, the brake torque left after the pipeline's
    brake projection — AND its planar contact speed
    ``sqrt(v_long^2 + v_lat^2)`` is below ``min(v_thr, V_ENGAGE_MAX)``
    (:attr:`V_ENGAGE_MAX` 0.5 m/s), at ANY pedal value. A duck-typed ctx
    without ``brake_rem`` engages no wheel. The lock never forces a wheel's
    spin (no ``ctx.omega_override``), and it TAKES OVER only through the
    creep test: the pipeline's held-wheel deadbeat binds on the raw tyre force
    (``|ctx.F_long| >= ctx.F_long_deadbeat``) and the planar speed did not
    fall this step (``v^2 >= CREEP_RATIO * v_prev^2``, :attr:`CREEP_RATIO`
    0.96) — how a stop is detected, where the deadbeat-limited brake balances
    the load at a small speed.

    Consequences (per wheel; not a guarantee about the vehicle): a wheel whose
    brake torque exceeds its drive torque is held and can be locked; a wheel
    whose DRIVE torque exceeds its brake torque turns (B_rem = 0), whatever the
    pedal — brake does not override drive. ``tank_skid_belt`` (``PerSide``)
    drives EVERY wheel of a side with the whole side torque
    ``30000*|0.3*throttle +- steer|`` N*m but brakes each of its n wheels with
    ``30000*brake/n`` N*m, so a full brake does not hold the tank at standstill
    against ``|steer| > 1/n`` or ``throttle > 1/(0.3*n)`` — on the 10-wheel
    reference tank (``tests/data/tank_ref.urdf``, 3000 N*m per wheel)
    ``|steer| > 0.1`` or ``throttle > 0.33``.

    **Through 2.0.3** the lock engaged only on the PEDAL, ``brake > brake_thr``
    (0.3), forced every engaged wheel's spin to 0 below ``v_thr``, and also
    took over below a rest speed of 0.05 m/s. Two defects followed: at a pedal
    at or below 0.3 it never engaged, so a braked vehicle crept down a slope at
    the held-wheel deadbeat speed ``g*tan(theta)*dt/s`` (reference car 5°:
    0.0250 m/s); above 0.3 the forced spin slid a vehicle whose brake could
    NOT hold it on locked wheels (reference car 10°, brake 0.31: 0.040 m/s
    with ω 0) and held wheels the drive should have turned. ``brake_thr`` and
    ``REST_SPEED`` were removed in 2.0.4 (reading or writing either raises
    ``AttributeError``).

    Until then the brake (the pipeline's Coulomb projection and held-wheel
    deadbeat) does the stopping and the anchor FOLLOWS the contact. After the
    takeover a 2D spring-damper holds the contact at the anchor:

        ``F_long  = -K_spring·d_long  - D·v_long``
        ``F_lat   = -K_spring·d_lat   - D·v_lat``

    where ``d_{long,lat}`` are displacements from the anchor (integrated from
    ``v_{long,lat}·dt`` per step). The anchor is placed so the lock holds the
    point where the contact came to REST:

    * the takeover step anchors at the END of the step (``d = 0`` after
      integration), so only the damper acts on it;
    * the first step after engagement on which ``s_eng·v_long <= 0`` (``s_eng``
      the travel sign at engagement: the contact stopped or reversed)
      re-anchors the longitudinal spring at the contact.

    The damper is ``D = max(k_damp, 2·sqrt(k_spring·N/g))`` — at least
    critical for the wheel's mass share ``N/g`` — and the damper force is
    bounded by ``s·(N/g)·|v|/dt`` (``s`` = ``ctx.deadbeat_scale``, the
    pipeline's rigid-body scale), the force that stops the contact in one step,
    so the damper cannot reverse a contact in any planar mode. The force is
    then projected onto the per-wheel friction ellipse

        ``(F_long / mu_long·N)^2 + (F_lat / mu_lat·N)^2  <=  1``

    and the longitudinal component is limited to ``ctx.F_long_bounds`` — what
    the wheel can transmit this step, which includes the brake capacity left on
    a held wheel. Where the ellipse clamps (``norm > 1``, slipping) or the
    bounds limit the force, the anchor advances so the spring asks for exactly
    the limit next step (kinetic-friction mode). Inside both, the wheel is
    stuck within ``F / K_spring`` of the anchor. Wheels whose force the lock
    replaced are ORed into ``ctx.hold_applied``; the pipeline then exempts them
    from its held-wheel and lateral deadbeats.

    **The hold is limited by the brake, not only by friction.** Through 1.7.0
    this docstring said "the vehicle truly stops when external force
    ``<= mu·N``". That was wrong: the hold force is bounded by what the wheel
    can transmit, so a vehicle stops and holds only while the external force
    is also within the brake's capacity (measured: the ``tank_skid_belt``
    preset, 3000 N·m per wheel, does not hold 10°, a 99 kN slope load against
    75 kN of brake; CPU/WSL2, genesis-world 1.4.5, dt 0.025 x 4 substeps).

    Why stick-slip rather than velocity damping (v0.5.7 vs v0.5.6 tanh):
      A pure velocity damper (``F = -K·v`` or ``F = -mu·N·tanh(v/scale)``)
      is fundamentally a *kinetic*-friction model: any non-zero ``v``
      produces an opposing force, but ``v=0`` produces ``F=0``. On a side
      slope where gravity continuously biases ``v_lat``, the damper reaches
      an equilibrium where ``F_damper = F_gravity``, which requires *non-zero*
      ``v_lat`` (~5 mm/s on the truck preset @ 20°). A position-anchored spring
      + ellipse projection is *static* friction: at ``v = 0`` the force is
      whatever ``-K_spring·d`` evaluates to, up to the friction and brake
      limits above.

    Tuning:
      - ``k_spring`` (per wheel): stiffness. Spring's max-allowable
        displacement before friction breaks is ``mu·N / K``; with the truck
        preset (``mu·N ≈ 8000 N`` per wheel) and ``K = 500_000 N/m`` this is
        16 mm. Setting K too high → discretization instability; too low →
        visible creep before stick. Default ``500_000`` is tuned around the
        explicit-Euler bound at ``dt = 0.02`` for a 5-ton chassis. The spring
        is integrated explicitly (displacement first, then the spring and
        damper force). A linear single-step analysis (derived, not measured)
        of one MODE of the rigid body, with ``c = k * dt^2 / m_mode`` its
        spring number and ``b = D * dt / m_mode`` its damper number, gives the
        per-step map ``[[1, 1], [-c, 1 - c - b]]``: stable iff ``c + 2b < 4``
        and ``0 < b < 2``; at ``b = 1`` its multiplier is ``1 - c`` (stable
        iff ``c < 2``). The deadbeat bound on the damper force puts each
        wheel's damper at ``b_i = s`` on its own mass share once it binds; in
        the planar YAW mode, for loads in moment balance about the centre of
        mass and ``s < 1``, that is modal ``b = s * sum m_i rho_i^2 / I_cg =
        1``, so the yaw mode is stable iff its spring number ``c_yaw =
        lambda_yaw(K) * dt^2`` is below 2 — and ``c_yaw`` is set by the
        vehicle's mass and yaw inertia, roughly ``c / s`` for a per-wheel
        ``c``, NOT by the wheel loads. **Through the first 2.0.4 drafts this
        paragraph said "``c < 4 - 2s`` is sufficient" and that the per-wheel
        cap made "``c < 2`` hold down to half the static load". Both were
        wrong**: they treated each wheel as an independent mass share, which
        misses the yaw branch ``c / s < 2``; a vehicle with a small yaw
        inertia relative to ``sum m_i rho_i^2`` (``s`` well below 1) cycled in
        yaw under a held brake with every wheel's ``c`` below 1
        (``tests/data/low_yaw_inertia_car.urdf``, ``s`` 0.288, per-wheel ``c``
        0.93-0.95: about 0.23 rad/s at brake 0.3 and 0.27 rad/s at brake 1.0,
        impl-review measurement, CPU/WSL2, genesis-world 1.4.5, dt 0.025 x 4,
        plane mu 1.0; the brake-1.0 cycle also on 2.0.3).

        **Spring cap (v2.0.4).** The lock does not use ``k_spring`` as given:
        every place it reads the spring rate (spring force, anchor
        re-placement under slip and under the bound, the damper floor) uses

            ``k_eff_i = min(k_spring, f * m_static_i / dt^2)``,
            ``f = min(SPRING_C_MAX, SPRING_C_RB / beta)``

        with :attr:`SPRING_C_MAX` = 1.0 (per wheel: ``c_i <= 1`` on the
        STATIC load) and :attr:`SPRING_C_RB` = 1.5, a constant chosen by the
        user (not derived; see its comment). ``beta`` is
        ``wheel_meta.static_rigid_factor.beta``
        (``dynamics.rigid_share_factor``: the largest eigenvalue of ``S v =
        beta P v``, ``S = sum_i m_static_i J_i^T J_i`` over the planar (x, y,
        yaw) contact Jacobians, ``P = diag(M, M, I_cg)``; ``beta >= 1``). For
        ``beta <= 1.5`` ``f`` is ``SPRING_C_MAX`` itself and ``k_eff`` is
        bit-for-bit the per-wheel cap alone; every reference vehicle is
        there (reference car beta 1.169, tank 1.069, truck demo 1.446).
        ``m_static_i`` is ``wheel_meta.static_mass_share``
        (``dynamics.static_mass_shares``, elastic-support statics) and ``beta``
        comes from the same link walk; both are computed once by the driver
        on its first step from the build-time link masses (a batched kind uses
        its proto's entity, ``entities[0]``; runtime mass changes and per-env
        mass differences are not reflected; ``wheel_meta.yaw_inertia`` is
        never read for ``beta``), and ``dt`` is ``ctx.dt``. A write to
        ``k_spring`` is capped too. ``beta`` is computed only together with
        the share: a ``static_mass_share`` written by the user after the first
        step keeps the OLD ``beta`` — set ``wheel_meta.static_rigid_factor``
        to ``None`` too (per-wheel cap only) or to a
        ``dynamics.rigid_share_factor`` of the new shares.

        What the cap enforces, where both ``static_mass_share`` and
        ``static_rigid_factor`` exist (by construction of ``k_eff`` in
        ``_compute_k_eff``; no eigenvalue is computed there): per
        wheel ``c_i <= SPRING_C_MAX``, and ``lambda(K_eff, P) * dt^2 <=
        SPRING_C_RB`` for the PLANAR model, for any subset of grounded wheels
        and any steer angle. That is a bound on ``lambda(K)`` ONLY. Linear
        stability additionally needs the modal damper number ``b <= 1``,
        which the deadbeat scale provides only for planar modes under
        moment-balanced loads. NOT covered: (i) roll / pitch-coupled modes
        (the contact force acts below the centre of mass; on a light 10-wheel
        tank the deadbeat-bounded damper reaches modal ``b`` 1.257 in a
        roll-coupled mode, linear spectral radius 1.31 before and after this
        cap — a linear computation, not a measurement); (ii) loads not in
        moment balance; (iii) a user-preset ``wheel_meta.yaw_inertia =
        YawInertia(inf, ...)`` (``s = 1``: the damper's yaw ``b`` is then
        ``sum m_i rho_i^2 / I_cg``, 3.47 on the low-yaw-inertia car, and the
        cap does not restore stability); (iv) a batched kind of mixed-mass
        vehicles (``entities[0]``); (v) a vehicle whose ``beta`` could not be
        computed (one "rigid-body factor unavailable" WARNING; per-wheel cap
        only) or whose ``static_mass_share`` was preset by the caller
        (``beta`` None; per-wheel cap only); (vi) a duck-typed meta without
        ``static_mass_share`` or a driver that could not read its mass
        (logged): uncapped; (vii) one wheel in contact (the deadbeat scale's
        own up-to-2x caveat). Where no wheel is capped the lock computes
        exactly as through 2.0.3. Measured inactive at ``dt = 0.025`` on the
        reference car (max ``c`` 0.947), the reference tank (0.053) and the
        tank with ``TankTuning`` (0.107); the reference car is capped above
        ``dt`` ≈ 0.0257 s. A capped wheel is softer, so its hold deflection
        ``F / k_eff <= mu*N/k_eff = mu*g*dt^2/f * (N / N_static)`` — vehicle
        dependent through ``f``: 6.1 mm at ``dt = 0.025`` and ``mu = 1`` per
        unit of static load where ``f = 1`` (derived), 14.2 mm at 0.025 and
        36.3 mm at 0.04 on the low-yaw-inertia car (``f`` 0.4324, a formula
        value: that URDF diverges at ``dt >= 0.03`` with no hooks and no
        brake on 2.0.3 too — independent of the lock, cause not established);
        more on a wheel carrying more than its static load. The first time a
        cap binds, one ``[genesis_vehicle:sfl-cap]`` WARNING per process per
        (wheel meta, ``k_spring``, ``dt``) names the vehicle, the capped
        wheels and the binding term.
      - ``k_damp`` (per wheel): since 1.8.0 a FLOOR on the damping
        coefficient: the lock uses ``max(k_damp, 2·sqrt(k_spring·N/g))``, and
        the damper force is then bounded by the one-step deadbeat. So a
        ``k_damp`` below critical for the wheel's mass share no longer takes
        effect (through 1.7.0 it was used as given). The default ``20_000``
        is BELOW critical on the reference vehicles: ζ ≈ 0.78 on the reference
        car's wheel share, ζ ≈ 0.185 on the reference tank's 5.8 t share
        (``tank_skid_belt``); through 1.7.0 this docstring called it
        "~half-critical for a 5-ton chassis", which matched neither.
      - ``v_thr``: planar contact speed below which a held wheel ENGAGES,
        capped at :attr:`V_ENGAGE_MAX` (0.5 m/s): since 2.0.4 values above
        0.5 m/s no longer widen the engage window (``TankTuning``'s 5.0 acts
        as 0.5). The force replacement waits for the creep test. Through
        1.7.0 the lock replaced the tyre force from ``v_thr``
        on and anchored at the engagement point. Combined with the 1.8.0
        full-strength brake, a stop the brake finished after engagement was
        pulled back toward that point (measured in the 1.8.0 design probe,
        reference car, brake 1.0 from 10 m/s, CPU/WSL2, genesis-world 1.4.5,
        dt 0.025 x 4 substeps: reverse speed -0.107 m/s after the stop; with
        the rest anchor -3e-5 m/s). (On 1.7.0 itself, whose brake faded near
        standstill, the same stop measured +2e-7 m/s.)

    State (:attr:`STATE_ATTRS`): ``_d_long``, ``_d_lat``, ``_was_active`` and,
    since 1.8.0, ``_s_eng``, ``_rested``, ``_rest``, ``_v_prev_sq`` — all
    ``(n_envs, n_wheels)``. Like the first three, the new state survives every
    reset (a documented known gap). Two consequences of the new state: a
    stale ``_v_prev_sq`` can trigger one early creep takeover on a wheel that
    engages on that exact step; and a stale ``_rest = True`` (with
    ``_was_active`` still True, i.e. the reset vehicle's wheels are still held
    by the brake and below the engage speed) skips the creep test entirely:
    no ``just_engaged`` edge clears it, so the lock re-applies the spring hold
    at once, on the stale ``_d_long`` / ``_d_lat`` anchor (1.7.0 had the
    equivalent through its stale anchor alone). Since 2.0.4 this is reachable
    at any pedal (through 2.0.3 only above ``brake_thr``). The anchor is
    velocity-integrated, so a teleport itself does not displace it; the
    deflection it can hold is bounded by ``mu*N/k_eff``. Clear every
    :attr:`STATE_ATTRS` entry yourself after a reset that teleports a braked
    vehicle if that offset matters. (The first 1.8.0 implementation named
    only the ``_v_prev_sq`` consequence here.)

    BREAKING from v0.5.6:
      The old ``hold_k``, ``hold_k_lat``, ``slip_scale`` constructor kwargs
      are GONE — they were proportional / tanh gains in the prior velocity-
      damper model and have no meaning in the stick-slip model. Replace with
      ``k_spring`` / ``k_damp`` (or just drop them and accept the defaults).
      Pre-1.0 SDK; no deprecation shim.
    """

    slots = ("POST_TIRE",)

    # (v2.0.4) Removed: reading or writing either name on an instance raises
    # AttributeError (a plain removal would let a write create an inert
    # attribute). Class-level access is covered by the metaclass.
    brake_thr = property(_removed_get("brake_thr"), _removed_set("brake_thr"),
                         _removed_set("brake_thr"))
    REST_SPEED = property(_removed_get("REST_SPEED"), _removed_set("REST_SPEED"),
                          _removed_set("REST_SPEED"))

    #: Creep equilibrium test (v1.8.0, A6): the held-wheel deadbeat binds and
    #: ``v_planar^2 >= CREEP_RATIO * v_planar_prev^2`` (the speed did not fall
    #: this step).
    CREEP_RATIO = 0.96
    #: The spring cap's per-wheel bound on the explicit-spring number
    #: ``c = k * dt^2 / m_static`` (v2.0.4): ``k_eff <= SPRING_C_MAX *
    #: m_static / dt^2`` per wheel (see the class docstring for the full
    #: two-term cap).
    SPRING_C_MAX = 1.0
    #: The cap's planar rigid-body constant (v2.0.4 Step D): ``k_eff <=
    #: SPRING_C_RB * m_static / (beta * dt^2)``, ``beta`` =
    #: ``wheel_meta.static_rigid_factor.beta``. Chosen by the user, not
    #: derived: at the damper's one-step bound (modal damper number b = 1) it
    #: keeps the planar yaw-mode multiplier ``1 - lambda * dt^2 >= -0.5`` per
    #: step (the oscillating side only; a small ``lambda * dt^2`` leaves the
    #: multiplier near +1). Any value in about 1.17-2 would leave the reference vehicles
    #: unchanged. Unifying both terms on one constant (1) was planned for 2.1.0
    #: through 2.0.4; WITHDRAWN by the user on 2026-11-24 (CHANGELOG 2.0.5).
    SPRING_C_RB = 1.5
    #: Upper bound (m/s) of the engage window (v2.0.4): a wheel with
    #: ``ctx.brake_rem > 0`` engages below ``min(v_thr, V_ENGAGE_MAX)``.
    V_ENGAGE_MAX = 0.5
    #: The hook's per-wheel INTEGRATOR state, every attribute
    #: ``control.plant._sync_hooks`` row-slices and every tensor this hook
    #: holds. Code that copies the lock's state between hooks by name must copy
    #: all of these (the last four are new in 1.8.0).
    STATE_ATTRS = ("_d_long", "_d_lat", "_was_active",
                   "_s_eng", "_rested", "_rest", "_v_prev_sq")

    def __init__(
        self,
        *args: Any,
        v_thr: float = 0.5,
        k_spring: float = 500_000.0,
        k_damp: float = 20_000.0,
        **kwargs: Any,
    ):
        # (v2.0.4) Keyword-only. The explicit checks replace Python's generic
        # messages: an old positional call (brake_thr, v_thr, k_spring,
        # k_damp) would otherwise not say why, and brake_thr was the first.
        if args:
            raise TypeError(
                "StaticFrictionLock takes keyword arguments only since 2.0.4 "
                f"(got {len(args)} positional); its old first parameter "
                "brake_thr was removed — pass v_thr= / k_spring= / k_damp= "
                "by name")
        if "brake_thr" in kwargs:
            raise TypeError(_BRAKE_THR_REMOVED)
        if kwargs:
            raise TypeError(
                "StaticFrictionLock got unexpected keyword argument(s) "
                f"{sorted(kwargs)}; it accepts v_thr, k_spring, k_damp")
        self.v_thr = float(v_thr)
        self.k_spring = float(k_spring)
        self.k_damp = float(k_damp)
        # Per-wheel state, allocated lazily in apply_post_tire when shape is
        # known. (n_envs, n_wheels) — displacement of contact from anchor.
        self._d_long: torch.Tensor | None = None
        self._d_lat: torch.Tensor | None = None
        self._was_active: torch.Tensor | None = None
        # v1.8.0 rest-anchor state (see the class docstring), also
        # (n_envs, n_wheels), created here as None so a generic attribute sweep
        # sees the same names before and after the first step:
        # travel sign at engagement (float), "came to rest since engagement"
        # (bool), "the lock has taken over" (bool), last step's planar contact
        # speed squared (float, +inf before the first step).
        self._s_eng: torch.Tensor | None = None
        self._rested: torch.Tensor | None = None
        self._rest: torch.Tensor | None = None
        self._v_prev_sq: torch.Tensor | None = None

    def _prime_derived(self, wheel_meta: Any, device: Any, dtype: Any) -> None:
        """Build-time hook (``_hotset.prime_derived``): derive ``_v_thr_sq``
        now, so the threshold is fixed AFTER every documented pre-build write
        (``TankTuning.apply_config`` sets ``v_thr = 5.0`` on the constructed
        hook) and before the first step."""
        self._v_thr_squared(torch.zeros((), device=device, dtype=dtype))
        # (v2.0.4) No dt is known at build (k_eff takes dt only from ctx.dt),
        # so this derives the uncapped placeholder; the first step's key
        # (ctx.dt, the static share) differs and re-derives.
        self._k_eff(wheel_meta, None, device, dtype)

    def _k_eff(self, wheel_meta: Any, dt: Any, device: Any, dtype: Any) -> Any:
        """The per-wheel lock stiffness ``k_eff`` (v2.0.4, the spring cap).

        ``k_eff_i = min(k_spring, f * m_static_i / dt^2)``, ``f =
        min(SPRING_C_MAX, SPRING_C_RB / beta)`` (Step D of 2.0.4), with
        ``m_static`` = ``wheel_meta.static_mass_share`` (set by the driver on
        its first step from the build-time link masses; never the
        instantaneous load), ``beta`` = ``wheel_meta.static_rigid_factor.beta``
        (``f = SPRING_C_MAX`` when the meta has no ``static_rigid_factor`` or
        it is None) and ``dt`` = ``ctx.dt`` only.

        Returns the python float ``k_spring`` itself — so every use is the
        2.0.3 expression, bit for bit — when no wheel is capped, when the meta
        has no static share (``getattr`` fallback: a duck-typed meta, a driver
        whose mass could not be read) or when ``dt`` is unknown / not
        positive. Otherwise a ``(1, n_wheels)`` tensor on ``device`` /
        ``dtype``. Cached with ``_hotset.derived``: ``ref`` the wheel meta
        (identity), key ``(k_spring, dt, static share tensor, its version
        counter, static_rigid_factor (by value), device, dtype)``, so a later ``k_spring`` write lands on the
        next step. The key holds the share TENSOR itself (never its ``id()``):
        a newly assigned share is compared by value (``torch.equal``), and an
        in-place edit of the same tensor changes its version counter, so it
        re-derives too — except for an inference-mode tensor, which has no
        version counter (an in-place edit of one is not seen). Raises
        ``AssertionError`` when a static share would give a non-positive or
        non-finite cap (``dynamics.static_mass_shares`` guards against it) or
        when ``beta`` is not finite or below 1 (``1 - 1e-6`` allows the
        rounding of shares that sum to the mass)."""
        share = getattr(wheel_meta, "static_mass_share", None)
        # (Step D) beta of the planar rigid-body term, a NamedTuple of python
        # floats compared by value in the key; None -> per-wheel cap only.
        rigid = getattr(wheel_meta, "static_rigid_factor", None)
        # The tensor's version counter (a python int, no device sync) makes an
        # in-place edit of the share a different key; an inference-mode tensor
        # has none (None: such an in-place edit is not seen).
        ver = (None if share is None or not torch.is_tensor(share)
               or share.is_inference() else share._version)
        k = self.k_spring
        return derived(
            self, "_k_eff", wheel_meta,
            (k, dt, share, ver, rigid, device, dtype),
            lambda: self._compute_k_eff(wheel_meta, share, k, dt, device, dtype,
                                        rigid))

    def _compute_k_eff(self, wheel_meta, share, k_raw, dt, device, dtype,
                       rigid=None):
        # Uncapped: the attribute value itself, unconverted (2.0.3 used it so).
        if share is None or dt is None or not float(dt) > 0.0:
            return k_raw
        k = float(k_raw)
        dt = float(dt)
        c_max = float(self.SPRING_C_MAX)
        # (Step D) f = min(C_MAX, C_RB / beta). For beta <= C_RB / C_MAX (1.5)
        # min() returns c_max itself, so the cap below is the pre-Step-D
        # expression with the same operands in the same order (bit-exact).
        beta = None
        f = c_max
        if rigid is not None:
            beta = float(rigid.beta)
            if not (math.isfinite(beta) and beta >= 1.0 - 1e-6):
                raise AssertionError(
                    f"StaticFrictionLock: static_rigid_factor.beta {beta!r} is "
                    f"not a finite number >= 1")
            f = min(c_max, float(self.SPRING_C_RB) / beta)
        cap = f * share.detach().to(dtype=torch.float64).reshape(-1) / (dt * dt)
        if not bool(torch.isfinite(cap).all()) or not bool((cap > 0.0).all()):
            raise AssertionError(
                f"StaticFrictionLock: the spring cap from static mass shares "
                f"{share.reshape(-1).tolist()} kg at dt {dt} is not positive and "
                f"finite on every wheel")
        capped = cap < k
        if not bool(capped.any()):
            return k_raw
        keff64 = torch.clamp(cap, max=k)
        keff = keff64.to(device=device, dtype=dtype).reshape(1, -1)
        if not bool((keff > 0.0).all()):
            raise AssertionError(
                f"StaticFrictionLock: k_eff {keff.reshape(-1).tolist()} is not "
                f"positive on every wheel")
        ent = _sfl_cap_entry(wheel_meta)
        rkey = (k, dt)
        if rkey not in ent["reported"]:
            ent["reported"].add(rkey)
            import logging
            idx = [int(i) for i in torch.nonzero(capped).reshape(-1)]
            ms = share.detach().to(dtype=torch.float64).reshape(-1)
            # (Step D) Name the binding term: the per-wheel one when f is
            # C_MAX, else the planar rigid-body one ("yaw mode" only when the
            # leading eigenvector is yaw-dominated: a labelling rule).
            if f == c_max:
                bound = "per-wheel static load (c_max %.1f)" % c_max
            else:
                bound = "planar rigid-body term (C_RB %.1f, beta %.3f%s)" % (
                    float(self.SPRING_C_RB), beta,
                    ", yaw mode" if float(rigid.yaw_fraction) > 0.5 else "")
            logging.getLogger("genesis_vehicle").warning(
                "[genesis_vehicle:sfl-cap] vehicle '%s': StaticFrictionLock "
                "k_spring %.0f N/m capped on wheel(s) %s to k_eff %s N/m "
                "by the %s (dt %g s, static share(s) %s kg); hold offset up "
                "to mu*g*dt^2/f = %.1f mm at mu 1.0 (f %.6g), times "
                "N/N_static on a wheel carrying more than its static load",
                ent["name"] or "<unnamed>", k, idx,
                [round(float(keff64[i]), 1) for i in idx], bound, dt,
                [round(float(ms[i]), 1) for i in idx],
                1000.0 * 9.81 * dt * dt / f, f)
        return keff

    def _v_thr_squared(self, ref: torch.Tensor) -> torch.Tensor:
        """The squared engage speed as a ``(1, 1)`` tensor on ``ref``'s
        device/dtype: ``min(v_thr, V_ENGAGE_MAX) ** 2`` since 2.0.4 (``v_thr
        ** 2`` through 2.0.3).

        Hoisted out of the step path (it was ``self.v_thr * self.v_thr`` on
        every call). Squared in python DOUBLE and only then cast — see
        ``_hotset.square_f64``. Per-ROW rather than a python scalar so a fused
        group can carry one threshold per vehicle; a write to ``v_thr`` still
        lands on the next step (the cache key carries it)."""
        return derived(
            self, "_v_thr_sq", None,
            (self.v_thr, self.V_ENGAGE_MAX, ref.device, ref.dtype),
            lambda: row_tensor(square_f64(self._engage_speed()), ref),
        )

    def _engage_speed(self) -> Any:
        """``min(v_thr, V_ENGAGE_MAX)`` (v2.0.4), a python float or, for a
        per-row tensor ``v_thr``, a tensor."""
        if torch.is_tensor(self.v_thr):
            return torch.clamp(self.v_thr, max=float(self.V_ENGAGE_MAX))
        return min(float(self.v_thr), float(self.V_ENGAGE_MAX))

    def _ensure_state(self, ref: torch.Tensor) -> None:
        # Allocated in groups, each on its own shape test, so state copied in by
        # NAME from another hook (external code copies `_d_long`, `_d_lat`,
        # `_was_active`) is kept where its shape fits and the rest starts fresh.
        if self._d_long is None or self._d_long.shape != ref.shape:
            self._d_long = torch.zeros_like(ref)
            self._d_lat = torch.zeros_like(ref)
            self._was_active = torch.zeros(ref.shape, device=ref.device, dtype=torch.bool)
        if self._s_eng is None or self._s_eng.shape != ref.shape:
            self._s_eng = torch.zeros_like(ref)
            self._rested = torch.zeros(ref.shape, device=ref.device, dtype=torch.bool)
        if self._rest is None or self._rest.shape != ref.shape:
            self._rest = torch.zeros(ref.shape, device=ref.device, dtype=torch.bool)
        if self._v_prev_sq is None or self._v_prev_sq.shape != ref.shape:
            self._v_prev_sq = torch.full_like(ref, float("inf"))

    def apply_post_tire(self, ctx: Any) -> None:
        # Activation (v2.0.4, pedal-free): the brake physically holds the
        # wheel this step (ctx.brake_rem > 0) AND the planar contact speed is
        # below min(v_thr, V_ENGAGE_MAX). We deliberately do NOT short-circuit
        # with `if not active.any()` — that forces a CPU<->GPU sync per step.
        # getattr: a duck-typed ctx without brake_rem -> no wheel held.
        v_planar_sq  = ctx.v_long * ctx.v_long + ctx.v_lat * ctx.v_lat  # (n_envs, n_wheels)
        active_v     = v_planar_sq < self._v_thr_squared(v_planar_sq)
        brake_rem = getattr(ctx, "brake_rem", None)
        if brake_rem is None:
            active = torch.zeros_like(active_v)
        else:
            active = (brake_rem > 0) & active_v                         # (n_envs, n_wheels)

        self._ensure_state(ctx.v_long)
        dt = ctx.dt
        # (v2.0.4) The per-wheel capped stiffness, used at EVERY place the
        # lock reads its spring rate (python float k_spring when uncapped).
        k_eff = self._k_eff(ctx.wheel_meta, dt, ctx.v_long.device, ctx.v_long.dtype)

        # On rising edge of active (just-engaged), reset the anchor → d = 0 and
        # record the travel sign (A) and clear the rest flag (A6).
        just_engaged = active & (~self._was_active)
        self._d_long = torch.where(just_engaged, torch.zeros_like(self._d_long), self._d_long)
        self._d_lat  = torch.where(just_engaged, torch.zeros_like(self._d_lat),  self._d_lat)
        self._s_eng = torch.where(just_engaged, torch.sign(ctx.v_long), self._s_eng)
        self._rested = torch.where(just_engaged, self._s_eng == 0, self._rested)
        self._rest = torch.where(just_engaged, torch.zeros_like(self._rest), self._rest)

        # [A6] Take over only through the creep test (v2.0.4; through 2.0.3
        # also below a 0.05 m/s rest speed): the held-wheel deadbeat binds on
        # the raw tyre force and the planar speed did not fall this step.
        rest_before = self._rest
        # getattr: a duck-typed ctx (a test SimpleNamespace, a pre-1.8.0 caller)
        # may lack the v1.8.0 fields; each then means "not available".
        db = getattr(ctx, "F_long_deadbeat", None)
        bind = (torch.zeros_like(active) if db is None
                else (torch.abs(ctx.F_long) >= db))
        creep = bind & (v_planar_sq >= self.CREEP_RATIO * self._v_prev_sq)
        self._rest = self._rest | (active & creep)
        self._v_prev_sq = v_planar_sq.detach().clone()
        # Before rest the anchor follows the contact (d = 0), and the takeover
        # step itself anchors at the END of the step (d = 0 after integration):
        # only the bounded damper acts on it.
        pre_rest = active & ((~self._rest) | (self._rest & ~rest_before))
        self._d_long = torch.where(pre_rest, torch.zeros_like(self._d_long), self._d_long)
        self._d_lat  = torch.where(pre_rest, torch.zeros_like(self._d_lat),  self._d_lat)

        # Integrate displacement (only while active).
        self._d_long = torch.where(
            active, self._d_long + ctx.v_long * dt, self._d_long)
        self._d_lat = torch.where(
            active, self._d_lat  + ctx.v_lat  * dt, self._d_lat)
        self._d_long = torch.where(pre_rest, torch.zeros_like(self._d_long), self._d_long)
        self._d_lat  = torch.where(pre_rest, torch.zeros_like(self._d_lat),  self._d_lat)

        # [A] The first step after engagement on which the contact came to rest
        # or reversed (s_eng * v_long <= 0) re-anchors the longitudinal spring
        # at the contact: the lock holds the STOP point, not the engagement point.
        came_to_rest = active & (~self._rested) & (self._s_eng * ctx.v_long <= 0)
        self._d_long = torch.where(came_to_rest, torch.zeros_like(self._d_long), self._d_long)
        self._rested = self._rested | came_to_rest

        # [C] Damping at least critical for the wheel's mass share N/g ...
        k_damp = torch.maximum(
            torch.full_like(ctx.N, float(self.k_damp)),
            2.0 * torch.sqrt(k_eff * torch.clamp(ctx.N, min=0.0) / DEADBEAT_G))
        damp_long = k_damp * ctx.v_long
        damp_lat = k_damp * ctx.v_lat
        # ... and never more than what stops the contact in one step on its
        # (rigid-body scaled) mass share, so the damper cannot reverse it.
        s = getattr(ctx, "deadbeat_scale", None)
        if s is None:
            s = 1.0
        m_share = ctx.N * s / DEADBEAT_G
        cap_long = m_share * torch.abs(ctx.v_long) / dt
        cap_lat = m_share * torch.abs(ctx.v_lat) / dt
        damp_long = torch.maximum(torch.minimum(damp_long, cap_long), -cap_long)
        damp_lat = torch.maximum(torch.minimum(damp_lat, cap_lat), -cap_lat)

        # Spring + damper force (will be projected onto friction ellipse below).
        hold_long = -k_eff * self._d_long - damp_long
        hold_lat  = -k_eff * self._d_lat  - damp_lat

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
        # behaviour: anchor drifts along with the contact). Not before rest:
        # there the anchor already follows the contact.
        new_d_long = -hold_long_clamped / k_eff
        new_d_lat  = -hold_lat_clamped  / k_eff
        advance_mask = slipping & active & ~pre_rest
        self._d_long = torch.where(advance_mask, new_d_long, self._d_long)
        self._d_lat  = torch.where(advance_mask, new_d_lat,  self._d_lat)

        # [L] The hold force cannot exceed what the wheel can transmit this
        # step (the pipeline's clamp interval, widened by the brake capacity
        # left on a held wheel); where that limits it, the anchor moves so the
        # spring asks for exactly the limit next step.
        b = getattr(ctx, "F_long_bounds", None)
        if b is not None:
            lim = torch.maximum(torch.minimum(hold_long_clamped, b[1]), b[0])
            limited = active & (lim != hold_long_clamped) & ~pre_rest
            self._d_long = torch.where(limited, -lim / k_eff, self._d_long)
            hold_long_clamped = lim

        # Replace the tyre force only where the lock has taken over (after
        # rest). (When N=0 the friction ellipse has μ·N ≈ 0 so `norm` is huge
        # and `scale ≈ 0` → hold_clamped ≈ 0 automatically.)
        apply = active & self._rest
        prev_hold = getattr(ctx, "hold_applied", None)
        ctx.hold_applied = apply if prev_hold is None else (prev_hold | apply)
        ctx.F_long = torch.where(apply, hold_long_clamped, ctx.F_long)
        ctx.F_lat  = torch.where(apply, hold_lat_clamped,  ctx.F_lat)

        # (v2.0.4) No omega override: through 2.0.3 the lock forced the spin
        # of every engaged wheel to 0; the brake projection holds a wheel the
        # brake can hold by itself, and a wheel it cannot hold must turn.

        # Persist edge-detect state for next step.
        self._was_active = active
