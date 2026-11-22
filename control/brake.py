"""Brake -> deceleration model of the gear-mode follower's stage 2 (v2.0.2).

Used ONLY in stage 2 (the inversion of an acceleration target into commands)
of ``PathFollower(command_mode="gear")``. The MPC prediction model keeps its
throttle-based deceleration bound, and ``DifferentiablePlant``'s unroll
predicts with its constant constructor ``brake``; neither contains this model.

Model, for a vehicle rolling at ``v`` with throttle 0 and brake ``b`` in
[0, 1]::

    a(b) = a_coast(v, pitch, roll) - sign(v) * k_b * b        (|v| > 0)

``a_coast`` is the plant's own forward prediction at throttle 0
(``accel_at``), ``k_b`` the brake gain (m/s^2 per unit brake). Where it was
measured (EXPLORATORY probe, reference car, plane mu 1.0, dt 0.025 x 4, 2 /
6 / 12 m/s): the deceleration is linear in ``b`` up to b = 1 (about 5.2 m/s^2
there, below mu*g, so no wheel locked) and the closed form below gave 5.0152
against 5.0204 measured. Not modelled: the friction-limited regime (wheel
lock, low mu), where the real deceleration is smaller than the model's;
``dynamics.brake_projection`` is not an anti-lock system.

| abbr | meaning |
|---|---|
| k_b | brake gain: modelled deceleration per unit brake input (m/s^2) |
| v_h | coast hold speed: the plant table's grid speed nearest 0, excluding 0, on the side used |
| v_f | coast fade speed (``COAST_FADE_SPEED``) |
| G | slope term of the coast: pitched/rolled minus flat table acceleration at the same speed |
| a_coast | modelled acceleration at throttle 0, brake 0 (m/s^2) |
| T_b,max,i | wheel i's brake torque at brake = 1 (N*m) |
| I_i, R_i | wheel i's spin inertia (kg*m^2) and radius (m) |
"""
from __future__ import annotations

import inspect
import math
from typing import Any, Callable, Optional

from ..strategies.stability import RollingResistance

#: Fade speed (m/s) of the low-speed flat coast in :func:`coast_for_brake`: the
#: default ``tanh_scale`` of ``RollingResistance``, read from its signature (one
#: source). A vehicle measured without ``RollingResistance`` ("raw" /
#: "research" stability profiles) or with another ``tanh_scale`` gets a fade of
#: an assumed shape below about 1 m/s.
COAST_FADE_SPEED = float(
    inspect.signature(RollingResistance.__init__).parameters["tanh_scale"].default)


def brake_for(a_target: float, a_coast: float, v: float, gear: int, k_b: float) -> float:
    """Brake in [0, 1] whose modelled deceleration gives ``a_target``:
    ``clip(s * (a_coast - a_target) / k_b, 0, 1)`` with ``s = sign(v)`` when
    ``|v| > 1e-3``, else ``s = gear`` (the direction the vehicle is about to
    move in). ``a_target`` / ``a_coast`` / ``v`` are signed along chassis +X."""
    s = math.copysign(1.0, v) if abs(v) > 1e-3 else (1.0 if gear >= 0 else -1.0)
    b = s * (float(a_coast) - float(a_target)) / float(k_b)
    return min(1.0, max(0.0, b))


def brake_gain_closed_form(resolved: Any, wheel_meta: Any, mass: float) -> float:
    """``k_b = sum_i(T_b,max,i / R_i) / (m + sum_i I_i / R_i^2)``: the
    deceleration at brake 1 of a vehicle whose wheels roll without slip and
    whose brake torque is all transmitted (below the friction limit).
    ``T_b,max,i`` comes from the vehicle's own drivetrain
    (``distribute_torque`` at throttle 0, brake 1), ``I_i`` / ``R_i`` from
    ``wheel_meta.i_wheel`` / ``wheel_meta.radius``, ``mass`` is the composite
    mass (``PlantMassProperties.mass``)."""
    import torch
    from ..inputs import VehicleInputs
    wm = wheel_meta
    dev, fdt = wm.radius.device, wm.radius.dtype
    n = int(wm.n_wheels)
    one = torch.ones(1, device=dev, dtype=fdt)
    ins = resolved.steering.InputType.from_unified(
        VehicleInputs(throttle=0.0 * one, brake=one, steer=0.0 * one))
    omega = torch.zeros(1, n, device=dev, dtype=fdt)
    _, t_brake = resolved.drivetrain.distribute_torque(ins, omega, wm, dev, fdt)
    tb = t_brake.reshape(-1).double().cpu()
    r = wm.radius.reshape(-1).double().cpu()
    i_w = wm.i_wheel.reshape(-1).double().cpu()
    k_b = float((tb / r).sum() / (float(mass) + (i_w / r ** 2).sum()))
    if not (math.isfinite(k_b) and k_b > 0.0):
        raise ValueError(f"closed-form brake gain is not a positive number: {k_b!r}")
    return k_b


def brake_gain_from_params(params: Any, accel_at: Callable[..., float]) -> Optional[float]:
    """``k_b`` from an MPC params record: its ``brake_gain`` key (written by
    the 2.0.2 identify tool; ``null`` counts as absent), else — for a record
    from before 2.0.2 — the mean over the measured speeds ``v`` of
    ``accel_at(v, 0, 0, 0) - decel_brake_by_speed[v]`` (the brake-1
    acceleration the record carries, against the plant's coast). ``None``
    when neither is available."""
    if params is None:
        return None
    data = getattr(params, "data", params)
    kb = data.get("brake_gain") if isinstance(data, dict) else None
    if kb is not None:
        kb = float(kb)
        if not (math.isfinite(kb) and kb > 0.0):
            raise ValueError(f"mpc_params brake_gain must be a positive number; got {kb!r}")
        return kb
    dec = data.get("decel_brake_by_speed") if isinstance(data, dict) else None
    if not dec:
        return None
    vals = [float(accel_at(float(v), 0.0, 0.0, 0.0)) - float(a) for v, a in dec.items()]
    kb = sum(vals) / len(vals)
    if not (math.isfinite(kb) and kb > 0.0):
        raise ValueError(
            f"brake gain derived from mpc_params decel_brake_by_speed is not positive: {kb!r}")
    return kb


def coast_for_brake(plant: Any, v: float, pitch_deg: float, roll_deg: float,
                    gear: int) -> float:
    """(v2.0.2, plan_r202 Amendment 1 Rev 1) The coast acceleration ``a_coast``
    the gear-mode brake conversion uses, from the plant's forward map
    ``accel_at`` at throttle 0.

    A plant without ``coast_hold_speeds`` (``DifferentiablePlant``, which
    predicts the coast from the live state through the real pipeline; any
    duck-typed plant) is passed through: ``plant.accel_at(v, 0, pitch, roll)``.
    A duck-typed plant gets no low-speed correction.

    For a ``SweepTable`` (``coast_hold_speeds = (v_neg, v_pos)``): the table is
    linear between its grid speed 0 (a vehicle AT REST, coast ~0) and its first
    non-zero speed, while the real coast of a vehicle with the default
    ``RollingResistance`` stays near its full value down to ~0.5 m/s. So, with
    ``s = sign(v)`` (``gear`` when ``|v| <= 1e-3``), ``sigma`` = ``s`` if that
    side has a non-zero grid speed else ``-s``, ``v_h`` = that side's grid
    speed nearest 0, ``u = |v|``, ``w = sigma * max(u, v_h)``::

        G     = accel_at(w, 0, pitch, roll) - accel_at(w, 0, 0, 0)
        flat  = accel_at(w, 0, 0, 0)                                  u >= v_h
              = accel_at(sigma*v_h, 0, 0, 0) * tanh(u/v_f) / tanh(v_h/v_f)   u < v_h
        flat  = -flat  when sigma != s   (mirror: the flat part only)
        a_coast = G + flat

    Enforced properties: on a sampled side with ``u >= v_h`` the result is
    ``accel_at(v, 0, pitch, roll)`` exactly; it is continuous at ``u = v_h``;
    at ``v = 0`` it is ``G`` (no rolling resistance at rest). It is continuous
    through ``v = 0`` only on FLAT ground: on a slope ``G`` switches between the
    two sides' rows at ``|v| = 1e-3`` (0.15 m/s^2 on the reference table at
    10 deg). The slope term is read at ``v_h`` or above and is never faded; the
    mirror on a side with no grid speed assumes a direction-symmetric coast
    (measured for the reference car only) and keeps ``G``'s sign.

    What the fade assumes: the shape ``tanh(|v| / COAST_FADE_SPEED)`` matches a
    table measured on a vehicle with the ``"control"`` stability profile
    (``RollingResistance``, ``tanh_scale`` 0.5). With ``"raw"`` / ``"research"``
    (no ``RollingResistance``) or another ``tanh_scale`` it is an assumption:
    below about 1 m/s the flat-coast error is then at most
    ``|accel_at(v_h, 0, 0, 0)|`` of that table (0.142 m/s^2 for the reference
    car). On a slope the conversion is not measured; uphill table rows are
    biased by the vehicle rolling back inside the measurement window (reference
    table, 10 deg: -1.472 vs about -1.627 m/s^2), which makes the modelled coast
    less decelerating and the commanded brake larger than needed.

    Known limitation (dead band): the plant's ``throttle_for`` still inverts the
    table's LINEAR low-speed coast. Below ``v_h``, for an acceleration target
    between the faded coast and the table's linear coast (about -0.13 and
    -0.071 m/s^2 at 1 m/s on the reference car) the plant returns a slightly
    negative throttle, :func:`brake_for` clips the brake to 0, and the vehicle
    coasts harder than asked, by at most the gap between the two coast values
    (0.07 m/s^2 there)."""
    hold = getattr(plant, "coast_hold_speeds", None)
    if hold is None:
        return float(plant.accel_at(v, 0.0, pitch_deg, roll_deg))
    v_neg, v_pos = hold
    s = (1.0 if v > 0 else -1.0) if abs(v) > 1e-3 else (1.0 if gear >= 0 else -1.0)
    side = {1.0: v_pos, -1.0: v_neg}
    sigma = s if side[s] is not None else -s
    if side[sigma] is None:
        return float(plant.accel_at(v, 0.0, pitch_deg, roll_deg))
    v_h = abs(float(side[sigma]))
    u = abs(float(v))
    if u >= v_h and sigma == s:
        # On a sampled side at or above v_h, G + flat IS accel_at(v, ...); return
        # it directly so the rev-2 value is kept bit for bit.
        return float(plant.accel_at(v, 0.0, pitch_deg, roll_deg))
    w = sigma * max(u, v_h)
    flat_w = float(plant.accel_at(w, 0.0, 0.0, 0.0))
    g_term = float(plant.accel_at(w, 0.0, pitch_deg, roll_deg)) - flat_w
    if u >= v_h:
        flat = flat_w
    else:
        f = COAST_FADE_SPEED
        flat = (float(plant.accel_at(sigma * v_h, 0.0, 0.0, 0.0))
                * math.tanh(u / f) / math.tanh(v_h / f))
    if sigma != s:
        flat = -flat
    return g_term + flat
