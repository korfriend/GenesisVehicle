"""Opt-in gear mode (v2.0.2) — the pure functions both batched drivers share.

A vehicle input carries an optional ``gear`` (``+1`` forward, ``-1`` reverse).
``gear=None`` (the default everywhere) is the 2.0.1 signed-throttle input and
takes none of the code below: :func:`encode_requests` returns ``None`` when no
input of a step carries a gear, and the drivers then run their 2.0.1 lines.

With a gear request, for that vehicle row:

* the throttle is read as a MAGNITUDE: ``clamp(throttle, 0, 1)``, applied in the
  direction of the ENGAGED gear (:func:`effective_throttle`);
* the engaged gear changes to the requested one only on a step where the
  clamped throttle is exactly 0 and the planar chassis speed
  ``hypot(vel_x, vel_y)`` (world frame) is below :data:`GEAR_SHIFT_MAX_SPEED`
  (:func:`update_gear`). A request that violates this is DEFERRED, not refused:
  the input persists, so it is re-evaluated every step and engages on the first
  step the condition holds. The drivers do not evaluate it on the first-step
  protection step (no chassis state is read there);
* after that update, every wheel's drive torque is clamped to the sign of the
  engaged gear (:func:`clamp_drive_to_gear`), so a custom drivetrain that
  produces torque against the gear (even at throttle 0) is covered too. The
  clamp looks at the torque sign only: driving in the gear's direction while
  the vehicle rolls the other way (a hill start) is allowed.
  A custom (non-``PerSide``) drivetrain that deliberately applies opposite-sign
  torque on some wheels (torque vectoring, a differential brake through the
  drive channel) has that torque ZEROED on those wheels in gear mode.

Where it is enforced: ``VehiclePhysics.step``, ``MultiVehicleKindPhysics.step``
and, through them, ``MultiVehiclePhysics.step`` and ``VehicleScene.step``.
``control.DifferentiablePlant``'s unroll does not use it (it predicts signed
throttle).

Skid-steer vehicles refuse gear mode (:func:`refuses_gear`): a ``PerSide``
drivetrain or a ``SkidSteer`` steering strategy. A custom skid-steer drivetrain
that does not subclass ``PerSide`` (on a non-``SkidSteer`` steering) is NOT
detected.

This module is Genesis-free (torch + ``.strategies`` only); the root package
imports it eagerly.

| abbr | meaning |
|---|---|
| v_xy | planar chassis speed ``hypot(vel_x, vel_y)``, world frame (m/s) |
"""
from __future__ import annotations

import dataclasses
import numbers
import types
from typing import Any, Optional, Sequence

import torch

from .strategies.drivetrain import PerSide
from .strategies.steering import SkidSteer

#: Engaged / requested forward gear.
GEAR_FORWARD = 1
#: Engaged / requested reverse gear.
GEAR_REVERSE = -1
#: A gear change engages only below this planar speed (m/s), at throttle 0.
GEAR_SHIFT_MAX_SPEED = 0.5


def refuses_gear(resolved: Any) -> bool:
    """True for a skid-steer vehicle, which refuses gear mode: its
    ``drivetrain`` is a ``PerSide`` or its ``steering`` a ``SkidSteer``.
    ``resolved`` is anything with those two attributes (a ``ResolvedConfig``
    or, before ``build()``, the ``VehicleConfig``). A custom skid-steer
    drivetrain that does not subclass ``PerSide`` is not detected."""
    return (isinstance(getattr(resolved, "drivetrain", None), PerSide)
            or isinstance(getattr(resolved, "steering", None), SkidSteer))


def _is_scalar(x: Any) -> bool:
    return isinstance(x, numbers.Real) and not torch.is_tensor(x)


def validate_request(gear: Any, throttle: Any, *, where: str) -> None:
    """Value checks on ONE vehicle's gear-mode input (only called when
    ``gear`` is not None): a Python scalar ``gear`` must be ``+1`` or ``-1``
    (a ``bool`` is refused), a Python scalar ``throttle`` must not be negative.
    Raises ``ValueError``.

    Tensors are not checked: a tensor ``gear`` is mapped by its sign (``0``
    reads as ``+1``) and a tensor ``throttle`` is clamped to ``[0, 1]``, so a
    negative tensor element reads as throttle 0 — that is not a refusal."""
    if gear is None:
        return
    if torch.is_tensor(gear):
        pass
    elif isinstance(gear, bool) or not _is_scalar(gear) or float(gear) not in (1.0, -1.0):
        raise ValueError(
            f"{where}: gear must be +1 (forward), -1 (reverse), None (signed "
            f"throttle) or a tensor; got {gear!r}")
    if _is_scalar(throttle) and float(throttle) < 0.0:
        raise ValueError(
            f"{where}: with a gear the throttle is a magnitude in [0, 1]; got "
            f"{throttle!r}. Drive backwards with gear=-1 and a positive throttle, "
            f"or pass gear=None for the signed throttle.")


def check_vehicle_request(resolved: Any, gear: Any, throttle: Any, *,
                          where: str, name: Optional[str] = None) -> None:
    """The full per-vehicle gear check (only when ``gear`` is not None):
    refuse a skid-steer vehicle (``ValueError``), then :func:`validate_request`."""
    if gear is None:
        return
    if refuses_gear(resolved):
        who = f" '{name}'" if name else ""
        raise ValueError(
            f"{where}: vehicle{who} is skid-steer (PerSide drivetrain or SkidSteer "
            f"steering) and refuses gear mode; drive it with the signed throttle "
            f"(gear=None)")
    validate_request(gear, throttle, where=where)


def check_request_shape(gear: Any, n_envs: int, *, where: str) -> None:
    """A tensor gear request must be a scalar tensor or shape ``(n_envs,)``
    (the rule :func:`encode_requests` applies); ``ValueError`` otherwise. Lets
    a multi-kind driver refuse it before any kind steps."""
    if torch.is_tensor(gear) and gear.dim() != 0 and tuple(gear.shape) != (n_envs,):
        raise ValueError(f"{where}: a gear tensor must be a scalar or shape ({n_envs},); "
                         f"got {tuple(gear.shape)}")


def _carries_throttle(inputs: Any) -> bool:
    return (dataclasses.is_dataclass(inputs) and not isinstance(inputs, type)
            and any(f.name == "throttle" for f in dataclasses.fields(inputs)))


def check_input_type(steering: Any, *, where: str) -> None:
    """``TypeError`` when ``steering.InputType.from_unified`` does not produce
    a dataclass with a ``throttle`` field (gear mode passes the effective
    throttle through ``dataclasses.replace``, :func:`with_throttle`). Probed
    with a duck-typed all-zero unified input (no state touched)."""
    probe = types.SimpleNamespace(throttle=0.0, brake=0.0, steer=0.0, gear=None)
    if not _carries_throttle(steering.InputType.from_unified(probe)):
        raise TypeError(
            f"{where}: gear mode needs the steering strategy's input type to be a "
            f"dataclass with a 'throttle' field; {steering.InputType.__name__} is not")


def encode_requests(gears: Sequence[Any], n_envs: int, device: Any,
                    dtype: Any) -> Optional[torch.Tensor]:
    """Per-vehicle gear requests -> a flat ``(n_envs * K,)`` tensor in the
    env-major / vehicle-minor row order of ``MultiVehicleKindPhysics`` (``K =
    len(gears)``; ``(n_envs,)`` for one vehicle). ``+1`` / ``-1`` for a row
    with a request, ``0`` for a row whose input has ``gear=None``. Returns
    ``None`` when EVERY entry is None (the legacy path). A tensor request is
    mapped by sign, ``0`` -> ``+1``; it must be a scalar tensor or shape
    ``(n_envs,)``."""
    if all(g is None for g in gears):
        return None
    cols = []
    for g in gears:
        if g is None:
            cols.append(torch.zeros(n_envs, device=device, dtype=dtype))
        elif torch.is_tensor(g):
            t = g.to(device=device, dtype=dtype)
            if t.dim() == 0:
                t = t.expand(n_envs)
            if tuple(t.shape) != (n_envs,):
                raise ValueError(
                    f"a gear tensor must be a scalar or shape ({n_envs},); "
                    f"got {tuple(t.shape)}")
            cols.append(torch.where(t < 0, -torch.ones_like(t), torch.ones_like(t)))
        else:
            cols.append(torch.full((n_envs,), float(g), device=device, dtype=dtype))
    return torch.stack(cols, dim=1).reshape(-1)


def effective_throttle(throttle: torch.Tensor, req: torch.Tensor,
                       state: torch.Tensor) -> torch.Tensor:
    """Signed throttle the drivetrain sees: ``state * clamp(throttle, 0, 1)``
    on gear rows (``req != 0``), the input throttle unchanged on legacy rows.
    All three are ``(B,)``."""
    return torch.where(req != 0, state * throttle.clamp(0.0, 1.0), throttle)


def update_gear(req: torch.Tensor, state: torch.Tensor, throttle: torch.Tensor,
                v_xy: torch.Tensor):
    """Rule 4: ``(new_state, shifted)``. A row shifts to its request when the
    request differs from the engaged gear, the CLAMPED throttle is exactly 0
    (a negative tensor throttle reads as 0 and allows the shift) and
    ``v_xy < GEAR_SHIFT_MAX_SPEED``. Legacy rows (``req == 0``) never shift."""
    shifted = ((req != 0) & (req != state)
               & (throttle.clamp(0.0, 1.0) == 0.0)
               & (v_xy < GEAR_SHIFT_MAX_SPEED))
    return torch.where(shifted, req, state), shifted


def clamp_drive_to_gear(T_drive: torch.Tensor, req: torch.Tensor,
                        state: torch.Tensor) -> torch.Tensor:
    """Clamp each gear row's per-wheel drive torque ``(B, n)`` to the sign of
    its engaged gear ``state`` (``>= 0`` forward, ``<= 0`` reverse). Legacy
    rows (``req == 0``) are returned unchanged."""
    pos = ((req != 0) & (state > 0)).unsqueeze(-1)
    neg = ((req != 0) & (state < 0)).unsqueeze(-1)
    T_drive = torch.where(pos, T_drive.clamp(min=0.0), T_drive)
    return torch.where(neg, T_drive.clamp(max=0.0), T_drive)


def with_throttle(inputs: Any, throttle: Any) -> Any:
    """A NEW inputs object equal to ``inputs`` but with ``throttle`` replaced
    (``dataclasses.replace``) — the caller's object is never modified. A
    steering ``InputType`` that is not a dataclass with a ``throttle`` field
    cannot carry the effective throttle: ``TypeError``."""
    if not _carries_throttle(inputs):
        raise TypeError(
            f"gear mode needs the steering strategy's input type to be a dataclass "
            f"with a 'throttle' field (to pass the gear-direction throttle to the "
            f"drivetrain); got {type(inputs).__name__}")
    return dataclasses.replace(inputs, throttle=throttle)
