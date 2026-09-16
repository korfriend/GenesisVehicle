"""Drivetrain strategies. Distributes (throttle, brake) across wheels.

All strategies return per-wheel drive torque and per-wheel brake torque,
both shape (n_envs, n_wheels), in N*m. Brake is always positive magnitude;
the per-wheel loop in core.py applies it opposite to wheel ω via tanh smoothing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from .._hotset import derived
from ..config import ConfigError


class DrivetrainStrategy(ABC):
    """Maps user (throttle, brake) to per-wheel (T_drive, T_brake)."""

    #: Drive-side wheel angular-velocity cap (rad/s); ``None`` = uncapped.
    #: This is the de-facto top-speed governor: with negligible aero drag a
    #: driven wheel spins up to this cap, so top speed = omega_max_drive x
    #: wheel_radius. Presets set it from a target top speed (m/s) via
    #: ``genesis_vehicle.units.omega_from_top_speed``. (v1.2.3)
    omega_max_drive: Optional[float] = None

    @abstractmethod
    def distribute_torque(
        self,
        inputs: Any,
        omega: torch.Tensor,        # (n_envs, n_wheels)
        wheel_meta: Any,
        device: Any,
        dtype: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (T_drive_per_wheel, T_brake_per_wheel) both (n_envs, n_wheels)."""

    def _drive_omega_cap(self) -> float:
        """``omega_max_drive`` with ``None`` normalised to ``+inf``.

        The PUBLIC attribute keeps ``None`` (``Vehicle.set_omega_max_drive(None)``
        reads back as ``None``); only this derived value is uncapped-as-``+inf``.
        Two reasons it is not an ``if`` at the read site: ``+inf`` makes the
        taper the exact identity (``1 - |w|/inf = 1``, and ``T * 1.0`` is
        exact — measured 0/120,000 mismatches over scalar, 0-d, all-inf and
        MIXED rows, plan v4), and a fused group cannot put ``None`` in a
        per-row tensor, so keeping the ``is None`` test would make the whole
        group behave like whichever row was tested."""
        return derived(
            self, "_omega_cap", None, (self.omega_max_drive,),
            lambda: (float("inf") if self.omega_max_drive is None
                     else float(self.omega_max_drive)),
        )

    def _rev_limit(self, T_drive: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """Soft rev-limiter (v1.2.3). Taper drive torque linearly to 0 as
        ``|omega|`` approaches ``omega_max_drive``, but only in the direction
        the wheel is already turning — engine braking (drive torque opposing
        rotation) keeps full torque. The identity when the cap is None (which
        is carried as ``+inf``; see :meth:`_drive_omega_cap`).

        Element-wise on ``(n_envs, n_wheels)``; non-driven wheels carry
        ``T_drive == 0`` so the taper leaves them untouched. PerSide has its own
        per-side variant and does not use this."""
        taper = torch.clamp(1.0 - torch.abs(omega) / self._drive_omega_cap(), min=0.0)
        driving = (T_drive * omega) > 0
        return torch.where(driving, T_drive * taper, T_drive)

    def _prime_derived(self, wheel_meta: Any, device: Any, dtype: Any) -> None:
        """Build-time hook (``_hotset.prime_derived``)."""
        self._drive_omega_cap()

    def validate(self, wheels: list[Any]) -> None:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_tensor(x: Any, n_envs: int, device: Any, dtype: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        t = x.to(device=device, dtype=dtype)
        if t.dim() == 0:
            t = t.expand(n_envs)
        return t.contiguous()
    return torch.full((n_envs,), float(x), device=device, dtype=dtype)


def _axles_key(axles: Any) -> Any:
    """Cache key for an axle tuple source: by VALUE, so an in-place edit of a
    list (``driven_axles[0] = 1``) is a different key."""
    return None if axles is None else tuple(axles)


def _bias_key(bias: Any) -> Any:
    """Cache key for a per-wheel weight list source: by VALUE (see above)."""
    if bias is None:
        return None
    if torch.is_tensor(bias):
        return tuple(bias.detach().flatten().tolist())
    return tuple(bias)


def _axle_drive_mask(
    wheel_meta: Any, axles: tuple[int, ...], device: Any, dtype: Any
) -> torch.Tensor:
    """Boolean mask (n_wheels,) marking driven wheels."""
    mask = torch.zeros(wheel_meta.n_wheels, device=device, dtype=torch.bool)
    for a in axles:
        mask = mask | (wheel_meta.axle_index == a)
    return mask


def _normalise_weights(
    weights: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Renormalise so weights[mask] sums to 1; weights[~mask] = 0."""
    w = weights * mask.to(dtype=weights.dtype)
    s = w.sum()
    if float(s) <= 1e-12:
        return w
    return w / s


def _default_brake_bias(wheel_meta: Any, device: Any, dtype: Any) -> torch.Tensor:
    """60/40 front/rear split (equal within each axle) on a 2-axle vehicle,
    uniform otherwise. The default used when ``brake_bias is None``; shared by
    RWD/FWD and AWD, which carried byte-identical copies of it."""
    n = wheel_meta.n_wheels
    axle = wheel_meta.axle_index
    n_axles = int(axle.max()) + 1
    # Assume axle 0 = front, axle n_axles-1 = rear. With > 2 axles, fall back to uniform.
    if n_axles == 2:
        front_mask = (axle == 0).to(dtype=dtype)
        rear_mask = (axle == 1).to(dtype=dtype)
        n_front = float(front_mask.sum().item()) or 1.0
        n_rear = float(rear_mask.sum().item()) or 1.0
        return 0.6 * front_mask / n_front + 0.4 * rear_mask / n_rear
    return torch.full((n,), 1.0 / n, device=device, dtype=dtype)


def _brake_bias_tensor(bias: Any, wheel_meta: Any, device: Any,
                       dtype: Any) -> torch.Tensor:
    """``brake_bias`` as an ``(n_wheels,)`` tensor, or the default."""
    if bias is None:
        return _default_brake_bias(wheel_meta, device, dtype)
    bb = torch.tensor(bias, device=device, dtype=dtype)
    if bb.numel() != wheel_meta.n_wheels:
        raise ConfigError(
            f"brake_bias length {bb.numel()} != n_wheels {wheel_meta.n_wheels}")
    return bb


# ---------------------------------------------------------------------------
# RWD / FWD
# ---------------------------------------------------------------------------


class RWD(DrivetrainStrategy):
    """Rear-wheel drive: torque split equally across rear axle wheels.
    Brake bias defaults to front-biased 60/40 (split equally within each axle)."""

    def __init__(
        self,
        t_drive_max: float,
        t_brake_max: float,
        driven_axles: Optional[tuple[int, ...]] = None,
        brake_bias: Optional[list[float]] = None,
        omega_max_drive: Optional[float] = None,
    ):
        self.t_drive_max = float(t_drive_max)
        self.t_brake_max = float(t_brake_max)
        # If driven_axles is None, it's resolved to the rear-most axle at distribute time.
        self.driven_axles = driven_axles
        self.brake_bias = brake_bias
        self.omega_max_drive = (
            float(omega_max_drive) if omega_max_drive is not None else None)

    def _resolve_driven(self, wheel_meta: Any) -> tuple[int, ...]:
        """The driven axle tuple, derived once per (wheel_meta, driven_axles).

        A post-build write to ``driven_axles`` still lands on the next call —
        the cache key carries it (``HOT_DEPENDENTS[("drivetrain",
        "driven_axles")]``)."""
        return derived(
            self, "_driven", wheel_meta, (_axles_key(self.driven_axles),),
            lambda: self._compute_driven(wheel_meta),
        )

    def _compute_driven(self, wheel_meta: Any) -> tuple[int, ...]:
        if self.driven_axles is not None:
            return tuple(int(a) for a in self.driven_axles)
        return (int(wheel_meta.axle_index.max()),)

    def _drive_share_pw(self, wheel_meta: Any, device: Any,
                        dtype: Any) -> torch.Tensor:
        """Per-wheel drive share ``drive_mask / n_driven`` — ``(n_wheels,)``.

        Hoists the mask build, the ``n_driven`` count (a device->host sync via
        ``.item()``) and the division out of the step path. Depends on the same
        source as :meth:`_resolve_driven`."""
        return derived(
            self, "_drive_share", wheel_meta,
            (_axles_key(self.driven_axles), device, dtype),
            lambda: self._compute_drive_share(wheel_meta, device, dtype),
        )

    def _compute_drive_share(self, wheel_meta: Any, device: Any,
                             dtype: Any) -> torch.Tensor:
        driven = self._resolve_driven(wheel_meta)
        drive_mask = _axle_drive_mask(wheel_meta, driven, device, dtype).to(dtype=dtype)
        n_driven = float(drive_mask.sum().item()) or 1.0
        # Equal split among driven wheels.
        return drive_mask / n_driven

    def _resolve_brake_bias(
        self, wheel_meta: Any, device: Any, dtype: Any
    ) -> torch.Tensor:
        """Per-wheel brake bias — ``(n_wheels,)``, derived once.

        In-place edits of the ``brake_bias`` LIST are seen: the cache key holds
        it as a tuple, by value."""
        return derived(
            self, "_brake_bias", wheel_meta,
            (_bias_key(self.brake_bias), device, dtype),
            lambda: _brake_bias_tensor(self.brake_bias, wheel_meta, device, dtype),
        )

    def _prime_derived(self, wheel_meta: Any, device: Any, dtype: Any) -> None:
        super()._prime_derived(wheel_meta, device, dtype)
        self._drive_share_pw(wheel_meta, device, dtype)
        self._resolve_brake_bias(wheel_meta, device, dtype)

    def distribute_torque(
        self,
        inputs: Any,
        omega: torch.Tensor,
        wheel_meta: Any,
        device: Any,
        dtype: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_envs = omega.shape[0]
        throttle = _to_tensor(inputs.throttle, n_envs, device, dtype)
        brake = _to_tensor(inputs.brake, n_envs, device, dtype)

        per_wheel_share = self._drive_share_pw(wheel_meta, device, dtype)  # (n_wheels,)
        T_drive = (throttle * self.t_drive_max).unsqueeze(-1) * per_wheel_share.unsqueeze(0)

        T_drive = self._rev_limit(T_drive, omega)
        bb = self._resolve_brake_bias(wheel_meta, device, dtype)   # (n_wheels,)
        T_brake = (brake * self.t_brake_max).unsqueeze(-1) * bb.unsqueeze(0)
        return T_drive, T_brake


class FWD(RWD):
    """Front-wheel drive: same split logic as RWD but driven axle defaults to front."""

    def _compute_driven(self, wheel_meta: Any) -> tuple[int, ...]:
        if self.driven_axles is not None:
            return tuple(int(a) for a in self.driven_axles)
        return (int(wheel_meta.axle_index.min()),)


class AWD(DrivetrainStrategy):
    """All-wheel drive: per-wheel drive weights (default uniform across all wheels)."""

    def __init__(
        self,
        t_drive_max: float,
        t_brake_max: float,
        drive_weights: Optional[list[float]] = None,
        brake_bias: Optional[list[float]] = None,
        omega_max_drive: Optional[float] = None,
    ):
        self.t_drive_max = float(t_drive_max)
        self.t_brake_max = float(t_brake_max)
        self.drive_weights = drive_weights
        self.brake_bias = brake_bias
        self.omega_max_drive = (
            float(omega_max_drive) if omega_max_drive is not None else None)

    def _norm_drive_weights(self, wheel_meta: Any, device: Any,
                            dtype: Any) -> torch.Tensor:
        """Normalised per-wheel drive weights — ``(n_wheels,)``, derived once.

        In-place edits of the ``drive_weights`` LIST are seen (keyed by value)."""
        return derived(
            self, "_drive_weights", wheel_meta,
            (_bias_key(self.drive_weights), device, dtype),
            lambda: self._compute_drive_weights(wheel_meta, device, dtype),
        )

    def _compute_drive_weights(self, wheel_meta: Any, device: Any,
                               dtype: Any) -> torch.Tensor:
        n = wheel_meta.n_wheels
        if self.drive_weights is None:
            return torch.full((n,), 1.0 / n, device=device, dtype=dtype)
        dw = torch.tensor(self.drive_weights, device=device, dtype=dtype)
        if dw.numel() != n:
            raise ConfigError(f"drive_weights length {dw.numel()} != n_wheels {n}")
        s = dw.sum()
        if float(s) > 1e-12:
            dw = dw / s
        return dw

    def _brake_bias_pw(self, wheel_meta: Any, device: Any,
                       dtype: Any) -> torch.Tensor:
        """Per-wheel brake bias — ``(n_wheels,)``, derived once.

        Deliberately NOT named ``_resolve_brake_bias``: ``tools/fusion_probe``
        keys its report off ``getattr(drivetrain, "_resolve_brake_bias")``, so
        adding that name to a class that never had it would change the probe's
        output."""
        return derived(
            self, "_brake_bias", wheel_meta,
            (_bias_key(self.brake_bias), device, dtype),
            lambda: _brake_bias_tensor(self.brake_bias, wheel_meta, device, dtype),
        )

    def _prime_derived(self, wheel_meta: Any, device: Any, dtype: Any) -> None:
        super()._prime_derived(wheel_meta, device, dtype)
        self._norm_drive_weights(wheel_meta, device, dtype)
        self._brake_bias_pw(wheel_meta, device, dtype)

    def distribute_torque(
        self,
        inputs: Any,
        omega: torch.Tensor,
        wheel_meta: Any,
        device: Any,
        dtype: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_envs = omega.shape[0]
        throttle = _to_tensor(inputs.throttle, n_envs, device, dtype)
        brake = _to_tensor(inputs.brake, n_envs, device, dtype)

        dw = self._norm_drive_weights(wheel_meta, device, dtype)
        bb = self._brake_bias_pw(wheel_meta, device, dtype)

        T_drive = (throttle * self.t_drive_max).unsqueeze(-1) * dw.unsqueeze(0)
        T_drive = self._rev_limit(T_drive, omega)
        T_brake = (brake * self.t_brake_max).unsqueeze(-1) * bb.unsqueeze(0)
        return T_drive, T_brake


# ---------------------------------------------------------------------------
# PerSide (skid-steer / tank)
# ---------------------------------------------------------------------------


class PerSide(DrivetrainStrategy):
    """Per-side drivetrain for skid-steer vehicles (tank tracks).

    Throttle in [-1, 1] is the SDK contract. PerSide treats the forward part as
    forward drive and uses the user's brake input for braking; reverse drive
    is achieved by passing negative throttle.

    Steer differential: ISO 8855 +steer_diff = right turn, so the LEFT side
    spins faster and the RIGHT side slower. Internally this is implemented as
    side_throttle_L = throttle + steer_gain * steer_diff
    side_throttle_R = throttle - steer_gain * steer_diff
    """

    def __init__(
        self,
        t_drive_max: float,
        t_brake_max: float,
        steer_gain: float = 1.0,
        omega_max_drive: float = 100.0,
        throttle_gear_cap: float = 1.0,
        use_per_side_taper: bool = True,
    ):
        self.t_drive_max = float(t_drive_max)
        self.t_brake_max = float(t_brake_max)
        self.steer_gain = float(steer_gain)
        self.omega_max_drive = float(omega_max_drive)
        self.throttle_gear_cap = float(throttle_gear_cap)
        self.use_per_side_taper = bool(use_per_side_taper)

    @staticmethod
    def _cap_torque_batched(
        omega_side: torch.Tensor, T_des: torch.Tensor, omega_max: float, T_max: float
    ) -> torch.Tensor:
        """Batched per-side torque cap. omega_side / T_des shape: (n_envs,).
        Cap drive only in the same direction as omega; opposite direction (decel)
        keeps full T_max."""
        taper_pos = torch.clamp(1.0 - torch.clamp(omega_side, min=0.0) / omega_max, min=0.0)
        taper_neg = torch.clamp(1.0 - torch.clamp(-omega_side, min=0.0) / omega_max, min=0.0)
        T_hi = T_max * taper_pos
        T_lo = -T_max * taper_neg
        return torch.maximum(T_lo, torch.minimum(T_hi, T_des))

    def distribute_torque(
        self,
        inputs: Any,
        omega: torch.Tensor,
        wheel_meta: Any,
        device: Any,
        dtype: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_envs = omega.shape[0]
        n = wheel_meta.n_wheels
        # Accept either SkidSteerInputs (steer_diff) or unified-style (steer).
        throttle = _to_tensor(inputs.throttle, n_envs, device, dtype)
        brake = _to_tensor(inputs.brake, n_envs, device, dtype)
        steer = getattr(inputs, "steer_diff", None)
        if steer is None:
            steer = getattr(inputs, "steer")
        steer = _to_tensor(steer, n_envs, device, dtype)

        # Gear cap: scale throttle before physics (reference-tank tuning).
        throttle = throttle * self.throttle_gear_cap

        # Per-side commanded throttle. ISO: +steer = right turn -> L faster.
        thr_L = throttle + self.steer_gain * steer
        thr_R = throttle - self.steer_gain * steer

        # Per-side mean omega for taper (over the wheels on each side).
        L_idx = wheel_meta.left_idx
        R_idx = wheel_meta.right_idx
        if L_idx.numel() == 0 or R_idx.numel() == 0:
            raise ConfigError("PerSide requires at least one left and one right wheel.")
        omega_L = omega[:, L_idx].mean(dim=-1)   # (n_envs,)
        omega_R = omega[:, R_idx].mean(dim=-1)

        T_des_L = thr_L * self.t_drive_max
        T_des_R = thr_R * self.t_drive_max
        if self.use_per_side_taper:
            # Same derived cap as _rev_limit (identical float for any non-None
            # setting; +inf, i.e. no taper, when a caller clears it).
            cap = self._drive_omega_cap()
            T_drive_L = self._cap_torque_batched(omega_L, T_des_L, cap, self.t_drive_max)
            T_drive_R = self._cap_torque_batched(omega_R, T_des_R, cap, self.t_drive_max)
        else:
            T_drive_L = T_des_L
            T_drive_R = T_des_R

        T_drive = torch.zeros(n_envs, n, device=device, dtype=dtype)
        T_drive[:, L_idx] = T_drive_L.unsqueeze(-1)
        T_drive[:, R_idx] = T_drive_R.unsqueeze(-1)

        # Brake distributed equally across all wheels.
        T_brake = (brake * self.t_brake_max / float(n)).unsqueeze(-1).expand(n_envs, n).contiguous()
        return T_drive, T_brake

    def validate(self, wheels: list[Any]) -> None:
        if any(w.side is None for w in wheels):
            raise ConfigError("PerSide requires every wheel to have side='L' or 'R'.")
