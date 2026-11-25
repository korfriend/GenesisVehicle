# SPDX-License-Identifier: Apache-2.0
# Derived from the field port (PathFollower_MPC/mpc/alpasim_mpc.py:24-59 and
# mpc_follower.py:72-89) of NVlabs/alpasim (Copyright (c) 2026 NVIDIA
# Corporation, Apache-2.0); the upstream revision was not recorded by the port.
# Modified by the GenesisVehicle contributors (see NOTICE).
"""MPC vehicle parameters, solver gains and acceleration limits (numpy only).

Three objects, kept apart from the solvers so that importing them needs
neither casadi/do_mpc nor osqp:

* :class:`MPCVehicleParams` — the MEASURED record of one vehicle, as written
  by the identify tool and stored as JSON (schema
  ``genesis_vehicle.mpc_params/1``).
* :class:`MPCGains` — the AlpaSim cost weights (defaults = AlpaSim).
* ``_AlpaSimParams`` — the solver-side parameter set the field port called
  ``VehicleParams`` (same fields, same meaning, same defaults); derived from
  an :class:`MPCVehicleParams` by :meth:`MPCVehicleParams.solver_params`.

(v2.1.0) ``solver_params(decel_limit=)`` selects the deceleration bound's
source table (``"throttle"`` default, ``"brake"``) of ONE parameter set — a
table selector with no gate. ``PathFollower(mpc_decel_limit="brake")`` builds
both sets and uses the brake one only on end-gated solves (see
``PathFollower``).

Schema rule (fixed for 2.x): :meth:`MPCVehicleParams.from_json` accepts
exactly ``genesis_vehicle.mpc_params/1``; every 2.x reader keeps accepting
``/1``; any other or missing ``schema`` is a ``ValueError``. A JSON written by
the field ``PathFollower_MPC/tools/identify.py`` has no ``schema`` key and is
therefore rejected — re-run the SDK identify tool.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SCHEMA = "genesis_vehicle.mpc_params/1"

#: Prediction models a params record can feed, per SDK release. 2.0.0 ships the
#: field bicycle model only.
SUPPORTED_MODELS = ("bicycle",)

#: (v2.1.0) Deceleration-table sources of ``solver_params(decel_limit=)``:
#: ``"throttle"`` (default, the 2.0.x bound) = the record's
#: ``decel_throttle_by_speed`` (throttle -1, brake 0); ``"brake"`` = its
#: ``decel_brake_by_speed`` (brake 1, throttle 0, flat ground, at the record's mu).
#: This selects a table only; the follower's ``mpc_decel_limit="brake"`` uses the
#: brake table on end-gated solves and the throttle table on all others.
DECEL_LIMITS = ("throttle", "brake")


# --- solver-side parameters (field VehicleParams) ------------------------------

@dataclass
class _AlpaSimParams:
    """The field port's ``VehicleParams`` (``alpasim_mpc.py:24-41``), verbatim.

    Defaults are the AlpaSim original (Ford Fusion). ``None`` limits mean the
    AlpaSim constraints (wheel angle +-pi/4, acceleration state -8..6, command
    -9..6). ``accel_*_tab`` = (speeds [m/s], values [m/s^2]), piecewise linear,
    clamped at both ends; a table wins over the constant.
    """
    mass: float = 2014.4
    inertia: float = 3414.2
    l_rig_to_cg: float = 1.59
    wheelbase: float = 2.85
    front_cornering_stiffness: float = 93534.5
    rear_cornering_stiffness: float = 176162.1
    steering_time_constant: float = 0.1
    acceleration_time_constant: float = 0.1
    kinematic_threshold_speed: float = 5.0
    steer_max: Optional[float] = None
    accel_max: Optional[float] = None
    accel_min: Optional[float] = None
    accel_max_tab: Optional[tuple] = None
    accel_min_tab: Optional[tuple] = None


def _decel_rows(data: dict, decel_limit: str):
    """The sorted ``[(v, a), ...]`` rows of the deceleration bound's source
    table. ``ValueError`` for an unknown ``decel_limit`` and, under
    ``"brake"``, for an empty ``decel_brake_by_speed`` or any row whose speed
    is not a finite number or whose value is not a finite number < 0."""
    if decel_limit not in DECEL_LIMITS:
        raise ValueError(f"mpc_decel_limit must be one of {list(DECEL_LIMITS)}; got {decel_limit!r}")
    if decel_limit == "throttle":
        return sorted((float(v), float(a)) for v, a in data["decel_throttle_by_speed"].items())
    tab = data.get("decel_brake_by_speed")
    hint = " — re-run `python -m genesis_vehicle.control.identify`"
    if not isinstance(tab, dict) or not tab:
        raise ValueError("mpc_decel_limit=\"brake\" needs a non-empty decel_brake_by_speed in the "
                         f"params record; got {tab!r}{hint}")
    rows = []
    for v, a in tab.items():
        try:
            fv, fa = float(v), float(a)
        except (TypeError, ValueError):
            raise ValueError(f"decel_brake_by_speed row {v!r}: {a!r} is not numeric{hint}") from None
        if not (np.isfinite(fv) and np.isfinite(fa) and fa < 0.0):
            raise ValueError(f"mpc_decel_limit=\"brake\": decel_brake_by_speed row {v!r}: {a!r} must be a "
                             f"finite deceleration < 0 at a finite speed{hint}")
        rows.append((fv, fa))
    return sorted(rows)


def accel_limits(p, v):
    """``(max deceleration, max acceleration)`` at speed ``v`` — the table if
    there is one, else the constant, else ``None`` (``alpasim_mpc.py:44-48``)."""
    lo = float(np.interp(v, *p.accel_min_tab)) if p.accel_min_tab else p.accel_min
    hi = float(np.interp(v, *p.accel_max_tab)) if p.accel_max_tab else p.accel_max
    return lo, hi


@dataclass
class MPCGains:
    """AlpaSim MPC cost weights (``alpasim_mpc.py:51-59``; defaults = AlpaSim).

    Tracking terms count only from horizon node ``idx_start_penalty`` on (the
    first second of the 2 s horizon is free). ``rel_*`` weigh the command
    (NMPC: command CHANGE, do_mpc ``rterm``; LMPC with ``rterm="abs"``: command
    magnitude).
    """
    long_position_weight: float = 2.0
    lat_position_weight: float = 1.0
    heading_weight: float = 1.0
    acceleration_weight: float = 0.1
    rel_front_steering_angle_weight: float = 5.0
    rel_acceleration_weight: float = 1.0
    idx_start_penalty: int = 10


# --- the measured vehicle record ------------------------------------------------

#: Keys a ``genesis_vehicle.mpc_params/1`` record must carry. The field
#: identify keys (minus ``steer_cap``, which the follower supplies) plus the
#: schema, provenance and the steering description the follower checks.
REQUIRED_KEYS = (
    "schema", "sdk_version", "genesis_version", "urdf", "preset", "config",
    "plant_overrides", "dt", "substeps",
    "mass", "izz_cg", "wheelbase", "rig_offset", "l_rig_to_cg", "max_steer_rad",
    "wheel_radius", "steer_input_sign",
    "front_cornering_stiffness", "rear_cornering_stiffness", "cornering_samples",
    "accel_max_by_speed", "decel_throttle_by_speed", "decel_brake_by_speed",
    "accel_tau", "steer_yaw_tau",
    "steering", "n_axles",
)

#: Keys that may be absent or ``null`` (filled by the identify tool from the
#: vehicle config; the bicycle model does not read them).
OPTIONAL_KEYS = ("tyre", "axle_loads", "deadbeat_scale", "v_star", "drivetrain",
                 "fit_residual", "envelope", "brake_gain")
# ``brake_gain`` (v2.0.2): the closed-form brake gain k_b (m/s^2 per unit
# brake, ``control.brake.brake_gain_closed_form``) the gear-mode follower's
# stage 2 uses; a record from before 2.0.2 lacks it (from_dict fills None), and
# the follower then derives k_b from ``decel_brake_by_speed``.

#: Steering strategies the MPC model describes (2-axle front-steer vehicles).
ACKERMANN_STEERING = ("Ackermann", "PartialAckermann")


@dataclass
class MPCVehicleParams:
    """One vehicle's measured MPC record (the identify tool's JSON).

    ``data`` holds every key of the record verbatim; the common ones are
    exposed as attributes. Build one with :meth:`from_json` /
    :meth:`from_dict`; derive the solver parameters with
    :meth:`solver_params`.
    """
    data: dict = field(repr=False)

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> "MPCVehicleParams":
        if not isinstance(d, dict):
            raise ValueError(f"MPC params must be a JSON object; got {type(d).__name__}")
        schema = d.get("schema")
        if schema != SCHEMA:
            hint = (" (a JSON written by the field PathFollower_MPC identify tool has "
                    "no schema key — re-run `python -m genesis_vehicle.control.identify`)"
                    if schema is None else "")
            raise ValueError(f"MPC params schema must be {SCHEMA!r}; got {schema!r}{hint}")
        missing = [k for k in REQUIRED_KEYS if k not in d]
        if missing:
            raise ValueError(f"MPC params are missing keys: {', '.join(missing)}")
        steering, n_axles = d["steering"], d["n_axles"]
        if steering not in ACKERMANN_STEERING or n_axles != 2:
            raise ValueError(
                "the MPC layer models a 2-axle Ackermann vehicle; this vehicle has "
                f"{steering}/{n_axles} axles — pass mpc_model=\"off\"")
        if not d["accel_max_by_speed"] or not d["decel_throttle_by_speed"]:
            raise ValueError("MPC params need non-empty accel_max_by_speed and "
                             "decel_throttle_by_speed")
        data = dict(d)
        for k in OPTIONAL_KEYS:
            data.setdefault(k, None)
        return cls(data=data)

    @classmethod
    def from_json(cls, path) -> "MPCVehicleParams":
        path = os.fspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"MPC params file not found: {path}")
        with open(path) as fh:
            d = json.load(fh)
        return cls.from_dict(d)

    def to_dict(self) -> dict:
        return dict(self.data)

    def to_json(self, path) -> None:
        with open(os.fspath(path), "w") as fh:
            json.dump(self.data, fh, indent=1)

    # -- attributes -------------------------------------------------------------

    def __getattr__(self, name):
        data = self.__dict__.get("data")
        if data is not None and name in data:
            return data[name]
        raise AttributeError(name)

    # -- derived solver parameters ------------------------------------------------

    def solver_params(self, model: str = "bicycle", *, steer_cap: float = 1.0,
                      dt_ctrl: float = 0.025, decel_limit: str = "throttle") -> _AlpaSimParams:
        """The solver-side parameter set for prediction ``model``.

        ``"bicycle"`` = the field ``params_from_json`` (``mpc_follower.py:72-89``)
        with the steering limit taken from the follower (field ``demo.py:97``):

        * ``steer_max = steer_cap * max_steer_rad``;
        * time constants floored at ``dt_ctrl`` (a command changes once per
          control period, so a faster response cannot be realised);
        * acceleration limit tables from the measured per-speed values; the
          deceleration table gets a 0 m/s entry equal to its lowest measured
          speed's value (identify does not measure decel at rest);
        * constant limits ``accel_max = min(table)``, ``accel_min = max(table)``.

        ``decel_limit`` (v2.1.0, keyword-only): the source of the deceleration
        table and of ``accel_min`` — ``"throttle"`` (default; the 2.0.x rule,
        ``decel_throttle_by_speed``, and the result is field-for-field the call
        without the keyword) or ``"brake"`` (``decel_brake_by_speed``: brake 1,
        throttle 0, flat ground, at the mu the record was measured at; the same
        0 m/s rule). Under ``"brake"`` an empty brake row or a row that is not a
        finite deceleration < 0 raises ``ValueError``, and a ``UserWarning``
        names each speed, among the speeds present in BOTH rows, at which the
        brake row decelerates LESS than the throttle row (that would narrow the
        bound there); a speed in only one row is not compared. Nothing here checks
        that the record's brake row is reachable by the vehicle being driven
        (a larger stage-2 ``brake_gain=``, lower live mu, payload or a slope
        make it unreachable, silently). This is a TABLE selector: the result
        is one parameter set with that bound everywhere; the follower's
        ``mpc_decel_limit="brake"`` builds both sets and uses the brake one only
        on end-gated solves (``PathFollower``).

        Raises ``ValueError`` for a model this release does not ship and for an
        unknown ``decel_limit``.
        """
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"mpc_model must be one of {list(SUPPORTED_MODELS) + ['off']}; "
                             f"got {model!r}")
        j = self.data
        dec = _decel_rows(j, decel_limit)
        if decel_limit == "brake":
            thr = dict(_decel_rows(j, "throttle"))
            weaker = [v for v, a in dec if v in thr and a > thr[v]]
            if weaker:
                import warnings
                warnings.warn("mpc_decel_limit=\"brake\": the record's brake row decelerates less than its "
                              f"throttle row at {weaker} m/s, so the brake bound is NARROWER there",
                              UserWarning, stacklevel=2)
        p = _AlpaSimParams()
        p.mass, p.inertia = float(j["mass"]), float(j["izz_cg"])
        p.wheelbase, p.l_rig_to_cg = float(j["wheelbase"]), float(j["l_rig_to_cg"])
        p.front_cornering_stiffness = float(j["front_cornering_stiffness"])
        p.rear_cornering_stiffness = float(j["rear_cornering_stiffness"])
        p.steer_max = float(steer_cap) * float(j["max_steer_rad"])
        amx = sorted((float(v), float(a)) for v, a in j["accel_max_by_speed"].items())
        p.accel_max_tab = ([v for v, _ in amx], [a for _, a in amx])
        p.accel_min_tab = ([0.0] + [v for v, _ in dec], [dec[0][1]] + [a for _, a in dec])
        p.accel_max, p.accel_min = min(a for _, a in amx), max(a for _, a in dec)
        p.acceleration_time_constant = max(float(j["accel_tau"]), dt_ctrl)
        p.steering_time_constant = max(float(j["steer_yaw_tau"]), dt_ctrl)
        return p


def load_mpc_params(path_or_params) -> MPCVehicleParams:
    """``MPCVehicleParams`` from a path or an existing instance (pass-through)."""
    if isinstance(path_or_params, MPCVehicleParams):
        return path_or_params
    if isinstance(path_or_params, (str, os.PathLike)):
        return MPCVehicleParams.from_json(path_or_params)
    raise TypeError("mpc_params must be a path to a params JSON or an "
                    f"MPCVehicleParams; got {type(path_or_params).__name__}")

