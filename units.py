"""Unit helpers.

The SDK interface is uniformly **SI**: every speed the API takes or returns is
m/s (and every angular rate rad/s). These helpers exist only for the human-
facing edges — a preset default written as "200 km/h", a CLI flag, a log line —
so km/h never leaks into the interface itself. Convert at the boundary, keep m/s
everywhere inside.
"""

from __future__ import annotations

# 1 km/h = 1000 m / 3600 s
_KMH_PER_MPS = 3.6


def kmh_to_mps(kmh: float) -> float:
    """Kilometres per hour -> metres per second."""
    return float(kmh) / _KMH_PER_MPS


def mps_to_kmh(mps: float) -> float:
    """Metres per second -> kilometres per hour."""
    return float(mps) * _KMH_PER_MPS


def omega_from_top_speed(top_speed_mps: float, wheel_radius_m: float) -> float:
    """Rolling angular velocity (rad/s) at a linear speed: ``v / r``.

    Used to turn a target top speed (m/s, the interface unit) into the
    ``omega_max_drive`` cap the drivetrain governs on — radius-independent from
    the caller's point of view. (v1.2.3)
    """
    if wheel_radius_m <= 0.0:
        raise ValueError(f"wheel_radius_m must be positive, got {wheel_radius_m}")
    return float(top_speed_mps) / float(wheel_radius_m)
