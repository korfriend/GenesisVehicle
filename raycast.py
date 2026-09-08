"""WheelRayPattern: a single canonical Genesis RaycastPattern for ray-wheels.

Replaces the legacy per-variant raycast implementations.
Each ray starts ``RAY_UP_OFFSET`` above a chassis-local wheel position and
points down (-z body); :func:`read_distances` subtracts the offset back out
so consumers see attachment-relative distances (the "high-cast" scheme —
see the constant's docstring for why).
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch
import genesis as gs
from genesis.options.sensors.raycaster import RaycastPattern

_logger = logging.getLogger(__name__)


#: High-cast margin (m): rays start this far ABOVE the wheel attachment
#: point (chassis-local +z) and the read layer subtracts it back out, so
#: the reported distance stays attachment-relative. Why: with the origin AT
#: the attachment point, a hard impact that bottoms the suspension can sink
#: the chassis far enough that the origins go BELOW the ground — the rays
#: then miss, the air mask kills N, and the vehicle rests on its chassis
#: collision box forever (a stable "buried" equilibrium; v1.1.16 fix).
#: With the high-cast origin the ray still sees the ground from above in
#: that state, the distance goes small/NEGATIVE, compression maxes out, and
#: N pushes the vehicle back onto its wheels. Keep this below any overhead
#: structure you expect within the margin above the wheel (tunnel ceilings).
RAY_UP_OFFSET = 1.0

#: Distances at or beyond this are treated as ray MISSES by the SDK
#: (air mask, visual air pose). The offset is NOT subtracted from misses so
#: the sentinel keeps its magnitude.
RAY_MISS_THRESHOLD = 19.9


class WheelRayPattern(RaycastPattern):
    """One downward ray (-z body) per wheel position (chassis-local).

    Rays are HIGH-CAST: each origin sits ``up_offset`` metres above the
    wheel attachment point (see :data:`RAY_UP_OFFSET`); pair with
    :func:`read_distances`, which subtracts the offset so consumers see
    attachment-relative distances (negative = ground above the attachment
    point, i.e. the chassis has sunk past it)."""

    def __init__(self, positions: Iterable[tuple[float, float, float]],
                 up_offset: float = RAY_UP_OFFSET):
        self._positions = [tuple(p) for p in positions]
        self.up_offset = float(up_offset)
        super().__init__()

    @classmethod
    def from_config(cls, resolved: "object") -> "WheelRayPattern":  # ResolvedConfig
        """Build from a ResolvedConfig (avoids importing it here to dodge cycles)."""
        return cls([w.position for w in resolved.wheels])

    def _get_return_shape(self) -> tuple[int]:
        return (len(self._positions),)

    def compute_ray_dirs(self) -> None:
        self._ray_dirs[:] = torch.tensor(
            [0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float
        )

    def compute_ray_starts(self) -> None:
        for i, (x, y, z) in enumerate(self._positions):
            self._ray_starts[i] = torch.tensor(
                [x, y, z + self.up_offset], device=gs.device, dtype=gs.tc_float
            )


def single_scene_up_offset(urdf_path: str, wheel_positions,
                           name: str = "vehicle") -> float:
    """The high-cast offset a SINGLE-SCENE vehicle can safely use.

    In ``single_scene`` the wheel rays are cast in the SAME scene the chassis
    collides in, so an origin lifted the full :data:`RAY_UP_OFFSET` above the
    wheel attachment point can end up above the vehicle's own collision box —
    every ray then hits its own roof a few centimetres down, the pipeline reads
    that as maximum compression, and the vehicle launches (the reference car has
    only 0.30 m between the wheel attachment at z=0.30 and the chassis box
    bottom at z=0.60, so 1.0 m was 0.7 m too high). This caps the offset at the
    vehicle's own collision ceiling — see
    :func:`genesis_vehicle.urdf.self_collision_ceiling`.

    ``dual_scene`` raycasts a scene that holds no vehicle collision geometry, so
    it is unaffected and keeps the full offset.

    A URDF whose collision geometry cannot be resolved (an unreadable mesh, a
    ``package://`` URI) reports no ceiling and keeps the full offset, which is
    the pre-v1.5.0 behaviour; run that vehicle in ``dual_scene`` if its rays
    self-hit."""
    from .urdf import self_collision_ceiling
    try:
        ceiling = self_collision_ceiling(urdf_path, wheel_positions)
    except Exception:
        _logger.debug("could not measure the self-collision ceiling for %r",
                      name, exc_info=True)
        return RAY_UP_OFFSET
    if ceiling is None or ceiling >= RAY_UP_OFFSET:
        return RAY_UP_OFFSET
    _logger.info(
        "[genesis_vehicle:single-scene] %s: wheel-ray high-cast offset capped at "
        "%.3f m (default %.2f m) — the vehicle's own collision geometry starts "
        "%.3f m above the wheel attachment point and the rays are cast in the "
        "same scene it collides in. A deep suspension bottom-out recovers less "
        "readily than in dual_scene; use raycast_mode='dual_scene' (the default) "
        "if that matters.", name, ceiling, RAY_UP_OFFSET, ceiling)
    return ceiling


#: Attribute :func:`set_sensor_up_offset` stamps on a Genesis raycaster so
#: :func:`read_distances` can recover the offset its rays were built with.
_UP_OFFSET_ATTR = "_gv_up_offset"


def set_sensor_up_offset(sensor: "object", up_offset: float) -> None:
    """Record on the sensor which high-cast offset its rays were built with.

    The offset is per vehicle since v1.5.0 (``single_scene`` caps it at the
    vehicle's own collision ceiling), and :func:`read_distances` has to subtract
    back exactly what :class:`WheelRayPattern` added. Stamping the sensor keeps
    that pairing with the object it belongs to instead of threading a parameter
    through every call site."""
    setattr(sensor, _UP_OFFSET_ATTR, float(up_offset))


def sensor_up_offset(sensor: "object") -> float:
    """The high-cast offset of ``sensor``'s rays; :data:`RAY_UP_OFFSET` if it was
    never stamped (a sensor built outside :class:`VehicleScene`)."""
    return float(getattr(sensor, _UP_OFFSET_ATTR, RAY_UP_OFFSET))


def read_distances(sensor: "object", n_envs: int,
                   up_offset: "float | None" = None) -> torch.Tensor:
    """Normalise sensor.read().distances to (n_envs, n_wheels), corrected
    back to ATTACHMENT-RELATIVE distances.

    Genesis returns shape (n_wheels,) when n_envs == 1 and (n_envs, n_wheels)
    otherwise; this collapses the two cases into one for the pipeline, and
    subtracts the high-cast offset from HITS (misses keep their sentinel value
    >= :data:`RAY_MISS_THRESHOLD` untouched). A hit closer than the offset
    yields a NEGATIVE distance — the ground is above the attachment point (deep
    over-compression); the pipeline turns that into maximum compression so the
    suspension can recover the vehicle.

    ``up_offset`` defaults to whatever the sensor was stamped with by
    :func:`set_sensor_up_offset` (:data:`RAY_UP_OFFSET` if it never was), so a
    ``single_scene`` vehicle whose offset is capped reads back correctly without
    the caller knowing."""
    d = sensor.read().distances
    if d.dim() == 1:
        d = d.unsqueeze(0)
    if up_offset is None:
        up_offset = sensor_up_offset(sensor)
    if up_offset:
        d = torch.where(d >= RAY_MISS_THRESHOLD, d, d - up_offset)
    return d
