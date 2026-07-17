"""WheelRayPattern: a single canonical Genesis RaycastPattern for ray-wheels.

Replaces the legacy per-variant raycast implementations.
Each ray starts ``RAY_UP_OFFSET`` above a chassis-local wheel position and
points down (-z body); :func:`read_distances` subtracts the offset back out
so consumers see attachment-relative distances (the "high-cast" scheme —
see the constant's docstring for why).
"""

from __future__ import annotations

from typing import Iterable

import torch
import genesis as gs
from genesis.options.sensors.raycaster import RaycastPattern


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


def read_distances(sensor: "object", n_envs: int,
                   up_offset: float = RAY_UP_OFFSET) -> torch.Tensor:
    """Normalise sensor.read().distances to (n_envs, n_wheels), corrected
    back to ATTACHMENT-RELATIVE distances.

    Genesis returns shape (n_wheels,) when n_envs == 1 and (n_envs, n_wheels)
    otherwise; this collapses the two cases into one for the pipeline, and
    subtracts the high-cast ``up_offset`` from HITS (misses keep their
    sentinel value >= :data:`RAY_MISS_THRESHOLD` untouched). A hit closer
    than the offset yields a NEGATIVE distance — the ground is above the
    attachment point (deep over-compression); the pipeline turns that into
    maximum compression so the suspension can recover the vehicle."""
    d = sensor.read().distances
    if d.dim() == 1:
        d = d.unsqueeze(0)
    if up_offset:
        d = torch.where(d >= RAY_MISS_THRESHOLD, d, d - up_offset)
    return d
