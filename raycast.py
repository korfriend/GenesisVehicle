"""WheelRayPattern: a single canonical Genesis RaycastPattern for ray-wheels.

Replaces the per-variant copies in HJW/car_raywheel.py and KDU/physics.py.
Each ray starts at a chassis-local wheel position and points down (-z body).
"""

from __future__ import annotations

from typing import Iterable

import torch
import genesis as gs
from genesis.options.sensors.raycaster import RaycastPattern


class WheelRayPattern(RaycastPattern):
    """One downward ray (-z body) per wheel position (chassis-local)."""

    def __init__(self, positions: Iterable[tuple[float, float, float]]):
        self._positions = [tuple(p) for p in positions]
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
                [x, y, z], device=gs.device, dtype=gs.tc_float
            )


def read_distances(sensor: "object", n_envs: int) -> torch.Tensor:
    """Normalise sensor.read().distances to (n_envs, n_wheels).

    Genesis returns shape (n_wheels,) when n_envs == 1 and (n_envs, n_wheels)
    otherwise; this collapses the two cases into one for the pipeline.
    """
    d = sensor.read().distances
    if d.dim() == 1:
        d = d.unsqueeze(0)
    return d
