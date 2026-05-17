"""Thin convenience helpers for the most repetitive Genesis-side boilerplate
when setting up a vehicle scene.

Each helper returns raw Genesis / SDK objects — the caller still owns
``scene.build()`` and ``scene.step()``. These helpers do NOT encapsulate
Genesis; they just bundle the canonical wheel-raycaster + URDF-entity
boilerplate into a single call. Use the core API directly when you need
viewer setup, custom materials, terrain, multi-agent scenes, or any other
Genesis feature these helpers don't expose.

For a fully encapsulated CARLA-/Gym-style API, see (future) modules under
``genesis_vehicle.adapters/``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import genesis as gs

from .raycast import WheelRayPattern
from .urdf import parse_urdf


def make_wheel_raycaster(
    scene: Any,
    entity: Any,
    urdf_path: str,
    *,
    max_range: float = 20.0,
    min_range: float = 0.0,
    return_world_frame: bool = True,
) -> Any:
    """Add a wheel raycaster sensor to ``scene`` using wheel positions parsed
    from ``urdf_path``. Returns the sensor handle.

    Equivalent to:

        parsed = parse_urdf(urdf_path)
        sensor = scene.add_sensor(gs.sensors.Raycaster(
            pattern=WheelRayPattern([w.position for w in parsed.wheels]),
            entity_idx=entity.idx,
            max_range=max_range, min_range=min_range,
            return_world_frame=return_world_frame,
        ))
    """
    parsed = parse_urdf(urdf_path)
    return scene.add_sensor(gs.sensors.Raycaster(
        pattern=WheelRayPattern([w.position for w in parsed.wheels]),
        entity_idx=entity.idx,
        max_range=max_range,
        min_range=min_range,
        return_world_frame=return_world_frame,
    ))


def add_vehicle(
    scene: Any,
    urdf_path: str,
    preset_fn: Optional[Callable[..., Any]] = None,
    *,
    pos: tuple[float, float, float] = (0.0, 0.0, 1.5),
    material: Any = None,
    raycaster_max_range: float = 20.0,
    stability: str = "control",
) -> tuple[Any, Any, Optional[Any]]:
    """Add a URDF vehicle entity + wheel raycaster to ``scene``.

    If ``preset_fn`` is given (e.g. ``car_4w_rwd_ackermann``), the matching
    ``VehicleConfig`` is also built. You still call ``scene.build()`` and
    construct ``VehiclePhysics`` yourself afterwards — so other entities,
    cameras, or sensors can be added between ``add_vehicle`` and ``build``.

    Returns ``(car_entity, sensor, cfg_or_None)``.

    Usage:

        car, sensor, cfg = add_vehicle(scene, URDF, car_4w_rwd_ackermann)
        scene.build(n_envs=1)
        physics = VehiclePhysics(scene, car, sensor, cfg, n_envs=1)
    """
    morph = gs.morphs.URDF(file=urdf_path, pos=pos)
    if material is None:
        car = scene.add_entity(morph)
    else:
        car = scene.add_entity(morph, material=material)
    sensor = make_wheel_raycaster(
        scene, car, urdf_path, max_range=raycaster_max_range,
    )
    cfg = preset_fn(urdf_path, stability=stability) if preset_fn is not None else None
    return car, sensor, cfg
