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

import os
from typing import Any, Callable, Optional

import genesis as gs

from .raycast import (
    WheelRayPattern, fan_height_offsets, fan_ray_positions,
    set_sensor_fan, set_sensor_miss_value, set_sensor_up_offset,
    single_scene_up_offset,
)
from .urdf import parse_urdf


def make_wheel_raycaster(
    scene: Any,
    entity: Any,
    urdf_path: str,
    *,
    max_range: float = 20.0,
    min_range: float = 0.0,
    return_world_frame: bool = True,
    up_offset: Optional[float] = None,
    wheel_contact: str = "point",
    contact_samples: int = 9,
) -> Any:
    """Add a wheel raycaster sensor to ``scene`` using wheel positions parsed
    from ``urdf_path``. Returns the sensor handle.

    This is inherently a SINGLE-SCENE raycaster: the sensor is anchored to the
    vehicle in the very scene the vehicle collides in, so the high-cast ray
    origins are capped at the vehicle's own collision ceiling
    (:func:`genesis_vehicle.raycast.single_scene_up_offset`) — without the cap
    the rays hit the vehicle's own chassis box and it launches. Pass
    ``up_offset`` to override the cap. ``read_distances`` recovers the offset
    off the sensor, so callers need not pass it on.

    ``wheel_contact`` selects the contact model — ``"point"`` (DEFAULT, one ray
    per wheel, unchanged) or ``"swept_envelope"`` (``contact_samples`` rays per
    wheel, collapsed back to one distance per wheel by ``read_distances``; see
    :class:`genesis_vehicle.raycast.WheelRayPattern`, and note that M is a
    discretisation count, not an accuracy dial). The fan's radii come from the
    URDF wheel geometry; build the pattern yourself from a resolved config if a
    ``WheelConfig`` overrides the radius. The offset cap is measured against
    EVERY fan ray, not the wheel centres.

    Equivalent to:

        parsed = parse_urdf(urdf_path)
        wheels = [w.position for w in parsed.wheels]
        offset = single_scene_up_offset(urdf_path, wheels)
        sensor = scene.add_sensor(gs.sensors.Raycaster(
            pattern=WheelRayPattern(wheels, up_offset=offset),
            entity_idx=entity.idx,
            max_range=max_range, min_range=min_range,
            return_world_frame=return_world_frame,
        ))
        set_sensor_up_offset(sensor, offset)
        set_sensor_miss_value(sensor)
    """
    parsed = parse_urdf(urdf_path)
    wheels = [w.position for w in parsed.wheels]
    fan_m = _fan_samples(wheel_contact, contact_samples)
    fan_radii = _urdf_wheel_radii(parsed, urdf_path) if fan_m > 1 else None
    fan_c = fan_height_offsets(fan_radii, fan_m) if fan_m > 1 else None
    if up_offset is None:
        # Cap against EVERY ray: an outer fan sample can sit under a low
        # overhang the wheel centre clears (see fan_ray_positions).
        ray_positions = (fan_ray_positions(wheels, fan_radii, fan_m)
                         if fan_m > 1 else wheels)
        up_offset = single_scene_up_offset(urdf_path, ray_positions,
                                           name=os.path.basename(urdf_path))
    sensor = scene.add_sensor(gs.sensors.Raycaster(
        pattern=WheelRayPattern(wheels, up_offset=up_offset,
                                fan_samples=fan_m, wheel_radii=fan_radii),
        entity_idx=entity.idx,
        max_range=max_range,
        min_range=min_range,
        return_world_frame=return_world_frame,
    ))
    set_sensor_up_offset(sensor, up_offset)
    # Swept-envelope geometry, so read_distances can collapse the M axis with no
    # argument from the call site (a no-op for the default point contact).
    set_sensor_fan(sensor, fan_c)
    # Record (and validate) the distance this sensor reports for a ray that hit
    # NOTHING — read_distances and the grounded predicate test against it
    # instead of a hardcoded threshold. Raises if it is below max_range.
    set_sensor_miss_value(sensor)
    return sensor


def add_vehicle(
    scene: Any,
    urdf_path: str,
    preset_fn: Optional[Callable[..., Any]] = None,
    *,
    pos: tuple[float, float, float] = (0.0, 0.0, 1.5),
    material: Any = None,
    raycaster_max_range: float = 20.0,
    stability: str = "control",
    wheel_contact: str = "point",
    contact_samples: int = 9,
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
        wheel_contact=wheel_contact, contact_samples=contact_samples,
    )
    cfg = preset_fn(urdf_path, stability=stability) if preset_fn is not None else None
    return car, sensor, cfg


def _urdf_wheel_radii(parsed: Any, urdf_path: str) -> list:
    """Per-wheel radii off the parsed URDF, for the swept-envelope fan.

    ``make_wheel_raycaster`` sees no ``VehicleConfig``, so the URDF geometry is
    the only radius source it has; a ``WheelConfig(radius=...)`` override is NOT
    visible here. Use ``VehicleScene.add_vehicle(wheel_contact=...)`` (which
    resolves the config) when the two differ."""
    radii = []
    for w in parsed.wheels:
        if w.radius is None:
            raise ValueError(
                f"make_wheel_raycaster({os.path.basename(urdf_path)}): wheel "
                f"{w.name!r} has no radius in the URDF, so a swept-envelope fan "
                f"cannot be sized for it. Use wheel_contact='point', or build "
                f"the WheelRayPattern yourself with explicit wheel_radii.")
        radii.append(float(w.radius))
    return radii


def _fan_samples(wheel_contact: str, contact_samples: int) -> int:
    """Rays per wheel for a ``wheel_contact`` selection — shared with
    ``VehicleScene.add_vehicle`` so the two entry points cannot disagree about
    what ``"point"`` means."""
    from .vehicle_scene import _fan_samples as _impl
    return _impl(wheel_contact, contact_samples)
