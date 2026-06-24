"""VehicleScene: the unified, high-level entry point of the SDK.

A ``VehicleScene`` owns the Genesis scene(s), the registered vehicles, and the
static/dynamic bodies, and drives the per-step loop — so the caller never
touches ``gs.init`` / ``scene.build`` / ``scene.step`` / sensor reads directly.

Two raycast modes (``raycast_mode=``):

- ``"raywheel"`` (default) — the ray-wheel-dedicated raycast optimization. The
  terrain is raycast in a SEPARATE scene where it lives as a *kinematic* body,
  so its BVH is built **once** and never re-fit (``maybe_static``), and the cast
  is shared across batch envs. Physics/collision (incl. rollover) run in the
  main scene with the terrain as a *rigid* body. Each step the vehicle's chassis
  pose is mirrored onto a lightweight rigid, fixed, collision-free proxy in the
  raycast scene; ``raycast_scene.step()`` refreshes the ray origins and re-casts
  against the static BVH; the distances are fed into the main-scene physics via
  :meth:`VehiclePhysics.step(distances=...) <genesis_vehicle.core.VehiclePhysics.step>`.

- ``"inline"`` — classic: one scene, each vehicle owns a wheel raycaster that
  casts against everything (terrain + vehicle). The raycast BVH is **re-fit
  every step** because the vehicle moves, so the per-step cost scales with
  terrain face count.

Why ``"raywheel"`` is the default (see ``docs/two-scene-raycast.md``): complex
terrain is the common case, and keeping the terrain BVH static stops the wheel
raycast from re-fitting it each step. The win is small at ``n_envs=1`` on GPU
(~1–1.3x; ~1.5–5.5x on CPU) but **grows strongly with L3 batch size** because the
static BVH is shared across envs (split is ~flat in ``n_envs``, inline re-fits
per env): GPU full-step **1.03x @1, 1.57x @64, 3.40x @256 envs** on a 51k-face
terrain. ``"raywheel"`` is also more *accurate* on non-convex mesh terrain (a
rigid mesh is convexified for collision, so an inline rigid-mesh raycast hits the
convex bulge while the kinematic raycast hits the true surface). It costs ~2x
terrain memory and is marginally slower than ``"inline"`` only on small/flat
terrain at ``n_envs=1`` — use ``raycast_mode="inline"`` there.

The legacy names ``"split"`` / ``"single"`` are accepted as aliases for
``"raywheel"`` / ``"inline"``.

Scope: one or more vehicles (L2 — each gets its own proxy + sensor in the
raycast scene; they still collide in the main scene), L3 (``n_envs >= 1``)
batching, static *terrain/mesh* raycast targets (``add_static`` /
``add_static_terrain``), and dynamic raycast obstacles the wheels must sense
(``add_obstacle`` — a synced rigid mirror in the raycast scene's rigid solver,
so only its small BVH re-fits while the terrain stays static).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import genesis as gs

from .core import VehiclePhysics
from .inputs import VehicleInputs
from .raycast import WheelRayPattern, read_distances
from .urdf import parse_urdf

# gs.init is process-global and may be called at most once. Track it so several
# VehicleScenes (or a user who already called gs.init) don't double-initialize.
_GENESIS_BACKEND: Optional[str] = None


def _ensure_genesis(backend: str) -> None:
    global _GENESIS_BACKEND
    # Already initialized (by the user, or a previous VehicleScene)? Reuse it.
    # Use Genesis's own flag rather than swallowing gs.init's exception, so a
    # REAL init failure (e.g. missing libcuda for the GPU backend) propagates
    # clearly instead of being mistaken for "already initialized".
    if getattr(gs, "_initialized", False) or _GENESIS_BACKEND is not None:
        _GENESIS_BACKEND = _GENESIS_BACKEND or backend
        return
    be = gs.gpu if backend == "gpu" else gs.cpu
    gs.init(backend=be, logging_level="warning")
    _GENESIS_BACKEND = backend


def _clone_morph(morph, **updates):
    """Copy a Genesis morph (pydantic model) with fields updated, so the same
    geometry can be added to two scenes with different flags (fixed/collision).
    Only fields the morph actually has are updated; falls back to the original
    if it cannot be copied."""
    try:
        valid = {k: v for k, v in updates.items()
                 if k in type(morph).model_fields}
        return morph.model_copy(update=valid, deep=True)
    except Exception:
        return morph


def _add_kwargs(material=None, surface=None, vis_mode=None):
    """Build the kwargs for ``scene.add_entity`` omitting any that are None, so
    callers can pass-through optional material / surface / vis_mode uniformly."""
    kw = {}
    if material is not None:
        kw["material"] = material
    if surface is not None:
        kw["surface"] = surface
    if vis_mode is not None:
        kw["vis_mode"] = vis_mode
    return kw


@dataclass
class StaticBody:
    """Handle for a registered static body. ``is_static``/``has_collision``/
    ``has_raycast`` are the "static physics" properties that drive routing:
    a collision body lands in the main scene (rigid); a raycast body lands in
    the raycast scene (kinematic, split mode) or is the same rigid body (single
    mode)."""
    name: str
    has_collision: bool
    has_raycast: bool
    is_static: bool = True
    entity_main: Any = None       # rigid collision entity (main scene)
    entity_raycast: Any = None    # kinematic raycast entity (raycast scene, split)


@dataclass
class Obstacle:
    """Handle for a registered moving body the wheels may need to *sense*
    (raycast), not just collide with (returned by :meth:`VehicleScene.add_dynamic`).
    Unlike a StaticBody its ``entity_raycast`` is re-synced every step, so it
    tracks the body as it moves.

    In dual_scene mode ``entity_raycast`` is a rigid, fixed, collision-enabled
    body in the raycast scene's RIGID solver — a separate BVH context from the
    kinematic terrain, so re-fitting it each step does NOT disturb the static
    terrain BVH. In single_scene mode it is None (the main entity is the target).
    """
    name: str
    is_dynamic: bool
    has_raycast: bool
    entity_main: Any = None       # rigid body in the main scene (physics)
    entity_raycast: Any = None    # synced raycast target in the raycast scene

    def set_pose(self, pos=None, quat=None, *, zero_velocity=True):
        """Move a user-controlled body in the main scene (the raycast mirror
        follows on the next ``VehicleScene.step``)."""
        if pos is not None:
            self.entity_main.set_pos(pos, zero_velocity=zero_velocity)
        if quat is not None:
            self.entity_main.set_quat(quat, zero_velocity=zero_velocity)


class Vehicle:
    """Handle for a registered vehicle. Set inputs each step; read pose anytime.

    Wraps the main-scene rigid entity, its :class:`VehiclePhysics`, and (split
    mode) the raycast-scene proxy + sensor."""

    def __init__(self, name: str, urdf_path: str, cfg: Any,
                 wheel_positions: list, pos, quat, material) -> None:
        self.name = name
        self.urdf_path = urdf_path
        self.cfg = cfg
        self._wheel_positions = wheel_positions
        self._pos = pos
        self._quat = quat
        self._material = material
        # filled during VehicleScene.add_vehicle / build
        self.entity: Any = None        # main-scene rigid URDF entity
        self.physics: Optional[VehiclePhysics] = None
        self.sensor: Any = None        # wheel raycaster (main in single, raycast-scene in split)
        self.proxy: Any = None         # raycast-scene pose carrier (split only)
        self._inputs = VehicleInputs(throttle=0.0, brake=0.0, steer=0.0)
        self._two_scene = False
        self._n_envs = 1

    # ---- per-step input ----
    def set_inputs(self, throttle: float = 0.0, brake: float = 0.0,
                   steer: float = 0.0) -> "Vehicle":
        self._inputs = VehicleInputs(throttle=throttle, brake=brake, steer=steer)
        return self

    # ---- pose accessors (main scene = physical truth) ----
    def get_pos(self):
        return self.entity.get_pos()

    def get_quat(self):
        return self.entity.get_quat()

    def get_vel(self):
        return self.entity.get_vel()

    def get_ang(self):
        return self.entity.get_ang()

    @property
    def distances(self):
        """Last wheel-ground distances used by the physics step."""
        return None if self.physics is None else self.physics.last_distances

    # ---- internals (called by VehicleScene) ----
    def _sync_proxy(self) -> None:
        """Mirror the main chassis base pose onto the raycast-scene proxy.
        ``set_pos``/``set_quat`` run FK, so the wheel ray origins update without
        stepping the raycast scene."""
        p = self.entity.get_pos()
        q = self.entity.get_quat()
        if p.dim() > 1:          # (n_envs, 3) — single-env keeps a leading dim
            if p.shape[0] == 1:
                p, q = p[0], q[0]
        self.proxy.set_pos(p, relative=False)
        self.proxy.set_quat(q, relative=False)



class VehicleScene:
    """Unified vehicle simulation scene — the center of the SDK API."""

    def __init__(
        self,
        *,
        n_envs: int = 1,
        dt: float = 1.0 / 200.0,
        backend: str = "gpu",
        raycast_mode: str = "dual_scene",
        gravity: tuple = (0.0, 0.0, -9.81),
        substeps: int = 4,
        sim_options: Any = None,
        rigid_options: Any = None,
        vis_options: Any = None,
        show_viewer: bool = False,
        init_genesis: bool = True,
    ) -> None:
        # Back-compat aliases for the pre-rename names.
        raycast_mode = {"raywheel": "dual_scene", "split": "dual_scene",
                        "inline": "single_scene", "single": "single_scene"}.get(
                            raycast_mode, raycast_mode)
        if raycast_mode not in ("dual_scene", "single_scene"):
            raise ValueError(
                f"raycast_mode must be 'dual_scene' (default) or 'single_scene', "
                f"got {raycast_mode!r}")
        if n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        # L3 batching: both scenes build with the same n_envs; the proxy and the
        # wheel sensor are batched, so _sync_proxy mirrors the full (n_envs, ...)
        # chassis pose and read_distances returns (n_envs, n_wheels).
        self.n_envs = n_envs
        self.dt = dt
        self.backend = backend
        self.raycast_mode = raycast_mode
        self._two_scene = raycast_mode == "dual_scene"
        self._built = False

        if init_genesis:
            _ensure_genesis(backend)

        _sim = sim_options or gs.options.SimOptions(dt=dt, substeps=substeps, gravity=gravity)
        _rigid = rigid_options or gs.options.RigidOptions(dt=dt, enable_collision=True)
        _vis = vis_options or gs.options.VisOptions()

        self.main_scene = gs.Scene(sim_options=_sim, rigid_options=_rigid,
                                   vis_options=_vis, show_viewer=show_viewer)
        # The raycast scene never advances dynamics; reuse the same dt for sanity.
        self.raycast_scene = (
            gs.Scene(sim_options=gs.options.SimOptions(dt=dt, substeps=1, gravity=(0, 0, 0)),
                     show_viewer=False)
            if self._two_scene else None
        )

        self._vehicles: list[Vehicle] = []
        self._statics: list[StaticBody] = []
        self._obstacles: list[Obstacle] = []
        self._pending_mass: list = []   # (entity, mass) applied after build()

    # -----------------------------------------------------------------
    # Registration (before build)
    # -----------------------------------------------------------------

    def add_static(
        self,
        *,
        morph: Any = None,
        raycast_morph: Any = None,
        collision_morph: Any = None,
        collision: bool = True,
        raycast: bool = True,
        material: Any = None,
        surface: Any = None,
        vis_mode: Any = None,
        name: Optional[str] = None,
    ) -> StaticBody:
        """Register a static body (terrain / mesh / plane / primitive).

        Provide one ``morph`` for both roles, or split into ``raycast_morph``
        (detailed surface the wheels cast against) and ``collision_morph``
        (coarse/convex body the chassis collides with). Splitting is recommended
        for high-poly or non-convex meshes: a rigid mesh is auto-convexified for
        collision (so a rigid-mesh raycast hits the convex bulge, not the true
        surface), whereas the kinematic raycast surface stays exact.

        VehicleScene owns the scene routing: the collision body lands in the main
        scene (rigid) and, in raywheel mode, a kinematic raycast mirror lands in
        the raycast scene. Callers never touch the underlying scenes.
        """
        self._require_not_built()
        if morph is None and raycast_morph is None and collision_morph is None:
            raise ValueError("add_static: provide morph (or raycast_morph/collision_morph).")
        rc_morph = raycast_morph or morph
        col_morph = collision_morph or morph
        name = name or f"static_{len(self._statics)}"

        body = StaticBody(name=name, has_collision=bool(collision), has_raycast=bool(raycast))

        if collision and col_morph is not None:
            mat = material if material is not None else gs.materials.Rigid()
            body.entity_main = self.main_scene.add_entity(
                col_morph, **_add_kwargs(mat, surface, vis_mode))

        if raycast and rc_morph is not None:
            if self._two_scene:
                # Kinematic visual-raycast body in the raycast scene → static BVH.
                body.entity_raycast = self.raycast_scene.add_entity(
                    rc_morph, **_add_kwargs(
                        gs.materials.Kinematic(use_visual_raycasting=True),
                        surface, vis_mode))
            else:
                # single mode: the rigid collision body IS the raycast target.
                if body.entity_main is None:
                    mat = material if material is not None else gs.materials.Rigid()
                    body.entity_main = self.main_scene.add_entity(
                        rc_morph, **_add_kwargs(mat, surface, vis_mode))
                body.entity_raycast = body.entity_main

        self._statics.append(body)
        return body

    def add_static_terrain(self, morph: Any, **kwargs) -> StaticBody:
        """Convenience alias for a heightfield/terrain morph. Same as
        :meth:`add_static` with ``morph=``."""
        return self.add_static(morph=morph, **kwargs)

    def add_ground_plane(self, *, friction: float = 0.85) -> StaticBody:
        """Convenience: an infinite flat ground (raycast + collision)."""
        return self.add_static(morph=gs.morphs.Plane(),
                               material=gs.materials.Rigid(friction=friction),
                               name="ground")

    def add_dynamic(
        self,
        morph: Any,
        *,
        physics: bool = True,
        raycast: bool = True,
        material: Any = None,
        surface: Any = None,
        vis_mode: Any = None,
        mass: Optional[float] = None,
        name: Optional[str] = None,
    ) -> Obstacle:
        """Register a MOVING body the wheels may need to **sense** (drive onto /
        over), not just collide with — e.g. a ramp, curb, or moving platform.
        Use :meth:`add_static` for bodies that never move.

        ``physics=True`` (default): a free rigid body that moves under physics.
        ``physics=False``: a fixed body you teleport yourself with
        ``handle.set_pose(...)`` (e.g. an externally / UE-driven obstacle).

        With ``raycast=True`` in dual_scene mode a synced rigid raycast mirror is
        added to the raycast scene (re-synced every ``step``; re-fits only its own
        small BVH, leaving the terrain static). In single_scene mode the one
        main-scene rigid body is already the raycast target. VehicleScene owns the
        routing — the caller never touches a scene.
        """
        self._require_not_built()
        name = name or f"dynamic_{len(self._obstacles)}"
        mat = material if material is not None else gs.materials.Rigid()

        obs = Obstacle(name=name, is_dynamic=bool(physics),
                       has_raycast=bool(raycast))
        main_morph = _clone_morph(morph, fixed=not physics)
        obs.entity_main = self.main_scene.add_entity(
            main_morph, **_add_kwargs(mat, surface, vis_mode))
        if mass is not None:
            self._pending_mass.append((obs.entity_main, float(mass)))

        if raycast and self._two_scene:
            # Rigid + fixed + collision mirror in the raycast scene's RIGID
            # solver (a separate BVH context from the kinematic terrain), so
            # re-syncing it each step re-fits only this small body.
            mirror_morph = _clone_morph(morph, fixed=True, collision=True)
            obs.entity_raycast = self.raycast_scene.add_entity(
                mirror_morph, **_add_kwargs(gs.materials.Rigid(), surface, vis_mode))

        self._obstacles.append(obs)
        return obs

    def add_vehicle(
        self,
        urdf_path: str,
        preset: Optional[Callable[..., Any]] = None,
        *,
        pos: tuple = (0.0, 0.0, 1.0),
        quat: Optional[tuple] = None,
        material: Any = None,
        surface: Any = None,
        vis_mode: Any = None,
        stability: str = "control",
        name: Optional[str] = None,
        raycaster_max_range: float = 20.0,
        cfg: Any = None,
        morph: Any = None,
    ) -> Vehicle:
        """Register a vehicle. VehicleScene builds the rigid entity in the main
        scene and, in dual_scene mode, the kinematic proxy + wheel sensor in the
        raycast scene — the caller never touches a scene.

        cfg: ``preset`` (a preset fn → cfg) OR a pre-built ``cfg``.
        entity geometry: pass ``morph`` (e.g. ``gs.morphs.URDF(stripped_path,…)``,
        built internally with ``material`` / ``surface`` / ``vis_mode``), or let
        it be built from ``urdf_path``. ``urdf_path`` always gives the wheel
        positions.
        """
        self._require_not_built()
        if cfg is None and preset is None:
            raise ValueError("add_vehicle: pass preset=<fn> or cfg=<VehicleConfig>.")
        name = name or f"vehicle_{len(self._vehicles)}"
        if cfg is None:
            cfg = preset(urdf_path, stability=stability)
        parsed = parse_urdf(urdf_path)
        wheel_positions = [w.position for w in parsed.wheels]

        veh = Vehicle(name, urdf_path, cfg, wheel_positions, pos, quat, material)
        veh._two_scene = self._two_scene
        veh._n_envs = self.n_envs

        if morph is None:
            morph_kw = dict(file=urdf_path, pos=pos)
            if quat is not None:
                morph_kw["quat"] = quat
            morph = gs.morphs.URDF(**morph_kw)
        veh.entity = self.main_scene.add_entity(
            morph, **_add_kwargs(material, surface, vis_mode))

        if self._two_scene:
            # raycast scene: lightweight pose-carrier proxy + wheel sensor.
            # The proxy MUST be a RIGID, fixed-base, collision-free body, NOT a
            # kinematic one. Why: (a) collision=False → no collision faces → not
            # a raycast target → no self-hit; (b) it lives in the rigid solver,
            # SEPARATE from the kinematic terrain — so teleporting it via set_pos
            # fires a GEOMETRY change on the rigid solver only, leaving the
            # kinematic-terrain raycast BVH untouched (it stays `maybe_static` →
            # built once, never re-fit). A kinematic proxy shares the terrain's
            # solver and its set_pos would invalidate the terrain BVH every step
            # (measured ~6x slower). The ray origins still track the vehicle
            # because raycast_scene.step() refreshes them each frame from the
            # proxy's link pose. See docs/two-scene-raycast.md.
            veh.proxy = self.raycast_scene.add_entity(
                gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=pos,
                              fixed=True, collision=False),
                material=gs.materials.Rigid())
            veh.sensor = self.raycast_scene.add_sensor(gs.sensors.Raycaster(
                pattern=WheelRayPattern(wheel_positions),
                entity_idx=veh.proxy.idx,
                max_range=raycaster_max_range, min_range=0.0, return_world_frame=True))
        else:
            veh.sensor = self.main_scene.add_sensor(gs.sensors.Raycaster(
                pattern=WheelRayPattern(wheel_positions),
                entity_idx=veh.entity.idx,
                max_range=raycaster_max_range, min_range=0.0, return_world_frame=True))

        self._vehicles.append(veh)
        return veh

    # -----------------------------------------------------------------
    # Build / step
    # -----------------------------------------------------------------

    def build(self) -> None:
        """Build both scenes and wire up per-vehicle physics."""
        if self._built:
            return
        self.main_scene.build(n_envs=self.n_envs)
        if self._two_scene:
            self.raycast_scene.build(n_envs=self.n_envs)
            # Populate the raycast sensors once so the static BVH is built and
            # the first read() returns valid data (Genesis sensors are empty
            # before the first step). The kinematic terrain/proxy don't move, so
            # this single step is cheap and never repeated.
            self.raycast_scene.step()

        for veh in self._vehicles:
            sensor = None if self._two_scene else veh.sensor
            veh.physics = VehiclePhysics(
                self.main_scene, veh.entity, sensor, veh.cfg, n_envs=self.n_envs)

        # Apply any per-obstacle mass overrides now that entities are built.
        for entity, mass in self._pending_mass:
            entity.set_mass(mass)
        self._built = True

    def _measure_distances(self) -> dict:
        """Internal: return ``{vehicle: wheel-ground distances}`` for the step.

        Raywheel: mirror each chassis pose onto its proxy + sync obstacle
        mirrors, step the raycast scene ONCE (re-cast against the static BVH),
        read each sensor → ``(n_envs, n_wheels)``. Inline: ``{vehicle: None}``.

        NOT a getter — it re-syncs poses and CASTS (advances the raycast scene),
        so it must run exactly once per :meth:`step`. ``step`` is the only caller.
        """
        self._require_built()
        if not self._two_scene:
            return {veh: None for veh in self._vehicles}
        for veh in self._vehicles:
            veh._sync_proxy()
        for obs in self._obstacles:
            if obs.entity_raycast is not None:
                self._sync_obstacle(obs)
        self.raycast_scene.step()
        return {veh: read_distances(veh.sensor, self.n_envs)
                for veh in self._vehicles}

    def step(self) -> None:
        """One simulation step.

        Raywheel mode: sync proxies/obstacle mirrors + re-cast the static raycast
        scene, feed each vehicle's main-scene physics, advance the main scene.
        Inline mode: each vehicle reads its own sensor.
        """
        self._require_built()
        dists = self._measure_distances()
        for veh in self._vehicles:
            veh.physics.step(veh._inputs, distances=dists[veh])
        self.main_scene.step()

    def reset(self) -> None:
        """Reset per-vehicle physics state and re-sync proxies. (Full scene
        re-randomization is out of scope for this first cut.)"""
        self._require_built()
        for veh in self._vehicles:
            if veh.physics is not None:
                veh.physics.reset()
            if self._two_scene:
                veh._sync_proxy()

    # ---- accessors ----
    @property
    def vehicles(self) -> list:
        return list(self._vehicles)

    @property
    def statics(self) -> list:
        return list(self._statics)

    @property
    def obstacles(self) -> list:
        return list(self._obstacles)

    # ---- internals ----
    def _sync_obstacle(self, obs: "Obstacle") -> None:
        """Mirror an obstacle's main-scene pose onto its raycast-scene mirror so
        the wheels' rays see it at its current position."""
        p = obs.entity_main.get_pos()
        q = obs.entity_main.get_quat()
        if p.dim() > 1 and p.shape[0] == 1:
            p, q = p[0], q[0]
        obs.entity_raycast.set_pos(p, relative=False)
        obs.entity_raycast.set_quat(q, relative=False)

    def _require_not_built(self) -> None:
        if self._built:
            raise RuntimeError("VehicleScene is already built; register bodies before build().")

    def _require_built(self) -> None:
        if not self._built:
            raise RuntimeError("VehicleScene.build() must be called before step()/reset().")
