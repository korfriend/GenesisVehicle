"""VehicleScene: the unified, high-level entry point of the SDK.

A ``VehicleScene`` owns the Genesis scene(s), the registered vehicles, and the
static/dynamic bodies, and drives the per-step loop — so the caller never
touches ``gs.init`` / ``scene.build`` / ``scene.step`` / sensor reads directly.

Two raycast modes (``raycast_mode=``):

- ``"dual_scene"`` (default) — the ray-wheel-dedicated raycast optimization. The
  terrain is raycast in a SEPARATE scene where it lives as a *kinematic* body,
  so its BVH is built **once** and never re-fit (``maybe_static``), and the cast
  is shared across batch envs. Physics/collision (incl. rollover) run in the
  main scene with the terrain as a *rigid* body. Each step the vehicle's chassis
  pose is mirrored onto a lightweight rigid, fixed, collision-free proxy in the
  raycast scene; ``raycast_scene.step()`` refreshes the ray origins and re-casts
  against the static BVH; the distances are fed into the main-scene physics via
  :meth:`VehiclePhysics.step(distances=...) <genesis_vehicle.core.VehiclePhysics.step>`.
  The raycast scene is **sensors-only**: it is never shown (``show_viewer=False``
  always) or rendered (stepped with ``update_visualizer=False``, no camera) —
  only the main scene takes a viewer / ``viewer_options``.

- ``"single_scene"`` — classic: one scene, each vehicle owns a wheel raycaster that
  casts against everything (terrain + vehicle). The raycast BVH is **re-fit
  every step** because the vehicle moves, so the per-step cost scales with
  terrain face count.

Why ``"dual_scene"`` is the default (see ``docs/two-scene-raycast.md``): complex
terrain is the common case, and keeping the terrain BVH static stops the wheel
raycast from re-fitting it each step. The win is small at ``n_envs=1`` on GPU
(~1–1.3x; ~1.5–5.5x on CPU) but **grows strongly with L3 batch size** because the
static BVH is shared across envs (dual_scene is ~flat in ``n_envs``, single_scene
re-fits per env): GPU full-step **1.03x @1, 1.57x @64, 3.40x @256 envs** on a 51k-face
terrain. ``"dual_scene"`` is also more *accurate* on non-convex mesh terrain (a
rigid mesh is convexified for collision, so a single_scene rigid-mesh raycast hits the
convex bulge while the kinematic raycast hits the true surface). It costs ~2x
terrain memory and is marginally slower than ``"single_scene"`` only on small/flat
terrain at ``n_envs=1`` — use ``raycast_mode="single_scene"`` there.

The legacy names ``"raywheel"`` / ``"inline"`` and ``"split"`` / ``"single"`` are
accepted as aliases for ``"dual_scene"`` / ``"single_scene"``.

Scope: one or more vehicles (L2 — each gets its own proxy + sensor in the
raycast scene; they still collide in the main scene), L3 (``n_envs >= 1``)
batching, static *terrain/mesh* raycast targets (``add_static``, always a
raycast target), and moving bodies (``add_dynamic``; ``raycast=True`` opt-in adds
a synced rigid mirror in the raycast scene's rigid solver, so only its small BVH
re-fits while the terrain stays static).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import genesis as gs

from .core import VehiclePhysics
from .inputs import VehicleInputs
from .raycast import WheelRayPattern, read_distances
from .urdf import parse_urdf

_logger = logging.getLogger("genesis_vehicle.vehicle_scene")

# Genesis primitive morph types (closed-form colliders) whose raycast BVH is
# trivially cheap — anything outside this set is a mesh/heightfield whose BVH
# re-fit cost scales with face count (see add_dynamic's guard).
_PRIMITIVE_MORPHS = frozenset({"Box", "Sphere", "Cylinder", "Capsule", "Plane"})

# Above this face count, a NON-CONVEX mesh used as a rigid collider/raycast
# target is refused (see _guard_collision_mesh): the cost — the SDF/collision
# build AND, in single_scene, the per-step wheel-raycaster BVH re-fit over every
# face — spikes memory/compute and can crash the process; under WSL it takes the
# whole VM down.
_MAX_NONCONVEX_COLLISION_FACES = 1000


def _guard_collision_mesh(morph: Any, where: str) -> None:
    """Refuse to build a large NON-CONVEX mesh as a rigid collision/raycast body.

    A ``gs.morphs.Mesh`` with ``convexify=False`` keeps its full concave
    geometry. As a *rigid* body that is expensive two ways: Genesis builds an
    SDF / collision structure over every face, AND — in ``single_scene`` — the
    wheel raycaster re-fits a BVH over every face *each step* (the vehicle moves,
    so the BVH is never static). Past ``_MAX_NONCONVEX_COLLISION_FACES`` either
    cost can spike memory/compute and hard-crash the process (under WSL, the
    whole VM). It fires even with a large ``sdf_cell_size`` (which only caps the
    SDF grid) because the per-step raycaster re-fit is unaffected by it. Raising
    here — before the entity is added/built — turns that silent crash into a
    clear, actionable error and asks for the mesh to be reviewed.

    Exempt:
    - primitives (Box/Sphere/…), heightfields — not a ``Mesh``;
    - ``convexify=True`` — convex decomposition keeps both the collider and the
      raycast BVH cheap;
    - ``collision=False`` — visual-only / kinematic wheel-raycast surfaces
      (``add_static(collision=False)`` in dual_scene): no SDF, and the kinematic
      raycast BVH is built once (static) — the recommended home for a high-poly
      surface."""
    if type(morph).__name__ != "Mesh":
        return
    if getattr(morph, "convexify", True):
        return
    if not getattr(morph, "collision", True):
        return
    f = getattr(morph, "file", None)
    if not f or not os.path.exists(f):
        return                      # can't introspect → best-effort, let it through
    try:
        import trimesh
        n_faces = int(len(trimesh.load(f, process=False, force="mesh").faces))
    except Exception:
        return
    if n_faces <= _MAX_NONCONVEX_COLLISION_FACES:
        return
    _logger.error(
        "[genesis_vehicle:mesh-guard] %s: %d-face non-convex mesh requested as a "
        "RIGID collision/raycast body with convexify=False (limit %d). >>> REVIEW "
        "THIS MESH <<< — as a rigid body it pays both an SDF/collision build and "
        "(single_scene) a per-step wheel-raycaster BVH re-fit over every face, "
        "either of which can crash the process (independent of sdf_cell_size, "
        "which only caps the SDF grid). Fix: decimate it, enable convexify=True "
        "(convex decomposition), or register it as a KINEMATIC wheel-raycast "
        "target (add_static(collision=False) in dual_scene), which needs no SDF "
        "and whose BVH is built once. File: %s",
        where, n_faces, _MAX_NONCONVEX_COLLISION_FACES, f)
    raise ValueError(
        f"[genesis_vehicle:mesh-guard] {where}: refusing to build a {n_faces}-face "
        f"non-convex mesh as a rigid collision/raycast body with convexify=False "
        f"(limit {_MAX_NONCONVEX_COLLISION_FACES}). As a rigid body it forces a "
        f"large SDF/collision build AND, in single_scene, a per-step "
        f"wheel-raycaster BVH re-fit over every face — either can exhaust "
        f"memory/compute and crash the process/WSL (independent of sdf_cell_size, "
        f"which only caps the SDF grid). Decimate the mesh, set convexify=True, or "
        f"use a kinematic wheel-raycast target (add_static(collision=False)) "
        f"instead. File: {f}")

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
class DynamicBody:
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


@dataclass
class Camera:
    """Handle for a camera registered on a :class:`VehicleScene` (the SDK wrapper
    around a Genesis camera). Lives on the main scene and works in every view mode
    (``"native"`` and ``"cv2"``). Returned by :meth:`VehicleScene.add_camera`."""
    name: str
    entity: Any = None        # underlying gs camera

    def render(self, *, rgb=True, depth=False, segmentation=False, normal=False):
        """Render a frame from this camera (offscreen). Returns whatever Genesis
        returns for the requested buffers — e.g. the RGB array for a cv2 window."""
        return self.entity.render(rgb=rgb, depth=depth,
                                  segmentation=segmentation, normal=normal)

    def set_pose(self, *, pos=None, lookat=None, up=None):
        """Move / aim the camera (e.g. a chase cam following the vehicle)."""
        self.entity.set_pose(pos=pos, lookat=lookat, up=up)


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
        self.entity_main: Any = None   # main-scene rigid URDF entity
        self.physics: Optional[VehiclePhysics] = None
        self.sensor: Any = None        # wheel raycaster (main in single, raycast-scene in split)
        self.proxy: Any = None         # raycast-scene pose carrier (split only)
        self._inputs = VehicleInputs(throttle=0.0, brake=0.0, steer=0.0)
        self._two_scene = False
        self._n_envs = 1

    # ---- per-step input ----
    def set_inputs(self, throttle=0.0, brake=0.0, steer=0.0) -> "Vehicle":
        """Set this vehicle's controls for the next ``step``(s). Each of
        ``throttle`` [-1,1] (+ = accel) / ``brake`` [0,1] / ``steer`` [-1,1]
        (+ = right) is a scalar (broadcast to all envs) **or a per-env tensor of
        shape ``(n_envs,)``** for L3 — e.g. per-env random controls in an RL / MPPI
        rollout. Persists until the next ``set_inputs``."""
        self._inputs = VehicleInputs(throttle=throttle, brake=brake, steer=steer)
        return self

    # ---- pose accessors (main scene = physical truth) ----
    def get_pos(self):
        return self.entity_main.get_pos()

    def get_quat(self):
        return self.entity_main.get_quat()

    def get_vel(self):
        return self.entity_main.get_vel()

    def get_ang(self):
        return self.entity_main.get_ang()

    @property
    def distances(self):
        """Last wheel-ground distances used by the physics step."""
        return None if self.physics is None else self.physics.last_distances

    # ---- internals (called by VehicleScene) ----
    def _sync_proxy(self) -> None:
        """Mirror the main chassis base pose onto the raycast-scene proxy.
        ``set_pos``/``set_quat`` run FK, so the wheel ray origins update without
        stepping the raycast scene."""
        p = self.entity_main.get_pos()
        q = self.entity_main.get_quat()
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
        viewer_options: Any = None,
        view: Optional[str] = None,
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
        # Render mode (all on the MAIN scene; the raycast scene never renders):
        #   None    — headless (no Genesis rendering)
        #   "native"— open the native Genesis viewer window
        #   "cv2"   — no native window; add cameras with add_camera() and render
        #             them yourself (e.g. to a cv2 window or a frame buffer)
        # add_camera() works in EVERY mode (native + cv2). show_viewer is a
        # back-compat alias for view="native".
        if view not in (None, "native", "cv2"):
            raise ValueError(f"view must be None, 'native', or 'cv2', got {view!r}")
        self.view = view
        self.show_viewer = (view == "native") or bool(show_viewer)
        self._cameras: list["Camera"] = []
        self._built = False

        if init_genesis:
            _ensure_genesis(backend)

        _sim = sim_options or gs.options.SimOptions(dt=dt, substeps=substeps, gravity=gravity)
        _rigid = rigid_options or gs.options.RigidOptions(dt=dt, enable_collision=True)
        _vis = vis_options or gs.options.VisOptions()

        # viewer_options (gs.options.ViewerOptions: camera_pos / camera_lookat /
        # res / max_FPS / refresh_rate …) configures the NATIVE viewer; only the
        # main scene can have one (the raycast scene is sensors-only). None lets
        # Genesis use its defaults. Pass show_viewer=True to actually open it.
        _scene_kw = dict(sim_options=_sim, rigid_options=_rigid, vis_options=_vis,
                         show_viewer=self.show_viewer)
        if viewer_options is not None:
            _scene_kw["viewer_options"] = viewer_options
        self.main_scene = gs.Scene(**_scene_kw)
        # The raycast scene is SENSORS-ONLY and is NEVER viewed or rendered:
        #   - created with show_viewer=False ALWAYS (independent of this
        #     VehicleScene's show_viewer / viewer_options, which apply to the main
        #     scene only) → no native viewer window, _visualizer._viewer is None;
        #   - no camera is ever added to it;
        #   - its step() is always called with update_visualizer=False (see build()
        #     and _measure_distances) → the visualizer is never updated for it.
        # Only the main scene can be viewed/rendered. It also never advances real
        # dynamics (kinematic terrain + fixed proxies); reuse the same dt for sanity.
        self.raycast_scene = (
            gs.Scene(sim_options=gs.options.SimOptions(dt=dt, substeps=1, gravity=(0, 0, 0)),
                     show_viewer=False)
            if self._two_scene else None
        )

        self._vehicles: list[Vehicle] = []
        self._statics: list[StaticBody] = []
        self._dynamics: list[DynamicBody] = []
        self._pending_mass: list = []   # (entity, mass) applied after build()

    # -----------------------------------------------------------------
    # Registration (before build)
    # -----------------------------------------------------------------

    def add_static(
        self,
        *,
        morph: Any = None,
        wheel_raycast_morph: Any = None,
        collision_morph: Any = None,
        collision: bool = True,
        material: Any = None,
        surface: Any = None,
        vis_mode: Any = None,
        name: Optional[str] = None,
    ) -> StaticBody:
        """Register a static body (terrain / mesh / plane / primitive). A static
        body is **always a wheel-raycast target** (the wheels drive on it) — there
        is no toggle here; use :meth:`add_dynamic` for a moving body the wheels
        only collide with.

        Provide one ``morph`` for both roles, or split into ``wheel_raycast_morph``
        (detailed surface the wheel rays cast against) and ``collision_morph``
        (coarse/convex body the chassis collides with). Splitting is recommended
        for high-poly or non-convex meshes: a rigid mesh is auto-convexified for
        collision (so a rigid-mesh raycast hits the convex bulge, not the true
        surface), whereas the kinematic raycast surface stays exact.

        VehicleScene owns the scene routing: the collision body lands in the main
        scene (rigid) and, in dual_scene mode, a kinematic raycast mirror lands in
        the raycast scene. Callers never touch the underlying scenes.
        """
        self._require_not_built()
        if morph is None and wheel_raycast_morph is None and collision_morph is None:
            raise ValueError("add_static: provide morph (or wheel_raycast_morph/collision_morph).")
        rc_morph = wheel_raycast_morph or morph
        col_morph = collision_morph or morph
        name = name or f"static_{len(self._statics)}"

        if (not self._two_scene and wheel_raycast_morph is not None
                and collision and col_morph is not None):
            # A distinct raycast surface vs collider split needs the two bodies of
            # dual_scene. In single_scene the one rigid collision body is the
            # raycast target, so a *separate* wheel_raycast_morph is dropped. (When
            # wheel_raycast_morph is the only geometry — collision=False — it IS the
            # body, so no warning.)
            _logger.warning(
                "[genesis_vehicle:single-scene] add_static(%r): wheel_raycast_morph "
                "only applies in dual_scene (a "
                "separate kinematic raycast surface). In single_scene the one rigid "
                "collision body is also the raycast target, so a distinct "
                "wheel_raycast_morph is ignored.", name)

        if not self._two_scene and not collision and rc_morph is not None:
            # single_scene has ONE body, and the wheel-raycast target must be a
            # rigid body the rays can hit — so collision=False cannot be honored:
            # the body still collides (it is created as a rigid below). Use
            # dual_scene for a true no-collision (kinematic) raycast surface.
            _logger.warning(
                "[genesis_vehicle:single-scene] add_static(%r): collision=False "
                "cannot be honored in single_scene "
                "— the lone rigid body is also the wheel-raycast target, so it "
                "still collides. Use raycast_mode='dual_scene' for a true "
                "no-collision (kinematic) raycast surface.", name)

        body = StaticBody(name=name, has_collision=bool(collision), has_raycast=True)

        if collision and col_morph is not None:
            _guard_collision_mesh(col_morph, f"add_static({name!r})")
            mat = material if material is not None else gs.materials.Rigid()
            body.entity_main = self.main_scene.add_entity(
                col_morph, **_add_kwargs(mat, surface, vis_mode))

        if rc_morph is not None:
            if self._two_scene:
                # Kinematic visual-raycast body in the raycast scene → static BVH.
                # Force vis_mode="visual": a KinematicEntity is visual-only (it has
                # vgeoms, not collision geoms), so the renderer's on_rigid must take
                # the vgeoms path. A non-"visual" vis_mode (e.g. the server's
                # --vis-mode=collision leaking through, or a changed Genesis default)
                # makes on_rigid touch entity.geoms → AttributeError at build. This
                # scene is sensors-only and never user-rendered, so "visual" is right.
                body.entity_raycast = self.raycast_scene.add_entity(
                    rc_morph, **_add_kwargs(
                        gs.materials.Kinematic(use_visual_raycasting=True),
                        surface, "visual"))
            else:
                # single mode: the rigid collision body IS the raycast target.
                if body.entity_main is None:
                    _guard_collision_mesh(rc_morph, f"add_static({name!r})")
                    mat = material if material is not None else gs.materials.Rigid()
                    body.entity_main = self.main_scene.add_entity(
                        rc_morph, **_add_kwargs(mat, surface, vis_mode))
                body.entity_raycast = body.entity_main

        self._statics.append(body)
        return body

    def add_ground_plane(self, *, friction: float = 0.85) -> StaticBody:
        """Convenience: an infinite flat ground (raycast + collision)."""
        return self.add_static(morph=gs.morphs.Plane(),
                               material=gs.materials.Rigid(friction=friction),
                               name="ground")

    def add_camera(self, *, res=(1280, 720), pos=(3.0, -3.0, 2.0),
                   lookat=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0), fov=50.0,
                   GUI=False, name=None, **kwargs) -> Camera:
        """Add a camera to the (main) scene — call before :meth:`build`. Render
        frames in your loop with ``cam.render()`` (e.g. into a cv2 window), or aim
        it with ``cam.set_pose(...)``. Works in every ``view`` mode (it is how the
        ``"cv2"`` mode renders, and you can also add cameras alongside ``"native"``).
        The caller never touches the underlying scene. Adding any camera also
        auto-enables the wheels' VisualJointSync at build, so rendered wheels
        animate."""
        self._require_not_built()
        cam = self.main_scene.add_camera(res=res, pos=pos, lookat=lookat, up=up,
                                         fov=fov, GUI=GUI, **kwargs)
        handle = Camera(name=name or f"camera_{len(self._cameras)}", entity=cam)
        self._cameras.append(handle)
        return handle

    def add_dynamic(
        self,
        morph: Any,
        *,
        physics: bool = True,
        wheel_raycast: bool = False,
        material: Any = None,
        surface: Any = None,
        vis_mode: Any = None,
        mass: Optional[float] = None,
        name: Optional[str] = None,
    ) -> DynamicBody:
        """Register a MOVING body. By default the wheels only **collide** with it
        (``wheel_raycast=False`` — e.g. another vehicle or a dynamic prop). Set
        ``wheel_raycast=True`` only for a moving surface the wheels must **sense /
        drive onto** (a ramp or moving platform). Use :meth:`add_static` for bodies
        that never move (those are always wheel-raycast targets).

        ``physics=True`` (default): a free rigid body that moves under physics.
        ``physics=False``: a fixed body you teleport yourself with
        ``handle.set_pose(...)`` (e.g. an externally / UE-driven body).

        With ``wheel_raycast=True`` in dual_scene mode a synced rigid raycast
        mirror is added to the raycast scene (re-synced every ``step``; re-fits
        only its own small BVH, leaving the terrain static). In single_scene mode
        the one main-scene rigid body is already the raycast target. VehicleScene
        owns the routing — the caller never touches a scene.

        Guard: a wheel-raycast dynamic body re-fits its BVH every step, so a
        non-primitive (mesh) morph logs a warning — prefer a primitive collider.
        """
        self._require_not_built()
        name = name or f"dynamic_{len(self._dynamics)}"
        mat = material if material is not None else gs.materials.Rigid()

        if wheel_raycast and not self._two_scene:
            # wheel_raycast adds a dedicated raycast-scene mirror, which only
            # exists in dual_scene. In single_scene there is no raycast scene, and
            # a rigid body is already a wheel-raycast target via the main scene, so
            # the flag changes nothing.
            _logger.warning(
                "[genesis_vehicle:single-scene] add_dynamic(%r): wheel_raycast=True "
                "has no effect in single_scene "
                "mode — there is no raycast scene to mirror into, and the rigid body "
                "is already a wheel-raycast target via the main scene. The flag only "
                "adds a dedicated mirror in dual_scene.", name)
        elif wheel_raycast and type(morph).__name__ not in _PRIMITIVE_MORPHS:
            _logger.warning(
                "[genesis_vehicle:refit-cost] add_dynamic(%r): wheel_raycast=True on "
                "a non-primitive (%s) morph — "
                "the dual_scene mirror's BVH re-fits every step (cost grows with face "
                "count). Prefer a primitive collider (Box/Sphere/Cylinder) for a "
                "wheel_raycast dynamic body.", name, type(morph).__name__)

        obs = DynamicBody(name=name, is_dynamic=bool(physics),
                          has_raycast=bool(wheel_raycast))
        main_morph = _clone_morph(morph, fixed=not physics)
        _guard_collision_mesh(main_morph, f"add_dynamic({name!r})")
        obs.entity_main = self.main_scene.add_entity(
            main_morph, **_add_kwargs(mat, surface, vis_mode))
        if mass is not None:
            self._pending_mass.append((obs.entity_main, float(mass)))

        if wheel_raycast and self._two_scene:
            # Rigid + fixed + collision mirror in the raycast scene's RIGID
            # solver (a separate BVH context from the kinematic terrain), so
            # re-syncing it each step re-fits only this small body.
            mirror_morph = _clone_morph(morph, fixed=True, collision=True)
            _guard_collision_mesh(mirror_morph,
                                  f"add_dynamic({name!r}) wheel_raycast mirror")
            obs.entity_raycast = self.raycast_scene.add_entity(
                mirror_morph, **_add_kwargs(gs.materials.Rigid(), surface, vis_mode))

        self._dynamics.append(obs)
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
        veh.entity_main = self.main_scene.add_entity(
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
                entity_idx=veh.entity_main.idx,
                max_range=raycaster_max_range, min_range=0.0, return_world_frame=True))

        self._vehicles.append(veh)
        return veh

    # -----------------------------------------------------------------
    # Build / step
    # -----------------------------------------------------------------

    def build(self, *, env_spacing=None, n_envs_per_row=None) -> None:
        """Build both scenes and wire up per-vehicle physics.

        ``env_spacing`` / ``n_envs_per_row`` are forwarded to the underlying
        ``Scene.build`` (both scenes get the SAME layout so the raycast proxy
        stays aligned with the main-scene vehicle per env) — use them to lay the
        L3 batch out on a grid (e.g. one cell per checkpoint) instead of stacked
        at the origin.

        Build ORDER matters in ``dual_scene`` with a native viewer: Genesis
        creates a GL ``RasterizerContext`` per scene, and building the
        sensors-only raycast scene *after* the main one makes the raycast context
        current — the viewer thread (on the main scene) then dies with
        "no valid context". So the raycast scene is built FIRST and the main
        scene (which may own the viewer) LAST, leaving the main context current;
        the per-step ``raycast_scene.step(update_visualizer=False)`` is CUDA-only
        and never touches GL, so it can't steal the context back.
        """
        if self._built:
            return
        _kw = {}
        if env_spacing is not None:
            _kw["env_spacing"] = env_spacing
        if n_envs_per_row is not None:
            _kw["n_envs_per_row"] = n_envs_per_row
        if self._two_scene:
            self.raycast_scene.build(n_envs=self.n_envs, **_kw)
            # Populate the raycast sensors once so the static BVH is built and
            # the first read() returns valid data (Genesis sensors are empty
            # before the first step). The kinematic terrain/proxy don't move, so
            # this single step is cheap and never repeated.
            # update_visualizer=False: this scene is sensors-only and never
            # user-rendered, so skip the per-step visualizer/render update.
            self.raycast_scene.step(update_visualizer=False)
        self.main_scene.build(n_envs=self.n_envs, **_kw)   # viewer (if any) starts LAST

        # VisualJointSync drives the URDF wheel VISUAL joints through the engine
        # each step so GENESIS's own renderer shows wheels spinning/steering. It is
        # useful ONLY when the main scene is actually rendered by Genesis — a native
        # viewer or a Genesis camera — so VehicleScene auto-manages it here (it is
        # not a user-facing option): on a headless / external-renderer run it stays
        # off, and wheel poses are read closed-form via wheel_visual_transforms().
        renders = self.show_viewer or bool(getattr(self.main_scene.visualizer, "cameras", None))
        for veh in self._vehicles:
            veh.cfg.enable_visual_joint_sync = renders
            sensor = None if self._two_scene else veh.sensor
            veh.physics = VehiclePhysics(
                self.main_scene, veh.entity_main, sensor, veh.cfg, n_envs=self.n_envs)

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
        for obs in self._dynamics:
            if obs.entity_raycast is not None:
                self._sync_dynamic(obs)
        # update_visualizer=False: the raycast scene is sensors-only and never
        # user-rendered, so skip the per-step visualizer/render update — the
        # sensor re-cast still runs inside sim.step(). Saves the render call the
        # raycast scene was making every frame.
        self.raycast_scene.step(update_visualizer=False)
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
    def dynamics(self) -> list:
        return list(self._dynamics)

    @property
    def cameras(self) -> list:
        return list(self._cameras)

    # ---- internals ----
    def _sync_dynamic(self, obs: "DynamicBody") -> None:
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
