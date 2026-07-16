"""VehicleScene: the unified, high-level entry point of the SDK.

A ``VehicleScene`` owns the Genesis scene(s), the registered vehicles, and the
static/dynamic bodies, and drives the per-step loop — so the caller never
touches ``gs.init`` / ``scene.build`` / ``scene.step`` / sensor reads directly.

Backends — physics vs renderer
------------------------------
The Genesis **physics** backend (CPU / GPU) is the COMPUTE target. It is
process-global and set ONCE, and is deliberately NOT a ``VehicleScene``
constructor argument: call ``VehicleScene.init_backend("cpu" | "gpu")`` BEFORE
constructing any scene (default **cpu**). A second call — or any double-init,
internal or external — warns and is ignored (the backend can't change within a
process). Constructing a ``VehicleScene`` without calling it first auto-initializes
the **cpu** backend.

The **renderer** is independent of the physics backend: the viewer / cameras
rasterize on the **GPU** graphics stack regardless of where physics runs. So
**physics-on-CPU + GPU-rendering** is valid (and the natural n_envs=1 visual
combo); "GPU physics + CPU rendering" is not a thing (there is no software-render
mode). If no GPU is present, rendering falls back to slow software and ``build()``
logs a warning.

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

Why ``"dual_scene"`` is the default (see ``docs/dual-scene-raycast.md``): complex
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


def _ensure_genesis(backend: Optional[str] = None) -> None:
    """Ensure the Genesis **physics** backend is up (process-global, set once).

    ``backend`` is an EXPLICIT request from ``VehicleScene.init_backend`` (``"cpu"``
    / ``"gpu"``); ``None`` is ``VehicleScene.__init__``'s silent auto-ensure (default
    cpu). If Genesis is already initialized, an EXPLICIT request for a *different*
    backend warns and is ignored — but the silent auto-ensure (``backend=None``)
    never warns, it just reuses whatever is up. Uses ``gs._initialized`` (not a
    swallowed exception) so a REAL init failure (e.g. missing libcuda on GPU) still
    propagates."""
    global _GENESIS_BACKEND
    if getattr(gs, "_initialized", False) or _GENESIS_BACKEND is not None:
        if (backend is not None and _GENESIS_BACKEND is not None
                and backend != _GENESIS_BACKEND):
            _logger.warning(
                "Genesis physics backend already initialized as %r; ignoring the "
                "requested %r (the backend is process-global and set once — call "
                "VehicleScene.init_backend(...) before any VehicleScene to choose it).",
                _GENESIS_BACKEND, backend)
        _GENESIS_BACKEND = _GENESIS_BACKEND or backend or "cpu"
        return
    be_str = backend or "cpu"     # auto-ensure (backend=None) defaults to cpu
    gs.init(backend=(gs.gpu if be_str == "gpu" else gs.cpu), logging_level="warning")
    _GENESIS_BACKEND = be_str


def _render_gpu_available() -> bool:
    """Best-effort: is a GPU present for the renderer? The renderer (rasterizer)
    uses GPU graphics independently of the physics backend; with no GPU it falls
    back to slow software rendering. We proxy availability via CUDA visibility."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


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
    the raycast scene (kinematic, dual_scene mode) or is the same rigid body
    (single_scene mode)."""
    name: str
    has_collision: bool
    has_raycast: bool
    is_static: bool = True
    entity_main: Any = None       # rigid collision entity (main scene)
    entity_raycast: Any = None    # kinematic raycast entity (raycast scene, dual_scene)


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

    Wraps the main-scene rigid entity, its :class:`VehiclePhysics`, and
    (dual_scene mode) the raycast-scene proxy + sensor."""

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
        self.sensor: Any = None        # wheel raycaster (main scene in single_scene, raycast scene in dual_scene)
        self.proxy: Any = None         # raycast-scene pose carrier (dual_scene only)
        self._inputs = VehicleInputs(throttle=0.0, brake=0.0, steer=0.0)
        self._dual_scene = False
        self._n_envs = 1
        self._scene = None         # back-ref to VehicleScene (for batched accessors)
        self._slot = -1            # this vehicle's flat index in the scene
        self._kind_key = None      # batched grouping key (set in add_vehicle)
        self._group_cfg = cfg      # shared cfg of its batched kind

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

    # ---- per-vehicle accessors (work in BOTH solver modes) ----
    # In solver="per_vehicle" they read this vehicle's own VehiclePhysics; in
    # solver="batched" they delegate to the shared MultiVehiclePhysics + this
    # vehicle's flat slot, so callers (the server, samples) never branch on solver.
    @property
    def distances(self):
        """Last wheel-ground distances of the physics step — ``(n_envs, n_wheels)``."""
        if self.physics is not None:
            return self.physics.last_distances
        if self._scene is not None and self._scene._mvp is not None:
            return self._scene._mvp.distances_list()[self._slot]
        return None

    @property
    def resolved(self):
        """The resolved ``VehicleConfig`` actually driving this vehicle."""
        if self.physics is not None:
            return self.physics.resolved
        if self._scene is not None and self._scene._mvp is not None:
            return self._scene._mvp.resolved_list[self._slot]
        return None

    def wheel_visual_transforms(self, frame: str = "world"):
        """Closed-form per-wheel visual poses ``(pos, quat)`` — ``(n_envs, n_wheels,
        3)`` / ``(…, 4)``. WheelJointInternalSync-independent (works headless)."""
        if self.physics is not None:
            return self.physics.wheel_visual_transforms(frame)
        return self._scene._mvp.wheel_visual_transforms(frame)[self._slot]

    def visual_parts_transforms(self, frame: str = "world"):
        """One-call render feed (chassis + wheels) for this vehicle."""
        if self.physics is not None:
            return self.physics.visual_parts_transforms(frame)
        return self._scene._mvp.visual_parts_transforms(frame)[self._slot]

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

    @staticmethod
    def init_backend(backend: str = "cpu") -> None:
        """Initialize the Genesis **physics** backend, explicitly, before constructing
        any ``VehicleScene``. ``backend`` is ``"cpu"`` (the default) or ``"gpu"`` and
        is the COMPUTE backend only — the renderer is separate (always GPU; see
        "Backends" in the class docs). Process-global and set once: if Genesis is
        already initialized, a request for a DIFFERENT backend warns and is ignored.

        Constructing a ``VehicleScene`` without calling this first auto-initializes
        the **cpu** backend. So: opt into GPU physics by calling this up front.

        >>> VehicleScene.init_backend("gpu")   # GPU physics (e.g. large-n_envs L3)
        >>> vs = VehicleScene(n_envs=4096)
        """
        if backend not in ("cpu", "gpu"):
            raise ValueError(f"backend must be 'cpu' or 'gpu', got {backend!r}")
        _ensure_genesis(backend)

    def __init__(
        self,
        *,
        n_envs: int = 1,
        dt: float = 1.0 / 200.0,
        raycast_mode: str = "dual_scene",
        gravity: tuple = (0.0, 0.0, -9.81),
        substeps: int = 4,
        sim_options: Any = None,
        rigid_options: Any = None,
        vis_options: Any = None,
        viewer_options: Any = None,
        view: Optional[str] = None,
        show_viewer: bool = False,
        solver: str = "batched",
        wheel_render_mode: str = "auto",
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
        self.raycast_mode = raycast_mode
        self._dual_scene = raycast_mode == "dual_scene"
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
        # solver: "per_vehicle" — one VehiclePhysics per vehicle (works in any
        # raycast_mode). "batched" — one MultiVehiclePhysics that groups vehicles
        # of the same kind (= same cfg object) into one batched compute; faster for
        # many same-kind vehicles (L2). In dual_scene the raycast-scene distances
        # are injected into the batched compute; in single_scene the batched solver
        # reads each vehicle's own sensor.
        if solver not in ("per_vehicle", "batched"):
            raise ValueError(f"solver must be 'per_vehicle' or 'batched', got {solver!r}")
        self.solver = solver
        # How wheel visuals reach the Genesis renderer when the scene renders:
        #   "auto"       — instanced solver-free rendering when supported
        #                  (n_envs == 1, wheel meshes harvestable), else
        #                  WheelJointInternalSync (the pre-1.1.17 mechanism)
        #   "instanced"  — force the instanced path (raises when unsupported)
        #   "internal_sync" — force WheelJointInternalSync
        # Headless runs render nothing either way (both stay off).
        if wheel_render_mode not in ("auto", "instanced", "internal_sync"):
            raise ValueError("wheel_render_mode must be 'auto', 'instanced', "
                             f"or 'internal_sync', got {wheel_render_mode!r}")
        self.wheel_render_mode = wheel_render_mode
        self._wheel_renderer = None      # InstancedWheelRenderer when active
        self._viewer_atomic_wheels = False   # wheel buffers stream inside viewer.update (set at build)
        self._mvp = None        # MultiVehiclePhysics, in batched mode
        # Kind grouping is lazy + dirty-tracked: add_vehicle / mark_config_dirty
        # bump _config_version; _ensure_grouped re-groups only when it differs from
        # _grouped_version (so step() pays an O(1) int compare when nothing changed).
        self._config_version = 0
        self._grouped_version = -1
        self._built = False
        # Batched proxy sync (dual_scene): lazy (main_idx, proxy_idx, rc_solver)
        # cache + tri-state health flag (None=untried / True / False=fell back).
        self._proxy_sync_cache = None
        self._proxy_sync_ok = None

        # Ensure the Genesis physics backend is up (default cpu). To run on GPU,
        # call VehicleScene.init_backend("gpu") BEFORE constructing any VehicleScene;
        # this is a no-op if it (or gs.init) already ran. init_genesis=False lets a
        # caller manage gs.init entirely on its own.
        if init_genesis:
            _ensure_genesis()

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
        self._main_scene = gs.Scene(**_scene_kw)
        # The raycast scene is SENSORS-ONLY and is NEVER viewed or rendered:
        #   - created with show_viewer=False ALWAYS (independent of this
        #     VehicleScene's show_viewer / viewer_options, which apply to the main
        #     scene only) → no native viewer window, _visualizer._viewer is None;
        #   - no camera is ever added to it;
        #   - its step() is always called with update_visualizer=False (see build()
        #     and _measure_distances) → the visualizer is never updated for it.
        # Only the main scene can be viewed/rendered. It also never advances real
        # dynamics (kinematic terrain + fixed proxies); reuse the same dt for sanity.
        self._raycast_scene = (
            gs.Scene(sim_options=gs.options.SimOptions(dt=dt, substeps=1, gravity=(0, 0, 0)),
                     show_viewer=False)
            if self._dual_scene else None
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

        ``collision=False`` makes the body a pure wheel-raycast surface with NO
        collision geometry — a **dual_scene-only** feature (the kinematic
        ``use_visual_raycasting`` body lives in the raycast scene). In
        single_scene it raises ``ValueError`` (since v1.0.7 — previously it
        warned and built a rigid the wheel rays could not hit when the morph
        itself carried ``collision=False``, i.e. a fall-through surface).
        """
        self._require_not_built()
        if morph is None and wheel_raycast_morph is None and collision_morph is None:
            raise ValueError("add_static: provide morph (or wheel_raycast_morph/collision_morph).")
        rc_morph = wheel_raycast_morph or morph
        col_morph = collision_morph or morph
        name = name or f"static_{len(self._statics)}"

        if (not self._dual_scene and wheel_raycast_morph is not None
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

        if not self._dual_scene and not collision:
            # A no-collision static is a KINEMATIC raycast surface, which lives in
            # the dual_scene raycast scene. single_scene has ONE body and the wheel
            # rays only hit rigid collision geoms there — a collision=False body
            # would be invisible to the rays (vehicles fall straight through it).
            # Fail fast instead of building a silently broken scene (pre-1.0.7 this
            # warned and built a rigid from the raycast morph; with the morph's own
            # collision=False that rigid was exactly the fall-through case).
            _logger.error(
                "[genesis_vehicle:single-scene] add_static(%r): collision=False "
                "requires raycast_mode='dual_scene' (kinematic raycast surface in "
                "the raycast scene). In single_scene the wheel rays cannot hit a "
                "no-collision body — refusing to build a fall-through scene.", name)
            raise ValueError(
                f"add_static({name!r}): collision=False requires "
                f"raycast_mode='dual_scene'; single_scene cannot host a "
                f"no-collision wheel-raycast surface.")

        body = StaticBody(name=name, has_collision=bool(collision), has_raycast=True)

        if collision and col_morph is not None:
            _guard_collision_mesh(col_morph, f"add_static({name!r})")
            mat = material if material is not None else gs.materials.Rigid()
            body.entity_main = self._main_scene.add_entity(
                col_morph, **_add_kwargs(mat, surface, vis_mode))

        if rc_morph is not None:
            if self._dual_scene:
                # Kinematic visual-raycast body in the raycast scene → static BVH.
                # Force vis_mode="visual": a KinematicEntity is visual-only (it has
                # vgeoms, not collision geoms), so the renderer's on_rigid must take
                # the vgeoms path. A non-"visual" vis_mode (e.g. the server's
                # --vis-mode=collision leaking through, or a changed Genesis default)
                # makes on_rigid touch entity.geoms → AttributeError at build. This
                # scene is sensors-only and never user-rendered, so "visual" is right.
                body.entity_raycast = self._raycast_scene.add_entity(
                    rc_morph, **_add_kwargs(
                        gs.materials.Kinematic(use_visual_raycasting=True),
                        surface, "visual"))
            else:
                # single_scene mode: the rigid collision body IS the raycast target.
                if body.entity_main is None:
                    _guard_collision_mesh(rc_morph, f"add_static({name!r})")
                    mat = material if material is not None else gs.materials.Rigid()
                    body.entity_main = self._main_scene.add_entity(
                        rc_morph, **_add_kwargs(mat, surface, vis_mode))
                body.entity_raycast = body.entity_main

        self._statics.append(body)
        return body

    def add_ground_plane(self, *, friction: float = 0.85) -> StaticBody:
        """Convenience: an infinite flat ground (raycast + collision)."""
        return self.add_static(morph=gs.morphs.Plane(),
                               material=gs.materials.Rigid(friction=friction),
                               name="ground")

    def add_raycast_surface(self, morph: Any, *, surface: Any = None,
                            name: Optional[str] = None) -> StaticBody:
        """Register a **wheel-raycast-ONLY** static surface (v1.0.10).

        The body exists solely for the wheel rays: a Kinematic
        ``use_visual_raycasting`` entity in the raycast scene — it has **zero
        rigid-solver presence** (no collision geoms, no FK/dynamics, no
        broadphase pairs, its BVH is built once). This is the first-class name
        for what ``add_static(collision=False, wheel_raycast_morph=...)`` did —
        the negative-flag spelling hid the intent (and its single_scene
        fall-through was a real bug pre-1.0.7), so express it directly:

        >>> vs.add_raycast_surface(gs.morphs.Mesh(file="road.obj",
        ...                        collision=False), name="road")

        Ideal for high-poly roads/terrain the vehicles drive on but whose
        chassis-collision response you don't need (``--road-raycast-only``).
        The mesh-guard face limit does not apply (kinematic, no SDF, static
        BVH). ``dual_scene`` only — in ``single_scene`` the wheel rays only hit
        rigid collision geoms, so this raises ``ValueError`` (same fail-fast
        as ``add_static(collision=False)``).
        """
        return self.add_static(wheel_raycast_morph=morph, collision=False,
                               surface=surface, name=name)

    def add_camera(self, *, res=(1280, 720), pos=(3.0, -3.0, 2.0),
                   lookat=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0), fov=50.0,
                   GUI=False, name=None, **kwargs) -> Camera:
        """Add a camera to the (main) scene — call before :meth:`build`. Render
        frames in your loop with ``cam.render()`` (e.g. into a cv2 window), or aim
        it with ``cam.set_pose(...)``. Works in every ``view`` mode (it is how the
        ``"cv2"`` mode renders, and you can also add cameras alongside ``"native"``).
        The caller never touches the underlying scene. Adding any camera also
        auto-enables wheel visuals at build (v1.1.17: the solver-free
        instanced renderer; legacy joint-sync only as fallback), so rendered wheels
        animate."""
        self._require_not_built()
        cam = self._main_scene.add_camera(res=res, pos=pos, lookat=lookat, up=up,
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

        if wheel_raycast and not self._dual_scene:
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
        obs.entity_main = self._main_scene.add_entity(
            main_morph, **_add_kwargs(mat, surface, vis_mode))
        if mass is not None:
            self._pending_mass.append((obs.entity_main, float(mass)))

        if wheel_raycast and self._dual_scene:
            # Rigid + fixed + collision mirror in the raycast scene's RIGID
            # solver (a separate BVH context from the kinematic terrain), so
            # re-syncing it each step re-fits only this small body.
            mirror_morph = _clone_morph(morph, fixed=True, collision=True)
            _guard_collision_mesh(mirror_morph,
                                  f"add_dynamic({name!r}) wheel_raycast mirror")
            obs.entity_raycast = self._raycast_scene.add_entity(
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

        The URDF is ALWAYS made ray-wheel ready first (v1.1.22; not optional
        since v1.1.23) — wheel colliders become render-only, a suspension
        attach point that sits off the wheel centre is corrected, and links
        missing an ``<inertial>`` get one. The original file is never
        modified, and a URDF that already satisfies the contracts (every
        SDK-authored vehicle) is used as-is with no copy. See
        :mod:`genesis_vehicle.urdf_prep` and ``docs/physics-contracts.md``
        §7.9. The prepared path is used for the entity, the parse AND the ray
        pattern — they must agree, which is why this is not a knob.
        """
        self._require_not_built()
        if cfg is None and preset is None:
            raise ValueError("add_vehicle: pass preset=<fn> or cfg=<VehicleConfig>.")
        name = name or f"vehicle_{len(self._vehicles)}"
        from .urdf_prep import prepare_vehicle_urdf
        if morph is None:
            urdf_path = prepare_vehicle_urdf(urdf_path)
        else:
            # A caller-supplied morph carries its OWN file. ``urdf_path`` still
            # decides where the wheel rays go (it is what gets parsed), so the
            # two must describe the same vehicle — a corrected morph paired with
            # an unprepared urdf_path silently puts the rays in the wrong place
            # (the v1.1.24 field report: an M1A2 whose suspension attach sits
            # below the wheel centre floated in UE, because the server passed
            # the original path with a stripped morph). Warn if the given path
            # is not already ray-wheel ready — the caller should prepare it once
            # and pass the SAME file to both.
            if prepare_vehicle_urdf(urdf_path, quiet=True) != urdf_path:
                _logger.warning(
                    "add_vehicle(morph=...) got an unprepared urdf_path (%s): the "
                    "wheel rays are placed from THIS file, so it must be the same "
                    "(prepared) URDF the morph was built from. Run "
                    "genesis_vehicle.urdf_prep.prepare_vehicle_urdf() once and pass "
                    "its result as both urdf_path and the morph's file.",
                    os.path.basename(urdf_path))
        user_cfg = cfg
        if cfg is None:
            cfg = preset(urdf_path, stability=stability)
        parsed = parse_urdf(urdf_path)
        wheel_positions = [w.position for w in parsed.wheels]

        veh = Vehicle(name, urdf_path, cfg, wheel_positions, pos, quat, material)
        veh._dual_scene = self._dual_scene
        veh._n_envs = self.n_envs
        # Kind key for the batched solver: vehicles registered the same way are one
        # kind and get batched. preset → grouped by (urdf, preset fn, stability);
        # a pre-built cfg → grouped by that cfg OBJECT (pass the same cfg to batch).
        # VehicleConfig has object fields (hooks/strategies) so value-equality is
        # unreliable; the registration-based key is robust and cheap.
        veh._kind_key = (("cfg", id(user_cfg)) if user_cfg is not None
                         else ("preset", urdf_path, preset, stability))
        veh._group_cfg = cfg
        veh._scene = self
        veh._slot = len(self._vehicles)   # flat index (== MVP order); set before append
        self._config_version += 1         # mark grouping dirty

        if morph is None:
            morph_kw = dict(file=urdf_path, pos=pos)
            if quat is not None:
                morph_kw["quat"] = quat
            morph = gs.morphs.URDF(**morph_kw)
        veh.entity_main = self._main_scene.add_entity(
            morph, **_add_kwargs(material, surface, vis_mode))

        if self._dual_scene:
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
            # proxy's link pose. See docs/dual-scene-raycast.md.
            veh.proxy = self._raycast_scene.add_entity(
                gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=pos,
                              fixed=True, collision=False),
                material=gs.materials.Rigid())
            veh.sensor = self._raycast_scene.add_sensor(gs.sensors.Raycaster(
                pattern=WheelRayPattern(wheel_positions),
                entity_idx=veh.proxy.idx,
                max_range=raycaster_max_range, min_range=0.0, return_world_frame=True))
        else:
            veh.sensor = self._main_scene.add_sensor(gs.sensors.Raycaster(
                pattern=WheelRayPattern(wheel_positions),
                entity_idx=veh.entity_main.idx,
                max_range=raycaster_max_range, min_range=0.0, return_world_frame=True))

        self._vehicles.append(veh)
        return veh

    # -----------------------------------------------------------------
    # Build / step
    # -----------------------------------------------------------------

    def build(self, *, env_spacing=None, n_envs_per_row=None,
              center_envs_at_origin=None) -> None:
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
        if center_envs_at_origin is not None:
            _kw["center_envs_at_origin"] = center_envs_at_origin
        # ---- wheel visual mode decision (v1.1.17) — BEFORE the main build:
        # the instanced path hides the URDF wheel vgeoms (renderer skips
        # geoms with an empty active_envs_idx), which must be set before the
        # visualizer creates its nodes at build.
        renders_pre = self.show_viewer or bool(self._cameras)
        self._use_instanced_wheels = False
        self._instanced_wheel_meshes = {}       # vehicle -> per-wheel trimeshes
        if renders_pre and self._vehicles and self.wheel_render_mode in ("auto", "instanced"):
            if self.n_envs != 1:
                if self.wheel_render_mode == "instanced":
                    raise ValueError(
                        "wheel_render_mode='instanced' supports n_envs == 1 only "
                        "(multi-env wheel visuals still use WheelJointInternalSync)")
            else:
                from .visual import InstancedWheelRenderer
                harvested = {}
                for veh in self._vehicles:
                    meshes = InstancedWheelRenderer.harvest_wheel_meshes(
                        veh.entity_main, veh.cfg.wheels)
                    if meshes is None:
                        harvested = None
                        break
                    harvested[veh] = meshes
                if harvested is not None:
                    self._use_instanced_wheels = True
                    self._instanced_wheel_meshes = harvested
                elif self.wheel_render_mode == "instanced":
                    raise ValueError(
                        "wheel_render_mode='instanced': could not harvest wheel "
                        "visual meshes from the URDF wheel links")
                else:
                    _logger.warning(
                        "wheel_render_mode='auto': wheel visual meshes not "
                        "harvestable — falling back to WheelJointInternalSync.")
        if self._use_instanced_wheels:
            import numpy as _np
            none_active = _np.array([], dtype=_np.int64)
            for veh in self._vehicles:
                for w in veh.cfg.wheels:
                    try:
                        link = veh.entity_main.get_link(w.name)
                    except Exception:
                        continue
                    for vg in getattr(link, "vgeoms", None) or []:
                        vg.active_envs_idx = none_active   # renderer skips it

        if self._dual_scene:
            self._raycast_scene.build(n_envs=self.n_envs, **_kw)
            # Populate the raycast sensors once so the static BVH is built and
            # the first read() returns valid data (Genesis sensors are empty
            # before the first step). The kinematic terrain/proxy don't move, so
            # this single step is cheap and never repeated.
            # update_visualizer=False: this scene is sensors-only and never
            # user-rendered, so skip the per-step visualizer/render update.
            self._raycast_scene.step(update_visualizer=False)
        self._main_scene.build(n_envs=self.n_envs, **_kw)   # viewer (if any) starts LAST

        # WheelJointInternalSync drives the URDF wheel VISUAL joints through the engine
        # each step so GENESIS's own renderer shows wheels spinning/steering. It is
        # useful ONLY when the main scene is actually rendered by Genesis — a native
        # viewer or a Genesis camera — so VehicleScene auto-manages it here (it is
        # not a user-facing option): on a headless / external-renderer run it stays
        # off, and wheel poses are read closed-form via wheel_visual_transforms().
        renders = self.show_viewer or bool(getattr(self._main_scene.visualizer, "cameras", None))
        # The renderer auto-selects GPU graphics regardless of the physics backend
        # (CPU physics + GPU render is fine). Warn only if rendering is on but no GPU
        # is available — Genesis then falls back to slow software (CPU) rendering.
        if renders and not _render_gpu_available():
            _logger.warning(
                "Rendering is enabled (viewer/camera) but no GPU was detected — "
                "Genesis falls back to software (CPU) rendering, which is slow.")
        # Instanced wheel rendering replaces WheelJointInternalSync (v1.1.17): keep
        # the joint-sync path OFF so wheel visuals cannot touch the solver.
        if self._use_instanced_wheels:
            renders = False        # only gates enable_wheel_joint_internal_sync below
        if self.solver == "batched":
            # Group same-kind vehicles (lazy/dirty) and give each kind one shared
            # cfg so MultiVehiclePhysics batches them. dual_scene → step() injects
            # the raycast-scene distances; single_scene → the MVP reads each sensor.
            # (No vehicles → no MVP; the scene can still hold static/dynamic bodies.)
            if self._vehicles:
                self._ensure_grouped()
                for veh in self._vehicles:
                    veh._group_cfg.enable_wheel_joint_internal_sync = renders
                self._build_mvp()
            # veh.physics stays None in batched mode; the shared solver is vs.physics.
        else:
            for veh in self._vehicles:
                veh.cfg.enable_wheel_joint_internal_sync = renders   # auto-managed (see above)
                sensor = None if self._dual_scene else veh.sensor
                veh.physics = VehiclePhysics(
                    self._main_scene, veh.entity_main, sensor, veh.cfg, n_envs=self.n_envs)

        # Wire the instanced wheel renderer now that physics exists: one unit
        # per KIND in batched mode (ONE closed-form call per kind per step —
        # per-vehicle calls would recompute the whole kind, O(K^2)), one per
        # vehicle in per_vehicle mode.
        if self._use_instanced_wheels:
            from .visual import InstancedWheelRenderer
            # Node-swap is needed only for the NATIVE viewer (its classic
            # renderer never re-uploads in-place poses writes); camera-only
            # scenes keep the cheap in-place update.
            r = InstancedWheelRenderer(self._main_scene,
                                       node_swap=self.show_viewer)
            if self.solver == "batched" and self._mvp is not None:
                veh_by_entity = {id(v.entity_main): v for v in self._vehicles}
                for kind in self._mvp.kinds:
                    veh0 = veh_by_entity[id(kind.entities[0])]
                    r.add_unit(self._instanced_wheel_meshes[veh0],
                               kind.wheel_visual_transforms, K=kind.K)
            else:
                for veh in self._vehicles:
                    r.add_unit(self._instanced_wheel_meshes[veh],
                               veh.physics.wheel_visual_transforms, K=1)
            self._wheel_renderer = r
            self._instanced_wheel_meshes = {}     # handed over; free the refs
            print(f"[genesis_vehicle] wheel visuals: instanced renderer "
                  f"({len(r._units)} kind(s); solver-free)")
        elif renders_pre and self._vehicles:
            print("[genesis_vehicle] wheel visuals: internal_sync "
                  "(WheelJointInternalSync fallback)")

        # Native viewer: make the per-frame render state atomic. Stock Genesis
        # sets the follow camera OUTSIDE the render lock and the node poses
        # inside it, so the async draw thread can pair a fresh camera with last
        # step's body pose — the followed vehicle "trembles" fore/aft at speed.
        # The patched viewer.update commits wheel buffers (_gv_pre_draw), the
        # follow camera and the node poses in ONE lock hold. (v1.1.25)
        self._viewer_atomic_wheels = False
        if self.viewer is not None:
            from .visual import patch_viewer_atomic_update
            if patch_viewer_atomic_update(self.viewer):
                if self._wheel_renderer is not None:
                    self.viewer._gv_pre_draw = self._wheel_renderer.update
                    self._viewer_atomic_wheels = True
            else:
                print("[genesis_vehicle] WARN: viewer internals unrecognized - "
                      "atomic camera/pose update patch skipped (a followed "
                      "vehicle may show slight draw-lag tremble)")

        # Apply any per-obstacle mass overrides now that entities are built.
        for entity, mass in self._pending_mass:
            entity.set_mass(mass)
        self._built = True
        # Resolved timing, logged on every run (v1.0.19) — dt is the one value
        # perf reports keep needing; make it impossible to miss.
        try:
            _ss = int(self.sim_options.substeps)
            print(f"[genesis_vehicle] timing: dt={self.dt * 1e3:.1f}ms "
                  f"({1.0 / self.dt:.0f}Hz) x substeps={_ss} "
                  f"(internal {self.dt / _ss * 1e3:.2f}ms) | n_envs={self.n_envs} "
                  f"| raycast_mode={self.raycast_mode}")
        except Exception:
            pass

    def _measure_distances(self) -> dict:
        """Internal: return ``{vehicle: wheel-ground distances}`` for the step.

        Raywheel: mirror each chassis pose onto its proxy + sync obstacle
        mirrors, step the raycast scene ONCE (re-cast against the static BVH),
        read each sensor → ``(n_envs, n_wheels)``. Inline: ``{vehicle: None}``.

        NOT a getter — it re-syncs poses and CASTS (advances the raycast scene),
        so it must run exactly once per :meth:`step`. ``step`` is the only caller.
        """
        self._require_built()
        if not self._dual_scene:
            return {veh: None for veh in self._vehicles}
        has_mirrors = any(o.entity_raycast is not None for o in self._dynamics)
        if self._vehicles or has_mirrors:
            if self._proxy_sync_ok is not False:
                try:
                    self._sync_proxies_batched()
                    self._proxy_sync_ok = True
                except Exception:
                    self._proxy_sync_ok = False
                    _logger.warning(
                        "[genesis_vehicle:proxy-sync] batched raycast-mirror sync "
                        "failed; falling back to the per-body loop (correct but "
                        "~1 ms/body slower).", exc_info=True)
            if self._proxy_sync_ok is False:
                for veh in self._vehicles:
                    veh._sync_proxy()
                for obs in self._dynamics:
                    if obs.entity_raycast is not None:
                        self._sync_dynamic(obs)
        # update_visualizer=False: the raycast scene is sensors-only and never
        # user-rendered, so skip the per-step visualizer/render update — the
        # sensor re-cast still runs inside sim.step(). Saves the render call the
        # raycast scene was making every frame.
        self._raycast_scene.step(update_visualizer=False)
        return {veh: read_distances(veh.sensor, self.n_envs)
                for veh in self._vehicles}

    def _sync_proxies_batched(self) -> None:
        """Mirror ALL raycast-scene followers — the K vehicle proxies AND every
        dynamic obstacle's raycast mirror — in one batched solver write + a
        SINGLE forward-kinematics pass (proxies v1.0.11; mirrors joined
        v1.0.13; both live in the raycast scene's rigid solver).

        The per-body loops (``Vehicle._sync_proxy`` / ``_sync_dynamic``) pay a
        fixed engine-entry overhead per call AND a whole-raycast-scene FK per
        ``set_pos``/``set_quat`` — 2 FK passes PER BODY (~1 ms/body: measured
        29.8 ms for 30 proxies alone). This method reads every followed base
        pose from the main solver in one batched ``get_links_pos/quat`` and
        writes them all via ``set_base_links_pos(skip_forward=True)`` +
        ``set_base_links_quat`` (which runs FK once for the whole batch).

        Pose semantics are identical to the loops: read the USER-frame base
        pose (what ``entity.get_pos()`` returns), write it WORLD-frame onto
        the follower (``relative=False``). ``_measure_distances`` falls back
        to the per-body loops (with a one-time warning) if this raises.
        """
        if self._proxy_sync_cache is None:
            main_idx = [v.entity_main.base_link_idx for v in self._vehicles]
            rc_idx = [v.proxy.base_link_idx for v in self._vehicles]
            mirrors = [o for o in self._dynamics if o.entity_raycast is not None]
            main_idx += [o.entity_main.base_link_idx for o in mirrors]
            rc_idx += [o.entity_raycast.base_link_idx for o in mirrors]
            rc_solver = (self._vehicles[0].proxy if self._vehicles
                         else mirrors[0].entity_raycast)._solver
            self._proxy_sync_cache = (main_idx, rc_idx, rc_solver)
        main_idx, rc_idx, rc_solver = self._proxy_sync_cache
        solver = self.rigid_solver
        pos = solver.get_links_pos(main_idx, relative=True)     # (n_envs, B, 3)
        quat = solver.get_links_quat(main_idx, relative=True)   # (n_envs, B, 4)
        rc_solver.set_base_links_pos(pos, rc_idx, relative=False,
                                     skip_forward=True)
        rc_solver.set_base_links_quat(quat, rc_idx, relative=False,
                                      skip_forward=False)       # ONE FK pass

    def step(self) -> None:
        """One simulation step.

        Raywheel mode: sync proxies/obstacle mirrors + re-cast the static raycast
        scene, feed each vehicle's main-scene physics, advance the main scene.
        Inline mode: each vehicle reads its own sensor.
        """
        self._require_built()
        if self.solver == "batched" and self._vehicles and self._ensure_grouped():
            self._build_mvp()   # config changed since last group → regroup + rebuild
        dists = self._measure_distances()   # dual: {veh:(N,n)} ; single: {veh:None}
        if self.solver == "batched":
            if self._mvp is not None:       # None → no vehicles; just advance the scene
                inputs_list = [veh._inputs for veh in self._vehicles]
                # dual_scene: inject the per-vehicle raycast-scene distances into the
                # batched compute; single_scene: None → the MVP reads each sensor.
                inj = [dists[veh] for veh in self._vehicles] if self._dual_scene else None
                self._mvp.step(inputs_list, distances=inj)
        else:
            for veh in self._vehicles:
                veh.physics.step(veh._inputs, distances=dists[veh])
        if self._wheel_renderer is not None:
            # Solver-free wheel visuals (v1.1.17): the closed-form wheel poses
            # must reach the renderer in the SAME frame as the chassis pose,
            # or the async viewer thread draws a mixed state and the vehicle
            # appears to tremble (v1.1.25). With the atomic viewer patch the
            # buffers stream inside viewer.update()'s render-lock hold
            # (_gv_pre_draw, together with the follow camera and node poses);
            # without a viewer (offscreen cameras render synchronously) or if
            # the patch declined, stream them between the physics step and the
            # visualizer update — the best non-atomic ordering.
            self._main_scene.step(update_visualizer=False)
            if not self._viewer_atomic_wheels:
                self._wheel_renderer.update()
            self._main_scene.visualizer.update(force=False, auto=True)
        else:
            self._main_scene.step()

    def reset(self) -> None:
        """Reset per-vehicle physics state and re-sync proxies. (Full scene
        re-randomization is out of scope for this first cut.)"""
        self._require_built()
        for veh in self._vehicles:
            if veh.physics is not None:
                veh.physics.reset()
            if self._dual_scene:
                veh._sync_proxy()

    # ---- kind grouping for the batched solver (lazy + dirty-tracked) ----
    def mark_config_dirty(self) -> None:
        """Mark the vehicle/cfg configuration changed so the batched solver
        re-groups before the next ``step`` (e.g. after you mutate a vehicle's cfg
        post-build). Cheap — re-grouping only happens on the next dirty step."""
        self._config_version += 1

    def _ensure_grouped(self) -> bool:
        """If the config changed since the last grouping, re-group vehicles into
        kinds (same ``_kind_key`` → one kind) and give each kind ONE shared cfg
        object so the batched solver batches them. Returns True iff it re-grouped;
        O(1) (an int compare) when nothing changed."""
        if self._config_version == self._grouped_version:
            return False
        groups: dict = {}
        for veh in self._vehicles:
            groups.setdefault(veh._kind_key, []).append(veh)
        for kind_vehicles in groups.values():
            shared = kind_vehicles[0].cfg          # one cfg object per kind
            for veh in kind_vehicles:
                veh._group_cfg = shared
        self._grouped_version = self._config_version
        return True

    def _build_mvp(self) -> None:
        """(Re)construct the batched MultiVehiclePhysics from the current groups."""
        from .multi_vehicle import MultiVehiclePhysics
        self._mvp = MultiVehiclePhysics(
            self._main_scene,
            [(veh.entity_main, veh.sensor, veh._group_cfg) for veh in self._vehicles],
            n_envs=self.n_envs)

    # ---- accessors ----
    @property
    def scene(self):
        """The underlying main ``gs.Scene`` — the escape hatch for Genesis
        APIs the wrapper does not re-export (e.g. ``draw_debug_line`` /
        ``draw_debug_spheres`` overlays after ``build()``). Read-only
        access; entity registration should still go through the
        ``add_*`` methods so bodies participate in wheel-raycast/proxy
        bookkeeping."""
        return self._main_scene

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

    @property
    def physics(self):
        """The batched ``MultiVehiclePhysics`` when ``solver="batched"`` (else
        ``None``; in ``per_vehicle`` mode each ``Vehicle.physics`` is its own
        ``VehiclePhysics``). Use it for batched closed-form visuals, e.g.
        ``vs.physics.wheel_visual_transforms()``."""
        return self._mvp

    # ---- thin scene accessors ----
    # VehicleScene owns the underlying Scene(s); the raw scenes are NOT public
    # (use add_* / step / add_camera). These narrow accessors cover the few
    # legitimate reads/tweaks external code still needs.
    @property
    def is_dual_scene(self) -> bool:
        """True in ``raycast_mode="dual_scene"`` (a separate raycast scene exists)."""
        return self._dual_scene

    @property
    def viewer(self):
        """The Genesis native viewer (``view="native"``), or ``None`` when headless
        / cv2. Use it to poll liveness or move the viewer camera."""
        return getattr(self._main_scene, "viewer", None)

    @property
    def rigid_solver(self):
        """The main scene's rigid solver — for read-only sim introspection
        (``n_geoms`` / ``n_links`` / ``faces_info`` …). Valid after ``build()``."""
        return self._main_scene.sim.rigid_solver

    @property
    def sim_options(self):
        """The main scene's ``sim_options`` — for the runtime physics tweaks the
        server makes (``dt`` / ``gravity``)."""
        return self._main_scene.sim_options

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
