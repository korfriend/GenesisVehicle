"""The SDK's single inventory of engine-private or engine-fragile Genesis reads.

Since genesis_vehicle 2.0.5 the SDK requires genesis-world 1.5.0 (``>=1.5.0,
<1.6``) and is written directly against it: every function here has ONE code
path, for 1.5.0. Through 2.0.4 this module branched between engine versions
(``<= 1.3.3`` force / inertial / ``Scene.sim_options`` spellings, warn-only
floor); those branches were removed in 2.0.5. What stays here is what reaches
past the engine's public surface, or what a later engine is likely to move, so
that a future break is fixed in one place:

``require_genesis`` / ``GenesisVersionError`` / ``GENESIS_FLOOR``
    The required-engine check, called right after ``import genesis`` in the
    Genesis-dependent SDK modules (``core``, ``vehicle_scene``,
    ``multi_vehicle``, ``raycast``, ``scene_helpers``, ``visual`` and
    ``server/__init__``). Never called at module level here, in
    ``_pipeline``, ``dynamics`` or ``control/*`` (``import genesis_vehicle``
    reaches those, and Genesis-free use must keep importing).

``apply_links_wrench``
    One call to ``RigidSolver.apply_links_external_wrench``; kept as a function
    so the server's per-instance monkey patch (``server/vehicle_builder``) is
    resolved per call.

``link_inertial`` / ``LinkInertialProps`` / ``check_link_inertial_api``
    Reads of ``link.desc`` (mass, inertial frame, inertia). The check RAISES if
    a field is missing, so an engine rename cannot read as zero inertia.

``sensor_miss_value``
    Reads the raycaster's private ``_options``.

``kinematic_raycast_entity``
    The dual_scene raycast mirror of a static surface, built on the 1.5.0 entity
    spec ``gs.entities.Kinematic(use_visual_raycasting=True)``.

``variant_index``
    The env -> variant mapping of a heterogeneous (per-env variant) entity: an
    ENGINE-PRIVATE read of the engine's own ``_balanced_variant_mapping``,
    imported at call time; raises if it is gone.

Wheel rendering (2.0.6, engine-node override; every target first appeared in
genesis-world 1.5.0, and a 1.5.x patch is admitted silently by ``require_genesis``)
    Checks: ``check_render_seg_api`` (at build, when the main scene renders) and
    ``check_render_fallback_api`` (before the first fallback node); both RAISE.
    ENGINE-PRIVATE READS: ``ctx.nodes_by_seg_keys`` (``engine_part_nodes``),
    ``ctx.rendered_envs_idx`` (``rendered_envs``), ``ctx.segmentation_level``,
    ``SEGMENTATION_LEVELS``, ``vgeom.desc.dynamic_vertices``
    (``part_is_posed``), ``ctx.jit._buffer`` (``queued_model``, wiring self-test
    only), ``vgeom.seg_keys`` / ``geom.seg_keys``, ``ctx.__dict__["_sync_shapes"]``
    (``on_render_sync`` / ``render_sync_depth``), ``node.mesh.primitives`` and
    ``primitive.poses`` (the renderer's per-sync ordering guard, and the rows a
    non-finite pose keeps), the module constant
    ``genesis.vis.rasterizer_context.POSE_MAX_ABS`` (``pose_max_abs``; the
    engine's own invalid-pose bound, read since the 2.0.6 fix round); the CALL
    ``ctx.update(force_render=True)`` (``force_render_sync``, wiring self-test).
    WRITES into engine-owned
    objects: the instance attribute ``ctx._sync_shapes`` (``on_render_sync``
    wraps it); ``primitive.poses``, ``mesh._bounds`` and the queued ``"model"``
    buffer of the ENGINE's wheel nodes (``queue_part_poses``); segmentation-map
    registration of SDK fallback nodes (``register_fallback_seg`` ->
    ``ctx.create_node_seg``). Kept: ``ctx.add_node`` (``add_render_node``,
    fallback nodes) and ``pyrender.Mesh.from_trimesh(envs=)`` (fallback nodes,
    called from ``visual``). Removed in 2.0.6:
    ``hide_vgeoms`` (the private ``vgeom.active_envs_mask`` write) and
    ``queue_instance_poses`` (generalised by ``queue_part_poses``).

``COARSE_SDF_MATERIAL_KW`` / ``coarse_sdf_nonconvex_geoms`` /
``warn_coarse_sdf_on_nonconvex``
    The SDF policy for SDK-owned collision bodies; reads
    ``collider.collider_config``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, NamedTuple, Optional

import numpy as np

__all__ = ["apply_links_wrench", "link_inertial",
           "LinkInertialProps", "check_link_inertial_api",
           "sensor_miss_value", "GENESIS_FLOOR", "GENESIS_UNVALIDATED_FROM",
           "GenesisVersionError", "require_genesis",
           "add_render_node", "queue_part_poses", "check_render_seg_api",
           "check_render_fallback_api", "rendered_envs", "engine_part_nodes",
           "part_is_posed", "pose_max_abs", "register_fallback_seg", "on_render_sync",
           "render_sync_depth", "force_render_sync", "queued_model",
           "WHEEL_RENDER_TAG", "RENDER_TESTED_GENESIS",
           "kinematic_raycast_entity", "variant_index",
           "COARSE_SDF_MATERIAL_KW",
           "coarse_sdf_nonconvex_geoms", "warn_coarse_sdf_on_nonconvex"]

_logger = logging.getLogger("genesis_vehicle")


def apply_links_wrench(solver: Any, force: Any, torque: Any,
                       links_idx: Any = None,
                       envs_idx: Optional[Any] = None) -> None:
    """Apply a world-frame force **and** torque to ``links_idx`` in one
    ``solver.apply_links_external_wrench`` call (genesis-world 1.5.0; the
    engine's default ``ref="link_origin"``, ``local=False``,
    ``accumulate=True``).

    ``force`` / ``torque`` are ``(n_envs, n_links, 3)`` (or ``(n_links, 3)`` for
    an unbatched scene), matching what the engine call expects.

    Resolved per call (not cached) so the server's monkey patch in
    ``server/vehicle_builder.apply_monkey_patches`` -- which rebinds this
    attribute on the solver instance -- stays in effect.
    """
    solver.apply_links_external_wrench(force=force, torque=torque,
                                       links_idx=links_idx, envs_idx=envs_idx)


class LinkInertialProps(NamedTuple):
    """A link's authored inertial frame.

    ``i`` is the 3x3 inertia tensor expressed in the inertial frame, which
    ``quat`` orients relative to the link frame -- so the link-frame tensor is
    ``R(quat) @ i @ R(quat).T``. (Genesis diagonalizes an authored tensor onto
    its principal axes and carries the rotation in ``quat``.)"""
    mass: float
    pos: Any                                        # (3,) COM offset, link frame
    quat: Any                                       # (4,) wxyz, inertial frame
    i: Any                                          # (3, 3) inertia, inertial frame


def link_inertial(link: Any) -> Optional[LinkInertialProps]:
    """Read a link's inertial properties from its resolved description
    (``link.desc.mass / .inertial_pos / .inertial_quat / .inertia``), or None
    if the link carries no mass (or has no ``desc``)."""
    desc = getattr(link, "desc", None)
    if desc is None:
        return None
    mass = getattr(desc, "mass", None)
    pos, quat = getattr(desc, "inertial_pos", None), getattr(desc, "inertial_quat", None)
    inertia = getattr(desc, "inertia", None)
    if mass is None:
        return None
    mass = float(mass)
    if mass <= 0.0:
        return None
    pos = np.zeros(3) if pos is None else np.asarray(pos, dtype=float).reshape(3)
    quat = None if quat is None else np.asarray(quat, dtype=float).reshape(4)
    inertia = None if inertia is None else np.asarray(inertia, dtype=float).reshape(3, 3)
    return LinkInertialProps(mass, pos, quat, inertia)


#: The ``link.desc`` field names :func:`check_link_inertial_api` requires.
_INERTIAL_FIELDS = ("mass", "inertial_pos", "inertial_quat", "inertia")


def check_link_inertial_api(link: Any) -> None:
    """Raise ``AttributeError`` unless ``link.desc`` exposes ALL FOUR inertial
    fields :func:`link_inertial` reads (``mass`` / ``inertial_pos`` /
    ``inertial_quat`` / ``inertia``; their VALUES may be ``None``: a massless
    link passes). The message names the missing ones.

    Why every field and not only the container: these values feed the physics
    (the yaw-inertia inputs of the low-speed deadbeat bounds, v1.8.0).
    :func:`link_inertial` reads each field with ``getattr(..., None)``, so a
    renamed ``inertia`` would read as zero link inertia and a renamed
    ``inertial_quat`` would skip the principal-axis rotate-back -- a silently
    wrong yaw inertia, no error. And a link with no ``desc`` would read as
    massless, so the caller would report "no mass" instead of "unknown engine".
    Call it on the ROOT link before a mass walk. A guard against a future
    engine rename, not a version branch (2.0.5 removed the ``<= 1.3.3``
    ``inertial_*`` spelling this used to accept)."""
    desc = getattr(link, "desc", None)
    if desc is None:
        raise AttributeError(
            f"{type(link).__name__} exposes no 'desc'; this genesis version's link "
            "inertial values are not recognized by genesis_vehicle._gs_compat."
            "link_inertial (written against genesis-world 1.5.0)")
    missing = [n for n in _INERTIAL_FIELDS if not hasattr(desc, n)]
    if missing:
        raise AttributeError(
            f"{type(link).__name__}: link.desc lacks {missing} (expected all of "
            f"{list(_INERTIAL_FIELDS)}); this genesis version's link inertial values "
            "are not recognized by genesis_vehicle._gs_compat.link_inertial")


def sensor_miss_value(sensor: Any) -> Optional[float]:
    """The distance ``sensor`` reports for a ray that hit NOTHING (the "miss
    sentinel"), or ``None`` if this sensor does not expose one.

    Genesis resolves the value onto the sensor's own options object:
    ``Raycaster.no_hit_value``, which ``model_post_init`` fills with
    ``max_range`` when the user leaves it unset. That happens at sensor
    CONSTRUCTION, so this read is valid both before and after ``scene.build()``
    -- which matters because the SDK stamps it at ``add_sensor`` time. An
    options object that only carries ``max_range`` falls back to it; a stub
    sensor with neither returns ``None``; the SDK's miss predicates then raise
    ``ValueError`` (v1.8.0).

    The ``is None`` test below is deliberate and must not become ``or``: a
    ``no_hit_value`` of ``0.0`` is legitimate and would otherwise silently fall
    through to ``max_range``.

    NOT sourced from ``sensor._shared_metadata.no_hit_values``: that tensor is
    empty before ``build()`` and is a scene-wide concatenation over EVERY
    raycaster in the scene, so indexing it correctly needs the sensor's slot.
    """
    options = getattr(sensor, "_options", None)
    if options is None:
        return None
    value = getattr(options, "no_hit_value", None)
    if value is None:
        value = getattr(options, "max_range", None)
    if value is None:
        return None
    return float(value)


# ---------------------------------------------------------------------------
# Required engine version (2.0.5: genesis-world >= 1.5.0, < 1.6)
# ---------------------------------------------------------------------------

#: The oldest genesis-world release this SDK runs on (REQUIRED since 2.0.5;
#: through 2.0.4 a warn-only floor of 1.4.5). The matching dependency set is
#: quadrants 1.4.0, coacd >= 1.0.14, PyGEL3D >= 0.8.0, Python >= 3.12 (the
#: 1.5.0 wheel's own Requires-Dist).
GENESIS_FLOOR = (1, 5, 0)

#: First release outside the validated range ``>=1.5.0,<1.6``: from here on
#: :func:`require_genesis` logs one WARNING and continues. A 1.5.x patch release
#: (1.5.1, ...) is admitted with no warning and is NOT validated either.
GENESIS_UNVALIDATED_FROM = (1, 6, 0)


class GenesisVersionError(ImportError):
    """genesis-world is older than the SDK's required floor (2.0.5).

    An ``ImportError`` subclass, because it is raised while a Genesis-dependent
    SDK module is being imported. NB a caller's ``try: ... except ImportError:``
    (the optional-dependency idiom) also catches it, so :func:`require_genesis`
    logs the message at ERROR on the ``genesis_vehicle`` logger, once per
    process, before raising."""


_floor_error_logged = False
_unvalidated_warned = False
_unparsed_warned = False

_VERSION_RE = re.compile(
    r"^\s*v?(\d+)\.(\d+)(?:\.(\d+))?"          # release: major.minor[.patch]
    r"(?:[.\-_]?(a|b|rc|alpha|beta|c|pre|preview|dev)\.?\d*(?=$|[.+]))?",   # pre-release tag
    re.IGNORECASE)


def _parse_genesis_version(text: Any) -> Optional[tuple]:
    """``(major, minor, patch, is_prerelease)`` or ``None`` if unparseable.

    A missing patch reads as 0 (``"1.5"`` -> ``(1, 5, 0)``). A post-release
    (``1.5.0.post1``), a local tag (``+g1234``) or a build suffix such as
    ``-cuda12`` / ``_cpu`` does not count as pre-release; ``a`` / ``b`` /
    ``rc`` / ``dev`` do -- ``1.5.0rc1`` sorts BEFORE ``1.5.0``."""
    if not isinstance(text, str):
        return None
    m = _VERSION_RE.match(text)
    if m is None:
        return None
    major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    return major, minor, patch, m.group(4) is not None


def _floor_message(found: Any) -> str:
    from ._version import __version__
    return (f"genesis_vehicle {__version__} requires genesis-world >= 1.5.0, < 1.6 "
            f"(found {found}). Upgrade: pip install 'genesis-world>=1.5.0,<1.6' "
            "(pulls quadrants 1.4.0, coacd >= 1.0.14, PyGEL3D >= 0.8.0; needs "
            "Python >= 3.12), or stay on genesis_vehicle 2.0.4 for genesis-world 1.4.5.")


def require_genesis(gs_module: Any = None) -> None:
    """Raise :class:`GenesisVersionError` if genesis-world is older than
    ``GENESIS_FLOOR`` (1.5.0); a 1.5.0 pre-release (``1.5.0rc1``) counts as
    older.

    Called right after ``import genesis`` at module level in ``core``,
    ``vehicle_scene``, ``multi_vehicle``, ``raycast``, ``scene_helpers``,
    ``visual`` and ``server/__init__`` -- so any SDK import that needs the engine
    (``VehicleScene``, ``VehiclePhysics``, ``MultiVehiclePhysics``,
    ``scene_helpers``, the server) fails at import, before a scene exists.
    ``genesis_vehicle.kinematics`` (imports ``genesis.utils.geom`` at module
    level), ``urdf`` (no genesis import; ``estimate_spin_inertia_from_genesis``
    reads a built entity through ``link_inertial``) and
    ``control.sweep_measure`` (imports genesis
    lazily; its ``VehicleScene`` import hits the check when a sweep builds a
    scene) are not guarded on their own.
    ``import genesis_vehicle`` itself, ``PathFollower`` and ``SweepTable`` are
    not blocked (genesis must still be importable: ``control`` reaches
    ``genesis.utils.geom``).

    Before raising, the message is logged at ERROR on the ``genesis_vehicle``
    logger, ONCE per process (a caller's ``except ImportError`` would otherwise
    hide it).

    Other outcomes: ``>= 1.6.0`` logs one WARNING per process ("not validated")
    and continues; an unparseable or missing version logs one WARNING per
    process and continues. A 1.5.x patch release (1.5.1, ...) is silent and is
    not validated."""
    global _floor_error_logged, _unvalidated_warned, _unparsed_warned
    if gs_module is None:
        import genesis as gs_module                 # noqa: PLC0415 - lazy on purpose
    raw = getattr(gs_module, "__version__", None)
    parsed = _parse_genesis_version(raw)
    if parsed is None:
        if not _unparsed_warned:
            _unparsed_warned = True
            _logger.warning(
                "[genesis_vehicle:genesis-version] could not parse genesis-world "
                "version %r; the required >= 1.5.0, < 1.6 was not checked. "
                "Continuing.", raw)
        return
    below = parsed[:3] < GENESIS_FLOOR or (parsed[:3] == GENESIS_FLOOR and parsed[3])
    if below:
        msg = _floor_message(raw)
        if not _floor_error_logged:
            _floor_error_logged = True
            _logger.error("[genesis_vehicle:genesis-version] %s", msg)
        raise GenesisVersionError(msg)
    if parsed[:3] >= GENESIS_UNVALIDATED_FROM and not _unvalidated_warned:
        _unvalidated_warned = True
        from ._version import __version__
        _logger.warning(
            "[genesis_vehicle:genesis-version] genesis-world %s is not validated "
            "with genesis_vehicle %s (validated: >= 1.5.0, < 1.6). Continuing.",
            raw, __version__)


# ---------------------------------------------------------------------------
# dual_scene raycast mirror on the 1.5.0 entity spec
# ---------------------------------------------------------------------------

def kinematic_raycast_entity(scene: Any, morph_or_morphs: Any, surface: Any = None) -> Any:
    """Add a visual-raycasting KINEMATIC body (no collision, no dynamics) built
    from ``morph_or_morphs`` to ``scene`` and return the entity:
    ``scene.add_entity(spec=gs.entities.Kinematic(morph=..., surface=...,
    use_visual_raycasting=True))`` (genesis-world 1.5.0 entity spec).

    ``morph_or_morphs`` is one morph, or a sequence of morphs (passed as a
    tuple: a heterogeneous entity, one variant per env). ``surface`` is copied
    with ``vis_mode=None``: the Kinematic spec raises on a non-"visual"
    ``vis_mode`` (e.g. the server's ``--vis_mode collision`` leaking through),
    and "visual" itself is deprecated on 1.5.0. No ``visualize_contact`` (the
    spec has no such field) and no material ``use_visual_raycasting``.

    Through 2.0.4 the SDK passed ``material=gs.materials.Kinematic(
    use_visual_raycasting=True), vis_mode="visual"``, which 1.5.0 accepts with
    two deprecation warnings."""
    import genesis as gs                             # noqa: PLC0415
    morph = (tuple(morph_or_morphs) if isinstance(morph_or_morphs, (list, tuple))
             else morph_or_morphs)
    kw = dict(morph=morph, use_visual_raycasting=True)
    if surface is not None:
        if getattr(surface, "vis_mode", None) is not None:
            surface = surface.model_copy(update={"vis_mode": None}, deep=True)
        kw["surface"] = surface
    return scene.add_entity(spec=gs.entities.Kinematic(**kw))


def variant_index(n_variants: int, n_envs: int) -> Any:
    """The env -> variant mapping genesis-world 1.5.0 applies to a heterogeneous
    entity of ``n_variants`` morphs in a scene built with ``n_envs`` envs, as a
    ``torch.long`` tensor of shape ``(max(n_envs, 1),)``.

    The engine exposes no read-back of that mapping; this calls the engine's
    OWN private rule, ``genesis.engine.entities.rigid_entity.description.
    _balanced_variant_mapping(n_variants, B)`` (contiguous blocks of nearly
    equal size; B = ``max(n_envs, 1)``), imported at call time, so the SDK never
    re-implements it. If an engine renames or drops it, this RAISES
    ``ImportError`` naming the symbol (a re-implemented guess could silently
    disagree with where the engine put each variant). ENGINE-PRIVATE read."""
    try:
        from genesis.engine.entities.rigid_entity.description import (  # noqa: PLC0415
            _balanced_variant_mapping)
    except ImportError as e:
        raise ImportError(
            "variant_index: genesis.engine.entities.rigid_entity.description."
            "_balanced_variant_mapping (the engine's env -> variant rule, "
            "genesis-world 1.5.0) is not available in this engine, so the SDK "
            "cannot tell which variant each env holds.") from e
    import torch                                     # noqa: PLC0415
    mapping = _balanced_variant_mapping(int(n_variants), max(int(n_envs), 1))
    return torch.as_tensor(np.asarray(mapping), dtype=torch.long)


# ---------------------------------------------------------------------------
# Wheel rendering (2.0.6, "engine-node override"): rasterizer-context access
# ---------------------------------------------------------------------------
#
# Every target below first appeared in genesis-world 1.5.0 (the 1.4.5
# rasterizer context was rewritten) and a 1.5.x patch release is admitted by
# require_genesis with no warning, so a same-name meaning change is the risk.
# The structure checks (check_render_seg_api / check_render_fallback_api) catch
# renames and signature changes; the ordering of the SDK's write after the
# engine's is checked at run time by the renderer, ONE SYNC LATE, and only for
# writes to primitive.poses (see visual.InstancedWheelRenderer._check_last_writes;
# an engine "model" buffer queued after the hook is covered only by the flush
# test tests/test_render_sync_pose.py). The check compares the poses written by
# the last COMPLETED hooked sync; after a hooked sync that did not complete (an
# exception or a BaseException anywhere in the wrapper) the next hooked sync is
# not checked and checking resumes on the one after. A non-finite closed-form
# pose is not an error: that wheel instance keeps the row the SDK last wrote
# (the engine's POSE_MAX_ABS rule, pose_max_abs), with no warning.

#: Prefix of every error / warning raised by the wheel render path. Hook
#: errors raised on the native viewer's thread surface there only as
#: "Unexpected viewer error", so the prefix names the source.
WHEEL_RENDER_TAG = "[genesis_vehicle:wheel-render]"

#: The genesis-world release the wheel render path was written and measured on.
RENDER_TESTED_GENESIS = "1.5.0"


def _render_api_error(what: str) -> RuntimeError:
    import genesis as gs                             # noqa: PLC0415
    return RuntimeError(
        f"{WHEEL_RENDER_TAG} {what} (genesis-world {getattr(gs, '__version__', '?')}; "
        f"the SDK's wheel renderer was written and tested against genesis-world "
        f"{RENDER_TESTED_GENESIS}). Rendering a VehicleScene (a camera or a viewer) "
        f"needs it; headless scenes (no camera, no viewer) are unaffected.")


def _require_attr(obj: Any, name: str, where: str) -> Any:
    """``getattr`` WITHOUT a default: a missing engine attribute raises the
    prefixed RuntimeError naming it (never a silent fallback value)."""
    try:
        return getattr(obj, name)
    except AttributeError:
        raise _render_api_error(f"{where} has no attribute {name!r}") from None


def check_render_seg_api(ctx: Any) -> None:
    """Structure check of the engine surface the wheel-pose override uses,
    called once at ``VehicleScene.build`` when the main scene renders. Raises
    ``RuntimeError`` naming the first missing / changed item.

    Checked: ``ctx.nodes_by_seg_keys`` is a dict; ``ctx._sync_shapes`` is
    callable; ``ctx.rendered_envs_idx`` is a sequence; ``ctx.jit.update_buffer``
    takes exactly ``(primitives, buffer_name, data)`` (the 1.4.5 signature was
    ``(node, buffer_name, data)``); ``pyrender.Mesh.from_trimesh`` takes
    ``envs``; ``RigidLink.vgeoms`` / ``.geoms`` and ``RigidVisGeom.seg_keys`` /
    ``RigidGeom.seg_keys`` exist. It checks existence and signatures only: a
    same-name change of MEANING (e.g. the engine's pose write moving after
    ``_sync_shapes``) is caught at run time by the renderer, not here. The
    fallback-only surface is checked by :func:`check_render_fallback_api`."""
    import inspect                                   # noqa: PLC0415
    nb = _require_attr(ctx, "nodes_by_seg_keys", "RasterizerContext")
    if not isinstance(nb, dict):
        raise _render_api_error(
            f"RasterizerContext.nodes_by_seg_keys is a {type(nb).__name__}, not a dict")
    if not callable(_require_attr(ctx, "_sync_shapes", "RasterizerContext")):
        raise _render_api_error("RasterizerContext._sync_shapes is not callable")
    ridx = _require_attr(ctx, "rendered_envs_idx", "RasterizerContext")
    if ridx is None or not hasattr(ridx, "__len__"):
        raise _render_api_error("RasterizerContext.rendered_envs_idx is not a sequence (context not built?)")
    jit = _require_attr(ctx, "jit", "RasterizerContext")
    ub = _require_attr(jit, "update_buffer", "RasterizerContext.jit")
    params = [p for p in inspect.signature(ub).parameters if p != "self"]
    if params != ["primitives", "buffer_name", "data"]:
        raise _render_api_error(
            f"RasterizerContext.jit.update_buffer takes {params}, expected "
            f"['primitives', 'buffer_name', 'data']")
    from genesis.ext import pyrender                 # noqa: PLC0415
    if "envs" not in inspect.signature(pyrender.Mesh.from_trimesh).parameters:
        raise _render_api_error("pyrender.Mesh.from_trimesh has no 'envs' parameter")
    from genesis.engine.entities import rigid_entity as _re   # noqa: PLC0415
    for cls_name, attrs in (("RigidLink", ("vgeoms", "geoms")),
                            ("RigidVisGeom", ("seg_keys",)),
                            ("RigidGeom", ("seg_keys",))):
        cls = _require_attr(_re, cls_name, "genesis.engine.entities.rigid_entity")
        for a in attrs:
            _require_attr(cls, a, cls_name)


def check_render_fallback_api(ctx: Any) -> None:
    """Structure check of the surface an SDK fallback node uses, called when the
    first fallback node is about to be made (a wheel the engine draws no node
    for, e.g. ``vis_mode="collision"``): ``ctx.create_node_seg(seg_key,
    seg_node)``, ``ctx.segmentation_level`` in ``SEGMENTATION_LEVELS``, and
    ``SEGMENTATION_LEVELS == SegmentationKeys._fields`` (the level names are
    also the key field names :func:`register_fallback_seg` reads)."""
    import inspect                                   # noqa: PLC0415
    cns = _require_attr(ctx, "create_node_seg", "RasterizerContext")
    params = [p for p in inspect.signature(cns).parameters if p != "self"]
    if params != ["seg_key", "seg_node"]:
        raise _render_api_error(
            f"RasterizerContext.create_node_seg takes {params}, expected ['seg_key', 'seg_node']")
    level = _require_attr(ctx, "segmentation_level", "RasterizerContext")
    from genesis.vis import rasterizer_context as _rc   # noqa: PLC0415
    levels = _require_attr(_rc, "SEGMENTATION_LEVELS", "genesis.vis.rasterizer_context")
    from genesis.engine.render_context import SegmentationKeys   # noqa: PLC0415
    if tuple(levels) != tuple(SegmentationKeys._fields):
        raise _render_api_error(
            f"SEGMENTATION_LEVELS {tuple(levels)} differ from SegmentationKeys fields "
            f"{tuple(SegmentationKeys._fields)}")
    if level not in levels:
        raise _render_api_error(f"segmentation_level {level!r} is not one of {tuple(levels)}")


def rendered_envs(ctx: Any) -> list:
    """The env indices the context renders (``ctx.rendered_envs_idx``); node
    instance r of an env-instanced node is env ``rendered_envs(ctx)[r]``."""
    return [int(i) for i in _require_attr(ctx, "rendered_envs_idx", "RasterizerContext")]


def engine_part_nodes(ctx: Any, part: Any) -> tuple:
    """The engine's render nodes of one rigid geom ``part`` (visual or
    collision), looked up by its segmentation keys in the engine-internal
    ``ctx.nodes_by_seg_keys``. ENGINE-PRIVATE read. An empty result is NOT a
    decision input for the caller (a key-form change would read as "no node");
    the renderer decides from the engine model which parts must have nodes and
    raises when one has none."""
    return tuple(ctx.nodes_by_seg_keys.get(_require_attr(part, "seg_keys", type(part).__name__), ()))


def part_is_posed(vgeom: Any) -> bool:
    """False when the visual geom draws ``DynamicVertices`` (its vertices are
    written in world space, so a pose override cannot apply): reads the
    ENGINE-PRIVATE ``vgeom.desc.dynamic_vertices``; raises if ``desc`` or the
    field is missing."""
    desc = _require_attr(vgeom, "desc", type(vgeom).__name__)
    return _require_attr(desc, "dynamic_vertices", "visual geom desc") is None


def pose_max_abs() -> float:
    """The engine's invalid-pose bound ``genesis.vis.rasterizer_context.
    POSE_MAX_ABS`` (1e20 on 1.5.0; ENGINE-PRIVATE read of a module constant,
    imported at call time, no default: a missing name raises the prefixed
    ``RuntimeError``). The engine keeps a node instance at the transform it was
    last drawn with when any entry of its new pose is not ``abs(x) < POSE_MAX_ABS``
    (a NaN fails that test); the wheel renderer applies the same rule to the
    closed-form poses it writes."""
    from genesis.vis import rasterizer_context as _rc   # noqa: PLC0415
    return float(_require_attr(_rc, "POSE_MAX_ABS", "genesis.vis.rasterizer_context"))


def queue_part_poses(ctx: Any, items: Any) -> None:
    """Pose render nodes: ``items`` is a sequence of ``(node, T)`` with ``T`` a
    ``(R, 4, 4)`` float32 C-contiguous world transform per rendered env.

    For every primitive of every node: ``mesh._bounds = None`` (ENGINE-PRIVATE
    write, mirroring the engine's own pose sync so the mesh bounds are
    recomputed), ``primitive.poses = T`` (WRITE; the renderer reads draw-sort
    centroids and shadow bounds from it), then ONE ``ctx.jit.update_buffer(
    primitives, "model", data)`` (WRITE into the engine's per-frame upload
    queue) with one TRANSPOSED array per primitive (1.5.0 uploads "model" with
    the translation in the last row). The queue is a dict keyed by (primitive,
    buffer), so the last write before the flush wins. Asserts the shapes: a bare
    ``(R, 4, 4)`` passed where a list is expected would be zipped row by row and
    silently pose one instance. The caller holds the viewer lock."""
    prims, data = [], []
    for node, T in items:
        if not (isinstance(T, np.ndarray) and T.dtype == np.float32 and T.ndim == 3
                and T.shape[1:] == (4, 4) and T.flags.c_contiguous):
            raise ValueError(
                f"{WHEEL_RENDER_TAG} queue_part_poses: each T must be a (R, 4, 4) float32 "
                f"C-contiguous array, got {type(T).__name__} "
                f"{getattr(T, 'shape', None)} {getattr(T, 'dtype', None)}")
        mesh = node.mesh
        mesh._bounds = None
        Tt = np.ascontiguousarray(T.transpose((0, 2, 1)))
        for prim in mesh.primitives:
            prim.poses = T
            prims.append(prim)
            data.append(Tt)
    assert len(data) == len(prims)
    if prims:
        ctx.jit.update_buffer(prims, "model", data)


def queued_model(ctx: Any, primitive: Any) -> Any:
    """The "model" buffer queued for ``primitive`` and not flushed yet, or None
    (ENGINE-PRIVATE read of ``ctx.jit._buffer``; used only by the renderer's
    wiring self-test)."""
    return _require_attr(ctx.jit, "_buffer", "RasterizerContext.jit").get((primitive, "model"))


def add_render_node(ctx: Any, obj: Any) -> Any:
    """Add a pyrender ``Mesh`` to the rasterizer context's scene and return its
    node (``RasterizerContext.add_node``, which takes the viewer lock itself).
    Used for fallback wheel nodes only (2.0.6)."""
    return ctx.add_node(obj)


def register_fallback_seg(ctx: Any, node: Any, seg_keys: Any) -> None:
    """Give an SDK fallback node the segmentation id of ``seg_keys`` at the
    context's level: ``ctx.create_node_seg(getattr(seg_keys,
    ctx.segmentation_level), node)`` (WRITE into the engine's segmentation
    maps). Deliberately NOT added to ``ctx.nodes_by_seg_keys``, so the engine
    and the override never treat it as an engine node. The integer it gets is
    assigned in registration order, after the engine's own keys."""
    ctx.create_node_seg(getattr(seg_keys, ctx.segmentation_level), node)


def on_render_sync(ctx: Any, fn: Any, *, before: Any = None) -> Any:
    """Run ``fn()`` after every engine node sync of ``ctx`` (and ``before()``
    ahead of it): replaces the INSTANCE attribute ``ctx._sync_shapes`` with a
    wrapper (ENGINE-PRIVATE WRITE). 1.5.0 calls ``_sync_shapes`` only from
    ``RasterizerContext.update`` (once per sync that passes its early return),
    inside ``with viewer_lock``, and the engine writes the poses of its mesh
    nodes only there; so ``fn`` sees -- and overrides -- every engine pose
    write of that sync, under the same lock (reentrant on the viewer).

    This is the enforcement point of the wheel-render contract, and it covers
    ONLY ``scene.visualizer.context``: the two syncs ``Visualizer.build`` makes
    before any SDK code runs, viewer frames drawn during build, camera SENSORS
    (``scene.add_sensor(gs.sensors.*CameraOptions)``; a rasterizer sensor uses
    its own context when there is no viewer), batch-renderer and raytracer
    sensors, and visual raycasting / picking are not hooked and show the URDF
    wheel at the solver's pose. The native viewer thread was not run here
    (no display); what it draws is inferred from the source.

    The wrapper depth is always 1: a second install on the same context adds
    ``fn`` to the existing wrapper, and installing an ``fn`` that is already
    there (compared with ``==``, so a bound method of the same object counts)
    raises ``RuntimeError``. An exception from ``fn`` / ``before`` is re-raised
    as ``RuntimeError`` carrying :data:`WHEEL_RENDER_TAG`. Returns the wrapper
    (its ``calls`` counter counts completed hooked syncs).

    Order inside one sync: every ``before()``, then the engine's original
    ``_sync_shapes``, then every ``fn()``; an exception at any point ends the
    sync there (``calls`` is not incremented). An exception from the original
    ``_sync_shapes`` propagates unwrapped. The renderer passes its ordering
    guard as ``before``: it compares, by value (``equal_nan=True``), the poses
    written by the last COMPLETED hooked sync with what the primitives hold
    now, and discards that record whether or not it raised; only a completed
    ``fn()`` records again. So an engine pose write that moved after this
    wrapper is detected ONE SYNC LATE (the frame of the sync where it happened
    is drawn at the solver's pose; the raise comes at the start of the next
    hooked sync; a frame with no later hooked sync is never checked); after a
    hooked sync that did not complete (the guard, the engine's
    ``_sync_shapes`` or ``fn`` raised, or a ``BaseException`` such as
    ``KeyboardInterrupt``) the next hooked sync is not checked and checking
    resumes on the one after, so a persistent late write raises on at least
    every second hooked sync (and while ``fn`` raises on every sync, its own
    exception surfaces on each); and an engine ``"model"`` buffer queued after
    the wrapper WITHOUT touching ``primitive.poses`` is not detected at run
    time at all (covered only by ``tests/test_render_sync_pose.py``'s flush
    test, to be run on every engine bump). A non-finite closed-form pose is
    not an error: that wheel instance keeps its last SDK-written row (see
    ``visual.InstancedWheelRenderer``)."""
    cur = ctx.__dict__.get("_sync_shapes")
    if cur is not None and getattr(cur, "_gv_render_hook", False):
        if any(f == fn for f, _ in cur._gv_fns):
            raise RuntimeError(f"{WHEEL_RENDER_TAG} on_render_sync: {fn!r} is already installed "
                               f"on this context (it would run twice per sync)")
        cur._gv_fns.append((fn, before))
        return cur
    original = _require_attr(ctx, "_sync_shapes", "RasterizerContext")
    fns = [(fn, before)]

    def _gv_sync_shapes(*args: Any, **kwargs: Any) -> Any:
        try:
            for _, b in fns:
                if b is not None:
                    b()
        except Exception as e:
            if str(e).startswith(WHEEL_RENDER_TAG):
                raise
            raise RuntimeError(f"{WHEEL_RENDER_TAG} {type(e).__name__}: {e}") from e
        out = original(*args, **kwargs)
        try:
            for f, _ in fns:
                f()
        except Exception as e:
            if str(e).startswith(WHEEL_RENDER_TAG):
                raise
            raise RuntimeError(f"{WHEEL_RENDER_TAG} {type(e).__name__}: {e}") from e
        _gv_sync_shapes.calls += 1
        return out

    _gv_sync_shapes._gv_render_hook = True
    _gv_sync_shapes._gv_original = original
    _gv_sync_shapes._gv_fns = fns
    _gv_sync_shapes.calls = 0
    ctx._sync_shapes = _gv_sync_shapes
    return _gv_sync_shapes


def render_sync_depth(ctx: Any) -> int:
    """Number of SDK wrappers stacked on ``ctx._sync_shapes`` (0 or 1)."""
    d, f = 0, ctx.__dict__.get("_sync_shapes")
    while f is not None and getattr(f, "_gv_render_hook", False):
        d += 1
        f = getattr(f, "_gv_original", None)
    return d


def force_render_sync(ctx: Any, viewer_lock: Any) -> None:
    """One forced node sync of ``ctx`` under ``viewer_lock``
    (``ctx.update(force_render=True)``); used by the renderer's wiring
    self-test."""
    with viewer_lock:
        ctx.update(force_render=True)


# ---------------------------------------------------------------------------
# Coarse SDF material kwargs for the collision bodies the SDK owns
# ---------------------------------------------------------------------------

#: ``gs.materials.Rigid(**COARSE_SDF_MATERIAL_KW)`` — the minimum SDF grid the
#: engine's validators accept (``ge=16`` on both ``sdf_min_res`` and
#: ``sdf_max_res``), for collision bodies whose SDF the SDK does not expect to be
#: read.
#:
#: What it does NOT do: it does not switch the SDF off. genesis-world 1.5.0 sizes
#: each geom's grid per axis as ``cell = clip(target, grid/(max_res-1),
#: grid/max(sdf_min_res-1, 2))`` with ``max_res = sdf_max_res`` (capped at 32 for
#: a primitive), where ``target`` is ``sdf_cell_size`` refined to half the convex
#: WIDTH of a convex geom (since 1.4.7) or to half the wall thickness of a
#: watertight non-convex one (``rigid_entity/description.py``
#: ``SignedDistanceFieldDescription.from_geometry``). With ``sdf_min_res =
#: sdf_max_res = 16`` both clip bounds are ``grid/15``, so EVERY geom carrying
#: these kwargs lands on 16 nodes per axis, whatever its shape.
#:
#: History, corrected in place (2.0.5): through 2.0.4 this dict had no
#: ``sdf_max_res`` and this comment said a convex geom always lands on 16 nodes
#: per axis because the huge ``sdf_cell_size`` is clipped to the upper bound.
#: That was true through genesis-world 1.4.6 only: 1.4.7 added the convex-width
#: refinement, which pulls a THIN convex geom (e.g. a 0.1 m wall) below the upper
#: bound and up to the default ``sdf_max_res`` (128, or 32 for a primitive) along
#: its thin axis. ``"sdf_max_res": 16`` restores the 16-node grid.
#:
#: Behaviour change in 2.0.5, stated: ``sdf_max_res`` 16 also caps the
#: wall-thickness refinement of a watertight NON-convex geom at 16 nodes per
#: axis (through 2.0.4 such a geom still refined up to 128). The post-build
#: check :func:`warn_coarse_sdf_on_nonconvex` names any non-convex geom carrying
#: these kwargs whose SDF a contact can read.
#:
#: Where the SDF is read: in a rigid-only scene, only by the non-convex vs
#: non-terrain narrowphase (gated on ``collider_config.has_nonconvex_nonterrain``);
#: terrain uses its heightfield and the SDK's ray-wheel contact never reads an
#: SDF. So the kwargs are only appropriate where every geom is convex.
#:
#: Why it matters: the collider computes EVERY geom's SDF at build (since
#: genesis-world 1.4.5), so the per-geom node count is paid in build time and
#: memory whether or not the SDF is ever read.
COARSE_SDF_MATERIAL_KW = {"sdf_cell_size": 1.0e4, "sdf_min_res": 16, "sdf_max_res": 16}


def _is_coarse_material(material: Any) -> bool:
    """True iff ``material`` carries ALL THREE ``COARSE_SDF_MATERIAL_KW`` values.
    A two-key copy made from the 2.0.4 dict (no ``sdf_max_res``) is NOT
    recognised as coarse since 2.0.5."""
    if material is None:
        return False
    return all(getattr(material, k, None) == v for k, v in COARSE_SDF_MATERIAL_KW.items())


def coarse_sdf_nonconvex_geoms(solver: Any) -> list:
    """Describe every non-convex, non-terrain geom of ``solver`` whose entity
    material carries ``COARSE_SDF_MATERIAL_KW``.

    Reads ``solver.geoms`` and, per geom, ``is_convex`` / ``type`` /
    ``entity.material`` (genesis-world 1.5.0). Returns
    ``"geom <idx> (entity <idx>, link <name>)"`` strings, empty if none."""
    import genesis as gs                             # noqa: PLC0415
    terrain = getattr(getattr(gs, "GEOM_TYPE", None), "TERRAIN", None)
    out = []
    for g in getattr(solver, "geoms", None) or []:
        if getattr(g, "is_convex", True):
            continue
        if terrain is not None and getattr(g, "type", None) == terrain:
            continue
        ent = getattr(g, "entity", None)
        if not _is_coarse_material(getattr(ent, "material", None)):
            continue
        link = getattr(g, "link", None)
        out.append(f"geom {getattr(g, 'idx', '?')} (entity {getattr(ent, 'idx', '?')}, "
                   f"link {getattr(link, 'name', '?')!s})")
    return out


def _nonconvex_narrowphase_compiled(solver: Any) -> bool:
    """``solver.collider.collider_config.has_nonconvex_nonterrain``
    (genesis-world 1.5.0 — True iff some valid collision pair holds a non-convex
    non-terrain geom, i.e. the SDF narrowphase is live). If the path is absent
    on this engine, answer True so the caller still reports (fail open)."""
    collider = getattr(solver, "collider", None)
    cfg = getattr(collider, "collider_config", None) if collider is not None else None
    flag = getattr(cfg, "has_nonconvex_nonterrain", None) if cfg is not None else None
    if flag is None:
        return True
    return bool(flag)


def warn_coarse_sdf_on_nonconvex(solver: Any, where: str) -> list:
    """Post-build check: log ONE warning listing every non-convex, non-terrain
    geom built with ``COARSE_SDF_MATERIAL_KW``, if the solver's non-convex
    narrowphase is compiled in (so that coarse SDF can actually be read in a
    contact). Returns the listed geoms (empty when nothing was logged).

    Called once per scene (main and, in dual_scene, raycast) from
    ``VehicleScene.build``, so every scene built through ``VehicleScene`` — the
    server's L2 and L3 scenes included — reports a coarse SDF that a contact can
    read. Scenes built without ``VehicleScene`` are not checked."""
    if solver is None or not _nonconvex_narrowphase_compiled(solver):
        return []
    bad = coarse_sdf_nonconvex_geoms(solver)
    if bad:
        _logger.warning(
            "[genesis_vehicle:coarse-sdf] %s: %d NON-convex geom(s) carry the SDK's "
            "coarse SDF material kwargs %s, and this solver's non-convex "
            "narrowphase reads their SDF in contacts — at 16 nodes/axis that "
            "contact may be inaccurate. Give these "
            "bodies convexify=True or a default-SDF material: %s",
            where, len(bad), COARSE_SDF_MATERIAL_KW, "; ".join(bad))
    return bad
