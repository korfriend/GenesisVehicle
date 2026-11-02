"""Thin compatibility shims over Genesis engine APIs that changed across releases.

The SDK supports the Genesis versions the deployed clones actually run, so a
renamed/reshaped engine call is branched **once, here**, rather than at every
call site. Each shim documents which Genesis versions take which branch.

Currently shimmed:

``apply_links_wrench``
    Genesis <= 1.3.3 exposed ``RigidSolver.apply_links_external_force`` and
    ``.apply_links_external_torque`` as two separate calls. Genesis 1.4.0
    removed both and replaced them with a single
    ``RigidSolver.apply_links_external_wrench(force=..., torque=...)``. The two
    are numerically identical for the SDK's usage (``ref="link_origin"``,
    ``local=False``): both accumulate into ``cfrc_applied_vel/ang``, and the
    wrench kernel's moment arm ``(link.pos - root_COM) x force`` is exactly what
    the old force kernel added. The wrench form is also one kernel launch
    instead of two.

``scene_sim_options``
    Genesis <= 1.3.3 assigned ``Scene.sim_options`` in ``Scene.__init__``.
    Genesis 1.4.0 dropped that attribute and nests the same object under
    ``Scene.options.sim`` (a ``SceneOptions`` field) — the constructor still
    ACCEPTS ``sim_options=``, so scene construction is unaffected and the
    breakage is read-side only.

``check_genesis_floor``
    Warn-only guard (once per process) when the installed genesis is older than
    the SDK's supported floor ``GENESIS_FLOOR`` (1.4.5). It never raises.

``COARSE_SDF_MATERIAL_KW`` / ``coarse_sdf_nonconvex_geoms``
    The ``gs.materials.Rigid`` kwargs the SDK puts on the collision bodies it
    owns, and the post-build check that names any NON-convex geom built with
    them — it runs only when the scene compiled the non-convex narrowphase
    (``has_nonconvex_nonterrain``), i.e. when some collision pair can read
    that SDF; otherwise it is silent. The grid formula they rely on is the same on 1.4.0 and 1.4.5; what
    changed in 1.4.5 is that every geom's SDF is computed eagerly at build.
"""

from __future__ import annotations

import logging
import re
from typing import Any, NamedTuple, Optional

import numpy as np

__all__ = ["apply_links_wrench", "has_wrench_api", "link_inertial",
           "LinkInertialProps", "scene_sim_options", "sensor_miss_value",
           "GENESIS_FLOOR", "check_genesis_floor", "COARSE_SDF_MATERIAL_KW",
           "coarse_sdf_nonconvex_geoms", "warn_coarse_sdf_on_nonconvex"]

_logger = logging.getLogger("genesis_vehicle")


def has_wrench_api(solver: Any) -> bool:
    """True on genesis >= 1.4.0 (single-call wrench API)."""
    return hasattr(solver, "apply_links_external_wrench")


def apply_links_wrench(solver: Any, force: Any, torque: Any,
                       links_idx: Any = None,
                       envs_idx: Optional[Any] = None) -> None:
    """Apply a world-frame force **and** torque to ``links_idx`` in one go.

    ``force`` / ``torque`` are ``(n_envs, n_links, 3)`` (or ``(n_links, 3)`` for
    an unbatched scene), matching what the underlying engine call expects.

    Resolved per call (not cached) so the server's monkey patches in
    ``server/vehicle_builder.apply_monkey_patches`` — which rebind these
    attributes on the solver instance — stay in effect.
    """
    fn = getattr(solver, "apply_links_external_wrench", None)
    if fn is not None:                                   # genesis >= 1.4.0
        fn(force=force, torque=torque, links_idx=links_idx, envs_idx=envs_idx)
        return
    solver.apply_links_external_force(force, links_idx, envs_idx)    # <= 1.3.3
    solver.apply_links_external_torque(torque, links_idx, envs_idx)


class LinkInertialProps(NamedTuple):
    """A link's authored inertial frame, engine-version independent.

    ``i`` is the 3x3 inertia tensor expressed in the inertial frame, which
    ``quat`` orients relative to the link frame — so the link-frame tensor is
    ``R(quat) @ i @ R(quat).T`` on every genesis version. (Genesis 1.4.0
    diagonalizes an authored tensor onto its principal axes and carries the
    rotation in ``quat``; 1.3.3 kept the authored tensor with an identity
    ``quat``. Both spell the same physical inertia.)"""
    mass: float
    pos: Any                                        # (3,) COM offset, link frame
    quat: Any                                       # (4,) wxyz, inertial frame
    i: Any                                          # (3, 3) inertia, inertial frame


def link_inertial(link: Any) -> Optional[LinkInertialProps]:
    """Read a link's inertial properties, or None if the link carries no mass.

    Genesis <= 1.3.3 exposed ``RigidLink.inertial_mass / .inertial_pos /
    .inertial_quat / .inertial_i``. Genesis 1.4.0 dropped all four and moved the
    authored values onto the link's resolved description
    (``link.desc.mass / .inertial_pos / .inertial_quat / .inertia``)."""
    mass = getattr(link, "inertial_mass", None)
    if mass is not None:                                        # genesis <= 1.3.3
        pos, quat = getattr(link, "inertial_pos", None), getattr(link, "inertial_quat", None)
        inertia = getattr(link, "inertial_i", None)
    else:                                                       # genesis >= 1.4.0
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


def scene_sim_options(scene: Any) -> Any:
    """The ``SimOptions`` object a ``gs.Scene`` was CONSTRUCTED with.

    Genesis <= 1.3.3 assigned it to ``Scene.sim_options``
    (``genesis/engine/scene.py`` ``self.sim_options = sim_options``). Genesis
    1.4.0 removed that attribute and nests the very same object under
    ``Scene.options.sim`` — measured on 1.4.0:
    ``gs.Scene(sim_options=so).options.sim is so`` → True, and
    ``hasattr(gs.Scene, "sim_options")`` → False. The constructor keyword
    ``sim_options=`` is unchanged on both, so only READS had to be branched.

    Probe order is 1.4.0-first (``options.sim``), then the <= 1.3.3 name. If
    neither exists this raises an ``AttributeError`` naming BOTH spellings —
    a silent ``None`` here would resurface as an unreadable
    ``'NoneType' has no attribute 'dt'`` far from the engine that changed.

    NOTE the object is the AUTHORED options: writing to it after
    ``scene.build()`` does NOT change the simulation on either version
    (``Simulator.__init__`` snapshots ``dt``/``substeps``). See
    ``docs/physics-contracts.md`` §7.14.
    """
    opts = getattr(scene, "options", None)               # genesis >= 1.4.0
    sim = getattr(opts, "sim", None) if opts is not None else None
    if sim is not None:
        return sim
    sim = getattr(scene, "sim_options", None)            # genesis <= 1.3.3
    if sim is not None:
        return sim
    raise AttributeError(
        f"{type(scene).__name__} exposes neither 'options.sim' (genesis >= "
        "1.4.0) nor 'sim_options' (genesis <= 1.3.3); this genesis version's "
        "scene options are not recognized by genesis_vehicle._gs_compat."
        "scene_sim_options")


def sensor_miss_value(sensor: Any) -> Optional[float]:
    """The distance ``sensor`` reports for a ray that hit NOTHING (the "miss
    sentinel"), or ``None`` if this engine's sensor does not expose one.

    Genesis (1.3.3 … 1.4.5, every version checked) resolves the value
    onto the sensor's own options object: ``Raycaster.no_hit_value``, which
    ``model_post_init`` fills with ``max_range`` when the user leaves it unset.
    That happens at sensor CONSTRUCTION, so this read is valid both before and
    after ``scene.build()`` — which matters because the SDK stamps it at
    ``add_sensor`` time. An engine shape that only carries ``max_range``
    (a hypothetical older/other raycaster) falls back to it; a stub sensor with
    neither returns ``None`` and callers keep the legacy threshold.

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
# Supported-floor guard
# ---------------------------------------------------------------------------

#: The oldest genesis-world release this SDK is validated against. The matching
#: dependency set is quadrants 1.3.3, trimesh >= 5.0 and zstandard (the 1.4.5
#: wheel's own Requires-Dist).
GENESIS_FLOOR = (1, 4, 5)

_floor_checked = False

_VERSION_RE = re.compile(
    r"^\s*v?(\d+)\.(\d+)(?:\.(\d+))?"          # release: major.minor[.patch]
    r"(?:[.\-_]?(a|b|rc|alpha|beta|c|pre|preview|dev)\.?\d*(?=$|[.+]))?",   # pre-release tag
    re.IGNORECASE)


def _parse_genesis_version(text: Any) -> Optional[tuple]:
    """``(major, minor, patch, is_prerelease)`` or ``None`` if unparseable.

    A missing patch reads as 0 (``"1.4"`` -> ``(1, 4, 0)``). A post-release
    (``1.4.5.post1``) or a local tag (``+g1234``) does not count as pre-release;
    ``a`` / ``b`` / ``rc`` / ``dev`` do — ``1.4.5rc1`` sorts BEFORE ``1.4.5``."""
    if not isinstance(text, str):
        return None
    m = _VERSION_RE.match(text)
    if m is None:
        return None
    major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    return major, minor, patch, m.group(4) is not None


def check_genesis_floor(gs_module: Any = None) -> None:
    """Warn ONCE per process if genesis-world is older than ``GENESIS_FLOOR``.

    Warn-only by design: it never raises, so a deployment whose venv lags the
    SDK (e.g. a clone that updates by ``git pull``) keeps running and says so.
    Logged at WARNING on the ``genesis_vehicle`` logger, independent of
    ``GENESIS_VEHICLE_QUIET`` (that variable only silences the version banner).

    Called from ``vehicle_scene._ensure_genesis`` (``VehicleScene.init_backend``
    and every ``VehicleScene`` built with the default ``init_genesis=True``,
    including the server) and from ``VehiclePhysics.__init__``, which the
    batched driver also reaches through its prototype (direct driver users such
    as the ``GeneVehicle_*`` demos). NOT covered: a ``VehicleScene(init_genesis
    =False)`` that never constructs a vehicle. Other entry points are not
    guarded.

    Branches: ``genesis.__version__`` parsed as ``major.minor[.patch][pre]``;
    ``< 1.4.5`` (or a 1.4.5 pre-release) warns; an unparseable or missing
    version warns that it could not be checked and continues."""
    global _floor_checked
    if _floor_checked:
        return
    _floor_checked = True
    if gs_module is None:
        import genesis as gs_module                 # noqa: PLC0415 - lazy on purpose
    raw = getattr(gs_module, "__version__", None)
    parsed = _parse_genesis_version(raw)
    floor = ".".join(str(x) for x in GENESIS_FLOOR)
    if parsed is None:
        _logger.warning(
            "[genesis_vehicle:genesis-floor] could not parse genesis-world version "
            "%r; the supported floor (%s) was not checked. Continuing.", raw, floor)
        return
    below = parsed[:3] < GENESIS_FLOOR or (parsed[:3] == GENESIS_FLOOR and parsed[3])
    if not below:
        return
    _logger.warning(
        "[genesis_vehicle:genesis-floor] genesis-world %s is older than this SDK's "
        "supported floor %s (with quadrants 1.3.3, trimesh>=5, zstandard). Not "
        "enforced — continuing. Known differences from the floor are listed in "
        "CHANGELOG 1.6.9 (SDF pre-processing at build, rigid-body hibernation); "
        "genesis-world releases before 1.4.0 are no longer supported.", raw, floor)


# ---------------------------------------------------------------------------
# Coarse SDF material kwargs for the collision bodies the SDK owns
# ---------------------------------------------------------------------------

#: ``gs.materials.Rigid(**COARSE_SDF_MATERIAL_KW)`` — the minimum SDF grid the
#: engine's validators accept, for collision bodies whose SDF the SDK does not
#: expect to be read.
#:
#: What it does NOT do: it does not switch the SDF off. genesis sizes each geom's
#: grid per axis as ``cell = clip(target, grid/(sdf_max_res-1),
#: grid/max(sdf_min_res-1, 2))`` (1.4.5 ``rigid_entity/description.py``
#: ``SignedDistanceFieldDescription.from_geometry``; same formula in 1.4.0
#: ``rigid_geom.py``), so for a convex geom the huge ``sdf_cell_size`` is clipped
#: to the UPPER bound and the grid lands on ``sdf_min_res`` = 16 nodes per axis
#: (the validator minimum, ``ge=16``). The previous ``sdf_cell_size=10000.0``
#: alone landed on the default ``sdf_min_res`` = 32.
#:
#: Effect by geom kind:
#:
#: * convex geom (primitive, ``convexify=True`` mesh, URDF collision geom under
#:   the default ``convexify``) — 16 nodes per axis. genesis resets a Primitive
#:   morph's ``sdf_max_res`` to 32 on the material object, which does not move
#:   the result: the upper clip still wins.
#: * NON-convex, watertight geom — genesis first shrinks the target to half the
#:   wall thickness. Where that is at or below ``grid/31`` these kwargs change
#:   nothing (thin walls still refine, up to ``sdf_max_res``); where it is above
#:   ``grid/31`` the upper clip moves from ``grid/31`` to ``grid/15`` and the
#:   grid DOES get coarser.
#: * NON-convex, non-watertight geom — no wall refinement; the grid goes from 32
#:   to 16 nodes per axis, i.e. it DOES get coarser.
#:
#: Where the SDF is read: in a rigid-only scene, only by the non-convex vs
#: non-terrain narrowphase (1.4.5 ``collider/narrowphase.py``, gated on
#: ``collider_config.has_nonconvex_nonterrain``); terrain uses its heightfield
#: and the SDK's ray-wheel contact never reads an SDF. So the kwargs are only
#: appropriate where every geom is convex — and ``warn_coarse_sdf_on_nonconvex``
#: (called from ``VehicleScene.build`` for both scenes) names any non-convex geom
#: that carries them in a solver whose non-convex narrowphase is compiled in.
#:
#: Why it matters on 1.4.5: the collider computes EVERY geom's SDF at build
#: (1.4.0 only sized the buffer and computed lazily), so the per-geom node count
#: is paid in build time and memory whether or not the SDF is ever read.
COARSE_SDF_MATERIAL_KW = {"sdf_cell_size": 1.0e4, "sdf_min_res": 16}


def _is_coarse_material(material: Any) -> bool:
    if material is None:
        return False
    return all(getattr(material, k, None) == v for k, v in COARSE_SDF_MATERIAL_KW.items())


def coarse_sdf_nonconvex_geoms(solver: Any) -> list:
    """Describe every non-convex, non-terrain geom of ``solver`` whose entity
    material carries ``COARSE_SDF_MATERIAL_KW``.

    Reads ``solver.geoms`` and, per geom, ``is_convex`` / ``type`` /
    ``entity.material`` — present on genesis 1.4.0 and 1.4.5. Returns
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
    """``solver.collider.collider_config.has_nonconvex_nonterrain`` (genesis
    1.4.0 and 1.4.5 — True iff some valid collision pair holds a non-convex
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
            "narrowphase reads their SDF in contacts — at 16 nodes/axis (or its "
            "wall-thickness refinement) that contact may be inaccurate. Give these "
            "bodies convexify=True or a default-SDF material: %s",
            where, len(bad), COARSE_SDF_MATERIAL_KW, "; ".join(bad))
    return bad
