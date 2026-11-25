"""Wheel visuals for the Genesis renderer (2.0.6: engine-node override).

``VehicleScene`` draws the wheels of a rendered scene (a camera or a viewer) by
overriding the poses of the ENGINE's own wheel render nodes with the closed-form
wheel transform (``wheel_visual_transforms``: steer + suspension + spin),
composed with each geom's local offset, on every node sync of the main scene's
rasterizer context. It writes no solver state: no joint is written, no FK runs,
no PD acts; headless physics and rendered physics are the same computation.

Since 2.0.6 the URDF wheel visual geoms are NOT hidden, so every wheel pixel
carries the segmentation id Genesis gives that wheel's URDF link part, at every
``n_envs`` and segmentation level. A wheel the engine draws no node for (e.g. an
entity added with ``vis_mode="collision"``, whose wheel links have no colliders
after ``urdf_prep``) gets an SDK FALLBACK node: the harvested wheel mesh,
registered for segmentation under the link's first visual geom (at geom level a
fallback wheel with several visual geoms is one merged mesh under visual geom
0's ``GeomKey``).

What is NOT covered (the hook sits on ``scene.visualizer.context`` only, see
``_gs_compat.on_render_sync``): camera SENSORS (``scene.add_sensor(
gs.sensors.{Rasterizer,BatchRenderer,Raytracer}CameraOptions)``; the rasterizer
one renders from its own context when there is no viewer), visual raycasting /
picking, the two syncs ``Visualizer.build`` makes and the viewer frames drawn
during ``build()`` show the URDF wheel at the SOLVER's pose (the solver's own
wheel joint state, which no SDK code writes and which does not follow the
commanded steer, suspension or spin; measured values in
``VehiclePhysics.link_transforms``). The native viewer thread was not run when
this was written (no display); what it draws is inferred from the source.

Removed in 2.0.6: ``WheelJointInternalSync`` and ``KindVisualBatch`` (the
joint-writing path, which moved the chassis trajectory: the reference car ended
+0.142 m ahead over 10.7 m, and ΣN lost the wheel-side link weight; the
coupling was not established). Importing either raises ``ImportError``. NB
because this module's ``__getattr__`` raises ``ImportError`` for them,
``hasattr(genesis_vehicle.visual, "WheelJointInternalSync")`` raises instead of
returning False. External renderers (UE / Unity) read
``VehiclePhysics.wheel_visual_transforms`` / ``visual_parts_transforms``.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Sequence

import numpy as np
import genesis as gs

from ._gs_compat import (WHEEL_RENDER_TAG, add_render_node,
                         check_render_fallback_api, check_render_seg_api,
                         engine_part_nodes, force_render_sync, on_render_sync,
                         part_is_posed, pose_max_abs, queue_part_poses, queued_model,
                         register_fallback_seg, rendered_envs, require_genesis)

require_genesis(gs)

_logger = logging.getLogger("genesis_vehicle")

_REMOVED = {
    "WheelJointInternalSync": (
        "WheelJointInternalSync was removed in 2.0.6: it wrote the URDF wheel "
        "joints into the rigid solver, which moved the chassis trajectory. Use "
        "VehicleScene (it draws the wheels from the closed form at every n_envs) "
        "or VehiclePhysics.wheel_visual_transforms() / visual_parts_transforms(); "
        "the spin angle it accumulated is VehiclePhysics.wheel_spin_angle."),
    "KindVisualBatch": (
        "KindVisualBatch was removed in 2.0.6: it wrote the URDF wheel joints of "
        "a vehicle kind into the rigid solver, which moved the chassis "
        "trajectory. Use VehicleScene (it draws the wheels from the closed form "
        "at every n_envs) or MultiVehiclePhysics.wheel_visual_transforms()."),
}


def __getattr__(name: str) -> Any:
    """``from genesis_vehicle.visual import WheelJointInternalSync`` and
    ``genesis_vehicle.visual.WheelJointInternalSync`` both raise ``ImportError``
    with the replacement named (2.0.6)."""
    if name in _REMOVED:
        raise ImportError(_REMOVED[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


#: One WARNING per process for a wheel the renderer can only draw as a
#: synthesized cylinder with no segmentation id.
_HARVEST_WARNED = False


def _pose_mats(pos: np.ndarray, quat_wxyz: np.ndarray, dtype: Any = np.float32) -> np.ndarray:
    """(N, 3) positions + (N, 4) wxyz quats -> (N, 4, 4) transforms."""
    w = quat_wxyz[:, 0]; x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]; z = quat_wxyz[:, 3]
    T = np.zeros((pos.shape[0], 4, 4), dtype=dtype)
    T[:, 0, 0] = 1 - 2 * (y * y + z * z)
    T[:, 0, 1] = 2 * (x * y - w * z)
    T[:, 0, 2] = 2 * (x * z + w * y)
    T[:, 1, 0] = 2 * (x * y + w * z)
    T[:, 1, 1] = 1 - 2 * (x * x + z * z)
    T[:, 1, 2] = 2 * (y * z - w * x)
    T[:, 2, 0] = 2 * (x * z - w * y)
    T[:, 2, 1] = 2 * (y * z + w * x)
    T[:, 2, 2] = 1 - 2 * (x * x + y * y)
    T[:, :3, 3] = pos
    T[:, 3, 3] = 1.0
    return T


def _local_T(geom: Any) -> np.ndarray:
    """A rigid geom's pose in its link frame (``init_pos`` / ``init_quat``), float64 4x4."""
    lp = np.asarray(geom.init_pos, dtype=np.float64).reshape(1, 3)
    lq = np.asarray(geom.init_quat, dtype=np.float64).reshape(1, 4)
    return _pose_mats(lp, lq, np.float64)[0]


# numpy floating-point warnings silenced while the closed-form poses are
# assembled: an inf in one env's pose (inf * 0 in the matmul, seen in the unit
# test; a float32 cast overflow) would otherwise print a RuntimeWarning for a
# row that ``_keep_invalid_rows`` then drops. A NaN pose printed none in the
# unit test (run with -W error::RuntimeWarning).
def _nonfinite_ok() -> Any:
    return np.errstate(invalid="ignore", over="ignore")


def _keep_invalid_rows(T: np.ndarray, prev: np.ndarray, max_abs: float) -> np.ndarray:
    """The engine's invalid-pose rule applied to a pose array about to be
    written: every instance row of ``T`` (shape ``(R, 4, 4)``) with an entry
    that is not ``abs(x) < max_abs`` (NaN included) is replaced by the same row
    of ``prev``. Returns ``T`` itself when every row is valid, else a copy."""
    bad = ~(np.abs(T) < max_abs).all(axis=(-2, -1))
    if not bad.any():
        return T
    out = T.copy()
    out[bad] = np.asarray(prev)[bad]
    return out


def _cylinder_mesh(radius: Optional[float]) -> Any:
    import trimesh as _tm
    r = float(radius or 0.35)
    width = max(0.08, 0.4 * r)
    tm = _tm.creation.cylinder(radius=r, height=width, sections=24)
    # cylinder axis is +z; a wheel spins about the chassis +y axis
    tm.apply_transform(_tm.transformations.rotation_matrix(math.pi / 2.0, (1, 0, 0)))
    return tm


class InstancedWheelRenderer:
    """VehicleScene's wheel renderer (2.0.6: engine-node override, "ENO").

    One unit per vehicle kind (batched solver) or per vehicle (per_vehicle
    solver), holding the K entities, the wheel list and the closed-form pose
    provider. On every node sync of the main scene's rasterizer context
    (``_gs_compat.on_render_sync``), :meth:`update` writes, into every engine
    render node of each wheel part, ``T_closed(env, k, i) @ T_local(part)`` for
    the rendered envs, and the closed form itself into fallback nodes.

    Which parts are drawn is decided from the ENGINE MODEL, never from an empty
    node lookup: entity ``surface.vis_mode`` None -> the wheel link's visual
    geoms; ``"collision"`` -> its collision geoms; a link with no such geoms, or
    a wheel name that does not resolve -> one SDK fallback node. A part that
    must be drawn but has no engine node raises ``RuntimeError``; a visual geom
    with ``DynamicVertices`` raises ``ValueError`` (defensive: not reachable
    through ``VehicleScene.add_vehicle``). A non-None ``vis_mode`` other than
    ``"collision"`` on a wheel link that carries collision geoms raises
    ``ValueError`` at ``build()`` of a rendered scene: on 1.5.0 only ``"sdf"``
    reaches here (``Scene.add_entity`` clears every other non-debug value to
    None), and its draw layout was never run. It needs an UNPREPARED URDF
    (``add_vehicle(morph=...)``; ``prepare_vehicle_urdf`` strips wheel
    colliders). With a prepared URDF the wheel link has no colliders and an
    ``"sdf"`` entity's wheels are drawn by fallback nodes -- untested.

    Ordering guard (per sync): the engine writes its nodes' poses inside
    ``_sync_shapes``, before the hook. At the end of every :meth:`update` that
    completes, the renderer records a private COPY of every pose array it
    wrote. At the START of every hooked sync it compares each recorded
    primitive's ``poses`` with that copy by value (``np.array_equal(...,
    equal_nan=True)``) and then discards the record, also when the comparison
    raised. A raise (``RuntimeError``) means: the wheel poses the SDK wrote on
    the previous completed hooked sync were changed before this one -- a
    re-assignment or an in-place write by anything after the hook, e.g. an
    engine whose pose write moved later in ``update()``. Limits:
    (1) ONE SYNC LATE: the first affected frame has already been drawn at the
    solver's pose, and the raise comes at the next hooked sync; a frame with no
    later hooked sync (the last render of a process, or a single render after a
    step) is never checked. (2) It does NOT detect an engine ``"model"`` buffer
    queued after the hook without touching ``primitive.poses`` (the queue is
    last-write-wins); that case is guarded only by the flush test in
    ``tests/test_render_sync_pose.py``, to be run on every engine bump.
    (3) After a hooked sync that did not complete -- the guard raised, the
    engine's ``_sync_shapes`` raised, the pose provider (or anything else in
    :meth:`update`) raised, or a ``BaseException`` such as
    ``KeyboardInterrupt`` -- nothing was recorded, so the next hooked sync is
    not checked; checking resumes on the one after. A persistent late write
    therefore raises on at least every second hooked sync; while the provider
    raises on every sync, its own exception surfaces on each.
    (4) A non-finite pose is not an error (next paragraph).

    Non-finite poses: before writing, every instance row (one rendered env of
    one wheel node) with an entry that is not ``abs(x) < POSE_MAX_ABS`` (NaN or
    inf; the engine's own invalid-pose rule, read through
    ``_gs_compat.pose_max_abs``) is replaced by the row the SDK last wrote to
    that node (the record of the last completed :meth:`update`, which survives
    the guard's discard), or by the node's current row when the SDK has written
    none. So a diverged env's wheel instance keeps its last drawn pose and no
    NaN reaches the renderer's bounds; no warning is issued (2.0.6). The record
    and the guard copy hold the array actually written. The SDK writes one
    array to every primitive of a node; the first primitive's row is the one
    kept (as in the engine). The wiring self-test (:meth:`wire`) runs one forced
    sync and checks the hook ran once and the poses / queued buffers equal the
    override -- a wiring / structure check: that sync cannot test the ordering,
    because on 1.5.0 the engine writes node poses only on a sync after a state
    change (measured)."""

    def __init__(self, gs_scene: Any):
        self._gs_scene = gs_scene
        self._units: list[dict] = []
        self._ctx = None
        self._hook = None
        self._resolved = False
        self._pose_max_abs: Optional[float] = None
        # primitive -> a private copy of the array the last COMPLETED update()
        # assigned to primitive.poses. ``_last`` is the guard's copy (compared by
        # value, then discarded, at the start of every hooked sync); ``_drawn``
        # is the same record kept for the non-finite-row rule (never discarded).
        self._last: dict = {}
        self._drawn: dict = {}

    # -- units ---------------------------------------------------------------

    def add_unit(self, entities: Sequence, wheels: Sequence, provider: Any, K: int) -> None:
        """Register one vehicle kind: ``entities`` (K of them, slot order),
        ``wheels`` (the kind's ``cfg.wheels``), ``provider(frame)`` returning the
        closed-form ``(pos, quat)`` shaped ``(n_envs, K, n, 3/4)`` (kind
        physics) or ``(n_envs, n, 3/4)`` (single VehiclePhysics, K == 1)."""
        entities = tuple(entities)
        if len(entities) != int(K):
            raise ValueError(f"add_unit: {len(entities)} entities for K={K}")
        self._units.append(dict(entities=entities, wheels=tuple(wheels), provider=provider,
                                K=int(K), n=len(wheels), parts=None, fallback=None))

    def rebind(self, providers: Sequence) -> None:
        """Re-point the units' pose providers, in unit order.

        ``providers`` is one ``(provider, K, n, entities)`` per unit, matched
        POSITIONALLY (``VehicleScene._rebind_wheel_renderer`` owns that
        ordering). K, n AND the identity of every entity (in slot order) must
        still match: the wheel parts are tied to entities, and two same-shaped
        kinds swapping places would otherwise pose one vehicle's wheels on the
        other. Raises ``ValueError`` on any mismatch and leaves EVERY unit
        untouched (validated in full before the first assignment). Used when
        ``VehicleScene`` rebuilds its batched driver: the old providers are
        bound methods of the DISCARDED kind, which is never stepped again."""
        if len(providers) != len(self._units):
            raise ValueError(
                f"rebind got {len(providers)} providers for "
                f"{len(self._units)} render units (the vehicle kinds changed)")
        for u, (_, K, n, ents) in zip(self._units, providers):
            if int(K) != int(u["K"]) or int(n) != int(u["n"]):
                raise ValueError(
                    f"rebind shape mismatch: unit has K={u['K']}, n={u['n']}; "
                    f"new provider has K={int(K)}, n={int(n)}")
            ents = tuple(ents)
            if len(ents) != len(u["entities"]) or any(a is not b for a, b in zip(ents, u["entities"])):
                raise ValueError(
                    "rebind entity mismatch: the new provider's kind holds different "
                    "entities (or a different slot order) than the render unit")
        for u, (provider, _, _, _) in zip(self._units, providers):
            u["provider"] = provider

    # -- wiring ----------------------------------------------------------------

    def wire(self) -> None:
        """Install on the main scene's rasterizer context (after ``build()``):
        structure check, hook, one build-time :meth:`update` under the viewer
        lock (decides the parts and the fallbacks and poses every node before
        any draw), then the behavioural self-test: one forced sync must run the
        hook exactly once and leave every posed primitive at the override
        (poses and, when queued, the engine's "model" buffer)."""
        vis = self._gs_scene.visualizer
        ctx = vis.context
        check_render_seg_api(ctx)
        self._ctx = ctx
        self._hook = on_render_sync(ctx, self.update, before=self._check_last_writes)
        with vis.viewer_lock:
            self.update()
        c0 = self._hook.calls
        force_render_sync(ctx, vis.viewer_lock)
        if self._hook.calls != c0 + 1:
            raise RuntimeError(
                f"{WHEEL_RENDER_TAG} wiring self-test: one forced sync ran the wheel hook "
                f"{self._hook.calls - c0} times (expected 1)")
        for prim, arr in self._last.items():
            if np.abs(np.asarray(prim.poses, dtype=np.float64) - arr).max() > 1e-6:
                raise RuntimeError(
                    f"{WHEEL_RENDER_TAG} wiring self-test: a wheel node's poses differ "
                    f"from the override after a forced sync")
            q = queued_model(ctx, prim)
            if q is not None and np.abs(np.asarray(q).transpose(0, 2, 1) - arr).max() > 1e-6:
                raise RuntimeError(
                    f"{WHEEL_RENDER_TAG} wiring self-test: the queued 'model' buffer of a "
                    f"wheel node is not the transposed override (engine buffer layout changed?)")

    def _check_last_writes(self) -> None:
        """The per-sync ordering guard (runs at the START of every hooked
        sync): every primitive posed by the last completed :meth:`update` must
        still hold, by value (``equal_nan=True``), the poses written there.
        The record is discarded afterwards, also when this raises, so only
        :meth:`update` re-creates it. See the class docstring for what it does
        and does not detect (one sync late; not a queued buffer; no check on
        the sync after an incomplete one)."""
        try:
            for prim, saved in self._last.items():
                if not np.array_equal(prim.poses, saved, equal_nan=True):
                    raise RuntimeError(
                        f"{WHEEL_RENDER_TAG} the wheel poses the SDK wrote on the previous "
                        f"completed hooked sync were changed before this one (a write after "
                        f"the SDK's sync hook; the engine's pose write is no longer inside "
                        f"_sync_shapes?); detected one sync late: the previous frame may "
                        f"have drawn that wheel at the solver's pose")
        finally:
            self._last = {}

    # -- parts -------------------------------------------------------------------

    def _resolve(self, ctx: Any) -> None:
        """Decide, from the engine model, each (unit, k, i)'s drawn parts and
        whether it needs a fallback node; create the fallback nodes (in unit, k,
        i order, so their segmentation ids follow the engine's in that order)."""
        R = len(rendered_envs(ctx))
        self._pose_max_abs = pose_max_abs()
        for u in self._units:
            parts, fallback = [], {}
            for k, ent in enumerate(u["entities"]):
                vis_mode = ent.surface.vis_mode
                for i, w in enumerate(u["wheels"]):
                    name = w.name
                    link = None
                    if name is not None:
                        try:
                            link = ent.get_link(name)
                        except gs.GenesisException:
                            _logger.debug("wheel link %r not found", name, exc_info=True)
                            link = None
                    if link is None:
                        fallback[(k, i)] = (None, None, w)
                        continue
                    vgeoms, cgeoms = list(link.vgeoms), list(link.geoms)
                    if vis_mode is None:
                        if vgeoms:
                            for vg in vgeoms:
                                if not part_is_posed(vg):
                                    raise ValueError(
                                        f"{WHEEL_RENDER_TAG} wheel {name!r}: its visual geom draws "
                                        f"DynamicVertices (world-space vertices), which a pose "
                                        f"override cannot place")
                                parts.append((k, i, vg, _local_T(vg)))
                        else:
                            fallback[(k, i)] = (link, cgeoms[0].seg_keys if cgeoms else None, w)
                    elif not cgeoms:
                        fallback[(k, i)] = (link, vgeoms[0].seg_keys if vgeoms else None, w)
                    elif vis_mode == "collision":
                        for g in cgeoms:
                            parts.append((k, i, g, _local_T(g)))
                    else:
                        raise ValueError(
                            f"{WHEEL_RENDER_TAG} wheel {name!r}: entity vis_mode={vis_mode!r} with "
                            f"collision geoms on the wheel link is not supported by the wheel "
                            f"renderer (with wheel colliders only vis_mode None and 'collision' "
                            f"are supported; the draw layout "
                            f"of {vis_mode!r} with wheel colliders was never run). Prepare the "
                            f"URDF with genesis_vehicle.urdf_prep.prepare_vehicle_urdf (it strips "
                            f"wheel colliders), or use vis_mode=None / 'collision'. Raised from "
                            f"build() of a rendered scene (a camera or a viewer); headless scenes "
                            f"are unaffected.")
            # C25: one node reached from two parts with different T_local
            seen: dict = {}
            for j, (k, i, part, Tl) in enumerate(parts):
                nodes = engine_part_nodes(ctx, part)
                if not nodes:
                    raise RuntimeError(
                        f"{WHEEL_RENDER_TAG} wheel {u['wheels'][i].name!r} (vehicle slot {k}): "
                        f"the engine draws this wheel part but no render node was found under its "
                        f"segmentation keys (the engine's node lookup changed?)")
                for node in nodes:
                    prev = seen.get(id(node))
                    if prev is not None and not np.allclose(prev, Tl, atol=1e-9):
                        raise RuntimeError(
                            f"{WHEEL_RENDER_TAG} one engine render node is reached from two wheel "
                            f"parts with different local offsets (a collider and a visual geom "
                            f"sharing a GeomKey?)")
                    seen[id(node)] = Tl
            u["parts"] = parts
            u["part_k"] = np.array([p[0] for p in parts], dtype=np.int64)
            u["part_i"] = np.array([p[1] for p in parts], dtype=np.int64)
            u["part_T"] = (np.stack([p[3] for p in parts]) if parts
                           else np.zeros((0, 4, 4)))
            u["fallback"] = {}
            if fallback:
                check_render_fallback_api(ctx)
            for (k, i), (link, seg_keys, w) in sorted(fallback.items()):
                u["fallback"][(k, i)] = self._make_fallback(ctx, u, k, i, link, seg_keys, w, R)
        self._resolved = True

    def _make_fallback(self, ctx: Any, u: dict, k: int, i: int, link: Any,
                       seg_keys: Any, w: Any, R: int) -> Any:
        """One SDK fallback node for wheel (k, i) of unit ``u``: the harvested
        mesh, R instances, registered for segmentation under ``seg_keys`` (or
        unregistered, with the one-per-process harvest WARNING)."""
        global _HARVEST_WARNED
        mesh = self._harvest(link, w)
        if seg_keys is None and not _HARVEST_WARNED:
            _HARVEST_WARNED = True
            _logger.warning(
                "[genesis_vehicle:wheel-harvest] wheel %r: no wheel link geometry to "
                "draw or to take a segmentation id from; drawn as a synthesized "
                "cylinder of its radius with no segmentation id. Its URDF wheel "
                "visual, if any, is drawn as well (twice). (one warning per process)",
                w.name)
        from genesis.ext import pyrender
        obj = pyrender.Mesh.from_trimesh(
            mesh, name=f"gv_wheel_fb_{id(u)}_{k}_{i}",
            poses=np.tile(np.eye(4, dtype=np.float32), (R, 1, 1)),
            smooth=False, is_marker=False,
            envs=(np.ones(R, dtype=bool) if self._gs_scene.n_envs > 1 else None))
        node = add_render_node(ctx, obj)
        if seg_keys is not None:
            register_fallback_seg(ctx, node, seg_keys)
        return node

    @staticmethod
    def _harvest(link: Any, wheel: Any) -> Any:
        """The fallback mesh of one wheel, in the wheel-link frame: its visual
        geoms, else its collision geoms, else a cylinder of the wheel radius."""
        import trimesh as _tm
        geoms = []
        if link is not None:
            geoms = list(link.vgeoms) or list(link.geoms)
        parts = []
        for g in geoms:
            tm = g.get_trimesh().copy()
            tm.apply_transform(_local_T(g))
            parts.append(tm)
        if not parts:
            parts.append(_cylinder_mesh(getattr(wheel, "radius", None)))
        return _tm.util.concatenate(parts) if len(parts) > 1 else parts[0]

    # -- per-sync update ---------------------------------------------------------

    def drawn_nodes(self) -> list:
        """``[(unit_idx, k, i, node, T_local or None)]`` for every node the
        renderer poses (engine nodes with the part's local offset, fallback
        nodes with None). For tests and harnesses."""
        out = []
        ctx = self._ctx
        for ui, u in enumerate(self._units):
            for (k, i, part, Tl) in (u["parts"] or []):
                for node in engine_part_nodes(ctx, part):
                    out.append((ui, k, i, node, Tl))
            for (k, i), node in (u["fallback"] or {}).items():
                out.append((ui, k, i, node, None))
        return out

    def update(self) -> None:
        """Write the current closed-form wheel poses into the wheel nodes
        (engine nodes and fallbacks) for the rendered envs. Called from the
        sync hook (and once at wiring); holds the viewer lock (reentrant)."""
        ctx = self._ctx
        if ctx is None or not self._units:
            return
        with self._gs_scene.visualizer.viewer_lock:
            if not self._resolved:
                self._resolve(ctx)
            ridx = rendered_envs(ctx)
            R = len(ridx)
            items = []
            for u in self._units:
                K, n = u["K"], u["n"]
                p, q = u["provider"]("world")
                p = p.detach().cpu().numpy().astype(np.float64).reshape(-1, K, n, 3)[ridx]
                q = q.detach().cpu().numpy().astype(np.float64).reshape(-1, K, n, 4)[ridx]
                with _nonfinite_ok():                  # a non-finite row is dropped below
                    T = _pose_mats(p.reshape(-1, 3), q.reshape(-1, 4), np.float64).reshape(R, K, n, 4, 4)
                parts = u["parts"]
                if parts:
                    with _nonfinite_ok():
                        Tw = T[:, u["part_k"], u["part_i"]] @ u["part_T"][None]      # (R, P, 4, 4)
                        Tw = np.ascontiguousarray(Tw.transpose(1, 0, 2, 3), dtype=np.float32)
                    done = set()
                    for j, (k, i, part, _) in enumerate(parts):
                        nodes = engine_part_nodes(ctx, part)
                        if not nodes:
                            raise RuntimeError(
                                f"{WHEEL_RENDER_TAG} wheel {u['wheels'][i].name!r} (vehicle slot "
                                f"{k}): its engine render node disappeared")
                        arr = Tw[j]
                        for node in nodes:
                            if id(node) in done:
                                continue
                            done.add(id(node))
                            for prim in node.mesh.primitives:
                                if len(prim.poses) != R:
                                    raise RuntimeError(
                                        f"{WHEEL_RENDER_TAG} an engine wheel node holds "
                                        f"{len(prim.poses)} instances, expected one per rendered "
                                        f"env ({R})")
                            items.append((node, arr))
                if u["fallback"]:
                    with _nonfinite_ok():
                        Tf = np.ascontiguousarray(T.transpose(1, 2, 0, 3, 4), dtype=np.float32)  # (K, n, R, 4, 4)
                    for (k, i), node in u["fallback"].items():
                        items.append((node, Tf[k, i]))
            items = [(node, self._valid_rows(node, arr)) for node, arr in items]
            queue_part_poses(ctx, items)
            self._drawn = self._last = {prim: arr.copy() for node, arr in items
                                        for prim in node.mesh.primitives}

    def _valid_rows(self, node: Any, arr: np.ndarray) -> np.ndarray:
        """``arr`` with every instance row that fails the engine's invalid-pose
        rule (``_gs_compat.pose_max_abs``) replaced by the row the SDK last
        wrote to ``node`` (its first primitive's record), else by that
        primitive's current row. See the class docstring."""
        prims = node.mesh.primitives
        if not prims:
            return arr
        prev = self._drawn.get(prims[0])
        if prev is None or prev.shape != arr.shape:
            prev = prims[0].poses
        return _keep_invalid_rows(arr, prev, self._pose_max_abs)
