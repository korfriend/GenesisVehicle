"""Fake-object tests of the 2.0.6 wheel renderer (``visual.InstancedWheelRenderer``,
engine-node override), no renderer and no Genesis scene.

| abbr | meaning |
|---|---|
| ENO | engine-node override: the engine's own wheel nodes get closed-form poses after every sync |
| T_local | a geom's pose in its link frame (``init_pos`` / ``init_quat``) |
| R | number of rendered envs (instances per engine node) |

What each test pins (plan / review condition in brackets):
- the fallback is decided from the engine MODEL, never from an empty node lookup;
  a drawn part with no node raises (api_review A3);
- a visual geom with DynamicVertices raises ValueError (C27, defensive);
- two parts reaching one node with different T_local raise (C25 / A6); in visual
  mode a collider sharing a visual geom's GeomKey is not a part (A6);
- ``rebind`` checks entity identity (A7);
- the per-sync ordering guard raises when a posed primitive was re-posed after the
  hook — the "hook before the engine's write" mutant (A2) — ONE SYNC LATE, compares
  by value (an in-place write is caught; a copying ``poses`` setter is not a false
  raise) (review_r206_6 R2a);
- the guard's record is discarded at every hooked sync, also when it raises, and
  re-created only by a completed ``update()``: a hooked sync that did not
  complete (engine ``_sync_shapes`` raised, the provider raised, the guard
  raised) leaves the next sync unchecked and the one after checked again; a
  finite -> NaN in-place write is still caught (fix round 1, F-C2 / F-C3);
- a non-finite closed-form row is never written: that instance keeps the row the
  SDK last wrote (the engine's ``POSE_MAX_ABS`` rule, read through
  ``_gs_compat.pose_max_abs``), and nothing raises (F-C1 / F-C3);
- ``vis_mode="sdf"`` on a wheel link with colliders raises ValueError naming the
  workaround; without colliders it takes the fallback (R3);
- every node gets ``T_closed @ T_local``; fallbacks are made in (unit, k, i) order
  and registered under the link's first visual geom.
"""

import contextlib

import numpy as np
import pytest

pytest.importorskip("genesis")

import torch

from genesis.engine.render_context import GeomKey, LinkKey, SegmentationKeys
from genesis_vehicle import _gs_compat as C
from genesis_vehicle import visual as V


# --- fakes ------------------------------------------------------------------

class _Prim:
    def __init__(self, R):
        self.poses = np.tile(np.eye(4, dtype=np.float32), (R, 1, 1))


class _Mesh:
    def __init__(self, R):
        self.primitives = [_Prim(R)]
        self._bounds = None


class _Node:
    def __init__(self, R):
        self.mesh = _Mesh(R)


class _Jit:
    def __init__(self):
        self._buffer = {}

    def update_buffer(self, primitives, buffer_name, data):
        for p, d in zip(primitives, data):
            self._buffer[(p, buffer_name)] = d


class _Ctx:
    def __init__(self, R=2, level="link"):
        self.nodes_by_seg_keys = {}
        self.rendered_envs_idx = list(range(R))
        self.jit = _Jit()
        self.segmentation_level = level
        self.seg = []
        self.added = []

    def _sync_shapes(self):
        pass

    def create_node_seg(self, seg_key, seg_node):
        self.seg.append((seg_key, seg_node))

    def add_node(self, obj):
        node = _Node(len(self.rendered_envs_idx))
        node.obj = obj
        self.added.append(node)
        return node


class _Desc:
    dynamic_vertices = None


class _Geom:
    def __init__(self, e, l, g, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
        self.seg_keys = SegmentationKeys(e, LinkKey(e, l), GeomKey(e, l, g))
        self.init_pos = np.array(pos)
        self.init_quat = np.array(quat)
        self.desc = _Desc()

    def get_trimesh(self):
        import trimesh
        return trimesh.creation.box((0.1, 0.1, 0.1))


class _Link:
    def __init__(self, vgeoms=(), geoms=()):
        self.vgeoms = list(vgeoms)
        self.geoms = list(geoms)


class _Surface:
    def __init__(self, vis_mode=None):
        self.vis_mode = vis_mode


class _Entity:
    def __init__(self, links, vis_mode=None):
        self._links = links
        self.surface = _Surface(vis_mode)

    def get_link(self, name):
        import genesis as gs
        if name not in self._links:
            gs.raise_exception(f"Link not found for name: {name}.")
        return self._links[name]


class _Wheel:
    def __init__(self, name, radius=0.3):
        self.name = name
        self.radius = radius


class _Vis:
    def __init__(self, ctx):
        self.context = ctx
        self.viewer_lock = contextlib.nullcontext()


class _Scene:
    def __init__(self, ctx, n_envs=2):
        self.visualizer = _Vis(ctx)
        self.n_envs = n_envs


def _provider(K, n, R):
    """Distinct poses per (env, k, i): translation (10 e + k, i, 0.3), yaw 0.1 i."""
    def prov(frame="world"):
        p = torch.zeros(R, K, n, 3)
        q = torch.zeros(R, K, n, 4)
        for e in range(R):
            for k in range(K):
                for i in range(n):
                    p[e, k, i] = torch.tensor([10.0 * e + k, float(i), 0.3])
                    a = 0.05 * i
                    q[e, k, i] = torch.tensor([np.cos(a), 0.0, 0.0, np.sin(a)])
        return p, q
    return prov


def _car(e, n=2, vis_mode=None, with_collider_key_clash=False, local=(0.0, 0.1, 0.0)):
    """An entity with n wheel links 'w0'..; wheel i's link idx 10 + i, vgeom idx i."""
    links = {}
    for i in range(n):
        vg = _Geom(e, 10 + i, i, pos=local)
        cg = [_Geom(e, 10 + i, i, pos=(0.0, -0.2, 0.0))] if with_collider_key_clash else []
        links[f"w{i}"] = _Link([vg], cg)
    return _Entity(links, vis_mode)


def _renderer(ctx, ents, n=2, R=2):
    r = V.InstancedWheelRenderer(_Scene(ctx, n_envs=R))
    r.add_unit(ents, [_Wheel(f"w{i}") for i in range(n)], _provider(len(ents), n, R), K=len(ents))
    r._ctx = ctx
    return r


def _register_engine_nodes(ctx, ents, n=2, R=2):
    nodes = {}
    for k, ent in enumerate(ents):
        for i in range(n):
            vg = ent.get_link(f"w{i}").vgeoms[0]
            node = _Node(R)
            ctx.nodes_by_seg_keys[vg.seg_keys] = [node]
            nodes[(k, i)] = (node, vg)
    return nodes


def _closed(r, k, i):
    p, q = r._units[0]["provider"]("world")
    return V._pose_mats(p[:, k, i].double().numpy(), q[:, k, i].double().numpy(), np.float64)


# --- tests ------------------------------------------------------------------

def test_engine_nodes_get_closed_form_times_local_offset():
    ctx = _Ctx()
    ents = [_car(0), _car(1)]
    nodes = _register_engine_nodes(ctx, ents)
    r = _renderer(ctx, ents)
    r.update()
    for (k, i), (node, vg) in nodes.items():
        want = _closed(r, k, i) @ V._local_T(vg)
        got = node.mesh.primitives[0].poses
        assert np.abs(got - want).max() <= 1e-6
        np.testing.assert_array_equal(ctx.jit._buffer[(node.mesh.primitives[0], "model")],
                                      got.transpose(0, 2, 1))
    assert ctx.added == [] and ctx.seg == []          # no fallback when the engine draws


def test_a_drawn_part_without_an_engine_node_raises_instead_of_a_fallback():
    """A3 mutant: a perturbed lookup key (here: the lookup is empty) must raise,
    never draw the wheel a second time through a fallback."""
    ctx = _Ctx()
    ents = [_car(0)]
    _register_engine_nodes(ctx, ents)
    key = next(iter(ctx.nodes_by_seg_keys))
    ctx.nodes_by_seg_keys[key._replace(entity=99)] = ctx.nodes_by_seg_keys.pop(key)
    r = _renderer(ctx, ents)
    with pytest.raises(RuntimeError, match="no render node was found"):
        r.update()
    assert ctx.added == []


def test_dynamic_vertices_wheel_visual_raises_value_error():
    """C27: not reachable through VehicleScene.add_vehicle; defensive."""
    ctx = _Ctx()
    ents = [_car(0)]
    _register_engine_nodes(ctx, ents)
    ents[0].get_link("w1").vgeoms[0].desc = type("D", (), {"dynamic_vertices": object()})()
    with pytest.raises(ValueError, match="DynamicVertices"):
        _renderer(ctx, ents).update()


def test_two_parts_with_different_offsets_on_one_node_raise():
    """C25: one engine node reached from two parts with different T_local."""
    ctx = _Ctx()
    ent = _car(0, n=1)
    link = ent.get_link("w0")
    extra = _Geom(0, 10, 0, pos=(0.0, 0.5, 0.0))      # same keys, different offset
    link.vgeoms.append(extra)
    node = _Node(2)
    ctx.nodes_by_seg_keys[link.vgeoms[0].seg_keys] = [node]
    with pytest.raises(RuntimeError, match="two wheel parts"):
        _renderer(ctx, [ent], n=1).update()


def test_visual_mode_ignores_a_collider_sharing_the_visual_geom_key():
    """A6: in visual mode only visual geoms are parts, so a collider whose GeomKey
    equals its visual geom's (two index spaces) does not pose the node."""
    ctx = _Ctx()
    ents = [_car(0, with_collider_key_clash=True)]
    nodes = _register_engine_nodes(ctx, ents)
    r = _renderer(ctx, ents)
    r.update()
    node, vg = nodes[(0, 1)]
    want = _closed(r, 0, 1) @ V._local_T(vg)
    assert np.abs(node.mesh.primitives[0].poses - want).max() <= 1e-6


def test_collision_mode_with_colliders_poses_the_collision_parts():
    ctx = _Ctx()
    ent = _car(0, vis_mode="collision", with_collider_key_clash=True)
    nodes = {}
    for i in range(2):
        cg = ent.get_link(f"w{i}").geoms[0]
        nodes[i] = (_Node(2), cg)
        ctx.nodes_by_seg_keys[cg.seg_keys] = [nodes[i][0]]
    r = _renderer(ctx, [ent])
    r.update()
    for i, (node, cg) in nodes.items():
        want = _closed(r, 0, i) @ V._local_T(cg)
        assert np.abs(node.mesh.primitives[0].poses - want).max() <= 1e-6
    assert ctx.added == []


def test_collision_mode_without_colliders_makes_ordered_registered_fallbacks():
    """D-S2: the engine draws no wheel node (no colliders after urdf_prep); one
    fallback per (k, i), in (unit, k, i) order, registered under the link's
    first visual geom at the context level, NOT added to nodes_by_seg_keys."""
    ctx = _Ctx(level="geom")
    ents = [_car(0, vis_mode="collision"), _car(1, vis_mode="collision")]
    r = _renderer(ctx, ents)
    r.update()
    assert len(ctx.added) == 4 and ctx.nodes_by_seg_keys == {}
    want_keys = [ents[k].get_link(f"w{i}").vgeoms[0].seg_keys.geom for k in range(2) for i in range(2)]
    assert [k for k, _ in ctx.seg] == want_keys
    assert [n for _, n in ctx.seg] == ctx.added
    for (k, i), node in r._units[0]["fallback"].items():
        assert np.abs(node.mesh.primitives[0].poses - _closed(r, k, i)).max() <= 1e-6


def test_an_unresolvable_wheel_gets_an_unregistered_cylinder_and_one_warning(caplog):
    ctx = _Ctx()
    ent = _car(0, n=1)
    r = V.InstancedWheelRenderer(_Scene(ctx))
    r.add_unit([ent], [_Wheel("nope", radius=0.4)], _provider(1, 1, 2), K=1)
    r._ctx = ctx
    V._HARVEST_WARNED = False
    import logging
    lg = logging.getLogger("genesis_vehicle")
    recs = []
    h = logging.Handler(); h.emit = recs.append
    lg.addHandler(h)
    try:
        r.update()
    finally:
        lg.removeHandler(h)
    assert len(ctx.added) == 1 and ctx.seg == []
    msgs = [x.getMessage() for x in recs if "wheel-harvest" in x.getMessage()]
    assert len(msgs) == 1 and "drawn as well (twice)" in msgs[0]


def test_rebind_checks_entity_identity_and_changes_nothing_on_mismatch():
    """A7: K and n alone would not notice two same-shaped kinds swapping places."""
    ctx = _Ctx()
    ents = [_car(0), _car(1)]
    r = _renderer(ctx, ents)
    old = r._units[0]["provider"]
    new = _provider(2, 2, 2)
    with pytest.raises(ValueError, match="entity mismatch"):
        r.rebind([(new, 2, 2, [ents[1], ents[0]])])
    assert r._units[0]["provider"] is old
    with pytest.raises(ValueError, match="shape mismatch"):
        r.rebind([(new, 2, 3, ents)])
    r.rebind([(new, 2, 2, ents)])
    assert r._units[0]["provider"] is new


def test_the_ordering_check_catches_a_write_after_the_hook():
    """A2 mutant: a wrapper that runs the SDK before the engine's pose write. On
    the next sync the posed primitive no longer holds the SDK's array -> raise."""
    ctx = _Ctx()
    ents = [_car(0)]
    nodes = _register_engine_nodes(ctx, ents)

    def engine_sync():                    # the engine re-poses every wheel node
        for node, _ in nodes.values():
            node.mesh.primitives[0].poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    ctx._sync_shapes = engine_sync
    r = _renderer(ctx, ents)
    original = ctx._sync_shapes

    def mutant():                         # fn BEFORE the original
        r._check_last_writes(); r.update(); original()
    ctx._sync_shapes = mutant
    ctx._sync_shapes()
    with pytest.raises(RuntimeError, match="were changed before this one"):
        ctx._sync_shapes()                # one sync late: the raise is on the 2nd sync

    # The real wrapper (engine first, SDK after) never trips it.
    ctx2 = _Ctx()
    nodes2 = _register_engine_nodes(ctx2, ents)

    def engine_sync2():
        for node, _ in nodes2.values():
            node.mesh.primitives[0].poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    ctx2._sync_shapes = engine_sync2
    r2 = _renderer(ctx2, ents)
    C.on_render_sync(ctx2, r2.update, before=r2._check_last_writes)
    for _ in range(3):
        ctx2._sync_shapes()


def test_the_ordering_guard_catches_an_in_place_write_after_the_hook():
    """R2a: an engine that writes INTO the primitive's existing poses array
    after the hook mutates the SDK's own array, which an identity test would
    miss; the guard compares with a private copy by value. Detected one sync
    late, through the real wrapper."""
    ctx = _Ctx()
    ents = [_car(0)]
    nodes = _register_engine_nodes(ctx, ents)
    r = _renderer(ctx, ents)
    C.on_render_sync(ctx, r.update, before=r._check_last_writes)
    hooked = ctx._sync_shapes

    def late_in_place():                  # the engine's write moved after the hook
        hooked()
        for node, _ in nodes.values():
            node.mesh.primitives[0].poses[...] = np.eye(4, dtype=np.float32)
    late_in_place()                       # frame drawn at the "solver" pose, no raise yet
    with pytest.raises(RuntimeError, match="one sync late"):
        late_in_place()


def test_the_ordering_guard_does_not_depend_on_the_poses_setter_keeping_the_object():
    """R2a: a ``poses`` setter that copies its input (a future pyrender) must not
    make the guard raise: it compares values, not identity."""
    class _CopyPrim:
        def __init__(self, R):
            self._p = np.tile(np.eye(4, dtype=np.float32), (R, 1, 1))

        @property
        def poses(self):
            return self._p

        @poses.setter
        def poses(self, v):
            self._p = np.array(v, dtype=np.float32, copy=True)

    ctx = _Ctx()
    ents = [_car(0)]
    nodes = _register_engine_nodes(ctx, ents)
    for node, _ in nodes.values():
        node.mesh.primitives = [_CopyPrim(2)]

    def engine_sync():
        for node, _ in nodes.values():
            node.mesh.primitives[0].poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    ctx._sync_shapes = engine_sync
    r = _renderer(ctx, ents)
    C.on_render_sync(ctx, r.update, before=r._check_last_writes)
    for _ in range(3):
        ctx._sync_shapes()


def test_sdf_vis_mode_with_wheel_colliders_raises_naming_the_workaround():
    """R3: only 'sdf' reaches the renderer among non-None, non-'collision'
    vis_modes on 1.5.0; with wheel colliders (an unprepared URDF) its draw layout
    was never run, so it raises; without colliders it takes the fallback."""
    ctx = _Ctx()
    ent = _car(0, vis_mode="sdf", with_collider_key_clash=True)
    with pytest.raises(ValueError) as ei:
        _renderer(ctx, [ent]).update()
    msg = str(ei.value)
    assert msg.startswith(C.WHEEL_RENDER_TAG)
    assert "prepare_vehicle_urdf" in msg and "vis_mode=None / 'collision'" in msg
    assert "headless scenes are unaffected" in msg and "build()" in msg
    assert "verified" not in msg and "'visual'" not in msg
    ctx2 = _Ctx()
    ent2 = _car(0, vis_mode="sdf")                   # prepared URDF: no wheel colliders
    r2 = _renderer(ctx2, [ent2])
    r2.update()
    assert len(ctx2.added) == 2 and len(ctx2.seg) == 2


def test_an_engine_node_with_the_wrong_instance_count_raises():
    ctx = _Ctx(R=2)
    ents = [_car(0)]
    nodes = _register_engine_nodes(ctx, ents, R=3)
    with pytest.raises(RuntimeError, match="instances"):
        _renderer(ctx, ents).update()


def test_removed_names_raise_import_error_with_the_replacement():
    """3b table / A13: both import forms keep the message; hasattr raises."""
    for name in ("WheelJointInternalSync", "KindVisualBatch"):
        with pytest.raises(ImportError, match="removed in 2.0.6"):
            exec(f"from genesis_vehicle.visual import {name}", {})
        with pytest.raises(ImportError, match="VehicleScene"):
            getattr(V, name)
        with pytest.raises(ImportError):
            hasattr(V, name)


# --- fix round 1: non-finite rows and the guard state machine (F-C1 .. F-C3) ---

_GUARD = "were changed before this one"


def _hooked(engine_writes=True, R=2):
    """A renderer wired through the REAL wrapper on a fake context whose engine
    ``_sync_shapes`` re-poses every wheel node at the identity ("solver pose"),
    as 1.5.0 does on a sync after a state change. Returns (ctx, r, nodes, state);
    ``state["engine_raise"]`` / ``state["prov_raise"]`` make the next engine call
    / provider call raise once."""
    ctx = _Ctx(R=R)
    ents = [_car(0)]
    nodes = _register_engine_nodes(ctx, ents, R=R)
    state = dict(engine_raise=False, prov_raise=False, nan=None)

    def engine_sync():
        if engine_writes:
            for node, _ in nodes.values():
                node.mesh.primitives[0].poses = np.tile(np.eye(4, dtype=np.float32), (R, 1, 1))
        if state["engine_raise"]:
            state["engine_raise"] = False
            raise ValueError("engine sync failed")
    ctx._sync_shapes = engine_sync
    r = _renderer(ctx, ents, R=R)
    base = r._units[0]["provider"]
    calls = dict(n=0)

    def prov(frame="world"):
        if state["prov_raise"]:
            state["prov_raise"] = False
            raise RuntimeError("transient provider failure")
        calls["n"] += 1
        p, q = base(frame)
        p = p.clone(); p[..., 2] += 0.01 * calls["n"]          # a new pose on every sync
        if state["nan"] is not None:
            e, k, i, val = state["nan"]
            p[e, k, i, 0] = val
        state["last"] = (p, q)
        return p, q
    r._units[0]["provider"] = prov
    C.on_render_sync(ctx, r.update, before=r._check_last_writes)
    return ctx, r, nodes, state


def _poses(nodes, k, i):
    return np.array(nodes[(k, i)][0].mesh.primitives[0].poses, copy=True)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1e21])
def test_a_non_finite_provider_row_keeps_the_last_written_row_and_nothing_raises(bad):
    """F-C3 (i): env 0 of wheel (0, 0) goes non-finite (or past the engine's
    POSE_MAX_ABS) after a completed sync: that instance keeps the row the SDK
    wrote on the previous sync (NOT the engine's identity written in between),
    the other env keeps following the closed form, the queued buffer is the
    transposed written array, and no later sync raises."""
    ctx, r, nodes, state = _hooked()
    ctx._sync_shapes()
    before = _poses(nodes, 0, 0)
    state["nan"] = (0, 0, 0, bad)
    for _ in range(3):
        ctx._sync_shapes()
        got = _poses(nodes, 0, 0)
        assert np.isfinite(got).all()
        np.testing.assert_array_equal(got[0], before[0])
        assert not np.array_equal(got[1], before[1])            # env 1 moved on
        prim = nodes[(0, 0)][0].mesh.primitives[0]
        np.testing.assert_array_equal(ctx.jit._buffer[(prim, "model")], got.transpose(0, 2, 1))
        other = _poses(nodes, 0, 1)
        p, q = state["last"]
        want = V._pose_mats(p[:, 0, 1].double().numpy(), q[:, 0, 1].double().numpy(),
                            np.float64) @ V._local_T(nodes[(0, 1)][1])
        assert np.abs(other - want).max() <= 1e-6


def test_a_finite_to_nan_in_place_write_is_still_caught():
    """F-C3 (ii): ``equal_nan=True`` hides nothing: a NaN written in place into a
    recorded primitive after a completed sync raises on the next sync."""
    ctx, r, nodes, _ = _hooked()
    ctx._sync_shapes()
    nodes[(0, 1)][0].mesh.primitives[0].poses[1, 0, 3] = np.nan
    with pytest.raises(RuntimeError, match=_GUARD):
        ctx._sync_shapes()


def test_an_engine_sync_that_raises_once_leaves_one_unchecked_sync_then_checks_again():
    """F-C3 (iii): the engine writes the solver pose and then raises. The
    exception propagates; the next sync does not raise the guard error (a
    record kept across the failed sync would compare the engine's write with the
    SDK's old array); a late write after that sync raises on the one after."""
    ctx, r, nodes, state = _hooked()
    ctx._sync_shapes()
    state["engine_raise"] = True
    with pytest.raises(ValueError, match="engine sync failed"):
        ctx._sync_shapes()
    ctx._sync_shapes()                                   # no guard error
    nodes[(0, 0)][0].mesh.primitives[0].poses[...] = np.eye(4, dtype=np.float32)
    with pytest.raises(RuntimeError, match=_GUARD):
        ctx._sync_shapes()


def test_a_provider_that_raises_once_propagates_once_then_checking_resumes():
    """F-C3 (iv) / F1-a: the provider raises once after the engine re-posed the
    nodes. The tagged exception propagates once; the next sync does not raise
    the guard error; a late write after that sync raises on the one after."""
    ctx, r, nodes, state = _hooked()
    ctx._sync_shapes()
    state["prov_raise"] = True
    with pytest.raises(RuntimeError, match="transient provider failure") as ei:
        ctx._sync_shapes()
    assert str(ei.value).startswith(C.WHEEL_RENDER_TAG)
    ctx._sync_shapes()                                   # no guard error
    ctx._sync_shapes()                                   # checked, clean
    nodes[(0, 1)][0].mesh.primitives[0].poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    with pytest.raises(RuntimeError, match=_GUARD):
        ctx._sync_shapes()


def test_a_guard_raise_on_a_real_late_write_does_not_poison_the_next_sync():
    """F-C3 (v): a real late write raises once; with a clean engine afterwards the
    next sync is clean (the record was discarded by the raising check)."""
    ctx, r, nodes, _ = _hooked()
    ctx._sync_shapes()
    nodes[(0, 0)][0].mesh.primitives[0].poses[...] = np.eye(4, dtype=np.float32)
    with pytest.raises(RuntimeError, match=_GUARD):
        ctx._sync_shapes()
    for _ in range(3):
        ctx._sync_shapes()


def test_pose_max_abs_reads_the_engine_constant_without_a_default(monkeypatch):
    """F-C1: the bound is the engine's own ``POSE_MAX_ABS`` (1e20 on 1.5.0); a
    missing constant raises the prefixed RuntimeError, never a fallback value."""
    from genesis.vis import rasterizer_context as rc
    assert C.pose_max_abs() == float(rc.POSE_MAX_ABS) == 1e20
    monkeypatch.delattr(rc, "POSE_MAX_ABS")
    with pytest.raises(RuntimeError, match="POSE_MAX_ABS") as ei:
        C.pose_max_abs()
    assert str(ei.value).startswith(C.WHEEL_RENDER_TAG)
