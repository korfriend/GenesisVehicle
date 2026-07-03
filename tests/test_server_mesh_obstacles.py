"""Regression tests for the server's MESH-obstacle paths through
``env_builder.build_obstacles``.

Why this file exists
--------------------
The 0.9.0 encapsulation refactor was validated only with PRIMITIVE box
obstacles, which skip the mesh morph-building branch. That branch is where
``road_raycast_only`` is read and where the trimesh double-siding runs — so two
real bugs hid there until UE integration hit real road meshes:

  * 0.9.3 — ``road_raycast_only`` was referenced but dropped from the signature
            → ``NameError`` on any ``obs_type == 5`` / ``[Complex]`` mesh.
  * 0.9.4 — ``make_double_sided_mesh`` called ``remove_degenerate_faces()`` /
            ``remove_duplicate_faces()``, removed in trimesh >= 4 → the road
            double-siding silently fell back to the single-sided mesh.

These tests exercise the mesh paths so that class of bug can't regress silently.

Genesis-gated: importing ``env_builder`` needs ``import genesis`` (fine on CPU);
the end-to-end path additionally inits Genesis on the CPU backend. Both are
``importorskip`` / ``skip``-guarded so a Genesis-less environment skips cleanly.
"""

import inspect
import os
import tempfile

import pytest

pytest.importorskip("genesis")
trimesh = pytest.importorskip("trimesh")

import genesis as gs

from genesis_vehicle import VehicleScene
from genesis_vehicle.server import env_builder


# ---------------------------------------------------------------------------
# Cheap, no-gs.init guards on the two exact regressions
# ---------------------------------------------------------------------------

def test_build_obstacles_keeps_road_raycast_only_param():
    """0.9.3 guard: the mesh branch reads ``road_raycast_only``; it must stay a
    parameter (a default value is fine) or that read NameErrors again."""
    params = inspect.signature(env_builder.build_obstacles).parameters
    assert "road_raycast_only" in params
    assert params["road_raycast_only"].default is False


def test_make_double_sided_mesh_doubles_without_legacy_api():
    """0.9.4 guard: doubling a closed mesh must succeed (no removed trimesh API)
    and roughly double the face count (front + back), not fall back to the input."""
    cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))   # 12 faces
    out = env_builder.make_double_sided_mesh(cube, thickness=0.001)
    assert len(out.faces) > len(cube.faces)       # actually doubled, not fallback
    assert len(out.faces) <= 2 * len(cube.faces)  # (minus any degenerate/dupes)


def test_build_cfg_shares_one_object_per_identical_target():
    """1.0.8 guard: the batched solver groups vehicles into kinds by cfg OBJECT
    identity, so identical targets (same urdf/mapping/friction) must get the
    SAME cfg instance. Per-target fresh cfgs split K vehicles into K kinds × 1
    vehicle — the batched pipeline never engages and the server's SDK compute
    scales ×K (measured: 10 tanks CPU 37.8 ms/step vs 2.8 ms as one kind)."""
    from genesis_vehicle.server import vehicle_builder
    urdf = os.path.join(os.path.dirname(env_builder.__file__),
                        "..", "samples", "urdf", "car_4w.urdf")
    urdf = os.path.abspath(urdf)
    mapping = {"driveType": 0}
    a = vehicle_builder.build_cfg(urdf, mapping, 1.0, target_id=0)
    b = vehicle_builder.build_cfg(urdf, mapping, 1.0, target_id=1)
    assert a is b                       # one kind → one batched pipeline
    c = vehicle_builder.build_cfg(urdf, mapping, 2.0, target_id=2)
    assert c is not a                   # different friction → its own kind


# ---------------------------------------------------------------------------
# End-to-end: every mesh-obstacle path REGISTERS without NameError / crash
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cpu_genesis():
    if not getattr(gs, "_initialized", False):
        try:
            VehicleScene.init_backend("cpu")
        except Exception as e:                     # pragma: no cover
            pytest.skip(f"genesis CPU backend unavailable: {e}")
    return gs


@pytest.fixture(scope="module")
def cube_obj():
    cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    fd, path = tempfile.mkstemp(suffix=".obj")
    os.close(fd)
    cube.export(path)
    yield path
    try:
        os.remove(path)
    except OSError:
        pass


def _mesh_init(b_dynamic, col_src, mesh_path):
    return {"obstacles": {0: dict(
        type=5, pos=[5.0, 0.0, 0.5], quat=[1.0, 0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0],
        collision_source=col_src, mesh_path=mesh_path, b_dynamic=b_dynamic,
        mass=2.0, friction=-1.0, restitution=-1.0)}}


# (name, b_dynamic, collision_source, build_kwargs, n_static, n_dynamic, is_ue)
MESH_PATHS = [
    ("road_raycast_only",     0, "[Complex]",     {"road_raycast_only": True},        1, 0, False),
    ("road_full_collision",   0, "[Complex]",     {"road_raycast_only": False},       1, 0, False),
    ("structures_primitive",  0, "[Complex]",     {"structures_as_primitive": True},  1, 0, False),
    ("convex_mesh",           0, "[User:Convex]", {},                                 1, 0, False),
    ("dynamic_mesh",          1, "[Complex]",     {},                                 0, 1, False),
    ("ue_driven_mesh",        2, "[Complex]",     {},                                 0, 1, True),
]


@pytest.mark.parametrize("name,b_dynamic,col_src,kw,n_static,n_dynamic,is_ue", MESH_PATHS)
def test_build_obstacles_mesh_path_registers(cpu_genesis, cube_obj, name, b_dynamic,
                                             col_src, kw, n_static, n_dynamic, is_ue):
    """Each mesh-obstacle path runs the morph-building + vs.add_* routing and
    registers the body (no NameError, no crash). Registration-only — the bugs all
    live before scene build, so we skip the (multi-scene) build for speed."""
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene",
                      init_genesis=False)
    obstacles, dyn, init_states, ue_ids, extra_mass = env_builder.build_obstacles(
        vs=vs, init_data=_mesh_init(b_dynamic, col_src, cube_obj),
        ue_friction=1.0, ue_restitution=0.0, vis_mode=None, **kw)

    assert len(obstacles) == 1
    assert len(vs.statics) == n_static
    assert len(vs.dynamics) == n_dynamic
    if b_dynamic in (1, 2):
        assert sorted(dyn) == [0]
    assert sorted(ue_ids) == ([0] if is_ue else [])


def test_rco_road_dual_scene_has_no_redundant_main_rigid(cpu_genesis, cube_obj):
    """0.9.6 cleanup: a road_raycast_only road in dual_scene is the kinematic
    raycast mirror ONLY — no redundant no-collision rigid in the main scene."""
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene",
                      init_genesis=False)
    env_builder.build_obstacles(
        vs=vs, init_data=_mesh_init(0, "[Complex]", cube_obj),
        ue_friction=1.0, ue_restitution=0.0, vis_mode=None, road_raycast_only=True)
    body = vs.statics[0]
    assert body.entity_main is None          # no main-scene collider built
    assert body.entity_raycast is not None   # just the raycast mirror


def test_kinematic_mirror_builds_with_collision_vis_mode(cpu_genesis):
    """A dual_scene kinematic raycast mirror must build even when the caller passes
    a non-'visual' vis_mode (e.g. the server's --vis-mode=collision). A
    KinematicEntity is visual-only (vgeoms, no collision geoms), so the renderer's
    on_rigid must take the vgeoms path; a 'collision' vis_mode made it touch the
    missing KinematicEntity.geoms and AttributeError at build. The mirror's
    vis_mode is forced to 'visual'; the caller's vis_mode is kept for the main
    (rendered) entity only."""
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene",
                      init_genesis=False)
    sb = vs.add_static(morph=gs.morphs.Plane(),
                       material=gs.materials.Rigid(friction=1.0), vis_mode="collision")
    vs.build()   # must NOT raise AttributeError('KinematicEntity' ... 'geoms')
    assert sb.entity_raycast.surface.vis_mode == "visual"   # mirror forced visual
    assert sb.entity_main.surface.vis_mode == "collision"   # caller honored for main


def test_add_raycast_surface_dual_scene(cpu_genesis, cube_obj):
    """1.0.10: add_raycast_surface is the first-class raycast-only entity —
    kinematic use_visual_raycasting body in the raycast scene, ZERO main-scene
    (rigid-solver) presence. The rco road path now goes through it."""
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene", init_genesis=False)
    body = vs.add_raycast_surface(
        gs.morphs.Mesh(file=cube_obj, fixed=True, collision=False,
                       convexify=False, decimate=False),
        name="road")
    assert body.entity_main is None            # no rigid-solver presence
    assert body.entity_raycast is not None     # raycast-scene kinematic body
    assert not body.has_collision
    assert type(body.entity_raycast).__name__ == "KinematicEntity"


def test_add_raycast_surface_single_scene_fails_fast(cpu_genesis, cube_obj):
    """single_scene rays only hit rigid collision geoms → same fail-fast as
    add_static(collision=False)."""
    import pytest
    vs = VehicleScene(n_envs=1, raycast_mode="single_scene", init_genesis=False)
    with pytest.raises(ValueError, match="dual_scene"):
        vs.add_raycast_surface(
            gs.morphs.Mesh(file=cube_obj, fixed=True, collision=False,
                           convexify=False, decimate=False))


def test_collision_false_single_scene_fails_fast(cpu_genesis, cube_obj):
    """1.0.7: collision=False is a dual_scene-only feature (kinematic raycast
    surface in the raycast scene). In single_scene the wheel rays only hit rigid
    collision geoms, so a no-collision static would be a fall-through surface —
    pre-1.0.7 this warned and built a rigid from the rco road morph (which itself
    carries collision=False): zero collision geoms, invisible to the raycaster,
    vehicles fell straight through rco roads in the per-entity (inline) server.
    add_static must now refuse (fail fast) instead of building a broken scene.
    (The server pairs --road-raycast-only with raycast_mode='dual_scene'.)"""
    import pytest
    vs = VehicleScene(n_envs=1, raycast_mode="single_scene",
                      init_genesis=False)
    with pytest.raises(ValueError, match="dual_scene"):
        env_builder.build_obstacles(
            vs=vs, init_data=_mesh_init(0, "[Complex]", cube_obj),
            ue_friction=1.0, ue_restitution=0.0, vis_mode=None, road_raycast_only=True)
