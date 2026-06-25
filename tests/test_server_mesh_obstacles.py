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


# ---------------------------------------------------------------------------
# End-to-end: every mesh-obstacle path REGISTERS without NameError / crash
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cpu_genesis():
    if not getattr(gs, "_initialized", False):
        try:
            gs.init(backend=gs.cpu, logging_level="warning")
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
    vs = VehicleScene(n_envs=1, backend="cpu", raycast_mode="dual_scene",
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
    vs = VehicleScene(n_envs=1, backend="cpu", raycast_mode="dual_scene",
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
    vs = VehicleScene(n_envs=1, backend="cpu", raycast_mode="dual_scene",
                      init_genesis=False)
    sb = vs.add_static(morph=gs.morphs.Plane(),
                       material=gs.materials.Rigid(friction=1.0), vis_mode="collision")
    vs.build()   # must NOT raise AttributeError('KinematicEntity' ... 'geoms')
    assert sb.entity_raycast.surface.vis_mode == "visual"   # mirror forced visual
    assert sb.entity_main.surface.vis_mode == "collision"   # caller honored for main


def test_rco_road_single_scene_no_spurious_warning(cpu_genesis, cube_obj, caplog):
    """0.9.6 guard refinement: in single_scene the rco road's wheel_raycast_morph
    is the sole geometry (the raycast body), so the 'ignored in single_scene'
    warning must NOT fire."""
    import logging
    vs = VehicleScene(n_envs=1, backend="cpu", raycast_mode="single_scene",
                      init_genesis=False)
    with caplog.at_level(logging.WARNING, logger="genesis_vehicle.vehicle_scene"):
        env_builder.build_obstacles(
            vs=vs, init_data=_mesh_init(0, "[Complex]", cube_obj),
            ue_friction=1.0, ue_restitution=0.0, vis_mode=None, road_raycast_only=True)
    assert not any("wheel_raycast_morph" in r.getMessage() for r in caplog.records)
    assert vs.statics[0].entity_main is not None   # single_scene: the body exists
