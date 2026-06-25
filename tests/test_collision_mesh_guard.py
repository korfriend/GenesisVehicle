"""Unit tests for the non-convex collision-mesh guard (_guard_collision_mesh).

Pure-Python: no Genesis / GPU. The guard keys on ``type(morph).__name__`` and
duck-typed attributes (``file`` / ``convexify`` / ``collision``), so we stand in
a tiny ``Mesh`` class and real temp .obj files of known face count.
"""

import os
import tempfile

import numpy as np
import pytest

from genesis_vehicle.vehicle_scene import (
    _guard_collision_mesh, _MAX_NONCONVEX_COLLISION_FACES,
)

trimesh = pytest.importorskip("trimesh")


class Mesh:  # stand-in: the guard checks type(morph).__name__ == "Mesh"
    def __init__(self, file, convexify=False, collision=True):
        self.file = file
        self.convexify = convexify
        self.collision = collision


class Box:  # a non-Mesh morph (primitive) — always exempt
    def __init__(self):
        self.collision = True


def _grid_obj(side: int) -> str:
    """An (side x side)-vertex grid mesh → (side-1)^2 * 2 faces; temp .obj path."""
    xs, ys = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(side * side)], axis=1)
    faces = []
    for i in range(side - 1):
        for j in range(side - 1):
            a = i * side + j; b = a + 1; c = a + side; d = c + 1
            faces.append([a, c, b]); faces.append([b, c, d])
    m = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces))
    fd, path = tempfile.mkstemp(suffix=".obj"); os.close(fd)
    m.export(path)
    return path


@pytest.fixture
def big_obj():
    p = _grid_obj(40)              # 39*39*2 = 3042 faces (> 1000)
    yield p
    os.unlink(p)


@pytest.fixture
def small_obj():
    p = _grid_obj(10)             # 9*9*2 = 162 faces (< 1000)
    yield p
    os.unlink(p)


def test_big_nonconvex_collision_mesh_is_refused(big_obj):
    with pytest.raises(ValueError, match="refusing to build"):
        _guard_collision_mesh(Mesh(big_obj, convexify=False, collision=True),
                              "add_static('terrain')")


def test_small_mesh_is_allowed(small_obj):
    # Under the face limit → no raise.
    _guard_collision_mesh(Mesh(small_obj, convexify=False, collision=True), "x")


def test_convexify_true_is_exempt(big_obj):
    # Convex decomposition keeps collision cheap → not guarded even when big.
    _guard_collision_mesh(Mesh(big_obj, convexify=True, collision=True), "x")


def test_collision_false_is_exempt(big_obj):
    # Visual-only / kinematic raycast surface (no SDF) → exempt.
    _guard_collision_mesh(Mesh(big_obj, convexify=False, collision=False), "x")


def test_non_mesh_primitive_is_exempt():
    # A Box (or any non-"Mesh" morph) never trips the guard.
    _guard_collision_mesh(Box(), "x")


def test_missing_file_is_best_effort():
    # Can't introspect a missing file → let it through (no raise).
    _guard_collision_mesh(Mesh("/nonexistent/none.obj", convexify=False), "x")


def test_face_limit_is_a_thousand():
    assert _MAX_NONCONVEX_COLLISION_FACES == 1000
