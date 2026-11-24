"""Per-env heterogeneous variants for obstacles and raycast surfaces (2.0.5).

genesis-world 1.5.0 builds a "heterogeneous" entity from a list of morphs and
gives each env one of them (contiguous blocks of nearly equal size). The SDK
routes such a list to the main body and to its dual_scene mirror, and reports
the env -> variant mapping as ``body.variant_index`` from the engine's own rule
(``_gs_compat.variant_index``). What is tested here:

* the mapping shim against the engine function, and that it raises if the
  engine symbol is gone;
* every refusal (before build): vehicle variants, Terrain / Plane in a list,
  mismatched / mixed lists, k > n_envs, free-body (``physics=True``) variants;
  k == 1 is a single morph;
* H1 (built scene): ``variant_index`` equals the variant the engine actually
  placed in each env, read back from per-env AABB heights;
* api-review condition 5: the variant morphs are copied per element (no
  "could not copy morph" fallback), ``fixed`` is set on every built variant of
  the main body and of the mirror, and the refit-cost check looks at each
  element (no false warning on a list of primitives).

Ray-level and collision behaviour per env (AC-HET H2-H4) is measured by the
release harness, not here.
"""
import logging

import pytest
import torch

gs = pytest.importorskip("genesis")

from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann            # noqa: E402
from genesis_vehicle._gs_compat import variant_index                      # noqa: E402


class _Capture(logging.Handler):
    """Records WARNING+ on the ``genesis_vehicle`` logger (not caplog: gs.init
    replaces the logging configuration; see test_bump_stop_dt_warning.py)."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())

    def __enter__(self):
        logging.getLogger("genesis_vehicle").addHandler(self)
        return self

    def __exit__(self, *exc):
        logging.getLogger("genesis_vehicle").removeHandler(self)


def _scene(n_envs, mode="dual_scene"):
    VehicleScene.init_backend("cpu")
    return VehicleScene(n_envs=n_envs, dt=0.025, substeps=4, raycast_mode=mode, show_viewer=False)


def _box(h, x=0.0):
    return gs.morphs.Box(size=(1.0, 1.0, h), pos=(x, 0.0, h / 2), fixed=True)


# --- the mapping shim (api-review condition 7) --------------------------------

def test_variant_index_matches_the_engine_function():
    from genesis.engine.entities.rigid_entity.description import _balanced_variant_mapping
    assert variant_index(3, 6).tolist() == [0, 0, 1, 1, 2, 2]
    assert variant_index(3, 3).tolist() == [0, 1, 2]
    assert variant_index(5, 7).tolist() == list(_balanced_variant_mapping(5, 7))
    assert variant_index(5, 7).tolist() == [0, 0, 1, 1, 2, 3, 4]
    assert variant_index(2, 0).tolist() == [0]          # unbatched scene: one slot
    assert variant_index(3, 6).dtype == torch.long


def test_variant_index_raises_naming_the_engine_symbol_when_it_is_gone(monkeypatch):
    import genesis.engine.entities.rigid_entity.description as desc
    monkeypatch.delattr(desc, "_balanced_variant_mapping")
    with pytest.raises(ImportError, match="_balanced_variant_mapping"):
        variant_index(3, 6)


# --- refusals, all before build (AC-HET H5) ------------------------------------

def test_a_vehicle_variant_list_is_refused():
    import os
    import genesis_vehicle
    urdf = os.path.join(os.path.dirname(genesis_vehicle.__file__), "samples", "urdf", "car_4w.urdf")
    vs = _scene(3)
    with pytest.raises(TypeError, match="per-env vehicle variants are not supported"):
        vs.add_vehicle([urdf, urdf], car_4w_rwd_ackermann)
    with pytest.raises(TypeError, match="per-env vehicle variants are not supported"):
        vs.add_vehicle(urdf, car_4w_rwd_ackermann, morph=[gs.morphs.URDF(file=urdf)])


@pytest.mark.parametrize("bad", ["terrain", "plane"])
def test_terrain_and_plane_are_refused_in_a_variant_list(bad):
    vs = _scene(3)
    m = (gs.morphs.Terrain(n_subterrains=(1, 1), subterrain_size=(8.0, 8.0), horizontal_scale=0.5,
                           vertical_scale=0.005, subterrain_types=[["flat_terrain"]])
         if bad == "terrain" else gs.morphs.Plane())
    with pytest.raises(ValueError, match="not supported in a per-env variant list"):
        vs.add_static(morph=[_box(0.1), m])


def test_variant_lists_of_different_lengths_or_mixed_with_a_single_morph_are_refused():
    vs = _scene(3)
    with pytest.raises(ValueError, match="lengths differ"):
        vs.add_static(collision_morph=[_box(0.1), _box(0.2)],
                      wheel_raycast_morph=[_box(0.1), _box(0.2), _box(0.3)])
    with pytest.raises(ValueError, match="single morph"):
        vs.add_static(collision_morph=[_box(0.1), _box(0.2)], wheel_raycast_morph=_box(0.1))
    with pytest.raises(ValueError, match="empty variant list"):
        vs.add_static(morph=[])


def test_more_variants_than_envs_is_refused():
    vs = _scene(2)
    with pytest.raises(ValueError, match="3 variants for n_envs=2"):
        vs.add_static(morph=[_box(0.1), _box(0.2), _box(0.3)])
    with pytest.raises(ValueError, match="3 variants for n_envs=2"):
        vs.add_dynamic([_box(0.1), _box(0.2), _box(0.3)], physics=False)


@pytest.mark.parametrize("mass", [None, 5.0])
def test_free_body_variants_are_refused(mass):
    vs = _scene(3)
    with pytest.raises(ValueError, match="physics=False"):
        vs.add_dynamic([_box(0.1), _box(0.2)], physics=True, mass=mass)


def test_a_one_element_list_is_a_single_morph():
    vs = _scene(2)
    vs.add_ground_plane(friction=1.0)
    s = vs.add_static(morph=[_box(0.2, x=3.0)])
    d = vs.add_dynamic([_box(0.1, x=-3.0)], physics=False, wheel_raycast=True)
    vs.build()
    assert s.variant_index is None and d.variant_index is None
    assert len(s.entity_main.spec.morphs) == 1 and len(d.entity_main.spec.morphs) == 1


# --- H1: the mapping the SDK reports is where the engine put each variant -------

@pytest.mark.parametrize("n_envs,want", [(6, [0, 0, 1, 1, 2, 2]), (3, [0, 1, 2])])
def test_variant_index_is_what_the_engine_placed_per_env(n_envs, want):
    hs = (0.10, 0.20, 0.30)
    vs = _scene(n_envs)
    vs.add_ground_plane(friction=1.0)
    body = vs.add_static(morph=[_box(h, x=5.0) for h in hs], name="steps")
    vs.build()
    assert body.variant_index.tolist() == want
    for env in range(n_envs):
        aabb = body.entity_main.get_AABB(envs_idx=[env]).reshape(2, 3)
        height = float(aabb[1, 2] - aabb[0, 2])
        placed = min(range(3), key=lambda i: abs(hs[i] - height))
        assert abs(hs[placed] - height) < 1e-5, (env, height)
        assert placed == int(body.variant_index[env]), (env, placed, body.variant_index.tolist())
    # the dual_scene mirror holds the same variants, in the same envs
    for env in range(n_envs):
        aabb = body.entity_raycast.get_vAABB(envs_idx=[env]).reshape(2, 3)   # Kinematic: visual AABB
        assert abs(float(aabb[1, 2] - aabb[0, 2]) - hs[int(body.variant_index[env])]) < 1e-5


# --- api-review condition 5: per-element copy, fixed on every variant ------------

def test_dynamic_variants_are_copied_per_element_and_fixed_in_both_scenes():
    import genesis_vehicle.vehicle_scene as vsm
    vs = _scene(3)
    vs.add_ground_plane(friction=1.0)
    morphs = [gs.morphs.Box(size=(2.0, 2.0, h), pos=(4.0, 0.0, h / 2)) for h in (0.1, 0.2, 0.3)]
    assert all(not m.fixed for m in morphs)
    before = vsm._clone_morph_failed_warned
    with _Capture() as cap:
        d = vs.add_dynamic(morphs, physics=False, wheel_raycast=True, name="plat")
        vs.build()
    assert vsm._clone_morph_failed_warned == before
    assert not [m for m in cap.msgs if "could not copy morph" in m or "refit-cost" in m], cap.msgs
    main = d.entity_main.spec.morphs
    mirror = d.entity_raycast.spec.morphs
    assert len(main) == len(mirror) == 3
    assert all(m.fixed for m in main) and all(m.fixed and m.collision for m in mirror)
    assert all(not m.fixed for m in morphs)                 # the caller's morphs untouched
    assert d.variant_index.tolist() == [0, 1, 2]
