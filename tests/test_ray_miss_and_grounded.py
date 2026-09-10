"""The ray-MISS sentinel and the "did every wheel find ground?" question.

Before v1.5.2 the SDK decided a wheel ray had missed by comparing the distance
with a hardcoded ``RAY_MISS_THRESHOLD = 19.9``, which is pinned to the
raycaster's DEFAULT ``max_range`` of 20.0. Genesis reports a miss as
``no_hit_value``, which itself defaults to ``max_range`` — so at
``raycaster_max_range=10.0`` a miss came back as 10.0, under the threshold, and
``read_distances`` handed the caller 9.0: the high-cast offset subtracted out of
a ray that measured nothing. Three copies of that wrong test existed
(``raycast.read_distances``, ``core._susp_visual_offset``,
``visual._susp_visual_target``), consistent only because all three were wrong
the same way.

Every test here fails on v1.5.1.

| abbr | meaning |
|---|---|
| miss | the distance a raycaster reports for a ray that hit nothing (``no_hit_value``) |
| raw  | a sensor distance BEFORE ``read_distances`` subtracts the high-cast offset |
"""

import os

import pytest

pytest.importorskip("genesis")

import torch
import genesis as gs

from genesis_vehicle import VehicleScene, VehiclePhysics, car_4w_rwd_ackermann
from genesis_vehicle.inputs import VehicleInputs
from genesis_vehicle._gs_compat import sensor_miss_value as engine_miss_value
from genesis_vehicle.core import _susp_visual_offset
from genesis_vehicle.raycast import (
    RAY_MISS_THRESHOLD, RAY_UP_OFFSET, WheelRayPattern, check_miss_supported,
    is_ray_hit, is_ray_hit_corrected, ray_miss_value, read_distances,
    set_sensor_miss_value, set_sensor_up_offset,
)
from genesis_vehicle.visual import _susp_visual_target

URDF = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    "..", "samples", "urdf", "car_4w.urdf"))


@pytest.fixture(scope="module")
def cpu_genesis():
    if not getattr(gs, "_initialized", False):
        try:
            VehicleScene.init_backend("cpu")
        except Exception as e:                     # pragma: no cover
            pytest.skip(f"genesis CPU backend unavailable: {e}")
    return gs


class _StubSensor:
    """A sensor whose distances and options we control. ``options`` may be a
    REAL ``gs.sensors.Raycaster`` options object (that is the point of the
    tests below) or None for a sensor exposing nothing."""

    def __init__(self, distances, options=None):
        self._d = torch.as_tensor(distances, dtype=torch.float32)
        if options is not None:
            self._options = options

    def read(self):
        return type("R", (), dict(distances=self._d))()


def _raycaster_options(**kw):
    """A REAL engine options object — so the default-resolution rule
    (``no_hit_value = max_range`` in ``model_post_init``) is the engine's, not
    a re-implementation of it here."""
    kw.setdefault("pattern", WheelRayPattern([(1.0, 0.5, 0.3), (-1.0, -0.5, 0.3)]))
    return gs.sensors.Raycaster(**kw)


# --- the sentinel read (engine compat) --------------------------------------

def test_engine_miss_value_defaults_to_max_range(cpu_genesis):
    s = _StubSensor([[0.0, 0.0]], _raycaster_options(max_range=10.0))
    assert engine_miss_value(s) == pytest.approx(10.0)
    assert ray_miss_value(s) == pytest.approx(10.0)


def test_engine_miss_value_keeps_an_explicit_zero(cpu_genesis):
    """`is None`, not `or`: 0.0 is a legitimate no_hit_value and must not fall
    through to max_range."""
    s = _StubSensor([[0.0, 0.0]], _raycaster_options(max_range=10.0,
                                                     no_hit_value=0.0))
    assert engine_miss_value(s) == 0.0


def test_engine_miss_value_is_none_for_a_sensor_without_options():
    assert engine_miss_value(_StubSensor([[1.0, 1.0]])) is None
    assert ray_miss_value(_StubSensor([[1.0, 1.0]])) is None


def test_real_vehicle_scene_sensor_reports_its_own_range(cpu_genesis):
    """The defect in one line: at max_range=10.0 the miss sentinel is 10.0, not
    the 19.9 the SDK used to assume. Read BOTH before and after build() — the
    SDK stamps at add_vehicle() time, and the value is resolved at sensor
    construction, so both must agree."""
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene", init_genesis=False)
    vs.add_ground_plane()
    veh = vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(0.0, 0.0, 1.0),
                         raycaster_max_range=10.0)
    assert engine_miss_value(veh.sensor) == pytest.approx(10.0)   # PRE-build
    assert ray_miss_value(veh.sensor) == pytest.approx(10.0)
    vs.build()
    assert engine_miss_value(veh.sensor) == pytest.approx(10.0)   # POST-build
    assert ray_miss_value(veh.sensor) == pytest.approx(10.0)
    assert veh.sensor._options.max_range == pytest.approx(10.0)


# --- the predicate -----------------------------------------------------------

def test_is_ray_hit_rejects_the_unpopulated_zero():
    """`Vehicle.distances` is a ZERO tensor before the first step, so a naive
    `d < threshold` passes vacuously."""
    assert is_ray_hit(0.0, 20.0) is False
    assert is_ray_hit(0.0) is False                  # legacy-fallback branch
    hit = is_ray_hit(torch.tensor([0.0, 0.5, 20.0]), 20.0)
    assert hit.tolist() == [False, True, False]


def test_is_ray_hit_uses_equality_not_a_threshold():
    """A miss at max_range=10.0 is UNDER the old 19.9 threshold."""
    d = torch.tensor([10.0, 9.5])
    assert is_ray_hit(d, 10.0).tolist() == [False, True]
    assert is_ray_hit(d).tolist() == [True, True]    # what 1.5.1 believed


def test_is_ray_hit_equality_is_exact_for_an_unrepresentable_sentinel():
    """float32 cannot represent 19.9 or 0.1 exactly, but genesis writes the
    sentinel VERBATIM, so `==` still holds."""
    for sentinel in (19.9, 0.1):
        d = torch.tensor([sentinel], dtype=torch.float32)
        assert is_ray_hit(d, float(d[0])).tolist() == [False]


# --- read_distances ----------------------------------------------------------

def test_read_distances_leaves_a_non_default_range_miss_alone(cpu_genesis):
    """v1.5.1 returned 9.0 here: the high-cast offset subtracted out of a ray
    that measured NOTHING."""
    s = _StubSensor([[10.0, 1.30]], _raycaster_options(max_range=10.0))
    out = read_distances(s, 1)[0]
    assert out[0] == pytest.approx(10.0)             # sentinel, untouched
    assert out[1] == pytest.approx(0.30)             # hit, offset removed


def test_read_distances_returns_the_hit_mask_on_request(cpu_genesis):
    s = _StubSensor([[10.0, 1.30, 0.0]],
                    _raycaster_options(max_range=10.0))
    out, hit = read_distances(s, 1, return_hit=True)
    assert hit.shape == (1, 3)
    assert hit[0].tolist() == [False, True, False]
    assert out[0][1] == pytest.approx(0.30)
    # The single-value default is unchanged (four in-tree consumers rely on it).
    assert torch.is_tensor(read_distances(s, 1))


def test_read_distances_needs_raw_equality_when_the_sentinel_is_zero(cpu_genesis):
    """With no_hit_value=0.0 a CORRECTED hit at exactly the offset lands on
    0.0 too, so only a comparison on the RAW distance can tell them apart.

    That sentinel is itself rejected on every path now (see the MAJOR-1 test
    below), so the property is exercised with an explicit `miss=` override —
    which is a value SUPPLIED by the caller, not one OBTAINED from a sensor."""
    s = _StubSensor([[0.0, RAY_UP_OFFSET]],
                    _raycaster_options(max_range=10.0, no_hit_value=0.0))
    out, hit = read_distances(s, 1, miss=0.0, return_hit=True)
    assert out[0][0] == pytest.approx(0.0)           # miss: raw sentinel kept
    assert out[0][1] == pytest.approx(0.0)           # hit AT the offset
    assert hit[0].tolist() == [False, True]


def test_read_distances_falls_back_to_the_threshold_without_options():
    """A stub/foreign sensor exposing no options keeps the pre-1.5.2 behaviour."""
    s = _StubSensor([[20.0, 1.30]])
    out = read_distances(s, 1)[0]
    assert out[0] >= RAY_MISS_THRESHOLD
    assert out[1] == pytest.approx(0.30)


# --- the unsupported configuration raises ------------------------------------

def test_a_sentinel_below_max_range_raises(cpu_genesis):
    """Physics reads compression = max(rest_d - distance, 0), so a 0.0 miss
    would be MAXIMUM compression and launch the vehicle. Fail loudly instead."""
    s = _StubSensor([[0.0, 0.0]], _raycaster_options(max_range=10.0,
                                                     no_hit_value=0.0))
    with pytest.raises(ValueError, match="no_hit_value"):
        set_sensor_miss_value(s)
    with pytest.raises(ValueError, match="max_range"):
        set_sensor_miss_value(s, 1.5)


def test_a_sentinel_at_or_above_max_range_is_accepted(cpu_genesis):
    s = _StubSensor([[0.0, 0.0]], _raycaster_options(max_range=10.0))
    assert set_sensor_miss_value(s) == pytest.approx(10.0)
    assert ray_miss_value(s) == pytest.approx(10.0)
    s2 = _StubSensor([[0.0, 0.0]], _raycaster_options(max_range=10.0,
                                                      no_hit_value=99.0))
    assert set_sensor_miss_value(s2) == pytest.approx(99.0)


def test_stamping_a_sensor_without_options_is_a_no_op():
    s = _StubSensor([[1.0, 1.0]])
    assert set_sensor_miss_value(s) is None
    assert set_sensor_miss_value(s, 20.0) == pytest.approx(20.0)   # explicit is fine
    assert ray_miss_value(s) == pytest.approx(20.0)


# --- the three definitions now agree -----------------------------------------

def test_read_layer_core_and_visual_agree_at_a_non_default_range(cpu_genesis):
    """The unification. At max_range=10.0 all three layers used to misjudge a
    miss as a 9.0 m hit; the physics air mask (compression <= 0) did not, so
    the read layer and the visuals disagreed with the simulation."""
    miss = 10.0
    s = _StubSensor([[10.0, 1.30]], _raycaster_options(max_range=miss))
    d, hit = read_distances(s, 1, return_hit=True)
    mesh_r, l_susp = 0.4, 0.1

    off = _susp_visual_offset(d, mesh_r, l_susp, 0.19, miss)
    tgt = _susp_visual_target(d, mesh_r, l_susp, None, None, 0.0, miss)

    air_read = ~hit[0]
    air_core = torch.tensor([bool(x == pytest.approx(-l_susp)) for x in off[0]])
    air_vis = torch.tensor([bool(x == pytest.approx(-l_susp)) for x in tgt[0]])
    assert air_read.tolist() == [True, False]
    assert air_core.tolist() == air_read.tolist()
    assert air_vis.tolist() == air_read.tolist()

    # ... and the legacy threshold path is the bug being fixed: it calls the
    # 10.0 sentinel a hit and drives the wheel mesh to a bogus pose.
    legacy = _susp_visual_offset(d, mesh_r, l_susp, 0.19, None)
    assert float(legacy[0][0]) != pytest.approx(-l_susp)


def test_susp_helpers_keep_their_legacy_positional_signature():
    """core._susp_visual_offset / visual._susp_visual_target gained `miss` as a
    trailing optional; existing positional callers must be unaffected."""
    d = torch.tensor([[0.4, 0.5, 20.0]], dtype=torch.float64)
    off = _susp_visual_offset(d, 0.4, 0.1)
    assert [round(float(x), 6) for x in off[0]] == [0.0, -0.1, -0.1]
    tgt = _susp_visual_target(d, 0.4, 0.1, None, None, 0.0)
    assert [round(float(x), 6) for x in tgt[0]] == [0.0, -0.1, -0.1]


# --- the public grounded API -------------------------------------------------

def test_wheels_grounded_is_false_before_the_first_step(cpu_genesis):
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene", init_genesis=False)
    vs.add_ground_plane()
    veh = vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(0.0, 0.0, 1.0))
    vs.build()
    # distances is still the zero tensor here — the naive `d < threshold`
    # check this API replaces would pass vacuously.
    assert bool(veh.distances.abs().max() == 0.0)
    assert veh.wheels_grounded.shape == (1, 4)
    assert not bool(veh.all_wheels_grounded.any())


def test_wheels_grounded_after_settling_on_the_ground(cpu_genesis):
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene", init_genesis=False)
    vs.add_ground_plane()
    veh = vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(0.0, 0.0, 1.0))
    vs.build()
    for _ in range(60):
        veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        vs.step()
    assert bool(veh.wheels_grounded.all())
    assert bool(veh.all_wheels_grounded.all())


def test_a_vehicle_over_a_hole_is_not_grounded(cpu_genesis):
    """No ground under the rays at all: every ray reports the sentinel."""
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene", init_genesis=False)
    vs.add_ground_plane()
    veh = vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(0.0, 0.0, 1.0))
    vs.build()
    for _ in range(5):
        veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        vs.step()
    # Teleport far above the plane — beyond the raycaster's range.
    veh.entity_main.set_pos(torch.tensor([0.0, 0.0, 100.0]), relative=False)
    veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
    vs.step()
    assert not bool(veh.wheels_grounded.any())
    assert not bool(veh.all_wheels_grounded.any())


def test_partial_reset_clears_only_that_envs_grounded_flag(cpu_genesis):
    """`reset(env_ids=[1])` zeroes env 1's last_distances. Before v1.5.2
    `_stepped_once` was a SCALAR cleared only on a full reset, so env 1 kept a
    stale True and its zeros read as ground."""
    vs = VehicleScene(n_envs=2, raycast_mode="dual_scene", solver="per_vehicle",
                      init_genesis=False)
    vs.add_ground_plane()
    veh = vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(0.0, 0.0, 1.0))
    vs.build()
    for _ in range(60):
        veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        vs.step()
    assert veh.all_wheels_grounded.tolist() == [True, True]

    veh.physics.reset(torch.tensor([1]))
    assert veh.physics._stepped_once.tolist() == [True, False]
    assert veh.all_wheels_grounded.tolist() == [True, False]

    veh.physics.reset()                       # full reset clears both
    assert veh.physics._stepped_once.tolist() == [False, False]
    assert veh.all_wheels_grounded.tolist() == [False, False]


def test_batched_solver_grounded_list_matches_per_vehicle(cpu_genesis):
    """MultiVehiclePhysics.grounded_list mirrors distances_list — same flat
    order, same (n_envs, n_wheels) slices."""
    vs = VehicleScene(n_envs=2, raycast_mode="dual_scene", init_genesis=False)
    vs.add_ground_plane()
    vehs = [vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(6.0 * i, 0.0, 1.0))
            for i in range(2)]
    vs.build()
    assert vs.physics is not None                 # default solver="batched"
    for g in vs.physics.grounded_list():          # before any step: all False
        assert not bool(g.any())
    for _ in range(60):
        for veh in vehs:
            veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        vs.step()
    grounded = vs.physics.grounded_list()
    assert len(grounded) == len(vs.physics.distances_list())
    for g, d in zip(grounded, vs.physics.distances_list()):
        assert g.shape == d.shape
        assert bool(g.all())
    for veh in vehs:                           # the Vehicle handle routes to it
        assert bool(veh.all_wheels_grounded.all())


# --- MAJOR 1: the guarantee holds where the sentinel is OBTAINED -------------
# Validating only inside set_sensor_miss_value guarded the two paths that
# STAMP (scene_helpers.make_wheel_raycaster, VehicleScene.add_vehicle) and left
# the DOCUMENTED "build your own gs.sensors.Raycaster and hand it to
# VehiclePhysics" path completely unchecked.

def test_the_read_path_validates_an_unstamped_sensor(cpu_genesis):
    """v1.5.2-pre defect: an UNSTAMPED no_hit_value=0.0 sensor gave
    `ray_miss_value(s) == 0.0` and `read_distances([[0.0, 1.4]])` -> [0.0, 0.40]
    — the miss stayed 0.0, `_pipeline` read `compression = rest_d - 0.0` (i.e.
    MAXIMUM compression) and launched the vehicle. Exactly what the raise
    exists to prevent."""
    s = _StubSensor([[0.0, 1.40]], _raycaster_options(max_range=20.0,
                                                      no_hit_value=0.0))
    assert engine_miss_value(s) == 0.0            # the engine really reports it
    assert not hasattr(s, "_gv_ray_miss")         # nothing stamped this sensor
    with pytest.raises(ValueError, match="max_range"):
        ray_miss_value(s)                         # the READ accessor refuses
    with pytest.raises(ValueError, match="no_hit_value"):
        read_distances(s, 1)                      # ... and so does every reader


def test_a_stamped_sentinel_is_re_validated_on_read(cpu_genesis):
    """Even a hand-stamped attribute cannot smuggle the bad value past the
    read path."""
    s = _StubSensor([[0.0, 1.40]], _raycaster_options(max_range=20.0))
    setattr(s, "_gv_ray_miss", 0.0)               # bypassing set_sensor_miss_value
    with pytest.raises(ValueError, match="max_range"):
        ray_miss_value(s)


def test_vehicle_physics_refuses_a_hand_built_bad_sensor(cpu_genesis):
    """The documented construction path: VehiclePhysics(scene, entity, sensor,
    cfg) with a raycaster the SDK never stamped. It must fail at construction,
    not on the step that launches the vehicle."""
    vs = VehicleScene(n_envs=1, raycast_mode="single_scene", init_genesis=False)
    vs.add_ground_plane()
    veh = vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(0.0, 0.0, 1.0))
    vs.build()
    bad = _StubSensor([[0.0, 0.0, 0.0, 0.0]],
                      _raycaster_options(max_range=20.0, no_hit_value=0.0))
    with pytest.raises(ValueError, match="no_hit_value"):
        VehiclePhysics(vs._main_scene, veh.entity_main, bad, veh.cfg, n_envs=1)


# --- MAJOR 2: RAW predicate vs CORRECTED predicate --------------------------
# `is_ray_hit`'s `!= 0.0` term is only meaningful on RAW distances, but all
# four in-tree consumers passed distances that had ALREADY had the high-cast
# offset subtracted.

def test_the_raw_predicate_is_wrong_on_corrected_distances():
    """The reproduction, in one test. raw [0.0, 1.5, 1.0] with a 1.0 m offset
    becomes corrected [-1.0, 0.5, 0.0]:

    * the unpopulated 0.0 is now -1.0 and PASSES the `!= 0.0` term;
    * a genuine hit at exactly the offset is now 0.0 and FAILS it.
    """
    miss = 20.0
    raw = torch.tensor([[0.0, 1.5, 1.0]])
    corrected = torch.tensor([[-1.0, 0.5, 0.0]])
    assert is_ray_hit(raw, miss).tolist() == [[False, True, True]]      # correct
    assert is_ray_hit(corrected, miss).tolist() == [[True, True, False]]  # the bug
    # The corrected-distance predicate is the sentinel test alone, which is
    # exact once a miss is guaranteed >= max_range (MAJOR 1).
    assert is_ray_hit_corrected(corrected, miss).tolist() == [[True, True, True]]
    assert is_ray_hit_corrected(torch.tensor([[miss]]), miss).tolist() == [[False]]


def test_read_distances_hit_mask_matches_the_raw_predicate(cpu_genesis):
    """The mask `read_distances` hands back is the RAW one — which is why it is
    carried to consumers rather than re-derived from the corrected values it
    returns alongside."""
    s = _StubSensor([[0.0, 1.5, RAY_UP_OFFSET, 20.0]],
                    _raycaster_options(max_range=20.0))
    d, hit = read_distances(s, 1, return_hit=True)
    assert [round(float(x), 6) for x in d[0]] == [-1.0, 0.5, 0.0, 20.0]
    assert hit[0].tolist() == [False, True, True, False]
    # re-deriving with the RAW predicate on the CORRECTED output disagrees:
    assert is_ray_hit(d, 20.0)[0].tolist() == [True, True, False, False]


def test_grounded_uses_the_carried_mask_not_a_recomputed_one(cpu_genesis):
    """End-to-end on the INJECTED (dual_scene) path: a wheel whose ray hits at
    exactly the high-cast offset reads back as a CORRECTED 0.0 and must count
    as grounded. v1.5.2-pre reported it airborne because `wheels_grounded`
    re-ran the raw predicate on `last_distances`."""
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene", solver="per_vehicle",
                      init_genesis=False)
    vs.add_ground_plane()
    veh = vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(0.0, 0.0, 1.0))
    vs.build()
    for _ in range(30):
        veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        vs.step()
    assert bool(veh.physics._stepped_once.all())
    miss = veh.physics._ray_miss
    # hit at exactly the offset (corrected 0.0) | normal hit | miss | deep
    # over-compression (negative)
    crafted = torch.tensor([[0.0, 0.30, miss, -0.20]], dtype=torch.float32)
    veh.physics.step(VehicleInputs(throttle=0.0, brake=1.0, steer=0.0),
                     distances=crafted)
    assert veh.physics.last_hit[0].tolist() == [True, True, False, True]
    assert veh.wheels_grounded[0].tolist() == [True, True, False, True]


def test_grounded_still_false_on_the_unpopulated_zero_buffer(cpu_genesis):
    """The other half: dropping the `!= 0.0` term is only safe because
    `_stepped_once` gates the zero buffer. Assert that dependency directly, so
    removing the gate fails a test instead of silently reporting ground."""
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene", solver="per_vehicle",
                      init_genesis=False)
    vs.add_ground_plane()
    veh = vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(0.0, 0.0, 1.0))
    vs.build()
    p = veh.physics
    assert not bool(p._stepped_once.any())
    # The zero buffer, run through the corrected predicate the injected path
    # uses, claims every wheel is a hit — the gate is what makes it False.
    assert bool(is_ray_hit_corrected(p.last_distances, p._ray_miss).all())
    assert not bool(veh.wheels_grounded.any())


# --- MINOR 1: the per-env gate on the CLOSED-FORM visual poses --------------
# `_stepped_any()` is batch-wide `.any()`. A partial reset(env_ids=...) leaves
# `.any()` True while the reset env's last_distances are zeros — and a
# CORRECTED 0.0 is a legitimate contact at exactly RAY_UP_OFFSET (MAJOR 2), so
# _susp_visual_offset renders that env FULLY COMPRESSED for one frame.

def _drive(vs, veh, n=40, throttle=0.2):
    for _ in range(n):
        veh.set_inputs(throttle=throttle, brake=0.0, steer=0.0)
        vs.step()


@pytest.mark.parametrize("mode", ["dual_scene", "single_scene"])
def test_partial_reset_keeps_the_rest_pose_not_full_compression(cpu_genesis, mode):
    """v1.5.2-pre defect: after ``reset(env_ids=[0])`` env 0's wheels rendered
    at rest + full compression (+susp_clamp) instead of the rest pose the
    all-unstepped branch returns for the very same ``_stepped_once == False``.

    The invariant asserted here is that the per-env gate and the batch-wide
    shortcut agree: an env with ``_stepped_once == False`` gets the SAME pose
    whether or not its neighbours have stepped."""
    vs = VehicleScene(n_envs=2, raycast_mode=mode, solver="per_vehicle",
                      init_genesis=False)
    vs.add_ground_plane()
    veh = vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(0.0, 0.0, 0.6))
    vs.build()
    p = veh.physics

    rest_z = p.wheel_visual_transforms(frame="local")[0][0, :, 2].clone()  # pre-step
    _drive(vs, veh)
    assert bool(p._stepped_once.all())
    live_z = p.wheel_visual_transforms(frame="local")[0][1, :, 2].clone()

    p.reset(torch.tensor([0], device=p.dev))
    assert p._stepped_once.tolist() == [False, True]
    pos, quat = p.wheel_visual_transforms(frame="local")

    # The reset env is back at rest — NOT rest + susp_clamp (the defect), and
    # exactly what the batch-wide branch would have returned for it.
    assert torch.allclose(pos[0, :, 2], rest_z, atol=1e-6)
    clamp = p._susp_clamp
    clamp = float(clamp.reshape(-1)[0]) if torch.is_tensor(clamp) else float(clamp)
    assert torch.all((pos[0, :, 2] - rest_z).abs() < 0.5 * clamp)
    # ... and the untouched env still carries its real suspension state.
    assert torch.allclose(pos[1, :, 2], live_z, atol=1e-6)
    # Orientation falls back too (steer/spin were zeroed by the reset anyway).
    assert torch.allclose(quat[0], p._rest_wheel_quat_local, atol=1e-6)


def test_partial_reset_rest_pose_on_the_batched_kind_path(cpu_genesis):
    """Same defect, same fix, in ``MultiVehicleKindPhysics.wheel_visual_transforms``
    (its rows are the flat NK batch: env-major over vehicles)."""
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene", solver="batched",
                      init_genesis=False)
    vs.add_ground_plane()
    v0 = vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(0.0, 0.0, 0.6))
    v1 = vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(0.0, 3.0, 0.6))
    vs.build()
    kind = vs._mvp.kinds[0]
    p = kind._proto
    assert p._stepped_once.shape[0] == kind.NK == 2

    rest_z = kind.wheel_visual_transforms(frame="local")[0][0, 0, :, 2].clone()
    for _ in range(40):
        v0.set_inputs(throttle=0.2, brake=0.0, steer=0.0)
        v1.set_inputs(throttle=0.2, brake=0.0, steer=0.0)
        vs.step()
    assert bool(p._stepped_once.all())
    live_z = kind.wheel_visual_transforms(frame="local")[0][0, 1, :, 2].clone()

    kind.reset(torch.tensor([0], device=p.dev))          # flat row 0 = env0/veh0
    assert p._stepped_once.tolist() == [False, True]
    pos = kind.wheel_visual_transforms(frame="local")[0]  # (N, K, n, 3)
    assert torch.allclose(pos[0, 0, :, 2], rest_z, atol=1e-6)
    assert torch.allclose(pos[0, 1, :, 2], live_z, atol=1e-6)


# --- MINOR 2: set_ray_miss_value ENFORCES what its docstring claims ---------

def test_set_ray_miss_value_rejects_a_sentinel_that_is_not_a_distance(cpu_genesis):
    """v1.5.2-pre defect: it cited set_sensor_miss_value's rule and then stored
    ``float(miss)`` unchecked, so ``set_ray_miss_value(0.0)`` re-opened MAJOR
    1's failure mode (a miss read as maximum suspension compression) on the
    dual_scene injection path, where the driver owns no sensor to check."""
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene", solver="per_vehicle",
                      init_genesis=False)
    vs.add_ground_plane()
    veh = vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(0.0, 0.0, 1.0))
    vs.build()
    p = veh.physics
    assert p.sensor is None                     # nothing to validate against
    good = p._ray_miss
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="no_hit_value"):
            p.set_ray_miss_value(bad)
    assert p._ray_miss == good                  # unchanged by the refusals
    p.set_ray_miss_value(None)                  # None = legacy-threshold opt-out
    assert p._ray_miss is None
    p.set_ray_miss_value(good)
    assert p._ray_miss == pytest.approx(good)


def test_set_ray_miss_value_checks_against_the_source_sensor(cpu_genesis):
    """With a sensor to check against (the keyword, as VehicleScene passes on
    the dual_scene path, or the driver's own) the full ``max_range`` rule
    applies — the same ``raycast.check_miss_supported`` the stamp and read
    paths use."""
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene", solver="per_vehicle",
                      init_genesis=False)
    vs.add_ground_plane()
    veh = vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(0.0, 0.0, 1.0),
                         raycaster_max_range=20.0)
    vs.build()
    p = veh.physics
    with pytest.raises(ValueError, match="max_range"):
        p.set_ray_miss_value(5.0, sensor=veh.sensor)      # inside the range
    p.set_ray_miss_value(25.0, sensor=veh.sensor)         # beyond it: fine
    assert p._ray_miss == pytest.approx(25.0)
    # A driver that DOES own a sensor is checked against it with no keyword.
    vs2 = VehicleScene(n_envs=1, raycast_mode="single_scene",
                       solver="per_vehicle", init_genesis=False)
    vs2.add_ground_plane()
    veh2 = vs2.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(0.0, 0.0, 1.0),
                           raycaster_max_range=20.0)
    vs2.build()
    assert veh2.physics.sensor is not None
    with pytest.raises(ValueError, match="max_range"):
        veh2.physics.set_ray_miss_value(5.0)


# --- MINOR 1: the finite-and-positive test lives in check_miss_supported ----
# It used to live ONLY in VehiclePhysics.set_ray_miss_value, so the STAMP and
# READ paths carried a weaker guarantee than the injection path. Two holes,
# both measured:
#   * a stub sensor (no `_options`) has no `max_range`, so `set_sensor_miss_value`
#     had NOTHING to check and accepted 0.0 / -1.0;
#   * on a real sensor, `nan < max_range` evaluates False, so a `nan` sentinel
#     passed the range test — and `is_ray_hit_corrected(d, nan)` is `d != nan`,
#     which is True everywhere: every genuine MISS reported as a HIT.

def test_check_miss_supported_rejects_a_non_finite_sentinel(cpu_genesis):
    """`nan < max_range` is False — the range test alone cannot catch it."""
    s = _StubSensor([[0.0, 1.40]], _raycaster_options(max_range=20.0))
    nan = float("nan")
    assert not (nan < 20.0)                       # why the range test misses it
    for bad in (nan, float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            check_miss_supported(s, bad)
        with pytest.raises(ValueError, match="finite and positive"):
            set_sensor_miss_value(s, bad)
    assert not hasattr(s, "_gv_ray_miss")         # nothing was stamped


def test_check_miss_supported_rejects_zero_and_negative_on_a_stub(cpu_genesis):
    """A sensor with no `_options` exposes no `max_range`, so stage 2 passes it
    through; stage 1 is unconditional and must still refuse."""
    s = _StubSensor([[0.0, 1.40]])                # no options at all
    assert engine_miss_value(s) is None
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            check_miss_supported(s, bad)
        with pytest.raises(ValueError, match="finite and positive"):
            set_sensor_miss_value(s, bad)
    assert not hasattr(s, "_gv_ray_miss")
    # ... and the legitimate "this stub exposes nothing" case still returns
    # None rather than raising (readers fall back to RAY_MISS_THRESHOLD).
    assert set_sensor_miss_value(s) is None


def test_a_stamped_nan_cannot_survive_the_read_path(cpu_genesis):
    """The consequence, if one did get through: `is_ray_hit_corrected` is an
    `!=` against the sentinel, and everything `!= nan`. This asserts BOTH that
    the predicate is defenceless (hence the guard belongs upstream) and that
    every accessor now refuses to hand it a nan."""
    d = torch.tensor([[20.0, 1.40]])              # col 0 IS a miss at max_range
    assert bool(is_ray_hit_corrected(d, float("nan")).all())    # both "hit"
    assert is_ray_hit_corrected(d, 20.0).tolist() == [[False, True]]
    s = _StubSensor([[20.0, 1.40]], _raycaster_options(max_range=20.0))
    setattr(s, "_gv_ray_miss", float("nan"))      # bypassing the stamp helper
    with pytest.raises(ValueError, match="finite and positive"):
        ray_miss_value(s)
    with pytest.raises(ValueError, match="finite and positive"):
        read_distances(s, 1)


def test_all_three_entry_points_share_one_guarantee(cpu_genesis):
    """stamp / read / driver-injection: same rejections, same function."""
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene", solver="per_vehicle",
                      init_genesis=False)
    vs.add_ground_plane()
    veh = vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(0.0, 0.0, 1.0),
                         raycaster_max_range=20.0)
    vs.build()
    p = veh.physics
    good = p._ray_miss
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            check_miss_supported(veh.sensor, bad)          # the shared check
        with pytest.raises(ValueError):
            set_sensor_miss_value(veh.sensor, bad)         # STAMP
        with pytest.raises(ValueError):
            p.set_ray_miss_value(bad)                      # INJECT (no sensor)
        with pytest.raises(ValueError):
            p.set_ray_miss_value(bad, sensor=veh.sensor)   # INJECT (sensor)
    assert p._ray_miss == pytest.approx(good)              # nothing moved
    assert ray_miss_value(veh.sensor) == pytest.approx(20.0)   # READ still fine
