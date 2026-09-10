"""The opt-in swept-envelope wheel contact (v1.6.0).

A single zero-radius downward ray reads ground height as a STEP FUNCTION, so a
wheel crossing a vertical edge takes the whole obstacle height of suspension
compression in one ``dt``: an unphysical rate, and a damper force that dominates
the spring. The swept-envelope mode casts M rays per wheel along body +X,
penalises sample ``j`` by the height ``c_j`` its tire surface stands off the
wheel's lowest point, and takes the MINIMUM — so a wheel of radius ``r`` starts
climbing a step of height ``h`` about ``sqrt(2rh)`` before it, as a real tire
does.

What is asserted here, in order:

1. the fan geometry (odd M, ``c_0 = 0``, per-wheel radii);
2. ``read_distances``' collapse of the M axis, including the miss / unpopulated
   cases and the ``d_eff <= d_center`` guarantee that makes a sinking regression
   structurally impossible;
3. the single-scene ceiling measured against EVERY fan ray — the function, and
   then that BOTH registration entry points really hand it the fan;
4. the mode selection (and its refusals) at both registration entry points;
5. a pure-python step-crossing harness with the SDK's own suspension force —
   the reduction in peak wheel N, as a RATIO. Absolute newtons from any harness
   are reference figures only.
"""

import math
import os
import textwrap

import pytest
import torch

from genesis_vehicle.dynamics import suspension_normal_force
from genesis_vehicle.multi_vehicle import check_fan_uniformity
from genesis_vehicle.raycast import (
    FAN_SPAN, RAY_UP_OFFSET, WheelRayPattern, fan_height_offsets,
    fan_longitudinal_offsets, fan_ray_positions, read_distances, sensor_fan,
    sensor_up_offset, set_sensor_fan, set_sensor_miss_value,
    set_sensor_up_offset, single_scene_up_offset,
)
from genesis_vehicle.urdf import self_collision_ceiling

CAR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                   "samples", "urdf", "car_4w.urdf"))

R = 0.358                      # the reference car's wheel radius


class _StubSensor:
    """Distances supplied by the test; ``max_range`` / ``no_hit_value`` on
    ``_options`` exactly as the Genesis raycaster carries them."""

    def __init__(self, distances, miss=20.0):
        self._d = torch.as_tensor(distances, dtype=torch.float32)
        self._options = type("O", (), dict(max_range=miss, no_hit_value=miss))()

    def read(self):
        return type("R", (), dict(distances=self._d))()


def _fan_sensor(distances, c, up_offset=1.0, miss=20.0):
    s = _StubSensor(distances, miss=miss)
    set_sensor_up_offset(s, up_offset)
    set_sensor_miss_value(s)
    set_sensor_fan(s, c)
    return s


# --- 1. fan geometry ---------------------------------------------------------

def test_offsets_span_the_wheel_radius_and_centre_on_zero():
    s = fan_longitudinal_offsets([R], 5)[0]
    assert s == pytest.approx([-R, -R / 2, 0.0, R / 2, R])
    assert s[2] == 0.0                      # exactly, not approximately


def test_c_is_the_swept_circle_height_and_c0_is_exactly_zero():
    c = fan_height_offsets([R], 9)[0]
    s = fan_longitudinal_offsets([R], 9)[0]
    assert c[4] == 0.0                      # the centre sample is unpenalised
    for cj, sj in zip(c, s):
        assert cj == pytest.approx(R - math.sqrt(max(R * R - sj * sj, 0.0)))
    assert c[0] == pytest.approx(R)         # at |s| = r the surface is at hub height
    assert c == sorted(c[:5], reverse=True) + sorted(c[5:])   # symmetric V


def test_each_wheel_uses_its_own_radius():
    """A mixed-radius vehicle cannot share one scalar — c_j approximates THAT
    wheel's swept circle."""
    c = fan_height_offsets([0.2, 0.5], 3)
    assert c[0][0] == pytest.approx(0.2)
    assert c[1][0] == pytest.approx(0.5)


def test_even_M_is_refused_with_the_reason():
    for M in (2, 4, 8):
        with pytest.raises(ValueError, match="ODD"):
            fan_height_offsets([R], M)
    with pytest.raises(ValueError):
        fan_height_offsets([R], 0)


def test_the_even_M_flat_ground_bias_is_what_the_docstrings_quote():
    """Both ``fan_height_offsets`` and ``_check_fan_samples``'s error message
    quote these numbers as the reason even M is refused: with no ``s = 0``
    sample the smallest penalty in the fan is not zero, so even FLAT ground
    reads a distance too large by ``min_j c_j`` — the ground reads that much too
    low. Computed from the documented ``c_j`` formula, since the library itself
    refuses to build an even fan."""
    r = 0.35

    def flat_ground_bias(M):
        s = [(2.0 * j / (M - 1) - 1.0) * r for j in range(M)]
        return min(r - math.sqrt(max(r * r - sj * sj, 0.0)) for sj in s)

    assert flat_ground_bias(4) == pytest.approx(0.020017, abs=5e-7)
    assert flat_ground_bias(2) == pytest.approx(0.350000, abs=5e-7)
    for M in (3, 9, 15, 101):                    # odd M is exact, not approx
        assert min(fan_height_offsets([r], M)[0]) == 0.0


def test_M_1_is_the_point_contact_and_needs_no_geometry():
    assert fan_height_offsets([R], 1) == [[0.0]]
    assert fan_ray_positions([(1.0, 0.5, 0.3)], [R], 1) == [(1.0, 0.5, 0.3)]


def test_radii_are_required_and_must_match_the_wheel_count():
    with pytest.raises(ValueError, match="per-wheel radii"):
        WheelRayPattern([(1.0, 0.5, 0.3)], fan_samples=9)
    _ensure_backend()
    with pytest.raises(ValueError, match="radii"):
        WheelRayPattern([(1.0, 0.5, 0.3), (-1.0, 0.5, 0.3)],
                        fan_samples=9, wheel_radii=[R])
    for bad in ([0.0], [-1.0], [float("nan")]):
        with pytest.raises(ValueError):
            WheelRayPattern([(1.0, 0.5, 0.3)], fan_samples=9, wheel_radii=bad)


# --- 1b. the pattern -------------------------------------------------------

def _ensure_backend():
    import genesis as gs
    if not getattr(gs, "_initialized", False):
        from genesis_vehicle import VehicleScene
        VehicleScene.init_backend("cpu")


def test_M_1_keeps_a_RANK_1_return_shape():
    """Not (n_wheels, 1): a rank-2 pattern of width 1 reads back with a trailing
    axis Genesis does not add to a rank-1 one, and the default path must stay
    bit-identical."""
    _ensure_backend()
    pat = WheelRayPattern([(1.0, 0.5, 0.3), (-1.0, -0.5, 0.3)])
    assert pat.return_shape == (2,)
    assert pat.ray_starts.shape == (2, 3)
    assert pat.fan_samples == 1
    assert pat.fan_c is None


def test_fan_pattern_is_rank_2_and_spreads_the_origins_along_body_x():
    _ensure_backend()
    pat = WheelRayPattern([(1.0, 0.5, 0.3)], up_offset=0.28,
                          fan_samples=5, wheel_radii=[R])
    assert pat.return_shape == (1, 5)
    starts = pat.ray_starts.reshape(5, 3)
    assert [float(v) for v in starts[:, 0]] == pytest.approx(
        [1.0 - R, 1.0 - R / 2, 1.0, 1.0 + R / 2, 1.0 + R])
    assert [float(v) for v in starts[:, 1]] == pytest.approx([0.5] * 5)
    assert [float(v) for v in starts[:, 2]] == pytest.approx([0.58] * 5)
    assert pat.ray_dirs.reshape(-1, 3)[0, 2] == pytest.approx(-1.0)


def test_pattern_ray_positions_is_the_full_fan_in_the_attachment_plane():
    _ensure_backend()
    pat = WheelRayPattern([(1.0, 0.5, 0.3)], up_offset=0.28,
                          fan_samples=3, wheel_radii=[R])
    assert pat.ray_positions() == pytest.approx(
        [(1.0 - R, 0.5, 0.3), (1.0, 0.5, 0.3), (1.0 + R, 0.5, 0.3)])


def test_from_config_carries_the_radii():
    _ensure_backend()
    from genesis_vehicle.config import resolve
    from genesis_vehicle import car_4w_rwd_ackermann
    resolved = resolve(car_4w_rwd_ackermann(CAR))
    pat = WheelRayPattern.from_config(resolved, fan_samples=3)
    assert pat.fan_samples == 3
    assert pat.wheel_radii == [w.radius for w in resolved.wheels]
    assert WheelRayPattern.from_config(resolved).fan_samples == 1


def test_from_config_takes_its_POSITIONS_from_the_config_not_the_urdf():
    """Why neither registration entry point uses it: ``from_config`` reads
    positions off ``resolved.wheels``, whose ORDER is the user's wheel list and
    whose ``position`` a ``WheelConfig`` may override, while ``add_vehicle`` /
    ``make_wheel_raycaster`` must place rays in ``parse_urdf`` order (that is
    the order every downstream index — ``Vehicle.wheel_positions``, the
    per-wheel radii, the pipeline's L1 axis — is built in). Swapping them in
    would silently move the rays of any hand-built config, so ``from_config``
    stays the CALLER-facing path for someone who already holds a
    ``ResolvedConfig``."""
    _ensure_backend()
    from dataclasses import replace
    from genesis_vehicle.config import resolve
    from genesis_vehicle import car_4w_rwd_ackermann
    resolved = resolve(car_4w_rwd_ackermann(CAR))
    moved = replace(resolved, wheels=[replace(resolved.wheels[0],
                                              position=(9.0, 9.0, 9.0))]
                    + list(resolved.wheels[1:]))
    assert WheelRayPattern.from_config(moved).ray_positions()[0] == (9.0, 9.0, 9.0)
    # ... whereas the URDF (and therefore both entry points) says otherwise.
    from genesis_vehicle.urdf import parse_urdf
    assert parse_urdf(CAR).wheels[0].position != (9.0, 9.0, 9.0)


# --- 2. read_distances collapses the M axis ---------------------------------

def test_the_minimum_over_the_fan_wins_and_M_disappears():
    # wheel 0: the +0.20-penalised sample sees ground 0.30 lower -> it wins.
    s = _fan_sensor([[[1.30, 1.30, 1.00]]], [[0.0, 0.05, 0.20]])
    d, hit = read_distances(s, 1, return_hit=True)
    assert d.shape == (1, 1) and hit.shape == (1, 1)
    assert float(d[0, 0]) == pytest.approx(0.20)     # min(0.30, 0.35, 0.20)
    assert bool(hit[0, 0])


def test_the_envelope_can_only_RAISE_the_ground():
    """``c_0 = 0`` makes ``d_eff <= d_center`` structural — the fan can never
    report the ground LOWER than the single ray would, so it cannot make a
    vehicle sink. Checked over a randomised set of ground profiles."""
    g = torch.Generator().manual_seed(7)
    c = fan_height_offsets([R], 9)
    for _ in range(200):
        raw = 1.0 + torch.rand(1, 1, 9, generator=g)          # (N, W, M)
        fan = _fan_sensor(raw, c)
        centre = _StubSensor(raw[:, :, 4])
        set_sensor_up_offset(centre, 1.0)
        assert float(read_distances(fan, 1)[0, 0]) <= float(
            read_distances(centre, 1)[0, 0]) + 1e-6


def test_flat_ground_reads_exactly_the_same_as_the_point_contact():
    """Odd M has a centre sample with c=0, and every off-centre sample is
    penalised by exactly the height it stands off the ground — so on level
    ground the fan reproduces the single ray to the bit."""
    for M in (3, 9, 15, 101):
        c = fan_height_offsets([R], M)
        raw = [[[1.30 + cj for cj in c[0]]]]      # ground is flat under all rays
        fan = _fan_sensor(raw, c)
        assert float(read_distances(fan, 1)[0, 0]) == pytest.approx(0.30, abs=1e-6)


def test_a_wheel_whose_whole_fan_missed_reports_the_sentinel():
    s = _fan_sensor([[[20.0, 20.0, 20.0], [1.30, 20.0, 20.0]]],
                    [[0.0, 0.05, 0.20], [0.0, 0.05, 0.20]])
    d, hit = read_distances(s, 1, return_hit=True)
    assert float(d[0, 0]) == pytest.approx(20.0)   # untouched sentinel
    assert not bool(hit[0, 0])
    assert float(d[0, 1]) == pytest.approx(0.30)   # one sample hit -> it wins
    assert bool(hit[0, 1])


def test_a_missing_sample_never_wins_the_minimum():
    """A miss is +inf inside the fan, not a 20 m 'hit' — otherwise the sentinel
    would be compared against real distances."""
    s = _fan_sensor([[[20.0, 1.40, 20.0]]], [[0.0, 0.05, 0.20]])
    assert float(read_distances(s, 1)[0, 0]) == pytest.approx(0.45)


def test_an_unpopulated_buffer_reads_as_a_MISS_not_as_ground():
    """Genesis allocates the distance buffer as zeros and fills it inside
    ``scene.step()``. At M=1 those zeros keep their historical ``-up_offset``
    shift; at M>1 every sample fails the raw hit test, so the wheel reports the
    sentinel. Both are discarded by the first-step protection — this asserts
    which one the fan produces, not that either is safe on its own."""
    s = _fan_sensor([[[0.0, 0.0, 0.0]]], [[0.0, 0.05, 0.20]])
    d, hit = read_distances(s, 1, return_hit=True)
    assert float(d[0, 0]) == pytest.approx(20.0)
    assert not bool(hit[0, 0])


def test_a_hit_is_clamped_under_the_sentinel():
    """A far-but-real hit must not land ON the miss value; the returned mask is
    the authority, and this keeps the value agreeing with it."""
    s = _fan_sensor([[[19.999, 19.999, 19.999]]], [[0.0, 0.05, 0.20]], miss=20.0)
    d, hit = read_distances(s, 1, return_hit=True)
    assert bool(hit[0, 0])
    assert float(d[0, 0]) < 20.0


def test_the_squeeze_is_against_the_pattern_rank_not_n_envs():
    """Genesis drops the env axis at n_envs == 0, so a fan reads back
    (n_wheels, M) there and (n_envs, n_wheels, M) otherwise."""
    c = [[0.0, 0.05, 0.20], [0.0, 0.05, 0.20]]
    row = [[1.30, 1.40, 1.50], [1.30, 1.40, 1.50]]
    assert read_distances(_fan_sensor(row, c), 1).shape == (1, 2)          # (W, M)
    assert read_distances(_fan_sensor([row], c), 1).shape == (1, 2)
    assert read_distances(_fan_sensor([row] * 3, c), 3).shape == (3, 2)


def test_a_shape_that_disagrees_with_the_stamp_is_rejected():
    s = _fan_sensor([[[1.30, 1.40, 1.50]]], [[0.0, 0.05, 0.20]])
    s._d = torch.tensor([[1.30, 1.40]])            # no longer a fan read
    with pytest.raises(ValueError, match="fan_samples"):
        read_distances(s, 1)


def test_an_unstamped_sensor_falls_back_to_its_own_pattern():
    """The 'build your own Raycaster and hand it to VehiclePhysics' path never
    stamps; reading its M axis as a wheel axis would be silent nonsense."""
    _ensure_backend()
    pat = WheelRayPattern([(1.0, 0.5, 0.3)], fan_samples=3, wheel_radii=[R])
    s = _StubSensor([[[1.30, 1.40, 1.50]]])
    s._options = type("O", (), dict(max_range=20.0, no_hit_value=20.0,
                                    pattern=pat))()
    assert sensor_fan(s)[0] == 3
    assert read_distances(s, 1).shape == (1, 1)


def test_point_sensors_still_report_no_fan():
    s = _StubSensor([[1.30, 1.30]])
    assert sensor_fan(s) == (1, None)
    set_sensor_fan(s, None)                        # idempotent, no attributes
    assert sensor_fan(s) == (1, None)


# --- 3. the single-scene ceiling sees every ray ------------------------------

def _urdf_with_low_outer_geom(path):
    """A body whose centre is high (a 0.28 m ceiling over a ray at x=0) but
    whose OUTER geom hangs low (0.12 m over a ray at x=+0.30)."""
    path.write_text(textwrap.dedent("""\
        <?xml version="1.0"?>
        <robot name="t">
          <link name="base_link">
            <inertial><origin xyz="0 0 0.5"/><mass value="100.0"/>
              <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
            <collision>
              <origin xyz="0 0 0.85" rpy="0 0 0"/>
              <geometry><box size="0.4 1.0 0.5"/></geometry>
            </collision>
            <collision>
              <origin xyz="0.4 0 0.54" rpy="0 0 0"/>
              <geometry><box size="0.4 1.0 0.2"/></geometry>
            </collision>
          </link>
        </robot>
        """))
    return str(path)


def test_the_ceiling_is_lowered_by_an_outer_fan_ray(tmp_path):
    """The v1.1.16 launch bug, per sample: the wheel centre clears the overhang
    and an outer fan ray does not. THIS is why the fan origins, not the wheel
    centres, are handed to self_collision_ceiling."""
    urdf = _urdf_with_low_outer_geom(tmp_path / "overhang.urdf")
    wheels = [(0.0, 0.0, 0.30)]
    assert self_collision_ceiling(urdf, wheels) == pytest.approx(0.28)
    fan = fan_ray_positions(wheels, [0.30], 3)              # x = -0.30, 0, +0.30
    assert self_collision_ceiling(urdf, fan) == pytest.approx(0.12)
    assert single_scene_up_offset(urdf, fan) == pytest.approx(0.12)
    # ... and every fan origin then stays under the geometry it sits below.
    assert 0.30 + 0.12 < 0.44                                # outer box bottom


def test_the_reference_car_ceiling_is_unchanged_by_the_fan():
    """True for THIS urdf only — its chassis AABB spans x in [-2.15, 2.15] and
    so covers every fan origin uniformly. Not a property of the function."""
    from genesis_vehicle.urdf import parse_urdf
    wheels = [w.position for w in parse_urdf(CAR).wheels]
    fan = fan_ray_positions(wheels, [R] * len(wheels), 9)
    assert self_collision_ceiling(CAR, fan) == pytest.approx(0.28)
    assert self_collision_ceiling(CAR, wheels) == pytest.approx(0.28)


# --- 3b. ... and BOTH registration entry points actually pass those rays -----
#
# The section above tests the FUNCTION. On the reference car the fan origins and
# the wheel centres give the same ceiling (its chassis AABB covers both), so a
# call site that passed the wheel centres would pass every test above. These
# tests register a vehicle whose ceiling DIFFERS between the two, so they fail if
# either entry point hands single_scene_up_offset the wheel centres.

_CAR_CHASSIS_COLLISION = """    <collision>
      <origin xyz="0 0 0.85" rpy="0 0 0"/>
      <geometry><box size="4.3 1.7 0.5"/></geometry>
    </collision>
"""

#: A low sill OUTBOARD of the front wheels: x in [1.60, 1.80], bottom z=0.44.
#: The front wheel centres are at x=1.35 (clear of it); the M=9 fan reaches
#: x = 1.35 + 0.358 = 1.708 (under it).
_CAR_OUTER_SILL = """    <collision>
      <origin xyz="1.70 0 0.49" rpy="0 0 0"/>
      <geometry><box size="0.20 1.7 0.10"/></geometry>
    </collision>
"""

_SILL_BOTTOM_Z = 0.44
_CENTRE_ONLY_CEILING = 0.28      # the chassis box at z=0.60, less the margin
_FAN_CEILING = 0.12              # the sill at z=0.44, less the margin


def _overhang_car(tmp_path):
    """The reference car with a low sill hanging outboard of the front wheels —
    a URDF on which the centre-only ceiling (0.28 m) and the swept-envelope fan
    ceiling (0.12 m) DISAGREE, which car_4w.urdf cannot do."""
    text = open(CAR).read()
    assert _CAR_CHASSIS_COLLISION in text          # guard against a URDF edit
    path = tmp_path / "overhang_car.urdf"
    path.write_text(text.replace(_CAR_CHASSIS_COLLISION,
                                 _CAR_CHASSIS_COLLISION + _CAR_OUTER_SILL))
    return str(path)


def test_the_overhang_car_really_distinguishes_the_two(tmp_path):
    """The discriminator itself: if this ever stops holding, the two wiring
    tests below stop being able to fail and must be re-authored."""
    from genesis_vehicle.urdf import parse_urdf
    urdf = _overhang_car(tmp_path)
    wheels = [w.position for w in parse_urdf(urdf).wheels]
    fan = fan_ray_positions(wheels, [R] * len(wheels), 9)
    assert self_collision_ceiling(urdf, wheels) == pytest.approx(_CENTRE_ONLY_CEILING)
    assert self_collision_ceiling(urdf, fan) == pytest.approx(_FAN_CEILING)


def test_add_vehicle_caps_the_offset_against_every_fan_ray(tmp_path):
    """``VehicleScene.add_vehicle(raycast_mode='single_scene')`` must measure the
    ceiling at the FAN origins. Passing the wheel centres would stamp 0.28 m and
    lift the outer rays to z=0.58 — inside the sill at z=0.44, i.e. the v1.1.16
    "hits its own body, reads maximum compression, launches" bug, one sample at
    a time."""
    _ensure_backend()
    from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann
    urdf = _overhang_car(tmp_path)

    vs = VehicleScene(n_envs=1, raycast_mode="single_scene", init_genesis=False)
    veh = vs.add_vehicle(urdf, car_4w_rwd_ackermann, pos=(0.0, 0.0, 1.0),
                         wheel_contact="swept_envelope", contact_samples=9)
    assert sensor_up_offset(veh.sensor) == pytest.approx(_FAN_CEILING)
    pattern = veh.sensor._options.pattern
    assert pattern.up_offset == pytest.approx(_FAN_CEILING)
    assert pattern.fan_samples == 9
    # ... and every ray the pattern actually casts starts under the sill.
    starts = pattern.ray_starts.reshape(-1, 3)
    assert float(starts[:, 2].max()) < _SILL_BOTTOM_Z

    # The point contact on the SAME urdf keeps the centre-only ceiling: that is
    # the value this test would read if the fan branch passed wheel centres.
    vs2 = VehicleScene(n_envs=1, raycast_mode="single_scene", init_genesis=False)
    point = vs2.add_vehicle(urdf, car_4w_rwd_ackermann, pos=(0.0, 0.0, 1.0))
    assert sensor_up_offset(point.sensor) == pytest.approx(_CENTRE_ONLY_CEILING)
    assert sensor_up_offset(veh.sensor) < sensor_up_offset(point.sensor)


def test_make_wheel_raycaster_caps_the_offset_against_every_fan_ray(tmp_path):
    """``scene_helpers.make_wheel_raycaster`` has its OWN wiring (its radii come
    from the URDF, not from a resolved config) and needs its own proof."""
    gs = pytest.importorskip("genesis")
    _ensure_backend()
    from genesis_vehicle.scene_helpers import make_wheel_raycaster
    urdf = _overhang_car(tmp_path)

    scene = gs.Scene(show_viewer=False)
    entity = scene.add_entity(gs.morphs.URDF(file=urdf, pos=(0.0, 0.0, 1.0)))
    sensor = make_wheel_raycaster(scene, entity, urdf,
                                  wheel_contact="swept_envelope",
                                  contact_samples=9)
    assert sensor_up_offset(sensor) == pytest.approx(_FAN_CEILING)
    pattern = sensor._options.pattern
    assert pattern.up_offset == pytest.approx(_FAN_CEILING)
    assert float(pattern.ray_starts.reshape(-1, 3)[:, 2].max()) < _SILL_BOTTOM_Z

    point = make_wheel_raycaster(scene, entity, urdf, wheel_contact="point")
    assert sensor_up_offset(point) == pytest.approx(_CENTRE_ONLY_CEILING)
    # An explicit up_offset still wins over the cap (documented override).
    forced = make_wheel_raycaster(scene, entity, urdf, up_offset=0.05,
                                  wheel_contact="swept_envelope",
                                  contact_samples=9)
    assert sensor_up_offset(forced) == pytest.approx(0.05)


# --- 4. mode selection ------------------------------------------------------

def test_wheel_contact_selection():
    from genesis_vehicle.vehicle_scene import _fan_samples
    assert _fan_samples("point", 9) == 1            # contact_samples ignored
    assert _fan_samples("point", 4) == 1
    assert _fan_samples("swept_envelope", 9) == 9
    with pytest.raises(ValueError, match="wheel_contact"):
        _fan_samples("envelope", 9)
    with pytest.raises(ValueError, match="ODD"):
        _fan_samples("swept_envelope", 8)
    with pytest.raises(ValueError, match="point"):
        _fan_samples("swept_envelope", 1)


def test_WHEEL_CONTACT_MODES_is_public_and_the_uniformity_check_is_not():
    """``wheel_contact``'s legal values are user-facing (a caller validating
    input needs them), so they are exported. ``check_fan_uniformity`` is an
    invariant check ``MultiVehicleKindPhysics.__init__`` makes on your behalf
    and stays internal."""
    import genesis_vehicle
    from genesis_vehicle.vehicle_scene import WHEEL_CONTACT_MODES as internal
    assert "WHEEL_CONTACT_MODES" in genesis_vehicle.__all__
    assert genesis_vehicle.WHEEL_CONTACT_MODES is internal
    assert genesis_vehicle.WHEEL_CONTACT_MODES == ("point", "swept_envelope")
    assert "check_fan_uniformity" not in genesis_vehicle.__all__


def test_both_entry_points_agree_on_what_point_means():
    from genesis_vehicle.scene_helpers import _fan_samples as helper
    from genesis_vehicle.vehicle_scene import _fan_samples as scene
    for args in (("point", 9), ("swept_envelope", 3), ("swept_envelope", 15)):
        assert helper(*args) == scene(*args)


def test_one_kind_cannot_mix_fan_sizes():
    c3 = [[0.0, 0.05, 0.20]]
    a = _fan_sensor([[[1.3, 1.3, 1.3]]], c3)
    b = _fan_sensor([[[1.3, 1.3, 1.3]]], c3)
    point = _StubSensor([[1.3]])
    assert check_fan_uniformity([a, b]) == 3
    assert check_fan_uniformity([point, point]) == 1
    with pytest.raises(ValueError, match="DIFFERENT wheel-contact fan sizes"):
        check_fan_uniformity([a, point])


# --- 5. the defect this mode addresses --------------------------------------
#
# A wheel is carried across a vertical step at constant height and speed; the
# distances go through the REAL read_distances and the normal force through the
# SDK's own suspension_normal_force, with the reference car's constants. Only
# the contact model changes between runs.

_REST_D = R + 0.10           # radius + rest_stroke (car preset)
_Z_ATTACH = 0.412            # the car's measured static ray distance
_K, _C_COMP, _C_EXT, _RATE_CLAMP = 70_000.0, 14_000.0, 4_000.0, 30.0
_DT = 0.025


def _cross_step(M, h=0.130, v=3.3, span=FAN_SPAN, n=200):
    """Peak (N, one-step compression jump, raw compression rate) crossing a step
    of height ``h`` at ``v`` m/s."""
    s = fan_longitudinal_offsets([R], M, span)[0] if M > 1 else [0.0]
    c = fan_height_offsets([R], M, span) if M > 1 else None
    sensor = _StubSensor([[0.0]])
    set_sensor_up_offset(sensor, RAY_UP_OFFSET)
    if M > 1:
        set_sensor_fan(sensor, c)
    prev = torch.zeros(1, 1)
    peak = [0.0, 0.0, 0.0]
    for k in range(n):
        x = -1.0 + v * _DT * k
        row = [_Z_ATTACH + RAY_UP_OFFSET - (h if x + sj >= 0.0 else 0.0)
               for sj in s]
        sensor._d = torch.tensor([[row]] if M > 1 else [row], dtype=torch.float32)
        d = read_distances(sensor, 1)
        comp = torch.clamp(_REST_D - d, min=0.0)
        raw_rate = (comp - prev) / _DT
        rate = torch.clamp(raw_rate, -_RATE_CLAMP, _RATE_CLAMP)
        N = suspension_normal_force(comp, rate, _K, _C_COMP, _C_EXT, comp <= 0)
        if x > -0.6:
            peak = [max(peak[0], float(N.max())),
                    max(peak[1], float((comp - prev).max())),
                    max(peak[2], float(raw_rate.max()))]
        prev = comp
    return peak


def test_the_point_contact_takes_the_whole_step_in_one_dt():
    """The defect, stated as a measurement: at 3.3 m/s a 0.130 m lip delivers
    0.130 m of compression in one 0.025 s step — a 5.2 m/s rate the damper term
    then dominates."""
    peak_N, dc, rate = _cross_step(1)
    assert dc == pytest.approx(0.130, abs=1e-6)
    assert rate == pytest.approx(5.2, abs=1e-3)
    assert peak_N == pytest.approx(85_120.0, rel=1e-3)


def test_the_swept_envelope_spreads_the_lip_over_several_steps():
    peak_N_point, dc_point, _ = _cross_step(1)
    peak_N_fan, dc_fan, rate_fan = _cross_step(9)
    assert dc_fan < dc_point / 1.5           # 0.0732 vs 0.1300
    assert rate_fan < 3.0                    # 2.930 vs 5.200
    ratio = peak_N_point / peak_N_fan
    # Measured 1.703 here (85,120 / 49,978). The bar is deliberately below it,
    # and deliberately a RATIO: the absolute newtons belong to this harness,
    # which has no chassis-terrain contact and no load transfer. Note the
    # planned >= 2.0 acceptance is NOT met at M=9 in this scenario — it is
    # reached at M=31 (2.04); see test_M_is_not_an_accuracy_dial.
    assert ratio >= 1.5, ratio


def test_M_is_not_an_accuracy_dial():
    """M is a DISCRETISATION count, not an accuracy knob: the peak force does
    not converge monotonically, it oscillates with M as the sample the minimum
    picks changes. Do not compare two runs at different M."""
    point = _cross_step(1)[0]
    ratios = {M: point / _cross_step(M)[0] for M in (9, 15, 31, 101)}
    assert ratios[9] == pytest.approx(1.703, rel=0.02)
    assert ratios[15] == pytest.approx(1.906, rel=0.02)
    assert ratios[31] == pytest.approx(2.039, rel=0.02)
    assert ratios[101] == pytest.approx(1.959, rel=0.02)
    # non-monotonic — more samples is not uniformly "better"
    assert ratios[101] < ratios[31]


def test_the_absolute_peaks_quoted_in_the_WheelRayPattern_docstring():
    """``WheelRayPattern``'s docstring quotes these newtons and names THIS
    harness for them. They are harness figures — no chassis-terrain contact, no
    load transfer — and are reference values only, but they must be what the
    committed harness produces, or the docstring is unreproducible again."""
    assert _cross_step(1)[0] == pytest.approx(85_120.0, rel=1e-3)
    for M, peak_N in ((9, 49_978.0), (15, 44_670.0),
                      (31, 41_745.0), (101, 43_455.0)):
        assert _cross_step(M)[0] == pytest.approx(peak_N, rel=1e-3), M
    # ... and the "about 20% across M in [9, 31]" spread the docstrings claim.
    peaks = [_cross_step(M)[0] for M in (9, 15, 31)]
    assert max(peaks) / min(peaks) == pytest.approx(1.20, abs=0.02)


def test_M_3_at_the_full_span_is_the_point_contact_again():
    """Its only off-centre samples sit at |s| = r, where c = r: for any obstacle
    shorter than the wheel radius they can never win the minimum. M=3 is legal
    (odd) but useless at the default span — the reason the docs recommend 9."""
    assert _cross_step(3)[0] == pytest.approx(_cross_step(1)[0])
    # A narrower span puts those two samples where they can see the lip.
    assert _cross_step(3, span=0.6)[0] < _cross_step(1)[0]


def test_the_envelope_never_raises_the_peak_force_above_the_point_contact():
    """The direction of the effect is the guarantee; its size is not."""
    point = _cross_step(1)[0]
    for M in (9, 15, 31, 101):
        for span in (1.0, 0.8, 0.6):
            assert _cross_step(M, span=span)[0] <= point + 1e-6


# --- 6. one real Genesis run ------------------------------------------------

def test_single_scene_fan_does_not_hit_its_own_body():
    """Only an integration run proves the ceiling cap actually holds: in
    ``single_scene`` the rays are cast in the scene the chassis collides in, and
    a fan origin above the car's own collision box reads as maximum compression
    and launches the vehicle (the v1.1.16 class of bug). Flat ground, so the fan
    must ALSO read exactly what the point contact reads."""
    genesis = pytest.importorskip("genesis")
    from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann

    out = {}
    for contact in ("point", "swept_envelope"):
        cfg = car_4w_rwd_ackermann(CAR, stability="control")
        VehicleScene.init_backend("cpu")
        vs = VehicleScene(raycast_mode="single_scene", dt=cfg.recommended_dt,
                          substeps=10, n_envs=1, show_viewer=False)
        vs.add_ground_plane(friction=1.0)
        veh = vs.add_vehicle(CAR, car_4w_rwd_ackermann, cfg=cfg,
                             pos=(0.0, 0.0, 1.0),
                             material=genesis.materials.Rigid(friction=1.0),
                             wheel_contact=contact, contact_samples=9)
        vs.build()
        for _ in range(60):
            veh.set_inputs(throttle=0.0, brake=1.0)
            vs.step()
        raw = veh.sensor.read().distances
        out[contact] = (tuple(raw.shape), veh.distances[0].clone(),
                        float(veh.get_pos()[0][2]),
                        bool(veh.wheels_grounded.all()))

    assert out["point"][0] == (1, 4)                 # (n_envs, n_wheels)
    assert out["swept_envelope"][0] == (1, 4, 9)     # ... M is the last axis
    assert out["point"][3] and out["swept_envelope"][3]
    # Not launched: the fan rests at the same ride height as the point contact.
    assert out["swept_envelope"][2] == pytest.approx(out["point"][2], abs=1e-3)
    assert torch.allclose(out["swept_envelope"][1], out["point"][1], atol=1e-4)
