"""The single-scene wheel rays must not start inside the vehicle's own body.

In ``raycast_mode="single_scene"`` the rays are cast in the SAME scene the
chassis collides in. The v1.1.16 high-cast offset (1.0 m above the wheel
attachment point) put the origin ABOVE the reference car's chassis collision
box, so every ray hit its own roof 0.2 m down, ``read_distances`` reported
-0.8 m, the pipeline read maximum compression and the vehicle launched. v1.5.0
caps the offset at the vehicle's own collision ceiling.
"""

import os
import textwrap

import pytest
import torch

from genesis_vehicle.raycast import (
    RAY_MISS_THRESHOLD, RAY_UP_OFFSET, WheelRayPattern, read_distances,
    sensor_up_offset, set_sensor_up_offset, single_scene_up_offset,
)
from genesis_vehicle.urdf import self_collision_ceiling

CAR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                   "samples", "urdf", "car_4w.urdf"))
TANK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "samples", "urdf", "tank_ray.urdf"))


# --- the ceiling measurement -------------------------------------------------

def test_reference_car_ceiling_is_the_chassis_box_bottom():
    """car_4w: wheel attachment z=0.30, chassis collision box z in [0.60, 1.10].
    0.30 m of clearance, less the 0.02 m margin."""
    from genesis_vehicle.urdf import parse_urdf
    wheels = [w.position for w in parse_urdf(CAR).wheels]
    assert self_collision_ceiling(CAR, wheels) == pytest.approx(0.28)


def test_reference_tank_ceiling():
    from genesis_vehicle.urdf import parse_urdf
    wheels = [w.position for w in parse_urdf(TANK).wheels]
    assert self_collision_ceiling(TANK, wheels) == pytest.approx(0.28)


def test_ceiling_is_none_when_nothing_sits_above_the_rays(tmp_path):
    """A ray outside the collision box's x/y footprint is not obstructed, so the
    full offset stands. (This is why the HJW/JMK demo cars, whose collision box
    is a narrow y=+/-0.25 spine, keep RAY_UP_OFFSET.)"""
    urdf = _urdf(collision_box="0.4 0.4 0.5", collision_z="0.85")
    p = tmp_path / "narrow.urdf"
    p.write_text(urdf)
    # Wheels at y=+/-0.75 are clear of a 0.4-wide box.
    assert self_collision_ceiling(str(p), [(1.35, 0.75, 0.30), (1.35, -0.75, 0.30)]) is None


def test_ceiling_is_zero_when_the_ray_starts_inside_the_body(tmp_path):
    """An origin already inside a collision geom cannot be lifted at all."""
    p = tmp_path / "low.urdf"
    p.write_text(_urdf(collision_box="4.0 2.0 1.0", collision_z="0.30"))
    assert self_collision_ceiling(str(p), [(0.0, 0.0, 0.30)]) == 0.0


def test_ceiling_ignores_geometry_below_the_rays(tmp_path):
    p = tmp_path / "belly.urdf"
    p.write_text(_urdf(collision_box="4.0 2.0 0.2", collision_z="0.05"))
    assert self_collision_ceiling(str(p), [(0.0, 0.0, 0.30)]) is None


# --- the offset the single-scene path picks ----------------------------------

def test_single_scene_offset_is_capped_for_the_reference_car():
    from genesis_vehicle.urdf import parse_urdf
    wheels = [w.position for w in parse_urdf(CAR).wheels]
    offset = single_scene_up_offset(CAR, wheels, "car")
    assert offset == pytest.approx(0.28)
    assert offset < RAY_UP_OFFSET


def test_single_scene_offset_keeps_the_default_without_an_obstruction(tmp_path):
    p = tmp_path / "narrow.urdf"
    p.write_text(_urdf(collision_box="0.4 0.4 0.5", collision_z="0.85"))
    assert single_scene_up_offset(str(p), [(1.35, 0.75, 0.30)]) == RAY_UP_OFFSET


def test_single_scene_offset_survives_an_unreadable_urdf():
    assert single_scene_up_offset("/nonexistent.urdf", [(0.0, 0.0, 0.3)]) == RAY_UP_OFFSET


def test_capped_offset_would_clear_the_reference_car_roof():
    """The whole point: origin at z = attach + offset must stay UNDER the box."""
    from genesis_vehicle.urdf import parse_urdf
    wheels = [w.position for w in parse_urdf(CAR).wheels]
    offset = single_scene_up_offset(CAR, wheels, "car")
    box_bottom_z = 0.60
    for (_, _, z) in wheels:
        assert z + offset < box_bottom_z
    # ... which the pre-v1.5.0 default did not.
    assert wheels[0][2] + RAY_UP_OFFSET > box_bottom_z


# --- the pattern and the read-back stay paired -------------------------------

def test_pattern_lifts_the_origins_by_the_given_offset():
    _ensure_backend()                        # RaycastPattern allocates gs tensors
    pat = WheelRayPattern([(1.0, 0.5, 0.3), (-1.0, -0.5, 0.3)], up_offset=0.28)
    starts = pat.ray_starts.reshape(-1, 3)
    assert starts[0, 2] == pytest.approx(0.58)
    assert starts[1, 2] == pytest.approx(0.58)
    assert pat.ray_dirs.reshape(-1, 3)[0, 2] == pytest.approx(-1.0)


def _ensure_backend():
    import genesis as gs
    if not getattr(gs, "_initialized", False):
        from genesis_vehicle import VehicleScene
        VehicleScene.init_backend("cpu")


class _StubSensor:
    def __init__(self, distances):
        self._d = torch.as_tensor(distances, dtype=torch.float32)

    def read(self):
        return type("R", (), dict(distances=self._d))()


def test_read_distances_subtracts_the_offset_the_sensor_carries():
    s = _StubSensor([[1.30, 1.30]])
    set_sensor_up_offset(s, 0.28)
    assert sensor_up_offset(s) == pytest.approx(0.28)
    assert read_distances(s, 1)[0].tolist() == pytest.approx([1.02, 1.02])


def test_read_distances_falls_back_to_the_default_offset():
    """A sensor built outside VehicleScene was never stamped."""
    s = _StubSensor([[1.30, 1.30]])
    assert sensor_up_offset(s) == RAY_UP_OFFSET
    assert read_distances(s, 1)[0].tolist() == pytest.approx([0.30, 0.30])


def test_read_distances_leaves_misses_at_their_sentinel():
    s = _StubSensor([[20.0, 1.30]])
    set_sensor_up_offset(s, 0.28)
    out = read_distances(s, 1)[0]
    assert out[0] >= RAY_MISS_THRESHOLD          # untouched
    assert out[1] == pytest.approx(1.02)


def test_read_distances_explicit_offset_still_wins():
    s = _StubSensor([[1.30, 1.30]])
    set_sensor_up_offset(s, 0.28)
    assert read_distances(s, 1, up_offset=1.0)[0].tolist() == pytest.approx([0.30, 0.30])


def _urdf(*, collision_box: str, collision_z: str) -> str:
    """A minimal one-link URDF carrying a single collision box."""
    return textwrap.dedent(f"""\
        <?xml version="1.0"?>
        <robot name="t">
          <link name="base_link">
            <inertial><origin xyz="0 0 0.5"/><mass value="100.0"/>
              <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
            <collision>
              <origin xyz="0 0 {collision_z}" rpy="0 0 0"/>
              <geometry><box size="{collision_box}"/></geometry>
            </collision>
          </link>
        </robot>
        """)
