"""Pure-Python unit tests for URDF parsing.

Validates that parse_urdf() correctly extracts wheel positions, sides, and
axle clusters from the reference car and tank URDFs.
"""

from __future__ import annotations

import os

import pytest

from genesis_vehicle.urdf import parse_urdf


_DATA = os.path.join(os.path.dirname(__file__), "data")
CAR_URDF = os.path.join(_DATA, "car_ref.urdf")
TANK_URDF = os.path.join(_DATA, "tank_ref.urdf")


# ---------------------------------------------------------------------------
# Reference car (4-wheel Ackermann)
# ---------------------------------------------------------------------------


def test_car_wheel_count():
    parsed = parse_urdf(CAR_URDF)
    assert len(parsed.wheels) == 4


def test_car_wheel_names():
    parsed = parse_urdf(CAR_URDF)
    names = {w.name for w in parsed.wheels}
    assert names == {
        "front_left_wheel", "front_right_wheel",
        "rear_left_wheel", "rear_right_wheel",
    }


def test_car_positions_match_constants():
    parsed = parse_urdf(CAR_URDF)
    by_name = {w.name: w for w in parsed.wheels}
    # Match the reference car's wheel-position constants.
    assert by_name["front_left_wheel"].position == pytest.approx((1.38, 0.80, 0.34))
    assert by_name["front_right_wheel"].position == pytest.approx((1.38, -0.80, 0.34))
    assert by_name["rear_left_wheel"].position == pytest.approx((-1.35, 0.80, 0.34))
    assert by_name["rear_right_wheel"].position == pytest.approx((-1.35, -0.80, 0.34))


def test_car_sides_detected():
    parsed = parse_urdf(CAR_URDF)
    by_name = {w.name: w for w in parsed.wheels}
    assert by_name["front_left_wheel"].side == "L"
    assert by_name["front_right_wheel"].side == "R"
    assert by_name["rear_left_wheel"].side == "L"
    assert by_name["rear_right_wheel"].side == "R"


def test_car_axles_clustered():
    parsed = parse_urdf(CAR_URDF)
    axle_count = len({w.axle_index for w in parsed.wheels})
    assert axle_count == 2
    # Front axle (x=1.38) should be axle 0; rear (x=-1.35) axle 1.
    by_name = {w.name: w for w in parsed.wheels}
    assert by_name["front_left_wheel"].axle_index == 0
    assert by_name["rear_left_wheel"].axle_index == 1


def test_car_steer_joint_axis_signs():
    parsed = parse_urdf(CAR_URDF)
    # Both steer joints declare axis="0 0 -1".
    assert parsed.steer_axis_signs["front_left_steer_joint"] == -1
    assert parsed.steer_axis_signs["front_right_steer_joint"] == -1


def test_car_steer_and_spin_joint_names():
    parsed = parse_urdf(CAR_URDF)
    by_name = {w.name: w for w in parsed.wheels}
    assert by_name["front_left_wheel"].steer_joint_name == "front_left_steer_joint"
    assert by_name["front_left_wheel"].spin_joint_name == "front_left_wheel_joint"
    # Rear wheels have no steer joint.
    assert by_name["rear_left_wheel"].steer_joint_name is None
    assert by_name["rear_left_wheel"].spin_joint_name == "rear_left_wheel_joint"


def test_car_susp_dynamics_zero():
    parsed = parse_urdf(CAR_URDF)
    # All reference-car suspension joints declare dynamics damping=0/stiffness=0.
    for sname, has_dyn in parsed.susp_has_dynamics.items():
        assert has_dyn is False, f"{sname} has unexpected non-zero dynamics"


# ---------------------------------------------------------------------------
# Reference tank (10-wheel skid-steer)
# ---------------------------------------------------------------------------


def test_tank_wheel_count():
    parsed = parse_urdf(TANK_URDF)
    assert len(parsed.wheels) == 10


def test_tank_sides_balanced():
    parsed = parse_urdf(TANK_URDF)
    sides = [w.side for w in parsed.wheels]
    assert sides.count("L") == 5
    assert sides.count("R") == 5


def test_tank_axles_clustered():
    parsed = parse_urdf(TANK_URDF)
    axle_count = len({w.axle_index for w in parsed.wheels})
    # 5 axles (x = +3.0, +1.5, 0.0, -1.5, -3.0).
    assert axle_count == 5


def test_tank_radius_extracted():
    parsed = parse_urdf(TANK_URDF)
    # All reference-tank wheel meshes are <cylinder radius="0.4">.
    for w in parsed.wheels:
        assert w.radius == pytest.approx(0.4)


def test_tank_no_steer_joints():
    parsed = parse_urdf(TANK_URDF)
    for w in parsed.wheels:
        assert w.steer_joint_name is None


def test_tank_susp_joint_names_match():
    parsed = parse_urdf(TANK_URDF)
    by_name = {w.name: w for w in parsed.wheels}
    assert by_name["l_sprocket"].susp_joint_name == "l_sprocket_susp"
    assert by_name["r_idler"].susp_joint_name == "r_idler_susp"


def test_chassis_mass_extracted():
    car = parse_urdf(CAR_URDF)
    tank = parse_urdf(TANK_URDF)
    assert car.chassis_mass == pytest.approx(2200.0)
    assert tank.chassis_mass == pytest.approx(50000.0)
