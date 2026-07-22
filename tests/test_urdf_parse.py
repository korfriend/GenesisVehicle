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


def test_sprung_unsprung_mass_split():
    """sprung + unsprung == total, and sprung includes non-wheel children
    (a turret) that chassis_mass (base link only) leaves out. (v1.2.1)"""
    tank = parse_urdf(TANK_URDF)
    assert tank.total_mass == pytest.approx(
        tank.sprung_mass + tank.unsprung_mass)
    # Base link alone is 50 t; the turret + barrel ride on the same springs.
    assert tank.sprung_mass > tank.chassis_mass
    assert tank.sprung_mass == pytest.approx(53200.0, rel=1e-3)
    # Unsprung = 10 x (wheel + carrier).
    assert tank.unsprung_mass == pytest.approx(5500.0, rel=1e-3)


def test_car_sprung_mass_equals_chassis_mass():
    """A URDF whose base link has no sprung children: the two agree, so the
    v1.2.1 switch to sprung_mass is a no-op for simple vehicles."""
    car = parse_urdf(CAR_URDF)
    assert car.sprung_mass == pytest.approx(car.chassis_mass)


_NO_SPIN_JOINT_URDF = """<?xml version="1.0"?>
<robot name="minimal">
  <link name="base_link">
    <inertial><mass value="1000.0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
  </link>
  <joint name="l_road1_susp" type="prismatic">
    <parent link="base_link"/><child link="l_road1"/>
    <origin xyz="1.0 0.8 0.0"/><axis xyz="0 0 1"/>
    <limit lower="-0.1" upper="0.1" effort="1" velocity="1"/>
  </joint>
  <link name="l_road1">
    <inertial><mass value="500.0"/>
      <inertia ixx="10" ixy="0" ixz="0" iyy="29.5" iyz="0" izz="10"/></inertial>
    <visual><geometry><cylinder radius="0.4" length="0.64"/></geometry></visual>
  </link>
  <joint name="r_road1_susp" type="prismatic">
    <parent link="base_link"/><child link="r_road1"/>
    <origin xyz="1.0 -0.8 0.0"/><axis xyz="0 0 1"/>
    <limit lower="-0.1" upper="0.1" effort="1" velocity="1"/>
  </joint>
  <link name="r_road1">
    <inertial><mass value="500.0"/>
      <inertia ixx="10" ixy="0" ixz="0" iyy="29.5" iyz="0" izz="10"/></inertial>
    <visual><geometry><cylinder radius="0.4" length="0.64"/></geometry></visual>
  </link>
</robot>
"""


def _urdf_with_susp_dynamics(attrs: str) -> str:
    return _NO_SPIN_JOINT_URDF.replace(
        '<limit lower="-0.1" upper="0.1" effort="1" velocity="1"/>',
        f'<limit lower="-0.1" upper="0.1" effort="1" velocity="1"/>'
        f'<dynamics {attrs}/>')


def test_urdf_declared_suspension_is_read(tmp_path):
    """Non-standard `<dynamics stiffness=...>` on the suspension joint is
    honoured — that attribute is the only unambiguous "this is a spring"
    marker, since standard URDF has no stiffness field. (v1.2.1)"""
    path = tmp_path / "declared.urdf"
    path.write_text(_urdf_with_susp_dynamics(
        'stiffness="250000" compression_damping="9000" extension_damping="5000"'))
    for w in parse_urdf(str(path)).wheels:
        assert w.k_susp == pytest.approx(250_000.0)
        assert w.c_compression == pytest.approx(9_000.0)
        assert w.c_extension == pytest.approx(5_000.0)


def test_urdf_symmetric_damping_applies_to_both_strokes(tmp_path):
    path = tmp_path / "sym.urdf"
    path.write_text(_urdf_with_susp_dynamics('stiffness="250000" damping="7000"'))
    for w in parse_urdf(str(path)).wheels:
        assert w.c_compression == pytest.approx(7_000.0)
        assert w.c_extension == pytest.approx(7_000.0)


def test_zero_stiffness_means_no_declared_suspension(tmp_path):
    """`stiffness="0.0"` says "no spring here" — it must NOT zero the
    suspension. The bundled car URDF writes exactly this."""
    path = tmp_path / "zero.urdf"
    path.write_text(_urdf_with_susp_dynamics(
        'damping="0.0" friction="0.0" stiffness="0.0"'))
    for w in parse_urdf(str(path)).wheels:
        assert w.k_susp is None
        assert w.c_compression is None


def test_bare_damping_is_not_a_suspension_declaration(tmp_path):
    """A `damping` with no stiffness is a solver hint in the standard schema,
    not a suspension characterization — the reference URDFs carry exactly that
    on their steer joints."""
    path = tmp_path / "bare.urdf"
    path.write_text(_urdf_with_susp_dynamics('damping="20.0" friction="0.0"'))
    for w in parse_urdf(str(path)).wheels:
        assert w.k_susp is None
        assert w.c_compression is None
        assert w.c_extension is None


def test_reference_urdfs_declare_no_suspension():
    """Neither reference vehicle declares a spring, so the v1.2.1 chain leaves
    both on their derived/default values."""
    for p in (CAR_URDF, TANK_URDF):
        for w in parse_urdf(p).wheels:
            assert w.k_susp is None


def test_wheel_geometry_read_without_a_spin_joint(tmp_path):
    """A wheel modelled as a single link straight off the suspension joint —
    no `continuous` spin joint to walk to. The suspension's own child IS the
    wheel, so its geometry/mass must still be read (v1.2.1); before the fix
    these came back None and silently fell through to module defaults."""
    path = tmp_path / "no_spin.urdf"
    path.write_text(_NO_SPIN_JOINT_URDF)
    parsed = parse_urdf(str(path))

    assert len(parsed.wheels) == 2
    for w in parsed.wheels:
        assert w.spin_joint_name is None          # there genuinely isn't one
        assert w.radius == pytest.approx(0.4)     # not DEFAULT_RADIUS (0.35)
        assert w.mass == pytest.approx(500.0)     # not DEFAULT_MASS (20.0)
        assert w.i_wheel == pytest.approx(29.5)   # not DEFAULT_I_WHEEL (1.5)
    # The wheels are unsprung even though they hang directly off the joint.
    assert parsed.sprung_mass == pytest.approx(1000.0)
    assert parsed.unsprung_mass == pytest.approx(1000.0)
