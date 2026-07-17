"""urdf_prep unit tests — pure XML, no Genesis needed."""
import os
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.urdf_prep import prepare_vehicle_urdf

_HERE = os.path.dirname(os.path.abspath(__file__))
SDK_TANK = os.path.join(_HERE, "..", "samples", "urdf", "tank_ray.urdf")
SDK_CAR = os.path.join(_HERE, "..", "samples", "urdf", "car_4w.urdf")


def _offender_urdf(tmp_path, *, collider_only_wheel=False):
    """A URDF that breaks all three ray-wheel contracts (heavy-tank export shape):
    body --susp(z=0)--> carrier(no inertial) --spin(z=+0.4)--> wheel(collider).
    """
    wheel_geom = "sphere"
    visual = "" if collider_only_wheel else """
    <visual><geometry><sphere radius="0.3"/></geometry></visual>"""
    xml = f"""<?xml version="1.0"?>
<robot name="offender">
  <link name="body">
    <inertial><mass value="1000"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
  </link>
  <link name="carrier"/>
  <link name="wheel">{visual}
    <collision><geometry><{wheel_geom} radius="0.3"/></geometry></collision>
    <inertial><mass value="50"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
  </link>
  <joint name="wheel_susp" type="prismatic">
    <parent link="body"/><child link="carrier"/>
    <origin xyz="1.0 0.5 0"/><axis xyz="0 0 1"/>
    <limit lower="-0.1" upper="0.1" effort="0" velocity="0"/>
  </joint>
  <joint name="carrier__wheel" type="continuous">
    <parent link="carrier"/><child link="wheel"/>
    <origin xyz="0 0 0.4"/><axis xyz="0 -1 0"/>
  </joint>
</robot>
"""
    p = tmp_path / "offender.urdf"
    p.write_text(xml)
    return str(p)


def test_sdk_urdfs_are_already_ray_wheel_ready():
    # No temp copy, no changes — the SDK's own vehicles satisfy the contracts.
    for src in (SDK_TANK, SDK_CAR):
        assert prepare_vehicle_urdf(src, quiet=True) == src


def test_offender_is_corrected(tmp_path):
    src = _offender_urdf(tmp_path)
    out = prepare_vehicle_urdf(src, quiet=True)
    assert out != src and os.path.exists(out)
    root = ET.parse(out).getroot()

    wheel = next(l for l in root.findall("link") if l.get("name") == "wheel")
    assert wheel.findall("collision") == [], "wheel collider must be removed"
    assert wheel.findall("visual"), "wheel must still have a visual to render"

    susp = next(j for j in root.findall("joint") if j.get("name") == "wheel_susp")
    spin = next(j for j in root.findall("joint") if j.get("name") == "carrier__wheel")
    sx = [float(v) for v in susp.find("origin").get("xyz").split()]
    px = [float(v) for v in spin.find("origin").get("xyz").split()]
    # spin offset folded into the susp origin; wheel rest pose unchanged
    assert sx == pytest.approx([1.0, 0.5, 0.4])
    assert px == pytest.approx([0.0, 0.0, 0.0])

    carrier = next(l for l in root.findall("link") if l.get("name") == "carrier")
    assert carrier.find("inertial") is not None, "missing inertial must be injected"

    # original file untouched
    orig = ET.parse(src).getroot()
    ow = next(l for l in orig.findall("link") if l.get("name") == "wheel")
    assert ow.findall("collision"), "the ORIGINAL urdf must not be modified"


def test_collider_only_wheel_is_promoted_to_visual(tmp_path):
    # A wheel whose only geometry is a collider must still RENDER: the
    # collider is promoted to a <visual> before it is stripped.
    src = _offender_urdf(tmp_path, collider_only_wheel=True)
    out = prepare_vehicle_urdf(src, quiet=True)
    wheel = next(l for l in ET.parse(out).getroot().findall("link")
                 if l.get("name") == "wheel")
    assert wheel.findall("collision") == []
    vis = wheel.findall("visual")
    assert len(vis) == 1
    assert vis[0].find("geometry").find("sphere") is not None


def test_opt_out_flags(tmp_path):
    src = _offender_urdf(tmp_path)
    out = prepare_vehicle_urdf(src, strip_wheel_colliders=False,
                               fold_spin_offsets=False, quiet=True)
    root = ET.parse(out).getroot()
    wheel = next(l for l in root.findall("link") if l.get("name") == "wheel")
    assert wheel.findall("collision"), "collider kept when opted out"
    susp = next(j for j in root.findall("joint") if j.get("name") == "wheel_susp")
    assert [float(v) for v in susp.find("origin").get("xyz").split()] == \
        pytest.approx([1.0, 0.5, 0.0]), "origin untouched when opted out"
    # inertial fix still ran (it was left enabled)
    carrier = next(l for l in root.findall("link") if l.get("name") == "carrier")
    assert carrier.find("inertial") is not None
