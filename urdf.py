"""URDF parsing for ray-wheel vehicles.

Generalises KDU/physics.py:parse_tank_urdf to support both KDU's `*_susp`
naming and HJW's `*_suspension_joint` naming. Returns a URDFParsedConfig
with per-wheel WheelConfig entries (URDF-derivable fields filled, others None).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import WheelConfig

# Suffixes that mark a prismatic joint as a wheel suspension.
_SUSP_SUFFIXES = ("_suspension_joint", "_susp")
# Tolerance (m) used when clustering wheels into axles by x-coordinate.
_AXLE_X_TOLERANCE = 0.05


@dataclass
class URDFParsedConfig:
    """Parsed URDF state used by the resolver and the visual layer."""
    base_link_name: str
    chassis_mass: Optional[float]
    wheels: list[WheelConfig]
    # steer_axis_signs[name] = +1 or -1 depending on URDF <axis xyz>; used by visual layer
    # to flip the visual command so users see the requested rotation in the viewer.
    steer_axis_signs: dict[str, int] = field(default_factory=dict)
    # Whether each suspension prismatic joint has non-zero stiffness/damping in the URDF.
    # If non-zero, visual sync should use control_dofs_position (KDU pattern); otherwise
    # set_dofs_position is preferred (HJW pattern).
    susp_has_dynamics: dict[str, bool] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_urdf(urdf_path: str) -> URDFParsedConfig:
    """Parse a URDF and extract per-wheel configuration.

    Wheel discovery: any prismatic joint whose name ends with `_susp` or
    `_suspension_joint`. For each such joint we walk down the child link chain
    to find a continuous joint with axis (0, +/-1, 0); that is the spin joint
    and its child is the wheel link.
    """
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints_by_name = {j.get("name"): j for j in root.findall("joint")}
    links_by_name = {l.get("name"): l for l in root.findall("link")}
    joints_by_parent: dict[str, list[ET.Element]] = {}
    for j in root.findall("joint"):
        parent = j.find("parent")
        if parent is None:
            continue
        joints_by_parent.setdefault(parent.get("link"), []).append(j)

    base_link_name = _find_base_link(root, joints_by_name)
    chassis_mass = _link_mass(links_by_name.get(base_link_name))

    susp_joints = [
        j for j in root.findall("joint")
        if j.get("type") == "prismatic" and any(j.get("name", "").endswith(s) for s in _SUSP_SUFFIXES)
    ]

    raw: list[dict[str, Any]] = []
    steer_axis_signs: dict[str, int] = {}
    susp_has_dynamics: dict[str, bool] = {}

    for sj in susp_joints:
        susp_name = sj.get("name")
        carrier_link_name = sj.find("child").get("link")
        position = _origin_xyz(sj)

        steer_joint, spin_joint, wheel_link_name, steer_sign = _walk_to_wheel(
            carrier_link_name, joints_by_parent
        )

        # Wheel name = the actual wheel link name when found by chain walk;
        # fall back to stripping the susp-joint suffix (KDU pattern where the
        # two coincide).
        wheel_name = wheel_link_name if wheel_link_name is not None else _strip_susp_suffix(susp_name)

        wheel_link = links_by_name.get(wheel_link_name) if wheel_link_name else None
        radius = _wheel_radius(wheel_link)
        mass = _link_mass(wheel_link)
        i_wheel = _wheel_inertia(wheel_link)

        side = _detect_side(wheel_name)
        susp_has_dynamics[susp_name] = _joint_has_dynamics(sj)
        if steer_joint is not None:
            steer_axis_signs[steer_joint.get("name")] = steer_sign

        raw.append({
            "name": wheel_name,
            "position": position,
            "side": side,
            "radius": radius,
            "mass": mass,
            "i_wheel": i_wheel,
            "susp_joint_name": susp_name,
            "steer_joint_name": steer_joint.get("name") if steer_joint is not None else None,
            "spin_joint_name": spin_joint.get("name") if spin_joint is not None else None,
        })

    axle_indices = _cluster_axles([r["position"] for r in raw], _AXLE_X_TOLERANCE)
    wheels: list[WheelConfig] = []
    for r, axle_idx in zip(raw, axle_indices):
        wheels.append(WheelConfig(
            name=r["name"],
            position=r["position"],
            side=r["side"],
            axle_index=axle_idx,
            radius=r["radius"],
            mass=r["mass"],
            i_wheel=r["i_wheel"],
            susp_joint_name=r["susp_joint_name"],
            steer_joint_name=r["steer_joint_name"],
            spin_joint_name=r["spin_joint_name"],
        ))

    return URDFParsedConfig(
        base_link_name=base_link_name,
        chassis_mass=chassis_mass,
        wheels=wheels,
        steer_axis_signs=steer_axis_signs,
        susp_has_dynamics=susp_has_dynamics,
    )


def parse_inertia_max_principal_genesis(entity: Any, link_name: str) -> float:
    """Extract max diagonal of `link.inertial_i` from a built Genesis entity.

    Used at runtime by VehiclePhysics.__init__ to override the URDF-parsed
    inertia: Genesis may rotate the inertial frame, so the URDF iyy is not
    necessarily the spin axis. The max principal moment is the spin MOI for
    a wheel (mass distributed around the spin axis).
    """
    import numpy as np
    link = entity.get_link(link_name)
    I_mat = np.asarray(link.inertial_i)
    return float(np.max(np.diag(I_mat)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_susp_suffix(name: str) -> str:
    for suf in _SUSP_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def _origin_xyz(joint: ET.Element) -> tuple[float, float, float]:
    origin = joint.find("origin")
    if origin is None or origin.get("xyz") is None:
        return (0.0, 0.0, 0.0)
    parts = origin.get("xyz").split()
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _axis_xyz(joint: ET.Element) -> tuple[float, float, float]:
    axis = joint.find("axis")
    if axis is None or axis.get("xyz") is None:
        return (1.0, 0.0, 0.0)
    parts = axis.get("xyz").split()
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _find_base_link(root: ET.Element, joints_by_name: dict[str, ET.Element]) -> str:
    """Find the unique link that never appears as a child."""
    children = set()
    for j in root.findall("joint"):
        c = j.find("child")
        if c is not None:
            children.add(c.get("link"))
    for l in root.findall("link"):
        if l.get("name") not in children:
            return l.get("name")
    return "base_link"


def _link_mass(link: Optional[ET.Element]) -> Optional[float]:
    if link is None:
        return None
    inertial = link.find("inertial")
    if inertial is None:
        return None
    mass = inertial.find("mass")
    if mass is None or mass.get("value") is None:
        return None
    return float(mass.get("value"))


def _wheel_radius(link: Optional[ET.Element]) -> Optional[float]:
    if link is None:
        return None
    for elem in link.findall(".//cylinder"):
        if elem.get("radius") is not None:
            return float(elem.get("radius"))
    for elem in link.findall(".//sphere"):
        if elem.get("radius") is not None:
            return float(elem.get("radius"))
    return None


def _wheel_inertia(link: Optional[ET.Element]) -> Optional[float]:
    """URDF-side fallback: max(ixx, iyy, izz). At runtime the SDK prefers
    parse_inertia_max_principal_genesis() which reads from the built entity."""
    if link is None:
        return None
    inertial = link.find("inertial")
    if inertial is None:
        return None
    inertia = inertial.find("inertia")
    if inertia is None:
        return None
    diag = []
    for k in ("ixx", "iyy", "izz"):
        v = inertia.get(k)
        if v is None:
            return None
        diag.append(float(v))
    return max(diag)


def _joint_has_dynamics(joint: ET.Element) -> bool:
    dyn = joint.find("dynamics")
    if dyn is None:
        return False
    for k in ("damping", "stiffness", "friction"):
        v = dyn.get(k)
        if v is not None and float(v) != 0.0:
            return True
    return False


def _walk_to_wheel(
    carrier_link: str,
    joints_by_parent: dict[str, list[ET.Element]],
    max_depth: int = 6,
) -> tuple[Optional[ET.Element], Optional[ET.Element], Optional[str], int]:
    """Walk from a carrier link down the joint tree to find:
      - steer joint: revolute joint with axis (0, 0, +-1)  (optional)
      - spin joint:  continuous joint with axis (0, +-1, 0) (required for a wheel)
    Returns (steer_joint, spin_joint, wheel_link_name, steer_axis_sign).
    steer_axis_sign is +1 if URDF axis is (0, 0, +1), -1 if (0, 0, -1), 0 if no steer joint.
    """
    steer_joint: Optional[ET.Element] = None
    steer_sign = 0
    cur = carrier_link
    for _ in range(max_depth):
        children = joints_by_parent.get(cur, [])
        if not children:
            return steer_joint, None, None, steer_sign
        # Prefer a continuous joint with y-axis; that's the spin joint.
        spin_candidate = None
        for j in children:
            jtype = j.get("type")
            ax = _axis_xyz(j)
            if jtype == "continuous" and abs(ax[1]) > 0.5 and abs(ax[0]) < 0.5 and abs(ax[2]) < 0.5:
                spin_candidate = j
                break
        if spin_candidate is not None:
            wheel_link_name = spin_candidate.find("child").get("link")
            return steer_joint, spin_candidate, wheel_link_name, steer_sign
        # Otherwise: revolute with z-axis is the steer joint; descend into its child.
        steer_candidate = None
        for j in children:
            jtype = j.get("type")
            ax = _axis_xyz(j)
            if jtype == "revolute" and abs(ax[2]) > 0.5 and abs(ax[0]) < 0.5 and abs(ax[1]) < 0.5:
                steer_candidate = j
                break
        if steer_candidate is not None:
            steer_joint = steer_candidate
            steer_sign = 1 if _axis_xyz(steer_candidate)[2] >= 0 else -1
            cur = steer_candidate.find("child").get("link")
            continue
        # Fallback: descend into any single child to keep walking.
        cur = children[0].find("child").get("link")
    return steer_joint, None, None, steer_sign


def _detect_side(wheel_name: str) -> Optional[str]:
    """Return 'L' / 'R' / None based on URDF naming conventions."""
    n = wheel_name.lower()
    if n.startswith("l_") or "_left_" in n or "_left" in n or n.startswith("left_"):
        return "L"
    if n.startswith("r_") or "_right_" in n or "_right" in n or n.startswith("right_"):
        return "R"
    return None


def _cluster_axles(positions: list[tuple[float, float, float]], tol: float) -> list[int]:
    """Cluster wheels by x-coordinate (within `tol`), then sort cluster centroids
    descending so axle 0 is the front-most."""
    if not positions:
        return []
    indexed = list(enumerate(positions))
    # Sort by x descending so we encounter front first.
    indexed.sort(key=lambda t: -t[1][0])
    axle_of: dict[int, int] = {}
    cluster_x: list[float] = []
    for orig_idx, pos in indexed:
        x = pos[0]
        # Find an existing cluster within tolerance.
        match = None
        for ci, cx in enumerate(cluster_x):
            if abs(cx - x) <= tol:
                match = ci
                break
        if match is None:
            match = len(cluster_x)
            cluster_x.append(x)
        axle_of[orig_idx] = match
    return [axle_of[i] for i in range(len(positions))]
