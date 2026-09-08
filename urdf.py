"""URDF parsing for ray-wheel vehicles.

Supports both the `*_susp` and `*_suspension_joint` suspension-joint
naming conventions. Returns a URDFParsedConfig
with per-wheel WheelConfig entries (URDF-derivable fields filled, others None).
"""

from __future__ import annotations

import math
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
    # Total mass of every link in the URDF.
    total_mass: Optional[float] = None
    # Mass carried by the springs = total - everything below a suspension joint
    # (carrier + wheel + any descendant). This is what a suspension rate must be
    # sized against; `chassis_mass` is the BASE LINK ALONE and silently omits
    # sprung children such as a turret or a cargo body. (v1.2.1)
    sprung_mass: Optional[float] = None
    unsprung_mass: Optional[float] = None
    # steer_axis_signs[name] = +1 or -1 depending on URDF <axis xyz>; used by visual layer
    # to flip the visual command so users see the requested rotation in the viewer.
    steer_axis_signs: dict[str, int] = field(default_factory=dict)
    # Whether each suspension prismatic joint has non-zero stiffness/damping in the URDF.
    # If non-zero, visual sync should use control_dofs_position; otherwise
    # set_dofs_position is preferred.
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
        # fall back to stripping the susp-joint suffix (for URDFs where the
        # two coincide).
        wheel_name = wheel_link_name if wheel_link_name is not None else _strip_susp_suffix(susp_name)

        # No spin joint below the suspension (a URDF that models the wheel as a
        # single link hanging straight off the prismatic joint): the suspension's
        # own child IS the wheel, so read its geometry/mass from there rather
        # than falling through to module defaults. (v1.2.1 — until then such a
        # vehicle silently got DEFAULT_RADIUS/DEFAULT_MASS and needed a
        # hard-coded radius override to behave.)
        geom_link_name = wheel_link_name or carrier_link_name
        wheel_link = links_by_name.get(geom_link_name) if geom_link_name else None
        radius = _wheel_radius(wheel_link)
        mass = _link_mass(wheel_link)
        # Project the inertia tensor onto the axis this wheel actually spins
        # about (v1.2.8). Falls back to the SDK's +Y convention when the URDF
        # models the wheel without a spin joint.
        i_wheel = _wheel_inertia(
            wheel_link,
            _axis_xyz(spin_joint) if spin_joint is not None else (0.0, 1.0, 0.0))

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
            "susp_dynamics": _susp_dynamics(sj),
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
            # k_susp / c_compression / c_extension stay None unless the URDF
            # declared them; resolve() then fills them from the caller's config
            # or the module defaults, giving the priority chain
            #   caller override > URDF <dynamics> > mass-derived / default.
            **r["susp_dynamics"],
        ))

    total_mass, sprung_mass, unsprung_mass = _mass_split(
        links_by_name, joints_by_parent, susp_joints
    )

    return URDFParsedConfig(
        base_link_name=base_link_name,
        chassis_mass=chassis_mass,
        wheels=wheels,
        total_mass=total_mass,
        sprung_mass=sprung_mass,
        unsprung_mass=unsprung_mass,
        steer_axis_signs=steer_axis_signs,
        susp_has_dynamics=susp_has_dynamics,
    )


def estimate_spin_inertia_from_genesis(
    entity: Any,
    link_name: str,
    spin_axis_local: Optional[tuple[float, float, float]] = None,
) -> float:
    """Estimate a wheel's spin moment of inertia from a built Genesis entity.

    This is a FALLBACK estimate consulted only when ``WheelConfig.i_wheel``
    is not explicitly supplied by the user (and the URDF-derived value is
    also unavailable). The user's `WheelConfig.i_wheel` is always the
    authoritative truth when set.

    Caveats:
      - Genesis stores the link inertia in the link's INERTIAL frame, which may
        be rotated relative to the body frame — and genesis >= 1.4.0 always
        diagonalizes an authored tensor onto its principal axes, permuting the
        components in the process. The tensor is therefore rotated into the BODY
        frame with the link's ``inertial_quat`` before anything is read off it,
        which makes the result identical on every genesis version. (Projecting
        the raw stored diagonal, as this did before v1.5.0, silently read the
        wrong component on >= 1.4.0: the reference car's wheel is
        ``ixx=1, iyy=2, izz=1`` and its spin MOI about +Y came back as 1.0
        instead of 2.0.)
      - With ``spin_axis_local`` a unit 3-vector in body coordinates, the spin
        MOI is the exact ``I_about_axis = a^T I_body a``.
      - Falls back to ``max(diag(I_body))`` when ``spin_axis_local`` is None or
        on numerical trouble: for a cylindrical wheel the largest principal
        moment is the one about the spin axis, but that is a heuristic for
        general shapes — pass ``spin_axis_local``.

    Reads the tensor through :func:`genesis_vehicle._gs_compat.link_inertial`
    (genesis <= 1.3.3 spells it ``link.inertial_i``, >= 1.4.0
    ``link.desc.inertia``). Raises ``ValueError`` for a link with no inertia,
    which the caller in ``core.py`` treats like any other failure — keep the
    authored ``i_wheel``.
    """
    import numpy as np
    from ._gs_compat import link_inertial
    link = entity.get_link(link_name)
    props = link_inertial(link)
    if props is None or props.i is None:
        raise ValueError(
            f"link {link_name!r} carries no inertia to estimate a spin MOI from")
    I_body = _inertia_in_link_frame(np.asarray(props.i, dtype=float), props.quat)
    diag = np.diag(I_body)
    if spin_axis_local is None:
        return float(np.max(diag))
    a = np.asarray(spin_axis_local, dtype=float)
    norm = float(np.linalg.norm(a))
    if norm < 1e-12:
        return float(np.max(diag))
    a = a / norm
    I_about = float(a @ I_body @ a)
    if not np.isfinite(I_about) or I_about <= 0.0:
        return float(np.max(diag))
    return I_about


def _inertia_in_link_frame(I_inertial, quat) -> "Any":
    """Rotate an inertia tensor from the link's inertial frame into the link
    frame: ``I_body = R I R^T`` with ``R`` from the wxyz ``quat``. Identity
    (a copy) when ``quat`` is None or degenerate."""
    import numpy as np
    I_inertial = np.asarray(I_inertial, dtype=float)
    if I_inertial.ndim == 1:                     # a bare diagonal
        I_inertial = np.diag(I_inertial)
    if quat is None:
        return I_inertial
    q = np.asarray(quat, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-12:
        return I_inertial
    w, x, y, z = q / n
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])
    return R @ I_inertial @ R.T


def self_collision_ceiling(
    urdf_path: str,
    ray_positions,
    *,
    margin: float = 0.02,
) -> Optional[float]:
    """How far a wheel ray may start ABOVE its attachment point before it would
    begin inside — or above — the vehicle's OWN collision geometry.

    Only ``raycast_mode="single_scene"`` needs this. There the wheel rays are
    cast in the SAME scene the chassis collides in, so a ray origin lifted above
    the chassis collision box (``raycast.RAY_UP_OFFSET``, 1.0 m by default) hits
    the vehicle's own roof instead of the ground — the "launches into sky" bug
    class the URDF comments warn about. ``dual_scene`` raycasts in a scene that
    holds no vehicle collision geometry at all, so it is unaffected and keeps
    the full offset.

    Returns the smallest gap over ``ray_positions``, minus ``margin``, clamped
    at 0.0; or ``None`` when no own-collision geometry sits above any ray, which
    means the offset needs no cap. Geometry is evaluated at the URDF REST pose
    (every joint at zero), which is the pose the ray origins are authored in.

    ``ray_positions`` is an iterable of chassis-local ``(x, y, z)`` wheel
    attachment points.

    A collision geom is treated as an obstruction when its rest-pose
    axis-aligned bounding box spans the ray's ``(x, y)``. Using the AABB rather
    than the exact surface over-approximates a rotated or round geom, which can
    only shrink the offset — never leave a ray starting inside the body.
    """
    boxes = _collision_aabbs(urdf_path)
    if not boxes:
        return None
    gap: Optional[float] = None
    for (x, y, z) in ray_positions:
        for (lo, hi) in boxes:
            if not (lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]):
                continue
            if hi[2] <= z:
                continue                       # entirely below the ray origin
            here = max(0.0, lo[2] - z)         # 0 if the ray starts inside it
            gap = here if gap is None else min(gap, here)
    if gap is None:
        return None
    return max(0.0, gap - margin)


def _collision_aabbs(urdf_path: str) -> list[tuple[Any, Any]]:
    """Rest-pose axis-aligned bounds of every ``<collision>`` in the URDF, in the
    BASE-LINK frame, as a list of ``(min_xyz, max_xyz)`` numpy pairs.

    Link poses come from composing the ``<origin>`` of each joint on the path
    from the base link, with every joint at zero — a URDF's rest pose. A geom
    whose extent cannot be determined (an unreadable mesh, an unknown shape) is
    skipped rather than guessed at: it drops out of the obstruction test, which
    is the same outcome as there being nothing there.
    """
    import numpy as np

    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return []
    joints_by_parent: dict[str, list[ET.Element]] = {}
    joints_by_name = {j.get("name"): j for j in root.findall("joint")}
    for j in root.findall("joint"):
        parent = j.find("parent")
        if parent is not None:
            joints_by_parent.setdefault(parent.get("link"), []).append(j)
    base = _find_base_link(root, joints_by_name)

    # Breadth-first walk from the base link, composing (R, t) per link.
    poses: dict[str, tuple[Any, Any]] = {base: (np.eye(3), np.zeros(3))}
    queue = [base]
    while queue:
        name = queue.pop()
        R_p, t_p = poses[name]
        for j in joints_by_parent.get(name, ()):
            child = j.find("child")
            if child is None or child.get("link") in poses:
                continue
            R_j = _rpy_to_R(_origin_rpy(j))
            t_j = np.asarray(_origin_xyz(j), dtype=float)
            poses[child.get("link")] = (R_p @ R_j, t_p + R_p @ t_j)
            queue.append(child.get("link"))

    out: list[tuple[Any, Any]] = []
    for link in root.findall("link"):
        pose = poses.get(link.get("name"))
        if pose is None:                        # not connected to the base link
            continue
        R_l, t_l = pose
        for col in link.findall("collision"):
            half = _collision_half_extents(col)
            if half is None:
                continue
            R_c = R_l @ _rpy_to_R(_origin_rpy(col))
            t_c = t_l + R_l @ np.asarray(_origin_xyz(col), dtype=float)
            # AABB of the rotated box: |R| @ half_extents (the standard bound).
            radius = np.abs(R_c) @ half
            out.append((t_c - radius, t_c + radius))
    return out


def _collision_half_extents(collision: ET.Element):
    """Half-extents of a ``<collision>``'s shape in its own frame, or None when
    the shape is unknown or a mesh whose bounds cannot be read."""
    import numpy as np

    geom = collision.find("geometry")
    if geom is None:
        return None
    box = geom.find("box")
    if box is not None and box.get("size"):
        return np.asarray([float(v) for v in box.get("size").split()], dtype=float) / 2.0
    sphere = geom.find("sphere")
    if sphere is not None and sphere.get("radius"):
        r = float(sphere.get("radius"))
        return np.asarray([r, r, r], dtype=float)
    cyl = geom.find("cylinder")
    if cyl is not None and cyl.get("radius") and cyl.get("length"):
        r, h = float(cyl.get("radius")), float(cyl.get("length"))
        return np.asarray([r, r, h / 2.0], dtype=float)      # URDF cylinders are +z
    mesh = geom.find("mesh")
    if mesh is not None and mesh.get("filename"):
        return _mesh_half_extents(mesh)
    return None


def _mesh_half_extents(mesh: ET.Element):
    """Half-extents of a ``<mesh>`` from its file's bounding box, scaled by the
    element's ``scale``. None if the file cannot be read — trimesh is a Genesis
    dependency, but the path may be a package:// URI we cannot resolve."""
    import numpy as np

    path = mesh.get("filename")
    if path.startswith(("package://", "model://")):
        return None
    try:
        import trimesh
        loaded = trimesh.load(path, force="mesh", skip_materials=True)
        lo, hi = np.asarray(loaded.bounds, dtype=float)
    except Exception:
        return None
    scale = mesh.get("scale")
    half = (hi - lo) / 2.0
    if scale:
        half = half * np.abs(np.asarray([float(v) for v in scale.split()], dtype=float))
    return half


def _rpy_to_R(rpy):
    """URDF fixed-axis roll-pitch-yaw (Rz @ Ry @ Rx) to a 3x3 rotation matrix."""
    import numpy as np

    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mass_split(
    links_by_name: dict[str, ET.Element],
    joints_by_parent: dict[str, list[ET.Element]],
    susp_joints: list[ET.Element],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Split the URDF's link masses into sprung / unsprung.

    Unsprung = every link at or below a suspension joint's child (the carrier,
    the wheel, and anything hanging off them). Sprung = the rest, i.e. what the
    springs actually carry: base link PLUS its non-wheel descendants (turret,
    cargo body, arm, ...). Sizing a spring rate against `chassis_mass` (the base
    link alone) undercounts whenever such a child exists.

    Returns (total, sprung, unsprung); all None if no link declares a mass.
    """
    masses = {n: _link_mass(l) for n, l in links_by_name.items()}
    if not any(m is not None for m in masses.values()):
        return (None, None, None)

    unsprung_links: set[str] = set()
    for sj in susp_joints:
        child = sj.find("child")
        if child is None:
            continue
        # Walk the subtree below the suspension joint. Guard against a malformed
        # URDF with a cycle by refusing to revisit a link.
        stack = [child.get("link")]
        while stack:
            link = stack.pop()
            if link is None or link in unsprung_links:
                continue
            unsprung_links.add(link)
            for j in joints_by_parent.get(link, []):
                c = j.find("child")
                if c is not None:
                    stack.append(c.get("link"))

    total = sum(m for m in masses.values() if m is not None)
    unsprung = sum(m for n, m in masses.items() if m is not None and n in unsprung_links)
    return (total, total - unsprung, unsprung)


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


def _origin_rpy(element: ET.Element) -> tuple[float, float, float]:
    """`<origin rpy="r p y">` of any element that carries one (rad). Zeros if absent."""
    origin = element.find("origin")
    if origin is None or origin.get("rpy") is None:
        return (0.0, 0.0, 0.0)
    parts = origin.get("rpy").split()
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


def _rpy_matrix(r: float, p: float, y: float) -> list[list[float]]:
    """URDF `rpy` (fixed-axis XYZ) -> rotation matrix ``Rz(y) @ Ry(p) @ Rx(r)``."""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ]


def _wheel_inertia(link: Optional[ET.Element],
                   spin_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
                   ) -> Optional[float]:
    """Wheel spin moment of inertia **about the spin axis** (kg*m^2).

    URDF states the full inertia tensor in the link's inertial frame; what the
    ray-wheel pipeline needs (``domega = T / I_WHEEL``) is the component about
    the axis the wheel actually turns on:

        I_spin = a^T (R I R^T) a          a = unit spin axis (child-link frame)
                                          R = rotation from `<inertial rpy>`

    Until v1.2.8 this returned ``max(ixx, iyy, izz)``, which is only correct
    for a disc-like wheel (length < sqrt(3)*radius) where the spin term IS the
    largest. For a WIDE, small-diameter wheel — a tracked/UGV road wheel, e.g.
    r=0.124 m and 0.367 m across — the transverse term is the larger one, so
    max() returned roughly 2x the true spin inertia and the wheel accelerated
    half as fast as the URDF describes.

    Falls back to ``max(diag)`` when the projection is unusable (degenerate
    axis, non-positive result), so a malformed tensor can never make it zero.
    """
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
    ixx, iyy, izz = diag
    ixy = float(inertia.get("ixy", 0.0) or 0.0)
    ixz = float(inertia.get("ixz", 0.0) or 0.0)
    iyz = float(inertia.get("iyz", 0.0) or 0.0)
    tensor = [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]

    # Rotate the tensor out of the inertial frame into the link frame, which is
    # the frame the joint axis is expressed in.
    rpy = _origin_rpy(inertial)
    if any(abs(v) > 1e-12 for v in rpy):
        R = _rpy_matrix(*rpy)
        RI = [[sum(R[i][k] * tensor[k][j] for k in range(3)) for j in range(3)]
              for i in range(3)]
        tensor = [[sum(RI[i][k] * R[j][k] for k in range(3)) for j in range(3)]
                  for i in range(3)]

    norm = math.sqrt(sum(c * c for c in spin_axis))
    if norm > 1e-9:
        a = [c / norm for c in spin_axis]
        val = sum(a[i] * tensor[i][j] * a[j] for i in range(3) for j in range(3))
        if math.isfinite(val) and val > 0.0:
            return val
    return max(diag)


def _susp_dynamics(joint: ET.Element) -> dict[str, float]:
    """Read ray-wheel suspension values off a suspension joint's `<dynamics>`.

    Standard URDF has **no spring-stiffness field** — `<dynamics>` carries only
    `damping` and `friction`, and on a prismatic joint those describe the
    articulated solver, not a suspension characterization. A `stiffness` /
    `spring_stiffness` attribute is a non-standard extension, and its presence
    with a non-zero value is the only unambiguous signal that the author meant
    "this is a suspension spring".

    So: honour the tag **only** when it declares a non-zero stiffness, and only
    then read its damping alongside. A bare `damping="20.0"` is ignored — the
    reference URDFs carry exactly that on their steer joints, and one of them
    writes `stiffness="0.0"` on the suspension joint to say "no spring here".

    Returns {} when nothing should be honoured. (v1.2.1)
    """
    dyn = joint.find("dynamics")
    if dyn is None:
        return {}
    raw = dyn.get("stiffness") or dyn.get("spring_stiffness")
    try:
        k = float(raw) if raw is not None else 0.0
    except ValueError:
        return {}
    if k <= 0.0:
        return {}

    out: dict[str, float] = {"k_susp": k}

    def _f(name: str) -> Optional[float]:
        v = dyn.get(name)
        try:
            return float(v) if v is not None else None
        except ValueError:
            return None

    sym = _f("damping")
    if sym is not None and sym > 0.0:
        out["c_compression"] = sym
        out["c_extension"] = sym
    comp = _f("compression_damping")
    if comp is not None and comp > 0.0:
        out["c_compression"] = comp
    ext = _f("extension_damping")
    if ext is not None and ext > 0.0:
        out["c_extension"] = ext
    return out


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
