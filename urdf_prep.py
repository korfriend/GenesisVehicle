"""Make an arbitrary vehicle URDF ray-wheel ready (v1.1.22).

The ray-wheel model has three URDF contracts (``docs/physics-contracts.md``).
The SDK's own vehicles satisfy them by construction; a URDF authored for a
normal rigid-body simulator usually does not. Rather than making every
caller hand-patch its file, ``prepare_vehicle_urdf`` writes a corrected
temp copy (next to the original, so relative mesh paths keep working) and
leaves the original untouched:

1. **Wheels must not collide.** Ground contact IS the raycast + suspension
   force model; a colliding wheel adds a second, fighting support (the
   vehicle then jitters in place, or rides on its colliders while the
   suspension pushes with several times its weight). Wheel colliders are
   removed — but a collider that is the wheel's ONLY geometry is first
   promoted to a ``<visual>``, so the wheel still RENDERS (the instanced
   wheel renderer draws the visual geoms; physics never touches them).

2. **The suspension attach point IS the wheel centre.** The ray is cast
   down from ``WheelConfig.position`` (= the prismatic joint's origin) and
   ``rest_d = radius + rest_stroke`` measures from there. A URDF that hangs
   the wheel off a carrier link (``body --susp--> carrier --spin(z=+h)-->
   wheel``) puts the attach ``h`` BELOW the wheel centre, and the hull then
   settles ``h`` too high with the wheels visibly floating. The spin-joint
   offset is folded into the suspension origin (and zeroed on the spin
   joint), which leaves every link's rest pose unchanged.

3. **Moving links need a valid inertial.** Links without ``<inertial>``
   (mass = inertia = 0) make the articulated chain degenerate: Genesis
   falls back to its legacy URDF parser and the hull stops responding to
   forces properly. A small inertial is injected where one is missing.

Idempotent and cheap: if the URDF already satisfies all three contracts
(every SDK-authored vehicle does), the original path is returned unchanged
and no temp file is created.
"""
from __future__ import annotations

import atexit
import copy
import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from typing import Optional

_logger = logging.getLogger("genesis_vehicle.urdf_prep")

_TEMP_FILES: list[str] = []

#: Inertial injected into links that declare none (kg, kg·m²).
_FALLBACK_MASS = 1.0
_FALLBACK_INERTIA = 0.01

_EPS = 1e-6


@atexit.register
def _cleanup_temp_urdfs() -> None:
    for path in _TEMP_FILES:
        try:
            os.remove(path)
        except OSError:
            pass


def _is_wheel_link(name: str, wheel_links: set[str]) -> bool:
    n = (name or "").lower()
    return name in wheel_links or "wheel" in n or "track" in n


def _wheel_link_names(root: ET.Element) -> set[str]:
    """Links reachable from a prismatic (suspension) joint — the carrier and
    everything below it (carrier -> [steer] -> spin -> wheel)."""
    children_of: dict[str, list[ET.Element]] = {}
    for j in root.findall("joint"):
        children_of.setdefault(j.find("parent").get("link"), []).append(j)

    names: set[str] = set()
    stack = [j.find("child").get("link")
             for j in root.findall("joint") if j.get("type") == "prismatic"]
    while stack:
        link = stack.pop()
        if link in names:
            continue
        names.add(link)
        for j in children_of.get(link, []):
            stack.append(j.find("child").get("link"))
    return names


def _promote_collision_to_visual(link: ET.Element) -> int:
    """A wheel whose only geometry is a collider still has to be DRAWN. Copy
    each collider into a visual (render-only) before the colliders go."""
    if link.findall("visual"):
        return 0
    made = 0
    for col in link.findall("collision"):
        vis = ET.SubElement(link, "visual")
        for child in col:
            vis.append(copy.deepcopy(child))
        made += 1
    return made


def prepare_vehicle_urdf(
    urdf_path: str,
    *,
    strip_wheel_colliders: bool = True,
    fold_spin_offsets: bool = True,
    fix_missing_inertials: bool = True,
    quiet: bool = False,
) -> str:
    """Return a ray-wheel-ready URDF path (the original when it already is).

    See the module docstring for what is corrected and why. The temp file is
    written next to the original (relative mesh paths keep working) and
    removed at process exit.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    wheel_links = _wheel_link_names(root)

    promoted = removed = inertials = shifted = 0

    # (1) wheels render, but never collide
    if strip_wheel_colliders:
        for link in root.findall("link"):
            if not _is_wheel_link(link.get("name") or "", wheel_links):
                continue
            cols = link.findall("collision")
            if not cols:
                continue
            promoted += _promote_collision_to_visual(link)
            for col in cols:
                link.remove(col)
                removed += 1

    # (2) suspension attach == wheel centre
    if fold_spin_offsets:
        children_of: dict[str, list[ET.Element]] = {}
        for j in root.findall("joint"):
            children_of.setdefault(j.find("parent").get("link"), []).append(j)
        for susp in root.findall("joint"):
            if susp.get("type") != "prismatic":
                continue
            carrier = susp.find("child").get("link")
            for spin in children_of.get(carrier, []):
                if spin.get("type") not in ("continuous", "revolute"):
                    continue
                s_org = spin.find("origin")
                if s_org is None:
                    continue
                off = [float(v) for v in s_org.get("xyz", "0 0 0").split()]
                if max(abs(v) for v in off) < _EPS:
                    continue
                j_org = susp.find("origin")
                if j_org is None:
                    j_org = ET.SubElement(susp, "origin",
                                          {"xyz": "0 0 0", "rpy": "0 0 0"})
                base = [float(v) for v in j_org.get("xyz", "0 0 0").split()]
                j_org.set("xyz", " ".join(f"{b + o:.9f}"
                                          for b, o in zip(base, off)))
                s_org.set("xyz", "0 0 0")
                shifted += 1

    # (3) every moving link needs a valid inertial
    if fix_missing_inertials:
        for link in root.findall("link"):
            if link.find("inertial") is not None:
                continue
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            ET.SubElement(inertial, "mass", {"value": str(_FALLBACK_MASS)})
            ET.SubElement(inertial, "inertia", {
                "ixx": str(_FALLBACK_INERTIA), "ixy": "0", "ixz": "0",
                "iyy": str(_FALLBACK_INERTIA), "iyz": "0",
                "izz": str(_FALLBACK_INERTIA)})
            inertials += 1

    if not (removed or shifted or inertials):
        return urdf_path                       # already ray-wheel ready

    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    fd, tmp = tempfile.mkstemp(suffix=".urdf", prefix="_gv_prep_", dir=urdf_dir)
    os.close(fd)
    tree.write(tmp)
    _TEMP_FILES.append(tmp)

    if not quiet:
        parts = []
        if removed:
            parts.append(f"{removed} wheel collider(s) -> render-only"
                         + (f" ({promoted} promoted to <visual>)" if promoted else ""))
        if shifted:
            parts.append(f"{shifted} suspension origin(s) moved onto the wheel centre")
        if inertials:
            parts.append(f"{inertials} missing inertial(s) injected")
        print(f"[genesis_vehicle] urdf prep ({os.path.basename(urdf_path)}): "
              + "; ".join(parts))
    return tmp
