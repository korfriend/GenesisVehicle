"""Thin compatibility shims over Genesis engine APIs that changed across releases.

The SDK supports the Genesis versions the deployed clones actually run, so a
renamed/reshaped engine call is branched **once, here**, rather than at every
call site. Each shim documents which Genesis versions take which branch.

Currently shimmed:

``apply_links_wrench``
    Genesis <= 1.3.3 exposed ``RigidSolver.apply_links_external_force`` and
    ``.apply_links_external_torque`` as two separate calls. Genesis 1.4.0
    removed both and replaced them with a single
    ``RigidSolver.apply_links_external_wrench(force=..., torque=...)``. The two
    are numerically identical for the SDK's usage (``ref="link_origin"``,
    ``local=False``): both accumulate into ``cfrc_applied_vel/ang``, and the
    wrench kernel's moment arm ``(link.pos - root_COM) x force`` is exactly what
    the old force kernel added. The wrench form is also one kernel launch
    instead of two.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import numpy as np

__all__ = ["apply_links_wrench", "has_wrench_api", "link_inertial", "LinkInertialProps"]


def has_wrench_api(solver: Any) -> bool:
    """True on genesis >= 1.4.0 (single-call wrench API)."""
    return hasattr(solver, "apply_links_external_wrench")


def apply_links_wrench(solver: Any, force: Any, torque: Any,
                       links_idx: Any = None,
                       envs_idx: Optional[Any] = None) -> None:
    """Apply a world-frame force **and** torque to ``links_idx`` in one go.

    ``force`` / ``torque`` are ``(n_envs, n_links, 3)`` (or ``(n_links, 3)`` for
    an unbatched scene), matching what the underlying engine call expects.

    Resolved per call (not cached) so the server's monkey patches in
    ``server/vehicle_builder.apply_monkey_patches`` — which rebind these
    attributes on the solver instance — stay in effect.
    """
    fn = getattr(solver, "apply_links_external_wrench", None)
    if fn is not None:                                   # genesis >= 1.4.0
        fn(force=force, torque=torque, links_idx=links_idx, envs_idx=envs_idx)
        return
    solver.apply_links_external_force(force, links_idx, envs_idx)    # <= 1.3.3
    solver.apply_links_external_torque(torque, links_idx, envs_idx)


class LinkInertialProps(NamedTuple):
    """A link's authored inertial frame, engine-version independent.

    ``i`` is the 3x3 inertia tensor expressed in the inertial frame, which
    ``quat`` orients relative to the link frame — so the link-frame tensor is
    ``R(quat) @ i @ R(quat).T`` on every genesis version. (Genesis 1.4.0
    diagonalizes an authored tensor onto its principal axes and carries the
    rotation in ``quat``; 1.3.3 kept the authored tensor with an identity
    ``quat``. Both spell the same physical inertia.)"""
    mass: float
    pos: Any                                        # (3,) COM offset, link frame
    quat: Any                                       # (4,) wxyz, inertial frame
    i: Any                                          # (3, 3) inertia, inertial frame


def link_inertial(link: Any) -> Optional[LinkInertialProps]:
    """Read a link's inertial properties, or None if the link carries no mass.

    Genesis <= 1.3.3 exposed ``RigidLink.inertial_mass / .inertial_pos /
    .inertial_quat / .inertial_i``. Genesis 1.4.0 dropped all four and moved the
    authored values onto the link's resolved description
    (``link.desc.mass / .inertial_pos / .inertial_quat / .inertia``)."""
    mass = getattr(link, "inertial_mass", None)
    if mass is not None:                                        # genesis <= 1.3.3
        pos, quat = getattr(link, "inertial_pos", None), getattr(link, "inertial_quat", None)
        inertia = getattr(link, "inertial_i", None)
    else:                                                       # genesis >= 1.4.0
        desc = getattr(link, "desc", None)
        if desc is None:
            return None
        mass = getattr(desc, "mass", None)
        pos, quat = getattr(desc, "inertial_pos", None), getattr(desc, "inertial_quat", None)
        inertia = getattr(desc, "inertia", None)
    if mass is None:
        return None
    mass = float(mass)
    if mass <= 0.0:
        return None
    pos = np.zeros(3) if pos is None else np.asarray(pos, dtype=float).reshape(3)
    quat = None if quat is None else np.asarray(quat, dtype=float).reshape(4)
    inertia = None if inertia is None else np.asarray(inertia, dtype=float).reshape(3, 3)
    return LinkInertialProps(mass, pos, quat, inertia)
