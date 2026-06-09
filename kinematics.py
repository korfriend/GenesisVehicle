"""Per-link transform extraction for a built Genesis rigid vehicle entity.

Genesis exposes every link's **world** pose
(``entity.get_links_pos`` / ``get_links_quat``). For most consumers that is
not the natural frame:

* telemetry → animation retargeting wants each joint **relative to its URDF
  parent** (that is exactly what a rig's local channels are),
* placing ghost copies / attaching external sensors or effects wants either
  world (to drop in place) or a chassis-relative frame.

``get_link_transforms`` composes those frames from the world poses Genesis
gives, walking the URDF parent topology that the built entity already knows
(``link.parent_idx``). It returns a batched, vectorised result — no per-link
Python loop over ``n_envs``.

Frames
------
``"world"``   each link in world coordinates (raw Genesis output).
``"base"``    each link relative to the entity **base/root link**
              (one frame for the whole vehicle; base link itself = identity).
``"parent"``  each link relative to its **immediate parent link** in the URDF
              tree (true hierarchy-local). The root link has no parent, so it
              is returned in world coordinates.

Conventions
-----------
Quaternions are ``wxyz`` (Genesis convention). Positions are metres. Output
tensors are ``(n_envs, n_links, …)`` when the entity was built batched
(``n_envs > 0``) and ``(n_links, …)`` for a single-env build, matching what
``get_links_pos`` returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import genesis.utils.geom as gu


_VALID_FRAMES = ("world", "base", "parent")


@dataclass
class LinkTransforms:
    """Per-link transforms of a vehicle entity in a chosen frame.

    Attributes
    ----------
    frame : str
        One of ``"world"``, ``"base"``, ``"parent"`` — the frame ``pos``/``quat``
        are expressed in.
    names : list[str]
        Link names, ordered by entity-local link index (``names[i]`` ↔ column
        ``i`` of ``pos``/``quat``).
    parent_local : list[int]
        For each link, the **entity-local index** of its URDF parent link, or
        ``-1`` if the link is the root (its ``"parent"`` frame is world).
    pos : torch.Tensor
        ``(n_envs, n_links, 3)`` (or ``(n_links, 3)`` single-env) translation.
    quat : torch.Tensor
        ``(n_envs, n_links, 4)`` (or ``(n_links, 4)`` single-env) ``wxyz``.
    """

    frame: str
    names: List[str]
    parent_local: List[int]
    pos: torch.Tensor
    quat: torch.Tensor

    @property
    def n_links(self) -> int:
        return len(self.names)

    def matrices(self) -> torch.Tensor:
        """Return homogeneous 4×4 transforms, shape ``(..., n_links, 4, 4)``."""
        R = gu.quat_to_R(self.quat)                       # (..., 3, 3)
        T = torch.zeros((*self.quat.shape[:-1], 4, 4), dtype=self.pos.dtype, device=self.pos.device)
        T[..., :3, :3] = R
        T[..., :3, 3] = self.pos
        T[..., 3, 3] = 1.0
        return T

    def index(self, name: str) -> int:
        """Local link index of ``name`` (raises ``KeyError`` if absent)."""
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(name) from None


def get_link_transforms(
    entity: Any,
    frame: str = "parent",
    *,
    envs_idx: Optional[Any] = None,
) -> LinkTransforms:
    """Extract every link's transform from a **built** Genesis rigid entity.

    Parameters
    ----------
    entity : RigidEntity
        A vehicle entity already added to a built scene
        (``car = scene.add_entity(gs.morphs.URDF(...))``; ``scene.build(...)``).
    frame : {"world", "base", "parent"}
        Target frame — see module docstring. Default ``"parent"``
        (URDF-hierarchy-local), the frame an animation rig consumes.
    envs_idx : optional
        Subset of envs to read (passed through to ``get_links_pos/quat``).

    Returns
    -------
    LinkTransforms
    """
    if frame not in _VALID_FRAMES:
        raise ValueError(f"frame must be one of {_VALID_FRAMES}, got {frame!r}")

    links = list(entity.links)
    n_links = len(links)
    link_start = entity.link_start

    # Link metadata in entity-local index order (column order of the output).
    names: List[str] = [None] * n_links            # type: ignore[list-item]
    parent_local: List[int] = [-1] * n_links
    for lk in links:
        i = lk.idx_local
        names[i] = lk.name
        p = lk.parent_idx                          # global idx, -1 if root
        pl = p - link_start if p is not None and p >= 0 else -1
        # Guard: a parent outside this entity (cross-entity attach) → treat as
        # root for "parent" framing (returned in world).
        parent_local[i] = pl if 0 <= pl < n_links else -1

    # World poses (Genesis): (n_envs, n_links, ·) batched or (n_links, ·) single.
    wp = entity.get_links_pos(envs_idx=envs_idx)
    wq = entity.get_links_quat(envs_idx=envs_idx)
    was_2d = wp.dim() == 2
    if was_2d:
        wp = wp.unsqueeze(0)                       # (1, n_links, 3)
        wq = wq.unsqueeze(0)                       # (1, n_links, 4)

    if frame == "world":
        out_p, out_q = wp, wq
    elif frame == "base":
        # Everything relative to base link (local idx 0). Base → identity.
        inv_qb = gu.inv_quat(wq[:, 0:1, :]).expand_as(wq)          # (n_envs, n_links, 4)
        out_q = gu.transform_quat_by_quat(wq, inv_qb)             # quatmul(inv_qb, wq)
        out_p = gu.transform_by_quat(wp - wp[:, 0:1, :], inv_qb)
    else:  # "parent"
        pl_t = torch.tensor(parent_local, device=wp.device)       # (n_links,)
        valid = (pl_t >= 0).view(1, n_links, 1)
        gather_idx = pl_t.clamp(min=0)                            # root→0 (overwritten by `valid`)
        q_par = wq[:, gather_idx, :]                             # (n_envs, n_links, 4)
        p_par = wp[:, gather_idx, :]
        inv_qp = gu.inv_quat(q_par)
        rel_q = gu.transform_quat_by_quat(wq, inv_qp)            # quatmul(inv_qp, wq)
        rel_p = gu.transform_by_quat(wp - p_par, inv_qp)
        out_q = torch.where(valid, rel_q, wq)                    # root keeps world pose
        out_p = torch.where(valid, rel_p, wp)

    if was_2d:
        out_p = out_p.squeeze(0)
        out_q = out_q.squeeze(0)

    return LinkTransforms(
        frame=frame, names=names, parent_local=parent_local, pos=out_p, quat=out_q,
    )
