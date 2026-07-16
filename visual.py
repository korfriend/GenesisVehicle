"""Wheel visual-joint sync layer (for the Genesis viewer).

`WheelJointInternalSync` drives the URDF **wheel** spin / steer / suspension joints
(via ``set_dofs_position`` — kinematic, no force) so that the **Genesis
viewer** shows wheels rotating, steered, and ground-following. Scope and
non-scope, important:

- It only touches the WHEEL visual joints. It does NOT move the chassis —
  the chassis (base link) is driven by real dynamics
  (``apply_links_external_force``) and is read with ``entity.get_pos/get_quat``
  regardless of whether this runs.
- It is purely cosmetic: it does NOT feed back into the dynamics (no
  double-apply of suspension force).
- It exists for the Genesis-native viewer. An **external renderer (UE /
  Unity) does NOT need it** — use ``VehiclePhysics.wheel_visual_transforms``
  (closed-form, viewer-independent) instead; that is also cleaner (no
  substep jitter during hard transients).

Suspension visualisation: if the URDF prismatic joint declares non-zero
dynamics (KDU pattern), we use control_dofs_position with high kp/kv;
otherwise (HJW pattern, dynamics=0) we use set_dofs_position directly.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch


# Defaults used when control_dofs_position path is taken (KDU-style).
_SUSP_VIS_KP = 1.0e7
_SUSP_VIS_KV = 1.0e5

# Visual-suspension slew-rate limit (m/s): the susp visual TARGET may move at
# most this fast. The control path's PD (kp above) applies REAL joint forces,
# and the set path teleports wheel-link mass — an instantaneous air<->ground
# target jump therefore injects a force impulse into the chassis (field
# report: VJS On added ~2-3 cm extra compression on hard drop landings,
# enough to flip a marginal landing into the buried state). Slew-limiting
# the target bounds that impulse while staying far faster than any visual
# need. (v1.1.16)
_SUSP_VIS_MAX_RATE = 2.0


def _susp_visual_target(d, mesh_radius, l_susp, clamp, prev, dt):
    """Shared susp visual-joint target for both writer paths and the batched
    writer: ``mesh_radius - d``; ray MISS (>= 19.9) or exact-zero
    (unpopulated sensor) -> fully extended ``-l_susp``; clamped to the
    stroke bound; slew-rate-limited against ``prev`` (pass None to skip).

    NB a NEGATIVE ``d`` is a valid deep-over-compression reading under the
    high-cast ray scheme (raycast.RAY_UP_OFFSET), NOT air — the old
    ``d <= 1e-6`` air test would have misclassified it."""
    air = (d >= 19.9) | (d == 0.0)
    jp = mesh_radius - d
    jp = torch.where(air, torch.full_like(jp, -l_susp), jp)
    if clamp is not None:
        jp = torch.maximum(-clamp, torch.minimum(clamp, jp))
    else:
        jp = torch.clamp(jp, -l_susp, l_susp)
    if prev is not None and dt > 0.0:
        max_step = _SUSP_VIS_MAX_RATE * dt
        jp = prev + torch.clamp(jp - prev, -max_step, max_step)
    return jp

# One-time-per-process perf advisory when WheelJointInternalSync is active.
_PERF_WARNED = False


def _warn_perf_once() -> None:
    global _PERF_WARNED
    if _PERF_WARNED:
        return
    _PERF_WARNED = True
    import os, sys
    if os.environ.get("GENESIS_VEHICLE_QUIET"):
        return
    print(
        "[genesis_vehicle] PERF: WheelJointInternalSync is ENABLED — it drives the URDF "
        "wheel visual joints through the engine's articulated-body forward "
        "kinematics every step (one collider/constraint reset + FK pass per "
        "entity; batched into a single set_dofs_position in v0.7.16, ~0.85 ms/step "
        "per vehicle on CPU). It is only needed for the Genesis viewer. For an "
        "external renderer (UE / Unity), or any headless run, set "
        "enable_wheel_joint_internal_sync=False and read wheel poses from "
        "VehiclePhysics.visual_parts_transforms() / wheel_visual_transforms() "
        "(closed-form, ~µs). Silence with GENESIS_VEHICLE_QUIET=1.",
        file=sys.stderr, flush=True,
    )


class WheelJointInternalSync:
    """(Formerly ``VisualJointSync``; renamed in v1.1.19 — it is an
    internal joint-sync mechanism, not a pure visual one.)

    Drives a vehicle's URDF WHEEL visual joints (spin, steer, suspension) to
    match physics state, for the Genesis viewer. Wheels only — never the
    chassis. INTENDED to be cosmetic, but not perfectly physics-neutral:
    the set path teleports wheel-link mass and the control path's PD
    (kp=_SUSP_VIS_KP) applies real joint forces — both targets are therefore
    stroke-clamped and slew-rate-limited (_SUSP_VIS_MAX_RATE) so the residual
    disturbance stays negligible (measured < 1 cm extra compression on hard
    drop landings; was ~2-3 cm unclamped). External renderers should use
    ``VehiclePhysics.wheel_visual_transforms`` instead (viewer-independent).

    Emits a one-time-per-process performance advisory on construction (it is the
    dominant SDK cost at scale, and is unnecessary for external/headless use)."""

    def __init__(
        self,
        entity: Any,
        resolved: Any,         # ResolvedConfig
        n_envs: int,
        device: Any,
        dtype: Any,
        wheel_mesh_radius: Optional[float] = None,
    ):
        _warn_perf_once()
        self.entity = entity
        self.resolved = resolved
        self.n_envs = n_envs
        self.device = device
        self.dtype = dtype
        self.wheels = resolved.wheels
        self.n_wheels = len(self.wheels)
        self.spin_enabled = bool(getattr(resolved, "visual_spin_enabled", True))

        # Visual mesh radius for suspension positioning. Default = avg wheel radius.
        if wheel_mesh_radius is None:
            radii = [float(w.radius) for w in self.wheels if w.radius is not None]
            self.wheel_mesh_radius = float(sum(radii) / len(radii)) if radii else 0.35
        else:
            self.wheel_mesh_radius = float(wheel_mesh_radius)

        # Average rest stroke (used for the airborne suspension visual fallback).
        rest_strokes = [float(w.rest_stroke) for w in self.wheels if w.rest_stroke is not None]
        self.l_susp = float(sum(rest_strokes) / len(rest_strokes)) if rest_strokes else 0.10

        # Resolve DoF indices for spin/steer/suspension joints.
        self.spin_dofs: list[int] = []
        self.steer_dofs: list[int] = []
        self.steer_axis_signs: list[int] = []
        self.steer_wheel_idx: list[int] = []
        self.susp_dofs: list[int] = []
        self.susp_uses_control: list[bool] = []   # True -> control_dofs_position; False -> set_dofs_position

        urdf = resolved.urdf
        # Top-level override for suspension visual mode. "auto" keeps the per-joint
        # decision based on URDF <dynamics>; "kinematic"/"control" force one path
        # for every wheel (heavy wheels need "control" — see VehicleConfig docstring).
        mode = getattr(resolved, "visual_susp_mode", "auto")
        for i, w in enumerate(self.wheels):
            if w.spin_joint_name is not None:
                self.spin_dofs.append(int(entity.get_joint(w.spin_joint_name).dofs_idx_local[0]))
            else:
                self.spin_dofs.append(-1)
            if w.steer_joint_name is not None:
                self.steer_dofs.append(int(entity.get_joint(w.steer_joint_name).dofs_idx_local[0]))
                self.steer_axis_signs.append(int(urdf.steer_axis_signs.get(w.steer_joint_name, 1)))
                self.steer_wheel_idx.append(i)
            if w.susp_joint_name is not None:
                self.susp_dofs.append(int(entity.get_joint(w.susp_joint_name).dofs_idx_local[0]))
                if mode == "control":
                    uses_ctrl = True
                elif mode == "kinematic":
                    uses_ctrl = False
                else:   # "auto"
                    uses_ctrl = bool(urdf.susp_has_dynamics.get(w.susp_joint_name, False))
                self.susp_uses_control.append(uses_ctrl)
            else:
                self.susp_dofs.append(-1)
                self.susp_uses_control.append(False)

        # Cull -1 entries for batch DoF writes (joints not present in URDF).
        self._spin_dofs_valid = [d for d in self.spin_dofs if d >= 0]
        self._spin_idx_valid = [i for i, d in enumerate(self.spin_dofs) if d >= 0]
        # Split suspension DoFs by control-vs-set path. Build once via a
        # zipped iterable of (wheel_idx, dof, uses_control) for valid wheels.
        susp_triples = [
            (i, d, self.susp_uses_control[i])
            for i, d in enumerate(self.susp_dofs)
            if d >= 0
        ]
        self._susp_set_dofs = [d for (_, d, u) in susp_triples if not u]
        self._susp_set_idx = [i for (i, _, u) in susp_triples if not u]
        self._susp_ctrl_dofs = [d for (_, d, u) in susp_triples if u]
        self._susp_ctrl_idx = [i for (i, _, u) in susp_triples if u]
        # Suspension-offset clamp. Kept consistent with core._susp_visual_offset
        # so the viewer matches the closed-form wheel_visual_transforms.
        #   resolved.susp_visual_clamp is None -> per-wheel = own rest_stroke
        #     (min 0.02 m); a float -> uniform clamp. Replaces the old fixed 0.19.
        _clamp_override = getattr(resolved, "susp_visual_clamp", None)
        if _clamp_override is not None:
            set_strokes = [float(_clamp_override) for _ in self._susp_set_idx]
        else:
            set_strokes = [max(0.02, float(self.wheels[i].rest_stroke)
                               if self.wheels[i].rest_stroke is not None else 0.10)
                           for i in self._susp_set_idx]
        self._susp_set_clamp = (
            torch.tensor(set_strokes, device=device, dtype=dtype).unsqueeze(0)
            if set_strokes else None)

        # Set high kp/kv on control-path suspension joints (KDU pattern).
        if self._susp_ctrl_dofs:
            import numpy as np
            n_ctrl = len(self._susp_ctrl_dofs)
            kp_arr = np.full(n_ctrl, _SUSP_VIS_KP, dtype=np.float32)
            kv_arr = np.full(n_ctrl, _SUSP_VIS_KV, dtype=np.float32)
            entity.set_dofs_kp(kp=kp_arr, dofs_idx_local=self._susp_ctrl_dofs)
            entity.set_dofs_kv(kv=kv_arr, dofs_idx_local=self._susp_ctrl_dofs)

        # Precompute steer axis signs (was rebuilt as a fresh tensor every step).
        self._steer_signs = (
            torch.tensor(self.steer_axis_signs, device=device, dtype=dtype).unsqueeze(0)
            if self.steer_dofs else None)

        # Batched-write plan. spin + steer + suspension(set-path) all go through
        # set_dofs_position, and EACH solver call triggers a full collider reset
        # + constraint reset + forward-kinematics pass over every link & geom.
        # Issuing 3 separate calls per step therefore pays for 3 FK passes; we
        # combine them into ONE call (one FK pass) — the dominant WheelJointInternalSync
        # cost at n=1. control_dofs_position (PD, heavy wheels) uses a different
        # API and stays separate. The position tensors are concatenated in the
        # SAME order the dof indices are gathered here.
        self._batch_spin = bool(self.spin_enabled and self._spin_dofs_valid)
        batch_dofs: list[int] = []
        if self._batch_spin:
            batch_dofs += list(self._spin_dofs_valid)
        if self.steer_dofs:
            batch_dofs += list(self.steer_dofs)
        if self._susp_set_dofs:
            batch_dofs += list(self._susp_set_dofs)
        self._batch_set_dofs = batch_dofs if batch_dofs else None

        # Visual-state accumulators.
        self.wheel_visual_angle = torch.zeros(n_envs, self.n_wheels, device=device, dtype=dtype)
        # Slew-rate state for the susp visual targets (see _SUSP_VIS_MAX_RATE).
        self._susp_set_prev: Optional[torch.Tensor] = None
        self._susp_ctrl_prev: Optional[torch.Tensor] = None

    def reset_visual_state(self, env_ids=None) -> None:
        """Clear the accumulators after a physics/scene reset. Dropping the
        slew state makes the next susp target SNAP instead of slewing from
        the stale pre-reset value (which, on the control path, would inject
        a real transient joint force right after the reset)."""
        if env_ids is None:
            self.wheel_visual_angle.zero_()
            self._susp_set_prev = None
            self._susp_ctrl_prev = None
        else:
            self.wheel_visual_angle[env_ids] = 0.0
            # Per-env slew snap: forget only those rows.
            if self._susp_set_prev is not None:
                self._susp_set_prev = None
            if self._susp_ctrl_prev is not None:
                self._susp_ctrl_prev = None

    def step(
        self,
        steer_per_wheel: torch.Tensor,    # (n_envs, n_wheels), physics-side angles
        distances: torch.Tensor,          # (n_envs, n_wheels)
        omega: torch.Tensor,              # (n_envs, n_wheels)
        dt: float,
    ) -> None:
        # Build the per-group position tensors, then issue ONE set_dofs_position
        # for all of spin + steer + suspension(set-path). Concatenation order
        # MUST match self._batch_set_dofs (spin, steer, susp — same conditions).
        parts: list[torch.Tensor] = []

        # Spin — integrate angle and wrap to [-pi, pi]. Skipped entirely when
        # disabled (e.g. tank cylindrical wheels: rotation isn't visible).
        if self._batch_spin:
            self.wheel_visual_angle.add_(omega * dt)
            two_pi = 2.0 * math.pi
            self.wheel_visual_angle = (
                (self.wheel_visual_angle + math.pi) % two_pi
            ) - math.pi
            parts.append(self.wheel_visual_angle[:, self._spin_idx_valid])

        # Steer. Physics-side `steer_per_wheel` is ISO 8855 (+ = right turn). The
        # URDF joint angle for a CW (right) visual rotation depends on the axis z
        # sign: visual_cmd = -phys * sign (sign = +1 for (0,0,1), -1 for (0,0,-1)).
        if self.steer_dofs:
            phys = steer_per_wheel[:, self.steer_wheel_idx]
            parts.append(-phys * self._steer_signs)

        # Suspension, set_dofs_position path (HJW): joint_pos = mesh_radius - hit.
        susp_joint_pos = None
        if self._susp_set_dofs:
            d = distances[:, self._susp_set_idx]
            jp = _susp_visual_target(d, self.wheel_mesh_radius, self.l_susp,
                                     self._susp_set_clamp,
                                     self._susp_set_prev, dt)
            self._susp_set_prev = jp
            susp_joint_pos = jp
            parts.append(jp)

        # Single batched write — one solver collider/constraint reset + one FK
        # pass for all visual joints, instead of one per group.
        if self._batch_set_dofs is not None:
            combined = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
            self.entity.set_dofs_position(
                combined, self._batch_set_dofs, zero_velocity=False,
            )

        # Suspension drift suppression: zero the susp joint velocity. skip_forward
        # avoids an extra velocity FK pass (the position write above already ran
        # forward kinematics; the joint is cosmetic).
        if susp_joint_pos is not None:
            self.entity.set_dofs_velocity(
                torch.zeros_like(susp_joint_pos), self._susp_set_dofs,
                skip_forward=True,
            )

        # Suspension joints. control_dofs_position path (heavy wheels):
        # Use the SAME ground-following formula as the set_dofs_position path
        # so the wheel mesh lands on the ground at equilibrium regardless of
        # chassis settle height. The earlier "command = compression" formula
        # was wrong for URDFs whose susp joint origin sits at the chassis z
        # plane (KDU pattern) — `compression` is always >= 0, so the wheel
        # could only ride UP from rest, never extend DOWN to reach the ground.
        if self._susp_ctrl_dofs:
            d = distances[:, self._susp_ctrl_idx]
            # PD path applies REAL joint forces (kp=_SUSP_VIS_KP) — the
            # target is stroke-clamped and slew-rate-limited so an air<->
            # ground jump cannot inject a large impulse into the chassis.
            joint_pos = _susp_visual_target(d, self.wheel_mesh_radius,
                                            self.l_susp, None,
                                            self._susp_ctrl_prev, dt)
            self._susp_ctrl_prev = joint_pos
            self.entity.control_dofs_position(
                joint_pos, dofs_idx_local=self._susp_ctrl_dofs,
            )


class KindVisualBatch:
    """Batched writer for K same-kind ``WheelJointInternalSync`` objects (v1.0.15).

    Each ``WheelJointInternalSync.step`` issues its own ``set_dofs_position`` (plus
    the susp velocity-zero and, for heavy wheels, ``control_dofs_position``) —
    and every solver entry pays a collider/constraint reset + FK pass. K
    same-kind vehicles therefore paid K× that cost per step (the reason the
    old perf advisory said ~0.85 ms/step *per vehicle*). Same-kind entities
    share one dof LAYOUT (identical URDF/joints), so all K writes collapse
    into ONE solver-level ``set_dofs_position`` over the concatenated GLOBAL
    dof indices (``entity._dof_start + local``) — one solver entry, one FK.

    Math is identical to ``WheelJointInternalSync.step``, just computed on
    ``(n_envs, K, n)`` slabs instead of K ``(n_envs, n)`` slices. The spin
    accumulator lives here in batched form. Construction raises if the K
    layouts differ (caller falls back to the per-entity loop).
    """

    def __init__(self, visuals: "list[WheelJointInternalSync]"):
        if not visuals:
            raise ValueError("KindVisualBatch needs at least one WheelJointInternalSync")
        v0 = visuals[0]
        for v in visuals[1:]:
            if (v._batch_set_dofs != v0._batch_set_dofs
                    or v._spin_idx_valid != v0._spin_idx_valid
                    or v.steer_dofs != v0.steer_dofs
                    or v.steer_wheel_idx != v0.steer_wheel_idx
                    or v._susp_set_dofs != v0._susp_set_dofs
                    or v._susp_set_idx != v0._susp_set_idx
                    or v._susp_ctrl_dofs != v0._susp_ctrl_dofs
                    or v._susp_ctrl_idx != v0._susp_ctrl_idx
                    or v.n_wheels != v0.n_wheels):
                raise ValueError("KindVisualBatch requires identical layouts")
        self.proto = v0
        self.K = len(visuals)
        self.n_envs = v0.n_envs
        self.solver = v0.entity._solver

        def _globals(local_dofs):
            return [int(v.entity._dof_start) + int(d)
                    for v in visuals for d in local_dofs]

        # Per-entity blocks, matching combined.reshape(n_envs, K*m) row order.
        self._set_dofs_g = (_globals(v0._batch_set_dofs)
                            if v0._batch_set_dofs else None)
        self._susp_set_g = (_globals(v0._susp_set_dofs)
                            if v0._susp_set_dofs else None)
        self._susp_ctrl_g = (_globals(v0._susp_ctrl_dofs)
                             if v0._susp_ctrl_dofs else None)

        # Batched spin accumulator (replaces the K per-visual accumulators).
        self._angle = torch.zeros(self.n_envs, self.K, v0.n_wheels,
                                  device=v0.device, dtype=v0.dtype)
        # Batched slew-rate state for the susp visual targets.
        self._susp_set_prev = None
        self._susp_ctrl_prev = None

    def reset_visual_state(self, env_ids=None) -> None:
        """Mirror of ``WheelJointInternalSync.reset_visual_state`` for the batched
        writer (spin accumulator + slew snap)."""
        if env_ids is None:
            self._angle.zero_()
        else:
            self._angle[env_ids] = 0.0
        self._susp_set_prev = None
        self._susp_ctrl_prev = None

    def step(self, steer_NK: torch.Tensor, dist_NK: torch.Tensor,
             omega_NK: torch.Tensor, dt: float) -> None:
        """Inputs are ``(n_envs, K, n_wheels)`` slabs (the kind's natural
        layout) — no per-entity slicing."""
        v0 = self.proto
        N, K = self.n_envs, self.K
        parts: list[torch.Tensor] = []

        if v0._batch_spin:
            self._angle.add_(omega_NK * dt)
            two_pi = 2.0 * math.pi
            self._angle = ((self._angle + math.pi) % two_pi) - math.pi
            parts.append(self._angle[:, :, v0._spin_idx_valid])

        if v0.steer_dofs:
            phys = steer_NK[:, :, v0.steer_wheel_idx]
            parts.append(-phys * v0._steer_signs.unsqueeze(0))   # (1,1,m) bcast

        susp_written = False
        if v0._susp_set_dofs:
            d = dist_NK[:, :, v0._susp_set_idx]
            clamp = (v0._susp_set_clamp.unsqueeze(0)
                     if v0._susp_set_clamp is not None else None)
            jp = _susp_visual_target(d, v0.wheel_mesh_radius, v0.l_susp,
                                     clamp, self._susp_set_prev, dt)
            self._susp_set_prev = jp
            parts.append(jp)
            susp_written = True

        if self._set_dofs_g is not None:
            combined = parts[0] if len(parts) == 1 else torch.cat(parts, dim=2)
            # (N, K, m) → (N, K·m): row blocks per entity = global-idx order.
            self.solver.set_dofs_position(
                combined.reshape(N, -1), self._set_dofs_g)

        if susp_written:
            self.solver.set_dofs_velocity(
                torch.zeros(N, len(self._susp_set_g),
                            device=v0.device, dtype=v0.dtype),
                self._susp_set_g, skip_forward=True)

        if self._susp_ctrl_g is not None:
            d = dist_NK[:, :, v0._susp_ctrl_idx]
            jp = _susp_visual_target(d, v0.wheel_mesh_radius, v0.l_susp,
                                     None, self._susp_ctrl_prev, dt)
            self._susp_ctrl_prev = jp
            self.solver.control_dofs_position(
                jp.reshape(N, -1), self._susp_ctrl_g)



# ----------------------------------------------------------------------
# Instanced (solver-free) wheel rendering — v1.1.17
# ----------------------------------------------------------------------

def _pose_mats(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """(N, 3) positions + (N, 4) wxyz quats -> (N, 4, 4) transforms."""
    w = quat_wxyz[:, 0]; x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]; z = quat_wxyz[:, 3]
    T = np.zeros((pos.shape[0], 4, 4), dtype=np.float32)
    T[:, 0, 0] = 1 - 2 * (y * y + z * z)
    T[:, 0, 1] = 2 * (x * y - w * z)
    T[:, 0, 2] = 2 * (x * z + w * y)
    T[:, 1, 0] = 2 * (x * y + w * z)
    T[:, 1, 1] = 1 - 2 * (x * x + z * z)
    T[:, 1, 2] = 2 * (y * z - w * x)
    T[:, 2, 0] = 2 * (x * z - w * y)
    T[:, 2, 1] = 2 * (y * z + w * x)
    T[:, 2, 2] = 1 - 2 * (x * x + y * y)
    T[:, :3, 3] = pos
    T[:, 3, 3] = 1.0
    return T


class InstancedWheelRenderer:
    """Solver-free wheel visuals (v1.1.17) — VehicleScene's default when the
    scene renders (wheel_render_mode="auto", n_envs == 1).

    One instanced pyrender node per (vehicle kind, wheel index); instance
    poses are streamed each step from the closed-form
    ``wheel_visual_transforms`` (steer + suspension + spin baked). The path
    touches the RENDERER only — no joint writes, no solver FK, no PD — so
    wheel visuals cannot perturb physics BY CONSTRUCTION (WheelJointInternalSync,
    the previous mechanism, drives solver joints and measurably can; it
    remains available via ``wheel_render_mode="internal_sync"`` and is still
    used automatically for ``n_envs > 1``).

    The URDF wheel-link visual geoms are hidden by the caller (their
    ``active_envs_idx`` set empty) and re-baked here into per-wheel
    trimeshes (vgeom local offsets applied), so the instanced copies look
    identical to what the URDF wheels would have shown. Nodes are plain
    external nodes (``is_marker=False``): every camera renders them without
    needing ``debug=True``. Node creation is lazy (first ``update()``);
    pose writes hold the viewer lock."""

    def __init__(self, gs_scene: Any, node_swap: bool = False):
        # (node_swap is accepted for call-site compatibility but no longer
        # used: v1.1.20 final streams poses through the engine's own
        # per-frame buffer-update queue — context.jit.update_buffer — the
        # SAME mechanism Genesis uses to stream link-frame instance poses,
        # consumed by BOTH the native viewer and offscreen cameras on the
        # render thread. Earlier attempts: in-place poses writes froze in
        # the viewer (classic renderer uploads the instance buffer only
        # once), and node-recreation caused shadow-state flicker / marker
        # translucency.)
        self._gs_scene = gs_scene
        self._units: list[dict] = []

    # -- harvesting ----------------------------------------------------------

    @staticmethod
    def harvest_wheel_meshes(entity: Any, wheels: Any) -> Optional[list]:
        """Per-wheel trimesh in the WHEEL-LINK frame (local geom offsets baked
        in). Source order per wheel (v1.1.22):

        1. the link's VISUAL geoms — the normal case;
        2. its COLLISION geoms, when the wheel was authored with a collider
           and no visual (``urdf_prep`` promotes those to visuals, so this is
           only a safety net for callers that skipped the prep);
        3. a cylinder synthesized from the wheel's ``radius`` — last resort,
           so a wheel with NO geometry at all still shows up.

        Returns None only if the wheel links cannot be resolved at all, in
        which case the caller falls back to WheelJointInternalSync."""
        import trimesh as _tm
        out = []
        for w in wheels:
            name = getattr(w, "name", None)
            link = None
            if name:
                try:
                    link = entity.get_link(name)
                except Exception:
                    link = None
            if link is None:
                return None
            geoms = list(getattr(link, "vgeoms", None) or [])
            if not geoms:
                geoms = list(getattr(link, "geoms", None) or [])   # colliders
            parts = []
            for g in geoms:
                try:
                    tm = g.get_trimesh().copy()
                except Exception:
                    continue
                lp = np.asarray(g.init_pos, dtype=np.float64).reshape(1, 3)
                lq = np.asarray(g.init_quat, dtype=np.float64).reshape(1, 4)
                tm.apply_transform(_pose_mats(lp, lq)[0].astype(np.float64))
                parts.append(tm)
            if not parts:
                r = float(getattr(w, "radius", None) or 0.35)
                width = max(0.08, 0.4 * r)
                tm = _tm.creation.cylinder(radius=r, height=width, sections=24)
                # cylinder axis is +z; a wheel spins about the chassis +y axis
                tm.apply_transform(_tm.transformations.rotation_matrix(
                    math.pi / 2.0, (1, 0, 0)))
                parts.append(tm)
            out.append(_tm.util.concatenate(parts) if len(parts) > 1 else parts[0])
        return out

    # -- units ---------------------------------------------------------------

    def add_unit(self, meshes: list, provider: Any, K: int) -> None:
        """Register one vehicle kind: ``meshes`` = per-wheel trimeshes (from
        :meth:`harvest_wheel_meshes`), ``provider(frame)`` returning the
        closed-form ``(pos, quat)`` shaped ``(n_envs, K, n, 3/4)`` (kind
        physics) or ``(n_envs, n, 3/4)`` (single VehiclePhysics, K == 1)."""
        self._units.append(dict(meshes=meshes, provider=provider,
                                K=int(K), n=len(meshes), nodes=None))

    # -- per-step update -------------------------------------------------------

    def update(self) -> None:
        """Stream the current closed-form wheel poses into the render nodes.

        Uses the engine's OWN per-frame instance-buffer update queue
        (``context.jit.update_buffer`` — the mechanism Genesis streams
        link-frame instance poses with): set ``primitive.poses`` for
        anything that (re)binds, and queue a model-buffer refresh that the
        render thread flushes on its next pass. Works identically for the
        native viewer and offscreen cameras; no node churn, no marker
        semantics, normal opaque depth-tested rendering."""
        if not self._units:
            return
        vis = getattr(self._gs_scene, "_visualizer", None)
        if vis is None:
            return
        from genesis.ext import pyrender
        ctx = vis.context
        with vis.viewer_lock:
            for u in self._units:
                p, q = u["provider"]("world")
                K, n = u["K"], u["n"]
                pos = p.detach().cpu().numpy().reshape(K, n, 3)
                quat = q.detach().cpu().numpy().reshape(K, n, 4)
                if u["nodes"] is None:
                    nodes = []
                    for i, mesh in enumerate(u["meshes"]):
                        name = f"gv_wheel_{id(u)}_{i}"
                        obj = pyrender.Mesh.from_trimesh(
                            mesh, name=name,
                            poses=_pose_mats(pos[:, i], quat[:, i]),
                            smooth=False, is_marker=False)
                        ctx.add_external_node(obj)
                        nodes.append(ctx.external_nodes[name])   # pyrender Node
                    u["nodes"] = nodes
                else:
                    buf_ids = u.get("buf_ids")
                    if buf_ids is None or any(b < 0 for b in buf_ids):
                        # Buffers bind lazily at the first render; re-lookup
                        # until every node has a valid id, then cache.
                        buf_ids = [ctx._scene.get_buffer_id(nd, "model")
                                   for nd in u["nodes"]]
                        u["buf_ids"] = buf_ids
                    for i, node in enumerate(u["nodes"]):
                        T = _pose_mats(pos[:, i], quat[:, i])
                        node.mesh.primitives[0].poses = T
                        if buf_ids[i] >= 0:
                            ctx.jit.update_buffer(
                                buf_ids[i], T.transpose((0, 2, 1)))


def patch_viewer_atomic_update(viewer) -> bool:
    """Rebind viewer.update so the follow camera, any queued wheel-instance
    buffers (viewer._gv_pre_draw) and the rigid node poses reach the
    renderer inside ONE render-lock hold.

    Genesis (verified against 1.2.0) updates the follow camera OUTSIDE the
    render lock and the node poses INSIDE it. The interactive viewer draws on
    its own thread under that same (re-entrant) lock, so it can slip a frame
    between the two updates and pair a fresh camera with LAST steps body
    pose. At speed that one-step offset flickers at the draw rate: the
    followed vehicle "trembles" fore/aft on screen while the wheels and the
    terrain look steady (each is self-consistent) and the physics is clean.
    The same window let the wheel-instance buffers (streamed by
    VehicleScene.step) race the body pose in the opposite direction.

    Returns False (and changes nothing) if the viewer internals do not match
    the expected layout, e.g. a future Genesis restructuring — callers fall
    back to the non-atomic ordering, which only ever produces the cosmetic
    flicker this patch removes.
    """
    import threading as _threading
    import time as _time
    import types as _types

    import genesis as _gs

    for attr in ("_pyrender_viewer", "lock", "context", "update_following",
                 "_followed_entity", "_last_refresh_time", "_refresh_rate",
                 "_realtime_pacer", "is_alive"):
        if not hasattr(viewer, attr):
            return False
    if not hasattr(viewer._pyrender_viewer, "update_on_sim_step"):
        return False

    def _update(self, auto_refresh=None, force=False):
        if not self.is_alive():
            _gs.raise_exception("Viewer closed.")

        self._pyrender_viewer.update_on_sim_step()

        with self.lock:
            # Everything the next drawn frame reads is committed in ONE hold:
            # wheel-instance buffers, the follow camera, then the node poses.
            pre_draw = getattr(self, "_gv_pre_draw", None)
            if pre_draw is not None:
                pre_draw()
            if self._followed_entity is not None:
                self.update_following()      # re-enters the RLock via set_camera_pose
            self.context.update(force)

            if auto_refresh is None:
                viewer_thread = (self._pyrender_viewer._thread
                                 or _threading.main_thread())
                auto_refresh = viewer_thread == _threading.current_thread()
            if auto_refresh and not self._pyrender_viewer.run_in_thread:
                now = _time.perf_counter()
                if (self._last_refresh_time is None
                        or now - self._last_refresh_time >= 1.0 / self._refresh_rate):
                    self._last_refresh_time = now
                    self._pyrender_viewer.refresh()

        # Real-time pacing stays OUTSIDE the lock (it can sleep most of the
        # frame; holding the render lock through it would starve the drawer).
        realtime_pacer = self._realtime_pacer
        if realtime_pacer is not None:
            realtime_pacer.sleep()

    viewer.update = _types.MethodType(_update, viewer)
    return True
