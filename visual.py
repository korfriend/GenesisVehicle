"""Wheel visual-joint sync layer (for the Genesis viewer).

`VisualJointSync` drives the URDF **wheel** spin / steer / suspension joints
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

import torch


# Defaults used when control_dofs_position path is taken (KDU-style).
_SUSP_VIS_KP = 1.0e7
_SUSP_VIS_KV = 1.0e5

# One-time-per-process perf advisory when VisualJointSync is active.
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
        "[genesis_vehicle] PERF: VisualJointSync is ENABLED — it drives the URDF "
        "wheel visual joints through the engine's articulated-body FK every step "
        "(~ms/step overhead; the dominant SDK cost at scale). It is only needed "
        "for the Genesis viewer. For an external renderer (UE / Unity), or any "
        "headless run, set enable_visual_joint_sync=False and read wheel poses from "
        "VehiclePhysics.visual_parts_transforms() / wheel_visual_transforms() "
        "(closed-form, ~µs). Silence with GENESIS_VEHICLE_QUIET=1.",
        file=sys.stderr, flush=True,
    )


class VisualJointSync:
    """Drives a vehicle's URDF WHEEL visual joints (spin, steer, suspension) to
    match physics state, for the Genesis viewer. Wheels only — never the
    chassis. Cosmetic (no force feedback). External renderers should use
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
        # combine them into ONE call (one FK pass) — the dominant VisualJointSync
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
            air = (d <= 1e-6) | (d >= 19.9)
            jp = self.wheel_mesh_radius - d
            jp = torch.where(air, torch.full_like(jp, -self.l_susp), jp)
            if self._susp_set_clamp is not None:
                jp = torch.maximum(
                    -self._susp_set_clamp, torch.minimum(self._susp_set_clamp, jp))
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
            air = (d <= 1e-6) | (d >= 19.9)
            joint_pos = self.wheel_mesh_radius - d
            joint_pos = torch.where(
                air, torch.full_like(joint_pos, -self.l_susp), joint_pos,
            )
            self.entity.control_dofs_position(
                joint_pos, dofs_idx_local=self._susp_ctrl_dofs,
            )
