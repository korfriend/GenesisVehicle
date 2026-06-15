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
        "headless run, set enable_visual_sync=False and read wheel poses from "
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

        # Set high kp/kv on control-path suspension joints (KDU pattern).
        if self._susp_ctrl_dofs:
            import numpy as np
            n_ctrl = len(self._susp_ctrl_dofs)
            kp_arr = np.full(n_ctrl, _SUSP_VIS_KP, dtype=np.float32)
            kv_arr = np.full(n_ctrl, _SUSP_VIS_KV, dtype=np.float32)
            entity.set_dofs_kp(kp=kp_arr, dofs_idx_local=self._susp_ctrl_dofs)
            entity.set_dofs_kv(kv=kv_arr, dofs_idx_local=self._susp_ctrl_dofs)

        # Visual-state accumulators.
        self.wheel_visual_angle = torch.zeros(n_envs, self.n_wheels, device=device, dtype=dtype)

    def step(
        self,
        steer_per_wheel: torch.Tensor,    # (n_envs, n_wheels), physics-side angles
        distances: torch.Tensor,          # (n_envs, n_wheels)
        omega: torch.Tensor,              # (n_envs, n_wheels)
        dt: float,
    ) -> None:
        # Spin joints — skip entirely when disabled (saves a Genesis call per
        # step + a few tensor ops). Useful for cylindrical wheels (e.g. tank
        # sprockets/road wheels) where rotation isn't visible anyway.
        if self.spin_enabled and self._spin_dofs_valid:
            # Integrate spin angle and wrap to [-pi, pi]. In-place to avoid
            # allocating a fresh tensor every step.
            self.wheel_visual_angle.add_(omega * dt)
            two_pi = 2.0 * math.pi
            self.wheel_visual_angle = (
                (self.wheel_visual_angle + math.pi) % two_pi
            ) - math.pi
            spin_cmd = self.wheel_visual_angle[:, self._spin_idx_valid]
            self.entity.set_dofs_position(
                spin_cmd, self._spin_dofs_valid, zero_velocity=False,
            )

        # Steer joints. Physics-side `steer_per_wheel` is ISO 8855 (+ = right
        # turn, computed as wheel fwd tilted from +X toward -Y). The URDF
        # joint angle that produces a CW visual rotation (= right turn from
        # above) depends on the URDF axis z sign:
        #   axis (0, 0,  1): +joint = CCW = LEFT  → visual_cmd = -phys
        #   axis (0, 0, -1): +joint = CW  = RIGHT → visual_cmd = +phys
        # Unified: visual_cmd = -phys * sign  (sign = +1 for (0,0,1), -1 for (0,0,-1))
        if self.steer_dofs:
            cols = self.steer_wheel_idx
            phys = steer_per_wheel[:, cols]
            signs = torch.tensor(
                self.steer_axis_signs, device=self.device, dtype=self.dtype
            ).unsqueeze(0)
            visual_cmd = -phys * signs
            self.entity.set_dofs_position(
                visual_cmd, self.steer_dofs, zero_velocity=False,
            )

        # Suspension joints. set_dofs_position path (HJW): joint_pos = mesh_radius - hit_distance.
        if self._susp_set_dofs:
            d = distances[:, self._susp_set_idx]
            air = (d <= 1e-6) | (d >= 19.9)
            joint_pos = self.wheel_mesh_radius - d
            joint_pos = torch.where(air, torch.full_like(joint_pos, -self.l_susp), joint_pos)
            joint_pos = torch.clamp(joint_pos, -0.19, 0.19)
            self.entity.set_dofs_position(
                joint_pos, self._susp_set_dofs, zero_velocity=False,
            )
            # Drift suppression: zero suspension joint velocity.
            zero_v = torch.zeros_like(joint_pos)
            self.entity.set_dofs_velocity(zero_v, self._susp_set_dofs)

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
