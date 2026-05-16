"""Visual sync layer.

Drives URDF spin / steer / suspension joints to match the physics state, so the
viewer shows wheels rotating, steered, and ground-following. The forces on the
chassis come from VehiclePhysics's apply_links_external_force; this layer is
purely cosmetic (does NOT double-apply).

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


class VisualSync:
    """Synchronises URDF visual joints (spin, steer, suspension) with physics state."""

    def __init__(
        self,
        entity: Any,
        resolved: Any,         # ResolvedConfig
        n_envs: int,
        device: Any,
        dtype: Any,
        wheel_mesh_radius: Optional[float] = None,
    ):
        self.entity = entity
        self.resolved = resolved
        self.n_envs = n_envs
        self.device = device
        self.dtype = dtype
        self.wheels = resolved.wheels
        self.n_wheels = len(self.wheels)

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
                self.susp_uses_control.append(bool(urdf.susp_has_dynamics.get(w.susp_joint_name, False)))
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
        # Integrate spin angle and wrap to [-pi, pi].
        self.wheel_visual_angle = self.wheel_visual_angle + omega * dt
        two_pi = 2.0 * math.pi
        self.wheel_visual_angle = ((self.wheel_visual_angle + math.pi) % two_pi) - math.pi

        # Spin joints.
        if self._spin_dofs_valid:
            spin_cmd = self.wheel_visual_angle[:, self._spin_idx_valid]
            self.entity.set_dofs_position(
                spin_cmd, self._spin_dofs_valid, zero_velocity=False,
            )

        # Steer joints. URDF axis (0,0,-1) requires negating the visual command
        # so the user-visible rotation matches the physics-side angle.
        if self.steer_dofs:
            cols = self.steer_wheel_idx
            phys = steer_per_wheel[:, cols]
            signs = torch.tensor(
                self.steer_axis_signs, device=self.device, dtype=self.dtype
            ).unsqueeze(0)
            visual_cmd = phys * signs
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

        # Suspension joints. control_dofs_position path (KDU): command = compression.
        if self._susp_ctrl_dofs:
            d = distances[:, self._susp_ctrl_idx]
            # Compute compression per visual wheel; use rest_stroke + radius as REST_D.
            rest_d = torch.tensor(
                [float(self.wheels[i].rest_stroke) + float(self.wheels[i].radius)
                 for i in self._susp_ctrl_idx],
                device=self.device, dtype=self.dtype,
            ).unsqueeze(0)
            compression = torch.clamp(rest_d - d, min=0.0)
            self.entity.control_dofs_position(
                compression, dofs_idx_local=self._susp_ctrl_dofs,
            )
