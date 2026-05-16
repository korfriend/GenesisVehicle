"""VehiclePhysics: orchestrates the 5-step ray-wheel pipeline batched over n_envs.

Lifted from HJW/car_raywheel.py with strategy hooks plugged in at the right
seams (steering, drivetrain, tire, coupling, stability). All runtime tensors
are (n_envs, n_wheels) or (n_envs, 3/4); single-env case is just a special
case of n_envs=1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import genesis as gs
from genesis.utils.geom import transform_by_quat

from .config import ResolvedConfig, VehicleConfig, resolve
from .dynamics import brake_torque_signed, suspension_normal_force
from .inputs import VehicleInputs, VehicleStepInputs
from .raycast import read_distances
from .urdf import estimate_spin_inertia_from_genesis
from .visual import VisualSync


@dataclass
class WheelMeta:
    """Cached per-wheel arrays built once at __init__ from ResolvedConfig."""
    n_wheels: int
    positions: torch.Tensor          # (n_wheels, 3) chassis-local
    side_mask_L: torch.Tensor        # (n_wheels,) bool
    side_mask_R: torch.Tensor        # (n_wheels,) bool
    left_idx: torch.Tensor           # (n_L,) long
    right_idx: torch.Tensor          # (n_R,) long
    axle_index: torch.Tensor         # (n_wheels,) long
    radius: torch.Tensor             # (n_wheels,)
    i_wheel: torch.Tensor            # (n_wheels,)
    rest_d: torch.Tensor             # (n_wheels,) = radius + rest_stroke
    k_susp: torch.Tensor             # (n_wheels,)
    c_compression: torch.Tensor      # (n_wheels,)
    c_extension: torch.Tensor        # (n_wheels,)
    comp_rate_clamp: torch.Tensor    # (n_wheels,)


@dataclass
class PipelineContext:
    """Mutable context passed to stability hooks. Filled by core.py before each
    pre_loop / per-wheel iteration."""
    # Per-wheel scalars (current wheel only, all (n_envs,))
    F_long: Optional[torch.Tensor] = None
    F_lat: Optional[torch.Tensor] = None
    N: Optional[torch.Tensor] = None
    v_long: Optional[torch.Tensor] = None
    v_lat: Optional[torch.Tensor] = None
    omega_i: Optional[torch.Tensor] = None
    air_mask: Optional[torch.Tensor] = None
    omega_override: Optional[Any] = None    # _OmegaOverride or None
    # Optional pull set by LowSpeedRegularizer
    omega_pull_factor: Optional[torch.Tensor] = None
    omega_pull_target: Optional[torch.Tensor] = None
    # Per-step shared state
    moving: Optional[torch.Tensor] = None
    throttle: Optional[torch.Tensor] = None
    brake: Optional[torch.Tensor] = None
    vel: Optional[torch.Tensor] = None
    ang: Optional[torch.Tensor] = None
    wheel_params: Any = None                 # list[WheelConfig]


class VehiclePhysics:
    """Top-level vehicle physics driver. One instance per vehicle entity."""

    def __init__(
        self,
        scene: Any,
        entity: Any,
        sensor: Any,
        config: VehicleConfig,
        n_envs: int = 1,
    ):
        assert n_envs >= 1
        self.scene = scene
        self.entity = entity
        self.sensor = sensor
        self.solver = scene.sim.rigid_solver
        self.n_envs = n_envs
        self.dev = gs.device
        self.fdt = gs.tc_float

        # Snapshot which wheels have an explicit i_wheel BEFORE resolve() fills
        # URDF defaults. Per the SDK contract: WheelConfig.i_wheel (when set
        # by the user or populated from URDF) is the authoritative value. The
        # Genesis-runtime refinement below only fires for wheels where neither
        # the user nor the URDF supplied a value (i.e. resolve fell back to
        # DEFAULT_I_WHEEL).
        user_explicit_i_wheel = [w.i_wheel is not None for w in config.wheels]

        self.resolved: ResolvedConfig = resolve(config)
        self.dt = float(self.resolved.dt)

        # base_link index for force/torque application.
        base_name = self.resolved.chassis.base_link_name
        try:
            base_idx = int(entity.base_link_idx)
        except Exception:
            base_link = [l for l in entity.links if l.name == base_name][0]
            base_idx = int(base_link.idx)
        self.base_idx_list = [base_idx]

        # Genesis-runtime spin-inertia estimate as a LAST-RESORT fallback.
        # Skipped for wheels whose i_wheel was explicit (user or URDF).
        for i, w in enumerate(self.resolved.wheels):
            if user_explicit_i_wheel[i]:
                continue
            spin_link_name = self._spin_child_link_name(entity, w.spin_joint_name)
            if spin_link_name is None:
                continue
            try:
                moi = estimate_spin_inertia_from_genesis(entity, spin_link_name)
                if moi > 0.0:
                    w.i_wheel = moi
            except Exception:
                # Keep the value resolve() already filled (URDF or DEFAULT_I_WHEEL).
                pass

        # Build WheelMeta.
        self.wheel_meta = self._build_wheel_meta(self.resolved)

        # Stability hook routing.
        self.pre_loop_hooks = [h for h in self.resolved.stability_hooks if "PRE_LOOP" in h.slots]
        self.post_tire_hooks = [h for h in self.resolved.stability_hooks if "POST_TIRE" in h.slots]

        # State buffers.
        n_wheels = self.wheel_meta.n_wheels
        self.omega = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.prev_compression = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self._prev_init = False

        # Diagnostics.
        self.last_distances = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_compression = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_N = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_F_long = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_F_lat = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_T_drive = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_T_brake = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_kappa = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_alpha = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)

        # Visual layer.
        self.visual: Optional[VisualSync] = None
        if self.resolved.enable_visual_sync:
            self.visual = VisualSync(
                entity=entity,
                resolved=self.resolved,
                n_envs=n_envs,
                device=self.dev,
                dtype=self.fdt,
            )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Reset omega / prev_compression / visual angles. None = full reset;
        otherwise only the rows in env_ids."""
        if env_ids is None:
            self.omega.zero_()
            self.prev_compression.zero_()
            self._prev_init = False
            self.last_distances.zero_()
            self.last_compression.zero_()
            self.last_N.zero_()
            self.last_F_long.zero_()
            self.last_F_lat.zero_()
            self.last_T_drive.zero_()
            self.last_T_brake.zero_()
            self.last_kappa.zero_()
            self.last_alpha.zero_()
            if self.visual is not None:
                self.visual.wheel_visual_angle.zero_()
            return
        idx = env_ids
        if idx.dtype == torch.bool:
            idx = torch.nonzero(idx, as_tuple=False).flatten()
        self.omega[idx] = 0.0
        self.prev_compression[idx] = 0.0
        self.last_distances[idx] = 0.0
        self.last_compression[idx] = 0.0
        self.last_N[idx] = 0.0
        self.last_F_long[idx] = 0.0
        self.last_F_lat[idx] = 0.0
        self.last_T_drive[idx] = 0.0
        self.last_T_brake[idx] = 0.0
        self.last_kappa[idx] = 0.0
        self.last_alpha[idx] = 0.0
        if self.visual is not None:
            self.visual.wheel_visual_angle[idx] = 0.0

    def step(self, inputs: VehicleStepInputs) -> None:
        """Advance the vehicle one physics step.

        Accepts either the strategy's expected input type or a unified
        VehicleInputs (which is auto-converted via from_unified)."""
        steering = self.resolved.steering
        if not isinstance(inputs, steering.InputType):
            if isinstance(inputs, VehicleInputs):
                inputs = steering.InputType.from_unified(inputs)
            else:
                raise TypeError(
                    f"step() expected {steering.InputType.__name__} or VehicleInputs, "
                    f"got {type(inputs).__name__}"
                )

        n_envs = self.n_envs
        n = self.wheel_meta.n_wheels
        dev, fdt = self.dev, self.fdt
        DT = self.dt

        # [PRE-LOOP] strategies
        steer_per_wheel = self.resolved.steering.per_wheel_steer(
            inputs, n_envs, self.wheel_meta, dev, fdt,
        )                                                # (n_envs, n)
        T_drive_pw, T_brake_pw = self.resolved.drivetrain.distribute_torque(
            inputs, self.omega, self.wheel_meta, dev, fdt,
        )                                                # both (n_envs, n)
        self.last_T_drive = T_drive_pw
        self.last_T_brake = T_brake_pw

        # Throttle/brake broadcast for stability hooks.
        throttle = self._scalar_to_envs(getattr(inputs, "throttle", 0.0))
        brake = self._scalar_to_envs(getattr(inputs, "brake", 0.0))

        ctx = PipelineContext(
            throttle=throttle,
            brake=brake,
            wheel_params=self.resolved.wheels,
        )

        # [RAYCAST] + first-step protection
        distances = read_distances(self.sensor, n_envs)
        self.last_distances = distances.detach().clone()
        if not self._prev_init and torch.all(distances < 1e-6):
            self._prev_init = True
            return

        # [CHASSIS STATE]
        pos = self.entity.get_pos()
        quat = self.entity.get_quat()
        vel = self.entity.get_vel()
        ang = self.entity.get_ang()
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)
            quat = quat.unsqueeze(0)
            vel = vel.unsqueeze(0)
            ang = ang.unsqueeze(0)

        ctx.vel = vel
        ctx.ang = ang
        for hook in self.pre_loop_hooks:
            hook.apply_pre_loop(ctx)

        up_world = torch.tensor(
            [0.0, 0.0, 1.0], device=dev, dtype=fdt,
        ).unsqueeze(0).expand(n_envs, 3).contiguous()

        total_F = torch.zeros(n_envs, 3, device=dev, dtype=fdt)
        total_T = torch.zeros(n_envs, 3, device=dev, dtype=fdt)

        wheel_body = self.wheel_meta.positions       # (n, 3)
        radius = self.wheel_meta.radius              # (n,)
        i_wheel = self.wheel_meta.i_wheel            # (n,)
        rest_d = self.wheel_meta.rest_d              # (n,)
        k_susp = self.wheel_meta.k_susp              # (n,)
        c_comp = self.wheel_meta.c_compression       # (n,)
        c_ext = self.wheel_meta.c_extension          # (n,)
        rate_clamp = self.wheel_meta.comp_rate_clamp # (n,)
        omega_max = float(self.resolved.chassis.omega_max)

        for i in range(n):
            # World-space wheel position.
            wb_i = wheel_body[i].unsqueeze(0).expand(n_envs, 3).contiguous()
            wheel_world = pos + transform_by_quat(wb_i, quat)        # (n_envs, 3)

            d = distances[:, i]                                       # (n_envs,)
            compression = torch.clamp(rest_d[i] - d, min=0.0)         # (n_envs,)
            air_mask = compression <= 0

            # (A) Suspension N — asymmetric damper, non-negative clamp, air-mask zero.
            if self._prev_init:
                raw_rate = (compression - self.prev_compression[:, i]) / DT
                comp_rate = torch.clamp(raw_rate, -float(rate_clamp[i]), float(rate_clamp[i]))
            else:
                comp_rate = torch.zeros_like(compression)
            self.prev_compression[:, i] = compression
            N = suspension_normal_force(
                compression, comp_rate,
                k_susp[i], c_comp[i], c_ext[i],
                air_mask,
            )

            # (B) Wheel-frame fwd/lat. ISO 8855 convention: positive steer
            # rotates fwd from +X toward -Y (right turn). NOTE: this is the
            # OPPOSITE sign of HJW, which pre-negates steer to compensate for
            # the URDF axis flip; the SDK does NOT pre-negate here.
            cs = torch.cos(steer_per_wheel[:, i])
            ss = torch.sin(steer_per_wheel[:, i])
            zer = torch.zeros_like(cs)
            wheel_fwd_local = torch.stack([cs, -ss, zer], dim=-1)
            wheel_lat_local = torch.stack([ss, cs, zer], dim=-1)
            wheel_fwd_world = transform_by_quat(wheel_fwd_local, quat)
            wheel_lat_world = transform_by_quat(wheel_lat_local, quat)

            r_vec = wheel_world - pos                                 # (n_envs, 3)
            v_hit = vel + torch.cross(ang, r_vec, dim=-1)
            v_long = (v_hit * wheel_fwd_world).sum(dim=-1)
            v_lat = (v_hit * wheel_lat_world).sum(dim=-1)
            v_roll = float(radius[i]) * self.omega[:, i]

            # (C) Tire force.
            F_long, F_lat, kappa, alpha = self.resolved.tire(
                v_long, v_lat, v_roll, N, self.resolved.wheels[i],
            )

            # POST_TIRE stability hooks.
            ctx.F_long = F_long
            ctx.F_lat = F_lat
            ctx.N = N
            ctx.v_long = v_long
            ctx.v_lat = v_lat
            ctx.omega_i = self.omega[:, i]
            ctx.air_mask = air_mask
            ctx.omega_override = None
            ctx.omega_pull_factor = None
            ctx.omega_pull_target = None
            for hook in self.post_tire_hooks:
                hook.apply_post_tire(ctx, i)
            F_long, F_lat = ctx.F_long, ctx.F_lat

            # (D) Wheel omega update. brake_torque_signed converts the positive
            # brake magnitude into a signed torque opposing wheel rotation, so
            # `- T_brake_eff` always decelerates the wheel (see API.md S7).
            T_brake_eff = brake_torque_signed(T_brake_pw[:, i], self.omega[:, i])
            T_friction = float(radius[i]) * F_long
            domega = (T_drive_pw[:, i] - T_brake_eff - T_friction) / float(i_wheel[i])
            new_omega = self.omega[:, i] + domega * DT
            # Air mask: friction is zero (no ground), only drive/brake apply.
            domega_air = (T_drive_pw[:, i] - T_brake_eff) / float(i_wheel[i])
            new_omega_air = self.omega[:, i] + domega_air * DT
            new_omega = torch.where(air_mask, new_omega_air, new_omega)
            new_omega = torch.clamp(new_omega, -omega_max, omega_max)

            # Optional omega pull from LowSpeedRegularizer.
            if ctx.omega_pull_factor is not None and ctx.omega_pull_target is not None:
                pull = ctx.omega_pull_factor
                target = ctx.omega_pull_target
                new_omega = new_omega * (1.0 - pull) + target * pull

            # Optional static lock override.
            if ctx.omega_override is not None:
                new_omega = torch.where(
                    ctx.omega_override.active,
                    torch.zeros_like(new_omega),
                    new_omega,
                )

            self.omega[:, i] = new_omega

            # (E) Accumulate chassis force/torque.
            F_world = (
                N.unsqueeze(-1) * up_world
                + F_long.unsqueeze(-1) * wheel_fwd_world
                + F_lat.unsqueeze(-1) * wheel_lat_world
            )
            torque = torch.cross(r_vec, F_world, dim=-1)
            total_F = total_F + F_world
            total_T = total_T + torque

            # Diagnostics.
            self.last_N[:, i] = N
            self.last_F_long[:, i] = F_long
            self.last_F_lat[:, i] = F_lat
            self.last_compression[:, i] = compression
            self.last_kappa[:, i] = kappa
            self.last_alpha[:, i] = alpha

        # [COUPLING]
        self.omega = self.resolved.coupling.apply(self.omega, self.wheel_meta)

        # [APPLY] chassis force/torque.
        F_arr = total_F.unsqueeze(1)
        T_arr = total_T.unsqueeze(1)
        self.solver.apply_links_external_force(F_arr, self.base_idx_list)
        self.solver.apply_links_external_torque(T_arr, self.base_idx_list)
        self._prev_init = True

        # [VISUAL]
        if self.visual is not None:
            self.visual.step(steer_per_wheel, distances, self.omega, DT)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _build_wheel_meta(self, resolved: ResolvedConfig) -> WheelMeta:
        wheels = resolved.wheels
        n = len(wheels)
        positions = torch.tensor(
            [w.position for w in wheels], device=self.dev, dtype=self.fdt,
        )
        side_L = torch.tensor([w.side == "L" for w in wheels], device=self.dev, dtype=torch.bool)
        side_R = torch.tensor([w.side == "R" for w in wheels], device=self.dev, dtype=torch.bool)
        left_idx = torch.nonzero(side_L, as_tuple=False).flatten().to(dtype=torch.long)
        right_idx = torch.nonzero(side_R, as_tuple=False).flatten().to(dtype=torch.long)
        axle_index = torch.tensor(
            [int(w.axle_index) if w.axle_index is not None else 0 for w in wheels],
            device=self.dev, dtype=torch.long,
        )
        radius = torch.tensor([float(w.radius) for w in wheels], device=self.dev, dtype=self.fdt)
        i_wheel = torch.tensor([float(w.i_wheel) for w in wheels], device=self.dev, dtype=self.fdt)
        rest_stroke = torch.tensor(
            [float(w.rest_stroke) for w in wheels], device=self.dev, dtype=self.fdt,
        )
        rest_d = radius + rest_stroke
        k_susp = torch.tensor([float(w.k_susp) for w in wheels], device=self.dev, dtype=self.fdt)
        c_comp = torch.tensor(
            [float(w.c_compression) for w in wheels], device=self.dev, dtype=self.fdt,
        )
        c_ext = torch.tensor(
            [float(w.c_extension) for w in wheels], device=self.dev, dtype=self.fdt,
        )
        rate_clamp = torch.tensor(
            [float(w.comp_rate_clamp) for w in wheels], device=self.dev, dtype=self.fdt,
        )
        return WheelMeta(
            n_wheels=n,
            positions=positions,
            side_mask_L=side_L,
            side_mask_R=side_R,
            left_idx=left_idx,
            right_idx=right_idx,
            axle_index=axle_index,
            radius=radius,
            i_wheel=i_wheel,
            rest_d=rest_d,
            k_susp=k_susp,
            c_compression=c_comp,
            c_extension=c_ext,
            comp_rate_clamp=rate_clamp,
        )

    def _scalar_to_envs(self, x: Any) -> torch.Tensor:
        if torch.is_tensor(x):
            t = x.to(device=self.dev, dtype=self.fdt)
            if t.dim() == 0:
                t = t.expand(self.n_envs)
            return t.contiguous()
        return torch.full((self.n_envs,), float(x), device=self.dev, dtype=self.fdt)

    @staticmethod
    def _spin_child_link_name(entity: Any, spin_joint_name: Optional[str]) -> Optional[str]:
        """Look up the wheel link name from the spin joint via Genesis entity."""
        if spin_joint_name is None:
            return None
        try:
            joint = entity.get_joint(spin_joint_name)
        except Exception:
            return None
        # Genesis Joint exposes its child link via a few possible attributes.
        for attr in ("child_link", "child"):
            obj = getattr(joint, attr, None)
            if obj is None:
                continue
            name = getattr(obj, "name", None)
            if isinstance(name, str):
                return name
            if isinstance(obj, str):
                return obj
        return None
