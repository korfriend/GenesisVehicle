"""VehiclePhysics: orchestrates the 5-step ray-wheel pipeline batched over
n_envs AND over n_wheels. All per-wheel work is a single batched tensor op
(no Python `for i in range(n_wheels)` loop in the hot path) — this matters
on small n_envs where Python + per-op CUDA launch overhead dominates GPU work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import genesis as gs
from genesis.utils.geom import transform_by_quat

from ._version import __version__
from .config import ResolvedConfig, VehicleConfig, resolve
from .dynamics import brake_torque_signed
from .inputs import VehicleInputs, VehicleStepInputs
from .raycast import read_distances
from .urdf import estimate_spin_inertia_from_genesis
from .visual import VisualSync


# Process-level flag so the version banner prints at most once per process,
# even if the user instantiates several VehiclePhysics objects.
_BANNER_PRINTED = False

# Process-level set of (recommended_dt, scene_dt) pairs we've already warned
# about. Keeps the warning to one line per distinct mismatch per process,
# instead of spamming on every VehiclePhysics construction.
_DT_MISMATCH_WARNED: set[tuple[float, float]] = set()


def _resolve_dt_from_scene(scene: Any, recommended_dt: float) -> float:
    """Return ``scene.sim.dt`` (the authoritative simulation step). If the
    scene's dt differs from the preset's ``recommended_dt``, emit a
    one-time-per-process warning so the user knows their physics may be
    less stable than the preset author tested.

    Genesis owns physical time — ``scene.step()`` advances by exactly
    ``scene.sim.dt`` per call. Our hooks (wheel ω forward-Euler, stick-slip
    integrator, ...) run once per outer step and must use the same value.
    By reading ``scene.sim.dt`` directly we make the two impossible to
    desync structurally; ``recommended_dt`` is advisory only.
    """
    try:
        scene_dt = float(scene.sim.dt)
    except (AttributeError, TypeError):
        # scene not built yet / no .sim.dt — fall back to preset's value
        # (best we can do until scene.build is called).
        return float(recommended_dt)
    if abs(scene_dt - recommended_dt) > 1e-9:
        key = (round(recommended_dt, 9), round(scene_dt, 9))
        if key not in _DT_MISMATCH_WARNED:
            _DT_MISMATCH_WARNED.add(key)
            print(
                f"[genesis_vehicle] WARN: scene.sim.dt={scene_dt:g} "
                f"differs from preset's recommended_dt={recommended_dt:g}. "
                f"Using scene.sim.dt (Genesis owns time). If hooks oscillate "
                f"or speed diverges, set SimOptions(dt={recommended_dt:g}, "
                f"...) or pick a different preset."
            )
    return scene_dt


def _print_version_banner(resolved: ResolvedConfig, n_envs: int) -> None:
    global _BANNER_PRINTED
    if _BANNER_PRINTED:
        return
    _BANNER_PRINTED = True
    hook_names = [type(h).__name__ for h in resolved.stability_hooks]
    print(
        f"[genesis_vehicle v{__version__}] Initialized: "
        f"{len(resolved.wheels)} wheels, "
        f"{type(resolved.steering).__name__}, "
        f"{type(resolved.drivetrain).__name__}, "
        f"{type(resolved.coupling).__name__}, "
        f"n_envs={n_envs}, "
        f"hooks={hook_names if hook_names else 'none'}"
    )


@dataclass
class WheelMeta:
    """Cached per-wheel arrays built once at __init__ from ResolvedConfig.
    All tensors are (n_wheels,) and broadcast against (n_envs, n_wheels) state."""
    n_wheels: int
    positions: torch.Tensor          # (n_wheels, 3) chassis-local ray origins
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
    # Tire / hook coefficients (added in v0.5.0 for batched per-wheel ops).
    mu_long: torch.Tensor            # (n_wheels,)
    mu_lat: torch.Tensor             # (n_wheels,)
    rolling_resistance_cr: torch.Tensor   # (n_wheels,)
    pb_x: torch.Tensor               # (n_wheels,)
    pc_x: torch.Tensor               # (n_wheels,)
    pe_x: torch.Tensor               # (n_wheels,)
    pb_y: torch.Tensor               # (n_wheels,)
    pc_y: torch.Tensor               # (n_wheels,)
    pe_y: torch.Tensor               # (n_wheels,)


@dataclass
class PipelineContext:
    """Mutable context passed to stability hooks.

    v0.5.0: now batched over wheels — F_long/F_lat/N/v_long/v_lat are all
    (n_envs, n_wheels) tensors, not (n_envs,) per-wheel scalars. Hook
    implementations have a single `apply_post_tire(ctx)` call per step
    (no `wheel_idx` argument)."""
    # Batched per-wheel state (all (n_envs, n_wheels))
    F_long: Optional[torch.Tensor] = None
    F_lat: Optional[torch.Tensor] = None
    N: Optional[torch.Tensor] = None
    v_long: Optional[torch.Tensor] = None
    v_lat: Optional[torch.Tensor] = None
    omega: Optional[torch.Tensor] = None
    air_mask: Optional[torch.Tensor] = None     # (n_envs, n_wheels) bool
    omega_override: Optional[Any] = None         # _OmegaOverride with (n_envs, n_wheels) bool
    omega_pull_factor: Optional[torch.Tensor] = None    # (n_envs, n_wheels)
    omega_pull_target: Optional[torch.Tensor] = None    # (n_envs, n_wheels)
    # Per-step shared state (n_envs,) or chassis-wide
    moving: Optional[torch.Tensor] = None        # (n_envs,)
    throttle: Optional[torch.Tensor] = None      # (n_envs,)
    brake: Optional[torch.Tensor] = None         # (n_envs,)
    vel: Optional[torch.Tensor] = None           # (n_envs, 3)
    ang: Optional[torch.Tensor] = None           # (n_envs, 3)
    # Simulation step duration in seconds (= cfg.dt). Hooks that integrate
    # state across steps (e.g., position-anchored stick-slip in
    # StaticFrictionLock) need this. Added in v0.5.7.
    dt: float = 0.0
    # WheelMeta with batched per-wheel coefficients (for hooks/tire to read).
    wheel_meta: Any = None


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

        user_explicit_i_wheel = [w.i_wheel is not None for w in config.wheels]

        self.resolved: ResolvedConfig = resolve(config)
        # Single source of truth: Genesis owns physical time. We pull dt
        # from scene.sim.dt; resolved.recommended_dt is advisory only.
        self.dt = _resolve_dt_from_scene(scene, self.resolved.recommended_dt)

        base_name = self.resolved.chassis.base_link_name
        try:
            base_idx = int(entity.base_link_idx)
        except Exception:
            base_link = [l for l in entity.links if l.name == base_name][0]
            base_idx = int(base_link.idx)
        self.base_idx_list = [base_idx]

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
                pass

        self.wheel_meta = self._build_wheel_meta(self.resolved)

        self.pre_loop_hooks = [h for h in self.resolved.stability_hooks if "PRE_LOOP" in h.slots]
        self.post_tire_hooks = [h for h in self.resolved.stability_hooks if "POST_TIRE" in h.slots]

        n_wheels = self.wheel_meta.n_wheels
        self.omega = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.prev_compression = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self._prev_init = False

        self.last_distances = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_compression = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_N = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_F_long = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_F_lat = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_T_drive = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_T_brake = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_kappa = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_alpha = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)

        self.visual: Optional[VisualSync] = None
        if self.resolved.enable_visual_sync:
            self.visual = VisualSync(
                entity=entity, resolved=self.resolved,
                n_envs=n_envs, device=self.dev, dtype=self.fdt,
            )

        self._up_world = torch.tensor(
            [0.0, 0.0, 1.0], device=self.dev, dtype=self.fdt,
        ).unsqueeze(0).expand(n_envs, 3).contiguous()
        # Pre-built (n_envs, n_wheels, 3) wheel-body broadcast.
        self._wheel_body_b = self.wheel_meta.positions.unsqueeze(0).expand(
            n_envs, n_wheels, 3
        ).contiguous()

        _print_version_banner(self.resolved, n_envs)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            self.omega.zero_()
            self.prev_compression.zero_()
            self._prev_init = False
            self.last_distances.zero_(); self.last_compression.zero_()
            self.last_N.zero_(); self.last_F_long.zero_(); self.last_F_lat.zero_()
            self.last_T_drive.zero_(); self.last_T_brake.zero_()
            self.last_kappa.zero_(); self.last_alpha.zero_()
            if self.visual is not None:
                self.visual.wheel_visual_angle.zero_()
            return
        idx = env_ids
        if idx.dtype == torch.bool:
            idx = torch.nonzero(idx, as_tuple=False).flatten()
        self.omega[idx] = 0.0
        self.prev_compression[idx] = 0.0
        self.last_distances[idx] = 0.0; self.last_compression[idx] = 0.0
        self.last_N[idx] = 0.0; self.last_F_long[idx] = 0.0; self.last_F_lat[idx] = 0.0
        self.last_T_drive[idx] = 0.0; self.last_T_brake[idx] = 0.0
        self.last_kappa[idx] = 0.0; self.last_alpha[idx] = 0.0
        if self.visual is not None:
            self.visual.wheel_visual_angle[idx] = 0.0

    def step(self, inputs: VehicleStepInputs) -> None:
        """Vectorized 5-step pipeline. Per-wheel work is a SINGLE batched
        tensor op set (no Python wheel loop)."""
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
        wm = self.wheel_meta

        # [PRE-LOOP] strategies — same as before, one call each
        steer_per_wheel = self.resolved.steering.per_wheel_steer(
            inputs, n_envs, wm, dev, fdt,
        )                                                # (n_envs, n)
        T_drive_pw, T_brake_pw = self.resolved.drivetrain.distribute_torque(
            inputs, self.omega, wm, dev, fdt,
        )                                                # both (n_envs, n)
        self.last_T_drive = T_drive_pw
        self.last_T_brake = T_brake_pw

        throttle = self._scalar_to_envs(getattr(inputs, "throttle", 0.0))
        brake = self._scalar_to_envs(getattr(inputs, "brake", 0.0))

        ctx = PipelineContext(throttle=throttle, brake=brake, wheel_meta=wm)

        # [RAYCAST]
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
            pos = pos.unsqueeze(0); quat = quat.unsqueeze(0)
            vel = vel.unsqueeze(0); ang = ang.unsqueeze(0)

        ctx.vel = vel
        ctx.ang = ang
        for hook in self.pre_loop_hooks:
            hook.apply_pre_loop(ctx)

        # ================================================================
        # BATCHED PER-WHEEL PIPELINE — no Python loop over wheels
        # All tensors are (n_envs, n_wheels) or (n_envs, n_wheels, 3).
        # ================================================================

        # World-space wheel positions: transform (n_envs * n, 3) at once.
        quat_b_flat = quat.unsqueeze(1).expand(n_envs, n, 4).reshape(n_envs * n, 4)
        wheel_body_flat = self._wheel_body_b.reshape(n_envs * n, 3)
        wheel_world = (transform_by_quat(wheel_body_flat, quat_b_flat)
                       .reshape(n_envs, n, 3) + pos.unsqueeze(1))   # (n_envs, n, 3)

        # (A) Compression / asymmetric damper / N.
        compression = torch.clamp(wm.rest_d.unsqueeze(0) - distances, min=0.0)
        air_mask = compression <= 0
        if self._prev_init:
            raw_rate = (compression - self.prev_compression) / DT
            rc = wm.comp_rate_clamp.unsqueeze(0)
            comp_rate = torch.clamp(raw_rate, -rc, rc)
        else:
            comp_rate = torch.zeros_like(compression)
        self.prev_compression = compression.detach().clone()

        c_damp = torch.where(
            comp_rate > 0.0,
            wm.c_compression.unsqueeze(0).expand_as(comp_rate),
            wm.c_extension.unsqueeze(0).expand_as(comp_rate),
        )
        N = wm.k_susp.unsqueeze(0) * compression + c_damp * comp_rate
        N = torch.clamp(N, min=0.0)
        N = torch.where(air_mask, torch.zeros_like(N), N)

        # (B) Wheel-frame fwd/lat (ISO 8855: +steer → fwd rotates +X toward -Y).
        cs = torch.cos(steer_per_wheel)
        ss = torch.sin(steer_per_wheel)
        zer = torch.zeros_like(cs)
        wheel_fwd_local = torch.stack([cs, -ss, zer], dim=-1)   # (n_envs, n, 3)
        wheel_lat_local = torch.stack([ss,  cs, zer], dim=-1)
        wheel_fwd_world = transform_by_quat(
            wheel_fwd_local.reshape(n_envs * n, 3), quat_b_flat
        ).reshape(n_envs, n, 3)
        wheel_lat_world = transform_by_quat(
            wheel_lat_local.reshape(n_envs * n, 3), quat_b_flat
        ).reshape(n_envs, n, 3)

        r_vec = wheel_world - pos.unsqueeze(1)                       # (n_envs, n, 3)
        ang_b = ang.unsqueeze(1).expand(n_envs, n, 3)
        v_hit = vel.unsqueeze(1) + torch.cross(ang_b, r_vec, dim=-1)  # (n_envs, n, 3)
        v_long = (v_hit * wheel_fwd_world).sum(dim=-1)               # (n_envs, n)
        v_lat = (v_hit * wheel_lat_world).sum(dim=-1)
        v_roll = wm.radius.unsqueeze(0) * self.omega                 # (n_envs, n)

        # (C) Tire force — single batched call.
        F_long, F_lat, kappa, alpha = self.resolved.tire(
            v_long, v_lat, v_roll, N, wm,
        )

        # POST_TIRE stability hooks (batched — one call per hook).
        ctx.F_long = F_long; ctx.F_lat = F_lat; ctx.N = N
        ctx.v_long = v_long; ctx.v_lat = v_lat
        ctx.omega = self.omega; ctx.air_mask = air_mask
        ctx.omega_override = None
        ctx.omega_pull_factor = None; ctx.omega_pull_target = None
        ctx.dt = float(DT)
        for hook in self.post_tire_hooks:
            hook.apply_post_tire(ctx)
        F_long, F_lat = ctx.F_long, ctx.F_lat

        # (D) Omega update.
        T_brake_eff = brake_torque_signed(T_brake_pw, self.omega)
        i_w = wm.i_wheel.unsqueeze(0)
        radius_b = wm.radius.unsqueeze(0)
        T_friction = radius_b * F_long
        domega = (T_drive_pw - T_brake_eff - T_friction) / i_w
        new_omega = self.omega + domega * DT
        domega_air = (T_drive_pw - T_brake_eff) / i_w
        new_omega_air = self.omega + domega_air * DT
        new_omega = torch.where(air_mask, new_omega_air, new_omega)
        omega_max = float(self.resolved.chassis.omega_max)
        new_omega = torch.clamp(new_omega, -omega_max, omega_max)

        if ctx.omega_pull_factor is not None and ctx.omega_pull_target is not None:
            pull = ctx.omega_pull_factor
            new_omega = new_omega * (1.0 - pull) + ctx.omega_pull_target * pull

        if ctx.omega_override is not None:
            new_omega = torch.where(
                ctx.omega_override.active, torch.zeros_like(new_omega), new_omega,
            )

        self.omega = new_omega

        # (E) Force accumulation (batched).
        F_world = (
            N.unsqueeze(-1) * self._up_world.unsqueeze(1)
            + F_long.unsqueeze(-1) * wheel_fwd_world
            + F_lat.unsqueeze(-1) * wheel_lat_world
        )                                                   # (n_envs, n, 3)
        torque = torch.cross(r_vec, F_world, dim=-1)
        total_F = F_world.sum(dim=1)                        # (n_envs, 3)
        total_T = torque.sum(dim=1)

        # Diagnostics (all already (n_envs, n)).
        self.last_N = N
        self.last_F_long = F_long
        self.last_F_lat = F_lat
        self.last_compression = compression
        self.last_kappa = kappa
        self.last_alpha = alpha

        # [COUPLING]
        self.omega = self.resolved.coupling.apply(self.omega, wm)

        # [APPLY]
        self.solver.apply_links_external_force(total_F.unsqueeze(1), self.base_idx_list)
        self.solver.apply_links_external_torque(total_T.unsqueeze(1), self.base_idx_list)
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
        d, f = self.dev, self.fdt
        positions = torch.tensor([w.position for w in wheels], device=d, dtype=f)
        side_L = torch.tensor([w.side == "L" for w in wheels], device=d, dtype=torch.bool)
        side_R = torch.tensor([w.side == "R" for w in wheels], device=d, dtype=torch.bool)
        left_idx = torch.nonzero(side_L, as_tuple=False).flatten().to(dtype=torch.long)
        right_idx = torch.nonzero(side_R, as_tuple=False).flatten().to(dtype=torch.long)
        axle_index = torch.tensor(
            [int(w.axle_index) if w.axle_index is not None else 0 for w in wheels],
            device=d, dtype=torch.long,
        )

        def _t(attr: str) -> torch.Tensor:
            return torch.tensor([float(getattr(w, attr)) for w in wheels], device=d, dtype=f)

        radius = _t("radius")
        i_wheel = _t("i_wheel")
        rest_d = radius + _t("rest_stroke")
        return WheelMeta(
            n_wheels=n,
            positions=positions,
            side_mask_L=side_L, side_mask_R=side_R,
            left_idx=left_idx, right_idx=right_idx,
            axle_index=axle_index,
            radius=radius, i_wheel=i_wheel, rest_d=rest_d,
            k_susp=_t("k_susp"),
            c_compression=_t("c_compression"),
            c_extension=_t("c_extension"),
            comp_rate_clamp=_t("comp_rate_clamp"),
            mu_long=_t("mu_long"), mu_lat=_t("mu_lat"),
            rolling_resistance_cr=_t("rolling_resistance_cr"),
            pb_x=_t("pb_x"), pc_x=_t("pc_x"), pe_x=_t("pe_x"),
            pb_y=_t("pb_y"), pc_y=_t("pc_y"), pe_y=_t("pe_y"),
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
        if spin_joint_name is None:
            return None
        try:
            joint = entity.get_joint(spin_joint_name)
        except Exception:
            return None
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
