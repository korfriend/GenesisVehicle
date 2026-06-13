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

from ._version import __version__
from .config import ResolvedConfig, VehicleConfig, resolve
from ._pipeline import compute_wheel_step
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
        global _BANNER_PRINTED
        if not _BANNER_PRINTED:
            import os, sys
            if not os.environ.get("GENESIS_VEHICLE_QUIET"):
                print(f"[genesis_vehicle] v{__version__}",
                      file=sys.stderr, flush=True)
            _BANNER_PRINTED = True
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

    def link_transforms(self, frame: str = "parent", *, envs_idx: Optional[Any] = None):
        """Per-link transforms of this vehicle's entity in ``frame``.

        Thin wrapper over :func:`genesis_vehicle.kinematics.get_link_transforms`
        bound to ``self.entity``. ``frame`` is ``"world"``, ``"base"``, or
        ``"parent"`` (URDF-hierarchy-local; default). Returns a
        ``LinkTransforms`` (link names + parent topology + batched pos/quat).

        Use for telemetry → animation retargeting, sensor/effect attachment, or
        placing ghost copies. See the kinematics module docstring for frames.
        """
        from .kinematics import get_link_transforms
        return get_link_transforms(self.entity, frame, envs_idx=envs_idx)

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
        # BATCHED PER-WHEEL PIPELINE — shared with MultiVehicleKindPhysics
        # via _pipeline.compute_wheel_step (single source of the math).
        # ================================================================
        res = compute_wheel_step(
            pos=pos, quat=quat, vel=vel, ang=ang, distances=distances,
            steer_per_wheel=steer_per_wheel,
            T_drive_pw=T_drive_pw, T_brake_pw=T_brake_pw, omega=self.omega,
            prev_compression=self.prev_compression, prev_init=self._prev_init,
            wheel_meta=wm, resolved=self.resolved,
            wheel_body_b=self._wheel_body_b, up_world=self._up_world,
            post_tire_hooks=self.post_tire_hooks, ctx=ctx,
            dt=DT, B=n_envs, n=n,
        )
        self.prev_compression = res.compression.detach().clone()
        self.omega = res.new_omega
        total_F, total_T = res.total_F, res.total_T

        # Diagnostics (all (n_envs, n)).
        self.last_N = res.N
        self.last_F_long = res.F_long
        self.last_F_lat = res.F_lat
        self.last_compression = res.compression
        self.last_kappa = res.kappa
        self.last_alpha = res.alpha

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
