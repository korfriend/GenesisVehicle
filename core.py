"""VehiclePhysics: orchestrates the 5-step ray-wheel pipeline batched over
n_envs AND over n_wheels. All per-wheel work is a single batched tensor op
(no Python `for i in range(n_wheels)` loop in the hot path) — this matters
on small n_envs where Python + per-op CUDA launch overhead dominates GPU work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import math
import os

import torch
import genesis as gs
from genesis.utils.geom import transform_by_quat

from ._version import __version__
from .config import ResolvedConfig, VehicleConfig, resolve
from ._gs_compat import apply_links_wrench
from ._hotset import prime_derived
from ._pipeline import compute_wheel_step, pw, pw3
from .inputs import VehicleInputs, VehicleStepInputs
from .raycast import (
    check_miss_supported, is_ray_hit_corrected, ray_miss_value, read_distances,
)
from .urdf import estimate_spin_inertia_from_genesis
from .visual import WheelJointInternalSync


# Process-level flag so the version banner prints at most once per process,
# even if the user instantiates several VehiclePhysics objects.
_BANNER_PRINTED = False

# One-time-per-process warning guard for reading visual link transforms while
# WheelJointInternalSync is disabled (the result is the rest pose — see link_transforms).
_VISUAL_OFF_WARNED = False


def _quat_axis_angle(axis: str, theta: torch.Tensor) -> torch.Tensor:
    """Batched quaternion (w, x, y, z) for a rotation of ``theta`` about a
    principal axis ('x'|'y'|'z'). ``theta`` shape ``(...,)`` → out ``(..., 4)``."""
    half = 0.5 * theta
    c = torch.cos(half)
    s = torch.sin(half)
    z = torch.zeros_like(theta)
    if axis == "x":
        return torch.stack([c, s, z, z], dim=-1)
    if axis == "y":
        return torch.stack([c, z, s, z], dim=-1)
    return torch.stack([c, z, z, s], dim=-1)   # "z"


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two (w, x, y, z) quaternion tensors, broadcasting on
    all but the last dim."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _susp_visual_offset(distance: torch.Tensor, mesh_radius: float,
                        l_susp: float, clamp=0.19, miss=None) -> torch.Tensor:
    """Vertical wheel-mesh offset (chassis +z) from ray hit distance.

    Mirror of the suspension command in ``visual.WheelJointInternalSync.step`` (joint_pos =
    mesh_radius − distance; air → −l_susp; clamp). Kept here so the closed-form
    ``wheel_visual_transforms`` matches what WheelJointInternalSync drives into the URDF
    joints. If you change one, change the other (the equivalence is unit-checked
    against ``entity.get_link`` in tests/smoke).

    ``clamp`` bounds the offset to ±clamp. It should be the **per-wheel
    suspension stroke** (so a vehicle whose travel exceeds the old fixed 0.19 m
    is not visually muted) — pass a ``(n_wheels,)`` / broadcastable tensor.
    A scalar is also accepted (the helper's unit-test default is 0.19).

    ``miss`` is the sensor's ray-miss sentinel; ``None`` falls back to the
    deprecated ``raycast.RAY_MISS_THRESHOLD`` (correct only at the default
    ``max_range=20.0``)."""
    # Air = ray MISS, i.e. the sensor's own sentinel. ONE definition, shared
    # with visual._susp_visual_target: raycast.is_ray_hit_corrected — the
    # CORRECTED-distance predicate, because `distance` here is post-offset
    # (last_distances). It carries no unpopulated-sensor term; callers reach
    # this either inside a full step, or from wheel_visual_transforms whose
    # result is masked back to rest for any env with _stepped_once False.
    # A NEGATIVE distance is a valid
    # deep-over-compression reading under the high-cast ray scheme
    # (raycast.RAY_UP_OFFSET), not air — the clamp below bounds it.
    # (WheelJointInternalSync additionally slew-rate-limits its target; this
    # closed-form mirror is stateless and does not.)
    air = ~is_ray_hit_corrected(distance, miss)
    jp = mesh_radius - distance
    # broadcast_to, not full_like: `full_like`'s fill_value must be a Number
    # and raises `TypeError: fill_value must be Number, not Tensor` the moment
    # `l_susp` arrives as a per-row tensor (fused group). Value-identical for a
    # python float — both round it to jp's dtype once.
    neg_l_susp = -l_susp
    if not torch.is_tensor(neg_l_susp):
        neg_l_susp = torch.as_tensor(neg_l_susp, device=jp.device, dtype=jp.dtype)
    jp = torch.where(air, torch.broadcast_to(neg_l_susp, jp.shape), jp)
    if torch.is_tensor(clamp):
        return torch.maximum(-clamp, torch.minimum(clamp, jp))
    return torch.clamp(jp, -clamp, clamp)

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


def build_wheel_meta(resolved: ResolvedConfig, device: Any, dtype: Any) -> "WheelMeta":
    """Build the cached per-wheel tensor bundle from a resolved config.

    Hoisted out of :class:`VehiclePhysics` in v1.3.0 so anything that needs the
    same per-wheel arrays without a built Genesis entity can get them — notably
    :class:`genesis_vehicle.control.DifferentiablePlant`, whose prediction
    unroll calls the very same ``_pipeline.compute_wheel_step``, and the
    Genesis-free unit tests that exercise it.
    """
    return VehiclePhysics._build_wheel_meta_impl(resolved, device, dtype)


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
    # Bump-stop (v1.2.6): extra rate beyond rest_stroke; all-zero = off.
    # `has_bump_stop` is a build-time python bool so the pipeline can skip the
    # term without a per-step GPU sync. It is the OR of `k_bump > 0` over every
    # element, so it stays a single whole-batch branch when `k_bump` becomes
    # per-row: one wheel with a bump stop makes every row take the branch,
    # which is exact because `bump_stop_force` is 0 where `k_bump == 0`.
    # It replaced `float(k_bump.max()) > 0.0`, and the two agree on every
    # ordinary value INCLUDING -0.0 and +/-inf, but NOT under NaN: torch.max
    # propagates NaN, so the max form was False for a whole vehicle as soon as
    # ONE wheel carried a NaN k_bump, even with real bump stops on the others.
    # The OR is True there. That is the intended behaviour, not an accident —
    # a NaN rate is a broken config, and silently disabling every wheel's bump
    # stop is the worse of the two answers. Pinned in tests/test_bump_stop.py.
    k_bump: torch.Tensor             # (n_wheels,)
    has_bump_stop: bool
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


@dataclass
class VisualPartsTransforms:
    """One-stop render feed for an external engine (UE / Unity), produced by
    :meth:`VehiclePhysics.visual_parts_transforms`. WheelJointInternalSync-independent.

    ``chassis_*`` is the real dynamics pose (always world). ``wheel_*`` is the
    closed-form visual pose in ``frame`` (``"world"`` absolute, or ``"local"``
    relative to the chassis — attach wheels under the chassis component).
    Tensors keep a leading ``n_envs`` dim (1 for a single-env build)."""
    frame: str                       # wheels' frame ("world" | "local")
    chassis_pos: torch.Tensor        # (n_envs, 3)
    chassis_quat: torch.Tensor       # (n_envs, 4)  wxyz
    wheel_names: list                # length n_wheels (renderer mesh mapping)
    wheel_pos: torch.Tensor          # (n_envs, n_wheels, 3)
    wheel_quat: torch.Tensor         # (n_envs, n_wheels, 4)  wxyz


#: Per-wheel runtime tensors carried across a config rebuild, all shaped
#: ``(batch, n_wheels)`` and allocated together in ``VehiclePhysics.__init__``.
#: The first four are INTEGRATOR state (dropping them restarts the wheel
#: dynamics mid-drive — the v1.6.2 defect); the rest are read-layer
#: diagnostics carried in the same loop because they cost one clone each and a
#: reader between the rebuild and the next step would otherwise see zeros.
#: Used by :meth:`VehiclePhysics.export_runtime_state` /
#: :meth:`VehiclePhysics.load_runtime_state`.
RUNTIME_STATE_ATTRS = (
    "omega", "prev_compression",
    "last_distances", "last_hit",
    "wheel_spin_angle", "last_steer_per_wheel",
    "last_compression", "last_N", "last_F_long", "last_F_lat",
    "last_T_drive", "last_T_brake", "last_kappa", "last_alpha",
)

#: ``(batch,)``-shaped runtime state (one entry per flat batch row).
RUNTIME_STATE_ROW_ATTRS = ("_stepped_once",)

#: Batch-wide scalars carried with the tensors.
RUNTIME_STATE_SCALARS = ("_prev_init",)

#: Rest-pose constants captured ONCE at construction, while the suspension is
#: still at zero. A rebuild re-captures them from a SAGGING vehicle, which is
#: why they are carried rather than recomputed (measured: z 0.3000 → 0.1499).
#: ``load_runtime_state`` SKIPS a ``None`` source: ``__init__`` leaves these
#: None when ``_capture_rest_wheel_pose`` raised, and overwriting the new
#: instance's good capture with that None would break the visual getters.
REST_POSE_ATTRS = ("_rest_wheel_pos_local", "_rest_wheel_quat_local")


class VehiclePhysics:
    """Top-level vehicle physics driver. One instance per vehicle entity."""

    def __init__(
        self,
        scene: Any,
        entity: Any,
        sensor: Any,
        config: VehicleConfig,
        n_envs: int = 1,
        name: Optional[str] = None,
    ):
        assert n_envs >= 1
        global _BANNER_PRINTED
        if not _BANNER_PRINTED:
            import sys
            if not os.environ.get("GENESIS_VEHICLE_QUIET"):
                print(f"[genesis_vehicle] v{__version__}",
                      file=sys.stderr, flush=True)
            _BANNER_PRINTED = True
        self.scene = scene
        self.entity = entity
        self.sensor = sensor
        #: Identifier for messages about THIS driver's config (the dt-stability
        #: warning). ``VehicleScene`` passes the ``Vehicle.name`` the user gave
        #: ``add_vehicle``; a direct ``VehiclePhysics(...)`` / plain
        #: ``MultiVehiclePhysics(...)`` caller supplies no name, so it falls
        #: back to the URDF file name, which is the only identifier that exists
        #: on that path.
        self.vehicle_name = name or os.path.basename(
            getattr(config, "urdf_path", "") or "") or "<unnamed>"
        # The distance a wheel ray reports when it hit NOTHING. Read off the
        # sensor (raycast.ray_miss_value → the engine's no_hit_value, which
        # defaults to max_range). None when this instance was built WITHOUT a
        # sensor — VehicleScene's dual_scene mode does exactly that and injects
        # the raycast-scene sensor's value via set_ray_miss_value().
        self._ray_miss = None if sensor is None else ray_miss_value(sensor)
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
                # Project onto the spin axis rather than taking max(diag)
                # (v1.2.8): a wide, small-diameter wheel has its LARGEST
                # principal moment about a transverse axis, so the max would
                # be the wrong component. Wheels spin about body +Y
                # (ISO 8855; urdf._walk_to_wheel only accepts a ±Y spin joint).
                moi = estimate_spin_inertia_from_genesis(
                    entity, spin_link_name, spin_axis_local=(0.0, 1.0, 0.0))
                if moi > 0.0:
                    w.i_wheel = moi
            except Exception:
                pass

        self.wheel_meta = self._build_wheel_meta(self.resolved)

        # BUILD TIME for everything the step path derives from config: the
        # Ackermann geometry, the driven-axle share, the brake-bias / AWD
        # weight vectors, the drive-omega cap, and the two squared thresholds
        # (`_v_thr_sq`, `_eps2`). Deliberately here and not in each strategy's
        # __init__: `TankTuning.apply_config` (samples/tank_tuning.py:35-37)
        # writes hook scalars onto the CONSTRUCTED hook, so an __init__-time
        # snapshot would silently revert v_thr 5.0 -> 0.5. A later write is
        # picked up too — every one of these is a dependent cache keyed on its
        # sources (see _hotset).
        prime_derived(self.resolved, self.wheel_meta, self.dev, self.fdt)

        self._warn_bump_stop_dt()

        self.pre_loop_hooks = [h for h in self.resolved.stability_hooks if "PRE_LOOP" in h.slots]
        self.post_tire_hooks = [h for h in self.resolved.stability_hooks if "POST_TIRE" in h.slots]

        n_wheels = self.wheel_meta.n_wheels
        self.omega = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.prev_compression = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self._prev_init = False

        self.last_distances = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        # (n_envs, n_wheels) bool — "this wheel's ray found ground", aligned with
        # last_distances. CARRIED from read_distances(return_hit=True) (evaluated
        # on the RAW distances, the only place raycast.is_ray_hit's
        # unpopulated-zero term is meaningful) when this driver owns a sensor;
        # derived with raycast.is_ray_hit_corrected when distances are INJECTED
        # (dual_scene), where only corrected values exist. The two agree except
        # on the pre-first-step zero buffer, which _stepped_once gates. v1.5.2.
        self.last_hit = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=torch.bool)
        self.last_compression = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_N = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_F_long = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_F_lat = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_T_drive = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_T_brake = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_kappa = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.last_alpha = torch.zeros(n_envs, n_wheels, device=self.dev, dtype=self.fdt)

        self.visual: Optional[WheelJointInternalSync] = None
        if self.resolved.enable_wheel_joint_internal_sync:
            self.visual = WheelJointInternalSync(
                entity=entity, resolved=self.resolved,
                n_envs=n_envs, device=self.dev, dtype=self.fdt,
            )
            # Same ray-miss sentinel the read layer uses, so the visual air
            # pose and the physics air mask cannot disagree (set_ray_miss_value
            # keeps them in sync when the scene injects it later).
            self.visual.ray_miss = self._ray_miss

        self._up_world = torch.tensor(
            [0.0, 0.0, 1.0], device=self.dev, dtype=self.fdt,
        ).unsqueeze(0).expand(n_envs, 3).contiguous()
        # Pre-built (n_envs, n_wheels, 3) wheel-body broadcast.
        self._wheel_body_b = pw3(self.wheel_meta.positions).expand(
            n_envs, n_wheels, 3
        ).contiguous()

        # ---- Visual-pose state (for wheel_visual_transforms / WheelJointInternalSync) ----
        # Per-wheel steer angle from the last step (exposed for external
        # renderers); accumulated spin angle (maintained whether or not
        # WheelJointInternalSync runs, so the closed-form getter works headless).
        self.last_steer_per_wheel = torch.zeros(
            n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        self.wheel_spin_angle = torch.zeros(
            n_envs, n_wheels, device=self.dev, dtype=self.fdt)
        # True only after a FULL step (not the first-step early-return, where
        # WheelJointInternalSync is skipped). Gates wheel_visual_transforms'
        # deltas and wheels_grounded. PER ENV (n_envs,) since v1.5.2: a partial
        # reset(env_ids=...) zeroes only those envs' last_distances, so a single
        # scalar flag would report their stale zeros as a valid reading.
        self._stepped_once = torch.zeros(n_envs, device=self.dev, dtype=torch.bool)
        radii = [float(w.radius) for w in self.resolved.wheels if w.radius is not None]
        self._mesh_radius = float(sum(radii) / len(radii)) if radii else 0.35
        strokes = [float(w.rest_stroke) for w in self.resolved.wheels
                   if getattr(w, "rest_stroke", None) is not None]
        self._l_susp = float(sum(strokes) / len(strokes)) if strokes else 0.10
        # Visual suspension-offset clamp (a safety bound vs raycast spikes, not a
        # physics limit). Override via VehicleConfig.susp_visual_clamp:
        #   None  -> auto: per-wheel = that wheel's stroke (rest_d − radius), with
        #            a 0.02 m floor so a ~zero-stroke wheel isn't frozen.
        #   float -> uniform clamp on every wheel.
        # Replaces the old fixed ±0.19 m, which muted large-travel vehicles.
        # (1, n_wheels) for broadcast against (n_envs, n_wheels) offsets.
        _clamp_override = getattr(self.resolved, "susp_visual_clamp", None)
        if _clamp_override is not None:
            susp_stroke = torch.full_like(self.wheel_meta.rest_d,
                                          float(_clamp_override))
        else:
            susp_stroke = torch.clamp(
                self.wheel_meta.rest_d - self.wheel_meta.radius, min=0.02)
        self._susp_clamp = pw(susp_stroke)
        # Skid-steer/tank presets disable the wheel spin visual (cylindrical
        # road wheels — spin is invisible). Match WheelJointInternalSync so the
        # closed-form pose agrees: no spin baked into the quat when disabled.
        self._visual_spin_enabled = bool(
            getattr(self.resolved, "visual_spin_enabled", True))

        # Capture each wheel link's REST pose relative to the chassis (joints
        # still at 0 — no step / WheelJointInternalSync yet). wheel_visual_transforms then
        # composes steer/spin/suspension deltas ON TOP of this, so it reproduces
        # entity.get_link(wheel) exactly (rest link frame may sit below the
        # raycast attach point and carry a rest orientation).
        self._rest_wheel_pos_local = None
        self._rest_wheel_quat_local = None
        # Initialised unconditionally: the capture below only ASSIGNS this on
        # the failure path, so `p._rest_capture_err` used to raise
        # AttributeError on every healthy build - an attribute that exists only
        # when something went wrong cannot be read by a health check.
        self._rest_capture_err = None
        try:
            self._capture_rest_wheel_pose(entity)
        except Exception as e:   # entity not yet readable → lazy-capture on first getter call
            self._rest_capture_err = e

        _print_version_banner(self.resolved, n_envs)

    def _warn_bump_stop_dt(self) -> None:
        """Bump-stop stability guard (v1.2.6), named per vehicle.

        The suspension force is recomputed once per dt (held through the
        substeps), so the total spring rate is stability-bounded by
        ``(k_susp + k_bump) * dt^2 / m_share``. Measured on a 38.5 t /
        14-wheel tracked hull at dt=0.025: ratio 0.61 shows mm-level rest
        chatter, >= 0.86 diverges outright.

        SCOPE, exactly. This driver carries ONE config, so the ratio is that
        config's and the message names the vehicle it was built for
        (``self.vehicle_name``); where several vehicles share the kind they
        share the cfg, hence the ``+N more`` suffix the caller composes. The
        ratio mixes ``sprung_mass`` with the MAXIMUM ``k_susp + k_bump`` over
        the wheels — worst wheel against the average wheel's mass share, which
        is the intended conservative reading WITHIN one vehicle. It becomes a
        real per-slot mix-up only when ``wheel_meta`` carries several vehicles'
        rows (fused groups, STEP 3); this method is the single site that then
        has to loop.
        """
        if not self.wheel_meta.has_bump_stop:
            return
        sprung = getattr(self.resolved.urdf, "sprung_mass", None)
        if not sprung:
            return
        m_share = float(sprung) / self.wheel_meta.n_wheels
        k_tot = float((self.wheel_meta.k_susp + self.wheel_meta.k_bump).max())
        ratio = k_tot * self.dt * self.dt / m_share
        if ratio > 0.7:
            import logging
            logging.getLogger("genesis_vehicle").warning(
                "vehicle '%s': bump-stop is too stiff for this timestep: "
                "(k_susp + k_bump)*dt^2/m_share = %.2f (> 0.7). The "
                "suspension will chatter or diverge - lower k_bump or dt.",
                self.vehicle_name, ratio,
            )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def _stepped_any(self) -> bool:
        """True if ANY env has taken a full physics step.

        This is ONLY the cheap all-or-nothing shortcut for the batch-wide
        rest-pose fallbacks (``wheel_visual_transforms`` and the two in
        ``MultiVehicleKindPhysics``): False means NO env has stepped, so the
        whole batch is at rest and no deltas need computing at all.

        It is NOT a substitute for the per-env flag. A partial
        ``reset(env_ids=...)`` un-steps SOME envs, and ``.any()`` is then True
        while those envs' ``last_distances`` are zeros — which
        ``_susp_visual_offset`` would read as a fully-compressed suspension.
        Callers must therefore mask per env with ``_stepped_once`` after the
        shortcut (``wheel_visual_transforms`` does; see v1.5.2). ``.any()``
        alone was bit-identical only for a batch that steps uniformly."""
        return bool(self._stepped_once.any())

    def set_ray_miss_value(self, miss: Optional[float], *,
                           sensor: Optional[Any] = None) -> None:
        """Tell this driver which distance means "the ray hit NOTHING".

        Normally read off ``sensor`` at construction. It has to be injectable
        because ``VehicleScene``'s ``dual_scene`` mode builds ``VehiclePhysics``
        with ``sensor=None`` (the rays live in a SEPARATE scene) and feeds
        ``step(distances=...)`` from there; the scene injects that sensor's
        value here. ``None`` falls back to the deprecated
        ``raycast.RAY_MISS_THRESHOLD``.

        VALIDATED, not merely stored (v1.5.2 — it used to keep ``float(miss)``
        unchecked while its docstring pointed at the stamp path's rule, so
        ``set_ray_miss_value(0.0)`` reopened the launch-the-vehicle failure
        mode through ``_susp_visual_offset`` / ``is_ray_hit_corrected``).
        Exactly what is enforced:

        * against ``sensor`` (the keyword, for the injected ``dual_scene``
          case where this driver owns none) or else ``self.sensor``, through
          the same :func:`genesis_vehicle.raycast.check_miss_supported` the
          stamp and read paths use: a sentinel below that raycaster's
          ``max_range`` raises ``ValueError``;
        * with no sensor on either side there is no ``max_range`` to compare
          with, so only that function's unconditional stage holds: a sentinel
          that cannot be a distance AT ALL (non-finite, or ``<= 0``) raises.

        Both stages live in ``check_miss_supported`` since v1.5.2 — this method
        used to carry its own copy of the unconditional one, which meant the
        stamp/read paths did NOT have it (a stub sensor accepted ``0.0``, a real
        one accepted ``nan`` because ``nan < max_range`` is False).

        See :func:`genesis_vehicle.raycast.set_sensor_miss_value` for why a
        sentinel below the raycaster's ``max_range`` is unsupported."""
        if miss is None:
            self._ray_miss = None
        else:
            src = sensor if sensor is not None else self.sensor
            self._ray_miss = check_miss_supported(src, float(miss))
        if self.visual is not None:
            self.visual.ray_miss = self._ray_miss

    @property
    def wheels_grounded(self) -> torch.Tensor:
        """``(n_envs, n_wheels)`` bool — did each wheel's ray find ground on the
        last step?

        ``last_hit`` — the mask the read layer produced alongside
        ``last_distances`` (see that attribute) — AND'ed with the per-env "has
        stepped" flag, so an env that has never stepped (or was just ``reset``)
        reports all-False instead of passing vacuously on the zero tensor
        ``last_distances`` starts out as. The gate is not decoration: the
        corrected-distance predicate that feeds ``last_hit`` on the injected
        path cannot recognise that zero buffer by itself.

        This is a READ-LAYER diagnostic, not the physics air mask: the pipeline
        decides air from ``compression <= 0`` and never looks at the sentinel.
        The two agree except within the suspension's rest gap, where a ray can
        hit ground that is still too far away to compress the spring."""
        return self.last_hit & self._stepped_once.unsqueeze(1)

    @property
    def all_wheels_grounded(self) -> torch.Tensor:
        """``(n_envs,)`` bool — True where EVERY wheel of that env found ground.
        The question "is this vehicle actually on the terrain?"."""
        return self.wheels_grounded.all(dim=1)

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            self.omega.zero_()
            self.prev_compression.zero_()
            self._prev_init = False
            self.last_distances.zero_(); self.last_hit.zero_()
            self.last_compression.zero_()
            self.last_N.zero_(); self.last_F_long.zero_(); self.last_F_lat.zero_()
            self.last_T_drive.zero_(); self.last_T_brake.zero_()
            self.last_kappa.zero_(); self.last_alpha.zero_()
            self.last_steer_per_wheel.zero_(); self.wheel_spin_angle.zero_()
            self._stepped_once.zero_()
            if self.visual is not None:
                self.visual.reset_visual_state()
            return
        idx = env_ids
        if idx.dtype == torch.bool:
            idx = torch.nonzero(idx, as_tuple=False).flatten()
        self.omega[idx] = 0.0
        self.prev_compression[idx] = 0.0
        self.last_distances[idx] = 0.0; self.last_hit[idx] = False
        self.last_compression[idx] = 0.0
        self.last_N[idx] = 0.0; self.last_F_long[idx] = 0.0; self.last_F_lat[idx] = 0.0
        self.last_T_drive[idx] = 0.0; self.last_T_brake[idx] = 0.0
        self.last_kappa[idx] = 0.0; self.last_alpha[idx] = 0.0
        self.last_steer_per_wheel[idx] = 0.0; self.wheel_spin_angle[idx] = 0.0
        # These envs' last_distances are now zeros again, so they have NOT
        # stepped as far as any reader is concerned (v1.5.2 — a scalar flag left
        # a stale True here and wheels_grounded reported the zeros as ground).
        self._stepped_once[idx] = False
        if self.visual is not None:
            self.visual.reset_visual_state(idx)

    # -----------------------------------------------------------------
    # Runtime-state carry (config rebuild) — v1.6.2
    # -----------------------------------------------------------------
    def _row_index(self, rows: Optional[Any]) -> Optional[torch.Tensor]:
        if rows is None:
            return None
        idx = torch.as_tensor(rows, device=self.dev)
        if idx.dtype == torch.bool:
            idx = torch.nonzero(idx, as_tuple=False).flatten()
        return idx.to(dtype=torch.long)

    def export_runtime_state(self, rows: Optional[Any] = None) -> dict:
        """Snapshot the runtime state a config rebuild must carry over.

        ``rows`` is an index into the FLAT batch (``env * K + slot`` for a
        multi-vehicle kind's proto, ``env`` for a plain driver); ``None`` takes
        every row. Every tensor is cloned, so the snapshot survives the old
        driver being discarded.

        What is in it: :data:`RUNTIME_STATE_ATTRS`,
        :data:`RUNTIME_STATE_ROW_ATTRS`, :data:`RUNTIME_STATE_SCALARS` and
        :data:`REST_POSE_ATTRS`, plus ``n_wheels`` / ``n_rows`` so
        :meth:`load_runtime_state` can reject a mismatched target. NOT in it:
        anything re-derived from the config (``resolved``, ``wheel_meta``,
        hooks, ``dt``, the ray-miss sentinel) — re-resolution is the point of a
        rebuild."""
        idx = self._row_index(rows)

        def _g(t):
            return (t.detach().clone() if idx is None
                    else t.detach()[idx].clone())

        state = {name: _g(getattr(self, name)) for name in RUNTIME_STATE_ATTRS}
        for name in RUNTIME_STATE_ROW_ATTRS:
            state[name] = _g(getattr(self, name))
        for name in RUNTIME_STATE_SCALARS:
            state[name] = bool(getattr(self, name))
        for name in REST_POSE_ATTRS:
            v = getattr(self, name, None)
            state[name] = None if v is None else v.detach().clone()
        state["n_wheels"] = int(self.wheel_meta.n_wheels)
        state["n_rows"] = int(self.n_envs if idx is None else int(idx.numel()))
        # Wheel IDENTITY, in column order. Every per-wheel tensor above is
        # carried by COLUMN INDEX, so a destination whose wheels are the same
        # COUNT in a different ORDER would silently land omega / compression /
        # spin / rest pose on the wrong physical wheel. The consumer
        # (VehicleScene._precheck_wheel_counts) compares this and refuses.
        state["wheel_names"] = [getattr(w, "name", None)
                                for w in self.resolved.wheels]
        return state

    def load_runtime_state(self, state: dict, rows: Optional[Any] = None) -> None:
        """Write a :meth:`export_runtime_state` snapshot back into this driver.

        ``rows`` selects the destination rows of the FLAT batch, exactly as on
        export. Raises ``ValueError`` when the wheel count or the row count
        differs — the per-wheel tensors would not line up, and the wheel count
        is fixed at build time by the raycast sensor anyway (see
        ``docs/physics-contracts.md`` §7.12).

        The rest pose is copied only when the snapshot HAS one: a source whose
        ``_capture_rest_wheel_pose`` failed carries ``None``, which must not
        overwrite this instance's good capture."""
        n = int(self.wheel_meta.n_wheels)
        src_n = int(state.get("n_wheels", n))
        if src_n != n:
            raise ValueError(
                f"runtime-state wheel count {src_n} != this driver's {n}; the "
                f"wheel count is fixed at build time (the raycast sensor's ray "
                f"count is), so the state cannot be carried")
        idx = self._row_index(rows)
        want = int(self.n_envs if idx is None else int(idx.numel()))
        src_rows = int(state.get("n_rows", want))
        if src_rows != want:
            raise ValueError(
                f"runtime-state row count {src_rows} != destination {want} "
                f"(batch rows are env * K + slot)")

        for name in RUNTIME_STATE_ATTRS + RUNTIME_STATE_ROW_ATTRS:
            src = state.get(name)
            if src is None:
                continue
            dst = getattr(self, name)
            src = src.to(device=dst.device, dtype=dst.dtype)
            if idx is None:
                dst.copy_(src)
            else:
                dst[idx] = src
        if idx is None:
            self._prev_init = bool(state.get("_prev_init", self._prev_init))
        else:
            # Partial load: _prev_init is batch-wide ("distances have been
            # populated"), so ANY carried row that had stepped initialises it.
            self._prev_init = bool(self._prev_init) or bool(
                state.get("_prev_init", False))
        for name in REST_POSE_ATTRS:
            src = state.get(name)
            if src is None:
                continue
            setattr(self, name, src.detach().clone().to(device=self.dev,
                                                        dtype=self.fdt))

    def link_transforms(self, frame: str = "parent", *, envs_idx: Optional[Any] = None):
        """Per-link transforms of this vehicle's entity in ``frame``.

        Thin wrapper over :func:`genesis_vehicle.kinematics.get_link_transforms`
        bound to ``self.entity``. ``frame`` is ``"world"``, ``"base"``, or
        ``"parent"`` (URDF-hierarchy-local; default). Returns a
        ``LinkTransforms`` (link names + parent topology + batched pos/quat).

        Use for telemetry → animation retargeting, sensor/effect attachment, or
        placing ghost copies. See the kinematics module docstring for frames.

        NOTE: the wheel links reflect steering / suspension / spin ONLY when
        WheelJointInternalSync is enabled (it drives those URDF joints). With WheelJointInternalSync
        off, wheel links sit at the rest pose. For an external renderer (UE /
        Unity), prefer :meth:`wheel_visual_transforms`, which is computed
        closed-form and works regardless of WheelJointInternalSync. A one-time warning is
        emitted if you call this with WheelJointInternalSync disabled.
        """
        global _VISUAL_OFF_WARNED
        if self.visual is None and not _VISUAL_OFF_WARNED:
            import sys
            print(
                "[genesis_vehicle] WARN: link_transforms() read with WheelJointInternalSync "
                "disabled — wheel links are at the REST pose (no steer/suspension/"
                "spin). Use wheel_visual_transforms() for a WheelJointInternalSync-independent "
                "visual pose.", file=sys.stderr, flush=True)
            _VISUAL_OFF_WARNED = True
        from .kinematics import get_link_transforms
        return get_link_transforms(self.entity, frame, envs_idx=envs_idx)

    def _capture_rest_wheel_pose(self, entity: Any) -> None:
        """Read each wheel link's rest pose relative to the chassis base (joints
        at 0). Stored as ``(n_wheels, 3)`` / ``(n_wheels, 4)`` constants."""
        bpos = entity.get_pos(); bquat = entity.get_quat()
        if bpos.dim() > 1:
            bpos = bpos[0]; bquat = bquat[0]
        bpos = bpos.to(self.dev, self.fdt); bquat = bquat.to(self.dev, self.fdt)
        bconj = torch.stack([bquat[0], -bquat[1], -bquat[2], -bquat[3]])
        rp, rq = [], []
        for w in self.resolved.wheels:
            lk = entity.get_link(w.name)
            wp = lk.get_pos(); wq = lk.get_quat()
            if wp.dim() > 1:
                wp = wp[0]; wq = wq[0]
            wp = wp.to(self.dev, self.fdt); wq = wq.to(self.dev, self.fdt)
            # rest pos/quat in base frame
            rp.append(transform_by_quat((wp - bpos).unsqueeze(0), bconj.unsqueeze(0))[0])
            rq.append(_quat_mul(bconj, wq))
        self._rest_wheel_pos_local = torch.stack(rp, dim=0)    # (n, 3)
        self._rest_wheel_quat_local = torch.stack(rq, dim=0)   # (n, 4)

    def wheel_visual_transforms(self, frame: str = "world", *,
                                envs_idx: Optional[Any] = None):
        """Closed-form per-wheel VISUAL pose — steer + suspension + spin applied
        — **without** driving Genesis joints (works whether or not WheelJointInternalSync is
        enabled). The intended feed for an external renderer (UE / Unity).

        Parameters
        ----------
        frame : {"world", "local"}
            ``"local"`` returns the wheel transform **relative to the chassis**
            (attach the wheel mesh under the chassis component and set this as
            its relative transform — the wheel then rides the chassis rigidly).
            ``"world"`` returns the absolute transform.
        envs_idx : optional
            Subset of envs (passed to the chassis read for the world frame).

        Returns
        -------
        (pos, quat) : tuple of tensors
            ``pos`` ``(n_envs, n_wheels, 3)``, ``quat`` ``(n_envs, n_wheels, 4)``
            (w, x, y, z). For ``n_envs == 1`` the leading dim is kept.

        Notes
        -----
        Assumes the conventional ray-wheel axes the presets use: steer about
        chassis +z, suspension travel along chassis ±z, spin about the wheel
        axle (+y). The steer sign follows the URDF steer-axis convention (same
        as WheelJointInternalSync). This matches ``entity.get_link(wheel)`` when WheelJointInternalSync
        is enabled (unit-checked in tests), but costs ~µs (a few quaternion
        ops per wheel) instead of the engine's articulated-body FK.
        """
        if frame not in ("world", "local"):
            raise ValueError(f"frame must be 'world' or 'local', got {frame!r}")
        if self._rest_wheel_pos_local is None:
            self._capture_rest_wheel_pose(self.entity)

        # Before the first FULL step (the first-step early-return skips the
        # pipeline AND WheelJointInternalSync), wheels are at the rest pose.
        # Apply no deltas. `_stepped_any()` is the batch-wide shortcut for
        # "nobody has stepped"; envs that a PARTIAL reset un-stepped are
        # masked back to the rest pose below.
        if not self._stepped_any():
            rest_pos = pw3(self._rest_wheel_pos_local).expand(
                self.n_envs, -1, 3).contiguous()
            rest_quat = pw3(self._rest_wheel_quat_local).expand(
                self.n_envs, -1, 4).contiguous()
            if frame == "local":
                return rest_pos, rest_quat
            cpos = self.entity.get_pos(); cquat = self.entity.get_quat()
            if cpos.dim() == 1:
                cpos = cpos.unsqueeze(0); cquat = cquat.unsqueeze(0)
            N, n = rest_pos.shape[0], rest_pos.shape[1]
            cqb = cquat.unsqueeze(1).expand(N, n, 4)
            wp = cpos.unsqueeze(1) + transform_by_quat(
                rest_pos.reshape(N * n, 3), cqb.reshape(N * n, 4)).reshape(N, n, 3)
            return wp, _quat_mul(cqb, rest_quat)

        # Per-wheel visual deltas, applied on top of the captured rest pose.
        # Net visual steer about chassis +z is -phys regardless of the URDF
        # steer-axis sign: WheelJointInternalSync's visual_cmd (= -phys·sign) rotated about
        # the axis (z-component = sign) gives (-phys·sign)·sign = -phys. So the
        # axis sign cancels — do NOT multiply by it here.
        steer_z = -self.last_steer_per_wheel                                 # (N, n)
        susp_off = _susp_visual_offset(
            self.last_distances, self._mesh_radius, self._l_susp,
            self._susp_clamp, self._ray_miss)                               # (N, n)
        spin = (self.wheel_spin_angle if self._visual_spin_enabled
                else torch.zeros_like(self.wheel_spin_angle))                # (N, n)

        rest_pos = pw3(self._rest_wheel_pos_local)                           # (1, n, 3)
        rest_quat = pw3(self._rest_wheel_quat_local)                         # (1, n, 4)

        # Position: rest + suspension translation along chassis +z.
        z_off = torch.stack(
            [torch.zeros_like(susp_off), torch.zeros_like(susp_off), susp_off], dim=-1)
        local_pos = rest_pos + z_off                                        # (N, n, 3)
        # Orientation: rest ∘ R_z(steer) ∘ R_y(spin).
        local_quat = _quat_mul(
            rest_quat,
            _quat_mul(_quat_axis_angle("z", steer_z), _quat_axis_angle("y", spin)),
        )                                                                   # (N, n, 4)

        # PER-ENV rest-pose fallback (v1.5.2). `_stepped_once` is per env, so a
        # partial reset(env_ids=...) leaves those envs un-stepped with zeroed
        # last_distances; feeding that zero to _susp_visual_offset renders them
        # FULLY COMPRESSED (a corrected 0.0 is a legitimate contact at exactly
        # RAY_UP_OFFSET, not air — see raycast.is_ray_hit_corrected). Give them
        # the same rest pose the all-unstepped branch above returns. Skipped
        # entirely — bit-identical, no extra kernels — when every env has
        # stepped, which is the steady state.
        if not bool(self._stepped_once.all()):
            m = self._stepped_once.view(-1, 1, 1)
            local_pos = torch.where(m, local_pos, rest_pos.expand_as(local_pos))
            local_quat = torch.where(m, local_quat, rest_quat.expand_as(local_quat))

        if frame == "local":
            return local_pos, local_quat

        # World: compose with the chassis (base-link) pose.
        cpos = self.entity.get_pos(envs_idx=envs_idx) if envs_idx is not None else self.entity.get_pos()
        cquat = self.entity.get_quat(envs_idx=envs_idx) if envs_idx is not None else self.entity.get_quat()
        if cpos.dim() == 1:
            cpos = cpos.unsqueeze(0); cquat = cquat.unsqueeze(0)
        N, n = local_pos.shape[0], local_pos.shape[1]
        cquat_b = cquat.unsqueeze(1).expand(N, n, 4)
        world_pos = cpos.unsqueeze(1) + transform_by_quat(
            local_pos.reshape(N * n, 3), cquat_b.reshape(N * n, 4)
        ).reshape(N, n, 3)
        world_quat = _quat_mul(cquat_b, local_quat)
        return world_pos, world_quat

    def visual_parts_transforms(self, frame: str = "world", *,
                          envs_idx: Optional[Any] = None) -> "VisualPartsTransforms":
        """One call returning everything an external renderer needs for this
        vehicle: the chassis pose **and** the wheel visual poses. Fully
        WheelJointInternalSync-independent (works headless).

        The chassis comes from real dynamics (``entity.get_pos/get_quat`` —
        always world, the physical truth). The wheels come from
        :meth:`wheel_visual_transforms` (closed-form steer + suspension + spin).
        ``frame`` applies to the WHEELS: ``"world"`` = absolute, ``"local"`` =
        relative to the chassis (attach wheel meshes under the chassis
        component). The chassis is always world.

        Returns a :class:`VisualPartsTransforms`. This is the recommended feed for a
        UE / Unity bridge — one call per vehicle, no get_link, no WheelJointInternalSync.
        """
        cpos = self.entity.get_pos(envs_idx=envs_idx) if envs_idx is not None else self.entity.get_pos()
        cquat = self.entity.get_quat(envs_idx=envs_idx) if envs_idx is not None else self.entity.get_quat()
        if cpos.dim() == 1:
            cpos = cpos.unsqueeze(0); cquat = cquat.unsqueeze(0)
        wp, wq = self.wheel_visual_transforms(frame, envs_idx=envs_idx)
        return VisualPartsTransforms(
            frame=frame, chassis_pos=cpos, chassis_quat=cquat,
            wheel_names=[w.name for w in self.resolved.wheels],
            wheel_pos=wp, wheel_quat=wq,
        )

    def step(self, inputs: VehicleStepInputs,
             distances: Optional[torch.Tensor] = None) -> None:
        """Vectorized 5-step pipeline. Per-wheel work is a SINGLE batched
        tensor op set (no Python wheel loop).

        Parameters
        ----------
        inputs : VehicleStepInputs | VehicleInputs
            Throttle/brake/steer for this step.
        distances : torch.Tensor, optional
            Externally-supplied wheel-ground ray distances, shape
            ``(n_envs, n_wheels)`` (``(n_wheels,)`` accepted for a single env).
            When given, the pipeline uses these instead of reading
            ``self.sensor`` — this is the hook :class:`VehicleScene`
            ``dual_scene`` mode uses to feed distances measured in a SEPARATE static-terrain
            raycast scene (whose BVH is built once, never re-fit per step; see
            ``docs/dual-scene-raycast.md``). When ``None`` (default) the sensor
            is read exactly as before — fully backward compatible.
        """
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

        # [RAYCAST] — distances may be injected by an external raycast source
        # (VehicleScene dual_scene mode reads them from a separate kinematic-terrain
        # scene whose BVH is static). When omitted, read this vehicle's own
        # sensor exactly as before.
        if distances is None:
            if self.sensor is None:
                raise ValueError(
                    "VehiclePhysics.step() needs wheel-ground distances: this "
                    "instance was built with sensor=None, so pass distances= "
                    "(shape (n_envs, n_wheels)) from your raycast source."
                )
            # Carry the sensor's own RAW-evaluated hit mask out with the
            # corrected distances; it is the only one that can tell an
            # unpopulated zero buffer from a hit at exactly the high-cast offset.
            distances, hit = read_distances(self.sensor, n_envs,
                                            miss=self._ray_miss, return_hit=True)
        else:
            distances = self._coerce_distances(distances)
            # Injected (dual_scene): only CORRECTED distances exist here, so the
            # sentinel test is the whole predicate and _stepped_once is the
            # guard for the pre-first-step zeros. See raycast.is_ray_hit_corrected.
            hit = is_ray_hit_corrected(distances, self._ray_miss)
        self.last_distances = distances.detach().clone()
        self.last_hit = hit.detach().clone()
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

        # Visual-pose bookkeeping (cheap; needed by wheel_visual_transforms even
        # when WheelJointInternalSync is disabled). Spin integrates the post-update omega,
        # matching WheelJointInternalSync's accumulator.
        self.last_steer_per_wheel = steer_per_wheel
        two_pi = 2.0 * math.pi
        self.wheel_spin_angle = (
            (self.wheel_spin_angle + self.omega * DT) + math.pi
        ) % two_pi - math.pi

        # Diagnostics (all (n_envs, n)).
        self.last_N = res.N
        self.last_F_long = res.F_long
        self.last_F_lat = res.F_lat
        self.last_compression = res.compression
        self.last_kappa = res.kappa
        self.last_alpha = res.alpha

        # [APPLY]
        apply_links_wrench(self.solver, total_F.unsqueeze(1), total_T.unsqueeze(1),
                           self.base_idx_list)
        self._prev_init = True
        self._stepped_once[:] = True

        # [VISUAL]
        if self.visual is not None:
            self.visual.step(steer_per_wheel, distances, self.omega, DT)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _build_wheel_meta(self, resolved: ResolvedConfig) -> WheelMeta:
        return build_wheel_meta(resolved, self.dev, self.fdt)

    @staticmethod
    def _build_wheel_meta_impl(resolved: ResolvedConfig, d, f) -> WheelMeta:
        wheels = resolved.wheels
        n = len(wheels)
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
        k_bump = _t("k_bump")
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
            k_bump=k_bump,
            has_bump_stop=bool((k_bump > 0.0).any()),
            mu_long=_t("mu_long"), mu_lat=_t("mu_lat"),
            rolling_resistance_cr=_t("rolling_resistance_cr"),
            pb_x=_t("pb_x"), pc_x=_t("pc_x"), pe_x=_t("pe_x"),
            pb_y=_t("pb_y"), pc_y=_t("pc_y"), pe_y=_t("pe_y"),
        )

    def _coerce_distances(self, distances: Any) -> torch.Tensor:
        """Normalize externally-injected wheel-ground distances to
        ``(n_envs, n_wheels)`` on this driver's device/dtype. Accepts a
        ``(n_wheels,)`` vector (single env) or ``(n_envs, n_wheels)``."""
        d = distances
        if not torch.is_tensor(d):
            d = torch.as_tensor(d)
        d = d.to(device=self.dev, dtype=self.fdt)
        if d.dim() == 1:
            d = d.unsqueeze(0)
        n = self.wheel_meta.n_wheels
        if tuple(d.shape) != (self.n_envs, n):
            raise ValueError(
                f"injected distances must be (n_envs, n_wheels)="
                f"({self.n_envs}, {n}); got {tuple(d.shape)}"
            )
        return d.contiguous()

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
