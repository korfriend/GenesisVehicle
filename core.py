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

from ._gs_compat import apply_links_wrench, check_link_inertial_api, require_genesis

require_genesis(gs)   # genesis-world >= 1.5.0 required (2.0.5); raises GenesisVersionError

from genesis.utils.geom import transform_by_quat  # noqa: E402

from ._version import __version__  # noqa: E402
from .config import ResolvedConfig, VehicleConfig, check_rolling_resistance_cr, resolve  # noqa: E402
from .dynamics import (YawInertia, rigid_share_factor, static_mass_shares,
                       yaw_inertia_from_mass_props)
from ._hotset import prime_derived
from .strategies.stability import register_sfl_cap_name
from ._pipeline import compute_wheel_step, pw, pw3
from .inputs import VehicleInputs, VehicleStepInputs
from .gear import (check_vehicle_request, clamp_drive_to_gear, effective_throttle,
                   encode_requests, refuses_gear, update_gear, with_throttle)
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

    ``miss`` is the sensor's ray-miss sentinel; ``None`` raises ``ValueError``
    since v1.8.0 (through 1.7.x it fell back to the deprecated
    ``raycast.RAY_MISS_THRESHOLD``, correct only at the default
    ``max_range=20.0``; the constant was removed in 1.8.0)."""
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
# One-shot flag for the `scene.sim.dt` read failing outright (see below).
_DT_FALLBACK_WARNED = False


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
        # Fall back to the preset's value — but say so. On the genesis the
        # SDK requires (1.5.0) ``scene.sim.dt`` reads fine even BEFORE
        # build(), so on a real Scene this branch is unreachable; reaching it
        # means the caller passed something that is not a Scene, or the engine
        # moved the time axis again. Either way the vehicle would then
        # integrate at a dt the solver does not use. Warn once per process.
        global _DT_FALLBACK_WARNED
        if not _DT_FALLBACK_WARNED:
            _DT_FALLBACK_WARNED = True
            import logging
            logging.getLogger("genesis_vehicle").warning(
                "could not read scene.sim.dt (scene=%r); falling back to the "
                "preset's recommended_dt=%g. The SDK's per-step integrators "
                "will use that value even if the solver steps at another.",
                type(scene).__name__, float(recommended_dt), exc_info=True)
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
    # Inputs of the rigid-body deadbeat scale (v1.8.0; dynamics.YawInertia:
    # composite yaw inertia about the centre of mass + squared horizontal
    # distance of each ray origin from it). None until the OWNING driver
    # computes it: VehiclePhysics._ensure_yaw_inertia(), once, on its first
    # step() (a batched kind computes it on its proto, the same object). It is
    # NOT recomputed later (a tank's composite inertia read moves by 2.7e-4
    # relative after 40 steps of driving, as its turret links settle; the
    # first-step value is kept), and a config rebuild re-derives it from the
    # carried mass properties and the rebuilt positions. A DifferentiablePlant
    # passes its OWN inputs to compute_wheel_step by keyword and never writes
    # this field (it shares this object with the live driver). Appended LAST
    # (keyword default) so the positional fields above keep their order.
    yaw_inertia: Optional[YawInertia] = None
    # (v2.0.4) Per-wheel STATIC mass share, kg, (n_wheels,) on this meta's
    # device/dtype: dynamics.static_mass_shares (elastic-support statics) from
    # the build-time link masses of the owning driver's entity (a batched
    # kind: its proto's, i.e. entities[0]). Input of StaticFrictionLock's
    # spring cap. None until VehiclePhysics._ensure_static_mass_share() sets it
    # on the first step(); stays None (the lock then runs UNCAPPED) when the
    # mass properties cannot be read. Runtime mass changes are not reflected.
    # Appended after yaw_inertia so every earlier field keeps its position.
    static_mass_share: Optional[torch.Tensor] = None
    # (v2.0.4 Step D) dynamics.RigidShareFactor (beta, yaw_fraction; python
    # floats) of the planar rigid-body term of StaticFrictionLock's spring cap,
    # from the same link walk and shares as static_mass_share (never from
    # yaw_inertia). Written by _ensure_static_mass_share just before the share;
    # None when the share was preset by the caller, could not be computed, or
    # beta could not be computed (logged) — the cap is then per wheel only.
    # A static_mass_share written by the user later keeps this (old) beta.
    # Appended last so every earlier field keeps its position.
    static_rigid_factor: Optional[Any] = None


@dataclass
class PipelineContext:
    """Mutable context passed to stability hooks.

    v0.5.0: now batched over wheels — F_long/F_lat/N/v_long/v_lat are all
    (n_envs, n_wheels) tensors, not (n_envs,) per-wheel scalars. Hook
    implementations have a single `apply_post_tire(ctx)` call per step
    (no `wheel_idx` argument). v1.7.0: a POST_TIRE hook also gets one
    `apply_pre_tire(ctx)` call per step, before the tyre."""
    # Batched per-wheel state (all (n_envs, n_wheels))
    F_long: Optional[torch.Tensor] = None
    F_lat: Optional[torch.Tensor] = None
    N: Optional[torch.Tensor] = None
    v_long: Optional[torch.Tensor] = None
    v_lat: Optional[torch.Tensor] = None
    # In apply_pre_tire: the pre-step omega. In apply_post_tire (v1.8.0):
    # ``omega_t``, the omega after this step's resisting torque, drive torque
    # and brake projection, before tyre friction; the tyre was evaluated at it
    # clipped to +-omega_max. Through 1.7.0 this comment said it was the omega
    # after the resisting torque only and equal to the pre-step omega when no
    # hook set T_resist; that is no longer true.
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
    # Per-wheel AXLE resisting torque (v1.7.0), (n_envs, n_wheels), N*m, with
    # the SAME sign as ``omega`` (a positive value opposes positive spin).
    # Reset to None by ``_pipeline.compute_wheel_step`` before the hooks run; a
    # hook ACCUMULATES into it in ``apply_pre_tire`` ONLY (``RollingResistance``
    # does) -- a write in ``apply_post_tire`` raises RuntimeError (an in-place
    # one is not detected under torch.inference_mode()). The pipeline
    # caps it at ``|omega| * i_wheel / dt`` (``dynamics.cap_resist_torque``)
    # against the pre-step omega and subtracts it from omega FIRST, before the
    # drive torque, the brake projection and the tyre. (Through 1.7.0 the brake
    # was then capped together with it and only the remainder entered the
    # overshoot clamp; since 1.8.0 the brake is ``dynamics.brake_projection``
    # and nothing of T_resist enters the clamp.) None = no resisting torque.
    # Appended after the fields above so positional construction keeps its
    # meaning.
    T_resist: Optional[torch.Tensor] = None
    # --- v1.8.0, reset to None by the pipeline every step. F_long_bounds,
    # F_long_deadbeat and deadbeat_scale are set for the hooks and are
    # READ-ONLY for them: the pipeline computes with its own copies, so
    # rebinding them or writing them in place changes nothing it computes
    # (``deadbeat_scale`` is set before the PRE_TIRE phase, the other two
    # before the POST_TIRE phase). hold_applied is the one a hook WRITES. ---
    # ``(lo, hi)``, each (n_envs, n_wheels), N: the interval of tyre F_long the
    # wheel can transmit this step before the held-wheel deadbeat -- the
    # overshoot clamp widened by the brake capacity left on a held wheel
    # (``B_rem / r`` both ways). Always contains 0. ``StaticFrictionLock``
    # limits its hold force to it.
    F_long_bounds: Optional[tuple] = None
    # (n_envs, n_wheels), N: the held-wheel deadbeat |F_long| bound,
    # ``s * (N / g) * |v_long - r * omega| / dt`` on a wheel the brake holds
    # (B_rem > 0), +inf elsewhere. ``StaticFrictionLock`` uses it to detect a
    # creep equilibrium.
    F_long_deadbeat: Optional[torch.Tensor] = None
    # (n_envs, n_wheels) bool, or None: wheels whose force a hook REPLACED with
    # a position-anchored hold (``StaticFrictionLock`` after rest). A hook ORs
    # its mask into it (do not overwrite another hook's). Those wheels get the
    # clamp WITHOUT the held-wheel deadbeat and no lateral deadbeat. None (no
    # hook set it) = no wheel is exempt.
    hold_applied: Optional[torch.Tensor] = None
    # The rigid-body deadbeat scale s (dynamics.deadbeat_scale), set before
    # the PRE_TIRE phase. A (B, 1) tensor, B = the pipeline's batch rows:
    # n_envs for VehiclePhysics, n_envs*K in a batched kind, plant members x
    # candidates in DifferentiablePlant. Value in (0, 1], computed from THIS
    # step's N, so a row with every wheel airborne reads 1.0. It is the python
    # float 1.0 only when compute_wheel_step ran with no yaw-inertia inputs
    # (a direct caller). A copy: writing it changes nothing the pipeline
    # computes. Hooks that bound a force by a one-step deadbeat multiply their
    # mass share ``N / g`` by it (``StaticFrictionLock``'s damper bound does).
    deadbeat_scale: Optional[Any] = None
    # (v2.0.2) The ENGAGED gear of each batch row after this step's shift
    # evaluation, a (B,) float tensor of +-1, or None when no input of the
    # step carried a gear (the signed-throttle path; then nothing below is
    # set). In a batch where only some rows are in gear mode, the rows without
    # a gear carry their stored (unused) state. Set before the PRE_LOOP hooks;
    # ``throttle`` above is then the EFFECTIVE signed throttle
    # (gear * clamp(throttle, 0, 1) on gear rows). Read-only for hooks.
    # Appended last so positional construction keeps its meaning.
    gear: Optional[torch.Tensor] = None
    # (v2.0.4) (n_envs, n_wheels), N*m, >= 0: the brake torque LEFT on each
    # wheel after this step's brake projection (dynamics.brake_projection's
    # ``B_rem``) -- > 0 only where the brake holds the wheel this step; a wheel
    # the brake stops with no torque to spare is held with 0 (at standstill:
    # an exact drive == brake tie). Pre-coupling value; not masked for airborne
    # wheels. Reset to None by the pipeline before the PRE_TIRE phase and set
    # before the POST_TIRE phase, so it is None in ``apply_pre_tire``. A
    # detached copy, READ-ONLY for hooks: the pipeline computes with its own
    # tensor, so a write changes nothing it computes -- but a later hook in the
    # list sees the written value. ``StaticFrictionLock`` engages on it.
    # Appended last so positional construction keeps its meaning.
    brake_rem: Optional[torch.Tensor] = None


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
#: ``_gear`` (v2.0.2): the engaged gear (+-1.0) of gear mode; a snapshot from
#: before 2.0.2 has no such key, so a load skips it and the row keeps +1.
RUNTIME_STATE_ROW_ATTRS = ("_stepped_once", "_gear")

#: Batch-wide scalars carried with the tensors.
RUNTIME_STATE_SCALARS = ("_prev_init",)

#: Rest-pose constants captured ONCE at construction, while the suspension is
#: still at zero. A rebuild re-captures them from a SAGGING vehicle, which is
#: why they are carried rather than recomputed (measured: z 0.3000 → 0.1499).
#: ``load_runtime_state`` SKIPS a ``None`` source: ``__init__`` leaves these
#: None when ``_capture_rest_wheel_pose`` raised, and overwriting the new
#: instance's good capture with that None would break the visual getters.
REST_POSE_ATTRS = ("_rest_wheel_pos_local", "_rest_wheel_quat_local")

#: Composite planar mass properties (``control.plant.PlantMassProperties``)
#: read ONCE, on the driver's first ``step()``, for the rigid-body deadbeat
#: scale (v1.8.0). Carried across a config rebuild like the rest pose (a
#: re-read mid-drive sees displaced suspension and turret links and differs in
#: the last bits); after the load ``wheel_meta.yaw_inertia`` is rebuilt from
#: them with the squared distances recomputed from the REBUILT
#: ``wheel_meta.positions``. ``load_runtime_state`` skips a ``None`` source (a
#: driver that never stepped).
MASS_PROPS_ATTRS = ("_mass_props",)


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
        # Planar mass properties for the deadbeat scale (v1.8.0), read on the
        # first step() by _ensure_yaw_inertia (the entity is built by then).
        self._mass_props = None
        # (v2.0.4) The wheel meta whose static-share read failed (attempted and
        # logged once), see _ensure_static_mass_share.
        self._static_share_failed_for = None
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
        except (AttributeError, TypeError, ValueError):
            # Narrow (v1.6.5): `base_link_idx` exists on genesis-world 1.5.0
            # (and did on 1.3.3 .. 1.4.5), so this is a fallback for an entity shape that lacks it or
            # returns something non-int — not a catch-all. The fallback is a
            # DEFINED behaviour (look the base link up by name), so it stays
            # silent; anything else must propagate.
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
        # (v2.0.2) Engaged gear of gear mode, +1.0 forward / -1.0 reverse, per
        # batch row (a batched kind's proto gets (NK,)). Starts forward; read
        # and changed only by a step whose input carries a gear (gear.py);
        # +1 again on reset(); carried across a rebuild
        # (RUNTIME_STATE_ROW_ATTRS). Public read: the ``gear`` property.
        self._gear = torch.ones(n_envs, device=self.dev, dtype=self.fdt)
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

    def set_ray_miss_value(self, miss: float, *,
                           sensor: Optional[Any] = None) -> None:
        """Tell this driver which distance means "the ray hit NOTHING".

        Normally read off ``sensor`` at construction. It has to be injectable
        because ``VehicleScene``'s ``dual_scene`` mode builds ``VehiclePhysics``
        with ``sensor=None`` (the rays live in a SEPARATE scene) and feeds
        ``step(distances=...)`` from there; the scene injects that sensor's
        value here. ``None`` raises ``ValueError`` since v1.8.0 (through 1.7.x
        it fell back to the deprecated ``raycast.RAY_MISS_THRESHOLD``, removed
        in 1.8.0).

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
            raise ValueError(
                "set_ray_miss_value(None): a driver needs the miss sentinel of "
                "the sensor its distances come from; RAY_MISS_THRESHOLD, the "
                "None fallback, was removed in 1.8.0. Pass the sensor's "
                "sentinel (genesis_vehicle.sensor_miss_value(sensor)).")
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

    @property
    def gear(self) -> Optional[torch.Tensor]:
        """(v2.0.2) The engaged gear of gear mode, ``(n_envs,)`` ``int8``
        (+1 forward, -1 reverse), a COPY. Meaningful only for a vehicle driven
        with gear inputs: a vehicle driven only with ``gear=None`` reads +1
        forever, whatever its throttle sign. ``None`` for a skid-steer vehicle,
        which refuses gear mode (``gear.refuses_gear``)."""
        if refuses_gear(self.resolved):
            return None
        return self._gear.detach().to(dtype=torch.int8).clone()

    def _ensure_yaw_inertia(self) -> None:
        """Compute ``wheel_meta.yaw_inertia`` once (v1.8.0).

        The inputs of the rigid-body deadbeat scale (``dynamics.YawInertia``):
        the composite yaw inertia about the centre of mass and each ray
        origin's squared horizontal distance from it, from
        ``control.plant.plant_mass_properties(self.entity)`` (a walk over every
        link's authored inertial; the base link's own inertia is rotated back
        from the principal axes genesis >= 1.4.0 stores it on). Called at the
        top of :meth:`step` and of ``MultiVehicleKindPhysics.step`` (on the
        kind's proto), so it runs on the FIRST step of every driver, before the
        first-step early return; afterwards it is one attribute test.

        Failure handling, exactly:

        * an entity with no ``links`` raises ``AttributeError`` naming the
          requirement (new in 1.8.0: through 1.7.0 the step path never read
          ``entity.links``, so a duck-typed non-genesis entity could be
          stepped; it now needs ``links`` or a ``wheel_meta.yaw_inertia`` set
          before the first step);
        * a root link exposing no ``desc``, or a ``desc`` missing one of its
          four inertial fields, raises ``AttributeError``
          (``_gs_compat.check_link_inertial_api``) — an unknown engine, not a
          massless vehicle;
        * a ``ValueError`` from the mass walk (no link carries mass) or from
          the inputs (yaw inertia about the centre of mass not positive) sets
          ``s = 1`` for this driver (``YawInertia(inf, ...)``: each wheel's
          deadbeat on its own mass share, unscaled) and logs ONE
          ``[genesis_vehicle:deadbeat-scale]`` WARNING naming
          ``self.vehicle_name``. Nothing else is caught.

        Carried mass properties (a config rebuild, ``MASS_PROPS_ATTRS``) are
        used instead of a fresh read."""
        wm = self.wheel_meta
        if wm.yaw_inertia is not None:
            return
        mp = self._mass_props
        try:
            if mp is None:
                from .control.plant import plant_mass_properties
                links = getattr(self.entity, "links", None)
                if not links:
                    raise AttributeError(
                        f"vehicle '{self.vehicle_name}': the entity "
                        f"({type(self.entity).__name__}) exposes no 'links'. "
                        "Since 1.8.0 VehiclePhysics reads the vehicle's mass "
                        "properties from entity.links on its first step (the "
                        "yaw-inertia inputs of the low-speed deadbeat bounds); "
                        "through 1.7.0 the step path did not need it. Pass a "
                        "genesis rigid entity, or set wheel_meta.yaw_inertia "
                        "(dynamics.yaw_inertia_from_mass_props) before the "
                        "first step.")
                check_link_inertial_api(links[0])
                mp = plant_mass_properties(self.entity)
            yi = yaw_inertia_from_mass_props(mp.mass, mp.com, mp.izz, wm.positions)
        except ValueError as exc:
            import logging
            logging.getLogger("genesis_vehicle").warning(
                "[genesis_vehicle:deadbeat-scale] vehicle '%s': could not build "
                "the yaw-inertia inputs of the low-speed deadbeat bounds (%s); "
                "using s = 1 (each wheel's own mass share, no rigid-body "
                "scale). A vehicle whose wheels sit far from its centre of "
                "mass relative to its yaw inertia can then alternate in yaw at "
                "low speed.", self.vehicle_name, exc)
            wm.yaw_inertia = YawInertia(
                float("inf"), torch.zeros_like(wm.positions[..., 0]))
            return
        self._mass_props = mp
        wm.yaw_inertia = yi

    def _ensure_static_mass_share(self) -> None:
        """Compute ``wheel_meta.static_mass_share`` once (v2.0.4).

        The input of ``StaticFrictionLock``'s per-wheel spring cap:
        ``dynamics.static_mass_shares`` (elastic-support statics) from the
        vehicle's mass and centre of mass and ``wheel_meta.positions`` /
        ``k_susp``. Its own computation with its own guard, called right after
        :meth:`_ensure_yaw_inertia` at the top of :meth:`step` and of
        ``MultiVehicleKindPhysics.step`` (on the proto), so it runs on the
        FIRST step, before the first-step early return, and also when
        ``wheel_meta.yaw_inertia`` was set by the caller.

        * Already set → nothing (one attribute test per step).
        * Mass properties: ``self._mass_props`` when :meth:`_ensure_yaw_inertia`
          (or a config rebuild) has them; otherwise one read of
          ``control.plant.plant_mass_properties(self.entity)``, NOT stored on
          the driver (so what :meth:`export_runtime_state` carries is
          unchanged).
        * Never raises: an entity without ``links``, an unknown inertial API
          (``AttributeError``) or a mass that is NaN, infinite or not positive
          (``ValueError``) leaves the share ``None`` — the lock then runs
          UNCAPPED — logs one ``[genesis_vehicle:sfl-cap]`` WARNING naming
          ``self.vehicle_name``, and records the failure on the driver
          (``_static_share_failed_for`` = this wheel meta), so later steps
          neither re-read the mass nor log again. A driver with a new wheel
          meta (a config rebuild builds a new driver) tries once more. An
          equal-share fallback (plan r204 condition C1, see
          ``dynamics.static_mass_shares``) logs one line and sets the share.
        * (Step D of 2.0.4) Also computes ``wheel_meta.static_rigid_factor``
          (``dynamics.rigid_share_factor``: beta of the cap's planar
          rigid-body term) from the same mass properties and the solved
          shares, written just before the share. Its ``ValueError`` is caught
          separately, after the share is solved: beta stays ``None``, the
          share IS set, the lock is capped per wheel only, and one
          ``[genesis_vehicle:sfl-cap]`` WARNING says "rigid-body factor
          unavailable". A share preset by the caller (the early return above)
          leaves beta ``None`` too. beta never reads
          ``wheel_meta.yaw_inertia``.
        """
        wm = self.wheel_meta
        if getattr(wm, "static_mass_share", None) is not None:
            return
        if getattr(self, "_static_share_failed_for", None) is wm:
            return
        import logging
        log = logging.getLogger("genesis_vehicle")
        mp = self._mass_props
        try:
            if mp is None:
                from .control.plant import plant_mass_properties
                links = getattr(self.entity, "links", None)
                if not links:
                    raise AttributeError(
                        f"the entity ({type(self.entity).__name__}) exposes no "
                        "'links'")
                check_link_inertial_api(links[0])
                mp = plant_mass_properties(self.entity)
            res = static_mass_shares(mp.mass, mp.com, wm.positions, wm.k_susp)
        except (AttributeError, ValueError) as exc:
            log.warning(
                "[genesis_vehicle:sfl-cap] vehicle '%s': no static load (%s); "
                "the StaticFrictionLock spring is uncapped.",
                self.vehicle_name, exc)
            log.debug("static mass share read failed", exc_info=True)
            self._static_share_failed_for = wm
            return
        if res.fallback is not None:
            log.warning(
                "[genesis_vehicle:sfl-cap] vehicle '%s': elastic-statics wheel "
                "shares rejected (%s); using equal shares M/n = %.1f kg for "
                "the StaticFrictionLock spring cap.",
                self.vehicle_name, res.fallback, float(res.shares[0]))
        share = torch.as_tensor(res.shares, device=wm.positions.device,
                                dtype=wm.positions.dtype)
        # (v2.0.4 Step D) The planar rigid-body factor beta, from the same
        # mass properties and the solved shares, in its OWN guard after the
        # share is solved: a ValueError here (e.g. a yaw inertia about the
        # centre of mass that is not positive) leaves beta None — the cap is
        # then per wheel only — and the share is still set.
        rigid = None
        try:
            rigid = rigid_share_factor(mp.mass, mp.com, mp.izz, wm.positions,
                                       res.shares)
        except ValueError as exc:
            log.warning(
                "[genesis_vehicle:sfl-cap] vehicle '%s': rigid-body factor "
                "unavailable (%s); the lock spring is capped per wheel only "
                "(C_MAX), the rigid-body term is not applied.",
                self.vehicle_name, exc)
            log.debug("rigid-body factor failed", exc_info=True)
        register_sfl_cap_name(wm, self.vehicle_name)
        wm.static_rigid_factor = rigid
        wm.static_mass_share = share

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
            self._gear.fill_(1.0)       # (v2.0.2) engaged gear back to forward
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
        self._gear[idx] = 1.0           # (v2.0.2) engaged gear back to forward
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
        :data:`RUNTIME_STATE_ROW_ATTRS`, :data:`RUNTIME_STATE_SCALARS`,
        :data:`REST_POSE_ATTRS` and (v1.8.0) :data:`MASS_PROPS_ATTRS` — the
        mass properties the deadbeat scale was built from, ``None`` if this
        driver never stepped — plus ``n_wheels`` / ``n_rows`` so
        :meth:`load_runtime_state` can reject a mismatched target. NOT in it:
        anything re-derived from the config (``resolved``, ``wheel_meta``,
        hooks, ``dt``, the ray-miss sentinel) — re-resolution is the point of a
        rebuild. ``wheel_meta.yaw_inertia`` is one such derived value: it is
        not carried itself, :meth:`load_runtime_state` rebuilds it from the
        carried mass properties and the NEW ``wheel_meta.positions``. (Through
        1.7.0 this said only "NOT in it: ... ``wheel_meta``"; the mass
        properties are the one ``wheel_meta`` input that is carried.)"""
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
        for name in MASS_PROPS_ATTRS:
            # PlantMassProperties: python floats + numpy arrays nobody mutates.
            state[name] = getattr(self, name, None)
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
        for name in MASS_PROPS_ATTRS:
            src = state.get(name)
            if src is None:
                continue
            setattr(self, name, src)
            # Rebuild the deadbeat-scale inputs from the CARRIED mass
            # properties and THIS driver's wheel positions (v1.8.0).
            self.wheel_meta.yaw_inertia = None
            self._ensure_yaw_inertia()

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
            ``dual_scene`` mode uses to feed distances measured in a SEPARATE
            raycast scene (see ``VehicleScene``'s module docstring for which
            ray-cast trees genesis-world 1.5.0 refits; through 2.0.4 this said
            that scene's BVH "is built once, never re-fit per step", a pre-1.5.0
            description). When ``None`` (default) the sensor
            is read exactly as before — fully backward compatible.
        """
        # (v2.0.2) Gear mode: read the request off the ORIGINAL object, before
        # any conversion (a custom InputType's from_unified may drop it), and
        # refuse a bad request before ANY state is written.
        gear_req = getattr(inputs, "gear", None)
        if gear_req is not None:
            check_vehicle_request(self.resolved, gear_req,
                                  getattr(inputs, "throttle", 0.0),
                                  where="VehiclePhysics.step", name=self.vehicle_name)

        # Deadbeat-scale inputs, once per driver, on the first call (the
        # entity is built by then); one attribute test afterwards.
        self._ensure_yaw_inertia()
        # (v2.0.4) StaticFrictionLock spring-cap input, same pattern, own guard.
        self._ensure_static_mass_share()

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
        req = (None if gear_req is None
               else encode_requests([gear_req], n_envs, dev, fdt))
        if req is None:
            T_drive_pw, T_brake_pw = self.resolved.drivetrain.distribute_torque(
                inputs, self.omega, wm, dev, fdt,
            )                                                # both (n_envs, n)
            self.last_T_drive = T_drive_pw
            self.last_T_brake = T_brake_pw

            throttle = self._scalar_to_envs(getattr(inputs, "throttle", 0.0))
        else:
            # (v2.0.2) Gear mode: the drivetrain gets gear * clamp(throttle,
            # 0, 1) on a NEW inputs object; T_drive stays local until it has
            # been clamped to the gear engaged AFTER this step's shift
            # (below), and only then becomes last_T_drive.
            throttle_req = self._scalar_to_envs(getattr(inputs, "throttle", 0.0))
            throttle = effective_throttle(throttle_req, req, self._gear)
            T_drive_pw, T_brake_pw = self.resolved.drivetrain.distribute_torque(
                with_throttle(inputs, throttle), self.omega, wm, dev, fdt,
            )
            self.last_T_brake = T_brake_pw
        brake = self._scalar_to_envs(getattr(inputs, "brake", 0.0))

        ctx = PipelineContext(throttle=throttle, brake=brake, wheel_meta=wm)

        # [RAYCAST] — distances may be injected by an external raycast source
        # (VehicleScene dual_scene mode reads them from a separate kinematic-terrain
        # raycast scene). When omitted, read this vehicle's own
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
            if req is not None:
                # No chassis state on this step, so no shift: clamp to the
                # gear engaged now.
                self.last_T_drive = clamp_drive_to_gear(T_drive_pw, req, self._gear)
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
        if req is not None:
            # (v2.0.2) Rule 4 on the planar speed, then the drive torque
            # clamped to the POST-shift gear (B1: a shift step must not apply
            # torque of the old gear's sign).
            v_xy = torch.hypot(vel[:, 0], vel[:, 1])
            self._gear, _ = update_gear(req, self._gear, throttle_req, v_xy)
            T_drive_pw = clamp_drive_to_gear(T_drive_pw, req, self._gear)
            self.last_T_drive = T_drive_pw
            ctx.gear = self._gear.clone()
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
        check_rolling_resistance_cr(wheels)
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
