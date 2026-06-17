"""Multi-vehicle batched physics (L2 vectorization).

The single-`VehiclePhysics`-per-entity pattern (the default since v0.1) is
ideal for the common cases — one driver vehicle per scene, or N parallel
rollouts via Genesis's `n_envs > 1` (L3 batching). It is NOT ideal when
you have K vehicles in the SAME scene at DIFFERENT positions (a traffic
scenario, an MPPI candidate visualization, a multi-agent benchmark): then
each vehicle needs its own `VehiclePhysics.step()` call, and the per-call
Python overhead + per-call CUDA kernel launches add up.

This module provides L2 vectorization: K vehicles of the SAME URDF / cfg
are processed in one batched compute pipeline. The batch dimension that
`VehiclePhysics` calls `n_envs` is reused as "n_vehicles_of_this_kind".
Different URDFs are still handled by separate per-kind batches (one
`MultiVehicleKindPhysics` per kind), dispatched by `MultiVehiclePhysics`.

What is batched
---------------
- Pacejka tire model: one call over (K, n_wheels) instead of K calls
- Stability hooks: one call per hook over (K, n_wheels) instead of K calls
- Slip / suspension / omega update: one batched op set
- Force + torque application: one `apply_links_external_force` call with K
  link indices instead of K separate calls

What is NOT batched
-------------------
- Per-entity raycaster reads (one `sensor.read()` per vehicle, then stack —
  small Python overhead that dominates only at K = 1-2)
- Per-entity visual joint updates (`set_dofs_position` per vehicle —
  Python loop over K, acceptable)
- Cross-kind dispatch (one Python iteration per vehicle kind in the
  outer `MultiVehiclePhysics`)

When to use
-----------
- Multi-agent traffic / MPPI candidate fleets visible in one scene
- Cross-vehicle comparison (FWD vs RWD on the same road)
- Anything where K vehicles of the same kind share one Genesis scene

For pure RL / MPPI throughput where you only need many independent
rollouts of one vehicle (no shared scene), `n_envs > 1` (L3) is still
the right tool — see `samples/perf_vectorization.py`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import torch
import genesis as gs
from genesis.utils.geom import transform_by_quat

from .config import VehicleConfig
from .core import (
    VehiclePhysics, PipelineContext, VisualPartsTransforms,
    _quat_axis_angle, _quat_mul, _susp_visual_offset,
)
from ._pipeline import compute_wheel_step
from .inputs import VehicleInputs, VehicleStepInputs
from .raycast import read_distances
from .visual import VisualJointSync


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Active rotation of `vec` by unit quaternion `quat` (w,x,y,z).
    Both tensors flat-batched (N, 4) and (N, 3)."""
    w, x, y, z = quat[:, 0:1], quat[:, 1:2], quat[:, 2:3], quat[:, 3:4]
    vx, vy, vz = vec[:, 0:1], vec[:, 1:2], vec[:, 2:3]
    # q * v * q_conj, expanded
    rx = (w*w + x*x - y*y - z*z) * vx + 2 * (x*y - w*z) * vy + 2 * (x*z + w*y) * vz
    ry = 2 * (x*y + w*z) * vx + (w*w - x*x + y*y - z*z) * vy + 2 * (y*z - w*x) * vz
    rz = 2 * (x*z - w*y) * vx + 2 * (y*z + w*x) * vy + (w*w - x*x - y*y + z*z) * vz
    return torch.cat([rx, ry, rz], dim=-1)


class MultiVehicleKindPhysics:
    """Batched physics driver for K vehicles of the SAME URDF / cfg, optionally
    across N parallel Genesis envs.

    Combined batch size is ``N * K`` (Genesis envs × vehicles per env). All
    compute (Pacejka, hooks, suspension, omega) runs once on shape
    ``(N*K, n_wheels)`` tensors; entity / sensor I/O is a single batched
    solver call per quantity per kind.

      n_envs = 1, K > 1   → L2 (same-scene multi-vehicle).
      n_envs > 1, K = 1   → L3 (parallel envs, but use plain VehiclePhysics
                                directly — this class adds no value over it).
      n_envs > 1, K > 1   → L2 + L3 combined. The headline use case is
                            "M parallel scenarios, each with K-vehicle
                            traffic around an ego" for autonomous-driving
                            MPPI / RL.

    Use ``MultiVehiclePhysics`` to dispatch across multiple kinds.
    """

    def __init__(
        self,
        scene: Any,
        entities: Sequence[Any],
        sensors: Sequence[Any],
        config: VehicleConfig,
        n_envs: int = 1,
    ):
        assert len(entities) == len(sensors) and len(entities) >= 1
        assert n_envs >= 1
        K = len(entities)
        NK = n_envs * K

        # Proto VehiclePhysics with batch dim = N * K. All its internal tensors
        # (omega, prev_compression, _wheel_body_b, _up_world, …) come out at
        # shape (NK, n_wheels) or (NK, n_wheels, 3), which is the flat-batch
        # form the compute pipeline operates on.
        self._proto = VehiclePhysics(scene, entities[0], sensors[0], config, n_envs=NK)

        self.entities = list(entities)
        self.sensors = list(sensors)
        self.K = K
        self.n_envs = n_envs
        self.NK = NK
        self.scene = scene
        self.solver = scene.sim.rigid_solver
        self.dev = self._proto.dev
        self.fdt = self._proto.fdt
        # _proto.dt was already resolved against this scene (proto is built
        # with the same scene), so just inherit it.
        self.dt = self._proto.dt

        # Re-resolve base link indices: K different links, one per entity.
        # (Same K indices apply to every Genesis env — n_envs is implicit.)
        base_name = self._proto.resolved.chassis.base_link_name
        base_idx_list = []
        for e in entities:
            try:
                base_idx_list.append(int(e.base_link_idx))
            except Exception:
                base_link = [l for l in e.links if l.name == base_name][0]
                base_idx_list.append(int(base_link.idx))
        self.base_idx_list = base_idx_list
        self.base_idx_tensor = torch.tensor(
            base_idx_list, dtype=torch.long, device=self.dev,
        )

        # Replace the proto's VisualJointSync with K per-entity VisualJointSync objects,
        # each one bound to its own entity and built for the actual Genesis
        # n_envs. Compute output gets sliced (N, n_wheels) per entity for
        # each visual.step(). (See the [VISUAL] block at the bottom of step().)
        self._proto.visual = None
        self.visuals: list[VisualJointSync] = []
        if self._proto.resolved.enable_visual_joint_sync:
            for ent in entities:
                self.visuals.append(VisualJointSync(
                    entity=ent, resolved=self._proto.resolved,
                    n_envs=n_envs, device=self.dev, dtype=self.fdt,
                ))

    # ------------------------------------------------------------------
    # Expose proto attributes for callers (omega, last_*, etc.)
    # ------------------------------------------------------------------
    @property
    def omega(self) -> torch.Tensor:
        return self._proto.omega

    @property
    def wheel_meta(self):
        return self._proto.wheel_meta

    @property
    def resolved(self):
        return self._proto.resolved

    def reset(self, vehicle_ids=None) -> None:
        self._proto.reset(vehicle_ids)

    # ------------------------------------------------------------------
    # Batched I/O — replaces the per-entity reads/writes in VehiclePhysics.step
    # ------------------------------------------------------------------
    def _read_state_batched(self) -> tuple[torch.Tensor, torch.Tensor,
                                            torch.Tensor, torch.Tensor]:
        """Read state for all N×K base links via the rigid solver's multi-link
        batched API. Returns ``(pos, quat, vel, ang)``, each flat-batched
        shape ``(NK, 3)`` / ``(NK, 4)`` — row-major over envs then vehicles
        (env 0 vehicle 0, env 0 vehicle 1, ..., env 0 vehicle K-1, env 1 v0, ...)."""
        # get_links_pos returns (n_envs, K, 3) when n_envs >= 1.
        pos  = self.solver.get_links_pos(self.base_idx_tensor)
        quat = self.solver.get_links_quat(self.base_idx_tensor)
        vel  = self.solver.get_links_vel(self.base_idx_tensor)
        ang  = self.solver.get_links_ang(self.base_idx_tensor)
        return (pos.reshape(self.NK, 3),
                quat.reshape(self.NK, 4),
                vel.reshape(self.NK, 3),
                ang.reshape(self.NK, 3))

    def _read_distances_batched(self) -> torch.Tensor:
        """Stack K raycaster reads into a flat ``(NK, n_wheels)`` tensor.

        Each sensor returns ``(n_envs, n_wheels)``. We stack across K along
        dim 1 → ``(N, K, n_wheels)`` → flatten → ``(NK, n_wheels)`` with the
        same env-major / vehicle-minor row ordering as the state reads."""
        out = []
        for s in self.sensors:
            d = read_distances(s, n_envs=self.n_envs)   # (N, n_wheels)
            out.append(d)
        # stack on dim 1 to interleave [N, K, n_wheels] then flatten
        stacked = torch.stack(out, dim=1)               # (N, K, n_wheels)
        return stacked.reshape(self.NK, -1)

    def _apply_force_torque_batched(self, total_F: torch.Tensor,
                                     total_T: torch.Tensor) -> None:
        """Apply per-vehicle base-link force + torque in ONE batched solver call.

        Compute outputs are flat ``(NK, 3)``. We reshape to ``(N, K, 3)`` to
        match what ``apply_links_external_force`` expects when the scene is
        built with n_envs=N and we target K different links per env."""
        F_NK3 = total_F.reshape(self.n_envs, self.K, 3)
        T_NK3 = total_T.reshape(self.n_envs, self.K, 3)
        self.solver.apply_links_external_force(F_NK3, self.base_idx_tensor)
        self.solver.apply_links_external_torque(T_NK3, self.base_idx_tensor)

    # ------------------------------------------------------------------
    # The step pipeline. Mirrors VehiclePhysics.step but with batched I/O.
    # ------------------------------------------------------------------
    def step(self, inputs_list: Sequence[VehicleStepInputs]) -> None:
        """Step K vehicles in one batched compute pipeline.

        Args
        ----
        inputs_list : sequence of K ``VehicleInputs`` (or steering-specific
                      inputs). Each vehicle's throttle / brake / steer
                      become element-k of the batched (K,) tensor.
        """
        assert len(inputs_list) == self.K, (
            f"expected K={self.K} inputs, got {len(inputs_list)}")
        p = self._proto
        K = self.K
        N = self.n_envs
        NK = self.NK
        n = p.wheel_meta.n_wheels
        dev, fdt = p.dev, p.fdt
        DT = p.dt
        wm = p.wheel_meta

        # Coerce per-vehicle inputs into a flat-batched tensor of length NK.
        # Each inputs_list[k] is a VehicleInputs whose throttle/brake/steer is
        # either a scalar (same across all N envs) or a shape-(N,) tensor.
        # Stack K → (N, K) → flatten in env-major order → (NK,).
        def _to_NK(attr: str) -> torch.Tensor:
            per_k = []
            for i in inputs_list:
                v = getattr(i, attr, 0.0)
                if torch.is_tensor(v):
                    t = v.to(device=dev, dtype=fdt)
                    if t.dim() == 0:
                        t = t.expand(N)
                    assert t.shape == (N,), (
                        f"input '{attr}' tensor must be shape ({N},), got {tuple(t.shape)}")
                else:
                    t = torch.full((N,), float(v), device=dev, dtype=fdt)
                per_k.append(t)
            return torch.stack(per_k, dim=1).reshape(NK)   # (N, K) → (NK,)

        throttle_t = _to_NK("throttle")
        brake_t    = _to_NK("brake")
        steer_t    = _to_NK("steer")

        steering = p.resolved.steering
        batched_in = VehicleInputs(throttle=throttle_t, brake=brake_t, steer=steer_t)
        batched_in = steering.InputType.from_unified(batched_in)

        steer_per_wheel = steering.per_wheel_steer(
            batched_in, NK, wm, dev, fdt,
        )                                                # (NK, n)
        T_drive_pw, T_brake_pw = p.resolved.drivetrain.distribute_torque(
            batched_in, p.omega, wm, dev, fdt,
        )
        p.last_T_drive = T_drive_pw
        p.last_T_brake = T_brake_pw

        ctx = PipelineContext(
            throttle=throttle_t, brake=brake_t, wheel_meta=wm,
        )

        # [RAYCAST]
        distances = self._read_distances_batched()       # (K, n)
        p.last_distances = distances.detach().clone()
        if not p._prev_init and torch.all(distances < 1e-6):
            p._prev_init = True
            return

        # [CHASSIS STATE]  — single batched solver read for K vehicles.
        pos, quat, vel, ang = self._read_state_batched()
        ctx.vel = vel
        ctx.ang = ang
        for hook in p.pre_loop_hooks:
            hook.apply_pre_loop(ctx)

        # [PIPELINE] — sections A–E shared with VehiclePhysics.step via
        # _pipeline.compute_wheel_step (batch dim = NK here). Single source
        # of the math, so fixes like the F_long overshoot clamp apply to
        # both single- and multi-vehicle paths.
        res = compute_wheel_step(
            pos=pos, quat=quat, vel=vel, ang=ang, distances=distances,
            steer_per_wheel=steer_per_wheel,
            T_drive_pw=T_drive_pw, T_brake_pw=T_brake_pw, omega=p.omega,
            prev_compression=p.prev_compression, prev_init=p._prev_init,
            wheel_meta=wm, resolved=p.resolved,
            wheel_body_b=p._wheel_body_b, up_world=p._up_world,
            post_tire_hooks=p.post_tire_hooks, ctx=ctx,
            dt=DT, B=NK, n=n,
        )
        p.prev_compression = res.compression.detach().clone()
        p.omega = res.new_omega
        total_F, total_T = res.total_F, res.total_T

        # Visual-pose bookkeeping on the proto (so wheel_visual_transforms works
        # for the multi-vehicle path too — same as VehiclePhysics.step).
        p.last_steer_per_wheel = steer_per_wheel
        two_pi = 2.0 * math.pi
        p.wheel_spin_angle = (
            (p.wheel_spin_angle + p.omega * DT) + math.pi
        ) % two_pi - math.pi

        # Diagnostics.
        p.last_N = res.N; p.last_F_long = res.F_long; p.last_F_lat = res.F_lat
        p.last_compression = res.compression
        p.last_kappa = res.kappa; p.last_alpha = res.alpha

        # [APPLY]  — single batched solver call for K vehicles.
        self._apply_force_torque_batched(total_F, total_T)
        p._prev_init = True
        p._stepped_once = True

        # [VISUAL] — per-entity Python loop over K VisualJointSync objects, each
        # built with n_envs=N. Compute outputs are flat (NK, n_wheels); we
        # reshape to (N, K, n_wheels) and slice the k-th vehicle's slab
        # ((N, n_wheels)) to feed its visual.step(). With N=1 this is the
        # original L2-only path; with N>1 each visual gets a real n_envs-
        # batched update.
        if self.visuals:
            # NB: 'N' got reassigned earlier in section (A) to the per-wheel
            # normal-force tensor — use the explicit self.n_envs / self.K
            # here instead of the local names.
            n_envs = self.n_envs
            steer_NK = steer_per_wheel.reshape((n_envs, K, n))
            dist_NK  = distances.reshape((n_envs, K, n))
            omega_NK = p.omega.reshape((n_envs, K, n))
            for k_i, vis in enumerate(self.visuals):
                vis.step(
                    steer_NK[:, k_i, :].contiguous(),
                    dist_NK[:, k_i, :].contiguous(),
                    omega_NK[:, k_i, :].contiguous(),
                    DT,
                )

    def wheel_visual_transforms(self, frame: str = "world"):
        """Closed-form wheel visual poses for this kind's K vehicles (× n_envs),
        the multi-vehicle analogue of ``VehiclePhysics.wheel_visual_transforms``.

        Returns ``(pos, quat)`` shaped ``(n_envs, K, n_wheels, 3)`` and
        ``(n_envs, K, n_wheels, 4)`` (env-major). VisualJointSync-independent;
        steer + suspension + spin baked in (spin honors ``visual_spin_enabled``).
        ``frame="local"`` is relative to each vehicle's chassis, ``"world"``
        absolute."""
        if frame not in ("world", "local"):
            raise ValueError(f"frame must be 'world' or 'local', got {frame!r}")
        p = self._proto
        NK, N, K = self.NK, self.n_envs, self.K
        n = p.wheel_meta.n_wheels
        if p._rest_wheel_pos_local is None:
            # All K vehicles share the kind's URDF → rest pose (relative to base)
            # is identical; capture from the first entity.
            p._capture_rest_wheel_pose(self.entities[0])
        rest_pos = p._rest_wheel_pos_local.unsqueeze(0)            # (1, n, 3)
        rest_quat = p._rest_wheel_quat_local.unsqueeze(0)          # (1, n, 4)

        if not p._stepped_once:
            local_pos = rest_pos.expand(NK, n, 3).contiguous()
            local_quat = rest_quat.expand(NK, n, 4).contiguous()
        else:
            steer_z = -p.last_steer_per_wheel                      # (NK, n)
            susp_off = _susp_visual_offset(
                p.last_distances, p._mesh_radius, p._l_susp, p._susp_clamp)  # (NK, n)
            spin = (p.wheel_spin_angle if p._visual_spin_enabled
                    else torch.zeros_like(p.wheel_spin_angle))
            z_off = torch.stack(
                [torch.zeros_like(susp_off), torch.zeros_like(susp_off), susp_off],
                dim=-1)
            local_pos = rest_pos + z_off                           # (NK, n, 3)
            local_quat = _quat_mul(
                rest_quat,
                _quat_mul(_quat_axis_angle("z", steer_z), _quat_axis_angle("y", spin)),
            )

        if frame == "local":
            return local_pos.reshape(N, K, n, 3), local_quat.reshape(N, K, n, 4)

        # World: compose with each vehicle's chassis (base-link) pose.
        pos, quat, _vel, _ang = self._read_state_batched()         # (NK, 3), (NK, 4)
        cqb = quat.unsqueeze(1).expand(NK, n, 4)
        world_pos = pos.unsqueeze(1) + transform_by_quat(
            local_pos.reshape(NK * n, 3), cqb.reshape(NK * n, 4)
        ).reshape(NK, n, 3)
        world_quat = _quat_mul(cqb, local_quat)
        return world_pos.reshape(N, K, n, 3), world_quat.reshape(N, K, n, 4)


def group_vehicles_by_cfg(vehicles: Sequence[tuple]):
    """Group a flat vehicle list by cfg identity (same Python object → same kind).

    Pure function (no Genesis objects touched — only ``veh[2]`` identity is
    read), so the grouping/dispatch bookkeeping that ``MultiVehiclePhysics``
    depends on can be unit-tested without a GPU. Returns:

    - ``group_order``: list of cfg-id keys in first-seen order (= kind order)
    - ``groups``: dict key → list of the vehicles in that kind (caller order
      within the kind preserved)
    - ``flat_to_kind``: list of ``(flat_i, kind_idx, slot_idx)``, sorted by
      ``flat_i`` (the caller's flat index). Maps ``inputs_list[flat_i]`` →
      kind ``kind_idx`` slot ``slot_idx``.
    """
    groups: dict = defaultdict(list)
    group_order: list = []
    for veh in vehicles:
        cfg = veh[2]
        key = id(cfg)
        if key not in groups:
            group_order.append(key)
        groups[key].append(veh)

    vehicles_list = list(vehicles)
    flat_to_kind = []
    for k_i, key in enumerate(group_order):
        for slot_i, veh in enumerate(groups[key]):
            flat_i = next(j for j, v in enumerate(vehicles_list) if v is veh)
            flat_to_kind.append((flat_i, k_i, slot_i))
    flat_to_kind.sort(key=lambda t: t[0])
    return group_order, groups, flat_to_kind


def rebucket_inputs(inputs_list: Sequence, flat_to_kind, kind_sizes):
    """Scatter a flat (caller-order) ``inputs_list`` into per-kind slot lists.

    Pure function (no Genesis). ``per_kind[kind_idx][slot_idx] =
    inputs_list[flat_i]`` for each mapping entry. ``kind_sizes[kind_idx]`` is
    the number of slots (K) in that kind. Returns a list (length n_kinds) of
    lists (length K_kind)."""
    per_kind = [[None] * ks for ks in kind_sizes]
    for flat_i, kind_idx, slot_idx in flat_to_kind:
        per_kind[kind_idx][slot_idx] = inputs_list[flat_i]
    return per_kind


class MultiVehiclePhysics:
    """Top-level multi-vehicle driver: groups vehicles by URDF / cfg and
    runs one ``MultiVehicleKindPhysics`` per kind. K vehicles of the same
    kind share a batched compute pipeline; M different kinds dispatch in
    a small (length-M) Python loop.

    Typical usage
    -------------
    ```python
    from genesis_vehicle import MultiVehiclePhysics, VehicleInputs

    # Build the scene with K vehicles of one or more kinds — same flow as
    # before, but you keep the (entity, sensor, cfg) tuples in a list.
    vehicles = []  # list of (entity, sensor, cfg)
    for kind in kinds:
        for j in range(K):
            ent, sens, _ = add_vehicle(scene, kind.urdf, preset_fn=None, ...)
            vehicles.append((ent, sens, kind.cfg))

    scene.build(n_envs=1)
    mphys = MultiVehiclePhysics(scene, vehicles)
    for step in range(...):
        mphys.step([VehicleInputs(...) for _ in vehicles])
        scene.step()
    ```
    """

    def __init__(self, scene: Any,
                 vehicles: Sequence[tuple],    # list of (entity, sensor, cfg)
                 n_envs: int = 1):
        if not vehicles:
            raise ValueError("MultiVehiclePhysics needs at least one vehicle.")
        # Group by cfg identity (same Python object → same kind). Callers
        # who want grouping by URDF *value* can pass the same cfg instance
        # for matching vehicles (the bundled presets return fresh cfgs per
        # call, so identity-by-call works — see road_loop.py). The grouping
        # bookkeeping is a pure function (group_vehicles_by_cfg) so it is
        # unit-tested without Genesis; only kind construction below needs GPU.
        group_order, groups, self._flat_to_kind = group_vehicles_by_cfg(vehicles)

        self.vehicles = list(vehicles)   # preserve caller's order
        self.n_envs = n_envs
        self.kinds = []                  # list of MultiVehicleKindPhysics
        self.kind_slices: list[slice] = []  # per-kind slice into flat inputs
        for key in group_order:
            kind_vehicles = groups[key]
            entities = [v[0] for v in kind_vehicles]
            sensors  = [v[1] for v in kind_vehicles]
            cfg      = kind_vehicles[0][2]
            self.kinds.append(
                MultiVehicleKindPhysics(scene, entities, sensors, cfg, n_envs=n_envs))

    @property
    def n_vehicles(self) -> int:
        return len(self.vehicles)

    @property
    def n_kinds(self) -> int:
        return len(self.kinds)

    def step(self, inputs_list: Sequence[VehicleStepInputs]) -> None:
        """Step all vehicles in batched per-kind pipelines.

        Args
        ----
        inputs_list : flat list of length ``n_vehicles``, in the SAME order
                      as the ``vehicles`` argument to ``__init__``.
        """
        assert len(inputs_list) == self.n_vehicles, (
            f"MultiVehiclePhysics expected {self.n_vehicles} inputs, "
            f"got {len(inputs_list)}")
        # Re-bucket inputs into per-kind slot order (pure helper — unit-tested).
        per_kind = rebucket_inputs(
            inputs_list, self._flat_to_kind, [k.K for k in self.kinds])
        # Dispatch.
        for kind, ins in zip(self.kinds, per_kind):
            kind.step(ins)

    def wheel_visual_transforms(self, frame: str = "world"):
        """Closed-form wheel visual poses for every vehicle, in the caller's flat
        order. Returns a list (length ``n_vehicles``) of ``(pos, quat)`` tuples,
        each ``(n_envs, n_wheels, 3)`` / ``(n_envs, n_wheels, 4)``. Per-vehicle
        because kinds may differ in wheel count. VisualJointSync-independent."""
        kind_out = [k.wheel_visual_transforms(frame) for k in self.kinds]   # (N,K,n,·)
        out = [None] * self.n_vehicles
        for flat_i, kind_idx, slot_idx in self._flat_to_kind:
            wp, wq = kind_out[kind_idx]
            out[flat_i] = (wp[:, slot_idx], wq[:, slot_idx])
        return out

    def visual_parts_transforms(self, frame: str = "world"):
        """One-call render feed per vehicle (chassis + wheels), caller flat order.
        Returns a list (length ``n_vehicles``) of ``VisualPartsTransforms``.
        Chassis = real dynamics pose; wheels = closed-form visual pose."""
        wheels = self.wheel_visual_transforms(frame)
        kind_of = {flat_i: kind_idx for flat_i, kind_idx, _ in self._flat_to_kind}
        out = []
        for i, (ent, _sensor, _cfg) in enumerate(self.vehicles):
            cpos = ent.get_pos(); cquat = ent.get_quat()
            if cpos.dim() == 1:
                cpos = cpos.unsqueeze(0); cquat = cquat.unsqueeze(0)
            names = [w.name for w in self.kinds[kind_of[i]].resolved.wheels]
            wp, wq = wheels[i]
            out.append(VisualPartsTransforms(
                frame=frame, chassis_pos=cpos, chassis_quat=cquat,
                wheel_names=names, wheel_pos=wp, wheel_quat=wq))
        return out
