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

from collections import defaultdict
from typing import Any, Sequence

import torch
import genesis as gs

from genesis.utils.geom import transform_by_quat

from .config import VehicleConfig
from .core import VehiclePhysics, PipelineContext
from .dynamics import brake_torque_signed
from .inputs import VehicleInputs, VehicleStepInputs
from .raycast import read_distances
from .visual import VisualSync


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
    """One batched physics driver for K vehicles of the SAME URDF / cfg.

    Use ``MultiVehiclePhysics`` instead if you have multiple vehicle kinds
    — it wraps one of these per kind.
    """

    def __init__(
        self,
        scene: Any,
        entities: Sequence[Any],
        sensors: Sequence[Any],
        config: VehicleConfig,
    ):
        assert len(entities) == len(sensors) and len(entities) >= 1
        K = len(entities)

        # Reuse VehiclePhysics's setup by constructing a "prototype" instance
        # with n_envs=K so wheel_meta, resolved config, hook lists, omega
        # tensor (shape (K, n_wheels)), pre-built broadcasts, etc. all come
        # out correctly sized. The prototype's `entity` / `sensor` are the
        # first-of-K; we override the I/O paths below.
        self._proto = VehiclePhysics(scene, entities[0], sensors[0], config, n_envs=K)

        # Override the entity / sensor handles with the full K-sized lists.
        self.entities = list(entities)
        self.sensors = list(sensors)
        self.K = K
        self.scene = scene
        self.solver = scene.sim.rigid_solver
        self.dev = self._proto.dev
        self.fdt = self._proto.fdt
        self.dt = self._proto.dt

        # Re-resolve base link indices: K different links, one per entity.
        base_name = self._proto.resolved.chassis.base_link_name
        base_idx_list = []
        for e in entities:
            try:
                base_idx_list.append(int(e.base_link_idx))
            except Exception:
                base_link = [l for l in e.links if l.name == base_name][0]
                base_idx_list.append(int(base_link.idx))
        self.base_idx_list = base_idx_list   # Python list of K ints
        self.base_idx_tensor = torch.tensor(
            base_idx_list, dtype=torch.long, device=self.dev,
        )

        # The proto's _wheel_body_b / _up_world were built for n_envs=K; we
        # reuse them as-is (shape (K, n_wheels, 3) / (K, 3)).

        # Replace the proto's VisualSync (which was built for n_envs=K and
        # would issue a shape-(K, n_dofs) set_dofs_position into a scene
        # that actually has n_envs=1 — Genesis would reject it) with K
        # per-entity VisualSync objects, each one bound to its own entity
        # and built for n_envs=1. Compute is still batched (K, n_wheels);
        # the visual writes are a Python loop over K (each call is a small
        # set_dofs_position, negligible vs the compute saving).
        self._proto.visual = None
        self.visuals: list[VisualSync] = []
        if self._proto.resolved.enable_visual_sync:
            for ent in entities:
                self.visuals.append(VisualSync(
                    entity=ent, resolved=self._proto.resolved,
                    n_envs=1, device=self.dev, dtype=self.fdt,
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
        """Read K base-link poses + velocities via the rigid solver's
        multi-link batched API. Returns (pos, quat, vel, ang), each shaped
        ``(K, 3)`` for pos/vel/ang and ``(K, 4)`` for quat."""
        # `get_links_pos` returns (n_envs, n_links_queried, 3); with the
        # scene built at n_envs=1, the first dim is 1 — squeeze it.
        pos  = self.solver.get_links_pos(self.base_idx_tensor)[0]    # (K, 3)
        quat = self.solver.get_links_quat(self.base_idx_tensor)[0]   # (K, 4)
        vel  = self.solver.get_links_vel(self.base_idx_tensor)[0]    # (K, 3)
        ang  = self.solver.get_links_ang(self.base_idx_tensor)[0]    # angular VELOCITY (K, 3)
        return pos, quat, vel, ang

    def _read_distances_batched(self) -> torch.Tensor:
        """Stack K raycaster reads into a (K, n_wheels) tensor."""
        out = []
        for s in self.sensors:
            d = read_distances(s, n_envs=1)     # (1, n_wheels)
            out.append(d[0])
        return torch.stack(out, dim=0)          # (K, n_wheels)

    def _apply_force_torque_batched(self, total_F: torch.Tensor,
                                     total_T: torch.Tensor) -> None:
        """Apply per-vehicle base-link force + torque in ONE solver call each.

        Shape: ``(1, K, 3)`` — scene has 1 env, K target links.
        """
        self.solver.apply_links_external_force(
            total_F.unsqueeze(0), self.base_idx_tensor,
        )
        self.solver.apply_links_external_torque(
            total_T.unsqueeze(0), self.base_idx_tensor,
        )

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
        assert len(inputs_list) == self.K
        p = self._proto
        K = self.K
        n = p.wheel_meta.n_wheels
        dev, fdt = p.dev, p.fdt
        DT = p.dt
        wm = p.wheel_meta

        # Coerce per-vehicle inputs into batched tensors.
        steering = p.resolved.steering
        # Build a synthetic batched input compatible with the strategy.
        throttle_t = torch.tensor(
            [float(getattr(i, "throttle", 0.0)) for i in inputs_list],
            device=dev, dtype=fdt,
        )
        brake_t = torch.tensor(
            [float(getattr(i, "brake", 0.0)) for i in inputs_list],
            device=dev, dtype=fdt,
        )
        steer_t = torch.tensor(
            [float(getattr(i, "steer", 0.0)) for i in inputs_list],
            device=dev, dtype=fdt,
        )
        # Strategies expect a single VehicleInputs-like object whose
        # throttle/brake/steer are (K,) tensors (matching the n_envs=K
        # contract).
        batched_in = VehicleInputs(throttle=throttle_t, brake=brake_t, steer=steer_t)
        batched_in = steering.InputType.from_unified(batched_in)

        steer_per_wheel = steering.per_wheel_steer(
            batched_in, K, wm, dev, fdt,
        )                                                # (K, n)
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

        # World-space wheel positions.
        quat_b_flat = quat.unsqueeze(1).expand(K, n, 4).reshape(K * n, 4)
        wheel_body_flat = p._wheel_body_b.reshape(K * n, 3)
        wheel_world = (transform_by_quat(wheel_body_flat, quat_b_flat)
                       .reshape(K, n, 3) + pos.unsqueeze(1))

        # (A) Suspension + damper + N.
        compression = torch.clamp(wm.rest_d.unsqueeze(0) - distances, min=0.0)
        air_mask = compression <= 0
        if p._prev_init:
            raw_rate = (compression - p.prev_compression) / DT
            rc = wm.comp_rate_clamp.unsqueeze(0)
            comp_rate = torch.clamp(raw_rate, -rc, rc)
        else:
            comp_rate = torch.zeros_like(compression)
        p.prev_compression = compression.detach().clone()

        c_damp = torch.where(
            comp_rate > 0.0,
            wm.c_compression.unsqueeze(0).expand_as(comp_rate),
            wm.c_extension.unsqueeze(0).expand_as(comp_rate),
        )
        N = wm.k_susp.unsqueeze(0) * compression + c_damp * comp_rate
        N = torch.clamp(N, min=0.0)
        N = torch.where(air_mask, torch.zeros_like(N), N)

        # (B) Wheel-frame fwd / lat.
        cs = torch.cos(steer_per_wheel)
        ss = torch.sin(steer_per_wheel)
        zer = torch.zeros_like(cs)
        wheel_fwd_local = torch.stack([cs, -ss, zer], dim=-1)
        wheel_lat_local = torch.stack([ss,  cs, zer], dim=-1)
        wheel_fwd_world = transform_by_quat(
            wheel_fwd_local.reshape(K * n, 3), quat_b_flat
        ).reshape(K, n, 3)
        wheel_lat_world = transform_by_quat(
            wheel_lat_local.reshape(K * n, 3), quat_b_flat
        ).reshape(K, n, 3)

        r_vec = wheel_world - pos.unsqueeze(1)
        ang_b = ang.unsqueeze(1).expand(K, n, 3)
        v_hit = vel.unsqueeze(1) + torch.cross(ang_b, r_vec, dim=-1)
        v_long = (v_hit * wheel_fwd_world).sum(dim=-1)
        v_lat  = (v_hit * wheel_lat_world).sum(dim=-1)
        v_roll = wm.radius.unsqueeze(0) * p.omega

        # (C) Tire force.
        F_long, F_lat, kappa, alpha = p.resolved.tire(
            v_long, v_lat, v_roll, N, wm,
        )

        # POST_TIRE hooks.
        ctx.F_long = F_long; ctx.F_lat = F_lat; ctx.N = N
        ctx.v_long = v_long; ctx.v_lat = v_lat
        ctx.omega = p.omega; ctx.air_mask = air_mask
        ctx.omega_override = None
        ctx.omega_pull_factor = None; ctx.omega_pull_target = None
        ctx.dt = float(DT)
        for hook in p.post_tire_hooks:
            hook.apply_post_tire(ctx)
        F_long, F_lat = ctx.F_long, ctx.F_lat

        # (D) Omega update.
        T_brake_eff = brake_torque_signed(T_brake_pw, p.omega)
        i_w = wm.i_wheel.unsqueeze(0)
        radius_b = wm.radius.unsqueeze(0)
        domega = (T_drive_pw - T_brake_eff - radius_b * F_long) / i_w
        new_omega = p.omega + domega * DT
        # Air mask: drive/brake torque still applies in air, but no tire force.
        domega_air = (T_drive_pw - T_brake_eff) / i_w
        new_omega_air = p.omega + domega_air * DT
        new_omega = torch.where(air_mask, new_omega_air, new_omega)
        omega_max = float(p.resolved.chassis.omega_max)
        new_omega = torch.clamp(new_omega, -omega_max, omega_max)

        if ctx.omega_pull_factor is not None and ctx.omega_pull_target is not None:
            new_omega = (new_omega * (1.0 - ctx.omega_pull_factor)
                         + ctx.omega_pull_target * ctx.omega_pull_factor)
        if ctx.omega_override is not None:
            new_omega = torch.where(
                ctx.omega_override.active, torch.zeros_like(new_omega), new_omega,
            )
        p.omega = new_omega

        # (E) Force accumulation.
        F_world = (
            N.unsqueeze(-1) * p._up_world.unsqueeze(1)
            + F_long.unsqueeze(-1) * wheel_fwd_world
            + F_lat.unsqueeze(-1) * wheel_lat_world
        )
        torque = torch.cross(r_vec, F_world, dim=-1)
        total_F = F_world.sum(dim=1)            # (K, 3)
        total_T = torque.sum(dim=1)

        # Diagnostics.
        p.last_N = N; p.last_F_long = F_long; p.last_F_lat = F_lat
        p.last_compression = compression
        p.last_kappa = kappa; p.last_alpha = alpha

        # [COUPLING]
        p.omega = p.resolved.coupling.apply(p.omega, wm)

        # [APPLY]  — single batched solver call for K vehicles.
        self._apply_force_torque_batched(total_F, total_T)
        p._prev_init = True

        # [VISUAL] — per-entity Python loop over K VisualSync objects. The
        # compute pipeline already produced batched (K, n_wheels) angles;
        # we slice into (1, n_wheels) for each entity and call its
        # VisualSync (n_envs=1) so set_dofs_position sees the shape it
        # expects. Loop cost is ~K small set_dofs_position calls — small
        # compared to the batched compute savings.
        if self.visuals:
            for k_i, vis in enumerate(self.visuals):
                vis.step(
                    steer_per_wheel[k_i:k_i + 1],
                    distances[k_i:k_i + 1],
                    p.omega[k_i:k_i + 1],
                    DT,
                )


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
                 vehicles: Sequence[tuple]):    # list of (entity, sensor, cfg)
        if not vehicles:
            raise ValueError("MultiVehiclePhysics needs at least one vehicle.")
        # Group by cfg identity (same Python object → same kind). Callers
        # who want grouping by URDF *value* can pass the same cfg instance
        # for matching vehicles (the bundled presets return fresh cfgs per
        # call, so identity-by-call works — see road_loop.py).
        groups: dict[int, list[tuple]] = defaultdict(list)
        group_order = []
        for veh in vehicles:
            entity, sensor, cfg = veh
            key = id(cfg)
            if key not in groups:
                group_order.append(key)
            groups[key].append(veh)

        self.vehicles = list(vehicles)   # preserve caller's order
        self.kinds = []                  # list of MultiVehicleKindPhysics
        self.kind_slices: list[slice] = []  # per-kind slice into flat inputs
        # Build per-kind physics, tracking which flat-input indices belong to it.
        # We need to map: input[i] (caller's flat order) → which kind / which slot.
        # Simplest: build kinds in groupwise order, then record which flat
        # indices belong to each kind; dispatch in step() reorders inputs.
        self._flat_to_kind = []          # length len(vehicles), each = (kind_idx, slot_idx)
        for k_i, key in enumerate(group_order):
            kind_vehicles = groups[key]
            entities = [v[0] for v in kind_vehicles]
            sensors  = [v[1] for v in kind_vehicles]
            cfg      = kind_vehicles[0][2]
            self.kinds.append(
                MultiVehicleKindPhysics(scene, entities, sensors, cfg))
            # Find the flat positions of these vehicles in the caller's order.
            for slot_i, veh in enumerate(kind_vehicles):
                flat_i = next(j for j, v in enumerate(self.vehicles) if v is veh)
                self._flat_to_kind.append((flat_i, k_i, slot_i))
        # Sort by flat_i so we can map quickly in step().
        self._flat_to_kind.sort(key=lambda t: t[0])

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
        # Re-bucket inputs into per-kind slot order.
        per_kind: list[list] = [[None] * k.K for k in self.kinds]
        for flat_i, kind_idx, slot_idx in self._flat_to_kind:
            per_kind[kind_idx][slot_idx] = inputs_list[flat_i]
        # Dispatch.
        for kind, ins in zip(self.kinds, per_kind):
            kind.step(ins)
