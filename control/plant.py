"""Differentiable plant model — the sweep table's replacement (v1.3.0).

A :class:`PathFollower` needs one thing from the vehicle: the *inverse* of its
input→response map,

    (v, a_target, pitch, roll)      -> throttle
    (v, omega_target, pitch, roll)  -> steer

Until v1.3.0 the only answer was a :class:`~genesis_vehicle.control.SweepTable`
— a 5-D grid of that map, measured offline by driving tens of thousands of
open-loop rollouts (``sweep_measure``) and written to CSV. It works, but the
CSV is valid for exactly one (URDF, preset, override) triple at one dt, it
takes a long batched GPU run to produce, and every plant change (a mass edit,
a friction edit, the v1.2.8 ``i_wheel`` fix) silently invalidates it.

:class:`DifferentiablePlant` answers the same two questions with **no CSV and
no offline measurement**: it differentiates the vehicle's own force model.

How
---
The SDK's ray-wheel physics lives in :func:`genesis_vehicle._pipeline.compute_wheel_step`,
which is pure batched ``torch`` with no I/O — suspension load, slip, the
Pacejka tire, the wheel-omega update and the resulting chassis wrench. That
makes it differentiable as written. This module unrolls it ``horizon`` steps
forward against a planar (surge/sway/yaw) rigid-body integrator built from the
vehicle's real mass matrix, reads off

    a       = (v_long(H) - v_long(0)) / (H * dt)
    omega_z = mean yaw rate over the horizon

— the same two quantities the sweep table tabulates — and gets

    J = d(a, omega_z) / d(throttle, steer)

from ``torch.autograd``. One Newton/chord solve on ``J`` inverts the plant at
the current operating point, warm-started from last step's command.

Because the unroll starts from the LIVE state (per-wheel normal loads, wheel
speeds, body velocity, attitude), the inverse is local to what the vehicle is
actually doing, rather than an interpolation of a grid measured at a handful of
(v, throttle, steer, pitch, roll) nodes.

Fleet batching (v1.4.0)
-----------------------
One plant serves **M members** — (vehicle, env) pairs of a single kind, the
same granularity ``MultiVehicleKindPhysics`` groups by. Every member carries
its own state, target, warm start and Jacobian; they share the force model,
the wheel meta and the mass properties, which is what "one kind" means.

This matters because the unroll is dominated by torch's per-op dispatch at
these tensor sizes, not by arithmetic: a batch of 128 costs 1.25x a batch of
1, so the per-member cost falls by two orders of magnitude across a fleet.
``M == 1`` keeps the scalar-in / scalar-out surface unchanged.

Why not the Genesis differentiable solver
-----------------------------------------
Genesis does ship a differentiable rigid solver (``SimOptions(requires_grad=True)``),
but its taped input set is exactly ``set_pos`` / ``set_quat`` /
``set_dofs_velocity`` / ``control_dofs_force`` — see
``RigidEntity.process_input_grad``. **``apply_links_external_force`` and
``apply_links_external_torque`` are not on the tape**, and they are the only
way a ray-wheel vehicle touches the solver: this SDK has no wheel joints to
motor. Verified empirically on genesis-world 1.3.3 — a leaf tensor passed to
``apply_links_external_force`` comes back from ``scene.backward()`` with
``grad is None``, while the same test through ``control_dofs_force`` returns a
finite gradient. So autodiff through Genesis can only differentiate a
joint-motor vehicle, not this one; differentiating the SDK's own force model
gets the same Jacobian, needs no ``requires_grad`` scene (no hibernation ban,
no integrator restriction, no dense-Hessian cost) and runs on CPU.

Accuracy envelope
-----------------
The unroll is the real force model driven by a reduced chassis model, so it is
exact in the wheel forces and approximate in the body motion:

- surge/sway/yaw only. Heave, pitch and roll RATES are frozen, and the
  measured wheel-ground distances are held over the horizon, i.e. the normal
  loads are the ones measured this step (no in-horizon load transfer).
- gravity enters through the live attitude, so slopes are modelled (this is
  what the sweep table's pitch/roll axes bought).
- terrain is assumed locally flat over the horizon (0.1 s at the default
  ``horizon=4`` and 40 Hz).

Those are the same assumptions a bicycle-model controller makes, and they cost
nothing in the closed loop: the Jacobian is a local gain, refreshed every step.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Optional

import numpy as np
import torch

from .._pipeline import compute_wheel_step


__all__ = ["DifferentiablePlant", "PlantMassProperties", "plant_mass_properties",
           "is_state_source", "STATE_SOURCE_SURFACE"]


# --- Mass properties ----------------------------------------------------------

class PlantMassProperties:
    """Planar rigid-body properties of one vehicle, in the CHASSIS BODY frame.

    ``mass`` (kg), ``com`` = centre of mass offset from the base-link ORIGIN
    (m, body frame), ``izz`` = yaw inertia about the base-link ORIGIN (kg*m^2).

    The 3x3 ``m3`` is the planar mass matrix over the (surge, sway, yaw) dofs
    referenced at the base-link origin::

        [[ m,      0,     -m*cy ],
         [ 0,      m,      m*cx ],
         [ -m*cy,  m*cx,   izz  ]]

    referencing at the origin (not the CoM) is deliberate:
    ``compute_wheel_step`` returns its wrench about the base-link origin, and
    ``entity.get_vel()`` — the velocity :func:`extract_state` turns into
    ``v_long`` — is the origin's.
    """

    __slots__ = ("mass", "com", "izz", "m3", "m3_inv")

    def __init__(self, mass: float, com, izz: float):
        self.mass = float(mass)
        self.com = (float(com[0]), float(com[1]), float(com[2]))
        self.izz = float(izz)
        m, cx, cy = self.mass, self.com[0], self.com[1]
        self.m3 = np.array([[m, 0.0, -m * cy],
                            [0.0, m, m * cx],
                            [-m * cy, m * cx, self.izz]], dtype=np.float64)
        self.m3_inv = np.linalg.inv(self.m3)

    def __repr__(self) -> str:
        return (f"PlantMassProperties(mass={self.mass:.1f}, "
                f"com=({self.com[0]:+.3f}, {self.com[1]:+.3f}, {self.com[2]:+.3f}), "
                f"izz={self.izz:.1f})")


def _as_np(x, shape=None):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    a = np.asarray(x, dtype=np.float64)
    return a.reshape(shape) if shape is not None else a


def _quat_to_R(q) -> np.ndarray:
    """Scalar-first (w, x, y, z) quaternion -> rotation matrix.

    Batched: ``(..., 4)`` in, ``(..., 3, 3)`` out.
    """
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def plant_mass_properties(entity: Any) -> PlantMassProperties:
    """Composite planar mass properties of a built Genesis entity, body frame.

    Walks every link's ``inertial_mass`` / ``inertial_pos`` / ``inertial_quat``
    / ``inertial_i``, transforms each into the BASE-LINK frame and accumulates
    mass, first moment and the yaw inertia about the base-link origin.

    Reading the entity's 6x6 mass matrix would give the same numbers but in
    WORLD axes at the current attitude, so it would only be usable while the
    vehicle happens to sit level. The link walk is attitude-independent.
    """
    from ..kinematics import get_link_transforms

    lt = get_link_transforms(entity, "base")
    pos = _as_np(lt.pos)          # (n_envs, n_links, 3)
    quat = _as_np(lt.quat)        # (n_envs, n_links, 4)
    if pos.ndim == 3:
        pos, quat = pos[0], quat[0]

    m_tot = 0.0
    mc = np.zeros(3)
    i_o = np.zeros((3, 3))
    for i, link in enumerate(entity.links):
        m_i = getattr(link, "inertial_mass", None)
        if m_i is None:
            continue
        m_i = float(m_i)
        if m_i <= 0.0:
            continue
        R_l = _quat_to_R(quat[i])
        c_local = _as_np(getattr(link, "inertial_pos", None))
        c_local = np.zeros(3) if c_local is None else c_local.reshape(3)
        c_i = pos[i] + R_l @ c_local
        q_in = _as_np(getattr(link, "inertial_quat", None))
        R_in = R_l if q_in is None else R_l @ _quat_to_R(q_in.reshape(4))
        I_local = _as_np(getattr(link, "inertial_i", None))
        if I_local is None:
            I_c = np.zeros((3, 3))
        else:
            I_local = I_local.reshape(3, 3)
            I_c = R_in @ I_local @ R_in.T
        # Parallel-axis onto the base-link origin.
        i_o += I_c + m_i * (float(c_i @ c_i) * np.eye(3) - np.outer(c_i, c_i))
        m_tot += m_i
        mc += m_i * c_i

    if m_tot <= 0.0:
        raise ValueError(
            "could not read any link mass off the entity — pass mass=/izz=/com= "
            "to DifferentiablePlant explicitly")
    return PlantMassProperties(m_tot, mc / m_tot, float(i_o[2, 2]))


# --- Live-state source --------------------------------------------------------

#: What :class:`DifferentiablePlant` reads every step. Anything providing all
#: of it can be handed to the plant directly, in place of a Genesis vehicle —
#: which is how the Genesis-free unit tests drive the prediction, and the hook
#: for wiring the plant to a non-Genesis state feed. A source serving more than
#: one member also sets ``n_members``; without it the plant assumes one.
STATE_SOURCE_SURFACE = ("resolved", "wheel_meta", "dt", "dev", "fdt", "physics",
                        "chassis", "omega", "prev_compression", "distances")


def is_state_source(obj: Any) -> bool:
    """True if ``obj`` already provides the whole state-source surface."""
    return all(hasattr(obj, name) for name in STATE_SOURCE_SURFACE)


class _StateSource:
    """Everything the unroll needs, pulled from M live (vehicle, env) members.

    Resolves both driver layouts: a plain :class:`VehiclePhysics`
    (``VehicleScene(solver="per_vehicle")``) and vehicle slots of a batched
    ``MultiVehicleKindPhysics`` (the default ``solver="batched"``), where the
    per-wheel state lives in the kind's shared ``(n_envs*K, n_wheels)`` proto
    and a member is row ``env * K + slot``.

    All members must belong to ONE kind: they share ``resolved``,
    ``wheel_meta`` and the force model, which is what a kind is. Mixed kinds
    take one plant each.
    """

    def __init__(self, source: Any, env_idx=0):
        sources = list(source) if isinstance(source, (list, tuple)) else [source]
        if not sources:
            raise ValueError("DifferentiablePlant needs at least one vehicle")
        envs = ([int(env_idx)] if np.ndim(env_idx) == 0
                else [int(e) for e in env_idx])

        parts = [self._resolve(s) for s in sources]
        physics = parts[0][0]
        if any(p[0] is not physics for p in parts):
            raise ValueError(
                "all vehicles of one DifferentiablePlant must belong to the same "
                "kind (they share the force model and the wheel meta) — build "
                "one plant per kind")
        self.physics = physics
        self.entities = [p[1] for p in parts]
        self.kind = parts[0][3]
        K = parts[0][4]
        slots = [p[2] for p in parts]

        # Members are the cartesian product, env-major so the rows they map to
        # follow the proto's own (env * K + slot) ordering.
        self.envs = np.array([e for e in envs for _ in slots], dtype=np.int64)
        self.slots = np.array([s for _ in envs for s in slots], dtype=np.int64)
        self.n_members = int(self.envs.size)
        self.K = K
        self.n_envs = int(getattr(physics, "n_envs", 1))
        rows = self.envs * K + self.slots
        self.rows = torch.as_tensor(rows, dtype=torch.long)
        self.n_rows_total = self.n_envs * K

        self.resolved = physics.resolved
        self.wheel_meta = physics.wheel_meta
        self.dt = float(physics.dt)
        self.dev = physics.dev
        self.fdt = physics.fdt
        # Single-member back-compat surface.
        self.env_idx = int(self.envs[0])
        self.row = int(rows[0])
        self.entity = self.entities[0]

    @staticmethod
    def _resolve(source):
        """-> (physics, entity, slot, kind_or_None, K)"""
        scene = getattr(source, "_scene", None)
        if scene is not None and hasattr(source, "_slot"):
            entity = source.entity_main
            if getattr(source, "physics", None) is not None:      # per_vehicle
                return source.physics, entity, 0, None, 1
            mvp = getattr(scene, "_mvp", None)
            if mvp is None:
                raise ValueError(
                    "this Vehicle has no physics yet — call VehicleScene.build() "
                    "before building a DifferentiablePlant")
            for flat_i, kind_idx, slot_idx in mvp._flat_to_kind:
                if flat_i == source._slot:
                    kind = mvp.kinds[kind_idx]
                    return kind._proto, entity, slot_idx, kind, kind.K
            raise ValueError(f"vehicle slot {source._slot} not found in the scene's kinds")

        if hasattr(source, "resolved") and hasattr(source, "wheel_meta"):
            entity = getattr(source, "entity", None)
            if entity is None:
                raise ValueError("state source has no .entity to read chassis state from")
            return source, entity, 0, None, 1

        raise TypeError(
            "DifferentiablePlant needs a VehicleScene Vehicle handle or a "
            f"VehiclePhysics; got {type(source).__name__}")

    # -- reads -----------------------------------------------------------------

    def _rows(self, t):
        """Gather this source's member rows out of a proto-batched tensor."""
        if t is None:
            return None
        if t.dim() < 2:
            return t.unsqueeze(0).expand(self.n_members, -1)
        return t[self.rows.to(t.device)]

    def chassis(self):
        """``(pos, quat, vel, ang)`` as numpy ``(M, 3)`` / ``(M, 4)``.

        A kind reads all its base links in ONE batched solver call however many
        members this plant serves; only the per-vehicle driver (which has a
        single entity anyway) falls back to entity reads.
        """
        if self.kind is not None:
            idx = self.kind.base_idx_tensor
            solver = self.kind.solver

            def _pick(t):
                a = _as_np(t)
                if a.ndim == 3:                       # (n_envs, K, d)
                    return a[self.envs, self.slots]
                return a[self.slots]                  # (K, d) when n_envs == 0
            return (_pick(solver.get_links_pos(idx)),
                    _pick(solver.get_links_quat(idx)),
                    _pick(solver.get_links_vel(idx)),
                    _pick(solver.get_links_ang(idx)))

        ent = self.entities[0]

        def _pick1(t):
            a = _as_np(t)
            if a.ndim == 2:                           # (n_envs, d)
                return a[self.envs]
            return np.repeat(a[None, :], self.n_members, axis=0)
        return (_pick1(ent.get_pos()), _pick1(ent.get_quat()),
                _pick1(ent.get_vel()), _pick1(ent.get_ang()))

    def omega(self) -> torch.Tensor:
        return self._rows(self.physics.omega).detach().clone()

    def prev_compression(self) -> torch.Tensor:
        return self._rows(self.physics.prev_compression).detach().clone()

    def distances(self) -> Optional[torch.Tensor]:
        d = getattr(self.physics, "last_distances", None)
        if d is None:
            return None
        return self._rows(d).detach().clone()


# --- The plant ----------------------------------------------------------------

class DifferentiablePlant:
    """Sweep-table-free inverse plant: autodiff Jacobian + Newton solve.

    Drop-in for :class:`~genesis_vehicle.control.SweepTable` in
    :class:`~genesis_vehicle.control.PathFollower` — it answers the same
    ``throttle_for`` / ``steer_for`` questions, and additionally offers the
    coupled :meth:`solve` (both channels from one 2x2 inversion), which the
    follower prefers when present.

    Serves **M members** — (vehicle, env) pairs of one kind. With ``M == 1``
    (the default) every entry point takes and returns scalars, exactly as
    before fleet batching existed; with ``M > 1`` they take and return arrays
    of length M, and one call covers the whole fleet in a single unroll.

    Args:
        source: the live vehicle — a ``VehicleScene`` ``Vehicle`` handle
            (either solver mode), a ``VehiclePhysics``, or a SEQUENCE of
            vehicles of the same kind.
        env_idx: which Genesis env to read; a sequence selects several. The
            members are the cartesian product of ``source`` x ``env_idx``,
            env-major.
        horizon: unroll length in SIM STEPS, and the plant's cost knob (the
            solve is linear in it). Throttle reaches the ground only after the
            wheel-omega update — omega first, slip second, force third — so a
            1-step unroll sees the response only through the overshoot clamp
            and is rejected. The default 4 (0.1 s at 40 Hz) measured both
            faster AND tighter than 8 on the reference tank course: a
            controller running every step wants the near-term response, and a
            longer horizon also drifts further from the frozen-normal-load
            assumption. Raise it if the vehicle's tyre transients are slower
            than 0.1 s.
        newton_iters: chord iterations after the Jacobian is taken. The
            Jacobian is computed once per :meth:`solve`; each extra iteration
            is one forward unroll (no backward), which buys most of the
            benefit of a full Newton at a fraction of the cost.
        coupled: solve the full 2x2 system (a damped least squares over both
            commands at once). Default ``False`` — channel-diagonal, throttle
            for ``a`` and steer for ``omega_z``, which is the priority path
            following wants; the chord iterations still see the cross terms.
            See :meth:`_step_command` for why the coupled solve is not the
            default despite being the more "complete" inverse.
        brake: brake command assumed during the prediction. 0.0 matches how
            the sweep tables were measured.
        damping: Levenberg damping of the least-squares inversion, on the
            row-normalised Jacobian. Larger = more conservative near a
            singular operating point (standstill, a skid-steer at full
            differential).
        min_gain_a / min_gain_w: a Jacobian row whose norm falls below this
            has no authority in that channel; its residual is dropped rather
            than divided by an almost-zero gain.
        max_delta: per-call cap on the command change, in input units. Keeps
            one bad Jacobian from throwing a full-scale command.
        scale_a / scale_w: the residual units that count as "one unit of
            error" when the line search compares candidate commands — an
            ``a`` error of ``scale_a`` m/s^2 weighs the same as an
            ``omega_z`` error of ``scale_w`` rad/s.
        line_search_steps: how many times a rejected step may be halved
            before the iteration gives up and keeps the current command.
        tol: weighted residual below which the solve stops early.
        probe_delta: command step used to rescue a channel whose autograd
            derivative is identically zero (see :meth:`jacobian`). 0 disables
            the rescue and lets such a channel report no authority.
        mass / izz / com: override the planar mass properties, which are
            otherwise composed from the entity's links. Giving both
            `mass` and `izz` skips that read entirely, and `com` then
            defaults to the base-link origin. Shared by every member, which is
            what "one kind" means.

    Example::

        plant = DifferentiablePlant(veh)          # no CSV anywhere
        follower = PathFollower(path, plant)

        fleet = DifferentiablePlant(all_tanks)    # one unroll for the lot
        thr, steer = fleet.solve(v_long, a_target, omega_target)   # arrays
    """

    #: Marks the coupled two-channel solve, which PathFollower prefers.
    has_coupled_solve = True

    def __init__(self, source: Any, *,
                 env_idx=0,
                 horizon: int = 4,
                 newton_iters: int = 2,
                 coupled: bool = False,
                 brake: float = 0.0,
                 mass: Optional[float] = None,
                 izz: Optional[float] = None,
                 com=None,
                 throttle_range: tuple = (-1.0, 1.0),
                 steer_range: tuple = (-1.0, 1.0),
                 min_gain_a: float = 1e-3,
                 min_gain_w: float = 1e-3,
                 damping: float = 1e-2,
                 max_delta: float = 1.0,
                 scale_a: float = 1.0,
                 scale_w: float = 0.2,
                 line_search_steps: int = 3,
                 tol: float = 1e-3,
                 probe_delta: float = 0.2,
                 probe_threshold: float = 1e-9):
        if horizon < 2:
            raise ValueError(
                "horizon must be >= 2: within a single step the tyre force is "
                "computed from the wheel omega the vehicle already has, so a "
                "1-step unroll reports throttle authority only through the "
                "F_long overshoot clamp — a numerical guard, not the response "
                "the controller is trying to invert")
        self.src = source if is_state_source(source) else _StateSource(source, env_idx)
        self.n_members = int(getattr(self.src, "n_members", 1))
        self.horizon = int(horizon)
        self.newton_iters = int(newton_iters)
        self.coupled = bool(coupled)
        self.brake = float(brake)
        self.throttle_range = (float(throttle_range[0]), float(throttle_range[1]))
        self.steer_range = (float(steer_range[0]), float(steer_range[1]))
        self.min_gain_a = float(min_gain_a)
        self.min_gain_w = float(min_gain_w)
        self.damping = float(damping)
        self.max_delta = float(max_delta)
        self.scale_a = float(scale_a)
        self.scale_w = float(scale_w)
        self.line_search_steps = int(line_search_steps)
        self.tol = float(tol)
        self.probe_delta = float(probe_delta)
        self.probe_threshold = float(probe_threshold)

        if mass is None or izz is None:
            mp = plant_mass_properties(self.src.entity)
            mass = mp.mass if mass is None else mass
            izz = mp.izz if izz is None else izz
            com = mp.com if com is None else com
        elif com is None:
            # Both inertial terms were given, so there is nothing to go and
            # read the entity for. The CoM offset only adds the surge/sway <->
            # yaw coupling terms of the planar mass matrix, and assuming the
            # base-link origin is a small, well-defined approximation — not a
            # reason to require a live entity from a caller who already knows
            # the vehicle's mass and yaw inertia.
            com = (0.0, 0.0, 0.0)
        self.mass_props = PlantMassProperties(mass, com, izz)

        # Prediction-only hook copies: the live hooks own cross-step state
        # (StaticFrictionLock anchors its stick-slip springs), which an
        # unrolled what-if must not advance. Re-synced from the live ones at
        # the top of every unroll so the prediction still starts where the
        # vehicles actually are.
        p = self.src.physics
        self._pred_pre_hooks = copy.deepcopy(list(p.pre_loop_hooks))
        self._pred_post_hooks = copy.deepcopy(list(p.post_tire_hooks))

        # Warm start + diagnostics, per member. Everything a caller needs to
        # see why a command came out the way it did: `last_jacobian` is the
        # 2x2 the solve linearised on, `last_response` the (a, omega_z) it
        # converged to, `last_target` what was asked for, `last_singular`
        # whether a channel had no authority at this operating point.
        M = self.n_members
        self._u = np.zeros((M, 2))
        self._jacobian = np.zeros((M, 2, 2))
        self._response = np.zeros((M, 2))
        self._target = np.zeros((M, 2))
        self._singular = np.zeros(M, dtype=bool)
        self._cost = np.zeros(M)
        self.last_iters = 0
        self._pending_a: Optional[float] = None
        self._has_solved = False

    # -- introspection ---------------------------------------------------------

    @property
    def dt(self) -> float:
        return self.src.dt

    @property
    def last_u(self):
        """Warm start: ``(throttle, steer)``, or an ``(M, 2)`` array."""
        return tuple(self._u[0]) if self.n_members == 1 else self._u.copy()

    @property
    def last_jacobian(self):
        if not self._has_solved:
            return None
        return self._jacobian[0] if self.n_members == 1 else self._jacobian.copy()

    @property
    def last_response(self):
        if not self._has_solved:
            return None
        return tuple(self._response[0]) if self.n_members == 1 else self._response.copy()

    @property
    def last_target(self):
        if not self._has_solved:
            return None
        return tuple(self._target[0]) if self.n_members == 1 else self._target.copy()

    @property
    def last_singular(self):
        return bool(self._singular[0]) if self.n_members == 1 else self._singular.copy()

    @property
    def last_cost(self):
        return float(self._cost[0]) if self.n_members == 1 else self._cost.copy()

    def __repr__(self) -> str:
        members = "" if self.n_members == 1 else f"members={self.n_members}, "
        return (f"DifferentiablePlant({members}"
                f"n_wheels={self.src.wheel_meta.n_wheels}, "
                f"horizon={self.horizon}, dt={self.dt:.4f}, "
                f"{self.mass_props!r})")

    # -- prediction ------------------------------------------------------------

    def _sync_hooks(self, cands: int = 1) -> None:
        """Reset the prediction hooks to the live per-member state.

        Two jobs at once. It re-anchors the copies on what the vehicles are
        actually doing, and it detaches — a hook carrying a graph tensor from
        the previous unroll would chain every unroll into one ever-growing
        graph, which turns repeated calls quadratic.

        Tensors indexed by the proto's flat batch are gathered down to this
        plant's members and then widened to the ``M * cands`` rows the unroll
        will run at, because a hook that re-allocates on a shape mismatch
        (StaticFrictionLock does) would silently zero its anchors instead.
        """
        rows = self.src.rows if hasattr(self.src, "rows") else None
        n_total = getattr(self.src, "n_rows_total", None)
        for live, pred in ((self.src.physics.pre_loop_hooks, self._pred_pre_hooks),
                           (self.src.physics.post_tire_hooks, self._pred_post_hooks)):
            for lh, ph in zip(live, pred):
                for k, v in lh.__dict__.items():
                    if not torch.is_tensor(v):
                        continue
                    t = v.detach()
                    if (rows is not None and n_total is not None
                            and t.dim() >= 1 and t.shape[0] == n_total):
                        t = t[rows.to(t.device)]
                    if cands > 1:
                        t = t.repeat_interleave(cands, dim=0)
                    setattr(ph, k, t.clone())

    def _snapshot(self) -> dict:
        """Freeze the live per-member state the next unroll starts from."""
        src = self.src
        M = self.n_members
        pos, quat, vel, ang = src.chassis()
        quat = np.atleast_2d(np.asarray(quat, dtype=np.float64))
        vel = np.atleast_2d(np.asarray(vel, dtype=np.float64))
        ang = np.atleast_2d(np.asarray(ang, dtype=np.float64))
        R = _quat_to_R(quat)                                    # (M, 3, 3)
        dev, fdt = src.dev, src.fdt
        dist = src.distances()
        if dist is None:
            raise RuntimeError(
                "the vehicle has no wheel-ground distances yet — step the scene "
                "at least once before asking the plant for a command")
        n = src.wheel_meta.n_wheels
        g_world = np.array([0.0, 0.0, -9.81])
        return {
            "distances": dist.reshape(M, n).to(device=dev, dtype=fdt),
            "omega": src.omega().reshape(M, n).to(device=dev, dtype=fdt),
            "prev_compression": src.prev_compression().reshape(M, n).to(
                device=dev, dtype=fdt),
            "v_b": np.einsum("mji,mj->mi", R, vel),             # R^T v, (M, 3)
            "w_z": np.einsum("mji,mj->mi", R, ang)[:, 2],       # (M,)
            "g_b": np.einsum("mji,j->mi", R, g_world),          # (M, 3)
        }

    def _unroll(self, throttle: torch.Tensor, steer: torch.Tensor,
                snap: dict, cands: int = 1) -> tuple:
        """H steps of the real wheel pipeline + a planar chassis integrator.

        ``throttle`` / ``steer`` are ``(M * cands,)``, member-major: member 0's
        candidates first, then member 1's. The batch axis therefore carries
        BOTH the fleet and the candidate commands of a line search, and at
        these tensor sizes the unroll is dominated by per-op dispatch, so
        widening it is close to free.

        Returns ``(a, omega_z)``, both ``(M * cands,)`` and graph-carrying.
        """
        src = self.src
        dev, fdt = src.dev, src.fdt
        wm = src.wheel_meta
        resolved = src.resolved
        DT = src.dt
        n = wm.n_wheels
        M = self.n_members
        B = M * cands

        from ..core import PipelineContext
        from ..inputs import VehicleInputs

        throttle = throttle.reshape(B)
        steer = steer.reshape(B)
        zeros3 = torch.zeros(B, 3, device=dev, dtype=fdt)
        zeroB = torch.zeros(B, device=dev, dtype=fdt)
        quat_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev,
                               dtype=fdt).expand(B, 4)
        up = torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=fdt).expand(B, 3)
        wheel_body_b = wm.positions.to(device=dev, dtype=fdt).unsqueeze(0).expand(B, n, 3)

        def _wide(t):
            return t if cands == 1 else t.repeat_interleave(cands, dim=0)

        def _np_wide(a):
            a = np.asarray(a, dtype=np.float64)
            return np.repeat(a, cands, axis=0) if cands > 1 else a

        brake_t = torch.full((B,), self.brake, device=dev, dtype=fdt)
        omega = _wide(snap["omega"])
        prev_comp = _wide(snap["prev_compression"])
        distances = _wide(snap["distances"])

        # Planar state, body frame, referenced at the base-link origin.
        v_b = _np_wide(snap["v_b"])
        vx = torch.as_tensor(v_b[:, 0], device=dev, dtype=fdt)
        vy = torch.as_tensor(v_b[:, 1], device=dev, dtype=fdt)
        wz = torch.as_tensor(_np_wide(snap["w_z"]), device=dev, dtype=fdt)
        vx0 = vx
        g_b = _np_wide(snap["g_b"])
        gx = torch.as_tensor(g_b[:, 0], device=dev, dtype=fdt)
        gy = torch.as_tensor(g_b[:, 1], device=dev, dtype=fdt)

        m = self.mass_props.mass
        cx, cy = self.mass_props.com[0], self.mass_props.com[1]
        m3_inv = torch.as_tensor(self.mass_props.m3_inv, device=dev, dtype=fdt)

        steering = resolved.steering
        drivetrain = resolved.drivetrain
        wz_sum = zeroB

        for _ in range(self.horizon):
            vel_b = torch.stack([vx, vy, zeroB], dim=-1)
            ang_b = torch.stack([zeroB, zeroB, wz], dim=-1)

            unified = VehicleInputs(throttle=throttle, brake=brake_t, steer=steer)
            ins = steering.InputType.from_unified(unified)
            steer_pw = steering.per_wheel_steer(ins, B, wm, dev, fdt)
            T_drive, T_brake = drivetrain.distribute_torque(ins, omega, wm, dev, fdt)

            ctx = PipelineContext(throttle=throttle, brake=brake_t, wheel_meta=wm)
            ctx.vel = vel_b
            ctx.ang = ang_b
            for hook in self._pred_pre_hooks:
                hook.apply_pre_loop(ctx)

            res = compute_wheel_step(
                pos=zeros3, quat=quat_id, vel=vel_b, ang=ang_b,
                distances=distances, steer_per_wheel=steer_pw,
                T_drive_pw=T_drive, T_brake_pw=T_brake, omega=omega,
                prev_compression=prev_comp, prev_init=True,
                wheel_meta=wm, resolved=resolved,
                wheel_body_b=wheel_body_b, up_world=up,
                post_tire_hooks=self._pred_post_hooks, ctx=ctx,
                dt=DT, B=B, n=n,
            )

            # Planar Newton-Euler about the base-link origin. The wrench comes
            # out of the pipeline in the same frame (chassis origin, body axes
            # because the unroll runs with quat == identity).
            Fx = res.total_F[:, 0] + m * gx
            Fy = res.total_F[:, 1] + m * gy
            Tz = res.total_T[:, 2] + m * (cx * gy - cy * gx)
            # Centrifugal / Coriolis terms of an origin-referenced planar body.
            rhs = torch.stack([
                Fx + m * (wz * vy + wz * wz * cx),
                Fy - m * (wz * vx - wz * wz * cy),
                Tz - m * (cx * wz * vx + cy * wz * vy),
            ])                                              # (3, B)
            acc = m3_inv @ rhs                              # (3, B)

            vx = vx + acc[0] * DT
            vy = vy + acc[1] * DT
            wz = wz + acc[2] * DT
            wz_sum = wz_sum + wz
            omega = res.new_omega
            prev_comp = res.compression

        a = (vx - vx0) / (self.horizon * DT)
        omega_z = wz_sum / self.horizon
        return a, omega_z

    def predict(self, throttle, steer, snap: Optional[dict] = None,
                cands: Optional[int] = None):
        """Forward-only prediction of ``(a, omega_z)`` for held commands.

        With one member and scalar inputs, returns scalars. Otherwise returns
        two arrays of length ``M * cands``, member-major — several candidate
        commands per member, all in ONE unroll, which is what the line search
        does.

        ``cands`` is inferred from the input length when omitted, so a
        single-member plant can be swept with a plain list of commands.
        """
        if snap is None:
            snap = self._snapshot()
        M = self.n_members
        dev, fdt = self.src.dev, self.src.fdt
        scalar_in = np.ndim(throttle) == 0 and np.ndim(steer) == 0
        t = torch.as_tensor(np.atleast_1d(np.asarray(throttle, dtype=np.float64)),
                            device=dev, dtype=fdt)
        s = torch.as_tensor(np.atleast_1d(np.asarray(steer, dtype=np.float64)),
                            device=dev, dtype=fdt)
        if t.shape != s.shape:
            t, s = torch.broadcast_tensors(t, s)
        n_in = int(t.numel())
        if cands is None:
            if n_in % M:
                raise ValueError(
                    f"predict got {n_in} commands, which is not a whole number "
                    f"of candidates for this plant's {M} members")
            cands = n_in // M
        elif n_in != M * cands:
            raise ValueError(
                f"predict got {n_in} commands, expected M*cands = {M}*{cands}")
        if scalar_in and M > 1:
            # One command for the whole fleet: hold it on every member.
            t = t.expand(M).contiguous()
            s = s.expand(M).contiguous()
            cands = 1
        self._sync_hooks(cands)
        with torch.no_grad():
            a, w = self._unroll(t, s, snap, cands)
        a = a.detach().cpu().numpy()
        w = w.detach().cpu().numpy()
        if scalar_in and M == 1:
            return float(a[0]), float(w[0])
        return a, w

    def jacobian(self, throttle=None, steer=None, snap: Optional[dict] = None):
        """``(J, y)`` at the given command.

        ``J`` is ``(2, 2)`` and ``y`` is ``(a, omega_z)`` for a single member;
        ``(M, 2, 2)`` and ``(M, 2)`` for a fleet. Rows are ``(a, omega_z)``,
        columns ``(throttle, steer)``.

        Both rows come from one backward pass each over the WHOLE fleet: batch
        members are independent in the unroll, so ``d(sum_i a_i)/d(thr_j)`` is
        exactly member j's own gradient, and M members cost what one does.
        """
        if snap is None:
            snap = self._snapshot()
        M = self.n_members
        u0 = self._u
        thr = u0[:, 0] if throttle is None else np.broadcast_to(
            np.asarray(throttle, dtype=np.float64), (M,))
        st = u0[:, 1] if steer is None else np.broadcast_to(
            np.asarray(steer, dtype=np.float64), (M,))
        dev, fdt = self.src.dev, self.src.fdt
        self._sync_hooks(1)
        t = torch.tensor(np.ascontiguousarray(thr), device=dev, dtype=fdt,
                         requires_grad=True)
        s = torch.tensor(np.ascontiguousarray(st), device=dev, dtype=fdt,
                         requires_grad=True)
        a, w = self._unroll(t, s, snap, cands=1)
        y = np.stack([a.detach().cpu().numpy(), w.detach().cpu().numpy()], axis=1)
        J = np.zeros((M, 2, 2))
        for i, out in enumerate((a, w)):
            g = torch.autograd.grad(out.sum(), (t, s), retain_graph=(i == 0),
                                    allow_unused=True)
            for c in (0, 1):
                if g[c] is not None:
                    J[:, i, c] = g[c].detach().cpu().numpy()

        # Secant rescue for a channel whose derivative is degenerate.
        #
        # The plant has genuine dead zones: the v0.6.0 F_long overshoot clamp
        # pins the tyre force to exactly zero whenever friction could not carry
        # the wheel past rolling within one step, and `torch.where` hands back
        # a zero gradient there. The DERIVATIVE is then honestly zero while the
        # SECANT over a control-sized command change is not — a tank cruising
        # in reverse sat in exactly that state and froze the controller, since
        # a zero Jacobian means "no authority, do not move".
        #
        # So: where a column of a member's Jacobian vanishes, replace it with a
        # one-sided difference over `probe_delta`, large enough to step out of
        # the dead zone. Both columns of every member ride one batched unroll.
        if self.probe_delta > 0.0:
            u = np.stack([thr, st], axis=1)                       # (M, 2)
            dead = np.abs(J).max(axis=1) <= self.probe_threshold  # (M, 2)
            if dead.any():
                lo = np.array([self.throttle_range[0], self.steer_range[0]])
                hi = np.array([self.throttle_range[1], self.steer_range[1]])
                # Probe towards whichever side has room (a saturated command
                # has room only inwards). Column c of every member at once.
                trials = np.repeat(u[:, None, :], 2, axis=1)      # (M, 2, 2)
                steps = np.zeros((M, 2))
                for c in (0, 1):
                    d = np.where(u[:, c] + self.probe_delta <= hi[c],
                                 self.probe_delta, -self.probe_delta)
                    d = np.clip(d, lo[c] - u[:, c], hi[c] - u[:, c])
                    trials[:, c, c] = u[:, c] + d
                    steps[:, c] = d
                a_t, w_t = self.predict(trials[:, :, 0].reshape(-1),
                                        trials[:, :, 1].reshape(-1),
                                        snap=snap, cands=2)
                a_t = np.asarray(a_t).reshape(M, 2)
                w_t = np.asarray(w_t).reshape(M, 2)
                ok = dead & (np.abs(steps) > 1e-9)
                for c in (0, 1):
                    sel = ok[:, c]
                    if not sel.any():
                        continue
                    J[sel, 0, c] = (a_t[sel, c] - y[sel, 0]) / steps[sel, c]
                    J[sel, 1, c] = (w_t[sel, c] - y[sel, 1]) / steps[sel, c]

        if M == 1:
            return J[0], (float(y[0, 0]), float(y[0, 1]))
        return J, y

    # -- inversion -------------------------------------------------------------

    def _step_command(self, J, err, u):
        """One command update per member, from Jacobians and response errors.

        ``J`` is ``(M, 2, 2)``; ``err`` and ``u`` are ``(M, 2)``.

        Damped least squares rather than a straight ``J^-1 e``. The plant has
        genuine local singularities — a skid-steer at full steer differential
        has no throttle authority over ``a`` at all, and every vehicle loses
        yaw authority at a standstill — where the exact inverse asks for an
        unbounded command and the loop turns bang-bang. DLS degrades to "this
        channel cannot buy anything here, so leave it alone", which is the
        correct action; :class:`PathFollower`'s low-speed KICK covers pulling
        away from rest, where no plant inverse can help.
        """
        J = np.nan_to_num(np.asarray(J, dtype=np.float64), nan=0.0,
                          posinf=0.0, neginf=0.0)
        e = np.asarray(err, dtype=np.float64)
        floors = np.array([self.min_gain_a, self.min_gain_w])

        if self.coupled:
            # Rows are normalised first so the m/s^2 and rad/s residuals are
            # weighted comparably and one damping constant serves both.
            rows = np.linalg.norm(J, axis=2)                       # (M, 2)
            weak = rows < floors
            scale = np.where(rows > 1e-12, rows, 1.0)
            Jn = J / scale[:, :, None]
            en = np.where(weak, 0.0, e / scale)
            A = np.matmul(np.swapaxes(Jn, 1, 2), Jn) + self.damping * np.eye(2)
            rhs = np.matmul(np.swapaxes(Jn, 1, 2), en[:, :, None])
            du = np.linalg.solve(A, rhs)[:, :, 0]
        else:
            # Channel-diagonal inversion: throttle answers for `a`, steer for
            # `omega_z`. The cross terms are NOT ignored, they are handled by
            # the chord iterations, which re-evaluate the full coupled response
            # at the updated command — the safe half of the coupling without
            # its pathology. A coupled least-squares solve is free to spend the
            # steer channel on a speed error whenever `|dA/dsteer|` exceeds
            # `|dA/dthrottle|`, which for a skid-steer near rest it does: the
            # controller then throws the vehicle into a spin to make it go
            # faster. Path following wants the opposite priority.
            g = np.stack([J[:, 0, 0], J[:, 1, 1]], axis=1)         # (M, 2)
            weak = np.abs(g) < floors
            du = np.where(weak, 0.0, e / np.where(weak, 1.0, g))

        du = np.clip(du, -self.max_delta, self.max_delta)
        lo = np.array([self.throttle_range[0], self.steer_range[0]])
        hi = np.array([self.throttle_range[1], self.steer_range[1]])
        return np.clip(u + du, lo, hi), weak.any(axis=1)

    def solve(self, v_long, a_target, omega_target,
              pitch_deg=0.0, roll_deg=0.0):
        """Inverse: the ``(throttle, steer)`` hitting ``(a_target, omega_target)``.

        Scalars in, ``(throttle, steer)`` out for a single member; arrays of
        length M in, two arrays out for a fleet — one unroll covers the lot.

        ``v_long`` / ``pitch_deg`` / ``roll_deg`` are accepted for signature
        symmetry with :class:`SweepTable` but are NOT used — the plant reads
        the live speed and attitude off the vehicles themselves, which is
        strictly more information than those three scalars carry.
        """
        M = self.n_members
        snap = self._snapshot()
        target = np.stack([
            np.broadcast_to(np.asarray(a_target, dtype=np.float64), (M,)),
            np.broadcast_to(np.asarray(omega_target, dtype=np.float64), (M,)),
        ], axis=1)                                                  # (M, 2)
        self._target = target.copy()
        w = np.array([1.0 / self.scale_a, 1.0 / self.scale_w])

        u = self._u.copy()
        J, y = self.jacobian(u[:, 0], u[:, 1], snap=snap)
        J = np.asarray(J, dtype=np.float64).reshape(M, 2, 2)
        y = np.asarray(y, dtype=np.float64).reshape(M, 2)
        best_cost = np.linalg.norm((target - y) * w, axis=1)        # (M,)
        singular = np.zeros(M, dtype=bool)
        iters = 0

        # Chord iterations with a backtracking line search. Reusing the same J
        # keeps each iteration to one FORWARD unroll (no backward), and the
        # line search is what makes the loop safe where the plant is kinked:
        # the rev limiter and the friction-circle clamp both put corners in the
        # response, and a Jacobian read on the wrong side of one can point
        # uphill. Rather than guess which derivatives to trust, take the step
        # only if the predicted residual actually shrinks.
        fracs = 0.5 ** np.arange(self.line_search_steps)
        C = self.line_search_steps
        rows = np.arange(M)
        for _ in range(max(1, self.newton_iters)):
            live = best_cost >= self.tol
            if not live.any():
                break
            # A converged member contributes no residual, so its step is zero
            # and the batched unroll simply re-evaluates it at no extra cost.
            err = np.where(live[:, None], target - y, 0.0)
            cand, sing = self._step_command(J, err, u)
            singular |= sing
            du = cand - u
            if not np.any(np.abs(du) > 1e-12):
                break

            # All backtracking candidates of all members in ONE batched
            # unroll: the halvings are independent, and a wider batch costs
            # almost the same at these sizes.
            trials = u[:, None, :] + du[:, None, :] * fracs[None, :, None]
            a_t, w_t = self.predict(trials[:, :, 0].reshape(-1),
                                    trials[:, :, 1].reshape(-1),
                                    snap=snap, cands=C)
            y_t = np.stack([np.asarray(a_t), np.asarray(w_t)], axis=1).reshape(M, C, 2)
            costs = np.linalg.norm((target[:, None, :] - y_t) * w, axis=2)  # (M, C)
            k = np.argmin(costs, axis=1)
            best_k = costs[rows, k]
            take = live & (best_k < best_cost)
            iters += 1
            if not take.any():
                break
            u = np.where(take[:, None], trials[rows, k], u)
            y = np.where(take[:, None], y_t[rows, k], y)
            best_cost = np.where(take, best_k, best_cost)

        self._u = u
        self._jacobian = J
        self._response = y
        self._cost = best_cost
        self._singular = singular
        self.last_iters = iters
        self._has_solved = True
        if M == 1:
            return float(u[0, 0]), float(u[0, 1])
        return u[:, 0].copy(), u[:, 1].copy()

    # -- SweepTable-compatible surface ----------------------------------------
    #
    # These exist so the plant can stand in for a SweepTable anywhere, but they
    # cost one solve each. PathFollower checks `has_coupled_solve` and calls
    # solve() once per step instead.

    def _require_single(self, what: str) -> None:
        if self.n_members != 1:
            raise ValueError(
                f"{what} is the scalar SweepTable protocol and only applies to a "
                f"single-member plant; this one serves {self.n_members}. Use "
                "solve() with arrays, or one plant per vehicle.")

    def throttle_for(self, v: float, a_target: float,
                     pitch_deg: float = 0.0, roll_deg: float = 0.0) -> float:
        """Throttle for ``a_target``, yaw rate left where it is.

        Remembers ``a_target`` so a following :meth:`steer_for` — the order
        :class:`PathFollower` calls them in — can solve both channels together
        instead of fighting this one.
        """
        self._require_single("throttle_for")
        self._pending_a = float(a_target)
        return self.solve(v, a_target, self._held_omega(), pitch_deg, roll_deg)[0]

    def steer_for(self, v: float, omega_target: float,
                  pitch_deg: float = 0.0, roll_deg: float = 0.0) -> float:
        """Steer for ``omega_target``, honouring a pending :meth:`throttle_for`."""
        self._require_single("steer_for")
        a_t = self._pending_a if self._pending_a is not None else self._held_a()
        self._pending_a = None
        return self.solve(v, a_t, omega_target, pitch_deg, roll_deg)[1]

    def _held_omega(self) -> float:
        return float(self._response[0, 1]) if self._has_solved else 0.0

    def _held_a(self) -> float:
        return float(self._response[0, 0]) if self._has_solved else 0.0

    def set_applied(self, throttle, steer) -> None:
        """Tell the plant what the vehicles were ACTUALLY commanded this step.

        The Jacobian is a linearisation about the warm-start command, so the
        warm start has to be the command the vehicle really got. A caller that
        post-processes the solve — :class:`PathFollower` clamps steer to its
        own ``steer_cap`` and overrides throttle outright with its low-speed
        KICK — must report the result back, or every following step
        differentiates the plant at an operating point the vehicle was never
        in. That mismatch is invisible while the two agree and divergent the
        moment they do not.

        Takes scalars for a single member, arrays of length M for a fleet.
        """
        M = self.n_members
        self._u = np.stack([
            np.broadcast_to(np.asarray(throttle, dtype=np.float64), (M,)),
            np.broadcast_to(np.asarray(steer, dtype=np.float64), (M,)),
        ], axis=1).copy()

    def set_applied_member(self, i: int, throttle: float, steer: float) -> None:
        """:meth:`set_applied` for ONE member of a fleet plant.

        The per-vehicle followers driving a fleet each report their own
        command, so they need to write a single row without disturbing the
        others.
        """
        self._u[int(i), 0] = float(throttle)
        self._u[int(i), 1] = float(steer)

    def reset(self) -> None:
        """Drop the warm start (after a teleport / scenario reset)."""
        M = self.n_members
        self._u = np.zeros((M, 2))
        self._jacobian = np.zeros((M, 2, 2))
        self._response = np.zeros((M, 2))
        self._target = np.zeros((M, 2))
        self._singular = np.zeros(M, dtype=bool)
        self._cost = np.zeros(M)
        self._has_solved = False
        self._pending_a = None
