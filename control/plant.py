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
    """Scalar-first (w, x, y, z) quaternion -> 3x3 rotation matrix."""
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


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
#: for wiring the plant to a non-Genesis state feed.
STATE_SOURCE_SURFACE = ("resolved", "wheel_meta", "dt", "dev", "fdt", "physics",
                        "chassis", "omega", "prev_compression", "distances")


def is_state_source(obj: Any) -> bool:
    """True if ``obj`` already provides the whole state-source surface."""
    return all(hasattr(obj, name) for name in STATE_SOURCE_SURFACE)


class _StateSource:
    """Everything the unroll needs, pulled from one live vehicle.

    Resolves both driver layouts: a plain :class:`VehiclePhysics`
    (``VehicleScene(solver="per_vehicle")``) and one vehicle slot of a batched
    :class:`MultiVehicleKindPhysics` (the default ``solver="batched"``), where
    the per-wheel state lives in the kind's shared ``(n_envs*K, n_wheels)``
    proto and this vehicle is row ``env_idx * K + slot``.
    """

    def __init__(self, source: Any, env_idx: int = 0):
        self.env_idx = int(env_idx)
        physics, entity, row, n_envs, K = self._resolve(source, self.env_idx)
        self.physics = physics
        self.entity = entity
        self.row = row
        self.n_envs = n_envs
        self.K = K
        self.resolved = physics.resolved
        self.wheel_meta = physics.wheel_meta
        self.dt = float(physics.dt)
        self.dev = physics.dev
        self.fdt = physics.fdt

    @staticmethod
    def _resolve(source, env_idx):
        # A VehicleScene Vehicle handle.
        scene = getattr(source, "_scene", None)
        if scene is not None and hasattr(source, "_slot"):
            entity = source.entity_main
            if getattr(source, "physics", None) is not None:      # per_vehicle
                p = source.physics
                return p, entity, env_idx, p.n_envs, 1
            mvp = getattr(scene, "_mvp", None)
            if mvp is None:
                raise ValueError(
                    "this Vehicle has no physics yet — call VehicleScene.build() "
                    "before building a DifferentiablePlant")
            for flat_i, kind_idx, slot_idx in mvp._flat_to_kind:
                if flat_i == source._slot:
                    kind = mvp.kinds[kind_idx]
                    return (kind._proto, entity, env_idx * kind.K + slot_idx,
                            kind.n_envs, kind.K)
            raise ValueError(f"vehicle slot {source._slot} not found in the scene's kinds")

        # A VehiclePhysics (or a kind's proto) directly.
        if hasattr(source, "resolved") and hasattr(source, "wheel_meta"):
            entity = getattr(source, "entity", None)
            if entity is None:
                raise ValueError("state source has no .entity to read chassis state from")
            return source, entity, env_idx, getattr(source, "n_envs", 1), 1

        raise TypeError(
            "DifferentiablePlant needs a VehicleScene Vehicle handle or a "
            f"VehiclePhysics; got {type(source).__name__}")

    # -- reads -----------------------------------------------------------------

    def _row(self, t):
        if t is None:
            return None
        return t[self.row] if t.dim() >= 2 else t

    def chassis(self):
        """(pos, quat, vel, ang) as numpy (3,)/(4,) for THIS vehicle+env."""
        ent = self.entity

        def _pick(t):
            a = _as_np(t)
            return a[self.env_idx] if a.ndim == 2 else a

        return (_pick(ent.get_pos()), _pick(ent.get_quat()),
                _pick(ent.get_vel()), _pick(ent.get_ang()))

    def omega(self) -> torch.Tensor:
        return self._row(self.physics.omega).detach().clone()

    def prev_compression(self) -> torch.Tensor:
        return self._row(self.physics.prev_compression).detach().clone()

    def distances(self) -> Optional[torch.Tensor]:
        d = getattr(self.physics, "last_distances", None)
        if d is None:
            return None
        return self._row(d).detach().clone()


# --- The plant ----------------------------------------------------------------

class DifferentiablePlant:
    """Sweep-table-free inverse plant: autodiff Jacobian + Newton solve.

    Drop-in for :class:`~genesis_vehicle.control.SweepTable` in
    :class:`~genesis_vehicle.control.PathFollower` — it answers the same
    ``throttle_for`` / ``steer_for`` questions, and additionally offers the
    coupled :meth:`solve` (both channels from one 2x2 inversion), which the
    follower prefers when present.

    Args:
        source: the live vehicle — a ``VehicleScene`` ``Vehicle`` handle
            (either solver mode) or a :class:`VehiclePhysics`.
        env_idx: which Genesis env to read when the scene is batched.
        horizon: unroll length in SIM STEPS, and the plant's cost knob (the
            solve is linear in it). Throttle reaches the ground only after the
            wheel-omega update — omega first, slip second, force third — so a
            1-step unroll sees the response only through the overshoot clamp
            and is rejected. The default 4 (0.1 s at 40 Hz) measured both faster
            AND tighter than 8 on the reference tank course: a controller
            running every step wants the near-term response, and a longer
            horizon also drifts further from the frozen-normal-load assumption.
            Raise it if the vehicle's tyre transients are slower than 0.1 s.
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
        mass / izz / com: override the planar mass properties, which are
            otherwise composed from the entity's links. Giving both
            `mass` and `izz` skips that read entirely, and `com` then
            defaults to the base-link origin.
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

    Example::

        plant = DifferentiablePlant(veh)          # no CSV anywhere
        follower = PathFollower(path, plant)
        thr, steer, brake = follower.step(*state)
    """

    #: Marks the coupled two-channel solve, which PathFollower prefers.
    has_coupled_solve = True

    def __init__(self, source: Any, *,
                 env_idx: int = 0,
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
        # the top of every solve so the prediction still starts where the
        # vehicle actually is.
        p = self.src.physics
        self._pred_pre_hooks = copy.deepcopy(list(p.pre_loop_hooks))
        self._pred_post_hooks = copy.deepcopy(list(p.post_tire_hooks))

        # Warm start + diagnostics. Everything a caller needs to see why a
        # command came out the way it did: `last_jacobian` is the 2x2 the solve
        # linearised on, `last_response` the (a, omega_z) it converged to,
        # `last_target` what was asked for, `last_singular` whether a channel
        # had no authority at this operating point.
        self.last_u = (0.0, 0.0)
        self.last_jacobian: Optional[np.ndarray] = None
        self.last_response: Optional[tuple] = None
        self.last_target: Optional[tuple] = None
        self.last_iters = 0
        self.last_cost = 0.0
        self.last_singular = False
        self._pending_a: Optional[float] = None

    # -- introspection ---------------------------------------------------------

    @property
    def dt(self) -> float:
        return self.src.dt

    def __repr__(self) -> str:
        return (f"DifferentiablePlant(n_wheels={self.src.wheel_meta.n_wheels}, "
                f"horizon={self.horizon}, dt={self.dt:.4f}, "
                f"{self.mass_props!r})")

    # -- prediction ------------------------------------------------------------

    def _sync_hooks(self) -> None:
        p = self.src.physics
        for live, pred in ((p.pre_loop_hooks, self._pred_pre_hooks),
                           (p.post_tire_hooks, self._pred_post_hooks)):
            for lh, ph in zip(live, pred):
                for k, v in lh.__dict__.items():
                    if torch.is_tensor(v):
                        setattr(ph, k, v.detach().clone())

    def _snapshot(self) -> dict:
        """Freeze the live state the next unroll starts from."""
        src = self.src
        pos, quat, vel, ang = src.chassis()
        R = _quat_to_R(quat)
        dev, fdt = src.dev, src.fdt
        dist = src.distances()
        if dist is None:
            raise RuntimeError(
                "the vehicle has no wheel-ground distances yet — step the scene "
                "at least once before asking the plant for a command")
        v_b = R.T @ np.asarray(vel, dtype=np.float64).reshape(3)
        w_b = R.T @ np.asarray(ang, dtype=np.float64).reshape(3)
        g_b = R.T @ np.array([0.0, 0.0, -9.81])
        return {
            "distances": dist.reshape(1, -1).to(device=dev, dtype=fdt),
            "omega": src.omega().reshape(1, -1).to(device=dev, dtype=fdt),
            "prev_compression": src.prev_compression().reshape(1, -1).to(device=dev, dtype=fdt),
            "v_b": v_b, "w_z": float(w_b[2]), "g_b": g_b,
        }

    def _unroll(self, throttle: torch.Tensor, steer: torch.Tensor,
                snap: dict) -> tuple:
        """H steps of the real wheel pipeline + a planar chassis integrator.

        ``throttle`` / ``steer`` are ``(B,)``: the whole method is batched over
        candidate COMMANDS, so a line search evaluates all of its candidates in
        one pass. On CPU at these sizes the unroll is dominated by torch's
        per-op dispatch, not by arithmetic, so B candidates cost barely more
        than one.

        Returns ``(a, omega_z)``, both ``(B,)`` and graph-carrying.
        """
        src = self.src
        dev, fdt = src.dev, src.fdt
        wm = src.wheel_meta
        resolved = src.resolved
        DT = src.dt
        n = wm.n_wheels
        B = int(throttle.reshape(-1).shape[0])

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

        brake_t = torch.full((B,), self.brake, device=dev, dtype=fdt)
        omega = snap["omega"].expand(B, n)
        prev_comp = snap["prev_compression"].expand(B, n)
        distances = snap["distances"].expand(B, n)

        # Planar state, body frame, referenced at the base-link origin.
        vx = torch.full((B,), float(snap["v_b"][0]), device=dev, dtype=fdt)
        vy = torch.full((B,), float(snap["v_b"][1]), device=dev, dtype=fdt)
        wz = torch.full((B,), float(snap["w_z"]), device=dev, dtype=fdt)
        vx0 = vx
        gx = float(snap["g_b"][0]); gy = float(snap["g_b"][1])

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

    def predict(self, throttle, steer, snap: Optional[dict] = None):
        """Forward-only prediction of ``(a, omega_z)`` for a held command.

        Scalars in, scalars out. Pass sequences to evaluate several candidate
        commands in ONE unroll — the line search does exactly that — and get
        two ``(B,)`` arrays back.
        """
        if snap is None:
            snap = self._snapshot()
        dev, fdt = self.src.dev, self.src.fdt
        batched = np.ndim(throttle) > 0 or np.ndim(steer) > 0
        t = torch.as_tensor(np.atleast_1d(np.asarray(throttle, dtype=np.float64)),
                            device=dev, dtype=fdt)
        s = torch.as_tensor(np.atleast_1d(np.asarray(steer, dtype=np.float64)),
                            device=dev, dtype=fdt)
        if t.shape != s.shape:
            t, s = torch.broadcast_tensors(t, s)
        # The prediction hooks carry cross-step state, so every unroll has to
        # start from the live vehicle's — otherwise candidate N inherits
        # candidate N-1's stick-slip anchors AND keeps its autograd graph
        # alive, which turns repeated calls quadratic.
        self._sync_hooks()
        with torch.no_grad():
            a, w = self._unroll(t, s, snap)
        a = a.detach().cpu().numpy()
        w = w.detach().cpu().numpy()
        return (a, w) if batched else (float(a[0]), float(w[0]))

    def jacobian(self, throttle: Optional[float] = None, steer: Optional[float] = None,
                 snap: Optional[dict] = None) -> tuple:
        """``(J, y)`` at the given command — ``J`` is 2x2, ``y`` is ``(a, omega_z)``.

        Rows are ``(a, omega_z)``, columns ``(throttle, steer)``.
        """
        if snap is None:
            snap = self._snapshot()
        u0 = self.last_u
        thr = u0[0] if throttle is None else float(throttle)
        st = u0[1] if steer is None else float(steer)
        dev, fdt = self.src.dev, self.src.fdt
        self._sync_hooks()
        t = torch.tensor([float(thr)], device=dev, dtype=fdt, requires_grad=True)
        s = torch.tensor([float(st)], device=dev, dtype=fdt, requires_grad=True)
        a, w = self._unroll(t, s, snap)
        y = (float(a[0].detach()), float(w[0].detach()))
        outs = torch.cat([a.reshape(1), w.reshape(1)])
        J = np.zeros((2, 2))
        try:
            # Both rows of the Jacobian in ONE vmapped backward instead of two
            # sequential ones — the unroll graph is deep (horizon x pipeline)
            # and walking it twice is the single largest cost in the solve.
            seeds = torch.eye(2, device=outs.device, dtype=outs.dtype)
            g = torch.autograd.grad(outs, (t, s), grad_outputs=seeds,
                                    is_grads_batched=True, allow_unused=True)
            for c in (0, 1):
                if g[c] is not None:
                    J[:, c] = g[c].reshape(2).detach().cpu().numpy()
        except (RuntimeError, NotImplementedError):
            # vmap-over-backward is not supported for every op; fall back to
            # the plain two-pass form.
            for i, out in enumerate((a, w)):
                gi = torch.autograd.grad(out.sum(), (t, s), retain_graph=(i == 0),
                                         allow_unused=True)
                J[i, 0] = 0.0 if gi[0] is None else float(gi[0])
                J[i, 1] = 0.0 if gi[1] is None else float(gi[1])

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
        # So: where a whole column of the Jacobian vanishes, replace it with a
        # one-sided difference over `probe_delta`, large enough to step out of
        # the dead zone. Both probes ride one batched unroll.
        if self.probe_delta > 0.0:
            dead = np.flatnonzero(np.abs(J).max(axis=0) <= self.probe_threshold)
            if dead.size:
                u = np.array([thr, st])
                lo = np.array([self.throttle_range[0], self.steer_range[0]])
                hi = np.array([self.throttle_range[1], self.steer_range[1]])
                trials = np.repeat(u[None, :], dead.size, axis=0)
                steps = np.zeros(dead.size)
                for i, c in enumerate(dead):
                    # Probe towards whichever side has room (a saturated
                    # command has room only inwards).
                    d = self.probe_delta if u[c] + self.probe_delta <= hi[c] \
                        else -self.probe_delta
                    d = max(min(d, hi[c] - u[c]), lo[c] - u[c])
                    trials[i, c] = u[c] + d
                    steps[i] = d
                ok = np.abs(steps) > 1e-9
                if ok.any():
                    a_t, w_t = self.predict(trials[:, 0], trials[:, 1], snap=snap)
                    for i, c in enumerate(dead):
                        if not ok[i]:
                            continue
                        J[0, c] = (float(a_t[i]) - y[0]) / steps[i]
                        J[1, c] = (float(w_t[i]) - y[1]) / steps[i]
        return J, y

    # -- inversion -------------------------------------------------------------

    def _step_command(self, J, err, u):
        """One command update from a Jacobian and a response error.

        Damped least squares rather than a straight ``J^-1 e``. The plant has
        genuine local singularities — a skid-steer at full steer differential
        has no throttle authority over ``a`` at all, and every vehicle loses
        yaw authority at a standstill — where the exact inverse asks for an
        unbounded command and the loop turns bang-bang. DLS degrades to "this
        channel cannot buy anything here, so leave it alone", which is the
        correct action; :class:`PathFollower`'s low-speed KICK covers pulling
        away from rest, where no plant inverse can help.

        Rows are normalised first so the ``m/s^2`` and ``rad/s`` residuals are
        weighted comparably and one damping constant serves both.
        """
        thr, st = u
        J = np.nan_to_num(np.asarray(J, dtype=np.float64), nan=0.0,
                          posinf=0.0, neginf=0.0)
        e = np.asarray(err, dtype=np.float64)
        floors = np.array([self.min_gain_a, self.min_gain_w])

        if self.coupled:
            rows = np.linalg.norm(J, axis=1)
            weak = rows < floors
            scale = np.where(rows > 1e-12, rows, 1.0)
            Jn = J / scale[:, None]
            en = np.where(weak, 0.0, e / scale)
            A = Jn.T @ Jn + self.damping * np.eye(2)
            du = np.linalg.solve(A, Jn.T @ en)
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
            g = np.array([J[0, 0], J[1, 1]])
            weak = np.abs(g) < floors
            du = np.where(weak, 0.0, e / np.where(weak, 1.0, g))

        du = np.clip(du, -self.max_delta, self.max_delta)
        thr = min(max(thr + float(du[0]), self.throttle_range[0]), self.throttle_range[1])
        st = min(max(st + float(du[1]), self.steer_range[0]), self.steer_range[1])
        return (thr, st), bool(weak.any())

    def solve(self, v_long: float, a_target: float, omega_target: float,
              pitch_deg: float = 0.0, roll_deg: float = 0.0) -> tuple:
        """Coupled inverse: ``(throttle, steer)`` hitting ``(a_target, omega_target)``.

        ``v_long`` / ``pitch_deg`` / ``roll_deg`` are accepted for signature
        symmetry with :class:`SweepTable` but are NOT used — the plant reads
        the live speed and attitude off the vehicle itself, which is strictly
        more information than those three scalars carry.
        """
        self._sync_hooks()
        snap = self._snapshot()
        target = np.array([float(a_target), float(omega_target)])
        self.last_target = (float(a_target), float(omega_target))
        w = np.array([1.0 / self.scale_a, 1.0 / self.scale_w])

        def cost(y):
            return float(np.linalg.norm((target - np.asarray(y)) * w))

        u = self.last_u
        J, y = self.jacobian(u[0], u[1], snap=snap)
        self.last_jacobian = J
        self.last_response = y
        best_cost = cost(y)
        singular = False
        iters = 0

        # Chord iterations with a backtracking line search. Reusing the same J
        # keeps each iteration to one FORWARD unroll (no backward), and the
        # line search is what makes the loop safe where the plant is kinked:
        # the rev limiter and the friction-circle clamp both put corners in the
        # response, and a Jacobian read on the wrong side of one can point
        # uphill. Rather than guess which derivatives to trust, take the step
        # only if the predicted residual actually shrinks.
        for _ in range(max(1, self.newton_iters)):
            if best_cost < self.tol:
                break
            cand, sing = self._step_command(J, target - np.asarray(y), u)
            singular = singular or sing
            du = np.array(cand) - np.array(u)
            if not np.any(np.abs(du) > 1e-12):
                break
            # All backtracking candidates in ONE batched unroll: the halvings
            # are independent, and a B-wide unroll costs almost the same as a
            # single one at these sizes.
            fracs = 0.5 ** np.arange(self.line_search_steps)
            trials = np.array(u)[None, :] + du[None, :] * fracs[:, None]
            a_t, w_t = self.predict(trials[:, 0], trials[:, 1], snap=snap)
            costs = np.linalg.norm((target[None, :] - np.stack([a_t, w_t], 1)) * w[None, :],
                                   axis=1)
            k = int(np.argmin(costs))
            iters += 1
            if costs[k] >= best_cost:
                break
            u = (float(trials[k, 0]), float(trials[k, 1]))
            y = (float(a_t[k]), float(w_t[k]))
            best_cost = float(costs[k])

        self.last_u = (float(u[0]), float(u[1]))
        self.last_response = y
        self.last_iters = iters
        self.last_cost = best_cost
        self.last_singular = singular
        return self.last_u

    # -- SweepTable-compatible surface ----------------------------------------
    #
    # These exist so the plant can stand in for a SweepTable anywhere, but they
    # cost one solve each. PathFollower checks `has_coupled_solve` and calls
    # solve() once per step instead.

    def throttle_for(self, v: float, a_target: float,
                     pitch_deg: float = 0.0, roll_deg: float = 0.0) -> float:
        """Throttle for ``a_target``, yaw rate left where it is.

        Remembers ``a_target`` so a following :meth:`steer_for` — the order
        :class:`PathFollower` calls them in — can solve both channels together
        instead of fighting this one.
        """
        self._pending_a = float(a_target)
        return self.solve(v, a_target, self._held_omega(), pitch_deg, roll_deg)[0]

    def steer_for(self, v: float, omega_target: float,
                  pitch_deg: float = 0.0, roll_deg: float = 0.0) -> float:
        """Steer for ``omega_target``, honouring a pending :meth:`throttle_for`."""
        a_t = self._pending_a if self._pending_a is not None else self._held_a()
        self._pending_a = None
        return self.solve(v, a_t, omega_target, pitch_deg, roll_deg)[1]

    def _held_omega(self) -> float:
        r = self.last_response
        return 0.0 if r is None else r[1]

    def _held_a(self) -> float:
        r = self.last_response
        return 0.0 if r is None else r[0]

    def set_applied(self, throttle: float, steer: float) -> None:
        """Tell the plant what the vehicle was ACTUALLY commanded this step.

        The Jacobian is a linearisation about the warm-start command, so the
        warm start has to be the command the vehicle really got. A caller that
        post-processes the solve — :class:`PathFollower` clamps steer to its
        own ``steer_cap`` and overrides throttle outright with its low-speed
        KICK — must report the result back, or every following step
        differentiates the plant at an operating point the vehicle was never
        in. That mismatch is invisible while the two agree and divergent the
        moment they do not.
        """
        self.last_u = (float(throttle), float(steer))

    def reset(self) -> None:
        """Drop the warm start (after a teleport / scenario reset)."""
        self.last_u = (0.0, 0.0)
        self.last_jacobian = None
        self.last_response = None
        self.last_target = None
        self._pending_a = None
