"""Path -> per-step (Steer, Throttle) controller.

Give ``PathFollower`` a path (waypoints with signed target speeds) and an
*inverse plant* — what turns a desired ``(a, omega_z)`` into a
``(throttle, steer)`` command — and call :meth:`PathFollower.step` every
simulation step with the current chassis state; it returns
``(throttle, steer, brake)`` ready for ``vehicle.set_inputs(...)``.

Since 2.0.0 a step has two stages. Stage 1 decides the physical response the
vehicle should produce, ``(a, omega_z)``: an MPC layer (``mpc_model="bicycle"``,
the default; needs ``mpc_params``) on forward blocks of the path, and the 1.x
PD law everywhere the MPC does not run — reverse blocks, a failed solve under
``on_solve_failure="pd"``, and every block when ``mpc_model="off"`` (the 1.x
behaviour). Stage 2, the inverse plant, turns ``(a, omega_z)`` into
``(throttle, steer)``.

The default stack since 2.0.0 is a sweep table + the MPC layer;
:class:`~genesis_vehicle.control.DifferentiablePlant` is the no-measurement
option. (Through 1.8.0 this docstring called ``DifferentiablePlant`` "the
default since v1.3.0"; that stopped being true in 2.0.0 and was corrected.)
Two inverse plants ship with the SDK, and ``PathFollower`` takes either in the
same argument slot:

- :class:`~genesis_vehicle.control.sweep.SweepTable` — a 5-D response grid
  measured offline by ``sweep_measure`` and loaded from CSV. It must be
  re-measured whenever the plant changes, and it is the only option for a
  process with no simulator.
- :class:`~genesis_vehicle.control.DifferentiablePlant` — differentiates the
  vehicle's own ray-wheel force model with ``torch.autograd`` and inverts the
  resulting 2x2 Jacobian each step. Needs the live vehicle handle and NO
  measurement, NO CSV.

The path is split into *direction blocks* — maximal runs of waypoints whose
``target_speed`` shares one sign. Within a block the controller is a plain
pursuit loop; a block boundary is a cusp (gear change): the vehicle
decelerates toward the boundary waypoint, brakes to a stop, then continues
into the next block with the flipped direction. Projection and lookahead
never cross the boundary of the active block, so approaching a cusp cannot
contaminate the current direction's geometry (the pre-SDK deliverable let
the lookahead leak across the cusp, which only behaved on collinear
forward/backward paths).

Per-step pipeline (within the active block):

    1. project the vehicle position onto the block's segments
    2. find the waypoint ``lookahead`` metres ahead (capped at block end)
    3. v_target from that waypoint; if the block end is inside the
       lookahead window and a cusp follows, taper |v_target| with the
       remaining distance (decelerate into the cusp)
    4. heading_err = 0.7 * yaw_err + 0.3 * position_err — the reference
       yaw per waypoint is its EXPLICIT yaw when given (optional 5th
       tuple element, world +X = 0, CCW positive), else the tangential
       direction (+pi-flipped on backward waypoints)
    5. P control: a_target = k_v * (v_target - v_long),
                  omega_target = k_w * heading_err
    6. plant inverse: (throttle, steer) = plant^-1(v, a_target,
                      omega_target, pitch, roll) — one coupled 2x2 solve for a
                      DifferentiablePlant, two independent grid inversions for
                      a SweepTable
    7. low-speed KICK (saturate throttle when far below target speed) and
       brake-at-rest handling

The controller itself is simulator-agnostic (numpy/math only): feed it state
from Genesis via :func:`extract_state`, or from any engine via
:func:`extract_state_from_arrays`. That holds end-to-end with a
:class:`SweepTable`; a :class:`DifferentiablePlant` reads the live vehicle, so
that combination is Genesis-bound by construction.

Tuned on skid-steer (tank) vehicles; Ackermann vehicles generally work,
but re-measure the sweep so the steer sign convention of the actual
drivetrain is baked into the table (see ``docs/path-following.md``).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

from .sweep import SweepTable
from ..gear import GEAR_SHIFT_MAX_SPEED, refuses_gear


# --- State extraction --------------------------------------------------------

def extract_state_from_arrays(pos_xyz, quat_wxyz, vel_xyz, ang_xyz=None) -> dict:
    """Simulator-agnostic chassis-state extraction.

    Args:
        pos_xyz: world position (3,).
        quat_wxyz: world orientation quaternion, scalar-first (w, x, y, z).
        vel_xyz: world linear velocity (3,).
        ang_xyz: world angular velocity (3,), optional (v2.0.0).

    Returns:
        dict with ``pos_xy`` (m), ``yaw`` (rad), ``v_long`` (m/s, signed along
        chassis +X), ``pitch``/``roll`` (DEGREES — the sweep grid unit). With
        ``ang_xyz`` also ``yaw_rate`` (rad/s: the WORLD z angular velocity,
        which equals the chassis yaw rate only on level ground) and ``v_lat``
        (m/s: ``-vx*sin(yaw) + vy*cos(yaw)``, + = left) — the two extra states
        the MPC layer needs.
    """
    w = float(quat_wxyz[0]); x = float(quat_wxyz[1])
    y = float(quat_wxyz[2]); z = float(quat_wxyz[3])
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    cy, sy = math.cos(yaw), math.sin(yaw)
    v_long = float(vel_xyz[0]) * cy + float(vel_xyz[1]) * sy
    out = {
        "pos_xy": (float(pos_xyz[0]), float(pos_xyz[1])),
        "yaw": yaw, "v_long": v_long,
        "pitch": math.degrees(pitch), "roll": math.degrees(roll),
    }
    if ang_xyz is not None:
        out["yaw_rate"] = float(ang_xyz[2])
        out["v_lat"] = -float(vel_xyz[0]) * sy + float(vel_xyz[1]) * cy
    return out


def extract_state(vehicle, env_idx: int = 0) -> dict:
    """State extraction for a Genesis entity or a ``VehicleScene`` Vehicle.

    Works with anything exposing ``get_pos()/get_quat()/get_vel()`` (both a
    raw Genesis entity and the SDK ``Vehicle`` wrapper do). For batched
    scenes (``n_envs > 1``) pass ``env_idx``. If the object also has
    ``get_ang()`` the dict carries ``yaw_rate`` and ``v_lat`` (see
    :func:`extract_state_from_arrays`); without it those two keys are absent.
    """
    def _row(t):
        arr = t.cpu().numpy() if hasattr(t, "cpu") else t
        return arr[env_idx] if getattr(arr, "ndim", 1) == 2 else arr
    get_ang = getattr(vehicle, "get_ang", None)
    return extract_state_from_arrays(
        _row(vehicle.get_pos()), _row(vehicle.get_quat()), _row(vehicle.get_vel()),
        None if get_ang is None else _row(get_ang()))


# --- The two halves of a control step -----------------------------------------

@dataclass
class FollowerPlan:
    """What :meth:`PathFollower.plan` decided, before any vehicle is consulted.

    ``command`` set means the step needs no inversion at all — the path state
    machine already knows the answer (arrived, or braking into a cusp).
    Otherwise ``a_target`` / ``omega_target`` are the physical response the
    inverse plant has to deliver.
    """
    mode: str
    a_target: float = 0.0
    omega_target: float = 0.0
    v_long: float = 0.0
    v_target: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    command: Optional[tuple] = None
    #: (v2.0.0, appended) which stage 1 produced this plan: ``"R-OFF"`` (PD,
    #: MPC layer off), ``"R-MPC"``, ``"R-REV"`` (PD in a reverse block),
    #: ``"R-FAIL"`` (PD after a failed solve), ``"CMD"`` (DONE / cusp brake /
    #: since v2.0.1 the terminal ``GOAL-MISSED`` and ``MPC-ABORT``, with the
    #: MPC layer on; with it off those are ``"R-OFF"``), and since v2.0.1
    #: ``"R-ALIGN"`` (PD on a
    #: forward block while the heading gate holds, ``mpc_align_gate``) and
    #: ``"R-STALL"`` (PD after the no-progress watchdog fired,
    #: ``mpc_stall_timeout``). ``a_ff`` / ``w_ff``: the MPC feedforward.
    #: Under R-MPC ``v_target`` is the PD's value, which since v2.0.1 can
    #: differ in sign from 1.x at a block end (W2); the MPC command does not
    #: use it.
    regime: str = "R-OFF"
    a_ff: Optional[float] = None
    w_ff: Optional[float] = None
    #: (v2.0.2, appended) gear mode only (``command_mode="gear"``; ``None``
    #: otherwise): the gear the command of this step requests — the active
    #: block's sign, except on a terminal step (``DONE`` / ``GOAL-MISSED`` /
    #: ``MPC-ABORT`` / ``GEAR-ABORT``), which keeps the gear the follower
    #: believes engaged. ``a_coast``: the plant's coast acceleration (throttle
    #: 0, brake 0) for the brake model, filled by :class:`FleetFollower` from
    #: one batched prediction; ``None`` lets :meth:`PathFollower.finish` ask
    #: the plant's ``accel_at`` when it needs it.
    gear: Optional[int] = None
    a_coast: Optional[float] = None


@dataclass(frozen=True)
class GearCommand:
    """(v2.0.2) What a gear-mode :class:`PathFollower` (``command_mode=
    "gear"``, the default with the MPC layer on) returns from :meth:`step` /
    :meth:`finish`: ``throttle`` in [0, 1] (a magnitude, applied in ``gear``'s
    direction by a vehicle in gear mode), ``steer`` in [-steer_cap, steer_cap],
    ``brake`` in [0, 1], ``gear`` +1 / -1. Apply it with
    ``veh.set_inputs(throttle=c.throttle, brake=c.brake, steer=c.steer,
    gear=c.gear)``.

    Iterates as ``throttle, steer, brake, gear = cmd`` (so the 2.0.1
    ``thr, st, br = cmd`` raises ``ValueError``). NOT indexable (``cmd[0]``
    raises ``TypeError``): an index would silently hand a legacy signed
    vehicle the unsigned magnitude and drive a reverse block forwards.
    ``cmd == (0, 0, 1)`` is always False (a dataclass is not a tuple) and
    ``np.asarray(cmd)`` is a 0-d object array — compare fields or
    :meth:`signed`. :meth:`signed` gives the 2.0.1 3-tuple for a consumer that
    has no gear (an OSC client, a legacy vehicle); a vehicle driven that way
    does not enforce the shift rule."""
    throttle: float
    steer: float
    brake: float
    gear: int

    def __iter__(self):
        return iter((self.throttle, self.steer, self.brake, self.gear))

    def signed(self) -> tuple:
        """``(gear * throttle, steer, brake)`` — the signed 2.0.1 command."""
        return (self.gear * self.throttle, self.steer, self.brake)


def _predict_brake0(plant, throttle, steer):
    """``plant.predict(throttle, steer)`` with the plant's constant
    prediction ``brake`` (a ``DifferentiablePlant`` constructor argument) held
    at 0 for the call, so the brake model does not count a brake twice."""
    saved = getattr(plant, "brake", None)
    if saved is not None:
        plant.brake = 0.0
    try:
        return plant.predict(throttle, steer)
    finally:
        if saved is not None:
            plant.brake = saved


class FleetFollower:
    """Drives N :class:`PathFollower` s off ONE batched inverse plant.

    A follower is per-vehicle by nature — its path, its block, its cusp state
    machine — but the expensive half of a step is the inversion, and a
    :class:`~genesis_vehicle.control.DifferentiablePlant` can invert a whole
    fleet in one unroll. So the step runs in three phases rather than N
    independent ones::

        plan   (per follower, cheap geometry)
        solve  (ONCE, batched over every member)
        finish (per follower, caps + KICK + brake)

    The plant must have one member per follower, in the same order.

    Example::

        plant = DifferentiablePlant(tanks)             # M members
        fleet = FleetFollower([PathFollower(p, plant, mpc_model="off")
                               for p in paths], plant)
        cmds = fleet.step([extract_state(t) for t in tanks])

    Each member stops on its own terminal mode (``DONE``, ``GOAL-MISSED``,
    and in gear mode ``GEAR-ABORT``, v2.0.2); the fleet does not aggregate — check
    ``fleet.last_modes`` against ``PathFollower.TERMINAL_MODES``.
    """

    def __init__(self, followers, plant):
        followers = list(followers)
        # Before anything is touched: a rejected call leaves the followers as
        # they were (the loop at the end of __init__ mutates them).
        mpc = [i for i, f in enumerate(followers)
               if getattr(f, "mpc_model", "off") != "off"]
        if mpc:
            raise TypeError(
                f"FleetFollower runs PD followers only; follower(s) {mpc} have the "
                "MPC layer on — construct them with mpc_model=\"off\" (the fleet "
                "step passes no yaw_rate / v_lat, and N MPC solves per step do not "
                "fit a control period)")
        modes = sorted({getattr(f, "command_mode", "signed") for f in followers})
        if len(modes) > 1:
            raise ValueError(
                f"FleetFollower needs one command_mode for every follower; got {modes}")
        n = int(getattr(plant, "n_members", 1))
        if n != len(followers):
            raise ValueError(
                f"the plant serves {n} members but there are {len(followers)} "
                "followers — they must correspond one to one, in order")
        if not getattr(plant, "has_coupled_solve", False):
            raise TypeError(
                "FleetFollower needs a plant with a batched solve() "
                "(DifferentiablePlant); a SweepTable is inverted per vehicle "
                "anyway, so plain PathFollower.step() is the right call there")
        self.followers = followers
        self.plant = plant
        for i, f in enumerate(followers):
            f.plant = plant
            f.plant_member = i
            f._coupled_plant = True
        self.last_modes = ["INIT"] * len(followers)

    def __len__(self) -> int:
        return len(self.followers)

    def step(self, states):
        """One control step for the whole fleet.

        ``states`` is a sequence of the dicts :func:`extract_state` returns,
        or of ``(pos_xy, yaw, v_long, pitch_deg, roll_deg)`` tuples — one per
        follower. Returns a list of ``(throttle, steer, brake)`` — or, when the
        followers run ``command_mode="gear"`` (v2.0.2), of
        :class:`GearCommand`. In gear mode a dict state's optional ``"gear"``
        key is that vehicle's engaged gear (feedback, see
        :meth:`PathFollower.step`); :func:`extract_state` does not set it.
        """
        states = list(states)
        if len(states) != len(self.followers):
            raise ValueError(
                f"got {len(states)} states for {len(self.followers)} followers")

        plans = []
        for f, st in zip(self.followers, states):
            fb = None
            if isinstance(st, dict):
                fb = st.get("gear")
                st = (st["pos_xy"], st["yaw"], st["v_long"],
                      st["pitch"], st["roll"])
            if fb is None:
                plans.append(f.plan(*st))
            else:
                plans.append(f.plan(*st, gear=fb))

        # One inversion for everyone. Followers whose step is already decided
        # still occupy a slot — asking for their current response makes their
        # solve a no-op, and a batched unroll does not care about the width.
        v_long = [p.v_long for p in plans]
        a_t = [p.a_target for p in plans]
        w_t = [p.omega_target for p in plans]
        thr, steer = self.plant.solve(v_long, a_t, w_t)
        if self.followers and getattr(self.followers[0], "command_mode", "signed") == "gear":
            # (v2.0.2) the brake model's coast acceleration, ONE batched
            # prediction at throttle 0 for every member, only when some
            # member will brake this step.
            need = [i for i, p in enumerate(plans)
                    if p.command is None and p.gear * float(thr[i]) < 0.0]
            if need:
                import numpy as np
                a0, _ = _predict_brake0(self.plant, np.zeros(len(plans)),
                                        np.asarray(steer, dtype=float))
                a0 = np.atleast_1d(a0)
                for i in need:
                    plans[i].a_coast = float(a0[i])

        out = [f.finish(p, thr[i], steer[i])
               for i, (f, p) in enumerate(zip(self.followers, plans))]
        self.last_modes = [f.last_mode for f in self.followers]
        return out


# --- Inverse-plant coercion ---------------------------------------------------

def _as_plant(plant, plant_kwargs: dict):
    """Coerce whatever the caller passed into an object with the inverse surface.

    Accepts (in order): something that already answers ``throttle_for`` /
    ``steer_for``; a CSV path; a live vehicle (``Vehicle`` handle or
    ``VehiclePhysics``), which is wrapped in a
    :class:`~genesis_vehicle.control.plant.DifferentiablePlant`.
    """
    if isinstance(plant, (str, os.PathLike)):
        return SweepTable.load(plant)
    if hasattr(plant, "throttle_for") and hasattr(plant, "steer_for"):
        return plant
    # A live vehicle: VehicleScene's Vehicle handle, or a VehiclePhysics.
    looks_live = (hasattr(plant, "_slot") and hasattr(plant, "_scene")) or (
        hasattr(plant, "resolved") and hasattr(plant, "wheel_meta"))
    if looks_live:
        from .plant import DifferentiablePlant
        return DifferentiablePlant(plant, **plant_kwargs)
    raise TypeError(
        "PathFollower's plant must be a DifferentiablePlant, a SweepTable, a "
        "sweep-CSV path, or a live vehicle (Vehicle handle / VehiclePhysics); "
        f"got {type(plant).__name__}")


# --- MPC-layer helpers (v2.0.0) ------------------------------------------------

class _Unset:
    """Default of an MPC-only constructor argument: lets ``mpc_model="off"``
    tell "not given" from "given with the default value". Reads as its value
    in ``inspect.signature`` / the docs."""
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return repr(self.value)


#: v2.0.1 watchdog distances (m), chosen values, not public: the anchor moves
#: once the chassis is this far from it; R-STALL ends once the reported
#: position is this far (straight-line displacement) from the R-STALL entry point.
STALL_ANCHOR_M = 0.25
STALL_EXIT_M = 1.0
#: v2.0.1 F5: R-STALL entries allowed per run of entries without
#: STALL_PROGRESS_M of new path progress; the next one aborts.
STALL_MAX_PER_BLOCK = 3
#: F5: new path progress (projection arc length beyond the furthest earlier
#: entry in the block) that starts a new run. Chosen value, not public.
STALL_PROGRESS_M = 3.0
#: v2.0.1 GP-AWAY2 (goal-passed stop), chosen values: the distance to the goal
#: must grow by more than GP_INC_M on each of two consecutive plan() calls, at
#: least GP_MARGIN_M beyond arrival_goal.
GP_INC_M = 1e-3
GP_MARGIN_M = 0.1
# _handover_step results.
_HO_LAW, _HO_PD, _HO_ABORT = 0, 1, 2
#: v2.0.2 gear mode, chosen values: GUARD latches while the vehicle moves
#: faster than V_AGAINST (m/s) against the desired gear; more than
#: SHIFT_MAX_CALLS consecutive SHIFT / GUARD calls end on GEAR-ABORT
#: ("shift_timeout"; 120 calls = 3 s at 40 Hz — a call count, not a time).
V_AGAINST = 0.3
SHIFT_MAX_CALLS = 120
#: Planar speed below which a SHIFT / GUARD call counts as "slow": 90 % of the
#: vehicle's shift limit (GEAR_SHIFT_MAX_SPEED). The margin covers the
#: differences between the follower's ``hypot(v_long, v_lat)`` and the
#: vehicle's own ``hypot(vel_x, vel_y)``: one control step of lag and float32
#: vs float64. They are the same quantity only when the state comes from the
#: same base-link velocity — ``extract_state`` and the drivers read the same
#: one for a base link without a morph offset (checked on the reference car
#: only); without ``v_lat`` the follower uses ``|v_long|``.
SHIFT_SLOW_SPEED = 0.9 * GEAR_SHIFT_MAX_SPEED
COMMAND_MODES = ("gear", "signed")


def _is_real(x) -> bool:
    import numbers
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


def _check_align_gate(gate):
    """``None`` or ``(theta_on, theta_off)`` as floats with
    ``0 < theta_off < theta_on <= pi``; ``ValueError`` otherwise."""
    if gate is None:
        return None
    msg = ("mpc_align_gate must be None or (theta_on, theta_off) in rad with "
           f"0 < theta_off < theta_on <= pi; got {gate!r}")
    if isinstance(gate, (str, bytes)):
        raise ValueError(msg)
    try:
        vals = list(gate)
    except TypeError:
        raise ValueError(msg) from None
    if len(vals) != 2 or not all(_is_real(v) for v in vals):
        raise ValueError(msg)
    on, off = float(vals[0]), float(vals[1])
    # Written so that a NaN fails every comparison and raises.
    if not (math.isfinite(on) and math.isfinite(off) and 0.0 < off < on <= math.pi):
        raise ValueError(msg)
    return (on, off)


def _check_stall_timeout(timeout):
    """``None`` or a finite real ``> 0`` (as float); ``ValueError`` otherwise."""
    if timeout is None:
        return None
    if not _is_real(timeout) or not (math.isfinite(float(timeout)) and float(timeout) > 0.0):
        raise ValueError(f"mpc_stall_timeout must be None or a finite number of seconds > 0; "
                         f"got {timeout!r}")
    return float(timeout)


#: The U1 error: an MPC follower (the 2.0.0 default) without its params.
U1_MESSAGE = (
    "PathFollower runs an MPC layer by default since genesis_vehicle 2.0.0 and needs "
    "this vehicle's MPC parameters: pass mpc_params=\"<vehicle>_mpc_params.json\" "
    "(measure it once with `python -m genesis_vehicle.control.identify --urdf <urdf> "
    "--preset <preset> --output <vehicle>_mpc_params.json`), or pass "
    "mpc_model=\"off\" to run the PD path follower alone (the 1.x behaviour).")


def _looks_live(plant) -> bool:
    return (hasattr(plant, "_slot") and hasattr(plant, "_scene")) or (
        hasattr(plant, "resolved") and hasattr(plant, "wheel_meta"))


def _live_steering(plant):
    """``(steering strategy, n_axles)`` of a live vehicle / DifferentiablePlant,
    or ``(None, None)`` when the plant cannot tell (SweepTable, CSV, other).

    Axles are counted from ``WheelConfig.axle_index``; a wheel whose axle is
    unknown (``None``) makes the count unknown, reported as ``"?"`` — which
    fails the 2-axle check rather than passing as one axle."""
    resolved = None
    src = getattr(plant, "src", None)
    if getattr(plant, "has_coupled_solve", False) and src is not None:
        resolved = getattr(src, "resolved", None)
    elif _looks_live(plant):
        from .plant import _StateSource
        resolved = _StateSource._resolve(plant)[0].resolved
    if resolved is None:
        return None, None
    axles = [getattr(w, "axle_index", None) for w in resolved.wheels]
    n_axles = "?" if any(a is None for a in axles) else len(set(axles))
    return resolved.steering, n_axles


def _is_two_axle_ackermann(steering, n_axles) -> bool:
    from ..strategies.steering import Ackermann        # PartialAckermann subclasses it
    return isinstance(steering, Ackermann) and n_axles == 2


def _cross_check_live(plant, params) -> None:
    """Warn when the params JSON describes a different vehicle than the live
    plant: mass, wheelbase, and (if the JSON has a tyre block) the per-axle
    Pacejka lateral coefficients and friction, relative tolerance 1e-6. Not
    checked on the sweep-table path (a table carries no vehicle data).

    The mass check needs the plant's ``mass_props``; the wheelbase check needs
    ``src.wheel_meta`` with ``positions`` / ``axle_index`` tensors (both
    present on a ``DifferentiablePlant``). When either is missing or cannot be
    read, that check is skipped and ONE ``UserWarning`` names what was not
    checked and why."""
    import warnings
    diffs = []

    def cmp(name, a, b):
        if a is None or b is None:
            return
        a, b = float(a), float(b)
        if abs(a - b) > 1e-6 * max(abs(a), abs(b), 1e-12):
            diffs.append(f"{name}: params {a!r} vs vehicle {b!r}")

    unchecked = []
    mp = getattr(plant, "mass_props", None)
    if mp is not None:
        cmp("mass", params.mass, mp.mass)
    else:
        unchecked.append("mass (the plant has no mass_props)")
    src = getattr(plant, "src", None)
    wm = getattr(src, "wheel_meta", None)
    resolved = getattr(src, "resolved", None)
    if wm is None:
        unchecked.append("wheelbase (the plant exposes no wheel_meta)")
    else:
        try:
            import numpy as np
            pos = wm.positions.detach().cpu().numpy().reshape(-1, 3)
            axle = wm.axle_index.detach().cpu().numpy().reshape(-1)
            xs = [float(pos[axle == k, 0].mean()) for k in np.unique(axle)]
            cmp("wheelbase", params.wheelbase, max(xs) - min(xs))
        except (AttributeError, ValueError) as e:
            unchecked.append(f"wheelbase ({type(e).__name__}: {e})")
    if unchecked:
        warnings.warn("mpc_params could not be cross-checked against this vehicle for: "
                      + "; ".join(unchecked), UserWarning, stacklevel=4)
    tyre = params.tyre
    if isinstance(tyre, dict) and resolved is not None:
        for ax in tyre.get("axles", []):
            wheels = [w for w in resolved.wheels if w.axle_index == ax.get("axle")]
            for key in ("pb_y", "pc_y", "pe_y", "mu_lat", "mu_long"):
                if wheels and key in ax:
                    cmp(f"axle {ax.get('axle')} {key}", ax[key], getattr(wheels[0], key))
    if diffs:
        warnings.warn("mpc_params do not describe this vehicle: " + "; ".join(diffs),
                      UserWarning, stacklevel=4)


# --- Follower -----------------------------------------------------------------

class PathFollower:
    """Pursuit-style follower over a (x, y, z, target_speed) path.

    Args:
        path: list of ``(x, y, z, target_speed)`` or
            ``(x, y, z, target_speed, yaw)``. ``z`` is ignored (the vehicle
            drives on whatever ground the sim provides).
            ``target_speed`` is m/s — positive = forward, negative =
            backward; a sign flip along the path is a cusp (gear change)
            and triggers an automatic decelerate-stop-reverse.
            ``yaw`` (optional 5th element, radians; world +X is 0, CCW
            about +Z positive — the same convention ``extract_state``
            returns) is the DESIRED CHASSIS HEADING at that waypoint,
            used VERBATIM (no backward flip) — give it where the tangential
            default is not what you want, e.g. to control which way the
            chassis faces during a reverse maneuver. ``None`` (or a
            4-tuple) falls back to the tangential direction toward the
            next waypoint, +pi-flipped on backward waypoints.
            Note the heading must stay drivable for a nonholonomic
            vehicle: it can only move along its chassis axis, so an
            explicit yaw far from the travel (or travel+pi) direction
            cannot be tracked.
            Adjacent waypoints should be ~0.3–1 m apart (densify long
            straight segments) for accurate projection/lookahead.
        plant: the inverse plant. Any of
            a :class:`~genesis_vehicle.control.DifferentiablePlant`;
            a ``VehicleScene`` ``Vehicle`` handle or a ``VehiclePhysics``
            (wrapped into a ``DifferentiablePlant`` for you — the no-CSV
            option; this said "default" through 1.8.0); a
            :class:`SweepTable`; or a path to a sweep CSV.
            Also accepted as the keyword ``sweep=`` for pre-v1.3.0 call sites.
        lookahead: pursuit distance along the path (m).
        arrival_goal: distance to the final waypoint that counts as DONE (m).
        cusp_goal: distance to a cusp waypoint that starts the stop-and-
            reverse transition (m).
        k_v / k_w: P gains for speed / heading.
        k_approach: cusp deceleration taper (1/s): approaching a cusp,
            |v_target| <= k_approach * remaining_distance + 0.3.
        v_stop: |v| below this counts as stopped for the cusp transition (m/s).
        steer_cap: |steer| command cap (keeps skid-steer from saturating).
        command_mode: (v2.0.2) ``"gear"`` or ``"signed"``; ``None`` (default)
            = ``"gear"`` with the MPC layer on, ``"signed"`` with
            ``mpc_model="off"``. ``"signed"`` returns the 2.0.1
            ``(throttle, steer, brake)`` 3-tuple: the same commands as 2.0.1
            for the same inputs. ``"gear"`` returns a :class:`GearCommand`
            (throttle in [0, 1], brake, gear = the active block's sign) for a
            vehicle driven with ``set_inputs(..., gear=)``. The COMMAND never
            asks for reverse outside a reverse block already in the path and
            never decelerates with a throttle against the commanded gear; the
            VEHICLE behaves the same with ``gear=`` feedback, or absent an
            external change of its engaged gear — without feedback, after an
            external reset (or a follower constructed on a vehicle rolling the
            wanted way in the other engaged gear) the vehicle can be driven in
            its engaged gear until ``GUARD`` (see :meth:`resync_gear`). The
            brake level comes from a brake -> deceleration model in STAGE 2 only
            (:mod:`genesis_vehicle.control.brake`; the MPC prediction keeps its
            throttle-based deceleration bound). Gear mode refuses a skid-steer
            ``DifferentiablePlant`` source (``ValueError``; a ``SweepTable``
            cannot tell — the vehicle refuses the gear on its first step) and
            needs a plant with ``accel_at`` (``TypeError`` otherwise). See
            :meth:`plan` for ``SHIFT`` / ``GUARD`` / ``GEAR-ABORT``.
        brake_gain: (v2.0.2, gear mode only) k_b, m/s^2 of deceleration per
            unit brake. ``None``: ``plant.brake_gain()`` (DifferentiablePlant,
            closed form), else the params record's ``brake_gain`` key, else
            derived from its ``decel_brake_by_speed``; none of these ->
            ``ValueError``.

    MPC layer (v2.0.0; every argument below is MPC-only and raises
    ``ValueError`` when given together with ``mpc_model="off"``):

        mpc_params: path to this vehicle's params JSON (written by
            ``python -m genesis_vehicle.control.identify``) or an
            :class:`~genesis_vehicle.control.mpc.MPCVehicleParams`. REQUIRED
            unless ``mpc_model="off"``.
        mpc_model: ``"bicycle"`` (default; the field AlpaSim bicycle model) or
            ``"off"`` (no MPC layer: the 1.x PD follower). Since 2.0.1, for
            the same sequence of inputs, identical to 1.8.0 on every step
            before the first ``GOAL-MISSED`` step, except at steps where 1.x
            planned a speed against the block direction (W2; none in the PD
            golden). In closed loop a W2 step changes the run from that step
            on. ``tests/test_pd_golden.py`` enforces it on its 38 scenarios
            (34 on every step, 4 up to their ``GOAL-MISSED`` step). (Through
            2.0.0 this said "bit-identical to 1.8.0", true then.)
        mpc_impl: ``"nonlinear"`` (default; do_mpc + IPOPT) or ``"linear"``
            (OSQP). The nonlinear solve is NOT guaranteed to fit a 25 ms
            control period: on the measured machine a share of the control
            steps overran it, and that share varied between measurement
            sessions. Measure your machine with
            ``python -m genesis_vehicle.samples.bench_path_follower`` and see
            the docs (``docs/path-following.md``) for the recorded sessions
            and their conditions. For a loop that must fit the period, use
            ``"linear"`` (measure it too).
        on_solve_failure: ``"pd"`` (default) — a solve that is not applicable
            hands the step to the PD law until the next applicable solve —
            or ``"hold"``, the parity mode with the field controller. **Since
            2.0.1 ``"hold"`` is field parity only with
            ``mpc_align_gate=None, mpc_stall_timeout=None`` and only on runs
            with no ``GOAL-MISSED`` and no W2 step** (the gate and watchdog
            are on by default under either policy; ``GOAL-MISSED`` and W2
            cannot be switched off). Under ``"pd"`` the PD drives after a
            failed solve only up to a travel budget per forward block, reset
            by :meth:`set_path` (F8: at most pi * R_min plus the travel of one
            :meth:`plan` interval; then ``MPC-ABORT``,
            ``stop_reason == "fail_budget"``), also with both new kwargs
            ``None``. The budget integrates the distance between successive
            positions passed to :meth:`plan`, so position noise alone can fill
            it — but only with ``mpc_stall_timeout=None``: with the watchdog on
            (default), a vehicle blocked in R-FAIL enters R-STALL after
            ``mpc_stall_timeout`` and stays there without a time limit
            (measured with a fake always-raising solver: R-STALL at 2.0 s,
            still there at 100 s). With ``mpc_stall_timeout=None`` a blocked
            vehicle stays in R-FAIL without a time limit if its pose is
            noise-free. What
            ``"hold"`` applies depends on ``mpc_impl``: with ``"nonlinear"``
            it APPLIES INFEASIBLE SOLUTIONS — every solve that did not raise,
            including an infeasible or iteration-capped one and the stale
            finite ``u`` IPOPT returns with ``Invalid_Number_Detected`` — and
            KEEPS THE PREVIOUS COMMAND on an exception or a non-finite ``u``
            (the field applied a non-finite one); with ``"linear"`` it applies
            only ``OSQP_SOLVED`` / ``OSQP_SOLVED_INACCURATE`` and keeps the
            previous command otherwise. Under ``"pd"`` a capped solve is
            applied when its iterate is feasible — feasible, not optimal.
            The classification is in :mod:`genesis_vehicle.control.mpc.law`.
        mpc_gains: :class:`~genesis_vehicle.control.mpc.MPCGains` (AlpaSim
            defaults if None).
        mpc_v_floor: speed floor for timing the MPC reference (m/s, default
            0.5).
        dt_ctrl: control period (s); default the params file's ``dt``.
        mpc_align_gate: (v2.0.1) ``(theta_on, theta_off)`` in rad, default
            ``(pi/2, pi/4)``, or ``None``. The heading gate, evaluated on
            every forward non-command step outside R-STALL (in R-STALL only
            at the 1 m exit): when the heading
            error e_psi = wrap(per-waypoint path yaw at the projection −
            chassis yaw) exceeds ``theta_on`` the PD law drives
            (``last_mode == "PD-ALIGN"``, regime ``"R-ALIGN"``, no solve);
            once |e_psi| < ``theta_off`` the MPC reference is re-timed from
            the current projection (``u_prev`` reset as in :meth:`set_path`)
            and the MPC drives again from that step. Needs
            ``0 < theta_off < theta_on <= pi``. It also takes over starts
            the 2.0.0 MPC drove away from: on the reference car (ray-wheel,
            NMPC, ``samples/mpc_follow_demo.py``) a start at |e_psi| = 1.67 rad
            reached DONE with the 2.0.0 behaviour and a start at 2.36 rad stood
            still; a Genesis-free sweep of the first MPC command on one stall
            path gave a positive acceleration up to 2.2 rad. It also takes
            over mid-path corners sharp enough to put |e_psi| above
            ``theta_on`` — such runs differ from 2.0.0 and from the field
            controller. Bounds (v2.0.1, chosen values): R-ALIGN travel is
            budgeted per forward block, reset by :meth:`set_path`, at most
            pi * R_min plus the travel of one :meth:`plan` interval (R_min =
            wheelbase / tan(steer_cap * max_steer_rad)); beyond it
            ``MPC-ABORT`` (``"align_budget"``). A final block too short for
            the turn is not refused: the R-ALIGN entry is handled like any
            other (F3 budget). Measured on the reference car, C-v150 with a
            5 m second leg (ray-wheel, NMPC, 60 s, final 2.0.1 tree, from the
            per-step log): 43.9 m of travel, of which 23.3 m in PD-ALIGN and
            1.5 m braking, CTE 23.95 m, then ``MPC-ABORT`` /
            ``"align_budget"``. The budgets
            integrate the distance between successive positions passed to
            :meth:`plan`, so position noise alone fills them: a blocked
            vehicle with noisy pose eventually aborts (``align_budget``; one
            measurement: pose noise 5 mm / 0.002 rad at 40 Hz,
            ``align_budget`` after 65.95 s); with a noise-free pose it stays
            in R-ALIGN without a time limit (the watchdog does not run in
            R-ALIGN). Distance driven under the MPC itself (R-MPC) has no
            budget.
        mpc_stall_timeout: (v2.0.1) seconds, default ``2.0``, or ``None``. The
            no-progress watchdog: while the MPC drives a forward block
            (regimes ``R-MPC`` / ``R-FAIL``), if the chassis position passed
            to :meth:`plan` stays within 0.25 m of an anchor for
            ``mpc_stall_timeout`` of follower time ``t`` while farther than
            ``arrival_goal`` from the last waypoint, the PD law drives from
            that step (``"PD-STALL"``, regime ``"R-STALL"``) until the
            vehicle has moved 1.0 m; then the reference is re-timed and the
            MPC (or the gate) takes over. A stop that creeps 0.25 m or more
            every ``mpc_stall_timeout`` is NOT caught. A vehicle held still
            by the user (own brake / e-stop) while the follower keeps being
            stepped also hands over; on release the PD law (its low-speed
            KICK saturates the throttle) drives 1 m. User-held stops long
            enough to fire the watchdog, the 2nd to 4th of a run each less
            than 3.0 m of new path progress beyond the earlier ones, end the
            run with ``"stall_budget"``. Time
            is the ``t`` passed to :meth:`step` (or the internal clock), in
            the caller's units; a forward jump of ``t`` by more than half the
            timeout between two calls re-anchors on that call, so the jump
            itself does not fire the watchdog (a fire can still follow
            ``mpc_stall_timeout`` later). An R-STALL entry aborts (``MPC-ABORT``,
            ``"stall_budget"``) when it is the fourth in a run. A run starts
            at the block's first entry, or at an entry whose projection lies
            at least 3.0 m (arc length along the block) beyond the furthest
            projection of every earlier entry in this block. Progress is
            measured along the PATH PROJECTION, not on the ground: on the
            inside of a path curve of radius R, at offset d from the path
            toward the centre, 1 m of ground motion is about R/(R − d) m of
            path progress, so a vehicle that stops repeatedly on the inside
            of a tight curve can start a new run at every entry without
            getting anywhere (constructed: hairpin of radius 7.4 m, vehicle
            1.9 m from the centre, entries 1.03 m apart on the ground: 6
            entries, 6 new runs, no abort). The entries in one forward block
            are bounded by the block length: at most
            3 * (floor(L_block / 3.0 m) + 1) entries before an abort.
            Stop-and-go where every stop lies at least 3.0 m of path beyond
            every earlier stop in the block does not end on
            ``"stall_budget"``. Reset by :meth:`set_path` and on a block
            change. An R-STALL episode ends on 1.0 m of displacement from its
            entry point; its travel is not limited. R-STALL has no timer: a
            blocked vehicle that does not reach the 1.0 m displacement stays
            in R-STALL without a time limit (zero-mean pose noise of a few mm
            is far below 1.0 m).

        For these two, ``None`` DISABLES the feature — unlike
        ``mpc_v_floor`` / ``mpc_gains`` / ``dt_ctrl``, where ``None`` means
        the default. Which kwarg turns off what: ``mpc_align_gate=None`` the
        gate with its travel budget (F3); ``mpc_stall_timeout=None`` the
        watchdog with its run-of-entries limit including the progress rule
        and the ``t``-jump rule (F5 / F6); never off: F8 (``"pd"`` only),
        ``GOAL-MISSED``, W2. So ``mpc_align_gate=None, mpc_stall_timeout=None`` gives the
        2.0.0 commands only on runs with no ``GOAL-MISSED``, no W2 step and,
        under ``"pd"``, no ``fail_budget`` abort — no kwarg combination
        restores 2.0.0's ``"pd"`` exactly (``mpc_stats`` also gains the
        ``handover`` key). MPC followers only: with ``mpc_model="off"``
        passing either, even as ``None``, raises like every MPC argument.
        Gate, watchdog, F3 / F5 / F8 also apply to ``mpc_impl="linear"``.
        The 2.0.1 ray-wheel acceptance courses were all measured on the
        nonlinear MPC; the linear MPC on two of them only (the misaligned
        stall course: ``MPC-ABORT`` / ``"fail_budget"``; a 150 degree vertex
        with a 20 m second leg: ``GOAL-MISSED``) and, for the F5 progress
        rule, in a kinematic search (40 random paths: 12 DONE, one run lost
        to F8). The linear MPC was not measured on the ray-wheel short-leg
        courses. While PD drives under either, the
        law is not called: ``mpc_stats["plan_age_s"]`` keeps growing and
        :attr:`last_plan` is the last pre-handover plan.

    The MPC layer runs on FORWARD blocks only and models a 2-axle Ackermann
    vehicle; reverse blocks are driven by the PD law. It needs the chassis yaw
    rate and lateral speed: pass ``yaw_rate=`` / ``v_lat=`` to :meth:`step`
    (:func:`extract_state` returns both). Not thread-safe (casadi's NumPy mode
    and OpenBLAS thread count are process-global).

    ``last_mode`` after each :meth:`step` is one of ``INIT`` / ``DRV+1`` /
    ``DRV-1`` / ``STOP`` / ``BRAKE_TRANS`` (stopping for a cusp) / ``DONE``,
    and with the MPC layer on also ``MPC`` / ``MPC-HOLD`` (standing, brake
    instead of a negative throttle) / ``MPC-HOLDCMD`` (``"hold"``: previous
    command kept after a failed solve) / ``PD-FALLBACK`` (``"pd"``: PD after
    a failed solve) / ``PD-ALIGN`` (heading gate) / ``PD-STALL`` (watchdog),
    and in gear mode (v2.0.2) the non-terminal ``SHIFT`` / ``GUARD`` and the
    terminal ``GEAR-ABORT`` (see :meth:`plan`),
    and since v2.0.1 the terminal ``GOAL-MISSED`` (every mode: on the final
    block, past the goal's normal plane on the last segment, at least
    ``arrival_goal + 0.1`` m from the goal, and the distance grew by more than
    1e-3 m on each of the last two :meth:`plan` calls — so a vehicle moving
    away slower than 1e-3 m per call, i.e. 1e-3/h m/s at call interval h, is
    not stopped, and calling :meth:`plan` more than once per control step
    shortens the window) / ``MPC-ABORT`` (MPC layer budgets, see
    :attr:`stop_reason`). Both command ``(0, 0, 1)`` and stay set until
    :meth:`set_path` or a ``DONE`` step. A ``DONE`` (inside ``arrival_goal``
    of the last waypoint) sets :attr:`stop_reason` to None, and the follower
    drives again if the vehicle leaves the circle; the block's MPC budgets
    stay as spent — only :meth:`set_path` resets them. A ``DONE`` can clear
    only an abort on the final block: an abort on an earlier block is sticky
    until :meth:`set_path`. See :attr:`TERMINAL_MODES`. Neither is a bound on
    travel, and ``"off"`` and R-REV have no MPC budget: the follower is NOT
    guaranteed not to run away. Observed in a kinematic test of the 1.x PD
    with these stops (``tests/test_pd_runaway_property.py``, seed 21, 120 s),
    runs that neither finished nor stopped: pinned mid-path and receding;
    on the near side of the last segment and receding; on a non-final
    forward block; and on the final block beyond the goal plane either
    approaching slowly (-1.8 m over the last 10 s) or moving at 2 m/s at a
    nearly constant distance from the goal (+0.4 m over the last 10 s), so
    the distance does not grow on two consecutive calls and GP-AWAY2 does
    not fire. Runs pinned near the path start and run-aways on reverse blocks
    were observed in the planner's emulation of these rules on other seeds,
    not on the shipped code.
    """

    #: Release-specific accepted values of ``mpc_model``.
    MPC_MODELS = ("bicycle", "off")
    #: (v2.0.1) ``last_mode`` values that end a run: ``DONE`` (inside
    #: ``arrival_goal`` of the last waypoint; recomputed every step, as in
    #: 1.x), ``GOAL-MISSED`` and ``MPC-ABORT`` (both stay set until
    #: :meth:`set_path` or a ``DONE`` step; a ``DONE`` can clear only an abort
    #: on the final block, sets ``stop_reason`` to None and resets no MPC
    #: budget). Loop on
    #: ``last_mode in PathFollower.TERMINAL_MODES`` and read :attr:`stop_reason`.
    #: (v2.0.2) plus ``GEAR-ABORT`` (gear mode: more than ``SHIFT_MAX_CALLS``
    #: consecutive ``SHIFT`` / ``GUARD`` calls, ``stop_reason ==
    #: "shift_timeout"``; sticky like ``MPC-ABORT``).
    TERMINAL_MODES = ("DONE", "GOAL-MISSED", "MPC-ABORT", "GEAR-ABORT")

    def __init__(self, path, plant=None, *, sweep=None,
                 mpc_params=_Unset(None), mpc_model: str = "bicycle",
                 mpc_impl=_Unset("nonlinear"), on_solve_failure=_Unset("pd"),
                 mpc_gains=_Unset(None), mpc_v_floor=_Unset(0.5),
                 dt_ctrl=_Unset(None),
                 mpc_align_gate=_Unset((math.pi / 2, math.pi / 4)),
                 mpc_stall_timeout=_Unset(2.0),
                 lookahead: float = 3.5, arrival_goal: float = 1.5,
                 cusp_goal: float = 1.0, k_v: float = 2.0, k_w: float = 1.5,
                 k_approach: float = 1.0, v_stop: float = 0.05,
                 steer_cap: float = 0.5, plant_kwargs=None,
                 command_mode: Optional[str] = None, brake_gain: Optional[float] = None):
        # (1) path length, (2) plant present — the 1.x checks, unchanged.
        if len(path) < 2:
            raise ValueError("path needs at least 2 waypoints")
        if plant is None:
            plant = sweep
        if plant is None:
            raise ValueError(
                "PathFollower needs an inverse plant: pass the vehicle (or a "
                "DifferentiablePlant) for the autodiff plant, or a SweepTable / "
                "sweep CSV path for the measured one")
        mpc_kw = {"mpc_params": mpc_params, "mpc_impl": mpc_impl,
                  "on_solve_failure": on_solve_failure, "mpc_gains": mpc_gains,
                  "mpc_v_floor": mpc_v_floor, "dt_ctrl": dt_ctrl,
                  "mpc_align_gate": mpc_align_gate, "mpc_stall_timeout": mpc_stall_timeout}
        # (3) the model name.
        if mpc_model not in self.MPC_MODELS:
            raise ValueError(f"mpc_model must be one of {list(self.MPC_MODELS)}; "
                             f"got {mpc_model!r}")
        mpc_on = mpc_model != "off"
        mpc_kw = {k: (v.value if isinstance(v, _Unset) else v) for k, v in mpc_kw.items()}
        if not mpc_on:
            # (4) no MPC-only argument with the layer off.
            given = [k for k, v in (("mpc_params", mpc_params), ("mpc_impl", mpc_impl),
                                    ("on_solve_failure", on_solve_failure),
                                    ("mpc_gains", mpc_gains), ("mpc_v_floor", mpc_v_floor),
                                    ("dt_ctrl", dt_ctrl), ("mpc_align_gate", mpc_align_gate),
                                    ("mpc_stall_timeout", mpc_stall_timeout))
                     if not isinstance(v, _Unset)]
            if given:
                raise ValueError(f"mpc_model=\"off\" takes no MPC argument; got {', '.join(given)}")
            mpc_setup = None
        else:
            mpc_setup = self._check_mpc_args(plant, mpc_model, **mpc_kw)
        # (v2.0.2) command mode and the brake model's gain — every check before
        # this object is touched (the plant is coerced into a local first).
        plant_obj = _as_plant(plant, plant_kwargs or {})
        command_mode, k_b = self._check_command_mode(
            plant_obj, mpc_on, command_mode, brake_gain,
            None if mpc_setup is None else mpc_setup["params"])

        self.mpc_model = mpc_model
        self.command_mode = command_mode
        self._gear_mode = command_mode == "gear"
        self._k_b = k_b
        self.plant = plant_obj
        #: Back-compat alias — ``.sweep`` was the pre-v1.3.0 attribute name.
        self.sweep = self.plant
        self._coupled_plant = bool(getattr(self.plant, "has_coupled_solve", False))
        # A Jacobian plant linearises about the command it last handed out, so
        # it has to work inside the same steer envelope this follower enforces
        # — otherwise it can sit at steer=-1.0 while the vehicle is being given
        # -steer_cap, and every Jacobian after that describes a vehicle state
        # that never existed. `set_applied` at the end of step() closes the
        # same loop for the throttle, which the KICK can override outright.
        if hasattr(self.plant, "steer_range"):
            self.plant.steer_range = (-abs(steer_cap), abs(steer_cap))
        (self.path, self.wp_yaws, self.n, self.blocks, self.yaws,
         self._block_end_yaw) = self._parse_path(path)
        self.lookahead = lookahead
        self.arrival_goal = arrival_goal
        self.cusp_goal = cusp_goal
        self.k_v = k_v
        self.k_w = k_w
        self.k_approach = k_approach
        self.v_stop = v_stop
        self.steer_cap = steer_cap

        #: Which member of a fleet plant this follower drives; None when the
        #: plant serves it alone. Set by :class:`FleetFollower`.
        self.plant_member = None

        self.block_i = 0
        blk_start = self.blocks[0][0]
        self.current_idx = blk_start
        self.projection_t = 0.0
        self.projection_xy = (self.path[blk_start][0], self.path[blk_start][1])
        self.transitioning = False
        self.last_mode = "INIT"
        self.regime = None
        self._law = None
        # v2.0.1 terminal state and GP-AWAY2 history (every mode, "off" too).
        self._terminal = None           # None | (mode, stop_reason); DONE is not stored
        self._gp_hist = []              # distance to the final waypoint, last 3 calls
        # v2.0.2 gear mode (plan_r202 section 3.3): the gear the follower
        # believes engaged, whether the construction-only motion rule has run,
        # consecutive slow SHIFT calls without feedback, consecutive
        # SHIFT/GUARD calls (GEAR-ABORT), the GUARD latch and its slow calls,
        # and this call's feedback / lateral speed.
        self._gear_assumed = None
        self._gear_init_done = False
        self._shift_ok = 0
        self._shift_n = 0
        self._guard = False
        self._guard_slow = 0
        self._fb_gear = None
        self._fb_vlat = None
        if mpc_on:
            self._init_mpc(mpc_setup, steer_cap)

    # -- gear mode: construction (v2.0.2) -------------------------------------------

    @staticmethod
    def _check_command_mode(plant, mpc_on, command_mode, brake_gain, params):
        """``(command_mode, k_b)``; raises before any side effect. Default
        mode: ``"gear"`` with the MPC layer on, ``"signed"`` with
        ``mpc_model="off"``. In gear mode: a skid-steer ``DifferentiablePlant``
        source refuses (``ValueError``); the plant needs a callable
        ``accel_at`` (``TypeError``); ``k_b`` comes from ``brake_gain=``, else
        a callable ``plant.brake_gain()`` (a non-callable attribute of that name
        is a ``TypeError``), else the params record
        (``control.brake.brake_gain_from_params``), else ``ValueError``."""
        if command_mode is None:
            command_mode = "gear" if mpc_on else "signed"
        if command_mode not in COMMAND_MODES:
            raise ValueError(f"command_mode must be one of {list(COMMAND_MODES)} or None; "
                             f"got {command_mode!r}")
        if brake_gain is not None:
            if command_mode != "gear":
                raise ValueError("brake_gain= is a gear-mode argument (command_mode=\"gear\")")
            if not _is_real(brake_gain) or not (math.isfinite(float(brake_gain))
                                                and float(brake_gain) > 0.0):
                raise ValueError(f"brake_gain must be a finite number > 0 (m/s^2 per unit "
                                 f"brake); got {brake_gain!r}")
        if command_mode == "signed":
            return command_mode, None
        src = getattr(plant, "src", None)
        resolved = getattr(src, "resolved", None) if src is not None else None
        if resolved is not None and refuses_gear(resolved):
            raise ValueError(
                "command_mode=\"gear\": this plant's vehicle is skid-steer (PerSide "
                "drivetrain or SkidSteer steering), which refuses gear mode — pass "
                "command_mode=\"signed\"")
        if not callable(getattr(plant, "accel_at", None)):
            raise TypeError(
                "command_mode=\"gear\" needs a plant with a forward accel_at(v, throttle, "
                f"pitch_deg, roll_deg) for the brake model; {type(plant).__name__} has "
                "none (SweepTable and DifferentiablePlant have it) — or pass "
                "command_mode=\"signed\"")
        if brake_gain is not None:
            return command_mode, float(brake_gain)
        bg = getattr(plant, "brake_gain", None)
        if bg is not None and not callable(bg):
            raise TypeError(
                f"plant.brake_gain must be a method returning the brake gain; got a "
                f"{type(bg).__name__} attribute — pass brake_gain= to PathFollower instead")
        if bg is not None:
            k_b = float(bg())
            if not (math.isfinite(k_b) and k_b > 0.0):
                raise ValueError(f"plant.brake_gain() must be a finite number > 0; got {k_b!r}")
            return command_mode, k_b
        from .brake import brake_gain_from_params
        k_b = brake_gain_from_params(params, plant.accel_at)
        if k_b is None:
            raise ValueError(
                "command_mode=\"gear\" needs the brake gain k_b (m/s^2 per unit brake): "
                "pass brake_gain=, a plant with brake_gain() (DifferentiablePlant), or "
                "mpc_params (its brake_gain key, or decel_brake_by_speed)")
        return command_mode, k_b

    # -- MPC layer: construction ------------------------------------------------

    def _check_mpc_args(self, plant, mpc_model, *, mpc_params, mpc_impl,
                        on_solve_failure, mpc_gains, mpc_v_floor, dt_ctrl,
                        mpc_align_gate, mpc_stall_timeout):
        """Steps 5-10 of the constructor's check order (each raises before any
        side effect): live-vehicle steering, params present, implementation /
        policy / heading gate / stall timeout (v2.0.1: after U1, before the
        dependencies), dependencies, params file, forward blocks."""
        # (5) a live vehicle can be checked for what the model describes.
        steering, n_axles = _live_steering(plant)
        if steering is not None and not _is_two_axle_ackermann(steering, n_axles):
            raise ValueError(
                "the MPC layer models a 2-axle Ackermann vehicle; this vehicle has "
                f"{type(steering).__name__}/{n_axles} axles — pass mpc_model=\"off\"")
        # (6) U1: no params, no MPC layer.
        if mpc_params is None:
            raise ValueError(U1_MESSAGE)
        # (7) implementation, policy.
        if mpc_impl not in ("nonlinear", "linear"):
            raise ValueError(f"mpc_impl must be 'nonlinear' or 'linear'; got {mpc_impl!r}")
        if on_solve_failure not in ("pd", "hold"):
            raise ValueError(f"on_solve_failure must be 'pd' or 'hold'; got {on_solve_failure!r}")
        gate = _check_align_gate(mpc_align_gate)
        stall = _check_stall_timeout(mpc_stall_timeout)
        # (8) dependencies.
        from .mpc._deps import require_mpc_deps
        require_mpc_deps(mpc_impl)
        # (9) the params record (missing file, schema, keys, JSON steering).
        from .mpc.params import load_mpc_params
        params = load_mpc_params(mpc_params)
        return dict(params=params, impl=mpc_impl, policy=on_solve_failure,
                    gains=mpc_gains, v_floor=0.5 if mpc_v_floor is None else float(mpc_v_floor),
                    dt_ctrl=dt_ctrl, model=mpc_model, align_gate=gate, stall_timeout=stall)

    def _init_mpc(self, st, steer_cap):
        import warnings
        params = st["params"]
        dt_ctrl = float(params.dt) if st["dt_ctrl"] is None else float(st["dt_ctrl"])
        if st["dt_ctrl"] is not None and abs(dt_ctrl - float(params.dt)) > 1e-12:
            warnings.warn(
                f"dt_ctrl={dt_ctrl} differs from the params file's dt={params.dt}: the "
                "identified time constants and limits belong to that dt", UserWarning,
                stacklevel=3)
        # (10) a path the MPC layer never drives.
        if not any(sgn > 0 for (_, _, sgn) in self.blocks):
            warnings.warn("the MPC layer never runs on this path (no forward block); "
                          "every block is driven by the PD law", UserWarning, stacklevel=3)
        # (11) a live plant runs the inversion in the control loop.
        if self._coupled_plant:
            warnings.warn(
                "MPC layer + a live-vehicle plant (DifferentiablePlant): the MPC solve "
                "and the plant inversion both run inside every control step; a sweep "
                "table keeps the step cheaper (see the docs for measured times)",
                UserWarning, stacklevel=3)
            _cross_check_live(self.plant, params)        # (12)
        sp = params.solver_params(st["model"], steer_cap=abs(steer_cap), dt_ctrl=dt_ctrl)
        # (v2.0.1) minimum turn radius from the solver's own steering limit:
        # the scale of the PD budgets (F3 / F8: pi * R_min of travel per block).
        # The check stays: an infinite R_min would silently disable F3 / F8.
        wb, smax = float(sp.wheelbase), float(sp.steer_max)
        if not (math.isfinite(wb) and wb > 0.0 and math.isfinite(smax)
                and 0.0 < smax < math.pi / 2):
            raise ValueError(
                "the MPC layer needs wheelbase > 0 and a steering limit "
                "steer_cap * max_steer_rad in (0, pi/2) rad; got wheelbase="
                f"{wb!r}, steering limit {smax!r}")
        r_min = wb / math.tan(smax)
        from .mpc.law import _MPCLaw
        if st["impl"] == "nonlinear":
            from .mpc.alpasim_nonlinear import NonlinearMPC
            solver = NonlinearMPC(sp, st["gains"], yaw_bound=math.pi / 2)
        else:
            from .mpc.alpasim_linear import LinearMPC
            solver = LinearMPC(sp, st["gains"])
        self.mpc_params = params
        self.mpc_impl = st["impl"]
        self.on_solve_failure = st["policy"]
        self.dt_ctrl = dt_ctrl
        self._law = _MPCLaw(solver, rig_offset=float(params.rig_offset),
                            on_solve_failure=st["policy"], v_floor=st["v_floor"],
                            impl=st["impl"])
        self._mpc_block = None          # block index the law's reference belongs to
        self._t_next = 0.0              # internal clock (when step() gets no t)
        self._t_now = None
        # v2.0.1 heading gate / no-progress watchdog (plan_r210 section 3.3-3.4).
        self._align_gate = st["align_gate"]        # (theta_on, theta_off) or None
        self._stall_timeout = st["stall_timeout"]  # s or None
        self._handover = {"align": 0, "stall": 0}  # cumulative, survives set_path
        self._ho_regime = None          # None | "R-ALIGN" | "R-STALL"
        self._ho_block = None           # block the handover state belongs to
        self._anchor = None             # (t, x, y) of the watchdog, or None
        self._stall_entry = None        # (x, y) where R-STALL began, or None
        # v2.0.1 per-block PD budgets (plan_r210 section 3B.3).
        self._r_min = r_min
        self._pd_budget = math.pi * r_min   # F3 (R-ALIGN) and F8 (R-FAIL) travel, m
        self._reset_budgets()

    def _reset_budgets(self) -> None:
        self._bud_block = None          # block the budgets below belong to
        self._d_align = 0.0             # travel after R-ALIGN steps in that block
        self._d_fail = 0.0              # travel after R-FAIL steps in that block
        self._n_stall = 0               # watchdog fires in the current run of that block
        self._stall_s_hw = None         # furthest projection arc length over the block's fires
        self._prev_regime = None        # regime of the previous plan() call
        self._prev_xy = None            # its position

    # -- Path geometry ---------------------------------------------------------

    def _parse_path(self, path):
        """Waypoints -> (path, wp_yaws, n, blocks, yaws, block_end_yaw), the
        1.x path model (unchanged): normalised waypoints, direction blocks,
        per-waypoint chassis yaw. Pure: raises before touching ``self``."""
        # Normalize waypoints to (x, y, z, speed) + optional explicit yaw.
        path_ = []
        wp_yaws = []            # per-waypoint explicit yaw or None
        for p in path:
            p = tuple(p)
            if len(p) == 4:
                wy = None
            elif len(p) == 5:
                wy = p[4]
                if wy is not None:
                    wy = (float(wy) + math.pi) % (2 * math.pi) - math.pi
            else:
                raise ValueError(
                    "waypoints must be (x, y, z, target_speed) or "
                    f"(x, y, z, target_speed, yaw); got length {len(p)}")
            path_.append((float(p[0]), float(p[1]), float(p[2]), float(p[3])))
            wp_yaws.append(wy)
        n = len(path_)
        # Direction blocks: maximal index runs [start, end] with one speed
        # sign. Zero-speed waypoints inherit the running sign (an explicit
        # stop point does not split a block).
        blocks = []          # list of (start_idx, end_idx, sign)
        sign = 0
        start = 0
        for i in range(n):
            s = path_[i][3]
            s_sign = (s > 1e-6) - (s < -1e-6)
            if s_sign == 0:
                continue
            if sign == 0:
                sign = s_sign
            elif s_sign != sign:
                blocks.append((start, i, sign))   # cusp AT waypoint i
                start, sign = i, s_sign
        blocks.append((start, n - 1, sign if sign != 0 else 1))

        # Per-waypoint chassis yaw, computed PER BLOCK: toward the next
        # waypoint, with a +pi flip in backward blocks (the chassis faces
        # away from the travel direction). A block-end waypoint is shared
        # with the next block, so each block also records its own
        # continue-straight end yaw (`_block_end_yaw`) — using the global
        # per-waypoint value there would leak the NEXT block's (flipped)
        # direction into the current block's pursuit. Zero-length segments
        # (duplicated cusp waypoints) carry the previous yaw.
        # (The pre-SDK deliverable's backward flip `(yaw + pi) % (2*pi) - pi`
        # was a plain normalization — a no-op — so backward chassis yaw was
        # off by pi; masked on collinear reversing where the position-error
        # term vanishes.)
        # An EXPLICIT per-waypoint yaw (5th tuple element) wins verbatim —
        # no backward flip, it already IS the desired chassis heading.
        yaws = [0.0] * n
        block_end_yaw = []
        for (bs, be, sgn) in blocks:
            prev = None
            for i in range(bs, be):
                if wp_yaws[i] is not None:
                    yaw = wp_yaws[i]
                else:
                    dx = path_[i + 1][0] - path_[i][0]
                    dy = path_[i + 1][1] - path_[i][1]
                    if dx * dx + dy * dy < 1e-12:
                        yaw = prev if prev is not None else 0.0
                    else:
                        yaw = math.atan2(dy, dx)
                        if sgn < 0:
                            yaw = (yaw % (2 * math.pi)) - math.pi   # +pi flip
                yaws[i] = yaw
                prev = yaw
            # Arrival heading at the block end: an explicit yaw on the
            # boundary waypoint wins; otherwise continue the last in-block
            # direction — but NOT a trailing segment that runs backwards.
            #
            # The boundary waypoint belongs to both blocks, and nothing forces
            # it to lie ahead of the previous one: a path that doubles back can
            # place the next block's first waypoint *behind* this block's last
            # (a real one hops 0.28 m backwards at the cusp). Taking that hop as
            # the arrival heading asks the vehicle to spin 180 degrees at the
            # end of a straight leg, which it will dutifully try to do — and
            # then the projection can no longer advance, so the cusp never
            # triggers and the vehicle drives off forever. Walk back to the last
            # segment that still agrees with the block's net travel. (v1.2.1)
            if wp_yaws[be] is not None:
                block_end_yaw.append(wp_yaws[be])
            else:
                block_end_yaw.append(
                    self._forward_end_yaw(bs, be, sgn, prev, path_))
        yaws[n - 1] = block_end_yaw[-1]

        return path_, wp_yaws, n, blocks, yaws, block_end_yaw

    def _forward_end_yaw(self, bs, be, sgn, fallback, path=None):
        """Arrival heading for block ``[bs, be]``: the heading of the last
        segment that does not run counter to the block's net travel.

        Segments are scanned from the block end backwards; one whose direction
        opposes the straight line from ``bs`` to ``be`` by more than 90 degrees
        is a path artifact (a boundary waypoint sitting behind its predecessor)
        and is skipped. Backward blocks get the usual +pi chassis flip. Falls
        back to ``fallback`` when the whole block is degenerate.
        """
        path = self.path if path is None else path
        nx = path[be][0] - path[bs][0]
        ny = path[be][1] - path[bs][1]
        have_net = (nx * nx + ny * ny) > 1e-12

        for i in range(be - 1, bs - 1, -1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            if dx * dx + dy * dy < 1e-12:
                continue
            if have_net and (dx * nx + dy * ny) < 0.0:
                continue                      # runs backwards — artifact
            yaw = math.atan2(dy, dx)
            if sgn < 0:
                yaw = (yaw % (2 * math.pi)) - math.pi
            return yaw
        return fallback if fallback is not None else 0.0

    def _advance(self, pos):
        """Project the position onto the nearest segment OF THE ACTIVE BLOCK
        and advance ``current_idx``/``projection_t``.

        Only segments within an arc-length window (lookahead + 2 m) ahead
        are searched, and the index may advance at most 5 segments per call
        — both guard against snapping to a far-away later segment when the
        path self-approaches.
        """
        blk_end = self.blocks[self.block_i][1]
        if self.current_idx >= blk_end:
            return
        search_arc = self.lookahead + 2.0
        accum = 0.0
        end = self.current_idx + 1
        while end <= blk_end and accum < search_arc:
            dx = self.path[end][0] - self.path[end - 1][0]
            dy = self.path[end][1] - self.path[end - 1][1]
            accum += math.hypot(dx, dy)
            end += 1

        best_idx = self.current_idx
        best_d = float("inf")
        best_t = self.projection_t
        for i in range(self.current_idx, min(end, blk_end)):
            p0 = self.path[i]; p1 = self.path[i + 1]
            sx = p1[0] - p0[0]; sy = p1[1] - p0[1]
            seg_len_sq = sx * sx + sy * sy
            if seg_len_sq < 1e-9:
                continue
            tx = pos[0] - p0[0]; ty = pos[1] - p0[1]
            t_c = max(0.0, min(1.0, (sx * tx + sy * ty) / seg_len_sq))
            cx = p0[0] + t_c * sx; cy = p0[1] + t_c * sy
            d = (pos[0] - cx) ** 2 + (pos[1] - cy) ** 2
            if d < best_d:
                best_d = d; best_idx = i; best_t = t_c

        MAX_STEP = 5
        target_idx = min(best_idx, self.current_idx + MAX_STEP)
        if target_idx > self.current_idx:
            self.current_idx = target_idx
            if self.current_idx < blk_end:
                p0 = self.path[self.current_idx]
                p1 = self.path[self.current_idx + 1]
                sx = p1[0] - p0[0]; sy = p1[1] - p0[1]
                seg_len_sq = sx * sx + sy * sy
                if seg_len_sq > 1e-9:
                    tx = pos[0] - p0[0]; ty = pos[1] - p0[1]
                    self.projection_t = max(0.0, min(1.0,
                        (sx * tx + sy * ty) / seg_len_sq))
                else:
                    self.projection_t = 0.0
            else:
                self.projection_t = 0.0
        else:
            self.projection_t = best_t

        if self.current_idx < blk_end:
            p0 = self.path[self.current_idx]
            p1 = self.path[self.current_idx + 1]
            self.projection_xy = (
                p0[0] + self.projection_t * (p1[0] - p0[0]),
                p0[1] + self.projection_t * (p1[1] - p0[1]),
            )
        else:
            p = self.path[blk_end]
            self.projection_xy = (p[0], p[1])

    def _lookahead(self):
        """Waypoint ``lookahead`` metres ahead of the projection point along
        the path, CAPPED at the active block's end. Returns
        ``(index, capped)`` — ``capped`` is True when the block end lies
        inside the lookahead window (i.e. a cusp or the goal is near)."""
        blk_end = self.blocks[self.block_i][1]
        if self.current_idx >= blk_end:
            return blk_end, True
        idx = self.current_idx
        p_next = self.path[idx + 1]
        accum = math.hypot(p_next[0] - self.projection_xy[0],
                           p_next[1] - self.projection_xy[1])
        idx += 1
        while accum < self.lookahead and idx < blk_end:
            p0 = self.path[idx]; p1 = self.path[idx + 1]
            accum += math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            idx += 1
        return idx, idx >= blk_end and accum < self.lookahead

    # -- Control ----------------------------------------------------------------

    def step(self, pos_xy, yaw, v_long, pitch_deg, roll_deg, *,
             yaw_rate=None, v_lat=None, t=None, gear=None):
        """One control step. Returns ``(throttle, steer, brake)``, each in
        the SDK input range (throttle/steer in [-1, 1], brake in [0, 1]) with
        ``command_mode="signed"``; with ``command_mode="gear"`` (v2.0.2, the
        default with the MPC layer on) a :class:`GearCommand` ``(throttle in
        [0, 1], steer, brake, gear)`` — decelerations come out as brake, never
        as a throttle against the commanded gear. The vehicle follows that with
        ``gear=`` feedback, or absent an external change of its engaged gear
        (see :meth:`resync_gear` for the case without feedback).

        ``gear`` (v2.0.2, gear mode): the vehicle's ENGAGED gear (+1 / -1),
        e.g. ``int(veh.gear[0])`` — optional feedback. With it, a call whose
        engaged gear differs from the one the follower wants is a ``SHIFT``
        (throttle 0, brake 1); without it the follower assumes the shift took
        after two slow ``SHIFT`` calls and guards against motion opposite the
        wanted gear (``GUARD``) — see :meth:`plan`. Ignored with
        ``command_mode="signed"``.

        ``yaw_rate`` (rad/s) and ``v_lat`` (m/s) are required while the MPC
        layer drives a forward block (``TypeError`` otherwise;
        :func:`extract_state` returns both) and ignored elsewhere. ``t`` (s):
        the control time; ``None`` advances an internal clock by ``dt_ctrl``
        per call (from 0 at construction / the ``set_path`` time), a given
        ``t`` also resets that clock. Ignored with ``mpc_model="off"``."""
        plan = self.plan(pos_xy, yaw, v_long, pitch_deg, roll_deg,
                         yaw_rate=yaw_rate, v_lat=v_lat, t=t, gear=gear)
        if plan.command is not None:
            return self.finish(plan, 0.0, 0.0)
        if self._coupled_plant:
            # One 2x2 inversion for both channels: a DifferentiablePlant knows
            # that steering bleeds speed and that throttle shifts the yaw
            # authority, which two independent grid lookups cannot express.
            throttle, steer = self.plant.solve(
                v_long, plan.a_target, plan.omega_target,
                plan.pitch_deg, plan.roll_deg)
        else:
            throttle = self.plant.throttle_for(
                v_long, plan.a_target, plan.pitch_deg, plan.roll_deg)
            steer = self.plant.steer_for(
                v_long, plan.omega_target, plan.pitch_deg, plan.roll_deg)
        return self.finish(plan, throttle, steer)

    def _report_applied(self, throttle, steer) -> None:
        """Tell a Jacobian plant what this vehicle was ACTUALLY commanded.

        Called on EVERY exit, not just the driving one. The cusp brake and the
        DONE stop return (0, 0, 1) without consulting the plant; leaving it
        warm-started on the last DRIVING command through a whole
        stop-and-reverse means the first command of the new direction is
        linearised about a vehicle that was still accelerating forwards —
        which spins a skid-steer instead of reversing it.
        """
        if self.plant_member is not None:
            per_member = getattr(self.plant, "set_applied_member", None)
            if per_member is not None:
                per_member(self.plant_member, throttle, steer)
                return
        report = getattr(self.plant, "set_applied", None)
        if report is not None:
            report(throttle, steer)

    def finish(self, plan, throttle, steer):
        """Turn a raw plant command into the applied ``(throttle, steer, brake)``.

        The second half of a control step: the supervisors that sit on top of
        the inversion (low-speed KICK, actuator caps, brake-at-rest) plus the
        report back to the plant. Split out of :meth:`step` so a fleet can run
        one batched inversion between everyone's :meth:`plan` and everyone's
        :meth:`finish` — see :class:`FleetFollower`.

        With ``command_mode="gear"`` (v2.0.2) it returns a :class:`GearCommand`
        from :meth:`_finish_gear` instead; the signed path below is the 2.0.1
        code, unchanged.
        """
        if self._gear_mode:
            return self._finish_gear(plan, throttle, steer)
        if plan.command is not None:
            self.last_mode = plan.mode
            self._report_applied(plan.command[0], plan.command[1])
            return plan.command

        if plan.regime == "R-MPC":
            # The field MPC finish (mpc_follower.py:175-183), same order:
            # caps, brake at rest, MPC-HOLD (decided on the MPC's own a_ff),
            # then the report to the plant. No KICK.
            from .mpc.law import finish_command
            throttle, steer, brake, mode = finish_command(
                plan.v_long, plan.a_ff, throttle, steer, self.steer_cap)
            self._report_applied(throttle, steer)
            self.last_mode = mode if mode == "MPC-HOLD" else plan.mode
            return throttle, steer, brake

        v_long, v_target = plan.v_long, plan.v_target
        # Low-speed KICK: far below target speed -> saturate throttle so the
        # vehicle actually gets moving (the P term alone can be too shy).
        # Signed comparison: v_long * sign(v_target), so moving the WRONG
        # way also kicks toward the target direction.
        if abs(v_target) > 0.1:
            v_along = v_long * math.copysign(1.0, v_target)
            if v_along < 0.4 * abs(v_target):
                throttle = math.copysign(1.0, v_target)

        throttle = max(-1.0, min(1.0, float(throttle)))
        steer = max(-self.steer_cap, min(self.steer_cap, float(steer)))
        brake = 1.0 if abs(throttle) < 0.02 and abs(v_long) < 0.1 else 0.0

        self._report_applied(throttle, steer)
        self.last_mode = plan.mode
        return throttle, steer, brake

    def _finish_gear(self, plan, throttle, steer):
        """(v2.0.2) The gear-mode finish: ``GearCommand(throttle >= 0, steer,
        brake, gear)``. ``u_g = gear * u`` is the plant's command in the
        gear's direction: ``u_g >= 0`` drives with throttle ``min(u_g, 1)``,
        ``u_g < 0`` brakes with ``control.brake.brake_for(a_target, a_coast,
        v, gear, k_b)`` at throttle 0 (``a_coast`` = ``plan.a_coast`` or
        ``control.brake.coast_for_brake(plant, v, pitch, roll, gear)``: the
        plant's ``accel_at`` at throttle 0, with a SweepTable's coast below its
        first non-zero grid speed replaced — Amendment 1). PD steps keep the low-speed
        KICK only while the vehicle is not moving against the gear by more
        than 0.05 m/s; the brake at rest (throttle < 0.02 and |v| < 0.1 m/s)
        is the 2.0.1 rule. MPC steps keep ``MPC-HOLD`` and the field finish's
        brake at rest. The plant is told ``gear * throttle`` (0 on a braking
        step)."""
        g = 1 if plan.gear is None or plan.gear >= 0 else -1
        if plan.command is not None:
            self.last_mode = plan.mode
            self._report_applied(plan.command[0], plan.command[1])
            c = plan.command
            return GearCommand(float(c[0]), float(c[1]), float(c[2]), g)
        v_long = plan.v_long

        def _brake():
            a_coast = plan.a_coast
            if a_coast is None:
                # Amendment 1: the table's coast below its first non-zero
                # speed is replaced (control.brake.coast_for_brake).
                from .brake import coast_for_brake
                a_coast = coast_for_brake(self.plant, v_long, plan.pitch_deg,
                                          plan.roll_deg, g)
            from .brake import brake_for
            return brake_for(plan.a_target, a_coast, v_long, g, self._k_b)

        if plan.regime == "R-MPC":
            from .mpc.law import finish_command
            thr, st, brk, mode = finish_command(
                plan.v_long, plan.a_ff, throttle, steer, self.steer_cap)
            if mode == "MPC-HOLD":
                thr_out, brake = 0.0, 1.0
            else:
                u_g = g * thr
                if u_g >= 0.0:
                    thr_out, brake = min(u_g, 1.0), brk
                else:
                    thr_out, brake = 0.0, max(_brake(), brk)
            self._report_applied(g * thr_out, st)
            self.last_mode = mode if mode == "MPC-HOLD" else plan.mode
            return GearCommand(float(thr_out), float(st), float(brake), g)

        v_target = plan.v_target
        u_g = g * float(throttle)
        kick = False
        if abs(v_target) > 0.1 and math.copysign(1.0, v_target) == g:
            v_along = v_long * g
            # 2.0.1 KICK condition, and (B2) not while moving against the gear.
            kick = v_along < 0.4 * abs(v_target) and v_along >= -0.05
        if kick:
            thr_out, brake = 1.0, 0.0
        elif u_g >= 0.0:
            thr_out, brake = min(u_g, 1.0), 0.0
        else:
            thr_out, brake = 0.0, _brake()
        st = max(-self.steer_cap, min(self.steer_cap, float(steer)))
        if thr_out < 0.02 and abs(v_long) < 0.1:
            brake = 1.0
        self._report_applied(g * thr_out, st)
        self.last_mode = plan.mode
        return GearCommand(float(thr_out), float(st), float(brake), g)

    def plan(self, pos_xy, yaw, v_long, pitch_deg, roll_deg, *,
             yaw_rate=None, v_lat=None, t=None, gear=None):
        """The vehicle-agnostic half of a control step (stage 1).

        Runs the path state machine (projection, blocks, cusp, DONE) every
        step, and returns a :class:`FollowerPlan`. With ``mpc_model="off"``
        that is the 1.x PD plan plus the v2.0.1 goal-passed stop and W2 (see
        the class docstring; through 2.0.0 this said "unchanged"). With the
        MPC layer on, a forward
        block is planned by the MPC (``regime="R-MPC"``); a reverse block
        (``"R-REV"``) and a step after a failed solve under
        ``on_solve_failure="pd"`` (``"R-FAIL"``) get the PD plan, and so do
        (v2.0.1) forward steps held by the heading gate (``"R-ALIGN"``) or
        handed over by the no-progress watchdog (``"R-STALL"``) — both decided
        here from this method's inputs, so a caller that runs its own finish
        gets them too. A terminal step (``DONE``, ``GOAL-MISSED``,
        ``MPC-ABORT``) carries ``command=(0, 0, 1)``.

        Gear mode (v2.0.2, ``command_mode="gear"``), decided here too — in
        this order after the DONE / terminal / cusp checks: ``SHIFT`` (before
        the projection advances), then after ``GOAL-MISSED`` ``GUARD``. Both
        are non-terminal ``command=(0, 0, 1)`` plans with ``gear`` = the
        active block's sign (regime ``"CMD"`` with the MPC layer: no solve, no
        budget charged), and neither counts toward ``GOAL-MISSED`` (that call's
        goal distance is taken back out of the history).

        * ``SHIFT`` — the engaged gear (feedback ``gear=``) differs from the
          wanted one; without feedback, while the gear the follower assumes
          differs: it is assumed engaged on the call AFTER two consecutive
          ``SHIFT`` calls at planar speed ``hypot(v_long, v_lat)`` below
          ``SHIFT_SLOW_SPEED`` (0.45 m/s). Given neither ``v_lat`` nor ``gear``
          feedback, the follower judges "slow" on ``|v_long|`` alone while the
          vehicle's shift rule uses its planar speed, so a vehicle sliding
          sideways can be assumed engaged while it has not shifted — two, because the first can fall on the vehicle's
          first-step protection step, where it does not shift. The assumed
          gear starts unknown, except that on the FIRST call that reaches this
          check a vehicle already moving at 0.5 m/s or more in the wanted
          direction is assumed in gear; :meth:`set_path` keeps it,
          :meth:`resync_gear` forgets it (then always ``SHIFT``).
        * ``GUARD`` — no feedback confirming the wanted gear, and the vehicle
          moves faster than ``V_AGAINST`` (0.3 m/s) against it: latched; it
          releases on the call after two consecutive slow ``GUARD`` calls
          (the vehicle, at throttle 0 and below 0.5 m/s, engages the wanted
          gear during them), and the wanted gear is then assumed engaged.
          A mismatch while the vehicle moves IN the block direction (e.g. an
          external reset of the vehicle's gear) is not detected; with
          feedback there is none.
        * ``GEAR-ABORT`` (terminal, ``stop_reason="shift_timeout"``) — more
          than ``SHIFT_MAX_CALLS`` (120) consecutive ``SHIFT`` / ``GUARD``
          calls; one driving call resets the count, so ``GUARD`` <-> drive
          cycles have no bound.
        """
        if self._gear_mode:
            self._fb_gear = self._check_feedback(gear)
            self._fb_vlat = v_lat
        if self._law is None:
            pd = self._plan_pd(pos_xy, yaw, v_long, pitch_deg, roll_deg)
            self.regime = pd.regime
            if self._gear_mode:
                self._stamp_gear(pd)
            return pd
        out = self._plan_mpc(pos_xy, yaw, v_long, pitch_deg, roll_deg, yaw_rate, v_lat, t)
        # (v2.0.1) what drove the step that ends here: the next call charges
        # the travel in between to the R-ALIGN / R-FAIL budget of this block.
        self._prev_regime = self.regime
        self._prev_xy = (float(pos_xy[0]), float(pos_xy[1]))
        if self._gear_mode:
            self._stamp_gear(out)
        return out

    # -- gear mode: the follower side (v2.0.2) -------------------------------------

    @staticmethod
    def _check_feedback(gear):
        """``None``, or the feedback gear as int +1 / -1 (anything ``float()``
        reads as exactly +-1: an int, a numpy scalar, a one-element tensor);
        ``ValueError`` otherwise (a ``bool`` included)."""
        if gear is None:
            return None
        try:
            g = None if isinstance(gear, bool) else float(gear)
        except (TypeError, ValueError):
            g = None
        if g not in (1.0, -1.0):
            raise ValueError(f"gear feedback must be +1, -1 or None; got {gear!r}")
        return int(g)

    def _desired_gear(self) -> int:
        """The gear the follower wants: the active block's direction sign
        (+1 forward block, -1 reverse block). The single source of the wanted
        gear — the 2.0.3 seam."""
        return 1 if self.blocks[self.block_i][2] >= 0 else -1

    def _stamp_gear(self, plan) -> None:
        if plan.mode in self.TERMINAL_MODES:
            plan.gear = (self._gear_assumed if self._gear_assumed is not None
                         else self._desired_gear())
        else:
            plan.gear = self._desired_gear()

    def _slow(self, v_long) -> bool:
        vl = self._fb_vlat
        return math.hypot(float(v_long), 0.0 if vl is None else float(vl)) < SHIFT_SLOW_SPEED

    def _shift_call(self, desired, v_long) -> bool:
        """True when this call is a ``SHIFT`` (counters advanced)."""
        fb = self._fb_gear
        if not self._gear_init_done:
            # Construction-only motion rule (plan-review C1): resync_gear()
            # never re-enters here.
            self._gear_init_done = True
            if fb is not None:
                self._gear_assumed = fb
            elif desired * float(v_long) >= GEAR_SHIFT_MAX_SPEED:
                self._gear_assumed = desired
        if fb is not None:
            self._gear_assumed = fb
            self._shift_ok = 0
            return fb != desired
        if self._gear_assumed == desired:
            self._shift_ok = 0
            return False
        if self._shift_ok >= 2:
            self._gear_assumed = desired
            self._shift_ok = 0
            return False
        self._shift_ok = self._shift_ok + 1 if self._slow(v_long) else 0
        return True

    def _guard_call(self, desired, v_long) -> bool:
        """True when this call is a ``GUARD`` (latch advanced)."""
        fb = self._fb_gear
        if fb is not None and fb == desired:
            self._guard = False
            self._guard_slow = 0
            return False
        if self._guard and self._guard_slow >= 2:
            self._guard = False
            self._guard_slow = 0
            self._gear_assumed = desired
        if not self._guard:
            if desired * float(v_long) < -V_AGAINST:
                self._guard = True
                self._guard_slow = 0
            else:
                return False
        self._guard_slow = self._guard_slow + 1 if self._slow(v_long) else 0
        return True

    def _gear_cmd_plan(self, mode, v_long, pitch_deg, roll_deg, gp_saved):
        """A SHIFT / GUARD plan, or GEAR-ABORT past the call limit. Takes
        this call's goal distance back out of the GOAL-MISSED history."""
        self._gp_hist = gp_saved
        self._shift_n += 1
        if self._shift_n > SHIFT_MAX_CALLS:
            self._terminal = ("GEAR-ABORT", "shift_timeout")
            mode = "GEAR-ABORT"
        return FollowerPlan(mode=mode, command=(0.0, 0.0, 1.0), v_long=v_long,
                            pitch_deg=pitch_deg, roll_deg=roll_deg)

    def resync_gear(self) -> None:
        """(v2.0.2, gear mode) Forget the gear the follower assumes engaged —
        call it after anything changed the vehicle's gear behind the
        follower's back (``VehicleScene.reset()``, ``reset(env_ids)``)
        when no ``gear=`` feedback is passed. Without feedback the next driving
        call is then a ``SHIFT`` (two slow calls, as at a cusp; a call that
        does pass ``gear=`` equal to the wanted gear drives instead) — the construction-time
        "already moving in the wanted direction" rule does not apply again.
        Also clears the ``SHIFT`` / ``GUARD`` counters and the ``GUARD``
        latch, like :meth:`set_path`; keeps a terminal mode. No effect with
        ``command_mode="signed"``.

        Without feedback and without this call, after an external reset of a
        vehicle that keeps moving in the block direction the follower goes on
        commanding throttle in the gear it assumes while the vehicle drives in
        its reset (opposite) gear, until the vehicle has stopped, moved back
        and tripped ``GUARD``. That stretch is NOT bounded: on flat or uphill
        ground it was estimated at about 68 control steps for the reference
        case (plan_r202 section 4.1 — an estimate, not a bound), and on a
        downhill grade (in the block direction) that equals or exceeds the
        rolling resistance it has no time bound. Passing ``gear=`` feedback
        removes it."""
        if not self._gear_mode:
            return
        self._gear_assumed = None
        self._gear_init_done = True
        self._shift_ok = 0
        self._shift_n = 0
        self._guard = False
        self._guard_slow = 0

    def _plan_mpc(self, pos_xy, yaw, v_long, pitch_deg, roll_deg, yaw_rate, v_lat, t):
        """:meth:`plan` with the MPC layer on."""
        missing = yaw_rate is None or v_lat is None
        if missing:                     # restorable: nothing moves if we raise below
            snap = (self.block_i, self.current_idx, self.projection_t,
                    self.projection_xy, self.transitioning, list(self._gp_hist),
                    self._terminal)
            gsnap = (self._gear_assumed, self._gear_init_done, self._shift_ok,
                     self._shift_n, self._guard, self._guard_slow)
        pd = self._plan_pd(pos_xy, yaw, v_long, pitch_deg, roll_deg)
        if missing and pd.command is None and self.blocks[self.block_i][2] > 0:
            (self.block_i, self.current_idx, self.projection_t,
             self.projection_xy, self.transitioning, self._gp_hist, self._terminal) = snap
            (self._gear_assumed, self._gear_init_done, self._shift_ok,
             self._shift_n, self._guard, self._guard_slow) = gsnap
            raise TypeError(
                "the MPC layer needs yaw_rate= and v_lat= on every step of a forward "
                "block — extract_state() returns both")
        t_now = self._t_next if t is None else float(t)
        self._t_next = t_now + self.dt_ctrl
        t_prev = self._t_now
        self._t_now = t_now
        handover_on = self._align_gate is not None or self._stall_timeout is not None
        if pd.command is not None:
            if handover_on:
                self._clear_handover()
            pd.regime = "CMD"
            self.regime = "CMD"
            return pd
        bs, be, sgn = self.blocks[self.block_i]
        if sgn < 0:
            if handover_on:
                self._clear_handover()
            pd.regime = "R-REV"
            self.regime = "R-REV"
            return pd
        # Per-block budgets (F3 / F5 / F8): reset on a block change, charged
        # with the travel that followed an R-ALIGN / R-FAIL step. Independent
        # of the gate / watchdog (F8 is live with both None).
        x, y = float(pos_xy[0]), float(pos_xy[1])
        if self._bud_block != self.block_i:
            self._bud_block = self.block_i
            self._d_align = self._d_fail = 0.0
            self._n_stall = 0
            self._stall_s_hw = None
        elif self._prev_xy is not None:
            d_step = math.hypot(x - self._prev_xy[0], y - self._prev_xy[1])
            if self._prev_regime == "R-ALIGN":
                self._d_align += d_step
            elif self._prev_regime == "R-FAIL":
                self._d_fail += d_step
        if handover_on:
            ho = self._handover_step(t_now, pos_xy, yaw, t_prev)
            if ho == _HO_ABORT:
                return self._abort_plan(v_long, pitch_deg, roll_deg)
            if ho == _HO_PD:
                # PD drives this step (R-ALIGN / R-STALL); the law is not called.
                pd.mode = "PD-ALIGN" if self._ho_regime == "R-ALIGN" else "PD-STALL"
                pd.regime = self._ho_regime
                self.regime = self._ho_regime
                return pd
        if self._mpc_block != self.block_i:
            self._law.start_block(self._block_waypoints(self.block_i), t0=t_now)
            self._mpc_block = self.block_i
        o = self._law.step(t_now, pos_xy, yaw, v_long, yaw_rate, v_lat)
        if o.mode == "PD-FALLBACK":
            if self._d_fail > self._pd_budget:          # F8
                self._abort("fail_budget")
                return self._abort_plan(v_long, pitch_deg, roll_deg)
            pd.mode = "PD-FALLBACK"
            pd.regime = "R-FAIL"
            self.regime = "R-FAIL"
            return pd
        self.regime = "R-MPC"
        return FollowerPlan(mode=o.mode, a_target=o.a_t, omega_target=o.w_t,
                            v_long=v_long, v_target=pd.v_target,
                            pitch_deg=pitch_deg, roll_deg=roll_deg,
                            regime="R-MPC", a_ff=o.a_t, w_ff=o.w_t)

    # -- terminal modes (v2.0.1) --------------------------------------------------------

    @property
    def stop_reason(self):
        """Why the follower stopped in a terminal mode other than ``DONE``.
        Exactly one of:

        * ``"goal_passed"`` (``GOAL-MISSED``): every mode, cannot be disabled;
        * ``"align_budget"`` (``MPC-ABORT``): MPC layer on and
          ``mpc_align_gate`` not None;
        * ``"stall_budget"`` (``MPC-ABORT``): MPC layer on and
          ``mpc_stall_timeout`` not None;
        * ``"fail_budget"`` (``MPC-ABORT``): MPC layer on and
          ``on_solve_failure="pd"``, whatever the two kwargs;
        * ``"shift_timeout"`` (``GEAR-ABORT``, v2.0.2): gear mode, more than
          ``SHIFT_MAX_CALLS`` consecutive ``SHIFT`` / ``GUARD`` calls.

        ``None`` before the first step, while driving, and on ``DONE`` (a
        ``DONE`` step clears it); reset by :meth:`set_path`. Read-only."""
        return None if self._terminal is None else self._terminal[1]

    def _abort(self, reason) -> int:
        """Enter the sticky ``MPC-ABORT`` terminal state (brake to a stop)."""
        self._terminal = ("MPC-ABORT", reason)
        self._clear_handover()
        return _HO_ABORT

    def _abort_plan(self, v_long, pitch_deg, roll_deg):
        self.regime = "CMD"
        return FollowerPlan(mode="MPC-ABORT", command=(0.0, 0.0, 1.0), v_long=v_long,
                            pitch_deg=pitch_deg, roll_deg=roll_deg, regime="CMD")

    def _goal_passed(self, pos_xy, d_goal) -> bool:
        """GP-AWAY2 (v2.0.1): on the final block with the projection on its
        last segment, beyond the goal's segment-normal plane, at least
        ``arrival_goal + 0.1`` m from the goal, and the distance to the goal
        grew by more than 1e-3 m on each of the last two :meth:`plan` calls
        (``_gp_hist`` holds the last three distances, this call's last)."""
        h = self._gp_hist
        if len(h) < 3 or not (h[1] - h[0] > GP_INC_M and h[2] - h[1] > GP_INC_M):
            return False
        if d_goal < self.arrival_goal + GP_MARGIN_M:
            return False
        if self.block_i != len(self.blocks) - 1:
            return False
        bs, be, _ = self.blocks[self.block_i]
        if self.current_idx < be - 1:
            return False
        g = self.path[be]
        j = be - 1
        while j > bs and math.hypot(g[0] - self.path[j][0], g[1] - self.path[j][1]) < 1e-9:
            j -= 1
        return ((pos_xy[0] - g[0]) * (g[0] - self.path[j][0])
                + (pos_xy[1] - g[1]) * (g[1] - self.path[j][1])) > 0

    def _heading_error(self, yaw) -> float:
        """e_psi = wrap(per-waypoint path yaw at the projection − chassis yaw)."""
        return (self.yaws[self.current_idx] - yaw + math.pi) % (2 * math.pi) - math.pi

    def _clear_handover(self) -> None:
        """Release R-ALIGN / R-STALL and drop the watchdog anchor (no re-seed);
        the ``handover`` counts are kept."""
        self._ho_regime = None
        self._ho_block = None
        self._anchor = None
        self._stall_entry = None

    def _proj_s(self) -> float:
        """Arc length along the active block from its first waypoint to the
        projection point (read-only; called at watchdog fires only)."""
        bs, be, _ = self.blocks[self.block_i]
        i = min(self.current_idx, be)
        s = 0.0
        for k in range(bs, i):
            s += math.hypot(self.path[k + 1][0] - self.path[k][0],
                            self.path[k + 1][1] - self.path[k][1])
        return s + math.hypot(self.projection_xy[0] - self.path[i][0],
                              self.projection_xy[1] - self.path[i][1])

    def _enter_align(self, e) -> int:
        """An R-ALIGN entry (both paths). F3: the block's R-ALIGN travel budget
        already spent — abort ``align_budget``. (``e`` is unused; kept,
        private.)"""
        if self._d_align > self._pd_budget:
            return self._abort("align_budget")
        self._ho_regime = "R-ALIGN"
        self._handover["align"] += 1
        self._anchor = None
        return _HO_PD

    def _handover_step(self, t_now, pos_xy, yaw, t_prev=None) -> int:
        """The gate / watchdog state machine for one forward, non-command step.
        Returns ``_HO_PD`` when the PD law drives this step
        (``self._ho_regime`` says which), ``_HO_LAW`` when the MPC law does,
        ``_HO_ABORT`` on an ``MPC-ABORT`` (F3 / F5). Reads the path state
        only; touches the law only through :meth:`_reseed_block` on a
        release. ``t_prev``: the previous :meth:`plan` call's time (F6)."""
        if self._ho_block != self.block_i:              # block changed: release
            self._clear_handover()
            self._ho_block = self.block_i
        gate, timeout = self._align_gate, self._stall_timeout
        x, y = float(pos_xy[0]), float(pos_xy[1])
        e = abs(self._heading_error(yaw))
        if self._ho_regime == "R-STALL":
            ex, ey = self._stall_entry
            if math.hypot(x - ex, y - ey) < STALL_EXIT_M:
                return _HO_PD
            self._ho_regime = None
            self._stall_entry = None
            self._reseed_block(t_now)
            if gate is not None and e > gate[0]:
                return self._enter_align(e)
            self._anchor = (t_now, x, y)
            return _HO_LAW
        if self._ho_regime == "R-ALIGN":
            if e >= gate[1]:
                if self._d_align > self._pd_budget:     # F3
                    return self._abort("align_budget")
                return _HO_PD
            self._ho_regime = None
            self._reseed_block(t_now)
            self._anchor = (t_now, x, y) if timeout is not None else None
            return _HO_LAW
        if gate is not None and e > gate[0]:
            return self._enter_align(e)
        if timeout is not None:
            a = self._anchor
            jump = t_prev is not None and t_now - t_prev > timeout / 2      # F6
            if (a is None or t_now < a[0] or jump
                    or math.hypot(x - a[1], y - a[2]) >= STALL_ANCHOR_M):
                self._anchor = (t_now, x, y)
            elif t_now - a[0] >= timeout:
                gx, gy = self.path[-1][0], self.path[-1][1]
                if math.hypot(x - gx, y - gy) > self.arrival_goal:
                    # F5 run of entries: an entry whose projection lies at least
                    # STALL_PROGRESS_M beyond every earlier entry of the block
                    # starts a new run (the count restarts at this entry).
                    s_now = self._proj_s()
                    hw = self._stall_s_hw
                    if hw is not None and max(hw, s_now) - hw >= STALL_PROGRESS_M:
                        self._n_stall = 0
                    self._stall_s_hw = s_now if hw is None else max(hw, s_now)
                    self._n_stall += 1
                    if self._n_stall > STALL_MAX_PER_BLOCK:             # F5
                        return self._abort("stall_budget")
                    self._ho_regime = "R-STALL"
                    self._handover["stall"] += 1
                    self._stall_entry = (x, y)
                    self._anchor = None
                    return _HO_PD
        return _HO_LAW

    def _reseed_block(self, t_now) -> None:
        """Re-time the active forward block's reference so the current
        projection is due at ``t_now``, and restart the law (command,
        acceleration history and ``u_prev`` reset by ``start_block``). Does not
        touch ``_t_next``."""
        bs, _, _ = self.blocks[self.block_i]
        wps = self._block_waypoints(self.block_i)
        self._law.start_block(wps, t0=0.0)
        tt = self._law.ref[0]
        k = self.current_idx - bs
        k1 = min(k + 1, len(tt) - 1)
        t_ref = float(tt[k] + self.projection_t * (tt[k1] - tt[k]))
        self._law.start_block(wps, t0=t_now - t_ref)
        self._mpc_block = self.block_i

    def _block_waypoints(self, i):
        """Block ``i``'s waypoints for the MPC reference: speeds floored at 0
        (the boundary waypoint of a forward block before a cusp carries the
        reverse speed; the reference stops there), explicit yaws only when
        every waypoint of the block has one."""
        bs, be, _ = self.blocks[i]
        idx = range(bs, be + 1)
        if all(self.wp_yaws[k] is not None for k in idx):
            return [(*self.path[k][:3], max(self.path[k][3], 0.0), self.wp_yaws[k]) for k in idx]
        return [(*self.path[k][:3], max(self.path[k][3], 0.0)) for k in idx]

    # -- MPC layer: reporting -----------------------------------------------------

    @property
    def mpc_stats(self):
        """``{n_solve, status: {converged, capped, capped_infeasible,
        "failed:<status>", exception, nan}, last_status, last_solve_ms,
        plan_age_s, handover: {align, stall}}`` — ``None`` with
        ``mpc_model="off"``. ``handover`` (v2.0.1) counts entries into
        R-ALIGN / R-STALL since construction (``set_path`` keeps it):
        cumulative, not a per-block count, and ``handover["stall"]`` is not
        bounded by 3 per block (the R-STALL limit is per run of entries)."""
        if self._law is None:
            return None
        stats = self._law.mpc_stats(self._t_now)
        stats["handover"] = dict(self._handover)
        return stats

    @property
    def last_plan(self):
        """``(times, states (N+1, 8))`` of the last applied MPC solve (rig
        frame at its solve time; a copy), or ``None`` — always ``None`` with
        ``mpc_impl="linear"`` (the QP keeps no state trajectory) and with
        ``mpc_model="off"``."""
        return None if self._law is None else self._law.last_plan

    # -- path replacement -----------------------------------------------------------

    def set_path(self, path, *, pos_xy=None, t=None):
        """Replace the path, keeping the plant (and its warm start) and the MPC
        problem (no NLP rebuild; the solver keeps its last solution as warm
        start).

        The new PATH is validated first: a path error (length, waypoint
        shape) changes nothing. A bad ``pos_xy`` or ``t`` raises AFTER the new
        path has been installed (it is not validated up front).
        The PD state restarts on the new path's first block (a cusp
        transition in progress is cancelled); with ``pos_xy`` the projection
        snaps to the globally nearest segment of that block (the 5-segment
        advance cap does not apply). With the MPC layer on, a forward first
        block gets a new reference timed so that the point nearest ``pos_xy``
        (or the block start) is due at ``t`` (default: the internal clock),
        and the MPC restarts: held command and acceleration history cleared,
        do_mpc's ``u_prev`` reset to the measured ``(d0, a0)`` before the
        next solve.

        (v2.0.2) In gear mode it also restarts the ``SHIFT`` / ``GUARD``
        counters and releases the ``GUARD`` latch (and, as the terminal state
        is cleared, a ``GEAR-ABORT``); it KEEPS the gear the follower assumes
        engaged, so a ``set_path`` while driving in the same direction issues
        no ``SHIFT``. A vehicle still moving against the wanted gear latches
        ``GUARD`` again on the next call.

        (v2.0.1) It also clears ``GOAL-MISSED`` / ``MPC-ABORT`` and
        :attr:`stop_reason`, the GP-AWAY2 history, and every MPC budget
        (R-ALIGN / R-FAIL travel, R-STALL count and progress mark): the
        budgets are per forward block per path, so a caller that re-plans
        repeatedly is not bounded by them. With ``mpc_model="off"`` it clears
        only the terminal state and the GP-AWAY2 history; with the MPC layer
        it also clears R-ALIGN / R-STALL, the watchdog anchor and entry point,
        the R-ALIGN and R-FAIL travel, the R-STALL count and progress mark;
        the ``handover`` counts are kept, and the heading gate is evaluated
        again on the next :meth:`plan` (a vehicle still misaligned enters
        R-ALIGN at once). Measured by the impl reviewer (fake solver, states
        fed directly): a car in R-ALIGN at 0.1 m per call with
        ``set_path(same path)`` every 100 calls travelled 40 m without an
        abort (budget 23.24 m).
        """
        if len(path) < 2:
            raise ValueError("path needs at least 2 waypoints")
        parsed = self._parse_path(path)
        (self.path, self.wp_yaws, self.n, self.blocks, self.yaws,
         self._block_end_yaw) = parsed
        self.block_i = 0
        blk_start, blk_end, sgn = self.blocks[0]
        self.current_idx = blk_start
        self.projection_t = 0.0
        self.projection_xy = (self.path[blk_start][0], self.path[blk_start][1])
        self.transitioning = False
        self._terminal = None           # (v2.0.1) terminal state and GP-AWAY2 history
        self._gp_hist = []
        # (v2.0.2) gear mode: SHIFT / GUARD counters and the GUARD latch
        # restart; the gear assumed engaged is KEPT (the vehicle's gear does
        # not depend on the path; resync_gear() forgets it).
        self._shift_ok = 0
        self._shift_n = 0
        self._guard = False
        self._guard_slow = 0
        if pos_xy is not None:
            self._snap(pos_xy)
        if self._law is None:
            return
        if not any(s_ > 0 for (_, _, s_) in self.blocks):
            import warnings
            warnings.warn("the MPC layer never runs on this path (no forward block); "
                          "every block is driven by the PD law", UserWarning, stacklevel=2)
        t_now = self._t_next if t is None else float(t)
        self._t_next = t_now
        self._mpc_block = None
        self._clear_handover()
        self._reset_budgets()
        if sgn > 0:
            # block 0; without pos_xy the projection is the block start, so
            # t_ref = tt[0] = 0.0 exactly (as through 2.0.0).
            self._reseed_block(t_now)

    def _snap(self, pos_xy):
        """Project ``pos_xy`` onto the nearest segment of the active block."""
        bs, be, _ = self.blocks[self.block_i]
        best = (float("inf"), bs, 0.0)
        for i in range(bs, be):
            p0, p1 = self.path[i], self.path[i + 1]
            sx, sy = p1[0] - p0[0], p1[1] - p0[1]
            l2 = sx * sx + sy * sy
            if l2 < 1e-9:
                continue
            tc = max(0.0, min(1.0, ((pos_xy[0] - p0[0]) * sx + (pos_xy[1] - p0[1]) * sy) / l2))
            d = (pos_xy[0] - p0[0] - tc * sx) ** 2 + (pos_xy[1] - p0[1] - tc * sy) ** 2
            if d < best[0]:
                best = (d, i, tc)
        _, self.current_idx, self.projection_t = best
        if self.current_idx < be:
            p0, p1 = self.path[self.current_idx], self.path[self.current_idx + 1]
            self.projection_xy = (p0[0] + self.projection_t * (p1[0] - p0[0]),
                                  p0[1] + self.projection_t * (p1[1] - p0[1]))

    def _plan_pd(self, pos_xy, yaw, v_long, pitch_deg, roll_deg):
        """The 1.x stage 1 (PD law + path state machine), plus two v2.0.1
        changes that apply in every mode, ``"off"`` included, and cannot be
        switched off: the goal-passed stop GP-AWAY2 (``GOAL-MISSED``, see
        :meth:`_goal_passed`) and W2 (at a block end the target speed keeps
        the CURRENT block's direction sign; 1.x read the next block's sign
        there when the lookahead sat exactly on the block-end waypoint
        without being capped). Otherwise the 1.x law, unchanged.

        Advances the path state machine and turns the geometry into the
        physical response the vehicle should produce — a desired acceleration
        and yaw rate. Returns a :class:`FollowerPlan`; when its ``command`` is
        set the step is already decided (DONE, cusp brake, a terminal mode)
        and no inversion is needed.
        """
        last_block = self.block_i == len(self.blocks) - 1
        blk_start, blk_end, dir_sign = self.blocks[self.block_i]

        # DONE: near the final waypoint while driving the final block.
        last = self.path[-1]
        d_goal = math.hypot(pos_xy[0] - last[0], pos_xy[1] - last[1])
        if last_block and d_goal < self.arrival_goal:
            self._terminal = None       # (v2.0.1) DONE takes precedence, clears stop_reason
            return FollowerPlan(mode="DONE", command=(0.0, 0.0, 1.0),
                                v_long=v_long, pitch_deg=pitch_deg,
                                roll_deg=roll_deg)
        # (v2.0.1) GOAL-MISSED / MPC-ABORT are sticky until set_path.
        if self._terminal is not None:
            return FollowerPlan(mode=self._terminal[0], command=(0.0, 0.0, 1.0),
                                v_long=v_long, pitch_deg=pitch_deg,
                                roll_deg=roll_deg)
        gp_saved = list(self._gp_hist) if self._gear_mode else None
        self._gp_hist = (self._gp_hist + [d_goal])[-3:]

        # Cusp state machine: brake to a stop at the block boundary, then
        # switch to the next block (flipped direction).
        if self.transitioning:
            if abs(v_long) > self.v_stop:
                return FollowerPlan(mode="BRAKE_TRANS", command=(0.0, 0.0, 1.0),
                                    v_long=v_long, pitch_deg=pitch_deg,
                                    roll_deg=roll_deg)
            self.transitioning = False
            self.block_i += 1
            blk_start, blk_end, dir_sign = self.blocks[self.block_i]
            last_block = self.block_i == len(self.blocks) - 1
            self.current_idx = blk_start
            self.projection_t = 0.0
            self.projection_xy = (self.path[blk_start][0],
                                  self.path[blk_start][1])
        elif not last_block:
            cusp = self.path[blk_end]
            # Reached when the projection sits at the block's end point
            # (robust to duplicated/zero-length boundary waypoints) or the
            # vehicle itself is within cusp_goal of the cusp.
            reached = (self.current_idx >= blk_end
                       or math.hypot(self.projection_xy[0] - cusp[0],
                                     self.projection_xy[1] - cusp[1]) < 0.25)
            if reached or math.hypot(pos_xy[0] - cusp[0],
                                     pos_xy[1] - cusp[1]) < self.cusp_goal:
                self.transitioning = True
                return FollowerPlan(mode="BRAKE_TRANS", command=(0.0, 0.0, 1.0),
                                    v_long=v_long, pitch_deg=pitch_deg,
                                    roll_deg=roll_deg)

        if self._gear_mode:
            # (v2.0.2) SHIFT: after the cusp machine, before the projection.
            desired = self._desired_gear()
            if self._shift_call(desired, v_long):
                return self._gear_cmd_plan("SHIFT", v_long, pitch_deg, roll_deg, gp_saved)

        self._advance(pos_xy)

        # (v2.0.1) GP-AWAY2: past the goal on the final segment, moving away.
        if self._goal_passed(pos_xy, d_goal):
            self._terminal = ("GOAL-MISSED", "goal_passed")
            return FollowerPlan(mode="GOAL-MISSED", command=(0.0, 0.0, 1.0),
                                v_long=v_long, pitch_deg=pitch_deg,
                                roll_deg=roll_deg)

        if self._gear_mode:
            # (v2.0.2) GUARD: after GOAL-MISSED (a terminal wins), before the
            # lookahead. A driving call resets the SHIFT/GUARD call count.
            if self._guard_call(desired, v_long):
                return self._gear_cmd_plan("GUARD", v_long, pitch_deg, roll_deg, gp_saved)
            self._shift_n = 0

        la_idx, capped = self._lookahead()
        la = self.path[la_idx]
        v_target = la[3]
        if abs(v_target) < 1e-6:
            v_target = dir_sign * 1e-6     # explicit stop wp: keep direction
        if capped and not last_block:
            # Cusp inside the lookahead window: taper the speed with the
            # remaining distance so the vehicle arrives slow enough to stop.
            d_cusp = math.hypot(pos_xy[0] - self.path[blk_end][0],
                                pos_xy[1] - self.path[blk_end][1])
            v_mag = min(abs(v_target), self.k_approach * d_cusp + 0.3)
            v_target = math.copysign(v_mag, dir_sign)
        if la_idx == blk_end and not last_block and v_target * dir_sign < 0:
            # (v2.0.1) W2: the block-end waypoint carries the NEXT block's
            # speed; keep this block's direction (forward->reverse and
            # reverse->forward). The magnitude is still that waypoint's.
            v_target = math.copysign(abs(la[3]), dir_sign)

        # Heading error: blend of path yaw and bearing-to-lookahead.
        dx = la[0] - pos_xy[0]; dy = la[1] - pos_xy[1]
        bearing = math.atan2(dy, dx)
        # The block-end waypoint is shared with the NEXT block; use this
        # block's continue-straight yaw there instead of the (next-block,
        # possibly flipped) per-waypoint value.
        if la_idx >= blk_end and not last_block:
            path_yaw = self._block_end_yaw[self.block_i]
        else:
            path_yaw = self.yaws[la_idx]
        yaw_err = (path_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        if dir_sign >= 0:
            pos_err = (bearing - yaw + math.pi) % (2 * math.pi) - math.pi
        else:
            pos_err = (bearing + math.pi - yaw + math.pi) % (2 * math.pi) - math.pi
        heading_err = 0.7 * yaw_err + 0.3 * pos_err

        return FollowerPlan(
            mode=(f"DRV{dir_sign:+d}" if dir_sign != 0 else "STOP"),
            a_target=self.k_v * (v_target - v_long),
            omega_target=self.k_w * heading_err,
            v_long=v_long, v_target=v_target,
            pitch_deg=pitch_deg, roll_deg=roll_deg)
