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
    #: ``"R-FAIL"`` (PD after a failed solve), ``"CMD"`` (DONE / cusp brake
    #: with the MPC layer on). ``a_ff`` / ``w_ff``: the MPC feedforward.
    regime: str = "R-OFF"
    a_ff: Optional[float] = None
    w_ff: Optional[float] = None


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
        follower. Returns a list of ``(throttle, steer, brake)``.
        """
        states = list(states)
        if len(states) != len(self.followers):
            raise ValueError(
                f"got {len(states)} states for {len(self.followers)} followers")

        plans = []
        for f, st in zip(self.followers, states):
            if isinstance(st, dict):
                st = (st["pos_xy"], st["yaw"], st["v_long"],
                      st["pitch"], st["roll"])
            plans.append(f.plan(*st))

        # One inversion for everyone. Followers whose step is already decided
        # still occupy a slot — asking for their current response makes their
        # solve a no-op, and a batched unroll does not care about the width.
        v_long = [p.v_long for p in plans]
        a_t = [p.a_target for p in plans]
        w_t = [p.omega_target for p in plans]
        thr, steer = self.plant.solve(v_long, a_t, w_t)

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

    MPC layer (v2.0.0; every argument below is MPC-only and raises
    ``ValueError`` when given together with ``mpc_model="off"``):

        mpc_params: path to this vehicle's params JSON (written by
            ``python -m genesis_vehicle.control.identify``) or an
            :class:`~genesis_vehicle.control.mpc.MPCVehicleParams`. REQUIRED
            unless ``mpc_model="off"``.
        mpc_model: ``"bicycle"`` (default; the field AlpaSim bicycle model) or
            ``"off"`` (no MPC layer: the 1.x PD follower, bit-identical to
            1.8.0 on the scenarios ``tests/test_pd_golden.py`` replays).
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
            or ``"hold"``, the parity mode with the field controller. What
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
    a failed solve).
    """

    #: Release-specific accepted values of ``mpc_model``.
    MPC_MODELS = ("bicycle", "off")

    def __init__(self, path, plant=None, *, sweep=None,
                 mpc_params=_Unset(None), mpc_model: str = "bicycle",
                 mpc_impl=_Unset("nonlinear"), on_solve_failure=_Unset("pd"),
                 mpc_gains=_Unset(None), mpc_v_floor=_Unset(0.5),
                 dt_ctrl=_Unset(None),
                 lookahead: float = 3.5, arrival_goal: float = 1.5,
                 cusp_goal: float = 1.0, k_v: float = 2.0, k_w: float = 1.5,
                 k_approach: float = 1.0, v_stop: float = 0.05,
                 steer_cap: float = 0.5, plant_kwargs=None):
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
                  "mpc_v_floor": mpc_v_floor, "dt_ctrl": dt_ctrl}
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
                                    ("dt_ctrl", dt_ctrl)) if not isinstance(v, _Unset)]
            if given:
                raise ValueError(f"mpc_model=\"off\" takes no MPC argument; got {', '.join(given)}")
            mpc_setup = None
        else:
            mpc_setup = self._check_mpc_args(plant, mpc_model, **mpc_kw)

        self.mpc_model = mpc_model
        self.plant = _as_plant(plant, plant_kwargs or {})
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
        if mpc_on:
            self._init_mpc(mpc_setup, steer_cap)

    # -- MPC layer: construction ------------------------------------------------

    def _check_mpc_args(self, plant, mpc_model, *, mpc_params, mpc_impl,
                        on_solve_failure, mpc_gains, mpc_v_floor, dt_ctrl):
        """Steps 5-10 of the constructor's check order (each raises before any
        side effect): live-vehicle steering, params present, implementation,
        dependencies, params file, forward blocks."""
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
        # (8) dependencies.
        from .mpc._deps import require_mpc_deps
        require_mpc_deps(mpc_impl)
        # (9) the params record (missing file, schema, keys, JSON steering).
        from .mpc.params import load_mpc_params
        params = load_mpc_params(mpc_params)
        return dict(params=params, impl=mpc_impl, policy=on_solve_failure,
                    gains=mpc_gains, v_floor=0.5 if mpc_v_floor is None else float(mpc_v_floor),
                    dt_ctrl=dt_ctrl, model=mpc_model)

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
             yaw_rate=None, v_lat=None, t=None):
        """One control step. Returns ``(throttle, steer, brake)``, each in
        the SDK input range (throttle/steer in [-1, 1], brake in [0, 1]).

        ``yaw_rate`` (rad/s) and ``v_lat`` (m/s) are required while the MPC
        layer drives a forward block (``TypeError`` otherwise;
        :func:`extract_state` returns both) and ignored elsewhere. ``t`` (s):
        the control time; ``None`` advances an internal clock by ``dt_ctrl``
        per call (from 0 at construction / the ``set_path`` time), a given
        ``t`` also resets that clock. Ignored with ``mpc_model="off"``."""
        plan = self.plan(pos_xy, yaw, v_long, pitch_deg, roll_deg,
                         yaw_rate=yaw_rate, v_lat=v_lat, t=t)
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
        """
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

    def plan(self, pos_xy, yaw, v_long, pitch_deg, roll_deg, *,
             yaw_rate=None, v_lat=None, t=None):
        """The vehicle-agnostic half of a control step (stage 1).

        Runs the path state machine (projection, blocks, cusp, DONE) every
        step, and returns a :class:`FollowerPlan`. With ``mpc_model="off"``
        that is the 1.x PD plan, unchanged. With the MPC layer on, a forward
        block is planned by the MPC (``regime="R-MPC"``); a reverse block
        (``"R-REV"``) and a step after a failed solve under
        ``on_solve_failure="pd"`` (``"R-FAIL"``) get the PD plan.
        """
        if self._law is None:
            pd = self._plan_pd(pos_xy, yaw, v_long, pitch_deg, roll_deg)
            self.regime = pd.regime
            return pd
        missing = yaw_rate is None or v_lat is None
        if missing:                     # restorable: nothing moves if we raise below
            snap = (self.block_i, self.current_idx, self.projection_t,
                    self.projection_xy, self.transitioning)
        pd = self._plan_pd(pos_xy, yaw, v_long, pitch_deg, roll_deg)
        if missing and pd.command is None and self.blocks[self.block_i][2] > 0:
            (self.block_i, self.current_idx, self.projection_t,
             self.projection_xy, self.transitioning) = snap
            raise TypeError(
                "the MPC layer needs yaw_rate= and v_lat= on every step of a forward "
                "block — extract_state() returns both")
        t_now = self._t_next if t is None else float(t)
        self._t_next = t_now + self.dt_ctrl
        self._t_now = t_now
        if pd.command is not None:
            pd.regime = "CMD"
            self.regime = "CMD"
            return pd
        bs, be, sgn = self.blocks[self.block_i]
        if sgn < 0:
            pd.regime = "R-REV"
            self.regime = "R-REV"
            return pd
        if self._mpc_block != self.block_i:
            self._law.start_block(self._block_waypoints(self.block_i), t0=t_now)
            self._mpc_block = self.block_i
        o = self._law.step(t_now, pos_xy, yaw, v_long, yaw_rate, v_lat)
        if o.mode == "PD-FALLBACK":
            pd.mode = "PD-FALLBACK"
            pd.regime = "R-FAIL"
            self.regime = "R-FAIL"
            return pd
        self.regime = "R-MPC"
        return FollowerPlan(mode=o.mode, a_target=o.a_t, omega_target=o.w_t,
                            v_long=v_long, v_target=pd.v_target,
                            pitch_deg=pitch_deg, roll_deg=roll_deg,
                            regime="R-MPC", a_ff=o.a_t, w_ff=o.w_t)

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
        plan_age_s}`` — ``None`` with ``mpc_model="off"``."""
        if self._law is None:
            return None
        return self._law.mpc_stats(self._t_now)

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
        if sgn > 0:
            wps = self._block_waypoints(0)
            self._law.start_block(wps, t0=0.0)
            t_ref = 0.0
            if pos_xy is not None:
                tt, X, Y, H = self._law.ref
                k = self.current_idx - blk_start
                k1 = min(k + 1, len(tt) - 1)
                t_ref = float(tt[k] + self.projection_t * (tt[k1] - tt[k]))
            self._law.start_block(wps, t0=t_now - t_ref)
            self._mpc_block = 0

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
        """The 1.x stage 1 (PD law + path state machine), unchanged.

        Advances the path state machine and turns the geometry into the
        physical response the vehicle should produce — a desired acceleration
        and yaw rate. Returns a :class:`FollowerPlan`; when its ``command`` is
        set the step is already decided (DONE, cusp brake) and no inversion is
        needed.
        """
        last_block = self.block_i == len(self.blocks) - 1
        blk_start, blk_end, dir_sign = self.blocks[self.block_i]

        # DONE: near the final waypoint while driving the final block.
        last = self.path[-1]
        if last_block and math.hypot(pos_xy[0] - last[0],
                                     pos_xy[1] - last[1]) < self.arrival_goal:
            return FollowerPlan(mode="DONE", command=(0.0, 0.0, 1.0),
                                v_long=v_long, pitch_deg=pitch_deg,
                                roll_deg=roll_deg)

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

        self._advance(pos_xy)

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
