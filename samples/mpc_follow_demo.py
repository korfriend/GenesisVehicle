"""mpc_follow_demo.py — the 2.0.0 default path follower: sweep table + MPC layer.

The reference car follows the field demo course (straight 20 m, a 30 m-radius
left turn of 90 degrees, straight 15 m; from rest, ramping at 1 m/s^2 to
8 m/s) with ``PathFollower`` running the MPC layer (AlpaSim bicycle model) on
the forward block and the inverse plant in stage 2.

    python -m genesis_vehicle.samples.mpc_follow_demo                    # NMPC + table
    python -m genesis_vehicle.samples.mpc_follow_demo --mpc-impl linear  # LMPC (OSQP)
    python -m genesis_vehicle.samples.mpc_follow_demo --mpc-model off    # the 1.x PD follower
    python -m genesis_vehicle.samples.mpc_follow_demo --plant diff       # DifferentiablePlant in stage 2
    python -m genesis_vehicle.samples.mpc_follow_demo --course cusp      # forward, cusp, reverse (PD)
    python -m genesis_vehicle.samples.mpc_follow_demo --course misaligned --t-max 60  # the 2.0.0 stall start
    python -m genesis_vehicle.samples.mpc_follow_demo --course vertex --vertex-deg 150
    python -m genesis_vehicle.samples.mpc_follow_demo --radius 17 --v 10  # tighter demo turn
    python -m genesis_vehicle.samples.mpc_follow_demo --course behind --behind-len 6 --kturn on --t-max 60  # 2.0.3 K-turn
    python -m genesis_vehicle.samples.mpc_follow_demo --decel-limit brake  # 2.1.0 opt-in (end-gated brake bound)
    python -m genesis_vehicle.samples.mpc_follow_demo --lead-in 60 --radius 11.3 --v 10 --final-len 3 --turn right

2.1.0 options (all opt-in; with none of them a run is the 2.0.6 run):
``--decel-limit throttle|brake`` (passes ``mpc_decel_limit``; absent = the follower's default
``"throttle"``; exits 2 with ``--mpc-model off``, and ``brake`` exits 2 with ``--command-mode signed``);
demo-course geometry ``--lead-in L`` (first straight, default 20 m), ``--final-len F`` (last
straight, default 15 m) and ``--turn left|right`` (default left; the defaults give the 2.0.6
waypoints exactly); ``--mu-lat`` / ``--mu-long`` set every wheel's ``WheelConfig.mu_lat`` /
``mu_long`` before ``build()`` through ``control.sweep_measure.apply_plant_overrides`` (the
identify / sweep_measure mechanism; the ground plane and chassis friction stay 1.0 — they act only
on Genesis rigid contacts, not on the ray-wheel tyre) and require both ``--table`` and
``--params`` (exit 2 otherwise, so the cache is never read or written with another mu). The demo
prints one ``WARNING record tyre mu != vehicle mu`` line when the params record's tyre mu
differs from the vehicle's; the SDK's ``SweepTable`` path itself checks nothing. ``--log`` rows
gain ``v_lat``, ``yaw_rate``, ``steer_raw`` (stage-2 steer before the ``steer_cap`` clip; ``null``
on steps that run no steer inversion) and ``front_roll_min`` (min over the front wheels of
``omega * r / max(v_long, 0.1)``) and ``decel_open`` (``mpc_stats["decel_gate"]["open"]`` after the
step: whether the last MPC solve used the brake row under ``--decel-limit brake``, held between
solves; ``null`` under the default throttle bound, with ``--mpc-model off`` and before the first
solve); the result gains ``mpc_decel_limit``, the asset sha256s, ``v_done``, ``rest_xy``,
``post_terminal``, ``path_sha256``, ``live_mu``, ``record_tyre_mu``, ``record_plant_overrides``,
``setup_ms``, ``decel_switch_steps`` (one ``{k, t, open, status}`` per solve that switched the
deceleration row) and ``decel_first_open_k``. ``--decel-limit brake`` uses the brake row only on
solves whose active forward block's reference end time is within 2.0 s of the solve time (in
reference time) and whose rig reference end point is ahead of the rear axle; every other solve uses
the throttle row — mid-path speed drops (and a speed profile that ramps down before the last 2 s of
reference time) get the throttle row, as does every solve after the end point has passed behind the
rear axle. Equality with the default before the first open solve is measured, not guaranteed. Full
statement: the ``mpc_decel_limit`` docstring of ``PathFollower``.

Courses (v2.0.1 adds ``misaligned``, ``vertex`` and ``--radius``; v2.0.3 adds
``behind``, described under K-turn below): ``demo`` (the field
course; ``--radius`` sets the turn radius, default 30 m); ``cusp``;
``misaligned`` — the path (30,0) -> (10,-8) -> (-10,-8) -> (-30,0) at 2 m/s,
the car spawned on its first waypoint with yaw ``--start-yaw`` (default 0,
i.e. 2.76 rad off the path heading: the start on which the 2.0.0 MPC stood
still; it needs ``--t-max 60``: on the reference setup it reaches DONE at
41.775 s, after the default 40 s); ``vertex`` — straight 20 m, a vertex with heading change
``--vertex-deg``, straight 20 m (0.5 m waypoint spacing, 4 m/s at every
waypoint, start from rest). ``--start-yaw`` (rad) spawns any course at that
yaw. ``--align-gate off`` / ``--stall-timeout off`` pass
``mpc_align_gate=None`` / ``mpc_stall_timeout=None``. Both off is NOT an
exact 2.0.0 restore: ``GOAL-MISSED``, the block-end speed sign (W2) and, under
``on_solve_failure="pd"``, the ``fail_budget`` abort stay on; a run equals 2.0.0
only if none of them acts. ``--log PATH`` writes one JSON line per control step (time,
state, regime, mode, command, solve status; v2.0.2 also the commanded and the engaged
gear).

K-turn (v2.0.3, opt-in): ``--kturn on`` passes ``mpc_kturn=KTurnParams()`` (default ``off``: the
follower's default ``mpc_kturn=None``, so every default run is unchanged); ``--lateral-limit L`` /
``--max-reverse M`` pass those ``KTurnParams`` fields and imply ``--kturn on``. ``--course behind`` is a
straight path ``--behind-len`` m (default 6) BEHIND the spawned car (path from the spawn along -X at
2 m/s, the car facing +X): the reference K-turn case — with the K-turn the follower stops, plans, and drives
forward/reverse legs before handing back to the MPC; without it, it makes the 2.0.2 forward U-turn. The
K-turn legs are reported as regime ``R-KTURN`` in ``--log`` and counted in ``mpc_stats["kturn"]``;
``reverse_entry_step`` counts only ``DRV-1`` (a reverse block), not a K-turn reverse leg.

Gear mode (v2.0.2): ``--command-mode gear`` (default, also with ``--mpc-model off``)
drives the car with ``set_inputs(..., gear=)`` from the follower's ``GearCommand``
(decelerations by brake, reverse only on the cusp course's reverse block —
and, with ``--kturn on``, on a K-turn's reverse legs);
``--gear-feedback on`` (default) passes the engaged gear ``int(veh.gear[0])`` back to
``PathFollower.step(gear=)``, ``off`` lets the follower run without it.
``--command-mode signed`` is the 2.0.1 signed-throttle loop.

On a terminal mode the demo applies the terminal command (throttle 0, brake 1) and
keeps stepping until the car is still (|v| < 0.01 m/s) or 2 s have passed, then
exits (v2.0.2; through 2.0.1 it left the loop without applying it). The reported
metrics (time, CTE, modes) stop at the terminal step.

Onboarding is two measurements, done on the first run and CACHED in
``samples/data/mpc_cache/`` (git-ignored; ``--remeasure`` refreshes):

1. a sweep table on a grid that covers the driving speeds — the default
   ``sweep_measure`` grid stops at |v| = 4 m/s, so the grid flags are part of
   the command: ``python -m genesis_vehicle.control.sweep_measure --urdf ...
   --preset ... --v-list=-2,0,2,4,6,8,10,12,14,16 --throttle-list=-1,-0.5,0,0.5,1
   --steer-list=-1,-0.5,0,0.5,1 --pitch-list=-10,-5,0,5,10 --roll-list=0
   --output table.csv``;
2. the MPC parameters: ``python -m genesis_vehicle.control.identify --urdf ...
   --preset ... --output params.json``.

Both with the same URDF, preset, dt and substeps (the code does not check that
they match). ``--table`` / ``--params`` use existing files instead.

The NMPC solve is NOT guaranteed to fit the 25 ms control period: the demo
prints how many control steps took longer than 25 ms (control + simulation);
on the measured machine that share varied between sessions (see
``bench_path_follower.py`` and the docs). The LMPC measured 0 % over 25 ms on
that machine (CPU/WSL2, genesis-world 1.4.5, this course) — measure yours.

PASS: DONE within ``--t-max`` and maximum cross-track error < 1.0 m (the
other courses: DONE within ``--t-max``). The run ends on any terminal mode
(``DONE``, ``GOAL-MISSED``, ``MPC-ABORT``, v2.0.2 ``GEAR-ABORT``; ``PathFollower.TERMINAL_MODES``)
and reports it with ``stop_reason``; only ``DONE`` is a PASS. The last line is a ``MPC_RESULT {json}``
record (consumed by ``bench_path_follower.py``).

| abbr | meaning |
|---|---|
| MPC | Model Predictive Control |
| NMPC | nonlinear MPC (do_mpc + IPOPT), ``--mpc-impl nonlinear`` |
| LMPC | linear MPC (OSQP quadratic program), ``--mpc-impl linear`` |
| PD | the 1.x proportional path-follower law, ``--mpc-model off`` |
| CTE | cross-track error (distance to the course polyline) |
"""
from __future__ import annotations

import sys
import pathlib
_SDK_PARENT = str(pathlib.Path(__file__).resolve().parents[2])
if _SDK_PARENT not in sys.path:
    sys.path.insert(0, _SDK_PARENT)

import argparse
import json
import math
import os
import subprocess
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(_HERE, "data", "mpc_cache")
URDF = os.path.join(_HERE, "urdf", "car_4w.urdf")
DT, SUBSTEPS, STEER_CAP = 0.025, 4, 0.5
GRID_FLAGS = ["--v-list=-2,0,2,4,6,8,10,12,14,16", "--throttle-list=-1,-0.5,0,0.5,1",
              "--steer-list=-1,-0.5,0,0.5,1", "--pitch-list=-10,-5,0,5,10", "--roll-list=0"]


def demo_course(v=8.0, ramp=1.0, radius=30.0, lead_in=20.0, final_len=15.0, turn="left"):
    """Field demo course (dense 0.1 m line -> 0.5 m waypoints): straight 20 m,
    a 90 degree left turn of ``radius`` (field: 30 m), straight 15 m. Returns
    ``(path, dense polyline (n, 2))``. (v2.1.0) ``lead_in`` / ``final_len`` set
    the first / last straight (m) and ``turn="right"`` mirrors the course in y;
    the defaults reproduce the 2.0.6 waypoints exactly."""
    if turn not in ("left", "right"):
        raise ValueError(f"turn must be 'left' or 'right'; got {turn!r}")
    mir = -1.0 if turn == "right" else 1.0
    lead = float(lead_in)
    pts = [(x, 0.0) for x in np.arange(0.0, lead, 0.1)]
    R = float(radius)
    for th in np.arange(0.0, math.pi / 2, 0.1 / R):
        pts.append((lead + R * math.sin(th), mir * (R - R * math.cos(th))))
    for d in np.arange(0.0, float(final_len) + 1e-9, 0.1):
        pts.append((lead + R, mir * (R + d)))
    P = np.array(pts)
    S = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    V = np.minimum(v, np.sqrt(2 * ramp * (S + 0.5)))
    return [(float(x), float(y), 0.0, float(s)) for (x, y), s in zip(P[::5], V[::5])], P


def cusp_course(v_fwd=4.0, v_rev=-2.0, ramp=1.0):
    """Forward 20 m (ramped), cusp, reverse 10 m along the same line."""
    xf = np.arange(0.0, 20.0 + 1e-9, 0.5)
    vf = np.minimum(v_fwd, np.sqrt(2 * ramp * (xf + 0.5)))
    xr = np.arange(20.0, 10.0 - 1e-9, -0.5)
    path = [(float(x), 0.0, 0.0, float(s)) for x, s in zip(xf, vf)] + \
           [(float(x), 0.0, 0.0, v_rev) for x in xr]
    P = np.array([(x, 0.0) for x in np.arange(0.0, 20.0 + 1e-9, 0.1)])
    return path, P


def _polyline(corners, step=0.1):
    """Dense ``(n, 2)`` polyline through ``corners`` (for the CTE)."""
    pts = []
    for (x0, y0), (x1, y1) in zip(corners[:-1], corners[1:]):
        n = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / step)))
        pts += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(n)]
    pts.append(corners[-1])
    return np.array(pts, float)


#: The path the 2.0.0 MPC stall was documented on (spawn on its first waypoint).
MISALIGNED_CORNERS = [(30.0, 0.0), (10.0, -8.0), (-10.0, -8.0), (-30.0, 0.0)]


def misaligned_course(v=2.0):
    """The documented 2.0.0 stall path: 4 sparse waypoints at ``v``."""
    path = [(x, y, 0.0, float(v)) for x, y in MISALIGNED_CORNERS]
    return path, _polyline(MISALIGNED_CORNERS)


def vertex_course(deg=90.0, v=4.0, spacing=0.5):
    """Straight 20 m, a vertex with heading change ``deg``, straight 20 m;
    waypoints every ``spacing`` m, all at ``v`` (start from rest)."""
    th = math.radians(deg)
    corners = [(0.0, 0.0), (20.0, 0.0), (20.0 + 20.0 * math.cos(th), 20.0 * math.sin(th))]
    path = []
    for (x0, y0), (x1, y1) in zip(corners[:-1], corners[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / spacing))
        path += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, 0.0, float(v)) for i in range(n)]
    path.append((*corners[-1], 0.0, float(v)))
    return path, _polyline(corners)


def behind_course(length=6.0, v=2.0, spacing=0.5):
    """(v2.0.3) A straight path ``length`` m behind the spawn: from (0, 0) along -X at ``v`` (the car
    spawns at (0, 0) facing +X) — the reference K-turn case."""
    n = max(1, int(length / spacing))
    path = [(-length * i / n, 0.0, 0.0, float(v)) for i in range(n + 1)]
    return path, _polyline([(0.0, 0.0), (-length, 0.0)])


COURSE_V = {"demo": 8.0, "cusp": 4.0, "misaligned": 2.0, "vertex": 4.0, "behind": 2.0}


def ensure_cache(urdf, preset, remeasure=False, gpu=False, log=print):
    """``(table_csv, params_json)`` in ``samples/data/mpc_cache/``, measured
    (one subprocess each) when missing or on ``remeasure``."""
    os.makedirs(CACHE, exist_ok=True)
    stem = f"{os.path.splitext(os.path.basename(urdf))[0]}_{preset}"
    table = os.path.join(CACHE, f"{stem}_sweep.csv")
    params = os.path.join(CACHE, f"{stem}_mpc_params.json")
    common = ["--urdf", urdf, "--preset", preset, "--dt", str(DT), "--substeps", str(SUBSTEPS)]
    if gpu:
        common.append("--gpu")
    if remeasure or not os.path.exists(table):
        log(f"[cache] measuring the sweep table -> {table}")
        subprocess.run([sys.executable, "-m", "genesis_vehicle.control.sweep_measure", *common,
                        *GRID_FLAGS, "--n-envs", "625", "--output", table],
                       check=True, cwd=_SDK_PARENT)
    if remeasure or not os.path.exists(params):
        log(f"[cache] identifying the MPC parameters -> {params}")
        subprocess.run([sys.executable, "-m", "genesis_vehicle.control.identify", *common,
                        "--output", params], check=True, cwd=_SDK_PARENT)
    return table, params


def run(*, table, params, preset="car_4w_rwd_ackermann", urdf=URDF, mpc_model="bicycle",
        mpc_impl="nonlinear", plant="table", course="demo", v=None, ramp=1.0, t_max=40.0,
        on_solve_failure="pd", gpu=False, viewer=False, log=print, radius=30.0,
        vertex_deg=90.0, start_yaw=None, align_gate=True, stall_timeout=True, log_path=None,
        command_mode="gear", gear_feedback=True, kturn=False, lateral_limit=None, max_reverse=None,
        behind_len=6.0, decel_limit=None, lead_in=20.0, final_len=15.0, turn="left",
        mu_lat=None, mu_long=None):
    """Drive one run; returns the result dict (also the ``MPC_RESULT`` record).

    ``command_mode`` ``"gear"`` (default) / ``"signed"`` is passed to
    ``PathFollower``; ``gear_feedback`` passes the engaged gear to
    ``PathFollower.step(gear=)`` in gear mode.

    ``v`` None = the course's speed (demo 8, cusp 4, misaligned 2, vertex 4 m/s).
    ``align_gate`` / ``stall_timeout`` False pass ``mpc_align_gate=None`` /
    ``mpc_stall_timeout=None`` (MPC only). ``log_path``: one JSON line per step.
    ``kturn`` (v2.0.3): pass ``mpc_kturn=KTurnParams(lateral_limit, max_reverse)`` (MPC only; False
    leaves the follower's default ``None``); ``behind_len``: the ``"behind"`` course length.
    (v2.1.0) ``decel_limit`` (None = the follower default) is passed as ``mpc_decel_limit`` with
    the MPC layer on; ``lead_in`` / ``final_len`` / ``turn``: demo-course geometry; ``mu_lat`` /
    ``mu_long`` (None = the preset's) are applied to every wheel before ``build()``."""
    import genesis as gs
    import genesis_vehicle as gv
    from genesis_vehicle import DifferentiablePlant, PathFollower, VehicleScene, presets
    from genesis_vehicle.control import extract_state

    v = COURSE_V[course] if v is None else float(v)
    if course == "demo":
        path, P = demo_course(v, ramp, radius, lead_in=lead_in, final_len=final_len, turn=turn)
    elif course == "cusp":
        path, P = cusp_course(v_fwd=v, ramp=ramp)
    elif course == "misaligned":
        path, P = misaligned_course(v)
    elif course == "vertex":
        path, P = vertex_course(vertex_deg, v)
    elif course == "behind":
        path, P = behind_course(behind_len, v)
    else:
        raise ValueError(f"unknown course {course!r}")
    spawn = MISALIGNED_CORNERS[0] if course == "misaligned" else (0.0, 0.0)
    yaw0 = 0.0 if start_yaw is None else float(start_yaw)
    quat = (math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2))
    VehicleScene.init_backend("gpu" if gpu else "cpu")
    vs = VehicleScene(dt=DT, substeps=SUBSTEPS, n_envs=1, raycast_mode="dual_scene",
                      show_viewer=viewer, solver="per_vehicle",
                      rigid_options=gs.options.RigidOptions(enable_collision=True))
    vs.add_ground_plane(friction=1.0)
    veh = vs.add_vehicle(urdf, getattr(presets, preset), pos=(spawn[0], spawn[1], 0.5),
                         quat=quat, material=gs.materials.Rigid(friction=1.0))
    if mu_lat is not None or mu_long is not None:
        # (v2.1.0) tyre mu on every wheel, before build (the identify / sweep_measure mechanism)
        from genesis_vehicle.control.sweep_measure import apply_plant_overrides
        apply_plant_overrides(veh.cfg, urdf, mu_lat=mu_lat, mu_long=mu_long, log=log)
    vs.build()
    for _ in range(int(0.5 / DT)):                  # settle on the brake
        veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        vs.step()
    stage2 = table if plant == "table" else DifferentiablePlant(veh.physics)
    kw = {} if mpc_model == "off" else dict(mpc_params=params, mpc_impl=mpc_impl,
                                            on_solve_failure=on_solve_failure)
    if mpc_model != "off" and not align_gate:
        kw["mpc_align_gate"] = None
    if mpc_model != "off" and not stall_timeout:
        kw["mpc_stall_timeout"] = None
    if mpc_model != "off" and decel_limit is not None:
        kw["mpc_decel_limit"] = decel_limit
    if kturn:
        from genesis_vehicle.control import KTurnParams
        kw["mpc_kturn"] = KTurnParams(lateral_limit=lateral_limit, max_reverse=max_reverse)
    if mpc_model == "off" and command_mode == "gear" and plant == "table":
        # The PD follower gets no mpc_params, so its brake gain comes from the
        # same params file explicitly (a DifferentiablePlant has brake_gain()).
        from genesis_vehicle.control import SweepTable, load_mpc_params
        from genesis_vehicle.control.brake import brake_gain_from_params
        stage2 = SweepTable.load(table)
        kw["brake_gain"] = brake_gain_from_params(load_mpc_params(params), stage2.accel_at)
    t0 = time.perf_counter()
    fol = PathFollower(path, stage2, mpc_model=mpc_model, steer_cap=STEER_CAP,
                       command_mode=command_mode, **kw)
    gear_mode = fol.command_mode == "gear"
    ctor_ms = (time.perf_counter() - t0) * 1e3
    # (v2.1.0 instrumentation, read-only) the stage-2 steer before the steer_cap clip, recorded by
    # wrapping the plant's steer inversion on this instance (returned values unchanged).
    raw = {"steer": None}
    if getattr(fol.plant, "has_coupled_solve", False):
        _solve = fol.plant.solve

        def _rec_solve(*a, **k):
            out = _solve(*a, **k)
            raw["steer"] = float(out[1])
            return out
        fol.plant.solve = _rec_solve
    else:
        _steer_for = fol.plant.steer_for

        def _rec_steer(*a, **k):
            out = _steer_for(*a, **k)
            raw["steer"] = float(out)
            return out
        fol.plant.steer_for = _rec_steer
    ph = veh.physics
    _ax = ph.wheel_meta.axle_index.detach().cpu().numpy().reshape(-1)
    front_idx = np.where(_ax == _ax.min())[0]
    radius_w = ph.wheel_meta.radius.detach().cpu().numpy().reshape(-1)

    def front_roll_min(v_long):
        om = ph.omega.detach().cpu().numpy().reshape(-1)
        return float(min(om[i] * radius_w[i] for i in front_idx) / max(float(v_long), 0.1))
    tc, ts, xy, modes, rev_entry, solve_ms = [], [], [], {}, None, []
    rows, seq = [], []
    terminal_cmd = None
    v_done = None
    gate_switches, gate_steps = 0, []           # (v2.1.0, read-only) the decel gate's switch steps
    for k in range(int(t_max / DT)):
        st = extract_state(veh)
        eng = int(veh.gear[0]) if gear_mode else None
        gkw = {"gear": eng} if (gear_mode and gear_feedback) else {}
        raw["steer"] = None
        t1 = time.perf_counter()
        cmd = fol.step(st["pos_xy"], st["yaw"], st["v_long"], st["pitch"],
                       st["roll"], yaw_rate=st["yaw_rate"], v_lat=st["v_lat"],
                       t=k * DT, **gkw)
        tc.append((time.perf_counter() - t1) * 1e3)
        if gear_mode:
            thr, steer, brake, gear = cmd
        else:
            (thr, steer, brake), gear = cmd, None
        solved = fol._law is not None and fol._law.n_solve > len(solve_ms)
        if solved:
            solve_ms.append(fol._law.last_solve_ms)
        modes[fol.last_mode] = modes.get(fol.last_mode, 0) + 1
        if seq and seq[-1][0] == fol.last_mode:
            seq[-1][1] += 1
        else:
            seq.append([fol.last_mode, 1, round(k * DT, 3)])
        gate = (fol.mpc_stats or {}).get("decel_gate") if fol._law is not None else None
        decel_open = None if gate is None else gate["open"]
        if gate is not None and gate["switches"] > gate_switches:
            gate_switches = gate["switches"]
            gate_steps.append(dict(k=k, t=k * DT, open=decel_open,
                                   status=fol._law.last_status if solved else None))
        if log_path is not None:
            rows.append(dict(k=k, t=k * DT, x=float(st["pos_xy"][0]), y=float(st["pos_xy"][1]),
                             yaw=float(st["yaw"]), v=float(st["v_long"]),
                             regime=fol.regime, mode=fol.last_mode, thr=float(thr),
                             steer=float(steer), brake=float(brake), gear=gear, eng=eng,
                             status=fol._law.last_status if solved else None,
                             v_lat=float(st["v_lat"]), yaw_rate=float(st["yaw_rate"]),
                             steer_raw=raw["steer"], front_roll_min=front_roll_min(st["v_long"]),
                             decel_open=decel_open))
        if rev_entry is None and fol.last_mode == "DRV-1":
            rev_entry = k
        xy.append(st["pos_xy"])
        if fol.last_mode in PathFollower.TERMINAL_MODES:   # DONE / GOAL-MISSED / MPC-ABORT / GEAR-ABORT
            v_done = float(st["v_long"])
            ts.append(0.0)
            terminal_cmd = (thr, steer, brake, gear)
            break
        t2 = time.perf_counter()
        veh.set_inputs(throttle=float(thr), brake=float(brake), steer=float(steer),
                       **({"gear": gear} if gear is not None else {}))
        vs.step()
        ts.append((time.perf_counter() - t2) * 1e3)
    # v2.0.2: APPLY the terminal command (throttle 0, brake 1) until the car is
    # still or 2 s have passed (not part of the metrics below).
    post_v, post_s = None, 0.0
    post = {"max_abs_beta_v1": 0.0, "locked_steps": 0, "n_samples": 0}
    if terminal_cmd is not None:
        thr_t, steer_t, brake_t, gear_t = terminal_cmd
        veh.set_inputs(throttle=float(thr_t), brake=float(brake_t), steer=float(steer_t),
                       **({"gear": gear_t} if gear_t is not None else {}))
        for _ in range(int(2.0 / DT)):
            sp_ = extract_state(veh)
            if abs(sp_["v_long"]) < 0.01:
                break
            vs.step()
            post_s += DT
            sp_ = extract_state(veh)          # (v2.1.0, read-only) the window after the terminal command
            post["n_samples"] += 1
            if sp_["v_long"] > 1.0:
                post["max_abs_beta_v1"] = max(post["max_abs_beta_v1"],
                                              abs(math.atan2(sp_["v_lat"], sp_["v_long"])))
                post["locked_steps"] += int(front_roll_min(sp_["v_long"]) < 0.5)
        post_v = abs(extract_state(veh)["v_long"])
    rest_xy = [float(c) for c in extract_state(veh)["pos_xy"]]
    xy = np.array(xy)
    cte = np.hypot(P[None, :, 0] - xy[:, None, 0], P[None, :, 1] - xy[:, None, 1]).min(1)
    tcs, tss = np.array(tc[20:]), np.array(ts[20:])
    tot = tcs + tss
    done = fol.last_mode == "DONE"
    res = dict(
        mpc_model=mpc_model, mpc_impl=mpc_impl if mpc_model != "off" else None, plant=plant,
        course=course, on_solve_failure=on_solve_failure if mpc_model != "off" else None,
        done=bool(done), t_end=round(len(xy) * DT, 3), n_steps=len(xy),
        cte_mean=float(cte.mean()), cte_max=float(cte.max()),
        cte_max_xy=[float(v_) for v_ in xy[int(cte.argmax())]],
        ctrl_mean_ms=float(tcs.mean()), ctrl_p95_ms=float(np.percentile(tcs, 95)),
        ctrl_max_ms=float(tcs.max()), ctrl_over25_pct=float(100 * (tcs > 25).mean()),
        tot_over25_pct=float(100 * (tot > 25).mean()), sim_median_ms=float(np.median(tss)),
        ctor_ms=ctor_ms, modes=modes, reverse_entry_step=rev_entry,
        solve_n=len(solve_ms),
        solve_p50_ms=float(np.median(solve_ms[1:])) if len(solve_ms) > 1 else None,
        solve_p95_ms=float(np.percentile(solve_ms[1:], 95)) if len(solve_ms) > 1 else None,
        solve_max_ms=float(np.max(solve_ms[1:])) if len(solve_ms) > 1 else None,
        mpc_stats=fol.mpc_stats, sdk=gv.__version__, genesis=gs.__version__,
        end_mode=fol.last_mode, stop_reason=fol.stop_reason,
        command_mode=fol.command_mode, gear_feedback=bool(gear_feedback) if gear_mode else None,
        post_stop_v=post_v, post_stop_s=round(post_s, 3),
        mode_seq=seq[:60], mode_seq_len=len(seq), start_xy=[float(c) for c in xy[0]],
        end_xy=[float(c) for c in xy[-1]], start_yaw=yaw0,
        align_gate=bool(align_gate) if mpc_model != "off" else None,
        stall_timeout=bool(stall_timeout) if mpc_model != "off" else None,
        conditions=dict(dt=DT, substeps=SUBSTEPS, steer_cap=STEER_CAP, v=v, ramp=ramp,
                        radius=radius if course == "demo" else None,
                        vertex_deg=vertex_deg if course == "vertex" else None,
                        behind_len=behind_len if course == "behind" else None,
                        kturn=bool(kturn), lateral_limit=lateral_limit, max_reverse=max_reverse,
                        t_max=t_max, backend="gpu" if gpu else "cpu",
                        env_threads={k_: os.environ.get(k_) for k_ in
                                     ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}))
    # (v2.1.0) arm identity, asset hashes, mu provenance (C1, C12, PV-7mu), waypoint hash (C3)
    import hashlib

    def _sha(f):
        if not isinstance(f, (str, os.PathLike)) or not os.path.exists(f):
            return None
        with open(f, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    from genesis_vehicle.control import load_mpc_params
    rec = load_mpc_params(params).data if mpc_model != "off" or params is not None else {}
    tyre = rec.get("tyre") if isinstance(rec, dict) else None
    record_tyre_mu = (sorted({(float(a["mu_lat"]), float(a["mu_long"])) for a in tyre["axles"]})
                      if isinstance(tyre, dict) else None)
    live_mu = sorted({(float(w.mu_lat), float(w.mu_long)) for w in veh.physics.resolved.wheels})
    res.update(
        mpc_decel_limit=fol.mpc_decel_limit, table_sha256=_sha(table), params_sha256=_sha(params),
        v_done=v_done, rest_xy=rest_xy, post_terminal=post,
        path_sha256=hashlib.sha256(json.dumps([list(map(float, p_)) for p_ in path]).encode()).hexdigest(),
        live_mu=[list(m) for m in live_mu],
        record_tyre_mu=None if record_tyre_mu is None else [list(m) for m in record_tyre_mu],
        record_plant_overrides=rec.get("plant_overrides") if isinstance(rec, dict) else None,
        setup_ms=getattr(getattr(fol._law, "solver", None), "setup_ms", None) if fol._law is not None else None,
        decel_switch_steps=gate_steps,
        decel_first_open_k=next((g["k"] for g in gate_steps if g["open"]), None))
    res["conditions"].update(lead_in=lead_in if course == "demo" else None,
                             final_len=final_len if course == "demo" else None,
                             turn=turn if course == "demo" else None, mu_lat=mu_lat, mu_long=mu_long,
                             decel_limit=decel_limit)
    if record_tyre_mu is not None and record_tyre_mu != live_mu:
        log(f"WARNING record tyre mu != vehicle mu: record {record_tyre_mu} vs vehicle {live_mu} "
            "(the SweepTable path checks nothing; see the docs)")
    ok = done and (course != "demo" or res["cte_max"] < 1.0)
    res["pass"] = bool(ok)
    if log_path is not None:
        with open(log_path, "w") as fh:
            for r_ in rows:
                fh.write(json.dumps(r_) + "\n")
    log(f"\n{'PASS' if ok else 'FAIL'}: done={done} end mode {fol.last_mode}"
        + (f" ({fol.stop_reason})" if fol.stop_reason else "")
        + f" t={res['t_end']:.2f}s  [{fol.command_mode}] CTE mean/max "
        f"{res['cte_mean']:.3f}/{res['cte_max']:.3f} m  ctrl mean {res['ctrl_mean_ms']:.2f} ms, "
        f"control+sim > 25 ms on {res['tot_over25_pct']:.1f} % of steps  modes {modes}")
    if fol.mpc_stats is not None:
        log(f"MPC solves: {fol.mpc_stats['n_solve']} {fol.mpc_stats['status']}")
    if post_v is not None:
        log(f"terminal command applied: |v| = {post_v:.4f} m/s after {post_s:.2f} s")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urdf", default=URDF)
    ap.add_argument("--preset", default="car_4w_rwd_ackermann")
    ap.add_argument("--mpc-model", choices=("bicycle", "off"), default="bicycle")
    ap.add_argument("--decel-limit", choices=("throttle", "brake"), default=None,
                    help="v2.1.0 PathFollower(mpc_decel_limit=); absent = the follower default (throttle)")
    ap.add_argument("--mpc-impl", choices=("nonlinear", "linear"), default="nonlinear")
    ap.add_argument("--on-solve-failure", choices=("pd", "hold"), default="pd")
    ap.add_argument("--plant", choices=("table", "diff"), default="table")
    ap.add_argument("--course", choices=("demo", "cusp", "misaligned", "vertex", "behind"), default="demo")
    ap.add_argument("--behind-len", type=float, default=6.0,
                    help="'behind' course: path length behind the car (m)")
    ap.add_argument("--kturn", choices=("off", "on"), default="off",
                    help="v2.0.3 K-turn: on = mpc_kturn=KTurnParams() (MPC layer, gear mode)")
    ap.add_argument("--lateral-limit", type=float, default=None,
                    help="KTurnParams.lateral_limit (m); implies --kturn on")
    ap.add_argument("--max-reverse", type=float, default=None,
                    help="KTurnParams.max_reverse (m); implies --kturn on")
    ap.add_argument("--radius", type=float, default=30.0, help="demo course turn radius (m)")
    ap.add_argument("--lead-in", type=float, default=20.0, help="v2.1.0 demo course: first straight (m)")
    ap.add_argument("--final-len", type=float, default=15.0, help="v2.1.0 demo course: last straight (m)")
    ap.add_argument("--turn", choices=("left", "right"), default="left", help="v2.1.0 demo course turn direction")
    ap.add_argument("--mu-lat", type=float, default=None,
                    help="v2.1.0: every wheel's WheelConfig.mu_lat (needs --table and --params)")
    ap.add_argument("--mu-long", type=float, default=None,
                    help="v2.1.0: every wheel's WheelConfig.mu_long (needs --table and --params)")
    ap.add_argument("--vertex-deg", type=float, default=90.0, help="vertex course heading change")
    ap.add_argument("--start-yaw", type=float, default=None, help="spawn yaw (rad), default 0")
    ap.add_argument("--align-gate", choices=("on", "off"), default="on")
    ap.add_argument("--stall-timeout", choices=("on", "off"), default="on")
    ap.add_argument("--log", default=None, help="per-step JSON-lines log path")
    ap.add_argument("--command-mode", choices=("gear", "signed"), default="gear",
                    help="PathFollower command_mode (v2.0.2; default gear)")
    ap.add_argument("--gear-feedback", choices=("on", "off"), default="on",
                    help="pass the engaged gear to PathFollower.step(gear=) (gear mode)")
    ap.add_argument("--table", default=None, help="existing sweep CSV (skips the cache)")
    ap.add_argument("--params", default=None, help="existing MPC params JSON (skips the cache)")
    ap.add_argument("--remeasure", action="store_true")
    ap.add_argument("--v", type=float, default=None, help="course speed (default per course)")
    ap.add_argument("--ramp", type=float, default=1.0)
    ap.add_argument("--t-max", type=float, default=40.0,
                    help="run length (s); --course misaligned needs 60 (DONE at 41.775 s)")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    if a.kturn == "on" or a.lateral_limit is not None or a.max_reverse is not None:
        if a.mpc_model == "off" or a.command_mode == "signed" or a.align_gate == "off":
            ap.error("--kturn on / --lateral-limit / --max-reverse need the MPC layer in gear mode with the "
                     "heading gate (not --mpc-model off, not --command-mode signed, not --align-gate off)")
    if a.decel_limit is not None and a.mpc_model == "off":
        ap.error("--decel-limit is an MPC-layer option (PathFollower refuses mpc_decel_limit with "
                 "--mpc-model off, even 'throttle')")
    if a.decel_limit == "brake" and a.command_mode == "signed":
        ap.error("--decel-limit brake needs gear mode (not --command-mode signed)")
    if (a.mu_lat is not None or a.mu_long is not None) and (a.table is None or a.params is None):
        ap.error("--mu-lat / --mu-long need both --table and --params (assets measured at that mu; the "
                 "cache is keyed on URDF + preset only)")
    table, params = a.table, a.params
    if table is None or params is None:
        ct, cp = ensure_cache(a.urdf, a.preset, a.remeasure, a.gpu)
        table, params = table or ct, params or cp
    res = run(table=table, params=params, preset=a.preset, urdf=a.urdf, mpc_model=a.mpc_model,
              mpc_impl=a.mpc_impl, plant=a.plant, course=a.course, v=a.v, ramp=a.ramp,
              t_max=a.t_max, on_solve_failure=a.on_solve_failure, gpu=a.gpu, viewer=a.viewer,
              radius=a.radius, vertex_deg=a.vertex_deg, start_yaw=a.start_yaw,
              align_gate=a.align_gate == "on", stall_timeout=a.stall_timeout == "on",
              log_path=a.log, command_mode=a.command_mode,
              gear_feedback=a.gear_feedback == "on",
              kturn=(a.kturn == "on" or a.lateral_limit is not None or a.max_reverse is not None),
              lateral_limit=a.lateral_limit, max_reverse=a.max_reverse, behind_len=a.behind_len,
              decel_limit=a.decel_limit, lead_in=a.lead_in, final_len=a.final_len, turn=a.turn,
              mu_lat=a.mu_lat, mu_long=a.mu_long)
    print("MPC_RESULT " + json.dumps(res, default=str), flush=True)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
