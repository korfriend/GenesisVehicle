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

Courses (v2.0.1 adds the last two and ``--radius``): ``demo`` (the field
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

Gear mode (v2.0.2): ``--command-mode gear`` (default, also with ``--mpc-model off``)
drives the car with ``set_inputs(..., gear=)`` from the follower's ``GearCommand``
(decelerations by brake, reverse only on the cusp course's reverse block);
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


def demo_course(v=8.0, ramp=1.0, radius=30.0):
    """Field demo course (dense 0.1 m line -> 0.5 m waypoints): straight 20 m,
    a 90 degree left turn of ``radius`` (field: 30 m), straight 15 m. Returns
    ``(path, dense polyline (n, 2))``."""
    pts = [(x, 0.0) for x in np.arange(0.0, 20.0, 0.1)]
    R = float(radius)
    for th in np.arange(0.0, math.pi / 2, 0.1 / R):
        pts.append((20.0 + R * math.sin(th), R - R * math.cos(th)))
    for d in np.arange(0.0, 15.0 + 1e-9, 0.1):
        pts.append((20.0 + R, R + d))
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


COURSE_V = {"demo": 8.0, "cusp": 4.0, "misaligned": 2.0, "vertex": 4.0}


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
        command_mode="gear", gear_feedback=True):
    """Drive one run; returns the result dict (also the ``MPC_RESULT`` record).

    ``command_mode`` ``"gear"`` (default) / ``"signed"`` is passed to
    ``PathFollower``; ``gear_feedback`` passes the engaged gear to
    ``PathFollower.step(gear=)`` in gear mode.

    ``v`` None = the course's speed (demo 8, cusp 4, misaligned 2, vertex 4 m/s).
    ``align_gate`` / ``stall_timeout`` False pass ``mpc_align_gate=None`` /
    ``mpc_stall_timeout=None`` (MPC only). ``log_path``: one JSON line per step."""
    import genesis as gs
    import genesis_vehicle as gv
    from genesis_vehicle import DifferentiablePlant, PathFollower, VehicleScene, presets
    from genesis_vehicle.control import extract_state

    v = COURSE_V[course] if v is None else float(v)
    if course == "demo":
        path, P = demo_course(v, ramp, radius)
    elif course == "cusp":
        path, P = cusp_course(v_fwd=v, ramp=ramp)
    elif course == "misaligned":
        path, P = misaligned_course(v)
    elif course == "vertex":
        path, P = vertex_course(vertex_deg, v)
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
    tc, ts, xy, modes, rev_entry, solve_ms = [], [], [], {}, None, []
    rows, seq = [], []
    terminal_cmd = None
    for k in range(int(t_max / DT)):
        st = extract_state(veh)
        eng = int(veh.gear[0]) if gear_mode else None
        gkw = {"gear": eng} if (gear_mode and gear_feedback) else {}
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
        if log_path is not None:
            rows.append(dict(k=k, t=k * DT, x=float(st["pos_xy"][0]), y=float(st["pos_xy"][1]),
                             yaw=float(st["yaw"]), v=float(st["v_long"]),
                             regime=fol.regime, mode=fol.last_mode, thr=float(thr),
                             steer=float(steer), brake=float(brake), gear=gear, eng=eng,
                             status=fol._law.last_status if solved else None))
        if rev_entry is None and fol.last_mode == "DRV-1":
            rev_entry = k
        xy.append(st["pos_xy"])
        if fol.last_mode in PathFollower.TERMINAL_MODES:   # DONE / GOAL-MISSED / MPC-ABORT / GEAR-ABORT
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
    if terminal_cmd is not None:
        thr_t, steer_t, brake_t, gear_t = terminal_cmd
        veh.set_inputs(throttle=float(thr_t), brake=float(brake_t), steer=float(steer_t),
                       **({"gear": gear_t} if gear_t is not None else {}))
        for _ in range(int(2.0 / DT)):
            if abs(extract_state(veh)["v_long"]) < 0.01:
                break
            vs.step()
            post_s += DT
        post_v = abs(extract_state(veh)["v_long"])
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
                        t_max=t_max, backend="gpu" if gpu else "cpu",
                        env_threads={k_: os.environ.get(k_) for k_ in
                                     ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}))
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
    ap.add_argument("--mpc-impl", choices=("nonlinear", "linear"), default="nonlinear")
    ap.add_argument("--on-solve-failure", choices=("pd", "hold"), default="pd")
    ap.add_argument("--plant", choices=("table", "diff"), default="table")
    ap.add_argument("--course", choices=("demo", "cusp", "misaligned", "vertex"), default="demo")
    ap.add_argument("--radius", type=float, default=30.0, help="demo course turn radius (m)")
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
              gear_feedback=a.gear_feedback == "on")
    print("MPC_RESULT " + json.dumps(res, default=str), flush=True)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
