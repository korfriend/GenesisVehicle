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
cusp course: DONE within ``--t-max``). The last line is a ``MPC_RESULT {json}``
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


def demo_course(v=8.0, ramp=1.0):
    """Field demo course (dense 0.1 m line -> 0.5 m waypoints). Returns
    ``(path, dense polyline (n, 2))``."""
    pts = [(x, 0.0) for x in np.arange(0.0, 20.0, 0.1)]
    R = 30.0
    for th in np.arange(0.0, math.pi / 2, 0.1 / R):
        pts.append((20.0 + R * math.sin(th), R - R * math.cos(th)))
    for d in np.arange(0.0, 15.0 + 1e-9, 0.1):
        pts.append((50.0, 30.0 + d))
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
        mpc_impl="nonlinear", plant="table", course="demo", v=8.0, ramp=1.0, t_max=40.0,
        on_solve_failure="pd", gpu=False, viewer=False, log=print):
    """Drive one run; returns the result dict (also the ``MPC_RESULT`` record)."""
    import genesis as gs
    import genesis_vehicle as gv
    from genesis_vehicle import DifferentiablePlant, PathFollower, VehicleScene, presets
    from genesis_vehicle.control import extract_state

    path, P = demo_course(v, ramp) if course == "demo" else cusp_course(ramp=ramp)
    VehicleScene.init_backend("gpu" if gpu else "cpu")
    vs = VehicleScene(dt=DT, substeps=SUBSTEPS, n_envs=1, raycast_mode="dual_scene",
                      show_viewer=viewer, solver="per_vehicle",
                      rigid_options=gs.options.RigidOptions(enable_collision=True))
    vs.add_ground_plane(friction=1.0)
    veh = vs.add_vehicle(urdf, getattr(presets, preset), pos=(0.0, 0.0, 0.5),
                         material=gs.materials.Rigid(friction=1.0))
    vs.build()
    for _ in range(int(0.5 / DT)):                  # settle on the brake
        veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        vs.step()
    stage2 = table if plant == "table" else DifferentiablePlant(veh.physics)
    kw = {} if mpc_model == "off" else dict(mpc_params=params, mpc_impl=mpc_impl,
                                            on_solve_failure=on_solve_failure)
    t0 = time.perf_counter()
    fol = PathFollower(path, stage2, mpc_model=mpc_model, steer_cap=STEER_CAP, **kw)
    ctor_ms = (time.perf_counter() - t0) * 1e3
    tc, ts, xy, modes, rev_entry, solve_ms = [], [], [], {}, None, []
    for k in range(int(t_max / DT)):
        st = extract_state(veh)
        t1 = time.perf_counter()
        thr, steer, brake = fol.step(st["pos_xy"], st["yaw"], st["v_long"], st["pitch"],
                                     st["roll"], yaw_rate=st["yaw_rate"], v_lat=st["v_lat"],
                                     t=k * DT)
        tc.append((time.perf_counter() - t1) * 1e3)
        if fol._law is not None and fol._law.n_solve > len(solve_ms):
            solve_ms.append(fol._law.last_solve_ms)
        modes[fol.last_mode] = modes.get(fol.last_mode, 0) + 1
        if rev_entry is None and fol.last_mode == "DRV-1":
            rev_entry = k
        xy.append(st["pos_xy"])
        if fol.last_mode == "DONE":
            ts.append(0.0)
            break
        t2 = time.perf_counter()
        veh.set_inputs(throttle=float(thr), brake=float(brake), steer=float(steer))
        vs.step()
        ts.append((time.perf_counter() - t2) * 1e3)
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
        conditions=dict(dt=DT, substeps=SUBSTEPS, steer_cap=STEER_CAP, v=v, ramp=ramp,
                        t_max=t_max, backend="gpu" if gpu else "cpu",
                        env_threads={k_: os.environ.get(k_) for k_ in
                                     ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}))
    ok = done and (course != "demo" or res["cte_max"] < 1.0)
    res["pass"] = bool(ok)
    log(f"\n{'PASS' if ok else 'FAIL'}: done={done} t={res['t_end']:.2f}s  CTE mean/max "
        f"{res['cte_mean']:.3f}/{res['cte_max']:.3f} m  ctrl mean {res['ctrl_mean_ms']:.2f} ms, "
        f"control+sim > 25 ms on {res['tot_over25_pct']:.1f} % of steps  modes {modes}")
    if fol.mpc_stats is not None:
        log(f"MPC solves: {fol.mpc_stats['n_solve']} {fol.mpc_stats['status']}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urdf", default=URDF)
    ap.add_argument("--preset", default="car_4w_rwd_ackermann")
    ap.add_argument("--mpc-model", choices=("bicycle", "off"), default="bicycle")
    ap.add_argument("--mpc-impl", choices=("nonlinear", "linear"), default="nonlinear")
    ap.add_argument("--on-solve-failure", choices=("pd", "hold"), default="pd")
    ap.add_argument("--plant", choices=("table", "diff"), default="table")
    ap.add_argument("--course", choices=("demo", "cusp"), default="demo")
    ap.add_argument("--table", default=None, help="existing sweep CSV (skips the cache)")
    ap.add_argument("--params", default=None, help="existing MPC params JSON (skips the cache)")
    ap.add_argument("--remeasure", action="store_true")
    ap.add_argument("--v", type=float, default=8.0)
    ap.add_argument("--ramp", type=float, default=1.0)
    ap.add_argument("--t-max", type=float, default=40.0)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    table, params = a.table, a.params
    if table is None or params is None:
        ct, cp = ensure_cache(a.urdf, a.preset, a.remeasure, a.gpu)
        table, params = table or ct, params or cp
    res = run(table=table, params=params, preset=a.preset, urdf=a.urdf, mpc_model=a.mpc_model,
              mpc_impl=a.mpc_impl, plant=a.plant, course=a.course, v=a.v, ramp=a.ramp,
              t_max=a.t_max, on_solve_failure=a.on_solve_failure, gpu=a.gpu, viewer=a.viewer)
    print("MPC_RESULT " + json.dumps(res, default=str), flush=True)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
