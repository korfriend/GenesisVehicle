"""fleet_follow_demo.py — K tanks path-following at once, autodiff vs lookup.

The two inverse plants side by side on the SAME fleet, so the trade they make
is visible rather than argued:

- ``jacobian`` — ONE :class:`DifferentiablePlant` covering every vehicle as a
  member, driven by :class:`FleetFollower`. No sweep measurement, no CSV. One
  unrolled batch per step for the whole fleet.
- ``sweep`` — ONE :class:`SweepTable`, shared by K independent
  :class:`PathFollower` s. The table is stateless, so a fleet of one kind
  needs exactly one of it; the inversion is per-vehicle but ~0.16 ms each.

    python -m genesis_vehicle.samples.fleet_follow_demo [--k 8]
        [--plant {both,jacobian,sweep}] [--viewer] [--gpu] [--mp4 [PATH]]

Each tank gets its own lane and its own S-curve, so no two vehicles ask the
plant for the same thing and a batched solve cannot look right by accident.

PASS = every vehicle finishes within 3 m of its goal.

What the comparison actually shows
----------------------------------
**Cost scales differently.** The jacobian plant pays per FLEET — one unrolled
batch however many vehicles — so its per-vehicle cost falls as the fleet
grows. The sweep table pays per vehicle, but so little (~0.4 ms) that it does
not matter at these sizes. What the table does not show is the large offline
measurement it already paid, and has to pay again on any change to the
vehicle: the bundled CSV is valid only for this exact (URDF, preset,
override) triple, and the ``TankTuning`` constants below reproduce the
overrides it was measured with. Edit any of them and the table is silently
stale while the jacobian plant stays correct.

**Tracking is course-dependent, and not in one plant's favour.** On this
continuous S-curve the sweep table holds a tighter line (mean cross-track
~0.34 m vs ~0.61 m at 4 tanks); on ``path_follow_demo``'s straights-and-
corners course the jacobian plant is the tighter of the two (0.370 m vs
0.384 m). Both stay far inside the 3 m criterion in both cases.

The gap here is NOT the plant missing its target: raising the outer heading
gain (``--k-w 3.0`` -> 0.695 m) and letting the inversion converge harder
(``--newton-iters 5`` -> 0.607 m) both leave it where it was, so it is
neither an outer-loop P limit nor an unconverged inverse. An accurate inverse
delivers exactly the `(a, omega_z)` the pursuit law asks for; whether that
keeps the vehicle ON the line is the pursuit law's business, and on sustained
curvature the measured table's own inaccuracy happens to bias it inward.
Worth understanding before either plant is tuned for a curvy course.
"""
from __future__ import annotations

# Bootstrap: allow `python path/to/this_file.py` (and `python -m ...`) both.
import sys, pathlib
_SDK_PARENT = str(pathlib.Path(__file__).resolve().parents[2])
if _SDK_PARENT not in sys.path:
    sys.path.insert(0, _SDK_PARENT)

import argparse
import math
import os
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(_HERE, "urdf", "tank_ray.urdf")
CSV = os.path.join(_HERE, "data", "tank_sweep_signed.csv")

from genesis_vehicle.samples.tank_tuning import TankTuning

LANE_SPACING = 14.0          # m between lanes — wide enough that tanks never meet
DT = 0.025
T_MAX = 60.0
GOAL_RADIUS = 3.0


def lane_path(k: int, n_lanes: int):
    """One lane's waypoints: a straight run with an S-curve, phase-shifted per
    lane so every vehicle is asking for a different yaw rate at any moment."""
    y0 = LANE_SPACING * (k - 0.5 * (n_lanes - 1))
    phase = 0.7 * k
    return [(1.0 * i,
             y0 + 3.5 * math.sin(0.13 * i + phase),
             0.0, 2.0)
            for i in range(61)]


def cross_track(xy, poly):
    """Distance from a point to the lane POLYLINE, not to its nearest waypoint.

    Waypoint distance has a floor set by the waypoint spacing — a vehicle
    exactly on a 1 m-spaced path still reads up to 0.5 m off — which would put
    a discretisation artefact straight into the comparison both plants are
    being judged on.
    """
    a = poly[:-1]
    b = poly[1:]
    ab = b - a
    ap = np.asarray(xy) - a
    denom = (ab * ab).sum(axis=1)
    t = np.clip((ap * ab).sum(axis=1) / np.maximum(denom, 1e-12), 0.0, 1.0)
    closest = a + t[:, None] * ab
    return float(np.hypot(*(np.asarray(xy) - closest).T).min())


def build_scene(args, n_lanes):
    import genesis as gs
    from genesis_vehicle import VehicleScene, tank_skid_belt

    VehicleScene.init_backend("gpu" if args.gpu else "cpu")
    span = LANE_SPACING * n_lanes
    vs = VehicleScene(
        dt=DT, substeps=4, n_envs=1, raycast_mode="dual_scene",
        show_viewer=args.viewer,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(30.0, -0.6 * span, 0.9 * span),
            camera_lookat=(30.0, 0.0, 0.0),
            camera_fov=50) if args.viewer else None,
        rigid_options=gs.options.RigidOptions(dt=DT, enable_collision=True),
        vis_options=gs.options.VisOptions(shadow=True),
    )
    vs.add_ground_plane(friction=1.0)

    paths = [lane_path(k, n_lanes) for k in range(n_lanes)]
    for k, path in enumerate(paths):
        # Goal marker per lane (non-physics visual).
        gx, gy = path[-1][0], path[-1][1]
        vs.add_dynamic(
            gs.morphs.Sphere(radius=0.6, pos=(gx, gy, 0.9), fixed=True,
                             collision=False),
            physics=False, wheel_raycast=False,
            surface=gs.surfaces.Default(color=(1.0, 0.2, 0.2, 1.0)),
            name=f"goal_{k}")

    cam = None
    if args.mp4:
        cam = vs.add_camera(res=(1280, 720),
                            pos=(30.0, -0.6 * span, 0.9 * span),
                            lookat=(30.0, 0.0, 0.0), up=(0, 0, 1), fov=50,
                            near=0.1, far=400.0, GUI=False, debug=True)

    tanks = []
    for k, path in enumerate(paths):
        t = vs.add_vehicle(URDF, tank_skid_belt,
                           pos=(path[0][0], path[0][1], 2.0),
                           material=gs.materials.Rigid(friction=1.0),
                           name=f"tank_{k}")
        TankTuning.apply_config(t.cfg)      # BEFORE build (batched solver bakes cfg)
        tanks.append(t)
    vs.build()
    for t in tanks:
        TankTuning.apply_resolved(t.resolved)

    # Lane polylines — cyan, drawn once (needs a built scene).
    try:
        for path in paths:
            for a, b in zip(path[:-1], path[1:]):
                vs.scene.draw_debug_line((a[0], a[1], 0.35), (b[0], b[1], 0.35),
                                         radius=0.03, color=(0.2, 0.7, 1.0, 0.8))
    except Exception as e:
        print(f"[note] lane polylines skipped (no visualizer context): {e}")

    for _ in range(int(0.6 / DT)):          # settle on the ground under brake
        for t in tanks:
            t.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        vs.step()
    return vs, tanks, paths, cam


def run(mode, args, n_lanes, recorder=None):
    """Drive the fleet with one plant kind. Returns a result dict."""
    from genesis_vehicle import (DifferentiablePlant, FleetFollower,
                                 PathFollower, SweepTable)
    from genesis_vehicle.control import extract_state

    vs, tanks, paths, cam = build_scene(args, n_lanes)

    tune = dict(k_w=args.k_w) if args.k_w is not None else {}
    if mode == "jacobian":
        plant = DifferentiablePlant(tanks, horizon=args.horizon,
                                    newton_iters=args.newton_iters)
        followers = [PathFollower(p, plant, **tune) for p in paths]
        driver = FleetFollower(followers, plant)
        print(f"  inverse plant: {plant}")
    else:
        table = SweepTable.load(CSV)        # stateless -> one table for the fleet
        followers = [PathFollower(p, table, **tune) for p in paths]
        driver = None
        print(f"  inverse plant: sweep table {os.path.basename(CSV)} "
              f"(shared by all {n_lanes} vehicles)")

    lanes_xy = [np.array([(w[0], w[1]) for w in p]) for p in paths]
    goals = [p[-1] for p in paths]
    devs = [[] for _ in range(n_lanes)]
    done_at = [None] * n_lanes
    t_ctrl = t_sim = 0.0
    n_steps = int(T_MAX / DT)
    REC_EVERY = 2

    for step in range(n_steps):
        states = [extract_state(t) for t in tanks]
        for k, st in enumerate(states):
            devs[k].append(cross_track(st["pos_xy"], lanes_xy[k]))

        t0 = time.perf_counter()
        if driver is not None:
            cmds = driver.step(states)
        else:
            cmds = [f.step(s["pos_xy"], s["yaw"], s["v_long"], s["pitch"], s["roll"])
                    for f, s in zip(followers, states)]
        t_ctrl += time.perf_counter() - t0

        for k, f in enumerate(followers):
            if f.last_mode == "DONE" and done_at[k] is None:
                done_at[k] = step * DT
        if all(d is not None for d in done_at):
            break

        for t, (thr, steer, brk) in zip(tanks, cmds):
            t.set_inputs(throttle=thr, brake=brk, steer=steer)
        t0 = time.perf_counter()
        vs.step()
        t_sim += time.perf_counter() - t0

        if recorder is not None and cam is not None and step % REC_EVERY == 0:
            n_done = sum(d is not None for d in done_at)
            recorder.add(cam.render()[0], lines=(
                f"[{mode}]  t={step * DT:6.2f}s   {n_done}/{n_lanes} arrived",
                f"control {t_ctrl / (step + 1) * 1e3:6.2f} ms/step   "
                f"sim {t_sim / (step + 1) * 1e3:6.2f} ms/step",
            ))
        if step % int(5.0 / DT) == 0:
            v = np.mean([abs(s["v_long"]) for s in states])
            print(f"    t={step * DT:5.1f}s  arrived {sum(d is not None for d in done_at)}"
                  f"/{n_lanes}  mean |v|={v:4.2f}  "
                  f"control {t_ctrl / (step + 1) * 1e3:6.2f} ms/step", flush=True)

    used = step + 1
    err = [math.hypot(float(t.get_pos()[0][0]) - g[0],
                      float(t.get_pos()[0][1]) - g[1])
           for t, g in zip(tanks, goals)]
    dev = np.array([np.mean(d) for d in devs])
    return {
        "mode": mode,
        "n": n_lanes,
        "arrived": sum(d is not None for d in done_at),
        "err": np.array(err),
        "dev_mean": float(dev.mean()),
        "dev_max": float(max(np.max(d) for d in devs)),
        "ctrl_ms": t_ctrl / used * 1e3,
        "sim_ms": t_sim / max(1, used - 1) * 1e3,
        "t_finish": max([d for d in done_at if d is not None], default=None),
    }


def report(results, n_lanes):
    print("\n" + "=" * 78)
    print(f" fleet path following — {n_lanes} tanks")
    print("=" * 78)
    print(f"{'plant':>10} {'arrived':>9} {'max err':>9} {'dev mean':>9} {'dev max':>9} "
          f"{'control':>12} {'per vehicle':>12}")
    for r in results:
        print(f"{r['mode']:>10} {r['arrived']:>4}/{r['n']:<4} "
              f"{r['err'].max():8.2f}m {r['dev_mean']:8.3f}m {r['dev_max']:8.3f}m "
              f"{r['ctrl_ms']:9.2f} ms {r['ctrl_ms'] / r['n']:9.2f} ms")
    if len(results) == 2:
        a, b = results
        print(f"\n  control cost ratio ({a['mode']} / {b['mode']}): "
              f"{a['ctrl_ms'] / b['ctrl_ms']:.1f}x  "
              f"({a['ctrl_ms'] / a['n']:.2f} vs {b['ctrl_ms'] / b['n']:.2f} ms per vehicle)")
        print(f"  mean cross-track: {a['mode']} {a['dev_mean']:.3f} m, "
              f"{b['mode']} {b['dev_mean']:.3f} m — course-dependent, see the "
              f"module docstring")
    print(f"\n  simulation itself: {results[0]['sim_ms']:.2f} ms/step for "
          f"{n_lanes} vehicles (L2 batched, unaffected by the plant choice)")
    ok = all(r["arrived"] == r["n"] and r["err"].max() < GOAL_RADIUS for r in results)
    print(f"\n  {'PASS' if ok else 'FAIL'} (criterion: every vehicle within "
          f"{GOAL_RADIUS} m of its goal)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8, help="number of tanks (lanes)")
    ap.add_argument("--plant", choices=("both", "jacobian", "sweep"), default="both",
                    help="which inverse plant to run; 'both' runs each on the "
                         "same fleet and compares")
    ap.add_argument("--horizon", type=int, default=4,
                    help="DifferentiablePlant unroll length in sim steps")
    ap.add_argument("--newton-iters", type=int, default=2, dest="newton_iters",
                    help="DifferentiablePlant chord iterations per solve")
    ap.add_argument("--k-w", type=float, default=None, dest="k_w",
                    help="PathFollower heading gain (default 1.5). The outer "
                         "loop is P-only, so this is what a standing "
                         "cross-track offset responds to")
    ap.add_argument("--viewer", action="store_true", help="show the Genesis viewer")
    ap.add_argument("--gpu", action="store_true", help="GPU physics backend")
    ap.add_argument("--mp4", nargs="?", const="fleet_follow_demo.mp4",
                    default=None, metavar="PATH",
                    help="record to an mp4 (works headless; needs opencv-python)")
    args = ap.parse_args()

    n_lanes = max(1, args.k)
    modes = ["jacobian", "sweep"] if args.plant == "both" else [args.plant]
    if args.viewer and len(modes) > 1:
        print("[note] --viewer with --plant both would open two scenes in turn; "
              "run one plant at a time to watch it.")

    recorder = None
    if args.mp4:
        from genesis_vehicle.samples import _hud
        recorder = _hud.Mp4Recorder(args.mp4, fps=1.0 / (2 * DT))

    results = []
    for mode in modes:
        print(f"\n--- {mode} ---")
        results.append(run(mode, args, n_lanes, recorder))
    if recorder is not None:
        recorder.close()

    return 0 if report(results, n_lanes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
