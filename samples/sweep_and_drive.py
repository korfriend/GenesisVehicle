"""sweep_and_drive.py — measure a sweep table for a URDF, then drive it.

The full offline-to-online loop in one sample: (1) MEASURE a sweep table for the
chosen vehicle's URDF with the current SDK preset (cached to
``data/<vehicle>_sweep_generated.csv``), then (2) DRIVE a waypoint path with
:class:`PathFollower` inverting that table each step.

    python -m genesis_vehicle.samples.sweep_and_drive --vehicle car  [--viewer]
    python -m genesis_vehicle.samples.sweep_and_drive --vehicle tank [--viewer]
    python -m genesis_vehicle.samples.sweep_and_drive --vehicle car  --remeasure --gpu

Measuring is a one-time ~batched run (a few minutes on GPU, longer on CPU) and
is CACHED — subsequent drives reuse the CSV. Pass ``--remeasure`` to force a new
measurement (e.g. after you change the preset, URDF or top speed).

Why one sample and not two: the table and the plant it is measured against MUST
match (URDF, preset, dt/substeps, top speed) — keeping measure + drive in one
place is what guarantees they stay in sync. Measuring is cached so it does not
re-run on every drive.

PASS criterion: final position within 3 m of the goal waypoint.
"""
from __future__ import annotations

import sys
import pathlib
_SDK_PARENT = str(pathlib.Path(__file__).resolve().parents[2])
if _SDK_PARENT not in sys.path:
    sys.path.insert(0, _SDK_PARENT)

import argparse
import math
import os
import subprocess

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# (urdf, preset name, following speed m/s, measurement top speed m/s)
VEHICLES = {
    "car":  (os.path.join(_HERE, "urdf", "car_4w.urdf"),  "car_4w_rwd_ackermann", 3.0, 55.6),
    "tank": (os.path.join(_HERE, "urdf", "tank_ray.urdf"), "tank_skid_belt",       2.0, 18.6),
}


def densify(corners, spacing=0.5):
    path = []
    for (x0, y0, s0), (x1, y1, _s1) in zip(corners[:-1], corners[1:]):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / spacing))
        for k in range(n):
            t = k / n
            path.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0), 0.0, s0))
    path.append((corners[-1][0], corners[-1][1], 0.0, corners[-1][2]))
    return path


def build_path(v):
    """A simple three-leg dogleg on open ground."""
    return densify([
        (0.0,   0.0, v),
        (20.0,  0.0, v),
        (30.0, 12.0, v),
        (45.0, 12.0, v),
    ])


def measure_table(urdf, preset, top_speed, csv_path, gpu, quick):
    """Run the sweep_measure CLI as a subprocess (own Genesis backend), caching
    the result. Kept out-of-process so measuring and driving don't share one
    Genesis init."""
    cmd = [sys.executable, "-m", "genesis_vehicle.control.sweep_measure",
           "--urdf", urdf, "--preset", preset,
           "--top-speed", str(top_speed), "--output", csv_path]
    if gpu:
        cmd.append("--gpu")
    if quick:
        cmd.append("--quick")
    print(f"[measure] {' '.join(cmd)}")
    print("[measure] one-time; cached to", os.path.basename(csv_path),
          "(re-run with --remeasure to refresh)")
    subprocess.run(cmd, check=True, cwd=_SDK_PARENT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle", choices=list(VEHICLES), default="car")
    ap.add_argument("--viewer", action="store_true", help="show the Genesis viewer")
    ap.add_argument("--gpu", action="store_true", help="GPU backend (measure + drive)")
    ap.add_argument("--remeasure", action="store_true",
                    help="force a fresh measurement even if the CSV exists")
    ap.add_argument("--quick", action="store_true",
                    help="measure a coarse smoke grid (fast; not for real control)")
    args = ap.parse_args()

    urdf, preset_name, v_follow, v_top = VEHICLES[args.vehicle]
    csv_path = os.path.join(_HERE, "data", f"{args.vehicle}_sweep_generated.csv")

    # ---- Phase 1: measure (cached) ------------------------------------------
    if args.remeasure or not os.path.exists(csv_path):
        measure_table(urdf, preset_name, v_top, csv_path, args.gpu, args.quick)
    else:
        print(f"[measure] reusing cached table {os.path.basename(csv_path)} "
              f"(--remeasure to refresh)")

    # ---- Phase 2: drive it --------------------------------------------------
    import torch
    import genesis as gs
    from genesis_vehicle import VehicleScene, PathFollower
    from genesis_vehicle import presets as _presets
    from genesis_vehicle.control import extract_state

    preset_fn = getattr(_presets, preset_name)
    VehicleScene.init_backend("gpu" if args.gpu else "cpu")

    DT = 0.025
    path = build_path(v_follow)
    start_xy, goal_xy = (path[0][0], path[0][1]), (path[-1][0], path[-1][1])
    print(f"[drive] {args.vehicle}: {len(path)} waypoints, "
          f"start={start_xy} goal={goal_xy}, table={os.path.basename(csv_path)}")

    vs = VehicleScene(
        dt=DT, substeps=4, n_envs=1, raycast_mode="dual_scene",
        show_viewer=args.viewer,
        viewer_options=(gs.options.ViewerOptions(
            camera_pos=(20.0, -30.0, 25.0), camera_lookat=(22.0, 6.0, 0.0),
            camera_fov=50) if args.viewer else None),
        rigid_options=gs.options.RigidOptions(dt=DT, enable_collision=True),
    )
    vs.add_ground_plane(friction=1.0)
    for i, wp in enumerate(path[::3]):
        vs.add_dynamic(gs.morphs.Sphere(radius=0.15, pos=(wp[0], wp[1], 0.3),
                       fixed=True, collision=False), physics=False,
                       wheel_raycast=False,
                       surface=gs.surfaces.Default(color=(0.2, 0.7, 1.0, 0.9)),
                       name=f"wp_{i}")

    veh = vs.add_vehicle(urdf, preset_fn, pos=(start_xy[0], start_xy[1], 1.0),
                         material=gs.materials.Rigid(friction=1.0), name=args.vehicle)
    vs.build()

    follower = PathFollower(path, csv_path)
    T_MAX = int(90.0 / DT)
    goal_r = 3.0
    for step in range(T_MAX):
        st = extract_state(veh)
        thr, steer, brake = follower.step(
            (st["pos_xy"][0], st["pos_xy"][1]), st["yaw"], st["v_long"],
            st.get("pitch_deg", 0.0), st.get("roll_deg", 0.0))
        veh.set_inputs(throttle=thr, brake=brake, steer=steer)
        vs.step()
        if follower.last_mode == "DONE":
            break
        if step % 40 == 0:
            print(f"  t={step * DT:5.1f}s pos=({st['pos_xy'][0]:+6.1f},"
                  f"{st['pos_xy'][1]:+6.1f}) v={st['v_long']:+.2f} [{follower.last_mode}]")

    st = extract_state(veh)
    err = math.hypot(st["pos_xy"][0] - goal_xy[0], st["pos_xy"][1] - goal_xy[1])
    ok = err < goal_r
    print(f"\nFINAL pos=({st['pos_xy'][0]:+.2f},{st['pos_xy'][1]:+.2f}) "
          f"goal=({goal_xy[0]:+.2f},{goal_xy[1]:+.2f})  err={err:.2f} m  "
          f"{'PASS' if ok else 'FAIL'} (criterion: err < {goal_r} m)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
