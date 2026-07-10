"""path_follow_reverse_demo.py — reverse maneuver with explicit waypoint yaw.

The "back into a parking bay" scenario for the optional 5th waypoint
element (v1.1.14): drive PAST the bay, brake to a stop (cusp), then REVERSE
along a curved path into the bay — with the desired chassis heading (psi)
supplied explicitly on every reverse waypoint, so the vehicle arrives
facing bay-north, front-out.

    python -m genesis_vehicle.samples.path_follow_reverse_demo [--viewer] [--gpu] [--mp4 [PATH]]

Why explicit yaw here: on reverse waypoints the follower's default heading
is the densified segment tangent +pi — fine when the path is smooth, but a
parking planner KNOWS the headings it planned (and the arrival heading it
wants at the bay). Supplying them verbatim (world +X = 0, CCW positive)
removes the dependency on densification quality and pins the arrival
orientation: an explicit yaw on the cusp/goal waypoints also defines the
block's arrival heading.

PASS criteria: final position within 2 m of the bay AND final chassis
heading within 0.4 rad of bay-north (+pi/2).

The waypoint markers and connecting polyline are always drawn — forward
segments cyan, the reverse leg orange (colored by speed sign) — so the
maneuver reads at a glance in the viewer and in ``--mp4`` recordings.
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

import numpy as np

from genesis_vehicle.samples.path_follow_demo import densify
from genesis_vehicle.samples.tank_tuning import TankTuning

_HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(_HERE, "urdf", "tank_ray.urdf")
CSV = os.path.join(_HERE, "data", "tank_sweep_signed.csv")

BAY_XY = (6.0, -5.0)
BAY_YAW = math.pi / 2.0        # arrival heading: face bay-north, front-out


def build_path():
    """Forward past the bay, then a curved reverse leg into it.

    The forward leg is plain 4-tuples (tangential default). The reverse
    leg is a quadratic-bezier arc from the cusp into the bay, emitted as
    5-tuples whose explicit yaw is the PLANNED chassis heading: the arc
    tangent + pi (chassis faces away from travel when backing), which
    lands exactly on BAY_YAW at the bay.
    """
    fwd = densify([(0.0, 0.0, 0.0, +2.0), (12.0, 0.0, 0.0, +2.0)])

    # Reverse arc: cusp (12, 0) -> bay (6, -5); control point shapes the
    # curve so the final tangent is straight south (=> chassis north).
    p0, p1, p2 = (12.0, 0.0), (BAY_XY[0], -0.5), BAY_XY
    rev = []
    N = 24
    for k in range(N + 1):
        t = k / N
        x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
        y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
        dx = 2*(1-t)*(p1[0]-p0[0]) + 2*t*(p2[0]-p1[0])   # bezier tangent
        dy = 2*(1-t)*(p1[1]-p0[1]) + 2*t*(p2[1]-p1[1])
        psi = (math.atan2(dy, dx) % (2*math.pi)) - math.pi   # tangent + pi
        rev.append((x, y, 0.0, -1.0, psi))
    return fwd + rev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true", help="show the Genesis viewer")
    ap.add_argument("--gpu", action="store_true", help="GPU physics backend")
    ap.add_argument("--mp4", nargs="?", const="path_follow_reverse_demo.mp4",
                    default=None, metavar="PATH",
                    help="record the run to an mp4 (works headless; needs "
                         "opencv-python). Default file: "
                         "path_follow_reverse_demo.mp4")
    args = ap.parse_args()

    import torch
    import genesis as gs
    from genesis_vehicle import VehicleScene, tank_10w_skid_belt, PathFollower
    from genesis_vehicle.control import extract_state

    VehicleScene.init_backend("gpu" if args.gpu else "cpu")

    DT = 0.025
    vs = VehicleScene(
        dt=DT, substeps=10, n_envs=1, raycast_mode="dual_scene",
        show_viewer=args.viewer,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(6.0, -22.0, 14.0), camera_lookat=(6.0, -2.0, 0.0),
            camera_fov=50) if args.viewer else None,
        rigid_options=gs.options.RigidOptions(dt=DT, enable_collision=True),
        vis_options=gs.options.VisOptions(shadow=False),
    )
    vs.add_ground_plane(friction=1.0)

    path = build_path()
    print(f"path: {len(path)} waypoints  "
          f"(forward 4-tuples + reverse 5-tuples with explicit yaw)")
    print(f"bay: pos={BAY_XY}, arrival yaw={BAY_YAW:+.2f} rad (north)")

    # Waypoint markers + bay — always present (non-physics fixed visuals).
    for i, wp in enumerate(path[::3]):
        vs.add_dynamic(
            gs.morphs.Sphere(radius=0.15, pos=(wp[0], wp[1], 0.4),
                             fixed=True, collision=False),
            physics=False, wheel_raycast=False,
            surface=gs.surfaces.Default(color=(0.2, 0.7, 1.0, 0.9)),
            name=f"wp_{i}")
    vs.add_dynamic(   # bay marker
        gs.morphs.Sphere(radius=0.5, pos=(BAY_XY[0], BAY_XY[1], 0.7),
                         fixed=True, collision=False),
        physics=False, wheel_raycast=False,
        surface=gs.surfaces.Default(color=(1.0, 0.2, 0.2, 1.0)),
        name="bay")

    cam = None
    if args.mp4:        # offscreen camera, same pose as the viewer camera
        cam = vs.add_camera(res=(1280, 720), pos=(6.0, -22.0, 14.0),
                            lookat=(6.0, -2.0, 0.0), up=(0, 0, 1), fov=50,
                            near=0.1, far=200.0, GUI=False,
                            debug=True)   # render marker overlays (path polyline) too

    tank = vs.add_vehicle(URDF, tank_10w_skid_belt, pos=(0.0, 0.0, 2.0),
                          material=gs.materials.Rigid(friction=1.0), name="tank")
    TankTuning.apply_config(tank.cfg)   # BEFORE build (batched solver bakes cfg)
    vs.build()
    TankTuning.apply_resolved(tank.resolved)

    # Path polyline — drawn unconditionally (guarded like sample #13).
    try:
        for a, b in zip(path[:-1], path[1:]):
            fwd_seg = a[3] >= 0
            vs.scene.draw_debug_line(
                (a[0], a[1], 0.35), (b[0], b[1], 0.35), radius=0.03,
                color=(0.2, 0.7, 1.0, 0.8) if fwd_seg else (1.0, 0.6, 0.1, 0.8))
    except Exception as e:
        print(f"[note] path polyline skipped (no visualizer context): {e}")

    # Hold turret/barrel at zero so they don't droop while driving.
    ent = tank.entity_main
    try:
        idx = np.array([ent.get_joint("turret_yaw_joint").dofs_idx_local[0],
                        ent.get_joint("barrel_pitch_joint").dofs_idx_local[0]])
        ent.set_dofs_kp(kp=np.array([50000.0, 5000000.0]), dofs_idx_local=idx)
        ent.set_dofs_kv(kv=np.array([45000.0, 500000.0]), dofs_idx_local=idx)
        ent.control_dofs_position(
            torch.zeros((1, 2), device=gs.device, dtype=gs.tc_float),
            dofs_idx_local=idx)
    except Exception:
        pass

    follower = PathFollower(path, CSV)

    recorder = None
    REC_EVERY = 2                       # record every 2nd step -> 20 fps
    if args.mp4:
        from genesis_vehicle.samples import _hud
        recorder = _hud.Mp4Recorder(args.mp4, fps=1.0 / (REC_EVERY * DT))

    for _ in range(int(0.6 / DT)):     # settle on the ground under brake
        tank.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        vs.step()

    T_MAX = 90.0
    for step in range(int(T_MAX / DT)):
        st = extract_state(tank)
        thr, steer, brk = follower.step(
            st["pos_xy"], st["yaw"], st["v_long"], st["pitch"], st["roll"])
        if follower.last_mode == "DONE":
            print(f"\n[DONE] t={step * DT:.2f}s")
            break
        tank.set_inputs(throttle=thr, brake=brk, steer=steer)
        vs.step()
        if recorder is not None and step % REC_EVERY == 0:
            recorder.add(cam.render()[0], lines=(
                f"t={step * DT:6.2f}s  [{follower.last_mode}]",
                f"pos=({st['pos_xy'][0]:+6.1f},{st['pos_xy'][1]:+6.1f})  "
                f"yaw={st['yaw']:+5.2f}  v={st['v_long']:+5.2f}",
            ))
        if step % int(2.0 / DT) == 0:
            print(f"  t={step * DT:5.1f}s  pos=({st['pos_xy'][0]:+6.1f},"
                  f"{st['pos_xy'][1]:+6.1f})  yaw={st['yaw']:+5.2f}"
                  f"  v={st['v_long']:+5.2f}  thr={thr:+5.2f} str={steer:+5.2f}"
                  f"  [{follower.last_mode}]", flush=True)

    if recorder is not None:
        recorder.close()
    st = extract_state(tank)
    pos_err = math.hypot(st["pos_xy"][0] - BAY_XY[0], st["pos_xy"][1] - BAY_XY[1])
    yaw_err = abs((st["yaw"] - BAY_YAW + math.pi) % (2 * math.pi) - math.pi)
    ok = pos_err < 2.0 and yaw_err < 0.4
    print(f"\nFINAL pos=({st['pos_xy'][0]:+.2f}, {st['pos_xy'][1]:+.2f}) "
          f"yaw={st['yaw']:+.2f}  bay={BAY_XY} yaw*={BAY_YAW:+.2f}")
    print(f"pos_err={pos_err:.2f}m (<2)  yaw_err={yaw_err:.2f}rad (<0.4)  "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
