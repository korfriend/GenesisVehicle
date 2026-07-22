"""path_follow_demo.py — closed-loop path following with the control utility.

End-to-end demo of ``genesis_vehicle.control``: the bundled 10-wheel tank
(``urdf/tank_ray.urdf``) follows a waypoint path around a central wall,
driven each step by :class:`PathFollower` inverting the bundled reference
sweep table (``data/tank_sweep_signed.csv``).

    python -m genesis_vehicle.samples.path_follow_demo [--viewer] [--gpu] [--mp4 [PATH]]

PASS criterion: final position within 3 m of the goal waypoint.

The bundled sweep CSV was measured for EXACTLY this (URDF, preset,
override) triple — the ``TankTuning`` constants below reproduce the
overrides it was measured with. Change any of the three and you must
re-measure: ``python -m genesis_vehicle.control.sweep_measure --urdf ...
--preset tank_skid_belt --config your_overrides.py --output new.csv``.
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

_HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(_HERE, "urdf", "tank_ray.urdf")
CSV = os.path.join(_HERE, "data", "tank_sweep_signed.csv")


# The override set the bundled sweep table was measured with — shared with
# the sweep_measure CLI via samples/tank_tuning.py (usable as --config).
from genesis_vehicle.samples.tank_tuning import TankTuning


def densify(corners, spacing=0.5):
    """(x, y, z, target_speed) corner waypoints -> full waypoint list at
    ~``spacing`` m intervals (the follower wants 0.3-1 m spacing). Each
    segment inherits its START corner's speed; z is interpolated (the
    follower ignores it — kept for format symmetry)."""
    path = []
    for (x0, y0, z0, s0), (x1, y1, z1, _s1) in zip(corners[:-1], corners[1:]):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / spacing))
        for k in range(n):
            t = k / n
            path.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0),
                         z0 + t * (z1 - z0), s0))
    path.append(corners[-1])
    return path


def build_path():
    """Corners around the south side of the wall, +2 m/s throughout."""
    return densify([
        (30.0,   0.0, 0.0, +2.0),
        (10.0,  -8.0, 0.0, +2.0),
        (-10.0, -8.0, 0.0, +2.0),
        (-30.0,  0.0, 0.0, +2.0),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true", help="show the Genesis viewer")
    ap.add_argument("--gpu", action="store_true", help="GPU physics backend")
    ap.add_argument("--mp4", nargs="?", const="path_follow_demo.mp4",
                    default=None, metavar="PATH",
                    help="record the run to an mp4 (bird's-eye camera; works "
                         "headless; needs opencv-python). Default file: "
                         "path_follow_demo.mp4")
    args = ap.parse_args()

    import torch
    import genesis as gs
    from genesis_vehicle import VehicleScene, tank_skid_belt, PathFollower
    from genesis_vehicle.control import extract_state

    VehicleScene.init_backend("gpu" if args.gpu else "cpu")

    # SDK-recommended timing (presets' recommended_dt): 40 Hz, internal 2.5 ms.
    # The bundled sweep CSV is measured at this same dt/substeps — keep them
    # in sync (dt is part of the table's validity contract).
    DT = 0.025
    vs = VehicleScene(
        # substeps 4 = the SDK default, which is also what `sweep_measure` and
        # the OSC server default to (v1.2.1) — the sibling osc demo drives this
        # same table through the server, so all three have to agree.
        dt=DT, substeps=4, n_envs=1, raycast_mode="dual_scene",
        show_viewer=args.viewer,
        # Bird's-eye start pose framing the whole course (the default viewer
        # camera spawns at the origin — inside the wall/tank).
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -45.0, 32.0), camera_lookat=(0.0, -4.0, 0.0),
            camera_fov=50) if args.viewer else None,
        rigid_options=gs.options.RigidOptions(dt=DT, enable_collision=True),
        vis_options=gs.options.VisOptions(shadow=True),
    )
    vs.add_ground_plane(friction=1.0)

    # The obstacle the path detours around.
    wall_morph = gs.morphs.Box(size=(5, 8, 3), pos=(0, 0, 1.5), fixed=True)
    vs.add_static(collision_morph=wall_morph, wheel_raycast_morph=wall_morph,
                  material=gs.materials.Rigid(friction=1.0),
                  surface=gs.surfaces.Default(color=(0.5, 0.4, 0.3, 1.0)),
                  name="wall")

    path = build_path()
    start_xy = (path[0][0], path[0][1])
    goal_xy = (path[-1][0], path[-1][1])
    print(f"path: {len(path)} waypoints, start={start_xy} goal={goal_xy}")

    # Waypoint markers + goal — always present (non-physics fixed visuals;
    # negligible cost headless, visible in viewer/video runs).
    for i, wp in enumerate(path[::3]):
        vs.add_dynamic(
            gs.morphs.Sphere(radius=0.18, pos=(wp[0], wp[1], 0.4),
                             fixed=True, collision=False),
            physics=False, wheel_raycast=False,
            surface=gs.surfaces.Default(color=(0.2, 0.7, 1.0, 0.9)),
            name=f"wp_{i}")
    vs.add_dynamic(   # goal marker
        gs.morphs.Sphere(radius=0.6, pos=(goal_xy[0], goal_xy[1], 0.9),
                         fixed=True, collision=False),
        physics=False, wheel_raycast=False,
        surface=gs.surfaces.Default(color=(1.0, 0.2, 0.2, 1.0)),
        name="goal")

    cam = None
    if args.mp4:        # offscreen camera, same bird's-eye pose as the viewer
        cam = vs.add_camera(res=(1280, 720), pos=(0.0, -45.0, 32.0),
                            lookat=(0.0, -4.0, 0.0), up=(0, 0, 1), fov=50,
                            near=0.1, far=250.0, GUI=False,
                            debug=True)   # render marker overlays (path polyline) too

    tank = vs.add_vehicle(URDF, tank_skid_belt,
                          pos=(start_xy[0], start_xy[1], 2.0),
                          material=gs.materials.Rigid(friction=1.0), name="tank")
    # cfg overrides BEFORE build — with the default batched solver, cfg
    # mutations after build are ignored unless vs.mark_config_dirty() is
    # called (the resolved config is baked at build time).
    TankTuning.apply_config(tank.cfg)
    vs.build()

    # Waypoint-connecting polyline (debug overlay — needs a built scene,
    # hence after build). Forward segments cyan, backward segments orange
    # (color keyed on the segment's speed sign, so cusp paths read at a
    # glance). Drawn unconditionally; guarded in case a headless Genesis
    # build has no visualizer context for debug draws.
    try:
        for a, b in zip(path[:-1], path[1:]):
            fwd_seg = a[3] >= 0
            vs.scene.draw_debug_line(
                (a[0], a[1], 0.35), (b[0], b[1], 0.35), radius=0.03,
                color=(0.2, 0.7, 1.0, 0.8) if fwd_seg else (1.0, 0.6, 0.1, 0.8))
    except Exception as e:
        print(f"[note] path polyline skipped (no visualizer context): {e}")
    # Post-build overrides go on the RESOLVED config; tank.resolved works in
    # both solver modes (tank.physics is None under the batched solver).
    TankTuning.apply_resolved(tank.resolved)

    # Face west (the first path segment direction): yaw = pi.
    hp = math.pi / 2.0
    q = np.array([math.cos(hp), 0.0, 0.0, math.sin(hp)], dtype=np.float32)
    tank.entity_main.set_quat(
        torch.tensor(q, device=gs.device, dtype=gs.tc_float).unsqueeze(0))

    # Hold turret/barrel at zero so they don't droop while driving.
    ent = tank.entity_main
    try:
        turret_idx = ent.get_joint("turret_yaw_joint").dofs_idx_local[0]
        barrel_idx = ent.get_joint("barrel_pitch_joint").dofs_idx_local[0]
        idx = np.array([turret_idx, barrel_idx])
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
    n_steps = int(T_MAX / DT)
    done_t = None
    for step in range(n_steps):
        st = extract_state(tank)
        thr, steer, brk = follower.step(
            st["pos_xy"], st["yaw"], st["v_long"], st["pitch"], st["roll"])
        if follower.last_mode == "DONE":
            done_t = step * DT
            print(f"\n[DONE] t={done_t:.2f}s")
            break
        tank.set_inputs(throttle=thr, brake=brk, steer=steer)
        vs.step()
        if recorder is not None and step % REC_EVERY == 0:
            recorder.add(cam.render()[0], lines=(
                f"t={step * DT:6.2f}s  [{follower.last_mode}]",
                f"pos=({st['pos_xy'][0]:+6.1f},{st['pos_xy'][1]:+6.1f})  "
                f"v={st['v_long']:+5.2f}  thr={thr:+5.2f} str={steer:+5.2f}",
            ))
        if step % int(2.0 / DT) == 0:
            print(f"  t={step * DT:5.1f}s  pos=({st['pos_xy'][0]:+6.1f},"
                  f"{st['pos_xy'][1]:+6.1f})  v={st['v_long']:+5.2f}"
                  f"  thr={thr:+5.2f} str={steer:+5.2f}  [{follower.last_mode}]",
                  flush=True)

    if recorder is not None:
        recorder.close()
    final_pos = tank.get_pos()[0].cpu().numpy()
    err = math.hypot(final_pos[0] - goal_xy[0], final_pos[1] - goal_xy[1])
    print(f"\nFINAL pos=({final_pos[0]:+.2f}, {final_pos[1]:+.2f}) "
          f"goal=({goal_xy[0]:+.2f}, {goal_xy[1]:+.2f})  err={err:.2f}m  "
          f"{'PASS' if err < 3.0 else 'FAIL'} (criterion: err < 3m)")
    return 0 if err < 3.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
