"""multi_env_render.py — visualize ALL parallel envs at once in a grid.

Demonstrates Genesis's ``env_separate_rigid`` rendering mode. With
``n_envs > 1`` Genesis simulates parallel "universes" that physically
overlap (every env runs at the same world coordinates). For
visualization you can lay them out on a grid:

    scene.build(n_envs=N,
                env_spacing=(dx, dy),
                n_envs_per_row=K)
    vis_options = gs.options.VisOptions(env_separate_rigid=True, ...)

Each env is then rendered as if shifted to its own grid cell. A wide
overhead camera frames the whole grid so you can watch every parallel
rollout simultaneously — handy for sanity-checking MPPI candidate
trajectories or eyeballing RL diversity.

Run
---
    python -m genesis_vehicle.samples.multi_env_render
    python -m genesis_vehicle.samples.multi_env_render --n_envs 64 --duration 8
    python -m genesis_vehicle.samples.multi_env_render --n_envs 16 --spacing 8
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
import torch
import genesis as gs

from genesis_vehicle import (
    VehiclePhysics, VehicleInputs,
    car_4w_rwd_ackermann,
    add_vehicle,
    __version__ as sdk_version,
)

URDF_PATH = os.path.join(os.path.dirname(__file__), "urdf", "car_4w.urdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_envs",   type=int,   default=16,
                    help="Number of parallel envs (default 16, will be laid out in a grid).")
    ap.add_argument("--spacing",  type=float, default=10.0,
                    help="Grid cell spacing in meters between envs (default 10).")
    ap.add_argument("--per_row",  type=int,   default=None,
                    help="Envs per row in the grid (default sqrt(n_envs)).")
    ap.add_argument("--duration", type=float, default=8.0,
                    help="Driving duration in seconds (default 8).")
    ap.add_argument("--throttle", type=float, default=0.4,
                    help="Base throttle (per-env value is throttle * (0.5 + rand) (default 0.4).")
    ap.add_argument("--viewer", action="store_true",
                    help="Open Genesis's interactive viewer window in addition to the "
                         "offscreen camera render (otherwise the demo is headless).")
    args = ap.parse_args()

    n_envs = int(args.n_envs)
    per_row = args.per_row if args.per_row else max(1, int(round(math.sqrt(n_envs))))
    n_rows = math.ceil(n_envs / per_row)

    print(f"genesis_vehicle v{sdk_version}  |  multi_env_render")
    print(f"  n_envs   : {n_envs}   grid: {per_row} per row × {n_rows} rows")
    print(f"  spacing  : {args.spacing:.1f} m between envs")
    print(f"  drive    : throttle {args.throttle:.2f} × per-env scale, "
          f"{args.duration:.1f} s")

    gs.init(backend=gs.gpu, logging_level="warning")
    cfg = car_4w_rwd_ackermann(URDF_PATH, stability="control")

    # Camera framing — used both for the offscreen cam below and the
    # interactive viewer (when --viewer).
    grid_w = args.spacing * per_row
    grid_h = args.spacing * n_rows
    cam_h  = max(grid_w, grid_h) * 1.5
    viewer_opts = gs.options.ViewerOptions(
        res=(1280, 720),
        camera_pos=(0.0, 0.0, cam_h),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_up=(1.0, 0.0, 0.0),
        camera_fov=70,
    ) if args.viewer else None

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=cfg.dt, substeps=20),
        rigid_options=gs.options.RigidOptions(
            dt=cfg.dt, enable_collision=True,
            enable_self_collision=False, enable_joint_limit=True,
        ),
        viewer_options=viewer_opts,
        vis_options=gs.options.VisOptions(
            shadow=True, ambient_light=(0.40, 0.40, 0.40),
            background_color=(0.05, 0.07, 0.10),
            env_separate_rigid=True,        # ← visualization grid layout
        ),
        show_viewer=args.viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(pos=(0, 0, 0), plane_size=(args.spacing * per_row * 2,
                                                    args.spacing * n_rows * 2)),
        material=gs.materials.Rigid(friction=1.0),
    )
    car, sensor, _ = add_vehicle(
        scene, URDF_PATH, preset_fn=None, pos=(0.0, 0.0, 1.0),
        material=gs.materials.Rigid(friction=1.0),
    )

    # Overhead offscreen camera — image-tensor render for inspection / mp4.
    cam = scene.add_camera(
        res=(1920, 1080),
        pos=(0.0, 0.0, cam_h), lookat=(0.0, 0.0, 0.0),
        up=(1.0, 0.0, 0.0),
        fov=70, near=0.1, far=cam_h * 4, GUI=False,
    )

    # Build with env_spacing + n_envs_per_row → physics-irrelevant grid for vis.
    scene.build(
        n_envs=n_envs,
        env_spacing=(args.spacing, args.spacing),
        n_envs_per_row=per_row,
        center_envs_at_origin=True,
    )
    physics = VehiclePhysics(scene, car, sensor, cfg, n_envs=n_envs)
    device = car.get_pos().device

    DT = cfg.dt

    # Per-env random controls: throttle in [throttle*0.5, throttle*1.5],
    # steer in [-0.3, +0.3] — visible diversity across envs in the grid.
    g = torch.Generator(device=device).manual_seed(0)
    throttle = args.throttle * (0.5 + torch.rand(n_envs, generator=g, device=device))
    steer    = 0.6 * (torch.rand(n_envs, generator=g, device=device) - 0.5)
    brake    = torch.zeros(n_envs, device=device)
    drive    = VehicleInputs(throttle=throttle, brake=brake, steer=steer)

    # Settle (brake held, uniform).
    print(f"\n[settle 1.5 s]")
    settle = VehicleInputs(throttle=0.0, brake=1.0, steer=0.0)
    for _ in range(int(1.5 / DT)):
        physics.step(settle)
        scene.step()
        cam.render()

    # Drive.
    n_steps = int(args.duration / DT)
    print(f"[drive {n_steps} steps  (per-env throttle/steer randomized)]")
    render_every = max(1, int(0.04 / DT))    # ~25 fps render

    for step in range(n_steps):
        physics.step(drive)
        scene.step()
        if step % render_every == 0:
            cam.render()

    # Final spread — confirm envs diverged.
    p = car.get_pos().cpu().numpy()    # shape (n_envs, 3) in WORLD coords (no grid offset)
    v = car.get_vel().cpu().numpy()
    speed = np.linalg.norm(v[:, :2], axis=1)
    print(f"\n=== FINAL  (per-env spread in chassis-local world frame) ===")
    print(f"  pos x  : {p[:, 0].min():+.2f} .. {p[:, 0].max():+.2f}  "
          f"(range {p[:, 0].max() - p[:, 0].min():.2f} m)")
    print(f"  pos y  : {p[:, 1].min():+.2f} .. {p[:, 1].max():+.2f}  "
          f"(range {p[:, 1].max() - p[:, 1].min():.2f} m)")
    print(f"  speed  : {speed.min():.2f} .. {speed.max():.2f} m/s  "
          f"(mean {speed.mean():.2f})")
    print(f"\nNote: get_pos() returns the chassis-local world position WITHOUT the")
    print(f"      env_spacing offset (which is a visualization-only transform).")
    print(f"      The renderer adds the offset so all envs appear in their grid cell.")


if __name__ == "__main__":
    main()
