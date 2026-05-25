"""quickstart.py — minimal genesis_vehicle hello-world.

Spawn a 4-wheel RWD car on flat ground, drive forward for 5 seconds with
throttle=0.5, print the final pose. About 50 lines of physics code.

What this demonstrates
----------------------
- The minimum-viable SDK call pattern: ``car_4w_rwd_ackermann`` preset →
  ``add_vehicle`` helper → ``scene.build`` → loop of
  ``physics.step(VehicleInputs(...)) + scene.step()``.
- Vehicle reaches a steady forward velocity once tire / drivetrain
  forces balance (no closed-loop controller, no scenarios — just
  open-loop throttle).

How to read this script
-----------------------
The pipeline is documented in detail in ``docs/pipeline-and-hooks.md``;
here you'll see only the top-level API surface, no per-wheel internals.

Run
---
    python -m genesis_vehicle.samples.quickstart
    python -m genesis_vehicle.samples.quickstart --viewer
"""

from __future__ import annotations

# Bootstrap: allow `python path/to/this_file.py` (and `python -m ...`) both.
# Adds the SDK's parent directory to sys.path so `from genesis_vehicle import ...`
# resolves regardless of cwd.
import sys, pathlib
_SDK_PARENT = str(pathlib.Path(__file__).resolve().parents[2])
if _SDK_PARENT not in sys.path:
    sys.path.insert(0, _SDK_PARENT)

import argparse
import os
import time

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
    ap.add_argument("--viewer", action="store_true",
                    help="Render a side-view camera each step (chase-cam-style).")
    args = ap.parse_args()

    print(f"genesis_vehicle v{sdk_version}  |  quickstart"
          + ("  (viewer ON)" if args.viewer else ""))

    cfg = car_4w_rwd_ackermann(URDF_PATH, stability="control")
    gs.init(backend=gs.gpu, logging_level="warning")

    viewer_opts = gs.options.ViewerOptions(
        res=(1280, 720),
        camera_pos=(-8.0, -6.0, 4.0),
        camera_lookat=(0.0, 0.0, 1.0),
        camera_up=(0.0, 0.0, 1.0),
        camera_fov=55,
    ) if args.viewer else None
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=cfg.dt, substeps=50),
        rigid_options=gs.options.RigidOptions(dt=cfg.dt, enable_collision=True),
        viewer_options=viewer_opts,
        vis_options=gs.options.VisOptions(
            shadow=True, ambient_light=(0.40, 0.40, 0.40),
            background_color=(0.05, 0.07, 0.10)),
        show_viewer=args.viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(pos=(0, 0, 0)),
        material=gs.materials.Rigid(friction=1.0),
    )
    car, sensor, _ = add_vehicle(
        scene, URDF_PATH, preset_fn=None,
        pos=(0.0, 0.0, 1.0),
        material=gs.materials.Rigid(friction=1.0),
    )

    cam = None
    if args.viewer:
        # Side chase-cam — camera trails the car at a fixed offset behind +
        # to the side, looking forward along +X.
        cam = scene.add_camera(
            res=(1280, 720),
            pos=(-8.0, -6.0, 4.0), lookat=(0.0, 0.0, 1.0),
            up=(0.0, 0.0, 1.0), fov=55, near=0.1, far=200.0, GUI=False,
        )

    scene.build(n_envs=1)
    physics = VehiclePhysics(scene, car, sensor, cfg, n_envs=1)

    DT = cfg.dt
    n_settle = int(1.5 / DT)
    n_drive  = int(5.0 / DT)
    render_every = max(1, int(0.04 / DT))    # ~25 fps render

    def _render():
        if cam is None:
            return
        # Trail the car: offset (-8, -6, 4) from current chassis pos.
        p = car.get_pos()[0].cpu().numpy()
        cam.set_pose(
            pos=p + np.array([-8.0, -6.0, 4.0]),
            lookat=p + np.array([0.0, 0.0, 1.0]),
            up=np.array([0.0, 0.0, 1.0]),
        )
        cam.render()

    # Phase 1 — settle (brake held while the car drops onto the ground).
    for step in range(n_settle):
        physics.step(VehicleInputs(throttle=0.0, brake=1.0, steer=0.0))
        scene.step()
        if step % render_every == 0:
            _render()

    # Phase 2 — open-loop forward throttle. Timed end-to-end with a
    # single CUDA sync before/after the loop (zero per-step overhead).
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for step in range(n_drive):
        physics.step(VehicleInputs(throttle=0.5, brake=0.0, steer=0.0))
        scene.step()
        if step % render_every == 0:
            _render()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t_start

    p = car.get_pos()[0].cpu().numpy()
    v = car.get_vel()[0].cpu().numpy()
    print(f"\nFinal pose: x={p[0]:+.2f} y={p[1]:+.2f} z={p[2]:.2f}  "
          f"speed={(v[0]**2 + v[1]**2)**0.5:.2f} m/s")
    print(f"[timing] {n_drive} steps in {wall:.2f}s  "
          f"= {wall/n_drive*1000:.2f} ms/step  ({n_drive/wall:.0f} steps/s)")


if __name__ == "__main__":
    main()
