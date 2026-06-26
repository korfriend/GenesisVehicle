"""quickstart.py — minimal genesis_vehicle hello-world (VehicleScene API).

Spawn a 4-wheel RWD car on flat ground, drive forward for 5 seconds with
throttle=0.5, print the final pose. About 40 lines of physics code.

What this demonstrates
----------------------
- The minimum-viable SDK call pattern through the unified ``VehicleScene`` entry
  point — no manual ``gs.init`` / ``gs.Scene`` / ``scene.build`` / ``scene.step``
  / ``sensor.read``:
  ``VehicleScene(...)`` → ``add_ground_plane`` → ``add_vehicle`` (preset) →
  ``build`` → loop of ``veh.set_inputs(...) + vs.step()``.
- Vehicle reaches a steady forward velocity once tire / drivetrain forces
  balance (no closed-loop controller, no scenarios — just open-loop throttle).

How to read this script
-----------------------
The pipeline is documented in detail in ``docs/pipeline-and-hooks.md``; here
you'll see only the top-level API surface, no per-wheel internals. For the
obstacle / ramp API (``add_static`` / ``add_dynamic``) see
``obstacles_and_ramp.py``; for heightfield terrain + the two-scene raycast see
``two_scene_terrain.py``.

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
    VehicleScene,
    car_4w_rwd_ackermann,
    __version__ as sdk_version,
)
from genesis_vehicle.samples import _hud

URDF_PATH = os.path.join(os.path.dirname(__file__), "urdf", "car_4w.urdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true",
                    help="cv2 HUD window each step (chase-cam-style). Needs opencv-python.")
    ap.add_argument("--native", action="store_true",
                    help="Genesis native interactive viewer (orbit/zoom/ESC) instead of cv2.")
    args = ap.parse_args()
    if args.native:
        args.viewer = False        # --native uses the Genesis viewer, not the cv2 HUD

    print(f"genesis_vehicle v{sdk_version}  |  quickstart"
          + ("  (native viewer)" if args.native
             else ("  (viewer ON)" if args.viewer else "")))

    cfg = car_4w_rwd_ackermann(URDF_PATH, stability="control")

    # VehicleScene owns gs.init + the scene(s) + build + step. Flat ground at
    # n_envs=1 → single_scene (the classic one-scene raycast; the dual_scene
    # optimization only pays off on heavy static terrain — see two_scene_terrain).
    VehicleScene.InitBackend("gpu")
    vs = VehicleScene(
        raycast_mode="single_scene",
        dt=cfg.recommended_dt, substeps=10,
        vis_options=gs.options.VisOptions(
            shadow=True, ambient_light=(0.40, 0.40, 0.40),
            background_color=(0.05, 0.07, 0.10)),
        viewer_options=(_hud.native_viewer_options((-8.0, -6.0, 4.0), (0.0, 0.0, 1.0))
                        if args.native else None),
        show_viewer=args.native,    # --viewer uses a cv2 HUD instead (see below)
    )
    if args.viewer and not _hud.have_cv2():
        print("WARN: --viewer needs opencv-python. Continuing headless.")
        args.viewer = False

    vs.add_ground_plane(friction=1.0)
    veh = vs.add_vehicle(URDF_PATH, car_4w_rwd_ackermann, cfg=cfg,
                         pos=(0.0, 0.0, 1.0),
                         material=gs.materials.Rigid(friction=1.0))

    # Camera only when --viewer. Adding a camera triggers per-step renderer-state
    # sync inside the step (slows pure-physics timing), so headless skips it and
    # reports physics-only numbers. vs.add_camera must be called before build().
    cam = None
    if args.viewer:
        cam = vs.add_camera(
            res=(1280, 720),
            pos=(-8.0, -6.0, 4.0), lookat=(0.0, 0.0, 1.0),
            up=(0.0, 0.0, 1.0), fov=55, near=0.1, far=200.0, GUI=False,
        )

    # VisualJointSync (Genesis-viewer wheel animation) is auto-managed by
    # VehicleScene at build() — it turns on iff the main scene is rendered by
    # Genesis (show_viewer or a Genesis camera, e.g. the --viewer cv2 HUD's).
    vs.build()

    DT = cfg.recommended_dt
    n_settle = int(1.5 / DT)
    n_drive  = int(5.0 / DT)
    render_every = max(1, int(0.04 / DT))    # ~25 fps HUD refresh
    hud_perf = _hud.PerfMeter(window=60)

    def _hud_render(t_sim: float, throttle: float):
        # Headless = pure physics (no render call in the timed loop).
        # Viewer = render + HUD overlay; cam pose follows the car.
        if args.native:                 # native viewer renders itself; just watch for close
            return _hud.native_alive(vs)
        if not args.viewer:
            return True
        p = veh.get_pos()[0].cpu().numpy()
        v = veh.get_vel()[0].cpu().numpy()
        speed = float((v[0] ** 2 + v[1] ** 2) ** 0.5)
        cam.set_pose(
            pos=p + np.array([-8.0, -6.0, 4.0]),
            lookat=p + np.array([0.0, 0.0, 1.0]),
            up=np.array([0.0, 0.0, 1.0]),
        )
        frame = _hud.render_hud_frame(
            cam,
            title=f"quickstart  v{sdk_version}",
            lines=[
                f"t = {t_sim:5.2f} s    throttle = {throttle:+.2f}",
                f"pos = ({p[0]:+6.2f}, {p[1]:+6.2f})    speed = {speed:5.2f} m/s",
                "[ESC] quit",
            ],
            perf_ms=hud_perf.ms_per_step(),
        )
        return _hud.cv2_show("genesis_vehicle quickstart", frame)

    # Phase 1 — settle (brake held while the car drops onto the ground).
    for step in range(n_settle):
        veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        vs.step()
        hud_perf.tick()
        if step % render_every == 0:
            if not _hud_render(step * DT, throttle=0.0):
                break

    # Phase 2 — open-loop forward throttle. Timed end-to-end with a single CUDA
    # sync before/after the loop (zero per-step overhead).
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    user_quit = False
    for step in range(n_drive):
        veh.set_inputs(throttle=0.5, brake=0.0, steer=0.0)
        vs.step()
        hud_perf.tick()
        if step % render_every == 0:
            if not _hud_render(1.5 + step * DT, throttle=0.5):
                user_quit = True
                break
    torch.cuda.synchronize()
    wall = time.perf_counter() - t_start
    _hud.cv2_cleanup()

    n_done = step + 1 if user_quit else n_drive
    p = veh.get_pos()[0].cpu().numpy()
    v = veh.get_vel()[0].cpu().numpy()

    # Separate render-only benchmark (viewer only — see comment near cam).
    r_ms, r_n = _hud.bench_render(cam, n=20) if cam is not None else (None, None)
    _hud.print_perf_summary(
        sample=f"quickstart  (v{sdk_version})",
        completed=not user_quit,
        n_done=n_done, n_target=n_drive, wall=wall,
        render_ms=r_ms, render_n=r_n,
        extra=[
            f"final pose : x={p[0]:+.2f}  y={p[1]:+.2f}  z={p[2]:.2f}",
            f"final speed: {(v[0]**2 + v[1]**2)**0.5:.2f} m/s",
        ],
    )

    if args.native:    # keep the interactive viewer open until closed/ESC
        print("\nviewer 유지 중 — 창 닫기(또는 ESC)로 종료.")
        try:
            while _hud.native_alive(vs):
                veh.set_inputs(throttle=0.0, brake=0.0, steer=0.0)
                vs.step()
        except gs.GenesisException:
            pass


if __name__ == "__main__":
    main()
