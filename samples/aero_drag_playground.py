"""aero_drag_playground.py — tune aerodynamic drag LIVE while a car drives.

Demonstrates that `ChassisConfig.drag_area` / `air_density` are **runtime**
parameters (v1.2.4): the pipeline reads `resolved.chassis` every step, so
mutating it mid-drive changes the physics on the next step. A small Tk control
panel gives you sliders for Cd·A and air density; the car holds full throttle so
you watch its top speed rise and fall as you drag the sliders.

    python -m genesis_vehicle.samples.aero_drag_playground [--viewer] [--gpu]
    python -m genesis_vehicle.samples.aero_drag_playground --sweep    # no GUI: scripted Cd*A sweep

The GUI needs Tk (`python3-tk`) and a display. Without either, or with
``--sweep``, the sample instead steps Cd·A through a scripted sequence and prints
the resulting top speed — the same point, headless.

Top speed here is set by drag AND the drive-omega cap together: with drag 0 the
car runs up to its cap; add drag and it settles lower where traction = drag.
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
import threading

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(_HERE, "urdf", "car_4w.urdf")


class DragControls:
    """Shared, thread-safe drag state written by the GUI (or the scripted
    sweep) and read by the drive loop each step."""

    def __init__(self, drag_area, air_density):
        self._lock = threading.Lock()
        self.drag_area = float(drag_area)
        self.air_density = float(air_density)

    def get(self):
        with self._lock:
            return self.drag_area, self.air_density

    def set(self, *, drag_area=None, air_density=None):
        with self._lock:
            if drag_area is not None:
                self.drag_area = float(drag_area)
            if air_density is not None:
                self.air_density = float(air_density)


def _try_start_gui(controls) -> bool:
    """Launch a Tk slider panel in a background thread. Returns False (no GUI)
    if Tk or a display is unavailable."""
    try:
        import tkinter as tk
    except Exception as e:
        print(f"[gui] Tk unavailable ({e}); falling back to scripted sweep.")
        return False

    def run():
        try:
            root = tk.Tk()
        except Exception as e:      # no display
            print(f"[gui] no display ({e}); falling back to scripted sweep.")
            return
        root.title("aero drag — live")
        tk.Label(root, text="Cd·A  (m²)").pack()
        a = tk.Scale(root, from_=0.0, to=15.0, resolution=0.1,
                     orient=tk.HORIZONTAL, length=320,
                     command=lambda v: controls.set(drag_area=float(v)))
        a.set(controls.drag_area); a.pack()
        tk.Label(root, text="air density  (kg/m³)").pack()
        rho = tk.Scale(root, from_=0.0, to=3.0, resolution=0.05,
                       orient=tk.HORIZONTAL, length=320,
                       command=lambda v: controls.set(air_density=float(v)))
        rho.set(controls.air_density); rho.pack()
        tk.Label(root, text="drag F = ½·ρ·CdA·v²  (0 = off)").pack()
        root.mainloop()

    threading.Thread(target=run, daemon=True).start()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true", help="show the Genesis viewer")
    ap.add_argument("--gpu", action="store_true", help="GPU backend")
    ap.add_argument("--sweep", action="store_true",
                    help="skip the GUI; step Cd·A through a scripted sequence")
    args = ap.parse_args()

    import torch
    import genesis as gs
    from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann

    VehicleScene.init_backend("gpu" if args.gpu else "cpu")
    DT = 0.025
    vs = VehicleScene(
        dt=DT, substeps=4, n_envs=1, raycast_mode="dual_scene",
        show_viewer=args.viewer,
        viewer_options=(gs.options.ViewerOptions(
            camera_pos=(-8.0, -12.0, 6.0), camera_lookat=(10.0, 0.0, 0.0),
            camera_fov=55) if args.viewer else None),
        rigid_options=gs.options.RigidOptions(dt=DT, enable_collision=True),
    )
    vs.add_ground_plane(friction=1.0)
    # Start with drag off; the car preset's own default (0.66) is replaced so
    # the playground starts from a clean baseline.
    veh = vs.add_vehicle(URDF, lambda u, **kw: car_4w_rwd_ackermann(u, **kw, drag_area=0.0),
                         pos=(0, 0, 0.5), material=gs.materials.Rigid(friction=1.0),
                         name="car")
    vs.build()
    # veh.resolved is the live config the pipeline reads each step (veh.physics
    # is None under the default batched solver). Mutating its chassis is what
    # makes drag runtime-tunable.
    ch = veh.resolved.chassis

    controls = DragControls(drag_area=0.0, air_density=ch.air_density)
    use_gui = (not args.sweep) and _try_start_gui(controls)

    # A scripted Cd*A sequence for the headless / no-GUI path.
    sweep_seq = [(0.0, 6.0), (1.0, 6.0), (3.0, 6.0), (8.0, 6.0), (0.0, 6.0)]
    seq_i, seq_t0 = 0, 0.0

    print(f"[drive] car at full throttle; "
          f"{'drag the sliders' if use_gui else 'scripted Cd·A sweep'} to change drag.")
    T_MAX = int((25.0 if use_gui else len(sweep_seq) * 6.0) / DT)
    last_print = -1.0
    for step in range(T_MAX):
        t = step * DT
        if not use_gui:                    # advance the scripted sequence
            cda, secs = sweep_seq[seq_i]
            controls.set(drag_area=cda)
            if t - seq_t0 >= secs:
                seq_i += 1; seq_t0 = t
                if seq_i >= len(sweep_seq):
                    break

        drag_area, air_density = controls.get()
        ch.drag_area = drag_area           # <-- RUNTIME mutation, read next step
        ch.air_density = air_density

        veh.set_inputs(throttle=1.0, brake=0.0, steer=0.0)
        vs.step()

        if t - last_print >= 1.0:
            last_print = t
            v = float(np.linalg.norm(
                veh.entity_main.get_vel().cpu().numpy().reshape(-1)[:2]))
            print(f"  t={t:5.1f}s  Cd·A={drag_area:5.2f}  "
                  f"v={v:6.2f} m/s ({v * 3.6:5.1f} km/h)")

    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
