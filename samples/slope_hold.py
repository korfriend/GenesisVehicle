"""slope_hold.py — StaticFrictionLock side-slope hold regression check.

Drops a 4-wheel car onto a 20° side-tilted ground, holds brake=1.0 for
10 seconds, and verifies the chassis does NOT creep laterally.

Why this exists
---------------
v0.5.5 had a long-standing bug: ``StaticFrictionLock`` only wrote
``ctx.F_long``, so on a side slope ``ctx.F_lat`` came from Pacejka which
is ill-conditioned near rest (``alpha = atan2(v_lat, max(|v_long|,
eps_v))`` artificially compresses the slip angle). A brake-locked car
on a 20° side slope creeped downhill ~10 cm over 10 s. v0.5.7 rewrote
the lock as a 2D position-anchored stick-slip model. This script is the
regression check — under v0.5.7+ the slip should be < 1 cm.

If this script ever reports REGRESSION, look at:
  - genesis_vehicle/strategies/stability.py — StaticFrictionLock spring /
    damper / friction-ellipse projection
  - genesis_vehicle/tire_models/pacejka.py — eps_v clamp, ellipse
  - genesis_vehicle/core.py — ctx.dt wiring, hook ordering

Run
---
    python -m genesis_vehicle.samples.slope_hold
    python -m genesis_vehicle.samples.slope_hold --slope 30
    python -m genesis_vehicle.samples.slope_hold --flat   # baseline
    python -m genesis_vehicle.samples.slope_hold --slope 20 --viewer
"""

from __future__ import annotations

import argparse
import math
import os

import genesis as gs

from genesis_vehicle import (
    VehiclePhysics, VehicleInputs,
    car_4w_rwd_ackermann,
    add_vehicle,
    __version__ as sdk_version,
)

URDF_PATH = os.path.join(os.path.dirname(__file__), "urdf", "car_4w.urdf")
HOLD_OK_M = 0.01     # 1 cm threshold — stick-slip lock should beat this easily


def _quat_to_roll_deg(q) -> float:
    """Roll component of (w, x, y, z) quat in degrees."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.degrees(math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slope", type=float, default=20.0,
                    help="Slope angle in degrees (around world X-axis). Default 20.")
    ap.add_argument("--flat", action="store_true",
                    help="Override --slope, use 0° as a baseline.")
    ap.add_argument("--duration", type=float, default=10.0,
                    help="Brake-hold duration in seconds (default 10).")
    ap.add_argument("--viewer", action="store_true",
                    help="Render a side view of the slope + car each step.")
    args = ap.parse_args()
    slope_deg = 0.0 if args.flat else float(args.slope)

    print(f"genesis_vehicle v{sdk_version}  |  slope_hold  "
          f"slope={slope_deg:+.1f}°  hold={args.duration:.1f}s"
          + ("  (viewer ON)" if args.viewer else ""))

    cfg = car_4w_rwd_ackermann(URDF_PATH, stability="control")
    gs.init(backend=gs.gpu, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=cfg.dt, substeps=50),
        rigid_options=gs.options.RigidOptions(dt=cfg.dt, enable_collision=True),
        vis_options=gs.options.VisOptions(
            shadow=True, ambient_light=(0.40, 0.40, 0.40),
            background_color=(0.05, 0.07, 0.10)),
        show_viewer=False,
    )
    # Tilt the ground around world X-axis. fixed=True is critical — without
    # it the ground itself falls under gravity, putting the car in a
    # co-moving frame and removing the apparent lateral gravity.
    scene.add_entity(
        gs.morphs.Box(
            size=(60.0, 60.0, 0.1),
            pos=(0.0, 0.0, -0.05),
            euler=(slope_deg, 0.0, 0.0),
            fixed=True,
        ),
        material=gs.materials.Rigid(friction=1.0),
    )
    car, sensor, _ = add_vehicle(
        scene, URDF_PATH, preset_fn=None, pos=(0.0, 0.0, 1.0),
        material=gs.materials.Rigid(friction=1.0),
    )

    cam = None
    if args.viewer:
        # Side view that shows the slope tilt AND the car. Camera looks at
        # the world origin (where the car spawns) from the +X side at the
        # height of typical chassis (~1 m), so the slope is visible as a
        # tilted plane and lateral drift is visible end-on.
        cam = scene.add_camera(
            res=(1280, 720),
            pos=(15.0, 0.0, 6.0), lookat=(0.0, 0.0, 1.0),
            up=(0.0, 0.0, 1.0), fov=50, near=0.1, far=200.0, GUI=False,
        )

    scene.build(n_envs=1)
    physics = VehiclePhysics(scene, car, sensor, cfg, n_envs=1)

    DT = cfg.dt
    n_settle = int(3.0 / DT)
    n_hold   = int(args.duration / DT)
    render_every = max(1, int(0.04 / DT))

    inputs = VehicleInputs(throttle=0.0, brake=1.0, steer=0.0)

    for step in range(n_settle):
        physics.step(inputs)
        scene.step()
        if cam is not None and step % render_every == 0:
            cam.render()
    p0 = car.get_pos()[0].cpu().numpy()
    roll0 = _quat_to_roll_deg(car.get_quat()[0].cpu().numpy())
    y0 = float(p0[1])
    print(f"  settled: pos=({p0[0]:+.3f}, {p0[1]:+.3f}, {p0[2]:+.3f}) m   "
          f"roll={roll0:+.2f}°  (expect roll ≈ {-slope_deg:+.1f}° on slope)")

    for step in range(n_hold):
        physics.step(inputs)
        scene.step()
        if cam is not None and step % render_every == 0:
            cam.render()
    p1 = car.get_pos()[0].cpu().numpy()
    slip = float(p1[1]) - y0
    abs_slip_mm = abs(slip) * 1000.0

    print(f"  after hold: y={p1[1]:+.4f} m   lateral slip={slip*1000:+.1f} mm "
          f"over {args.duration:.1f}s")
    if abs_slip_mm < HOLD_OK_M * 1000:
        mu_needed = math.tan(math.radians(abs(slope_deg))) if slope_deg else 0
        print(f"  → OK  (under {HOLD_OK_M*1000:.0f} mm threshold; expected for "
              f"mu=1.0 > tan({slope_deg:.0f}°)={mu_needed:.3f})")
    else:
        print(f"  → REGRESSION  ({abs_slip_mm:.1f} mm > {HOLD_OK_M*1000:.0f} mm threshold)")
        print(f"      Likely cause: stability.py StaticFrictionLock anchor / spring-damper")
        print(f"      logic regression, or pacejka.py / core.py wiring change.")


if __name__ == "__main__":
    main()
