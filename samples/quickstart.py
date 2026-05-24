"""quickstart.py — minimal genesis_vehicle hello-world.

Spawn a 4-wheel RWD car on flat ground, drive forward for 5 seconds with
throttle=0.5, print the final pose. About 40 lines of physics code.

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
    # or:
    python genesis_vehicle/samples/quickstart.py
"""

from __future__ import annotations

import os

import genesis as gs

from genesis_vehicle import (
    VehiclePhysics, VehicleInputs,
    car_4w_rwd_ackermann,
    add_vehicle,
    __version__ as sdk_version,
)

URDF_PATH = os.path.join(os.path.dirname(__file__), "urdf", "car_4w.urdf")


def main():
    print(f"genesis_vehicle v{sdk_version}  |  quickstart")

    cfg = car_4w_rwd_ackermann(URDF_PATH, stability="control")
    gs.init(backend=gs.gpu, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=cfg.dt, substeps=50),
        rigid_options=gs.options.RigidOptions(dt=cfg.dt, enable_collision=True),
        show_viewer=False,
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

    scene.build(n_envs=1)
    physics = VehiclePhysics(scene, car, sensor, cfg, n_envs=1)

    DT = cfg.dt
    n_settle = int(1.5 / DT)
    n_drive  = int(5.0 / DT)

    # Phase 1 — settle (brake held while the car drops onto the ground).
    for _ in range(n_settle):
        physics.step(VehicleInputs(throttle=0.0, brake=1.0, steer=0.0))
        scene.step()

    # Phase 2 — open-loop forward throttle.
    for _ in range(n_drive):
        physics.step(VehicleInputs(throttle=0.5, brake=0.0, steer=0.0))
        scene.step()

    p = car.get_pos()[0].cpu().numpy()
    v = car.get_vel()[0].cpu().numpy()
    print(f"\nFinal pose: x={p[0]:+.2f} y={p[1]:+.2f} z={p[2]:.2f}  "
          f"speed={(v[0]**2 + v[1]**2)**0.5:.2f} m/s")


if __name__ == "__main__":
    main()
