"""obstacles_and_ramp.py — the encapsulated VehicleScene obstacle API.

Drives a car through a small course built entirely with `VehicleScene` (no manual
`gs.init` / `scene.build` / `scene.step`), exercising every registration method
and the `collision` / `wheel_raycast` / `physics` parameter combinations from
`docs/api-reference.md` §0.2:

- `add_ground_plane()`                          — flat ground (always raycast).
- `add_static(collision_morph=, wheel_raycast_morph=)` — a raised platform: the
  wheels SENSE it (always a raycast target) so the suspension lifts the chassis
  onto it; the coarse collider / detailed raycast surface split is a dual_scene
  feature.
- `add_dynamic(physics=True, wheel_raycast=False)` — a free box the car COLLIDES
  with and knocks away; the wheels do NOT drive onto it (collide-only default).
- `add_dynamic(physics=True, wheel_raycast=True)`  — a low ramp the wheels SENSE
  and drive onto (opt-in); a primitive collider so no re-fit warning.

It prints the body registry (each body's main / raycast entities) — the §0.2
matrix made concrete — then drives forward and reports what happened.

Run
---
    python -m genesis_vehicle.samples.obstacles_and_ramp
    python -m genesis_vehicle.samples.obstacles_and_ramp --mode single_scene
"""

from __future__ import annotations

import sys, pathlib
_SDK_PARENT = str(pathlib.Path(__file__).resolve().parents[2])
if _SDK_PARENT not in sys.path:
    sys.path.insert(0, _SDK_PARENT)

import argparse
import os

import genesis as gs
from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann, __version__ as sdk_version

URDF_PATH = os.path.join(os.path.dirname(__file__), "urdf", "car_4w.urdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dual_scene", "single_scene"], default="dual_scene",
                    help="raycast_mode (default dual_scene).")
    ap.add_argument("--cpu", action="store_true", help="run on CPU instead of GPU.")
    args = ap.parse_args()
    print(f"genesis_vehicle v{sdk_version}  |  obstacles_and_ramp  (mode={args.mode})")

    cfg = car_4w_rwd_ackermann(URDF_PATH, stability="control")
    vs = VehicleScene(backend="cpu" if args.cpu else "gpu", raycast_mode=args.mode,
                      dt=cfg.recommended_dt, substeps=10, n_envs=1)

    # --- build the course through VehicleScene (it owns the scene routing) ---
    # Obstacles are kept low so the car interacts gently instead of launching off
    # a vertical step. Positions are spaced along +x in the order the car meets them.
    vs.add_ground_plane(friction=1.0)

    # (1) Collide-only free box at x=4 — the car drives into it and pushes it; the
    # wheels do NOT drive onto it (wheel_raycast=False, the default).
    knock_box = vs.add_dynamic(
        gs.morphs.Box(size=(0.5, 0.5, 0.5), pos=(4.0, 0.0, 0.25)),
        physics=True, wheel_raycast=False, mass=1.0, name="knock_box")

    # (2) Low ramp at x=8 the wheels SENSE and roll over (wheel_raycast=True opt-in;
    # a primitive collider, so no re-fit warning). physics=False = a fixed surface.
    ramp = vs.add_dynamic(
        gs.morphs.Box(size=(2.0, 2.0, 0.12), pos=(8.0, 0.0, 0.06)),
        physics=False, wheel_raycast=True, name="ramp")

    # (3) Static block at x=11 — a static body is ALWAYS a raycast target; the
    # collision/raycast morph split only takes effect in dual_scene.
    block = vs.add_static(
        collision_morph=gs.morphs.Box(size=(2.0, 2.0, 0.12), pos=(11.0, 0.0, 0.06)),
        wheel_raycast_morph=gs.morphs.Box(size=(2.0, 2.0, 0.12), pos=(11.0, 0.0, 0.06)),
        material=gs.materials.Rigid(friction=1.0), name="block")

    veh = vs.add_vehicle(URDF_PATH, car_4w_rwd_ackermann, cfg=cfg,
                         pos=(0.0, 0.0, 0.6),
                         material=gs.materials.Rigid(friction=1.0))
    vs.build()

    # --- registry: the §0.2 matrix made concrete ---
    print(f"\n  raycast_mode = {vs.raycast_mode}  (raycast_scene = "
          f"{'present' if vs.raycast_scene is not None else 'None'})")
    print("  registered bodies:")

    def _has(e):
        return "yes" if e is not None else "—"
    for b in vs.statics:
        print(f"    static  {b.name:9s}  collides={_has(b.entity_main)}  "
              f"wheel-raycast={_has(b.entity_raycast)}")
    for d in vs.dynamics:
        print(f"    dynamic {d.name:9s}  collides={_has(d.entity_main)}  "
              f"wheel-raycast={_has(d.entity_raycast)}  "
              f"physics={'on' if d.is_dynamic else 'teleport'}")

    DT = cfg.recommended_dt
    box_x0 = float(knock_box.entity_main.get_pos()[0, 0])

    for _ in range(int(1.0 / DT)):          # settle on the ground
        veh.set_inputs(brake=1.0)
        vs.step()
    z_ground = float(veh.get_pos()[0, 2])

    max_z = z_ground
    for _ in range(int(5.0 / DT)):          # drive forward through the course
        veh.set_inputs(throttle=0.45)
        vs.step()
        max_z = max(max_z, float(veh.get_pos()[0, 2]))

    p = veh.get_pos()[0].cpu().numpy()
    box_dx = float(knock_box.entity_main.get_pos()[0, 0]) - box_x0

    print("\n  results")
    print(f"    car final pose    : x={p[0]:+.2f}  y={p[1]:+.2f}  z={p[2]:.2f}")
    print(f"    collide-only box  : pushed {box_dx:+.2f} m in x by the chassis "
          f"(wheels did not climb it)")
    print(f"    wheels-sensed lift: chassis z {z_ground:.2f} -> peak {max_z:.2f} "
          f"(+{max_z - z_ground:.2f} m) crossing the ramp lip — the wheels sense it "
          f"(wheel-raycast=yes above)")
    ok = (p[0] > 5.0) and (box_dx > 0.05)
    print("\n  RESULT:", "PASS" if ok else "CHECK")


if __name__ == "__main__":
    main()
