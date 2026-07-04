"""obstacles_and_ramp.py — the encapsulated VehicleScene obstacle API.

Drives a car through a small course built entirely with `VehicleScene` (no manual
`gs.init` / `scene.build` / `scene.step`), exercising every registration method
and the `collision` / `wheel_raycast` / `physics` parameter combinations from
`docs/api-reference.md` §0.2:

- `add_ground_plane()`                          — flat ground (always raycast).
- `add_static(collision_morph=, wheel_raycast_morph=)` — a static block: the
  wheels SENSE it (always a raycast target); the coarse-collider / detailed-
  raycast split is a dual_scene feature.
- `add_dynamic(physics=True, wheel_raycast=False)` — a free box the car COLLIDES
  with and knocks away; the wheels do NOT drive onto it (collide-only default).
- `add_dynamic(physics=True, wheel_raycast=True)`  — a low ramp the wheels SENSE
  and drive onto (opt-in); a primitive collider so no re-fit warning.

It prints the body registry (each body's main / raycast entities) — the §0.2
matrix made concrete — then drives forward and reports what happened.

`--bench` instead times single_scene vs dual_scene over the drive loop (optionally
swept across L3 `--n-envs`), to show how dual_scene pays off here. Spoiler: these
obstacles are PRIMITIVES (cheap BVH), so single_scene's per-step BVH re-fit is
nearly free and dual_scene's extra scene + synced ramp-mirror make it ~even-to-
slower at small n_envs — dual_scene's win needs a heavy *static mesh* terrain
and/or large `n_envs` (see `dual_scene_terrain.py`).

Run
---
    python -m genesis_vehicle.samples.obstacles_and_ramp
    python -m genesis_vehicle.samples.obstacles_and_ramp --mode single_scene
    python -m genesis_vehicle.samples.obstacles_and_ramp --bench
    python -m genesis_vehicle.samples.obstacles_and_ramp --bench --n-envs 64
"""

from __future__ import annotations

import sys, pathlib
_SDK_PARENT = str(pathlib.Path(__file__).resolve().parents[2])
if _SDK_PARENT not in sys.path:
    sys.path.insert(0, _SDK_PARENT)

import argparse
import os
import time

import genesis as gs
from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann, __version__ as sdk_version

URDF_PATH = os.path.join(os.path.dirname(__file__), "urdf", "car_4w.urdf")


def _build_course(mode, n_envs, backend, cfg):
    """Build the obstacle course on a fresh VehicleScene. Returns (vs, veh, box)."""
    VehicleScene.init_backend(backend)
    vs = VehicleScene(raycast_mode=mode, dt=cfg.recommended_dt,
                      substeps=10, n_envs=n_envs)
    vs.add_ground_plane(friction=1.0)
    # (1) collide-only free box at x=4 — pushed by the chassis, wheels don't climb it.
    knock_box = vs.add_dynamic(
        gs.morphs.Box(size=(0.5, 0.5, 0.5), pos=(4.0, 0.0, 0.25)),
        physics=True, wheel_raycast=False, mass=1.0, name="knock_box")
    # (2) low ramp at x=8 the wheels SENSE and roll over (wheel_raycast=True opt-in).
    vs.add_dynamic(
        gs.morphs.Box(size=(2.0, 2.0, 0.12), pos=(8.0, 0.0, 0.06)),
        physics=False, wheel_raycast=True, name="ramp")
    # (3) static block at x=11 — always a raycast target; collision/raycast split
    #     only takes effect in dual_scene.
    vs.add_static(
        collision_morph=gs.morphs.Box(size=(2.0, 2.0, 0.12), pos=(11.0, 0.0, 0.06)),
        wheel_raycast_morph=gs.morphs.Box(size=(2.0, 2.0, 0.12), pos=(11.0, 0.0, 0.06)),
        material=gs.materials.Rigid(friction=1.0), name="block")
    veh = vs.add_vehicle(URDF_PATH, car_4w_rwd_ackermann, cfg=cfg,
                         pos=(0.0, 0.0, 0.6), material=gs.materials.Rigid(friction=1.0))
    vs.build()
    return vs, veh, knock_box


def _bench(backend, n_envs, cfg, drive_s=3.0, settle_s=1.0):
    """Time the drive loop in each mode; return {mode: ms_per_step}."""
    import torch
    DT = cfg.recommended_dt
    out = {}
    for mode in ("single_scene", "dual_scene"):
        vs, veh, _ = _build_course(mode, n_envs, backend, cfg)
        veh.set_inputs(throttle=0.0, brake=1.0)
        for _ in range(int(settle_s / DT)):      # settle + warm up kernels
            vs.step()
        n = int(drive_s / DT)
        veh.set_inputs(throttle=0.45)
        if backend == "gpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            vs.step()
        if backend == "gpu":
            torch.cuda.synchronize()
        out[mode] = (time.perf_counter() - t0) / n * 1000.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dual_scene", "single_scene"], default="dual_scene",
                    help="raycast_mode for the educational run (default dual_scene).")
    ap.add_argument("--gpu", action="store_true",
                    help="opt into the GPU backend (default: CPU — faster at small n_envs).")
    ap.add_argument("--bench", action="store_true",
                    help="time single_scene vs dual_scene over the drive loop.")
    ap.add_argument("--n-envs", type=int, default=1,
                    help="L3 batch size for --bench (default 1).")
    args = ap.parse_args()
    backend = "gpu" if args.gpu else "cpu"
    cfg = car_4w_rwd_ackermann(URDF_PATH, stability="control")

    if args.bench:
        print(f"genesis_vehicle v{sdk_version}  |  obstacles_and_ramp  --bench  "
              f"(backend={backend}, n_envs={args.n_envs})")
        ms = _bench(backend, args.n_envs, cfg)
        s, d = ms["single_scene"], ms["dual_scene"]
        print(f"\n  {'mode':12s} {'ms/step':>9s} {'ms/env-step':>12s}")
        for mode, t in (("single_scene", s), ("dual_scene", d)):
            print(f"  {mode:12s} {t:9.3f} {t / args.n_envs:12.4f}")
        verdict = ("dual faster" if d < s else
                   "dual slower — primitive obstacles, no heavy static mesh to amortize")
        print(f"\n  dual_scene speedup vs single_scene: {s / d:.2f}x  ({verdict})")
        return

    print(f"genesis_vehicle v{sdk_version}  |  obstacles_and_ramp  (mode={args.mode})")
    vs, veh, knock_box = _build_course(args.mode, 1, backend, cfg)

    # --- registry: the §0.2 matrix made concrete ---
    print(f"\n  raycast_mode = {vs.raycast_mode}  (is_dual_scene = {vs.is_dual_scene})")
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
