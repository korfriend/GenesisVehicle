"""two_scene_terrain.py — VehicleScene unified API + the ray-wheel raycast split.

Drives a 4-wheel car over a heightfield terrain using the high-level
``VehicleScene`` API (no manual ``gs.init`` / ``scene.build`` / ``scene.step``).
Demonstrates both raycast modes and (with ``--compare``) times them head to head.

What this demonstrates
----------------------
- ``VehicleScene`` as the single entry point: ``add_static`` +
  ``add_vehicle`` + ``build`` + a loop of ``veh.set_inputs(...) / vs.step()``.
- ``raycast_mode="dual_scene"`` (default): the terrain is raycast in a separate
  scene as a *kinematic* body (BVH built once, never re-fit, shared across batch
  envs) while collision/rollover run in the main scene with the terrain as a
  *rigid* body. The wheel distances are identical to ``"single_scene"`` mode; only the
  per-step raycast cost differs.

Why dual_scene helps (and when it does not)
-----------------------------------------
The ``"single_scene"`` (one-scene) wheel raycaster re-fits a BVH over every face each
step (the vehicle moves → the solver is non-static). ``"dual_scene"`` keeps the
terrain BVH static, so the *raycast* cost stops scaling with terrain face count
AND with ``n_envs`` (the static BVH is shared across envs). The win grows with
both: marginal at ``n_envs=1`` on GPU (~1x), large for batched rollouts
(~3.4x @256 envs). ``"single_scene"`` can be marginally faster only on small/flat
terrain at ``n_envs=1``. See ``docs/two-scene-raycast.md``.

Run
---
    python -m genesis_vehicle.samples.two_scene_terrain                 # dual_scene (default)
    python -m genesis_vehicle.samples.two_scene_terrain --mode single_scene
    python -m genesis_vehicle.samples.two_scene_terrain --compare           # dual_scene vs single_scene
    python -m genesis_vehicle.samples.two_scene_terrain --compare --n-envs 64  # L3: dual_scene pulls ahead
"""

from __future__ import annotations

import sys, pathlib
_SDK_PARENT = str(pathlib.Path(__file__).resolve().parents[2])
if _SDK_PARENT not in sys.path:
    sys.path.insert(0, _SDK_PARENT)

import argparse
import os
import time

import numpy as np
import genesis as gs

from genesis_vehicle import (
    VehicleScene, car_4w_rwd_ackermann, __version__ as sdk_version,
)

URDF_PATH = os.path.join(os.path.dirname(__file__), "urdf", "car_4w.urdf")


def _terrain(horizontal_scale: float):
    """A single flat subterrain. Bump face count with a smaller horizontal_scale
    (faces ~= 2*(size/horizontal_scale)^2) to see the split advantage grow."""
    return gs.morphs.Terrain(
        n_subterrains=(1, 1), subterrain_size=(40.0, 40.0),
        horizontal_scale=horizontal_scale, subterrain_types="flat_terrain")


def run(mode: str, backend: str, horizontal_scale: float, n_envs: int = 1,
        settle_s: float = 1.0, drive_s: float = 4.0, measure: bool = False):
    # SDK default timing (v1.0.19): 40 Hz (dt=0.025), substeps=10 → internal 2.5 ms.
    VehicleScene.init_backend(backend)
    vs = VehicleScene(raycast_mode=mode, dt=0.025,
                      substeps=10, n_envs=n_envs)
    vs.add_static(morph=_terrain(horizontal_scale))
    veh = vs.add_vehicle(URDF_PATH, car_4w_rwd_ackermann, pos=(0.0, 0.0, 3.0))
    vs.build()
    n_faces = int(vs.rigid_solver.faces_info.geom_idx.shape[0])

    for _ in range(int(settle_s / vs.dt)):
        veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        vs.step()

    ms = None
    n = int(drive_s / vs.dt)
    if measure:
        for _ in range(10):
            veh.set_inputs(throttle=0.6, steer=0.0); vs.step()
        t0 = time.perf_counter()
        for _ in range(n):
            veh.set_inputs(throttle=0.6, steer=0.0); vs.step()
        ms = (time.perf_counter() - t0) / n * 1e3
    else:
        for _ in range(n):
            veh.set_inputs(throttle=0.6, steer=0.0); vs.step()

    p = veh.get_pos()[0].cpu().numpy()
    v = veh.get_vel()[0].cpu().numpy()
    speed = float(np.linalg.norm(v[:2]))
    return dict(mode=mode, faces=n_faces, x=float(p[0]), z=float(p[2]),
                speed=speed, ms=ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dual_scene", "single_scene"], default="dual_scene")
    ap.add_argument("--gpu", action="store_true",
                    help="Opt into the GPU backend (default: CPU — faster at small n_envs).")
    ap.add_argument("--compare", action="store_true",
                    help="Run single AND split, print pose match + timing.")
    ap.add_argument("--horizontal-scale", type=float, default=0.25,
                    help="Terrain cell size; smaller = more faces (split wins more).")
    ap.add_argument("--n-envs", type=int, default=1,
                    help="L3 batch size. Split's win grows strongly with n_envs "
                         "(the static terrain BVH is shared across envs).")
    args = ap.parse_args()
    backend = "gpu" if args.gpu else "cpu"

    print(f"genesis_vehicle v{sdk_version}  |  two_scene_terrain  "
          f"(backend={backend}, horizontal_scale={args.horizontal_scale}, "
          f"n_envs={args.n_envs})")

    if not args.compare:
        r = run(args.mode, backend, args.horizontal_scale, args.n_envs, measure=True)
        print(f"\n[{r['mode']}]  faces={r['faces']}  "
              f"x={r['x']:+.2f}  z={r['z']:.2f}  speed={r['speed']:.2f} m/s  "
              f"{r['ms']:.3f} ms/step  ({args.n_envs / r['ms'] * 1e3:.0f} env-steps/s)")
        return

    ri = run("single_scene", backend, args.horizontal_scale, args.n_envs, measure=True)
    rr = run("dual_scene", backend, args.horizontal_scale, args.n_envs, measure=True)
    print(f"\n{'mode':>9} | {'faces':>7} | {'x':>7} | {'speed':>6} | {'ms/step':>8} | "
          f"{'env-steps/s':>11}")
    print("-" * 64)
    for r in (ri, rr):
        print(f"{r['mode']:>9} | {r['faces']:>7} | {r['x']:>7.2f} | "
              f"{r['speed']:>6.2f} | {r['ms']:>8.3f} | "
              f"{args.n_envs / r['ms'] * 1e3:>11.0f}")
    dx = abs(ri["x"] - rr["x"])
    print(f"\npose match (|Δx|={dx:.3f} m): {'OK' if dx < 0.5 else 'DIVERGED'}")
    print(f"dual_scene speedup vs single_scene: {ri['ms'] / rr['ms']:.2f}x  "
          f"(grows with n_envs and terrain face count; ~1x at n_envs=1 on GPU)")


if __name__ == "__main__":
    main()
