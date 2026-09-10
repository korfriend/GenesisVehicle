"""dual_scene_terrain.py — VehicleScene unified API + the wheel-raycast dual/single scene modes.

Drives a 4-wheel car over a heightfield terrain using the high-level
``VehicleScene`` API (no manual ``gs.init`` / ``scene.build`` / ``scene.step``).
Demonstrates both raycast modes and (with ``--compare``) checks that they
produce the same drive, reporting each mode's step cost WITHOUT judging it.

What this demonstrates
----------------------
- ``VehicleScene`` as the single entry point: ``add_static`` +
  ``add_vehicle`` + ``build`` + a loop of ``veh.set_inputs(...) / vs.step()``.
- ``raycast_mode="dual_scene"`` (default): the terrain is raycast in a separate
  scene as a *static* body (BVH built once, never re-fit, shared across batch
  envs) while collision/rollover run in the main scene with the terrain as a
  *rigid* body. The wheel distances are identical to ``"single_scene"`` mode; only the
  per-step raycast cost differs.
- That the car is actually ON the terrain: the FINAL block asserts every wheel
  ray hits and that the chassis stays at ride height. This sample shipped for several releases silently "passing"
  while the car fell off the corner-origin terrain to z = -84 m.

What this does NOT demonstrate
------------------------------
- It does NOT benchmark the two modes. ``--compare`` prints each mode's
  ms/step — the absolute cost on YOUR machine, which is useful — but no ratio
  and no speedup verdict, because this harness cannot produce a trustworthy
  one: it runs both modes back to back inside ONE process and always in the
  SAME order (single_scene first), and the two slots are not equivalent.
  Measured over ten runs, single_scene is consistently slower when it is run
  in the second slot than in the first, and reversing the mode order moves the
  comparison; WHY the slots differ was not established, so do not attribute it
  to a mechanism. On top of that, the run-to-run spread on this CPU/WSL setup
  is of the same size as the difference between the modes. A ms/step from this
  sample describes this machine at this moment; a comparison between the two
  numbers is not a reproducible speedup and must not be quoted as one.
- It does not demonstrate the batched (``--n-envs``) or GPU win, which is where
  the static shared BVH is supposed to pay off; that needs a machine with a GPU
  and is not measurable here.

Why dual_scene helps (and when it does not)
-----------------------------------------
The ``"single_scene"`` (one-scene) wheel raycaster re-fits a BVH over every face each
step (the vehicle moves → the solver is non-static). ``"dual_scene"`` keeps the
terrain BVH static, so the *raycast* cost stops scaling with terrain face count
AND with ``n_envs`` (the static BVH is shared across envs) — that is a structural
argument, not a measured one, and this sample does not upgrade it to a measured
one; see ``docs/dual-scene-raycast.md`` for that explanation. Producing a
figure worth quoting needs a dedicated harness: a fresh process per mode,
repeats, and an alternating order (cf. ``perf_vectorization.py``, which
subprocesses every measurement). That is deliberately not this sample's job —
its job is the API and the pose match.

Run
---
    python -m genesis_vehicle.samples.dual_scene_terrain                 # dual_scene (default)
    python -m genesis_vehicle.samples.dual_scene_terrain --mode single_scene
    python -m genesis_vehicle.samples.dual_scene_terrain --compare           # dual_scene vs single_scene
    python -m genesis_vehicle.samples.dual_scene_terrain --compare --n-envs 64  # L3 batch
    python -m genesis_vehicle.samples.dual_scene_terrain --gpu                  # CPU is the default

Exits 0 only if every criterion in the FINAL block passes.
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
from genesis_vehicle.raycast import RAY_MISS_THRESHOLD

URDF_PATH = os.path.join(os.path.dirname(__file__), "urdf", "car_4w.urdf")

N_SUBTERRAINS = (1, 1)
SUBTERRAIN_SIZE = (40.0, 40.0)          # metres per subterrain (x, y)
TERRAIN_SPAN = (N_SUBTERRAINS[0] * SUBTERRAIN_SIZE[0],
                N_SUBTERRAINS[1] * SUBTERRAIN_SIZE[1])
# Half-extent after re-centring; the (d) criterion's bound is derived from it.
TERRAIN_HALF = (TERRAIN_SPAN[0] / 2.0, TERRAIN_SPAN[1] / 2.0)


def _terrain(horizontal_scale: float):
    """A single flat subterrain, re-centred on the world origin.

    Bump face count with a smaller horizontal_scale
    (faces ~= 2*(size/horizontal_scale)^2) to see the dual_scene advantage grow.
    """
    # gs.morphs.Terrain's ORIGIN IS A CORNER, not the centre: the mesh spans
    # [0, span_x] x [0, span_y], so a morph at the default pos leaves the world
    # origin ON the -X/-Y corner and a vehicle spawned at (0, 0) half off it.
    # This pos= shifts the corner to (-span/2, -span/2), which is what puts the
    # terrain's CENTRE at (0, 0) and gives the spawn 20 m of margin every way.
    return gs.morphs.Terrain(
        n_subterrains=N_SUBTERRAINS, subterrain_size=SUBTERRAIN_SIZE,
        pos=(-TERRAIN_HALF[0], -TERRAIN_HALF[1], 0.0),
        horizontal_scale=horizontal_scale, subterrain_types="flat_terrain")


def run(mode: str, backend: str, horizontal_scale: float, n_envs: int = 1,
        settle_s: float = 3.0, drive_s: float = 4.0, measure: bool = False):
    # SDK default timing (v1.0.19): 40 Hz (dt=0.025), substeps=10 → internal 2.5 ms.
    VehicleScene.init_backend(backend)
    vs = VehicleScene(raycast_mode=mode, dt=0.025,
                      substeps=10, n_envs=n_envs)
    vs.add_static(morph=_terrain(horizontal_scale))
    veh = vs.add_vehicle(URDF_PATH, car_4w_rwd_ackermann, pos=(0.0, 0.0, 3.0))
    vs.build()
    # genesis >= 1.4.0 dropped the RigidSolver.faces_info accessor; n_faces is
    # the same count and is public on every version the SDK supports.
    n_faces = int(vs.rigid_solver.n_faces)

    # settle_s=3.0, not 1.0: the car is dropped from z=3 and is still bouncing at
    # 1.0 s (z=0.479 vs a settled 0.112), which would put 0.37 m of bounce into
    # the z-stability criterion. By 3.0 s the residual z drift over the last 10
    # steps is 0.0004 m.
    for _ in range(int(settle_s / vs.dt)):
        veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        vs.step()

    # Read the wheel rays AFTER the settle loop: before the first scene.step()
    # the sensor buffer is a zero tensor (core.py), which is not "all rays hit".
    d0 = veh.distances.detach().cpu().numpy()
    d0_min, d0_max = float(d0.min()), float(d0.max())
    z_settle = float(veh.get_pos()[0, 2])

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

    P = veh.get_pos().detach().cpu().numpy()          # (n_envs, 3)
    p = P[0]
    v = veh.get_vel()[0].cpu().numpy()
    speed = float(np.linalg.norm(v[:2]))
    return dict(
        mode=mode, faces=n_faces,
        # env-0 scalars: the compare table and the |Δx| pose match use these.
        x=float(p[0]), y=float(p[1]), z=float(p[2]), speed=speed, ms=ms,
        # envelopes over all envs — the verdicts use these so a single stray env
        # in an L3 batch cannot hide behind env 0.
        x_min=float(P[:, 0].min()), x_max=float(P[:, 0].max()),
        y_min=float(P[:, 1].min()), y_max=float(P[:, 1].max()),
        z_min=float(P[:, 2].min()), z_max=float(P[:, 2].max()),
        d0_min=d0_min, d0_max=d0_max, z_settle=z_settle,
    )


def verdicts(r: dict) -> "list[tuple[str, bool, str]]":
    """(label, ok, detail) for each pass criterion. Every threshold below is
    stated with the measurement it was set from (CPU backend, horizontal_scale
    0.25, faces=51212, n_envs=1)."""
    out = []

    # (a) Every wheel ray must HIT the terrain after settling.
    #     d0_max < RAY_MISS_THRESHOLD: a miss reads back as the sentinel (19.9 is
    #     the DEFAULT raycaster_max_range=20.0 less a margin — this check is
    #     valid only at that default range). d0_min > 0.0: a pre-step sensor
    #     read is a ZERO tensor (core.py), which would satisfy the max check
    #     vacuously, so the min half is what makes the criterion mean "hit".
    #     Measured on the centred terrain: d0 = [0.411, 0.411, 0.412, 0.412].
    #     Reproduced with the pos= line deleted: d0 = [20, 20, 20, 20] (all four
    #     rays missing, the car already 17 m below the world) and z_end = -185 —
    #     the exact bug that used to exit 0.
    ok = (r["d0_max"] < RAY_MISS_THRESHOLD) and (r["d0_min"] > 0.0)
    out.append(("(a) all wheel rays hit terrain",
                ok, f"d0 min={r['d0_min']:.3f} max={r['d0_max']:.3f} m "
                    f"(miss sentinel >= {RAY_MISS_THRESHOLD})"))

    # (b) The chassis sits at ride height and STAYS there.
    #     0 < z < 1.0: the settled ride height is 0.112 m; 1.0 m is far above any
    #     terrain feature here and far below a launch, and z <= 0 means the car
    #     is at or under the surface. |z_end - z_settle| < 0.25: measured 0.001
    #     after a 3 s settle, so 0.25 is ~200x margin — it fires on a fall
    #     (z_end was -74.3 off-terrain) or a suspension blow-up, not on noise.
    dz = abs(r["z"] - r["z_settle"])
    ok = (0.0 < r["z_min"]) and (r["z_max"] < 1.0) and (dz < 0.25)
    out.append(("(b) chassis at ride height, stable",
                ok, f"z_settle={r['z_settle']:.3f} z_end={r['z']:.3f} "
                    f"(env range {r['z_min']:.3f}..{r['z_max']:.3f})  |Δz|={dz:.3f} "
                    f"(need 0 < z < 1.0, |Δz| < 0.25)"))

    # (c) The car actually drove. 4 s at throttle 0.6 reaches x=10.34 m; the
    #     bound is 2.0 m — under a fifth of that, so it separates "drove" from
    #     "sat still / spun in place", without pinning tire tuning.
    ok = r["x_min"] > 2.0
    out.append(("(c) drove forward",
                ok, f"x_end={r['x']:+.3f} (worst env {r['x_min']:+.3f})  "
                    f"speed={r['speed']:.2f} m/s  (need x > 2.0)"))

    # (d) Still ON the terrain at the end. Half-extent is 20 m after re-centring;
    #     18 m leaves ~2 m for the car's own footprint.
    #     NOTE: this CANNOT fire at default arguments — 4 s of driving only
    #     reaches 10.34 m. It is a guard for users who raise --drive-s (or add
    #     steering): past ~7 s the car leaves the terrain and every other number
    #     here becomes a measurement of a fall. Do not delete it as dead code.
    bound = TERRAIN_HALF[0] - 2.0
    ax = max(abs(r["x_min"]), abs(r["x_max"]))
    ay = max(abs(r["y_min"]), abs(r["y_max"]))
    ok = (ax < bound) and (ay < TERRAIN_HALF[1] - 2.0)
    out.append(("(d) stayed within terrain bounds",
                ok, f"|x|max={ax:.2f} |y|max={ay:.2f}  (need < {bound:.1f}; "
                    f"terrain spans ±{TERRAIN_HALF[0]:.0f} m about the origin)"))
    return out


def report(r: dict) -> bool:
    """Print the FINAL block for one run; return True if it passed."""
    vs_ = verdicts(r)
    print(f"\n=== FINAL [{r['mode']}] ===")
    print(f"  faces          : {r['faces']}")
    print(f"  final pose     : x={r['x']:+.3f}  y={r['y']:+.3f}  z={r['z']:.3f}  "
          f"speed={r['speed']:.2f} m/s")
    if r["ms"] is not None:
        print(f"  step cost      : {r['ms']:.3f} ms/step  "
              f"(reported, not a benchmark result)")
    for label, ok, detail in vs_:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:34s} {detail}")
    allok = all(ok for _, ok, _ in vs_)
    print(f"FINAL: {'PASS' if allok else 'FAIL'}")
    return allok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dual_scene", "single_scene"], default="dual_scene")
    ap.add_argument("--gpu", action="store_true",
                    help="Opt into the GPU backend (default: CPU — faster at small n_envs).")
    ap.add_argument("--compare", action="store_true",
                    help="Run single_scene AND dual_scene, print pose match + timing.")
    ap.add_argument("--horizontal-scale", type=float, default=0.25,
                    help="Terrain cell size; smaller = more faces (which is what "
                         "dual_scene's static BVH stops paying per step).")
    ap.add_argument("--n-envs", type=int, default=1,
                    help="L3 batch size. dual_scene shares ONE static terrain BVH "
                         "across envs, so its raycast cost does not scale with n_envs "
                         "(structural; not benchmarked here).")
    ap.add_argument("--drive-s", type=float, default=4.0,
                    help="Seconds of open-loop throttle after settling (default 4). "
                         "Past ~7 s the car runs off the 40x40 m terrain.")
    args = ap.parse_args()
    backend = "gpu" if args.gpu else "cpu"

    print(f"genesis_vehicle v{sdk_version}  |  dual_scene_terrain  "
          f"(backend={backend}, horizontal_scale={args.horizontal_scale}, "
          f"n_envs={args.n_envs})")

    if not args.compare:
        r = run(args.mode, backend, args.horizontal_scale, args.n_envs,
                drive_s=args.drive_s, measure=True)
        print(f"\n[{r['mode']}]  faces={r['faces']}  "
              f"x={r['x']:+.2f}  z={r['z']:.2f}  speed={r['speed']:.2f} m/s  "
              f"{r['ms']:.3f} ms/step  ({args.n_envs / r['ms'] * 1e3:.0f} env-steps/s)")
        return 0 if report(r) else 1

    ri = run("single_scene", backend, args.horizontal_scale, args.n_envs,
             drive_s=args.drive_s, measure=True)
    rr = run("dual_scene", backend, args.horizontal_scale, args.n_envs,
             drive_s=args.drive_s, measure=True)
    print(f"\n{'mode':>12} | {'faces':>7} | {'x':>7} | {'speed':>6} | {'ms/step':>8} | "
          f"{'env-steps/s':>11}")
    print("-" * 67)
    for r in (ri, rr):
        print(f"{r['mode']:>12} | {r['faces']:>7} | {r['x']:>7.2f} | "
              f"{r['speed']:>6.2f} | {r['ms']:>8.3f} | "
              f"{args.n_envs / r['ms'] * 1e3:>11.0f}")
    ok_i, ok_r = report(ri), report(rr)
    dx = abs(ri["x"] - rr["x"])
    # Both modes raycast the same terrain, so the poses must agree; 0.5 m is
    # loose enough for solver-order float noise (measured |Δx| ~ 1e-3) and
    # tight enough to catch one mode raycasting the wrong geometry.
    ok_dx = dx < 0.5
    print(f"\n{'PASS' if ok_dx else 'FAIL'}  pose match |Δx|={dx:.3f} m (need < 0.5)")
    # Each mode's ms/step is REPORTED so a user learns the absolute per-step cost
    # on their own machine. NO ratio is printed, and none should be computed from
    # the two numbers, for two reasons:
    #   1. ORDERING BIAS. This function always runs single_scene first and
    #      dual_scene second, in the same process, and the two slots are NOT
    #      equivalent: measured repeatedly, single_scene comes out slower when it
    #      runs in the second slot than in the first, and swapping the mode order
    #      moves the comparison. What has been established is only that the order
    #      changes the answer — the cause was not identified, so do not write one
    #      down here.
    #   2. RUN-TO-RUN SPREAD. On CPU at n_envs=1 the spread between identical
    #      invocations is comparable to the difference between the modes.
    # Fixing this inside the sample would mean a fresh process per mode plus
    # repeats in an alternating order — a benchmark harness, which this sample is
    # not. Build that separately (see perf_vectorization.py).
    print(f"      ms/step single={ri['ms']:.3f} dual={rr['ms']:.3f}  "
          f"(per-mode cost on this machine; order-dependent — the modes are run "
          f"back to back in a fixed order, so these two are NOT a fair "
          f"comparison and no speedup may be derived from them)")
    allok = ok_i and ok_r and ok_dx
    print(f"FINAL: {'PASS' if allok else 'FAIL'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
