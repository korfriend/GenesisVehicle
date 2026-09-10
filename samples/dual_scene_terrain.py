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
  ray hits (``veh.all_wheels_grounded``) and that the chassis stays at
  ride height. This sample shipped for several releases silently "passing"
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
its job is the API and the pose match. That harness now exists as
``samples/bench_raycast_mode.py``; it drives THIS module's ``run()`` and
``verdicts()`` in a fresh subprocess per measurement — which is why ``run()``
takes ``terrain_size=`` and returns its own conditions (dt, substeps, faces,
window speeds, ...) in the result dict. Whether it prints a ratio at all is
decided by its own fail-closed rule; it may legitimately report that the ratio
is not measurable on your machine.

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

URDF_PATH = os.path.join(os.path.dirname(__file__), "urdf", "car_4w.urdf")

N_SUBTERRAINS = (1, 1)
SUBTERRAIN_SIZE = (40.0, 40.0)          # metres per subterrain (x, y)
TERRAIN_SPAN = (N_SUBTERRAINS[0] * SUBTERRAIN_SIZE[0],
                N_SUBTERRAINS[1] * SUBTERRAIN_SIZE[1])
# DEFAULT half-extent after re-centring; the (d) criterion's bound is derived
# from it. This is the DEFAULT only: ``run(terrain_size=...)`` builds a
# different plate and reports the half-extent it actually used as
# ``terrain_half`` in its result dict, which is what ``verdicts()`` reads.
# ``verdicts()`` falls back to this constant only for a dict with no
# ``terrain_half`` key (i.e. any result produced before v1.6.1).
TERRAIN_HALF = (TERRAIN_SPAN[0] / 2.0, TERRAIN_SPAN[1] / 2.0)


def _as_size(size) -> "tuple[float, float]":
    """Normalise a terrain size to (Lx, Ly) metres.

    ``None`` -> the module default ``SUBTERRAIN_SIZE``; a scalar -> a square
    plate of that side; a 2-sequence -> (Lx, Ly) as given.
    """
    if size is None:
        return (float(SUBTERRAIN_SIZE[0]), float(SUBTERRAIN_SIZE[1]))
    if isinstance(size, (int, float)):
        return (float(size), float(size))
    lx, ly = size
    return (float(lx), float(ly))


def _terrain(horizontal_scale: float, size=None):
    """A single flat subterrain, re-centred on the world origin.

    Bump face count with a smaller horizontal_scale
    (faces ~= 2*(size/horizontal_scale)^2) to see the dual_scene advantage grow.

    ``size`` (metres; scalar or (Lx, Ly)) overrides ``SUBTERRAIN_SIZE`` for a
    bigger plate; ``None`` keeps the module default, so the default call is
    unchanged. The re-centring ``pos=`` below is derived from whichever size is
    in force. NOTE genesis requires size/horizontal_scale to be a whole number
    of cells (e.g. 640/4.0 = 160).
    """
    # gs.morphs.Terrain's ORIGIN IS A CORNER, not the centre: the mesh spans
    # [0, span_x] x [0, span_y], so a morph at the default pos leaves the world
    # origin ON the -X/-Y corner and a vehicle spawned at (0, 0) half off it.
    # This pos= shifts the corner to (-span/2, -span/2), which is what puts the
    # terrain's CENTRE at (0, 0) and gives the spawn half the plate as margin
    # every way (20 m at the default 40x40 m size).
    span = _as_size(size)
    half = (span[0] / 2.0, span[1] / 2.0)
    return gs.morphs.Terrain(
        n_subterrains=N_SUBTERRAINS, subterrain_size=span,
        pos=(-half[0], -half[1], 0.0),
        horizontal_scale=horizontal_scale, subterrain_types="flat_terrain")


def run(mode: str, backend: str, horizontal_scale: float, n_envs: int = 1,
        settle_s: float = 3.0, drive_s: float = 4.0, measure: bool = False,
        terrain_size=None):
    """Settle a car on a flat plate, then drive it open-loop; return a result dict.

    ``terrain_size`` (v1.6.1, metres, scalar or (Lx, Ly); default ``None`` =
    the module's 40x40 m plate) lets a caller ask for a bigger plate so a long
    timed window stays on the terrain. The half-extent actually used is
    returned as ``terrain_half`` and is what ``verdicts()`` bounds against.

    ``measure=True`` adds a 10-step warm-up and times the drive loop.

    Result keys (v1.6.1 added the conditions block so a caller in another
    process can print the conditions it MEASURED rather than assume them):
    the pre-existing pose/settle keys are unchanged, and the dict now also
    carries dt, substeps, terrain_size, terrain_half, settle_s, drive_s,
    throttle, steer, n_envs, backend, horizontal_scale, genesis_version,
    sdk_version, plus the
    timing-window block ``ms``, ``warmup_steps``, ``n_timed_steps``,
    ``timing_definition``, ``x_at_window_start``, ``speed_at_window_start``,
    ``speed_at_window_end``.

    ON THE ``measure=False`` PATH EVERY KEY IN THAT TIMING-WINDOW BLOCK IS
    ``None`` — following the convention ``ms`` has always used. That path has
    no warm-up loop and no timer, so there is no window whose speeds could be
    reported; the drive steps still run, they are simply not timed. Callers
    must not treat ``n_timed_steps=None`` as zero steps driven.
    """
    # SDK default timing (v1.0.19): 40 Hz (dt=0.025), substeps=10 → internal 2.5 ms.
    span = _as_size(terrain_size)
    terrain_half = (span[0] / 2.0, span[1] / 2.0)
    VehicleScene.init_backend(backend)
    vs = VehicleScene(raycast_mode=mode, dt=0.025,
                      substeps=10, n_envs=n_envs)
    vs.add_static(morph=_terrain(horizontal_scale, size=span))
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

    # Read the wheel rays AFTER the settle loop. `all_wheels_grounded` is the
    # supported "did every ray find ground?" question (v1.5.2): it tests the
    # raycaster's OWN miss sentinel and is gated on the per-env has-stepped
    # flag, so it cannot pass on the zero tensor `distances` starts out as.
    # The raw distances below are reported for context only.
    grounded = bool(veh.all_wheels_grounded.all())
    d0 = veh.distances.detach().cpu().numpy()
    d0_min, d0_max = float(d0.min()), float(d0.max())
    z_settle = float(veh.get_pos()[0, 2])

    # The drive-phase inputs, as ONE binding used by every drive loop below
    # and reported in the result dict — so a caller in another process reads
    # what actually drove the car instead of restating a literal.
    throttle, steer = 0.6, 0.0

    ms = None
    warmup_steps = n_timed_steps = None
    x_w0 = v_w0 = v_w1 = None
    n = int(drive_s / vs.dt)
    if measure:
        warmup_steps, n_timed_steps = 10, n
        for _ in range(warmup_steps):
            veh.set_inputs(throttle=throttle, steer=steer); vs.step()
        # State at the window's START, read AFTER the warm-up and BEFORE the
        # timer starts. It must not be read inside the timed loop: every read
        # here is a device->host sync, and putting one in the loop would
        # contaminate the very ms/step this function exists to produce.
        x_w0 = float(veh.get_pos()[0, 0])
        v_w0 = float(np.linalg.norm(veh.get_vel()[0].cpu().numpy()[:2]))
        t0 = time.perf_counter()
        for _ in range(n):
            veh.set_inputs(throttle=throttle, steer=steer); vs.step()
        ms = (time.perf_counter() - t0) / n * 1e3
        # ... and the END state after the timer has stopped.
        v_w1 = float(np.linalg.norm(veh.get_vel()[0].cpu().numpy()[:2]))
    else:
        for _ in range(n):
            veh.set_inputs(throttle=throttle, steer=steer); vs.step()

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
        d0_min=d0_min, d0_max=d0_max, z_settle=z_settle, grounded=grounded,
        # --- conditions (v1.6.1). Everything below is READ from this run, so a
        # parent process can print the conditions instead of restating them.
        dt=float(vs.dt), substeps=10,
        terrain_size=(span[0], span[1]), terrain_half=terrain_half,
        settle_s=float(settle_s), drive_s=float(drive_s),
        # The open-loop drive inputs actually used (warm-up and drive loops
        # alike). Reported on BOTH paths — they describe the drive, not the
        # timing window, so measure=False does not null them.
        throttle=float(throttle), steer=float(steer),
        n_envs=int(n_envs), backend=str(backend),
        horizontal_scale=float(horizontal_scale),
        genesis_version=str(getattr(gs, "__version__", "unknown")),
        sdk_version=str(sdk_version),
        # --- timing window (all None when measure=False; see the docstring)
        warmup_steps=warmup_steps, n_timed_steps=n_timed_steps,
        timing_definition=("wall time of the full vs.step() loop including "
                           "veh.set_inputs, divided by n_timed_steps"),
        x_at_window_start=x_w0,
        speed_at_window_start=v_w0, speed_at_window_end=v_w1,
    )


def verdicts(r: dict) -> "list[tuple[str, bool, str]]":
    """(label, ok, detail) for each pass criterion. Every threshold below is
    stated with the measurement it was set from (CPU backend, horizontal_scale
    0.25, faces=51212, n_envs=1).

    The (d) bounds come from ``r["terrain_half"]`` — the half-extent the run
    ACTUALLY built (v1.6.1) — falling back to the module default
    ``TERRAIN_HALF`` (20 m, bound 18.0) for a dict without that key, so a
    result recorded before v1.6.1 verdicts exactly as it did then. The (a)-(c)
    thresholds are absolute and do not scale with the plate."""
    out = []

    # (a) Every wheel ray must HIT the terrain after settling.
    #     `veh.all_wheels_grounded` (v1.5.2) compares each ray distance with the
    #     raycaster's own miss sentinel (no_hit_value, defaulting to max_range)
    #     and AND's in the per-env has-stepped flag — the mask is produced by
    #     the read layer on the RAW distances where it can be (single_scene) and
    #     from the sentinel alone on injected, offset-corrected ones
    #     (dual_scene); the has-stepped gate covers the difference. It replaces this
    #     sample's old hand-rolled `d0_max < RAY_MISS_THRESHOLD and d0_min > 0`,
    #     which was only valid at the DEFAULT raycaster_max_range=20.0 — at a
    #     lower range a miss came back UNDER 19.9 and the check passed forever.
    #     Measured on the centred terrain: d0 = [0.411, 0.411, 0.412, 0.412].
    #     Reproduced with the pos= line deleted: d0 = [20, 20, 20, 20] (all four
    #     rays missing, the car already 17 m below the world) and z_end = -185 —
    #     the exact bug that used to exit 0.
    ok = bool(r["grounded"])
    out.append(("(a) all wheel rays hit terrain",
                ok, f"all_wheels_grounded={r['grounded']}  "
                    f"d0 min={r['d0_min']:.3f} max={r['d0_max']:.3f} m"))

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
    #     steering). MEASURED on the default plate (40x40 m, horizontal_scale
    #     0.25, CPU, genesis-world 1.4.0, dt=0.025/substeps=10, n_envs=1, 3 s
    #     brake settle then constant throttle 0.6, no steering): x crosses this
    #     18 m bound at t = 5.65 s and the terrain edge at 20 m at t = 5.95 s.
    #     (A comment here previously said "~7 s"; that was wrong — the car is
    #     already off the plate by then and every other number in this block
    #     becomes a measurement of a fall.) Do not delete this as dead code.
    #     On a larger plate (``run(terrain_size=...)``) both bounds and the
    #     crossing time scale with the plate; the times above are the default's.
    half = r.get("terrain_half", TERRAIN_HALF)
    bound = half[0] - 2.0
    ax = max(abs(r["x_min"]), abs(r["x_max"]))
    ay = max(abs(r["y_min"]), abs(r["y_max"]))
    ok = (ax < bound) and (ay < half[1] - 2.0)
    out.append(("(d) stayed within terrain bounds",
                ok, f"|x|max={ax:.2f} |y|max={ay:.2f}  (need < {bound:.1f}; "
                    f"terrain spans ±{half[0]:.0f} m about the origin)"))
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
                         "Measured on this 40x40 m terrain (CPU, horizontal_scale "
                         "0.25, 3 s settle, constant throttle 0.6, no steering): "
                         "x passes the (d) bound of 18 m at t=5.65 s and the "
                         "terrain edge at 20 m at t=5.95 s, so anything past ~5.6 s "
                         "fails (d) and then measures a fall.")
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
