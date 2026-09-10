"""obstacles_and_ramp.py — the VehicleScene obstacle API, and what a wheel
contact model does at a vertical edge.

Drives a car through a small course built entirely with ``VehicleScene`` (no
manual ``gs.init`` / ``scene.build`` / ``scene.step``).

What this demonstrates
----------------------
- Every registration method and the ``collision`` / ``wheel_raycast`` /
  ``physics`` combination from ``docs/api-reference.md`` §0.2, made concrete by
  printing the body registry and then driving into each body:

  * ``add_ground_plane()`` — flat ground (always a raycast target).
  * ``add_dynamic(physics=True, wheel_raycast=False)`` — a free box the CHASSIS
    collides with and knocks away; the wheels do NOT sense it (collide-only is
    the default).
  * ``add_dynamic(physics=False, wheel_raycast=True)`` — a low ramp the wheels
    SENSE and drive onto (opt-in), a teleportable body, primitive collider.
  * ``add_static(collision_morph=, wheel_raycast_morph=)`` — a static block: the
    wheels always sense it; the coarse-collider / detailed-raycast split is a
    dual_scene feature. Note the default 5 s drive only brings the front axle
    ONTO this block's leading edge — the block's registration is what is shown
    here, not a crossing. ``--drive-s 7.5`` crosses it, but by then the
    open-loop car (no steering feedback, twice launched) has wandered 1-4 m in
    y, which criterion (f) reports as a FAIL. That is the sample working.

- ``--wheel-contact`` (SDK v1.6.0): ``point`` (DEFAULT, one ray per wheel,
  every earlier release's behaviour) vs ``swept_envelope`` (``--contact-samples``
  rays per wheel, reduced by the swept-circle lower envelope). The ramp lip is
  the one place on this course where the two differ, and the FINAL block prints
  the number that separates them: how much of the lip arrives in ONE step,
  ``Δc at the ramp lip``.

What this does NOT demonstrate
------------------------------
- It is NOT a physically credible obstacle crossing, in either contact mode.
  A 0.12 m ramp lip at ~3 m/s throws this 1200 kg car's chassis roughly a metre
  into the air. That is a characterised limitation of a ray-based wheel contact,
  not a tuning miss: a zero-radius downward probe reads ground height as a step
  function, so the whole obstacle height arrives as compression within one
  ``dt`` (measured below: ``Δc at the ramp lip`` ~= the ramp's own 0.12 m in the
  default ``point`` mode) and the damper
  term ``C·ċ`` — proportional to that jump over ``dt`` — dominates the contact
  force. ``swept_envelope`` spreads the same edge over several steps and takes
  the peak down, but it does not remove the artifact. The pass criteria for the
  lift and the peak normal force are therefore REGRESSION bands around today's
  measured values, not physical acceptance bounds. Do not read a passing run as
  "the obstacle crossing is correct".
- It does NOT compare the two contact modes for you, and prints no ratio
  between them. The magnitudes here are phase-sensitive — they depend on
  exactly where in a step the wheel meets the lip, which moves with the
  approach speed — so a ratio measured on one course at one speed is not a
  characteristic number (``samples/dual_scene_terrain.py`` carries the same
  warning about its two timings). Run both and read both.
- ``--bench`` reports each raycast mode's ms/step and NOTHING else: no ratio,
  no verdict. Both modes run back to back in one process in a fixed order, and
  these obstacles are primitives (cheap BVH), so the number that mode ordering
  and cache state produce here is not a speedup. dual_scene's win needs a heavy
  static MESH terrain and/or large ``n_envs``; see ``dual_scene_terrain.py``
  and ``docs/dual-scene-raycast.md``.

Run
---
    python -m genesis_vehicle.samples.obstacles_and_ramp
    python -m genesis_vehicle.samples.obstacles_and_ramp --mode single_scene
    python -m genesis_vehicle.samples.obstacles_and_ramp --wheel-contact swept_envelope
    python -m genesis_vehicle.samples.obstacles_and_ramp --viewer      # headless by default
    python -m genesis_vehicle.samples.obstacles_and_ramp --gpu         # CPU by default
    python -m genesis_vehicle.samples.obstacles_and_ramp --bench --n-envs 64

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
from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann, __version__ as sdk_version

URDF_PATH = os.path.join(os.path.dirname(__file__), "urdf", "car_4w.urdf")

# ---- the course -------------------------------------------------------------
# car_4w.urdf's chassis COLLISION box is <box 4.3 x 1.7 x 0.5> centred at body
# z = 0.85, i.e. its bottom sits 0.60 m above the base link and ~0.71 m above the
# ground at ride height (it is deliberately high so the wheel rays cannot hit the
# car's own body). A knock box has to reach INTO that band to be knocked: the
# 0.5 m box this sample used until v1.6.0 passed clean underneath the chassis and
# was never touched while driving — it only ever moved because the car, launched
# by the ramp lip, swung its nose up and clipped the box with its tail 1.5 m
# further on. So the box is 1.0 m tall (top at z = 1.0, overlapping the collider
# band by ~0.3 m) and light (1 kg), and the chassis meets it head-on.
BOX_SIZE, BOX_POS, BOX_MASS = (0.5, 0.5, 1.0), (4.0, 0.0, 0.5), 1.0
# Ramp and block are both 0.12 m tall and 2 m long. The front axle is at body
# x = +1.35, so the front wheels meet the ramp's leading edge (x = 7.0) when the
# chassis is at x ~= 5.6, and the block's (x = 10.0) at chassis x ~= 8.65.
OBSTACLE_H = 0.12
RAMP_POS, BLOCK_POS = (8.0, 0.0, OBSTACLE_H / 2), (11.0, 0.0, OBSTACLE_H / 2)
RAMP_LIP_X = RAMP_POS[0] - 1.0                  # leading edge of the 2 m ramp
# Δc that counts as "the wheel just met an edge". Flat-ground steady state on
# this course is |Δc| ~ 2e-4 m, so 0.02 is two orders of magnitude clear of the
# noise and an order below the 0.12 m obstacle — it cannot miss the lip and
# cannot fire before it.
DC_LIP_TRIGGER = 0.02
THROTTLE = 0.45
SETTLE_S, DRIVE_S = 1.0, 5.0


def _build_course(mode, n_envs, backend, cfg, wheel_contact="point",
                  contact_samples=9, viewer=False):
    """Build the obstacle course on a fresh VehicleScene. Returns (vs, veh, box)."""
    VehicleScene.init_backend(backend)
    # dt comes from the preset; the internal rate is asked for with substeps=
    # (never with RigidOptions(dt=...)).
    vs = VehicleScene(raycast_mode=mode, dt=cfg.recommended_dt,
                      substeps=10, n_envs=n_envs, show_viewer=viewer)
    vs.add_ground_plane(friction=1.0)
    # (1) collide-only free box at x=4 — the chassis shoves it, wheels don't sense it.
    knock_box = vs.add_dynamic(
        gs.morphs.Box(size=BOX_SIZE, pos=BOX_POS),
        physics=True, wheel_raycast=False, mass=BOX_MASS, name="knock_box")
    # (2) low ramp at x=8 the wheels SENSE and roll over (wheel_raycast=True opt-in).
    vs.add_dynamic(
        gs.morphs.Box(size=(2.0, 2.0, OBSTACLE_H), pos=RAMP_POS),
        physics=False, wheel_raycast=True, name="ramp")
    # (3) static block at x=11 — always a raycast target; the collision/raycast
    #     split only takes effect in dual_scene.
    vs.add_static(
        collision_morph=gs.morphs.Box(size=(2.0, 2.0, OBSTACLE_H), pos=BLOCK_POS),
        wheel_raycast_morph=gs.morphs.Box(size=(2.0, 2.0, OBSTACLE_H), pos=BLOCK_POS),
        material=gs.materials.Rigid(friction=1.0), name="block")
    veh = vs.add_vehicle(URDF_PATH, car_4w_rwd_ackermann, cfg=cfg,
                         pos=(0.0, 0.0, 0.6), material=gs.materials.Rigid(friction=1.0),
                         wheel_contact=wheel_contact, contact_samples=contact_samples)
    vs.build()
    return vs, veh, knock_box


def _proto(vs):
    """The kind proto that carries the per-step diagnostics (last_N,
    last_compression). There is no public per-vehicle accessor for the contact
    FORCES — `Vehicle` exposes `.distances` / `.wheels_grounded` but not
    `.last_N` — so this sample reaches into the batched solver's single kind.
    Only diagnostics are read; nothing is written."""
    return vs.physics.kinds[0]._proto


def run(mode, backend, cfg, wheel_contact="point", contact_samples=9,
        viewer=False, drive_s=DRIVE_S, quiet=False):
    """Drive the course once; return every number the FINAL block reports."""
    vs, veh, knock_box = _build_course(mode, 1, backend, cfg, wheel_contact,
                                       contact_samples, viewer)
    if not quiet:
        print(f"\n  raycast_mode = {vs.raycast_mode}  (is_dual_scene = {vs.is_dual_scene})"
              f"   wheel_contact = {wheel_contact}"
              f"{f' (M={contact_samples})' if wheel_contact != 'point' else ''}")
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

    DT = vs.dt
    box_x0 = float(knock_box.entity_main.get_pos()[0, 0])

    for _ in range(int(SETTLE_S / DT)):          # settle on the ground
        veh.set_inputs(brake=1.0)
        vs.step()
    z_ground = float(veh.get_pos()[0, 2])
    p_ = _proto(vs)
    prev_c = p_.last_compression.clone()

    max_z, n_peak = z_ground, 0.0
    dc_max, dc_lip, lip = 0.0, None, {}
    z_min = z_ground
    v_lip = None
    prev_x, prev_speed = float(veh.get_pos()[0, 0]), 0.0
    for i in range(int(drive_s / DT)):           # drive forward through the course
        veh.set_inputs(throttle=THROTTLE)
        vs.step()
        z = float(veh.get_pos()[0, 2])
        x = float(veh.get_pos()[0, 0])
        speed = float(np.linalg.norm(veh.get_vel()[0, :2].cpu().numpy()))
        c = p_.last_compression
        dc = float((c - prev_c).max())
        prev_c = c.clone()
        n_now = float(p_.last_N.max())
        max_z, z_min = max(max_z, z), min(z_min, z)
        n_peak = max(n_peak, n_now)
        dc_max = max(dc_max, dc)                 # includes landing impacts
        # THE LIP EVENT: the first ground-height step this drive sees. The course
        # is flat until the ramp's leading edge (|Δc| ~ 2e-4 on flat ground), so
        # the first Δc over DC_LIP_TRIGGER is the front wheels meeting the lip —
        # and that, not the global max, is what the contact model decides. The
        # global max lands later and is a re-entry impact after the launch, whose
        # magnitude depends on where the car happens to come down.
        if dc_lip is None and dc > DC_LIP_TRIGGER:
            dc_lip = dc
            # v_lip is the speed on the step BEFORE the event, not on it: the
            # contact impulse lands inside the event step and adds ~1 m/s to the
            # reading there. The APPROACH speed is what the crossing's phase (and
            # therefore every magnitude below) depends on, so report that.
            lip = dict(t=i * DT, x=prev_x, speed=prev_speed, N=n_now)
            v_lip = prev_speed
        prev_x, prev_speed = x, speed

    p = veh.get_pos()[0].cpu().numpy()
    speed = float(np.linalg.norm(veh.get_vel()[0, :2].cpu().numpy()))
    return dict(
        mode=mode, wheel_contact=wheel_contact, contact_samples=contact_samples,
        x=float(p[0]), y=float(p[1]), z=float(p[2]), speed=speed,
        z_ground=z_ground, lift=max_z - z_ground, z_min=z_min,
        n_peak=n_peak, dc_max=dc_max, dc_lip=dc_lip, lip=lip,
        v_lip=v_lip if v_lip else 0.0,
        box_dx=float(knock_box.entity_main.get_pos()[0, 0]) - box_x0,
        grounded=bool(veh.all_wheels_grounded.all()),
        dt=DT, drive_s=drive_s,
    )


# ---- pass criteria ----------------------------------------------------------
# Every band below was measured on this course, CPU backend, genesis-world 1.4.0,
# n_envs=1, throttle 0.45, 1 s settle + 5 s drive, and holds for BOTH raycast
# modes (dual_scene and single_scene agree to ~1e-3 here).
#
#   wheel_contact     lift (m)   peak N (N)   Δc at lip (m)   x_end (m)   box Δx (m)
#   point               0.836        82_010          0.120         9.06        1.82
#   swept_envelope      0.686        50_298          0.072        10.36        1.04
#
# single_scene reproduces the same run: lift and Δc identical to 3 decimals,
# peak N within 0.04% (82_304 / 50_320), x_end identical to 2 decimals.
#
# The lift and force bands are REGRESSION bands, not physical acceptance: a
# 0.12 m lip must not throw a car a metre into the air, and no threshold here
# claims otherwise (see the module docstring). They are one-sided upward with
# ~25% headroom, so a change that makes the artifact WORSE fails while ordinary
# solver noise does not. The lower guards are physical, not pinned: a car that
# drives over a 0.12 m obstacle must rise at least ~half its height, and a wheel
# that senses it must see some compression step.
BANDS = {
    # lift_max: measured 0.836 -> 1.05 (~26% headroom).
    # n_max:    measured 82_010 -> 105_000 (~28%; the peak is the re-entry impact
    #           after the launch, so it carries more phase spread than Δc at the lip).
    # dc_lo/hi: the point contact takes the WHOLE obstacle in one dt. Measured
    #           0.120 == OBSTACLE_H to 3 decimals; the band 0.09..0.13 pins that
    #           characterisation (it cannot meaningfully exceed the obstacle height).
    "point": dict(lift_max=1.05, n_max=105_000.0, dc_lo=0.09, dc_hi=0.13),
    # The fan's guarantee is DIRECTIONAL: the same edge is spread over several
    # steps, so Δc at the lip must land clearly below the obstacle height. Measured
    # 0.072; 0.095 is under the point contact's 0.120 by more than the spread
    # between raycast modes (0.000) and dc_lo=0.02 (== DC_LIP_TRIGGER) still
    # proves the wheel SAW the ramp.
    # lift/n bands: measured 0.686 / 50_298 -> 0.86 / 65_000 (~25-29% headroom).
    # Measured at the default M=9 and re-checked at M=15 (Δc 0.035, N 41_501,
    # lift 0.640) and M=31 (Δc 0.070, N 57_466, lift 0.612) — note M=31 is WORSE
    # than M=15 on both. M is a discretisation count, not an accuracy dial, so
    # these bands are not promised to hold at an arbitrary --contact-samples.
    "swept_envelope": dict(lift_max=0.86, n_max=65_000.0, dc_lo=0.02, dc_hi=0.095),
}


def verdicts(r):
    """(label, ok, detail) per criterion. Thresholds and their reasons above."""
    b = BANDS[r["wheel_contact"]]
    out = []

    # (a) The chassis actually knocked the collide-only box away. Measured 3.30 m
    #     in both contact modes; 0.5 m is well clear of "nudged by a passing
    #     shockwave" and would have caught the pre-v1.6.0 course, where the box
    #     sat under the chassis collider and moved 0.00 m (swept_envelope) or
    #     1.58 m for the wrong reason (point: clipped by the launched car's tail).
    ok = r["box_dx"] > 0.5
    out.append(("(a) chassis knocked the collide-only box", ok,
                f"box Δx={r['box_dx']:+.2f} m  (need > 0.5; wheels never sensed it)"))

    # (b) The car crossed the ramp completely: the front axle (body x = +1.35)
    #     is past the ramp's trailing edge (x = 9.0) once the chassis is at 7.65.
    #     Measured x_end after the default 5 s drive: 9.06 (point) / 10.36
    #     (swept_envelope), so the bound has >= 1.4 m of margin. It separates
    #     "crossed" from "stopped at the lip / spun in place / bounced back",
    #     without pinning tire tuning. (The static block's leading edge, x = 10.0,
    #     is only just reached by the front axle at the end of the default drive —
    #     see the module docstring.)
    ok = r["x"] > 7.65
    out.append(("(b) crossed the ramp", ok,
                f"x_end={r['x']:+.2f} m  speed={r['speed']:.2f} m/s  (need x > 7.65 "
                f"= front axle past the ramp's trailing edge)"))

    # (c) The wheels SENSED the raycast bodies (this is what wheel_raycast=True
    #     buys). A car crossing a 0.12 m obstacle must rise by at least about half
    #     its height; 0.05 m is that floor and fires if the ramp mirror stops
    #     being a raycast target. The upper bound is the regression band.
    ok = (r["lift"] > 0.05) and (r["lift"] < b["lift_max"])
    out.append(("(c) wheels sensed the ramp; lift in band", ok,
                f"z {r['z_ground']:.3f} -> peak +{r['lift']:.3f} m  "
                f"(need 0.05 < lift < {b['lift_max']:.2f}; REGRESSION band, "
                f"not a physical bound)"))

    # (d) THE CONTACT-MODEL CRITERION: how much of the lip arrives in one dt.
    #     For "point" the one-step compression jump IS the obstacle height (the
    #     defect, pinned as a characterisation); for "swept_envelope" it must be
    #     clearly smaller (the feature: the same edge spread over several steps).
    #     Measured at the lip, not at the global max — the global max is a
    #     re-entry impact after the launch and depends on where the car lands.
    lip = r["lip"]
    ok = (r["dc_lip"] is not None) and (b["dc_lo"] < r["dc_lip"] < b["dc_hi"])
    out.append((f"(d) Δc at the ramp lip in one dt", ok,
                f"{(r['dc_lip'] or 0.0):.3f} m at t={lip.get('t', 0):.2f}s "
                f"x={lip.get('x', 0):.2f} ({lip.get('speed', 0):.2f} m/s, "
                f"N={lip.get('N', 0):,.0f})  (need {b['dc_lo']:.2f} < Δc < "
                f"{b['dc_hi']:.3f} for wheel_contact={r['wheel_contact']}; "
                f"the obstacle is {OBSTACLE_H:.2f} m tall)"))

    # (e) Peak normal force regression band. Purely a "did this get worse?" pin.
    ok = r["n_peak"] < b["n_max"]
    out.append(("(e) peak wheel normal force in band", ok,
                f"{r['n_peak']:,.0f} N  (need < {b['n_max']:,.0f}; REGRESSION band. "
                f"Static load is ~2.9 kN/wheel — this is the damper spike, "
                f"not a credible contact force)"))

    # (f) The car stayed on the course: never below the ground plane and never
    #     thrown clear of it. z_min > 0 catches falling through; z < 2.5 m catches
    #     the launch class of bug (a car at z=+1495 once passed a forward-distance
    #     check in this repo for a whole release). 2.5 is above the measured
    #     1.16 m peak and far below any real blow-up.
    z_peak = r["z_ground"] + r["lift"]
    ok = (r["z_min"] > 0.0) and (z_peak < 2.5) and (abs(r["y"]) < 2.0)
    out.append(("(f) stayed on the course", ok,
                f"z ∈ [{r['z_min']:.3f}, {z_peak:.3f}]  |y|={abs(r['y']):.2f} m  "
                f"(need z > 0, z_peak < 2.5, |y| < 2.0)"))
    return out


def report(r):
    """Print the FINAL block for one run; return True if it passed."""
    vs_ = verdicts(r)
    tag = (f"{r['mode']} / {r['wheel_contact']}"
           + (f" M={r['contact_samples']}" if r["wheel_contact"] != "point" else ""))
    print(f"\n=== FINAL [{tag}] ===")
    print(f"  final pose     : x={r['x']:+.2f}  y={r['y']:+.2f}  z={r['z']:.2f}  "
          f"speed={r['speed']:.2f} m/s  (all wheels grounded: {r['grounded']})")
    print(f"  ramp approach  : {r['v_lip']:.2f} m/s on the step before the lip "
          f"(every magnitude below is phase-sensitive to this)")
    print(f"  ride height    : {r['z_ground']:.3f} m  ->  peak +{r['lift']:.3f} m")
    print(f"  Δc at ramp lip : {(r['dc_lip'] or 0.0):.3f} m in one dt "
          f"(obstacle is {OBSTACLE_H:.2f} m; dt = {r['dt']:.3f} s)")
    print(f"  max Δc per dt  : {r['dc_max']:.3f} m  (anywhere on the course — the "
          f"later re-entry impact, not the lip)")
    print(f"  peak wheel N   : {r['n_peak']:,.0f} N")
    for label, ok, detail in vs_:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:40s} {detail}")
    allok = all(ok for _, ok, _ in vs_)
    print(f"FINAL: {'PASS' if allok else 'FAIL'}")
    return allok


def _bench(backend, n_envs, cfg, wheel_contact, contact_samples,
           drive_s=3.0, settle_s=1.0):
    """Time the drive loop in each raycast mode; return {mode: ms_per_step}."""
    import torch
    out = {}
    for mode in ("single_scene", "dual_scene"):
        vs, veh, _ = _build_course(mode, n_envs, backend, cfg, wheel_contact,
                                   contact_samples)
        veh.set_inputs(throttle=0.0, brake=1.0)
        for _ in range(int(settle_s / vs.dt)):      # settle + warm up kernels
            vs.step()
        n = int(drive_s / vs.dt)
        veh.set_inputs(throttle=THROTTLE)
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
    ap.add_argument("--wheel-contact", choices=["point", "swept_envelope"],
                    default="point",
                    help="wheel-ground contact model (default point — one ray per "
                         "wheel, the historical behaviour).")
    ap.add_argument("--contact-samples", type=int, default=9,
                    help="rays per wheel (M, odd) for --wheel-contact swept_envelope. "
                         "M is a discretisation count, NOT an accuracy dial: the "
                         "pass bands were measured at M=9 and re-checked at 15/31, "
                         "where the peak force does not fall monotonically.")
    ap.add_argument("--gpu", action="store_true",
                    help="opt into the GPU backend (default: CPU — faster at small n_envs).")
    ap.add_argument("--viewer", action="store_true",
                    help="open the interactive viewer (default: headless).")
    ap.add_argument("--drive-s", type=float, default=DRIVE_S,
                    help=f"seconds of open-loop throttle after settling (default "
                         f"{DRIVE_S}). The pass bands were measured at the default.")
    ap.add_argument("--bench", action="store_true",
                    help="report single_scene / dual_scene ms/step. Reporting only: "
                         "no criteria are checked and no ratio is printed.")
    ap.add_argument("--n-envs", type=int, default=1,
                    help="L3 batch size for --bench (default 1).")
    args = ap.parse_args()
    backend = "gpu" if args.gpu else "cpu"
    cfg = car_4w_rwd_ackermann(URDF_PATH, stability="control")

    if args.bench:
        print(f"genesis_vehicle v{sdk_version}  |  obstacles_and_ramp  --bench  "
              f"(backend={backend}, n_envs={args.n_envs}, "
              f"wheel_contact={args.wheel_contact})")
        ms = _bench(backend, args.n_envs, cfg, args.wheel_contact,
                    args.contact_samples)
        print(f"\n  {'mode':12s} {'ms/step':>9s} {'ms/env-step':>12s}")
        for mode in ("single_scene", "dual_scene"):
            print(f"  {mode:12s} {ms[mode]:9.3f} {ms[mode] / args.n_envs:12.4f}")
        print("\n  Per-mode cost on THIS machine only. No ratio is printed: both "
              "modes run\n  back to back in one process in a fixed order, and these "
              "obstacles are\n  primitives (cheap BVH re-fit), so nothing here is a "
              "speedup. --bench checks\n  no criteria and always exits 0.")
        return 0

    print(f"genesis_vehicle v{sdk_version}  |  obstacles_and_ramp  "
          f"(mode={args.mode}, wheel_contact={args.wheel_contact}, backend={backend})")
    r = run(args.mode, backend, cfg, args.wheel_contact, args.contact_samples,
            viewer=args.viewer, drive_s=args.drive_s)
    if args.drive_s != DRIVE_S:
        print(f"\n  NOTE: --drive-s {args.drive_s} != the {DRIVE_S} s the pass bands "
              f"were measured at; a FAIL may just mean a different course.")
    return 0 if report(r) else 1


if __name__ == "__main__":
    sys.exit(main())
