"""terrain_drive.py — drive a car over an UNDULATING (bumpy) mesh terrain.

Built on the high-level **`VehicleScene`** API (0.9.0): no manual ``gs.init`` /
``scene.build`` / ``scene.step`` — just ``add_static`` + ``add_vehicle`` +
``build`` + a loop of ``veh.set_inputs(...) / vs.step()``.

Unlike the flat-`Plane` samples this generates a real wavy ground mesh and, in
the default ``raycast_mode="dual_scene"``, registers it with
``add_static(collision=False)`` so it lives ONLY in the raycast scene as a
**`Kinematic` raycast target**: the wheel raycaster follows every bump (the
chassis pitches/heaves), but that scene's BVH is *static* → the raycaster
**skips rebuilding it every step**. A high-poly mesh as a `Rigid` collider would
instead be re-fit into the wheel-raycaster BVH on every step (the dominant cost
on big maps — see ``docs/dual-scene-raycast.md`` / CHANGELOG v0.7.17). Pass
``--rigid-terrain`` to run that slow path (``raycast_mode="single_scene"`` with a
rigid mesh) for an A/B comparison.

What this demonstrates
----------------------
- Ray-cast wheels following a non-flat surface via ``VehicleScene``: the chassis
  z / pitch vary as the car crosses the bumps; the suspension absorbs them.
- The dual_scene kinematic-terrain pattern (the productized ``--road-raycast-only``
  server flag does the same for UE maps). Because dual_scene keeps the terrain in
  a SEPARATE raycast scene, a collision-free VISUAL copy is added to the main
  scene purely so the camera/viewer can see the bumps.

The terrain is x-PERIODIC and the car is wrapped back one period each time it
crosses one, so with a viewer it drives **forever** (a side-following camera
trails the car) until you press **ESC** (or Ctrl+C). The wrap is seamless: the
surface repeats exactly, so neither the car nor the camera notices.

Run
---
    python -m genesis_vehicle.samples.terrain_drive --viewer   # drive forever, ESC to stop
    python -m genesis_vehicle.samples.terrain_drive --native   # Genesis 3D viewer (orbit)
    python -m genesis_vehicle.samples.terrain_drive            # headless, capped at --duration
    python -m genesis_vehicle.samples.terrain_drive --viewer --amp 1.0   # extreme relief
    python -m genesis_vehicle.samples.terrain_drive --rigid-terrain      # single_scene A/B: slower

Note: ``--viewer`` opens the cv2 HUD window (placed off-corner + kept on top).
``--native`` opens the Genesis OpenGL viewer — NOTE it may not surface a window
under WSLg / some pyglet setups (it runs but no window shows); use ``--viewer``
there.
"""

from __future__ import annotations

import sys, pathlib
_SDK_PARENT = str(pathlib.Path(__file__).resolve().parents[2])
if _SDK_PARENT not in sys.path:
    sys.path.insert(0, _SDK_PARENT)

import argparse
import os
import tempfile
import time

import numpy as np
import torch
import genesis as gs

from genesis_vehicle import (
    VehicleScene, car_4w_rwd_ackermann, __version__ as sdk_version,
)
from genesis_vehicle.samples import _hud

URDF_PATH = os.path.join(os.path.dirname(__file__), "urdf", "car_4w.urdf")


def make_bumpy_terrain(period=100.0, width=60.0, res=0.5, amp=0.7, ahead=35.0):
    """An x-PERIODIC undulating ground mesh (period ``period`` m), temp .obj.

    The car drives along +X forever; when it crosses one period it is wrapped
    back by ``period`` m (see the drive loop). That is seamless because the
    surface repeats EXACTLY every ``period`` — the frequencies are integer
    multiples of 2π/period. The mesh spans x ∈ [-10, period+ahead] so there is
    always terrain ahead of the car before it wraps."""
    k1 = 4 * 2 * np.pi / period          # 4 big rolling hills per period
    k2 = 11 * 2 * np.pi / period         # 11 secondary swells per period
    length = period + ahead + 10.0
    nx = int(length / res) + 1
    ny = int(width / res) + 1
    xs = np.linspace(-10.0, period + ahead, nx)
    ys = np.linspace(-width / 2, width / 2, ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    gz = (amp * np.sin(gx * k1)                  # big SMOOTH rolling hills (periodic)
          + 0.18 * amp * np.sin(gx * k2)         # secondary swells (periodic)
          + 0.28 * amp * np.cos(gy * 0.45))      # cross-slope undulation
    verts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j; b = a + 1; c = a + ny; d = c + 1
            faces.append([a, c, b]); faces.append([b, c, d])
    import trimesh
    m = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces))
    m.fix_normals()                # consistent winding → correct lighting
    _ = m.vertex_normals           # compute smooth normals so slopes shade
    fd, path = tempfile.mkstemp(suffix=".obj"); os.close(fd)
    m.export(path, include_normals=True)
    return path, len(faces)


def _vis_options():
    # Low ambient + a strong GRAZING key light so the bumps cast long shadows and
    # slopes go light/dark sharply (max relief). A weak cool fill from the
    # opposite side keeps shadowed slopes from going black. High ambient (the
    # other samples use 0.40) washes the bumps flat.
    return gs.options.VisOptions(
        shadow=True, ambient_light=(0.07, 0.07, 0.08),
        lights=[
            {"type": "directional", "dir": (-1.0, -0.35, -0.22),
             "color": (1.0, 0.95, 0.85), "intensity": 7.0},
            {"type": "directional", "dir": (0.7, 0.5, -0.5),
             "color": (0.35, 0.40, 0.55), "intensity": 1.5},
        ],
        background_color=(0.05, 0.07, 0.10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true",
                    help="Render a chase-cam HUD each step (needs opencv-python).")
    ap.add_argument("--rigid-terrain", action="store_true",
                    help="Load the terrain as a Rigid mesh in single_scene mode "
                         "instead of dual_scene Kinematic raycast (A/B: the rigid "
                         "mesh is re-fit in the raycaster BVH every step → slower). "
                         "HEAVY: a full-resolution non-convex mesh as a rigid "
                         "collider is the pathological case the dual_scene path "
                         "exists to avoid; the SDK now REFUSES it (mesh-guard) "
                         "rather than risk an out-of-memory crash. Decimate the "
                         "terrain (raise --res-equivalent) to actually run it.")
    ap.add_argument("--gpu", action="store_true",
                    help="Opt into the GPU backend (default: CPU — faster at n_envs=1).")
    ap.add_argument("--amp", type=float, default=0.7,
                    help="Terrain undulation amplitude (m). Default 0.7 (severe); "
                         "try 1.0+ for extreme, 0.2 for gentle.")
    ap.add_argument("--duration", type=float, default=20.0,
                    help="Headless run length (s). With --viewer/--native it "
                         "drives FOREVER until you press ESC (or Ctrl+C).")
    ap.add_argument("--native", action="store_true",
                    help="Open the Genesis native 3D viewer (orbit with the mouse) "
                         "instead of the cv2 HUD window. Stays open after the drive "
                         "until you close it (Ctrl+C).")
    args = ap.parse_args()
    if args.native:
        args.viewer = False        # --native uses the Genesis viewer, not the cv2 HUD

    rigid = args.rigid_terrain
    mode = "single_scene" if rigid else "dual_scene"
    mode_label = "RIGID terrain (single_scene)" if rigid else \
                 "KINEMATIC raycast terrain (dual_scene)"
    view = "  (native viewer)" if args.native else ("  (cv2 viewer)" if args.viewer else "")
    print(f"genesis_vehicle v{sdk_version}  |  terrain_drive  ({mode_label}){view}")

    need_render = args.viewer or args.native

    # car_4w_rwd_ackermann recommends dt≈1/48 with substeps=10 (internal ~2 ms).
    cfg = car_4w_rwd_ackermann(URDF_PATH, stability="control")
    DT = cfg.recommended_dt

    VehicleScene.init_backend("gpu" if args.gpu else "cpu")
    # --native: camera_pos doubles as the follow OFFSET for viewer.follow_entity
    # below (Genesis adds it to the followed entity's position every update).
    viewer_opts = gs.options.ViewerOptions(
        camera_pos=(0.0, -13.0, 2.8),
        camera_lookat=(0.0, 0.0, 0.7)) if args.native else None
    vs = VehicleScene(
        raycast_mode=mode,
        dt=DT, substeps=10, n_envs=1,
        vis_options=_vis_options(), show_viewer=args.native,
        viewer_options=viewer_opts)
    if args.viewer and not _hud.have_cv2():
        print("WARN: --viewer needs opencv-python. Continuing headless.")
        args.viewer = False
        need_render = args.native

    WRAP_PERIOD = 100.0    # terrain repeats every this many metres → seamless wrap
    terrain_obj, n_faces = make_bumpy_terrain(amp=args.amp, period=WRAP_PERIOD)
    vis_obj = None         # coarse VISUAL terrain copy (viewer modes only)
    # Earthy Rough surface so the slopes catch the directional light (a bare
    # default surface + high ambient is what made the bumps look flat).
    terrain_surface = gs.surfaces.Rough(color=(0.52, 0.46, 0.34))

    if rigid:
        # single_scene: the rigid mesh is itself the wheel-raycast target, so it
        # is re-fit into the raycaster BVH every step (the slow A/B path). A
        # full-resolution non-convex mesh as a rigid collider is exactly the
        # OOM-risk case the SDK's mesh-guard refuses — catch it and exit cleanly
        # instead of crashing (decimate the terrain to actually run this path).
        print(f"WARNING: --rigid-terrain builds a {n_faces}-face non-convex mesh "
              f"as a RIGID collider (the heavy path). The SDK mesh-guard will "
              f"refuse it above 1000 faces.")
        try:
            vs.add_static(
                morph=gs.morphs.Mesh(file=terrain_obj, fixed=True, align=False,
                                     convexify=False, decimate=False),
                material=gs.materials.Rigid(friction=1.0, sdf_cell_size=10000.0),
                surface=terrain_surface, name="terrain")
        except ValueError as e:
            print(f"\n[mesh-guard] {e}")
            os.unlink(terrain_obj)
            return
    else:
        # dual_scene: bumpy ground as a Kinematic raycast target (exact surface,
        # BVH static-skipped). collision=False → it lives ONLY in the raycast
        # scene, so the wheels sense it but nothing collides with it.
        vs.add_static(
            morph=gs.morphs.Mesh(file=terrain_obj, fixed=True, align=False,
                                 collision=False, visualization=True, convexify=False),
            collision=False, surface=terrain_surface, name="terrain")
        # dual_scene keeps that terrain in the SEPARATE raycast scene, so the
        # main-scene camera/viewer would not see it. Add a collision-free VISUAL
        # copy to the main scene purely for rendering (no physics, no raycast).
        # The copy is COARSER (res 1.0 vs 0.5, ~17k faces vs ~70k): the offscreen
        # raster is geometry-bound (~15 ms/frame at 70k faces — alone over the
        # 20.8 ms real-time budget; pixel count barely matters), while the coarse
        # grid deviates ≤ ~5 mm from the exact full-res raycast surface the
        # wheels ride — invisible, and the HUD/native view stays real-time.
        if need_render:
            vis_obj, _ = make_bumpy_terrain(amp=args.amp, period=WRAP_PERIOD,
                                            res=1.0)
            vs.add_dynamic(
                gs.morphs.Mesh(file=vis_obj, fixed=True, align=False,
                               collision=False, visualization=True, convexify=False),
                physics=False, wheel_raycast=False,
                material=gs.materials.Rigid(), surface=terrain_surface)

    veh = vs.add_vehicle(URDF_PATH, cfg=cfg, pos=(0.0, 0.0, 1.0),
                         material=gs.materials.Rigid(friction=1.0))

    cam = None
    if args.viewer:
        # 1024x576 (not 720p): the offscreen raster + HUD overlay + imshow cost
        # ~25 ms/frame at 1280x720 — over the 20.8 ms real-time step budget on
        # its own. At 1024x576 and 24 fps (every 2nd step) the whole loop fits
        # the budget, so the 1x pacer below holds true real time.
        cam = vs.add_camera(res=(1024, 576), pos=(-8.0, -6.0, 4.0),
                            lookat=(0.0, 0.0, 1.0), up=(0.0, 0.0, 1.0),
                            fov=55, near=0.1, far=200.0, GUI=False)

    vs.build()

    # Native-viewer follow must happen INSIDE viewer.update() — the same call
    # that pushes the fresh chassis pose to the renderer (and then sleeps for
    # the realtime pacer). Setting the camera from the drive loop instead
    # (after vs.step() returns) leaves it one physics step behind the car for
    # most of each frame: at 7 m/s that is a ~15 cm camera-vs-car offset that
    # flickers at the viewer's draw rate — the car visibly "trembles" while
    # the cv2 --viewer (synchronous set_pose + render) looks rock solid.
    # follow_entity closes the gap; no smoothing, so the seamless 100 m wrap
    # snap stays invisible (a smoothed camera would sweep 100 m backwards).
    follow_fallback = False
    if args.native:
        try:
            vs.viewer.follow_entity(veh.entity_main)
        except Exception:
            follow_fallback = True     # older Genesis: per-step follow below

    def _follow_native():
        # Fallback camera follow for Genesis versions without follow_entity.
        # Updated from the drive loop → lags the car by up to one step (the
        # trembling this fallback tolerates is what follow_entity fixes).
        if not (args.native and follow_fallback):
            return
        p = veh.get_pos()[0].cpu().numpy()
        try:
            vs.viewer.set_camera_pose(
                pos=p + np.array([0.0, -13.0, 2.8]),
                lookat=p + np.array([0.0, 0.0, 0.7]))
        except Exception:
            pass

    n_settle = int(1.0 / DT)
    n_drive = int(args.duration / DT)
    render_every = max(1, round((1.0 / 24.0) / DT))   # HUD ~24 fps (real-time budget)
    hud_perf = _hud.PerfMeter(window=60)

    def _hud_render(t_sim, throttle, p, speed):
        # p / speed are passed in from the drive loop (already read there this
        # step) — re-reading them here would add 2 more engine reads per frame,
        # which the real-time budget can't spare.
        if not args.viewer:
            return True
        # Follow ALONGSIDE the car (camera to its side, not behind) so the
        # chassis pitch/heave over the bumps is seen in profile against the
        # terrain skyline.
        cam.set_pose(pos=p + np.array([0.0, -13.0, 2.8]),
                     lookat=p + np.array([0.0, 0.0, 0.7]), up=np.array([0.0, 0.0, 1.0]))
        frame = _hud.render_hud_frame(
            cam, title=f"terrain_drive  v{sdk_version}",
            lines=[f"t={t_sim:5.2f}s  throttle={throttle:+.2f}",
                   f"pos=({p[0]:+6.2f},{p[1]:+6.2f},{p[2]:+5.2f})  speed={speed:4.1f}",
                   "[ESC] quit"],
            perf_ms=hud_perf.ms_per_step())
        return _hud.cv2_show("genesis_vehicle terrain_drive", frame)

    # Settle onto the terrain.
    for step in range(n_settle):
        veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        vs.step(); hud_perf.tick()

    # Drive FOREVER. With a viewer: ESC in the window (or Ctrl+C) stops it.
    # Headless (no viewer) is capped at --duration so it can't hang. Whenever the
    # car crosses one terrain period it is wrapped back by it — seamless, because
    # the surface repeats exactly, so neither the car nor the camera notices.
    infinite = args.viewer or args.native
    x0 = float(veh.get_pos()[0][0])
    z_lo, z_hi, wraps = 1e9, -1e9, 0
    torch.cuda.synchronize() if args.gpu else None
    t_start = time.perf_counter()
    user_quit = False
    step = 0
    pace_next = time.perf_counter()   # real-time (1x) pacer for the cv2 HUD
    if infinite:
        print("\n[driving — press ESC in the window, or Ctrl+C here, to stop]")
    # ONE pos read + ONE vel read per step, reused by the governor, centerline
    # steer, z stats, wrap check and HUD. The naive version re-read them at
    # each use (~6 engine reads/step ≈ +3 ms) — enough to blow the 20.8 ms
    # real-time budget with the HUD on.
    p_np = veh.get_pos()[0].cpu().numpy()
    try:
        while True:
            # Speed governor: coast above ~7 m/s so the car rides the hills at a
            # steady pace forever instead of accelerating until it launches off a
            # crest and tumbles (with kinematic terrain there's no floor to catch
            # it). Plus a gentle pull to the centerline so it doesn't drift off
            # the side (+steer = right; left of centre → +). p_np is last step's
            # pose — a 1-step-stale centerline correction is imperceptible.
            v_np = veh.get_vel()[0].cpu().numpy()
            speed = float((v_np[0] ** 2 + v_np[1] ** 2) ** 0.5)
            thr = 0.45 if speed < 7.0 else 0.0
            steer = max(-0.18, min(0.18, 0.06 * float(p_np[1])))
            veh.set_inputs(throttle=thr, brake=0.0, steer=steer)
            vs.step(); hud_perf.tick()
            p_np = veh.get_pos()[0].cpu().numpy()
            zc = float(p_np[2]); z_lo = min(z_lo, zc); z_hi = max(z_hi, zc)
            if float(p_np[0]) - x0 >= WRAP_PERIOD:
                pos = veh.entity_main.get_pos().clone()
                pos[..., 0] -= WRAP_PERIOD          # absolute pose, keep velocity
                veh.entity_main.set_pos(pos, zero_velocity=False, relative=False)
                p_np[0] -= WRAP_PERIOD
                wraps += 1
            if step % render_every == 0:
                _follow_native()
                if not _hud_render(step * DT, thr, p_np, speed):  # False = ESC
                    user_quit = True
                    break
            if args.viewer:
                # Real-time (1x) pacing for the cv2 HUD. --native gets this
                # from the Genesis viewer's realtime_factor pacer (inside
                # vs.step()); the cv2 path has no viewer, so it would free-run
                # on a fast machine and looked sped-up. Debt must CARRY OVER
                # between steps: the HUD renders every render_every-th step, so
                # a render step overruns the budget and the steps between pay
                # it back — resyncing on every overrun locks in ~0.8x. Only a
                # real hitch (>0.25 s behind) resyncs, to avoid a catch-up rush.
                pace_next += DT
                now = time.perf_counter()
                if pace_next > now:
                    time.sleep(pace_next - now)
                elif now - pace_next > 0.25:
                    pace_next = now
            step += 1
            if not infinite and step >= n_drive:
                break
    except KeyboardInterrupt:
        user_quit = True
        print("\n  stopped.")
    torch.cuda.synchronize() if args.gpu else None
    wall = time.perf_counter() - t_start
    _hud.cv2_cleanup()

    p = veh.get_pos()[0].cpu().numpy()
    z_var = (z_hi - z_lo) if z_hi > z_lo else 0.0
    total_dist = wraps * WRAP_PERIOD + (float(p[0]) - x0)
    on_terrain = -4.0 < float(p[2]) < 6.0
    undulates = z_var > 0.03

    r_ms, r_n = _hud.bench_render(cam, n=20) if cam is not None else (None, None)
    _hud.print_perf_summary(
        sample=f"terrain_drive  (v{sdk_version})  [{mode_label}]",
        completed=not user_quit,
        n_done=step, n_target=step, wall=wall,
        render_ms=r_ms, render_n=r_n,
        extra=[
            f"terrain    : {n_faces} faces, amp {args.amp:.2f} m, period {WRAP_PERIOD:.0f} m",
            f"drove      : {total_dist:+.1f} m over {wraps} wrap(s)   "
            f"({'OK' if total_dist > 3 else 'FAIL'} drive)",
            f"chassis z  : {float(p[2]):+.2f} m   ({'OK' if on_terrain else 'FAIL'} on terrain)",
            f"z undulation (max-min) : {z_var:.3f} m   "
            f"({'OK' if undulates else 'FAIL'} follows bumps)",
        ],
    )
    os.unlink(terrain_obj)
    if vis_obj is not None:
        os.unlink(vis_obj)


if __name__ == "__main__":
    main()
