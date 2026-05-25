"""city_traffic_ego.py — flat highway scenario: ego + traffic vehicles.

A multi-lane straight road on a flat plane. One **ego** (red, sporty
AWD) plus N traffic agents (mix of FWD sedans, RWD coupes, yellow
trucks). All driving in +X under a simple lane-keeping controller
(P on lateral offset from target lane) + constant throttle. Top-down
camera frames the highway segment.

The sample is built on top of ``MultiVehiclePhysics`` so it supports
both the visual single-env case AND the L2 × L3 combined case via
``--n_envs N`` — exactly the autonomous-driving "K vehicles per
scenario × N parallel scenarios" pattern that motivates L2 × L3
batching. See [`../docs/batching.md`](../docs/batching.md).

Vehicle composition (default)
-----------------------------
  ego        1 × AWD car, RED sporty body
  traffic    3 × RWD blue coupe
             3 × FWD compact sedan (smaller, dark red)
             1 × Truck yellow 6-wheel
  Total      8 vehicles per env

Layout (default 4 lanes, ~3.5 m wide each)
------------------------------------------
   lane 0   y = +5.25   ┃
   lane 1   y = +1.75   ┃    ego drives lane 1 by default
   lane 2   y = -1.75   ┃
   lane 3   y = -5.25   ┃

Run
---
    python -m genesis_vehicle.samples.city_traffic_ego
    python -m genesis_vehicle.samples.city_traffic_ego --duration 12 --viewer
    python -m genesis_vehicle.samples.city_traffic_ego --n_envs 16
        ↑ 16 parallel scenarios, L2 × L3 combined batching

Timing block always prints at the end (single CUDA sync before/after the
drive loop — zero per-step overhead).
"""

from __future__ import annotations

# Bootstrap: allow `python path/to/this_file.py` (and `python -m ...`) both.
import sys, pathlib
_SDK_PARENT = str(pathlib.Path(__file__).resolve().parents[2])
if _SDK_PARENT not in sys.path:
    sys.path.insert(0, _SDK_PARENT)

import argparse
import math
import os
import tempfile
import time

import numpy as np
import torch
import genesis as gs

from genesis_vehicle import (
    MultiVehiclePhysics, VehicleInputs,
    car_4w_fwd_ackermann, car_4w_rwd_ackermann, car_4w_awd_ackermann,
    truck_6w_partial_ackermann,
    __version__ as sdk_version,
)
from genesis_vehicle.scene_helpers import make_wheel_raycaster

# Reuse the parametric URDF generators from road_loop.
from genesis_vehicle.samples.road_loop import (
    _car_urdf, _truck_urdf, _save_urdf,
)


# ---------------------------------------------------------------------------
# Scenario constants
# ---------------------------------------------------------------------------

LANE_Y = [+5.25, +1.75, -1.75, -5.25]        # 4 lanes, center y
LANE_W = 3.5                                  # lane width (visual only)
ROAD_LENGTH = 200.0                           # plane size in X
ROAD_HALF_WIDTH = 12.0                        # plane size in Y

# Vehicle catalogue. Each entry: (name, urdf_builder, preset_fn, wheelbase, kind_color_label)
# Ego gets its own URDF entry so it's visually distinct (RED + slightly larger body).
EGO_KIND = ("EGO_AWD",
            lambda: _car_urdf("ego_awd",  (0.95, 0.10, 0.10, 1.0),    # bright RED
                              body_size=(4.8, 1.9, 1.3),
                              cabin_size=(2.4, 1.7, 0.6)),
            car_4w_awd_ackermann, 2.7)

TRAFFIC_KINDS = [
    ("RWD",   lambda: _car_urdf("traffic_rwd", (0.15, 0.35, 0.85, 1.0),
                                 body_size=(4.5, 1.8, 0.7),
                                 cabin_size=(2.2, 1.6, 0.4)),
              car_4w_rwd_ackermann, 2.7),
    ("FWD",   lambda: _car_urdf("traffic_fwd", (0.55, 0.10, 0.10, 1.0),  # dark red
                                 body_size=(3.8, 1.6, 1.0),
                                 cabin_size=(2.0, 1.5, 0.5)),
              car_4w_fwd_ackermann, 2.7),
    ("Truck", lambda: _truck_urdf((0.95, 0.80, 0.15, 1.0)),
              truck_6w_partial_ackermann, 3.6),
]


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

def _add_lane_markers(scene, n_dashes: int = 60):
    """White dashed lane separators between adjacent lanes.

    3 separators (between 4 lanes), each a row of small white boxes
    spaced ~3.5 m apart along the road's +X direction."""
    for i in range(len(LANE_Y) - 1):
        y = (LANE_Y[i] + LANE_Y[i + 1]) / 2.0
        for d in range(n_dashes):
            x = -ROAD_LENGTH / 2 + (d + 0.5) * (ROAD_LENGTH / n_dashes)
            scene.add_entity(
                gs.morphs.Box(size=(1.5, 0.20, 0.05), pos=(x, y, 0.03),
                              fixed=True, collision=False),
                surface=gs.surfaces.Plastic(color=(1.0, 1.0, 1.0, 1.0)),
            )
    # Road edges (solid white lines, brighter).
    for y in (LANE_Y[0] + LANE_W / 2, LANE_Y[-1] - LANE_W / 2):
        scene.add_entity(
            gs.morphs.Box(size=(ROAD_LENGTH, 0.30, 0.05), pos=(0, y, 0.03),
                          fixed=True, collision=False),
            surface=gs.surfaces.Plastic(color=(1.0, 1.0, 1.0, 1.0)),
        )


def _make_lane_keeper_steer(vehicle_y: torch.Tensor, target_y: float,
                             vehicle_yaw: torch.Tensor) -> torch.Tensor:
    """Simple lane-keeping P controller. Returns steer ∈ [-0.3, 0.3] rad.

    Args
    ----
    vehicle_y    : (N,) current Y position of each env's vehicle.
    target_y     : float — desired lane center.
    vehicle_yaw  : (N,) current yaw (rad).
    """
    lateral_err = vehicle_y - target_y      # > 0 if to the LEFT of target
    # heading_err: deviation from "facing +X" — small angles only.
    heading_err = vehicle_yaw
    # Desired steer = - (K_lat * lateral + K_yaw * heading), ISO 8855
    # (+steer = right turn). Vehicle is left of target → steer right → +steer.
    K_lat = 0.15
    K_yaw = 0.8
    steer = K_lat * lateral_err + K_yaw * heading_err
    return torch.clamp(steer, -0.3, 0.3)


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw component of (..., 4) quaternion (w, x, y, z)."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_envs",   type=int,   default=1,
                    help="Parallel envs (L3). Default 1 = visual scene; "
                         "> 1 enables L2 × L3 combined batching.")
    ap.add_argument("--duration", type=float, default=10.0,
                    help="Drive duration in seconds (default 10).")
    ap.add_argument("--ego_throttle", type=float, default=0.4,
                    help="Ego constant throttle (default 0.4).")
    ap.add_argument("--viewer", action="store_true",
                    help="Render top-down camera per step.")
    args = ap.parse_args()

    print(f"genesis_vehicle v{sdk_version}  |  city_traffic_ego")
    print(f"  scene  : 4-lane highway, flat plane ({ROAD_LENGTH:.0f} × "
          f"{2*ROAD_HALF_WIDTH:.0f} m)")
    print(f"  fleet  : 1 ego (RED AWD) + 7 traffic (3 RWD blue, "
          f"3 FWD small dark-red, 1 yellow truck)")
    print(f"  n_envs : {args.n_envs}   "
          f"({'L2 only' if args.n_envs == 1 else f'L2 × L3 = {8 * args.n_envs} total vehicles batched'})")

    # ------------------------------------------------------------------
    # Stamp URDFs to tempdir (ego + 3 traffic kinds).
    # ------------------------------------------------------------------
    tmpdir = tempfile.mkdtemp(prefix="gv_city_traffic_")
    ego_urdf = _save_urdf(EGO_KIND[1](), tmpdir, "ego_awd")
    traffic_urdfs = [
        _save_urdf(uf(), tmpdir, name.lower())
        for (name, uf, _p, _wb) in TRAFFIC_KINDS
    ]

    # ------------------------------------------------------------------
    # Scene + Genesis init.
    # ------------------------------------------------------------------
    gs.init(backend=gs.gpu, logging_level="warning")
    DT = 0.02
    cam_h = 55.0
    from genesis_vehicle.samples import _hud
    if args.viewer and not _hud.have_cv2():
        print("WARN: --viewer needs opencv-python. Continuing headless.")
        args.viewer = False

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=20),
        rigid_options=gs.options.RigidOptions(
            dt=DT, enable_collision=True,
            enable_self_collision=False, enable_joint_limit=True,
        ),
        vis_options=gs.options.VisOptions(
            shadow=True, ambient_light=(0.40, 0.40, 0.40),
            background_color=(0.05, 0.07, 0.10),
            # If n_envs > 1, lay parallel scenarios out on a grid so the
            # camera can show all of them. n_envs=1 → no offset.
            env_separate_rigid=(args.n_envs > 1),
        ),
        show_viewer=False,    # --viewer uses cv2 HUD instead
    )
    scene.add_entity(
        gs.morphs.Plane(pos=(0, 0, 0),
                         plane_size=(ROAD_LENGTH, 2 * ROAD_HALF_WIDTH + 4.0)),
        material=gs.materials.Rigid(friction=1.0),
    )
    _add_lane_markers(scene, n_dashes=60)

    # ------------------------------------------------------------------
    # Spawn ego + traffic.
    # Layout: ego at x=0 lane 1. Traffic at x ∈ [-30, +30], scattered across lanes.
    # ------------------------------------------------------------------
    cfg_ego     = EGO_KIND[2](ego_urdf, stability="control")
    cfg_traffic = [tk[2](traffic_urdfs[i], stability="control")
                   for i, tk in enumerate(TRAFFIC_KINDS)]

    vehicles = []                # list[(entity, sensor, cfg)]
    target_lanes = []            # parallel list: target Y per vehicle for lane-keeping
    labels = []                  # parallel list: label per vehicle

    def spawn(urdf_path: str, cfg, pos_xyz, target_y, label):
        morph = gs.morphs.URDF(file=urdf_path, pos=pos_xyz)
        ent = scene.add_entity(morph, material=gs.materials.Rigid(friction=1.0))
        sens = make_wheel_raycaster(scene, ent, urdf_path)
        vehicles.append((ent, sens, cfg))
        target_lanes.append(target_y)
        labels.append(label)

    # Ego — center-ish lane (lane index 1), x = 0.
    spawn(ego_urdf, cfg_ego,
          pos_xyz=(0.0, LANE_Y[1], 1.0),
          target_y=LANE_Y[1],
          label="ego")

    # Traffic — fixed scripted layout for reproducibility.
    # (lane_idx, x_offset, kind_index in TRAFFIC_KINDS, label)
    TRAFFIC_LAYOUT = [
        (0, +25.0, 0, "rwd_0"),    # RWD ahead-left
        (0, -15.0, 0, "rwd_1"),    # RWD behind-left
        (1, +18.0, 0, "rwd_2"),    # RWD ahead-same-lane
        (2, +10.0, 1, "fwd_0"),    # FWD ahead-right
        (2, -22.0, 1, "fwd_1"),    # FWD behind-right
        (3, +30.0, 1, "fwd_2"),    # FWD far-right ahead
        (3, -10.0, 2, "truck_0"),  # Truck behind-far-right
    ]
    for lane_i, x_off, kind_i, label in TRAFFIC_LAYOUT:
        urdf  = traffic_urdfs[kind_i]
        cfg_t = cfg_traffic[kind_i]
        spawn(urdf, cfg_t,
              pos_xyz=(x_off, LANE_Y[lane_i], 1.0),
              target_y=LANE_Y[lane_i],
              label=label)

    K_total = len(vehicles)

    # ------------------------------------------------------------------
    # Offscreen camera — produces image tensors for recording / inspection.
    # The interactive viewer window (when --viewer) is configured separately
    # via the viewer_options above; this offscreen camera is independent
    # and runs whether or not the viewer is on.
    # ------------------------------------------------------------------
    # Camera only when --viewer. Having a camera in the scene adds per-step
    # renderer-state-sync cost inside scene.step(), so skipping it in
    # headless gives a clean physics-only ms/step number. With
    # env_separate_rigid (args.n_envs > 1 + --viewer) the render returns a
    # per-env stack and the HUD tiles + downsizes cells, so a modest per-
    # cell res is enough; with single env, keep 1080p.
    cam = None
    if args.viewer:
        cam_res = (640, 360) if args.n_envs > 1 else (1920, 1080)
        cam = scene.add_camera(
            res=cam_res,
            pos=(0.0, 0.0, cam_h), lookat=(0.0, 0.0, 0.0),
            up=(1.0, 0.0, 0.0),       # +X is "up" on screen (driving away from viewer)
            fov=60, near=0.1, far=cam_h * 4, GUI=False,
        )

    # ------------------------------------------------------------------
    # Build + MultiVehiclePhysics (n_envs = L3 axis; K = L2 axis = K_total here).
    # ------------------------------------------------------------------
    scene.build(
        n_envs=args.n_envs,
        env_spacing=(ROAD_LENGTH + 20.0, 2 * ROAD_HALF_WIDTH + 20.0) if args.n_envs > 1 else (0.0, 0.0),
        n_envs_per_row=max(1, int(round(math.sqrt(args.n_envs)))),
    )
    mphys = MultiVehiclePhysics(scene, vehicles, n_envs=args.n_envs)
    device = vehicles[0][0].get_pos().device
    print(f"  L2 groups : {mphys.n_kinds} kinds  "
          f"(K per kind = {[k.K for k in mphys.kinds]})")

    target_y_t = [torch.full((args.n_envs,), float(t), device=device,
                              dtype=gs.tc_float) for t in target_lanes]

    # Per-vehicle, per-env throttle. Ego = constant; traffic = different
    # constant per kind + small per-env noise (so envs diverge under L3).
    g = torch.Generator(device=device).manual_seed(7)
    base_throttle = {"ego": args.ego_throttle,
                     "rwd": 0.35, "fwd": 0.28, "truck": 0.22}
    throttle_t = []
    for lab in labels:
        key = "ego" if lab == "ego" else lab.split("_")[0]
        base = base_throttle[key]
        # +/- 10% per-env noise to keep envs from being identical.
        noise = 0.10 * base * (torch.rand(args.n_envs, generator=g, device=device) - 0.5) * 2
        throttle_t.append(torch.clamp(torch.full((args.n_envs,), base, device=device) + noise,
                                       0.0, 1.0).to(dtype=gs.tc_float))

    # ------------------------------------------------------------------
    # Settle (brake held).
    # ------------------------------------------------------------------
    print(f"\n[settle 1.5 s]")
    settle_in = [VehicleInputs(throttle=0.0, brake=1.0, steer=0.0)
                 for _ in vehicles]
    for _ in range(int(1.5 / DT)):
        mphys.step(settle_in)
        scene.step()
        if args.viewer:
            cam.render()

    # ------------------------------------------------------------------
    # Drive — lane-keeping P + constant per-vehicle throttle.
    # ------------------------------------------------------------------
    n_steps = int(args.duration / DT)
    print(f"[drive {n_steps} steps  ego throttle={args.ego_throttle:.2f}]\n")

    # Always-on timing — single sync before/after, zero per-step overhead.
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    render_every = max(1, int(0.04 / DT))      # ~25 fps
    hud_perf = _hud.PerfMeter(window=60)

    def _hud_render(step: int):
        # Headless = pure physics (cam is None); viewer = render + HUD.
        if not args.viewer:
            return True
        # env 0 ego state.
        ego_ent = vehicles[0][0]
        ep = ego_ent.get_pos()[0].cpu().numpy()
        ev = ego_ent.get_vel()[0].cpu().numpy()
        ego_speed = float(np.linalg.norm(ev[:2]))
        frame = _hud.render_hud_frame(
            cam,
            title=f"city_traffic_ego  v{sdk_version}",
            lines=[
                f"step {step:>4}/{n_steps}    "
                f"{K_total} vehicles × {args.n_envs} envs = "
                f"{K_total * args.n_envs} total batched",
                f"ego (env 0): pos=({ep[0]:+6.2f}, {ep[1]:+5.2f})  "
                f"lane Δy={ep[1] - target_lanes[0]:+5.3f}  speed={ego_speed:4.1f} m/s",
                f"L2 kinds: {mphys.n_kinds}    "
                f"K per kind = {[k.K for k in mphys.kinds]}",
                "[ESC] quit",
            ],
            perf_ms=hud_perf.ms_per_step(),
            grid_per_row=max(1, int(round(math.sqrt(args.n_envs)))) if args.n_envs > 1 else None,
        )
        return _hud.cv2_show("genesis_vehicle city_traffic_ego", frame)

    user_quit = False
    for step in range(n_steps):
        # Read state once per step. Multi-entity reads aren't auto-batched
        # at the entity-API level — loop is short (K_total = 8).
        inputs = []
        for v_i, (ent, _sens, _cfg) in enumerate(vehicles):
            pos  = ent.get_pos()         # (n_envs, 3)
            quat = ent.get_quat()        # (n_envs, 4)
            yaw  = _yaw_from_quat(quat)
            steer = _make_lane_keeper_steer(pos[:, 1], target_lanes[v_i], yaw)
            inputs.append(VehicleInputs(
                throttle=throttle_t[v_i],
                brake=torch.zeros(args.n_envs, device=device, dtype=gs.tc_float),
                steer=steer,
            ))
        mphys.step(inputs)
        scene.step()
        hud_perf.tick()
        if step % render_every == 0:
            if not _hud_render(step):
                user_quit = True
                break

    torch.cuda.synchronize()
    wall = time.perf_counter() - t_start
    _hud.cv2_cleanup()
    n_done = step + 1 if user_quit else n_steps

    # ------------------------------------------------------------------
    # Report.
    # ------------------------------------------------------------------
    print(f"=== AFTER {args.duration:.1f}s  (env 0) ===")
    print(f"  {'label':<8}  {'pos':<22}  {'lane Δy':>8}  {'speed':>7}")
    print(f"  {'-'*8}  {'-'*22}  {'-'*8}  {'-'*7}")
    for v_i, (ent, _s, _c) in enumerate(vehicles):
        p = ent.get_pos()[0].cpu().numpy()
        v = ent.get_vel()[0].cpu().numpy()
        speed = float(np.linalg.norm(v[:2]))
        dy = float(p[1]) - target_lanes[v_i]
        print(f"  {labels[v_i]:<8}  ({p[0]:+7.2f}, {p[1]:+6.2f}, {p[2]:.2f})  "
              f"{dy:+8.3f}  {speed:7.2f} m/s")

    r_ms, r_n = _hud.bench_render(cam, n=20) if cam is not None else (None, None)
    _hud.print_perf_summary(
        sample=f"city_traffic_ego  (v{sdk_version})",
        completed=not user_quit,
        n_done=n_done, n_target=n_steps, wall=wall,
        batch=args.n_envs * K_total, batch_label="vehicle",
        render_ms=r_ms, render_n=r_n,
        extra=[
            f"L2 kinds   : {mphys.n_kinds}   K per kind = {[k.K for k in mphys.kinds]}",
            f"L3 envs    : {args.n_envs}    (batch = n_envs x K = "
            f"{args.n_envs} x {K_total} = {args.n_envs*K_total})",
        ],
    )


if __name__ == "__main__":
    main()
