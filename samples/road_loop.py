"""road_loop.py — multi-vehicle loop drive visual demo.

Drops 4 KINDS of vehicles (FWD sedan / RWD coupe / AWD SUV / 6-wheel truck),
``--n_per_kind`` of each, evenly spaced around a circular track. Each
vehicle gets a constant Ackermann steering angle sized to the track radius
so the whole fleet orbits indefinitely under constant throttle. A
top-down chase camera shows the whole scene.

Vehicle identification (shape + color)
--------------------------------------
  FWD     red    compact sedan        ~3.8 × 1.6 × 1.0 m
  RWD     blue   low sports coupe     ~4.5 × 1.8 × 0.7 m
  AWD     green  tall SUV             ~4.3 × 2.0 × 1.4 m
  Truck   yellow 6-wheel cargo truck  ~6.0 × 2.2 × 2.5 m

Run
---
    python -m genesis_vehicle.samples.road_loop
    python -m genesis_vehicle.samples.road_loop --n_per_kind 8
    python -m genesis_vehicle.samples.road_loop --duration 30 --viewer

For a benchmark of `n_envs` batching speedup see
``samples/perf_vectorization.py`` (a separate concern from this visual demo
since true n_envs batching requires a different scene build).
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

import numpy as np
import genesis as gs

import time

from genesis_vehicle import (
    VehicleScene,
    car_4w_fwd_ackermann, car_4w_rwd_ackermann, car_4w_awd_ackermann,
    truck_6w_partial_ackermann,
    __version__ as sdk_version,
)


# ---------------------------------------------------------------------------
# URDF generation (parametric — written to a tempdir at runtime)
# ---------------------------------------------------------------------------

def _car_urdf(name: str, color_rgba: tuple, body_size: tuple,
              cabin_size: tuple, wheelbase: float = 2.7, track: float = 1.5,
              wheel_r: float = 0.32, mass: float = 1200.0) -> str:
    """Render a 4-wheel car URDF. Same topology as samples/urdf/car_4w.urdf
    but with parameterized body box + color so callers can stamp out a
    visually distinct variant per drivetrain."""
    bx, by, bz = body_size
    cx, cy, cz = cabin_size
    half_wb = wheelbase / 2
    half_tk = track / 2
    rgba = " ".join(f"{c:.3f}" for c in color_rgba)
    ixx = mass * (by*by + bz*bz) / 12
    iyy = mass * (bx*bx + bz*bz) / 12
    izz = mass * (bx*bx + by*by) / 12

    def wheel_block(side: str, axle: str, y: float, signed_x: float) -> str:
        steer_block = ""
        wheel_parent = f"{side}_axle_carrier"
        if axle == "front":
            steer_block = f"""
  <joint name="{side}_steer_joint" type="revolute">
    <parent link="{side}_axle_carrier"/><child link="{side}_steer_link"/>
    <origin xyz="0 0 0"/>
    <axis xyz="0 0 -1"/>
    <limit lower="-0.7" upper="0.7" effort="500000.0" velocity="10.0"/>
    <dynamics damping="20.0" friction="0.0"/>
  </joint>
  <link name="{side}_steer_link">
    <inertial><origin xyz="0 0 0"/><mass value="5.0"/>
      <inertia ixx="0.05" iyy="0.05" izz="0.05" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>"""
            wheel_parent = f"{side}_steer_link"

        return f"""
  <joint name="{side}_suspension_joint" type="prismatic">
    <parent link="base_link"/><child link="{side}_axle_carrier"/>
    <origin xyz="{signed_x:+.3f} {y:+.3f} {wheel_r:.3f}"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.15" upper="0.15" effort="0" velocity="10.0"/>
    <dynamics damping="0.0" friction="0.0" stiffness="0.0"/>
  </joint>
  <link name="{side}_axle_carrier">
    <inertial><origin xyz="0 0 0"/><mass value="10.0"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>{steer_block}
  <joint name="{side}_wheel_joint" type="continuous">
    <parent link="{wheel_parent}"/><child link="{side}_wheel"/>
    <origin xyz="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit effort="0" velocity="200"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>
  <link name="{side}_wheel">
    <inertial><origin xyz="0 0 0"/><mass value="20.0"/>
      <inertia ixx="1.0" iyy="2.0" izz="1.0" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry><cylinder radius="{wheel_r:.3f}" length="0.20"/></geometry>
      <material name="wheel_dark"><color rgba="0.10 0.10 0.10 1.0"/></material>
    </visual>
  </link>"""

    return f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 {bz/2 + 0.3:.3f}"/>
      <mass value="{mass:.1f}"/>
      <inertia ixx="{ixx:.2f}" iyy="{iyy:.2f}" izz="{izz:.2f}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 {bz/2 + 0.3:.3f}"/>
      <geometry><box size="{bx:.3f} {by:.3f} {bz:.3f}"/></geometry>
      <material name="body"><color rgba="{rgba}"/></material>
    </visual>
    <visual>
      <origin xyz="-0.2 0 {bz + cz/2 + 0.3:.3f}"/>
      <geometry><box size="{cx:.3f} {cy:.3f} {cz:.3f}"/></geometry>
      <material name="cabin"><color rgba="0.18 0.18 0.22 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 {bz/2 + 0.5:.3f}"/>
      <geometry><box size="{bx:.3f} {by:.3f} {bz:.3f}"/></geometry>
    </collision>
  </link>
{wheel_block("front_left",  "front", +half_tk, +half_wb)}
{wheel_block("front_right", "front", -half_tk, +half_wb)}
{wheel_block("rear_left",   "rear",  +half_tk, -half_wb)}
{wheel_block("rear_right",  "rear",  -half_tk, -half_wb)}
</robot>
"""


def _truck_urdf(color_rgba=(0.95, 0.80, 0.15, 1.0)) -> str:
    """6-wheel truck. Yellow cargo box + light-gray cab + steered front axle."""
    rgba = " ".join(f"{c:.3f}" for c in color_rgba)

    def axle_set(name: str, x: float, y: float, steered: bool) -> str:
        steer_block = ""
        wheel_parent = f"{name}_axle_carrier"
        if steered:
            steer_block = f"""
  <joint name="{name}_steer_joint" type="revolute">
    <parent link="{name}_axle_carrier"/><child link="{name}_steer_link"/>
    <origin xyz="0 0 0"/>
    <axis xyz="0 0 -1"/>
    <limit lower="-0.55" upper="0.55" effort="500000.0" velocity="10.0"/>
    <dynamics damping="20.0" friction="0.0"/>
  </joint>
  <link name="{name}_steer_link">
    <inertial><origin xyz="0 0 0"/><mass value="10.0"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>"""
            wheel_parent = f"{name}_steer_link"
        return f"""
  <joint name="{name}_suspension_joint" type="prismatic">
    <parent link="base_link"/><child link="{name}_axle_carrier"/>
    <origin xyz="{x:+.3f} {y:+.3f} 0.40"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.20" upper="0.20" effort="0" velocity="10.0"/>
    <dynamics damping="0.0" friction="0.0" stiffness="0.0"/>
  </joint>
  <link name="{name}_axle_carrier">
    <inertial><origin xyz="0 0 0"/><mass value="20.0"/>
      <inertia ixx="0.2" iyy="0.2" izz="0.2" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>{steer_block}
  <joint name="{name}_wheel_joint" type="continuous">
    <parent link="{wheel_parent}"/><child link="{name}_wheel"/>
    <origin xyz="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit effort="0" velocity="200"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>
  <link name="{name}_wheel">
    <inertial><origin xyz="0 0 0"/><mass value="40.0"/>
      <inertia ixx="2.0" iyy="3.6" izz="2.0" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry><cylinder radius="0.40" length="0.32"/></geometry>
      <material name="wheel_dark"><color rgba="0.10 0.10 0.10 1.0"/></material>
    </visual>
  </link>"""

    return f"""<?xml version="1.0"?>
<robot name="truck_6w">
  <link name="base_link">
    <inertial><origin xyz="0 0 0.9"/><mass value="5000.0"/>
      <inertia ixx="3000" iyy="9000" izz="10000" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual>
      <origin xyz="-0.6 0 1.25"/>
      <geometry><box size="4.2 2.0 1.1"/></geometry>
      <material name="cargo"><color rgba="{rgba}"/></material>
    </visual>
    <visual>
      <origin xyz="2.15 0 1.55"/>
      <geometry><box size="1.3 2.0 1.6"/></geometry>
      <material name="cab"><color rgba="0.85 0.85 0.88 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 1.35"/>
      <geometry><box size="5.5 2.1 1.1"/></geometry>
    </collision>
  </link>
{axle_set("front_left",   1.80,  0.85, steered=True)}
{axle_set("front_right",  1.80, -0.85, steered=True)}
{axle_set("mid_left",     0.00,  0.95, steered=False)}
{axle_set("mid_right",    0.00, -0.95, steered=False)}
{axle_set("rear_left",   -1.80,  0.95, steered=False)}
{axle_set("rear_right",  -1.80, -0.95, steered=False)}
</robot>
"""


# ---------------------------------------------------------------------------
# Vehicle kind table
# ---------------------------------------------------------------------------

KINDS = [
    # (name,  color RGBA,                URDF builder lambda,                   preset fn,                   wheelbase, n_wheels)
    ("FWD",   (0.85, 0.15, 0.15, 1.0),  lambda: _car_urdf("car_fwd", (0.85, 0.15, 0.15, 1.0),
                                                          body_size=(3.8, 1.6, 1.0),
                                                          cabin_size=(2.0, 1.5, 0.5)),
              car_4w_fwd_ackermann,    2.7, 4),
    ("RWD",   (0.15, 0.35, 0.85, 1.0),  lambda: _car_urdf("car_rwd", (0.15, 0.35, 0.85, 1.0),
                                                          body_size=(4.5, 1.8, 0.7),
                                                          cabin_size=(2.2, 1.6, 0.4)),
              car_4w_rwd_ackermann,    2.7, 4),
    ("AWD",   (0.10, 0.70, 0.20, 1.0),  lambda: _car_urdf("car_awd", (0.10, 0.70, 0.20, 1.0),
                                                          body_size=(4.3, 2.0, 1.4),
                                                          cabin_size=(2.6, 1.8, 0.6)),
              car_4w_awd_ackermann,    2.7, 4),
    ("Truck", (0.95, 0.80, 0.15, 1.0),  lambda: _truck_urdf((0.95, 0.80, 0.15, 1.0)),
              truck_6w_partial_ackermann, 3.6, 6),
]


# ---------------------------------------------------------------------------
# Scene + run
# ---------------------------------------------------------------------------

def _add_loop_markers(vs, radius: float, n: int = 24, z: float = 0.05):
    """Tall posts around the loop at target radius — visual cue for the
    'road' the cars are following (no physical track geometry). Registered as
    visual-only dynamics (physics=False, wheel_raycast=False) so the wheel
    raycaster never treats them as ground."""
    for i in range(n):
        theta = 2 * math.pi * i / n
        x, y = radius * math.cos(theta), radius * math.sin(theta)
        color = (1.0, 0.55, 0.15, 1.0) if i % 2 == 0 else (1.0, 1.0, 1.0, 1.0)
        vs.add_dynamic(
            gs.morphs.Box(size=(0.3, 0.3, 0.5), pos=(x, y, z),
                          fixed=True, collision=False),
            physics=False, wheel_raycast=False,
            surface=gs.surfaces.Plastic(color=color),
        )


def _save_urdf(text: str, tmpdir: str, name: str) -> str:
    path = os.path.join(tmpdir, f"{name}.urdf")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_kind", type=int, default=4,
                    help="Vehicles per kind (default 4 → 16 total).")
    ap.add_argument("--radius",     type=float, default=25.0,
                    help="Loop radius in meters (default 25).")
    ap.add_argument("--duration",   type=float, default=15.0,
                    help="Driving duration in seconds (default 15).")
    ap.add_argument("--throttle",   type=float, default=0.4,
                    help="Constant throttle for all vehicles (default 0.4).")
    ap.add_argument("--viewer", action="store_true",
                    help="Render the top-down camera each step.")
    ap.add_argument("--native", action="store_true",
                    help="Genesis native interactive viewer (orbit/zoom/ESC) instead of cv2.")
    ap.add_argument("--solver", default="multi_batched",
                    choices=["per_vehicle", "multi_batched"],
                    help="Solver: 'multi_batched' (default; MultiVehiclePhysics — "
                         "kinds grouped, compute pipeline batched within each kind, "
                         "much faster for the fleet) or 'per_vehicle' (N separate "
                         "VehiclePhysics in a Python loop — simpler but slow).")
    args = ap.parse_args()
    if args.native:
        args.viewer = False        # --native uses the Genesis viewer, not the cv2 HUD

    K = args.n_per_kind
    N_TOTAL = K * len(KINDS)

    print(f"genesis_vehicle v{sdk_version}  |  road_loop")
    print(f"  fleet  : {N_TOTAL} vehicles ({K} × FWD red, {K} × RWD blue, "
          f"{K} × AWD green, {K} × Truck yellow)")
    print(f"  track  : circle radius {args.radius:.1f} m")
    print(f"  drive  : throttle {args.throttle:.2f} for {args.duration:.1f} s")

    # Stamp URDFs to tempdir.
    tmpdir = tempfile.mkdtemp(prefix="gv_road_loop_")
    urdf_paths = [_save_urdf(urdf_fn(), tmpdir, name.lower())
                  for name, _c, urdf_fn, _p, _wb, _nw in KINDS]

    VehicleScene.init_backend("gpu")
    DT = 0.02
    cam_height = args.radius * 2.5
    from genesis_vehicle.samples import _hud
    if args.viewer and not _hud.have_cv2():
        print("WARN: --viewer needs opencv-python. Continuing headless.")
        args.viewer = False
    # VehicleScene owns the scene / build / step. view: None headless, "native"
    # the Genesis viewer, "cv2" renders the loop camera for the cv2 HUD. The
    # benchmark's solver maps to VE's: multi_batched → "batched", per_vehicle →
    # one VehiclePhysics per vehicle.
    view = "native" if args.native else ("cv2" if args.viewer else None)
    vs = VehicleScene(
        n_envs=1, raycast_mode="single_scene", view=view,
        solver=("batched" if args.solver == "multi_batched" else "per_vehicle"),
        # substeps=30: the cars are stable at 10, but the 5000 kg Truck's stiff
        # suspension blows the constraint forces up to NaN at coarse dt/substeps
        # the moment it drives (20 still NaNs, 30 is stable). multi_batched offsets
        # the cost by batching each kind's pipeline; per_vehicle at 30 is slow.
        dt=DT, substeps=30,
        rigid_options=gs.options.RigidOptions(
            dt=DT, enable_collision=True,
            enable_self_collision=False, enable_joint_limit=True,
        ),
        vis_options=gs.options.VisOptions(
            shadow=True, ambient_light=(0.40, 0.40, 0.40),
            background_color=(0.05, 0.07, 0.10)),
        viewer_options=(_hud.native_viewer_options((0.0, 0.0, cam_height), (0.0, 0.0, 0.0))
                        if args.native else None),
    )
    vs.add_ground_plane(friction=1.0)
    _add_loop_markers(vs, radius=args.radius, n=24)

    # One cfg per kind, shared across that kind's K vehicles → the batched solver
    # groups them into one kind each. VJS is auto-managed by VehicleScene.
    cfg_per_kind = [preset_fn(urdf_paths[k_i], stability="control")
                    for k_i, (_n, _c, _u, preset_fn, _wb, _nw) in enumerate(KINDS)]

    # Spawn vehicles, interleaved around the loop so kinds are mixed visually.
    vehs = []
    entities = []   # (kind_name, Vehicle, wheelbase)
    for global_idx in range(N_TOTAL):
        kind_idx  = global_idx % len(KINDS)
        kind_name, _color, _urdf_fn, _preset_fn, wheelbase, _nw = KINDS[kind_idx]
        theta = 2 * math.pi * global_idx / N_TOTAL
        pos   = (args.radius * math.cos(theta),
                 args.radius * math.sin(theta), 1.0)
        yaw_deg = math.degrees(theta + math.pi / 2)   # tangent CCW
        veh = vs.add_vehicle(
            urdf_paths[kind_idx], cfg=cfg_per_kind[kind_idx],
            morph=gs.morphs.URDF(file=urdf_paths[kind_idx], pos=pos,
                                 euler=(0.0, 0.0, yaw_deg)),
            material=gs.materials.Rigid(friction=1.0))
        vehs.append(veh)
        entities.append((kind_name, veh, wheelbase))

    # Offscreen camera framing the whole loop, only created when --viewer
    # (a camera adds a per-step renderer-sync cost; headless skips it).
    cam = None
    if args.viewer:
        cam = vs.add_camera(
            res=(1280, 720),
            pos=(0.0, 0.0, cam_height), lookat=(0.0, 0.0, 0.0),
            up=(1.0, 0.0, 0.0),       # +X is up on screen
            fov=60, near=0.1, far=cam_height * 4, GUI=False,
        )

    vs.build()

    if args.solver == "multi_batched":
        print(f"  solver : multi_batched — {vs.physics.n_kinds} kinds, "
              f"K per kind = {[k.K for k in vs.physics.kinds]}")

    # Constant Ackermann steering — for ISO 8855 (+steer = right turn, CW),
    # a CCW loop needs negative steer. Per-vehicle (wheelbase differs by kind).
    drive_steer = [-math.atan(wb / args.radius) for (_kn, _veh, wb) in entities]

    # ------------------------------------------------------------------
    # Phase 1 — settle on brake. set_inputs persists across steps.
    # ------------------------------------------------------------------
    print(f"\n[settle 1.5 s]")
    for veh in vehs:
        veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
    for _ in range(int(1.5 / DT)):
        vs.step()
        if args.viewer:
            cam.render()

    # ------------------------------------------------------------------
    # Phase 2 — drive the loop.
    # ------------------------------------------------------------------
    n_steps = int(args.duration / DT)
    print(f"[drive {n_steps} steps  throttle={args.throttle:.2f}  "
          f"steer per-vehicle ≈ {math.degrees(-math.atan(2.7 / args.radius)):.1f}° (car) / "
          f"{math.degrees(-math.atan(3.6 / args.radius)):.1f}° (truck)  "
          f"solver={args.solver}]")

    # Always-on timing — single sync before/after, zero per-step overhead.
    import torch
    hud_perf = _hud.PerfMeter(window=60)

    def _hud_render(step: int):
        # Headless = pure physics (cam is None); viewer = render + HUD.
        if args.native:                 # native viewer renders itself; just watch for close
            return _hud.native_alive(vs)
        if not args.viewer:
            return True
        # Pick the first vehicle of each kind for HUD speed display.
        speeds = []
        for kind_i in range(len(KINDS)):
            veh = entities[kind_i][1]
            v = veh.get_vel()[0].cpu().numpy()
            speeds.append(float(np.linalg.norm(v[:2])))
        frame = _hud.render_hud_frame(
            cam,
            title=f"road_loop  v{sdk_version}   solver={args.solver}",
            lines=[
                f"step {step:>4}/{n_steps}    "
                f"{N_TOTAL} vehicles, {len(KINDS)} kinds (×{K} each)",
                f"speeds: FWD {speeds[0]:4.1f}  RWD {speeds[1]:4.1f}  "
                f"AWD {speeds[2]:4.1f}  Truck {speeds[3]:4.1f} m/s",
                "[ESC] quit",
            ],
            perf_ms=hud_perf.ms_per_step(),
        )
        return _hud.cv2_show("genesis_vehicle road_loop", frame)

    # Apply per-vehicle constant Ackermann steer for the loop (persists).
    for veh, steer in zip(vehs, drive_steer):
        veh.set_inputs(throttle=args.throttle, brake=0.0, steer=steer)
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    user_quit = False
    for step in range(n_steps):
        vs.step()
        hud_perf.tick()
        if step % 2 == 0:    # ~25 fps render
            if not _hud_render(step):
                user_quit = True
                break
    torch.cuda.synchronize()
    wall = time.perf_counter() - t_start
    _hud.cv2_cleanup()
    n_done = step + 1 if user_quit else n_steps
    r_ms, r_n = _hud.bench_render(cam, n=20) if cam is not None else (None, None)
    _hud.print_perf_summary(
        sample=f"road_loop  (v{sdk_version})",
        completed=not user_quit,
        n_done=n_done, n_target=n_steps, wall=wall,
        batch=N_TOTAL, batch_label="vehicle",
        render_ms=r_ms, render_n=r_n,
        extra=[
            f"solver     : {args.solver}",
            f"fleet      : {N_TOTAL} vehicles "
            f"({', '.join(f'{nm}={k}' for nm, k, *_ in KINDS)})",
        ],
    )

    # ------------------------------------------------------------------
    # Final pose summary (one sample per kind — should be near radius).
    # ------------------------------------------------------------------
    print(f"\n=== FINAL  (after {args.duration:.1f} s of driving) ===")
    print(f"  {'kind':<5}  {'pos':<20}  {'radius':>7}  {'speed':>7}")
    print(f"  {'-'*5}  {'-'*20}  {'-'*7}  {'-'*7}")
    for kind_idx, (kind_name, _c, _u, _p, _wb, _nw) in enumerate(KINDS):
        veh = entities[kind_idx][1]    # first vehicle of this kind
        p = veh.get_pos()[0].cpu().numpy()
        v = veh.get_vel()[0].cpu().numpy()
        r = float(np.linalg.norm(p[:2]))
        s = float(np.linalg.norm(v[:2]))
        print(f"  {kind_name:<5}  ({p[0]:+6.2f}, {p[1]:+6.2f})    "
              f"{r:6.2f}   {s:5.2f} m/s")
    print(f"  (target radius {args.radius:.1f} m)")

    if args.native:    # keep the interactive viewer open until closed/ESC
        print("\nviewer 유지 중 — 창 닫기(또는 ESC)로 종료.")
        for veh in vehs:
            veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        try:
            while _hud.native_alive(vs):
                vs.step()
        except gs.GenesisException:
            pass


if __name__ == "__main__":
    main()
