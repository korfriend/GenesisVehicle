"""batched_rollout.py — batched n_envs rollout for RL / MPPI.

Spawns N copies of the 4-wheel car in parallel, applies a different
random throttle / steer per env every step, prints state-tensor shapes
and per-step wall time. Demonstrates the batched (``n_envs > 1``) API
that the velocity-damper or stick-slip ``StaticFrictionLock`` will apply
across all envs simultaneously without a Python-level loop.

What this demonstrates
----------------------
- Build with ``scene.build(n_envs=N)``.
- ``VehicleInputs`` accepts per-env tensors of shape ``(N,)`` — throttle,
  brake, steer all broadcast to per-env values.
- ``car.get_pos() / get_vel() / get_quat()`` return ``(N, 3)`` or ``(N, 4)``
  — index by env.
- Sensor / physics state stays GPU-resident; no per-env Python loop.
- Per-step wall time stays roughly constant up to a few hundred envs
  (CUDA throughput limit) — important for sample-efficient MPPI / RL.

Run
---
    python -m genesis_vehicle.samples.batched_rollout
    python -m genesis_vehicle.samples.batched_rollout --n_envs 256 --steps 200
    python -m genesis_vehicle.samples.batched_rollout --n_envs 16 --viewer
        ↑ --viewer lays the N envs out on a grid (env_separate_rigid) so
          you can SEE all parallel rollouts. Note: rendering adds per-step
          overhead and makes the throughput numbers unrepresentative —
          run without --viewer for true bench timings.
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
import time

import torch
import genesis as gs

from genesis_vehicle import (
    VehiclePhysics, VehicleInputs,
    car_4w_rwd_ackermann,
    add_vehicle,
    __version__ as sdk_version,
)

URDF_PATH = os.path.join(os.path.dirname(__file__), "urdf", "car_4w.urdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_envs", type=int, default=64,
                    help="Number of parallel rollouts (default 64).")
    ap.add_argument("--steps",  type=int, default=100,
                    help="Steps to measure after warmup (default 100).")
    ap.add_argument("--warmup", type=int, default=20,
                    help="Warmup steps to compile kernels (default 20).")
    ap.add_argument("--viewer", action="store_true",
                    help="Render all N envs in a grid (env_separate_rigid). "
                         "Adds per-step overhead — for clean bench numbers, "
                         "use perf_vectorization.py instead.")
    ap.add_argument("--grid_spacing", type=float, default=12.0,
                    help="Grid cell spacing (m) when --viewer is on (default 12).")
    args = ap.parse_args()

    print(f"genesis_vehicle v{sdk_version}  |  batched_rollout  "
          f"n_envs={args.n_envs}  steps={args.steps}"
          + ("  (viewer ON — grid)" if args.viewer else ""))

    cfg = car_4w_rwd_ackermann(URDF_PATH, stability="control")
    gs.init(backend=gs.gpu, logging_level="warning")

    from genesis_vehicle.samples import _hud
    if args.viewer and not _hud.have_cv2():
        print("WARN: --viewer needs opencv-python. Continuing headless.")
        args.viewer = False

    per_row = max(1, int(round(math.sqrt(args.n_envs))))
    n_rows  = math.ceil(args.n_envs / per_row)
    grid_w  = args.grid_spacing * per_row
    grid_h  = args.grid_spacing * n_rows
    cam_h   = max(grid_w, grid_h) * 1.2

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=cfg.recommended_dt, substeps=10),
        rigid_options=gs.options.RigidOptions(dt=cfg.recommended_dt, enable_collision=True),
        vis_options=gs.options.VisOptions(
            shadow=True, ambient_light=(0.40, 0.40, 0.40),
            background_color=(0.05, 0.07, 0.10),
            env_separate_rigid=args.viewer,
        ),
        show_viewer=False,    # --viewer uses cv2 HUD instead
    )
    scene.add_entity(
        gs.morphs.Plane(pos=(0, 0, 0)),
        material=gs.materials.Rigid(friction=1.0),
    )
    car, sensor, _ = add_vehicle(
        scene, URDF_PATH, preset_fn=None, pos=(0.0, 0.0, 1.0),
        material=gs.materials.Rigid(friction=1.0),
    )

    cam = None
    if args.viewer:
        # env_separate_rigid → N per-env frames. HUD downsizes each cell to
        # ≤480 px, so a modest per-cell render res is plenty.
        cam = scene.add_camera(
            res=(640, 360),
            pos=(0.0, 0.0, cam_h), lookat=(0.0, 0.0, 0.0),
            up=(1.0, 0.0, 0.0), fov=70, near=0.1, far=cam_h * 4, GUI=False,
        )
        scene.build(
            n_envs=args.n_envs,
            env_spacing=(args.grid_spacing, args.grid_spacing),
            n_envs_per_row=per_row,
        )
        print(f"  viewer grid: {per_row} × {n_rows} cells, spacing {args.grid_spacing} m")
    else:
        scene.build(n_envs=args.n_envs)
    physics = VehiclePhysics(scene, car, sensor, cfg, n_envs=args.n_envs)
    device = car.get_pos().device

    print(f"\n[shapes after build (n_envs={args.n_envs})]")
    print(f"  car.get_pos()        : {tuple(car.get_pos().shape)}")
    print(f"  car.get_vel()        : {tuple(car.get_vel().shape)}")
    print(f"  car.get_quat()       : {tuple(car.get_quat().shape)}")
    print(f"  sensor.read().distances: {tuple(sensor.read().distances.shape)}")
    print(f"  physics.omega        : {tuple(physics.omega.shape)}")

    # 1.5 s settle, brake held, identical across envs.
    settle_in = VehicleInputs(throttle=0.0, brake=1.0, steer=0.0)
    render_every = max(1, int(0.04 / cfg.recommended_dt))
    hud_perf = _hud.PerfMeter(window=60)

    def _hud_render(phase: str, step: int, total: int):
        if cam is None:
            return True
        if not args.viewer:
            cam.render()
            return True
        # Quick spread stats for HUD.
        v = car.get_vel().cpu().numpy()
        speed = (v[:, :2] ** 2).sum(axis=1) ** 0.5
        frame = _hud.render_hud_frame(
            cam,
            title=f"batched_rollout  n_envs={args.n_envs}   v{sdk_version}",
            lines=[
                f"phase: {phase}    step {step:>4}/{total}",
                f"grid : {per_row} × {n_rows}    spacing {args.grid_spacing} m",
                f"speed across envs: {speed.min():.2f} .. {speed.max():.2f} m/s "
                f"(mean {speed.mean():.2f})",
                "[ESC] quit",
            ],
            perf_ms=hud_perf.ms_per_step(),
            grid_per_row=per_row,
        )
        return _hud.cv2_show("genesis_vehicle batched_rollout", frame)

    user_quit = False
    for step in range(int(1.5 / cfg.recommended_dt)):
        physics.step(settle_in)
        scene.step()
        hud_perf.tick()
        if step % render_every == 0:
            if not _hud_render("settle", step, int(1.5 / cfg.recommended_dt)):
                user_quit = True
                break
    if user_quit:
        # Don't enter the measure loop if the user already wanted out during
        # settle — just emit a partial summary and return.
        _hud.cv2_cleanup()
        _hud.print_perf_summary(
            sample=f"batched_rollout  (v{sdk_version})",
            completed=False, n_done=step + 1, n_target=int(1.5 / cfg.recommended_dt),
            wall=0.0, batch=args.n_envs, batch_label="env",
            extra=["quit during settle phase — no measured drive timing"],
        )
        return
    print(f"\n[settled — running {args.warmup + args.steps} steps "
          f"({args.warmup} warmup + {args.steps} measured)]")

    # Per-env random controls. Throttle ∈ [0, 0.6], steer ∈ [-0.4, +0.4],
    # brake = 0. These are (n_envs,) tensors — broadcast by VehicleInputs.
    g = torch.Generator(device=device).manual_seed(0)
    throttle = 0.6 * torch.rand(args.n_envs, generator=g, device=device)
    steer    = 0.8 * (torch.rand(args.n_envs, generator=g, device=device) - 0.5)
    brake    = torch.zeros(args.n_envs, device=device)
    inputs   = VehicleInputs(throttle=throttle, brake=brake, steer=steer)

    for _ in range(args.warmup):
        physics.step(inputs)
        scene.step()
        hud_perf.tick()

    user_quit = False
    t0 = time.perf_counter()
    for step in range(args.steps):
        physics.step(inputs)
        scene.step()
        hud_perf.tick()
        if step % render_every == 0:
            if not _hud_render("measure", step, args.steps):
                user_quit = True
                break
    wall = time.perf_counter() - t0
    _hud.cv2_cleanup()
    n_done = step + 1 if user_quit else args.steps

    p = car.get_pos().cpu().numpy()    # (n_envs, 3)
    v = car.get_vel().cpu().numpy()
    speed = (v[:, :2] ** 2).sum(axis=1) ** 0.5

    r_ms, r_n = _hud.bench_render(cam, n=20) if cam is not None else (None, None)
    _hud.print_perf_summary(
        sample=f"batched_rollout  (v{sdk_version})",
        completed=not user_quit,
        n_done=n_done, n_target=args.steps, wall=wall,
        batch=args.n_envs, batch_label="env",
        render_ms=r_ms, render_n=r_n,
        extra=[
            f"warmup     : {args.warmup} steps (not counted)",
            f"spread x   : {p[:,0].min():+.2f} .. {p[:,0].max():+.2f}",
            f"spread y   : {p[:,1].min():+.2f} .. {p[:,1].max():+.2f}",
            f"speed range: {speed.min():.2f} .. {speed.max():.2f} m/s",
        ],
    )


if __name__ == "__main__":
    main()
