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
"""

from __future__ import annotations

import argparse
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
    args = ap.parse_args()

    print(f"genesis_vehicle v{sdk_version}  |  batched_rollout  "
          f"n_envs={args.n_envs}  steps={args.steps}")

    cfg = car_4w_rwd_ackermann(URDF_PATH, stability="control")
    gs.init(backend=gs.gpu, logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=cfg.dt, substeps=20),
        rigid_options=gs.options.RigidOptions(dt=cfg.dt, enable_collision=True),
        show_viewer=False,
    )
    scene.add_entity(
        gs.morphs.Plane(pos=(0, 0, 0)),
        material=gs.materials.Rigid(friction=1.0),
    )
    car, sensor, _ = add_vehicle(
        scene, URDF_PATH, preset_fn=None, pos=(0.0, 0.0, 1.0),
        material=gs.materials.Rigid(friction=1.0),
    )

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
    for _ in range(int(1.5 / cfg.dt)):
        physics.step(settle_in)
        scene.step()
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

    t0 = time.perf_counter()
    for _ in range(args.steps):
        physics.step(inputs)
        scene.step()
    wall = time.perf_counter() - t0

    p = car.get_pos().cpu().numpy()    # (n_envs, 3)
    v = car.get_vel().cpu().numpy()
    speed = (v[:, :2] ** 2).sum(axis=1) ** 0.5

    print(f"\n[per-step wall time]  {wall / args.steps * 1000:.2f} ms / step  "
          f"→ {args.n_envs * args.steps / wall:.0f} env-steps / s")
    print(f"[chassis spread]   x: {p[:, 0].min():+.2f}..{p[:, 0].max():+.2f}  "
          f"y: {p[:, 1].min():+.2f}..{p[:, 1].max():+.2f}  "
          f"speed: {speed.min():.2f}..{speed.max():.2f} m/s")


if __name__ == "__main__":
    main()
