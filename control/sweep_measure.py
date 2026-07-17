"""Measure a sweep table for an arbitrary URDF + SDK preset combination.

CLI::

    python -m genesis_vehicle.control.sweep_measure \\
        --urdf my_vehicle.urdf \\
        --preset tank_skid_belt \\
        --config my_overrides.py \\
        --output my_vehicle_sweep.csv [--gpu] [--quick]

The (v_init, throttle, steer, pitch, roll) grid is measured in parallel
envs: each combination is one env, slopes are emulated by rotating gravity
per env on flat ground, and the chassis response (a, omega_z) is averaged
over a 2 s window. Chunks of ``--n-envs`` (default 500) run until the grid
(17,325 combos by default) is exhausted.

``--config`` points at a Python file defining ``apply_config(cfg)``
(called after ``add_vehicle``, before ``build``) and/or
``apply_runtime_config(physics)`` (called after ``VehiclePhysics`` exists).
The measured table is ONLY valid for the exact (URDF, preset, overrides)
triple — re-measure when any of the three changes.

Measurement correctness (adopted from the deliverables_v3 revision):

- ``a_measured`` is taken in the BODY-LONGITUDINAL frame — the world
  velocity projected through yaw (``v_long = vx*cos(psi) + vy*sin(psi)``),
  the same definition ``PathFollower``'s state extraction consumes. A
  world-x measurement under-reads by cos(yaw) once steer != 0 rotates the
  vehicle during the 2 s window (up to ~12 m/s^2 error at high speed +
  full steer).
- Initial wheel spin uses each wheel's resolved radius (no hard-coded
  tire radius), so an arbitrary URDF starts at rolling condition.

Performance (L3, also from v3): the scene is built ONCE and reused across
chunks via ``scene.reset()`` + ``physics.reset()`` — the last chunk is
padded to the chunk size (excess rows discarded) so the batch shape stays
fixed and nothing re-JITs. Per-env gravity is set in one batched call, and
the measure loop reads velocity only at the window's start/end (omega_z as
a running sum).

Backend: CPU by default like the rest of the SDK, but this workload is a
~500-env L3 batch — well past the measured GPU crossover (~300 envs, see
``docs/backends.md``) — so ``--gpu`` is recommended when CUDA is available
(measured full grid: GPU ~3.6 min vs CPU ~20 min).
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import time

import numpy as np

from .sweep import SweepTable

# Timing defaults = the SDK-wide recommendation (presets' recommended_dt
# 0.025 @ substeps 10 -> internal 2.5 ms). dt/substeps are PART OF THE
# TABLE'S VALIDITY CONTRACT: discrete effects (brake, stability hooks) are
# baked into the measured response, so measure at the dt you will drive at
# (--dt/--substeps below).
DEFAULT_DT = 0.025
DEFAULT_SUBSTEPS = 10
T_GROUND_SETTLE = 0.50   # s — drop to the ground under zero input
T_SETTLE = 0.30          # s — held inputs before the measure window
T_MEASURE = 2.00         # s — response averaging window

# --- Default measurement grid (shared by all vehicles) ------------------------
# throttle == 0 and steer == 0 levels are REQUIRED: SweepTable builds its
# steer inverse from the throttle == 0 rows and vice versa.
V_INIT_LIST = [-4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0]  # m/s
THROTTLE_LIST = [-1.0, -0.5, 0.0, 0.5, 1.0]
STEER_LIST = [-1.0, -0.5, 0.0, 0.5, 1.0]
PITCH_DEG_LIST = [-35.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 35.0]
ROLL_DEG_LIST = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]

# --quick: a coarse smoke grid (CLI plumbing check, NOT a usable table).
QUICK_V = [-2.0, 0.0, 2.0]
QUICK_TS = [-1.0, 0.0, 1.0]
QUICK_PITCH = [-10.0, 0.0, 10.0]
QUICK_ROLL = [0.0]


def compute_slope_gravity(pitch_rad, roll_rad, g=9.8):
    """Gravity vector rotated so a flat-ground vehicle feels a slope."""
    gx = +g * math.sin(pitch_rad)
    gy = -g * math.sin(roll_rad) * math.cos(pitch_rad)
    gz = -g * math.cos(pitch_rad) * math.cos(roll_rad)
    return (gx, gy, gz)


def build_combo_arrays(v_list, thr_list, steer_list, pitch_list, roll_list):
    """Full factorial grid, flattened to parallel per-combo arrays."""
    v_arr, t_arr, s_arr, p_arr, r_arr = [], [], [], [], []
    for v in v_list:
        for th in thr_list:
            for st in steer_list:
                for p in pitch_list:
                    for r in roll_list:
                        v_arr.append(v); t_arr.append(th); s_arr.append(st)
                        p_arr.append(p); r_arr.append(r)
    return tuple(np.array(a, dtype=np.float32)
                 for a in (v_arr, t_arr, s_arr, p_arr, r_arr))


def _load_preset(preset_name: str):
    import genesis_vehicle
    if not hasattr(genesis_vehicle, preset_name):
        available = [n for n in dir(genesis_vehicle)
                     if not n.startswith("_")
                     and ("tank_" in n or "car_" in n or "truck_" in n)]
        raise ValueError(f"unknown preset '{preset_name}'. Available: {available}")
    return getattr(genesis_vehicle, preset_name)


def _load_config_module(config_path):
    """Load the optional override module (apply_config / apply_runtime_config)."""
    if not config_path:
        return None
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config file not found: {config_path}")
    spec = importlib.util.spec_from_file_location("_user_config", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_scene(n_envs, urdf_path, preset_fn, config_mod,
                dt=DEFAULT_DT, substeps=DEFAULT_SUBSTEPS):
    """Build a VehicleScene (dual_scene raycast) for the sweep. (v1.1.26)

    This tool used a raw ``gs.Scene`` + the low-level ``add_vehicle`` before,
    i.e. single_scene raycast — the wheel rays hit the MAIN scene's rigid
    geoms. Two failure modes for an arbitrary URDF, both producing a table of
    pure ballistic noise: (a) the raw file was never urdf-prepped, so wheel
    COLLIDERS fight the suspension as a double support; (b) the v1.1.16
    high-cast ray (start = attach + 1 m) begins INSIDE a tall hull and hits
    the vehicle's OWN roof — a constant self-hit that rides along with the
    vehicle (measured on a 27 t tracked vehicle: d frozen at -0.405 m on all 14 wheels,
    N = 317 kN each, z = +16 m / vz = +56 m/s within the settle). VehicleScene
    preps the URDF automatically, and in dual_scene the rays only see the
    RAYCAST scene's static mirrors — self-hits are impossible.
    """
    from genesis_vehicle import VehicleScene

    vs = VehicleScene(n_envs=n_envs, dt=dt, substeps=substeps,
                      solver="per_vehicle", show_viewer=False)
    vs.add_ground_plane(friction=1.0)
    veh = vs.add_vehicle(urdf_path, preset_fn, pos=(0.0, 0.0, 0.5))
    if config_mod is not None and hasattr(config_mod, "apply_config"):
        config_mod.apply_config(veh.cfg)
    vs.build()
    physics = veh.physics
    if config_mod is not None and hasattr(config_mod, "apply_runtime_config"):
        config_mod.apply_runtime_config(physics)
    return vs, veh, physics


def _wheel_radii_tensor(physics, device):
    """Per-wheel resolved radii (n_wheels,); unresolved wheels fall back to
    the mean of the known ones."""
    import torch
    import genesis as gs
    radii = [w.radius for w in physics.resolved.wheels]
    known = [float(r) for r in radii if r is not None]
    fallback = (sum(known) / len(known)) if known else 0.35
    vals = [float(r) if r is not None else fallback for r in radii]
    return torch.tensor(vals, device=device, dtype=gs.tc_float)


def set_per_env_gravity(scene, pitch_deg, roll_deg, device):
    """Set per-env gravity in ONE batched (n_envs, 3) call (Genesis
    ``set_gravity`` supports batched input natively)."""
    import torch
    n_envs = len(pitch_deg)
    g_vecs = np.zeros((n_envs, 3), dtype=np.float32)
    for i in range(n_envs):
        g_vecs[i] = compute_slope_gravity(
            math.radians(pitch_deg[i]), math.radians(roll_deg[i]))
    scene.sim.set_gravity(torch.tensor(g_vecs, device=device))


def set_initial_velocity(veh, physics, v_init, wheel_radii, device):
    import torch
    import genesis as gs
    entity = veh.entity_main
    n_envs = len(v_init)
    v_t = torch.tensor(v_init, device=device, dtype=gs.tc_float)
    dofs_vel = torch.zeros(n_envs, entity.n_dofs, device=device, dtype=gs.tc_float)
    dofs_vel[:, 0] = v_t
    entity.set_dofs_velocity(dofs_vel)
    # Spin the wheels consistently with the chassis speed (per-wheel resolved
    # radius — no hard-coded tire radius).
    physics.omega[:, :] = v_t.unsqueeze(-1) / wheel_radii.unsqueeze(0)


def _v_long(veh):
    """Body-longitudinal speed: world velocity projected through yaw — the
    SAME definition PathFollower's state extraction consumes, so the table
    is produced and consumed in one frame."""
    import torch
    vel = veh.get_vel()                        # (n_envs, 3) world
    q = veh.get_quat()                         # (n_envs, 4) wxyz
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return vel[:, 0] * torch.cos(yaw) + vel[:, 1] * torch.sin(yaw)


def measure_one_chunk(vs, veh, physics, wheel_radii,
                      v_init, throttle, steer, pitch_deg, roll_deg, device,
                      dt=DEFAULT_DT):
    import torch
    import genesis as gs

    n_ground_settle = int(T_GROUND_SETTLE / dt)
    n_settle = int(T_SETTLE / dt)
    n_measure = int(T_MEASURE / dt)

    n_envs = len(v_init)
    set_per_env_gravity(vs.scene, pitch_deg, roll_deg, device)

    zeros = torch.zeros(n_envs, device=device, dtype=gs.tc_float)
    veh.set_inputs(throttle=zeros, brake=zeros, steer=zeros)
    for _ in range(n_ground_settle):
        vs.step()

    set_initial_velocity(veh, physics, v_init, wheel_radii, device)

    veh.set_inputs(
        throttle=torch.tensor(throttle, device=device, dtype=gs.tc_float),
        brake=zeros,
        steer=torch.tensor(steer, device=device, dtype=gs.tc_float),
    )
    for _ in range(n_settle):
        vs.step()

    # Velocity (+quat) only at the window's start/end; omega_z as a running
    # sum — no per-step history stacking.
    v_long_start = _v_long(veh)
    omega_z_sum = torch.zeros(n_envs, device=device, dtype=gs.tc_float)
    for _ in range(n_measure):
        vs.step()
        omega_z_sum += veh.get_ang()[:, 2]
    v_long_end = _v_long(veh)

    a_measured = ((v_long_end - v_long_start) / (n_measure * dt)).cpu().numpy()
    omega_z_measured = (omega_z_sum / n_measure).cpu().numpy()

    return {"v_init": v_init, "throttle": throttle, "steer": steer,
            "pitch": pitch_deg, "roll": roll_deg,
            "a_measured": a_measured, "omega_z_measured": omega_z_measured}


def run_sweep(urdf_path, preset_fn, config_mod, n_envs,
              v_arr, t_arr, s_arr, p_arr, r_arr, device, log=print,
              dt=DEFAULT_DT, substeps=DEFAULT_SUBSTEPS):
    """Build the scene ONCE and reuse it across chunks via reset.

    The last chunk is padded to ``n_envs`` by repeating the final combo
    (excess rows discarded afterwards), so every chunk has the same batch
    shape and nothing re-JITs.
    """
    n_total = len(v_arr)
    n_chunks = (n_total + n_envs - 1) // n_envs
    n_padded = n_chunks * n_envs

    def _pad(a):
        return np.concatenate([a, np.repeat(a[-1:], n_padded - n_total, axis=0)])
    v_p, t_p, s_p, p_p, r_p = (_pad(a) for a in (v_arr, t_arr, s_arr, p_arr, r_arr))

    vs, veh, physics = build_scene(
        n_envs, urdf_path, preset_fn, config_mod, dt=dt, substeps=substeps)
    wheel_radii = _wheel_radii_tensor(physics, device)

    results = {k: [] for k in ("v_init", "throttle", "steer", "pitch", "roll",
                               "a_measured", "omega_z_measured")}
    for ci in range(n_chunks):
        lo, hi = ci * n_envs, (ci + 1) * n_envs
        log(f"\n  [chunk {ci + 1}/{n_chunks}] {lo}..{min(hi, n_total)}  "
            f"(n_envs={n_envs})", flush=True)
        if ci > 0:
            vs.scene.reset()   # main-scene poses/velocities back to build time
            physics.reset()    # wheel omega, slip state, first-step guard
        t0 = time.perf_counter()
        out = measure_one_chunk(
            vs, veh, physics, wheel_radii,
            v_p[lo:hi], t_p[lo:hi], s_p[lo:hi], p_p[lo:hi], r_p[lo:hi], device,
            dt=dt)
        log(f"    elapsed: {time.perf_counter() - t0:.1f}s", flush=True)
        for k in results:
            results[k].append(out[k])

    return {k: np.concatenate(v)[:n_total] for k, v in results.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Measure a sweep table (see genesis_vehicle/docs/path-following.md)")
    ap.add_argument("--urdf", required=True, help="vehicle URDF path")
    ap.add_argument("--preset", required=True,
                    help="genesis_vehicle preset name (e.g. tank_skid_belt, "
                         "car_4w_rwd_ackermann)")
    ap.add_argument("--config", default=None,
                    help="optional Python override file defining "
                         "apply_config(cfg) / apply_runtime_config(physics)")
    ap.add_argument("--output", required=True, help="output CSV path")
    ap.add_argument("--n-envs", "--n_envs", dest="n_envs", type=int, default=500,
                    help="parallel envs per chunk (default 500; lower it on "
                         "GPU out-of-memory)")
    ap.add_argument("--gpu", action="store_true",
                    help="use the GPU backend. Recommended when CUDA is "
                         "available: at ~500 envs this workload is past the "
                         "measured CPU/GPU crossover (~300 envs, docs/backends.md); "
                         "measured full grid GPU ~3.6 min vs CPU ~20 min")
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke grid (81 combos) to check the CLI wiring; "
                         "the resulting table is too coarse for real control")
    ap.add_argument("--dt", type=float, default=DEFAULT_DT,
                    help="sim dt for the measurement (default 0.025 — the "
                         "presets' recommended_dt). MEASURE AT THE dt YOU "
                         "WILL DRIVE AT: dt/substeps are part of the table's "
                         "validity contract")
    ap.add_argument("--substeps", type=int, default=DEFAULT_SUBSTEPS,
                    help="solver substeps (default 10 -> internal 2.5 ms at "
                         "dt=0.025)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.urdf):
        raise FileNotFoundError(f"URDF not found: {args.urdf}")
    preset_fn = _load_preset(args.preset)
    config_mod = _load_config_module(args.config)

    import genesis as gs
    from genesis_vehicle import VehicleScene
    VehicleScene.init_backend("gpu" if args.gpu else "cpu")
    device = gs.device

    if args.quick:
        grids = (QUICK_V, QUICK_TS, QUICK_TS, QUICK_PITCH, QUICK_ROLL)
    else:
        grids = (V_INIT_LIST, THROTTLE_LIST, STEER_LIST,
                 PITCH_DEG_LIST, ROLL_DEG_LIST)
    v_arr, t_arr, s_arr, p_arr, r_arr = build_combo_arrays(*grids)
    n_total = len(v_arr)
    n_envs = min(args.n_envs, n_total)   # don't build more envs than combos
    print("\n=== Sweep Table Measurement ===")
    print(f"  URDF: {args.urdf}")
    print(f"  Preset: {args.preset}")
    print(f"  Config: {args.config or '(none)'}")
    print(f"  Output: {args.output}")
    print(f"  Backend: {'gpu' if args.gpu else 'cpu'}")
    print(f"  Combos: {n_total} "
          f"({len(grids[0])}v x {len(grids[1])}thr x {len(grids[2])}str "
          f"x {len(grids[3])}pit x {len(grids[4])}rol)"
          f"{'  [--quick smoke grid]' if args.quick else ''}")
    print(f"  Chunk size: {n_envs}")
    print(f"  Timing: dt={args.dt} x substeps={args.substeps} "
          f"(internal {args.dt / args.substeps * 1000:.2f} ms)")

    results = run_sweep(args.urdf, preset_fn, config_mod, n_envs,
                        v_arr, t_arr, s_arr, p_arr, r_arr, device,
                        dt=args.dt, substeps=args.substeps)
    table = SweepTable(results)
    table.save(args.output)
    print(f"\n[done] saved {n_total} rows -> {args.output}")


if __name__ == "__main__":
    main()
