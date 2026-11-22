"""Measure a vehicle's MPC parameters (the MPC layer's ``mpc_params`` JSON).

CLI::

    python -m genesis_vehicle.control.identify \\
        --urdf car.urdf --preset car_4w_rwd_ackermann [--config overrides.py] \\
        --output car_mpc_params.json [--dt 0.025 --substeps 4] [plant overrides] [--gpu]

Same flags, scene and plant overrides as ``sweep_measure`` (``build_scene``),
so a table and a params file measured with the same flags describe the same
plant. The code does NOT check that a given table and params file belong to
the same vehicle (a sweep CSV carries no metadata).

Measurements (the field ``PathFollower_MPC/tools/identify.py`` procedure,
run in parallel envs, flat ground):

1. geometry and mass: axle positions, base_link -> rear axle (``rig_offset``),
   rear axle -> centre of mass, mass, yaw inertia about the centre of mass,
   maximum steering angle, wheel radius;
2. cornering stiffness (front/rear, per wheel): constant speed + fixed
   steering for 5 s, mean ``(vx, vy, r)`` after 3 s; the two steady-state
   equations of the linear bicycle are linear in ``(C_af, C_ar)``, solved by
   least squares (:func:`fit_cornering_stiffness`);
3. acceleration limits per speed: throttle +1 / throttle -1 / brake 1 for
   0.5 s from several speeds;
4. response times: a throttle step (+0.3) and a steering step (-0.2) at
   6 m/s -> time to 63 % (:func:`t63`).

Recorded from the vehicle config (no fit): per-axle Pacejka lateral
coefficients and friction (``tyre``; ``None`` for a non-Pacejka tyre),
static axle loads (settled ``last_N`` and the moment-balance values),
the low-speed deadbeat scale ``s`` and speed ``v_star``
(:func:`lateral_deadbeat_speed`), steering, axle count, driven axles. The fit
residual of step 2 (:func:`cornering_fit_residual`) and the sampled envelope
are recorded too; a steady-state yaw-rate residual above ``RESIDUAL_WARN``
(3 %) warns.

Only 2-axle Ackermann vehicles are measured (that is what the MPC model
describes): anything else raises ``ValueError`` naming ``mpc_model="off"``
before any measurement.
"""
from __future__ import annotations

import argparse
import math
import os
import warnings

import numpy as np

from .sweep_measure import (DEFAULT_DT, DEFAULT_SUBSTEPS, _load_config_module, _load_preset,
                            add_plant_override_args, build_scene, plant_overrides_fn)

#: (speed m/s, steer input) cases of the cornering fit (lateral acc. 2..5 m/s^2).
CORNER = [(v, s) for v in (6.0, 8.0, 10.0, 12.0) for s in (0.03, 0.06, -0.06)] + \
         [(4.0, 0.3), (6.0, 0.2), (6.0, 0.3), (8.0, 0.15), (8.0, 0.2)]
#: (speed, throttle, brake) cases of the acceleration limits.
LONG = [(v, thr, brk) for v in (0.0, 3.0, 6.0, 10.0, 14.0)
        for thr, brk in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0))]
#: Steady-state yaw-rate residual (relative, max over samples) that warns.
RESIDUAL_WARN = 0.03
#: Static-load measurement: braked settle after the drop, and the averaging window (s).
T_LOAD_SETTLE, T_LOAD_AVG = 1.5, 0.25


# --- pure helpers (Genesis-free) ------------------------------------------------

def hold_throttle(v_target, v_x):
    """Speed-hold throttle of the field tool: ``clip(0.8*(v0 - vx) + 0.08, -1, 1)``."""
    return np.clip(0.8 * (np.asarray(v_target) - np.asarray(v_x)) + 0.08, -1, 1)


def t63(y, dt):
    """Time for a normalised step response ``y`` (final value 1, sampled
    every ``dt`` from the step) to reach 0.63, linearly interpolated."""
    y = np.concatenate([[0.0], np.asarray(y, float)])
    k = int(np.argmax(y >= 0.63))
    if k == 0 or y[k] < 0.63:
        return float("nan")
    return float((k - 1 + (0.63 - y[k - 1]) / (y[k] - y[k - 1])) * dt)


def _bicycle_rows(vx, vy, r, d, mass, lf, lr):
    """The two steady-state rows (lateral force, yaw moment), linear in
    ``(C_af, C_ar)`` (per wheel; 2 wheels per axle)."""
    return ([-2 * vy - 2 * lf * r + 2 * d * vx, -2 * vy + 2 * lr * r], mass * vx * vx * r,
            [-2 * lf * vy - 2 * lf * lf * r + 2 * lf * d * vx, 2 * lr * vy - 2 * lr * lr * r], 0.0)


def fit_cornering_stiffness(samples, mass, lf, lr, max_steer_rad, steer_sign):
    """``(C_af, C_ar)`` per wheel [N/rad] from steady-state samples
    ``(vx, vy, r, steer_input)`` — the field tool's least squares, rows
    scaled to unit max. ``steer_sign`` maps a steer input to the front angle
    (``+steer = right`` gives -1 on a left-positive angle)."""
    A, b = [], []
    for vx, vy, r, s in samples:
        d = steer_sign * s * max_steer_rad
        r1, b1, r2, b2 = _bicycle_rows(vx, vy, r, d, mass, lf, lr)
        A += [r1, r2]
        b += [b1, b2]
    A, b = np.array(A, float), np.array(b, float)
    sc = np.abs(A).max(1, keepdims=True)
    (caf, car), *_ = np.linalg.lstsq(A / sc, b / sc[:, 0], rcond=None)
    return float(caf), float(car)


def steady_state_bicycle(vx, d, caf, car, mass, izz, lf, lr):
    """Steady-state ``(vy, r)`` of the linear bicycle at speed ``vx`` and front
    angle ``d`` (the field model's dynamic branch with zero derivatives)."""
    a00 = -2 * (caf + car) / (mass * vx)
    a01 = -vx - 2 * (lf * caf - lr * car) / (mass * vx)
    a10 = -2 * (lf * caf - lr * car) / (izz * vx)
    a11 = -2 * (lf * lf * caf + lr * lr * car) / (izz * vx)
    b0, b1 = 2 * caf / mass * d, 2 * lf * caf / izz * d
    vy, r = np.linalg.solve(np.array([[a00, a01], [a10, a11]]), -np.array([b0, b1]))
    return float(vy), float(r)


def cornering_fit_residual(samples, caf, car, mass, izz, lf, lr, max_steer_rad, steer_sign):
    """How well the fitted linear bicycle reproduces the samples.

    Returns ``yaw_rate_rel_max`` (max over samples of ``|r_pred - r| / |r|``),
    ``v_lat_abs_max`` (m/s), the sampled lateral-acceleration and speed
    ranges, and ``warn`` (``yaw_rate_rel_max > RESIDUAL_WARN``)."""
    rel, dvy, ay, vs = [], [], [], []
    for vx, vy, r, s in samples:
        d = steer_sign * s * max_steer_rad
        vy_p, r_p = steady_state_bicycle(vx, d, caf, car, mass, izz, lf, lr)
        rel.append(abs(r_p - r) / max(abs(r), 1e-9))
        dvy.append(abs(vy_p - vy))
        ay.append(abs(vx * r))
        vs.append(vx)
    out = {"yaw_rate_rel_max": float(max(rel)), "v_lat_abs_max": float(max(dvy)),
           "a_y_range": [float(min(ay)), float(max(ay))],
           "speed_range": [float(min(vs)), float(max(vs))]}
    out["warn"] = out["yaw_rate_rel_max"] > RESIDUAL_WARN
    return out


def lateral_deadbeat_speed(tyre_axles, s, dt, g=None):
    """``v* = max over axles of B_y * C_y * mu_lat * lat_scale * g * dt / s``:
    the speed below which the 1.8.0 lateral deadbeat binds at static load on
    at least one axle (``physics-contracts.md`` section 7.16; reference car
    2.98 m/s at dt 0.025)."""
    if g is None:
        from ..dynamics import DEADBEAT_G as g
    return float(max(a["pb_y"] * a["pc_y"] * a["mu_lat"] * a.get("lat_scale", 1.0) * g * dt / s
                     for a in tyre_axles))


def check_identifiable(resolved):
    """``(steering name, n_axles)``; raises ``ValueError`` unless the vehicle
    is a 2-axle Ackermann (the MPC model). An unknown axle index counts as
    unknown, not as one axle."""
    from ..strategies.steering import Ackermann
    axles = [w.axle_index for w in resolved.wheels]
    n = "?" if any(a is None for a in axles) else len(set(axles))
    if not isinstance(resolved.steering, Ackermann) or n != 2:
        raise ValueError(
            "the MPC layer models a 2-axle Ackermann vehicle; this vehicle has "
            f"{type(resolved.steering).__name__}/{n} axles — there are no MPC "
            "parameters to identify; use PathFollower(..., mpc_model=\"off\")")
    return type(resolved.steering).__name__, n


def tyre_block(resolved):
    """Per-axle Pacejka lateral coefficients + friction from the config, or
    ``None`` for a non-Pacejka tyre. Raises ``ValueError`` if the wheels of one
    axle differ (an axle cannot be lumped)."""
    from ..tire_models.pacejka import PacejkaAnisotropic
    tire = resolved.tire
    if not isinstance(tire, PacejkaAnisotropic):
        return None
    out = []
    for ax in sorted({w.axle_index for w in resolved.wheels}):
        ws = [w for w in resolved.wheels if w.axle_index == ax]
        rec = {}
        for key in ("pb_y", "pc_y", "pe_y", "mu_lat", "mu_long"):
            vals = {float(getattr(w, key)) for w in ws}
            if len(vals) != 1:
                raise ValueError(f"axle {ax}: wheels differ in {key} ({sorted(vals)}); "
                                 "the MPC model lumps an axle and cannot represent that")
            rec[key] = vals.pop()
        rec.update(axle=int(ax), lat_scale=float(tire.lat_scale), eps_v=float(tire.eps_v),
                   n_wheels=len(ws))
        out.append(rec)
    return {"model": type(tire).__name__, "axles": out}


# --- measurement (needs Genesis) -------------------------------------------------

def identify_mpc_params(urdf_path, preset_fn, config_mod=None, *, dt=DEFAULT_DT,
                        substeps=DEFAULT_SUBSTEPS, overrides_fn=None, device=None, log=print):
    """Measure ``urdf_path`` + ``preset_fn`` (+ ``config_mod``) and return its
    :class:`~genesis_vehicle.control.mpc.MPCVehicleParams` (call
    ``.to_json(path)`` to save). Builds its own ``VehicleScene`` (the
    ``sweep_measure`` scene); call ``VehicleScene.init_backend`` first."""
    import torch
    import genesis as gs
    import genesis_vehicle
    from ..dynamics import deadbeat_scale
    from .mpc.params import SCHEMA, MPCVehicleParams
    from .plant import plant_mass_properties

    N = max(len(CORNER), len(LONG))
    vs, veh, phys = build_scene(N, urdf_path, preset_fn, config_mod, dt=dt, substeps=substeps,
                                overrides_fn=overrides_fn)
    steering, n_axles = check_identifiable(phys.resolved)
    tyre = tyre_block(phys.resolved)
    ent, cfg = veh.entity_main, veh.cfg
    dev = device if device is not None else gs.device
    DT = float(dt)

    # (1) geometry and mass
    wm = phys.wheel_meta
    pos = wm.positions.detach().cpu().numpy().reshape(-1, 3)
    axle = wm.axle_index.detach().cpu().numpy().reshape(-1)
    rad = wm.radius.detach().cpu().numpy().reshape(-1)
    ax_x = {int(k): float(pos[axle == k, 0].mean()) for k in np.unique(axle)}
    fa, ra = max(ax_x, key=ax_x.get), min(ax_x, key=ax_x.get)
    L = ax_x[fa] - ax_x[ra]
    mp = plant_mass_properties(ent)
    mass = float(mp.mass)
    izz_cg = float(mp.izz) - mass * (float(mp.com[0]) ** 2 + float(mp.com[1]) ** 2)
    LR = float(mp.com[0]) - ax_x[ra]
    LF = L - LR
    MAXS = float(cfg.steering.max_steer_rad)
    R = float(rad.mean())
    log(f"[geometry] wheelbase {L:.3f} m, base_link->rear axle {-ax_x[ra]:.3f} m, rear axle->CoM "
        f"{LR:.3f} m, mass {mass:.1f} kg, yaw inertia (CoM) {izz_cg:.1f} kg*m^2, max steer "
        f"{MAXS:.3f} rad, wheel radius {R:.3f} m")

    def T(x):
        return torch.as_tensor(np.asarray(x, np.float32), device=dev)

    def body():
        v = ent.get_vel().reshape(N, 3).cpu().numpy()
        q = ent.get_quat().reshape(N, 4).cpu().numpy()
        w = ent.get_ang().reshape(N, 3).cpu().numpy()
        h = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        return v[:, 0] * np.cos(h) + v[:, 1] * np.sin(h), -v[:, 0] * np.sin(h) + v[:, 1] * np.cos(h), w[:, 2]

    def start(v0):
        vs.scene.reset()
        phys.reset()
        z = np.zeros(N)
        veh.set_inputs(throttle=T(z), brake=T(z), steer=T(z))
        for _ in range(int(0.5 / DT)):
            vs.step()
        v0 = np.asarray(v0, float)
        dv = torch.zeros(N, ent.n_dofs, device=dev, dtype=gs.tc_float)
        dv[:, 0] = T(v0)
        ent.set_dofs_velocity(dv)
        phys.omega[:, :] = T(v0).unsqueeze(-1) / T(rad).unsqueeze(0)

    # static loads + deadbeat scale: settle at rest under full brake for
    # T_LOAD_SETTLE after the 0.5 s drop, average env 0's last_N over the
    # final T_LOAD_AVG
    start(np.zeros(N))
    n_log = []
    veh.set_inputs(throttle=T(np.zeros(N)), brake=T(np.ones(N)), steer=T(np.zeros(N)))
    n_settle, n_avg = int(T_LOAD_SETTLE / DT), int(T_LOAD_AVG / DT)
    for k in range(n_settle):
        vs.step()
        if k >= n_settle - n_avg:
            n_log.append(phys.last_N[0].detach().cpu().numpy().astype(float).copy())
    n_w = np.mean(n_log, 0)
    loads = {int(k): float(n_w[axle == k].sum()) for k in np.unique(axle)}
    g = 9.81
    com_x = float(mp.com[0])
    balance = {fa: mass * g * (com_x - ax_x[ra]) / L, ra: mass * g * (ax_x[fa] - com_x) / L}
    s_db = float(deadbeat_scale(torch.as_tensor(n_w[None], device=wm.positions.device,
                                                dtype=wm.positions.dtype), wm.yaw_inertia)[0, 0])
    v_star = lateral_deadbeat_speed(tyre["axles"], s_db, DT) if tyre else None
    log(f"[loads] settled axle loads {loads} N (moment balance {balance}); deadbeat s {s_db:.3f}, "
        f"v* {v_star}")

    # (2) cornering
    v0 = np.array([c[0] for c in CORNER] + [0.0] * (N - len(CORNER)))
    s_in = np.array([c[1] for c in CORNER] + [0.0] * (N - len(CORNER)))
    start(v0)
    acc = []
    for k in range(int(5.0 / DT)):
        vx, vy, r = body()
        veh.set_inputs(throttle=T(hold_throttle(v0, vx)), brake=T(np.zeros(N)), steer=T(s_in))
        vs.step()
        if k * DT > 3.0:
            acc.append(np.stack([vx, vy, r], 1))
    A = np.mean(acc, 0)[:len(CORNER)]
    rk = A[:, 0] * np.tan(s_in[:len(CORNER)] * MAXS) / L
    sign = float(np.sign(np.median(A[:, 2] / rk)))
    samples = [[float(vx), float(vy), float(r), float(s)] for (vx, vy, r), s in zip(A, s_in)]
    caf, car = fit_cornering_stiffness(samples, mass, LF, LR, MAXS, sign)
    rows = [[vx, vy, r, sign * s * MAXS] for vx, vy, r, s in samples]
    resid = cornering_fit_residual(samples, caf, car, mass, izz_cg, LF, LR, MAXS, sign)
    log(f"[cornering] steer input sign {sign:+.0f}, front {caf:.0f} / rear {car:.0f} N/rad; "
        f"steady-state yaw-rate residual max {100 * resid['yaw_rate_rel_max']:.2f} %")
    if resid["warn"]:
        warnings.warn(
            f"the linear bicycle reproduces the measured steady-state yaw rate only to "
            f"{100 * resid['yaw_rate_rel_max']:.1f} % (threshold {100 * RESIDUAL_WARN:.0f} %): "
            "the bicycle MPC model is a poor description of this vehicle", UserWarning)

    # (3) acceleration limits per speed
    v0 = np.array([c[0] for c in LONG] + [0.0] * (N - len(LONG)))
    thr = np.array([c[1] for c in LONG] + [0.0] * (N - len(LONG)))
    brk = np.array([c[2] for c in LONG] + [0.0] * (N - len(LONG)))
    start(v0)
    va, _, _ = body()
    veh.set_inputs(throttle=T(thr), brake=T(brk), steer=T(np.zeros(N)))
    for _ in range(int(0.5 / DT)):
        vs.step()
    vb, _, _ = body()
    acc_l = (vb - va) / 0.5
    amx, dthr, dbrk = {}, {}, {}
    for i, (v, t_, b_) in enumerate(LONG):
        if v == 0.0 and (t_ < 0 or b_ > 0):
            continue
        (amx if t_ > 0 else dthr if t_ < 0 else dbrk)["%g" % v] = float(acc_l[i])
    log(f"[limits] throttle +1 {amx} | throttle -1 {dthr} | brake {dbrk}")

    # (4) response times: env 0 throttle step, env 1 steering step
    v0 = np.zeros(N)
    v0[:2] = 6.0
    start(v0)
    for _ in range(80):
        vx, _, _ = body()
        veh.set_inputs(throttle=T(hold_throttle(v0, vx) * (v0 > 0)), brake=T(np.zeros(N)),
                       steer=T(np.zeros(N)))
        vs.step()
    vx, _, _ = body()
    thr0 = hold_throttle(6.0, vx[0])
    vs0, rs1 = [], []
    for _ in range(int(1.5 / DT)):
        vx, _, r = body()
        th = hold_throttle(v0, vx) * (v0 > 0)
        th[0] = thr0 + 0.3
        st = np.zeros(N)
        st[1] = -0.2
        veh.set_inputs(throttle=T(th), brake=T(np.zeros(N)), steer=T(st))
        vs.step()
        vx, _, r = body()
        vs0.append(vx[0])
        rs1.append(r[1])
    ag = np.gradient(np.array(vs0), DT)
    tau_a = t63(ag / ag[int(0.8 / DT):].mean(), DT)
    rs1 = np.array(rs1)
    tau_s = t63(rs1 / rs1[int(1.0 / DT):].mean(), DT)
    log(f"[response] acceleration t63 {tau_a:.3f} s, yaw rate t63 {tau_s:.3f} s")

    dtr = phys.resolved.drivetrain
    driven = getattr(dtr, "driven_axles", None)
    # (v2.0.2) closed-form brake gain for the gear-mode follower's stage 2.
    from .brake import brake_gain_closed_form
    k_b = brake_gain_closed_form(phys.resolved, wm, mass)
    log(f"[brake] closed-form brake gain {k_b:.4f} m/s^2 per unit brake (measured "
        f"brake-1 acceleration {dbrk})")
    out = dict(
        schema=SCHEMA, sdk_version=genesis_vehicle.__version__, genesis_version=gs.__version__,
        urdf=os.path.abspath(urdf_path), preset=getattr(preset_fn, "__name__", str(preset_fn)),
        config=getattr(config_mod, "__file__", None),
        plant_overrides=dict(getattr(overrides_fn, "overrides", {}) or {}),
        dt=DT, substeps=int(substeps),
        mass=mass, izz_cg=izz_cg, wheelbase=L, rig_offset=-ax_x[ra], l_rig_to_cg=LR,
        max_steer_rad=MAXS, wheel_radius=R, steer_input_sign=sign,
        front_cornering_stiffness=caf, rear_cornering_stiffness=car, cornering_samples=rows,
        accel_max_by_speed=amx, decel_throttle_by_speed=dthr, decel_brake_by_speed=dbrk,
        accel_tau=tau_a, steer_yaw_tau=tau_s, steering=steering, n_axles=n_axles,
        tyre=tyre,
        axle_loads={"settled": {str(k): v for k, v in loads.items()},
                    "moment_balance": {str(k): v for k, v in balance.items()},
                    "front_axle": int(fa), "rear_axle": int(ra)},
        deadbeat_scale=s_db, v_star=v_star,
        drivetrain={"type": type(dtr).__name__,
                    "driven_axles": None if driven is None else [int(a) for a in driven]},
        fit_residual=resid,
        envelope={"a_y_range": resid["a_y_range"], "speed_range": resid["speed_range"]},
        brake_gain=k_b,
    )
    return MPCVehicleParams.from_dict(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m genesis_vehicle.control.identify",
        description="Measure a 2-axle Ackermann vehicle's MPC parameters (mpc_params JSON).")
    ap.add_argument("--urdf", required=True, help="vehicle URDF path")
    ap.add_argument("--preset", required=True, help="genesis_vehicle preset name")
    ap.add_argument("--config", default=None,
                    help="optional override file (apply_config / apply_runtime_config), "
                         "as for sweep_measure")
    ap.add_argument("--output", required=True, help="output JSON path")
    ap.add_argument("--dt", type=float, default=DEFAULT_DT, help="sim dt (default 0.025)")
    ap.add_argument("--substeps", type=int, default=DEFAULT_SUBSTEPS, help="solver substeps (default 4)")
    ap.add_argument("--gpu", action="store_true", help="use the GPU backend")
    add_plant_override_args(ap)
    args = ap.parse_args(argv)
    if not os.path.exists(args.urdf):
        raise FileNotFoundError(f"URDF not found: {args.urdf}")
    preset_fn = _load_preset(args.preset)
    config_mod = _load_config_module(args.config)
    overrides_fn = plant_overrides_fn(args, args.urdf)
    from genesis_vehicle import VehicleScene
    VehicleScene.init_backend("gpu" if args.gpu else "cpu")
    p = identify_mpc_params(args.urdf, preset_fn, config_mod, dt=args.dt,
                            substeps=args.substeps, overrides_fn=overrides_fn)
    p.to_json(args.output)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
