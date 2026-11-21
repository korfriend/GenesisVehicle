"""Capture harness for ``tests/data/mpc_field_replay.json`` (the P1 regression fixture).

Runs the FIELD controller (``PathFollower_MPC/mpc/mpc_follower.py`` —
``MPCFollower`` with the nonlinear MPC, unmodified, read from the field folder)
for 200 control steps against a synthetic ``SweepTable`` and a synthetic
numpy plant, and records every step: the state fed in, and the field's
outputs (stage-1 command, stage-2 target, ``(throttle, steer, brake)``,
mode, solve status). ``tests/test_mpc_field_replay.py`` feeds the recorded
states to the SDK port and requires the same outputs to 1e-9.

An IPOPT solution is a floating-point result of this machine's CPU, BLAS and
compiler output, so the fixture carries a CAPTURE PLATFORM KEY
(``sys.platform``-``platform.machine()``, casadi and do_mpc versions, the
``/proc/cpuinfo`` model name); the test skips on any other key. Capture and
replay both run with ``OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`` set before
the process starts (this harness re-executes itself with them if needed).

    python genesis_vehicle/tests/data/make_mpc_field_replay.py \\
        --field-dir /path/to/PathFollower_MPC --output mpc_field_replay.json

Needs casadi, do_mpc and the field folder; no Genesis.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SDK_PARENT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _SDK_PARENT not in sys.path:
    sys.path.insert(0, _SDK_PARENT)

SCHEMA = "genesis_vehicle.mpc_field_replay/1"
THREAD_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
DT = 0.025
N_STEPS = 200
STEER_CAP = 0.5
RIG = 1.35

#: A car-like synthetic params record (SDK schema). Values are plausible for
#: the reference car but are NOT a measurement of it.
PARAMS_SDK = {
    "schema": "genesis_vehicle.mpc_params/1", "sdk_version": "synthetic",
    "genesis_version": "synthetic", "urdf": "synthetic", "preset": "synthetic",
    "config": None, "plant_overrides": {}, "dt": DT, "substeps": 4,
    "mass": 1330.0, "izz_cg": 2714.4, "wheelbase": 2.7, "rig_offset": RIG,
    "l_rig_to_cg": 1.36, "max_steer_rad": 0.7, "wheel_radius": 0.358,
    "steer_input_sign": -1.0, "front_cornering_stiffness": 33200.0,
    "rear_cornering_stiffness": 32500.0, "cornering_samples": [],
    "accel_max_by_speed": {"0": 1.9, "3": 1.75, "6": 1.63, "10": 1.47, "14": 1.29},
    "decel_throttle_by_speed": {"3": -2.15, "6": -2.16, "10": -2.18, "14": -2.2},
    "decel_brake_by_speed": {"3": -5.17, "6": -5.17, "10": -5.19, "14": -5.21},
    "accel_tau": 0.015, "steer_yaw_tau": 0.08, "steering": "Ackermann", "n_axles": 2,
}


def platform_key() -> dict:
    import casadi
    import do_mpc
    cpu = None
    try:
        with open("/proc/cpuinfo") as fh:
            cpu = next((ln.split(":", 1)[1].strip() for ln in fh
                        if ln.startswith("model name")), None)
    except OSError:
        pass
    return {"platform": f"{sys.platform}-{platform.machine()}",
            "casadi": casadi.__version__, "do_mpc": do_mpc.__version__, "cpu": cpu}


# --- synthetic table and plant --------------------------------------------------

def _a_model(thr, v, pitch):
    return 2.4 * np.asarray(thr) - 0.04 * np.asarray(v) - 0.1 * np.asarray(pitch)


def _w_model(steer, v):
    v = np.asarray(v)
    return -0.12 * np.asarray(steer) * (1.0 + np.abs(v))


def make_table():
    from genesis_vehicle.control.sweep import SweepTable
    vg = [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    tg = [-1.0, -0.5, 0.0, 0.5, 1.0]
    v, t, s, p, r = np.meshgrid(vg, tg, tg, [0.0], [0.0], indexing="ij")
    cols = {"v_init": v.ravel(), "throttle": t.ravel(), "steer": s.ravel(),
            "pitch": p.ravel(), "roll": r.ravel()}
    cols["a_measured"] = _a_model(cols["throttle"], cols["v_init"], cols["pitch"])
    cols["omega_z_measured"] = _w_model(cols["steer"], cols["v_init"])
    return SweepTable(cols)


def course():
    pts = [(x, 0.0) for x in np.arange(0.0, 8.0, 0.1)]
    R = 15.0
    for th in np.arange(0.0, math.pi / 2, 0.1 / R):
        pts.append((8.0 + R * math.sin(th), R - R * math.cos(th)))
    P = np.array(pts)
    S = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    V = np.minimum(5.0, np.sqrt(2 * 1.2 * (S + 0.5)))
    return [(float(x), float(y), 0.0, float(v)) for (x, y), v in zip(P[::5], V[::5])]


def _capture(field_dir):
    sys.path.insert(0, os.path.join(field_dir, "mpc"))
    from mpc_follower import MPCFollower, params_from_json  # field code
    from alpasim_mpc import accel_limits as field_accel_limits
    field_json = {k: v for k, v in PARAMS_SDK.items()
                  if k not in ("schema", "genesis_version", "plant_overrides", "steering", "n_axles")}
    field_json["steer_cap"] = 1.0
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(field_json, fh)
        pj = fh.name
    try:
        pm = params_from_json(pj, DT)
    finally:
        os.unlink(pj)
    pm.steer_max = STEER_CAP * float(field_json["max_steer_rad"])   # field demo.py:97
    path = course()
    tab = make_table()
    fol = MPCFollower(path, tab, pm, rig_offset=RIG, dt_ctrl=DT, steer_cap=STEER_CAP,
                      impl="nonlinear")
    x, y, yaw, v, r, vy = 0.0, 0.2, 0.03, 0.0, 0.0, 0.0
    rec = []
    for k in range(N_STEPS):
        t = k * DT
        pitch = 1.5 * math.sin(0.05 * k)
        n0 = fol.n_solve
        thr, steer, brake = fol.step((x, y), yaw, v, pitch, 0.0, yaw_rate=r, v_lat=vy, t=t)
        lo, hi = field_accel_limits(fol.mpc.p, max(v, 0.0))
        a_t = float(np.clip(fol.cmd["a"], -9.0 if lo is None else lo, 6.0 if hi is None else hi))
        w_t = max(v, 0.0) * math.tan(fol.cmd["delta"]) / fol.L
        rec.append([t, x, y, yaw, v, pitch, 0.0, r, vy,
                    float(thr), float(steer), float(brake), fol.last_mode,
                    int(fol.n_solve > n0), fol.cmd["st"] if fol.n_solve > n0 else None,
                    float(fol.cmd["delta"]), float(fol.cmd["a"]), a_t, w_t])
        if fol.last_mode == "DONE":
            break
        # synthetic plant: the table's own model + a lagged lateral slip
        a = float(_a_model(thr, v, pitch)) - (6.0 * v if brake > 0.5 and abs(v) < 0.5 else 0.0)
        w = float(_w_model(steer, v))
        v = v + a * DT
        r = r + (w - r) * min(1.0, DT / 0.05)
        vy = vy + (0.08 * r * v - vy) * min(1.0, DT / 0.1)
        yaw = yaw + r * DT
        x = x + (v * math.cos(yaw) - vy * math.sin(yaw)) * DT
        y = y + (v * math.sin(yaw) + vy * math.cos(yaw)) * DT
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--field-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    if any(os.environ.get(k) != v for k, v in THREAD_ENV.items()):
        env = dict(os.environ, **THREAD_ENV)
        sys.exit(subprocess.call([sys.executable, os.path.abspath(__file__)]
                                 + (argv if argv is not None else sys.argv[1:]), env=env))
    rec = _capture(os.path.abspath(args.field_dir))
    data = {"schema": SCHEMA, "platform_key": platform_key(), "thread_env": THREAD_ENV,
            "dt": DT, "steer_cap": STEER_CAP, "rig_offset": RIG, "params": PARAMS_SDK,
            "record": ["t", "x", "y", "yaw", "v_long", "pitch", "roll", "yaw_rate", "v_lat",
                       "throttle", "steer", "brake", "mode", "solved", "field_status",
                       "cmd_delta", "cmd_a", "a_t", "w_t"],
            "steps": rec}
    with open(args.output, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    modes = sorted({s[12] for s in rec})
    print(f"saved {len(rec)} steps, {sum(s[13] for s in rec)} solves, modes {modes} -> {args.output}")


if __name__ == "__main__":
    main()
