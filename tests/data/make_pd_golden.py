"""Capture harness for ``tests/data/pd_golden_1x.json`` (the 1.x PD golden).

The golden is the per-step output of the 1.x ``PathFollower`` (the PD law,
the path state machine, the KICK and the brake-at-rest rule) on Genesis-free
scenarios: synthetic paths driven in closed loop by a kinematic unicycle,
through a synthetic ``SweepTable`` and through a fake coupled plant.
``tests/test_pd_golden.py`` replays the same scenarios and compares every
step with ``==``.

It was captured ONCE on the unmodified 1.8.0 tree (``3810c79``), before any
2.0.0 edit to ``control/path_follower.py``. Re-running it on a later tree
re-captures THAT tree, which defeats the purpose; it is kept so the capture
is reproducible from the 1.8.0 tree::

    python genesis_vehicle/tests/data/make_pd_golden.py --output pd_golden_1x.json

Pure Python + numpy; no Genesis, no MPC dependencies. Everything the
scenarios need is defined here (the table model, the plant, the integrator),
so a later change to another test helper cannot move the golden.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SDK_PARENT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _SDK_PARENT not in sys.path:
    sys.path.insert(0, _SDK_PARENT)

from genesis_vehicle.control import PathFollower  # noqa: E402
from genesis_vehicle.control.sweep import SweepTable  # noqa: E402

SCHEMA = "genesis_vehicle.pd_golden/1"


# --- synthetic plant models (fixed here, not imported) ------------------------

def _a_model(thr, v, pitch):
    return 3.0 * np.asarray(thr) - 0.5 * np.asarray(v) - 0.1 * np.asarray(pitch)


def _w_model(steer):
    return 2.0 * np.asarray(steer)


def make_table() -> SweepTable:
    v_grid = [-3.0, -1.5, 0.0, 1.5, 3.0, 5.0]
    ts_grid = [-1.0, -0.5, 0.0, 0.5, 1.0]
    pitch_grid = [-10.0, 0.0, 10.0]
    roll_grid = [0.0]
    v, t, s, p, r = np.meshgrid(v_grid, ts_grid, ts_grid, pitch_grid, roll_grid,
                                indexing="ij")
    cols = {"v_init": v.ravel(), "throttle": t.ravel(), "steer": s.ravel(),
            "pitch": p.ravel(), "roll": r.ravel()}
    cols["a_measured"] = _a_model(cols["throttle"], cols["v_init"], cols["pitch"])
    cols["omega_z_measured"] = _w_model(cols["steer"])
    return SweepTable(cols)


class FakeCoupledPlant:
    """A deterministic stand-in for ``DifferentiablePlant``: a coupled
    ``solve`` whose answer depends on the last command reported through
    ``set_applied`` — so the ORDER of ``finish``'s report is observable."""
    has_coupled_solve = True

    def __init__(self):
        self.steer_range = (-1.0, 1.0)
        self.applied = (0.0, 0.0)
        self.n_applied = 0

    def solve(self, v, a, w, pitch_deg=0.0, roll_deg=0.0):
        thr = (a + 0.5 * v + 0.1 * pitch_deg) / 3.0 + 0.05 * w + 0.1 * self.applied[0]
        steer = w / 2.0 - 0.02 * a + 0.1 * self.applied[1]
        lo, hi = self.steer_range
        return max(-1.0, min(1.0, thr)), max(lo, min(hi, steer))

    def set_applied(self, throttle, steer):
        self.applied = (float(throttle), float(steer))
        self.n_applied += 1

    # The inverse surface ``_as_plant`` duck-types on; a coupled follower
    # never calls these (it calls ``solve``), and they say so if it does.
    def throttle_for(self, v, a, pitch_deg, roll_deg):
        raise AssertionError("coupled plant: throttle_for must not be called")

    def steer_for(self, v, w, pitch_deg, roll_deg):
        raise AssertionError("coupled plant: steer_for must not be called")


# --- scenarios ----------------------------------------------------------------

def densify(corners, spacing=0.5):
    path = []
    for (x0, y0, z0, s0), (x1, y1, z1, _s1) in zip(corners[:-1], corners[1:]):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / spacing))
        for k in range(n):
            t = k / n
            path.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0),
                         z0 + t * (z1 - z0), s0))
    path.append(corners[-1])
    return path


def scenarios():
    """name -> (path, follower kwargs, start pose (x, y, yaw), max steps)."""
    straight = densify([(0.0, 0.0, 0.0, 2.0), (20.0, 0.0, 0.0, 2.0)])
    square = densify([(0.0, 0.0, 0.0, 2.0), (20.0, 0.0, 0.0, 2.0),
                      (20.0, 20.0, 0.0, 2.0), (0.0, 20.0, 0.0, 2.0),
                      (0.0, 8.0, 0.0, 2.0)])
    cusp = (densify([(0.0, 0.0, 0.0, 1.5), (10.0, 0.0, 0.0, 1.5)])
            + densify([(10.0, 0.0, 0.0, -1.5), (6.0, -4.0, 0.0, -1.5)]))
    fwd = densify([(0.0, 0.0, 0.0, 2.0), (10.0, 0.0, 0.0, 2.0)])
    bwd = [(x, y, z, s, 0.0) for (x, y, z, s) in
           densify([(10.0, 0.0, 0.0, -1.0), (5.0, 0.0, 0.0, -1.0)])]
    fwd2 = densify([(5.0, 0.0, 0.0, 2.0), (15.0, 0.0, 0.0, 2.0)])
    explicit_yaw = fwd + bwd + fwd2
    # Zero target speed everywhere: v_target = 1e-6, no KICK, a ~ 0, so the
    # brake-at-rest rule (|throttle| < 0.02 and |v| < 0.1) fires in DRV+1.
    at_rest = densify([(0.0, 0.0, 0.0, 0.0), (10.0, 0.0, 0.0, 0.0)])
    sc = {
        "at_rest": (at_rest, {}, (0.0, 0.0, 0.0), 60),
        "straight": (straight, {}, (0.0, 0.3, 0.05), 1500),
        "square": (square, {}, (0.0, 0.0, 0.0), 900),
        "square_tuned": (square, dict(lookahead=2.5, k_v=1.5, k_w=2.0,
                                      steer_cap=0.7, arrival_goal=1.0),
                         (0.0, -0.4, 0.1), 900),
        "cusp": (cusp, {}, (0.0, 0.0, 0.0), 4000),
        "explicit_yaw_cusp": (explicit_yaw, dict(cusp_goal=0.8, k_approach=1.2,
                                                 v_stop=0.08),
                              (0.0, 0.0, 0.0), 6000),
    }
    # KICK threshold sweep: straight starts at a range of target speeds,
    # integrated at a fine step (KICK_DT) so the speed crosses
    # 0.4 * v_target within a few hundredths of a m/s — a small change of the
    # KICK threshold moves at least one of these runs.
    for i in range(13):
        v = 1.0 + 0.25 * i
        sc[f"kick_{v:.2f}"] = (densify([(0.0, 0.0, 0.0, v), (8.0, 0.0, 0.0, v)]), {},
                               (0.0, 0.0, 0.0), 150)
    return sc


KICK_DT = 0.01


def run_scenario(name, plant_kind, follower_kwargs=None):
    """Build the follower for scenario ``name`` and run it; returns
    ``(follower, records)``. ``follower_kwargs`` are added to the scenario's
    own constructor arguments (the 2.0.0 tree passes ``mpc_model="off"``)."""
    path, kw, (x0, y0, yaw0), n = scenarios()[name]
    plant = make_table() if plant_kind == "table" else FakeCoupledPlant()
    f = PathFollower(path, plant, **kw, **(follower_kwargs or {}))
    pitch_fn = (lambda k: 4.0 * math.sin(0.01 * k)) if name == "square" else None
    dt = KICK_DT if name.startswith("kick_") else 0.05
    return f, run_unicycle(f, x0, y0, yaw0, dt=dt, max_steps=n, pitch_fn=pitch_fn)


def run_unicycle(follower, x, y, yaw, dt=0.05, max_steps=4000, pitch_fn=None):
    """Closed loop on the unicycle the synthetic table describes. Returns the
    list of per-step records (inputs and outputs)."""
    v = 0.0
    rec = []
    for k in range(max_steps):
        pitch = pitch_fn(k) if pitch_fn else 0.0
        thr, steer, brake = follower.step((x, y), yaw, v, pitch, 0.0)
        rec.append([x, y, yaw, v, pitch, float(thr), float(steer), float(brake),
                    follower.last_mode, follower.block_i, follower.current_idx])
        if follower.last_mode == "DONE":
            break
        if brake > 0.5:
            v *= 0.5
            if abs(v) < 1e-3:
                v = 0.0
        else:
            v += (3.0 * thr - 0.5 * v - 0.1 * pitch) * dt
        yaw += (2.0 * steer) * dt
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
    return rec


def capture():
    out = {}
    for name in scenarios():
        for plant_kind in ("table", "coupled"):
            f, rec = run_scenario(name, plant_kind)
            extra = {}
            if plant_kind == "coupled":
                extra = {"n_applied": f.plant.n_applied,
                         "applied": list(f.plant.applied)}
            out[f"{name}/{plant_kind}"] = {"steps": rec, **extra}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    import platform
    import genesis_vehicle
    data = {
        "schema": SCHEMA,
        "captured_from": "genesis_vehicle " + genesis_vehicle.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "record": ["x", "y", "yaw", "v_long", "pitch_deg", "throttle", "steer",
                   "brake", "last_mode", "block_i", "current_idx"],
        "runs": capture(),
    }
    with open(args.output, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    n = sum(len(r["steps"]) for r in data["runs"].values())
    print(f"saved {len(data['runs'])} runs, {n} steps -> {args.output}")


if __name__ == "__main__":
    main()
