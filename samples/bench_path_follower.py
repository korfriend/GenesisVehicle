"""bench_path_follower.py — how often does a PathFollower control step overrun 25 ms here?

Measures, on THIS machine, the share of control steps whose control time plus
simulation time exceeds the 25 ms control period (dt 0.025), for the MPC
layer's solvers (``nonlinear`` = do_mpc/IPOPT, ``linear`` = OSQP) and the 1.x
PD follower (``pd``), on the ``mpc_follow_demo`` course (reference car,
sweep table in stage 2, CPU unless ``--gpu``).

    python -m genesis_vehicle.samples.bench_path_follower                 # nonlinear,linear x 5
    python -m genesis_vehicle.samples.bench_path_follower --arms nonlinear,linear,pd --repeats 5

| abbr | meaning |
|---|---|
| tot>25 | % of control steps (first 20 excluded) where control + simulation > 25 ms |
| ctrl>25 | the same for the control time alone |
| A / B | runs on odd / even rounds (the arm order is reversed on even rounds) |

How it measures (the ``bench_raycast_mode.py`` discipline): one FRESH PROCESS
per measurement (each runs ``mpc_follow_demo.run``), one process at a time,
the arm order alternating between rounds. It ALWAYS prints every run's values.
It prints a per-arm MEDIAN only if the two order groups agree — each group's
median lies inside the other group's [min, max] — and otherwise says the
number is not measurable here. That is a disclosure heuristic, not a
statistical test. A failed worker aborts the run with no aggregate.

It measures a closed loop that also simulates, so the numbers depend on the
simulation cost of this machine, the background load, and the thread settings
of the process (``OPENBLAS_NUM_THREADS`` / ``OMP_NUM_THREADS`` are reported).
Nothing here makes the nonlinear MPC real-time; the bench reports how often it
is not.
"""
from __future__ import annotations

import sys
import pathlib
_SDK_PARENT = str(pathlib.Path(__file__).resolve().parents[2])
if _SDK_PARENT not in sys.path:
    sys.path.insert(0, _SDK_PARENT)

import argparse
import json
import os
import subprocess

import numpy as np

ARMS = {"nonlinear": dict(mpc_model="bicycle", mpc_impl="nonlinear"),
        "linear": dict(mpc_model="bicycle", mpc_impl="linear"),
        "pd": dict(mpc_model="off")}
METRICS = ("tot_over25_pct", "ctrl_over25_pct", "ctrl_mean_ms", "solve_p50_ms", "solve_p95_ms",
           "solve_max_ms")


def schedule(arms, repeats):
    """``[(round, arm), ...]``: forward order on odd rounds, reversed on even."""
    out = []
    for r in range(1, repeats + 1):
        out += [(r, a) for a in (arms if r % 2 == 1 else arms[::-1])]
    return out


def publication(values_a, values_b):
    """Fail-closed rule: publish ``(median, min, max, n)`` of all runs only if
    each group's median lies inside the other group's [min, max]."""
    if not values_a or not values_b:
        return None, "needs runs in both order groups"
    ma, mb = float(np.median(values_a)), float(np.median(values_b))
    if min(values_b) <= ma <= max(values_b) and min(values_a) <= mb <= max(values_a):
        allv = list(values_a) + list(values_b)
        return (float(np.median(allv)), float(min(allv)), float(max(allv)), len(allv)), "ok"
    return None, f"order groups disagree (A median {ma:.3g}, B median {mb:.3g})"


def _worker(a):
    from genesis_vehicle.samples import mpc_follow_demo as demo
    res = demo.run(table=a.table, params=a.params, t_max=a.t_max, gpu=a.gpu,
                   log=lambda *x, **k: None, **ARMS[a.worker])
    print("MPC_RESULT " + json.dumps(res, default=str), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", default="nonlinear,linear")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--table", default=None)
    ap.add_argument("--params", default=None)
    ap.add_argument("--t-max", type=float, default=40.0)
    ap.add_argument("--json", default=None, help="write all runs + the aggregate here")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        return _worker(a)
    from genesis_vehicle.samples import mpc_follow_demo as demo
    if a.table is None or a.params is None:
        ct, cp = demo.ensure_cache(demo.URDF, "car_4w_rwd_ackermann", gpu=a.gpu)
        a.table, a.params = a.table or ct, a.params or cp
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    for x in arms:
        if x not in ARMS:
            raise SystemExit(f"unknown arm {x!r}; choose from {sorted(ARMS)}")
    runs = []
    for r, arm in schedule(arms, a.repeats):
        cmd = [sys.executable, "-m", "genesis_vehicle.samples.bench_path_follower", "--worker", arm,
               "--table", a.table, "--params", a.params, "--t-max", str(a.t_max)]
        if a.gpu:
            cmd.append("--gpu")
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=_SDK_PARENT)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("MPC_RESULT ")), None)
        if p.returncode != 0 or line is None:
            print(f"[round {r} {arm}] worker FAILED (rc {p.returncode}):\n{p.stderr[-2000:]}")
            print("No aggregate is produced: a benchmark that drops a failed worker publishes "
                  "a biased number.")
            return 2
        res = json.loads(line[len("MPC_RESULT "):])
        if not res["done"]:
            print(f"[round {r} {arm}] the run did not finish the course; aborting (no aggregate)")
            return 3
        runs.append(dict(round=r, arm=arm, **res))
        print(f"[round {r} {arm:9s}] tot>25 {res['tot_over25_pct']:6.2f} %  ctrl>25 "
              f"{res['ctrl_over25_pct']:6.2f} %  ctrl mean {res['ctrl_mean_ms']:6.2f} ms  solve p50/p95/max "
              f"{res['solve_p50_ms']}/{res['solve_p95_ms']}/{res['solve_max_ms']} ms  CTE "
              f"{res['cte_mean']:.3f}/{res['cte_max']:.3f} m", flush=True)
    agg = {}
    print("\naggregate (published only where the two order groups agree):")
    for arm in arms:
        agg[arm] = {}
        for m in METRICS:
            va = [x[m] for x in runs if x["arm"] == arm and x["round"] % 2 == 1 and x[m] is not None]
            vb = [x[m] for x in runs if x["arm"] == arm and x["round"] % 2 == 0 and x[m] is not None]
            pub, why = publication(va, vb)
            agg[arm][m] = dict(published=pub, reason=why, A=va, B=vb)
            vals = sorted(va + vb)
            txt = (f"{pub[0]:.3g} [{pub[1]:.3g} .. {pub[2]:.3g}] n={pub[3]}" if pub
                   else f"not measurable here ({why}); values {['%.3g' % v for v in vals]}")
            print(f"  {arm:9s} {m:16s} {txt}")
    env = {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}
    print(f"conditions: CPU={not a.gpu}, env threads {env}, repeats {a.repeats}, "
          f"table {a.table}, params {a.params}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(dict(runs=runs, aggregate=agg, env=env), fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
