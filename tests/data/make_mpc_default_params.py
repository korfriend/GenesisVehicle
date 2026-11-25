"""Capture harness for ``tests/data/mpc_default_params_2_0_6.json`` (the 2.1.0 I-2 identity fixture).

Captured ONCE on the 2.0.6 tree (``d68ef4d``), before any 2.1.0 change: every field of the solver-side
parameter set the DEFAULT ``PathFollower`` hands its solver (NMPC in gear and in signed mode, LMPC) and of
``MPCVehicleParams.solver_params`` at its defaults and at ``steer_cap=0.5, dt_ctrl=0.025``, for the test
record ``make_mpc_field_replay.PARAMS_SDK`` and the synthetic ``test_sweep_table.make_table()``.
``tests/test_mpc_default_identity.py`` requires the 2.1.0 tree to reproduce every field with ``==``.
Floats are stored by JSON (shortest repr, exact round trip).

    python genesis_vehicle/tests/data/make_mpc_default_params.py --output mpc_default_params_2_0_6.json

Needs casadi, do_mpc, osqp (the follower builds its solvers); no Genesis.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
_SDK_PARENT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _SDK_PARENT not in sys.path:
    sys.path.insert(0, _SDK_PARENT)

import numpy as np  # noqa: E402

PATH = [(float(x), 0.0, 0.0, 3.0) for x in np.arange(0.0, 30.0, 0.5)]


def as_record(p) -> dict:
    """``{"type": class name, "fields": dataclasses.asdict with tuples as lists}``."""
    def conv(v):
        if isinstance(v, (tuple, list)):
            return [conv(x) for x in v]
        return v
    return {"type": type(p).__name__, "fields": {k: conv(v) for k, v in dataclasses.asdict(p).items()}}


def capture() -> dict:
    from genesis_vehicle.control import PathFollower
    from genesis_vehicle.control.mpc import MPCVehicleParams
    from genesis_vehicle.tests.data.make_mpc_field_replay import PARAMS_SDK
    from genesis_vehicle.tests.test_sweep_table import make_table
    rec = MPCVehicleParams.from_dict(copy.deepcopy(PARAMS_SDK))
    out = {"solver_params_default": as_record(rec.solver_params()),
           "solver_params_bicycle_cap05_dt0025": as_record(rec.solver_params("bicycle", steer_cap=0.5, dt_ctrl=0.025))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, kw in (("follower_nmpc_gear", {}), ("follower_nmpc_signed", {"command_mode": "signed"}),
                         ("follower_lmpc_gear", {"mpc_impl": "linear"})):
            f = PathFollower(PATH, make_table(), mpc_params=MPCVehicleParams.from_dict(copy.deepcopy(PARAMS_SDK)), **kw)
            out[name] = as_record(f._law.solver.p)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    if os.path.exists(a.output):
        raise SystemExit(f"{a.output} exists; the fixture is captured once on 2.0.6 and never re-captured")
    from genesis_vehicle._version import __version__
    d = {"captured_on": {"sdk": __version__, "tree": "d68ef4d"}, **capture()}
    with open(a.output, "w") as fh:
        json.dump(d, fh, indent=1)
    print("wrote", a.output)
