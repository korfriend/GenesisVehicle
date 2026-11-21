"""The MPC layer of the path follower (2.0.0): parameters, solvers, control law.

Importing this package needs only numpy: the parameter record
(:class:`MPCVehicleParams`), the gains (:class:`MPCGains`) and the
dependency gate (:func:`require_mpc_deps`) are imported eagerly; the solvers
(``NonlinearMPC`` — casadi/do_mpc; ``LinearMPC`` — osqp/scipy/threadpoolctl)
and the law are resolved on first attribute access, so ``import
genesis_vehicle`` never loads an optional dependency.
"""
from __future__ import annotations

from ._deps import require_mpc_deps
from .params import SCHEMA, MPCGains, MPCVehicleParams, accel_limits, load_mpc_params

_LAZY = {
    "NonlinearMPC": (".alpasim_nonlinear", "NonlinearMPC"),
    "LinearMPC": (".alpasim_linear", "LinearMPC"),
    "SolveResult": (".alpasim_nonlinear", "SolveResult"),
}

__all__ = ["MPCGains", "MPCVehicleParams", "SCHEMA", "accel_limits", "load_mpc_params",
           "require_mpc_deps", *_LAZY]


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(mod, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
