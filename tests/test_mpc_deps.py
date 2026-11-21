"""The MPC layer's optional dependencies: gated, named, never loaded by import.

Mutant this file must fail on (plan section 11): ``import casadi`` added at
the top of ``control/mpc/__init__.py``.

A missing ``mpc_params`` raises the U1 ``ValueError`` even with the
dependencies blocked (check 6 runs before check 8).
"""
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control.mpc import _deps  # noqa: E402

_SDK_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.mark.parametrize("impl,blocked", [("nonlinear", ["casadi", "do_mpc"]),
                                          ("linear", ["osqp"]),
                                          ("linear", ["threadpoolctl"])])
def test_blocked_dependency_raises_import_error_naming_it(monkeypatch, impl, blocked):
    for name in blocked:
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError) as ei:
        _deps.require_mpc_deps(impl)
    msg = str(ei.value)
    assert "requirements-mpc.txt" in msg
    for name in blocked:
        assert name in msg
    assert 'mpc_model="off"' in msg


def test_cached_do_mpc_is_not_enough_when_casadi_is_blocked(monkeypatch):
    # api-review: a cached do_mpc imports fine with casadi blocked, so the gate
    # must check casadi itself.
    monkeypatch.setitem(sys.modules, "casadi", None)
    with pytest.raises(ImportError, match="casadi"):
        _deps.require_mpc_deps("nonlinear")


def test_solver_construction_is_gated(monkeypatch):
    from genesis_vehicle.control.mpc import alpasim_nonlinear, alpasim_linear
    monkeypatch.setitem(sys.modules, "do_mpc", None)
    with pytest.raises(ImportError, match="do_mpc"):
        alpasim_nonlinear.NonlinearMPC()
    monkeypatch.setitem(sys.modules, "osqp", None)
    with pytest.raises(ImportError, match="osqp"):
        alpasim_linear.LinearMPC()


def test_unknown_impl():
    with pytest.raises(ValueError, match="nonlinear"):
        _deps.require_mpc_deps("quadratic")


def test_import_loads_no_optional_dependency():
    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {_SDK_PARENT!r})\n"
        "import genesis_vehicle\n"
        "import genesis_vehicle.control.mpc\n"
        "from genesis_vehicle.control.mpc import MPCVehicleParams, MPCGains\n"
        "import genesis_vehicle.control.mpc.law\n"
        "print(json.dumps([m for m in ('casadi', 'do_mpc', 'osqp', 'threadpoolctl') if m in sys.modules]))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    loaded = json.loads(out.stdout.strip().splitlines()[-1])
    assert loaded == [], f"imported eagerly: {loaded}"


def test_u1_comes_before_the_dependency_check(monkeypatch):
    from genesis_vehicle.control import PathFollower
    from genesis_vehicle.tests.test_sweep_table import make_table
    for name in ("casadi", "do_mpc", "osqp"):
        monkeypatch.setitem(sys.modules, name, None)
    path = [(0.0, 0.0, 0.0, 1.0), (5.0, 0.0, 0.0, 1.0)]
    with pytest.raises(ValueError, match="mpc_params="):
        PathFollower(path, make_table())
    from genesis_vehicle.tests.data.make_mpc_field_replay import PARAMS_SDK
    from genesis_vehicle.control.mpc import MPCVehicleParams
    with pytest.raises(ImportError, match="requirements-mpc.txt"):
        PathFollower(path, make_table(), mpc_params=MPCVehicleParams.from_dict(dict(PARAMS_SDK)))
