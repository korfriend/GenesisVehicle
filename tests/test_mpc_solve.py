"""Real solvers: NMPC (do_mpc/IPOPT) and LMPC (OSQP) on simple references, a
Genesis-free closed loop, and the process-global state they touch.

* straight from rest: positive acceleration, ~zero steering; left turn:
  positive (CCW) steering; the per-speed acceleration limit is respected;
* the law + a synthetic table + a numpy plant reach the goal of a curved path
  with CTE (cross-track error) max < 0.3 m;
* no casadi FutureWarning escapes (``-W error::FutureWarning`` equivalent),
  and casadi's NumPy mode is restored after setup and after every solve;
* casadi's OpenBLAS thread cap reaches the copy IPOPT uses: read back through
  a second handle opened by soname (api-review C5).

Mutant this file must fail on (plan section 11): the NumPy-mode restore
removed from ``_deps.casadi_numpy_mode``. (Also fails if the OpenBLAS handle is
opened by file path instead of soname.)
"""
import math
import os
import sys
import warnings

import numpy as np
import pytest

casadi = pytest.importorskip("casadi")
pytest.importorskip("do_mpc")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control.mpc import _deps  # noqa: E402
from genesis_vehicle.control.mpc.alpasim_nonlinear import NonlinearMPC  # noqa: E402
from genesis_vehicle.control.mpc.law import _MPCLaw, finish_command  # noqa: E402
from genesis_vehicle.control.mpc.params import accel_limits  # noqa: E402
from genesis_vehicle.tests._mpc_fakes import solver_params  # noqa: E402
from genesis_vehicle.tests.data import make_mpc_field_replay as rp  # noqa: E402


def _straight(v=0.0):
    x = np.linspace(0.0, 20.0, 21) * 0.1 * max(v, 3.0)
    return np.column_stack([x, np.zeros(21), np.zeros(21)])


def _left(v=6.0, R=20.0):
    s = np.arange(21) * 0.1 * v
    th = s / R
    return np.column_stack([R * np.sin(th), R - R * np.cos(th), th])


def _solvers():
    out = [("nonlinear", lambda: NonlinearMPC(solver_params(), yaw_bound=math.pi / 2))]
    try:
        import osqp  # noqa: F401
        import threadpoolctl  # noqa: F401
        from genesis_vehicle.control.mpc.alpasim_linear import LinearMPC
        out.append(("linear", lambda: LinearMPC(solver_params())))
    except ImportError:
        pass
    return out


@pytest.mark.parametrize("name,make", _solvers(), ids=lambda v: v if isinstance(v, str) else "")
def test_straight_from_rest_and_left_turn(name, make):
    s = make()
    r = s.solve(np.zeros(8), _straight())
    assert r.exception is None and r.u[1] > 0.0 and abs(r.u[0]) < 1e-3, r
    s2 = make()
    r2 = s2.solve(np.array([0, 0, 0, 6.0, 0, 0, 0, 0]), _left())
    assert r2.u[0] > 0.0, r2


def test_nmpc_respects_the_per_speed_acceleration_limit():
    s = NonlinearMPC(solver_params(), yaw_bound=math.pi / 2)
    for v in (0.0, 2.0, 5.0, 9.0, 13.0):
        far = np.column_stack([np.linspace(0, 80, 21), np.zeros(21), np.zeros(21)])
        r = s.solve(np.array([0, 0, 0, v, 0, 0, 0, 0]), far)
        lo, hi = accel_limits(s.p, v)
        assert r.u[1] <= hi + 1e-4, (v, r.u, hi)


def test_closed_loop_on_a_synthetic_plant_reaches_the_goal():
    pts = [(x, 0.0) for x in np.arange(0.0, 8.0, 0.1)]
    R = 25.0
    for th in np.arange(0.0, math.pi / 3, 0.1 / R):
        pts.append((8.0 + R * math.sin(th), R - R * math.cos(th)))
    P = np.array(pts)
    S = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    V = np.minimum(4.0, np.sqrt(2 * 1.0 * (S + 0.5)))
    path = [(float(x), float(y), 0.0, float(v)) for (x, y), v in zip(P[::5], V[::5])]
    tab = rp.make_table()
    law = _MPCLaw(NonlinearMPC(solver_params(), yaw_bound=math.pi / 2), rig_offset=rp.RIG,
                  on_solve_failure="pd")
    law.start_block(path, t0=0.0)
    x, y, yaw, v, r, vy = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt, cte, done = 0.025, [], False
    for k in range(int(30 / dt)):
        if math.hypot(x - path[-1][0], y - path[-1][1]) < 1.5:
            done = True
            break
        o = law.step(k * dt, (x, y), yaw, v, r, vy)
        assert o.mode == "MPC", (k, law.last_status)
        thr = tab.throttle_for(v, o.a_t, 0.0, 0.0)
        st = tab.steer_for(v, o.w_t, 0.0, 0.0)
        thr, st, brake, _ = finish_command(v, o.a_t, thr, st, 0.5)
        a = float(rp._a_model(thr, v, 0.0)) if brake < 0.5 else -2.0 * v
        r = float(rp._w_model(st, v))
        v = max(0.0, v + a * dt)
        yaw += r * dt
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        cte.append(float(np.hypot(P[:, 0] - x, P[:, 1] - y).min()))
    assert done, "never reached the goal"
    assert max(cte) < 0.3, max(cte)


def test_no_future_warning_and_numpy_mode_restored():
    go = casadi.GlobalOptions
    old = go.getNumpyMode()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        s = NonlinearMPC(solver_params(), yaw_bound=math.pi / 2)
        assert go.getNumpyMode() == old
        s.solve(np.zeros(8), _straight())
    assert go.getNumpyMode() == old


def test_numpy_mode_restored_even_when_the_block_raises():
    go = casadi.GlobalOptions
    old = go.getNumpyMode()
    with pytest.raises(RuntimeError):
        with _deps.casadi_numpy_mode(-1 if old != -1 else 0):
            raise RuntimeError("x")
    assert go.getNumpyMode() == old


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="casadi OpenBLAS soname is Linux-only")
def test_openblas_cap_reaches_the_live_copy():
    import ctypes
    NonlinearMPC(solver_params(), yaw_bound=math.pi / 2)   # maps the library (nlpsol)
    lib = _deps.casadi_openblas()
    assert lib is not None
    probe = ctypes.CDLL("libcasadi-tp-openblas.so.0", mode=os.RTLD_NOLOAD | ctypes.RTLD_GLOBAL)
    probe.openblas_get_num_threads.restype = ctypes.c_int
    before = probe.openblas_get_num_threads()
    target = 2 if before != 2 else 3
    with _deps.casadi_openblas_limit(target):
        assert probe.openblas_get_num_threads() == target
    assert probe.openblas_get_num_threads() == before
