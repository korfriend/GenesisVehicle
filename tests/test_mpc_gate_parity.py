"""Heading gate / watchdog vs the 2.0.0 path, real nonlinear MPC, closed loop (v2.0.1).

Each run drives a KINEMATIC bicycle plant (not the ray-wheel vehicle) with
``PathFollower.plan()`` — the stage-1 targets are integrated directly
(acceleration clamped to [-3, 1.9] m/s^2, yaw rate to the 0.35 rad steering
limit, wheelbase 2.7 m), as in the plan-review probe
``.pm_r2/r210_probe/review1/corner_epsi_probe.py``. MPC-HOLD is the field rule
on the stage-1 output (standing and the plan wants to slow -> no acceleration).
Synthetic in-repo params and table. Courses: straight 20 m, a vertex with
heading change 90 / 150 deg, straight 20 m (0.5 m spacing, 4 m/s), and a gentle
arc (straight 20 m, R 30 m quarter circle, 4 m/s); aligned start from rest.

* 90 deg vertex and the gentle arc: the default (gate + watchdog on) and both
  ``None`` give the SAME per-step targets and modes (``==``) and no handover —
  evaluating the gate does not touch the law;
* 150 deg vertex: the default hands over (``handover["align"] >= 1``) and
  reaches DONE; with both ``None`` the MPC stands in MPC-HOLD (the 2.0.0
  mid-path stall) for at
  least 400 steps mid-path (measured 1024 on linux-x86_64, Ultra 9 285H,
  casadi 3.8.2, do_mpc 5.1.2 — an IPOPT path can differ on another platform;
  if that assertion fails there, record the platform).

Mutants this file must fail on (plan_r210 section 6): the gate evaluated on
block entry only (the 150 deg assertions); a gate evaluation that touches the
law state (the 90 deg ``==``); F3 removed or F4 restored at the 2.0.1-draft
threshold (the 5 m leg ends on ``align_budget``).
"""
import copy
import math
import os
import sys

import pytest

pytest.importorskip("casadi")
pytest.importorskip("do_mpc")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control import PathFollower  # noqa: E402
from genesis_vehicle.control.mpc import MPCVehicleParams  # noqa: E402
from genesis_vehicle.tests.data import make_mpc_field_replay as rp  # noqa: E402

DT, L, DMAX = 0.025, 2.7, 0.35


def vertex(deg, spacing=0.5, v=4.0, leg=20.0):
    th = math.radians(deg)
    corners = [(0.0, 0.0), (20.0, 0.0), (20.0 + leg * math.cos(th), leg * math.sin(th))]
    path = []
    for (x0, y0), (x1, y1) in zip(corners[:-1], corners[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / spacing))
        path += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n, 0.0, v) for i in range(n)]
    path.append((*corners[-1], 0.0, v))
    return path


def arc(R=30.0, v=4.0):
    pts = [(0.5 * i, 0.0) for i in range(40)]
    n = int(R * math.pi / 2 / 0.5)
    pts += [(20.0 + R * math.sin(i / n * math.pi / 2), R - R * math.cos(i / n * math.pi / 2))
            for i in range(n + 1)]
    return [(x, y, 0.0, v) for x, y in pts]


def run(path, gate_on, t_max=60.0):
    kw = {} if gate_on else dict(mpc_align_gate=None, mpc_stall_timeout=None)
    f = PathFollower(path, rp.make_table(),
                     mpc_params=MPCVehicleParams.from_dict(copy.deepcopy(rp.PARAMS_SDK)),
                     command_mode="signed", **kw)
    x = y = yaw = v = 0.0
    out, mode = [], None
    for k in range(int(t_max / DT)):
        pl = f.plan((x, y), yaw, v, 0.0, 0.0, yaw_rate=0.0, v_lat=0.0, t=k * DT)
        if pl.command is not None:
            mode, a, w = pl.mode, (-3.0 if v > 0 else 0.0), 0.0
        else:
            a, w, mode = pl.a_target, pl.omega_target, pl.mode
            if pl.regime == "R-MPC" and v < 0.1 and pl.a_ff <= 0.0:
                a, mode = 0.0, "MPC-HOLD"
        out.append((a, w, mode))
        if mode == "DONE":
            break
        a = max(-3.0, min(1.9, a))
        wmax = abs(v) * math.tan(DMAX) / L
        w = max(-wmax, min(wmax, w))
        v = max(0.0, v + a * DT)
        yaw += w * DT
        x += v * math.cos(yaw) * DT
        y += v * math.sin(yaw) * DT
    return out, f.mpc_stats["handover"], f


@pytest.mark.parametrize("name,path", [("v90", vertex(90.0)), ("arc", arc())])
def test_gentle_courses_are_unchanged(name, path):
    on, ho_on, _ = run(path, True)
    off, ho_off, _ = run(path, False)
    assert on[-1][2] == "DONE"
    assert ho_on == {"align": 0, "stall": 0} and ho_off == {"align": 0, "stall": 0}
    assert on == off


def test_150_deg_vertex_hands_over_and_finishes():
    on, ho_on, _ = run(vertex(150.0), True)
    assert ho_on["align"] >= 1 and on[-1][2] == "DONE"
    assert any(m == "PD-ALIGN" for _, _, m in on)
    off, ho_off, _ = run(vertex(150.0), False)
    assert ho_off == {"align": 0, "stall": 0}
    assert sum(m == "MPC-HOLD" for _, _, m in off) >= 400


def test_150_deg_vertex_with_a_5_m_leg_ends_on_the_align_budget():
    """T1 (v2.0.1): with only 5 m of path after the vertex the R-ALIGN hand-over
    is not refused (the first 2.0.1 draft's no-room refusal was deleted); the
    PD turn ends on the F3 travel budget.

    Mutants this fails on: F3 removed; the deleted no-room refusal restored at
    the 2.0.1-draft threshold (R_min * |e_psi| + arrival_goal)."""
    out, ho, f = run(vertex(150.0, leg=5.0), True)
    assert out[-1][2] == "MPC-ABORT" and f.stop_reason == "align_budget"
    assert ho["align"] >= 1 and any(m == "PD-ALIGN" for _, _, m in out)
