"""2.1.0 ``mpc_follow_demo`` course geometry and CLI guards (no Genesis).

C3: ``demo_course(lead_in=L, final_len=F, turn=T)`` reproduces, ``==``, (a) the 2.0.6 ``demo_course`` at
the defaults (copied verbatim below), and (b) the Step-0 capture harness formula (copied verbatim from the
2.1.0 acceptance harness, which emulated these flags on the 2.0.6 tree) for L in {20, 60, 80}, F in {15, 3},
both turn directions; plus the waypoint sha256 recorded by that harness on 2.0.6 for three courses.
A10 / L1: the CLI exits 2 for the refused flag combinations, before any simulation.

Mutant this file must fail on: a default waypoint change (e.g. the lead-in default moved, the final
straight's sample count changed, the right-turn mirror applied to x).
"""
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.samples.mpc_follow_demo import demo_course  # noqa: E402


def demo_course_2_0_6(v=8.0, ramp=1.0, radius=30.0):
    """``samples/mpc_follow_demo.demo_course`` of 2.0.6 (d68ef4d), verbatim."""
    pts = [(x, 0.0) for x in np.arange(0.0, 20.0, 0.1)]
    R = float(radius)
    for th in np.arange(0.0, math.pi / 2, 0.1 / R):
        pts.append((20.0 + R * math.sin(th), R - R * math.cos(th)))
    for d in np.arange(0.0, 15.0 + 1e-9, 0.1):
        pts.append((20.0 + R, R + d))
    P = np.array(pts)
    S = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    V = np.minimum(v, np.sqrt(2 * ramp * (S + 0.5)))
    return [(float(x), float(y), 0.0, float(s)) for (x, y), s in zip(P[::5], V[::5])], P


def harness_course(lead_, fin_, mir, v=8.0, ramp=1.0, radius=30.0):
    """The Step-0 capture harness's course (``.pm_r2/r21/tools/head_harness.py``), verbatim."""
    R = float(radius)
    pts = [(x, 0.0) for x in np.arange(0.0, lead_, 0.1)]
    for th in np.arange(0.0, math.pi / 2, 0.1 / R):
        pts.append((lead_ + R * math.sin(th), mir * (R - R * math.cos(th))))
    for d in np.arange(0.0, fin_ + 1e-9, 0.1):
        pts.append((lead_ + R, mir * (R + d)))
    P = np.array(pts); S = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    V = np.minimum(v, np.sqrt(2 * ramp * (S + 0.5)))
    return [(float(x), float(y), 0.0, float(s)) for (x, y), s in zip(P[::5], V[::5])], P


def _eq(a, b):
    assert a[0] == b[0]
    assert a[1].shape == b[1].shape and np.array_equal(a[1], b[1])


def test_default_waypoints_are_the_2_0_6_waypoints():
    for args in ((), (8.0, 1.0, 30.0), (10.0, 1.0, 17.0), (12.0, 0.5, 8.5)):
        _eq(demo_course(*args), demo_course_2_0_6(*args))


@pytest.mark.parametrize("lead", [20.0, 60.0, 80.0])
@pytest.mark.parametrize("fin", [15.0, 3.0])
@pytest.mark.parametrize("turn", ["left", "right"])
def test_waypoints_equal_the_step0_harness(lead, fin, turn):
    for v, R in ((8.0, 30.0), (10.0, 11.3), (12.0, 10.0), (10.0, 18.8333)):
        _eq(demo_course(v, 1.0, R, lead_in=lead, final_len=fin, turn=turn),
            harness_course(lead, fin, -1.0 if turn == "right" else 1.0, v, 1.0, R))


@pytest.mark.parametrize("kw,sha", [
    (dict(v=8.0, radius=30.0), "de4230345f636d6d6f3ab0ed65881b876f8872d1414d9a9117b19d8555036797"),
    (dict(v=10.0, radius=30.0, lead_in=60.0), "639e5ff8f8555115df36a867a30c979a1b667a40eccdc4e52f6f91d3865bf5ca"),
    (dict(v=12.0, radius=10.0, lead_in=80.0), "869daafaf69b72aeb6364513ea4e79afe9eae0991f60e3eacd33808962584aeb"),
    (dict(v=10.0, radius=14.1667, lead_in=60.0, final_len=3.0), "da1c82dda8922808dc1a78a9caad811bb8f9f0865ba57ae6873064367cf08658"),
])
def test_waypoint_hash_recorded_on_2_0_6(kw, sha):
    path, _ = demo_course(**kw)
    assert hashlib.sha256(json.dumps([list(map(float, p)) for p in path]).encode()).hexdigest() == sha


def test_bad_turn_raises():
    with pytest.raises(ValueError, match="turn"):
        demo_course(turn="up")


@pytest.mark.parametrize("argv", [
    ["--decel-limit", "brake", "--mpc-model", "off"],
    ["--decel-limit", "throttle", "--mpc-model", "off"],
    ["--decel-limit", "brake", "--command-mode", "signed"],
    ["--mu-lat", "0.6"],
    ["--mu-long", "0.6", "--table", "x.csv"],
])
def test_cli_refuses_with_exit_2(argv):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    p = subprocess.run([sys.executable, "-m", "genesis_vehicle.samples.mpc_follow_demo", *argv], cwd=root,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    assert p.returncode == 2, (p.returncode, p.stderr[-500:])
