"""P1 regression: the SDK MPC port replays the field controller's recorded run to 1e-9.

``tests/data/mpc_field_replay.json`` holds 200 control steps of the FIELD
``MPCFollower`` (nonlinear MPC) driving a synthetic plant through a synthetic
``SweepTable``, captured by ``tests/data/make_mpc_field_replay.py``. Here the
recorded states are fed to the SDK port — ``NonlinearMPC`` + ``_MPCLaw`` with
``on_solve_failure="hold"`` (field parity), the same table, the field finish —
and every step's stage-1 command, stage-2 target, ``(throttle, steer, brake)``
and mode must match to 1e-9 — once through ``_MPCLaw`` + the finish rule, and
once through the public ``PathFollower`` (2.0.0).

Scope: an IPOPT result is a floating-point result of one machine, so the
comparison runs only on the fixture's CAPTURE PLATFORM KEY (platform tag,
casadi / do_mpc versions, CPU model name) and SKIPS elsewhere, printing why.
The replay runs in a subprocess started with ``OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1`` — the capture's thread setting.

What the fixture does NOT exercise (reported by the capture): MPC-HOLD and any
non-converged IPOPT status — those branches are pinned only by the fake-solver
tests (``test_mpc_law``, ``test_mpc_status_classes``).

Mutants this file must fail on (plan section 11): ``MPCGains.idx_start_penalty``
10 -> 9; the platform-key check removed (``test_forged_platform_key_skips``
then fails on a fixture forged with another key and other outputs).
"""
import copy
import json
import os
import subprocess
import sys
import tempfile

import pytest

pytest.importorskip("casadi")
pytest.importorskip("do_mpc")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.tests.data import make_mpc_field_replay as rp  # noqa: E402

_FIXTURE = os.path.join(os.path.dirname(__file__), "data", "mpc_field_replay.json")
_SDK_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOL = 1e-9

_REPLAY = r'''
import json, math, sys
sys.path.insert(0, SDK_PARENT)
from genesis_vehicle.control.mpc import MPCVehicleParams
from genesis_vehicle.control.mpc.alpasim_nonlinear import NonlinearMPC
from genesis_vehicle.control.mpc.law import _MPCLaw, finish_command
from genesis_vehicle.tests.data import make_mpc_field_replay as rp
fx = json.load(open(FIXTURE))
sp = MPCVehicleParams.from_dict(fx["params"]).solver_params(
    "bicycle", steer_cap=fx["steer_cap"], dt_ctrl=fx["dt"])
law = _MPCLaw(NonlinearMPC(sp, yaw_bound=math.pi / 2), rig_offset=fx["rig_offset"],
              on_solve_failure="hold", v_floor=0.5)
path = rp.course()
law.start_block(path, t0=0.0)
goal = path[-1]
tab = rp.make_table()
out = []
for s in fx["steps"]:
    t, x, y, yaw, v, pitch, roll, r, vy = s[:9]
    if math.hypot(x - goal[0], y - goal[1]) < 1.5:
        out.append([0.0, 0.0, 1.0, "DONE", 0, law.cmd_delta, law.cmd_a, None, None])
        break
    o = law.step(t, (x, y), yaw, v, r, vy)
    thr = tab.throttle_for(v, o.a_t, pitch, roll)
    st = tab.steer_for(v, o.w_t, pitch, roll)
    thr, st, brake, mode = finish_command(v, o.a_t, thr, st, fx["steer_cap"])
    out.append([thr, st, brake, mode, int(o.solved), law.cmd_delta, law.cmd_a, o.a_t, o.w_t])
print("REPLAY " + json.dumps(out))
'''

_REPLAY_PF = r'''
import json, sys
sys.path.insert(0, SDK_PARENT)
from genesis_vehicle.control import PathFollower
from genesis_vehicle.control.mpc import MPCVehicleParams
from genesis_vehicle.tests.data import make_mpc_field_replay as rp
fx = json.load(open(FIXTURE))
fol = PathFollower(rp.course(), rp.make_table(), mpc_params=MPCVehicleParams.from_dict(fx["params"]),
                   mpc_model="bicycle", on_solve_failure="hold", mpc_v_floor=0.5,
                   steer_cap=fx["steer_cap"])
out = []
for s in fx["steps"]:
    t, x, y, yaw, v, pitch, roll, r, vy = s[:9]
    n0 = fol._law.n_solve
    thr, st, brake = fol.step((x, y), yaw, v, pitch, roll, yaw_rate=r, v_lat=vy, t=t)
    out.append([thr, st, brake, fol.last_mode, int(fol._law.n_solve > n0),
                fol._law.cmd_delta, fol._law.cmd_a, None, None])
    if fol.last_mode == "DONE":
        break
print("REPLAY " + json.dumps(out))
'''


def _load():
    with open(_FIXTURE) as fh:
        return json.load(fh)


def replay_or_skip(fixture: dict, script=None):
    """Replay ``fixture`` and compare; ``pytest.skip`` on another platform key."""
    assert fixture["schema"] == rp.SCHEMA
    here = rp.platform_key()
    if fixture["platform_key"] != here:
        pytest.skip(f"fixture captured on {fixture['platform_key']}, this machine is {here}: "
                    "an IPOPT result is only reproducible on its capture platform")
    fd, path = tempfile.mkstemp(suffix=".json", prefix="mpc_replay_")
    with os.fdopen(fd, "w") as fh:
        json.dump(fixture, fh)
    try:
        code = (script or _REPLAY).replace("SDK_PARENT", repr(_SDK_PARENT)).replace("FIXTURE", repr(path))
        env = dict(os.environ, **fixture["thread_env"])
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env=env, timeout=600)
    finally:
        os.unlink(path)
    assert p.returncode == 0, p.stderr[-3000:]
    got = json.loads(next(ln for ln in p.stdout.splitlines() if ln.startswith("REPLAY "))[7:])
    want = fixture["steps"]
    assert len(got) == len(want), (len(got), len(want))
    for k, (g, w) in enumerate(zip(got, want)):
        thr, st, brake, mode, solved, cd, ca, a_t, w_t = g
        exp = (w[9], w[10], w[11], w[15], w[16], w[17], w[18])
        for name, a, b in zip(("throttle", "steer", "brake", "cmd_delta", "cmd_a", "a_t", "w_t"),
                              (thr, st, brake, cd, ca, a_t, w_t), exp):
            if a is None:                       # the PathFollower replay does not expose a_t / w_t
                continue
            assert abs(a - b) <= TOL, f"step {k}: {name} {a!r} != field {b!r}"
        assert mode == w[12], f"step {k}: mode {mode} != field {w[12]}"
        assert solved == w[13], f"step {k}: solve cadence differs"


def test_sdk_law_replays_the_field_run():
    replay_or_skip(_load())


def test_path_follower_replays_the_field_run():
    """The same replay through the public 2.0.0 PathFollower (stage 2 = the
    synthetic SweepTable, field finish, ``on_solve_failure="hold"``)."""
    replay_or_skip(_load(), _REPLAY_PF)


def test_forged_platform_key_skips():
    fx = copy.deepcopy(_load())
    fx["platform_key"] = dict(fx["platform_key"], cpu="Some Other CPU @ 1.0GHz")
    for s in fx["steps"]:                       # what another machine might have recorded
        s[9] += 1e-6
    with pytest.raises(pytest.skip.Exception):
        replay_or_skip(fx)


def test_fixture_is_not_vacuous():
    fx = _load()
    steps = fx["steps"]
    assert len(steps) == 200 and sum(s[13] for s in steps) == 50
    assert max(s[4] for s in steps) > 5.0       # the dynamic branch (v >= 5 m/s) is reached
    assert fx["thread_env"] == {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
