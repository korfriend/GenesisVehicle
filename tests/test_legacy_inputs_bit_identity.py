"""AC-1b / AC-1c (v2.0.2): signed (``gear=None``) inputs are bit-identical to the
capture — the 2.0.4 capture since v2.0.4 (the 2.0.1 one through 2.0.3).

What it proves: on the platform and genesis-world version the fixture was
captured on, 300 scripted steps of LEGACY inputs (negative throttle, brake,
steer; Python scalars at n_envs 1, per-env throttle tensors at n_envs 3) give,
byte for byte, the captured pose / quat / velocity / angular velocity / wheel
omega / ``last_T_drive`` / ``last_T_brake`` of every vehicle after every step —
through ``VehicleScene.step`` on both solvers x both raycast modes x n_envs
1 and 3 (the batched scene holds two car kinds' worth: the reference car twice
= ONE kind with K=2, plus the reference tank = a second kind, so the
multi-kind path runs), and through ``VehiclePhysics.step`` called directly with
injected ``distances=``. AC-1c: the same scenes with car "b" given ``gear=1``
(it only ever gets throttle >= 0): car "a" and the tank, which still pass
``gear=None``, stay byte-identical to the capture — a gear row in the kind does
not touch the legacy rows.

What it does NOT prove: anything on another platform / genesis version (the
fixture is a float32 bit pattern; an uncaptured pair SKIPS, like
``test_fan_default_identity.py``), or anything about gear mode itself.

Fixture: ``tests/data/legacy_inputs_2_0_4.json``, captured 2026-11-23 from the
2.0.4 working tree (pedal-free ``StaticFrictionLock``) with this file's scenario
code (``run_scene``), genesis-world 1.4.5, CPU, linux-x86_64 (WSL2). Each vehicle
entry holds a SHA-256 over all 300 steps and the raw values at
``CHECKPOINTS`` (the values only make a failure readable).

Why it was re-captured (2.0.4): the lock change moves all three vehicles
(first at checkpoint 179 for both cars, 239 / 299 for the tank). It was written
only after a PHYSICAL acceptance proof on all 9 scenes and all three vehicles
(plan r204 §6.1, harness ``.pm_r2/r204/fixture_proof204.py`` outside the repo):
per vehicle, env and braked segment the per-wheel torque balance
(H = sum max(T_b - |T_d|, 0), U = sum max(|T_d| - T_b, 0)) predicts REST or
MOVE; no lock hold may be applied on a wheel the drive overpowers; no wheel
may turn against its contact while its brake dominates; post-stop back-travel
<= 2 mm; and on a run with every braked segment stretched by 160 steps the
REST-predicted segments must be at rest (<= 1e-4 m/s over 80 steps) and the
MOVE-predicted ones must not. ``direct_single_3`` injects a 1 cm, 0.45 Hz
ray-distance forcing, so its base link cannot be at rest on any tree; for that
scene (amended rule) the full proof ran with the forcing off, the hold / turning
checks with it on, and the net base-link movement over 4 whole forcing periods
was bounded (<= 1 mm, and <= the 2.0.3 value + 0.1 mm where 2.0.3 had its lock
engaged). This scene is not a rest measurement. ``legacy_inputs_2_0_1.json``
stays in the repository and is no longer asserted.
"""
import hashlib
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
N_STEPS = 300
CHECKPOINTS = (59, 119, 179, 239, 299)
SCENES = [  # (name, solver, raycast_mode, n_envs, direct)
    ("pv_dual_1", "per_vehicle", "dual_scene", 1, False),
    ("pv_dual_3", "per_vehicle", "dual_scene", 3, False),
    ("pv_single_1", "per_vehicle", "single_scene", 1, False),
    ("pv_single_3", "per_vehicle", "single_scene", 3, False),
    ("b_dual_1", "batched", "dual_scene", 1, False),
    ("b_dual_3", "batched", "dual_scene", 3, False),
    ("b_single_1", "batched", "single_scene", 1, False),
    ("b_single_3", "batched", "single_scene", 3, False),
    ("direct_single_3", "per_vehicle", "single_scene", 3, True),
]
ENV_SCALE = {"a": (1.0, 0.5, -0.5), "b": (1.0, 0.5, 0.25), "tank": (1.0, -0.5, 0.75)}


def script(who, i):
    """(throttle, brake, steer) python floats for vehicle ``who`` at step ``i``."""
    if i < 40:
        return 0.0, 1.0, 0.0
    if who == "a":
        if i < 120: return 0.6, 0.0, 0.2
        if i < 160: return 0.0, 0.5, 0.0
        if i < 240: return -0.5, 0.0, -0.3
        return 0.3, 0.2, 0.1
    if who == "b":                          # throttle >= 0 only
        if i < 150: return 0.4, 0.0, -0.2
        if i < 200: return 0.0, 0.7, 0.0
        return 0.8, 0.0, 0.1
    if i < 140: return 0.5, 0.0, 0.3        # tank
    if i < 220: return -0.6, 0.0, -0.2
    return 0.2, 0.6, 0.0


def inputs_for(who, i, n_envs, gs):
    thr, brk, st = script(who, i)
    if n_envs == 1:
        return thr, brk, st
    import torch
    sc = torch.tensor(ENV_SCALE[who], device=gs.device, dtype=gs.tc_float)
    return thr * sc, brk, st


def _rows(vs, veh):
    """(omega, last_T_drive, last_T_brake) of this vehicle, (n_envs, n_wheels)."""
    if veh.physics is not None:
        p = veh.physics
        return p.omega, p.last_T_drive, p.last_T_brake
    mvp = vs._mvp
    for flat_i, kind_idx, slot in mvp._flat_to_kind:
        if flat_i == veh._slot:
            kind = mvp.kinds[kind_idx]
            r = kind.slot_rows(slot)
            p = kind._proto
            return p.omega[r], p.last_T_drive[r], p.last_T_brake[r]
    raise AssertionError("vehicle not in the batched driver")


def _snap(vs, veh):
    om, td, tb = _rows(vs, veh)
    parts = [veh.get_pos(), veh.get_quat(), veh.get_vel(), veh.get_ang(), om, td, tb]
    return [t.detach().cpu().numpy().astype(np.float32).reshape(-1) for t in parts]


NAMES = ("pos", "quat", "vel", "ang", "omega", "T_drive", "T_brake")


def run_scene(name, solver, raycast_mode, n_envs, direct, *, gear_b=False):
    """Drive one scene; returns {who: {"sha": hex, "cp": {step: {field: [...]}}}}.
    ``gear_b``: car "b" gets ``gear=1`` (only on a tree that has gear mode)."""
    import genesis as gs
    import torch
    from genesis_vehicle import VehicleInputs, VehicleScene, car_4w_rwd_ackermann, tank_skid_belt
    from genesis_vehicle.raycast import read_distances
    import genesis_vehicle
    car_urdf = os.path.join(os.path.dirname(genesis_vehicle.__file__), "samples", "urdf", "car_4w.urdf")
    tank_urdf = os.path.join(os.path.dirname(genesis_vehicle.__file__), "tests", "data", "tank_ref.urdf")
    VehicleScene.init_backend("cpu")
    vs = VehicleScene(dt=0.025, substeps=4, n_envs=n_envs, raycast_mode=raycast_mode,
                      show_viewer=False, solver=solver)
    vs.add_ground_plane(friction=1.0)
    mat = gs.materials.Rigid(friction=1.0)
    vehs = {
        "a": vs.add_vehicle(car_urdf, car_4w_rwd_ackermann, pos=(0.0, 0.0, 0.6), material=mat, name="a"),
        "b": vs.add_vehicle(car_urdf, car_4w_rwd_ackermann, pos=(0.0, 8.0, 0.6), material=mat, name="b"),
        "tank": vs.add_vehicle(tank_urdf, tank_skid_belt, pos=(0.0, -10.0, 1.5), material=mat, name="tank"),
    }
    vs.build()
    hs = {w: hashlib.sha256() for w in vehs}
    cps = {w: {} for w in vehs}
    for i in range(N_STEPS):
        if direct:
            for w, veh in vehs.items():
                thr, brk, st = inputs_for(w, i, n_envs, gs)
                d, hit = read_distances(veh.sensor, n_envs, return_hit=True)
                bump = 0.01 * torch.sin(torch.arange(d.shape[1], device=d.device, dtype=d.dtype)
                                        + 0.07 * i)
                d = torch.where(hit, d + bump, d)
                veh.physics.step(VehicleInputs(throttle=thr, brake=brk, steer=st), distances=d)
            vs.scene.step()
        else:
            for w, veh in vehs.items():
                thr, brk, st = inputs_for(w, i, n_envs, gs)
                if gear_b and w == "b":
                    veh.set_inputs(throttle=thr, brake=brk, steer=st, gear=1)
                else:
                    veh.set_inputs(throttle=thr, brake=brk, steer=st)
            vs.step()
        for w, veh in vehs.items():
            arrs = _snap(vs, veh)
            for a in arrs:
                hs[w].update(a.tobytes())
            if i in CHECKPOINTS:
                cps[w][str(i)] = {n: [float(x) for x in a] for n, a in zip(NAMES, arrs)}
    return {w: {"sha": hs[w].hexdigest(), "cp": cps[w]} for w in vehs}


FIXTURE = os.path.join(HERE, "data", "legacy_inputs_2_0_4.json")


def _platform_key():
    import platform
    import sys
    return f"{sys.platform}-{platform.machine()}"


def _fixture():
    import json
    pytest.importorskip("genesis")
    import genesis
    with open(FIXTURE) as fh:
        fx = json.load(fh)
    if (fx["genesis"], fx["platform"]) != (genesis.__version__, _platform_key()):
        pytest.skip(f"legacy fixture captured on genesis-world {fx['genesis']} / "
                    f"{fx['platform']}; this is {genesis.__version__} / {_platform_key()}")
    return fx


def _compare(name, who, got, want):
    if got["sha"] == want["sha"]:
        return
    for step, fields in want["cp"].items():
        for field_name, vals in fields.items():
            g = got["cp"][step][field_name]
            assert g == vals, (f"{name}/{who}: {field_name} at step {step} differs from "
                               f"the capture:\n  captured {vals}\n  now      {g}")
    raise AssertionError(f"{name}/{who}: per-step digest differs from the capture (the "
                         f"checkpoint values agree, so a non-checkpoint step moved)")


@pytest.mark.parametrize("scene", SCENES, ids=[s[0] for s in SCENES])
def test_legacy_inputs_are_bit_identical_to_the_capture(scene):
    fx = _fixture()
    got = run_scene(*scene)
    want = fx["scenes"][scene[0]]
    for who in ("a", "b", "tank"):
        _compare(scene[0], who, got[who], want[who])


GEAR_SCENES = [s for s in SCENES if not s[4]]


@pytest.mark.parametrize("scene", GEAR_SCENES, ids=[s[0] for s in GEAR_SCENES])
def test_a_gear_vehicle_leaves_the_legacy_rows_bit_identical(scene):
    fx = _fixture()
    got = run_scene(*scene, gear_b=True)
    want = fx["scenes"][scene[0]]
    for who in ("a", "tank"):
        _compare(scene[0] + "+gear_b", who, got[who], want[who])
