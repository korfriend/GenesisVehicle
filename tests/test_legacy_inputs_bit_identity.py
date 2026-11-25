"""AC-1b / AC-1c (v2.0.2): signed (``gear=None``) inputs match the capture bit
for bit — since v2.0.6 the capture from the 2.0.6 tree on genesis-world 1.5.0
(through v2.0.5 the capture from the v2.0.4 tree on 1.5.0; the 2.0.4 capture on
1.4.5 through v2.0.4, the 2.0.1 one through 2.0.3).

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

Fixture (v2.0.6): ``tests/data/legacy_inputs_2_0_6.json``. Captured from the
2.0.6 tree (the ``StaticFrictionLock`` reversal re-anchor removed and the ground
gate ``ctx.N > 0`` added; before the version bump) on genesis-world 1.5.0
(linux-x86_64), with this file's own ``run_scene``, signed inputs; CPU, WSL2;
2026-11-24 (harness ``.pm_r2/r206/h/capture206.py`` outside the repo). The gear
rows were checked at capture time: with car "b" in gear, "a" and the tank gave
the signed capture's digests on all 8 gear scenes. It is a regression lock on
exactly this engine version and platform, not evidence that the 2.0.6 physics
is right: see "Why it was re-captured in 2.0.6" below for the evidence that was
required before it was written.

Fixture through v2.0.5: ``tests/data/legacy_inputs_2_0_5.json``. Captured from the
v2.0.4 tree + the two-line ``skip_forward`` patch on genesis-world 1.5.0
(linux-x86_64); 2.0.5 reproduces it bit for bit. (2026-11-24: ``git
archive 9e5c4d5`` with ``skip_forward=`` deleted from the two
``set_base_links_pos`` / ``_quat`` calls in ``vehicle_scene.py``; this file's own
``run_scene`` from that tree, signed inputs; CPU, WSL2. The gear rows were
checked at capture time: with car "b" in gear, "a" and the tank gave the signed
capture's digests on all 8 gear scenes.) Each vehicle entry holds a SHA-256
over all 300 steps and the raw values at ``CHECKPOINTS`` (the values only make
a failure readable). On another engine version, a 1.5.x patch release included,
or another platform, the tests SKIP.

``direct_single_3`` — the tank: through 2.0.5 this entry encoded the +1
branch of ``StaticFrictionLock``'s travel sign and was round-off-branch
sensitive. At touchdown the lock recorded ``_s_eng = sign(v_long)`` while
``v_long`` was free-fall round-off (instrumented by plan-review: -8.0e-10 m/s
on 1.4.5, which took the -1 branch, +2.5e-9 m/s on 1.5.0, which takes +1); the
two branches end 0.126 m apart, and a 1e-6 m spawn shift on 1.4.5 already
reproduced the 1.5.0 outcome to 2e-5 m. That was the knife-edge (present on
2.0.4 and 2.0.5, engine-independent). 2.0.6 removed the reversal re-anchor
that read the sign (and ``_s_eng`` with it); the tree with only that removal
reproduces ``legacy_inputs_2_0_5.json`` byte for byte (all 9 scenes, measured on
1.5.0, linux-x86_64); the 2.0.6 capture comes from the tree that also has the
ground gate. ``tests/test_sfl_touchdown_continuity.py`` checks that the tank's
final position no longer depends on the sign of a tiny perturbation. The lock
still has discrete decisions (the creep-test takeover step), so this entry is
not claimed to be free of every round-off sensitivity.
The 2.0.5 capture measured against the 1.4.5 capture (final position, same scenario): the two
cars of ``direct_single_3`` within 7.7e-4 m, every vehicle of the other 8
scenes within 1.3e-4 m (CPU, WSL2; a comparison, not a bound for other scenes).

Fixture history: ``legacy_inputs_2_0_5.json`` (genesis-world 1.5.0, asserted
v2.0.5), ``legacy_inputs_2_0_4.json`` (genesis-world 1.4.5, asserted v2.0.4) and
``legacy_inputs_2_0_1.json`` (asserted through v2.0.3) stay in the repository,
no longer asserted.

Why it was re-captured in 2.0.6 (on 1.5.0): the lock's ground gate moves all
three vehicles of every scene (first at checkpoint 59). Measured before the
capture, against ``legacy_inputs_2_0_5.json`` (CPU/WSL2; harnesses in
``.pm_r2/r206`` outside the repo): the 2.0.6 tree with only the gate removed
reproduces ``legacy_inputs_2_0_5.json`` byte for byte (all 17 tests), so every
difference is the gate's (the first differing step, 12 for the cars and 17
for the tank, also comes after the first step on which a braked wheel had no
normal force, 0 or 1 — but every scene starts with a braked drop, so that
ordering does not discriminate and is not evidence); the 2.0.4 physical
acceptance proof described below (``fixture_proof204.py``, unstretched and
stretched by 160 steps; for ``direct_single_3`` the amended rule, with 2.0.5
in place of 2.0.3 and its net-movement check run stretched by 400 steps)
passes on the 9 signed scenes with no lock
engagement or takeover on a wheel at ``N <= 0``; and the largest position
change at a checkpoint is 1.92 mm (``direct_single_3``, car "b"), 7.3e-5 m in
every other scene. The re-captured file is not itself evidence; those checks
are, and they cover the 9 signed scenes only.

Why it was re-captured in 2.0.4 (on 1.4.5): the lock change moves all three vehicles
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
engaged). This scene is not a rest measurement.
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


FIXTURE = os.path.join(HERE, "data", "legacy_inputs_2_0_6.json")


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
def test_legacy_inputs_match_the_2_0_6_capture_on_1_5_0(scene):
    fx = _fixture()
    got = run_scene(*scene)
    want = fx["scenes"][scene[0]]
    for who in ("a", "b", "tank"):
        _compare(scene[0], who, got[who], want[who])


GEAR_SCENES = [s for s in SCENES if not s[4]]


@pytest.mark.parametrize("scene", GEAR_SCENES, ids=[s[0] for s in GEAR_SCENES])
def test_a_gear_vehicle_matches_the_2_0_6_capture_on_1_5_0(scene):
    fx = _fixture()
    got = run_scene(*scene, gear_b=True)
    want = fx["scenes"][scene[0]]
    for who in ("a", "tank"):
        _compare(scene[0] + "+gear_b", who, got[who], want[who])
