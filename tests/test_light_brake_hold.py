"""2.0.4: the pedal-free StaticFrictionLock on real CPU scenes (plan r204 §5.1).

A braked vehicle whose brake can hold the load rests at any pedal; one whose
brake cannot rolls at any pedal; the outcome is continuous across the old
``brake_thr`` (0.3); no brake-dominated wheel turns against its contact; a
wheel the drive overpowers turns (D11). Classes come from the configuration
(plan r204 ``acceptance.json``), not from these results.

| abbr | meaning |
|---|---|
| SFL | `StaticFrictionLock` |
| REST | mean planar speed over the last 2 s < 1e-3 m/s, net planar displacement over the second half <= 1 mm, max wheel rim speed over the last 2 s <= 1e-3 m/s |
| ROLL | planar speed at the end > 0.05 m/s and mean wheel rim speed within 10 % of it |
| CR | counter-rotation: omega*v_long < 0 with both above 1e-3 m/s, on a wheel whose drive torque does not exceed its brake torque |
| rho | brake capacity / slope load: brake * sum(T_brake_max) / (r * M * g * sin(theta)) |
| D11 | plan r204 decision: per wheel, drive above brake turns the wheel |

Conditions: reference car ``car_4w.urdf`` + ``car_4w_rwd_ackermann`` /
reference tank ``tank_ref.urdf`` + ``tank_skid_belt``, "control" profile,
per_vehicle, dual_scene, plane mu 1.0, dt 0.025 x 4, gravity-tilt slopes, CPU.
"""
from __future__ import annotations

import math
import os
import re

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CAR = os.path.abspath(os.path.join(HERE, "..", "samples", "urdf", "car_4w.urdf"))
TANK = os.path.join(HERE, "data", "tank_ref.urdf")
DT = 0.025


def _scene(urdf, preset, rows, secs, z0=0.5, cfg=None, inputs_fn=None):
    """rows: dicts with dir ('dn'|'up'|'cross'), theta, v0, brake [, throttle, steer]. Returns per-row traces."""
    import genesis as gs
    from genesis_vehicle import VehicleScene
    from genesis_vehicle.strategies import stability as stab
    N = len(rows)
    VehicleScene.init_backend("cpu")
    vs = VehicleScene(dt=DT, substeps=4, n_envs=N, raycast_mode="dual_scene", show_viewer=False, solver="per_vehicle")
    vs.add_ground_plane(friction=1.0)
    kw = {"cfg": cfg} if cfg is not None else {"stability": "control"}
    veh = vs.add_vehicle(urdf, preset, pos=(0.0, 0.0, z0), material=gs.materials.Rigid(friction=1.0), **kw)
    vs.build()
    ent, p = veh.entity_main, veh.physics
    hook = [h for h in p.post_tire_hooks if isinstance(h, stab.StaticFrictionLock)][0]
    rec = {}
    orig = type(hook).apply_post_tire

    def wrap(ctx):
        orig(hook, ctx)
        rec["vl"] = ctx.v_long.detach().clone()
    hook.apply_post_tire = wrap
    for i, r in enumerate(rows):
        t = math.radians(r.get("theta", 0.0)); g = 9.81
        gv = {"dn": (g * math.sin(t), 0.0, -g * math.cos(t)), "up": (-g * math.sin(t), 0.0, -g * math.cos(t)),
              "cross": (0.0, g * math.sin(t), -g * math.cos(t))}[r.get("dir", "dn")]
        vs.set_gravity(gv, envs_idx=[i])
    T = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=gs.device)  # noqa: E731
    z = np.zeros(N)
    veh.set_inputs(throttle=T(z), brake=T(np.ones(N)), steer=T(z))
    for _ in range(40):
        vs.step()
    v0 = np.array([r.get("v0", 0.0) for r in rows])
    if np.any(v0):
        dv = torch.zeros(N, ent.n_dofs, device=gs.device, dtype=gs.tc_float); dv[:, 0] = T(v0)
        ent.set_dofs_velocity(dv)
        p.omega[:, :] = T(v0).unsqueeze(-1) / p.wheel_meta.radius.reshape(1, -1)
    thr = T([r.get("throttle", 0.0) for r in rows]); brk = T([r["brake"] for r in rows])
    st = T([r.get("steer", 0.0) for r in rows])
    veh.set_inputs(throttle=thr, brake=brk, steer=st)
    rr = p.wheel_meta.radius.reshape(1, -1).cpu().numpy()
    out = dict(vp=[], pos=[], wr=[], yaw=[], cr=np.zeros(N, bool), engaged=[], vx=[])
    for k in range(int(round(secs / DT))):
        if inputs_fn is not None:
            inputs_fn(k, veh, T)
        vs.step()
        v = ent.get_vel().reshape(N, 3).cpu().numpy()
        out["vp"].append(np.hypot(v[:, 0], v[:, 1])); out["vx"].append(v[:, 0].copy())
        out["pos"].append(ent.get_pos().reshape(N, 3).cpu().numpy()[:, :2].copy())
        q = ent.get_quat().reshape(N, 4).cpu().numpy()
        out["yaw"].append(np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)))
        om = p.omega.detach().cpu().numpy(); out["wr"].append(np.abs(om) * rr)
        vl = rec["vl"].cpu().numpy()
        Td = np.abs(p.last_T_drive.detach().cpu().numpy()); Tb = p.last_T_brake.detach().cpu().numpy()
        out["cr"] |= ((om * rr * vl < 0) & (np.abs(om * rr) > 1e-3) & (np.abs(vl) > 1e-3) & (Td <= Tb)).any(1)
        out["engaged"].append(hook._was_active.detach().cpu().numpy().copy())
    for k in ("vp", "pos", "wr", "yaw", "engaged", "vx"):
        out[k] = np.array(out[k])
    out["yaw"] = np.unwrap(out["yaw"], axis=0)
    return out


def _rest(o, i, t0=None):
    L2 = int(2 / DT)
    if t0 is None:
        half = len(o["vp"]) // 2
    else:
        half = int(t0 / DT)
    return (o["vp"][-L2:, i].mean() < 1e-3 and np.hypot(*(o["pos"][-1, i] - o["pos"][half, i])) <= 1e-3
            and o["wr"][-L2:, i, :].max() <= 1e-3)


def _roll(o, i):
    v = o["vp"][-1, i]
    return v > 0.05 and abs(o["wr"][-1, i, :].mean() - v) <= 0.1 * v


ROWS = [
    # must REST (rho >= 1.10, or cross-slope)
    dict(dir="dn", theta=5, brake=0.3), dict(dir="dn", theta=5, brake=0.2), dict(dir="dn", theta=3, brake=0.3),
    dict(dir="cross", theta=5, brake=0.1), dict(dir="dn", theta=5, brake=0.3, v0=2.0),
    # must ROLL (rho_eff <= 0.90) at any pedal, from rest
    dict(dir="dn", theta=15, brake=0.31), dict(dir="dn", theta=5, brake=0.1), dict(dir="dn", theta=10, brake=0.2),
    # pedal continuity, 10 deg from rest
    dict(dir="dn", theta=10, brake=0.29), dict(dir="dn", theta=10, brake=0.30), dict(dir="dn", theta=10, brake=0.31),
    dict(dir="dn", theta=10, brake=0.35),
    # light brake flat stop from 1 m/s
    dict(dir="dn", theta=0, brake=0.05, v0=1.0),
]


@pytest.fixture(scope="module")
def car_matrix():
    from genesis_vehicle import car_4w_rwd_ackermann
    return _scene(CAR, car_4w_rwd_ackermann, ROWS, 20.0)


def test_light_brake_holds_when_capacity_suffices(car_matrix):
    """2.0.3 crept here at the held-wheel deadbeat speed g*tan(theta)*dt/s
    (5 deg: 0.0250 m/s) because its lock engaged only above brake 0.3."""
    o = car_matrix
    for i in range(5):
        assert _rest(o, i), ROWS[i]
        a, b = int(10 / DT), len(o["vp"]) - 1
        assert np.hypot(*(o["pos"][b, i] - o["pos"][a, i])) <= 1e-3, ROWS[i]
        assert o["vp"][a:, i].max() <= 1e-4, ROWS[i]


def test_too_weak_brake_rolls_at_any_pedal(car_matrix):
    """10 deg / 0.31 (rho 0.96) is EITHER by the rule; 15 deg / 0.31, 5 deg / 0.1
    and 10 deg / 0.2 must roll — on 2.0.3 a pedal above 0.3 slid the car on
    locked wheels (wheel rim speed 0) instead."""
    o = car_matrix
    for i in (5, 6, 7):
        assert _roll(o, i), ROWS[i]


def test_pedal_continuity_10deg(car_matrix):
    """Speed at 20 s is non-increasing along the pedal (tolerance 1e-3 m/s),
    with no jump to a locked slide at 0.31 (2.0.3: 0.606 -> 0.040 m/s, wheels
    locked)."""
    o = car_matrix
    v = [o["vp"][-1, i] for i in (8, 9, 10, 11)]
    assert all(v[k + 1] <= v[k] + 1e-3 for k in range(3)), v
    assert o["wr"][-1, 10, :].mean() > 0.9 * v[2]           # 0.31 rolls on turning wheels


def test_light_brake_flat_stop_does_not_reverse(car_matrix):
    o = car_matrix
    i = 12
    assert _rest(o, i)
    assert o["vx"][:, i].min() >= -1e-4


def test_no_counter_rotation_on_any_row(car_matrix):
    assert not car_matrix["cr"].any(), [ROWS[i] for i in np.nonzero(car_matrix["cr"])[0]]


def test_held_is_per_wheel():
    """A wheel with zero brake torque (brake_bias 0 on the front axle) is never
    held, so the lock never engages it; the rear wheels hold the car on 5 deg
    at brake 0.6 (rear capacity alone: 1500 N*m, rho 3.7)."""
    from genesis_vehicle import car_4w_rwd_ackermann
    cfg = car_4w_rwd_ackermann(CAR, stability="control")
    cfg.drivetrain.brake_bias = [0.0, 0.0, 0.5, 0.5]       # per wheel: FL, FR, RL, RR
    o = _scene(CAR, car_4w_rwd_ackermann, [dict(dir="dn", theta=5, brake=0.6)], 10.0, cfg=cfg)
    eng = o["engaged"][:, 0, :]
    assert not eng[:, :2].any(), "an unbraked front wheel engaged the lock"
    assert eng[-1, 2:].all()
    assert not o["cr"].any()


def test_light_vehicle_rests(tmp_path):
    """A 0.2x-mass reference car (266 kg, every link mass and inertia x 0.2,
    written here from car_4w.urdf) built like an OSC-server generic vehicle
    (``server.vehicle_builder.build_cfg``: mass-derived suspension, maxTorque
    200 / maxBrake 500 N*m, rear drive) with the "control" hook set: the lock
    spring is capped per wheel (k_eff 107 200 / 105 600 N/m at dt 0.025) and
    the car rests on the flat and on 3 deg at brake 0.3. With the uncapped
    spring the plan's probe measured 1.75e-3 m/s on the flat (review r204-3)."""
    from genesis_vehicle.presets import stability_hooks_for_profile
    from genesis_vehicle.server.vehicle_builder import build_cfg
    src = open(CAR).read()

    def scale(m):
        return '%s"%.9g"' % (m.group(1), float(m.group(2)) * 0.2)
    light = re.sub(r'(<mass value=)"([-0-9.eE+]+)"', scale, src)
    light = re.sub(r'(\b(?:ixx|ixy|ixz|iyy|iyz|izz)=)"([-0-9.eE+]+)"', scale, light)
    path = tmp_path / "car_light.urdf"
    path.write_text(light)                  # car_4w.urdf has primitive geometry only (no mesh files)
    mapping = {"driveType": 3, "drivetrainStrategy": 1, "maxTorque": 200.0, "maxBrake": 500.0,
               "drivingJoints": [{"jointName": "rear_left_wheel_joint"}, {"jointName": "rear_right_wheel_joint"}],
               "steeringJoints": [{"jointName": "front_left_steer_joint"}, {"jointName": "front_right_steer_joint"}]}
    cfg = build_cfg(str(path), mapping, 1.0, target_id=0)
    cfg.stability_hooks = stability_hooks_for_profile("control", "car")
    o = _scene(str(path), None, [dict(dir="dn", theta=0, brake=0.3), dict(dir="dn", theta=3, brake=0.3)], 15.0, cfg=cfg)
    assert _rest(o, 0) and _rest(o, 1)
    assert not o["cr"].any()


def test_d11_tank_brake_vs_steer_at_rest():
    """D11 (documented physics, no brake priority): tank_skid_belt puts the
    whole side torque 30000*|0.3*throttle +- steer| N*m on every wheel of a
    side and brakes each with 3000*brake N*m. Brake 1.0 + steer 0.3 (9000 >
    3000 N*m): the wheels turn and the tank yaws more than 2 deg in 5 s (2.0.3
    held it by forcing the wheel spin to 0). Brake 1.0 + throttle 0.3 (2700 <
    3000 N*m): it rests."""
    from genesis_vehicle import tank_skid_belt
    rows = [dict(dir="dn", theta=0, brake=1.0, steer=0.3), dict(dir="dn", theta=0, brake=1.0, throttle=0.3)]
    o = _scene(TANK, tank_skid_belt, rows, 6.0, z0=1.5)
    k5 = int(5 / DT) - 1
    assert abs(math.degrees(o["yaw"][k5, 0] - o["yaw"][0, 0])) > 2.0
    assert _rest(o, 1)
    assert not o["cr"][1]


def test_release_matches_a_full_brake_hold():
    """AC-5(i): after a 6 s hold at a light brake, the throttle-0.5 release
    travels like the release after a brake-1.0 hold (within 1 mm at 2 s), on
    5 deg downhill and uphill."""
    from genesis_vehicle import car_4w_rwd_ackermann
    rows = [dict(dir="dn", theta=5, brake=0.3), dict(dir="dn", theta=5, brake=1.0),
            dict(dir="up", theta=5, brake=0.3), dict(dir="up", theta=5, brake=1.0)]
    n_hold = int(6 / DT)

    def inputs(k, veh, T):
        if k == n_hold:
            veh.set_inputs(throttle=T([0.5] * 4), brake=T([0.0] * 4), steer=T([0.0] * 4))
    o = _scene(CAR, car_4w_rwd_ackermann, rows, 8.0, inputs_fn=inputs)
    x = o["pos"][:, :, 0]
    rel = x[-1] - x[n_hold - 1]
    assert abs(rel[0] - rel[1]) <= 1e-3 and abs(rel[2] - rel[3]) <= 1e-3, rel
    assert not o["cr"].any()


@pytest.mark.parametrize("solver", ["per_vehicle", "batched"])
def test_partial_reset_leaves_other_envs_untouched(solver):
    """AC-9: n_envs 3 resting on 5 deg at brake 0.3; at 6 s env 1 is reset
    (VehiclePhysics.reset(env_ids=[1]) / MultiVehicleKindPhysics.reset(rows=[1]))
    and teleported back to its start pose. Envs 0 and 2 are byte-identical to a
    run without the reset; env 1 rests again within 5 s."""
    import genesis as gs
    from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann

    def run(do_reset):
        VehicleScene.init_backend("cpu")
        vs = VehicleScene(dt=DT, substeps=4, n_envs=3, raycast_mode="dual_scene", show_viewer=False, solver=solver)
        vs.add_ground_plane(friction=1.0)
        veh = vs.add_vehicle(CAR, car_4w_rwd_ackermann, pos=(0, 0, 0.5), stability="control",
                             material=gs.materials.Rigid(friction=1.0))
        vs.build()
        t = math.radians(5.0); vs.set_gravity((9.81 * math.sin(t), 0.0, -9.81 * math.cos(t)))
        veh.set_inputs(throttle=0.0, brake=0.3, steer=0.0)
        ent = veh.entity_main
        p0, q0 = ent.get_pos().clone(), ent.get_quat().clone()
        tr = []
        for k in range(int(12 / DT)):
            if do_reset and k == int(6 / DT):
                if veh.physics is not None:
                    veh.physics.reset(env_ids=torch.tensor([1]))
                else:
                    vs._mvp.kinds[0].reset(rows=[1])
                ent.set_pos(p0[1:2], envs_idx=[1]); ent.set_quat(q0[1:2], envs_idx=[1])
                ent.set_dofs_velocity(torch.zeros(1, ent.n_dofs, device=gs.device, dtype=gs.tc_float), envs_idx=[1])
            vs.step()
            tr.append(np.concatenate([ent.get_pos().cpu().numpy().reshape(3, -1),
                                      ent.get_vel().cpu().numpy().reshape(3, -1)], axis=1))
        return np.array(tr)
    a, b = run(False), run(True)
    assert np.array_equal(a[:, [0, 2]], b[:, [0, 2]])
    k0 = int(6 / DT)
    v = np.hypot(b[:, 1, 3], b[:, 1, 4])
    assert v[k0 + int(3 / DT):k0 + int(5 / DT)].mean() < 1e-3
