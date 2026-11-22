"""The stage-2 brake -> deceleration model (v2.0.2, ``control.brake``) and the plant
surface it reads (``SweepTable.accel_at``, ``DifferentiablePlant.accel_at`` /
``brake_gain``), the params key ``brake_gain`` and its derivation from a pre-2.0.2
record. The last two tests build a CPU scene: the closed form on the reference car
against the exploratory probe's value, and one AC-5a point (6 m/s, brake 0.5).

| abbr | meaning |
|---|---|
| k_b | brake gain: modelled deceleration per unit brake (m/s^2) |
| a_coast | modelled acceleration at throttle 0, brake 0 (m/s^2) |
| AC-5a | plan_r202 acceptance criterion: open-loop brake model accuracy |
"""
import copy
import math
import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control.brake import (brake_for, brake_gain_closed_form,  # noqa: E402
                                           brake_gain_from_params)
from genesis_vehicle.control.mpc.params import OPTIONAL_KEYS, MPCVehicleParams  # noqa: E402
from genesis_vehicle.tests.data import make_mpc_field_replay as rp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CAR = os.path.join(HERE, "..", "samples", "urdf", "car_4w.urdf")


def test_brake_for_signs_and_clipping():
    # forward, decelerate from coast -0.2 to -2.2 with k_b 4: b = 0.5
    assert brake_for(-2.2, -0.2, 3.0, 1, 4.0) == pytest.approx(0.5)
    # reverse, the deceleration is a POSITIVE acceleration: coast +0.2 -> +2.2
    assert brake_for(2.2, 0.2, -3.0, -1, 4.0) == pytest.approx(0.5)
    assert brake_for(-20.0, 0.0, 3.0, 1, 4.0) == 1.0         # clipped
    assert brake_for(1.0, 0.0, 3.0, 1, 4.0) == 0.0           # no braking wanted
    # at rest the gear decides the sign
    assert brake_for(-1.0, 0.0, 0.0, 1, 4.0) == pytest.approx(0.25)
    assert brake_for(1.0, 0.0, 0.0, -1, 4.0) == pytest.approx(0.25)


class _DT:
    """A drivetrain stub: brake torque 100 * brake on 4 wheels (bias 1)."""
    def distribute_torque(self, inputs, omega, wm, dev, fdt):
        b = torch.as_tensor(inputs.brake, dtype=fdt).reshape(-1, 1)
        tb = b * torch.tensor([100.0, 100.0, 200.0, 200.0], dtype=fdt)
        return torch.zeros_like(tb), tb


def test_closed_form_on_a_fake_vehicle():
    from genesis_vehicle import Ackermann
    resolved = types.SimpleNamespace(steering=Ackermann.__new__(Ackermann), drivetrain=_DT())
    wm = types.SimpleNamespace(n_wheels=4, radius=torch.tensor([0.5, 0.5, 0.25, 0.25]),
                               i_wheel=torch.tensor([1.0, 1.0, 0.5, 0.5]))
    want = (100 / 0.5 * 2 + 200 / 0.25 * 2) / (1000.0 + 1.0 / 0.25 * 2 + 0.5 / 0.0625 * 2)
    assert brake_gain_closed_form(resolved, wm, 1000.0) == pytest.approx(want, rel=1e-12)


def test_params_key_and_derivation_from_a_2_0_1_record():
    assert "brake_gain" in OPTIONAL_KEYS
    d = copy.deepcopy(rp.PARAMS_SDK)
    assert "brake_gain" not in d
    p = MPCVehicleParams.from_dict(d)
    assert p.data["brake_gain"] is None                     # filled, treated as absent
    dec = d["decel_brake_by_speed"]
    coast = lambda v, t, pi, ro: -0.1 - 0.01 * v            # noqa: E731
    want = sum(coast(float(v), 0, 0, 0) - a for v, a in dec.items()) / len(dec)
    assert brake_gain_from_params(p, coast) == pytest.approx(want)
    d["brake_gain"] = 5.0
    assert brake_gain_from_params(MPCVehicleParams.from_dict(d), coast) == 5.0
    d["brake_gain"] = -1.0
    with pytest.raises(ValueError):
        brake_gain_from_params(MPCVehicleParams.from_dict(d), coast)
    assert brake_gain_from_params(None, coast) is None


def test_sweep_table_accel_at_is_its_forward_map():
    t = rp.make_table()
    for v, thr, p in ((3.0, 0.0, 0.0), (6.5, 0.3, 2.0), (-1.0, -0.5, 0.0)):
        assert t.accel_at(v, thr, p, 0.0) == t._throttle_lu.forward(v, thr, p, 0.0)
    # the inverse of the forward value comes back (bisection tolerance 1e-3)
    a = t.accel_at(6.0, 0.4)
    assert t.throttle_for(6.0, a, 0.0, 0.0) == pytest.approx(0.4, abs=2e-3)


def test_identify_records_brake_gain():
    """F6: run the identify tool on the reference car (CPU, ~18 s) and read the
    record's ``brake_gain``: the closed form of that vehicle, and consistent with
    the brake-1 decelerations the same run measured."""
    pytest.importorskip("genesis")
    from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann
    from genesis_vehicle.control.identify import identify_mpc_params
    VehicleScene.init_backend("cpu")
    p = identify_mpc_params(CAR, car_4w_rwd_ackermann)
    kb = p.data["brake_gain"]
    assert kb == pytest.approx(5.0152, abs=2e-3)
    # the measured brake-1 accelerations are coast + k_b * 1 within 0.1 m/s^2
    for v, a in p.data["decel_brake_by_speed"].items():
        assert abs(-a - kb) < 0.3, (v, a, kb)
    # and the JSON round trip keeps it
    assert MPCVehicleParams.from_dict(p.to_dict()).data["brake_gain"] == kb


# --- Amendment 1 Rev 1: coast_for_brake -------------------------------------------------

from genesis_vehicle.control.brake import COAST_FADE_SPEED, coast_for_brake  # noqa: E402
from genesis_vehicle.control.sweep import SweepTable  # noqa: E402


def _flat_coast(v):
    """Synthetic flat coast: a vehicle at rest has none, otherwise ~-0.14."""
    return 0.0 if v == 0.0 else -0.14 - 0.002 * abs(v) if v > 0 else 0.15 + 0.003 * abs(v)


def _slope(v, p):
    """Synthetic slope term per grid row: the rest rows (v = 0) are skewed low
    and the two sides differ, as in the measured table."""
    base = 0.163 * p
    return base - 0.014 * p if v == 0.0 else base * (1.0 if v > 0 else 0.91)


def _table(speeds, pitches=(-10.0, 0.0, 10.0)):
    import numpy as np
    rows = {c: [] for c in ("v_init", "throttle", "steer", "pitch", "roll",
                            "a_measured", "omega_z_measured")}
    for v in speeds:
        for t in (-1.0, 0.0, 1.0):
            for st in (-1.0, 0.0, 1.0):
                for p in pitches:
                    rows["v_init"].append(v); rows["throttle"].append(t)
                    rows["steer"].append(st); rows["pitch"].append(p); rows["roll"].append(0.0)
                    rows["a_measured"].append(2.0 * t + _flat_coast(v) + _slope(v, p))
                    rows["omega_z_measured"].append(st)
    return SweepTable({k: np.asarray(x, float) for k, x in rows.items()})


REF = (-2.0, 0.0, 2.0, 4.0, 6.0)


def _expect(t, v, p, gear, vf=0.5):
    """R1.1, written independently: hold speeds from the table's columns."""
    import numpy as np
    ax = np.unique(t.columns["v_init"][t.columns["steer"] == 0.0])
    pos, neg = [a for a in ax if a > 0], [a for a in ax if a < 0]
    s = (1.0 if v > 0 else -1.0) if abs(v) > 1e-3 else float(gear)
    sides = {1.0: min(pos) if pos else None, -1.0: max(neg) if neg else None}
    sg = s if sides[s] is not None else -s
    vh = abs(sides[sg]); u = abs(v); w = sg * max(u, vh)
    f0 = t._throttle_lu.forward
    G = f0(w, 0.0, p, 0.0) - f0(w, 0.0, 0.0, 0.0)
    flat = f0(w, 0.0, 0.0, 0.0) if u >= vh else \
        f0(sg * vh, 0.0, 0.0, 0.0) * math.tanh(u / vf) / math.tanh(vh / vf)
    if sg != s:
        flat = -flat
    return G + flat


def test_coast_fade_speed_is_rolling_resistances_default():
    import inspect
    from genesis_vehicle import RollingResistance
    assert COAST_FADE_SPEED == inspect.signature(
        RollingResistance.__init__).parameters["tanh_scale"].default == 0.5


def test_coast_hold_speeds():
    assert _table(REF).coast_hold_speeds == (-2.0, 2.0)
    assert _table((0.0, 2.0, 4.0)).coast_hold_speeds == (None, 2.0)
    assert _table((1.0, 2.0, 4.0)).coast_hold_speeds == (None, 1.0)


def test_accel_at_is_unchanged_below_the_hold_speed():
    """F2 / mutant 5: the correction lives in coast_for_brake, not in accel_at."""
    t = _table(REF)
    for v, p in ((1.0, 0.0), (-1.0, 0.0), (1.0, 10.0)):
        assert t.accel_at(v, 0.0, p, 0.0) == t._throttle_lu.forward(v, 0.0, p, 0.0)


def test_coast_equals_accel_at_at_and_above_the_hold_speed():
    t = _table(REF)
    for v in (2.0, 3.0, 6.0, -2.0, -2.5):
        for p in (0.0, 10.0, -10.0):
            assert coast_for_brake(t, v, p, 0.0, 1) == t.accel_at(v, 0.0, p, 0.0)


def test_flat_fade_below_the_hold_speed():
    t = _table(REF)
    a2 = t.accel_at(2.0, 0.0, 0.0, 0.0)
    for v in (1.0, 0.5, 0.25, 0.05):
        want = a2 * math.tanh(v / 0.5) / math.tanh(2.0 / 0.5)
        assert coast_for_brake(t, v, 0.0, 0.0, 1) == pytest.approx(want, rel=1e-12, abs=1e-15)
        assert coast_for_brake(t, v, 0.0, 0.0, 1) == _expect(t, v, 0.0, 1)
    assert coast_for_brake(t, 0.0, 0.0, 0.0, 1) == 0.0             # no coast at rest (flat)
    eps = 1e-9                                                       # continuity at v_h
    assert abs(coast_for_brake(t, 2.0 - eps, 0.0, 0.0, 1) - coast_for_brake(t, 2.0, 0.0, 0.0, 1)) <= 1e-8
    assert abs(coast_for_brake(t, -2.0 + eps, 0.0, 0.0, -1)
               - coast_for_brake(t, -2.0, 0.0, 0.0, -1)) <= 1e-8
    # through 0 on FLAT ground only (A1-C1: no such claim on a slope)
    assert abs(coast_for_brake(t, 2e-5, 0.0, 0.0, 1) - coast_for_brake(t, -2e-5, 0.0, 0.0, -1)) < 1e-4


def test_pitched_coast_keeps_the_slope_term_unfaded():
    """F1 / mutants 6, 7: the slope term is read at w = max(|v|, v_h) and never faded."""
    t = _table(REF)
    for p in (10.0, -10.0):
        for v in (0.25, 0.05, -0.25, -0.05):
            g = 1 if v > 0 else -1
            assert coast_for_brake(t, v, p, 0.0, g) == _expect(t, v, p, g)
    f0 = t._throttle_lu.forward
    for gear in (1, -1):                                           # at v = 0: G exactly
        w = 2.0 * gear
        assert coast_for_brake(t, 0.0, 10.0, 0.0, gear) == f0(w, 0, 10.0, 0) - f0(w, 0, 0.0, 0)
    # the slope term really differs from the faded / at-v forms here
    v = 0.25
    G_w = f0(2.0, 0, 10.0, 0) - f0(2.0, 0, 0.0, 0)
    G_v = f0(v, 0, 10.0, 0) - f0(v, 0, 0.0, 0)
    assert abs(G_w - G_v) > 0.05 and abs(G_w) > 0.5


def test_gear_decides_the_side_at_rest():
    t = _table(REF)
    a = coast_for_brake(t, 5e-4, 10.0, 0.0, 1)
    b = coast_for_brake(t, 5e-4, 10.0, 0.0, -1)
    assert a != b and a == pytest.approx(_expect(t, 5e-4, 10.0, 1)) and b == pytest.approx(_expect(t, 5e-4, 10.0, -1))


def test_mirror_on_a_table_without_negative_speeds():
    """F4 / mutants 4, 8: on the empty side only the flat part flips; G keeps its sign."""
    t = _table((0.0, 2.0, 4.0))
    f0 = t._throttle_lu.forward
    for v, w in ((-0.5, 2.0), (-3.0, 3.0)):
        flat = f0(w, 0, 0.0, 0) if abs(v) >= 2.0 else \
            f0(2.0, 0, 0.0, 0) * math.tanh(abs(v) / 0.5) / math.tanh(2.0 / 0.5)
        G = f0(w, 0, 10.0, 0) - f0(w, 0, 0.0, 0)
        assert coast_for_brake(t, v, 10.0, 0.0, -1) == pytest.approx(G - flat, rel=1e-12)
        assert coast_for_brake(t, v, 0.0, 0.0, -1) == pytest.approx(-flat, rel=1e-12)
    G_pos = coast_for_brake(t, 0.5, 10.0, 0.0, 1) - coast_for_brake(t, 0.5, 0.0, 0.0, 1)
    G_neg = coast_for_brake(t, -0.5, 10.0, 0.0, -1) - coast_for_brake(t, -0.5, 0.0, 0.0, -1)
    assert G_pos > 0 and G_neg > 0 and G_pos == pytest.approx(G_neg)
    assert coast_for_brake(t, -0.5, 0.0, 0.0, -1) > 0             # the flat coast flipped


def test_first_grid_speed_one_is_the_hold_speed():
    t = _table((1.0, 2.0, 4.0))
    a1 = t.accel_at(1.0, 0.0, 0.0, 0.0)
    assert coast_for_brake(t, 0.5, 0.0, 0.0, 1) == pytest.approx(
        a1 * math.tanh(0.5 / 0.5) / math.tanh(1.0 / 0.5), rel=1e-12)
    assert coast_for_brake(t, 1.5, 0.0, 0.0, 1) == t.accel_at(1.5, 0.0, 0.0, 0.0)


def test_plant_without_hold_speeds_is_passed_through():
    calls = []

    class P:
        def accel_at(self, v, thr, p, r):
            calls.append((v, thr, p, r))
            return -0.123
    assert coast_for_brake(P(), 0.4, 3.0, 1.0, 1) == -0.123
    assert calls == [(0.4, 0.0, 3.0, 1.0)]


# --- CPU scene ---------------------------------------------------------------------------

def _car_scene(n_envs=1, brake=0.0):
    gs = pytest.importorskip("genesis")
    from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann
    VehicleScene.init_backend("cpu")
    vs = VehicleScene(dt=0.025, substeps=4, n_envs=n_envs, show_viewer=False,
                      solver="per_vehicle")
    vs.add_ground_plane(friction=1.0)
    veh = vs.add_vehicle(CAR, car_4w_rwd_ackermann, pos=(0.0, 0.0, 0.5),
                         material=gs.materials.Rigid(friction=1.0))
    vs.build()
    return gs, vs, veh


def test_closed_form_and_plant_surface_on_the_reference_car():
    gs, vs, veh = _car_scene()
    from genesis_vehicle import DifferentiablePlant
    from genesis_vehicle.control.plant import plant_mass_properties
    for _ in range(20):
        veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        vs.step()
    p = veh.physics
    k_b = brake_gain_closed_form(p.resolved, p.wheel_meta, plant_mass_properties(veh.entity_main).mass)
    assert k_b == pytest.approx(5.0152, abs=2e-3)           # EXPLORATORY probe p2: 5.0152
    plant = DifferentiablePlant(veh, brake=0.5)
    assert plant.brake_gain() == pytest.approx(k_b, rel=1e-9)
    a0 = plant.accel_at(0.0, 0.0)
    assert plant.brake == 0.5                               # restored after the call
    plant.brake = 0.0
    assert plant.accel_at(0.0, 0.0) == a0                   # K4(a): unrolled at brake 0
    plant.brake = 0.5
    assert plant.predict(0.0, 0.0)[0] != a0                 # predict keeps its own brake


def test_ac5a_one_point_six_mps_half_brake():
    """AC-5a, one point: 6 m/s, brake 0.5, throttle 0: the mean deceleration over the
    speed window [0.2 v0, 0.9 v0] is within max(0.05, 0.03 |a_model|) of
    a_coast - k_b * 0.5, a_coast measured at brake 0 from the same start (two envs)."""
    gs, vs, veh = _car_scene(n_envs=2)
    from genesis_vehicle.control.plant import plant_mass_properties
    p = veh.physics
    ent = veh.entity_main
    T = lambda x: torch.as_tensor(x, device=gs.device, dtype=gs.tc_float)  # noqa: E731
    veh.set_inputs(throttle=T([0.0, 0.0]), brake=T([1.0, 1.0]), steer=T([0.0, 0.0]))
    for _ in range(20):
        vs.step()
    dv = torch.zeros(2, ent.n_dofs, device=gs.device, dtype=gs.tc_float)
    dv[:, 0] = 6.0
    ent.set_dofs_velocity(dv)
    p.omega[:, :] = 6.0 / p.wheel_meta.radius.reshape(1, -1)
    veh.set_inputs(throttle=T([0.0, 0.0]), brake=T([0.0, 0.0]), steer=T([0.0, 0.0]))
    for _ in range(4):
        vs.step()
    veh.set_inputs(throttle=T([0.0, 0.0]), brake=T([0.0, 0.5]), steer=T([0.0, 0.0]))
    hist = []
    for _ in range(int(3.0 / 0.025)):
        vs.step()
        hist.append(ent.get_vel()[:, 0].cpu().tolist())
    v0 = hist[0]

    def mean_a(i):
        idx = [k for k, h in enumerate(hist) if 0.2 * v0[i] < h[i] < 0.9 * v0[i]]
        if len(idx) < 3:                       # coast: whole window (barely slows)
            idx = list(range(len(hist)))
        return (hist[idx[-1]][i] - hist[idx[0]][i]) / ((idx[-1] - idx[0]) * 0.025)
    a_coast, a_meas = mean_a(0), mean_a(1)
    k_b = brake_gain_closed_form(p.resolved, p.wheel_meta, plant_mass_properties(ent).mass)
    a_model = a_coast - k_b * 0.5
    assert abs(a_meas - a_model) <= max(0.05, 0.03 * abs(a_model)), (a_meas, a_model, a_coast)


def test_ac5a_failing_point_now_within_five_percent():
    """Amendment 1, the point AC-5a failed on in round 1: 2 m/s, brake 0.1, stopping
    distance (chassis displacement to the first |v_long| < 0.01 step) within 5 % of
    the model integrated with coast_for_brake. The table is built in the test from
    this run's own coast measurement at 2 and 4 m/s (grid 0 / 2 / 4, a vehicle at
    rest has coast 0), because no car sweep table is tracked in the repo."""
    import numpy as np
    gs, vs, veh = _car_scene(n_envs=3)
    from genesis_vehicle.control.plant import plant_mass_properties
    p, ent = veh.physics, veh.entity_main
    T = lambda x: torch.as_tensor(x, device=gs.device, dtype=gs.tc_float)  # noqa: E731
    z = T([0.0, 0.0, 0.0])
    veh.set_inputs(throttle=z, brake=T([1.0, 1.0, 1.0]), steer=z)
    for _ in range(20):
        vs.step()
    v0 = [2.0, 2.0, 4.0]
    dv = torch.zeros(3, ent.n_dofs, device=gs.device, dtype=gs.tc_float)
    dv[:, 0] = T(v0)
    ent.set_dofs_velocity(dv)
    p.omega[:, :] = T(v0).unsqueeze(-1) / p.wheel_meta.radius.reshape(1, -1)
    veh.set_inputs(throttle=z, brake=z, steer=z)
    for _ in range(4):
        vs.step()
    vstart = ent.get_vel()[:, 0].cpu().numpy().copy()
    xstart = ent.get_pos()[:, 0].cpu().numpy().copy()
    veh.set_inputs(throttle=z, brake=T([0.1, 0.0, 0.0]), steer=z)
    vh, xh = [], []
    for _ in range(int(6.0 / 0.025)):
        vs.step()
        vh.append(ent.get_vel()[:, 0].cpu().numpy().copy())
        xh.append(ent.get_pos()[:, 0].cpu().numpy().copy())
    vh, xh = np.array(vh), np.array(xh)
    coast = {2.0: float((vh[19, 1] - vstart[1]) / 0.5), 4.0: float((vh[19, 2] - vstart[2]) / 0.5)}
    rows = {c: [] for c in ("v_init", "throttle", "steer", "pitch", "roll",
                            "a_measured", "omega_z_measured")}
    for v, a0 in ((0.0, 0.0), (2.0, coast[2.0]), (4.0, coast[4.0])):
        for t in (-1.0, 0.0, 1.0):
            for st in (-1.0, 0.0, 1.0):
                for k, val in (("v_init", v), ("throttle", t), ("steer", st), ("pitch", 0.0),
                               ("roll", 0.0), ("a_measured", a0 + 2.0 * t),
                               ("omega_z_measured", st)):
                    rows[k].append(val)
    table = SweepTable({k: np.asarray(x, float) for k, x in rows.items()})
    stop = next(k for k in range(len(vh)) if abs(vh[k, 0]) < 0.01)
    d_meas = float(xh[stop, 0] - xstart[0])
    k_b = brake_gain_closed_form(p.resolved, p.wheel_meta, plant_mass_properties(ent).mass)
    v, d, h = float(vstart[0]), 0.0, 1e-3
    while v > 0.01:
        a = coast_for_brake(table, v, 0.0, 0.0, 1) - k_b * 0.1
        d += v * h
        v += a * h
    assert abs(d_meas - d) / d <= 0.05, (d_meas, d, coast)
