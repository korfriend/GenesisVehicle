"""v1.8.0 low-speed behaviour in real CPU scenes: the Coulomb brake, the
held-wheel / lateral deadbeats with their rigid-body scale, and the
``StaticFrictionLock`` rest anchor.

| abbr | meaning |
|---|---|
| D / Y / Z | held-wheel deadbeat / lateral deadbeat / rigid-body scale ``s`` of both |
| SFL | ``StaticFrictionLock`` |
| A / A6 / C / L | SFL: anchor at the first reversal / take over only at rest / damping at least critical / force limited by the clamp bounds |
| back / vmin | back-travel after the first \\|v\\| < 0.05 m/s / minimum speed along the travel direction after it |
| alt@tol | per wheel, consecutive ω steps of opposite sign with both changes > tol (rad/s); max over wheels |
| flip_vlat / flip_flat / alt_wz | lateral 2-cycle metrics (as defined in this module's helpers): v_lat sign flips (both > 1e-3 m/s) / per-wheel F_lat sign flips (both > 10 N) / yaw-rate change sign flips (both > 1e-3) |
| N-numbers | test ids of the 1.8.0 plan |

Scene, every test unless stated (the scene of the 1.8.0 design measurements,
CHANGELOG 1.8.0): CPU, dual_scene, ``solver="per_vehicle"``, dt 0.025 x 4 substeps,
ground plane μ 1.0, the preset's default stability profile; reference car =
``samples/urdf/car_4w.urdf`` + ``car_4w_rwd_ackermann`` spawned at z 0.5,
reference tank = ``samples/urdf/tank_ray.urdf`` + ``tank_skid_belt`` at z 1.0.
Each run: genesis scene reset, ``VehicleScene.reset()``, the SFL hook's state
cleared (it survives resets — a documented known gap; clearing it makes a
shared scene behave like a fresh process), 1 s settle, then the launch.
Slopes are emulated by tilting gravity (``VehicleScene.set_gravity``), set
back to (0, 0, -9.81) before every run.

The thresholds are the plan's acceptance criteria (AC-*), fixed before the
implementation was measured; the one documented exception is the user's D-PS
decision (car throttle 1.0 + brake 1.0, see ``test_n17``).
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

genesis = pytest.importorskip("genesis")

from genesis_vehicle import VehicleScene, presets  # noqa: E402
from genesis_vehicle.control.plant import plant_mass_properties  # noqa: E402

DT, SUB = 0.025, 4
_SD = os.path.join(os.path.dirname(__file__), "..", "samples", "urdf")
CAR_URDF = os.path.abspath(os.path.join(_SD, "car_4w.urdf"))
TANK_URDF = os.path.abspath(os.path.join(_SD, "tank_ray.urdf"))
LOW_IZZ_URDF = os.path.abspath(os.path.join(os.path.dirname(__file__), "data",
                                            "low_yaw_inertia_car.urdf"))


class _Rig:
    """One vehicle in one scene, reused across runs."""

    def __init__(self, kind, urdf=None, solver="per_vehicle"):
        self.kind = kind
        self.solver = solver
        urdf = urdf or (TANK_URDF if kind == "tank" else CAR_URDF)
        preset = presets.tank_skid_belt if kind == "tank" else presets.car_4w_rwd_ackermann
        self.z0 = 1.0 if kind == "tank" else 0.5
        VehicleScene.init_backend("cpu")
        self.vs = VehicleScene(dt=DT, substeps=SUB, n_envs=1, raycast_mode="dual_scene",
                               show_viewer=False, solver=solver)
        self.vs.add_ground_plane(friction=1.0)
        self.veh = self.vs.add_vehicle(urdf, preset, cfg=preset(urdf),
                                       pos=(0.0, 0.0, self.z0),
                                       material=genesis.materials.Rigid(friction=1.0))
        self.vs.build()
        self.ent = self.veh.entity_main
        self._cache = {}

    @property
    def phys(self):
        p = self.veh.physics
        return p if p is not None else self.vs._mvp.kinds[0]._proto

    def sfl(self):
        hs = [h for h in self.phys.post_tire_hooks
              if type(h).__name__ == "StaticFrictionLock"]
        return hs[0] if hs else None

    def _state(self):
        v = self.ent.get_vel().reshape(3).cpu().numpy()
        q = self.ent.get_quat().reshape(4).cpu().numpy()
        p = self.ent.get_pos().reshape(3).cpu().numpy()
        w = self.ent.get_ang().reshape(3).cpu().numpy()
        h = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        vf = v[0] * math.cos(h) + v[1] * math.sin(h)
        vl = -v[0] * math.sin(h) + v[1] * math.cos(h)
        return vf, vl, p, w[2]

    def run(self, *, V0=0.0, THR=0.0, BRK=0.0, STEER=0.0, SECS=10.0, DEG=0.0,
            DEGY=0.0, VLAT=0.0, WZ0=0.0, kick=False, settle_brake=None,
            zoff=False, every=None):
        key = (V0, THR, BRK, STEER, SECS, DEG, DEGY, VLAT, WZ0, kick, settle_brake, zoff)
        if every is None and key in self._cache:
            return self._cache[key]
        vs, ent, phys = self.vs, self.ent, self.phys
        vs.set_gravity((0.0, 0.0, -9.81))
        vs.scene.reset()
        vs.reset()
        sfl = self.sfl()
        if sfl is not None:
            # (the fallback names are the pre-1.8.0 state, for the non-vacuity
            # run of this file against the 1.7.0 archive)
            for a in getattr(sfl, "STATE_ATTRS", ("_d_long", "_d_lat", "_was_active")):
                setattr(sfl, a, None)
        sb = (1.0 if DEG else 0.0) if settle_brake is None else settle_brake
        self.veh.set_inputs(0.0, sb, 0.0)
        for _ in range(int(round(1.0 / DT))):
            vs.step()
        wm = phys.wheel_meta
        if zoff:
            # The N26 control: Z disabled (s = 1 exactly) on this driver only.
            from genesis_vehicle.dynamics import YawInertia
            wm.yaw_inertia = YawInertia(float("inf"), wm.yaw_inertia.rho_sq)
        if DEG or DEGY:
            th, ty = math.radians(DEG), math.radians(DEGY)
            vs.set_gravity((9.81 * math.sin(th), 9.81 * math.sin(ty),
                            -9.81 * math.cos(th) * math.cos(ty)))
        if kick or V0 != 0.0:
            dv = torch.zeros(1, ent.n_dofs, device=genesis.device, dtype=genesis.tc_float)
            dv[0, 0] = V0
            dv[0, 1] = VLAT
            dv[0, 5] = WZ0
            ent.set_dofs_velocity(dv)
            phys.omega[:, :] = V0 / wm.radius
        self.veh.set_inputs(THR, BRK, STEER)
        V, VL, X, WZ, W, FL = [], [], [], [], [], []
        p0 = self._state()[2]
        for _ in range(int(round(SECS / DT))):
            vs.step()
            vf, vl, p, wz = self._state()
            V.append(vf); VL.append(vl); X.append(p.copy()); WZ.append(wz)
            W.append(phys.omega.reshape(-1).cpu().numpy().copy())
            FL.append(phys.last_F_lat.reshape(-1).cpu().numpy().copy())
            if every is not None:
                every(self)
        if zoff:
            wm.yaw_inertia = None                 # recomputed from the carried mass props
            phys._ensure_yaw_inertia()
        vs.set_gravity((0.0, 0.0, -9.81))
        out = dict(V=np.array(V), VL=np.array(VL), X=np.array(X), WZ=np.array(WZ),
                   W=np.array(W), FL=np.array(FL), p0=p0)
        if every is None:
            self._cache[key] = out
        return out


@pytest.fixture(scope="module")
def car():
    return _Rig("car")


@pytest.fixture(scope="module")
def tank():
    return _Rig("tank")


def _pd13(r, sign):
    """back-travel (m) and vmin (m/s) after the first |v| < 0.05 m/s; travel
    sign `sign` (the PD13 metric of CHANGELOG 1.8.0 §6)."""
    V, X, p0 = r["V"], r["X"], r["p0"]
    xa = X[:, 0] - p0[0]
    st = np.where(np.abs(V) < 0.05)[0]
    assert st.size, "never stopped"
    k = int(st[0])
    back = float(max(0.0, np.max(sign * (xa[k] - xa[k:]))))
    vmin = float(np.min(sign * V[k:]))
    return back, vmin, (k + 1) * DT


def _alt(W, tol):
    d = np.diff(W, axis=0)
    sc = ((np.sign(d[1:]) * np.sign(d[:-1]) < 0)
          & (np.abs(d[1:]) > tol) & (np.abs(d[:-1]) > tol))
    return int(sc.sum(axis=0).max())


def _lat(r, w0=int(round(0.5 / DT))):
    VL, FL, WZ = r["VL"], r["FL"], r["WZ"]

    def flips(x, tol):
        a, b = x[w0:-1], x[w0 + 1:]
        s = (np.sign(a) * np.sign(b) < 0) & (np.abs(a) > tol) & (np.abs(b) > tol)
        return int(s.sum(axis=0).max()) if x.ndim > 1 else int(s.sum())

    d = np.diff(WZ[w0:])
    alt_wz = int(((np.sign(d[1:]) * np.sign(d[:-1]) < 0)
                  & (np.abs(d[1:]) > 1e-3) & (np.abs(d[:-1]) > 1e-3)).sum())
    return flips(VL, 1e-3), flips(FL, 10.0), alt_wz


# --- N17: PD13 post-stop matrix ------------------------------------------------

_PD13 = [
    # id, kind, scenario, travel sign
    ("car_b05", "car", dict(V0=10, BRK=0.5, SECS=12), 1),
    ("car_b10", "car", dict(V0=10, BRK=1.0, SECS=10), 1),
    ("car_t05b10", "car", dict(V0=5, THR=0.5, BRK=1.0, SECS=10), 1),
    ("car_t10b10", "car", dict(V0=5, THR=1.0, BRK=1.0, SECS=10), 1),
    ("car_t05b05", "car", dict(V0=5, THR=0.5, BRK=0.5, SECS=10), 1),
    ("car_t03b05", "car", dict(V0=5, THR=0.3, BRK=0.5, SECS=10), 1),
    ("car_rev_b10", "car", dict(V0=-5, BRK=1.0, SECS=8), -1),
    ("car_rev_both", "car", dict(V0=-5, THR=-0.5, BRK=1.0, SECS=10), -1),
    ("car_revdown10", "car", dict(DEG=-10, V0=-3, BRK=1.0, SECS=10), -1),
    ("car_down10", "car", dict(DEG=10, V0=3, BRK=1.0, SECS=10), 1),
    ("tank_b05", "tank", dict(V0=5, BRK=0.5, SECS=16), 1),
    ("tank_b10", "tank", dict(V0=5, BRK=1.0, SECS=12), 1),
    ("tank_t03b10", "tank", dict(V0=5, THR=0.3, BRK=1.0, SECS=20), 1),
    ("tank_rev_b10", "tank", dict(V0=-5, BRK=1.0, SECS=12), -1),
    ("tank_down05", "tank", dict(DEG=5, V0=2, BRK=1.0, SECS=12), 1),
]

#: D-PS (a), user decision 2026-11-20: car throttle 1.0 + brake 1.0 reverses at
#: about -0.0104 m/s for one step with zero back-travel. Accepted and documented
#: AS MEASURED (design probe and this tree -0.010357 m/s; this test's bound is
#: that value rounded outward at the 4th decimal) — not a relaxed gate for any
#: other row. (The first version of this file used -0.0105, wider than this
#: comment said.) This row also catches the A mutant (no re-anchor at the
#: first reversal: -0.01214 m/s), see test_n20.
_DPS_VMIN = -0.0104


@pytest.mark.parametrize("row", _PD13, ids=[r[0] for r in _PD13])
def test_n17_no_back_travel_or_reverse_speed_after_a_stop(row, request):
    """PD13 (AC-PS1): after the first |v| < 0.05 m/s, back-travel ≤ 2 mm and
    the speed along the travel direction ≥ -0.01 m/s, on every row of the
    plan's §2.4 (the tank's 10° hold is brake-capacity limited and not in
    the list). On the 1.7.0 tree five rows fail because the vehicle never
    stops (car throttle 1 + brake 1, car 10° downhill forward and reversing,
    tank throttle 0.3 + brake 1, tank 5° downhill); the other ten pass there
    too, by design — the 1.7.0 brake faded near standstill and never
    over-stopped — and guard against the back-slide a full-strength brake
    produced with the 1.7.0 engagement anchor (design probe, round 2: car
    brake 1.0 from 10 m/s -0.107 m/s, tank brake 0.5 22.8 mm)."""
    rid, kind, scen, sign = row
    rig = request.getfixturevalue(kind)
    back, vmin, t_stop = _pd13(rig.run(**scen), sign)
    assert back <= 0.002, f"{rid}: back-travel {back * 1000:.2f} mm (stop {t_stop:.3f} s)"
    floor = _DPS_VMIN if rid == "car_t10b10" else -0.01
    assert vmin >= floor, f"{rid}: vmin {vmin:+.5f} m/s (stop {t_stop:.3f} s)"


# --- N11 / N13: both pedals stop and hold --------------------------------------

@pytest.mark.parametrize("kind, scen", [
    ("car", dict(V0=5, THR=1.0, BRK=1.0, SECS=10)),
    ("tank", dict(V0=5, THR=0.3, BRK=1.0, SECS=20)),
])
def test_n11_brake_above_drive_stops_and_holds(kind, scen, request):
    """AC-B1: both pedals with the brake above the drive torque on every
    wheel: stop within 10 s, last 5 s max |v| < 0.02 m/s and displacement
    < 0.02 m (tank run 20 s). On 1.7.0 neither stops (car creeps at 0.728
    m/s, tank at 1.09 m/s in the design probe)."""
    rig = request.getfixturevalue(kind)
    r = rig.run(**scen)
    V, X = r["V"], r["X"]
    st = np.where(np.abs(V) < 0.05)[0]
    assert st.size and (st[0] + 1) * DT <= 10.0, "did not stop within 10 s"
    n5 = int(round(5.0 / DT))
    assert float(np.abs(V[-n5:]).max()) < 0.02
    assert float(np.linalg.norm((X[-1] - X[-n5])[:2])) < 0.02


def test_n13_the_tank_hold_does_not_wind_up_after_both_pedal_stop(tank):
    """After the both-pedal stop (tank throttle 0.3 + brake 1.0) the lock does
    not wind up and push the tank back: from the first |v| < 0.05 m/s on, the
    speed never reverses by more than PD13's 0.01 m/s (measured -0.00112),
    and over the last 5 s |v| < 0.02 m/s. (The round-2 design candidate, whose
    lock force was not limited by the clamp bounds, reversed at -0.12 m/s
    here.) The reverse-speed half restates N17[tank_t03b10] on the same cached
    run (which also bounds back-travel); what this test adds is the last-5-s
    hold. (Its first version used -0.02, a looser restatement of that row.)"""
    r = tank.run(V0=5, THR=0.3, BRK=1.0, SECS=20)
    V = r["V"]
    st = np.where(np.abs(V) < 0.05)[0]
    assert st.size, "never stopped"          # the 1.7.0 tree: creeps at ~1.09 m/s
    k = int(st[0])
    assert float(V[k:].min()) >= -0.01, float(V[k:].min())
    n5 = int(round(5.0 / DT))
    assert float(np.abs(V[-n5:]).max()) < 0.02


# --- N12 / N18: slope holds ------------------------------------------------------

@pytest.mark.parametrize("deg", [10.0, 20.0])
def test_n12_car_stops_and_holds_downhill_from_3_mps(car, deg):
    """AC-H1: car, brake 1.0 from 3 m/s down a 10° / 20° slope: stop ≤ 3 s,
    last 5 s max |v| < 0.01 m/s, displacement < 5 mm. On 1.7.0 it creeps
    (10°: 0.907 m/s) or speeds up (20°)."""
    r = car.run(DEG=deg, V0=3, BRK=1.0, SECS=10)
    V, X = r["V"], r["X"]
    st = np.where(np.abs(V) < 0.05)[0]
    assert st.size and (st[0] + 1) * DT <= 3.0
    n5 = int(round(5.0 / DT))
    assert float(np.abs(V[-n5:]).max()) < 0.01
    assert float(np.linalg.norm((X[-1] - X[-n5])[:2])) < 0.005


def test_n18_a_downhill_stop_is_taken_over_by_the_creep_test(car):
    """A6's creep branch: on 20° the deadbeat-limited brake balances gravity
    at about g*sin(20°)*dt/s ≈ 0.1 m/s, above ``REST_SPEED`` (0.05). The lock
    must take over there (held-wheel deadbeat binding AND speed no longer
    falling) — a fixed 0.05 m/s threshold never fires on this slope and the
    car creeps (0.089 m/s in the design probe). Asserts the takeover happened
    while a wheel's planar contact speed was still ≥ ``REST_SPEED``, and the
    AC-H1 hold."""
    seen = []

    def watch(rig):
        h = rig.sfl()
        if h._rest is not None and not seen and bool(h._rest.any()):
            seen.append(float(h._v_prev_sq.max()) ** 0.5)

    r = car.run(DEG=20.0, V0=3, BRK=1.0, SECS=10, every=watch)
    assert seen, "the lock never took over"
    assert seen[0] >= car.sfl().REST_SPEED, (
        f"takeover at planar contact speed {seen[0]:.4f} m/s: not the creep branch")
    n5 = int(round(5.0 / DT))
    assert float(np.abs(r["V"][-n5:]).max()) < 0.01


# --- N19 / N20: the lock holds the rest point ----------------------------------

# (N19 / N20 rewritten in the fix round: see their docstrings)

def _back_from_furthest(r, sign):
    """Back-travel (m) from the FURTHEST point reached after the first
    |v| < 0.05 m/s, along travel sign ``sign``. Unlike PD13's back-travel
    (``_pd13``: relative to the position AT that first step), this also sees
    a vehicle that rolls on after the first slow step and is then pulled back
    part of the way."""
    V, X, p0 = r["V"], r["X"], r["p0"]
    xa = sign * (X[:, 0] - p0[0])
    st = np.where(np.abs(V) < 0.05)[0]
    assert st.size, "never stopped"
    k = int(st[0])
    return float(np.max(np.maximum.accumulate(xa[k:]) - xa[k:]))


def test_n19_tank_downhill_hold_does_not_ring(tank):
    """C (damping at least critical on the wheel's mass share, 5.8 t on the
    reference tank, where the default ``k_damp`` is ζ ≈ 0.185): tank, brake
    1.0 from 2 m/s down a 3° slope. After the first |v| < 0.05 m/s the lock
    holds the slope load with its spring; with C it does not ring: back-travel
    from the furthest point ≤ 1 mm and speed along the slope ≥ -0.005 m/s.

    Discriminates C (one-line mutant: the critical floor multiplied by 0, so
    D = ``k_damp``): this tree 0.000 mm / -7.4e-6 m/s, mutant 4.89 mm /
    -0.0227 m/s with 11 speed sign changes (this tree 0). Measured with
    ``_Rig`` (CPU/WSL2, genesis-world 1.4.5, per_vehicle). The first version
    of this test (flat ground, brake 1.0 from 5 m/s, residual ≤ 1 mm over the
    last 3 s) passed on the mutant too (0.37 µm vs 2.5 µm): with no external
    load the spring is barely excited, so it did not guard C; it was
    replaced. N17[tank_down05] guards C as well (mutant: 7.83 mm back from the
    furthest point, vmin -0.0364). On the 1.7.0 tree this test fails because
    the tank never stops on the 3° slope (1.43 m in the last 3 s)."""
    r = tank.run(DEG=3, V0=2, BRK=1.0, SECS=12)
    back = _back_from_furthest(r, 1)
    k = int(np.where(np.abs(r["V"]) < 0.05)[0][0])
    vmin = float(r["V"][k:].min())
    assert back <= 0.001, f"rang back {back * 1000:.2f} mm from the furthest point"
    assert vmin >= -0.005, f"vmin {vmin:+.5f} m/s"


@pytest.mark.parametrize("V0, sign", [(5, 1), (-5, -1)], ids=["fwd", "rev"])
def test_n20_the_lock_holds_where_the_tank_came_to_rest(tank, V0, sign):
    """A (the first step after engagement on which the contact stopped or
    reversed re-anchors the longitudinal spring at the contact): tank, brake
    1.0 from 5 m/s forward and from -5 m/s. The lock takes over before the
    contact has quite stopped and the tank rolls on ~0.63 mm; with A the
    spring is re-anchored where it stopped, so the back-travel from the
    furthest point reached after the first |v| < 0.05 m/s is ≤ 0.5 mm.

    Discriminates A (one-line mutant: ``came_to_rest`` forced False, so the
    spring keeps its takeover anchor and pulls the tank back past where it
    stopped): this tree 0.137 / 0.138 mm, mutant 0.978 / 0.977 mm (fwd / rev;
    ``_Rig``, CPU/WSL2, genesis-world 1.4.5, per_vehicle). The car's brake
    stops do not discriminate A (car brake 1.0 from 10 m/s: identical on
    both); N17[car_t10b10] (the D-PS row) does, at -0.01214 m/s on the mutant.

    The PD13 back-travel gate is NOT this test: it is N17 (e.g. N17[car_b10],
    back-travel from the position at the first |v| < 0.05 m/s ≤ 2 mm, 0 mm
    measured), which this test does not replace. The first version of this
    test (car, brake 1.0 from 10 m/s, displacement ≤ 1 mm after the step on
    which the lock took over on every wheel, and no motion behind that
    point) passed on the mutant too: it measured from the takeover point,
    which is exactly where the mutant's spring pulls the car back to.

    Passes on the 1.7.0 tree too (0.000 mm both ways): its brake faded near
    standstill, so the lock engaged on a contact that had already stopped
    and there was nothing to re-anchor."""
    r = tank.run(V0=V0, BRK=1.0, SECS=12)
    back = _back_from_furthest(r, sign)
    assert back <= 0.0005, f"pulled back {back * 1000:.3f} mm from where it stopped"


# --- N21: the plant carries the lock's rest state --------------------------------

def test_n21_a_plant_on_a_held_car_starts_from_the_lock_state(car):
    """``DifferentiablePlant._sync_hooks`` copies EVERY tensor attribute of a
    hook, so the prediction copy of the lock starts with the live
    ``_rest`` / ``_s_eng`` / ``_rested`` / ``_v_prev_sq`` (v1.8.0 state), and a
    plant built on a held car predicts no motion at brake 1.0."""
    from genesis_vehicle.control.plant import DifferentiablePlant
    car.run(V0=10, BRK=1.0, SECS=6)
    live = car.sfl()
    assert bool(live._rest.all())
    p = DifferentiablePlant(car.veh, brake=1.0)
    p._sync_hooks()
    pred = [h for h in p._pred_post_hooks if hasattr(h, "STATE_ATTRS")][0]
    for a in live.STATE_ATTRS:
        assert torch.equal(getattr(pred, a), getattr(live, a)), a
    a, wz = p.predict(0.0, 0.0)
    assert abs(a) < 0.05 and abs(wz) < 0.01, (a, wz)


# --- N3: no step-to-step alternation under light inputs --------------------------

@pytest.mark.parametrize("kind", ["car", "tank"])
def test_n3_throttle_brake_and_reverse_do_not_alternate(kind, request):
    """AC-A1..A3 on a shortened window (10 s from 1 m/s): throttle 0.15 —
    alt@1e-4 = alt@0.01 = 0, max |dω| ≤ 0.05 rad/s, chassis speed never falls;
    brake 0.05 — alt@1e-4 = 0, speed rises on ≤ 5 steps; reverse throttle
    -0.3 — alt@0.01 = 0, max |dω| ≤ 0.06. On 1.7.0 throttle 0.15 alternated
    on 798 (car) / 228 (tank) of 800 steps in the design probe."""
    rig = request.getfixturevalue(kind)
    r = rig.run(V0=1, THR=0.15, SECS=10)
    assert _alt(r["W"], 1e-4) == 0 and _alt(r["W"], 1e-2) == 0
    assert float(np.abs(np.diff(r["W"], axis=0)).max()) <= 0.05
    assert int((np.diff(r["V"]) < -1e-5).sum()) == 0
    r = rig.run(V0=1, BRK=0.05, SECS=10)
    assert _alt(r["W"], 1e-4) == 0
    assert int((np.diff(r["V"]) > 1e-5).sum()) <= 5
    r = rig.run(V0=-1, THR=-0.3, SECS=10)
    assert _alt(r["W"], 1e-2) == 0
    assert float(np.abs(np.diff(r["W"], axis=0)).max()) <= 0.06


# --- N25 / N26: the lateral / yaw 2-cycle ----------------------------------------

_LAT = {
    "k_thr03": dict(THR=0.3, VLAT=0.05, WZ0=0.02, kick=True, SECS=3.0),
    "n_thr03": dict(THR=0.3, SECS=3.0),
    "s_v1": dict(V0=1.0, VLAT=0.05, WZ0=0.02, kick=True, SECS=2.0),
    "k_turn_slow": dict(V0=1.0, THR=0.05, STEER=0.3, kick=True, SECS=6.0),
}


def test_n25_reference_car_has_no_lateral_two_cycle(car):
    """AC-L1 on the reference car, throttle 0.3 after a v_lat 0.05 m/s / ω_z
    0.02 rad/s kick, 3 s, metrics after the first 0.5 s: flip_vlat = flip_flat
    = alt_wz = 0. On 1.7.0: 99 / 99 / 66 (the explicit lateral gain
    ``B_y*C_y*mu*g*dt/max(|v_long|, eps_v)`` is 5.1 below 0.5 m/s)."""
    assert _lat(car.run(**_LAT["k_thr03"])) == (0, 0, 0)


@pytest.fixture(scope="module")
def low_izz():
    return _Rig("car", urdf=LOW_IZZ_URDF)


def test_n26_low_yaw_inertia_car_scale_is_what_the_urdf_implies(low_izz):
    """The test vehicle really is a ρ²/k² > 2 vehicle: composite yaw inertia
    about the centre of mass ≈ 914 kg*m^2 (design probe 914.4), static s ≈
    0.29 (probe 0.288)."""
    low_izz.run(**_LAT["n_thr03"])
    yi = low_izz.phys.wheel_meta.yaw_inertia
    assert yi.i_cg == pytest.approx(914.4, rel=0.01)
    m = low_izz.phys._mass_props.mass
    k2 = yi.i_cg / m
    assert float(yi.rho_sq.mean()) / k2 > 2.0


@pytest.mark.parametrize("case", list(_LAT))
def test_n26_low_yaw_inertia_car_has_no_two_cycle_with_z(low_izz, case):
    """A vehicle whose wheels sit far from its centre of mass relative to its
    yaw inertia (``tests/data/low_yaw_inertia_car.urdf``, ρ²/k² ≈ 3.5): with
    the rigid-body scale Z the lateral deadbeat is deadbeat in the yaw mode
    too — 0 / 0 / 0 on all four cases."""
    assert _lat(low_izz.run(**_LAT[case])) == (0, 0, 0), case


def test_n26_without_z_the_same_vehicle_alternates_in_yaw(low_izz):
    """Non-vacuity of the test above: with Z disabled on the driver
    (``YawInertia(inf, rho_sq)``, s = 1 exactly — the per-wheel deadbeat on
    each wheel's own mass share) the same vehicle alternates in yaw on every
    case (design probe, Y without Z: alt_wz 98 / 98 / 58 / 218)."""
    for case, scen in _LAT.items():
        _, _, alt_wz = _lat(low_izz.run(zoff=True, **scen))
        assert alt_wz >= 20, (case, alt_wz)


# --- N23: Z inputs on every driver path -------------------------------------------

def test_n23_reference_car_static_scale(car):
    """The reference car's composite properties (m 1330 kg, I_cg 2714.4
    kg*m^2) give s ≈ 0.856 at static load; within 1%."""
    from genesis_vehicle.dynamics import DEADBEAT_G, deadbeat_scale
    car.run(**_LAT["n_thr03"])
    phys = car.phys
    yi = phys.wheel_meta.yaw_inertia
    assert yi.i_cg == pytest.approx(2714.4129782959144, rel=1e-6)
    m = phys._mass_props.mass
    N = torch.full((1, phys.wheel_meta.n_wheels), m * DEADBEAT_G / phys.wheel_meta.n_wheels)
    assert float(deadbeat_scale(N, yi)) == pytest.approx(0.856, rel=0.01)


def _scene(solver, z=0.5):
    VehicleScene.init_backend("cpu")
    vs = VehicleScene(dt=DT, substeps=SUB, n_envs=1, raycast_mode="dual_scene",
                      show_viewer=False, solver=solver)
    vs.add_ground_plane(friction=1.0)
    veh = vs.add_vehicle(CAR_URDF, presets.car_4w_rwd_ackermann,
                         pos=(0.0, 0.0, z), material=genesis.materials.Rigid(friction=1.0))
    vs.build()
    return vs, veh


def _driver(vs, veh):
    return veh.physics if veh.physics is not None else vs._mvp.kinds[0]._proto


def test_n23_n29_z_inputs_on_both_solvers_and_after_a_rebuild():
    """Z's inputs exist after the first step on a per_vehicle driver AND on a
    batched kind's proto (the batched kind never calls the proto's step()),
    with equal values; a ``mark_config_dirty()`` rebuild carries the mass
    properties (the same object) and recomputes ρ² from the rebuilt wheel
    positions (PC7: moved here by editing a wheel position before the
    rebuild)."""
    pv_vs, pv = _scene("per_vehicle")
    b_vs, b = _scene("batched")
    for vs, veh in ((pv_vs, pv), (b_vs, b)):
        veh.set_inputs(0.3, 0.0, 0.0)
        vs.step()
    zp = _driver(pv_vs, pv).wheel_meta.yaw_inertia
    zb = _driver(b_vs, b).wheel_meta.yaw_inertia
    assert zp is not None and zb is not None
    assert zp.i_cg == zb.i_cg and torch.equal(zp.rho_sq, zb.rho_sq)
    old = _driver(b_vs, b)
    mp = old._mass_props
    for _ in range(20):
        b_vs.step()
    b.cfg.wheels[0].position = (b.cfg.wheels[0].position[0] + 0.1,
                                *b.cfg.wheels[0].position[1:])
    b_vs.mark_config_dirty()
    b_vs.step()
    new = _driver(b_vs, b)
    assert new is not old
    assert new._mass_props is mp                          # carried, not re-read
    yi = new.wheel_meta.yaw_inertia
    pos = new.wheel_meta.positions
    cx, cy = mp.com[0], mp.com[1]
    want = (pos[:, 0] - cx) ** 2 + (pos[:, 1] - cy) ** 2
    assert torch.equal(yi.rho_sq, want)                   # recomputed, new positions
    assert not torch.equal(yi.rho_sq, zb.rho_sq)          # ... which really moved


def test_n23_a_mass_read_failure_means_s_1_and_one_warning(monkeypatch):
    """A ``ValueError`` from the mass walk (no link carries mass) sets s = 1
    for that driver (``YawInertia(inf, ...)``) and logs ONE
    ``[genesis_vehicle:deadbeat-scale]`` warning naming the vehicle; nothing
    else is caught. A root link with neither inertial spelling raises
    ``AttributeError`` (an unknown engine, not a massless vehicle)."""
    import logging
    import genesis_vehicle.control.plant as PL
    vs, veh = _scene("per_vehicle")
    phys = veh.physics

    def boom(entity):
        raise ValueError("could not read any link mass off the entity")

    monkeypatch.setattr(PL, "plant_mass_properties", boom)
    lines = []

    class H(logging.Handler):
        def emit(self, rec):
            lines.append(rec.getMessage())

    h = H()
    logging.getLogger("genesis_vehicle").addHandler(h)
    try:
        for _ in range(3):
            veh.set_inputs(0.3, 0.0, 0.0)
            vs.step()
    finally:
        logging.getLogger("genesis_vehicle").removeHandler(h)
    hits = [l for l in lines if "[genesis_vehicle:deadbeat-scale]" in l]
    assert len(hits) == 1 and phys.vehicle_name in hits[0], lines
    assert phys.wheel_meta.yaw_inertia.i_cg == float("inf")

    phys.wheel_meta.yaw_inertia = None
    phys._mass_props = None

    class NoSpelling:
        name = "base_link"

    class FakeEnt:
        links = [NoSpelling()]

    phys.entity, real = FakeEnt(), phys.entity
    try:
        with pytest.raises(AttributeError, match="inertial_mass"):
            phys._ensure_yaw_inertia()
    finally:
        phys.entity = real


# --- N28 / N29: the plant never writes the driver's Z; solvers agree --------------

@pytest.mark.parametrize("solver", ["per_vehicle", "batched"])
def test_n28_a_plant_with_explicit_masses_never_touches_the_driver(solver):
    """PD20: drive 60 steps, build ``DifferentiablePlant(veh, mass=600,
    izz=300, com=(0, 0, 0))`` (a plant-local s far from the car's), run
    ``jacobian`` and ``solve``; the driver's Z inputs are the same object with
    the same values, and its next 40 steps are bit-identical to a twin scene
    that never built a plant."""
    from genesis_vehicle.control.plant import DifferentiablePlant
    trajs = {}
    for build_plant in (False, True):
        vs, veh = _scene(solver)
        for _ in range(60):
            veh.set_inputs(0.3, 0.0, 0.05)
            vs.step()
        wm = _driver(vs, veh).wheel_meta
        z0 = wm.yaw_inertia
        snap = (z0.i_cg, z0.rho_sq.clone())
        if build_plant:
            p = DifferentiablePlant(veh, mass=600.0, izz=300.0, com=(0.0, 0.0, 0.0))
            p.jacobian(0.3, 0.05)
            p.solve(5.0, 0.5, 0.05)
            assert p._yaw_inertia.i_cg == pytest.approx(300.0)
            assert wm.yaw_inertia is z0
            assert wm.yaw_inertia.i_cg == snap[0]
            assert torch.equal(wm.yaw_inertia.rho_sq, snap[1])
        traj = []
        for _ in range(40):
            veh.set_inputs(0.3, 0.0, 0.05)
            vs.step()
            traj.append(veh.get_pos()[0].cpu().tolist() + veh.get_vel()[0].cpu().tolist())
        trajs[build_plant] = traj
    assert trajs[False] == trajs[True]


def test_n29_the_two_solvers_give_the_same_rollout():
    """The default batched solver runs the same pipeline with the same Z
    inputs as per_vehicle: the kicked throttle-0.3 rollout (``_LAT``
    ``k_thr03``, 1 s settle + 3 s) is bit-identical between them (the design
    probe found all 121 scenario rows identical across solvers). Passes on
    the 1.7.0 tree too, by design: there NEITHER solver had Z. It fails if one
    solver gets the deadbeat-scale inputs and the other does not (the design
    probe's round-3 defect: Z absent on the batched kind); the inputs
    themselves are asserted in ``test_n23_n29_...``."""
    out = {}
    for solver in ("per_vehicle", "batched"):
        rig = _Rig("car", solver=solver)
        r = rig.run(**_LAT["k_thr03"])
        out[solver] = (r["V"], r["VL"], r["X"], r["W"])
    for a, b in zip(out["per_vehicle"], out["batched"]):
        assert np.array_equal(a, b)


# --- N7: closed-form windows -------------------------------------------------------

def _a_closed(rig, v, thr, brk):
    """The closed form of the 1.8.0 design measurements: (sum of drive torque /
    r - sum of brake torque * tanh(ω/0.5) / r - cr*M*g*tanh(v/0.5) - aero) /
    M_eff, with M the composite mass and M_eff = M + sum(i_wheel / r^2)."""
    from genesis_vehicle.inputs import VehicleInputs
    phys = rig.phys
    wm, R = phys.wheel_meta, phys.resolved
    M = float(plant_mass_properties(rig.ent).mass)
    m_eff = M + float((wm.i_wheel / wm.radius ** 2).sum())
    r = float(wm.radius.mean())
    cr = float(wm.rolling_resistance_cr.mean())
    ch = R.chassis
    ins = R.steering.InputType.from_unified(VehicleInputs(
        throttle=torch.tensor([thr]), brake=torch.tensor([brk]), steer=torch.tensor([0.0])))
    om = torch.full((1, wm.n_wheels), v / r)
    Td, Tb = R.drivetrain.distribute_torque(ins, om, wm, torch.device("cpu"), torch.float32)
    aero = 0.5 * ch.air_density * (getattr(ch, "drag_area", 0.0) or 0.0) * v * v
    return (float(Td.sum()) / r - float(Tb.sum()) * math.tanh((v / r) / 0.5) / r
            - cr * M * 9.81 * math.tanh(v / 0.5) - aero) / m_eff


@pytest.mark.parametrize("mode, scen, va, vb", [
    ("brake", dict(V0=4, BRK=1.0, SECS=2), 3.0, 0.6),
    ("accel", dict(V0=2, THR=1.0, SECS=12), 5.0, 15.0),
])
def test_n7_brake_and_drive_follow_their_closed_forms(car, mode, scen, va, vb):
    """AC-R1 windows: car brake 1.0 from 3 to 0.6 m/s (the regime where the
    1.7.0 brake faded: ratio 0.40) and full throttle from 5 to 15 m/s (1.7.0
    0.98): measured mean acceleration over the window / the closed form's
    window average within [0.99, 1.01]. Design probe 0.9996 / 1.0014."""
    r = car.run(**scen)
    V = r["V"]
    T = (np.arange(len(V)) + 1) * DT
    if mode == "brake":
        ia, ib = int(np.argmax(V <= va)), int(np.argmax(V <= vb))
    else:
        ia, ib = int(np.argmax(V >= va)), int(np.argmax(V >= vb))
    assert ib > ia
    meas = (V[ib] - V[ia]) / (T[ib] - T[ia])
    vv = np.linspace(V[ia], V[ib], 200)
    a_c = np.array([_a_closed(car, x, scen.get("THR", 0.0), scen.get("BRK", 0.0)) for x in vv])
    closed = (V[ib] - V[ia]) / np.trapezoid(1.0 / a_c, vv)
    assert 0.99 <= meas / closed <= 1.01, (mode, meas, closed)
