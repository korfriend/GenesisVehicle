"""2.0.4 Step D: the planar rigid-body term of the StaticFrictionLock spring cap.

``k_eff_i = min(k_spring, f * m_static_i / dt^2)``, ``f = min(SPRING_C_MAX,
SPRING_C_RB / beta)``, ``SPRING_C_MAX`` 1.0, ``SPRING_C_RB`` 1.5 (the user's
constant), ``beta`` = ``wheel_meta.static_rigid_factor.beta``
(``dynamics.rigid_share_factor``). For ``beta <= 1.5`` ``f`` is
``SPRING_C_MAX`` itself and ``k_eff`` is bit-for-bit the per-wheel cap.

| abbr | meaning |
|---|---|
| SFL | `StaticFrictionLock` |
| k_eff | the capped per-wheel lock stiffness, N/m |
| m_static | per-wheel static mass share, kg (`WheelMeta.static_mass_share`) |
| beta | largest eigenvalue of `S v = beta P v`, `S = sum m_i J_i^T J_i`, `P = diag(M, M, I_cg)` |
| I_cg | composite yaw inertia about the centre of mass, kg*m^2 |
| COM | centre of mass |
| REST | mean planar speed over the last 2 s < 1e-3 m/s, net planar displacement over the second half <= 1 mm, max wheel rim speed over the last 2 s <= 1e-3 m/s |

The literal mass properties below were read off built CPU entities with the
link walk (``control.plant.plant_mass_properties`` convention; genesis-world
1.4.5): reference car ``car_4w.urdf``, ``tests/data/low_yaw_inertia_car.urdf``,
reference tank ``tests/data/tank_ref.urdf`` and the truck demo URDF
``GeneVehicle_Truck6w/urdf/truck_6w.urdf`` (``I`` below is ``I_cg``).
"""

from __future__ import annotations

import copy
import itertools
import logging
import math
import os
import types

import numpy as np
import pytest
import torch

from genesis_vehicle import dynamics
from genesis_vehicle.dynamics import RigidShareFactor, rigid_share_factor
from genesis_vehicle.strategies.stability import StaticFrictionLock

HERE = os.path.dirname(os.path.abspath(__file__))
CAR_URDF = os.path.abspath(os.path.join(HERE, "..", "samples", "urdf", "car_4w.urdf"))
LOWYAW_URDF = os.path.join(HERE, "data", "low_yaw_inertia_car.urdf")
DEV, FDT, F64 = torch.device("cpu"), torch.float32, torch.float64

_CAR_POS = [[1.350000023841858, 0.75, 0.30000001192092896],
            [1.350000023841858, -0.75, 0.30000001192092896],
            [-1.350000023841858, 0.75, 0.30000001192092896],
            [-1.350000023841858, -0.75, 0.30000001192092896]]
_TANK_POS = ([[x, 1.4199999570846558, 0.0] for x in (3.0, 1.5, 0.0, -1.5, -3.0)]
             + [[x, -1.4199999570846558, 0.0] for x in (3.0, 1.5, 0.0, -1.5, -3.0)])
_TANK_SH = [5911.33349609375, 5890.66650390625, 5870.0, 5849.33349609375, 5828.66650390625] * 2
_TRUCK_POS = [[1.7999999523162842, 0.8500000238418579, 0.4000000059604645],
              [1.7999999523162842, -0.8500000238418579, 0.4000000059604645],
              [0.0, 0.949999988079071, 0.4000000059604645],
              [0.0, -0.949999988079071, 0.4000000059604645],
              [-1.7999999523162842, 0.949999988079071, 0.4000000059604645],
              [-1.7999999523162842, -0.949999988079071, 0.4000000059604645]]
_TRUCK_SH = [901.6666870117188, 901.6666870117188, 896.6666870117188,
             896.6666870117188, 891.6666870117188, 891.6666870117188]

#: name -> (mass, (cx, cy), I_cg, positions, shares, expected beta)
VEH = {
    "car": (1330.0, (0.01015036536338634, 0.0), 2714.4129782959144, _CAR_POS,
            [335.0, 335.0, 330.0, 330.0], 1.168545),
    "lowyaw": (1330.0, (0.01015037567095649, 0.0), 914.412978295915, _CAR_POS,
               [335.0, 335.0, 330.0, 330.0], 3.468797),
    "tank": (58700.0, (0.010345581011587163, 0.0), 357730.65080716606, _TANK_POS,
             _TANK_SH, 1.069257),
    "truck": (5380.0, (0.006691452738801692, 0.0), 11173.309063446772, _TRUCK_POS,
              _TRUCK_SH, 1.445535),
}


def _izz(mass, com, i_cg):
    """Yaw inertia about the base-link ORIGIN (the helper's input convention)."""
    return i_cg + mass * (com[0] ** 2 + com[1] ** 2)


def _rsf(name, i_scale=1.0):
    m, com, icg, pos, sh, _ = VEH[name]
    return rigid_share_factor(m, com, _izz(m, com, icg * i_scale), pos, sh)


class _Meta:
    """A weak-referenceable duck-typed wheel meta for direct hook calls."""

    def __init__(self, share, rigid=None, dtype=FDT):
        n = len(share)
        self.n_wheels = n
        self.mu_long = torch.ones(n)
        self.mu_lat = torch.ones(n)
        self.static_mass_share = torch.tensor(share, dtype=dtype)
        self.static_rigid_factor = rigid


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, rec):
        msg = rec.getMessage()
        if "[genesis_vehicle:sfl-cap]" in msg:
            self.lines.append(msg)


@pytest.fixture
def caplines():
    h = _Capture()
    lg = logging.getLogger("genesis_vehicle")
    lg.addHandler(h)
    try:
        yield h.lines
    finally:
        lg.removeHandler(h)


def _pre_d_k_eff(share, k_raw, dt, dtype=FDT):
    """The pre-Step-D tree's ``_compute_k_eff`` expression, verbatim in its
    operand order (``c_max * share64 / (dt*dt)``)."""
    k = float(k_raw)
    c_max = float(StaticFrictionLock.SPRING_C_MAX)
    cap = c_max * share.detach().to(dtype=torch.float64).reshape(-1) / (dt * dt)
    capped = cap < k
    if not bool(capped.any()):
        return k_raw
    return torch.clamp(cap, max=k).to(device=DEV, dtype=dtype).reshape(1, -1)


# --- beta ---------------------------------------------------------------------

@pytest.mark.parametrize("name", ["car", "lowyaw", "tank", "truck"])
def test_rigid_share_factor_values(name):
    r = _rsf(name)
    assert isinstance(r, RigidShareFactor)
    assert isinstance(r.beta, float) and isinstance(r.yaw_fraction, float)
    assert abs(r.beta - VEH[name][5]) <= 1e-6
    assert r.yaw_fraction > 0.5            # every one of these is yaw-led


def test_rigid_share_factor_raises_on_bad_inertia():
    m, com, icg, pos, sh, _ = VEH["car"]
    with pytest.raises(ValueError):                       # I_cg < 0
        rigid_share_factor(m, (3.0, 0.0), 1.0, pos, sh)
    with pytest.raises(ValueError):                       # I_cg == 0
        rigid_share_factor(m, (0.0, 0.0), 0.0, pos, sh)
    with pytest.raises(ValueError):
        rigid_share_factor(m, com, float("nan"), pos, sh)
    with pytest.raises(ValueError):
        rigid_share_factor(float("inf"), com, _izz(m, com, icg), pos, sh)
    with pytest.raises(ValueError):
        rigid_share_factor(m, com, _izz(m, com, icg), pos, [335.0, float("nan"), 330.0, 330.0])
    with pytest.raises(ValueError):
        rigid_share_factor(m, com, _izz(m, com, icg), pos, [335.0, 335.0, 330.0])


def test_rigid_share_factor_raises_below_one():
    """Shares summing to less than the mass give beta < 1: ValueError (the
    driver then falls back to the logged per-wheel cap); 1 - 1e-7 passes."""
    m, com, icg, pos, sh, _ = VEH["car"]
    izz = _izz(m, com, icg * 100.0)                 # translation-led
    with pytest.raises(ValueError, match="beta"):
        rigid_share_factor(m, com, izz, pos, [s * 0.5 for s in sh])
    r = rigid_share_factor(m, com, izz, pos, [s * (1 - 1e-7) for s in sh])
    assert 1 - 1e-6 <= r.beta < 1.0


def test_beta_below_one_falls_back_to_the_per_wheel_cap(caplines):
    """Live driver path: a beta < 1 ValueError is caught by the driver's own
    guard: one 'rigid-body factor unavailable' line, share set, beta None."""
    import genesis_vehicle.core as core
    from genesis_vehicle.control.plant import PlantMassProperties
    m, com, icg, _, _, _ = VEH["car"]
    d = _fake_driver(PlantMassProperties(m, (com[0], com[1], 0.46), _izz(m, com, icg)))
    real = core.rigid_share_factor
    core.rigid_share_factor = lambda mass, c, izz, pos, sh: real(mass, c, izz, pos, [s * 0.5 for s in sh])
    try:
        core.VehiclePhysics._ensure_static_mass_share(d)
    finally:
        core.rigid_share_factor = real
    assert d.wheel_meta.static_mass_share is not None and d.wheel_meta.static_rigid_factor is None
    assert len(caplines) == 1 and "rigid-body factor unavailable" in caplines[0] and "beta" in caplines[0]


# --- k_eff --------------------------------------------------------------------

def test_low_yaw_k_eff_values():
    """beta 3.468797 -> f 0.432427: 231 781 / 228 321 N/m at dt 0.025;
    f*m/0.0016 at dt 0.04; uncapped at dt 0.0125 (927 123 > 500 000)."""
    meta = _Meta([335.0, 335.0, 330.0, 330.0], _rsf("lowyaw"))
    hook = StaticFrictionLock()
    k = hook._k_eff(meta, 0.025, DEV, FDT).reshape(-1).tolist()
    assert abs(k[0] - 231781) <= 1 and abs(k[1] - 231781) <= 1
    assert abs(k[2] - 228321) <= 1 and abs(k[3] - 228321) <= 1
    f = 1.5 / _rsf("lowyaw").beta
    k04 = hook._k_eff(meta, 0.04, DEV, FDT).reshape(-1).tolist()
    assert abs(k04[0] - f * 335.0 / 0.0016) <= 1 and abs(k04[2] - f * 330.0 / 0.0016) <= 1
    assert abs(k04[0] - 90539) <= 1 and abs(k04[2] - 89188) <= 1
    assert hook._k_eff(meta, 0.0125, DEV, FDT) is hook.k_spring


@pytest.mark.parametrize("beta", [1.0, 1.17, 1.4455, 1.5])
def test_f_is_c_max_bit_exact_below_1p5(beta):
    """For beta <= 1.5 the cap is the pre-Step-D expression bit for bit, at
    every dt and k_spring tried, in float32 AND float64 output; the
    light-car shares (67 / 66 / 58.3) at dt 1/30 and beta 1.5 are the
    switch-over case. (An operation-order mutant such as ``min(c_max*m/dt^2,
    C_RB*m/(beta*dt*dt))`` differs there by one float64 ulp — 60299.99999999999
    vs 60300.0 — which the cast to float32 hides, hence the float64 pass.)"""
    shares = ([335.0, 335.0, 330.0, 330.0], [67.0, 67.0, 66.0, 66.0], [58.3, 58.3, 66.0, 67.0],
              _TRUCK_SH, _TANK_SH)
    for sh, dt, ks, dtype in itertools.product(shares, (0.0125, 0.025, 1.0 / 30.0, 0.04, 0.05),
                                               (500_000.0, 50_000.0, 5e6), (FDT, F64)):
        hook = StaticFrictionLock(k_spring=ks)
        meta = _Meta(sh, RigidShareFactor(beta, 1.0))
        got = hook._k_eff(meta, dt, DEV, dtype)
        want = _pre_d_k_eff(meta.static_mass_share, hook.k_spring, dt, dtype)
        if torch.is_tensor(want):
            assert torch.is_tensor(got) and got.dtype == want.dtype and torch.equal(got, want), (sh, dt, ks)
        else:
            assert got is hook.k_spring, (sh, dt, ks)


def test_beta_write_re_derives():
    """The cache key carries static_rigid_factor by value: writing it on the
    same meta re-derives (uncapped -> rigid-term values)."""
    meta = _Meta([335.0, 335.0, 330.0, 330.0], None)
    hook = StaticFrictionLock()
    assert hook._k_eff(meta, 0.025, DEV, FDT) is hook.k_spring
    meta.static_rigid_factor = _rsf("lowyaw")
    k = hook._k_eff(meta, 0.025, DEV, FDT)
    assert torch.is_tensor(k) and abs(float(k[0, 0]) - 231781) <= 1
    meta.static_rigid_factor = _rsf("car")
    assert hook._k_eff(meta, 0.025, DEV, FDT) is hook.k_spring


def test_beta_none_is_the_d12_cap():
    """No static_rigid_factor (None, or a duck-typed meta without the field)
    gives exactly the per-wheel cap."""
    for dt in (0.025, 1.0 / 30.0, 0.04):
        for sh in ([335.0, 335.0, 330.0, 330.0], [67.0, 67.0, 66.0, 66.0]):
            a = _Meta(sh, None)
            b = _Meta(sh, None)
            del b.static_rigid_factor
            for meta in (a, b):
                hook = StaticFrictionLock()
                got = hook._k_eff(meta, dt, DEV, FDT)
                want = _pre_d_k_eff(meta.static_mass_share, hook.k_spring, dt)
                if torch.is_tensor(want):
                    assert torch.equal(got, want)
                else:
                    assert got is hook.k_spring


def test_beta_assert():
    for bad in (0.5, float("nan"), float("inf")):
        with pytest.raises(AssertionError):
            StaticFrictionLock()._k_eff(_Meta([335.0] * 4, RigidShareFactor(bad, 1.0)),
                                        0.025, DEV, FDT)


def _ctx(meta, n, v, N, dt=0.025):
    return types.SimpleNamespace(
        v_long=torch.full((1, n), v), v_lat=torch.zeros(1, n), N=torch.full((1, n), N),
        F_long=torch.zeros(1, n), F_lat=torch.zeros(1, n), dt=dt, wheel_meta=meta,
        deadbeat_scale=torch.ones(1, 1), brake_rem=torch.full((1, n), 100.0),
        F_long_deadbeat=torch.zeros(1, n))


def test_beta_is_static():
    """k_eff comes from the static share and the static beta, never from the
    instantaneous N: the same value at N 0, the static load and 3x it."""
    from genesis_vehicle._hotset import derived_value
    vals = []
    for N in (0.0, 335.0 * 9.81, 3 * 335.0 * 9.81):
        meta = _Meta([335.0, 335.0, 330.0, 330.0], _rsf("lowyaw"))
        h = StaticFrictionLock()
        h.apply_post_tire(_ctx(meta, 4, 0.01, N))
        h.apply_post_tire(_ctx(meta, 4, 0.01, N))
        vals.append(derived_value(h, "_k_eff").clone())
    assert torch.equal(vals[0], vals[1]) and torch.equal(vals[1], vals[2])
    assert abs(float(vals[0][0, 0]) - 231781) <= 1


def test_cap_log_names_the_binding_bound(caplines):
    # Yaw-dominated (the low-yaw car): the rigid-body term, beta, "yaw mode".
    StaticFrictionLock()._k_eff(_Meta([335.0, 335.0, 330.0, 330.0], _rsf("lowyaw")), 0.025, DEV, FDT)
    assert len(caplines) == 1
    line = caplines[0]
    assert "planar rigid-body term" in line and "C_RB 1.5" in line
    assert "beta 3.469" in line and "yaw mode" in line
    assert "231780.6" in line and "228321.2" in line and "14.2 mm" in line
    # Translation-dominated: shares summing to twice the mass on a large yaw
    # inertia -> beta ~2 with a translational leading mode: no "yaw mode".
    m, com, icg, pos, _, _ = VEH["car"]
    tr = rigid_share_factor(m, com, _izz(m, com, icg * 10.0), pos, [665.0] * 4)
    assert tr.beta > 1.5 and tr.yaw_fraction < 0.5
    StaticFrictionLock()._k_eff(_Meta([335.0] * 4, tr), 0.025, DEV, FDT)
    assert len(caplines) == 2
    assert "planar rigid-body term" in caplines[1] and "yaw mode" not in caplines[1]
    # f == C_MAX: the per-wheel term (reference car at dt 0.04).
    StaticFrictionLock()._k_eff(_Meta([335.0, 335.0, 330.0, 330.0], _rsf("car")), 0.04, DEV, FDT)
    assert len(caplines) == 3
    assert "per-wheel static load (c_max 1.0)" in caplines[2]
    assert "209375" in caplines[2] and "206250" in caplines[2] and "15.7 mm" in caplines[2]
    assert "rigid-body" not in caplines[2]


# --- the planar bound (D-4) ------------------------------------------------------

def _planar_check(mass, com, i_cg, pos, shares, k_eff, dt, steer):
    """lambda_max(K, P) * dt^2 and max_i k_i dt^2 / m_i for the planar model,
    springs written in each wheel's rotated (long, lat) axes."""
    P = np.diag([mass, mass, i_cg])
    K = np.zeros((3, 3))
    for (x, y, _), k, d in zip(pos, k_eff, steer):
        a, b = x - com[0], y - com[1]
        J = np.array([[1.0, 0.0, -b], [0.0, 1.0, a]])
        R = np.array([[math.cos(d), -math.sin(d)], [math.sin(d), math.cos(d)]])
        Kw = R @ np.diag([k, k]) @ R.T
        K += J.T @ Kw @ J
    ph = np.diag(np.diag(P) ** -0.5)
    lam = float(np.linalg.eigvalsh(ph @ K @ ph).max()) if len(pos) else 0.0
    c = max((k * dt * dt / m for k, m in zip(k_eff, shares)), default=0.0)
    return lam * dt * dt, c


def _k_eff_list(rigid, shares, dt, k_spring):
    hook = StaticFrictionLock(k_spring=k_spring)
    meta = _Meta(shares, rigid, dtype=F64)
    k = hook._k_eff(meta, dt, DEV, F64)
    return ([float(k)] * len(shares)) if not torch.is_tensor(k) else k.reshape(-1).tolist()


@pytest.mark.parametrize("name,i_scale", [("lowyaw", 1.0), ("car", 1.0), ("car", 0.3),
                                          ("tank", 1.0), ("tank", 0.25)])
def test_planar_bound_subsets_and_steer(name, i_scale):
    """lambda(K_eff, P) * dt^2 <= C_RB and c_i <= C_MAX for EVERY non-empty
    grounded-wheel subset (one side, three wheels, ...) and per-wheel steer
    angles in [-0.6, 0.6] rad, on a 4-wheel car and a 10-wheel tank fixture
    (also with the yaw inertia scaled down, so the rigid term binds)."""
    m, com, icg, pos, sh, _ = VEH[name]
    icg = icg * i_scale
    rigid = rigid_share_factor(m, com, _izz(m, com, icg), pos, sh)
    rng = np.random.default_rng(1234)
    n = len(pos)
    for dt, ks in ((0.025, 500_000.0), (0.04, 500_000.0), (0.025, 1e12)):
        k = _k_eff_list(rigid, sh, dt, ks)
        for r in range(1, n + 1):
            for sub in itertools.combinations(range(n), r):
                for _ in range(2):
                    steer = rng.uniform(-0.6, 0.6, size=len(sub))
                    lam, c = _planar_check(m, com, icg, [pos[i] for i in sub], [sh[i] for i in sub],
                                           [k[i] for i in sub], dt, steer)
                    assert lam <= StaticFrictionLock.SPRING_C_RB * (1 + 1e-9), (name, sub, dt, lam)
                    assert c <= StaticFrictionLock.SPRING_C_MAX * (1 + 1e-9), (name, sub, dt, c)
    if name == "lowyaw" or i_scale < 1.0:
        # The rigid term is the binding one here: all wheels at 1e12 reach C_RB.
        k = _k_eff_list(rigid, sh, 0.025, 1e12)
        lam, _ = _planar_check(m, com, icg, pos, sh, k, 0.025, [0.0] * n)
        assert lam == pytest.approx(min(StaticFrictionLock.SPRING_C_RB, rigid.beta), rel=1e-9)


def test_planar_bound_unbalanced_fallback():
    """Equal shares with the COM 0.4 m off the wheel centroid (the
    equal-share fallback's situation): the exact 3x3 beta keeps the bound;
    the closed form max(1, sum m rho^2 / I_cg) would not."""
    m = 1330.0
    com = (0.4, 0.0)
    pos = _CAR_POS
    sh = [m / 4] * 4
    icg = 900.0
    rigid = rigid_share_factor(m, com, _izz(m, com, icg), pos, sh)
    rho2 = [(p[0] - com[0]) ** 2 + (p[1] - com[1]) ** 2 for p in pos]
    closed = max(1.0, sum(s * r for s, r in zip(sh, rho2)) / icg)
    assert rigid.beta > closed * (1 + 1e-3)
    for dt in (0.025, 0.04):
        k = _k_eff_list(rigid, sh, dt, 1e12)
        lam, c = _planar_check(m, com, icg, pos, sh, k, dt, [0.0] * 4)
        assert lam <= 1.5 * (1 + 1e-9) and c <= 1.0 * (1 + 1e-9)
        k_closed = _k_eff_list(RigidShareFactor(closed, 1.0), sh, dt, 1e12)
        lam_c, _ = _planar_check(m, com, icg, pos, sh, k_closed, dt, [0.0] * 4)
        assert lam_c > 1.5 * (1 + 1e-6)          # the closed form misses it


# --- driver side (Genesis-free fake drivers) ---------------------------------------

def _fake_driver(mass_props):
    from genesis_vehicle.config import resolve
    from genesis_vehicle.core import build_wheel_meta
    from genesis_vehicle.presets import car_4w_rwd_ackermann
    wm = build_wheel_meta(resolve(car_4w_rwd_ackermann(CAR_URDF, stability="control")), DEV, FDT)
    wm.yaw_inertia = object()
    return types.SimpleNamespace(wheel_meta=wm, _mass_props=mass_props, entity=object(),
                                 vehicle_name="fake_car")


def test_driver_computes_beta_from_the_link_walk():
    from genesis_vehicle.control.plant import PlantMassProperties
    from genesis_vehicle.core import VehiclePhysics
    m, com, icg, _, _, beta = VEH["lowyaw"]
    d = _fake_driver(PlantMassProperties(m, (com[0], com[1], 0.46), _izz(m, com, icg)))
    VehiclePhysics._ensure_static_mass_share(d)
    rf = d.wheel_meta.static_rigid_factor
    assert isinstance(rf, RigidShareFactor) and abs(rf.beta - beta) <= 1e-6


def test_beta_failure_keeps_the_per_wheel_cap(caplines, monkeypatch):
    """A ValueError from rigid_share_factor is caught on its own: the share IS
    set, beta is None, one 'rigid-body factor unavailable' line, k_eff is the
    per-wheel cap (209 375 / 206 250 at dt 0.04 on the reference car)."""
    import genesis_vehicle.core as core
    from genesis_vehicle.control.plant import PlantMassProperties

    def boom(*a, **k):
        raise ValueError("forced by the test")
    monkeypatch.setattr(core, "rigid_share_factor", boom)
    m, com, icg, _, _, _ = VEH["car"]
    d = _fake_driver(PlantMassProperties(m, (com[0], com[1], 0.46), _izz(m, com, icg)))
    core.VehiclePhysics._ensure_static_mass_share(d)
    wm = d.wheel_meta
    assert [round(float(v), 1) for v in wm.static_mass_share] == [335.0, 335.0, 330.0, 330.0]
    assert wm.static_rigid_factor is None
    assert len(caplines) == 1 and "rigid-body factor unavailable" in caplines[0]
    assert "forced by the test" in caplines[0] and "fake_car" in caplines[0]
    core.VehiclePhysics._ensure_static_mass_share(d)       # set: no further line
    assert len(caplines) == 1
    k = StaticFrictionLock()._k_eff(wm, 0.04, DEV, FDT)
    assert k.reshape(-1).tolist() == [209375.0, 209375.0, 206250.0, 206250.0]
    assert len(caplines) == 2 and "per-wheel static load" in caplines[1]


# --- real scenes ------------------------------------------------------------------

def _scene(urdf, dt, n_envs=1, cfg=None, preset=None):
    import genesis as gs
    from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann
    VehicleScene.init_backend("cpu")
    vs = VehicleScene(dt=dt, substeps=4, n_envs=n_envs, raycast_mode="dual_scene", show_viewer=False,
                      solver="per_vehicle")
    vs.add_ground_plane(friction=1.0)
    kw = {"cfg": cfg} if cfg is not None else {"stability": "control"}
    veh = vs.add_vehicle(urdf, preset or car_4w_rwd_ackermann, pos=(0.0, 0.0, 0.5),
                         material=gs.materials.Rigid(friction=1.0), **kw)
    vs.build()
    return vs, veh


def _sfl(veh):
    return [h for h in veh.physics.post_tire_hooks if isinstance(h, StaticFrictionLock)][0]


def test_low_yaw_car_rests_flat(caplines):
    """The low-yaw-inertia car (beta 3.469) rests on the flat at brake 0.3
    and 1.0 after 20 braked spawn steps (through the first 2.0.4 drafts it
    cycled in yaw at about 0.23 / 0.27 rad/s): REST over 15 s and |yaw rate|
    <= 6.44e-5 rad/s (1e-4 / rho_max, rho_max 1.5532 m) over the last 5 s."""
    import genesis as gs
    dt = 0.025
    vs, veh = _scene(LOWYAW_URDF, dt, n_envs=2)
    ent, p = veh.entity_main, veh.physics
    T = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=gs.device)  # noqa: E731
    veh.set_inputs(throttle=T([0, 0]), brake=T([1.0, 1.0]), steer=T([0, 0]))
    for _ in range(20):
        vs.step()
    veh.set_inputs(throttle=T([0, 0]), brake=T([0.3, 1.0]), steer=T([0, 0]))
    rr = p.wheel_meta.radius.reshape(1, -1).cpu().numpy()
    vp, pos, wr, yr = [], [], [], []
    nst = int(round(15.0 / dt))
    for _ in range(nst):
        vs.step()
        v = ent.get_vel().reshape(2, 3).cpu().numpy()
        vp.append(np.hypot(v[:, 0], v[:, 1]))
        pos.append(ent.get_pos().reshape(2, 3).cpu().numpy()[:, :2].copy())
        wr.append(np.abs(p.omega.detach().cpu().numpy()) * rr)
        yr.append(ent.get_ang().reshape(2, 3).cpu().numpy()[:, 2].copy())
    vp, pos, wr, yr = map(np.array, (vp, pos, wr, yr))
    L2, L5, half = int(2 / dt), int(5 / dt), nst // 2
    for i in range(2):
        assert vp[-L2:, i].mean() < 1e-3, (i, vp[-L2:, i].mean())
        assert np.hypot(*(pos[-1, i] - pos[half, i])) <= 1e-3
        assert wr[-L2:, i, :].max() <= 1e-3
        assert np.abs(yr[-L5:, i]).max() <= 6.44e-5, (i, np.abs(yr[-L5:, i]).max())
    rf = p.wheel_meta.static_rigid_factor
    assert abs(rf.beta - 3.468797) <= 1e-4 * 3.468797
    k = _sfl(veh)._k_eff(p.wheel_meta, dt, DEV, FDT).reshape(-1).tolist()
    assert abs(k[0] - 231781) <= 1 and abs(k[3] - 228321) <= 1
    assert len(caplines) == 1 and "yaw mode" in caplines[0] and "beta 3.469" in caplines[0]


def test_preset_yaw_inertia_does_not_change_beta():
    """beta comes from the link walk, never from wheel_meta.yaw_inertia: a
    preset YawInertia(inf, rho_sq) (s = 1) leaves beta and k_eff as computed."""
    from genesis_vehicle.dynamics import YawInertia
    vs, veh = _scene(LOWYAW_URDF, 0.025)
    p = veh.physics
    wm = p.wheel_meta
    wm.yaw_inertia = YawInertia(float("inf"), torch.zeros_like(wm.positions[..., 0]))
    veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
    vs.step()
    assert abs(wm.static_rigid_factor.beta - 3.468797) <= 1e-4 * 3.468797
    k = _sfl(veh)._k_eff(wm, 0.025, DEV, FDT).reshape(-1).tolist()
    assert abs(k[0] - 231781) <= 1 and abs(k[3] - 228321) <= 1


def test_plant_before_first_step_predicts_capped():
    """A DifferentiablePlant built BEFORE the live driver's first step makes
    the driver compute its static shares and beta at bind, so its prediction
    hooks use the capped spring (reference car, dt 0.04: 209 375 / 206 250)."""
    from genesis_vehicle import DifferentiablePlant
    from genesis_vehicle._hotset import derived_value
    vs, veh = _scene(CAR_URDF, 0.04)
    wm = veh.physics.wheel_meta
    assert wm.static_mass_share is None
    plant = DifferentiablePlant(veh)
    assert wm.static_mass_share is not None and wm.static_rigid_factor is not None
    plant.predict(0.0, 0.0)
    hooks = [h for h in plant._pred_post_hooks if isinstance(h, StaticFrictionLock)]
    k = derived_value(hooks[0], "_k_eff")
    assert torch.is_tensor(k) and k.reshape(-1).tolist() == [209375.0, 209375.0, 206250.0, 206250.0]


def test_reference_vehicles_print_no_cap_line_at_0p025(caplines):
    """Reference car preset and the server generic car (``build_cfg``): no
    sfl-cap line at dt 0.025 (beta 1.169 <= 1.5, per-wheel c < 1)."""
    from genesis_vehicle.server import vehicle_builder
    mapping = {"driveType": 3, "drivetrainStrategy": 1, "maxTorque": 1000.0, "maxBrake": 2500.0,
               "drivingJoints": [{"jointName": "rear_left_wheel_joint"}, {"jointName": "rear_right_wheel_joint"}],
               "steeringJoints": [{"jointName": "front_left_steer_joint"}, {"jointName": "front_right_steer_joint"}]}
    vehicle_builder._cfg_cache.clear()
    cfg = vehicle_builder.build_cfg(CAR_URDF, mapping, 1.0, target_id=0)
    for c in (None, cfg):
        vs, veh = _scene(CAR_URDF, 0.025, cfg=c)
        veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        for _ in range(20):
            vs.step()
        veh.set_inputs(throttle=0.0, brake=0.3, steer=0.0)
        for _ in range(200):
            vs.step()
        assert abs(veh.physics.wheel_meta.static_rigid_factor.beta - 1.168545) <= 1e-4 * 1.168545
    assert caplines == []
