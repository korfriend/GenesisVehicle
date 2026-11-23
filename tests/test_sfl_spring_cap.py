"""2.0.4 Step K: the per-wheel StaticFrictionLock spring cap (Genesis-free).

``k_eff_i = min(k_spring, SPRING_C_MAX * m_static_i / dt^2)``, ``SPRING_C_MAX``
1.0, ``m_static`` from elastic-support statics
(``dynamics.static_mass_shares``), stored on ``wheel_meta.static_mass_share``.

| abbr | meaning |
|---|---|
| SFL | `StaticFrictionLock` (`strategies/stability.py`) |
| k_eff | the capped per-wheel lock stiffness, N/m |
| m_static | per-wheel static mass share, kg (`WheelMeta.static_mass_share`) |
| c | explicit-spring number `k * dt^2 / m_static` |
| COM | centre of mass |

The reference-car mass properties below were read with
``control.plant.plant_mass_properties`` off a built ``car_4w.urdf`` entity
(genesis-world 1.4.5, CPU): mass 1330.0 kg, COM x 0.010150376 m, y 0. The
reference tank (``tests/data/tank_ref.urdf``): 58700.0 kg, COM x 0.010562 m.
"""

from __future__ import annotations

import logging
import os
import types

import pytest
import torch

from genesis_vehicle import dynamics
from genesis_vehicle.config import resolve
from genesis_vehicle.core import WheelMeta, build_wheel_meta
from genesis_vehicle.presets import car_4w_rwd_ackermann, tank_skid_belt
from genesis_vehicle.strategies.stability import StaticFrictionLock

HERE = os.path.dirname(__file__)
CAR_URDF = os.path.abspath(os.path.join(HERE, "..", "samples", "urdf", "car_4w.urdf"))
TANK_URDF = os.path.abspath(os.path.join(HERE, "data", "tank_ref.urdf"))
DEV, FDT = torch.device("cpu"), torch.float32

CAR_MASS, CAR_COM = 1330.0, (0.010150376119111714, 0.0, 0.48045112898475245)
TANK_MASS, TANK_COM = 58700.0, (0.010562181878942851, 0.0, 0.6916524569870462)


def _wm(preset, urdf):
    return build_wheel_meta(resolve(preset(urdf, stability="control")), DEV, FDT)


class _Meta:
    """A weak-referenceable duck-typed wheel meta for direct hook calls."""

    def __init__(self, share, n=None):
        n = n if n is not None else (len(share) if share is not None else 4)
        self.n_wheels = n
        self.mu_long = torch.ones(n)
        self.mu_lat = torch.ones(n)
        self.static_mass_share = (None if share is None
                                  else torch.tensor(share, dtype=FDT))


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


# --- C1: test_static_mass_shares (four cases) ---------------------------------

def test_static_mass_shares_reference_car():
    wm = _wm(car_4w_rwd_ackermann, CAR_URDF)
    res = dynamics.static_mass_shares(CAR_MASS, CAR_COM, wm.positions, wm.k_susp)
    assert res.fallback is None
    assert [round(float(v), 1) for v in res.shares] == [335.0, 335.0, 330.0, 330.0]
    assert float(res.shares.sum()) == pytest.approx(CAR_MASS, rel=1e-12)


def test_static_mass_shares_ten_wheel_tank_sums_to_mass():
    wm = _wm(tank_skid_belt, TANK_URDF)
    assert wm.n_wheels == 10
    res = dynamics.static_mass_shares(TANK_MASS, TANK_COM, wm.positions, wm.k_susp)
    assert res.fallback is None
    assert float(res.shares.sum()) == pytest.approx(TANK_MASS, rel=1e-12)
    assert (res.shares > 0).all()
    # COM slightly forward -> front wheels carry more, linear in x.
    assert float(res.shares[0]) > float(res.shares[4])


def test_static_mass_shares_com_outside_wheelbase_falls_back_and_k_eff_positive():
    """A COM 3 m ahead of a 2.7 m wheelbase: the elastic statics give NEGATIVE
    rear shares (an anti-spring). The guard must return equal shares and the
    lock's k_eff must be positive on every wheel."""
    wm = _wm(car_4w_rwd_ackermann, CAR_URDF)
    res = dynamics.static_mass_shares(CAR_MASS, (3.0, 0.0, 0.5), wm.positions, wm.k_susp)
    assert res.fallback is not None
    assert [float(v) for v in res.shares] == [CAR_MASS / 4] * 4
    meta = _Meta([float(v) for v in res.shares])
    k_eff = StaticFrictionLock()._k_eff(meta, 0.05, DEV, FDT)   # dt where it caps
    t = torch.as_tensor(k_eff)
    assert bool((t > 0).all())


def test_static_mass_shares_zero_rate_wheel():
    wm = _wm(car_4w_rwd_ackermann, CAR_URDF)
    k = wm.k_susp.clone()
    k[2] = 0.0
    res = dynamics.static_mass_shares(CAR_MASS, CAR_COM, wm.positions, k)
    import numpy as np
    shares = np.asarray(res.shares)
    assert np.isfinite(shares).all() and (shares >= 0).all()
    assert res.fallback is not None          # its share is 0 <= 0.05 M/n
    assert float(shares.sum()) == pytest.approx(CAR_MASS)


def test_static_mass_shares_rejects_a_massless_or_nan_vehicle():
    wm = _wm(car_4w_rwd_ackermann, CAR_URDF)
    for m in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            dynamics.static_mass_shares(m, CAR_COM, wm.positions, wm.k_susp)


def test_static_mass_shares_non_finite_com_falls_back():
    wm = _wm(car_4w_rwd_ackermann, CAR_URDF)
    res = dynamics.static_mass_shares(CAR_MASS, (float("nan"), 0.0, 0.0),
                                      wm.positions, wm.k_susp)
    assert res.fallback is not None
    assert [float(v) for v in res.shares] == [CAR_MASS / 4] * 4


# --- the cap value (M18), its inactivity, and its inputs ----------------------

def test_sfl_cap_value():
    """66 kg at dt 0.025: k_eff = 1.0 * 66 / 0.025^2 = 105 600 N/m (M18: a
    c_max of 2.0 would give 211 200)."""
    assert StaticFrictionLock.SPRING_C_MAX == 1.0
    k = StaticFrictionLock()._k_eff(_Meta([66.0, 66.0, 335.0, 335.0]), 0.025, DEV, FDT)
    assert torch.is_tensor(k) and k.shape == (1, 4) and k.dtype == FDT
    assert [float(v) for v in k[0]] == [105600.0, 105600.0, 500000.0, 500000.0]


def test_uncapped_is_the_k_spring_object_itself():
    """No wheel capped -> the lock reads exactly the 2.0.3 python scalar."""
    hook = StaticFrictionLock()
    for meta in (_Meta([335.0, 335.0, 330.0, 330.0]),       # reference car, c 0.947
                 _Meta(None),                               # no static share
                 types.SimpleNamespace(n_wheels=4)):        # duck type without the field
        assert hook._k_eff(meta, 0.025, DEV, FDT) is hook.k_spring
    # dt unknown / non-positive: uncapped.
    assert hook._k_eff(_Meta([66.0] * 4), None, DEV, FDT) is hook.k_spring
    assert hook._k_eff(_Meta([66.0] * 4), 0.0, DEV, FDT) is hook.k_spring


def test_reference_car_threshold_dt():
    """The reference car's rear share (330 kg) caps above dt = sqrt(330/5e5)."""
    meta = _Meta([335.0, 335.0, 330.0, 330.0])
    hook = StaticFrictionLock()
    assert hook._k_eff(meta, 0.0256, DEV, FDT) is hook.k_spring
    k = hook._k_eff(meta, 0.0258, DEV, FDT)
    assert torch.is_tensor(k) and float(k[0, 2]) < 500000.0


def test_k_spring_write_after_derive_lands_on_next_call():
    """HOT_DEPENDENTS gate for ("hook", "k_spring") -> "_k_eff": derive first,
    then write the source, then the next call moves."""
    from genesis_vehicle._hotset import HOT_DEPENDENTS, derived_value
    assert HOT_DEPENDENTS[("hook", "k_spring")] == ("_k_eff",)
    meta = _Meta([66.0, 66.0, 335.0, 335.0])
    hook = StaticFrictionLock()
    k1 = hook._k_eff(meta, 0.025, DEV, FDT)
    assert derived_value(hook, "_k_eff") is k1
    hook.k_spring = 100_000.0                     # below every cap now
    assert hook._k_eff(meta, 0.025, DEV, FDT) == 100_000.0
    hook.k_spring = 1_000_000.0
    k3 = hook._k_eff(meta, 0.025, DEV, FDT)
    assert [float(v) for v in k3[0]] == [105600.0, 105600.0, 536000.0, 536000.0]


def test_a_different_wheel_meta_or_dt_re_derives():
    hook = StaticFrictionLock()
    a = hook._k_eff(_Meta([66.0] * 4), 0.025, DEV, FDT)
    b = hook._k_eff(_Meta([100.0] * 4), 0.025, DEV, FDT)
    assert float(a[0, 0]) == 105600.0 and float(b[0, 0]) == 160000.0
    c = hook._k_eff(_Meta([100.0] * 4), 0.02, DEV, FDT)
    assert float(c[0, 0]) == 250000.0


def test_same_meta_new_share_value_re_derives():
    """The static share is keyed by VALUE (never by id())."""
    meta = _Meta([66.0] * 4)
    hook = StaticFrictionLock()
    assert float(hook._k_eff(meta, 0.025, DEV, FDT)[0, 0]) == 105600.0
    meta.static_mass_share = torch.tensor([100.0] * 4)
    assert float(hook._k_eff(meta, 0.025, DEV, FDT)[0, 0]) == 160000.0


def test_a_bad_static_share_is_asserted():
    for bad in ([66.0, -1.0, 66.0, 66.0], [66.0, float("nan"), 66.0, 66.0], [0.0] * 4):
        with pytest.raises(AssertionError):
            StaticFrictionLock()._k_eff(_Meta(bad), 0.025, DEV, FDT)


def test_wheel_meta_field_is_appended_last_and_defaults_none():
    import dataclasses
    names = [f.name for f in dataclasses.fields(WheelMeta)]
    # (Step D) static_rigid_factor is appended after static_mass_share.
    assert names[-3:] == ["yaw_inertia", "static_mass_share", "static_rigid_factor"]
    wm = _wm(car_4w_rwd_ackermann, CAR_URDF)
    assert wm.static_mass_share is None and wm.static_rigid_factor is None


# --- the report (M13 / M16, api-review condition 10) --------------------------

def test_sfl_cap_report(caplines):
    """Static share 66 kg at dt 0.025 reports ONCE per (meta, k_spring, dt),
    also across a second (copied) hook on the same meta; 335 kg is silent; a
    ctx whose N is 0 on the first engagement changes neither k_eff nor the
    report (the cap is computed from the static share, never from N)."""
    import copy
    meta = _Meta([66.0, 66.0, 335.0, 335.0])
    hook = StaticFrictionLock()
    hook._k_eff(meta, 0.025, DEV, FDT)
    hook._k_eff(meta, 0.025, DEV, FDT)
    copy.deepcopy(hook)._k_eff(meta, 0.025, DEV, FDT)     # a plant prediction copy
    assert len(caplines) == 1
    assert "105600" in caplines[0] and "c_max 1.0" in caplines[0]
    assert "dt 0.025" in caplines[0] and "6.1 mm" in caplines[0]
    StaticFrictionLock()._k_eff(_Meta([335.0, 335.0, 330.0, 330.0]), 0.025, DEV, FDT)
    assert len(caplines) == 1
    # A different dt on the same meta is a new report key.
    hook._k_eff(meta, 0.04, DEV, FDT)
    assert len(caplines) == 2

    # Through apply_post_tire with N = 0 (a spawn drop): k_eff from the share.
    meta2 = _Meta([66.0] * 4)
    h2 = StaticFrictionLock()
    ctx = _ctx(meta2, n=4, v=0.01, N=0.0)
    h2.apply_post_tire(ctx)
    from genesis_vehicle._hotset import derived_value
    assert float(derived_value(h2, "_k_eff")[0, 0]) == 105600.0
    assert len(caplines) == 3


def _ctx(meta, n, v, N, brake=1.0, dt=0.025):
    """A held wheel (brake_rem > 0) whose deadbeat binds (F_long_deadbeat 0):
    the lock engages on the first call and takes over through the creep test
    on the second (v_prev is +inf before the first)."""
    return types.SimpleNamespace(
        brake=torch.tensor([brake]), v_long=torch.full((1, n), v),
        v_lat=torch.zeros(1, n), N=torch.full((1, n), N), dt=dt,
        wheel_meta=meta, F_long=torch.zeros(1, n), F_lat=torch.zeros(1, n),
        omega_override=None, brake_rem=torch.full((1, n), 100.0),
        F_long_deadbeat=torch.zeros(1, n))


def test_damper_floor_uses_k_eff():
    """M17: the damper floor is 2*sqrt(k_eff * N / g). On the creep takeover
    step at 0.01 m/s the anchor is at the contact, so the
    lock's force is the bounded damper alone (engage, then the creep
    takeover). N = 9 x the static load: with
    k_eff 105 600 the floor (15 840 N*s/m) is under the one-step bound
    (N/g/dt = 23 763 N*s/m); with the raw 500 000 it would be clipped there."""
    meta = _Meta([66.0] * 4)
    N = 9 * 66.0 * 9.81
    hook = StaticFrictionLock(k_damp=0.0)     # the floor term alone
    hook.apply_post_tire(_ctx(meta, n=4, v=0.01, N=N))   # engage
    ctx = _ctx(meta, n=4, v=0.01, N=N)
    hook.apply_post_tire(ctx)                             # creep takeover, d = 0
    D = 2.0 * (105600.0 * N / 9.81) ** 0.5
    assert float(ctx.F_long[0, 0]) == pytest.approx(-D * 0.01, rel=1e-5)
    raw_bound = (N / 9.81) * 0.01 / 0.025
    assert abs(float(ctx.F_long[0, 0])) < 0.75 * raw_bound


def test_spring_force_and_anchor_use_k_eff():
    """The spring term and the slip / bound anchor re-placement read k_eff."""
    meta = _Meta([66.0] * 4)
    N = 66.0 * 9.81
    hook = StaticFrictionLock(k_damp=0.0)
    hook.apply_post_tire(_ctx(meta, n=4, v=0.01, N=N))   # engage
    ctx = _ctx(meta, n=4, v=0.01, N=N)
    hook.apply_post_tire(ctx)                 # creep takeover step: d = 0
    # Keep moving forward (a stop or reversal would re-anchor), at 1e-6 m/s
    # so the damper term (~5e-3 N) is negligible next to the spring.
    hook._d_long = torch.full((1, 4), 0.001)  # 1 mm from the anchor
    ctx = _ctx(meta, n=4, v=1e-6, N=N)
    hook.apply_post_tire(ctx)
    assert float(ctx.F_long[0, 0]) == pytest.approx(-105600.0 * 0.001, rel=1e-3)
    # Slip: displacement far beyond mu*N/k_eff -> anchor re-placed at mu*N/k_eff.
    hook._d_long = torch.full((1, 4), 0.1)
    hook.apply_post_tire(_ctx(meta, n=4, v=1e-6, N=N))
    assert float(hook._d_long[0, 0]) == pytest.approx(N / 105600.0, rel=1e-3)


def test_hook_tensor_attributes_stay_state_attrs_after_a_capped_step():
    meta = _Meta([66.0] * 4)
    hook = StaticFrictionLock()
    hook.apply_post_tire(_ctx(meta, n=4, v=0.01, N=600.0))
    tensors = {k for k, v in hook.__dict__.items() if torch.is_tensor(v)}
    assert tensors == set(StaticFrictionLock.STATE_ATTRS)


def test_k_eff_dt_comes_from_ctx_only():
    """api-review condition 5: the only dt the cap reads is ctx.dt."""
    import inspect
    from genesis_vehicle.strategies import stability
    src = inspect.getsource(stability.StaticFrictionLock.apply_post_tire)
    assert "self._k_eff(ctx.wheel_meta, dt," in src and "dt = ctx.dt" in src
    meta = _Meta([66.0] * 4)
    hook = StaticFrictionLock()
    hook.apply_post_tire(_ctx(meta, n=4, v=0.01, N=600.0, dt=0.02))
    from genesis_vehicle._hotset import derived_value
    assert float(derived_value(hook, "_k_eff")[0, 0]) == pytest.approx(66.0 / 0.0004)


def test_the_cache_key_holds_the_share_tensor_not_its_id():
    """api-review condition 4: the share enters the cache key as the TENSOR
    (compared with ``torch.equal`` when another tensor is assigned), never as
    ``id()``. Deterministic check: the key keeps the old share ALIVE after the
    meta drops it — an ``id()`` key would let it be freed (and its id reused
    by a new share of another value, which would then be served the stale
    k_eff)."""
    import gc
    import weakref
    meta = _Meta([66.0] * 4)
    hook = StaticFrictionLock()
    assert float(hook._k_eff(meta, 0.025, DEV, FDT)[0, 0]) == 105600.0
    old = weakref.ref(meta.static_mass_share)
    meta.static_mass_share = None
    gc.collect()
    assert old() is not None, "the cache key must hold the share tensor itself"
    meta.static_mass_share = torch.tensor([100.0] * 4)
    assert float(hook._k_eff(meta, 0.025, DEV, FDT)[0, 0]) == 160000.0


def test_an_in_place_edit_of_the_share_re_derives():
    """The key carries the share's version counter, so an in-place edit of the
    SAME tensor re-derives (an inference-mode tensor has no counter)."""
    meta = _Meta([66.0] * 4)
    hook = StaticFrictionLock()
    assert float(hook._k_eff(meta, 0.025, DEV, FDT)[0, 0]) == 105600.0
    meta.static_mass_share.fill_(100.0)
    assert float(hook._k_eff(meta, 0.025, DEV, FDT)[0, 0]) == 160000.0
    meta.static_mass_share[0] = 66.0
    assert float(hook._k_eff(meta, 0.025, DEV, FDT)[0, 0]) == 105600.0
