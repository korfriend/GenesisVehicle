"""``eps_v`` is LIVE on both tire models — G5a / G5b.

This file is a CHARACTERISATION baseline, not a red-first test. Before the
build-time hoist, ``CoulombIsotropic.__call__`` recomputed ``eps_v * eps_v`` on
every call, so every number below was already produced by the tree that had no
hoist at all. That is the point: the hoist is a change of WHEN the square is
computed, and the only way to catch it silently freezing at build time is to
have pinned the numbers BEFORE it landed. Asserting instead that ``_eps2`` is a
tensor would prove the hoist happened and say nothing about the values.

Why here and not somewhere existing: ``tire_models/coulomb.py`` had NO
behavioural test in the tree (``grep -rin coulomb tests/`` found exactly one
hit, an import in ``_check_import.py``), so the Coulomb arm has no baseline to
regress against and must create its own.

The two arms move in OPPOSITE directions as ``eps_v`` falls — Pacejka's
``F_long`` drops, Coulomb's rises — so a copy-paste of one arm's expected value
into the other fails immediately.

| abbr | meaning |
|---|---|
| eps_v | velocity floor (m/s) under which slip ratio stops being computed from |v_long| |
| kappa | longitudinal slip ratio |
| N | wheel normal load (N) |
"""

from __future__ import annotations

import os

import pytest
import torch

from genesis_vehicle.config import resolve
from genesis_vehicle.core import build_wheel_meta
from genesis_vehicle.presets import car_4w_rwd_ackermann
from genesis_vehicle.tire_models import CoulombIsotropic, PacejkaAnisotropic

CAR_URDF = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "samples", "urdf", "car_4w.urdf"))

# Conditions for every number in this file (v10 plan §0): CPU, n_envs=1,
# the reference car's wheel_meta, v_long=0.05, v_lat=0.02, v_roll=0.30, N=3000.
V_LONG, V_LAT, V_ROLL, N_LOAD = 0.05, 0.02, 0.30, 3000.0

PACEJKA_F_LONG = {0.5: 2529.112, 0.05: 1482.747}    # G5a
COULOMB_F_LONG = {0.5: 1500.000, 0.05: 2990.446}    # G5b
KAPPA = {0.5: 0.500, 0.05: 5.000}


@pytest.fixture(scope="module")
def wm():
    resolved = resolve(car_4w_rwd_ackermann(CAR_URDF, stability="control"))
    return build_wheel_meta(resolved, torch.device("cpu"), torch.float32)


def _inputs(n_wheels: int):
    f = lambda v: torch.full((1, n_wheels), v, dtype=torch.float32)
    return f(V_LONG), f(V_LAT), f(V_ROLL), f(N_LOAD)


def _f_long(model, wm) -> tuple[float, float]:
    F_long, _F_lat, kappa, _alpha = model(*_inputs(wm.n_wheels), wm)
    return float(F_long[0, 0]), float(kappa[0, 0])


# --- G5a: Pacejka reads eps_v directly (no derived square) --------------------

@pytest.mark.parametrize("eps_v", [0.5, 0.05])
def test_g5a_pacejka_eps_v_fresh_model(eps_v, wm):
    f, k = _f_long(PacejkaAnisotropic(eps_v=eps_v), wm)
    assert f == pytest.approx(PACEJKA_F_LONG[eps_v], abs=1e-3)
    assert k == pytest.approx(KAPPA[eps_v], abs=1e-3)


def test_g5a_pacejka_eps_v_is_live_after_construction(wm):
    """The MUTATION path: write eps_v on a live model, no rebuild."""
    tire = PacejkaAnisotropic(eps_v=0.5)
    assert _f_long(tire, wm)[0] == pytest.approx(PACEJKA_F_LONG[0.5], abs=1e-3)
    tire.eps_v = 0.05
    assert _f_long(tire, wm)[0] == pytest.approx(PACEJKA_F_LONG[0.05], abs=1e-3)


# --- G5b: Coulomb — this is the arm that observes the `_eps2` hoist -----------

@pytest.mark.parametrize("eps_v", [0.5, 0.05])
def test_g5b_coulomb_eps_v_fresh_model(eps_v, wm):
    f, k = _f_long(CoulombIsotropic(eps_v=eps_v), wm)
    assert f == pytest.approx(COULOMB_F_LONG[eps_v], abs=1e-3)
    assert k == pytest.approx(KAPPA[eps_v], abs=1e-3)


def test_g5b_coulomb_eps_v_is_live_after_the_square_was_derived(wm):
    """The gate for the hoist. Call once (which derives and caches eps_v**2),
    then write eps_v — the next call must use the NEW square. A build-time
    snapshot without the dependency would return 1500.000 twice."""
    tire = CoulombIsotropic(eps_v=0.5)
    assert _f_long(tire, wm)[0] == pytest.approx(COULOMB_F_LONG[0.5], abs=1e-3)
    tire.eps_v = 0.05
    assert _f_long(tire, wm)[0] == pytest.approx(COULOMB_F_LONG[0.05], abs=1e-3)


def test_g5b_coulomb_eps_v_survives_a_resolved_config_round_trip(wm):
    """The shape a user actually reaches it in: `resolved.tire.eps_v = x`."""
    resolved = resolve(car_4w_rwd_ackermann(CAR_URDF, stability="control"))
    resolved.tire = CoulombIsotropic(eps_v=0.5)
    assert _f_long(resolved.tire, wm)[0] == pytest.approx(
        COULOMB_F_LONG[0.5], abs=1e-3)
    resolved.tire.eps_v = 0.05
    assert _f_long(resolved.tire, wm)[0] == pytest.approx(
        COULOMB_F_LONG[0.05], abs=1e-3)


def test_the_two_arms_do_not_share_an_expected_value():
    """Guard on the file itself: the arms move in opposite directions, so a
    copy-paste between them is a test failure and not a silent pass."""
    assert PACEJKA_F_LONG[0.5] > PACEJKA_F_LONG[0.05]
    assert COULOMB_F_LONG[0.5] < COULOMB_F_LONG[0.05]


# --- tensor acceptance (what STEP 3 needs; asserted, not inspected) ----------

def test_coulomb_accepts_a_per_row_eps_v(wm):
    """`coulomb.py`'s two eps_v sites under a per-ROW tensor.

    The old `torch.full_like(sq, eps2)` raised `TypeError: fill_value must be
    Number, not Tensor` here; `torch.clamp(..., min=self.eps_v)` takes a tensor
    min. Two rows carrying the SAME value must reproduce the scalar answer
    exactly, and two rows carrying DIFFERENT values must give each row its own
    answer — which a python-scalar square could not do."""
    v_long, v_lat, v_roll, N = (t.expand(2, wm.n_wheels).contiguous()
                                for t in _inputs(wm.n_wheels))
    tire = CoulombIsotropic(eps_v=0.5)

    tire.eps_v = torch.tensor([[0.5], [0.5]])
    F_long, _, _, _ = tire(v_long, v_lat, v_roll, N, wm)
    assert float(F_long[0, 0]) == pytest.approx(COULOMB_F_LONG[0.5], abs=1e-3)
    assert float(F_long[1, 0]) == pytest.approx(COULOMB_F_LONG[0.5], abs=1e-3)

    tire.eps_v = torch.tensor([[0.5], [0.05]])
    F_long, _, _, _ = tire(v_long, v_lat, v_roll, N, wm)
    assert float(F_long[0, 0]) == pytest.approx(COULOMB_F_LONG[0.5], abs=1e-3)
    assert float(F_long[1, 0]) == pytest.approx(COULOMB_F_LONG[0.05], abs=1e-3)


def test_pacejka_accepts_a_per_row_eps_v(wm):
    """`pacejka.py`'s `torch.clamp(torch.abs(v_long), min=self.eps_v)` — there
    is no derived square on this model, only the tensor `min`."""
    v_long, v_lat, v_roll, N = (t.expand(2, wm.n_wheels).contiguous()
                                for t in _inputs(wm.n_wheels))
    # Assigned, not constructed: `__init__` still does `float(eps_v)` on both
    # models (`pacejka.py:33`, `coulomb.py:21`), so a promoted per-row source
    # arrives by ASSIGNMENT onto the built object. Widening the constructors is
    # STEP 3's promotion work, not this change's — the site named here is the
    # clamp at `pacejka.py:45`.
    tire = PacejkaAnisotropic(eps_v=0.5)
    tire.eps_v = torch.tensor([[0.5], [0.05]])
    F_long, _, _, _ = tire(v_long, v_lat, v_roll, N, wm)
    assert float(F_long[0, 0]) == pytest.approx(PACEJKA_F_LONG[0.5], abs=1e-3)
    assert float(F_long[1, 0]) == pytest.approx(PACEJKA_F_LONG[0.05], abs=1e-3)


def test_a_reassigned_eps_v_tensor_is_not_mistaken_for_the_cached_one(wm):
    """The derived-cache key holds the source. Comparing two DIFFERENT tensors
    with `==` inside a tuple raises "Boolean value of Tensor ... is ambiguous";
    comparing them by identity alone would return a stale square. Neither is
    acceptable, so the key compares tensors with torch.equal."""
    v_long, v_lat, v_roll, N = (t.expand(2, wm.n_wheels).contiguous()
                                for t in _inputs(wm.n_wheels))
    tire = CoulombIsotropic(eps_v=0.5)
    tire.eps_v = torch.tensor([[0.5], [0.5]])
    tire(v_long, v_lat, v_roll, N, wm)
    tire.eps_v = torch.tensor([[0.05], [0.05]])          # different object
    F_long, _, _, _ = tire(v_long, v_lat, v_roll, N, wm)
    assert float(F_long[0, 0]) == pytest.approx(COULOMB_F_LONG[0.05], abs=1e-3)
