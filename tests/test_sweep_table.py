"""SweepTable unit tests — numpy-only, no Genesis needed.

A synthetic full-factorial table with a KNOWN linear response
    a       = 3*throttle - 0.5*v - 0.05*pitch
    omega_z = 2*steer
lets us assert that the inverse lookups recover the exact input (linear
surface -> multilinear interpolation is exact; bisection xtol is 1e-3).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control import SweepTable, SWEEP_COLUMNS


V_GRID = [-4.0, -2.0, 0.0, 2.0, 4.0]
TS_GRID = [-1.0, -0.5, 0.0, 0.5, 1.0]
PITCH_GRID = [-10.0, 0.0, 10.0]
ROLL_GRID = [-10.0, 0.0, 10.0]


def _a_model(thr, v, pitch):
    return 3.0 * thr - 0.5 * v - 0.05 * pitch


def _omega_model(steer):
    return 2.0 * steer


def make_table() -> SweepTable:
    v, t, s, p, r = np.meshgrid(V_GRID, TS_GRID, TS_GRID, PITCH_GRID, ROLL_GRID,
                                indexing="ij")
    cols = {"v_init": v.ravel(), "throttle": t.ravel(), "steer": s.ravel(),
            "pitch": p.ravel(), "roll": r.ravel()}
    cols["a_measured"] = _a_model(cols["throttle"], cols["v_init"], cols["pitch"])
    cols["omega_z_measured"] = _omega_model(cols["steer"])
    return SweepTable(cols)


@pytest.fixture(scope="module")
def table():
    return make_table()


def test_throttle_inverse_recovers_input(table):
    for thr_true in (0.5, -0.25, 0.8):
        for v in (0.0, 1.0, -2.0):
            a = _a_model(thr_true, v, 0.0)
            thr = table.throttle_for(v, a, 0.0, 0.0)
            assert abs(thr - thr_true) < 2e-3, (thr_true, v, thr)


def test_steer_inverse_recovers_input(table):
    for steer_true in (0.3, -0.7, 0.0):
        steer = table.steer_for(1.0, _omega_model(steer_true), 0.0, 0.0)
        assert abs(steer - steer_true) < 2e-3


def test_pitch_dependence_and_clamp(table):
    # On an uphill (positive pitch reduces a), more throttle is needed for
    # the same acceleration.
    thr_flat = table.throttle_for(0.0, 1.0, 0.0, 0.0)
    thr_up = table.throttle_for(0.0, 1.0, 10.0, 0.0)
    assert thr_up > thr_flat
    # Out-of-grid pitch clamps to the boundary — same answer as at the edge.
    assert table.throttle_for(0.0, 1.0, 50.0, 0.0) == pytest.approx(
        table.throttle_for(0.0, 1.0, 10.0, 0.0), abs=1e-9)


def test_unachievable_target_saturates(table):
    # a beyond the surface range -> endpoint throttle.
    assert table.throttle_for(0.0, 100.0, 0.0, 0.0) == 1.0
    assert table.throttle_for(0.0, -100.0, 0.0, 0.0) == -1.0


def test_csv_round_trip(table, tmp_path):
    p = tmp_path / "sweep.csv"
    table.save(p)
    loaded = SweepTable.load(p)
    assert len(loaded) == len(table)
    for c in SWEEP_COLUMNS:
        np.testing.assert_allclose(loaded.columns[c], table.columns[c],
                                   atol=1e-4)
    assert loaded.throttle_for(1.0, 1.0, 0.0, 0.0) == pytest.approx(
        table.throttle_for(1.0, 1.0, 0.0, 0.0), abs=2e-3)


def test_missing_grid_cell_raises():
    t = make_table()
    # Drop one row that belongs to the throttle-lookup subset (steer == 0) —
    # a hole in a lookup's factorial grid must be a hard error.
    drop = int(np.flatnonzero(t.columns["steer"] == 0.0)[0])
    keep = np.ones(len(t), dtype=bool)
    keep[drop] = False
    cols = {c: a[keep] for c, a in t.columns.items()}
    with pytest.raises(ValueError, match="full factorial"):
        SweepTable(cols)


def test_requires_zero_levels():
    t = make_table()
    mask = t.columns["steer"] != 0.0   # no steer==0 rows left
    cols = {c: a[mask] for c, a in t.columns.items()}
    with pytest.raises(ValueError, match="steer == 0"):
        SweepTable(cols)


def test_missing_column_raises():
    t = make_table()
    cols = dict(t.columns)
    del cols["a_measured"]
    with pytest.raises(ValueError, match="missing columns"):
        SweepTable(cols)
