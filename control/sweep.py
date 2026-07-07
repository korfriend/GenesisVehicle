"""Sweep-table container + inverse lookup for path following.

A *sweep table* is the measured open-loop response of one specific vehicle
(URDF + preset + config overrides) on flat ground with rotated gravity
(emulating pitch/roll slopes):

    (v_init, throttle, steer, pitch, roll) -> (a_measured, omega_z_measured)

``SweepTable`` loads that grid from CSV (or takes it as arrays) and answers
the inverse question the controller needs each step:

    throttle_for(v, a_target, pitch, roll)   -> throttle in [-1, 1]
    steer_for(v, omega_target, pitch, roll)  -> steer in [-1, 1]

Implementation is numpy-only (no scipy/pandas): 4-D multilinear
interpolation on the rectilinear grid + bisection for the inverse.
Out-of-grid queries are clamped to the grid boundary — extrapolating a
measured response surface is less trustworthy than saturating it.

Measure a table for a new vehicle with
``python -m genesis_vehicle.control.sweep_measure`` (needs Genesis); this
module itself has no Genesis/torch dependency, so a game/runtime process
can use a pre-measured CSV without a simulator installed.

The table is only valid for the exact (URDF, preset, config-override)
triple it was measured with — re-measure when any of the three changes.
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np


#: CSV schema, in column order. ``pitch``/``roll`` are degrees; ``v_init``
#: and ``a_measured`` are m/s and m/s²; ``omega_z_measured`` is rad/s.
SWEEP_COLUMNS = ("v_init", "throttle", "steer", "pitch", "roll",
                 "a_measured", "omega_z_measured")


def _build_grid(cols: dict, in_cols: Sequence[str], out_col: str):
    """Reshape flat rows into (axes, values) on the full factorial grid.

    Every (in_cols) combination must appear at least once (duplicates: the
    last row wins, matching pandas-era behaviour). A hole in the grid is a
    hard error — silent zeros would corrupt the interpolation.
    """
    axes = [np.unique(np.asarray(cols[c], dtype=np.float64)) for c in in_cols]
    grid = np.full(tuple(len(a) for a in axes), np.nan, dtype=np.float64)
    idx = tuple(
        np.searchsorted(axes[i], np.asarray(cols[c], dtype=np.float64))
        for i, c in enumerate(in_cols)
    )
    grid[idx] = np.asarray(cols[out_col], dtype=np.float64)
    if np.isnan(grid).any():
        n_missing = int(np.isnan(grid).sum())
        raise ValueError(
            f"sweep table is not a full factorial grid over {tuple(in_cols)}: "
            f"{n_missing}/{grid.size} cells missing")
    return axes, grid


class _Interp4D:
    """Multilinear interpolation on a 4-D rectilinear grid, clamped at the
    boundary in every dimension (no extrapolation)."""

    def __init__(self, axes, values):
        self.axes = [np.asarray(a, dtype=np.float64) for a in axes]
        self.values = np.asarray(values, dtype=np.float64)

    def __call__(self, *pt: float) -> float:
        lo_idx = []
        frac = []
        for ax, val in zip(self.axes, pt):
            n = len(ax)
            if n == 1:                        # degenerate axis
                lo_idx.append(0)
                frac.append(0.0)
                continue
            v = min(max(float(val), float(ax[0])), float(ax[-1]))
            i = int(np.searchsorted(ax, v, side="right")) - 1
            i = min(max(i, 0), n - 2)
            lo_idx.append(i)
            frac.append((v - float(ax[i])) / (float(ax[i + 1]) - float(ax[i])))
        out = 0.0
        for corner in range(16):              # 2^4 corners
            w = 1.0
            idx = []
            for d in range(4):
                bit = (corner >> d) & 1
                if len(self.axes[d]) == 1:
                    if bit:
                        w = 0.0
                        break
                    idx.append(0)
                else:
                    idx.append(lo_idx[d] + bit)
                    w *= frac[d] if bit else (1.0 - frac[d])
            if w > 0.0:
                out += w * float(self.values[tuple(idx)])
        return out


class _InverseLookup:
    """(v, target, pitch, roll) -> x_in (throttle or steer) by bisection on
    the interpolated forward map x_in -> response."""

    XTOL = 1e-3

    def __init__(self, cols: dict, in_col: str, out_col: str):
        axes, grid = _build_grid(cols, ("v_init", in_col, "pitch", "roll"),
                                 out_col)
        self._interp = _Interp4D(axes, grid)
        self.in_min = float(axes[1][0])
        self.in_max = float(axes[1][-1])

    def forward(self, v: float, x: float, pitch: float, roll: float) -> float:
        return self._interp(v, x, pitch, roll)

    def inverse(self, v: float, target: float, pitch: float, roll: float) -> float:
        lo, hi = self.in_min, self.in_max
        f_lo = self.forward(v, lo, pitch, roll) - target
        f_hi = self.forward(v, hi, pitch, roll) - target
        if f_lo * f_hi > 0.0:
            # Target outside the achievable range — saturate to the nearer end.
            return lo if abs(f_lo) < abs(f_hi) else hi
        while hi - lo > self.XTOL:
            mid = 0.5 * (lo + hi)
            f_mid = self.forward(v, mid, pitch, roll) - target
            if f_mid == 0.0:
                return mid
            if (f_lo < 0.0) != (f_mid < 0.0):
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        return 0.5 * (lo + hi)


class SweepTable:
    """Measured (input -> response) grid of ONE vehicle, with inverse lookup.

    Construct from arrays (``SweepTable(columns)`` where ``columns`` maps
    each name in ``SWEEP_COLUMNS`` to a same-length 1-D array) or from a
    CSV produced by ``sweep_measure`` (``SweepTable.load(path)``).

    The throttle inverse is built from the ``steer == 0`` rows and the steer
    inverse from the ``throttle == 0`` rows, so both zero levels must be in
    the measured grid (the default measurement grid includes them).
    """

    def __init__(self, columns: dict):
        missing = [c for c in SWEEP_COLUMNS if c not in columns]
        if missing:
            raise ValueError(f"sweep table missing columns: {missing}")
        n = len(np.asarray(columns[SWEEP_COLUMNS[0]]))
        self.columns = {}
        for c in SWEEP_COLUMNS:
            arr = np.asarray(columns[c], dtype=np.float64).ravel()
            if len(arr) != n:
                raise ValueError(f"column {c!r} length {len(arr)} != {n}")
            self.columns[c] = arr

        thr_rows = self.columns["steer"] == 0.0
        str_rows = self.columns["throttle"] == 0.0
        if not thr_rows.any() or not str_rows.any():
            raise ValueError(
                "sweep table needs steer == 0 rows (throttle lookup) and "
                "throttle == 0 rows (steer lookup)")
        self._throttle_lu = _InverseLookup(
            {c: a[thr_rows] for c, a in self.columns.items()},
            "throttle", "a_measured")
        self._steer_lu = _InverseLookup(
            {c: a[str_rows] for c, a in self.columns.items()},
            "steer", "omega_z_measured")

    # -- IO -----------------------------------------------------------------

    @classmethod
    def load(cls, csv_path: str | os.PathLike) -> "SweepTable":
        """Load a sweep CSV (header row required, ``SWEEP_COLUMNS`` schema)."""
        data = np.genfromtxt(csv_path, delimiter=",", names=True,
                             dtype=np.float64)
        names = data.dtype.names or ()
        missing = [c for c in SWEEP_COLUMNS if c not in names]
        if missing:
            raise ValueError(
                f"{csv_path}: missing sweep columns {missing} (found {list(names)})")
        return cls({c: data[c] for c in SWEEP_COLUMNS})

    def save(self, csv_path: str | os.PathLike) -> None:
        """Write the table in the same CSV schema ``load`` reads."""
        parent = os.path.dirname(os.path.abspath(os.fspath(csv_path)))
        os.makedirs(parent or ".", exist_ok=True)
        stacked = np.column_stack([self.columns[c] for c in SWEEP_COLUMNS])
        np.savetxt(csv_path, stacked, delimiter=",",
                   header=",".join(SWEEP_COLUMNS), comments="",
                   fmt=["%.4f"] * 5 + ["%.6f"] * 2)

    # -- Inverse lookups (the controller-facing API) --------------------------

    def throttle_for(self, v: float, a_target: float,
                     pitch_deg: float, roll_deg: float) -> float:
        """Throttle whose measured acceleration at (v, pitch, roll) is closest
        to ``a_target`` (saturates when the target is unachievable)."""
        return self._throttle_lu.inverse(v, a_target, pitch_deg, roll_deg)

    def steer_for(self, v: float, omega_target: float,
                  pitch_deg: float, roll_deg: float) -> float:
        """Steer whose measured yaw rate at (v, pitch, roll) is closest to
        ``omega_target`` (saturates when the target is unachievable)."""
        return self._steer_lu.inverse(v, omega_target, pitch_deg, roll_deg)

    def __len__(self) -> int:
        return len(self.columns[SWEEP_COLUMNS[0]])
