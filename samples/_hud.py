"""Minimal cv2 HUD helper for sample `--viewer` modes.

Purpose
-------
When a sample runs with ``--viewer`` we want a visible window with
perf indicators (ms/step, fps) and per-sample state (speed, throttle,
position, etc.) drawn on top of the offscreen camera frame.

Genesis's built-in viewer (``gs.Scene(show_viewer=True)``) gives a 3D
interactive window but has no text-overlay API. The classic workaround
is to render the offscreen camera each step and use OpenCV to draw
text on top and ``cv2.imshow``. That's what this helper does.

Dependency
----------
``opencv-python`` only. No pynput, no _demo_hud (the umbrella helper
which adds a keyboard listener / chase-cam presets — not needed at the
SDK sample level). If cv2 is missing, ``have_cv2()`` returns False and
the caller should skip HUD setup (or fall back to plain ``cam.render()``).

Usage pattern
-------------
```python
from genesis_vehicle.samples import _hud

# Per-sample setup at start of main():
hud_perf = _hud.PerfMeter(window=60)   # rolling-mean ms/step over last N steps

# Per step inside the main loop:
hud_perf.tick()
if args.viewer and _hud.have_cv2():
    frame = _hud.render_hud_frame(
        cam,
        title="quickstart  v0.5.23",
        lines=[
            f"t = {t:5.2f} s   throttle = {throttle:+.2f}",
            f"pos = ({p[0]:+6.2f}, {p[1]:+6.2f})   speed = {speed:5.2f} m/s",
        ],
        perf_ms=hud_perf.ms_per_step(),
    )
    if not _hud.cv2_show("quickstart", frame):
        break   # ESC pressed

# At end:
_hud.cv2_cleanup()
```
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Optional, Sequence

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# ----------------------------------------------------------------------
# PerfMeter — rolling-mean ms/step over a configurable window
# ----------------------------------------------------------------------

class PerfMeter:
    """Tracks wall-clock time per ``tick()`` call and returns the rolling
    mean ms/step over the last ``window`` ticks.

    The cost of a tick is one ``time.perf_counter()`` call + a deque
    append + a sum + a division — well under 1 μs. Safe to call inside
    a tight simulation loop without distorting throughput.
    """

    def __init__(self, window: int = 60):
        self._dt = deque(maxlen=int(window))
        self._last: Optional[float] = None

    def tick(self) -> None:
        """Call once per simulation step. The first tick just records
        the start time (no interval yet); subsequent ticks accumulate
        intervals into the rolling window."""
        t = time.perf_counter()
        if self._last is not None:
            self._dt.append(t - self._last)
        self._last = t

    def ms_per_step(self) -> float:
        """Rolling-mean ms/step. Returns 0.0 before the second tick."""
        if not self._dt:
            return 0.0
        return (sum(self._dt) / len(self._dt)) * 1000.0

    def fps(self) -> float:
        ms = self.ms_per_step()
        return 1000.0 / ms if ms > 0 else 0.0

    def reset(self) -> None:
        self._dt.clear()
        self._last = None


# ----------------------------------------------------------------------
# HUD rendering
# ----------------------------------------------------------------------

def have_cv2() -> bool:
    """``True`` if ``opencv-python`` is importable; ``False`` otherwise."""
    return cv2 is not None


def _tile_grid(arr: np.ndarray, per_row: Optional[int], max_cell_size: Optional[int]) -> np.ndarray:
    """Tile an (N, H, W, 3) stack into a (R*h, C*w, 3) mosaic.

    Used when ``env_separate_rigid=True`` makes ``cam.render()`` return a
    stack of per-env frames. ``per_row`` defaults to ``round(sqrt(N))``;
    cells are downsized to ``max_cell_size`` on their longer axis to
    keep the mosaic display-sized.
    """
    n, h, w, c = arr.shape
    if per_row is None or per_row <= 0:
        per_row = max(1, int(round(math.sqrt(n))))
    n_rows = math.ceil(n / per_row)
    if max_cell_size is not None and max(h, w) > max_cell_size:
        scale = max_cell_size / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        arr = np.stack([cv2.resize(arr[i], (new_w, new_h), interpolation=cv2.INTER_AREA)
                        for i in range(n)])
        h, w = new_h, new_w
    mosaic = np.zeros((n_rows * h, per_row * w, c), dtype=arr.dtype)
    for i in range(n):
        r, col = divmod(i, per_row)
        mosaic[r * h:(r + 1) * h, col * w:(col + 1) * w] = arr[i]
    return mosaic


def render_hud_frame(
    cam,
    *,
    title: str,
    lines: Sequence[str],
    perf_ms: Optional[float] = None,
    grid_per_row: Optional[int] = None,
    max_cell_size: Optional[int] = 480,
) -> Optional[np.ndarray]:
    """Render one camera frame and overlay the HUD on top.

    Args
    ----
    cam            : Genesis camera (the one created with ``scene.add_camera``).
    title          : Bold title text line (one line).
    lines          : Per-sample state lines (throttle, speed, etc.).
    perf_ms        : Rolling ms/step from ``PerfMeter.ms_per_step()``, drawn as
                     a cyan footer with computed fps. Pass ``None`` to skip
                     the perf footer.
    grid_per_row   : When ``env_separate_rigid=True`` causes the render to
                     return a per-env stack, lay the cells out ``per_row``
                     wide. ``None`` → ``round(sqrt(N))``.
    max_cell_size  : Downsize each cell so its longer axis is at most this
                     many pixels (default 480). Set ``None`` for no
                     downsizing — useful for high-res single-env views.

    Returns
    -------
    BGR ``np.ndarray`` ready for ``cv2.imshow`` — or ``None`` if cv2 is
    not installed (caller should ``have_cv2()``-guard).
    """
    if cv2 is None:
        return None
    rgb, *_ = cam.render()
    arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
    if arr.ndim == 4:
        # env_separate_rigid → (N, H, W, 3) stack of per-env frames.
        frame = _tile_grid(arr.astype(np.uint8), grid_per_row, max_cell_size)
    else:
        frame = arr.astype(np.uint8)
    frame = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)

    h, w = frame.shape[:2]
    n_lines = 1 + len(lines) + (1 if perf_ms is not None else 0)
    pad = 18
    line_h = 28
    header_h = pad + n_lines * line_h + pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    y = pad + line_h
    cv2.putText(frame, title, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.80, (255, 255, 255), 2, cv2.LINE_AA)
    y += line_h
    for line in lines:
        cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (220, 220, 220), 1, cv2.LINE_AA)
        y += line_h

    if perf_ms is not None and perf_ms > 0:
        fps = 1000.0 / perf_ms
        perf_text = f"perf: {perf_ms:6.2f} ms/step  ({fps:6.1f} fps)"
        cv2.putText(frame, perf_text, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.70, (90, 220, 255), 2, cv2.LINE_AA)

    return frame


def cv2_show(window: str, frame: Optional[np.ndarray]) -> bool:
    """Display ``frame`` in the named cv2 window. Returns ``False`` if ESC
    was pressed (caller should ``break`` the main loop). Returns ``True``
    otherwise (continue)."""
    if cv2 is None or frame is None:
        return True
    cv2.imshow(window, frame)
    key = cv2.waitKey(1) & 0xFF
    return key != 27   # 27 = ESC


def cv2_cleanup() -> None:
    """Close all cv2 windows. Safe to call even if cv2 isn't installed."""
    if cv2 is not None:
        cv2.destroyAllWindows()


# ----------------------------------------------------------------------
# End-of-run perf summary — surfaced regardless of completion vs ESC
# ----------------------------------------------------------------------

def print_perf_summary(
    *,
    sample: str,
    completed: bool,
    n_done: int,
    n_target: int,
    wall: float,
    batch: Optional[int] = None,
    batch_label: str = "unit",
    extra: Optional[Sequence[str]] = None,
) -> None:
    """Print a multi-line perf summary block at the end of a sample run.

    Args
    ----
    sample      : Sample name (e.g. ``"multi_env_render"``) — shown in header.
    completed   : True if the run ran to its planned step count, False if the
                  user ESC-quit early. The header line distinguishes the two.
    n_done      : Steps actually executed (use ``step + 1 if user_quit else
                  n_target``).
    n_target    : Steps the loop was asked to run.
    wall        : Wall time over those ``n_done`` steps, in seconds.
    batch       : Items processed per step (n_envs, K_total, or n_envs*K).
                  Pass ``None`` for single-vehicle, single-env samples.
    batch_label : Label for the batched throughput line — e.g. ``"env"`` →
                  "X env/step → Y env-steps/s".
    extra       : Extra lines appended after the standard block (e.g.
                  diagnostic context the sample wants to surface).

    The summary is always printed (regardless of ``completed``), so the user
    can compare runs cut short by ESC against full runs on the same line
    numbers.
    """
    bar = "=" * 70
    status = "COMPLETED" if completed else "USER QUIT (ESC)"
    n_done = max(1, int(n_done))
    pct = 100.0 * n_done / max(1, int(n_target))
    ms = wall / n_done * 1000.0
    sps = n_done / wall if wall > 0 else 0.0
    print()
    print(bar)
    print(f" {sample}  —  {status}")
    print(bar)
    print(f"  steps      : {n_done:>6} / {n_target:<6}   ({pct:5.1f}% of requested)")
    print(f"  wall time  : {wall:>6.2f} s")
    print(f"  ms / step  : {ms:>6.2f}")
    print(f"  steps / s  : {sps:>6.2f}")
    if batch is not None and batch > 1:
        bps = batch * sps
        print(f"  batch      : {batch} {batch_label}/step  ->  "
              f"{bps:,.0f} {batch_label}-steps/s")
    if extra:
        for line in extra:
            print(f"  {line}")
    print(bar)
