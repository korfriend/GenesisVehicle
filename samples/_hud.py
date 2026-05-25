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


def render_hud_frame(
    cam,
    *,
    title: str,
    lines: Sequence[str],
    perf_ms: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Render one camera frame and overlay the HUD on top.

    Args
    ----
    cam        : Genesis camera (the one created with ``scene.add_camera``).
    title      : Bold title text line (one line).
    lines      : Per-sample state lines (throttle, speed, etc.).
    perf_ms    : Rolling ms/step from ``PerfMeter.ms_per_step()``, drawn as
                 a cyan footer with computed fps. Pass ``None`` to skip
                 the perf footer.

    Returns
    -------
    BGR ``np.ndarray`` ready for ``cv2.imshow`` — or ``None`` if cv2 is
    not installed (caller should ``have_cv2()``-guard).
    """
    if cv2 is None:
        return None
    rgb, *_ = cam.render()
    frame = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb)
    if frame.ndim == 4:
        frame = frame[0]
    frame = cv2.cvtColor(frame.astype(np.uint8).copy(), cv2.COLOR_RGB2BGR)

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
