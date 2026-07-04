"""Adaptive catch-up pacer (v1.0.20).

| abbr | meaning |
|---|---|
| cap | upper bound on physics steps run in one loop (catch-up cap) |
| BURST | cap = max — mode that bursts through backlog to try to return to real-time |
| SMOOTH | cap = 1 — mode that gives up recovery and degrades to uniform-speed slow motion |
| steps/loop | physics steps executed per loop (~1.0 = holding real-time) |

The fixed-cap dilemma: under overload where a step exceeds the dt budget,
cap=5 causes 5-step bursts that make frame intervals stutter, while cap=1 is
smooth but never catches up on backlog even after a transient hiccup (e.g. a
loading spike).

``AdaptiveCatchup`` monitors a windowed average of steps/loop and switches
between the two automatically:

- **BURST → SMOOTH**: window-average steps/loop ≥ ``hi`` (default 1.5) —
  sustained overload. Bursting cannot recover, so drop to cap=1 for uniform
  slow speed.
- **SMOOTH → BURST**: the duration of 'loops that executed a step' stays
  below ``sim_dt × recover_ratio`` (default 0.8) for a full window
  **consecutively** — hardware headroom has returned, so restore cap to max
  and resume catching up on backlog.
- Right after a switch, re-switching is forbidden for ``cooldown`` loops
  (anti-oscillation hysteresis).

If ``--max-catchup-steps N`` is given (fixed), the adaptive logic is off and
cap=N is fixed. Every switch leaves a ``[Pacing] [AdaptiveCatchup]`` log, and
the current mode is shown on the [STATS] line.
"""

from __future__ import annotations

import time
from collections import deque


class AdaptiveCatchup:
    """steps/loop-based automatic catch-up cap switcher. Shared by both server loops.

    With ``profile=True`` (server ``--pacing-profile``), dumps details at the
    moment a switch triggers: the window's steps/loop history, loop-duration
    stats (avg/p95), estimated speed vs the dt budget, and time since the last
    switch. Benchmarks always enable this option.
    """

    def __init__(self, max_cap: int, sim_dt: float, *,
                 fixed: "int | None" = None,
                 window: int = 25, hi: float = 1.5,
                 recover_ratio: float = 0.9, cooldown: int = 50,
                 grace: int = 100, profile: bool = False):
        self.max_cap = max(1, int(max_cap))
        self.sim_dt = float(sim_dt)
        self.fixed = max(1, int(fixed)) if fixed is not None else None
        self.window = int(window)
        self.hi = float(hi)
        self.recover_ratio = float(recover_ratio)
        self.cooldown = int(cooldown)
        # Startup grace: the first `grace` loops only observe, never switch —
        # prevents the post-build JIT/settle transient (50-200ms step spikes)
        # from polluting the window and causing spurious SMOOTH switches
        # (bench measurement: L2×30 false-triggered at t=1.4s).
        self.grace = int(grace)
        self.profile = bool(profile)

        self._mode = "BURST"          # BURST(cap=max) | SMOOTH(cap=1)
        self._steps_hist: deque = deque(maxlen=self.window)
        self._dur_hist: deque = deque(maxlen=self.window)   # loop duration (sec) history
        self._good_streak = 0         # consecutive counter for SMOOTH recovery decision
        self._cooldown_left = 0
        self._t0 = time.monotonic()
        self._last_switch_t: "float | None" = None
        self._n_switches = 0

    # -- Values the loop reads --------------------------------------------
    def cap(self) -> int:
        if self.fixed is not None:
            return self.fixed
        return self.max_cap if self._mode == "BURST" else 1

    @property
    def mode(self) -> str:
        if self.fixed is not None:
            return f"fixed:{self.fixed}"
        return "burst" if self._mode == "BURST" else "smooth"

    # -- Called by the loop at the end of every iteration --------------------
    def update(self, catchup_steps: int, loop_dur: float) -> None:
        """``catchup_steps``: steps executed in this loop. ``loop_dur``: total
        duration of this loop (sec; includes physics + capture + send,
        non-sleep path only)."""
        if self.fixed is not None:
            return
        if self.grace > 0:                 # startup grace — observe only, no switching
            self.grace -= 1
            if self.grace == 0:
                self._steps_hist.clear()   # discard transient samples from the window
                self._dur_hist.clear()
            else:
                self._steps_hist.append(catchup_steps)
                self._dur_hist.append(float(loop_dur))
            return
        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        self._steps_hist.append(catchup_steps)
        self._dur_hist.append(float(loop_dur))

        if self._mode == "BURST":
            if (self._cooldown_left == 0
                    and len(self._steps_hist) == self.window
                    and sum(self._steps_hist) / self.window >= self.hi):
                self._switch("SMOOTH",
                             f"창 평균 steps/loop "
                             f"{sum(self._steps_hist) / self.window:.1f} ≥ {self.hi}"
                             f" (지속 과부하) → cap=1 균일 슬로모션")
        else:  # SMOOTH — recovery judged by duration of loops that executed a step
            if catchup_steps >= 1:
                if loop_dur < self.sim_dt * self.recover_ratio:
                    self._good_streak += 1
                else:
                    self._good_streak = 0
                if (self._cooldown_left == 0
                        and self._good_streak >= self.window):
                    self._switch("BURST",
                                 f"루프 소요 < dt×{self.recover_ratio} 가 "
                                 f"{self.window}회 연속 (여유 회복) → cap="
                                 f"{self.max_cap} 실시간 복귀 재개")

    def _switch(self, mode: str, why: str) -> None:
        prev = self._mode
        self._mode = mode
        self._n_switches += 1
        now = time.monotonic()
        print(f" [Pacing] [AdaptiveCatchup] {why}")
        if self.profile:
            durs = sorted(self._dur_hist) or [0.0]
            avg_ms = sum(durs) / len(durs) * 1e3
            p95_ms = durs[min(len(durs) - 1, int(len(durs) * 0.95))] * 1e3
            steps = list(self._steps_hist)
            avg_steps = (sum(steps) / len(steps)) if steps else 0.0
            since = (f"{now - self._last_switch_t:.1f}s since last switch"
                     if self._last_switch_t is not None else "first switch")
            # Estimated speed: sim progress per loop (steps×dt) / real time per loop
            speed = ((avg_steps * self.sim_dt) / (avg_ms / 1e3)
                     if avg_ms > 0 else 0.0)
            print(f" [Pacing] [AdaptiveCatchup] [profile] switch#{self._n_switches} "
                  f"{prev}→{mode} @t={now - self._t0:.1f}s ({since}) | "
                  f"window steps/loop={steps} (avg {avg_steps:.2f}) | "
                  f"loop_dur avg {avg_ms:.2f} ms p95 {p95_ms:.2f} ms | "
                  f"budget {self.sim_dt * 1e3:.1f} ms | est speed {speed:.2f}x")
        self._last_switch_t = now
        self._steps_hist.clear()
        self._dur_hist.clear()
        self._good_streak = 0
        self._cooldown_left = self.cooldown
