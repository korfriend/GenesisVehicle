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


class SendRateLimiter:
    """Wall-clock gate for the L2 server's state send (``--send-hz``, v1.6.7).

    | abbr | meaning |
    |---|---|
    | H | requested send rate (Hz) |
    | P | send period = 1/H (sec) |
    | L2 | per-entity server mode (K interacting vehicles, n_envs=1) |

    Physics keeps running at ``dt``; only the TargetBulk/SimTime send to the
    client is downsampled. ``period=None`` disables the gate entirely and
    ``should_send`` then returns ``True`` unconditionally — that is the default
    and the historical behaviour (one send per loop).

    **One budget.** Every send, ordinary or flushed, goes through ``_record``.
    The deadline (``_next_send_t``) and the last send time (``_last_send_t``)
    are separate fields only so the deadline can CARRY: when H does not divide
    the loop rate, adding P to the previous deadline (rather than to ``now``)
    keeps the long-run average at H. The two are tied by the invariant

        ``_next_send_t <= _last_send_t + P``   (holds after every _record)

    - snap path: ``_next_send_t = now + P = _last_send_t + P`` (equality).
    - carry path: entered only when ``now >= _next_send_t_old``, so
      ``_next_send_t_new = _next_send_t_old + P <= now + P``.

    That invariant is what bounds the realised rate at H (+ the reset flush
    rate). An earlier design ran a SECOND, independent budget for a "soft
    flush" on override/teleport edges; because ordinary sends never consumed
    it, the two could fire in the same period and the realised rate reached
    ~1.65H on a 30 Hz override stream against a 40 Hz loop. There is now one
    budget and no soft path. Do not re-split it — ``tests/
    test_server_send_downsample.py`` pins the realised rate, not just the
    fields.

    ``now`` is injected by the caller (the server passes
    ``time.perf_counter()``), so the whole class is testable without a clock.

    NB the caller is responsible for rejecting a non-positive rate; the server
    does it in argparse (``--send-hz`` must be > 0) and demotes H >= 1/sim_dt
    to ``period=None``.
    """

    def __init__(self, period: "float | None"):
        self.period = float(period) if period is not None else None
        self.enabled = period is not None
        # -inf, so the FIRST should_send() after build fires: `now <
        # _next_send_t` is False for any finite now. Initialising to 0.0 or to
        # a perf_counter() reading would delay the first TargetBulk by up to
        # one period, and that first one is what the client is waiting for.
        # The invariant holds at t0 too: -inf <= -inf + P.
        self._next_send_t = float("-inf")
        self._last_send_t = float("-inf")

    def should_send(self, now: float) -> bool:
        """Ordinary (non-flush) send decision. Consumes the budget when True."""
        if not self.enabled:
            return True
        if now < self._next_send_t:
            return False
        # More than a whole period late (a stall, or the very first call from
        # -inf): snap the deadline to `now` instead of carrying, so a 1 s hitch
        # does not queue 20 catch-up sends.
        self._record(now, snap=(self._next_send_t + self.period <= now))
        return True

    def notify_sent(self, now: float) -> None:
        """Tell the gate a send went out outside ``should_send`` (the reset
        flush). Snaps the deadline, so the flush does not leave the next
        ordinary send due immediately."""
        if not self.enabled:
            return
        self._record(now, snap=True)

    def _record(self, now: float, snap: bool) -> None:
        self._last_send_t = now
        self._next_send_t = (now + self.period) if snap \
            else (self._next_send_t + self.period)
