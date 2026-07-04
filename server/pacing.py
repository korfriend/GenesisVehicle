"""Adaptive catch-up pacer (v1.0.20).

| 약자 | 의미 |
|---|---|
| cap | 한 루프에서 몰아 돌릴 물리 스텝 상한 (catch-up cap) |
| BURST | cap = max — 밀린 시간을 몰아 돌려 실시간 복귀를 시도하는 모드 |
| SMOOTH | cap = 1 — 복귀를 포기하고 균일한 배속의 슬로모션으로 열화하는 모드 |
| steps/loop | 루프당 실행한 물리 스텝 수 (~1.0 = 실시간 유지) |

고정 cap 의 딜레마: 스텝이 dt 예산을 초과하는 과부하에서 cap=5 는 5-스텝
버스트로 프레임 간격이 출렁(뚝뚝)이고, cap=1 은 부드럽지만 일시적 히컵
(로딩 스파이크 등) 뒤에도 밀린 시간을 영원히 따라잡지 못한다.

``AdaptiveCatchup`` 은 steps/loop 를 창(window) 평균으로 모니터링해 둘을
자동 전환한다:

- **BURST → SMOOTH**: 창 평균 steps/loop ≥ ``hi``(기본 1.5) — 지속적 과부하.
  버스트로 복귀가 안 되는 상태이므로 cap=1 로 내려 균일 저속으로 전환.
- **SMOOTH → BURST**: '스텝을 실행한 루프'의 소요가 ``sim_dt × recover_ratio``
  (기본 0.8) 미만으로 창 길이만큼 **연속** 유지 — 하드웨어에 여유가 돌아왔다는
  뜻이므로 cap 을 max 로 되돌려 밀린 시간 따라잡기를 재개.
- 전환 직후 ``cooldown`` 루프 동안은 재전환 금지 (진동 방지 히스테리시스).

``--max-catchup-steps N`` 을 준 경우(fixed) 적응 로직은 꺼지고 cap=N 고정.
전환 시마다 ``[Pacing] [AdaptiveCatchup]`` 로그를 남기고, [STATS] 라인에
현재 모드가 표기된다.
"""

from __future__ import annotations

import time
from collections import deque


class AdaptiveCatchup:
    """steps/loop 기반 catch-up cap 자동 전환기. 두 서버 루프가 공유한다.

    ``profile=True`` (서버 ``--pacing-profile``)면 전환 트리거 시점의 상황을
    상세 덤프한다: 창의 steps/loop 이력, 루프 소요 통계(avg/p95), dt 예산
    대비 추정 배속, 마지막 전환 이후 경과. 벤치마크는 이 옵션을 항상 켠다.
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
        # 기동 유예: 첫 grace 루프는 관측만 하고 전환하지 않는다 — 빌드 직후의
        # JIT/settle 과도기(스텝 50~200ms 스파이크)가 창을 오염시켜 스퓨리어스
        # SMOOTH 전환을 만들던 것을 방지 (벤치 실측: L2×30이 t=1.4s에 오탐).
        self.grace = int(grace)
        self.profile = bool(profile)

        self._mode = "BURST"          # BURST(cap=max) | SMOOTH(cap=1)
        self._steps_hist: deque = deque(maxlen=self.window)
        self._dur_hist: deque = deque(maxlen=self.window)   # 루프 소요(초) 이력
        self._good_streak = 0         # SMOOTH 회복 판정용 연속 카운터
        self._cooldown_left = 0
        self._t0 = time.monotonic()
        self._last_switch_t: "float | None" = None
        self._n_switches = 0

    # -- 루프가 읽는 값 ---------------------------------------------------
    def cap(self) -> int:
        if self.fixed is not None:
            return self.fixed
        return self.max_cap if self._mode == "BURST" else 1

    @property
    def mode(self) -> str:
        if self.fixed is not None:
            return f"fixed:{self.fixed}"
        return "burst" if self._mode == "BURST" else "smooth"

    # -- 루프가 매회 끝에서 호출 -------------------------------------------
    def update(self, catchup_steps: int, loop_dur: float) -> None:
        """``catchup_steps``: 이번 루프에서 실행한 스텝 수. ``loop_dur``: 이번
        루프의 전체 소요(초; 물리 + capture + 송신 포함, sleep 제외 경로만)."""
        if self.fixed is not None:
            return
        if self.grace > 0:                 # 기동 유예 — 관측만, 전환 없음
            self.grace -= 1
            if self.grace == 0:
                self._steps_hist.clear()   # 과도기 표본은 창에서 버린다
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
        else:  # SMOOTH — 스텝을 실행한 루프의 소요로 회복 판정
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
            # 추정 배속: 루프당 시뮬 진행(steps×dt) / 루프당 현실 시간
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
