"""State-machine guards for the v1.0.20 adaptive catch-up pacer.

Pure-python (no genesis): the pacer only sees (catchup_steps, loop_dur)
pairs, so overload/recovery scenarios are simulated directly.

| 약자 | 의미 |
|---|---|
| cap | 루프당 물리 스텝 상한 |
| burst / smooth | cap=max (실시간 복귀 시도) / cap=1 (균일 슬로모션) |
"""

from genesis_vehicle.server.pacing import AdaptiveCatchup

DT = 0.025


def _pacer(**kw):
    kw.setdefault("max_cap", 5)
    kw.setdefault("sim_dt", DT)
    kw.setdefault("window", 10)
    kw.setdefault("cooldown", 5)
    kw.setdefault("grace", 0)      # tests exercise the steady-state machine
    return AdaptiveCatchup(**kw)


def test_grace_period_blocks_startup_false_positive():
    """빌드 직후 과도기(스텝 스파이크)가 창을 오염시켜도 grace 동안은 전환
    금지 — grace 뒤 정상 부하면 burst 유지, 진짜 과부하면 그때 전환."""
    p = _pacer(grace=20)
    for _ in range(20):                                   # startup transient
        p.update(catchup_steps=5, loop_dur=5 * DT * 4)
    assert p.mode == "burst"                              # no spurious switch
    for _ in range(50):                                   # healthy afterwards
        p.update(catchup_steps=1, loop_dur=DT * 0.5)
    assert p.mode == "burst"
    p2 = _pacer(grace=20)
    for _ in range(20 + 10):                              # overload persists past grace
        p2.update(catchup_steps=5, loop_dur=5 * DT * 2)
    assert p2.mode == "smooth"                            # genuine overload still caught


def test_starts_in_burst_with_max_cap():
    p = _pacer()
    assert p.cap() == 5
    assert p.mode == "burst"


def test_sustained_overload_switches_to_smooth():
    p = _pacer()
    # saturated: every loop runs the full 5-step burst, each step over budget
    for _ in range(10):
        p.update(catchup_steps=5, loop_dur=5 * DT * 2)
    assert p.cap() == 1
    assert p.mode == "smooth"


def test_healthy_load_stays_burst():
    p = _pacer()
    for _ in range(100):
        p.update(catchup_steps=1, loop_dur=DT * 0.5)   # comfortably real-time
    assert p.cap() == 5
    assert p.mode == "burst"


def test_recovery_returns_to_burst():
    p = _pacer()
    for _ in range(10):
        p.update(catchup_steps=5, loop_dur=5 * DT * 2)   # overload → smooth
    assert p.mode == "smooth"
    # cooldown first, then `window` consecutive fast step-loops
    for _ in range(5 + 10):
        p.update(catchup_steps=1, loop_dur=DT * 0.5)
    assert p.cap() == 5
    assert p.mode == "burst"


def test_recovery_streak_resets_on_slow_loop():
    p = _pacer()
    for _ in range(10):
        p.update(catchup_steps=5, loop_dur=5 * DT * 2)
    assert p.mode == "smooth"
    # alternate fast/slow — never `window` consecutive good loops → stays smooth
    for i in range(60):
        dur = DT * 0.5 if i % 5 else DT * 1.5
        p.update(catchup_steps=1, loop_dur=dur)
    assert p.mode == "smooth"


def test_zero_step_loops_do_not_fake_recovery():
    p = _pacer()
    for _ in range(10):
        p.update(catchup_steps=5, loop_dur=5 * DT * 2)
    assert p.mode == "smooth"
    # idle loops (0 steps, tiny durations) must not count as recovery evidence
    for _ in range(100):
        p.update(catchup_steps=0, loop_dur=0.001)
    assert p.mode == "smooth"


def test_fixed_mode_never_switches():
    p = _pacer(fixed=1)
    for _ in range(50):
        p.update(catchup_steps=1, loop_dur=5 * DT)       # heavy overload
    assert p.cap() == 1
    assert p.mode == "fixed:1"
    p2 = _pacer(fixed=3)
    for _ in range(50):
        p2.update(catchup_steps=3, loop_dur=5 * DT)
    assert p2.cap() == 3


def test_profile_dump_on_switch(capsys):
    p = _pacer(profile=True)
    for _ in range(10):
        p.update(catchup_steps=5, loop_dur=5 * DT * 2)
    out = capsys.readouterr().out
    assert "[AdaptiveCatchup] [profile] switch#1 BURST→SMOOTH" in out
    assert "window steps/loop=" in out and "est speed" in out
    # off by default: no profile dump, only the one-line switch message
    p2 = _pacer()
    for _ in range(10):
        p2.update(catchup_steps=5, loop_dur=5 * DT * 2)
    out2 = capsys.readouterr().out
    assert "[profile]" not in out2 and "[AdaptiveCatchup]" in out2


def test_cooldown_blocks_immediate_flapping():
    p = _pacer()
    for _ in range(10):
        p.update(catchup_steps=5, loop_dur=5 * DT * 2)   # → smooth, cooldown=5
    # 5 fast loops during cooldown + window(10) needed after — at loop 14 the
    # streak is 14 but cooldown ended at 5; ensure it did NOT flip before
    # accumulating a full window of evidence AND cooldown expiry.
    flipped_at = None
    for i in range(20):
        p.update(catchup_steps=1, loop_dur=DT * 0.5)
        if p.mode == "burst" and flipped_at is None:
            flipped_at = i + 1
    assert flipped_at is not None and flipped_at >= 10
