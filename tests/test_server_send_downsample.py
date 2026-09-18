"""v1.6.7 ``--send-hz``: the L2 state-send rate gate (``SendRateLimiter``), its
wiring into the server loop, the ``[SERVE]`` counters it adds and the benchmark
forwarding that makes it observable.

| abbr | meaning |
|---|---|
| L2 | per-entity server mode (K interacting vehicles, n_envs=1) |
| L3 | multi-env server mode (``--multi-env``) |
| H | requested send rate (Hz), the ``--send-hz`` value |
| P | send period = 1/H (sec) |
| SERVE | the server's per-window serving-counter stdout line |
| STATS | the server's frozen per-window timing line |
| AST | Abstract Syntax Tree |

Pure Python: no scene is built and no server process is started. The live
arrival-rate counterpart is ``test_server_send_downsample_live.py``, gated
behind ``GENESIS_VEHICLE_LIVE_SERVER_TEST``.
"""
import ast
import inspect
import math
import random
import textwrap

import pytest

from genesis_vehicle.server.pacing import SendRateLimiter

# Source text of the v1.6.8 wall-clock window trigger in physics_server.main()
# (it replaced ``if log_count >= 50:``). Assertions that split on it check it
# is PRESENT first: ``str.split`` on an absent separator returns the whole
# string, so a renamed trigger would silently make them vacuous — which is
# exactly what happened to these tests when the trigger changed.
_WINDOW_TRIGGER = "if _t_end - _win_t0 >= _stats_interval:"

# The OSC calls that ship this loop's state to the client. `_t_end` must be
# stamped after all of them — that is the whole content of "after the loop's
# work". Deliberately NOT `send_message`: the window block sends
# /Genesis/Init/TimeDilation with it, AFTER the stamp, by design.
_STATE_SEND_CALLS = ("send_target_states_bulk", "send_dynamic_states_bulk",
                     "send_sim_time", "send_step_ack")


def _main_loop(node):
    """The `while True:` whose body carries the window counters."""
    loops = [n for n in ast.walk(node)
             if isinstance(n, ast.While)
             and isinstance(n.test, ast.Constant) and n.test.value is True
             and any(isinstance(c, ast.AugAssign)
                     and getattr(c.target, "id", "") == "log_count"
                     for c in ast.walk(n))]
    assert len(loops) == 1, "expected exactly one main loop"
    return loops[0]


def _stmt_index(body, pred):
    """Indices of the DIRECT statements of `body` satisfying pred(stmt)."""
    return [i for i, st in enumerate(body) if pred(st)]


def _assign_to(name):
    return lambda st: (isinstance(st, ast.Assign)
                       and any(getattr(t, "id", "") == name for t in st.targets))


def _contains_state_send(st):
    return any(isinstance(c, ast.Call)
               and getattr(c.func, "attr", "") in _STATE_SEND_CALLS
               for c in ast.walk(st))


# ==========================================================================
# 1. SendRateLimiter — pure unit
# ==========================================================================

def test_off_is_off_on_every_adversarial_clock():
    """``period=None`` is the default arm. It must return exactly ``True`` (the
    object, not a truthy value) for any clock at all — repeated, backwards,
    huge, zero, negative — because the server's OFF path is expected to be
    behaviourally identical to pre-v1.6.7."""
    lim = SendRateLimiter(None)
    assert lim.enabled is False
    assert lim.period is None
    for t in (0.0, 0.0, -1.0, 1e9, 1e-9, 1e9, 0.0, -1e9, 12345.6789):
        assert lim.should_send(t) is True
    # notify_sent on a disabled limiter must not raise (period is None).
    lim.notify_sent(3.0)
    assert lim.should_send(3.0) is True


def test_the_first_send_after_build_is_not_delayed():
    """Initial state pin. ``_next_send_t`` starts at -inf so the FIRST
    ``should_send`` fires immediately; initialising it to 0.0 or to a
    ``perf_counter()`` reading would hold the first TargetBulk back by up to
    one period, and that first one is the state the client is waiting for
    before it can draw anything."""
    for first_now in (0.0, 1e-9, 12345.678, 1e9):
        lim = SendRateLimiter(0.05)
        assert lim.should_send(first_now) is True
        # ...and the deadline is snapped to that moment, not carried from -inf.
        assert lim._next_send_t == pytest.approx(first_now + 0.05)


def test_dividing_rate_sends_on_every_other_loop():
    """H=20 on a 40 Hz (25 ms) loop = exactly every other loop."""
    lim = SendRateLimiter(0.05)
    ticks = [100.0 + 0.025 * i for i in range(21)]
    got = [lim.should_send(t) for t in ticks]
    assert got == [True, False] * 10 + [True]


def test_non_dividing_rate_carries_to_the_right_average():
    """H=25 (P=0.04) on a 40 Hz loop does not divide. The deadline CARRIES
    (next = previous deadline + P, not now + P), so the realised indices are
    uneven but the long-run average period converges on 0.04 s, not on the
    0.05 s a snap-every-time implementation would give."""
    lim = SendRateLimiter(0.04)
    ticks = [0.025 * i for i in range(81)]          # 2.0 s at 40 Hz
    idx = [i for i, t in enumerate(ticks) if lim.should_send(t)]
    assert idx == [0, 2, 4, 5, 7, 8, 10, 12, 13, 15, 16, 18, 20, 21, 23, 24,
                   26, 28, 29, 31, 33, 34, 36, 37, 39, 41, 42, 44, 45, 47, 49,
                   50, 52, 53, 55, 57, 58, 60, 61, 63, 65, 66, 68, 69, 71, 73,
                   74, 76, 77, 79]
    assert len(idx) == 50                       # 2.0 s at 25 Hz
    span = ticks[idx[-1]] - ticks[idx[0]]
    assert span / (len(idx) - 1) == pytest.approx(0.04, abs=0.001)
    # a snap-every-time (carry-free) implementation would land on 0.05 here:
    snapped = []
    nxt = float("-inf")
    for t in ticks:
        if t >= nxt:
            snapped.append(t)
            nxt = t + 0.04
    assert len(snapped) == 41                   # i.e. 20 Hz, not the 50 above


def test_a_stall_snaps_forward_instead_of_queueing_catch_up_sends():
    lim = SendRateLimiter(0.05)
    assert lim.should_send(10.0) is True
    # 1 s hitch: one send, and the deadline lands one period after NOW.
    assert lim.should_send(11.0) is True
    assert lim._next_send_t == pytest.approx(11.05)
    assert lim.should_send(11.0) is False
    assert lim.should_send(11.049) is False
    assert lim.should_send(11.05) is True


def test_notify_sent_snaps_the_deadline_one_period_past_the_flush():
    lim = SendRateLimiter(0.05)
    lim.should_send(10.0)
    lim.notify_sent(10.01)                    # a reset flush mid-period
    assert lim._next_send_t == pytest.approx(10.06)
    assert lim.should_send(10.05) is False    # the flush moved the deadline
    assert lim.should_send(10.06) is True


def test_single_budget_invariant_holds_after_every_record():
    """DOCUMENTATION assertion, not the regression pin.

    It states the field-level invariant the design rests on --
    ``_next_send_t <= _last_send_t + P`` after every ``_record`` -- which is
    what makes an ordinary deadline always come due before (or exactly with) a
    "one period since the last send" test, and therefore what makes a second,
    soft-flush budget provably redundant.

    It is deliberately labelled documentation rather than a pin, because a
    future re-split would add a NEW field rather than violate this inequality
    between these two. The pin for the realised rate is
    ``test_realised_rate_is_bounded_by_H_plus_hard_flushes``.
    """
    P = 0.05
    lim = SendRateLimiter(P)
    rng = random.Random(20250918)
    t = 100.0
    checked = 0
    for _ in range(400):
        t += rng.choice([0.0, 0.001, 0.025, 0.025, 0.05, 0.4])
        if rng.random() < 0.1:
            lim.notify_sent(t)
            assert lim._next_send_t <= lim._last_send_t + P + 1e-12
            checked += 1
        elif lim.should_send(t):
            assert lim._next_send_t <= lim._last_send_t + P + 1e-12
            checked += 1
    assert checked > 50


# --------------------------------------------------------------------------
# the realised-rate pin
# --------------------------------------------------------------------------

def _drive(lim, ticks, flushes=None, events=None):
    """Mirror of the server's send gate (physics_server.main, the
    ``if not args.lockstep:`` block) over an injected clock.

    ``flushes[i]`` is the loop's ``_flush_hard`` (a reset). ``events[i]`` is
    the loop's ``_need_capture``-style "something moved the solver" flag; the
    shipped design does NOT flush on it, but if a second budget is ever
    re-introduced on the limiter (``soft_flush_allowed``), this harness
    consults it exactly the way the rejected v3 gate did, on the rising edge
    -- so the re-split shows up as extra sends here instead of in the field.

    Returns a list of ``(t, kind)`` with kind in {"flush", "soft", "ordinary"}.
    """
    out = []
    prev_ev = False
    soft_api = getattr(lim, "soft_flush_allowed", None)
    for i, t in enumerate(ticks):
        flush = bool(flushes[i]) if flushes else False
        ev = bool(events[i]) if events else False
        if flush:
            lim.notify_sent(t)
            out.append((t, "flush"))
        elif soft_api is not None and ev and not prev_ev and soft_api(t):
            lim.notify_sent(t)
            out.append((t, "soft"))
        elif lim.should_send(t):
            out.append((t, "ordinary"))
        prev_ev = ev
    return out


@pytest.mark.parametrize("name,H,loop_hz,events,flushes_every", [
    # (i) alternating events
    ("alternating", 20.0, 40.0, lambda i: i % 2 == 0, 0),
    # (ii) THE regression series: a 30 Hz override stream against a 40 Hz loop
    # makes the T,T,F,T,T,F... pattern that re-armed the rejected v3 soft flush
    # ~13 Hz on top of the 20 Hz ordinary rate -> a realised ~33 Hz = 1.65H.
    ("30hz_vs_40hz", 20.0, 40.0, lambda i: i % 3 != 2, 0),
    # (iii) seeded random EVENT FLAGS on the same uniform grid (see docstring)
    ("random_events", 20.0, 40.0, None, 0),
    # non-dividing H, and one with resets
    ("non_dividing", 25.0, 40.0, lambda i: i % 2 == 0, 0),
    ("with_resets", 20.0, 40.0, lambda i: i % 3 != 2, 37),
])
def test_realised_rate_is_bounded_by_H_plus_hard_flushes(name, H, loop_hz,
                                                         events, flushes_every):
    """MAJOR regression pin: the number of sends the gate REALISES, not the
    fields it holds.

    Time grid: pinned UNIFORM at ``1/loop_hz``; series (iii) randomises only
    the event FLAGS, never the timestamps. That is deliberate. The assertion
    is stated in AMORTISED form --

        ordinary + soft sends over a span T  <=  ceil(T/P) + 1

    -- and NOT as "at most one send in any P-length window", because the
    latter is false for a CORRECT implementation: with a carrying deadline a
    late send can be followed by a shorter-than-P gap (H=25 on a 40 Hz loop
    sends at t=0.100 and t=0.125). A maintainer who "fixed" that by dropping
    the carry would break
    ``test_non_dividing_rate_carries_to_the_right_average``.

    Hard flushes are exempt and counted separately: a reset is unbounded by
    design (its payload is the exact reset pose and exists on one loop only).
    """
    P = 1.0 / H
    n = 400
    ticks = [50.0 + i / loop_hz for i in range(n)]
    rng = random.Random(4242)
    if events is None:
        ev = [rng.random() < 0.4 for _ in range(n)]
    else:
        ev = [events(i) for i in range(n)]
    fl = [(flushes_every and i % flushes_every == 0) for i in range(n)]

    sends = _drive(SendRateLimiter(P), ticks, flushes=fl, events=ev)
    ordinary = [t for t, k in sends if k != "flush"]
    hard = [t for t, k in sends if k == "flush"]

    span = ticks[-1] - ticks[0]
    bound = math.ceil(span / P) + 1
    assert len(ordinary) <= bound, (
        f"{name}: {len(ordinary)} rate-gated sends over {span:.3f}s at H={H} "
        f"exceeds the amortised bound {bound} "
        f"(realised {len(ordinary) / span:.1f} Hz vs H={H})")
    # and the effective rate including resets stays at H + the reset rate
    eff = (len(ordinary) + len(hard)) / span
    assert eff <= H + len(hard) / span + 1.0 / span
    assert len(sends) >= 1


def test_the_rejected_two_budget_design_would_fail_the_realised_rate_pin():
    """Shows the pin has teeth: the v3 shape (a SECOND budget, consumed only by
    flushes) re-armed a soft flush ~13 Hz on the 30 Hz-vs-40 Hz series while
    ordinary sends kept their own 20 Hz, for a realised ~33 Hz = 1.65H. Run
    against the same harness it breaks the bound the shipped class satisfies.
    """
    P = 0.05

    class _V3Limiter(SendRateLimiter):
        """The rejected design, reconstructed for this test only."""

        def __init__(self, period):
            super().__init__(period)
            self._last_soft = float("-inf")

        def soft_flush_allowed(self, now):
            return now - self._last_soft >= self.period

        def notify_sent(self, now):        # v3: flush-only budget
            self._last_soft = now

    n, loop_hz = 400, 40.0
    ticks = [50.0 + i / loop_hz for i in range(n)]
    ev = [i % 3 != 2 for i in range(n)]
    sends = _drive(_V3Limiter(P), ticks, events=ev)
    span = ticks[-1] - ticks[0]
    realised = len([1 for _, k in sends if k != "flush"]) / span
    assert realised > 1.5 * (1.0 / P), realised          # ~33 Hz at H=20
    # ...and the shipped class on the identical series stays at H.
    shipped = _drive(SendRateLimiter(P), ticks, events=ev)
    assert len([1 for _, k in shipped if k != "flush"]) / span \
        <= 1.0 / P + 1.0 / span


# ==========================================================================
# 2. physics_server wiring — AST / source
# ==========================================================================

def _ps():
    pytest.importorskip("genesis")
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import physics_server
    return physics_server


def _main_src():
    return textwrap.dedent(inspect.getsource(_ps().main))


def _main_ast():
    return ast.parse(_main_src()).body[0]


def _gate(main_node):
    gates = [n for n in ast.walk(main_node)
             if isinstance(n, ast.If) and getattr(n.test, "id", "") == "_do_send"]
    assert len(gates) == 1, f"expected exactly one `if _do_send:` gate, got {len(gates)}"
    return gates[0]


def test_gate_wraps_the_whole_send_and_nothing_else():
    """Everything the client sees on the non-lockstep path — the lerp, the
    sim-time stamp and both bulk sends — must be inside the gate; the lockstep
    branch and the step ack must be outside it."""
    main_node = _main_ast()
    g = _gate(main_node)
    inside = ast.dump(g)
    for fn in ("lerp_state", "send_sim_time", "send_target_states_bulk",
               "send_dynamic_states_bulk"):
        assert fn in inside, fn
    assert "send_step_ack" not in inside

    # the gate lives inside `if not args.lockstep:`
    hosts = [n for n in ast.walk(main_node)
             if isinstance(n, ast.If) and isinstance(n.test, ast.UnaryOp)
             and "lockstep" in ast.dump(n.test)
             and any(c is g for c in ast.walk(n))]
    assert hosts, "the gate is not inside an `if not args.lockstep:` block"
    # ...and the ack is not.
    acks = [n for n in ast.walk(main_node)
            if isinstance(n, ast.Call)
            and getattr(n.func, "attr", "") == "send_step_ack"]
    assert acks and not any(a is c for a in acks for c in ast.walk(g))


_GATE_BODY_FROZEN = (
    "Assign([Name('alpha', Store())], BinOp(Name('accumulator', Load()), Div(),"
    " Name('SIM_DT', Load())))\n"
    "Assign([Name('alpha', Store())], Call(Name('float', Load()), "
    "[Call(Attribute(Name('np', Load()), 'clip', Load()), [Name('alpha', Load()),"
    " Constant(0.0), Constant(0.9999)], [])], []))\n"
    "Assign([Name('interpolated', Store())], Call(Name('lerp_state', Load()), "
    "[Name('prev_state', Load()), Name('curr_state', Load()), "
    "Name('alpha', Load())], []))\n"
    "If(Subscript(Name('interpolated', Load()), Constant('targets'), Load()), "
    "[Expr(Call(Attribute(Name('osc', Load()), 'send_sim_time', Load()), "
    "[BinOp(BinOp(BinOp(Name('step_count', Load()), Sub(), Constant(1)), Add(), "
    "Name('alpha', Load())), Mult(), Name('SIM_DT', Load()))], [])), "
    "Expr(Call(Attribute(Name('osc', Load()), 'send_target_states_bulk', Load()),"
    " [Subscript(Name('interpolated', Load()), Constant('targets'), Load())], "
    "[]))], [])\n"
    "If(Subscript(Name('interpolated', Load()), Constant('dynamic_obstacles'), "
    "Load()), [Expr(Call(Attribute(Name('osc', Load()), "
    "'send_dynamic_states_bulk', Load()), [Subscript(Name('interpolated', "
    "Load()), Constant('dynamic_obstacles'), Load())], []))], [])"
)


def test_gate_body_is_frozen_byte_for_byte_against_the_pre_gate_code():
    """The gate was introduced by INDENTING the existing send block one level
    and adding ``log_sends += 1`` in front of it. Nothing else may change: the
    0.9999 alpha clip, the ``np.clip`` call, the ``(step_count - 1 + alpha) *
    SIM_DT`` stamp expression, and the fact that ``if interpolated['targets']``
    wraps the stamp and the bulk send TOGETHER (the stamp is contracted to
    precede the TargetBulk it describes — osc_manager.send_sim_time).

    IF THIS FAILS: the gate body changed. That is not automatically wrong —
    but it is never incidental. Confirm the change is intended, then
    regenerate the constant with

        python -c "import ast,inspect,textwrap; \\
          from genesis_vehicle.server import physics_server as ps; \\
          t=ast.parse(textwrap.dedent(inspect.getsource(ps.main))).body[0]; \\
          g=[n for n in ast.walk(t) if isinstance(n,ast.If) and \\
             getattr(n.test,'id','')=='_do_send'][0]; \\
          print(repr(chr(10).join(ast.dump(s,annotate_fields=False) \\
                                  for s in g.body[1:])))"

    and paste it in. Do NOT delete the test. (It is CPython-version sensitive:
    ast.dump's rendering can change between releases. Measured on 3.12.)
    """
    g = _gate(_main_ast())
    first = g.body[0]
    assert isinstance(first, ast.AugAssign) and first.target.id == "log_sends"
    dumped = "\n".join(ast.dump(s, annotate_fields=False) for s in g.body[1:])
    assert dumped == _GATE_BODY_FROZEN


def test_the_skip_branch_only_counts():
    """The else arm of the gate must do nothing but bump send_skips — anything
    else there is work the flag was added to avoid."""
    g = _gate(_main_ast())
    assert len(g.orelse) == 1
    only = g.orelse[0]
    assert isinstance(only, ast.AugAssign) and only.target.id == "log_send_skips"


def test_off_path_costs_one_attribute_read_and_one_branch():
    """The OFF arm (the default, and every deployment before v1.6.7) must not
    call perf_counter: the clock read lives INSIDE `if _send_lim.enabled:`."""
    main_node = _main_ast()
    g = _gate(main_node)
    # the enabling `if _send_lim.enabled:` immediately precedes the gate
    enabled_ifs = [n for n in ast.walk(main_node)
                   if isinstance(n, ast.If) and "_send_lim" in ast.dump(n.test)
                   and getattr(n.test, "attr", "") == "enabled"]
    assert len(enabled_ifs) == 1
    blk = enabled_ifs[0]
    # every perf_counter() reachable from the send path sits inside it
    src = _main_src()
    seg = src.split("_do_send = True", 1)[1].split("if _do_send:", 1)[0]
    assert "perf_counter" in seg
    assert "perf_counter" in ast.dump(blk)
    # and the pre-gate statement is just the default assignment
    # (presence guard: `str.split` on an absent separator returns the WHOLE
    # source, and this is a `not in` assertion — it would pass vacuously in
    # the wrong direction if the anchor were ever reindented or renamed)
    assert "            _do_send = True" in src
    assert "_now_s" not in src.split("            _do_send = True")[0][-400:]


def test_flush_hard_is_armed_per_loop_outside_if_recv_and_set_once():
    """``_reset_ran`` is armed inside ``if recv:``; ``_flush_hard`` must be
    armed OUTSIDE it, at the top of the ``while True:`` body — an empty recv
    skips that block entirely and would leave last loop's flag standing.
    Mirrors test_reset_ran_is_armed_per_loop_and_set_only_by_the_reset_branch.
    """
    main_node = _main_ast()
    while_true = [n for n in ast.walk(main_node)
                  if isinstance(n, ast.While)
                  and getattr(n.test, "value", None) is True]
    assert len(while_true) == 1
    body = while_true[0].body
    # statement 0 is `loop_start = ...`; statement 1 is the arming
    armed = body[1]
    assert isinstance(armed, ast.Assign)
    assert [t.id for t in armed.targets] == ["_flush_hard"]
    assert armed.value.value is False

    # ...and it is NOT inside `if recv:`
    if_recv = [n for n in ast.walk(main_node)
               if isinstance(n, ast.If) and getattr(n.test, "id", "") == "recv"]
    assert len(if_recv) == 1
    assert not any(c is armed for c in ast.walk(if_recv[0]))

    # exactly one True assignment, and it is in the reset branch next to
    # `_reset_ran = True`
    trues = [n for n in ast.walk(main_node)
             if isinstance(n, ast.Assign)
             and any(getattr(t, "id", "") == "_flush_hard" for t in n.targets)
             and getattr(n.value, "value", None) is True]
    assert len(trues) == 1
    reset_ifs = [n for n in ast.walk(main_node)
                 if isinstance(n, ast.If) and "reset" in ast.dump(n.test)
                 and any(t is trues[0] for t in ast.walk(n))]
    assert reset_ifs and trues[0] in reset_ifs[-1].body
    src = _main_src()
    assert "_reset_ran = True" in src.split("_flush_hard = True")[0][-300:]


def test_no_soft_flush_path_exists():
    """The rejected v3 design armed a SECOND flush on the rising edge of
    ``_need_capture`` with its own budget. Under one budget an ordinary
    deadline always comes due first (or simultaneously), so that path could
    never fire strictly earlier — it was provably dead code that only
    mislabelled counters, and it doubled the worst-case rate when the budgets
    drifted apart. If either name comes back, this test and
    ``test_realised_rate_is_bounded_by_H_plus_hard_flushes`` both fail; fix the
    design, not the tests."""
    src = _main_src()
    assert "_prev_need_capture" not in src
    assert "soft_flush_allowed" not in src
    assert not hasattr(SendRateLimiter, "soft_flush_allowed")
    assert not hasattr(SendRateLimiter(0.05), "_last_soft")


def test_legacy_override_capture_is_not_ORed_into_the_flush():
    """--legacy-override-capture means 'capture even when nothing changed', a
    rollback switch — not 'something changed'. ORing it into the flush would
    turn the rate gate off whenever that switch is on."""
    src = _main_src()
    flush_line = [ln for ln in src.splitlines() if "_flush_hard = True" in ln]
    assert len(flush_line) == 1
    assert "_legacy_override_capture" not in flush_line[0]


# --------------------------------------------------------------------------
# argparse / banner / L3
# --------------------------------------------------------------------------

def _add_argument_kwargs(flag):
    tree = _main_ast()
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call)
                and getattr(n.func, "attr", "") == "add_argument"
                and n.args and getattr(n.args[0], "value", None) == flag):
            return {kw.arg: kw.value for kw in n.keywords}
    raise AssertionError(f"{flag} not found in main()'s argparse")


def test_send_hz_argparse_shape_and_help():
    src = _main_src()
    assert '"--send-hz", type=float, default=None' in src
    kw = _add_argument_kwargs("--send-hz")
    assert kw["default"].value is None
    assert kw["type"].id == "float"
    help_text = kw["help"].value
    assert help_text.startswith("(L2)"), help_text[:40]
    # the two things the plan requires the help to carry: the lag cost, and
    # the numeric teleport-delay bound (P + one loop interval = 75 ms at H=20)
    assert "LAG" in help_text or "lag" in help_text
    assert "75" in help_text
    # it must not promise a snap for teleports — only reset is flushed
    assert "RESET" in help_text


def test_non_positive_send_hz_is_a_parser_error():
    src = _main_src()
    blk = src.split("if args.send_hz is not None and args.send_hz <= 0.0:")[1]
    assert blk.lstrip().startswith("parser.error(")


def test_l3_warns_instead_of_silently_ignoring_the_flag():
    src = _main_src()
    multi = src.split("if args.multi_env:")[1].split("run_l3(args)")[0]
    assert "--send-hz" in multi
    warn = [ln for ln in multi.splitlines() if "--send-hz" in ln][0]
    assert "[WARN]" in multi.split("--send-hz")[0].splitlines()[-1] or "[WARN]" in warn
    # the token discipline: [SERVE]/[STATS] mark counter/timing lines only
    # `str.index` raises ValueError on a missing anchor, so this slice fails
    # LOUDLY rather than widening — no presence guard is needed here, unlike
    # the `str.split` sites elsewhere in this file. (A dead `seg` variable
    # computed by rsplit lived here from v1.6.7 until v1.6.8; nothing read it,
    # and the comment above it claimed to protect a check that uses `block`.)
    block = multi[multi.index("if args.send_hz"):]
    assert "[SERVE]" not in block and "[STATS]" not in block


def test_demotion_is_judged_against_the_pacing_dt_not_the_engine_dt():
    """Send opportunities are produced by the catch-up loop, which consumes
    time in SIM_DT (= sim_dt) units. Judging the demotion against
    ``_eff_dt`` would let an H between the two slip through and then send on
    every loop while the banner still claimed H. ``_eff_dt`` is the banner's
    physics LABEL only."""
    src = _main_src()
    # both anchors must exist: either split falling through would silently
    # widen `region` to the rest of main() and weaken every assertion below
    assert '_send_hz = getattr(args, "send_hz", None)' in src
    assert "SIM_DT = sim_dt" in src
    region = src.split('_send_hz = getattr(args, "send_hz", None)')[1] \
                .split("SIM_DT = sim_dt")[0]
    demote = [ln for ln in region.splitlines()
              if "_send_hz is not None and _send_hz >=" in ln]
    assert len(demote) == 1
    assert "sim_dt" in demote[0] and "_eff_dt" not in demote[0]
    # a non-positive H is demoted too (unreachable via argparse, but a
    # hand-built args namespace would divide by zero one line later)
    assert "_send_hz is not None and _send_hz <= 0.0" in region
    # the banner's physics-Hz label uses the engine dt
    assert "1.0 / _eff_dt" in region
    assert "[Send-Rate]" in region
    # the demotion actually turns the gate off, it does not merely warn
    assert "_send_hz = None" in region
    # ...and OFF is decided by `is not None`, never by truthiness: 0.0 is a
    # REJECTED value, not "not asked for", and the banner must not conflate
    # the two.
    assert "if _send_hz is not None else None)" in region
    assert "if _send_hz:" not in region


def test_banner_is_printed_in_both_arms():
    """A flag whose effect you cannot see in the log is a flag you cannot
    A/B. The [Send-Rate] line is unconditional."""
    src = _main_src()
    # both anchors must exist: either split falling through would silently
    # widen `region` to the rest of main() and weaken every assertion below
    assert '_send_hz = getattr(args, "send_hz", None)' in src
    assert "SIM_DT = sim_dt" in src
    region = src.split('_send_hz = getattr(args, "send_hz", None)')[1] \
                .split("SIM_DT = sim_dt")[0]
    # the banner print sits at region top level (no leading `if` indent deeper
    # than the surrounding function body)
    banner = [ln for ln in region.splitlines()
              if 'print(f" [Pacing] [Send-Rate] physics' in ln]
    assert len(banner) == 1
    assert banner[0].startswith("    print("), banner[0]
    assert "매 루프" in region                  # the OFF label
    assert "--send-hz 미지정" in region
    # ...and a DEMOTED run is not labelled "기본값": the operator passed the
    # flag, it is simply not in effect.
    assert "_send_hz_req" in region
    assert "강등됨(OFF)" in region


# --------------------------------------------------------------------------
# [SERVE] counters
# --------------------------------------------------------------------------

def test_new_serve_counters_are_initialised_printed_and_reset():
    src = _main_src()
    assert _WINDOW_TRIGGER in src, \
        "the window trigger was renamed; the splits below would be vacuous"
    for n in ("log_sends", "log_send_skips", "log_send_flushes"):
        assert src.count(n) >= 3, n
        assert f"{n} = 0" in src
        assert f"{n} = 0" in src.split(_WINDOW_TRIGGER)[-1], \
            f"{n} is not reset on the window boundary"
    # the window origin is carried, not re-read, so windows do not gap
    assert "_win_t0 = time.perf_counter()" in src
    assert "_win_t0 = _win_now" in src.split(_WINDOW_TRIGGER)[-1]


def test_window_ms_is_measured_to_a_clock_read_after_the_loop_body():
    """INTENT (unchanged since v1.6.7; the mechanism changed in v1.6.8).

    ``window_ms`` must end at a stamp taken AFTER this loop's work, so the
    loop that closes the window is inside the window. Reusing the loop's
    ``now`` — bound at the TOP of the loop, and only in the non-lockstep
    branch — would drop that loop and shorten the window by ~2%, biasing the
    derived effective send rate UPWARD, i.e. in exactly the direction that
    would hide a rate regression.

    v1.6.7 satisfied this with a dedicated ``_win_now = time.perf_counter()``
    inside the window block. v1.6.8 needs the same stamp to TRIGGER the
    window, so it is read once per loop as ``_t_end`` right after the loop
    body, and ``_win_now`` binds it — no extra clock read, and the property
    is preserved because ``_t_end``'s read still sits after the loop's work.
    This test therefore pins the ORDERING, not the identity of the call site.

    WHAT IS ENFORCED, exactly (an upper bound alone is not enough — a
    ``_t_end`` hoisted to the top of the loop body satisfies "before the
    counters" and reintroduces the defect, with ``loop_dur`` collapsing to
    ~0 and ``serve_ms`` going NEGATIVE):

      * ``_t_end`` is a DIRECT statement of the ``while True:`` body, not
        nested in a branch that some loops skip;
      * its statement INDEX is greater than that of ``loop_start`` (lower
        bound), of the catch-up step loop, and of every OSC state-send
        statement (the loop's work);
      * and less than that of the ``log_count`` increment and the window
        block (upper bound).
    """
    main_node = _main_ast()
    body = _main_loop(main_node).body

    # `_t_end = time.perf_counter()` is read once, as a direct statement of
    # the loop body (a read inside a branch would skip whole loops).
    t_end = [n for n in ast.walk(main_node)
             if isinstance(n, ast.Assign)
             and any(getattr(t, "id", "") == "_t_end" for t in n.targets)]
    assert len(t_end) == 1, "expected exactly one _t_end clock read"
    assert "perf_counter" in ast.dump(t_end[0].value)
    i_te = _stmt_index(body, _assign_to("_t_end"))
    assert len(i_te) == 1, "_t_end is not a direct statement of the loop body"
    i_te = i_te[0]

    # LOWER bounds — the stamp follows the loop's work.
    i_ls = _stmt_index(body, _assign_to("loop_start"))
    assert i_ls == [0], "loop_start is not the first statement of the loop"
    assert i_te > 0, "_t_end is hoisted to the top of the loop: loop_dur " \
                     "would collapse to ~0 and serve_ms would go negative"
    i_send = _stmt_index(body, _contains_state_send)
    assert i_send, "no OSC state send found in the loop body"
    assert i_te > max(i_send), \
        "_t_end is stamped before the state send — the send's cost would " \
        "fall outside loop_dur and out of serve_ms"
    i_step = _stmt_index(
        body, lambda st: any(isinstance(c, ast.AugAssign)
                             and getattr(c.target, "id", "") == "catchup_steps"
                             for c in ast.walk(st)))
    assert i_step and i_te > max(i_step), \
        "_t_end is stamped before the catch-up steps"

    # UPPER bounds — the stamp precedes the counters it feeds.
    incs = [n for n in ast.walk(main_node)
            if isinstance(n, ast.AugAssign)
            and getattr(n.target, "id", "") == "log_count"]
    assert len(incs) == 1
    assert t_end[0].lineno < incs[0].lineno, \
        "_t_end must be read before the window counters, i.e. after the body"

    blocks = [n for n in ast.walk(main_node)
              if isinstance(n, ast.If)
              and "_win_t0" in ast.dump(n.test)
              and "_stats_interval" in ast.dump(n.test)]
    assert len(blocks) == 1, "expected exactly one wall-clock window block"
    # ...and the window block runs AFTER the increment, which is what makes
    # log_count >= 1 and the `/ _n` denominators inside it division-safe.
    assert incs[0].lineno < blocks[0].lineno

    assigns = [n for n in ast.walk(blocks[0])
               if isinstance(n, ast.Assign)
               and any(getattr(t, "id", "") == "window_ms" for t in n.targets)]
    assert len(assigns) == 1
    expr = ast.dump(assigns[0])
    assert "_win_now" in expr and "_win_t0" in expr
    assert "'now'" not in expr and "Name('now'" not in expr
    fresh = [n for n in ast.walk(blocks[0])
             if isinstance(n, ast.Assign)
             and any(getattr(t, "id", "") == "_win_now" for t in n.targets)]
    assert len(fresh) == 1 and "_t_end" in ast.dump(fresh[0].value)


def test_serve_line_carries_the_four_new_keys_and_stays_parseable():
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm
    from genesis_vehicle.server import benchmark_collision as bc
    src = _main_src()
    for key in ("sends=", "send_skips=", "send_flushes=", "window_ms="):
        assert f'f"{key}' in src or f'{key}{{' in src, key
    line = (" [SERVE] [L2] recv_loops=50 nonskip_loops=0 skipped_captures=50 "
            "post_step_captures=50 post_step_captures_ref=50 sends=25 "
            "send_skips=25 send_flushes=1 loops=50 steps=50 "
            "zero_step_loops=0 serve_ms=312.500 phys_ms_sum=687.500 "
            "window_ms=1252.4\n")
    assert bm._SERVE_RE.search(line)
    kv = {k: float(v) for k, v in bm._SERVE_KV_RE.findall(line)}
    assert {"sends", "send_skips", "send_flushes", "window_ms"} <= set(kv)
    # the identities the counters are built to make decidable. Since v1.6.8
    # the window is a wall-clock interval, so the right-hand side is the
    # window's OWN loop count, not the constant 50 it used to be.
    assert kv["sends"] + kv["send_skips"] == kv["loops"]
    assert kv["send_flushes"] <= kv["sends"]
    # ...and the new keys must not disturb either frozen [STATS] parser
    assert bm._STATS_RE.search(line) is None
    assert bc._STATS_RE.search(line) is None


def test_the_identity_comment_names_the_path_it_holds_on():
    """CLAUDE.md: a guarantee may only be stated unqualified if it holds on
    every reachable path. `sends + send_skips == loops` holds on the
    NON-LOCKSTEP path only — both counters sit inside `if not args.lockstep:`
    while log_count does not. Lockstep is unreachable today, so no user can
    see an unbalanced line, but the sentence is scheduled to be copied into
    docs/server.md, and the unqualified version must not be what gets
    copied."""
    src = _main_src()
    ident = [ln for ln in src.splitlines()
             if "sends + send_skips == loops" in ln]
    assert ident, "the identity is not documented at all"
    # ...and the pre-v1.6.8 form, which the wall-clock window made false,
    # must be gone: a window no longer holds a constant 50 loops.
    assert "sends + send_skips == 50" not in src
    where = src.split("sends + send_skips == loops")[0]
    # the qualification sits on the comment lines immediately around it
    near = "\n".join(where.splitlines()[-3:]) + "\n".join(
        src.split("sends + send_skips == loops")[1].splitlines()[:8])
    assert "LOCKSTEP" in near.upper(), near


def test_stats_fstring_gained_no_send_tokens():
    """[STATS] is frozen: two separate _STATS_RE copies parse it."""
    src = _main_src()
    stats_print = src.split('f" [STATS] [L2]')[1].split("[cap=")[0]
    for tok in ("sends", "send_skips", "send_flushes", "window_ms", "[SERVE]"):
        assert tok not in stats_print, tok


def test_serve_keys_are_unconditional_unlike_cap_override_us():
    """cap_override_us is appended under `if _serve_timers:`; the four send
    keys must be in the base f-string, or an A/B arm without --serve-timers
    would lose them."""
    src = _main_src()
    # `.split(")\n")` is correct only while no line of the f-string ends in
    # `")` — assert the anchors and that the slice really stopped inside the
    # [SERVE] statement, rather than swallowing the rest of the block
    assert '_serve = (f" [SERVE] [L2]' in src
    base = src.split('_serve = (f" [SERVE] [L2]')[1].split(")\n")[0]
    assert base and "print(" not in base, base
    for key in ("sends=", "send_skips=", "send_flushes=", "window_ms="):
        assert key in base, key
    assert "cap_override_us" not in base


# ==========================================================================
# 3. benchmark forwarding
# ==========================================================================

def test_benchmark_forwards_send_hz_and_records_it():
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm
    sig = inspect.signature(bm.run_config).parameters
    assert "send_hz" in sig and sig["send_hz"].default is None
    src = inspect.getsource(bm.run_config)
    assert 'cmd += ["--send-hz", str(send_hz)]' in src
    assert "send_hz=send_hz" in src
    msrc = inspect.getsource(bm.main)
    assert '"--send-hz", type=float, default=None' in msrc
    assert "send_hz=args.send_hz" in msrc
    # exposed in the summary table next to the switch count
    assert "send_hz" in msrc.split("| mode | terrain")[1][:400]
    assert "pacing_switches" in inspect.getsource(bm)


def test_benchmark_default_arm_passes_no_flag():
    """The A arm must be the untouched server command line."""
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm
    src = inspect.getsource(bm.run_config)
    before = [ln.strip() for ln
              in src.split('cmd += ["--send-hz", str(send_hz)]')[0].splitlines()
              if ln.strip()]
    # `is not None`, not truthiness: send_hz=0.0 must be FORWARDED (and then
    # rejected loudly by the server's argparse), never silently dropped while
    # the record still reports send_hz=0.0.
    assert before[-1] == "if send_hz is not None:"


def test_benchmark_replay_still_parses_with_the_new_serve_keys(monkeypatch):
    """The reader sums [SERVE] keys per window; the four --send-hz keys and
    the five v1.6.8 window-shape keys are all summable, so a canned stdout
    carrying them must still produce aligned windows and correct sums — and
    the effective send rate must still come out of Σsends / Σwindow_ms."""
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm

    class _FakeProc:
        returncode = 0

        def __init__(self, lines):
            self.stdout = iter(lines)

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def send_init_burst(self):
            pass

        def start_inputs(self):
            pass

        def stop(self):
            pass

    serve = (" [SERVE] [L2] recv_loops=50 nonskip_loops=0 skipped_captures=50 "
             "post_step_captures=50 post_step_captures_ref=50 sends=25 "
             "send_skips=25 send_flushes=0 loops=50 steps=50 "
             "zero_step_loops=0 serve_ms=125.000 phys_ms_sum=125.000 "
             "window_ms=1250.0\n")
    lines = [f"{bm._TARGET_TOKEN}0\n",
             " [Genesis] Initialization Complete. Obstacles: 0\n"]
    for loop, phys in ((9.00, 5.00), (4.00, 2.50), (6.00, 3.50)):
        lines.append(f" [STATS] [L2] Loop Avg: {loop:.2f} ms | "
                     f"Physics Avg: {phys:.2f} ms "
                     f"(1.0 steps/loop, 2.50 ms/step) [cap=5:burst]\n")
        lines.append(serve)

    monkeypatch.setattr(bm.subprocess, "Popen", lambda *a, **kw: _FakeProc(lines))
    monkeypatch.setattr(bm, "MockUEClient", _FakeClient)
    r = bm.run_config("L2", "simple", 1, "/tmp/tank.urdf", "/tmp/hull.obj",
                      False, 3, 30.0, "python", False, send_hz=20.0)
    assert r["ok"] is True
    assert r["send_hz"] == 20.0
    assert r["serve_windows"] == 2 and r["serve_windows_match"] is True
    assert r["serve_sum"]["sends"] == 50
    assert r["serve_sum"]["send_skips"] == 50
    assert r["serve_sum"]["window_ms"] == pytest.approx(2500.0)
    # the effective send rate the refuse rule uses
    eff = 1000.0 * r["serve_sum"]["sends"] / r["serve_sum"]["window_ms"]
    assert eff == pytest.approx(20.0)
    # v1.6.8: with the window keys present the headline averages are POOLED
    # over loops. Both windows are 50 loops here, so the values match the
    # per-window means, and the record says which estimator produced them.
    assert r["weighting"] == "loops"
    assert r["loops_total"] == 100
    # 2 windows x (125 ms serve + 125 ms phys) over 2 x 50 loops
    assert r["loop_ms"] == pytest.approx(5.0)
    assert r["phys_ms"] == pytest.approx(2.5)
    assert r["serving_ms"] == pytest.approx(2.5)
    assert r["duty"] == pytest.approx(250.0 / 2500.0)     # D = 0.1
    assert r["s_prime"] == pytest.approx(2.5)             # 250 ms / 100 steps
