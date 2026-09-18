"""v1.6.8: the ``[STATS]`` / ``[SERVE]`` reporting window is a fixed
WALL-CLOCK interval (``--stats-interval``, default 1.0 s) in BOTH server
modes, replacing the fixed 50-LOOP window, plus the benchmark re-weighting
that the variable loop count forces.

| abbr | meaning |
|---|---|
| L2 | per-entity server mode (K interacting vehicles, ``n_envs=1``) |
| L3 | multi-env server mode (``--multi-env``: 1 entity x ``n_envs``) |
| STATS | the server's frozen per-window timing stdout line |
| SERVE | the server's per-window serving-counter stdout line |
| AST | Abstract Syntax Tree |
| dt | physics step duration (``SIM_DT``) |
| D | serving duty cycle = Sum(``serve_ms``) / Sum(``window_ms``) |
| S' | serving ms per physics step = Sum(``serve_ms``) / Sum(``steps``) |

WHY the trigger moved. The server busy-waits through the last <= 2 ms before
each step deadline, so on a server with headroom most loops take no physics
step at all. Fifty of those close in ~0.1 ms of wall clock, and every
per-window statistic then describes ~0.1 ms of nothing while ``[STATS]``'s
``.2f`` / ``.1f`` rounding prints ``0.00`` / ``0.0``. Measured on the v1.6.7
tree at ``--send-hz 20``: median ``window_ms`` 0.0-0.1, and
``run_config --stats 5`` returned having observed ~25 ms of simulation
(against ~4.0 s after this change). The WINDOW COUNT of that arm is NOT a
stable figure — repeats of the identical harness on this machine spread over
roughly 2.0k-2.7k windows per 8 s — so no single count is quoted here.

WHAT THIS FILE DOES NOT CLAIM. It does not claim the printed ``Loop Avg`` /
``steps/loop`` became meaningful on a server with headroom — they are still
diluted by the spin loops, and no window length fixes that. It pins the
window MECHANISM, the frozen ``[STATS]`` string, the new summable keys, and
the benchmark's pooled weighting.

Pure Python: no scene is built and no server process is started. The live
counterpart is ``test_server_stats_window_live.py``, gated behind
``GENESIS_VEHICLE_LIVE_SERVER_TEST``.
"""
import ast
import inspect
import textwrap

import pytest

# The exact source text of the wall-clock window trigger in both loops. Every
# split-on-the-boundary assertion checks it is PRESENT first: `str.split` on
# an absent separator returns the whole string, which is how the pre-existing
# reset assertions stayed green when the trigger changed under them.
_WINDOW_TRIGGER = "if _t_end - _win_t0 >= _stats_interval:"

# The five window-shape keys shared by the L2 and L3 [SERVE] lines. All are
# counts or sums, i.e. SUMMABLE across windows: server/benchmark.py adds
# [SERVE] keys together, so a ratio or an average would be meaningless there.
_SHARED_KEYS = ("loops", "steps", "zero_step_loops", "serve_ms", "phys_ms_sum")

# Keys that exist only on the L2 path. They must be ABSENT from the L3 line,
# not zero: the reader sums BY KEY NAME, so `sends=0` from L3 would read as
# "L3 sent state zero times" when L3 sends every loop.
_L2_ONLY_KEYS = ("recv_loops", "nonskip_loops", "skipped_captures",
                 "post_step_captures", "post_step_captures_ref",
                 "sends", "send_skips", "send_flushes", "cap_override_us")


def _ps():
    pytest.importorskip("genesis")
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import physics_server
    return physics_server


def _l3():
    pytest.importorskip("genesis")
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import l3_runtime
    return l3_runtime


def _src(fn):
    return textwrap.dedent(inspect.getsource(fn))


def _node(fn):
    return ast.parse(_src(fn)).body[0]


def _loops():
    """(name, source, AST) for both server loops."""
    return [("L2", _src(_ps().main), _node(_ps().main)),
            ("L3", _src(_l3().run_l3), _node(_l3().run_l3))]


# The OSC calls that ship this loop's state to the client. `_t_end` must be
# stamped after all of them — that is what "after the loop's work" means.
# Deliberately NOT `send_message`: the window block sends
# /Genesis/Init/TimeDilation with it, AFTER the stamp, by design.
_STATE_SEND_CALLS = ("send_target_states_bulk", "send_dynamic_states_bulk",
                     "send_sim_time", "send_step_ack")


def _main_loop(node):
    """The `while True:` whose body carries the window counters (both loops
    also contain an init-wait `while True:`, which this excludes)."""
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


def _window_blocks(node):
    return [n for n in ast.walk(node)
            if isinstance(n, ast.If)
            and "_win_t0" in ast.dump(n.test)
            and "_stats_interval" in ast.dump(n.test)]


def _render(joined: ast.JoinedStr) -> str:
    """Reconstruct an f-string's TEMPLATE: literal text verbatim, and every
    placeholder as ``{expr:spec}``. Format specs are part of the contract —
    ``.2f`` / ``.1f`` are exactly what destroy the headroom numbers."""
    out = []
    for part in joined.values:
        if isinstance(part, ast.Constant):
            out.append(part.value)
        else:
            spec = ""
            if part.format_spec is not None:
                spec = ":" + "".join(
                    c.value for c in part.format_spec.values
                    if isinstance(c, ast.Constant))
            out.append("{" + ast.unparse(part.value) + spec + "}")
    return "".join(out)


def _printed_lines(node, marker):
    """Rendered templates of every print() whose f-string starts with marker."""
    found = []
    for n in ast.walk(node):
        if not isinstance(n, ast.JoinedStr):
            continue
        head = n.values[0]
        if isinstance(head, ast.Constant) and head.value.lstrip().startswith(marker):
            found.append(_render(n))
    return found


# --------------------------------------------------------------------------
# 1-4: the trigger, the denominators and the division-by-zero invariant
# --------------------------------------------------------------------------

def test_window_trigger_is_wall_clock_not_a_loop_count():
    for name, src, node in _loops():
        blocks = _window_blocks(node)
        assert len(blocks) == 1, f"{name}: expected exactly one window block"
        test_dump = ast.dump(blocks[0].test)
        assert "log_count" not in test_dump, \
            f"{name}: the window still triggers on the loop count"
        assert "50" not in test_dump, f"{name}: a literal 50 survives in the test"
        assert _WINDOW_TRIGGER in src, name


def test_no_literal_50_denominator_remains():
    for name, src, _node in _loops():
        assert "/ 50.0" not in src, f"{name}: a hard-coded 50-loop denominator"
        assert "log_count >= 50" not in src, name


def test_window_block_runs_after_the_log_count_increment():
    """The ONLY thing standing between `/ _n` and a ZeroDivisionError is the
    order: the window test sits below `log_count += 1`, so log_count >= 1
    whenever the block runs. Pinned by line number in both loops."""
    for name, _src, node in _loops():
        incs = [n for n in ast.walk(node)
                if isinstance(n, ast.AugAssign)
                and getattr(n.target, "id", "") == "log_count"]
        assert len(incs) == 1, name
        block = _window_blocks(node)[0]
        assert incs[0].lineno < block.lineno, \
            f"{name}: the window block runs BEFORE log_count is incremented"


def test_denominators_use_the_window_loop_count():
    """avg_loop / avg_phys / steps_per_loop / avg_loop_sec divide by the
    window's own loop count `_n`; per_step keeps dividing by the STEP count
    (guarded by max(log_step_sum, 1), because a window can legitimately
    contain zero physics steps)."""
    for name, _src, node in _loops():
        block = _window_blocks(node)[0]
        got = {}
        for n in ast.walk(node):
            if isinstance(n, ast.Assign) and len(n.targets) == 1:
                tid = getattr(n.targets[0], "id", "")
                if tid in ("avg_loop", "avg_phys", "steps_per_loop",
                           "per_step", "avg_loop_sec", "_n"):
                    got[tid] = ast.dump(n.value)
        assert "_n" in got and "log_count" in got["_n"], name
        for key in ("avg_loop", "avg_phys", "steps_per_loop", "avg_loop_sec"):
            assert key in got, f"{name}: {key} missing"
            assert "'_n'" in got[key], f"{name}: {key} does not divide by _n"
        assert "log_step_sum" in got["per_step"] and "max" in got["per_step"], \
            f"{name}: per_step must stay a per-STEP average with its guard"
        assert block is not None


# --------------------------------------------------------------------------
# 5-6: the [STATS] line is frozen
# --------------------------------------------------------------------------

_L2_STATS = (" [STATS] [L2] Loop Avg: {avg_loop:.2f} ms | "
             "Physics Avg: {avg_phys:.2f} ms "
             "({steps_per_loop:.1f} steps/loop, {per_step:.2f} ms/step) "
             "[cap={pacer.cap()}:{pacer.mode}]")
_L3_STATS = (" [STATS] [L3 n_envs={n_envs}] Loop Avg: {avg_loop:.2f} ms | "
             "Physics Avg: {avg_phys:.2f} ms "
             "({steps_per_loop:.1f} steps/loop, {per_step:.2f} ms/step) "
             "[cap={pacer.cap()}:{pacer.mode}]")


def test_stats_fstring_is_byte_identical_to_the_pre_change_string():
    """Two independent `_STATS_RE` copies parse this line (benchmark.py and
    benchmark_collision.py), and a UE-side fork may read it too, so v1.6.8
    changed the window TRIGGER and left the string alone. The expected values
    below are the pre-change templates, transcribed.

    NB the L3 tag KEEPS `n_envs=` — `_STATS_RE` reads four numeric fields and
    never sees the tag. The [SERVE] line is the one that must not carry it
    (test_l3_serve_tag_is_bare_and_yields_no_stray_keys)."""
    l2 = _printed_lines(_node(_ps().main), "[STATS]")
    l3 = _printed_lines(_node(_l3().run_l3), "[STATS]")
    assert l2 == [_L2_STATS], l2
    assert l3 == [_L3_STATS], l3


def test_both_stats_regex_copies_still_match_and_ignore_serve():
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm
    from genesis_vehicle.server import benchmark_collision as bc
    stats_l2 = (" [STATS] [L2] Loop Avg: 25.04 ms | Physics Avg: 7.81 ms "
                "(1.0 steps/loop, 7.81 ms/step) [cap=5:smooth]\n")
    stats_l3 = (" [STATS] [L3 n_envs=100] Loop Avg: 25.04 ms | "
                "Physics Avg: 7.81 ms (1.0 steps/loop, 7.81 ms/step) "
                "[cap=5:smooth]\n")
    serve_l2 = (" [SERVE] [L2] recv_loops=40 nonskip_loops=0 "
                "skipped_captures=40 post_step_captures=40 "
                "post_step_captures_ref=40 sends=20 send_skips=20 "
                "send_flushes=0 loops=40 steps=40 zero_step_loops=0 "
                "serve_ms=312.500 phys_ms_sum=687.500 window_ms=1000.0\n")
    serve_l3 = (" [SERVE] [L3] loops=40 steps=40 zero_step_loops=0 "
                "serve_ms=312.500 phys_ms_sum=687.500 window_ms=1000.0\n")
    for rx in (bm._STATS_RE, bc._STATS_RE):
        assert rx.search(stats_l2)
        assert rx.search(stats_l3)
        assert rx.search(serve_l2) is None
        assert rx.search(serve_l3) is None


# --------------------------------------------------------------------------
# 7-10: the [SERVE] lines
# --------------------------------------------------------------------------

def test_l3_serve_tag_is_bare_and_yields_no_stray_keys():
    """`_SERVE_KV_RE` harvests `key=value` from ANYWHERE on the line, so an
    `[L3 n_envs=100]` tag (the shape [STATS] uses) would put `n_envs` into the
    benchmark's summed metrics dict — it would be added up across windows as
    if it were a counter. The [SERVE] tag is therefore bare `[L3]`."""
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm
    lines = _printed_lines(_node(_l3().run_l3), "[SERVE]")
    assert len(lines) == 1, lines
    line = lines[0]
    assert line.lstrip().startswith("[SERVE] [L3] "), line
    assert "n_envs" not in line, line
    # ...and the same check against the parser, on a rendered line.
    rendered = (" [SERVE] [L3] loops=40 steps=40 zero_step_loops=0 "
                "serve_ms=312.500 phys_ms_sum=687.500 window_ms=1000.0\n")
    kv = {k: float(v) for k, v in bm._SERVE_KV_RE.findall(rendered)}
    assert set(kv) == set(_SHARED_KEYS) | {"window_ms"}
    assert "n_envs" not in kv


def test_serve_lines_carry_the_shared_window_keys_in_both_modes():
    """Both modes emit the same five names with the same meaning, and on L2
    they sit in the BASE f-string — not behind `if _serve_timers:`, which is
    where cap_override_us lives. An arm run without --serve-timers must still
    carry the weights, or the benchmark silently falls back to the unweighted
    estimator."""
    l2_lines = _printed_lines(_node(_ps().main), "[SERVE]")
    l3_lines = _printed_lines(_node(_l3().run_l3), "[SERVE]")
    assert len(l2_lines) == 1 and len(l3_lines) == 1
    for name, line in (("L2", l2_lines[0]), ("L3", l3_lines[0])):
        for key in _SHARED_KEYS:
            assert f"{key}=" in line, f"{name} line lacks {key}="
        assert "window_ms=" in line, name
    # the L2 base f-string is the one assigned to `_serve`, before the
    # conditional cap_override_us append. (Split on the LAST _serve_timers
    # gate: the earlier ones wrap the capture timer, not the print.)
    src = _src(_ps().main)
    parts = src.split("if _serve_timers:")
    assert len(parts) >= 2, "the _serve_timers gate disappeared"
    base, tail = "if _serve_timers:".join(parts[:-1]), parts[-1]
    assert "cap_override_us" in tail
    assert "cap_override_us" not in _printed_lines(_node(_ps().main), "[SERVE]")[0]
    for key in _SHARED_KEYS:
        assert f"{key}={{" in base, f"{key} is not in the base [SERVE] f-string"


def test_serve_identities():
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm
    line = (" [SERVE] [L2] recv_loops=37 nonskip_loops=5 skipped_captures=32 "
            "post_step_captures=41 post_step_captures_ref=40 sends=20 "
            "send_skips=20 send_flushes=0 loops=40 steps=40 "
            "zero_step_loops=3 serve_ms=312.500 phys_ms_sum=687.500 "
            "window_ms=1000.0\n")
    kv = {k: float(v) for k, v in bm._SERVE_KV_RE.findall(line)}
    assert kv["sends"] + kv["send_skips"] == kv["loops"]
    assert kv["skipped_captures"] + kv["nonskip_loops"] == kv["recv_loops"]
    assert kv["zero_step_loops"] <= kv["loops"]
    assert kv["phys_ms_sum"] >= 0.0
    # serve_ms >= 0 is an invariant, not a formality: serve_ms + phys_ms_sum
    # = Sum(loop_dur)*1000 and phys_ms_sum <= Sum(loop_dur), so a NEGATIVE
    # serve_ms means loop_dur was measured from a stamp taken before the
    # loop's work (see test_win_now_binds_t_end_read_after_the_loop_body).
    assert kv["serve_ms"] >= 0.0
    # serve_ms <= window_ms is PROVEN, not assumed: the loops'
    # [loop_start, _t_end] spans are disjoint (one thread, sequential) and all
    # lie inside [_win_t0, _win_now], so Sum(loop_dur) <= the window; sleep and
    # the spin loops that `continue` add to window_ms only.
    assert kv["serve_ms"] <= kv["window_ms"]
    # and the construction identity the benchmark rebuilds Loop Avg from
    assert kv["serve_ms"] + kv["phys_ms_sum"] == pytest.approx(1000.0)


def test_l2_only_keys_are_absent_from_the_l3_line():
    line = _printed_lines(_node(_l3().run_l3), "[SERVE]")[0]
    for key in _L2_ONLY_KEYS:
        assert key not in line, \
            f"{key} must be ABSENT from the L3 line, not printed as 0 " \
            f"(benchmark.py sums by key name)"


# --------------------------------------------------------------------------
# 11-13: the new counter and the window origin
# --------------------------------------------------------------------------

def test_zero_step_loops_counter_is_initialised_incremented_and_reset():
    for name, src, node in _loops():
        assert src.count("log_zero_step_loops") >= 3, name
        assert "log_zero_step_loops = 0" in src, name
        assert _WINDOW_TRIGGER in src, name
        assert "log_zero_step_loops = 0" in src.split(_WINDOW_TRIGGER)[-1], \
            f"{name}: zero_step_loops is not reset on the window boundary"
        incs = [n for n in ast.walk(node)
                if isinstance(n, ast.AugAssign)
                and getattr(n.target, "id", "") == "log_zero_step_loops"]
        assert len(incs) == 1, name
        # incremented exactly on the zero-step branch
        guards = [n for n in ast.walk(node)
                  if isinstance(n, ast.If)
                  and "catchup_steps" in ast.dump(n.test)
                  and any(g is incs[0] for g in ast.walk(n))]
        assert guards, f"{name}: the increment is not guarded by catchup_steps"


def test_window_origin_is_carried_not_re_read():
    """Consecutive windows must TILE the wall clock: the next window's origin
    is the stamp that closed the previous one, so no time falls between them
    and Sum(window_ms) equals the measured span. Re-reading the clock here
    would leak the print + TimeDilation cost out of every window."""
    for name, src, node in _loops():
        assert _WINDOW_TRIGGER in src, \
            f"{name}: the window trigger was renamed; the split is vacuous"
        tail = src.split(_WINDOW_TRIGGER)[-1]
        assert ("_win_t0 = _win_now" in tail) or ("_win_t0 = _t_end" in tail), \
            f"{name}: the window origin is not carried"
        assert "_win_t0 = time.perf_counter()" not in tail, \
            f"{name}: the origin is re-read inside the window block"
        # exactly one fresh read, at loop setup
        assert src.count("_win_t0 = time.perf_counter()") == 1, name
        assert node is not None


def test_win_now_binds_t_end_read_after_the_loop_body():
    """v1.6.7 read a dedicated `_win_now = time.perf_counter()` inside the
    window block; v1.6.8 needs that stamp to TRIGGER the window, so `_t_end`
    is read once per loop right after the loop body and reused. The property
    being preserved is the ORDER (the stamp follows the loop's work, so the
    closing loop is inside its window), not the call site — and it costs no
    extra clock read, which matters because the spin loop is what is being
    measured.

    WHAT IS ENFORCED, exactly. An upper bound alone does NOT pin this: a
    `_t_end` hoisted to the top of the loop body still sits before the
    counters and before the window block, and it is precisely the "reuses a
    stamp taken before the loop's work" defect — `loop_dur` collapses to ~0,
    `Loop Avg` prints 0.00 and `serve_ms = (Sum loop_dur - Sum phys) * 1000`
    goes NEGATIVE. So both bounds are asserted, by statement INDEX within the
    `while True:` body:

      * `_t_end` is a DIRECT statement of the loop body (not nested in a
        branch that some loops skip);
      * after `loop_start` (which must be the loop's first statement), after
        the catch-up step loop, and after every OSC state-send statement;
      * before the `log_count` increment and before the window block.

    The runtime backstop for the same class is `serve_ms >= 0`, asserted on
    real windows in test_server_stats_window_live.py."""
    for name, _src, node in _loops():
        t_end = [n for n in ast.walk(node)
                 if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", "") == "_t_end" for t in n.targets)]
        assert len(t_end) == 1, name
        assert "perf_counter" in ast.dump(t_end[0].value), name
        block = _window_blocks(node)[0]
        assert t_end[0].lineno < block.lineno, name

        body = _main_loop(node).body
        i_te = _stmt_index(body, _assign_to("_t_end"))
        assert len(i_te) == 1, \
            f"{name}: _t_end is not a direct statement of the loop body"
        i_te = i_te[0]
        assert _stmt_index(body, _assign_to("loop_start")) == [0], \
            f"{name}: loop_start is not the loop's first statement"
        assert i_te > 0, \
            f"{name}: _t_end is hoisted to the top of the loop — loop_dur " \
            f"collapses to ~0 and serve_ms goes negative"
        i_send = _stmt_index(body, _contains_state_send)
        assert i_send, f"{name}: no OSC state send in the loop body"
        assert i_te > max(i_send), \
            f"{name}: _t_end is stamped before the state send, so the " \
            f"send's cost falls outside loop_dur and out of serve_ms"
        i_step = _stmt_index(
            body, lambda st: any(isinstance(c, ast.AugAssign)
                                 and getattr(c.target, "id", "") == "catchup_steps"
                                 for c in ast.walk(st)))
        assert i_step and i_te > max(i_step), \
            f"{name}: _t_end is stamped before the catch-up steps"
        i_inc = _stmt_index(
            body, lambda st: isinstance(st, ast.AugAssign)
            and getattr(st.target, "id", "") == "log_count")
        assert i_inc and i_te < min(i_inc), name

        # loop_dur is derived from the same stamp: no second clock read
        loop_dur = [n for n in ast.walk(node)
                    if isinstance(n, ast.Assign)
                    and any(getattr(t, "id", "") == "loop_dur" for t in n.targets)]
        assert len(loop_dur) == 1, name
        dump = ast.dump(loop_dur[0].value)
        assert "_t_end" in dump and "perf_counter" not in dump, name
    # L2 additionally keeps the `_win_now` name that window_ms is built from
    l2 = _node(_ps().main)
    win_now = [n for n in ast.walk(l2)
               if isinstance(n, ast.Assign)
               and any(getattr(t, "id", "") == "_win_now" for t in n.targets)]
    assert len(win_now) == 1
    assert "_t_end" in ast.dump(win_now[0].value)
    assert "perf_counter" not in ast.dump(win_now[0].value)


# --------------------------------------------------------------------------
# 14-15: the flag and its forwarding
# --------------------------------------------------------------------------

class _Args:
    """Hand-built args namespace — the path that never reaches argparse
    (a test harness or an embedder calling main()/run_l3() directly)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_non_positive_stats_interval_falls_back_to_the_default():
    """`getattr(args, "stats_interval", 1.0) or 1.0` is NOT sufficient: `or`
    only replaces FALSY values, so 0.0 and None fall back but **-1.0
    survives** — and a negative interval makes `_t_end - _win_t0 >=
    _stats_interval` true on EVERY loop, closing a window per iteration.
    That is the collapsed-window pathology v1.6.8 exists to remove, reached
    by setting the option to the value that most looks like "turn it off".
    Same falsy-vs-invalid confusion as the v1.6.7 `send_hz=0.0` bug.

    argparse rejects <= 0 before the L2/L3 split, so this clamp covers the
    remaining path only — but with both in place the guarantee is
    unconditional rather than path-qualified."""
    from genesis_vehicle.server.pacing import (resolve_stats_interval,
                                               DEFAULT_STATS_INTERVAL)
    assert DEFAULT_STATS_INTERVAL == 1.0
    # the case `or` misses, and the one this test exists for
    assert resolve_stats_interval(_Args(stats_interval=-1.0)) == 1.0
    assert resolve_stats_interval(_Args(stats_interval=-1e-9)) == 1.0
    # the falsy cases `or` already handled, which must not regress
    assert resolve_stats_interval(_Args(stats_interval=0.0)) == 1.0
    assert resolve_stats_interval(_Args(stats_interval=-0.0)) == 1.0
    assert resolve_stats_interval(_Args(stats_interval=None)) == 1.0
    assert resolve_stats_interval(_Args()) == 1.0            # attribute absent
    # NaN would make the window trigger false FOREVER (no window ever closes)
    assert resolve_stats_interval(_Args(stats_interval=float("nan"))) == 1.0
    # unparseable falls back rather than raising: the loop is already running
    assert resolve_stats_interval(_Args(stats_interval="x")) == 1.0
    # valid values pass through untouched, including strings argparse would
    # have converted, and the caller may override the default
    assert resolve_stats_interval(_Args(stats_interval=2.5)) == 2.5
    assert resolve_stats_interval(_Args(stats_interval="0.25")) == 0.25
    assert resolve_stats_interval(_Args(stats_interval=0.0), default=5.0) == 5.0


def test_a_rejected_stats_interval_is_rejected_LOUDLY(capsys):
    """Falling back in silence would RELOCATE the defect, not close it: the
    hand-built args namespace is the only caller this resolver exists for, so
    a value it ignores is a value nobody is told about. Same standing as the
    `--send-hz` non-positive demotion, which WARNs for exactly this reason.

    NaN is the non-obvious case: `float("nan")` PARSES, so it never reaches
    the `except` — it reaches the fallback through `not value > 0.0`, and the
    warning has to fire on that path too."""
    from genesis_vehicle.server.pacing import resolve_stats_interval
    for bad in (-1.0, 0.0, float("nan"), "x", [1.0]):
        resolve_stats_interval(_Args(stats_interval=bad))
        out = capsys.readouterr().out
        assert "[WARN]" in out, (bad, out)
        assert "--stats-interval" in out, (bad, out)
        assert "1" in out, (bad, out)       # names the value it fell back to
    # NaN specifically, by the path it takes
    resolve_stats_interval(_Args(stats_interval=float("nan")))
    assert "양수가 아닙니다" in capsys.readouterr().out
    # ...and UNSET is not a bad value: no warning for a missing attribute or
    # an explicit None, or every embedder that never set it gets a scary line
    for quiet in (_Args(), _Args(stats_interval=None)):
        resolve_stats_interval(quiet)
        assert capsys.readouterr().out == ""
    # a VALID value is silent too
    resolve_stats_interval(_Args(stats_interval=2.5))
    assert capsys.readouterr().out == ""


def test_both_loops_clamp_the_interval_through_the_same_resolver():
    """L2 and L3 must not diverge here: two loops with different window
    lengths produce [SERVE] keys that cannot be pooled or compared, which is
    the entire purpose of the keys. One shared resolver, called in both, and
    neither may re-introduce a local `or`-style fallback."""
    for name, src, _node in _loops():
        assert "resolve_stats_interval(args)" in src, \
            f"{name} does not use the shared resolver"
        # comment-stripped, so the explanatory comment at the call site (which
        # quotes the `or 1.0` shape it replaced) does not trip the check
        code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
        assert 'getattr(args, "stats_interval"' not in code, \
            f"{name} re-reads stats_interval directly, bypassing the clamp"
        assert "or 1.0" not in code, f"{name} still has an `or` fallback"
    # ...and the resolver lives in ONE place that both import
    ps_src = inspect.getsource(_ps())
    l3_src = inspect.getsource(_l3())
    for name, src in (("L2", ps_src), ("L3", l3_src)):
        assert "resolve_stats_interval" in src.split("\n\n")[0] \
            or "from .pacing import" in src, name
        imports = [ln for ln in src.splitlines()
                   if ln.startswith("from .pacing import")]
        assert imports and "resolve_stats_interval" in imports[0], name


def test_stats_interval_flag_is_validated_and_shared_by_both_modes():
    src = _src(_ps().main)
    assert 'parser.add_argument("--stats-interval"' in src
    assert "default=1.0" in src.split('"--stats-interval"')[1][:400]
    # rejected at <= 0: a zero or negative interval closes a window on every
    # loop, which is worse than the 50-loop behaviour it replaced.
    reject = src.split("args.stats_interval <= 0.0")
    assert len(reject) == 2, "--stats-interval is not validated"
    assert "parser.error" in reject[1][:200]
    # ...and NO "L3 ignores this" warning, unlike --send-hz / --single-scene:
    # both modes honour it. The L3 branch's warnings must not mention it.
    l3_branch = src.split("if args.multi_env:")[1].split("run_l3(args)")[0]
    assert "stats" not in l3_branch.lower(), l3_branch
    # both loops read it — through the shared clamp, never raw (see
    # test_non_positive_stats_interval_falls_back_to_the_default)
    for name, loop_src, _n in _loops():
        assert "resolve_stats_interval(args)" in loop_src, name


def test_benchmark_forwards_stats_interval():
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm
    sig = inspect.signature(bm.run_config)
    assert "stats_interval" in sig.parameters
    assert sig.parameters["stats_interval"].default == 1.0
    src = inspect.getsource(bm.run_config)
    assert '"--stats-interval", str(stats_interval)' in src
    assert "stats_interval=stats_interval" in src      # recorded in the result
    main_src = inspect.getsource(bm.main)
    assert 'ap.add_argument("--stats-interval"' in main_src
    assert "stats_interval=args.stats_interval" in main_src
    assert "args.stats_interval <= 0.0" in main_src    # rejected here too


def test_budget_ms_is_milliseconds_not_seconds():
    """`budget_ms` feeds the realtime O/X verdict against `loop_ms`, which is
    in MILLISECONDS. Hoisting the literal 25.0 onto WIRE_DT (0.025 s) without
    the x1000 would make every config fail the verdict."""
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm
    assert bm.WIRE_DT == 0.025
    src = inspect.getsource(bm.main)
    assert "budget_ms = WIRE_DT * 1000.0" in src
    assert bm.WIRE_DT * 1000.0 == 25.0                 # the historical budget


# --------------------------------------------------------------------------
# 16-18: the benchmark's pooled re-weighting
# --------------------------------------------------------------------------

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


def _replay(bm, monkeypatch, lines, measure_stats):
    monkeypatch.setattr(bm.subprocess, "Popen", lambda *a, **kw: _FakeProc(lines))
    monkeypatch.setattr(bm, "MockUEClient", _FakeClient)
    return bm.run_config("L2", "simple", 1, "/tmp/tank.urdf", "/tmp/hull.obj",
                         False, measure_stats, 30.0, "python", False)


def _stats(loop, phys):
    return (f" [STATS] [L2] Loop Avg: {loop:.2f} ms | "
            f"Physics Avg: {phys:.2f} ms "
            f"(1.0 steps/loop, 2.50 ms/step) [cap=5:burst]\n")


def _serve(loops, steps, serve_ms, phys_ms, window_ms, with_keys=True):
    head = (" [SERVE] [L2] recv_loops=0 nonskip_loops=0 skipped_captures=0 "
            "post_step_captures=0 post_step_captures_ref=0 sends=0 "
            "send_skips=0 send_flushes=0 ")
    if with_keys:
        head += (f"loops={loops} steps={steps} zero_step_loops=0 "
                 f"serve_ms={serve_ms:.3f} phys_ms_sum={phys_ms:.3f} ")
    return head + f"window_ms={window_ms:.1f}\n"


def test_run_config_reweights_by_loops(monkeypatch):
    """The headline averages must be POOLED over the windows' own loop counts.
    While every window was exactly 50 loops the unweighted mean of the
    per-window means happened to equal the pooled mean; with wall-clock
    windows it does not, and the error is unbounded — the replay below is
    built so the two answers differ by ~2x.

    Windows (window 0 is the dropped warm-up):
      w1: 4000 loops,   40 steps,  serve 400 ms, phys  100 ms, 1000 ms
      w2:   40 loops,   40 steps,  serve  20 ms, phys  980 ms, 1000 ms
    """
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm

    lines = [f"{bm._TARGET_TOKEN}0\n",
             " [Genesis] Initialization Complete. Obstacles: 0\n",
             _stats(9.00, 5.00), _serve(38, 38, 100.0, 100.0, 1000.0),
             _stats(0.13, 0.03), _serve(4000, 40, 400.0, 100.0, 1000.0),
             _stats(25.00, 24.50), _serve(40, 40, 20.0, 980.0, 1000.0)]
    r = _replay(bm, monkeypatch, lines, 3)

    assert r["ok"] is True and r["n"] == 2
    assert r["weighting"] == "loops"
    loops, steps = 4040.0, 80.0
    serve, phys, win = 420.0, 1080.0, 2000.0
    assert r["loop_ms"] == pytest.approx((serve + phys) / loops)      # 0.3713
    assert r["phys_ms"] == pytest.approx(phys / loops)                # 0.2673
    assert r["serving_ms"] == pytest.approx(serve / loops)            # 0.1040
    assert r["steps_per_loop"] == pytest.approx(steps / loops)        # 0.0198
    # ms/step is the ONE column weighted by `steps` rather than by `loops`
    assert r["ms_per_step"] == pytest.approx(phys / steps)            # 13.5
    assert r["duty"] == pytest.approx(serve / win)                    # 0.21
    assert r["s_prime"] == pytest.approx(serve / steps)               # 5.25

    # ...and every one of them differs from what the pre-v1.6.8 unweighted
    # estimator would have produced, so a regression cannot slip past by
    # looking plausible. Unweighted: mean of the two [STATS] rows.
    assert r["loop_ms"] != pytest.approx((0.13 + 25.00) / 2)          # 12.565
    assert r["phys_ms"] != pytest.approx((0.03 + 24.50) / 2)          # 12.265
    assert r["serving_ms"] != pytest.approx(
        ((0.13 - 0.03) + (25.00 - 24.50)) / 2)                        # 0.30
    assert r["ms_per_step"] != pytest.approx(2.50)                    # both rows


def test_run_config_falls_back_loudly_when_loops_key_is_missing(monkeypatch):
    """A server without the v1.6.8 keys (or an unaligned [SERVE] stream) must
    fall back to the old estimator and SAY SO. Silently mixing estimators
    between two arms of a comparison is the failure this guards."""
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm

    lines = [f"{bm._TARGET_TOKEN}0\n",
             " [Genesis] Initialization Complete. Obstacles: 0\n"]
    for loop, phys in ((9.00, 5.00), (4.00, 2.50), (6.00, 3.50)):
        lines.append(_stats(loop, phys))
        lines.append(_serve(0, 0, 0, 0, 1250.0, with_keys=False))
    r = _replay(bm, monkeypatch, lines, 3)

    assert r["ok"] is True
    assert r["weighting"] == "unweighted"
    assert r["loop_ms"] == pytest.approx(5.0)          # (4.00 + 6.00) / 2
    assert r["phys_ms"] == pytest.approx(3.0)
    assert r["serving_ms"] == pytest.approx(2.0)
    assert r["loops_total"] is None
    assert r["low_sample"] is False                    # unknown, not flagged
    assert r["duty"] != r["duty"] and r["s_prime"] != r["s_prime"]   # NaN


def test_run_config_records_sample_size(monkeypatch):
    """`loops_min` is computed over the MEASURED windows only. Including the
    dropped warm-up window (whose origin predates the loop and which can be
    arbitrarily short) would fire !!low-sample on a perfectly good run."""
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm

    lines = [f"{bm._TARGET_TOKEN}0\n",
             " [Genesis] Initialization Complete. Obstacles: 0\n",
             _stats(9.00, 5.00), _serve(3, 3, 10.0, 10.0, 1000.0),   # warm-up
             _stats(4.00, 2.50), _serve(40, 40, 100.0, 100.0, 1000.0),
             _stats(6.00, 3.50), _serve(60, 60, 100.0, 100.0, 1000.0)]
    r = _replay(bm, monkeypatch, lines, 3)
    assert r["loops_total"] == 100
    assert r["loops_min"] == 40 and r["loops_max"] == 60   # 3 is NOT the min
    assert r["low_sample"] is False

    lines2 = [f"{bm._TARGET_TOKEN}0\n",
              " [Genesis] Initialization Complete. Obstacles: 0\n",
              _stats(9.00, 5.00), _serve(40, 40, 100.0, 100.0, 1000.0),
              _stats(200.0, 190.0), _serve(5, 5, 50.0, 950.0, 1000.0),
              _stats(200.0, 190.0), _serve(6, 6, 50.0, 950.0, 1000.0)]
    r2 = _replay(bm, monkeypatch, lines2, 3)
    assert r2["loops_min"] == 5
    assert r2["low_sample"] is True
    assert bm.MIN_WINDOW_LOOPS == 20


def test_ms_per_step_and_s_prime_survive_a_zero_step_window(monkeypatch):
    """Every loop in every measured window can be a zero-step busy-wait loop,
    making Sum(steps) == 0. The server guards the same quotient with
    max(log_step_sum, 1); the reader must not die with a ZeroDivisionError
    instead of recording a result."""
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm

    lines = [f"{bm._TARGET_TOKEN}0\n",
             " [Genesis] Initialization Complete. Obstacles: 0\n",
             _stats(9.00, 5.00), _serve(40, 0, 1000.0, 0.0, 1000.0),
             _stats(0.01, 0.00), _serve(4000, 0, 1000.0, 0.0, 1000.0),
             _stats(0.01, 0.00), _serve(4000, 0, 1000.0, 0.0, 1000.0)]
    r = _replay(bm, monkeypatch, lines, 3)
    assert r["ok"] is True
    assert r["steps_per_loop"] == 0.0
    assert r["ms_per_step"] == 0.0        # 0 ms of physics / guarded 1 step
    assert r["s_prime"] == pytest.approx(2000.0)   # serve_ms / guarded 1 step
    assert r["duty"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# the summary artifact: the weighting must travel with the TABLE, not only
# with the progress line (the table is what gets pasted into docs)
# --------------------------------------------------------------------------

def _record(weighting="loops", low_sample=False, **over):
    r = dict(mode="L2", terrain="simple", k=1, ok=True, n=4,
             kinds=1, kinds_observable=True, kinds_requested=1,
             k_observed=1, reuse_lines=0,
             ms_per_step=18.3, steps_per_loop=0.9, loop_ms=20.0,
             serving_ms=1.7, phys_ms=18.3, send_hz=None,
             pacing_mode="burst", pacing_switches=0, pacing_events=[],
             profile="", serve_windows=4, serve_windows_match=True,
             serve_sum={"loops": 160.0, "serve_ms": 400.0,
                        "window_ms": 4000.0, "steps": 160.0},
             weighting=weighting, low_sample=low_sample,
             duty=0.1, s_prime=2.5,
             loops_total=160, loops_min=5 if low_sample else 38,
             loops_max=42, build_s=1.0)
    r.update(over)
    return r


def _run_main(monkeypatch, capsys, records):
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm
    it = iter(records)
    monkeypatch.setattr(bm, "_make_hull_obj", lambda: "/tmp/hull.obj")
    monkeypatch.setattr(bm.os.path, "exists", lambda p: True)
    monkeypatch.setattr(bm, "run_config", lambda *a, **kw: next(it))
    monkeypatch.setattr(bm.sys, "argv",
                        ["benchmark", "--modes", "L2", "--terrain", "simple",
                         "--tanks", ",".join(["1"] * len(records))])
    bm.main()
    return capsys.readouterr().out


def test_summary_table_carries_the_weighting_of_every_row(monkeypatch, capsys):
    """The markdown table is the artifact that ends up in docs/server.md and
    the CHANGELOG. A row computed with the pre-v1.6.8 unweighted estimator
    must SAY so there, not only on the transient progress line — otherwise
    two estimators get pasted into one table under one caption."""
    out = _run_main(monkeypatch, capsys,
                    [_record(), _record(weighting="unweighted")])
    table = [ln for ln in out.splitlines() if ln.startswith("| L2 |")]
    assert len(table) == 2, out
    assert "| loops |" in table[0], table[0]
    assert "!!unweighted" not in table[0]
    assert "unweighted !!unweighted" in table[1], table[1]
    # the header gained the column, and the separator row still matches it
    header = [ln for ln in out.splitlines() if ln.startswith("| mode |")][0]
    sep = out.splitlines()[out.splitlines().index(header) + 1]
    assert header.count("|") == sep.count("|") == table[0].count("|")
    assert "weighting" in header


def test_footer_does_not_claim_pooled_weighting_when_a_row_fell_back(
        monkeypatch, capsys):
    """The footer used to assert 'every headline average above is POOLED'
    unconditionally. The fallback is reachable in normal operation (the
    reader-race grace, or a pre-v1.6.8 server), so the caption could swear
    to a property the table did not have."""
    good = _run_main(monkeypatch, capsys, [_record()])
    assert "every headline average above is POOLED" in good
    assert "MIXED ESTIMATORS" not in good

    mixed = _run_main(monkeypatch, capsys,
                      [_record(), _record(weighting="unweighted")])
    assert "every headline average above is POOLED" not in mixed
    assert "!! MIXED ESTIMATORS" in mixed
    assert "1 of 2 rows" in mixed
    assert "L2/simple/1" in mixed


def test_low_sample_is_flagged_in_the_table_and_the_serve_line(
        monkeypatch, capsys):
    """Refuse rule 7 is phrased in loops_min, so loops_min has to be visible
    on the [serve] line too — `loops` alone is the SUM and hides a window of
    5 loops among windows of 40."""
    out = _run_main(monkeypatch, capsys, [_record(low_sample=True)])
    row = [ln for ln in out.splitlines() if ln.startswith("| L2 |")][0]
    assert "!!low-sample (loops_min=5)" in row, row
    serve = [ln for ln in out.splitlines() if ln.startswith("[serve]")][0]
    assert "loops=160 (5..42)" in serve, serve
    assert "!!low-sample" in serve
