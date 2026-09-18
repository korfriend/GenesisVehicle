"""v1.6.6 serving work: the post-override capture skip, the ``tag=`` seam, the
``[SERVE]`` counter line, and the benchmark wiring that makes them observable.

| abbr | meaning |
|---|---|
| L2 | per-entity server mode (K interacting vehicles, n_envs=1) |
| kind | one batched vehicle configuration; targets sharing (URDF, mapping, friction) share one |
| SERVE | the server's per-window serving-counter stdout line (separate from the frozen STATS line) |

v1.6.8 update: the reporting window is a fixed WALL-CLOCK interval
(``--stats-interval``, default 1.0 s), not a fixed 50 loops, so the canned
``[SERVE]`` lines below carry the six window-shape keys and the reset-boundary
assertions split on the new trigger instead of ``if log_count >= 50:``.

These are pure-Python: no scene is built and no server process is started.
"""
import inspect
import re

import pytest

# The source text of the window trigger in physics_server.main(). Every
# "is it reset on the boundary?" assertion splits on this, and each one
# asserts the token is PRESENT first: `str.split` on an absent separator
# returns the whole string, so a silently renamed trigger would turn those
# assertions into vacuous passes rather than failures. (It did exactly that
# when the trigger moved off `if log_count >= 50:` — the tests stayed green.)
_WINDOW_TRIGGER = "if _t_end - _win_t0 >= _stats_interval:"


# --------------------------------------------------------------------------
# physics_server — skip predicate + capture_state seam
# --------------------------------------------------------------------------

def _ps():
    pytest.importorskip("genesis")
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import physics_server
    return physics_server


def test_needs_override_capture_truth_table():
    ps = _ps()
    f = ps.needs_override_capture
    # Nothing happened -> the held state is still current -> skip.
    assert f({}, {}, {}, False) is False
    # Any one of the four inputs forces the capture.
    assert f({0: {'pos': [0, 0, 0]}}, {}, {}, False) is True
    assert f({}, {0: [{'type': 'AddWorldOffset'}]}, {}, False) is True
    assert f({}, {}, {1000: {'pos': [0, 0, 0]}}, False) is True
    assert f({}, {}, {}, True) is True
    # Empty containers of any flavour are falsy, None too (defensive).
    assert f(None, None, None, False) is False
    assert f([], [], [], False) is False


def test_needs_override_capture_ignores_target_forces():
    """target_forces goes through control_dofs_force, which moves nothing until
    the next step — it must not be an input to the skip decision."""
    ps = _ps()
    assert "target_forces" not in inspect.signature(
        ps.needs_override_capture).parameters
    src = inspect.getsource(ps.needs_override_capture)
    assert "target_forces" in src, "the docstring must say WHY it is excluded"


def test_capture_state_tag_is_keyword_only_with_default():
    ps = _ps()
    params = inspect.signature(ps.capture_state).parameters
    assert "tag" in params
    assert params["tag"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["tag"].default == ""
    # Every pre-1.6.6 positional argument keeps its position, so external
    # positional callers are unaffected.
    positional = [n for n, p in params.items()
                  if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert positional == [
        "target_entities", "dynamic_obstacles", "is_urdf_active", "controllers",
        "ue_driven_obstacle_ids", "accumulated_wheel_angles", "sim_dt",
        "update_angles", "mvp", "readers"]


def test_all_four_loop_capture_sites_are_tagged():
    ps = _ps()
    src = inspect.getsource(ps.main)
    for tag in ("init", "reset", "override", "post_step"):
        assert f'tag="{tag}"' in src, f"capture_state call site {tag!r} untagged"
    # ...and every capture_state call inside main() carries a tag.
    calls = re.findall(r"capture_state\((?:[^()]|\([^()]*\))*\)", src)
    assert len(calls) == 4
    assert all("tag=" in c for c in calls)


def test_serve_counters_exist_and_reset_with_the_stats_counters():
    ps = _ps()
    src = inspect.getsource(ps.main)
    names = ["log_recv_loops", "log_skipped_captures", "log_nonskip_loops",
             "log_post_step_captures", "log_post_step_captures_ref"]
    for n in names:
        # once initialised, once printed/incremented, once reset
        assert src.count(n) >= 3, n
        assert f"{n} = 0" in src
    # The reset happens in the same window block that resets log_count.
    assert "log_count = 0" in src
    assert _WINDOW_TRIGGER in src, \
        "the window trigger was renamed; the split below would be vacuous"
    for n in names:
        assert f"{n} = 0" in src.split(_WINDOW_TRIGGER)[-1], \
            f"{n} is not reset on the [STATS] window boundary"


def test_serve_line_is_separate_from_the_frozen_stats_line():
    ps = _ps()
    src = inspect.getsource(ps.main)
    # The [SERVE] text must not be appended to the [STATS] f-string: the two
    # benchmarks each hold their own _STATS_RE copy of that string.
    stats_print = src.split('f" [STATS] [L2]')[1].split("[cap=")[0]
    assert "[SERVE]" not in stats_print
    assert 'f" [SERVE] [L2] recv_loops=' in src


def test_recv_loops_is_counted_at_the_skip_decision_not_at_if_recv():
    """A loop carrying 'stop' breaks out before the decision; counting it would
    break skipped_captures + nonskip_loops == recv_loops by one."""
    ps = _ps()
    src = inspect.getsource(ps.main)
    before_decision = src.split("log_recv_loops += 1")[0]
    # the 'stop' break must already have happened by the counting point
    assert "cmd == 'stop'" in before_decision
    # and the counter must sit next to the skip branch, not at `if recv:`
    assert "log_recv_loops += 1" in src
    after = src.split("log_recv_loops += 1")[1]
    # The guard belongs on `after`, not on `src`: the name also appears in the
    # initialisation block ABOVE the loop, so asserting it in `src` is always
    # true and guards nothing. A split on an absent separator returns the
    # whole remainder, which would turn both "sits BEFORE the post-step
    # captures" assertions into "exists anywhere after the counter".
    assert "log_post_step_captures" in after
    assert "log_nonskip_loops += 1" in after.split("log_post_step_captures")[0]
    assert "log_skipped_captures += 1" in after.split("log_post_step_captures")[0]


def test_serve_token_marks_only_the_counter_line():
    """The bench counts [SERVE] windows by the token, so no other line may carry
    it (a token-only line would shift the window alignment by one)."""
    import ast
    import textwrap
    ps = _ps()
    tree = ast.parse(textwrap.dedent(inspect.getsource(ps.main)))
    # ast never sees comments, so this looks at real string literals only.
    line_literals = [n.value for n in ast.walk(tree)
                     if isinstance(n, ast.Constant) and isinstance(n.value, str)
                     and n.value.lstrip().startswith("[SERVE]")]
    assert line_literals, "main() builds no [SERVE]-prefixed output line"
    assert all("recv_loops" in v for v in line_literals), line_literals


def test_serve_line_parses_with_the_benchmark_parser_and_balances():
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm
    line = (" [SERVE] [L2] recv_loops=50 nonskip_loops=3 skipped_captures=47 "
            "post_step_captures=52 post_step_captures_ref=50 "
            "loops=50 steps=40 zero_step_loops=10 serve_ms=312.500 "
            "phys_ms_sum=687.500 window_ms=1000.0 "
            "cap_override_us=4071.3\n")
    assert bm._SERVE_RE.search(line)
    kv = {k: float(v) for k, v in bm._SERVE_KV_RE.findall(line)}
    assert set(kv) == {"recv_loops", "nonskip_loops", "skipped_captures",
                       "post_step_captures", "post_step_captures_ref",
                       "loops", "steps", "zero_step_loops", "serve_ms",
                       "phys_ms_sum", "window_ms",
                       "cap_override_us"}
    # the accounting identity the counters are built to make decidable
    assert kv["skipped_captures"] + kv["nonskip_loops"] == kv["recv_loops"]
    # the v1.6.8 window-shape identities (see the [SERVE] comment block in
    # physics_server.main): the window no longer has a constant loop count, so
    # `loops` is what every per-loop average must be weighted by.
    assert kv["zero_step_loops"] <= kv["loops"]
    assert kv["serve_ms"] <= kv["window_ms"]        # proven, not assumed: the
    # loops' [loop_start, _t_end] spans are disjoint and all lie inside
    # [_win_t0, _win_now], and sleep is counted in window_ms only.
    assert kv["phys_ms_sum"] >= 0.0
    # the "[L2]" token has no '=' and must not leak in as a key
    assert not any(k.startswith("L") for k in kv)


def test_serve_line_does_not_disturb_either_stats_parser():
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm
    from genesis_vehicle.server import benchmark_collision as bc
    stats = (" [STATS] [L2] Loop Avg: 3.86 ms | Physics Avg: 2.61 ms "
             "(1.0 steps/loop, 2.61 ms/step) [cap=5:burst]\n")
    serve = (" [SERVE] [L2] recv_loops=50 nonskip_loops=0 skipped_captures=50 "
             "loops=50 steps=40 zero_step_loops=10 serve_ms=312.500 "
             "phys_ms_sum=687.500 window_ms=1000.0\n")
    # v1.6.8: L3 prints a [SERVE] line too, and it must be just as invisible
    # to both frozen [STATS] parsers.
    serve_l3 = (" [SERVE] [L3] loops=41 steps=40 zero_step_loops=1 "
                "serve_ms=312.500 phys_ms_sum=687.500 window_ms=1000.0\n")
    assert bm._STATS_RE.search(stats)
    assert bc._STATS_RE.search(stats)
    assert bm._STATS_RE.search(serve) is None
    assert bc._STATS_RE.search(serve) is None
    assert bm._STATS_RE.search(serve_l3) is None
    assert bc._STATS_RE.search(serve_l3) is None


# --------------------------------------------------------------------------
# benchmark — --kinds / --input-hz / observed-kind anchors
# --------------------------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.msgs = []

    def send_message(self, addr, args):
        self.msgs.append((addr, args))


def _client(k, kinds=1, complex_terrain=False, input_hz=30.0):
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server.benchmark import MockUEClient
    c = MockUEClient(k, "/tmp/tank.urdf", complex_terrain, "/tmp/hull.obj",
                     kinds=kinds, input_hz=input_hz)
    c.cli = _Recorder()
    return c


def _target_frictions(client):
    client.send_init_burst()
    return [args[13] for addr, args in client.cli.msgs if addr == "/Init/Target"]


def test_kinds_default_is_bit_identical_to_the_historical_constant():
    fr = _target_frictions(_client(6))
    assert fr == [2.0] * 6
    assert all(f.hex() == (2.0).hex() for f in fr)


def test_kinds_n_splits_the_fleet_into_exactly_n_kinds():
    for n in (2, 3, 5):
        fr = _target_frictions(_client(6, kinds=n))
        assert len(set(fr)) == min(n, 6)
        assert fr == [2.0 + 1e-3 * (t % n) for t in range(6)]


def test_kinds_does_not_touch_global_or_obstacle_friction():
    """The 88 static hull meshes and the global physics friction must stay one
    value — splitting them would create static-mesh kinds as a side effect."""
    c = _client(6, kinds=3, complex_terrain=True)
    c.send_init_burst()
    phys = [a for addr, a in c.cli.msgs if addr == "/Genesis/Init/Physics"]
    assert phys and phys[0][2] == 2.0
    obs_fr = set()
    for addr, args in c.cli.msgs:
        if addr != "/Init/Obstacle":
            continue
        for i in range(0, len(args), 18):
            obs_fr.add(args[i + 13])
    assert obs_fr == {2.0}


def test_input_hz_default_reproduces_the_historical_sleep_exactly():
    c = _client(1)
    assert c.input_hz == 30.0
    assert 1.0 / c.input_hz == 1.0 / 30.0
    assert 1.0 / _client(1, input_hz=100.0).input_hz == 0.01


def test_bench_argparse_defaults_are_the_historical_behaviour():
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm
    src = inspect.getsource(bm.main)
    assert '"--kinds", type=int, default=1' in src
    assert '"--input-hz", type=float, default=30.0' in src


def test_observed_kind_anchors_match_what_the_server_actually_prints():
    """kinds = ('Target ID' lines) - ('reusing shared cfg' lines). Both anchors
    are asserted against the producing source, so a reworded print breaks this
    test instead of silently zeroing the bench's kind count."""
    pytest.importorskip("genesis")
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm
    from genesis_vehicle.server import vehicle_builder as vb

    reuse_src = inspect.getsource(vb.build_cfg)
    assert bm._REUSE_TOKEN in reuse_src

    table_src = inspect.getsource(vb.print_resolved_table)
    assert f'print(f"{bm._TARGET_TOKEN}' in table_src
    # exactly one line per target: the token must not recur in the table body
    assert inspect.getsource(vb).count("Target ID") == 1
    # and the line as printed starts with the token (leading spaces included)
    assert f"{bm._TARGET_TOKEN}7".startswith(bm._TARGET_TOKEN)


def test_run_config_records_kinds_n_and_serving():
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm
    src = inspect.getsource(bm.run_config)
    for field in ("kinds=k_observed - reuse_lines", "k_observed=k_observed",
                  "reuse_lines=reuse_lines", "n=n", "serving_ms="):
        assert field in src, field
    # serving is still Loop Avg - Physics Avg, but v1.6.8 computes it POOLED
    # over the windows' own loop counts (the windows are wall-clock now and
    # hold different loop counts, so the mean of the per-window means is
    # biased). Both estimators must be present: the pooled one and the loud
    # unweighted fallback for a server that prints no window keys.
    assert "serving_ms = _serve / _loops" in src              # pooled
    assert "loop_ms = (_serve + _phys) / _loops" in src
    assert 'weighting = "loops"' in src
    assert ("serving_ms = (sum(r[0] for r in rows) - sum(r[1] for r in rows)) / n"
            in src)                                           # fallback
    assert 'weighting = "unweighted"' in src
    # the summary table carries the serving column
    assert "serving" in inspect.getsource(bm.main)


def test_pooled_headline_is_an_identity_not_an_approximation():
    """`serving_ms == loop_ms - phys_ms` must hold EXACTLY under the pooled
    estimator, because loop_ms is rebuilt from the same two [SERVE] sums
    (`serve_ms + phys_ms_sum` is Sum(loop_dur) by construction) rather than
    re-weighted from the rounded [STATS] print."""
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm
    src = inspect.getsource(bm.run_config)
    assert "loop_ms = (_serve + _phys) / _loops" in src
    assert "phys_ms = _phys / _loops" in src
    assert "serving_ms = _serve / _loops" in src
    # ms/step is the ONE column weighted by steps rather than by loops, and
    # Sum(steps) can legitimately be 0 (every loop a zero-step spin loop), so
    # it is guarded exactly as the server guards it with max(log_step_sum, 1).
    assert "ms_per_step = _phys / _steps_d" in src
    assert "_steps_d = _steps if _steps > 0 else 1.0" in src


def test_bench_reader_keeps_the_lines_the_gate_depends_on():
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm
    src = inspect.getsource(bm.run_config)
    assert "counts[\"reuse_lines\"] += 1" in src
    assert "counts[\"k_observed\"] += 1" in src
    assert "serve_rows.append" in src


def _replay(bm, monkeypatch, mode, lines, k=6, kinds=1, measure_stats=3):
    """Run the real run_config against a canned server stdout."""
    class _FakeProc:
        returncode = 0

        def __init__(self):
            self.stdout = iter(lines)

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.kwargs = kw

        def send_init_burst(self):
            pass

        def start_inputs(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(bm.subprocess, "Popen", lambda *a, **kw: _FakeProc())
    monkeypatch.setattr(bm, "MockUEClient", _FakeClient)
    return bm.run_config(mode, "simple", k, "/tmp/tank.urdf", "/tmp/hull.obj",
                         False, measure_stats, 30.0, "python", False, kinds=kinds)


def test_run_config_parses_a_replayed_server_stdout(monkeypatch):
    """Drive run_config's reader with a canned server stdout — the live bench
    cannot be used as a test (it spawns a real server and takes minutes), but
    the parsing, the kind arithmetic and the serving split are all here."""
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm

    stats_line = (" [STATS] [L2] Loop Avg: {:.2f} ms | Physics Avg: {:.2f} ms "
                  "(1.0 steps/loop, 2.50 ms/step) [cap=5:burst]\n")
    # Every window here is 50 loops, which is the ONE case where the pooled
    # estimator and the old unweighted one agree — deliberately, so the
    # numbers this test has always asserted keep their meaning. The unequal
    # case is test_run_config_reweights_by_loops in test_server_stats_window.py.
    serve_line = (" [SERVE] [L2] recv_loops=50 nonskip_loops=0 "
                  "skipped_captures=50 post_step_captures=50 "
                  "post_step_captures_ref=50 loops=50 steps=50 "
                  "zero_step_loops=0 serve_ms={:.3f} phys_ms_sum={:.3f} "
                  "window_ms=1250.0\n")
    lines = [" [Genesis] [MODE] === L2 (per-entity) ===\n"]
    for tid in range(6):
        if tid >= 3:
            lines.append(f" [Genesis] [Batch] Vehicle {tid}: "
                         f"{bm._REUSE_TOKEN} (same URDF/mapping/friction)\n")
        lines.append(f"{bm._TARGET_TOKEN}{tid}\n")
    lines.append(" [Genesis] Initialization Complete. Obstacles: 0\n")
    # window 0 is the warm-up window and must be dropped by both parsers
    for loop, phys in ((9.00, 5.00), (4.00, 2.50), (6.00, 3.50)):
        lines.append(stats_line.format(loop, phys))
        # serve_ms / phys_ms_sum are the per-window SUMS matching the same
        # per-loop averages: 50 loops x (loop - phys) ms and 50 x phys ms.
        lines.append(serve_line.format(50 * (loop - phys), 50 * phys))

    r = _replay(bm, monkeypatch, "L2", lines, k=6, kinds=3, measure_stats=3)
    assert r["ok"] is True
    assert r["k_observed"] == 6 and r["reuse_lines"] == 3
    assert r["kinds"] == 3 and r["kinds_requested"] == 3
    assert r["n"] == 2                                  # first window dropped
    assert r["weighting"] == "loops"                    # v1.6.8 pooled
    assert r["loop_ms"] == pytest.approx(5.0)           # (4.00 + 6.00) / 2
    assert r["phys_ms"] == pytest.approx(3.0)           # (2.50 + 3.50) / 2
    assert r["serving_ms"] == pytest.approx(2.0)        # Loop Avg - Physics Avg
    assert r["loops_total"] == 100 and r["loops_min"] == 50
    assert r["low_sample"] is False
    # D = Sum(serve_ms) / Sum(window_ms) = 200 ms / 2500 ms
    assert r["duty"] == pytest.approx(0.08)
    # S' = Sum(serve_ms) / Sum(steps) = 200 ms / 100 steps
    assert r["s_prime"] == pytest.approx(2.0)
    assert r["serve_windows"] == 2                      # aligned with n
    assert r["serve_windows_match"] is True
    assert r["kinds_observable"] is True                # L2
    assert r["serve_sum"]["recv_loops"] == 100
    assert (r["serve_sum"]["skipped_captures"]
            + r["serve_sum"]["nonskip_loops"]) == r["serve_sum"]["recv_loops"]


def _main_ast():
    import ast
    import textwrap
    ps = _ps()
    return ast, ast.parse(textwrap.dedent(inspect.getsource(ps.main))).body[0]


def _assigns_to(ast, tree, name, value=None):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == name for t in node.targets):
            if value is None or getattr(node.value, "value", None) == value:
                out.append(node)
    return out


def test_reset_ran_is_armed_per_loop_and_set_only_by_the_reset_branch():
    """Wiring, not predicate: `_reset_ran` must be re-armed False at the top of
    the `if recv:` body (per loop, before any branch can read it) and set True
    only inside the reset branch.

    This term is REDUNDANT today — the reset branch re-captures for itself
    immediately above — so a broken wiring here would have no symptom now. If
    that re-capture is ever removed as an optimisation, a stale `_reset_ran`
    becomes a silently broadcast stale state. Hence a test rather than trust.
    """
    ast, main_node = _main_ast()

    if_recv = [n for n in ast.walk(main_node)
               if isinstance(n, ast.If) and getattr(n.test, "id", "") == "recv"]
    assert len(if_recv) == 1, "expected exactly one `if recv:` in the loop"
    first = if_recv[0].body[0]
    assert isinstance(first, ast.Assign), ast.dump(first)
    assert [t.id for t in first.targets] == ["_reset_ran"]
    assert first.value.value is False

    # ...and it lives INSIDE the per-loop `while True:` body, so it cannot
    # survive from one loop into the next.
    while_true = [n for n in ast.walk(main_node)
                  if isinstance(n, ast.While)
                  and getattr(n.test, "value", None) is True]
    assert len(while_true) == 1
    assert while_true[0].lineno < first.lineno <= while_true[0].end_lineno

    # The only True assignment sits in the `cmd == 'reset'` branch.
    trues = _assigns_to(ast, main_node, "_reset_ran", True)
    assert len(trues) == 1
    reset_ifs = [n for n in ast.walk(main_node)
                 if isinstance(n, ast.If)
                 and "reset" in ast.dump(n.test)
                 and any(t is trues[0] for t in ast.walk(n))]
    assert reset_ifs, "the True assignment is not inside the reset branch"
    # and it is a direct statement of that branch, not nested in a sub-condition
    assert trues[0] in reset_ifs[-1].body


def test_post_step_ref_is_counted_outside_the_catchup_loop():
    """`post_step_captures_ref` must use the loop's FINAL catchup_steps, so it
    has to sit after the catch-up while, while `post_step_captures` counts each
    executed capture and has to sit inside it."""
    ast, main_node = _main_ast()

    catchup = [n for n in ast.walk(main_node)
               if isinstance(n, ast.While) and "catchup_steps" in ast.dump(n.test)
               and "steps_limit" in ast.dump(n.test)]
    assert len(catchup) == 1, "expected exactly one catch-up while loop"
    lo, hi = catchup[0].lineno, catchup[0].end_lineno

    aug = {}
    for node in ast.walk(main_node):
        if isinstance(node, ast.AugAssign) and getattr(node.target, "id", "") in (
                "log_post_step_captures", "log_post_step_captures_ref"):
            aug.setdefault(node.target.id, []).append(node)

    inside = aug["log_post_step_captures"]
    assert len(inside) == 1 and lo < inside[0].lineno <= hi, \
        "post_step_captures must be incremented inside the catch-up loop"

    outside = aug["log_post_step_captures_ref"]
    assert len(outside) == 1 and outside[0].lineno > hi, \
        "post_step_captures_ref must be summed AFTER the catch-up loop"
    # ...and it is min(catchup_steps, 2), the count interpolation consumes
    assert "min" in ast.dump(outside[0].value) and \
        "catchup_steps" in ast.dump(outside[0].value)


def test_override_capture_is_the_only_one_behind_the_skip_gate():
    """Only the tag="override" call may be conditional; init / reset / post_step
    must stay unconditional."""
    ast, main_node = _main_ast()
    gates = [n for n in ast.walk(main_node)
             if isinstance(n, ast.If) and "_need_capture" in ast.dump(n.test)]
    assert len(gates) == 1
    gated = [c for c in ast.walk(gates[0])
             if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "capture_state"]
    assert len(gated) == 1
    assert [kw.value.value for kw in gated[0].keywords if kw.arg == "tag"] == ["override"]


def test_l3_reports_one_kind_and_flags_the_count_as_not_observable(monkeypatch):
    """L3 builds ONE shared kind and prints the resolved table once
    (l3_runtime: print_resolved_table("L3-shared")), so k_observed is 1 for any
    fleet size. kinds=1 is right; k_observed is NOT a target count there, and
    the kind gate is L2-only.

    v1.6.8: the L3 server DOES print a [SERVE] line now — bare `[SERVE] [L3]`
    with the six window-shape keys and none of the L2-only serving counters.
    Until v1.6.7 it printed none, and this test asserted serve_windows == 0;
    that assertion would have stayed GREEN after the change (the canned input
    is the test's own, and simply had no [SERVE] line in it), so the replay
    input is updated here rather than left to pass while describing a server
    that no longer exists."""
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm

    lines = [" [Genesis] [MODE] === L3 (multi-env) ===\n",
             f"{bm._TARGET_TOKEN}L3-shared\n",
             " [Genesis] Initialization Complete. Obstacles: 0\n"]
    for loop, phys in ((9.00, 5.00), (4.00, 2.50), (6.00, 3.50)):
        lines.append(f" [STATS] [L3] Loop Avg: {loop:.2f} ms | "
                     f"Physics Avg: {phys:.2f} ms "
                     f"(1.0 steps/loop, 2.50 ms/step) [cap=5:burst]\n")
        lines.append(f" [SERVE] [L3] loops=50 steps=50 zero_step_loops=0 "
                     f"serve_ms={50 * (loop - phys):.3f} "
                     f"phys_ms_sum={50 * phys:.3f} window_ms=1250.0\n")

    r = _replay(bm, monkeypatch, "L3", lines, k=100, kinds=1, measure_stats=3)
    assert r["ok"] is True
    assert r["k_observed"] == 1 and r["reuse_lines"] == 0
    assert r["kinds"] == 1
    assert r["kinds_observable"] is False        # 100 vehicles, K reads 1
    assert r["serve_windows"] == 2               # L3 has [SERVE] since v1.6.8
    assert r["serve_windows_match"] is True
    assert r["weighting"] == "loops"             # L3 is re-weighted like L2
    assert r["serving_ms"] == pytest.approx(2.0)
    # The L2-only counters must be ABSENT from the sums, not zero: the reader
    # sums BY KEY NAME, so a `sends=0` on an L3 line would read as "L3 sent
    # state zero times" when L3 sends every loop.
    for absent in ("recv_loops", "sends", "send_skips", "send_flushes",
                   "post_step_captures", "post_step_captures_ref",
                   "nonskip_loops", "skipped_captures", "cap_override_us"):
        assert absent not in r["serve_sum"], absent
    # ...and the tag must not leak `n_envs` in as a summed metric.
    assert "n_envs" not in r["serve_sum"]


def test_serve_window_alignment_is_guarded_against_the_reader_race():
    """The reader sets stats_done on the [STATS] line while the matching [SERVE]
    line is still unread, so run_config must both wait for it and record a
    mismatch when it never arrives."""
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm
    src = inspect.getsource(bm.run_config)
    assert "_grace_deadline" in src
    assert "while len(serve_rows) < len(stats)" in src
    assert "serve_windows_match=" in src
    # rows and the [SERVE] alignment must come from ONE snapshot of stats
    assert "_stats = list(stats)" in src
    assert "rows = _stats[1:]" in src
    assert "serve_rows[1:len(_stats)]" in src


def test_run_config_short_serve_stream_is_flagged_not_silently_averaged(monkeypatch):
    """One [SERVE] line missing (the race, or a server killed mid-window) must
    surface as serve_windows_match False rather than a quietly smaller sum."""
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import benchmark as bm

    serve = (" [SERVE] [L2] recv_loops=50 nonskip_loops=0 skipped_captures=50 "
             "post_step_captures=50 post_step_captures_ref=50\n")
    lines = [f"{bm._TARGET_TOKEN}0\n",
             " [Genesis] Initialization Complete. Obstacles: 0\n"]
    for i, (loop, phys) in enumerate(((9.00, 5.00), (4.00, 2.50), (6.00, 3.50))):
        lines.append(f" [STATS] [L2] Loop Avg: {loop:.2f} ms | "
                     f"Physics Avg: {phys:.2f} ms "
                     f"(1.0 steps/loop, 2.50 ms/step) [cap=5:burst]\n")
        if i < 2:                     # the LAST window's [SERVE] never arrives
            lines.append(serve)

    r = _replay(bm, monkeypatch, "L2", lines, k=1, kinds=1, measure_stats=3)
    assert r["n"] == 2
    assert r["serve_windows"] == 1
    assert r["serve_windows_match"] is False
    # v1.6.8: an unaligned [SERVE] stream also costs the pooled weighting —
    # the record must SAY so rather than quietly mixing estimators between
    # two arms of a comparison.
    assert r["weighting"] == "unweighted"
    assert r["loops_total"] is None and r["loops_min"] is None
    assert r["duty"] != r["duty"]        # NaN: D is not computable here
