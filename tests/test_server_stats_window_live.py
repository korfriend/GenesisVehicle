"""LIVE window-length check for the v1.6.8 wall-clock ``[STATS]``/``[SERVE]``
window. EXCLUDED from the default run.

| abbr | meaning |
|---|---|
| L2 / L3 | per-entity (K interacting vehicles, ``n_envs=1``) / multi-env (1 entity x ``n_envs``) server mode |
| SERVE | the server's per-window serving-counter stdout line |
| STATS | the server's frozen per-window timing line |
| dt | physics step duration (0.025 s here) |
| UDP | User Datagram Protocol (the OSC transport) |

This spawns real server processes and binds UDP ports, so it is gated behind
an environment variable rather than a pytest marker: this repo has no
``pytest.ini`` / ``pyproject.toml`` / ``conftest.py``, so a custom
``@pytest.mark.slow`` would only raise ``PytestUnknownMarkWarning`` and still
run. Enable with::

    GENESIS_VEHICLE_LIVE_SERVER_TEST=1 python -m pytest \\
        genesis_vehicle/tests/test_server_stats_window_live.py -q -s

WHAT IT PROVES. Over a fixed wall-clock measurement window, each server
``[SERVE]`` window is one ``--stats-interval`` long in BOTH arms (gate off and
``--send-hz 20``) and in BOTH modes, and the L3 line parses into exactly the
six window-shape keys with no ``n_envs`` leaking in from the tag.

WHAT IT DOES NOT PROVE. Nothing about the QUALITY of ``Loop Avg`` /
``steps/loop`` on a server with headroom: those are still diluted by the
zero-step busy-wait loops (this test prints ``zero_step_loops`` so that is
visible rather than hidden). And the ms figures here are not a benchmark —
one process per arm, no order alternation, no repeats.

Conditions for every number printed: CPU/WSL2, genesis-world 1.4.0, dt 0.025,
K=1 tank (``samples/urdf/tank_ray.urdf``), ``--road-raycast-only``, input
30 Hz, ``--stats-interval 1.0``.
"""
import os
import re
import subprocess
import sys
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("GENESIS_VEHICLE_LIVE_SERVER_TEST"),
    reason="live server test: set GENESIS_VEHICLE_LIVE_SERVER_TEST=1 to run "
           "(spawns real server processes and binds UDP ports)")

# Dedicated ports, distinct from the default server ports (7001/7002/7004),
# benchmark.py's (7101/7102/7104), benchmark_collision.py's (7111/...) and
# test_server_send_downsample_live.py's (7141/...).
LIVE_RECV_PORT = 7151
LIVE_SEND_PORT = 7152
LIVE_OBS_PORT = 7154

SIM_DT = 0.025          # MockUEClient.send_init_burst hard-codes it
STATS_INTERVAL = 1.0
WARMUP_S = 2.0          # after the first [SERVE] line: JIT/settle transient
MEASURE_S = 8.0         # fixed wall-clock measurement window

_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(-?[\d.]+)")


def _run_arm(send_hz=None, multi_env=False, stats_interval=STATS_INTERVAL):
    """Spawn one server, drive it with the benchmark's MockUEClient, and
    collect the ``[SERVE]`` lines that land inside a fixed wall-clock window.

    Returns ``(rows, tags)``: the parsed key/value dicts and the raw tag
    tokens (``[L2]`` / ``[L3]``) of those same lines.
    """
    from genesis_vehicle.server import benchmark as bm
    bm.RECV_PORT, bm.SEND_PORT, bm.OBS_PORT = (LIVE_RECV_PORT, LIVE_SEND_PORT,
                                               LIVE_OBS_PORT)
    urdf = bm._default_urdf()
    assert os.path.exists(urdf), urdf
    hull = bm._make_hull_obj()

    cmd = [sys.executable, "-m", "genesis_vehicle.server", "--headless",
           "--road-raycast-only",
           "--stats-interval", str(stats_interval),
           "--recv_port", str(LIVE_RECV_PORT),
           "--send_port", str(LIVE_SEND_PORT),
           "--send_port_obs", str(LIVE_OBS_PORT)]
    if multi_env:
        cmd.append("--multi-env")
    if send_hz:
        cmd += ["--send-hz", str(send_hz)]
    env = dict(os.environ)
    env.setdefault("GENESIS_VEHICLE_QUIET", "1")
    env["PYTHONUNBUFFERED"] = "1"
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env=env, cwd=repo_root)

    init_done = threading.Event()
    first_serve = threading.Event()
    lock = threading.Lock()
    serve = []              # (t_seen, tag, kv)

    def reader():
        for line in proc.stdout:
            if "Initialization Complete" in line:
                init_done.set()
            elif "[SERVE]" in line:
                tag = line.split("[SERVE]", 1)[1].strip().split()[0]
                kv = {k: float(v) for k, v in _KV_RE.findall(line)}
                if kv:
                    with lock:
                        serve.append((time.perf_counter(), tag, kv))
                    first_serve.set()

    threading.Thread(target=reader, daemon=True).start()

    client = bm.MockUEClient(1, urdf, False, hull, kinds=1, input_hz=30.0)
    try:
        t0 = time.time()
        while not init_done.is_set():
            assert proc.poll() is None and time.time() - t0 < 300.0, \
                "server died or never finished init"
            client.send_init_burst()
            init_done.wait(timeout=1.0)
        client.start_inputs()
        # Anchor on the first [SERVE] line, not on init: the server prints
        # "Initialization Complete" seconds before the main loop produces a
        # window, and folding that dead time in would understate the count.
        assert first_serve.wait(timeout=180.0), \
            "the server printed no [SERVE] line at all"
        time.sleep(WARMUP_S)
        t_a = time.perf_counter()
        time.sleep(MEASURE_S)
        t_b = time.perf_counter()
    finally:
        # client.stop() sends /Genesis/Control ["stop"], which the loop only
        # consumes on a loop that RECEIVES data — not a reliable exit signal.
        client.stop()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        time.sleep(0.3)
    with lock:
        inside = [(tag, kv) for (t, tag, kv) in serve if t_a <= t <= t_b]
    return [kv for _tag, kv in inside], [tag for tag, _kv in inside], (t_b - t_a)


def test_stats_window_length_tracks_the_interval_in_both_arms():
    pytest.importorskip("pythonosc")
    pytest.importorskip("genesis")

    rows_a, tags_a, span_a = _run_arm(send_hz=None)
    rows_b, tags_b, span_b = _run_arm(send_hz=20.0)

    for name, rows, span in (("A (--send-hz off)", rows_a, span_a),
                             ("B (--send-hz 20)", rows_b, span_b)):
        wm = [r["window_ms"] for r in rows]
        print(f"\n[live] arm {name}: {len(rows)} [SERVE] windows in "
              f"{span:.2f} s | window_ms min {min(wm):.1f} "
              f"median {sorted(wm)[len(wm) // 2]:.1f} max {max(wm):.1f}")
        print(f"[live]   loops {[int(r['loops']) for r in rows]}")
        print(f"[live]   zero_step_loops "
              f"{[int(r['zero_step_loops']) for r in rows]}")
        print(f"[live]   steps {[int(r['steps']) for r in rows]}  "
              f"serve_ms {[round(r['serve_ms'], 1) for r in rows]}")

    assert all(t == "[L2]" for t in tags_a + tags_b), (tags_a, tags_b)

    for name, rows, span in (("A", rows_a, span_a), ("B", rows_b, span_b)):
        # The headline: a fixed-length window. Before v1.6.8 arm B produced
        # THOUSANDS of ~0.1 ms windows over the same span (measured on the
        # v1.6.7 tree: median window_ms 0.0-0.1; the count itself repeats
        # only to about +-25%, so the median is what carries the claim).
        assert rows, f"arm {name} produced no [SERVE] window at all"
        expect = span / STATS_INTERVAL
        assert expect - 2 <= len(rows) <= expect + 2, \
            f"arm {name}: {len(rows)} windows in {span:.2f} s at " \
            f"{STATS_INTERVAL} s/window"
        for r in rows:
            # Upper bound is loose by one loop: the window closes on the
            # FIRST loop at or after the deadline, so a long loop overshoots.
            assert 900.0 <= r["window_ms"] <= 1200.0, (name, r)
            # the identity the serving counters are built to make decidable
            assert r["sends"] + r["send_skips"] == r["loops"], (name, r)
            assert r["zero_step_loops"] <= r["loops"], (name, r)
            # proven in the [SERVE] comment block; confirmed at runtime here
            assert r["serve_ms"] <= r["window_ms"], (name, r)
            assert r["phys_ms_sum"] >= 0.0, (name, r)
            # ...and the lower bound, which is the RUNTIME backstop for a
            # mis-ordered clock read: serve_ms = (Sum loop_dur - Sum phys)
            # * 1000, so a _t_end stamped before the loop's work makes
            # loop_dur ~0 and drives this NEGATIVE. `serve_ms <= window_ms`
            # above is satisfied by any negative value and cannot see it.
            assert r["serve_ms"] >= 0.0, (name, r)
            # sample size: a 1 s window on this config must not be tiny
            assert r["loops"] >= 20, (name, r)

    # Both arms simulated the same amount, or the serving comparison the new
    # keys exist for would be meaningless (refuse rule 2).
    steps_a = sum(r["steps"] for r in rows_a)
    steps_b = sum(r["steps"] for r in rows_b)
    print(f"[live] steps A={steps_a:.0f} B={steps_b:.0f}  "
          f"ratio {steps_b / steps_a:.4f}")
    assert abs(steps_b - steps_a) <= 0.05 * max(steps_a, steps_b), \
        (steps_a, steps_b)

    # D and S' are now computable for both arms. NOT published as a speedup:
    # one process per arm, fixed order, no repeats (refuse rule 5).
    for name, rows in (("A", rows_a), ("B", rows_b)):
        d = sum(r["serve_ms"] for r in rows) / sum(r["window_ms"] for r in rows)
        sp = sum(r["serve_ms"] for r in rows) / max(sum(r["steps"] for r in rows), 1)
        print(f"[live] arm {name}: D={d:.4f}  S'={sp:.4f} ms/step "
              f"(single process, fixed order — not a benchmark)")
        assert 0.0 <= d <= 1.0


def test_l3_prints_serve_windows():
    """L3 had no [SERVE] line at all before v1.6.8, so its benchmark record
    could not be re-weighted and reported no serving cost. The line it gained
    must carry exactly the six shared keys, with a BARE [L3] tag: the
    benchmark's key/value parser harvests `k=v` from anywhere on the line, so
    an `[L3 n_envs=N]` tag would be summed as a metric."""
    pytest.importorskip("pythonosc")
    pytest.importorskip("genesis")

    rows, tags, span = _run_arm(multi_env=True)
    print(f"\n[live] L3: {len(rows)} [SERVE] windows in {span:.2f} s")
    if rows:
        wm = [r["window_ms"] for r in rows]
        print(f"[live]   window_ms min {min(wm):.1f} max {max(wm):.1f} | "
              f"loops {[int(r['loops']) for r in rows]} | "
              f"zero_step_loops {[int(r['zero_step_loops']) for r in rows]}")
    assert rows, "L3 printed no [SERVE] line"
    assert all(t == "[L3]" for t in tags), tags
    for r in rows:
        assert set(r) == {"loops", "steps", "zero_step_loops", "serve_ms",
                          "phys_ms_sum", "window_ms"}, r
        assert "n_envs" not in r
        assert 900.0 <= r["window_ms"] <= 1200.0, r
        assert 0.0 <= r["serve_ms"] <= r["window_ms"], r
        assert r["phys_ms_sum"] >= 0.0, r
    expect = span / STATS_INTERVAL
    assert expect - 2 <= len(rows) <= expect + 2, (len(rows), span)
