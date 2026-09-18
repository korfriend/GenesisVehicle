"""LIVE arrival-rate check for ``--send-hz`` (v1.6.7). EXCLUDED from the
default run.

| abbr | meaning |
|---|---|
| L2 | per-entity server mode (K interacting vehicles, n_envs=1) |
| H | requested send rate (Hz), the ``--send-hz`` value |
| SERVE | the server's per-window serving-counter stdout line |
| STATS | the server's frozen per-window timing line |
| UDP | User Datagram Protocol (the OSC transport; lossy by design) |

This spawns TWO real server processes and binds UDP ports, so it is gated
behind an environment variable rather than a pytest marker: this repo has no
``pytest.ini`` / ``pyproject.toml`` / ``conftest.py``, so a custom
``@pytest.mark.slow`` would only raise ``PytestUnknownMarkWarning`` and still
run. Enable with::

    GENESIS_VEHICLE_LIVE_SERVER_TEST=1 python -m pytest \\
        genesis_vehicle/tests/test_server_send_downsample_live.py -q -s

MEASUREMENT WINDOW — why this file drives the server itself instead of calling
``benchmark.run_config``. Through v1.6.7 one ``[STATS]``/``[SERVE]`` window was
50 LOOPS. The server busy-waits through the last <= 2 ms before each step
deadline (the ``time_to_wait > 0.002`` branch), and each of those zero-step
loops used to pay for a full lerp + OSC send (~0.2 ms). With the gate ON they
cost ~2 us instead, so the loop rate rises by orders of magnitude and a
50-loop window collapsed to ~0.1 ms: measured on the v1.6.7 tree, the gated
arm produced THOUSANDS of ``[SERVE]`` windows in an 8 s span at a median
``window_ms`` of 0.0-0.1, and ``run_config --stats 5`` returned having
observed ~25 ms of simulation. (The window COUNT repeats only to about
+-25% run to run, so the median and the observed span are the figures that
carry the claim, not the count.)
That finding is FIXED in v1.6.8: the window is a fixed wall-clock interval
(``--stats-interval``, default 1.0 s) in both L2 and L3, so ``run_config`` now
measures ``--stats`` x interval seconds. This file still drives the server
directly because it must count arrivals AT THE CLIENT, which ``run_config``
does not observe (it binds no listener).

WHAT IT PROVES: the arrival RATE at the client drops to ~H, and the server's
own ``[SERVE] sends`` / ``send_skips`` agree with it. Note the un-gated arm is
NOT at 1/dt for the same reason as above: it sends on every loop including the
zero-step busy-wait loops, so it measures well above the 40 Hz pacing rate.
Observed across SEVEN runs of this harness at K=1 / dt=25 ms on CPU/WSL2,
genesis-world 1.4.0, input 30 Hz: 123.0-146.3 Hz. That spread is the reason
only a LOWER bound is asserted for it — the figure is a description of this
machine over those runs, not a target, and it is not a bound: an eighth run
may well land outside it. An earlier revision of this docstring said
"123-144 Hz", which the very next independent run (146.25 Hz) exceeded.

WHAT IT DOES NOT PROVE, explicitly:
  (i)  bit-identity of the packets on the OFF arm. That is a structural
       argument, made by ``test_gate_body_is_frozen_byte_for_byte_against_the_
       pre_gate_code`` (the gate body is the untouched pre-v1.6.7 block) plus
       ``test_off_is_off_on_every_adversarial_clock`` (the OFF predicate is
       unconditionally True).
  (ii) the reset FLUSH path. ``MockUEClient`` never sends a reset or an
       override, so no flush fires here at all; the flush is covered by the
       realised-rate and wiring tests, and by the second-stage field check.
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
           "(spawns two real server processes and binds UDP ports)")

# Dedicated ports, distinct from both the default server ports (7001/7002/7004)
# and the benchmark's own (7101/7102/7104): benchmark.py picks 7102 on the
# premise that NOBODY listens there, and this test does listen.
LIVE_RECV_PORT = 7141
LIVE_SEND_PORT = 7142
LIVE_OBS_PORT = 7144

SIM_DT = 0.025          # MockUEClient.send_init_burst hard-codes it
WARMUP_S = 2.0          # after the FIRST packet: JIT/settle transient
MEASURE_S = 8.0         # fixed wall-clock measurement window


class _BulkCounter:
    """Same shape as benchmark_collision.StateListener, reduced to arrival
    timestamps (``max_packet_size`` raised for the same reason: one TargetBulk
    datagram carries every target and overruns socketserver's 8192 default)."""

    def __init__(self, port):
        from pythonosc.dispatcher import Dispatcher
        from pythonosc.osc_server import ThreadingOSCUDPServer
        self.lock = threading.Lock()
        self.stamps = []
        disp = Dispatcher()
        disp.map("/Genesis/Vehicle/TargetBulk", self._on_bulk)
        disp.set_default_handler(lambda *a: None)
        self._srv = ThreadingOSCUDPServer(("127.0.0.1", port), disp)
        self._srv.max_packet_size = 65535
        self._thread = threading.Thread(target=self._srv.serve_forever,
                                        daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._srv.shutdown()
        self._srv.server_close()

    def _on_bulk(self, address, *args):
        t = time.perf_counter()
        with self.lock:
            self.stamps.append(t)

    def count_between(self, t0, t1):
        with self.lock:
            return sum(1 for t in self.stamps if t0 <= t <= t1)

    def wait_for_first(self, timeout):
        """The server prints "Initialization Complete" well before the main
        loop produces its first TargetBulk — measured 4.6 s on CPU/WSL2 at
        K=1. Anchoring the measurement window on init instead of on the first
        packet silently folds that dead time into the rate (it read 13 Hz for
        a stream whose gaps were a clean 50.0 ms)."""
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            with self.lock:
                if self.stamps:
                    return True
            time.sleep(0.05)
        return False


def _run_arm(send_hz):
    """Spawn one server, drive it with the benchmark's MockUEClient, and count
    TargetBulk arrivals over a fixed wall-clock window.

    Returns ``(rate_hz, serve_rows)``.
    """
    from genesis_vehicle.server import benchmark as bm
    bm.RECV_PORT, bm.SEND_PORT, bm.OBS_PORT = (LIVE_RECV_PORT, LIVE_SEND_PORT,
                                               LIVE_OBS_PORT)
    urdf = bm._default_urdf()
    assert os.path.exists(urdf), urdf
    hull = bm._make_hull_obj()

    cmd = [sys.executable, "-m", "genesis_vehicle.server", "--headless",
           "--road-raycast-only",
           "--recv_port", str(LIVE_RECV_PORT),
           "--send_port", str(LIVE_SEND_PORT),
           "--send_port_obs", str(LIVE_OBS_PORT)]
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
    serve_rows = []
    banner = []

    def reader():
        for line in proc.stdout:
            if "Initialization Complete" in line:
                init_done.set()
            elif "[Send-Rate]" in line:
                banner.append(line.strip())
            elif "[SERVE]" in line:
                kv = {k: float(v) for k, v in
                      re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=(-?[\d.]+)", line)}
                if kv:
                    serve_rows.append(kv)

    threading.Thread(target=reader, daemon=True).start()

    listener = _BulkCounter(LIVE_SEND_PORT)
    listener.start()
    client = bm.MockUEClient(1, urdf, False, hull, kinds=1, input_hz=30.0)
    try:
        t0 = time.time()
        while not init_done.is_set():
            assert proc.poll() is None and time.time() - t0 < 180.0, \
                "server died or never finished init"
            client.send_init_burst()
            init_done.wait(timeout=1.0)
        client.start_inputs()
        assert listener.wait_for_first(120.0), \
            "no TargetBulk arrived at all — the server never started serving"
        time.sleep(WARMUP_S)
        t_a = time.perf_counter()
        time.sleep(MEASURE_S)
        t_b = time.perf_counter()
        n = listener.count_between(t_a, t_b)
        rate = n / (t_b - t_a)
    finally:
        # client.stop() sends /Genesis/Control ["stop"], which the server's
        # `while True:` only consumes on a loop that RECEIVES data — it is not
        # a reliable exit signal, and the loop has no other exit condition.
        # Waiting on it first cost ~20 s of dead time per arm and then killed
        # the process anyway. Terminate first; keep kill() as the fallback.
        client.stop()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        time.sleep(0.3)
        listener.stop()
    assert banner, "the server printed no [Send-Rate] banner"
    return rate, serve_rows, banner[-1] if len(banner) == 1 else " || ".join(banner)


def test_send_hz_gates_the_arrival_rate_at_the_client():
    pytest.importorskip("pythonosc")
    pytest.importorskip("genesis")

    rate_a, rows_a, banner_a = _run_arm(None)
    rate_b, rows_b, banner_b = _run_arm(20.0)

    print(f"\n[live] arm A (--send-hz off): {rate_a:6.2f} Hz  "
          f"{len(rows_a)} [SERVE] windows")
    print(f"[live]   {banner_a}")
    print(f"[live] arm B (--send-hz 20):  {rate_b:6.2f} Hz  "
          f"{len(rows_b)} [SERVE] windows")
    print(f"[live]   {banner_b}")
    print(f"[live] B/A = {rate_b / rate_a:.3f}")

    assert "매 루프" in banner_a
    assert "--send-hz 20.0" in banner_b

    # A: one send per loop. The loop rate is NOT 1/dt (see the module
    # docstring), so only a lower bound at the pacing rate is sound.
    assert rate_a >= 0.8 / SIM_DT, \
        f"arm A {rate_a:.1f} Hz is below the pacing rate {1 / SIM_DT:.0f} Hz " \
        f"— the server is not keeping up, the comparison is not meaningful"
    # B: the gate rate, absolute and independent of the loop rate. +-20%.
    assert 16.0 <= rate_b <= 24.0, rate_b
    assert rate_b < 0.75 * rate_a, (rate_b, rate_a)

    # The server's own counters must tell the same story as the wire.
    assert rows_a and rows_b
    # The identity is against the window's OWN loop count: since v1.6.8 the
    # window is 1 s of wall clock and holds a variable number of loops.
    assert all(r["sends"] + r["send_skips"] == r["loops"] for r in rows_a), rows_a[:3]
    assert all(r["sends"] + r["send_skips"] == r["loops"] for r in rows_b), rows_b[:3]
    assert all(r["send_skips"] == 0 for r in rows_a), "OFF arm skipped a send"
    assert sum(r["send_skips"] for r in rows_b) > 0, "ON arm skipped nothing"
    # MockUEClient sends no reset, so no flush may fire in either arm.
    assert sum(r["send_flushes"] for r in rows_a) == 0
    assert sum(r["send_flushes"] for r in rows_b) == 0
    # Effective send rate from the counters (refuse rule 1: <= 1.2 x H).
    # Consecutive windows TILE the wall clock (_win_t0 = _win_now carries the
    # origin with no gap), so summing sends over summed window_ms is exact —
    # do NOT filter short windows out, that biases the ratio. The first window
    # is dropped because its origin is set before the loop starts and so
    # includes the wait for the client, exactly as benchmark.py drops the
    # first [STATS]/[SERVE] pair.
    for name, rows in (("A", rows_a), ("B", rows_b)):
        body = rows[1:]
        # NOT `if not body: continue` — that let the ONLY quantitative
        # assertion in this test be skipped while the test still reported
        # green. An arm that produced a single window did not measure the
        # send rate at all, and that is a failed measurement, not a pass.
        assert body, (
            f"arm {name} produced {len(rows)} [SERVE] window(s); at least 2 "
            f"are needed (the first is dropped: its origin predates the loop "
            f"and includes the wait for the client). Nothing was measured.")
        eff = 1000.0 * sum(r["sends"] for r in body) / \
            sum(r["window_ms"] for r in body)
        print(f"[live] arm {name} effective send rate from [SERVE]: "
              f"{eff:.2f} Hz over {len(body)} windows")
        if name == "B":
            assert eff <= 1.2 * 20.0, eff
            assert eff >= 0.8 * 20.0, eff
