"""Official OSC-server benchmark — the SDK server driven end-to-end over the
real wire by a mock UE client (v1.0.20).

| abbr | meaning |
|---|---|
| L2 / L3 | per-entity (K interacting entities × 1 env) / multi-env (1 entity × n_envs) |
| simple / complex | plane ground only / plane + 88 static convex-hull structures |
| ms/step | cost of one physics step (the `[STATS]` per-step value) |
| steps/loop | catch-up steps per loop (~1.0 = holding real-time, pinned at cap = saturated) |
| serving | Loop Avg - Physics Avg: the non-physics part of the loop (state capture, interpolation, OSC encode/send, pacing) |
| kind | one batched vehicle configuration; targets sharing (URDF, mapping, friction) share one |
| n | number of `[STATS]` windows averaged for a config (the first is dropped) |
| loops | loops the server ran inside one window; the WEIGHT of every headline average since v1.6.8 |
| D | serving duty cycle = Σ`serve_ms` / Σ`window_ms` (share of wall clock spent on non-physics loop work) |
| S' | serving ms per physics step = Σ`serve_ms` / Σ`steps` (dt-free) |

Since v1.6.8 a window is a fixed WALL-CLOCK interval (`--stats-interval`,
default 1.0 s) rather than a fixed 50 loops, so its loop count varies and the
headline averages are pooled over `loops` instead of averaged over windows.
Numbers from an older run (50-loop windows, unweighted means) are not directly
comparable; the `weighting` field of each record says which estimator ran.

For every configuration in {L2, L3} × {simple, complex} × {1, 10, 30, 100}
tanks this script (default matrix now includes 200 and 400):

1. launches the REAL server (`python -m genesis_vehicle.server [--multi-env]
   --headless`) as a subprocess (CPU backend — the SDK default; ``--gpu`` to
   opt in),
2. acts as the UE client over OSC loopback: sends ``/Genesis/Init/Physics``
   (dt = 0.025), ``/Genesis/Vehicle/Init`` (tank URDF + SkidSteer mapping →
   ``tank_skid_belt``), K ``/Init/Target``s, the 88 convex-hull
   ``/Init/Obstacle``s (complex only), ``/Init/Done``,
3. streams ``/Genesis/Vehicle/Control`` driving inputs at ~30 Hz
   (``--input-hz`` to change it — the default is the rate every published
   table here was measured at),
4. collects the server's ``[MODE]`` / ``[PROFILE]`` / ``[STATS]`` / ``[SERVE]``
   lines plus the per-target init lines that reveal the batched-kind count,
   then sends ``/Genesis/Control "stop"``,
5. prints one summary table (per-mode speed + serving split + real-time
   verdict), with the OBSERVED kind count and sample size n in the record.

Run (from the workspace root, venv active):

    python -m genesis_vehicle.server.benchmark                     # full 16-config matrix
    python -m genesis_vehicle.server.benchmark --tanks 1,10 --terrain simple
    python -m genesis_vehicle.server.benchmark --modes L3 --gpu    # L3 on GPU
    python -m genesis_vehicle.server.benchmark --urdf /path/to/tank.urdf

The tank URDF defaults to the SDK's bundled ``samples/urdf/tank_ray.urdf``
repo (the workspace layout); pass ``--urdf`` explicitly elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

from pythonosc.udp_client import SimpleUDPClient

# Non-default ports so a stray real server/UE session doesn't collide.
RECV_PORT = 7101      # server listens here (we send to it)
SEND_PORT = 7102      # server sends state here (nobody listens — fine over UDP)
OBS_PORT = 7104

# Physics step duration this benchmark asks the server for, in SECONDS. One
# constant because all three uses are the same physical quantity: the
# ``/Genesis/Init/Physics`` payload, the banner, and the real-time verdict's
# per-loop budget. They were three separate literals until v1.6.8, where
# changing the payload alone would have silently falsified both the banner and
# the O/X column. NB the budget below is ``WIRE_DT * 1000.0`` — milliseconds.
WIRE_DT = 0.025
# Minimum loops in the SMALLEST measured window before the record stops being
# flagged !!low-sample. A saturated server at a 1 s window holds ~40 loops; a
# K=400 config at ~200 ms/loop holds ~5 and needs --stats-interval raised.
MIN_WINDOW_LOOPS = 20

_STATS_RE = re.compile(
    r"\[STATS\].*Loop Avg:\s*([\d.]+)\s*ms\s*\|\s*Physics Avg:\s*([\d.]+)\s*ms"
    r"\s*\((\d+(?:\.\d+)?)\s*steps/loop,\s*([\d.]+)\s*ms/step\)")
_CAP_RE = re.compile(r"\[cap=(\d+):([a-z0-9:]+)\]")
# The server's per-window serving counters, printed on their OWN line (the
# ``[STATS]`` string is frozen — two separate ``_STATS_RE`` copies parse it, here
# and in benchmark_collision.py). Shape:
#   " [SERVE] [L2] recv_loops=50 nonskip_loops=0 skipped_captures=50 ..."
_SERVE_RE = re.compile(r"\[SERVE\]")
_SERVE_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(-?[\d.]+)")
# Per-target / per-kind lines the server prints during init. ``reusing shared cfg``
# is emitted once per target that JOINS an existing batched kind
# (vehicle_builder.build_cfg), ``  Target ID: `` once per target from
# print_resolved_table — so kinds == (targets seen) - (reuses seen).
_REUSE_TOKEN = "reusing shared cfg"
_TARGET_TOKEN = "  Target ID: "


def _default_urdf() -> str:
    here = os.path.dirname(os.path.abspath(__file__))          # .../genesis_vehicle/server
    return os.path.abspath(os.path.join(here, "..", "samples", "urdf", "tank_ray.urdf"))


def _make_hull_obj() -> str:
    """A small box mesh the 88 'convex hull' structures point at (obs_type=5 +
    [Simple:Aggregate] → env_builder's exact-convex-hull path, same as UE)."""
    import trimesh
    box = trimesh.creation.box(extents=(2.0, 2.0, 3.0))
    path = os.path.join(tempfile.gettempdir(), "gv_bench_hull.obj")
    box.export(path)
    return path


def _tank_grid(k: int):
    """K spawn positions on a 10-per-row grid, 15 m spacing, clear of the hulls."""
    return [(-70.0 + (i % 10) * 15.0, -70.0 + (i // 10) * 15.0, 1.2) for i in range(k)]


class MockUEClient:
    """The UE side of the wire, minimally: init burst + input streaming."""

    def __init__(self, k: int, urdf: str, complex_terrain: bool, hull_obj: str,
                 kinds: int = 1, input_hz: float = 30.0):
        self.cli = SimpleUDPClient("127.0.0.1", RECV_PORT)
        self.k = k
        self.urdf = urdf
        self.complex_terrain = complex_terrain
        self.hull_obj = hull_obj
        # ``kinds``: how many DISTINCT batched vehicle kinds the K targets should
        # resolve to. ``build_cfg``'s cache key is
        # ``(abspath(urdf), float(friction), mapping)`` (vehicle_builder.build_cfg),
        # so perturbing ONLY the per-target friction of ``/Init/Target`` splits the
        # fleet into N kinds with no wire-format change. kinds=1 reproduces the
        # historical constant 2.0 bit-for-bit (2.0 + 1e-3*0 == 2.0).
        # NB: the obstacle friction (``/Init/Obstacle``) and the global
        # ``/Genesis/Init/Physics`` friction are deliberately NOT touched — the 88
        # static hull meshes must stay one kind.
        self.kinds = max(1, int(kinds))
        # ``input_hz``: /Genesis/Vehicle/Control streaming rate. Default 30.0 keeps
        # the historical ``time.sleep(1.0 / 30.0)`` exactly, so docs/server.md §2.1
        # ("~30 Hz") and its measured table stay valid. Raise it above the loop rate
        # (1/dt) only for experiments that need every loop to see fresh input.
        self.input_hz = float(input_hz)
        self._stop = threading.Event()
        self._streamer = None

    def send_init_burst(self) -> None:
        c = self.cli
        c.send_message("/Genesis/Init/Physics", [-9.81, WIRE_DT, 2.0])
        c.send_message("/Genesis/Vehicle/Init",
                       [self.urdf, json.dumps({"driveType": 2})])   # 2 = SkidSteer → tank preset
        for tid, (x, y, z) in enumerate(_tank_grid(self.k)):
            # [ID, Type, Px,Py,Pz, Qx,Qy,Qz,Qw, Sx,Sy,Sz, mass, friction, restitution]
            c.send_message("/Init/Target",
                           [tid, 1, x, y, z, 0.0, 0.0, 0.0, 1.0,
                            1.0, 1.0, 1.0, 1.0,
                            2.0 + 1e-3 * (tid % self.kinds), 0.0])
        if self.complex_terrain:
            fields = []
            for s in range(88):
                px = 95.0 + (s % 4) * 6.0
                py = -70.0 + (s // 4) * 6.5
                # [ID, Type=5(mesh), P, Q(x,y,z,w), S, mass, fric, rest, mesh, b_dyn, src]
                fields.append([1000 + s, 5, px, py, 1.5, 0.0, 0.0, 0.0, 1.0,
                               1.0, 1.0, 1.0, 1.0, 2.0, 0.0,
                               self.hull_obj, 0, "[Simple:Aggregate]"])
                if len(fields) == 8:                      # chunk: 8 × 18 args/packet
                    c.send_message("/Init/Obstacle", sum(fields, []))
                    fields = []
            if fields:
                c.send_message("/Init/Obstacle", sum(fields, []))
        c.send_message("/Init/Done", [])

    def start_inputs(self) -> None:
        def run():
            frame = 0
            while not self._stop.is_set():
                frame += 1
                payload = [frame]
                for tid in range(self.k):
                    payload += [tid, 0.1, 0.5, 0.0, 0.0, 0.0]   # steer, throttle, brake, aux
                self.cli.send_message("/Genesis/Vehicle/Control", payload)
                time.sleep(1.0 / self.input_hz)
        self._streamer = threading.Thread(target=run, daemon=True)
        self._streamer.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self.cli.send_message("/Genesis/Control", ["stop"])
        except Exception:
            pass


def run_config(mode: str, terrain: str, k: int, urdf: str, hull_obj: str,
               gpu: bool, measure_stats: int, build_timeout: float,
               python_exe: str, verbose: bool, kinds: int = 1,
               input_hz: float = 30.0, legacy_override_capture: bool = False,
               serve_timers: bool = False, send_hz: "float | None" = None,
               stats_interval: float = 1.0) -> dict:
    """Launch server + mock client for one config; return parsed results.

    The returned record carries the OBSERVED kind count (``kinds``,
    ``k_observed``, ``reuse_lines``) and the per-arm sample size ``n`` so a
    comparison does not depend on the operator remembering to note them.

    ``stats_interval`` (v1.6.8) is the server's ``--stats-interval``: the
    wall-clock length of one ``[STATS]``/``[SERVE]`` window. It sets this
    call's runtime (about ``measure_stats * stats_interval`` seconds of
    simulation) and the per-window sample count, reported as
    ``loops_total`` / ``loops_min`` / ``loops_max``.

    WEIGHTING (v1.6.8). The five headline averages are pooled over the
    window's own ``loops``, taken from the ``[SERVE]`` keys, NOT averaged over
    windows. Before v1.6.8 every window was exactly 50 loops, so an unweighted
    mean of the per-window means happened to equal the pooled mean; with
    wall-clock windows the loop count varies and it does not. The record's
    ``weighting`` field says which estimator produced the numbers
    (``"loops"``, or ``"unweighted"`` when the ``[SERVE]`` keys are missing or
    unaligned) — figures from the two are NOT comparable.
    """
    cmd = [python_exe, "-m", "genesis_vehicle.server", "--headless",
           "--road-raycast-only", "--pacing-profile",
           "--stats-interval", str(stats_interval),
           "--recv_port", str(RECV_PORT), "--send_port", str(SEND_PORT),
           "--send_port_obs", str(OBS_PORT)]
    if mode == "L3":
        cmd.append("--multi-env")
    if legacy_override_capture:
        cmd.append("--legacy-override-capture")
    if serve_timers:
        cmd.append("--serve-timers")
    # `is not None`, not truthiness: send_hz=0.0 must NOT be silently dropped
    # here while the record below still reports send_hz=0.0 — that would make
    # the record claim a rate the server never received. Forwarded, the server
    # rejects it in argparse and this arm fails loudly instead.
    if send_hz is not None:
        cmd += ["--send-hz", str(send_hz)]
    if gpu:
        # Both modes accept --gpu (since v1.0.14). For L2 the GPU parallelizes
        # over LINKS within the one env (23·K of them) rather than over envs —
        # measured here to locate the L2 GPU crossover, if any.
        cmd.append("--gpu")

    env = dict(os.environ)
    env.setdefault("GENESIS_VEHICLE_QUIET", "1")
    env["PYTHONUNBUFFERED"] = "1"     # piped child stdout is block-buffered otherwise
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env,
                            cwd=os.path.abspath(os.path.join(
                                os.path.dirname(os.path.abspath(__file__)), "..", "..")))

    init_done = threading.Event()
    stats: list[tuple] = []
    profile_line: list[str] = []
    mode_line: list[str] = []
    pacing_events: list[str] = []
    serve_rows: list[dict] = []
    counts = {"reuse_lines": 0, "k_observed": 0}
    stats_done = threading.Event()

    def reader():
        for line in proc.stdout:                      # noqa: B007
            if verbose:
                sys.stdout.write(f"    | {line}")
            if "Initialization Complete" in line:
                init_done.set()
            elif "[PROFILE]" in line:
                profile_line.append(line.strip())
            elif "[MODE]" in line:
                mode_line.append(line.strip())
            elif "[AdaptiveCatchup]" in line:
                pacing_events.append(line.strip())
            # The two init-time counters that make the kind count OBSERVABLE.
            # Without these branches both lines fall into the else and are
            # discarded unless -v is passed, and the bench can only ASSUME how
            # many batched kinds the server actually built.
            elif _REUSE_TOKEN in line:
                counts["reuse_lines"] += 1
            elif line.startswith(_TARGET_TOKEN):
                counts["k_observed"] += 1
            elif _SERVE_RE.search(line):
                # Only key=value windows count as rows; a [SERVE]-tagged line
                # with no pairs would shift the window alignment below.
                _kv = {kk: float(vv) for kk, vv in _SERVE_KV_RE.findall(line)}
                if _kv:
                    serve_rows.append(_kv)
            else:
                m = _STATS_RE.search(line)
                if m:
                    cm = _CAP_RE.search(line)
                    stats.append(tuple(float(g) for g in m.groups())
                                 + ((cm.group(2) if cm else "?"),))
                    if len(stats) >= measure_stats:
                        stats_done.set()
    t_reader = threading.Thread(target=reader, daemon=True)
    t_reader.start()

    client = MockUEClient(k, urdf, terrain == "complex", hull_obj,
                          kinds=kinds, input_hz=input_hz)
    result = dict(mode=mode, terrain=terrain, k=k, ok=False,
                  kinds_requested=kinds, input_hz=input_hz,
                  legacy_override_capture=legacy_override_capture,
                  serve_timers=serve_timers, send_hz=send_hz,
                  stats_interval=stats_interval, dt=WIRE_DT)
    t0 = time.time()
    try:
        # init burst until the server acknowledges (it polls RequestInit at 1 Hz)
        while not init_done.is_set():
            if proc.poll() is not None or time.time() - t0 > 60.0:
                result["error"] = "init timeout / server died"
                return result
            client.send_init_burst()
            init_done.wait(timeout=1.0)

        client.start_inputs()
        if not stats_done.wait(timeout=build_timeout):
            result["error"] = f"no {measure_stats} [STATS] within {build_timeout:.0f}s (got {len(stats)})"
            return result

        # RACE: the reader sets stats_done while handling the [STATS] line, and
        # the matching [SERVE] line is the NEXT line on the same stream — so the
        # main thread can get here before that line is parsed and lose the last
        # window. Since serve_sum scales with the window count, that would make
        # two arms silently incomparable. Both guards are applied: a bounded
        # grace wait here, AND serve_windows != n recorded as an explicit
        # rejection reason below (a slow/absent [SERVE] stream still exits).
        _grace_deadline = time.time() + 2.0
        while len(serve_rows) < len(stats) and time.time() < _grace_deadline:
            time.sleep(0.02)
        # Snapshot once: the reader keeps appending until the server stops, so
        # rows and the [SERVE] alignment must be taken from the SAME view.
        _stats = list(stats)
        # drop the first STATS line (JIT/settle warm-up), average the rest
        rows = _stats[1:] if len(_stats) > 1 else _stats
        n = len(rows)
        # [SERVE] windows align 1:1 with [STATS] windows (both printed at the
        # same window boundary), so drop the first one too. A short/absent
        # [SERVE] stream (a server without the counters) leaves empty sums.
        # Since v1.6.8 BOTH modes print the line; a pre-1.6.8 L3 server does not.
        s_rows = serve_rows[1:len(_stats)] if len(serve_rows) > 1 else []
        serve_sum: dict = {}
        for sr in s_rows:
            for kk, vv in sr.items():
                serve_sum[kk] = serve_sum.get(kk, 0.0) + vv
        k_observed = counts["k_observed"]
        reuse_lines = counts["reuse_lines"]

        # ---- headline averages: pooled over loops, from the [SERVE] sums ----
        # The windows are wall-clock since v1.6.8 and hold DIFFERENT loop
        # counts, so the historical `mean of the per-window means` is biased.
        # The weights come from [SERVE] (`loops`, `steps`, `serve_ms`,
        # `phys_ms_sum`), never from [STATS]: [STATS] prints .2f/.1f, so on a
        # server with headroom Loop Avg reads 0.00 and steps/loop 0.0 — the
        # information is already gone and no weighting recovers it.
        _windows_aligned = (len(s_rows) == n) and bool(s_rows)
        _have_keys = _windows_aligned and all(
            {"loops", "steps", "serve_ms", "phys_ms_sum"} <= set(sr)
            for sr in s_rows)
        if _have_keys:
            _loops = serve_sum["loops"]
            _steps = serve_sum["steps"]
            _serve = serve_sum["serve_ms"]
            _phys = serve_sum["phys_ms_sum"]
            _win = serve_sum.get("window_ms", 0.0)
            # Sum(loops) and Sum(window_ms) cannot structurally be zero here
            # (a window exists only because a loop closed it), but Sum(steps)
            # CAN be zero: every loop in every measured window may be a
            # zero-step busy-wait loop. Guard it the way the server guards the
            # same quotient (max(log_step_sum, 1)) rather than dying.
            _steps_d = _steps if _steps > 0 else 1.0
            weighting = "loops"
            loop_ms = (_serve + _phys) / _loops
            phys_ms = _phys / _loops
            steps_per_loop = _steps / _loops
            # NB: unlike the other five columns, this one weights by `steps`,
            # not `loops`.
            ms_per_step = _phys / _steps_d
            serving_ms = _serve / _loops
            # D = serving duty cycle; S' = serving ms per physics step. Both
            # are DERIVED here and never printed per window, because a ratio
            # is not summable across windows.
            duty = (_serve / _win) if _win > 0 else float("nan")
            s_prime = _serve / _steps_d
            _lo = [sr["loops"] for sr in s_rows]
            loops_total, loops_min, loops_max = _loops, min(_lo), max(_lo)
        else:
            # LOUD fallback: the pre-v1.6.8 unweighted estimator. Recorded in
            # `weighting` and marked !!unweighted on the progress line — an
            # arm computed this way is not comparable with a "loops" arm.
            weighting = "unweighted"
            loop_ms = sum(r[0] for r in rows) / n
            phys_ms = sum(r[1] for r in rows) / n
            steps_per_loop = sum(r[2] for r in rows) / n
            ms_per_step = sum(r[3] for r in rows) / n
            serving_ms = (sum(r[0] for r in rows) - sum(r[1] for r in rows)) / n
            duty = s_prime = float("nan")
            loops_total = loops_min = loops_max = None
        result.update(
            ok=True,
            loop_ms=loop_ms,
            phys_ms=phys_ms,
            steps_per_loop=steps_per_loop,
            ms_per_step=ms_per_step,
            serving_ms=serving_ms,
            weighting=weighting,
            duty=duty,
            s_prime=s_prime,
            loops_total=loops_total,
            loops_min=loops_min,
            loops_max=loops_max,
            low_sample=(loops_min is not None and loops_min < MIN_WINDOW_LOOPS),
            # n IS part of the record: a timeout that shortens ONE arm silently
            # changes what the averages mean, so an n mismatch between two arms
            # is a rejection reason, not a footnote.
            n=n,
            # L2 ONLY. Observed, not intended: a dropped /Init/Target makes the
            # fleet smaller with no other symptom. Under L3 the server builds ONE
            # shared kind and prints the resolved table once
            # (l3_runtime.print_resolved_table("L3-shared")), so k_observed is 1
            # for ANY fleet size — kinds=1 is then correct, but k_observed is NOT
            # a target count and the kind gate does not apply.
            k_observed=k_observed,
            reuse_lines=reuse_lines,
            kinds=k_observed - reuse_lines,
            kinds_observable=(mode == "L2"),
            serve_windows=len(s_rows),
            # A serve_windows != n arm is NOT comparable with another arm: the
            # [SERVE] sums below cover a different number of windows than the
            # ms averages do — and since v1.6.8 it also forces the headline
            # averages onto the unweighted fallback. Same standing as an n
            # mismatch.
            serve_windows_match=(len(s_rows) == n or not s_rows),
            serve_sum=serve_sum,
            serve_rows=list(serve_rows),
            build_s=time.time() - t0,
            profile=profile_line[-1] if profile_line else "",
            mode_banner=mode_line[-1] if mode_line else "",
            pacing_events=list(pacing_events),
            pacing_mode=rows[-1][4] if rows else "?",           # mode from the last [STATS]
            pacing_switches=sum("switch#" in e for e in pacing_events),
        )
        return result
    finally:
        client.stop()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modes", default="L2,L3")
    ap.add_argument("--terrain", default="simple,complex")
    ap.add_argument("--tanks", default="1,10,30,100,200,400")
    ap.add_argument("--urdf", default=_default_urdf())
    ap.add_argument("--gpu", action="store_true",
                    help="L3 on the GPU backend (CPU is the SDK default)")
    ap.add_argument("--stats", type=int, default=5,
                    help="[STATS] lines per config (one --stats-interval each; "
                         "first is dropped)")
    ap.add_argument("--stats-interval", type=float, default=1.0,
                    help="pass --stats-interval SEC to the server: the "
                         "wall-clock length of one [STATS]/[SERVE] window "
                         "(default 1.0). Before v1.6.8 the window was 50 "
                         "LOOPS, which on a server with headroom closed in "
                         "~0.1 ms and measured nothing. This value fixes the "
                         "runtime of each config (about --stats x this many "
                         "seconds) and the per-window sample count: check "
                         "loops_min on the progress line and RAISE the "
                         "interval if the arm is flagged !!low-sample "
                         f"(loops_min < {MIN_WINDOW_LOOPS}). Two arms must "
                         "use the SAME interval.")
    ap.add_argument("--build-timeout", type=float, default=900.0)
    ap.add_argument("--kinds", type=int, default=1,
                    help="split the K targets into N distinct batched vehicle "
                         "KINDS by varying only the per-target /Init/Target "
                         "friction (2.0 + 1e-3*(tid %% N)); build_cfg's cache "
                         "key is (urdf, friction, mapping). Default 1 = the "
                         "historical single shared kind, bit-identical. The "
                         "OBSERVED kind count (targets seen minus cfg reuses) "
                         "is reported on the per-config progress line and as the "
                         "'kinds' column of the summary table; it is L2-only "
                         "(L3 builds one shared kind). NB: N>1 is not purely a "
                         "batching change - that friction is also the chassis "
                         "material friction and the wheel mu fallback, so arms "
                         "at different --kinds are not physically identical.")
    ap.add_argument("--input-hz", type=float, default=30.0,
                    help="/Genesis/Vehicle/Control streaming rate (default 30.0 "
                         "= the historical rate docs/server.md §2.1's table was "
                         "measured at; do not compare tables across rates). "
                         "benchmark_collision.py has its own streamer pinned at "
                         "30 Hz and does NOT take this flag, so using it splits "
                         "the two benchmarks' input rates.")
    ap.add_argument("--legacy-override-capture", action="store_true",
                    help="pass --legacy-override-capture to the server (the "
                         "pre-1.6.6 unconditional post-override capture_state). "
                         "For A/B arms only.")
    ap.add_argument("--serve-timers", action="store_true",
                    help="pass --serve-timers to the server. When comparing two "
                         "arms, run it in BOTH arms — otherwise the arms differ "
                         "by one flag.")
    ap.add_argument("--send-hz", type=float, default=None,
                    help="pass --send-hz H to the server (L2 state-send "
                         "downsampling; physics still runs at dt). Default "
                         "None = the server default, one send per loop. For "
                         "A/B arms: run one arm with it and one without, and "
                         "ALTERNATE the arm order — a single-process "
                         "sequential measurement in this repo has been shown "
                         "to depend on slot order. Check the B arm's "
                         "effective rate 1000*sum(sends)/sum(window_ms) from "
                         "the [serve] summary line: above 1.2*H the arm is "
                         "not comparable.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="echo the server's stdout")
    args = ap.parse_args()

    if not os.path.exists(args.urdf):
        sys.exit(f"tank URDF not found: {args.urdf}  (pass --urdf)")
    hull_obj = _make_hull_obj()
    modes = [m.strip() for m in args.modes.split(",")]
    terrains = [t.strip() for t in args.terrain.split(",")]
    tanks = [int(x) for x in args.tanks.split(",")]

    if args.stats_interval <= 0.0:
        sys.exit(f"--stats-interval must be > 0 seconds (got {args.stats_interval})")

    print(f"[bench] server benchmark — modes={modes} terrain={terrains} "
          f"tanks={tanks} backend={'GPU(L3)' if args.gpu else 'CPU'} "
          f"dt={WIRE_DT:g} rco=on "
          f"kinds={args.kinds} input_hz={args.input_hz:g} "
          f"send_hz={'off' if args.send_hz is None else args.send_hz} "
          f"stats_interval={args.stats_interval:g}s")
    results = []
    for mode in modes:
        for terrain in terrains:
            for k in tanks:
                print(f"[bench] >>> {mode} | {terrain} | {k} tanks ...", flush=True)
                r = run_config(mode, terrain, k, args.urdf, hull_obj, args.gpu,
                               args.stats, args.build_timeout, sys.executable,
                               args.verbose, kinds=args.kinds,
                               input_hz=args.input_hz,
                               legacy_override_capture=args.legacy_override_capture,
                               serve_timers=args.serve_timers,
                               send_hz=args.send_hz,
                               stats_interval=args.stats_interval)
                results.append(r)
                if r.get("ok"):
                    kind_note = (f"kinds {r['kinds']}/{r['kinds_requested']} "
                                 f"(K={r['k_observed']}, reuse={r['reuse_lines']})"
                                 if r["kinds_observable"] else
                                 "kinds 1 (L3: one shared kind, K not observable)")
                    warn = "" if r["serve_windows_match"] else \
                        f"  !! serve_windows={r['serve_windows']} != n={r['n']}"
                    if r.get("weighting") != "loops":
                        warn += "  !!unweighted"
                    if r.get("low_sample"):
                        warn += f"  !!low-sample (loops_min={r['loops_min']}; " \
                                f"raise --stats-interval)"
                    loops_note = (
                        f" | loops={r['loops_total']:g} "
                        f"({r['loops_min']:g}..{r['loops_max']:g})"
                        if r.get("loops_total") is not None else "")
                    print(f"[bench]     {r['ms_per_step']:.2f} ms/step | "
                          f"{r['steps_per_loop']:.1f} steps/loop | "
                          f"Loop {r['loop_ms']:.2f} ms | "
                          f"serving {r['serving_ms']:.2f} ms | "
                          f"{kind_note} | n={r['n']}{loops_note}{warn}",
                          flush=True)
                else:
                    print(f"[bench]     FAILED: {r.get('error')}", flush=True)

    budget_ms = WIRE_DT * 1000.0        # MILLISECONDS — the per-loop dt budget
    # The markdown table below is the artifact that gets pasted into docs and
    # the CHANGELOG, so the weighting has to travel WITH it: a `weighting`
    # column plus the same !! markers the progress line carries. A caption
    # alone is not enough — the fallback to the unweighted estimator is
    # reachable in normal operation (the reader-race grace above, or a
    # pre-v1.6.8 server binary), so a row can differ from its neighbours.
    _ok_rows = [r for r in results if r.get("ok")]
    _all_pooled = bool(_ok_rows) and all(
        r.get("weighting") == "loops" for r in _ok_rows)
    print("\n| mode | terrain | tanks | kinds | n | send_hz | ms/step | "
          "steps/loop | Loop Avg | serving | pacing | realtime | weighting |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if r.get("ok"):
            rt = "O" if (r["steps_per_loop"] <= 1.05 and r["loop_ms"] <= budget_ms) else "X"
            sw = r.get("pacing_switches", 0)
            pacing = r.get("pacing_mode", "?") + (f" ({sw}sw)" if sw else "")
            wtag = r.get("weighting", "?")
            if wtag != "loops":
                wtag += " !!unweighted"
            if r.get("low_sample"):
                wtag += f" !!low-sample (loops_min={r['loops_min']:g})"
            print(f"| {r['mode']} | {r['terrain']} | {r['k']} | "
                  f"{r['kinds'] if r['kinds_observable'] else '1*'} | {r['n']} | "
                  f"{'off' if r.get('send_hz') is None else r['send_hz']} | "
                  f"{r['ms_per_step']:.2f} | {r['steps_per_loop']:.1f} | "
                  f"{r['loop_ms']:.2f} | {r['serving_ms']:.2f} | {pacing} | {rt} | "
                  f"{wtag} |")
        else:
            print(f"| {r['mode']} | {r['terrain']} | {r['k']} | | | "
                  f"{'off' if r.get('send_hz') is None else r['send_hz']} | "
                  f"FAIL: {r.get('error')} | | | | | | |")
    print(f"\n[bench] realtime O = steps/loop <= 1.05 AND Loop Avg <= "
          f"{budget_ms:g} ms (dt budget)")
    print("[bench] serving = Loop Avg - Physics Avg (everything in the loop that "
          "is not a physics step: capture, interpolation, OSC encode/send, pacing)")
    if _all_pooled:
        print(f"[bench] every headline average above is POOLED over the "
              f"windows' own loop counts (weighting='loops'), not averaged "
              f"over windows — the windows are {args.stats_interval:g} s of "
              f"WALL CLOCK each (v1.6.8) and hold different loop counts. "
              f"ms/step is the only one weighted by `steps` rather than "
              f"`loops`. Figures from a pre-v1.6.8 run (50-loop windows, "
              f"unweighted means) are NOT directly comparable")
    else:
        _bad = [f"{r['mode']}/{r['terrain']}/{r['k']}" for r in _ok_rows
                if r.get("weighting") != "loops"]
        print(f"[bench] !! MIXED ESTIMATORS — {len(_bad)} of {len(_ok_rows)} "
              f"rows fell back to the pre-v1.6.8 UNWEIGHTED average "
              f"({', '.join(_bad)}); the rest are pooled over loops. The "
              f"rows are NOT comparable with each other. Cause: the [SERVE] "
              f"stream was short/unaligned (see serve_windows != n) or the "
              f"server predates v1.6.8 and prints no window keys. Re-run "
              f"before quoting anything from the table")
    print("[bench] Loop Avg / steps/loop / serving are diluted by the busy-wait "
          "spin loops on a server WITH HEADROOM and mean little there; read "
          "them on a SATURATED server (steps/loop ~ 1.0, Loop Avg ~ dt). The "
          "[serve] D= below does not have that problem")
    print("[bench] kinds = OBSERVED batched vehicle kinds ('Target ID' lines minus "
          "'reusing shared cfg' lines) — L2 ONLY; '1*' means L3, where the server "
          "builds one shared kind and prints the table once, so the count is not "
          "observed. n = [STATS] windows averaged (first dropped)")
    print("[bench] REFUSE to publish a comparison if ANY of these holds: "
          "(1) serve_windows != n on either arm; (2) the arms' total `steps` "
          "differ by more than 5% (they did not simulate the same amount); "
          "(3) only one arm is saturated, or their zero_step_loops/loops "
          "differ materially (different loop populations); (4) the arms' "
          "pacing_switches differ; (5) the runs were not fresh processes with "
          "ALTERNATED arm order and a repeated median; (6) either arm is "
          "flagged !!unweighted (its headline averages came from the "
          "pre-v1.6.8 estimator); (7) either arm is flagged !!low-sample "
          f"(loops_min < {MIN_WINDOW_LOOPS} — raise --stats-interval and "
          "re-measure). n mismatch alone invalidates the comparison")
    print("[bench] --kinds N>1 is NOT a pure batching knob: the per-target "
          "friction it varies is also the chassis material friction, and the "
          "wheel mu_long/mu_lat fall back to it when the URDF omits them, so a "
          "--kinds 3 arm differs from a --kinds 1 arm by up to 0.1% in friction "
          "and its trajectories diverge. Compare arms at the SAME --kinds")
    print("[bench] pacing = final adaptive-catchup mode (Nsw = switch count); "
          "trigger contexts below as [pacing] lines. Compare two --send-hz "
          "arms by pacing_switches, NOT by the [cap=N:mode] token: a window "
          "can round-trip BURST<->SMOOTH and end on the same token. When the "
          "arms differ in switch count, report the mode change and do not "
          "quote a ms delta")
    print("[bench] send_hz = the --send-hz passed to the server ('off' = the "
          "default one-send-per-loop). The [serve] line below carries sends / "
          "send_skips / send_flushes / window_ms for the same windows")
    print("[bench] D = Sum(serve_ms)/Sum(window_ms), the serving DUTY CYCLE: "
          "the share of wall clock spent on non-physics loop work. It EXCLUDES "
          "the windows' own reporting cost (the [STATS]/[SERVE] prints are "
          "stamped after loop_dur, so they land in the next window's "
          "window_ms and in no serve_ms). S' = Sum(serve_ms)/Sum(steps), the "
          "serving cost per physics step (divide by dt for a per-simulated-"
          "second figure). Both need a server that prints the v1.6.8 window "
          "keys; they are omitted otherwise")
    for r in results:
        if r.get("ok") and r.get("profile"):
            print(f"[profile] {r['mode']}/{r['terrain']}/{r['k']}: {r['profile']}")
    for r in results:
        if r.get("ok") and r.get("serve_windows"):
            kv = "  ".join(f"{kk}={vv:g}" for kk, vv in sorted(r["serve_sum"].items()))
            # D / S' are DERIVED, never summed — omitted silently on a server
            # that does not print the v1.6.8 window keys.
            d = r.get("duty")
            sp = r.get("s_prime")
            extra = ""
            if d is not None and d == d:            # not NaN
                extra += f"  D={d:.4f}"
            if sp is not None and sp == sp:
                extra += f"  S'={sp:.4f}ms/step"
            # loops_min/loops_max are NOT in serve_sum (it only sums), and
            # refuse rule 7 is phrased in loops_min — so print the spread
            # explicitly rather than leaving only the summed `loops`.
            if r.get("loops_total") is not None:
                extra += (f"  [loops={r['loops_total']:g} "
                          f"({r['loops_min']:g}..{r['loops_max']:g})"
                          + ("  !!low-sample" if r.get("low_sample") else "")
                          + "]")
            print(f"[serve]   {r['mode']}/{r['terrain']}/{r['k']}: "
                  f"windows={r['serve_windows']}  {kv}{extra}")
    for r in results:
        for ev in r.get("pacing_events", []):
            print(f"[pacing]  {r['mode']}/{r['terrain']}/{r['k']}: {ev}")


if __name__ == "__main__":
    main()
