"""Collision-stress OSC-server benchmark — a ring of tanks converging on the
center of a plane until they all pile up (v1.1.1).

| abbr | meaning |
|---|---|
| L2 | per-entity mode (K interacting entities × 1 env) — the only mode in this bench |
| ms/step | cost of one physics step (the `[STATS]` per-step value) |
| R | spawn circle radius (m) |
| v_peak / v_end | peak mean speed during approach / mean speed at the end (m/s) |
| steps/loop | catch-up steps per loop (~1.0 = holding real-time) |

The official matrix benchmark (``genesis_vehicle.server.benchmark``) spawns
tanks on a 15 m grid, so they never touch — it measures the contact-free
cost. This companion measures the OTHER regime: K tanks are placed evenly on
a circle of radius R, each rotated to face the center, and driven straight
at identical constant throttle until they all collide in the middle. Two
"speed drops" fall out. Steering is a small P-controller that keeps each
tank aimed at the origin (see ``RingClient``), so after the first impact the
tanks stay pressed together instead of ricocheting out of the ring — the
measured "post" regime is sustained K-way contact:

1. **Physical** — mean tank speed collapses when the pile-up forms
   (v_peak → v_end), which doubles as proof the collision actually happened.
2. **Simulation** — ms/step before vs after the pile-up, i.e. what K-way
   persistent vehicle-vehicle contact costs the solver.

The server is the REAL one (``python -m genesis_vehicle.server --headless``,
L2, vehicle-vehicle collision on by default), driven over OSC loopback
exactly like ``benchmark.py``; tank positions come back over the wire via
``/Genesis/Vehicle/TargetBulk``, so the whole path (contact solving + state
encode) is exercised.

Run (from the workspace root, venv active):

    python -m genesis_vehicle.server.benchmark_collision              # 10, 30 tanks
    python -m genesis_vehicle.server.benchmark_collision --tanks 30 --duration 40
    python -m genesis_vehicle.server.benchmark_collision --radius 25 --throttle 1.0
    python -m genesis_vehicle.server.benchmark_collision --urdf /path/to/tank.urdf
    python -m genesis_vehicle.server.benchmark_collision --tanks 10 --viewer
        (--viewer: server runs WITH the Genesis viewer window so you can watch
         the pile-up — rendering skews the timing, numbers indicative only)

The tank URDF defaults to ``GeneVehicle_KDU/tank_ray.urdf`` next to the SDK
repo (the workspace layout); pass ``--urdf`` explicitly elsewhere.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import threading
import time

from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

# Non-default ports, distinct from benchmark.py's 7101/7102/7104, so the two
# benchmarks (or a stray real server/UE session) never collide.
RECV_PORT = 7111      # server listens here (we send to it)
SEND_PORT = 7112      # server streams state here (we DO listen — TargetBulk)
OBS_PORT = 7114

_STATS_RE = re.compile(
    r"\[STATS\].*Loop Avg:\s*([\d.]+)\s*ms\s*\|\s*Physics Avg:\s*([\d.]+)\s*ms"
    r"\s*\((\d+(?:\.\d+)?)\s*steps/loop,\s*([\d.]+)\s*ms/step\)")


def _default_urdf() -> str:
    here = os.path.dirname(os.path.abspath(__file__))          # .../genesis_vehicle/server
    return os.path.abspath(os.path.join(here, "..", "..", "GeneVehicle_KDU", "tank_ray.urdf"))


def _ring_spawns(k: int, radius: float, z: float = 1.2):
    """K spawn poses on a circle: position + yaw quat facing the center.

    Tank i sits at angle θ = 2πi/K; body +X (ISO 8855 forward) must point at
    the origin, so yaw ψ = θ + π. /Init/Target takes Genesis coordinates
    directly, wire quat order (Qx, Qy, Qz, Qw).
    """
    poses = []
    for i in range(k):
        theta = 2.0 * math.pi * i / k
        psi = theta + math.pi
        qz, qw = math.sin(psi / 2.0), math.cos(psi / 2.0)
        poses.append((radius * math.cos(theta), radius * math.sin(theta), z,
                      0.0, 0.0, qz, qw))
    return poses


class StateListener:
    """Listens on SEND_PORT for /Genesis/Vehicle/TargetBulk and keeps the
    latest chassis position per tank (converted back from UE cm / y-flip)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest: dict[int, tuple[float, float, float]] = {}
        self.latest_t: float = 0.0
        disp = Dispatcher()
        disp.map("/Genesis/Vehicle/TargetBulk", self._on_bulk)
        disp.set_default_handler(lambda *a: None)   # RequestInit, Step/Ack, ...
        self._srv = ThreadingOSCUDPServer(("127.0.0.1", SEND_PORT), disp)
        # TargetBulk is ONE datagram for all K tanks (~89 floats/tank incl.
        # wheels ≈ 10.7 KB at K=30) — socketserver's default max_packet_size
        # of 8192 silently truncates it and the stream never parses.
        self._srv.max_packet_size = 65535
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._srv.shutdown()
        self._srv.server_close()

    def _on_bulk(self, address, *args):
        # [ID, Px,Py,Pz(cm, y-flip), Qx,Qy,Qz,Qw(UE frame), NumWheels,
        #  (wheel: 3+4+1)×N, ...] , -1
        now = time.perf_counter()
        pos = {}
        i, n = 0, len(args)
        try:
            while i < n and int(args[i]) != -1:
                tid = int(args[i])
                x = float(args[i + 1]) / 100.0
                y = -float(args[i + 2]) / 100.0
                z = float(args[i + 3]) / 100.0
                # Undo the UE mirror (qx,qy,qz,qw)_UE = (-x, y, -z, w)_Genesis,
                # then yaw about +Z from the Genesis (w,x,y,z) quat.
                qx, qy = -float(args[i + 4]), float(args[i + 5])
                qz, qw = -float(args[i + 6]), float(args[i + 7])
                yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                                 1.0 - 2.0 * (qy * qy + qz * qz))
                pos[tid] = (x, y, z, yaw)
                m = int(args[i + 8])
                i += 9 + m * 8
        except (ValueError, IndexError):
            return          # malformed / truncated packet — skip
        if pos:
            with self.lock:
                self.latest = pos
                self.latest_t = now

    def snapshot(self):
        with self.lock:
            return self.latest_t, dict(self.latest)


class RingClient:
    """The UE side of the wire: ring init burst + center-seeking streaming.

    Steering is a P-controller on heading toward the origin (fed by the
    TargetBulk state stream) rather than a fixed 0: after the first impact
    tanks deflect off each other, and with straight-ahead inputs they would
    simply exit the far side of the ring and disperse. Aiming at the center
    keeps every tank pressing into the pile — the sustained K-way contact
    regime this benchmark exists to measure.
    """

    STEER_KP = 1.0      # steer per rad of heading error (+steer = right = -yaw)

    def __init__(self, k: int, urdf: str, radius: float, throttle: float,
                 listener: StateListener):
        self.cli = SimpleUDPClient("127.0.0.1", RECV_PORT)
        self.k = k
        self.urdf = urdf
        self.radius = radius
        self.throttle = throttle
        self.listener = listener
        self._stop = threading.Event()
        self._streamer = None

    def send_init_burst(self) -> None:
        c = self.cli
        c.send_message("/Genesis/Init/Physics", [-9.81, 0.025, 2.0])
        c.send_message("/Genesis/Vehicle/Init",
                       [self.urdf, json.dumps({"driveType": 2})])   # 2 = SkidSteer → tank preset
        for tid, (x, y, z, qx, qy, qz, qw) in enumerate(_ring_spawns(self.k, self.radius)):
            # [ID, Type, Px,Py,Pz, Qx,Qy,Qz,Qw, Sx,Sy,Sz, mass, friction, restitution]
            c.send_message("/Init/Target",
                           [tid, 1, x, y, z, qx, qy, qz, qw,
                            1.0, 1.0, 1.0, 1.0, 2.0, 0.0])
        c.send_message("/Init/Done", [])

    def _steer_to_center(self, state) -> float:
        x, y, _z, yaw = state
        err = (math.atan2(-y, -x) - yaw + math.pi) % (2.0 * math.pi) - math.pi
        return max(-1.0, min(1.0, -self.STEER_KP * err))

    def start_inputs(self) -> None:
        def run():
            frame = 0
            while not self._stop.is_set():
                frame += 1
                _t, states = self.listener.snapshot()
                payload = [frame]
                for tid in range(self.k):
                    # [ID, Steer, Throttle, Brake, Aux1, Aux2]
                    steer = (self._steer_to_center(states[tid])
                             if tid in states else 0.0)
                    payload += [tid, steer, self.throttle, 0.0, 0.0, 0.0]
                self.cli.send_message("/Genesis/Vehicle/Control", payload)
                time.sleep(1.0 / 30.0)
        self._streamer = threading.Thread(target=run, daemon=True)
        self._streamer.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self.cli.send_message("/Genesis/Control", ["stop"])
        except Exception:
            pass


def run_config(k: int, urdf: str, radius: float, throttle: float,
               duration: float, init_timeout: float,
               python_exe: str, verbose: bool, viewer: bool = False) -> dict:
    """One ring run: launch server, converge, sample, return the timeline."""
    cmd = [python_exe, "-m", "genesis_vehicle.server",
           "--road-raycast-only", "--pacing-profile",
           "--recv_port", str(RECV_PORT), "--send_port", str(SEND_PORT),
           "--send_port_obs", str(OBS_PORT)]
    if not viewer:
        cmd.append("--headless")

    env = dict(os.environ)
    env.setdefault("GENESIS_VEHICLE_QUIET", "1")
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env,
                            cwd=os.path.abspath(os.path.join(
                                os.path.dirname(os.path.abspath(__file__)), "..", "..")))

    init_done = threading.Event()
    stats: list[tuple[float, float, float, float]] = []   # (t, loop_ms, steps/loop, ms/step)

    def reader():
        for line in proc.stdout:                      # noqa: B007
            if verbose:
                sys.stdout.write(f"    | {line}")
            if "Initialization Complete" in line:
                init_done.set()
            else:
                m = _STATS_RE.search(line)
                if m:
                    stats.append((time.perf_counter(), float(m.group(1)),
                                  float(m.group(3)), float(m.group(4))))
    threading.Thread(target=reader, daemon=True).start()

    listener = StateListener()
    listener.start()
    client = RingClient(k, urdf, radius, throttle, listener)
    result = dict(k=k, radius=radius, ok=False)
    t0 = time.time()
    try:
        while not init_done.is_set():
            if proc.poll() is not None or time.time() - t0 > init_timeout:
                result["error"] = "init timeout / server died"
                return result
            client.send_init_burst()
            init_done.wait(timeout=1.0)

        client.start_inputs()

        # Sample the ring every 0.5 s: mean radius + mean speed (finite
        # difference of per-tank xy positions between consecutive samples).
        samples = []            # (t_rel, mean_radius, mean_speed)
        prev = None             # (t, {tid: pos})
        t_run0 = time.perf_counter()
        while time.perf_counter() - t_run0 < duration:
            time.sleep(0.5)
            t_pkt, pos = listener.snapshot()
            if len(pos) < k or (prev is not None and t_pkt <= prev[0]):
                continue        # state stream not up yet / no fresh packet
            mean_r = sum(math.hypot(p[0], p[1]) for p in pos.values()) / k
            if prev is not None:
                dt = t_pkt - prev[0]
                common = [tid for tid in pos if tid in prev[1]]
                speed = sum(math.hypot(pos[tid][0] - prev[1][tid][0],
                                       pos[tid][1] - prev[1][tid][1])
                            for tid in common) / (len(common) * dt)
                samples.append((time.perf_counter() - t_run0, mean_r, speed))
            prev = (t_pkt, pos)

        if len(samples) < 4:
            result["error"] = f"too few state samples ({len(samples)}) — no TargetBulk stream?"
            return result

        # Collision time: first sample (after the ring got moving) where the
        # mean speed falls below half of the running peak.
        v_peak, t_collide = 0.0, None
        for t_rel, _r, v in samples:
            v_peak = max(v_peak, v)
            if t_collide is None and v_peak > 1.0 and v < 0.5 * v_peak:
                t_collide = t_rel
        v_end = sum(v for _t, _r, v in samples[-3:]) / 3
        r_end = samples[-1][1]

        # Split the [STATS] series at the collision instant.
        rows = [(t - t_run0, lm, spl, ms) for (t, lm, spl, ms) in stats if t >= t_run0]
        rows = rows[1:] if len(rows) > 1 else rows      # drop warm-up window
        if t_collide is not None:
            pre = [ms for (t, _l, _s, ms) in rows if t < t_collide]
            post = [ms for (t, _l, _s, ms) in rows if t >= t_collide]
        else:
            pre, post = [ms for (_t, _l, _s, ms) in rows], []

        result.update(
            ok=True,
            samples=samples, stats_rows=rows,
            v_peak=v_peak, v_end=v_end, r_end=r_end, t_collide=t_collide,
            ms_pre=(sum(pre) / len(pre)) if pre else None,
            ms_post=(sum(post) / len(post)) if post else None,
        )
        return result
    finally:
        client.stop()
        listener.stop()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tanks", default="10,30")
    ap.add_argument("--urdf", default=_default_urdf())
    ap.add_argument("--radius", type=float, default=None,
                    help="ring radius (m). Default: max(20, 8·K/2π) — 8 m arc "
                         "spacing so neighbours spawn clear of each other.")
    ap.add_argument("--throttle", type=float, default=0.8)
    ap.add_argument("--duration", type=float, default=30.0,
                    help="measured seconds per config (server is real-time paced)")
    ap.add_argument("--init-timeout", type=float, default=300.0)
    ap.add_argument("--viewer", action="store_true",
                    help="run the server WITH the Genesis viewer window (drops "
                         "--headless) to watch the ring converge and pile up. "
                         "Rendering adds per-step overhead — treat the numbers "
                         "as indicative only, not as reference results.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="echo the server's stdout")
    args = ap.parse_args()

    if not os.path.exists(args.urdf):
        sys.exit(f"tank URDF not found: {args.urdf}  (pass --urdf)")
    tanks = [int(x) for x in args.tanks.split(",")]

    print(f"[bench] collision benchmark — L2 ring convergence | tanks={tanks} "
          f"throttle={args.throttle} dt=0.025 plane-only rco=on CPU")
    if args.viewer:
        print("[bench] VIEWER mode — rendering adds per-step overhead; "
              "numbers are indicative only, NOT reference results")
    results = []
    for k in tanks:
        radius = args.radius if args.radius else max(20.0, 8.0 * k / (2.0 * math.pi))
        print(f"[bench] >>> {k} tanks on R={radius:.1f} m ring ...", flush=True)
        r = run_config(k, args.urdf, radius, args.throttle, args.duration,
                       args.init_timeout, sys.executable, args.verbose,
                       viewer=args.viewer)
        results.append(r)
        if not r.get("ok"):
            print(f"[bench]     FAILED: {r.get('error')}", flush=True)
            continue
        if r["t_collide"] is not None:
            pre = f"{r['ms_pre']:.2f}" if r["ms_pre"] is not None else "—"
            post = f"{r['ms_post']:.2f}" if r["ms_post"] is not None else "—"
            print(f"[bench]     v_peak {r['v_peak']:.2f} m/s → v_end {r['v_end']:.2f} m/s | "
                  f"collision @ {r['t_collide']:.1f}s | ms/step {pre} → {post}", flush=True)
        else:
            print(f"[bench]     no collision detected within {args.duration:.0f}s "
                  f"(v_peak {r['v_peak']:.2f} m/s, mean R still {r['r_end']:.1f} m)",
                  flush=True)

        # Per-config timeline (0.5 s state samples + nearest preceding STATS).
        print(f"\n  timeline (tanks={k}, R={radius:.1f} m):")
        print(f"  {'t (s)':>6}  {'mean R (m)':>10}  {'speed (m/s)':>11}  {'ms/step':>8}")
        rows = r["stats_rows"]
        for t_rel, mean_r, v in r["samples"]:
            ms = next((ms for (ts, _l, _s, ms) in reversed(rows) if ts <= t_rel), None)
            ms_s = f"{ms:8.2f}" if ms is not None else f"{'—':>8}"
            marker = "  <-- collision" if (r["t_collide"] is not None
                                           and abs(t_rel - r["t_collide"]) < 0.26) else ""
            print(f"  {t_rel:>6.1f}  {mean_r:>10.2f}  {v:>11.2f}  {ms_s}{marker}")
        print()

    print("\n| tanks | R (m) | v_peak | v_end | collision t | ms/step pre | ms/step post | slowdown |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        if not r.get("ok"):
            print(f"| {r['k']} | {r['radius']:.1f} | FAIL: {r.get('error')} | | | | | |")
            continue
        tc = f"{r['t_collide']:.1f} s" if r["t_collide"] is not None else "—"
        pre = f"{r['ms_pre']:.2f}" if r["ms_pre"] is not None else "—"
        post = f"{r['ms_post']:.2f}" if r["ms_post"] is not None else "—"
        slow = (f"{r['ms_post'] / r['ms_pre']:.2f}×"
                if r["ms_pre"] and r["ms_post"] else "—")
        print(f"| {r['k']} | {r['radius']:.1f} | {r['v_peak']:.2f} | {r['v_end']:.2f} | "
              f"{tc} | {pre} | {post} | {slow} |")
    print("\n[bench] pre/post = mean ms/step before/after the detected collision "
          "(speed < 50% of peak); slowdown = post/pre — the cost of the pile-up's "
          "persistent vehicle-vehicle contacts")


if __name__ == "__main__":
    main()
