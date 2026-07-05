"""Official OSC-server benchmark — the SDK server driven end-to-end over the
real wire by a mock UE client (v1.0.20).

| abbr | meaning |
|---|---|
| L2 / L3 | per-entity (K interacting entities × 1 env) / multi-env (1 entity × n_envs) |
| simple / complex | plane ground only / plane + 88 static convex-hull structures |
| ms/step | cost of one physics step (the `[STATS]` per-step value) |
| steps/loop | catch-up steps per loop (~1.0 = holding real-time, pinned at cap = saturated) |

For every configuration in {L2, L3} × {simple, complex} × {1, 10, 30, 100}
tanks this script (default matrix now includes 200 and 400):

1. launches the REAL server (`python -m genesis_vehicle.server [--multi-env]
   --headless`) as a subprocess (CPU backend — the SDK default; ``--gpu`` to
   opt in),
2. acts as the UE client over OSC loopback: sends ``/Genesis/Init/Physics``
   (dt = 0.025), ``/Genesis/Vehicle/Init`` (tank URDF + SkidSteer mapping →
   ``tank_10w_skid_belt``), K ``/Init/Target``s, the 88 convex-hull
   ``/Init/Obstacle``s (complex only), ``/Init/Done``,
3. streams ``/Genesis/Vehicle/Control`` driving inputs at ~30 Hz,
4. collects the server's ``[MODE]`` / ``[PROFILE]`` / ``[STATS]`` lines,
   then sends ``/Genesis/Control "stop"``,
5. prints one summary table (per-mode speed + real-time verdict).

Run (from the workspace root, venv active):

    python -m genesis_vehicle.server.benchmark                     # full 16-config matrix
    python -m genesis_vehicle.server.benchmark --tanks 1,10 --terrain simple
    python -m genesis_vehicle.server.benchmark --modes L3 --gpu    # L3 on GPU
    python -m genesis_vehicle.server.benchmark --urdf /path/to/tank.urdf

The tank URDF defaults to ``GeneVehicle_KDU/tank_ray.urdf`` next to the SDK
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

_STATS_RE = re.compile(
    r"\[STATS\].*Loop Avg:\s*([\d.]+)\s*ms\s*\|\s*Physics Avg:\s*([\d.]+)\s*ms"
    r"\s*\((\d+(?:\.\d+)?)\s*steps/loop,\s*([\d.]+)\s*ms/step\)")
_CAP_RE = re.compile(r"\[cap=(\d+):([a-z0-9:]+)\]")


def _default_urdf() -> str:
    here = os.path.dirname(os.path.abspath(__file__))          # .../genesis_vehicle/server
    cand = os.path.abspath(os.path.join(here, "..", "..", "GeneVehicle_KDU", "tank_ray.urdf"))
    return cand


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

    def __init__(self, k: int, urdf: str, complex_terrain: bool, hull_obj: str):
        self.cli = SimpleUDPClient("127.0.0.1", RECV_PORT)
        self.k = k
        self.urdf = urdf
        self.complex_terrain = complex_terrain
        self.hull_obj = hull_obj
        self._stop = threading.Event()
        self._streamer = None

    def send_init_burst(self) -> None:
        c = self.cli
        c.send_message("/Genesis/Init/Physics", [-9.81, 0.025, 2.0])
        c.send_message("/Genesis/Vehicle/Init",
                       [self.urdf, json.dumps({"driveType": 2})])   # 2 = SkidSteer → tank preset
        for tid, (x, y, z) in enumerate(_tank_grid(self.k)):
            # [ID, Type, Px,Py,Pz, Qx,Qy,Qz,Qw, Sx,Sy,Sz, mass, friction, restitution]
            c.send_message("/Init/Target",
                           [tid, 1, x, y, z, 0.0, 0.0, 0.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 2.0, 0.0])
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
                time.sleep(1.0 / 30.0)
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
               python_exe: str, verbose: bool) -> dict:
    """Launch server + mock client for one config; return parsed results."""
    cmd = [python_exe, "-m", "genesis_vehicle.server", "--headless",
           "--road-raycast-only", "--pacing-profile",
           "--recv_port", str(RECV_PORT), "--send_port", str(SEND_PORT),
           "--send_port_obs", str(OBS_PORT)]
    if mode == "L3":
        cmd.append("--multi-env")
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

    client = MockUEClient(k, urdf, terrain == "complex", hull_obj)
    result = dict(mode=mode, terrain=terrain, k=k, ok=False)
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

        # drop the first STATS line (JIT/settle warm-up), average the rest
        rows = stats[1:] if len(stats) > 1 else stats
        n = len(rows)
        result.update(
            ok=True,
            loop_ms=sum(r[0] for r in rows) / n,
            phys_ms=sum(r[1] for r in rows) / n,
            steps_per_loop=sum(r[2] for r in rows) / n,
            ms_per_step=sum(r[3] for r in rows) / n,
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
                    help="[STATS] lines per config (50 loops each; first is dropped)")
    ap.add_argument("--build-timeout", type=float, default=900.0)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="echo the server's stdout")
    args = ap.parse_args()

    if not os.path.exists(args.urdf):
        sys.exit(f"tank URDF not found: {args.urdf}  (pass --urdf)")
    hull_obj = _make_hull_obj()
    modes = [m.strip() for m in args.modes.split(",")]
    terrains = [t.strip() for t in args.terrain.split(",")]
    tanks = [int(x) for x in args.tanks.split(",")]

    print(f"[bench] server benchmark — modes={modes} terrain={terrains} "
          f"tanks={tanks} backend={'GPU(L3)' if args.gpu else 'CPU'} dt=0.025 rco=on")
    results = []
    for mode in modes:
        for terrain in terrains:
            for k in tanks:
                print(f"[bench] >>> {mode} | {terrain} | {k} tanks ...", flush=True)
                r = run_config(mode, terrain, k, args.urdf, hull_obj, args.gpu,
                               args.stats, args.build_timeout, sys.executable,
                               args.verbose)
                results.append(r)
                if r.get("ok"):
                    print(f"[bench]     {r['ms_per_step']:.2f} ms/step | "
                          f"{r['steps_per_loop']:.1f} steps/loop | "
                          f"Loop {r['loop_ms']:.2f} ms", flush=True)
                else:
                    print(f"[bench]     FAILED: {r.get('error')}", flush=True)

    budget_ms = 25.0
    print("\n| mode | terrain | tanks | ms/step | steps/loop | Loop Avg | pacing | realtime |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        if r.get("ok"):
            rt = "O" if (r["steps_per_loop"] <= 1.05 and r["loop_ms"] <= budget_ms) else "X"
            sw = r.get("pacing_switches", 0)
            pacing = r.get("pacing_mode", "?") + (f" ({sw}sw)" if sw else "")
            print(f"| {r['mode']} | {r['terrain']} | {r['k']} | "
                  f"{r['ms_per_step']:.2f} | {r['steps_per_loop']:.1f} | "
                  f"{r['loop_ms']:.2f} | {pacing} | {rt} |")
        else:
            print(f"| {r['mode']} | {r['terrain']} | {r['k']} | FAIL: {r.get('error')} | | | | |")
    print("\n[bench] realtime O = steps/loop <= 1.05 AND Loop Avg <= 25 ms (dt budget)")
    print("[bench] pacing = final adaptive-catchup mode (Nsw = switch count); "
          "trigger contexts below as [pacing] lines")
    for r in results:
        if r.get("ok") and r.get("profile"):
            print(f"[profile] {r['mode']}/{r['terrain']}/{r['k']}: {r['profile']}")
    for r in results:
        for ev in r.get("pacing_events", []):
            print(f"[pacing]  {r['mode']}/{r['terrain']}/{r['k']}: {ev}")


if __name__ == "__main__":
    main()
