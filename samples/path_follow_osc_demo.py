"""path_follow_osc_demo.py — trajectory following THROUGH the OSC server.

The physics runs in a separate ``genesis_vehicle.server`` process (exactly
as it would for UE / Unity); THIS script plays the game-client role: it
receives vehicle state over OSC, runs :class:`PathFollower` on it, and
streams ``(steer, throttle, brake)`` back. I.e. the trajectory controller
lives on the CLIENT side of the wire — where a game would put it.

    python -m genesis_vehicle.samples.path_follow_osc_demo [-v] [--viewer]

``--viewer`` opens the SERVER's Genesis viewer (the physics process is the
one with a scene to show). Lifecycle is coupled both ways: the client
finishing sends the server a stop, and the server exiting — e.g. its
viewer window being closed — is detected by the client, which shuts down
with it (no orphan processes).

    server subprocess                        this script (mock UE client)
    /Genesis/Vehicle/TargetBulk  ─────────▶  UE frame → Genesis frame,
      (pos+quat per step, UE frame)          finite-difference velocity,
                                             PathFollower.step(state)
    /Genesis/Vehicle/Control     ◀─────────  [frame, id, S, T, B, 0, 0]

Wire-frame conversions (see ``docs/server.md``): TargetBulk positions are
UE centimetres with Y flipped — ``pos_g = (Px/100, -Py/100, Pz/100)`` —
and quaternions are UE XYZW with ``(x, z)`` negated —
``quat_g(wxyz) = (Qw, -Qx, Qy, -Qz)``. The client has no velocity channel,
so ``v_long`` comes from finite-differencing positions over a ~0.3 s
window in the server's SIM time base (/Genesis/State/SimTime) —
immune to slow motion and interpolated sends.

PASS criterion: final position within 3 m of the goal waypoint.

Caveat: the server builds its own vehicle tuning from the URDF +
``driveType`` json (NOT ``samples/tank_tuning.py``), so the bundled sweep
table is an approximate match here — fine for this forward-only path.
For production, measure a table against your actual server-built config
(``docs/path-following.md``).
"""
from __future__ import annotations

# Bootstrap: allow `python path/to/this_file.py` (and `python -m ...`) both.
import sys, pathlib
_SDK_PARENT = str(pathlib.Path(__file__).resolve().parents[2])
if _SDK_PARENT not in sys.path:
    sys.path.insert(0, _SDK_PARENT)

import argparse
import json
import math
import os
import subprocess
import threading
import time
from collections import deque

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

from genesis_vehicle import PathFollower
from genesis_vehicle.control import extract_state_from_arrays
from genesis_vehicle.samples.path_follow_demo import densify

_HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(_HERE, "urdf", "tank_ray.urdf")
CSV = os.path.join(_HERE, "data", "tank_sweep_signed.csv")

RECV_PORT = 7101      # server listens here (we send control/init to it)
SEND_PORT = 7102      # server streams state here (we listen)
OBS_PORT = 7104

TARGET_ID = 0
T_MAX_WALL = 120.0
SERVER_DT = 0.025     # sim dt sent in /Genesis/Init/Physics — one TargetBulk
                      # packet per sim step, so FD velocity divides by THIS
                      # (wall-clock gaps are uneven under catch-up pacing)


class StateReceiver:
    """Listens for /Genesis/Vehicle/TargetBulk and keeps the latest pose of
    TARGET_ID, converted from the UE wire frame back to Genesis metres."""

    def __init__(self):
        self.lock = threading.Lock()
        self.seq = 0
        self.pos = None          # (x, y, z) Genesis metres
        self.quat = None         # (w, x, y, z) Genesis
        self.t = None            # arrival wall time
        self.sim_t = None        # server SIM time of the latest pose (v1.1.20)
        self._pending_sim_t = None

        disp = Dispatcher()
        disp.map("/Genesis/Vehicle/TargetBulk", self._on_bulk)
        disp.map("/Genesis/State/SimTime", self._on_sim_time)
        self._osc = ThreadingOSCUDPServer(("127.0.0.1", SEND_PORT), disp)
        threading.Thread(target=self._osc.serve_forever, daemon=True).start()

    def _on_sim_time(self, address, *args):
        # Sent immediately BEFORE the TargetBulk it describes.
        if args:
            self._pending_sim_t = float(args[0])

    def _on_bulk(self, address, *args):
        # Per target: ID, Px,Py,Pz, Qx,Qy,Qz,Qw, numWheels, wheels(8 each)...
        # (packet ends with a -1 sentinel; we only run one vehicle).
        i = 0
        while i < len(args) and int(args[i]) != -1:
            tid = int(args[i])
            px, py, pz = args[i + 1], args[i + 2], args[i + 3]
            qx, qy, qz, qw = args[i + 4], args[i + 5], args[i + 6], args[i + 7]
            n_wheels = int(args[i + 8])
            i += 9 + 8 * n_wheels
            if tid != TARGET_ID:
                continue
            with self.lock:
                self.pos = (px / 100.0, -py / 100.0, pz / 100.0)
                self.quat = (qw, -qx, qy, -qz)          # UE xyzw -> Genesis wxyz
                self.t = time.perf_counter()
                self.sim_t = self._pending_sim_t
                self.seq += 1

    def latest(self):
        with self.lock:
            return self.seq, self.pos, self.quat, self.t, self.sim_t

    def shutdown(self):
        self._osc.shutdown()


def build_path():
    """Forward L: east 25 m, then south 18 m, +2 m/s throughout."""
    return densify([
        (0.0,   0.0, 0.0, +2.0),
        (25.0,  0.0, 0.0, +2.0),
        (25.0, -18.0, 0.0, +2.0),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="echo server stdout")
    ap.add_argument("--viewer", action="store_true",
                    help="open the SERVER's Genesis viewer (drops --headless). "
                         "Closing the viewer window ends the server process; "
                         "this client detects that and shuts down with it")
    args = ap.parse_args()

    path = build_path()
    goal_xy = (path[-1][0], path[-1][1])
    follower = PathFollower(path, CSV)
    print(f"path: {len(path)} waypoints, goal={goal_xy}")

    receiver = StateReceiver()

    # --- Server subprocess (the UE-facing physics process) -------------------
    cmd = [sys.executable, "-m", "genesis_vehicle.server",
           "--road-raycast-only", "--max-catchup-steps", "1",
           "--recv_port", str(RECV_PORT), "--send_port", str(SEND_PORT),
           "--send_port_obs", str(OBS_PORT)]
    if not args.viewer:
        cmd.append("--headless")
    env = dict(os.environ)
    env.setdefault("GENESIS_VEHICLE_QUIET", "1")
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env=env, cwd=_SDK_PARENT)
    init_done = threading.Event()

    def reader():
        for line in proc.stdout:
            if args.verbose:
                sys.stdout.write(f"    | {line}")
            if "Initialization Complete" in line:
                init_done.set()
    threading.Thread(target=reader, daemon=True).start()

    cli = SimpleUDPClient("127.0.0.1", RECV_PORT)

    def send_init_burst():
        cli.send_message("/Genesis/Init/Physics", [-9.81, 0.025, 2.0])
        # driveType 2 = skid-steer (tank preset). The other keys replicate
        # samples/tank_tuning.py on the server-built vehicle so the bundled
        # sweep table matches the plant: omega cap (top speed), brake
        # authority, wheel spin inertia + lateral friction ("susp" substring-
        # matches every wheel spin joint of tank_ray.urdf).
        cli.send_message("/Genesis/Vehicle/Init",
                         [URDF, json.dumps({
                             "driveType": 2,
                             "maxBrake": 200000.0,
                             "omegaMaxDrive": 10.4,
                             "wheelOverrides": [{"wheelName": "susp",
                                                 "inertia": 100.0,
                                                 "muLat": 0.5}],
                         })])
        # [ID, Type, P(Genesis metres), Q(x,y,z,w), S, mass, friction, restit.]
        cli.send_message("/Init/Target",
                         [TARGET_ID, 1, path[0][0], path[0][1], 1.2,
                          0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0])
        cli.send_message("/Init/Done", [])

    ok = False
    try:
        t0 = time.time()
        while not init_done.is_set():           # server polls init at 1 Hz
            if proc.poll() is not None or time.time() - t0 > 60.0:
                raise RuntimeError("server init timeout / died")
            send_init_burst()
            init_done.wait(timeout=1.0)
        print("server initialized; driving...")

        # Draw the planned path in the SERVER's viewer (debug overlay wire,
        # v1.1.20): [r, g, b, a, radius, x0, y0, z0, ...] — Genesis metres.
        # Segments colored by direction: forward cyan (this demo is
        # forward-only; a reverse block would be sent as a second orange
        # polyline).
        cli.send_message(
            "/Genesis/Debug/Polyline",
            [0.2, 0.7, 1.0, 1.0, 0.05] +
            [c for wp in path for c in (wp[0], wp[1], 0.35)])
        # Waypoint sphere markers (every 3rd) + a red goal marker.
        cli.send_message(
            "/Genesis/Debug/Spheres",
            [0.2, 0.7, 1.0, 1.0, 0.15] +
            [c for wp in path[::3] for c in (wp[0], wp[1], 0.4)])
        cli.send_message(
            "/Genesis/Debug/Spheres",
            [1.0, 0.2, 0.2, 1.0, 0.5, goal_xy[0], goal_xy[1], 0.7])

        # --- Client-side control loop: one command per received state -------
        frame = 0
        last_seq = 0
        hist = deque()                 # (t_wall, pos) ring for window FD
        vel = (0.0, 0.0, 0.0)
        last_print = 0.0
        server_gone = False
        t0 = time.time()
        while time.time() - t0 < T_MAX_WALL:
            # Shut down WITH the server: if its process ended (e.g. the
            # viewer window was closed in --viewer mode), stop driving too.
            if proc.poll() is not None:
                server_gone = True
                print("\n[server exited — shutting down with it]")
                break
            seq, pos, quat, t_pkt, sim_t = receiver.latest()
            if seq == last_seq or pos is None:
                time.sleep(0.002)
                continue
            last_seq = seq
            # Velocity by finite difference over a ~0.3 s WINDOW of packets,
            # in the server's SIMULATION time base (/Genesis/State/SimTime,
            # v1.1.20). Two reasons wall-clock FD fails here: (a) the send
            # cadence is not 1:1 with sim steps (catch-up bursts,
            # interpolated states) so per-packet deltas jitter; (b) under
            # load (e.g. viewer on) the server runs in SLOW MOTION — wall
            # deltas then under-read velocity, the follower over-throttles,
            # and the vehicle overshoots in sim terms. Falls back to wall
            # time if the server predates the SimTime stamp.
            tb = sim_t if sim_t is not None else t_pkt
            hist.append((tb, pos))
            while hist and tb - hist[0][0] > 0.3:
                hist.popleft()
            t_old, p_old = hist[0]
            span = tb - t_old
            if span > 0.1:
                vel = tuple((p - q) / span for p, q in zip(pos, p_old))

            st = extract_state_from_arrays(pos, quat, vel)
            thr, steer, brk = follower.step(
                st["pos_xy"], st["yaw"], st["v_long"], st["pitch"], st["roll"])
            if follower.last_mode == "DONE":
                print(f"\n[DONE] wall t={time.time() - t0:.1f}s")
                break
            frame += 1
            cli.send_message("/Genesis/Vehicle/Control",
                             [frame, TARGET_ID, steer, thr, brk, 0.0, 0.0])
            if time.time() - last_print > 2.0:
                last_print = time.time()
                print(f"  pos=({st['pos_xy'][0]:+6.1f},{st['pos_xy'][1]:+6.1f}) "
                      f"yaw={st['yaw']:+5.2f} v={st['v_long']:+5.2f} "
                      f"thr={thr:+5.2f} str={steer:+5.2f} idx={follower.current_idx} "
                      f"[{follower.last_mode}]", flush=True)

        _, pos, _, _, _ = receiver.latest()
        if server_gone:
            ok = False
            print("aborted by server shutdown (no PASS/FAIL verdict)")
        elif pos is None:
            ok = False
            print("no state ever received — FAIL")
        else:
            err = math.hypot(pos[0] - goal_xy[0], pos[1] - goal_xy[1])
            ok = err < 3.0
            print(f"\nFINAL pos=({pos[0]:+.2f}, {pos[1]:+.2f}) "
                  f"goal=({goal_xy[0]:+.2f}, {goal_xy[1]:+.2f})  err={err:.2f}m  "
                  f"{'PASS' if ok else 'FAIL'} (criterion: err < 3m)")
    finally:
        try:
            cli.send_message("/Genesis/Control", ["stop"])
        except Exception:
            pass
        receiver.shutdown()
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
