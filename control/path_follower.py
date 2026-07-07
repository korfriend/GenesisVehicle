"""Path -> per-step (Steer, Throttle) controller driven by a sweep table.

Give ``PathFollower`` a path (waypoints with signed target speeds) and the
:class:`~genesis_vehicle.control.sweep.SweepTable` measured for the vehicle,
and call :meth:`PathFollower.step` every simulation step with the current
chassis state; it returns ``(throttle, steer, brake)`` ready for
``vehicle.set_inputs(...)``.

The path is split into *direction blocks* — maximal runs of waypoints whose
``target_speed`` shares one sign. Within a block the controller is a plain
pursuit loop; a block boundary is a cusp (gear change): the vehicle
decelerates toward the boundary waypoint, brakes to a stop, then continues
into the next block with the flipped direction. Projection and lookahead
never cross the boundary of the active block, so approaching a cusp cannot
contaminate the current direction's geometry (the pre-SDK deliverable let
the lookahead leak across the cusp, which only behaved on collinear
forward/backward paths).

Per-step pipeline (within the active block):

    1. project the vehicle position onto the block's segments
    2. find the waypoint ``lookahead`` metres ahead (capped at block end)
    3. v_target from that waypoint; if the block end is inside the
       lookahead window and a cusp follows, taper |v_target| with the
       remaining distance (decelerate into the cusp)
    4. heading_err = 0.7 * yaw_err + 0.3 * position_err
    5. P control: a_target = k_v * (v_target - v_long),
                  omega_target = k_w * heading_err
    6. sweep inverse: throttle = table^-1(v, a_target, pitch, roll),
                      steer    = table^-1(v, omega_target, pitch, roll)
    7. low-speed KICK (saturate throttle when far below target speed) and
       brake-at-rest handling

The controller is simulator-agnostic (numpy/math only): feed it state from
Genesis via :func:`extract_state`, or from any engine via
:func:`extract_state_from_arrays`.

Tuned on skid-steer (tank) vehicles; Ackermann vehicles generally work,
but re-measure the sweep so the steer sign convention of the actual
drivetrain is baked into the table (see ``docs/path-following.md``).
"""
from __future__ import annotations

import math
import os

from .sweep import SweepTable


# --- State extraction --------------------------------------------------------

def extract_state_from_arrays(pos_xyz, quat_wxyz, vel_xyz) -> dict:
    """Simulator-agnostic chassis-state extraction.

    Args:
        pos_xyz: world position (3,).
        quat_wxyz: world orientation quaternion, scalar-first (w, x, y, z).
        vel_xyz: world linear velocity (3,).

    Returns:
        dict with ``pos_xy`` (m), ``yaw`` (rad), ``v_long`` (m/s, signed along
        chassis +X), ``pitch``/``roll`` (DEGREES — the sweep grid unit).
    """
    w = float(quat_wxyz[0]); x = float(quat_wxyz[1])
    y = float(quat_wxyz[2]); z = float(quat_wxyz[3])
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    cy, sy = math.cos(yaw), math.sin(yaw)
    v_long = float(vel_xyz[0]) * cy + float(vel_xyz[1]) * sy
    return {
        "pos_xy": (float(pos_xyz[0]), float(pos_xyz[1])),
        "yaw": yaw, "v_long": v_long,
        "pitch": math.degrees(pitch), "roll": math.degrees(roll),
    }


def extract_state(vehicle, env_idx: int = 0) -> dict:
    """State extraction for a Genesis entity or a ``VehicleScene`` Vehicle.

    Works with anything exposing ``get_pos()/get_quat()/get_vel()`` (both a
    raw Genesis entity and the SDK ``Vehicle`` wrapper do). For batched
    scenes (``n_envs > 1``) pass ``env_idx``.
    """
    def _row(t):
        arr = t.cpu().numpy() if hasattr(t, "cpu") else t
        return arr[env_idx] if getattr(arr, "ndim", 1) == 2 else arr
    return extract_state_from_arrays(
        _row(vehicle.get_pos()), _row(vehicle.get_quat()), _row(vehicle.get_vel()))


# --- Follower -----------------------------------------------------------------

class PathFollower:
    """Pursuit-style follower over a (x, y, z, target_speed) path.

    Args:
        path: list of ``(x, y, z, target_speed)``. ``z`` is ignored (the
            vehicle drives on whatever ground the sim provides).
            ``target_speed`` is m/s — positive = forward, negative =
            backward; a sign flip along the path is a cusp (gear change)
            and triggers an automatic decelerate-stop-reverse.
            Adjacent waypoints should be ~0.3–1 m apart (densify long
            straight segments) for accurate projection/lookahead.
        sweep: a :class:`SweepTable`, or a path to its CSV.
        lookahead: pursuit distance along the path (m).
        arrival_goal: distance to the final waypoint that counts as DONE (m).
        cusp_goal: distance to a cusp waypoint that starts the stop-and-
            reverse transition (m).
        k_v / k_w: P gains for speed / heading.
        k_approach: cusp deceleration taper (1/s): approaching a cusp,
            |v_target| <= k_approach * remaining_distance + 0.3.
        v_stop: |v| below this counts as stopped for the cusp transition (m/s).
        steer_cap: |steer| command cap (keeps skid-steer from saturating).

    ``last_mode`` after each :meth:`step` is one of ``INIT`` / ``DRV+1`` /
    ``DRV-1`` / ``STOP`` / ``BRAKE_TRANS`` (stopping for a cusp) / ``DONE``.
    """

    def __init__(self, path, sweep, *,
                 lookahead: float = 3.5, arrival_goal: float = 1.5,
                 cusp_goal: float = 1.0, k_v: float = 2.0, k_w: float = 1.5,
                 k_approach: float = 1.0, v_stop: float = 0.05,
                 steer_cap: float = 0.5):
        if len(path) < 2:
            raise ValueError("path needs at least 2 waypoints")
        if isinstance(sweep, (str, os.PathLike)):
            sweep = SweepTable.load(sweep)
        self.sweep = sweep
        self.path = [tuple(p) for p in path]
        self.n = len(self.path)
        self.lookahead = lookahead
        self.arrival_goal = arrival_goal
        self.cusp_goal = cusp_goal
        self.k_v = k_v
        self.k_w = k_w
        self.k_approach = k_approach
        self.v_stop = v_stop
        self.steer_cap = steer_cap

        # Direction blocks: maximal index runs [start, end] with one speed
        # sign. Zero-speed waypoints inherit the running sign (an explicit
        # stop point does not split a block).
        self.blocks = []          # list of (start_idx, end_idx, sign)
        sign = 0
        start = 0
        for i in range(self.n):
            s = self.path[i][3]
            s_sign = (s > 1e-6) - (s < -1e-6)
            if s_sign == 0:
                continue
            if sign == 0:
                sign = s_sign
            elif s_sign != sign:
                self.blocks.append((start, i, sign))   # cusp AT waypoint i
                start, sign = i, s_sign
        self.blocks.append((start, self.n - 1, sign if sign != 0 else 1))

        # Per-waypoint chassis yaw, computed PER BLOCK: toward the next
        # waypoint, with a +pi flip in backward blocks (the chassis faces
        # away from the travel direction). A block-end waypoint is shared
        # with the next block, so each block also records its own
        # continue-straight end yaw (`_block_end_yaw`) — using the global
        # per-waypoint value there would leak the NEXT block's (flipped)
        # direction into the current block's pursuit. Zero-length segments
        # (duplicated cusp waypoints) carry the previous yaw.
        # (The pre-SDK deliverable's backward flip `(yaw + pi) % (2*pi) - pi`
        # was a plain normalization — a no-op — so backward chassis yaw was
        # off by pi; masked on collinear reversing where the position-error
        # term vanishes.)
        self.yaws = [0.0] * self.n
        self._block_end_yaw = []
        for (bs, be, sgn) in self.blocks:
            prev = None
            for i in range(bs, be):
                dx = self.path[i + 1][0] - self.path[i][0]
                dy = self.path[i + 1][1] - self.path[i][1]
                if dx * dx + dy * dy < 1e-12:
                    yaw = prev if prev is not None else 0.0
                else:
                    yaw = math.atan2(dy, dx)
                    if sgn < 0:
                        yaw = (yaw % (2 * math.pi)) - math.pi   # +pi flip
                self.yaws[i] = yaw
                prev = yaw
            self._block_end_yaw.append(prev if prev is not None else 0.0)
        self.yaws[self.n - 1] = self._block_end_yaw[-1]

        self.block_i = 0
        blk_start = self.blocks[0][0]
        self.current_idx = blk_start
        self.projection_t = 0.0
        self.projection_xy = (self.path[blk_start][0], self.path[blk_start][1])
        self.transitioning = False
        self.last_mode = "INIT"

    # -- Path geometry ---------------------------------------------------------

    def _advance(self, pos):
        """Project the position onto the nearest segment OF THE ACTIVE BLOCK
        and advance ``current_idx``/``projection_t``.

        Only segments within an arc-length window (lookahead + 2 m) ahead
        are searched, and the index may advance at most 5 segments per call
        — both guard against snapping to a far-away later segment when the
        path self-approaches.
        """
        blk_end = self.blocks[self.block_i][1]
        if self.current_idx >= blk_end:
            return
        search_arc = self.lookahead + 2.0
        accum = 0.0
        end = self.current_idx + 1
        while end <= blk_end and accum < search_arc:
            dx = self.path[end][0] - self.path[end - 1][0]
            dy = self.path[end][1] - self.path[end - 1][1]
            accum += math.hypot(dx, dy)
            end += 1

        best_idx = self.current_idx
        best_d = float("inf")
        best_t = self.projection_t
        for i in range(self.current_idx, min(end, blk_end)):
            p0 = self.path[i]; p1 = self.path[i + 1]
            sx = p1[0] - p0[0]; sy = p1[1] - p0[1]
            seg_len_sq = sx * sx + sy * sy
            if seg_len_sq < 1e-9:
                continue
            tx = pos[0] - p0[0]; ty = pos[1] - p0[1]
            t_c = max(0.0, min(1.0, (sx * tx + sy * ty) / seg_len_sq))
            cx = p0[0] + t_c * sx; cy = p0[1] + t_c * sy
            d = (pos[0] - cx) ** 2 + (pos[1] - cy) ** 2
            if d < best_d:
                best_d = d; best_idx = i; best_t = t_c

        MAX_STEP = 5
        target_idx = min(best_idx, self.current_idx + MAX_STEP)
        if target_idx > self.current_idx:
            self.current_idx = target_idx
            if self.current_idx < blk_end:
                p0 = self.path[self.current_idx]
                p1 = self.path[self.current_idx + 1]
                sx = p1[0] - p0[0]; sy = p1[1] - p0[1]
                seg_len_sq = sx * sx + sy * sy
                if seg_len_sq > 1e-9:
                    tx = pos[0] - p0[0]; ty = pos[1] - p0[1]
                    self.projection_t = max(0.0, min(1.0,
                        (sx * tx + sy * ty) / seg_len_sq))
                else:
                    self.projection_t = 0.0
            else:
                self.projection_t = 0.0
        else:
            self.projection_t = best_t

        if self.current_idx < blk_end:
            p0 = self.path[self.current_idx]
            p1 = self.path[self.current_idx + 1]
            self.projection_xy = (
                p0[0] + self.projection_t * (p1[0] - p0[0]),
                p0[1] + self.projection_t * (p1[1] - p0[1]),
            )
        else:
            p = self.path[blk_end]
            self.projection_xy = (p[0], p[1])

    def _lookahead(self):
        """Waypoint ``lookahead`` metres ahead of the projection point along
        the path, CAPPED at the active block's end. Returns
        ``(index, capped)`` — ``capped`` is True when the block end lies
        inside the lookahead window (i.e. a cusp or the goal is near)."""
        blk_end = self.blocks[self.block_i][1]
        if self.current_idx >= blk_end:
            return blk_end, True
        idx = self.current_idx
        p_next = self.path[idx + 1]
        accum = math.hypot(p_next[0] - self.projection_xy[0],
                           p_next[1] - self.projection_xy[1])
        idx += 1
        while accum < self.lookahead and idx < blk_end:
            p0 = self.path[idx]; p1 = self.path[idx + 1]
            accum += math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            idx += 1
        return idx, idx >= blk_end and accum < self.lookahead

    # -- Control ----------------------------------------------------------------

    def step(self, pos_xy, yaw, v_long, pitch_deg, roll_deg):
        """One control step. Returns ``(throttle, steer, brake)``, each in
        the SDK input range (throttle/steer in [-1, 1], brake in [0, 1])."""
        last_block = self.block_i == len(self.blocks) - 1
        blk_start, blk_end, dir_sign = self.blocks[self.block_i]

        # DONE: near the final waypoint while driving the final block.
        last = self.path[-1]
        if last_block and math.hypot(pos_xy[0] - last[0],
                                     pos_xy[1] - last[1]) < self.arrival_goal:
            self.last_mode = "DONE"
            return 0.0, 0.0, 1.0

        # Cusp state machine: brake to a stop at the block boundary, then
        # switch to the next block (flipped direction).
        if self.transitioning:
            if abs(v_long) > self.v_stop:
                self.last_mode = "BRAKE_TRANS"
                return 0.0, 0.0, 1.0
            self.transitioning = False
            self.block_i += 1
            blk_start, blk_end, dir_sign = self.blocks[self.block_i]
            last_block = self.block_i == len(self.blocks) - 1
            self.current_idx = blk_start
            self.projection_t = 0.0
            self.projection_xy = (self.path[blk_start][0],
                                  self.path[blk_start][1])
        elif not last_block:
            cusp = self.path[blk_end]
            # Reached when the projection sits at the block's end point
            # (robust to duplicated/zero-length boundary waypoints) or the
            # vehicle itself is within cusp_goal of the cusp.
            reached = (self.current_idx >= blk_end
                       or math.hypot(self.projection_xy[0] - cusp[0],
                                     self.projection_xy[1] - cusp[1]) < 0.25)
            if reached or math.hypot(pos_xy[0] - cusp[0],
                                     pos_xy[1] - cusp[1]) < self.cusp_goal:
                self.transitioning = True
                self.last_mode = "BRAKE_TRANS"
                return 0.0, 0.0, 1.0

        self._advance(pos_xy)

        la_idx, capped = self._lookahead()
        la = self.path[la_idx]
        v_target = la[3]
        if abs(v_target) < 1e-6:
            v_target = dir_sign * 1e-6     # explicit stop wp: keep direction
        if capped and not last_block:
            # Cusp inside the lookahead window: taper the speed with the
            # remaining distance so the vehicle arrives slow enough to stop.
            d_cusp = math.hypot(pos_xy[0] - self.path[blk_end][0],
                                pos_xy[1] - self.path[blk_end][1])
            v_mag = min(abs(v_target), self.k_approach * d_cusp + 0.3)
            v_target = math.copysign(v_mag, dir_sign)

        # Heading error: blend of path yaw and bearing-to-lookahead.
        dx = la[0] - pos_xy[0]; dy = la[1] - pos_xy[1]
        bearing = math.atan2(dy, dx)
        # The block-end waypoint is shared with the NEXT block; use this
        # block's continue-straight yaw there instead of the (next-block,
        # possibly flipped) per-waypoint value.
        if la_idx >= blk_end and not last_block:
            path_yaw = self._block_end_yaw[self.block_i]
        else:
            path_yaw = self.yaws[la_idx]
        yaw_err = (path_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        if dir_sign >= 0:
            pos_err = (bearing - yaw + math.pi) % (2 * math.pi) - math.pi
        else:
            pos_err = (bearing + math.pi - yaw + math.pi) % (2 * math.pi) - math.pi
        heading_err = 0.7 * yaw_err + 0.3 * pos_err

        a_target = self.k_v * (v_target - v_long)
        omega_target = self.k_w * heading_err

        throttle = self.sweep.throttle_for(v_long, a_target, pitch_deg, roll_deg)
        steer = self.sweep.steer_for(v_long, omega_target, pitch_deg, roll_deg)

        # Low-speed KICK: far below target speed -> saturate throttle so the
        # vehicle actually gets moving (the P term alone can be too shy).
        # Signed comparison: v_long * sign(v_target), so moving the WRONG
        # way also kicks toward the target direction.
        if abs(v_target) > 0.1:
            v_along = v_long * math.copysign(1.0, v_target)
            if v_along < 0.4 * abs(v_target):
                throttle = math.copysign(1.0, v_target)

        throttle = max(-1.0, min(1.0, throttle))
        steer = max(-self.steer_cap, min(self.steer_cap, steer))
        brake = 1.0 if abs(throttle) < 0.02 and abs(v_long) < 0.1 else 0.0

        self.last_mode = f"DRV{dir_sign:+d}" if dir_sign != 0 else "STOP"
        return throttle, steer, brake
