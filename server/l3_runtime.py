"""L3 multi-env runtime (genesis_vehicle.server) — implements ``--multi-env`` mode.

Premise: all targets are vehicles of the **same URDF** and there is **no
vehicle-to-vehicle interaction**. Instead of "N entities in one scene"
(per-entity, linear cost), N vehicles are driven as **one vehicle +
``scene.build(n_envs=N)``** parallel environments. All physics kernels process
the N environments simultaneously, so step cost is nearly independent of
vehicle count (measured: 30/50/100 vehicles all ~19 ms/step, 57x vs
per-entity GPU 30 vehicles at 1,115 ms).

Mapping: target_id (sorted order) ↔ env index. The environment
(floor/obstacles) is added once; Genesis auto-replicates it per env.

Intentional differences / limitations vs the L2 (per-entity) mode:
  - Backend defaults to CPU (same as L2 mode — at n_envs≲200 the GPU is
    kernel-launch bound, so CPU is faster; measured 30 tanks CPU 8.4 vs GPU
    ~19 ms/step). Opt in with ``--gpu`` beyond the measured crossover at
    ~250–300 envs (v1.1.6: L3×400 GPU 27.8 vs CPU 46.2 ms/step).
  - Dynamic obstacles get N env copies, so interactions like "vehicle B sees
    the obstacle vehicle A pushed" cannot be expressed. State is sent from env 0.
  - ``target_forces`` (persistent external forces) unsupported — warned once
    and ignored on receipt.
  - lockstep unsupported (OSC pacing mode fixed).
"""

import os
import re
import sys
import time
import argparse

import numpy as np
import torch
import genesis as gs

from genesis_vehicle import VehicleScene

from .osc_manager import OSCManager
from .pacing import AdaptiveCatchup
from . import env_builder
from . import vehicle_builder


def _to_host_batched(tensors, force_batch=False):
    """Concat a list of torch tensors to 1-D on-device, download with a
    **single** ``.cpu()`` call, and return a list of numpy arrays with the
    original shapes (v1.1.1).

    On the GPU backend, capture paid a blocking CUDA stream flush + DtoH
    round-trip (~0.3-0.5 ms/call, WSL2) at every read site — the cost is the
    number of calls, not the data volume (a few KB). Here the synchronization
    is collapsed to one per capture (measured: mitigates the gap of the same
    capture code, 0.72 ms CPU vs 3.19 ms GPU @n_envs=10).
    On the CPU backend the concat copy is actually a loss, so per-tensor
    conversion is kept (``force_batch`` is a test-only override switch). If
    dtypes are mixed, safely fall back to per-tensor conversion."""
    if not tensors:
        return []
    ts = list(tensors)
    all_torch = all(torch.is_tensor(t) for t in ts)
    use_batch = (force_batch or (all_torch and ts[0].is_cuda))
    if not use_batch or not all_torch or len({t.dtype for t in ts}) != 1:
        return [t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
                for t in ts]
    shapes = [tuple(t.shape) for t in ts]
    host = torch.cat([t.reshape(-1) for t in ts]).detach().cpu().numpy()  # 1 sync
    out, o = [], 0
    for sh in shapes:
        n = 1
        for d in sh:
            n *= d
        out.append(host[o:o + n].reshape(sh))
        o += n
    return out


def _parse_input_triplet(raw):
    """Same input interpretation as legacy: len==4 → [_, steer, throttle, brake],
    len>=3 → [steer, throttle, brake]."""
    raw = list(raw)
    if len(raw) == 4:
        return float(raw[1]), float(raw[2]), float(raw[3])
    if len(raw) >= 3:
        return float(raw[0]), float(raw[1]), float(raw[2])
    return 0.0, 0.0, 0.0


def _q_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float32)


def _q_rot(v, q):
    qw, qv = q[0], q[1:4]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


class L3State:
    """Handles batch tensor ↔ per-tid state dict conversion + spin angle accumulation."""

    def __init__(self, car, veh, tids, n_envs, sim_dt):
        self.car = car
        self.veh = veh                          # Vehicle handle (solver-agnostic)
        self.tids = list(tids)                  # target ids in env-index order
        self.n_envs = n_envs
        self.sim_dt = sim_dt
        self.n_wheels = len(veh.resolved.wheels)

    def reset_angles(self):
        pass   # spin is owned by the SDK (wheel_visual_transforms); nothing to reset here

    def capture(self, dynamic_obstacles, ue_driven_obstacle_ids, update_angles):
        """Build the state dict via batched reads.

        Reads all N vehicles with 2 body pos/quat reads + 1 SDK closed-form
        wheel_visual_transforms call. Wheel pos/quat is the visual pose with
        steer+suspension+spin all applied (regardless of WheelJointInternalSync
        on/off). Spin is included in the quat, so w_angle=0."""
        # [GPU-mode serving architecture v1.1.3] Only pure physics on GPU,
        # capture math on CPU: the 5 raw reads (pos/quat/steer/dist/spin) +
        # obstacle poses are downloaded in **one DtoH** via _to_host_batched,
        # and the closed-form wheel pose computation runs **on CPU** via
        # wheel_visual_transforms_host — removes the dozens of small kernel
        # launches GPU-backend capture used to trigger. On the CPU backend the
        # same math was already on CPU, so behavior/cost is identical.
        obs_items = [(o_id, ent) for o_id, ent in dynamic_obstacles.items()
                     if o_id not in ue_driven_obstacle_ids]
        mvp = getattr(getattr(self.veh, "_scene", None), "_mvp", None)
        kind = mvp.kinds[0] if (mvp is not None and len(mvp.kinds) == 1) else None
        reads = kind.wheel_visual_reads() if kind is not None else None

        if reads is not None:
            tensors = list(reads)
            for _, ent in obs_items:
                tensors.append(ent.get_pos())
                tensors.append(ent.get_quat())
            hosts = _to_host_batched(tensors)            # ONE DtoH sync (GPU)
            pos_h = torch.from_numpy(np.ascontiguousarray(hosts[0]))
            quat_h = torch.from_numpy(np.ascontiguousarray(hosts[1]))
            steer_h = torch.from_numpy(np.ascontiguousarray(hosts[2]))
            dist_h = torch.from_numpy(np.ascontiguousarray(hosts[3]))
            spin_h = torch.from_numpy(np.ascontiguousarray(hosts[4]))
            wp_t, wq_t = kind.wheel_visual_transforms_host(
                pos_h, quat_h, steer_h, dist_h, spin_h, "world")
            wp = wp_t[:, 0].numpy()                      # L3: K=1 → (N, n, 3)
            wq = wq_t[:, 0].numpy()
            bp = np.atleast_2d(hosts[0])                 # (N, 3)
            bq = np.atleast_2d(hosts[1])
            obs_hosts = hosts[5:]
        else:
            # Fallback: before the first step (rest pose) / multiple kinds / exceptions — original path.
            bp = self.car.get_pos()
            bq = self.car.get_quat()
            wp, wq = self.veh.wheel_visual_transforms("world")   # (N, n, 3/4)
            fb = [bp, bq, wp, wq]
            for _, ent in obs_items:
                fb.append(ent.get_pos())
                fb.append(ent.get_quat())
            hosts = _to_host_batched(fb)
            bp, bq, wp, wq = hosts[0], hosts[1], hosts[2], hosts[3]
            bp = np.atleast_2d(bp); bq = np.atleast_2d(bq)
            obs_hosts = hosts[4:]

        state = {'targets': {}, 'dynamic_obstacles': {}}
        # NB: bp/bq/wp/wq are freshly created THIS capture (host download)
        # and never mutated — per-row .copy() removed in v1.0.13.
        for k, tid in enumerate(self.tids):
            wheels_states = [
                (wp[k, j], wq[k, j], 0.0)
                for j in range(self.n_wheels)
            ]
            state['targets'][tid] = (bp[k], bq[k], wheels_states)

        # Dynamic obstacles: sent from env 0 due to the env-replication limitation
        for i, (o_id, _ent) in enumerate(obs_items):
            p = np.atleast_2d(obs_hosts[2 * i])[0]
            q = np.atleast_2d(obs_hosts[2 * i + 1])[0]
            state['dynamic_obstacles'][o_id] = (p, q)
        return state


def run_l3(args):
    # 1. Backend: default CPU — at this scale (n_envs≲100) the GPU is
    #    kernel-launch bound, so CPU is faster (measured: 30 tanks CPU 8.4 vs
    #    GPU ~19 ms/step). Opt in with --gpu only for hundreds-of-envs batching.
    use_gpu = bool(getattr(args, 'gpu', False))
    if use_gpu and not torch.cuda.is_available():
        print(" [Genesis] [L3] [WARN] --gpu requested but CUDA is unavailable — falling back to CPU.")
        use_gpu = False
    use_cpu = not use_gpu
    VehicleScene.init_backend("cpu" if use_cpu else "gpu")
    print(f" [Genesis] [L3] Multi-env batched mode | backend = {'CPU' if use_cpu else 'GPU'}")

    # [Engine Hack] RigidGeom.n_cells monkey-patch (same as legacy)
    import genesis.engine.entities.rigid_entity.rigid_geom as rigid_geom
    rigid_geom.RigidGeom.n_cells = property(lambda self: 1)

    # 2. OSC handshake (same protocol as legacy)
    osc = OSCManager(
        send_ip=args.send_ip,
        recv_port=args.recv_port,
        send_port_cpp=args.send_port,
        send_port_obs=args.send_port_obs,
        subject_name='Genesis',
        verbose=args.verbose
    )
    print("\n" + "="*50)
    print(" Genesis Vehicle Physics Server — L3 Multi-Env Mode")
    print(" 클라이언트의 초기화 데이터를 대기합니다...")
    print("="*50 + "\n")

    init_data = osc.wait_for_initialization(timeout=300.0)
    init_physics = init_data.get('physics', {})
    ue_gravity = init_physics.get('gravity', -9.81)
    ue_dt = args.override_dt if args.override_dt is not None else init_physics.get('dt', 0.025)
    ue_friction = init_physics.get('friction', 2.0)
    ue_restitution = 0.0

    if not (osc.urdf_init_request and osc.urdf_init_request.get('path')):
        print(" [Genesis] [L3] [ERROR] multi-env 모드는 URDF 차량 전용입니다. URDF 설정이 수신되지 않아 종료합니다.")
        osc.close()
        return
    urdf_path = osc.urdf_init_request['path']
    mapping = osc.urdf_init_request.get('mapping', {})

    target_dict = init_data.get('target', {})
    if not isinstance(target_dict, dict) or len(target_dict) == 0:
        target_dict = {0: {'pos': [0, 0, 2], 'quat': [1, 0, 0, 0]}}
    tids = sorted(target_dict.keys())
    n_envs = len(tids)
    tid_to_env = {tid: k for k, tid in enumerate(tids)}
    print(f" [Genesis] [L3] {n_envs} targets → scene.build(n_envs={n_envs}) | URDF: {urdf_path}")

    # 3. Scene setup — add the environment once (auto-replicated per env)
    _rigid_kwargs = dict(
        enable_self_collision=False,
        enable_adjacent_collision=False,
        enable_neutral_collision=False,
        enable_collision=True,
        prefer_parallel_linesearch=False,
        tolerance=0.001,
        use_gjk_collision=False,
        box_box_detection=True,
        use_hibernation=True,
        broadphase_traversal=gs.broadphase_traversal.SAP,
        max_collision_pairs=2048,
    )
    # Some options may be absent depending on the genesis build (e.g. the
    # 1.2.0 PyPI build lacks prefer_parallel_linesearch) — drop unrecognized
    # keys and retry (v1.0.20: previously the server hard-crashed here).
    while True:
        try:
            _rigid_opts = gs.options.RigidOptions(**_rigid_kwargs)
            break
        except Exception as e:
            m = re.search(r"Unrecognized attribute '([^']+)'", str(e))
            if m and m.group(1) in _rigid_kwargs:
                print(f" [Genesis] [L3] [WARN] RigidOptions '{m.group(1)}' 은 이 "
                      f"genesis 빌드에 없음 — 제외하고 재시도.")
                _rigid_kwargs.pop(m.group(1))
                continue
            raise
    # Unified dual-scene wheel-raycast (VehicleScene raycast_mode=dual_scene): the road is RIGID in the
    # main scene (collision / rollover) and a KINEMATIC mirror is raycast in a
    # SEPARATE scene whose BVH is static and shared across envs — same trick as
    # the high-level API (see docs/dual-scene-raycast.md). Supersedes the old
    # single-scene --road-raycast-only (kinematic road, no collision). Genesis is
    # already initialized above via VehicleScene.init_backend, so init_genesis=False.
    vs = VehicleScene(
        n_envs=n_envs, dt=ue_dt,
        raycast_mode="dual_scene", gravity=(0, 0, ue_gravity), substeps=2,
        rigid_options=_rigid_opts, show_viewer=not args.headless,
        init_genesis=False,
    )

    plane = None
    if not args.no_floor:
        # VehicleScene routes the floor: rigid in main + kinematic raycast mirror
        # (dual_scene) — no manual main/raycast handling here.
        plane = vs.add_static(
            morph=gs.morphs.Plane(),
            material=gs.materials.Rigid(friction=ue_friction, coup_restitution=ue_restitution),
            name="ground")

    obstacles, dynamic_obstacles, initial_dynamic_states, ue_driven_obstacle_ids, extra_mass_entities = \
        env_builder.build_obstacles(
            vs=vs, init_data=init_data,
            ue_friction=ue_friction, ue_restitution=ue_restitution,
            vis_mode=args.vis_mode, verbose=args.verbose,
            # v1.0.7: forward --road-raycast-only (previously SILENTLY IGNORED in
            # multi-env — argparse accepted it but it never reached build_obstacles,
            # so "no difference" was literal). In dual_scene the wheels already ride
            # the exact kinematic mirror; rco additionally drops the main-scene
            # CoACD road collider → no chassis-vs-road narrow-phase/SDF per env.
            road_raycast_only=getattr(args, "road_raycast_only", False),
            structures_as_primitive=getattr(args, "structures_as_primitive", False),
        )

    # Vehicle: pass only cfg + morph and VehicleScene creates the main entity +
    # raycast proxy/sensor, then constructs VehiclePhysics (sensor=None,
    # distance injection) at build.
    first_info = target_dict[tids[0]]
    t_fric = first_info.get('friction', ue_friction)
    temp_urdf = vehicle_builder.strip_wheel_collisions(urdf_path)
    cfg = vehicle_builder.build_cfg(urdf_path, mapping, t_fric, target_id="L3-shared")
    # WheelJointInternalSync is auto-managed by VehicleScene.build() (on iff the main
    # scene is rendered — show_viewer); headless reads wheel poses closed-form via
    # wheel_visual_transforms, so no per-step engine FK is paid.
    veh = vs.add_vehicle(
        urdf_path, cfg=cfg,
        morph=gs.morphs.URDF(file=temp_urdf, pos=first_info.get('pos', [0, 0, 2]),
                             quat=first_info.get('quat', [1, 0, 0, 0]), fixed=False, align=False),
        material=gs.materials.Rigid(friction=t_fric, coup_restitution=0.0, sdf_cell_size=10000.0),
        surface=gs.surfaces.Rough(color=(1.0, 0.3, 0.3, 0.5)), vis_mode=args.vis_mode,
        name="L3-shared")
    car = veh.entity_main

    # 4. Batched build — VehicleScene builds the main + raycast scenes together and creates VehiclePhysics
    vs.build()
    print(f" [DEBUG] Total rigid geoms after build: {vs.rigid_solver.n_geoms}")
    print(f" [DEBUG] Total rigid links after build: {vs.rigid_solver.n_links}")

    vehicle_builder.print_resolved_table("L3-shared", veh.resolved)

    # Per-env initial poses
    init_pos = np.array([target_dict[tid].get('pos', [0, 0, 2]) for tid in tids], dtype=np.float32)
    init_quat = np.array([target_dict[tid].get('quat', [1, 0, 0, 0]) for tid in tids], dtype=np.float32)
    car.set_pos(init_pos)
    car.set_quat(init_quat)
    try:
        car.set_dofs_velocity(np.zeros((n_envs, car.n_dofs), dtype=np.float32))
    except Exception:
        pass

    st = L3State(car, veh, tids, n_envs, ue_dt)

    # 5. Hardware profiling + pacing (same procedure as legacy)
    print("\n" + "="*50)
    print(" [INFO] [GENESIS] [L3] 하드웨어 연산 성능 실측 프로파일링 중...")
    print("="*50)
    # [PROFILE] 2 warmup steps before measuring: taichi/torch kernel JIT
    # compile cost piles onto the first step and grossly inflates per-section
    # numbers (measured: GPU SDK compute 100ms (PROFILE) vs 23ms (steady)).
    # Measure only 5 steps after warmup.
    for _ in range(2):
        veh.set_inputs(throttle=0.0, brake=0.0, steer=0.0)
        vs.step()
        if not use_cpu:
            torch.cuda.synchronize()

    # [PROFILE] Measure per-section time inside the step during the 5 warmup
    # steps and print once — a diagnostic log to pinpoint the cause of
    # slowness (raycast/proxy vs SDK compute vs genesis solver) on the spot.
    # GPU launches are async, so synchronize at each section boundary to
    # attribute actual time (cost limited to the 5 warmup steps).
    _prof = {'ray': 0.0, 'sdk': 0.0, 'solver': 0.0}
    def _psync():
        if not use_cpu:
            torch.cuda.synchronize()
    _o_md = vs._measure_distances
    def _p_md():
        _psync(); t0 = time.perf_counter(); r = _o_md()
        _psync(); _prof['ray'] += time.perf_counter() - t0; return r
    vs._measure_distances = _p_md
    _o_ph = vs.physics.step if vs.physics is not None else None
    if _o_ph is not None:
        def _p_ph(*a, **kw):
            _psync(); t0 = time.perf_counter(); r = _o_ph(*a, **kw)
            _psync(); _prof['sdk'] += time.perf_counter() - t0; return r
        vs.physics.step = _p_ph
    _o_sc = vs._main_scene.step
    def _p_sc(*a, **kw):
        _psync(); t0 = time.perf_counter(); r = _o_sc(*a, **kw)
        _psync(); _prof['solver'] += time.perf_counter() - t0; return r
    vs._main_scene.step = _p_sc

    warmup_starts = time.perf_counter()
    for _ in range(5):
        veh.set_inputs(throttle=0.0, brake=0.0, steer=0.0)
        vs.step()                       # = sync proxy/mirrors + raycast + physics + main.step
        if not use_cpu:
            torch.cuda.synchronize()
    avg_step_time = (time.perf_counter() - warmup_starts) / 5.0

    # Remove instrumentation (main loop restored to unmeasured originals)
    vs._measure_distances = _o_md
    if _o_ph is not None:
        vs.physics.step = _o_ph
    vs._main_scene.step = _o_sc

    _rest = avg_step_time * 1e3 - (_prof['ray'] + _prof['sdk'] + _prof['solver']) / 5.0 * 1e3
    print(f"  - 실측된 1스텝 평균 연산 속도: {avg_step_time * 1000.0:.2f} ms  (차량 {n_envs}대 분 배치)")
    print(f"  - [PROFILE] 스텝 구간별: raycast/proxy {_prof['ray']/5*1e3:.2f} ms | "
          f"SDK compute {_prof['sdk']/5*1e3:.2f} ms | "
          f"genesis solver {_prof['solver']/5*1e3:.2f} ms | 기타 {_rest:.2f} ms")

    sim_dt = ue_dt
    print(f"  [OK] [Determinism] 물리 해상도(sim_dt) {sim_dt * 1000.0:.1f}ms ({1.0/sim_dt:.1f}Hz) 고정.")
    print("="*50 + "\n")
    print(f" [Pacing] [Auto-Pacing] 클라이언트로 동조 주기({sim_dt * 1000.0:.1f}ms)를 전송합니다...")
    osc.client_cpp.send_message("/Genesis/Init/Pacing", [float(sim_dt)])

    SIM_DT = sim_dt
    # catch-up cap — adaptive since v1.0.20 (same as physics_server; see pacing.py).
    pacer = AdaptiveCatchup(max_cap=max(5, int(0.1 / sim_dt)), sim_dt=SIM_DT,
                            fixed=getattr(args, "max_catchup_steps", None),
                            profile=bool(getattr(args, "pacing_profile", False)))
    if getattr(args, "max_catchup_steps", None) is not None:
        print(f" [Pacing] [Catch-up] 고정 cap={pacer.cap()} (--max-catchup-steps; 적응 전환 꺼짐)")
    else:
        print(f" [Pacing] [Catch-up] 적응형 cap: burst={pacer.max_cap} ↔ smooth=1 (steps/loop 모니터링)")
    accumulator = 0.0
    last_time = time.perf_counter()
    last_slow_motion_warn_time = 0.0

    from .physics_server import lerp_state   # reuse interpolation logic (same state format)

    prev_state = st.capture(dynamic_obstacles, ue_driven_obstacle_ids, update_angles=False)
    curr_state = prev_state

    step_count = 0
    last_urdf_inputs = {}
    last_frame_id = 0
    last_printed_inputs = {}
    warned_target_forces = False

    log_loop_dur_sum = 0.0
    log_phys_dur_sum = 0.0
    log_step_sum = 0
    log_count = 0

    # Input buffers (reused without reallocation)
    steer_arr = np.zeros(n_envs, dtype=np.float32)
    throttle_arr = np.zeros(n_envs, dtype=np.float32)
    brake_arr = np.zeros(n_envs, dtype=np.float32)

    # =========================================================================
    # Main simulation loop (same accumulator pacing as legacy)
    # =========================================================================
    while True:
        loop_start = time.perf_counter()

        now = time.perf_counter()
        frame_time = min(now - last_time, 0.25)
        last_time = now
        accumulator += frame_time

        time_to_wait = SIM_DT - accumulator
        if time_to_wait > 0.002:
            time.sleep(time_to_wait - 0.001)
            continue

        recv = osc.get_received_data() or {}

        if recv:
            cmd = recv.get('command')
            if cmd == 'stop':
                print(" [Genesis] [L3] 정지 명령 수신. 종료합니다.")
                break
            elif cmd == 'reset':
                print(" [Genesis] [L3] 리셋 명령 수신 — 전체 env 초기 포즈 복원.")
                car.set_pos(init_pos)
                car.set_quat(init_quat)
                try:
                    car.set_dofs_velocity(np.zeros((n_envs, car.n_dofs), dtype=np.float32))
                except Exception:
                    pass
                osc.clear_all_forces()
                step_count = 0
                osc.received_data['urdf_inputs'] = {}
                last_urdf_inputs = {}
                last_frame_id = 0
                st.reset_angles()
                osc.client_cpp.send_message("/Genesis/Init/TimeDilation", [1.0])
                if hasattr(run_l3, '_dilation_sent'):
                    delattr(run_l3, '_dilation_sent')
                last_time = time.perf_counter()
                accumulator = 0.0
                prev_state = st.capture(dynamic_obstacles, ue_driven_obstacle_ids, update_angles=False)
                curr_state = prev_state
            recv['command'] = None

            safe_overrides = osc.pop_overrides()
            safe_relative_cmds = osc.pop_relative_cmds()

            # [UE-DRIVEN OBSTACLE SYNC] — broadcast to all envs
            safe_obstacle_overrides = osc.pop_obstacle_overrides()
            for obs_id, obs_ent in dynamic_obstacles.items():
                if obs_id in safe_obstacle_overrides:
                    ovrd = safe_obstacle_overrides[obs_id]
                    if 'pos' in ovrd:
                        obs_ent.set_pos(np.tile(np.array(ovrd['pos'], dtype=np.float32), (n_envs, 1)))
                    if 'quat' in ovrd:
                        obs_ent.set_quat(np.tile(np.array(ovrd['quat'], dtype=np.float32), (n_envs, 1)))

            # Target teleport/relative commands — update only that env via envs_idx
            for tid, k in tid_to_env.items():
                needs_pos = needs_quat = needs_vel = False
                t_pos = t_quat = None

                if tid in safe_overrides:
                    ovrd = safe_overrides[tid]
                    if 'pos' in ovrd:
                        t_pos = np.array(ovrd['pos'], dtype=np.float32)
                        needs_pos = True
                    if 'quat' in ovrd:
                        t_quat = np.array(ovrd['quat'], dtype=np.float32)
                        needs_quat = True
                    if ovrd.get('teleport', True):
                        needs_vel = True
                    if args.verbose:
                        print(f" [Genesis] [L3] Target {tid} (env {k}) Override applied.")

                if tid in safe_relative_cmds:
                    if t_pos is None:
                        t_pos = np.atleast_2d(car.get_pos().cpu().numpy())[k].copy()
                    if t_quat is None:
                        t_quat = np.atleast_2d(car.get_quat().cpu().numpy())[k].copy()
                    for r_cmd in safe_relative_cmds[tid]:
                        ctype = r_cmd['type']
                        cdata = np.array(r_cmd['data'], dtype=np.float32)
                        tp = r_cmd.get('teleport', True)
                        if ctype == 'AddLocalOffset':
                            t_pos = t_pos + _q_rot(cdata, t_quat); needs_pos = True
                            if tp: needs_vel = True
                        elif ctype == 'AddWorldOffset':
                            t_pos = t_pos + cdata; needs_pos = True
                            if tp: needs_vel = True
                        elif ctype == 'AddLocalRotation':
                            t_quat = _q_mul(t_quat, cdata); needs_quat = True
                            if tp: needs_vel = True
                        elif ctype == 'AddWorldRotation':
                            t_quat = _q_mul(cdata, t_quat); needs_quat = True
                            if tp: needs_vel = True
                        else:
                            # AddWorldImpulse / AddWorldTorque: per-env velocity
                            # manipulation unsupported in batched mode v1 — warn and ignore
                            print(f" [Genesis] [L3] [WARN] relative cmd '{ctype}' 은 multi-env 모드에서 미지원 (무시).")

                if needs_pos:
                    car.set_pos(t_pos[None, :], envs_idx=[k])
                if needs_quat:
                    car.set_quat(t_quat[None, :], envs_idx=[k])
                if needs_vel:
                    try:
                        car.set_dofs_velocity(np.zeros((1, car.n_dofs), dtype=np.float32), envs_idx=[k])
                    except Exception:
                        pass

                if tid in recv.get('target_forces', {}) and np.any(np.array(recv['target_forces'][tid]) != 0):
                    if not warned_target_forces:
                        warned_target_forces = True
                        print(" [Genesis] [L3] [WARN] target_forces(지속 외력)는 multi-env 모드 v1 에서 미지원 — 무시합니다.")

            curr_state = st.capture(dynamic_obstacles, ue_driven_obstacle_ids, update_angles=False)

        catchup_steps = 0
        physics_dur_total = 0.0
        steps_limit = pacer.cap()

        while accumulator >= SIM_DT and catchup_steps < steps_limit:
            # Inputs: tid dict → (N,) arrays → one batched VehicleInputs
            queued_input = osc.pop_urdf_input()
            if queued_input:
                last_frame_id = queued_input['frame_id']
                last_urdf_inputs = queued_input['inputs']

            if isinstance(last_urdf_inputs, dict):
                for tid, raw in last_urdf_inputs.items():
                    k = tid_to_env.get(tid, tid_to_env.get(int(tid) if str(tid).isdigit() else tid))
                    if k is None:
                        continue
                    s, t, b = _parse_input_triplet(raw)
                    steer_arr[k] = s; throttle_arr[k] = t; brake_arr[k] = b
                    curr_inp = (round(s, 3), round(t, 3), round(b, 3))
                    if last_printed_inputs.get(tid) != curr_inp:
                        last_printed_inputs[tid] = curr_inp
                        print(f" [DEBUG] Vehicle {tid} (env {k}) Inputs: steer={s:.3f}, throttle={t:.3f}, brake={b:.3f}")

            veh.set_inputs(
                steer=torch.as_tensor(steer_arr, device=gs.device),
                throttle=torch.as_tensor(throttle_arr, device=gs.device),
                brake=torch.as_tensor(brake_arr, device=gs.device),
            )

            prev_state = curr_state
            physics_start = time.perf_counter()
            try:
                vs.step()               # measure_distances + physics.step(distances) + main.step
            except gs.GenesisException as e:
                if "Viewer closed" in str(e):
                    print("\n [Genesis] 시각화 창이 닫혔습니다. 정상 종료합니다.")
                    osc.close()
                    sys.exit(0)
                raise
            # [Instrumentation v1.1.3] GPU launches are async, so we must
            # synchronize here for Physics Avg to reflect actual execution
            # time — previously the execution tail after the step spilled into
            # the loop-bottom synchronize and was misattributed to 'serving'.
            if not use_cpu:
                torch.cuda.synchronize()
            physics_dur_total += time.perf_counter() - physics_start

            curr_state = st.capture(dynamic_obstacles, ue_driven_obstacle_ids, update_angles=True)

            accumulator -= SIM_DT
            catchup_steps += 1
            step_count += 1

        # Death-spiral prevention (same as legacy)
        if catchup_steps == steps_limit and accumulator >= SIM_DT:
            t_warn = time.perf_counter()
            if t_warn - last_slow_motion_warn_time >= 5.0:
                sim_ratio = (steps_limit * SIM_DT) / frame_time if frame_time > 0 else 1.0
                print(f" [WARNING] [Slow-Motion] Simulation lagging behind real-time. Running at {sim_ratio:.2f}x speed. (Next warning in 5s)")
                last_slow_motion_warn_time = t_warn
            accumulator = 0.0

        if catchup_steps > 0 and not use_cpu:
            torch.cuda.synchronize()

        # Interpolate + send (reuses legacy lerp_state / osc encoding)
        alpha = float(np.clip(accumulator / SIM_DT, 0.0, 0.9999))
        interpolated = lerp_state(prev_state, curr_state, alpha)
        if interpolated['targets']:
            osc.send_target_states_bulk(interpolated['targets'])
        if interpolated['dynamic_obstacles']:
            osc.send_dynamic_states_bulk(interpolated['dynamic_obstacles'])

        loop_dur = time.perf_counter() - loop_start
        pacer.update(catchup_steps, loop_dur)   # adaptive cap switch decision
        log_loop_dur_sum += loop_dur
        log_phys_dur_sum += physics_dur_total
        log_step_sum += catchup_steps
        log_count += 1

        if log_count >= 50:
            avg_loop = (log_loop_dur_sum / 50.0) * 1000.0
            avg_phys = (log_phys_dur_sum / 50.0) * 1000.0
            # Physics Avg is the 'sum' of catch-up steps per loop — print
            # steps/loop and per-step together to avoid confusion with the
            # per-step value (v1.0.7).
            steps_per_loop = log_step_sum / 50.0
            per_step = (log_phys_dur_sum / max(log_step_sum, 1)) * 1000.0
            print(f" [STATS] [L3 n_envs={n_envs}] Loop Avg: {avg_loop:.2f} ms | "
                  f"Physics Avg: {avg_phys:.2f} ms "
                  f"({steps_per_loop:.1f} steps/loop, {per_step:.2f} ms/step) "
                  f"[cap={pacer.cap()}:{pacer.mode}]")

            if not hasattr(run_l3, '_dilation_sent'):
                run_l3._dilation_sent = True
                avg_loop_sec = log_loop_dur_sum / 50.0
                dilation = sim_dt / avg_loop_sec if avg_loop_sec > sim_dt else 1.0
                print(f"  [Pacing] [TimeDilation] 실측 Loop Avg({avg_loop:.2f}ms) 기준 Dilation {dilation:.4f} 전송.")
                osc.client_cpp.send_message("/Genesis/Init/TimeDilation", [float(dilation)])

            log_loop_dur_sum = 0.0
            log_phys_dur_sum = 0.0
            log_step_sum = 0
            log_count = 0

    osc.close()
