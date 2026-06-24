"""L3 멀티-env 런타임 (genesis_vehicle.server) — ``--multi-env`` 모드 구현.

전제: 모든 타겟이 **동일 URDF** 의 차량이고 **차량 간 인터랙션이 없다**.
이때 N대의 차량을 "한 씬에 N개 엔티티"(per-entity, 선형 비용) 대신
**차량 1대 + ``scene.build(n_envs=N)``** 평행 환경으로 구동한다. 모든 물리
커널이 N개 환경을 동시에 처리하므로 스텝 비용이 차량 수와 거의 무관해진다
(실측: 30/50/100대 모두 ~19 ms/step, per-entity GPU 30대 1,115 ms 대비 57×).

매핑: target_id(정렬 순) ↔ env index. 환경(바닥/장애물)은 1세트만 추가하면
Genesis 가 env 마다 자동 복제한다.

per-entity 모드와 의도적으로 다른 점 / 한계:
  - 백엔드 기본 GPU (per-entity 모드는 CPU 가 정답이었던 것과 반대 —
    배칭이 있어야 GPU 커널 런치 고정비가 분산된다). ``--force-cpu`` 로 강제 가능.
  - 동적 장애물은 env 복사본이 N개 생기므로 "차량 A가 민 장애물을 차량 B가
    보는" 상호작용은 표현 불가. 상태 송신은 env 0 기준.
  - ``target_forces`` (지속 외력) 미지원 — 수신 시 1회 경고 후 무시.
  - lockstep 미지원 (OSC pacing 모드 고정).
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import genesis as gs

from genesis_vehicle import VehicleInputs, VehicleScene

from .osc_manager import OSCManager
from . import env_builder
from . import vehicle_builder


def _parse_input_triplet(raw):
    """legacy 와 동일한 입력 해석: len==4 → [_, steer, throttle, brake],
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
    """배치 텐서 ↔ tid 별 상태 dict 변환 + 스핀각 누적을 담당."""

    def __init__(self, car, physics, tids, n_envs, sim_dt):
        self.car = car
        self.physics = physics
        self.tids = list(tids)                  # env index 순서의 target id
        self.n_envs = n_envs
        self.sim_dt = sim_dt
        self.n_wheels = len(physics.resolved.wheels)

    def reset_angles(self):
        pass   # spin is owned by the SDK (wheel_visual_transforms); nothing to reset here

    def capture(self, dynamic_obstacles, ue_driven_obstacle_ids, update_angles):
        """state dict 를 배치 읽기로 생성.

        차체 pos/quat 2회 + SDK 닫힌형 wheel_visual_transforms 1회로 N대 전체를
        읽는다. 바퀴 pos/quat 은 steer+suspension+spin 이 모두 반영된 visual 포즈
        (VisualJointSync on/off 무관). spin 은 quat 에 포함되므로 w_angle=0."""
        bp = self.car.get_pos()
        bq = self.car.get_quat()
        if hasattr(bp, 'cpu'):
            bp = bp.cpu().numpy(); bq = bq.cpu().numpy()
        bp = np.atleast_2d(bp); bq = np.atleast_2d(bq)

        wp, wq = self.physics.wheel_visual_transforms("world")   # (N, n, 3/4)
        if hasattr(wp, 'cpu'):
            wp = wp.cpu().numpy(); wq = wq.cpu().numpy()

        state = {'targets': {}, 'dynamic_obstacles': {}}
        for k, tid in enumerate(self.tids):
            wheels_states = [
                (wp[k, j].copy(), wq[k, j].copy(), 0.0)
                for j in range(self.n_wheels)
            ]
            state['targets'][tid] = (bp[k].copy(), bq[k].copy(), wheels_states)

        # 동적 장애물: env 복제 한계로 env 0 기준 송신
        for o_id, ent in dynamic_obstacles.items():
            if o_id in ue_driven_obstacle_ids:
                continue
            p = ent.get_pos(); q = ent.get_quat()
            if hasattr(p, 'cpu'):
                p = p.cpu().numpy(); q = q.cpu().numpy()
            p = np.atleast_2d(p)[0]; q = np.atleast_2d(q)[0]
            state['dynamic_obstacles'][o_id] = (p.copy(), q.copy())
        return state


def run_l3(args):
    # 1. 백엔드: 배칭 모드에선 GPU 가 기본 (커널 런치 고정비를 n_envs 로 분산)
    use_cpu = bool(getattr(args, 'force_cpu', False)) or not torch.cuda.is_available()
    backend = gs.cpu if use_cpu else gs.gpu
    gs.init(backend=backend, logging_level="warning")
    print(f" [Genesis] [L3] Multi-env batched mode | backend = {'CPU' if use_cpu else 'GPU'}")

    # [Engine Hack] RigidGeom.n_cells Monkey-patch (legacy 와 동일)
    import genesis.engine.entities.rigid_entity.rigid_geom as rigid_geom
    rigid_geom.RigidGeom.n_cells = property(lambda self: 1)

    # 2. OSC 핸드셰이크 (legacy 와 동일 프로토콜)
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
    ue_dt = args.override_dt if args.override_dt is not None else init_physics.get('dt', 0.02)
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

    # 3. 씬 구성 — 환경은 1세트만 추가 (env 마다 자동 복제)
    _rigid_opts = gs.options.RigidOptions(
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
    # Unified two-scene raycast (VehicleScene raywheel): the road is RIGID in the
    # main scene (collision / rollover) and a KINEMATIC mirror is raycast in a
    # SEPARATE scene whose BVH is static and shared across envs — same trick as
    # the high-level API (see docs/two-scene-raycast.md). Supersedes the old
    # single-scene --road-raycast-only (kinematic road, no collision). Genesis is
    # already initialized by the server, so init_genesis=False.
    vs = VehicleScene(
        n_envs=n_envs, dt=ue_dt, backend=("cpu" if use_cpu else "gpu"),
        raycast_mode="raywheel", gravity=(0, 0, ue_gravity), substeps=2,
        rigid_options=_rigid_opts, show_viewer=not args.headless,
        init_genesis=False,
    )
    scene = vs.main_scene
    raycast_scene = vs.raycast_scene

    plane = None
    if not args.no_floor:
        plane = scene.add_entity(gs.morphs.Plane(),
                                 material=gs.materials.Rigid(friction=ue_friction, coup_restitution=ue_restitution))
        # Kinematic floor mirror so the wheels sense flat ground in the raycast scene.
        raycast_scene.add_entity(gs.morphs.Plane(),
                                 material=gs.materials.Kinematic(use_visual_raycasting=True))

    obstacles, dynamic_obstacles, initial_dynamic_states, ue_driven_obstacle_ids, extra_mass_entities = \
        env_builder.build_obstacles(
            scene=scene, init_data=init_data,
            ue_friction=ue_friction, ue_restitution=ue_restitution,
            vis_mode=args.vis_mode, verbose=args.verbose,
            road_raycast_only=False,   # road RIGID in main (collision); mirror handles raycast
            structures_as_primitive=getattr(args, "structures_as_primitive", False),
            raycast_scene=raycast_scene,
        )

    # 차량: 엔티티 1개 + Raycaster 1개 + 공유 cfg
    first_info = target_dict[tids[0]]
    t_fric = first_info.get('friction', ue_friction)
    temp_urdf = vehicle_builder.strip_wheel_collisions(urdf_path)
    car = scene.add_entity(
        gs.morphs.URDF(file=temp_urdf, pos=first_info.get('pos', [0, 0, 2]),
                       quat=first_info.get('quat', [1, 0, 0, 0]), fixed=False, align=False),
        material=gs.materials.Rigid(friction=t_fric, coup_restitution=0.0, sdf_cell_size=10000.0),
        surface=gs.surfaces.Rough(color=(1.0, 0.3, 0.3, 0.5)),
        vis_mode=args.vis_mode,
    )
    cfg = vehicle_builder.build_cfg(urdf_path, mapping, t_fric, target_id="L3-shared")
    # VisualJointSync only for the Genesis viewer; headless uses the closed-form
    # wheel_visual_transforms capture (skip the per-step engine FK cost).
    cfg.enable_visual_joint_sync = not args.headless
    # Register with VehicleScene: it adds the raycast-scene proxy + wheel sensor
    # and (at build) constructs the VehiclePhysics with sensor=None (distances
    # are injected from the raycast scene each step).
    veh = vs.add_vehicle(urdf_path, cfg=cfg, entity=car, name="L3-shared")

    # 4. 배치 빌드 — VehicleScene 이 main + raycast 씬을 함께 빌드하고 VehiclePhysics 생성
    vs.build()
    print(f" [DEBUG] Total rigid geoms after build: {scene.sim.rigid_solver.n_geoms}")
    print(f" [DEBUG] Total rigid links after build: {scene.sim.rigid_solver.n_links}")

    physics = veh.physics
    vehicle_builder.print_resolved_table("L3-shared", physics.resolved)

    # env 별 초기 포즈
    init_pos = np.array([target_dict[tid].get('pos', [0, 0, 2]) for tid in tids], dtype=np.float32)
    init_quat = np.array([target_dict[tid].get('quat', [1, 0, 0, 0]) for tid in tids], dtype=np.float32)
    car.set_pos(init_pos)
    car.set_quat(init_quat)
    try:
        car.set_dofs_velocity(np.zeros((n_envs, car.n_dofs), dtype=np.float32))
    except Exception:
        pass

    st = L3State(car, physics, tids, n_envs, ue_dt)

    # 5. 하드웨어 프로파일링 + Pacing (legacy 와 동일 절차)
    print("\n" + "="*50)
    print(" [INFO] [GENESIS] [L3] 하드웨어 연산 성능 실측 프로파일링 중...")
    print("="*50)
    warmup_starts = time.perf_counter()
    idle_inputs = VehicleInputs(throttle=0.0, brake=0.0, steer=0.0)
    for _ in range(5):
        physics.step(idle_inputs, distances=vs.measure_distances()[veh])
        scene.step()
        if not use_cpu:
            torch.cuda.synchronize()
    avg_step_time = (time.perf_counter() - warmup_starts) / 5.0
    print(f"  - 실측된 1스텝 평균 연산 속도: {avg_step_time * 1000.0:.2f} ms  (차량 {n_envs}대 분 배치)")

    sim_dt = ue_dt
    print(f"  [OK] [Determinism] 물리 해상도(sim_dt) {sim_dt * 1000.0:.1f}ms ({1.0/sim_dt:.1f}Hz) 고정.")
    print("="*50 + "\n")
    print(f" [Pacing] [Auto-Pacing] 클라이언트로 동조 주기({sim_dt * 1000.0:.1f}ms)를 전송합니다...")
    osc.client_cpp.send_message("/Genesis/Init/Pacing", [float(sim_dt)])

    SIM_DT = sim_dt
    MAX_SUBSTEPS = max(5, int(0.1 / sim_dt))
    accumulator = 0.0
    last_time = time.perf_counter()
    last_slow_motion_warn_time = 0.0

    from .physics_server import lerp_state   # 보간 로직 재사용 (state 포맷 동일)

    prev_state = st.capture(dynamic_obstacles, ue_driven_obstacle_ids, update_angles=False)
    curr_state = prev_state

    step_count = 0
    last_urdf_inputs = {}
    last_frame_id = 0
    last_printed_inputs = {}
    warned_target_forces = False

    log_loop_dur_sum = 0.0
    log_phys_dur_sum = 0.0
    log_count = 0

    # 입력 버퍼 (재할당 없이 재사용)
    steer_arr = np.zeros(n_envs, dtype=np.float32)
    throttle_arr = np.zeros(n_envs, dtype=np.float32)
    brake_arr = np.zeros(n_envs, dtype=np.float32)

    # =========================================================================
    # 메인 시뮬레이션 루프 (legacy 와 동일한 accumulator pacing)
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

            # [UE-DRIVEN OBSTACLE SYNC] — env 전체 브로드캐스트
            safe_obstacle_overrides = osc.pop_obstacle_overrides()
            for obs_id, obs_ent in dynamic_obstacles.items():
                if obs_id in safe_obstacle_overrides:
                    ovrd = safe_obstacle_overrides[obs_id]
                    if 'pos' in ovrd:
                        obs_ent.set_pos(np.tile(np.array(ovrd['pos'], dtype=np.float32), (n_envs, 1)))
                    if 'quat' in ovrd:
                        obs_ent.set_quat(np.tile(np.array(ovrd['quat'], dtype=np.float32), (n_envs, 1)))

            # 타겟 텔레포트/상대 명령 — envs_idx 로 해당 env 만 갱신
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
                            # AddWorldImpulse / AddWorldTorque: env 단위 속도 조작은
                            # 배치 모드 v1 미지원 — 경고 후 무시
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

        while accumulator >= SIM_DT and catchup_steps < MAX_SUBSTEPS:
            # 입력: tid dict → (N,) 배열 → 배치 VehicleInputs 1개
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

            inputs = VehicleInputs(
                steer=torch.as_tensor(steer_arr, device=gs.device),
                throttle=torch.as_tensor(throttle_arr, device=gs.device),
                brake=torch.as_tensor(brake_arr, device=gs.device),
            )

            prev_state = curr_state
            physics_start = time.perf_counter()
            try:
                physics.step(inputs, distances=vs.measure_distances()[veh])
                scene.step()
            except gs.GenesisException as e:
                if "Viewer closed" in str(e):
                    print("\n [Genesis] 시각화 창이 닫혔습니다. 정상 종료합니다.")
                    osc.close()
                    sys.exit(0)
                raise
            physics_dur_total += time.perf_counter() - physics_start

            curr_state = st.capture(dynamic_obstacles, ue_driven_obstacle_ids, update_angles=True)

            accumulator -= SIM_DT
            catchup_steps += 1
            step_count += 1

        # 데스 스파이럴 방지 (legacy 와 동일)
        if catchup_steps == MAX_SUBSTEPS and accumulator >= SIM_DT:
            t_warn = time.perf_counter()
            if t_warn - last_slow_motion_warn_time >= 5.0:
                sim_ratio = (MAX_SUBSTEPS * SIM_DT) / frame_time if frame_time > 0 else 1.0
                print(f" [WARNING] [Slow-Motion] Simulation lagging behind real-time. Running at {sim_ratio:.2f}x speed. (Next warning in 5s)")
                last_slow_motion_warn_time = t_warn
            accumulator = 0.0

        if catchup_steps > 0 and not use_cpu:
            torch.cuda.synchronize()

        # 보간 + 송신 (legacy lerp_state / osc 인코딩 재사용)
        alpha = float(np.clip(accumulator / SIM_DT, 0.0, 0.9999))
        interpolated = lerp_state(prev_state, curr_state, alpha)
        if interpolated['targets']:
            osc.send_target_states_bulk(interpolated['targets'])
        if interpolated['dynamic_obstacles']:
            osc.send_dynamic_states_bulk(interpolated['dynamic_obstacles'])

        loop_dur = time.perf_counter() - loop_start
        log_loop_dur_sum += loop_dur
        log_phys_dur_sum += physics_dur_total
        log_count += 1

        if log_count >= 50:
            avg_loop = (log_loop_dur_sum / 50.0) * 1000.0
            avg_phys = (log_phys_dur_sum / 50.0) * 1000.0
            print(f" [STATS] [L3 n_envs={n_envs}] Loop Avg: {avg_loop:.2f} ms | Physics Avg: {avg_phys:.2f} ms")

            if not hasattr(run_l3, '_dilation_sent'):
                run_l3._dilation_sent = True
                avg_loop_sec = log_loop_dur_sum / 50.0
                dilation = sim_dt / avg_loop_sec if avg_loop_sec > sim_dt else 1.0
                print(f"  [Pacing] [TimeDilation] 실측 Loop Avg({avg_loop:.2f}ms) 기준 Dilation {dilation:.4f} 전송.")
                osc.client_cpp.send_message("/Genesis/Init/TimeDilation", [float(dilation)])

            log_loop_dur_sum = 0.0
            log_phys_dur_sum = 0.0
            log_count = 0

    osc.close()
