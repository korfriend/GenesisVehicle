"""OSC 기반 Genesis 차량 물리 서버 (genesis_vehicle.server).

원본: genesis_unreal_plugin/physics_server.py — UE 전용이 아닌 범용 OSC
클라이언트 서버로 SDK 안에 편입. 변경점:
  - Windows 전용 코드(ctypes 패치, HIGH_PRIORITY_CLASS)에 플랫폼 가드
  - 패키지 상대 임포트 (sys.path 핵 제거)
  - ``--multi-env`` 플래그: 동일 URDF·비인터랙션 다수 차량을 L3(n_envs)
    배칭으로 구동 (l3_runtime.run_l3 로 디스패치)
"""

import os
import sys

# [Fix] PyInstaller ctypes wrapper crash monkey-patch (Windows 배포용 —
# Linux 에서는 Genesis 가 ctypes.CDLL(None) 으로 libc 를 여는 것을 깨뜨리므로 가드)
if sys.platform == "win32":
    import ctypes
    orig_CDLL = ctypes.CDLL
    class PatchedCDLL(orig_CDLL):
        def __init__(self, name, *args, **kwargs):
            if name is None:
                raise TypeError("name cannot be None")
            super().__init__(name, *args, **kwargs)
    ctypes.CDLL = PatchedCDLL

import time
import torch
import argparse
import numpy as np
import genesis as gs

# HIGH PRIORITY CLASS 설정 (Windows 전용 상수 — 타 플랫폼은 통상 우선순위 유지)
if sys.platform == "win32":
    try:
        import psutil
        psutil.Process(os.getpid()).nice(psutil.HIGH_PRIORITY_CLASS)
    except Exception:
        pass

from .osc_manager import OSCManager
from genesis_vehicle import (
    PacejkaAnisotropic, VehicleScene
)

# 쪼갠 모듈들 임포트 (패키지 내부 모듈 — 기존 코드의 참조 이름을 유지하기 위한 별칭)
from . import env_builder as genesis_env_builder
from . import vehicle_builder as genesis_vehicle_builder

# [Fix] Quadrants CUDA Graph 버그(환경변수 인식 오류) 해결
if "CUDA_PATH" in os.environ and "CUDA_HOME" not in os.environ:
    os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]


def slerp(q0, q1, t):
    """
    Spherical linear interpolation between two quaternions q0 and q1 with ratio t in [0, 1].
    Quaternions format: [w, x, y, z]
    """
    q0 = np.array(q0, dtype=np.float32)
    q1 = np.array(q1, dtype=np.float32)
    
    # Normalize inputs
    q0_norm = np.linalg.norm(q0)
    q1_norm = np.linalg.norm(q1)
    if q0_norm > 0: q0 /= q0_norm
    if q1_norm > 0: q1 /= q1_norm
    
    dot = np.sum(q0 * q1)
    
    # Take the shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot
        
    if dot > 0.9995:
        res = q0 + t * (q1 - q0)
        res_norm = np.linalg.norm(res)
        if res_norm > 0: res /= res_norm
        return res
        
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


def capture_state(target_entities, dynamic_obstacles, is_urdf_active, controllers, ue_driven_obstacle_ids, accumulated_wheel_angles=None, sim_dt=0.02, update_angles=False):
    """
    Captures positions and rotations for all target entities (including wheels if active)
    and dynamic obstacles in Genesis physics engine.
    """
    state = {
        'targets': {},
        'dynamic_obstacles': {}
    }
    
    # 1. Target entities (Chassis & Wheels)
    if target_entities:
        tids = list(target_entities.keys())
        pos_tensors  = [ent.get_pos()  for ent in target_entities.values()]
        quat_tensors = [ent.get_quat() for ent in target_entities.values()]
        
        if hasattr(pos_tensors[0], 'cpu'):
            pos_batch  = torch.stack(pos_tensors).cpu().numpy()
            quat_batch = torch.stack(quat_tensors).cpu().numpy()
        else:
            pos_batch  = np.array(pos_tensors)
            quat_batch = np.array(quat_tensors)
            
        for i, tid in enumerate(tids):
            p = pos_batch[i]
            q = quat_batch[i]
            if p.ndim > 1: p = p[0]
            if q.ndim > 1: q = q[0]
            
            p = p.copy()
            q = q.copy()
            
            wheels_states = []
            if is_urdf_active and controllers and tid in controllers:
                ctrl = controllers[tid]
                # [VISUAL] SDK closed-form wheel pose: steer + suspension + spin
                # baked into pos/quat, computed WITHOUT driving Genesis joints
                # (works even with VisualJointSync off). Replaces the get_link read,
                # which required VisualJointSync on and returned a rest pose otherwise.
                # The quat already includes spin → send w_angle=0 (UE uses the
                # quat directly; no separate spin to re-apply).
                if hasattr(ctrl, 'wheel_visual_transforms'):
                    try:
                        wp_all, wq_all = ctrl.wheel_visual_transforms("world")
                        if hasattr(wp_all, 'cpu'):
                            wp_all = wp_all.cpu().numpy(); wq_all = wq_all.cpu().numpy()
                        wp_all = wp_all[0]; wq_all = wq_all[0]   # env 0 (single-env)
                        for j in range(wp_all.shape[0]):
                            wheels_states.append((wp_all[j].copy(), wq_all[j].copy(), 0.0))
                    except Exception:
                        pass
            state['targets'][tid] = (p, q, wheels_states)
            
    # 2. Dynamic obstacles
    if dynamic_obstacles:
        oids = list(dynamic_obstacles.keys())
        opos_tensors  = [ent.get_pos()  for ent in dynamic_obstacles.values()]
        oquat_tensors = [ent.get_quat() for ent in dynamic_obstacles.values()]
        
        if hasattr(opos_tensors[0], 'cpu'):
            opos_batch  = torch.stack(opos_tensors).cpu().numpy()
            oquat_batch = torch.stack(oquat_tensors).cpu().numpy()
        else:
            opos_batch  = np.array(opos_tensors)
            oquat_batch = np.array(oquat_tensors)
            
        for i, o_id in enumerate(oids):
            if o_id in ue_driven_obstacle_ids:
                continue
            p = opos_batch[i]
            q = oquat_batch[i]
            if p.ndim > 1: p = p[0]
            if q.ndim > 1: q = q[0]
            
            state['dynamic_obstacles'][o_id] = (p.copy(), q.copy())
            
    return state


def lerp_state(prev, curr, a):
    """
    Interpolates states between prev and curr with factor a in [0, 1).
    """
    interpolated = {
        'targets': [],
        'dynamic_obstacles': []
    }
    
    # 1. Target entities interpolation
    for tid, curr_data in curr['targets'].items():
        if tid in prev['targets']:
            prev_p, prev_q, prev_wheels = prev['targets'][tid]
            curr_p, curr_q, curr_wheels = curr_data
            
            # Position LERP
            interp_p = prev_p * (1.0 - a) + curr_p * a
            # Rotation SLERP
            interp_q = slerp(prev_q, curr_q, a)
            
            # Wheels LERP/SLERP
            interp_wheels = []
            for j in range(min(len(prev_wheels), len(curr_wheels))):
                pw_tuple = prev_wheels[j]
                cw_tuple = curr_wheels[j]
                
                pw_p, pw_q = pw_tuple[0], pw_tuple[1]
                cw_p, cw_q = cw_tuple[0], cw_tuple[1]
                pw_angle = pw_tuple[2] if len(pw_tuple) > 2 else 0.0
                cw_angle = cw_tuple[2] if len(cw_tuple) > 2 else 0.0
                
                iw_p = pw_p * (1.0 - a) + cw_p * a
                iw_q = slerp(pw_q, cw_q, a)
                
                # Boundary wrap-around LERP for wheel angle
                diff = cw_angle - pw_angle
                diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
                iw_angle = (pw_angle + diff * a) % (2.0 * np.pi)
                
                interp_wheels.append((iw_p, iw_q, iw_angle))
                
            interpolated['targets'].append((tid, interp_p, interp_q, interp_wheels))
        else:
            interpolated['targets'].append((tid, curr_data[0], curr_data[1], curr_data[2]))
            
    # 2. Dynamic obstacles interpolation
    for o_id, curr_data in curr['dynamic_obstacles'].items():
        if o_id in prev['dynamic_obstacles']:
            prev_p, prev_q = prev['dynamic_obstacles'][o_id]
            curr_p, curr_q = curr_data
            
            interp_p = prev_p * (1.0 - a) + curr_p * a
            interp_q = slerp(prev_q, curr_q, a)
            interpolated['dynamic_obstacles'].append((o_id, interp_p, interp_q))
        else:
            interpolated['dynamic_obstacles'].append((o_id, curr_data[0], curr_data[1]))
            
    return interpolated


def main():
    parser = argparse.ArgumentParser(description="Genesis Ball Kick Sample with Unreal Engine OSC Bridge")
    parser.add_argument("--send_ip", type=str, default="127.0.0.1", help="Target IP address for Unreal Engine (default: 127.0.0.1)")
    parser.add_argument("--recv_port", type=int, default=7001, help="Port to receive data from UE (default: 7001)")
    parser.add_argument("--send_port", type=int, default=7002, help="Port to send State data to UE (default: 7002)")
    parser.add_argument("--send_port_obs", type=int, default=7004, help="Port to send Observation data to UE (default: 7004)")
    # parser.add_argument("--cpu", action="store_true", help="Force backend to CPU instead of GPU")
    parser.add_argument("--headless", action="store_true", help="Run without Genesis visualizer window")
    # parser.add_argument("--lockstep", action="store_true", help="Enable strict Frame-ID based synchronization (Default: False)")
    parser.add_argument("--no-floor", action="store_true", help="Disable the default ground plane")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose OSC logging")
    parser.add_argument("--vis_mode", type=str, default="collision", choices=["visual", "collision"], help="Visualization mode for the scene (default: collision)")
    parser.add_argument("--override_dt", type=float, default=None, help="Override simulation time step (dt) in seconds (e.g. 0.01 for 100Hz)")
    parser.add_argument("--no-target-collision", action="store_true", help="Disable collision between target entities")
    parser.add_argument("--road-raycast-only", action="store_true",
                        help="Load complex road/terrain meshes ([Complex]) as VISUAL raycast "
                             "targets with no collision geometry. Skips CoACD convex "
                             "decomposition and the chassis-vs-road narrow-phase entirely; "
                             "the ray-cast wheels still follow the surface. Big win for large "
                             "chassis (e.g. tanks) on complex maps.")
    parser.add_argument("--structures-as-primitive", action="store_true",
                        help="Replace every MESH collider (obstacles/structures) with its "
                             "bounding BOX. Mesh colliders carry a per-geom SDF processed each "
                             "step even with zero contact (~0.6ms/mesh on CPU → 100s of "
                             "structures = 100s of ms). Box collision is analytic (~0 when not "
                             "touching), so cost scales with actual contacts, not structure "
                             "count. Use with --road-raycast-only (roads stay raycast surfaces).")
    parser.add_argument("--multi-env", action="store_true",
                        help="L3 batched mode: N identical, non-interacting vehicles as n_envs=N "
                             "(one vehicle entity, GPU-batched). Requires all targets to share one URDF.")
    parser.add_argument("--force-cpu", action="store_true",
                        help="(multi-env mode) force CPU backend instead of GPU")
    args = parser.parse_args()

    # [L3] 멀티-env 배칭 모드 — 동일 URDF 다수 차량 전용 경로로 분기
    if args.multi_env:
        from .l3_runtime import run_l3
        run_l3(args)
        return

    # 배포용 강제 안전가드: 기본 CPU 가동 및 비-락스텝(OSC Pacing) 처리
    # (per-entity 모드에선 CPU 가 GPU 보다 빠름 — n_envs=1 다중 엔티티는 커널 런치 바운드)
    args.cpu = True
    args.lockstep = False

    # 1. 제네시스 물리 엔진 초기화 및 백엔드 결정
    if args.cpu:
        backend = gs.cpu
        # print(" [Genesis] [Init] 백엔드 모드: CPU (기본 강제 지정)")
    else:
        if torch.cuda.is_available():
            backend = gs.gpu
            # print(" [Genesis] [Init] 백엔드 모드: GPU (CUDA 가용)")
        else:
            backend = gs.cpu
            # print(" [Genesis] [Warning] GPU 백엔드가 요청되었으나 CUDA를 사용할 수 없습니다. CPU 백엔드로 안전하게 폴백합니다.")

    VehicleScene.init_backend("cpu" if backend is gs.cpu else "gpu")

    # [Engine Hack] RigidGeom.n_cells Monkey-patch
    import genesis.engine.entities.rigid_entity.rigid_geom as rigid_geom
    rigid_geom.RigidGeom.n_cells = property(lambda self: 1)

    # 2. OSC 통신 매니저 생성
    osc = OSCManager(
        send_ip=args.send_ip,
        recv_port=args.recv_port,
        send_port_cpp=args.send_port,
        send_port_obs=args.send_port_obs,
        subject_name='Genesis',
        verbose=args.verbose
    )
    
    print("\n" + "="*50)
    print(" Genesis Ball Kick Sample (with Unreal Engine) - [Modularized]")
    print(" 언리얼 엔진에서 '플레이(Play)' 버튼을 누를 때까지 대기합니다...")
    print("="*50 + "\n")

    # [Correct Fix] osc_manager.py의 wait_for_initialization 내부에서 자동으로 1초 주기로 RequestInit을 보내며 대기하므로,
    # 여기서 이중 루프를 돌며 초기 데이터 버퍼를 비워버리는 병목을 원천 제거합니다.
    init_data = osc.wait_for_initialization(timeout=300.0)
    init_physics = init_data.get('physics', {})
    ue_gravity = init_physics.get('gravity', -9.81)
    ue_dt = args.override_dt if args.override_dt is not None else init_physics.get('dt', 0.02)
    ue_friction = init_physics.get('friction', 2.0)
    ue_restitution = 0.0
    
    initial_physics_state = {
        'gravity': ue_gravity,
        'dt': ue_dt,
        'friction': ue_friction,
        'restitution': ue_restitution
    }
    
    # 4. Genesis Scene Setup — VehicleScene(inline) for unified vehicle handling.
    # Per-entity mode is interacting vehicles at n_envs=1 on CPU, where the
    # two-scene raycast has no benefit, so inline == one scene == prior behavior.
    # All geometry is registered via vs.add_* (no raw scene access); build() /
    # step() and sim reads/tweaks route through vs accessors. Genesis is already
    # initialized above, so init_genesis=False.
    vs = VehicleScene(
        n_envs=1, dt=ue_dt, raycast_mode="inline",
        gravity=(0, 0, ue_gravity), substeps=2, show_viewer=not args.headless,
        init_genesis=False,
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=False,
            enable_adjacent_collision=False,
            enable_neutral_collision=False,
            enable_collision=True,
            tolerance=0.001,
            use_gjk_collision=False, # [최적화] MPR 전환
            box_box_detection=True,
            use_hibernation=True, # [최적화] Sleep 모드 활성화
            broadphase_traversal=gs.broadphase_traversal.SAP, # [최적화] Sweep-and-Prune
            max_collision_pairs=2048, # [최적화] 충돌 쌍 사전할당
        ),
    )
    target_entities = {}
    controllers = {}        # {tid: VehiclePhysics} — populated after vs.build()
    vehicles = {}           # {tid: Vehicle handle} — for set_inputs in the loop
    entities_to_set_mass = []
    is_urdf_active = False
    urdf_path = ""
    
    # 바닥면(Plane) 생성
    plane = None
    if not args.no_floor:
        plane = vs.add_static(morph=gs.morphs.Plane(),
                              material=gs.materials.Rigid(friction=ue_friction, coup_restitution=ue_restitution),
                              name="ground")
    
    # [MULTI-TARGET INITIALIZATION]
    if osc.urdf_init_request and osc.urdf_init_request.get('path'):
        urdf_path = osc.urdf_init_request['path']
        is_urdf_active = True
        print(f" [Genesis] [Init] Multi-Target (URDF) Active: {urdf_path}")

    target_dict = init_data.get('target', {})
    if not isinstance(target_dict, dict) or len(target_dict) == 0:
        target_dict = { 0: {'pos': [0,0,2], 'quat': [1,0,0,0], 'mass': 1.0, 'friction': ue_friction, 'restitution': ue_restitution, 'type': 2, 'scale': [0.1, 0.1, 0.1]} }
        
    for target_id, target_info in target_dict.items():
        if is_urdf_active:
            # 헬퍼 모듈을 이용해 URDF 차량 세팅 및 Raycaster 인스턴스화 수행
            mapping = osc.urdf_init_request.get('mapping', {})
            genesis_vehicle_builder.build_vehicle(
                vs=vs,
                target_entities=target_entities,
                vehicles=vehicles,
                target_id=target_id,
                target_info=target_info,
                urdf_path=urdf_path,
                mapping=mapping,
                ue_friction=ue_friction,
                ue_restitution=ue_restitution,
                vis_mode=args.vis_mode,
                # VisualJointSync is auto-managed by VehicleScene.build() (on iff
                # the main scene is rendered); headless uses closed-form
                # wheel_visual_transforms, so no per-step engine FK is paid.
            )
        else:
            # 일반 타겟 빌딩
            t_pos = target_info.get('pos', [0, 0, 2])
            t_quat = target_info.get('quat', [1, 0, 0, 0])
            obs_type = target_info.get('type', 2)
            scale = target_info.get('scale', [1.0, 1.0, 1.0])
            size = (scale[0] * 1.0, scale[1] * 1.0, scale[2] * 1.0)
            
            if obs_type == 1:
                size = (max(size[0], 0.01), max(size[1], 0.01), max(size[2], 0.01))
                target_morph = gs.morphs.Box(size=size, pos=t_pos, quat=t_quat)
            elif obs_type == 2:
                target_morph = gs.morphs.Sphere(radius=max(size[0] / 2.0, 0.01), pos=t_pos, quat=t_quat)
            elif obs_type == 3:
                target_morph = gs.morphs.Cylinder(radius=max(size[0] / 2.0, 0.01), height=max(size[2], 0.01), pos=t_pos, quat=t_quat)
            else:
                size = (max(size[0], 0.01), max(size[1], 0.01), 0.01)
                target_morph = gs.morphs.Box(size=size, pos=t_pos, quat=t_quat)

            t_color = (1.0, 0.8, 0.1, 0.6)
            # Generic (non-vehicle) target — a free rigid body the OSC layer
            # teleports. Routed through VehicleScene like any obstacle.
            _obs = vs.add_dynamic(
                target_morph, physics=True,
                material=gs.materials.Rigid(friction=ue_friction, coup_restitution=ue_restitution, sdf_cell_size=10000.0),
                surface=gs.surfaces.Rough(color=t_color), vis_mode=args.vis_mode,
                mass=target_info.get('mass', 1.0), name=f"target_{target_id}")
            target_entities[target_id] = _obs.entity_main
            print(f" [Genesis] Created Target {target_id} at {t_pos}")
            
    # 5. 언리얼 엔진 환경 동기화 (헬퍼 모듈에 위임)
    obstacles, dynamic_obstacles, initial_dynamic_states, ue_driven_obstacle_ids, extra_mass_entities = genesis_env_builder.build_obstacles(
        vs=vs,
        init_data=init_data,
        ue_friction=ue_friction,
        ue_restitution=ue_restitution,
        vis_mode=args.vis_mode,
        verbose=args.verbose,
        road_raycast_only=args.road_raycast_only,
        structures_as_primitive=args.structures_as_primitive,
    )
    entities_to_set_mass.extend(extra_mass_entities)

    # 6. 배경 및 시뮬레이션 설정
    if args.no_target_collision:
        print(" [Genesis] 옵션에 의해 차량(Target) 상호 간의 충돌을 비활성화합니다.")
        for idx, (tid, tentity) in enumerate(target_entities.items()):
            for link in tentity.links:
                for geom in link.geoms:
                    geom._contype = (1 << (idx + 1))
                    geom._conaffinity = 1

    vs.build()      # builds vs.main_scene + constructs each vehicle's VehiclePhysics
    print(f" [DEBUG] Total rigid geoms after build: {vs.rigid_solver.n_geoms}")
    print(f" [DEBUG] Total rigid links after build: {vs.rigid_solver.n_links}")

    # Populate the controllers dict the OSC / state-capture code reads with the
    # Vehicle HANDLES (solver-agnostic — veh.wheel_visual_transforms / veh.resolved
    # work in both per_vehicle and batched modes), and print the resolved table.
    for tid, veh in vehicles.items():
        controllers[tid] = veh
        genesis_vehicle_builder.print_resolved_table(tid, veh.resolved)

    # 텐서 관련 Monkey Patch 적용
    genesis_vehicle_builder.apply_monkey_patches(vs.rigid_solver)

    # 질량 덮어쓰기 적용
    for entity, mass in entities_to_set_mass:
        entity.set_mass(mass)
        
    # [NEW] 하드웨어 성능 자율 측정 기반 1배속 정속 자동 튜닝 (Auto-Pacing)
    print("\n" + "="*50)
    print(" [INFO] [GENESIS] 하드웨어 연산 성능 실측 프로파일링 중...")
    print("="*50)
    
    # 5회 시험 step 구동하여 스텝당 평균 물리 연산 시간 실측 (GPU일 때는 명시적 동기화 적용)
    warmup_starts = time.perf_counter()
    for _ in range(5):
        vs.step()
        if not args.cpu and torch.cuda.is_available():
            torch.cuda.synchronize()
    warmup_ends = time.perf_counter()
    avg_step_time = (warmup_ends - warmup_starts) / 5.0
    
    print(f"  - 실측된 1스텝 평균 연산 속도: {avg_step_time * 1000.0:.2f} ms")
    
    # [CRITICAL FIX] 물리적 시간 흐름 및 동역학 일관성(Determinism)을 위해 sim_dt는 항상 고정 고수합니다.
    sim_dt = ue_dt
    vs.sim_options.dt = sim_dt
    print(f"  [OK] [Determinism] 물리 해상도(sim_dt)가 표준 {sim_dt * 1000.0:.1f}ms ({1.0/sim_dt:.1f}Hz)로 설정되었습니다.")
    print("="*50 + "\n")
    
    # 언리얼 엔진에 고정 동기 주기를 확인 전송합니다.
    print(f" [Pacing] [Auto-Pacing] 언리얼 엔진으로 동조 주기({sim_dt * 1000.0:.1f}ms)를 전송합니다...")
    osc.client_cpp.send_message("/Genesis/Init/Pacing", [float(sim_dt)])
        
    SIM_DT = sim_dt
    MAX_SUBSTEPS = max(5, int(0.1 / sim_dt))
    accumulator = 0.0
    last_time = time.perf_counter()
    last_slow_motion_warn_time = 0.0

    accumulated_wheel_angles = {}
    prev_state = capture_state(target_entities, dynamic_obstacles, is_urdf_active, controllers, ue_driven_obstacle_ids, accumulated_wheel_angles, sim_dt, False)
    curr_state = prev_state

    step_count = 0
    last_urdf_inputs = {}
    last_frame_id = 0
    last_printed_inputs = {}
    last_step_time = None
    
    log_loop_dur_sum = 0.0
    log_phys_dur_sum = 0.0
    log_count = 0

    # =========================================================================
    # 메인 시뮬레이션 루프
    # =========================================================================
    last_physics_time = time.perf_counter()
    last_real_time = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        if args.lockstep:
            recv = osc.wait_for_next_frame(timeout=3.0)
            if not recv:
                continue
            steps_to_run = 1
        else:
            now = time.perf_counter()
            frame_time = now - last_time
            last_time = now
            frame_time = min(frame_time, 0.25)  # Prevents death spiral

            accumulator += frame_time

            # CPU 보호용 Sleep 기법: 남은 시간이 너무 크면 대기하여 100% CPU 방지
            time_to_wait = SIM_DT - accumulator
            if time_to_wait > 0.002:
                time.sleep(time_to_wait - 0.001)
                continue

            steps_to_run = 0
            recv = osc.get_received_data() or {}

        if recv:
            cmd = recv.get('command')
            if cmd == 'stop':
                print(" [Genesis] 언리얼 엔진으로부터 정지 명령 수신. 종료합니다.")
                break
            elif cmd == 'reset':
                print(" [Genesis] Multi-Target 리셋 명령 수신.")
                for tid, tentity in target_entities.items():
                    t_info = init_data.get('target', {}).get(tid, {})
                    i_pos = t_info.get('pos', [0,0,2])
                    i_quat = t_info.get('quat', [1,0,0,0])
                    tentity.set_pos(np.array(i_pos, dtype=np.float32))
                    tentity.set_quat(np.array(i_quat, dtype=np.float32))
                    tentity.set_dofs_velocity(np.zeros(tentity.n_dofs, dtype=np.float32))
                    
                    if hasattr(tentity, 'control_dofs_force'):
                        tentity.control_dofs_force(np.zeros(6, dtype=np.float32), slice(0, 6))
                        tentity.control_dofs_force(np.zeros(tentity.n_dofs, dtype=np.float32), slice(0, tentity.n_dofs))
                    
                    if hasattr(tentity, 'set_friction'): 
                        tentity.set_friction(initial_physics_state['friction'])
                    if hasattr(tentity, 'set_restitution'): 
                        tentity.set_restitution(initial_physics_state['restitution'])

                for obs_id, obs_ent in dynamic_obstacles.items():
                    if obs_id in initial_dynamic_states:
                        i_pos, i_quat = initial_dynamic_states[obs_id]
                        obs_ent.set_pos(i_pos)
                        obs_ent.set_quat(i_quat)
                        if hasattr(obs_ent, 'set_dofs_velocity'):
                            obs_ent.set_dofs_velocity(np.zeros(obs_ent.n_dofs, dtype=np.float32))
                
                osc.clear_all_forces()
                step_count = 0
                osc.received_data['urdf_inputs'] = {}
                last_urdf_inputs = {}
                last_frame_id = 0
                
                print(f" [Genesis] Physics Resetting to Initial Values...")
                vs.sim_options.gravity = (0, 0, initial_physics_state['gravity'])
                vs.sim_options.dt = initial_physics_state['dt']
                
                accumulated_wheel_angles.clear()

                # 리셋 신호 발생 시 언리얼 Dilation 배속도 1.0배속으로 초기화 복원
                osc.client_cpp.send_message("/Genesis/Init/TimeDilation", [1.0])
                if hasattr(main, '_dilation_sent'):
                    delattr(main, '_dilation_sent')
                if hasattr(main, 'debug_step_count'):
                    delattr(main, 'debug_step_count')

                if plane and hasattr(plane, 'set_friction'): plane.set_friction(initial_physics_state['friction'])
                if plane and hasattr(plane, 'set_restitution'): plane.set_restitution(initial_physics_state['restitution'])
                
                # 타이밍 및 상태 초기화
                last_time = time.perf_counter()
                accumulator = 0.0
                prev_state = capture_state(target_entities, dynamic_obstacles, is_urdf_active, controllers, ue_driven_obstacle_ids, accumulated_wheel_angles, sim_dt, False)
                curr_state = prev_state

            recv['command'] = None

            safe_overrides = osc.pop_overrides()
            safe_relative_cmds = osc.pop_relative_cmds()

            # [UE-DRIVEN OBSTACLE SYNC]
            safe_obstacle_overrides = osc.pop_obstacle_overrides()
            for obs_id, obs_ent in dynamic_obstacles.items():
                if obs_id in safe_obstacle_overrides:
                    ovrd = safe_obstacle_overrides[obs_id]
                    if 'pos' in ovrd:
                        obs_ent.set_pos(np.array(ovrd['pos'], dtype=np.float32))
                    if 'quat' in ovrd:
                        obs_ent.set_quat(np.array(ovrd['quat'], dtype=np.float32))

            for tid, tentity in target_entities.items():
                current_needs_pos = False
                current_needs_quat = False
                current_needs_vel = False
                t_pos = None
                t_quat = None
                t_vel = None

                if tid in safe_overrides:
                    ovrd = safe_overrides[tid]
                    if 'pos' in ovrd:
                        t_pos = np.array(ovrd['pos'], dtype=np.float32)
                        current_needs_pos = True
                    if 'quat' in ovrd:
                        t_quat = np.array(ovrd['quat'], dtype=np.float32)
                        current_needs_quat = True
                    if ovrd.get('teleport', True):
                        t_vel = np.zeros(tentity.n_dofs, dtype=np.float32)
                        current_needs_vel = True

                if tid in safe_relative_cmds:
                    def q_mul(q1, q2):
                        w1, x1, y1, z1 = q1
                        w2, x2, y2, z2 = q2
                        return np.array([
                            w1*w2 - x1*x2 - y1*y2 - z1*z2,
                            w1*x2 + x1*w2 + y1*z2 - z1*y2,
                            w1*y2 - x1*z2 + y1*w2 + z1*x2,
                            w1*z2 + x1*y2 - y1*x2 + z1*w2
                        ], dtype=np.float32)

                    def q_rot(v, q):
                        qw, qv = q[0], q[1:4]
                        t = 2.0 * np.cross(qv, v)
                        return v + qw * t + np.cross(qv, t)

                    if t_pos is None: t_pos = tentity.get_pos().cpu().numpy()
                    if t_quat is None: t_quat = tentity.get_quat().cpu().numpy()
                    if t_vel is None: t_vel = tentity.get_dofs_velocity().cpu().numpy()

                    for r_cmd in safe_relative_cmds[tid]:
                        ctype = r_cmd['type']
                        cdata = np.array(r_cmd['data'], dtype=np.float32)
                        tp = r_cmd.get('teleport', True)

                        if ctype == 'AddLocalOffset':
                            t_pos += q_rot(cdata, t_quat)
                            current_needs_pos = True
                            if tp: t_vel[:] = 0; current_needs_vel = True
                        elif ctype == 'AddWorldOffset':
                            t_pos += cdata
                            current_needs_pos = True
                            if tp: t_vel[:] = 0; current_needs_vel = True
                        elif ctype == 'AddLocalRotation':
                            t_quat = q_mul(t_quat, cdata)
                            current_needs_quat = True
                            if tp: t_vel[:] = 0; current_needs_vel = True
                        elif ctype == 'AddWorldRotation':
                            t_quat = q_mul(cdata, t_quat)
                            current_needs_quat = True
                            if tp: t_vel[:] = 0; current_needs_vel = True
                        elif ctype == 'AddWorldImpulse':
                            mass = tentity.get_mass()
                            if hasattr(mass, 'cpu'): mass = mass.cpu().item()
                            mass = max(float(mass), 1e-3)
                            t_vel[0:3] += cdata / mass
                            current_needs_vel = True
                        elif ctype == 'AddWorldTorque':
                            q_conj = np.array([t_quat[0], -t_quat[1], -t_quat[2], -t_quat[3]])
                            local_t = q_rot(cdata, q_conj)
                            t_vel[3:6] += local_t
                            current_needs_vel = True

                if current_needs_pos: tentity.set_pos(t_pos)
                if current_needs_quat: tentity.set_quat(t_quat)
                if current_needs_vel: tentity.set_dofs_velocity(t_vel)

                if tid in recv.get('target_forces', {}):
                    fw = np.array(recv['target_forces'][tid], dtype=np.float32)
                    if np.any(fw != 0):
                        ft = np.zeros(6, dtype=np.float32)
                        ft[0:3] = fw
                        tentity.control_dofs_force(ft, slice(0, 6))

            # 오버라이드 반영 후 상태 갱신
            curr_state = capture_state(target_entities, dynamic_obstacles, is_urdf_active, controllers, ue_driven_obstacle_ids, accumulated_wheel_angles, sim_dt, False)

        catchup_steps = 0
        physics_dur_total = 0.0

        if args.lockstep:
            # Lockstep 모드는 고정 1스텝씩 가동
            steps_limit = 1
        else:
            steps_limit = MAX_SUBSTEPS

        # Catch-up Multi-Step Loop (물리 20ms 고정 단위 소비)
        while (args.lockstep and steps_limit > 0) or (not args.lockstep and accumulator >= SIM_DT and catchup_steps < MAX_SUBSTEPS):
            if controllers:
                if not args.lockstep:
                    queued_input = osc.pop_urdf_input()
                    if queued_input:
                        last_frame_id = queued_input['frame_id']
                        last_urdf_inputs = queued_input['inputs']
                else:
                    if 'urdf_inputs' in recv:
                        last_frame_id = recv.get('frame_id', last_frame_id)
                        last_urdf_inputs = recv['urdf_inputs']

                if isinstance(last_urdf_inputs, dict):
                    for tid, veh in vehicles.items():
                        steer = throttle = brake = 0.0
                        if tid in last_urdf_inputs:
                            raw_in = list(last_urdf_inputs[tid])
                            if len(raw_in) == 4:
                                steer    = float(raw_in[1])
                                throttle = float(raw_in[2])
                                brake    = float(raw_in[3])
                            elif len(raw_in) >= 3:
                                steer    = float(raw_in[0])
                                throttle = float(raw_in[1])
                                brake    = float(raw_in[2])

                        # Stored on the handle; vs.step() (below) applies them.
                        veh.set_inputs(steer=steer, throttle=throttle, brake=brake)

                        curr_inp = (round(steer, 3), round(throttle, 3), round(brake, 3))
                        if last_printed_inputs.get(tid) != curr_inp:
                            last_printed_inputs[tid] = curr_inp
                            print(f" [DEBUG] Vehicle {tid} Inputs: steer={steer:.3f}, throttle={throttle:.3f}, brake={brake:.3f}")
            
            # 이전 물리 상태 보존
            prev_state = curr_state
            
            try:
                physics_start = time.perf_counter()
                vs.step()       # applies each vehicle's inputs (physics) + steps the scene
                physics_end = time.perf_counter()
                last_step_time = physics_end
                physics_dur_total += (physics_end - physics_start)
            except gs.GenesisException as e:
                if "Viewer closed" in str(e):
                    print("\n [Genesis] 시각화 창이 닫혔습니다. 시뮬레이션을 정상 종료합니다.")
                    osc.close()
                    sys.exit(0)
                else:
                    raise e
            
            # 최신 물리 상태 기록
            curr_state = capture_state(target_entities, dynamic_obstacles, is_urdf_active, controllers, ue_driven_obstacle_ids, accumulated_wheel_angles, sim_dt, True)
            
            if not args.lockstep:
                accumulator -= SIM_DT
            else:
                steps_limit -= 1
                
            catchup_steps += 1
            step_count += 1






        # 데스 스파이럴 방지: 따라잡지 못하면 어큐뮬레이터 탕감 및 슬로우 모션 경고 (5초 간격으로 스로틀링 출력)
        if not args.lockstep and catchup_steps == MAX_SUBSTEPS and accumulator >= SIM_DT:
            current_warn_time = time.perf_counter()
            if current_warn_time - last_slow_motion_warn_time >= 5.0:
                sim_ratio = (MAX_SUBSTEPS * SIM_DT) / frame_time if frame_time > 0 else 1.0
                print(f" [WARNING] [Slow-Motion] Simulation lagging behind real-time. Running at {sim_ratio:.2f}x speed. (Next warning in 5s)")
                last_slow_motion_warn_time = current_warn_time
            accumulator = 0.0

        # 모든 캐치업 물리 연산 완료 시 GPU 동기화
        if catchup_steps > 0:
            if not args.cpu and torch.cuda.is_available():
                torch.cuda.synchronize()

        # [상태 보간(LERP/SLERP) 및 언리얼 텔레메트리 송신]
        if not args.lockstep:
            alpha = accumulator / SIM_DT
            alpha = float(np.clip(alpha, 0.0, 0.9999))
            interpolated = lerp_state(prev_state, curr_state, alpha)
            
            # 최종 보간 상태 벌크 전송
            if interpolated['targets']:
                osc.send_target_states_bulk(interpolated['targets'])
            if interpolated['dynamic_obstacles']:
                osc.send_dynamic_states_bulk(interpolated['dynamic_obstacles'])
        else:
            # Lockstep 모드는 보간 없이 최신 상태 전송
            if target_entities:
                target_states_to_send = []
                for tid, target_data in curr_state['targets'].items():
                    target_states_to_send.append((tid, target_data[0], target_data[1], target_data[2]))
                osc.send_target_states_bulk(target_states_to_send)
                
            if dynamic_obstacles:
                dynamic_states_to_send = []
                for o_id, obs_data in curr_state['dynamic_obstacles'].items():
                    dynamic_states_to_send.append((o_id, obs_data[0], obs_data[1]))
                osc.send_dynamic_states_bulk(dynamic_states_to_send)

        if args.lockstep and 'frame_id' in recv:
            osc.send_step_ack(recv['frame_id'])

        loop_dur = time.perf_counter() - loop_start
            
        log_loop_dur_sum += loop_dur
        log_phys_dur_sum += physics_dur_total
        log_count += 1
        
        if log_count >= 50:
            avg_loop = (log_loop_dur_sum / 50.0) * 1000.0
            avg_phys = (log_phys_dur_sum / 50.0) * 1000.0
            print(f" [STATS] Loop Avg: {avg_loop:.2f} ms | Physics Avg: {avg_phys:.2f} ms")
            
            # 최초 50프레임 평균 루프 시간 통계가 나오면, 언리얼에 해당 지연 배속 비율(TimeDilation)을 1회 전송합니다.
            if not hasattr(main, '_dilation_sent'):
                main._dilation_sent = True
                avg_loop_sec = log_loop_dur_sum / 50.0
                if avg_loop_sec > sim_dt:
                    dilation = sim_dt / avg_loop_sec
                else:
                    dilation = 1.0
                print(f"  [Pacing] [TimeDilation] 실측 Loop Avg({avg_loop:.2f}ms)에 기초하여 Unreal Dilation을 {dilation:.4f}배속으로 전송 및 자동 매칭합니다.")
                osc.client_cpp.send_message("/Genesis/Init/TimeDilation", [float(dilation)])
                
            log_loop_dur_sum = 0.0
            log_phys_dur_sum = 0.0
            log_count = 0

    osc.close()

def cli():
    """`python -m genesis_vehicle.server` 진입점 — main() 종료 시 임시 파일 청소 보장."""
    try:
        main()
    finally:
        # 분할된 env_builder의 created_temp_files 목록을 호출하여 청소
        if hasattr(genesis_env_builder, 'created_temp_files') and genesis_env_builder.created_temp_files:
            print(f"\n [Genesis] Cleaning up {len(genesis_env_builder.created_temp_files)} temporary preprocessed meshes...")
            for temp_file in genesis_env_builder.created_temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                        print(f"  └ Successfully removed: {os.path.basename(temp_file)}")
                    except Exception as e:
                        print(f"  └ Failed to remove temp file {temp_file}: {e}")
                        
        # 임시 생성된 차량 URDF 파일들 청소
        if hasattr(genesis_vehicle_builder, 'created_temp_urdfs') and genesis_vehicle_builder.created_temp_urdfs:
            print(f"\n [Genesis] Cleaning up {len(genesis_vehicle_builder.created_temp_urdfs)} temporary preprocessed URDFs...")
            for temp_urdf in genesis_vehicle_builder.created_temp_urdfs:
                if os.path.exists(temp_urdf):
                    try:
                        os.remove(temp_urdf)
                        print(f"  └ Successfully removed: {os.path.basename(temp_urdf)}")
                    except Exception as e:
                        print(f"  └ Failed to remove temp URDF {temp_urdf}: {e}")


if __name__ == "__main__":
    cli()
