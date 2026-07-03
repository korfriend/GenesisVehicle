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


class _BatchPoseReader:
    """One batched solver read for many entities' base poses (v1.0.13).

    ``capture_state`` was calling ``entity.get_pos()``/``get_quat()`` once per
    target AND per dynamic obstacle — 2·K engine entries per capture (and the
    server captures twice per physics step). All those entities live in the
    same rigid solver, so ONE ``get_links_pos``/``get_links_quat`` over their
    base links returns the same env-0 user-frame poses (identical semantics:
    ``entity.get_pos`` is ``solver.get_links_pos(base_link_idx,
    relative=True)``). Build once after ``vs.build()``; ``read()`` per capture.
    """

    def __init__(self, entities):
        self.entities = [e for e in entities]
        self.idx = [e.base_link_idx for e in self.entities]
        self.solver = self.entities[0]._solver if self.entities else None

    def read(self):
        """→ ``(pos, quat)`` numpy ``(K, 3)`` / ``(K, 4)``, env 0 — matching
        capture_state's previous per-entity ``get_pos()[0]`` semantics."""
        if not self.entities:
            return np.zeros((0, 3), np.float32), np.zeros((0, 4), np.float32)
        pos = self.solver.get_links_pos(self.idx, relative=True)
        quat = self.solver.get_links_quat(self.idx, relative=True)
        pos = pos.reshape(-1, len(self.idx), 3)[0]
        quat = quat.reshape(-1, len(self.idx), 4)[0]
        return pos.cpu().numpy(), quat.cpu().numpy()


def capture_state(target_entities, dynamic_obstacles, is_urdf_active, controllers, ue_driven_obstacle_ids, accumulated_wheel_angles=None, sim_dt=0.02, update_angles=False, mvp=None, readers=None):
    """
    Captures positions and rotations for all target entities (including wheels if active)
    and dynamic obstacles in Genesis physics engine.

    ``mvp``: the scene's batched ``MultiVehiclePhysics`` (``vs.physics``) when
    solver="batched". Per-vehicle ``Vehicle.wheel_visual_transforms`` recomputes
    the WHOLE K-vehicle batch and slices one vehicle, so calling it per target
    is O(K²); with ``mvp`` the batch is computed ONCE here and sliced per slot.

    ``readers``: optional ``(_BatchPoseReader(targets), _BatchPoseReader(
    obstacles))`` — replaces the 2·K per-entity ``get_pos/get_quat`` engine
    calls with ONE batched solver read each (v1.0.13). Entity order must match
    the dicts' iteration order (build the readers from the same dicts).
    """
    state = {
        'targets': {},
        'dynamic_obstacles': {}
    }
    t_reader, o_reader = readers if readers is not None else (None, None)

    # [PERF] One batched closed-form wheel-pose compute for all K vehicles
    # (flat list in Vehicle._slot order), instead of K per-vehicle calls.
    wheel_tf_flat = None
    if is_urdf_active and controllers and mvp is not None:
        try:
            wheel_tf_flat = mvp.wheel_visual_transforms("world")
        except Exception:
            wheel_tf_flat = None

    # 1. Target entities (Chassis & Wheels)
    if target_entities:
        tids = list(target_entities.keys())
        if t_reader is not None:
            pos_batch, quat_batch = t_reader.read()      # ONE solver read
        else:
            pos_tensors  = [ent.get_pos()  for ent in target_entities.values()]
            quat_tensors = [ent.get_quat() for ent in target_entities.values()]

            if hasattr(pos_tensors[0], 'cpu'):
                pos_batch  = torch.stack(pos_tensors).cpu().numpy()
                quat_batch = torch.stack(quat_tensors).cpu().numpy()
            else:
                pos_batch  = np.array(pos_tensors)
                quat_batch = np.array(quat_tensors)

        for i, tid in enumerate(tids):
            # NB: rows are views into arrays freshly created THIS capture
            # (reader output / .cpu().numpy()), never mutated afterwards —
            # per-row .copy() removed in v1.0.13.
            p = pos_batch[i]
            q = quat_batch[i]
            if p.ndim > 1: p = p[0]
            if q.ndim > 1: q = q[0]

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
                        slot = getattr(ctrl, '_slot', -1)
                        if wheel_tf_flat is not None and 0 <= slot < len(wheel_tf_flat):
                            wp_all, wq_all = wheel_tf_flat[slot]   # pre-computed batch
                        else:
                            wp_all, wq_all = ctrl.wheel_visual_transforms("world")
                        if hasattr(wp_all, 'cpu'):
                            wp_all = wp_all.cpu().numpy(); wq_all = wq_all.cpu().numpy()
                        wp_all = wp_all[0]; wq_all = wq_all[0]   # env 0 (single-env)
                        for j in range(wp_all.shape[0]):
                            wheels_states.append((wp_all[j], wq_all[j], 0.0))
                    except Exception:
                        pass
            state['targets'][tid] = (p, q, wheels_states)
            
    # 2. Dynamic obstacles
    if dynamic_obstacles:
        oids = list(dynamic_obstacles.keys())
        if o_reader is not None:
            opos_batch, oquat_batch = o_reader.read()    # ONE solver read
        else:
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

            state['dynamic_obstacles'][o_id] = (p, q)
            
    return state


def _slerp_batch(q0, q1, t):
    """Vectorized quaternion slerp — ``q0``/``q1`` are ``(N, 4)`` ``[w,x,y,z]``,
    ``t`` a scalar in [0, 1]. Semantics match the scalar ``slerp`` above
    (normalize inputs, shortest path, nlerp fallback for near-parallel pairs);
    one numpy pass replaces N python calls.
    """
    q0 = np.asarray(q0, dtype=np.float32).copy()
    q1 = np.asarray(q1, dtype=np.float32).copy()
    n0 = np.linalg.norm(q0, axis=1, keepdims=True)
    n1 = np.linalg.norm(q1, axis=1, keepdims=True)
    np.divide(q0, n0, out=q0, where=n0 > 0)
    np.divide(q1, n1, out=q1, where=n1 > 0)

    dot = np.sum(q0 * q1, axis=1)
    q1 = np.where((dot < 0.0)[:, None], -q1, q1)          # shortest path
    dot = np.abs(dot)

    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin0 = np.sin(theta0)
    spherical = (dot <= 0.9995) & (sin0 > 1e-6)
    sin0_safe = np.where(sin0 > 1e-6, sin0, 1.0)
    # spherical weights; near-parallel falls back to linear (nlerp) weights
    s0 = np.where(spherical, np.sin((1.0 - t) * theta0) / sin0_safe, 1.0 - t)
    s1 = np.where(spherical, np.sin(t * theta0) / sin0_safe, t)

    out = s0[:, None] * q0 + s1[:, None] * q1
    n = np.linalg.norm(out, axis=1, keepdims=True)
    np.divide(out, n, out=out, where=n > 0)
    return out


def lerp_state(prev, curr, a):
    """
    Interpolates states between prev and curr with factor a in [0, 1).

    v1.0.10: vectorized — all chassis/wheel/obstacle quats are gathered into
    flat arrays and slerped in ONE numpy pass (`_slerp_batch`) instead of one
    python `slerp` call per quaternion. The old per-wheel loop cost the server
    ~0.4 ms/vehicle/loop (30 tanks x 10 wheels = 300 python slerps -> ~14 ms,
    rivaling the physics step itself). Output format is unchanged.
    """
    interpolated = {
        'targets': [],
        'dynamic_obstacles': []
    }

    # 1. Target entities — gather matched pairs, batch-interpolate, scatter.
    tids, P0, P1, Q0, Q1 = [], [], [], [], []
    wheel_counts = []
    WP0, WP1, WQ0, WQ1, WA0, WA1 = [], [], [], [], [], []
    for tid, curr_data in curr['targets'].items():
        if tid not in prev['targets']:
            interpolated['targets'].append((tid, curr_data[0], curr_data[1], curr_data[2]))
            continue
        prev_p, prev_q, prev_wheels = prev['targets'][tid]
        curr_p, curr_q, curr_wheels = curr_data
        tids.append(tid)
        P0.append(prev_p); P1.append(curr_p)
        Q0.append(prev_q); Q1.append(curr_q)
        m = min(len(prev_wheels), len(curr_wheels))
        wheel_counts.append(m)
        for j in range(m):
            pw = prev_wheels[j]; cw = curr_wheels[j]
            WP0.append(pw[0]); WQ0.append(pw[1])
            WA0.append(pw[2] if len(pw) > 2 else 0.0)
            WP1.append(cw[0]); WQ1.append(cw[1])
            WA1.append(cw[2] if len(cw) > 2 else 0.0)

    if tids:
        P0a = np.asarray(P0, dtype=np.float32); P1a = np.asarray(P1, dtype=np.float32)
        iP = P0a * (1.0 - a) + P1a * a
        iQ = _slerp_batch(Q0, Q1, a)
        if WP0:
            WP0a = np.asarray(WP0, dtype=np.float32); WP1a = np.asarray(WP1, dtype=np.float32)
            iWP = WP0a * (1.0 - a) + WP1a * a
            iWQ = _slerp_batch(WQ0, WQ1, a)
            WA0a = np.asarray(WA0, dtype=np.float32); WA1a = np.asarray(WA1, dtype=np.float32)
            # Boundary wrap-around LERP for wheel angle
            diff = (WA1a - WA0a + np.pi) % (2.0 * np.pi) - np.pi
            iWA = (WA0a + diff * a) % (2.0 * np.pi)
        off = 0
        for i, tid in enumerate(tids):
            m = wheel_counts[i]
            wheels = [(iWP[off + j], iWQ[off + j], float(iWA[off + j])) for j in range(m)]
            off += m
            interpolated['targets'].append((tid, iP[i], iQ[i], wheels))

    # 2. Dynamic obstacles — same batched treatment.
    oids, OP0, OP1, OQ0, OQ1 = [], [], [], [], []
    for o_id, curr_data in curr['dynamic_obstacles'].items():
        if o_id not in prev['dynamic_obstacles']:
            interpolated['dynamic_obstacles'].append((o_id, curr_data[0], curr_data[1]))
            continue
        prev_p, prev_q = prev['dynamic_obstacles'][o_id]
        oids.append(o_id)
        OP0.append(prev_p); OP1.append(curr_data[0])
        OQ0.append(prev_q); OQ1.append(curr_data[1])
    if oids:
        OP0a = np.asarray(OP0, dtype=np.float32); OP1a = np.asarray(OP1, dtype=np.float32)
        iOP = OP0a * (1.0 - a) + OP1a * a
        iOQ = _slerp_batch(OQ0, OQ1, a)
        for i, o_id in enumerate(oids):
            interpolated['dynamic_obstacles'].append((o_id, iOP[i], iOQ[i]))

    return interpolated


def main():
    parser = argparse.ArgumentParser(description="Genesis Ball Kick Sample with Unreal Engine OSC Bridge")
    parser.add_argument("--send_ip", type=str, default="127.0.0.1", help="Target IP address for Unreal Engine (default: 127.0.0.1)")
    parser.add_argument("--recv_port", type=int, default=7001, help="Port to receive data from UE (default: 7001)")
    parser.add_argument("--send_port", type=int, default=7002, help="Port to send State data to UE (default: 7002)")
    parser.add_argument("--send_port_obs", type=int, default=7004, help="Port to send Observation data to UE (default: 7004)")
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
    parser.add_argument("--single-scene", action="store_true",
                        help="(per-entity mode) legacy single-scene raycast: the wheel rays hit "
                             "the rigid collision geoms of the ONE main scene directly (the "
                             "pre-v1.0.12 default). Cheaper to build, but the raycast BVH re-fits "
                             "every step and wheels ride convex (CoACD) road colliders instead of "
                             "the exact mesh surface. Incompatible with --road-raycast-only "
                             "(rco roads only exist in the dual raycast scene).")
    parser.add_argument("--structures-as-primitive", action="store_true",
                        help="Replace every MESH collider (obstacles/structures) with its "
                             "bounding BOX. Mesh colliders carry a per-geom SDF processed each "
                             "step even with zero contact (~0.6ms/mesh on CPU → 100s of "
                             "structures = 100s of ms). Box collision is analytic (~0 when not "
                             "touching), so cost scales with actual contacts, not structure "
                             "count. Use with --road-raycast-only (roads stay raycast surfaces).")
    parser.add_argument("--multi-env", action="store_true",
                        help="L3 batched mode: N identical, non-interacting vehicles as n_envs=N "
                             "(one vehicle entity, batched). Requires all targets to share one URDF.")
    parser.add_argument("--gpu", action="store_true",
                        help="Opt into the GPU backend (default: CPU in BOTH modes). GPU is "
                             "kernel-launch bound at small batch sizes, so it only pays off "
                             "in --multi-env (L3) mode with hundreds of envs; per-entity (L2) "
                             "and small fleets (n_envs<~100) are faster on CPU (measured: 30 "
                             "tanks L3 CPU 8.4 vs GPU ~19 ms/step).")
    parser.add_argument("--max-catchup-steps", type=int, default=None,
                        help="Max physics steps per loop when behind real-time "
                             "(default: max(5, 0.1/dt)). The cap does NOT speed "
                             "anything up — when a step exceeds the dt budget it "
                             "trades catch-up bursts (jerky pacing) for a steady "
                             "slow-motion: 1 = one step per loop, smoothest "
                             "degradation. Irrelevant once a step fits the budget.")
    args = parser.parse_args()

    if args.single_scene and args.road_raycast_only:
        parser.error("--single-scene is incompatible with --road-raycast-only: "
                     "an rco road is a kinematic raycast surface, which only the "
                     "dual raycast scene can host.")

    # [L3] 멀티-env 배칭 모드 — 동일 URDF 다수 차량 전용 경로로 분기
    if args.multi_env:
        print(" [Genesis] [MODE] === MULTI-ENV (L3 batched) === "
              "(1 vehicle entity x n_envs, non-interacting, CPU default — --gpu to opt in)")
        if args.single_scene:
            print(" [Genesis] [WARN] --single-scene is a per-entity-mode flag; "
                  "multi-env (L3) is always dual_scene — ignoring it.")
        from .l3_runtime import run_l3
        run_l3(args)
        return

    # 백엔드 기본 CPU + 비-락스텝(OSC Pacing) 처리
    # (per-entity 모드에선 CPU 가 GPU 보다 빠름 — n_envs=1 다중 엔티티는 커널 런치 바운드)
    args.cpu = not args.gpu
    args.lockstep = False
    print(" [Genesis] [MODE] === PER-ENTITY (L2) === "
          "(K interacting vehicles, n_envs=1, CPU default; use --multi-env for L3)")

    # 1. 제네시스 물리 엔진 초기화 및 백엔드 결정
    if args.cpu:
        backend = gs.cpu
    elif torch.cuda.is_available():
        backend = gs.gpu
        print(" [Genesis] [Init] backend = GPU (--gpu; NB: per-entity mode is "
              "usually FASTER on CPU — kernel-launch bound at n_envs=1)")
    else:
        backend = gs.cpu
        args.cpu = True
        print(" [Genesis] [WARN] --gpu requested but CUDA is unavailable — falling back to CPU.")

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
    
    # 4. Genesis Scene Setup — VehicleScene for unified vehicle handling.
    # Per-entity mode is interacting vehicles at n_envs=1 on CPU. v1.0.12:
    # dual_scene is the DEFAULT (matching the SDK default and L3) — statics get a
    # kinematic raycast mirror whose BVH is static (no per-step re-fit), wheels
    # ride the exact mesh surface, and --road-raycast-only composes on top (it
    # additionally drops the main-scene road collider). Dynamic obstacles keep
    # their pre-dual semantics via wheel_raycast mirrors (env_builder). The
    # legacy one-scene behavior (rays hit the rigid colliders themselves) stays
    # available as --single-scene; rco is rejected there at arg-parse time.
    # All geometry is registered via vs.add_* (no raw scene access); build() /
    # step() and sim reads/tweaks route through vs accessors. Genesis is already
    # initialized above, so init_genesis=False.
    vs = VehicleScene(
        n_envs=1, dt=ue_dt,
        raycast_mode="single_scene" if args.single_scene else "dual_scene",
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
    # [PROFILE] 계측 전 2스텝 예열: 첫 스텝에 taichi/torch 커널 JIT 컴파일 비용이
    # 몰려 구간별 수치가 크게 과대측정된다 (실측: GPU SDK compute 100ms(PROFILE)
    # vs 23ms(steady)). 예열 후 5스텝만 계측한다.
    for _ in range(2):
        vs.step()
        if not args.cpu and torch.cuda.is_available():
            torch.cuda.synchronize()

    # [PROFILE] 워밍업 5스텝 동안 스텝 내부 구간별 시간을 실측해 1회 출력 —
    # 느릴 때 원인(raycast/proxy vs SDK compute vs genesis solver)을 현장에서 특정.
    _prof = {'ray': 0.0, 'sdk': 0.0, 'solver': 0.0}
    _o_md = vs._measure_distances
    def _p_md():
        t0 = time.perf_counter(); r = _o_md()
        _prof['ray'] += time.perf_counter() - t0; return r
    vs._measure_distances = _p_md
    _o_ph = vs.physics.step if vs.physics is not None else None
    if _o_ph is not None:
        def _p_ph(*a, **kw):
            t0 = time.perf_counter(); r = _o_ph(*a, **kw)
            _prof['sdk'] += time.perf_counter() - t0; return r
        vs.physics.step = _p_ph
    _o_sc = vs._main_scene.step
    def _p_sc(*a, **kw):
        t0 = time.perf_counter(); r = _o_sc(*a, **kw)
        _prof['solver'] += time.perf_counter() - t0; return r
    vs._main_scene.step = _p_sc

    warmup_starts = time.perf_counter()
    for _ in range(5):
        vs.step()
        if not args.cpu and torch.cuda.is_available():
            torch.cuda.synchronize()
    warmup_ends = time.perf_counter()
    avg_step_time = (warmup_ends - warmup_starts) / 5.0

    # 계측 해제 (본 루프는 무계측 원상 복구)
    vs._measure_distances = _o_md
    if _o_ph is not None:
        vs.physics.step = _o_ph
    vs._main_scene.step = _o_sc

    _rest = avg_step_time * 1e3 - (_prof['ray'] + _prof['sdk'] + _prof['solver']) / 5.0 * 1e3
    print(f"  - 실측된 1스텝 평균 연산 속도: {avg_step_time * 1000.0:.2f} ms")
    print(f"  - [PROFILE] 스텝 구간별: raycast/proxy {_prof['ray']/5*1e3:.2f} ms | "
          f"SDK compute {_prof['sdk']/5*1e3:.2f} ms | "
          f"genesis solver {_prof['solver']/5*1e3:.2f} ms | 기타 {_rest:.2f} ms")
    
    # [CRITICAL FIX] 물리적 시간 흐름 및 동역학 일관성(Determinism)을 위해 sim_dt는 항상 고정 고수합니다.
    sim_dt = ue_dt
    vs.sim_options.dt = sim_dt
    print(f"  [OK] [Determinism] 물리 해상도(sim_dt)가 표준 {sim_dt * 1000.0:.1f}ms ({1.0/sim_dt:.1f}Hz)로 설정되었습니다.")
    print("="*50 + "\n")
    
    # 언리얼 엔진에 고정 동기 주기를 확인 전송합니다.
    print(f" [Pacing] [Auto-Pacing] 언리얼 엔진으로 동조 주기({sim_dt * 1000.0:.1f}ms)를 전송합니다...")
    osc.client_cpp.send_message("/Genesis/Init/Pacing", [float(sim_dt)])
        
    SIM_DT = sim_dt
    # catch-up 상한: 기본 max(5, 0.1/dt). --max-catchup-steps 로 재정의 가능 —
    # 1 이면 루프당 1스텝(버스트 없는 균일 슬로모션), 값이 커질수록 실시간
    # 복귀를 더 공격적으로 시도(밀린 만큼 몰아 돌려 프레임 간격이 출렁).
    if args.max_catchup_steps is not None:
        MAX_CATCHUP_STEPS = max(1, int(args.max_catchup_steps))
        print(f" [Pacing] [Catch-up] MAX_CATCHUP_STEPS 재정의: {MAX_CATCHUP_STEPS} (--max-catchup-steps)")
    else:
        MAX_CATCHUP_STEPS = max(5, int(0.1 / sim_dt))
    accumulator = 0.0
    last_time = time.perf_counter()
    last_slow_motion_warn_time = 0.0

    accumulated_wheel_angles = {}
    # [PERF] 배치 포즈 리더 (v1.0.13): capture_state 의 타깃/장애물 chassis 포즈를
    # per-entity get_pos/get_quat 대신 solver 일괄 read 1회씩으로. dict 순서와
    # 리더의 엔티티 순서가 일치해야 하므로 같은 dict 로 생성.
    try:
        _readers = (_BatchPoseReader(target_entities.values()),
                    _BatchPoseReader(dynamic_obstacles.values()))
    except Exception:
        _readers = None
    prev_state = capture_state(target_entities, dynamic_obstacles, is_urdf_active, controllers, ue_driven_obstacle_ids, accumulated_wheel_angles, sim_dt, False, mvp=vs.physics, readers=_readers)
    curr_state = prev_state

    step_count = 0
    last_urdf_inputs = {}
    last_frame_id = 0
    last_printed_inputs = {}
    last_step_time = None
    
    log_loop_dur_sum = 0.0
    log_phys_dur_sum = 0.0
    log_step_sum = 0
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
                prev_state = capture_state(target_entities, dynamic_obstacles, is_urdf_active, controllers, ue_driven_obstacle_ids, accumulated_wheel_angles, sim_dt, False, mvp=vs.physics, readers=_readers)
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
            curr_state = capture_state(target_entities, dynamic_obstacles, is_urdf_active, controllers, ue_driven_obstacle_ids, accumulated_wheel_angles, sim_dt, False, mvp=vs.physics, readers=_readers)

        catchup_steps = 0
        physics_dur_total = 0.0

        if args.lockstep:
            # Lockstep 모드는 고정 1스텝씩 가동
            steps_limit = 1
        else:
            steps_limit = MAX_CATCHUP_STEPS

        # Catch-up Multi-Step Loop (물리 20ms 고정 단위 소비)
        while (args.lockstep and steps_limit > 0) or (not args.lockstep and accumulator >= SIM_DT and catchup_steps < MAX_CATCHUP_STEPS):
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
            curr_state = capture_state(target_entities, dynamic_obstacles, is_urdf_active, controllers, ue_driven_obstacle_ids, accumulated_wheel_angles, sim_dt, True, mvp=vs.physics, readers=_readers)
            
            if not args.lockstep:
                accumulator -= SIM_DT
            else:
                steps_limit -= 1
                
            catchup_steps += 1
            step_count += 1






        # 데스 스파이럴 방지: 따라잡지 못하면 어큐뮬레이터 탕감 및 슬로우 모션 경고 (5초 간격으로 스로틀링 출력)
        if not args.lockstep and catchup_steps == MAX_CATCHUP_STEPS and accumulator >= SIM_DT:
            current_warn_time = time.perf_counter()
            if current_warn_time - last_slow_motion_warn_time >= 5.0:
                sim_ratio = (MAX_CATCHUP_STEPS * SIM_DT) / frame_time if frame_time > 0 else 1.0
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
        log_step_sum += catchup_steps
        log_count += 1

        if log_count >= 50:
            avg_loop = (log_loop_dur_sum / 50.0) * 1000.0
            avg_phys = (log_phys_dur_sum / 50.0) * 1000.0
            # Physics Avg 는 루프당 catch-up 스텝의 '합' — 스텝당 값과 혼동을
            # 막기 위해 steps/loop 와 per-step 을 함께 표기 (v1.0.7).
            steps_per_loop = log_step_sum / 50.0
            per_step = (log_phys_dur_sum / max(log_step_sum, 1)) * 1000.0
            print(f" [STATS] [per-entity] Loop Avg: {avg_loop:.2f} ms | "
                  f"Physics Avg: {avg_phys:.2f} ms "
                  f"({steps_per_loop:.1f} steps/loop, {per_step:.2f} ms/step)")
            
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
            log_step_sum = 0
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
