import os
import trimesh
import numpy as np
import genesis as gs

# [Fix] CoACD/Complex 콜라이더의 Face 꼬임 방지를 위해 생성된 전처리 임시 파일 목록
created_temp_files = []

def make_double_sided_mesh(mesh, thickness=0.02):
    """
    오픈 메쉬의 모든 페이스를 복제하고 와인딩을 뒤집은 뒤 아래로 미세하게 이동하여,
    양면에서 백페이스 컬링 없이 완벽히 투명하지 않게 보이도록 하는 이중면 메쉬를 만듭니다.
    """
    v_orig = mesh.vertices.copy()
    f_orig = mesh.faces.copy()
    num_v = len(v_orig)
    
    # Z축 아래로 thickness만큼 내린 정점 생성
    v_back = v_orig.copy()
    v_back[:, 2] -= thickness
    
    new_vertices = np.vstack([v_orig, v_back])
    
    # 뒷면 face 생성 (정점 인덱스 오프셋 적용 및 와인딩 순서 반전 [c, b, a])
    f_back = f_orig + num_v
    f_back = f_back[:, [2, 1, 0]]
    
    # 앞면과 뒷면 결합
    all_faces = np.vstack([f_orig, f_back])
    
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=all_faces)
    new_mesh.remove_degenerate_faces()
    new_mesh.remove_duplicate_faces()
    
    return new_mesh

def build_obstacles(scene, init_data, ue_friction, ue_restitution, vis_mode, verbose=False):
    """
    언리얼 엔진으로부터 수신한 초기 장애물 리스트를 파싱하여
    물질 특성(마찰력, 반발력 등)과 충돌 기하(SDF/Box/Convex 등)를 자동 튜닝하여
    Genesis 물리 씬에 배치합니다.
    """
    global created_temp_files
    obstacles = []
    dynamic_obstacles = {}
    initial_dynamic_states = {}
    ue_driven_obstacle_ids = set()
    entities_to_set_mass = []

    if not init_data or not init_data.get('obstacles'):
        return obstacles, dynamic_obstacles, initial_dynamic_states, ue_driven_obstacle_ids, entities_to_set_mass

    print(f" [Genesis] [Env] 언리얼 엔진으로부터 {len(init_data['obstacles'])}개의 장애물을 동기화합니다...")
    
    for obs_id, obs_data in init_data['obstacles'].items():
        obs_type = obs_data.get('type', 1) 
        pos = obs_data['pos']
        quat = obs_data['quat']
        scale = obs_data['scale'] 
        
        obs_mass = obs_data.get('mass', 1.0)
        obs_friction = ue_friction if obs_data.get('friction', -1.0) < 0 else obs_data['friction']
        obs_restitution = ue_restitution if obs_data.get('restitution', -1.0) < 0 else obs_data['restitution']
        
        b_dynamic = obs_data.get('b_dynamic', 0)
        is_fixed = (b_dynamic == 0 or b_dynamic == 2)
        if b_dynamic == 2:
            ue_driven_obstacle_ids.add(obs_id)

        # [User Choice] 사용자 지정 태그 확인
        col_src = obs_data.get('collision_source', '')
        is_user_simple = "[User:Simple]" in col_src
        is_user_convex = "[User:Convex]" in col_src
        is_user_complex = "[User:Complex]" in col_src

        # 직관적인 물리 엔진 로딩 방식 이름으로 변환 (로그 출력용)
        sim_handling = "Primitive"
        if is_user_simple: sim_handling = "PrimitiveBox_Override"
        elif obs_type == 5:
            if "[Simple:Aggregate]" in col_src: sim_handling = "Exact_Mesh (AutoConvex / Fast)"
            elif "[Complex" in col_src or is_user_complex: sim_handling = "Exact_Mesh (Raw HighPoly / Lag-Warning)"
            else: sim_handling = "Decomposed_Convex (CoACD)"

        mesh_path_str = f" | Mesh: {os.path.basename(obs_data.get('mesh_path', ''))}" if obs_type == 5 else ""
        print(f"  └ [Obs {obs_id}] SimTag: [{sim_handling}] | UE: {col_src} | Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]{mesh_path_str}")
        
        if obs_mass <= 0:
            obs_mass = 1.0 
        
        # [NEW] Enforce Minimum Physical Volume to prevent tunneling
        size = (
            max(scale[0] * 1.0, 0.01), 
            max(scale[1] * 1.0, 0.01), 
            max(scale[2] * 1.0, 0.01)
        )

        if any(s <= 0.01 for s in size):
            if verbose:
                print(f" [Genesis] [Warning] Obstacle {obs_id} is extremely thin. Clamped to 1cm for stability.")

        morph = gs.morphs.Box(size=size, pos=pos, quat=quat, fixed=is_fixed)
        
        if is_user_simple:
            morph = gs.morphs.Box(size=size, pos=pos, quat=quat, fixed=is_fixed)
        elif obs_type == 2:
            morph = gs.morphs.Sphere(radius=size[0] / 2.0, pos=pos, quat=quat, fixed=is_fixed)
        elif obs_type == 3:
            morph = gs.morphs.Cylinder(radius=size[0] / 2.0, height=size[2], pos=pos, quat=quat, fixed=is_fixed)
        elif obs_type == 4:
            morph = gs.morphs.Box(size=size, pos=pos, quat=quat, fixed=is_fixed)
        elif obs_type == 5 or is_user_complex or is_user_convex:
            mesh_path = obs_data.get('mesh_path', '')
            
            # [Fix] 패키지 빌드 시 절대 경로가 다를 수 있으므로 자동 상대 경로 스캔 및 역추적을 수행합니다.
            if mesh_path and not os.path.exists(mesh_path):
                filename = os.path.basename(mesh_path)
                candidate_paths = [
                    # 파이썬 실행 디렉토리 기준 (Content 폴더 우대)
                    os.path.join(os.getcwd(), "Content", "GenesisCache", filename),
                    os.path.join(os.getcwd(), "Content", "GenesisAutoMeshes", filename),
                    os.path.join(os.getcwd(), "Saved", "GenesisCache", filename),
                    
                    # 소스코드 경로 기준
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Content", "GenesisCache", filename),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Saved", "GenesisCache", filename),
                    
                    # Unreal 빌드 아웃풋 예상 폴더들 (Content 폴더 우대)
                    os.path.join(os.getcwd(), "..", "Content", "GenesisCache", filename),
                    os.path.join(os.getcwd(), "Windows", "Content", "GenesisCache", filename),
                    
                    # 프로젝트명 포함 폴더
                    os.path.join(os.getcwd(), "Windows", "Genesis_Unreal", "Content", "GenesisCache", filename),
                    os.path.join(os.getcwd(), "WindowsNoEditor", "Genesis_Unreal", "Content", "GenesisCache", filename)
                ]
                for candidate in candidate_paths:
                    if os.path.exists(candidate):
                        print(f" [Genesis] [Path Fix] Remapped packaged mesh path to: {candidate}")
                        mesh_path = candidate
                        break
            
            if mesh_path and os.path.exists(mesh_path):
                do_decimate = False
                mesh_to_load = mesh_path
                is_road = "[Complex" in col_src or is_user_complex
                
                if is_road:
                    try:
                        import tempfile
                        print(f"    -> [Trimesh] Pre-processing complex road (Double-sided): {os.path.basename(mesh_path)}")
                        mesh = trimesh.load(mesh_path)
                        if isinstance(mesh, trimesh.Scene):
                            mesh = mesh.dump(concatenate=True)
                            
                        mesh = make_double_sided_mesh(mesh, thickness=0.001)
                        
                        temp_fd, temp_mesh_path = tempfile.mkstemp(suffix='.obj')
                        os.close(temp_fd)
                        mesh.export(temp_mesh_path)
                        created_temp_files.append(temp_mesh_path)
                        
                        mesh_to_load = temp_mesh_path
                    except Exception as e:
                        print(f" [Genesis] [Warning] Failed to preprocess road mesh with trimesh ({e}). Fallback to original mesh.")
                        mesh_to_load = mesh_path

                if "[Complex" in col_src or is_user_complex:
                    should_convexify = True
                    error_threshold = 0.01
                    coacd_opt = gs.options.CoacdOptions(threshold=0.03, extrude_margin=0.0, preprocess_mode="off", merge=False)
                    print(f"    -> [Optimize] Applying Fast & Zero-Margin CoACD for road (Threshold=0.03, Res=1000, ExtrudeMrg=0.0, Preprocess=off, Merge=off)...")
                else:
                    should_convexify = True
                    error_threshold = float('inf')
                    coacd_opt = None
                    print(f"    -> [Optimize] Loading exact Unreal Convex Hulls without CoACD merging.")

                morph = gs.morphs.Mesh(
                    file=mesh_to_load,
                    scale=size,
                    pos=pos,
                    quat=quat,
                    fixed=is_fixed,
                    align=False,
                    convexify=should_convexify,
                    decompose_object_error_threshold=error_threshold,
                    coacd_options=coacd_opt,
                    decimate=do_decimate
                )
            else:
                print(f" [Genesis] [WARNING] Mesh file not found: {mesh_path}. Fallback to Box.")
                morph = gs.morphs.Box(size=size, pos=pos, quat=quat, fixed=is_fixed)
        
        obs_colors = {
            1: (0.2, 0.4, 1.0, 0.5), # Standard Box: Blue
            2: (1.0, 0.6, 0.2, 0.5), # Sphere: Orange
            3: (0.2, 1.0, 0.6, 0.5), # Cylinder: Cyan
            5: (0.6, 0.2, 1.0, 0.4)  # Mesh (SDF): Purple
        }
        current_color = obs_colors.get(obs_type, (0.7, 0.7, 0.7, 0.5))
        
        mat = gs.materials.Rigid(
            friction=obs_friction,
            coup_restitution=obs_restitution,
            sdf_cell_size=10000.0
        )
        
        is_road_mesh = (obs_type == 5 and ("[Complex" in col_src or is_user_complex))
        
        obs_entity = scene.add_entity(
            morph,
            material=mat,
            surface=gs.surfaces.Rough(color=current_color, double_sided=is_road_mesh),
            vis_mode=vis_mode
        )
        obstacles.append(obs_entity)
        
        if b_dynamic in [1, 2]:
            dynamic_obstacles[obs_id] = obs_entity
            initial_dynamic_states[obs_id] = (np.array(pos, dtype=np.float32), np.array(quat, dtype=np.float32))

        if not is_fixed:
            entities_to_set_mass.append((obs_entity, obs_mass))

    return obstacles, dynamic_obstacles, initial_dynamic_states, ue_driven_obstacle_ids, entities_to_set_mass
