import os
import trimesh
import numpy as np
import genesis as gs

# [Fix] List of preprocessing temp files created to prevent face winding issues in CoACD/Complex colliders
created_temp_files = []

def make_double_sided_mesh(mesh, thickness=0.02):
    """
    Build a double-sided mesh from an open mesh by duplicating all faces,
    flipping their winding, and shifting them slightly downward so the mesh
    is visible from both sides without backface culling.
    """
    v_orig = mesh.vertices.copy()
    f_orig = mesh.faces.copy()
    num_v = len(v_orig)
    
    # Create vertices shifted down along Z by thickness
    v_back = v_orig.copy()
    v_back[:, 2] -= thickness
    
    new_vertices = np.vstack([v_orig, v_back])
    
    # Create back faces (apply vertex index offset and reverse winding order [c, b, a])
    f_back = f_orig + num_v
    f_back = f_back[:, [2, 1, 0]]
    
    # Combine front and back faces
    all_faces = np.vstack([f_orig, f_back])
    
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=all_faces)
    # trimesh >= 4 removed remove_degenerate_faces() / remove_duplicate_faces();
    # the replacement is update_faces() with a boolean face mask. Fall back to the
    # legacy methods on trimesh 3.x.
    if hasattr(new_mesh, "nondegenerate_faces"):
        new_mesh.update_faces(new_mesh.nondegenerate_faces())
        new_mesh.update_faces(new_mesh.unique_faces())
    else:
        new_mesh.remove_degenerate_faces()
        new_mesh.remove_duplicate_faces()

    return new_mesh


def _rotate_vec_by_quat(v, q):
    """Rotate vector ``v`` (3,) by quaternion ``q`` = (w, x, y, z) — Genesis
    convention. Uses the standard t = 2·(q_xyz × v); v' = v + w·t + q_xyz × t."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    qv = np.array([x, y, z], dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def mesh_to_primitive_box(mesh_path, pos, quat, scale, fixed):
    """Replace a mesh collider with its **local axis-aligned bounding box**,
    emitted as a ``gs.morphs.Box`` primitive.

    Why: Genesis collides meshes via a per-geom SDF that is processed EVERY step
    regardless of contact (measured ~0.6 ms/mesh on CPU, even with zero contact).
    A Box is collided analytically (``box_box_detection``) and costs ~0 when not
    touching — so hundreds/thousands of structures scale with *actual contacts*,
    not total count. The box rides the entity's ``quat``, so a building that is
    axis-aligned in its own local frame still gets a correctly oriented box
    (effectively an OBB) once the entity rotation is applied.

    Returns a ``gs.morphs.Box``. Raises on mesh-load failure (caller falls back).
    """
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    lo, hi = mesh.bounds                       # local AABB (3,), (3,)
    extents = np.asarray(hi, float) - np.asarray(lo, float)
    center = (np.asarray(hi, float) + np.asarray(lo, float)) * 0.5
    sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
    box_size = (max(extents[0] * sx, 0.01),
                max(extents[1] * sy, 0.01),
                max(extents[2] * sz, 0.01))
    # Local (scaled) center offset, rotated into world by the entity quat.
    off_local = np.array([center[0] * sx, center[1] * sy, center[2] * sz], float)
    off_world = _rotate_vec_by_quat(off_local, quat)
    box_pos = [pos[0] + off_world[0], pos[1] + off_world[1], pos[2] + off_world[2]]
    return gs.morphs.Box(size=box_size, pos=box_pos, quat=quat, fixed=fixed)

def build_obstacles(vs, init_data, ue_friction, ue_restitution, vis_mode,
                    verbose=False, road_raycast_only=False,
                    structures_as_primitive=False):
    """
    Parse the initial obstacle list received from Unreal Engine, auto-tune
    material properties (friction, restitution, etc.) and collision geometry
    (SDF/Box/Convex, etc.), and place them into the Genesis physics scene.

    With road_raycast_only=True, complex road/terrain meshes (obs_type==5 +
    [Complex]) are loaded as **KINEMATIC visual meshes**
    (material.Kinematic(use_visual_raycasting=True)). In the raycast-wheel
    model the chassis floats on suspension forces and the wheels follow the
    ground via raycasts, so the road only needs to "be hit by rays" and does
    not need to be a rigid collision body. Placing it in the Kinematic solver
    makes its raycast BVH maybe_static=True, so the **per-step rebuild is
    skipped** (even while vehicles move in the rigid solver). This path
    removes not only CoACD/narrow-phase but also the **raycaster's per-frame
    BVH rebuild**, saving tens to hundreds of ms per step on large maps.

    With structures_as_primitive=True, **all mesh colliders** (obs_type==5 /
    convex / complex) are replaced by their bounding box (``gs.morphs.Box``
    primitive). Meshes process a per-geom SDF every step (~0.6 ms/mesh even
    with zero contact), slowing down proportionally to structure count, while
    a Box collides analytically and costs ~0 when not touching. So even with
    hundreds/thousands of structures, only "the few actually in contact" cost
    anything. (A box is unsuitable for roads, so road_raycast_only takes
    precedence there; only the other mesh structures become boxes.)
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

        # [User Choice] Check user-specified tags
        col_src = obs_data.get('collision_source', '')
        is_user_simple = "[User:Simple]" in col_src
        is_user_convex = "[User:Convex]" in col_src
        is_user_complex = "[User:Complex]" in col_src

        # Convert to an intuitive physics-engine loading-mode name (for log output)
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
            
            # [Fix] Absolute paths may differ in packaged builds, so perform automatic relative-path scanning and backtracking.
            if mesh_path and not os.path.exists(mesh_path):
                filename = os.path.basename(mesh_path)
                candidate_paths = [
                    # Relative to the Python working directory (Content folder preferred)
                    os.path.join(os.getcwd(), "Content", "GenesisCache", filename),
                    os.path.join(os.getcwd(), "Content", "GenesisAutoMeshes", filename),
                    os.path.join(os.getcwd(), "Saved", "GenesisCache", filename),
                    
                    # Relative to the source code path
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Content", "GenesisCache", filename),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Saved", "GenesisCache", filename),
                    
                    # Expected Unreal build output folders (Content folder preferred)
                    os.path.join(os.getcwd(), "..", "Content", "GenesisCache", filename),
                    os.path.join(os.getcwd(), "Windows", "Content", "GenesisCache", filename),
                    
                    # Folders including the project name
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
                rc_only = road_raycast_only and is_road

                if rc_only:
                    # [Raycast-Only Road] No collision geometry → no CoACD, no
                    # chassis-vs-road narrow-phase. The mesh is added with a
                    # Kinematic + use_visual_raycasting material (set below) so the
                    # wheel raycaster hits it AND its BVH is static-skipped each
                    # step. Downward wheel rays hit the up-facing road surface;
                    # Surface(double_sided=...) handles rendering, so we skip the
                    # geometry-doubling preprocessing too (keeps the visual BVH small).
                    print(f"    -> [Raycast-Only] Road loaded as KINEMATIC visual mesh "
                          f"(no collision/CoACD, static-skipped BVH): {os.path.basename(mesh_path)}")
                    morph = gs.morphs.Mesh(
                        file=mesh_path,
                        scale=size,
                        pos=pos,
                        quat=quat,
                        fixed=is_fixed,
                        align=False,
                        collision=False,
                        visualization=True,
                        convexify=False,
                        decimate=False,
                    )
                elif structures_as_primitive:
                    # [Primitive Override] Replace the mesh collider with its
                    # bounding box → analytic collision, no per-geom SDF cost.
                    try:
                        morph = mesh_to_primitive_box(
                            mesh_path, pos, quat, size, is_fixed)
                        bs = morph.size
                        print(f"    -> [Primitive] Mesh collider replaced by BOX "
                              f"(size={bs[0]:.2f}x{bs[1]:.2f}x{bs[2]:.2f}, no SDF): "
                              f"{os.path.basename(mesh_path)}")
                    except Exception as e:
                        print(f" [Genesis] [Warning] mesh→box failed ({e}); "
                              f"fallback to Box(scale).")
                        morph = gs.morphs.Box(size=size, pos=pos, quat=quat, fixed=is_fixed)
                else:
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
        
        is_road_mesh = (obs_type == 5 and ("[Complex" in col_src or is_user_complex))
        surface = gs.surfaces.Rough(color=current_color, double_sided=is_road_mesh)
        mat = gs.materials.Rigid(friction=obs_friction,
                                 coup_restitution=obs_restitution, sdf_cell_size=10000.0)

        # Route through VehicleScene — it owns the single_scene-vs-dual_scene
        # distribution (collision body in main + a static/synced raycast mirror in
        # the raycast scene). The caller never touches a scene.
        #   b_dynamic 0 = static structure/road  -> add_static
        #   b_dynamic 1 = physics-dynamic         -> add_dynamic(physics=True)
        #   b_dynamic 2 = UE-driven (OSC set_pos) -> add_dynamic(physics=False)
        if b_dynamic == 0:
            if is_road_mesh and mesh_path and os.path.exists(mesh_path):
                # Detailed kinematic raycast surface (collision=False) the wheels
                # cast against — the true surface, not a convex bulge.
                rc_morph = gs.morphs.Mesh(file=mesh_path, scale=size, pos=pos, quat=quat,
                                          fixed=is_fixed, align=False, collision=False,
                                          visualization=True, convexify=False, decimate=False)
                if road_raycast_only:
                    # rco road: raycast surface ONLY — no chassis collision, no CoACD,
                    # no rigid-solver presence at all (v1.0.10 first-class API; the
                    # kinematic use_visual_raycasting body lives in the raycast scene
                    # with a build-once BVH). material/vis_mode are irrelevant for a
                    # kinematic surface, so they are not passed.
                    handle = vs.add_raycast_surface(rc_morph, surface=surface,
                                                    name=f"obs_{obs_id}")
                else:
                    # full road: convex CoACD collider (morph) + detailed raycast surface.
                    handle = vs.add_static(collision_morph=morph, wheel_raycast_morph=rc_morph,
                                           material=mat, surface=surface, vis_mode=vis_mode,
                                           name=f"obs_{obs_id}")
            else:
                handle = vs.add_static(morph=morph, material=mat, surface=surface,
                                       vis_mode=vis_mode, name=f"obs_{obs_id}")
            obs_entity = handle.entity_main if handle.entity_main is not None else handle.entity_raycast
        else:
            # wheel_raycast: in dual_scene a dynamic body is invisible to the wheel
            # rays unless it gets a synced raycast mirror — without it the wheels
            # would clip through moving ramps/platforms that single_scene handled
            # implicitly (rays hit rigid colliders there, so the flag is a no-op
            # and is skipped to avoid the SDK's single-scene warning per obstacle).
            handle = vs.add_dynamic(morph, physics=(b_dynamic == 1),
                                    wheel_raycast=vs.is_dual_scene,
                                    material=mat, surface=surface, vis_mode=vis_mode,
                                    mass=(obs_mass if b_dynamic == 1 else None),
                                    name=f"obs_{obs_id}")
            obs_entity = handle.entity_main

        obstacles.append(obs_entity)

        if b_dynamic in [1, 2]:
            dynamic_obstacles[obs_id] = obs_entity
            initial_dynamic_states[obs_id] = (np.array(pos, dtype=np.float32), np.array(quat, dtype=np.float32))

    # Obstacle masses are applied by VehicleScene (add_dynamic mass=); the
    # returned list stays empty for back-compat with the call sites.
    return obstacles, dynamic_obstacles, initial_dynamic_states, ue_driven_obstacle_ids, entities_to_set_mass
