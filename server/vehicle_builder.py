"""Vehicle builder (genesis_vehicle.server) — origin: genesis_unreal_plugin/genesis_vehicle_builder.py.

Changes made during migration:
  - cfg construction logic extracted into ``build_cfg()`` — the L2 path
    (build_vehicle, per-entity) and the L3 multi-env path (l3_runtime) share
    the same vehicle configuration.
  - Temporary URDF generation with wheel colliders removed extracted into
    ``strip_wheel_collisions()``.
  - [FIX] Steering range key mismatch: UE ``FGenesisVehicleMapping.SteerScale``
    serializes to JSON as ``steerScale``, but the old code only read
    ``maxSteerRad``, so the UE setting was always ignored and the preset
    default (0.7 rad) applied. Now steerScale is also recognized.
"""
import os
import xml.etree.ElementTree as ET
import tempfile
import math
import numpy as np
import genesis as gs
import torch

from genesis_vehicle import (
    VehicleConfig, parse_urdf,
    AWD, PartialAckermann, Independent, PacejkaAnisotropic,
    RollingResistance, LowSpeedRegularizer
)

created_temp_urdfs = []


def _mapping_steer_rad(mapping, default=None):
    """Read the max steering angle (rad) from the mapping. Recognizes all
    UE-side field-name variants: maxSteerRad / MaxSteerRad / steerScale / SteerScale."""
    for key in ("maxSteerRad", "MaxSteerRad", "steerScale", "SteerScale"):
        if key in mapping:
            return float(mapping[key])
    return default


def apply_monkey_patches(rigid_solver):
    """
    Monkey patch that converts GenesisVehicle's 3D tensor (batched) force
    inputs to the current Genesis engine's 2D tensor requirement (avoids
    modifying the SDK source).
    """
    orig_apply_force = rigid_solver.apply_links_external_force
    def patched_apply_force(force, links_idx=None, envs_idx=None, **kwargs):
        if isinstance(force, torch.Tensor) and force.dim() == 3:
            if force.shape[1] == 1:
                force = force.squeeze(1)
            else:
                force = force.reshape(-1, force.shape[-1])
        return orig_apply_force(force, links_idx, envs_idx, **kwargs)
    rigid_solver.apply_links_external_force = patched_apply_force

    orig_apply_torque = rigid_solver.apply_links_external_torque
    def patched_apply_torque(torque, links_idx=None, envs_idx=None, **kwargs):
        if isinstance(torque, torch.Tensor) and torque.dim() == 3:
            if torque.shape[1] == 1:
                torque = torque.squeeze(1)
            else:
                torque = torque.reshape(-1, torque.shape[-1])
        return orig_apply_torque(torque, links_idx, envs_idx, **kwargs)
    rigid_solver.apply_links_external_torque = patched_apply_torque

    orig_set_dofs_pos = rigid_solver.set_dofs_position
    def patched_set_dofs_pos(position, dofs_idx=None, envs_idx=None, **kwargs):
        if isinstance(position, torch.Tensor) and position.dim() == 2:
            if position.shape[0] == 1:
                position = position.squeeze(0)
            else:
                position = position.reshape(-1)
        return orig_set_dofs_pos(position, dofs_idx, envs_idx, **kwargs)
    rigid_solver.set_dofs_position = patched_set_dofs_pos

    orig_control_dofs_pos = rigid_solver.control_dofs_position
    def patched_control_dofs_pos(position, dofs_idx=None, envs_idx=None, **kwargs):
        if isinstance(position, torch.Tensor) and position.dim() == 2:
            if position.shape[0] == 1:
                position = position.squeeze(0)
            else:
                position = position.reshape(-1)
        return orig_control_dofs_pos(position, dofs_idx, envs_idx, **kwargs)
    rigid_solver.control_dofs_position = patched_control_dofs_pos


def is_wheel_match_fuzzy(override_wheel_name, wheel_config):
    """
    Checks if an override wheel name matches a WheelConfig object,
    supporting exact matches and position-based abbreviation fuzzy matches.
    """
    o_name = override_wheel_name.lower()
    w_spin = wheel_config.spin_joint_name.lower() if wheel_config.spin_joint_name else ""
    w_node_name = wheel_config.name.lower() if wheel_config.name else ""

    # 1. Exact matches
    if o_name == w_spin or o_name == w_node_name:
        return True

    # 2. Extract and compare positions
    def get_pos(name):
        n = name.lower()
        if 'fl' in n or ('front' in n and 'left' in n):
            return 'fl'
        if 'fr' in n or ('front' in n and 'right' in n):
            return 'fr'
        if 'rl' in n or 'bl' in n or ('rear' in n and 'left' in n) or ('back' in n and 'left' in n):
            return 'rl'
        if 'rr' in n or 'br' in n or ('rear' in n and 'right' in n) or ('back' in n and 'right' in n):
            return 'rr'
        return None

    o_pos = get_pos(o_name)
    if o_pos:
        # Check if either the spin joint name or node name matches this position
        if get_pos(w_spin) == o_pos or get_pos(w_node_name) == o_pos:
            return True

    # 3. Substring matching fallback
    if o_name and (o_name in w_spin or w_spin in o_name or o_name in w_node_name or w_node_name in o_name):
        return True

    return False


def strip_wheel_collisions(urdf_path):
    """Create a temporary URDF with only the wheel colliders removed, without
    modifying the original URDF, and return its path.

    Ray-wheel physics does not use wheel colliders (raycast handles ground
    contact), so leaving them in creates double contact with the floor. The
    temp file is created in the same directory as the original to preserve
    relative mesh paths, and is registered in created_temp_urdfs for cleanup
    on exit.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall('link'):
        if 'wheel' in link.get('name', ''):
            for col in link.findall('collision'):
                link.remove(col)

    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    temp_fd, temp_path = tempfile.mkstemp(suffix='.urdf', dir=urdf_dir)
    os.close(temp_fd)
    tree.write(temp_path)
    created_temp_urdfs.append(temp_path)
    return temp_path


# [Batch] Shared cfg cache — VehicleScene's batched solver groups kinds by
# cfg "object identity" (multi_vehicle.group_vehicles_by_cfg). If the L2
# server creates a new cfg per target, K vehicles split into K kinds × 1
# vehicle, defeating the batched pipeline and multiplying SDK compute by K
# (measured: 10 tanks CPU 37.8ms vs 1 kind 2.8ms). For the same
# (urdf, mapping, friction), return the same cfg object so they group into
# one kind. target_id is excluded from the key (log-only argument, no effect
# on the result).
_cfg_cache: dict = {}


def build_cfg(urdf_path, mapping, t_fric, target_id=0):
    """Build and return a VehicleConfig from the URDF + UE mapping JSON.

    Single source of vehicle configuration shared by the L2 path
    (build_vehicle, per-entity) and the L3 multi-env path (l3_runtime). Does
    not include entity creation/Raycaster/VehiclePhysics. Calls with the same
    (urdf_path, mapping, t_fric) return the **same cfg object** (batched kind
    sharing — see the _cfg_cache comment above).
    """
    _key = (os.path.abspath(urdf_path), float(t_fric),
            repr(sorted(mapping.items())) if isinstance(mapping, dict) else repr(mapping))
    _cached = _cfg_cache.get(_key)
    if _cached is not None:
        print(f" [Genesis] [Batch] Vehicle {target_id}: reusing shared cfg "
              f"(same URDF/mapping/friction → one batched kind)")
        return _cached

    # 1. Parse URDF to extract wheel information
    urdf_parsed = parse_urdf(urdf_path)
    wheels = urdf_parsed.wheels

    drive_type_val = mapping.get('driveType', mapping.get('DriveType', 0))
    # EGenesisDriveType: 0 = Ackermann, 1 = Truck, 2 = SkidSteer, 3 = Manual
    is_skid = (drive_type_val == 2)
    is_truck = (drive_type_val == 1)

    # If SkidSteer with 10 wheels, use tank preset
    if is_skid and len(wheels) == 10:
        print(f" [Genesis] [Preset] Loading tank_10w_skid_belt preset for Vehicle {target_id}")
        from genesis_vehicle.presets import tank_10w_skid_belt
        cfg = tank_10w_skid_belt(urdf_path, n_envs=1)
    # If Truck with 6 wheels, use truck preset
    elif is_truck and len(wheels) == 6:
        print(f" [Genesis] [Preset] Loading truck_6w_partial_ackermann preset for Vehicle {target_id}")
        from genesis_vehicle.presets import truck_6w_partial_ackermann
        cfg = truck_6w_partial_ackermann(urdf_path, n_envs=1)
    # If Ackermann with 4 wheels, use car preset
    elif drive_type_val == 0 and len(wheels) == 4:
        print(f" [Genesis] [Preset] Loading car_4w_rwd_ackermann preset for Vehicle {target_id}")
        from genesis_vehicle.presets import car_4w_rwd_ackermann
        cfg = car_4w_rwd_ackermann(urdf_path, n_envs=1)
    else:
        # 2. Extract Drive/Steer rules from Unreal mapping
        _drv_joints = mapping.get('drivingJoints', mapping.get('DriveJoints', []))
        _str_joints = mapping.get('steeringJoints', mapping.get('SteeringJoints', []))
        drive_joint_names = [d.get('jointName', d.get('JointName', '')).lower() for d in _drv_joints]
        steer_joint_names = [s.get('jointName', s.get('JointName', '')).lower() for s in _str_joints]

        drive_weights = []
        steered_axles = []

        for w in wheels:
            w_spin = w.spin_joint_name.lower() if w.spin_joint_name else ""
            w_steer = w.steer_joint_name.lower() if w.steer_joint_name else ""

            # Drive Weights Mapping
            if w_spin in drive_joint_names:
                drive_weights.append(1.0)
            else:
                drive_weights.append(0.0)

            # Steering Axle Mapping
            if w_steer in steer_joint_names:
                if w.axle_index not in steered_axles:
                    steered_axles.append(w.axle_index)

        # Normalize drive weights
        sw = sum(drive_weights)
        if sw > 0:
            drive_weights = [x / sw for x in drive_weights]

        # 3. Dynamic Strategy Build
        t_drive = float(mapping.get('maxTorque', mapping.get('MaxTorque', 5000.0)))
        t_brake = float(mapping.get('maxBrake', mapping.get('MaxBrake', t_drive)))
        if t_brake <= 0.0:
            t_brake = t_drive

        print(f" [Genesis] [Config] Vehicle {target_id} Drivetrain Configured: MaxTorque={t_drive} Nm, MaxBrake={t_brake} Nm")

        # Resolve drivetrain strategy from Unreal
        drivetrain_strategy_val = mapping.get('drivetrainStrategy', mapping.get('DrivetrainStrategy', 0))
        if drivetrain_strategy_val == 1: # RWD
            from genesis_vehicle.strategies import RWD
            drivetrain = RWD(t_drive_max=t_drive, t_brake_max=t_brake, driven_axles=(1,))
        elif drivetrain_strategy_val == 2: # FWD
            from genesis_vehicle.strategies import FWD
            drivetrain = FWD(t_drive_max=t_drive, t_brake_max=t_brake, driven_axles=(0,))
        elif drivetrain_strategy_val == 3: # PerSide
            from genesis_vehicle.strategies import PerSide
            drivetrain = PerSide(t_drive_max=t_drive, t_brake_max=t_brake)
        else: # AWD (Default: 0)
            drivetrain = AWD(t_drive_max=t_drive, t_brake_max=t_brake, drive_weights=drive_weights)

        # Resolve steering strategy
        max_steer_rad = _mapping_steer_rad(mapping, default=0.7)
        if steered_axles:
            steering = PartialAckermann(max_steer_rad=max_steer_rad, steered_axles=tuple(steered_axles))
        else:
            from genesis_vehicle import NoSteer
            steering = NoSteer()

        # Resolve coupling strategy
        coupling_strategy_val = mapping.get('couplingStrategy', mapping.get('CouplingStrategy', 0))
        if coupling_strategy_val == 1: # SameSideBelt
            from genesis_vehicle.strategies import SameSideBelt
            coupling = SameSideBelt()
        else: # Independent
            coupling = Independent()

        # 4. Dynamic suspension parameter computation
        chassis_mass = urdf_parsed.chassis_mass if urdf_parsed.chassis_mass else 1500.0
        m_per_wheel = chassis_mass / len(wheels) if len(wheels) > 0 else chassis_mass / 4.0

        if chassis_mass < 1000.0:
            fn = 1.8  # ultra-light / sports car / kart
            damping_ratio_comp = 0.7
            damping_ratio_ext = 0.45
        elif chassis_mass > 3000.0:
            fn = 1.2  # heavy truck / bus
            damping_ratio_comp = 0.7
            damping_ratio_ext = 0.45
        else:
            fn = 1.5  # regular sedan / SUV
            damping_ratio_comp = 0.7
            damping_ratio_ext = 0.45

        omega_n = 2 * math.pi * fn
        dynamic_k_susp = m_per_wheel * (omega_n ** 2)

        c_critical = 2 * math.sqrt(dynamic_k_susp * m_per_wheel)
        dynamic_c_comp = damping_ratio_comp * c_critical
        dynamic_c_ext = damping_ratio_ext * c_critical

        urdf_root = None
        try:
            if os.path.exists(urdf_path):
                urdf_tree = ET.parse(urdf_path)
                urdf_root = urdf_tree.getroot()
        except Exception as e:
            print(f" [Genesis] [Warning] Failed to parse URDF XML for joint limits: {e}")

        for w in wheels:
            w.k_susp = dynamic_k_susp
            w.c_compression = dynamic_c_comp
            w.c_extension = dynamic_c_ext

            rest_stroke = 0.2  # default 20cm
            if urdf_root is not None and w.susp_joint_name:
                joint_elem = urdf_root.find(f".//joint[@name='{w.susp_joint_name}']")
                if joint_elem is not None:
                    limit_elem = joint_elem.find("limit")
                    if limit_elem is not None:
                        lower = float(limit_elem.get("lower", 0.0))
                        upper = float(limit_elem.get("upper", 0.0))
                        if (upper - lower) > 0.05:
                            rest_stroke = upper - lower

                    # Support parsing stiffness and damping attributes from the dynamics tag
                    dyn_elem = joint_elem.find("dynamics")
                    if dyn_elem is not None:
                        stiff = dyn_elem.get("stiffness") or dyn_elem.get("spring_stiffness")
                        if stiff is not None:
                            w.k_susp = float(stiff)
                            print(f" [Genesis] URDF {w.susp_joint_name} stiffness override: {w.k_susp} N/m")

                        damp = dyn_elem.get("damping")
                        if damp is not None:
                            w.c_compression = float(damp)
                            w.c_extension = float(damp)

                        comp_damp = dyn_elem.get("compression_damping")
                        if comp_damp is not None:
                            w.c_compression = float(comp_damp)

                        ext_damp = dyn_elem.get("extension_damping")
                        if ext_damp is not None:
                            w.c_extension = float(ext_damp)

                        print(f" [Genesis] URDF {w.susp_joint_name} damping overrides: comp={w.c_compression}, ext={w.c_extension}")

            w.rest_stroke = rest_stroke
            w.comp_rate_clamp = max(2.0, (w.radius if w.radius else 0.3) * 6.0)

        # 5. Construct Final Config
        cfg = VehicleConfig(
            urdf_path=urdf_path,
            wheels=wheels,
            steering=steering,
            drivetrain=drivetrain,
            coupling=coupling,
            tire=PacejkaAnisotropic(),
            stability_hooks=[RollingResistance(), LowSpeedRegularizer()],
            visual_susp_mode="kinematic"
        )

    # 6. Apply global strategies to loaded preset configurations if explicitly overridden in Unreal
    # [FIX] UE FGenesisVehicleMapping serializes as 'steerScale', so recognize that key too
    steer_override = _mapping_steer_rad(mapping, default=None)
    if steer_override is not None:
        if hasattr(cfg.steering, 'max_steer_rad'):
            cfg.steering.max_steer_rad = steer_override
            print(f" [Genesis] Override max_steer_rad = {steer_override} on preset steering strategy.")

    # Override brake bias front ratio if specified
    if 'brakeBiasFrontRatio' in mapping or 'BrakeBiasFrontRatio' in mapping:
        ratio = float(mapping.get('brakeBiasFrontRatio', mapping.get('BrakeBiasFrontRatio', 0.6)))
        # Ackermann reference has 4 wheels: split it to [ratio/2, ratio/2, (1-ratio)/2, (1-ratio)/2]
        if hasattr(cfg.drivetrain, 'brake_bias') and cfg.drivetrain.brake_bias is not None:
            if len(cfg.drivetrain.brake_bias) == 4:
                cfg.drivetrain.brake_bias = [ratio / 2.0, ratio / 2.0, (1.0 - ratio) / 2.0, (1.0 - ratio) / 2.0]
                print(f" [Genesis] Override brake_bias = {cfg.drivetrain.brake_bias} based on ratio {ratio}")

    # Override drive max torque and brake torque if specified
    if 'maxTorque' in mapping or 'MaxTorque' in mapping:
        t_val = float(mapping.get('maxTorque', mapping.get('MaxTorque', 5000.0)))
        if hasattr(cfg.drivetrain, 't_drive_max'):
            cfg.drivetrain.t_drive_max = t_val
            print(f" [Genesis] Override maxTorque = {t_val} Nm on preset drivetrain.")
    if 'maxBrake' in mapping or 'MaxBrake' in mapping:
        b_val = float(mapping.get('maxBrake', mapping.get('MaxBrake', 5000.0)))
        if hasattr(cfg.drivetrain, 't_brake_max'):
            cfg.drivetrain.t_brake_max = b_val
            print(f" [Genesis] Override maxBrake = {b_val} Nm on preset drivetrain.")
    # Override the drive wheel-omega cap (top-speed limiter) if specified
    if 'omegaMaxDrive' in mapping or 'OmegaMaxDrive' in mapping:
        o_val = float(mapping.get('omegaMaxDrive', mapping.get('OmegaMaxDrive', 100.0)))
        if hasattr(cfg.drivetrain, 'omega_max_drive'):
            cfg.drivetrain.omega_max_drive = o_val
            print(f" [Genesis] Override omegaMaxDrive = {o_val} rad/s on preset drivetrain.")

    # Resolve wheel friction coefficients from URDF, falling back to chassis material friction (t_fric) or defaults
    for w in cfg.wheels:
        w_name = w.name or w.spin_joint_name or ""

        if w.mu_long is not None:
            print(f" [Genesis] Resolved mu_long = {w.mu_long} for wheel {w_name} from URDF.")
        else:
            w.mu_long = t_fric
            print(f" [Genesis] Falling back mu_long = {w.mu_long} for wheel {w_name} to chassis friction (t_fric={t_fric:.2f}).")

        if w.mu_lat is not None:
            print(f" [Genesis] Resolved mu_lat = {w.mu_lat} for wheel {w_name} from URDF.")
        else:
            w.mu_lat = t_fric
            print(f" [Genesis] Falling back mu_lat = {w.mu_lat} for wheel {w_name} to chassis friction (t_fric={t_fric:.2f}).")

    # 7. Apply Wheel Overrides if present in mapping JSON
    wheel_overrides = mapping.get('wheelOverrides', mapping.get('WheelOverrides', []))
    if wheel_overrides:
        print(f" [Genesis] Applying {len(wheel_overrides)} wheel override settings...")
        for override in wheel_overrides:
            override_w_name = override.get('wheelName', override.get('WheelName', ''))
            if not override_w_name:
                continue
            for w in cfg.wheels:
                if is_wheel_match_fuzzy(override_w_name, w):
                    # Wheel physical dimensions
                    if 'radius' in override or 'Radius' in override:
                        val = float(override.get('radius', override.get('Radius', -1.0)))
                        if val > 0: w.radius = val
                    if 'mass' in override or 'Mass' in override:
                        val = float(override.get('mass', override.get('Mass', -1.0)))
                        if val > 0: w.mass = val
                    if 'inertia' in override or 'Inertia' in override:
                        val = float(override.get('inertia', override.get('Inertia', -1.0)))
                        # WheelConfig's spin-inertia field is `i_wheel` —
                        # `w.inertia` was a dead attribute (silent no-op).
                        if val > 0: w.i_wheel = val

                    # Suspension properties
                    if 'stiffness' in override or 'Stiffness' in override:
                        w.k_susp = float(override.get('stiffness', override.get('Stiffness', w.k_susp)))
                    if 'compressionDamping' in override or 'CompressionDamping' in override:
                        w.c_compression = float(override.get('compressionDamping', override.get('CompressionDamping', w.c_compression)))
                    if 'extensionDamping' in override or 'ExtensionDamping' in override:
                        w.c_extension = float(override.get('extensionDamping', override.get('ExtensionDamping', w.c_extension)))
                    if 'restStroke' in override or 'RestStroke' in override:
                        w.rest_stroke = float(override.get('restStroke', override.get('RestStroke', w.rest_stroke)))

                    # Friction properties
                    if 'muLong' in override or 'MuLong' in override:
                        val = float(override.get('muLong', override.get('MuLong', -1.0)))
                        if val > 0.0: w.mu_long = val
                    if 'muLat' in override or 'MuLat' in override:
                        val = float(override.get('muLat', override.get('MuLat', -1.0)))
                        if val > 0.0: w.mu_lat = val
                    if 'rollingResistance' in override or 'RollingResistance' in override:
                        w.rolling_resistance = float(override.get('rollingResistance', override.get('RollingResistance', 0.015)))

                    # Pacejka parameters (Bx, Cx, Ex, By, Cy, Ey)
                    if 'pbX' in override or 'PbX' in override:
                        w.bx = float(override.get('pbX', override.get('PbX', 10.0)))
                    if 'pcX' in override or 'PcX' in override:
                        w.cx = float(override.get('pcX', override.get('PcX', 1.65)))
                    if 'peX' in override or 'PeX' in override:
                        w.ex = float(override.get('peX', override.get('PeX', 0.4)))
                    if 'pbY' in override or 'PbY' in override:
                        w.by = float(override.get('pbY', override.get('PbY', 8.0)))
                    if 'pcY' in override or 'PcY' in override:
                        w.cy = float(override.get('pcY', override.get('PcY', 1.30)))
                    if 'peY' in override or 'PeY' in override:
                        w.ey = float(override.get('peY', override.get('PeY', 0.4)))

                    print(f"   - Applied overrides to wheel: {override_w_name} (matched to URDF link: {w.name})")

    _cfg_cache[_key] = cfg
    return cfg


def print_resolved_table(target_id, resolved_cfg):
    """Print a summary table of the final resolved vehicle physics settings."""
    print("\n=========================================================================================================")
    print("🚗 GENESIS RESOLVED VEHICLE PHYSICS SETTINGS")
    print(f"  Target ID: {target_id}")
    print(f"  Chassis Mass: {resolved_cfg.chassis.mass if resolved_cfg.chassis.mass else 1500.0} kg")
    print("---------------------------------------------------------------------------------------------------------")
    print("  Wheel Settings Summary:")
    print("  +-----------------+-------+--------+---------+--------+--------+-----------+-----------+-----------+--------+")
    print("  | Wheel Link Name | Mass  | Radius | Inertia | MuLong | MuLat  | Stiffness | Damp Comp | Damp Ext  | Stroke |")
    print("  +-----------------+-------+--------+---------+--------+--------+-----------+-----------+-----------+--------+")
    for w in resolved_cfg.wheels:
        w_name = w.name if w.name else "unknown"
        w_mass = w.mass if w.mass is not None else 0.0
        w_radius = w.radius if w.radius is not None else 0.0
        w_inertia = w.i_wheel if w.i_wheel is not None else 0.0
        w_mu_long = w.mu_long if w.mu_long is not None else 0.0
        w_mu_lat = w.mu_lat if w.mu_lat is not None else 0.0
        w_k_susp = w.k_susp if w.k_susp is not None else 0.0
        w_c_comp = w.c_compression if w.c_compression is not None else 0.0
        w_c_ext = w.c_extension if w.c_extension is not None else 0.0
        w_stroke = w.rest_stroke if w.rest_stroke is not None else 0.0
        print(f"  | {w_name:<15} | {w_mass:<5.1f} | {w_radius:<6.3f} | {w_inertia:<7.4f} | {w_mu_long:<6.2f} | {w_mu_lat:<6.2f} | {w_k_susp:<9.1f} | {w_c_comp:<9.1f} | {w_c_ext:<9.1f} | {w_stroke:<6.3f} |")
    print("  +-----------------+-------+--------+---------+--------+--------+-----------+-----------+-----------+--------+")
    print("=========================================================================================================\n")


def build_vehicle(vs, target_entities, vehicles, target_id, target_info,
                  urdf_path, mapping, ue_friction, ue_restitution, vis_mode):
    """
    Register a vehicle via vs.add_vehicle based on the given URDF path and
    mapping settings, auto-tuning the cfg dynamically and registering it with
    the VehicleScene.
    (L2 path — one entity per vehicle (per-entity). proxy/sensor/VehiclePhysics
    are created in vs.build(), so the caller fills controllers after build.)

    WheelJointInternalSync (driving the visual wheel joints) is auto-managed by
    VehicleScene.build() — ON only when the main scene is rendered by Genesis
    (show_viewer). The server is drawn by an external renderer (UE) and
    captures via wheel_visual_transforms (closed-form), so it stays OFF when
    headless.
    """
    t_pos = target_info.get('pos', [0, 0, 2])
    t_quat = target_info.get('quat', [1, 0, 0, 0])
    t_fric = target_info.get('friction', ue_friction)
    t_rest = 0.0  # forced to 0 to suppress chassis bouncing

    temp_path = strip_wheel_collisions(urdf_path)
    target_morph = gs.morphs.URDF(file=temp_path, pos=t_pos, quat=t_quat, fixed=False, align=False)
    t_color = (1.0, 0.3, 0.3, 0.5)   # debug color (semi-transparent red)

    # Build the vehicle config (single source shared with the L3 path).
    # enable_wheel_joint_internal_sync is set automatically by vs.build() based on
    # rendering, so it is not touched here.
    cfg = build_cfg(urdf_path, mapping, t_fric, target_id=target_id)

    # VehicleScene builds the entity into the main scene + creates
    # raycaster/proxy/VehiclePhysics in vs.build(). The caller never touches
    # a scene directly.
    veh = vs.add_vehicle(
        urdf_path, cfg=cfg, morph=target_morph,
        material=gs.materials.Rigid(friction=t_fric, coup_restitution=t_rest, sdf_cell_size=10000.0),
        surface=gs.surfaces.Rough(color=t_color), vis_mode=vis_mode,
        name=f"target_{target_id}")
    vehicles[target_id] = veh
    target_entities[target_id] = veh.entity_main
    print(f" [Genesis] Created Vehicle Target {target_id} at {t_pos}")
