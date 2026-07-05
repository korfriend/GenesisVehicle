import sys
import os

from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import threading
import time
import json
import genesis as gs
from genesis_vehicle import parse_urdf

class OSCManager:
    """
    OSCManager
    
    A bidirectional UDP OSC communication manager designed to bridge the Genesis
    Python Simulator with Unreal Engine. It manages three dedicated ports:
    - 7000: Sending legacy location data to Level Blueprints.
    - 7001: Receiving all incoming data (Init, Physics, Setup) from UE on a single thread.
    - 7002/7004: Sending state and observation tensors to the Unreal C++ GenesisBridge Plugin.
    """
    def __init__(self, send_ip='127.0.0.1', recv_port=7001, send_port_cpp=7002, send_port_obs=7004, subject_name='GenesisCar', verbose=False):
        self.verbose = verbose
        self.send_ip = send_ip
        
        # Output Ports (To Unreal Engine GenesisBridge)
        self.send_port_cpp = send_port_cpp # Default: 7002 (C++ GenesisBridge Main)
        self.send_port_obs = send_port_obs # Default: 7004 (C++ GenesisBridge Obs)
        
        # Input Port (From Unreal Engine)
        self.recv_port = recv_port         # Default: 7001 (Unified Receiver)
        
        self.subject_name = subject_name
        
        # Sender 1 (to UE 7002 - GenesisBridge C++ Main)
        self.client_cpp = udp_client.SimpleUDPClient(self.send_ip, self.send_port_cpp)
        
        # Sender 3 (to UE 7004 - GenesisBridge C++ Observation)
        self.client_obs = udp_client.SimpleUDPClient(self.send_ip, self.send_port_obs)
        
        # Receiver Status Data
        self.received_data = {
            'location': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0, 1.0], # w, x, y, z
            'command': None,
            'updated': False,
            'input_queue': [],  # [NEW] Queue for URDF inputs to handle Catch-up logic
            'urdf_inputs': [],  # Fallback/Legacy
            'frame_id': 0,      # Current frame ID from Unreal
            'last_frame_id': -1, # Last processed frame ID
            'target_forces': {},        # {TID: [fx, fy, fz]} (Persistent)
            'target_overrides': {},     # {TID: {'pos':, 'quat':, 'teleport':}} (One-shot)
            'target_relative_cmds': {}, # {TID: [{type:, data:, teleport:}, ...]} (Queue)
            'obstacle_overrides': {}    # {ObsID: {'pos':, 'quat':}} (UE-Driven Obstacle sync)
        }
        
        self.urdf_init_request = {'path': '', 'mapping': {}} # Initialize properly
        self.init_physics_data = {}
        self.init_target_data = {}
        self.init_done_event = threading.Event()
        self._new_frame_event = threading.Event()  # [LOCKSTEP FIX] Event-based sync for wait_for_next_frame
        self._count = 0 # Initialize debug counter
        
        # Shared Dispatcher (Receives everything on 7001)
        self.dispatcher = Dispatcher()
        self.dispatcher.map(f"/{self.subject_name}/Location", self._handle_location)
        self.dispatcher.map(f"/{self.subject_name}/Rotation", self._handle_rotation)
        self.dispatcher.map("/Init/Obstacle", self._handle_init_obstacle)
        self.dispatcher.map("/Init/Obstacle/", self._handle_init_obstacle)
        self.dispatcher.map("/Init/Done", self._handle_init_done)
        self.dispatcher.map("/Init/Done/", self._handle_init_done)
        self.dispatcher.map("/Init/Target", self._handle_init_target)
        self.dispatcher.map("/Init/Target/", self._handle_init_target)
        self.dispatcher.map("/Obstacle/Update", self._handle_obstacle) 
        self.dispatcher.map("/Genesis/Init/Physics", self._handle_init_physics)
        self.dispatcher.map("/Genesis/Init/Physics/", self._handle_init_physics)
        
        # --- NEW: Control Handler ---
        # UE with PushContainer sends trailing slash!
        self.dispatcher.map("/Genesis/Control/", self._handle_control) # Trailing slash
        self.dispatcher.map("/Genesis/Config/Physics/", self._handle_physics_settings) # Trailing slash
        
        # --- Multi-Target Control Handlers (New Container) ---
        TC_ADDR = "/Genesis/Vehicle/TargetControl"
        self.dispatcher.map(f"{TC_ADDR}/Transform/", self._handle_target_transform_override)
        self.dispatcher.map(f"{TC_ADDR}/Position/", self._handle_target_position_override)
        self.dispatcher.map(f"{TC_ADDR}/Rotation/", self._handle_target_rotation_override)
        self.dispatcher.map(f"{TC_ADDR}/AddLocalOffset/", self._handle_target_add_local_offset)
        self.dispatcher.map(f"{TC_ADDR}/AddWorldOffset/", self._handle_target_add_world_offset)
        self.dispatcher.map(f"{TC_ADDR}/AddLocalRotation/", self._handle_target_add_local_rotation)
        self.dispatcher.map(f"{TC_ADDR}/AddWorldRotation/", self._handle_target_add_world_rotation)
        self.dispatcher.map(f"{TC_ADDR}/AddWorldForce/", self._handle_target_add_world_force)
        self.dispatcher.map(f"{TC_ADDR}/AddWorldImpulse/", self._handle_target_add_world_impulse)
        self.dispatcher.map(f"{TC_ADDR}/AddWorldTorque/", self._handle_target_add_world_torque)

        # Fallback without trailing slash
        self.dispatcher.map(f"{TC_ADDR}/Transform", self._handle_target_transform_override)
        self.dispatcher.map(f"{TC_ADDR}/Position", self._handle_target_position_override)
        self.dispatcher.map(f"{TC_ADDR}/Rotation", self._handle_target_rotation_override)
        self.dispatcher.map(f"{TC_ADDR}/AddLocalOffset", self._handle_target_add_local_offset)
        self.dispatcher.map(f"{TC_ADDR}/AddWorldOffset", self._handle_target_add_world_offset)
        self.dispatcher.map(f"{TC_ADDR}/AddLocalRotation", self._handle_target_add_local_rotation)
        self.dispatcher.map(f"{TC_ADDR}/AddWorldRotation", self._handle_target_add_world_rotation)
        self.dispatcher.map(f"{TC_ADDR}/AddWorldForce", self._handle_target_add_world_force)
        self.dispatcher.map(f"{TC_ADDR}/AddWorldImpulse", self._handle_target_add_world_impulse)
        self.dispatcher.map(f"{TC_ADDR}/AddWorldTorque", self._handle_target_add_world_torque)

        # --- UE-Driven Obstacle Handlers ---
        self.dispatcher.map("/Genesis/Obstacle/Transform/", self._handle_obstacle_transform_override)
        self.dispatcher.map("/Genesis/Obstacle/Transform", self._handle_obstacle_transform_override)

        # Fallback for old cached Blueprints sending to /GenesisCar/
        self.dispatcher.map(f"/{self.subject_name}/Control/", self._handle_control)
        self.dispatcher.map(f"/{self.subject_name}/Control/Force/", self._handle_force)
        self.dispatcher.map(f"/{self.subject_name}/Control/Torque/", self._handle_torque)
        self.dispatcher.map(f"/{self.subject_name}/Control/Transform/", self._handle_target_transform_override)
        self.dispatcher.map(f"/{self.subject_name}/Control/Position/", self._handle_target_position_override)
        self.dispatcher.map(f"/{self.subject_name}/Control/Rotation/", self._handle_target_rotation_override)
        
        # Also map without slash just in case
        self.dispatcher.map("/Genesis/Control", self._handle_control)
        self.dispatcher.map("/Genesis/Config/Physics", self._handle_physics_settings)
        self.dispatcher.map("/Genesis/Control/Force", self._handle_force)
        self.dispatcher.map("/Genesis/Control/Torque", self._handle_torque)
        self.dispatcher.map("/Genesis/Control/Transform", self._handle_target_transform_override)
        self.dispatcher.map("/Genesis/Control/Position", self._handle_target_position_override)
        self.dispatcher.map("/Genesis/Control/Rotation", self._handle_target_rotation_override)
        
        # URDF Control System
        self.dispatcher.map("/Genesis/Vehicle/Init", self._handle_urdf_init)
        self.dispatcher.map("/Genesis/Vehicle/Init/", self._handle_urdf_init)
        
        # Robust mappings for Control
        self.dispatcher.map("/Genesis/Vehicle/Control", self._handle_urdf_control)
        self.dispatcher.map("/Genesis/Vehicle/Control/", self._handle_urdf_control)
        self.dispatcher.map(f"/{self.subject_name}/Vehicle/Control", self._handle_urdf_control)
        self.dispatcher.map(f"/{self.subject_name}/Vehicle/Control/", self._handle_urdf_control)
        
        self.dispatcher.map("/Genesis/Sync/Reset", self._handle_sync_reset)
        self.dispatcher.map("/Genesis/Sync/Reset/", self._handle_sync_reset)
        
        self.dispatcher.map("/Genesis/Vehicle/FetchMetadata", self._handle_fetch_metadata)
        
        # Default handler for debugging
        # Start Server (Listening ONLY on 7001 for EVERYTHING)
        # [CRITICAL FIX] Use BlockingOSCUDPServer in a single background thread 
        # instead of ThreadingOSCUDPServer to avoid spawning thousands of threads.
        self.server = BlockingOSCUDPServer(('0.0.0.0', self.recv_port), self.dispatcher)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        
        # Track obstacles
        self.received_obstacles = {}
        
        print(f" [Genesis] Manager Started.")
        print(f"       C++ Main Port:  Send->{self.send_port_cpp}")
        print(f" [Genesis] OUT: Genesis -> UE (Observations) : {self.send_port_obs}")
        print(f" [Genesis] IN : UE -> Genesis (All Data)     : {self.recv_port}")
        print("-" * 50)

    def wait_for_initialization(self, timeout=60.0):
        print(f" [Genesis] Requesting Initialization Data (Obstacles + Physics) (Target UE Port: {self.send_port_cpp})...")
        self.received_obstacles = {} 
        self.init_physics_data = {}
        self.init_target_data = {}
        self.init_done_event = threading.Event()
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Send Handshake Ping to Unreal Engine asking them to execute BroadcastInitializationData()
            self.client_cpp.send_message("/Genesis/RequestInit", [])
            
            # Check for a response for 1 second
            if self.init_done_event.wait(timeout=1.0):
                print(f" [Genesis] Initialization Complete. Obstacles: {len(self.received_obstacles)}, Physics: {self.init_physics_data}")
                return {'obstacles': self.received_obstacles, 'physics': self.init_physics_data, 'target': self.init_target_data}
            
            print(f" [Genesis] Waiting for Unreal Engine to respond... ({int(time.time() - start_time)}s / {int(timeout)}s)")
            
        print(f" [Genesis] Initialization Timed Out.")
        return {'obstacles': self.received_obstacles, 'physics': self.init_physics_data, 'target': self.init_target_data}

    def _handle_init_obstacle(self, address, *args):
        # args contains multiple obstacles, flattened.
        # Legacy: 16, New: 17, Latest: 18 fields (CollisionSource included)
        num_fields = 18 if (len(args) % 18 == 0) else (17 if (len(args) % 17 == 0) else 16)
        
        if len(args) == 0 or len(args) % num_fields != 0:
            if self.verbose:
                print(f" [Genesis] [WARNING] Invalid obstacle data length: {len(args)}. Expected multiple of 16, 17 or 18")
            return
            
        if not hasattr(self, 'received_obstacles'):
            self.received_obstacles = {}
            
        num_obstacles = len(args) // num_fields
        
        for i in range(num_obstacles):
            offset = i * num_fields
            obs_id = int(args[offset])
            obs_type = int(args[offset+1]) # <--- Capture TYPE
            
            # Received as Genesis-compatible coordinates from UE
            px = float(args[offset+2])
            py = float(args[offset+3])
            pz = float(args[offset+4])
            pos = [px, py, pz]
            
            quat = [float(args[offset+8]), float(args[offset+5]), float(args[offset+6]), float(args[offset+7])]
            scale = [float(args[offset+9]), float(args[offset+10]), float(args[offset+11])]
            
            # Physics Properties
            mass = float(args[offset+12])
            friction = float(args[offset+13])
            restitution = float(args[offset+14])
            mesh_path = str(args[offset+15])
            b_is_dynamic = int(args[offset+16]) if num_fields >= 17 else 0
            collision_source = str(args[offset+17]) if num_fields == 18 else ("[Complex:Mesh]" if obs_type == 5 else "[Simple:Box]")
            
            self.received_obstacles[obs_id] = {
                'type': obs_type,
                'pos': pos,
                'quat': quat,
                'scale': scale,
                'mass': mass,
                'friction': friction,
                'restitution': restitution,
                'mesh_path': mesh_path,
                'b_dynamic': b_is_dynamic,
                'collision_source': collision_source
            }
            if self.verbose:
                print(f" [Genesis] [DEBUG] Processed Obs_{obs_id}: Type={obs_type}, Source={collision_source}, Pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

    def _handle_init_physics(self, address, *args):
        # Expected Args: [GravityZ(float), TimeStep(float), DefaultFriction(float)]
        if len(args) >= 1:
            if not hasattr(self, 'init_physics_data'):
                self.init_physics_data = {}
            if len(args) > 0: self.init_physics_data['gravity'] = float(args[0])
            if len(args) > 1: self.init_physics_data['dt'] = float(args[1])
            if len(args) > 2: self.init_physics_data['friction'] = float(args[2])
            if self.verbose:
                print(f" [Genesis] [DEBUG] Processed Init Physics: {self.init_physics_data}")

    def _handle_init_target(self, address, *args):
        # args: [ID], Type, Px, Py, Pz, Qx, Qy, Qz, Qw, Sx, Sy, Sz, Mass, Friction, Restitution
        # Depending on the C++ GenesisBridge plugin version, the ID may be omitted (14 args) or included (15 args)
        if len(args) in [14, 15]:
            offset = 1 if len(args) == 15 else 0
            target_id = int(args[0]) if len(args) == 15 else 0
            target_type = int(args[offset])
            
            # Received as Genesis-compatible coordinates from UE.
            px = float(args[offset+1])
            py = float(args[offset+2])
            pz = float(args[offset+3])
            pos = [px, py, pz]
            
            # Quat logic: [W, X, Y, Z]. 
            quat = [float(args[offset+7]), float(args[offset+4]), float(args[offset+5]), float(args[offset+6])]
            
            scale = [float(args[offset+8]), float(args[offset+9]), float(args[offset+10])]
            
            mass = float(args[offset+11])
            friction = float(args[offset+12])
            restitution = float(args[offset+13])
            
            if not hasattr(self, 'init_target_data'):
                self.init_target_data = {}
                
            self.init_target_data[target_id] = {
                'type': target_type,
                'pos': pos,
                'quat': quat,
                'scale': scale,
                'mass': mass,
                'friction': friction,
                'restitution': restitution
            }
            if self.verbose:
                print(f" [Genesis] [DEBUG] Target {target_id} Info Received: Type={target_type}, Pos={pos}")

    def _handle_init_done(self, address, *args):
        print(f" [Genesis] Initialization Done. Total Obstacles: {len(self.received_obstacles)}")
        if hasattr(self, 'init_done_event'):
            self.init_done_event.set()

    def _handle_obstacle(self, address, *args):
        # Same logic for update
        self._handle_init_obstacle(address, *args)

    def _default_handler(self, address, *args):
        if self.verbose:
            print(f" [Genesis] Address: {address} | Args: {args}")

    def _handle_location(self, address, *args):
        # UE sends this already converted, so we trust it fully and take it as-is.
        self.received_data['location'] = [args[0], args[1], args[2]]
        self.received_data['updated'] = True

    def _handle_rotation(self, address, *args):
        # We trust UE to send this already ordered as [w, x, y, z] (or whatever order is required).
        self.received_data['rotation'] = [args[0], args[1], args[2], args[3]]
        self.received_data['updated'] = True


    def _handle_control(self, address, *args):
        """
        Handle control commands: Start, Pause, Stop, Reset
        Args: [CommandString]
        """
        if len(args) > 0:
            cmd = str(args[0]).lower()
            self.received_data['command'] = cmd
            self.received_data['updated'] = True # Force update flag
            self._new_frame_event.set()  # [LOCKSTEP FIX] Unblock wait_for_next_frame on commands (stop/reset)
            if self.verbose:
                print(f" [Genesis] Received Command: {cmd}")

    def _handle_force(self, address, *args):
        """
        Handle directional force command from Unreal Engine
        Args: [FX(float), FY(float), FZ(float)]
        """
        if len(args) >= 3:
            # Received as Genesis-compatible Force.
            # Divide by roughly 100 or specific multiplier if applying raw Unreal Units as Impulses
            fx = float(args[0])
            fy = float(args[1])
            fz = float(args[2])
            
            if 'force' not in self.received_data:
                self.received_data['force'] = []
                
            self.received_data['force'] = [fx, fy, fz]
            self.received_data['updated'] = True
            
            if self.verbose:
                print(f" [Genesis] Received Force: [{fx:.1f}, {fy:.1f}, {fz:.1f}]")

    def _handle_torque(self, address, *args):
        """
        Handle rotational force (torque) command from Unreal Engine
        Args: [TX(float), TY(float), TZ(float)]
        """
        if len(args) >= 3:
            # Received as Genesis-compatible Torque.
            tx = float(args[0])
            ty = float(args[1])
            tz = float(args[2])
            
            if 'torque' not in self.received_data:
                self.received_data['torque'] = []
                
            self.received_data['torque'] = [tx, ty, tz]
            self.received_data['updated'] = True
            
            if self.verbose:
                print(f" [Genesis] Received Torque: [{tx:.1f}, {ty:.1f}, {tz:.1f}]")

    def _handle_target_transform_override(self, address, *args):
        if len(args) >= 8:
            tid = int(args[0])
            pos = [float(args[1]), float(args[2]), float(args[3])]
            quat = [float(args[7]), float(args[4]), float(args[5]), float(args[6])]
            teleport = float(args[8]) > 0.5
            self.received_data['target_overrides'][tid] = {'pos': pos, 'quat': quat, 'teleport': teleport}
            self.received_data['updated'] = True

    def _handle_obstacle_transform_override(self, address, *args):
        if len(args) >= 8:
            obs_id = int(args[0])
            pos = [float(args[1]), float(args[2]), float(args[3])]
            quat = [float(args[7]), float(args[4]), float(args[5]), float(args[6])]
            self.received_data['obstacle_overrides'][obs_id] = {'pos': pos, 'quat': quat}
            self.received_data['updated'] = True

    def _handle_target_position_override(self, address, *args):
        if len(args) >= 4:
            tid = int(args[0])
            pos = [float(args[1]), float(args[2]), float(args[3])]
            teleport = float(args[4]) > 0.5
            if tid not in self.received_data['target_overrides']: self.received_data['target_overrides'][tid] = {}
            self.received_data['target_overrides'][tid].update({'pos': pos, 'teleport': teleport})
            self.received_data['updated'] = True

    def _handle_target_rotation_override(self, address, *args):
        if len(args) >= 5:
            tid = int(args[0])
            quat = [float(args[4]), float(args[1]), float(args[2]), float(args[3])]
            teleport = float(args[5]) > 0.5
            if tid not in self.received_data['target_overrides']: self.received_data['target_overrides'][tid] = {}
            self.received_data['target_overrides'][tid].update({'quat': quat, 'teleport': teleport})
            self.received_data['updated'] = True

    def _queue_target_relative_cmd(self, tid, cmd_type, data, teleport=True):
        if tid not in self.received_data['target_relative_cmds']:
            self.received_data['target_relative_cmds'][tid] = []
        self.received_data['target_relative_cmds'][tid].append({'type': cmd_type, 'data': data, 'teleport': teleport})
        self.received_data['updated'] = True

    def _handle_target_add_local_offset(self, address, *args):
        if len(args) >= 4:
            tid, dx, dy, dz = int(args[0]), float(args[1]), float(args[2]), float(args[3])
            tp = float(args[4]) > 0.5
            self._queue_target_relative_cmd(tid, 'AddLocalOffset', [dx, dy, dz], tp)

    def _handle_target_add_world_offset(self, address, *args):
        if len(args) >= 4:
            tid, dx, dy, dz = int(args[0]), float(args[1]), float(args[2]), float(args[3])
            tp = float(args[4]) > 0.5
            self._queue_target_relative_cmd(tid, 'AddWorldOffset', [dx, dy, dz], tp)

    def _handle_target_add_local_rotation(self, address, *args):
        if len(args) >= 5:
            tid = int(args[0])
            quat = [float(args[4]), float(args[1]), float(args[2]), float(args[3])]
            tp = float(args[5]) > 0.5
            self._queue_target_relative_cmd(tid, 'AddLocalRotation', quat, tp)

    def _handle_target_add_world_rotation(self, address, *args):
        if len(args) >= 5:
            tid = int(args[0])
            quat = [float(args[4]), float(args[1]), float(args[2]), float(args[3])]
            tp = float(args[5]) > 0.5
            self._queue_target_relative_cmd(tid, 'AddWorldRotation', quat, tp)

    def _handle_target_add_world_force(self, address, *args):
        if len(args) >= 4:
            tid, fx, fy, fz = int(args[0]), float(args[1]), float(args[2]), float(args[3])
            prev = self.received_data['target_forces'].get(tid, [0.0, 0.0, 0.0])
            self.received_data['target_forces'][tid] = [prev[0]+fx, prev[1]+fy, prev[2]+fz]
            self.received_data['updated'] = True
            if self.verbose:
                print(f" [Genesis] AddWorldForce Target={tid}: [{fx:.1f}, {fy:.1f}, {fz:.1f}]")

    def _handle_target_add_world_impulse(self, address, *args):
        if len(args) >= 4:
            tid, fx, fy, fz = int(args[0]), float(args[1]), float(args[2]), float(args[3])
            self._queue_target_relative_cmd(tid, 'AddWorldImpulse', [fx, fy, fz])
            if self.verbose:
                print(f" [Genesis] AddWorldImpulse Target={tid}: [{fx:.1f}, {fy:.1f}, {fz:.1f}]")

    def _handle_target_add_world_torque(self, address, *args):
        if len(args) >= 4:
            tid, tx, ty, tz = int(args[0]), float(args[1]), float(args[2]), float(args[3])
            self._queue_target_relative_cmd(tid, 'AddWorldTorque', [tx, ty, tz])
            if self.verbose:
                print(f" [Genesis] AddWorldTorque Target={tid}: [{tx:.1f}, {ty:.1f}, {tz:.1f}]")


    def _handle_fetch_metadata(self, address, *args):
        """
        Extracts joint and link names from a URDF file and sends them back to UE.
        Args: [URDFPath] (string)
        """
        if not args: return
        path = args[0]
        
        try:
            # Create a temporary scene to parse the URDF and extract names
            # This is safer than raw XML parsing as it uses Genesis's internal loader
            temp_scene = gs.Scene(show_viewer=False)
            entity = temp_scene.add_entity(gs.morphs.URDF(file=path))
            
            joint_names = [joint.name for joint in entity.joints]
            link_names  = [link.name for link in entity.links]
            
            # Send back to UE (7002)
            self.client_cpp.send_message("/Genesis/Vehicle/JointList", joint_names)
            self.client_cpp.send_message("/Genesis/Vehicle/LinkList", link_names)
            
            # Send ordered wheel list back to UE
            try:
                urdf_parsed = parse_urdf(path)
                wheel_names = [w.name for w in urdf_parsed.wheels]
                self.client_cpp.send_message("/Genesis/Vehicle/WheelNamesList", wheel_names)
                if self.verbose:
                    print(f" [Genesis] Sent wheel names list for {path}: {wheel_names}")
            except Exception as urdf_err:
                print(f" [Genesis] Failed to parse wheels from URDF: {urdf_err}")
            
            if self.verbose:
                print(f" [Genesis] Sent metadata for {path}: {len(joint_names)} joints, {len(link_names)} links.")
                
        except Exception as e:
            print(f" [Genesis] Failed to fetch URDF metadata: {e}")

    def clear_target_forces(self, tid):
        if tid in self.received_data['target_forces']:
            self.received_data['target_forces'][tid] = [0.0, 0.0, 0.0]

    def clear_all_forces(self):
        """Reset all persistent forces for all targets."""
        self.received_data['target_forces'].clear()
        self.received_data['urdf_inputs'].clear()
        self.received_data['input_queue'].clear()
        if self.verbose:
            print(" [Genesis] ClearAllForces: all persistent forces and URDF inputs reset.")

    def _handle_urdf_init(self, address, *args):
        """
        Handles /Genesis/Vehicle/Init
        Args: [URDFPath (str), MappingJSON (str)]
        """
        if len(args) < 2:
            return
            
        path = args[0]
        json_str = args[1]
        
        try:
            mapping = json.loads(json_str)
            self.urdf_init_request = {
                'path': path,
                'mapping': mapping
            }
            if self.verbose:
                print(f" [OSC] URDF Init Received: {path}")
                print(f" [OSC] URDF Mapping JSON: {mapping}")
        except Exception as e:
            print(f" [OSC] Error parsing URDF Mapping JSON: {e}")

    def _handle_urdf_control(self, address, *args):
        """
        Handles /Genesis/Vehicle/Control
        Format: [FrameId (int), ID1 (int), S1, T1, B1, A1, A2, ID2 (int), S2, T2, ...]
        """
        if len(args) < 1:
            return

        frame_id = int(args[0])
        self.received_data['frame_id'] = frame_id
        
        # Performance metrics
        if hasattr(self, 'last_recv_time'):
            now = time.perf_counter()
            self.received_data['recv_gap_ms'] = (now - self.last_recv_time) * 1000.0
            self.last_recv_time = now
        else:
            self.last_recv_time = time.perf_counter()
            self.received_data['recv_gap_ms'] = 0.0

        # Reset detection
        if frame_id == 1 and self.received_data['last_frame_id'] > 10:
            self.received_data['last_frame_id'] = -1
        
        # Multi-target parsing
        # Each vehicle has 6 fields: [ID, Steer, Throttle, Brake, Aux1, Aux2]
        control_payload = args[1:]
        multi_inputs = {}
        
        VEHICLE_FIELDS = 6
        for i in range(0, len(control_payload) // VEHICLE_FIELDS):
            offset = i * VEHICLE_FIELDS
            target_id = int(control_payload[offset])
            inputs = [float(x) for x in control_payload[offset+1 : offset+VEHICLE_FIELDS]]
            multi_inputs[target_id] = inputs

        self.received_data['urdf_inputs'] = multi_inputs
        self.received_data['input_queue'].append({
            'frame_id': frame_id,
            'inputs': multi_inputs
        })
        self.received_data['updated'] = True
        self._new_frame_event.set()  # [LOCKSTEP FIX] Signal main loop that new control data arrived
        
        if self.verbose:
            print(f" [Genesis] VehicleControl Frame={frame_id} Inputs={multi_inputs}")

    def pop_urdf_input(self):
        """
        [CRITICAL FIX FOR NO-LOCKSTEP RACE CONDITION]
        When Unreal Engine's tick rate (e.g. 60Hz~120Hz) is faster than the Genesis
        physics loop (50Hz), data piles up in the queue indefinitely, causing lag.
        Returns only the most recently received control input, while atomically
        swapping the queue so no in-flight packet is lost in the multithreaded
        situation (OSC receiver thread vs physics loop thread).
        """
        # [RACE CONDITION FIX] Using .clear() has a Python list thread-safety problem:
        # a 2nd Unreal packet (e.g. Set Steering) arriving in that split second would evaporate.
        # So we take the accumulated list wholesale and overwrite with a fresh empty list
        # (perfectly atomic thanks to the GIL).
        queue_copy = self.received_data['input_queue']
        if queue_copy:
            self.received_data['input_queue'] = []
            return queue_copy[-1] # Use only the newest (last) packet from the grabbed batch
        return None

    def pop_relative_cmds(self):
        """Safely pops all queued relative commands (Impulses) atomically to prevent Race Conditions."""
        cmds = self.received_data.get('target_relative_cmds', {})
        if cmds:
            self.received_data['target_relative_cmds'] = {}
            return cmds
        return {}

    def pop_overrides(self):
        """Safely pops all overrides (Teleports) atomically."""
        ovrds = self.received_data.get('target_overrides', {})
        if ovrds:
            self.received_data['target_overrides'] = {}
            return ovrds
        return {}

    def pop_obstacle_overrides(self):
        """Safely pops all obstacle overrides atomically."""
        ovrds = self.received_data.get('obstacle_overrides', {})
        if ovrds:
            self.received_data['obstacle_overrides'] = {}
            return ovrds
        return {}

    def wait_for_next_frame(self, timeout=5.0):
        """
        [LOCKSTEP FIX] Blocks until a new control frame is received from UE.
        
        Uses threading.Event instead of polling frame_id comparison.
        This fixes the critical deadlock where C++ bLockStep=false causes frame_id
        to always be 0, making the old `frame_id > last_frame_id` check (0 > 0)
        fail permanently after the first frame.
        
        The Event is set by _handle_urdf_control (on every new control packet)
        and _handle_control (on stop/reset commands), ensuring the main loop
        unblocks regardless of whether frame IDs increment or stay at 0.
        """
        start_time = time.time()
        while True:
            remaining = timeout - (time.time() - start_time)
            if remaining <= 0:
                return None
            
            got_event = self._new_frame_event.wait(timeout=min(remaining, 1.0))
            if got_event:
                self._new_frame_event.clear()
                # Update tracking for diagnostics
                self.received_data['last_frame_id'] = self.received_data['frame_id']
                return self.received_data

    def send_step_ack(self, frame_id):
        """
        Sends an acknowledgment back to Unreal that a simulation step is complete.
        """
        self.client_cpp.send_message("/Genesis/Step/Ack", [int(frame_id)])
        if self.verbose:
            print(f" [OSC] Sent Step Ack for Frame {frame_id}")




    def _handle_sync_reset(self, address, *args):
        """
        Handles /Genesis/Sync/Reset
        Explicitly zeroes out frame counters to sync with a fresh Unreal 'Play' session.
        """
        print(f" [Genesis] [Sync] Received Reset Signal from Unreal. Zeroing Frame IDs (last_id=-1, next_id=0).")
        self.received_data['frame_id'] = 0
        self.received_data['last_frame_id'] = -1

    def _handle_add_world_torque(self, address, *args):
        if len(args) >= 3:
            tx = float(args[0])
            ty = float(args[1])
            tz = float(args[2])
            self._queue_target_relative_cmd(0, 'AddWorldTorque', [tx, ty, tz])
            if self.verbose: print(f" [Genesis] AddWorldTorque [{tx:.1f}, {ty:.1f}, {tz:.1f}]")

    def _handle_physics_settings(self, address, *args):
        """
        Handle physics settings: Gravity, dt, Friction, Viscosity, Restitution, etc.
        Expected Args: [Gravity(float), dt(float), Friction(float), Viscosity(float), Restitution(float)]
        """
        if len(args) >= 1:
            # Update shared data holder
            # We add a 'physics_settings' key to received_data
            if 'physics_settings' not in self.received_data:
                self.received_data['physics_settings'] = {}
            
            # Simple mapping based on index
            # This requires agreement on the param order with UE
            # 0: Gravity, 1: dt, 2: Friction, 3: Viscosity, 4: Restitution
            settings = {}
            if len(args) > 0: settings['gravity'] = float(args[0])
            if len(args) > 1: settings['dt'] = float(args[1])
            if len(args) > 2: settings['friction'] = float(args[2])
            if len(args) > 3: settings['viscosity'] = float(args[3])
            if len(args) > 4: settings['restitution'] = float(args[4])
            
            self.received_data['physics_settings'].update(settings)
            self.received_data['updated'] = True
            if self.verbose:
                print(f" [Genesis] Received Settings: {settings}")
                

    def send_transform(self, location, rotation):
        """
        Sends Genesis transform to Unreal.
        location: [x, y, z] (Genesis coordinates)
        rotation: [w, x, y, z] (Genesis quaternion)
        """
        # Convert Coordinate System (Genesis Z-up -> Unreal Z-up, but check scale)
        # Genesis (m) -> Unreal (cm)
        # Explicitly cast to float() to avoid numpy type errors in python-osc
        ue_loc = [float(location[0] * 100.0), float(location[1] * -100.0), float(location[2] * 100.0)]  # Y flipped for LH
        
        # Rotation conversion (LHS vs RHS coordinate sync)
        # Genesis (RHS) [w, x, y, z] -> Unreal (LHS) [x, y, z, w]
        # In a mirrored Y-system (x, -y, z), the visual 'sense' of rotation is preserved 
        # by negating the X and Z components of the quaternion: (-qx, qy, -qz, qw).
        # rotation[0]=w, [1]=x, [2]=y, [3]=z
        ue_quat = [float(-rotation[1]), float(rotation[2]), float(-rotation[3]), float(rotation[0])] # [-x, y, -z, w]
        # Simple pass-through for now, user can adjust in UE

        # Send to Cpp Actor (GenesisBridge - 7002)
        self.client_cpp.send_message(f"/{self.subject_name}/Location", ue_loc)
        self.client_cpp.send_message(f"/{self.subject_name}/Rotation", ue_quat)

        # Debug print every 100 packets
        if self.verbose:
            if not hasattr(self, '_count'): self._count = 0
            self._count += 1
            if self._count % 100 == 0:
                print(f" [Genesis] Sent 100 packets... Current Loc: {ue_loc}")

    def send_dynamic_transform(self, obs_id, location, rotation):
        """
        Sends dynamic obstacle transform back to Unreal.
        """
        ue_loc = [float(location[0] * 100.0), float(location[1] * -100.0), float(location[2] * 100.0)]
        ue_quat = [float(-rotation[1]), float(rotation[2]), float(-rotation[3]), float(rotation[0])]
        
        # Packing: [ID, Px, Py, Pz, Qx, Qy, Qz, Qw]
        payload = [int(obs_id), ue_loc[0], ue_loc[1], ue_loc[2], ue_quat[0], ue_quat[1], ue_quat[2], ue_quat[3]]
        
        self.client_cpp.send_message("/Genesis/Dynamic/State", payload)

    def send_dynamic_states_bulk(self, states_list):
        """
        Sends [ID, Px, Py, Pz, Qx, Qy, Qz, Qw, ID, Px,...] in one bulk packet.
        Chunked to prevent UDP MTU overflow.
        """
        if not states_list:
            return
            
        payload = []
        for o_id, location, rotation in states_list:
            ue_loc = [float(location[0] * 100.0), float(location[1] * -100.0), float(location[2] * 100.0)]
            ue_quat = [float(-rotation[1]), float(rotation[2]), float(-rotation[3]), float(rotation[0])]
            payload.extend([int(o_id), ue_loc[0], ue_loc[1], ue_loc[2], ue_quat[0], ue_quat[1], ue_quat[2], ue_quat[3]])
            
        # UDP MTU is ~1500 bytes. Each float is 4 bytes.
        # Let's chunk every 30 objects = 240 elements = 960 bytes (very safe)
        CHUNK_SIZE = 240 
        for i in range(0, len(payload), CHUNK_SIZE):
            chunk = payload[i : i + CHUNK_SIZE]
            chunk.append(-1)
            self.client_cpp.send_message("/Genesis/Dynamic/StateBulk", chunk)

    def send_target_states_bulk(self, target_list):
        """
        Sends [ID, Px, Py, Pz, Qx, Qy, Qz, Qw, NumWheels, Wheel0Px, ... ID, Px,...] for ALL targets in one bulk packet.
        Address: /Genesis/Vehicle/TargetBulk
        Each element in target_list can be:
          - (tid, location, rotation)
          - (tid, location, rotation, wheel_transforms_list)
        """
        if not target_list:
            return

        # [PERF v1.0.13] Vectorized UE-frame conversion: gather every chassis +
        # wheel pose, do ONE numpy pass (scale + axis flips + quat reorder),
        # then assemble the payload from plain-python .tolist() rows. The old
        # loop cast ~8 numpy scalars through float() per pose (30 targets x
        # 10 wheels ≈ 2,600 casts per send, twice per step). Wire format is
        # byte-identical (OSC packs float32 either way).
        import numpy as _np
        tids, locs, rots, n_wheels = [], [], [], []
        w_locs, w_rots, w_omegas = [], [], []
        for item in target_list:
            if len(item) == 4:
                tid, location, rotation, wheels = item
            else:
                tid, location, rotation = item
                wheels = []
            tids.append(int(tid))
            locs.append(location)
            rots.append(rotation)
            n_wheels.append(len(wheels))
            for w_tuple in wheels:
                w_locs.append(w_tuple[0])
                w_rots.append(w_tuple[1])
                w_omegas.append(float(w_tuple[2]) if len(w_tuple) > 2 else 0.0)

        loc_a = _np.asarray(locs, dtype=_np.float64) * (100.0, -100.0, 100.0)
        rot_a = _np.asarray(rots, dtype=_np.float64)
        quat_a = _np.stack([-rot_a[:, 1], rot_a[:, 2], -rot_a[:, 3], rot_a[:, 0]], axis=1)
        loc_l, quat_l = loc_a.tolist(), quat_a.tolist()
        if w_locs:
            wl_a = _np.asarray(w_locs, dtype=_np.float64) * (100.0, -100.0, 100.0)
            wr_a = _np.asarray(w_rots, dtype=_np.float64)
            wq_a = _np.stack([-wr_a[:, 1], wr_a[:, 2], -wr_a[:, 3], wr_a[:, 0]], axis=1)
            wl_l, wq_l = wl_a.tolist(), wq_a.tolist()

        # [v1.1.6] CHUNKED sends: a single datagram holding ALL targets bursts
        # the 64 KB UDP limit at ~180 tanks (each 10-wheel target ≈ 90 args ≈
        # 450 B encoded; 200 targets ≈ 71 KB → OSError "Message too long" and
        # the server died). Split into packets of ≤ _BULK_CHUNK_TARGETS
        # targets, each self-contained and ending with the -1 sentinel — the
        # client parses per-packet [ID, ...] records, so chunking is
        # wire-compatible (states of one frame just arrive in a few packets).
        _BULK_CHUNK_TARGETS = 120     # ≈54 KB worst-case (10 wheels/target)
        payload = []
        wi = 0
        chunk_count = 0
        for i, tid in enumerate(tids):
            payload.append(tid)
            payload.extend(loc_l[i])
            payload.extend(quat_l[i])
            m = n_wheels[i]
            payload.append(m)
            for _ in range(m):
                payload.extend(wl_l[wi])
                payload.extend(wq_l[wi])
                payload.append(w_omegas[wi])
                wi += 1
            chunk_count += 1
            if chunk_count >= _BULK_CHUNK_TARGETS:
                payload.append(-1)    # sentinel closes this packet
                self.client_cpp.send_message("/Genesis/Vehicle/TargetBulk", payload)
                payload = []
                chunk_count = 0

        if payload:
            # Add sentinel to prevent OOB
            payload.append(-1)
            self.client_cpp.send_message("/Genesis/Vehicle/TargetBulk", payload)

    def send_observation(self, obs_array):
        """
        Sends Genesis observation tensor to Unreal.
        obs_array: list of floats or numpy array (usually 25 dims)
        """
        # Convert to list of python floats explicitly to avoid OSC packing errors
        obs_list = [float(val) for val in obs_array]
        
        # Send via C++ Observation client (Port 7004)
        self.client_obs.send_message("/Genesis/State/Observation", obs_list)

    def send_observation_schema(self, labels):
        """
        Sends the semantic names for the observation tensor to Unreal. (Run once)
        labels: list of strings (e.g. ['GravityX', 'Steering_L', ...])
        """
        # python-osc can pack multiple strings in an array natively.
        # Send via C++ Observation client (Port 7004)
        self.client_obs.send_message("/Genesis/State/ObservationSchema", labels)
        if self.verbose:
            print(f" [Genesis] Sent Observation Schema ({len(labels)} parameters)")

    def get_received_data(self):
        if self.received_data['updated']:
            self.received_data['updated'] = False
            return self.received_data
        return None

    def get_active_forces(self):
        """Returns currently active persistent forces without clearing them.
        Used by the sim loop to apply F*dt/mass every step (UE AddForce semantics).
        Returns dict with 'local' and 'world' keys, each a [fx, fy, fz] list or None."""
        return {
            'local': self.received_data.get('active_local_force', None),
            'world': self.received_data.get('active_world_force', None),
        }


    def close(self):
        if hasattr(self, 'server'):
            self.server.shutdown()
            self.server.server_close()
            
        print(" [Genesis] Manager Closed.")
