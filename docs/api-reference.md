# API Reference

Full public surface. Imported as `from genesis_vehicle import <name>` unless
otherwise noted. For the mental model behind these names, read
[`concepts.md`](concepts.md) first.

> **Runnable references:** the scripts under [`../samples/`](../samples/)
> exercise the most common call patterns:
> [`quickstart.py`](../samples/quickstart.py) (the minimum-viable API),
> [`slope_hold.py`](../samples/slope_hold.py) (StaticFrictionLock behavior),
> [`batched_rollout.py`](../samples/batched_rollout.py) and
> [`perf_vectorization.py`](../samples/perf_vectorization.py) (the
> `n_envs > 1` shapes and the throughput benefit of batching), and
> [`road_loop.py`](../samples/road_loop.py) (multi-vehicle visual demo
> with 4 distinct kinds driving a closed loop).

## 0. `VehicleScene` — the unified entry point (recommended)

The high-level API that owns the Genesis scene(s), the registered vehicles and
static bodies, and the per-step loop — no manual `gs.init` / `scene.build` /
`scene.step` / `sensor.read`. It wraps `VehiclePhysics` (§1) internally. Full
notes in [`dual-scene-raycast.md`](dual-scene-raycast.md). **Start here and drop to
the low-level `VehiclePhysics` / `MultiVehiclePhysics` (§1) only for control it
doesn't expose** — see the two-API-layers guide in
[`concepts.md`](concepts.md#start-here-the-two-api-layers).

**Backends — physics vs renderer.** The **physics** backend (CPU/GPU compute) is
process-global and set ONCE — it is NOT a constructor arg. Call
`VehicleScene.init_backend("cpu" | "gpu")` *before* constructing any scene (default
**cpu**); a second call (or any double-init) warns and is ignored. The
**renderer** is separate — viewer/cameras rasterize on the **GPU** regardless of
the physics backend, so physics-CPU + GPU-render is valid; there is no CPU-render
mode. No GPU present → software render + a `build()` warning.

```python
class VehicleScene:
    @staticmethod
    def init_backend(backend="cpu") -> None    # PHYSICS backend, once; "cpu"|"gpu"

    def __init__(*, n_envs=1, dt=1/200,
                 raycast_mode="dual_scene",        # "dual_scene" (default) | "single_scene"
                 solver="batched",                 # "batched" (default) | "per_vehicle"
                 gravity=(0,0,-9.81), substeps=4,
                 sim_options=None, rigid_options=None, vis_options=None,
                 viewer_options=None,
                 view=None,                         # None headless | "native" | "cv2"
                 show_viewer=False,
                 wheel_render_mode="auto",          # "auto" | "instanced" | "internal_sync"
                 init_genesis=True)

    # --- registration (before build) ---
    def add_vehicle(urdf_path, preset=None, *, pos=(0,0,1), quat=None,
                    material=None, surface=None, vis_mode=None,
                    stability="control", name=None, raycaster_max_range=20.0,
                    cfg=None, morph=None) -> Vehicle
    #   preset (fn→cfg) OR a pre-built cfg=; and a morph= the VehicleScene
    #   builds into an entity internally (custom material/surface, e.g. the L3
    #   server) OR built from urdf_path. urdf_path always gives the wheel
    #   positions.
    #   The URDF is ALWAYS made ray-wheel ready first (not a knob, v1.1.23):
    #   wheel colliders become render-only, a suspension attach point off the
    #   wheel centre is corrected, missing <inertial>s injected. The original
    #   file is never modified; a compliant URDF is used as-is (no copy).
    #   See physics-contracts.md §7.9 / genesis_vehicle.urdf_prep.
    def add_static(*, morph=None, wheel_raycast_morph=None, collision_morph=None,
                   collision=True, material=None, surface=None, vis_mode=None,
                   name=None) -> StaticBody
    #   a static body is ALWAYS a wheel-raycast target (no raycast= toggle).
    def add_ground_plane(*, friction=0.85) -> StaticBody
    def add_dynamic(morph, *, physics=True, wheel_raycast=False,   # collide-only by default;
                    material=None, surface=None, vis_mode=None,    # set wheel_raycast=True only
                    mass=None, name=None) -> DynamicBody           # for a surface wheels must sense
    def add_camera(*, res=(1280,720), pos=(3,-3,2), lookat=(0,0,0), up=(0,0,1),
                   fov=50.0, GUI=False, name=None, **kw) -> Camera   # any view mode

    def build(*, env_spacing=None, n_envs_per_row=None,
              center_envs_at_origin=None) -> None
    #   env_spacing / n_envs_per_row / center_envs_at_origin → grid the L3 batch
    #   (forwarded to BOTH scenes, identical layout). dual_scene builds the raycast
    #   scene first + the main scene (viewer, if any) last so the native viewer's
    #   GL context stays current.
    def step() -> None
    def reset() -> None
    def mark_config_dirty() -> None      # force the batched solver to re-group next step

    # --- accessors (the raw Genesis scenes are PRIVATE — not exposed) ---
    viewer                               # native viewer | None  (property)
    rigid_solver                         # n_geoms / n_links / faces_info (read-only)
    sim_options                          # runtime dt / gravity tweaks
    is_dual_scene: bool                  # raycast_mode == "dual_scene"
    physics                              # batched MultiVehiclePhysics (solver="batched"), else None
    cameras: list[Camera]                # property
    vehicles: list[Vehicle]              # property
    statics: list[StaticBody]            # property
    dynamics: list[DynamicBody]          # property

class Vehicle:                           # handle returned by add_vehicle
    def set_inputs(throttle=0.0, brake=0.0, steer=0.0) -> Vehicle
    def get_pos() / get_quat() / get_vel() / get_ang()   # (n_envs, ...) main-scene truth
    distances                            # property: last (n_envs, n_wheels) wheel-ground d
    entity_main, physics, sensor, cfg, proxy  # underlying objects

class StaticBody:                        # handle returned by add_static
    name; is_static; has_collision; has_raycast
    entity_main                          # rigid collision entity (main scene)
    entity_raycast                       # kinematic raycast entity (raycast scene; dual_scene)

class DynamicBody:                       # handle returned by add_dynamic
    name; is_dynamic; has_raycast
    entity_main                          # rigid body in the main scene (physics)
    entity_raycast                       # synced raycast target in the raycast scene (dual_scene)
    def set_pose(pos=None, quat=None)    # move a user-controlled body (raycast target follows)
```

`raycast_mode="dual_scene"` (default) raycasts static terrain in a separate scene
(BVH built once, shared across envs); `"single_scene"` is the classic one scene.
The legacy names `"raywheel"`/`"inline"` and `"split"`/`"single"` are accepted as
aliases for `"dual_scene"`/`"single_scene"`. Use `collision_morph` to give a
coarse/convex collider while raycasting a detailed surface (non-convex meshes are
convexified for collision, so a single_scene rigid-mesh raycast hits the convex bulge
— the dual_scene kinematic raycast stays exact). A static body is always a
wheel-raycast target; use `wheel_raycast_morph` to raycast a detailed surface
while colliding against `collision_morph`. `add_dynamic` adds a moving body that
is collide-only by default — pass `wheel_raycast=True` only for a moving surface
the wheels must sense/drive onto (ramp, moving platform); on a non-primitive
(mesh) morph that logs a warning, since its mirror BVH re-fits every step
(prefer a primitive Box/Sphere/Cylinder collider). Scope: one or more vehicles
(L2), L3 (`n_envs >= 1`), static terrain/mesh targets, and dynamic raycast
targets (`add_dynamic`).

### 0.1 Parameters per method

**`VehicleScene(...)`**

| param | default | meaning |
|---|---|---|
| `n_envs` | `1` | L3 batch size (parallel envs) |
| `dt` | `1/200` | sim step time (s) |
| `raycast_mode` | `"dual_scene"` | `"dual_scene"` (separate static-BVH raycast scene) / `"single_scene"` (one scene). Aliases: `raywheel`/`split`, `inline`/`single` |
| `solver` | `"batched"` | `"batched"` (one `MultiVehiclePhysics`, same-kind vehicles grouped) / `"per_vehicle"` (one `VehiclePhysics` each) |
| `gravity` | `(0,0,-9.81)` | world gravity |
| `substeps` | `4` | engine solver substeps per `step()` |
| `sim_options` / `rigid_options` / `vis_options` | `None` | inject Genesis option objects (else built from the args above) |
| `viewer_options` | `None` | native-viewer config — `gs.options.ViewerOptions(camera_pos, camera_lookat, camera_fov, res, max_FPS, refresh_rate, …)`. Main scene only (the raycast scene is never shown). Needs `view="native"` to actually open a window |
| `view` | `None` | `None` headless / `"native"` (Genesis viewer) / `"cv2"` (render cameras for a cv2 HUD) |
| `show_viewer` | `False` | back-compat alias for `view="native"` |
| `wheel_render_mode` | `"auto"` | how wheel visuals reach the Genesis renderer when the scene renders. Whenever a viewer/camera is present, wheel visuals are ALWAYS active (there is no off switch — headless runs simply have nothing to render). `"auto"` (default) uses the **instanced renderer** (v1.1.17) when supported (`n_envs == 1`): wheel poses are computed closed-form and drawn through Genesis's external render-node channel (the same machinery as the debug-draw overlays) — **the wheels are NOT updated through the Genesis rigid solver**, so physics is bit-identical to a headless run; expect only a slight per-step cost for the pose streaming (~2–3 ms at 30 vehicles, CPU). `"instanced"` forces that path (raises when unsupported); `"internal_sync"` forces the legacy `WheelJointInternalSync` fallback (drives solver joints — small physical side effects, clamped since v1.1.16), which is also what `n_envs > 1` rendered scenes fall back to automatically |
| `init_genesis` | `True` | auto-init the (cpu-default) physics backend if not already up; set `False` to manage `gs.init` yourself. **The backend itself is chosen by `VehicleScene.init_backend()`, not here** |

> **Physics backend** is set by the classmethod `VehicleScene.init_backend("cpu" \| "gpu")` (process-global, once, default **cpu**) BEFORE any scene — not a constructor arg. The renderer is separate (always GPU). See "Backends" above.

**`add_vehicle(urdf_path, preset=None, *, …)`** — registers a driven vehicle (always collides + always wheel-raycast).

| param | default | meaning |
|---|---|---|
| `urdf_path` | — | URDF; always parsed for wheel positions |
| `preset` | `None` | preset fn → cfg. Pass this **or** `cfg=` |
| `cfg` | `None` | pre-built `VehicleConfig` (e.g. server `build_cfg`) |
| `morph` | `None` | entity morph built internally (custom collider/visual); else built from `urdf_path` |
| `pos` / `quat` | `(0,0,1)` / `None` | spawn pose when `morph` is not given |
| `material` / `surface` / `vis_mode` | `None` | passed to the main-scene `add_entity` |
| `stability` | `"control"` | stability profile passed to `preset` |
| `raycaster_max_range` | `20.0` | wheel ray max length (m) |
| `name` | `None` | handle label |

**`add_static(*, …)`** — a body that never moves; **always a wheel-raycast target**.

| param | default | meaning |
|---|---|---|
| `morph` | `None` | one morph for both collision + raycast |
| `collision_morph` | `None` | coarse/convex collider (overrides `morph` for collision) |
| `wheel_raycast_morph` | `None` | detailed surface the wheel rays hit (overrides `morph` for raycast) |
| `collision` | `True` | build a main-scene rigid collider (a `convexify=False` mesh > 1000 faces is **refused** — see the mesh guard below; see matrix for the `False` + single_scene caveat) |
| `material` / `surface` / `vis_mode` | `None` | passed to `add_entity` |
| `name` | `None` | handle label |

**`add_dynamic(morph, *, …)`** — a moving body; **collide-only by default**.

| param | default | meaning |
|---|---|---|
| `morph` | — | the body geometry |
| `physics` | `True` | `True`: free rigid (moves under physics). `False`: fixed base you teleport via `handle.set_pose` (e.g. UE-driven) |
| `wheel_raycast` | `False` | `True`: wheels can sense / drive onto it (adds a synced raycast mirror in dual_scene). On a non-primitive morph this logs a re-fit-cost warning |
| `mass` | `None` | override mass (applied after build; only meaningful for `physics=True`) |
| `material` / `surface` / `vis_mode` | `None` | passed to `add_entity` |
| `name` | `None` | handle label |

**`add_ground_plane(*, friction=0.85)`** — convenience for `add_static(morph=Plane(...))`.

### 0.2 Parameter → behavior matrix

What each registration produces. **Collides** = the chassis physically collides
with it in the main scene; **Wheels sense** = the wheel rays detect it as ground;
**Mirror** = where the wheel-raycast target lives.

**`add_static`** (the `collision` flag × `raycast_mode`):

| `collision` | `raycast_mode` | main scene | raycast scene | collides | wheels sense |
|---|---|---|---|---|---|
| `True` (default) | `dual_scene` | rigid (`collision_morph`) | kinematic mirror (`wheel_raycast_morph`) | ✅ | ✅ |
| `True` | `single_scene` | rigid — serves both roles | — (same body) | ✅ | ✅ |
| `False` | `dual_scene` | — (none) | kinematic mirror | ❌ | ✅ |
| `False` | `single_scene` | ⛔ **`ValueError`** (since v1.0.7) | — | — | — |

**`add_raycast_surface(morph, *, surface=None, name=None)`** (v1.0.10) is the
first-class spelling of the `collision=False` row: a wheel-raycast-ONLY
surface — kinematic `use_visual_raycasting` body in the raycast scene, zero
rigid-solver presence (no collision geoms / FK / broadphase pairs, build-once
BVH, mesh-guard exempt). Prefer it over `add_static(collision=False, ...)`
for roads/terrain the wheels drive on but the chassis need not collide with
(the server's `--road-raycast-only` path uses it). dual_scene only — raises
the same single_scene `ValueError`.

⚠️ **single_scene caveats:** `collision=False` is a **dual_scene-only** feature
(the kinematic `use_visual_raycasting` surface lives in the raycast scene) — in
single_scene the wheel rays only hit rigid collision geoms, so a no-collision
static would be a **fall-through** surface. Since v1.0.7 `add_static` **fails
fast** with a `ValueError` (and a `[genesis_vehicle:single-scene]` error log);
pre-1.0.7 it warned and built a rigid from the raycast morph, which (the rco
road morph itself carries `collision=False`) had zero collision geoms and was
invisible to the raycaster. Likewise `wheel_raycast_morph` (a detailed raycast
surface separate from the collider) needs the two bodies of dual_scene, so it is
**ignored in single_scene and logs a warning** — the rigid collider is also the
raycast target (the wheels ride its convex bulge, not the true surface).

⚠️ **Non-convex mesh guard (rigid collider / raycast target):** any rigid body
built from a `gs.morphs.Mesh` with `convexify=False` and **> 1000 faces**
(`_MAX_NONCONVEX_COLLISION_FACES`) is *refused* — `_guard_collision_mesh` raises
a `ValueError` and logs a `[genesis_vehicle:mesh-guard] >>> REVIEW THIS MESH <<<`
error. As a *rigid* body it is expensive **two** ways, either of which can crash
the process (under WSL, the whole VM): (1) an SDF/collision build over every
face, and (2) in `single_scene`, a **per-step wheel-raycaster BVH re-fit** over
every face (the vehicle moves → the BVH is never static). Note (2) fires **even
with a large `sdf_cell_size`** — that only caps the SDF grid (1), not the raycast
re-fit, which is why a terrain that set `sdf_cell_size=10000` could still take the
VM down. Applies to **every rigid path**: `add_static` (main collider, and the
single_scene raycast target) and `add_dynamic` (main body, and the dual_scene
`wheel_raycast` mirror). **Exempt:** primitives & heightfields (not a `Mesh`),
`convexify=True` (convex decomposition keeps both the collider and the raycast
BVH cheap), and `collision=False` kinematic wheel-raycast surfaces (no SDF, and
the BVH is built once — the recommended home for a high-poly mesh). Fix: decimate
the mesh, set `convexify=True`, or register it as `add_static(collision=False)`
(kinematic) in dual_scene.

**Log prefixes.** All `VehicleScene` warnings/errors carry a greppable
`[genesis_vehicle:<slug>]` prefix so an issue class is easy to spot/filter:
`mesh-guard` (mesh refused), `single-scene` (a dual_scene-only option used in
single_scene), `refit-cost` (a non-primitive `wheel_raycast` mirror that re-fits
each step). Grep `\[genesis_vehicle:` for all, or a slug for one class.

**`add_dynamic`** — `physics` sets motion; `wheel_raycast` only controls the
**dual_scene** raycast mirror (see the caveat below):

| `physics` | motion | main scene |
|---|---|---|
| `True` (default) | moves under physics | rigid, free |
| `False` | you teleport via `set_pose` (e.g. UE-driven) | rigid, fixed |

| `wheel_raycast` | `raycast_mode` | raycast scene | wheels sense |
|---|---|---|---|
| `False` (default) | `dual_scene` | — (no mirror) | ❌ |
| `False` | `single_scene` | — (same scene) | ⚠️ **yes** |
| `True` | `dual_scene` | **synced** rigid mirror | ✅ |
| `True` | `single_scene` | — (same scene) | ✅ |

In every case the body **collides** (it is a rigid body in the main scene).

⚠️ **single_scene caveat:** with one scene the wheel rays cast against the whole
main scene, so *every* rigid body is a raycast target regardless of
`wheel_raycast` — the flag can only suppress sensing in `dual_scene` (by not
building the mirror). For "collide but the wheels ignore it" you need
`dual_scene` + `wheel_raycast=False`. Setting `wheel_raycast=True` in
single_scene therefore has no effect and **logs a warning**.

The dual_scene mirror for a `wheel_raycast=True` dynamic body is a *rigid* body in
the raycast scene's rigid solver (re-synced every `step`), so only its own small
BVH re-fits — the heavy static terrain BVH stays untouched. That re-fit cost
scales with the morph's face count, which is why a non-primitive morph warns.

**`add_vehicle`** — always collides (main-scene rigid) and the wheels always
raycast (a proxy + wheel sensor in the raycast scene in dual_scene; the sensor
rides the entity directly in single_scene). Moves under physics.

**Server mapping** (`server/env_builder.py`, from the UE `b_dynamic` field):
`0` static structure/road → `add_static`; `1` physics-dynamic → `add_dynamic(physics=True)`;
`2` UE-driven → `add_dynamic(physics=False)`. Server obstacles are collide-only
(`wheel_raycast` left `False`); roads carry the wheel surface via `add_static`'s
`collision_morph` (convex) + `wheel_raycast_morph` (detailed).

## 1. `VehiclePhysics` — the driver

The low-level driver that `VehicleScene` (§0) wraps. Use it directly when you own
the `gs.Scene` and the step loop, or via `MultiVehiclePhysics` for the L2 batched
solver — otherwise prefer `VehicleScene`. (See the two-API-layers guide in
[`concepts.md`](concepts.md#start-here-the-two-api-layers).)

```python
class VehiclePhysics:
    def __init__(scene, entity, sensor, config: VehicleConfig, n_envs: int = 1)
    def step(inputs: VehicleStepInputs, distances: torch.Tensor | None = None) -> None
    def reset(env_ids: torch.Tensor | None = None) -> None
```

`step(distances=...)` injects externally-measured wheel-ground distances
(shape `(n_envs, n_wheels)`) instead of reading `self.sensor` — the hook
`VehicleScene`'s dual_scene mode uses. `distances=None` reads the sensor as before
(`sensor=None` is then allowed at construction).

### Multi-vehicle (L2 batching, v0.5.11+)

When K vehicles share ONE scene at DIFFERENT positions (traffic,
multi-agent, MPPI candidate visualization), use `MultiVehiclePhysics` —
**not** a hand-rolled Python loop of K separate `VehiclePhysics` objects.
The loop is correct but leaves L2 batching on the table (K compute calls,
K state reads/writes); `MultiVehiclePhysics` groups vehicles by URDF / cfg
identity and batches the compute pipeline per kind. (Exception: if you
need per-vehicle solver ops — independent external forces/impulses, or
per-vehicle teleport mid-rollout — the loop is still legitimate; see
[`batching.md`](batching.md#l2--cross-vehicle-batching-in-one-env).)

```python
class MultiVehiclePhysics:
    def __init__(scene, vehicles: list[tuple[Entity, Sensor, VehicleConfig]])
    def step(inputs_list: list[VehicleStepInputs]) -> None
```

Vehicles of the same kind must share the SAME ``cfg`` instance — group
by passing ``cfg_per_kind[k]`` instead of calling the preset fresh per
vehicle (a fresh cfg per vehicle splits K vehicles into K kinds × 1 and
the batching never engages — measured 10 tanks: SDK compute 33.8 vs
2.8 ms/step). Since v1.0.15 multi-kind steps also batch the solver I/O
across kinds (one combined state read + one force/torque apply pair)
and same-kind WheelJointInternalSync writers collapse into one solver call
(``KindVisualBatch`` — 30 tanks with visuals: 23.3 → 14.2 ms/step);
bounded by Genesis's per-entity ``scene.step()`` cost. For RL/MPPI
throughput use ``n_envs > 1`` instead (L3 batching — see
[`../samples/perf_vectorization.py`](../samples/perf_vectorization.py)).

State (read-only, all `(n_envs, n_wheels)`):

- `omega` — wheel angular velocities
- `last_distances`, `last_compression`, `last_N`
- `last_F_long`, `last_F_lat`, `last_kappa`, `last_alpha`
- `last_T_drive`, `last_T_brake`

Resolved internals:

- `resolved: ResolvedConfig` — post-merge config; the only source of truth at runtime
- `wheel_meta: WheelMeta` — cached per-wheel tensors (positions, radius, side masks, axle indices, etc.)

`step()` accepts either the strategy's `InputType` (e.g. `AckermannInputs`)
or a unified `VehicleInputs`. The latter is auto-converted via
`from_unified()`.

`reset(env_ids=None)` zeroes `omega`, `prev_compression`, and the visual
angle accumulators. Pass a `(K,)` long tensor or a `(n_envs,)` bool mask to
reset only specific environments (RL / MPPI partial reset).

`VehiclePhysics.__init__` prints a one-line version banner the first time
it's called per process. Suppress with `GENESIS_VEHICLE_QUIET=1`.

## 2. Inputs

```python
@dataclass
class VehicleInputs:
    throttle: float | torch.Tensor    # [-1, +1] signed
    brake:    float | torch.Tensor    # [0, 1] positive magnitude
    steer:    float | torch.Tensor    # [-1, +1], + = right turn

@dataclass
class AckermannInputs:           # for Ackermann / PartialAckermann
    throttle, brake, steer
@dataclass
class SkidSteerInputs:           # for SkidSteer + PerSide
    throttle, brake, steer_diff  # + = right turn
@dataclass
class NoSteerInputs:             # for NoSteer + any drivetrain
    throttle, brake

# Convenience alias for VehiclePhysics.step() type hints.
VehicleStepInputs = Union[
    VehicleInputs, AckermannInputs, PartialAckermannInputs,
    SkidSteerInputs, NoSteerInputs,
]
```

Each typed input has a `from_unified(VehicleInputs) -> Self` classmethod for
pipelines that prefer the unified schema.

## 3. Config

```python
from dataclasses import field

@dataclass
class WheelConfig:
    # All fields Optional. None = derive from URDF or module default.
    name, position, side, axle_index
    radius, mass, i_wheel
    rest_stroke, k_susp, c_compression, c_extension, comp_rate_clamp
    mu_long, mu_lat, rolling_resistance_cr
    pb_x, pc_x, pe_x, pb_y, pc_y, pe_y          # Pacejka coefficients
    susp_joint_name, steer_joint_name, spin_joint_name

@dataclass
class ChassisConfig:
    base_link_name: str = "base_link"
    mass: float | None = None              # None = read from URDF
    frame: str = "ISO8855"
    omega_max: float = 100.0
    eps_v: float = 0.5

@dataclass
class VehicleConfig:
    urdf_path: str
    wheels: list[WheelConfig]
    steering, drivetrain, coupling, tire
    chassis: ChassisConfig = field(default_factory=ChassisConfig)
    stability_hooks: list[StabilityHook] = field(default_factory=list)
    recommended_dt: float = 0.025    # 40 Hz SDK default (v1.0.19); `dt` is a deprecated alias
    enable_wheel_joint_internal_sync: bool = False   # auto-managed by VehicleScene.build() (not a user knob)
    susp_visual_clamp: float | None = None   # v0.7.14: None=per-wheel rest_stroke

    @classmethod
    def from_urdf(urdf_path, *, steering, drivetrain, coupling, tire,
                  wheel_overrides: dict[str, WheelConfig] = None,
                  chassis: ChassisConfig = None, **kwargs) -> VehicleConfig
```

> Mutable defaults on dataclass fields (`chassis: ChassisConfig = ChassisConfig()` or `list = []`) are unsafe — they are shared across instances. The SDK uses `field(default_factory=...)` throughout. When constructing your own dataclasses, follow the same pattern.

`from_urdf()` is the easiest path: pass strategies + a dict of per-wheel
overrides keyed by URDF wheel link name; the wheel list is auto-populated.

`resolve(config) -> ResolvedConfig` runs the URDF → user → default merge and
each strategy's `validate()`. Called automatically by `VehiclePhysics.__init__`.

`ConfigError` — raised on bad config (missing required fields, wheel count
mismatch, sides missing for skid-steer, etc.).

## 4. URDF utilities

```python
parse_urdf(urdf_path: str) -> URDFParsedConfig

estimate_spin_inertia_from_genesis(
    entity, link_name: str,
    spin_axis_local: tuple[float, float, float] | None = None,
) -> float
```

**`prepare_vehicle_urdf(urdf_path, **flags) -> str`** (`from genesis_vehicle.urdf_prep import ...`, v1.1.22) — make an arbitrary URDF ray-wheel ready: wheel colliders become render-only (promoted to `<visual>` when that is the wheel's only geometry), a suspension attach point that sits off the wheel centre is corrected, and links missing an `<inertial>` get one. Returns the ORIGINAL path when the file already complies; never modifies it otherwise (temp copy). `VehicleScene.add_vehicle` calls it by default (`prepare_urdf=True`) — see physics-contracts.md §7.9.

`parse_urdf()` is a **convention-based helper** that discovers wheels by
walking the URDF joint tree:

- Wheels = prismatic joints whose name ends with `_susp` or `_suspension_joint`
- Spin joint = first descendant `continuous` joint with axis `(0, ±1, 0)`
- Steer joint = `revolute` joint with axis `(0, 0, ±1)` along the chain
- Side: `l_*` / `_left_*` → `'L'`; `r_*` / `_right_*` → `'R'`; else `None`
- Axle index: clusters by x-coordinate, sorted descending so axle 0 = front-most

> **For non-conforming URDFs**, construct `VehicleConfig` explicitly or supply
> `wheel_overrides` with the relevant joint names. `parse_urdf()` is a
> best-effort convenience for our supported vehicle naming conventions, not a
> general URDF interpreter.

`URDFParsedConfig` fields:

- `base_link_name`, `chassis_mass`, `wheels: list[WheelConfig]`
- `steer_axis_signs: dict[str, int]` — used by the visual layer for `<axis 0 0 -1>` flip
- `susp_has_dynamics: dict[str, bool]` — picks `set_dofs_position` vs `control_dofs_position`

`estimate_spin_inertia_from_genesis()` is a **fallback estimate** consulted
only when `WheelConfig.i_wheel` is not set (by the user OR by the URDF).
When `spin_axis_local` is provided, the helper projects the inertia tensor
onto that axis (`a^T diag(I) a`); otherwise it returns `max(diag(inertial_i))`,
which is the spin MOI for cylindrical wheels but a heuristic for general
shapes. **`WheelConfig.i_wheel` (when supplied) is always authoritative.**

## 5. Strategies

### 5.1 Steering

```python
class Ackermann(SteeringStrategy):
    def __init__(max_steer_rad=0.7, front_axle=0,
                 wheelbase=None, track_width=None)
    # Geometry derived from wheel positions if not supplied.
    InputType = AckermannInputs

class PartialAckermann(Ackermann):
    def __init__(max_steer_rad=0.7, steered_axles=(0,),
                 wheelbase=None, track_width=None)
    InputType = PartialAckermannInputs

class SkidSteer(SteeringStrategy):
    # No steer angle; turning happens via PerSide drivetrain.
    InputType = SkidSteerInputs

class NoSteer(SteeringStrategy):
    InputType = NoSteerInputs
```

### 5.2 Drivetrain

```python
class RWD(DrivetrainStrategy):
    def __init__(t_drive_max, t_brake_max,
                 driven_axles=None,    # default: rear-most axle
                 brake_bias=None)      # default: 60/40 front/rear (2-axle); uniform otherwise
class FWD(RWD):                       # default driven_axles = front-most axle
class AWD(DrivetrainStrategy):
    def __init__(t_drive_max, t_brake_max,
                 drive_weights=None,   # per-wheel weights, default uniform
                 brake_bias=None)
class PerSide(DrivetrainStrategy):
    """For tank-style skid steer.
    Unified-input mapping (ISO +steer = right turn):
        left_cmd  = throttle + steer_gain * steer
        right_cmd = throttle - steer_gain * steer
    So +steer makes the LEFT side faster → right turn."""
    def __init__(t_drive_max, t_brake_max,
                 steer_gain=1.0, omega_max_drive=100.0,
                 throttle_gear_cap=1.0, use_per_side_taper=True)
```

### 5.3 Coupling

```python
class Independent(CouplingStrategy):  # passthrough; the default
class SameSideBelt(CouplingStrategy):
    # Tank track: omega[L] := mean(omega[L]) per env; same for R.
```

`CouplingStrategy.apply()` runs once per step after the per-wheel omega
integration and before visual sync. See
[`physics-contracts.md`](physics-contracts.md#75-coupling-order) for the
one-step lag detail.

### 5.4 Stability hooks (via `stability=` profile)

```python
class RollingResistance(StabilityHook):
    def __init__(tanh_scale=0.5)
    # F_long -= cr * N * tanh(v_long / tanh_scale)
    # cr is per-wheel via WheelConfig.rolling_resistance_cr.

class LowSpeedRegularizer(StabilityHook):
    def __init__(v_kin_com=0.5, ang_kin=0.5,
                 disable_when_control_active=True, control_threshold=0.01)

class StaticFrictionLock(StabilityHook):
    def __init__(brake_thr=0.3, v_thr=0.5, k_spring=500_000.0, k_damp=20_000.0)
    # 2D stick-slip: per-wheel position anchor + spring-damper,
    # projected onto the friction ellipse; omega forced to 0 on locked wheels.
    # Active when brake > brake_thr AND sqrt(v_long^2 + v_lat^2) < v_thr.
```

Hooks are selected via `stability=` profile in the presets (see
[`stability-profiles.md`](stability-profiles.md)), not assembled by hand
in the common path.

## 6. Tire models

```python
class PacejkaAnisotropic(TireModel):
    def __init__(eps_v=0.5)
    # Per-wheel (B, C, E) for long + lat, mu_long, mu_lat come from wheel_params.
    # Returns (F_long, F_lat, kappa, alpha) with friction-ellipse clamp.
class CoulombIsotropic(TireModel):
    def __init__(eps_v=0.5)
    # F = -mu * N * v_slip / |v_slip|, isotropic.
```

## 7. Sensor / dynamics utilities

```python
class WheelRayPattern(genesis.options.sensors.raycaster.RaycastPattern):
    def __init__(positions: list[tuple[float, float, float]])
    @classmethod
    def from_config(resolved: ResolvedConfig) -> WheelRayPattern

read_distances(sensor, n_envs: int) -> torch.Tensor
    # Returns (n_envs, n_wheels). Handles the n_envs=1 sensor shape quirk.

# --- Pure-Python dynamics primitives (testable without Genesis) ---
brake_torque_signed(
    t_brake: torch.Tensor, omega: torch.Tensor, smoothing_scale: float = 0.5,
) -> torch.Tensor
suspension_normal_force(
    compression, comp_rate, k_susp, c_compression, c_extension, air_mask,
) -> torch.Tensor

# --- Version ---
genesis_vehicle.__version__               # str, e.g. "0.6.0"
genesis_vehicle.VERSION_INFO              # tuple, e.g. (0, 6, 0)
genesis_vehicle.version() -> str          # same as __version__
genesis_vehicle.version_info() -> tuple   # same as VERSION_INFO
```

### 7.5 Per-link transforms (telemetry / animation / attach, v0.6.0)

Genesis exposes only each link's **world** pose. `get_link_transforms`
composes per-link transforms in the frame you actually want — typically each
link relative to its **URDF parent** (what an animation rig consumes). Batched
over `n_envs` and `n_links`, no Python per-link loop.

```python
get_link_transforms(
    entity,                       # built Genesis RigidEntity (the vehicle)
    frame: str = "parent",        # "world" | "base" | "parent"
    *, envs_idx=None,
) -> LinkTransforms

# Convenience method bound to the vehicle's entity:
VehiclePhysics.link_transforms(frame="parent", *, envs_idx=None) -> LinkTransforms

@dataclass
class LinkTransforms:
    frame: str                    # the frame pos/quat are in
    names: list[str]              # link names, entity-local index order
    parent_local: list[int]       # each link's parent local idx, -1 if root
    pos: torch.Tensor             # (n_envs, n_links, 3)  (or (n_links,3) single-env)
    quat: torch.Tensor            # (n_envs, n_links, 4)  wxyz
    def matrices() -> torch.Tensor   # (..., n_links, 4, 4) homogeneous
    def index(name: str) -> int      # local index of a link by name
    n_links: int                     # property
```

Frames: `"world"` = raw Genesis output; `"base"` = relative to the entity
base/root link; `"parent"` (default) = relative to each link's immediate URDF
parent (root → world). See [`physics-contracts.md`](physics-contracts.md) for
the quaternion/position conventions.

> **For wheel VISUAL pose, prefer `wheel_visual_transforms` (below), not
> `link_transforms`.** `link_transforms` reads the engine's link state, which
> reflects steering/suspension/spin only when WheelJointInternalSync is ON (it drives those
> joints). Called with WheelJointInternalSync off, it returns the rest pose and emits a
> one-time warning.

### 7.6 Wheel visual pose for external renderers (`wheel_visual_transforms`, v0.7.7)

Closed-form per-wheel VISUAL transform — steer + suspension + spin applied —
computed **without driving Genesis joints**, so it works whether or not
WheelJointInternalSync is enabled. This is the intended feed for an external renderer
(Unreal / Unity): the SDK owns the steer-sign / spin-axis / suspension
conventions, so the client just places the wheel.

```python
VehiclePhysics.wheel_visual_transforms(frame="world", *, envs_idx=None)
    -> (pos, quat)
    # pos  (n_envs, n_wheels, 3)
    # quat (n_envs, n_wheels, 4)  wxyz, includes steer + spin
```

- `frame="local"` — pose **relative to the chassis** (attach the wheel mesh
  under the chassis component and set this as its relative transform → the
  wheel rides the chassis rigidly, no detach). Recommended for hierarchy-based
  renderers.
- `frame="world"` — absolute pose (place the mesh directly).

The quat already carries spin, so a client should NOT additionally rotate the
wheel. Costs a few quaternion ops per wheel (~µs) vs the engine's articulated-
body FK; matches `get_link(wheel)` in the steady regime (the closed-form is
actually cleaner — it has no substep jitter during hard impacts/steer). Assumes
the conventional ray-wheel axes the presets use (steer about chassis +z,
suspension along ±z, spin about the wheel axle +y).

Works for all bundled vehicle classes — **4-wheel cars, 6-wheel trucks
(`PartialAckermann`, front-axle steer), and 10-wheel skid-steer tanks**
(verified vs `get_link`: car/truck ≈ 3.5 mm / 0.04°, tank ≈ 0.6 mm / 0°).
Non-steered wheels get steer 0; the spin component honors
`enable_visual_spin` / the preset's `visual_spin_enabled` (skid-steer/tank
presets disable it — cylindrical road wheels have no visible spin — so the
closed-form omits spin too, matching the viewer).

### 7.7 One-call visual-parts feed (`visual_parts_transforms`, v0.7.8; renamed v0.7.10)

The high-level convenience for an external renderer: chassis **and** wheels in
one call, WheelJointInternalSync-independent. (Named `render_transforms` in v0.7.8–0.7.9;
renamed to `visual_parts_transforms` in v0.7.10.)

```python
VehiclePhysics.visual_parts_transforms(frame="world", *, envs_idx=None) -> VisualPartsTransforms

@dataclass
class VisualPartsTransforms:
    frame: str                  # wheels' frame ("world" | "local")
    chassis_pos: torch.Tensor   # (n_envs, 3)       real dynamics, always world
    chassis_quat: torch.Tensor  # (n_envs, 4) wxyz
    wheel_names: list[str]       # n_wheels (mesh mapping)
    wheel_pos: torch.Tensor      # (n_envs, n_wheels, 3)  closed-form visual pose
    wheel_quat: torch.Tensor     # (n_envs, n_wheels, 4)
```

The chassis is the real **dynamics** pose (`get_pos/get_quat`, always world —
the physical truth, unaffected by WheelJointInternalSync). The wheels are the closed-form
visual pose (`wheel_visual_transforms`). `frame` applies to the wheels:
`"world"` absolute, `"local"` relative to the chassis. This is the recommended
one-stop feed for a UE / Unity bridge — no `get_link`, no WheelJointInternalSync.

**Multi-vehicle (L2 / L2×L3):** `MultiVehiclePhysics` exposes the same two
methods — `wheel_visual_transforms(frame)` returns a **list** (length
`n_vehicles`, caller order) of `(pos, quat)` each `(n_envs, n_wheels, 3/4)`,
and `visual_parts_transforms(frame)` a list of `VisualPartsTransforms`
(per-vehicle, since kinds may differ in wheel count). Verified identical to a
single `VehiclePhysics` at K=1 (Δ = 0).

> **Naming:** the legacy viewer-side wheel-joint driver (internal since v1.1.18) is **`WheelJointInternalSync`** (formerly `VisualJointSync`, renamed v1.1.19)
> (renamed from `VisualSync` in v0.7.8; the old alias was removed in v0.7.9).
> The name is explicit on purpose — it drives the **wheel** visual joints
> (spin/steer/suspension) for the **Genesis viewer** only; it never moves the
> chassis and does not affect physics. External renderers don't need it at
> all — use `visual_parts_transforms` / `wheel_visual_transforms`.

> **Auto-managed by `VehicleScene` (v0.9.18):** `enable_wheel_joint_internal_sync` is
> **not a user-facing knob**. `VehicleScene.build()` sets it automatically — ON iff
> the main scene is rendered by Genesis (`view="native"` **or** a camera was added
> with `vs.add_camera(...)`), OFF otherwise (headless /
> external UE·Unity renderer, the dominant path). It defaults to `False` (was `True`
> through v0.7.13). When driving the low-level `VehiclePhysics` directly you may
> still set it on the config yourself.

> **Perf advisory (v0.7.10):** when `WheelJointInternalSync` is active
> (`enable_wheel_joint_internal_sync=True`) it logs a one-time-per-process
> `[genesis_vehicle] PERF:` warning to stderr — it drives the URDF wheel joints
> through the engine's articulated-body forward kinematics every step and is
> only needed for the **Genesis viewer**. For an external renderer or any
> headless run, set `enable_wheel_joint_internal_sync=False` and read wheel poses from
> `visual_parts_transforms` / `wheel_visual_transforms` (closed-form, ~µs).
> Silence the warning with `GENESIS_VEHICLE_QUIET=1`.
>
> **v0.7.16** cut this cost ≈5× by batching spin + steer + suspension into a
> single `set_dofs_position` call per step (each call triggers a full
> collider/constraint reset + FK pass, so 3 calls = 3 passes). Measured on CPU
> at 1 vehicle the tax dropped from **+4.46 ms/step** (1.53×) to **+0.85 ms/step**
> (1.10×). It is lower, not free — the closed-form path is still ~µs and the
> right choice for headless / external rendering.

## 8. Presets

Every preset takes a keyword-only `stability` argument that picks the
stability profile (see [`stability-profiles.md`](stability-profiles.md)).

```python
def car_4w_fwd_ackermann(
    urdf_path: str, n_envs: int = 1, *, stability: str = "control",
) -> VehicleConfig
def car_4w_rwd_ackermann(
    urdf_path: str, n_envs: int = 1, *, stability: str = "control",
) -> VehicleConfig
def car_4w_awd_ackermann(
    urdf_path: str, n_envs: int = 1, *, stability: str = "control",
) -> VehicleConfig
def truck_6w_partial_ackermann(
    urdf_path: str, n_envs: int = 1, *, stability: str = "control",
) -> VehicleConfig
def tank_10w_skid_belt(
    urdf_path: str, n_envs: int = 1, *, stability: str = "control",
) -> VehicleConfig

stability_hooks_for_profile(
    profile: str = "control", vehicle_kind: str = "car",
) -> list[StabilityHook]
```

| Function | Topology | Steering | Drive | Coupling |
|---|---|---|---|---|
| `car_4w_fwd_ackermann(urdf_path)` | 4 wheels | Ackermann front | FWD | Independent |
| `car_4w_rwd_ackermann(urdf_path)` | 4 wheels | Ackermann front | RWD | Independent |
| `car_4w_awd_ackermann(urdf_path)` | 4 wheels | Ackermann front | AWD | Independent |
| `truck_6w_partial_ackermann(urdf_path)` | 6 wheels | Ackermann on axle 0 | Drive on mid + rear axles | Independent |
| `tank_10w_skid_belt(urdf_path)` | 10 wheels | SkidSteer | PerSide (gear cap 0.3) | SameSideBelt |

Tune by editing the returned config (`cfg.dt = ...`, replace a strategy,
override `cfg.stability_hooks`) before passing it to `VehiclePhysics`.

## 9. Common patterns

### 9.1 Custom 4-wheel car (no preset)

```python
from genesis_vehicle import (
    VehicleConfig, ChassisConfig, WheelConfig,
    Ackermann, RWD, Independent, PacejkaAnisotropic,
    RollingResistance, LowSpeedRegularizer,
)

cfg = VehicleConfig.from_urdf(
    URDF_PATH,
    steering=Ackermann(max_steer_rad=0.6),
    drivetrain=RWD(t_drive_max=1500.0, t_brake_max=3000.0),
    coupling=Independent(),
    tire=PacejkaAnisotropic(),
    wheel_overrides={
        "front_left_wheel":  WheelConfig(c_compression=20_000, c_extension=6_000),
        "front_right_wheel": WheelConfig(c_compression=20_000, c_extension=6_000),
    },
    chassis=ChassisConfig(omega_max=120.0),
    stability_hooks=[RollingResistance(), LowSpeedRegularizer()],
    dt=1/60,
)
```

### 9.2 RL rollout (batched, partial reset)

```python
N_ENVS = 256
physics = VehiclePhysics(scene, car, sensor, cfg, n_envs=N_ENVS)
throttle = torch.zeros(N_ENVS, device='cuda')
brake    = torch.zeros(N_ENVS, device='cuda')
steer    = torch.zeros(N_ENVS, device='cuda')

for t in range(T):
    physics.step(VehicleInputs(throttle, brake, steer))
    scene.step()
    if t % 100 == 0:
        done = some_termination_mask              # (N_ENVS,) bool
        physics.reset(env_ids=done)
```

### 9.3 Custom strategy

```python
import torch
from genesis_vehicle import AWD

class TorqueVectoringAWD(AWD):
    def distribute_torque(self, inputs, omega, wheel_meta, device, dtype):
        T_drive, T_brake = super().distribute_torque(
            inputs, omega, wheel_meta, device, dtype)
        # Per-wheel bias from yaw rate, etc.
        return T_drive, T_brake
```

Same pattern works for `SteeringStrategy.per_wheel_steer`,
`CouplingStrategy.apply`, `TireModel.__call__`, and stability hooks.

## 10. Module-level defaults (from `config.py`)

| Constant | Value | Notes |
|---|---|---|
| `DEFAULT_K_SUSP` | 70_000 N/m | spring stiffness |
| `DEFAULT_C_COMPRESSION` | 14_000 N·s/m | compression damping (HJW direction) |
| `DEFAULT_C_EXTENSION` | 4_000 N·s/m | rebound damping |
| `DEFAULT_REST_STROKE` | 0.10 m | suspension rest length |
| `DEFAULT_MU` | 1.0 | friction (long & lat) |
| `DEFAULT_ROLLING_RESISTANCE` | 0.015 | `Cr` for `RollingResistance` hook |
| `DEFAULT_COMP_RATE_CLAMP` | 30 m/s | numerical clamp on `c_dot` |
| `DEFAULT_RADIUS` | 0.35 m | wheel radius (used only if URDF has no `<cylinder>`) |
| `DEFAULT_MASS` | 20 kg | wheel mass fallback |
| `DEFAULT_I_WHEEL` | 1.5 kg·m² | wheel spin MOI fallback (last-resort) |
| `DEFAULT_PACEJKA` | `PB_X=10, PC_X=1.65, PE_X=0.4, PB_Y=8, PC_Y=1.30, PE_Y=0.4` | mid-grip car defaults |

These are last-resort fallbacks. URDF + `WheelConfig` overrides take
precedence per [`concepts.md`](concepts.md#4-urdf-is-the-default-source-explicit-api-overrides-are-the-final-truth).

## 11. `control` — path following (v1.1.11)

Full guide: [`path-following.md`](path-following.md). Numpy-only (no
Genesis/torch needed at control time).

| Symbol | Import | What it is |
|---|---|---|
| `PathFollower(path, sweep, **tuning)` | `from genesis_vehicle import PathFollower` | path `(x, y, z, speed[, yaw])` → per-step `(throttle, steer, brake)`; the optional per-waypoint `yaw` (rad, world +X = 0, CCW+) pins the desired chassis heading verbatim, else tangential default; `sweep` is a `SweepTable` or a CSV path; cusps (speed-sign flips) handled as stop-and-reverse. `last_mode` ∈ `INIT/DRV±1/STOP/BRAKE_TRANS/DONE` |
| `SweepTable` | `from genesis_vehicle import SweepTable` | measured (v, throttle, steer, pitch, roll) → (a, ω_z) grid; `.load(csv)` / `.save(csv)`; inverse lookups `throttle_for(v, a, pitch, roll)` / `steer_for(v, ω, pitch, roll)` |
| `extract_state(vehicle, env_idx=0)` | `from genesis_vehicle.control import extract_state` | Genesis entity / `Vehicle` wrapper → state dict for `PathFollower.step` |
| `extract_state_from_arrays(pos, quat_wxyz, vel)` | `from genesis_vehicle.control import extract_state_from_arrays` | same, from raw arrays (simulator-agnostic) |
| `sweep_measure` CLI | `python -m genesis_vehicle.control.sweep_measure` | measure the sweep CSV for a (URDF, preset, config) triple; `--gpu` recommended (~500-env L3 batch) |
