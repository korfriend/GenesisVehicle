# Changelog — genesis_vehicle

All notable changes are documented here. Versions follow
[Semantic Versioning](https://semver.org/) (major.minor.patch).

`__version__` lives in `genesis_vehicle/_version.py` and is re-exported as
`genesis_vehicle.__version__` plus the `version()` / `version_info()`
functions. `VehiclePhysics.__init__` prints a one-line banner with the
running version the first time it is instantiated in a process.

---

## [0.9.27] — 2026-06-26

### Changed — `solver="batched"` is the default; server on solver-agnostic accessors (1.0.0 phase 3.1–3.2)

- VehicleScene's default solver is now **`"batched"`** (was `per_vehicle`). With
  the dirty-tracked kind grouping (0.9.25), a single vehicle is a batched
  kind-of-1 and K same-preset vehicles batch automatically; correctness is
  identical (quickstart x=+12.49 / 5.02 m/s; `two_scene_terrain --compare`
  |Δx| = 0.000 m). A vehicle-less scene builds no MVP and steps fine (guarded).
- The OSC server (`l3_runtime`, `physics_server`) now reads per-vehicle capture
  through the **solver-agnostic `Vehicle` accessors** (`veh.wheel_visual_transforms()`
  / `veh.resolved`) instead of `veh.physics.*`, so it works under either solver
  (and under the new batched default). `controllers[tid]` and `L3State` hold the
  `Vehicle` handle. 96 pytest.

---

## [0.9.26] — 2026-06-26

### Added — per-vehicle accessors that work in both solver modes (1.0.0 phase 3.0)

`Vehicle` gained `wheel_visual_transforms()`, `visual_parts_transforms()`, a
`resolved` property, and a solver-agnostic `distances` — each reads this vehicle's
own `VehiclePhysics` in `per_vehicle` mode, or delegates to the shared
`MultiVehiclePhysics` + the vehicle's flat slot in `batched` mode. So callers (the
server, samples) never branch on `solver`. `MultiVehiclePhysics` gained
`resolved_list` / `distances_list()` to back the batched delegation. This unblocks
making `batched` the default without breaking per-vehicle capture. 96 pytest.

---

## [0.9.25] — 2026-06-26

### Added — lazy, dirty-tracked kind grouping for the batched solver (1.0.0 phase 2c)

`solver="batched"` now groups vehicles by KIND, so the common pattern —
`add_vehicle(preset=…)` called K times — actually batches. (Previously each fresh
preset cfg was its own kind → `MultiVehiclePhysics` made K kinds-of-1 → no
batching.) The kind key is **registration-based**: `preset` → `(urdf, preset fn,
stability)`; a pre-built `cfg` → that cfg object (pass the same cfg to batch).
VehicleConfig has object fields (hooks/strategies) and is not value-comparable, so
the registration key is the robust choice. Same-kind vehicles share one cfg object
so the MVP batches them.

Grouping is **lazy + dirty-tracked**: `add_vehicle` / `mark_config_dirty()` bump a
config version; the grouping re-runs (and the MVP is rebuilt) only before a `step`
where the version changed — otherwise `step()` pays an O(1) int compare. Verified:
`add_vehicle(preset)` ×3 → **1 kind** (batched); rwd+awd → 2 kinds;
`mark_config_dirty()` re-groups on the next step. 96 pytest.

---

## [0.9.24] — 2026-06-26

### Added — VehicleScene `solver="batched"`, working in dual_scene too (1.0.0 phase 2b)

`VehicleScene` gained `solver="per_vehicle"` (default) | `"batched"`. `"batched"`
runs one `MultiVehiclePhysics` that groups same-kind vehicles (**same cfg object**
→ pass `cfg=shared` to each `add_vehicle`) into one batched compute — faster for
many same-kind vehicles (L2).

It now works in **both** raycast modes. The batched solver
(`MultiVehicleKindPhysics.step` / `MultiVehiclePhysics.step`) gained a
`distances=` injection: in dual_scene `VehicleScene` feeds the raycast-scene
distances into the batched compute (the kinds skip their own sensor read); in
single_scene the solver reads each vehicle's sensor as before. Previously the
batched solver only read its own sensors → single-scene only; injecting the
pre-computed distances lifts that. Verified: `batched` == `per_vehicle` in both
single_scene and dual_scene (`[2.4, 2.4]` either way). `vs.physics` exposes the
MVP in batched mode (per-vehicle `Vehicle.physics` handles otherwise). 96 pytest.

---

## [0.9.23] — 2026-06-26

### Docs — per-env `set_inputs` (1.0.0 phase 2a)

`Vehicle.set_inputs(throttle, brake, steer)` already accepted **per-env tensors of
shape `(n_envs,)`** (it forwards to `VehicleInputs`, whose fields are
`ScalarOrTensor`) — only the signature/docstring implied scalars. Documented it:
each control is a scalar (broadcast to all envs) or a per-env `(n_envs,)` tensor
for L3 RL/MPPI rollouts. Verified through `VehicleScene` (n_envs=4, per-env
throttle → the four envs diverge as expected). No behavior change.

---

## [0.9.22] — 2026-06-26

### Added — VehicleScene rendering API: `view=`, `Camera`, `add_camera()` (1.0.0 phase 1)

First step of the 1.0.0 "VehicleScene owns everything" refactor — so callers never
touch the underlying scene to render:

- `view=None | "native" | "cv2"` (default `None`): `None` headless (no Genesis
  render), `"native"` opens the Genesis viewer, `"cv2"` renders via cameras you add
  (offscreen → e.g. a cv2 window). `show_viewer=True` is kept as a back-compat
  alias for `view="native"`.
- A `Camera` handle (SDK wrapper around a Genesis camera) + `vs.add_camera(...)`
  (works in **every** view mode) + the `vs.cameras` property. Render frames with
  `cam.render()`, aim with `cam.set_pose(...)`.
- A camera (or `view="native"`) auto-enables the wheels' VisualJointSync at build
  (0.9.20), so rendered wheels animate with no extra code.

Additive / non-breaking. **Next phases** (toward 1.0.0): per-env `set_inputs` + a
batched `solver=`, migrate all samples + the server onto `VehicleScene`, then
remove the `main_scene` / `raycast_scene` properties and tag 1.0.0.

---

## [0.9.21] — 2026-06-26

### Fixed — `road_loop` NaN with the Truck kind (+ faster default solver)

- `samples/road_loop.py`: the 5000 kg 6-wheel **Truck** blew the constraint
  forces up to NaN the instant the brake released into the drive phase (the 3
  car kinds were fine) — a pre-existing genesis-1.2.0 instability that the new
  `--native` viewer made visible (cars launching / wheels sinking, then crash).
  Root cause: the truck's stiff suspension + heavy-chassis/light-wheel mass
  ratio is unstable at the coarse internal dt of `substeps=10`. **Raised
  substeps 10 → 30** (measured floor with the truck in the fleet: 20 still NaNs,
  30 is stable; the standalone `GeneVehicle_Truck6w` demo uses 50 for a single
  truck).
- **Default `--solver` `per_vehicle` → `multi_batched`**: batches each kind's
  compute pipeline, much faster for the 16-vehicle fleet — offsets the higher
  substeps (~125 ms/step). `per_vehicle` is still available for the simple
  N-independent-`VehiclePhysics` path.

---

## [0.9.20] — 2026-06-26

### Changed — `enable_visual_joint_sync` is auto-managed (no longer a user knob)

`VisualJointSync` drives the URDF wheel *visual* joints through the engine each
step so **Genesis's own renderer** shows wheels spinning/steering — useful only
when Genesis actually renders (the wheels are decorative; physics is unaffected).
`VehicleScene.build()` now sets `enable_visual_joint_sync` **automatically**: ON iff
the main scene is rendered by Genesis — `show_viewer=True` **or** a Genesis camera
was added (`vs.main_scene.add_camera(...)`) — OFF otherwise.

So it is no longer exposed as something callers set: the samples (`quickstart`,
`slope_hold`) and the server (`l3_runtime`, `physics_server`, `vehicle_builder`)
dropped their manual `cfg.enable_visual_joint_sync = …` lines. A headless /
external-renderer run leaves it off and reads wheel poses closed-form via
`wheel_visual_transforms()` / `visual_parts_transforms()` (forward kinematics, ~µs,
no per-step engine FK); a `--viewer` (cv2 camera) or `--native` (Genesis viewer)
sample turns it on with no extra code. (Driving the low-level `VehiclePhysics`
directly, you may still set it on the config yourself.) 96 pytest.

---

## [0.9.19] — 2026-06-26

### Added — `--native` (Genesis interactive viewer) option across the visual samples

- The visual samples now take **`--native`** to open the Genesis native
  interactive viewer (orbit / zoom / ESC) instead of the cv2 HUD (`--viewer`):
  `quickstart`, `slope_hold`, `road_loop`, `city_traffic_ego`,
  `multi_env_render`, `batched_rollout` (`terrain_drive` already had it). With
  the 0.9.18 build-order fix the native viewer works for `VehicleScene`
  dual_scene too. `--native` implies no cv2; headless / `--viewer` unchanged.
- `samples/_hud.py`: added `native_viewer_options(camera_pos, camera_lookat, …)`
  (builds `gs.options.ViewerOptions`) and `native_alive(vs)` (viewer-open check),
  the two bits each sample reuses for the native path.

---

## [0.9.18] — 2026-06-26

### Added — `VehicleScene.build()` grid layout + dual_scene native-viewer build order

| Abbr. | Meaning |
|---|---|
| GL | OpenGL render context |
| BVH | Bounding Volume Hierarchy (raycast acceleration tree) |

- **`VehicleScene.build(env_spacing=None, n_envs_per_row=None)`** — forwards the
  L3 grid layout to *both* scenes' `Scene.build` (identical layout, so the
  dual_scene raycast proxy stays aligned with the main-scene vehicle per env).
  Lay a batch out on a grid (e.g. one cell per checkpoint) instead of stacked at
  the origin. No args → unchanged (stacked at origin).
- **dual_scene + native viewer now works.** `build()` builds the sensors-only
  raycast scene **first** and the main scene (which may own the viewer) **last**.
  Genesis creates a GL context per scene; building the raycast scene *after* the
  main one left the raycast context current and crashed the viewer thread
  (`OpenGL ... no valid context` in the pyrender draw). With the main scene built
  last its context stays current, and the per-step
  `raycast_scene.step(update_visualizer=False)` is CUDA-only so it never steals
  the context back. No effect in single_scene / headless (identical output).

---

## [0.9.17] — 2026-06-26

### Docs — carry the "raycast scene never viewed/rendered" statement into the docs

Extends 0.9.16 (code comment + docstring) to the markdown docs:
`two-scene-raycast.md` gains a "The raycast scene is never viewed or rendered"
subsection (sensors-only; `show_viewer=False` always; no camera;
`update_visualizer=False`; only the main scene takes a viewer / `viewer_options`),
and the `api-reference.md` §0 `raycast_scene` property is annotated the same way.

---

## [0.9.16] — 2026-06-26

### Docs — state explicitly that the raycast scene is never viewed or rendered

Made the invariant explicit in the code (a comment at the raycast-scene
construction and the module docstring): the dual_scene raycast scene is
**sensors-only and never views or renders** — `show_viewer=False` always
(independent of `VehicleScene`'s `show_viewer` / `viewer_options`, which apply to
the main scene only), no camera is ever added to it, and its `step()` is always
called with `update_visualizer=False`. Only the main scene can have a viewer.

---

## [0.9.15] — 2026-06-26

### Added — `viewer_options=` on `VehicleScene` (configure the native viewer)

`VehicleScene` exposed `show_viewer` (open/close) and `vis_options` (rendering)
but not the native viewer's own config. Added a `viewer_options=` parameter
(`gs.options.ViewerOptions`: `camera_pos` / `camera_lookat` / `camera_fov` / `res`
/ `max_FPS` / `refresh_rate` / …), passed through to the main scene. `None` keeps
Genesis defaults; pair with `show_viewer=True` to open the window. Only the main
scene takes it (the raycast scene is sensors-only, never shown).

```python
vs = VehicleScene(show_viewer=True,
                  viewer_options=gs.options.ViewerOptions(
                      camera_pos=(8, -6, 4), camera_lookat=(0, 0, 1), max_FPS=60))
```

---

## [0.9.14] — 2026-06-26

### Fixed (docs) — correct the 0.9.13 perf claim; Genesis already no-ops headless render

The 0.9.13 note claimed a ~16.5% raycast-step saving from `update_visualizer=False`.
That was a **benchmarking artifact** (a single non-interleaved GPU run);
interleaved re-measurement shows no real difference. Reading Genesis confirms why:
`Scene.step()` calls `visualizer.update(force=not advance)`, whose body is

```python
if force:                         # only when the sim did NOT advance (paused)
    self.reset()
elif self._viewer is not None:    # only when a native viewer is attached
    self._viewer.update(...)
# else: nothing
```

So a **headless scene (no viewer) during a normal advancing step renders nothing**
— Genesis already skips it; cameras render only on an explicit `cam.render()`,
never inside `step()`. The criterion is "is a viewer attached / is the sim
paused", and Genesis applies it itself. The 0.9.13 `update_visualizer=False` on the
raycast scene is therefore **not a speedup**; it is kept only as explicit intent
(the sensors-only raycast scene must never render) and is a no-op in practice.
`main_scene` needs no such change — when headless it already renders nothing.

## [0.9.13] — 2026-06-26

### Changed — be explicit that the sensors-only raycast scene is never rendered

`VehicleScene` steps the dual_scene raycast scene with `update_visualizer=False`
(both the per-step re-cast in `_measure_distances` and the one-time populate step
in `build`) — it is sensors-only and never user-rendered. The wheel-ray re-cast
still runs inside `sim.step()`, so distances/poses are unchanged
(`two_scene_terrain --compare` matches single_scene, |Δx| = 0.000 m). **Not a
measurable speedup** — see the 0.9.14 correction (Genesis already no-ops a headless
scene's visualizer update). Kept as explicit intent. 96 pytest.

---

## [0.9.12] — 2026-06-25

### Changed — mesh-guard message: name the per-step raycaster BVH re-fit cost

- The `[genesis_vehicle:mesh-guard]` error/log and docstring (and
  `docs/api-reference.md` §0.2) now state that a large non-convex *rigid* mesh is
  expensive **two** ways — the SDF/collision build **and**, in `single_scene`, a
  **per-step wheel-raycaster BVH re-fit over every face** — and that the latter
  fires **independent of `sdf_cell_size`** (which only caps the SDF grid). This is
  why a terrain that set `sdf_cell_size=10000` (small SDF) could still crash the
  VM. No behavior change: same trigger (`convexify=False` + > 1000 faces), same
  exempt cases — message accuracy only.

---

## [0.9.11] — 2026-06-25

### Fixed — dual_scene `vs.build()` crash on a non-"visual" mirror vis_mode

Building a dual_scene `VehicleScene` could crash in Genesis's renderer:

```
AttributeError: 'KinematicEntity' object has no attribute 'geoms'. Did you mean: 'vgeoms'?
  genesis/vis/rasterizer_context.py … on_rigid() → geoms = entity.geoms
```

The kinematic raycast-scene mirrors (`use_visual_raycasting=True`) are visual-only
(they have `vgeoms`, not collision `geoms`). Genesis's `on_rigid` only takes the
`vgeoms` path when `surface.vis_mode == "visual"`; for any other vis_mode it
touches `entity.geoms`, which a `KinematicEntity` lacks. The caller's vis_mode
(e.g. the OSC server's `--vis-mode=collision`) was leaking onto the sensors-only
mirror. On Genesis 1.2.0 the kinematic default is `"visual"`, so it was latent;
the L3 server with `--vis-mode=collision` (or a Genesis build whose kinematic
default differs) hit it at `vs.build()` — the failure the UE team reported.

Fix: the kinematic raycast-scene mirror is now always added with
`vis_mode="visual"` (it is never user-rendered — the external engine renders); the
caller's vis_mode still applies to the main-scene entity. Reproduced on 1.2.0
(`vis_mode="collision"` → same crash, `"visual"` → OK) and regression-tested. 96
pytest.

---

## [0.9.10] — 2026-06-25

### Docs — the two API layers (high-level `VehicleScene` vs low-level `VehiclePhysics`)

The recommendation to use `VehicleScene` was scattered and `concepts.md` (the
mental-model doc) never mentioned it. Added a cohesive explanation:

- `concepts.md` opens with a **"Start here: the two API layers"** section — what
  each layer owns, a decision table, and the rule of thumb (start high-level,
  drop to `VehiclePhysics` / `MultiVehiclePhysics` only for control it doesn't
  expose, e.g. the L2 batched solver or a hand-written step loop).
- `docs/index.md` gains a row pointing to it.
- `api-reference.md` §0 (`VehicleScene`) and §1 (`VehiclePhysics`) cross-link to
  the guide.

---

## [0.9.9] — 2026-06-25

### Added — `obstacles_and_ramp --bench` (dual_scene vs single_scene)

The obstacle sample gained `--bench` (and `--n-envs`) to time the two raycast
modes over the drive loop. Finding on this PRIMITIVE-obstacle course (GPU):

| n_envs | single_scene | dual_scene | dual speedup |
|---|---|---|---|
| 1   | 24.5 ms/step | 26.3 ms/step | 0.93x (≈7% slower) |
| 64  | 46.8 ms/step (0.73 ms/env) | 50.8 ms/step (0.79 ms/env) | 0.92x (≈8% slower) |

dual_scene is **slower** here, and L3 does not rescue it: the obstacles are boxes
(trivial BVH), so single_scene's per-step BVH re-fit is nearly free while
dual_scene pays for a second `scene.step()`, the ramp's synced mirror, and the
proxy sync — and the static-BVH-shared-across-envs benefit is negligible for
primitives. dual_scene's win needs a heavy static **mesh** terrain (see
`two_scene_terrain.py`: 3.40x @256 envs); prefer **single_scene** for
primitive-obstacle scenes.

---

## [0.9.8] — 2026-06-25

### Added — formatted package logging by default

- **`import genesis_vehicle` now attaches a formatted handler to the
  `genesis_vehicle` logger**, so its warnings/errors print as
  `HH:MM:SS LEVEL    genesis_vehicle.<mod> | [genesis_vehicle:<slug>] message`
  out of the box (previously, with no logging config, Python's last-resort
  handler printed the bare message only). Only the `genesis_vehicle` namespace
  is configured and `propagate` is turned **off**, so the root logger / app
  config is untouched and messages are never double-emitted — even if the app
  later calls `logging.basicConfig()`.
- New public **`configure_logging(level=…, fmt=…, datefmt=…, stream=…,
  force=…)`** to change the level/format or redirect the stream; if the logger
  already has handlers it keeps them and only updates the level. Auto-setup can
  be disabled with the env var **`GENESIS_VEHICLE_LOG=0`**.

---

## [0.9.7] — 2026-06-25

### Added — single_scene `collision=False` warning + greppable log prefixes

| Abbr. | Meaning |
|---|---|
| BVH | Bounding Volume Hierarchy (raycast acceleration tree) |
| SDF | Signed Distance Field (rigid mesh collision) |

- **`add_static(collision=False)` in `single_scene` now logs a warning.** With one
  scene the wheel-raycast target *is* a rigid body, so `collision=False` cannot be
  honored — the body still collides. This was a documented caveat but was applied
  silently; it now emits a `[genesis_vehicle:single-scene]` warning pointing to
  `dual_scene` for a true no-collision (kinematic) raycast surface.
- **All `VehicleScene` warnings/errors now carry a greppable
  `[genesis_vehicle:<slug>]` prefix** so an issue class is easy to spot and
  filter. Slugs: `mesh-guard` (non-convex collider refused), `single-scene` (a
  dual_scene-only option used in single_scene), `refit-cost` (a non-primitive
  `wheel_raycast` mirror whose BVH re-fits each step). Grep `\[genesis_vehicle:`
  for all, or a single slug for one class.
- **Docs:** `docs/api-reference.md` §0.2 now documents the 0.9.2 non-convex mesh
  guard (every rigid collision path, with the exempt cases) and the log-prefix
  convention, and notes the single_scene `collision=False` warning.

---

## [0.9.6] — 2026-06-25

### Changed — `road_raycast_only` routing drops the redundant collider

When `road_raycast_only=True` a road mesh is a wheel-raycast surface only (no
chassis collision). The 0.9.3 fix restored the flag but still routed it through
`add_static(collision_morph=<a collision=False mesh>, wheel_raycast_morph=…)`,
which added a useless no-collision rigid to the main scene — doubling road-mesh
memory, exactly what rco exists to avoid on big maps. It now routes as
`add_static(collision=False, wheel_raycast_morph=…)`: the kinematic raycast
mirror only, no main-scene collider. (This completes the team's rco fix in the
direction they intended — a leaner raycast-only road — rather than reverting it.)

`add_static`'s single_scene `wheel_raycast_morph` warning is refined to fire only
when a collision body also exists; the rco case (where `wheel_raycast_morph` is
the sole geometry and so IS the raycast body) no longer warns spuriously.

Tests: a dual_scene rco road has no main rigid (just the mirror); a single_scene
rco road logs no warning. 95 pytest.

---

## [0.9.5] — 2026-06-25

### Added — regression tests for the server mesh-obstacle paths

`tests/test_server_mesh_obstacles.py` (Genesis-gated via `importorskip`; CPU).
The root cause of the 0.9.3 / 0.9.4 bugs was that the 0.9.0 obstacle test only
used primitive boxes, which skip the mesh morph-building branch where both bugs
lived. New tests close that gap:

- `build_obstacles` must keep the `road_raycast_only` parameter (signature guard
  for the 0.9.3 NameError);
- `make_double_sided_mesh` doubles a closed mesh without the removed trimesh API
  (guard for 0.9.4);
- `build_obstacles` registers cleanly across all six mesh paths (road
  raycast-only / full-collision, structures-as-primitive, convex, dynamic,
  UE-driven). 93 pytest (8 new).

---

## [0.9.4] — 2026-06-25

### Fixed — double-sided road-mesh preprocessing on trimesh 4.x

`env_builder.make_double_sided_mesh` called `remove_degenerate_faces()` /
`remove_duplicate_faces()`, which **trimesh ≥ 4 removed** (the installed venv has
4.12.2). The `[Complex]` road path caught the resulting `AttributeError` and fell
back to the *original* (single-sided) mesh — so the double-siding silently never
happened. Switched to the `update_faces(mesh.nondegenerate_faces())` /
`update_faces(mesh.unique_faces())` face-mask API, with a `hasattr` fallback to
the legacy methods on trimesh 3.x.

Verified: a `[Complex]` road mesh is now doubled before CoACD (e.g. a 12-face
cube → 24 faces) instead of falling back; the other mesh-obstacle paths
(`structures_as_primitive`, convex, dynamic, UE-driven) all register cleanly;
85 pytest.

---

## [0.9.3] — 2026-06-25

### Fixed — server road-mesh path (regression from 0.9.0)

- `env_builder.build_obstacles` still referenced `road_raycast_only` in the mesh
  morph-building branch, but the 0.9.0 encapsulation refactor had dropped it from
  the signature — so a **road mesh** (`obs_type == 5` / `[Complex]`) hit a
  `NameError` at runtime. The 0.9.0 routing test only used primitive boxes, which
  never reach that branch, so it slipped through; UE integration surfaced it.
  Restored `road_raycast_only=False` to the signature and `physics_server` passes
  `--road-raycast-only` again (`l3_runtime` uses the default). So
  **`--road-raycast-only` is NOT superseded** (correcting the 0.9.0 note): a road
  can still load as a no-collision kinematic wheel-raycast surface, while the
  default (`False`) routes it through `add_static` (convex `collision_morph` +
  detailed `wheel_raycast_morph`).
- `physics_server` dropped `RigidOptions(prefer_parallel_linesearch=False)` — the
  installed Genesis (1.2.0) rejects that attribute ("Unrecognized attribute"),
  which crashed `VehicleScene` construction in the per-entity server. Latent
  since the pre-SDK server hard-coded it but was never run headless on 1.2.0.

Verified: a road mesh registers under both `road_raycast_only` values with no
NameError; 85 pytest.

---

## [0.9.2] — 2026-06-25

### Added — non-convex collision-mesh guard

| Abbr. | Meaning |
|---|---|
| SDF | Signed Distance Field (rigid mesh collision representation) |
| BVH | Bounding Volume Hierarchy (raycast acceleration tree) |
| OOM | Out Of Memory (process/VM killed) |

- **`VehicleScene` now refuses a large non-convex mesh as a rigid collider.**
  A `gs.morphs.Mesh` with `convexify=False` keeps its full concave geometry for
  collision, so Genesis builds an SDF over every face. Past **1000 faces**
  (`_MAX_NONCONVEX_COLLISION_FACES`) that build explodes in memory and can
  hard-crash the process — under WSL it takes the whole VM down. `add_static` /
  `add_dynamic` now call `_guard_collision_mesh()` before adding any rigid
  collision entity (main-scene collider, single_scene raycast target, and the
  dual_scene dynamic raycast mirror): it `raise`s a `ValueError` with an
  actionable message and logs a `>>> REVIEW THIS MESH <<<` error asking the mesh
  to be decimated, `convexify=True`-d, or moved to a **kinematic wheel-raycast
  target** (`add_static(collision=False)`, which needs no SDF).
- **Exempt** (correctly not blocked): primitives / heightfields (not a `Mesh`),
  `convexify=True` (convex decomposition keeps collision cheap), and
  `collision=False` visual / kinematic raycast surfaces — i.e. the recommended
  home for a high-poly surface, where a big face count is fine.

### Added — `samples/terrain_drive.py` (bumpy-terrain demo on `VehicleScene`)

- Drive a car forever over an x-periodic **undulating mesh terrain** using the
  `VehicleScene` API. Default `dual_scene` registers the terrain as a
  `Kinematic` wheel-raycast target (`add_static(collision=False)`, exact surface,
  BVH built once) plus a collision-free **visual copy** in the main scene for the
  camera. Side-following chase cam, seamless period-wrap, speed governor, ESC to
  quit. `--rigid-terrain` runs the slow `single_scene` rigid-mesh A/B path — at
  full resolution it now trips the mesh-guard above and exits cleanly instead of
  crashing.

---

## [0.9.1] — 2026-06-25

### Added / Changed — samples on the `VehicleScene` API

- **New sample `obstacles_and_ramp.py`** — builds a course entirely with
  `VehicleScene` (`add_ground_plane` + `add_static` with the
  `collision_morph`/`wheel_raycast_morph` split + `add_dynamic` for a collide-only
  box and a `wheel_raycast=True` ramp). Prints the body registry (each body's
  main / raycast entities — `docs/api-reference.md` §0.2 made concrete), then
  drives through and reports. Sample #12.
- **`quickstart.py` and `slope_hold.py` migrated to `VehicleScene`** — they no
  longer call `gs.init` / `gs.Scene` / `scene.build` / `VehiclePhysics` directly;
  the flat / tilted ground is an `add_static` (slope) / `add_ground_plane` body
  and the loop is `veh.set_inputs(...) + vs.step()`. Behaviour is identical
  (quickstart final pose x=+12.49, 5.02 m/s; slope_hold 0.0 mm lateral slip).
- The other samples intentionally stay on the lower-level
  `VehiclePhysics` / `MultiVehiclePhysics` API they exist to demonstrate (L2/L3
  batching, per-env control, solver benchmarks) — that API is unchanged by 0.9.0.

All 12 samples verified running on the new version (GPU); 78 pytest.

---

## [0.9.0] — 2026-06-25

### Changed — full server encapsulation behind `VehicleScene`

The OSC physics server no longer touches a Genesis scene directly — **all**
geometry (floor, vehicles, obstacles, generic targets) is registered through
`VehicleScene.add_*`, and VehicleScene owns the inline-vs-two-scene routing.

- **Per-entity path ported** (`server/physics_server.py`): builds via
  `VehicleScene(raycast_mode="single_scene")` (interacting vehicles, n_envs=1, CPU,
  where the two-scene raycast has no benefit). With the earlier L3 port, **both**
  server paths now run on `VehicleScene`. The server no longer constructs
  `VehiclePhysics` / `VehicleInputs` directly.
- **`env_builder.build_obstacles(vs, …)`** registers every obstacle through
  `add_static` (b_dynamic 0) / `add_dynamic` (1 = physics, 2 = UE-driven). The
  obstacle-mirror gap is closed: all obstacles — not just roads — become
  wheel-raycast targets in two-scene mode. Roads use `collision_morph` (convex)
  + `wheel_raycast_morph` (detailed). Supersedes single-scene `--road-raycast-only`.
- `add_vehicle` / `add_static` / `add_dynamic` gained `morph=` / `surface=` /
  `vis_mode=` (and `add_dynamic` `mass=`) so callers build entities through
  VehicleScene rather than poking `main_scene` and passing a pre-built entity.

### Changed — `VehicleScene` naming pass (BREAKING, pre-1.0)

Old names accepted only where noted; otherwise these are hard renames.

- `raycast_mode` values `"raywheel"` / `"inline"` → **`"dual_scene"` /
  `"single_scene"`** (legacy `raywheel`/`inline`/`split`/`single` still accepted
  as aliases).
- `add_obstacle` → **`add_dynamic`**; the `Obstacle` handle → **`DynamicBody`**;
  the `vs.obstacles` property → **`vs.dynamics`**.
- `add_dynamic`: the `dynamic` param → **`physics`** (`True` = moves under
  physics; `False` = you teleport via `set_pose`); the `raycast` param →
  **`wheel_raycast`** and now **defaults to `False`** (a moving body is
  collide-only unless you opt the wheels into sensing it).
- `add_static`: the `raycast` toggle is **removed** (a static body is always a
  wheel-raycast target); `raycast_morph` → **`wheel_raycast_morph`**.
- `add_static_terrain` **removed** (it was a pure alias of `add_static(morph=)`;
  no heightfield-specific behavior).
- Handle fields unified: `Vehicle.entity` → **`entity_main`**;
  `Obstacle.entity` / `.mirror` → `DynamicBody.entity_main` / `.entity_raycast`
  (StaticBody already used these).
- `is_two_scene` property removed — use the `raycast_mode` string.

### Added — parameter guards (logged warnings)

- `add_dynamic(wheel_raycast=True)` on a non-primitive (mesh) morph in two-scene
  mode: its synced mirror BVH re-fits every step (cost ∝ face count); prefer a
  primitive collider.
- Two-scene-only options used in single_scene (no raycast scene): a warning fires
  for `add_dynamic(wheel_raycast=True)` (no effect — a rigid body is already a
  raycast target via the main scene) and `add_static(wheel_raycast_morph=…)` (the
  detailed-raycast/coarse-collider split needs two bodies, so it is ignored).

### Docs

- `docs/api-reference.md` §0.1 (per-method parameter tables) and §0.2
  (parameter → behavior matrix for `add_static` and `add_dynamic`, each with the
  single_scene caveat that one scene makes every rigid body a raycast target and
  `collision=False` cannot be honored).
- `two-scene-raycast.md`, the `VehicleScene` docstrings, and the
  `two_scene_terrain` sample updated to the new names.

### Notes

OSC round-trip behavior (UE / Unity client) is unchanged at the wire level but
remains untested headless — needs UE integration testing. 78 pytest pass.

---

## [0.8.0] — 2026-06-24

### Added — `VehicleScene` unified API + ray-wheel two-scene raycast (default)

| abbr | meaning |
|---|---|
| BVH | Bounding Volume Hierarchy (ray/collision acceleration tree) |
| FK | Forward Kinematics (link world transforms from base/joint state) |
| re-cast | shooting rays through an existing BVH (cheap, ~flat in face count) |
| rebuild | re-fitting the BVH from all faces (scales with face count) |

New high-level entry point `VehicleScene` (plus `Vehicle` / `StaticBody` /
`DynamicBody` handles) that owns the Genesis scene(s), registered vehicles / static
bodies / dynamic bodies, and the per-step loop — no manual `gs.init` / `scene.build` /
`scene.step` / `sensor.read`. The existing `VehiclePhysics` / `add_vehicle` / presets are
unchanged and used internally.

`raycast_mode="dual_scene"` (**default**) raycasts the terrain in a **separate
scene** as a *kinematic* body, so its BVH is built once and never re-fit (and is
shared across batch envs), while collision/rollover keep the terrain as a *rigid*
body in the main scene. Each step the chassis pose is mirrored onto a rigid,
fixed, collision-free proxy in the raycast scene; `raycast_scene.step()` re-casts
against the static BVH; the distances are fed to the main-scene physics.
`raycast_mode="single_scene"` reproduces the classic one-scene behavior. The
legacy names `"raywheel"` / `"inline"` and `"split"` / `"single"` are accepted as
aliases for `"dual_scene"` / `"single_scene"`. See `docs/two-scene-raycast.md`.

- **`VehiclePhysics.step(inputs, distances=None)`** — new optional `distances`
  arg injects externally-measured wheel-ground distances (the hook the dual_scene
  mode uses). `distances=None` reads `self.sensor` exactly as before — fully
  backward compatible. `sensor=None` is now allowed when distances are injected.

### Performance (CPU)

`raycast_mode` changes only the *raycast* cost; vehicle physics is shared.
Measured single vs split (tank/car, flat terrain, CPU):

| terrain faces | raycast single (rebuild) | raycast split (re-cast) | full-step ratio |
|---|---|---|---|
| 3 k   | ~4 ms  | ~2.5 ms | 0.94x (split slightly slower) |
| 51 k  | ~17 ms | ~2.5 ms | 2.79x |
| 205 k | ~44 ms | ~2.5 ms | 5.49x |

The *raycast* cost stops scaling with face count (rebuild → flat re-cast, up to
~18x cheaper at 205 k); the *full-step* speedup is smaller because the shared
vehicle physics (~6 ms) dominates once the rebuild is gone. `dual_scene` is
slightly slower than `single_scene` on small/flat terrain and costs ~2x terrain memory.

On **GPU at `n_envs=1`** the gap is much smaller — the rebuild parallelizes, so
single barely grows with face count and split's two-scene/launch overhead
dominates: full-step **0.98x @13 k, 1.10x @51 k, 1.31x @205 k**.

**But split's GPU win grows strongly with L3 batch size** because the static
terrain BVH is built once and shared across envs (split is ~flat in `n_envs`,
single re-fits per env): full-step **1.03x @1, 1.13x @16, 1.57x @64, 3.40x @256
envs** (51 k-face terrain; split 42 → 8576 env-steps/s ≈ near-linear vs single
41 → 2521). So **`dual_scene` is the default** (complex terrain is the common case
and the win grows with `n_envs`); switch to `single_scene` only for a flat ground at
`n_envs=1`. `VehicleScene` supports `n_envs > 1` (L3). The cast is already shared
across envs (so split is ~flat in `n_envs`), but the BVH *allocation* still
replicates per env, hitting a memory ceiling at very high `n_envs`
(Genesis #2914 lifts it).

Split also helps independent of speed via the accuracy benefit on non-convex
mesh (a rigid mesh is convexified for collision, so a single-scene rigid-mesh
raycast hits the convex bulge while the split kinematic raycast hits the true
surface). Pose/distance output is identical to `single_scene` mode (verified:
|Δx| < 1e-3 m on a 2 s drive, CPU and GPU, n_envs=1).

### Notes

- An earlier exploratory "no-step" two-scene benchmark reported ~30–47x; that was
  an artifact of a *stationary* proxy (whose stale sensor cache happened to be
  correct) and a physics-free scene. `sensor.read()` returns a cache filled by
  `scene.step()`, so a moving proxy needs a `scene.step()` to re-cast. The
  corrected mechanism and honest numbers are above.
- Scope: one or more vehicles (L2 — each gets its own proxy + sensor, still
  colliding in the main scene), L3 (`n_envs >= 1`), static terrain/mesh targets
  (`add_static`), and **dynamic raycast targets** the wheels must sense
  (`add_dynamic` — ramp / curb / moving platform; collide-only by default, pass
  `wheel_raycast=True` to make it a raycast target). In dual_scene mode the
  dynamic body gets a rigid mirror in the raycast scene's *rigid* solver (a
  separate BVH context from the kinematic terrain), re-synced each step via
  `handle.set_pose(...)`, so only its small BVH re-fits while the terrain stays
  static. Verified: the wheel distance tracks the body and matches `single_scene`
  as it moves.
- **`add_vehicle(cfg=, morph=)`** — register a vehicle the caller built itself
  (custom URDF / material / surface) with a pre-built `cfg`, instead of a
  `preset` fn. `urdf_path` is still used for the wheel positions.
- **Server L3 unification**: `server/l3_runtime.py` now builds via
  `VehicleScene` (dual_scene) — the road is rigid in the main scene (collision /
  rollover) with a kinematic raycast mirror, superseding the single-scene
  `--road-raycast-only` on the L3 path. (`env_builder.build_obstacles` gained
  `raycast_scene=` for the road mirror.) The **per-entity (non-L3) path** is also
  ported — `physics_server` builds via `VehicleScene(raycast_mode="single_scene")`
  (interacting vehicles at n_envs=1 on CPU, where the two-scene raycast has no
  benefit), `build_vehicle` registers via `add_vehicle(cfg=, morph=)`, and the
  loop uses `veh.set_inputs(...)` + `vs.step()` — so the server no longer
  constructs `VehiclePhysics` / `VehicleInputs` directly. Non-road obstacle
  mirroring remains a follow-up; the OSC round-trip needs UE integration testing.
- Upstream-correct fix (no second scene): Genesis splitting the rigid BVH into
  static + dynamic subsets — Genesis issue #2878 (open).

---

## [0.7.17] — 2026-06-24

### Performance — server collider options for large maps (`--road-raycast-only`, `--structures-as-primitive`)

| abbr | meaning |
|---|---|
| BVH | Bounding Volume Hierarchy (ray/collision acceleration tree) |
| SDF | Signed Distance Field (mesh-collider contact representation) |
| CoACD | Convex Approximate Convex Decomposition (concave mesh → many convex hulls) |

Diagnosing a report of ~150 ms/step with a vehicle on a map carrying a few
hundred imported structure meshes. The rigid contact solver was **not** the
cost (it handles fixed, non-contacting mesh geoms for ~free). The cost is the
**wheel `Raycaster`**: upstream Genesis rebuilds **one flat collision BVH over
every collision face in the rigid solver — including all fixed/static geometry
— on every `scene.step`** (the static-rebuild skip added upstream only engages
when *all* solver links are fixed, which a moving vehicle defeats; the proper
fix is upstream PR #2878, static/dynamic BVH split, still in review). So per-step
cost scaled with the *total triangle count of all mesh colliders*, contact or
not. Reproduced standalone (genesis + trimesh): 300 torus structures, nothing
hit, add a raycaster → **+135 ms/step on CPU**; the same as `Box` primitives →
flat.

Two opt-in server flags (both default off; no behavior change unless passed):

- **`--road-raycast-only`** — load complex road/terrain meshes (`[Complex]`) as
  a **`Kinematic`** entity with **`use_visual_raycasting=True`** instead of a
  rigid collision mesh. The wheel raycaster casts against both the rigid and
  kinematic solvers, but the kinematic solver's BVH is `maybe_static` (no
  physics-movable link) → its rebuild is **skipped every step** even while the
  vehicle moves. The ray-cast wheels still follow the surface (the chassis is
  held by suspension forces, so the road need not be a contact body). This also
  skips CoACD and the chassis-vs-road narrow-phase. Verified end-to-end (SDK car
  on a 28k-face road, genesis 1.2.0): car drives + stays up; **scene.step 6.3 ms
  vs 19.6 ms** for the earlier rigid+visual-raycast path (which still rebuilt in
  the rigid BVH).

- **`--structures-as-primitive`** — replace every mesh collider with its
  bounding `Box` (`mesh_to_primitive_box`: local AABB, scaled, with the entity
  quat applied → effectively an OBB). Box collision is analytic (no per-geom
  SDF), and a box contributes few faces to the rebuilt BVH. For structures that
  must physically block the vehicle (rigid contact still needed), this keeps
  them cheap; cost then scales with *actual contacts*, not structure count.

A standalone reproduction + an upstream issue write-up are kept outside the SDK
repo (`genesis_raycaster_bvh_repro.py`, `genesis_raycaster_bvh_issue.md`).

## [0.7.16] — 2026-06-18

### Performance — `VisualJointSync` batches its per-step joint writes (≈5× less overhead)

Investigating why `VisualJointSync` costs so much even at 1 vehicle: each
`entity.set_dofs_position` call lowers to `solver.set_dofs_position`, which does
a **collider reset + constraint-solver reset + a full forward-kinematics pass
over every link and geom** — every call. `VisualJointSync.step` issued these
separately for spin, steer, and suspension (plus a `set_dofs_velocity`), so a
single step paid for 3–4 collider/constraint resets and FK passes.

Now spin + steer + suspension(set-path) are concatenated and written with **one
`set_dofs_position` call** (one FK pass instead of three), and the
drift-suppression `set_dofs_velocity` passes `skip_forward=True` (no extra
velocity FK). `control_dofs_position` (PD, heavy wheels) uses a different API
and stays separate. Also precomputes the steer-axis-sign tensor once instead of
per step.

Measured (CPU, 1 vehicle, `car_4w_rwd_ackermann`):

| | ms/step | VisualJointSync tax |
|---|---|---|
| physics only | 8.30 | — |
| + VisualJointSync (before) | 12.81 | **+4.46** (1.53×) |
| + VisualJointSync (after)  | 9.16  | **+0.85** (1.10×) |

~5× reduction in the per-step tax. Applies to both `VehiclePhysics` and
`MultiVehicleKindPhysics` (both drive a per-entity `VisualJointSync`).

Correctness unchanged: the final joint state is identical (same values, one call
instead of three — only the wasted intermediate FK passes are removed). Verified
on CPU that the closed-form `wheel_visual_transforms` still matches the
engine-driven `get_link` (Δpos ≈ 4.3 mm steady-state jitter, Δquat = 0.0°).

Note: this lowers the cost but does not eliminate it — for headless / external
(UE) runs keep `enable_visual_joint_sync=False` and read poses from the
closed-form `wheel_visual_transforms` (~µs). The Genesis native viewer still
needs `VisualJointSync` (the closed-form cannot drive the viewer).

---

## [0.7.15] — 2026-06-17

### Changed — `enable_visual_sync` → **`enable_visual_joint_sync`** (rename, breaking)

Renamed the flag to match the class it gates, `VisualJointSync`. The flag
toggles `VisualJointSync` (which drives the URDF wheel **joints**), not a
`VisualSync`, so `enable_visual_joint_sync` is the accurate name. No alias is
kept — same policy as the `VisualSync` alias removal in 0.7.9.

- `VehicleConfig.enable_visual_sync` → `VehicleConfig.enable_visual_joint_sync`
  (default stays `False` from 0.7.14).
- Updated the `ResolvedConfig` field, the `resolve()` mapping, the `core` /
  `multi_vehicle` consumers, and the `VisualJointSync` perf-advisory text.
- Server: `vehicle_builder.build_vehicle(*, enable_visual_joint_sync=False)`
  keyword, and the `physics_server` / `l3_runtime` `= not args.headless` setter.
- All 6 `--viewer` samples: `cfg.enable_visual_joint_sync = args.viewer`.
- Docs updated (`api-reference.md`, `README.md`, `samples/README.md`). The
  0.7.13/0.7.14 CHANGELOG entries keep the old name (it is what those versions
  actually shipped).

Migration: replace `cfg.enable_visual_sync` with `cfg.enable_visual_joint_sync`.
The old name no longer exists (AttributeError).

---

## [0.7.14] — 2026-06-17

Follow-up to v0.7.13 team feedback: (1) `VisualJointSync` is viewer-only but
defaulted ON, silently costing performance in headless / external-render runs
→ flip the default off. (2) The suspension visual clamp should be settable, not
locked to a derived value → expose it as a config field.

### Changed — `enable_visual_sync` default `True` → **`False`** (breaking)

`VisualJointSync` (drives the URDF wheel visual joints through engine FK every
step) is only needed for the **Genesis native viewer**. The closed-form
`wheel_visual_transforms` cannot update the Genesis viewer (it only returns
pose tensors), so it is for external renderers (UE / Unity), while the Genesis
viewer still needs `VisualJointSync`. Given the dominant path is headless /
external render, and default-ON is a silent perf trap (the 0.7.13 server 14ms
case), the flag is now **opt-in**.

- `VehicleConfig.enable_visual_sync` defaults to `False`.
- Set it `True` explicitly only when opening the Genesis viewer (or a sample
  `--viewer`).
- The server already sets it explicitly (`= not args.headless`) → unaffected.
- The 6 samples (`quickstart` / `slope_hold` / `multi_env_render` /
  `batched_rollout` / `road_loop` / `city_traffic_ego`) enable it only with
  `--viewer`. Headless runs are now faster (closed-form path).

Migration: code that relied on Genesis-viewer wheel animation must set
`cfg.enable_visual_sync = True` (otherwise wheels freeze at the rest pose — the
chassis still drives normally).

### Added — `VehicleConfig.susp_visual_clamp` (suspension visual clamp is settable)

0.7.13 derived the clamp per-wheel from `rest_stroke`; feedback was that it
should be configurable rather than locked to a derived value. The clamp is a
**visual safety bound** against raycast spikes, not a physics limit.

- `susp_visual_clamp: Optional[float] = None`
  - `None` (default) → **auto**: per-wheel = that wheel's `rest_stroke`
    (= `rest_d − radius`, min 0.02 m). Since `rest_d ≡ radius + rest_stroke`,
    this is just the wheel's own stroke. The 0.02 m floor keeps a ~zero-stroke
    wheel from freezing at rest.
  - `float` (e.g. `0.30`) → uniform clamp on every wheel.
  - `≤ 0` → `ConfigError`.
- `core` / `multi_vehicle` / `VisualJointSync` share the value, so the
  closed-form `wheel_visual_transforms` stays equal to what the viewer drives.

---

## [0.7.13] — 2026-06-17

Two team reports addressed: (1) the SDK-provided physics server is slower than
the team's existing server (CPU, 1 vehicle: 4–5 ms vs 14–15 ms/step), and
(2) the suspension visual clamp is fixed at 0.19 m, muting large-travel
vehicles.

### Fixed — server disables `VisualJointSync` when headless / external-render (perf)

Cause: the SDK's `physics_server` / `l3_runtime` ran with `VisualJointSync`
ON. `VisualJointSync` drives the URDF wheel visual joints through the engine's
articulated-body FK every step (~ms/step) — viewer-only, and unnecessary for an
external renderer (UE / Unity) or any headless run. The team's existing (fast)
server was already running with `enable_visual_sync=False`; that was the
difference.

- `vehicle_builder.build_vehicle(*, enable_visual_sync=False)` keyword added;
  applied to `cfg.enable_visual_sync` after build.
- `physics_server` / `l3_runtime`: `enable_visual_sync = not args.headless`.
  Headless (= external UE render) turns `VisualJointSync` off and captures wheel
  poses via the closed-form `wheel_visual_transforms("world")` (no per-step
  engine FK cost).

Step time is normalized to the same `enable_visual_sync=False` path as the
existing server. It is only on when the viewer is needed (`--headless` omitted).

### Fixed — suspension visual clamp is per-wheel `rest_stroke` (removed fixed 0.19)

`core._susp_visual_offset` and `visual.VisualJointSync` hardcoded a `±0.19 m`
suspension visual offset, so vehicles whose travel exceeds 0.19 m (trucks /
tanks) had their wheel motion clipped and looked muted. The clamp is now each
wheel's **own stroke** (`rest_d − radius`, min 0.02 m).

- `core._susp_visual_offset(clamp=…)` accepts a tensor clamp (a scalar still
  works — the helper's unit-test default stays 0.19).
- `VehiclePhysics.__init__`: builds `self._susp_clamp = clamp(rest_d − radius,
  ≥0.02)` and passes it to the getter / `wheel_visual_transforms`.
- `multi_vehicle` uses the same per-wheel clamp.
- `VisualJointSync` (set path) applies the same per-wheel clamp — keeping the
  viewer equal to the closed-form `wheel_visual_transforms`.

Vehicles with stroke ≤ 0.19 m (e.g. the default car preset) are unchanged; only
those exceeding 0.19 m get their visible travel restored.

---

## [0.7.12] — 2026-06-17

### Added — multi-vehicle (L2 / L2×L3) wheel visual transforms

`wheel_visual_transforms` / `visual_parts_transforms` were only on the
single-vehicle `VehiclePhysics`; the multi-vehicle path didn't expose them
(the report: "not yet updated for multi_vehicle"). Now:

- `MultiVehicleKindPhysics.step` maintains the visual-pose state on its proto
  (`last_steer_per_wheel`, accumulated `wheel_spin_angle`, `_stepped_once`) —
  the same bookkeeping `VehiclePhysics.step` does.
- `MultiVehicleKindPhysics.wheel_visual_transforms(frame)` → closed-form
  `(n_envs, K, n_wheels, 3/4)` (rest pose captured from the kind's first
  entity — identical across the kind's same-URDF vehicles; honors
  `visual_spin_enabled`).
- `MultiVehiclePhysics.wheel_visual_transforms(frame)` → per-vehicle list
  (caller order) of `(pos, quat)`; `visual_parts_transforms(frame)` → per-
  vehicle list of `VisualPartsTransforms` (chassis + wheels).

Verified: single `VehiclePhysics` vs `MultiVehiclePhysics` at K=1 produce
**identical** wheel visual transforms (Δpos = 0, Δquat = 0); K=2 × n_envs=2
returns correct per-vehicle shapes. No change to the single-vehicle API.

---

## [0.7.11] — 2026-06-16

### Fixed — `wheel_visual_transforms` / `visual_parts_transforms` now correct for trucks & tanks

The closed-form wheel visual pose now honors `visual_spin_enabled`: skid-steer
/ tank presets (`tank_10w_skid_belt`) disable the wheel spin visual (cylindrical
road wheels — spin is invisible), and the closed-form previously baked spin
into the wheel quat anyway, disagreeing with the viewer. It now omits spin when
disabled, matching `VisualJointSync`. Verified against `get_link` (VisualJointSync
on) across all bundled vehicle classes: 4-wheel car ≈ 3.5 mm / 0.04°, 6-wheel
truck (`truck_6w_partial_ackermann`, front-axle steer) ≈ 3.5 mm / 0.04°,
10-wheel skid-steer tank (`tank_10w_skid_belt`) ≈ 0.6 mm / 0°. Trucks already
worked (same conventional axes as cars); tanks needed the spin-flag fix. No API
change.

---

## [0.7.10] — 2026-06-16

### Changed — `render_transforms` → `visual_parts_transforms` (`RenderTransforms` → `VisualPartsTransforms`)

Renamed the one-call render feed (method + dataclass) for a clearer name that
says what it returns: the visual parts (chassis + wheels) of the vehicle. No
behavior change. Old names existed for v0.7.8–0.7.9 only; no alias kept
(the SDK/server/team code is updated). `__init__` export, `_check_import`,
and docs unified on the new name.

### Added — one-time perf advisory when `VisualJointSync` is enabled

On first construction of a `VisualJointSync` (i.e. `enable_visual_sync=True`,
the default), the SDK logs a one-time-per-process `[genesis_vehicle] PERF:`
line to stderr: it drives the URDF wheel joints through the engine's
articulated-body FK every step (~ms/step, the dominant SDK cost at scale) and
is only needed for the Genesis viewer — external / headless users should set
`enable_visual_sync=False` and read `visual_parts_transforms` /
`wheel_visual_transforms` instead. Silence with `GENESIS_VEHICLE_QUIET=1`.

---

## [0.7.9] — 2026-06-16

### Removed — `VisualSync` alias (use `VisualJointSync`)

The deprecated `VisualSync` alias added in v0.7.8 is removed — the class is
now only `VisualJointSync`. Dropped from `__init__` exports and
`_check_import`; all internal/doc references unified on `VisualJointSync`.
External code that imported `VisualSync` must switch to `VisualJointSync`
(it was a one-release alias). The bundled `genesis_unreal_plugin` / D:\ team
server do not import it, so they are unaffected.

---

## [0.7.8] — 2026-06-16

### Added — `VehiclePhysics.render_transforms(frame=...)` + `RenderTransforms`

One-call render feed for an external engine (UE / Unity): chassis pose **and**
wheel visual poses together, VisualSync-independent. The chassis is the real
dynamics pose (`get_pos/get_quat`, always world); the wheels are the closed-
form visual pose (`wheel_visual_transforms`). `frame` applies to the wheels
(`"world"` / `"local"`). Returns a `RenderTransforms` dataclass (chassis_pos/
quat, wheel_names, wheel_pos/quat). The recommended bridge feed — one call per
vehicle, no `get_link`, no VisualSync.

### Changed — `VisualSync` renamed to `VisualJointSync` (alias kept)

The viewer-side class was renamed to make its scope explicit: it drives the
**wheel** visual joints (spin/steer/suspension) for the **Genesis viewer**
only — it never moves the chassis (real dynamics) and does not affect physics.
External renderers don't need it (use `render_transforms` /
`wheel_visual_transforms`). `VisualSync` remains as a **deprecated alias**, so
existing code keeps working. Internal references, `__init__` exports,
`_check_import`, and docs updated; the class docstring now spells out the
scope/non-scope.

---

## [0.7.7] — 2026-06-16

### Added — `VehiclePhysics.wheel_visual_transforms(frame=...)` (closed-form wheel visual pose)

New getter returning each wheel's VISUAL transform (steer + suspension +
spin applied) **without driving Genesis joints** — works whether or not
VisualSync is enabled. The blessed feed for an external renderer (UE /
Unity): `frame="local"` gives the pose relative to the chassis (attach the
wheel under the chassis component → it rides rigidly), `frame="world"` the
absolute pose. Computed closed-form (~µs/wheel) instead of the engine's
articulated-body FK; verified to match `entity.get_link(wheel)` in the
steady regime (≈3 mm / 0.04°, the residual being get_link's own kinematic-
suspension substep jitter, from which the closed-form is free).

Why this exists: reading `entity.get_link(wheel)` only reflects steering/
suspension/spin when VisualSync is ON (it drives those URDF joints). Teams
that disabled VisualSync for performance but kept reading get_link saw
wheels frozen at the rest pose (no suspension travel, no steer). This getter
decouples the visual pose from VisualSync and from the engine.

- `core.py`: `wheel_visual_transforms`, plus `last_steer_per_wheel` and a
  `wheel_spin_angle` accumulator (maintained every step so the getter works
  headless), and quaternion helpers `_quat_axis_angle` / `_quat_mul` /
  `_susp_visual_offset` (the suspension formula shared with VisualSync).
- `link_transforms()` now emits a one-time warning if called with VisualSync
  disabled (wheel links are at the rest pose then) — pointing to
  `wheel_visual_transforms`.
- `tests/test_quat_helpers.py`: 8 pure-Python tests for the quaternion /
  suspension-offset math. 70 → 78 tests.

### Changed — `genesis_vehicle.server` feeds wheels from `wheel_visual_transforms`

`server/physics_server.py` and `server/l3_runtime.py` now build wheel state
from `ctrl.wheel_visual_transforms("world")` instead of `get_link` reads +
a separately-accumulated spin angle. The pos/quat already include
steer/suspension/spin, so the per-wheel spin scalar is sent as 0 (the client
uses the quat directly). Fixes the "no suspension travel / wrong wheel
rotation" symptom when running the server headless (VisualSync off), and
drops the now-unused wheel-link-index cache.

---

## [0.7.6] — 2026-06-13

### Changed — docs/server.md architecture diagram → Mermaid

Replaced the ASCII client↔server box diagram with a Mermaid `flowchart`
(renders as a real diagram on GitHub / VS Code / Obsidian). Other docs'
text diagrams (the pipeline pseudocode, the README directory tree) are
intentionally left as text.

---

## [0.7.5] — 2026-06-13

### Removed — docs/migration.md

Dropped the legacy HJW/JMK/KDU → SDK migration guide (no longer needed).
Removed its links from docs/index.md and README.md; the one cross-reference
in physics-contracts.md (KDU steer-sign flip) is now stated inline.

---

## [0.7.4] — 2026-06-13

### Added — docs/tire-and-contact.md (tire models vs ground-contact mechanisms)

New doc separating the two orthogonal axes that are easily conflated:
tire friction model (Pacejka vs Coulomb) and ground-detection mechanism
(raycast wheel / shapecast / collider contact / contact-patch), with
standard names for each. States the recommendation (Pacejka + raycast
wheel), explains why Coulomb is not a performance win (lighter compute is
negligible; worse low-speed stability), and clears up the historical
"Coulomb = collider-vs-collider" confusion — today's `CoulombIsotropic`
is ray-hit-based, same pipeline as Pacejka. Linked from docs/index.md.

---

## [0.7.3] — 2026-06-13

### Fixed — multi-vehicle (L2 / L2×L3) now gets the `F_long` overshoot clamp

The per-wheel `step()` math was mirrored between `VehiclePhysics.step` and
`MultiVehicleKindPhysics.step`, and had **silently drifted**: the v0.6.0
longitudinal-friction overshoot clamp was added to the single-vehicle path
only. Multi-vehicle (`MultiVehiclePhysics`, i.e. L2 and L2×L3) was missing
it, so a batched traffic/MPPI scene launched with slightly different wheel
dynamics than the same vehicle run singly.

Both paths now call one shared `_pipeline.compute_wheel_step`, so the clamp
(and any future physics fix) applies to both. Verified: a single
`VehiclePhysics(n_envs=1)` and a `MultiVehiclePhysics` with K=1 now produce
**identical** trajectories (Δpos = 0.0 mm, Δω = 0.0) over an accel+steer
sequence — previously they diverged in the acceleration phase.

### Changed — shared per-wheel pipeline extracted to `_pipeline.py`

Sections A–E (suspension → tire → omega → force accumulation → coupling)
of the ray-wheel step are now a single batch-shape-agnostic pure function
`compute_wheel_step(...)` in `genesis_vehicle/_pipeline.py`. `core.py` and
`multi_vehicle.py` both call it; their `step()` methods keep only their own
I/O (entity/sensor reads, solver force writes, VisualSync). Removes ~120
lines of duplicated math and the mirror-drift maintenance hazard. No public
API change; behavior for single-vehicle is byte-identical (math moved
verbatim).

---

## [0.7.2] — 2026-06-13

### Added — docs/server.md (purpose + modes + full OSC schema reference)

Dedicated user doc for the OSC physics server: purpose (Genesis = physics
truth, external client = render/control), when to use the server vs the
in-process SDK API, the three modes + CPU/GPU rationale, and a complete
OSC wire-schema reference (ports, handshake sequence, every inbound/
outbound address with payload layout, RHS->LHS coordinate conversion, and
the vehicle mapping JSON keys incl. the steerScale/maxSteerRad note).
Linked from docs/index.md.

### Changed — genesis_unreal_plugin reduced to a thin launcher

The repo-external `genesis_unreal_plugin/` no longer keeps its own copy of
the server: `physics_server.py` is now a ~20-line shim that forwards to
`genesis_vehicle.server`, and the duplicate `osc_manager.py` /
`genesis_env_builder.py` / `genesis_vehicle_builder.py` were removed (their
canonical, improved versions live in genesis_vehicle/server/). Prevents
the edit-the-wrong-copy hazard that previously dropped a perf patch. (The
plugin folder is outside the SDK git root, so this change is not in this
commit; it is documented here for traceability.)

---

## [0.7.1] — 2026-06-13

### Added — `samples/l2l3_minimal.py` + L2/L3 docs: API-selection & two-class rationale

- **`samples/l2l3_minimal.py`** — the shortest runnable L2 × L3 program
  (~90 lines): K interacting vehicles in one world × N parallel scenarios
  via a single `MultiVehiclePhysics(scene, vehicles, n_envs=N)`. Shows
  per-(scenario, vehicle) control (lead car brakes in scenario 0 only and
  diverges from the rolling copies). Fills the gap between the full
  `city_traffic_ego.py` demo and the `perf_l2_l3_combined.py` benchmark.
- **`docs/batching.md`** — new "Why two classes? (`VehiclePhysics` vs
  `MultiVehiclePhysics`)" section: confirms K=1 `MultiVehiclePhysics` ≈
  `VehiclePhysics(n_envs=N)`, explains that Multi is built *on top of*
  Single (proto reuse) and they differ only in the I/O layer, why they
  stay separate (common-case ergonomics; composition), and the honest
  caveat that `step()` math is currently mirrored between the two (a
  future-cleanup wart, not a correctness issue). Decision matrix + perf
  table now point to `l2l3_minimal`.
- **`docs/batching.md` + `docs/api-reference.md`** — explicit guidance: for
  K > 1 vehicles in one scene, prefer `MultiVehiclePhysics` over a manual
  Python loop of K `VehiclePhysics` objects (the loop is correct but skips
  L2 batching). Documents the one legitimate exception (per-vehicle solver
  ops — independent forces/impulses or mid-rollout teleport — which is why
  `genesis_vehicle.server`'s per-entity mode still loops) and that K = 1
  should just use plain `VehiclePhysics`.

### Changed — `MultiVehiclePhysics` grouping/dispatch extracted to pure functions

`group_vehicles_by_cfg()` and `rebucket_inputs()` are now module-level
pure functions (no Genesis objects touched), so the kind-grouping and
flat→(kind, slot) input routing that L2 × L3 depends on is unit-tested
without a GPU. Behavior unchanged; `MultiVehiclePhysics.__init__` /
`.step` call the extracted helpers.

- `tests/test_multi_vehicle_grouping.py` — 5 pure-Python tests
  (single-kind, interleaved kinds, caller-order preservation, input
  re-bucketing round-trip, single-vehicle). 65 → 70 tests.

---

## [0.7.0] — 2026-06-13

### Added — `genesis_vehicle.server` subpackage (OSC physics server, moved in from genesis_unreal_plugin)

The UE-bridge physics server now lives inside the SDK as
`genesis_vehicle/server/` so it is version-controlled with the package
(previous home `genesis_unreal_plugin/` was an unversioned folder — a
hand-off overwrite silently dropped a perf patch once). It is a
client-agnostic OSC/UDP server; UE is just one client.

Run it with:

```bash
python -m genesis_vehicle.server               # per-entity mode (legacy; heterogeneous / few vehicles, CPU)
python -m genesis_vehicle.server --multi-env   # L3 batched mode (same-URDF fleet, GPU)
```

- `server/physics_server.py` — legacy per-entity loop (one entity +
  `VehiclePhysics(n_envs=1)` per target). Unchanged semantics, plus:
  Windows-only code (PyInstaller `ctypes.CDLL` patch, `HIGH_PRIORITY_CLASS`)
  is now platform-guarded so the server also runs on Linux; package-relative
  imports (no `sys.path` hacks); `capture_state` reads wheel poses via
  cached link indices + 2 batched `get_links_pos/quat` calls instead of
  per-wheel `get_link(name)+get_pos+get_quat` (≈6× fewer Genesis calls
  per step at 100 vehicles) and converts `omega` once per vehicle instead
  of one `.item()` per wheel.
- `server/l3_runtime.py` — **new `--multi-env` mode**: N same-URDF,
  non-interacting targets run as ONE vehicle entity ×
  `scene.build(n_envs=N)` with a single batched `VehiclePhysics(n_envs=N)`.
  Measured on the dev laptop (dt=0.02, substeps=2, GPU): 30/50/100 vehicles
  all ≈ 19 ms/step **including** state capture — vs 1,115 ms/step for the
  per-entity loop at 30 vehicles on the same GPU (57×). Backend defaults to
  GPU (`--force-cpu` to override). Known v1 limits (logged at runtime):
  dynamic obstacles are per-env copies (states sent from env 0),
  `target_forces` and `AddWorldImpulse/Torque` relative commands are not
  supported, no lockstep.
- `server/vehicle_builder.py` — cfg construction extracted into
  `build_cfg()` shared by both modes; `strip_wheel_collisions()` extracted.
  **Fixed:** steering-range mapping key mismatch — UE serializes
  `FGenesisVehicleMapping.SteerScale` as `steerScale`, but only
  `maxSteerRad` was read, so the UE setting was silently ignored and the
  preset default (0.7 rad) always applied. `_mapping_steer_rad()` now
  accepts `maxSteerRad` / `MaxSteerRad` / `steerScale` / `SteerScale`
  (in that priority).
- `server/env_builder.py`, `server/osc_manager.py` — moved as-is
  (osc_manager loses its `sys.path` hacks).
- `tests/test_server_import.py` — import smoke + steer-key mapping unit
  tests (auto-skip when genesis/pythonosc absent, so the pure-Python CI
  story is unchanged). 62 → 65 tests.

Server-only dependencies (`pythonosc`, `psutil`, `trimesh`) are NOT
required by the SDK core — `genesis_vehicle.server.__init__` is
intentionally empty so `import genesis_vehicle` works without them.

The old `genesis_unreal_plugin/` copies remain for the team's current
workflow but `genesis_vehicle/server/` is now the canonical, git-tracked
home; edit there.

---

## [0.6.0] — 2026-06-09

### Fixed — longitudinal friction-force overshoot clamp (wheel-spin oscillation / "front tire slips at launch")

`VehiclePhysics.step` now caps the tire longitudinal force `F_long` so the
resulting friction torque cannot carry the wheel **past the rolling speed**
(`omega_target = v_long / R`) in a single step:

```
F_long_limit = (omega_nofric − omega_target)·I_wheel / (dt·R)
omega_nofric > omega_target → F_long ∈ [0, F_long_limit]
omega_nofric < omega_target → F_long ∈ [F_long_limit, 0]
```

where `omega_nofric` is the wheel speed after drive/brake but before tire
friction. This is the tire-friction analogue of the existing
`brake_torque_signed` anti-overshoot clamp (target `omega = 0`).

**Why.** Explicit-Euler integration of the slip-dependent friction torque is
stiff near rolling (relaxation rate `R²·C_kappa/(I·|v_long|) → ∞` as
`v_long → 0`). Below the stability limit it oscillates — the wheel ping-pongs
across the rolling point (forward force → reverse slip → backward force → …),
seen as wheel "trembling" and, on undriven wheels, a stuck `kappa ≈ −1` drag
at launch (the RWD front-tire-slip report). The clamp binds **only near
rolling** (where `omega_nofric − omega_target` is small), so it kills the
oscillation while leaving the high-slip saturated regime — driven-wheel launch
slip — untouched. Verified: `quickstart` launch preserved (x = 12.59 m vs
12.43 m baseline); 62/62 unit tests pass; JMK / Truck6w / KDU scenarios
unchanged (HJW slightly slower — removes the spurious overshoot thrust, may
want per-vehicle torque re-tune). Lets the wheel mass / inertia "band-aid"
(inflated `i_wheel`) be reverted to realistic values without re-introducing
the oscillation.

| 약자 | 의미 |
|---|---|
| F_long | 종방향 타이어 마찰력 |
| omega_target | 구름조건 각속도 (= v_long/R) |
| C_kappa | 종슬립 강성 ∂F_long/∂kappa |

### Added — per-link transforms in world / base / **parent (URDF-hierarchy-local)** frames

New public API `get_link_transforms(entity, frame=...)` (module
`genesis_vehicle.kinematics`) and the convenience method
`VehiclePhysics.link_transforms(frame=...)`. Returns a `LinkTransforms`
dataclass — link `names`, parent topology (`parent_local`), and batched
`pos (n_envs, n_links, 3)` / `quat (n_envs, n_links, 4)` (wxyz) — plus
`.matrices()` (4×4) and `.index(name)`.

**Why.** Genesis only exposes each link's **world** pose
(`get_links_pos`/`get_links_quat`). For telemetry → animation retargeting,
external sensor/effect attachment, or placing ghost copies you usually want
each component relative to its **URDF parent** — that is what an animation
rig's local channels are. The frame choice was previously the caller's to
compose by hand from world poses + quaternion math.

Frames:

| `frame` | each link expressed relative to |
|---|---|
| `"world"`  | world (raw Genesis output) |
| `"base"`   | the entity base/root link (one frame for the whole vehicle) |
| `"parent"` (default) | its **immediate parent link** in the URDF tree (hierarchy-local); root → world |

Vectorised over `n_envs` and `n_links` (no Python per-link loop): parent
poses are gathered and the relative transform is `q_PC = inv(q_WP) ⊗ q_WC`,
`p_PC = R_WP⁻¹·(p_WC − p_WP)`. Single-env builds return `(n_links, …)`,
batched builds `(n_envs, n_links, …)`, matching `get_links_pos`. Re-exported
as `genesis_vehicle.get_link_transforms` / `LinkTransforms` (lazy import —
needs a built entity).

| 약자 | 의미 |
|---|---|
| P / C | Parent link / Child link |
| W | World frame |
| `q_WC` | 월드→자식 링크 쿼터니언 (wxyz) |
| `R_WP` | 월드→부모 링크 회전행렬 |

---

## [0.5.33] — 2026-05-30

### Added — version banner on first `VehiclePhysics()` construction

`VehiclePhysics.__init__` now prints a single stderr line
`[genesis_vehicle] vX.Y.Z` the first time it is instantiated in a process
(module-level `_BANNER_PRINTED` guard prevents duplicates across multiple
vehicles). Lets users and bug reports confirm which SDK build is loaded
at runtime without an explicit `print(genesis_vehicle.__version__)`.

Suppress with `GENESIS_VEHICLE_QUIET=1`. The banner does **not** fire on
plain `import genesis_vehicle` or `genesis_vehicle.version()` — only on
actual vehicle construction — so doc builds, URDF-only tooling, and unit
tests that never instantiate a vehicle remain silent.

### Changed — validated against the Genesis 1.0.0 backend (was 0.4.6)

The Genesis physics backend reached its `1.0.0` release. genesis_vehicle
is now developed and validated against **genesis-world ≥ 1.0.0**; the
previously validated floor was `0.4.6`.

**No SDK code change was required.** Every Genesis API surface the SDK
binds to is signature-compatible between 0.4.6 and 1.0.0 (verified by
inspecting the installed `genesis-world 1.0.0` source):

| SDK call site | Genesis 1.0.0 symbol | Status |
|---|---|---|
| `raycast.WheelRayPattern(RaycastPattern)` overriding `_get_return_shape` / `compute_ray_dirs` / `compute_ray_starts` / `_ray_dirs` / `_ray_starts` | `genesis.options.sensors.raycaster.RaycastPattern` | unchanged base class + hook names |
| `gs.sensors.Raycaster(pattern=, entity_idx=, max_range=, min_range=, return_world_frame=)` | `genesis.options.sensors.Raycaster` (`KinematicSensorOptionsMixin` + `SimpleSensorOptions`) | all five kwargs still valid |
| `scene.add_sensor(...)` → `sensor.read().distances` | `Scene.add_sensor`, `RaycasterData.distances` (NamedTuple) | unchanged |
| `solver.apply_links_external_force(F, links_idx)` / `apply_links_external_torque(T, links_idx)` | `RigidSolver.apply_links_external_*` | unchanged positional signature; default `ref="link_origin"` |
| `gs.init(backend=gs.gpu, logging_level=...)` | `genesis.init` | unchanged |
| `gs.options.SimOptions(dt=, substeps=)` / `RigidOptions(enable_collision=, ...)` / `VisOptions(env_separate_rigid=)` | `genesis.options.solvers` / `genesis.options.vis` | all fields present |

The 58 pure-Python tests do not exercise Genesis and are unaffected.

### GPU regression (RTX 5070 Laptop, genesis-world 1.0.0, CUDA)

The default reference frame of `apply_links_external_force` is
`ref="link_origin"` in 1.0.0. `core.py` measures the torque arm
`r_vec = wheel_world - entity.get_pos()` from the base-link origin and
applies `total_F` / `total_T` as a force + pure-torque pair about that
same origin, so the decomposition is only correct if the force lands at
the link origin. Confirmed empirically that it does — the validated
samples reproduce their 0.4.6 numbers:

| Sample | Path | Result on 1.0.0 |
|---|---|---|
| `quickstart` | L1 single env | drives straight: `x=+12.43 m`, `y=+0.01 m`, `4.97 m/s` after 240 steps |
| `slope_hold` | StaticFrictionLock on 20° slope | lateral slip **+0.1 mm / 10 s** (unchanged from v0.5.32); roll settles +20.31° |
| `batched_rollout` | L3, `n_envs=64` | `sensor.read().distances` → `(64, 4)`; all shapes correct, realistic env spread |
| `perf_l2_l3_combined` | L2 × L3, up to `K=4 × N=64 = 256` veh | clean, 126× scaling; `MultiVehiclePhysics` batched force/torque apply OK |

### Notes

- `RigidOptions.use_contact_island` still defaults to `False` in 1.0.0.
  The multi-vehicle samples continue to leave it off — see the v0.5.28
  entry for the 0.4.6 kernel-compile error that motivated avoiding it;
  that finding has not been re-validated on 1.0.0, so the samples keep
  the safe default.
- Installation docs (`README.md` Installation, `docs/quickstart.md`)
  now state the `genesis-world ≥ 1.0.0` requirement explicitly.

---

## [0.5.32] — 2026-05-26

### Fixed — brake torque could overshoot zero and act as propulsion

`brake_torque_signed` previously returned ``t_brake * tanh(omega/0.5)``
only. tanh smooths the discontinuity around ``omega ≈ 0`` but does
**not** prevent a single forward-Euler step from reversing ω when
``t_brake`` is large and ``dt`` non-trivial:

```
omega = +0.1 rad/s, t_brake = 100 Nm, I = 0.5, dt = 0.01
T_brake_eff = 100 * tanh(0.2) ≈ 19.7 Nm
new_omega = 0.1 - 19.7/0.5 * 0.01 = -0.294    # sign flip!
next step: omega=-0.294 -> T_brake flips -> overshoots to +0.76
... amplitude grows; user reports brake "acting like propulsion"
```

Classic stiff-friction instability with explicit Euler. The tanh helps
inside ``|omega| < smoothing_scale`` but saturates beyond that, so the
discrete overshoot is unbounded.

Fix: when `brake_torque_signed` is called with the new `dt` and
`i_wheel` kwargs, the magnitude is capped at
``|omega| * i_wheel / dt`` — the torque that exactly zeroes ω this
step. Brake can decelerate to rest but never past it. `core.py` and
`multi_vehicle.py` now pass these kwargs. The static-hold case
(``omega → 0`` with brake held) is handed off to `StaticFrictionLock`
as before.

### Back-compat

The new kwargs default to ``dt=0.0, i_wheel=None`` → clamp disabled,
behavior exactly matches v0.5.31 (only tanh applies). Existing 8 tests
in `tests/test_dynamics.py` still pass without modification.

### New tests

`tests/test_dynamics.py`:
- `test_brake_clamp_prevents_omega_sign_flip` — at ω ∈ {±0.1, ±5}
  with the canonical overshoot config (t_brake=100, I=0.5, dt=0.01),
  asserts (a) sign(eff) == sign(ω), (b) |eff| ≤ |ω|·I/dt, (c) one
  forward-Euler step preserves the sign of ω.
- `test_brake_clamp_legacy_when_dt_zero` — without dt/i_wheel kwargs,
  output matches the v0.5.31 tanh-only formula exactly.

62 SDK pytest pass (60 + 2 new). `slope_hold` regression still
OK (lateral slip 0.1 mm).

---

## [0.5.31] — 2026-05-26

### Changed — `VehicleConfig.dt` → `VehicleConfig.recommended_dt` (advisory)

v0.5.30 added a validator that raised on `cfg.dt != scene.sim.dt`. The
two-place duplication was the underlying problem, not the lack of
validation. v0.5.31 removes the duplication: `VehiclePhysics` now pulls
`self.dt` directly from `scene.sim.dt` (Genesis owns physical time). The
preset's old `dt` field is renamed `recommended_dt` and is purely
advisory — what the sample wires into `SimOptions` so the user gets the
preset author's recommended physics rate by default.

If `scene.sim.dt != cfg.recommended_dt`, `VehiclePhysics.__init__` emits
a **one-time-per-process warning** (one line, gated on the
`(recommended_dt, scene_dt)` pair) and uses `scene.sim.dt`. No more
hard error — Genesis wins, the user gets a heads-up if their picked dt
is outside the preset's tested range.

```
[genesis_vehicle] WARN: scene.sim.dt=0.002 differs from preset's
recommended_dt=0.0208. Using scene.sim.dt (Genesis owns time). If hooks
oscillate or speed diverges, set SimOptions(dt=0.0208, ...) or pick a
different preset.
```

### Back-compat (kept through v0.5.x, removed in v0.6.0)

- `VehicleConfig.dt` works as a read/write @property aliasing
  `recommended_dt`, emits `DeprecationWarning`.
- `VehicleConfig.from_urdf(dt=...)` and the preset constructors that
  accept `dt=...` forward it to `recommended_dt`, also with
  `DeprecationWarning`.

Migration path:

```python
# before (still works, deprecated)
cfg = car_4w_rwd_ackermann(URDF, dt=0.005, ...)
scene = gs.Scene(sim_options=SimOptions(dt=cfg.dt, ...))

# after
cfg = car_4w_rwd_ackermann(URDF, recommended_dt=0.005, ...)
scene = gs.Scene(sim_options=SimOptions(dt=cfg.recommended_dt, ...))
```

All bundled samples and presets have been migrated. Tests cover both
the new and legacy spellings.

### Files

- `genesis_vehicle/config.py` — field rename + `dt` @property alias +
  `from_urdf(dt=...)` deprecation forwarder
- `genesis_vehicle/core.py` — `_validate_dt_matches_scene` replaced
  with `_resolve_dt_from_scene` (returns scene's dt, warns on mismatch)
- `genesis_vehicle/multi_vehicle.py` — inherits `dt` from prototype
  (already resolved against scene)
- `genesis_vehicle/presets.py` — uses `recommended_dt=` directly
- 4 samples (`quickstart`, `slope_hold`, `multi_env_render`,
  `batched_rollout`) — `cfg.dt` → `cfg.recommended_dt`
- `genesis_vehicle/tests/test_config_resolve.py` — assert on new field

---

## [0.5.30] — 2026-05-26

### Added — `VehiclePhysics` now validates `cfg.dt` against `scene.sim.dt`

Two dts existed in the system without an explicit relationship:

- `VehicleConfig.dt` — used by hooks inside `VehiclePhysics.step()` to
  integrate state across steps (wheel ω in `core.py:349`, stick-slip
  displacement in `stability.py:233`, etc.)
- `SimOptions.dt` — used by Genesis's `scene.step()` to advance the
  rigid-body state.

These MUST match. If they don't, the wheels integrate at one rate and
the chassis integrates at another, and the system oscillates or
diverges (user-visible as "출렁임"). The samples all do
``SimOptions(dt=cfg.dt, ...)`` to wire them together, but nothing
enforced this — a typo or a hand-built scene could silently break.

Fix: `VehiclePhysics.__init__` and `MultiVehiclePhysics.__init__` now
call ``_validate_dt_matches_scene(scene, cfg.dt)``. On mismatch they
raise `ValueError` with both fix options spelled out:

```
ValueError: VehicleConfig.dt (0.02) does not match scene.sim.dt (0.002).
Genesis advances time by scene.sim.dt per scene.step(); if
VehiclePhysics integrates wheel omega and stick-slip displacements at
a different dt, the wheels and the chassis drift apart (oscillation /
velocity divergence). Fix one of:
  - SimOptions(dt=0.02, ...)        # follow the preset
  - cfg.dt = 0.002                  # follow the scene
All bundled samples wire the preset's dt into SimOptions.
```

### Why keep `cfg.dt` at all?

The preset's recommended dt is part of its physical character: tank
presets need 200 Hz (`dt=0.005`) to keep the suspension stack stable,
car presets are happy at 100 Hz. The preset declares its requirement,
the sample wires that into both Scene and Physics — DRY, single source
of truth. The validation in `__init__` is what makes "single source"
real instead of conventional.

---

## [0.5.29] — 2026-05-26

### Performance — substeps=10 across all remaining samples (1.4-2.2× faster)

v0.5.28 dropped `road_loop` and `city_traffic_ego` to substeps=10
(verified safe). Extended the same change to every other sample:

- `quickstart` and `slope_hold` were on substeps=**50** (Genesis-default
  ×2.5). Internal dt of 0.2 ms — total overkill for a 2 Hz suspension
  natural frequency. Dropped to 10 → internal dt = 1 ms.
- `batched_rollout`, `multi_env_render`, `perf_vectorization`,
  `perf_multi_vehicle`, `perf_l2_l3_combined` were on substeps=20.
  Dropped to 10.

Physics verified identical:

- `quickstart`: final pose (12.42, 0.01, 0.12) and speed 5.02 m/s
  (was (12.38, 0.02, 0.12) and 4.90 m/s — same trajectory).
- `slope_hold`: settled roll +20.31° (matches expected +20°), lateral
  slip ≤0.1 mm (was 0.5 mm — still well under the 10 mm threshold).

### Re-bench (headless, pure physics)

| Sample                          | v0.5.27 | v0.5.28 | v0.5.29 | total speedup |
|---------------------------------|--------:|--------:|--------:|--------------:|
| `quickstart`                    | 42      | 42      | **19**  | 2.2×          |
| `slope_hold`                    | 49      | 49      | **23**  | 2.1×          |
| `batched_rollout` n_envs=16     | 37      | 37      | **24**  | 1.5×          |
| `multi_env_render` n_envs=4     | 37      | 37      | **24**  | 1.5×          |
| `road_loop` 16 veh multi_batched| 740     | 447     | **457** | 1.6×          |
| `city_traffic_ego` 8 veh        | 198     | 138     | **140** | 1.4×          |
| `perf_vectorization` n=256      | 35      | 35      | **25**  | 1.4×          |
| `perf_multi_vehicle` K=4 batched| 776     | 776     | **439** | 1.8×          |

`perf_vectorization` n=256 throughput went from 7,200 → **10,300
env-steps/s** at the same physics fidelity.

60 SDK pytest pass.

### Rationale

Genesis's default `substeps=20` (and the older `quickstart`/`slope_hold`
`substeps=50`) was chosen for broad robustness across arbitrary stiff
contact stacks. Our ray-cast wheels never expose Genesis's constraint
solver to wheel-ground contact (we apply the wheel forces as external
forces to the chassis), so the tightest natural frequency the solver
has to integrate is the suspension spring-damper (~2 Hz at K~50 kN/m,
m~375 kg quarter-car). `internal dt = 1 ms` (substeps=10 with dt=0.01,
or substeps=20 with dt=0.02) is the validated floor; going to internal
dt = 2 ms also works for these samples but is at the edge.

---

## [0.5.28] — 2026-05-25

### Performance — `road_loop` and `city_traffic_ego` substeps 20 → 10

Diagnosed why `road_loop` 16 vehicles measured 824 ms/step in headless
mode. Tried in order:

- `enable_collision=False` → 689 ms/step (only 50 ms savings → collision
  was not the bottleneck)
- `use_contact_island=True` → Genesis 0.4.6 kernel compile error
  (`Cannot subscript NdarrayType` in contact_island.py); fail-closed
- `enable_multi_contact=False` → 775 ms/step (regression)
- `max_collision_pairs` bumped → no change (cap wasn't binding)
- `substeps=20 → 10` → **447 ms/step** (1.66× speedup), physics identical
- `substeps=8` → 363 ms/step but car speeds collapse to 0 m/s
  (suspension stack no longer integrates correctly)

`substeps=10` (internal dt = 1 ms) is the floor for this vehicle stack.
The suspension's natural frequency is ~2 Hz (K~50 kN/m at 375 kg
quarter-car mass) so 1 ms is still ≫ enough; Genesis's default
`substeps=20` was 2× the safe margin. The constraint solver runs once
per substep, so the saving is roughly linear.

Applied the same to `city_traffic_ego` (198 → 138 ms/step, −30 %).
Other samples already had small entity counts; changing substeps there
wouldn't move the needle.

| Sample                          | Before (ms/step) | After (ms/step) | Change                |
|---------------------------------|-----------------:|----------------:|-----------------------|
| `road_loop` 16 veh (per_vehicle)| 824              | ~500 (est)      | substeps 20 → 10      |
| `road_loop` 16 veh (multi_batched)| 740            | 447             | substeps 20 → 10      |
| `city_traffic_ego` 8 veh        | 198              | 138             | substeps 20 → 10      |

### Added — `_hud.warn_if_unused_camera`

Helper that prints a one-line warning when a camera exists but neither
`--viewer` nor `--record` is in use. Documents the gotcha (Genesis
pays a per-step renderer-state sync inside `scene.step()` for any
camera in the scene, even one whose `.render()` is never called — see
v0.5.27 for the impact this had on `multi_env_render`'s 70 % overhead).

The samples in this repo already follow the preferred pattern
(`cam = None; if args.viewer: cam = scene.add_camera(...)`), so the
warning is for user code that copies a sample and forgets to gate the
camera. Optional opt-in:

```python
cam = scene.add_camera(...)   # always create
_hud.warn_if_unused_camera(cam, used=args.viewer or args.record,
                           sample="my_sample")
```

---

## [0.5.27] — 2026-05-25

### Fixed — perf summaries had rendering folded into physics ms/step

In every sample that wrote out a `[timing]` / `print_perf_summary` line,
the headless main loop was calling `cam.render()` every few steps even
though the result was thrown away. The reported `ms/step` therefore
mixed physics with render cost, and the gap was huge for samples that
render a large image: `multi_env_render --n_envs 4` measured 106 ms/step
when the actual pure-physics cost is 37 ms/step. ~70 % was wasted
rendering frames into the bit bucket.

### Changed — strict separation of physics vs render in samples

- In **headless** mode (no `--viewer`), no camera is created at all and
  no render call happens in the main loop. The reported `ms/step` is
  now pure physics + `scene.step()`. (Genesis's `scene.step()` does a
  per-step renderer-state sync if any camera exists, so even an unused
  camera measurably slows physics — hence creating none.)
- In **viewer** mode, the main loop renders inline (same as v0.5.26),
  and a separate post-loop standalone render benchmark of 20 frames is
  printed on its own line in the summary block.
- New `_hud.bench_render(cam, n=20)` helper — single `cuda.synchronize()`
  on each side, returns `(ms_per_frame, n)`.
- `_hud.print_perf_summary` gained `render_ms` / `render_n` kwargs.
  Render line is omitted when no camera exists.

### Re-bench (headless, pure physics)

| Sample                          | Before (ms/step) | After (ms/step) | Notes                                |
|---------------------------------|-----------------:|----------------:|--------------------------------------|
| `quickstart`                    | 41–55            | 42              | unchanged (render was rare)          |
| `slope_hold`                    | 49               | 49              | unchanged                            |
| `batched_rollout n_envs=16`     | 36               | 37              | unchanged (cam was None already)     |
| `multi_env_render n_envs=4`     | 106              | **37**          | **−65 %** — render was 70 % of cost  |
| `road_loop` 16 vehicles         | 829              | 824             | unchanged (`scene.step` dominates)   |
| `city_traffic_ego` 8 vehicles   | 208              | 198             | small win                            |
| `perf_vectorization` n=1..1024  | 23→37            | 25→38           | unchanged (already cam-less)         |

### Behavior change for `multi_env_render`

Previously the sample always called `cam.render()` even in headless
("always renders" tag in the README). With this change, headless
`multi_env_render` runs pure physics and prints a clean ms/step;
`--viewer` keeps the cv2 grid HUD behavior unchanged. If you need the
old "render to a tensor every step" behavior for mp4 recording, run
with `--viewer` — the sample's identity (rendering N parallel envs in
one tiled view) is in the viewer path now, not the headless path.

---

## [0.5.26] — 2026-05-25

### Fixed — `slope_hold` settled-roll print had the wrong sign

The ground tilt is created with `euler=(slope_deg, 0.0, 0.0)`, which
rotates the box +slope_deg around X. The car settled on the slope picks
up the same +slope_deg roll. The "expected" value in the settled-pose
print was `{-slope_deg:+.1f}°` (negated) — so the user saw
`roll=+20.30° (expect roll ≈ -20.0° on slope)` and would reasonably
suspect a sign convention bug in the SDK. The bug was in the print
line; the physics was already correct (lateral slip OK = 0.5 mm).

Fix: drop the negation. The settled-roll print now reads
`expect roll ≈ +20.0° on slope` for a +20° wedge.

### Full-sample bench notes (informational, no code change)

Ran every sample headless and observed:

- `quickstart` / `slope_hold` — 41 / 49 ms per step. Stable.
- `batched_rollout n_envs=16` — 35 ms/step → 451 env-steps/s. L3
  batching working as designed.
- `multi_env_render n_envs=4` — 106 ms/step (~10 fps). Higher than
  raw physics because the sample always calls `cam.render()` (that's
  the sample's purpose; see README "always renders" tag).
- `road_loop` 16 vehicles — 754 ms/step with `multi_batched`, 829
  with `per_vehicle` (9 % gap). The solver-choice gap is small because
  `scene.step()` dominates at 16 entities in one scene.
- `perf_vectorization` — 23 → 37 ms across n_envs ∈ [1, 1024]; 646×
  speedup at n_envs=1024 vs n_envs=1. L3 batching is the right axis
  for raw throughput.
- `perf_multi_vehicle` K=4 n_per_kind ∈ [1, 2, 4] — 66 → 200 → 896 ms;
  per-vehicle cost grows super-linearly because Genesis's `scene.step`
  cost in number-of-bodies dominates. Both `per_vehicle` and
  `multi_batched` solvers hit this equally (multi_batched is only
  ~10 % faster), as already documented in that sample's docstring.
- `perf_l2_l3_combined` — same K=N cell measured 64 ms vs 176 ms on
  separate runs. Variance comes from per-subprocess JIT-cache warm
  state, not from the SDK. The reported `gain` columns are still
  meaningful as a ratio within one run, but absolute ms values
  benchmark-to-benchmark are noisy by ±2-3×.

---

## [0.5.25] — 2026-05-25

### Added — `print_perf_summary` end-of-run report on every sample

When a sample finishes — whether it ran to completion or the user ESC-quit
mid-run — it now prints a banner-separated multi-line block instead of a
single `[timing]` line. The header makes the completion state explicit
(`COMPLETED` vs `USER QUIT (ESC)`) so a short run cut at step 12/200 is
obvious at a glance.

```
======================================================================
 multi_env_render  (v0.5.25)  —  USER QUIT (ESC)
======================================================================
  steps      :     42 / 200      ( 21.0% of requested)
  wall time  :   4.21 s
  ms / step  : 100.24
  steps / s  :   9.98
  batch      : 4 env/step  ->  40 env-steps/s
  grid       : 2 x 2   spacing 10.0 m
======================================================================
```

The helper lives in `samples/_hud.py`:

```python
_hud.print_perf_summary(
    sample=f"multi_env_render  (v{sdk_version})",
    completed=not user_quit,
    n_done=n_done, n_target=n_steps, wall=wall,
    batch=n_envs, batch_label="env",
    extra=[...],
)
```

Applied to all six viewer-supporting samples (`quickstart`, `slope_hold`,
`batched_rollout`, `multi_env_render`, `road_loop`, `city_traffic_ego`).
The three `perf_*` benches were left alone — they have their own
purpose-built reports.

### Fixed — `batched_rollout` accuracy when ESC-quit

`batched_rollout` previously computed `wall / args.steps` even when the
measure loop broke early. With an ESC quit at step 30/100 it would
under-report ms/step by ~3.3x. Now tracks `user_quit` through both the
settle and measure loops, uses the actual `n_done` step count, and
short-circuits with a partial summary if the user quits during the
settle phase.

### Smoke check

60 SDK pytest pass. `quickstart` headless prints the new summary block
on completion.

---

## [0.5.24] — 2026-05-25

### Fixed — `--viewer` only showed env 0 when `env_separate_rigid=True`

`multi_env_render --viewer` (and `batched_rollout --viewer`,
`city_traffic_ego --viewer --n_envs N` with `N>1`) was rendering only
one of the parallel envs and updating at far below the requested rate.

Root cause: Genesis's `cam.render()` returns a per-env stack of shape
`(N, H, W, 3)` whenever `env_separate_rigid=True`; the v0.5.23
`_hud.render_hud_frame` only checked `if frame.ndim == 4: frame =
frame[0]`, silently dropping all but env 0. The "only one car visible"
bug followed directly.

### Changed

- `_hud.render_hud_frame` now tiles a 4-D per-env stack into a
  `per_row × ceil(N/per_row)` mosaic. Two new kwargs:
  - `grid_per_row` — cells per row in the mosaic; default
    `round(sqrt(N))`.
  - `max_cell_size` — clamp each cell's longer axis to this many
    pixels before tiling (default 480). Keeps the displayed mosaic
    a sane size regardless of camera resolution.
- `multi_env_render`, `batched_rollout`, `city_traffic_ego` lower the
  per-cell camera resolution to 640×360 when running with
  `env_separate_rigid=True`. Rendering at 1920×1080 per env when each
  cell is downsized to ≤480 px wastes ~10× the GPU + GPU→CPU work and
  was the main cause of the sub-10 fps display rate the user observed.
  Headless and single-env paths keep the original 1920×1080.
- Samples now pass `grid_per_row` matching their build-time
  `n_envs_per_row` so the HUD layout matches the physics-side grid.

### Effect on the reported case

`multi_env_render --viewer --n_envs 4` (per_row=2, spacing 10):
- Before: env-0-only view, ~2.5 fps display (1080p × 4 = 8 MP/frame
  rendered then 7/8 discarded).
- After: 2×2 grid mosaic (960×540 final), per-cell render at 640×360,
  display refresh follows the existing `~25 fps render_every` cadence
  (actual frame-to-frame rate now limited by physics step time, not
  the render).

---

## [0.5.23] — 2026-05-25

### Changed — `--viewer` now opens a cv2 window with live HUD overlay

`--viewer` previously opened Genesis's built-in interactive 3D window
(v0.5.21) but had no way to overlay text — performance numbers, vehicle
state, sample-specific info — on top of the camera frame. Switched to a
cv2-based pattern that does:

- Renders the offscreen camera each step
- Draws a semi-transparent header rectangle
- Overlays a title + per-sample state lines + a perf footer
  (rolling-mean `ms/step` + `fps`)
- Shows via `cv2.imshow`; `ESC` quits

The trade-off: lose Genesis's mouse-rotate interactivity, gain a
fixed-POV camera with informative overlay (the camera's pose is the one
each sample already set for its offscreen rendering).

### Added — `samples/_hud.py`

Minimal helper used by all 6 viewer-supporting samples:

- `PerfMeter` — rolling-mean ms/step over a configurable window (default 60).
  ``tick()`` cost is well under 1 μs; safe to call every step.
- `render_hud_frame(cam, title, lines, perf_ms)` — renders the camera and
  draws the HUD overlay; returns BGR `np.ndarray` for `cv2.imshow`.
- `cv2_show(window, frame)` — wraps `cv2.imshow + cv2.waitKey(1)`;
  returns `False` on `ESC`.
- `cv2_cleanup()` — `cv2.destroyAllWindows()`.

No `pynput` dependency (unlike the umbrella `_demo_hud.py`); cv2 only.
If cv2 isn't importable, `--viewer` falls back to headless with a
warning.

### Per-sample HUD content

| Sample | HUD lines |
|---|---|
| `quickstart`         | t, throttle, pos, speed |
| `slope_hold`         | t, brake, pos, roll, lateral slip |
| `batched_rollout`    | phase, step, grid, speed spread across envs |
| `road_loop`          | step, vehicle/kind count, per-kind speeds |
| `multi_env_render`   | step, grid spec, speed spread across envs |
| `city_traffic_ego`   | step, total batch, ego pose/speed, L2 kinds |

All include a perf footer (`X.XX ms/step (Y fps)`) and an `[ESC] quit`
prompt.

### `ESC`-to-quit

All 6 samples now break out of the main loop on `ESC`. The end-of-run
`[timing]` print uses the actual `n_done` step count (steps that ran
before `ESC`), so the throughput numbers are honest even when the user
cuts the run short.

### No SDK code changes

Pure sample / docs work. 60 SDK tests pass.

---

## [0.5.22] — 2026-05-25

### Added — zero-overhead timing across all non-bench samples

Every sample now prints a `[timing]` line at the end of its main drive
loop showing total wall time, ms/step, and throughput. The measurement
adds essentially zero overhead — one `torch.cuda.synchronize()` before
the loop and one after, no per-step sync. Sample output:

```
[timing] 240 steps in 11.01s  = 45.88 ms/step  (22 steps/s)
[timing] 150 steps in 35.57s  → 237.13 ms/step  (135 vehicle-steps/s, batch=4×8=32 per step)
```

Per-sample variants of the throughput field:

| Sample | extra throughput info |
|---|---|
| `quickstart`         | `steps/s` |
| `slope_hold`         | `steps/s` |
| `batched_rollout`    | `env-steps/s` (existing, unchanged) |
| `road_loop`          | `vehicle-steps/s, solver=<name>` |
| `multi_env_render`   | `env-steps/s, batch <n_envs>` |
| `city_traffic_ego`   | `vehicle-steps/s, batch=N×K` |

The three perf benches (`perf_vectorization`, `perf_multi_vehicle`,
`perf_l2_l3_combined`) are unchanged — they were already comprehensive
benchmarks.

### Removed — `--bench` flag (now always-on)

`road_loop` and `city_traffic_ego` previously gated their wall-time
print behind `--bench` because of a perceived overhead concern. The
overhead turned out to be negligible (single sync before/after), so the
flag is removed and the timing prints unconditionally. Scripts passing
`--bench` to these two samples will need to drop the flag — argparse
will error otherwise.

No SDK code changes; samples only. 60 SDK tests still pass.

---

## [0.5.21] — 2026-05-25

### Fixed — `--viewer` now actually opens a viewer window

Previously the `--viewer` flag on the 6 viewer-supporting samples
(`quickstart`, `slope_hold`, `batched_rollout`, `road_loop`,
`multi_env_render`, `city_traffic_ego`) only added an offscreen
camera that called `cam.render()` each step. That renders into a
GPU tensor but DOES NOT open any visible window — `--viewer` did
nothing the user could see. Bug report from korfriend:

> "gen_vesis/bin/python -m genesis_vehicle.samples.city_traffic_ego
> --viewer  이거 뷰어 안 나오던데"

Fixed by wiring `--viewer` to Genesis's built-in interactive viewer:

```python
viewer_opts = gs.options.ViewerOptions(
    res=(1280, 720),
    camera_pos=(...), camera_lookat=(...), camera_up=(...), camera_fov=...,
) if args.viewer else None
scene = gs.Scene(
    ...,
    viewer_options=viewer_opts,
    show_viewer=args.viewer,
)
```

Each sample's `ViewerOptions` uses the same camera framing as its
existing offscreen camera (side chase-cam for `quickstart`, side view
of the slope for `slope_hold`, top-down for `road_loop` /
`city_traffic_ego` / `multi_env_render`, grid top-down for
`batched_rollout`). The offscreen camera is kept — it still produces
image tensors for inspection / mp4 recording, independent of the
viewer window.

`multi_env_render` also gained a `--viewer` flag (was always
headless-render-only, no window).

### Caveats

- The viewer window is Genesis's, not OpenCV — mouse-rotate, zoom,
  ESC to close.
- WSL / WSLg with a software OpenGL fallback won't open a window; you
  need hardware GL (D3D12 via `GALLIUM_DRIVER=d3d12` on WSL, or native
  Windows / Linux). See `docs/batching.md` notes about WSL GL.

All 60 SDK tests still pass.

---

## [0.5.20] — 2026-05-25

### Fixed — samples now run via direct file path too

Previously every sample required either `python -m genesis_vehicle.samples.X`
from the project root OR a `PYTHONPATH` export. Running them by full
file path —

```bash
gen_vesis/bin/python genesis_vehicle/samples/city_traffic_ego.py --viewer
```

— failed with `ModuleNotFoundError: No module named 'genesis_vehicle'`
because Python only adds the file's own directory to `sys.path`, not
the SDK's parent.

Fixed by a 3-line bootstrap at the top of each of the 9 samples that
prepends the SDK parent directory to `sys.path`:

```python
import sys, pathlib
_SDK_PARENT = str(pathlib.Path(__file__).resolve().parents[2])
if _SDK_PARENT not in sys.path:
    sys.path.insert(0, _SDK_PARENT)
```

For the 3 perf samples (which re-invoke themselves as subprocesses to
get a clean GPU per measurement), the same `_SDK_PARENT` is also
exported as `PYTHONPATH` to the subprocess so the child process can
resolve `genesis_vehicle` regardless of cwd.

Net effect: all samples now work via any invocation pattern —

```bash
python -m genesis_vehicle.samples.X            # module form (always worked)
python path/to/samples/X.py                    # direct file path (new)
PYTHONPATH=... python ...                       # explicit env var (still works)
```

All 60 SDK tests still pass.

---

## [0.5.19] — 2026-05-25

### Docs — first-class samples link from README

The 9 bundled samples were referenced throughout the docs but the
catalog [`samples/README.md`](samples/README.md) wasn't linked from
the main README itself. Fixed:

- **README.md** new "Samples" section between "Bundled Presets" and
  "Documentation". Lists all 9 samples with their `python -m
  genesis_vehicle.samples.<name>` invocation in a single code block,
  short note about `--viewer` vs headless-by-design benches, and
  pointer to `samples/README.md` for full catalog with descriptions.
- **README.md** Documentation table gets a `samples/README.md` row at
  the top alongside `docs/index.md`.
- **docs/index.md** TOC gains a "See all 9 runnable samples" entry
  linking to `samples/README.md`.

No code changes.

---

## [0.5.18] — 2026-05-25

### Docs — align sample references with the v0.5.17 viewer additions

After v0.5.17 added `--viewer` to `quickstart`, `slope_hold`, and
`batched_rollout` (and made it explicit that the three perf benches
are headless by design), the doc references hadn't yet been updated.
This release fixes those:

- **README.md** "Three-axis vectorization" section — closing paragraph
  now separates "runnable benchmarks (headless)" and "visual demos
  (with `--viewer`)" into two groups, and adds the
  `city_traffic_ego.py` reference under L2 × L3 visuals (was missing).
- **docs/quickstart.md** — TL;DR shows BOTH `python -m
  genesis_vehicle.samples.quickstart` and the `--viewer` variant.
- **docs/batching.md** — L1, L2, L3, and L2 × L3 sections all flag
  whether each referenced sample has `--viewer` or is headless by
  design. New "See also" entry under L2 × L3 points to both the
  headless `perf_l2_l3_combined.py` AND the visual
  `city_traffic_ego.py`. Performance reference table gets a row for
  `city_traffic_ego.py`.

No code changes.

---

## [0.5.17] — 2026-05-25

### Added — `--viewer` for the 3 samples that didn't have it

[`samples/quickstart.py`](samples/quickstart.py),
[`samples/slope_hold.py`](samples/slope_hold.py),
[`samples/batched_rollout.py`](samples/batched_rollout.py) now accept
`--viewer`:

- **quickstart**: side chase-cam trailing the car.
- **slope_hold**: side view showing the slope tilt + the brake-locked
  car (so the v0.5.7 stick-slip fix is visible as the car staying put).
- **batched_rollout**: lays the N envs out on a grid via
  `env_separate_rigid=True` + `env_spacing` so all parallel rollouts are
  visible in one frame. (Same mechanism as `multi_env_render.py`.) The
  flag explicitly notes that rendering distorts the throughput numbers
  — use `perf_vectorization.py` for clean bench timings.

### Clarified — three perf samples are intentionally headless

[`samples/perf_vectorization.py`](samples/perf_vectorization.py),
[`samples/perf_multi_vehicle.py`](samples/perf_multi_vehicle.py),
[`samples/perf_l2_l3_combined.py`](samples/perf_l2_l3_combined.py)
docstrings now explicitly say they have NO `--viewer` flag because
camera rendering adds per-step overhead that distorts the throughput
numbers they're trying to measure. Each one points to the matching
visual sample for the same scenario.

### Viewer support across all 9 samples

| Sample | --viewer | Notes |
|---|---|---|
| quickstart           | ✓ (new) | side chase-cam |
| slope_hold           | ✓ (new) | side view of tilted ground + car |
| batched_rollout      | ✓ (new) | env grid via env_separate_rigid |
| road_loop            | ✓        | top-down loop |
| perf_vectorization   | ✗ headless by design | bench |
| multi_env_render     | always renders | env grid is the demo |
| perf_multi_vehicle   | ✗ headless by design | bench |
| perf_l2_l3_combined  | ✗ headless by design | bench |
| city_traffic_ego     | ✓        | top-down highway |

No SDK code changes; samples only.

---

## [0.5.16] — 2026-05-25

### Added — `city_traffic_ego` sample (highway ego + traffic scenario)

[`samples/city_traffic_ego.py`](samples/city_traffic_ego.py) — the
reference setup for "autonomous-driving ego in surrounding traffic"
workflows. A 4-lane straight highway on a flat plane with:

- **1 ego** — bright red AWD car (sportier body)
- **7 traffic agents** — 3 RWD blue coupes, 3 FWD small dark-red
  sedans, 1 yellow 6-wheel truck
- **Lane-keeping P controller** on every vehicle (target lane center
  + small yaw correction)
- **Top-down camera** framing the action area
- **`--n_envs N`** enables the L2 × L3 combined batching pattern —
  one MPPI candidate (or RL rollout) per env, all 8·N vehicles
  processed in batched compute per kind

Measured (RTX 5070 Laptop, 3-second drive, 150 steps):

| n_envs | total batch | ms/step | vehicle-steps/s |
|-------:|------------:|--------:|----------------:|
|      1 |           8 |   205.9 |              39 |
|      4 |          32 |   234.9 |             136 |
|     16 |         128 |   ~290  |            ~440 |

n_envs=4 costs +14% ms/step for 3.5× more vehicle-steps/s — the
expected L2 × L3 combined scaling.

The 8 vehicle fleet groups into 4 kinds in `MultiVehiclePhysics`:
`K per kind = [1, 3, 3, 1]` (ego is its own group because it uses
a different URDF / cfg from the RWD traffic).

### Usage

```bash
python -m genesis_vehicle.samples.city_traffic_ego                    # 1 env, visual
python -m genesis_vehicle.samples.city_traffic_ego --viewer
python -m genesis_vehicle.samples.city_traffic_ego --n_envs 16        # 16 parallel scenarios
python -m genesis_vehicle.samples.city_traffic_ego --n_envs 16 --bench
```

### Docs

- `samples/README.md` — row #9 added.

---

## [0.5.15] — 2026-05-25

### Added — `docs/batching.md` (L1 / L2 / L3 reference)

The three orthogonal vectorization axes are now a first-class topic in
the docs and README:

- **README.md** — new top-level "Three-axis vectorization (L1 / L2 / L3)"
  section above "Installation" with the headline table, rule-of-thumb
  decision guide, and measured numbers.
- **docs/batching.md** (new) — full treatment: each axis explained, when
  it matters, when it doesn't, code patterns, measured speedups
  (RTX 5070 Laptop), L2 × L3 combined pattern, decision matrix, what's
  NOT batched.
- **docs/concepts.md** — "Batched by default" section now sketches the
  three axes and links to `batching.md`.
- **docs/index.md** — added "Pick the right batching axis" entry to
  the landing TOC.

No code changes; doc-only release.

---

## [0.5.14] — 2026-05-25

### Added — L2 × L3 combined batching (`n_envs > 1` for MultiVehiclePhysics)

`MultiVehicleKindPhysics` and `MultiVehiclePhysics` now accept an
``n_envs`` parameter. The internal compute pipeline batch dimension
becomes ``N * K`` (parallel envs × vehicles per kind), and all I/O
collapses to ONE batched solver call per quantity per kind:

  - State reads (`get_links_{pos,quat,vel,ang}`): return ``(N, K, 3)``,
    flattened to ``(NK, 3)`` for compute.
  - Force / torque writes: compute output ``(NK, 3)`` reshaped to
    ``(N, K, 3)`` and applied in one ``apply_links_external_force`` call.
  - VisualSync: K per-entity objects, each built with ``n_envs=N``;
    compute output sliced ``(N, n_wheels)`` per entity.

```python
mphys = MultiVehiclePhysics(scene, vehicles, n_envs=64)
mphys.step([VehicleInputs(throttle=tensor_of_shape_N, ...) for _ in vehicles])
```

Per-vehicle inputs accept ``(N,)`` tensors so each parallel env can have
distinct controls (RL-style per-env action diversity).

**n_envs default is 1**, so existing v0.5.11-v0.5.13 call sites continue
to work unchanged.

### Added — `perf_l2_l3_combined` sample

[`samples/perf_l2_l3_combined.py`](samples/perf_l2_l3_combined.py)
sweeps a 2D ``(K, N)`` grid and reports a scaling table. Each cell is
a fresh subprocess (clean GPU state). Sample result on RTX 5070 Laptop:

|  K |  N | total | ms/step | per veh (μs) | gain |
|---:|---:|------:|--------:|-------------:|-----:|
|  1 |  1 |     1 |   26.31 |       26,315 | 1.0× |
|  1 |  4 |     4 |   37.83 |        9,458 | 2.8× |
|  2 |  1 |     2 |   35.98 |       17,992 | 1.5× |
|  2 |  4 |     8 |   45.79 |        5,724 | **4.6×** |

The L2-only (K=2) and L3-only (N=4) gains (1.5×, 2.8×) multiply close
to the combined gain (4.6× ≈ 1.5×2.8 = 4.2×, with a small bonus from
better GPU utilization at larger batch). This is the headline pattern
for autonomous-driving simulation: K vehicles per scenario (ego + N_t
traffic) × M parallel scenarios.

### Use case decision matrix

| Scenario | Solver |
|---|---|
| 1 vehicle, RL rollouts | `VehiclePhysics(n_envs=N)` |
| K vehicles in 1 visible scene (traffic) | `MultiVehiclePhysics(n_envs=1)` |
| 1 vehicle but parallel scenarios | `VehiclePhysics(n_envs=N)` |
| **K vehicles × N parallel scenarios** | **`MultiVehiclePhysics(n_envs=N)`** |

### Migration

None — `n_envs=1` is the default; existing code keeps working.

All 60 SDK tests pass; new sample smoke-tested.

---

## [0.5.13] — 2026-05-25

### Added — `perf_multi_vehicle` sample (L2 solver comparison)

[`samples/perf_multi_vehicle.py`](samples/perf_multi_vehicle.py) — the
L2 counterpart to `perf_vectorization.py`. Sweeps `--n_per_kind` and
runs the same 4-kind fleet under both solvers (`per_vehicle` and
`multi_batched`) in fresh subprocesses, prints a scaling table:

| n_per_kind | total | per_vehicle (ms) | multi_batched (ms) | speedup |
|-----------:|------:|-----------------:|-------------------:|--------:|
|          1 |     4 |            75.27 |              78.84 |   0.95× |
|          2 |     8 |           208.20 |             183.25 |   1.14× |

At K=1 (4 vehicles, 1 per kind) `multi_batched` is slightly SLOWER —
the batching machinery has setup overhead with nothing to batch (each
"kind" group is just 1 vehicle). From K=2 upward the gap widens; in
practice expect 1.1-1.2× speedup for realistic multi-vehicle scenes.

Use this script to:
- Decide which solver to use for your specific fleet size + kind mix.
- Catch regressions if either solver's perf drifts.

The output makes it concrete that L2 batching is helpful but bounded
(see also `MultiVehiclePhysics` docstring) — for pure throughput on
one vehicle kind, n_envs > 1 (L3) remains the right tool.

### Docs

- `samples/README.md` — added row #7.

---

## [0.5.12] — 2026-05-25

### Fixed — `MultiVehicleKindPhysics` now supports VisualSync

v0.5.11 disabled the internal VisualSync entirely because the proto
`VehiclePhysics(n_envs=K)` it inherits from would issue a
`set_dofs_position` with shape `(K, n_dofs)` into a scene actually
built with `n_envs=1`, which Genesis rejects (the K dim is "vehicles
in this env", not "parallel envs").

Fix: build K independent `VisualSync` instances (one per entity, each
`n_envs=1`). After the batched compute pipeline produces (K, n_wheels)
spin / steer / suspension targets, the visual step slices them into K
per-entity (1, n_wheels) chunks and dispatches K small
`set_dofs_position` calls — a Python loop, but each call is tiny.

Measured overhead: ~34 ms added to the 16-vehicle road_loop step
(760 → 794 ms / step). Net L2 speedup vs per_vehicle (848 → 794)
shrinks slightly from 10% to ~6%, but every vehicle now gets correct
wheel-spin and steering visuals — making `multi_batched` a drop-in
replacement for `per_vehicle` in visual demos.

### Migration

None — same `MultiVehiclePhysics` constructor signature. Existing code
written against v0.5.11 keeps working and gets visuals for free.

---

## [0.5.11] — 2026-05-25

### Added — `MultiVehiclePhysics` (L2 cross-vehicle batching)

New SDK class for the case where K vehicles share ONE Genesis scene at
DIFFERENT positions (traffic, multi-agent, MPPI candidate fleets). The
default ``VehiclePhysics`` is one-per-entity, so K vehicles in the same
scene meant a Python loop of K ``step()`` calls per simulation step.

``MultiVehiclePhysics`` groups vehicles by URDF / cfg identity and runs
ONE batched compute pipeline per kind:

- Pacejka tire model: one call over ``(K_kind, n_wheels)`` instead of
  K_kind separate calls.
- Stability hooks: one batched invocation per hook per kind.
- Force + torque application: one ``apply_links_external_force`` call
  with K_kind link indices instead of K_kind separate calls.

The compute batching uses the rigid solver's existing batched APIs
(``get_links_pos``, ``get_links_quat``, ``get_links_vel``,
``get_links_ang``) so cross-vehicle reads are also a single call per
quantity per kind.

```python
from genesis_vehicle import MultiVehiclePhysics, VehicleInputs

# Build the scene as usual: K different entities, each spawned at its own
# pose, each with its own sensor. SHARE the cfg instance across vehicles
# of the same kind — MultiVehiclePhysics groups by ``id(cfg)``.
cfg_per_kind = [preset_fn(urdf) for ...]
vehicles = []
for k, (urdf, preset_fn) in enumerate(kinds):
    for _ in range(K):
        ent, sens, _ = add_vehicle(scene, urdf, preset_fn=None, pos=(...))
        vehicles.append((ent, sens, cfg_per_kind[k]))

scene.build(n_envs=1)
mphys = MultiVehiclePhysics(scene, vehicles)
for step in range(N):
    mphys.step([VehicleInputs(...) for _ in vehicles])
    scene.step()
```

### Performance

L2 batching helps but is bounded by Genesis's per-entity ``scene.step()``
cost (which dominates as the number of in-scene entities grows). For a
16-vehicle ``road_loop`` scenario the measured speedup is **~10%**
(848 → 760 ms/step on an RTX 5070 Laptop).

For dramatic batching speedups, ``n_envs > 1`` (L3) is still the right
tool — see ``samples/perf_vectorization.py`` (44× at n_envs=64). L2's
value is enabling multi-vehicle scenes (visualization, multi-agent
interaction, mixed-kind comparisons) at a modest extra cost vs the
per-vehicle loop, NOT replacing L3 for pure-throughput RL/MPPI.

### Sample integration

``samples/road_loop.py`` gains ``--solver {per_vehicle, multi_batched}``
(default ``per_vehicle``) and ``--bench`` flags so users can compare the
two on the same 4-kind, K-per-kind fleet:

```bash
python -m genesis_vehicle.samples.road_loop --n_per_kind 4 --bench --solver per_vehicle
python -m genesis_vehicle.samples.road_loop --n_per_kind 4 --bench --solver multi_batched
```

### Known limitations

- `MultiVehicleKindPhysics` disables its internal `VisualSync` (the
  visual joint updates would need K separate VisualSync objects, not
  in v0.5.11). Wheels won't visibly spin in the chase-cam view; chassis
  motion is correct.
- Vehicles of the SAME kind must share the SAME cfg INSTANCE (not just
  the same URDF) — group by passing `cfg_per_kind[k]` instead of calling
  the preset fresh per vehicle. `MultiVehiclePhysics` groups by
  `id(cfg)`.

All 60 SDK tests pass; no changes to `VehiclePhysics` or other existing
APIs.

---

## [0.5.10] — 2026-05-25

### Added — `multi_env_render` sample

[`samples/multi_env_render.py`](samples/multi_env_render.py) closes the
loop on the v0.5.9 batching story: ``perf_vectorization`` shows you the
44× speedup at ``n_envs=64`` as numbers, but until now there was no
way to actually SEE what those 64 parallel rollouts were doing —
Genesis simulates them at the same world coordinates (the env axis is
a "parallel universe" axis, not a spatial one).

The fix is a Genesis feature pair we hadn't exposed in samples yet:

```python
scene = gs.Scene(
    vis_options=gs.options.VisOptions(env_separate_rigid=True, ...),
    ...,
)
scene.build(
    n_envs=N,
    env_spacing=(dx, dy),     # visualization-only offset
    n_envs_per_row=K,         # √N by default
)
```

The renderer then offsets each env's rigid entities to its grid cell.
Physics is unchanged (still parallel universes at the same coords);
only the visualization adds the per-env transform. The overhead
camera frames the entire grid, so MPPI candidate trajectories or RL
rollout diversity is visible at a glance.

Per-env throttle / steer are randomized so every cell looks different —
useful as a sanity check that batched simulation isn't accidentally
making all envs identical.

### Docs

- `samples/README.md` updated with row #6.

---

## [0.5.9] — 2026-05-24

### Added — two more samples (`road_loop`, `perf_vectorization`)

- [`samples/road_loop.py`](samples/road_loop.py) — multi-vehicle visual
  demo. 4 distinct kinds (FWD red sedan, RWD blue coupe, AWD green SUV,
  yellow 6-wheel truck), `--n_per_kind` each, all driving a circular
  track under constant Ackermann steering. Top-down camera frames the
  whole fleet. URDFs are generated parametrically at runtime to a
  tempdir so the demo stays self-contained.

- [`samples/perf_vectorization.py`](samples/perf_vectorization.py) —
  `n_envs` batching speedup benchmark. Sweeps
  `n_envs ∈ [1, 4, 16, 64, 256, 1024]` (one fresh subprocess per
  measurement to keep GPU state clean) and prints a scaling table.
  Typical result on an RTX 5070 Laptop:

  | n_envs | ms / step | env-steps / s | per env (μs) | speedup |
  |-------:|----------:|--------------:|-------------:|--------:|
  |      1 |     26.28 |            38 |       26 277 |    1.0× |
  |      4 |     36.78 |           109 |        9 196 |    2.9× |
  |     16 |     37.28 |           429 |        2 330 |   11.3× |
  |     64 |     37.86 |         1 691 |          592 |  44.4× |

  `ms / step` stays roughly constant from n_envs=4 upward — Genesis +
  the SDK saturate the GPU, so each additional parallel env is nearly
  free. This is the headline batching benefit that RL / MPPI workloads
  exploit. The sweep is honest (each row a separate GPU process) rather
  than relying on simulated kernel-launch overhead.

### Docs

- `docs/api-reference.md` runnable-references callout extended to list
  all five samples.
- `samples/README.md` updated.

---

## [0.5.8] — 2026-05-24

### Added — bundled `samples/` directory

Three minimal, self-contained reference scripts ship with the SDK so
that `pip install genesis-vehicle` is immediately runnable without
hunting for assets or umbrella-level helpers:

- [`samples/quickstart.py`](samples/quickstart.py) — minimum-viable
  call pattern (preset → `add_vehicle` → step loop). Drives a car
  forward for 5 s on flat ground.
- [`samples/slope_hold.py`](samples/slope_hold.py) — `StaticFrictionLock`
  side-slope hold regression check. Reports OK / REGRESSION based on
  lateral slip over 10 s under `brake=1.0`.
- [`samples/batched_rollout.py`](samples/batched_rollout.py) — batched
  `n_envs > 1` API for RL / MPPI. Demonstrates per-env random controls
  and per-step throughput.

Bundled asset:
- [`samples/urdf/car_4w.urdf`](samples/urdf/car_4w.urdf) — minimal
  1.2 t 4-wheel RWD sedan, primitive geometry only (no mesh files),
  follows project URDF conventions.

Run as Python modules:

```bash
python -m genesis_vehicle.samples.quickstart
python -m genesis_vehicle.samples.slope_hold --slope 20
python -m genesis_vehicle.samples.batched_rollout --n_envs 256
```

Docs (`quickstart.md`, `stability-profiles.md`, `api-reference.md`)
now cross-reference the samples as runnable examples.

### Changed — `"control"` profile is now uniform across vehicle kinds

`stability_hooks_for_profile("control", vehicle_kind=...)` now returns
the same hook set (`[RollingResistance, LowSpeedRegularizer,
StaticFrictionLock]`) regardless of `vehicle_kind`. Previously only
`vehicle_kind="tank"` got the lock — `vehicle_kind="car"` was hook[0:2]
only. That was a real footgun: any preset car on a slope (or after an
impact-induced lateral velocity) would creep indefinitely under brake.

The new stick-slip lock (v0.5.7) is cheap enough that there's no
reason to leave cars without it. The truck preset's manual `append` of
the lock is removed since the base profile now includes it.

`vehicle_kind` is kept as a parameter for forward compatibility with
future vehicle-specific tweaks, but currently has no effect on the
profile hook set.

Test update: `test_profile_control_car_includes_static_friction_lock`
replaces the pre-v0.5.8 `..._returns_two_hooks` test.

---

## [0.5.7] — 2026-05-24

### Fixed — `StaticFrictionLock` is now true static friction (stick-slip)

User physics critique on v0.5.6 (the tanh velocity-damper version): a
real asphalt-tire contact at μ=1.0 should hold a vehicle stationary on
any slope where μ > tan(slope_angle) — for a 20° slope that's μ > 0.36,
well within μ=1.0. v0.5.6 still showed ~5 cm drift over 10 s at 20°
because `F = -μN·tanh(v/scale)` is fundamentally a *kinetic* friction
model: any non-zero `v` produces an opposing force, but `v=0` produces
`F=0`. Equilibrium on a slope therefore requires non-zero v_lat — visible
creep, even though physics says the vehicle should be motionless.

### Changes

- `genesis_vehicle/strategies/stability.py` — rewrote `StaticFrictionLock`
  as a **position-anchored stick-slip** model:
  - When the lock first engages on a wheel, the contact position becomes
    its anchor (displacement `d = 0`).
  - Each step while active: integrate `d += v · dt`; compute force as a
    spring + damper: `F = -K_spring·d - K_damp·v`.
  - Project `(F_long, F_lat)` onto the per-wheel friction ellipse (same
    form as `tire_models/pacejka.py`).
  - If the projection clamps (i.e., wheel is slipping), advance the anchor
    so the next-step spring force matches the friction limit — kinetic-
    friction behaviour without spring runaway.
  - Otherwise (stuck), the spring quietly absorbs any external lateral
    force up to μN. Vehicle is **truly stationary**.
  - New tuning parameters: `k_spring` (default 500_000 N/m per wheel,
    around the explicit-Euler stability limit at dt=0.02 for a 5 t
    chassis) and `k_damp` (default 20_000 N·s/m per wheel, ~half-critical
    at the natural frequency).
  - Legacy `hold_k`, `hold_k_lat`, `slip_scale` constructor kwargs are
    REMOVED (see BREAKING section below).
- `genesis_vehicle/core.py` — added `ctx.dt: float` to PipelineContext
  so hooks can integrate per-step state (the stick-slip lock needs it).
- `genesis_vehicle/presets.py` — tank and truck preset call sites updated
  to the new signature (`StaticFrictionLock(brake_thr=0.3, v_thr=0.5)`,
  no `hold_k` arg).
- `genesis_vehicle/urdf.py` + `__init__.py` — removed deprecated alias
  `parse_inertia_max_principal_genesis` (was a back-compat shim from
  v0.4.x → v0.5.0 rename; no longer needed in pre-1.0).

### Verified — `samples/slope_lateral_slip.py` truck preset

| slope    | v0.5.5 (1D lock)  | v0.5.6 (2D tanh) | v0.5.7 (stick-slip) |
|---|---|---|---|
| flat 0°  | -72 mm            | 0 mm             | **0.0 mm**          |
| 20°      | -114 mm           | -54 mm           | **2.9 mm**          |
| 30°      | (not tested)      | (not tested)     | **0.4 mm**          |
| 40°      | (rolls over)      | (rolls over)     | (rolls over)        |

The 40° case is a real physical limit: μ=1.0 → max-hold-able slope ≈
arctan(1.0) = 45°, but a high-COG 5 t truck tips over at ~35-40° well
before reaching the friction limit. The lock cannot prevent tip-over
(would need a separate "rollover restraint" hook, out of scope).

### BREAKING — Migration required

`StaticFrictionLock` no longer accepts `hold_k`, `hold_k_lat`, or
`slip_scale`. They were proportional / tanh gains in the prior velocity-
damper models (v0.5.5, v0.5.6) and have no meaning in the new stick-slip
model. Replace with `k_spring` / `k_damp` (or drop them and use the
defaults, which are tuned for typical 1-5 t vehicles):

```python
# Before:
StaticFrictionLock(brake_thr=0.3, v_thr=0.5, hold_k=400_000.0)
# After:
StaticFrictionLock(brake_thr=0.3, v_thr=0.5)
# or explicitly:
StaticFrictionLock(brake_thr=0.3, v_thr=0.5,
                   k_spring=500_000.0, k_damp=20_000.0)
```

Pre-1.0 SDK — no deprecation shim. Bundled presets (tank, truck) have
been updated.

---

## [0.5.6] — 2026-05-24

### Fixed — `StaticFrictionLock` 2D extension (lateral slip on slopes)

Student bug report from MPPI work: a truck holding `brake=1.0` on a 20°
side slope continued to creep downhill ~11 cm over 10 s, despite the
control-profile stability hooks all being active. Root cause:
``StaticFrictionLock`` only wrote ``ctx.F_long``; ``ctx.F_lat`` came
straight from Pacejka, which is ill-conditioned near rest because
``alpha = atan2(v_lat, max(|v_long|, eps_v=0.5))`` artificially
compresses the slip angle (so ``F_lat`` from the magic formula was
much smaller than what a real tire would produce at rest).

### Changes

- `genesis_vehicle/strategies/stability.py` — `StaticFrictionLock` now:
  - Activates on planar wheel speed ``sqrt(v_long² + v_lat²) < v_thr``
    (was: ``|v_long| < v_thr`` only).
  - Overrides both ``F_long`` and ``F_lat`` with a smooth saturated
    hold ``-mu·N · tanh(v / slip_scale)`` per axis, then projects onto
    the per-wheel friction ellipse — same form as the Pacejka clamp.
  - Switched from `clamp(-hold_k · v, ±mu·N)` (proportional + hard
    clamp) to `tanh(v / slip_scale)`: the original form was a
    bang-bang controller whose effective gain in the saturated regime
    exceeded the explicit-Euler stability bound (`K_total · dt / mass
    = 9.6` for the truck preset, well above the ~2 stability limit) —
    fine in 1D where ``v_long`` was always tiny, but on a side slope
    the sustained lateral gravity excited the discretization into a
    growing oscillation. The tanh form keeps `|F| ≤ mu·N` smoothly,
    eliminating overshoot.
  - New `hold_k_lat` parameter (defaults to `hold_k`) and `slip_scale`
    (defaults to `v_thr / 5` — the stable upper limit on the truck
    preset at dt=0.02).
- `_version.py`: 0.5.5 → 0.5.6.
- `docs/pipeline-and-hooks.md`, `docs/stability-profiles.md`: updated
  hook descriptions.

### Verified — `samples/slope_lateral_slip.py`

| slope | pre-patch slip | post-patch slip | verdict |
|---|---|---|---|
| flat (0°)     | -72 mm  | **0 mm**   | OK |
| 10° side      | n/a     | **32 mm**  | NO BUG OBSERVED (under 80 mm threshold) |
| 20° side      | -114 mm | **54 mm**  | NO BUG OBSERVED |

### Known limitation

The tanh saturated form is a pure velocity damper: any sustained external
lateral force (gravity component on a slope) produces an equilibrium drift
velocity proportional to (force) / (gain near v=0). At the maximum stable
gain, this is ~5-10 mm/s on the truck preset at 20°. Truly zero-drift
hold on steep slopes (> ~25°) would require a position-tracked stick-slip
model (store a per-wheel anchor at lock-on time, spring + damper to it,
release when force exceeds the static-friction circle). Filed as future
work; current behavior is the standard fix for the Real2Sim / MPPI use
cases that motivated the bug report.

### Migration

`StaticFrictionLock(brake_thr=..., v_thr=..., hold_k=...)` continues to
work — `hold_k_lat` defaults to `hold_k`, `slip_scale` defaults from
`v_thr`. No call-site changes required in `presets.py` or external code.

---

## [0.5.5] — 2026-05-18

### Changed — `truck_6w_partial_ackermann` brake behavior

User report from the Truck6w demo: pressing SPACE (brake) while the
scenario kept driving throttle on produced visibly slow deceleration —
the brake was working mechanically (~1.8 m/s² decel for a 5-ton truck
with throttle still applied), but not enough to feel responsive.

Two preset tweaks:

- `t_brake_max` 8000 → 15_000 N·m. Realistic emergency-brake decel for a
  5-ton chassis is ~4-5 m/s²; the old value capped at ~2 m/s² (and that
  was while fighting throttle). Now ~4 m/s² with throttle off.
- The `"control"` stability profile for the truck preset now also includes
  `StaticFrictionLock(brake_thr=0.3, v_thr=0.5, hold_k=400_000)` (same
  pattern as the tank preset). The truck used to creep at < 0.5 m/s after
  the brake's `tanh(omega/0.5)` smoothing weakened — now it holds at rest.

The `"raw"` and `"research"` profiles get no StaticFrictionLock (consistent
with their "no hooks" semantics).

### Demo behavior — `GeneVehicle_Truck6w/demo_drive.py`

SPACE is now an **emergency brake**: it forces throttle to 0 in addition
to setting brake = 1. Mirrors real-car panic-brake semantics (brake pedal
overrides the accelerator). The scenario throttle resumes when SPACE is
released. Reason for the change: the previous behavior (SPACE adds brake
on top of scenario throttle) made `T_drive` and `T_brake` fight each other
and decelerated very gently — confusing to the human.

For raw "brake-while-throttling" research, hold SPACE and observe — but be
aware that the scenario throttle is now zeroed under SPACE.

---

## [0.5.4] — 2026-05-18

### Documentation — URDF steer joint axis recommendation

Added a recommendation in `docs/physics-contracts.md` §7.4: new URDFs
should declare steer joint axes as `<axis xyz="0 0 -1"/>` so that
`+joint_angle` and `+steer` (user-facing ISO 8855) share the same sign
domain. The SDK's `VisualSync` still compensates for either axis convention
via `visual_cmd = -phys * sign`, so existing URDFs with `(0, 0, 1)` (e.g.
JMK) keep working — the recommendation is only for new URDFs.

### Fixed in repo
- `GeneVehicle_Truck6w/urdf/truck_6w.urdf` — steer joint axes changed from
  `(0, 0, 1)` to `(0, 0, -1)` to match the recommendation. Visual behavior
  was already correct (v0.5.3 fix), but the URDF now follows the
  recommended convention internally.

---

## [0.5.3] — 2026-05-18

### Fixed — steer wheel visual rotated the wrong direction

`VisualSync` was driving the steer joints **opposite to the physics-side
steering direction** for both URDF axis conventions:

- URDF axis `(0, 0, 1)` (e.g. truck preset, JMK URDF) — visual wheels rotated LEFT when physics rotated RIGHT, and vice versa.
- URDF axis `(0, 0, -1)` (HJW URDF) — same inversion.

The old formula `visual_cmd = phys * sign` assumed the only conversion
needed was the URDF axis flip captured in `sign`. It missed that the
physics-side `steer_per_wheel` is **opposite-handed** from the URDF joint
convention to begin with (physics +θ = right turn = CW from above; URDF
axis `(0,0,1)` +joint = CCW from above). The correct formula is
`visual_cmd = -phys * sign`:

- axis `(0, 0, 1)`: `sign=+1` → `visual_cmd = -phys` → joint goes CW for +phys → right turn visual ✓
- axis `(0, 0, -1)`: `sign=-1` → `visual_cmd = +phys` → joint goes CW for +phys → right turn visual ✓

Demos affected: HJW (4-wheel car, axis -1) and Truck6w (axis +1). The
4-wheel HJW demo never had the wheel direction verified visually; the
truck demo just hit it (front wheels turned left when the truck arced
right). KDU (skid-steer, no steer joints) and JMK demos are unaffected.

---

## [0.5.2] — 2026-05-18

### Changed — `truck_6w_partial_ackermann` preset

- **Drive layout fixed to match real trucks.** Was `AWD` with uniform 1/6 split across all 6 wheels (front-axle steered wheels were also driven). Now `RWD(driven_axles=(1, 2))` — only the middle and rear axles drive, the front axle is steered-only. This matches conventional truck drivetrain layouts (front steer, rear/mid drive) and stops the visible weirdness of the steering wheels also producing forward torque.
- **Torque sized for the truck's mass.** `t_drive_max` 1500 → 10_000 N·m (was producing ~0.13 m/s² acceleration on a 5-ton truck after rolling resistance — barely visible). `t_brake_max` 3500 → 8000 N·m for symmetry. At full throttle the truck now hits ~2-3 m/s² acceleration without wheelspin (Pacejka peak per driven wheel at equilibrium ≈ 8800 N vs ~2500 N required).

---

## [0.5.1] — 2026-05-18

### Fixed — preset vehicles could not start from rest under throttle

The `"control"` stability profile built `LowSpeedRegularizer` with
`disable_when_control_active=False` (set in v0.3.0). At `v=0, omega=0`,
the regularizer:

1. Scaled `F_long` and `F_lat` by `moving = 0` → no propulsion force.
2. Set `omega_pull_factor = 1`, `omega_pull_target = v_long / radius = 0`
   → forced `omega` back to 0 every step.

Combined: any preset vehicle (`car_4w_rwd_ackermann`, `tank_10w_skid_belt`,
`truck_6w_partial_ackermann`, …) issuing `throttle > 0` at rest would
spin its wheels for one step and then have `omega` snapped back to 0.
**Vehicle stuck at rest indefinitely.** Discovered while running the
6-wheel truck demo where the truck wouldn't accelerate.

### Reverted — `disable_when_control_active=True` is back in `"control"` profile

This restores the original HJW behavior: the regularizer is off when the
user is actively throttling or braking. The vehicle can accelerate from
rest as expected. The regularizer still fires when the chassis is at
rest with no input (suppressing drift jitter), which is its original
intent.

### Trade-off note

The v0.3.0 change was made on the hypothesis that an always-on
regularizer would help MPPI low-speed uphill oscillation. That
hypothesis was never validated and the resulting behavior breaks every
demo. MPPI users who want an always-on regularizer can opt in
explicitly via `stability="research"` + a custom hook list:

```python
cfg.stability_hooks = [
    RollingResistance(),
    LowSpeedRegularizer(disable_when_control_active=False),
]
```

### Test update

`test_profile_control_uses_mppi_friendly_lowspeed_default` →
`test_profile_control_disables_regularizer_under_throttle`. Same assertion
flipped (`False` → `True`).

---

## [0.5.0] — 2026-05-18

### Performance — vectorized per-wheel pipeline

The Python `for i in range(n_wheels)` loop inside `VehiclePhysics.step()` is gone. All per-wheel work — suspension N, wheel-frame fwd/lat transforms, slip, tire force, stability hooks, omega update, force accumulation — is now a single batched tensor op set operating on `(n_envs, n_wheels)` tensors.

Measured on the KDU 10-wheel tank interactive demo:
- Before (v0.4.4): physics ≈ 50 ms / step (chase-cam interactive, `--profile` mode)
- 250+ CUDA kernel launches per step (10 wheels × ~25 ops per wheel)
- At `n_envs=1`, launch overhead (~10-30 µs each) dominated the actual GPU work, costing 5-10 ms in pure dispatch overhead alone.

After v0.5.0: ~25-30 kernel launches per step regardless of `n_wheels`. Helps interactive single-env use (where overhead was the bottleneck) AND batched RL/MPPI (cleaner code, same speed or better).

### BREAKING — stability hook API

- `StabilityHook.apply_post_tire(ctx, wheel_idx)` → `StabilityHook.apply_post_tire(ctx)`. No more per-wheel index. All `ctx.F_long`, `ctx.F_lat`, `ctx.N`, `ctx.v_long`, `ctx.v_lat` are now `(n_envs, n_wheels)` tensors. Hooks operate on the whole batch at once.
- `PipelineContext.wheel_params` (list of WheelConfig) → `PipelineContext.wheel_meta` (the cached `WheelMeta` with batched coefficient tensors). Read per-wheel coefficients via `ctx.wheel_meta.mu_long`, `ctx.wheel_meta.pb_x`, etc.
- All three bundled hooks (`RollingResistance`, `LowSpeedRegularizer`, `StaticFrictionLock`) updated. Custom hooks need to be ported.

### BREAKING — TireModel API

- `TireModel.__call__(v_long, v_lat, v_roll, N, wheel_params)` now expects `(n_envs, n_wheels)` tensors and a `WheelMeta` (not a single `WheelConfig`). Both `PacejkaAnisotropic` and `CoulombIsotropic` read per-wheel coefficients from the meta. Custom tire models need to be ported.

### Added — `WheelMeta` batched coefficient tensors

`WheelMeta` now exposes per-wheel `(n_wheels,)` tensors for: `mu_long`, `mu_lat`, `rolling_resistance_cr`, `pb_x`, `pc_x`, `pe_x`, `pb_y`, `pc_y`, `pe_y`. Built once at `VehiclePhysics.__init__` from the resolved `WheelConfig` list. Cheap to broadcast against `(n_envs, n_wheels)` state.

### Removed — `dynamics.suspension_normal_force` (still importable)

The pure-Python helper is still exported and unit-tested, but `core.py` no longer calls it — the suspension N calculation is inlined as a batched op in `step()` for one less function-call layer. Helper remains for external use.

### Other

- `StaticFrictionLock.apply_post_tire` no longer does the `if not active.any(): return` short-circuit (forced a CPU↔GPU sync per step). The branchless batched path is always taken.

### Migration

External hooks / tire models with the v0.4.x per-wheel API will break — update to the batched signature. No change needed for code that only uses the public `VehiclePhysics` / `VehicleInputs` / presets / strategy APIs.

---

## [0.4.4] — 2026-05-18

### Performance
- **Tank chase-cam fps fix** — interactive demo on the KDU tank ran at ~15 fps vs ~25 fps for the original `KDU/example_interactive.py`. Two contributions:
  - `VisualSync` was syncing wheel spin angle (`set_dofs_position` for the 10 continuous spin joints) every step. The original KDU intentionally skipped this since the cylinder primitive wheels are rotationally symmetric and a spinning visual is invisible.
  - The chassis `up_world` reference tensor was being re-allocated every step inside `VehiclePhysics.step()` instead of being cached at init time.

### Added
- `VehicleConfig.visual_spin_enabled: bool = True` (also on `ResolvedConfig`). Set `False` to skip the per-step spin-angle `set_dofs_position` call. Saves ~3-5 ms / step in interactive mode (one fewer Genesis call). Cars keep the default `True` so mesh wheels visibly roll; tanks (`tank_10w_skid_belt` preset) now default to `False`.

### Changed
- `VehiclePhysics.__init__` caches `_up_world` once instead of re-creating it each `step()`.
- `VisualSync` integrates the visual spin angle in place (`add_`) when enabled.

---

## [0.4.3] — 2026-05-18

### Fixed
- **Heavy-wheel suspension visual fix (two-part)** — on the KDU 10-wheel tank (each wheel ≈ 500 kg):
  - *Part 1 (free-fall)*: the wheel meshes were sinking below the ground while the chassis stayed up. `VisualSync` chose `set_dofs_position` vs `control_dofs_position` by detecting `<dynamics>` declarations in the URDF, which KDU's URDF omits entirely. With no PD control (`kp`/`kv` = 0), Genesis lets the heavy wheel free-fall between substeps and the kinematic `set_dofs_position` cannot snap it back fast enough.
  - *Part 2 (spinning in air)*: after the PD fix above, wheels stopped falling but were spinning in air ~5 cm above ground. The `control_dofs_position` path used `target = max(0, rest_d - d)` (a non-negative compression), which can only push the wheel UP from rest, never DOWN to reach the ground. This is fine when the carrier rest position sits *above* ground level (HJW URDF: susp origin z = 0.34), but breaks when the carrier rest sits *at* base_link z (KDU URDF: susp origin z = 0). The `control_dofs_position` path now uses the same `joint_pos = mesh_radius - d` formula as the `set_dofs_position` path so the wheel mesh lands on the ground in both cases.

### Added
- `VehicleConfig.visual_susp_mode: str = "auto"` (also on `ResolvedConfig`). Three values:
  - `"auto"` (default) — per-joint decision based on URDF `<dynamics>` presence; preserves the v0.4.2 behavior for HJW-style URDFs.
  - `"kinematic"` — force `set_dofs_position` everywhere (light wheels, HJW-style).
  - `"control"` — force `control_dofs_position` with PD kp=1e7 / kv=1e5 everywhere (heavy wheels, KDU-style).
  - `resolve()` validates the value and raises `ConfigError` on unknown strings.
- `tank_10w_skid_belt` preset now sets `visual_susp_mode="control"` so the wheels stay attached on the KDU tank without the user having to know.

### Migration
- HJW / JMK / car presets unchanged — `"auto"` keeps their existing behavior.
- If you constructed a `VehicleConfig` manually for a heavy-wheel vehicle and saw the wheel-sinking bug, set `visual_susp_mode="control"` explicitly.

---

## [0.4.2] — 2026-05-18

### Added
- `genesis_vehicle.scene_helpers` module with two thin Genesis-side
  convenience helpers:
  - `make_wheel_raycaster(scene, entity, urdf_path, ...)` — replaces the
    five-line `parse_urdf` + `WheelRayPattern` + `gs.sensors.Raycaster` +
    `scene.add_sensor` boilerplate with one call.
  - `add_vehicle(scene, urdf_path, preset_fn=None, ...)` — adds the URDF
    entity + raycaster (and optionally builds the preset `VehicleConfig`)
    in one call. Returns `(car, sensor, cfg)`. Caller still owns
    `scene.build()` and `VehiclePhysics(...)` construction so other
    entities / sensors can be added between.
- Both are lazy-imported through the top-level package, so
  `from genesis_vehicle import add_vehicle` works without Genesis
  installed at import time.

### Design note
The SDK intentionally does NOT fully encapsulate Genesis. Helpers cut the
canonical boilerplate for the common case; full-encapsulation facades for
CARLA-style or Gym-style use will live under `genesis_vehicle.adapters/`
when needed. This keeps the core API research-friendly (viewer, terrain,
custom materials, multi-agent scenes all stay accessible) while still
offering a one-liner for the common path.

---

## [0.4.1] — 2026-05-17

### Added
- `car_4w_fwd_ackermann` preset — 4-wheel front-wheel-drive car with front Ackermann steering. Fills the obvious gap between the existing `car_4w_rwd_ackermann` and `car_4w_awd_ackermann` presets (real-world passenger cars are mostly FWD). Mirrors the RWD preset's parameters with `FWD(driven_axles=(0,))`.

---

## [0.4.0] — 2026-05-17

### Changed (BREAKING)
- **Stability profile renamed**: `"physical"` → `"raw"`. No deprecated alias kept.
  - **Reason for break.** The original `"physical"` name implied "the physically correct config" or "the right profile for parameter fitting", which is incorrect. Fitted parameters in this SDK are always model-coupled (Pacejka `B` is the coefficient that fits real inside our SDK; it is not "rubber's true stiffness"). Hooks are part of the model. The right Real2Sim rule is *"fit with the same profile you'll deploy with"* — for our SDK that is almost always `"control"`. The renamed `"raw"` profile is now correctly framed: hook-free raw dynamics for **debugging / ablation**, not a general fitting recommendation.
  - **No alias kept.** Pre-1.0 versioning + essentially no external users; keeping the misleading name even as a deprecated alias would perpetuate the confusion the rename is meant to resolve.
- API.md S6 rewritten around "one rule: fitting forward model = deployment forward model", with `"raw"` reframed as an analysis tool rather than a Real2Sim recommendation.

### Migration
- Replace every `stability="physical"` with `stability="raw"` (or, more likely, with the default `"control"`, since most users were on `"physical"` because the doc misled them).
- If you were on `"physical"` for Real2Sim fitting and plan to deploy with `"control"`, **change fitting to `"control"`** so the forward models match.

---

## [0.3.0] — 2026-05-17

### Added
- **Stability profile** on every preset: `stability="control" | "physical" | "research"` (default `"control"`).
  - `"control"` — RL/MPPI-friendly hooks: `RollingResistance` + `LowSpeedRegularizer(disable_when_control_active=False)` (+ `StaticFrictionLock` for tank). Default for `car_4w_rwd_ackermann`, `car_4w_awd_ackermann`, `truck_6w_partial_ackermann`, `tank_10w_skid_belt`.
  - `"physical"` — empty hook list. Use for Real2Sim parameter identification so numerical stabilizers don't leak into fitted parameters.
  - `"research"` — empty hook list. Caller assembles hooks manually.
- `stability_hooks_for_profile(profile, vehicle_kind)` public helper to materialize the same hook lists used by the presets.
- Version API: `__version__`, `VERSION_INFO`, `version()`, `version_info()` exported from `genesis_vehicle`.
- One-time version banner printed on first `VehiclePhysics` initialization:
  ```
  [genesis_vehicle v0.3.0] Initialized: 4 wheels, Ackermann, RWD, Independent, n_envs=1, hooks=[RollingResistance, LowSpeedRegularizer]
  ```
- `CHANGELOG.md` (this file) tracking version history.

### Changed
- **Footgun fix** — `"control"` profile builds `LowSpeedRegularizer` with `disable_when_control_active=False`. The legacy default (`True`) made the regularizer a no-op exactly when MPPI was applying throttle, so the documented "low-speed stabilization" never fired under control. Direct construction (`LowSpeedRegularizer()`) keeps the legacy default for backwards compatibility.

### Notes
- The hook system is now treated as a stability profile rather than a free-form list assembled by the user. This aligns with industry vehicle SDKs (PhysX sticky tire mode, Jolt anti-rollback, Chaos low-speed threshold, Unity WheelCollider stick-slip) where low-speed numerical stabilization is part of the model, not an opt-in plug-in. Advanced users can still pass `stability="research"` and assemble `cfg.stability_hooks` manually.

---

## [0.2.0]

Reviewer-driven revisions of the v0.1.0 reference implementation. The SDK's
public contracts and helper boundaries were tightened ahead of the first
variant migration.

### Added
- `genesis_vehicle/dynamics.py` — pure-Python primitives:
  - `brake_torque_signed(t_brake, omega, smoothing_scale=0.5)` — positive brake magnitude → signed torque opposing wheel rotation.
  - `suspension_normal_force(compression, comp_rate, k_susp, c_compression, c_extension, air_mask)` — asymmetric damper + non-negative clamp + air-mask zero in one call.
- `VehicleStepInputs` typing alias on `VehiclePhysics.step()` (Union of `VehicleInputs` + every typed input).
- `estimate_spin_inertia_from_genesis(entity, link_name, spin_axis_local=None)` with optional axis projection (`a^T diag(I) a`).
- New tests:
  - `tests/test_dynamics.py` (8 tests) for `brake_torque_signed` and `suspension_normal_force`.
  - `tests/test_config_resolve.py::test_user_explicit_i_wheel_wins_over_urdf`, `test_user_explicit_radius_wins_over_urdf`.

### Changed
- **`i_wheel` truth policy** — `WheelConfig.i_wheel` set by the user (or populated by URDF) is now authoritative; the Genesis-runtime spin-inertia refinement only fires for wheels where neither the user nor the URDF supplied a value. Previously the Genesis estimate always overrode.
- **Inertia helper renamed**: `parse_inertia_max_principal_genesis` → `estimate_spin_inertia_from_genesis`. The old name is kept as a deprecated alias and will be removed in a later release.
- `core.py` refactored to call the new `dynamics.py` helpers instead of inlining the brake/suspension formulas.
- `__init__.py` exposes the full public surface (strategies, tire models, presets) as a single import path: `from genesis_vehicle import ...`.

### Documentation
- API.md restructured into 11 sections; new "Physics contracts" section codifies brake-sign, non-negative N, `i_wheel` policy, ISO 8855 steer convention, and coupling order. Migration cheatsheet now uses the "URDF default, API override wins" wording instead of the misleading "URDF as truth".

---

## [0.1.0]

Initial Phase 3 reference implementation of the genesis_vehicle SDK.

### Added
- Top-level `VehiclePhysics` driver running the shared 5-step ray-wheel pipeline batched over `n_envs ≥ 1`.
- Strategy abstractions: `SteeringStrategy` (`Ackermann`, `PartialAckermann`, `SkidSteer`, `NoSteer`), `DrivetrainStrategy` (`FWD`, `RWD`, `AWD`, `PerSide`), `CouplingStrategy` (`Independent`, `SameSideBelt`), `StabilityHook` (`RollingResistance`, `LowSpeedRegularizer`, `StaticFrictionLock`).
- Tire models: `PacejkaAnisotropic`, `CoulombIsotropic`.
- Config & merge: `WheelConfig`, `ChassisConfig`, `VehicleConfig`, `ResolvedConfig`, `resolve()`, `ConfigError`. URDF defaults + per-wheel overrides.
- Inputs: unified `VehicleInputs` + typed (`AckermannInputs`, `SkidSteerInputs`, etc.) with `from_unified()` adapters.
- `parse_urdf()` convention-based parser supporting HJW (`*_suspension_joint`) and KDU (`*_susp`) naming.
- Visual layer (`VisualSync`) absorbing URDF axis-sign quirks.
- Presets for four reference vehicles: `car_4w_rwd_ackermann`, `car_4w_awd_ackermann`, `truck_6w_partial_ackermann`, `tank_10w_skid_belt`.
- Pure-Python tests for URDF parsing, config resolve, strategy math.
