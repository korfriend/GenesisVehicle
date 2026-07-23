# Changelog — genesis_vehicle

All notable changes are documented here. Versions follow
[Semantic Versioning](https://semver.org/) (major.minor.patch).

`__version__` lives in `genesis_vehicle/_version.py` and is re-exported as
`genesis_vehicle.__version__` plus the `version()` / `version_info()`
functions. `VehiclePhysics.__init__` prints a one-line banner with the
running version the first time it is instantiated in a process.

---

## [1.2.2] — 2026-07-23

Parameterise sweep-table measurement from the CLI, and a note on tank top speed.

| abbr | meaning |
|---|---|
| omega_max_drive | drive-side wheel angular-velocity cap (rad/s); with negligible drag it is the de-facto top-speed governor |
| CLI | command-line interface |
| MBT | main battle tank |

### Added — `sweep_measure` plant overrides on the command line

`python -m genesis_vehicle.control.sweep_measure` gains a **plant overrides**
group so a user can measure a table end-to-end without writing an
`apply_config` Python file: `--top-speed`, `--omega-max-drive`, `--i-wheel`,
`--mu-long`, `--mu-lat`, `--k-susp`, `--rest-stroke`, `--brake-max`. Only the
flags passed are applied — everything else keeps the preset value, so a tracked
preset's suspension stays mass-derived (v1.2.1) unless `--k-susp` is given. CLI
flags win over `--config`.

`--top-speed` is radius-independent: it sets `omega_max_drive = top_speed /
mean_wheel_radius`, so you ask for a speed in m/s rather than a wheel rad/s that
depends on the URDF's wheel size. `apply_plant_overrides()` is also importable.

### Note — the tank preset's `omega_max_drive` default is unrealistically high

`tank_skid_belt` sets `omega_max_drive = 100` rad/s. This model has almost no
aero/rolling drag, so the wheels simply spin up to that cap: at a 0.33 m wheel
that is ~32 m/s (**115 km/h**), versus an M1A2's ~67 km/h road / ~48 km/h
off-road, or 40–72 km/h for MBTs generally. Every bundled demo masked this by
overriding the cap in its tuning file, so a user driving the **raw** preset gets
a runaway-speed tank. This is a bad default rather than a crash, so it is left
as-is for now (changing it is a behaviour change for anyone relying on it); the
new `--top-speed` flag is the intended way to pin a realistic speed
(`--top-speed 18.6` ≈ 67 km/h). Empirically the cap — not wheel mass — governs
top speed, and at any capped speed the plant is hunt-free regardless of
`i_wheel`; the earlier belief that a heavier wheel was needed for stability was
wrong (it only changes spin-up feel).

## [1.2.1] — 2026-07-23

Suspension sizing. A tracked vehicle reported as "far too soft vs the client
engine, wallows badly" turned out to be running on the car-sized module default
spring, because the tracked preset's tuning constants were keyed by another
vehicle's wheel-link names and matched nothing. Three fixes plus two guards so
the same class of failure reports itself instead of feeling like bad tuning.

| abbr | meaning |
|---|---|
| sprung mass | the mass the springs carry: everything not below a suspension joint |
| unsprung mass | carrier + wheel + anything hanging off them |
| sag | static suspension deflection under the vehicle's own weight |
| f_n | natural heave (vertical bounce) frequency |
| zeta | damping ratio, `c / c_critical` |
| pk-pk | peak-to-peak amplitude |

### Fixed — `tank_skid_belt` sizes its suspension from the URDF's mass

`_tank_wheel_overrides()` returned a dict keyed by the reference vehicle's ten
wheel-link names, and `VehicleConfig.from_urdf` applies overrides by exact name
match. Any other tracked URDF therefore matched **zero** keys and silently kept
module defaults — `k_susp = 70,000 N/m`, sized for a 2 t car. Measured on a
38.5 t / 14-wheel hull: 386 mm of static sag against 50 mm of travel (7.7x),
f_n 0.8 Hz, rebound zeta 0.14; the hull settled 330 mm below its design ride
height and heaved 272 mm pk-pk with 5.0 deg of pitch under acceleration.

The preset now derives `k_susp` / `c_compression` / `c_extension` from the
URDF's own sprung mass and keys the overrides by the URDF's own wheel names, so
it holds its ride frequency across tracked vehicles of any mass and wheel count.
Same hull after the fix: 50 mm sag, f_n 2.23 Hz, 4.9 mm pk-pk heave, 1.0 deg
pitch.

The derivation reproduces the literals it replaces to within 4% on the vehicle
they came from (k 1.00e6 -> 1.044e6, c 120,000 -> 119,229), so that vehicle's
behaviour and its measured sweep tables are unchanged. The hard-coded
`radius=0.4` override is gone as well — the URDF's own wheel geometry is
authoritative, and the literal silently rescaled every other tracked vehicle.

`target_sag` is now a `tank_skid_belt` keyword (default 0.05 m) for callers who
want a softer or tighter vehicle.

### Added — `suspension_from_mass()`

Public helper (`from genesis_vehicle import suspension_from_mass`) returning
`(k_susp, c_compression, c_extension)` for one wheel from a sprung mass, a wheel
count and a target sag:

    k      = (m_sprung * g / n) / target_sag
    c_crit = 2 * sqrt(k * m_sprung / n)

A fixed spring rate cannot serve both a 2 t car and a 40 t tank; deriving from
mass keeps sag — and hence ride frequency — constant across vehicle scales.

### Added — `parse_urdf()` reports sprung / unsprung / total mass

`URDFParsedConfig` gains `total_mass`, `sprung_mass` and `unsprung_mass`.
Unsprung is every link at or below a suspension joint's child; sprung is the
rest. `chassis_mass` is the **base link alone** and omits sprung children such
as a turret or a cargo body — on the reported vehicle it read 27,134 kg against
a true sprung mass of 38,517 kg, undersizing any spring derived from it by 42%.

The OSC server's generic (mapping-path) suspension autotune now sizes against
`sprung_mass` for the same reason. Simple car URDFs, where the two are equal,
are unaffected.

### Fixed — `parse_urdf()` reads geometry from a spin-jointless wheel

A URDF that models the wheel as a single link hanging straight off the
suspension joint (no separate `continuous` spin joint) failed the chain walk, so
`radius` / `mass` / `i_wheel` came back `None` and fell through to module
defaults — the wheel silently became 0.35 m and 20 kg no matter what the URDF
declared. `parse_urdf` now falls back to the suspension joint's own child link,
which in that topology **is** the wheel. This is what the tracked preset's
`radius=0.4` literal had been papering over: with the parser fixed, the bundled
reference vehicle keeps its 0.4 m wheels from its own geometry, and picks up its
declared 500 kg / I = 29.5 kg·m² instead of the 20 kg / 1.5 kg·m² defaults.
`WheelConfig.mass` is informational (the pipeline does not read it); `i_wheel`
is not, so a tracked vehicle relying on the old default will spin its wheels up
more slowly now — which is the physically correct behaviour for a 500 kg wheel.

The bundled `samples/data/tank_sweep_signed.csv` was re-measured against the
resulting plant.

### Fixed — a backwards cusp hop no longer spins the path follower

`PathFollower` derives a direction block's arrival heading from its last
segment. The boundary waypoint belongs to both blocks, and nothing forces it to
lie ahead of the previous one: a path that doubles back can put the reverse
leg's first waypoint *behind* the forward leg's last. One real path hops 0.28 m
backwards there, which made the forward block's arrival heading **-94.9 deg** —
a 180 deg spin demanded at the end of a straight leg. The vehicle obeyed, left
the path, and then the projection could no longer advance, so the cusp never
fired and it drove forward forever. Only exactly-duplicated waypoints were
guarded before.

The block-end heading now skips trailing segments that run counter to the
block's net travel (`_forward_end_yaw`). An explicit per-waypoint yaw still
wins, and a genuine corner is unaffected — only a reversing tail is skipped.

This was latent: the same path used to survive because the vehicle spun slowly
enough to reach the cusp trigger first. It surfaced when the tracked preset
started applying its tire model (below), which raised skid-steer yaw authority
~21% and let the spin win the race.

### Changed — the tracked preset's tire model now reaches every tracked vehicle

A consequence of the wheel-name fix above worth calling out on its own: the
tracked preset's Pacejka shape and rolling resistance (`pb_x` 5.0, `pb_y` 4.0,
`rolling_resistance_cr` 0.05, `mu` 0.9 / 0.63) previously applied only to URDFs
whose wheel links matched the reference names. Every other tracked vehicle
silently ran on the **car** defaults (`pb_x` 10.0, `pb_y` 8.0, `cr` 0.015).

They now apply to all of them, which is the intent of a tracked-vehicle preset —
a car's tire curve on a 40 t tracked hull was never right. Handling does change:
measured on a 38.5 t / 14-wheel vehicle, pivot yaw rate +21% (0.775 -> 0.939
rad/s) and stopping distance -16% (0.62 -> 0.52 m). Unlike the suspension this
is not a mass-scaled quantity, so if you had tuned around the car defaults,
re-check that vehicle — and re-measure any sweep table taken against it.

### Added — one suspension priority chain, honoured everywhere

    caller / OSC override  >  URDF <dynamics stiffness=...>  >  mass-derived

Standard URDF has **no spring-stiffness field** — `<joint><dynamics>` carries
only `damping` and `friction`, and on a prismatic joint those describe the
articulated solver, not a suspension. A `stiffness` / `spring_stiffness`
attribute is a non-standard extension, so a **non-zero** value there is the only
unambiguous "the author meant a suspension spring" marker. `parse_urdf()` now
reads it (plus `damping` / `compression_damping` / `extension_damping` from the
same tag) into the parsed `WheelConfig`, and `resolve()`'s existing
URDF-then-default merge turns that into the chain above for free. The tracked
preset yields per field: anything the URDF declared is left alone, anything it
did not still gets the derived value.

The rule matters because reading `damping` unconditionally is wrong: the bundled
car URDF writes `damping="20.0"` on its **steer** joints and `stiffness="0.0"`
on its suspension joints — "no spring here". The OSC server's mapping-path
branch did read both at face value, so that URDF silently resolved to
`k_susp = 0` — no suspension at all. Both paths now share `_susp_dynamics()` and
the same rule; neither reference URDF declares a spring, so both stay on their
derived values.

### Changed — `sweep_measure` default substeps 10 -> 4

The measurement tool defaulted to `--substeps 10` while both runtimes that
consume its tables default to 4 (`VehicleScene(substeps=4)`,
`genesis_vehicle.server --substeps 4`). dt/substeps are part of a sweep table's
validity contract, so the out-of-the-box table was measured on a
finer-integrated plant than the one it would be driven on. All three now agree
at dt 0.025 x substeps 4 (internal 6.25 ms). Pass `--substeps 10` explicitly to
reproduce an older table.

### Added — two silent-failure guards

- `VehicleConfig.from_urdf` warns when a `wheel_overrides` key matches no wheel,
  naming the dropped keys and listing the URDF's actual wheel names. This is the
  guard that would have caught the bug above on the first run; it is the same
  class of check the server's `wheelOverrides` handler gained in v1.1.26.
- `resolve()` warns when the resolved spring cannot hold the vehicle up — static
  sag past 1.25x `rest_stroke`. The threshold is not 1.0x because `rest_stroke`
  is a ray budget rather than a mechanical bump stop and the tracked preset sits
  at ~1.0 by design.

Both are `logging.WARNING` on the `genesis_vehicle` logger, consistent with the
v1.1.24 severity split.

---

## [1.2.0] — 2026-07-17

Minor-version milestone (released as 1.1.27 for a few minutes, retitled):
it consolidates the 1.1.26 BREAKING preset rename
(`tank_10w_skid_belt` → `tank_skid_belt`) and the official-surface cleanup
below — together they change what the public package looks like, which is
minor-version territory, not a patch.

### Changed — internal codenames removed from public surfaces

Author/sandbox codenames (the pre-SDK variant initials) and external vehicle
model names no longer appear in the SDK's docs, sample docstrings, preset
docstrings or code comments — they carried no information a reader of the
public repo could use ("reference car" / "reference tank" / "a 14-wheel
tracked vehicle" say the same thing without the internal context). Purely
internal helper names in `presets.py` were renamed to match
(`_car_brake_bias`, `_car_wheel_overrides`, `_tank_wheel_overrides`). The
retired `tank_10w_skid_belt` name is likewise no longer mentioned outside
this changelog's history.

One behavioral change: `server/benchmark.py` and `server/benchmark_collision.py`
now default their tank URDF to the SDK's **bundled** `samples/urdf/tank_ray.urdf`
instead of a sibling repo checkout — the benchmarks are self-contained now
(`--urdf` still overrides). Two stray temp URDFs left in `samples/urdf/` by a
killed process were removed.

`tests/` is now SELF-CONTAINED: the two reference URDFs the parsing/config
tests exercise are vendored as `tests/data/car_ref.urdf` and
`tests/data/tank_ref.urdf` (they were read from sibling sandbox checkouts by
path before — a clone of this repo alone could not run them), and test/helper
names were neutralised to match. This changelog's history entries were also
reworded to the same neutral vocabulary.

## [1.1.26] — 2026-07-16

### Fixed — OSC server: skid-steer vehicles with ≠10 wheels got a DEAD drivetrain

| abbr | meaning |
|---|---|
| OSC | Open Sound Control (the UE/Unity wire protocol) |
| UE  | Unreal Engine (the external client) |

`driveType 2` (skid steer) only loaded the tank preset when the URDF had
EXACTLY 10 wheels. Every other tracked vehicle — e.g. a 14-wheel tank —
fell into the generic mapping branch, where an empty `drivingJoints` list
produced all-zero drive weights and `NoSteer`: zero drive torque, steering
ignored. The vehicle only crept at cm/s from residual motion. Field impact
(team report 2026-07-16): a UE waypoint test where the tank "moves at
5 cm/sec and won't follow waypoints" — and a sweep table (CSV) measured
against that dead plant came out as pure noise (`omega_z ≈ -0.5 rad/s at
every steer value, including 0`), which then made the client-side follower
command nonsense steering.

- The skid-steer branch now loads the tank preset for ANY wheel count
  (`SkidSteer`/`PerSide`/`SameSideBelt` are wheel-count-generic; the preset
  discovers wheels from the URDF).
- The generic mapping branch no longer accepts an all-zero drive-weight
  vector silently: it falls back to all-wheel drive and prints a warning
  naming the cause (no `drivingJoints` matched any wheel spin joint).
- `wheelOverrides` matching now uses the WHOLE URDF joint chain the parse
  already knows — wheel link, spin, SUSPENSION and steer joint names (it
  only checked spin + wheel link before, so `"wheelName": "susp"` matched
  the reference tank, whose spin joints carry the word, but nothing in an
  vehicle where only the suspension joints do — and the override was skipped
  SILENTLY, so its mass-derived suspension never applied and the 27 t hull
  bounced on the default spring). `"*"` / `"all"` is a new wildcard for
  every wheel, and an entry that still matches nothing prints a warning
  naming the URDF's actual wheel links.
- Verified end-to-end on the reported scenario (14-wheel tank, the team's
  9 waypoints incl. a doubling-back reverse leg, start heading +y):
  PASS twice, final error 1.48/1.50 m in ~50 s — with a valid sweep table.
  Sweep tables measured against a pre-1.1.26 server plant must be
  REGENERATED; they encode the dead drivetrain, not the vehicle.

### Fixed — sweep_measure: raw-scene build let the wheel rays hit the vehicle itself

`genesis_vehicle.control.sweep_measure` built a raw `gs.Scene` + the
low-level `add_vehicle` (single_scene raycast, no URDF prep). For an
arbitrary URDF that combination measures a BALLISTIC vehicle, not a driving
one: (a) un-prepped wheel colliders fight the suspension as a double
support, and (b) the v1.1.16 high-cast ray (start = attach + 1 m) begins
inside a tall hull and hits the vehicle's OWN roof — a self-hit that rides
along with the vehicle. Measured on a 27 t tracked vehicle: distance frozen at
-0.405 m on all 14 wheels, N = 317 kN each, the hull at z = +16 m climbing
at 56 m/s during the "ground settle" — and the sweep CSV came out as pure
noise (e.g. `a = +18 m/s^2`, `omega_z = -0.57 rad/s at steer 0`).

The tool now builds a `VehicleScene` (dual_scene raycast, `per_vehicle`
solver): the URDF is auto-prepped and the rays only see the raycast scene's
static mirrors, so self-hits are impossible. Methodology (settle → v_init
teleport → 0.3 s hold → 2 s window; per-env slope gravity; chunk reuse via
scene reset) is unchanged. Verified: the same combo that measured
`a = -0.92` / ballistic garbage now measures `a = +1.01, omega_z = 0.0001`
at (v=0, thr=+1, steer=0), matching an independent in-process probe.

### Changed — BREAKING: preset `tank_10w_skid_belt` renamed to `tank_skid_belt`

The preset was always wheel-count-generic (`from_urdf` discovers the wheels;
`SkidSteer`/`PerSide`/`SameSideBelt` scale to any count) — and now that the
server applies it to any skid-steer vehicle, the "10w" in the name actively
misled (it is validated on 10- and 14-wheel tracked vehicles).
The old name is REMOVED, not aliased: update imports to
`from genesis_vehicle import tank_skid_belt`. All docs, samples, tests and
the server updated.

## [1.1.25] — 2026-07-15

### Fixed — native-viewer vehicle "tremble": draw-thread races (two of them)

The interactive viewer draws on its own thread under the render lock, so
anything committed in a DIFFERENT lock hold than the rigid node poses can be
drawn one physics step apart from them — a one-step offset (~15 cm at 7 m/s)
that flickers at the draw rate. Physics is clean; cv2/offscreen cameras never
show it (they render synchronously after the full step). Reported on
`terrain_drive`. Two such races existed:

1. **Wheel buffers vs body.** `VehicleScene.step()` streamed the instanced
   wheel buffers AFTER `scene.step()`, whose internal visualizer update had
   already signalled the draw — a frame could show the chassis at step N with
   wheels at N-1 (the whole vehicle appears to tremble).
2. **Follow camera vs body.** Stock Genesis `Viewer.update()` sets the
   `follow_entity` camera OUTSIDE the render lock and the node poses INSIDE
   it — a frame could pair the fresh camera with last step's body pose. With
   race 1 fixed this became visible in isolation: wheels and terrain steady,
   ONLY the body trembling fore/aft.

Fix: `visual.patch_viewer_atomic_update(viewer)` (applied automatically at
`build()` when the native viewer is on) rebinds `viewer.update` so the wheel
buffers (`viewer._gv_pre_draw`), the follow camera and the node poses are all
committed in ONE render-lock hold; real-time pacing stays outside the lock.
`VehicleScene.step()` runs `scene.step(update_visualizer=False)` and updates
the visualizer after the wheel stream when the patch is not active (offscreen
cameras, or unrecognized future viewer internals — the patch verifies the
expected layout and declines gracefully with a warning). Headless scenes keep
the old single-call path. Physics is untouched in every mode.

### Added — OSC server `--substeps` and a `--follow-cam` viewer follow

| abbr | meaning |
|---|---|
| OSC | Open Sound Control (the UE/Unity wire protocol) |
| UE  | Unreal Engine (the external client) |

- `--substeps N` sets the server's internal substep count, which was hard-coded
  to 2. The internal physics step is `dt / substeps`; a stiff suspension spring
  needs a small internal step or it rings. Lower it (e.g. 1) to reproduce
  coarse-step instability, raise it for stiffer models. Honoured in both
  per-entity (L2) and multi-env (L3) modes. **The default is now 4, up from the
  old hard-coded 2** — the extra integration is stable for stiffer models and
  measured negligible at n_envs=1 (step time is dominated by raycast + per-step
  overhead, not the internal solver). Large fleets (tens of contact-heavy
  vehicles) should measure, since substep cost can scale with contacts there.
- `--follow-cam {side,chase}` (with `--follow-target ID`) makes the viewer
  camera track a vehicle instead of the one-shot bird's-eye framing. `side`
  views it from -Y (good for watching the wheels); `chase` from behind (-X);
  world-fixed offset, smoothed. Implemented with the viewer's own
  `follow_entity`, NOT a per-loop `set_camera_pose`: the viewer draws on its
  own thread, so a camera set from the serve loop lags the vehicle by up to
  one physics step within each drawn frame, and that offset flickers at the
  draw rate — the vehicle "trembles" on screen even when the physics is
  steady (the `terrain_drive` sample hit and documented exactly this;
  `follow_entity` updates the camera inside `viewer.update()`, atomically
  with the pose push). A per-loop smoothed camera remains only as a fallback
  for Genesis versions without `follow_entity`. Ignored when headless.

`--follow-cam` defaults to `none` (prior behaviour). Motivated by
`Team_Issue_Test/wheel_vibration_test.py`, an issue repro that drives one URDF
through both the in-process SDK and the server and needs the two to run at the
same substep count with a matching, jitter-free side view.

## [1.1.24] — 2026-07-15

### Fixed — the OSC server bypassed URDF prep (vehicles floated in UE)

| abbr | meaning |
|---|---|
| URDF | Unified Robot Description Format (the vehicle model file) |
| OSC  | Open Sound Control (the UE/Unity wire protocol) |
| UE   | Unreal Engine (the external client) |

1.1.22/1.1.23 made `VehicleScene.add_vehicle` prepare the URDF, but the
server built its own `gs.morphs.URDF` and therefore hit the
`if morph is None` guard — it passed the ORIGINAL, unprepared path to
`add_vehicle`, whose `parse_urdf` is what places the wheel rays. A URDF that
hangs its wheels off a carrier link (`body --susp--> carrier --spin(z=+h)-->
wheel`) then cast its rays `h` below the wheel centres, and the hull settled
`h` too high — the vehicle floated 0.433 m in UE even though the same model
was correct in every in-process sample.

- `server/vehicle_builder.py` and `server/l3_runtime.py` now call
  `prepare_vehicle_urdf()` ONCE and feed that single prepared path to the
  morph, `build_cfg()` and `add_vehicle()`, so the entity, the config and
  the ray pattern all come from the same file. (This replaces the older
  `strip_wheel_collisions()` helper, which only handled contract 1.)
- `VehicleScene.add_vehicle` warns when it is given `morph=...` together
  with an unprepared `urdf_path` — the mismatch that caused this bug can no
  longer happen silently.
- Verified through the server path: settled chassis z = −0.008 m (was
  +0.400 m); path-following OSC sample passes with 1.48 m final error.

### Changed — missing `<inertial>` is now a warning, not a log line

`prepare_vehicle_urdf` corrects three URDF issues, but they are not equally
severe, and lumping them into one informational line hid the one that
matters:

- Wheel colliders removed, and suspension origins folded onto the wheel
  centre, stay **informational**. Neither is a URDF defect — the file is
  legal for a normal rigid-body simulator (a prismatic joint's origin along
  its own axis is a gauge freedom); it simply does not match what the
  ray-wheel model reads out of it.
- A link with no `<inertial>` is a **real defect in any engine** (a
  zero-mass moving link makes the articulated chain degenerate), and the fix
  injects a placeholder mass the author never chose. It now raises a
  `logging.WARNING` naming the offending links and telling the author to fix
  the URDF at the source.

## [1.1.23] — 2026-07-11

### Changed — URDF prep is mandatory; samples render shadows by default

- `VehicleScene.add_vehicle` no longer takes `prepare_urdf`: the URDF is
  ALWAYS made ray-wheel ready (1.1.22's three corrections). It was never a
  meaningful knob — the prepared file feeds the entity, the `parse_urdf`
  call AND the ray pattern, and those must agree; opting out could only
  produce a vehicle whose rays disagree with its geometry. Per-correction
  flags still exist on `urdf_prep.prepare_vehicle_urdf` for tooling.
- Shadows are ON in every sample/demo viewer (Genesis's own
  `VisOptions(shadow=True)` default): `path_follow_demo`,
  `path_follow_reverse_demo` and the legacy tank interactive demo had explicitly
  disabled them.

---

## [1.1.22] — 2026-07-11

### Added — `urdf_prep`: arbitrary URDFs are made ray-wheel ready automatically

Driving an externally authored 14-wheel tank model surfaced three
URDF contracts the SDK had only ever met by construction. All three are now
auto-corrected by `VehicleScene.add_vehicle` (`prepare_urdf=True` by
default; the original file is never modified, and a compliant URDF — every
SDK vehicle — is used as-is with no temp copy):

- **Wheel colliders -> render-only.** Ground contact IS the raycast +
  suspension model; a colliding wheel is a second, fighting support. The
  vehicle sat on its wheel colliders while the suspension pushed with 4x its
  weight, jittering in place instead of driving. Colliders are stripped —
  and a collider that is the wheel's ONLY geometry is first promoted to a
  `<visual>`, so the wheel still renders (the instanced renderer draws
  visuals; physics never touches them).
- **Suspension attach point moved onto the wheel centre.** The model chains
  `body --susp(z=0)--> carrier --spin(z=+0.433)--> wheel`, so the ray
  origin sat 0.433 m BELOW the wheel: the hull settled that much too high
  and the wheels visibly floated. The spin-joint offset is folded into the
  suspension origin (link rest poses unchanged). NB `VehicleScene`
  re-parses the URDF for the ray pattern, so this must be fixed in the FILE
  — a config-level override does not reach the rays.
- **Missing inertials injected.** The model's 14 carrier links declare no
  `<inertial>` at all (zero mass and inertia) — Genesis falls back to its
  legacy URDF parser and the articulated chain goes degenerate: a 5 MN push
  moved the 40-t hull as if it weighed ~900 t.

Also: `InstancedWheelRenderer.harvest_wheel_meshes` now falls back
visual -> collision -> cylinder-from-`radius`, so a wheel with no visual (or
no geometry at all) still shows up. New `docs/physics-contracts.md` §7.9
documents all three contracts and what rendering does/does not depend on;
`api-reference.md` §4 gains the helper. 5 new pure-XML tests (compliant
URDFs untouched, offender corrected, collider-only wheel promoted, original
file preserved, opt-out flags) — 155 pass.

---

## [1.1.21] — 2026-07-11

### Docs — OSC velocity-estimation guidance added to the path-following guide

- New paragraph in the quick-start section of `docs/path-following.md`:
  driving the follower over the OSC server requires finite-differencing
  velocity in the server SIM time base (`/Genesis/State/SimTime`, ~0.3 s
  window) — wall-clock FD under-reads under server slow motion and blows
  the course (the measured 1.1.20 failure); sample #14 is the reference
  implementation. Docs-only; no code changes.

---

## [1.1.20] — 2026-07-10

### Fixed — native-viewer wheel streaming; server viewer framing; client path overlay

| abbr | meaning |
|---|---|
| node-swap | replacing the render mesh node per update (both render paths re-diff it) |
| in-place | writing primitive poses on the existing node (camera/jit path streams it) |

Field report on ``path_follow_osc_demo --viewer``: wheels frozen at spawn,
camera inside the vehicle, path invisible. All three fixed:

- **Wheels frozen in the NATIVE viewer (all rendered-vehicle scenes, not
  just the OSC demo)**: Genesis's classic viewer renderer uploads a
  primitive's instance-pose buffer only ONCE when the mesh enters the GL
  context — in-place ``primitive.poses`` writes are never re-uploaded
  there (the camera/jit path rebuilds per render, which is why mp4
  recordings streamed fine and masked this). Final fix:
  ``InstancedWheelRenderer.update`` now streams poses through the
  engine's OWN per-frame instance-buffer queue
  (``context.jit.update_buffer`` — the same mechanism Genesis uses for
  link-frame instance poses, flushed by the render thread on its next
  pass), identical for the native viewer and offscreen cameras: no node
  churn, plain opaque depth-tested nodes. Two interim approaches were
  tried and REJECTED during viewer testing: node-recreation per step
  (shadow-state churn → brightness flicker) and marker nodes (markers
  render with overlay semantics — no depth write — so wheels looked
  translucent with wheels-behind-wheels showing through; the user's
  "depth-test flicker" diagnosis). Verified in the viewer run end-to-end
  (PASS, and faster than the node-swap interim: 32.7 s vs 35.7 s wall);
  physics stays bit-identical. Samples using cv2-HUD ``--viewer``
  (camera-based) were never affected; every ``show_viewer``/native path
  was. Cost re-measured after buf-id caching: rendering-on adds
  +2.7 ms/step at 30 tanks over headless (18.1 -> 20.8; the retired
  joint-sync mechanism added +4.2) and ~0.1 ms at demo scale.
- **Server viewer camera spawned inside the vehicle**: both server modes
  now frame the fleet after build — bird's-eye at the spawn centroid,
  span-scaled (``viewer.set_camera_pose``); orbit freely afterwards.
- **Client path invisible in the server viewer**: new debug-overlay wire
  message ``/Genesis/Debug/Polyline`` —
  ``[r, g, b, a, radius, x0, y0, z0, x1, y1, z1, ...]`` in Genesis
  metres; the server draws consecutive-point segments once built (both
  modes; any number of messages). ``path_follow_osc_demo`` sends its
  planned path (forward cyan) right after init.
- ``VehicleScene.build`` now logs which wheel-visual path was chosen
  (``instanced renderer`` / ``internal_sync fallback``) — the silence
  made this diagnosis slower than it should have been.
- Viewer-feedback follow-ups (same day): node-swap is now
  DOUBLE-BUFFERED (add the new node under an alternating name, THEN
  remove the old one — a frame landing between the two mutations
  briefly renders both copies at identical poses instead of neither,
  killing the wheel flicker); new `/Genesis/Debug/Spheres` wire
  message (same format as Polyline, one sphere per point) so the
  trajectory client can show waypoint/goal markers; and the server
  viewer REFRAMES to cover the received overlays (the overlay usually
  IS the course — spawn-only framing sat too close), with a more
  top-down elevation. `path_follow_osc_demo` sends waypoint
  spheres (every 3rd, cyan) and a red goal marker after init.
- **Slow-motion control bug (exposed by the full viewer run)**: with
  the viewer on, the server can run slower than real time; the
  client's velocity estimate was WALL-clock-based, so it under-read
  speed, the KICK over-throttled, and the vehicle overshot the course
  in sim terms (FAIL err 157 m). Fix: new additive wire message
  `/Genesis/State/SimTime [t]` — both server modes stamp each
  TargetBulk with its SIMULATION time (interpolation-fractional:
  `(step_count - 1 + alpha) * dt`), and the demo client
  finite-differences in that time base (wall-clock fallback when the
  server predates the stamp). Control is now immune to slow motion
  and to interpolated/duplicate sends.
- Server viewer camera calls were failing SILENTLY (framing looked
  unchanged): ``viewer.set_camera_pose`` rejects tuples (ndarray
  required), and numpy 2.0 removed the ``ndarray.ptp()`` method — both
  the spawn framing and the overlay reframe hit these and were skipped.
  Fixed (ndarray args, ``np.ptp()``); the viewer now jumps to a
  course-wide top-down overview as soon as the path overlay arrives.
  Demo overlay colors made fully opaque (translucent overlays under the
  vehicle join the depth-sorted blend pass and can shimmer).
- Verified: OSC demo headless PASS (err 1.47 m) with polyline + both
  sphere batches acknowledged by the server; 151 tests pass.

---

## [1.1.19] — 2026-07-10

### Changed — renamed: `VisualJointSync` → `WheelJointInternalSync` (+ related parameters)

- The legacy joint-sync mechanism is renamed to say what it IS: an
  internal wheel-joint synchronizer that goes through the solver — not a
  pure "visual" layer (the misnomer is what made the drop-impact field
  report so confusing). Renames, applied repo-wide (code, docs, comments;
  historical CHANGELOG entries keep the old class name):
  - class `visual.VisualJointSync` → `visual.WheelJointInternalSync`
    (a "(formerly VisualJointSync)" note stays on the class docstring and
    the api-reference naming note for traceability);
  - config field `enable_visual_joint_sync` →
    `enable_wheel_joint_internal_sync` (`VehicleConfig` +
    `ResolvedConfig`; internal/legacy — auto-managed by `VehicleScene`).
    No compatibility alias: a repo-wide search found zero external users
    of the old field, and the class itself left the public API in 1.1.18;
  - `VehicleScene(wheel_render_mode=...)` value `"joint_sync"` →
    `"internal_sync"` (introduced unreleased in 1.1.17, renamed before
    first release; the 1.1.17/1.1.18 entries below were updated to the
    final string so readers can copy-paste).
- 151 tests pass; rendered-path smoke re-verified.
- Docs pass for `wheel_render_mode` across `api-reference.md` (ctor
  signature + parameter table), `pipeline-and-hooks.md` step [6],
  `samples/README.md`, `index.md`, and `physics-contracts.md` §7.8 —
  stating explicitly: whenever a viewer/camera is present wheel visuals
  are ALWAYS active (no off switch; headless pays nothing); the wheels
  are **NOT updated through the Genesis rigid solver** but drawn via
  Genesis's external render-node channel (the debug-draw machinery) with
  closed-form poses; the only cost is the pose streaming itself (a
  slight per-step overhead, ~2–3 ms at 30 vehicles on CPU) with physics
  bit-identical to headless.

---

## [1.1.18] — 2026-07-10

### Removed — `VisualJointSync` dropped from the public API

- `VisualJointSync` is no longer exported from `genesis_vehicle` (removed
  from the lazy-export table and `__all__`). Rationale: since v1.1.17 the
  instanced solver-free renderer is the default wheel-visual mechanism and
  is physics-identical to headless; the joint-sync path survives ONLY as
  an internal fallback (`genesis_vehicle.visual.VisualJointSync` — used
  automatically for `n_envs > 1` rendered scenes and the raw
  `VehiclePhysics` path, or forced via
  `VehicleScene(wheel_render_mode="internal_sync")`). Users never need to
  name the class.
- `VehicleConfig.enable_visual_joint_sync` stays (it wires the internal
  fallback) but is documented as internal/legacy — `VehicleScene` manages
  wheel visuals itself.
- Docs/comments sweep: `index.md` Built-in utilities row is now "Viewer
  wheel visuals" (entry point `VehicleScene(wheel_render_mode=...)`),
  `samples/README.md`'s wheel-animation section rewritten for the
  instanced renderer, `api-reference.md` naming note marks the class
  internal, sample comments no longer name it, `testing.md` lazy-name
  list updated, `tests/_check_import.py` updated. 151 tests pass.

---

## [1.1.17] — 2026-07-10

### Changed — wheel visuals: solver-free instanced rendering replaces VisualJointSync as the default

| abbr | meaning |
|---|---|
| VJS | VisualJointSync — the pre-1.1.17 wheel-visual mechanism (drives solver joints) |
| instanced | one pyrender mesh node per (kind, wheel index) carrying K instance poses, streamed per step |

- New `InstancedWheelRenderer` (visual.py): when a `VehicleScene` renders
  (viewer or camera) and `n_envs == 1`, wheel visuals are now drawn by
  streaming the closed-form `wheel_visual_transforms` poses into instanced
  pyrender nodes — the same data source external renderers (UE) use. The
  path touches the RENDERER only (no joint writes, no solver FK, no PD),
  so wheel visuals cannot perturb physics BY CONSTRUCTION. The URDF
  wheel-link vgeoms are hidden (`active_envs_idx` set empty — the Genesis
  renderer then skips them) and re-baked into per-wheel trimeshes with
  their local offsets, so the vehicles look unchanged; nodes are plain
  external nodes (`is_marker=False`), rendered by every camera without
  `debug=True`.
- New `VehicleScene(wheel_render_mode=...)`: `"auto"` (default — instanced
  when supported, else VJS), `"instanced"` (forced; raises when
  unsupported), `"internal_sync"` (the old behavior). `n_envs > 1` and
  non-harvestable wheel URDFs still fall back to VJS automatically. Raw
  `VehiclePhysics` users are unaffected (VJS unchanged there).
- Measured (30 tanks, CPU, 450-step steered drive, same-day baselines):
  headless 20.9 ms/step; **instanced 23.3 ms/step with final positions
  IDENTICAL to headless to 1e-6 m** ("on or off, same physics" — the goal
  of this line of work); VJS 25.2 ms/step with a 9.3 cm final-position
  deviation (the post-1.1.16 residual of its PD micro-forces). I.e. the
  instanced path is now both faster than VJS AND physics-pure.
- Rendered output verified: real URDF wheel meshes visible from ordinary
  offscreen cameras; `path_follow_demo --mp4` PASS unchanged (err 1.49 m).
  `tests/test_kind_visual_batch.py` pins `wheel_render_mode="internal_sync"`
  (it tests VJS itself). VJS is retained for multi-env viewing, the raw
  API path, and as an explicit opt-in; its 1.1.16 clamps stay in place.

---

## [1.1.16] — 2026-07-09

### Fixed — hard-landing "buried" lock-in (high-cast rays) + VisualJointSync physics disturbance bounded

| abbr | meaning |
|---|---|
| VJS | VisualJointSync — viewer-only wheel visual joint driver |
| high-cast | wheel rays start `RAY_UP_OFFSET` above the attachment point; read layer subtracts it back |
| min_d | minimum mean wheel-ray distance (≈ remaining suspension travel) during a drop |

Triggered by a field report: two tanks dropped from z=6.4/6.5 m —
one landed normally, the other ended with its chassis on the ground and
"wheels buried", and toggling VJS flipped the outcome.

- **Root cause 1 (the buried state, VJS-independent): ray-miss lock-in.**
  With ray origins AT the wheel attachment points, a hard landing that
  bottoms out the suspension can sink the chassis far enough that the
  origins go BELOW the ground; the rays then miss, the air mask kills
  `N`, and the vehicle rests on its chassis collision box forever — a
  stable equilibrium (reproduced: 6.4 m and 6.5 m drops ended at
  chassis z = −0.30 with rays reading the 20 m miss sentinel, with or
  without VJS). **Fix: high-cast rays** — `WheelRayPattern` starts each
  ray `RAY_UP_OFFSET` (1.0 m) above the attachment point and
  `read_distances` subtracts the offset from hits (misses keep their
  sentinel), so a bottomed-out chassis still measures the ground; the
  distance goes small/NEGATIVE (a valid over-compression reading, NOT
  air), compression maxes out, and `N` recovers the vehicle. Safe in
  both raycast modes: vehicles are never raycast targets
  (`use_visual_raycasting` defaults False) so the elevated origin cannot
  self-hit. After the fix all drop heights (2.5/4.5/6.4/6.5 m × VJS
  on/off) settle standing at z = 0.393. Normal-regime physics is
  unchanged (same ray line, offset subtracted — the path demos reproduce
  identical results, so existing sweep tables stay valid).
- **Root cause 2 (the VJS sensitivity): the "cosmetic" suspension visual
  is not physics-neutral.** The control path (URDF susp joints with
  dynamics — the tank) drives a PD with kp=1e7 that applies REAL joint
  forces; at the air→ground transition its target jumped by up to the
  full stroke, injecting an impulse into the chassis (measured: VJS On
  compressed 2–3 cm deeper at 2.5–4.5 m drops — enough to flip a
  marginal landing into the buried state, which is exactly what the team
  observed at 6.4 vs 6.5 m). **Fix: suspension visual targets are now
  stroke-clamped and slew-rate-limited** (`_SUSP_VIS_MAX_RATE` = 2 m/s)
  in BOTH writer paths and in the batched `KindVisualBatch` (shared
  `_susp_visual_target` helper); measured disturbance drops to ~0.9 cm.
  The air test also no longer misclassifies negative (over-compressed)
  distances (`d <= 1e-6` → miss-or-exact-zero), in the writers and in
  the closed-form `_susp_visual_offset` mirror.
- Docs: `concepts.md` pipeline step 1 and new `physics-contracts.md`
  §7.8 (high-cast contract: miss sentinel preserved, negative distance =
  valid over-compression, tunnel-ceiling caveat; VJS neutrality bounds);
  `visual.py` header no longer claims "no force feedback".
- Team-facing guidance: UE serving should stay headless (VJS is
  viewer-only; UE reads `wheel_visual_transforms`), but with this fix
  drop-spawns land correctly with the viewer on as well — 151 tests
  pass, path demos PASS unchanged.
- `--mp4` cameras on the path demos now pass `debug=True`: Genesis
  offscreen cameras skip marker/debug-draw nodes unless the camera is a
  debug camera (`rasterizer.py: skip_markers = not camera.debug`), so the
  waypoint polyline was silently missing from every recording (viewer
  runs were unaffected). Found during the solver-free wheel-rendering
  spike (below); reverse-demo recording re-verified with the polyline
  visible.
- Spike result (recorded here, no SDK change): wheels CAN be rendered
  with ZERO solver contact via a pyrender instanced debug-mesh node —
  one node carrying K×n_wheels poses, updated per step from the
  closed-form `wheel_visual_transforms` (`primitive.poses` is a
  streaming setter, `GL_STREAM_DRAW`). Measured at 30 tanks: ~0 ms/step
  added and bit-pure physics, vs VisualJointSync's +0.2 ms and ~9 cm /
  135 m trajectory drift (post-clamp residual), and vs +26 ms for
  solver-entity detached wheels (`set_base_links_pos/quat` is the
  bottleneck). A future VJS replacement should take the instanced
  debug-mesh route (needs: real wheel meshes instead of primitives,
  hiding the URDF wheel visuals, native-viewer path validation).
- Physics-review follow-ups (no CRITICAL findings; core sign/semantics,
  first-step guard, miss sentinel, and all distance-consumer paths
  verified clean): `reset_visual_state()` added to `VisualJointSync` /
  `KindVisualBatch` and wired into `VehiclePhysics.reset` AND
  `MultiVehicleKindPhysics.reset` (which previously reset no visual
  state at all) — a stale slew origin would otherwise inject a transient
  PD force on the first post-reset steps; new regression test pins
  "negative distance = clamped over-compression, exact 0.0 = air" in
  `_susp_visual_offset`. Known pre-existing gap left as-is and
  documented here: `VehicleScene.reset()` in batched-solver mode calls
  no physics-level reset (`veh.physics` is None; nothing invokes the
  kind-level reset) — to be wired in a follow-up.

---

## [1.1.15] — 2026-07-09

### Added — OSC trajectory-client sample; server json gains `omegaMaxDrive`; wheel-inertia override fixed

| abbr | meaning |
|---|---|
| FD | finite difference (velocity from consecutive positions) |
| window FD | FD over a ~0.3 s ring of packets instead of adjacent pairs |

- **New sample #14 `path_follow_osc_demo.py`** — trajectory following
  THROUGH the OSC server: physics in a separate `genesis_vehicle.server`
  process, this script as the game client (receive
  `/Genesis/Vehicle/TargetBulk`, run `PathFollower`, send
  `/Genesis/Vehicle/Control`). Verified: L-path PASS, err 1.46 m in
  30.2 s wall. The client-side reference for UE-style integration,
  including the wire details that bit during bring-up:
  - TargetBulk is UE-frame — decode `pos_g = (Px/100, -Py/100, Pz/100)`,
    `quat_g(wxyz) = (Qw, -Qx, Qy, -Qz)`.
  - There is no velocity channel, and per-packet FD is unusable (the send
    cadence is not 1:1 with sim steps — catch-up bursts and repeated/
    interpolated states make adjacent deltas alternate between ~0 and a
    full step, which bang-bangs the speed loop through the KICK). Use a
    ~0.3 s **window FD**; the sample also pins `--max-catchup-steps 1`.
  - Server-built tuning is matched to the bundled sweep table via the
    `Vehicle/Init` json (`maxBrake`, `omegaMaxDrive`, `wheelOverrides`
    with a `"susp"` substring match) — without the omega cap the
    server-default tank tops out ~9 m/s, far outside the table grid
    (±4 m/s), and the follower loses it at the first corner.
- **Server**: `Vehicle/Init` json now accepts `omegaMaxDrive` (drive
  wheel-omega cap = top-speed limiter, → `drivetrain.omega_max_drive`);
  `wheelOverrides[].inertia` now actually applies — it wrote to a dead
  `w.inertia` attribute (silent no-op) instead of `WheelConfig.i_wheel`.
  NB for the UE plugin: both are json-payload keys, no OSC address
  changes.
- `path_follow_demo --viewer`: bird's-eye default camera
  (`viewer_options` camera at (0, -45, 32) framing the whole course) —
  the Genesis default spawned at the origin, inside the wall/tank.
- `path_follow_osc_demo` gains `--viewer`: opens the SERVER process's
  Genesis viewer (drops `--headless`), and the client polls the server
  process each control tick — closing the viewer window ends the server
  and the client shuts down with it (verified by killing the server
  mid-run: client exits immediately, no orphan processes; the normal
  headless path still PASSes, err 1.49 m).
- **`--mp4 [PATH]` recording** on the path demos (#13, #15): records the
  run to an mp4 headless — offscreen camera at the viewer's bird's-eye
  pose, 20 fps (every 2nd step), HUD overlay (t / pos / v / mode). Shared
  `Mp4Recorder` added to `samples/_hud.py`; opencv-python is required
  only when `--mp4` is used (lazy, actionable error otherwise). Waypoint
  markers + path polyline are now ALWAYS drawn (viewer, video, and
  headless runs alike — the debug-line overlay works headless too;
  guarded with a printed note just in case). Verified: #13 638 frames
  PASS, #15 310 frames PASS; headless regression PASS for both.
- **New sample #15 `path_follow_reverse_demo.py`** — the runnable showcase
  for the 1.1.14 explicit-waypoint-yaw feature: "back into a parking bay"
  (drive past the bay, cusp, reverse along a bezier arc whose 5-tuple
  waypoints carry the planned chassis heading; arrival heading = bay-north).
  Verified: PASS in 15.4 s — pos_err 1.49 m (< 2), yaw_err 0.34 rad
  (< 0.4). `--viewer` draws the forward leg cyan / reverse leg orange.

---

## [1.1.14] — 2026-07-09

### Added — optional per-waypoint yaw for path following (reverse-maneuver headings)

| abbr | meaning |
|---|---|
| ψ | chassis yaw; world +X is ψ = 0, CCW about +Z positive |

- `PathFollower` waypoints may now carry an explicit desired chassis
  heading as an optional 5th tuple element:
  `(x, y, z, target_speed, yaw)`. Convention: radians, world +X = 0, CCW
  positive — identical to what `extract_state` returns. Motivated by
  reverse maneuvers, where "which way should the chassis FACE while
  backing" is a planning input, not something derivable from positions.
- Semantics: an explicit yaw is used VERBATIM as the reference heading
  (no backward +π flip — it already IS the chassis heading); the heading
  loop, and through it the sweep steer inversion, tracks it. `None` or a
  4-tuple falls back to the previous behavior (tangential direction,
  flipped on backward waypoints); 4- and 5-tuples mix freely. An explicit
  yaw on a cusp waypoint also defines the block's arrival heading
  (`_block_end_yaw`). Values are normalized to [-π, π); wrong tuple
  lengths raise.
- Docs: path format updated in `path-following.md` (quick start + §3 with
  the ψ convention and a nonholonomic-drivability note) and the
  api-reference §11 row. 6 new tests (verbatim no-flip, None fallback,
  normalization/mixing, cusp arrival heading, length validation, and a
  closed-loop straight-reverse with pinned yaw) — 150 total pass.
- **`VehicleScene.scene`** — new read-only property exposing the main
  ``gs.Scene`` as the escape hatch for Genesis APIs the wrapper does not
  re-export (debug-draw overlays etc.); registration still goes through
  the ``add_*`` methods.
- Demo/tests polish: corner helpers unified to the waypoint format
  (`(x, y, z, speed)` corners in both the demo's `densify` and the tests'
  `_densify` — z interpolated, previously the test corners were a
  z-less local shorthand); `--viewer` now draws the waypoint-connecting
  polyline via `scene.draw_debug_line` (forward segments cyan, backward
  orange, keyed on the segment speed sign) plus a red goal marker.

---

## [1.1.13] — 2026-07-08

### Changed — path-following stack moved to the recommended dt (0.025 × 10); demo override bug found and fixed

| abbr | meaning |
|---|---|
| quadruple | sweep validity contract: (URDF, preset, config overrides, dt/substeps) |
| MVP | MultiVehiclePhysics (VehicleScene's default batched solver) |

- **Timing**: `sweep_measure` gains `--dt` / `--substeps`, defaulting to
  **0.025 × 10** (the presets' `recommended_dt`, internal 2.5 ms; was a
  hard-coded 0.01 × 4 inherited from the deliverable). Settle/measure
  windows are now defined in seconds and derived from dt.
  `path_follow_demo` runs at 0.025 × 10 and no longer trips the
  recommended-dt WARN. dt/substeps are documented as part of the table's
  validity contract (measure at the dt you drive at) — the triple is now
  a quadruple.
- **Demo bug found while switching dt (present since 1.1.11)**: the demo
  applied `TankTuning` AFTER `vs.build()` — with the default batched
  solver, post-build cfg mutations are ignored without
  `mark_config_dirty()`, and the `if tank.physics is not None` guard
  silently skipped the runtime overrides (`physics` is None under MVP).
  **The demo had been driving on preset defaults** while its CSV was
  measured WITH the overrides. At 10 ms the mismatch was benign (forward
  drive is tolerant); at 25 ms the un-tuned tank cannot break away from
  standstill under combined throttle+steer (43 s stall), which exposed
  the bug. Isolated by experiment: not raycast_mode / initial yaw /
  brake-settle / RigidOptions / solver mode / SFL — a config-dirty
  rebuild alone fixed the launch. Fix: cfg overrides BEFORE `build()`;
  post-build overrides via `Vehicle.resolved` (works in both solver
  modes) — new `TankTuning.apply_resolved`. Documented as the
  "override-ordering trap" in `path-following.md` §1.
- Bundled reference table re-measured at 0.025 × 10 (GPU full grid,
  ~3.6 min). Verified with overrides actually applied: demo PASS
  (err 1.46 m, 31.8 s) and the forward–reverse–forward cusp run PASS in
  13.8 s (was 64 s — the overrides' `T_BRAKE_MAX` makes the cusp stop
  immediate; the old 20 s creep-stop was a symptom of the unapplied
  overrides, not a follower property).
- Docs: `path-following.md` §2 restructured — new "two-stage controller"
  concept section (stage 1: path errors → desired physical response
  `a_target`/`ω_target`, vehicle-agnostic; stage 2: sweep-table inverse at
  the current operating point → actuator commands, all vehicle specifics
  in the table) with a Mermaid dataflow diagram, plus the step-by-step
  mechanics. NOT changed: the legacy `DT = 0.02` samples
  (`road_loop`, `city_traffic_ego`, `l2l3_minimal`, perf benches) — their
  published benchmark figures are tied to that dt.

---

## [1.1.12] — 2026-07-07

### Changed — sweep measurement: deliverables_v3 accuracy + build-once fixes adopted

| abbr | meaning |
|---|---|
| A1/A2 | v3 accuracy fix IDs (measurement frame / wheel radius) |
| P1–P3 | v3 performance fix IDs (build-once / batched gravity / fewer reads) |
| build-once | build the scene once, reuse across chunks via reset |

`genesis_vehicle.control.sweep_measure` now carries the deliverables_v3
revision of the measurement pipeline (CLI, CSV schema, grid unchanged):

- **[A1] `a_measured` frame: world-x → body-longitudinal.** Acceleration
  is now measured on the yaw-projected longitudinal speed
  (`v_long = vx·cos ψ + vy·sin ψ`) — the same definition
  `extract_state` feeds `PathFollower`, so the table is produced and
  consumed in one frame. The old world-x read under-measured by cos(yaw)
  whenever steer ≠ 0 rotated the vehicle during the 2 s window (full-grid
  v2-vs-v3 audit: mean |Δa| 1.6–1.8 m/s², p95 ≈ 7, max 12 m/s² on
  |steer| ≥ 0.5 rows; high-speed + full-steer combos turned past 90° yaw,
  making the old values meaningless).
- **[A2] wheel-spin init**: per-wheel resolved radii with a mean fallback
  for unresolved wheels (1.1.11 had already removed the deliverable's
  `TIRE_R = 0.4` hard-code; this adds the fallback).
- **[P1] build-once**: the scene/physics are built ONCE and reused across
  chunks via `scene.reset()` + `physics.reset()`; the last chunk is padded
  to the chunk size (excess rows discarded) so the batch shape stays fixed
  and nothing re-JITs. Chunk count also stops over-building
  (`n_envs = min(n_envs, combos)`).
- **[P2] batched per-env gravity**: one `(n_envs, 3)` `set_gravity` call
  instead of an env-indexed Python loop (17,500 solver calls on the full
  grid).
- **[P3] measure-loop reads**: velocity (+quat) only at the window's
  start/end, omega_z as a running sum — no per-step history stacking.
- v3 reference timings: full grid GPU 3 m 37 s (vs v2 4 m 40 s, 1.29×);
  CPU ≈ 20 min. Reset-vs-rebuild physics parity on steer = 0 rows:
  max |Δa| ≈ 0.05 m/s² (verified here too: 3-chunk-with-padding vs
  single-chunk max |Δa| 0.006, |Δω_z| 0.000 on the smoke grid).
- **Bundled reference table re-measured** with the corrected CLI
  (`samples/data/tank_sweep_signed.csv`, full grid, GPU): the v2-era
  steer ≠ 0 rows were A1-biased. NB: the deliverables_v3 canonical CSV
  (`tank_sweep_v3_gpu.csv`) was measured WITHOUT the TankTuning overrides
  (`Config: (none)` in its logs), so it is not a valid drop-in for the
  demo's (URDF, preset, overrides) triple — hence the re-measurement.
  The overrides now live in `samples/tank_tuning.py` (importable
  `TankTuning` + module-level `apply_config`/`apply_runtime_config`,
  directly usable as `sweep_measure --config`), documenting the triple
  next to the CSV.
- Re-verified with the new table: `path_follow_demo` PASS (err 1.50 m,
  32.2 s) and the forward–reverse–forward cusp run PASS (lateral
  deviation 0.00 m). 144 tests pass.
- Docs: `index.md` gains a **Built-in utilities** section — a categorized
  map (control & workflow / model preparation & scene assembly /
  telemetry & rendering feed) of every shipped utility with its one-line
  function, entry point, and detail-doc link.

---

## [1.1.11] — 2026-07-07

### Added — `genesis_vehicle.control`: path following as an official SDK utility

| abbr | meaning |
|---|---|
| sweep table | measured (v, throttle, steer, pitch, roll) → (a, ω_z) grid of one vehicle |
| cusp | sign flip of a waypoint's `target_speed` = gear change (stop, then reverse) |
| block | maximal run of same-speed-sign waypoints (the follower's driving unit) |

- The `deliverables_v2` path2ST pipeline (path + signed target speeds →
  per-step throttle/steer/brake via sweep-table inversion) is now the
  built-in `genesis_vehicle/control/` subpackage:
  - `PathFollower` / `SweepTable` exported at top level;
    `extract_state(_from_arrays)` in `genesis_vehicle.control`.
  - `python -m genesis_vehicle.control.sweep_measure` measures the sweep
    CSV for any (URDF, preset, config-override) triple. CPU default per
    SDK convention; `--gpu` recommended (the measurement is a ~500-env L3
    batch, past the crossover in `docs/backends.md`). Wheel-spin
    initialization now takes per-wheel radii from the resolved config
    (the deliverable hard-coded `TIRE_R = 0.4`).
  - Dependencies dropped: pandas/scipy removed (the SDK never used them) —
    CSV IO via numpy, `RegularGridInterpolator` replaced by an equivalent
    clamped 4-D multilinear interpolation, `brentq` by bisection at the
    same 1e-3 tolerance. Control-side code needs numpy only (usable
    without Genesis/torch installed).
- Two deliverable bugs fixed during adoption:
  - **Backward-yaw flip was a no-op**: `(yaw + pi) % (2*pi) - pi` is a
    plain normalization, so backward waypoints stored chassis yaw off by
    pi; masked on collinear reversing where the position-error term
    vanishes.
  - **Cross-cusp lookahead contamination**: v_target and the steering
    geometry took the direction of the lookahead waypoint BEYOND the cusp
    while the transition state machine keyed on the current index, so
    approaching a cusp applied backward geometry while still driving
    forward (the follower could leave the path without ever triggering
    the stop-and-reverse). The follower now splits the path into
    direction blocks: projection/lookahead never cross the active block's
    boundary, |v_target| tapers into the cusp
    (`k_approach * distance + 0.3`), and the stop-and-reverse triggers on
    reaching the block end. Tuning knobs (`lookahead`, `cusp_goal`,
    `k_approach`, `v_stop`, ...) are constructor kwargs.
- Bundled assets: `samples/urdf/tank_ray.urdf` (primitive-only 10-wheel
  tank) + `samples/data/tank_sweep_signed.csv` (reference sweep for that
  URDF + the tank preset + the demo's `TankTuning` overrides).
- New sample `samples/path_follow_demo.py` (#13): tank follows a
  wall-detour path closed-loop; PASS = final error < 3 m. Verified:
  32.2 s sim, err 1.49 m PASS (CPU). A forward–reverse–forward cusp run
  on the real tank also passes (modes DRV+1 → BRAKE_TRANS → DRV-1 →
  BRAKE_TRANS → DRV+1 → DONE, lateral deviation 0.00 m).
- Docs: new `docs/path-following.md` (pipeline, measurement CLI, tuning
  table, path requirements); `api-reference.md` §11; index + samples
  README rows. 19 new pure-Python tests (sweep inversion recovery on a
  known-linear synthetic table, CSV round-trip, grid-hole/zero-level
  validation, closed-loop unicycle straight/L-shape/diagonal-cusp/
  straight-cusp scenarios) — 144 total pass.

---

## [1.1.10] — 2026-07-06

### Changed — CHANGELOG glossary tables translated to English

- All 15 `| 약자 | 의미 |` abbreviation tables in this file are now
  `| abbr | meaning |` with English descriptions (headers + every
  Korean cell). Content unchanged — translation only.

## [1.1.9] — 2026-07-05

### Changed — terminology sweep completed: "split"/"single"/"raywheel" retired from all prose

- The 1.1.0 unification retired "two-scene" but left the older mode
  shorthands in tables, docstrings and comments. All prose now uses the
  official names `dual_scene` / `single_scene`:
  - `docs/dual-scene-raycast.md` — every performance table header/cell and
    surrounding prose (was "raycast: split (re-cast)", "single (GPU)", …);
    the "in the tables below, split = dual_scene" legend is gone.
  - Code docstrings/comments: `core.py` (distance-injection hook),
    `vehicle_scene.py` (StaticBody/Vehicle handles, add_static routing),
    `samples/dual_scene_terrain.py` help texts, `server/env_builder.py`
    ("inline-vs-raywheel" → single_scene-vs-dual_scene),
    `server/l3_runtime.py` ("in raywheel" → dual_scene),
    `samples/terrain_drive.py` (stale `two-scene-raycast.md` doc link).
- Deliberately kept: the alias documentation (`raywheel`/`split`/`inline`/
  `single` remain ACCEPTED by `raycast_mode`), the `two_scene_terrain.py` /
  `two-scene-raycast.md` redirect stubs, legacy sandbox filenames
  (`car_raywheel.py`), and English uses of "split" (torque split, morph
  split, string.split).

## [1.1.8] — 2026-07-05

### Docs — L2×L3 combined GPU crossover measured: the invariant is K×N ≈ 300–400 total vehicles

- Measured the CPU/GPU crossover for the combined L2×L3 path
  (`MultiVehiclePhysics(n_envs=N)`, SDK-direct via
  `samples/perf_l2_l3_combined.py`): K ∈ {1,2,5,10} × N ∈ {1,10,50,100,300}
  × both backends, 40 cells, `car_4w`, all clean.
- Headline: the crossover N shifts left as K grows (K=1 → N~310, K=2 →
  ~150, K=5 → ~65, K=10 → ~40) — CPU cost is ~linear in K·N while GPU
  stays launch-bound-flat in N — so the practical rule is **GPU once
  K×N ≳ 300–400 total batched vehicles**, consistent with the v1.1.6
  L3-only tank figure. The GPU floor still rises mildly with K
  (N=1: 37.8 → 59.2 ms for K 1 → 10, the residual L2 anti-scaling term),
  so tiny-N configs stay CPU regardless of K.
- Throughput ceilings differ 4×: CPU saturates at ~6,000 veh-steps/s for
  any K/N mix; GPU reaches 24,200 veh-steps/s at K=10×N=300 (3,000
  vehicles, 4.1× over CPU) and is still climbing.
- Full table + per-workload rules recorded in `docs/backends.md` §3
  (section restructured into "L2 and L3 separately" / "L2 × L3 combined").
  Docs-only; no code changes.

---

## [1.1.7] — 2026-07-05

### Docs — new `docs/backends.md`: consolidated CPU-vs-GPU physics backend guide

- Backend-selection guidance was scattered (vehicle_scene docstrings,
  `server.md` §2.1/§3, `batching.md` perf notes). New `docs/backends.md`
  consolidates it: physics backend vs renderer, `VehicleScene.init_backend`
  / server `--gpu` selection semantics (process-global, set once; explicit
  mismatch warns, missing CUDA propagates), why CPU is the default
  (launch-bound at small batch width), the v1.1.6 measured crossover table
  with rules of thumb per workload (L2 always CPU; L3 `--gpu` at ~300+
  envs), the GPU-serving architecture notes (physics-only on GPU, capture
  math on host, sync-inside-timing), and WSL2 notes (`LD_LIBRARY_PATH`,
  crossover shifts left on native hardware).
- Cross-linked from `docs/index.md` (nav table), `server.md` §3 rationale,
  and `batching.md` perf section. No code changes.

---

## [1.1.6] — 2026-07-05

### Fixed — server crashed at ≥~180 targets: TargetBulk burst the 64 KB UDP datagram limit

- `send_target_states_bulk` packed ALL targets into ONE datagram. A 10-wheel
  target is ~90 args ≈ 450 B encoded, so ~180 tanks crossed the 65,507-byte
  UDP maximum — `OSError: [Errno 90] Message too long` and the server died
  (discovered the moment 200-tank configs were added to the benchmark; 100
  tanks ≈ 36 KB had always been safe). Now chunked into self-contained
  packets of ≤120 targets, each ending with the `-1` sentinel — the client
  parses per-packet `[ID, …]` records, so this is wire-compatible (one
  frame's states simply arrive in a few packets). `send_dynamic_states_bulk`
  already chunked; TargetBulk had been missed. Verified: L3×200 now serves
  (27.9 ms/step) where it previously died before the first `[STATS]`.

### Changed — samples/docs stop overriding `raycast_mode`: dual_scene (the default) everywhere

- Removed the explicit `raycast_mode="single_scene"` from `quickstart.py`,
  `batched_rollout.py`, `city_traffic_ego.py`, `l2l3_minimal.py`,
  `multi_env_render.py`, `perf_vectorization.py`, `perf_l2_l3_combined.py`,
  `perf_multi_vehicle.py` and the `docs/quickstart.md` snippet — they now
  use the SDK default (`dual_scene`). The L3 samples were directly
  contradicting the docs (dual is 3.4× at 256 envs); the flat-plane ones
  were saving ~6 % (~0.7 ms full-step) at the cost of teaching every reader
  a non-default configuration. Perf-sample baselines shift accordingly
  (n_envs>1 rows improve).
- `docs/dual-scene-raycast.md` recommendation wording demoted: flat-ground
  `n_envs=1` `single_scene` is now described as an optional
  micro-optimization, not the recommended mode ("switch to single_scene
  only for..." removed). Comparison samples (`dual_scene_terrain.py`,
  `obstacles_and_ramp.py --mode/--bench`) keep both modes on purpose.

### Changed — benchmark: default tank matrix gains 200 and 400; `--gpu` now valid for L2 too

- Default `--tanks` is now `1,10,30,100,200,400`.
- `--gpu` is applied to BOTH modes (previously the benchmark only forwarded
  it for L3). Measured crossover campaign (simple terrain, ms/step):

  | tanks | L2 CPU | L2 GPU | L3 CPU | L3 GPU |
  |---|---|---|---|---|
  | 30 | 15.7 | 109.5 | 12.1 | 14.5 |
  | 100 | 31.1 | 684.0 | 17.4 | 15.6 |
  | 200 | 73.2 | (impractical) | 27.4 | 31.5 |
  | 400 | — | — | 46.2 | **27.8** |

  **L3 crosses over at ~250–300 envs** (400: GPU wins 1.66×) — the
  long-standing "hundreds of envs" guidance now has a measured location.
  **L2 has NO practical GPU crossover — it anti-scales** (30 tanks 7×,
  100 tanks 22× slower than CPU; the GPU build alone took >14 min): growing
  K grows ONE env's system, so every per-vehicle sensor read / solver stage
  pays launch+sync latency with no env-axis batch width to amortize it.
  Recorded in `docs/server.md` §2.1/§3; `--gpu` help and the L3 module
  docstring updated with the measured numbers.

## [1.1.5] — 2026-07-05

### Added — `--vis_mode` on the collision benchmark's viewer

- `benchmark_collision.py --viewer` now passes `--vis_mode` through to the
  server, defaulting to **`visual`** (URDF visual meshes — turret, hull and
  the ten ray-cast wheels). The server's own default is `collision`, which
  for `tank_ray.urdf` renders only the single chassis collision box (the
  wheels are visual-only by wheel-raycast design: ground contact comes from
  the raycast + tire model, not wheel meshes), so viewer runs previously
  showed bare green boxes. `--vis_mode collision` restores that view.
- Headless (measurement) runs are unchanged — no vis flag is passed.

## [1.1.4] — 2026-07-05

### Changed — server mode terminology unified: L2 / L3 ("per-entity" demoted to a gloss)

- The two server modes were named asymmetrically: one by implementation
  ("per-entity"), the other by batching axis ("L3"). Official names are now
  batching-axis based — **L2 mode** (K interacting entities in one world)
  and **L3 mode** (1 entity × n_envs) — matching what the benchmarks
  already called them. "per-entity" survives only as a parenthetical gloss
  on first mention.
- User-facing surfaces updated: `[MODE] === L2 (per-entity) === / === L3
  (multi-env) ===` banners, the `[STATS] [L2]` tag (was `[per-entity]`), a
  new **`--l3` alias** for `--multi-env`, and the mode naming in help
  texts, `docs/server.md` / `batching.md` / `dual-scene-raycast.md`, and
  server-module comments.
- Deliberately kept: phrases like "per-entity loop / per-entity get_pos"
  that DESCRIBE per-entity iteration — those are not mode names and
  renaming them would change the meaning. No external-parser impact (both
  benchmarks' `[STATS]` regexes are tag-agnostic).

### Changed — server package comments/docstrings translated to English

- All Korean comments and docstrings across `genesis_vehicle/server/*.py`
  (legacy-migrated `osc_manager` / `physics_server` / `env_builder` /
  `vehicle_builder` plus the newer `l3_runtime` / `pacing` / `benchmark*`)
  are now English, matching the SDK-core convention. Korean **log/print
  strings are intentionally kept** (operator-facing console output). No
  code or behavior change — verified by py_compile, the full test suite,
  and an end-to-end benchmark smoke run.

### Added — `--viewer` on the collision benchmark

- `benchmark_collision.py --viewer` runs the server WITH the Genesis viewer
  window (drops `--headless`) so the ring convergence + pile-up can be
  watched live. Rendering adds per-step overhead (~22 vs ~12 ms/step at 4
  tanks), so viewer runs print a warning and their numbers are indicative
  only — reference results stay headless.

## [1.1.3] — 2026-07-05

| abbr | meaning |
|---|---|
| host path | closed-form wheel-pose capture math computed on the CPU (physics stays on GPU) |
| DtoH / HtoD | GPU→CPU / CPU→GPU transfer |
| launch-bound | regime dominated by the fixed latency per kernel launch |

### Changed — "GPU mode = physics-only on GPU, serving math on CPU" completed (host-side capture math)

- Added `MultiVehicleKindPhysics.wheel_visual_reads()` (the 5 raw DEVICE
  tensors: pos/quat/steer/dist/spin) and `wheel_visual_transforms_host(...)`
  (the identical closed-form math on CPU tensors; static inputs cached on
  the host once). `L3State.capture` now downloads the 5 raw reads + the
  obstacle poses in ONE DtoH sync via `_to_host_batched` and computes the
  wheel world poses **on the CPU** — removing the dozens of small kernel
  launches that GPU-backend capture used to trigger. The chassis pos/quat
  reuse the same download (double read removed). Parity is pinned by
  `test_wheel_visual_transforms_host_matches_device_path` (the math is
  device-agnostic, so CPU parity == GPU parity). Pre-first-step and
  multi-kind cases fall back to the device path.

### Fixed — GPU Physics Avg mis-attributed the execution tail to "serving"

- The L3 loop's `torch.cuda.synchronize()` sat OUTSIDE the physics timing
  (bottom of the loop), so ~2 ms of asynchronous GPU execution tail was
  missing from Physics Avg and booked as serving instead. The synchronize
  moved inside the step timing — `[STATS]` per-step now reflects the real
  execution time on GPU too.

### Conclusion — final steady-state decomposition, GPU L3×100 (all hypotheses measured)

| | physics ms/step (honest timing) | serving ms/loop | Loop |
|---|---|---|---|
| CPU | 16.2 | ~20.3 | 36.5 |
| GPU | **21.7** | ~25.6 | ~47.3 |

- Neither the capture sync batching (1.1.1) nor the CPU migration of the
  capture math (this release) moved the steady-state Loop meaningfully;
  after the timing fix the dominant term is **GPU physics itself**
  (launch-bound: still +5.5 ms over CPU physics at n_envs=100). The
  residual ~5 ms serving delta is the per-step HtoD upload of the 3 input
  tensors + capture getter kernels, within ±3 ms run noise.
- Meaning: the "physics-only on GPU, serving on CPU" architecture is now in
  place, so at the hundreds-of-envs scale where GPU physics starts winning,
  serving will not be the bottleneck. For fleets ≤100 envs the server
  recommendation stays CPU (unchanged).

## [1.1.2] — 2026-07-05

### Fixed — stale "per-entity forces CPU by design" wording in docs/server.md

- The §2.1 GPU-table intro claimed `--gpu` is L3-only. In fact BOTH server
  modes accept `--gpu` since v1.0.14 (per-entity prints a CPU-is-faster
  warning and proceeds on GPU); it is the BENCHMARK matrix that only
  measures GPU on L3, because per-entity is `n_envs=1` — no GPU batch
  width, so L2+GPU only pays kernel-launch overhead. Wording now says
  exactly that. No code change.

## [1.1.1] — 2026-07-05

### Changed — L3 capture downloads with ONE DtoH sync (`_to_host_batched`) + honest GPU-serving attribution

- `L3State.capture` now gathers every read tensor (chassis pos/quat, wheel
  visual poses, obstacle poses) and downloads them via a single on-device
  concat + ONE `.cpu()` (`l3_runtime._to_host_batched`; CPU backend keeps
  per-tensor conversion — the helper auto-detects; mixed dtypes fall back
  safely). Pinned by `tests/test_to_host_batched.py` (4 cases incl. a
  forced-concat parity run).
- **Measured honestly: steady-state Loop did NOT move** (GPU L3×100 simple:
  46.7 ms before/after) — so the earlier "per-read sync latency" attribution
  of the +8 ms GPU serving gap was wrong as the *dominant* term. Per-section
  [PROFILE] (sync-accurate) shows the gap is the **kernel-launch overhead of
  capture-side GPU compute** (`wheel_visual_transforms` = dozens of small
  kernels; 0.72 ms CPU vs 3.19 ms GPU for identical bytes), plus the
  post-step `synchronize()` execution tail and HtoD input uploads. The
  single-sync change is kept (strictly fewer round-trips, free on CPU);
  the next lever, if GPU serving ever matters, is computing capture-side
  wheel poses on the CPU from one raw-state download. `docs/server.md` GPU
  section rewritten accordingly. CPU remains the server recommendation.

### Added — collision-stress server benchmark (`server/benchmark_collision.py`)

- New `python -m genesis_vehicle.server.benchmark_collision`: K tanks spawn
  evenly on a ring (radius auto-scaled from K, `--radius` to override), each
  rotated to face the center, and drive at identical constant throttle until
  they all collide in the middle — the collision-regime companion to the
  official grid benchmark (`server.benchmark`), whose tanks never touch.
- Drives the REAL L2 server over the OSC wire (same mock-UE pattern as
  `benchmark.py`, on its own ports 7111/7112/7114) and — new — LISTENS to
  the `/Genesis/Vehicle/TargetBulk` state stream to track chassis positions
  and yaw, so the whole path (contact solving + state encode) is exercised.
  The listener sets `max_packet_size = 65535`: `TargetBulk` is one datagram
  for all K tanks (~10.7 KB at K=30) and socketserver's 8 KB default
  silently truncates it.
- Steering is a P-controller aiming each tank at the origin; after the first
  impact the tanks stay pressed together (sustained K-way contact) instead
  of ricocheting out of the ring, which is the regime being measured.
- Reports both "speed drops": physical (v_peak → v_end mean tank speed,
  collision instant = first drop below 50 % of peak) and simulation
  (mean ms/step before vs after the pile-up + slowdown factor), plus a
  0.5 s-resolution timeline (mean ring radius, mean speed, ms/step).
- Reference results (WSL2 laptop, CPU, dt = 0.025, throttle 0.8, in
  `docs/server.md` §2.2): 10 tanks 12.9 → 13.8 ms/step (1.07×);
  30 tanks 14.0 → 19.5 ms/step (1.39×, peak ~24) — the 30-way pile-up
  stays inside the 25 ms real-time budget.

## [1.1.0] — 2026-07-05

### Changed — terminology unification: "two-scene" → dual-scene; official subsystem name = wheel-raycast

- Official naming, everywhere in docs and APIs: the subsystem is the
  **wheel-raycast** (ray-cast wheels sensing the ground); its
  ``raycast_mode`` has two modes — **``dual_scene``** (default; separate
  static-BVH raycast scene) and **``single_scene``**. The older "two-scene"
  phrasing (which meant ``dual_scene``) is retired from all current docs and
  docstrings; legacy aliases (``raywheel``/``split``/``inline``/``single``)
  remain accepted by ``raycast_mode``.
- Renames (with back-compat shims):
  - ``docs/two-scene-raycast.md`` → ``docs/dual-scene-raycast.md``
    (old path keeps a one-line redirect stub; the doc now opens with the
    official-terminology note).
  - ``samples/two_scene_terrain.py`` → ``samples/dual_scene_terrain.py``
    (old module name still runs via a deprecation shim that forwards to the
    new one).
  - Internal ``VehicleScene._two_scene`` → ``_dual_scene`` (private).
  - ``l3_runtime`` now passes the canonical ``raycast_mode="dual_scene"``
    (was the ``"raywheel"`` alias).
- All cross-references updated (``docs/index.md``, ``api-reference.md``,
  ``concepts.md``, ``samples/README.md``, sample docstrings, module
  comments). CHANGELOG history is intentionally left as written.
- Version jumps to **1.1.0**: the terminology freeze plus the v1.0.x
  performance campaign (40 Hz default, batched proxy/mirror sync, kinds fix,
  adaptive catch-up pacer, official server benchmark) make a natural minor
  milestone.

---

## [1.0.20] — 2026-07-04

| abbr | meaning |
|---|---|
| L2 / L3 | per-entity (K entities × 1 env) / multi-env (1 entity × n_envs) |
| ms/step | cost of one physics step (the `[STATS]` per-step value) |
| steps/loop | catch-up steps per loop (~1.0 = real-time, pinned at the cap = saturated) |

### Added — `genesis_vehicle.server.benchmark` (official end-to-end server test)

- Drives the REAL server subprocess over the OSC wire with a built-in mock
  UE client: full handshake (`Init/Physics` dt=0.025 → `Vehicle/Init`
  SkidSteer→tank preset → K `Init/Target` → 88 convex-hull `Init/Obstacle`
  chunks → `Init/Done`), ~30 Hz `Vehicle/Control` input streaming, `[STATS]`
  collection (first line dropped as warm-up), then `stop`.
- Matrix: {L2, L3} × {simple(plane), complex(88 hulls)} × {1, 10, 30, 100}
  tanks; CPU default (`--gpu` = L3 opt-in), `--road-raycast-only` on,
  non-default ports (7101/7102/7104) so a live UE session is never hit.
  Prints one summary table: ms/step, steps/loop, Loop Avg, and a real-time
  verdict (steps/loop ≤ 1.05 AND Loop Avg ≤ 25 ms). Filters: `--modes`,
  `--terrain`, `--tanks`, `--urdf`, `--stats`. See `docs/server.md` §2.1.

### Added — adaptive catch-up pacer (`server/pacing.py`, both server modes)

- The catch-up cap is now ADAPTIVE by default (`AdaptiveCatchup`): the server
  monitors **steps/loop** and auto-switches between
  - **burst** (cap = max(5, 0.1/dt)) — normal mode; backlog from a hiccup is
    caught up and real-time is recovered, and
  - **smooth** (cap = 1) — entered on sustained overload (window-average
    steps/loop ≥ 1.5): recovery is impossible anyway, so degrade to a steady,
    burst-free slow motion instead of jerky 5-step bursts. Returns to burst
    once step-loops run a full window consecutively under 0.9×dt (headroom
    back), with a switch cooldown as hysteresis and a startup **grace** of
    100 loops (observe-only: the build/JIT transient right after init
    polluted the window and spuriously switched a healthy L2×30 run to
    smooth at t=1.4 s — with grace it stays burst, while genuine overload
    still triggers right after the grace expires; bench-verified both ways).
- `--max-catchup-steps N` now PINS the cap and disables the pacer (the old
  fixed behavior; `1` = always-smooth). Switches log as
  `[Pacing] [AdaptiveCatchup] …`; every `[STATS]` line carries the live mode
  (`[cap=N:burst|smooth|fixed:N]`).
- **`--pacing-profile`** (off by default; the server benchmark enables it):
  every switch additionally dumps its trigger context on one greppable line —
  the window's steps/loop history, loop-duration avg/p95, the dt budget, the
  estimated speed ratio, and time since the previous switch. Example:
  `[profile] switch#1 BURST→SMOOTH @t=1.7s | window steps/loop=[0,1,2,…]
  (avg 2.40) | loop_dur avg 65.5 ms p95 111.1 ms | budget 25.0 ms | est
  speed 0.92x`. The benchmark collects these per config and prints them as
  `[pacing]` lines in the final report.
- Verified end-to-end via the new server benchmark: overloaded L3×100 tanks
  auto-switched within the first window (steps/loop 2.5 → cap=1) and settled
  at 1.0 steps/loop with **Loop Avg 89.6 → ~38 ms** (bursts gone); healthy
  L2×30 stayed `[cap=5:burst]` throughout. State machine pinned by
  `tests/test_adaptive_catchup.py` (8 pure-python cases: overload switch,
  recovery, streak reset, idle loops don't fake recovery, fixed mode,
  cooldown).

### Fixed — L3 server crashed on genesis builds lacking `prefer_parallel_linesearch`

- `run_l3` passed `prefer_parallel_linesearch=False` unconditionally;
  genesis builds without that RigidOptions attribute (e.g. the 1.2.0 PyPI
  build) raised at startup. Unrecognized RigidOptions keys are now dropped
  with a warning and construction retried.

---

## [1.0.19] — 2026-07-04

| abbr | meaning |
|---|---|
| ss | substeps (integration subdivisions within one step) |
| internal dt | dt/ss — the solver's internal integration interval |

### Changed — SDK-wide default timing is now 40 Hz (dt = 0.025)

- Follow-through on 1.0.17 (server fallback → 0.025): the presets and samples
  now default to 40 Hz too, so the whole SDK speaks one default.
  - `VehicleConfig.recommended_dt` default `1/48 → 0.025`.
  - All five presets: car/awd/truck `1/48 → 0.025`; **tank `0.005 → 0.025`**
    (the 200 Hz recommendation was legacy-variant conservatism — the server has
    been running tanks at dt 0.02–0.025 / ss2 = internal 10–12.5 ms
    throughout the perf campaign, and the 1.0.17 bumpy-terrain A/B showed
    cruise/z-oscillation/yaw within noise; samples use ss10 → internal
    2.5 ms, far finer still).
  - Samples pick this up via `cfg.recommended_dt` automatically;
    `two_scene_terrain`'s hardcoded `1/48` updated to 0.025.
- Users can still pick any dt (`VehicleScene(dt=...)` /
  `SimOptions.dt` / server `--override_dt`); a client-sent dt still wins in
  the server, and the preset-mismatch warning stays informational.
- **The resolved timing is now logged on every run**:
  `VehicleScene.build()` prints
  `[genesis_vehicle] timing: dt=25.0ms (40Hz) x substeps=10 (internal
  2.50ms) | n_envs=… | raycast_mode=…` — alongside the server's existing
  `[Determinism]` line. Docs updated (`quickstart.md`, `README.md`,
  `two-scene-raycast.md`, `api-reference.md`).

---

## [1.0.18] — 2026-07-04

| abbr | meaning |
|---|---|
| HUD | the `--viewer` cv2 overlay window (offscreen camera + synchronous render) |
| pacer | sleep logic that holds the loop to wall-clock real time (1×) |
| budget | real-time budget per step = dt (48 Hz → 20.8 ms) |

### Changed — `terrain_drive --viewer` (cv2 HUD) now runs at TRUE real time (1x)

The cv2 path had no pacing at all — but it was not "too fast": with the HUD
rendering every step it ran at 0.6x (35 ms/step), i.e. slow motion, and would
free-run on a faster machine. Measured on CPU/WSLg and fixed in four parts:

- **Pacer with carry-over debt** (the subtle one): the HUD renders every
  `render_every`-th step, so a render step overruns the budget and the steps
  between must pay the debt back. A naive "resync on overrun" pacer zeroes
  that debt at every render step and locks in ~0.8x (measured 40.1/48
  steps/s with per-phase accounting). Debt now carries over; only a real
  hitch (>0.25 s behind) resyncs. Result: **47.98 steps/s vs 48.0 target**.
- **Coarse VISUAL terrain copy**: the offscreen raster is geometry-bound
  (~15 ms/frame at the full 70k-face terrain, ~10 ms at 17k; pixel count
  barely matters — 1280×720 vs 1024×576 was 15.9 vs 14.6 ms). The main-scene
  visual copy (viewer modes only) now uses `res=1.0` (~17k faces), deviating
  ≤ ~5 mm from the exact full-res raycast surface the wheels ride. The
  raycast mesh (physics) is untouched.
- **HUD at 24 fps, 1024×576**: render every 2nd step (was every step at
  1280×720) — the whole loop now fits the 20.8 ms budget.
- **One pos + one vel engine read per step**, reused by the governor,
  centerline steer, z stats, wrap check and HUD (was ~6 reads ≈ +3 ms/step;
  `_hud_render` takes `p`/`speed` as arguments now).

`--native` needs none of this (the Genesis viewer's `realtime_factor=1.0`
pacer already holds 1×, and it now draws the coarse visual copy too:
47.2 steps/s). Headless stays UNPACED by design (benchmark/CI: ~102
steps/s). 110 pytest pass; drive/on-terrain/bumps checks OK on all three
paths.

---

## [1.0.17] — 2026-07-04

| abbr | meaning |
|---|---|
| ss | substeps (integration subdivisions within one step; the server uses 2) |
| internal dt | dt/ss — the solver's internal integration interval |
| z osc-std | std-dev of chassis z oscillation during steady cruise (suspension-stability metric) |

### Changed — server default dt fallback 0.02 → 0.025 (40 Hz budget)

- The 20 ms budget was tight for large fleets/maps; 25 ms gives +25 % per-step
  budget and ~−20 % total CPU (40 loops/s instead of 50). Verified the physics
  is indistinguishable before changing the default — tank on bumpy rco
  terrain, CPU, settle + 4 s cruise + 2 s turn:

  | | 0.02/ss2 (old) | **0.025/ss2 (new)** | 0.025/ss3 |
  |---|---|---|---|
  | internal dt | 10.0 ms | 12.5 ms | 8.3 ms |
  | cruise speed | 5.596 m/s | 5.594 m/s | 5.587 m/s |
  | z osc-std | 14.3 mm | 14.3 mm | 14.4 mm |
  | turn yaw-rate | −1.127 rad/s | −1.112 (−1.3 %) | −1.113 |
  | NaN / divergence | none | none | none |

  ss3 (finer internal dt) buys nothing → substeps stays 2.
- Scope: this changes the server **fallback** only — a client-sent dt still
  wins (`/Genesis/Init/Physics` arg 1). To actually run 40 Hz against a
  client that sends 0.02, either change the client's dt or launch with
  `--override_dt 0.025` (help text + `docs/server.md` updated with the
  rationale). The resolved value is printed in the `[Determinism]` line and
  echoed to the client via the Pacing message, as before.

---

## [1.0.16] — 2026-07-04

| abbr | meaning |
|---|---|
| pacer | the Genesis viewer's realtime_factor pacer (sleeps at the end of `viewer.update()`) |
| HUD | the `--viewer` cv2 overlay window (offscreen camera + synchronous render) |

### Fixed — `terrain_drive --native`: the car visibly "trembled" (camera one step behind)

- The native-viewer follow set the camera **from the drive loop, after
  `vs.step()` returned**. But the realtime pacer sleeps INSIDE
  `viewer.update()` (i.e. inside `vs.step()`), right after the renderer
  receives the fresh chassis pose — so for most of each ~21 ms frame the
  async viewer thread drew car pose *k* against camera pose *k−1*. At
  7 m/s that is a ~15 cm car-vs-camera offset flickering at the draw rate:
  the car appeared to tremble/judder while `--viewer` (cv2 HUD: synchronous
  `cam.set_pose` + render, always same-step) looked rock solid. Physics was
  never affected — pure render-phase artifact.
- Fix: use Genesis's `viewer.follow_entity(veh.entity_main)` — the camera
  is updated inside `viewer.update()`, microseconds from the pose push, in
  the same call. No smoothing, so the seamless 100 m wrap snap stays
  invisible (a smoothed camera would sweep 100 m backwards through the
  scene). `ViewerOptions.camera_pos` doubles as the follow offset
  (`(0, -13, 2.8)`, matching the old side-follow view).
- The per-step `set_camera_pose` follow is kept only as a fallback for
  Genesis versions without `follow_entity`.
- Verified: `--native` runs at the pacer's real-time 48 Hz (46.4 steps/s
  measured) with the follow engaged; headless + cv2 paths unchanged;
  110 pytest pass.

---

## [1.0.15] — 2026-07-04

| abbr | meaning |
|---|---|
| kind | the batched solver's batching unit (vehicles sharing one cfg object) |
| VJS | VisualJointSync (drives the wheel visual joints for the Genesis viewer) |
| MVP | `MultiVehiclePhysics` |
| FK | Forward Kinematics (was triggered on every solver entry) |

### Changed — cross-kind solver I/O batching (batching audit #9)

- A multi-kind ``MultiVehiclePhysics.step`` paid 6 solver entries PER KIND
  (4 state reads + 2 force/torque applies). Kinds cannot share the *compute*
  (different wheel counts / strategy code by definition), but the I/O can:
  the step now pre-reads every kind's base-link state in ONE batched
  ``get_links_*`` set (4 calls total), injects per-kind slices
  (``kind.step(state=...)``), and applies all kinds' deferred force/torque in
  ONE combined call pair (``defer_apply=True``). Single-kind scenes take the
  unchanged fast path. First-step-protection kinds contribute zeros
  (physically identical to the old skip).
- Measured (10 kinds × 1 tank, dual, CPU — the worst case the pre-1.0.8
  server used to create): step **27.7 → 20.2 ms** (MVP 19.2 → 12.2; ~54 solver
  entries/step removed). The remaining per-kind pipeline overhead is the
  documented per-kind compute floor.

### Changed — `KindVisualBatch`: K visual writers → one solver call (audit #10)

- With the Genesis viewer on, each of a kind's K ``VisualJointSync.step``
  calls issued its own ``set_dofs_position`` (+ susp velocity-zero, + control
  path) — each a solver entry with collider/constraint reset + FK. Same-kind
  entities share one dof LAYOUT, so all K writes now collapse into ONE
  solver-level ``set_dofs_position`` over concatenated global dof indices
  (``entity._dof_start + local``), one velocity-zero, one control call —
  identical math on ``(n_envs, K, n)`` slabs (equivalence pinned by
  ``tests/test_kind_visual_batch.py``). Falls back to the per-entity loop if
  layouts differ.
- Measured (30 tanks, dual, CPU, VJS forced on): step **23.3 → 14.2 ms**
  (MVP 12.4 → 3.4) — the viewer overlay now costs ~0.8 ms over the headless
  baseline instead of ~9 ms. Headless / UE runs are unaffected (VJS off).

### Docs — synced to v1.0.11–v1.0.15

- `two-scene-raycast.md` step list now describes the batched proxy/mirror
  sync; `batching.md` visual-write + cross-kind sections and the stale
  road_loop "6 % faster" figures updated; `api-reference.md` MVP notes gain
  the kinds-split warning and v1.0.15 numbers; `server.md` gains
  `--max-catchup-steps` and the `[MODE]`/`[PROFILE]`/`[STATS]` diagnostics
  reference.

---

## [1.0.14] — 2026-07-04

| abbr | meaning |
|---|---|
| L2 | per-entity mode (K interacting vehicles, `n_envs=1`) |
| L3 | multi-env batched mode (`--multi-env`, one URDF × `n_envs=N`) |
| env | parallel sim instance (one slot on the `n_envs` axis) |

### Changed — CPU is now the default backend EVERYWHERE; `--gpu` opts in

GPU is kernel-launch bound at small batch sizes: measured, L3 GPU is a flat
≈ 19 ms/step (30/50/100 tanks alike) while L3 CPU is 8.4 ms at 30 — CPU wins
until roughly hundreds of envs, and per-entity (L2, `n_envs=1`) is not even
close (10 vehicles: CPU 47 vs GPU 160 ms). The defaults now match that
reality across the whole tree:

- **Server**: `--multi-env` (L3) now defaults to CPU like per-entity (L2)
  already did. `--force-cpu` is REMOVED; the new `--gpu` flag opts into GPU
  in either mode (falls back to CPU with a warning when CUDA is
  unavailable). L2 with `--gpu` warns that CPU is usually faster.
- **Samples**: every sample that hard-coded `init_backend("gpu")` or had a
  `--cpu` flag now defaults to CPU with a `--gpu` opt-in (`quickstart`,
  `slope_hold`, `terrain_drive`, `obstacles_and_ramp`, `two_scene_terrain`,
  `l2l3_minimal`, `road_loop`, `city_traffic_ego`, `batched_rollout`,
  `multi_env_render`, `perf_vectorization`, `perf_multi_vehicle`,
  `perf_l2_l3_combined` — the perf sweeps thread `--gpu` through their
  `--internal` subprocesses). NB: `--cpu` flags are gone; the perf tables in
  `docs/batching.md` were GPU-measured, so pass `--gpu` to reproduce them.
- **Legacy vehicle demos**: `demo_drive.py` (all four legacy variants) and
  the tank `demo_interactive.py` likewise default to `gs.cpu` with `--gpu`.
- Bare `torch.cuda.synchronize()` timing calls are now guarded
  (`torch.cuda.is_available()` / backend flag) so everything runs on
  CUDA-less machines.

No physics change; `VehicleScene.init_backend` itself already defaulted to
"cpu". Docs updated: `server.md` (mode table + "why CPU by default"),
`batching.md`, `quickstart.md`, `two-scene-raycast.md`, `README.md`.

---

## [1.0.13] — 2026-07-04

| abbr | meaning |
|---|---|
| mirror | a dynamic obstacle's raycast-scene follower body (re-synced every step) |
| FK | Forward Kinematics (ran over the whole raycast scene on every set call) |
| capture | the server function that reads poses for the UE stream |

### Batching audit — remaining per-item python loops in the step path

Full audit of the SDK + server for unbatched per-item loops, with dispositions
(✅ done earlier / 🔧 batched in this release / ❌ engine-limited / ➖ by design):

| # | loop | disposition |
|---|---|---|
| 1 | vehicle proxy sync (dual) | ✅ 1.0.11 |
| 2 | dynamic-obstacle mirror sync (dual) | 🔧 this release |
| 3 | server capture chassis/obstacle pose reads | 🔧 this release |
| 4 | capture wheel poses (K× full-batch recompute) | ✅ 1.0.7 |
| 5 | `lerp_state` per-quaternion slerp | ✅ 1.0.10 |
| 6 | OSC bulk-encode per-scalar `float()` casts | 🔧 this release |
| 7 | capture per-row `.copy()` (fresh arrays anyway) | 🔧 this release |
| 8 | K raycaster sensor reads | ❌ genesis per-sensor API (measured ~0.5 ms @K=10) |
| 9 | MVP per-kind loop | ➖ kinds have different shapes by definition |
| 10 | VisualJointSync per-entity loop | ⏸ viewer-only (headless/UE unaffected); revisit if native-viewer K grows |
| 11 | per-vehicle `set_inputs` + rebucket | ➖ µs-scale python, negligible |

### Changed — dynamic-obstacle raycast mirrors join the batched sync (#2)

- 1.0.12 gave dynamic obstacles raycast mirrors, re-synced per step via
  `_sync_dynamic` — a per-obstacle `set_pos`/`set_quat` loop with the same
  2-FK-per-body cost the vehicle proxies had (~0.8 ms/mirror). The mirrors
  live in the SAME raycast-scene rigid solver as the proxies, so
  `_sync_proxies_batched` now writes proxies + mirrors in ONE batched set +
  ONE FK. Measured (CPU, dual, 30 tanks + 20 mirrors, plane):
  **30.5 → 16.5 ms/step** (measure_distances 22.5 → 8.7). Fallback loop keeps
  both body types.

### Changed — batched capture pose reads + zero-copy rows (#3, #7)

- `capture_state` called `entity.get_pos()/get_quat()` per target and per
  dynamic obstacle — 2·K engine entries per capture, twice per step. New
  `_BatchPoseReader` does ONE `get_links_pos/quat` solver read per group
  (identical env-0 user-frame semantics; falls back to the old path when
  `readers=None`). Redundant per-row `.copy()` dropped in both servers'
  captures (the source arrays are freshly created each capture).

### Changed — vectorized OSC bulk encode (#6)

- `send_target_states_bulk` cast ~8 numpy scalars through `float()` per pose
  (30 targets × 10 wheels ≈ 2,600 casts per send). The UE-frame conversion
  (cm scale + Y flip + quat reorder) is now one numpy pass with plain-python
  `.tolist()` rows. Wire format unchanged (OSC packs float32 either way).

---

## [1.0.12] — 2026-07-04

| abbr | meaning |
|---|---|
| rco | `--road-raycast-only` (road as a raycast-only surface, main-scene collider omitted) |
| L2 / L3 | per-entity (K entities × 1 env) / multi-env (1 entity × n_envs) |
| BVH | Bounding Volume Hierarchy (raycast acceleration structure) |

### Changed — per-entity (L2) server defaults to `dual_scene`

- The per-entity server built `raycast_mode="inline"` (single_scene) unless
  rco was given (`physics_server.py`) — the only entry point whose default
  diverged from the SDK (`VehicleScene` default) and L3 (always `raywheel`).
  Now **both server modes default to `dual_scene`**: statics get a kinematic
  raycast mirror (static BVH — no per-step re-fit; wheels ride the exact mesh
  surface), and rco composes on top (drops the main-scene road collider).
- **Dynamic obstacles keep their wheel-raycast semantics**:
  `env_builder.build_obstacles` now passes `wheel_raycast=vs.is_dual_scene`
  to `add_dynamic`, so moving ramps/platforms get a per-step-synced raycast
  mirror. Without this, switching to dual_scene would have made wheels clip
  through every dynamic obstacle (rays only hit the raycast scene there;
  single_scene sensed rigid colliders implicitly).
- New `--single-scene` flag (per-entity only) restores the pre-1.0.12
  one-scene behavior; rejected with `--road-raycast-only` at arg-parse time
  (an rco road is a kinematic raycast surface — it needs the raycast scene),
  ignored with a warning under `--multi-env`.
- Docs: `docs/server.md` §3 documents the raycast-scene default + opt-out.

### Changed — terminology: server modes labeled with their batching axes

- "per-entity" IS the L2 axis (K interacting vehicles × 1 env) — the name
  predates 1.0.8, when each vehicle really ran its own `VehiclePhysics`;
  since then identical targets are batched per *kind*, so the mode is L2 in
  both senses. Docs and the startup banner now say **per-entity (L2)** /
  **multi-env (L3)** consistently (`docs/server.md` §3 mode table gains a
  "Batching axis" column; the stale "N × VehiclePhysics loop" solver cell is
  corrected). CLI flags are unchanged (`--multi-env` stays; no `--l3`).
- Tests: `test_dynamic_obstacle_gets_wheel_raycast_mirror_in_dual_scene`
  guards the mirror (dual: mirror present; single: flag skipped, no warning
  spam). 106 pytest pass.

---

## [1.0.11] — 2026-07-04

| abbr | meaning |
|---|---|
| proxy sync | copying each vehicle's pose to its raycast-scene proxy every step (dual_scene) |
| FK | Forward Kinematics (ran over the whole raycast scene on every set_pos/set_quat) |
| L2 / L3 | per-entity (K entities × 1 env) / multi-env (1 entity × n_envs) |

### Changed — batched proxy sync (dual_scene): 2·K whole-scene FK passes → 1

- `VehicleScene._measure_distances` synced each vehicle's proxy via
  `Vehicle._sync_proxy()` — a python loop where every `set_pos`/`set_quat`
  triggers a WHOLE-raycast-scene FK: 2·K FK passes + 4·K engine entries per
  step (~1 ms/vehicle; measured **29.8 ms at K=30**, i.e. 80 % of the
  L2-vs-L3 dual gap and the reason "no-collision L2 ≈ L3" did not hold).
- New `_sync_proxies_batched()`: ONE batched `get_links_pos/quat` read from
  the main solver + ONE `set_base_links_pos(skip_forward=True)` +
  `set_base_links_quat` write (single FK) onto all proxies. Pose semantics
  identical (user-frame read / world-frame write, same as `_sync_proxy`,
  which is kept for `reset()` and as the automatic fallback — one-time
  `[genesis_vehicle:proxy-sync]` warning if the batched path ever fails).
- Measured (CPU, plane, 30 tanks, zero inter-vehicle contact, dual both):

  | ms/step | L2 dual 이전 | L2 dual 이후 | L3 dual |
  |---|---|---|---|
  | raycast/proxy | 29.77 | **6.79** | 4.92 |
  | vs.step() 합계 | 37.57 | **13.45** | 8.95 |

  "충돌 없으면 L2 ≈ L3" now holds within ~1.5× (both real-time at 30
  vehicles); L3 also gains slightly (its single proxy went 2 FK → 1:
  10.4 → 9.0 ms). Equivalence pinned by `tests/test_proxy_sync_batched.py`
  (batched == loop poses; step path takes the batched branch, no fallback).

---

## [1.0.10] — 2026-07-03

### Added — `VehicleScene.add_raycast_surface()` (first-class raycast-only entity)

- A wheel-raycast-ONLY static surface: a Kinematic `use_visual_raycasting`
  body in the raycast scene with **zero rigid-solver presence** — no
  collision geoms, no FK/dynamics, no broadphase pairs, build-once BVH, and
  the mesh-guard face limit does not apply. This is the first-class name for
  `add_static(collision=False, wheel_raycast_morph=...)` — the negative-flag
  spelling hid the intent (its single_scene fall-through was a real bug
  pre-1.0.7). `dual_scene` only; single_scene raises the same fail-fast
  `ValueError`. env_builder's `--road-raycast-only` road path now uses it.
- Context: ray-cast wheels never need the road as a *collider* — the solver
  already gates contact work (SAP broadphase + hibernation), but a road
  mesh's AABB spans the whole map, so chassis-vs-road narrow-phase/SDF ran
  every step with zero actual contact. A raycast-only surface removes those
  pairs at the source. Measured (CPU L3, 4.4k-face road, 85 hulls): step
  9.90 vs 11.09 ms — and with a CoACD road collider the chassis rubs the
  convex bulge above the true surface, dragging cruise speed 5.53 → 4.66 m/s
  (−16 %), so the raycast-only surface is also the *more correct* physics.

### Changed — vectorized `lerp_state` (server interpolation, O(N·wheels) python → one numpy pass)

- The server interpolates every chassis/wheel/obstacle pose each loop before
  the OSC send. This was one python `slerp` call per quaternion — 30 tanks ×
  10 wheels = 300+ calls/loop. Field data (CPU L3, 30 tanks): non-physics
  loop overhead grew to **14.2 ms/loop**, rivaling the physics step itself
  (15.5 ms) and pushing the loop past the 20 ms budget.
- `lerp_state` now gathers all quats into flat arrays and slerps them in ONE
  numpy pass (`_slerp_batch`: normalize → shortest path → spherical weights
  with nlerp fallback — same semantics as the scalar `slerp`, which is kept).
  Measured: **3.42 → 0.36 ms/call** at 30 targets × 10 wheels (9.4×),
  1.12 → 0.18 ms at 10 targets. Output format unchanged (per-target tuples;
  entries missing from `prev` still pass through untouched). Both server
  modes benefit (L3 imports the same function).
- Equivalence pinned by `tests/test_lerp_state_vectorized.py`: vectorized
  output matches a verbatim copy of the old per-quaternion loop (positions,
  quats, wheel-angle wrap lerp, passthrough) at α ∈ {0, 0.3, 0.9999}.

---

## [1.0.9] — 2026-07-02

### Added — `--max-catchup-steps` (server pacing knob)

- The catch-up cap (max physics steps per loop when behind real-time) was
  hardcoded to `max(5, 0.1/dt)` in both server modes. It is now overridable
  via `--max-catchup-steps N` (floored at 1; default unchanged).
- The cap does **not** make anything faster — once a step exceeds the dt
  budget (20 ms at dt=0.02) slow-motion is unavoidable. It selects the
  degradation mode: the default 5 tries to recover real-time by bursting up
  to 5 steps in one loop (a 5×-step blocking burst → jerky frame pacing),
  while `1` runs exactly one step per loop → a steady, burst-free slow
  motion at a constant ratio (smoothest visual result under overload, paired
  with the TimeDilation the server already sends). Irrelevant once the step
  fits the budget (steps/loop reads ~1.0 either way).
- Renamed the loop variable `MAX_SUBSTEPS` → `MAX_CATCHUP_STEPS` (both server
  modes): it is a **server pacing cap** (steps per loop), unrelated to the
  Genesis `SimOptions.substeps` physics sub-integration parameter the old
  name suggested.

---

## [1.0.8] — 2026-07-02

| abbr | meaning |
|---|---|
| kind | the batched solver's batching unit (vehicles sharing one cfg object) |
| MVP | `MultiVehiclePhysics` |
| JIT | Just-In-Time kernel compilation (taichi/torch first-step cost) |

### Fixed — per-entity server split K same-URDF vehicles into K kinds (batching never engaged)

- The batched solver groups vehicles into kinds by **cfg object identity**
  (`group_vehicles_by_cfg`), but `build_vehicle` called `build_cfg()` fresh
  **per target** — so 10 identical tanks became **10 kinds × 1 vehicle** and
  `MVP.step` ran 10 sequential single-vehicle pipelines instead of one
  (B=10) batched pipeline.
- Field report matched exactly: team measured `[PROFILE] SDK compute
  37.79 ms` (CPU, 10 tanks); reproducing the fresh-cfg-per-target pattern in
  a bench gives **MVP 33.8 ms vs 2.8 ms as one kind** (~12×). This — not the
  map — was the dominant per-entity CPU cost.
- Fixed: `build_cfg` caches and returns the **same cfg object** for identical
  `(urdf_path, mapping, t_fric)` (`target_id` excluded — log-only), so
  same-URDF targets land in one kind. Logs
  `[Batch] Vehicle N: reusing shared cfg`. Targets with different
  friction/mapping still get their own kind (correctness preserved).
  Regression test: `test_build_cfg_shares_one_object_per_identical_target`.

### Fixed — `[PROFILE]` overstated GPU section times (JIT in first steps)

- The startup `[PROFILE]` measured the first 5 steps ever, which include
  taichi/torch kernel JIT compilation — field report showed `SDK compute
  100 ms` in PROFILE vs 23 ms steady-state on GPU (L3). Both servers now run
  2 unprofiled warmup steps before the 5 profiled ones.

---

## [1.0.7] — 2026-07-02

| abbr | meaning |
|---|---|
| MVP | `MultiVehiclePhysics` (K vehicles batched into one compute) |
| rco | `--road-raycast-only` (road mesh as wheel-raycast surface, no collider) |
| OSC | Open Sound Control (UE ↔ physics-server wire protocol) |
| K | number of vehicles (L2 batch size) |
| CoACD | convex decomposition preprocessing for mesh colliders |
| SDF | Signed Distance Field (per-geom cost a mesh collider pays every step) |
| BVH | Bounding Volume Hierarchy (raycast acceleration tree) |
| FK | Forward Kinematics (recomputes link poses on `set_pos`/`set_quat`) |

### Fixed — rco roads were fall-through in the per-entity server (single_scene)

- **`--road-raycast-only` + per-entity server mode silently built a
  fall-through road.** The per-entity server ran `raycast_mode="inline"`
  (single_scene), where `add_static(collision=False, ...)` warned and built a
  plain rigid from the raycast morph. env_builder's rco road morph itself
  carries `collision=False`, so that rigid had **no collision geoms and was
  invisible to the wheel raycaster** — vehicles fell straight through the road.
  Reproduced with 10 tanks on a 4.4k-face mesh road: tank z → −76 m (free
  fall).
- Two-part fix, keeping the design intent (a no-collision raycast surface is a
  **dual_scene** feature — the kinematic `use_visual_raycasting` body lives in
  the raycast scene, whose static BVH is the whole point):
  - **`add_static(collision=False)` in single_scene now FAILS FAST** with a
    `ValueError` + `[genesis_vehicle:single-scene]` error log instead of
    building a silently broken scene (pre-1.0.7: warning + fall-through rigid).
    dual_scene behavior is untouched.
  - **The per-entity server switches to `raycast_mode="dual_scene"` when
    `--road-raycast-only` is given** (default stays inline). The rco road then
    lands where it was designed to: a kinematic exact-surface raycast body with
    a build-once BVH, and no chassis-vs-road narrow-phase in the main scene.
  Verified: same 10-tank mesh-road run drives normally on dual_scene + rco
  (z ≈ surface, 5.6 m/s).
- **Per-frame numbers** (10 tanks, 4.4k-face mesh road, CPU, `vs.step()` only):

  | path | ms/step (p95) | wheel surface | build |
  |---|---|---|---|
  | inline + full CoACD collider (default) | 13.8 (19.3) | convex bulge (**+1.7 m** above the true concave surface) | **40.7 s** (CoACD) |
  | dual_scene + rco (`--road-raycast-only`) | 20.2 (23.7) | exact mesh | 16 s |

  dual+rco trades the road-collider cost (main_scene.step drops to 2.4 ms) for
  a **per-vehicle proxy-sync cost**: `_sync_proxy` is a Python loop doing
  `set_pos`/`set_quat` (each an FK) per vehicle — **9.9 ms at K=10** (~1
  ms/vehicle), plus 5.3 ms raycast-scene step. On this *small* road inline+full
  is faster; on real UE maps the full-collider path grows with mesh count/faces
  (per-geom SDF, narrow-phase, refit) while dual+rco stays flat — and rides the
  exact surface. **Known optimization target:** batch the K proxy syncs into
  one solver write + single FK (est. 9.9 → ~1–2 ms, bringing dual+rco to
  ~12–13 ms at K=10).
- A/B-verified no dual_scene regression from this change
  (`two_scene_terrain --compare --n-envs 64`, GPU: dual 37.3 ms pre-edit vs
  39.7 ms post-edit — within run noise; single_scene varied 46–55 ms across
  runs).

### Fixed — `--road-raycast-only` was silently ignored in `--multi-env` (L3)

- `run_l3` never forwarded `args.road_raycast_only` to
  `env_builder.build_obstacles` — argparse accepted the flag but it changed
  nothing, so an A/B test of the flag in multi-env mode was literally
  no-op-vs-no-op. Now forwarded: in raywheel the wheels already ride the exact
  kinematic mirror, so rco's effect in L3 is dropping the **main-scene CoACD
  road collider** (no chassis-vs-road narrow-phase/SDF, replicated per env).

### Added — server mode banner + per-step [STATS] + startup [PROFILE]

- **Mode banner**: startup prints `[MODE] === PER-ENTITY ===` (K interacting
  vehicles, n_envs=1, CPU-forced) or `[MODE] === MULTI-ENV (L3 batched) ===`
  so reports are unambiguous about which path ran. `[STATS]` lines are tagged
  `[per-entity]` / `[L3 n_envs=N]` too.
- **`[STATS]` now shows per-step physics**: `Physics Avg` is the SUM of the
  loop's catch-up steps (up to MAX_SUBSTEPS=5), which reads 4–5× the true
  per-step cost when the server is saturated. Both modes now print
  `(X steps/loop, Y ms/step)` alongside. `steps/loop` pinned at 5.0 = the
  server cannot keep real-time (permanent slow-motion).
- **`[PROFILE]` one-shot breakdown at startup**: during the 5 warmup steps the
  server times each step's sections — `raycast/proxy | SDK compute | genesis
  solver | 기타` — and prints one line (instrumentation removed afterwards; on
  GPU each section boundary synchronizes for true attribution). Pinpoints
  where a slow step goes without a repro environment.

### Changed — OSC server `capture_state` batched wheel readback (O(K²) → O(K))

- In batched solver mode, `Vehicle.wheel_visual_transforms()` recomputes the
  **whole K-vehicle batch** and slices one vehicle
  (`vehicle_scene.py` → `MultiVehiclePhysics.wheel_visual_transforms`). The
  server's `capture_state` called it once **per target**, doing K× the full-K
  compute every capture (twice per sim step). At K=10 tanks that readback cost
  **5.6 ms/step** — 34 % of the 20 ms real-time budget on CPU.
- `capture_state` now takes `mvp=` (the scene's `vs.physics`) and computes the
  batch **once**, slicing each vehicle by its `_slot`. Measured: 5.64 →
  **0.72 ms/step** (7.8×) at K=10; whole server loop ~16.3 → ~11.5 ms
  (real-time headroom ×1.22 → ×1.73). Per-vehicle fallback kept for
  `solver="per_vehicle"` / missing slot. Outputs bit-identical
  (`np.allclose` parity-checked old vs new).

---

## [1.0.6] — 2026-06-28

### Fixed — `car_4w.urdf` wheels floating above the ground

- **The visual wheel radius (`0.32 m`) did not match the physics radius
  (`0.358 m`)** that `car_4w_rwd_ackermann`'s wheel overrides impose
  (the car wheel-override table, matching the reference car URDF). The
  raycast suspension rests the chassis for a `0.358 m` wheel, so the smaller
  `0.32 m` visual cylinder was drawn ~38 mm above the contact point — the
  wheels visibly **floated** over the terrain (most obvious in
  `samples/terrain_drive.py --viewer`, where the bumpy surface made the gap and
  the chassis pitch stand out).
- Fixed by setting the four wheel `<cylinder>` visuals in `car_4w.urdf` to
  `radius="0.358"` so the rendered wheel matches the physics radius. **Pure
  visual fix — physics is unchanged** (the pipeline already used `0.358`), so no
  numeric drift: 96 pytest pass, `terrain_drive` / `quickstart` drive
  identically.
- NB: this does **not** change how much the chassis pitches/heaves over
  `terrain_drive`'s default `--amp 0.7` relief — that is the rigid car
  geometrically tracking a deliberately *severe*, short-wavelength surface
  (terrain forcing 0.3–0.8 Hz ≪ 2.4 Hz suspension natural freq, so no spring/
  damper tuning isolates it). Drive a gentler surface (`--amp 0.3`) or lower the
  cruise speed for a calm ride.

---

## [1.0.5] — 2026-06-27

### Fixed — road_loop vehicles flying off (revert 1.0.4 collision; drop the Truck)

- **1.0.4's `enable_collision=False` made vehicles launch UPWARD** (no rigid
  ground backstop → raycast-suspension overshoot, z→thousands). Reverted to
  `enable_collision=True`: all cars stay grounded (z≈0.09), 0/12 flown.
- **The 6-wheel Truck is removed from road_loop.** In the 16-vehicle tight loop
  it destabilizes the step and ~9/16 vehicles explode to z=thousands — and that
  is **independent of solver** (batched/per_vehicle) **and substeps** (30/50),
  with deterministic per-index blow-ups, so it is not a tuning knob. road_loop is
  now the 3 car kinds (FWD/RWD/AWD): stable at `substeps=10`, ~56 ms/step, all 12
  grounded. The truck drives fine standalone in its own demo (single
  vehicle, substeps=50). (`--truck` removed.)
- The final-pose summary now lists **every** vehicle with `z` + a `FLOWN` flag —
  the old one-per-kind summary hid the fly-aways (only the first-of-each-kind,
  which happened to survive, was printed).

---

## [1.0.4] — 2026-06-27

### Added — live FPS; Fixed — `--truck` launching the cars

- `samples/road_loop.py` prints a **live `fps` / `ms-per-step` line** to the
  console every ~0.5 s of the drive loop — works in every view (`--native`,
  `--viewer` cv2, headless).
- **`enable_collision=False`** (road_loop scene): ray-wheel vehicles float on
  raycast suspension (wheels sense the ground regardless), so chassis↔chassis
  rigid collision isn't needed for a spaced loop — and **with** it the heavy
  truck (`--truck`), which drifts wide, **rammed neighbouring cars and launched
  them** ("cars bouncing out"). Off → cars stay put; also a bit faster. (The
  truck itself still understeers wide off-frame under `--truck` — an opt-in,
  documented quirk; tame it later with a lower truck throttle.)

---

## [1.0.3] — 2026-06-27

### Changed — `road_loop` default = 3 car kinds (fast + all-visible); Truck is `--truck`

- The 6-wheel **Truck is now opt-in via `--truck`**. It was the source of both
  reported problems: (1) it needs `substeps=30` (3x slower) to avoid a
  constraint-force NaN, and (2) being heavy + partial-Ackermann + overpowered it
  understeers **wide off the loop** (spirals out to r≈33 m → "not all vehicles
  visible"). The **default is now the 3 car kinds** (FWD/RWD/AWD): they hold the
  loop (r≈25) and are stable at `substeps=10` → **~60 ms/step vs ~130**, and all
  vehicles stay in frame. `--truck` restores the 4th kind (at `substeps=30`).
- HUD speed line + final-pose summary are now **kind-count agnostic** (no
  hardcoded 4-kind assumption, which would `IndexError` on the 3-kind default).

---

## [1.0.2] — 2026-06-27

### Changed — `road_loop --solver` choices match `VehicleScene(solver=)`

- `samples/road_loop.py`: `--solver` choices `multi_batched` / `per_vehicle` →
  **`batched` / `per_vehicle`** (default `batched`), matching the 1.0.0 SDK's
  `VehicleScene(solver=...)` naming; the value is now passed straight through.
  The road_loop Truck stability fix (`substeps=30`, so the 5000 kg 6-wheel truck
  doesn't blow constraint forces to NaN the moment it drives) carries over to
  1.0.x unchanged — re-verified on 1.0.1: 16 vehicles, no NaN, ~130 ms/step.

---

## [1.0.1] — 2026-06-27

### Changed — `VehicleScene.InitBackend` → `init_backend` (PEP 8); fix spurious backend warning

- Renamed the physics-backend classmethod from PascalCase `InitBackend` to
  snake_case **`init_backend`** — to match the rest of the SDK (`set_inputs`,
  `add_vehicle`, `add_camera`, `mark_config_dirty`) and PEP 8. The PascalCase name
  is **removed (no alias)**; it shipped only in 0.9.37 / v1.0.0 (hours earlier), so
  nothing depends on it. New call: `VehicleScene.init_backend("gpu")`. All 13
  samples + the server + tests + docs updated.
- Fixed a spurious warning: `init_backend("gpu")` followed by `VehicleScene(...)`
  warned "backend already initialized as 'gpu'; ignoring 'cpu'" on **every** GPU
  scene, because the constructor's silent auto-ensure passed its cpu default into
  the mismatch check. The auto-ensure (in `__init__`) now reuses whatever backend
  is up **without warning**; only an *explicit* `init_backend(...)` requesting a
  different backend warns. (Bug was present in v1.0.0.)

Verified: quickstart drives on GPU with **no** spurious warning; an explicit
cpu→gpu `init_backend` mismatch still warns; 96 pytest.

> The `v1.0.0` tag is left in place; this is the follow-up `v1.0.1`.

---

## [1.0.0] — 2026-06-27

**VehicleScene is the SDK.** 1.0.0 makes `VehicleScene` the single high-level entry
point that owns everything — `gs.init`, the Genesis scene(s), `build` / `step`,
rendering, and L1/L2/L3 batching — so application code (and every sample + the OSC
server) never touches `gs.Scene` / `VehiclePhysics` / `MultiVehiclePhysics`
directly. The low-level layer is still public for control `VehicleScene` doesn't
expose (see api-reference §1).

This release is the sum of phases 1–5 (0.9.22 → 0.9.39); the headlines:

- **Rendering API** (0.9.22): `view=None | "native" | "cv2"`, an SDK `Camera` +
  `vs.add_camera()` (any view mode), `vs.cameras`. VisualJointSync auto-managed.
- **Per-env inputs** (0.9.23): `veh.set_inputs(...)` takes per-env `(n_envs,)`
  tensors.
- **Batched solver, default** (0.9.24–0.9.27): `solver="batched"` (default) wraps
  `MultiVehiclePhysics`; works in BOTH raycast modes (dual_scene injects the
  raycast-scene distances); same-kind vehicles auto-group (lazy, dirty-tracked);
  per-vehicle accessors (`veh.wheel_visual_transforms()` / `.resolved` /
  `.distances`) work in either solver mode.
- **Full scene encapsulation** (0.9.36): `main_scene` / `raycast_scene` are private;
  the thin accessors `vs.viewer` / `vs.rigid_solver` / `vs.sim_options` /
  `vs.is_dual_scene` cover the legitimate reads.
- **Physics backend via `VehicleScene.init_backend("cpu" | "gpu")`** (0.9.37),
  process-global + set once, **default CPU**; double-init warns + is ignored. The
  renderer is separate (always GPU; physics-CPU + GPU-render is valid).
- **All 12 samples + the OSC server migrated** onto `VehicleScene` (0.9.28–0.9.35,
  0.9.27); docs/README swept to the new API (0.9.39).

Regression for the release: **96 pytest**, **all 13 SDK samples** drive headless,
**all 4 legacy vehicle demos** run (one variant's `z anchored` sub-check is a pre-existing
Genesis-1.2.0 numeric-drift quirk, not a regression).

### Migration from 0.9.x

| was | now |
|---|---|
| `VehicleScene(backend="gpu", …)` | `VehicleScene.init_backend("gpu")` then `VehicleScene(…)` (default cpu) |
| `vs.main_scene` / `vs.raycast_scene` | `vs.viewer` / `vs.rigid_solver` / `vs.sim_options` / `vs.is_dual_scene` |
| `vs.main_scene.add_camera(...)` | `vs.add_camera(...)` |
| `show_viewer=True` | `view="native"` (alias still accepted) |
| low-level `MultiVehiclePhysics(scene, …)` for L2/L3 | `VehicleScene(solver="batched", n_envs=N)` |

---

## [0.9.39] — 2026-06-27

### Docs — sweep all docs/README for the 1.0.0 API (1.0.0 phase 5)

Brought the prose docs in line with phases 1–5 after a full drift scan:

- **quickstart.md** + **README.md** "Getting Started" + **two-scene-raycast.md**
  example: rewrote the old low-level `gs.init` + `VehiclePhysics` +
  `add_vehicle(scene,…)` snippets to the `VehicleScene` pattern
  (`VehicleScene.init_backend("gpu")` → `add_ground_plane` → `add_vehicle(preset=)`
  → `build` → `veh.set_inputs / vs.step`). Removed the now-false "the SDK does NOT
  fully encapsulate Genesis" claim from the README.
- **api-reference.md** §0 + **samples/README.md**: the VJS auto-manage note now
  says `view="native"` / `vs.add_camera(...)` (was `show_viewer=True` /
  `vs.main_scene.add_camera`).
- **batching.md** + **concepts.md**: added the high-level shortcut note —
  `VehicleScene(solver="batched", n_envs=N)` drives L2/L3 (the page documents the
  underlying layer it wraps).

No stale `backend=` / `vs.main_scene` / `vs.raycast_scene` left in the SDK docs.

---

## [0.9.38] — 2026-06-26

### Fixed — `terrain_drive` wrap used the old `veh.entity` name (regression sweep)

A full headless sweep of all 13 SDK samples at 0.9.37 caught `terrain_drive`
crashing with `AttributeError('Vehicle' object has no attribute 'entity')` in its
wrap-around teleport — only reached after `WRAP_PERIOD` of driving, so missed by
earlier spot checks. A leftover from the `entity` → `entity_main` rename; fixed to
`veh.entity_main`. Re-verified: drove +116 m over 1 wrap (OK). **All 13 SDK samples
pass.**

---

## [0.9.37] — 2026-06-26

### Changed — physics backend via `VehicleScene.init_backend()`, default CPU; renderer separate (1.0.0 phase 5)

The physics backend is no longer a `VehicleScene(...)` argument. Set it (process-
global, once) with the classmethod **`VehicleScene.init_backend("cpu" | "gpu")`**
BEFORE constructing any scene; the **default is now CPU**. Constructing a
`VehicleScene` without it auto-initializes CPU. Any double-init — a second
`init_backend`, or a stray `gs.init` — **warns and is ignored** (the backend can't
change within a process).

The **renderer is separate from the physics backend**: the viewer / cameras
rasterize on the GPU regardless, so **physics-CPU + GPU-render is valid** (and
tested — a real frame renders under `gs.cpu`); "GPU physics + CPU render" is not a
thing. With no GPU present, `build()` warns that rendering falls back to slow
software.

Consolidated the 7 scattered `gs.init` sites (server ×2, visual samples ×4, tests)
onto `init_backend`; removed `backend=` from all 13 sample `VehicleScene` calls +
both server entry points + the tests. Documented in the module docstring +
api-reference §0 ("Backends — physics vs renderer"). Verified: `init_backend`
default cpu + mismatch warning; `two_scene_terrain --cpu` drives on CPU;
`quickstart` drives on GPU; 96 pytest.

---

## [0.9.36] — 2026-06-26

### Changed — remove the public `main_scene` / `raycast_scene` properties (1.0.0 phase 4)

`VehicleScene` now fully encapsulates the underlying Genesis Scene(s): `main_scene`
/ `raycast_scene` are private (`_main_scene` / `_raycast_scene`). External code goes
through the high-level API + four narrow accessors added for the legitimate
reads/tweaks that remained:

- `vs.viewer` — the native viewer (or `None`) — replaces `vs.main_scene.viewer`
  (cv2 / native HUD + terrain_drive camera follow).
- `vs.rigid_solver` — read-only sim introspection (`n_geoms` / `n_links` /
  `faces_info`) — replaces `vs.main_scene.(sim.)rigid_solver`.
- `vs.sim_options` — the runtime physics tweaks (`dt` / `gravity`) the server makes.
- `vs.is_dual_scene` — replaces `vs.raycast_scene is not None` mode checks.

Migrated all callers: samples (quickstart / slope_hold / terrain_drive cameras →
`vs.add_camera`; terrain_drive's visual terrain copy → `vs.add_dynamic`;
two_scene_terrain faces → `vs.rigid_solver`; obstacles_and_ramp →
`vs.is_dual_scene`), `_hud.native_alive` → `vs.viewer`, and the OSC server
(l3_runtime / physics_server scene-alias reads → `vs.rigid_solver` / `vs.step()` /
`vs.sim_options`; `apply_monkey_patches(scene)` → `apply_monkey_patches(rigid_solver)`).
**No raw scene access remains outside `vehicle_scene.py`.** 96 pytest; server
imports clean; `two_scene_terrain --compare` |Δx| = 0.000 m; obstacles_and_ramp
prints `is_dual_scene`.

---

## [0.9.35] — 2026-06-26

### Changed — `city_traffic_ego` sample on VehicleScene — phase 3.3 COMPLETE (8/8)

Migrated `city_traffic_ego.py` (ego + 7 traffic, lane-keeping P-control, L2 kinds
× L3 envs) onto `VehicleScene`: `view=` display mode; lane markers as visual-only
dynamics; per-step lane-keeper steer via `veh.set_inputs` (recomputed from each
vehicle's pose); the 4 kinds (ego AWD + RWD×3 + FWD×3 + Truck) auto-group;
`vs.physics.n_kinds` for the L2 report; the `_vs` shim is gone. Headless-verified
n_envs=4: 4 kinds [1, 3, 3, 1], all vehicles hold their lanes (Δy < 0.03 m).

**Phase 3.3 complete** — all 8 low-level samples (`l2l3_minimal`,
`perf_vectorization`, `perf_l2_l3_combined`, `perf_multi_vehicle`,
`batched_rollout`, `multi_env_render`, `road_loop`, `city_traffic_ego`) now run on
`VehicleScene`. Every sample + the server uses the high-level API; none touch
`gs.Scene` / `VehiclePhysics` / `MultiVehiclePhysics` directly. Remaining for
1.0.0: phase 4 (remove the `main_scene` / `raycast_scene` properties) + phase 5
(release).

---

## [0.9.34] — 2026-06-26

### Changed — `road_loop` sample on VehicleScene (1.0.0 phase 3.3, 7/8)

Migrated `road_loop.py` (the 16-vehicle / 4-kind loop demo with cv2 / native HUD +
solver comparison) onto `VehicleScene`: `view=` maps the display mode; `--solver`
maps to `VehicleScene(solver=…)`; the 4 kinds (FWD/RWD/AWD/Truck, one shared cfg
each) auto-group into 4 batched kinds; the loop markers register as **visual-only
dynamics** (`add_dynamic(physics=False, wheel_raycast=False)`) so the wheel
raycaster never treats them as ground; the loop camera is `vs.add_camera()`;
per-vehicle Ackermann steer via `veh.set_inputs`; the `_vs` shim is gone. The URDF
generators (`_car_urdf` / `_truck_urdf` / `_save_urdf`, imported by `l2l3_minimal`
+ `perf_multi_vehicle`) are unchanged. Headless-verified both solvers:
multi_batched → 4 kinds [4, 4, 4, 4], all 16 vehicles track radius ~25 m;
per_vehicle runs clean.

---

## [0.9.33] — 2026-06-26

### Changed — `multi_env_render` sample on VehicleScene; `build(center_envs_at_origin=)` (1.0.0 phase 3.3, 6/8)

Migrated `multi_env_render.py` (the n_envs grid-render demo) onto `VehicleScene`
(same pattern as batched_rollout: `view=`, `vs.add_camera()`, grid `vs.build()`,
`veh.set_inputs`, `native_alive(vs)`). Also added `center_envs_at_origin=` to
`VehicleScene.build` (forwarded to `Scene.build`) so grid-render samples can
center the L3 grid under the overhead camera. Headless-verified: n_envs=16 → envs
spread (pos x range 1.49 m), 605 env-steps/s.

---

## [0.9.32] — 2026-06-26

### Changed — `batched_rollout` sample on VehicleScene (1.0.0 phase 3.3, 5/8)

Migrated `batched_rollout.py` (the L3 per-env random-control rollout with cv2 /
native HUD) onto `VehicleScene`: `view=` maps `--native`→`"native"` /
`--viewer`→`"cv2"`; the cv2 HUD camera is now `vs.add_camera()` (SDK `Camera`
handle — `_hud` uses `cam.render()`); the grid build uses `vs.build(env_spacing=,
n_envs_per_row=)`; per-env controls via `veh.set_inputs((N,) tensor)`; the
`SimpleNamespace` `_vs` shim is gone (`native_alive(vs)` reads `vs.main_scene`).
VJS auto-managed. Headless-verified: n_envs=16 → (16, 3) state, 662 env-steps/s,
envs spread as expected. First of the 4 HUD-heavy visual samples.

---

## [0.9.31] — 2026-06-26

### Changed — `perf_multi_vehicle` sample on VehicleScene; Truck NaN fixed (1.0.0 phase 3.3, 4/8)

Migrated `perf_multi_vehicle.py` (the `per_vehicle` vs batched solver benchmark)
onto `VehicleScene`: the `--solver` flag now maps to
`VehicleScene(solver="per_vehicle" | "batched")`, so the benchmark compares the
two solvers **through the public API**. 4 vehicle kinds (FWD/RWD/AWD/Truck), one
shared cfg per kind → auto-grouped into 4 batched kinds. Also bumped `substeps`
10→30: the 6-wheel Truck NaN'd the rigid solver at `substeps=10` in the
`per_vehicle` path (**pre-existing** — the original `gs.Scene` benchmark crashed
identically; same fix road_loop took at 0.9.21), so **both** solvers now run.
Verified n_per_kind=4 (16 vehicles): per_vehicle 223 ms/step vs batched 125
ms/step (**1.78×** — the L2 batching win).

---

## [0.9.30] — 2026-06-26

### Changed — `perf_l2_l3_combined` sample on VehicleScene (1.0.0 phase 3.3, 3/8)

Migrated `perf_l2_l3_combined.py` (the combined L2 × L3 benchmark) from
`gs.Scene` + `make_wheel_raycaster` + `MultiVehiclePhysics` to `VehicleScene`
(default `solver="batched"`): K `add_vehicle(preset=…)` (one batched kind) +
`n_envs=N` + per-(vehicle, env) `veh.set_inputs((N,) tensor)` + `vs.step()`.
Verified `--internal --K 4 --N 16` → 64 vehicles, 2,078 vehicle-steps/s.

---

## [0.9.29] — 2026-06-26

### Changed — `perf_vectorization` sample on VehicleScene (1.0.0 phase 3.3, 2/8)

Migrated `perf_vectorization.py` (the n_envs L3-scaling benchmark) from
`gs.Scene` + `VehiclePhysics` to `VehicleScene` (default `solver="batched"`,
single vehicle = batched kind-of-1). Per-env random `veh.set_inputs((n_envs,)
tensor)` + `vs.step()`; now measures the default VE path's L3 scaling. Verified
n_envs 1→256: 43 → 8,903 env-steps/s (~207× batching throughput).

---

## [0.9.28] — 2026-06-26

### Changed — `l2l3_minimal` sample on VehicleScene (1.0.0 phase 3.3, 1/8)

Migrated `l2l3_minimal.py` from the low-level `MultiVehiclePhysics` to
`VehicleScene` (default `solver="batched"`): `add_ground_plane` + K
`add_vehicle(preset=…)` (auto-grouped into 1 batched kind) + `n_envs=N` (L3) +
per-(scenario, vehicle) `veh.set_inputs((N,) tensor)` + `vs.step()`. Same
behavior — K=4 × N=8 = 32 vehicles, 1 kind, scenario-0 lead brake diverges from
the rolling copies. First of the 8 low-level samples to move onto VehicleScene.

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
  30 is stable; the standalone truck demo uses 50 for a single
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
/ tank presets disable the wheel spin visual (cylindrical
road wheels — spin is invisible), and the closed-form previously baked spin
into the wheel quat anyway, disagreeing with the viewer. It now omits spin when
disabled, matching `VisualJointSync`. Verified against `get_link` (VisualJointSync
on) across all bundled vehicle classes: 4-wheel car ≈ 3.5 mm / 0.04°, 6-wheel
truck (`truck_6w_partial_ackermann`, front-axle steer) ≈ 3.5 mm / 0.04°,
10-wheel skid-steer tank preset ≈ 0.6 mm / 0°. Trucks already
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

Dropped the legacy-variant → SDK migration guide (no longer needed).
Removed its links from docs/index.md and README.md; the one cross-reference
in physics-contracts.md (legacy steer-sign flip) is now stated inline.

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
12.43 m baseline); 62/62 unit tests pass; the other legacy scenarios
unchanged (the reference car slightly slower — removes the spurious overshoot thrust, may
want per-vehicle torque re-tune). Lets the wheel mass / inertia "band-aid"
(inflated `i_wheel`) be reverted to realistic values without re-introducing
the oscillation.

| abbr | meaning |
|---|---|
| F_long | longitudinal tire friction force |
| omega_target | rolling-condition wheel angular velocity (= v_long/R) |
| C_kappa | longitudinal slip stiffness ∂F_long/∂kappa |

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

| abbr | meaning |
|---|---|
| P / C | Parent link / Child link |
| W | World frame |
| `q_WC` | world→child-link quaternion (wxyz) |
| `R_WP` | world→parent-link rotation matrix |

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

### Demo behavior — the 6-wheel truck demo

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
one external-author URDF) keep working — the recommendation is only for new URDFs.

### Fixed in repo
- the 6-wheel truck URDF — steer joint axes changed from
  `(0, 0, 1)` to `(0, 0, -1)` to match the recommendation. Visual behavior
  was already correct (v0.5.3 fix), but the URDF now follows the
  recommended convention internally.

---

## [0.5.3] — 2026-05-18

### Fixed — steer wheel visual rotated the wrong direction

`VisualSync` was driving the steer joints **opposite to the physics-side
steering direction** for both URDF axis conventions:

- URDF axis `(0, 0, 1)` (e.g. truck preset, one external URDF) — visual wheels rotated LEFT when physics rotated RIGHT, and vice versa.
- URDF axis `(0, 0, -1)` (the reference car URDF) — same inversion.

The old formula `visual_cmd = phys * sign` assumed the only conversion
needed was the URDF axis flip captured in `sign`. It missed that the
physics-side `steer_per_wheel` is **opposite-handed** from the URDF joint
convention to begin with (physics +θ = right turn = CW from above; URDF
axis `(0,0,1)` +joint = CCW from above). The correct formula is
`visual_cmd = -phys * sign`:

- axis `(0, 0, 1)`: `sign=+1` → `visual_cmd = -phys` → joint goes CW for +phys → right turn visual ✓
- axis `(0, 0, -1)`: `sign=-1` → `visual_cmd = +phys` → joint goes CW for +phys → right turn visual ✓

Demos affected: the 4-wheel car (axis -1) and the truck (axis +1). The
4-wheel car demo never had the wheel direction verified visually; the
truck demo just hit it (front wheels turned left when the truck arced
right). The tank (skid-steer, no steer joints) and the external-author demo are unaffected.

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

Combined: any preset vehicle (`car_4w_rwd_ackermann`, the tank preset,
`truck_6w_partial_ackermann`, …) issuing `throttle > 0` at rest would
spin its wheels for one step and then have `omega` snapped back to 0.
**Vehicle stuck at rest indefinitely.** Discovered while running the
6-wheel truck demo where the truck wouldn't accelerate.

### Reverted — `disable_when_control_active=True` is back in `"control"` profile

This restores the original reference-car behavior: the regularizer is off when the
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

Measured on the 10-wheel tank interactive demo:
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
- **Tank chase-cam fps fix** — interactive demo on the reference tank ran at ~15 fps vs ~25 fps for the original legacy script. Two contributions:
  - `VisualSync` was syncing wheel spin angle (`set_dofs_position` for the 10 continuous spin joints) every step. The original legacy variant intentionally skipped this since the cylinder primitive wheels are rotationally symmetric and a spinning visual is invisible.
  - The chassis `up_world` reference tensor was being re-allocated every step inside `VehiclePhysics.step()` instead of being cached at init time.

### Added
- `VehicleConfig.visual_spin_enabled: bool = True` (also on `ResolvedConfig`). Set `False` to skip the per-step spin-angle `set_dofs_position` call. Saves ~3-5 ms / step in interactive mode (one fewer Genesis call). Cars keep the default `True` so mesh wheels visibly roll; tanks (the tank preset) now default to `False`.

### Changed
- `VehiclePhysics.__init__` caches `_up_world` once instead of re-creating it each `step()`.
- `VisualSync` integrates the visual spin angle in place (`add_`) when enabled.

---

## [0.4.3] — 2026-05-18

### Fixed
- **Heavy-wheel suspension visual fix (two-part)** — on the 10-wheel reference tank (each wheel ≈ 500 kg):
  - *Part 1 (free-fall)*: the wheel meshes were sinking below the ground while the chassis stayed up. `VisualSync` chose `set_dofs_position` vs `control_dofs_position` by detecting `<dynamics>` declarations in the URDF, which that tank's URDF omits entirely. With no PD control (`kp`/`kv` = 0), Genesis lets the heavy wheel free-fall between substeps and the kinematic `set_dofs_position` cannot snap it back fast enough.
  - *Part 2 (spinning in air)*: after the PD fix above, wheels stopped falling but were spinning in air ~5 cm above ground. The `control_dofs_position` path used `target = max(0, rest_d - d)` (a non-negative compression), which can only push the wheel UP from rest, never DOWN to reach the ground. This is fine when the carrier rest position sits *above* ground level (reference car URDF: susp origin z = 0.34), but breaks when the carrier rest sits *at* base_link z (reference tank URDF: susp origin z = 0). The `control_dofs_position` path now uses the same `joint_pos = mesh_radius - d` formula as the `set_dofs_position` path so the wheel mesh lands on the ground in both cases.

### Added
- `VehicleConfig.visual_susp_mode: str = "auto"` (also on `ResolvedConfig`). Three values:
  - `"auto"` (default) — per-joint decision based on URDF `<dynamics>` presence; preserves the v0.4.2 behavior for light-wheel URDFs.
  - `"kinematic"` — force `set_dofs_position` everywhere (light wheels).
  - `"control"` — force `control_dofs_position` with PD kp=1e7 / kv=1e5 everywhere (heavy wheels).
  - `resolve()` validates the value and raises `ConfigError` on unknown strings.
- The tank preset now sets `visual_susp_mode="control"` so the wheels stay attached on the heavy-wheel reference tank without the user having to know.

### Migration
- The car presets are unchanged — `"auto"` keeps their existing behavior.
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
  - `"control"` — RL/MPPI-friendly hooks: `RollingResistance` + `LowSpeedRegularizer(disable_when_control_active=False)` (+ `StaticFrictionLock` for tank). Default for `car_4w_rwd_ackermann`, `car_4w_awd_ackermann`, `truck_6w_partial_ackermann` and the tank preset.
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
- `parse_urdf()` convention-based parser supporting both `*_suspension_joint` and `*_susp` naming.
- Visual layer (`VisualSync`) absorbing URDF axis-sign quirks.
- Presets for four reference vehicles: `car_4w_rwd_ackermann`, `car_4w_awd_ackermann`, `truck_6w_partial_ackermann` and the tank preset.
- Pure-Python tests for URDF parsing, config resolve, strategy math.
