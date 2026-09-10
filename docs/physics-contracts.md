# Physics contracts

The SDK's promises about ambiguous physical conventions. These contracts are
enforced by `genesis_vehicle.dynamics` (pure-Python, unit-tested in
`tests/test_dynamics.py`) and consumed by `core.py`.

| abbr | meaning |
|---|---|
| `N` | suspension normal force (N) |
| `kappa` | longitudinal slip ratio |
| `rest_d` | suspension rest length; `compression = max(rest_d - distance, 0)` |
| `no_hit_value` | Genesis `Raycaster` option: the distance a ray reports when it hit nothing (defaults to `max_range`) |
| `max_range` | the raycaster's maximum ray length (m); `raycaster_max_range`, default 20.0 |
| RAW / CORRECTED distance | straight off `sensor.read().distances` / after `read_distances` subtracts the high-cast offset |
| `n_envs` / L3 | the parallel-scenario batching axis |
| `dual_scene` / `single_scene` | the two `raycast_mode` values: separate static-BVH raycast scene / one shared scene |
| BVH | Bounding Volume Hierarchy (the acceleration structure rays are cast against) |
| AABB | Axis-Aligned Bounding Box |
| VJS | `WheelJointInternalSync` — the legacy wheel-visual writer that drives solver joints |
| PD | proportional-derivative controller (the VJS `control_dofs_position` path) |
| RL | reinforcement learning (the partial-reset rollout pattern) |
| ISO 8855 | the vehicle-axis standard this SDK follows: `+X` forward, `+Y` left, `+Z` up |
| M | `contact_samples` — rays per wheel in a swept-envelope fan; 1 is the point contact |
| `s_j` / `c_j` | sample `j`'s longitudinal offset from the wheel centre / its swept-circle height penalty `r - sqrt(r^2 - s_j^2)` |
| `d_eff` | the one distance per wheel a fan reduces to, `min_j(d_j + c_j)` |
| `r` / `h` | wheel radius / obstacle (step) height, metres |

## 7.1 Brake torque is a positive command magnitude

`brake` (user input) is always in `[0, 1]`. Internally, the SDK converts it
to a signed torque opposing wheel rotation:

```
T_brake_eff = T_brake * tanh(omega / smoothing_scale)
domega      = (T_drive - T_brake_eff - R * F_long) / I_wheel
```

- For `omega > 0`, `T_brake_eff > 0`, so `-T_brake_eff` decelerates the wheel.
- For `omega < 0` (reverse spin), `T_brake_eff < 0`, so `-T_brake_eff > 0`
  again decelerates.
- For `omega ≈ 0`, `T_brake_eff ≈ 0` — the smooth brake cannot pin the
  wheel. Pair with `StaticFrictionLock` for hard hold-at-rest behaviour.

Implementation: `genesis_vehicle.dynamics.brake_torque_signed`.

## 7.2 Normal force is non-negative; air-mask wheels contribute nothing

Per-wheel suspension force uses the asymmetric damper (different coefficient
on compression vs extension) and is clamped non-negative — the ground cannot
pull a wheel down:

```
c_damp = c_compression if c_dot > 0 else c_extension
N_raw  = K_susp * compression + c_damp * c_dot
N      = max(N_raw, 0)
N      = 0 if the ray missed the ground (air_mask)
```

When `N = 0`, the per-wheel `F_long`, `F_lat` are also zero (no contact),
though `T_drive` and `T_brake` still update `omega` (the wheel spins freely
in the air).

Implementation: `genesis_vehicle.dynamics.suspension_normal_force`.

## 7.3 `WheelConfig.i_wheel` truth policy

```
1. WheelConfig.i_wheel set by the user            -> AUTHORITATIVE (used as-is)
2. URDF inertia (via parse_urdf, projected on the spin axis) -> default
3. Genesis-runtime metadata (projected on +Y)     -> fallback estimate
4. DEFAULT_I_WHEEL                                -> last-resort fallback
```

`i_wheel` is the moment about the **spin axis**, `a^T (R I R^T) a` — not the
largest principal moment. Since v1.2.8 both estimating paths project onto that
axis (`<inertial rpy>` honoured). Before then they took `max(ixx, iyy, izz)`,
which is the spin moment only for a disc-like wheel (length < √3·radius); a
wide, small-diameter road wheel got roughly **2× its true spin inertia** and
accelerated half as fast as its URDF describes.

For ray-wheel dynamics, the wheel spin inertia is often different from the
URDF hinge inertia (e.g. URDF wheel hinge is visual-only while real
ray-wheel inertia comes from a coarser estimate). In Real2Sim / parameter
fitting, **always set `WheelConfig.i_wheel` explicitly** to take this out of
the estimation pipeline.

## 7.4 Steering sign convention (ISO 8855)

`+steer` is right turn under all strategies (`Ackermann`,
`PartialAckermann`, `SkidSteer`). Unit-tested:

- **Ackermann right turn**: both front wheels turn positive; the right wheel
  (the inner wheel for a right turn) has the larger angle.
- **SkidSteer right turn**: the left side commands more torque than the
  right side (`left_cmd > right_cmd` via
  `left_cmd = throttle + steer_gain * steer`).

The legacy `+steer = LEFT` convention is **not** carried forward; skid-steer here is
`+steer = right` (ISO 8855), so any code ported from a `+steer = LEFT`
convention must flip the steer sign.

### URDF authoring recommendation — steer joint axis = `(0, 0, -1)`

For new URDFs, declare steer joint axes as `(0, 0, -1)` so the joint angle
follows the same sign convention as the user-facing `+steer = right`:

```xml
<joint name="front_left_steer_joint" type="revolute">
  ...
  <axis xyz="0 0 -1"/>   <!-- ISO 8855: +angle = CW from above = RIGHT turn -->
  ...
</joint>
```

Vehicle frame z is up, so +joint_angle around `(0, 0, -1)` is CW viewed
from above = right turn — matches `+steer`. Using `(0, 0, 1)` instead makes
+joint_angle = CCW = left turn, which is **opposite-handed** to the SDK's
user-facing convention.

`WheelJointInternalSync` does compensate either way (`visual_cmd = -phys * sign`), so
existing URDFs with `(0, 0, 1)` still render correctly. The recommendation
is only for NEW URDFs: declaring `(0, 0, -1)` keeps URDF joint values and
user-facing steer values in the same sign domain, which makes URDF-side
inspection / debugging less surprising.

Examples surveyed:
- the reference car URDF — `(0, 0, -1)` ✓ (matches recommendation)
- the 6-wheel truck URDF — `(0, 0, -1)` ✓ (fixed in v0.5.4)
- one externally authored URDF — `(0, 0, 1)` (SDK handles via WheelJointInternalSync sign flip)

## 7.5 Coupling order

`CouplingStrategy.apply(omega)` runs after the per-wheel omega integration
in the current step and before the next step. Drive torque distribution in
the same step uses **pre-coupling** omegas (one-step lag), matching the
reference implementation. Strategies must not assume they run inside the
per-wheel loop.

## 7.6 First-step protection

The wheel raycaster is not populated until the first `scene.step()`. To
avoid a NaN cascade, `VehiclePhysics.step()` skips force application on the
first call when all distances are zero, sets `_prev_init = True`, and runs
normally from the second step onward.

### An env that has not completed a step renders at the REST pose (v1.5.2)

`_stepped_once` is **per env** — an `(n_envs,)` bool tensor, cleared per index
by `reset(env_ids=...)`. Every closed-form visual path
(`VehiclePhysics.wheel_visual_transforms`,
`MultiVehicleKindPhysics.wheel_visual_transforms`) masks a row whose flag is
False back to the rest pose, and `wheels_grounded` reports all-False for it.

The contract is that such an env renders at REST, not in an air pose, and the
distinction is deliberate:

- `_stepped_once == False` means **"no distance data"**, not "the ray missed".
  An air pose would assert a MISS that was never measured. Rest is the only
  pose that asserts nothing.
- `reset` already zeroes `wheel_spin_angle` and `last_steer_per_wheel`, so the
  wheel ORIENTATION is already the rest orientation. Giving it an air POSITION
  would produce a hybrid — rest yaw and spin, extended suspension — that
  corresponds to no state the vehicle was ever in.

Before v1.5.2 the flag was a single Python `bool`, so a partial reset left it
`True` and those envs' freshly-zeroed `last_distances` were fed to
`_susp_visual_offset` as a CORRECTED `0.0` — a legitimate contact at exactly
`RAY_UP_OFFSET`, i.e. **maximum compression**. The reset wheels rendered
jammed into the chassis and `wheels_grounded` (had it existed) would have
called them grounded.

Known gap: `MultiVehicleKindPhysics.wheel_visual_reads` /
`wheel_visual_transforms_host` — the split device-read / host-compute pair —
still gate batch-wide and carry no per-row mask, so for one frame after a
partial reset they render those wheels fully compressed (measured device
`0.3000` vs host `0.4000`). The shipped OSC server cannot reach it (its L3
reset restores the pose with `set_pos`/`set_quat` and never calls
`physics.reset()`, `server/l3_runtime.py:539-542`); user-written RL and host
render harnesses can.

## 7.7 Longitudinal friction-force overshoot clamp (v0.6.0)

The tire-friction analogue of §7.1. Explicit-Euler integration of the
slip-dependent friction torque `T_fric = R·F_long` is stiff near rolling
(its relaxation rate `R²·C_kappa/(I·|v_long|) → ∞` as `v_long → 0`); below
the stability limit the wheel oscillates across the rolling point
(forward force → reverse slip → backward force → …), seen as wheel
"trembling" and a stuck `kappa ≈ −1` drag on undriven wheels at launch.

`VehiclePhysics.step()` caps `F_long` so the friction torque cannot carry
the wheel **past the rolling speed** `omega_target = v_long/R` in one step:

```
omega_nofric = omega + DT·(T_drive - T_brake_eff)/I_wheel
F_long_limit = (omega_nofric - omega_target)·I_wheel / (DT·R)
omega_nofric > omega_target → F_long ∈ [0, F_long_limit]
omega_nofric < omega_target → F_long ∈ [F_long_limit, 0]
```

The clamp binds **only near rolling** (small `omega_nofric − omega_target`),
so it removes the oscillation while leaving the high-slip saturated regime —
driven-wheel launch slip — untouched (`quickstart` launch is preserved). The
**clamped** `F_long` is what is applied to both the wheel-ω update and the
chassis force, so a custom `TireModel` or parameter-fit sees a force that may
be reduced from its raw output near rolling. Implementation: inline in
`core.py` step (D); cf. `brake_torque_signed` (§7.1).

## 7.8 High-cast rays and over-compression (v1.1.16)

Wheel rays start `RAY_UP_OFFSET` (default 1.0 m) ABOVE the wheel
attachment point and the read layer (`read_distances`) subtracts the
offset from hits, so every consumer still sees attachment-relative
distances. In `single_scene` the offset is capped per vehicle (v1.5.0, see
below); `read_distances` takes the offset off the sensor, so it always
subtracts back exactly what the pattern added. Contract points:

- A ray MISS keeps its sentinel value; the offset is NOT subtracted from
  misses. **The sentinel is the sensor's own `no_hit_value`** (a Genesis
  `Raycaster` option, which `model_post_init` fills with `max_range` when left
  unset), read through `raycast.ray_miss_value` — it is NOT a fixed number, and
  the correction that follows is:

  > **Correction (v1.5.2).** This bullet used to read "keeps its sentinel value
  > (>= 19.9)". That was **wrong** for every raycaster not built at the default
  > `raycaster_max_range=20.0`: `19.9` is that default less a margin, so at
  > `max_range=10.0` a miss (10.0) fell UNDER the threshold, was taken for a
  > hit, and `read_distances` returned 9.0 — the high-cast offset subtracted
  > from a distance that measured nothing. Three separate copies of the
  > threshold (`raycast.py`, `core._susp_visual_offset`,
  > `visual._susp_visual_target`) encoded the same false claim.

- The MISS test is an **equality**, `d_raw == miss`, evaluated on the **RAW**
  distance before the offset is taken off, plus a `d_raw != 0.0` term. Both
  halves are load-bearing:
  - Equality is exact because Genesis writes the sentinel verbatim and its own
    test is likewise an equality (`distances != no_hit_value`); a value with no
    float32 representation (19.9, 0.1) still round-trips.
  - RAW, because with `no_hit_value = 0.0` a genuine hit at exactly the offset
    corrects to `0.0` and would collide with the sentinel.
  - `d_raw != 0.0` rejects the UNPOPULATED buffer (Genesis allocates it as
    zeros and only fills it inside `scene.step()`), which is otherwise
    indistinguishable from four rays hitting ground exactly `up_offset` above
    the attachment points. It is only meaningful on RAW values.

  The two predicates are public: `is_ray_hit(d_raw, miss)` for raw distances
  and `is_ray_hit_corrected(d, miss)` (`d != miss`, no zero term) for corrected
  ones — everything in `last_distances`. Carry the mask
  `read_distances(..., return_hit=True)` produces rather than re-deriving one
  downstream.

- **A CORRECTED distance of `0.0` is a HIT, not air**: ground exactly
  `RAY_UP_OFFSET` below the ray origin, i.e. exactly at the wheel attachment
  point — full compression. `is_ray_hit_corrected` deliberately omits the
  `!= 0.0` term for this reason. Both visual mirrors carried the term until
  v1.5.2 and rendered that contact as fully-extended air; three tests encoded
  the same mistake for several releases because the contract above never said
  what a corrected `0.0` meant.

- **`no_hit_value` below `max_range` is UNSUPPORTED and raises.** A sentinel
  inside the measurable range is indistinguishable from a very close hit, and
  the pipeline reads `compression = max(rest_d - distance, 0)` — with
  `no_hit_value = 0.0` that is maximum compression and the vehicle launches. A
  non-finite or non-positive sentinel raises too (`nan < max_range` is False,
  so `nan` used to pass the range test and then made the equality never fire).
  `raycast.check_miss_supported` enforces both, and is called wherever the
  sentinel is OBTAINED — the stamp (`set_sensor_miss_value`), the read
  (`ray_miss_value`, which `VehiclePhysics.__init__` goes through) and the
  `dual_scene` injection (`VehiclePhysics.set_ray_miss_value`) — not only where
  the SDK happens to stamp it. A hand-built `gs.sensors.Raycaster` passed
  straight to `VehiclePhysics` is never stamped, so a stamp-only guard would
  have covered nothing but the SDK's own path.
- A hit may therefore report a **negative** distance: the ground is above
  the attachment point (the chassis has sunk past its wheels). That is a
  VALID reading — compression maxes out and `N` pushes the vehicle back
  up. Do not "sanitize" negative distances back to air.
- Why: with origins AT the attachment points, a hard landing could sink
  the origins below the ground; the rays then missed, the air mask killed
  `N`, and the vehicle rested on its chassis collision box forever (a
  stable buried equilibrium — the v1.1.16 field report).
- **The RAW read is `(n_envs, n_wheels)` — EXCEPT with a swept-envelope fan,
  where it is rank 3.** Genesis returns `(n_wheels,)` at `n_envs == 1` and
  `(n_envs, n_wheels)` otherwise; with `wheel_contact="swept_envelope"` (§7.11)
  it returns `(n_wheels, M)` and `(n_envs, n_wheels, M)`, **M last**.

  > **Correction (v1.6.0).** This doc and `docs/index.md` used to describe the
  > raw read as `(n_envs, N_WHEELS)` with no qualification. That is now
  > incomplete: it is the M=1 case. What holds unconditionally is the
  > post-`read_distances` shape below — state that, not the raw one.

  `read_distances` normalises BOTH the `n_envs == 1` squeeze and the M axis, so
  **every downstream consumer sees `(n_envs, n_wheels)` in every mode**:
  `last_distances`, the pipeline, `wheels_grounded`, `distances_list()` and both
  visual mirrors. Use `read_distances`; do not index the raw buffer.

- **`single_scene` CAN self-hit, and the offset is capped so it does not**
  (v1.5.0). A vehicle is never a *visual* raycast target
  (`use_visual_raycasting` defaults to False), which is why `dual_scene` —
  whose raycast scene holds no vehicle collision geometry at all — is safe at
  the full offset. But a raycaster also casts against the rigid solver's
  COLLISION BVH, and in `single_scene` that BVH contains the vehicle's own
  chassis box: from v1.1.16 to v1.5.0 the elevated origin sat above the
  reference car's roof and every ray self-hit, reporting -0.80 m and launching
  the vehicle. `raycast.single_scene_up_offset` now caps the offset at the
  vehicle's own collision ceiling (`urdf.self_collision_ceiling`, from the
  URDF's rest-pose collision AABBs) — 0.28 m for the reference car. A ray with
  nothing of its own above it keeps the full 1.0 m.
- The capped offset buys proportionally less over-compression headroom, so a
  `single_scene` vehicle recovers from a deep bottom-out less readily than a
  `dual_scene` one. `dual_scene` is the default for this reason among others.
- Keep overhead raycast terrain (tunnel ceilings) more than the vehicle's
  effective offset above the wheel attachment points.

Related: `WheelJointInternalSync` is intended to be cosmetic but is not
perfectly physics-neutral (the control path's PD applies real joint
forces; the set path teleports wheel mass). Its suspension targets are
stroke-clamped and slew-rate-limited (`_SUSP_VIS_MAX_RATE`), bounding
the measured disturbance to < 1 cm extra compression on hard landings
(was 2-3 cm unclamped — enough to flip a marginal landing into the
buried state before the high-cast fix removed that failure mode).

Since v1.1.17 `VehicleScene` no longer uses VJS by default: rendered
scenes with `n_envs == 1` draw wheels via `InstancedWheelRenderer` —
closed-form poses streamed into instanced render nodes (Genesis's
external render-node channel — the debug-draw machinery — NOT the rigid
solver), verified physics-identical to headless to 1e-6 m at a slight
pose-streaming cost (~2–3 ms/step at 30 vehicles, CPU)
(`wheel_render_mode="internal_sync"` restores the old behavior; multi-env
and raw-`VehiclePhysics` use still fall back to VJS).

Since v1.1.25 the native viewer additionally commits the wheel buffers, the
`follow_entity` camera and the rigid node poses in ONE render-lock hold
(`visual.patch_viewer_atomic_update`, applied automatically at `build()`).
Before that, the async draw thread could pair state from adjacent steps and
a followed vehicle appeared to "tremble" fore/aft at speed — a pure draw
artifact; physics was never affected (see CHANGELOG 1.1.25).


## 7.9 URDF contracts for ray-wheels (auto-corrected since v1.1.22)

Ray-wheel physics makes three demands on a vehicle URDF. Every
SDK-authored vehicle satisfies them; an arbitrary URDF (e.g. one exported
for a normal rigid-body sim) usually does not, so
`VehicleScene.add_vehicle` ALWAYS runs `genesis_vehicle.urdf_prep` on
it (not a knob — the prepared file feeds the entity, the parse AND the
ray pattern, which must agree), writing a corrected temp copy next to the
original — the original file is never modified, and a URDF that already
complies is used as-is (no copy).

| contract | why | auto-correction |
|---|---|---|
| **Wheels must not collide** | ground contact IS the raycast + suspension force model; a colliding wheel is a SECOND support that fights it (vehicle jitters in place, or rides on its colliders while the suspension pushes several times its weight) | wheel colliders are removed. A collider that is the wheel's ONLY geometry is first promoted to a `<visual>`, so the wheel still RENDERS — the instanced wheel renderer draws visual geoms, and physics never touches them |
| **The suspension attach point IS the wheel centre** | the ray is cast down from `WheelConfig.position` (= the prismatic joint origin) and `rest_d = radius + rest_stroke` measures from there | a chain that hangs the wheel off a carrier (`body --susp--> carrier --spin(z=+h)--> wheel`) puts the attach `h` below the wheel centre — the hull then settles `h` too high with the wheels visibly floating (measured on a 14-wheel tracked-vehicle model: h = 0.433 m). The spin-joint offset is folded into the suspension origin, leaving every link's rest pose unchanged |
| **Moving links need a valid inertial** | links with no `<inertial>` have zero mass AND inertia; Genesis warns ("mass and inertia of moving bodies must be larger than mjMINVAL"), falls back to its legacy URDF parser, and the articulated chain goes degenerate — the hull stops responding to force properly | a small inertial is injected where one is missing |

The OSC server builds its own morph, so it prepares the URDF itself and
hands the SAME prepared path to `build_cfg()` and `add_vehicle()` (fixed in
v1.1.24 — before that the server passed the original path and its vehicles
floated). If you supply `morph=...` to `add_vehicle` yourself, prepare the
file first and build the morph from the prepared path; `add_vehicle` warns
when the two disagree.

**These three are not equally severe, and the SDK reports them differently.**
Contracts 1 and 2 are *convention gaps*, not URDF defects: a wheel collider
is mandatory in a normal rigid-body sim, and a prismatic joint's origin may
sit anywhere along its own axis (a gauge freedom — the child chain
compensates, so the kinematics are identical). Such a file is perfectly
valid; it just does not match what the ray-wheel model reads out of it. Both
corrections are therefore reported as an informational line. A link with no
`<inertial>`, by contrast, is a **genuine defect in any engine**, and the fix
injects a placeholder mass the author never chose — so it raises a
`logging.WARNING` naming the offending links. Fix that one at the source.

Wheel RENDERING is independent of all of this: the instanced renderer
harvests the wheel link's visual geoms (falling back to its colliders, and
finally to a cylinder synthesized from `radius`) and draws them at the
closed-form ray-wheel pose. The wheel link's own solver pose is never used
— its visuals are hidden and its body only contributes mass.

## 7.10 `wheels_grounded` is a read-layer diagnostic, not the air mask (v1.5.2)

`Vehicle.wheels_grounded` / `VehiclePhysics.wheels_grounded` /
`MultiVehiclePhysics.grounded_list()` answer **"did this wheel's ray find
ground on the last step?"** — a question about the SENSOR. They are

```
last_hit & _stepped_once.unsqueeze(1)
```

where `last_hit` is the mask the read layer produced alongside
`last_distances` (§7.8) and the per-env gate is §7.6's. The gate is not
decoration: on the injected `dual_scene` path only corrected distances exist,
and the corrected predicate cannot recognise the pre-first-step zero buffer by
itself.

**The physics air mask is a different quantity.** The pipeline decides air from
`compression <= 0` (§7.2) and never looks at the sentinel. The two agree
everywhere except inside the suspension's rest gap: a ray can hit ground that
is still farther away than `rest_d`, so `wheels_grounded` is True while the
wheel carries no normal force and contributes nothing. Use `wheels_grounded` to
answer "is the vehicle over terrain at all" (fall detection, spawn validation,
RL termination); use `last_N > 0` or `last_compression > 0` if what you mean is
"is this wheel loaded".

`all_wheels_grounded` is `wheels_grounded.all(dim=1)` — `(n_envs,)`, True where
EVERY wheel of that env found ground. It is the supported answer to "is this
vehicle actually on the terrain?", replacing the ad-hoc
`distances.max() < 19.9` that samples and user code were writing (and which
§7.8 shows was wrong at any non-default `max_range`).

## 7.11 Swept-envelope wheel contact is OPT-IN and can only RAISE the ground (v1.6.0)

`VehicleScene.add_vehicle(wheel_contact=...)` selects the wheel-ground contact
model. **`"point"` is the default and is the single-ray behaviour of every
earlier release, bit-for-bit** (`torch.equal` on a 200-step reference rollout,
v1.6.0 CHANGELOG). Nothing below applies unless a caller opts in.

**The defect `"swept_envelope"` addresses.** A single zero-radius downward ray
reads ground height as a STEP FUNCTION. A wheel at road level on one step and
0.13 m into a kerb on the next takes the whole obstacle height as compression in
one `dt`; the damper sees a rate no tire can produce and dominates the spring
(measured on the reference car at a 0.130 m lip at 3.3 m/s: `dcompression =
0.130` in one 0.025 s step, raw rate `5.200 m/s`, peak `N` `85,120 N`).

**The model.** M rays per wheel, uniformly spaced along **body +X** over
`[-r*fan_span, +r*fan_span]` using EACH wheel's own radius, each penalised by
the height its tire surface stands off the wheel's lowest point:

```
s_j   = (2j/(M-1) - 1) * r * fan_span        j = 0 .. M-1
c_j   = r - sqrt(r^2 - s_j^2)                c(|s| = r) = r exactly
d_eff = min_j (d_j + c_j)                    a MISSING sample goes to +inf
```

A wheel of radius `r` therefore begins to climb a step of height `h` about
`sqrt(2*r*h)` before it — where a real tire touches it.

Contract points:

- **`c_0 = 0` exactly, so `d_eff <= d_center` always.** The envelope can only
  RAISE the ground it reports, never lower it; a sinking regression is
  structurally impossible rather than merely untested. Verified over 2000
  randomised profiles including 1784 mixed hit/miss cases: zero violations.
- **M must be 1 or ODD.** An even fan has no centre sample, loses the guarantee
  above, and reads even FLAT ground biased (`r=0.35`: `-0.020017 m` at M=4,
  `-0.350000 m` at M=2, against exactly `0.000000` for any odd M). Even counts
  raise.
- **M=3 at the default `fan_span=1.0` IS the point contact.** Its two off-centre
  samples sit at `|s| = r`, where `c = r`, so for any obstacle shorter than the
  wheel radius they can never win the minimum (measured: identical peak `N`).
  That is correct behaviour, not a bug — M=3 is legal and useful at
  `fan_span < 1` (at `0.6` the same three rays see the lip). The default M is 9
  because it is the smallest count that does useful work at the default span.
- **M is a DISCRETISATION count, not an accuracy dial.** The peak force does not
  converge monotonically in M — it oscillates as which sample wins the minimum
  changes (ratio vs the point contact on the harness above: M=9 → 1.703,
  M=15 → 1.906, M=31 → 2.039, M=101 → 1.959, i.e. M=101 BELOW M=31).
  Convergence needs M around 101; across M ∈ [9, 31] the peak still swings by
  roughly ±20%. Pick M for cost, read the result as "no longer a step function",
  and **do not compare two runs at different M**. "Low phase spread" and
  "accurate" are separate claims; conflating them is how a bogus acceptance bar
  got written during this work.
- **The approximation errs late, never early.** M samples of the continuous
  envelope can miss a contact, never invent one — the discretisation is
  conservative in the safe direction. The model's real limitation is that the
  rays are **vertical** and therefore blind to terrain SLOPE, not the size of
  `c_j`. An intermediate `s_j` winning the minimum with a smaller penalty than
  the outermost sample is the model working.
- **The M axis never escapes `read_distances`** (§7.8). The per-wheel hit mask is
  the OR over the fan, and `d_eff` is clamped just under the miss sentinel so
  "the mask says hit" and "the value is not the sentinel" cannot disagree.
- **`single_scene`'s self-collision ceiling is measured against EVERY fan ray**,
  not the wheel centre, and takes the minimum gap (`raycast.fan_ray_positions`).
  An outer fan origin can sit under a low overhang the wheel centre clears, and
  a centre-only cap would leave that one ray starting inside the body — the
  v1.1.16 "the ray hits its own roof, reads maximum compression, the vehicle
  launches" failure, reproduced one sample at a time. For the reference car both
  caps give the same answer (its chassis AABB covers every fan origin), so the
  tests use a synthetic variant with a low sill outboard of the front wheels:
  centre-only `0.28 m`, M=9 fan `0.12 m`, and the centre-only value would put
  the outer ray origins at `z=0.58`, inside the sill at `z=0.44`.
- **One kind, one M.** Every vehicle of a `MultiVehicleKindPhysics` kind must
  share one fan size; `multi_vehicle.check_fan_uniformity` enforces it once at
  construction. Mixed sizes would either fail deep inside the stacked read or,
  when the wheel counts line up, silently mix two contact models in one batch.
- **`DifferentiablePlant` does not model the envelope.** It snapshots
  `distances` and holds them frozen over the horizon, so a `swept_envelope`
  vehicle's kerb crossing is still PREDICTED as a point contact. The simulated
  vehicle gets the envelope; the controller's internal model does not. See
  `docs/path-following.md`.

Nothing else in the pipeline changes shape or behaviour: the cost is M raycasts
per wheel per step where there was 1, plus one `min` reduction.
