# Authoring a URDF by vehicle type

How the SDK reads a vehicle URDF, what each preset's topology needs, and —
new in v1.2.1 — where suspension values come from when the URDF does not
carry them. For the three hard *ray-wheel contracts* (no wheel colliders,
suspension origin = wheel centre, every moving link inertial) and their
auto-correction, see [`physics-contracts.md` §7.9](physics-contracts.md#79-urdf-contracts-for-ray-wheels-auto-corrected-since-v1122);
this guide is about getting the vehicle *recognised and tuned*.

| abbr | meaning |
|---|---|
| k_susp | suspension spring rate (N/m); `N = k·compression + C·ċ` |
| c_comp / c_ext | compression / extension damping coefficient (N·s/m) |
| sprung mass | mass the springs carry: everything not below a suspension joint |
| sag | static suspension deflection under the vehicle's own weight |
| ζ | damping ratio, `c / c_critical` |
| axle | a cluster of wheels sharing an x-position; axle 0 = front-most |

## 1. How the parser recognises a vehicle

`parse_urdf()` walks the joint tree by **convention** — there is no metadata
tag that says "this is a wheel". Discovery rules:

| element | rule |
|---|---|
| **Wheel** | a `prismatic` joint whose name ends with `_susp` or `_suspension_joint`. Its child is the suspension carrier |
| **Spin joint** | the first descendant `continuous` joint with axis `(0, ±1, 0)`. Its child is the wheel link. (A wheel hanging *straight* off the suspension joint with no spin joint is also accepted since v1.2.1 — the suspension child is then the wheel) |
| **Steer joint** | a `revolute` joint with axis `(0, 0, ±1)` along the chain (optional; tracked vehicles have none) |
| **Side** | name contains `l_`/`_left_` → `L`; `r_`/`_right_` → `R`; else `None` |
| **Axle index** | wheels are clustered by x-coordinate (tol 0.05 m) and sorted front-to-back; axle 0 is front-most |

So the *joint names and axes* are load-bearing, not decorative. Keep the
`_susp` suffix, the spin axis on `±Y`, and the steer axis on `±Z`. Full
reference: [`api-reference.md` §4](api-reference.md).

## 2. By vehicle type

Each preset assumes a topology. Pick the preset that matches how many wheels
the URDF has and whether they steer.

| preset | wheels | steer joints | drivetrain | coupling |
|---|---|---|---|---|
| `car_4w_fwd_ackermann` | 4 | front axle | FWD | Independent |
| `car_4w_rwd_ackermann` | 4 | front axle | RWD | Independent |
| `car_4w_awd_ackermann` | 4 | front axle | AWD | Independent |
| `truck_6w_partial_ackermann` | 6 | front axle only | mid + rear axles | Independent |
| `tank_skid_belt` | **any** | **none** | PerSide | SameSideBelt |

Concretely, per type:

- **Car (4-wheel Ackermann).** Four `*_susp` prismatic joints; each front
  wheel needs a `revolute` steer joint with axis `(0, 0, ±1)`; the spin joint
  hangs below the steer link. Name wheels by position
  (`front_left_wheel`, …) so both the parser's side detection and
  position-based `wheelOverrides` (`"FL"`, `"front_left"`) work.

- **Truck (6-wheel partial Ackermann).** Six wheels in three axles; only axle
  0 carries steer joints. Axle clustering by x decides which is "front", so
  keep the three axles clearly separated in x.

- **Tank / tracked (skid-steer, any wheel count).** No steer joints at all.
  `tank_skid_belt` discovers however many wheels the URDF has (validated on 10-
  and 14-wheel hulls) and scales `SkidSteer` / `PerSide` / `SameSideBelt` to
  them. Wheels are often **numbered** (`L_wheel_01` … `R_wheel_07`) rather than
  named by position — that is fine for the parser (side comes from the `L_`/`R_`
  prefix), but see §5: position-based `wheelName` overrides will NOT match a
  numbered hull.

## 3. Suspension — the URDF does NOT need to carry it

Standard URDF has **no spring-stiffness field**. `<joint><dynamics>` carries
only `damping` and `friction`, and on a prismatic joint those describe the
articulated solver, not a ray-wheel suspension. So by default you author the
geometry and mass, and the SDK supplies the spring. The value is resolved by
this priority chain (v1.2.1, honoured by the Python API and the OSC server
alike):

```
caller / OSC override   >   URDF <dynamics stiffness=…>   >   auto (mass / preset / default)
```

### 3a. No `<dynamics>` tag → how the spring is computed

This is the normal case. What fills `k_susp` / `c_comp` / `c_ext` depends on
how you build the vehicle:

| construction path | suspension when URDF is silent |
|---|---|
| `tank_skid_belt(urdf)` | **mass-derived** from the URDF's own `sprung_mass` and wheel count: `k = (m_sprung·g / n) / target_sag` (`target_sag` default 0.05 m → ~2.2 Hz), symmetric damper at ζ = 0.80 |
| `car_4w_*` / `truck_6w_*` presets | fixed preset constants (k ≈ 70 kN/m, tuned for a ~2 t car) — adequate because those vehicles ARE ~2 t |
| OSC server, mapping path (`driveType` 3) | fn-based autotune from `sprung_mass`: `k = m_share·(2π·f_n)²`, `f_n` 1.2–1.8 Hz by mass bucket, ζ 0.70 / 0.45 |
| bare `VehicleConfig` with no `wheel_overrides` | module default `DEFAULT_K_SUSP` = 70 kN/m |

The tank path is mass-derived precisely because a fixed rate cannot serve both
a 2 t car and a 40 t hull — a car-sized 70 kN/m spring under 40 t sags four
times its own travel and wallows at 0.8 Hz. If you want the spring by hand,
`suspension_from_mass(sprung_mass, n_wheels, target_sag=…)` returns the same
`(k, c_comp, c_ext)` triple. See [`api-reference.md` §3](api-reference.md).

### 3b. `<dynamics>` present → the URDF wins over auto

If you *do* want the URDF to pin the suspension, add the **non-standard**
extension attributes to each `*_susp` joint:

```xml
<joint name="L_wheel_01_susp" type="prismatic">
  <origin xyz="2.44 1.47 0.43"/>
  <parent link="body"/><child link="L_wheeltrack_01"/>
  <axis xyz="0 0 1"/>
  <limit lower="-0.17" upper="0.17" effort="20000" velocity="5"/>
  <!-- non-standard: read by the SDK, ignored by standard URDF tools -->
  <dynamics stiffness="450000"
            compression_damping="40000"
            extension_damping="25000"/>
</joint>
```

Rules the SDK applies (`urdf._susp_dynamics`, v1.2.1):

- Only a **non-zero `stiffness`** (or `spring_stiffness`) marks the tag as a
  suspension declaration. `stiffness="0.0"` means "no spring here" and is
  ignored — it does **not** zero the suspension.
- `damping` (symmetric) or `compression_damping` / `extension_damping`
  (asymmetric) are read only alongside a non-zero stiffness. A bare
  `damping="20.0"` — legal standard URDF, e.g. on a steer joint — is ignored.
- Values are in **our contract units**: `stiffness` in N/m, damping in N·s/m.
  Do NOT paste a UE Chaos `SpringRate` / `SuspensionDampingRatio` verbatim —
  Chaos uses different units and expresses damping as a 0–1 ratio, not a
  coefficient. Convert first, or omit the tag and let the mass-derived value
  handle it.
- A caller / OSC `wheelOverrides` value still outranks the URDF.

### 3c. Version note

URDF `<dynamics>` suspension reading on the **preset / tank path** is v1.2.1+.
On v1.2.0 and earlier, `driveType 2` (skid) took the preset branch, which never
read `<dynamics>`; only the server's mapping branch (`driveType 3`) read it. A
tank on ≤1.2.0 that put its spring in `<dynamics>` was silently ignored and ran
on the module default. Upgrade to ≥1.2.1, or send the value via `wheelOverrides`
(§5).

## 4. Geometry, mass, inertia

Read from the **wheel link**, with fallbacks:

| field | source | fallback |
|---|---|---|
| `radius` | wheel link `<cylinder radius>` or `<sphere radius>` | `DEFAULT_RADIUS` 0.35 m |
| `mass` | wheel link `<inertial><mass>` | `DEFAULT_MASS` 20 kg (informational; the pipeline does not read wheel mass) |
| `i_wheel` | wheel link inertia (max diagonal, or `estimate_spin_inertia_from_genesis`) | `DEFAULT_I_WHEEL` 1.5 kg·m² |
| chassis / sprung mass | base link `<inertial><mass>` + its non-wheel descendants | — |

`i_wheel` **is** read by the pipeline (it sets how fast a wheel spins up under
torque), so give heavy wheels a real inertia — a 500 kg tracked wheel left at
the 1.5 kg·m² default spins up unrealistically fast. An explicit
`WheelConfig.i_wheel` / `wheelOverrides.inertia` always wins over the URDF.

> **Sprung vs chassis mass.** `chassis_mass` is the base link **alone**;
> `sprung_mass` adds every non-wheel descendant (a turret, a cargo body).
> Size a spring against `sprung_mass` — on one tracked model the base link
> read 27 t against a true sprung mass of 38.5 t (a 42 % undersize). The tank
> preset and the server autotune both use `sprung_mass` since v1.2.1.

## 5. Addressing wheels in an override

`wheelOverrides` (OSC) / `wheel_overrides` (Python) key by `wheelName`, matched
exact / position / substring against **every** name in the wheel's joint chain
(wheel link, spin, suspension, steer), plus a `"*"` / `"all"` wildcard.

**Position matching (`FL`, `FR`, `RL`, `RR`, `front_left`, …) only fires on
wheels whose names encode a position.** A numbered tracked hull — `L_wheel_01`
… `R_wheel_07` — has no front/rear token, so `"FL"` matches nothing and the
override is dropped (with a warning since v1.2.1). For such a vehicle use:

- `"*"` — every wheel, or
- `"L_wheel"` / `"R_wheel"` — one track (substring), or
- the exact link name `"L_wheel_01"` — one wheel.

An override key that matches no wheel is reported (Python `logging.WARNING`;
server `[WARN]`) instead of vanishing silently. See [`server.md`](server.md)
for the full `wheelOverrides` field list.

## 6. Minimal skeletons

The three prep contracts still apply to all of these — the SDK auto-corrects,
but authoring them right avoids the informational churn. See
[`physics-contracts.md` §7.9](physics-contracts.md#79-urdf-contracts-for-ray-wheels-auto-corrected-since-v1122).

**One steered, driven car wheel (front-left):**

```xml
<joint name="front_left_suspension_joint" type="prismatic">
  <parent link="base_link"/><child link="fl_carrier"/>
  <origin xyz="1.35 0.75 0.30"/><axis xyz="0 0 1"/>
  <limit lower="-0.15" upper="0.15" effort="0" velocity="10"/>
</joint>
<link name="fl_carrier"><inertial><mass value="10"/>
  <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial></link>
<joint name="front_left_steer_joint" type="revolute">
  <parent link="fl_carrier"/><child link="fl_steer"/>
  <origin xyz="0 0 0"/><axis xyz="0 0 -1"/>   <!-- +steer → +angle -->
  <limit lower="-0.7" upper="0.7" effort="500000" velocity="10"/>
</joint>
<link name="fl_steer"><inertial><mass value="5"/>
  <inertia ixx="0.05" iyy="0.05" izz="0.05" ixy="0" ixz="0" iyz="0"/></inertial></link>
<joint name="front_left_wheel_joint" type="continuous">
  <parent link="fl_steer"/><child link="front_left_wheel"/>
  <origin xyz="0 0 0"/><axis xyz="0 1 0"/>    <!-- spin on +Y -->
  <limit effort="0" velocity="200"/>
</joint>
<link name="front_left_wheel"><inertial><mass value="20"/>
  <inertia ixx="1.5" iyy="1.5" izz="1.5" ixy="0" ixz="0" iyz="0"/></inertial>
  <visual><geometry><cylinder radius="0.35" length="0.25"/></geometry></visual></link>
```

**One tracked wheel (no steer joint):** drop the `*_steer_joint` /
`fl_steer` link — the suspension carrier's child is the spin joint directly.
Keep the `L_`/`R_` prefix for side detection.

---

*Related:* [`physics-contracts.md` §7.9](physics-contracts.md#79-urdf-contracts-for-ray-wheels-auto-corrected-since-v1122)
(ray-wheel prep contracts), [`api-reference.md` §4](api-reference.md) (`parse_urdf`),
[`server.md`](server.md) (`wheelOverrides` wire schema), [`tire-and-contact.md`](tire-and-contact.md)
(what the ray hit becomes).
