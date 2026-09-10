# Testing

## Running the tests

From the repo root:

```bash
python -m pytest tests/ -v
```

375 tests, almost all pure-Python. Runs in ~100s on CPU (v1.6.0; the handful of
real-`VehicleScene` rollouts dominate — collection alone is ~5s). The reference
URDFs the parsing tests read live in `tests/data/` (self-contained since v1.2.0).

A handful build a real `VehicleScene` on the CPU backend (the batched-visual /
proxy-sync / to-host parity tests, and the one ray pattern that allocates
Genesis tensors), so `genesis-world` must be importable — but nothing needs a
GPU.

GPU integration is exercised by user-side demo scripts in your downstream
project — the SDK's own test suite stays pure-Python so it can run in any
CI without GPU.

## Test inventory

| Coverage area | Test file | Notes |
|---|---|---|
| Version reporting | `test_version_and_profile.py` | `__version__`, `VERSION_INFO`, `version()`, `version_info()` |
| Stability profile semantics | `test_version_and_profile.py` | `control` / `raw` / `research` materialize the right hook lists; tank-vs-car difference; unknown profile raises |
| Preset profile integration | `test_version_and_profile.py` | `car_4w_rwd_ackermann(stability="raw")` etc. produce expected hook lists |
| Config resolve | `test_config_resolve.py` | URDF default + user override + module default merge; user-explicit `i_wheel` / `radius` wins over URDF |
| URDF parsing (both susp-joint naming conventions) | `test_urdf_parse.py` | wheel discovery, side detection, axle clustering, chain walk for deep joint trees |
| Sprung / unsprung mass split | `test_urdf_parse.py` | `sprung + unsprung == total`; sprung includes non-wheel children (turret) that `chassis_mass` omits; equals `chassis_mass` for a simple car |
| Mass-derived suspension sizing | `test_config_resolve.py` | `suspension_from_mass()` formula, sag held constant across vehicle scale, input validation |
| `tank_skid_belt` suspension derivation | `test_config_resolve.py` | keyed by the URDF's own wheel names, `k` from that URDF's sprung mass, `target_sag` scales it, URDF wheel radius wins |
| Silent-failure guards | `test_config_resolve.py` | unmatched `wheel_overrides` key warns; undersprung suspension (sag > 1.25× stroke) warns; neither fires on a sane config |
| Wheel spin inertia about the spin axis | `test_urdf_parse.py` | wide wheel takes `iyy` not `max(diag)`; `<inertial rpy>` rotated first; disc-shaped wheel keeps its pre-v1.2.8 value |
| UE restitution drop + SimTag labels | `test_server_mesh_obstacles.py` | non-zero `Restitution` dropped with ONE summary line (none when all zero); logged SimTag matches the CoACD/convex-hull branch actually taken |
| URDF-declared suspension | `test_urdf_parse.py` | non-zero `<dynamics stiffness>` honoured (symmetric + asymmetric damping); `stiffness="0.0"` and bare `damping` ignored; reference URDFs declare none |
| Suspension priority chain | `test_config_resolve.py` | caller override > URDF `<dynamics>` > mass-derived |
| Cusp arrival heading | `test_path_follower.py` | a backwards boundary hop doesn't become the block's arrival heading; a genuine corner still does; explicit waypoint yaw wins |
| Sweep CLI plant overrides | `test_sweep_plant_overrides.py` | `--top-speed` → omega from mean radius, `--omega-max-drive` direct, top-speed wins; only passed flags applied (suspension stays mass-derived); per-wheel + drivetrain knobs; no-op when none given |
| Top-speed governor + units | `test_top_speed_governor.py` | `kmh_to_mps`/`mps_to_kmh`/`omega_from_top_speed`; `_rev_limit` taper (untouched below cap, 0 at cap, half at half, engine-braking kept, no-op uncapped); presets cap from `top_speed`, radius-independent |
| Aerodynamic drag | `test_aero_drag.py` | `aero_drag_force` = ½ρ·CdA·v² opposing horizontal velocity (v² scaling, direction, vertical ignored, zero when off, batched); `ChassisConfig` drag defaults off; car preset has drag / tank does not; `drag_area` kwarg survives resolve; chassis mutable for runtime tuning |
| Runtime setters | `test_runtime_setters.py` | `Vehicle.set_aero_drag` (only given args, clamps negative), `set_omega_max_drive` (None clears), `set_top_speed` (radius-independent) mutate the live resolved config; setters chain |
| Bump-stop spring | `test_bump_stop.py` | `bump_stop_force` zero inside stroke / linear beyond / off at 0 / batched; default off; tank preset enables at `TANK_BUMP_FACTOR`×k within the dt-stability bound; sweep `--k-bump` override |
| Ackermann sign + inner/outer | `test_strategies_unit.py` | `+steer` → both wheels positive, FR > FL |
| SkidSteer sign (left faster on +steer) | `test_strategies_unit.py` | `test_perside_iso_right_turn_left_faster` |
| SameSideBelt averages each side | `test_strategies_unit.py` | |
| RWD front-drive-zero invariant | `test_strategies_unit.py` | T_drive[FL] == 0, T_drive[RL] == T/2 |
| PerSide gear cap | `test_strategies_unit.py` | throttle=1.0, gear_cap=0.3 → effective 0.3 |
| `brake_torque_signed` reverses with omega | `test_dynamics.py` | `omega < 0` → `T_brake_eff < 0` |
| Suspension N clamped non-negative | `test_dynamics.py` | strong rebound → `N = 0`, not negative |
| Suspension air-mask → `N = 0` | `test_dynamics.py` | |
| Asymmetric damper (compression vs extension) | `test_dynamics.py` | same |c_dot| produces different N when c_compression ≠ c_extension |
| MultiVehicle grouping / input routing | `test_multi_vehicle_grouping.py` | `group_vehicles_by_cfg` kind grouping + caller-order preservation; `rebucket_inputs` flat→(kind, slot) routing round-trip (the L2×L3 input-routing logic, GPU-free) |
| Differentiable plant: autodiff vs finite differences | `test_differentiable_plant.py` | the autodiff Jacobian through the ray-wheel pipeline matches a finite difference of the same unroll |
| Differentiable plant: ISO sign contract | `test_differentiable_plant.py` | `d(a)/d(throttle) > 0` and `d(omega_z)/d(steer) < 0` on both car and tank; yaw authority grows with the horizon |
| Differentiable plant: prediction | `test_differentiable_plant.py` | batched over candidate commands (matches the scalar path); more throttle accelerates harder; slope attitude enters through gravity |
| Differentiable plant: inversion | `test_differentiable_plant.py` | the solve hits self-calibrated reachable targets; respects command ranges; a degenerate derivative is rescued by the secant probe; `set_applied` / `reset` move the linearisation point |
| Fleet batching equivalence | `test_differentiable_plant.py` | a member's predict / jacobian / solve inside an M-member plant equals the same member inverted alone; diagnostics are per-member; `set_applied_member` touches one row; mixed kinds and the scalar sweep protocol are refused |
| plan / finish split | `test_differentiable_plant.py` | `plan()` then `finish()` reproduces `step()` exactly, including `last_mode`; a decided plan (DONE / cusp brake) needs no inversion |
| `FleetFollower` | `test_differentiable_plant.py` | drives every member off one solve, reports each command to its own plant row, matches independent followers, rejects a count mismatch or an unbatched plant |
| Plant / PathFollower wiring | `test_differentiable_plant.py` | plant vs sweep-table dispatch, `.sweep` back-compat alias, steer range tightened to `steer_cap`, applied command reported on early-return exits, bad plant rejected |
| Ray-MISS sentinel | `test_ray_miss_and_grounded.py` | the engine's `no_hit_value` defaults to `max_range` and an explicit `0.0` is kept (not coerced); a real `VehicleScene` raycaster reports its OWN range; equality is exact for an unrepresentable sentinel (19.9, 0.1); `read_distances` leaves a non-default-range miss alone; the raw predicate is wrong when applied to corrected distances |
| Miss-sentinel validation, all three entry points | `test_ray_miss_and_grounded.py` | a sentinel below `max_range` raises at the stamp, at the read, and at `VehiclePhysics.set_ray_miss_value`; non-finite and non-positive rejected; a stamped `nan` cannot survive the read path; `VehiclePhysics` refuses a hand-built bad sensor at construction |
| `wheels_grounded` / `grounded_list` | `test_ray_miss_and_grounded.py` | all-False before the first step and on the unpopulated zero buffer; True after settling; False over a hole; the carried mask is used, not a recomputed one; batched `grounded_list()` matches per-vehicle |
| Partial reset renders at rest, not full compression | `test_ray_miss_and_grounded.py` | per-env `_stepped_once` on both `VehiclePhysics.wheel_visual_transforms` and the batched kind path, in both raycast modes; read layer / core / visual agree at a non-default range; the susp helpers keep their legacy positional signature |
| Swept-envelope fan geometry | `test_swept_envelope.py` | `s_j` spans `±r*fan_span` and centres exactly on 0; `c_j = r - sqrt(r^2 - s_j^2)` with `c_0 == 0` exactly and `c(±r) == r`; each wheel uses its OWN radius; even M refused with the flat-ground bias measurement in the message; M=1 needs no geometry; radii required and count-matched |
| Swept-envelope read layer | `test_swept_envelope.py` | `read_distances` collapses `(n_envs, n_wheels, M)` to `(n_envs, n_wheels)`; M=1 keeps a genuinely RANK-1 return shape (not a width-1 fan); misses pushed to `+inf` and an all-miss wheel reports the sentinel; the unpopulated buffer reads all-miss; `d_eff <= d_center` over 2000 randomised profiles (1784 mixed hit/miss) |
| Swept-envelope `single_scene` ceiling | `test_swept_envelope.py` | `self_collision_ceiling` / `single_scene_up_offset` take a per-RAY minimum, and both registration entry points (`VehicleScene.add_vehicle`, `make_wheel_raycaster`) hand them the whole fan rather than the wheel centres. Proved on a synthetic URDF — the reference car plus a low sill outboard of the front wheels — where the two caps actually disagree (0.28 m vs 0.12 m); a guard test asserts that discriminator still discriminates, since the reference car alone cannot make these tests fail |
| Wheel-contact mode selection | `test_swept_envelope.py` | `wheel_contact` / `contact_samples` on `VehicleScene.add_vehicle` and `make_wheel_raycaster`; `"swept_envelope"` with `contact_samples=1` refused; unknown mode refused; `check_fan_uniformity` rejects a kind with mixed M |
| The defect the envelope addresses (analytic harness) | `test_swept_envelope.py` | a wheel crossing a 0.130 m lip at 3.3 m/s through the real `read_distances` + `suspension_normal_force`: point contact takes the whole lip in one `dt` (`85,120 N`); M=9 spreads it (`49,978 N`, ratio 1.703); **M is not an accuracy dial** — ratios 1.703 / 1.906 / 2.039 / 1.959 at M = 9 / 15 / 31 / 101, non-monotone; M=3 at full span IS the point contact; the envelope never RAISES the peak, at any M or span |
| Swept-envelope in a real scene | `test_swept_envelope.py` | `single_scene` + M=9 does not self-hit and does not launch: same ride height and wheel distances as the point contact on flat ground, raw read `(1, 4, 9)` vs `(1, 4)` |
| The DEFAULT path did not move by one bit | `test_fan_default_identity.py` | a 200-step Genesis rollout of the reference car in the default configuration compared with `torch.equal` — **not `allclose`** — against wheel distances and the final pose captured from the pre-v1.5.1 tree (commit `20f7380`). Skips itself off genesis-world 1.4.0, because the baseline is a bit-pattern: re-capture, do not loosen. Also drives the M > 1 branch through a stub sensor, so the file witnesses the branch the rollout does not take |
| Server subpackage import + steer-key mapping | `test_server_import.py` | `genesis_vehicle.server` imports; `steerScale`/`maxSteerRad` mapping-key resolution (auto-skips without genesis/pythonosc) |

## Public-surface import smoke check

```bash
python -m genesis_vehicle.tests._check_import
```

Imports every symbol from the top-level package and prints the version +
the lazy-import names. Confirms that the eager surface loads without Genesis
and that the lazy names (`VehiclePhysics`, `WheelRayPattern`,
...) are properly registered.

## Module map

| File | Purpose |
|---|---|
| `core.py` | `VehiclePhysics` — 5-step pipeline orchestrator |
| `config.py` | `WheelConfig`, `ChassisConfig`, `VehicleConfig`, `ResolvedConfig`, `resolve()`, `ConfigError`, `DEFAULT_*` |
| `inputs.py` | `VehicleInputs`, `VehicleStepInputs`, typed inputs |
| `urdf.py` | `parse_urdf()`, `URDFParsedConfig`, `estimate_spin_inertia_from_genesis` |
| `dynamics.py` | `brake_torque_signed`, `suspension_normal_force` — pure helpers |
| `raycast.py` | `WheelRayPattern`, `read_distances()`, the ray-MISS sentinel (`is_ray_hit` / `is_ray_hit_corrected` / `ray_miss_value` / `check_miss_supported`), high-cast offset helpers, the swept-envelope fan (`fan_longitudinal_offsets` / `fan_height_offsets` / `fan_ray_positions` / `set_sensor_fan` / `sensor_fan`, v1.6.0) |
| `kinematics.py` | `get_link_transforms`, `LinkTransforms` — per-link transforms (world / base / parent frame) |
| `visual.py` | `InstancedWheelRenderer` (solver-free wheel visuals), `WheelJointInternalSync` (legacy joint-sync fallback), `patch_viewer_atomic_update` |
| `tire_models/` | `TireModel` ABC + `PacejkaAnisotropic`, `CoulombIsotropic` |
| `strategies/steering.py` | `SteeringStrategy` + 4 concrete |
| `strategies/drivetrain.py` | `DrivetrainStrategy` + 4 concrete |
| `strategies/coupling.py` | `CouplingStrategy` + 2 concrete |
| `strategies/stability.py` | `StabilityHook` + 3 concrete |
| `presets.py` | 4 ready-to-use `VehicleConfig` builders + `stability_hooks_for_profile` |
| `_version.py` | `__version__`, `VERSION_INFO` (single source of truth) |
| `control/plant.py` | `DifferentiablePlant` — autodiff inverse plant (the default for `PathFollower`), batched over (vehicle, env) members |
| `tests/` | Pure-Python unit tests (no Genesis runtime needed) |
| `CHANGELOG.md` | Per-version release notes |
