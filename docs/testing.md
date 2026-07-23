# Testing

## Running the tests

From the repo root:

```bash
python -m pytest tests/ -v
```

206 pure-Python tests; no Genesis runtime required. Runs in ~40s on CPU. The
reference URDFs the parsing tests read live in `tests/data/` (self-contained
since v1.2.0).

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
| URDF-declared suspension | `test_urdf_parse.py` | non-zero `<dynamics stiffness>` honoured (symmetric + asymmetric damping); `stiffness="0.0"` and bare `damping` ignored; reference URDFs declare none |
| Suspension priority chain | `test_config_resolve.py` | caller override > URDF `<dynamics>` > mass-derived |
| Cusp arrival heading | `test_path_follower.py` | a backwards boundary hop doesn't become the block's arrival heading; a genuine corner still does; explicit waypoint yaw wins |
| Sweep CLI plant overrides | `test_sweep_plant_overrides.py` | `--top-speed` → omega from mean radius, `--omega-max-drive` direct, top-speed wins; only passed flags applied (suspension stays mass-derived); per-wheel + drivetrain knobs; no-op when none given |
| Top-speed governor + units | `test_top_speed_governor.py` | `kmh_to_mps`/`mps_to_kmh`/`omega_from_top_speed`; `_rev_limit` taper (untouched below cap, 0 at cap, half at half, engine-braking kept, no-op uncapped); presets cap from `top_speed`, radius-independent |
| Aerodynamic drag | `test_aero_drag.py` | `aero_drag_force` = ½ρ·CdA·v² opposing horizontal velocity (v² scaling, direction, vertical ignored, zero when off, batched); `ChassisConfig` drag defaults off; car preset has drag / tank does not; `drag_area` kwarg survives resolve; chassis mutable for runtime tuning |
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
| `raycast.py` | `WheelRayPattern`, `read_distances()` |
| `kinematics.py` | `get_link_transforms`, `LinkTransforms` — per-link transforms (world / base / parent frame) |
| `visual.py` | `InstancedWheelRenderer` (solver-free wheel visuals), `WheelJointInternalSync` (legacy joint-sync fallback), `patch_viewer_atomic_update` |
| `tire_models/` | `TireModel` ABC + `PacejkaAnisotropic`, `CoulombIsotropic` |
| `strategies/steering.py` | `SteeringStrategy` + 4 concrete |
| `strategies/drivetrain.py` | `DrivetrainStrategy` + 4 concrete |
| `strategies/coupling.py` | `CouplingStrategy` + 2 concrete |
| `strategies/stability.py` | `StabilityHook` + 3 concrete |
| `presets.py` | 4 ready-to-use `VehicleConfig` builders + `stability_hooks_for_profile` |
| `_version.py` | `__version__`, `VERSION_INFO` (single source of truth) |
| `tests/` | Pure-Python unit tests (no Genesis runtime needed) |
| `CHANGELOG.md` | Per-version release notes |
