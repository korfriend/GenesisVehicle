# Testing

## Running the tests

From the repo root:

```bash
python -m pytest tests/ -v
```

155 pure-Python tests; no Genesis runtime required. Runs in ~30s on CPU. The
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
