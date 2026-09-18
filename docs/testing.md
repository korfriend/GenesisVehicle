# Testing

| abbr | meaning |
|---|---|
| AST | Abstract Syntax Tree (Python's `ast` module — used as a source-level tripwire) |
| L2 | per-entity batching axis (K vehicles in one scene) |
| L3 | `n_envs` batching axis (same-URDF fleet) |
| M | swept-envelope contact samples per wheel (`contact_samples`) |
| SFL | `StaticFrictionLock` — the hold-at-rest stability hook |
| dt | one simulation STEP duration (s) |
| H | requested state-send rate (the server's `--send-hz` value, Hz) |
| P | send period, `1/H` (s) |
| substeps | solver iterations per step; internal interval is `dt / substeps` |

## Running the tests

From the repo root:

```bash
python -m pytest tests/ -v
```

605 tests, almost all pure-Python, plus three env-gated live tests that skip by
default. Measured on the v1.6.8 tree: `605 passed, 3 skipped in 135.07s`
(`python -m pytest tests/ -q --ignore=tests/test_fusion_probe.py`, CPU/WSL2,
genesis-world 1.4.0, one run, no warm cache control). `test_fusion_probe.py` is
an untracked file belonging to a parallel workstream and is not part of this
repo; a bare `pytest tests/` on the release working tree collects its 53 cases
too and reports `658 passed, 3 skipped`. The 3 skips are
`test_server_send_downsample_live.py` (1) and
`test_server_stats_window_live.py` (2), which spawn real server processes and
run only with `GENESIS_VEHICLE_LIVE_SERVER_TEST=1`. Earlier figures on the same machine:
`578 passed, 1 skipped in 137.19s` at v1.6.7,
`544 passed in 144.45s` at v1.6.6, `519 passed in 144.33s` at v1.6.5,
`507 passed in 170.53s` at v1.6.4,
~116s / 450 tests at v1.6.2-v1.6.3, ~88s at v1.6.1. **Runtime is not a stable
metric here** — v1.6.5 added 12 tests and measured 26s FASTER than v1.6.4, so
the run-to-run spread on this machine exceeds the difference a release makes;
read the count, not the seconds. The count breaks down as v1.6.3's 450 + 57 for
v1.6.4's build-time hoists + 12 for v1.6.5 (7 in
`test_sim_options_and_timing.py`, 3 engine-shape stubs in `test_gs_compat.py`,
2 `sim_options` write-guard cases in `test_server_import.py`) + 25 for v1.6.6
(all in `test_server_serving_counters.py`) + 34 for v1.6.7
(`test_server_send_downsample.py`; the 35th, the live arrival-rate test, is a
skip) + 27 for v1.6.8 (26 in `test_server_stats_window.py` and one more in
`test_server_serving_counters.py`, 25 → 26; the two cases of
`test_server_stats_window_live.py` are skips). Collection alone is ~6s. The reference URDFs the
parsing tests read live in `tests/data/` (self-contained since v1.2.0).

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
| Raycast-mode benchmark harness | `test_bench_raycast_mode.py` | 61 tests, all pure Python: the shipped entry point `main(argv, runner=fake)` is driven with a STUB runner, so the whole schedule (even/adjacent pairing, alternating slot order), the validity + x-margin gates, the cross-worker invariants, `paired_ratios` / `split_groups` and the fail-closed publication rule run without spawning a process. The only test in the suite that patches `genesis.init` — to RAISE, proving the PARENT path never calls it; that is NOT a claim that genesis is absent from the parent (`__init__.py` imports it eagerly via `control/plant.py`). Also pins `verdicts()` backward compatibility: no `terrain_half` key -> bound 18.0 and "±20 m", `terrain_half=(320, 320)` -> 318.0 |
| Config rebuild carries the runtime state | `test_rebuild_state_carry.py` | 14 tests, all on a real CPU `VehicleScene` (`samples/urdf/car_4w.urdf`, `dt = cfg.recommended_dt`, substeps 10, friction 1.0); two of them add a 160×120 offscreen camera, which is what turns the wheel-visual path on headless. Nothing skips. Covers: `torch.equal` copy fidelity of every carried attribute; 40-step continuity against a control scene built in the same process; the wheel rest pose (a regression reproduces z `0.3000` → `0.1499`); `VehicleScene.reset()` actually resetting the batched driver; a post-build wheel-count change and a wheel-ORDER change both raising `ValueError` with `_grouped_version` rolled back so the next step raises the same thing; the authoritative post-construction check driven directly; `MultiVehicleKindPhysics.reset(rows=)` vs `MultiVehiclePhysics.reset(vehicle_ids=)` semantics at K=2 / `n_envs`=2 and the `TypeError` on `vehicle_ids=`; a per-vehicle reset not wiping another vehicle's visuals; the instanced renderer rebind, including that the FIRST frame after it does not raise; plant freshness by identity AND by solve equality, plus `RuntimeError` on a structural change; the whole carry again under `raycast_mode="single_scene"`; and that one `mark_config_dirty()` rebuilds exactly ONCE over the next 10 steps |
| Build-time hoists stay LIVE | `test_build_time_hoists.py` | 23 tests (v1.6.4). Every value the step path now derives once — Ackermann geometry, driven-axle share, brake bias, AWD weight normalisation, the drive-omega cap, SFL's `v_thr²` — is derived FIRST, then its source is written, then the next call must move. Written that way round deliberately: a "snapshot at construction" implementation passes any test that writes the source before the first call, and fails these (confirmed by a reviewer who substituted a snapshot: 16 red). Numbers are characterisation baselines captured from the pre-hoist tree |
| `eps_v` is live on both tire models | `test_tire_coulomb_eps.py` | 11 tests (v1.6.4). Same characterisation shape for `CoulombIsotropic._eps2` and the Pacejka arm; also the file that gives `tire_models/coulomb.py` its first behavioural coverage (it had none) |
| Per-wheel broadcast rank convention | `test_rank_helpers.py` | 6 tests (v1.6.4). `_pipeline.pw` / `pw3` return `unsqueeze(0)` at rank 1 and pass a rank-2/3 field through unchanged. The rank-2 branch is not taken anywhere in this tree yet, so it is exercised explicitly rather than left to the promotion step |
| The bump-stop/`dt` warning names the vehicle | `test_bump_stop_dt_warning.py` | 3 tests (v1.6.4). RED before the change — the old message carried no vehicle name, URDF name or slot index. Scope: one driver = one config, so "per vehicle" means "per config, named after the vehicle it was registered as"; a per-SLOT ratio inside a fused group is not this test |
| L2 serving counters, the capture skip gate and the benchmark's kind/serving wiring | `test_server_serving_counters.py` | 26 tests (25 at v1.6.6, +1 at v1.6.8), pure Python — no server process is started. `needs_override_capture`'s truth table and its deliberate exclusion of `target_forces` (which goes through `control_dofs_force` and moves nothing this loop); `capture_state`'s `tag=` being keyword-only with a default and all four loop call sites carrying one; the five `[SERVE]` counters existing, incrementing and resetting on the same window boundary as the `[STATS]` counters (the split anchor is the v1.6.8 wall-clock trigger, asserted PRESENT before splitting — see the vacuous-pass note below); `recv_loops` counted at the skip decision rather than at `if recv:` (so the `skipped_captures + nonskip_loops == recv_loops` identity survives a `stop` loop); the `[SERVE]` token marking ONLY the counter line; the line parsing with the benchmark's parser and disturbing NEITHER `_STATS_RE` copy; `--kinds 1` being bit-identical to the historical friction constant and `--kinds N` splitting into exactly N kinds without touching global or obstacle friction; `--input-hz` reproducing the historical sleep exactly; the observed-kind anchor strings matching what the server actually prints; a replayed server stdout driven through `run_config`; `reset_ran` armed per loop and set only by the reset branch; `post_step_captures_ref` counted outside the catch-up loop; the override capture being the only call behind the gate; L3 reporting one kind and flagging it not observable — with an L3 `[SERVE]` line in the canned input since v1.6.8, so the case asserts the real window count and that no L2-only key (nor `n_envs`) reached the summed dictionary; `serving_ms == loop_ms − phys_ms` being an IDENTITY under the pooled estimator rather than an approximation (v1.6.8); and the `serve_windows` reader race guard plus the short-stream rejection, which now also pins `weighting == "unweighted"` and a NaN `D`. Source-level assertions where the behaviour lives in `main()`'s body — they read `server/physics_server.py` via `inspect.getsource` / `ast` rather than run the loop, and auto-skip without `genesis` / `pythonosc` |
| The L2 `--send-hz` state-send gate | `test_server_send_downsample.py` | 34 tests (v1.6.7), pure Python. `SendRateLimiter`: the OFF predicate returning exactly `True` on adversarial clocks (repeats, backwards time, huge jumps, zero), the first send after build not being delayed (`_next_send_t` starts at `-inf`), exact send indices for a dividing H (20 on a 40 Hz loop) and a non-dividing one (25, where the deadline CARRIES), the stall snap, and the single-budget invariant `_next_send_t <= _last_send_t + P` after every record. The REALISED rate is pinned separately from the fields — a mirror of the server's gate is driven with three injected event sequences, including the 30 Hz-override-vs-40 Hz-loop one that made the rejected two-budget design reach 1.65 H, and the rejected design is shown to fail that pin. Source-level: an `ast.dump` FREEZE of the gate body against the pre-v1.6.7 block (the `0.9999` clip, the `if interpolated[...]` structure, the `(step_count - 1 + alpha) * SIM_DT` stamp), the gate wrapping the whole send with `send_step_ack` outside it, the skip branch doing nothing but counting, the OFF path calling no `perf_counter()`, `_flush_hard` armed per loop OUTSIDE `if recv:` and set in exactly one place, no soft-flush path existing, `--legacy-override-capture` not ORed into the flush, the argparse shape and help text (the lag cost and the 75 ms teleport bound), the non-positive `parser.error`, the L3 WARN, demotion judged against `sim_dt` not `_eff_dt`, the banner in both arms, `window_ms` being measured to a clock read AFTER the loop body (v1.6.8 rewrote this one as `test_window_ms_is_measured_to_a_clock_read_after_the_loop_body`: it now pins the statement ORDERING of `_t_end` rather than the identity of the call site, because an upper bound alone would accept a `_t_end` hoisted to the top of the loop, which collapses `loop_dur` to ~0 and drives `serve_ms` negative), the four new `[SERVE]` keys parsing with the benchmark regex while matching NEITHER `_STATS_RE` copy, the identity comment naming the non-lockstep path it holds on (`sends + send_skips == loops` since v1.6.8, with the old `== 50` form asserted GONE), and the benchmark's forwarding + record |
| `--send-hz` arrival rate at a real client (env-gated) | `test_server_send_downsample_live.py` | 1 test (v1.6.7), SKIPPED unless `GENESIS_VEHICLE_LIVE_SERVER_TEST=1`. Spawns the real server twice (arm A without the flag, arm B with `--send-hz 20`) and counts `/Genesis/Vehicle/TargetBulk` arrivals at a `ThreadingOSCUDPServer` listener over a fixed wall-clock window. Measured: A **136.37 Hz**, B **20.00 Hz** (CPU/WSL2, genesis-world 1.4.0, L2, K=1 tank, dt 25 ms, 8 s window anchored on the first TargetBulk after a 2 s warm-up, three repeats). A fixed WALL-CLOCK window is used because it must count arrivals AT THE CLIENT, which `run_config` does not observe. Through v1.6.7 there was a second reason — a `[STATS]`/`[SERVE]` window was 50 LOOPS and the gated arm's windows collapsed to a median `window_ms` of 0.0–0.1 — and that reason is FIXED in v1.6.8 (the window is `--stats-interval` seconds). Its hard `sends + send_skips == 50` assertions are now `== r["loops"]`. Scope, stated in the file: it proves the arrival RATE, not packet bit-identity (a structural argument) and not the reset FLUSH path (the mock client sends neither reset nor override). An env-var skip rather than a custom marker because the repo has no `pytest.ini` / `pyproject.toml` / `conftest.py` |
| The wall-clock `[STATS]`/`[SERVE]` window and the benchmark's pooled weighting | `test_server_stats_window.py` | 26 tests (v1.6.8), pure Python — no scene, no server process. The trigger being wall-clock in BOTH `main()` and `run_l3()` with no literal `50` in either, zero `/ 50.0` denominators left, and the window block running AFTER `log_count += 1` (the AST lineno invariant that makes `/ _n` division-safe). A BYTE-IDENTITY freeze of both `[STATS]` f-strings and both `_STATS_RE` copies still matching a synthetic `[STATS]` while matching NEITHER `[SERVE]` line. The bare `[SERVE] [L3]` tag yielding exactly six keys with no `n_envs` (the benchmark's k=v regex harvests pairs from anywhere on the line, so a `[L3 n_envs=100]` tag would be summed as a metric); the five shared keys present in both modes and, on L2, in the BASE f-string outside the `--serve-timers` gate; the identities `sends + send_skips == loops`, `zero_step_loops <= loops`, `serve_ms >= 0`, `serve_ms <= window_ms`, `phys_ms_sum >= 0` and `serve_ms + phys_ms_sum == Σloop_dur × 1000` — `serve_ms >= 0` is on the list because the upper bound ALONE lets a negative through, which is how a mutation of the stamp position escaped detection; the L2-only keys ABSENT from the L3 line; `zero_step_loops` initialised/incremented/reset in both modes; the window origin carried rather than re-read; `--stats-interval` validated with no L3-ignores warning; `budget_ms` being MILLISECONDS; and the resolver — `pacing.resolve_stats_interval` clamping `-1.0` / `-1e-9` / `0.0` / NaN / an unparseable string back to 1.0 while letting valid values (including the string `"0.25"`) through, warning LOUDLY on every rejected value while staying SILENT for unset (missing attribute or explicit `None`) and for valid ones, and BOTH loops reading through that one resolver with no local `or`-style fallback left. And the estimator guards: `run_config` re-weighting by `loops` on three synthetic windows of deliberately unequal length (40 / 4000 / 38 loops) with the value the unweighted estimator WOULD have produced asserted to differ, the loud `"unweighted"` fallback, the sample-size record, survival of a zero-`steps` window, the summary table's `weighting` column on every row, the `!! MIXED ESTIMATORS` footer replacing the pooled caption when any row fell back, and `!!low-sample` |
| Window length at a real server (env-gated) | `test_server_stats_window_live.py` | 2 tests (v1.6.8), SKIPPED unless `GENESIS_VEHICLE_LIVE_SERVER_TEST=1`. Spawns real servers on dedicated ports: every `[SERVE]` window is one `--stats-interval` long in BOTH `--send-hz` arms and in BOTH modes, the per-line identities hold on the wire, and the L3 server prints a `[SERVE] [L3]` parsing into exactly the six keys with no `n_envs`. It prints `zero_step_loops` so the spin dilution is visible. Stated in the file: it proves NOTHING about the quality of `Loop Avg` / `steps/loop` (still diluted), and its ms figures are not a benchmark — one process per arm, no order alternation, no repeats. Conditions: CPU/WSL2, genesis-world 1.4.0, dt 0.025, K=1 tank, `--road-raycast-only`, 30 Hz input, `--stats-interval 1.0` |
| Server subpackage import + steer-key mapping + the `sim_options` write guard | `test_server_import.py` | `genesis_vehicle.server` imports; `steerScale`/`maxSteerRad` mapping-key resolution (auto-skips without genesis/pythonosc). **v1.6.5: an AST source guard** — any `Assign`/`AugAssign`/`AnnAssign` in `server/*.py` whose target attribute chain mentions `sim_options`, plus a literal `setattr`, fails the suite, with a second test proving the guard catches the shapes its docstring claims (`=`, `+=`, subscripted target, `setattr(getattr(...))`). It is a LITERAL-form tripwire, not runtime proof: aliasing, dynamic `setattr`, `exec`, and three literal forms named in the docstring (tuple-unpacking target, `for x.sim_options.dt in ...`, `with ... as x.sim_options.dt`) all escape it — the runtime facts live in `test_sim_options_and_timing.py` |
| Simulation time is fixed at `build()` | `test_sim_options_and_timing.py` | 7 cases (v1.6.5), on a real CPU `VehicleScene`. The `[genesis_vehicle] timing:` line is asserted to be EMITTED (`capsys` + regex) and its parsed dt / substeps must equal `effective_dt` / `substeps` — the line printed zero times for eight releases inside an `except Exception: pass`, so "the code path exists" is exactly what does not count here. Also: `sim_options` returns the AUTHORED object; post-build writes to `sim_options.dt` / `.gravity` are pinned INERT (a tripwire on the ENGINE — a future genesis that makes them live turns this test red instead of rotting the contract); `set_gravity` is pinned LIVE; `substeps` / `effective_dt` must equal `scene.sim.*`; `set_gravity` before `build()` raises; `envs_idx` passes through at `n_envs > 1` |

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
| `_hotset.py` | the HOT set and its build-time derivations: `derived()` (dependent cache, off-instance), `prime_derived()`, `HOT_DEPENDENTS`, `square_f64`, `row_tensor` (v1.6.4; contract in `docs/physics-contracts.md` §7.13) |
| `_version.py` | `__version__`, `VERSION_INFO` (single source of truth) |
| `control/plant.py` | `DifferentiablePlant` — autodiff inverse plant (the default for `PathFollower`), batched over (vehicle, env) members |
| `tests/` | Pure-Python unit tests (no Genesis runtime needed) |
| `CHANGELOG.md` | Per-version release notes |
