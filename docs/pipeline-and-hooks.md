# Pipeline and hook insertion points

The 5-step pipeline with every strategy and stability-hook seam marked.

| abbr | meaning |
|---|---|
| derived value | a strategy value computed once at build from config, re-computed when a source moves (`_hotset`, v1.6.4) |
| `pw` / `pw3` | `_pipeline.pw` / `pw3` — rank-checked per-wheel broadcast helpers |
| SFL | `StaticFrictionLock` |
| M | `contact_samples` — rays per wheel in a swept-envelope fan |
| RR | rolling resistance (`RollingResistance`) |
| ω, ω′ | wheel spin rate before / after this step's pre-tire resisting torque (ω′ = `omega_r` since v1.8.0) |
| `omega_d`, `omega_t`, `omega_e` | v1.8.0: ω after the drive torque / after the brake projection / `omega_t` clipped to ±`OMEGA_MAX` |
| `B_rem` | brake capacity left on a wheel the brake stopped this step (N·m); exposed as `ctx.brake_rem` since v2.0.4 |
| `k_eff` | `StaticFrictionLock`'s per-wheel capped spring rate (v2.0.4, physics-contracts.md §7.22) |
| D / Y / Z, `s` | held-wheel deadbeat / lateral deadbeat / their rigid-body scale (physics-contracts.md §7.16) |
| B2 | `F_lat` and κ taken at the transmitted (clamped) `F_long` (`TireModel.at_long_force`) |
| `T_resist` / `T_rr` | the axle resisting torque a pre-tire hook sets / its capped value `cap_resist_torque(T_resist, ω)` |
| R, `radius` | wheel radius (m) |
| `DT`, dt | simulation step duration (s) |
| `I_wheel`, `i_wheel` | wheel spin moment of inertia (kg·m²) |
| `OMEGA_MAX` | wheel spin-rate clamp (rad/s) |
| μ | tyre-ground friction coefficient |

```
[build]  VehiclePhysics.__init__ → _hotset.prime_derived(resolved, wheel_meta,
         dev, fdt)
         # v1.6.4: the Ackermann geometry, the driven-axle share, the
         # brake-bias / AWD weight vectors, the drive-omega cap and the two
         # squared thresholds (SFL v_thr², Coulomb eps_v²) are computed HERE,
         # not per step. They stay LIVE: a post-build write to any source in
         # _hotset.HOT_DEPENDENTS is picked up on the next call, including an
         # in-place list edit. Per-wheel config is NOT live — it is read off
         # WheelMeta and needs mark_config_dirty(). (physics-contracts.md
         # S7.13, S7.12)

VehiclePhysics.step(inputs)

[0] Input adaptation
    gear_req = getattr(inputs, "gear", None)   # v2.0.2: read off the ORIGINAL object; refuse a
                                               # skid-steer vehicle / bad scalar here (physics-contracts S7.20)
    typed = inputs if isinstance(inputs, steering.InputType)
            else steering.InputType.from_unified(inputs)

[1] PRE-LOOP
    steer_per_wheel = SteeringStrategy.per_wheel_steer(typed)
    T_drive_pw, T_brake_pw = DrivetrainStrategy.distribute_torque(typed, omega, meta)
        # v2.0.2 gear rows: typed' = replace(typed, throttle = gear * clamp(throttle, 0, 1));
        # ctx.throttle = that effective throttle. After the chassis state read [3]:
        # update_gear (clamped throttle == 0 AND v_xy < 0.5; not on the first-step
        # protection step), then T_drive clamped to the POST-shift gear's sign,
        # last_T_drive = the clamped tensor, ctx.gear = the engaged gear (before the
        # PRE_LOOP hooks). No gear on any input: none of this runs (the 2.0.1 path).
    for hook in stability_hooks if PRE_LOOP in hook.slots:
        hook.apply_pre_loop(ctx)                     # e.g. LowSpeedRegularizer

[2] Raycast (read_distances, return_hit=True) + first-step protection
    # the hit mask is evaluated on the RAW distances (the only place
    # raycast.is_ray_hit is correct) and carried into last_hit /
    # wheels_grounded. It is a read-layer diagnostic; the physics AIR MASK
    # below is compression <= 0, not the sentinel. (physics-contracts.md S7.8, S7.10)
    # A swept-envelope vehicle (wheel_contact="swept_envelope", v1.6.0) reads
    # back (n_envs, n_wheels, M) here; read_distances reduces the M axis with
    # d_eff = min_j(d_j + c_j) BEFORE anything below sees it, so no step of the
    # pipeline and no hook changes shape. (physics-contracts.md S7.11)

[3] Chassis state read

[per-wheel loop i = 0 .. n_wheels-1, all batched over n_envs]
    (A) compression, comp_rate, asymmetric damper -> N     (physics-contracts.md S7.2)
        # per-wheel WheelMeta fields enter through _pipeline.pw / pw3 (v1.6.4):
        # rank 1 -> unsqueeze(0), rank 2/3 passed through. Every field is rank 1
        # today, so this is value-identical. (S7.13)
        # bump-stop term guarded by the build-time bool WheelMeta.has_bump_stop
        # = OR over k_bump > 0 — a WHOLE-BATCH branch, exact because
        # bump_stop_force is 0 where k_bump == 0. (S7.13)
    (B) wheel-frame fwd/lat using steer_per_wheel[:, i]
    (B') PRE-TIRE phase (v1.7.0) — ctx.T_resist = None, F_long/F_lat = None, brake_rem = None (v2.0.4)
        s = deadbeat_scale(N, yaw_inertia)           # Z (v1.8.0, S7.16); ctx.deadbeat_scale
        for hook in stability_hooks if POST_TIRE in hook.slots:
            hook.apply_pre_tire(ctx)  if present     # RollingResistance
    (P) axle torques BEFORE the tyre (v1.8.0)
        omega_r = omega - cap_resist_torque(T_resist, omega) * DT / I_wheel   # if T_resist set (S7.15)
        omega_d = omega_r + DT * T_drive / I_wheel
        omega_t, B_rem = brake_projection(omega_d, T_brake)   # Coulomb, cannot cross 0 (S7.1)
        omega_e = clamp(omega_t, ±OMEGA_MAX)                  # M2
    (Q) bounds, fixed before the tyre
        L = (omega_t - v_long/R) * I_wheel / (DT * R)
        [lo, hi] = [min(0,L) - B_rem/R, max(0,L) + B_rem/R]   # ctx.F_long_bounds (S7.7)
        D = s * (N/g) * |v_long - R*omega_t| / DT  where B_rem > 0, else inf   # ctx.F_long_deadbeat
        ctx.brake_rem = B_rem.detach().clone()                # v2.0.4, read-only for hooks (S7.22)
    (C) F_long, F_lat, kappa = TireModel(..., v_roll = R * omega_e)
        where clamp(F_long, lo, hi, ±D) binds:                # B2 (S7.7)
            F_lat, kappa = tire.at_long_force(..., F_long_target = clamped F_long)
        for hook in stability_hooks if POST_TIRE in hook.slots:
            hook.apply_post_tire(ctx)  if present    # ctx.omega = omega_t; SFL may set ctx.hold_applied
                                                     # SFL engages on ctx.brake_rem > 0 (v2.0.4)
        # a write to ctx.T_resist here raises RuntimeError
    (D) F_long = clamp(F_long, lo, hi, ±D)          # held wheels (ctx.hold_applied): [lo, hi] only
        omega = omega_t - DT * R * F_long / I_wheel, then |omega| -= B_rem*DT/I_wheel (not past 0)
        airborne: omega = omega_t
        F_lat = clamp(F_lat, ±s*(N/g)*|v_lat|/DT)    # Y, not on held wheels (S7.16)
        omega[:, i] = clamp(omega, ±OMEGA_MAX); LSR pull; a hook's ctx.omega_override
        # (SFL set the override through 2.0.3; since v2.0.4 no SDK hook does)
        # Through v1.7.0 (D) was: T_brake_eff = brake_torque_signed(T_brake, omega)
        # (that function was deprecated in v1.8.0 and removed in 2.0.0)
        # (a tanh brake), capped with T_resist, and the drive and brake torques went
        # through the overshoot clamp with the tyre at omega' (pre-step omega less RR). That
        # ordering made omega alternate step to step under throttle or brake.
    (E) accumulate F_world, torque into total_F, total_T

[4] CouplingStrategy.apply(omega, meta)              # post-loop, pre-force-apply
[5] _gs_compat.apply_links_wrench(solver, F, T) (chassis)
    → solver.apply_links_external_wrench  (the one path since v2.0.5,
      genesis-world 1.5.0; the <= 1.3.3 force + torque fallback was removed)
[6] wheel visuals (rendered scenes only): VehicleScene streams closed-form
    wheel poses into instanced render nodes AFTER the physics advance — NOT
    via the rigid solver. WheelJointInternalSync, the legacy solver-joint
    fallback, runs here instead for n_envs > 1 /
    wheel_render_mode="internal_sync"; it writes solver joints and DOES
    move the chassis trajectory (physics-contracts.md §7.8, corrected in
    v2.0.5). ext. renderers use visual_parts_transforms.
    (Through v2.0.4 this step said that with the native viewer the stream
    ran inside viewer.update()'s render-lock hold, atomic with the camera +
    node poses (v1.1.25). That was true on genesis-world <= 1.3.3 only: the
    patch that did it, patch_viewer_atomic_update, returned False on >= 1.4
    and was removed in v2.0.5.)
```

## Hook slots

A stability hook declares which slots it implements via the `slots` class
attribute (a tuple containing some of `"PRE_LOOP"`, `"POST_TIRE"`).

| Hook | Slots | What it does |
|---|---|---|
| `RollingResistance` | `("POST_TIRE",)` | Pre-tire phase (v1.7.0): adds the axle torque `cr * N * r * tanh(ω * r / tanh_scale)` to `ctx.T_resist`; the pipeline caps it and applies it to ω first, before the drive torque, the brake and the tyre (v1.8.0 order). No post-tire work. **Through v1.6.10 this row said it "subtracts `cr * N * tanh(v_long / scale)` from `F_long`". That was what the code did, but it was wrong as a description of an effect: the overshoot clamp ran after the hook and cut the force back to zero, so the hook changed nothing (measured coast decel 0.0073 m/s² with or without it on the reference car; CHANGELOG 1.7.0 §1.1).** |
| `LowSpeedRegularizer` | `("PRE_LOOP", "POST_TIRE")` | Pre-loop: compute `moving ∈ [0,1]` from chassis speed. Post-tire: scale `F_long`/`F_lat` by `moving`; record an omega pull target so core blends ω toward `v_long / radius`. |
| `StaticFrictionLock` | `("POST_TIRE",)` | **Since v2.0.4** engages per wheel where the brake holds the wheel (`ctx.brake_rem > 0`) and planar contact speed `sqrt(v_long² + v_lat²) < min(v_thr, 0.5)`, at any pedal; never sets `ctx.omega_override`; replaces the tyre force only after its creep test (the held-wheel deadbeat binds and the speed did not fall this step); before that the anchor follows the contact and the brake does the stopping. Its spring rate is `k_eff`, `k_spring` capped per wheel ([`physics-contracts.md` §7.22](physics-contracts.md)). **Through 2.0.3 this row said: engages when `brake > brake_thr` and speed `< v_thr`, requests `omega = 0` via `ctx.omega_override` from the first engaged step, takes over below `REST_SPEED` (0.05 m/s) or on the creep test** — the 2.0.3 code; the pedal rule, the forced ω and the rest-speed takeover were removed in v2.0.4 (a pedal ≤ 0.3 never held a vehicle on a slope; the forced ω slid locked wheels where the brake could not hold). Then a 2D spring-damper `F = -K_spring·displacement - D·velocity`, `D = max(k_damp, critical)` and bounded by the one-step deadbeat, projected onto the per-wheel friction ellipse and limited to `ctx.F_long_bounds`; the wheels it holds go into `ctx.hold_applied`. Holds the point where the contact came to rest ([`physics-contracts.md` §7.17](physics-contracts.md)). **This row said, through v1.7.0, "stores anchor at lock engagement" and "holds with zero drift on any slope up to μ ≈ tan(slope_angle) (truck preset … rock-solid up to ~30°)". The second was wrong: the hold is limited by the brake capacity (the tank preset does not hold 10°), and on 1.7.0 the reference car crept down 10° under full brake (CHANGELOG 1.8.0 §2.2). The ~30° truck figure was not re-measured for 1.8.0.** |

**Hook config stays live (v1.6.4).** `StaticFrictionLock`'s speed threshold is
now squared once at build (`v_thr²`, in double, then cast) instead of on every
`apply_post_tire`. Writing `hook.v_thr` after `build()` — which
`samples/tank_tuning.py` does — still takes effect on the next step: the cached
square is keyed on `v_thr`. The cache lives off the hook instance, not in its
`__dict__`, because `control/plant._sync_hooks` row-slices every TENSOR
attribute of a live hook by the flat batch; a config tensor sitting there would
be sliced as if it were integrator state. The SOURCES (`v_thr`, `k_spring`,
`k_damp`) are still plain public attributes — set them freely. (`brake_thr`
was one of them through 2.0.3; it was removed in v2.0.4 and reading or writing
it raises `AttributeError`.) Since v2.0.4 the squared engage speed is
`min(v_thr, 0.5)²`, and the lock's capped stiffness `k_eff` is derived the same
way, keyed on `k_spring`, `ctx.dt`, `wheel_meta.static_mass_share` and
`wheel_meta.static_rigid_factor`, so a `k_spring` write is capped and lands on
the next step.

**Two calls per step for a `POST_TIRE` hook (v1.7.0).** The pipeline calls
`apply_pre_tire(ctx)` on every hook of the POST_TIRE list before the tyre, then
`apply_post_tire(ctx)` after it. Both are looked up with `getattr`: a hook
object that lacks one of them is skipped for that phase (in v1.6.10 a missing
`apply_post_tire` raised `AttributeError`); `StabilityHook` gives both as
no-ops. In `apply_pre_tire`, `ctx.N`, `v_long`, `v_lat`, `omega` (the pre-step
ω), `air_mask`, `wheel_meta` and `dt` are valid; `F_long` / `F_lat` are `None`.
It is the only phase that may set `ctx.T_resist` (add to it, do not overwrite
another hook's value). In `apply_post_tire`, `ctx.omega` is `omega_t` (v1.8.0:
after the resisting torque, the drive torque and the brake projection; through
v1.7.0 it was ω′, after the resisting torque only); `ctx.F_lat` is already the
B2 value where the clamp binds; `ctx.F_long_bounds`, `ctx.F_long_deadbeat`,
`ctx.deadbeat_scale` and (v2.0.4) `ctx.brake_rem` are read-only (the pipeline
keeps its own copies — but a write is not detected, and a LATER hook in the
list sees the written value; `StaticFrictionLock` reads `brake_rem`, so a hook
listed before it that writes `brake_rem` changes what it engages on);
`ctx.brake_rem` is `None` in `apply_pre_tire`, is the pre-coupling value, is
not masked for airborne wheels and reads 0 on a wheel the brake stops with no
torque to spare (at standstill: an exact drive == brake tie);
`ctx.hold_applied` is the one field a hook writes (OR its mask in).
`PipelineResult.kappa` / `last_kappa` are the slip at `omega_e` (`omega_t` clipped to ±`omega_max`, the speed the tyre is evaluated at), replaced by
the slip of the transmitted force where the clamp binds.
A rebinding of `ctx.T_resist` there raises `RuntimeError`; an in-place write
raises too, except under `torch.inference_mode()`, where it goes undetected but
cannot change the physics because the pipeline already took a snapshot.
`physics-contracts.md` §7.15.

**A hook's `F_long` / `F_lat` edit may not survive.** The overshoot clamp runs
after every POST_TIRE hook, so a hook's `F_long` is a request the clamp can
still cut, to zero on a freely rolling wheel; and since v1.8.0 a hook's `F_lat`
is a request the lateral deadbeat Y can cut, on every wheel not in
`ctx.hold_applied`. A resistance that must reach the chassis belongs in
`ctx.T_resist`.

Hook order is the list order (see
[`stability-profiles.md`](stability-profiles.md#hook-ordering-inside-a-profile)
for the recommended ordering).

## Coupling timing detail

`CouplingStrategy.apply()` runs at step [4], after the per-wheel loop
completes and after `omega` has been integrated for every wheel, but
**before** the chassis wrench is applied (step [5]). The chassis
force applied in the same step still reflects the pre-coupling per-wheel
F_long (because F_long was computed inside the loop using pre-coupling
omegas). The coupled omega becomes visible to the next step. This is the
reference-tank behavior and is documented as the SDK contract.

## What hooks CANNOT do

- Mutate `T_drive_per_wheel` / `T_brake_per_wheel` (those are owned by the
  drivetrain; if you need custom torque routing, subclass the strategy).
- Change what the brake projection computed: `ctx.brake_rem` (v2.0.4) is a
  copy; writing it changes nothing the pipeline computes (only later hooks
  see it).
- Insert at custom seams; only `PRE_LOOP` and `POST_TIRE` exist (a
  `POST_TIRE` hook's `apply_pre_tire` call, v1.7.0, is a phase of that slot,
  not a third slot).
- Run during the air-mask branch with non-zero `N` (the air mask is applied
  before the hook sees the wheel).

For anything else, subclass the relevant strategy ABC instead.
