# Differentiable control — source of truth

Status of everything the SDK does with derivatives, what was measured to get
there, and what is still open. Written 2026-08-20 against SDK v1.4.0 /
genesis-world 1.3.3, to be picked up in a separate research session.

| abbr | meaning |
|---|---|
| autodiff | automatic differentiation (`torch.autograd`) |
| tape | the input set autograd records for a backward pass |
| inverse plant | what turns a desired `(a, ω_z)` into a `(throttle, steer)` command |
| J | 2×2 Jacobian `d(a, ω_z) / d(throttle, steer)` |
| a / ω_z | body-longitudinal acceleration (m/s²) / chassis yaw rate (rad/s) |
| horizon | prediction unroll length, in sim steps |
| member | one (vehicle, env) pair a plant serves |
| M | number of members |
| L2 / L3 | batching axes: K interacting vehicles / `n_envs` parallel scenarios |
| MOI | moment of inertia |
| DOF | degree of freedom |
| free joint | the 6-DOF joint attaching a base link to the world |
| wrench | force + torque pair |
| dead zone | operating point where the model's derivative is identically zero |
| secant | finite difference over a control-sized step, as opposed to a derivative |
| CoM | centre of mass |
| SFL | `StaticFrictionLock` stability hook |

---

## 1. The question this answers

`PathFollower` needs the inverse of the vehicle's input→response map. Until
v1.3.0 the only answer was `SweepTable`: a 5-D grid measured offline by tens
of thousands of open-loop rollouts, written to CSV, valid for exactly one
(URDF, preset, overrides) triple at one dt. Every plant change silently
invalidated it — including the SDK's own v1.2.8 `i_wheel` fix, which moved
tracked-vehicle response by up to 1.96×.

The goal was to replace the measurement with a derivative taken online.

---

## 2. The load-bearing finding: Genesis autodiff cannot serve a ray-wheel vehicle

Genesis ships a differentiable rigid solver (`SimOptions(requires_grad=True)`).
**It cannot produce `∂(response)/∂(command)` for this SDK's vehicles.**

The taped input set is exactly, from `RigidEntity.process_input_grad` and
`_tgt_keys`:

| taped input | gradient path |
|---|---|
| `set_pos` | yes |
| `set_quat` | yes |
| `set_dofs_velocity` | yes |
| `control_dofs_force` | yes |
| `control_dofs_position` / `_velocity` / `_position_velocity` | **raises** — "Gradients with respect to PD control targets are not supported yet. Use 'control_dofs_force' for differentiable control inputs." |
| `apply_links_external_force` / `_torque` | **not on the tape at all** |

A ray-wheel vehicle reaches the solver *exclusively* through
`apply_links_external_force` + `_torque`: the SDK has no wheel joints to
motor, because the wheels are raycast points, not simulated bodies.

Measured on genesis-world 1.3.3 (scratch probes, both re-runnable):

| channel | result of `scene.backward()` |
|---|---|
| `apply_links_external_force`, leaf tensor with `requires_grad` | `grad is None` |
| `control_dofs_force` on a free-joint DOF | finite, correct (`4e-06` for the probe) |

This is why the reference code the team supplied *does* work: its vehicle
(`blender_car_suspension.urdf`) is a joint-motor car driven by
`control_dofs_force` on rear-wheel and front-steer joints, with contact
handled by Genesis. Its PD steering servo is computed in Python precisely so
the result enters through the differentiable channel — which is what the
Genesis error message above recommends.

### 2b. There IS a route, and it is verified

`control_dofs_force` on the base link's free-joint 6 DOFs is **the same
physical input** as `apply_links_external_force` + `_torque` at the link
origin. Measured on the reference tank (CoM 0.69 m above the base origin, so
a pure horizontal force at the origin produces a pitch moment that
discriminates the two reference points), `F = (1e5, 0, 0)` held 3 steps:

```
apply_links_external_force   vel=[0.056346, 0, -2.6e-05]  ang=[0, -0.007574, 0]
control_dofs_force (free 6)  vel=[0.056346, 0, -2.6e-05]  ang=[0, -0.007574, 0]
-> identical: True
```

So routing the SDK's chassis wrench through the taped channel is a **drop-in
swap of the apply site, no wrench transfer needed**. See §7 for why it was not
taken, and what it would buy.

---

## 3. What was built instead (v1.3.0)

`DifferentiablePlant` differentiates **the SDK's own force model**, not the
simulator.

An SDK physics step has two halves:

```
[1] SDK force model (python/torch)        [2] Genesis solver (C++/kernel)
    _pipeline.compute_wheel_step
    raycast → suspension N → slip →   ──▶ apply_links_external_force ──▶ chassis
    tyre force → wheel ω → chassis wrench                                integration
```

[2] is untouched — no `requires_grad` scene, simulation bit-identical to
v1.2.8 (verified: quickstart final pose and the sweep-table demos reproduce
exactly). Only [1] is differentiated, and [2]'s role inside the *prediction*
is played by a small planar integrator:

| part of the prediction | status |
|---|---|
| suspension N, slip, Pacejka, friction circle, wheel ω, coupling, hooks, aero drag | **the original — the same function the sim calls**, not a copy |
| chassis rigid-body integration | **stand-in**: planar surge/sway/yaw Newton-Euler on the vehicle's real composite mass and yaw MOI about the base-link origin |
| heave / pitch / roll rates, ray distances (hence normal loads), terrain | **frozen** over the horizon |

Per control step: snapshot the live state → unroll `horizon` steps → read off
`a = Δv_long/(H·dt)` and mean ω_z (the two quantities the sweep table
tabulates) → `torch.autograd` for J → Newton step with a backtracking line
search, warm-started from the previous command.

### Naming caution

"the differentiable solver" is the wrong phrase for what exists. The
differentiable object is a *replica of the vehicle's dynamics* sharing the
hard half with the simulator, not the simulator.

---

## 4. Measured results

### Prediction accuracy vs the real simulator

Command held for `horizon` steps from the live state, sequential (no scene
reset — the SDK's per-wheel state does not survive one):

| vehicle | horizon | max \|Δa\| | max \|Δω_z\| |
|---|---|---|---|
| `tank_ref`, 10 wheels, 58.7 t | 4 | 0.177 m/s² | 0.0014 rad/s |
| `tank_ref` | 8 | 0.149 m/s² | 0.0058 rad/s |
| `car_4w`, 1.33 t | 4 | 0.103 m/s² | 0.0029 rad/s |

Autodiff J vs central finite differences of the same unroll: ≤ 6e-4 relative.
Composite mass properties vs the entity's own 6×6 mass matrix: 6 digits
(m = 58700.0, izz = 357640.2, m·cx = 606.6, both routes).

### Closed loop vs the measured table

Same courses, same tuning, bundled 10-wheel tank:

| plant | forward course deviation | reverse-into-bay cusp |
|---|---|---|
| `SweepTable` (78,302 measured rollouts) | mean 0.384 m, max 1.034 m | 1.50 m / 0.33 rad PASS |
| `DifferentiablePlant` (no measurement) | mean 0.370 m, max 0.917 m | 1.50 m / 0.32 rad PASS |

### Cost, and the horizon knob

One `solve()`, 10-wheel tank, CPU. Linear in `horizon`:

| horizon | predicted ahead | per solve | vs a 25 ms control step |
|---|---|---|---|
| 2 | 0.05 s | 9.7 ms | 39 % |
| 3 | 0.075 s | 15.3 ms | 61 % |
| **4** (default) | **0.1 s** | **20.1 ms** | **80 %** |
| 6 | 0.15 s | 29.6 ms | 118 % |
| 8 | 0.2 s | 39.1 ms | 156 % |
| 12 | 0.3 s | 56.1 ms | 224 % |

For scale: the sim step is ~10 ms, a `SweepTable` inversion of both channels
0.16 ms.

`horizon=4` is the default because 8 measured both **slower and looser**
(0.433 m vs 0.370 m mean deviation): a controller running every step wants the
near-term response, and a longer horizon drifts further from the frozen-load
assumption. This is one vehicle on one course — see §8.

### Fleet batching (v1.4.0)

The unroll's batch axis carries `M members × candidates`. Dispatch-bound at
these sizes, so widening is nearly free — forward unroll, `horizon=4`:

| batch | ms/call | ms per item | vs B=1 |
|---|---|---|---|
| 1 | 6.83 | 6.83 | 1.00× |
| 8 | 6.93 | 0.87 | 1.01× |
| 32 | 8.34 | 0.26 | 1.22× |
| 128 | 8.57 | 0.067 | 1.25× |

End to end, control cost per step:

| K tanks | K separate plants | 1 batched plant | per vehicle |
|---|---|---|---|
| 8 | 200.7 ms | 26.2 ms (7.7×) | 25.1 → 3.27 ms |
| 30 | 743.9 ms | 28.1 ms (**26.5×**) | 24.8 → 0.94 ms |

Equivalence at K=30 over 40 steps vs per-vehicle plants: max |Δthrottle|
3.3e-05, max |Δsteer| 1.8e-04, final position spread 1.5e-05 m — float32
noise.

The Jacobian batches for free: members are independent in the unroll, so
`d(Σᵢ aᵢ)/d(thrⱼ)` is member j's own gradient, and one backward per row covers
the fleet.

---

## 5. Four things that had to be fixed or worked around

Each was found by actually running the thing; none were predicted.

**1. Two tyre models were not autograd-safe.** `PacejkaAnisotropic` and
`CoulombIsotropic` both took `sqrt` of a quantity that is exactly zero at zero
slip, where the derivative is infinite — and `torch.where` backpropagates
through *both* branches, so every non-sliding wheel produced NaN gradients.
Fixed with a double-`where` (Pacejka) and a clamp inside the `sqrt`
(Coulomb). Forward values unchanged, verified by identical demo trajectories.

**2. Genesis's `transform_by_quat` raises on gradient-carrying input.** It is a
TorchScript kernel writing into a preallocated buffer with `copy_`, so a
gradient input hits *"a leaf Variable that requires grad is being used in an
in-place operation"*. Any steered vehicle triggers it (wheel-frame axes depend
on the steer command); a skid-steer never did, because its per-wheel steer is
a constant zero. `_pipeline._rotate_by_quat` keeps the fast kernel for the
ordinary path and takes the functional quaternion rotation only when a
gradient is flowing.

**3. The plant has genuine dead zones.** The v0.6.0 `F_long` overshoot clamp
pins tyre force to exactly zero whenever friction could not carry the wheel
past rolling within one step. The derivative there is *honestly* zero, and a
zero Jacobian reads as "no authority, do not move" — which froze the
controller mid-manoeuvre on the reverse course (tank circling at a constant
0.76 rad/s with the command frozen). Where a Jacobian column vanishes the
plant now re-estimates it with a one-sided **secant** over `probe_delta` (0.2)
of command, big enough to step out of the zone. The derivative is a local
object; the secant over a control-sized step is not.

**4. The follower must report what it actually applied.** The Jacobian
linearises about the warm-start command. `PathFollower` clamps steer to
`steer_cap` and overrides throttle outright with its low-speed KICK, so a
stale warm start linearised a vehicle state that never existed. Fixed by
reporting on *every* exit — including the DONE and cusp-brake early returns —
and by tightening the plant's steer range to the follower's own cap.

Related design trap, documented rather than fixed: **a coupled 2×2
least-squares solve will spin the vehicle to make it go faster.** Near rest a
skid-steer has `|∂a/∂steer| > |∂a/∂throttle|`, so a least-squares inverse
"fixes" a speed error by throwing the vehicle into a spin. The default is
channel-diagonal (throttle→a, steer→ω_z); the cross terms come back safely
through the chord iterations, which re-evaluate the full coupled response.

---

## 6. What exists in the code

| thing | where |
|---|---|
| `DifferentiablePlant` | `control/plant.py` |
| `PlantMassProperties`, `plant_mass_properties(entity)` | `control/plant.py` |
| `STATE_SOURCE_SURFACE`, `is_state_source` — the hook for a non-Genesis state feed | `control/plant.py` |
| `FleetFollower`, `FollowerPlan`, `PathFollower.plan/finish` | `control/path_follower.py` |
| the differentiated force model | `_pipeline.compute_wheel_step` (unchanged except `_rotate_by_quat`) |
| `build_wheel_meta(resolved, device, dtype)` — per-wheel tensors without an entity | `core.py` |
| docs | `docs/path-following.md` §1 |
| tests (33, Genesis-free) | `tests/test_differentiable_plant.py` |
| demos | `path_follow_demo.py`, `path_follow_reverse_demo.py`, both `--plant {jacobian,sweep}` |

`SweepTable` is unchanged and still supported. It remains the right tool for a
hard real-time loop (0.16 ms, stateless, shareable across every vehicle of a
kind) and the **only** option for a controller in a process with no simulator
— the Jacobian plant reads the live vehicle by construction.

---

## 7. Open: routing the wrench through the taped channel

§2b established the swap is physically exact. What it would buy is **the one
"stand-in" row of §3's table** — real chassis integration instead of the
planar model — by making the whole chain
`SDK force model → control_dofs_force → Genesis solver → qpos` one graph.

Costs and constraints, all confirmed by reading 1.3.3:

- `requires_grad=True` is a **scene-construction** option, not a runtime
  toggle, and it bans hibernation (the OSC server uses it), forces the
  `approximate_implicitfast` integrator, rejects torsional/rolling friction,
  noslip and the elliptic friction cone, disables contact islands and takes a
  dense Hessian.
- The tape lives on the **entity** (`RigidEntity._tgt_keys`), so the
  solver-level batched apply that `MultiVehicleKindPhysics` uses today would
  have to become K entity calls — eroding the L2 advantage.
- The raycast sensor is outside the graph either way, so `∂/∂(suspension)`
  stays cut; this route does not improve that.
- Per-step cost would resemble the team's reference structure: `scene.reset()`
  plus a forward and backward unroll per control step.

Shape if taken: `VehicleScene(..., differentiable=True)` → sets
`SimOptions(requires_grad=True)` and flips `VehiclePhysics.step`'s apply site.
Default `False` = today's behaviour, bit-identical.

**Judgement as of v1.4.0:** the stand-in's error is ≤ 0.18 m/s² and ≤ 0.0014
rad/s, so the row it would replace is not currently the limiting factor. Worth
revisiting if a scenario makes the frozen heave/pitch/roll actually bite — a
steep or badly broken surface is the obvious candidate.

---

## 8. Open questions worth a research session

Ordered roughly by expected value.

1. **Where does the planar stand-in actually break?** The whole accuracy claim
   rests on flat-ish ground. Sweep slope and bump severity until |Δa| stops
   being small, and see whether the failure is the frozen normal loads, the
   frozen pitch/roll rates, or the flat-terrain assumption. That measurement
   decides whether §7 is worth doing.

2. **Is `horizon=4` right for anything other than this tank?** The default
   comes from one vehicle on one course. Vehicles with slow tyre transients
   (soft tyres, very large `i_wheel`) should want longer. A per-vehicle
   auto-tune — pick the shortest horizon whose prediction still tracks the sim
   within a tolerance — is a small, self-validating feature.

3. **Can the dead zones be removed rather than probed around?** The `F_long`
   overshoot clamp is a discrete guard on a continuous model. A smooth
   formulation with the same stability property would make the derivative
   informative everywhere and delete the secant rescue.

4. **Gradients through the raycast.** Wheel-ground distance is a sensor read,
   so `∂N/∂state` is cut in every variant considered here. On a heightfield
   the distance IS differentiable in closed form. That would make load
   transfer and terrain response part of the Jacobian.

5. **What else is the Jacobian good for?** It is currently consumed by one
   Newton step. The same object supports: a stability/controllability
   diagnostic per state (the plant already reports `last_singular`), an MPC
   horizon rather than a one-shot inverse, and gradient-based tuning of the
   preset's own parameters (mu, k_susp, i_wheel) against a recorded
   trajectory — i.e. system identification against real vehicle logs, which
   is the natural sim2real use.

6. **GPU.** Every cost figure here is CPU, where the unroll is dispatch-bound.
   On GPU the batching arithmetic changes shape entirely and a fleet of
   hundreds may behave very differently.

7. **Does the fleet plant hold up in an L3 scene?** Members are the cartesian
   product of vehicles × envs and the row mapping is implemented, but every
   measurement so far is `n_envs=1`. Worth an explicit L2×L3 run.

---

## 9. Reproducing the measurements

Every number above came from a script in this session's scratchpad; the ones
worth keeping are listed here so they can be rebuilt.

| measurement | how |
|---|---|
| tape contents | read `RigidEntity.process_input_grad` / `_tgt_keys` |
| `apply_links_external_force` grad is None | scene with `requires_grad=True`, leaf tensor in, `scene.backward(qpos[0])`, read `.grad` |
| the two wrench channels are identical | tank with an offset CoM, same force via both, compare `get_vel` / `get_ang` |
| prediction vs sim | hold a command `horizon` steps from the live state, sequentially, compare `(Δv_long/(H·dt), mean ω_z)` |
| autodiff vs finite differences | `plant.jacobian()` against central differences of `plant.predict()` on the same `snap` |
| closed loop | `path_follow_demo --plant {jacobian,sweep}`, which reports path deviation |
| cost vs horizon | time `plant.solve()` at each horizon |
| batch scaling | time `plant.predict(thr_B, steer_B, snap)` at increasing B |
| fleet equivalence | run K tanks twice from the same state, per-vehicle plants vs one batched plant, diff the command streams |
