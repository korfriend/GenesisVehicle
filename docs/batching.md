# Batching axes (L1 / L2 / L3)

The SDK has three orthogonal batching axes. Each compresses a Python
`for`-loop into a single GPU dispatch, but they target different kinds
of repetition. Knowing which one to reach for is the single biggest
performance decision in a `genesis_vehicle` codebase.

| Axis | Meaning | Built into | Headline win |
|---|---|---|---|
| **L1** | Wheels of ONE vehicle (4–10 wheels) | `VehiclePhysics` (always on) | Wheel loop → batched ops on `(*, n_wheels)` |
| **L2** | K vehicles of the SAME URDF/cfg in one Genesis env | `MultiVehiclePhysics` | K per-vehicle `step()` calls → 1 batched call per kind |
| **L3** | N parallel "universes" of the same scene | `scene.build(n_envs=N)` + `VehiclePhysics(n_envs=N)` | N independent rollouts → 1 batched call across envs |

They COMPOSE multiplicatively: L1 is always on; L2 and L3 can be stacked
via `MultiVehiclePhysics(scene, vehicles, n_envs=N)` for `N·K` total
vehicles in a single batched compute pipeline.

> **High-level shortcut.** You rarely construct these classes directly.
> `VehicleScene` (the recommended entry point) drives all three axes for you:
> `solver="batched"` (the **default**) groups same-kind vehicles into one
> `MultiVehiclePhysics` (L2), `n_envs=N` gives L3, and `veh.set_inputs(...)` accepts
> per-env `(n_envs,)` tensors — `add_vehicle(preset=…)` ×K then `vs.step()`. This
> page explains the underlying layer `VehicleScene` wraps; see the two API layers
> in [`concepts.md`](concepts.md).

---

## L1 — wheel batching (transparent)

Every `VehiclePhysics.step()` processes all wheels of the vehicle in
one batched tensor pipeline. Before v0.5.0 the SDK had a Python loop
over wheels (one CUDA kernel chain per wheel); v0.5.0 collapsed that
to a single `(n_envs, n_wheels)` tensor pipeline. There's no
user-facing knob — using `VehiclePhysics` at all gets you L1.

The compute steps all operate on `(*, n_wheels)`:

```
raycast       → distances     (n_envs, n_wheels)
suspension N  → N             (n_envs, n_wheels)
slip κ, α     → kappa, alpha  (n_envs, n_wheels)
tire force    → F_long, F_lat (n_envs, n_wheels)
omega update  → omega         (n_envs, n_wheels)
chassis force → total_F       (n_envs, 3)
```

The Pacejka tire model, all stability hooks, and the omega-coupling
strategy all process every wheel in one call. Nothing to tune.

### When L1 matters

- Always. Even a single vehicle benefits — for a tank with 10 wheels,
  pre-v0.5.0 you'd pay 10× kernel-launch overhead per step. Now it's 1×.

### See also

- [`samples/quickstart.py`](../samples/quickstart.py) — the smallest
  VehiclePhysics call (and therefore the smallest L1 use). Add
  `--viewer` for a side chase-cam.

---

## L2 — cross-vehicle batching in one env

> **Rule of thumb: for K (> 1) vehicles in one scene, reach for
> `MultiVehiclePhysics` — do NOT hand-roll a Python loop of K separate
> `VehiclePhysics` objects.**
>
> The manual loop
> ```python
> physics = [VehiclePhysics(scene, e, s, cfg) for (e, s) in vehicles]   # ✗ avoid
> for p, inp in zip(physics, inputs):
>     p.step(inp)
> ```
> is **not wrong** — it produces identical physics — but it leaves L2
> batching on the table: K separate compute calls, K separate state
> reads/writes, K Python `step()` round-trips. `MultiVehiclePhysics`
> ```python
> mphys = MultiVehiclePhysics(scene, vehicles)                          # ✓ prefer
> mphys.step(inputs)
> ```
> collapses those into one batched compute + one batched read/write per
> kind. Cleaner API and modestly faster (1.1–1.2× at K = 2–4; the gain is
> bounded because `scene.step()` itself isn't an L2 axis — see
> [What's NOT batched](#whats-not-batched)).
>
> **When the manual loop is still legitimate:** you need per-vehicle
> solver operations the batched path doesn't expose the same way —
> independent per-vehicle external forces / impulses, or per-vehicle
> teleport+reset mid-rollout. (This is exactly why
> `genesis_vehicle.server`'s default L2 (per-entity) mode still loops: the OSC
> protocol must apply `target_forces` / `AddWorldImpulse` to individual
> vehicles.) If you don't need those, prefer `MultiVehiclePhysics`.
>
> And for K = 1 it's the reverse — just use plain `VehiclePhysics`
> (L2 has nothing to batch; see [When L2 does NOT help](#when-l2-does-not-help)).

`MultiVehiclePhysics(scene, vehicles, n_envs=1)` processes K vehicles
that share one Genesis scene. It groups vehicles by `id(cfg)` (same
config object → same kind = same URDF + same preset + same wheel
overrides) and runs a single batched compute per kind:

- **Compute**: Pacejka, hooks, suspension calculation collapse from K
  separate calls to 1 batched call per kind operating on `(K_kind,
  n_wheels)` tensors.
- **State reads**: One `solver.get_links_{pos,quat,vel,ang}(K_idx)`
  call returns `(N=1, K_kind, ...)` for all K vehicles. No Python loop.
- **Force / torque writes**: One `solver.apply_links_external_force`
  call with K link indices applies forces to all K base links.
- **Sensor reads**: K small `sensor.read()` calls in a Python loop
  (one raycaster per vehicle — unavoidable per-vehicle I/O).
- **Visual writes** (only when `enable_visual_joint_sync=True`): since
  v1.0.15 the K same-kind writers collapse into **one** solver-level
  `set_dofs_position` over concatenated global dof indices
  (`KindVisualBatch` — one engine reset + FK pass for ALL K entities;
  measured 30 tanks: visual-write cost 12.4 → 3.4 ms/step). The per-entity
  `VisualJointSync.step()` loop (v0.7.16: one call per entity) remains as
  the layout-mismatch fallback. For headless / external rendering leave it
  off (closed-form `wheel_visual_transforms`, ~µs).

### When L2 matters

- **Traffic / multi-agent scenes**: K different vehicles at different
  positions in one Genesis scene.
- **Visual demos**: a top-down view of mixed vehicles ([`road_loop.py`](../samples/road_loop.py)).
- **Mixed-kind comparisons**: 4 FWD + 4 RWD + 4 AWD + 4 Truck in one scene,
  comparing their behaviors at a glance.

### When L2 does NOT help

- **Pure throughput on one vehicle kind**: just use L3 (`n_envs > 1`).
  L2 is for "same scene, different positions"; L3 is for "different
  scenes (universes), same positions".
- **K = 1 per kind**: L2 has setup overhead. With nothing to batch the
  default `VehiclePhysics` is faster — see [`samples/perf_multi_vehicle.py`](../samples/perf_multi_vehicle.py)
  showing 0.95× at K=1.

### Caveats / contracts

- **Vehicles of the same kind must share the SAME cfg INSTANCE** — the
  dispatcher groups by `id(cfg)`. Call `preset_fn()` once per kind and
  reuse the returned `VehicleConfig`, don't call it fresh per vehicle:

  ```python
  # WRONG — each call returns a fresh cfg, so K vehicles → K kinds.
  for i in range(K):
      _, _, cfg = add_vehicle(scene, urdf, preset_fn)
      vehicles.append((ent, sens, cfg))   # ← K different cfgs

  # RIGHT — one cfg per kind, shared across vehicles of that kind.
  cfg = preset_fn(urdf)
  for i in range(K):
      ent, sens, _ = add_vehicle(scene, urdf, preset_fn=None)
      vehicles.append((ent, sens, cfg))   # ← same instance
  ```

### See also

- [`samples/perf_multi_vehicle.py`](../samples/perf_multi_vehicle.py) — solver comparison sweep at varying K. **Headless by design** (bench).
- [`samples/road_loop.py`](../samples/road_loop.py) — visual demo with `--solver multi_batched`. Add `--viewer` for top-down camera.

---

## L3 — cross-env batching (`n_envs > 1`)

Built into Genesis. `scene.build(n_envs=N)` makes N parallel "universes"
of the same scene, and `VehiclePhysics(n_envs=N)` treats them as a
batched compute dimension. Every state read returns an `(N, ...)`
tensor; per-env inputs are accepted as `(N,)` tensors. One CUDA
dispatch chain handles all N envs.

This is the workhorse for RL / MPPI: you typically want many parallel
rollouts of the same vehicle setup with different action sequences.

### When L3 matters

- **RL training**: hundreds or thousands of parallel rollouts is the
  whole point of policy gradient + GPU throughput.
- **MPPI candidate evaluation**: 256 candidate action sequences scored
  in parallel.
- **Hyperparameter sweeps**: each env runs the scenario with a
  different cfg perturbation.

### Throughput characteristic

`ms / step` stays roughly constant from `n_envs = 4` upward — Genesis +
the SDK saturate the GPU, so each additional parallel env is nearly
free. The headline number on an RTX 5070 Laptop:

| n_envs | ms / step | env-steps / s | per env (μs) | gain |
|-------:|----------:|--------------:|-------------:|-----:|
|      1 |     26.3  |            38 |       26 315 | 1.0× |
|      4 |     36.8  |           109 |        9 196 | 2.9× |
|     16 |     37.3  |           429 |        2 330 | 11.3× |
|     64 |     37.9  |         1 691 |          592 | **44.4×** |

(See [`samples/perf_vectorization.py`](../samples/perf_vectorization.py) for the bench script.)

### Visualization caveat

All N envs simulate at the SAME world coordinates (they're parallel
universes, not spatial offsets). To SEE all N in one render, enable

```python
vis_options = gs.options.VisOptions(env_separate_rigid=True, ...)
scene.build(n_envs=N, env_spacing=(dx, dy), n_envs_per_row=K)
```

— Genesis lays the envs out in a grid for rendering only (physics
unchanged). See [`samples/multi_env_render.py`](../samples/multi_env_render.py).

### See also

- [`samples/batched_rollout.py`](../samples/batched_rollout.py) — minimal `n_envs > 1` call pattern. Add `--viewer` to lay all N envs out in a grid (env_separate_rigid).
- [`samples/perf_vectorization.py`](../samples/perf_vectorization.py) — scaling sweep. **Headless by design** (rendering would distort throughput numbers).
- [`samples/multi_env_render.py`](../samples/multi_env_render.py) — render all envs in a grid (dedicated visual demo, always renders).

---

## L2 × L3 combined (`MultiVehiclePhysics(n_envs=N)`)

For autonomous-driving simulation the natural pattern is "K vehicles
per scenario (ego + traffic agents) × N parallel scenarios (MPPI
candidates / RL rollouts)". This is L2 and L3 stacked.

Since v0.5.14, `MultiVehiclePhysics(scene, vehicles, n_envs=N)` does
exactly this: internal compute batch dim is `N·K`, with batched I/O
along both axes.

### API

```python
import torch
from genesis_vehicle import MultiVehiclePhysics, VehicleInputs, ...

# Build the scene as before: K vehicles of one or more kinds, share
# cfg instance per kind.
cfg = preset_fn(URDF)
vehicles = [(add_vehicle(scene, URDF, preset_fn=None, pos=spawn[k])[:2] + (cfg,))
            for k in range(K)]

# n_envs is now a constructor argument:
scene.build(n_envs=N)
mphys = MultiVehiclePhysics(scene, vehicles, n_envs=N)

# Per-vehicle inputs accept (N,) tensors per env, OR scalars (broadcast
# across all N envs). One step advances N·K vehicles total.
inputs = [
    VehicleInputs(throttle=torch.rand(N, device='cuda'),
                  brake=torch.zeros(N, device='cuda'),
                  steer=0.0)        # ← scalar broadcasts across envs
    for _ in range(K)
]
mphys.step(inputs)
scene.step()
```

### Measured stacking

L2 × L3 stack close to MULTIPLICATIVELY. From [`samples/perf_l2_l3_combined.py`](../samples/perf_l2_l3_combined.py)
on an RTX 5070 Laptop:

|  K |  N | total | ms/step | per veh (μs) | gain |
|---:|---:|------:|--------:|-------------:|-----:|
|  1 |  1 |     1 |   26.3  |       26 315 | 1.0× |
|  1 |  4 |     4 |   37.8  |        9 458 | 2.8× (L3 only) |
|  2 |  1 |     2 |   36.0  |       17 992 | 1.5× (L2 only) |
|  2 |  4 |     8 |   45.8  |        5 724 | **4.6×** (L2 × L3 ≈ 2.8 · 1.5 = 4.2 + GPU bonus) |

The combined gain (4.6×) is close to the product of the individual
gains (4.2×), with a small bonus from better GPU saturation at the
larger total batch.

### Use case (autonomous-driving MPPI)

Concretely: each MPPI scenario contains 1 ego + (K−1) traffic agents.
You evaluate N parallel scenarios per planning step (different ego
action sequences). With L2 × L3:

- N MPPI rollouts × K vehicles per rollout = `N·K` total
- One batched compute dispatch per vehicle kind
- Scales with GPU throughput, not vehicle count

This is the headline workflow the v0.5.14 release was built around.

### See also

- [`samples/perf_l2_l3_combined.py`](../samples/perf_l2_l3_combined.py) — 2D `(K, N)` scaling sweep. **Headless by design** (bench).
- [`samples/city_traffic_ego.py`](../samples/city_traffic_ego.py) — concrete autonomous-driving demo: 1 ego + 7 traffic on a 4-lane highway. `--n_envs N` enables L2 × L3, `--viewer` for top-down camera.

---

## Decision matrix

| Your scenario | Solver |
|---|---|
| 1 vehicle, parallel rollouts for RL / MPPI | `VehiclePhysics(n_envs=N)` (L3) |
| 1 vehicle, just visualization | `VehiclePhysics` (n_envs=1) |
| K vehicles in 1 visible scene (traffic demo, multi-kind comparison) | `MultiVehiclePhysics(scene, vehicles)` (L2) |
| **K vehicles × N parallel scenarios** | **`MultiVehiclePhysics(scene, vehicles, n_envs=N)`** (L2 × L3) — minimal example: [`samples/l2l3_minimal.py`](../samples/l2l3_minimal.py) |
| K=1, large N | Use plain `VehiclePhysics(n_envs=N)` — L2 only adds overhead with nothing to batch |

If you find yourself manually Python-looping over vehicles or envs,
you're probably leaving one of these axes on the table — check the
table above.

---

## Why two classes? (`VehiclePhysics` vs `MultiVehiclePhysics`)

A natural question: if `MultiVehiclePhysics` with K = 1 vehicle is just
`VehiclePhysics`, why are there two classes — couldn't one class branch
internally?

**They are NOT two parallel implementations.** `MultiVehiclePhysics`
(via its per-kind worker `MultiVehicleKindPhysics`) is built *on top of*
`VehiclePhysics`: it constructs a proto `VehiclePhysics(..., n_envs=N·K)`
and reuses it for all config resolution and batched tensor state
(`omega`, `wheel_meta`, `resolved`, …). The two classes differ only in
the **I/O layer**:

| | `VehiclePhysics` | `MultiVehiclePhysics` |
|---|---|---|
| Entities | ONE (1 base link, 1 sensor) | K (K base links, K sensors, grouped by cfg) |
| Force apply | 1 link | K links in one batched call |
| Sensor read | 1 raycaster | K raycasters stacked |
| Batch dim | `n_envs` | `n_envs · K` |

So with K = 1, `MultiVehiclePhysics` *is* functionally
`VehiclePhysics(n_envs=N)` plus a thin K-loop wrapper — your intuition is
correct.

**Why keep them separate rather than one branching class:**

1. **Common-case ergonomics.** ~90 % of use is ONE vehicle (your car,
   optionally batched over RL envs). `VehiclePhysics(scene, car, sensor,
   cfg, n_envs=N)` is the clean API for that. The multi-vehicle path
   needs a *list of `(entity, sensor, cfg)` tuples* and cfg-identity
   grouping — folding both input shapes into one constructor would make
   the simple case carry the complex case's surface area.
2. **No hot-path tax.** The single-vehicle path skips K-stacking of
   sensors, scatter-to-K-base-links, and per-kind dispatch — all pure
   overhead when K = 1.
3. **Composition over a god-class.** `MultiVehiclePhysics` *delegates* to
   `VehiclePhysics`; the physics math has one owner.

**Honest caveat (a real maintenance wart):** the `step()` *pipeline*
(suspension → tire → omega) is currently **mirrored** between
`VehiclePhysics.step` and `MultiVehicleKindPhysics.step` because the
multi-entity I/O is interleaved with the math. A physics fix (e.g. the
v0.6.0 `F_long` overshoot clamp) must be applied to both. A future
refactor could extract the shared pipeline into a pure function fed by an
I/O adapter — removing the duplication **without** merging the two
public classes. Tracked as a cleanup, not a correctness issue (the
grouping/dispatch bookkeeping is now unit-tested in
`tests/test_multi_vehicle_grouping.py`).

---

## What's NOT batched

- **`scene.step()` cost grows with entity count.** Genesis processes
  every entity in the scene per step (collision detection, integration).
  L2 batches the SDK-side compute, not Genesis's entity loop, so the
  per-step cost still grows roughly linearly with `K_total` (vehicles
  across all kinds in the scene). This is a Genesis-level constraint;
  L3 (parallel envs of the same scene) sidesteps it because the entity
  set is shared across envs.
- **Cross-kind compute**. Different URDFs / wheel counts / cfgs cannot
  share one batched call — they're dispatched as separate per-kind
  batches. With 4 kinds × 4 vehicles each, that's 4 batched calls per
  step (vs 16 in the per-vehicle pattern). Since v1.0.15 the cross-kind
  solver **I/O** IS batched, though: one combined state read (4
  `get_links_*` calls) + one combined force/torque apply (2 calls) for
  ALL kinds via `kind.step(state=…, defer_apply=True)` — was 6 solver
  entries per kind (measured 10 kinds × 1 tank: step 27.7 → 20.2 ms).
- **Sensor reads** in L2 are per-vehicle (one raycaster per vehicle —
  Genesis doesn't expose a multi-sensor batch API). Cost is small but
  scales with K (~0.5 ms at K=10, CPU).

---

## Performance numbers (reference, RTX 5070 Laptop)

| Setup | Sample | Result |
|---|---|---|
| L3 sweep | [`samples/perf_vectorization.py`](../samples/perf_vectorization.py) | 44× at n_envs=64 |
| L2 sweep | [`samples/perf_multi_vehicle.py`](../samples/perf_multi_vehicle.py) | 1.14× at K=2, 1.07× at K=4 (4-kind fleet) |
| L2 × L3 minimal | [`samples/l2l3_minimal.py`](../samples/l2l3_minimal.py) | shortest runnable L2 × L3 (K interacting × N scenarios, per-scenario control) |
| L2 × L3 combined | [`samples/perf_l2_l3_combined.py`](../samples/perf_l2_l3_combined.py) | 4.6× at K=2 N=4 (≈ product of L2 × L3 individual) |
| Multi-vehicle visual | [`samples/road_loop.py`](../samples/road_loop.py) | pre-1.0.15 figure: 6% faster than per-vehicle loop (16 vehicles, 4 kinds). v1.0.15 `KindVisualBatch` batches the visual writes themselves — 30 tanks CPU: 23.3 → 14.2 ms/step with VJS on |
| L2 × L3 ego+traffic visual | [`samples/city_traffic_ego.py`](../samples/city_traffic_ego.py) | 8 vehicles × 4 envs = 32 batched; 3.5× throughput vs n_envs=1 |

Re-run the samples on your machine — absolute numbers depend on GPU
and WSL/native setup. These tables were measured on the **GPU** backend;
since v1.0.14 every sample defaults to **CPU** (faster below ~100 envs —
GPU is kernel-launch bound at small batch), so pass `--gpu` to reproduce
them. For the consolidated CPU-vs-GPU decision guide (measured crossover
per batching axis, v1.1.6), see [`backends.md`](backends.md).
