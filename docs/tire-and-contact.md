# Tire models & ground-contact mechanisms

Two **independent** axes are often confused because of project history. This
doc separates them, gives the standard names, and states the recommended
combination.

## Glossary

| Term | Meaning |
|---|---|
| Pacejka (파세카) | Magic-formula empirical tire model |
| Coulomb (쿨롱) | Simple μ·N friction law |
| slip curve | tire-force vs slip curve (rise → peak → saturation) |
| contact patch | the tire's footprint on the ground |
| N | road normal force |
| μ | friction coefficient |

---

## The two axes are orthogonal

|  | **collider-vs-collider contact** | **ray-hit (analytic force)** |
|---|---|---|
| friction / tire | engine built-in Coulomb contact friction | `CoulombIsotropic` **or** `PacejkaAnisotropic` |
| cost | per-step cylinder-vs-ground collision test | one ray + analytic formula |

- **Axis A — ground detection**: collider contact ↔ ray-hit ↔ shapecast ↔ contact-patch
- **Axis B — tire friction model**: Coulomb ↔ Pacejka

This SDK **fixes Axis A to ray-hit** and lets you pick Axis B (Coulomb or
Pacejka) on top. The wheel has no collision geometry for tire forces; a
downward ray measures ground distance and the SDK applies suspension +
tire forces analytically (`apply_links_external_force`).

> **History note.** The old "Coulomb = always test cylinder-vs-ground
> collision" memory refers to Axis A being *collider contact*, where
> friction came from the engine's built-in Coulomb contact solver — heavy
> and jittery. The migration replaced that with ray-hit + an analytic tire
> model (Pacejka), and kept an analytic `CoulombIsotropic` as a baseline.
> Today's `CoulombIsotropic` is **ray-hit-based**, NOT the old collider
> approach — it consumes the same `N` (from ray-hit suspension compression)
> and slip (from ray-hit wheel kinematics) as Pacejka; only the final
> friction formula is simpler. There is no collider-contact tire path in
> this SDK.

---

## Axis B — which tire model? → **Pacejka** (effectively better on every axis)

| Criterion | Pacejka | Coulomb |
|---|---|---|
| Realism | **Industry standard** — slip-curve peak & saturation, separate long/lat, friction-circle (combined slip) | Crude — μN always opposes slip, no peak/saturation, single isotropic μ, unrealistic combined slip |
| Compute cost | slightly heavier (sin/atan ×6 + friction-circle) | slightly lighter |
| **Effective wall-clock** | **practically identical** — tire is a tiny fraction of step cost (dominated by `scene.step()` + state I/O) | identical |
| Numerical stability | smooth saturation → stable | discontinuity at zero slip → **low-speed chatter** risk |

Coulomb's only edge (marginally lighter) is negligible in the full
pipeline, and its low-speed stability is *worse*. So Coulomb is not a
performance win in practice. **Use Pacejka** (all bundled presets do);
reach for `CoulombIsotropic` only as a comparison baseline, for debugging,
or when you explicitly want simple predictable friction.

Note: `CoulombIsotropic` uses a single μ (`wheel_meta.mu_long`) — `mu_lat`
is ignored. If you need distinct longitudinal/lateral friction, use
Pacejka.

Swap is config-level:

```python
from genesis_vehicle import car_4w_rwd_ackermann, CoulombIsotropic
cfg = car_4w_rwd_ackermann(URDF)
cfg.tire = CoulombIsotropic()        # default: PacejkaAnisotropic()
```

Both run on the same shared `_pipeline.compute_wheel_step`, so the choice
applies identically to `VehiclePhysics` and `MultiVehiclePhysics`.

---

## Axis A — ground-detection mechanisms (standard names)

This SDK uses the first one. The others are listed for context / future work.

| Mechanism | Standard name | Notes |
|---|---|---|
| **Single downward ray** (current) | **Raycast wheel / ray-cast suspension** (a.k.a. single-point raycast contact) | Lightest, most stable. Used by Unity *Wheel Collider*, UE *Chaos Vehicles*. Limitation: one contact point can step over a thin curb/edge. |
| Multiple rays / shape sweep | **Shapecast (sweep) wheel / spherecast suspension** | Approximates the contact patch → catches edges a single ray misses. Slightly heavier. |
| Real collision shape | **Rigid contact wheel / collider-based wheel** | Wheel is a cylinder collider; engine contact solver resolves penetration + Coulomb friction. General (multi-point, edges) but heavy, jittery, hard to tune (the *old* approach). |
| Discretized footprint | **Contact-patch / brush model** (e.g. FTire, brush) | Splits the patch into elements for high fidelity. Very expensive; real vehicle-dynamics / R&D use. |

The SDK's term for this is **"ray-cast wheel + Pacejka tire"** — the 5-step
ray-wheel pipeline (raycast → suspension N → slip → tire force → wheel ω +
chassis force).

---

## Recommended combination

**Pacejka + raycast wheel** is the standard sweet spot for realism,
performance, and stability — and is what the presets ship. If you later
need higher accuracy over curbs / rough terrain, upgrade *Axis A* to a
shapecast (sweep) wheel while keeping Pacejka on *Axis B*; that does not
require touching the tire model.

### See also
- [`batching.md`](batching.md) — how the pipeline (incl. `resolved.tire(...)`) is vectorized.
- [`physics-contracts.md`](physics-contracts.md) — friction-circle clamp, N clamp, sign conventions.
- `tire_models/pacejka.py`, `tire_models/coulomb.py` — the two implementations.
