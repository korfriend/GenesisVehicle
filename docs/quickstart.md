# Quickstart

The shortest working program. Drop a car, drive 10 seconds, print the final
position. Requires Genesis (`genesis-world ≥ 1.0.0`, CUDA) installed and a 4-wheel-car URDF whose
suspension joints follow the SDK's naming convention (see
[`api-reference.md`](api-reference.md#4-urdf-utilities)).

> **TL;DR — runnable file:** [`../samples/quickstart.py`](../samples/quickstart.py)
> uses the bundled `samples/urdf/car_4w.urdf` so you can run it right after
> `pip install` without supplying your own URDF:
>
> ```bash
> python -m genesis_vehicle.samples.quickstart            # headless
> python -m genesis_vehicle.samples.quickstart --viewer   # side chase-cam
> ```

## Minimal example

```python
from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann

URDF = "<path to your 4-wheel-car URDF>"

# 0. Physics backend — process-global, set ONCE, before any scene. Default is
#    "cpu" (faster below ~100 envs; GPU is kernel-launch bound at small batch).
#    The renderer is separate (always GPU). Use "gpu" only for large-n_envs L3.
VehicleScene.init_backend("cpu")

# 1. VehicleScene owns gs.init / the scene(s) / build / step — the single entry point.
vs = VehicleScene(raycast_mode="single_scene", dt=0.025, substeps=10)
vs.add_ground_plane(friction=1.0)

# 2. Register a vehicle from a preset (it discovers wheels + builds the raycaster).
veh = vs.add_vehicle(URDF, preset=car_4w_rwd_ackermann, pos=(0, 0, 1.0))
vs.build()

# 3. Drive: set inputs (scalars or per-env (n_envs,) tensors), then step.
for step in range(480):                                  # 10 s @ 48 Hz
    veh.set_inputs(throttle=0.5, brake=0.0, steer=0.0)
    vs.step()

print(veh.get_pos()[0].cpu().numpy())
```

That's the whole API surface for a basic demo. `VehicleScene` is the recommended
entry point; the rest of the docs cover what to reach for when you want custom
topology, RL inputs, batching, or your own strategy.

`add_vehicle` takes a `preset=` (fn → cfg) **or** a pre-built `cfg=`, plus optional
`morph=` / `material=` / `surface=` / `raycaster_max_range=` for per-vehicle
customization — so you rarely need the raw Genesis API. To drop *below*
`VehicleScene` (its own `gs.Scene` + `VehiclePhysics` / `MultiVehiclePhysics`), see
the two-API-layers note in [`concepts.md`](concepts.md) and §1 of
[`api-reference.md`](api-reference.md).

## Banner

On first `VehiclePhysics` construction, the SDK prints a single line:

```
[genesis_vehicle v0.6.0] Initialized: 4 wheels, Ackermann, RWD, Independent, n_envs=1, hooks=[RollingResistance, LowSpeedRegularizer]
```

Telling you, at a glance: version, topology, which strategies are wired up,
batch size, and which stability hooks are active.

## What's next?

- New to the concepts → [`concepts.md`](concepts.md).
- Want the full API → [`api-reference.md`](api-reference.md).
- Doing RL / MPPI or Real2Sim → [`stability-profiles.md`](stability-profiles.md).
