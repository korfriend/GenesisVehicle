# Quickstart

The shortest working program. Drop a car, drive 10 seconds, print the final
position. Requires Genesis (CUDA) installed and a 4-wheel-car URDF whose
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
import genesis as gs
from genesis_vehicle import (
    VehiclePhysics, VehicleInputs, add_vehicle, car_4w_rwd_ackermann,
)

URDF = "<path to your car_raywheel.urdf>"

# 1. Genesis scene as usual.
gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/48, substeps=50))
scene.add_entity(gs.morphs.Plane())

# 2. SDK helper bundles URDF entity + wheel raycaster + preset cfg.
car, sensor, cfg = add_vehicle(scene, URDF, car_4w_rwd_ackermann)
scene.build(n_envs=1)

# 3. Construct VehiclePhysics and step.
physics = VehiclePhysics(scene, car, sensor, cfg, n_envs=1)
for step in range(480):                                       # 10s @ 48 Hz
    physics.step(VehicleInputs(throttle=0.5, brake=0.0, steer=0.0))
    scene.step()

print(car.get_pos()[0].cpu().numpy())
```

That's the whole API surface for a basic demo. The rest of the docs cover
what to reach for when you want custom topology, RL inputs, or your own
strategy.

`add_vehicle` is a thin helper (see `genesis_vehicle/scene_helpers.py`). The
hand-wired equivalent — useful when you need to customize URDF position,
material, or the raycaster's `max_range` per-side — is:

```python
from genesis_vehicle import WheelRayPattern, parse_urdf

car = scene.add_entity(gs.morphs.URDF(file=URDF, pos=(0, 0, 1.5)))
parsed = parse_urdf(URDF)
sensor = scene.add_sensor(gs.sensors.Raycaster(
    pattern=WheelRayPattern([w.position for w in parsed.wheels]),
    entity_idx=car.idx, max_range=20.0, return_world_frame=True,
))
cfg = car_4w_rwd_ackermann(URDF)
```

Both forms are first-class; pick whichever fits.

## Banner

On first `VehiclePhysics` construction, the SDK prints a single line:

```
[genesis_vehicle v0.4.0] Initialized: 4 wheels, Ackermann, RWD, Independent, n_envs=1, hooks=[RollingResistance, LowSpeedRegularizer]
```

Telling you, at a glance: version, topology, which strategies are wired up,
batch size, and which stability hooks are active.

## What's next?

- New to the concepts → [`concepts.md`](concepts.md).
- Want the full API → [`api-reference.md`](api-reference.md).
- Doing RL / MPPI or Real2Sim → [`stability-profiles.md`](stability-profiles.md).
