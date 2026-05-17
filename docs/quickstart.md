# Quickstart

The shortest working program. Drop a car, drive 10 seconds, print the final
position. Requires Genesis (CUDA) installed and a 4-wheel-car URDF whose
suspension joints follow the SDK's naming convention (see
[`api-reference.md`](api-reference.md#4-urdf-utilities)).

## Minimal example

```python
import genesis as gs
from genesis_vehicle import (
    VehiclePhysics, VehicleInputs, WheelRayPattern,
    car_4w_rwd_ackermann, parse_urdf,
)

URDF = "<path to your car_raywheel.urdf>"

# 1. Genesis scene as usual.
gs.init(backend=gs.gpu)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1/48, substeps=50))
scene.add_entity(gs.morphs.Plane())
car = scene.add_entity(gs.morphs.URDF(file=URDF, pos=(0, 0, 1.5)))

# 2. Wheel raycaster — positions come from the URDF.
parsed = parse_urdf(URDF)
sensor = scene.add_sensor(gs.sensors.Raycaster(
    pattern=WheelRayPattern([w.position for w in parsed.wheels]),
    entity_idx=car.idx, max_range=20.0, return_world_frame=True,
))
scene.build(n_envs=1)

# 3. Pick a preset and step.
cfg = car_4w_rwd_ackermann(URDF)
physics = VehiclePhysics(scene, car, sensor, cfg, n_envs=1)

for step in range(480):                                       # 10s @ 48 Hz
    physics.step(VehicleInputs(throttle=0.5, brake=0.0, steer=0.0))
    scene.step()

print(car.get_pos()[0].cpu().numpy())
```

That's the whole API surface for a basic demo. The rest of the docs cover
what to reach for when you want custom topology, RL inputs, or your own
strategy.

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
