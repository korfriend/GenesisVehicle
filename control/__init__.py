"""Vehicle control utilities on top of the physics SDK.

The default path-following stack since 2.0.0 is a sweep table + the MPC layer;
:class:`DifferentiablePlant` is the no-measurement option. (Through 1.8.0
this docstring called ``DifferentiablePlant`` "the default inverse plant
(v1.3.0)"; that stopped being true in 2.0.0 and was corrected.)

- :class:`PathFollower` — path (waypoints + signed target speeds) ->
  per-step ``(throttle, steer, brake)``. Stage 1: the MPC layer on forward
  blocks (needs ``mpc_params``; optional dependencies, see
  ``requirements-mpc.txt``) and the 1.x PD law elsewhere, or the PD law alone
  with ``mpc_model="off"``. Stage 2: either inverse plant, in the same slot.
- :class:`SweepTable` — a measured (input -> response) grid of one vehicle,
  loaded from CSV (numpy-only, no Genesis, so a simulator-free runtime can
  still path-follow).
- :class:`DifferentiablePlant` — differentiates the SDK's own ray-wheel force
  model with ``torch.autograd`` and inverts the resulting 2x2 Jacobian every
  step. No sweep measurement, no CSV.
- :class:`FleetFollower` — N PD followers driven off ONE batched plant.
- :class:`MPCVehicleParams` / :class:`MPCGains` / :func:`load_mpc_params` —
  the MPC layer's measured vehicle record and cost weights.
- :func:`identify_mpc_params` — measure that record (needs Genesis); CLI
  ``python -m genesis_vehicle.control.identify``. Resolved lazily (the module
  is also the ``-m`` entry point).
- :func:`extract_state` / :func:`extract_state_from_arrays` — chassis-state
  extraction helpers (Genesis entity / any simulator).
- ``python -m genesis_vehicle.control.sweep_measure`` — measure the sweep
  CSV for a new (URDF, preset, config) triple (needs Genesis).

See ``docs/path-following.md`` for the full pipeline.
"""
from .sweep import SweepTable, SWEEP_COLUMNS
from .plant import DifferentiablePlant, PlantMassProperties, plant_mass_properties
from .path_follower import (PathFollower, FleetFollower, FollowerPlan,
                            extract_state, extract_state_from_arrays)
from .mpc import MPCGains, MPCVehicleParams, load_mpc_params

__all__ = [
    "DifferentiablePlant", "PlantMassProperties", "plant_mass_properties",
    "SweepTable", "SWEEP_COLUMNS",
    "PathFollower", "FleetFollower", "FollowerPlan",
    "extract_state", "extract_state_from_arrays",
    "MPCVehicleParams", "MPCGains", "load_mpc_params", "identify_mpc_params",
]


def __getattr__(name):
    # Lazy: `control.identify` is also run as `python -m`; importing it here
    # eagerly would make runpy warn and execute it twice.
    if name == "identify_mpc_params":
        from .identify import identify_mpc_params
        return identify_mpc_params
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
