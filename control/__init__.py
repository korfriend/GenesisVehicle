"""Vehicle control utilities on top of the physics SDK.

- :class:`DifferentiablePlant` — the **default** inverse plant (v1.3.0):
  differentiates the SDK's own ray-wheel force model with ``torch.autograd``
  and inverts the resulting 2x2 Jacobian every step. No sweep measurement, no
  CSV, and it tracks plant edits (mass, friction, ``i_wheel``) automatically.
- :class:`SweepTable` — the pre-v1.3.0 inverse plant: a measured
  (input -> response) grid of one vehicle, loaded from CSV (numpy-only, no
  Genesis, so a simulator-free runtime can still path-follow).
- :class:`PathFollower` — path (waypoints + signed target speeds) ->
  per-step ``(throttle, steer, brake)``. Takes either plant in the same slot.
- :class:`FleetFollower` — N followers driven off ONE batched plant, so a
  fleet pays for one inversion instead of N.
- :func:`extract_state` / :func:`extract_state_from_arrays` — chassis-state
  extraction helpers (Genesis entity / any simulator).
- ``python -m genesis_vehicle.control.sweep_measure`` — measure the sweep
  CSV for a new (URDF, preset, config) triple (needs Genesis). Only needed
  for the SweepTable path.

See ``docs/path-following.md`` for the full pipeline.
"""
from .sweep import SweepTable, SWEEP_COLUMNS
from .plant import DifferentiablePlant, PlantMassProperties, plant_mass_properties
from .path_follower import (PathFollower, FleetFollower, FollowerPlan,
                            extract_state, extract_state_from_arrays)

__all__ = [
    "DifferentiablePlant", "PlantMassProperties", "plant_mass_properties",
    "SweepTable", "SWEEP_COLUMNS",
    "PathFollower", "FleetFollower", "FollowerPlan",
    "extract_state", "extract_state_from_arrays",
]
