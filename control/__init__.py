"""Vehicle control utilities on top of the physics SDK.

- :class:`SweepTable` — measured (input -> response) grid of one vehicle,
  with the inverse lookups a controller needs (numpy-only, no Genesis).
- :class:`PathFollower` — path (waypoints + signed target speeds) ->
  per-step ``(throttle, steer, brake)`` via sweep-table inversion.
- :func:`extract_state` / :func:`extract_state_from_arrays` — chassis-state
  extraction helpers (Genesis entity / any simulator).
- ``python -m genesis_vehicle.control.sweep_measure`` — measure the sweep
  CSV for a new (URDF, preset, config) triple (needs Genesis).

See ``docs/path-following.md`` for the full pipeline.
"""
from .sweep import SweepTable, SWEEP_COLUMNS
from .path_follower import PathFollower, extract_state, extract_state_from_arrays

__all__ = [
    "SweepTable", "SWEEP_COLUMNS",
    "PathFollower", "extract_state", "extract_state_from_arrays",
]
