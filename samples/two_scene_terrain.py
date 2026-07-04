"""Deprecated alias — renamed to ``dual_scene_terrain`` in v1.1.0.

Official terminology: the wheel-raycast subsystem (``raycast_mode``) has two
modes, ``dual_scene`` (default) and ``single_scene``; "two-scene" in older
notes meant ``dual_scene``. This shim keeps
``python -m genesis_vehicle.samples.two_scene_terrain`` working.
"""

import sys

from .dual_scene_terrain import main   # noqa: F401  (re-export)

if __name__ == "__main__":
    print("[genesis_vehicle] NOTE: samples.two_scene_terrain was renamed to "
          "samples.dual_scene_terrain (v1.1.0) — forwarding.", file=sys.stderr)
    main()
