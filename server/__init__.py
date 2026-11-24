"""genesis_vehicle.server — OSC-based general-purpose vehicle physics server.

Originally the UE integration server in genesis_unreal_plugin/, but folded
into the SDK as a subpackage so it works with any client that speaks the
protocol (OSC/UDP) — UE, Unity, custom visualizers, etc.

Run:
    python -m genesis_vehicle.server                # per-entity mode (heterogeneous/few vehicles, CPU)
    python -m genesis_vehicle.server --multi-env    # L3 batched mode (many same-URDF vehicles, GPU)

Extra dependencies: ``pythonosc`` (server-only — the SDK core does not need
it), ``psutil``, and ``trimesh`` for obstacle mesh preprocessing.

This ``__init__`` imports only ``genesis`` and runs the SDK's required-engine
check (``genesis_vehicle._gs_compat.require_genesis``: genesis-world >= 1.5.0,
< 1.6 since 2.0.5), so ``python -m genesis_vehicle.server`` on an older engine
stops with ``GenesisVersionError`` before any socket is bound. It does NOT import
pythonosc — the SDK core (``import genesis_vehicle``) does not import this
package and must work in environments without pythonosc. Import server modules
explicitly, e.g. ``from genesis_vehicle.server import physics_server``.
(Through 2.0.4 this ``__init__`` imported nothing.)
"""

import genesis as _gs

from .._gs_compat import require_genesis as _require_genesis

_require_genesis(_gs)
