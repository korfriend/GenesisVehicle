"""genesis_vehicle.server — OSC-based general-purpose vehicle physics server.

Originally the UE integration server in genesis_unreal_plugin/, but folded
into the SDK as a subpackage so it works with any client that speaks the
protocol (OSC/UDP) — UE, Unity, custom visualizers, etc.

Run:
    python -m genesis_vehicle.server                # per-entity mode (heterogeneous/few vehicles, CPU)
    python -m genesis_vehicle.server --multi-env    # L3 batched mode (many same-URDF vehicles, GPU)

Extra dependencies: ``pythonosc`` (server-only — the SDK core does not need
it), ``psutil``, and ``trimesh`` for obstacle mesh preprocessing.

This ``__init__`` intentionally imports nothing — the SDK core
(``import genesis_vehicle``) must work in environments without pythonosc.
Import server modules explicitly, e.g.
``from genesis_vehicle.server import physics_server``.
"""
