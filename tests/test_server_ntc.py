"""``--no-target-collision`` (2.0.5): works for primitive /Init/Target targets,
refused for URDF vehicle targets and for ``--multi-env``.

Through 2.0.4 the flag wrote ``geom._contype`` / ``_conaffinity`` after the
entities existed, which no genesis-world >= 1.4.0 reads, so it did nothing
(CHANGELOG 1.6.9 section 12, a code-read). Since 2.0.5:

* primitive targets carry ``contype=0b10, conaffinity=0b01`` on their morph
  (public ``Primitive`` fields), so target-vs-target pairs are filtered and
  target-vs-everything-else pairs are kept;
* a URDF target cannot be masked through public API (``gs.morphs.URDF`` has no
  such fields on 1.5.0), so the L2 server refuses the flag (rc 2) as soon as
  the URDF init is known -- right after the OSC init wait, and again at the
  target-build point in case ``/Genesis/Vehicle/Init`` arrives late;
* ``--multi-env`` refuses it at argument parsing.

The physical behaviour (targets pass through each other, still rest on the
ground, still stop at obstacles and at a URDF chassis) is the AC-NTC harness's
job; this file pins the construction and the refusal paths with a fake OSC
manager, no UE client.
"""

import ast
import os
import sys

import pytest

pytest.importorskip("genesis")
pytest.importorskip("pythonosc")

import genesis as gs

from genesis_vehicle.server import physics_server as ps

SDK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# --- the morph carries the masks ---------------------------------------------

@pytest.mark.parametrize("obs_type, cls", [(1, "Box"), (2, "Sphere"), (3, "Cylinder"), (9, "Box")])
def test_primitive_target_morph_carries_the_ntc_masks(obs_type, cls, cpu_genesis_for_ntc):
    info = dict(pos=[0, 0, 1], quat=[1, 0, 0, 0], type=obs_type, scale=[0.4, 0.4, 0.4])
    m = ps.generic_target_morph(info, no_target_collision=True)
    assert type(m).__name__ == cls
    assert (m.contype, m.conaffinity) == (0b10, 0b01)
    m0 = ps.generic_target_morph(info, no_target_collision=False)
    assert (m0.contype, m0.conaffinity) == (0xFFFF, 0xFFFF)


def test_ntc_masks_filter_target_pairs_only():
    """The engine's pair filter is ``(ct_a & ca_b) or (ct_b & ca_a)``."""
    def collides(a, b):
        return bool((a[0] & b[1]) or (b[0] & a[1]))
    tgt = (ps.NTC_TARGET_CONTYPE, ps.NTC_TARGET_CONAFFINITY)
    assert not collides(tgt, tgt)                 # target vs target: filtered
    assert collides(tgt, (0xFFFF, 0xFFFF))        # vs ground / obstacles (default)
    assert collides(tgt, (1, 1))                  # vs terrain / URDF chassis (1/1)


def test_built_target_geoms_keep_the_masks(cpu_genesis_for_ntc):
    """``add_dynamic`` copies the morph for the main body (``_clone_morph``);
    the masks must survive into the BUILT geoms."""
    from genesis_vehicle import VehicleScene
    vs = VehicleScene(n_envs=1, raycast_mode="dual_scene", init_genesis=False)
    vs.add_ground_plane()
    info = dict(pos=[0, 0, 1], quat=[1, 0, 0, 0], type=1, scale=[0.5, 0.5, 0.5])
    on = ps.build_generic_target(vs, 0, info, ue_friction=1.0, ue_restitution=0.0,
                                 vis_mode=None, no_target_collision=True)
    off = ps.build_generic_target(vs, 1, dict(info, pos=[3, 0, 1]), ue_friction=1.0,
                                  ue_restitution=0.0, vis_mode=None, no_target_collision=False)
    vs.build()
    assert [(int(g.contype), int(g.conaffinity)) for g in on.entity_main.geoms] == [(2, 1)]
    assert [(int(g.contype), int(g.conaffinity)) for g in off.entity_main.geoms] == [(0xFFFF, 0xFFFF)]


@pytest.fixture(scope="module")
def cpu_genesis_for_ntc():
    from genesis_vehicle import VehicleScene
    if not getattr(gs, "_initialized", False):
        VehicleScene.init_backend("cpu")
    return gs


# --- the refusals (fake OSC, no UE) -------------------------------------------

class _Events:
    def __init__(self):
        self.log = []


class _FakeOSC:
    """Stands in for OSCManager. ``urdf_at`` = when the URDF init becomes visible:
    "init" (before /Init/Done, i.e. when wait_for_initialization returns) or
    "late" (after it: set when VehicleScene is constructed), or None."""
    urdf_at = None
    events = None

    def __init__(self, **kw):
        _FakeOSC.events.log.append("osc")
        self.urdf_init_request = None
        _FakeOSC.instance = self

    def wait_for_initialization(self, timeout=60.0):
        _FakeOSC.events.log.append("init")
        if _FakeOSC.urdf_at == "init":
            self.urdf_init_request = {"path": "/x/car.urdf", "mapping": {}}
        return {"obstacles": {}, "physics": {"dt": 0.025},
                "target": {0: {"pos": [0, 0, 1], "quat": [1, 0, 0, 0], "type": 1}}}

    def close(self):
        _FakeOSC.events.log.append("close")


class _FakeVehicleScene:
    def __init__(self, **kw):
        _FakeOSC.events.log.append("VehicleScene")
        if _FakeOSC.urdf_at == "late":
            _FakeOSC.instance.urdf_init_request = {"path": "/x/car.urdf", "mapping": {}}

    @staticmethod
    def init_backend(*a, **k):
        _FakeOSC.events.log.append("init_backend")

    def add_static(self, *a, **k):
        _FakeOSC.events.log.append("add_static")

    def add_dynamic(self, *a, **k):
        _FakeOSC.events.log.append("TARGET:add_dynamic")
        raise AssertionError("a target was built")


def _run_main(monkeypatch, argv, urdf_at, capsys):
    ev = _Events()
    _FakeOSC.events, _FakeOSC.urdf_at = ev, urdf_at
    monkeypatch.setattr(ps, "OSCManager", _FakeOSC)
    monkeypatch.setattr(ps, "VehicleScene", _FakeVehicleScene)

    def _no_vehicle(**kw):
        ev.log.append("TARGET:build_vehicle")
        raise AssertionError("a URDF target was built")
    monkeypatch.setattr(ps.genesis_vehicle_builder, "build_vehicle", _no_vehicle)
    monkeypatch.setattr(sys, "argv", ["physics_server"] + argv)
    with pytest.raises(SystemExit) as ei:
        ps.main()
    out = capsys.readouterr()
    return ei.value.code, ev.log, out.out + out.err


def test_urdf_target_with_ntc_is_refused_before_the_scene_exists(monkeypatch, capsys):
    rc, log, out = _run_main(monkeypatch, ["--headless", "--no-target-collision"], "init", capsys)
    assert rc == 2
    assert ps.NTC_URDF_ERROR in out and "ERROR" in out
    assert "VehicleScene" not in log                      # no scene constructed
    assert not [e for e in log if e.startswith("TARGET:")]
    assert log[-1] == "close"


def test_urdf_init_after_init_done_is_refused_before_any_target(monkeypatch, capsys):
    """api-review condition 2: /Genesis/Vehicle/Init can arrive after /Init/Done
    (the early check then sees no URDF); the target-build point refuses too,
    with the same message and rc, before any target is built."""
    rc, log, out = _run_main(monkeypatch, ["--headless", "--no-target-collision"], "late", capsys)
    assert rc == 2
    assert ps.NTC_URDF_ERROR in out
    assert "VehicleScene" in log                          # the late URDF arrived after it
    assert not [e for e in log if e.startswith("TARGET:")]
    assert log[-1] == "close"


def test_multi_env_with_ntc_fails_at_parsing(monkeypatch, capsys):
    rc, log, out = _run_main(monkeypatch, ["--multi-env", "--headless", "--no-target-collision"],
                             None, capsys)
    assert rc == 2
    assert ps.NTC_URDF_ERROR in out
    assert log == []                                      # before any OSC / engine work


# --- public API only: no mask keyword ever reaches gs.morphs.URDF -------------

def _urdf_calls_with_mask_kw(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name == "URDF" and any(k.arg in ("contype", "conaffinity") for k in n.keywords):
                bad.append(f"{path}:{n.lineno}")
    return bad


def test_no_contype_keyword_reaches_a_urdf_morph():
    bad = []
    for root, dirs, files in os.walk(SDK):
        dirs[:] = [d for d in dirs if d not in ("tools", "__pycache__")]
        for f in files:
            if f.endswith(".py"):
                bad += _urdf_calls_with_mask_kw(os.path.join(root, f))
    assert not bad, bad
