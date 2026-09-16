"""genesis_vehicle.server 서브패키지 임포트 스모크 테스트.

서버는 SDK 본체와 달리 genesis/pythonosc/psutil 런타임이 필요하므로,
해당 의존성이 없는 순수-Python CI 에서는 자동 skip 된다.
"""
import pytest


def test_server_init_is_dependency_free():
    # __init__ 은 의도적으로 빈 모듈 — pythonosc 없는 환경에서도 임포트돼야 한다.
    import genesis_vehicle.server  # noqa: F401


def test_server_modules_import_and_expose_entry_points():
    pytest.importorskip("genesis")
    pytest.importorskip("pythonosc")
    from genesis_vehicle.server import physics_server, l3_runtime, vehicle_builder

    # CLI 진입점과 두 실행 모드
    assert callable(physics_server.main)
    assert callable(physics_server.cli)
    assert callable(l3_runtime.run_l3)

    # per-entity / L3 경로가 공유하는 cfg 빌더 표면
    assert callable(vehicle_builder.build_cfg)
    assert callable(vehicle_builder.build_vehicle)
    assert callable(vehicle_builder.strip_wheel_collisions)


def test_steer_mapping_recognizes_ue_field_names():
    pytest.importorskip("genesis")
    from genesis_vehicle.server.vehicle_builder import _mapping_steer_rad

    # UE FGenesisVehicleMapping.SteerScale 은 JSON 직렬화 시 'steerScale'
    assert _mapping_steer_rad({"steerScale": 0.6}) == 0.6
    assert _mapping_steer_rad({"SteerScale": 0.5}) == 0.5
    assert _mapping_steer_rad({"maxSteerRad": 0.7}) == 0.7
    assert _mapping_steer_rad({"MaxSteerRad": 0.65}) == 0.65
    # 우선순위: maxSteerRad > steerScale
    assert _mapping_steer_rad({"maxSteerRad": 0.7, "steerScale": 0.6}) == 0.7
    assert _mapping_steer_rad({}, default=0.7) == 0.7
    assert _mapping_steer_rad({}) is None


# --- source guard: no post-build writes to the sim options (v1.6.5) ----------

def _sim_options_writes_in(path):
    """Return `(lineno, source_line)` for every LITERAL assignment in `path`
    whose target attribute chain mentions ``sim_options``."""
    import ast

    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    lines = src.splitlines()
    hits = []

    def mentions_sim_options(node):
        """`sim_options` anywhere in this subtree — as an attribute access
        (``vs.sim_options``) or as a string handed to ``getattr``."""
        for n in ast.walk(node):
            if isinstance(n, ast.Attribute) and n.attr == "sim_options":
                return True
            if isinstance(n, ast.Constant) and n.value == "sim_options":
                return True
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "setattr" and node.args):
            # A literal setattr: its FIRST argument is the object being written.
            if mentions_sim_options(node.args[0]):
                hits.append((node.lineno, lines[node.lineno - 1].strip()))
            continue
        else:
            continue
        for t in targets:
            # Only a chain that ENDS in an attribute/subscript write can mutate
            # the options object; `so = vs.sim_options` (a plain Name target)
            # is a read and is not flagged.
            if isinstance(t, (ast.Attribute, ast.Subscript)) and mentions_sim_options(t):
                hits.append((node.lineno, lines[node.lineno - 1].strip()))
    return hits


def test_server_never_writes_to_sim_options():
    """Tripwire against the v1.5.0 regression: ``vs.sim_options.dt = sim_dt``
    on the L2 startup path raised ``AttributeError`` on genesis 1.4.0 and killed
    the server on the first client connection — and on genesis 1.3.3, where it
    did not raise, it was silently INERT (``Simulator`` snapshots dt/substeps at
    construction). Either way the server must not write there; gravity goes
    through ``VehicleScene.set_gravity``, dt is fixed at ``build()``.

    SCOPE, deliberately narrow: this is an AST tripwire for the LITERAL forms —
    ``x.sim_options.a = v``, ``+=``, ``x.sim_options.a[i] = v``, annotated
    assignment, and a literal ``setattr(<...sim_options...>, ...)``. It is NOT
    proof that nothing writes at runtime: binding the object to a name and
    passing it into a dict / an argument / a tuple, a dynamic ``setattr``, or
    ``exec`` all escape it. Three LITERAL forms escape it too, and are named
    here rather than implied away — a tuple-unpacking target
    (``a, vs.sim_options.dt = 1, 0.02``), a ``for vs.sim_options.dt in [...]``
    loop target, and ``with ... as vs.sim_options.dt`` — because the walk only
    inspects ``Assign`` / ``AugAssign`` / ``AnnAssign`` targets that are
    themselves an ``Attribute`` or ``Subscript`` (verified empty on all three,
    v1.6.5). None occurs in this tree; if one is ever wanted, widen the target
    check instead of trusting this docstring. The runtime facts are pinned by
    ``tests/test_sim_options_and_timing.py`` instead.
    """
    import pathlib

    server_dir = pathlib.Path(__file__).resolve().parent.parent / "server"
    offenders = []
    for py in sorted(server_dir.glob("*.py")):
        for lineno, text in _sim_options_writes_in(py):
            offenders.append(f"{py.name}:{lineno}: {text}")
    assert not offenders, (
        "server writes to sim_options (inert on genesis <= 1.3.3, "
        "AttributeError on >= 1.4.0):\n  " + "\n  ".join(offenders))


def test_sim_options_write_guard_catches_the_shapes_it_claims(tmp_path):
    """The guard's own coverage claim, exercised — including the three forms a
    regex misses (``+=``, subscripted target, ``setattr(getattr(...))``)."""
    bad = tmp_path / "bad.py"
    bad.write_text(
        "vs.sim_options.dt = 0.02\n"
        "vs.sim_options.dt += 0.001\n"
        "vs.sim_options.gravity[2] = -9.8\n"
        "setattr(getattr(vs, 'sim_options'), 'dt', 0.02)\n",
        encoding="utf-8")
    assert len(_sim_options_writes_in(bad)) == 4

    ok = tmp_path / "ok.py"
    ok.write_text(
        "so = vs.sim_options\n"                 # a read, and a binding
        "if vs.sim_options.dt == 0.02:\n"
        "    print(vs.sim_options.substeps)\n"
        "vs.set_gravity((0, 0, -9.81))\n",
        encoding="utf-8")
    assert _sim_options_writes_in(ok) == []
