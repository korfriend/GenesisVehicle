"""Unit tests for `_gs_compat`, the SDK's inventory of engine-private reads.

Since 2.0.5 the SDK requires genesis-world 1.5.0 and every shim has ONE code
path; the stub tests of the removed `<= 1.3.3` / 1.4.x shapes (two-call force
API, `link.inertial_*`, `Scene.sim_options`, warn-only floor) were deleted with
those branches (ledger: CHANGELOG 2.0.5). What stays is exercised here with
stubs, so the behaviour on an engine shape this machine does not have (a
renamed field, an old or unvalidated version) is still covered.
"""

import numpy as np
import pytest
import torch

from genesis_vehicle._gs_compat import (
    apply_links_wrench, link_inertial, sensor_miss_value,
)


# --- apply_links_wrench: one apply_links_external_wrench call ----------------

class _NewSolver:
    """genesis-world 1.5.0: one wrench entry point."""
    def __init__(self):
        self.calls = []

    def apply_links_external_wrench(self, force=None, torque=None,
                                    links_idx=None, envs_idx=None, **kw):
        self.calls.append(("wrench", force, torque, links_idx, envs_idx, kw))


def test_apply_links_wrench_new_solver_uses_one_call():
    s = _NewSolver()
    F = torch.ones(1, 2, 3)
    T = torch.full((1, 2, 3), 2.0)
    apply_links_wrench(s, F, T, [0, 1], envs_idx=[0])
    assert len(s.calls) == 1
    kind, force, torque, links_idx, envs_idx, _ = s.calls[0]
    assert kind == "wrench"
    assert torch.equal(force, F) and torch.equal(torque, T)
    assert links_idx == [0, 1] and envs_idx == [0]


def test_apply_links_wrench_has_no_two_call_fallback():
    """2.0.5 removed the <= 1.3.3 force/torque pair: a solver without the
    wrench API raises instead of being driven through two calls."""
    class _OldSolver:
        def apply_links_external_force(self, *a, **k): pass
        def apply_links_external_torque(self, *a, **k): pass
    with pytest.raises(AttributeError, match="apply_links_external_wrench"):
        apply_links_wrench(_OldSolver(), torch.zeros(1, 1, 3), torch.zeros(1, 1, 3), [0])


def test_apply_links_wrench_honours_a_rebound_attribute():
    """The server monkey-patches the solver instance; the shim must resolve the
    attribute per call rather than caching the bound method."""
    s = _NewSolver()
    seen = []
    orig = s.apply_links_external_wrench
    def patched(force=None, torque=None, links_idx=None, envs_idx=None, **kw):
        seen.append("patched")
        return orig(force, torque, links_idx, envs_idx, **kw)
    s.apply_links_external_wrench = patched
    apply_links_wrench(s, torch.zeros(1, 1, 3), torch.zeros(1, 1, 3), [0])
    assert seen == ["patched"] and len(s.calls) == 1


# --- link_inertial: link.desc.* ---------------------------------------------

_I = np.diag([450.0, 2200.0, 2400.0])
_IDENT = np.array([1.0, 0.0, 0.0, 0.0])


class _Desc:
    mass = 1200.0
    inertial_pos = np.array([0.0, 0.0, 0.5])
    inertial_quat = _IDENT
    inertia = _I


class _NewLink:
    desc = _Desc()


class _MasslessLink:
    desc = type("D", (), dict(mass=None, inertial_pos=None,
                              inertial_quat=None, inertia=None))()


def test_link_inertial_reads_link_desc():
    p = link_inertial(_NewLink())
    assert p is not None
    assert p.mass == pytest.approx(1200.0)
    assert np.allclose(p.pos, [0.0, 0.0, 0.5])
    assert np.allclose(p.quat, _IDENT)
    assert np.allclose(p.i, _I)


def test_link_inertial_no_longer_reads_the_1_3_3_spelling():
    """2.0.5: `link.inertial_*` (genesis <= 1.3.3) is not read any more; such a
    link has no `desc` and reads as massless (None), and the API check raises."""
    class _OldLink:
        inertial_mass = 1200.0
        inertial_pos = np.zeros(3)
        inertial_quat = _IDENT
        inertial_i = _I
    assert link_inertial(_OldLink()) is None
    from genesis_vehicle._gs_compat import check_link_inertial_api
    with pytest.raises(AttributeError, match=r"no 'desc'"):
        check_link_inertial_api(_OldLink())


def test_link_inertial_returns_none_without_mass():
    assert link_inertial(_MasslessLink()) is None
    assert link_inertial(object()) is None


# --- check_link_inertial_api (v1.8.0): link.desc fields, else AttributeError ---

@pytest.mark.parametrize("link", [_NewLink(), _MasslessLink()], ids=["desc", "massless"])
def test_check_link_inertial_api_accepts_desc_links(link):
    """A complete `desc` passes — including a massless link: the check is about
    the API, not the mass."""
    from genesis_vehicle._gs_compat import check_link_inertial_api
    check_link_inertial_api(link)


def test_check_link_inertial_api_raises_on_a_link_without_desc():
    """A link with no `desc` is an unknown engine. link_inertial alone would
    read it as massless (None), indistinguishable from a link that carries no
    mass; the check raises instead."""
    from genesis_vehicle._gs_compat import check_link_inertial_api
    with pytest.raises(AttributeError, match=r"no 'desc'"):
        check_link_inertial_api(object())


@pytest.mark.parametrize("drop", ["inertia", "inertial_quat", "mass", "inertial_pos"])
def test_check_link_inertial_api_names_a_missing_desc_field(drop):
    """C15 (v1.8.0): the container alone is not enough. A ``desc`` without
    ``inertia`` would make ``link_inertial`` read zero link inertia -- a
    silently wrong yaw inertia feeding the deadbeat scale. The check raises and
    names the missing field."""
    from genesis_vehicle._gs_compat import check_link_inertial_api
    fields = {k: getattr(_Desc, k) for k in
              ("mass", "inertial_pos", "inertial_quat", "inertia") if k != drop}
    link = type("L", (), dict(desc=type("D", (), fields)()))()
    with pytest.raises(AttributeError, match=rf"lacks \['{drop}'\]"):
        check_link_inertial_api(link)


def test_link_inertial_principal_axis_form_reconstructs_the_link_tensor():
    """genesis >= 1.4.0 diagonalizes an authored tensor onto its principal axes
    and carries the rotation in `inertial_quat`. R I R^T must give the authored
    link-frame tensor back (car_4w base_link: ixx=450, iyy=2200, izz=2400)."""
    class _D:
        mass = 1200.0
        inertial_pos = np.zeros(3)
        # 90 deg about +Y maps the principal diag back onto the authored one.
        inertial_quat = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])
        inertia = np.diag([2400.0, 2200.0, 450.0])

    p = link_inertial(type("L", (), dict(desc=_D()))())
    w, x, y, z = p.quat
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])
    assert np.allclose(R @ p.i @ R.T, _I, atol=1e-9)


# --- RigidOptions.dt normalization ------------------------------------------

class _Opts:
    """Stand-in for a pydantic options object (mutable, tracks explicit fields)."""
    def __init__(self, **kw):
        self.dt = kw.get("dt")
        self.model_fields_set = set(kw)


def test_normalize_rigid_dt_strips_a_dt_restating_the_step():
    from genesis_vehicle.vehicle_scene import _normalize_rigid_dt
    rigid = _Opts(dt=0.02, enable_collision=True)
    _normalize_rigid_dt(rigid, _Opts(dt=0.02, substeps=4))
    assert rigid.dt is None and "dt" not in rigid.model_fields_set


def test_normalize_rigid_dt_keeps_a_deliberate_substep_interval():
    from genesis_vehicle.vehicle_scene import _normalize_rigid_dt
    rigid = _Opts(dt=0.005)                      # = 0.02 / 4, meant for 1.4.0
    _normalize_rigid_dt(rigid, _Opts(dt=0.02, substeps=4))
    assert rigid.dt == 0.005 and "dt" in rigid.model_fields_set


def test_normalize_rigid_dt_ignores_options_without_dt():
    from genesis_vehicle.vehicle_scene import _normalize_rigid_dt
    rigid = _Opts(enable_collision=True)
    _normalize_rigid_dt(rigid, _Opts(dt=0.02, substeps=4))
    assert rigid.dt is None


# --- server monkey patch over the wrench API ---------------------------------

class _PatchTargetNew(_NewSolver):
    """Minimal solver surface `apply_monkey_patches` rebinds."""
    def set_dofs_position(self, position, dofs_idx=None, envs_idx=None, **kw): pass
    def control_dofs_position(self, position, dofs_idx=None, envs_idx=None, **kw): pass


def test_server_patch_squeezes_3d_input_on_the_wrench_api():
    from genesis_vehicle.server.vehicle_builder import apply_monkey_patches
    s = _PatchTargetNew()
    apply_monkey_patches(s)
    apply_links_wrench(s, torch.ones(1, 1, 3), torch.zeros(1, 1, 3), [0])
    _, force, torque, _, _, _ = s.calls[0]
    assert force.shape == (1, 3) and torque.shape == (1, 3)


# --- spin-MOI estimate must not read a permuted principal component ---------

def _fake_entity(inertia, quat):
    """An entity whose one link reports `inertia` in a frame oriented by `quat`
    (`link.desc`)."""
    desc = type("D", (), dict(mass=20.0, inertial_pos=np.zeros(3),
                              inertial_quat=quat, inertia=inertia))()
    link = type("L", (), dict(desc=desc))()
    return type("E", (), dict(get_link=lambda self, name: link))()


def test_spin_moi_is_read_in_the_link_frame():
    """The reference car's wheel is ixx=1, iyy=2, izz=1 and spins about +Y, so
    its spin MOI is 2.0. genesis >= 1.4.0 sorts that onto its principal axes as
    diag(2, 1, 1) plus a 90 deg rotation about +Z; reading the raw diagonal
    (what this did before v1.5.0) returns the transverse 1.0 instead."""
    from genesis_vehicle.urdf import estimate_spin_inertia_from_genesis
    s = np.sqrt(0.5)
    authored = _fake_entity(np.diag([1.0, 2.0, 1.0]), np.array([1.0, 0, 0, 0]))
    principal = _fake_entity(np.diag([2.0, 1.0, 1.0]), np.array([s, 0.0, 0.0, s]))
    # The two describe the same physical wheel; the raw diagonal does not.
    assert principal.get_link("w") is not authored.get_link("w")
    for e in (authored, principal):
        assert estimate_spin_inertia_from_genesis(
            e, "w", spin_axis_local=(0.0, 1.0, 0.0)) == pytest.approx(2.0)
        assert estimate_spin_inertia_from_genesis(e, "w") == pytest.approx(2.0)


def test_spin_moi_raises_on_a_link_without_inertia():
    from genesis_vehicle.urdf import estimate_spin_inertia_from_genesis
    e = type("E", (), dict(get_link=lambda self, n: _MasslessLink()))()
    with pytest.raises(ValueError):
        estimate_spin_inertia_from_genesis(e, "w", spin_axis_local=(0.0, 1.0, 0.0))


# --- deterministic-solver opt-in ---------------------------------------------

def test_init_backend_accepts_deterministic():
    """gs.init(use_deterministic_algorithms=True) pins the GPU constraint
    solver's arm instead of letting the runtime autotuner pick it — the fix for
    the run-to-run divergence reported against genesis-world v1.1.1."""
    import inspect

    from genesis_vehicle import VehicleScene
    assert "deterministic" in inspect.signature(VehicleScene.init_backend).parameters


def test_this_genesis_carries_both_upstream_determinism_fixes():
    """PR #3187 (call-count re-benchmark) and #3222 (use_deterministic_algorithms)
    both landed in genesis-world 1.4.0. Pinned so a backend downgrade is loud."""
    import inspect

    import genesis as gs
    assert "use_deterministic_algorithms" in inspect.signature(gs.init).parameters
    from genesis.engine.solvers.rigid.constraint import solver as _solver
    src = inspect.getsource(_solver)
    assert "repeat_after_count=300" in src and "repeat_after_seconds=0" in src


# --- sensor_miss_value: both raycaster option shapes -------------------------
#
# genesis-world 1.5.0 resolves `no_hit_value` (defaulting to `max_range`) onto
# the sensor's options at CONSTRUCTION. The shim also has to
# survive an options object that carries only `max_range`, and a sensor that
# exposes no options at all — branches this machine's engine never takes.

class _Options:
    """A raycaster options object; omit a field to model the other shape."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Sensor:
    def __init__(self, options=None):
        if options is not None:
            self._options = options


def test_sensor_miss_value_reads_no_hit_value():
    s = _Sensor(_Options(no_hit_value=20.0, max_range=20.0))
    assert sensor_miss_value(s) == 20.0


def test_sensor_miss_value_prefers_no_hit_value_over_max_range():
    s = _Sensor(_Options(no_hit_value=50.0, max_range=10.0))
    assert sensor_miss_value(s) == 50.0


def test_sensor_miss_value_falls_back_to_max_range():
    """An options shape with no `no_hit_value` field at all, and one where it is
    still None (the pre-`model_post_init` state)."""
    assert sensor_miss_value(_Sensor(_Options(max_range=7.5))) == 7.5
    assert sensor_miss_value(
        _Sensor(_Options(no_hit_value=None, max_range=7.5))) == 7.5


def test_sensor_miss_value_keeps_a_zero_sentinel():
    """`is None`, not `or` — 0.0 is falsy but legitimate, and must NOT fall
    through to max_range (that would silently mark every miss a hit)."""
    assert sensor_miss_value(_Sensor(_Options(no_hit_value=0.0,
                                              max_range=10.0))) == 0.0


def test_sensor_miss_value_is_none_when_the_engine_exposes_nothing():
    assert sensor_miss_value(_Sensor()) is None
    assert sensor_miss_value(_Sensor(_Options())) is None


# --- hide_vgeoms: an ENGINE-PRIVATE write, guarded ---------------------------

class _VGeom150:
    """genesis-world 1.5.0 RigidVisGeom shape: `active_envs_mask` exists (None
    until the engine or the SDK sets it)."""
    def __init__(self):
        self.active_envs_mask = None
        self.active_envs_idx = None


class _VGeomNoMask:
    """An engine that renamed / dropped the private mask."""
    def __init__(self):
        self.active_envs_idx = None


def test_hide_vgeoms_writes_an_all_false_mask_per_env():
    from genesis_vehicle._gs_compat import hide_vgeoms
    vgs = [_VGeom150(), _VGeom150()]
    assert hide_vgeoms(vgs, 3) == 2
    for vg in vgs:
        assert vg.active_envs_mask.dtype == torch.bool
        assert vg.active_envs_mask.shape == (3,) and not bool(vg.active_envs_mask.any())
        assert len(vg.active_envs_idx) == 0
    one = _VGeom150()
    hide_vgeoms([one], 0)                       # n_envs 0 (unbatched) -> length 1
    assert one.active_envs_mask.shape == (1,)


def test_hide_vgeoms_raises_when_the_private_mask_is_missing():
    """review_gs150_5.md 3(i): without `active_envs_mask` the write would add a
    new attribute the engine never reads (wheel drawn twice, silently). It must
    raise, naming the attribute, BEFORE writing anything to that geom."""
    from genesis_vehicle._gs_compat import hide_vgeoms
    ok, bad = _VGeom150(), _VGeomNoMask()
    with pytest.raises(AttributeError, match="active_envs_mask"):
        hide_vgeoms([ok, bad], 1)
    assert not hasattr(bad, "active_envs_mask")
    assert bad.active_envs_idx is None          # nothing written to the bad geom


def test_hide_vgeoms_targets_an_attribute_this_engine_has():
    """The guard is only meaningful if the engine's visual geom really carries
    the attribute: read it off the installed genesis class."""
    pytest.importorskip("genesis")
    import inspect
    from genesis.engine.entities.rigid_entity import rigid_geom
    src = inspect.getsource(rigid_geom.RigidVisGeom.__init__)
    assert "self.active_envs_mask" in src


# --- require_genesis: the REQUIRED engine floor (2.0.5) ----------------------
#
# Captured with a handler on the `genesis_vehicle` logger, not caplog: gs.init()
# replaces the logging configuration and pytest's capture handler does not
# survive it (see tests/test_bump_stop_dt_warning.py::_Capture).

import logging as _logging

from genesis_vehicle import _gs_compat


class _VersionCapture(_logging.Handler):
    def __init__(self):
        super().__init__(level=_logging.WARNING)
        self.recs = []

    def emit(self, record):
        if "genesis-version" in record.getMessage():
            self.recs.append((record.levelname, record.getMessage()))

    @property
    def msgs(self):
        return [m for _, m in self.recs]

    def __enter__(self):
        _logging.getLogger("genesis_vehicle").addHandler(self)
        return self

    def __exit__(self, *exc):
        _logging.getLogger("genesis_vehicle").removeHandler(self)
        return False


class _GsStub:
    def __init__(self, version):
        if version is not None:
            self.__version__ = version


@pytest.fixture
def fresh_floor(monkeypatch):
    """Reset the once-per-process flags for the duration of one test."""
    for flag in ("_floor_error_logged", "_unvalidated_warned", "_unparsed_warned"):
        monkeypatch.setattr(_gs_compat, flag, False)
    return _gs_compat


@pytest.mark.parametrize("version", ["1.4.5", "1.4.0", "1.3.3", "1.4", "1.4.7",
                                     "1.5.0rc1", "1.5.0.dev3", "0.4.6"])
def test_floor_raises_below_1_5_0(fresh_floor, version):
    with _VersionCapture() as cap:
        with pytest.raises(fresh_floor.GenesisVersionError) as ei:
            fresh_floor.require_genesis(_GsStub(version))
    msg = str(ei.value)
    assert f"(found {version})" in msg
    assert "genesis-world>=1.5.0,<1.6" in msg and "2.0.4" in msg
    assert isinstance(ei.value, ImportError)
    assert [lvl for lvl, _ in cap.recs] == ["ERROR"] and msg in cap.recs[0][1]


@pytest.mark.parametrize("version", ["1.5.0", "1.5.0.post1", "1.5.0+local",
                                     "1.5.0-cuda12", "1.5.0_cpu", "1.5.3", "1.5"])
def test_floor_is_silent_in_the_validated_range(fresh_floor, version):
    with _VersionCapture() as cap:
        assert fresh_floor.require_genesis(_GsStub(version)) is None
    assert cap.recs == []


@pytest.mark.parametrize("version", ["1.6.0", "1.6.0rc1", "2.0.0"])
def test_floor_warns_once_from_1_6_and_continues(fresh_floor, version):
    with _VersionCapture() as cap:
        fresh_floor.require_genesis(_GsStub(version))
        fresh_floor.require_genesis(_GsStub(version))
    assert [lvl for lvl, _ in cap.recs] == ["WARNING"]
    assert "not validated" in cap.msgs[0] and version in cap.msgs[0]


@pytest.mark.parametrize("version", ["garbage", "", None])
def test_floor_unparseable_version_warns_once_and_continues(fresh_floor, version):
    with _VersionCapture() as cap:
        assert fresh_floor.require_genesis(_GsStub(version)) is None   # no raise
        assert fresh_floor.require_genesis(_GsStub(version)) is None
    assert len(cap.recs) == 1 and cap.recs[0][0] == "WARNING"
    assert "could not parse" in cap.msgs[0]


def test_floor_error_is_logged_once_per_process_but_raised_every_time(fresh_floor):
    """A caller's `except ImportError` would swallow the error, so it is logged
    at ERROR -- once per process; every import attempt still raises."""
    with _VersionCapture() as cap:
        for _ in range(3):
            with pytest.raises(fresh_floor.GenesisVersionError):
                fresh_floor.require_genesis(_GsStub("1.4.5"))
    assert [lvl for lvl, _ in cap.recs] == ["ERROR"]


def test_floor_names_are_exported_from_the_package():
    import genesis_vehicle
    assert genesis_vehicle.GenesisVersionError is _gs_compat.GenesisVersionError
    assert genesis_vehicle.require_genesis is _gs_compat.require_genesis
    assert {"GenesisVersionError", "require_genesis"} <= set(genesis_vehicle.__all__)
    assert _gs_compat.GENESIS_FLOOR == (1, 5, 0)
    assert _gs_compat.GENESIS_UNVALIDATED_FROM == (1, 6, 0)


def test_this_engine_passes_the_floor(fresh_floor):
    """The installed engine (the test run's own) is in the validated range."""
    import genesis
    with _VersionCapture() as cap:
        fresh_floor.require_genesis(genesis)
    assert cap.recs == [], cap.recs


_FLOOR_CALLERS = ["core.py", "vehicle_scene.py", "multi_vehicle.py", "raycast.py",
                  "scene_helpers.py", "visual.py", "server/__init__.py"]
_FLOOR_FREE = ["_pipeline.py", "_gs_compat.py", "dynamics.py"]


def _module_level_calls(path, name):
    import ast
    tree = ast.parse(open(path).read())
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")).endswith(name):
                out.append(n.lineno)
    return out


def test_floor_is_called_at_module_level_in_every_genesis_module():
    import os
    import genesis_vehicle
    root = os.path.dirname(genesis_vehicle.__file__)
    for rel in _FLOOR_CALLERS:
        assert _module_level_calls(os.path.join(root, rel), "require_genesis"), rel


def test_floor_is_never_called_at_module_level_on_the_genesis_free_path():
    """AC-FL (e): `import genesis_vehicle` reaches _pipeline / dynamics /
    control/* (control/__init__ -> plant -> _pipeline), so a check there would
    block Genesis-free use of PathFollower / SweepTable."""
    import os
    import genesis_vehicle
    root = os.path.dirname(genesis_vehicle.__file__)
    rels = list(_FLOOR_FREE) + [os.path.join("control", f) for f in
                                sorted(os.listdir(os.path.join(root, "control")))
                                if f.endswith(".py")]
    rels += [os.path.join("control", "mpc", f) for f in
             sorted(os.listdir(os.path.join(root, "control", "mpc"))) if f.endswith(".py")] \
        if os.path.isdir(os.path.join(root, "control", "mpc")) else []
    bad = {rel: _module_level_calls(os.path.join(root, rel), "require_genesis") for rel in rels}
    assert not {k: v for k, v in bad.items() if v}, bad


# --- coarse-SDF post-build check: stub solvers (both collider shapes) --------

class _Mat:
    def __init__(self, **kw):
        self.sdf_cell_size = kw.get("sdf_cell_size", 0.005)
        self.sdf_min_res = kw.get("sdf_min_res", 32)
        self.sdf_max_res = kw.get("sdf_max_res", 128)


class _Ent:
    def __init__(self, idx, mat):
        self.idx, self.material = idx, mat


class _Link:
    def __init__(self, name):
        self.name = name


class _Geom:
    def __init__(self, idx, convex, mat, gtype=None, link="body"):
        self.idx, self.is_convex, self.type = idx, convex, gtype
        self.entity, self.link = _Ent(idx, mat), _Link(link)


class _Cfg:
    def __init__(self, flag):
        self.has_nonconvex_nonterrain = flag


class _Collider:
    def __init__(self, flag):
        self.collider_config = _Cfg(flag)


class _StubRigidSolver:
    def __init__(self, geoms, flag="absent"):
        self.geoms = geoms
        if flag != "absent":
            self.collider = _Collider(flag)


def _coarse():
    return _Mat(**_gs_compat.COARSE_SDF_MATERIAL_KW)


def test_coarse_sdf_kwargs_are_the_validator_minimum():
    """16 is the `ge=16` floor of `sdf_min_res` AND `sdf_max_res` on
    genesis-world 1.5.0; with both at 16 the two clip bounds coincide, so the
    grid is 16 nodes per axis whatever the geom's width (2.0.5 added
    `sdf_max_res`: since 1.4.7 the convex-width refinement pulled a thin convex
    geom past 16 without it)."""
    kw = _gs_compat.COARSE_SDF_MATERIAL_KW
    assert kw == {"sdf_cell_size": 1.0e4, "sdf_min_res": 16, "sdf_max_res": 16}


def test_coarse_sdf_lists_only_nonconvex_nonterrain_coarse_geoms():
    import genesis as gs
    geoms = [_Geom(0, True, _coarse()),                                  # convex: fine
             _Geom(1, False, _coarse(), link="bad"),                     # the one
             _Geom(2, False, _Mat()),                                    # default material
             _Geom(3, False, _coarse(), gtype=gs.GEOM_TYPE.TERRAIN),     # heightfield
             _Geom(4, False, _Mat(sdf_cell_size=1.0e4)),                 # old kwargs, not ours
             _Geom(5, False, _Mat(sdf_cell_size=1.0e4, sdf_min_res=16))]  # 2.0.4 two-key copy: not ours
    got = _gs_compat.coarse_sdf_nonconvex_geoms(_StubRigidSolver(geoms))
    assert len(got) == 1 and "geom 1" in got[0] and "bad" in got[0]


def test_coarse_sdf_warning_is_gated_on_the_nonconvex_narrowphase():
    geoms = [_Geom(1, False, _coarse())]
    assert _gs_compat.warn_coarse_sdf_on_nonconvex(_StubRigidSolver(geoms, False), "s") == []
    assert len(_gs_compat.warn_coarse_sdf_on_nonconvex(_StubRigidSolver(geoms, True), "s")) == 1


def test_coarse_sdf_warning_fails_open_without_the_collider_flag():
    """An engine without `collider.collider_config.has_nonconvex_nonterrain`
    (the path 1.4.0 and 1.4.5 share) must still report, not go quiet."""
    geoms = [_Geom(1, False, _coarse())]
    assert len(_gs_compat.warn_coarse_sdf_on_nonconvex(_StubRigidSolver(geoms), "s")) == 1
    assert _gs_compat.warn_coarse_sdf_on_nonconvex(None, "s") == []


# --- coarse-SDF kwargs on a REAL engine: node counts + the post-build check ---

def _sdf_nodes(geom):
    """Per-axis SDF node count of a built geom (``desc.sdf.n_nodes`` on
    genesis-world 1.5.0; the 1.4.0 ``RigidGeom.sdf_res`` branch was removed in
    2.0.5)."""
    return tuple(int(n) for n in geom.desc.sdf.n_nodes)


def test_coarse_sdf_kwargs_give_a_16_node_grid_on_this_engine(tmp_path, monkeypatch):
    """`COARSE_SDF_MATERIAL_KW` must land on 16 nodes per axis for every kind of
    convex geom the SDK gives it to — a primitive, a THIN primitive (a 0.1 m
    wall, which the 1.4.7+ convex-width refinement refines past 16 without
    `sdf_max_res`), a convexified mesh, and the two SDK-owned dual_scene raycast
    mirrors (rigid Terrain mirror, add_dynamic wheel_raycast mirror) — while
    the DEFAULT material stays finer (so this cannot pass vacuously). Also checks the post-build check runs on BOTH scenes
    and lists a real non-convex coarse geom, but stays quiet while no collision
    pair can read its SDF (every body here is fixed)."""
    trimesh = pytest.importorskip("trimesh")
    gs = pytest.importorskip("genesis")
    from genesis_vehicle import VehicleScene, vehicle_scene
    from genesis_vehicle._gs_compat import COARSE_SDF_MATERIAL_KW as C

    box = str(tmp_path / "box.obj")
    trimesh.creation.box(extents=(1.0, 2.0, 3.0)).export(box)
    ring = str(tmp_path / "ring.obj")
    trimesh.creation.annulus(r_min=0.3, r_max=0.6, height=0.4).export(ring)

    calls = []
    real_check = vehicle_scene.warn_coarse_sdf_on_nonconvex

    def _recording_check(solver, where):
        out = real_check(solver, where)
        calls.append((where, out))
        return out

    monkeypatch.setattr(vehicle_scene, "warn_coarse_sdf_on_nonconvex", _recording_check)

    VehicleScene.init_backend("cpu")
    vs = VehicleScene(dt=0.02, substeps=2, n_envs=1, show_viewer=False)
    prim = vs.add_static(morph=gs.morphs.Box(size=(1, 1, 1), pos=(5, 0, 0.5), fixed=True),
                         material=gs.materials.Rigid(**C), name="prim")
    thin = vs.add_static(morph=gs.morphs.Box(size=(0.1, 10.0, 3.0), pos=(12, 0, 1.5), fixed=True),
                         material=gs.materials.Rigid(**C), name="thin")
    thin_d = vs.add_static(morph=gs.morphs.Box(size=(0.1, 10.0, 3.0), pos=(14, 0, 1.5), fixed=True),
                           material=gs.materials.Rigid(sdf_cell_size=1.0e4, sdf_min_res=16),
                           name="thin_two_key")
    cmesh = vs.add_static(morph=gs.morphs.Mesh(file=box, pos=(-5, 0, 1.5), fixed=True,
                                               convexify=True),
                          material=gs.materials.Rigid(**C), name="cmesh")
    dmesh = vs.add_static(morph=gs.morphs.Mesh(file=box, pos=(-9, 0, 1.5), fixed=True,
                                               convexify=True),
                          material=gs.materials.Rigid(), name="dmesh")
    ring_h = vs.add_static(morph=gs.morphs.Mesh(file=ring, pos=(0, 6, 0.2), fixed=True,
                                                convexify=False),
                           material=gs.materials.Rigid(**C), name="ring")
    terr = vs.add_static(
        morph=gs.morphs.Terrain(n_subterrains=(1, 1), subterrain_size=(8.0, 8.0),
                                horizontal_scale=0.5, vertical_scale=0.005,
                                subterrain_types=[["flat_terrain"]], pos=(20.0, 20.0, 0.0)),
        material=gs.materials.Rigid(**C), name="terrain")
    dyn = vs.add_dynamic(gs.morphs.Box(size=(0.5, 0.5, 0.5), pos=(0, -6, 1.0)),
                         physics=False, wheel_raycast=True, name="dyn")
    vs.build()

    def nodes(entity):
        return [_sdf_nodes(g) for g in entity.geoms]

    assert nodes(prim.entity_main) == [(16, 16, 16)]
    assert nodes(thin.entity_main) == [(16, 16, 16)]
    # Without sdf_max_res (the 2.0.4 two-key dict) the thin axis refines past 16
    # on this engine -- the reason the key was added.
    assert max(nodes(thin_d.entity_main)[0]) > 16, nodes(thin_d.entity_main)
    assert nodes(cmesh.entity_main) == [(16, 16, 16)]
    assert all(min(n) > 16 for n in nodes(dmesh.entity_main)), nodes(dmesh.entity_main)
    # SDK-owned raycast-scene mirrors (vehicle_scene add_static Terrain branch,
    # add_dynamic wheel_raycast mirror): material chosen by the SDK.
    assert type(terr.entity_raycast).__name__ != "KinematicEntity"  # rigid Terrain mirror
    assert nodes(terr.entity_raycast) == [(16, 16, 16)]
    assert nodes(dyn.entity_raycast) == [(16, 16, 16)]
    assert all(min(n) > 16 for n in nodes(dyn.entity_main))         # user default kept

    # The post-build check ran for both scenes ...
    assert [w for w, _ in calls] == ["main scene", "raycast scene"]
    # ... finds the real non-convex coarse geom on this engine's geom objects ...
    listed = _gs_compat.coarse_sdf_nonconvex_geoms(vs.rigid_solver)
    ring_idx = ring_h.entity_main.geoms[0].idx
    assert len(listed) == 1 and f"geom {ring_idx} " in listed[0]
    # ... and stays quiet: every body is fixed, so no collision pair reads an SDF.
    assert not bool(vs.rigid_solver.collider.collider_config.has_nonconvex_nonterrain)
    assert calls[0][1] == [] and calls[1][1] == []
