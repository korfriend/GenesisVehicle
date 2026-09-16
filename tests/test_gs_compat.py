"""Unit tests for the genesis-version compatibility shims (`_gs_compat`).

Genesis 1.4.0 renamed/moved three APIs the SDK depends on. Each shim is
exercised here against BOTH engine shapes with stubs, so the compatibility
branch that this machine's genesis does not take is still covered.
"""

import numpy as np
import pytest
import torch

from genesis_vehicle._gs_compat import (
    apply_links_wrench, has_wrench_api, link_inertial, scene_sim_options,
    sensor_miss_value,
)


# --- apply_links_wrench: two-call (<= 1.3.3) vs one-call (>= 1.4.0) ----------

class _OldSolver:
    """genesis <= 1.3.3: separate force / torque entry points."""
    def __init__(self):
        self.calls = []

    def apply_links_external_force(self, force, links_idx=None, envs_idx=None):
        self.calls.append(("force", force, links_idx, envs_idx))

    def apply_links_external_torque(self, torque, links_idx=None, envs_idx=None):
        self.calls.append(("torque", torque, links_idx, envs_idx))


class _NewSolver:
    """genesis >= 1.4.0: one wrench entry point."""
    def __init__(self):
        self.calls = []

    def apply_links_external_wrench(self, force=None, torque=None,
                                    links_idx=None, envs_idx=None, **kw):
        self.calls.append(("wrench", force, torque, links_idx, envs_idx, kw))


def test_has_wrench_api_detects_both_shapes():
    assert has_wrench_api(_NewSolver()) is True
    assert has_wrench_api(_OldSolver()) is False


def test_apply_links_wrench_old_solver_splits_into_two_calls():
    s = _OldSolver()
    F = torch.ones(1, 2, 3)
    T = torch.full((1, 2, 3), 2.0)
    apply_links_wrench(s, F, T, [0, 1])
    assert [c[0] for c in s.calls] == ["force", "torque"]
    assert torch.equal(s.calls[0][1], F) and s.calls[0][2] == [0, 1]
    assert torch.equal(s.calls[1][1], T) and s.calls[1][2] == [0, 1]


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


# --- link_inertial: link.inertial_* (<= 1.3.3) vs link.desc.* (>= 1.4.0) -----

_I = np.diag([450.0, 2200.0, 2400.0])
_IDENT = np.array([1.0, 0.0, 0.0, 0.0])


class _OldLink:
    inertial_mass = 1200.0
    inertial_pos = np.array([0.0, 0.0, 0.5])
    inertial_quat = _IDENT
    inertial_i = _I


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


@pytest.mark.parametrize("link", [_OldLink(), _NewLink()])
def test_link_inertial_reads_both_engine_shapes(link):
    p = link_inertial(link)
    assert p is not None
    assert p.mass == pytest.approx(1200.0)
    assert np.allclose(p.pos, [0.0, 0.0, 0.5])
    assert np.allclose(p.quat, _IDENT)
    assert np.allclose(p.i, _I)


def test_link_inertial_returns_none_without_mass():
    assert link_inertial(_MasslessLink()) is None
    assert link_inertial(object()) is None


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


# --- server monkey patches over whichever force API the engine exposes -------

class _PatchTargetNew(_NewSolver):
    """Minimal solver surface `apply_monkey_patches` rebinds on genesis >= 1.4.0."""
    def set_dofs_position(self, position, dofs_idx=None, envs_idx=None, **kw): pass
    def control_dofs_position(self, position, dofs_idx=None, envs_idx=None, **kw): pass


class _PatchTargetOld(_OldSolver):
    def set_dofs_position(self, position, dofs_idx=None, envs_idx=None, **kw): pass
    def control_dofs_position(self, position, dofs_idx=None, envs_idx=None, **kw): pass


def test_server_patch_squeezes_3d_input_on_the_wrench_api():
    from genesis_vehicle.server.vehicle_builder import apply_monkey_patches
    s = _PatchTargetNew()
    apply_monkey_patches(s)
    apply_links_wrench(s, torch.ones(1, 1, 3), torch.zeros(1, 1, 3), [0])
    _, force, torque, _, _, _ = s.calls[0]
    assert force.shape == (1, 3) and torque.shape == (1, 3)


def test_server_patch_squeezes_3d_input_on_the_legacy_api():
    from genesis_vehicle.server.vehicle_builder import apply_monkey_patches
    s = _PatchTargetOld()
    apply_monkey_patches(s)
    apply_links_wrench(s, torch.ones(1, 1, 3), torch.zeros(1, 1, 3), [0])
    assert [c[0] for c in s.calls] == ["force", "torque"]
    assert s.calls[0][1].shape == (1, 3) and s.calls[1][1].shape == (1, 3)


# --- spin-MOI estimate must not read a permuted principal component ---------

def _fake_entity(inertia, quat, old_style=False):
    """An entity whose one link reports `inertia` in a frame oriented by `quat`,
    spelled the genesis <= 1.3.3 way (old_style) or the >= 1.4.0 way."""
    if old_style:
        link = type("L", (), dict(inertial_mass=20.0, inertial_pos=np.zeros(3),
                                  inertial_quat=quat, inertial_i=inertia))()
    else:
        desc = type("D", (), dict(mass=20.0, inertial_pos=np.zeros(3),
                                  inertial_quat=quat, inertia=inertia))()
        link = type("L", (), dict(desc=desc))()
    return type("E", (), dict(get_link=lambda self, name: link))()


@pytest.mark.parametrize("old_style", [True, False])
def test_spin_moi_is_read_in_the_link_frame(old_style):
    """The reference car's wheel is ixx=1, iyy=2, izz=1 and spins about +Y, so
    its spin MOI is 2.0. genesis >= 1.4.0 sorts that onto its principal axes as
    diag(2, 1, 1) plus a 90 deg rotation about +Z; reading the raw diagonal
    (what this did before v1.5.0) returns the transverse 1.0 instead."""
    from genesis_vehicle.urdf import estimate_spin_inertia_from_genesis
    s = np.sqrt(0.5)
    authored = _fake_entity(np.diag([1.0, 2.0, 1.0]), np.array([1.0, 0, 0, 0]),
                            old_style)
    principal = _fake_entity(np.diag([2.0, 1.0, 1.0]), np.array([s, 0.0, 0.0, s]),
                             old_style)
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


# --- deterministic-solver opt-in (genesis >= 1.4.0) -------------------------

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
    if tuple(int(p) for p in gs.__version__.split(".")[:2]) < (1, 4):
        pytest.skip(f"genesis-world {gs.__version__} predates both fixes")
    assert "use_deterministic_algorithms" in inspect.signature(gs.init).parameters
    from genesis.engine.solvers.rigid.constraint import solver as _solver
    src = inspect.getsource(_solver)
    assert "repeat_after_count=300" in src and "repeat_after_seconds=0" in src


# --- sensor_miss_value: both raycaster option shapes -------------------------
#
# Every genesis the SDK supports resolves `no_hit_value` (defaulting to
# `max_range`) onto the sensor's options at CONSTRUCTION. The shim also has to
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


# --- scene_sim_options: Scene.sim_options (<= 1.3.3) vs options.sim (>= 1.4.0) ---

class _SimOptionsStub:
    """Stands in for ``gs.options.SimOptions``; identity is what we assert."""
    def __init__(self, dt=0.02, substeps=2):
        self.dt, self.substeps = dt, substeps


class _NewSceneShape:
    """genesis >= 1.4.0: ``Scene.options.sim``, no ``Scene.sim_options``."""
    def __init__(self, sim_options):
        self.options = type("SceneOptions", (), {"sim": sim_options})()


class _OldSceneShape:
    """genesis <= 1.3.3: ``Scene.sim_options`` assigned in ``Scene.__init__``."""
    def __init__(self, sim_options):
        self.sim_options = sim_options


class _AlienSceneShape:
    """Neither spelling — a future engine that moved the options again."""


def test_scene_sim_options_reads_the_1_4_0_shape():
    so = _SimOptionsStub()
    assert scene_sim_options(_NewSceneShape(so)) is so


def test_scene_sim_options_reads_the_1_3_3_shape():
    so = _SimOptionsStub()
    assert scene_sim_options(_OldSceneShape(so)) is so


def test_scene_sim_options_raises_naming_both_spellings():
    """A silent ``None`` would resurface as ``'NoneType' has no attribute 'dt'``
    somewhere else entirely; the error must name both engine spellings so the
    reader knows which versions were probed."""
    with pytest.raises(AttributeError) as ei:
        scene_sim_options(_AlienSceneShape())
    msg = str(ei.value)
    assert "options.sim" in msg
    assert "sim_options" in msg
