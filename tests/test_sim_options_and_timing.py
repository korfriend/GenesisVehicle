"""Simulation time is fixed at ``build()``, and the build log must SAY so (v1.6.5).

Two defects shipped together in v1.5.0 and survived seven releases
(v1.5.0 … v1.6.4), both from the same genesis 1.4.0 drift — ``Scene.sim_options``
moved to ``Scene.options.sim``:

* ``server/physics_server.py``'s ``vs.sim_options.dt = sim_dt`` sat on the L2
  startup path outside any ``try``, so the L2 OSC server died with
  ``AttributeError`` on the first client connection;
* ``VehicleScene.build()``'s ``[genesis_vehicle] timing:`` line sat INSIDE
  ``except Exception: pass``, so it printed zero times and hid the same drift.

The second fact is what makes this more than an accessor fix: post-build writes
to the sim options are INERT on both supported engine versions.
``Simulator.__init__`` snapshots ``_dt`` / ``_substep_dt`` / ``_substeps`` and
``Simulator.dt`` is read-only, so the server's
``[Determinism] … 설정되었습니다`` line was false on L2 even on genesis 1.3.3
where the write did not raise. ``sim.set_gravity`` IS live.

The tests below pin all three facts. In particular
``test_post_build_sim_options_writes_are_inert`` is a tripwire on the ENGINE: if
a future genesis makes those writes live, it fails and sends the reader back to
``docs/physics-contracts.md`` §7.14 rather than letting the SDK quietly
contradict its own docs.

Harness: real CPU ``VehicleScene``, genesis-world 1.4.0, WSL2. The gravity /
inertness tests build their OWN scene (they deliberately corrupt dt/gravity, so
they must not share one).

| abbr | meaning |
|---|---|
| dt | simulation STEP duration (s) |
| substeps | solver iterations per step; internal interval is ``dt / substeps`` |
| L2 | per-entity batching axis (K vehicles in one scene) |
"""

import re

import pytest

pytest.importorskip("genesis")

import genesis as gs

from genesis_vehicle import VehicleScene

TIMING_RE = re.compile(
    r"\[genesis_vehicle\] timing: dt=([\d.]+)ms \((\d+)Hz\) x substeps=(\d+)")


@pytest.fixture(scope="module")
def cpu_genesis():
    if not getattr(gs, "_initialized", False):
        try:
            VehicleScene.init_backend("cpu")
        except Exception as e:                     # pragma: no cover
            pytest.skip(f"genesis CPU backend unavailable: {e}")
    return gs


def _plain_scene(dt=0.02, substeps=2, n_envs=1, gravity=(0.0, 0.0, -9.81)):
    """A ground plane only — no vehicle. Enough for every timing assertion and
    fast enough to build once per test."""
    vs = VehicleScene(dt=dt, substeps=substeps, n_envs=n_envs, gravity=gravity,
                      raycast_mode="single_scene", show_viewer=False,
                      init_genesis=False)
    vs.add_ground_plane(friction=1.0)
    return vs


def _falling_box(vs, z=5.0):
    return vs.add_dynamic(
        gs.morphs.Box(size=(0.4, 0.4, 0.4), pos=(0.0, 0.0, z)), physics=True)


def _z(body):
    import numpy as np
    p = body.entity_main.get_pos()
    p = p.cpu().numpy() if hasattr(p, "cpu") else np.asarray(p)
    return float(np.asarray(p).reshape(-1)[2] if np.asarray(p).ndim == 1
                 else np.asarray(p)[0][2])


def test_timing_line_is_actually_emitted(cpu_genesis, capsys):
    """The regression test this release exists for.

    Through v1.6.4 the timing line was inside ``except Exception: pass`` and,
    on genesis 1.4.0, never printed. Asserting "the code path exists" is
    exactly what failed; this asserts the BYTES reach stdout, and that the
    numbers in them are the engine's, not the caller's authored arguments.
    """
    vs = _plain_scene(dt=0.02, substeps=2)
    capsys.readouterr()                             # drop construction noise
    vs.build()
    out = capsys.readouterr().out
    m = TIMING_RE.search(out)
    assert m is not None, (
        "[genesis_vehicle] timing: line was NOT printed by build(). Captured "
        f"stdout was:\n{out}")
    assert float(m.group(1)) == pytest.approx(vs.effective_dt * 1e3, abs=0.05)
    assert int(m.group(3)) == vs.substeps
    # ...and effective_dt / substeps are the ENGINE's, not our ctor args.
    assert vs.effective_dt == pytest.approx(float(vs._main_scene.sim.dt))
    assert vs.substeps == int(vs._main_scene.sim.substeps)


def test_sim_options_is_the_engines_authored_object(cpu_genesis):
    """``VehicleScene.sim_options`` must resolve on this engine at all (it
    raised ``AttributeError`` for seven releases) and hand back the very object
    the scene was constructed with."""
    vs = _plain_scene(dt=0.02, substeps=2)
    vs.build()
    so = vs.sim_options
    assert so is not None
    assert float(so.dt) == pytest.approx(0.02)
    # Same object the engine holds, by identity, through whichever spelling
    # this genesis uses.
    from genesis_vehicle._gs_compat import scene_sim_options
    assert so is scene_sim_options(vs._main_scene)


def test_substeps_and_effective_dt_come_from_the_engine(cpu_genesis):
    """``substeps`` must read ``sim.substeps``, NOT the authored argument:
    genesis 1.4.0 re-derives it (measured authored 1 → effective 4 on the
    reference scene), and only the engine's value describes the integration."""
    vs = _plain_scene(dt=0.02, substeps=1)
    vs.build()
    assert vs.substeps == int(vs._main_scene.sim.substeps)
    assert vs.effective_dt == pytest.approx(float(vs._main_scene.sim.dt))
    assert isinstance(vs.substeps, int)


def test_post_build_sim_options_writes_are_inert(cpu_genesis):
    """ENGINE tripwire. Writing ``dt`` / ``gravity`` on the authored options
    after ``build()`` changes nothing, on genesis 1.3.3 and 1.4.0 alike. If a
    future genesis makes them live, this test fails — go re-read
    ``docs/physics-contracts.md`` §7.14 and the server's ``[Determinism]``
    line before "fixing" it.
    """
    vs = _plain_scene(dt=0.02, substeps=2)
    box = _falling_box(vs, z=5.0)
    vs.build()
    dt_before, ss_before = vs.effective_dt, vs.substeps
    z0 = _z(box)

    vs.sim_options.dt = 0.2                         # 10x
    vs.sim_options.gravity = (0.0, 0.0, 0.0)        # and no gravity

    assert vs.effective_dt == pytest.approx(dt_before), (
        "post-build sim_options.dt became LIVE on this engine")
    assert vs.substeps == ss_before

    for _ in range(10):
        vs.step()
    z1 = _z(box)
    # gravity write ignored -> it still fell about 1/2 g t^2 over 10 * 0.02 s
    assert z0 - z1 > 0.15, (
        f"box did not fall (z {z0} -> {z1}): post-build sim_options.gravity "
        "became LIVE on this engine")


def test_set_gravity_is_the_live_knob(cpu_genesis):
    """``VehicleScene.set_gravity`` routes to ``sim.set_gravity``, which IS
    live — this is what the server's reset path must use to restore gravity."""
    vs = _plain_scene(dt=0.02, substeps=2)
    box = _falling_box(vs, z=5.0)
    vs.build()
    for _ in range(10):                             # build up some velocity
        vs.step()
    vs.set_gravity((0.0, 0.0, 0.0))
    a = _z(box)
    for _ in range(10):
        vs.step()
    b = _z(box)
    for _ in range(10):
        vs.step()
    c = _z(box)
    # With gravity off the motion is constant-velocity: equal drops per window.
    assert (a - b) == pytest.approx(b - c, rel=0.02, abs=1e-4), (
        f"drops {a - b} then {b - c}: gravity still accelerating after "
        "set_gravity((0,0,0))")


def test_set_gravity_requires_build(cpu_genesis):
    vs = _plain_scene()
    with pytest.raises(RuntimeError):
        vs.set_gravity((0.0, 0.0, 0.0))


def test_set_gravity_accepts_envs_idx_at_n_envs_gt_1(cpu_genesis):
    """``envs_idx`` is passed through for per-env gravity (L3)."""
    vs = _plain_scene(n_envs=2)
    vs.build()
    vs.set_gravity((0.0, 0.0, -1.0), envs_idx=[0])
    vs.step()
