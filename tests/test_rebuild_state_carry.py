"""A config rebuild must not restart the vehicle (v1.6.2).

``VehicleScene.mark_config_dirty()`` + ``step()`` is the documented way to make
a post-``build()`` cfg change take effect. Through 1.6.1 it also, silently:

* zeroed the integrator state — wheel ``omega`` fell 51-60 % in one step
  (measured ``[-17.4207, -16.7303, -17.3659, -14.2479]`` →
  ``[-8.5504, -8.1484, -6.2335, -5.8315]``) and the spin accumulator jumped up
  to 2.08 rad,
* re-captured the wheel REST pose off a SAGGING vehicle (z 0.3000 → 0.1499 —
  a permanent 0.150 m visual reference error),
* left the instanced wheel renderer bound to the discarded kind, which is
  never stepped again (the rendered wheels stop turning permanently),
* left ``DifferentiablePlant`` reading the discarded proto,
* and consumed the dirty flag even when the rebuild failed.

Separately, ``VehicleScene.reset()`` reset nothing at all on the default
``solver="batched"`` scene (``veh.physics`` is None there), and
``MultiVehicleKindPhysics.reset``'s ``vehicle_ids`` parameter has always taken
FLAT ``env * K + slot`` ROWS, so a "per-vehicle" reset at ``n_envs > 1`` only
reset env 0's copy.

Harness: real CPU ``VehicleScene``, ``samples/urdf/car_4w.urdf``,
``dt = cfg.recommended_dt``, ``substeps=10``, friction 1.0 ground, genesis-world
1.4.0, WSL2. Comparisons are ``torch.equal``, not a tolerance: re-resolving the
same config was measured bit-identical, and two scenes in one process match
bit-for-bit over 40 steps.

| abbr | meaning |
|---|---|
| MVP | MultiVehiclePhysics (the batched multi-vehicle driver) |
| K | vehicles of one kind in a scene (L2 batch axis) |
| NK | flat batch rows, ``n_envs * K`` (env-major, vehicle-minor) |
| n_envs | parallel sim instances (L3 batch axis) |
| MOI | moment of inertia |
"""

import os

import numpy as np
import pytest

pytest.importorskip("genesis")

import torch
import genesis as gs

from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann
from genesis_vehicle.core import (
    REST_POSE_ATTRS, RUNTIME_STATE_ATTRS, RUNTIME_STATE_ROW_ATTRS,
)

URDF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "samples", "urdf", "car_4w.urdf"))

#: Everything a rebuild must carry, in the order the assertions report it.
CARRIED = RUNTIME_STATE_ATTRS + RUNTIME_STATE_ROW_ATTRS


@pytest.fixture(scope="module")
def cpu_genesis():
    if not getattr(gs, "_initialized", False):
        try:
            VehicleScene.init_backend("cpu")
        except Exception as e:                     # pragma: no cover
            pytest.skip(f"genesis CPU backend unavailable: {e}")
    return gs


def _scene(*, n_envs=1, raycast_mode="dual_scene", n_vehicles=1,
           camera=False, own_cfgs=False):
    """A ground plane + ``n_vehicles`` reference cars, built and ready to step.

    ``own_cfgs`` gives every vehicle its OWN cfg object, which is what makes it
    a separate kind (``_kind_key`` is the cfg identity)."""
    cfg0 = car_4w_rwd_ackermann(URDF, stability="control")
    vs = VehicleScene(dt=cfg0.recommended_dt, substeps=10, n_envs=n_envs,
                      raycast_mode=raycast_mode, show_viewer=False,
                      init_genesis=False)
    vs.add_ground_plane(friction=1.0)
    if camera:
        vs.add_camera(res=(160, 120), GUI=False)
    cfgs, vehs = [], []
    for i in range(n_vehicles):
        cfg = cfg0 if (i == 0 or not own_cfgs) else car_4w_rwd_ackermann(
            URDF, stability="control")
        cfgs.append(cfg)
        vehs.append(vs.add_vehicle(
            URDF, car_4w_rwd_ackermann, cfg=cfg, pos=(0.0, 4.0 * i, 0.6),
            material=gs.materials.Rigid(friction=1.0)))
    vs.build()
    return vs, vehs, cfgs


def _drive(vs, vehs, steps, throttle=0.5, steer=0.1):
    for _ in range(steps):
        for i, veh in enumerate(vehs):
            veh.set_inputs(throttle=throttle + 0.1 * i, brake=0.0, steer=steer)
        vs.step()


def _snapshot(proto):
    snap = {a: getattr(proto, a).detach().clone() for a in CARRIED}
    snap["_prev_init"] = bool(proto._prev_init)
    for a in REST_POSE_ATTRS:
        v = getattr(proto, a)
        snap[a] = None if v is None else v.detach().clone()
    return snap


# --- T1 / T3: what the rebuild carries ---------------------------------------

def test_t1_rebuild_carries_every_runtime_tensor(cpu_genesis):
    """White box: snapshot the proto, force the rebuild, compare every attr the
    carry claims to cover with ``torch.equal``."""
    vs, vehs, _ = _scene()
    _drive(vs, vehs, 60)
    old = vs.physics.kinds[0]._proto
    before = _snapshot(old)
    assert bool(old.omega.abs().max() > 1.0), "premise: the car is rolling"

    vs.mark_config_dirty()
    assert vs._ensure_grouped() is True
    vs._build_mvp()
    new = vs.physics.kinds[0]._proto
    assert new is not old, "premise: the rebuild really replaced the driver"

    for a in CARRIED:
        assert torch.equal(getattr(new, a), before[a]), (
            f"{a} was not carried across the rebuild:\n"
            f"  before: {before[a].flatten()[:4].tolist()}\n"
            f"  after:  {getattr(new, a).flatten()[:4].tolist()}")
    assert new._prev_init is before["_prev_init"]


def test_t3_rest_pose_is_not_recaptured_from_a_sagging_vehicle(cpu_genesis):
    """The rest pose is a construction-time constant (joints at 0). A rebuild
    re-capturing it while the suspension carries the car's weight moved wheel
    z 0.3000 → 0.1499 and tilted the rest quat off identity."""
    vs, vehs, _ = _scene()
    _drive(vs, vehs, 60)
    old = vs.physics.kinds[0]._proto
    rest_pos = old._rest_wheel_pos_local.clone()
    rest_quat = old._rest_wheel_quat_local.clone()
    assert float(rest_pos[0, 2]) == pytest.approx(0.30, abs=1e-3)

    vs.mark_config_dirty()
    vs.step()
    new = vs.physics.kinds[0]._proto
    assert new is not old
    assert torch.equal(new._rest_wheel_pos_local, rest_pos)
    assert torch.equal(new._rest_wheel_quat_local, rest_quat)
    assert float(new._rest_wheel_pos_local[0, 2]) == pytest.approx(0.30, abs=1e-3)


# --- T2: the vehicle keeps driving -------------------------------------------

def _run(rebuild_at, total, *, raycast_mode="dual_scene"):
    vs, vehs, _ = _scene(raycast_mode=raycast_mode)
    for i in range(total):
        if i == rebuild_at:
            vs.mark_config_dirty()
        vehs[0].set_inputs(throttle=0.5, brake=0.0, steer=0.1)
        vs.step()
    p = vs.physics.kinds[0]._proto
    return dict(omega=p.omega.clone(), prev_compression=p.prev_compression.clone(),
                spin=p.wheel_spin_angle.clone(),
                pos=vehs[0].get_pos().clone(), quat=vehs[0].get_quat().clone())


def _assert_same_rollout(got, want):
    for k in ("omega", "prev_compression", "spin", "pos", "quat"):
        assert torch.equal(got[k], want[k]), (
            f"{k} diverged from the no-rebuild control:\n"
            f"  rebuilt: {got[k].flatten()[:4].tolist()}\n"
            f"  control: {want[k].flatten()[:4].tolist()}")


def test_t2_forty_steps_after_a_rebuild_match_a_control_scene(cpu_genesis):
    """A one-step comparison hides carried hook / slew errors, so this compares
    40 steps PAST the rebuild against a scene that never rebuilt."""
    rebuilt = _run(rebuild_at=60, total=101)
    control = _run(rebuild_at=None, total=101)
    _assert_same_rollout(rebuilt, control)


# --- T4: a legitimate reset still resets --------------------------------------

def test_t4_vehicle_scene_reset_actually_resets_the_batched_driver(cpu_genesis):
    """Through 1.6.1 this reset NOTHING on the batched solver (only the proxy
    sync ran): omega came out of ``vs.reset()`` unchanged at
    ``[3.0921, 3.0967, 3.0921, 3.0967]``."""
    vs, vehs, _ = _scene()
    _drive(vs, vehs, 60)
    p = vs.physics.kinds[0]._proto
    assert float(p.omega.abs().max()) > 1.0
    assert bool(p._stepped_once.all())

    vs.reset()

    assert float(p.omega.abs().max()) == 0.0
    assert float(p.prev_compression.abs().max()) == 0.0
    assert float(p.wheel_spin_angle.abs().max()) == 0.0
    assert float(p.last_distances.abs().max()) == 0.0
    assert not bool(p._stepped_once.any())
    assert p._prev_init is False


# --- T5: a wheel-count change raises, and the dirty flag survives -------------

def test_t5_a_post_build_wheel_count_change_raises_and_keeps_the_scene(cpu_genesis):
    """The raycast sensor's ray count is fixed at ``build()``, so a driver
    rebuilt with fewer wheels dies on its next step with a tensor-size
    RuntimeError. It must raise HERE, before the working driver is replaced —
    and the dirty flag must be rolled back, or the broken config would be
    reported once and then silently accepted."""
    vs, vehs, cfgs = _scene(n_vehicles=2, own_cfgs=True)
    _drive(vs, vehs, 30)
    assert len(vs.physics.kinds) == 2, "premise: two separate kinds"
    mvp_before = vs.physics
    kept = vs.physics.kinds[1]._proto.omega.clone()

    cfgs[0].wheels = list(cfgs[0].wheels)[:3]
    cfgs[0].drivetrain.brake_bias = list(cfgs[0].drivetrain.brake_bias)[:3]
    vs.mark_config_dirty()

    with pytest.raises(ValueError, match="wheel count") as e1:
        vs.step()
    assert "4 -> 3" in str(e1.value)
    assert vehs[0].name in str(e1.value)
    # The cheap cfg check is the one that fires, BEFORE a replacement driver is
    # constructed (constructing one is not side-effect free — a new
    # WheelJointInternalSync writes kp/kv onto the live entity).
    assert "[config precheck]" in str(e1.value)

    # No zombie: the previous driver is still installed, still 4-wheeled.
    assert vs.physics is mvp_before
    assert [int(k.wheel_meta.n_wheels) for k in vs.physics.kinds] == [4, 4]
    assert torch.equal(vs.physics.kinds[1]._proto.omega, kept)

    # The dirty flag was NOT consumed: the next step retries and says the same.
    with pytest.raises(ValueError, match="wheel count"):
        vs.step()


def test_t5b_reordering_the_wheel_list_raises_and_keeps_the_scene(cpu_genesis):
    """Same wheel COUNT, different ORDER, is the silent-corruption case: every
    per-wheel column (omega, prev_compression, spin, steer, ray state, rest
    pose) is carried by index, and the raycast sensor's ray order is fixed at
    build() — so a swapped front-left/front-right would put each wheel's state
    on the other wheel with no exception and no warning."""
    vs, vehs, cfgs = _scene(n_vehicles=2, own_cfgs=True)
    _drive(vs, vehs, 30)
    mvp_before = vs.physics
    kept = vs.physics.kinds[1]._proto.omega.clone()

    w = list(cfgs[0].wheels)
    w[0], w[1] = w[1], w[0]
    cfgs[0].wheels = w
    vs.mark_config_dirty()

    with pytest.raises(ValueError, match="wheel order") as e:
        vs.step()
    assert vehs[0].name in str(e.value)
    assert "[config precheck]" in str(e.value)

    assert vs.physics is mvp_before, "the working driver must survive"
    assert torch.equal(vs.physics.kinds[1]._proto.omega, kept)
    with pytest.raises(ValueError, match="wheel order"):
        vs.step()


def test_minor1_the_authoritative_wheel_count_check_is_live(cpu_genesis):
    """``_check_rebuild_shapes`` reads the BUILT kind's ``wheel_meta`` and is
    the backstop behind the cfg precheck that T5 triggers. No user-reachable
    input reaches it (the precheck sees every ``len(cfg.wheels)`` change
    first), so it is driven here directly with a doctored snapshot — and the
    two messages are distinguishable, which is what lets T5 assert it was the
    precheck that fired."""
    vs, vehs, _ = _scene()
    _drive(vs, vehs, 5)
    snap = vs.physics.export_state()
    snap[0]["n_wheels"] = 3
    with pytest.raises(ValueError, match=r"wheel count changed 3 -> 4 .*\[rebuilt driver\]"):
        vs._check_rebuild_shapes(vs.physics, snap)
    # Trailing entries are tolerated the same way on both checks.
    vs._check_rebuild_shapes(vs.physics, [])

    # The identity half of the same backstop.
    snap = vs.physics.export_state()
    names = list(snap[0]["wheel_names"])
    names[0], names[1] = names[1], names[0]
    snap[0]["wheel_names"] = names
    with pytest.raises(ValueError, match=r"wheel order.*\[rebuilt driver\]"):
        vs._check_rebuild_shapes(vs.physics, snap)


# --- T6: reset row / vehicle semantics ----------------------------------------

def test_t6_kind_reset_takes_rows_and_mvp_reset_takes_vehicles(cpu_genesis):
    """K=2, n_envs=2 → 4 flat rows, env-major: row = env * K + slot.

    ``MultiVehicleKindPhysics.reset`` takes ROWS (it always did — the parameter
    was misnamed ``vehicle_ids`` through 1.6.1, which is why a per-vehicle reset
    at n_envs>1 half-worked: ``[True, False, True, True]``).
    ``MultiVehiclePhysics.reset`` takes VEHICLES and expands each to its rows."""
    vs, vehs, _ = _scene(n_envs=2, n_vehicles=2)
    kind = vs.physics.kinds[0]
    assert (kind.K, kind.n_envs, kind.NK) == (2, 2, 4)
    assert kind.slot_rows(1) == [1, 3]
    _drive(vs, vehs, 30)
    p = kind._proto
    assert bool(p._stepped_once.all())
    assert float(p.omega.abs().min()) > 0.0

    # Flat-row semantics: row 1 only (env 0's vehicle 1).
    kind.reset(rows=[1])
    assert p._stepped_once.tolist() == [True, False, True, True]
    assert float(p.omega[1].abs().max()) == 0.0
    assert float(p.omega[3].abs().max()) > 0.0

    _drive(vs, vehs, 10)
    assert bool(p._stepped_once.all())

    # Per-vehicle semantics: vehicle 1 in EVERY env → rows 1 and 3.
    vs.physics.reset(vehicle_ids=[1])
    assert p._stepped_once.tolist() == [True, False, True, False]
    assert float(p.omega[1].abs().max()) == 0.0
    assert float(p.omega[3].abs().max()) == 0.0
    assert float(p.omega[0].abs().max()) > 0.0
    assert float(p.omega[2].abs().max()) > 0.0

    # The old keyword meant rows; it now refuses rather than running wrong.
    with pytest.raises(TypeError, match="renamed to rows"):
        kind.reset(vehicle_ids=[1])


def test_t6b_a_per_vehicle_reset_does_not_wipe_another_vehicles_visuals(cpu_genesis):
    """A per-vehicle reset must be local in the VISUAL state too.

    The visual resets used to run with no index, so resetting vehicle 1 wiped
    every vehicle of the kind: vehicle 0's batched spin accumulator went
    ``[0.4316, 0.3281, -2.7312, 2.6947]`` → all zeros (K=2, n_envs=2, measured).
    Joint sync is forced on here (headless builds pick the instanced renderer
    otherwise) so both writers — the per-entity ``WheelJointInternalSync`` and
    the batched ``KindVisualBatch`` — are live."""
    cfg = car_4w_rwd_ackermann(URDF, stability="control")
    cfg.enable_wheel_joint_internal_sync = True
    vs = VehicleScene(dt=cfg.recommended_dt, substeps=10, n_envs=2,
                      show_viewer=False, init_genesis=False)
    vs.add_ground_plane(friction=1.0)
    # A camera turns wheel visuals on at build (n_envs>1 takes the joint-sync
    # fallback rather than the instanced renderer), which is what creates the
    # per-entity writers and KindVisualBatch.
    vs.add_camera(res=(160, 120), GUI=False)
    vehs = [vs.add_vehicle(URDF, car_4w_rwd_ackermann, cfg=cfg,
                           pos=(0.0, 4.0 * i, 0.6),
                           material=gs.materials.Rigid(friction=1.0))
            for i in range(2)]
    vs.build()
    _drive(vs, vehs, 30)

    kind = vs.physics.kinds[0]
    vb = kind._visual_batch
    assert len(kind.visuals) == 2 and vb is not None, (
        "premise: both visual writers are live")
    keep_batch = vb._angle[:, 0].clone()
    assert float(keep_batch.abs().max()) > 0.0, (
        "premise: the batched writer accumulated spin while driving")
    # With KindVisualBatch live it is the batch that accumulates, so the
    # per-entity writers' own accumulators stay at zero in this configuration.
    # Seed them so the per-entity reset path is observable too (the batch
    # writer can fail to be built — construction is guarded — and then these
    # are the live ones).
    for i, vis in enumerate(kind.visuals):
        vis.wheel_visual_angle = torch.full_like(
            vis.wheel_visual_angle, 0.25 + i)
    keep_vis = kind.visuals[0].wheel_visual_angle.clone()

    vs.physics.reset(vehicle_ids=[1])

    assert torch.equal(vb._angle[:, 0], keep_batch), (
        "vehicle 0's batched spin accumulator was wiped by vehicle 1's reset")
    assert torch.equal(kind.visuals[0].wheel_visual_angle, keep_vis), (
        "vehicle 0's per-entity spin accumulator was wiped")
    assert float(vb._angle[:, 1].abs().max()) == 0.0
    assert float(kind.visuals[1].wheel_visual_angle.abs().max()) == 0.0

    # A single flat ROW is just as local (env 0's vehicle 1 only).
    _drive(vs, vehs, 10)
    keep_batch = vb._angle[:, 0].clone()
    env1_of_veh1 = vb._angle[1, 1].clone()
    assert float(env1_of_veh1.abs().max()) > 0.0
    kind.reset(rows=[1])
    assert torch.equal(vb._angle[:, 0], keep_batch)
    assert float(vb._angle[0, 1].abs().max()) == 0.0
    assert torch.equal(vb._angle[1, 1], env1_of_veh1), (
        "row 1 is env0/veh1 — env 1's copy of that vehicle must survive")


# --- T7: the instanced wheel renderer follows the rebuild ---------------------

def test_t7_instanced_wheel_renderer_is_rebound(cpu_genesis):
    """One offscreen camera turns the instanced renderer on headless. Through
    1.6.1 its pose providers stayed bound to the discarded kind, which is never
    stepped again — the rendered wheels froze (old proto spin
    ``[0.3005, 0.3005, 0.5585, 0.5585]``) while physics ran on
    ``[0.3867, 0.3864, 0.6415, 0.6420]``."""
    vs, vehs, _ = _scene(camera=True)
    r = vs._wheel_renderer
    assert r is not None and r._units, "premise: the instanced path is live"
    _drive(vs, vehs, 30)

    vs.mark_config_dirty()
    vs.step()

    kinds = list(vs.physics.kinds)
    for u in r._units:
        assert u["provider"].__self__ in kinds, (
            "a render unit is still bound to the discarded kind")

    # D8 falsifier: the FIRST update after rebinding must complete. (If the
    # renderer needed its nodes recreated on a provider swap, it would raise or
    # crash here rather than merely draw a stale pose.)
    r.update()

    p0, q0 = r._units[0]["provider"]("world")
    p0, q0 = p0.clone(), q0.clone()
    spin0 = vs.physics.kinds[0]._proto.wheel_spin_angle.clone()
    _drive(vs, vehs, 10)
    r.update()
    p1, q1 = r._units[0]["provider"]("world")
    spin1 = vs.physics.kinds[0]._proto.wheel_spin_angle

    assert not torch.equal(q0, q1), "the rendered wheel orientation froze"
    assert not torch.equal(p0, p1), "the rendered wheel position froze"
    assert not torch.equal(spin0, spin1)


# --- T8: the differentiable plant re-binds ------------------------------------

def _plant_scene(rebuild_at, total):
    from genesis_vehicle.control import DifferentiablePlant
    vs, vehs, _ = _scene()
    plant = DifferentiablePlant(vehs[0], horizon=4, newton_iters=2)
    for i in range(total):
        if i == rebuild_at:
            vs.mark_config_dirty()
        vehs[0].set_inputs(throttle=0.5, brake=0.0, steer=0.1)
        vs.step()
    thr, st = plant.solve(v_long=0.0, a_target=1.0, omega_target=0.2)
    return vs, plant, np.array([thr, st], dtype=np.float64)


def test_t8_plant_rebinds_to_the_live_driver_and_solves_the_same(cpu_genesis):
    """Identity alone would pass even if the re-bind happened too late (inside
    ``_sync_hooks``, after ``_snapshot`` had already read the dead driver), so
    the first solve after the rebuild is compared NUMERICALLY against a scene
    that never rebuilt."""
    vs_r, plant_r, u_r = _plant_scene(rebuild_at=40, total=61)
    vs_c, plant_c, u_c = _plant_scene(rebuild_at=None, total=61)

    kind = vs_r.physics.kinds[0]
    assert plant_r.src.physics is kind._proto
    assert plant_r.src.kind is kind
    assert plant_r.src.resolved is kind._proto.resolved
    assert np.array_equal(u_r, u_c), (
        f"the first solve after the rebuild differs from the control: "
        f"{u_r.tolist()} vs {u_c.tolist()}")


def test_t8b_a_structural_change_under_the_plant_raises(cpu_genesis):
    """The plant refuses to re-bind across a SHAPE change.

    NB the scene-level route to this is now closed: ``_build_mvp`` raises
    ``ValueError`` on a wheel-count change before the driver is replaced (T5),
    so the plant can never be handed a 3-wheel driver by ``mark_config_dirty``.
    This drives the plant's own guard directly — it is the second line of
    defence for any other path that swaps a driver — by telling the source it
    was bound to a 3-wheel one and moving the build id."""
    from genesis_vehicle.control import DifferentiablePlant
    vs, vehs, _ = _scene()
    plant = DifferentiablePlant(vehs[0], horizon=4, newton_iters=2)
    _drive(vs, vehs, 20)
    plant.solve(v_long=0.0, a_target=1.0, omega_target=0.0)

    class _Meta:
        n_wheels = 3
    plant.src.wheel_meta = _Meta()
    vs.physics.build_id += 1

    with pytest.raises(RuntimeError, match="mark_config_dirty"):
        plant.solve(v_long=0.0, a_target=1.0, omega_target=0.0)


# --- T9: the same, in single_scene --------------------------------------------

def test_t9_single_scene_rebuild_carries_state_and_keeps_driving(cpu_genesis):
    """Both raycast modes share the rebuild path; single_scene casts in the
    scene the chassis collides in, so it gets its own rollout check."""
    vs, vehs, _ = _scene(raycast_mode="single_scene")
    _drive(vs, vehs, 40)
    old = vs.physics.kinds[0]._proto
    before = _snapshot(old)
    vs.mark_config_dirty()
    assert vs._ensure_grouped() is True
    vs._build_mvp()
    new = vs.physics.kinds[0]._proto
    assert new is not old
    for a in CARRIED:
        assert torch.equal(getattr(new, a), before[a]), a
    for a in REST_POSE_ATTRS:
        assert torch.equal(getattr(new, a), before[a]), a

    rebuilt = _run(rebuild_at=40, total=71, raycast_mode="single_scene")
    control = _run(rebuild_at=None, total=71, raycast_mode="single_scene")
    _assert_same_rollout(rebuilt, control)


# --- T10: exactly one rebuild per mark_config_dirty ---------------------------

def test_t10_one_mark_config_dirty_rebuilds_exactly_once(cpu_genesis):
    """After the fix the rebuild is invisible in the state, so a "rebuilds every
    step" regression would cost only performance and show no symptom. The
    ``build_id`` counter is process-global (a fresh MVP must never reuse the id
    its predecessor had, or staleness detection would fail), so this asserts a
    DELTA captured from THIS scene and builds no second scene."""
    vs, vehs, _ = _scene()
    _drive(vs, vehs, 5)
    b0 = vs.physics.build_id

    vs.mark_config_dirty()
    _drive(vs, vehs, 10)

    assert vs.physics.build_id == b0 + 1
