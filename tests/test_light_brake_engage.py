"""2.0.4: ``ctx.brake_rem`` and the pedal-free StaticFrictionLock engage rule.

A wheel the brake physically holds this step (``ctx.brake_rem > 0``) engages
the lock below ``min(v_thr, V_ENGAGE_MAX)`` at ANY pedal; the lock takes over
only through the creep test and never forces omega. The 2.0.3 pedal term
(``brake > brake_thr``), its forced omega and the 0.05 m/s rest-speed takeover
are gone, and ``brake_thr`` / ``REST_SPEED`` raise.

| abbr | meaning |
|---|---|
| SFL | `StaticFrictionLock` |
| B_rem | brake torque left on a wheel after `dynamics.brake_projection`, N*m |
| ctx | the `PipelineContext` passed to stability hooks |
"""

from __future__ import annotations

import math
import os
import types

import pytest
import torch

from genesis_vehicle._pipeline import compute_wheel_step
from genesis_vehicle.config import resolve
from genesis_vehicle.core import PipelineContext, build_wheel_meta
from genesis_vehicle.presets import car_4w_rwd_ackermann
from genesis_vehicle.strategies.stability import StaticFrictionLock

HERE = os.path.dirname(__file__)
CAR_URDF = os.path.abspath(os.path.join(HERE, "..", "samples", "urdf", "car_4w.urdf"))
FDT = torch.float32


def _step(*, omega, v_long, hooks, T_brake, dt=0.025, ctx=None):
    resolved = resolve(car_4w_rwd_ackermann(CAR_URDF))
    wm = build_wheel_meta(resolved, torch.device("cpu"), FDT)
    n = wm.n_wheels
    distances = (wm.rest_d - 0.5 * (wm.rest_d - wm.radius)).reshape(1, n)
    omega = torch.as_tensor(omega, dtype=FDT).expand(1, n).clone()
    zeros_n = torch.zeros(1, n, dtype=FDT)
    vel = torch.tensor([[v_long, 0.0, 0.0]], dtype=FDT)
    ang = torch.zeros(1, 3, dtype=FDT)
    if ctx is None:
        ctx = PipelineContext(throttle=torch.zeros(1), brake=torch.zeros(1), wheel_meta=wm)
    ctx.wheel_meta = wm
    ctx.vel, ctx.ang = vel, ang
    res = compute_wheel_step(
        pos=torch.zeros(1, 3, dtype=FDT), quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=FDT),
        vel=vel, ang=ang, distances=distances, steer_per_wheel=zeros_n,
        T_drive_pw=zeros_n, T_brake_pw=torch.full_like(zeros_n, T_brake), omega=omega,
        prev_compression=torch.clamp(wm.rest_d.reshape(1, n) - distances, min=0.0),
        prev_init=True, wheel_meta=wm, resolved=resolved,
        wheel_body_b=wm.positions.unsqueeze(0), up_world=torch.tensor([[0.0, 0.0, 1.0]]),
        post_tire_hooks=hooks, ctx=ctx, dt=dt, B=1, n=n)
    return res, ctx


class _Probe:
    slots = ("POST_TIRE",)

    def __init__(self, write=False):
        self.pre = "unset"
        self.post = None
        self.write = write

    def apply_pre_tire(self, ctx):
        self.pre = ctx.brake_rem

    def apply_post_tire(self, ctx):
        self.post = ctx.brake_rem
        if self.write:
            ctx.brake_rem.fill_(0.0)
            ctx.brake_rem = None


def test_brake_rem_is_none_pre_tire_and_a_detached_copy_post_tire():
    probe = _Probe()
    _step(omega=0.0, v_long=0.0, hooks=[probe], T_brake=500.0)
    assert probe.pre is None
    t = probe.post
    assert torch.is_tensor(t) and t.shape == (1, 4)
    assert t._base is None and not t.requires_grad
    assert bool((t > 0).all())                        # a stopped braked wheel is held
    assert float(t[0, 0]) == pytest.approx(500.0)


def test_brake_rem_is_reset_before_the_pre_tire_phase_of_the_next_step():
    probe = _Probe()
    _, ctx = _step(omega=0.0, v_long=0.0, hooks=[probe], T_brake=500.0)
    assert torch.is_tensor(ctx.brake_rem)
    _step(omega=0.0, v_long=0.0, hooks=[probe], T_brake=500.0, ctx=ctx)
    assert probe.pre is None and torch.is_tensor(probe.post)


def test_brake_rem_is_detached_from_a_grad_carrying_brake():
    """API-3: in a differentiable unroll B_rem carries grad (here through the
    brake torque); the hook copy must not."""
    resolved = resolve(car_4w_rwd_ackermann(CAR_URDF))
    wm = build_wheel_meta(resolved, torch.device("cpu"), FDT)
    n = wm.n_wheels
    distances = (wm.rest_d - 0.5 * (wm.rest_d - wm.radius)).reshape(1, n)
    zeros_n = torch.zeros(1, n, dtype=FDT)
    T_b = torch.full((1, n), 500.0, requires_grad=True)
    vel = torch.zeros(1, 3, dtype=FDT)
    ctx = PipelineContext(throttle=torch.zeros(1), brake=torch.zeros(1), wheel_meta=wm)
    ctx.vel, ctx.ang = vel, vel
    probe = _Probe()
    compute_wheel_step(
        pos=torch.zeros(1, 3, dtype=FDT), quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=FDT),
        vel=vel, ang=vel, distances=distances, steer_per_wheel=zeros_n, T_drive_pw=zeros_n,
        T_brake_pw=T_b, omega=torch.full((1, n), 1.0),
        prev_compression=torch.clamp(wm.rest_d.reshape(1, n) - distances, min=0.0),
        prev_init=True, wheel_meta=wm, resolved=resolved,
        wheel_body_b=wm.positions.unsqueeze(0), up_world=torch.tensor([[0.0, 0.0, 1.0]]),
        post_tire_hooks=[probe], ctx=ctx, dt=0.025, B=1, n=n)
    assert bool((probe.post > 0).all())
    assert not probe.post.requires_grad and probe.post.grad_fn is None


def test_brake_rem_is_zero_without_brake_torque():
    probe = _Probe()
    _step(omega=0.0, v_long=0.0, hooks=[probe], T_brake=0.0)
    assert bool((probe.post == 0).all())


def test_brake_rem_write_has_no_effect_on_the_pipeline():
    a, _ = _step(omega=1.0, v_long=0.3, hooks=[_Probe()], T_brake=500.0)
    b, _ = _step(omega=1.0, v_long=0.3, hooks=[_Probe(write=True)], T_brake=500.0)
    import dataclasses
    for f in (x.name for x in dataclasses.fields(a)):
        va, vb = getattr(a, f), getattr(b, f)
        assert (torch.equal(va, vb) if torch.is_tensor(va) else va == vb), f


def test_context_field_appended_last():
    import dataclasses
    names = [f.name for f in dataclasses.fields(PipelineContext)]
    assert names[-1] == "brake_rem" and PipelineContext().brake_rem is None


# --- the SFL engage term (Genesis-free hook calls) -----------------------------

class _Meta:
    def __init__(self, n=4):
        self.n_wheels = n
        self.mu_long = torch.ones(n)
        self.mu_lat = torch.ones(n)


def _ctx(*, v, brake, brake_rem, n=4, N=3000.0, F_long=0.0, db=None):
    c = types.SimpleNamespace(
        brake=torch.tensor([brake]), v_long=torch.full((1, n), v), v_lat=torch.zeros(1, n),
        N=torch.full((1, n), N), dt=0.025, wheel_meta=_Meta(n),
        F_long=torch.full((1, n), F_long), F_lat=torch.zeros(1, n), omega_override=None)
    if brake_rem is not None:
        c.brake_rem = torch.full((1, n), brake_rem)
    if db is not None:
        c.F_long_deadbeat = torch.full((1, n), db)
    return c


def test_held_wheel_engages_at_a_light_pedal_without_forcing_omega():
    hook = StaticFrictionLock()
    ctx = _ctx(v=0.2, brake=0.2, brake_rem=100.0)
    hook.apply_post_tire(ctx)
    assert bool(hook._was_active.all())
    assert ctx.omega_override is None                  # the lock never forces omega


def test_not_held_or_too_fast_does_not_engage():
    for kw in (dict(v=0.2, brake=0.2, brake_rem=0.0), dict(v=0.2, brake=0.2, brake_rem=None),
               dict(v=0.55, brake=0.2, brake_rem=100.0)):
        hook = StaticFrictionLock()
        hook.apply_post_tire(_ctx(**kw))
        assert not bool(hook._was_active.any()), kw


def test_engage_window_is_min_v_thr_and_half_a_metre_per_second():
    hook = StaticFrictionLock(v_thr=5.0)               # TankTuning's value acts as 0.5
    hook.apply_post_tire(_ctx(v=0.6, brake=0.2, brake_rem=100.0))
    assert not bool(hook._was_active.any())
    hook = StaticFrictionLock(v_thr=0.2)
    hook.apply_post_tire(_ctx(v=0.3, brake=0.2, brake_rem=100.0))
    assert not bool(hook._was_active.any())
    hook.apply_post_tire(_ctx(v=0.1, brake=0.2, brake_rem=100.0))
    assert bool(hook._was_active.all())


def test_held_term_takes_over_only_through_the_creep_test():
    # Below the 2.0.3 rest speed (0.05) but no deadbeat binding: no takeover.
    hook = StaticFrictionLock()
    for _ in range(3):
        ctx = _ctx(v=0.01, brake=0.2, brake_rem=100.0, F_long=-10.0, db=1e9)
        hook.apply_post_tire(ctx)
    assert not bool(hook._rest.any())
    assert ctx.hold_applied is not None and not bool(ctx.hold_applied.any())
    assert float(ctx.F_long[0, 0]) == -10.0              # tyre force untouched
    # Creep: the deadbeat binds and the speed did not fall -> takeover.
    ctx = _ctx(v=0.01, brake=0.2, brake_rem=100.0, F_long=-10.0, db=5.0)
    hook.apply_post_tire(ctx)
    assert bool(hook._rest.all()) and bool(ctx.hold_applied.all())


def test_the_pedal_alone_engages_nothing():
    """Since 2.0.4 a pedal above the old brake_thr (0.3) without a held wheel
    engages nothing, and a held wheel at brake 0.5 below the old 0.05 m/s rest
    speed does NOT take over without the creep test, nor force omega."""
    hook = StaticFrictionLock()
    hook.apply_post_tire(_ctx(v=0.01, brake=0.5, brake_rem=0.0))
    assert not bool(hook._was_active.any())
    hook = StaticFrictionLock()
    ctx = _ctx(v=0.01, brake=0.5, brake_rem=100.0, F_long=-10.0, db=1e9)
    hook.apply_post_tire(ctx)
    assert bool(hook._was_active.all()) and not bool(hook._rest.any())
    assert ctx.omega_override is None


def test_brake_thr_and_rest_speed_are_removed():
    """M11: keyword, positional, unknown keyword, attribute read / write /
    delete, class-level read / write each raise; nothing is silently inert."""
    with pytest.raises(TypeError, match="brake_thr was removed in 2.0.4"):
        StaticFrictionLock(brake_thr=0.3)
    with pytest.raises(TypeError, match="keyword arguments only"):
        StaticFrictionLock(0.3, 0.5)
    with pytest.raises(TypeError, match="k_sprng"):
        StaticFrictionLock(k_sprng=1)
    hook = StaticFrictionLock(v_thr=0.4, k_spring=3e5, k_damp=1e4)
    assert (hook.v_thr, hook.k_spring, hook.k_damp) == (0.4, 3e5, 1e4)
    for name in ("brake_thr", "REST_SPEED"):
        with pytest.raises(AttributeError, match="removed in 2.0.4"):
            getattr(hook, name)
        with pytest.raises(AttributeError, match="removed in 2.0.4"):
            setattr(hook, name, 0.0)
        with pytest.raises(AttributeError, match="removed in 2.0.4"):
            delattr(hook, name)
        with pytest.raises(AttributeError, match="removed in 2.0.4"):
            getattr(StaticFrictionLock, name)
        with pytest.raises(AttributeError, match="removed in 2.0.4"):
            setattr(StaticFrictionLock, name, 0.0)
        assert name not in vars(hook)
    # A hook pickled under <= 2.0.3 carries brake_thr in its __dict__: the
    # class property shadows it, so reading it still raises.
    import pickle
    h2 = StaticFrictionLock()
    h2.__dict__["brake_thr"] = 0.3
    h3 = pickle.loads(pickle.dumps(h2))
    with pytest.raises(AttributeError):
        h3.brake_thr


# --- live: the 5 degree light-brake hold ------------------------------------------------

def test_light_brake_on_a_5_degree_downhill_comes_to_rest():
    """Reference car, 5 deg downhill (gravity tilt), brake 0.3 from 2 m/s:
    2.0.3 creeps at 0.0250 m/s (the pedal term never engages at 0.3); with
    the brake-held term it rests (REST definition of plan r204's
    acceptance.json over the last 2 s)."""
    from genesis_vehicle import VehicleScene
    import genesis as gs
    import numpy as np
    VehicleScene.init_backend("cpu")
    vs = VehicleScene(dt=0.025, substeps=4, n_envs=1, show_viewer=False, solver="per_vehicle")
    vs.add_ground_plane(friction=1.0)
    veh = vs.add_vehicle(CAR_URDF, car_4w_rwd_ackermann, pos=(0.0, 0.0, 0.5), stability="control",
                         material=gs.materials.Rigid(friction=1.0))
    vs.build()
    th = math.radians(5.0)
    vs.set_gravity((9.81 * math.sin(th), 0.0, -9.81 * math.cos(th)))
    ent, p = veh.entity_main, veh.physics
    veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
    for _ in range(20):
        vs.step()
    dv = torch.zeros(1, ent.n_dofs, device=gs.device, dtype=gs.tc_float)
    dv[:, 0] = 2.0
    ent.set_dofs_velocity(dv)
    p.omega[:, :] = 2.0 / p.wheel_meta.radius.reshape(1, -1)
    veh.set_inputs(throttle=0.0, brake=0.0, steer=0.0)
    for _ in range(4):
        vs.step()
    veh.set_inputs(throttle=0.0, brake=0.3, steer=0.0)
    vp, om, pos = [], [], []
    for _ in range(int(8.0 / 0.025)):
        vs.step()
        v = ent.get_vel().reshape(-1).cpu().numpy()
        vp.append(math.hypot(v[0], v[1]))
        om.append(np.abs(p.omega.cpu().numpy().reshape(-1)) * float(p.wheel_meta.radius[0]))
        pos.append(ent.get_pos().reshape(-1).cpu().numpy()[:2].copy())
    last = int(2.0 / 0.025)
    assert float(np.mean(vp[-last:])) < 1e-3
    assert float(np.hypot(*(pos[-1] - pos[len(pos) // 2]))) <= 1e-3
    assert float(np.max(om[-last:])) <= 1e-3
