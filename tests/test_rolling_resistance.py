"""Rolling resistance is an axle resisting torque (v1.7.0).

Through v1.6.10 ``RollingResistance`` subtracted ``cr * N`` from the tyre
``F_long`` before the overshoot clamp, which cut it to zero on a freely rolling
wheel; while driving, the same ``F_long`` re-entered the wheel update and the
slip re-balanced it away. The hook had no measurable effect (a 5 m/s coast of
the reference car decelerated at 0.0073 m/s^2 against a closed-form 0.1478).
Since v1.7.0 the hook sets ``ctx.T_resist = cr * N * r * tanh(omega * r /
tanh_scale)`` in its ``apply_pre_tire`` phase, and
``_pipeline.compute_wheel_step`` caps it with ``dynamics.cap_resist_torque`` and
subtracts it from ω BEFORE the tyre. In v1.7.0 the brake was then capped
together with it and only the remainder entered the overshoot clamp; since
v1.8.0 the drive torque and the brake (``dynamics.brake_projection``, a Coulomb
projection toward zero) are ALSO applied to ω before the tyre, after the
resisting torque, and nothing of the brake enters the clamp as a torque. (An
intermediate, unreleased 1.7.0 design added ``T_resist`` to the brake INSIDE
the clamp instead; ω then alternated step to step between rolling and rolling
minus ``T_rr*dt/i`` -- measured on the reference tank up to 1.24 rad/s, on 281
of 800 steps -- and the plant's prediction depended on whether its horizon was
odd or even. The tests marked "flutter" below pin the fix.)

Three layers:

* the closed form, Genesis-free: ``DifferentiablePlant.predict`` runs the very
  same ``compute_wheel_step``, so its coast deceleration with vs without the hook
  must differ by ``sum(cr * N) / M_eff`` (``M_eff = M + sum(i_wheel / r^2)``);
* the single-step contract, Genesis-free: the cap, airborne and locked wheels,
  the hook's accumulate-don't-overwrite rule, the untouched path when no
  hook sets ``T_resist``, a rolling wheel staying rolling while the chassis gets
  ``-T_rr / r`` on the same step, and the refusal of a post-tyre ``T_resist``;
* a real CPU scene: the reference car's coast-down matches the closed form, a
  coast from 1 m/s is monotone with no step-to-step ω alternation (car and
  tank), and a low-speed coast runs down toward rest without reversing (a MEASURED
  property of this car, dt and stability profile, not a guarantee the code
  enforces on every path — what the code enforces is that brake + rolling
  resistance cannot flip ω in one step, which the single-step test pins).
"""
import dataclasses
import math
import os

import numpy as np
import pytest
import torch

from genesis_vehicle._pipeline import compute_wheel_step
from genesis_vehicle.config import resolve
from genesis_vehicle.core import PipelineContext, build_wheel_meta
from genesis_vehicle.dynamics import brake_torque_signed, cap_resist_torque
from genesis_vehicle.presets import car_4w_rwd_ackermann, tank_skid_belt
from genesis_vehicle.strategies.stability import (
    LowSpeedRegularizer, RollingResistance, StabilityHook,
)
from genesis_vehicle.tests.test_differentiable_plant import (
    CAR_URDF, TANK_URDF, _FakeSource, _plant,
)

G = 9.81


# --- 1. closed form through the plant (Genesis-free) --------------------------

def _without_rr(cfg):
    cfg.stability_hooks = [h for h in cfg.stability_hooks
                           if not isinstance(h, RollingResistance)]
    return cfg


@pytest.mark.parametrize("v", [2.0, 5.0, 10.0])
@pytest.mark.parametrize("kind", ["car", "tank"])
def test_plant_coast_includes_rolling_resistance_at_the_closed_form(kind, v):
    """``predict(0, 0)`` with the default "control" hooks minus the same
    prediction without ``RollingResistance`` equals ``-sum(cr*N)/M_eff``.

    Fake state: level chassis, every wheel at half its rest stroke (so
    ``N = k_susp * stroke / 2``, not ``m * g``), wheels exactly rolling
    (``omega = v / r``), 16-step horizon. Measured ratio 1.0008 .. 1.0017 on
    both reference vehicles (1.002 .. 1.004 under the intermediate 1.7.0
    design); through v1.6.10 it was -0.0000 .. 0.0056."""
    preset, urdf = ((car_4w_rwd_ackermann, CAR_URDF) if kind == "car"
                    else (tank_skid_belt, TANK_URDF))
    a = {}
    for keep in (True, False):
        cfg = preset(urdf)
        if not keep:
            cfg = _without_rr(cfg)
        src = _FakeSource(cfg, v_long=v)
        wm = src.wheel_meta
        r = float(wm.radius.mean())
        p = _plant(cfg, v_long=v, omega=v / r, horizon=16)
        a[keep], _ = p.predict(0.0, 0.0)
    N = wm.k_susp * 0.5 * (wm.rest_d - wm.radius)
    m_eff = src.mass_kg + float((wm.i_wheel / wm.radius ** 2).sum())
    expect = -float((wm.rolling_resistance_cr * N).sum()) / m_eff
    ratio = (a[True] - a[False]) / expect
    assert 0.85 <= ratio <= 1.05, (
        f"{kind} v={v}: plant coast decel with RR {a[True]:+.5f}, without "
        f"{a[False]:+.5f}; difference / closed form {expect:+.5f} = {ratio:.4f}")


# --- 2. the single-step contract (Genesis-free) -------------------------------

def _step(cfg, *, omega, v_long, hooks, distances=None, cr=None, dt=0.025,
          T_brake=None):
    """One ``compute_wheel_step`` on a level, evenly loaded vehicle."""
    resolved = resolve(cfg)
    dev, fdt = torch.device("cpu"), torch.float32
    wm = build_wheel_meta(resolved, dev, fdt)
    if cr is not None:
        wm.rolling_resistance_cr = torch.full_like(wm.rolling_resistance_cr, cr)
    n = wm.n_wheels
    if distances is None:
        distances = (wm.rest_d - 0.5 * (wm.rest_d - wm.radius)).reshape(1, n)
    distances = torch.as_tensor(distances, dtype=fdt).reshape(1, n)
    omega = torch.as_tensor(omega, dtype=fdt).expand(1, n).clone()
    zeros_n = torch.zeros(1, n, dtype=fdt)
    vel = torch.tensor([[v_long, 0.0, 0.0]], dtype=fdt)
    ang = torch.zeros(1, 3, dtype=fdt)
    ctx = PipelineContext(throttle=torch.zeros(1), brake=torch.zeros(1),
                          wheel_meta=wm)
    ctx.vel, ctx.ang = vel, ang
    res = compute_wheel_step(
        pos=torch.zeros(1, 3, dtype=fdt),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=fdt),
        vel=vel, ang=ang, distances=distances, steer_per_wheel=zeros_n,
        T_drive_pw=zeros_n,
        T_brake_pw=zeros_n if T_brake is None else torch.full_like(zeros_n, T_brake),
        omega=omega,
        prev_compression=torch.clamp(wm.rest_d.reshape(1, n) - distances, min=0.0),
        prev_init=True, wheel_meta=wm, resolved=resolved,
        wheel_body_b=wm.positions.unsqueeze(0), up_world=torch.tensor([[0.0, 0.0, 1.0]]),
        post_tire_hooks=hooks, ctx=ctx, dt=dt, B=1, n=n,
    )
    return res, ctx, wm


@pytest.mark.parametrize("bad", [-0.01, -1e-9, float("nan"), float("inf")])
def test_a_negative_or_non_finite_cr_is_refused_at_resolve(bad):
    """Since v1.7.0 a negative ``cr`` would make ``RollingResistance`` a
    torque that PROPELS the wheel (through 1.6.10 the overshoot clamp hid it),
    so ``resolve`` refuses it (``ConfigError`` is a ``ValueError``)."""
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    cfg.wheels[2].rolling_resistance_cr = bad
    with pytest.raises(ValueError, match="rolling_resistance_cr"):
        resolve(cfg)


def test_a_cr_mutated_after_resolve_is_refused_at_wheel_meta_build():
    """The rebuild path (``mark_config_dirty()``, ``DifferentiablePlant``) goes
    through ``build_wheel_meta`` without re-running ``resolve``."""
    resolved = resolve(tank_skid_belt(TANK_URDF))
    build_wheel_meta(resolved, torch.device("cpu"), torch.float32)   # fine as is
    resolved.wheels[0].rolling_resistance_cr = -0.05
    with pytest.raises(ValueError, match="rolling_resistance_cr"):
        build_wheel_meta(resolved, torch.device("cpu"), torch.float32)


def test_zero_cr_is_accepted():
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    for w in cfg.wheels:
        w.rolling_resistance_cr = 0.0
    wm = build_wheel_meta(resolve(cfg), torch.device("cpu"), torch.float32)
    assert bool((wm.rolling_resistance_cr == 0.0).all())


@pytest.mark.parametrize("preset, urdf", [
    (car_4w_rwd_ackermann, CAR_URDF), (tank_skid_belt, TANK_URDF),
])
def test_presets_resolve_with_a_non_negative_cr(preset, urdf):
    wm = build_wheel_meta(resolve(preset(urdf)), torch.device("cpu"), torch.float32)
    assert bool((wm.rolling_resistance_cr >= 0.0).all())


def test_t_resist_is_followed_only_by_the_1_8_0_context_fields():
    """``T_resist`` was appended at the END in v1.7.0 so a positional
    construction keeps its meaning; v1.8.0 appended its four fields after it,
    in this order, all defaulting to None (so positional construction up to
    ``T_resist`` still means what it meant)."""
    names = [f.name for f in dataclasses.fields(PipelineContext)]
    i = names.index("T_resist")
    assert names[i:] == ["T_resist", "F_long_bounds", "F_long_deadbeat",
                         "hold_applied", "deadbeat_scale"]
    ctx = PipelineContext()
    for name in names[i:]:
        assert getattr(ctx, name) is None, name


def test_no_hook_leaves_t_resist_none():
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    for hooks in ([], [LowSpeedRegularizer()]):
        _, ctx, _ = _step(cfg, omega=10.0, v_long=3.5, hooks=hooks)
        assert ctx.T_resist is None


def test_hook_writes_the_axle_torque_formula():
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    hook = RollingResistance()
    res, ctx, wm = _step(cfg, omega=10.0, v_long=3.5, hooks=[hook])
    r = wm.radius.unsqueeze(0)
    want = (wm.rolling_resistance_cr.unsqueeze(0) * res.N * r
            * torch.tanh(10.0 * r / hook.tanh_scale))
    assert torch.allclose(ctx.T_resist, want, rtol=1e-6, atol=0.0)
    assert bool((ctx.T_resist > 0).all())        # same sign as omega


def test_hooks_accumulate_into_t_resist():
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    _, one, _ = _step(cfg, omega=10.0, v_long=3.5, hooks=[RollingResistance()])
    _, two, _ = _step(cfg, omega=10.0, v_long=3.5,
                      hooks=[RollingResistance(), RollingResistance()])
    assert torch.allclose(two.T_resist, 2.0 * one.T_resist, rtol=1e-6, atol=0.0)


def test_tiny_omega_with_huge_cr_cannot_flip_the_wheel():
    """cr = 5 at omega = 0.01 rad/s asks for a torque that would reverse the
    wheel in one step many times over; the cap stops it at zero."""
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    dt = 0.025
    w0 = 0.01
    r = float(build_wheel_meta(resolve(cfg), torch.device("cpu"),
                               torch.float32).radius.mean())
    res, ctx, wm = _step(cfg, omega=w0, v_long=w0 * r, hooks=[RollingResistance()],
                         cr=5.0, dt=dt)
    # The request really is larger than what zeroes omega this step...
    assert bool((ctx.T_resist * dt / wm.i_wheel.unsqueeze(0) > w0).all())
    # ...and the wheel still ends at or above zero (float noise allowed).
    assert bool((res.new_omega >= -1e-6).all()), res.new_omega
    assert bool((res.new_omega <= w0).all()), res.new_omega


def test_airborne_and_locked_wheels_get_no_rolling_resistance():
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    wm0 = build_wheel_meta(resolve(cfg), torch.device("cpu"), torch.float32)
    n = wm0.n_wheels
    dist = (wm0.rest_d - 0.5 * (wm0.rest_d - wm0.radius)).clone()
    dist[0] = wm0.rest_d[0] + 0.1                  # wheel 0 airborne: N == 0
    omega = torch.full((n,), 10.0)
    omega[1] = 0.0                                 # wheel 1 locked
    res, ctx, _ = _step(cfg, omega=omega, v_long=3.5, distances=dist,
                        hooks=[RollingResistance()])
    assert float(res.N[0, 0]) == 0.0
    assert float(ctx.T_resist[0, 0]) == 0.0
    assert float(ctx.T_resist[0, 1]) == 0.0
    assert bool((ctx.T_resist[0, 2:] > 0).all())   # the others do get it


def test_the_cap_never_weakens_braking():
    """``|cap(Tb + Trr)| >= |Tb|`` for a capped brake torque Tb and a resisting
    torque Trr of the same sign — the combined cap cannot make the brake alone
    weaker than ``brake_torque_signed`` made it."""
    from genesis_vehicle.dynamics import cap_resist_torque
    g = torch.Generator().manual_seed(0)
    omega = (torch.rand(4096, generator=g) - 0.5) * 40.0
    i_w = torch.rand(4096, generator=g) * 5.0 + 0.1
    dt = 0.025
    with pytest.warns(DeprecationWarning):          # deprecated since 1.8.0
        tb = brake_torque_signed(torch.rand(4096, generator=g) * 3000.0, omega,
                                 dt=dt, i_wheel=i_w)
    trr = torch.rand(4096, generator=g) * 200.0 * torch.tanh(omega / 0.5)
    tot = cap_resist_torque(tb + trr, omega, dt, i_w)
    assert bool((tot.abs() >= tb.abs() - 1e-4).all())
    assert bool((tot.abs() <= omega.abs() * i_w / dt + 1e-4).all())
    assert bool((torch.sign(tot) * torch.sign(omega) >= 0).all())


def _kind(kind):
    return ((car_4w_rwd_ackermann, CAR_URDF) if kind == "car"
            else (tank_skid_belt, TANK_URDF))


@pytest.mark.parametrize("v", [2.0, 5.0, -2.0])
@pytest.mark.parametrize("kind", ["car", "tank"])
def test_a_rolling_wheel_stays_rolling_and_the_chassis_gets_the_rr_force(kind, v):
    """flutter: one step from exact rolling (``omega = v / r``), only
    ``RollingResistance``. The torque is applied to ω before the tyre, so the
    tyre sees the slip it causes, the overshoot clamp stops the wheel at exactly
    rolling again, and the chassis receives ``F_long = -T_rr / r`` on THIS step.

    Under the intermediate 1.7.0 design (T_resist inside the clamp) the same
    step returned ``F_long = 0`` and left ω at rolling minus ``T_rr*dt/i``; the
    force arrived one step later, and ω alternated."""
    preset, urdf = _kind(kind)
    cfg = preset(urdf)
    r = float(build_wheel_meta(resolve(cfg), torch.device("cpu"),
                               torch.float32).radius.mean())
    dt = 0.025
    res, ctx, wm = _step(cfg, omega=v / r, v_long=v, hooks=[RollingResistance()],
                         dt=dt)
    i_w = wm.i_wheel.unsqueeze(0)
    t_rr = cap_resist_torque(ctx.T_resist, torch.full_like(ctx.T_resist, v / r),
                             dt, i_w)
    assert bool((t_rr * math.copysign(1.0, v) > 0).all())     # torque is live
    want_omega = torch.full_like(res.new_omega, v / r)
    assert torch.allclose(res.new_omega, want_omega, rtol=1e-4, atol=0.0), (
        f"{kind} v={v}: omega {res.new_omega.tolist()} vs rolling {v / r:.5f}")
    want_f = -t_rr / wm.radius.unsqueeze(0)
    assert torch.allclose(res.F_long, want_f, rtol=1e-3, atol=0.0), (
        f"{kind} v={v}: F_long {res.F_long.tolist()} vs -T_rr/r {want_f.tolist()}")


def test_brake_plus_rr_never_reverses_omega():
    """Brake + rolling resistance cannot reverse ω in one step (the guards:
    ``cap_resist_torque`` on ``T_resist`` before the tyre, then, since v1.8.0,
    the brake as ``dynamics.brake_projection`` — a projection toward zero that
    cannot cross it — and the remaining brake capacity applied to the ω update
    the same way; through v1.7.0 the second guard was the cap of the brake +
    resistance sum, of which only the remainder entered the clamp).
    Deterministic grid over ω, cr, brake torque and chassis speed (rolling, or
    a stopped chassis under a spinning wheel)."""
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    r = float(build_wheel_meta(resolve(cfg), torch.device("cpu"),
                               torch.float32).radius.mean())
    worst = 0.0
    for w0 in (-1.0, -0.3, -0.01, 0.01, 0.3, 1.0):
        for cr in (0.0, 0.05, 1.0, 5.0):
            for tb in (0.0, 50.0, 500.0, 5000.0):
                for v in (w0 * r, 0.0):
                    res, _, _ = _step(cfg, omega=w0, v_long=v,
                                      hooks=[RollingResistance()], cr=cr,
                                      T_brake=tb)
                    worst = min(worst, float((res.new_omega * w0).min()))
    assert worst >= -1e-6, f"omega reversed: min(new_omega * omega) = {worst}"


def test_zero_cr_rolling_resistance_is_value_identical_to_no_hook():
    """``cr = 0`` gives ``T_resist = 0``: ω before the tyre (after the brake
    projection) is unchanged, every output equals the no-hook step."""
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    for tb in (0.0, 200.0):
        a, _, _ = _step(cfg, omega=10.0, v_long=3.5, hooks=[RollingResistance()],
                        cr=0.0, T_brake=tb)
        b, _, _ = _step(cfg, omega=10.0, v_long=3.5, hooks=[], cr=0.0, T_brake=tb)
        for f in ("new_omega", "F_long", "F_lat", "kappa", "total_F", "total_T"):
            assert torch.equal(getattr(a, f), getattr(b, f)), (tb, f)


class _PostTireWriter(StabilityHook):
    slots = ("POST_TIRE",)

    def __init__(self, in_place):
        self.in_place = in_place

    def apply_post_tire(self, ctx):
        if self.in_place:
            ctx.T_resist.add_(1.0)
        else:
            ctx.T_resist = torch.ones_like(ctx.N)


@pytest.mark.parametrize("hooks", ["post_only", "rr_then_rebind", "rr_then_in_place"])
def test_t_resist_written_after_the_tyre_raises(hooks):
    """``T_resist`` is consumed BEFORE the tyre; a POST_TIRE write would be
    silently ignored, so the pipeline refuses it (rebinding is detected by
    identity, an in-place write by the tensor's version counter)."""
    hk = {"post_only": [_PostTireWriter(False)],
          "rr_then_rebind": [RollingResistance(), _PostTireWriter(False)],
          "rr_then_in_place": [RollingResistance(), _PostTireWriter(True)]}[hooks]
    with pytest.raises(RuntimeError, match="apply_pre_tire"):
        _step(car_4w_rwd_ackermann(CAR_URDF), omega=10.0, v_long=3.5, hooks=hk)


def test_inference_mode_runs_and_an_in_place_post_write_cannot_change_the_physics():
    """An inference-mode tensor has no version counter, so the in-place check
    is skipped there (rebinding is still refused). The pipeline consumes its
    own snapshot of ``T_resist``, so a post-tyre in-place write changes
    nothing: the result equals the plain rolling-resistance step."""
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    ref, _, _ = _step(cfg, omega=10.0, v_long=3.5, hooks=[RollingResistance()])
    with torch.inference_mode():
        a, _, _ = _step(cfg, omega=10.0, v_long=3.5, hooks=[RollingResistance()])
        b, _, _ = _step(cfg, omega=10.0, v_long=3.5,
                        hooks=[RollingResistance(), _PostTireWriter(True)])
        with pytest.raises(RuntimeError, match="apply_pre_tire"):
            _step(cfg, omega=10.0, v_long=3.5,
                  hooks=[RollingResistance(), _PostTireWriter(False)])
    for f in ("new_omega", "F_long", "total_F"):
        assert torch.equal(getattr(a, f), getattr(ref, f)), f
        assert torch.equal(getattr(b, f), getattr(ref, f)), f


def test_a_duck_typed_hook_with_one_phase_method_gets_only_that_call():
    """The pipeline looks both phase methods up with ``getattr``: a hook object
    that has only one of them (not a ``StabilityHook``) is called for that phase
    and skipped for the other."""
    calls = []

    class OnlyPost:
        def apply_post_tire(self, ctx):
            calls.append(("post", ctx.F_long is not None))

    class OnlyPre:
        def apply_pre_tire(self, ctx):
            calls.append(("pre", ctx.F_long is None))

    _step(car_4w_rwd_ackermann(CAR_URDF), omega=10.0, v_long=3.5,
          hooks=[OnlyPost(), OnlyPre()])
    assert calls == [("pre", True), ("post", True)]


@pytest.mark.parametrize("throttle", [0.0, 0.3])
@pytest.mark.parametrize("kind", ["car", "tank"])
def test_plant_prediction_does_not_depend_on_horizon_parity(kind, throttle):
    """flutter / N4: under the intermediate 1.7.0 design ω alternated inside
    the plant's unroll too, so ``predict`` / ``jacobian`` depended on whether
    the horizon was odd or even. Now h3 and h4 agree within 1%, at coast AND
    under throttle 0.3.

    The throttle row is new in v1.8.0: on the released 1.7.0 the drive torque
    still went through the clamp the old way and the parity held at coast only.
    The values this test compares, on the 1.8.0 tree (Genesis-free
    ``_FakeSource``, its explicit masses, v 5 m/s, ω = v/r, CPU): d a /
    d throttle at throttle 0.3, h3 vs h4 — car 1.0707 vs 1.0679, tank
    3.7665 vs 3.7585. The fake source passes explicit mass properties, so the
    plant's rigid-body deadbeat scale (Z) is active here."""
    preset, urdf = _kind(kind)
    v = 5.0
    r = float(_FakeSource(preset(urdf), v_long=v).wheel_meta.radius.mean())
    out = {}
    for h in (3, 4):
        p = _plant(preset(urdf), v_long=v, omega=v / r, horizon=h)
        a, _ = p.predict(throttle, 0.0)
        j = float(p.jacobian(throttle, 0.0)[0][0, 0])
        out[h] = (a, j)
    for i, name in enumerate(("predict a", "d a / d throttle")):
        x3, x4 = out[3][i], out[4][i]
        assert abs(x3 - x4) <= 0.01 * abs(x4), (
            f"{kind} throttle {throttle}: {name} h3 {x3:+.5f} vs h4 {x4:+.5f}")


# --- 3. a real CPU scene ------------------------------------------------------

genesis = pytest.importorskip("genesis")

DT, SUBSTEPS = 0.025, 4
URDF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "samples", "urdf", "car_4w.urdf"))


TANK_SCENE_URDF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                               "samples", "urdf", "tank_ray.urdf"))


def _coast_scene(urdf, preset):
    from genesis_vehicle import VehicleScene
    from genesis_vehicle.control.plant import plant_mass_properties
    VehicleScene.init_backend("cpu")
    vs = VehicleScene(dt=DT, substeps=SUBSTEPS, n_envs=1,
                      raycast_mode="dual_scene", show_viewer=False,
                      solver="per_vehicle")
    vs.add_ground_plane(friction=1.0)
    cfg = preset(urdf, stability="control")
    veh = vs.add_vehicle(urdf, preset, cfg=cfg, pos=(0.0, 0.0, 1.0),
                         material=genesis.materials.Rigid(friction=1.0))
    vs.build()
    phys, ent = veh.physics, veh.entity_main
    wm = phys.wheel_meta
    M = plant_mass_properties(ent).mass
    m_eff = M + float((wm.i_wheel / wm.radius ** 2).sum())

    def v_forward():
        v = ent.get_vel().reshape(3).cpu().numpy()
        q = ent.get_quat().reshape(4).cpu().numpy()
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                         1 - 2 * (q[2] ** 2 + q[3] ** 2))
        return float(v[0] * math.cos(yaw) + v[1] * math.sin(yaw))

    def launch(v0):
        """Settle at rest for 1 s, then set chassis speed v0 with the wheels
        exactly rolling."""
        vs.scene.reset()
        phys.reset()
        veh.set_inputs(throttle=0.0, brake=0.0, steer=0.0)
        for _ in range(int(1.0 / DT)):
            vs.step()
        dv = torch.zeros(1, ent.n_dofs, device=genesis.device, dtype=genesis.tc_float)
        dv[0, 0] = v0
        ent.set_dofs_velocity(dv)
        phys.omega[:, :] = v0 / wm.radius

    def coast(seconds, omegas=None):
        veh.set_inputs(throttle=0.0, brake=0.0, steer=0.0)
        out = []
        for _ in range(int(round(seconds / DT))):
            vs.step()
            out.append(v_forward())
            if omegas is not None:
                omegas.append(phys.omega.reshape(-1).cpu().numpy().copy())
        return out

    return dict(cfg=cfg, wm=wm, M=M, m_eff=m_eff, launch=launch, coast=coast)


@pytest.fixture(scope="module")
def car_scene():
    return _coast_scene(URDF, car_4w_rwd_ackermann)


@pytest.fixture(scope="module")
def tank_scene():
    return _coast_scene(TANK_SCENE_URDF, tank_skid_belt)


def test_coast_down_matches_the_closed_form(car_scene):
    """5 m/s coast, 1 s window after 0.5 s settling: deceleration within 10% of
    ``(cr*g + F_aero/M) * M / M_eff``. Measured 0.1473 vs 0.1472 over a 2 s
    window (CPU, genesis-world 1.4.5); through v1.6.10, 0.0073."""
    s = car_scene
    s["launch"](5.0)
    v_a = s["coast"](0.5)[-1]
    v_b = s["coast"](1.0)[-1]
    decel = v_a - v_b
    v_mid = 0.5 * (v_a + v_b)
    ch = s["cfg"].chassis
    aero = 0.5 * ch.air_density * ch.drag_area * v_mid * v_mid / s["M"]
    cr = float(s["wm"].rolling_resistance_cr.mean())
    model = (cr * G + aero) * s["M"] / s["m_eff"]
    assert decel == pytest.approx(model, rel=0.10), (
        f"coast decel {decel:.4f} m/s^2 vs closed form {model:.4f}")


def test_low_speed_coast_runs_down_without_reversing(car_scene):
    """From 0.5 m/s, 10 s of coasting: the car slows well below its start and
    never goes backwards. Below 0.5 m/s ``LowSpeedRegularizer`` scales the tyre
    force by ``moving`` and the ``tanh`` is in its linear region, so the tail is
    slower than the closed form (measured on this car: 1.0 -> 0.0063 m/s in
    20 s, never negative; 0.032 under the intermediate 1.7.0 design); this
    pins that behaviour for the reference car at dt 0.025, it is not a
    code-enforced guarantee for every vehicle and dt."""
    s = car_scene
    s["launch"](0.5)
    v = s["coast"](10.0)
    assert v[-1] < 0.25, f"barely slowed: {v[0]:.4f} -> {v[-1]:.4f} m/s"
    assert min(v) >= -1e-4, f"reversed: min {min(v):.6f} m/s"


@pytest.mark.parametrize("scene", ["car_scene", "tank_scene"])
def test_coast_is_monotone_and_omega_does_not_alternate(scene, request):
    """flutter: coast from 1.0 m/s for 10 s, default "control" profile, CPU,
    dt 0.025 x 4 substeps. No wheel's ω changes direction on consecutive steps
    with both steps above 0.01 rad/s (1.6.10 measured 0 on both vehicles at
    that amplitude over 20 s), no step changes ω by more than 0.05 rad/s, and
    chassis speed never rises by more than 1e-5 m/s in a step.

    Measured over 20 s on this machine: max |d omega| 0.0124 (car) / 0.0316
    (tank) rad/s, 0 rises. The intermediate 1.7.0 design alternated from
    t = 0.05 s: car 479 / tank 245 alternations at 0.01 rad/s, max |d omega|
    0.227 / 1.237 rad/s, speed rose on 315 / 133 steps."""
    s = request.getfixturevalue(scene)
    s["launch"](1.0)
    W = []
    v = np.asarray(s["coast"](10.0, W))
    d = np.diff(np.asarray(W), axis=0)
    tol = 0.01
    alt = ((np.sign(d[1:]) * np.sign(d[:-1]) < 0)
           & (np.abs(d[1:]) > tol) & (np.abs(d[:-1]) > tol)).sum(axis=0)
    assert int(alt.max()) == 0, f"omega alternations per wheel: {alt.tolist()}"
    assert float(np.abs(d).max()) <= 0.05, f"max |d omega| {np.abs(d).max():.4f}"
    rises = np.diff(v)
    assert int((rises > 1e-5).sum()) == 0, (
        f"speed rose on {(rises > 1e-5).sum()} steps, max {rises.max():.2e} m/s")
