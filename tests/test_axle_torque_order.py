"""Axle torques before the tyre, the Coulomb brake and the friction ellipse on
the transmitted force (v1.8.0) — the Genesis-free single-step contract.

Every test here runs ONE ``_pipeline.compute_wheel_step`` (or a
``DifferentiablePlant`` on the Genesis-free ``_FakeSource``) on the reference
car / tank presets with ``tests/data`` URDFs, CPU, float32. The real-scene
tests of the same release are in ``tests/test_low_speed_deadbeats.py``.

| abbr | meaning |
|---|---|
| ω, ω_d, ω_t | wheel spin before the step / after resisting + drive torque / after the brake projection |
| B_rem | brake capacity left on a wheel the brake stopped this step (N*m) |
| B2 | friction ellipse evaluated at the TRANSMITTED (clamped) longitudinal force |
| κ_eq | slip at which the tyre's own combined curve gives the transmitted force |
| D / Y / Z | held-wheel deadbeat / lateral deadbeat / rigid-body scale of both |
| N-numbers | test ids of the 1.8.0 plan (N1 ... N29) |

Which tests pass on the 1.7.0 tree (``git archive af879d3``) BY DESIGN, and
why, is stated in each docstring that does; every other test here fails there.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from genesis_vehicle._pipeline import compute_wheel_step
from genesis_vehicle.config import resolve
from genesis_vehicle.core import PipelineContext, build_wheel_meta
from genesis_vehicle.presets import car_4w_rwd_ackermann, tank_skid_belt
from genesis_vehicle.strategies.stability import RollingResistance, StabilityHook
from genesis_vehicle.tests.test_differentiable_plant import (
    CAR_URDF, TANK_URDF, _FakeSource, _plant,
)
from genesis_vehicle.tire_models import PacejkaAnisotropic, TireModel

FDT = torch.float32
DT = 0.025


def _kind(kind):
    return ((car_4w_rwd_ackermann, CAR_URDF) if kind == "car"
            else (tank_skid_belt, TANK_URDF))


def _setup(cfg):
    resolved = resolve(cfg)
    wm = build_wheel_meta(resolved, torch.device("cpu"), FDT)
    return resolved, wm


def _one_step(resolved, wm, *, omega, v_long, v_lat=0.0, T_drive=0.0,
              T_brake=0.0, hooks=(), dt=DT, yaw_inertia=None, distances=None,
              w_z=0.0, B=1):
    """One compute_wheel_step on a level, evenly loaded vehicle (every wheel at
    half its rest stroke). Scalars broadcast to (B, n); tensors pass through."""
    n = wm.n_wheels

    def bn(x):
        return torch.as_tensor(x, dtype=FDT).expand(B, n).clone()

    if distances is None:
        distances = (wm.rest_d - 0.5 * (wm.rest_d - wm.radius)).reshape(1, n)
    distances = torch.as_tensor(distances, dtype=FDT).reshape(-1, n).expand(B, n).clone()
    v_long_t = torch.as_tensor(v_long, dtype=FDT).reshape(-1).expand(B)
    v_lat_t = torch.as_tensor(v_lat, dtype=FDT).reshape(-1).expand(B)
    vel = torch.stack([v_long_t, v_lat_t, torch.zeros(B, dtype=FDT)], dim=1)
    ang = torch.zeros(B, 3, dtype=FDT)
    ang[:, 2] = w_z
    ctx = PipelineContext(throttle=torch.zeros(B), brake=torch.zeros(B),
                          wheel_meta=wm)
    ctx.vel, ctx.ang = vel, ang
    res = compute_wheel_step(
        pos=torch.zeros(B, 3, dtype=FDT),
        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=FDT).expand(B, 4).clone(),
        vel=vel, ang=ang, distances=distances,
        steer_per_wheel=torch.zeros(B, n, dtype=FDT),
        T_drive_pw=bn(T_drive), T_brake_pw=bn(T_brake), omega=bn(omega),
        prev_compression=torch.clamp(wm.rest_d.reshape(1, n) - distances, min=0.0),
        prev_init=True, wheel_meta=wm, resolved=resolved,
        wheel_body_b=wm.positions.unsqueeze(0).expand(B, n, 3),
        up_world=torch.tensor([[0.0, 0.0, 1.0]]).expand(B, 3),
        post_tire_hooks=list(hooks), ctx=ctx, dt=dt, B=B, n=n,
        # Passed only when given, so the same helper runs on a pre-1.8.0 tree
        # (the non-vacuity check runs these tests against the 1.7.0 archive).
        **({} if yaw_inertia is None else dict(yaw_inertia=yaw_inertia)),
    )
    return res, ctx


class _Recorder(StabilityHook):
    """POST_TIRE hook that snapshots what the pipeline hands the hooks."""
    slots = ("POST_TIRE",)

    def __init__(self):
        self.seen = {}

    def apply_post_tire(self, ctx):
        for k in ("omega", "F_long", "F_lat", "F_long_deadbeat", "hold_applied",
                  "deadbeat_scale"):
            v = getattr(ctx, k, None)
            self.seen[k] = v.clone() if torch.is_tensor(v) else v
        b = getattr(ctx, "F_long_bounds", None)
        self.seen["F_long_bounds"] = None if b is None else (b[0].clone(), b[1].clone())


class _RecordingTire(PacejkaAnisotropic):
    """Pacejka whose EVERY call is recorded. Overriding __call__ sends
    at_long_force to the generic grid (the subclass guard), so the first record
    of a step is the main tyre call."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def __call__(self, v_long, v_lat, v_roll, N, wm):
        self.calls.append(v_roll.detach().clone())
        return PacejkaAnisotropic.__call__(self, v_long, v_lat, v_roll, N, wm)


# --- N1: the tyre sees r * clamp(omega_t, +-omega_max) ------------------------

@pytest.mark.parametrize("case", ["drive", "brake", "over_omega_max", "rr"])
def test_n1_the_tyre_sees_the_wheel_speed_after_every_axle_torque(case):
    """The main tyre call's ``v_roll`` is ``r * clamp(omega_t, +-omega_max)``
    with ``omega_t`` = ω minus the capped rolling resistance, plus the drive
    torque, then the brake projection — all of THIS step. Through 1.7.0 the
    tyre saw ω after the resisting torque only (drive and brake acted after
    it); so the ``rr`` case passes on the 1.7.0 tree too, by design, and the
    other three fail there."""
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    resolved, wm = _setup(cfg)
    tire = _RecordingTire()
    resolved.tire = tire
    r = wm.radius.unsqueeze(0)
    i_w = wm.i_wheel.unsqueeze(0)
    om_max = float(resolved.chassis.omega_max)
    v = 5.0
    omega = v / float(wm.radius[0])
    T_d, T_b, hooks = 0.0, 0.0, []
    if case == "drive":
        T_d = 200.0
    elif case == "brake":
        T_b = 100.0
    elif case == "over_omega_max":
        omega, T_d = om_max - 0.5, 900.0          # drive pushes past omega_max
    else:
        hooks = [RollingResistance()]
    rec = _Recorder()
    _one_step(resolved, wm, omega=omega, v_long=v, T_drive=T_d, T_brake=T_b,
              hooks=hooks + [rec])
    om = torch.full((1, wm.n_wheels), omega, dtype=FDT)
    if case == "rr":
        from genesis_vehicle.dynamics import cap_resist_torque
        t_rr = (wm.rolling_resistance_cr.unsqueeze(0) * _one_step(
            resolve(cfg), wm, omega=omega, v_long=v)[0].N * r
            * torch.tanh(om * r / 0.5))
        om = om - cap_resist_torque(t_rr, om, DT, i_w) * DT / i_w
    om = om + T_d / i_w * DT
    om = om - torch.sign(om) * T_b / i_w * DT
    want = r * torch.clamp(om, -om_max, om_max)
    assert torch.allclose(tire.calls[0], want, rtol=1e-6, atol=1e-6), (
        case, tire.calls[0].tolist(), want.tolist())
    assert torch.allclose(rec.seen["omega"], om, rtol=1e-6, atol=1e-6)


# --- N2: a rolling wheel under drive / below-lock brake stays rolling ---------

@pytest.mark.parametrize("v", [2.0, 5.0, -2.0])
@pytest.mark.parametrize("kind", ["car", "tank"])
@pytest.mark.parametrize("torque", ["drive", "brake"])
def test_n2_a_rolling_wheel_passes_the_axle_torque_to_the_chassis_in_one_step(
        kind, v, torque):
    """From exact rolling, a drive torque T (or a brake torque below what
    stops the wheel) leaves the wheel at rolling and puts ``F_long = T / r``
    (``-T_b / r`` braking, sign of travel) on the chassis on THIS step: the
    tyre is evaluated after the torque, its force reaches the clamp limit and
    the clamp passes exactly the torque. Through 1.7.0 the tyre was evaluated
    BEFORE the torque, so from exact rolling it returned 0 and the force
    arrived one step later. No hooks (isolates the axle torques)."""
    preset, urdf = _kind(kind)
    resolved, wm = _setup(preset(urdf))
    r = float(wm.radius.mean())
    T = 60.0 if kind == "car" else 600.0
    sgn = math.copysign(1.0, v)
    if torque == "drive":
        T_d, T_b, want_f = sgn * T, 0.0, sgn * T / r
    else:
        T_d, T_b, want_f = 0.0, T, -sgn * T / r
    stop_T = abs(v / r) * float(wm.i_wheel.mean()) / DT
    assert T < stop_T                          # below lock: the wheel keeps turning
    res, _ = _one_step(resolved, wm, omega=v / r, v_long=v, T_drive=T_d, T_brake=T_b)
    assert torch.allclose(res.new_omega, torch.full_like(res.new_omega, v / r),
                          rtol=1e-4, atol=0.0), res.new_omega.tolist()
    assert torch.allclose(res.F_long, torch.full_like(res.F_long, want_f),
                          rtol=1e-3, atol=0.0), (res.F_long.tolist(), want_f)


# --- N8: rolling resistance first, exactly the 1.7.0 omega' -------------------

@pytest.mark.parametrize("kind", ["car", "tank"])
def test_n8_with_no_drive_and_no_brake_omega_t_is_the_1_7_0_omega_prime(kind):
    """``omega_t`` with zero drive and zero brake is, bit for bit, the v1.7.0
    ``omega' = omega - cap_resist_torque(T_resist, omega) * dt / i_wheel``:
    rolling resistance keeps its v1.7.0 position and form (first, against the
    pre-step ω). Passes on the 1.7.0 tree too (there POST_TIRE ``ctx.omega``
    IS ω′) — it pins that 1.8.0 did not move it."""
    from genesis_vehicle.dynamics import cap_resist_torque
    preset, urdf = _kind(kind)
    resolved, wm = _setup(preset(urdf))
    rec = _Recorder()
    rr = RollingResistance()
    omega = torch.tensor([[7.0, -3.0, 0.01] + [5.0] * (wm.n_wheels - 3)], dtype=FDT)
    res, ctx = _one_step(resolved, wm, omega=omega, v_long=2.0, hooks=[rr, rec])
    i_w = wm.i_wheel.unsqueeze(0)
    T_rr = cap_resist_torque(ctx.T_resist, omega, DT, i_w)
    assert torch.equal(rec.seen["omega"], omega - T_rr * DT / i_w)


# --- N9: B2 accuracy: closed form vs a K = 512 grid; the K = 16 grid bound ----

class _PacejkaGrid(PacejkaAnisotropic):
    """The Pacejka combined law with the GENERIC grid (overriding __call__
    trips the closed-form guard)."""

    def __init__(self, K):
        super().__init__()
        self.AT_LONG_FORCE_GRID = K

    def __call__(self, *a):
        return PacejkaAnisotropic.__call__(self, *a)


def _random_states(wm, B, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = wm.n_wheels
    v = (torch.rand(B, generator=g) * 14.7 + 0.3) * torch.where(
        torch.rand(B, generator=g) < 0.2, -1.0, 1.0)
    v_lat = torch.rand(B, generator=g) * 2 - 1
    omega = (v.unsqueeze(1) / wm.radius.unsqueeze(0)) * (
        1 + (torch.rand(B, n, generator=g) * 0.1 - 0.05))
    T_d = torch.zeros(B, n)
    T_d[:, wm.positions[:, 0] < 0] = (torch.rand(B, 1, generator=g) * 500) * torch.sign(v).unsqueeze(1)
    T_b = torch.where(torch.rand(B, 1, generator=g) < 0.3,
                      torch.rand(B, 1, generator=g) * 750,
                      torch.zeros(B, 1)).expand(B, n).clone()
    return v, v_lat, omega, T_d, T_b


def _b2_errors(tire, ref_tire, B=4000):
    """kappa rel. error and F_lat error / muN of `tire` vs `ref_tire` over the
    rows where B2 is active."""
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    cfg.stability_hooks = []
    out = {}
    for name, t in (("t", tire), ("ref", ref_tire)):
        resolved, wm = _setup(cfg)
        resolved.tire = t
        v, vl, om, T_d, T_b = _random_states(wm, B)
        rec = _Recorder()
        res, _ = _one_step(resolved, wm, omega=om, v_long=v, v_lat=vl,
                           T_drive=T_d, T_brake=T_b, hooks=[rec], B=B)
        out[name] = (res, rec.seen, wm)
    res, seen, wm = out["t"]
    ref, ref_seen, _ = out["ref"]
    muN = wm.mu_long.unsqueeze(0) * ref.N
    lo, hi = ref_seen["F_long_bounds"]
    db = ref_seen["F_long_deadbeat"]
    raw = ref_seen["F_long"]
    active = torch.maximum(torch.minimum(raw, torch.minimum(hi, db)),
                           torch.maximum(lo, -db)) != raw
    dk = ((res.kappa - ref.kappa).abs() / ref.kappa.abs().clamp(min=1e-3))[active]
    # F_lat as the hooks see it (before the lateral deadbeat): the B2 value.
    dfl = ((seen["F_lat"] - ref_seen["F_lat"]).abs() / muN.clamp(min=1))[active]
    return dk, dfl, int(active.sum())


def test_n9_pacejka_closed_form_matches_a_k512_grid():
    """The closed form (``PacejkaAnisotropic.at_long_force``) against the
    generic grid at K = 512 on the SAME combined law, 4000 random car states
    (v 0.3..15 m/s either sign, v_lat -1..1, ω rolling ±5%, rear drive
    0..500 N*m, brake 0..750 N*m on 30% of rows), over the rows where B2 is
    active: κ relative error ≤ 1e-3, F_lat error ≤ 1e-3 μN. Measured on the
    implemented tree: κ 4.0e-4, F_lat 4.3e-5 μN (the design probe printed
    F_lat 0). Fails on 1.7.0 (no
    ``at_long_force``)."""
    dk, dfl, n_active = _b2_errors(PacejkaAnisotropic(), _PacejkaGrid(512))
    assert n_active > 1000, n_active
    assert float(dk.max()) <= 1e-3, f"kappa rel err max {float(dk.max()):.2e}"
    assert float(dfl.max()) <= 1e-3, f"F_lat err / muN max {float(dfl.max()):.2e}"


def test_n9_the_generic_k16_grid_stays_within_its_stated_bound():
    """The DEFAULT ``TireModel.at_long_force`` (K = 16) on a model that is not
    the plain Pacejka (here: a Pacejka subclass, which the guard sends to the
    grid), against K = 512: κ error ≤ 0.23 relative (p99 ≤ 0.056) and F_lat
    error ≤ 2.9% μN — the accuracy the ``TireModel.at_long_force`` docstring
    states (measured in the design probe, same states)."""
    dk, dfl, n_active = _b2_errors(_PacejkaGrid(16), _PacejkaGrid(512))
    assert n_active > 1000, n_active
    assert float(dk.max()) <= 0.23 + 1e-3, float(dk.max())
    assert float(torch.quantile(dk, 0.99)) <= 0.056 + 1e-3
    assert float(dfl.max()) <= 0.029 + 1e-4, float(dfl.max())


def test_at_long_force_default_calls_the_tyre_twice_more_and_pacejka_once():
    """C3 contract: the generic default calls ``__call__`` with K*B rows once
    and B rows once (3 calls per pipeline step incl. the main one); the
    closed form makes none."""
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    resolved, wm = _setup(cfg)
    rec = _RecordingTire()
    resolved.tire = rec
    _one_step(resolved, wm, omega=20.0, v_long=5.0, T_drive=100.0, B=3)
    assert [c.shape[0] for c in rec.calls] == [3, 16 * 3, 3]

    calls = []

    class Counting(PacejkaAnisotropic):
        pass                                   # __call__ NOT overridden

    t = Counting()
    orig = PacejkaAnisotropic.__call__

    def spy(self, *a):
        calls.append(a[0].shape[0])
        return orig(self, *a)

    PacejkaAnisotropic.__call__ = spy
    try:
        resolved.tire = t
        _one_step(resolved, wm, omega=20.0, v_long=5.0, T_drive=100.0, B=3)
    finally:
        PacejkaAnisotropic.__call__ = orig
    # spy replaced PacejkaAnisotropic.__call__ itself, so the guard still sees
    # the base __call__ and the closed form runs: only the main call.
    assert calls == [3]


def test_a_pacejka_subclass_that_overrides_call_gets_the_grid():
    class Scaled(PacejkaAnisotropic):
        def __call__(self, *a):
            Fl, Ft, k, al = PacejkaAnisotropic.__call__(self, *a)
            return 0.5 * Fl, 0.5 * Ft, k, al

    t = Scaled()
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    _, wm = _setup(cfg)
    n = wm.n_wheels
    vl = torch.full((1, n), 5.0)
    out = t.at_long_force(vl, torch.full((1, n), 0.3), torch.full((1, n), 6.0),
                          torch.full((1, n), 3000.0), wm,
                          F_long_target=torch.full((1, n), 200.0))
    grid = TireModel.at_long_force(
        t, vl, torch.full((1, n), 0.3), torch.full((1, n), 6.0),
        torch.full((1, n), 3000.0), wm, F_long_target=torch.full((1, n), 200.0))
    assert torch.equal(out[0], grid[0]) and torch.equal(out[1], grid[1])


def test_a_plain_callable_tyre_gets_the_generic_grid():
    """C17: ``cfg.tire`` was never type-checked, so a plain callable with the
    tyre signature (no ``TireModel`` base, no ``at_long_force``, no
    ``AT_LONG_FORCE_GRID``) is a supported tyre. The pipeline calls
    ``TireModel.at_long_force`` unbound on it: the step runs (no
    AttributeError on the first step, which the first 1.8.0 implementation
    raised) and gives bit for bit what a ``TireModel`` subclass wrapping the
    same function gives (K = 16 grid, 3 calls per step)."""
    pac = PacejkaAnisotropic()
    calls = []

    def plain(v_long, v_lat, v_roll, N, wm):
        calls.append(v_long.shape[0])
        return PacejkaAnisotropic.__call__(pac, v_long, v_lat, v_roll, N, wm)

    class Wrapped(TireModel):
        def __call__(self, v_long, v_lat, v_roll, N, wm):
            return PacejkaAnisotropic.__call__(pac, v_long, v_lat, v_roll, N, wm)

    cfg = car_4w_rwd_ackermann(CAR_URDF)
    resolved, wm = _setup(cfg)
    kw = dict(omega=20.0, v_long=5.0, v_lat=0.3, T_drive=100.0, B=3)
    resolved.tire = plain
    a, _ = _one_step(resolved, wm, **kw)
    assert calls == [3, 16 * 3, 3], calls
    resolved.tire = Wrapped()
    b, _ = _one_step(resolved, wm, **kw)
    for f in ("F_long", "F_lat", "new_omega", "kappa", "total_F", "total_T"):
        assert torch.equal(getattr(a, f), getattr(b, f)), f
    assert bool(torch.isfinite(a.F_lat).all())


# --- N10: the brake never reverses omega --------------------------------------

def test_n10_brake_drive_and_rr_never_reverse_omega():
    """Deterministic grid over ω, drive torque in the direction of spin (or
    zero), brake torque, rolling-resistance cr and chassis speed (rolling, or a
    stopped chassis): with no torque opposing the spin other than brake and
    rolling resistance, and the chassis not moving against it, the wheel never
    ends the step spinning the other way. Guards: the cap on the resisting
    torque, the brake projection, the leftover brake capacity applied to the
    update the same way, and the clamp (friction cannot carry ω past rolling).
    Passes on the 1.7.0 tree too, BY DESIGN: there the guards were the
    brake + resistance cap of the clamp; the property is the same."""
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    resolved, wm = _setup(cfg)
    r = float(wm.radius.mean())
    worst = 0.0
    for w0 in (-8.0, -1.0, -0.01, 0.01, 1.0, 8.0):
        for cr in (0.0, 0.05, 2.0):
            wm.rolling_resistance_cr = torch.full_like(wm.rolling_resistance_cr, cr)
            for tb in (0.0, 30.0, 500.0, 5000.0):
                for td in (0.0, 50.0, 800.0):
                    for v in (w0 * r, 0.0):
                        res, _ = _one_step(resolved, wm, omega=w0, v_long=v,
                                           T_drive=math.copysign(td, w0),
                                           T_brake=tb, hooks=[RollingResistance()])
                        worst = min(worst, float((res.new_omega * w0).min()))
    assert worst >= -1e-6, f"omega reversed: min(new_omega * omega) = {worst}"


# --- N22 / brake projection unit contract -------------------------------------

def test_n22_a_zero_brake_never_holds():
    """``brake_projection`` at ``t_brake = 0`` and ``omega_d = 0`` does NOT
    hold the wheel: ``b_rem = 0`` and ``d omega_t / d omega_d = 1``.

    What the ``t_brake > 0`` term changes is ONLY the gradient. The forward
    values are the same with or without it (at ``t_brake = 0``,
    ``omega_d = 0`` both branches give ``omega_t = 0``, ``b_rem = 0``), so
    the forward assertions below pass on a tree without the term; the
    gradient assertion is the one that fails there (the held branch gives
    ``d omega_t / d omega_d = 0``, which REDUCES -- does not remove -- a
    plant's throttle authority at a brake-free standstill: without the term
    ``d a / d throttle`` at throttle 0 is 0.881 (car) / 2.825 (tank) against
    1.141 / 3.758 at throttle 0.01, in the state of
    ``test_n22_plant_throttle_gain_is_continuous_at_zero_throttle``, which
    fails on that tree). The first version of this docstring said the
    clamp "would pin a free wheel at standstill" without the term; that was
    wrong, the forward step is unchanged. The plant-level consequence is
    ``test_n22_plant_throttle_gain_is_continuous_at_zero_throttle``."""
    from genesis_vehicle.dynamics import brake_projection
    om = torch.tensor([[0.0, 0.0, 1e-3, -2.0]], requires_grad=True)
    tb = torch.tensor([[0.0, 5.0, 0.0, 1e4]])
    i_w = torch.tensor([2.0])
    om_t, b_rem = brake_projection(om, tb, DT, i_w)
    (g,) = torch.autograd.grad(om_t.sum(), om)
    assert g.tolist() == [[1.0, 0.0, 1.0, 0.0]], g.tolist()
    om_t, b_rem = om_t.detach(), b_rem.detach()
    assert om_t.tolist() == [[0.0, 0.0, pytest.approx(1e-3), 0.0]]
    assert float(b_rem[0, 0]) == 0.0              # zero brake: not held
    assert float(b_rem[0, 1]) == 5.0              # held at rest, all of it left
    assert float(b_rem[0, 2]) == 0.0
    assert float(b_rem[0, 3]) == pytest.approx(1e4 - 2.0 * 2.0 / DT)


def test_brake_projection_never_crosses_zero():
    from genesis_vehicle.dynamics import brake_projection
    g = torch.Generator().manual_seed(1)
    om = (torch.rand(20000, generator=g) - 0.5) * 60.0
    tb = torch.rand(20000, generator=g) * 3000.0
    i_w = torch.rand(20000, generator=g) * 20.0 + 0.1
    om_t, b_rem = brake_projection(om, tb, DT, i_w)
    assert bool((om_t * om >= 0).all())
    assert bool((om_t.abs() <= om.abs() + 1e-6).all())
    assert bool((b_rem >= 0).all())
    held = b_rem > 0
    assert bool((om_t[held] == 0).all())


@pytest.mark.parametrize("kind", ["car", "tank"])
def test_n22_plant_throttle_gain_is_continuous_at_zero_throttle(kind):
    """With the wheels at ω = 0 and the chassis at 2 m/s, d a / d throttle at
    throttle 0 is within 5% of that at throttle 0.01: a zero brake never holds,
    so zero throttle is not a dead zone. The values this test compares, on
    the 1.8.0 tree (CPU): car 1.1711 / 1.1408, tank 3.7584 / 3.7584; with the
    ``t_brake > 0`` term of ``dynamics.brake_projection`` removed, the value
    at throttle 0 drops to 0.881 (car) / 2.825 (tank) and this test fails.
    1.7.0 had no brake projection, so there was no such dead zone to guard;
    this pins that the ``t_brake > 0`` guard keeps the property."""
    preset, urdf = _kind(kind)
    p = _plant(preset(urdf), v_long=2.0, omega=0.0)
    j0 = float(p.jacobian(0.0, 0.0)[0][0, 0])
    j1 = float(p.jacobian(0.01, 0.0)[0][0, 0])
    assert j0 > 0 and abs(j0 - j1) <= 0.05 * abs(j1), (j0, j1)


# --- N14 / N15: friction ellipse and slip at the transmitted force ------------

def _bisect_pure_slip(F, B_, C_, E_, D_):
    f = lambda x: D_ * math.sin(C_ * math.atan(B_ * x - E_ * (B_ * x - math.atan(B_ * x))))
    lo, hi = (0.0, 0.3) if F >= 0 else (-0.3, 0.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) < F:              # f is increasing in the slip
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@pytest.mark.parametrize("T_d", [50.0, 150.0, 467.8])
def test_n15_last_kappa_is_the_pure_curve_slip_of_the_transmitted_force(T_d):
    """Car, 5 m/s, wheels rolling, rear drive ``T_d``: the rear wheels' returned
    ``kappa`` (``last_kappa``) equals, within 0.1%, the slip at which the PURE
    longitudinal magic formula gives the transmitted ``F_long`` (the clamp
    binds: the tyre at ω_t asks for more than ``T_d / r``). This test's own
    state gives 0.00242 / 0.00729 / 0.02379. On 1.7.0 the transmitted
    force of this step is 0 and the slip reported is the raw one."""
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    cfg.stability_hooks = []
    resolved, wm = _setup(cfg)
    n = wm.n_wheels
    V = 5.0
    T = torch.zeros(1, n)
    T[0, wm.positions[:, 0] < 0] = T_d
    res, _ = _one_step(resolved, wm, omega=V / wm.radius.reshape(1, n), v_long=V,
                       T_drive=T)
    for i in torch.nonzero(wm.positions[:, 0] < 0).flatten().tolist():
        Fl = float(res.F_long[0, i])
        assert Fl == pytest.approx(T_d / float(wm.radius[i]), rel=1e-3)
        want = _bisect_pure_slip(Fl, float(wm.pb_x[i]), float(wm.pc_x[i]),
                                 float(wm.pe_x[i]), float(wm.mu_long[i] * res.N[0, i]))
        assert float(res.kappa[0, i]) == pytest.approx(want, rel=1e-3), (i, want)


def test_n14_lateral_force_is_on_the_ellipse_of_the_transmitted_force():
    """Car, 5 m/s, slip angle from v_lat 0.3 m/s, wheels 20% over rolling, no
    drive or brake: the tyre's raw force is near saturation but the overshoot
    clamp transmits only what stops the wheel at rolling. The lateral force the
    hooks see must be the tyre's own combined-law ``F_lat`` at the slip where
    its combined ``F_long`` equals the TRANSMITTED force (here inside the
    ellipse: the pure lateral force), not the ellipse cut against the raw
    force. On the 1.7.0 tree the lateral force is cut against the raw force:
    measured -1666 N against the combined-law value -1881 N on wheel 0 (11%
    lower)."""
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    cfg.stability_hooks = []
    resolved, wm = _setup(cfg)
    rec = _Recorder()
    V, VLAT = 5.0, 0.3
    n = wm.n_wheels
    res, _ = _one_step(resolved, wm, omega=1.2 * V / wm.radius.reshape(1, n),
                       v_long=V, v_lat=VLAT, hooks=[rec])
    tire = resolved.tire
    for i in range(n):
        F_t = float(res.F_long[0, i])                 # transmitted
        F_raw = float(rec.seen["F_long"][0, i])
        assert abs(F_t) < abs(F_raw) - 1.0            # the clamp binds
        # Bisection on the COMBINED law for the rolling speed giving F_t.
        vl = torch.full((1, n), V)
        vlat = torch.full((1, n), VLAT)
        N = res.N

        def comb(vr):
            Fl, Ft, _, _ = tire(vl, vlat, torch.full((1, n), vr), N, wm)
            return float(Fl[0, i]), float(Ft[0, i])

        lo, hi = V, 1.2 * V
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if comb(mid)[0] < F_t:
                lo = mid
            else:
                hi = mid
        want = comb(0.5 * (lo + hi))[1]
        mu_n = float(wm.mu_long[i] * N[0, i])
        assert float(rec.seen["F_lat"][0, i]) == pytest.approx(want, abs=1e-3 * mu_n)
        sq = (F_t / mu_n) ** 2 + (float(rec.seen["F_lat"][0, i]) / mu_n) ** 2
        assert sq <= 1.0 + 1e-5


# --- N16: ceiling, F-vs-WH check ----------------------------------------------

def test_n16_the_tyre_never_sees_a_wheel_speed_above_omega_max():
    """At the wheel-speed ceiling (ω = ω_max, full drive torque on the driven
    rear axle) the tyre is evaluated at ``r * ω_max`` at dt 0.025 AND 0.0125,
    so the DRIVEN wheels' transmitted force does not depend on dt (M2).
    Without the clip the tyre would see ``ω_max + T*dt/i`` — dt-dependent.
    (The undriven front wheels, also at ω_max here, are limited by the
    overshoot clamp, whose budget is dt-dependent by construction; they are
    not what this checks.) An F-vs-M2 check, NOT a 1.7.0 discriminator: 1.7.0
    evaluated the tyre at the pre-step ω (≤ ω_max) and passes it too."""
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    out = {}
    for dt in (0.025, 0.0125):
        resolved, wm = _setup(cfg)
        tire = _RecordingTire()
        resolved.tire = tire
        om_max = float(resolved.chassis.omega_max)
        n = wm.n_wheels
        T = torch.zeros(1, n)
        T[0, wm.positions[:, 0] < 0] = 1500.0
        res, _ = _one_step(resolved, wm, omega=om_max, v_long=34.0, T_drive=T, dt=dt)
        assert torch.allclose(tire.calls[0], wm.radius.unsqueeze(0) * om_max)
        out[dt] = res.F_long[:, wm.positions[:, 0] < 0]
    assert torch.allclose(out[0.025], out[0.0125], rtol=1e-5, atol=1e-3), out


# --- N27: the ctx contract -----------------------------------------------------

class _Vandal(StabilityHook):
    """Writes every 1.8.0 read-only ctx field, in place and by rebinding."""
    slots = ("POST_TIRE",)

    def apply_pre_tire(self, ctx):
        if torch.is_tensor(ctx.deadbeat_scale):
            ctx.deadbeat_scale.fill_(1e-6)

    def apply_post_tire(self, ctx):
        ctx.F_long_bounds[0].fill_(-1e9)
        ctx.F_long_bounds[1].fill_(1e9)
        ctx.F_long_deadbeat.fill_(1e9)
        if torch.is_tensor(ctx.deadbeat_scale):
            ctx.deadbeat_scale.fill_(1e-6)
        ctx.F_long_bounds = (ctx.F_long * 0 - 5e9, ctx.F_long * 0 + 5e9)
        ctx.F_long_deadbeat = ctx.F_long * 0 + 5e9
        ctx.deadbeat_scale = 1e-6


def test_n27_ctx_fields_are_populated_for_post_tire_hooks():
    resolved, wm = _setup(car_4w_rwd_ackermann(CAR_URDF))
    rec = _Recorder()
    _one_step(resolved, wm, omega=0.0, v_long=0.3, T_brake=700.0, hooks=[rec])
    lo, hi = rec.seen["F_long_bounds"]
    assert bool((lo <= 0).all()) and bool((hi >= 0).all())
    db = rec.seen["F_long_deadbeat"]
    assert bool(torch.isfinite(db).all())             # every wheel held here
    assert rec.seen["hold_applied"] is None           # no holding hook
    assert torch.is_tensor(rec.seen["deadbeat_scale"]) is False   # no Z inputs: 1.0
    rec2 = _Recorder()
    _one_step(resolved, wm, omega=20.0, v_long=5.0, hooks=[rec2])
    assert bool(torch.isinf(rec2.seen["F_long_deadbeat"]).all())  # nothing held


def test_n27_a_hook_writing_the_bounds_does_not_move_the_clamp():
    """The read-only fields (``F_long_bounds``, ``F_long_deadbeat``,
    ``deadbeat_scale``) written in place AND rebound by a hook, in both
    phases: every output is bit-identical to the step without that hook (with
    and without yaw-inertia inputs, i.e. a tensor and a float scale)."""
    from genesis_vehicle.dynamics import yaw_inertia_from_mass_props
    resolved, wm = _setup(car_4w_rwd_ackermann(CAR_URDF))
    yi = yaw_inertia_from_mass_props(1330.0, (0.01, 0.0, 0.0), 2714.5, wm.positions)
    for kw in (dict(omega=0.0, v_long=0.3, T_brake=700.0),
               dict(omega=1.2 * 5.0 / 0.358, v_long=5.0, v_lat=0.2),
               dict(omega=0.0, v_long=0.3, v_lat=0.05, T_brake=700.0, yaw_inertia=yi)):
        a, _ = _one_step(resolved, wm, **kw)
        b, _ = _one_step(resolved, wm, hooks=[_Vandal()], **kw)
        for f in ("F_long", "F_lat", "new_omega", "total_F", "total_T"):
            assert torch.equal(getattr(a, f), getattr(b, f)), (kw, f)


def test_n27_hold_applied_exempts_wheels_from_both_deadbeats():
    """A hook that ORs a wheel into ``ctx.hold_applied`` gets its force
    clamped to the bounds WITHOUT the held-wheel deadbeat, and no lateral
    deadbeat."""
    class Holder(StabilityHook):
        slots = ("POST_TIRE",)

        def apply_post_tire(self, ctx):
            ctx.F_long = torch.full_like(ctx.F_long, -1e5)
            ctx.F_lat = torch.full_like(ctx.F_lat, 4.0e3)
            m = torch.zeros_like(ctx.F_long, dtype=torch.bool)
            m[:, 0] = True
            ctx.hold_applied = m if ctx.hold_applied is None else ctx.hold_applied | m

    resolved, wm = _setup(car_4w_rwd_ackermann(CAR_URDF))
    rec = _Recorder()
    res, _ = _one_step(resolved, wm, omega=0.0, v_long=0.05, v_lat=0.01,
                       T_brake=700.0, hooks=[Holder(), rec])
    lo, _ = rec.seen["F_long_bounds"]
    db = rec.seen["F_long_deadbeat"]
    assert float(res.F_long[0, 0]) == pytest.approx(float(lo[0, 0]))       # bounds only
    assert float(res.F_long[0, 1]) == pytest.approx(float(max(lo[0, 1], -db[0, 1])))
    assert float(res.F_lat[0, 0]) == pytest.approx(4.0e3)                    # no Y
    assert float(res.F_lat[0, 1]) < 4.0e3                                    # Y binds


# --- Z: deadbeat_scale and its inputs -------------------------------------------

def test_deadbeat_scale_formula_and_the_inf_control():
    from genesis_vehicle.dynamics import (DEADBEAT_G, YawInertia, deadbeat_scale,
                                          yaw_inertia_from_mass_props)
    pos = torch.tensor([[1.35, 0.75, 0.3], [1.35, -0.75, 0.3],
                        [-1.35, 0.75, 0.3], [-1.35, -0.75, 0.3]])
    yi = yaw_inertia_from_mass_props(1330.0, (0.010, 0.0, 0.5),
                                     2714.4 + 1330.0 * 0.010 ** 2, pos)
    assert yi.i_cg == pytest.approx(2714.4)
    rho = (pos[:, 0] - 0.010) ** 2 + pos[:, 1] ** 2
    assert torch.allclose(yi.rho_sq, rho)
    N = torch.full((2, 4), 1330.0 * DEADBEAT_G / 4)
    s = deadbeat_scale(N, yi)
    assert s.shape == (2, 1)
    want = min(1.0, 2714.4 / float((1330.0 / 4 * rho).sum()))
    assert float(s[0, 0]) == pytest.approx(want, rel=1e-5)
    assert want == pytest.approx(0.856, abs=0.01)          # the reference car
    off = deadbeat_scale(N, YawInertia(float("inf"), yi.rho_sq))
    assert torch.equal(off, torch.ones(2, 1))
    air = deadbeat_scale(torch.zeros(1, 4), yi)            # all wheels airborne
    assert torch.equal(air, torch.ones(1, 1))


@pytest.mark.parametrize("izz", [100.0, float("nan"), 0.0])
def test_yaw_inertia_inputs_refuse_a_non_physical_rigid_body(izz):
    from genesis_vehicle.dynamics import yaw_inertia_from_mass_props
    # C18: the message also says izz is about the base-link ORIGIN and how to
    # convert a centre-of-mass value (the likely user mistake).
    with pytest.raises(ValueError, match=r"not a physical rigid body\. `izz` is the yaw "
                       r"inertia about the base-link ORIGIN.*izz_cg \+ mass\*\(cx\^2 \+ cy\^2\)"):
        yaw_inertia_from_mass_props(1330.0, (1.0, 0.0, 0.0), izz, torch.zeros(4, 3))


def test_the_keyword_takes_precedence_over_the_wheel_meta():
    """PC6: ``compute_wheel_step(yaw_inertia=...)`` wins over
    ``wheel_meta.yaw_inertia`` (how a plant runs with its own inputs)."""
    from genesis_vehicle.dynamics import YawInertia
    resolved, wm = _setup(car_4w_rwd_ackermann(CAR_URDF))
    wm.yaw_inertia = YawInertia(1.0, torch.full((wm.n_wheels,), 2.0))   # s tiny
    kw = YawInertia(float("inf"), torch.full((wm.n_wheels,), 2.0))      # s = 1
    rec = _Recorder()
    _one_step(resolved, wm, omega=0.0, v_long=0.3, T_brake=700.0, hooks=[rec],
              yaw_inertia=kw)
    assert torch.equal(rec.seen["deadbeat_scale"], torch.ones(1, 1))
    rec2 = _Recorder()
    _one_step(resolved, wm, omega=0.0, v_long=0.3, T_brake=700.0, hooks=[rec2])
    assert float(rec2.seen["deadbeat_scale"][0, 0]) < 0.01


def test_no_yaw_inertia_means_s_1_and_one_log_line(monkeypatch):
    import logging
    import genesis_vehicle._pipeline as P
    monkeypatch.setattr(P, "_NO_YAW_INERTIA_WARNED", False)
    lines = []

    class H(logging.Handler):
        def emit(self, rec):
            lines.append(rec.getMessage())

    h = H()
    logging.getLogger("genesis_vehicle").addHandler(h)
    try:
        resolved, wm = _setup(car_4w_rwd_ackermann(CAR_URDF))
        rec = _Recorder()
        for _ in range(3):
            _one_step(resolved, wm, omega=0.0, v_long=0.3, T_brake=700.0, hooks=[rec])
    finally:
        logging.getLogger("genesis_vehicle").removeHandler(h)
    assert rec.seen["deadbeat_scale"] == 1.0
    hits = [l for l in lines if "[genesis_vehicle:deadbeat-scale]" in l]
    assert len(hits) == 1, lines


# --- N5: the plant is monotone in throttle --------------------------------------

def test_n5_reverse_tank_plant_is_monotone_in_throttle():
    """AC-P2: tank backing at -2 m/s (fake source, explicit masses — Z
    active): predicted acceleration is monotone in throttle over -1.0 / -0.8
    / -0.5. On 1.7.0 it was not (design probe: -2.3787 / -2.7154 / -1.6965,
    the -1.0 point less negative than -0.8): its drive torque went through the
    overshoot clamp, which capped it at the clamp limit."""
    p = _plant(tank_skid_belt(TANK_URDF), v_long=-2.0)
    a = [p.predict(t, 0.0)[0] for t in (-1.0, -0.8, -0.5)]
    assert a[0] < a[1] < a[2], a


# --- FX6: the closed form with pe_x >= 1 ----------------------------------------

@pytest.mark.parametrize("pe_x", [1.0, 1.2, 1.5, 2.0])
def test_closed_form_kappa_stays_finite_for_pe_x_at_or_above_one(pe_x):
    """``pe_x >= 1`` is accepted by the config and the OSC server (only
    finiteness is checked). The closed form's Newton divisor ``(1-E) +
    E/(1+x^2)`` then reaches zero (E > 1: at ``x^2 = 1/(E-1)``).

    Where it is EXACTLY zero the unfloored step is inf and the next one NaN:
    the constructed state below (``pc_x`` 1.6 -- the tank preset's value --,
    ``pe_x`` 2.0, ``mu*N`` = 1, target 0.95105654 makes the first iterate
    exactly 1.0 in float32) gives NaN ``kappa`` and a NaN gradient without the
    floor -- in ``DifferentiablePlant`` such a NaN sits on the untaken
    ``torch.where`` branch and turns every wheel's gradient NaN. With the
    floor (exact 0 -> +1e-3, sign of a non-zero divisor kept): finite
    (accuracy at ``pe_x >= 1`` is NOT characterised and not asserted). The
    exact zero is what the floor removes; non-zero divisors were never
    observed to produce a non-finite value (dense sweep, see
    ``PacejkaAnisotropic.at_long_force``). The sweep part below is a
    smoke check that the floor does not introduce non-finite values; the
    exact-zero state is the part that fails on the unfloored code. A full
    pipeline step with that tyre also has a finite gradient of the chassis
    force with respect to ω."""
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    _, wm0 = _setup(cfg)
    n = wm0.n_wheels
    wm0.pe_x = torch.full_like(wm0.pe_x, 2.0)
    wm0.pc_x = torch.full_like(wm0.pc_x, 1.6)
    wm0.mu_long = torch.full_like(wm0.mu_long, 1.0)
    tgt0 = torch.full((1, n), 0.9510565400123596, requires_grad=True)
    F_lat0, kappa0 = PacejkaAnisotropic().at_long_force(
        torch.full((1, n), 5.0), torch.zeros(1, n), torch.full((1, n), 9.0),
        torch.ones(1, n), wm0, F_long_target=tgt0)
    (g0,) = torch.autograd.grad(kappa0.sum(), tgt0)
    assert bool(torch.isfinite(kappa0).all()) and bool(torch.isfinite(g0).all()), (
        kappa0, g0)

    cfg = car_4w_rwd_ackermann(CAR_URDF)
    resolved, wm = _setup(cfg)
    wm.pe_x = torch.full_like(wm.pe_x, pe_x)
    n = wm.n_wheels
    t = PacejkaAnisotropic()
    Nn = torch.full((64, n), 3000.0, requires_grad=True)
    frac = torch.linspace(0.0, 0.999, 64).unsqueeze(1).expand(64, n)
    target = (frac * 3000.0 * wm.mu_long.unsqueeze(0)).clone().requires_grad_(True)
    v_lat = torch.full((64, n), 0.05, requires_grad=True)
    F_lat, kappa = t.at_long_force(torch.full((64, n), 5.0), v_lat,
                                   torch.full((64, n), 9.0), Nn, wm,
                                   F_long_target=target)
    assert bool(torch.isfinite(kappa).all()) and bool(torch.isfinite(F_lat).all())
    grads = torch.autograd.grad((kappa.sum() + F_lat.sum()), (Nn, target, v_lat))
    assert all(bool(torch.isfinite(g).all()) for g in grads)

    om = torch.full((1, n), 30.0, requires_grad=True)
    res, _ = _one_step(resolved, wm, omega=om, v_long=5.0, v_lat=0.3, T_drive=400.0)
    (g,) = torch.autograd.grad(res.total_F.sum() + res.kappa.sum(), om)
    assert bool(torch.isfinite(g).all()), g


def _newton_kappa_reference(y, C, B, E, *, floor):
    """The closed form's 3 Newton steps, written out. Returns (kappa, divisors)."""
    t = math.tan(math.asin(y) / C)
    x, dens = t, []
    for _ in range(3):
        g = x * (1 - E) + E * math.atan(x) - t
        den = (1 - E) + E / (1 + x * x)
        dens.append(den)
        x = x - g / floor(den)
    return x / B, dens


def test_closed_form_keeps_the_newton_direction_where_the_divisor_is_negative():
    """F2-1 (1.8.0): for ``pe_x`` > 1 the Newton divisor ``(1-E) + E/(1+x^2)``
    is truly negative for ``x^2 > 1/(E-1)``. The floor keeps the divisor's sign
    and floors only its MAGNITUDE at 1e-3, so a divisor of ordinary negative
    size is used as is (the true Newton step). The earlier
    ``clamp(min=1e-3)`` replaced it by +1e-3, reversing the step and
    multiplying it by up to 1000/|g'|; this test fails on that code.

    State: ``pe_x`` 2.0, ``pc_x`` 1.6, ``pb_x`` 10 (reference car), ``mu*N``
    = 1, pure longitudinal (``v_lat`` 0), target 0.999. The first iterate
    sits at divisor -0.33. The result is only asserted finite and equal to
    the hand-written unfloored Newton steps: there is no root here (the target
    lies above the curve's reachable maximum, see the ``at_long_force``
    docstring), so the slip's value is not meaningful and is not asserted."""
    cfg = car_4w_rwd_ackermann(CAR_URDF)
    _, wm = _setup(cfg)
    n = wm.n_wheels
    E, C = 2.0, 1.6
    wm.pe_x = torch.full_like(wm.pe_x, E)
    wm.pc_x = torch.full_like(wm.pc_x, C)
    wm.mu_long = torch.full_like(wm.mu_long, 1.0)
    B = float(wm.pb_x[0])
    tgt = torch.full((1, n), 0.999, requires_grad=True)
    F_lat, kappa = PacejkaAnisotropic().at_long_force(
        torch.full((1, n), 5.0), torch.zeros(1, n), torch.full((1, n), 9.0),
        torch.ones(1, n), wm, F_long_target=tgt)
    (g,) = torch.autograd.grad(kappa.sum(), tgt)
    assert bool(torch.isfinite(kappa).all()) and bool(torch.isfinite(g).all()), (kappa, g)

    y = float(torch.tensor(0.999, dtype=FDT))
    ref, dens = _newton_kappa_reference(y, C, B, E, floor=lambda d: d)
    assert dens[0] < -0.1, dens                          # the negative branch is exercised
    assert all(abs(d) >= 1e-3 for d in dens), dens       # the floor does not engage
    assert float(kappa.detach()[0, 0]) == pytest.approx(ref, rel=1e-4, abs=1e-6), (kappa, ref)
    old, _ = _newton_kappa_reference(y, C, B, E, floor=lambda d: max(d, 1e-3))
    assert abs(old - ref) > 1.0, (old, ref)              # the wrong-signed clamp differs


def test_preset_tyres_never_reach_the_newton_divisor_floor():
    """F2-1: the divisor floor must not engage for any shipped preset, so the
    preset rollouts stay bit-identical. For x >= 0 the divisor is >= min(1,
    1-E) when E < 1 (and g' is even in x), so a preset is safe whenever every
    wheel has ``pe_x`` well below 0.999 (asserted < 0.99). Checked on the resolved wheel tables of the
    car presets (``tests/data/car_ref.urdf``) and the tank preset
    (``tests/data/tank_ref.urdf``), plus the config default
    ``DEFAULT_PACEJKA`` the truck preset falls back to (it sets no
    per-wheel tyre); then the minimum |divisor| over the three iterates is
    computed on a dense target grid for each preset's (pc_x, pe_x)."""
    from genesis_vehicle.config import DEFAULT_PACEJKA
    from genesis_vehicle.presets import car_4w_awd_ackermann, car_4w_fwd_ackermann
    params = set()
    for fn, urdf in ((car_4w_rwd_ackermann, CAR_URDF), (car_4w_fwd_ackermann, CAR_URDF),
                     (car_4w_awd_ackermann, CAR_URDF), (tank_skid_belt, TANK_URDF)):
        _, wm = _setup(fn(urdf))
        params |= {(float(c), float(e)) for c, e in zip(wm.pc_x, wm.pe_x)}
    params.add((DEFAULT_PACEJKA["pc_x"], DEFAULT_PACEJKA["pe_x"]))
    y = torch.linspace(0.0, 1.0 - 1e-6, 20001)
    for C, E in params:
        assert E < 0.99, (C, E)
        t = torch.tan(torch.asin(y) / C)
        x, m = t, float("inf")
        for _ in range(3):
            den = (1 - E) + E / (1 + x * x)
            m = min(m, float(den.abs().min()))
            x = x - (x * (1 - E) + E * torch.atan(x) - t) / den
        # g' is even in x, so the bound min(1, 1-E) holds for negative iterates too.
        assert m >= min(1.0, 1.0 - E) - 1e-6 and m > 1e-3, (C, E, m)


def test_mass_read_on_an_entity_without_links_names_the_requirement():
    """FX3: since 1.8.0 the first step reads the vehicle's mass properties from
    ``entity.links`` (the yaw-inertia inputs). Through 1.7.0 the step path
    never needed ``links``, so a duck-typed entity without it must get an
    error that names the new requirement, not a bare ``'X' object has no
    attribute 'links'``."""
    from types import SimpleNamespace
    from genesis_vehicle.core import VehiclePhysics
    me = SimpleNamespace(wheel_meta=SimpleNamespace(yaw_inertia=None),
                         _mass_props=None, entity=object(), vehicle_name="stub")
    with pytest.raises(AttributeError, match=r"vehicle 'stub'.*no 'links'.*Since 1\.8\.0"):
        VehiclePhysics._ensure_yaw_inertia(me)
