"""2.0.6: the StaticFrictionLock ground gate — the lock is inactive on a wheel with ``ctx.N <= 0``.

| abbr | meaning |
|---|---|
| SFL | `StaticFrictionLock` (`strategies/stability.py`) |
| N | per-wheel suspension normal force (`ctx.N`; `VehiclePhysics.last_N`) |
| air mask | the pipeline's `compression <= 0` (`ctx.air_mask`); N is 0 there, and also on a wheel in contact whose spring + damper force is not positive |
| U1-U4 | unit checks on a fake ctx; G1 the Genesis landing check |

What is enforced, and where: in ``StaticFrictionLock.apply_post_tire``,
``active = (brake_rem > 0) & active_v & (ctx.N > 0)``, per wheel, no threshold,
no hysteresis. On 2.0.5 the creep test's bind ``|F_long| >= F_long_deadbeat``
held vacuously at N = 0 (the deadbeat is 0 there), so a braked wheel engaged
and was taken over in the air.

U1  wheel 0 airborne (N 0, air mask True), brake_rem > 0, v 0; wheel 1 loaded:
    wheel 0 is not active, never latches ``_rest`` (the bind holds), gets no
    hold and its ``F_long`` is untouched.
U2  a taken-over wheel goes through one step at N 0 and lands: the landing
    step re-engages (``_rest`` cleared) and applies no hold unless the creep
    test passes; the next step whose creep test passes takes over again.
U3  N 0 with the air mask False (rebound, in contact) is gated too.
U4  the other wheels' state and forces equal a run where wheel 0 is loaded.
G1  Genesis, the reference car dropped onto a plane (heights 0.6 / 1.0 /
    2.0 m, flat / 5° gravity tilt, brake 1.0 / 0.3, plus two rows with
    vx 0.3 m/s), 120 steps: no engagement and no takeover edge on a wheel
    with N <= 0. (On 2.0.5: dozens of each.)
"""

from __future__ import annotations

import math
import os
import types

import numpy as np
import pytest
import torch

from genesis_vehicle.strategies.stability import StaticFrictionLock


class _Meta:
    def __init__(self, n):
        self.n_wheels = n
        self.mu_long = torch.ones(n)
        self.mu_lat = torch.ones(n)
        self.static_mass_share = None       # uncapped: k_eff = k_spring


def _ctx(meta, N, v=0.0, air=None, F_long=5.0, dt=0.025):
    """A held wheel on every wheel (brake_rem > 0) whose deadbeat binds."""
    N = torch.tensor([N], dtype=torch.float32)
    n = N.shape[1]
    v = torch.full((1, n), float(v)) if not torch.is_tensor(v) else v
    if air is None:
        air = N <= 0
    return types.SimpleNamespace(
        brake=torch.tensor([1.0]), v_long=v.clone(), v_lat=torch.zeros(1, n), N=N, dt=dt,
        wheel_meta=meta, F_long=torch.full((1, n), float(F_long)), F_lat=torch.zeros(1, n),
        omega_override=None, brake_rem=torch.full((1, n), 100.0),
        F_long_deadbeat=torch.zeros(1, n), air_mask=torch.as_tensor(air, dtype=torch.bool).reshape(1, n))


def test_u1_an_airborne_braked_wheel_is_neither_engaged_nor_taken_over():
    meta = _Meta(2)
    h = StaticFrictionLock()
    for _ in range(3):                      # step 1 primes v_prev; steps 2-3: creep test passes
        c = _ctx(meta, [0.0, 3000.0], v=0.0, air=[True, False])
        h.apply_post_tire(c)
    assert h._was_active.tolist() == [[False, True]]
    assert h._rest.tolist() == [[False, True]], "wheel 0 latched _rest in the air"
    assert c.hold_applied.tolist() == [[False, True]]
    assert float(c.F_long[0, 0]) == 5.0, "the lock replaced an airborne wheel's force"


def test_u2_landing_re_engages_and_takes_over_only_through_the_creep_test():
    meta = _Meta(1)
    h = StaticFrictionLock()
    for _ in range(2):                      # engage, then creep takeover
        h.apply_post_tire(_ctx(meta, [3000.0], v=0.0))
    assert h._rest.tolist() == [[True]]
    c = _ctx(meta, [0.0], v=0.01)           # one step unloaded (gated)
    h.apply_post_tire(c)
    assert h._was_active.tolist() == [[False]]
    assert not bool(c.hold_applied.any()) and float(c.F_long[0, 0]) == 5.0
    c = _ctx(meta, [3000.0], v=0.005)       # landing step: speed fell -> creep test fails
    h.apply_post_tire(c)
    assert h._was_active.tolist() == [[True]]
    assert h._rest.tolist() == [[False]], "_rest not cleared on re-engagement"
    assert not bool(c.hold_applied.any()) and float(c.F_long[0, 0]) == 5.0, "hold applied on landing"
    c = _ctx(meta, [3000.0], v=0.005)       # speed no longer falls -> creep takeover
    h.apply_post_tire(c)
    assert h._rest.tolist() == [[True]] and bool(c.hold_applied.all())


def test_u3_zero_normal_force_in_contact_is_gated_too():
    meta = _Meta(2)
    h = StaticFrictionLock()
    for _ in range(3):
        c = _ctx(meta, [0.0, 3000.0], v=0.0, air=[False, False])   # rebound: in contact, N clamped 0
        h.apply_post_tire(c)
    assert h._was_active.tolist() == [[False, True]]
    assert h._rest.tolist() == [[False, True]]
    assert c.hold_applied.tolist() == [[False, True]]
    assert float(c.F_long[0, 0]) == 5.0


def test_u4_the_gate_is_per_wheel():
    meta = _Meta(4)
    gated, loaded = StaticFrictionLock(), StaticFrictionLock()
    vs = [0.0, 0.0, 2e-3, 2e-3, 1e-3, -1e-3, 0.0, 0.0]
    for k, v in enumerate(vs):
        vt = torch.tensor([[v, v, -v, 0.5 * v]])
        cg = _ctx(meta, [0.0 if k % 3 else 3000.0, 3000.0, 2500.0, 3500.0], v=vt)
        cl = _ctx(meta, [3000.0, 3000.0, 2500.0, 3500.0], v=vt)
        gated.apply_post_tire(cg)
        loaded.apply_post_tire(cl)
        assert torch.equal(cg.F_long[:, 1:], cl.F_long[:, 1:]), k
        assert torch.equal(cg.F_lat[:, 1:], cl.F_lat[:, 1:]), k
        assert torch.equal(cg.hold_applied[:, 1:], cl.hold_applied[:, 1:]), k
        for a in StaticFrictionLock.STATE_ATTRS:
            assert torch.equal(getattr(gated, a)[:, 1:], getattr(loaded, a)[:, 1:]), (k, a)
    assert bool(loaded._rest[:, 1:].any()), "the other wheels never took over: U4 does not exercise the hold"


# --- G1: Genesis landing matrix -------------------------------------------------

def test_g1_no_engagement_or_takeover_on_a_wheel_without_normal_force(monkeypatch):
    import genesis as gs
    import genesis_vehicle
    from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann

    counts = dict(engage=0, takeover=0, engage_N0=0, takeover_N0=0)
    orig = StaticFrictionLock.apply_post_tire

    def counted(self, ctx):
        r0 = None if self._rest is None else self._rest.clone()
        w0 = None if self._was_active is None else self._was_active.clone()
        n0 = (ctx.N <= 0).clone()
        orig(self, ctx)
        if r0 is None:
            r0, w0 = torch.zeros_like(self._rest), torch.zeros_like(self._was_active)
        e, t = self._was_active & ~w0, self._rest & ~r0
        counts["engage"] += int(e.sum()); counts["takeover"] += int(t.sum())
        counts["engage_N0"] += int((e & n0).sum()); counts["takeover_N0"] += int((t & n0).sum())

    monkeypatch.setattr(StaticFrictionLock, "apply_post_tire", counted)
    urdf = os.path.join(os.path.dirname(genesis_vehicle.__file__), "samples", "urdf", "car_4w.urdf")
    rows = [dict(h=h, th=th, b=b, vx=0.0) for h in (0.6, 1.0, 2.0) for th in (0, 5) for b in (1.0, 0.3)]
    rows += [dict(h=1.0, th=th, b=1.0, vx=0.3) for th in (0, 5)]
    n = len(rows)
    VehicleScene.init_backend("cpu")
    vs = VehicleScene(dt=0.025, substeps=4, n_envs=n, raycast_mode="dual_scene", show_viewer=False,
                      solver="per_vehicle")
    vs.add_ground_plane(friction=1.0)
    veh = vs.add_vehicle(urdf, car_4w_rwd_ackermann, pos=(0, 0, 0.6), stability="control",
                         material=gs.materials.Rigid(friction=1.0))
    vs.build()
    ent = veh.entity_main
    for i, r in enumerate(rows):
        t = math.radians(r["th"])
        vs.set_gravity((9.81 * math.sin(t), 0.0, -9.81 * math.cos(t)), envs_idx=[i])
    pos0 = ent.get_pos().clone()
    pos0[:, 2] = torch.tensor([r["h"] for r in rows], dtype=pos0.dtype)
    ent.set_pos(pos0)
    v = torch.zeros((n, 1))
    v[:, 0] = torch.tensor([r["vx"] for r in rows])
    ent.set_dofs_velocity(v, dofs_idx_local=[0])
    T = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=gs.device)  # noqa: E731
    veh.set_inputs(throttle=T(np.zeros(n)), brake=T([r["b"] for r in rows]), steer=T(np.zeros(n)))
    for _ in range(120):
        vs.step()
    assert counts["engage"] > 0 and counts["takeover"] > 0, f"the lock never acted: {counts}"
    assert counts["engage_N0"] == 0 and counts["takeover_N0"] == 0, counts
