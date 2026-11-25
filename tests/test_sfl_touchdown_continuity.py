"""2.0.6: the StaticFrictionLock reversal re-anchor ("rule A") is gone.

| abbr | meaning |
|---|---|
| SFL | `StaticFrictionLock` (`strategies/stability.py`) |
| rule A | 1.8.0-2.0.5: the first step after engagement with `s_eng * v_long <= 0` zeroed `_d_long` once (`s_eng` = sign of `v_long` at engagement, latched by `_rested`) |
| vx, vy, wx, wy, wz | the tank free joint's linear (x, y) and angular (x, y, z) velocity dofs (0, 1, 3, 4, 5) |

Rule A read ``sign(v_long)`` at engagement. At a touchdown ``v_long`` is
round-off, so the branch it took was round-off: on 2.0.5 the tank of
``test_legacy_inputs_bit_identity``'s ``direct_single_3`` scene ends 0.121 m
apart for an initial tank velocity of -1e-6 vs +1e-6 m/s (plan measurement,
genesis-world 1.5.0, CPU/WSL2). What these tests check:

(a) Genesis, that scene with the tank's free joint started at vx -1e-6, vx
    +1e-6 and wz -1e-6: the three final tank positions agree within 1e-3 m per
    env. This is the sign-independence of THESE perturbations in THIS scene,
    not continuity of the lock (the creep-test takeover is still a discrete
    decision). It fails on 2.0.5.
(b) Unit, fake ctx: two locks engaged at v_long = +1e-12 and -1e-12 and then
    fed the same contact history give identical F_long / F_lat on every later
    step. With rule A they differ once the contact reverses.
(c) ``_s_eng`` / ``_rested`` raise on read and write (instance and class).
"""

from __future__ import annotations

import types

import pytest
import torch

from genesis_vehicle.strategies.stability import StaticFrictionLock


# --- (a) Genesis: sign of a tiny initial velocity does not pick a branch ------

def _direct_single_3_tank_final(monkeypatch, dof, val):
    """Run ``direct_single_3`` with the tank's free-joint dof ``dof`` started at
    ``val`` (set right after ``build()``); the tank's final position per env."""
    from genesis_vehicle.tests import test_legacy_inputs_bit_identity as T
    from genesis_vehicle import VehicleScene

    orig = VehicleScene.build

    def build(self, *a, **kw):
        r = orig(self, *a, **kw)
        for veh in self._vehicles:
            if veh.name == "tank":
                v = torch.zeros((self.n_envs, 1))
                v[:, 0] = val
                veh.entity_main.set_dofs_velocity(v, dofs_idx_local=[dof])
        return r

    monkeypatch.setattr(VehicleScene, "build", build)
    sc = [s for s in T.SCENES if s[0] == "direct_single_3"][0]
    res = T.run_scene(*sc)
    pos = res["tank"]["cp"][str(T.CHECKPOINTS[-1])]["pos"]
    return torch.tensor(pos, dtype=torch.float64).reshape(-1, 3)


def test_direct_single_3_tank_final_does_not_depend_on_the_perturbation_sign(monkeypatch):
    finals = {
        "vx-1e-6": _direct_single_3_tank_final(monkeypatch, 0, -1e-6),
        "vx+1e-6": _direct_single_3_tank_final(monkeypatch, 0, +1e-6),
        "wz-1e-6": _direct_single_3_tank_final(monkeypatch, 5, -1e-6),
    }
    names = list(finals)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = finals[names[i]], finals[names[j]]
            d = (a - b).abs().max(dim=1).values      # per env, max over x, y, z
            assert bool((d <= 1e-3).all()), (
                f"{names[i]} vs {names[j]}: per-env max |dpos| {d.tolist()} m "
                f"(> 1e-3 m): a round-off branch is back")


# --- (b) unit: the engagement sign is not remembered ---------------------------

class _Meta:
    def __init__(self, n=4):
        self.n_wheels = n
        self.mu_long = torch.ones(n)
        self.mu_lat = torch.ones(n)
        self.static_mass_share = None       # uncapped: k_eff = k_spring


def _ctx(meta, v_long, v_lat, n=4, N=3000.0, dt=0.025):
    """A held wheel (brake_rem > 0) whose deadbeat binds (F_long_deadbeat 0)."""
    return types.SimpleNamespace(
        brake=torch.tensor([1.0]), v_long=torch.full((1, n), v_long),
        v_lat=torch.full((1, n), v_lat), N=torch.full((1, n), N), dt=dt,
        wheel_meta=meta, F_long=torch.zeros(1, n), F_lat=torch.zeros(1, n),
        omega_override=None, brake_rem=torch.full((1, n), 100.0),
        F_long_deadbeat=torch.zeros(1, n))


def test_engagement_at_plus_and_minus_round_off_gives_identical_forces():
    meta = _Meta()
    hooks = {s: StaticFrictionLock() for s in (+1.0, -1.0)}
    # history after engagement: takeover (creep test passes: v^2 >= 0.96 *
    # v_prev^2), roll forward, then reverse — rule A fired on the reversal for
    # the +1 hook and on the first forward step for the -1 hook.
    history = [1e-3] * 5 + [-1e-3] * 5 + [0.0] * 3
    out = {}
    for s, h in hooks.items():
        h.apply_post_tire(_ctx(meta, s * 1e-12, 5e-4))      # engagement step
        fl, ft = [], []
        for v in history:
            c = _ctx(meta, v, 5e-4)
            h.apply_post_tire(c)
            fl.append(c.F_long.clone())
            ft.append(c.F_lat.clone())
        out[s] = (torch.stack(fl), torch.stack(ft))
        assert bool(h._rest.all()), "the lock never took over: the test does not exercise the hold"
    assert torch.equal(out[+1.0][0], out[-1.0][0]), (out[+1.0][0][:, 0, 0], out[-1.0][0][:, 0, 0])
    assert torch.equal(out[+1.0][1], out[-1.0][1])


# --- (c) the removed state names raise ----------------------------------------

@pytest.mark.parametrize("name", ["_s_eng", "_rested"])
def test_removed_rule_a_state_raises(name):
    h = StaticFrictionLock()
    with pytest.raises(AttributeError, match="removed in 2.0.6"):
        getattr(h, name)
    with pytest.raises(AttributeError, match="removed in 2.0.6"):
        setattr(h, name, torch.zeros(1, 4))
    with pytest.raises(AttributeError, match="removed in 2.0.6"):
        getattr(StaticFrictionLock, name)
    with pytest.raises(AttributeError, match="removed in 2.0.6"):
        setattr(StaticFrictionLock, name, None)
    assert name not in StaticFrictionLock.STATE_ATTRS
    assert StaticFrictionLock.STATE_ATTRS == (
        "_d_long", "_d_lat", "_was_active", "_rest", "_v_prev_sq")
