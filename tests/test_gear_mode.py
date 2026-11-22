"""Gear mode at the vehicle level (v2.0.2): ``genesis_vehicle.gear`` and its use by
``VehiclePhysics.step`` / ``MultiVehicleKindPhysics.step`` / ``MultiVehiclePhysics.step``
/ ``Vehicle.set_inputs``.

Genesis-free unit tests first (truth tables of the pure functions, input dataclasses),
then CPU scenes: rule 2 (drive torque never against the engaged gear, including a custom
drivetrain that always produces negative torque, B1), rule 4 (a shift only at clamped
throttle 0 and planar speed < 0.5 m/s, and always when allowed), no shift on the
first-step protection step, state carry (rebuild / partial reset / full reset), the
skid-steer refusals before any state is written (M2), input handling (M3, M6, Q3).
``drive_random_and_check`` is also the core of the AC-2/AC-3 acceptance harness.

| abbr | meaning |
|---|---|
| v_xy | planar chassis speed ``hypot(vel_x, vel_y)`` (m/s) |
| M2 / M3 / M6 / B1 / Q3 | plan_r202 review items: tank refusal sites / gear read before conversion / hooks see the effective throttle / clamp after the shift / scalar-vs-tensor checks |
"""
import dataclasses
import math
import os

import pytest
import torch

from genesis_vehicle import gear as G
from genesis_vehicle.inputs import (AckermannInputs, NoSteerInputs, PartialAckermannInputs,
                                    SkidSteerInputs, VehicleInputs)

HERE = os.path.dirname(os.path.abspath(__file__))
CAR = os.path.join(HERE, "..", "samples", "urdf", "car_4w.urdf")
TANK = os.path.join(HERE, "data", "tank_ref.urdf")


# --- Genesis-free: the pure functions ----------------------------------------------

def _t(*v):
    return torch.tensor(v, dtype=torch.float32)


def test_constants():
    assert (G.GEAR_FORWARD, G.GEAR_REVERSE, G.GEAR_SHIFT_MAX_SPEED) == (1, -1, 0.5)


def test_update_gear_truth_table():
    # rows: req, state, throttle, v_xy -> shifts?
    req = _t(-1, -1, -1, -1, -1, 1, 0, 1, -1)
    st = _t(1, 1, 1, 1, -1, -1, 1, 1, 1)
    thr = _t(0, 0.1, 0, -0.3, 0, 0, 0, 0, 0)
    vxy = _t(0.0, 0.0, 0.5, 0.2, 0.0, 0.49, 0.0, 0.0, 0.4999)
    new, shifted = G.update_gear(req, st, thr, vxy)
    assert shifted.tolist() == [True, False, False, True, False, True, False, False, True]
    assert new.tolist() == [-1, 1, 1, -1, -1, 1, 1, 1, -1]


def test_effective_throttle_and_legacy_rows():
    thr = _t(0.5, -0.4, 1.7, -0.4)
    req = _t(1, -1, -1, 0)
    st = _t(1, -1, -1, 1)
    assert G.effective_throttle(thr, req, st).tolist() == pytest.approx([0.5, -0.0, -1.0, -0.4])


def test_clamp_drive_to_gear():
    T = _t(5.0, -5.0, 0.0).repeat(4, 1)          # (4 rows, 3 wheels)
    req = _t(1, 1, 0, -1)
    st = _t(1, -1, 1, -1)
    out = G.clamp_drive_to_gear(T, req, st)
    assert out[0].tolist() == [5.0, 0.0, 0.0]      # forward: >= 0
    assert out[1].tolist() == [0.0, -5.0, 0.0]     # reverse engaged: <= 0
    assert out[2].tolist() == [5.0, -5.0, 0.0]     # legacy row untouched
    assert out[3].tolist() == [0.0, -5.0, 0.0]


def test_encode_requests():
    assert G.encode_requests([None, None], 3, "cpu", torch.float32) is None
    r = G.encode_requests([1, None, _t(-1, 0, 1)], 3, "cpu", torch.float32)
    # env-major / vehicle-minor: (N=3, K=3) -> 9
    assert r.reshape(3, 3).tolist() == [[1, 0, -1], [1, 0, 1], [1, 0, 1]]
    with pytest.raises(ValueError):
        G.encode_requests([_t(1, 1)], 3, "cpu", torch.float32)


def test_validate_request():
    G.validate_request(None, -1.0, where="x")              # gear None: nothing checked
    G.validate_request(1, 0.5, where="x")
    G.validate_request(-1.0, 0.0, where="x")
    G.validate_request(_t(0, 2, -3), _t(-1, 2, 0), where="x")   # tensors are not refused
    for bad in (0, 2, 0.5, True, "1"):
        with pytest.raises(ValueError):
            G.validate_request(bad, 0.5, where="x")
    with pytest.raises(ValueError, match="magnitude"):
        G.validate_request(1, -0.2, where="x")


def test_refuses_gear_four_combinations():
    from genesis_vehicle import RWD, PerSide, Ackermann, SkidSteer

    class R:
        def __init__(self, d, s):
            self.drivetrain, self.steering = d, s
    rwd, ps = RWD.__new__(RWD), PerSide.__new__(PerSide)      # type is all that is read
    ack, sk = Ackermann.__new__(Ackermann), SkidSteer.__new__(SkidSteer)
    assert not G.refuses_gear(R(rwd, ack))
    assert G.refuses_gear(R(ps, ack))
    assert G.refuses_gear(R(rwd, sk))
    assert G.refuses_gear(R(ps, sk))


def test_with_throttle_never_mutates_and_needs_a_dataclass():
    a = AckermannInputs(0.5, 0.0, 0.1, gear=-1)
    b = G.with_throttle(a, -0.5)
    assert a.throttle == 0.5 and b.throttle == -0.5 and b.gear == -1 and b is not a

    class Plain:
        throttle = 0.0
    with pytest.raises(TypeError, match="dataclass"):
        G.with_throttle(Plain(), 0.0)


# --- Genesis-free: inputs --------------------------------------------------------------

def test_inputs_gear_is_keyword_only_and_positional_construction_unchanged():
    vi = VehicleInputs(0.1, 0.2, 0.3)
    assert (vi.throttle, vi.brake, vi.steer, vi.gear) == (0.1, 0.2, 0.3, None)
    with pytest.raises(TypeError):
        VehicleInputs(0.1, 0.2, 0.3, 1)           # R1: no positional 4th slot
    assert VehicleInputs(0.1, 0.2, 0.3, gear=-1).gear == -1
    assert AckermannInputs.from_unified(VehicleInputs(1, 0, 0, gear=-1)).gear == -1
    assert PartialAckermannInputs.from_unified(VehicleInputs(1, 0, 0, gear=1)).gear == 1
    assert NoSteerInputs.from_unified(VehicleInputs(1, 0, 0, gear=-1)).gear == -1
    assert [f.name for f in dataclasses.fields(SkidSteerInputs)] == ["throttle", "brake",
                                                                     "steer_diff"]


def test_skidsteer_from_unified_refuses_a_gear():                    # AC-13 (iv)
    assert SkidSteerInputs.from_unified(VehicleInputs(0.5, 0, 0.1)).throttle == 0.5
    with pytest.raises(ValueError, match="skid-steer"):
        SkidSteerInputs.from_unified(VehicleInputs(0.5, 0, 0.1, gear=1))


# --- CPU scenes -------------------------------------------------------------------------

def _genesis():
    return pytest.importorskip("genesis")


def _scene(vehicles, *, solver="batched", n_envs=1, raycast_mode="dual_scene"):
    """vehicles: [(name, urdf, preset or None, cfg or None, pos)]"""
    gs = _genesis()
    from genesis_vehicle import VehicleScene
    VehicleScene.init_backend("cpu")
    vs = VehicleScene(dt=0.025, substeps=4, n_envs=n_envs, raycast_mode=raycast_mode,
                      show_viewer=False, solver=solver)
    vs.add_ground_plane(friction=1.0)
    out = {}
    for name, urdf, preset, cfg, pos in vehicles:
        out[name] = vs.add_vehicle(urdf, preset, cfg=cfg, pos=pos, name=name,
                                   material=gs.materials.Rigid(friction=1.0))
    vs.build()
    return vs, out


def _phys_rows(vs, veh):
    """(VehiclePhysics holding the rows, row index) for this vehicle."""
    if veh.physics is not None:
        return veh.physics, slice(None)
    mvp = vs._mvp
    for flat_i, kind_idx, slot in mvp._flat_to_kind:
        if flat_i == veh._slot:
            kind = mvp.kinds[kind_idx]
            return kind._proto, kind.slot_rows(slot)
    raise AssertionError


def drive_random_and_check(vs, vehs, *, steps, seed, n_envs, gear_fn=None, thr_lo=-0.5):
    """Drive ``vehs`` (dict name -> Vehicle) with random gear-mode inputs for ``steps``
    steps and count, per step and env, violations of rule 2 and rule 4:

    * ``tneg``: a wheel with ``last_T_drive * engaged_gear < 0`` after the step
      (engaged = the post-shift gear of that step);
    * ``bad_shift``: the engaged gear changed although the clamped throttle was not 0
      or the planar speed read before the step was >= 0.5 m/s, or on the first-step
      protection step;
    * ``missed_shift``: request != engaged, clamped throttle 0, speed < 0.5 m/s
      (more than 1e-4 from the limit), not the first-step step — and no shift.
    Inputs: phases of 30 steps alternate "drive" (throttle uniform in [thr_lo, 1] per
    env, brake 0) and "stop" (throttle 0 or a negative tensor value, brake 1); each env's
    gear request flips with probability 1/40 per step."""
    gen = torch.Generator().manual_seed(seed)
    import genesis as gs
    dev, fdt = gs.device, gs.tc_float
    req = {n: torch.ones(n_envs) for n in vehs}
    cnt = dict(tneg=0, bad_shift=0, missed_shift=0, shifts=0, steps=0)
    for k in range(steps):
        stop = (k // 30) % 2 == 1
        pre = {}
        for n, veh in vehs.items():
            flip = torch.rand(n_envs, generator=gen) < (1.0 / 40.0)
            req[n] = torch.where(flip, -req[n], req[n])
            if stop:
                thr = torch.where(torch.rand(n_envs, generator=gen) < 0.5,
                                  torch.zeros(n_envs), -0.3 * torch.ones(n_envs))
                brk = torch.ones(n_envs)
            else:
                thr = thr_lo + (1.0 - thr_lo) * torch.rand(n_envs, generator=gen)
                brk = torch.zeros(n_envs)
            st = 0.4 * (2 * torch.rand(n_envs, generator=gen) - 1)
            g = req[n] if gear_fn is None else gear_fn(n, k, req[n])
            veh.set_inputs(throttle=thr.to(dev, fdt), brake=brk.to(dev, fdt),
                           steer=st.to(dev, fdt), gear=g.to(dev, fdt))
            p, rows = _phys_rows(vs, veh)
            vel = veh.get_vel().reshape(n_envs, 3)
            pre[n] = dict(g=veh.gear.clone().float(), thr=thr, req=g.clone(),
                          vxy=torch.hypot(vel[:, 0], vel[:, 1]).cpu())
        vs.step()
        cnt["steps"] += 1
        for n, veh in vehs.items():
            p, rows = _phys_rows(vs, veh)
            ga = veh.gear.float()
            td = p.last_T_drive[rows].cpu()
            cnt["tneg"] += int(((td * ga.unsqueeze(-1)) < 0).any(dim=1).sum())
            b = pre[n]
            # The first-step protection step returns before the chassis read and
            # leaves _stepped_once False: no shift may happen on it.
            first = ~p._stepped_once[rows].cpu()
            changed = ga != b["g"]
            cnt["shifts"] += int(changed.sum())
            ok_cond = (b["thr"].clamp(0, 1) == 0) & (b["vxy"] < 0.5) & ~first
            cnt["bad_shift"] += int((changed & ~ok_cond).sum())
            want = (b["req"].sign() != b["g"]) & ok_cond & ((b["vxy"] - 0.5).abs() > 1e-4)
            cnt["missed_shift"] += int((want & ~changed).sum())
    return cnt


@pytest.mark.parametrize("solver", ["batched", "per_vehicle"])
def test_rules_2_and_4_hold_under_random_gear_inputs(solver):
    from genesis_vehicle import car_4w_awd_ackermann, car_4w_fwd_ackermann, car_4w_rwd_ackermann
    vs, vehs = _scene([("rwd", CAR, car_4w_rwd_ackermann, None, (0.0, 0.0, 0.5)),
                       ("fwd", CAR, car_4w_fwd_ackermann, None, (0.0, 8.0, 0.5)),
                       ("awd", CAR, car_4w_awd_ackermann, None, (0.0, -8.0, 0.5))],
                      solver=solver, n_envs=2)
    c = drive_random_and_check(vs, vehs, steps=240, seed=3, n_envs=2)
    assert c["tneg"] == 0 and c["bad_shift"] == 0 and c["missed_shift"] == 0, c
    assert c["shifts"] >= 4, c                    # the rules were exercised


def _always_negative_rwd():
    """An RWD whose drive torque is -100 N*m on every wheel ALWAYS (throttle 0 too)."""
    from genesis_vehicle import RWD

    class NegRWD(RWD):
        def distribute_torque(self, inputs, omega, wheel_meta, device, dtype):
            _, tb = super().distribute_torque(inputs, omega, wheel_meta, device, dtype)
            return torch.full_like(tb, -100.0), tb
    return NegRWD


@pytest.mark.parametrize("solver", ["batched", "per_vehicle"])
def test_clamp_uses_the_post_shift_gear_with_a_custom_drivetrain(solver):
    """B1: the shift step from -1 to +1 must already clamp to +1 (a clamp with the
    pre-shift gear would let this drivetrain's -100 N*m through on that step)."""
    from genesis_vehicle import car_4w_rwd_ackermann
    cfg = car_4w_rwd_ackermann(CAR)
    cfg.drivetrain.__class__ = _always_negative_rwd()
    vs, vehs = _scene([("car", CAR, None, cfg, (0.0, 0.0, 0.5))], solver=solver)
    veh = vehs["car"]
    p, rows = _phys_rows(vs, veh)
    seen = []
    for k in range(60):
        g = -1 if k < 30 else 1
        veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0, gear=g)
        before = int(veh.gear[0])
        vs.step()
        after = int(veh.gear[0])
        td = p.last_T_drive[rows]
        assert bool(((td * after) >= 0).all()), (k, before, after, td.tolist())
        seen.append((before, after))
    assert (-1, 1) in seen and (1, -1) in seen     # both shift steps happened


def test_no_shift_on_the_first_step_protection_step():
    from genesis_vehicle import car_4w_rwd_ackermann
    vs, vehs = _scene([("car", CAR, car_4w_rwd_ackermann, None, (0.0, 0.0, 0.5))],
                      solver="per_vehicle", raycast_mode="single_scene")
    veh = vehs["car"]
    assert not veh.physics._prev_init
    veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0, gear=-1)
    vs.step()
    assert int(veh.gear[0]) == 1                   # first-step protection: no state read
    vs.step()
    assert int(veh.gear[0]) == -1                  # deferred request engages next step


def test_slope_start_drive_with_the_gear_against_the_roll():
    """AC-2: rolling backwards (a 10 degree grade emulated by tilted gravity) in
    forward gear at throttle 1: the driven (rear) wheels get positive drive torque."""
    from genesis_vehicle import car_4w_rwd_ackermann
    vs, vehs = _scene([("car", CAR, car_4w_rwd_ackermann, None, (0.0, 0.0, 0.5))],
                      solver="per_vehicle")
    veh = vehs["car"]
    th = math.radians(10.0)
    vs.set_gravity((-9.81 * math.sin(th), 0.0, -9.81 * math.cos(th)))
    for _ in range(20):
        veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0, gear=1)
        vs.step()
    v = 0.0
    for _ in range(200):
        veh.set_inputs(throttle=0.0, brake=0.0, steer=0.0, gear=1)
        vs.step()
        v = float(veh.get_vel().reshape(-1, 3)[0, 0])
        if v < -0.5:
            break
    assert v < -0.5
    veh.set_inputs(throttle=1.0, brake=0.0, steer=0.0, gear=1)
    vs.step()
    td = veh.physics.last_T_drive[0]
    rear = veh.physics.wheel_meta.axle_index == veh.physics.wheel_meta.axle_index.max()
    assert bool((td[rear] > 0).all()), td.tolist()
    assert int(veh.gear[0]) == 1


def test_state_carry_rebuild_partial_and_full_reset():                    # AC-9
    from genesis_vehicle import car_4w_rwd_ackermann
    vs, vehs = _scene([("a", CAR, car_4w_rwd_ackermann, None, (0.0, 0.0, 0.5)),
                       ("b", CAR, car_4w_rwd_ackermann, None, (0.0, 8.0, 0.5))],
                      solver="batched", n_envs=2)
    for _ in range(10):
        for veh in vehs.values():
            veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0, gear=-1)
        vs.step()
    assert vehs["a"].gear.tolist() == [-1, -1] and vehs["b"].gear.tolist() == [-1, -1]
    old = vs._mvp
    vs.mark_config_dirty()
    for veh in vehs.values():
        veh.set_inputs(throttle=0.3, brake=0.0, steer=0.0, gear=1)   # cannot shift (thr > 0)
    vs.step()
    assert vs._mvp is not old                       # really rebuilt
    assert vehs["a"].gear.tolist() == [-1, -1] and vehs["b"].gear.tolist() == [-1, -1]
    vs._mvp.reset(vehicle_ids=[1])                  # per-vehicle: every env of "b"
    assert vehs["a"].gear.tolist() == [-1, -1] and vehs["b"].gear.tolist() == [1, 1]
    kind = vs._mvp.kinds[0]
    kind._proto._gear.fill_(-1.0)
    kind.reset(rows=[0])                            # flat row 0 = env 0, slot 0 ("a")
    assert vehs["a"].gear.tolist() == [1, -1] and vehs["b"].gear.tolist() == [-1, -1]
    vs.reset()
    assert vehs["a"].gear.tolist() == [1, 1] and vehs["b"].gear.tolist() == [1, 1]
    snap = kind._proto.export_runtime_state()
    assert "_gear" in snap
    del snap["_gear"]                               # a pre-2.0.2 snapshot: skipped
    kind._proto._gear.fill_(-1.0)
    kind._proto.load_runtime_state(snap)
    assert kind._proto._gear.tolist() == [-1.0] * 4


def test_per_vehicle_partial_reset_touches_only_those_envs():
    from genesis_vehicle import car_4w_rwd_ackermann
    vs, vehs = _scene([("a", CAR, car_4w_rwd_ackermann, None, (0.0, 0.0, 0.5))],
                      solver="per_vehicle", n_envs=3)
    p = vehs["a"].physics
    p._gear.fill_(-1.0)
    p.reset(env_ids=torch.tensor([1]))
    assert vehs["a"].gear.tolist() == [-1, 1, -1]


# --- M2: skid-steer refusals before any state is written -------------------------------

def _state(p):
    return [t.clone() for t in (p.omega, p.prev_compression, p.last_T_drive, p._gear)]


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def test_tank_refusals_leave_every_state_untouched():                      # AC-13
    from genesis_vehicle import (MultiVehiclePhysics, VehicleInputs, car_4w_rwd_ackermann,
                                 tank_skid_belt)
    vs, vehs = _scene([("car", CAR, car_4w_rwd_ackermann, None, (0.0, 0.0, 0.5)),
                       ("tank", TANK, tank_skid_belt, None, (0.0, -10.0, 1.5))],
                      solver="batched")
    car, tank = vehs["car"], vehs["tank"]
    for _ in range(5):
        car.set_inputs(throttle=0.5, brake=0.0, steer=0.0)
        tank.set_inputs(throttle=0.5, brake=0.0, steer=0.0)
        vs.step()
    # (i) set_inputs
    before = tank._inputs
    with pytest.raises(ValueError, match="skid-steer"):
        tank.set_inputs(throttle=0.5, brake=0.0, steer=0.0, gear=1)
    assert tank._inputs is before
    # (vii) tank reads no gear
    assert tank.gear is None and car.gear.tolist() == [1]
    # (iii) MultiVehiclePhysics.step with two kinds: the car kind must not move
    mvp = vs._mvp
    car_kind = mvp.kinds[[k for f, k, s in mvp._flat_to_kind if f == car._slot][0]]
    st_car = _state(car_kind._proto)
    ins = [VehicleInputs(0.5, 0.0, 0.0), VehicleInputs(0.5, 0.0, 0.0, gear=1)]
    with pytest.raises(ValueError, match="skid-steer"):
        mvp.step(ins)
    assert _same(st_car, _state(car_kind._proto))
    tank_kind = mvp.kinds[[k for f, k, s in mvp._flat_to_kind if f == tank._slot][0]]
    st_tank = _state(tank_kind._proto)
    with pytest.raises(ValueError, match="skid-steer"):           # the kind called directly
        tank_kind.step([VehicleInputs(0.5, 0.0, 0.0, gear=-1)])
    assert _same(st_tank, _state(tank_kind._proto))
    # (vi) GearCommand.signed() is a signed input: accepted
    from genesis_vehicle import GearCommand
    thr, st, br = GearCommand(0.4, 0.1, 0.0, -1).signed()
    tank.set_inputs(throttle=thr, brake=br, steer=st)
    vs.step()


def test_tank_refusal_in_a_single_kind_and_per_vehicle_driver():
    from genesis_vehicle import VehicleInputs, tank_skid_belt
    vs, vehs = _scene([("tank", TANK, tank_skid_belt, None, (0.0, 0.0, 1.5))],
                      solver="batched")
    mvp = vs._mvp
    vs.step()
    st = _state(mvp.kinds[0]._proto)
    with pytest.raises(ValueError, match="skid-steer"):
        mvp.step([VehicleInputs(0.5, 0.0, 0.0, gear=1)])
    assert _same(st, _state(mvp.kinds[0]._proto))
    vs2, vehs2 = _scene([("tank", TANK, tank_skid_belt, None, (0.0, 0.0, 1.5))],
                        solver="per_vehicle")
    p = vehs2["tank"].physics
    vs2.step()
    st = _state(p)
    with pytest.raises(ValueError, match="skid-steer"):            # (ii)
        p.step(VehicleInputs(0.5, 0.0, 0.0, gear=1))
    assert _same(st, _state(p))


def test_tank_gear_follower_refusals():                                    # AC-13 (v)
    from genesis_vehicle import DifferentiablePlant, PathFollower, tank_skid_belt
    from genesis_vehicle.tests.data.make_pd_golden import make_table
    vs, vehs = _scene([("tank", TANK, tank_skid_belt, None, (0.0, 0.0, 1.5))],
                      solver="per_vehicle")
    tank = vehs["tank"]
    path = [(0, 0, 0, 1.0), (5, 0, 0, 1.0), (10, 0, 0, 1.0)]
    with pytest.raises(ValueError, match="skid-steer"):
        PathFollower(path, DifferentiablePlant(tank), mpc_model="off", command_mode="gear")
    f = PathFollower(path, make_table(), mpc_model="off", command_mode="gear", brake_gain=5.0)
    cmd = f.step((0.0, 0.0), 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="skid-steer"):
        tank.set_inputs(throttle=cmd.throttle, brake=cmd.brake, steer=cmd.steer, gear=cmd.gear)
    assert PathFollower(path, DifferentiablePlant(tank), mpc_model="off").command_mode == "signed"


# --- M3 / M6 / Q3: input handling -----------------------------------------------------

@dataclasses.dataclass
class _DropsGear:
    """A custom steering InputType whose from_unified DROPS the gear."""
    throttle: object
    brake: object
    steer: object

    @classmethod
    def from_unified(cls, vi):
        return cls(vi.throttle, vi.brake, vi.steer)


class _PlainInputs:
    """A custom steering InputType that is not a dataclass."""
    def __init__(self, throttle, brake, steer):
        self.throttle, self.brake, self.steer = throttle, brake, steer

    @classmethod
    def from_unified(cls, vi):
        return cls(vi.throttle, vi.brake, vi.steer)


class _Recorder:
    """A PRE_LOOP hook recording what the hooks see (``ctx.throttle`` / ``ctx.gear``)."""
    slots = ("PRE_LOOP",)

    def __init__(self):
        self.throttle, self.gear = [], []

    def apply_pre_loop(self, ctx):
        self.throttle.append(ctx.throttle.clone())
        self.gear.append(None if ctx.gear is None else ctx.gear.clone())


def _car_with(input_type=None, hook=None, solver="per_vehicle"):
    from genesis_vehicle import Ackermann, car_4w_rwd_ackermann
    cfg = car_4w_rwd_ackermann(CAR)
    if input_type is not None:
        cls = type("CustomSteer", (type(cfg.steering),), {"InputType": input_type})
        cfg.steering.__class__ = cls
        assert isinstance(cfg.steering, Ackermann)
    if hook is not None:
        cfg.stability_hooks = list(cfg.stability_hooks) + [hook]
    return _scene([("car", CAR, None, cfg, (0.0, 0.0, 0.5))], solver=solver)


def test_gear_is_read_before_a_custom_from_unified_drops_it():             # AC-14 (M3)
    from genesis_vehicle import VehicleInputs
    vs, vehs = _car_with(_DropsGear)
    p = vehs["car"].physics
    for _ in range(10):                             # at rest, throttle 0: engage reverse
        vehs["car"].set_inputs(throttle=0.0, brake=1.0, steer=0.0, gear=-1)
        vs.step()
    assert int(vehs["car"].gear[0]) == -1
    vi = VehicleInputs(0.5, 0.0, 0.0, gear=-1)
    d = vs._measure_distances()[vehs["car"]]
    p.step(vi, distances=d)
    assert (vi.throttle, vi.gear) == (0.5, -1)      # the caller's object is untouched
    rear = p.wheel_meta.axle_index == p.wheel_meta.axle_index.max()
    assert bool((p.last_T_drive[0][rear] < 0).all()), p.last_T_drive.tolist()


def test_a_non_dataclass_input_type_cannot_carry_gear_mode():
    from genesis_vehicle import VehicleInputs
    vs, vehs = _car_with(_PlainInputs)
    p = vehs["car"].physics
    d = vs._measure_distances()[vehs["car"]]
    p.step(VehicleInputs(0.2, 0.0, 0.0), distances=d)          # legacy: fine
    with pytest.raises(TypeError, match="dataclass"):
        p.step(VehicleInputs(0.2, 0.0, 0.0, gear=1), distances=d)


def test_scalar_checks_only_with_a_gear_and_the_hooks_see_the_effective_throttle():
    import genesis as gs_
    from genesis_vehicle import VehicleInputs
    rec = _Recorder()
    vs, vehs = _car_with(hook=rec)
    veh = vehs["car"]
    p = veh.physics
    with pytest.raises(ValueError, match="magnitude"):        # Q3 at set_inputs
        veh.set_inputs(throttle=-0.5, brake=0.0, steer=0.0, gear=1)
    d = vs._measure_distances()[veh]
    with pytest.raises(ValueError, match="magnitude"):        # and at the driver
        p.step(VehicleInputs(-0.5, 0.0, 0.0, gear=1), distances=d)
    for _ in range(5):
        veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        vs.step()
    veh.set_inputs(throttle=-0.5, brake=0.0, steer=0.0)       # legacy reverse: allowed
    vs.step()
    rear = p.wheel_meta.axle_index == p.wheel_meta.axle_index.max()
    assert bool((p.last_T_drive[0][rear] < 0).all())
    assert rec.gear[-1] is None and float(rec.throttle[-1][0]) == -0.5
    neg = torch.tensor([-0.7], device=gs_.device, dtype=gs_.tc_float)
    veh.set_inputs(throttle=neg, brake=0.0, steer=0.0, gear=1)  # tensor: clamped, not refused
    vs.step()
    assert float(rec.throttle[-1][0]) == 0.0 and float(rec.gear[-1][0]) == 1.0
    assert bool((p.last_T_drive == 0).all())


def test_multi_kind_precheck_refuses_shape_and_input_type_before_any_kind_steps():
    """impl-review F5: a gear tensor of the wrong shape, or a steering input type that
    cannot carry the effective throttle, on the SECOND kind raises from the
    MultiVehiclePhysics precheck — the first kind's state is untouched."""
    import genesis as gs_
    from genesis_vehicle import VehicleInputs, car_4w_fwd_ackermann, car_4w_rwd_ackermann
    cfg = car_4w_fwd_ackermann(CAR)
    cls = type("PlainSteer", (type(cfg.steering),), {"InputType": _PlainInputs})
    cfg.steering.__class__ = cls
    vs, vehs = _scene([("a", CAR, car_4w_rwd_ackermann, None, (0.0, 0.0, 0.5)),
                       ("b", CAR, car_4w_fwd_ackermann, None, (0.0, 8.0, 0.5)),
                       ("c", CAR, None, cfg, (0.0, -8.0, 0.5))],
                      solver="batched", n_envs=2)
    for _ in range(5):
        for veh in vehs.values():
            veh.set_inputs(throttle=0.4, brake=0.0, steer=0.0)
        vs.step()
    mvp = vs._mvp
    first = mvp.kinds[[k for f, k, s in mvp._flat_to_kind if f == vehs["a"]._slot][0]]
    st = _state(first._proto)
    bad = torch.ones(3, device=gs_.device, dtype=gs_.tc_float)          # n_envs is 2
    ok = VehicleInputs(0.4, 0.0, 0.0, gear=1)
    with pytest.raises(ValueError, match="shape"):
        mvp.step([ok, VehicleInputs(0.4, 0.0, 0.0, gear=bad), VehicleInputs(0.4, 0.0, 0.0)])
    assert _same(st, _state(first._proto))
    with pytest.raises(TypeError, match="dataclass"):
        mvp.step([ok, VehicleInputs(0.4, 0.0, 0.0), VehicleInputs(0.4, 0.0, 0.0, gear=1)])
    assert _same(st, _state(first._proto))
    mvp.step([ok, VehicleInputs(0.4, 0.0, 0.0), VehicleInputs(0.4, 0.0, 0.0)])   # legal: steps
    assert not _same(st, _state(first._proto))
