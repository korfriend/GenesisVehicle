"""The build-time hoists stay LIVE — G1, G2, G3, G4, G6.

Five quantities the step path used to recompute on every call are now derived
once and cached (``_hotset.derived``): the Ackermann geometry, the driven-axle
share, the brake-bias vector, the AWD weight normalisation, and
``StaticFrictionLock``'s squared speed threshold.

Every test here is a CHARACTERISATION baseline — each number was produced by
the tree BEFORE the hoist, so a failure after it is attributable to the hoist.
They are written the strict way round: derive the value FIRST (one call), then
write the source, then assert the NEXT call moved. A "compute it once at
construction" hoist passes a test that writes the source before the first call
and fails every test below.

GATE 0 (a 200-step value-equality rollout) cannot see any of this: it never
mutates config. That is the whole reason these gates exist.

| abbr | meaning |
|---|---|
| AWD | all-wheel drive (`strategies.drivetrain.AWD`) |
| RWD | rear-wheel drive |
| SFL | `StaticFrictionLock` (`strategies/stability.py`) |
| T_drive_pw / T_brake_pw | per-wheel drive / brake torque, N*m, shape (n_envs, n_wheels) |
| v_thr | SFL activation speed threshold (m/s) |
"""

from __future__ import annotations

import os
import types

import pytest
import torch

from genesis_vehicle.config import resolve
from genesis_vehicle.core import build_wheel_meta
from genesis_vehicle.inputs import AckermannInputs
from genesis_vehicle.presets import car_4w_awd_ackermann, car_4w_rwd_ackermann
from genesis_vehicle.strategies.stability import StaticFrictionLock

CAR_URDF = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "samples", "urdf", "car_4w.urdf"))

DEV, FDT = torch.device("cpu"), torch.float32


def _built(preset):
    resolved = resolve(preset(CAR_URDF, stability="control"))
    return resolved, build_wheel_meta(resolved, DEV, FDT)


def _row(t) -> list[float]:
    return [round(float(x), 6) for x in t[0]]


# --- G1: Ackermann geometry (`_geom`) ----------------------------------------

G1_BASE = [0.320031, 0.385855, 0.0, 0.0]
G1_WB_3 = [0.322803, 0.381958, 0.0, 0.0]


def test_g1_wheelbase_still_reaches_the_next_per_wheel_steer():
    resolved, wm = _built(car_4w_rwd_ackermann)
    steering = resolved.steering
    ins = AckermannInputs(throttle=0.0, brake=0.0, steer=0.5)

    assert _row(steering.per_wheel_steer(ins, 1, wm, DEV, FDT)) == G1_BASE
    steering.wheelbase = 3.0                 # AFTER the geometry was derived
    assert _row(steering.per_wheel_steer(ins, 1, wm, DEV, FDT)) == G1_WB_3


def test_g1b_track_width_is_read_but_inert():
    """`steering.py` binds `wb, tw` and never reads `tw` again, so a
    track_width write is bit-equal by design. Pinned so that a future change
    making it live turns this test — not a user's trajectory — red."""
    resolved, wm = _built(car_4w_rwd_ackermann)
    ins = AckermannInputs(throttle=0.0, brake=0.0, steer=0.5)
    base = resolved.steering.per_wheel_steer(ins, 1, wm, DEV, FDT).clone()
    resolved.steering.track_width = 9.0
    assert torch.equal(resolved.steering.per_wheel_steer(ins, 1, wm, DEV, FDT), base)


def test_g1_geometry_is_derived_once_not_per_call():
    """The hoist itself: two calls with an unchanged source must not re-run the
    reduction. Counted through `_compute_geometry`, the uncached half."""
    resolved, wm = _built(car_4w_rwd_ackermann)
    steering = resolved.steering
    ins = AckermannInputs(throttle=0.0, brake=0.0, steer=0.5)
    calls = []
    real = steering._compute_geometry
    steering._compute_geometry = lambda meta: (calls.append(1), real(meta))[1]

    for _ in range(5):
        steering.per_wheel_steer(ins, 1, wm, DEV, FDT)
    assert len(calls) == 1
    steering.wheelbase = 3.0
    steering.per_wheel_steer(ins, 1, wm, DEV, FDT)
    assert len(calls) == 2


# --- G2: driven axles (`_driven` / `_drive_share`) ---------------------------

def test_g2_driven_axles_still_reaches_the_next_distribute():
    resolved, wm = _built(car_4w_rwd_ackermann)
    dt = resolved.drivetrain
    omega = torch.zeros(1, wm.n_wheels)
    ins = AckermannInputs(throttle=1.0, brake=0.0, steer=0.0)

    T_drive, _ = dt.distribute_torque(ins, omega, wm, DEV, FDT)
    assert _row(T_drive) == [0.0, 0.0, 500.0, 500.0]

    dt.driven_axles = (0,)                   # AFTER the share was derived
    T_drive, _ = dt.distribute_torque(ins, omega, wm, DEV, FDT)
    assert _row(T_drive) == [500.0, 500.0, 0.0, 0.0]
    assert dt._resolve_driven(wm) == (0,)


def test_g2_an_in_place_edit_of_driven_axles_is_seen():
    """The cache key holds the axle tuple BY VALUE, so mutating the caller's
    list in place is not mistaken for "unchanged"."""
    resolved, wm = _built(car_4w_rwd_ackermann)
    dt = resolved.drivetrain
    dt.driven_axles = [1]
    omega = torch.zeros(1, wm.n_wheels)
    ins = AckermannInputs(throttle=1.0, brake=0.0, steer=0.0)
    assert _row(dt.distribute_torque(ins, omega, wm, DEV, FDT)[0]) == [0.0, 0.0, 500.0, 500.0]
    dt.driven_axles[0] = 0
    assert _row(dt.distribute_torque(ins, omega, wm, DEV, FDT)[0]) == [500.0, 500.0, 0.0, 0.0]


# --- G3: brake bias (`_brake_bias`) ------------------------------------------

def test_g3_brake_bias_still_reaches_the_next_distribute():
    resolved, wm = _built(car_4w_rwd_ackermann)
    dt = resolved.drivetrain
    omega = torch.zeros(1, wm.n_wheels)
    ins = AckermannInputs(throttle=0.0, brake=1.0, steer=0.0)

    assert dt.brake_bias == [0.3, 0.3, 0.2, 0.2]
    _, T_brake = dt.distribute_torque(ins, omega, wm, DEV, FDT)
    assert _row(T_brake) == [750.0, 750.0, 500.0, 500.0]

    dt.brake_bias = [0.1, 0.1, 0.9, 0.9]
    _, T_brake = dt.distribute_torque(ins, omega, wm, DEV, FDT)
    assert _row(T_brake) == [250.0, 250.0, 2250.0, 2250.0]


def test_g3_a_wrong_length_brake_bias_still_raises_every_call():
    """A cached DERIVED value must not cache a REFUSAL away: the ConfigError
    has to fire on the second call too."""
    from genesis_vehicle.config import ConfigError
    resolved, wm = _built(car_4w_rwd_ackermann)
    dt = resolved.drivetrain
    dt.brake_bias = [0.5, 0.5]
    omega = torch.zeros(1, wm.n_wheels)
    ins = AckermannInputs(throttle=0.0, brake=1.0, steer=0.0)
    for _ in range(2):
        with pytest.raises(ConfigError):
            dt.distribute_torque(ins, omega, wm, DEV, FDT)


# --- G4: AWD drive-weight normalisation (`_drive_weights`) -------------------

def test_g4_drive_weights_still_reach_the_next_distribute():
    resolved, wm = _built(car_4w_awd_ackermann)
    dt = resolved.drivetrain
    omega = torch.zeros(1, wm.n_wheels)
    ins = AckermannInputs(throttle=1.0, brake=0.0, steer=0.0)

    assert dt.drive_weights == [0.25, 0.25, 0.25, 0.25]
    T_drive, _ = dt.distribute_torque(ins, omega, wm, DEV, FDT)
    assert _row(T_drive) == [250.0, 250.0, 250.0, 250.0]

    dt.drive_weights = [0.1, 0.1, 0.4, 0.4]
    T_drive, _ = dt.distribute_torque(ins, omega, wm, DEV, FDT)
    assert _row(T_drive) == [100.0, 100.0, 400.0, 400.0]


def test_g4_awd_brake_bias_is_also_live():
    """AWD carried its own byte-identical copy of the default-bias block; the
    two now share one helper, so the AWD arm needs its own gate."""
    resolved, wm = _built(car_4w_awd_ackermann)
    dt = resolved.drivetrain
    omega = torch.zeros(1, wm.n_wheels)
    ins = AckermannInputs(throttle=0.0, brake=1.0, steer=0.0)
    before = _row(dt.distribute_torque(ins, omega, wm, DEV, FDT)[1])
    dt.brake_bias = [0.1, 0.1, 0.9, 0.9]
    after = _row(dt.distribute_torque(ins, omega, wm, DEV, FDT)[1])
    total = sum(before)
    assert after != before
    assert after == [round(0.1 * total, 6), round(0.1 * total, 6),
                     round(0.9 * total, 6), round(0.9 * total, 6)]


# --- the omega cap: None -> +inf is the identity, and stays live -------------

def test_omega_max_drive_none_is_the_identity_taper():
    from genesis_vehicle.strategies.drivetrain import RWD
    d = RWD(t_drive_max=1.0, t_brake_max=1.0)          # omega_max_drive None
    assert d.omega_max_drive is None                    # PUBLIC value unchanged
    assert d._drive_omega_cap() == float("inf")
    T = torch.tensor([[3.0, -4.0, 0.0, 250.0]])
    omega = torch.tensor([[10.0, -20.0, 5.0, 0.0]])
    assert torch.equal(d._rev_limit(T, omega), T)       # VALUE equality, not approx


def test_omega_max_drive_is_still_live_after_the_cap_was_derived():
    from genesis_vehicle.strategies.drivetrain import RWD
    d = RWD(t_drive_max=1.0, t_brake_max=1.0, omega_max_drive=50.0)
    T = torch.tensor([[1.0]])
    omega = torch.tensor([[25.0]])
    assert float(d._rev_limit(T, omega)[0, 0]) == pytest.approx(0.5)
    d.omega_max_drive = None
    assert torch.equal(d._rev_limit(T, omega), T)
    d.omega_max_drive = 25.0
    assert float(d._rev_limit(T, omega)[0, 0]) == pytest.approx(0.0)


# --- G6: SFL `v_thr` (`_v_thr_sq`) ------------------------------------------

def _sfl_f_long(hook, wm, n_calls: int = 5) -> list[float]:
    """SFL's contribution to ctx.F_long after `n_calls` steps at
    v_long=0.7, brake=0.5, N=3000 — the G6 observable."""
    ctx = types.SimpleNamespace()
    for _ in range(n_calls):
        ctx.brake = torch.tensor([0.5])
        ctx.v_long = torch.full((1, wm.n_wheels), 0.7)
        ctx.v_lat = torch.zeros(1, wm.n_wheels)
        ctx.N = torch.full((1, wm.n_wheels), 3000.0)
        ctx.dt = 0.02
        ctx.wheel_meta = wm
        ctx.F_long = torch.zeros(1, wm.n_wheels)
        ctx.F_lat = torch.zeros(1, wm.n_wheels)
        ctx.omega_override = None
        hook.apply_post_tire(ctx)
    return _row(ctx.F_long)


def test_g6_v_thr_below_the_speed_leaves_the_lock_inactive():
    _resolved, wm = _built(car_4w_rwd_ackermann)
    hook = StaticFrictionLock(brake_thr=0.3, v_thr=0.5)
    assert _sfl_f_long(hook, wm) == [0.0] * wm.n_wheels


def test_g6_v_thr_is_still_live_after_the_square_was_derived():
    """The gate for the `_v_thr_sq` hoist, in the shape `TankTuning` uses it
    (write v_thr on the constructed hook). Derive first, then write."""
    _resolved, wm = _built(car_4w_rwd_ackermann)
    hook = StaticFrictionLock(brake_thr=0.3, v_thr=0.5)
    assert _sfl_f_long(hook, wm) == [0.0] * wm.n_wheels
    hook.v_thr = 5.0                          # TankTuning's value
    assert _sfl_f_long(hook, wm) == [-3000.0] * wm.n_wheels


def test_g6_the_square_is_computed_in_double_and_stored_per_row():
    """float32(s)*float32(s) != float32(double(s)**2) for some s, so the square
    is taken in python double; and it is a per-ROW tensor, not a scalar,
    because a fused group needs one threshold per vehicle."""
    from genesis_vehicle._hotset import derived_value
    s = 0.1
    hook = StaticFrictionLock(v_thr=s)
    hook._v_thr_squared(torch.zeros((), dtype=torch.float32))
    sq = derived_value(hook, "_v_thr_sq")
    assert torch.is_tensor(sq) and sq.shape == (1, 1)
    assert sq.dtype == torch.float32
    f32 = torch.tensor(s, dtype=torch.float32)
    assert float(sq) == float(torch.tensor(float(s) ** 2, dtype=torch.float32))
    assert float(sq) != float(f32 * f32)      # the two really do differ at 0.1


# --- where the derived values live, and who must not see them ---------------

def test_a_derived_cache_is_invisible_to_a_generic_attribute_sweep():
    """Two generic walkers already traverse these config objects and neither
    may see a derived cache:

    * the fusion key (``tools/fusion_key._sweep``) is FAIL-CLOSED over every
      NON-DUNDER attribute, so a visible cache slot becomes a key item and
      splits two structurally identical vehicles;
    * ``control/plant._sync_hooks`` copies every TENSOR attribute of a hook,
      row-slicing and interleaving it by candidate count — right for the
      stick-slip integrator state, wrong for a config tensor.

    Keeping the cache OFF the owner keeps both correct with no name list in
    either — nothing about the owner changes at all.
    """
    from genesis_vehicle._hotset import prime_derived

    resolved, wm = _built(car_4w_rwd_ackermann)
    hook = StaticFrictionLock(brake_thr=0.3, v_thr=0.5)
    resolved.stability_hooks = list(resolved.stability_hooks) + [hook]
    roles = [resolved.steering, resolved.drivetrain, resolved.tire, hook]
    before = [set(vars(o)) for o in roles]

    prime_derived(resolved, wm, DEV, FDT)
    _sfl_f_long(hook, wm, n_calls=1)

    # Not "no new NON-dunder attribute" but no new attribute AT ALL: the cache
    # lives off the owner, so deriving must leave the instance untouched.
    for obj, names in zip(roles, before):
        assert set(vars(obj)) == names, (
            f"{type(obj).__name__} grew an attribute: "
            f"{set(vars(obj)) - names}")
    tensors = {k for k, v in hook.__dict__.items() if torch.is_tensor(v)}
    assert tensors == {"_d_long", "_d_lat", "_was_active"}, (
        "the only tensor attributes of a hook may be its INTEGRATOR state — "
        "the three control/plant._sync_hooks row-slices and interleaves. A "
        f"config tensor in here would be sliced as if it were state: {tensors}")


def test_a_different_wheel_meta_re_derives():
    """The DifferentiablePlant builds its OWN WheelMeta and calls the same
    strategies with it. The cache is keyed on meta IDENTITY, so the plant's
    meta must not be answered with the driver's geometry."""
    resolved, wm = _built(car_4w_rwd_ackermann)
    ins = AckermannInputs(throttle=0.0, brake=0.0, steer=0.5)
    assert _row(resolved.steering.per_wheel_steer(ins, 1, wm, DEV, FDT)) == G1_BASE

    other = build_wheel_meta(resolved, DEV, FDT)     # a different object
    other.positions = other.positions.clone()
    other.positions[:, 0] *= 2.0                     # a longer wheelbase
    moved = _row(resolved.steering.per_wheel_steer(ins, 1, other, DEV, FDT))
    assert moved != G1_BASE
    # ... and the original meta still gets its own answer.
    assert _row(resolved.steering.per_wheel_steer(ins, 1, wm, DEV, FDT)) == G1_BASE


# --- the table and the sites do not drift -----------------------------------

def test_every_declared_dependent_is_really_derived_at_build_time():
    """`_hotset.HOT_DEPENDENTS` is documentation unless something checks it.

    For every declared (role, source) -> (slot, ...) pair: some shipped role
    object must actually carry that source, and `prime_derived` — the BUILD
    step — must leave every declared slot populated on it. A pair naming a
    source nothing reads, or a slot nothing derives, fails here rather than
    misleading STEP 3's refresh, which reads this table."""
    from genesis_vehicle._hotset import (
        HOT_DEPENDENTS, derived_value, prime_derived)
    from genesis_vehicle.tire_models import CoulombIsotropic

    _UNSET = object()
    candidates: dict[str, list] = {"steering": [], "drivetrain": [],
                                   "tire": [], "hook": []}
    for preset in (car_4w_rwd_ackermann, car_4w_awd_ackermann):
        resolved, wm = _built(preset)
        resolved.tire = CoulombIsotropic(eps_v=0.5)   # the model with a square
        prime_derived(resolved, wm, DEV, FDT)
        candidates["steering"].append(resolved.steering)
        candidates["drivetrain"].append(resolved.drivetrain)
        candidates["tire"].append(resolved.tire)
        candidates["hook"] += [h for h in resolved.stability_hooks
                               if isinstance(h, StaticFrictionLock)]

    assert candidates["hook"], "the control profile must ship a SFL hook"
    for (role, source), slots in HOT_DEPENDENTS.items():
        owners = [o for o in candidates[role] if hasattr(o, source)]
        assert owners, f"no shipped {role} carries {source!r}"
        assert slots, f"{role}.{source} declares no dependent"
        for owner in owners:
            for slot in slots:
                assert derived_value(owner, slot, _UNSET) is not _UNSET, (
                    f"{type(owner).__name__}.{slot} was not derived at build "
                    f"time although {role}.{source} declares it")


# --- the cache must cost the owner NOTHING: pickling, and duck-typed metas ---

def test_a_config_is_still_picklable_after_it_has_derived():
    """A derived cache holds a weak reference to the ``wheel_meta``. Keeping it
    in the owner's ``__dict__`` made the owner unpicklable
    (``TypeError: cannot pickle 'weakref.ReferenceType' object``) — and only
    AFTER the first derive, so a config pickled fine right after ``build()``
    and failed one step later. RL vector-env workers (spawn/fork), tuned-config
    checkpoints and ``torch.save(cfg)`` all pickle configs; nothing in the SDK,
    the samples or the server does, which is why the rest of the suite is blind
    to this."""
    import pickle

    resolved, wm = _built(car_4w_rwd_ackermann)
    pickle.dumps(resolved)                       # ... before any derive
    ins = AckermannInputs(throttle=0.0, brake=0.0, steer=0.5)
    resolved.steering.per_wheel_steer(ins, 1, wm, DEV, FDT)
    resolved.drivetrain.distribute_torque(
        ins, torch.zeros(1, wm.n_wheels), wm, DEV, FDT)

    pickle.dumps(resolved.steering)
    pickle.dumps(resolved.drivetrain)
    round_trip = pickle.loads(pickle.dumps(resolved))
    # ... and the unpickled copy derives its own value, correctly.
    wm2 = build_wheel_meta(round_trip, DEV, FDT)
    assert _row(round_trip.steering.per_wheel_steer(ins, 1, wm2, DEV, FDT)) == G1_BASE


def test_a_deepcopied_config_still_derives_its_own_value():
    import copy
    resolved, wm = _built(car_4w_rwd_ackermann)
    ins = AckermannInputs(throttle=0.0, brake=0.0, steer=0.5)
    resolved.steering.per_wheel_steer(ins, 1, wm, DEV, FDT)

    clone = copy.deepcopy(resolved.steering)
    clone.wheelbase = 3.0
    assert _row(clone.per_wheel_steer(ins, 1, wm, DEV, FDT)) == G1_WB_3
    assert _row(resolved.steering.per_wheel_steer(ins, 1, wm, DEV, FDT)) == G1_BASE


def _duck_metas(wm):
    """``wheel_meta`` is a documented DUCK TYPE — ``_WheelMetaProto``
    (``strategies/steering.py:25-31``) asks for five fields and nothing else.
    These three shapes satisfy it and are all refused by ``weakref.ref``."""
    import collections
    import types

    fields = dict(n_wheels=wm.n_wheels, positions=wm.positions,
                  axle_index=wm.axle_index, side_mask_L=wm.side_mask_L,
                  side_mask_R=wm.side_mask_R)
    Tup = collections.namedtuple("Tup", list(fields))

    class Slotted:
        __slots__ = tuple(fields)

        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    return [("SimpleNamespace", types.SimpleNamespace(**fields)),
            ("namedtuple", Tup(**fields)),
            ("__slots__ class", Slotted(**fields))]


def test_a_wheel_meta_that_cannot_be_weakly_referenced_still_works():
    """Holding the meta weakly must not become an undeclared requirement on the
    duck type: all three shapes below raise ``TypeError: cannot create weak
    reference to ...``. They are simply not cached — the pre-hoist cost, never
    a wrong value. ``prime_derived`` swallows build-time failures by design, so
    without this the user would meet it as "build succeeded, first step
    exploded"."""
    resolved, wm = _built(car_4w_rwd_ackermann)
    ins = AckermannInputs(throttle=0.0, brake=0.0, steer=0.5)
    for label, meta in _duck_metas(wm):
        got = _row(resolved.steering.per_wheel_steer(ins, 1, meta, DEV, FDT))
        assert got == G1_BASE, f"{label}: {got}"


def test_an_un_referenceable_meta_is_re_derived_every_call_not_mis_answered():
    """The uncached path must still be LIVE: two different such metas must not
    answer with each other's geometry."""
    resolved, wm = _built(car_4w_rwd_ackermann)
    ins = AckermannInputs(throttle=0.0, brake=0.0, steer=0.5)
    _label, meta = _duck_metas(wm)[0]
    assert _row(resolved.steering.per_wheel_steer(ins, 1, meta, DEV, FDT)) == G1_BASE

    resolved.steering.wheelbase = 3.0
    assert _row(resolved.steering.per_wheel_steer(ins, 1, meta, DEV, FDT)) == G1_WB_3


def test_the_drivetrain_also_accepts_an_un_referenceable_meta():
    resolved, wm = _built(car_4w_rwd_ackermann)
    ins = AckermannInputs(throttle=1.0, brake=1.0, steer=0.0)
    _label, meta = _duck_metas(wm)[0]
    T_drive, T_brake = resolved.drivetrain.distribute_torque(
        ins, torch.zeros(1, wm.n_wheels), meta, DEV, FDT)
    assert _row(T_drive) == [0.0, 0.0, 500.0, 500.0]
    assert _row(T_brake) == [750.0, 750.0, 500.0, 500.0]


def test_per_side_uses_the_same_derived_cap_and_no_longer_raises_on_none():
    """SCOPE NOTE, pinned because it is a real behaviour change the plan did
    not ask for. Item 5 was scoped to ``DrivetrainStrategy._rev_limit``
    (``drivetrain.py:48-52``); ``PerSide`` has its own per-side taper and was
    routed through the same derived cap for consistency.

    For any non-None cap that is bit-identical (asserted below). For
    ``omega_max_drive = None`` — which ``PerSide``'s constructor never produces
    but a caller can assign — the pre-change tree raised
    ``TypeError: unsupported operand type(s) for /: 'Tensor' and 'NoneType'``
    and the current one tapers not at all (the ``+inf`` identity). Arguably the
    fix, but it is a change, so it is written down and tested rather than left
    for a user to find."""
    from genesis_vehicle.strategies.drivetrain import PerSide

    ps = PerSide(t_drive_max=1000.0, t_brake_max=1000.0, omega_max_drive=100.0)
    omega_side = torch.tensor([50.0, -50.0, 0.0])
    T_des = torch.tensor([1000.0, -1000.0, 1000.0])
    capped = ps._cap_torque_batched(omega_side, T_des, ps._drive_omega_cap(),
                                    ps.t_drive_max)
    # bit-identical to passing the public attribute, as it did before
    assert torch.equal(
        capped,
        ps._cap_torque_batched(omega_side, T_des, ps.omega_max_drive,
                               ps.t_drive_max))
    assert float(capped[0]) == pytest.approx(500.0)

    ps.omega_max_drive = None
    assert ps._drive_omega_cap() == float("inf")
    uncapped = ps._cap_torque_batched(omega_side, T_des, ps._drive_omega_cap(),
                                      ps.t_drive_max)
    assert torch.equal(uncapped, T_des)          # no taper, no TypeError
