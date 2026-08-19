"""DifferentiablePlant — the autodiff inverse plant (v1.3.0).

Genesis-free: the plant's whole prediction path is `_pipeline.compute_wheel_step`
plus a planar integrator, both pure torch, so a fake state source standing in
for a built vehicle exercises everything that matters. What needs a running
simulator (that the prediction tracks the real vehicle) is covered by the
`path_follow_demo --plant jacobian` sample instead.
"""
import math
import os

import numpy as np
import pytest
import torch

from genesis_vehicle.config import resolve
from genesis_vehicle.control import (DifferentiablePlant, PathFollower,
                                     PlantMassProperties, SweepTable)
from genesis_vehicle.control.plant import is_state_source
from genesis_vehicle.core import build_wheel_meta
from genesis_vehicle.presets import car_4w_rwd_ackermann, tank_skid_belt
from genesis_vehicle.urdf import parse_urdf

DATA = os.path.join(os.path.dirname(__file__), "data")
CAR_URDF = os.path.join(DATA, "car_ref.urdf")
TANK_URDF = os.path.join(DATA, "tank_ref.urdf")


# --- a Genesis-free stand-in for a live vehicle -------------------------------

class _FakeSource:
    """Everything `DifferentiablePlant` reads off a live vehicle.

    Mirrors `_StateSource`'s surface exactly (the plant only ever touches these
    attributes), with the chassis parked level and every wheel evenly loaded.
    """

    #: Wheel overspeed of the default state, as a fraction of rolling speed.
    #: A driving vehicle always carries some slip, and EXACT rolling
    #: (kappa == 0) is a cusp of the force model: the Pacejka curve is at its
    #: steepest there and the v0.6.0 F_long overshoot clamp switches branch on
    #: it, so the derivative at that one point says nothing about the regime
    #: around it. Fixtures sit just off it.
    DEFAULT_SLIP = 0.02

    def __init__(self, cfg, *, v_long=4.0, v_lat=0.0, wz=0.0, dt=0.025,
                 pitch_deg=0.0, roll_deg=0.0, omega=None):
        self.cfg = cfg
        self.resolved = resolve(cfg)
        self.dev = torch.device("cpu")
        self.fdt = torch.float32
        self.wheel_meta = build_wheel_meta(self.resolved, self.dev, self.fdt)
        self.dt = float(dt)
        self.env_idx = 0
        self.row = 0
        self.n_envs = 1
        self.K = 1
        self.entity = None
        self.physics = self
        n = self.wheel_meta.n_wheels
        # Sit each wheel at half its rest stroke -> a sane, equal normal load.
        rest_d = self.wheel_meta.rest_d
        stroke = rest_d - self.wheel_meta.radius
        self._dist = (rest_d - 0.5 * stroke).reshape(1, n)
        self._comp = (0.5 * stroke).reshape(1, n)
        if omega is None:
            omega = (1.0 + self.DEFAULT_SLIP) * v_long / self.wheel_meta.radius
        self._omega = torch.as_tensor(omega, dtype=self.fdt).expand(1, n).clone()
        self.pre_loop_hooks = list(self.resolved.stability_hooks)
        self.post_tire_hooks = list(self.resolved.stability_hooks)
        self.pre_loop_hooks = [h for h in self.pre_loop_hooks
                               if hasattr(h, "apply_pre_loop")]
        self.post_tire_hooks = [h for h in self.post_tire_hooks
                                if hasattr(h, "apply_post_tire")]
        # Mass properties consistent with THIS urdf. A fixed constant would
        # give the 58 t reference tank a 1.5 t chassis, and its drive torque
        # (sized for the real mass) would then predict 70 m/s^2 accelerations
        # in a regime no tyre model is meaningful in.
        self.mass_kg = float(parse_urdf(cfg.urdf_path).total_mass)
        pos = self.wheel_meta.positions.numpy()
        length = float(pos[:, 0].max() - pos[:, 0].min()) * 1.3 + 1.0
        width = float(pos[:, 1].max() - pos[:, 1].min()) * 1.3 + 0.5
        self.izz_kgm2 = self.mass_kg * (length ** 2 + width ** 2) / 12.0

        p, r = math.radians(pitch_deg), math.radians(roll_deg)
        cp, sp, cr, sr = math.cos(p / 2), math.sin(p / 2), math.cos(r / 2), math.sin(r / 2)
        self._quat = np.array([cp * cr, cp * sr, sp * cr, -sp * sr])
        self._vel = np.array([v_long, v_lat, 0.0])
        self._ang = np.array([0.0, 0.0, wz])

    # -- the _StateSource surface --
    def chassis(self):
        return (np.zeros(3), self._quat, self._vel, self._ang)

    def omega(self):
        return self._omega.clone()

    def prev_compression(self):
        return self._comp.clone()

    def distances(self):
        return self._dist.clone()


_SRC_KEYS = ("v_long", "v_lat", "wz", "dt", "pitch_deg", "roll_deg", "omega")


def _plant(cfg, **kw):
    """A plant on a fake vehicle.

    `DifferentiablePlant` takes any object providing the state-source surface,
    so the fake goes in whole — no reaching into the constructor.
    """
    src = _FakeSource(cfg, **{k: v for k, v in kw.items() if k in _SRC_KEYS})
    plant_kw = {k: v for k, v in kw.items() if k not in _SRC_KEYS}
    plant_kw.setdefault("mass", src.mass_kg)
    plant_kw.setdefault("izz", src.izz_kgm2)
    plant_kw.setdefault("com", (0.0, 0.0, 0.5))
    return DifferentiablePlant(src, **plant_kw)


def test_the_fake_source_satisfies_the_documented_surface():
    # If this ever fails, DifferentiablePlant grew a state read that the fake
    # (and any non-Genesis state feed) does not provide.
    assert is_state_source(_FakeSource(car_4w_rwd_ackermann(CAR_URDF)))


# --- mass properties ----------------------------------------------------------

def test_planar_mass_matrix_matches_its_definition():
    mp = PlantMassProperties(1200.0, (0.1, -0.05, 0.5), 1800.0)
    m, cx, cy = 1200.0, 0.1, -0.05
    assert mp.m3[0, 0] == pytest.approx(m)
    assert mp.m3[1, 1] == pytest.approx(m)
    assert mp.m3[2, 2] == pytest.approx(1800.0)
    assert mp.m3[0, 2] == pytest.approx(-m * cy)
    assert mp.m3[1, 2] == pytest.approx(m * cx)
    assert mp.m3[0, 2] == pytest.approx(mp.m3[2, 0])     # symmetric
    assert np.allclose(mp.m3 @ mp.m3_inv, np.eye(3), atol=1e-9)


# --- the Jacobian -------------------------------------------------------------

def test_autodiff_jacobian_matches_finite_differences():
    """The whole premise: torch.autograd through the ray-wheel pipeline gives
    the same derivative a finite difference of the same unroll does."""
    p = _plant(car_4w_rwd_ackermann(CAR_URDF), v_long=5.0)
    u = (0.4, 0.2)
    snap = p._snapshot()
    J, y = p.jacobian(u[0], u[1], snap=snap)

    eps = 1e-3
    for c in (0, 1):
        up = list(u); dn = list(u)
        up[c] += eps; dn[c] -= eps
        yp = p.predict(up[0], up[1], snap=snap)
        ym = p.predict(dn[0], dn[1], snap=snap)
        for r in (0, 1):
            fd = (yp[r] - ym[r]) / (2 * eps)
            assert J[r, c] == pytest.approx(fd, rel=0.02, abs=1e-4), \
                f"J[{r}][{c}] autodiff {J[r, c]} vs finite difference {fd}"


def test_jacobian_signs_follow_the_iso_input_contract():
    """+throttle accelerates; +steer = RIGHT = negative yaw rate (ISO 8855)."""
    for cfg in (car_4w_rwd_ackermann(CAR_URDF), tank_skid_belt(TANK_URDF)):
        p = _plant(cfg, v_long=4.0)
        J, _ = p.jacobian(0.2, 0.1)
        assert J[0, 0] > 0.0, f"d(a)/d(throttle) must be positive, got {J[0, 0]}"
        assert J[1, 1] < 0.0, f"d(omega_z)/d(steer) must be negative, got {J[1, 1]}"


def test_horizon_below_two_is_rejected():
    # One step cannot show a throttle response at all: throttle moves wheel
    # omega, and only the NEXT step's slip turns that into force.
    with pytest.raises(ValueError, match="horizon"):
        _plant(car_4w_rwd_ackermann(CAR_URDF), horizon=1)


def test_yaw_authority_grows_with_the_horizon():
    """A longer unroll sees more of the yaw response, monotonically.

    Steering acts through the tyre's lateral force, which builds over the
    horizon rather than landing in one step — so the gain the plant reports is
    a statement about how far ahead it is looking, and the controller's
    aggressiveness follows the horizon directly.
    """
    p = _plant(car_4w_rwd_ackermann(CAR_URDF), v_long=4.0, horizon=2)
    gains = []
    for h in (2, 4, 8):
        p.horizon = h
        gains.append(float(p.jacobian(0.2, 0.0)[0][1, 1]))
    assert all(g < 0.0 for g in gains), gains
    assert abs(gains[0]) < abs(gains[1]) < abs(gains[2]), gains


# --- prediction ---------------------------------------------------------------

def test_prediction_is_batched_over_candidate_commands():
    p = _plant(car_4w_rwd_ackermann(CAR_URDF), v_long=4.0)
    snap = p._snapshot()
    thr = [0.0, 0.3, 0.6]
    a_b, w_b = p.predict(thr, [0.1, 0.1, 0.1], snap=snap)
    assert a_b.shape == (3,) and w_b.shape == (3,)
    for i, t in enumerate(thr):
        a_s, w_s = p.predict(t, 0.1, snap=snap)
        assert a_b[i] == pytest.approx(a_s, rel=1e-5, abs=1e-6)
        assert w_b[i] == pytest.approx(w_s, rel=1e-5, abs=1e-6)


def test_more_throttle_accelerates_harder():
    p = _plant(car_4w_rwd_ackermann(CAR_URDF), v_long=3.0)
    snap = p._snapshot()
    a, _ = p.predict([0.0, 0.5, 1.0], [0.0, 0.0, 0.0], snap=snap)
    assert a[0] < a[1] < a[2]


def test_slope_attitude_reaches_the_prediction_through_gravity():
    """The live attitude is what the sweep table's pitch/roll axes bought.

    ISO 8855 pitch is positive nose-DOWN, so +10 deg is a downhill: the same
    command has to accelerate harder, and -10 deg (uphill) harder still in the
    other direction. The gap should be near the in-plane gravity component,
    2 * g * sin(10 deg) = 3.4 m/s^2.
    """
    flat = _plant(car_4w_rwd_ackermann(CAR_URDF), v_long=3.0)
    down = _plant(car_4w_rwd_ackermann(CAR_URDF), v_long=3.0, pitch_deg=10.0)
    up = _plant(car_4w_rwd_ackermann(CAR_URDF), v_long=3.0, pitch_deg=-10.0)
    a_flat, _ = flat.predict(0.2, 0.0)
    a_down, _ = down.predict(0.2, 0.0)
    a_up, _ = up.predict(0.2, 0.0)
    assert a_up < a_flat < a_down
    assert (a_down - a_up) == pytest.approx(2 * 9.81 * math.sin(math.radians(10)),
                                            rel=0.15)


# --- inversion ----------------------------------------------------------------

def test_solve_hits_the_requested_response():
    """Targets are calibrated against what this vehicle can actually do at
    this state, so a failure means the INVERSION missed, not that the request
    was outside the car's authority."""
    p = _plant(car_4w_rwd_ackermann(CAR_URDF), v_long=5.0, newton_iters=4)
    snap = p._snapshot()
    a_reach, _ = p.predict(0.8, 0.0, snap=snap)
    _, w_reach = p.predict(0.0, 0.4, snap=snap)
    a_t, w_t = 0.5 * a_reach, 0.5 * w_reach
    assert a_t > 0.05 and w_t < -0.005, (a_t, w_t)     # a meaningful ask

    thr, steer = p.solve(5.0, a_t, w_t)
    a, w = p.predict(thr, steer)
    assert a == pytest.approx(a_t, abs=0.1)
    assert w == pytest.approx(w_t, abs=0.02)
    assert -1.0 <= thr <= 1.0 and -1.0 <= steer <= 1.0


def test_solve_respects_the_command_ranges():
    p = _plant(car_4w_rwd_ackermann(CAR_URDF), v_long=5.0,
               steer_range=(-0.3, 0.3), newton_iters=4)
    # Ask for a yaw rate far beyond anything the vehicle can produce.
    thr, steer = p.solve(5.0, 0.0, -5.0)
    assert -0.3 <= steer <= 0.3
    assert -1.0 <= thr <= 1.0


def test_a_dead_gradient_is_rescued_by_the_secant_probe():
    """A zero Jacobian reads as "no authority" and freezes the command, so a
    degenerate channel is re-estimated by a secant over a control-sized step.

    Raising ``probe_threshold`` declares every column dead, which is how the
    rescue path gets exercised without having to manufacture one of the
    plant's real dead zones (the F_long overshoot clamp).
    """
    p = _plant(car_4w_rwd_ackermann(CAR_URDF), v_long=4.0)
    J_auto, _ = p.jacobian(0.2, 0.1)
    assert J_auto[0, 0] > 0.0 and J_auto[1, 1] < 0.0

    p.probe_threshold = 1e9
    J_secant, _ = p.jacobian(0.2, 0.1)
    assert J_secant[0, 0] > 0.0 and J_secant[1, 1] < 0.0
    # A secant over 0.2 of command is a much coarser estimator than the
    # derivative — the yaw response saturates well inside that step — so the
    # contract is the sign and the order of magnitude, which is all the
    # inversion needs from a channel it had no derivative for.
    for r, c in ((0, 0), (1, 1)):
        ratio = J_secant[r, c] / J_auto[r, c]
        assert 0.25 < ratio < 4.0, f"J[{r}][{c}] secant/autograd = {ratio}"

    p.probe_delta = 0.0                    # rescue disabled -> autograd stands
    J_off, _ = p.jacobian(0.2, 0.1)
    assert np.allclose(J_off, J_auto)


def test_set_applied_moves_the_linearisation_point():
    p = _plant(car_4w_rwd_ackermann(CAR_URDF), v_long=4.0)
    p.set_applied(0.8, -0.2)
    assert p.last_u == (0.8, -0.2)
    J_at_applied, _ = p.jacobian()
    J_elsewhere, _ = p.jacobian(0.0, 0.0)
    assert not np.allclose(J_at_applied, J_elsewhere)


def test_reset_clears_the_warm_start():
    p = _plant(car_4w_rwd_ackermann(CAR_URDF), v_long=4.0)
    p.solve(4.0, 1.0, -0.1)
    assert p.last_u != (0.0, 0.0)
    p.reset()
    assert p.last_u == (0.0, 0.0)
    assert p.last_jacobian is None and p.last_response is None


# --- PathFollower integration -------------------------------------------------

_PATH = [(float(i), 0.0, 0.0, 2.0) for i in range(20)]


def test_path_follower_takes_a_plant_and_prefers_the_coupled_solve():
    p = _plant(car_4w_rwd_ackermann(CAR_URDF), v_long=4.0)
    f = PathFollower(_PATH, p)
    assert f.plant is p
    assert f.sweep is p                 # pre-v1.3.0 attribute name still works
    assert f._coupled_plant is True
    thr, steer, brake = f.step((0.0, 0.0), 0.0, 4.0, 0.0, 0.0)
    assert -1.0 <= thr <= 1.0 and abs(steer) <= f.steer_cap and brake in (0.0, 1.0)


def test_path_follower_forwards_plant_kwargs_when_it_builds_the_plant():
    # Documented in docs/path-following.md as one of the three ways to set a
    # plant option, and the only one that goes through PathFollower.
    src = _FakeSource(car_4w_rwd_ackermann(CAR_URDF))
    f = PathFollower(_PATH, src, plant_kwargs={
        "horizon": 6, "mass": src.mass_kg, "izz": src.izz_kgm2})
    assert isinstance(f.plant, DifferentiablePlant)
    assert f.plant.horizon == 6


def test_path_follower_tightens_the_plant_steer_range_to_its_own_cap():
    p = _plant(car_4w_rwd_ackermann(CAR_URDF), v_long=4.0)
    PathFollower(_PATH, p, steer_cap=0.35)
    assert p.steer_range == (-0.35, 0.35)


def test_path_follower_reports_the_applied_command_on_every_exit():
    p = _plant(car_4w_rwd_ackermann(CAR_URDF), v_long=4.0)
    f = PathFollower(_PATH, p)
    # Sitting on the final waypoint takes the DONE early-return branch.
    thr, steer, brake = f.step((19.0, 0.0), 0.0, 0.0, 0.0, 0.0)
    assert f.last_mode == "DONE" and brake == 1.0
    assert p.last_u == (thr, steer) == (0.0, 0.0)


def test_path_follower_still_accepts_a_sweep_table():
    csv = os.path.join(os.path.dirname(__file__), "..", "samples", "data",
                       "tank_sweep_signed.csv")
    if not os.path.exists(csv):
        pytest.skip("bundled sweep CSV not present")
    f = PathFollower(_PATH, SweepTable.load(csv))
    assert f._coupled_plant is False
    thr, steer, brake = f.step((0.0, 0.0), 0.0, 2.0, 0.0, 0.0)
    assert -1.0 <= thr <= 1.0


def test_path_follower_without_any_plant_is_an_error():
    with pytest.raises(ValueError, match="inverse plant"):
        PathFollower(_PATH)
    with pytest.raises(TypeError, match="plant"):
        PathFollower(_PATH, object())
