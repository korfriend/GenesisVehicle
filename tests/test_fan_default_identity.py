"""The v1.6.0 swept-envelope work must not move the default CONTACT PATH by one bit.

``wheel_contact="swept_envelope"`` is opt-in; every existing vehicle, preset,
sample and demo keeps the single-ray point contact. The proof is a real
200-step Genesis rollout of the reference car on the default contact path,
compared with ``torch.equal`` — bit-identical, not ``allclose`` — against wheel
distances captured from the tree BEFORE any of this landed.

"Contact path" means the reference car's default configuration,
``car_4w_rwd_ackermann(stability="control")``, with ``RollingResistance``
removed from its hook list (see "Re-anchoring for v1.7.0" below). Through
v1.6.10 this file rolled out the unmodified default and its first line said the
DEFAULT VEHICLE could not move by one bit; since v1.7.0 that is no longer true
— the rolling-resistance fix moves the default vehicle on purpose — so the claim
is now the contact path, not the whole default vehicle.

Baseline capture
----------------
* commit ``37b8b7943f7000245e462e53090d2ce3b303dd70`` (``genesis_vehicle/``) —
  the tree as it stood at the START of the swept-envelope change, with none of
  it applied. (Capturing after the change would prove nothing.)
* CPU backend, ``n_envs=1``, dual_scene (the default), the default
  ``solver="batched"``, ``dt=0.025``, ``substeps=10``, car spawned at ``z=1.0``
  on a friction-1.0 ground plane, 60 steps of ``brake=1`` then 140 of
  ``throttle=0.5``.

Re-anchoring for v1.7.0
------------------------
v1.7.0 made ``RollingResistance`` an axle resisting torque; through v1.6.10 it
had no effect (see the ``RollingResistance`` docstring). That fix moves the
default rollout on purpose: with the unmodified "control" profile the final
chassis x went from 5.920430660247803 to 5.224289894104004 m (-0.70 m) on
genesis-world 1.4.5, linux-x86_64, CPU (with rolling resistance applied to ω
before the tyre; the intermediate 1.7.0 design that added it to the brake
inside the overshoot clamp ended at 5.230594158172607 m). This file's mandate is that the FAN
work did not move the contact path, so the rollout now runs the "control"
profile minus ``RollingResistance`` and the baseline was RE-CAPTURED for that
rollout from a ``git archive`` of the unmodified ``BASELINE_COMMIT`` (genesis-world
1.4.5 / quadrants 1.3.3 / trimesh 5.2.0, torch 2.11.0, CPU, linux-x86_64 WSL2),
with ``PYTHONPATH`` pointing at the archive. The same rollout on the v1.6.10
tree (``a71b2a6``) and on the v1.7.0 tree gave the identical bit pattern, and
the archive WITH ``RollingResistance`` still reproduced the previous 1.4.5
entry bit-for-bit, so the capture method is the one the earlier entries used.

What this costs: the rolling-resistance code path is no longer covered by this
file. It is covered by ``tests/test_rolling_resistance.py`` (an analytic
coast-down against ``sum(cr*N) / M_eff``, Genesis-free and in a CPU scene), and
``test_default_control_profile_rolling_resistance_is_live`` below fails if the
default profile's rolling resistance silently goes back to having no effect.

The 1.4.0 entry was DELETED in v1.7.0. genesis-world 1.4.0 has been below the
supported floor (1.4.5) since SDK v1.6.9, and the 1.4.0 overlay environment the
earlier entry was captured with no longer exists, so it could not be re-captured
for the contact-path rollout; on 1.4.0 the identity test now skips. (Through
v1.6.10 this docstring said the 1.4.0 entry was kept to guard a rollback; that
guard is gone.)

Platform scope (v1.6.10)
------------------------
A baseline is a float32 BIT PATTERN, and it is only valid on the platform it was
captured on. Every entry above was captured on Linux x86_64 (WSL2), CPU backend.
Another OS, CPU instruction set or compiler output can round differently: a field
run on Windows 11 differed from the (pre-v1.7.0) 1.4.5 entry by about
0.00036 mm in the wheel distances (~12 float32 ULP at 0.41 m after 200 steps).
That is not a change of the default contact path. The baselines are therefore
keyed by (genesis version, platform) — ``platform`` being
``f"{sys.platform}-{platform.machine()}"`` — and an uncaptured pair SKIPS, it
does not fail. Through v1.6.9 this file keyed by version only and compared on
every platform, and this docstring did not say the pattern was platform-bound;
that was wrong. A matching key is still not a guarantee across CPUs: two Linux
x86_64 machines with different instruction-set support may differ too, which
calls for a re-capture on that machine, not a looser comparison.

A failure here, on a platform whose baseline was captured there, means the
default contact path changed. It is NOT a tolerance knob: loosen it and the
mandate this file exists to enforce is gone. If the Genesis version, the backend
or the platform changes, RE-CAPTURE against the pre-change tree (``git archive``
of ``BASELINE_COMMIT``, the rollout below — "control" profile MINUS
``RollingResistance``, exactly as ``_contact_path_cfg`` builds it) and say so in
the CHANGELOG. A capture of the unmodified "control" profile is the wrong
baseline since v1.7.0.

The second half of the file drives the M > 1 branch through a stub sensor, so
this file also witnesses that the branch the rollout does NOT take is live.
"""

import os
import platform
import sys

import pytest
import torch

from genesis_vehicle.raycast import (
    read_distances, sensor_fan, set_sensor_fan, set_sensor_up_offset,
)

URDF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "samples", "urdf", "car_4w.urdf"))

BASELINE_COMMIT = "37b8b7943f7000245e462e53090d2ce3b303dd70"

#: (genesis-world version, platform key) -> {step index -> the four wheel
#: distances of the reference car (float32), "final_pos" -> the chassis position
#: after step 199}, for the CONTACT-PATH rollout ("control" minus
#: RollingResistance). Platform key: see ``platform_key()``.
BASELINES = {
    ("1.4.5", "linux-x86_64"): {
        49:  [0.4148988723754883,  0.4148988723754883,
              0.41740942001342773, 0.4174095392227173],
        99:  [0.4119826555252075,  0.4117391109466553,
              0.4115445613861084,  0.41130101680755615],
        149: [0.41162002086639404, 0.41161978244781494,
              0.4112098217010498,  0.4112098217010498],
        199: [0.41160404682159424, 0.41160404682159424,
              0.4112248420715332,  0.41122496128082275],
        "final_pos": (5.920469284057617, -0.030104611068964005, 0.11141446977853775),
    },
}


def platform_key() -> str:
    """``"linux-x86_64"``, ``"win32-AMD64"``, ``"darwin-arm64"`` ..."""
    return f"{sys.platform}-{platform.machine()}"


def baseline_for(version, plat):
    """The captured baseline for this (engine, platform), or None — never the
    nearest other entry: a bit pattern from another engine or platform proves
    nothing about this one."""
    return BASELINES.get((version, plat))


def _contact_path_cfg():
    """The rollout's configuration: the reference car's default "control"
    profile with ``RollingResistance`` removed (v1.7.0 re-anchor)."""
    from genesis_vehicle import RollingResistance, car_4w_rwd_ackermann
    cfg = car_4w_rwd_ackermann(URDF, stability="control")
    cfg.stability_hooks = [h for h in cfg.stability_hooks
                           if not isinstance(h, RollingResistance)]
    return cfg


def _drive(vs, vehicles, record=None):
    """The rollout's inputs: 60 steps of full brake, then 140 of throttle 0.5.
    ``record(i)`` is called after every step."""
    for i in range(200):
        for veh in vehicles:
            if i < 60:
                veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
            else:
                veh.set_inputs(throttle=0.5, brake=0.0, steer=0.0)
        vs.step()
        if record is not None:
            record(i)


def test_baseline_lookup_is_exact_on_version_and_platform():
    assert baseline_for("1.4.5", "linux-x86_64") is BASELINES[("1.4.5", "linux-x86_64")]
    assert baseline_for("1.4.5", "win32-AMD64") is None       # the field report's platform
    assert baseline_for("1.4.4", "linux-x86_64") is None
    assert baseline_for("1.4.0", "linux-x86_64") is None      # deleted in v1.7.0
    assert all(isinstance(k, tuple) and len(k) == 2 for k in BASELINES)


def test_contact_path_cfg_drops_only_rolling_resistance():
    from genesis_vehicle import car_4w_rwd_ackermann
    full = [type(h).__name__ for h in
            car_4w_rwd_ackermann(URDF, stability="control").stability_hooks]
    contact = [type(h).__name__ for h in _contact_path_cfg().stability_hooks]
    assert "RollingResistance" in full
    assert contact == [n for n in full if n != "RollingResistance"]


def test_default_rollout_is_bit_identical_to_the_pre_change_tree():
    genesis = pytest.importorskip("genesis")
    from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann

    version = getattr(genesis, "__version__", None)
    plat = platform_key()
    baseline = baseline_for(version, plat)
    if baseline is None:
        pytest.skip(f"no baseline captured for genesis-world {version!r} on "
                    f"{plat!r} (have {sorted(BASELINES)}); the baseline is a "
                    f"float32 bit pattern bound to the platform it was captured "
                    f"on — re-capture against the pre-change tree there")

    cfg = _contact_path_cfg()
    VehicleScene.init_backend("cpu")
    vs = VehicleScene(dt=cfg.recommended_dt, substeps=10, n_envs=1,
                      show_viewer=False)
    vs.add_ground_plane(friction=1.0)
    veh = vs.add_vehicle(URDF, car_4w_rwd_ackermann, cfg=cfg, pos=(0.0, 0.0, 1.0),
                         material=genesis.materials.Rigid(friction=1.0))
    vs.build()

    # The default must not have become a fan.
    assert sensor_fan(veh.sensor)[0] == 1
    assert veh.sensor.read().distances.dim() == 2      # (n_envs, n_wheels)

    def check(i):
        want = baseline.get(i)
        if want is None:
            return
        got = veh.distances[0].cpu()
        expect = torch.tensor(want, dtype=got.dtype)
        assert torch.equal(got, expect), (
            f"step {i}: the DEFAULT point-contact path changed.\n"
            f"  baseline ({BASELINE_COMMIT[:12]}, genesis {version}, {plat}): {want}\n"
            f"  now:                              {got.tolist()}")

    _drive(vs, [veh], record=check)
    pos = veh.get_pos()[0].cpu()
    assert torch.equal(pos, torch.tensor(baseline["final_pos"], dtype=pos.dtype))


@pytest.mark.parametrize("solver", ["batched", "per_vehicle"])
def test_default_control_profile_rolling_resistance_is_live(solver):
    """Guard for the re-anchor above: the identity rollout no longer covers
    ``RollingResistance``, so this checks — in ONE scene, against itself, not
    against ``BASELINES`` — that the full default profile ends measurably
    short of the contact path on the same rollout, in both solver modes.

    Measured on genesis-world 1.4.5, linux-x86_64, CPU, one car per scene:
    5.9204 m without rolling resistance, 5.2243 m with it (-0.6961 m, both
    solvers; the intermediate 1.7.0 design measured 5.2305 m, -0.6899 m). Through
    v1.6.10 the difference was 0.04 mm (the hook had no effect). The lower
    bound catches that regression; the upper bound catches a resistance
    counted twice (about -1.4 m)."""
    genesis = pytest.importorskip("genesis")
    from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann

    full_cfg = car_4w_rwd_ackermann(URDF, stability="control")
    contact_cfg = _contact_path_cfg()
    VehicleScene.init_backend("cpu")
    vs = VehicleScene(dt=full_cfg.recommended_dt, substeps=10, n_envs=1,
                      show_viewer=False, solver=solver)
    vs.add_ground_plane(friction=1.0)
    mat = genesis.materials.Rigid(friction=1.0)
    full = vs.add_vehicle(URDF, car_4w_rwd_ackermann, cfg=full_cfg,
                          pos=(0.0, 0.0, 1.0), material=mat, name="full")
    contact = vs.add_vehicle(URDF, car_4w_rwd_ackermann, cfg=contact_cfg,
                             pos=(0.0, 6.0, 1.0), material=mat, name="contact")
    vs.build()
    _drive(vs, [full, contact])

    x_full = float(full.get_pos()[0, 0])
    x_contact = float(contact.get_pos()[0, 0])
    dx = x_contact - x_full
    assert 0.3 < dx < 1.2, (
        f"[{solver}] rolling resistance shortened the rollout by {dx:.4f} m "
        f"(full {x_full:.4f} m, contact path {x_contact:.4f} m); expected "
        f"about 0.70 m")


# --- the branch the rollout does not take ------------------------------------

class _StubSensor:
    """A raycaster whose distances are supplied by the test. ``max_range`` /
    ``no_hit_value`` live on ``_options`` exactly as the Genesis raycaster's do,
    so the miss sentinel resolves through the normal path."""

    def __init__(self, distances, miss=20.0):
        self._d = torch.as_tensor(distances, dtype=torch.float32)
        self._options = type("O", (), dict(max_range=miss, no_hit_value=miss))()

    def read(self):
        return type("R", (), dict(distances=self._d))()


def test_the_fan_branch_is_live_on_a_rank_3_sensor():
    """A stub shaped like a real M=3 fan read (n_envs, n_wheels, M) collapses to
    (n_envs, n_wheels), so the branch the default rollout skips is exercised."""
    s = _StubSensor([[[1.30, 1.40, 1.50],
                      [1.30, 1.31, 1.32]]])            # (1, 2, 3)
    set_sensor_up_offset(s, 1.0)
    set_sensor_fan(s, [[0.0, 0.05, 0.20],
                       [0.0, 0.05, 0.20]])
    assert sensor_fan(s)[0] == 3
    d, hit = read_distances(s, 1, return_hit=True)
    assert d.shape == (1, 2)
    assert hit.shape == (1, 2)
    # min over (raw - up_offset + c): wheel 0 -> min(0.30, 0.45, 0.70) = 0.30
    assert d[0].tolist() == pytest.approx([0.30, 0.30])


def test_a_fan_sensor_never_leaks_its_M_axis_to_last_distances():
    """The shape contract the whole design rests on: nothing downstream of
    read_distances can tell a fan sensor from a point one, at any n_envs."""
    row = [[1.30, 1.40, 1.50], [1.30, 1.31, 1.32]]
    s = _StubSensor([row, row, row])                   # (3 envs, 2 wheels, 3)
    set_sensor_up_offset(s, 1.0)
    set_sensor_fan(s, [[0.0, 0.05, 0.20], [0.0, 0.05, 0.20]])
    assert read_distances(s, 3).shape == (3, 2)
