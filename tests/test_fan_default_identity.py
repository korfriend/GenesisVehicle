"""The v1.6.0 swept-envelope work must not move the DEFAULT vehicle by one bit.

``wheel_contact="swept_envelope"`` is opt-in; every existing vehicle, preset,
sample and demo keeps the single-ray point contact. The proof is a real
200-step Genesis rollout of the reference car in the default configuration,
compared with ``torch.equal`` — bit-identical, not ``allclose`` — against wheel
distances captured from the tree BEFORE any of this landed.

Baseline capture
----------------
* commit ``37b8b7943f7000245e462e53090d2ce3b303dd70`` (``genesis_vehicle/``),
  plus the then-uncommitted v1.5.1 terrain-sample fix and the v1.5.2
  ray-miss/grounded work that were already in the working tree — i.e. the tree
  exactly as it stood at the START of the swept-envelope change, with none of it
  applied. (Capturing after the change would prove nothing.)
* genesis-world 1.4.0, CPU backend, ``n_envs=1``, dual_scene (the default),
  ``dt=0.025``, ``substeps=10``, car spawned at ``z=1.0`` on a friction-1.0
  ground plane, 60 steps of ``brake=1`` then 140 of ``throttle=0.5``.

Re-capture for genesis-world 1.4.5 (SDK v1.6.9)
-----------------------------------------------
The baselines are kept PER ENGINE VERSION (``BASELINES`` below); a version that
is not in the table skips, it does not compare against another engine's
numbers. The 1.4.5 entry was captured from the same pre-change commit
``37b8b79`` (a ``git archive`` of the commit itself; the uncommitted v1.5.1 /
v1.5.2 working-tree additions above are not in it), with genesis-world 1.4.5 /
quadrants 1.3.3 / trimesh 5.2.0, CPU backend, and the identical rollout. The
method was validated by running that same archive on genesis-world 1.4.0, where
it reproduced the 1.4.0 entry bit-for-bit. Running the then-current tree
(``54b3959``, v1.6.8) on 1.4.5 gave the same 1.4.5 numbers. The 1.4.0 entry is
kept so the test still guards a rollback to 1.4.0.

Platform scope (v1.6.10)
------------------------
A baseline is a float32 BIT PATTERN, and it is only valid on the platform it was
captured on. Every entry above was captured on Linux x86_64 (WSL2), CPU backend.
Another OS, CPU instruction set or compiler output can round differently: a field
run on Windows 11 differed from the 1.4.5 entry by about 0.00036 mm in the wheel
distances (~12 float32 ULP at 0.41 m after 200 steps). That is not a change of
the default contact path. The baselines are therefore keyed by
(genesis version, platform) — ``platform`` being ``f"{sys.platform}-{platform.machine()}"``
— and an uncaptured pair SKIPS, it does not fail. Through v1.6.9 this file keyed
by version only and compared on every platform, and this docstring did not say
the pattern was platform-bound; that was wrong. A matching key is still not a
guarantee across CPUs: two Linux x86_64 machines with different instruction-set
support may differ too, which calls for a re-capture on that machine, not a
looser comparison.

A failure here, on a platform whose baseline was captured there, means the
default contact path changed. It is NOT a tolerance knob: loosen it and the
mandate this file exists to enforce is gone. If the Genesis version, the backend
or the platform changes, RE-CAPTURE against the pre-change tree (``git archive``
of ``BASELINE_COMMIT``, the rollout below) and say so in the CHANGELOG.

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
#: after step 199}. Platform key: see ``platform_key()``.
BASELINES = {
    ("1.4.0", "linux-x86_64"): {
        49:  [0.41489899158477783, 0.41489899158477783,
              0.41740989685058594, 0.41740989685058594],
        99:  [0.411739706993103,   0.41198277473449707,
              0.4113011360168457,  0.41154420375823975],
        149: [0.41162025928497314, 0.41162025928497314,
              0.41120994091033936, 0.41121017932891846],
        199: [0.41160452365875244, 0.41160428524017334,
              0.41122496128082275, 0.41122472286224365],
        "final_pos": (5.92042875289917, 0.03004419431090355, 0.1114146038889885),
    },
    ("1.4.5", "linux-x86_64"): {
        49:  [0.4148988723754883,  0.4148988723754883,
              0.41740942001342773, 0.4174095392227173],
        99:  [0.4119824171066284,  0.4117391109466553,
              0.411543607711792,   0.4113004207611084],
        149: [0.4116201400756836,  0.4116201400756836,
              0.4112095832824707,  0.41120970249176025],
        199: [0.4116039276123047,  0.4116036891937256,
              0.41122496128082275, 0.4112248420715332],
        "final_pos": (5.920430660247803, -0.030101105570793152, 0.11141432076692581),
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


def test_baseline_lookup_is_exact_on_version_and_platform():
    assert baseline_for("1.4.5", "linux-x86_64") is BASELINES[("1.4.5", "linux-x86_64")]
    assert baseline_for("1.4.5", "win32-AMD64") is None       # the field report's platform
    assert baseline_for("1.4.4", "linux-x86_64") is None
    assert all(isinstance(k, tuple) and len(k) == 2 for k in BASELINES)


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

    cfg = car_4w_rwd_ackermann(URDF, stability="control")
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

    for i in range(200):
        if i < 60:
            veh.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
        else:
            veh.set_inputs(throttle=0.5, brake=0.0, steer=0.0)
        vs.step()
        want = baseline.get(i)
        if want is None:
            continue
        got = veh.distances[0].cpu()
        expect = torch.tensor(want, dtype=got.dtype)
        assert torch.equal(got, expect), (
            f"step {i}: the DEFAULT point-contact path changed.\n"
            f"  baseline ({BASELINE_COMMIT[:12]}, genesis {version}, {plat}): {want}\n"
            f"  now:                              {got.tolist()}")

    pos = veh.get_pos()[0].cpu()
    assert torch.equal(pos, torch.tensor(baseline["final_pos"], dtype=pos.dtype))


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
