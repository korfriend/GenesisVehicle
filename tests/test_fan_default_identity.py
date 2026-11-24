"""A regression lock on the default CONTACT PATH, plus the cross-tree witnesses.

What each test here proves (v2.0.5):

* ``test_contact_path_rollout_matches_the_2_0_4_tree_on_1_5_0``: the 200-step
  brake-then-throttle rollout below gives, bit for bit, what the v2.0.4 tree +
  the two-line ``skip_forward`` patch gave on genesis-world 1.5.0 when
  ``BASELINES`` was captured. So 2.0.5 reproduces the 2.0.4 tree's physics on
  1.5.0 on this rollout, and later releases are locked to it on exactly this
  engine version and platform. It is NOT evidence that 2.0.4 is right, and it
  says nothing across engines (1.4.5 <-> 1.5.0).
* ``test_witness_rollout_matches_the_2_0_4_tree_on_1_5_0`` (N24): a rollout on
  which no 1.8.0 mechanism acts (10 m/s, wheels rolling, no throttle / brake,
  no landing), captured the same way. Through v2.0.4 this test was the
  CROSS-TREE witness against the pre-fan tree ``BASELINE_COMMIT`` (37b8b79) on
  genesis-world 1.4.5; that statement cannot be re-checked on 1.5.0 (the old
  trees do not run their batched/rendering paths there), so since v2.0.5 it is
  a lock on the 2.0.4 tree's no-mechanism rollout on 1.5.0. Proves nothing
  about driving, braking, low speed, saturation or the fan.
* ``test_point_contact_equals_a_degenerate_fan_bit_for_bit`` (N6'): point
  contact == ``swept_envelope`` at M = 3 on the SAME tree. Proves the fan
  degenerates to the point contact; proves nothing across trees.
* ``test_default_control_profile_rolling_resistance_is_live``: the default
  profile's rolling resistance still takes effect (the lock rollout runs
  without it).

"Contact path" means the reference car's default configuration,
``car_4w_rwd_ackermann(stability="control")``, with ``RollingResistance``
removed from its hook list (``_contact_path_cfg``; the v1.7.0 re-anchor below).

Baseline capture (v2.0.5)
-------------------------
* Captured from the v2.0.4 tree + the two-line ``skip_forward`` patch on
  genesis-world 1.5.0 (linux-x86_64); 2.0.5 reproduces it bit for bit. A
  regression lock for later releases on exactly this engine version and
  platform. (2026-11-24: ``git archive 9e5c4d5`` with ``skip_forward=``
  deleted from the two ``set_base_links_pos`` / ``_quat`` calls in
  ``vehicle_scene.py``, nothing else; this file's own ``_contact_path_cfg`` /
  ``_drive`` / ``_witness`` from that tree; genesis-world 1.5.0 / quadrants
  1.4.0, torch 2.11.0, CPU backend, WSL2.)
* ``n_envs=1``, dual_scene (the default), the default ``solver="batched"``,
  ``dt=0.025``, ``substeps=10``, car spawned at ``z=1.0`` on a friction-1.0
  ground plane, 60 steps of ``brake=1`` then 140 of ``throttle=0.5``.
* Why not the 2.0.4 tree as is: on 1.5.0 its proxy sync silently takes a
  per-body fallback (equal to the batched path only to 1e-5), and its
  rendering paths do not run; the patch removes exactly that and nothing else.
* The 1.4.5 entries (``BASELINES``, captured from the 1.8.0 tree; and
  ``WITNESS_BASELINES``, captured from ``BASELINE_COMMIT``) were DELETED in
  v2.0.5: 1.4.5 is below the required floor 1.5.0. On 1.4.5 the contact path
  rollout ended at x 6.018499851226807 m; on 1.5.0 at x 6.018531322479248 m.

History of this rollout's baseline
----------------------------------
* v1.6.0 .. v1.7.0: captured from ``git archive`` of ``BASELINE_COMMIT``
  (37b8b79, the tree at the START of the swept-envelope change). The file's
  first line then read "The v1.6.0 swept-envelope work must not move the
  default CONTACT PATH by one bit", and that claim was what the test enforced.
* v1.7.0 re-anchor: v1.7.0 made ``RollingResistance`` an axle resisting
  torque (through v1.6.10 it had no effect), which moved the unmodified default
  rollout on purpose (final x 5.920430660247803 -> 5.224289894104004 m on
  genesis-world 1.4.5, linux-x86_64, CPU). The rollout therefore runs the
  "control" profile minus ``RollingResistance``, re-captured from the archive
  of ``BASELINE_COMMIT``; v1.6.10 (``a71b2a6``) and v1.7.0 gave the identical
  bit pattern (contact-path final x 5.920469284057617).
* v1.8.0: 1.8.0 changed the contact path on purpose (axle torques before the
  tyre, the Coulomb brake, the friction ellipse at the transmitted force, the
  low-speed deadbeats, the ``StaticFrictionLock`` rest anchor), so the rollout
  no longer reproduces the 37b8b79 capture (it first differs at step 49).
  ``BASELINES`` was re-captured from the 1.8.0 tree (above) and the test was
  renamed from ``test_default_rollout_is_bit_identical_to_the_pre_change_tree``
  -- a name that would now claim a cross-tree identity it no longer checks.
  The cross-tree claim moved to N24.

The 1.4.0 entry was DELETED in v1.7.0 (1.4.0 was then below the supported
floor 1.4.5); the 1.4.5 entries were deleted in v2.0.5 (above). A 1.5.x patch
release (1.5.1, ...) has no entry: the lock tests skip there until a capture
is made on it.

Platform scope (v1.6.10)
------------------------
A baseline is a float32 BIT PATTERN, and it is only valid on the platform it was
captured on. Every entry was captured on Linux x86_64 (WSL2), CPU backend.
Another OS, CPU instruction set or compiler output can round differently: a field
run on Windows 11 differed from the (pre-v1.7.0) 1.4.5 entry by about
0.00036 mm in the wheel distances (~12 float32 ULP at 0.41 m after 200 steps).
The baselines are therefore keyed by (genesis version, platform) --
``platform`` being ``f"{sys.platform}-{platform.machine()}"`` -- and an
uncaptured pair SKIPS, it does not fail. A matching key is still not a
guarantee across CPUs: two Linux x86_64 machines with different instruction-set
support may differ too, which calls for a re-capture on that machine, not a
looser comparison.

A failure of the lock test, on a platform whose baseline was captured there,
means the default contact path changed SINCE THE CAPTURE. It is NOT a
tolerance knob. If the change was intended, or the Genesis version, backend or
platform changed, RE-CAPTURE from the tree that is to be locked (this file's
rollout, "control" profile MINUS ``RollingResistance`` exactly as
``_contact_path_cfg`` builds it), say in the CHANGELOG why the path moved, and
update the "Baseline capture" section above. Re-capturing against
``BASELINE_COMMIT`` is no longer the method (that was the pre-1.8.0 mandate).

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

#: The pre-fan tree (start of the v1.6.0 swept-envelope change). Since v1.8.0
#: it is the capture source of ``WITNESS_BASELINES`` (N24) only, not of
#: ``BASELINES``.
BASELINE_COMMIT = "37b8b7943f7000245e462e53090d2ce3b303dd70"

#: (genesis-world version, platform key) -> {step index -> the four wheel
#: distances of the reference car (float32), "final_pos" -> the chassis position
#: after step 199}, for the CONTACT-PATH rollout ("control" minus
#: RollingResistance), captured from the v2.0.4 tree + the two-line
#: ``skip_forward`` patch on genesis-world 1.5.0 (module docstring, "Baseline
#: capture"). Platform key: see ``platform_key()``.
BASELINES = {
    # Captured 2026-11-24 (gs150 step 8, harness capture_locks.py outside the
    # repo, which imports this file's _contact_path_cfg / _drive from the
    # patched 2.0.4 tree); equal bit for bit to an independent run of the same
    # tree in the 2.0.5 AC-E determinism probe.
    ("1.5.0", "linux-x86_64"): {
        49:  [0.41484057903289795, 0.4148404598236084,
              0.4172234535217285, 0.4172234535217285],
        99:  [0.4116325378417969, 0.4116325378417969,
              0.4111868143081665, 0.41118693351745605],
        149: [0.41161608695983887, 0.4116159677505493,
              0.4111919403076172, 0.4111919403076172],
        199: [0.4116004705429077, 0.4116004705429077,
              0.411207914352417, 0.411207914352417],
        "final_pos": (6.018531322479248, -5.587245865967816e-08, 0.11140414327383041),
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
    assert baseline_for("1.5.0", "linux-x86_64") is BASELINES[("1.5.0", "linux-x86_64")]
    assert baseline_for("1.5.0", "win32-AMD64") is None       # another platform
    assert baseline_for("1.5.1", "linux-x86_64") is None      # an uncaptured 1.5.x patch
    assert baseline_for("1.4.5", "linux-x86_64") is None      # deleted in v2.0.5 (below the floor)
    assert baseline_for("1.4.0", "linux-x86_64") is None      # deleted in v1.7.0
    assert all(isinstance(k, tuple) and len(k) == 2 for k in BASELINES)
    assert sorted(BASELINES) == sorted(WITNESS_BASELINES) == [("1.5.0", "linux-x86_64")]


def test_contact_path_cfg_drops_only_rolling_resistance():
    from genesis_vehicle import car_4w_rwd_ackermann
    full = [type(h).__name__ for h in
            car_4w_rwd_ackermann(URDF, stability="control").stability_hooks]
    contact = [type(h).__name__ for h in _contact_path_cfg().stability_hooks]
    assert "RollingResistance" in full
    assert contact == [n for n in full if n != "RollingResistance"]


def test_contact_path_rollout_matches_the_2_0_4_tree_on_1_5_0():
    """Regression lock (see the module docstring): the contact-path rollout
    equals, bit for bit, its capture from the v2.0.4 tree + the two-line
    ``skip_forward`` patch on genesis-world 1.5.0 (linux-x86_64). 2.0.5
    reproduces it bit for bit; later releases are locked to it on exactly
    this engine version and platform."""
    genesis = pytest.importorskip("genesis")
    from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann

    version = getattr(genesis, "__version__", None)
    plat = platform_key()
    baseline = baseline_for(version, plat)
    if baseline is None:
        pytest.skip(f"no baseline captured for genesis-world {version!r} on "
                    f"{plat!r} (have {sorted(BASELINES)}); the baseline is a "
                    f"float32 bit pattern bound to the platform it was captured "
                    f"on — re-capture from the tree to be locked, there")

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
            f"step {i}: the default contact path changed since its capture "
            f"from the 2.0.4 tree on genesis-world 1.5.0.\n"
            f"  baseline (2.0.4 tree, genesis {version}, {plat}): {want}\n"
            f"  now:                                   {got.tolist()}")

    _drive(vs, [veh], record=check)
    pos = veh.get_pos()[0].cpu()
    assert torch.equal(pos, torch.tensor(baseline["final_pos"], dtype=pos.dtype))


@pytest.mark.parametrize("solver", ["batched", "per_vehicle"])
def test_default_control_profile_rolling_resistance_is_live(solver):
    """Guard for the re-anchor above: the identity rollout no longer covers
    ``RollingResistance``, so this checks — in ONE scene, against itself, not
    against ``BASELINES`` — that the full default profile ends measurably
    short of the contact path on the same rollout, in both solver modes.

    The values this test computes on the 1.8.0 tree (genesis-world 1.4.5,
    linux-x86_64, CPU, this test's two-car scene; identical in both solver
    modes): 6.0185 m without rolling resistance, 5.3498 m with it
    (-0.6687 m). Through v1.6.10 the difference was 0.04 mm (the hook had no
    effect). The lower bound catches that regression; the upper bound
    catches a resistance counted twice (about -1.4 m)."""
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
        f"about 0.67 m (1.8.0)")


# --- N24: the cross-tree witness (v1.8.0) --------------------------------------

#: (genesis-world version, platform key) -> steer -> {step -> the four wheel
#: distances, "final_pos"} of the WITNESS rollout (``_witness`` below).
#: Captured from the v2.0.4 tree + the two-line ``skip_forward`` patch on
#: genesis-world 1.5.0 (linux-x86_64); 2.0.5 reproduces it bit for bit. A
#: regression lock for later releases on exactly this engine version and
#: platform (2026-11-24, same capture as ``BASELINES``; reference car,
#: "control" minus ``RollingResistance``, per_vehicle, dt 0.025 x 10, spawned
#: at z 0.1114 at 10 m/s with the wheels rolling). Through v2.0.4 the entry was
#: ("1.4.5", "linux-x86_64"), captured from a ``git archive`` of
#: ``BASELINE_COMMIT`` (37b8b79) and reproduced bit for bit by the 1.7.0 and
#: 1.8.0 .. 2.0.4 trees; deleted in v2.0.5 (below the floor).
WITNESS_BASELINES = {
    ("1.5.0", "linux-x86_64"): {
        0.0: {
            9: [0.4093639850616455, 0.4093637466430664, 0.4105031490325928, 0.4105030298233032],
            49: [0.41104423999786377, 0.41104400157928467, 0.4118410348892212, 0.41184091567993164],
            99: [0.41100239753723145, 0.41100239753723145, 0.4118032455444336, 0.4118032455444336],
            149: [0.411002516746521, 0.41100239753723145, 0.41180288791656494, 0.4118027687072754],
            199: [0.4110032320022583, 0.41100335121154785, 0.4118025302886963, 0.41180264949798584],
            "final_pos": (49.64012908935547, -4.90921721052473e-08, 0.11140291392803192),
        },
        0.05: {
            9: [0.4082777500152588, 0.41092801094055176, 0.40942180156707764, 0.4120720624923706],
            49: [0.4096797704696655, 0.41239941120147705, 0.41050171852111816, 0.41322147846221924],
            99: [0.4096475839614868, 0.41235601902008057, 0.4104726314544678, 0.4131814241409302],
            149: [0.4096618890762329, 0.4123421907424927, 0.4104856252670288, 0.4131661653518677],
            199: [0.40967607498168945, 0.41232848167419434, 0.41049885749816895, 0.4131511449813843],
            "final_pos": (46.26096725463867, -14.972697257995605, 0.11141368001699448),
        },
    },
}


def _witness(steer, record):
    """Reference car, "control" profile minus ``RollingResistance``,
    ``solver="per_vehicle"``, dual_scene, dt 0.025 x 10 substeps, plane μ 1.0,
    spawned at the settled height z 0.1114 with chassis vx 10 m/s and every
    wheel rolling (ω = 10/r) BEFORE the first step, then 200 steps at
    ``steer``, no throttle, no brake."""
    genesis = pytest.importorskip("genesis")
    from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann
    cfg = _contact_path_cfg()
    VehicleScene.init_backend("cpu")
    vs = VehicleScene(dt=cfg.recommended_dt, substeps=10, n_envs=1,
                      show_viewer=False, solver="per_vehicle")
    vs.add_ground_plane(friction=1.0)
    veh = vs.add_vehicle(URDF, car_4w_rwd_ackermann, cfg=cfg,
                         pos=(0.0, 0.0, 0.1114),
                         material=genesis.materials.Rigid(friction=1.0))
    vs.build()
    ent = veh.entity_main
    dv = torch.zeros(1, ent.n_dofs, device=genesis.device, dtype=genesis.tc_float)
    dv[0, 0] = 10.0
    ent.set_dofs_velocity(dv)
    veh.physics.omega[:, :] = 10.0 / veh.physics.wheel_meta.radius
    for i in range(200):
        veh.set_inputs(throttle=0.0, brake=0.0, steer=steer)
        vs.step()
        record(i, veh)
    return veh


@pytest.mark.parametrize("steer", [0.0, 0.05])
def test_witness_rollout_matches_the_2_0_4_tree_on_1_5_0(steer):
    """N24 — the no-mechanism witness. A rollout on which NO 1.8.0 mechanism
    acts: no axle torque (no throttle, no brake; rolling resistance removed),
    the chassis at 10 m/s with the wheels rolling from the first step (so the
    friction ellipse never binds on the raw slip), and no low-speed contact
    (the lateral / held-wheel deadbeats only bind below a few m/s or under
    wheel-load spikes; the car is spawned at its settled height, so there is
    no landing). Compared with ``torch.equal`` against the capture from the
    v2.0.4 tree + the two-line ``skip_forward`` patch on genesis-world 1.5.0.

    What it proves: on genesis-world 1.5.0 / linux-x86_64, the raycast, the
    suspension normal force, the rolling tyre force and the force-application
    path give the 2.0.4 tree's bits where no 1.8.0 mechanism acts — on both
    the straight and a gentle turn. What it does NOT prove: anything about
    driving, braking, low speed, the friction ellipse at saturation, or the
    swept-envelope fan, nor anything across engines. Through v2.0.4 (on
    genesis-world 1.4.5) this compared against the PRE-FAN tree
    ``BASELINE_COMMIT`` and passed on the 1.7.0 tree too; that cross-tree
    statement is not re-checked on 1.5.0."""
    import genesis
    version = getattr(genesis, "__version__", None)
    plat = platform_key()
    base = WITNESS_BASELINES.get((version, plat))
    if base is None:
        pytest.skip(f"no witness captured for genesis-world {version!r} on {plat!r}")
    want = base[steer]

    def check(i, veh):
        if i in want:
            got = veh.distances[0].cpu()
            assert torch.equal(got, torch.tensor(want[i], dtype=got.dtype)), (
                f"step {i}: witness changed: {want[i]} -> {got.tolist()}")

    veh = _witness(steer, check)
    pos = veh.get_pos()[0].cpu()
    assert torch.equal(pos, torch.tensor(want["final_pos"], dtype=pos.dtype))


# --- N6': the point contact is a degenerate fan, on the same tree -------------

def test_point_contact_equals_a_degenerate_fan_bit_for_bit():
    """N6' — the identity rollout (``_drive``: 60 brake + 140 throttle-0.5
    steps, the contact-path config, batched, dual_scene, dt 0.025 x 10, z 1.0)
    with ``wheel_contact="point"`` and with ``wheel_contact="swept_envelope",
    contact_samples=3`` at the default span (whose off-centre samples sit at
    ``|s| = r``, where ``c = r``, so the envelope reduces to the centre ray):
    every recorded distance and the final pose are bit-identical.

    What it proves: the fan degenerates to the point contact on the SAME tree.
    What it does NOT prove: anything across trees — a change to shared code
    moves both rollouts and still passes. Passes on the 1.7.0 tree too, BY
    DESIGN. Measured on the 1.8.0 tree: both end at x 6.018499851226807."""
    genesis = pytest.importorskip("genesis")
    from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann
    res = {}
    for mode in ("point", "swept_envelope"):
        cfg = _contact_path_cfg()
        VehicleScene.init_backend("cpu")
        vs = VehicleScene(dt=cfg.recommended_dt, substeps=10, n_envs=1,
                          show_viewer=False)
        vs.add_ground_plane(friction=1.0)
        kw = (dict(wheel_contact=mode) if mode == "point"
              else dict(wheel_contact=mode, contact_samples=3))
        veh = vs.add_vehicle(URDF, car_4w_rwd_ackermann, cfg=cfg, pos=(0.0, 0.0, 1.0),
                             material=genesis.materials.Rigid(friction=1.0), **kw)
        vs.build()
        rec = {}
        _drive(vs, [veh], record=lambda i: rec.__setitem__(i, veh.distances[0].cpu().clone()))
        rec["final_pos"] = veh.get_pos()[0].cpu().clone()
        res[mode] = rec
    for k in res["point"]:
        assert torch.equal(res["point"][k], res["swept_envelope"][k]), k


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
