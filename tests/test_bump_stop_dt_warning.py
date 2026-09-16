"""The bump-stop dt-stability warning names the VEHICLE — G7a.

``(k_susp + k_bump) * dt^2 / m_share > 0.7`` means the suspension will chatter
or diverge. The warning that says so carried no identifier at all
(``core.py``'s message took ``ratio`` as its only argument), so a user running
a fleet was told a vehicle was mis-tuned without being told WHICH. This is the
gate for adding the name.

It is RED before the change: the pre-change message contains no vehicle name,
no URDF name and no slot index.

Scope, stated exactly, because the plan's G7b is NOT this test: one driver
carries ONE config today, so "per vehicle" here means "per config, named after
the vehicle it was registered as". Evaluating the ratio PER SLOT inside one
fused group — where ``sprung_mass`` comes from slot 0 and ``k_tot`` is the
group maximum — needs a fused group to exist, which is STEP 3. G7b lives there.

| abbr | meaning |
|---|---|
| k_susp / k_bump | suspension spring rate / bump-stop extra rate (N/m) |
| m_share | sprung mass carried per wheel (kg) |
| dt | simulation step duration (s) |
"""

from __future__ import annotations

import logging
import os

import pytest

CAR_URDF = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "samples", "urdf", "car_4w.urdf"))

#: The reference car: sprung_mass 1200 kg / 4 wheels = 300 kg per wheel,
#: k_susp 70 kN/m, dt 0.025 s. k_bump 600 kN/m puts the ratio at 1.396 (over
#: the 0.7 bound); the preset default k_bump=0 sits at 0.146 (under it).
K_BUMP_OVER = 600_000.0
RATIO_OVER = 1.40           # what the message must print, to 2 decimals


def _build_car(name: str, k_bump: float):
    genesis = pytest.importorskip("genesis")
    from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann

    cfg = car_4w_rwd_ackermann(CAR_URDF, stability="control")
    for w in cfg.wheels:
        w.k_bump = k_bump
    VehicleScene.init_backend("cpu")
    vs = VehicleScene(dt=cfg.recommended_dt, substeps=4, n_envs=1,
                      show_viewer=False)
    vs.add_ground_plane(friction=1.0)
    kw = {} if name is None else {"name": name}
    vs.add_vehicle(CAR_URDF, car_4w_rwd_ackermann, cfg=cfg,
                   pos=(0.0, 0.0, 0.8),
                   material=genesis.materials.Rigid(friction=1.0), **kw)
    vs.build()
    return vs


class _Capture(logging.Handler):
    """Records on the ``genesis_vehicle`` logger, captured directly.

    NOT pytest's ``caplog``: ``gs.init()`` installs Genesis's own logging
    configuration and pytest's ``LogCaptureHandler`` does not survive it (the
    warning below is visible on stderr while ``caplog.records`` stays empty).
    Attaching to the logger itself is independent of what the engine does to
    the root logger, and of whether some earlier test already ran ``gs.init``.
    """

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.msgs: list[str] = []

    def emit(self, record):
        if "bump-stop is too stiff" in record.getMessage():
            self.msgs.append(record.getMessage())

    def __enter__(self):
        logging.getLogger("genesis_vehicle").addHandler(self)
        return self

    def __exit__(self, *exc):
        logging.getLogger("genesis_vehicle").removeHandler(self)
        return False


def test_g7a_the_warning_names_the_vehicle():
    with _Capture() as cap:
        _build_car("stiff_car", K_BUMP_OVER)

    assert len(cap.msgs) == 1, f"expected exactly one warning, got {cap.msgs}"
    assert "stiff_car" in cap.msgs[0], (
        "the dt-stability warning does not say WHICH vehicle is mis-tuned: "
        + cap.msgs[0])
    assert f"{RATIO_OVER:.2f}" in cap.msgs[0], cap.msgs[0]


def test_g7a_a_vehicle_inside_the_bound_is_not_warned_about():
    """The trigger, not just the string: the preset default (ratio 0.146) must
    stay silent, or the test above would pass on a warning that always fires."""
    with _Capture() as cap:
        _build_car("calm_car", 0.0)
    assert cap.msgs == []


def test_the_driver_carries_the_vehicle_name_it_was_registered_under():
    """``VehiclePhysics.vehicle_name`` is new public surface (it is what the
    warning above prints). ``VehicleScene`` feeds it the ``Vehicle.name`` the
    user passed to ``add_vehicle``, or the auto-generated ``vehicle_N``.

    NOT covered here: the URDF-basename fallback for a caller who constructs
    ``VehiclePhysics`` / ``MultiVehiclePhysics`` directly and supplies no name
    (``scene_helpers.add_vehicle``'s documented flow). No scene path reaches
    it, so it is asserted as the expression it is and nothing more."""
    vs = _build_car("named_car", 0.0)
    proto = vs.physics.kinds[0]._proto
    assert proto.vehicle_name == "named_car" == vs.vehicles[0].name

    vs2 = _build_car(None, 0.0)          # auto-named by add_vehicle
    proto2 = vs2.physics.kinds[0]._proto
    assert proto2.vehicle_name == vs2.vehicles[0].name
    assert proto2.vehicle_name.startswith("vehicle_")
