"""UE ``wheelOverrides`` rolling resistance and Pacejka keys reach the physics (v1.7.0).

Through v1.6.10 ``server/vehicle_builder.build_cfg`` wrote the UE keys
``rollingResistance`` and ``pbX`` / ``pcX`` / ``peX`` / ``pbY`` / ``pcY`` /
``peY`` to ``w.rolling_resistance`` and ``w.bx`` .. ``w.ey``. ``WheelConfig``
names those fields ``rolling_resistance_cr`` and ``pb_x`` .. ``pe_y`` and has no
``__slots__``, so each write created an attribute nothing reads and the value
was silently dropped — while ``docs/server.md`` said the keys worked.

These tests drive the real chain ``build_cfg -> resolve -> build_wheel_meta``
on BOTH branches ``build_cfg`` has (a preset, and the generic mapping branch),
with override values that differ from the preset's values AND from the
``.get`` fallbacks in ``build_cfg`` (0.015 / 10.0 / 1.65 / 0.4 / 8.0 / 1.30 /
0.4), so neither an unapplied override nor a fallback can pass by accident.
"""
import math
import os

import pytest
import torch

pytest.importorskip("genesis")

from genesis_vehicle.config import resolve
from genesis_vehicle.core import build_wheel_meta
from genesis_vehicle.server import vehicle_builder

URDF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "samples", "urdf", "car_4w.urdf"))

#: UE key -> (WheelMeta field, override value).
OVERRIDES = {
    "rollingResistance": ("rolling_resistance_cr", 0.031),
    "pbX": ("pb_x", 11.5),
    "pcX": ("pc_x", 1.55),
    "peX": ("pe_x", 0.25),
    "pbY": ("pb_y", 9.25),
    "pcY": ("pc_y", 1.45),
    "peY": ("pe_y", 0.15),
}

#: driveType 0 + 4 wheels -> car_4w_rwd_ackermann preset; 3 (Manual) -> the
#: generic mapping branch.
BRANCHES = {"preset": 0, "generic": 3}


def _wheel_meta(mapping):
    cfg = vehicle_builder.build_cfg(URDF, mapping, 1.0, target_id=0)
    return build_wheel_meta(resolve(cfg), torch.device("cpu"), torch.float32)


@pytest.fixture(autouse=True)
def _fresh_cfg_cache(monkeypatch):
    # build_cfg returns a cached cfg for an identical (urdf, mapping, friction).
    monkeypatch.setattr(vehicle_builder, "_cfg_cache", {})


@pytest.mark.parametrize("branch", sorted(BRANCHES))
def test_wheel_overrides_reach_the_resolved_wheel_meta(branch):
    mapping = {"driveType": BRANCHES[branch]}
    base = _wheel_meta(dict(mapping))
    entry = {"wheelName": "*"}
    entry.update({k: v for k, (_, v) in OVERRIDES.items()})
    wm = _wheel_meta(dict(mapping, wheelOverrides=[entry]))
    for key, (field, value) in OVERRIDES.items():
        before = getattr(base, field)
        got = getattr(wm, field)
        assert not torch.allclose(before, torch.full_like(before, value)), (
            f"[{branch}] {key}: the override value {value} equals the "
            f"un-overridden {field} — the test would pass vacuously")
        assert torch.allclose(got, torch.full_like(got, value)), (
            f"[{branch}] UE override {key}={value} did not reach "
            f"WheelMeta.{field}: {got.tolist()} (without override "
            f"{before.tolist()})")


@pytest.mark.parametrize("branch", sorted(BRANCHES))
@pytest.mark.parametrize("key, bad", [
    ("pbX", 0.0), ("pcY", -1.0), ("pbY", float("nan")),
    ("rollingResistance", -0.01), ("peX", float("inf")),
])
def test_an_invalid_override_is_rejected_and_the_value_kept(branch, key, bad, capsys):
    """A UE placeholder (0 for a Pacejka B/C, a negative cr, a non-finite
    value) is refused like the neighbouring ``muLong`` / ``muLat`` overrides,
    instead of zeroing the tyre force or propelling the vehicle."""
    mapping = {"driveType": BRANCHES[branch]}
    field = OVERRIDES[key][0]
    base = getattr(_wheel_meta(dict(mapping)), field)
    wm = _wheel_meta(dict(mapping, wheelOverrides=[{"wheelName": "*", key: bad}]))
    assert torch.equal(getattr(wm, field), base)
    assert "rejected" in capsys.readouterr().out


def test_the_setter_refuses_a_name_wheel_config_does_not_have():
    from genesis_vehicle.config import WheelConfig
    w = WheelConfig()
    vehicle_builder._set_wheel_field(w, "rolling_resistance_cr", 0.02)
    assert w.rolling_resistance_cr == 0.02
    for dead in ("rolling_resistance", "bx", "cx", "ex", "by", "cy", "ey", "inertia"):
        with pytest.raises(AttributeError, match="no field"):
            vehicle_builder._set_wheel_field(w, dead, 1.0)
        assert not hasattr(w, dead)


def test_every_live_override_targets_a_wheel_config_field():
    from dataclasses import fields
    from genesis_vehicle.config import WheelConfig
    names = {f.name for f in fields(WheelConfig)}
    for _, field, _, _ in vehicle_builder._LIVE_TIRE_OVERRIDES:
        assert field in names
    assert not any(math.isnan(v) for _, v in OVERRIDES.values())
