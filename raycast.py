"""WheelRayPattern: a single canonical Genesis RaycastPattern for ray-wheels.

Replaces the legacy per-variant raycast implementations.
Each ray starts ``RAY_UP_OFFSET`` above a chassis-local wheel position and
points down (-z body); :func:`read_distances` subtracts the offset back out
so consumers see attachment-relative distances (the "high-cast" scheme —
see the constant's docstring for why).

Whether a ray hit anything at all is decided against the sensor's own miss
sentinel (:func:`ray_miss_value`), never against a hardcoded threshold. There
are exactly TWO predicates, one per distance convention, and which one you
want depends on whether the high-cast offset has been taken off yet:

* :func:`is_ray_hit` — RAW sensor distances. Used once, inside
  :func:`read_distances`, whose ``return_hit=True`` mask is the authoritative
  one; carry that mask to consumers rather than re-deriving it.
* :func:`is_ray_hit_corrected` — OFFSET-CORRECTED distances (everything the
  SDK stores in ``last_distances`` and hands to the visual mirrors). It cannot
  see the unpopulated-sensor zero, because the correction has already moved it;
  ``VehiclePhysics._stepped_once`` is the guard for that case.

The miss sentinel itself is validated wherever it is OBTAINED
(:func:`ray_miss_value`), not only where it is stamped
(:func:`set_sensor_miss_value`) — a sentinel below ``max_range`` would be read
as a very close hit and launch the vehicle, and a non-finite one would make the
equality test above never fire, reporting every miss as a hit. Both tests live
in :func:`check_miss_supported`, the one function all three entry points
(stamp / read / ``VehiclePhysics.set_ray_miss_value``) go through.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable

import torch
import genesis as gs
from genesis.options.sensors.raycaster import RaycastPattern

from ._gs_compat import sensor_miss_value

_logger = logging.getLogger(__name__)


#: High-cast margin (m): rays start this far ABOVE the wheel attachment
#: point (chassis-local +z) and the read layer subtracts it back out, so
#: the reported distance stays attachment-relative. Why: with the origin AT
#: the attachment point, a hard impact that bottoms the suspension can sink
#: the chassis far enough that the origins go BELOW the ground — the rays
#: then miss, the air mask kills N, and the vehicle rests on its chassis
#: collision box forever (a stable "buried" equilibrium; v1.1.16 fix).
#: With the high-cast origin the ray still sees the ground from above in
#: that state, the distance goes small/NEGATIVE, compression maxes out, and
#: N pushes the vehicle back onto its wheels. Keep this below any overhead
#: structure you expect within the margin above the wheel (tunnel ceilings).
RAY_UP_OFFSET = 1.0

#: DEPRECATED since v1.5.2, removal targeted v1.7.0. Unexported fallback
#: threshold used only when the miss sentinel of a sensor cannot be determined
#: (a stub/foreign sensor exposing no ``_options``). It is pinned to the
#: raycaster's DEFAULT ``max_range`` of 20.0, so it silently misjudges any
#: sensor built with a different range — which is the whole reason
#: :func:`is_ray_hit` now tests the sensor's own sentinel instead. Do not add
#: new uses.
RAY_MISS_THRESHOLD = 19.9


#: Half-width of the swept-envelope ray fan, in WHEEL RADII: sample ``j`` sits
#: at body-longitudinal offset ``s_j`` spanning ``[-r*FAN_SPAN, +r*FAN_SPAN]``
#: around the wheel attachment point. 1.0 = the full contact patch a wheel of
#: radius ``r`` can reach, which is where the swept circle meets the wheel
#: centre height; smaller values sample only the lower part of the envelope and
#: therefore see a step later.
FAN_SPAN = 1.0


def fan_longitudinal_offsets(wheel_radii: Iterable[float], fan_samples: int,
                             fan_span: float = FAN_SPAN) -> list[list[float]]:
    """Per-wheel body-longitudinal (+X) sample offsets ``s_j``, shape
    ``(n_wheels, fan_samples)``.

    ``fan_samples`` uniform samples over ``[-r*fan_span, +r*fan_span]`` using
    EACH wheel's own radius. ``fan_samples`` must be 1 (the single-ray point
    contact — an empty offset list) or an ODD number ≥ 3; see
    :func:`fan_height_offsets` for why even counts are refused."""
    M = _check_fan_samples(fan_samples)
    radii = _check_wheel_radii(wheel_radii)
    if M == 1:
        return [[0.0] for _ in radii]
    span = float(fan_span)
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError(f"fan_span must be finite and > 0, got {fan_span!r}")
    return [[(2.0 * j / (M - 1) - 1.0) * r * span for j in range(M)]
            for r in radii]


def fan_height_offsets(wheel_radii: Iterable[float], fan_samples: int,
                       fan_span: float = FAN_SPAN) -> list[list[float]]:
    """Per-wheel swept-envelope height offsets ``c_j``, shape
    ``(n_wheels, fan_samples)``.

    ``c_j = r - sqrt(r^2 - s_j^2)`` is how much HIGHER than the wheel's lowest
    point the tire surface sits at longitudinal offset ``s_j`` — a discrete
    approximation of the swept-circle lower envelope, conservative because
    discretisation always sees true contact late (the true envelope is the
    continuum of all ``s``; M samples of it can only miss contacts, never invent
    them). :func:`read_distances` adds ``c_j`` back to sample ``j``'s distance
    before taking the minimum, so a sample that is only *nearly* on the ground
    is penalised by exactly the height its tire surface stands off the wheel's
    bottom.

    The real limitation of the model is that the rays are VERTICAL and therefore
    ignore terrain slope, NOT the size of ``c_j``: an intermediate ``s_j``
    winning the minimum with a smaller penalty than the outermost sample is the
    model working, not a leak.

    ``fan_samples`` must be ODD (or 1). An even count has no ``s_j = 0`` sample,
    so the guarantee ``d_eff <= d_center`` — the one thing that makes the
    envelope structurally unable to LOWER the ground and thus unable to cause a
    sinking regression — is lost, and even FLAT ground reads biased by the
    smallest penalty in the fan, ``min_j c_j``: at r=0.35 that is 0.020017 m at
    M=4 (its innermost samples sit at ``|s| = r/3``) and 0.350000 m at M=2
    (``|s| = r``), i.e. the ground reads that much too LOW, whereas any odd M
    has ``c = 0`` at ``s = 0`` and is exact to 0.000000. Asserted by
    ``tests/test_swept_envelope.py::
    test_the_even_M_flat_ground_bias_is_what_the_docstrings_quote``."""
    return [[r - math.sqrt(max(r * r - s * s, 0.0)) for s in s_row]
            for r, s_row in zip(_check_wheel_radii(wheel_radii),
                                fan_longitudinal_offsets(wheel_radii, fan_samples,
                                                         fan_span))]


def fan_ray_positions(wheel_positions: Iterable[tuple[float, float, float]],
                      wheel_radii: Iterable[float], fan_samples: int,
                      fan_span: float = FAN_SPAN
                      ) -> list[tuple[float, float, float]]:
    """Every ray's chassis-local ``(x, y, z)`` for a fan of ``fan_samples``
    rays per wheel — the wheel attachment points spread along body +X by
    :func:`fan_longitudinal_offsets` (``z`` is the ATTACHMENT plane; the
    high-cast lift is added by the pattern, not here).

    Hand this — not the wheel centres — to
    :func:`genesis_vehicle.urdf.self_collision_ceiling` /
    :func:`single_scene_up_offset`, which judge each ray's ``(x, y)`` against
    the vehicle's own collision boxes and take the MINIMUM gap: an outer sample
    can sit under a low overhang the wheel centre clears, and capping on the
    centre alone would leave those rays starting inside the body — the v1.1.16
    "the ray hits its own roof, reads maximum compression, and the vehicle
    launches" bug, per-sample. (For the reference car the answer happens to be
    the same either way, because its chassis AABB spans x∈[-2.15, 2.15] and so
    covers every fan origin uniformly. That is a property of that URDF, not of
    the function: add a low sill outboard of the reference car's front wheels
    and the centre-only cap measures 0.28 m, while an M=9 fan — whose outermost
    front sample reaches x = 1.35 + r = 1.708 m, under the sill — measures
    0.12 m. That URDF is committed as the ``_overhang_car`` fixture and both
    figures are asserted by ``tests/test_swept_envelope.py::
    test_the_overhang_car_really_distinguishes_the_two``.)"""
    positions = [tuple(float(v) for v in p) for p in wheel_positions]
    s_all = fan_longitudinal_offsets(wheel_radii, fan_samples, fan_span)
    if len(s_all) != len(positions):
        raise ValueError(
            f"fan_ray_positions: got {len(positions)} wheel positions but "
            f"{len(s_all)} wheel radii — one radius per wheel is required.")
    return [(x + s, y, z) for (x, y, z), s_row in zip(positions, s_all)
            for s in s_row]


def _check_fan_samples(fan_samples: int) -> int:
    M = int(fan_samples)
    if M < 1:
        raise ValueError(f"fan_samples must be >= 1, got {fan_samples!r}")
    if M > 1 and M % 2 == 0:
        raise ValueError(
            f"fan_samples must be ODD (or 1), got {fan_samples!r}. An even fan "
            f"has no centre sample, so the swept envelope loses its "
            f"d_eff <= d_center guarantee and reads flat ground biased "
            f"(-0.020017 m at M=4, -0.350000 m at M=2 for r=0.35, vs exactly "
            f"0.000000 for any odd M).")
    return M


def _check_wheel_radii(wheel_radii: Iterable[float]) -> list[float]:
    if wheel_radii is None:
        raise ValueError(
            "a swept-envelope wheel fan (fan_samples > 1) needs the per-wheel "
            "radii: the height offset c_j = r - sqrt(r^2 - s_j^2) approximates "
            "THAT wheel's swept circle, so a mixed-radius vehicle cannot share "
            "one scalar. Pass wheel_radii=[w.radius for w in resolved.wheels].")
    radii = [float(r) for r in wheel_radii]
    for r in radii:
        if not math.isfinite(r) or r <= 0.0:
            raise ValueError(f"wheel radii must be finite and > 0, got {radii!r}")
    return radii


class WheelRayPattern(RaycastPattern):
    """Downward rays (-z body) per wheel position (chassis-local).

    Rays are HIGH-CAST: each origin sits ``up_offset`` metres above the
    wheel attachment point (see :data:`RAY_UP_OFFSET`); pair with
    :func:`read_distances`, which subtracts the offset so consumers see
    attachment-relative distances (negative = ground above the attachment
    point, i.e. the chassis has sunk past it).

    ``fan_samples`` (M) selects the contact model, and the DEFAULT M=1 is the
    single-ray point contact this class has always been:

    * **M = 1** — one ray per wheel, ``return_shape == (n_wheels,)``. A
      zero-radius probe: it reads ground height as a step function, so crossing
      a vertical edge delivers the whole obstacle height of suspension
      compression in ONE ``dt``, and the damper term then dominates because the
      rate is unphysical. Measured by the committed step-crossing harness
      ``tests/test_swept_envelope.py::_cross_step`` — reference-car constants
      (r=0.358 m, K=70,000 N/m, C_comp=14,000 N·s/m, dt=0.025 s), a 0.130 m lip
      taken at 3.3 m/s: ``dcompression = 0.130`` m in one step,
      ``raw_rate = 5.200`` m/s, peak wheel N = 85,120.
    * **M ≥ 3, odd** — a swept-envelope fan, ``return_shape == (n_wheels, M)``.
      M rays spread along body +X over ``[-r*fan_span, +r*fan_span]`` (each
      wheel's OWN radius, hence ``wheel_radii``), each penalised by
      :func:`fan_height_offsets`' ``c_j``; :func:`read_distances` takes the
      minimum over the fan and hands the pipeline ONE distance per wheel, so
      nothing downstream ever sees the M axis. A wheel of radius ``r`` then
      begins to climb a step of height ``h`` about ``sqrt(2rh)`` before it,
      which is where a real tire touches it.

    **M is not an accuracy dial.** It is a discretisation count, and the peak
    contact force does not converge in it: it moves NON-monotonically and spans
    about 20% over M ∈ [9, 31] (the same ``_cross_step`` harness and conditions
    as above — M=9 → 49,978 N, M=15 → 44,670 N, M=31 → 41,745 N,
    M=101 → 43,455 N, against 85,120 N for the point contact; asserted by
    ``test_M_is_not_an_accuracy_dial`` as ratios and by
    ``test_the_absolute_peaks_quoted_in_the_WheelRayPattern_docstring`` as
    newtons). Pick M for cost, take the result as "no longer a step function"
    rather than as a converged force, and do not compare two runs at different
    M.

    ``fan_samples`` must be 1 or ODD — see :func:`fan_height_offsets`.

    ``wheel_radii`` is keyword-with-default and REQUIRED when M > 1 (there is no
    ``None`` fallback: the pattern is constructed from bare positions at every
    call site, so it has nothing to fall back TO, and a pattern-level scalar
    would be wrong for a mixed-radius vehicle). ``positions`` and ``up_offset``
    keep their positional signature — out-of-repo callers construct this
    positionally.

    .. versionchanged:: 1.6.0
       ``fan_samples`` / ``wheel_radii`` / ``fan_span`` added. M=1 (the default)
       is bit-identical to the previous behaviour."""

    def __init__(self, positions: Iterable[tuple[float, float, float]],
                 up_offset: float = RAY_UP_OFFSET,
                 *,
                 fan_samples: int = 1,
                 wheel_radii: "Iterable[float] | None" = None,
                 fan_span: float = FAN_SPAN):
        self._positions = [tuple(p) for p in positions]
        self.up_offset = float(up_offset)
        self.fan_samples = _check_fan_samples(fan_samples)
        self.fan_span = float(fan_span)
        if self.fan_samples == 1:
            self.wheel_radii = None
            self.fan_s = None
            self.fan_c = None
        else:
            self.wheel_radii = _check_wheel_radii(wheel_radii)
            if len(self.wheel_radii) != len(self._positions):
                raise ValueError(
                    f"WheelRayPattern: got {len(self._positions)} wheel "
                    f"positions but {len(self.wheel_radii)} radii — the "
                    f"swept-envelope offsets are per wheel.")
            self.fan_s = fan_longitudinal_offsets(
                self.wheel_radii, self.fan_samples, self.fan_span)
            self.fan_c = fan_height_offsets(
                self.wheel_radii, self.fan_samples, self.fan_span)
        super().__init__()

    @classmethod
    def from_config(cls, resolved: "object", up_offset: float = RAY_UP_OFFSET,
                    *, fan_samples: int = 1,
                    fan_span: float = FAN_SPAN) -> "WheelRayPattern":
        """Build from a ResolvedConfig — positions AND radii straight off
        ``resolved.wheels`` (avoids importing ResolvedConfig here to dodge a
        cycle). This is the only construction path that can supply the per-wheel
        radii a swept-envelope fan needs without the caller assembling them.

        For CALLERS who already hold a ``ResolvedConfig``. The SDK's own
        registration paths (``VehicleScene.add_vehicle`` /
        ``scene_helpers.make_wheel_raycaster``) deliberately do NOT use it: they
        must place the rays in ``parse_urdf`` order — the order
        ``Vehicle.wheel_positions``, the per-wheel radii and the pipeline's L1
        axis are all built in — whereas ``resolved.wheels`` is ordered by the
        user's wheel list and its ``position`` may be overridden per wheel. They
        take positions from the URDF and name-match the radii into that order
        instead (``vehicle_scene._wheel_radii``)."""
        return cls([w.position for w in resolved.wheels], up_offset,
                   fan_samples=fan_samples, fan_span=fan_span,
                   wheel_radii=([w.radius for w in resolved.wheels]
                                if int(fan_samples) > 1 else None))

    def ray_positions(self) -> list[tuple[float, float, float]]:
        """Every ray's chassis-local ``(x, y, z)`` in the ATTACHMENT plane (the
        high-cast lift excluded) — the wheel centres at M=1, the full fan
        otherwise. This is what :func:`single_scene_up_offset` must be capped
        against; see :func:`fan_ray_positions`."""
        if self.fan_samples == 1:
            return list(self._positions)
        return [(x + s, y, z) for (x, y, z), s_row in zip(self._positions,
                                                          self.fan_s)
                for s in s_row]

    def _get_return_shape(self) -> tuple[int, ...]:
        if self.fan_samples > 1:
            return (len(self._positions), self.fan_samples)
        return (len(self._positions),)

    def compute_ray_dirs(self) -> None:
        self._ray_dirs[:] = torch.tensor(
            [0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float
        )

    def compute_ray_starts(self) -> None:
        if self.fan_samples == 1:
            for i, (x, y, z) in enumerate(self._positions):
                self._ray_starts[i] = torch.tensor(
                    [x, y, z + self.up_offset], device=gs.device, dtype=gs.tc_float
                )
            return
        for i, (x, y, z) in enumerate(self._positions):
            for j, s in enumerate(self.fan_s[i]):
                self._ray_starts[i, j] = torch.tensor(
                    [x + s, y, z + self.up_offset],
                    device=gs.device, dtype=gs.tc_float
                )


def single_scene_up_offset(urdf_path: str, ray_positions,
                           name: str = "vehicle") -> float:
    """The high-cast offset a SINGLE-SCENE vehicle can safely use.

    In ``single_scene`` the wheel rays are cast in the SAME scene the chassis
    collides in, so an origin lifted the full :data:`RAY_UP_OFFSET` above the
    wheel attachment point can end up above the vehicle's own collision box —
    every ray then hits its own roof a few centimetres down, the pipeline reads
    that as maximum compression, and the vehicle launches (the reference car has
    only 0.30 m between the wheel attachment at z=0.30 and the chassis box
    bottom at z=0.60, so 1.0 m was 0.7 m too high). This caps the offset at the
    vehicle's own collision ceiling — see
    :func:`genesis_vehicle.urdf.self_collision_ceiling`.

    ``dual_scene`` raycasts a scene that holds no vehicle collision geometry, so
    it is unaffected and keeps the full offset.

    A URDF whose collision geometry cannot be resolved (an unreadable mesh, a
    ``package://`` URI) reports no ceiling and keeps the full offset, which is
    the pre-v1.5.0 behaviour; run that vehicle in ``dual_scene`` if its rays
    self-hit.

    ``ray_positions`` must be EVERY ray's chassis-local ``(x, y, z)``, not the
    wheel centres, when the vehicle uses a swept-envelope fan
    (``WheelRayPattern.ray_positions()`` / :func:`fan_ray_positions` give
    exactly that): the ceiling is a per-ray minimum, and an outer fan sample can
    sit under a low overhang the wheel centre clears."""
    from .urdf import self_collision_ceiling
    try:
        ceiling = self_collision_ceiling(urdf_path, ray_positions)
    except Exception:
        _logger.debug("could not measure the self-collision ceiling for %r",
                      name, exc_info=True)
        return RAY_UP_OFFSET
    if ceiling is None or ceiling >= RAY_UP_OFFSET:
        return RAY_UP_OFFSET
    _logger.info(
        "[genesis_vehicle:single-scene] %s: wheel-ray high-cast offset capped at "
        "%.3f m (default %.2f m) — the vehicle's own collision geometry starts "
        "%.3f m above the wheel attachment point and the rays are cast in the "
        "same scene it collides in. A deep suspension bottom-out recovers less "
        "readily than in dual_scene; use raycast_mode='dual_scene' (the default) "
        "if that matters.", name, ceiling, RAY_UP_OFFSET, ceiling)
    return ceiling


#: Attribute :func:`set_sensor_up_offset` stamps on a Genesis raycaster so
#: :func:`read_distances` can recover the offset its rays were built with.
_UP_OFFSET_ATTR = "_gv_up_offset"


def set_sensor_up_offset(sensor: "object", up_offset: float) -> None:
    """Record on the sensor which high-cast offset its rays were built with.

    The offset is per vehicle since v1.5.0 (``single_scene`` caps it at the
    vehicle's own collision ceiling), and :func:`read_distances` has to subtract
    back exactly what :class:`WheelRayPattern` added. Stamping the sensor keeps
    that pairing with the object it belongs to instead of threading a parameter
    through every call site."""
    setattr(sensor, _UP_OFFSET_ATTR, float(up_offset))


def sensor_up_offset(sensor: "object") -> float:
    """The high-cast offset of ``sensor``'s rays; :data:`RAY_UP_OFFSET` if it was
    never stamped (a sensor built outside :class:`VehicleScene`)."""
    return float(getattr(sensor, _UP_OFFSET_ATTR, RAY_UP_OFFSET))


#: Attributes :func:`set_sensor_fan` stamps on a Genesis raycaster: the sample
#: count M and the ``(n_wheels, M)`` height offsets ``c_j``.
_FAN_M_ATTR = "_gv_fan_m"
_FAN_C_ATTR = "_gv_fan_c"


def set_sensor_fan(sensor: "object", fan_c) -> None:
    """Record the swept-envelope fan geometry ``sensor``'s rays were built with:
    the ``(n_wheels, M)`` height offsets ``c_j`` (:func:`fan_height_offsets`),
    from which M is taken.

    Stamped for the same reason as :func:`set_sensor_up_offset` — the read layer
    has to undo exactly what the pattern did, and ``VehicleScene`` calls
    ``read_distances(veh.sensor, self.n_envs)`` with no other arguments, so
    every parameter of the ray geometry has to travel ON the sensor.

    Pass ``None`` (or M=1 geometry) for the single-ray point contact; the fan
    attributes are then cleared and the read layer takes the historical path.

    .. versionadded:: 1.6.0"""
    if fan_c is None:
        for attr in (_FAN_M_ATTR, _FAN_C_ATTR):
            if hasattr(sensor, attr):
                delattr(sensor, attr)
        return
    c = _as_gs_tensor(fan_c)
    if c.dim() != 2:
        raise ValueError(f"fan height offsets must be (n_wheels, M), got "
                         f"shape {tuple(c.shape)}")
    M = _check_fan_samples(int(c.shape[1]))
    if M == 1:
        set_sensor_fan(sensor, None)
        return
    setattr(sensor, _FAN_M_ATTR, M)
    setattr(sensor, _FAN_C_ATTR, c)


def sensor_fan(sensor: "object") -> "tuple[int, torch.Tensor | None]":
    """``(M, c)`` for ``sensor``: the swept-envelope sample count and its
    ``(n_wheels, M)`` height offsets, or ``(1, None)`` for the single-ray point
    contact (the default, and what every sensor built before v1.6.0 reports).

    Falls back to the sensor's own ``WheelRayPattern`` when nothing was stamped,
    so the documented "build your own ``gs.sensors.Raycaster`` and hand it to
    ``VehiclePhysics``" path cannot silently read a fan sensor's M axis as if it
    were the wheel axis.

    .. versionadded:: 1.6.0"""
    c = getattr(sensor, _FAN_C_ATTR, None)
    if c is not None:
        return int(getattr(sensor, _FAN_M_ATTR, c.shape[1])), c
    pattern = getattr(getattr(sensor, "_options", None), "pattern", None)
    fan_c = getattr(pattern, "fan_c", None)
    if fan_c is not None and int(getattr(pattern, "fan_samples", 1)) > 1:
        return int(pattern.fan_samples), _as_gs_tensor(fan_c)
    return 1, None


def _as_gs_tensor(x) -> "torch.Tensor":
    """``x`` as a float tensor on the Genesis device when there is one (the fan
    offsets are read every step and must not sit on the wrong device), on the
    torch default otherwise — a stub sensor in a test can be stamped before
    ``gs.init``. :func:`read_distances` converts and re-caches either way."""
    try:
        return torch.as_tensor(x, dtype=gs.tc_float, device=gs.device)
    except (AttributeError, RuntimeError, TypeError):
        _logger.debug("Genesis backend not initialised; keeping the fan "
                      "offsets on the torch default device", exc_info=True)
        return torch.as_tensor(x, dtype=torch.float32)


#: Attribute :func:`set_sensor_miss_value` stamps on a Genesis raycaster.
_MISS_ATTR = "_gv_ray_miss"


def set_sensor_miss_value(sensor: "object",
                          miss: "float | None" = None) -> "float | None":
    """Validate and record the miss sentinel ``sensor``'s rays report.

    ``miss`` defaults to the sensor's own resolved option
    (:func:`sensor_miss_value` — ``no_hit_value``, itself defaulting to
    ``max_range``). Stamping keeps the value with the object it belongs to, the
    same way :func:`set_sensor_up_offset` does for the high-cast offset, and
    lets a caller override it for a sensor the SDK did not build.

    Raises ``ValueError`` when the sentinel is not a usable distance at all
    (non-finite or ``<= 0``) or is BELOW the sensor's ``max_range`` — see
    :func:`check_miss_supported`, which is where both tests live so the READ
    path (:func:`ray_miss_value`) and ``VehiclePhysics.set_ray_miss_value``
    enforce them too.

    Returns the stamped value, or ``None`` if the sensor exposes no sentinel at
    all (a stub) — in which case readers fall back to
    :data:`RAY_MISS_THRESHOLD`."""
    if miss is None:
        miss = sensor_miss_value(sensor)
    if miss is None:
        _logger.debug("sensor %r exposes no miss sentinel; readers will fall "
                      "back to RAY_MISS_THRESHOLD", sensor)
        return None
    miss = check_miss_supported(sensor, float(miss))
    setattr(sensor, _MISS_ATTR, miss)
    return miss


def check_miss_supported(sensor: "object", miss: float) -> float:
    """Return ``miss``, or raise ``ValueError`` if it cannot serve as a miss
    sentinel for ``sensor``.

    TWO stages, and the first one is unconditional:

    1. **Is it a distance at all?** A non-finite (``nan``/``inf``) or
       non-positive sentinel raises, whatever the sensor is. This stage cannot
       be folded into stage 2: ``nan < max_range`` evaluates False, so a ``nan``
       sentinel used to pass the range test silently, and then
       ``is_ray_hit_corrected(d, nan)`` — an ``!=`` against ``nan`` — reports
       every genuine MISS as a HIT.
    2. **Is it outside the measurable range?** If ``sensor`` exposes a
       ``max_range``, a sentinel BELOW it raises. A sensor exposing none (a stub
       / foreign sensor) cannot be range-checked and passes this stage.

    Both stages are shared by all three entry points — the STAMP path
    (:func:`set_sensor_miss_value`), the READ path (:func:`ray_miss_value`) and
    the driver-level injection (``VehiclePhysics.set_ray_miss_value``) — so the
    guarantee holds wherever the sentinel is OBTAINED, not only where the SDK
    happens to stamp it. Only ``scene_helpers.make_wheel_raycaster`` and
    ``VehicleScene.add_vehicle`` stamp; the documented "build your own
    ``gs.sensors.Raycaster`` and hand it to
    ``VehiclePhysics(scene, entity, sensor, cfg)``" path never does, and
    validating only at the stamp let a ``no_hit_value=0.0`` sensor through to
    the pipeline.

    Why either configuration is unsupported: the SDK's read layer leaves a miss
    at its raw sentinel, and the physics core turns a distance into
    ``compression = max(rest_d - distance, 0)`` — so a small sentinel (``0.0``
    being the worst case) reads as MAXIMUM suspension compression and launches
    the vehicle, while a ``nan`` one defeats the miss test entirely. A loud
    failure beats either. Use the default (``no_hit_value = max_range``) or set
    ``no_hit_value`` to something at least as large as ``max_range``.

    .. versionadded:: 1.5.2
    .. versionchanged:: 1.5.2
       The finite-and-positive stage moved here from
       ``VehiclePhysics.set_ray_miss_value``, which was the only path that had
       it; a stub sensor accepted ``0.0``/``-1.0`` and a real one accepted
       ``nan`` before this."""
    miss = float(miss)
    if not math.isfinite(miss) or miss <= 0.0:
        raise ValueError(
            f"unsupported ray-miss sentinel no_hit_value={miss!r} on "
            f"{sensor!r}: a miss distance must be finite and positive (it is "
            f"the raycaster's max_range by default). genesis_vehicle leaves a "
            f"MISS at its raw sentinel and the suspension reads compression = "
            f"rest_d - distance, so a sentinel of 0.0 is maximum compression "
            f"and the vehicle launches; a non-finite one makes the miss test "
            f"(an equality against the sentinel) never fire, so every miss is "
            f"reported as a hit."
        )
    max_range = getattr(getattr(sensor, "_options", None), "max_range", None)
    if max_range is not None and miss < float(max_range):
        raise ValueError(
            f"unsupported raycaster configuration on {sensor!r}: the ray-miss "
            f"sentinel (no_hit_value={miss!r}) is below max_range="
            f"{float(max_range)!r}. genesis_vehicle leaves a MISS at its raw "
            f"sentinel and the suspension reads compression = rest_d - "
            f"distance, so a sentinel inside the measurable range would be "
            f"taken for a very close hit — maximum compression, and the "
            f"vehicle launches. Leave no_hit_value unset (it defaults to "
            f"max_range) or set it >= max_range."
        )
    return miss


def ray_miss_value(sensor: "object") -> "float | None":
    """The miss sentinel to test ``sensor``'s distances against: the value
    stamped by :func:`set_sensor_miss_value` if there is one, else the sensor's
    own resolved option, else ``None`` (a stub sensor — readers then fall back
    to :data:`RAY_MISS_THRESHOLD`).

    Validates through :func:`check_miss_supported` on BOTH paths, so a sentinel
    that is non-finite, non-positive, or below the sensor's ``max_range`` raises
    here exactly as it does at the stamp. This is the read half of the guarantee: ``VehiclePhysics.__init__``
    obtains the sentinel through this function, so handing it a hand-built
    raycaster with ``no_hit_value=0.0`` now fails at construction instead of
    reaching the suspension as maximum compression.

    .. versionchanged:: 1.5.2
       Validates instead of returning whatever it finds."""
    stamped = getattr(sensor, _MISS_ATTR, None)
    if stamped is not None:
        return check_miss_supported(sensor, float(stamped))
    miss = sensor_miss_value(sensor)
    if miss is None:
        return None
    return check_miss_supported(sensor, float(miss))


def _is_miss(d_raw, miss):
    """The sentinel half of :func:`is_ray_hit` — shared so the offset
    correction in :func:`read_distances` and the hit predicate cannot drift."""
    if miss is None:
        return d_raw >= RAY_MISS_THRESHOLD           # legacy fallback
    return d_raw == miss


def is_ray_hit(d_raw, miss: "float | None" = None):
    """True where a wheel ray found ground, tested on a **RAW** sensor distance:
    ``(d_raw != miss) & (d_raw != 0.0)``.

    ``d_raw`` MUST be the value straight off ``sensor.read().distances``, BEFORE
    :func:`read_distances` subtracts the high-cast offset. Pass a corrected
    distance and both terms misjudge it — see :func:`is_ray_hit_corrected`,
    which is what every in-SDK consumer of ``last_distances`` uses.

    Why an EQUALITY and not a threshold:

    * Genesis writes the sentinel VERBATIM into the distance buffer on a miss
      and its own hit test is likewise an equality
      (``distances != self._options.no_hit_value``), so the comparison is exact
      even for a value with no float32 representation (19.9, 0.1 — probed).
    * A threshold cannot work: ``no_hit_value`` defaults to ``max_range``, which
      the caller chooses. At ``max_range=10.0`` a miss returns 10.0, which is
      UNDER the old hardcoded 19.9 — the read layer then subtracted the
      high-cast offset from a distance that measured nothing.
    * The test must be on the RAW value: with a sentinel of ``0.0`` a genuine
      hit at exactly the offset corrects to 0.0 and would collide with it.

    The ``d_raw != 0.0`` term rejects the UNPOPULATED sensor: Genesis allocates
    the distance buffer as zeros and only fills it inside ``scene.step()``, and
    an all-zeros buffer is otherwise indistinguishable from four rays that hit
    ground ``up_offset`` above the wheel attachment points. The term is only
    meaningful HERE, on raw values, which is why the mask produced by
    :func:`read_distances` (``return_hit=True``) is the one to carry to
    consumers rather than re-deriving a predicate downstream.

    ``miss=None`` falls back to the deprecated :data:`RAY_MISS_THRESHOLD`, for
    callers holding a sensor that exposes no sentinel."""
    if miss is None:
        return (d_raw < RAY_MISS_THRESHOLD) & (d_raw != 0.0)   # legacy fallback
    return (d_raw != miss) & (d_raw != 0.0)


def is_ray_hit_corrected(d, miss: "float | None" = None):
    """True where a wheel ray found ground, tested on an **OFFSET-CORRECTED**
    distance: ``d != miss``.

    ``d`` is what :func:`read_distances` returns and what the SDK stores in
    ``VehiclePhysics.last_distances`` — the high-cast offset already subtracted
    from every hit, with a miss left at its raw sentinel. A miss therefore still
    compares equal to ``miss`` (that is exactly why
    :func:`check_miss_supported` forbids a sentinel inside the measurable
    range), so the sentinel term carries over unchanged.

    What does NOT carry over is :func:`is_ray_hit`'s ``!= 0.0`` term, and this
    function deliberately omits it:

    * an UNPOPULATED buffer of zeros has been shifted to ``-up_offset`` by the
      time it gets here, so the term would not fire on the case it exists for;
    * a GENUINE hit at exactly ``up_offset`` corrects to exactly 0.0, so the
      term would report a real, fully-compressed contact as air.

    The unpopulated-buffer case is instead guarded by
    ``VehiclePhysics._stepped_once`` (per env since v1.5.2), which every
    consumer of this predicate is behind: ``wheels_grounded`` /
    ``MultiVehiclePhysics.grounded_list`` AND it directly; the visual mirrors
    (``core._susp_visual_offset``, ``visual._susp_visual_target``) run inside
    the step (past the first-step protection) or, for the closed-form
    ``wheel_visual_transforms``, have their result MASKED back to the rest pose
    for any env whose ``_stepped_once`` is False. That dependency is
    load-bearing — do not expose a new caller of this predicate without the
    gate. Note the gate must be applied PER ENV: a partial
    ``reset(env_ids=...)`` re-zeroes only those envs' distances, and a
    batch-wide ``.any()`` lets them through (the v1.5.2 defect this note
    exists to prevent recurring).

    ``miss=None`` falls back to the deprecated :data:`RAY_MISS_THRESHOLD`.

    .. versionadded:: 1.5.2"""
    if miss is None:
        return d < RAY_MISS_THRESHOLD                          # legacy fallback
    return d != miss


def read_distances(sensor: "object", n_envs: int,
                   up_offset: "float | None" = None,
                   *, miss: "float | None" = None,
                   return_hit: bool = False):
    """Normalise sensor.read().distances to (n_envs, n_wheels), corrected
    back to ATTACHMENT-RELATIVE distances.

    Genesis returns shape (n_wheels,) when n_envs == 1 and (n_envs, n_wheels)
    otherwise; this collapses the two cases into one for the pipeline, and
    subtracts the high-cast offset from every distance EXCEPT a miss, which
    keeps its raw sentinel. A hit closer than the offset yields a NEGATIVE
    distance — the ground is above the attachment point (deep over-compression);
    the pipeline turns that into maximum compression so the suspension can
    recover the vehicle.

    ``up_offset`` defaults to whatever the sensor was stamped with by
    :func:`set_sensor_up_offset` (:data:`RAY_UP_OFFSET` if it never was), so a
    ``single_scene`` vehicle whose offset is capped reads back correctly without
    the caller knowing. ``miss`` likewise defaults to the sensor's own sentinel
    (:func:`ray_miss_value`).

    ``return_hit=True`` additionally returns the ``(n_envs, n_wheels)`` bool
    mask from :func:`is_ray_hit`, evaluated on the RAW distances — the only
    place that predicate can be applied correctly, and therefore the mask to
    CARRY to consumers (``VehiclePhysics.last_hit`` does exactly that) instead
    of re-deriving one from the corrected values this returns. The default
    single-tensor return is unchanged.

    **Swept-envelope sensors** (``fan_samples`` M > 1 — see
    :class:`WheelRayPattern`) read back ``(n_envs, n_wheels, M)``, and the M
    axis is ELIMINATED here, so no consumer downstream ever sees it: each
    sample's distance is corrected by the high-cast offset AND raised by its own
    swept-circle height offset ``c_j``, a MISSING sample is pushed to ``+inf``,
    and the per-wheel result is the MINIMUM over the fan (its hit mask, the OR
    over the fan). Because the centre sample has ``c_0 = 0`` exactly, the result
    is always ``<= `` the single-ray distance — the envelope can only RAISE the
    ground it reports, never lower it, so it cannot cause a sinking regression.
    The minimum is clamped just under the miss sentinel so a hit can never
    masquerade as one.

    The returned ``hit`` mask is the SOLE authority on miss-vs-hit; do not
    re-derive one by comparing the returned distance to the sentinel when M > 1
    (the clamp is what keeps even that working, but the mask is the contract).

    Unpopulated-buffer handling differs by mode, and BOTH sit behind
    ``VehiclePhysics._stepped_once`` (the first-step protection), which is what
    actually discards the step: at M=1 the all-zeros buffer keeps its historical
    ``-up_offset`` shift; at M > 1 every ``hit_j`` is False, so the wheel reports
    the miss sentinel instead. Neither reaches the suspension.

    .. versionchanged:: 1.5.2
       Misses are recognised by the sensor's OWN sentinel instead of a
       hardcoded 19.9 threshold. At the default ``raycaster_max_range=20.0``
       nothing changes; at any other range this fixes a miss having the
       high-cast offset subtracted out of it (at ``max_range=10.0`` a miss used
       to be reported as 9.0).
    .. versionchanged:: 1.6.0
       Collapses the M axis of a swept-envelope sensor. M=1 (the default) is
       bit-identical to 1.5.2."""
    d = sensor.read().distances
    fan_m, fan_c = sensor_fan(sensor)
    if fan_m > 1:
        return _read_distances_fan(sensor, d, fan_m, fan_c, up_offset, miss,
                                   return_hit)
    if d.dim() == 1:
        d = d.unsqueeze(0)
    if up_offset is None:
        up_offset = sensor_up_offset(sensor)
    if miss is None:
        miss = ray_miss_value(sensor)
    hit = is_ray_hit(d, miss)                       # on the RAW distances
    if up_offset:
        # Shift everything EXCEPT a miss, which keeps its raw sentinel. Note
        # this is the sentinel test alone, not `hit`: an unpopulated 0.0 keeps
        # its historical -up_offset shift, which the first-step protection in
        # VehiclePhysics.step already discards.
        d = torch.where(_is_miss(d, miss), d, d - up_offset)
    if return_hit:
        return d, hit
    return d


#: How far under the miss sentinel a swept-envelope HIT is clamped (m). The fan
#: minimum is a real measurement and cannot legitimately land ON the sentinel,
#: but the sentinel is what every downstream miss test compares against, so the
#: clamp makes "the mask says hit" and "the value is not the sentinel" agree
#: unconditionally.
_FAN_HIT_EPS = 1e-3


def _read_distances_fan(sensor, d_raw, fan_m: int, fan_c, up_offset, miss,
                        return_hit: bool):
    """The M > 1 half of :func:`read_distances` — collapse the swept-envelope
    fan to ONE distance per wheel.

    ``d_raw`` is ``(n_wheels, M)`` at ``n_envs == 0`` and ``(n_envs, n_wheels,
    M)`` otherwise (M is always the LAST axis; verified on genesis-world 1.4.0
    for n_envs 0/1/3). The squeeze is therefore against the pattern's RANK, not
    against ``n_envs``: rank 2 for a fan, rank 1 for the single ray — and a
    rank-2 pattern with M=1 would read back ``(n_wheels, 1)``, which is NOT the
    same tensor as the rank-1 ``(n_wheels,)``, which is why M=1 keeps a genuinely
    rank-1 return shape instead of a width-1 fan.
    """
    rank = 2                                        # M > 1 → (n_wheels, M)
    if d_raw.dim() == rank:
        d_raw = d_raw.unsqueeze(0)
    if d_raw.dim() != rank + 1 or d_raw.shape[-1] != fan_m:
        raise ValueError(
            f"swept-envelope sensor {sensor!r} was stamped with fan_samples="
            f"{fan_m} but reads back shape {tuple(d_raw.shape)}; expected "
            f"(n_envs, n_wheels, {fan_m}). The fan geometry stamped on a sensor "
            f"must match the WheelRayPattern its rays were built from.")
    if up_offset is None:
        up_offset = sensor_up_offset(sensor)
    if miss is None:
        miss = ray_miss_value(sensor)
    # A miss sentinel is required as the FILL value for a wheel whose whole fan
    # missed; a stub sensor exposing none falls back to the deprecated threshold,
    # exactly as the hit predicate does.
    miss_fill = float(RAY_MISS_THRESHOLD if miss is None else miss)

    c = torch.as_tensor(fan_c)
    if c.device != d_raw.device or c.dtype != d_raw.dtype:
        c = c.to(device=d_raw.device, dtype=d_raw.dtype)
        setattr(sensor, _FAN_C_ATTR, c)             # cache the conversion
    if c.shape != d_raw.shape[1:]:
        raise ValueError(
            f"swept-envelope height offsets have shape {tuple(c.shape)} but the "
            f"sensor reads {tuple(d_raw.shape[1:])} (n_wheels, M) — they are "
            f"per wheel and per sample.")

    hit_j = is_ray_hit(d_raw, miss)                 # on the RAW distances
    # Each sample's ground height, expressed at the WHEEL: the high-cast offset
    # off, plus c_j — how much higher than the wheel's lowest point the tire
    # surface stands at that longitudinal offset. A missing sample must not win
    # the minimum, so it goes to +inf.
    d_j = torch.where(hit_j, d_raw - up_offset + c,
                      torch.full_like(d_raw, float("inf")))
    d_eff = d_j.min(dim=-1).values
    hit = hit_j.any(dim=-1)
    d_eff = d_eff.clamp(max=miss_fill - _FAN_HIT_EPS)
    d = torch.where(hit, d_eff, torch.full_like(d_eff, miss_fill))
    if return_hit:
        return d, hit
    return d
