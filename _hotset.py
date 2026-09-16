"""HOT-set bookkeeping: config fields the step path reads LIVE, and the
build-time values DERIVED from them.

Why this module exists
----------------------
Several quantities the per-step pipeline needs are pure functions of config
that used to be recomputed on **every** call — the Ackermann geometry
(``steering._resolve_geometry``), the driven-axle mask, the brake-bias vector,
the AWD weight normalisation, the drive-omega cap, and the two squared
thresholds (``StaticFrictionLock``'s ``v_thr**2``, ``CoulombIsotropic``'s
``eps_v**2`` — slots ``_v_thr_sq`` / ``_eps2``).
Hoisting them to build time is what lets a future fused group carry ONE
per-row tensor instead of re-deriving per slot per step.

A plain "compute it once in ``__init__``" hoist would have been a silent
behaviour change, and the tree has a caller that proves it:
``samples/tank_tuning.py:35-37`` (``TankTuning.apply_config``) writes
``hook.v_thr = 5.0`` **after** the preset constructed the hook, and
``apply_resolved`` (:47-48) writes ``drivetrain.omega_max_drive`` /
``t_brake_max`` **after** ``vs.build()``. Those writes take effect on the next
step today.

So every hoist here is a DEPENDENT cache, not a snapshot:

  derived value  =  f(source fields, wheel_meta, device, dtype)

and :func:`derived` re-runs ``f`` whenever any of those changes. What that
does and does not guarantee, exactly:

* a write to any source named in :data:`HOT_DEPENDENTS` (including an
  **in-place** edit of a list source such as ``brake_bias[0] = 0.9`` — list
  sources are keyed as tuples, i.e. by value) is picked up on the next call;
* a change of ``wheel_meta`` IDENTITY (a rebuilt driver, or the
  ``DifferentiablePlant``'s own meta) is picked up;
* an **in-place** mutation of a ``wheel_meta`` TENSOR is NOT picked up. That
  is the COLD set: ``WheelMeta`` is built once per driver
  (``VehiclePhysics._build_wheel_meta``) and the pipeline reads ``wm.*``, never
  ``resolved.wheels[i].*``; re-resolving wheel config after ``build()``
  requires ``VehicleScene.mark_config_dirty()``, which rebuilds the driver and
  therefore the meta. Nothing in the SDK writes into a built ``WheelMeta``
  tensor.

The table is also the input STEP 3's fused per-step refresh will read, which
is why the dependents are declared here rather than only implied by the call
sites.
"""

from __future__ import annotations

import weakref
from typing import Any, Callable

import torch


# ---------------------------------------------------------------------------
# The dependent cache
# ---------------------------------------------------------------------------

#: ``id(owner) -> {slot: (weak ref-or-None, key, value)}``.
#:
#: The cache lives HERE, not on the owner, and that is load-bearing three
#: times over:
#:
#: * **the owner stays picklable.** A cache entry holds a weak reference to the
#:   ``wheel_meta``; a ``weakref`` inside an instance ``__dict__`` makes the
#:   object unpicklable (``TypeError: cannot pickle 'weakref.ReferenceType'
#:   object``) — and only AFTER its first derive, so a config would pickle
#:   right after ``build()`` and fail one step later. RL vector-env workers,
#:   tuned-config checkpoints and ``torch.save(cfg)`` all pickle configs; the
#:   SDK itself never does, which is why no test would have caught it.
#: * **two generic attribute sweeps stay correct with no name list.**
#:   The structural fusion predicate — a diagnostic that ships in its own
#:   release, not this one — is FAIL-CLOSED over every non-dunder attribute of
#:   every role object, so a visible cache slot becomes a key item and splits
#:   vehicles that must fuse; that was observed as real failures in that
#:   workstream's tests while this change was being written. And
#:   ``control/plant._sync_hooks`` (in this tree) copies every TENSOR attribute of a live
#:   hook, row-slicing it by the flat batch and ``repeat_interleave``-ing it by
#:   the candidate count — right for the stick-slip integrator state it was
#:   written for, wrong for a ``(1, 1)`` config tensor, which comes out
#:   ``(cands, 1)`` and breaks the next compare against
#:   ``(M*cands, n_wheels)`` for any fleet with M > 1. An off-instance cache is
#:   invisible to both by construction, which is stronger than a dunder name.
#: * **a copy gets its own entry.** ``copy.copy`` (the fused per-group copy of
#:   STEP 2/3) and ``copy.deepcopy`` (the plant's prediction hooks) produce a
#:   new object, hence a new id and an empty cache, so a copy can never serve
#:   its original's derived value.
#:
#: Keyed by ``id`` rather than by the object: a ``WeakKeyDictionary`` would
#: hash the owner, and a user strategy that is an ``eq=True`` dataclass is
#: either unhashable or — worse — EQUAL to a different instance, which would
#: hand one strategy another's geometry. The entry is removed by a
#: ``weakref.finalize`` callback that runs while the owner is being
#: deallocated, i.e. before its id can be reused.
#:
#: STEP 3 note: this covers DERIVED values only. A PROMOTED SOURCE
#: (``hook.v_thr``, ``k_spring``, ...) is user-facing config and must stay a
#: public attribute, so it still meets ``_sync_hooks``' shape heuristic from
#: the source side.
_CACHES: dict[int, dict] = {}


def _cache_for(owner: Any) -> Any:
    """The cache dict for ``owner``, or ``None`` if it cannot have one.

    ``None`` means "derive every call" — exactly the pre-hoist behaviour, so an
    owner that cannot be weakly referenced (a user's ``__slots__`` strategy)
    keeps working, just without the saving."""
    key = id(owner)
    cache = _CACHES.get(key)
    if cache is not None:
        return cache
    try:
        weakref.finalize(owner, _CACHES.pop, key, None)
    except TypeError:
        return None
    cache = _CACHES[key] = {}
    return cache


def derived(owner: Any, slot: str, ref: Any, key: tuple,
            compute: Callable[[], Any]) -> Any:
    """Memoise ``compute()`` for ``owner``/``slot``, recomputing when an input
    moves.

    ``ref`` is compared by IDENTITY (``is``) and held WEAKLY — it is the
    ``wheel_meta`` (or ``None``). ``key`` is compared by VALUE and must contain
    every other input ``compute`` reads; a mutable source (a list) must enter
    it as a tuple, so an in-place edit is a different key.

    ``ref`` deliberately never reaches ``==``: ``WheelMeta`` is a dataclass, so
    ``==`` on two DIFFERENT metas compares tensor fields and raises "Boolean
    value of Tensor ... is ambiguous". CPython's ``PyObject_RichCompareBool``
    identity shortcut would hide that for the same-object case and expose it
    for every other, which is the worst shape a cache can have.

    A ``ref`` that cannot be weakly referenced is NOT cached. ``wheel_meta`` is
    a documented DUCK TYPE (``strategies.steering._WheelMetaProto`` asks for
    five fields and nothing else), and ``types.SimpleNamespace``, a
    ``namedtuple`` and a ``__slots__`` class all satisfy it while refusing
    ``weakref.ref``. Caching such a meta by identity alone would risk answering
    a NEW meta with a dead id's value, so the honest option is the pre-hoist
    one: derive every call.
    """
    cache = _cache_for(owner)
    if cache is None:
        return compute()
    rec = cache.get(slot)
    if rec is not None and _keys_equal(rec[1], key):
        held = rec[0]
        if (ref is None and held is None) or (
                held is not None and held() is ref):
            return rec[2]
    val = compute()
    if ref is None:
        cache[slot] = (None, key, val)
    else:
        try:
            cache[slot] = (weakref.ref(ref), key, val)
        except TypeError:
            cache.pop(slot, None)      # un-referenceable meta: never cache it
    return val


def _keys_equal(a: tuple, b: tuple) -> bool:
    """Element-wise key comparison that survives a TENSOR source.

    A plain ``a == b`` on tuples uses ``PyObject_RichCompareBool``, whose
    identity shortcut makes "the same tensor object" work and "a different
    tensor of the same value" raise "Boolean value of Tensor with more than one
    element is ambiguous" — i.e. it would work until the first time a caller
    re-assigned a promoted per-row source, which is the worst possible moment
    to find out. Scalars take the ordinary ``!=`` path, so nothing changes for
    the un-promoted sources STEP 0 has."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x is y:
            continue
        if torch.is_tensor(x) or torch.is_tensor(y):
            if not (torch.is_tensor(x) and torch.is_tensor(y)
                    and x.shape == y.shape and x.dtype == y.dtype
                    and x.device == y.device and torch.equal(x, y)):
                return False
            continue
        if x != y:
            return False
    return True


def derived_value(owner: Any, slot: str, default: Any = None) -> Any:
    """The currently cached value for ``owner``/``slot`` (``default`` if it has
    not been derived yet). For tests and for STEP 3's promotion — reading it
    does NOT derive."""
    cache = _CACHES.get(id(owner))
    if not cache or slot not in cache:
        return default
    return cache[slot][2]


def square_f64(x: Any) -> Any:
    """``x ** 2`` in DOUBLE — a python float, or a float64 tensor for a
    per-row source.

    Both squared thresholds are computed this way and only then cast to
    float32, because ``float32(s) * float32(s) != float32(double(s) ** 2)`` on
    6 of 10 sampled values — squaring in the storage precision moves the
    threshold."""
    if torch.is_tensor(x):
        d = x.to(dtype=torch.float64)
        return d * d
    return float(x) ** 2


def row_tensor(value: float, ref: torch.Tensor) -> torch.Tensor:
    """A derived scalar as a ``(1, 1)`` per-ROW tensor on ``ref``'s device and
    dtype.

    Per-row, not a python scalar: one row today (one vehicle = one value), and
    a fused group of V vehicles becomes ``(NV, 1)`` in STEP 3 by stacking. A
    python scalar cannot carry a per-slot value and would silently run the
    whole group on slot 0's threshold.
    """
    if torch.is_tensor(value):
        return value.reshape(-1, 1).to(device=ref.device, dtype=ref.dtype)
    return torch.full((1, 1), float(value),
                      device=ref.device, dtype=ref.dtype)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

#: ``(role, source attribute) -> (derived attribute, ...)``.
#:
#: ROLE is the ``ResolvedConfig`` attribute name the object is reached through
#: (``"hook"`` = any entry of ``resolved.stability_hooks``). The right-hand
#: names are :func:`derived` SLOTS — keys in this owner's entry of
#: :data:`_CACHES`, readable with :func:`derived_value`. They are NOT
#: attributes of the owner (see :data:`_CACHES` for why).
#:
#: Enforcement: every pair below is covered by a named gate in
#: ``tests/test_build_time_hoists.py`` / ``tests/test_tire_coulomb_eps.py``,
#: which writes the source AFTER the value has been derived once and asserts
#: the next call moves. The table is descriptive of the call sites; it does not
#: itself invalidate anything.
HOT_DEPENDENTS: dict[tuple[str, str], tuple[str, ...]] = {
    ("steering", "wheelbase"):        ("_geom",),
    ("steering", "track_width"):      ("_geom",),
    ("steering", "front_axle"):       ("_geom",),
    ("drivetrain", "driven_axles"):   ("_driven", "_drive_share"),
    ("drivetrain", "brake_bias"):     ("_brake_bias",),
    ("drivetrain", "drive_weights"):  ("_drive_weights",),
    ("drivetrain", "omega_max_drive"): ("_omega_cap",),
    ("tire", "eps_v"):                ("_eps2",),
    ("hook", "v_thr"):                ("_v_thr_sq",),
}


def prime_derived(resolved: Any, wheel_meta: Any, device: Any,
                  dtype: Any) -> None:
    """Compute every hoisted value ONCE, at BUILD time.

    Called from ``VehiclePhysics.__init__`` after ``resolve()`` and after the
    ``WheelMeta`` exists — i.e. after every documented pre-build config write
    (``TankTuning.apply_config``) and before the first step. Purely an
    optimisation of WHEN: each site would otherwise derive the same value on
    its first call, and every site re-derives on a later source change either
    way. A role that raises here (e.g. an Ackermann whose ``front_axle``
    matches no wheel) is left to raise from the step path exactly as before, so
    priming cannot turn a step-time error into a build-time one.
    """
    roles = [getattr(resolved, r, None)
             for r in ("steering", "drivetrain", "coupling", "tire")]
    roles += list(getattr(resolved, "stability_hooks", None) or [])
    for obj in roles:
        fn = getattr(obj, "_prime_derived", None)
        if fn is None:
            continue
        try:
            fn(wheel_meta, device, dtype)
        except Exception:
            # Deriving early must never change WHETHER a config is accepted.
            # The site re-raises on the first real call if the config is bad.
            import logging
            logging.getLogger("genesis_vehicle").debug(
                "prime_derived: %s could not pre-compute its build-time "
                "values; the step path will derive (and report) them",
                type(obj).__name__, exc_info=True)
