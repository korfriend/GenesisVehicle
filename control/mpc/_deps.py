"""Optional-dependency gate and process-scoped helpers for the MPC layer.

The MPC layer needs packages the rest of the SDK does not:

* ``impl="nonlinear"`` (the default): ``casadi`` and ``do_mpc``;
* ``impl="linear"``: ``osqp``, ``scipy`` and ``threadpoolctl``.

None of them is imported by ``import genesis_vehicle`` or by
``import genesis_vehicle.control.mpc``; they are imported when a solver is
constructed. :func:`require_mpc_deps` turns a missing one into an
``ImportError`` that names the packages and the requirements file.

The helpers below change PROCESS-GLOBAL state for the duration of a ``with``
block and restore it afterwards (try/finally). None of them is thread-safe:
casadi's NumPy mode, warning filters and the OpenBLAS thread count are global
to the process, so two threads using them at once can observe each other's
setting.
"""
from __future__ import annotations

import contextlib
import ctypes
import importlib
import logging
import os
import sys
import warnings

_log = logging.getLogger(__name__)

#: Packages per solver implementation (import names).
DEPS = {
    "nonlinear": ("casadi", "do_mpc"),
    "linear": ("osqp", "scipy", "threadpoolctl"),
}

#: Where the pinned optional requirements live (shipped next to the package).
REQUIREMENTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "requirements-mpc.txt")


def require_mpc_deps(impl: str) -> None:
    """Import every package ``impl`` needs, or raise ``ImportError`` naming them.

    Raises ``ValueError`` for an unknown ``impl``. Never falls back to PD.
    """
    if impl not in DEPS:
        raise ValueError(f"mpc_impl must be one of {sorted(DEPS)}; got {impl!r}")
    missing, first = [], None
    for name in DEPS[impl]:
        try:
            if name == "do_mpc":
                import_do_mpc()                        # its import warnings scoped
            else:
                importlib.import_module(name)
        except ImportError as e:                       # includes sys.modules[name] = None
            missing.append(name)
            first = first or e
    if missing:
        raise ImportError(
            f"the MPC layer (mpc_impl={impl!r}) needs optional packages that are "
            f"not installed. Missing: {', '.join(missing)}. Install them with "
            f"`pip install -r {REQUIREMENTS_FILE}` (check first with "
            "`pip install --dry-run --report - -r ...` that no installed package "
            "changes version), or pass mpc_model=\"off\" to run the PD path "
            "follower alone.") from first


def import_do_mpc():
    """``import do_mpc`` with ONLY its two import-time feature warnings silenced
    (ONNX, opcua) and a ``SyntaxWarning`` it triggers; the process-wide warning
    filters are left as they were (the field port installed them globally)."""
    mod = sys.modules.get("do_mpc")
    if mod is not None:
        return mod
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The ONNX feature is not available")
        warnings.filterwarnings("ignore", message="The opcua feature is not available")
        warnings.filterwarnings("ignore", message='"is" with a literal',
                                category=SyntaxWarning)
        import do_mpc
    return do_mpc


@contextlib.contextmanager
def casadi_numpy_mode(mode: int = -1):
    """Set ``casadi.GlobalOptions`` NumPy mode for the block, then restore it.

    do_mpc 5.1.2 relies on casadi's pre-3.8 NumPy interop; under casadi 3.8's
    default (0) its ``setup()`` emits a FutureWarning (casadi #2959). The field
    port set ``-1`` for the whole process at import; here it is scoped to the
    calls that need it. Not thread-safe (process-global option).
    """
    import casadi
    go = casadi.GlobalOptions
    get = getattr(go, "getNumpyMode", None)
    put = getattr(go, "setNumpyMode", None)
    if get is None or put is None:                    # casadi < 3.8: nothing to do
        yield
        return
    old = get()
    put(mode)
    try:
        yield
    finally:
        put(old)


# --- casadi's own OpenBLAS (used by IPOPT's MUMPS) ------------------------------

_CASADI_OPENBLAS_SONAME = "libcasadi-tp-openblas.so.0"
_openblas_handle = None
_openblas_failed = False


def casadi_openblas():
    """The LIVE copy of casadi's bundled OpenBLAS, or ``None``.

    Opened by soname (``libcasadi-tp-openblas.so.0``), which the dynamic loader
    dedups onto the copy IPOPT/MUMPS already mapped. Opening the file by path
    (``<casadi>/libcasadi-tp-openblas.so`` or ``.so.0.3``) would map a SECOND
    copy whose thread setting IPOPT never reads. The library is mapped only
    after the first ``nlpsol()`` construction (``MPC.setup()``), so call this
    after that. Linux only; on any other platform, or any ``OSError``, returns
    ``None`` ("not applied") and never raises. ``threadpoolctl`` does not see
    this library.
    """
    global _openblas_handle, _openblas_failed
    if _openblas_handle is not None or _openblas_failed:
        return _openblas_handle
    if not sys.platform.startswith("linux"):
        _openblas_failed = True
        return None
    try:
        lib = ctypes.CDLL(_CASADI_OPENBLAS_SONAME, mode=os.RTLD_NOLOAD | ctypes.RTLD_GLOBAL)
        lib.openblas_get_num_threads.restype = ctypes.c_int
        lib.openblas_set_num_threads.argtypes = [ctypes.c_int]
    except (OSError, AttributeError):
        _log.debug("casadi OpenBLAS not mapped / not found by soname", exc_info=True)
        return None                                   # not mapped yet: retry later
    _openblas_handle = lib
    return lib


def casadi_openblas_threads():
    """Current thread count of casadi's OpenBLAS, or ``None`` if unavailable."""
    lib = casadi_openblas()
    return None if lib is None else int(lib.openblas_get_num_threads())


@contextlib.contextmanager
def casadi_openblas_limit(n_threads):
    """Cap casadi's OpenBLAS at ``n_threads`` for the block, then restore it.

    ``None`` = do nothing. If the library cannot be reached (see
    :func:`casadi_openblas`) the block runs unchanged. Not thread-safe.
    """
    lib = casadi_openblas() if n_threads is not None else None
    if lib is None:
        yield
        return
    old = int(lib.openblas_get_num_threads())
    lib.openblas_set_num_threads(int(n_threads))
    try:
        yield
    finally:
        lib.openblas_set_num_threads(old)
