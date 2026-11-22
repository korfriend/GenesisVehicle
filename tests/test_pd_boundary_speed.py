"""W2, the block-boundary speed sign (v2.0.1, every mode, cannot be switched off).

1.x: when the lookahead point sits exactly on a block's end waypoint and the
window is not capped (accumulated arc >= lookahead), ``v_target`` was that
waypoint's speed — which carries the NEXT block's direction — and the capped
taper's sign correction did not apply; the projection never moves back, so
the vehicle drove the wrong way for good. W2 keeps the current block's sign
there (magnitude unchanged), for a forward block before a reverse one and for
a reverse block before a forward one.

Literal runs (``replan2/lb_validate_gen.gen(rnd, "cusp")``, seed 14; paths and
start poses extracted by ``.pm_r2/r201b/extract_literals.py``): i = 9 (the
wrong sign first appears in the FORWARD block 0 at 3.68 s on 1.x) and i = 0
(first in the REVERSE block 1 at 14.1 s). On a kinematic bicycle (wheelbase
2.7 m, |delta| <= 0.35, a in [-4, 2] m/s^2, dt 0.025), ``plan()`` targets
applied directly: no driving step plans against its block, and the run ends
in a terminal mode within 120 s. The golden cusp entries stay identical.

Mutants this file must fail on (plan_r210 section 6): W2 removed (both
literals plan against their block); W2 applied to forward blocks only (i = 0).
"""
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.control import PathFollower  # noqa: E402
from genesis_vehicle.tests.data import make_pd_golden as G  # noqa: E402

L_WB, DMAX, DT = 2.7, 0.35, 0.025

# (v1, v2, l1, l2, theta, join, ext, v3, x0, y0, yaw0)
LITERALS = {
    9: (2.0, -2.0, 6.706954468243538, 11.651569613587924, 0.5758582696550925, True, True, 2.0,
        -1.2703933045685107, -1.1685917495511027, 1.131757707698167),
    0: (1.0, -1.0, 9.066728398404848, 6.046204107464313, 0.49021278593391837, False, True, 2.0,
        -0.424761510565383, 1.6356928085803855, -1.4858700597378678),
}


def _path(v1, v2, l1, l2, th, join, ext, v3):
    p1 = G.densify([(0, 0, 0, v1), (l1, 0, 0, v1)])
    p2 = G.densify([(l1, 0, 0, v2), (l1 - l2 * math.cos(th), -l2 * math.sin(th), 0, v2)])
    path = p1 + p2[1:] if join else p1 + p2
    if ext:
        e = path[-1]
        path = path + G.densify([(e[0], e[1], 0, v3), (e[0] - 8, e[1] + 4, 0, v3)])[1:]
    return path


@pytest.mark.parametrize("i,first_block_sign", [(9, 1), (0, -1)])
def test_literal_runs_never_plan_against_their_block(i, first_block_sign):
    *pp, x, y, yaw = LITERALS[i]
    f = PathFollower(_path(*pp), G.make_table(), mpc_model="off")
    v, wrong, boundary_hits, end = 0.0, 0, [], None
    for k in range(int(120.0 / DT)):
        pl = f.plan((x, y), yaw, v, 0.0, 0.0)
        if pl.mode == "DONE":
            end = "DONE"
            break
        if pl.command is None:
            bs, be, sgn = f.blocks[f.block_i]
            li, capped = f._lookahead()
            if li == be and not capped and f.block_i != len(f.blocks) - 1:
                boundary_hits.append(sgn)          # the 1.x wrong-sign situation
            if pl.v_target * sgn < 0:
                wrong += 1
        if pl.command is not None:
            a, w = (-4.0 * math.copysign(1, v) if abs(v) > 0.05 else -v / DT), 0.0
        else:
            a, w = pl.a_target, pl.omega_target
        a = max(-4, min(2, a)); wm = abs(v) * math.tan(DMAX) / L_WB; w = max(-wm, min(wm, w))
        v += a * DT; yaw += w * DT; x += v * math.cos(yaw) * DT; y += v * math.sin(yaw) * DT
        if pl.mode in PathFollower.TERMINAL_MODES and abs(v) < 1e-6:
            end = pl.mode
            break
    assert boundary_hits and boundary_hits[0] == first_block_sign   # the case is exercised
    assert wrong == 0
    assert end in PathFollower.TERMINAL_MODES


def test_golden_cusp_entries_unchanged():
    with open(os.path.join(os.path.dirname(__file__), "data", "pd_golden_1x.json")) as fh:
        runs = json.load(fh)["runs"]
    for name in ("cusp", "explicit_yaw_cusp"):
        for kind in ("table", "coupled"):
            _, got = G.run_scenario(name, kind, {"mpc_model": "off"})
            assert got == runs[f"{name}/{kind}"]["steps"]
