"""Equivalence guard for the v1.0.10 vectorized ``lerp_state``.

The server interpolates every chassis/wheel/obstacle pose each loop before the
OSC send. Pre-1.0.10 this was one python ``slerp`` call per quaternion
(30 tanks x 10 wheels = 300 calls/loop ~ 14 ms — rivaling the physics step);
1.0.10 gathers them into flat arrays and slerps in one numpy pass
(``_slerp_batch``). This test pins the vectorized path to the original
per-quaternion semantics: same positions, same quaternions (up to fp), same
wheel-angle wrap-around lerp, same passthrough behavior for entries missing
from ``prev``.
"""

import numpy as np
import pytest

pytest.importorskip("genesis")

from genesis_vehicle.server.physics_server import lerp_state, slerp


def _reference_lerp_state(prev, curr, a):
    """Verbatim copy of the pre-1.0.10 per-quaternion loop (ground truth)."""
    interpolated = {'targets': [], 'dynamic_obstacles': []}
    for tid, curr_data in curr['targets'].items():
        if tid in prev['targets']:
            prev_p, prev_q, prev_wheels = prev['targets'][tid]
            curr_p, curr_q, curr_wheels = curr_data
            interp_p = prev_p * (1.0 - a) + curr_p * a
            interp_q = slerp(prev_q, curr_q, a)
            interp_wheels = []
            for j in range(min(len(prev_wheels), len(curr_wheels))):
                pw, cw = prev_wheels[j], curr_wheels[j]
                pw_angle = pw[2] if len(pw) > 2 else 0.0
                cw_angle = cw[2] if len(cw) > 2 else 0.0
                iw_p = pw[0] * (1.0 - a) + cw[0] * a
                iw_q = slerp(pw[1], cw[1], a)
                diff = (cw_angle - pw_angle + np.pi) % (2.0 * np.pi) - np.pi
                iw_angle = (pw_angle + diff * a) % (2.0 * np.pi)
                interp_wheels.append((iw_p, iw_q, iw_angle))
            interpolated['targets'].append((tid, interp_p, interp_q, interp_wheels))
        else:
            interpolated['targets'].append((tid, curr_data[0], curr_data[1], curr_data[2]))
    for o_id, curr_data in curr['dynamic_obstacles'].items():
        if o_id in prev['dynamic_obstacles']:
            prev_p, prev_q = prev['dynamic_obstacles'][o_id]
            interp_p = prev_p * (1.0 - a) + curr_data[0] * a
            interp_q = slerp(prev_q, curr_data[1], a)
            interpolated['dynamic_obstacles'].append((o_id, interp_p, interp_q))
        else:
            interpolated['dynamic_obstacles'].append((o_id, curr_data[0], curr_data[1]))
    return interpolated


def _rand_quat(rng):
    q = rng.normal(size=4).astype(np.float32)
    return q / np.linalg.norm(q)


def _make_states(n_targets=30, n_wheels=10, n_obs=5, seed=7):
    rng = np.random.default_rng(seed)
    prev = {'targets': {}, 'dynamic_obstacles': {}}
    curr = {'targets': {}, 'dynamic_obstacles': {}}
    for tid in range(n_targets):
        def wheels(rng):
            return [(rng.normal(size=3).astype(np.float32), _rand_quat(rng),
                     float(rng.uniform(0, 2 * np.pi))) for _ in range(n_wheels)]
        prev['targets'][tid] = (rng.normal(size=3).astype(np.float32), _rand_quat(rng), wheels(rng))
        curr['targets'][tid] = (rng.normal(size=3).astype(np.float32), _rand_quat(rng), wheels(rng))
    # a target with no prev entry → must pass through untouched
    curr['targets'][999] = (np.ones(3, np.float32), _rand_quat(rng), [])
    for o in range(n_obs):
        prev['dynamic_obstacles'][o] = (rng.normal(size=3).astype(np.float32), _rand_quat(rng))
        curr['dynamic_obstacles'][o] = (rng.normal(size=3).astype(np.float32), _rand_quat(rng))
    return prev, curr


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.9999])
def test_lerp_state_matches_scalar_reference(alpha):
    prev, curr = _make_states()
    ref = _reference_lerp_state(prev, curr, alpha)
    out = lerp_state(prev, curr, alpha)

    ref_t = {t[0]: t for t in ref['targets']}
    out_t = {t[0]: t for t in out['targets']}
    assert set(ref_t) == set(out_t)
    for tid in ref_t:
        _, rp, rq, rw = ref_t[tid]
        _, op, oq, ow = out_t[tid]
        np.testing.assert_allclose(op, rp, atol=1e-5)
        # quats equal up to fp (both take the shortest path; same sign)
        np.testing.assert_allclose(oq, np.asarray(rq) / np.linalg.norm(rq), atol=1e-4)
        assert len(ow) == len(rw)
        for j in range(len(rw)):
            np.testing.assert_allclose(ow[j][0], rw[j][0], atol=1e-5)
            np.testing.assert_allclose(ow[j][1], np.asarray(rw[j][1]) / np.linalg.norm(rw[j][1]), atol=1e-4)
            assert abs(ow[j][2] - rw[j][2]) < 1e-4

    ref_o = {t[0]: t for t in ref['dynamic_obstacles']}
    out_o = {t[0]: t for t in out['dynamic_obstacles']}
    assert set(ref_o) == set(out_o)
    for oid in ref_o:
        np.testing.assert_allclose(out_o[oid][1], ref_o[oid][1], atol=1e-5)
        np.testing.assert_allclose(out_o[oid][2], np.asarray(ref_o[oid][2]) / np.linalg.norm(ref_o[oid][2]), atol=1e-4)


def test_lerp_state_passthrough_when_prev_missing():
    prev, curr = _make_states(n_targets=2)
    out = lerp_state(prev, curr, 0.5)
    out_t = {t[0]: t for t in out['targets']}
    assert 999 in out_t                       # tid only in curr → passed through
    np.testing.assert_allclose(out_t[999][1], np.ones(3, np.float32))
