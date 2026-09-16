"""``_pipeline.pw`` / ``pw3`` — the per-wheel broadcast convention.

Every ``WheelMeta`` field the pipeline reads used to be broadcast with a bare
``.unsqueeze(0)``. That is correct only while the field is rank 1 (one value
per wheel, shared by every row). The moment a fused group carries one value per
ROW per wheel, ``.unsqueeze(0)`` on a ``(B, n)`` field yields ``(1, B, n)`` and
broadcasts the WRONG axis — silently, because ``(1, B, n)`` still multiplies
against ``(B, n)`` whenever ``B == n``.

The rank-2 branch is the one this tree never takes today (nothing is promoted
yet), so it is exercised here explicitly rather than left for STEP 3 to
discover.
"""

from __future__ import annotations

import torch

from genesis_vehicle._pipeline import pw, pw3


def test_pw_on_a_per_wheel_vector_is_exactly_unsqueeze():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = pw(t)
    assert out.shape == (1, 4)
    assert torch.equal(out, t.unsqueeze(0))       # VALUE equality, not approx


def test_pw_leaves_a_per_row_field_alone():
    t = torch.arange(8.0).reshape(2, 4)           # (B, n)
    assert pw(t) is t
    assert pw(t).shape == (2, 4)


def test_pw_would_have_broadcast_the_wrong_axis_without_the_rank_test():
    """The failure this helper exists to prevent, made visible: with B == n a
    bare unsqueeze is not a shape error, it is a wrong answer."""
    t = torch.arange(16.0).reshape(4, 4)          # B == n == 4
    state = torch.ones(4, 4)
    assert (state * t.unsqueeze(0)).shape == (1, 4, 4)   # rank grew, silently
    assert (state * pw(t)).shape == (4, 4)


def test_pw3_on_per_wheel_vectors_is_exactly_unsqueeze():
    t = torch.arange(12.0).reshape(4, 3)          # (n, 3) wheel positions
    out = pw3(t)
    assert out.shape == (1, 4, 3)
    assert torch.equal(out, t.unsqueeze(0))


def test_pw3_leaves_a_per_row_field_alone():
    t = torch.arange(24.0).reshape(2, 4, 3)       # (B, n, 3)
    assert pw3(t) is t


def test_pw3_handles_quaternions_too():
    t = torch.arange(16.0).reshape(4, 4)          # (n, 4) wheel quats
    assert pw3(t).shape == (1, 4, 4)
