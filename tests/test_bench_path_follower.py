"""bench_path_follower's schedule and fail-closed publication rule (no Genesis).

Mutant this file must fail on: the publication rule loosened to "always publish".
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from genesis_vehicle.samples.bench_path_follower import publication, schedule  # noqa: E402


def test_schedule_alternates_the_arm_order():
    assert schedule(["n", "l", "p"], 2) == [(1, "n"), (1, "l"), (1, "p"), (2, "p"), (2, "l"), (2, "n")]


def test_publication_is_fail_closed():
    assert publication([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])[0] == (2.5, 1.0, 4.0, 6)
    pub, why = publication([1.0, 1.1, 1.2], [5.0, 6.0])
    assert pub is None and "disagree" in why
    assert publication([1.0], [])[0] is None
