"""bench_raycast_mode.py — order-independent benchmark of the two wheel-raycast modes.

Answers ONE question: on THIS machine, at THESE settings, is
``raycast_mode="single_scene"`` slower or faster per simulation step than
``raycast_mode="dual_scene"`` — and is the difference big enough, relative to
this machine's own run-to-run noise, to be worth quoting at all?

It is allowed to answer "not measurable here", and on a noisy CPU/WSL box that
is the expected answer. See FAIL-CLOSED below.

| abbr | meaning |
|---|---|
| ms/step | wall-clock milliseconds per simulation step (one `veh.set_inputs` + one `vs.step`) |
| r_k | paired ratio of repeat k: ms(single_scene, k) / ms(dual_scene, k) |
| A / B | repeat groups by slot order: A = dual_scene ran first, B = single_scene ran first |
| hs | `horizontal_scale`, the terrain cell size in metres (faces ~= 2*(size/hs)^2) |
| BVH | Bounding Volume Hierarchy (the raycast acceleration structure) |
| L3 | the `n_envs` batching axis (see docs/batching.md) |

Why this exists
---------------
``dual_scene_terrain.py --compare`` cannot produce a quotable ratio: it runs
both modes back to back in ONE process in a FIXED order, and the slots are not
equivalent — measured, the full-step ratio median is 1.10 single-then-dual vs
1.28 dual-then-single, so the ORDER alone moves the answer (cause never
established). This harness removes the three defects that made that number
unquotable:

1. **Fresh process per measurement.** Every ms/step is produced by a child
   process that does its own ``gs.init`` / build / settle / drive, exactly like
   ``perf_vectorization.py``.
2. **Alternating order, paired.** Repeat k alternates which mode occupies slot
   1. The published statistic is the PAIRED ratio r_k within a repeat, so a
   drift that moves both slots of a repeat cancels; the slot effect itself does
   NOT cancel inside r_k, which is precisely why the groups are compared (rule
   (i) below).
3. **Fail-closed publication.** A ratio is printed only if two conditions hold
   at once. Otherwise the harness prints the full paired interval and says the
   ratio is not measurable at these settings, and still exits 0.

What it does NOT do
-------------------
- It does NOT time the raycast in isolation. The SDK's raycast is not
  separately observable through a public API, so every number here is a FULL
  simulation step: raycast + the 5-step wheel pipeline + ``scene.step``. A
  raycast-only speedup is therefore diluted by everything else in the step, and
  a ratio near 1.0 does not mean the two raycasters cost the same.
- It does NOT produce a confidence interval, a p-value or a PASS/FAIL verdict
  on speed. Nothing here is a statistical test.
- It has NOT been validated on GPU. ``--gpu`` is wired through to the same code
  path, but every number, default and threshold in this file was measured on
  CPU (WSL2), and the GPU path is unverified on the machine this harness was
  written on. On GPU also see the determinism note in the repo CLAUDE.md.

The parent process property
---------------------------
"The parent produces no measurement itself — every ms/step comes from a child's
RESULT_JSON. The parent never calls gs.init and never constructs a
VehicleScene. This does NOT mean genesis is absent from parent memory —
__init__.py:150 → control/plant.py:97 → _pipeline.py:23 eagerly imports
genesis.utils.geom, so that is impossible."

Concretely: every heavy import lives inside ``_worker_main``; the parent path is
covered by a test that patches ``genesis.init`` to raise and still runs a full
schedule through an injected runner.

The measurement
---------------
Each worker calls ``dual_scene_terrain.run(..., measure=True)``: settle on the
brake, 10 warm-up steps, then a timed loop at a constant throttle with no
steering (0.6 / 0.0 at the time of writing — the harness prints and stores the
values ``run()`` reports rather than these literals, and aborts if two workers
disagree on them). The window's start speed and x are read AFTER the warm-up and BEFORE
the timer starts, and the end speed after the timer stops — a device->host sync
inside the timed loop would contaminate the number being measured.

Because the window is a constant-throttle ACCELERATION, the vehicle state is
not stationary across it: at the defaults it covers roughly v = 0.33 .. 19 m/s.
That is a condition of the measurement, not a defect, and it is printed and
stored (read from the worker, never assumed).

The plate is 640x640 m for one reason only: it lets the timed window be long
enough that process-fixed cost is a smaller share of each worker. Computed from
the 24 non-warm-up workers of the two full runs shipped with this change
(bench1 + bench2, --drive-s 20, CPU/WSL2, this machine): ms/step 8.41 .. 12.44,
so the 800-step window took 6.73 .. 9.95 s inside a worker whose total wall
time was 16.43 .. 22.58 s — a timed fraction of 32 .. 51%, median 39%. No
comparable figure for the old 40 m plate is quoted here, because none was
measured under this harness.

The two modes do the same physical work over that window, so r_k compares two
runs that drove the same distance rather than one that did more work than the
other. Measured at these exact defaults (640x640 m plate, hs 4.0, CPU, WSL2,
genesis-world 1.4.0, dt 0.025 / substeps 10, n_envs 1, 3 s brake settle then
20 s at constant throttle 0.6) by this harness's own 12 workers: x_end =
208.148712 m for every dual_scene worker and 208.148788 m for every
single_scene worker — the modes agree to 6 significant digits (208.149 m),
with |dx| = 7.6e-5 m between them, and are bit-repeatable WITHIN a mode across
all 6 repeats. An independent probe (plan-review's probe2.py) reproduced the
same pair. Each run re-measures both x_end values and stores them per worker,
so a machine on which that stops being true is visible in the JSON instead of
being silently folded into the ratio.

Validity gates (a run that fell off the world is FASTER, so this is not optional)
--------------------------------------------------------------------------------
An invalid run can be a biased-fast run. One pair of fresh-process
``dual_scene_terrain.run()`` calls made while this harness was being planned
(default 40 m plate, hs 0.25, CPU/WSL2, genesis-world 1.4.0, dt 0.025 /
substeps 10, n_envs 1, throttle 0.6): the ``drive_s=12`` run, which left the
plate at ~5.7 s and free-fell for the rest, timed 7.66 ms/step; the valid
``drive_s=4`` run timed 10.85 ms/step. The valid run was that session's FIRST
(cold) process, so the pair is indicative, not a controlled comparison — the
point it supports is only that an invalid run cannot be assumed to look
slower, so a gate that drops or averages it would bias the result in the
direction least likely to be noticed. Therefore:

- every worker's payload must carry every key the parent reads, and must
  report the mode / repeat / slot / config it was DISPATCHED with — the pairing
  is done on the payload's own labels, so a mislabelled worker would pair the
  wrong two runs;
- every worker, INCLUDING the warm-up worker, must pass all four of
  ``dual_scene_terrain.verdicts()`` — the worker exits non-zero if it does not,
  and the parent re-checks the payload rather than trusting the exit code;
- every worker must also satisfy an x-margin gate: max(|x_min|, |x_max|) <
  0.75 x the (d) bound. This is an X-AXIS gate ONLY (it says nothing about y;
  y stays covered by (d)). It exists to fail while the run is still comfortably
  on the plate instead of at the edge, and it fires before (d) is examined;
- cross-worker invariants: identical ``faces`` within a config, and identical
  ``genesis_version``, ``dt``, ``substeps``, ``terrain_half`` across ALL
  workers, plus every other value the printed conditions block takes from one
  worker (backend, sdk/python version, terrain_size, settle_s, drive_s,
  throttle, steer, warm-up steps, timed steps, the ms/step definition).
  ``speed_at_window_start`` is deliberately NOT an equality invariant:
  it differs slightly between the modes in practice (measured 0.332 dual vs
  0.329 single m/s), so its per-worker values are recorded and the printed
  condition is the min..max envelope.

ANY failure aborts the whole run with a non-zero exit and no aggregate. Nothing
is skipped and averaged over: a benchmark that silently drops its failures is
how a biased number gets published.

FAIL-CLOSED publication rule
----------------------------
Both conditions must hold before a ratio line is printed:

  (i)  mutual group inclusion: median(A) lies within [min(B), max(B)] AND
       median(B) lies within [min(A), max(A)]. If the dual-first and
       single-first groups separate, what is being measured is the slot, not
       the mode.
  (ii) 1.0 lies OUTSIDE [min(r_k), max(r_k)]. If a single repeat crossed 1.0,
       the sign of the effect is not established on this machine.

The paired ratios, the pooled median with its [min .. max] and n, and both
group summaries are printed ALWAYS, published or not. When a ratio IS printed
it is always in interval form with n; there is no bare point estimate.

This is a HEURISTIC DISCLOSURE RULE, NOT A STATISTICAL TEST. At repeats=6 no
exact test can reach significance, so no exact test is attempted; [min .. max]
is a lower bound on the uncertainty and is NOT a confidence interval. Its
behaviour with more repeats is not "run longer and you will get a number":
condition (ii) gets HARDER with more samples (more chances to straddle 1.0)
while (i) gets easier. Simulated by plan-review with this machine's measured
run-to-run spread as the noise model (5 fresh dual_scene runs: 9.438, 8.620,
8.597, 10.065, 7.799 ms/step — a range of 26% of the median), P(publish) at
repeats=6 is: null effect 0.47%; a pure slot effect of 1.0 ms 0.072% and of
2.0 or 4.0 ms 0.000%; a true 1.13x 17.1%; a true 2.79x 48.6%, rising to 77.4%
at repeats=10. So more repeats help a LARGE effect publish and make a small or
null one publish LESS. Those percentages describe that simulation's noise
model, not a guarantee of this code.

A consequence worth stating before you run it twice: at a true 1.13x and
repeats=6 the publish probability is 17.1%, so two honest runs disagree about
publishability roughly 28% of the time even with a perfect rule. A flip between
runs is a result about this machine's noise; it is written up as a flip, not
resolved by picking the run you liked.

Run
---
    python -m genesis_vehicle.samples.bench_raycast_mode
    python -m genesis_vehicle.samples.bench_raycast_mode --json out.json
    python -m genesis_vehicle.samples.bench_raycast_mode --repeats 10
    python -m genesis_vehicle.samples.bench_raycast_mode --horizontal-scale 4.0,2.0

Exit code: 0 when every worker was valid (whether or not a ratio was
publishable), non-zero when a worker failed, a gate fired, or an invariant
broke.
"""

from __future__ import annotations

# Bootstrap: allow `python path/to/this_file.py` (and `python -m ...`) both.
import sys, pathlib
_SDK_PARENT = str(pathlib.Path(__file__).resolve().parents[2])
if _SDK_PARENT not in sys.path:
    sys.path.insert(0, _SDK_PARENT)

# STDLIB ONLY at module scope. The parent must not import genesis or the SDK:
# every heavy import lives inside _worker_main(), which only ever runs in a
# child process (or in a test that deliberately calls it).
import argparse
import json
import os
import statistics
import subprocess
import time

HARNESS_VERSION = "bench_raycast_mode/1"

MODES = ("dual_scene", "single_scene")

# Fraction of the (d) bound a worker's |x| may reach. 0.75 keeps the timed
# window off the plate's edge; see "Validity gates" in the module docstring.
X_MARGIN_FRACTION = 0.75

_RESULT_PREFIX = "RESULT_JSON "


# ---------------------------------------------------------------------------
# Pure functions (no I/O, no genesis) — these are what the tests drive
# ---------------------------------------------------------------------------

def build_schedule(configs, repeats: int):
    """The full worker schedule, in execution order.

    ``configs`` is a list of (horizontal_scale, n_envs); ``repeats`` must be
    even and >= 4 (enforced by the CLI, re-asserted here).

    Repeat k runs BOTH modes back to back — the two workers of a repeat are
    adjacent and no other config is interleaved between them, which is the
    precondition for pairing them. Even k puts dual_scene in slot 1, odd k puts
    single_scene in slot 1, so over an even number of repeats each mode occupies
    each slot the same number of times and the slot effect cancels in the
    aggregate (it does NOT cancel inside a single r_k — that is what the group
    comparison is for).
    """
    if repeats % 2 != 0 or repeats < 4:
        raise ValueError("repeats must be even and >= 4")
    jobs = []
    for ci, (hs, n_envs) in enumerate(configs):
        config_id = config_id_of(hs, n_envs)
        for k in range(repeats):
            order = ("dual_scene", "single_scene") if k % 2 == 0 else \
                    ("single_scene", "dual_scene")
            for slot, mode in enumerate(order, start=1):
                jobs.append(dict(config_index=ci, config_id=config_id,
                                 horizontal_scale=hs, n_envs=n_envs,
                                 repeat=k, slot=slot, mode=mode, warmup=False))
    return jobs


def config_id_of(horizontal_scale: float, n_envs: int) -> str:
    return f"hs{horizontal_scale:g}_env{int(n_envs)}"


def paired_ratios(samples):
    """[(repeat, r_k)] for ONE config's non-warm-up samples, sorted by repeat.

    r_k = ms(single_scene, k) / ms(dual_scene, k): > 1.0 means single_scene
    took longer per step in that repeat. Raises if a repeat is missing a mode
    or a mode appears twice — an incomplete pair must abort, never be dropped.
    """
    by_repeat: dict = {}
    for s in samples:
        if s.get("warmup"):
            continue
        slot = by_repeat.setdefault(int(s["repeat"]), {})
        mode = s["mode"]
        if mode in slot:
            raise ValueError(f"repeat {s['repeat']} has two {mode} samples")
        slot[mode] = float(s["ms"])
    out = []
    for k in sorted(by_repeat):
        pair = by_repeat[k]
        missing = [m for m in MODES if m not in pair]
        if missing:
            raise ValueError(f"repeat {k} is missing {missing}")
        out.append((k, pair["single_scene"] / pair["dual_scene"]))
    return out


def split_groups(paired):
    """(group_A, group_B) r-values: A = even repeats (dual_scene ran in slot 1),
    B = odd repeats (single_scene ran in slot 1). Matches build_schedule."""
    a = [r for k, r in paired if k % 2 == 0]
    b = [r for k, r in paired if k % 2 == 1]
    return a, b


def _interval(values):
    return dict(median=statistics.median(values), min=min(values),
                max=max(values), n=len(values))


def publication_decision(paired, group_a, group_b) -> dict:
    """Apply the fail-closed rule. Returns a dict, never raises on data shape.

    keys: publishable (bool), reason (str), cond_i / cond_ii (bool),
    detail_i / detail_ii (str). See the module docstring: this is a disclosure
    heuristic, not a test.
    """
    r = [v for _, v in paired]
    if len(r) < 2 or not group_a or not group_b:
        return dict(publishable=False, cond_i=False, cond_ii=False,
                    detail_i="not enough paired repeats to compare groups",
                    detail_ii="not enough paired repeats",
                    reason="not enough paired repeats")
    ma, mb = statistics.median(group_a), statistics.median(group_b)
    lo_a, hi_a = min(group_a), max(group_a)
    lo_b, hi_b = min(group_b), max(group_b)
    cond_i = (lo_b <= ma <= hi_b) and (lo_a <= mb <= hi_a)
    detail_i = (f"median(A)={ma:.3f} in [{lo_b:.3f} .. {hi_b:.3f}]=B? "
                f"{'yes' if lo_b <= ma <= hi_b else 'NO'}; "
                f"median(B)={mb:.3f} in [{lo_a:.3f} .. {hi_a:.3f}]=A? "
                f"{'yes' if lo_a <= mb <= hi_a else 'NO'}")

    lo, hi = min(r), max(r)
    cond_ii = not (lo <= 1.0 <= hi)
    detail_ii = (f"1.0 {'outside' if cond_ii else 'INSIDE'} "
                 f"[{lo:.3f} .. {hi:.3f}] (n={len(r)})")

    if cond_i and cond_ii:
        reason = "both conditions hold"
    else:
        failed = []
        if not cond_i:
            failed.append("(i) group separation")
        if not cond_ii:
            failed.append("(ii) 1.0 inside the paired range")
        reason = "failed " + " and ".join(failed)
    return dict(publishable=bool(cond_i and cond_ii), cond_i=bool(cond_i),
                cond_ii=bool(cond_ii), detail_i=detail_i, detail_ii=detail_ii,
                reason=reason)


def format_ratio_line(interval) -> str:
    """The ONLY shape a published ratio may take: median + [min .. max] + n."""
    return (f"single/dual = median {interval['median']:.3f} "
            f"[{interval['min']:.3f} .. {interval['max']:.3f}] "
            f"n={interval['n']}")


def parse_result_line(stdout: str):
    """The LAST RESULT_JSON line in a child's stdout, or None.

    Last, not first: genesis prints banners and warnings freely, and a future
    version could print something that looks like a result. The worker emits
    exactly one.
    """
    found = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(_RESULT_PREFIX):
            try:
                found = json.loads(line[len(_RESULT_PREFIX):])
            except json.JSONDecodeError:
                continue
    return found


# Every payload key the parent reads. Checked up front so a payload from an
# OLDER dual_scene_terrain (pre-1.6.1, which returns none of the conditions
# block) aborts by name instead of dying in whichever accessor got there first.
_REQUIRED_PAYLOAD_KEYS = (
    "mode", "repeat", "slot", "config_id", "warmup", "ms", "faces", "x",
    "x_min", "x_max", "verdicts", "terrain_half", "terrain_size",
    "genesis_version", "sdk_version", "python_version", "backend",
    "dt", "substeps", "settle_s", "drive_s", "throttle", "steer",
    "horizontal_scale", "n_envs", "warmup_steps", "n_timed_steps",
    "timing_definition", "speed_at_window_start", "speed_at_window_end",
    "wall_s",
)


def missing_payload_keys(payload) -> list:
    """Required keys absent from a worker payload, in declaration order."""
    return [k for k in _REQUIRED_PAYLOAD_KEYS if k not in payload]


def x_margin_check(payload) -> "tuple[bool, str]":
    """X-AXIS margin gate: max(|x_min|, |x_max|) < 0.75 x the (d) bound.

    Uses the env envelope, not the env-0 scalar, so a single stray env in an L3
    batch cannot hide behind env 0 — the same reason verdicts() uses it. Says
    nothing about y; (d) still bounds y.

    Assumes the payload is complete: callers must run missing_payload_keys()
    first (``_check_worker`` does), so a payload without ``terrain_half``
    aborts by name rather than raising here.
    """
    half = payload["terrain_half"]
    bound = float(half[0]) - 2.0
    limit = bound * X_MARGIN_FRACTION
    ax = max(abs(float(payload["x_min"])), abs(float(payload["x_max"])))
    ok = ax < limit
    return ok, (f"|x|max={ax:.2f} m vs limit {limit:.2f} m "
                f"(= {X_MARGIN_FRACTION:.2f} x the (d) bound {bound:.1f} m; "
                f"x-axis only)")


# ---------------------------------------------------------------------------
# Worker — the ONLY place that touches genesis
# ---------------------------------------------------------------------------

def _worker_main(args) -> int:
    """One measurement in this process; prints one RESULT_JSON line.

    Exits non-zero when the run is invalid, but prints the payload FIRST so the
    parent can name the criterion that failed instead of guessing from stderr.
    """
    from genesis_vehicle.samples import dual_scene_terrain as dst

    backend = "gpu" if args.gpu else "cpu"
    t_start = time.time()
    r = dst.run(args.mode, backend, args.horizontal_scale[0], args.n_envs[0],
                settle_s=args.settle_s, drive_s=args.drive_s, measure=True,
                terrain_size=args.terrain_size)
    t_end = time.time()
    checks = dst.verdicts(r)

    payload = dict(r)
    payload.update(
        mode=args.mode, slot=int(args.slot), repeat=int(args.repeat),
        config_id=args.config_id, warmup=bool(args.warmup),
        verdicts=[[label, bool(ok), detail] for label, ok, detail in checks],
        valid=all(ok for _, ok, _ in checks),
        t_start=t_start, t_end=t_end, wall_s=t_end - t_start,
        python_version=sys.version.split()[0],
        argv=list(sys.argv[1:]),
        harness_version=HARNESS_VERSION,
    )
    print(_RESULT_PREFIX + json.dumps(payload), flush=True)
    return 0 if payload["valid"] else 1


# ---------------------------------------------------------------------------
# Parent — spawns workers, gates them, aggregates
# ---------------------------------------------------------------------------

def _worker_argv(job, args) -> list:
    argv = [
        sys.executable, "-m", "genesis_vehicle.samples.bench_raycast_mode",
        "--_worker",
        "--mode", job["mode"],
        "--repeat", str(job["repeat"]),
        "--slot", str(job["slot"]),
        "--config-id", job["config_id"],
        "--horizontal-scale", f"{job['horizontal_scale']:g}",
        "--n-envs", str(job["n_envs"]),
        "--terrain-size", f"{args.terrain_size:g}",
        "--settle-s", f"{args.settle_s:g}",
        "--drive-s", f"{args.drive_s:g}",
    ]
    if job["warmup"]:
        argv.append("--warmup")
    if args.gpu:
        argv.append("--gpu")
    return argv


def _spawn(job, args, runner) -> dict:
    """Run one worker. Returns {'payload':..., 'returncode':..., 'stderr':...}."""
    cmd = _worker_argv(job, args)
    env = os.environ.copy()
    env["PYTHONPATH"] = _SDK_PARENT + os.pathsep + env.get("PYTHONPATH", "")
    out = runner(cmd, capture_output=True, text=True, env=env)
    return dict(payload=parse_result_line(out.stdout or ""),
                returncode=out.returncode, stderr=out.stderr or "",
                stdout=out.stdout or "", argv=cmd)


def _fail(msg, lines=()) -> int:
    print(f"\nABORT: {msg}")
    for line in lines:
        print(f"  {line}")
    print("No aggregate is produced: a benchmark that drops a failed worker "
          "and averages the rest publishes a biased number.")
    return 1


def _describe(job) -> str:
    tag = "warmup" if job["warmup"] else f"rep{job['repeat']} slot{job['slot']}"
    return f"{job['config_id']} {tag} {job['mode']}"


def _check_worker(job, res) -> "str | None":
    """None if the worker is acceptable, else a message naming what failed.

    Order is deliberate: completeness, then the job/payload identity, then the
    x-margin gate, then the worker's own verdicts — so a run that outgrew the
    plate is named by the margin rather than by (d) firing at the edge, and a
    stale or mis-dispatched worker is named before either.
    """
    if res["payload"] is None:
        return (f"{_describe(job)}: no RESULT_JSON in stdout "
                f"(exit {res['returncode']})")
    p = res["payload"]
    gone = missing_payload_keys(p)
    if gone:
        return (f"{_describe(job)}: RESULT_JSON is missing {gone} — this "
                f"worker is not the dual_scene_terrain.run() this harness "
                f"needs (pre-v1.6.1 on PYTHONPATH?)")
    # The payload must be the job that was dispatched: paired_ratios() pairs on
    # the PAYLOAD's mode/repeat, so a worker that reported someone else's
    # identity would silently produce a ratio between the wrong two runs.
    for key in ("mode", "repeat", "slot", "config_id", "warmup"):
        want, got = job[key], p[key]
        if isinstance(want, bool):
            same = bool(got) is want
        elif isinstance(want, int):
            same = isinstance(got, int) and got == want
        else:
            same = got == want
        if not same:
            return (f"{_describe(job)}: worker reported {key}={got!r} but was "
                    f"dispatched with {key}={want!r} — refusing to pair "
                    f"mislabelled measurements")
    ok, detail = x_margin_check(p)
    if not ok:
        return (f"{_describe(job)}: x-margin gate FAILED — {detail}. "
                f"Lower --drive-s or raise --terrain-size; do not raise the "
                f"gate.")
    bad = [(label, detail) for label, okv, detail in p["verdicts"] if not okv]
    if bad:
        first = bad[0]
        return (f"{_describe(job)}: run INVALID — {first[0]}: {first[1]}"
                + (f"  (+{len(bad) - 1} more failing criteria)"
                   if len(bad) > 1 else ""))
    if res["returncode"] != 0:
        return (f"{_describe(job)}: worker exited {res['returncode']} although "
                f"its payload passed every criterion")
    return None


# Everything the conditions block prints from ONE worker must be equal across
# ALL of them, or the printed conditions would describe only that worker.
_GLOBAL_INVARIANTS = (
    "genesis_version", "sdk_version", "python_version", "backend",
    "dt", "substeps", "terrain_size", "terrain_half",
    "settle_s", "drive_s", "throttle", "steer",
    "warmup_steps", "n_timed_steps", "timing_definition",
)


def _check_invariants(samples) -> "str | None":
    """Cross-worker equality invariants.

    ``speed_at_window_start`` is NOT here on purpose: it differs slightly
    between the modes (measured 0.332 dual vs 0.329 single m/s), so it is
    reported as a min..max envelope instead of asserted equal.
    """
    for key in _GLOBAL_INVARIANTS:
        vals = {json.dumps(s.get(key)) for s in samples}
        if len(vals) != 1:
            return (f"workers disagree on {key}: {sorted(vals)} — they are not "
                    f"measuring the same thing")
    by_config: dict = {}
    for s in samples:
        by_config.setdefault(s["config_id"], set()).add(int(s["faces"]))
    for cid, faces in by_config.items():
        if len(faces) != 1:
            return f"workers in config {cid} disagree on faces: {sorted(faces)}"
    return None


def _sig2(x) -> str:
    return f"{float(x):.2g}"


def _conditions(samples) -> dict:
    """The conditions block. EVERY value here is read from a worker payload,
    except the four parent-owned entries flagged below (harness_version,
    repeats, warmup_workers, config_ids). Each worker-read entry is also a
    cross-worker equality invariant (_GLOBAL_INVARIANTS) or a min..max envelope
    over all workers, so taking it from samples[0] cannot misdescribe the
    others."""
    s0 = samples[0]
    starts = [s["speed_at_window_start"] for s in samples]
    ends = [s["speed_at_window_end"] for s in samples]
    return {
        "genesis_version": s0["genesis_version"],
        "sdk_version": s0["sdk_version"],
        "python_version": s0["python_version"],
        "backend": s0["backend"],
        "dt": s0["dt"], "substeps": s0["substeps"],
        "terrain_size": s0["terrain_size"], "terrain_half": s0["terrain_half"],
        "settle_s": s0["settle_s"], "drive_s": s0["drive_s"],
        "throttle": s0["throttle"], "steer": s0["steer"],
        "warmup_steps_per_worker": s0["warmup_steps"],
        "n_timed_steps": s0["n_timed_steps"],
        "timing_definition": s0["timing_definition"],
        "window_speed": {
            "start_min": min(starts), "start_max": max(starts),
            "end_min": min(ends), "end_max": max(ends),
            "note": (f"constant throttle {s0['throttle']:g}, steer "
                     f"{s0['steer']:g}; the window is an acceleration, so the "
                     f"vehicle state changes across it"),
        },
        # --- parent-owned (the only values in this block not read from a
        # --- worker): the schedule the parent chose and its own identity.
        "harness_version": HARNESS_VERSION,
        "repeats": None,          # filled by the caller
        "warmup_workers": None,   # filled by the caller
        "config_ids": None,       # filled by the caller
    }


def _print_conditions(cond, per_config):
    print("\nCONDITIONS (read from the worker payloads unless marked parent):")
    print(f"  genesis-world {cond['genesis_version']}  |  SDK "
          f"v{cond['sdk_version']}  |  python {cond['python_version']}  |  "
          f"backend {cond['backend']}")
    print(f"  dt={cond['dt']} substeps={cond['substeps']}  |  terrain "
          f"{cond['terrain_size'][0]:g}x{cond['terrain_size'][1]:g} m "
          f"(half {cond['terrain_half'][0]:g} m)  |  settle "
          f"{cond['settle_s']:g} s, drive {cond['drive_s']:g} s")
    print(f"  timed window: {cond['n_timed_steps']} steps after "
          f"{cond['warmup_steps_per_worker']} warm-up steps; ms/step = "
          f"{cond['timing_definition']}")
    w = cond["window_speed"]
    print(f"  timed window covers v = {_sig2(w['start_min'])} .. "
          f"{_sig2(w['end_max'])} m/s under constant throttle "
          f"{cond['throttle']:g}, steer {cond['steer']:g}")
    print(f"  (parent) repeats={cond['repeats']}  warm-up workers="
          f"{cond['warmup_workers']}  configs={cond['config_ids']}  "
          f"harness={cond['harness_version']}")
    for cid, c in per_config.items():
        print(f"  [{cid}] horizontal_scale={c['horizontal_scale']:g}  "
              f"n_envs={c['n_envs']}  faces={c['faces']}")


def _aggregate(samples, repeats) -> dict:
    """Per-config aggregates. Warm-up samples are excluded from every statistic
    (they are kept in the JSON so the exclusion is auditable)."""
    aggregates: dict = {}
    for s in samples:
        if s.get("warmup"):
            continue
        aggregates.setdefault(s["config_id"], []).append(s)
    out: dict = {}
    for cid, rows in aggregates.items():
        per_mode = {}
        for mode in MODES:
            vals = [float(r["ms"]) for r in rows if r["mode"] == mode]
            per_mode[mode] = _interval(vals) if vals else None
        for slot in (1, 2):
            for mode in MODES:
                vals = [float(r["ms"]) for r in rows
                        if r["mode"] == mode and int(r["slot"]) == slot]
                per_mode[f"{mode}@slot{slot}"] = _interval(vals) if vals else None
        paired = paired_ratios(rows)
        ga, gb = split_groups(paired)
        decision = publication_decision(paired, ga, gb)
        pooled = _interval([r for _, r in paired])
        out[cid] = dict(
            conditions=dict(horizontal_scale=rows[0]["horizontal_scale"],
                            n_envs=rows[0]["n_envs"], faces=rows[0]["faces"]),
            n_samples=len(rows), repeats=repeats,
            per_mode=per_mode,
            paired_r=[dict(repeat=k, r=r,
                           slot1_mode=("dual_scene" if k % 2 == 0
                                       else "single_scene")) for k, r in paired],
            paired_median=pooled["median"], paired_min=pooled["min"],
            paired_max=pooled["max"], paired_n=pooled["n"],
            groupA=(_interval(ga) if ga else None),
            groupB=(_interval(gb) if gb else None),
            decision=decision,
            ratio=(dict(pooled) if decision["publishable"] else None),
        )
    return out


def _print_aggregate(cid, agg):
    print(f"\n=== RESULT [{cid}] ===")
    print(f"  per-mode ms/step (n={agg['n_samples'] // 2} each):")
    for mode in MODES:
        i = agg["per_mode"][mode]
        print(f"    {mode:<13} median {i['median']:7.3f}  "
              f"[{i['min']:7.3f} .. {i['max']:7.3f}]  n={i['n']}")
        for slot in (1, 2):
            si = agg["per_mode"][f"{mode}@slot{slot}"]
            if si:
                print(f"      slot{slot}      median {si['median']:7.3f}  "
                      f"[{si['min']:7.3f} .. {si['max']:7.3f}]  n={si['n']}")
    print("  paired r_k = ms(single_scene) / ms(dual_scene), per repeat:")
    for row in agg["paired_r"]:
        print(f"    repeat {row['repeat']}  r={row['r']:.4f}   "
              f"(slot 1 was {row['slot1_mode']})")
    print(f"  pooled  : median {agg['paired_median']:.3f} "
          f"[{agg['paired_min']:.3f} .. {agg['paired_max']:.3f}] "
          f"n={agg['paired_n']}")
    for name, g, who in (("group A", agg["groupA"], "dual_scene ran slot 1"),
                         ("group B", agg["groupB"], "single_scene ran slot 1")):
        if g:
            print(f"  {name} : median {g['median']:.3f} "
                  f"[{g['min']:.3f} .. {g['max']:.3f}] n={g['n']}   ({who})")
    d = agg["decision"]
    print(f"  rule (i)  mutual group inclusion : "
          f"{'PASS' if d['cond_i'] else 'FAIL'}  — {d['detail_i']}")
    print(f"  rule (ii) 1.0 outside the range  : "
          f"{'PASS' if d['cond_ii'] else 'FAIL'}  — {d['detail_ii']}")
    if d["publishable"]:
        print(f"  RATIO: {format_ratio_line(agg['ratio'])}")
        print("  (quote it WITH the interval, the n and the conditions block "
              "below; it is not a confidence interval and not a verdict)")
    else:
        print(f"  NO RATIO — not measurable on this machine at these settings "
              f"({d['reason']}).")
        print("  The interval above is the whole result. Do not quote a median "
              "from it, and do not re-run until it publishes.")


def main(argv=None, runner=None) -> int:
    ap = argparse.ArgumentParser(
        description="Order-independent, paired, fail-closed benchmark of the "
                    "dual_scene vs single_scene wheel raycast modes.")
    ap.add_argument("--terrain-size", type=float, default=640.0,
                    help="Square plate side in metres (default 640). Big "
                         "enough that a 20 s constant-throttle window stays on "
                         "it: measured x_end=208 m against a (d) bound of 318 "
                         "m. terrain-size/horizontal-scale must be a whole "
                         "number of cells.")
    ap.add_argument("--horizontal-scale", default="4.0",
                    help="Comma-separated terrain cell sizes in metres "
                         "(default 4.0 -> 51,212 faces at 640 m). Each value "
                         "is a separate config; halving it quadruples the "
                         "faces AND the run time.")
    ap.add_argument("--n-envs", default="1",
                    help="Comma-separated L3 batch sizes (default 1). Each "
                         "value is a separate config.")
    ap.add_argument("--repeats", type=int, default=6,
                    help="Paired repeats per config (default 6). Must be EVEN "
                         "and >= 4.")
    ap.add_argument("--settle-s", type=float, default=3.0,
                    help="Seconds on the brake before the drive (default 3).")
    ap.add_argument("--drive-s", type=float, default=20.0,
                    help="Seconds of the TIMED constant-throttle window "
                         "(default 20 = 800 steps). Raising it without raising "
                         "--terrain-size trips the x-margin gate.")
    ap.add_argument("--no-warmup", action="store_true",
                    help="Skip the single discarded warm-up worker. It exists "
                         "because the FIRST process of a session MAY pay a "
                         "cold-start cost (page cache, CPU frequency); running "
                         "one worker that lands in no statistic means that "
                         "cost, IF it exists, is not in the numbers. It is "
                         "gated like any other worker and is kept in the JSON "
                         "with warmup=true so a reader can check whether it "
                         "existed. In the two full runs shipped with this "
                         "change (--drive-s 20, this machine) it did not: the "
                         "cold worker took 19.33 s against 16.91..22.23 s for "
                         "the warm ones, and 16.54 s against 16.43..22.58 s — "
                         "inside the warm range both times. (An earlier version "
                         "of this help paired '~19 s cold vs ~11 s warm'. That "
                         "was WRONG: the two numbers came from different "
                         "--drive-s settings and were never comparable.)")
    ap.add_argument("--gpu", action="store_true",
                    help="Use the GPU backend. UNVERIFIED: every number and "
                         "default in this harness was measured on CPU.")
    ap.add_argument("--json", dest="json_path", default=None,
                    help="Write the full record (samples, conditions, "
                         "aggregates) to this path.")
    # --- worker-only ---
    ap.add_argument("--_worker", dest="worker", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--mode", choices=list(MODES), default=MODES[0],
                    help=argparse.SUPPRESS)
    ap.add_argument("--repeat", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--slot", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--config-id", default="", help=argparse.SUPPRESS)
    ap.add_argument("--warmup", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    args.horizontal_scale = [float(x) for x in
                             str(args.horizontal_scale).split(",") if x.strip()]
    args.n_envs = [int(x) for x in str(args.n_envs).split(",") if x.strip()]

    if args.worker:
        return _worker_main(args)

    if args.repeats % 2 != 0 or args.repeats < 4:
        # argparse.error exits before a single worker is spawned.
        ap.error(
            f"--repeats must be EVEN and >= 4 (got {args.repeats}). The "
            f"schedule alternates which mode occupies slot 1; with an odd "
            f"count one mode gets an extra slot-1 run and the slot bias stops "
            f"cancelling, which is the exact defect this harness exists to "
            f"remove.")

    configs = [(hs, n) for hs in args.horizontal_scale for n in args.n_envs]
    jobs = build_schedule(configs, args.repeats)
    if not args.no_warmup:
        first = dict(jobs[0])
        first.update(warmup=True, repeat=-1, slot=0)
        jobs = [first] + jobs

    runner = runner or subprocess.run

    n_workers = len(jobs)
    print(f"bench_raycast_mode ({HARNESS_VERSION}) — "
          f"{len(configs)} config(s) x {args.repeats} paired repeats x 2 modes"
          f"{'' if args.no_warmup else ' + 1 warm-up'} = {n_workers} fresh "
          f"processes")
    print(f"  configs : {[config_id_of(hs, n) for hs, n in configs]}")
    print(f"  plate   : {args.terrain_size:g} x {args.terrain_size:g} m, "
          f"settle {args.settle_s:g} s, timed drive {args.drive_s:g} s")
    print(f"  ESTIMATE ONLY (not a measurement of this machine; from a CPU/WSL2 "
          f"probe at 640 m / hs 4.0: ~11 s fixed + ~7 ms/step): "
          f"~{n_workers * (11.0 + (130 + 40 * args.drive_s) * 0.007) / 60:.1f} "
          f"min. Halving --horizontal-scale quadruples the faces and can push "
          f"this past 10 min.")
    print()

    samples = []
    for job in jobs:
        print(f"  [{_describe(job)}] spawning...", flush=True)
        res = _spawn(job, args, runner)
        bad = _check_worker(job, res)
        if bad is not None:
            lines = []
            if res["payload"] is None and res["stderr"].strip():
                lines = ["stderr: " + ln for ln in
                         res["stderr"].strip().splitlines()[-5:]]
            return _fail(bad, lines)
        p = res["payload"]
        print(f"      {p['ms']:.3f} ms/step   x_end={p['x']:+.2f} m   "
              f"v {p['speed_at_window_start']:.3f} -> "
              f"{p['speed_at_window_end']:.2f} m/s   faces={p['faces']}   "
              f"wall {p['wall_s']:.1f} s", flush=True)
        samples.append(p)

    inv = _check_invariants(samples)
    if inv is not None:
        return _fail(inv)

    measured = [s for s in samples if not s.get("warmup")]
    if not measured:
        return _fail("no non-warm-up samples")

    cond = _conditions(measured)
    cond["repeats"] = args.repeats
    cond["warmup_workers"] = sum(1 for s in samples if s.get("warmup"))
    cond["config_ids"] = [config_id_of(hs, n) for hs, n in configs]

    aggregates = _aggregate(samples, args.repeats)
    for cid in cond["config_ids"]:
        if cid in aggregates:
            _print_aggregate(cid, aggregates[cid])
    _print_conditions(cond, {cid: aggregates[cid]["conditions"]
                             for cid in aggregates})

    record = dict(harness=HARNESS_VERSION, conditions=cond,
                  samples=samples, aggregates=aggregates)
    if args.json_path:
        with open(args.json_path, "w") as fh:
            json.dump(record, fh, indent=1, sort_keys=True)
        print(f"\nwrote {args.json_path}  "
              f"({len(samples)} samples incl. "
              f"{cond['warmup_workers']} warm-up)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
