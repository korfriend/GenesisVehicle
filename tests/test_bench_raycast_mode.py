"""Unit tests for samples/bench_raycast_mode.py — the raycast-mode benchmark.

Everything here is pure python: no ``gs.init``, no ``VehicleScene``, no
timing. The subprocess layer is exercised through the harness's own injectable
``runner=`` seam (the shipped ``main(argv, runner=...)`` entry point), so the
tests drive the SAME code path a real run does, minus the child processes.

| abbr | meaning |
|---|---|
| r_k | paired ratio of repeat k: ms(single_scene, k) / ms(dual_scene, k) |
| A / B | repeat groups by slot order: A = dual_scene ran first, B = single_scene ran first |
| ms/step | wall-clock milliseconds per simulation step |
"""

import json
import types

import pytest

from genesis_vehicle.samples import bench_raycast_mode as B


# ---------------------------------------------------------------------------
# helpers — a payload shaped exactly like a real worker's RESULT_JSON
# ---------------------------------------------------------------------------

def _payload(mode="dual_scene", repeat=0, slot=1, ms=7.0, warmup=False,
             config_id="hs4_env1", x=208.0, valid=True, **over):
    p = dict(
        mode=mode, repeat=repeat, slot=slot, config_id=config_id,
        warmup=warmup, ms=ms, faces=51212,
        x=x, y=0.02, z=0.111, speed=18.86,
        x_min=x, x_max=x, y_min=0.02, y_max=0.02, z_min=0.111, z_max=0.111,
        d0_min=0.411, d0_max=0.412, z_settle=0.1116, grounded=True,
        dt=0.025, substeps=10,
        terrain_size=[640.0, 640.0], terrain_half=[320.0, 320.0],
        settle_s=3.0, drive_s=20.0, throttle=0.6, steer=0.0,
        n_envs=1, backend="cpu",
        horizontal_scale=4.0, genesis_version="1.4.0", sdk_version="1.6.1",
        warmup_steps=10, n_timed_steps=800,
        timing_definition="wall time of the full vs.step() loop",
        x_at_window_start=0.011, speed_at_window_start=0.332,
        speed_at_window_end=18.86,
        python_version="3.10.0", wall_s=17.5, t_start=0.0, t_end=17.5,
        harness_version=B.HARNESS_VERSION, argv=[],
        verdicts=[["(a) all wheel rays hit terrain", valid, "d0 ..."],
                  ["(b) chassis at ride height, stable", True, "z ..."],
                  ["(c) drove forward", True, "x ..."],
                  ["(d) stayed within terrain bounds", True, "|x|max ..."]],
        valid=valid,
    )
    p.update(over)
    p["valid"] = all(ok for _, ok, _ in p["verdicts"])
    return p


def _fake_runner(ms_for, payload_hook=None, calls=None):
    """A stand-in for subprocess.run driven by ms_for(mode, repeat, warmup)."""
    def run(cmd, capture_output=True, text=True, env=None):
        argv = cmd[cmd.index("--_worker"):]
        def val(flag):
            return argv[argv.index(flag) + 1]
        mode = val("--mode")
        repeat = int(val("--repeat"))
        warmup = "--warmup" in argv
        p = _payload(mode=mode, repeat=repeat, slot=int(val("--slot")),
                     config_id=val("--config-id"), warmup=warmup,
                     ms=ms_for(mode, repeat, warmup))
        if payload_hook is not None:
            p = payload_hook(p)
        if calls is not None:
            calls.append(dict(cmd=cmd, mode=mode, repeat=repeat,
                              slot=int(val("--slot")), warmup=warmup))
        out = ("[Genesis] some banner\n"
               + B._RESULT_PREFIX + json.dumps(p) + "\n")
        return types.SimpleNamespace(returncode=0 if p["valid"] else 1,
                                     stdout=out, stderr="")
    return run


# ---------------------------------------------------------------------------
# build_schedule
# ---------------------------------------------------------------------------

def test_schedule_slot_balance_and_adjacency():
    jobs = B.build_schedule([(4.0, 1)], 6)
    assert len(jobs) == 12
    for mode in B.MODES:
        for slot in (1, 2):
            n = sum(1 for j in jobs if j["mode"] == mode and j["slot"] == slot)
            assert n == 3, (mode, slot, n)
    # the two workers of a repeat are ADJACENT and are the two modes
    for i in range(0, 12, 2):
        a, b = jobs[i], jobs[i + 1]
        assert a["repeat"] == b["repeat"]
        assert (a["slot"], b["slot"]) == (1, 2)
        assert {a["mode"], b["mode"]} == set(B.MODES)
    # even repeats put dual_scene first, odd repeats single_scene first
    assert [j["mode"] for j in jobs if j["slot"] == 1] == [
        "dual_scene", "single_scene"] * 3


def test_schedule_does_not_interleave_configs():
    jobs = B.build_schedule([(4.0, 1), (2.0, 1)], 4)
    assert len(jobs) == 16
    ids = [j["config_id"] for j in jobs]
    assert ids == ["hs4_env1"] * 8 + ["hs2_env1"] * 8


@pytest.mark.parametrize("repeats", [0, 1, 2, 3, 5, 7])
def test_schedule_rejects_odd_or_too_few(repeats):
    with pytest.raises(ValueError):
        B.build_schedule([(4.0, 1)], repeats)


@pytest.mark.parametrize("repeats", ["1", "3"])
def test_cli_rejects_odd_repeats_before_any_worker(repeats, capsys):
    calls = []
    runner = _fake_runner(lambda m, k, w: 7.0, calls=calls)
    with pytest.raises(SystemExit) as exc:
        B.main(["--repeats", repeats], runner=runner)
    assert exc.value.code != 0
    msg = capsys.readouterr().err.lower()
    assert "even" in msg and "slot" in msg
    assert calls == []          # zero workers spawned


def test_cli_accepts_repeats_4():
    calls = []
    runner = _fake_runner(lambda m, k, w: 7.0 + (0.5 if m == "single_scene" else 0),
                          calls=calls)
    assert B.main(["--repeats", "4", "--no-warmup"], runner=runner) == 0
    assert len(calls) == 8


# ---------------------------------------------------------------------------
# paired ratios / grouping
# ---------------------------------------------------------------------------

def test_paired_ratios_pairs_within_a_repeat():
    rows = []
    for k in range(4):
        rows.append(_payload(mode="dual_scene", repeat=k, ms=10.0))
        rows.append(_payload(mode="single_scene", repeat=k, ms=11.0 + k))
    paired = B.paired_ratios(rows)
    assert [k for k, _ in paired] == [0, 1, 2, 3]
    assert [round(r, 4) for _, r in paired] == [1.1, 1.2, 1.3, 1.4]
    a, b = B.split_groups(paired)
    assert [round(x, 4) for x in a] == [1.1, 1.3]
    assert [round(x, 4) for x in b] == [1.2, 1.4]


def test_paired_ratios_reject_incomplete_pair():
    rows = [_payload(mode="dual_scene", repeat=0, ms=10.0)]
    with pytest.raises(ValueError):
        B.paired_ratios(rows)


def test_paired_ratios_ignore_warmup():
    rows = [_payload(mode="dual_scene", repeat=-1, ms=99.0, warmup=True)]
    for k in range(2):
        rows.append(_payload(mode="dual_scene", repeat=k, ms=10.0))
        rows.append(_payload(mode="single_scene", repeat=k, ms=12.0))
    assert [k for k, _ in B.paired_ratios(rows)] == [0, 1]


# ---------------------------------------------------------------------------
# fail-closed publication rule
# ---------------------------------------------------------------------------

def _decide(rs):
    paired = list(enumerate(rs))
    a, b = B.split_groups(paired)
    return B.publication_decision(paired, a, b), paired


def test_publish_when_groups_overlap_and_1_excluded():
    # A (even k) = 1.09, 1.12, 1.14 ; B (odd k) = 1.10, 1.13, 1.11 — interleaved,
    # so neither group's median falls outside the other's range.
    d, paired = _decide([1.09, 1.10, 1.12, 1.13, 1.14, 1.11])
    assert d["publishable"] is True
    assert d["cond_i"] and d["cond_ii"]
    iv = B._interval([r for _, r in paired])
    assert iv["n"] == 6
    assert iv["min"] == 1.09 and iv["max"] == 1.14
    line = B.format_ratio_line(iv)
    assert "1.090" in line and "1.140" in line and "n=6" in line


def test_suppressed_when_groups_separate():
    # A (even k) all near 1.05, B (odd k) all near 1.30 — a slot effect.
    d, paired = _decide([1.04, 1.29, 1.05, 1.30, 1.06, 1.31])
    assert d["cond_ii"] is True          # 1.0 is still outside the range
    assert d["cond_i"] is False
    assert d["publishable"] is False
    assert "(i)" in d["reason"]
    # the interval is still computable — suppression hides the RATIO, not the data
    iv = B._interval([r for _, r in paired])
    assert iv["n"] == 6 and iv["min"] == 1.04 and iv["max"] == 1.31


def test_suppressed_when_range_straddles_one():
    # groups overlap (A = 0.98, 1.03, 1.00 ; B = 1.01, 0.99, 1.02) but the
    # pooled range contains 1.0, so the SIGN of the effect is not established.
    d, _ = _decide([0.98, 1.01, 1.03, 0.99, 1.00, 1.02])
    assert d["cond_i"] is True
    assert d["cond_ii"] is False
    assert d["publishable"] is False
    assert "(ii)" in d["reason"]


def test_suppressed_when_both_conditions_fail():
    d, _ = _decide([0.90, 1.30, 0.92, 1.31, 0.94, 1.32])
    assert not d["cond_i"] and not d["cond_ii"]
    assert d["publishable"] is False
    assert "(i)" in d["reason"] and "(ii)" in d["reason"]


def test_ratio_below_one_can_publish():
    d, paired = _decide([0.89, 0.90, 0.92, 0.93, 0.94, 0.91])
    assert d["publishable"] is True
    assert B._interval([r for _, r in paired])["median"] < 1.0


def test_format_ratio_line_always_has_interval_and_n():
    line = B.format_ratio_line(dict(median=1.234, min=1.1, max=1.4, n=6))
    assert line.startswith("single/dual = median 1.234")
    assert "[1.100 .. 1.400]" in line and "n=6" in line


# ---------------------------------------------------------------------------
# parse_result_line
# ---------------------------------------------------------------------------

def test_parse_result_line_takes_the_last_one_among_noise():
    out = ("[Genesis] warning blah\n"
           + B._RESULT_PREFIX + json.dumps({"ms": 1.0}) + "\n"
           "[Genesis] more noise\n"
           + B._RESULT_PREFIX + json.dumps({"ms": 2.0}) + "\n"
           "trailing noise\n")
    assert B.parse_result_line(out)["ms"] == 2.0


def test_parse_result_line_none_when_absent_or_broken():
    assert B.parse_result_line("nothing here\n") is None
    assert B.parse_result_line(B._RESULT_PREFIX + "{not json\n") is None


# ---------------------------------------------------------------------------
# runner protocol / parent contract
# ---------------------------------------------------------------------------

def test_runner_called_once_per_config_repeat_mode_with_expected_argv():
    calls = []
    runner = _fake_runner(lambda m, k, w: 7.0 + (0.4 if m == "single_scene" else 0.0),
                          calls=calls)
    assert B.main(["--repeats", "6", "--no-warmup"], runner=runner) == 0
    assert len(calls) == 12
    seen = {(c["mode"], c["repeat"]) for c in calls}
    assert seen == {(m, k) for m in B.MODES for k in range(6)}
    for c in calls:
        for flag in ("--_worker", "--mode", "--repeat", "--slot",
                     "--config-id", "--terrain-size", "--drive-s"):
            assert flag in c["cmd"]
    # the alternation the parent asked for matches build_schedule
    assert [(c["mode"], c["slot"]) for c in calls] == [
        (j["mode"], j["slot"]) for j in B.build_schedule([(4.0, 1)], 6)]


def test_warmup_worker_is_run_and_excluded_from_statistics(capsys):
    calls = []
    # give the warm-up an absurd ms so its inclusion would be obvious
    runner = _fake_runner(
        lambda m, k, w: 99.0 if w else (7.0 + (0.4 if m == "single_scene" else 0.0)),
        calls=calls)
    assert B.main(["--repeats", "4"], runner=runner) == 0
    assert len(calls) == 9
    assert sum(1 for c in calls if c["warmup"]) == 1
    assert calls[0]["warmup"] is True
    out = capsys.readouterr().out
    assert "99.000" not in out.split("=== RESULT")[1]


def test_parent_never_calls_gs_init(monkeypatch):
    """The parent path must not reach gs.init or VehicleScene.

    This proves the PATH, not that genesis is absent from the process: the SDK
    package imports genesis.utils.geom eagerly, so absence is impossible.
    """
    import genesis as gs

    def boom(*a, **k):
        raise AssertionError("parent called gs.init")

    monkeypatch.setattr(gs, "init", boom)
    runner = _fake_runner(lambda m, k, w: 7.0 + (0.4 if m == "single_scene" else 0.0))
    assert B.main(["--repeats", "4", "--no-warmup"], runner=runner) == 0


# ---------------------------------------------------------------------------
# validity gates
# ---------------------------------------------------------------------------

def test_invalid_worker_aborts_with_the_criterion_named(capsys):
    def hook(p):
        if p["mode"] == "single_scene" and p["repeat"] == 1:
            p["verdicts"][0][1] = False
            p["verdicts"][0][2] = "all_wheels_grounded=False  d0 min=20.0"
            p["valid"] = False
        return p
    runner = _fake_runner(lambda m, k, w: 7.0, payload_hook=hook)
    rc = B.main(["--repeats", "4", "--no-warmup"], runner=runner)
    assert rc != 0
    out = capsys.readouterr().out
    assert "ABORT" in out
    assert "(a) all wheel rays hit terrain" in out
    assert "rep1" in out and "single_scene" in out
    assert "=== RESULT" not in out          # no aggregate


def test_invalid_warmup_worker_aborts_too(capsys):
    def hook(p):
        if p["warmup"]:
            p["verdicts"][1][1] = False
            p["verdicts"][1][2] = "z_end=-84.2"
        return p
    runner = _fake_runner(lambda m, k, w: 7.0, payload_hook=hook)
    assert B.main(["--repeats", "4"], runner=runner) != 0
    out = capsys.readouterr().out
    assert "warmup" in out and "(b) chassis at ride height" in out
    assert "=== RESULT" not in out


def test_x_margin_gate_fires_before_verdicts(capsys):
    # 250 m is inside the (d) bound of 318 m but outside 0.75 x 318 = 238.5 m,
    # so every criterion still passes and only the margin gate can catch it.
    def hook(p):
        p.update(x=250.0, x_min=250.0, x_max=250.0)
        return p
    runner = _fake_runner(lambda m, k, w: 7.0, payload_hook=hook)
    assert B.main(["--repeats", "4", "--no-warmup"], runner=runner) != 0
    out = capsys.readouterr().out
    assert "x-margin gate FAILED" in out
    assert "238.50" in out
    assert "=== RESULT" not in out


def test_x_margin_check_uses_the_env_envelope_not_env_zero():
    p = _payload()
    p.update(x=10.0, x_min=10.0, x_max=300.0)     # one stray env in an L3 batch
    ok, detail = B.x_margin_check(p)
    assert ok is False and "300.00" in detail


def test_missing_result_line_aborts(capsys):
    def runner(cmd, capture_output=True, text=True, env=None):
        return types.SimpleNamespace(returncode=1, stdout="boom\n",
                                     stderr="Traceback\nRuntimeError: nope\n")
    assert B.main(["--repeats", "4", "--no-warmup"], runner=runner) != 0
    out = capsys.readouterr().out
    assert "no RESULT_JSON" in out and "RuntimeError: nope" in out
    assert "=== RESULT" not in out


@pytest.mark.parametrize("key,bad", [("genesis_version", "1.3.3"),
                                     ("sdk_version", "1.5.2"),
                                     ("python_version", "3.11.9"),
                                     ("backend", "gpu"),
                                     ("dt", 0.02),
                                     ("substeps", 4),
                                     ("terrain_size", [40.0, 40.0]),
                                     ("terrain_half", [20.0, 20.0]),
                                     ("settle_s", 1.0),
                                     ("drive_s", 8.0),
                                     ("throttle", 0.4),
                                     ("steer", 0.1),
                                     ("warmup_steps", 30),
                                     ("n_timed_steps", 320),
                                     ("timing_definition", "something else")])
def test_cross_worker_invariants_abort(key, bad, capsys):
    def hook(p):
        if p["mode"] == "single_scene" and p["repeat"] == 3:
            p[key] = bad
            if key == "terrain_half":
                p.update(x=10.0, x_min=10.0, x_max=10.0)   # keep the margin OK
        return p
    runner = _fake_runner(lambda m, k, w: 7.0, payload_hook=hook)
    assert B.main(["--repeats", "4", "--no-warmup"], runner=runner) != 0
    out = capsys.readouterr().out
    assert f"disagree on {key}" in out
    assert "=== RESULT" not in out


def test_faces_mismatch_within_a_config_aborts(capsys):
    def hook(p):
        if p["repeat"] == 2:
            p["faces"] = 12800
        return p
    runner = _fake_runner(lambda m, k, w: 7.0, payload_hook=hook)
    assert B.main(["--repeats", "4", "--no-warmup"], runner=runner) != 0
    assert "disagree on faces" in capsys.readouterr().out


def test_speed_at_window_start_is_not_an_equality_invariant():
    """It differs between modes in practice (0.332 dual vs 0.329 single) and
    must NOT abort the run; it is reported as a min..max envelope."""
    def hook(p):
        p["speed_at_window_start"] = 0.329 if p["mode"] == "single_scene" else 0.332
        return p
    runner = _fake_runner(lambda m, k, w: 7.0 + (0.4 if m == "single_scene" else 0.0),
                          payload_hook=hook)
    assert B.main(["--repeats", "4", "--no-warmup"], runner=runner) == 0


# ---------------------------------------------------------------------------
# printed output + JSON record
# ---------------------------------------------------------------------------

def _run_to_json(tmp_path, ms_for, extra=()):
    path = tmp_path / "bench.json"
    runner = _fake_runner(ms_for)
    rc = B.main(["--repeats", "6", "--json", str(path)] + list(extra),
                runner=runner)
    return rc, json.loads(path.read_text())


def test_always_prints_r_k_pooled_and_groups_even_when_suppressed(tmp_path, capsys):
    # straddles 1.0 -> suppressed
    def ms_for(mode, k, warmup):
        return 7.0 + (0.2 if (mode == "single_scene") == (k % 2 == 0) else -0.2)
    rc, rec = _run_to_json(tmp_path, ms_for)
    assert rc == 0
    out = capsys.readouterr().out
    assert "NO RATIO" in out and "not measurable" in out
    for k in range(6):
        assert f"repeat {k}  r=" in out
    assert "pooled  : median" in out and "n=6" in out
    assert "group A : median" in out and "group B : median" in out
    agg = rec["aggregates"]["hs4_env1"]
    assert agg["decision"]["publishable"] is False
    assert agg["ratio"] is None
    assert len(agg["paired_r"]) == 6


def test_json_record_contract_when_published(tmp_path):
    def ms_for(mode, k, warmup):
        return (7.0 + 0.01 * k) * (1.12 if mode == "single_scene" else 1.0)
    rc, rec = _run_to_json(tmp_path, ms_for)
    assert rc == 0
    samples = rec["samples"]
    assert len(samples) == 13                       # 12 + 1 warm-up
    assert sum(1 for s in samples if s["warmup"]) == 1
    measured = [s for s in samples if not s["warmup"]]
    for mode in B.MODES:
        for slot in (1, 2):
            assert sum(1 for s in measured
                       if s["mode"] == mode and s["slot"] == slot) == 3
    agg = rec["aggregates"]["hs4_env1"]
    assert agg["decision"]["publishable"] is True
    assert agg["ratio"] is not None
    assert agg["ratio"]["n"] == 6
    assert agg["ratio"]["min"] <= agg["ratio"]["median"] <= agg["ratio"]["max"]
    assert len(agg["paired_r"]) == 6
    cond = rec["conditions"]
    # parent-owned entries, and only these
    assert cond["repeats"] == 6 and cond["warmup_workers"] == 1
    assert cond["config_ids"] == ["hs4_env1"]
    assert cond["harness_version"] == B.HARNESS_VERSION
    # every other condition equals the value the worker reported
    w = measured[0]
    for key in ("genesis_version", "sdk_version", "python_version", "backend",
                "dt", "substeps", "terrain_size", "terrain_half", "settle_s",
                "drive_s", "throttle", "steer", "n_timed_steps",
                "timing_definition"):
        assert cond[key] == w[key], key
    assert cond["warmup_steps_per_worker"] == w["warmup_steps"]
    assert cond["window_speed"]["start_min"] == w["speed_at_window_start"]
    assert cond["window_speed"]["end_max"] == w["speed_at_window_end"]


def test_conditions_line_shows_the_window_speed_envelope(tmp_path, capsys):
    def hook_ms(mode, k, warmup):
        return 7.0 + (0.4 if mode == "single_scene" else 0.0)
    runner = _fake_runner(hook_ms)
    assert B.main(["--repeats", "4", "--no-warmup"], runner=runner) == 0
    out = capsys.readouterr().out
    assert ("timed window covers v = 0.33 .. 19 m/s under constant throttle "
            "0.6, steer 0") in out
    assert "genesis-world 1.4.0" in out
    assert "faces=51212" in out


def test_throttle_and_steer_are_read_from_the_worker(tmp_path, capsys):
    """The printed drive inputs must come from the payload, not a literal."""
    def hook(p):
        p.update(throttle=0.45, steer=0.02)
        return p
    runner = _fake_runner(lambda m, k, w: 7.0, payload_hook=hook)
    path = tmp_path / "b.json"
    assert B.main(["--repeats", "4", "--no-warmup", "--json", str(path)],
                  runner=runner) == 0
    out = capsys.readouterr().out
    assert "constant throttle 0.45, steer 0.02" in out
    cond = json.loads(path.read_text())["conditions"]
    assert cond["throttle"] == 0.45 and cond["steer"] == 0.02


def test_payload_missing_a_key_aborts_by_name(capsys):
    """A pre-v1.6.1 dual_scene_terrain on PYTHONPATH returns no conditions
    block; that must abort naming the key, not raise inside a gate."""
    def hook(p):
        p.pop("terrain_half")
        return p
    runner = _fake_runner(lambda m, k, w: 7.0, payload_hook=hook)
    assert B.main(["--repeats", "4", "--no-warmup"], runner=runner) != 0
    out = capsys.readouterr().out
    assert "missing ['terrain_half']" in out
    assert "pre-v1.6.1" in out
    assert "=== RESULT" not in out


def test_missing_payload_keys_lists_every_absent_key():
    p = _payload()
    for k in ("throttle", "n_timed_steps", "wall_s"):
        p.pop(k)
    assert B.missing_payload_keys(p) == ["throttle", "n_timed_steps", "wall_s"]
    assert B.missing_payload_keys(_payload()) == []


@pytest.mark.parametrize("key,bad", [("mode", "dual_scene"),
                                     ("repeat", 5),
                                     ("slot", 2),
                                     ("config_id", "hs2_env1")])
def test_worker_reporting_the_wrong_job_aborts(key, bad, capsys):
    """paired_ratios() pairs on the PAYLOAD's labels, so a worker that reports
    someone else's identity must abort rather than pair the wrong two runs."""
    def hook(p):
        if p["mode"] == "single_scene" and p["repeat"] == 1:
            p[key] = bad
        return p
    runner = _fake_runner(lambda m, k, w: 7.0, payload_hook=hook)
    assert B.main(["--repeats", "4", "--no-warmup"], runner=runner) != 0
    out = capsys.readouterr().out
    assert f"worker reported {key}=" in out and "was dispatched with" in out
    assert "=== RESULT" not in out


def test_warmup_flag_mismatch_aborts(capsys):
    def hook(p):
        p["warmup"] = True          # a measured worker claiming to be warm-up
        return p
    runner = _fake_runner(lambda m, k, w: 7.0, payload_hook=hook)
    assert B.main(["--repeats", "4", "--no-warmup"], runner=runner) != 0
    assert "worker reported warmup=True" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# dual_scene_terrain backward compatibility (the harness depends on it)
# ---------------------------------------------------------------------------

def test_verdicts_without_terrain_half_uses_the_default_plate():
    from genesis_vehicle.samples import dual_scene_terrain as dst
    r = dict(grounded=True, d0_min=0.411, d0_max=0.412, z=0.111, z_settle=0.112,
             z_min=0.111, z_max=0.111, x=10.342, speed=4.83,
             x_min=10.342, x_max=10.342, y_min=-0.022, y_max=-0.022)
    detail = dict((lab, det) for lab, _, det in dst.verdicts(r))
    d = detail["(d) stayed within terrain bounds"]
    assert "need < 18.0; terrain spans ±20 m about the origin" in d
    assert all(ok for _, ok, _ in dst.verdicts(r))


def test_verdicts_with_terrain_half_uses_the_run_plate():
    from genesis_vehicle.samples import dual_scene_terrain as dst
    r = dict(grounded=True, d0_min=0.411, d0_max=0.412, z=0.111, z_settle=0.112,
             z_min=0.111, z_max=0.111, x=208.149, speed=18.86,
             x_min=208.149, x_max=208.149, y_min=0.026, y_max=0.026,
             terrain_half=(320.0, 320.0))
    detail = dict((lab, det) for lab, _, det in dst.verdicts(r))
    d = detail["(d) stayed within terrain bounds"]
    assert "need < 318.0; terrain spans ±320 m about the origin" in d
    assert all(ok for _, ok, _ in dst.verdicts(r))
    # ... and the same pose on the DEFAULT plate must fail (d)
    r.pop("terrain_half")
    ok_d = dict((lab, ok) for lab, ok, _ in dst.verdicts(r))
    assert ok_d["(d) stayed within terrain bounds"] is False


def test_as_size_normalisation():
    from genesis_vehicle.samples import dual_scene_terrain as dst
    assert dst._as_size(None) == (40.0, 40.0)
    assert dst._as_size(640) == (640.0, 640.0)
    assert dst._as_size(640.0) == (640.0, 640.0)
    assert dst._as_size((640.0, 320.0)) == (640.0, 320.0)
