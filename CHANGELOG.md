# Changelog — genesis_vehicle

All notable changes are documented here. Versions follow
[Semantic Versioning](https://semver.org/) (major.minor.patch).

`__version__` lives in `genesis_vehicle/_version.py` and is re-exported as
`genesis_vehicle.__version__` plus the `version()` / `version_info()`
functions. `VehiclePhysics.__init__` prints a one-line banner with the
running version the first time it is instantiated in a process.

---

## [0.5.28] — 2026-05-25

### Performance — `road_loop` and `city_traffic_ego` substeps 20 → 10

Diagnosed why `road_loop` 16 vehicles measured 824 ms/step in headless
mode. Tried in order:

- `enable_collision=False` → 689 ms/step (only 50 ms savings → collision
  was not the bottleneck)
- `use_contact_island=True` → Genesis 0.4.6 kernel compile error
  (`Cannot subscript NdarrayType` in contact_island.py); fail-closed
- `enable_multi_contact=False` → 775 ms/step (regression)
- `max_collision_pairs` bumped → no change (cap wasn't binding)
- `substeps=20 → 10` → **447 ms/step** (1.66× speedup), physics identical
- `substeps=8` → 363 ms/step but car speeds collapse to 0 m/s
  (suspension stack no longer integrates correctly)

`substeps=10` (internal dt = 1 ms) is the floor for this vehicle stack.
The suspension's natural frequency is ~2 Hz (K~50 kN/m at 375 kg
quarter-car mass) so 1 ms is still ≫ enough; Genesis's default
`substeps=20` was 2× the safe margin. The constraint solver runs once
per substep, so the saving is roughly linear.

Applied the same to `city_traffic_ego` (198 → 138 ms/step, −30 %).
Other samples already had small entity counts; changing substeps there
wouldn't move the needle.

| Sample                          | Before (ms/step) | After (ms/step) | Change                |
|---------------------------------|-----------------:|----------------:|-----------------------|
| `road_loop` 16 veh (per_vehicle)| 824              | ~500 (est)      | substeps 20 → 10      |
| `road_loop` 16 veh (multi_batched)| 740            | 447             | substeps 20 → 10      |
| `city_traffic_ego` 8 veh        | 198              | 138             | substeps 20 → 10      |

### Added — `_hud.warn_if_unused_camera`

Helper that prints a one-line warning when a camera exists but neither
`--viewer` nor `--record` is in use. Documents the gotcha (Genesis
pays a per-step renderer-state sync inside `scene.step()` for any
camera in the scene, even one whose `.render()` is never called — see
v0.5.27 for the impact this had on `multi_env_render`'s 70 % overhead).

The samples in this repo already follow the preferred pattern
(`cam = None; if args.viewer: cam = scene.add_camera(...)`), so the
warning is for user code that copies a sample and forgets to gate the
camera. Optional opt-in:

```python
cam = scene.add_camera(...)   # always create
_hud.warn_if_unused_camera(cam, used=args.viewer or args.record,
                           sample="my_sample")
```

---

## [0.5.27] — 2026-05-25

### Fixed — perf summaries had rendering folded into physics ms/step

In every sample that wrote out a `[timing]` / `print_perf_summary` line,
the headless main loop was calling `cam.render()` every few steps even
though the result was thrown away. The reported `ms/step` therefore
mixed physics with render cost, and the gap was huge for samples that
render a large image: `multi_env_render --n_envs 4` measured 106 ms/step
when the actual pure-physics cost is 37 ms/step. ~70 % was wasted
rendering frames into the bit bucket.

### Changed — strict separation of physics vs render in samples

- In **headless** mode (no `--viewer`), no camera is created at all and
  no render call happens in the main loop. The reported `ms/step` is
  now pure physics + `scene.step()`. (Genesis's `scene.step()` does a
  per-step renderer-state sync if any camera exists, so even an unused
  camera measurably slows physics — hence creating none.)
- In **viewer** mode, the main loop renders inline (same as v0.5.26),
  and a separate post-loop standalone render benchmark of 20 frames is
  printed on its own line in the summary block.
- New `_hud.bench_render(cam, n=20)` helper — single `cuda.synchronize()`
  on each side, returns `(ms_per_frame, n)`.
- `_hud.print_perf_summary` gained `render_ms` / `render_n` kwargs.
  Render line is omitted when no camera exists.

### Re-bench (headless, pure physics)

| Sample                          | Before (ms/step) | After (ms/step) | Notes                                |
|---------------------------------|-----------------:|----------------:|--------------------------------------|
| `quickstart`                    | 41–55            | 42              | unchanged (render was rare)          |
| `slope_hold`                    | 49               | 49              | unchanged                            |
| `batched_rollout n_envs=16`     | 36               | 37              | unchanged (cam was None already)     |
| `multi_env_render n_envs=4`     | 106              | **37**          | **−65 %** — render was 70 % of cost  |
| `road_loop` 16 vehicles         | 829              | 824             | unchanged (`scene.step` dominates)   |
| `city_traffic_ego` 8 vehicles   | 208              | 198             | small win                            |
| `perf_vectorization` n=1..1024  | 23→37            | 25→38           | unchanged (already cam-less)         |

### Behavior change for `multi_env_render`

Previously the sample always called `cam.render()` even in headless
("always renders" tag in the README). With this change, headless
`multi_env_render` runs pure physics and prints a clean ms/step;
`--viewer` keeps the cv2 grid HUD behavior unchanged. If you need the
old "render to a tensor every step" behavior for mp4 recording, run
with `--viewer` — the sample's identity (rendering N parallel envs in
one tiled view) is in the viewer path now, not the headless path.

---

## [0.5.26] — 2026-05-25

### Fixed — `slope_hold` settled-roll print had the wrong sign

The ground tilt is created with `euler=(slope_deg, 0.0, 0.0)`, which
rotates the box +slope_deg around X. The car settled on the slope picks
up the same +slope_deg roll. The "expected" value in the settled-pose
print was `{-slope_deg:+.1f}°` (negated) — so the user saw
`roll=+20.30° (expect roll ≈ -20.0° on slope)` and would reasonably
suspect a sign convention bug in the SDK. The bug was in the print
line; the physics was already correct (lateral slip OK = 0.5 mm).

Fix: drop the negation. The settled-roll print now reads
`expect roll ≈ +20.0° on slope` for a +20° wedge.

### Full-sample bench notes (informational, no code change)

Ran every sample headless and observed:

- `quickstart` / `slope_hold` — 41 / 49 ms per step. Stable.
- `batched_rollout n_envs=16` — 35 ms/step → 451 env-steps/s. L3
  batching working as designed.
- `multi_env_render n_envs=4` — 106 ms/step (~10 fps). Higher than
  raw physics because the sample always calls `cam.render()` (that's
  the sample's purpose; see README "always renders" tag).
- `road_loop` 16 vehicles — 754 ms/step with `multi_batched`, 829
  with `per_vehicle` (9 % gap). The solver-choice gap is small because
  `scene.step()` dominates at 16 entities in one scene.
- `perf_vectorization` — 23 → 37 ms across n_envs ∈ [1, 1024]; 646×
  speedup at n_envs=1024 vs n_envs=1. L3 batching is the right axis
  for raw throughput.
- `perf_multi_vehicle` K=4 n_per_kind ∈ [1, 2, 4] — 66 → 200 → 896 ms;
  per-vehicle cost grows super-linearly because Genesis's `scene.step`
  cost in number-of-bodies dominates. Both `per_vehicle` and
  `multi_batched` solvers hit this equally (multi_batched is only
  ~10 % faster), as already documented in that sample's docstring.
- `perf_l2_l3_combined` — same K=N cell measured 64 ms vs 176 ms on
  separate runs. Variance comes from per-subprocess JIT-cache warm
  state, not from the SDK. The reported `gain` columns are still
  meaningful as a ratio within one run, but absolute ms values
  benchmark-to-benchmark are noisy by ±2-3×.

---

## [0.5.25] — 2026-05-25

### Added — `print_perf_summary` end-of-run report on every sample

When a sample finishes — whether it ran to completion or the user ESC-quit
mid-run — it now prints a banner-separated multi-line block instead of a
single `[timing]` line. The header makes the completion state explicit
(`COMPLETED` vs `USER QUIT (ESC)`) so a short run cut at step 12/200 is
obvious at a glance.

```
======================================================================
 multi_env_render  (v0.5.25)  —  USER QUIT (ESC)
======================================================================
  steps      :     42 / 200      ( 21.0% of requested)
  wall time  :   4.21 s
  ms / step  : 100.24
  steps / s  :   9.98
  batch      : 4 env/step  ->  40 env-steps/s
  grid       : 2 x 2   spacing 10.0 m
======================================================================
```

The helper lives in `samples/_hud.py`:

```python
_hud.print_perf_summary(
    sample=f"multi_env_render  (v{sdk_version})",
    completed=not user_quit,
    n_done=n_done, n_target=n_steps, wall=wall,
    batch=n_envs, batch_label="env",
    extra=[...],
)
```

Applied to all six viewer-supporting samples (`quickstart`, `slope_hold`,
`batched_rollout`, `multi_env_render`, `road_loop`, `city_traffic_ego`).
The three `perf_*` benches were left alone — they have their own
purpose-built reports.

### Fixed — `batched_rollout` accuracy when ESC-quit

`batched_rollout` previously computed `wall / args.steps` even when the
measure loop broke early. With an ESC quit at step 30/100 it would
under-report ms/step by ~3.3x. Now tracks `user_quit` through both the
settle and measure loops, uses the actual `n_done` step count, and
short-circuits with a partial summary if the user quits during the
settle phase.

### Smoke check

60 SDK pytest pass. `quickstart` headless prints the new summary block
on completion.

---

## [0.5.24] — 2026-05-25

### Fixed — `--viewer` only showed env 0 when `env_separate_rigid=True`

`multi_env_render --viewer` (and `batched_rollout --viewer`,
`city_traffic_ego --viewer --n_envs N` with `N>1`) was rendering only
one of the parallel envs and updating at far below the requested rate.

Root cause: Genesis's `cam.render()` returns a per-env stack of shape
`(N, H, W, 3)` whenever `env_separate_rigid=True`; the v0.5.23
`_hud.render_hud_frame` only checked `if frame.ndim == 4: frame =
frame[0]`, silently dropping all but env 0. The "only one car visible"
bug followed directly.

### Changed

- `_hud.render_hud_frame` now tiles a 4-D per-env stack into a
  `per_row × ceil(N/per_row)` mosaic. Two new kwargs:
  - `grid_per_row` — cells per row in the mosaic; default
    `round(sqrt(N))`.
  - `max_cell_size` — clamp each cell's longer axis to this many
    pixels before tiling (default 480). Keeps the displayed mosaic
    a sane size regardless of camera resolution.
- `multi_env_render`, `batched_rollout`, `city_traffic_ego` lower the
  per-cell camera resolution to 640×360 when running with
  `env_separate_rigid=True`. Rendering at 1920×1080 per env when each
  cell is downsized to ≤480 px wastes ~10× the GPU + GPU→CPU work and
  was the main cause of the sub-10 fps display rate the user observed.
  Headless and single-env paths keep the original 1920×1080.
- Samples now pass `grid_per_row` matching their build-time
  `n_envs_per_row` so the HUD layout matches the physics-side grid.

### Effect on the reported case

`multi_env_render --viewer --n_envs 4` (per_row=2, spacing 10):
- Before: env-0-only view, ~2.5 fps display (1080p × 4 = 8 MP/frame
  rendered then 7/8 discarded).
- After: 2×2 grid mosaic (960×540 final), per-cell render at 640×360,
  display refresh follows the existing `~25 fps render_every` cadence
  (actual frame-to-frame rate now limited by physics step time, not
  the render).

---

## [0.5.23] — 2026-05-25

### Changed — `--viewer` now opens a cv2 window with live HUD overlay

`--viewer` previously opened Genesis's built-in interactive 3D window
(v0.5.21) but had no way to overlay text — performance numbers, vehicle
state, sample-specific info — on top of the camera frame. Switched to a
cv2-based pattern that does:

- Renders the offscreen camera each step
- Draws a semi-transparent header rectangle
- Overlays a title + per-sample state lines + a perf footer
  (rolling-mean `ms/step` + `fps`)
- Shows via `cv2.imshow`; `ESC` quits

The trade-off: lose Genesis's mouse-rotate interactivity, gain a
fixed-POV camera with informative overlay (the camera's pose is the one
each sample already set for its offscreen rendering).

### Added — `samples/_hud.py`

Minimal helper used by all 6 viewer-supporting samples:

- `PerfMeter` — rolling-mean ms/step over a configurable window (default 60).
  ``tick()`` cost is well under 1 μs; safe to call every step.
- `render_hud_frame(cam, title, lines, perf_ms)` — renders the camera and
  draws the HUD overlay; returns BGR `np.ndarray` for `cv2.imshow`.
- `cv2_show(window, frame)` — wraps `cv2.imshow + cv2.waitKey(1)`;
  returns `False` on `ESC`.
- `cv2_cleanup()` — `cv2.destroyAllWindows()`.

No `pynput` dependency (unlike the umbrella `_demo_hud.py`); cv2 only.
If cv2 isn't importable, `--viewer` falls back to headless with a
warning.

### Per-sample HUD content

| Sample | HUD lines |
|---|---|
| `quickstart`         | t, throttle, pos, speed |
| `slope_hold`         | t, brake, pos, roll, lateral slip |
| `batched_rollout`    | phase, step, grid, speed spread across envs |
| `road_loop`          | step, vehicle/kind count, per-kind speeds |
| `multi_env_render`   | step, grid spec, speed spread across envs |
| `city_traffic_ego`   | step, total batch, ego pose/speed, L2 kinds |

All include a perf footer (`X.XX ms/step (Y fps)`) and an `[ESC] quit`
prompt.

### `ESC`-to-quit

All 6 samples now break out of the main loop on `ESC`. The end-of-run
`[timing]` print uses the actual `n_done` step count (steps that ran
before `ESC`), so the throughput numbers are honest even when the user
cuts the run short.

### No SDK code changes

Pure sample / docs work. 60 SDK tests pass.

---

## [0.5.22] — 2026-05-25

### Added — zero-overhead timing across all non-bench samples

Every sample now prints a `[timing]` line at the end of its main drive
loop showing total wall time, ms/step, and throughput. The measurement
adds essentially zero overhead — one `torch.cuda.synchronize()` before
the loop and one after, no per-step sync. Sample output:

```
[timing] 240 steps in 11.01s  = 45.88 ms/step  (22 steps/s)
[timing] 150 steps in 35.57s  → 237.13 ms/step  (135 vehicle-steps/s, batch=4×8=32 per step)
```

Per-sample variants of the throughput field:

| Sample | extra throughput info |
|---|---|
| `quickstart`         | `steps/s` |
| `slope_hold`         | `steps/s` |
| `batched_rollout`    | `env-steps/s` (existing, unchanged) |
| `road_loop`          | `vehicle-steps/s, solver=<name>` |
| `multi_env_render`   | `env-steps/s, batch <n_envs>` |
| `city_traffic_ego`   | `vehicle-steps/s, batch=N×K` |

The three perf benches (`perf_vectorization`, `perf_multi_vehicle`,
`perf_l2_l3_combined`) are unchanged — they were already comprehensive
benchmarks.

### Removed — `--bench` flag (now always-on)

`road_loop` and `city_traffic_ego` previously gated their wall-time
print behind `--bench` because of a perceived overhead concern. The
overhead turned out to be negligible (single sync before/after), so the
flag is removed and the timing prints unconditionally. Scripts passing
`--bench` to these two samples will need to drop the flag — argparse
will error otherwise.

No SDK code changes; samples only. 60 SDK tests still pass.

---

## [0.5.21] — 2026-05-25

### Fixed — `--viewer` now actually opens a viewer window

Previously the `--viewer` flag on the 6 viewer-supporting samples
(`quickstart`, `slope_hold`, `batched_rollout`, `road_loop`,
`multi_env_render`, `city_traffic_ego`) only added an offscreen
camera that called `cam.render()` each step. That renders into a
GPU tensor but DOES NOT open any visible window — `--viewer` did
nothing the user could see. Bug report from korfriend:

> "gen_vesis/bin/python -m genesis_vehicle.samples.city_traffic_ego
> --viewer  이거 뷰어 안 나오던데"

Fixed by wiring `--viewer` to Genesis's built-in interactive viewer:

```python
viewer_opts = gs.options.ViewerOptions(
    res=(1280, 720),
    camera_pos=(...), camera_lookat=(...), camera_up=(...), camera_fov=...,
) if args.viewer else None
scene = gs.Scene(
    ...,
    viewer_options=viewer_opts,
    show_viewer=args.viewer,
)
```

Each sample's `ViewerOptions` uses the same camera framing as its
existing offscreen camera (side chase-cam for `quickstart`, side view
of the slope for `slope_hold`, top-down for `road_loop` /
`city_traffic_ego` / `multi_env_render`, grid top-down for
`batched_rollout`). The offscreen camera is kept — it still produces
image tensors for inspection / mp4 recording, independent of the
viewer window.

`multi_env_render` also gained a `--viewer` flag (was always
headless-render-only, no window).

### Caveats

- The viewer window is Genesis's, not OpenCV — mouse-rotate, zoom,
  ESC to close.
- WSL / WSLg with a software OpenGL fallback won't open a window; you
  need hardware GL (D3D12 via `GALLIUM_DRIVER=d3d12` on WSL, or native
  Windows / Linux). See `docs/batching.md` notes about WSL GL.

All 60 SDK tests still pass.

---

## [0.5.20] — 2026-05-25

### Fixed — samples now run via direct file path too

Previously every sample required either `python -m genesis_vehicle.samples.X`
from the project root OR a `PYTHONPATH` export. Running them by full
file path —

```bash
gen_vesis/bin/python genesis_vehicle/samples/city_traffic_ego.py --viewer
```

— failed with `ModuleNotFoundError: No module named 'genesis_vehicle'`
because Python only adds the file's own directory to `sys.path`, not
the SDK's parent.

Fixed by a 3-line bootstrap at the top of each of the 9 samples that
prepends the SDK parent directory to `sys.path`:

```python
import sys, pathlib
_SDK_PARENT = str(pathlib.Path(__file__).resolve().parents[2])
if _SDK_PARENT not in sys.path:
    sys.path.insert(0, _SDK_PARENT)
```

For the 3 perf samples (which re-invoke themselves as subprocesses to
get a clean GPU per measurement), the same `_SDK_PARENT` is also
exported as `PYTHONPATH` to the subprocess so the child process can
resolve `genesis_vehicle` regardless of cwd.

Net effect: all samples now work via any invocation pattern —

```bash
python -m genesis_vehicle.samples.X            # module form (always worked)
python path/to/samples/X.py                    # direct file path (new)
PYTHONPATH=... python ...                       # explicit env var (still works)
```

All 60 SDK tests still pass.

---

## [0.5.19] — 2026-05-25

### Docs — first-class samples link from README

The 9 bundled samples were referenced throughout the docs but the
catalog [`samples/README.md`](samples/README.md) wasn't linked from
the main README itself. Fixed:

- **README.md** new "Samples" section between "Bundled Presets" and
  "Documentation". Lists all 9 samples with their `python -m
  genesis_vehicle.samples.<name>` invocation in a single code block,
  short note about `--viewer` vs headless-by-design benches, and
  pointer to `samples/README.md` for full catalog with descriptions.
- **README.md** Documentation table gets a `samples/README.md` row at
  the top alongside `docs/index.md`.
- **docs/index.md** TOC gains a "See all 9 runnable samples" entry
  linking to `samples/README.md`.

No code changes.

---

## [0.5.18] — 2026-05-25

### Docs — align sample references with the v0.5.17 viewer additions

After v0.5.17 added `--viewer` to `quickstart`, `slope_hold`, and
`batched_rollout` (and made it explicit that the three perf benches
are headless by design), the doc references hadn't yet been updated.
This release fixes those:

- **README.md** "Three-axis vectorization" section — closing paragraph
  now separates "runnable benchmarks (headless)" and "visual demos
  (with `--viewer`)" into two groups, and adds the
  `city_traffic_ego.py` reference under L2 × L3 visuals (was missing).
- **docs/quickstart.md** — TL;DR shows BOTH `python -m
  genesis_vehicle.samples.quickstart` and the `--viewer` variant.
- **docs/batching.md** — L1, L2, L3, and L2 × L3 sections all flag
  whether each referenced sample has `--viewer` or is headless by
  design. New "See also" entry under L2 × L3 points to both the
  headless `perf_l2_l3_combined.py` AND the visual
  `city_traffic_ego.py`. Performance reference table gets a row for
  `city_traffic_ego.py`.

No code changes.

---

## [0.5.17] — 2026-05-25

### Added — `--viewer` for the 3 samples that didn't have it

[`samples/quickstart.py`](samples/quickstart.py),
[`samples/slope_hold.py`](samples/slope_hold.py),
[`samples/batched_rollout.py`](samples/batched_rollout.py) now accept
`--viewer`:

- **quickstart**: side chase-cam trailing the car.
- **slope_hold**: side view showing the slope tilt + the brake-locked
  car (so the v0.5.7 stick-slip fix is visible as the car staying put).
- **batched_rollout**: lays the N envs out on a grid via
  `env_separate_rigid=True` + `env_spacing` so all parallel rollouts are
  visible in one frame. (Same mechanism as `multi_env_render.py`.) The
  flag explicitly notes that rendering distorts the throughput numbers
  — use `perf_vectorization.py` for clean bench timings.

### Clarified — three perf samples are intentionally headless

[`samples/perf_vectorization.py`](samples/perf_vectorization.py),
[`samples/perf_multi_vehicle.py`](samples/perf_multi_vehicle.py),
[`samples/perf_l2_l3_combined.py`](samples/perf_l2_l3_combined.py)
docstrings now explicitly say they have NO `--viewer` flag because
camera rendering adds per-step overhead that distorts the throughput
numbers they're trying to measure. Each one points to the matching
visual sample for the same scenario.

### Viewer support across all 9 samples

| Sample | --viewer | Notes |
|---|---|---|
| quickstart           | ✓ (new) | side chase-cam |
| slope_hold           | ✓ (new) | side view of tilted ground + car |
| batched_rollout      | ✓ (new) | env grid via env_separate_rigid |
| road_loop            | ✓        | top-down loop |
| perf_vectorization   | ✗ headless by design | bench |
| multi_env_render     | always renders | env grid is the demo |
| perf_multi_vehicle   | ✗ headless by design | bench |
| perf_l2_l3_combined  | ✗ headless by design | bench |
| city_traffic_ego     | ✓        | top-down highway |

No SDK code changes; samples only.

---

## [0.5.16] — 2026-05-25

### Added — `city_traffic_ego` sample (highway ego + traffic scenario)

[`samples/city_traffic_ego.py`](samples/city_traffic_ego.py) — the
reference setup for "autonomous-driving ego in surrounding traffic"
workflows. A 4-lane straight highway on a flat plane with:

- **1 ego** — bright red AWD car (sportier body)
- **7 traffic agents** — 3 RWD blue coupes, 3 FWD small dark-red
  sedans, 1 yellow 6-wheel truck
- **Lane-keeping P controller** on every vehicle (target lane center
  + small yaw correction)
- **Top-down camera** framing the action area
- **`--n_envs N`** enables the L2 × L3 combined batching pattern —
  one MPPI candidate (or RL rollout) per env, all 8·N vehicles
  processed in batched compute per kind

Measured (RTX 5070 Laptop, 3-second drive, 150 steps):

| n_envs | total batch | ms/step | vehicle-steps/s |
|-------:|------------:|--------:|----------------:|
|      1 |           8 |   205.9 |              39 |
|      4 |          32 |   234.9 |             136 |
|     16 |         128 |   ~290  |            ~440 |

n_envs=4 costs +14% ms/step for 3.5× more vehicle-steps/s — the
expected L2 × L3 combined scaling.

The 8 vehicle fleet groups into 4 kinds in `MultiVehiclePhysics`:
`K per kind = [1, 3, 3, 1]` (ego is its own group because it uses
a different URDF / cfg from the RWD traffic).

### Usage

```bash
python -m genesis_vehicle.samples.city_traffic_ego                    # 1 env, visual
python -m genesis_vehicle.samples.city_traffic_ego --viewer
python -m genesis_vehicle.samples.city_traffic_ego --n_envs 16        # 16 parallel scenarios
python -m genesis_vehicle.samples.city_traffic_ego --n_envs 16 --bench
```

### Docs

- `samples/README.md` — row #9 added.

---

## [0.5.15] — 2026-05-25

### Added — `docs/batching.md` (L1 / L2 / L3 reference)

The three orthogonal vectorization axes are now a first-class topic in
the docs and README:

- **README.md** — new top-level "Three-axis vectorization (L1 / L2 / L3)"
  section above "Installation" with the headline table, rule-of-thumb
  decision guide, and measured numbers.
- **docs/batching.md** (new) — full treatment: each axis explained, when
  it matters, when it doesn't, code patterns, measured speedups
  (RTX 5070 Laptop), L2 × L3 combined pattern, decision matrix, what's
  NOT batched.
- **docs/concepts.md** — "Batched by default" section now sketches the
  three axes and links to `batching.md`.
- **docs/index.md** — added "Pick the right batching axis" entry to
  the landing TOC.

No code changes; doc-only release.

---

## [0.5.14] — 2026-05-25

### Added — L2 × L3 combined batching (`n_envs > 1` for MultiVehiclePhysics)

`MultiVehicleKindPhysics` and `MultiVehiclePhysics` now accept an
``n_envs`` parameter. The internal compute pipeline batch dimension
becomes ``N * K`` (parallel envs × vehicles per kind), and all I/O
collapses to ONE batched solver call per quantity per kind:

  - State reads (`get_links_{pos,quat,vel,ang}`): return ``(N, K, 3)``,
    flattened to ``(NK, 3)`` for compute.
  - Force / torque writes: compute output ``(NK, 3)`` reshaped to
    ``(N, K, 3)`` and applied in one ``apply_links_external_force`` call.
  - VisualSync: K per-entity objects, each built with ``n_envs=N``;
    compute output sliced ``(N, n_wheels)`` per entity.

```python
mphys = MultiVehiclePhysics(scene, vehicles, n_envs=64)
mphys.step([VehicleInputs(throttle=tensor_of_shape_N, ...) for _ in vehicles])
```

Per-vehicle inputs accept ``(N,)`` tensors so each parallel env can have
distinct controls (RL-style per-env action diversity).

**n_envs default is 1**, so existing v0.5.11-v0.5.13 call sites continue
to work unchanged.

### Added — `perf_l2_l3_combined` sample

[`samples/perf_l2_l3_combined.py`](samples/perf_l2_l3_combined.py)
sweeps a 2D ``(K, N)`` grid and reports a scaling table. Each cell is
a fresh subprocess (clean GPU state). Sample result on RTX 5070 Laptop:

|  K |  N | total | ms/step | per veh (μs) | gain |
|---:|---:|------:|--------:|-------------:|-----:|
|  1 |  1 |     1 |   26.31 |       26,315 | 1.0× |
|  1 |  4 |     4 |   37.83 |        9,458 | 2.8× |
|  2 |  1 |     2 |   35.98 |       17,992 | 1.5× |
|  2 |  4 |     8 |   45.79 |        5,724 | **4.6×** |

The L2-only (K=2) and L3-only (N=4) gains (1.5×, 2.8×) multiply close
to the combined gain (4.6× ≈ 1.5×2.8 = 4.2×, with a small bonus from
better GPU utilization at larger batch). This is the headline pattern
for autonomous-driving simulation: K vehicles per scenario (ego + N_t
traffic) × M parallel scenarios.

### Use case decision matrix

| Scenario | Solver |
|---|---|
| 1 vehicle, RL rollouts | `VehiclePhysics(n_envs=N)` |
| K vehicles in 1 visible scene (traffic) | `MultiVehiclePhysics(n_envs=1)` |
| 1 vehicle but parallel scenarios | `VehiclePhysics(n_envs=N)` |
| **K vehicles × N parallel scenarios** | **`MultiVehiclePhysics(n_envs=N)`** |

### Migration

None — `n_envs=1` is the default; existing code keeps working.

All 60 SDK tests pass; new sample smoke-tested.

---

## [0.5.13] — 2026-05-25

### Added — `perf_multi_vehicle` sample (L2 solver comparison)

[`samples/perf_multi_vehicle.py`](samples/perf_multi_vehicle.py) — the
L2 counterpart to `perf_vectorization.py`. Sweeps `--n_per_kind` and
runs the same 4-kind fleet under both solvers (`per_vehicle` and
`multi_batched`) in fresh subprocesses, prints a scaling table:

| n_per_kind | total | per_vehicle (ms) | multi_batched (ms) | speedup |
|-----------:|------:|-----------------:|-------------------:|--------:|
|          1 |     4 |            75.27 |              78.84 |   0.95× |
|          2 |     8 |           208.20 |             183.25 |   1.14× |

At K=1 (4 vehicles, 1 per kind) `multi_batched` is slightly SLOWER —
the batching machinery has setup overhead with nothing to batch (each
"kind" group is just 1 vehicle). From K=2 upward the gap widens; in
practice expect 1.1-1.2× speedup for realistic multi-vehicle scenes.

Use this script to:
- Decide which solver to use for your specific fleet size + kind mix.
- Catch regressions if either solver's perf drifts.

The output makes it concrete that L2 batching is helpful but bounded
(see also `MultiVehiclePhysics` docstring) — for pure throughput on
one vehicle kind, n_envs > 1 (L3) remains the right tool.

### Docs

- `samples/README.md` — added row #7.

---

## [0.5.12] — 2026-05-25

### Fixed — `MultiVehicleKindPhysics` now supports VisualSync

v0.5.11 disabled the internal VisualSync entirely because the proto
`VehiclePhysics(n_envs=K)` it inherits from would issue a
`set_dofs_position` with shape `(K, n_dofs)` into a scene actually
built with `n_envs=1`, which Genesis rejects (the K dim is "vehicles
in this env", not "parallel envs").

Fix: build K independent `VisualSync` instances (one per entity, each
`n_envs=1`). After the batched compute pipeline produces (K, n_wheels)
spin / steer / suspension targets, the visual step slices them into K
per-entity (1, n_wheels) chunks and dispatches K small
`set_dofs_position` calls — a Python loop, but each call is tiny.

Measured overhead: ~34 ms added to the 16-vehicle road_loop step
(760 → 794 ms / step). Net L2 speedup vs per_vehicle (848 → 794)
shrinks slightly from 10% to ~6%, but every vehicle now gets correct
wheel-spin and steering visuals — making `multi_batched` a drop-in
replacement for `per_vehicle` in visual demos.

### Migration

None — same `MultiVehiclePhysics` constructor signature. Existing code
written against v0.5.11 keeps working and gets visuals for free.

---

## [0.5.11] — 2026-05-25

### Added — `MultiVehiclePhysics` (L2 cross-vehicle batching)

New SDK class for the case where K vehicles share ONE Genesis scene at
DIFFERENT positions (traffic, multi-agent, MPPI candidate fleets). The
default ``VehiclePhysics`` is one-per-entity, so K vehicles in the same
scene meant a Python loop of K ``step()`` calls per simulation step.

``MultiVehiclePhysics`` groups vehicles by URDF / cfg identity and runs
ONE batched compute pipeline per kind:

- Pacejka tire model: one call over ``(K_kind, n_wheels)`` instead of
  K_kind separate calls.
- Stability hooks: one batched invocation per hook per kind.
- Force + torque application: one ``apply_links_external_force`` call
  with K_kind link indices instead of K_kind separate calls.

The compute batching uses the rigid solver's existing batched APIs
(``get_links_pos``, ``get_links_quat``, ``get_links_vel``,
``get_links_ang``) so cross-vehicle reads are also a single call per
quantity per kind.

```python
from genesis_vehicle import MultiVehiclePhysics, VehicleInputs

# Build the scene as usual: K different entities, each spawned at its own
# pose, each with its own sensor. SHARE the cfg instance across vehicles
# of the same kind — MultiVehiclePhysics groups by ``id(cfg)``.
cfg_per_kind = [preset_fn(urdf) for ...]
vehicles = []
for k, (urdf, preset_fn) in enumerate(kinds):
    for _ in range(K):
        ent, sens, _ = add_vehicle(scene, urdf, preset_fn=None, pos=(...))
        vehicles.append((ent, sens, cfg_per_kind[k]))

scene.build(n_envs=1)
mphys = MultiVehiclePhysics(scene, vehicles)
for step in range(N):
    mphys.step([VehicleInputs(...) for _ in vehicles])
    scene.step()
```

### Performance

L2 batching helps but is bounded by Genesis's per-entity ``scene.step()``
cost (which dominates as the number of in-scene entities grows). For a
16-vehicle ``road_loop`` scenario the measured speedup is **~10%**
(848 → 760 ms/step on an RTX 5070 Laptop).

For dramatic batching speedups, ``n_envs > 1`` (L3) is still the right
tool — see ``samples/perf_vectorization.py`` (44× at n_envs=64). L2's
value is enabling multi-vehicle scenes (visualization, multi-agent
interaction, mixed-kind comparisons) at a modest extra cost vs the
per-vehicle loop, NOT replacing L3 for pure-throughput RL/MPPI.

### Sample integration

``samples/road_loop.py`` gains ``--solver {per_vehicle, multi_batched}``
(default ``per_vehicle``) and ``--bench`` flags so users can compare the
two on the same 4-kind, K-per-kind fleet:

```bash
python -m genesis_vehicle.samples.road_loop --n_per_kind 4 --bench --solver per_vehicle
python -m genesis_vehicle.samples.road_loop --n_per_kind 4 --bench --solver multi_batched
```

### Known limitations

- `MultiVehicleKindPhysics` disables its internal `VisualSync` (the
  visual joint updates would need K separate VisualSync objects, not
  in v0.5.11). Wheels won't visibly spin in the chase-cam view; chassis
  motion is correct.
- Vehicles of the SAME kind must share the SAME cfg INSTANCE (not just
  the same URDF) — group by passing `cfg_per_kind[k]` instead of calling
  the preset fresh per vehicle. `MultiVehiclePhysics` groups by
  `id(cfg)`.

All 60 SDK tests pass; no changes to `VehiclePhysics` or other existing
APIs.

---

## [0.5.10] — 2026-05-25

### Added — `multi_env_render` sample

[`samples/multi_env_render.py`](samples/multi_env_render.py) closes the
loop on the v0.5.9 batching story: ``perf_vectorization`` shows you the
44× speedup at ``n_envs=64`` as numbers, but until now there was no
way to actually SEE what those 64 parallel rollouts were doing —
Genesis simulates them at the same world coordinates (the env axis is
a "parallel universe" axis, not a spatial one).

The fix is a Genesis feature pair we hadn't exposed in samples yet:

```python
scene = gs.Scene(
    vis_options=gs.options.VisOptions(env_separate_rigid=True, ...),
    ...,
)
scene.build(
    n_envs=N,
    env_spacing=(dx, dy),     # visualization-only offset
    n_envs_per_row=K,         # √N by default
)
```

The renderer then offsets each env's rigid entities to its grid cell.
Physics is unchanged (still parallel universes at the same coords);
only the visualization adds the per-env transform. The overhead
camera frames the entire grid, so MPPI candidate trajectories or RL
rollout diversity is visible at a glance.

Per-env throttle / steer are randomized so every cell looks different —
useful as a sanity check that batched simulation isn't accidentally
making all envs identical.

### Docs

- `samples/README.md` updated with row #6.

---

## [0.5.9] — 2026-05-24

### Added — two more samples (`road_loop`, `perf_vectorization`)

- [`samples/road_loop.py`](samples/road_loop.py) — multi-vehicle visual
  demo. 4 distinct kinds (FWD red sedan, RWD blue coupe, AWD green SUV,
  yellow 6-wheel truck), `--n_per_kind` each, all driving a circular
  track under constant Ackermann steering. Top-down camera frames the
  whole fleet. URDFs are generated parametrically at runtime to a
  tempdir so the demo stays self-contained.

- [`samples/perf_vectorization.py`](samples/perf_vectorization.py) —
  `n_envs` batching speedup benchmark. Sweeps
  `n_envs ∈ [1, 4, 16, 64, 256, 1024]` (one fresh subprocess per
  measurement to keep GPU state clean) and prints a scaling table.
  Typical result on an RTX 5070 Laptop:

  | n_envs | ms / step | env-steps / s | per env (μs) | speedup |
  |-------:|----------:|--------------:|-------------:|--------:|
  |      1 |     26.28 |            38 |       26 277 |    1.0× |
  |      4 |     36.78 |           109 |        9 196 |    2.9× |
  |     16 |     37.28 |           429 |        2 330 |   11.3× |
  |     64 |     37.86 |         1 691 |          592 |  44.4× |

  `ms / step` stays roughly constant from n_envs=4 upward — Genesis +
  the SDK saturate the GPU, so each additional parallel env is nearly
  free. This is the headline batching benefit that RL / MPPI workloads
  exploit. The sweep is honest (each row a separate GPU process) rather
  than relying on simulated kernel-launch overhead.

### Docs

- `docs/api-reference.md` runnable-references callout extended to list
  all five samples.
- `samples/README.md` updated.

---

## [0.5.8] — 2026-05-24

### Added — bundled `samples/` directory

Three minimal, self-contained reference scripts ship with the SDK so
that `pip install genesis-vehicle` is immediately runnable without
hunting for assets or umbrella-level helpers:

- [`samples/quickstart.py`](samples/quickstart.py) — minimum-viable
  call pattern (preset → `add_vehicle` → step loop). Drives a car
  forward for 5 s on flat ground.
- [`samples/slope_hold.py`](samples/slope_hold.py) — `StaticFrictionLock`
  side-slope hold regression check. Reports OK / REGRESSION based on
  lateral slip over 10 s under `brake=1.0`.
- [`samples/batched_rollout.py`](samples/batched_rollout.py) — batched
  `n_envs > 1` API for RL / MPPI. Demonstrates per-env random controls
  and per-step throughput.

Bundled asset:
- [`samples/urdf/car_4w.urdf`](samples/urdf/car_4w.urdf) — minimal
  1.2 t 4-wheel RWD sedan, primitive geometry only (no mesh files),
  follows project URDF conventions.

Run as Python modules:

```bash
python -m genesis_vehicle.samples.quickstart
python -m genesis_vehicle.samples.slope_hold --slope 20
python -m genesis_vehicle.samples.batched_rollout --n_envs 256
```

Docs (`quickstart.md`, `stability-profiles.md`, `api-reference.md`)
now cross-reference the samples as runnable examples.

### Changed — `"control"` profile is now uniform across vehicle kinds

`stability_hooks_for_profile("control", vehicle_kind=...)` now returns
the same hook set (`[RollingResistance, LowSpeedRegularizer,
StaticFrictionLock]`) regardless of `vehicle_kind`. Previously only
`vehicle_kind="tank"` got the lock — `vehicle_kind="car"` was hook[0:2]
only. That was a real footgun: any preset car on a slope (or after an
impact-induced lateral velocity) would creep indefinitely under brake.

The new stick-slip lock (v0.5.7) is cheap enough that there's no
reason to leave cars without it. The truck preset's manual `append` of
the lock is removed since the base profile now includes it.

`vehicle_kind` is kept as a parameter for forward compatibility with
future vehicle-specific tweaks, but currently has no effect on the
profile hook set.

Test update: `test_profile_control_car_includes_static_friction_lock`
replaces the pre-v0.5.8 `..._returns_two_hooks` test.

---

## [0.5.7] — 2026-05-24

### Fixed — `StaticFrictionLock` is now true static friction (stick-slip)

User physics critique on v0.5.6 (the tanh velocity-damper version): a
real asphalt-tire contact at μ=1.0 should hold a vehicle stationary on
any slope where μ > tan(slope_angle) — for a 20° slope that's μ > 0.36,
well within μ=1.0. v0.5.6 still showed ~5 cm drift over 10 s at 20°
because `F = -μN·tanh(v/scale)` is fundamentally a *kinetic* friction
model: any non-zero `v` produces an opposing force, but `v=0` produces
`F=0`. Equilibrium on a slope therefore requires non-zero v_lat — visible
creep, even though physics says the vehicle should be motionless.

### Changes

- `genesis_vehicle/strategies/stability.py` — rewrote `StaticFrictionLock`
  as a **position-anchored stick-slip** model:
  - When the lock first engages on a wheel, the contact position becomes
    its anchor (displacement `d = 0`).
  - Each step while active: integrate `d += v · dt`; compute force as a
    spring + damper: `F = -K_spring·d - K_damp·v`.
  - Project `(F_long, F_lat)` onto the per-wheel friction ellipse (same
    form as `tire_models/pacejka.py`).
  - If the projection clamps (i.e., wheel is slipping), advance the anchor
    so the next-step spring force matches the friction limit — kinetic-
    friction behaviour without spring runaway.
  - Otherwise (stuck), the spring quietly absorbs any external lateral
    force up to μN. Vehicle is **truly stationary**.
  - New tuning parameters: `k_spring` (default 500_000 N/m per wheel,
    around the explicit-Euler stability limit at dt=0.02 for a 5 t
    chassis) and `k_damp` (default 20_000 N·s/m per wheel, ~half-critical
    at the natural frequency).
  - Legacy `hold_k`, `hold_k_lat`, `slip_scale` constructor kwargs are
    REMOVED (see BREAKING section below).
- `genesis_vehicle/core.py` — added `ctx.dt: float` to PipelineContext
  so hooks can integrate per-step state (the stick-slip lock needs it).
- `genesis_vehicle/presets.py` — tank and truck preset call sites updated
  to the new signature (`StaticFrictionLock(brake_thr=0.3, v_thr=0.5)`,
  no `hold_k` arg).
- `genesis_vehicle/urdf.py` + `__init__.py` — removed deprecated alias
  `parse_inertia_max_principal_genesis` (was a back-compat shim from
  v0.4.x → v0.5.0 rename; no longer needed in pre-1.0).

### Verified — `samples/slope_lateral_slip.py` truck preset

| slope    | v0.5.5 (1D lock)  | v0.5.6 (2D tanh) | v0.5.7 (stick-slip) |
|---|---|---|---|
| flat 0°  | -72 mm            | 0 mm             | **0.0 mm**          |
| 20°      | -114 mm           | -54 mm           | **2.9 mm**          |
| 30°      | (not tested)      | (not tested)     | **0.4 mm**          |
| 40°      | (rolls over)      | (rolls over)     | (rolls over)        |

The 40° case is a real physical limit: μ=1.0 → max-hold-able slope ≈
arctan(1.0) = 45°, but a high-COG 5 t truck tips over at ~35-40° well
before reaching the friction limit. The lock cannot prevent tip-over
(would need a separate "rollover restraint" hook, out of scope).

### BREAKING — Migration required

`StaticFrictionLock` no longer accepts `hold_k`, `hold_k_lat`, or
`slip_scale`. They were proportional / tanh gains in the prior velocity-
damper models (v0.5.5, v0.5.6) and have no meaning in the new stick-slip
model. Replace with `k_spring` / `k_damp` (or drop them and use the
defaults, which are tuned for typical 1-5 t vehicles):

```python
# Before:
StaticFrictionLock(brake_thr=0.3, v_thr=0.5, hold_k=400_000.0)
# After:
StaticFrictionLock(brake_thr=0.3, v_thr=0.5)
# or explicitly:
StaticFrictionLock(brake_thr=0.3, v_thr=0.5,
                   k_spring=500_000.0, k_damp=20_000.0)
```

Pre-1.0 SDK — no deprecation shim. Bundled presets (tank, truck) have
been updated.

---

## [0.5.6] — 2026-05-24

### Fixed — `StaticFrictionLock` 2D extension (lateral slip on slopes)

Student bug report from MPPI work: a truck holding `brake=1.0` on a 20°
side slope continued to creep downhill ~11 cm over 10 s, despite the
control-profile stability hooks all being active. Root cause:
``StaticFrictionLock`` only wrote ``ctx.F_long``; ``ctx.F_lat`` came
straight from Pacejka, which is ill-conditioned near rest because
``alpha = atan2(v_lat, max(|v_long|, eps_v=0.5))`` artificially
compresses the slip angle (so ``F_lat`` from the magic formula was
much smaller than what a real tire would produce at rest).

### Changes

- `genesis_vehicle/strategies/stability.py` — `StaticFrictionLock` now:
  - Activates on planar wheel speed ``sqrt(v_long² + v_lat²) < v_thr``
    (was: ``|v_long| < v_thr`` only).
  - Overrides both ``F_long`` and ``F_lat`` with a smooth saturated
    hold ``-mu·N · tanh(v / slip_scale)`` per axis, then projects onto
    the per-wheel friction ellipse — same form as the Pacejka clamp.
  - Switched from `clamp(-hold_k · v, ±mu·N)` (proportional + hard
    clamp) to `tanh(v / slip_scale)`: the original form was a
    bang-bang controller whose effective gain in the saturated regime
    exceeded the explicit-Euler stability bound (`K_total · dt / mass
    = 9.6` for the truck preset, well above the ~2 stability limit) —
    fine in 1D where ``v_long`` was always tiny, but on a side slope
    the sustained lateral gravity excited the discretization into a
    growing oscillation. The tanh form keeps `|F| ≤ mu·N` smoothly,
    eliminating overshoot.
  - New `hold_k_lat` parameter (defaults to `hold_k`) and `slip_scale`
    (defaults to `v_thr / 5` — the stable upper limit on the truck
    preset at dt=0.02).
- `_version.py`: 0.5.5 → 0.5.6.
- `docs/pipeline-and-hooks.md`, `docs/stability-profiles.md`: updated
  hook descriptions.

### Verified — `samples/slope_lateral_slip.py`

| slope | pre-patch slip | post-patch slip | verdict |
|---|---|---|---|
| flat (0°)     | -72 mm  | **0 mm**   | OK |
| 10° side      | n/a     | **32 mm**  | NO BUG OBSERVED (under 80 mm threshold) |
| 20° side      | -114 mm | **54 mm**  | NO BUG OBSERVED |

### Known limitation

The tanh saturated form is a pure velocity damper: any sustained external
lateral force (gravity component on a slope) produces an equilibrium drift
velocity proportional to (force) / (gain near v=0). At the maximum stable
gain, this is ~5-10 mm/s on the truck preset at 20°. Truly zero-drift
hold on steep slopes (> ~25°) would require a position-tracked stick-slip
model (store a per-wheel anchor at lock-on time, spring + damper to it,
release when force exceeds the static-friction circle). Filed as future
work; current behavior is the standard fix for the Real2Sim / MPPI use
cases that motivated the bug report.

### Migration

`StaticFrictionLock(brake_thr=..., v_thr=..., hold_k=...)` continues to
work — `hold_k_lat` defaults to `hold_k`, `slip_scale` defaults from
`v_thr`. No call-site changes required in `presets.py` or external code.

---

## [0.5.5] — 2026-05-18

### Changed — `truck_6w_partial_ackermann` brake behavior

User report from the Truck6w demo: pressing SPACE (brake) while the
scenario kept driving throttle on produced visibly slow deceleration —
the brake was working mechanically (~1.8 m/s² decel for a 5-ton truck
with throttle still applied), but not enough to feel responsive.

Two preset tweaks:

- `t_brake_max` 8000 → 15_000 N·m. Realistic emergency-brake decel for a
  5-ton chassis is ~4-5 m/s²; the old value capped at ~2 m/s² (and that
  was while fighting throttle). Now ~4 m/s² with throttle off.
- The `"control"` stability profile for the truck preset now also includes
  `StaticFrictionLock(brake_thr=0.3, v_thr=0.5, hold_k=400_000)` (same
  pattern as the tank preset). The truck used to creep at < 0.5 m/s after
  the brake's `tanh(omega/0.5)` smoothing weakened — now it holds at rest.

The `"raw"` and `"research"` profiles get no StaticFrictionLock (consistent
with their "no hooks" semantics).

### Demo behavior — `GeneVehicle_Truck6w/demo_drive.py`

SPACE is now an **emergency brake**: it forces throttle to 0 in addition
to setting brake = 1. Mirrors real-car panic-brake semantics (brake pedal
overrides the accelerator). The scenario throttle resumes when SPACE is
released. Reason for the change: the previous behavior (SPACE adds brake
on top of scenario throttle) made `T_drive` and `T_brake` fight each other
and decelerated very gently — confusing to the human.

For raw "brake-while-throttling" research, hold SPACE and observe — but be
aware that the scenario throttle is now zeroed under SPACE.

---

## [0.5.4] — 2026-05-18

### Documentation — URDF steer joint axis recommendation

Added a recommendation in `docs/physics-contracts.md` §7.4: new URDFs
should declare steer joint axes as `<axis xyz="0 0 -1"/>` so that
`+joint_angle` and `+steer` (user-facing ISO 8855) share the same sign
domain. The SDK's `VisualSync` still compensates for either axis convention
via `visual_cmd = -phys * sign`, so existing URDFs with `(0, 0, 1)` (e.g.
JMK) keep working — the recommendation is only for new URDFs.

### Fixed in repo
- `GeneVehicle_Truck6w/urdf/truck_6w.urdf` — steer joint axes changed from
  `(0, 0, 1)` to `(0, 0, -1)` to match the recommendation. Visual behavior
  was already correct (v0.5.3 fix), but the URDF now follows the
  recommended convention internally.

---

## [0.5.3] — 2026-05-18

### Fixed — steer wheel visual rotated the wrong direction

`VisualSync` was driving the steer joints **opposite to the physics-side
steering direction** for both URDF axis conventions:

- URDF axis `(0, 0, 1)` (e.g. truck preset, JMK URDF) — visual wheels rotated LEFT when physics rotated RIGHT, and vice versa.
- URDF axis `(0, 0, -1)` (HJW URDF) — same inversion.

The old formula `visual_cmd = phys * sign` assumed the only conversion
needed was the URDF axis flip captured in `sign`. It missed that the
physics-side `steer_per_wheel` is **opposite-handed** from the URDF joint
convention to begin with (physics +θ = right turn = CW from above; URDF
axis `(0,0,1)` +joint = CCW from above). The correct formula is
`visual_cmd = -phys * sign`:

- axis `(0, 0, 1)`: `sign=+1` → `visual_cmd = -phys` → joint goes CW for +phys → right turn visual ✓
- axis `(0, 0, -1)`: `sign=-1` → `visual_cmd = +phys` → joint goes CW for +phys → right turn visual ✓

Demos affected: HJW (4-wheel car, axis -1) and Truck6w (axis +1). The
4-wheel HJW demo never had the wheel direction verified visually; the
truck demo just hit it (front wheels turned left when the truck arced
right). KDU (skid-steer, no steer joints) and JMK demos are unaffected.

---

## [0.5.2] — 2026-05-18

### Changed — `truck_6w_partial_ackermann` preset

- **Drive layout fixed to match real trucks.** Was `AWD` with uniform 1/6 split across all 6 wheels (front-axle steered wheels were also driven). Now `RWD(driven_axles=(1, 2))` — only the middle and rear axles drive, the front axle is steered-only. This matches conventional truck drivetrain layouts (front steer, rear/mid drive) and stops the visible weirdness of the steering wheels also producing forward torque.
- **Torque sized for the truck's mass.** `t_drive_max` 1500 → 10_000 N·m (was producing ~0.13 m/s² acceleration on a 5-ton truck after rolling resistance — barely visible). `t_brake_max` 3500 → 8000 N·m for symmetry. At full throttle the truck now hits ~2-3 m/s² acceleration without wheelspin (Pacejka peak per driven wheel at equilibrium ≈ 8800 N vs ~2500 N required).

---

## [0.5.1] — 2026-05-18

### Fixed — preset vehicles could not start from rest under throttle

The `"control"` stability profile built `LowSpeedRegularizer` with
`disable_when_control_active=False` (set in v0.3.0). At `v=0, omega=0`,
the regularizer:

1. Scaled `F_long` and `F_lat` by `moving = 0` → no propulsion force.
2. Set `omega_pull_factor = 1`, `omega_pull_target = v_long / radius = 0`
   → forced `omega` back to 0 every step.

Combined: any preset vehicle (`car_4w_rwd_ackermann`, `tank_10w_skid_belt`,
`truck_6w_partial_ackermann`, …) issuing `throttle > 0` at rest would
spin its wheels for one step and then have `omega` snapped back to 0.
**Vehicle stuck at rest indefinitely.** Discovered while running the
6-wheel truck demo where the truck wouldn't accelerate.

### Reverted — `disable_when_control_active=True` is back in `"control"` profile

This restores the original HJW behavior: the regularizer is off when the
user is actively throttling or braking. The vehicle can accelerate from
rest as expected. The regularizer still fires when the chassis is at
rest with no input (suppressing drift jitter), which is its original
intent.

### Trade-off note

The v0.3.0 change was made on the hypothesis that an always-on
regularizer would help MPPI low-speed uphill oscillation. That
hypothesis was never validated and the resulting behavior breaks every
demo. MPPI users who want an always-on regularizer can opt in
explicitly via `stability="research"` + a custom hook list:

```python
cfg.stability_hooks = [
    RollingResistance(),
    LowSpeedRegularizer(disable_when_control_active=False),
]
```

### Test update

`test_profile_control_uses_mppi_friendly_lowspeed_default` →
`test_profile_control_disables_regularizer_under_throttle`. Same assertion
flipped (`False` → `True`).

---

## [0.5.0] — 2026-05-18

### Performance — vectorized per-wheel pipeline

The Python `for i in range(n_wheels)` loop inside `VehiclePhysics.step()` is gone. All per-wheel work — suspension N, wheel-frame fwd/lat transforms, slip, tire force, stability hooks, omega update, force accumulation — is now a single batched tensor op set operating on `(n_envs, n_wheels)` tensors.

Measured on the KDU 10-wheel tank interactive demo:
- Before (v0.4.4): physics ≈ 50 ms / step (chase-cam interactive, `--profile` mode)
- 250+ CUDA kernel launches per step (10 wheels × ~25 ops per wheel)
- At `n_envs=1`, launch overhead (~10-30 µs each) dominated the actual GPU work, costing 5-10 ms in pure dispatch overhead alone.

After v0.5.0: ~25-30 kernel launches per step regardless of `n_wheels`. Helps interactive single-env use (where overhead was the bottleneck) AND batched RL/MPPI (cleaner code, same speed or better).

### BREAKING — stability hook API

- `StabilityHook.apply_post_tire(ctx, wheel_idx)` → `StabilityHook.apply_post_tire(ctx)`. No more per-wheel index. All `ctx.F_long`, `ctx.F_lat`, `ctx.N`, `ctx.v_long`, `ctx.v_lat` are now `(n_envs, n_wheels)` tensors. Hooks operate on the whole batch at once.
- `PipelineContext.wheel_params` (list of WheelConfig) → `PipelineContext.wheel_meta` (the cached `WheelMeta` with batched coefficient tensors). Read per-wheel coefficients via `ctx.wheel_meta.mu_long`, `ctx.wheel_meta.pb_x`, etc.
- All three bundled hooks (`RollingResistance`, `LowSpeedRegularizer`, `StaticFrictionLock`) updated. Custom hooks need to be ported.

### BREAKING — TireModel API

- `TireModel.__call__(v_long, v_lat, v_roll, N, wheel_params)` now expects `(n_envs, n_wheels)` tensors and a `WheelMeta` (not a single `WheelConfig`). Both `PacejkaAnisotropic` and `CoulombIsotropic` read per-wheel coefficients from the meta. Custom tire models need to be ported.

### Added — `WheelMeta` batched coefficient tensors

`WheelMeta` now exposes per-wheel `(n_wheels,)` tensors for: `mu_long`, `mu_lat`, `rolling_resistance_cr`, `pb_x`, `pc_x`, `pe_x`, `pb_y`, `pc_y`, `pe_y`. Built once at `VehiclePhysics.__init__` from the resolved `WheelConfig` list. Cheap to broadcast against `(n_envs, n_wheels)` state.

### Removed — `dynamics.suspension_normal_force` (still importable)

The pure-Python helper is still exported and unit-tested, but `core.py` no longer calls it — the suspension N calculation is inlined as a batched op in `step()` for one less function-call layer. Helper remains for external use.

### Other

- `StaticFrictionLock.apply_post_tire` no longer does the `if not active.any(): return` short-circuit (forced a CPU↔GPU sync per step). The branchless batched path is always taken.

### Migration

External hooks / tire models with the v0.4.x per-wheel API will break — update to the batched signature. No change needed for code that only uses the public `VehiclePhysics` / `VehicleInputs` / presets / strategy APIs.

---

## [0.4.4] — 2026-05-18

### Performance
- **Tank chase-cam fps fix** — interactive demo on the KDU tank ran at ~15 fps vs ~25 fps for the original `KDU/example_interactive.py`. Two contributions:
  - `VisualSync` was syncing wheel spin angle (`set_dofs_position` for the 10 continuous spin joints) every step. The original KDU intentionally skipped this since the cylinder primitive wheels are rotationally symmetric and a spinning visual is invisible.
  - The chassis `up_world` reference tensor was being re-allocated every step inside `VehiclePhysics.step()` instead of being cached at init time.

### Added
- `VehicleConfig.visual_spin_enabled: bool = True` (also on `ResolvedConfig`). Set `False` to skip the per-step spin-angle `set_dofs_position` call. Saves ~3-5 ms / step in interactive mode (one fewer Genesis call). Cars keep the default `True` so mesh wheels visibly roll; tanks (`tank_10w_skid_belt` preset) now default to `False`.

### Changed
- `VehiclePhysics.__init__` caches `_up_world` once instead of re-creating it each `step()`.
- `VisualSync` integrates the visual spin angle in place (`add_`) when enabled.

---

## [0.4.3] — 2026-05-18

### Fixed
- **Heavy-wheel suspension visual fix (two-part)** — on the KDU 10-wheel tank (each wheel ≈ 500 kg):
  - *Part 1 (free-fall)*: the wheel meshes were sinking below the ground while the chassis stayed up. `VisualSync` chose `set_dofs_position` vs `control_dofs_position` by detecting `<dynamics>` declarations in the URDF, which KDU's URDF omits entirely. With no PD control (`kp`/`kv` = 0), Genesis lets the heavy wheel free-fall between substeps and the kinematic `set_dofs_position` cannot snap it back fast enough.
  - *Part 2 (spinning in air)*: after the PD fix above, wheels stopped falling but were spinning in air ~5 cm above ground. The `control_dofs_position` path used `target = max(0, rest_d - d)` (a non-negative compression), which can only push the wheel UP from rest, never DOWN to reach the ground. This is fine when the carrier rest position sits *above* ground level (HJW URDF: susp origin z = 0.34), but breaks when the carrier rest sits *at* base_link z (KDU URDF: susp origin z = 0). The `control_dofs_position` path now uses the same `joint_pos = mesh_radius - d` formula as the `set_dofs_position` path so the wheel mesh lands on the ground in both cases.

### Added
- `VehicleConfig.visual_susp_mode: str = "auto"` (also on `ResolvedConfig`). Three values:
  - `"auto"` (default) — per-joint decision based on URDF `<dynamics>` presence; preserves the v0.4.2 behavior for HJW-style URDFs.
  - `"kinematic"` — force `set_dofs_position` everywhere (light wheels, HJW-style).
  - `"control"` — force `control_dofs_position` with PD kp=1e7 / kv=1e5 everywhere (heavy wheels, KDU-style).
  - `resolve()` validates the value and raises `ConfigError` on unknown strings.
- `tank_10w_skid_belt` preset now sets `visual_susp_mode="control"` so the wheels stay attached on the KDU tank without the user having to know.

### Migration
- HJW / JMK / car presets unchanged — `"auto"` keeps their existing behavior.
- If you constructed a `VehicleConfig` manually for a heavy-wheel vehicle and saw the wheel-sinking bug, set `visual_susp_mode="control"` explicitly.

---

## [0.4.2] — 2026-05-18

### Added
- `genesis_vehicle.scene_helpers` module with two thin Genesis-side
  convenience helpers:
  - `make_wheel_raycaster(scene, entity, urdf_path, ...)` — replaces the
    five-line `parse_urdf` + `WheelRayPattern` + `gs.sensors.Raycaster` +
    `scene.add_sensor` boilerplate with one call.
  - `add_vehicle(scene, urdf_path, preset_fn=None, ...)` — adds the URDF
    entity + raycaster (and optionally builds the preset `VehicleConfig`)
    in one call. Returns `(car, sensor, cfg)`. Caller still owns
    `scene.build()` and `VehiclePhysics(...)` construction so other
    entities / sensors can be added between.
- Both are lazy-imported through the top-level package, so
  `from genesis_vehicle import add_vehicle` works without Genesis
  installed at import time.

### Design note
The SDK intentionally does NOT fully encapsulate Genesis. Helpers cut the
canonical boilerplate for the common case; full-encapsulation facades for
CARLA-style or Gym-style use will live under `genesis_vehicle.adapters/`
when needed. This keeps the core API research-friendly (viewer, terrain,
custom materials, multi-agent scenes all stay accessible) while still
offering a one-liner for the common path.

---

## [0.4.1] — 2026-05-17

### Added
- `car_4w_fwd_ackermann` preset — 4-wheel front-wheel-drive car with front Ackermann steering. Fills the obvious gap between the existing `car_4w_rwd_ackermann` and `car_4w_awd_ackermann` presets (real-world passenger cars are mostly FWD). Mirrors the RWD preset's parameters with `FWD(driven_axles=(0,))`.

---

## [0.4.0] — 2026-05-17

### Changed (BREAKING)
- **Stability profile renamed**: `"physical"` → `"raw"`. No deprecated alias kept.
  - **Reason for break.** The original `"physical"` name implied "the physically correct config" or "the right profile for parameter fitting", which is incorrect. Fitted parameters in this SDK are always model-coupled (Pacejka `B` is the coefficient that fits real inside our SDK; it is not "rubber's true stiffness"). Hooks are part of the model. The right Real2Sim rule is *"fit with the same profile you'll deploy with"* — for our SDK that is almost always `"control"`. The renamed `"raw"` profile is now correctly framed: hook-free raw dynamics for **debugging / ablation**, not a general fitting recommendation.
  - **No alias kept.** Pre-1.0 versioning + essentially no external users; keeping the misleading name even as a deprecated alias would perpetuate the confusion the rename is meant to resolve.
- API.md S6 rewritten around "one rule: fitting forward model = deployment forward model", with `"raw"` reframed as an analysis tool rather than a Real2Sim recommendation.

### Migration
- Replace every `stability="physical"` with `stability="raw"` (or, more likely, with the default `"control"`, since most users were on `"physical"` because the doc misled them).
- If you were on `"physical"` for Real2Sim fitting and plan to deploy with `"control"`, **change fitting to `"control"`** so the forward models match.

---

## [0.3.0] — 2026-05-17

### Added
- **Stability profile** on every preset: `stability="control" | "physical" | "research"` (default `"control"`).
  - `"control"` — RL/MPPI-friendly hooks: `RollingResistance` + `LowSpeedRegularizer(disable_when_control_active=False)` (+ `StaticFrictionLock` for tank). Default for `car_4w_rwd_ackermann`, `car_4w_awd_ackermann`, `truck_6w_partial_ackermann`, `tank_10w_skid_belt`.
  - `"physical"` — empty hook list. Use for Real2Sim parameter identification so numerical stabilizers don't leak into fitted parameters.
  - `"research"` — empty hook list. Caller assembles hooks manually.
- `stability_hooks_for_profile(profile, vehicle_kind)` public helper to materialize the same hook lists used by the presets.
- Version API: `__version__`, `VERSION_INFO`, `version()`, `version_info()` exported from `genesis_vehicle`.
- One-time version banner printed on first `VehiclePhysics` initialization:
  ```
  [genesis_vehicle v0.3.0] Initialized: 4 wheels, Ackermann, RWD, Independent, n_envs=1, hooks=[RollingResistance, LowSpeedRegularizer]
  ```
- `CHANGELOG.md` (this file) tracking version history.

### Changed
- **Footgun fix** — `"control"` profile builds `LowSpeedRegularizer` with `disable_when_control_active=False`. The legacy default (`True`) made the regularizer a no-op exactly when MPPI was applying throttle, so the documented "low-speed stabilization" never fired under control. Direct construction (`LowSpeedRegularizer()`) keeps the legacy default for backwards compatibility.

### Notes
- The hook system is now treated as a stability profile rather than a free-form list assembled by the user. This aligns with industry vehicle SDKs (PhysX sticky tire mode, Jolt anti-rollback, Chaos low-speed threshold, Unity WheelCollider stick-slip) where low-speed numerical stabilization is part of the model, not an opt-in plug-in. Advanced users can still pass `stability="research"` and assemble `cfg.stability_hooks` manually.

---

## [0.2.0]

Reviewer-driven revisions of the v0.1.0 reference implementation. The SDK's
public contracts and helper boundaries were tightened ahead of the first
variant migration.

### Added
- `genesis_vehicle/dynamics.py` — pure-Python primitives:
  - `brake_torque_signed(t_brake, omega, smoothing_scale=0.5)` — positive brake magnitude → signed torque opposing wheel rotation.
  - `suspension_normal_force(compression, comp_rate, k_susp, c_compression, c_extension, air_mask)` — asymmetric damper + non-negative clamp + air-mask zero in one call.
- `VehicleStepInputs` typing alias on `VehiclePhysics.step()` (Union of `VehicleInputs` + every typed input).
- `estimate_spin_inertia_from_genesis(entity, link_name, spin_axis_local=None)` with optional axis projection (`a^T diag(I) a`).
- New tests:
  - `tests/test_dynamics.py` (8 tests) for `brake_torque_signed` and `suspension_normal_force`.
  - `tests/test_config_resolve.py::test_user_explicit_i_wheel_wins_over_urdf`, `test_user_explicit_radius_wins_over_urdf`.

### Changed
- **`i_wheel` truth policy** — `WheelConfig.i_wheel` set by the user (or populated by URDF) is now authoritative; the Genesis-runtime spin-inertia refinement only fires for wheels where neither the user nor the URDF supplied a value. Previously the Genesis estimate always overrode.
- **Inertia helper renamed**: `parse_inertia_max_principal_genesis` → `estimate_spin_inertia_from_genesis`. The old name is kept as a deprecated alias and will be removed in a later release.
- `core.py` refactored to call the new `dynamics.py` helpers instead of inlining the brake/suspension formulas.
- `__init__.py` exposes the full public surface (strategies, tire models, presets) as a single import path: `from genesis_vehicle import ...`.

### Documentation
- API.md restructured into 11 sections; new "Physics contracts" section codifies brake-sign, non-negative N, `i_wheel` policy, ISO 8855 steer convention, and coupling order. Migration cheatsheet now uses the "URDF default, API override wins" wording instead of the misleading "URDF as truth".

---

## [0.1.0]

Initial Phase 3 reference implementation of the genesis_vehicle SDK.

### Added
- Top-level `VehiclePhysics` driver running the shared 5-step ray-wheel pipeline batched over `n_envs ≥ 1`.
- Strategy abstractions: `SteeringStrategy` (`Ackermann`, `PartialAckermann`, `SkidSteer`, `NoSteer`), `DrivetrainStrategy` (`FWD`, `RWD`, `AWD`, `PerSide`), `CouplingStrategy` (`Independent`, `SameSideBelt`), `StabilityHook` (`RollingResistance`, `LowSpeedRegularizer`, `StaticFrictionLock`).
- Tire models: `PacejkaAnisotropic`, `CoulombIsotropic`.
- Config & merge: `WheelConfig`, `ChassisConfig`, `VehicleConfig`, `ResolvedConfig`, `resolve()`, `ConfigError`. URDF defaults + per-wheel overrides.
- Inputs: unified `VehicleInputs` + typed (`AckermannInputs`, `SkidSteerInputs`, etc.) with `from_unified()` adapters.
- `parse_urdf()` convention-based parser supporting HJW (`*_suspension_joint`) and KDU (`*_susp`) naming.
- Visual layer (`VisualSync`) absorbing URDF axis-sign quirks.
- Presets for four reference vehicles: `car_4w_rwd_ackermann`, `car_4w_awd_ackermann`, `truck_6w_partial_ackermann`, `tank_10w_skid_belt`.
- Pure-Python tests for URDF parsing, config resolve, strategy math.
