"""Wheel pixels carry the wheel's segmentation id at every n_envs (2.0.6,
AC2-SEG gate 2 in-suite).

Through 2.0.5 the n_envs 1 instanced wheel renderer hid the URDF wheel visuals
and drew SDK nodes with no segmentation id, so wheel pixels read as background
(0 of 160 red wheel pixels carried a wheel key in the plan's prototype). Since
2.0.6 the engine's own wheel nodes are drawn (posed from the closed form), so
they keep the engine's registration at every level.

| abbr | meaning |
|---|---|
| gate 2 | recall: the fraction of red (wheel-coloured) pixels whose segmentation id maps to a wheel link's key |
| L1 / L2 / L3 | n_envs 1 / n_envs 2 with ``rendered_envs_idx=[1]`` / n_envs 2 with ``[0, 1]`` and env spacing 8 (top camera) |
| R1 | the 2026-11-25 plan-review ruling that amended gate 2: L1 / L2 graded at >= 0.90 on frames with >= 100 red pixels, every listed L3 frame graded at >= 0.85 regardless of pixel count, L3 frame 0 not graded |

Thresholds as amended by R1. **They are segmentation-presence regression bounds
fixed after the prototype; they do not detect a wheel drawn at the wrong pose**
(a no-pose-override mutant passes them) — the pose is checked by
``test_render_sync_pose.py``.

Scene: the acceptance scene (``tests/data/car_4w_redwheels.urdf``, K = 2, dt
0.025 x 10, 60 brake steps then throttle 0.2 / steer on the last env), visual
mode, 100-step prefix of the 180-step schedule, frames 60 and 100 (200x150, fov
50, ambient light only, black background). Measured once before commit
(implement agent, 2026-11-25, CPU/WSL2, genesis-world 1.5.0): red / wheel-keyed
pixel counts equal the acceptance run's (``.pm_r2/r206/s5r/SEG_{L1,L2,L3}_*``)
at both frames. Pixel thresholds are PLATFORM-GATED (linux-x86_64,
genesis-world 1.5.0, where they were measured); elsewhere the test skips with
the reason. No collision-mode layout here (its colours are random per process).

Wall time: about 20 s on the measuring machine (18.9 s in the suite run; 5 scene builds).
"""

import os
import platform
import sys

import numpy as np
import pytest

pytest.importorskip("genesis")

import torch
import genesis as gs

from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann

URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                    "car_4w_redwheels.urdf")
K = 2
MEASURED_ON = ("1.5.0", "linux-x86_64")


@pytest.fixture(scope="module")
def cpu_genesis():
    if not getattr(gs, "_initialized", False):
        try:
            VehicleScene.init_backend("cpu")
        except Exception as e:                     # pragma: no cover
            pytest.skip(f"genesis CPU backend unavailable: {e}")
    return gs


@pytest.fixture(scope="module")
def measured_platform(cpu_genesis):
    here = (gs.__version__, f"{sys.platform}-{platform.machine()}")
    if here != MEASURED_ON:
        pytest.skip(f"pixel thresholds measured on genesis-world {MEASURED_ON[0]} / "
                    f"{MEASURED_ON[1]}; this is {here[0]} / {here[1]}")
    try:
        sc = gs.Scene(show_viewer=False)
        sc.add_entity(gs.morphs.Plane())
        cam = sc.add_camera(res=(32, 24), GUI=False)
        sc.build()
        cam.render(rgb=True)
    except Exception as e:                         # pragma: no cover
        pytest.skip(f"no offscreen rasterizer context: {type(e).__name__}: {e}")
    return True


def _frames(layout, level, frames=(60, 100)):
    """Render the layout and return {k: (red_px, red_and_wheel_keyed_px)}."""
    from genesis.vis.rasterizer_context import SEGMENTATION_LEVELS
    ne, ridx, sp, top = {"L1": (1, None, 0.0, False), "L2": (2, [1], 0.0, False),
                         "L3": (2, [0, 1], 8.0, True)}[layout]
    st = np.random.get_state()
    np.random.seed(0)
    try:
        vk = dict(segmentation_level=level, shadow=False, ambient_light=(1.0, 1.0, 1.0),
                  background_color=(0.0, 0.0, 0.0), lights=[])
        if ridx is not None:
            vk["rendered_envs_idx"] = ridx
        vs = VehicleScene(dt=0.025, substeps=10, n_envs=ne, show_viewer=False,
                          vis_options=gs.options.VisOptions(**vk),
                          raycast_mode="dual_scene", init_genesis=False)
        vs.add_ground_plane(friction=1.0)
        cam = vs.add_camera(pos=(8, 6, 6), lookat=(0, -3, 0.3), GUI=False,
                            res=(200, 150), fov=50.0).entity
        cfg = car_4w_rwd_ackermann(URDF, stability="control")
        vehs = [vs.add_vehicle(URDF, car_4w_rwd_ackermann, cfg=cfg,
                               pos=(0.0, -6.0 * i, 1.0), name=f"v{i}",
                               material=gs.materials.Rigid(friction=1.0))
                for i in range(K)]
        vs.build(**({"env_spacing": (sp, sp)} if sp else {}))
    finally:
        np.random.set_state(st)
    gscene = vs.scene
    eoff = np.asarray(gscene.envs_offset) if ne > 1 else np.zeros((1, 3))
    obs = ridx[-1] if ridx else ne - 1
    L = SEGMENTATION_LEVELS.index(level)
    wheel_keys = {g.seg_keys[L]
                  for v in vehs for w in cfg.wheels
                  for g in list(v.entity_main.get_link(w.name).vgeoms)
                  + list(v.entity_main.get_link(w.name).geoms)}
    out = {}
    for k in range(1, max(frames) + 1):
        for v in vehs:
            if k <= 60:
                v.set_inputs(throttle=0.0, brake=1.0, steer=0.0)
            else:
                st_ = 1.0 if k <= 140 else -1.0
                v.set_inputs(throttle=0.2, brake=0.0,
                             steer=torch.tensor([0.0] * (ne - 1) + [st_]) if ne > 1 else st_)
        vs.step()
        if k not in frames:
            continue
        if top:
            envs = ridx
            c = np.mean([np.mean([v.get_pos().reshape(ne, 3)[e].cpu().numpy() for v in vehs], axis=0)
                         + eoff[e] for e in envs], axis=0)
            d = 1.0 + 0.6 * sp * (len(envs) - 1) / 8.0
            cam.set_pose(pos=c + d * np.array([9.0, 7.0, 6.0]),
                         lookat=c + np.array([0.0, 0.0, 0.2]), up=(0.0, 0.0, 1.0))
        else:
            c = np.mean([v.get_pos().reshape(ne, 3)[obs].cpu().numpy() for v in vehs], axis=0)
            c = c + (eoff[obs] if ne > 1 else 0)
            cam.set_pose(pos=c + np.array([7.0, 5.0, 6.0]),
                         lookat=c + np.array([0.0, 0.0, 0.2]), up=(0.0, 0.0, 1.0))
        rgb, _, seg, _ = cam.render(rgb=True, depth=True, segmentation=True)
        rgb, seg = (np.asarray(x.cpu() if torch.is_tensor(x) else x) for x in (rgb, seg))
        if rgb.ndim == 4:
            j = (ridx.index(obs) if ridx else obs) if rgb.shape[0] > 1 else 0
            rgb, seg = rgb[j], seg[j]
        f = rgb[..., :3].astype(np.float32) / 255.0
        red = (f[..., 0] > 0.6) & (f[..., 1] < 0.25) & (f[..., 2] < 0.25)
        idmap = gscene.segmentation_idx_dict
        seg = seg.astype(np.int64)
        widx = [int(ix) for ix in np.unique(seg)
                if int(ix) in idmap and idmap[int(ix)] in wheel_keys]
        isw = np.isin(seg, widx) if widx else np.zeros(seg.shape, bool)
        out[k] = (int(red.sum()), int((red & isw).sum()))
    return out


@pytest.mark.parametrize("level", ["entity", "link", "geom"])
def test_l3_both_envs_rendered_wheel_pixels_are_wheel_keyed(measured_platform, level):
    """L3, frames 60 and 100: recall >= 0.85 on every frame regardless of pixel
    count (R1)."""
    got = _frames("L3", level)
    for k, (red, hit) in got.items():
        assert red > 0, f"frame {k}: no red wheel pixel in view (scene premise)"
        assert hit / red >= 0.85, f"{level} frame {k}: recall {hit}/{red}"


@pytest.mark.parametrize("layout", ["L1", "L2"])
def test_l1_l2_wheel_pixels_are_wheel_keyed(measured_platform, layout):
    """L1 (the n_envs 1 case 2.0.5 got wrong) and L2 at link level, frames 60
    and 100: recall >= 0.90 on frames with >= 100 red pixels (R1); both frames
    carry more than 100 red pixels in this scene, so both are graded."""
    got = _frames(layout, "link")
    for k, (red, hit) in got.items():
        assert red >= 100, f"frame {k}: {red} red px (scene premise: graded frame)"
        assert hit / red >= 0.90, f"{layout} frame {k}: recall {hit}/{red}"
