"""``VehicleScene.step`` hands the driver a fresh inputs object per step (v1.7.0).

Through v1.6.10 ``VehicleScene.step`` passed the SAME ``VehicleInputs`` object
(``Vehicle._inputs``, created only by ``set_inputs``) on every step, in both
solver modes. A documented extension point wraps ``physics.step`` (e.g. a
traction-control throttle cut installed through ``apply_runtime_config``); a
wrapper that rewrote a field of the object it received therefore rewrote what
``set_inputs`` had set, and every later step drove with the rewritten value.
Since v1.7.0 each step gets a shallow copy. The copy is shallow on purpose
(no per-step tensor clone), so an in-place write into a TENSOR value still
persists — pinned below so the documented limitation stays true.
"""
import os

import pytest
import torch

genesis = pytest.importorskip("genesis")

URDF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                    "samples", "urdf", "car_4w.urdf"))


@pytest.fixture(scope="module", params=["per_vehicle", "batched"])
def scene(request):
    from genesis_vehicle import VehicleScene, car_4w_rwd_ackermann
    VehicleScene.init_backend("cpu")
    vs = VehicleScene(dt=0.025, substeps=4, n_envs=1, show_viewer=False,
                      solver=request.param)
    vs.add_ground_plane(friction=1.0)
    veh = vs.add_vehicle(URDF, car_4w_rwd_ackermann, pos=(0.0, 0.0, 0.6),
                         material=genesis.materials.Rigid(friction=1.0))
    vs.build()
    vs.step()
    return request.param, vs, veh


class _Rewriter:
    """Wraps the driver's ``step``: records what it was handed, then rewrites
    it the way a throttle-cutting wrapper would, then steps."""

    def __init__(self, solver, vs, veh, rewrite):
        # The objects themselves are kept (not their id()): an id is only
        # unique among LIVE objects, and a dropped per-step copy's address is
        # reused by the next one.
        self.seen, self.objs = [], []
        self.rewrite = rewrite
        if solver == "per_vehicle":
            self.owner, self.orig = veh.physics, veh.physics.step

            def step(inputs, *a, **k):
                self._handle(inputs)
                return self.orig(inputs, *a, **k)
        else:
            self.owner, self.orig = vs._mvp, vs._mvp.step

            def step(inputs_list, *a, **k):
                self._handle(inputs_list[0])
                return self.orig(inputs_list, *a, **k)
        self.owner.step = step

    def _handle(self, inputs):
        t = inputs.throttle
        self.seen.append(float(t) if not torch.is_tensor(t) else float(t.reshape(-1)[0]))
        self.objs.append(inputs)
        self.rewrite(inputs)

    def restore(self):
        del self.owner.step          # back to the class method


def _zero_throttle(inputs):
    inputs.throttle = 0.0


def test_a_rewritten_field_does_not_persist_into_the_next_step(scene):
    solver, vs, veh = scene
    spy = _Rewriter(solver, vs, veh, _zero_throttle)
    try:
        veh.set_inputs(throttle=0.7, brake=0.0, steer=0.0)
        vs.step()
        vs.step()
    finally:
        spy.restore()
    assert spy.seen == [pytest.approx(0.7), pytest.approx(0.7)], (
        f"[{solver}] the driver saw throttle {spy.seen}: a wrapper's rewrite "
        f"leaked into the next step")
    assert spy.objs[0] is not spy.objs[1]
    assert spy.objs[0] is not veh._inputs
    assert veh._inputs.throttle == pytest.approx(0.7)   # set_inputs' state intact


def test_an_in_place_tensor_write_still_persists(scene):
    """The documented limit of the SHALLOW copy: a tensor value is shared."""
    solver, vs, veh = scene

    def halve_in_place(inputs):
        inputs.throttle.mul_(0.5)

    spy = _Rewriter(solver, vs, veh, halve_in_place)
    try:
        veh.set_inputs(throttle=torch.tensor([0.8]), brake=0.0, steer=0.0)
        vs.step()
        vs.step()
    finally:
        spy.restore()
    assert spy.seen == [pytest.approx(0.8), pytest.approx(0.4)]
