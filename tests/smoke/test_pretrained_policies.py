"""Contract checks on the official pretrained weights (when downloaded).

Behaviour quality is *reported* by `bipedlab eval-policy`, not asserted here:
we run MuJoCo position actuators, not the BAM actuator the policies were trained
with, so walking quality is an open question (backlog WP1-06), not a pass/fail gate.
"""

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from bipedlab import policies, upstream  # noqa: E402
from bipedlab.sim import microduck_policy as mp  # noqa: E402

pytestmark = [pytest.mark.upstream, pytest.mark.policy]

NAMES = ["alpha_walking.onnx", "alpha_stand.onnx"]


@pytest.fixture(params=NAMES)
def weights(request):
    try:
        return policies.verified_path(request.param)
    except policies.PolicyFileError as exc:
        pytest.skip(str(exc))


def test_contract_and_finite_rollout(weights, microduck_rl_path):
    policy = mp.OnnxPolicy(weights)  # raises unless 61 -> 14
    result, ctrl, _ = mp.run(upstream.microduck_scene(), policy, duration_s=2.0)
    assert result.finite
    actions = np.stack([e["action"] for e in ctrl.log])
    assert np.isfinite(actions).all()
