"""WP4-10: the runtime runs a biped_v1 policy in the loop, as training defined it.

Every period the bridge records the state it sent and the targets it got back.
Replaying those states through the Python reference (observation builder of
``runtime_bridge`` + numpy MLP) must give the same targets: that pins the
observation layout, the last-action feedback and the action scaling of
``runtime/src/policy.rs`` to the training env.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from bipedlab import mlp_export as me
from bipedlab import runtime_bridge as rb

pytestmark = pytest.mark.runtime

BINARY = Path(os.environ.get("BIPEDLAB_RUNTIME_BIN", rb.DEFAULT_BINARY))
OBS, ACT = 39, 10
SCALE = 0.5


@pytest.fixture(scope="module", autouse=True)
def _binary():
    if not BINARY.exists():
        if os.environ.get("BIPEDLAB_REQUIRE_RUNTIME") == "1":
            pytest.fail(f"runtime binary missing: {BINARY}")
        pytest.skip(f"runtime binary not built: {BINARY}")


def small_policy(seed: int, gain: float = 0.05) -> me.Mlp:
    """A random 39 -> 64 -> 64 -> 10 ELU actor with small outputs (the robot keeps standing)."""
    rng = np.random.default_rng(seed)
    dims = [OBS, 64, 64, ACT]
    ops: list[me.Op] = []
    for k, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        w = (rng.normal(size=(a, b)) / np.sqrt(a)).astype(np.float32)
        if k == len(dims) - 2:
            w *= gain
        ops.append(me.Op(me.DENSE, w, (rng.normal(size=b) * 0.01).astype(np.float32)))
        if k < len(dims) - 2:
            ops.append(me.Op(me.ELU, params=(1.0,)))
    return me.Mlp(OBS, ops)


def replay(mlp: me.Mlp, trace: list[dict], stand: np.ndarray) -> np.ndarray:
    last = np.zeros(ACT, dtype=np.float32)
    out = []
    for f in trace:
        obs = rb.policy_observation(f["gyro"], f["quat"], f["qpos"], f["qvel"], stand, last)
        last = mlp.forward(obs)
        out.append(stand + SCALE * last.astype(np.float64))
    return np.array(out)


def test_runtime_policy_matches_the_python_reference(tmp_path):
    mlp = small_policy(seed=0)
    path = mlp.save(tmp_path / "policy.bpml")
    r = rb.run_with_runtime(
        BINARY, ["--controller", f"policy:{path}"], rb.BridgeConfig(seconds=3.0, record=True)
    )
    assert r["exit_code"] == 0, r["stderr"]
    sim, rt = r["sim"], r["runtime"]
    assert sim["steps"] == 150 and rt["latched"] == 0
    assert sim["bits_seen"] == 0, rt  # no clamp, slew or derate event altered the targets
    trace = sim["trace"]
    got = np.array([f["targets"] for f in trace])
    want = replay(mlp, trace, rb.SimBridge().default)
    np.testing.assert_allclose(got, want, atol=1e-5, rtol=0)
    # The policy did move the joints, so the comparison is not trivially the stand pose.
    assert np.abs(got - rb.SimBridge().default).max() > 1e-3


def test_supervisor_still_filters_policy_targets(tmp_path):
    """A policy that asks for large steps is slew-limited, not obeyed (ADR-0006)."""
    path = small_policy(seed=1, gain=20.0).save(tmp_path / "wild.bpml")
    r = rb.run_with_runtime(
        BINARY, ["--controller", f"policy:{path}"], rb.BridgeConfig(seconds=1.0)
    )
    assert r["exit_code"] == 0, r["stderr"]
    assert r["sim"]["bits_seen"] & (rb.EVENT_SLEW_LIMIT | rb.EVENT_POSITION_CLAMP)
    assert r["sim"]["max_target_step_rad"] <= 10.0 * 0.02 + 1e-9


def test_wrong_size_policy_is_refused(tmp_path):
    rng = np.random.default_rng(0)
    w = rng.normal(size=(61, 14)).astype(np.float32)
    path = me.Mlp(61, [me.Op(me.DENSE, w, np.zeros(14, np.float32))]).save(tmp_path / "md.bpml")
    r = rb.run_with_runtime(BINARY, ["--controller", f"policy:{path}"], rb.BridgeConfig())
    assert r["exit_code"] == 2 and "error" in r["sim"]
    assert "policy is 61 -> 14, the robot needs 39 -> 10" in r["stderr"]
