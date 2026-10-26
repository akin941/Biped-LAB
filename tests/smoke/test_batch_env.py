"""Batched Microduck environment used by the PPO benchmark (WP3-01)."""

import numpy as np
import pytest

from bipedlab import upstream
from bipedlab.rl.batch_env import MicroduckBatchEnv
from bipedlab.sim import microduck_policy as mp

pytestmark = pytest.mark.upstream


@pytest.fixture(scope="module")
def scene(microduck_rl_path):
    return upstream.microduck_scene("scene.xml")


def test_shapes_and_upright_reward(scene):
    env = MicroduckBatchEnv(scene, num_envs=3)
    assert env.obs.shape == (3, mp.OBS_DIM) and env.obs.dtype == np.float32
    out = env.step(np.zeros((3, mp.ACT_DIM)))
    assert out.reward.shape == (3,) and out.reward.dtype == np.float32
    # Holding DEFAULT_POSE from the upstream start stays upright: alive + upright ~ 2.
    assert np.all(out.reward > 1.9)
    assert not out.terminated.any() and not out.truncated.any()


def test_same_seed_is_deterministic(scene):
    actions = np.random.default_rng(1).uniform(-0.3, 0.3, (20, 2, mp.ACT_DIM))
    runs = []
    for _ in range(2):
        env = MicroduckBatchEnv(scene, num_envs=2, seed=7)
        runs.append(np.stack([env.step(a).obs for a in actions]))
    np.testing.assert_array_equal(runs[0], runs[1])


def test_time_limit_truncates_and_resets(scene):
    env = MicroduckBatchEnv(scene, num_envs=2, episode_s=0.1)  # 5 control steps
    for _ in range(4):
        assert not env.step(np.zeros((2, mp.ACT_DIM))).truncated.any()
    out = env.step(np.zeros((2, mp.ACT_DIM)))
    assert out.truncated.all() and not out.terminated.any()
    assert np.all(env.episode_steps == 0)
    assert not np.array_equal(out.final_obs, out.obs)


def test_fall_terminates(scene):
    env = MicroduckBatchEnv(scene, num_envs=1, episode_s=20.0)
    rng = np.random.default_rng(0)
    fell = False
    for _ in range(500):
        if env.step(rng.uniform(-1.5, 1.5, (1, mp.ACT_DIM))).terminated.any():
            fell = True
            break
    assert fell


def test_rejects_bad_actions(scene):
    env = MicroduckBatchEnv(scene, num_envs=2)
    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros((1, mp.ACT_DIM)))
    with pytest.raises(ValueError, match="finite"):
        env.step(np.full((2, mp.ACT_DIM), np.nan))


def test_batched_observation_matches_observation_builder(scene):
    env = MicroduckBatchEnv(scene, num_envs=4, seed=5)
    actions = np.random.default_rng(2).uniform(-0.4, 0.4, (15, 4, mp.ACT_DIM))
    for a in actions:
        last = env.last_action.copy()
        out = env.step(a)
        for i, d in enumerate(env.datas):
            if out.terminated[i] or out.truncated[i]:
                continue  # env i was reset after the step
            # Bitwise equal to the per-env builder the ONNX evaluation uses.
            expected = env.observe(d, env.last_action[i], env.command)
            np.testing.assert_array_equal(out.obs[i], expected)
            gz = float(expected[mp.OBS_SLICES["projected_gravity"]][2])
            rate = float(np.sum(np.square(a[i].astype(np.float32) - last[i])))
            np.testing.assert_allclose(out.reward[i], 1.0 - gz - 0.05 * rate, rtol=1e-6)
