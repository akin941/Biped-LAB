"""Process-parallel batch env matches the serial one (WP3-06)."""

import numpy as np
import pytest

from bipedlab import upstream
from bipedlab.rl.batch_env import MicroduckBatchEnv
from bipedlab.rl.parallel_env import ParallelBatchEnv, shard_sizes
from bipedlab.sim import microduck_policy as mp

pytestmark = pytest.mark.upstream


@pytest.fixture(scope="module")
def scene(microduck_rl_path):
    return upstream.microduck_scene("scene.xml")


def test_shard_sizes():
    assert shard_sizes(10, 3) == [4, 3, 3]
    assert shard_sizes(4, 4) == [1, 1, 1, 1]
    with pytest.raises(ValueError):
        shard_sizes(2, 3)
    with pytest.raises(ValueError):
        shard_sizes(2, 0)


def test_matches_serial_env_including_resets(scene):
    # Short episodes and large actions so both falls and time-limit resets happen.
    n, steps = 5, 40
    actions = np.random.default_rng(3).uniform(-1.0, 1.0, (steps, n, mp.ACT_DIM))
    serial = MicroduckBatchEnv(scene, n, episode_s=0.3, seed=11)
    with ParallelBatchEnv(scene, n, workers=2, episode_s=0.3, seed=11) as parallel:
        np.testing.assert_array_equal(serial.obs, parallel.obs)
        resets = 0
        for a in actions:
            s, p = serial.step(a), parallel.step(a)
            for field in ("obs", "reward", "terminated", "truncated", "final_obs"):
                np.testing.assert_array_equal(getattr(s, field), getattr(p, field))
            np.testing.assert_array_equal(serial.episode_steps, parallel.episode_steps)
            resets += int((s.terminated | s.truncated).sum())
    assert resets > n  # the comparison covered resets, not just the first episode


def test_rejects_bad_actions_and_closes(scene):
    with ParallelBatchEnv(scene, 2, workers=2) as env:
        with pytest.raises(ValueError, match="shape"):
            env.step(np.zeros((3, mp.ACT_DIM)))
        with pytest.raises(ValueError, match="finite"):
            env.step(np.full((2, mp.ACT_DIM), np.inf))
        env.step(np.zeros((2, mp.ACT_DIM)))  # still usable
    assert env._procs == []


def test_worker_construction_error_is_reported(tmp_path):
    with pytest.raises(RuntimeError, match="env worker failed"):
        ParallelBatchEnv(tmp_path / "missing.xml", 2, workers=1)
