"""Upstream flat velocity task on CPU MuJoCo (WP3-02)."""

import math

import numpy as np
import pytest

from bipedlab import upstream
from bipedlab.rl import velocity_env as ve
from bipedlab.rl.parallel_env import ParallelBatchEnv
from bipedlab.rl.velocity_env import MicroduckVelocityEnv
from bipedlab.sim import microduck_policy as mp

pytestmark = pytest.mark.upstream

S = mp.OBS_SLICES


@pytest.fixture(scope="module")
def scene(microduck_rl_path):
    return upstream.microduck_scene("scene_walk.xml")


def small_actions(steps, n, seed=0, scale=0.2):
    return np.random.default_rng(seed).uniform(-scale, scale, (steps, n, mp.ACT_DIM))


def test_shapes_and_clean_actor_equals_critic_prefix(scene):
    env = MicroduckVelocityEnv(scene, num_envs=3, noise=False)
    assert env.model.opt.iterations == 10 and env.model.opt.ls_iterations == 20
    assert env.max_episode_steps == 1000
    assert env.obs.shape == (3, 61) and env.critic_obs.shape == (3, ve.CRITIC_DIM) == (3, 70)
    for a in small_actions(10, 3):
        out = env.step(a)
        assert out.obs.dtype == np.float32 and out.critic_obs.dtype == np.float32
        # Without noise and delays the actor sees exactly the critic's clean terms.
        np.testing.assert_array_equal(out.obs, out.critic_obs[:, :61])
        np.testing.assert_allclose(out.reward, sum(out.terms.values()), rtol=1e-5, atol=1e-7)
        assert all(np.isfinite(v).all() for v in out.terms.values())


def test_reset_distribution(scene):
    env = MicroduckVelocityEnv(scene, num_envs=16, seed=3)
    z = np.array([d.qpos[2] for d in env.datas])
    xy = np.array([d.qpos[0:2] for d in env.datas])
    assert np.all((z >= 0.12) & (z <= 0.13)) and np.all(np.abs(xy) <= 0.5)
    for d in env.datas:
        np.testing.assert_allclose(d.qpos[env.qpos_idx], mp.DEFAULT_POSE, atol=1e-7)
        quat = d.qpos[3:7]
        assert abs(quat[1]) < 1e-12 and abs(quat[2]) < 1e-12  # yaw only
    assert np.ptp([2 * math.atan2(d.qpos[6], d.qpos[3]) for d in env.datas]) > 1.0


def test_same_seed_deterministic_and_shards_match(scene):
    actions = small_actions(25, 4, seed=1)
    full = MicroduckVelocityEnv(scene, num_envs=4, seed=7)
    a = MicroduckVelocityEnv(scene, num_envs=2, seed=7, first_env=0)
    b = MicroduckVelocityEnv(scene, num_envs=2, seed=7, first_env=2)
    again = MicroduckVelocityEnv(scene, num_envs=4, seed=7)
    for act in actions:
        out, out2 = full.step(act), again.step(act)
        oa, ob = a.step(act[:2]), b.step(act[2:])
        np.testing.assert_array_equal(out.obs, out2.obs)
        np.testing.assert_array_equal(out.obs, np.concatenate([oa.obs, ob.obs]))
        np.testing.assert_array_equal(out.reward, np.concatenate([oa.reward, ob.reward]))


def test_parallel_matches_serial(scene):
    actions = small_actions(12, 4, seed=2)
    serial = MicroduckVelocityEnv(scene, num_envs=4, seed=5)
    with ParallelBatchEnv(
        scene, 4, workers=2, seed=5, episode_s=20.0, env_cls=MicroduckVelocityEnv
    ) as par:
        np.testing.assert_array_equal(par.critic_obs, serial.critic_obs)
        for act in actions:
            s, p = serial.step(act), par.step(act)
            np.testing.assert_array_equal(s.obs, p.obs)
            np.testing.assert_array_equal(s.critic_obs, p.critic_obs)
            for k in s.terms:
                np.testing.assert_array_equal(s.terms[k], p.terms[k])


def test_twist_sampling_fractions(scene):
    env = MicroduckVelocityEnv(scene, num_envs=1, seed=11)
    draws = []
    for _ in range(6000):
        env._sample_twist(0)
        draws.append(env.twist[0].copy())
    t = np.array(draws)
    zero = np.all(t == 0, axis=1)
    forward = (t[:, 0] >= 0.3) & (t[:, 1] == 0) & (t[:, 2] == 0)
    turn = (t[:, 0] == 0) & (t[:, 1] == 0) & (np.abs(t[:, 2]) >= 0.4)
    # Standing (2 %) is applied after the forward / turn draw, so it removes 2 % of each.
    assert abs(zero.mean() - 0.02) < 0.01
    assert abs(forward.mean() - 0.20 * 0.98) < 0.02
    assert abs(turn.mean() - 0.15 * 0.98) < 0.02
    assert np.all(np.abs(t) <= ve.TWIST_RANGE + 1e-12)


def test_curricula_follow_env_steps(scene):
    assert ve.stage_value(ve.ACTION_RATE_STAGES, 0) == -0.1
    assert ve.stage_value(ve.ACTION_RATE_STAGES, 500 * 24 - 1) == -0.1
    assert ve.stage_value(ve.ACTION_RATE_STAGES, 500 * 24) == -0.2
    assert ve.stage_value(ve.ACTION_RATE_STAGES, 10**9) == -1.0
    assert ve.stage_value(ve.HEAD_BIAS_STAGES, 599 * 24) == 0.0
    assert ve.stage_value(ve.STANDING_STAGES, 2000 * 24) == 0.25
    env = MicroduckVelocityEnv(scene, num_envs=2, noise=False, common_step=2000 * 24)
    for _ in range(50):
        env._sample_head(0)
        assert np.all(np.abs(env.head[0]) <= [1.10, 1.10, 1.40, 0.31])
    acts = small_actions(3, 2, seed=4)
    env.step(acts[0])
    out = env.step(acts[1])
    rate = np.sum(np.square(acts[1].astype(np.float32) - acts[0].astype(np.float32)), axis=1)
    np.testing.assert_allclose(out.terms["action_rate_l2"], -1.0 * rate * 0.02, rtol=1e-5)


def test_tracking_and_upright_terms_match_definitions(scene):
    env = MicroduckVelocityEnv(scene, num_envs=3, noise=False)
    env.freeze_commands(np.array([0.3, -0.1, 0.5]))
    for a in small_actions(15, 3, seed=6, scale=0.1):
        out = env.step(a)
        ok = ~(out.terminated | out.truncated)
        c = out.critic_obs[ok].astype(np.float64)
        lin = c[:, 61:64]
        lin_err = (0.3 - lin[:, 0]) ** 2 + (-0.1 - lin[:, 1]) ** 2 + lin[:, 2] ** 2
        np.testing.assert_allclose(
            out.terms["track_linear_velocity"][ok], 2.0 * 0.02 * np.exp(-lin_err / 0.1), rtol=1e-4
        )
        g = c[:, S["projected_gravity"]]
        upright = 2.0 * 0.02 * np.exp(-(g[:, 0] ** 2 + g[:, 1] ** 2) / 0.05)
        np.testing.assert_allclose(out.terms["upright"][ok], upright, rtol=1e-4)
        np.testing.assert_array_equal(
            out.obs[:, S["twist_cmd"]], np.float32([[0.3, -0.1, 0.5]] * 3)
        )


def test_actor_noise_and_joint_velocity_delay(scene):
    env = MicroduckVelocityEnv(scene, num_envs=2, seed=9)
    prev_critic = env.critic_obs
    for a in small_actions(20, 2, seed=8, scale=0.1):
        out = env.step(a)
        ok = ~(out.terminated | out.truncated)
        diff = out.obs[ok, S["joint_pos_rel"]] - out.critic_obs[ok, S["joint_pos_rel"]]
        assert np.all(np.abs(diff) <= 0.001 + 1e-6)
        # joint_vel is one step old, plus noise of at most 0.25.
        lagged = out.obs[ok, S["joint_vel"]] - prev_critic[ok, S["joint_vel"]]
        assert np.all(np.abs(lagged) <= 0.25 + 1e-5)
        prev_critic = out.critic_obs


def test_fall_terminates_and_time_limit_truncates(scene):
    env = MicroduckVelocityEnv(scene, num_envs=2, seed=1)
    rng = np.random.default_rng(0)
    fell = False
    for _ in range(300):
        if env.step(rng.uniform(-1.5, 1.5, (2, mp.ACT_DIM))).terminated.any():
            fell = True
            break
    assert fell
    short = MicroduckVelocityEnv(scene, num_envs=2, episode_s=0.1, noise=False)
    for _ in range(4):
        assert not short.step(np.zeros((2, mp.ACT_DIM))).truncated.any()
    out = short.step(np.zeros((2, mp.ACT_DIM)))
    assert out.truncated.all() and np.all(short.episode_steps == 0)


def test_bam_actuator_runs(scene):
    env = MicroduckVelocityEnv(scene, num_envs=2, actuator="bam")
    for a in small_actions(5, 2, seed=3, scale=0.1):
        out = env.step(a)
        assert np.isfinite(out.obs).all() and np.isfinite(out.reward).all()
