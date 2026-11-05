"""Velocity task for our own biped_v1 (G4.3 preparation). No upstream needed."""

import math

import numpy as np
import pytest

from bipedlab.rl import biped_velocity_env as bve
from bipedlab.rl.biped_velocity_env import BipedVelocityEnv
from bipedlab.rl.parallel_env import ParallelBatchEnv

S = bve.OBS_SLICES


def actions(steps, n, seed=0, scale=0.2):
    return np.random.default_rng(seed).uniform(-scale, scale, (steps, n, bve.ACT_DIM))


def test_contract_shapes_and_clean_actor_equals_critic_prefix():
    env = BipedVelocityEnv(num_envs=3, noise=False)
    assert bve.OBS_DIM == 39 and bve.CRITIC_DIM == 48 and bve.ACT_DIM == 10
    assert env.obs.shape == (3, 39) and env.critic_obs.shape == (3, 48)
    assert env.max_episode_steps == 1000
    assert S["command"] == slice(36, 39)
    for a in actions(10, 3):
        out = env.step(a)
        assert out.obs.dtype == np.float32 and out.critic_obs.dtype == np.float32
        np.testing.assert_array_equal(out.obs, out.critic_obs[:, :39])
        np.testing.assert_allclose(out.reward, sum(out.terms.values()), rtol=1e-5, atol=1e-7)
        assert out.joint_torque.shape == (3, 10) and np.all(out.joint_torque >= 0)
        np.testing.assert_array_equal(out.obs[:, S["last_action"]], a.astype(np.float32))


def test_reset_from_stand_keyframe_with_random_yaw():
    env = BipedVelocityEnv(num_envs=16, seed=3)
    for d in env.datas:
        np.testing.assert_allclose(
            d.qpos[env.qpos_idx], env.default_pose, atol=bve.RESET_JOINT_NOISE + 1e-12
        )
        assert env.stand_height <= d.qpos[2] <= env.stand_height + bve.RESET_Z_LIFT[1]
        assert abs(d.qpos[4]) < 1e-12 and abs(d.qpos[5]) < 1e-12  # yaw only
    yaws = [2 * math.atan2(d.qpos[6], d.qpos[3]) for d in env.datas]
    assert np.ptp(yaws) > 1.0
    # Joint order is the documented one, knees bent in the stand pose.
    assert env.default_pose[bve.KNEES].min() > 0.3


def test_zero_action_stands_without_falling():
    env = BipedVelocityEnv(num_envs=4, seed=1, noise=False)
    env.freeze_commands(np.zeros(3))
    for _ in range(150):  # 3 s
        out = env.step(np.zeros((4, bve.ACT_DIM)))
        assert not out.terminated.any()
    gravity = out.obs[:, S["projected_gravity"]]
    assert np.all(gravity[:, 2] < -math.cos(math.radians(10)))
    # Standing still: tracking a zero command is near its maximum.
    assert np.all(out.terms["track_linear_velocity"] > 0.8 * 2.0 * bve.STEP_DT)


def test_large_actions_fall_and_auto_reset():
    env = BipedVelocityEnv(num_envs=2, seed=0)
    rng = np.random.default_rng(0)
    fell = False
    for _ in range(300):
        out = env.step(rng.uniform(-3, 3, (2, bve.ACT_DIM)))
        if out.terminated.any():
            fell = True
            i = int(np.flatnonzero(out.terminated)[0])
            assert env.episode_steps[i] == 0
            assert not np.array_equal(out.final_obs[i], out.obs[i])
            break
    assert fell


def test_same_seed_deterministic_and_shards_match():
    acts = actions(25, 4, seed=1)
    full = BipedVelocityEnv(num_envs=4, seed=7)
    again = BipedVelocityEnv(num_envs=4, seed=7)
    a = BipedVelocityEnv(num_envs=2, seed=7, first_env=0)
    b = BipedVelocityEnv(num_envs=2, seed=7, first_env=2)
    for act in acts:
        o1, o2 = full.step(act), again.step(act)
        oa, ob = a.step(act[:2]), b.step(act[2:])
        np.testing.assert_array_equal(o1.obs, o2.obs)
        np.testing.assert_array_equal(o1.obs, np.concatenate([oa.obs, ob.obs]))
        np.testing.assert_array_equal(o1.reward, np.concatenate([oa.reward, ob.reward]))


def test_commands_resample_and_freeze():
    env = BipedVelocityEnv(num_envs=8, seed=2)
    first = env.twist.copy()
    assert np.all(np.abs(first) <= bve.TWIST_RANGE + 1e-12)
    for _ in range(round(8.5 / bve.STEP_DT)):
        env.step(np.zeros((8, bve.ACT_DIM)))
    assert not np.array_equal(first, env.twist)
    env.freeze_commands(np.array([0.2, 0.0, 0.0]))
    np.testing.assert_allclose(
        env.obs[:, S["command"]], np.tile([0.2, 0.0, 0.0], (8, 1)), atol=1e-7
    )


def test_tracking_reward_prefers_matching_velocity():
    env = BipedVelocityEnv(num_envs=2, seed=0, noise=False)
    env.freeze_commands(np.array([0.2, 0.0, 0.0]))
    env.datas[0].qvel[0:2] = 0.0
    env.datas[1].qvel[0:2] = 0.0
    # Rotate env 0's base velocity into its yaw so that it moves forward at 0.2 m/s.
    q = env.datas[0].qpos[3:7]
    yaw = 2 * math.atan2(q[3], q[0])
    env.datas[0].qvel[0:2] = 0.2 * np.array([math.cos(yaw), math.sin(yaw)])
    s = env._state()
    terms = env._rewards(s, np.zeros((2, bve.ACT_DIM), dtype=np.float32))
    assert terms["track_linear_velocity"][0] > terms["track_linear_velocity"][1]


def test_domain_randomization_changes_physics_but_not_commands():
    plain = BipedVelocityEnv(num_envs=3, seed=4)
    dr = BipedVelocityEnv(num_envs=3, seed=4, dr=True)
    np.testing.assert_array_equal(plain.twist, dr.twist)
    assert np.all((dr.foot_friction >= 0.7) & (dr.foot_friction <= 1.3))
    assert np.ptp(dr.mass_scale) > 0
    for a in actions(20, 3, seed=2, scale=0.1):
        out = dr.step(a)
        assert np.all(np.isfinite(out.obs))
    # The shared model holds the last stepped env's parameters.
    m = dr.model
    assert m.geom_friction[dr.foot_geoms[0], 0] == pytest.approx(dr.foot_friction[2])


def test_rejects_bad_actions_and_models(tmp_path):
    env = BipedVelocityEnv(num_envs=1)
    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros((1, 14)))
    with pytest.raises(ValueError, match="finite"):
        env.step(np.full((1, 10), np.nan))
    bad = tmp_path / "bad.xml"
    bad.write_text('<mujoco><option timestep="0.002"/><worldbody/></mujoco>')
    with pytest.raises(ValueError, match="timestep"):
        BipedVelocityEnv(scene=bad)


def test_parallel_env_matches_serial():
    acts = actions(10, 4, seed=5)
    serial = BipedVelocityEnv(num_envs=4, seed=9)
    with ParallelBatchEnv(
        bve.DEFAULT_SCENE, 4, 2, episode_s=bve.EPISODE_S, seed=9, env_cls=BipedVelocityEnv
    ) as par:
        assert par.act_dim == bve.ACT_DIM
        np.testing.assert_array_equal(par.obs, serial.obs)
        for a in acts:
            o1, o2 = serial.step(a), par.step(a)
            np.testing.assert_array_equal(o1.obs, o2.obs)
            np.testing.assert_array_equal(o1.joint_torque, o2.joint_torque)
