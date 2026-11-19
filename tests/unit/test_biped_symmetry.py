"""WP4-23: the observation/action mirror matches a physically mirrored biped_v1 state."""

import mujoco
import numpy as np
import pytest

from bipedlab.rl import biped_symmetry as sym
from bipedlab.rl import biped_velocity_env as bve
from bipedlab.rl.biped_velocity_env import BipedVelocityEnv


def _mirror_state(env, rng):
    m = env.model
    a, b = env.datas
    a.qpos[:] = env.stand_qpos
    a.qpos[0:3] += rng.uniform(-0.05, 0.05, 3)
    axis = rng.normal(size=3)
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, axis / np.linalg.norm(axis), 0.3)
    a.qpos[3:7] = quat
    a.qpos[env.qpos_idx] += rng.uniform(-0.2, 0.2, bve.ACT_DIM)
    a.qvel[:] = rng.uniform(-0.5, 0.5, m.nv)
    # Mirror y -> -y: positions/linear velocities flip y, the quaternion flips its x and
    # z parts, angular velocities flip x and z, joints swap sides with yaw/roll negated.
    b.qpos[:] = a.qpos
    b.qpos[1] = -a.qpos[1]
    b.qpos[3:7] = a.qpos[3:7] * [1, -1, 1, -1]
    b.qvel[:] = a.qvel
    b.qvel[1] = -a.qvel[1]
    b.qvel[3:6] = a.qvel[3:6] * [-1, 1, -1]
    qa = a.qpos[env.qpos_idx] - env.default_pose
    b.qpos[env.qpos_idx] = env.default_pose + sym.mirror_actions(qa)
    b.qvel[env.qvel_idx] = sym.mirror_actions(a.qvel[env.qvel_idx])
    for d in (a, b):
        mujoco.mj_forward(m, d)


def test_default_pose_is_mirror_symmetric():
    # yaw/roll defaults are zero and the two legs match, so the mirror keeps the pose
    env = BipedVelocityEnv(num_envs=1, noise=False)
    np.testing.assert_allclose(sym.mirror_actions(env.default_pose), env.default_pose, atol=1e-12)


def test_mirror_matches_a_mirrored_state():
    rng = np.random.default_rng(0)
    env = BipedVelocityEnv(num_envs=2, noise=False)
    env.lag[:] = False
    for _ in range(3):
        _mirror_state(env, rng)
        env.twist[0] = rng.uniform(-0.3, 0.3, 3)
        env.twist[1] = env.twist[0] * [1, -1, -1]
        env.last_action[0] = rng.uniform(-1, 1, bve.ACT_DIM)
        env.last_action[1] = sym.mirror_actions(env.last_action[0])
        env.air_time[0] = rng.uniform(0, 0.3, 2)
        env.air_time[1] = env.air_time[0][::-1]
        env._observe_into(np.arange(2))
        np.testing.assert_allclose(env.obs[1], sym.mirror_obs(env.obs[0]), atol=1e-5)
        np.testing.assert_allclose(
            env.critic_obs[1], sym.mirror_critic(env.critic_obs[0]), atol=1e-5
        )


def test_mirror_is_an_involution():
    x = np.random.default_rng(1).normal(size=(4, bve.CRITIC_DIM))
    np.testing.assert_array_equal(sym.mirror_critic(sym.mirror_critic(x)), x)


def test_augmentation_doubles_the_batch():
    torch = pytest.importorskip("torch")
    TensorDict = pytest.importorskip("tensordict").TensorDict
    obs = TensorDict(
        {"policy": torch.randn(5, bve.OBS_DIM), "critic": torch.randn(5, bve.CRITIC_DIM)},
        batch_size=[5],
    )
    acts = torch.randn(5, bve.ACT_DIM)
    o, a = sym.data_augmentation(obs=obs, actions=acts)
    assert o.batch_size[0] == 10 and a.shape == (10, bve.ACT_DIM)
    torch.testing.assert_close(o["policy"][5:], sym.mirror_obs(obs["policy"]))
    torch.testing.assert_close(a[5:], sym.mirror_actions(acts))


def test_mirrored_state_is_geometrically_mirrored():
    # The joint mirror (swap legs, negate yaw/roll) is an assumption about the model's
    # axes; check it against geometry: every body lands on its partner's mirror image.
    env = BipedVelocityEnv(num_envs=2, noise=False)
    _mirror_state(env, np.random.default_rng(2))
    m = env.model
    a, b = env.datas
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    for i, name in enumerate(names):
        partner = name.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
        j = names.index(partner)
        np.testing.assert_allclose(b.xpos[j], a.xpos[i] * [1, -1, 1], atol=1e-9)


def test_actuators_follow_the_mirrored_joint_order():
    env = BipedVelocityEnv(num_envs=1, noise=False)
    m = env.model
    joints = [m.jnt_qposadr[m.actuator_trnid[k, 0]] for k in range(m.nu)]
    np.testing.assert_array_equal(joints, env.qpos_idx)
