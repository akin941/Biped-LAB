"""WP3-16: the Microduck mirror matches a physically mirrored MuJoCo state."""

import mujoco
import numpy as np
import pytest

from bipedlab import upstream
from bipedlab.rl import microduck_symmetry as sym
from bipedlab.rl import velocity_env as ve
from bipedlab.sim import microduck_policy as mp

LEFT = ["yaw2roll", "hip_l", "upper_leg_left", "leg", "ankle_left"]
RIGHT = ["bearing_roll", "hip_l_2", "upper_leg_right", "leg_2", "ankle_right"]
# Head bodies on the sagittal plane (the neck frames sit 14.5 mm off it, on the pitch
# axis, so their origins are not mirror images even though the joints are).
CENTRE = ["trunk_base", "yaw_roll_motion", "jaw_soft"]


@pytest.fixture(scope="module")
def env():
    scene = upstream.microduck_scene("scene_walk.xml")
    if not scene.exists():
        pytest.skip("upstream Microduck scene not available")
    return ve.MicroduckVelocityEnv(scene, num_envs=2, noise=False)


def _mirror_state(env, rng):
    m = env.model
    a, b = env.datas
    a.qpos[:] = 0.0
    a.qpos[2] = 0.3
    axis = rng.normal(size=3)
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, axis / np.linalg.norm(axis), 0.3)
    a.qpos[3:7] = quat
    a.qpos[0:2] = rng.uniform(-0.05, 0.05, 2)
    a.qpos[env.qpos_idx] = mp.DEFAULT_POSE + rng.uniform(-0.3, 0.3, mp.ACT_DIM)
    a.qvel[:] = rng.uniform(-0.5, 0.5, m.nv)
    # Mirror y -> -y: positions and linear velocities flip y, the quaternion flips its
    # x and z parts, angular velocities flip x and z; joints use the module's map.
    b.qpos[:] = a.qpos
    b.qpos[1] = -a.qpos[1]
    b.qpos[3:7] = a.qpos[3:7] * [1, -1, 1, -1]
    b.qvel[:] = a.qvel
    b.qvel[1] = -a.qvel[1]
    b.qvel[3:6] = a.qvel[3:6] * [-1, 1, -1]
    b.qpos[env.qpos_idx] = sym.mirror_actions(a.qpos[env.qpos_idx])
    b.qvel[env.qvel_idx] = sym.mirror_actions(a.qvel[env.qvel_idx])
    for d in (a, b):
        mujoco.mj_forward(m, d)


def test_default_pose_is_mirror_symmetric():
    np.testing.assert_allclose(sym.mirror_actions(mp.DEFAULT_POSE), mp.DEFAULT_POSE, atol=1e-7)


def test_mirrored_state_is_geometrically_mirrored(env):
    _mirror_state(env, np.random.default_rng(0))
    m = env.model
    a, b = env.datas

    def pos(d, name):
        return d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)]

    for left, right in zip(LEFT + RIGHT + CENTRE, RIGHT + LEFT + CENTRE, strict=True):
        np.testing.assert_allclose(pos(b, right), pos(a, left) * [1, -1, 1], atol=1e-6)
    # Frame origins sit on the joint axes, so a wrong head-roll sign would not move
    # them; points 3 cm off the jaw frame along each axis catch it.
    jaw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")

    def point(d, local):
        return d.xpos[jaw] + d.xmat[jaw].reshape(3, 3) @ local

    for local in np.eye(3) * 0.03:
        np.testing.assert_allclose(
            point(b, local * [1, -1, 1]), point(a, local) * [1, -1, 1], atol=1e-6
        )


def test_mirror_matches_a_mirrored_state(env):
    rng = np.random.default_rng(1)
    env.lag[:] = False
    for _ in range(3):
        _mirror_state(env, rng)
        env.twist[0] = rng.uniform(-0.3, 0.3, 3)
        env.head[0] = rng.uniform(-0.2, 0.2, 4)
        env.body[0] = rng.uniform(-0.05, 0.05, 6)
        cmd = sym.mirror(np.concatenate([env.twist[0], env.head[0], env.body[0]]),
                         sym.OBS_PERM[48:] - 48, sym.OBS_SIGN[48:])  # fmt: skip
        env.twist[1], env.head[1], env.body[1] = cmd[:3], cmd[3:7], cmd[7:]
        env.last_action[0] = rng.uniform(-1, 1, mp.ACT_DIM)
        env.last_action[1] = sym.mirror_actions(env.last_action[0])
        env.air_time[0] = rng.uniform(0, 0.3, 2)
        env.air_time[1] = env.air_time[0][::-1]
        env._observe_into(np.arange(2))
        obs, critic = env._obs, env._critic
        np.testing.assert_allclose(obs[1], sym.mirror_obs(obs[0]), atol=1e-5)
        np.testing.assert_allclose(critic[1], sym.mirror_critic(critic[0]), atol=1e-5)


def test_maps_are_involutions_of_the_right_size():
    assert sym.CRITIC_PERM.size == ve.CRITIC_DIM
    x = np.random.default_rng(2).normal(size=(4, ve.CRITIC_DIM))
    np.testing.assert_array_equal(sym.mirror_critic(sym.mirror_critic(x)), x)
