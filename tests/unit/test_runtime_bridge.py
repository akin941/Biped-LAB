"""Sim bridge protocol and lockstep loop, with a Python stand-in for the runtime."""

import socket
import struct
import threading

import numpy as np

from bipedlab import runtime_bridge as rb


def test_frame_sizes_match_the_rust_protocol():
    # Sizes asserted in runtime/src/protocol.rs tests for n = 2.
    hello = rb.encode_hello(0.02, [-1, -0.1], [1, 2.2], [0, 0.6], [5, 5], [0.52, 0.52])
    assert len(hello) == rb.hello_size(2) == 4 + 2 + 2 + 8 + 2 * 40
    assert rb.action_size(2) == 1 + 8 + 1 + 4 + 16
    state = rb.encode_state(7, 0.14, [1, 0, 0, 0], [0.1, -0.2, 0.3], [0.5, 0.6], [-1, 1])
    assert len(state) == rb.state_size(2) == 1 + 8 + 8 * (1 + 4 + 3 + 2 + 4) + 1


def test_decode_action():
    buf = struct.pack("<BQBI2d", rb.TAG_ACTION, 7, 1, 5, 0.1, 0.2)
    step, enable, bits, targets = rb.decode_action(buf, 2)
    assert (step, enable, bits) == (7, True, 5)
    np.testing.assert_allclose(targets, [0.1, 0.2])


def test_tilt_from_quat():
    h = 2**-0.5
    assert rb.tilt_from_quat([1, 0, 0, 0]) == 0.0
    assert abs(rb.tilt_from_quat([h, h, 0, 0]) - np.pi / 2) < 1e-9
    assert abs(rb.tilt_from_quat([h, 0, 0, h])) < 1e-6


def _python_runtime(conn: socket.socket, n: int, torque_off_after: int | None) -> None:
    """Holds the stand pose from the hello; optionally sends torque off."""
    hello = rb._recv_exact(conn, rb.hello_size(n))
    default = np.array(struct.unpack_from(f"<{rb.JOINT_FIELDS * n}d", hello, 16))[2::5]
    state_size = rb.state_size(n)
    while True:
        try:
            frame = rb._recv_exact(conn, state_size)
        except ConnectionError:
            return
        step = struct.unpack_from("<Q", frame, 1)[0]
        enable = torque_off_after is None or step <= torque_off_after
        conn.sendall(struct.pack(f"<BQBI{n}d", rb.TAG_ACTION, step, int(enable), 0, *default))


def _run(torque_off_after: int | None = None, **cfg: float) -> dict:
    bridge = rb.SimBridge(cfg=rb.BridgeConfig(seconds=2.0, **cfg))
    a, b = socket.socketpair()
    t = threading.Thread(target=_python_runtime, args=(b, bridge.n, torque_off_after))
    t.start()
    with a:
        out = bridge.serve(a)
    t.join(timeout=10)
    b.close()
    return out


def test_bridge_stands_with_stand_pose_client():
    out = _run()
    assert out["steps"] == 100 and out["torque_off_step"] is None
    assert out["max_tilt_deg"] < 2.0 and out["trunk_height_m"] > 0.19


def test_torque_off_makes_the_robot_sag():
    out = _run(torque_off_after=10)
    assert out["torque_off_step"] == 11
    assert out["trunk_height_m"] < 0.19  # limp joints: the robot sags or falls


def test_projected_gravity_matches_the_env_rotation():
    from bipedlab.rl.velocity_env import rotate_inverse

    q = np.random.default_rng(0).normal(size=(20, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    want = rotate_inverse(q, np.broadcast_to([0.0, 0.0, -1.0], (20, 3)))
    got = np.array([rb.projected_gravity(x) for x in q])
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_policy_observation_matches_the_env_actor_terms():
    """The runtime-side observation (WP4-10) equals the clean actor terms of training.

    The critic observation starts with the actor terms without noise, lag or DR; the
    bridge builds its observation from what it sends the runtime: the IMU sensors,
    joint positions and velocities, and the stand pose from the keyframe ctrl.
    """
    from bipedlab.rl.biped_velocity_env import OBS_DIM, BipedVelocityEnv

    env = BipedVelocityEnv(num_envs=1, seed=1, noise=False)
    bridge = rb.SimBridge(model=env.model)
    np.testing.assert_allclose(bridge.default, env.default_pose, atol=1e-12)
    m, d = env.model, env.datas[0]
    quat_adr = m.sensor_adr[m.sensor("orientation").id]
    gyro_adr = m.sensor_adr[m.sensor("gyro").id]
    rng = np.random.default_rng(1)
    for _ in range(5):
        r = env.step(rng.uniform(-0.3, 0.3, size=(1, 10)).astype(np.float32))
        assert not (r.terminated[0] or r.truncated[0])
        obs = rb.policy_observation(
            d.sensordata[gyro_adr : gyro_adr + 3],
            d.sensordata[quat_adr : quat_adr + 4],
            d.qpos[bridge.qadr],
            d.qvel[bridge.vadr],
            bridge.default,
            env.last_action[0],
            env.twist[0],
        )
        np.testing.assert_allclose(obs, r.critic_obs[0, :OBS_DIM], atol=1e-6)


def test_body_velocity_is_in_the_trunk_frame():
    # WP4-20: a trunk yawed 90 deg moving along world +y moves forward in its own frame.
    import math

    import mujoco

    m = mujoco.MjModel.from_xml_string(rb.biped.to_mjcf())
    d = mujoco.MjData(m)
    d.qpos[3:7] = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
    d.qvel[0:3] = [0.0, 0.2, 0.0]
    d.qvel[5] = 0.3
    np.testing.assert_allclose(rb.body_velocity(d), [0.2, 0.0, 0.3], atol=1e-12)


def test_servo_model_override_and_curve_still_stand():
    # WP4-30: the STS3215 limit and curve reach the hello and the physics.
    bridge = rb.SimBridge(cfg=rb.BridgeConfig(torque_limit=1.91, motor_no_load_speed=5.45))
    assert np.allclose(bridge.torque_limit, 1.91)
    out = _run(torque_limit=1.91, motor_no_load_speed=5.45)
    assert out["steps"] == 100 and out["max_tilt_deg"] < 2.0


def test_motor_force_range_drives_less_along_the_motion():
    from bipedlab.design import actuators

    lo, hi = actuators.motor_force_range(np.ones(3), np.array([-5.0, 0.0, 2.5]), 5.0)
    assert np.allclose(hi, [1.0, 1.0, 0.5]) and np.allclose(lo, [0.0, -1.0, -1.0])


def test_init_seed_perturbs_the_start_and_still_stands():
    # WP4-36: seeded starts differ from each other and from the keyframe, and stand.
    base = _run()
    a, b, a2 = _run(init_seed=1), _run(init_seed=2), _run(init_seed=1)
    assert a["trunk_height_m"] == a2["trunk_height_m"]
    assert a["trunk_height_m"] != b["trunk_height_m"] != base["trunk_height_m"]
    for out in (a, b):
        assert out["steps"] == 100 and not out["fell"] and out["max_tilt_deg"] < 5.0
