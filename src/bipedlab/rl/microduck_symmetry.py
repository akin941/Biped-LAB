"""Left/right mirror of Microduck observations and actions (WP3-16).

Same idea as ``biped_symmetry`` (MAC-25 fixed biped_v1's yaw drift with the mirror
loss), but Microduck's joint axes differ: the right leg's axes are the left leg's
mirrored (its STAND2 default pose is the left one negated), so in the trunk frame
every leg joint, including the pitch joints, changes sign when the legs are swapped.
The head joints are their own partners: neck and head pitch keep their sign, head yaw
and roll flip. The 13-D command is twist (vx, vy, wz), head (4, joint space) and body
pose (x, y, z, roll, pitch, yaw). ``tests/unit/test_microduck_symmetry.py`` checks all
of it against a physically mirrored MuJoCo state.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bipedlab.rl.biped_symmetry import mirror
from bipedlab.sim import microduck_policy as mp

_LEFT, _RIGHT = np.arange(0, 5), np.arange(9, 14)
JOINT_PERM = np.arange(mp.ACT_DIM)
JOINT_PERM[_LEFT], JOINT_PERM[_RIGHT] = _RIGHT, _LEFT
JOINT_SIGN = np.full(mp.ACT_DIM, -1.0)
JOINT_SIGN[mp.HEAD_JOINTS] = [1.0, 1.0, -1.0, -1.0]  # neck_pitch, head_pitch, yaw, roll

_SIGNS = {
    "angvel": [-1.0, 1.0, -1.0],
    "vec": [1.0, -1.0, 1.0],
    "twist": [1.0, -1.0, -1.0],
    "head": list(JOINT_SIGN[mp.HEAD_JOINTS]),
    "body": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
}


def _build(blocks: list[tuple[str, int]]) -> tuple[np.ndarray, np.ndarray]:
    perm: list[np.ndarray] = []
    sign: list[np.ndarray] = []
    start = 0
    for kind, width in blocks:
        if kind == "joints":
            p, s = start + JOINT_PERM, JOINT_SIGN
        elif kind == "pair":
            p, s = start + np.array([1, 0]), np.ones(2)
        else:
            p, s = start + np.arange(width), np.array(_SIGNS[kind])
        assert p.size == width == s.size, kind
        perm.append(p)
        sign.append(s)
        start += width
    return np.concatenate(perm), np.concatenate(sign)


_ACTOR = [("angvel", 3), ("vec", 3), ("joints", mp.ACT_DIM), ("joints", mp.ACT_DIM),
          ("joints", mp.ACT_DIM), ("twist", 3), ("head", 4), ("body", 6)]  # fmt: skip
OBS_PERM, OBS_SIGN = _build(_ACTOR)
CRITIC_PERM, CRITIC_SIGN = _build(_ACTOR + [("vec", 3), ("pair", 2), ("pair", 2), ("pair", 2)])
assert OBS_PERM.size == mp.OBS_DIM


def mirror_obs(x: Any) -> Any:
    return mirror(x, OBS_PERM, OBS_SIGN)


def mirror_critic(x: Any) -> Any:
    return mirror(x, CRITIC_PERM, CRITIC_SIGN)


def mirror_actions(x: Any) -> Any:
    return mirror(x, JOINT_PERM, JOINT_SIGN)


def data_augmentation(env: Any = None, obs: Any = None, actions: Any = None) -> tuple[Any, Any]:
    """rsl-rl symmetry hook: return [original; mirrored] observations and actions."""
    import torch
    from tensordict import TensorDict

    out_obs = None
    if obs is not None:
        out_obs = TensorDict(
            {
                "policy": torch.cat([obs["policy"], mirror_obs(obs["policy"])]),
                "critic": torch.cat([obs["critic"], mirror_critic(obs["critic"])]),
            },
            batch_size=[2 * obs.batch_size[0]],
        )
    out_act = None if actions is None else torch.cat([actions, mirror_actions(actions)])
    return out_obs, out_act
