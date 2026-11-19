"""Left/right mirror of biped_v1 observations and actions (WP4-23).

biped_v1 is mirror-symmetric about its sagittal (x-z) plane, but a policy can still
learn an asymmetric gait: MAC-14 yaws -0.13 to -0.24 rad/s while walking straight
(MAC-18). rsl-rl's symmetry extension trains on mirrored copies of every sample
(data augmentation) and/or penalizes ``pi(mirror(o)) != mirror(pi(o))`` (mirror loss).

The mirror y -> -y maps
- body angular rates (gyro) (wx, wy, wz) -> (-wx, wy, -wz)
- projected gravity and body linear velocity (x, y, z) -> (x, -y, z)
- the command (vx, vy, wz) -> (vx, -vy, -wz)
- joints: swap left and right; hip yaw and hip roll change sign (axes z and x), while
  hip pitch, knee and ankle pitch (axis y) keep it
- per-foot critic terms (foot height, air time, contact): swap left and right.

Each mirror is a fixed permutation plus sign flip, ``out = x[:, PERM] * SIGN``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bipedlab.design import biped
from bipedlab.rl import biped_velocity_env as bve

_N = len(biped.LEG_JOINTS)  # joints per leg
_JOINT_SIGN = np.array([-1.0 if j in ("hip_yaw", "hip_roll") else 1.0 for j in biped.LEG_JOINTS])
JOINT_PERM = np.concatenate([np.arange(_N, 2 * _N), np.arange(_N)])
JOINT_SIGN = np.tile(_JOINT_SIGN, 2)


def _build(blocks: list[tuple[str, int]]) -> tuple[np.ndarray, np.ndarray]:
    perm: list[np.ndarray] = []
    sign: list[np.ndarray] = []
    start = 0
    for kind, width in blocks:
        idx = np.arange(start, start + width)
        if kind == "angvel":
            p, s = idx, np.array([-1.0, 1.0, -1.0])
        elif kind == "vec":
            p, s = idx, np.array([1.0, -1.0, 1.0])
        elif kind == "command":
            p, s = idx, np.array([1.0, -1.0, -1.0])
        elif kind == "joints":
            p, s = start + JOINT_PERM, JOINT_SIGN
        elif kind == "pair":
            p, s = start + np.array([1, 0]), np.ones(2)
        else:  # pragma: no cover - programming error
            raise ValueError(kind)
        perm.append(p)
        sign.append(s)
        start += width
    return np.concatenate(perm), np.concatenate(sign)


_ACTOR = [("angvel", 3), ("vec", 3), ("joints", bve.ACT_DIM), ("joints", bve.ACT_DIM),
          ("joints", bve.ACT_DIM), ("command", 3)]  # fmt: skip
OBS_PERM, OBS_SIGN = _build(_ACTOR)
CRITIC_PERM, CRITIC_SIGN = _build(_ACTOR + [("vec", 3), ("pair", 2), ("pair", 2), ("pair", 2)])
assert OBS_PERM.size == bve.OBS_DIM and CRITIC_PERM.size == bve.CRITIC_DIM


def mirror(x: Any, perm: np.ndarray, sign: np.ndarray) -> Any:
    """Mirror the last axis of ``x`` (numpy array or torch tensor)."""
    if isinstance(x, np.ndarray):
        return x[..., perm] * sign.astype(x.dtype)
    import torch

    p = torch.as_tensor(perm, device=x.device)
    s = torch.as_tensor(sign, dtype=x.dtype, device=x.device)
    return x[..., p] * s


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
