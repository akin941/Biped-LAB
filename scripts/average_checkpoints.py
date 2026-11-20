"""WP4-34: average the weights of neighbouring rsl-rl checkpoints into one checkpoint.

Terrain fine-tunes (MAC-40c..41) swing the flat-floor yaw from one saved checkpoint to
the next by about as much as the acceptance bar, so single checkpoints pass or miss by
chance. Averaging the weights of a few consecutive checkpoints (uniform weight average,
as in stochastic weight averaging) gives one policy between them.

Every floating-point tensor of the actor and critic state dicts is averaged; the
observation normaliser's ``_std`` is then recomputed as ``sqrt(_var)`` so it matches the
averaged ``_var``. Integer tensors (the normaliser's sample count) and ``infos`` come
from the last checkpoint. The optimizer state of the last checkpoint is kept so
``train_ppo.py --resume`` accepts the file; export the averaged policy with
``train_ppo.py --resume AVG.pt --iters 0 --run-dir DIR`` (writes DIR/policy.onnx).

Usage: python scripts/average_checkpoints.py runs/a/model_6950.pt runs/a/model_7000.pt \
    runs/a/model_7050.pt --out runs/avg/model_7000_avg3.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

STATE_KEYS = ("actor_state_dict", "critic_state_dict")


def average(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    """Uniform average of the actor/critic weights; the rest comes from the last one."""
    if not checkpoints:
        raise ValueError("need at least one checkpoint")
    last = checkpoints[-1]
    out = dict(last)
    for key in STATE_KEYS:
        dicts = [c[key] for c in checkpoints]
        names = list(dicts[0])
        if any(list(d) != names for d in dicts):
            raise ValueError(f"{key}: checkpoints have different tensors")
        avg = type(dicts[0])()
        for name in names:
            ts = [d[name] for d in dicts]
            if any(t.shape != ts[0].shape for t in ts):
                raise ValueError(f"{key}.{name}: shapes differ")
            if ts[0].is_floating_point():
                avg[name] = torch.stack(ts).mean(dim=0)
            else:
                avg[name] = ts[-1].clone()
        for name in names:
            if name.endswith("._std"):
                var = name[: -len("_std")] + "_var"
                if var in avg:
                    avg[name] = torch.sqrt(avg[var])
        out[key] = avg
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoints", type=Path, nargs="+", help="model_N.pt files, in order")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    loaded = [torch.load(p, map_location="cpu", weights_only=False) for p in args.checkpoints]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(average(loaded), args.out)
    print(json.dumps({"out": str(args.out), "sources": [str(p) for p in args.checkpoints]}))


if __name__ == "__main__":
    main()
