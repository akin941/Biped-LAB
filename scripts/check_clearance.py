"""Can the biped be built? Servo-box clearance check (WP5-13).

Loads a committed biped model and measures the distance between every pair of servo boxes
(and each servo against the feet) in the stand pose and with each joint, one at a time, at
both ends of its range. A pair on the same leg closer than ``--min-gap`` (default 3 mm, a
bracket wall) cannot be built and fails the check. Pairs on different legs that touch are
listed as ``legs_touch``: the robot can drive its legs into each other there, which the
model must make the policy avoid (``leg_collisions``), not a build error. Hip-yaw servos
inside the trunk box are expected and not checked against it.

Usage: uv run python scripts/check_clearance.py [--model biped_ds3218b] [--min-gap 0.003]
       [--out report.json]
Exit 1 when a pair is too close.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from bipedlab import REPO_ROOT

SCENES = REPO_ROOT / "simulation/custom-biped"


def box_gap(c1: np.ndarray, r1: np.ndarray, h1: np.ndarray,
            c2: np.ndarray, r2: np.ndarray, h2: np.ndarray) -> float:  # fmt: skip
    """Separating-axis gap between two boxes (centre, rotation, half sizes).

    Positive: the largest separation along the 15 SAT axes, a lower bound on the true
    distance (exact for face-to-face). Negative: the smallest overlap, the penetration
    depth. Used instead of ``mj_geomDistance``, which returns 0 for some separated boxes.
    """
    axes = [r1[:, i] for i in range(3)] + [r2[:, i] for i in range(3)]
    axes += [np.cross(r1[:, i], r2[:, j]) for i in range(3) for j in range(3)]
    best = -np.inf
    for ax in axes:
        n = np.linalg.norm(ax)
        if n < 1e-9:
            continue
        ax = ax / n
        e1 = float(np.sum(h1 * np.abs(r1.T @ ax)))
        e2 = float(np.sum(h2 * np.abs(r2.T @ ax)))
        best = max(best, abs(float((c2 - c1) @ ax)) - e1 - e2)
    return best


def _poses(model: mujoco.MjModel) -> list[tuple[str, np.ndarray]]:
    stand = model.key_qpos[0].copy()
    poses = [("stand", stand)]
    for j in range(1, model.njnt):  # skip the free root
        lo, hi = model.jnt_range[j]
        for tag, value in (("min", lo), ("max", hi)):
            q = stand.copy()
            q[model.jnt_qposadr[j]] = value
            poses.append((f"{model.joint(j).name}={tag}", q))
    return poses


def check(model_name: str, min_gap: float = 0.003) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(SCENES / f"{model_name}.xml"))
    data = mujoco.MjData(model)
    names = [model.geom(i).name for i in range(model.ngeom)]
    servos = [i for i, n in enumerate(names) if n.endswith("_servo")]
    feet = [i for i, n in enumerate(names) if n.endswith("_foot")]
    pairs = list(itertools.combinations(servos, 2)) + [(s, f) for s in servos for f in feet]
    pairs = [(a, b) for a, b in pairs if model.geom_bodyid[a] != model.geom_bodyid[b]]
    worst: dict[tuple[int, int], tuple[float, str]] = {}
    for tag, q in _poses(model):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        for a, b in pairs:
            dist = box_gap(
                data.geom_xpos[a], data.geom_xmat[a].reshape(3, 3), model.geom_size[a],
                data.geom_xpos[b], data.geom_xmat[b].reshape(3, 3), model.geom_size[b],
            )  # fmt: skip
            if (a, b) not in worst or dist < worst[(a, b)][0]:
                worst[(a, b)] = (dist, tag)
    rows = sorted(
        (
            {"a": names[a], "b": names[b], "gap_mm": round(d * 1000, 1), "pose": tag}
            for (a, b), (d, tag) in worst.items()
        ),
        key=lambda r: r["gap_mm"],
    )

    def same_leg(r: dict[str, Any]) -> bool:
        return bool(r["a"].split("_")[0] == r["b"].split("_")[0])

    close = [r for r in rows if r["gap_mm"] < min_gap * 1000]
    fails = [r for r in close if same_leg(r)]
    return {
        "model": model_name,
        "min_gap_mm": min_gap * 1000,
        "ok": not fails,
        "failures": fails,
        "legs_touch": [r for r in close if not same_leg(r)],
        "closest_same_leg": [r for r in rows if same_leg(r)][:4],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="biped_ds3218b")
    parser.add_argument("--min-gap", type=float, default=0.003, help="metres")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = check(args.model, args.min_gap)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
