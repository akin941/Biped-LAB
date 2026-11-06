"""WP4-09: export an ONNX policy to the runtime MLP file and check parity.

Compares onnxruntime, the numpy reference of the interpreter, and (with
``--rust-bin``) the Rust ``mlp-eval`` binary on random observations.

Usage:
  uv run python scripts/export_mlp.py POLICY.onnx|NAME --out policy.bpml
      [--rust-bin runtime/target/release/mlp-eval] [--samples 256] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from bipedlab import mlp_export, policies

ATOL, RTOL = 1e-5, 1e-4


def onnx_outputs(path: Path, xs: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return np.stack([sess.run(None, {name: x[None, :]})[0].reshape(-1) for x in xs])


def rust_outputs(binary: Path, model: Path, xs: np.ndarray, out_dim: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "in.f32", Path(tmp) / "out.f32"
        inp.write_bytes(xs.astype("<f4").tobytes())
        subprocess.run([str(binary), str(model), str(inp), str(out)], check=True)
        return np.frombuffer(out.read_bytes(), dtype="<f4").reshape(len(xs), out_dim)


def check(ref: np.ndarray, got: np.ndarray) -> dict[str, float | bool]:
    err = np.abs(ref - got)
    return {
        "max_abs_err": float(err.max()),
        "ok": bool(np.all(err <= ATOL + RTOL * np.abs(ref))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("policy", help="ONNX file or a name from the policy lockfile")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rust-bin", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = Path(args.policy)
    if not path.exists():
        path = policies.verified_path(args.policy)
    mlp = mlp_export.from_onnx(path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mlp.save(args.out)
    xs = np.random.default_rng(args.seed).normal(size=(args.samples, mlp.in_dim))
    xs = xs.astype(np.float32)
    ref = onnx_outputs(path, xs)
    report: dict[str, object] = {
        "onnx": str(path),
        "onnx_sha256": policies.sha256(path),
        "mlp": str(args.out),
        "mlp_sha256": policies.sha256(args.out),
        "in_dim": mlp.in_dim,
        "out_dim": mlp.out_dim,
        "ops": [mlp_export.OP_NAMES[op.kind] for op in mlp.ops],
        "numpy_vs_onnxruntime": check(ref, np.stack([mlp.forward(x) for x in xs])),
    }
    if args.rust_bin:
        got = rust_outputs(args.rust_bin, args.out, xs, mlp.out_dim)
        report["rust_vs_onnxruntime"] = check(ref, got)
    print(json.dumps(report, indent=2))
    checks = [v for k, v in report.items() if k.endswith("_vs_onnxruntime")]
    if not all(c["ok"] for c in checks if isinstance(c, dict)):
        sys.exit(1)


if __name__ == "__main__":
    main()
