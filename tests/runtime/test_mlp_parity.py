"""WP4-09: the Rust MLP interpreter matches onnxruntime on an exported actor."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bipedlab import runtime_bridge as rb
from tests.unit.test_mlp_export import mlp_onnx, ort_run

pytestmark = pytest.mark.runtime

BINARY = Path(os.environ.get("BIPEDLAB_MLP_EVAL_BIN", rb.DEFAULT_BINARY.with_name("mlp-eval")))
SCRIPT = Path(__file__).resolve().parents[2] / "scripts/export_mlp.py"


@pytest.fixture(scope="module", autouse=True)
def _binary():
    if not BINARY.exists():
        if os.environ.get("BIPEDLAB_REQUIRE_RUNTIME") == "1":
            pytest.fail(f"mlp-eval binary missing: {BINARY}")
        pytest.skip(f"mlp-eval not built: {BINARY}")


def test_rust_matches_onnxruntime(tmp_path):
    from bipedlab import mlp_export as me

    path = mlp_onnx(tmp_path / "actor.onnx", seed=3)
    model = me.from_onnx(path).save(tmp_path / "actor.bpml")
    xs = np.random.default_rng(2).normal(size=(256, 61)).astype(np.float32) * 3
    inp, out = tmp_path / "in.f32", tmp_path / "out.f32"
    inp.write_bytes(xs.astype("<f4").tobytes())
    subprocess.run([str(BINARY), str(model), str(inp), str(out)], check=True)
    got = np.frombuffer(out.read_bytes(), dtype="<f4").reshape(256, 14)
    np.testing.assert_allclose(got, ort_run(path, xs), atol=1e-5, rtol=1e-4)


def test_export_script_reports_parity(tmp_path):
    path = mlp_onnx(tmp_path / "actor.onnx", seed=4)
    r = subprocess.run(
        [sys.executable, str(SCRIPT), str(path), "--out", str(tmp_path / "a.bpml"),
         "--rust-bin", str(BINARY)],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert r.returncode == 0, r.stdout + r.stderr
    assert '"rust_vs_onnxruntime"' in r.stdout


def test_rust_rejects_a_corrupt_file(tmp_path):
    bad = tmp_path / "bad.bpml"
    bad.write_bytes(b"BPML\x01\x00" + b"\xff" * 8)
    inp = tmp_path / "in.f32"
    inp.write_bytes(b"")
    r = subprocess.run([str(BINARY), str(bad), str(inp), str(tmp_path / "o")],
                       capture_output=True, text=True, check=False)  # fmt: skip
    assert r.returncode == 2 and "out of range" in r.stderr
