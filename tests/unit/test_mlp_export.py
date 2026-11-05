"""ONNX -> runtime MLP export: numpy reference vs onnxruntime (WP4-07)."""

from pathlib import Path

import numpy as np
import pytest

from bipedlab import mlp_export as me

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402


def mlp_onnx(path: Path, extra_node=None, seed: int = 0) -> Path:
    """61 -> 64 -> 32 -> 14 actor with input normalisation, as policy exporters write."""
    rng = np.random.default_rng(seed)
    f = np.float32
    inits = {
        "mean": rng.normal(size=61).astype(f),
        "std": (rng.uniform(0.5, 2.0, size=61)).astype(f),
        "W1": (rng.normal(size=(64, 61)) * 0.2).astype(f),  # Gemm transB layout
        "b1": rng.normal(size=64).astype(f),
        "W2": (rng.normal(size=(64, 32)) * 0.2).astype(f),
        "b2": rng.normal(size=32).astype(f),
        "W3": (rng.normal(size=(32, 14)) * 0.2).astype(f),
        "b3": rng.normal(size=14).astype(f),
        "scale": np.array(0.5, dtype=f),
        "lo": np.array(-1.0, dtype=f),
        "hi": np.array(1.0, dtype=f),
    }
    nodes = [
        helper.make_node("Sub", ["obs", "mean"], ["n0"]),
        helper.make_node("Div", ["n0", "std"], ["n1"]),
        helper.make_node("Gemm", ["n1", "W1", "b1"], ["h1"], transB=1),
        helper.make_node("Elu", ["h1"], ["a1"], alpha=1.0),
        helper.make_node("MatMul", ["a1", "W2"], ["m2"]),
        helper.make_node("Add", ["m2", "b2"], ["h2"]),
        helper.make_node("Relu", ["h2"], ["a2"]),
        helper.make_node("MatMul", ["a2", "W3"], ["m3"]),
        helper.make_node("Add", ["m3", "b3"], ["h3"]),
        helper.make_node("Tanh", ["h3"], ["t3"]),
        helper.make_node("Mul", ["t3", "scale"], ["s3"]),
        helper.make_node("Clip", ["s3", "lo", "hi"], ["actions"]),
    ]
    if extra_node is not None:
        nodes[-1] = helper.make_node("Clip", ["s3", "lo", "hi"], ["c3"])
        nodes.append(extra_node)
    graph = helper.make_graph(
        nodes,
        "actor",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])],
        initializer=[numpy_helper.from_array(v, k) for k, v in inits.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def ort_run(path: Path, xs: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.stack([sess.run(None, {"obs": x[None, :]})[0].reshape(-1) for x in xs])


def test_export_matches_onnxruntime(tmp_path):
    path = mlp_onnx(tmp_path / "actor.onnx")
    mlp = me.from_onnx(path)
    assert (mlp.in_dim, mlp.out_dim) == (61, 14)
    names = [me.OP_NAMES[o.kind] for o in mlp.ops]
    assert names == ["shift", "scale", "dense", "elu", "dense", "relu", "dense", "tanh",
                     "scale", "clip"]  # fmt: skip
    xs = np.random.default_rng(1).normal(size=(64, 61)).astype(np.float32) * 3
    ref = ort_run(path, xs)
    got = np.stack([mlp.forward(x) for x in xs])
    np.testing.assert_allclose(got, ref, atol=1e-5, rtol=1e-4)
    assert (np.abs(ref) > 0.05).mean() > 0.5  # the check is not trivially on zeros


def test_file_round_trip_size(tmp_path):
    mlp = me.from_onnx(mlp_onnx(tmp_path / "actor.onnx"))
    data = mlp.to_bytes()
    dense = 4 * (61 * 64 + 64 + 64 * 32 + 32 + 32 * 14 + 14) + 3 * 9
    vectors = 2 * (1 + 4 + 4 * 61) + (1 + 4 + 4 * 14)
    assert len(data) == 4 + 2 + 4 + 4 + dense + vectors + 1 + 4 + 1 + 1 + 1 + 8


def test_unsupported_op_is_rejected(tmp_path):
    extra = helper.make_node("Softmax", ["c3"], ["actions"])
    with pytest.raises(ValueError, match="unsupported op Softmax"):
        me.from_onnx(mlp_onnx(tmp_path / "softmax.onnx", extra_node=extra))
