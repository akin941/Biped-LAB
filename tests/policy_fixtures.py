"""Build tiny ONNX policies in memory so tests never need downloaded weights."""

from pathlib import Path

import numpy as np


def linear_policy(path: Path, weight: np.ndarray, bias: np.ndarray) -> Path:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    in_dim, out_dim = weight.shape
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["obs", "W"], ["xw"]),
            helper.make_node("Add", ["xw", "b"], ["actions"]),
        ],
        "linear_policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, in_dim])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, out_dim])],
        initializer=[
            numpy_helper.from_array(weight.astype(np.float32), "W"),
            numpy_helper.from_array(bias.astype(np.float32), "b"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path
