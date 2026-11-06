"""Export a feed-forward ONNX policy to the runtime's MLP file (WP4-09, ADR-0013).

The Rust runtime runs policies with a small std-only interpreter instead of an
ONNX runtime library. This module turns an ONNX graph that is a single chain of
supported ops into that file, and has a numpy reference of the same interpreter
so parity can be checked against onnxruntime.

Supported chain ops: MatMul (+ Add bias), Gemm, Add/Sub/Mul/Div with a constant
(per-feature or scalar), Relu, Elu, Tanh, Clip, Identity/Flatten/Reshape on a
``[1, n]`` tensor. Anything else is rejected with the op name, never skipped.

File format (little-endian):

```text
magic "BPML" | version u16 | in_dim u32 | n_ops u32 | ops...
op DENSE 1 : rows u32 | cols u32 | W rows*cols f32 (row-major, x @ W) | b cols f32
op SCALE 2 : len u32 | len f32          (x * v)
op SHIFT 3 : len u32 | len f32          (x + v)
op RELU 4  | op TANH 6
op ELU 5   : alpha f32
op CLIP 7  : lo f32 | hi f32
```
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"BPML"
VERSION = 1
DENSE, SCALE, SHIFT, RELU, ELU, TANH, CLIP = 1, 2, 3, 4, 5, 6, 7
OP_NAMES = {DENSE: "dense", SCALE: "scale", SHIFT: "shift", RELU: "relu", ELU: "elu",
            TANH: "tanh", CLIP: "clip"}  # fmt: skip


@dataclass
class Op:
    kind: int
    w: np.ndarray | None = None  # DENSE weight (in, out); SCALE/SHIFT vector
    b: np.ndarray | None = None  # DENSE bias
    params: tuple[float, ...] = field(default_factory=tuple)  # ELU alpha, CLIP lo/hi


@dataclass
class Mlp:
    in_dim: int
    ops: list[Op]

    @property
    def out_dim(self) -> int:
        dim = self.in_dim
        for op in self.ops:
            if op.kind == DENSE and op.w is not None:
                dim = op.w.shape[1]
        return dim

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Reference evaluation in float32, the same arithmetic as the runtime."""
        y = np.asarray(x, dtype=np.float32).reshape(-1)
        for op in self.ops:
            if op.kind == DENSE:
                assert op.w is not None and op.b is not None
                y = (y @ op.w + op.b).astype(np.float32)
            elif op.kind == SCALE:
                y = (y * op.w).astype(np.float32)
            elif op.kind == SHIFT:
                y = (y + op.w).astype(np.float32)
            elif op.kind == RELU:
                y = np.maximum(y, np.float32(0))
            elif op.kind == ELU:
                a = np.float32(op.params[0])
                y = np.where(y > 0, y, a * (np.exp(y) - np.float32(1))).astype(np.float32)
            elif op.kind == TANH:
                y = np.tanh(y)
            elif op.kind == CLIP:
                y = np.clip(y, op.params[0], op.params[1]).astype(np.float32)
        return y

    def to_bytes(self) -> bytes:
        out = [MAGIC, struct.pack("<HII", VERSION, self.in_dim, len(self.ops))]
        for op in self.ops:
            out.append(struct.pack("<B", op.kind))
            if op.kind == DENSE:
                assert op.w is not None and op.b is not None
                out.append(struct.pack("<II", *op.w.shape))
                out.append(op.w.astype("<f4").tobytes() + op.b.astype("<f4").tobytes())
            elif op.kind in (SCALE, SHIFT):
                assert op.w is not None
                out.append(struct.pack("<I", op.w.size) + op.w.astype("<f4").tobytes())
            elif op.kind == ELU:
                out.append(struct.pack("<f", op.params[0]))
            elif op.kind == CLIP:
                out.append(struct.pack("<ff", *op.params))
        return b"".join(out)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_bytes(self.to_bytes())
        return path


def _attr(node: Any, name: str, default: Any) -> Any:
    from onnx import helper

    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


def _vector(c: np.ndarray, dim: int, what: str) -> np.ndarray:
    v = np.asarray(c, dtype=np.float32).reshape(-1)
    if v.size == 1:
        v = np.full(dim, v[0], dtype=np.float32)
    if v.size != dim:
        raise ValueError(f"{what}: constant of size {v.size} for a tensor of size {dim}")
    return v


def from_onnx(path: str | Path) -> Mlp:
    """Read an ONNX feed-forward chain (one float input ``[1, n]``) into an ``Mlp``."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    g = model.graph
    consts = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for node in g.node:
        if node.op_type == "Constant":
            consts[node.output[0]] = numpy_helper.to_array(_attr(node, "value", None))
    inputs = [i for i in g.input if i.name not in consts]
    if len(inputs) != 1 or len(g.output) != 1:
        raise ValueError("expected exactly one graph input and one output")
    shape = [d.dim_value for d in inputs[0].type.tensor_type.shape.dim]
    if len(shape) != 2 or shape[-1] <= 0:
        raise ValueError(f"expected an input of shape [batch, n], got {shape}")
    dim = shape[-1]
    current = inputs[0].name
    ops: list[Op] = []
    in_dim = dim
    for node in g.node:
        if node.op_type == "Constant":
            continue
        ins = list(node.input)
        if current not in ins:
            raise ValueError(f"graph is not a single chain at {node.op_type} ({node.name})")
        others = [consts[i] for i in ins if i != current and i]
        t = node.op_type
        if t == "MatMul":
            if ins[0] != current or len(others) != 1:
                raise ValueError("MatMul must be x @ W with a constant W")
            w = others[0].astype(np.float32)
            ops.append(Op(DENSE, w, np.zeros(w.shape[1], np.float32)))
            dim = w.shape[1]
        elif t == "Gemm":
            if ins[0] != current or _attr(node, "transA", 0):
                raise ValueError("Gemm must be x @ W (+ c) without transA")
            w = others[0].astype(np.float32)
            if _attr(node, "transB", 0):
                w = w.T
            w = w * np.float32(_attr(node, "alpha", 1.0))
            b = others[1] if len(others) > 1 else np.zeros(w.shape[1])
            b = _vector(b, w.shape[1], "Gemm bias") * np.float32(_attr(node, "beta", 1.0))
            ops.append(Op(DENSE, np.ascontiguousarray(w), b))
            dim = w.shape[1]
        elif t in ("Add", "Sub", "Mul", "Div"):
            if len(others) != 1:
                raise ValueError(f"{t} needs one constant operand")
            v = _vector(others[0], dim, t)
            if t == "Sub" and ins[0] != current:
                raise ValueError("Sub with the tensor on the right is not supported")
            if t == "Div" and ins[0] != current:
                raise ValueError("Div with the tensor on the right is not supported")
            last = ops[-1] if ops else None
            if t == "Add" and last is not None and last.kind == DENSE and last.b is not None:
                last.b = (last.b + v).astype(np.float32)  # MatMul + Add bias
            elif t in ("Add", "Sub"):
                ops.append(Op(SHIFT, v if t == "Add" else -v))
            else:
                ops.append(Op(SCALE, v if t == "Mul" else (np.float32(1) / v)))
        elif t == "Relu":
            ops.append(Op(RELU))
        elif t == "Elu":
            ops.append(Op(ELU, params=(float(_attr(node, "alpha", 1.0)),)))
        elif t == "Tanh":
            ops.append(Op(TANH))
        elif t == "Clip":
            lo = float(others[0]) if len(others) > 0 else float(_attr(node, "min", -np.inf))
            hi = float(others[1]) if len(others) > 1 else float(_attr(node, "max", np.inf))
            ops.append(Op(CLIP, params=(lo, hi)))
        elif t in ("Identity", "Flatten", "Reshape"):
            pass  # a [1, n] tensor stays [1, n]
        else:
            raise ValueError(f"unsupported op {t} ({node.name})")
        current = node.output[0]
    if current != g.output[0].name:
        raise ValueError("the chain does not end at the graph output")
    return Mlp(in_dim, ops)
