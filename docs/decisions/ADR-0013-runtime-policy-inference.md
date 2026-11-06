# ADR-0013: Policy inference in the runtime with a std-only MLP interpreter

Status: accepted (2026-11-05). Amends the "policy inference via `ort`" line of
ADR-0007; the rest of ADR-0007 stands.

## Context
ADR-0007 planned ONNX inference in the Rust runtime with `ort`. `ort` links the
ONNX Runtime C++ library, which its build downloads as a prebuilt binary by
default; that is a large, opaque dependency on the robot's control path, and the
download host is outside our pinned sources. Our policies (Microduck's, and the
ones we will train with rsl-rl) are small feed-forward MLPs: input normalisation,
a few dense layers, ELU/ReLU/Tanh, clipping.

## Decision
- Python (`bipedlab.mlp_export`) reads the ONNX file, accepts only a single chain
  of supported ops, and writes a flat little-endian MLP file. Unsupported ops
  are rejected, never skipped.
- The runtime (`runtime/src/mlp.rs`) interprets that file with std only, with
  bounds on every size and finite-value checks on every parameter.
- Parity is a test: the numpy reference and the Rust interpreter must match
  onnxruntime (atol 1e-5, rtol 1e-4) on a fixture actor in every CI run, and on
  the official Microduck policies in the CI job that can fetch them.
- The exported file's sha256 is recorded next to the source ONNX sha256.

## Consequences
- No new Rust dependency; the robot never loads an ONNX runtime.
- A policy with recurrent state or other ops needs this ADR revisited (then
  `tract` or `ort` would be evaluated with their own supply-chain review).
