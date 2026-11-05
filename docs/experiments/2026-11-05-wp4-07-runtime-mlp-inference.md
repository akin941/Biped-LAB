# WP4-07: policy inference in the runtime (std-only MLP, ADR-0013)

**Environment:** cloud (container) for the fixture results below. The official
Microduck policies cannot be downloaded in the cloud (huggingface.co blocked), so
their export and Rust parity run in the CI `pretrained-policy` job (gha-linux).
Stacked on WP4-04/06.

## Change
- `bipedlab.mlp_export`: ONNX single-chain → MLP file (dense, scale, shift, relu,
  elu, tanh, clip); numpy reference interpreter; unsupported ops rejected.
- `runtime/src/mlp.rs`: std-only interpreter with bounded sizes, finite
  parameters, size-consistent op chain, no trailing bytes.
- `runtime/src/bin/mlp-eval.rs`: batch evaluation for parity tests.
- `scripts/export_mlp.py`: export + parity report (onnxruntime vs numpy vs Rust),
  exits non-zero when parity fails.

## Results (cloud)
- Fixture actor 61 → 64 → 32 → 14 (input normalisation, Gemm/MatMul, ELU, ReLU,
  Tanh, Mul, Clip): numpy reference and Rust both match onnxruntime within
  atol 1e-5 / rtol 1e-4 on 64 and 256 random inputs.
- A corrupt file is rejected with an error (exit 2), and so is an unsupported op
  (Softmax) at export.
- Rust unit tests in `mlp.rs`: hand-computed forward, and truncated / trailing /
  bad magic / oversized / mismatched / NaN files rejected.

## Results (gha-linux, official policies)
Pending the CI run of this change (alpha_walking, alpha_stand, roulade).

## Not done yet
- A policy controller in the runtime: it needs the observation builder of our
  own robot, which comes with its training env (G4.3).
