# Apple Silicon (MacBook Pro M5) compatibility assessment

Status: **paper assessment + GitHub macOS arm64 CI only. Nothing has run on the
owner's M5 yet** (tracked in `docs/handoffs/MAC_LOCAL_HANDOFF.md`).

| Component | Expectation on M5 | Basis | Evidence level |
|---|---|---|---|
| Python 3.12 via uv | works | uv ships arm64 builds | U until mac-m5 |
| mujoco 3.10.0 | works (wheel `macosx_11_0_arm64`) | `uv.lock` | V (lockfile); gha-macos pending first CI run |
| onnxruntime 1.24.4 CPU | works, needs macOS ≥ 14 | wheel `macosx_14_0_arm64` | V (lockfile) |
| Our tests (`pytest`) | pass | CI matrix includes `macos-15` | gha-macos once CI runs |
| MuJoCo interactive viewer | `python -m mujoco.viewer` works; `launch_passive` needs `mjpython` on macOS | MuJoCo docs behaviour | U → handoff MAC-02 |
| CoreML / Metal EP for ONNX | not needed (policy ≈ 0.03 ms on x86 CPU) | microduck docs | — |
| torch MPS | available in torch ≥ 2.x; useful only for large PPO minibatches | microduck-lab auto-uses it only for large batches | U → Phase 3 experiment |
| mjlab / MuJoCo-Warp training | **does not run** (CUDA required) | microduck_rl README:24, uv.lock markers | V |
| microduck_rl `uv sync` | likely resolves (CUDA deps are linux-marked) but training won't run | uv.lock markers | U |
| BAM (Rhoban/bam) | pure Python + numpy, expected OK | git source; to be checked in WP1-06 | U |
| microduck runtime `cargo test` | upstream says passes on macOS | microduck docs | U (not built here) |
| microduck-lab | targets Macs; CI on macos-14; M5 untested | its CI config | U |

Performance: the cloud baseline (~18.7k Microduck physics steps/s single-thread on a
2.1 GHz Xeon) is **not** a prediction for M5. MAC-03 measures it natively.

Memory: 32 GB is ample for CPU-vectorized MuJoCo (each Microduck env is a few MB).
Process-parallel envs ≈ number of performance cores is the expected sweet spot (U).
