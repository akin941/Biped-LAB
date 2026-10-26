# Evidence: Phase 1 cloud baseline (WP1-02, WP1-03, WP1-04)

```
date: 2026-10-26
env: cloud (Linux 6.18 x86_64, Intel Xeon @ 2.10GHz, 4 vCPU, Python 3.12.3)
deps: mujoco 3.10.0, numpy 2.4.1, onnxruntime 1.24.4 (uv.lock)
upstream: microduck_rl b02fdea4902c2b93b7d34a83a6f05da4ececb7d3
command: uv run pytest -rs
result: 25 passed, 2 skipped (pretrained weights not downloadable from this container: HF proxy 403)
```

## Observations

| Item | Value |
|---|---|
| Microduck scene.xml | 14 position actuators kp 0.55, forcerange ±0.96 N·m, mass 0.737 kg, dt 0.002 |
| Keyframes | INIT, STAND, SIT, FOLD |
| Physics speed (dt 0.005) | 16–19k steps/s single thread (two runs) ≈ 80–93× real time |
| Open-loop STAND (MJCF actuators held at STAND ctrl) | falls (tilt > 60°) at 0.86 s, forward pitch |
| toy_biped_v0 open-loop STAND, 5 s | stands; max tilt 0.26°, trunk 0.213 m |
| Determinism | identical state digest across repeated runs, same platform |

Not covered here: behaviour of the real pretrained policies (needs download → CI
job `pretrained-policy`), BAM actuator (WP1-06), any Mac result (MAC-01..04).
