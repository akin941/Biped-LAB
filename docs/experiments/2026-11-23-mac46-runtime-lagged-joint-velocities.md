# MAC-42 averages through the runtime with lagged joint velocities, M5 (MAC-46, WP4-38)

```
env: mac-m5 (macOS 27.0.1 arm64), 2026-11-23
revision: 1ded258 (main; runtime feeds the previous period's joint velocities by default); runtime rebuilt
run: runtime_sim.py --seconds 20 --init-seed 0..7 --torque-limit 1.91 --motor-no-load-speed 5.45, no --realtime,
     0.15 and 0.25 m/s, MAC-44 policy.bpml exports; eval_biped --obs-mode runtime (16 envs × 20 s) for comparison
```

All 64 runtime runs: no fall, no latched fault, 1000 periods.

| candidate | cmd | runtime mean vx | runtime mean wz [min..max] | MAC-44 mean wz | eval --obs-mode runtime wz | pass |
|---|---|---|---|---|---|---|
| **k3_6900** | 0.15 | 0.163 (109 %) | **−0.044** [−0.056..−0.029] | +0.076 | −0.063 | ✔ |
| **k3_6900** | 0.25 | 0.239 (96 %) | **+0.009** [−0.009..+0.031] | +0.088 | −0.002 | worst 0.056 |
| k3_7450 | 0.15 | 0.181 (121 %) | −0.074 [−0.081..−0.063] | +0.021 | −0.080 | ✔ |
| k3_7450 | 0.25 | 0.260 (104 %) | +0.045 [+0.008..+0.093] | +0.051 | +0.035 | worst 0.093 |
| k5_6950 | 0.15 | 0.168 (112 %) | −0.013 [−0.023..−0.004] | +0.089 | −0.028 | ✔ |
| k5_6950 | 0.25 | 0.247 (99 %) | +0.047 [+0.013..+0.073] | +0.106 | +0.045 | worst 0.073 |
| k5_7250 | 0.15 | 0.174 (116 %) | +0.037 [+0.011..+0.057] | +0.084 | +0.033 | ✗ |
| k5_7250 | 0.25 | 0.257 (103 %) | +0.104 [+0.075..+0.125] | +0.224 | +0.103 | |

## Verdict
- **Seed part met.** Three candidates pass (no fall or fault, |mean wz| ≤ 0.08, vx ≥ 50 % at both commands). **Chosen: k3_6900**: it has the smallest
  worst-case |wz| (0.056), and its neighbours also passed MAC-42. ONNX sha256 `2e94457133cef85e234858e483a1a2dbe1c9f83d223ac306f032aff023807db3`.
- **WP4-38 fixes the mismatch.** With lagged joint velocities, the runtime's mean wz is within about 0.02 of the Python eval for every candidate,
  and the MAC-44 positive bias is gone (k3_6900 at 0.25: +0.088 → +0.009).
- **Realtime row pending:** a quiet-Mac `--realtime` run of k3_6900 from `--init-seed 0` at both commands. The wait script is running.

Files: `mac46/<candidate>-summary.json` (per-start rows, mean/min/max, MAC-44 means, eval --obs-mode runtime).
