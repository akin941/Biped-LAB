# MAC-42 terrain candidates through the runtime from 8 starts, M5 (MAC-44, WP4-36)

```
env: mac-m5 (macOS 27.0.1 arm64), 2026-11-20
revision: 32a0ac7 (main); runtime built with cargo build --locked --release
run: runtime_sim.py --seconds 20 --init-seed 0..7 (perturbed keyframe, env reset ranges), no --realtime,
     --torque-limit 1.91 --motor-no-load-speed 5.45, at 0.15 and 0.25 m/s
candidates: the MAC-42 averages that passed in Python (k3_6900, k3_7450, k5_6950, k5_7250)
```

All 64 runs: no fall, no latched fault, 1000 periods.

| candidate | 0.15: mean vx / mean wz [min..max] | 0.25: mean vx / mean wz [min..max] | worst \|wz\| | pass |
|---|---|---|---|---|
| k3_6900 | 0.173 / +0.076 [+0.069..+0.086] | 0.252 / +0.088 [+0.072..+0.100] | 0.100 | no |
| **k3_7450** | 0.192 / **+0.021** [−0.022..+0.045] | 0.264 / **+0.051** [+0.013..+0.078] | **0.078** | **yes** |
| k5_6950 | 0.178 / +0.089 [+0.074..+0.103] | 0.261 / +0.106 [+0.078..+0.123] | 0.123 | no |
| k5_7250 | 0.185 / +0.084 [+0.059..+0.110] | 0.266 / +0.224 [+0.203..+0.252] | 0.252 | no |

## Verdict
- **Seed part met by k3_7450** (MAC-41 models 7400 + 7450 + 7500), ONNX sha256 `959ab77b081d202e87e767ff12feb73a3547812285c7d8c1e9e682776c36bce0`.
  |mean wz| is 0.021 / 0.051 and vx is 128 % / 105 % at 0.15 / 0.25, with no fall or fault from any start.
- **Caveat:** k3_7450 is not a stable window like k3_6900. In MAC-42 its own k=3 neighbours failed the Python bar
  (k3_7400 |wz| 0.096, k3_7500 0.112 with one flat fall). k3_6900, which passed there with both neighbours, yaws +0.088 at 0.25
  through the runtime. All candidates yaw positive through the runtime.
- **Realtime part pending:** a quiet `--realtime` run of k3_7450 from `--init-seed 0` at both commands. The wait script is running,
  and its result will be added to this record.

Files: `mac44/<candidate>-summary.json` (per-start rows, mean/min/max, ONNX and bpml sha256).
