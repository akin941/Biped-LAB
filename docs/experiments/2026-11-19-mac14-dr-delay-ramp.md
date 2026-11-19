# biped_v1 with the DR action delay ramped in, M5: walks and tolerates latency (MAC-14, WP4-19)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-19
revision: a944b05 (main), git_dirty false
train: biped_v1, --dr (default: DR_DELAY_STAGES 0 ms until it 1000, 0-20 ms until 1500,
       then 0-40 ms), 256 envs, 3000 it, 8 workers, seed 0; 1815 s
eval: eval_biped.py --dr --robustness --envs 8 --seconds 10
```

## Results (vx in m/s, 0 falls in every case)
| case | MAC-14 (ramp) | MAC-13 A (no delay) | MAC-13 B / MAC-11 (delay from it 0) |
|---|---|---|---|
| 0.15 / 0.25, DR on | 0.150 / 0.228 (100 / 91 %) | 0.101 / 0.163 | 0 / 0 |
| 0.15 / 0.25, DR off | 0.179 / 0.264 | 0.135 / 0.216 | 0 / 0 |
| latency 20 / 40 / 60 ms (of 0.15) | 99 / 86 / 76 % | 60 / 42 / 31 % | 0 |
| friction_0.5 / 1.5 | 0.175 / 0.185 | 0.134 / 0.136 | 0 |
| push fwd / side 0.3 m/s | 0.180 / 0.177 | 0.137 / 0.135 | 0 |
| knee p95 / max, DR on 0.15 | 0.188 / 0.41 N·m | 0.195 / 0.52 | 0.066 / 0.26 |

`air_time` (100-iteration means): 0.007 (start) → 0.0025 (it 200) → 0.024 (it 500) →
0.0279 (it 1000) → 0.0277 (it 1500) → 0.0289 (it 2900). It did not fall when the delay
widened at 1000 or at 1500. Reward/step rose 0.053 → 0.110 (first → last quarter).

## Conclusion
- **The ramp works.** The gait forms while there is no delay. After that, adding the
  0–20 ms and 0–40 ms delay does not push the policy back into the standing optimum.
- G4.3 first pass is met with margin. The MAC-11 latency target (`latency_40ms` ≥ 50 %)
  is met at 86 %, and `friction_0.5` differs from nominal.
- Knee p95 stays around 0.19–0.22 N·m. The max reaches the 0.52 clip only in the push_sideways and
  latency_60ms cases. This does not change the WP4-05 servo note.

Files: `mac14/` holds `run.json`, `eval.json` and `report.json` (the `history` keeps
every 10th iteration, `history_stride: 10`).
policy.onnx sha256 `83766219233e60a14315ea78e9ee0ff68fcfa680787542f215d0cc8e8be6f78f` (not committed).
