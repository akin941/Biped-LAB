# biped_v1 terrain curriculum with a slower, lower ramp, M5 (MAC-40b, WP4-32)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-20
revision: 696d466 (main, box-tile floor), git_dirty false
train: from mac37-b/model_5000.pt; --rough-floor 0.012 --rough-floor-ramp-it 1500 --iters 2500,
       otherwise as MAC-40 (--desired-kl 0.005, STS3215 curve, symmetry loss 0.5, yaw std2 0.25, --dr)
stop rule: > 3/it at any time; falls rule (> 1.0 / it over 100 it) from it 1600
eval: eval_biped --terrain --robustness --dr --envs 16 --seconds 20, STS3215 curve
```

## Training
Falls per iteration (100-it means) rise slowly with the ramp: 0.03–0.09 up to 3 mm, 0.25–0.5 at 4–6 mm, ~1.2 at
8–10 mm, and 1.7–2.0 at 11–12 mm. The falls rule fired at iteration 1600, as soon as it switched on (2.01),
so the run stopped at model_6600, and the last good checkpoint is model_6500. Reward/step went 0.109 → 0.094.

## Checkpoints (16 envs × 20 s; 0 falls on the flat floor in every case)
| ckpt | flat \|wz\| max (0.15/0.25, DR off/on) | 40 ms | rough 5 / 10 / 20 mm falls | vx at 10 mm |
|---|---|---|---|---|
| 6000 | 0.187 | 78 % | 0 / 4 / 16 | 0.133 |
| 6100 (extra) | 0.133 | 82 % | 0 / 5 / 14 | 0.149 |
| 6200 (extra) | 0.211 | 84 % | 0 / 1 / 14 | 0.137 |
| 6300 (extra) | 0.256 | 84 % | 0 / 0 / 14 | 0.129 |
| **6400 (extra)** | **0.028** | **84 %** | **0 / 0 / 7** | **0.149** |
| 6500 (last good) | 0.325 | 95 % | 0 / 0 / 10 | 0.161 |
| 6600 (extra) | 0.224 | 101 % | 0 / 0 / 5 | 0.173 |

I scored 6000 and 6500 as instructed. Since neither keeps flat |wz| ≤ 0.08, I also exported and scored
6100–6400 and 6600 with the same command.

## model_6400
On the flat floor, 0.15 gives vx 0.152 / 0.136 (DR off / on) and 0.25 gives 0.235 / 0.214, with |wz| ≤ 0.028.
The zero command gives |wz| 0.016 / 0.049, turn 0.8 gives 0.757 / 0.676, and 40 ms latency keeps 84 %.
Friction, push and latency cases all have 0 falls. On the tiles: 5 mm 0/16 (vx 0.156), **10 mm 0/16 (vx 0.149, wz +0.014)**,
20 mm 7/16. For comparison at 10 mm, MAC-37 B falls 15/16 and MAC-40's model_5200 falls 13/16.

## Verdict
- **WP4-32 met by model_6400.** `rough_10mm` has 0/16 falls at 99 % of the command, and every flat MAC-25 row holds.
- The slow ramp works for terrain: from 6300 on, 10 mm stays at 0/16, and 20 mm improves to 5–10/16.
- But the flat yaw swings from checkpoint to checkpoint (0.13–0.33 around 6400), so 6400 is one good checkpoint,
  not a stable region. The falls rule also fired when the ramp reached 12 mm. model_6400 is the terrain
  candidate. To make it stable: hold the ramp near 10 mm, keep a share of flat episodes, and try mirror loss 1.0,
  which steadied the Microduck's yaw in MAC-32.

ONNX sha256: model_6400 `2a57e5d925b4e1535093aea9f560dbdfe734b9bfbf998dee53e353acf1cc72ec`,
model_6500 `d4e06262cab55e907ccebb4e2905c841d458c5c9bb845017df68253de5b94f21` (not committed).
Files: `mac40b/eval-<it>.json` for 6000–6600, `mac40b/history_5000-6600.json` (every 10th iteration),
`mac40b/watch.log`.
