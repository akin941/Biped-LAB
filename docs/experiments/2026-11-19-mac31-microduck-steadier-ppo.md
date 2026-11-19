# Microduck: steadier PPO and more turn commands, M5: steadier, but one passing checkpoint (MAC-31, WP3-19)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-19
revision: e4ebe56 (main), git_dirty false
train: microduck, MAC-27 flags + --desired-kl 0.005 --turn-fraction 0.3, 256 envs, 1500 it
       (17300 -> 18800), 8 workers, seed 0, --resume mac28-seed0/model_17300.pt; 1756 s
       (a VM used ~4–6 cores)
```

## Training
All 1500 iterations ran. **The falls rule did not fire.** The watchdog's 200-iteration mean
stayed between 0.81 and 1.19. Falls per iteration (100-it means) went from 1.98 in the first
100 iterations down to 0.74–0.88 at the end (MAC-29: 1.28–2.15). Reward/step rose 0.026 → 0.034,
the yaw term 0.011 → 0.013 and tracking 0.020 → 0.021. loss_symmetry went 0.013 → 0.010.

## Checkpoints (BAM; G1.3 = 4 × 10 s, 0 falls in every case)
| ckpt | 0.2 vx | wz at 0.2 | back 0.15 | sweep 0.3 / 0.4 | turn 1.0 wz (env / sweep) | G1.3 pass? |
|---|---|---|---|---|---|---|
| 17600 | 0.134–0.136 | −0.064 | −0.140 | 0.239 / 0.315 | 0.004 / 0.003 | no (speed, wz) |
| 17900 | 0.132–0.134 | −0.061 | −0.001 | 0.234 / 0.276 | 0.003 / 0.001 | no |
| 18200 | 0.118–0.120 | −0.052 | −0.001 | 0.182 / 0.204 | **0.575** / −0.183 | no |
| **18500** | **0.151–0.154** | **+0.016** | −0.132 | 0.218 / 0.246 | 0.240 / 0.021 | **yes** |
| 18800 | 0.144–0.145 | +0.088 | −0.147 | 0.206 / 0.262 | 0.005 / 0.004 | no (wz) |
| extra 18400 | 0.131–0.135 | −0.093 | – | – | – | no |
| extra 18600 | 0.147–0.148 | −0.107 | – | – | – | no (wz) |

Turn 0.5 gives |wz| ≤ 0.006 everywhere, in both the G1.3 case and the env eval. The extra checkpoints
18400 and 18600 were evaluated only to see whether 18500's pass was isolated. They are not in the
protocol list.

Best is 18500. `eval_velocity_env` at 0.4 gives 0.271 / 0.273, 0 falls. It stands 4 × 20 s and
4 × 30 s with no falls (tilt ≤ 1.8°). At 0.2 for 4 × 20 s it gives 0.149–0.155, 0 falls.

## Verdict
- **G3.3 not confirmed.** One evaluated checkpoint passes, and two consecutive are needed. On either
  side of 18500 the policy is either fast enough with |wz| 0.06–0.11, or straight enough but slower.
  The yaw at 0.2 still changes sign from checkpoint to checkpoint.
- **The training is steadier** than MAC-29: the rule did not fire, falls came down, and reward rose
  throughout. So `--desired-kl 0.005` helped the stability, but the policy still swings around the
  G1.3 boundary.
- **WP3-18 not met.** Turning at 1.0 came back partly at 18200 (0.575) and 18500 (0.240) and then went
  away again. Turn 0.5 stays at 0. The sweep and the env eval disagree on turning at 18200
  (−0.18 vs 0.58). They start differently (upstream init vs. the env reset), so the turn is not robust yet.
- What to try next (cloud): keep these knobs and run longer (reward still rising), raise
  `--turn-fraction`, or average or select checkpoints using a G1.3 + |wz| score.

Files: `mac31/ckpt-<it>/` holds `g1-3.json`, `sweep-bam.json` and `turn-eval.json`, plus `eval.json`
and `long-g13.json` for 18500. `extra-18400/` and `extra-18600/` hold their `g1-3.json`.
Also `report.json` (history every 10th iteration), `run.json` (resume path shortened) and `watch.log`.
policy.onnx sha256: 18500 `70053902f4fa400e83d523cb2d0d09ff3df8ba73585bd92c1fc4ab7ac429cc76` (not committed).
