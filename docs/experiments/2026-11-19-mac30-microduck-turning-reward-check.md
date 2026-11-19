# Microduck: why turning was lost, scored under the training reward, M5: verdict (b) (MAC-30, WP3-18)

```
env: mac-m5 (macOS 27.0.1 arm64), 2026-11-19; eval only (no training)
revision: 70b875d (main), git_dirty false
eval: eval_velocity_env.py --actuators bam --envs 8 --seconds 10 --track-lin-std2 0.05
      --posture-gate --track-lin-ref-speed 0.4 --twist 0 0 1.0 / 0 0 0.5 / 0.2 0 0 / 0 0 0
policies: MAC-24 policy.onnx (turned 0.965 at 1.0 in its sweep), MAC-28 model_17300, MAC-29
          model_17600 (both exported with --resume <ckpt> --iters 0)
```

## Results (noise off; 0 falls everywhere)
| policy | twist | wz | vx | reward/step | track_ang | track_lin | upright | pose |
|---|---|---|---|---|---|---|---|---|
| MAC-24 | turn 1.0 | **0.880** | 0.011 | **0.1064** | 0.0267 | 0.0316 | 0.0214 | 0.0066 |
| MAC-28 17300 | turn 1.0 | 0.003 | 0.000 | 0.0520 | 0.0070 | 0.0399 | 0.0069 | 0.0022 |
| MAC-29 17600 | turn 1.0 | 0.005 | 0.000 | 0.0475 | 0.0074 | 0.0399 | 0.0067 | 0.0023 |
| MAC-24 | turn 0.5 | 0.002 | 0.000 | 0.1159 | 0.0250 | 0.0399 | 0.0247 | 0.0114 |
| MAC-28 17300 | turn 0.5 | 0.004 | 0.000 | 0.1120 | 0.0249 | 0.0399 | 0.0248 | 0.0089 |
| MAC-29 17600 | turn 0.5 | 0.002 | 0.000 | 0.1142 | 0.0249 | 0.0399 | 0.0246 | 0.0114 |
| MAC-24 | fwd 0.2 | −0.001 | 0.000 | 0.0317 | 0.0398 | 0.0017 | 0.0017 | 0.0008 |
| MAC-28 17300 | fwd 0.2 | −0.007 | 0.165 | 0.0651 | 0.0261 | 0.0116 | 0.0081 | 0.0005 |
| MAC-29 17600 | fwd 0.2 | −0.050 | 0.140 | 0.0677 | 0.0304 | 0.0106 | 0.0075 | 0.0002 |
| MAC-24 | zero | 0.001 | 0.000 | 0.1657 | 0.0399 | 0.0399 | 0.0399 | 0.0172 |
| MAC-28 17300 | zero | −0.001 | 0.000 | 0.1565 | 0.0398 | 0.0399 | 0.0398 | 0.0106 |
| MAC-29 17600 | zero | −0.001 | 0.000 | 0.1626 | 0.0399 | 0.0399 | 0.0400 | 0.0158 |

## Verdict: (b), an optimisation problem
At turn 1.0, MAC-24's turning earns 0.106 reward/step under the reward MAC-27 / MAC-29 train
with. That is about twice the standing policies (0.052 and 0.048). Its yaw term is about 4× theirs,
and the gated upright term about 3×. Turning still pays more, so the later policies did not lose it
because the reward prefers standing. They forgot it during the slow-command and mirror-loss
fine-tunes. Rehearsing turn commands should bring it back, for example by oversampling
turn-only commands, or by mixing MAC-24's turning into the fine-tune (distillation or a
replay of its commands).

Caveat: at turn 0.5, all three policies stand, MAC-24 included. Standing there keeps 62 % of
the yaw term (0.025 of 0.040) and about 0.11–0.12 reward/step, so under the fixed yaw kernel
0.5 has little pressure. A speed-relative yaw kernel addresses that (MAC-28 tried it and
the falls rose), and it could be revisited once turning at 1.0 is back.

Also visible: at fwd 0.2, MAC-24 stands (vx 0.000) and earns 0.032. The two stepping
policies earn about 2× that, which matches the posture gate doing its job.

Files: `mac30/mac24-seed0-policy.json`, `mac30/mac28-model_17300.json`,
`mac30/mac29-model_17600.json` (policy paths shortened).
