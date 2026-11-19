# Microduck: DR ablation and ONNX check on the G3.3 policy, M5 (MAC-34, WP3-05, G3.4 / G3.5)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-20
revision: dd646be (main), git_dirty false
A: model_19700 (MAC-32, trained with --dr), ONNX from the MAC-32 export
B: model_19700 continued 1000 it WITHOUT --dr (MAC-32 flags otherwise), 861 s; training falls
   0.04–0.12 / it, falls rule not fired. B = 20400, the newest no-DR checkpoint that meets G1.3
eval: eval_velocity_env.py --actuators bam --envs 8 --twist 0.2 0 0 --twist 0.3 0 0 --twist 0 0 0,
      without and with --dr; bipedlab eval-policy --model microduck-walk --actuator bam --repeats 4
```

## G3.4 table (vx m/s, |wz| rad/s; each cell noise on / noise off; falls of 8 envs)
| | A, DR off | A, DR on | B, DR off | B, DR on |
|---|---|---|---|---|
| vx at 0.2 | 0.179 / 0.177 | 0.174 / 0.171 | 0.191 / 0.182 | 0.206 / 0.195 |
| vx at 0.3 | 0.268 / 0.271 | 0.268 / 0.269 | 0.282 / 0.279 | 0.299 / 0.295 |
| \|wz\| at 0.2 | 0.006 / 0.027 | 0.018 / 0.029 | 0.028 / 0.035 | **0.087** / 0.048 |
| \|wz\| at 0.3 | 0.039 / 0.009 | 0.044 / 0.048 | 0.037 / 0.029 | **0.097** / 0.018 |
| \|wz\| at 0 | 0.000 / 0.001 | 0.002 / 0.008 | 0.000 / 0.000 | 0.016 / 0.010 |
| falls (0.2 / 0.3 / 0) | 0 / 0 / 0 | **0 / 0 / 0** | 0 / 0 / 0 | **1 / 2 / 1** |

G1.3 (BAM, 4 × 10 s): A gives 0.2 → 0.168–0.170, wz −0.023, 0 falls, and **meets it**. B gives 0.171–0.173,
wz −0.032, 0 falls, and also meets it. In the no-DR run, 19800, 20300 and 20400 meet G1.3; 19900 fell once
in turn_left_0.5, and the rest fail on |wz| (0.05–0.10) or speed.

## Verdict
- **G3.4: the policy trained with DR (A) degrades less under DR.** With DR on, A keeps its speed (−3 % at 0.2,
  0 % at 0.3) and never falls. B, trained without DR, falls 4 times in the DR runs, and its yaw with
  observation noise roughly doubles. Without DR, B is a few percent faster. It only shows its
  weakness once the dynamics are randomized.
- **G3.5 (mac-m5 part): met.** A's exported ONNX meets G1.3 through `bipedlab eval-policy`, the same runner
  the official policies use, on the M5.
- model_19700 stays the reference.

policy.onnx sha256: A `398f399aab64332d13083a81a8b31ff96c06c40b2622b76a79eb30e424bb2846`,
B `f7cac8af4a7151aec4c2d195b33ee72e5f1bfbb467590ba59bb6c0f39835e996` (not committed).
Files: `mac34/A-model_19700/` and `mac34/B-model_20400/` each hold `env-nodr.json`, `env-dr.json`, `g1-3.json`,
`run.json` and `report.json` (history every 10th iteration; A's are from MAC-32). `mac34/nodr-checkpoints/`
holds G1.3 for every no-DR checkpoint 19800–20700. Also `watch.log`.
