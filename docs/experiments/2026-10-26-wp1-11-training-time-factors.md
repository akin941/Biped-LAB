# Training-time factors and the walking tracking gap (WP1-11)

```
env: gha-linux, CI run 40309206573, job pretrained-policy (PR #11), CPU AMD EPYC 7763
policy: alpha_walking.onnx, scene.xml, init upstream, BAM port (ADR-0010)
each row: 8 runs, 1 from the upstream start + 7 from starts perturbed by < 1 encoder
count (uniform ±1e-3 rad on the 14 joints, seeds 1..7); mean ± std over the 8 runs
```

## Why repeats

The first CI run of this PR (run 40308058923) gave different BAM numbers at default
settings than PR #10 (forward vx 0.0595 vs 0.0616 m/s, different state digests) although
the default code path is unchanged. Locally (cloud) a synthetic policy gives identical
digests on main and on this branch; disabling numpy's AVX-512 dispatch changes the BAM
digest but not the MJCF one. BAM's friction model (`exp`, `**`) goes through numpy's
CPU-dispatched math, and the walking gait amplifies last-bit differences, so GitHub
runners with different CPUs give slightly different single runs. MJCF runs are not
affected. A single BAM run is therefore one sample; tracking is now reported as a spread.

## Results

Speeds in m/s, yaw in rad/s (gyro z, second half of the run). No fall in any run.

| factor | forward vx (cmd 0.2) | backward vx (cmd −0.15) | turn yaw (cmd 0.5) |
|---|---|---|---|
| none (upstream CPU sim) | 0.059 ± 0.003 (30 %) | −0.026 ± 0.000 (17 %) | 0.313 ± 0.010 (63 %) |
| delay 3 physics steps | 0.076 ± 0.000 (38 %) | −0.029 ± 0.001 | 0.357 ± 0.014 (71 %) |
| delay 6 physics steps | 0.088 ± 0.000 (44 %) | −0.033 ± 0.001 | **0.405 ± 0.007 (81 %)** |
| joint_vel lag 1 step | 0.072 ± 0.001 (36 %) | −0.031 ± 0.000 | 0.349 ± 0.018 (70 %) |
| vin 6.5 V | 0.061 ± 0.000 | **+0.005 (stops)** | 0.291 ± 0.014 |
| vin 8.2 V | 0.065 ± 0.001 | −0.031 ± 0.001 | 0.263 ± 0.011 |
| delay 4 + vel lag 1 | **0.090 ± 0.001 (45 %)** | −0.036 ± 0.001 (24 %) | 0.335 ± 0.010 (67 %) |

The run-to-run spread is small (forward vx std ≤ 0.003 m/s), so the differences
between rows are real effects, not noise.

## Interpretation

- Each training-time effect moves the policy closer to its command. Command delay is
  the largest: 6 physics steps take forward speed from 30 % to 44 % and turning from
  63 % to 81 % of the command. The joint_vel lag adds a smaller gain. This matches the
  policy having been trained with both.
- Voltage matters in the expected direction for backward walking only: at 6.5 V the
  policy no longer walks backward. 8.2 V does not help.
- **G1.3 still fails.** Best forward speed is 0.090 m/s; the gate needs 0.14–0.26 m/s.
  The turn command is within ±30 % with a 6-step delay, but G1.3 is defined on forward
  speed and is not relaxed.
- What is left is not explained by the three effects. Remaining training-time
  differences (IMU misalignment, encoder bias, 0–1 step gravity/gyro delay) are
  zero-mean randomisations and are unlikely to cause a systematic shortfall. Contact
  and friction parameters of the training simulator (mjlab, MuJoCo Warp) differ from
  this CPU scene and are the next candidate (WP1-13).

## Reproduce

```
uv run bipedlab eval-policy alpha_walking.onnx --actuator bam --bam-delay 6 --repeats 8
uv run bipedlab eval-policy alpha_walking.onnx --actuator bam --bam-delay 4 --vel-lag 1 --repeats 8
```
