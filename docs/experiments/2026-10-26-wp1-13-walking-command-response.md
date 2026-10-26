# alpha_walking command response: start threshold and the upstream reference (WP1-13)

```
env: gha-linux, CI run 40313722550, job pretrained-policy (PR #13)
policy: alpha_walking.onnx, scene.xml, 10 s per scenario, no fall in any run
bam: our port (ADR-0010) with the WP1-12 start-state fix; repeats where marked
```

## Command sweep from the upstream start

| command | MJCF vx / yaw | BAM vx / yaw (2 runs) |
|---|---|---|
| forward 0.1 | 0.001 | 0.001 ± 0.000 |
| forward 0.2 (WP1-12) | 0.001 | 0.001 ± 0.000 |
| forward 0.3 | **0.124** | **0.097 ± 0.003** |
| forward 0.4 | **0.155** | **0.156 ± 0.001** |
| backward −0.3 | 0.000 | 0.000 |
| backward −0.15 (WP1-12) | 0.000 | 0.000 |
| lateral 0.3 (vy) | 0.041 | 0.045 |
| turn 1.0 rad/s | **0.359** | 0.018 ± 0.002 |
| turn 0.5 rad/s (WP1-12) | 0.000 | 0.001 |

From a still stand the policy has a **start threshold**: it does not step for forward
commands up to 0.2 m/s or backward up to 0.3 m/s. It starts walking at 0.3–0.4 m/s forward,
reaching 0.10–0.16 m/s, and lateral 0.3 gives a slow sidestep. MJCF and BAM now agree
closely, except turning in place, which starts only with MJCF.

## Once moving

Started from the straight-leg pose of the pre-fix bug (`--init straight`, a diagnostic only:
the fall sets the robot in motion), the same commands are followed (4 runs each):

| command | MJCF | BAM | BAM delay 4 + vel lag 1 |
|---|---|---|---|
| forward 0.2 | 0.088 | 0.061 ± 0.003 | 0.090 ± 0.001 |
| backward −0.15 | −0.040 | −0.027 ± 0.000 | −0.036 ± 0.001 |
| turn 0.5 | 0.208 | 0.301 ± 0.006 | 0.348 ± 0.013 |

So the gait exists below the threshold; the policy just does not start it from a still stand.

## Upstream reference

`microduck_rl` `docs/velstand_policy.md` (pinned sha), v8 sim eval, flat + backlash:
"vx tracking ≈ v7 and `alpha_walking` (0.4 m/s → 0.17 achieved, same as the teacher)".
Our measurement for the same command is **0.155–0.156 m/s** (both actuators), within
10 % of upstream's own number. The same document shows the policy family also under-tracks
turning in place (v7: 1.0 → 0.45 rad/s) and that turning was reworked in later versions.

## Interpretation

- Our CPU simulation reproduces upstream's published walking speed for this policy. The
  short tracking is a property of `alpha_walking`, not of our runner, actuator port or
  scene.
- **G1.3 fails as written** (vx 0.2 ± 30 %): from a still stand the policy does not walk at
  0.2 m/s, and at 0.4 m/s it reaches 0.16 m/s. Upstream's own number says no simulator
  fix will change that. G1.3 is not relaxed here. The roadmap records the finding next to
  the gate; meeting G1.3 needs a different policy (ours, Phase 3) or a later upstream
  policy (the `velstand` family).
- A useful sim-fidelity check came out of this: "forward 0.4 m/s → 0.17 m/s (upstream)
  reproduced within ±15 %" passes on gha-linux with both actuators.

## Reproduce

```
uv run bipedlab eval-policy alpha_walking.onnx --actuator bam --scenarios sweep --repeats 2
uv run bipedlab eval-policy alpha_walking.onnx --actuator bam --init straight --repeats 4
```
