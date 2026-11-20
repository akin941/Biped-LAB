# The MAC-42 terrain policy through the Rust runtime, M5 (MAC-43, WP4-35)

```
env: mac-m5 (macOS 27.0.1 arm64), 2026-11-20
revision: 5ad58d1 (main); runtime built with cargo build --locked --release
policy: MAC-42 k=3 average centred on 6900, ONNX sha256 2e94457133cef85e234858e483a1a2dbe1c9f83d223ac306f032aff023807db3,
        policy.bpml sha256 f91559700e24878b6cc0937adf46a64398e6bc1defef3b186f1042abae0466f0
run: runtime_sim.py --seconds 20 --realtime --torque-limit 1.91 --motor-no-load-speed 5.45 at 0.15 and 0.25 m/s
```

MLP parity: Rust vs onnxruntime 1.22e-6, numpy vs onnxruntime 1.67e-6.

## Tries (VM load during the runs; 100 % = one core)
| try | time (UTC) | VM max / mean | quiet? |
|---|---|---|---|
| noisy1 | 09:32 | 668 % / 439 % | no (the 2-min quiet window before it did not last) |
| try1 | 10:27 | 243 % / 152 % | no |
| **try2** | 13:18 | **22 % / 2 %** | **yes** |

## Results (try2, quiet; the other tries give identical vx / wz)
| command | vx_body_mean | of cmd | wz | fell | latched | periods | misses | period_max |
|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.172 | 115 % | +0.079 | no | 0 | 1000 | 0 | 25.0 ms |
| 0.25 | 0.253 | 101 % | **+0.132** | no | 0 | 1000 | 0 | 24.9 ms |

Max torque demand was 0.87 / 1.02 N·m, and the 0.25 run had 61 slew-limit events. Under load only the timing moved: noisy1 at 0.25 had 5 misses and
period_max 141 ms.

## Verdict
**Not met.** Safety and timing are clean on a quiet Mac: no fall, no fault, 1000 periods, 0 misses. But at 0.25 m/s the policy turns
+0.132 rad/s, against a limit of 0.08, and 0.15 is just inside at +0.079. Load is not the cause, since all three tries give the same yaw.
The Python env scored the same policy with wz −0.005 at 0.25 (mean over 16 envs with reset noise). The runtime runs one
env from a fixed start, so this single deterministic trajectory lands on one side of the yaw spread seen across MAC-41/42.
`safety.rs` was not changed.

Next (cloud): score the runtime at 0.25 over several start seeds or initial poses, and run the other passing averages
(k=3 7450, k=5 6950, k=5 7250) through the same check.

Files: `mac43/mlp.json`, `mac43/{noisy1,try1,try2}/v0.15.json`, `v0.25.json`, `vmload.log`, `mac43/steps.log`.
