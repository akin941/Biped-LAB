# The MAC-14 biped_v1 policy walks through the Rust runtime on the M5 (MAC-16, WP4-20)

```
env: mac-m5 (macOS 27.0.1 arm64), 2026-11-19; load avg ~9, no training running
revision: e9efa90 (main); runtime built with cargo build --locked --release
policy: MAC-14 policy.onnx (sha256 83766219233e60a14315ea78e9ee0ff68fcfa680787542f215d0cc8e8be6f78f)
        -> policy.bpml (sha256 7a917195ce120ed5965d725ac2195250583da210b6348c9cdef018c7c7ecbba7)
run: runtime_sim.py --seconds 20 --realtime, --controller policy:..., --command 0:20:v:0:0
```

## Results
MLP parity: Rust vs onnxruntime max abs error 1.43e-6, numpy vs onnxruntime 1.91e-6.

| command | vx_body_mean | of command | distance_xy | fell | latched | misses | periods | wz_mean | tilt max |
|---|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.194 | 129 % | 1.714 m | no | 0 | 0 | 1000 | −0.190 rad/s | 2.7° |
| 0.25 | 0.285 | 114 % | 3.286 m | no | 0 | 0 | 1000 | −0.148 rad/s | 2.4° |

period_max_us 29735 / 29807, compute max 736 / 153 µs, slew_limit events 23 / 71,
no torque or position clamps, wall 19.98 / 19.99 s.

## Conclusion
- **Acceptance met for both commands.** The supervised runtime at 50 Hz wall clock walks
  the robot without a fall or a fault, and vx is ≥ 50 % of the command.
- vx is above MAC-14's Python-env numbers with DR on (0.150 / 0.228) and close to those
  with DR off (0.179 / 0.264), as expected, since the runtime sim has no DR.
- **Yaw drift:** at a zero yaw command the robot turns right by 0.15–0.19 rad/s, so it walks
  a curve. That is also why `distance_xy` (net displacement) is less than vx × 18 s.
  The cloud should check this (yaw tracking in training and eval) before hardware.
- **Timing jitter at the bridge:** the recorder's `wall_dt` has mean 20.0 ms, p1 / p50 / p99 of
  12 / 20 / 29 ms, and about 430 of 1000 steps above 20.5 ms, with long and short steps
  alternating. The runtime counted 0 deadline misses and the 20 s schedule held.
  Where the jitter comes from (bridge or runtime) is open.

Files: `mac16/v0.15.json`, `mac16/v0.25.json`, `mac16/mac16-mlp.json` (paths shortened
to `<mac14 worktree>/…`). `.bplog` / `.npz` / `.bpml` are not committed.
