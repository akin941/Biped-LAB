# The STS3215 policy walks through the Rust runtime with the servo model, M5 (MAC-38, WP4-30)

```
env: mac-m5 (macOS 27.0.1 arm64), 2026-11-20, quiet Mac (VM < 0.03 cores during the runs, load avg ~2)
revision: 397076e (main); runtime built with cargo build --locked --release
policy: MAC-37 fine-tune B (STS3215 @ 7.4 V curve), ONNX sha256 13070c6a…63bfb,
        policy.bpml sha256 a7e090fbf4ac2f391fe9386df9f050f7b435ff101a760fa39f541fb0f1912b70
run: runtime_sim.py --seconds 20 --realtime --torque-limit 1.91 --motor-no-load-speed 5.45,
     --command 0:20:v:0:0 for v = 0.15 and 0.25
```

MLP parity: Rust vs onnxruntime 1.19e-6, numpy vs onnxruntime 1.91e-6.

| command | vx_body_mean | of cmd | wz | distance | fell | latched | periods | misses | period_max | max torque demand |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.15 | 0.163 | 109 % | +0.022 | 2.91 m | no | 0 | 1000 | 0 | 29.9 ms | 0.78 N·m |
| 0.25 | 0.248 | 99 % | +0.043 | 4.38 m | no | 0 | 1000 | 0 | 30.0 ms | 1.02 N·m |

Supervisor events: no torque clamp, position clamp, derate or non-finite target. Slew-limit events: 0 at 0.15 and 37 at 0.25.
Compute mean / max: 105 / 326 µs and 103 / 180 µs. Bridge `wall_dt`: mean 20.0 ms, p1 / p50 / p99 13 / 20 / 29 ms.

## Verdict
**WP4-30 acceptance met for both commands.** No fall, no fault, 1000 periods, 0 deadline misses, vx ≥ 99 % of the
command and |wz| ≤ 0.043. Through the runtime it walks closer to the command and straighter than MAC-16
(the MAC-14 policy with an ideal actuator: 0.194 / 0.285, wz −0.19 / −0.15). The runs waited for a quiet
window (2 min of VM < 2 cores), and with the Mac quiet there were no deadline misses. `safety.rs` was not changed.

Files: `mac38/mlp.json`, `mac38/v0.15.json`, `mac38/v0.25.json` (paths shortened; `.bplog` / `.npz` not committed).
