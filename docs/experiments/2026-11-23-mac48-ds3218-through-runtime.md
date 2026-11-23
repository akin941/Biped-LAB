# The DS3218MG policy through the Rust runtime with commanded feedback, M5 (MAC-48)

```
env: mac-m5 (macOS 27.0.1 arm64), 2026-11-23
revision: bc59934 (main, WP5-06: runtime --joint-feedback commanded, runtime_sim --model biped_ds3218); runtime rebuilt
policy: MAC-47 run A, ONNX sha256 d9a1d6ae1c82b4fc3dab53e4d9f2f5dd07d2231788f3867fc2bc3f100e4bdb5c,
        policy.bpml sha256 08be6b7116315e2fe2dcbdfb2cde65c12dd3da92144fd73c50f820abec52dcb1 (rust vs onnxruntime 1.3e-6)
run: runtime_sim.py --model biped_ds3218 --torque-limit 1.96 --motor-no-load-speed 7.0 ... -- --joint-feedback commanded
```

## Seed part (8 starts, no --realtime; 0 falls, 0 latched faults, 1000 periods in all 16 runs)
| cmd | mean vx [min..max] | mean wz [min..max] | max torque demand | supervisor events |
|---|---|---|---|---|
| 0.15 | 0.170 (113 %) [0.170..0.170] | +0.003 [+0.000..+0.007] | 1.48 N·m | none |
| 0.25 | 0.250 (100 %) [0.250..0.251] | +0.044 [+0.041..+0.046] | 2.03 N·m | slew_limit 27–32, torque_clamp 0 |

## Realtime (--init-seed 0; not quiet: VM ~7.1–7.2 cores, host load1 up to ~106)
| cmd | vx | wz | fell | latched | periods | misses | period_max |
|---|---|---|---|---|---|---|---|
| 0.15 | 0.170 (113 %) | +0.000 | no | 0 | 1000 | 0 | 30.5 ms |
| 0.25 | 0.250 (100 %) | +0.045 | no | 0 | 1000 | 0 | 30.0 ms |

## Verdict
**Met.** The DS3218MG policy walks through the supervised runtime with commanded-only feedback, from every start and in real time, with
|mean wz| ≤ 0.044. It matches the Python eval: MAC-47 gave vx 0.169 / 0.247 and runtime-obs wz +0.006 / +0.050. All MAC-38 rows pass despite
heavy machine load.
Note: at 0.25 the bridge's `max_torque_demand` peaks at about 2.0 N·m, just above the 1.96 N·m stall. That is a demanded torque, capped by the servo curve,
and the runtime's torque_clamp stayed 0. The Python sizing's walking max was 1.13 N·m, which measures the applied actuator force instead. This is worth reconciling
before hardware. No hardware was powered.

Files: `mac48/summary.json` (per-start rows, realtime rows, load note, sha256s).
