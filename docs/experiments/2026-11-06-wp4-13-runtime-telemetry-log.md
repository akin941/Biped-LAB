# WP4-13: runtime telemetry log

**Environment:** cloud (container, 4 vCPU, x86_64) for every result below; the CI
`runtime` job repeats the 30 s real-time run with logging on gha-linux and uploads
the log, its `.npz` and the HTML report as the `runtime-telemetry` artifact.
Simulation only.

## Change
- `runtime/src/telemetry.rs`, `biped-runtime --log FILE`: one fixed-size
  little-endian record per period, written after the action is sent. It holds the
  received state (IMU, joints, bus voltage, temperature, e-stop), the gated
  command, the controller's wanted targets, the sent targets, torque enable,
  fault/event bits and compute time. A failed write turns logging off and is
  counted (`log_write_errors`). It never stops the loop. An unwritable path is
  refused at start (exit 2). The log is flushed on every exit, including the
  watchdog exit.
- `bipedlab.telemetry.runtime_log`: reader (a cut-off last record is dropped) and
  conversion to the WP2-01 schema plus runtime channels (`joint_wanted`,
  `torque_enable`, `fault_bits`, `compute_dt`, `bus_voltage`, `max_temperature`).
  The robot does not measure trunk position, body velocity or foot contact, so
  those channels are left out. Joint torque is estimated as `kp·(target − q)`
  clipped to the limit, and the sidecar marks it as estimated.
- `summarize`, `profile` and the HTML report skip channels that are missing. The
  report gains a "Güvenlik denetçisi" chart (torque enable, supervisor events).
- `bipedlab runtime-log LOG [--html]`, `scripts/runtime_sim.py --log FILE`.

## Results (cloud)
- Runtime log vs what the bridge received, 150 periods with a target jump, a push
  and a forward command: steps, joint positions, sent targets, bits and torque
  enable are bit-identical. The wanted jump (0.6 rad) is logged before the slew
  limit.
- Watchdog exit (stall at 1 s): 50 records flushed. Unwritable log path: exit 2
  before the loop.

| 30 s, 50 Hz | periods | deadline misses | mean / max compute | log records |
|---|---|---|---|---|
| real time, log on | 1500 | 0 | 47 µs / 392 µs | 1500, 0 errors |
| real time, log off | 1500 | 0 | 52 µs / 2.1 ms | – |
| as fast as possible, log on | 1500 | 0 | 7.5 µs / 125 µs | 1500, 0 errors |

Logging adds no measurable compute time here. The worst wall-clock period, 34–40 ms
with and without the log, comes from the container's sleep jitter, not from the
loop. The M5 number comes from MAC-09.

Size: 450 bytes per period for 10 joints, 22.5 kB/s, about 81 MB per hour.
On a Raspberry Pi SD card, write in chunks or rotate files before long runs
(WP5).
