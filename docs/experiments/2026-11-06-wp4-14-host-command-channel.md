# WP4-14: host command channel (UDP) and keyboard teleop

**Environment:** cloud (container) for every result below. The CI `runtime` job
runs the same tests on gha-linux. Simulation only. A gamepad needs a new Python
dependency (owner review) and a device on the Mac, so it is not part of this
change. The packet format is the same whatever the input device.

## Change
- `runtime/src/channel.rs`, `biped-runtime --command-udp HOST:PORT`: 34-byte UDP
  packets (`BPCM`, version, sequence number, vx, vy, wz). The socket is
  non-blocking, at most 64 datagrams are read per period, and the newest accepted
  command goes to the command gate (ADR-0006), which still clamps the range, limits
  the rate, and stops 0.5 s after the last packet.
- The channel locks to the first sender's address and counts packets from any
  other address (`foreign`). A packet whose sequence number is not newer than the
  last accepted one is dropped (`stale`), and so is any packet that is not a
  command (`malformed`). The runtime summary reports these counts.
- No stop request travels over the channel. Stopping is the gate timeout (stop
  sending) or the e-stop (S-3), never a packet that could be lost.
- `bipedlab.command_channel` (sender, `KeyTeleop`), `scripts/teleop.py --hold` /
  `--keyboard` (curses, standard library; w/s a/d q/e, space = stop, x = quit).

## Results (cloud, real time, logged with WP4-13)
- A host sends vx 0.2, wz 0.5 for 1.5 s. The gated command ramps at ≤ 0.5 m/s²,
  holds 0.2 / 0.5, and returns to zero after the sender stops (0.5 s timeout, then
  the ramp).
- vx 5, vy −5, wz 9 is clamped to 0.3, −0.15, 1.0.
- A second sender starting 0.3 s later is ignored (`foreign` > 10). The command
  never exceeds the first sender's 0.1.
- Rust unit tests: packet round trip, duplicate, reordered, foreign and malformed
  packets, peer lock not taken by a malformed packet, socket poll returns the
  newest command. The three end-to-end tests passed 4 times in a row.

## Before hardware (WP5)
The default address is 127.0.0.1. Binding to a network interface on the robot
exposes the channel to the Wi-Fi. That needs an owner-reviewed decision on
authentication: the address lock alone does not stop spoofing. Done in WP4-16
(ADR-0014): packets are now authenticated and the 34-byte v1 format is gone.
