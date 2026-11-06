# WP4-15: Dynamixel Protocol 2.0 codec (no hardware)

**Environment:** cloud. Code and unit tests only. Nothing here opens a serial port
or talks to a servo; the hardware backend is WP5 and needs the owner's approval.

## Change
`runtime/src/dxl.rs` (std-only):
- Builds the instructions the runtime will need: ping, read, write, sync write,
  sync read, and `goal_positions` (one sync-write packet for all 10 joints, rad →
  ticks). It refuses a NaN target or a size mismatch.
- Parses status packets with checks for header, length, CRC-16 (0x8005) and
  instruction; it reports an incomplete, corrupt or non-status packet as such.
- Byte stuffing in both directions.
- X-series units: 4096 ticks per revolution, 2048 = 0 rad, 0.229 rpm per
  velocity tick. It also has the RAM control-table addresses used by the loop
  (torque enable 64, goal position 116, present current / velocity / position
  126 / 128 / 132, input voltage 144, temperature 146).
- `SimBus`: a simulated bus of servos with byte-level control tables that answers
  like the firmware (silent on a corrupt packet, no answer to broadcast writes,
  status packets in id order for sync read, no answer past the end of the table).

## Results (cloud)
- The ping (ID 1) and read-present-position (ID 1, 132, 4 bytes) packets equal
  the reference packets of the ROBOTIS Protocol 2.0 manual byte for byte,
  including the CRC.
- 10-joint goal positions → sim bus → sync read back: every joint within one
  tick (0.0015 rad).
- Corrupt packets are not executed, and status packets that are cut short,
  corrupted or not status packets are rejected.
- Rust tests: 29 in total (6 new).

## Before hardware (WP5)
The control-table addresses come from the X-series manual. On the first power-on
(owner present, S-5 low-torque mode), confirm them with ping and read before the
first write. The serial backend must read every status reply with a timeout and
treat a missing reply as a state timeout (S-2).
