# Phase 5 bring-up plan (WP5-00, draft)

**Status:** plan only, written in the cloud on 2026-11-23. Nothing here has run on hardware.
Every step below is `env: hw`, needs the owner's approval to start (no power-on without it),
and is closed only by evidence from the robot (S-6). It assumes the G4.5 decision (servo,
bus voltage, e-stop form) is made; servo-specific notes are for the STS3215 candidate.

## Rules for every session
- E-stop in hand and tested at the start of the session (S-3). It cuts motor power in
  hardware; the software stop is a second layer, not a replacement.
- Battery: fused, balance-charged, never charged unattended (S-4). Bench supply with a
  current limit for single-joint work.
- Runtime started with a low torque budget (`--torque-fraction`, S-5), raised one step per
  passed stage. Telemetry log on (`--log`) for every run; the HTML report goes with the result.
- One change at a time. A failed stage goes back to the previous one; no skipping.
- Results are written to `docs/handoffs/` with `env: hw`, date, revision and what was seen.

## Stages (the Phase 5 exit gates, in order)
| # | Stage | Setup | Pass | Abort / back off |
|---|---|---|---|---|
| 1 | E-stop test | Powered, no servo torque | Pressing the e-stop removes motor power within one control period; latching; the runtime sees it as a fault | Any motion or delay; anything that needs software to stop |
| 2 | Bus and ids | Servos on the bench, one at a time | Ping by the id in the joint map; voltage and temperature read back; no comm errors over 1 min | Id clash, checksum errors |
| 3 | Single-joint calibration | Each joint assembled, robot on a stand | Sign: a +0.1 rad sim target turns the joint the sim way. Offset: joint held at its sim zero, read servo angle → `offset_rad` (within ±0.8 rad). Range: the supervisor's clamp stops before the hard stop | Wrong direction, offset outside ±0.8 rad (horn spline off by a tooth), hitting a hard stop |
| 4 | Suspended motion | Robot hanging, feet free, torque fraction low | Stand pose holds; a slow sine on each joint tracks; the policy runs in the air at 0 command without faults; tilt/overspeed faults latch on purpose tests | Oscillation, overheating, any fault not explained |
| 5 | Supported standing | Feet on the floor, harness or hand support | Stands 60 s on the stand policy, then the walking policy at 0 command, temperatures stable | Falls into the harness, current/temperature derate |
| 6 | First steps | Harness, 0.05 → 0.15 m/s | Steps forward without falls; yaw drift recorded | Any fall, foot slip that grows |
| 7 | Sim-to-real report | Logs from 4–6 | Same commands replayed in sim with the same policy; speed, yaw drift, joint tracking and torque compared | — |

## Software that exists today (cloud / CI evidence only)
- Runtime control loop, safety supervisor model, command gate, telemetry (WP4-04/13).
- Walking policy k3_6900 through the runtime with the STS3215 torque-speed model (MAC-46).
- Servo packet codecs and simulated buses: STS3215 (`runtime/src/sts.rs`), Dynamixel X
  (`runtime/src/dxl.rs`). Neither opens a device yet.
- Joint map and calibration file (`runtime/src/jointmap.rs`,
  `hardware/calibration/biped_v1_joints.example.txt`): sim angles ↔ servo angles with id,
  direction and offset per joint (WP5-00).
- STS bus backend (`runtime/src/stsbus.rs`, WP5-01): packets over a port with timeouts and
  retries; can open a serial device file, never used on one yet.

## Software still missing before stage 2 (cloud work, no hardware needed)
1. ~~Serial backend~~ done for STS (WP5-01, `runtime/src/stsbus.rs`): ping, scan, register
   read/write, sync goals, sync state read, timeouts, retries and error counters, tested on the
   simulated bus and over a socket pair. The device's baud rate is set outside the runtime
   (`stty -F <dev> 1000000 raw -echo`); first real use is stage 2. Dynamixel needs the same
   wrapper if G4.5 picks it.
2. ~~Hardware state source~~ done (WP5-02, `runtime/src/hwstate.rs`): servo positions and
   speeds through the joint map, bus voltage and temperature, in the same `State` the sim
   bridge sends; actions go out as one sync write. The IMU is a trait with an upright
   placeholder, which is only allowed with the robot on a stand (stages 2-4). Wired into
   `biped-runtime` (WP5-03): `biped-runtime --device /dev/ttyUSB0 --hello
   hardware/calibration/biped_v1_sts3215.hello --map MAP --imu upright --torque-fraction 0.2
   --seconds 30 --log run.bplog` is the stage 4 run; `--hw-sim` runs the same path in CI.
3. Bench tools: ~~scan, read every joint, 1-minute bus check~~ done (WP5-02, `biped-bench`,
   read-only, e.g. `biped-bench check --device /dev/ttyUSB0 --map MAP --seconds 60` is the
   stage 2 pass test). Still missing: move one joint by a small step with the torque capped,
   and record the calibration offset into the joint-map file (stage 3).
4. IMU driver: depends on the IMU part in the BOM (G4.5).
