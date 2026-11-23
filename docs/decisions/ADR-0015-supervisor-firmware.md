# ADR-0015: Supervisor MCU firmware in no_std Rust on the RP2040

Status: accepted (2026-11-23). Applies to the DS3218MG build (G4.5, WP5-08).
Nothing has run on an MCU yet.

## Context
The owner chose ten DS3218MG PWM hobby servos for the prototype (G4.5). ADR-0006 and
S-2 put the safety supervisor on separate hardware from the policy computer, and the
BOM (`hardware/bom/biped_ds3218_bom.md`) has an RP2040 board for that. ADR-0007 left
the firmware language to a later ADR. The supervisor's checks already exist in Rust
(`runtime/src/safety.rs`), and the runtime keeps no third-party crates.

## Decision
- **Language:** `no_std` Rust for the RP2040 (`thumbv6m-none-eabi`), one language
  across the robot software.
- **Split:** `firmware/supervisor-core` holds everything that does not touch a
  peripheral, with no third-party crates, unit-tested on the host and built for the
  RP2040 target in CI:
  - UART frames, `A5 5A`-synced with a CRC-16/CCITT. The Pi sends joint targets
    (1e-4 rad, sim frame) and enable / stop / clear flags. The MCU replies with the
    pulses it outputs, fault bits and the battery voltage.
  - Per-joint angle → pulse mapping (centre, µs/rad, sign, offset, end stops).
  - Joint-range clamp and a 10 rad/s slew limit, as in the runtime.
  - Command watchdog: hold after 60 ms, then pulses off and latched after 500 ms,
    also counted from power-up.
  - Battery warning (7.0 V) and stop (6.6 V).
  - A software stop. Latched faults clear only with torque off.
- **Hardware crate:** a separate binary crate adds the RP2040 HAL (PWM slices at
  50 Hz, UART, ADC for the battery divider). Its third-party crates get their own
  review, as `ort` did (ADR-0013).
- **Limits of PWM servos, stated:** no torque limit, no current and no temperature
  readback. A digital servo may also hold its position when the pulses stop, so
  "pulses off" is a best effort. The hardware e-stop (S-3) cutting servo power is
  the real stop and stays mandatory before first motion.
- **IMU placement:** the MPU6050 driver runs on the Pi for now (`runtime/src/mpu6050.rs`,
  enough for stand work). S-2 tilt detection on the supervisor needs the IMU on the
  MCU's I2C or a second sensor; that is an open point before walking on the floor.

## Consequences
The supervisor's logic is testable in CI without hardware. The runtime gets a PWM
link that speaks these frames instead of the STS bus. Changes to the supervisor's
limits need owner approval, like `runtime/src/safety.rs`. Torque and temperature
protections of the bus-servo design are lost with PWM servos; the stand procedure
(S-5) and the e-stop carry more of the risk.

The Pi side of the link is `runtime/src/pwmlink.rs` (`biped-runtime --servos pwm`,
WP5-09), which uses this crate's frame code directly, so both ends share one
definition. The runtime's over-temperature check cannot fire on this build (no
sensor), and its joint terms are the commanded targets.
