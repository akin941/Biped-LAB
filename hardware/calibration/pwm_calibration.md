# PWM servo calibration (WP5-18, Phase 5 stage 3, DS3218MG build)

**Status:** tools only, tested in the cloud on the simulated supervisor
(`SimMcu`, the firmware's own logic). Nothing has run on a device. Powering a
servo needs the owner's approval, and so does applying the result to
`firmware/supervisor-core/src/joints.rs` (a supervisor limit, ADR-0015).

## Why
The supervisor sends `pulse = 1500 + sign · 636.6 · (q + offset)` µs for a
joint target `q` (sim frame, rad). On the real robot the joint does not land
exactly on `q`:
- the horn sits on a 25-tooth spline (14.4° per tooth), so up to ±7.2° bias;
- mirrored servos (right side) may turn the other way (sign);
- each servo's µs-per-degree differs a few percent (gain).

## Steps (per joint, servo in its bracket, robot on a stand)
1. `uv run python scripts/pwm_calibration.py plan > readings.csv` lists five
   angles per joint in the middle 60 % of its range, with the nominal pulse.
2. For each row run, through the USB-TTL adapter to the Pico:
   `biped-bench pwm-hold --device /dev/tty.usbserial-XXXX --hello
   hardware/calibration/biped_ds3218.hello --joint left_knee --to 1.27`
   (set the port first: `stty -f DEV 115200 raw -echo` on macOS). It holds
   the stand pose with that one joint moved for 10 s, then switches torque off.
3. During the hold, measure the real joint angle in degrees, with 0 = the
   joint's sim zero (phone inclinometer app or protractor; knee 0 = straight
   leg), and type it into the `measured_deg` column.
4. `uv run python scripts/pwm_calibration.py fit readings.csv --out
   calib_report.json` fits `measured = gain · commanded + bias` per joint.
   - Bias over 7.2°: move the horn one tooth and measure that joint again.
   - |gain| outside 0.85–1.15 or residual over 2°: check the servo model, the
     bracket for slack, and the readings.
   - Gain ≈ −1: the servo is mirrored; the fit flips its sign.
   - `unreachable_deg` > 0: that much of the range end is past the servo's
     end stop after trimming (hip pitch already loses ~1.7°); over 5° is refused.
5. The fit prints a `biped_ds3218_calibrated()` snippet for `joints.rs`. Send it
   for review: it goes into the firmware only with the owner's approval.

Joints not measured keep the nominal mapping and are listed as `missing`.

## Also measure: unpowered holding torque (WP5-19)
With the servo unpowered, hang a known weight on a known arm from the horn and
find the smallest torque that turns it (e.g. 50 g at 20 cm = 0.098 N·m). In sim
(`scripts/sit_stand.py`) the robot falls when the servos switch off if this is
below ~0.1 N·m, and stays where it was at ≥ 0.2 N·m.
