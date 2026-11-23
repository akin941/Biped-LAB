//! PWM servo link (WP5-09, ADR-0015): the Pi side of the UART to the safety
//! supervisor MCU of the DS3218MG build.
//!
//! The Pi never drives a servo itself. Each control period it sends one
//! targets frame (`supervisor_core::frame`) and the RP2040 answers with one
//! status frame: the pulses it now outputs, its fault bits and the battery
//! voltage. The MCU applies its own joint limits, slew limit and watchdog
//! whatever the Pi sends (S-2).
//!
//! PWM hobby servos report nothing back, so the state is built from what was
//! commanded, and this is stated rather than hidden:
//! - `qpos` is the last target sent (sim frame), `qvel` its rate of change.
//!   The policy runs with `--joint-feedback commanded` (WP4-39) and never sees
//!   these as measurements; the runtime supervisor's slew and torque-budget
//!   clamps then act relative to the last command, not to the real joint.
//! - `max_temperature` is a fixed `NO_TEMPERATURE` (no sensor), so the
//!   over-temperature fault cannot fire on this build.
//! - `bus_voltage` is the MCU's battery reading.
//! - `estop` is raised while the MCU has a latched fault (watchdog,
//!   undervoltage, soft stop), so the runtime latches and stops too.
//!
//! A runtime fault (any latching bit in the action) sets the frame's stop
//! flag, which latches the MCU's soft stop. Latched MCU faults are cleared
//! only once, at start, with torque off; a cleared fault that comes straight
//! back (low battery) fails the start.
//!
//! The UART is 115200 8N1 raw, set outside this crate, e.g.
//! `stty -F /dev/serial0 115200 raw -echo`. A frame each way takes ~2.5 ms.
//! `SimMcu` runs the real `supervisor_core::Supervisor` behind a port, so the
//! link is tested in CI. Nothing here has run on an MCU yet (`env: hw`).

use crate::hwstate::Imu;
use crate::protocol::{Action, State};
use crate::safety::LATCHING;
use crate::stsbus::{Fault, Port};
use std::collections::VecDeque;
use std::io;
use std::time::{Duration, Instant};
use supervisor_core::frame::{Reader, Status, Targets, STATUS_LEN, TARGETS_LEN};
use supervisor_core::servo::JointPwm;
use supervisor_core::{bits, Limits, Supervisor, N_JOINTS};

/// Time for the MCU's status after a targets frame (two frames at 115200 baud
/// take ~5.3 ms).
pub const REPLY_TIMEOUT: Duration = Duration::from_millis(10);
/// Reported as the hottest servo: PWM servos have no temperature readback.
pub const NO_TEMPERATURE: f64 = 25.0;

pub struct PwmSource<P: Port, I: Imu> {
    pub port: P,
    pub imu: I,
    /// Last status from the MCU (pulses, bits, battery) for logs and tests.
    pub last_status: Option<Status>,
    /// Status frames that failed their CRC or framing.
    pub bad_frames: u64,
    reader: Reader<STATUS_LEN>,
    /// Reply to the last frame, not yet turned into a state.
    reply: Option<Result<Status, String>>,
    seq: u8,
    torque: bool,
    stop: bool,
    sent: Vec<f64>,
    prev: Vec<f64>,
    dt: f64,
    step: u64,
    t0: Instant,
}

impl<P: Port, I: Imu> PwmSource<P, I> {
    /// Opens the link: torque off, clear the MCU's latched faults (it latches
    /// its watchdog 0.5 s after power-up), and check they stay clear.
    /// `default` is the stand pose: the first `qpos` until a target is sent.
    pub fn new(port: P, imu: I, default: &[f64], dt: f64) -> Result<Self, String> {
        if default.len() != N_JOINTS {
            return Err(format!(
                "the PWM supervisor drives {N_JOINTS} joints, the hello has {}",
                default.len()
            ));
        }
        let mut s = PwmSource {
            port,
            imu,
            last_status: None,
            bad_frames: 0,
            reader: Reader::default(),
            reply: None,
            seq: 0,
            torque: false,
            stop: false,
            sent: default.to_vec(),
            prev: default.to_vec(),
            dt,
            step: 0,
            t0: Instant::now(),
        };
        s.port.clear();
        let status = s.exchange(true)?;
        if status.bits & bits::LATCHED != 0 {
            return Err(format!(
                "supervisor MCU faults stay latched after clear (bits {:#06x}; battery {:.2} V)",
                status.bits,
                status.battery_mv as f64 / 1000.0
            ));
        }
        s.reply = Some(Ok(status));
        Ok(s)
    }

    /// Sends one targets frame (current torque flag, stop flag and targets)
    /// and waits for the status with the same sequence number.
    fn exchange(&mut self, clear: bool) -> Result<Status, String> {
        self.seq = self.seq.wrapping_add(1);
        let mut targets = [0.0f32; N_JOINTS];
        for (t, q) in targets.iter_mut().zip(&self.sent) {
            *t = *q as f32;
        }
        let frame = Targets {
            seq: self.seq,
            torque_enable: self.torque,
            stop: self.stop,
            clear,
            targets,
        };
        self.port
            .send(&frame.encode())
            .map_err(|e| format!("supervisor link: {e}"))?;
        let deadline = Instant::now() + REPLY_TIMEOUT;
        let mut buf = Vec::new();
        loop {
            buf.clear();
            match self.port.recv(deadline, &mut buf) {
                Ok(true) => {}
                Ok(false) => return Err(format!("no status from the supervisor MCU (seq {})", self.seq)),
                Err(e) => return Err(format!("supervisor link: {e}")),
            }
            let mut found = None;
            for &b in &buf {
                if let Some(raw) = self.reader.push(b) {
                    match Status::decode(&raw) {
                        Ok(st) if st.seq == self.seq => found = Some(st),
                        Ok(_) => {} // a late reply to an earlier frame
                        Err(_) => self.bad_frames += 1,
                    }
                }
            }
            if let Some(st) = found {
                self.last_status = Some(st);
                return Ok(st);
            }
        }
    }

    /// One control period's state: the reply to the last frame sent (or to a
    /// keep-alive frame if none was sent since the last read), plus the IMU.
    pub fn read(&mut self) -> Result<State, String> {
        let status = match self.reply.take() {
            Some(r) => r?,
            None => self.exchange(false)?,
        };
        let (quat, gyro) = self.imu.read().map_err(|e| format!("imu: {e}"))?;
        let qvel = self
            .sent
            .iter()
            .zip(&self.prev)
            .map(|(q, p)| (q - p) / self.dt)
            .collect();
        self.prev.clone_from(&self.sent);
        let state = State {
            step: self.step,
            time: self.t0.elapsed().as_secs_f64(),
            quat,
            gyro,
            bus_voltage: status.battery_mv as f64 / 1000.0,
            max_temperature: NO_TEMPERATURE,
            qpos: self.sent.clone(),
            qvel,
            estop: status.bits & bits::LATCHED != 0,
        };
        self.step += 1;
        Ok(state)
    }

    /// Sends the runtime supervisor's action. With torque off the targets may
    /// be empty and the last ones are kept. A missing reply is reported by the
    /// next `read`, as a state timeout.
    pub fn apply(&mut self, action: &Action) -> Result<(), String> {
        if action.torque_enable {
            if action.targets.len() != N_JOINTS {
                return Err(format!("{} targets for {N_JOINTS} joints", action.targets.len()));
            }
            if action.targets.iter().any(|t| !t.is_finite()) {
                return Err("non-finite target".into());
            }
            self.sent.clone_from(&action.targets);
        }
        self.torque = action.torque_enable;
        // A runtime fault latches the MCU's soft stop; it stays set.
        self.stop |= action.fault_bits & LATCHING != 0;
        self.reply = Some(self.exchange(false));
        Ok(())
    }

    pub fn torque_on(&self) -> bool {
        self.torque
    }

    /// False when the MCU reports no tilt sensor (its S-2 tilt stop is off).
    pub fn mcu_tilt_sensor(&self) -> bool {
        self.last_status.is_some_and(|s| s.bits & bits::NO_TILT_SENSOR == 0)
    }
}

/// The supervisor MCU in software: `supervisor_core::Supervisor` behind a
/// port, with a joint table (the firmware's own with `SimMcu::ds3218`). Its clock
/// advances `ms_per_frame` per targets frame received, so runs are
/// reproducible; `battery_v` and `faults` (one per frame) are test inputs.
pub struct SimMcu {
    pub sup: Supervisor,
    pub now_ms: u32,
    pub ms_per_frame: u32,
    pub battery_v: f32,
    /// The MCU's own accelerometer (body frame), sampled once per frame;
    /// `None` = no sensor fitted (the tilt stop is off).
    pub accel: Option<[f32; 3]>,
    pub faults: VecDeque<Fault>,
    /// Targets frames received with a valid CRC.
    pub frames: u64,
    reader: Reader<TARGETS_LEN>,
    pending: Vec<u8>,
}

impl SimMcu {
    /// The firmware's joint table for the DS3218MG build (WP5-10).
    pub fn ds3218() -> Self {
        SimMcu::new(supervisor_core::joints::biped_ds3218())
    }

    pub fn new(joints: [JointPwm; N_JOINTS]) -> Self {
        SimMcu {
            sup: Supervisor::new(joints, Limits::default()),
            // Powered up a while ago: its start-up watchdog has latched.
            now_ms: 1000,
            ms_per_frame: 20,
            battery_v: 7.4,
            accel: None,
            faults: VecDeque::new(),
            frames: 0,
            reader: Reader::default(),
            pending: Vec::new(),
        }
    }
}

impl Port for SimMcu {
    fn send(&mut self, bytes: &[u8]) -> io::Result<()> {
        for &b in bytes {
            let Some(raw) = self.reader.push(b) else { continue };
            self.now_ms = self.now_ms.wrapping_add(self.ms_per_frame);
            if let Some(a) = self.accel {
                self.sup.on_accel(a, self.now_ms);
            }
            match Targets::decode(&raw) {
                Ok(t) => {
                    self.frames += 1;
                    self.sup.on_targets(&t, self.now_ms);
                }
                Err(_) => self.sup.bad_frame(),
            }
            let out = self.sup.tick(self.now_ms, self.battery_v);
            let mut reply = Status {
                seq: self.sup.last_seq,
                bits: out.bits,
                battery_mv: (self.battery_v * 1000.0) as u16,
                pulses_us: out.pulses_us.unwrap_or([0; N_JOINTS]),
            }
            .encode()
            .to_vec();
            match self.faults.pop_front() {
                Some(Fault::Drop) => reply.clear(),
                Some(Fault::Corrupt) => reply[6] ^= 0x5A,
                Some(Fault::Garbage) => reply = [&[0x00, 0xA5, 0x13][..], &reply].concat(),
                _ => {}
            }
            self.pending.extend(reply);
        }
        Ok(())
    }

    fn recv(&mut self, _deadline: Instant, buf: &mut Vec<u8>) -> io::Result<bool> {
        if self.pending.is_empty() {
            return Ok(false);
        }
        buf.append(&mut self.pending);
        Ok(true)
    }

    fn clear(&mut self) {
        self.pending.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hwstate::UprightImu;
    use crate::safety::FAULT_TILT;

    fn mcu() -> SimMcu {
        SimMcu::new([JointPwm::ds3218(-1.0, 1.0); N_JOINTS])
    }

    fn source() -> PwmSource<SimMcu, UprightImu> {
        let mcu = mcu();
        PwmSource::new(mcu, UprightImu, &[0.1; N_JOINTS], 0.02).unwrap()
    }

    fn action(on: bool, targets: Vec<f64>, fault_bits: u32) -> Action {
        Action {
            step: 0,
            torque_enable: on,
            fault_bits,
            targets,
        }
    }

    fn us(q: f64) -> u16 {
        JointPwm::ds3218(-1.0, 1.0).pulse_us(q as f32)
    }

    #[test]
    fn start_clears_the_power_up_watchdog_with_torque_off() {
        let mut hw = source();
        let s = hw.read().unwrap();
        assert_eq!((s.step, s.estop, s.qpos.clone()), (0, false, vec![0.1; N_JOINTS]));
        assert!((s.bus_voltage - 7.4).abs() < 1e-9 && s.max_temperature == NO_TEMPERATURE);
        assert_eq!(hw.last_status.unwrap().pulses_us, [0; N_JOINTS]); // torque off
        assert!(!hw.torque_on());
    }

    #[test]
    fn targets_become_pulses_and_the_commanded_state() {
        let mut hw = source();
        hw.read().unwrap();
        let t: Vec<f64> = (0..N_JOINTS).map(|k| 0.1 + 0.01 * k as f64).collect();
        hw.apply(&action(true, t.clone(), 0)).unwrap();
        let st = hw.last_status.unwrap();
        let want: Vec<u16> = t.iter().map(|&q| us(q)).collect();
        assert_eq!(st.pulses_us.to_vec(), want);
        let s = hw.read().unwrap();
        assert!(s.qpos.iter().zip(&t).all(|(a, b)| a == b));
        assert!((s.qvel[9] - 0.09 / 0.02).abs() < 1e-9 && s.qvel[0].abs() < 1e-9);
        // No new frame since: the read sends a keep-alive with the same targets.
        let s = hw.read().unwrap();
        assert!(s.qvel.iter().all(|v| *v == 0.0) && !s.estop);
        assert_eq!(hw.port.frames, 3);
    }

    #[test]
    fn the_mcu_clamps_and_slew_limits_on_its_own() {
        let mut hw = source();
        hw.read().unwrap();
        hw.apply(&action(true, vec![0.0; N_JOINTS], 0)).unwrap();
        let mut t = vec![0.0; N_JOINTS];
        t[0] = 5.0; // outside the range and far beyond one period's slew
        hw.apply(&action(true, t, 0)).unwrap();
        let st = hw.last_status.unwrap();
        assert!(st.bits & bits::SLEW_LIMIT != 0 && st.bits & bits::POSITION_CLAMP != 0);
        assert_eq!(st.pulses_us[0], us(10.0 * 0.02)); // 10 rad/s for 20 ms
    }

    #[test]
    fn a_runtime_fault_latches_the_mcu_soft_stop() {
        let mut hw = source();
        hw.read().unwrap();
        hw.apply(&action(true, vec![0.0; N_JOINTS], 0)).unwrap();
        hw.read().unwrap();
        hw.apply(&action(false, vec![], FAULT_TILT)).unwrap();
        let s = hw.read().unwrap();
        assert!(s.estop && hw.port.sup.latched() & bits::SOFT_STOP != 0);
        // The stop stays set: torque on again is refused by the MCU.
        hw.apply(&action(true, vec![0.0; N_JOINTS], 0)).unwrap();
        assert_eq!(hw.last_status.unwrap().pulses_us, [0; N_JOINTS]);
    }

    #[test]
    fn mcu_faults_show_as_estop() {
        let mut hw = source();
        hw.read().unwrap();
        hw.apply(&action(true, vec![0.0; N_JOINTS], 0)).unwrap();
        hw.read().unwrap();
        hw.port.battery_v = 6.5;
        hw.apply(&action(true, vec![0.0; N_JOINTS], 0)).unwrap();
        let s = hw.read().unwrap();
        assert!(s.estop && (s.bus_voltage - 6.5).abs() < 1e-9);
        assert_eq!(hw.last_status.unwrap().pulses_us, [0; N_JOINTS]);
    }

    #[test]
    fn a_low_battery_fails_the_start() {
        let mut mcu = mcu();
        mcu.battery_v = 6.0;
        let e = PwmSource::new(mcu, UprightImu, &[0.0; N_JOINTS], 0.02).err().unwrap();
        assert!(e.contains("latched") && e.contains("6.00 V"), "{e}");
    }

    #[test]
    fn lost_corrupt_and_noisy_replies() {
        let mut hw = source();
        hw.read().unwrap();
        hw.port.faults.extend([Fault::Garbage, Fault::Corrupt, Fault::Drop]);
        hw.apply(&action(true, vec![0.0; N_JOINTS], 0)).unwrap();
        assert!(hw.read().is_ok()); // noise before the frame is skipped
        hw.apply(&action(true, vec![0.0; N_JOINTS], 0)).unwrap();
        let e = hw.read().unwrap_err(); // CRC error: no valid status in time
        assert!(e.contains("no status"), "{e}");
        assert_eq!(hw.bad_frames, 1);
        hw.apply(&action(true, vec![0.0; N_JOINTS], 0)).unwrap();
        assert!(hw.read().unwrap_err().contains("no status"));
        hw.apply(&action(true, vec![0.0; N_JOINTS], 0)).unwrap();
        assert!(hw.read().is_ok());
    }

    #[test]
    fn bad_actions_and_joint_counts_are_refused() {
        let mut hw = source();
        assert!(hw.apply(&action(true, vec![0.0; 9], 0)).is_err());
        assert!(hw.apply(&action(true, vec![f64::NAN; N_JOINTS], 0)).is_err());
        assert!(PwmSource::new(mcu(), UprightImu, &[0.0; 4], 0.02).is_err());
    }

    #[test]
    fn the_firmware_table_holds_the_knee_at_mid_range() {
        let mut hw = PwmSource::new(SimMcu::ds3218(), UprightImu, &[0.0; N_JOINTS], 0.02).unwrap();
        let mut t = vec![0.0; N_JOINTS];
        t[3] = supervisor_core::joints::KNEE_CENTER as f64;
        t[8] = t[3];
        hw.apply(&action(true, t, 0)).unwrap();
        let p = hw.last_status.unwrap().pulses_us;
        assert_eq!((p[0], p[3], p[8]), (1500, 1500, 1500));
    }

    #[test]
    fn the_mcu_tilt_stop_shows_as_estop() {
        let hw = source();
        assert!(!hw.mcu_tilt_sensor());
        let mut mcu = mcu();
        mcu.accel = Some([0.0, 0.0, 9.8]);
        let mut hw = PwmSource::new(mcu, UprightImu, &[0.0; N_JOINTS], 0.02).unwrap();
        assert!(hw.mcu_tilt_sensor());
        hw.apply(&action(true, vec![0.0; N_JOINTS], 0)).unwrap();
        assert!(!hw.read().unwrap().estop);
        hw.port.accel = Some([9.8, 0.0, 0.0]); // fallen forward
        for _ in 0..10 {
            hw.apply(&action(true, vec![0.0; N_JOINTS], 0)).unwrap();
        }
        assert!(hw.read().unwrap().estop);
        assert!(hw.port.sup.latched() & bits::TILT != 0);
    }
}
