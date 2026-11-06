//! Safety supervisor model (S-2 in docs/security/SECURITY_AND_SAFETY.md).
//!
//! Runs every control period between the controller and the actuators. On the
//! real robot this logic belongs on the low-level controller, separate from the
//! high-level computer; here it is the executable model that the sim bridge
//! tests. The hardware e-stop (S-3) cuts motor power on its own and is not part
//! of this model; the software stop is.
//!
//! Per period, in order:
//! 1. state checks -> latching fault: non-finite state, software e-stop, tilt,
//!    joint overspeed, bus undervoltage, servo overtemperature;
//! 2. latched fault -> torque off (the servos go limp; damping remains);
//! 3. non-finite target -> hold the last safe targets this period;
//! 4. clamp targets inside the joint range minus a margin;
//! 5. slew-limit targets to the servo speed;
//! 6. torque budget: keep `kp * |target - q|` within the allowed fraction of the
//!    servo torque limit (first power-on uses a low fraction, S-5); low voltage
//!    or high temperature halves the budget before the stop thresholds.

use crate::protocol::{Hello, State};
use crate::tilt_from_quat;

/// Latching faults: once set, torque stays off until the runtime restarts.
pub const FAULT_TILT: u32 = 1 << 0;
pub const FAULT_OVERSPEED: u32 = 1 << 1;
pub const FAULT_NONFINITE_STATE: u32 = 1 << 2;
pub const FAULT_STATE_TIMEOUT: u32 = 1 << 3;
pub const FAULT_ESTOP: u32 = 1 << 4;
pub const FAULT_UNDERVOLTAGE: u32 = 1 << 5;
pub const FAULT_OVERTEMPERATURE: u32 = 1 << 6;
pub const N_FAULTS: usize = 7;
pub const LATCHING: u32 = (1 << N_FAULTS) - 1;
/// Per-period events (not latching): what the supervisor changed this period.
pub const EVENT_NONFINITE_TARGET: u32 = 1 << 8;
pub const EVENT_POSITION_CLAMP: u32 = 1 << 9;
pub const EVENT_SLEW_LIMIT: u32 = 1 << 10;
pub const EVENT_TORQUE_CLAMP: u32 = 1 << 11;
pub const EVENT_DERATE: u32 = 1 << 12;
pub const N_EVENTS: usize = 5;

#[derive(Debug, Clone, Copy)]
pub struct SafetyLimits {
    /// Targets stay this far inside each joint's range (rad).
    pub position_margin: f64,
    /// Max target change per second (rad/s); ~ the servo's no-load speed.
    pub max_target_speed: f64,
    /// Measured joint speed above this latches a fault (rad/s).
    pub max_joint_speed: f64,
    /// Body tilt above this latches a fault (rad).
    pub max_tilt: f64,
    /// Fraction of each servo's torque limit the targets may ask for. `None`
    /// leaves torque to the servo's own limit (until a derate applies).
    pub torque_fraction: Option<f64>,
    /// Bus voltage: below `derate` the torque budget halves, below `stop` latch.
    pub voltage_derate: f64,
    pub voltage_stop: f64,
    /// Hottest servo: above `derate` the torque budget halves, above `stop` latch.
    pub temperature_derate: f64,
    pub temperature_stop: f64,
}

impl Default for SafetyLimits {
    fn default() -> Self {
        SafetyLimits {
            position_margin: 0.05,
            max_target_speed: 10.0, // XL330/XC330 class, WP4-03
            max_joint_speed: 20.0,
            max_tilt: 45f64.to_radians(),
            torque_fraction: None,
            // 2S LiPo: 3.4 V and 3.2 V per cell.
            voltage_derate: 6.8,
            voltage_stop: 6.4,
            // Below the servo firmware's own shutdown (~70 deg C class default).
            temperature_derate: 60.0,
            temperature_stop: 68.0,
        }
    }
}

#[derive(Debug)]
pub struct Supervisor {
    limits: SafetyLimits,
    lower: Vec<f64>,
    upper: Vec<f64>,
    /// Max |target - q| per joint at the full torque limit: torque_limit / kp.
    full_error: Vec<f64>,
    last: Vec<f64>,
    pub latched: u32,
    /// Control step at which each latching fault first fired (index = bit).
    pub first_fault_step: [Option<u64>; N_FAULTS],
    /// Periods in which each event happened (index = bit - 8).
    pub event_counts: [u64; N_EVENTS],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    pub targets: Vec<f64>,
    pub torque_enable: bool,
    pub bits: u32,
}

impl Supervisor {
    pub fn new(hello: &Hello, limits: SafetyLimits) -> Self {
        let j = &hello.joints;
        Supervisor {
            limits,
            lower: j.iter().map(|j| j.lower + limits.position_margin).collect(),
            upper: j.iter().map(|j| j.upper - limits.position_margin).collect(),
            full_error: j.iter().map(|j| j.torque_limit / j.kp).collect(),
            last: j.iter().map(|j| j.default).collect(),
            latched: 0,
            first_fault_step: [None; N_FAULTS],
            event_counts: [0; N_EVENTS],
        }
    }

    pub fn latch(&mut self, fault: u32, step: u64) {
        if self.latched & fault == 0 {
            self.first_fault_step[fault.trailing_zeros() as usize] = Some(step);
        }
        self.latched |= fault;
    }

    /// Targets to send with torque off (the last safe ones).
    pub fn filter_off(&self) -> Vec<f64> {
        self.last.clone()
    }

    fn check_state(&mut self, s: &State) {
        let all = s.quat.iter().chain(&s.gyro).chain(&s.qpos).chain(&s.qvel);
        let power = [s.bus_voltage, s.max_temperature];
        let finite = all.chain(&power).all(|v| v.is_finite());
        if !finite || s.qpos.len() != self.last.len() || s.qvel.len() != self.last.len() {
            self.latch(FAULT_NONFINITE_STATE, s.step);
            return;
        }
        let l = self.limits;
        if s.estop {
            self.latch(FAULT_ESTOP, s.step);
        }
        if tilt_from_quat(s.quat) > l.max_tilt {
            self.latch(FAULT_TILT, s.step);
        }
        if s.qvel.iter().any(|v| v.abs() > l.max_joint_speed) {
            self.latch(FAULT_OVERSPEED, s.step);
        }
        if s.bus_voltage < l.voltage_stop {
            self.latch(FAULT_UNDERVOLTAGE, s.step);
        }
        if s.max_temperature > l.temperature_stop {
            self.latch(FAULT_OVERTEMPERATURE, s.step);
        }
    }

    fn count(&mut self, bits: u32) {
        for (k, c) in self.event_counts.iter_mut().enumerate() {
            if bits & (1 << (8 + k)) != 0 {
                *c += 1;
            }
        }
    }

    pub fn filter(&mut self, s: &State, wanted: &[f64], dt: f64) -> Output {
        self.check_state(s);
        let mut bits = self.latched;
        if self.latched != 0 {
            return Output {
                targets: self.last.clone(),
                torque_enable: false,
                bits,
            };
        }
        if wanted.len() != self.last.len() || !wanted.iter().all(|v| v.is_finite()) {
            bits |= EVENT_NONFINITE_TARGET;
            self.count(bits);
            return Output {
                targets: self.last.clone(),
                torque_enable: true,
                bits,
            };
        }
        let l = self.limits;
        let derate = s.bus_voltage < l.voltage_derate || s.max_temperature > l.temperature_derate;
        let fraction = match (l.torque_fraction, derate) {
            (f, true) => Some(0.5 * f.unwrap_or(1.0)),
            (f, false) => f,
        };
        if derate {
            bits |= EVENT_DERATE;
        }
        let max_step = l.max_target_speed * dt;
        let mut out = Vec::with_capacity(wanted.len());
        for (i, &w) in wanted.iter().enumerate() {
            let mut c = w.clamp(self.lower[i], self.upper[i]);
            if c != w {
                bits |= EVENT_POSITION_CLAMP;
            }
            let step = (c - self.last[i]).clamp(-max_step, max_step);
            if step != c - self.last[i] {
                bits |= EVENT_SLEW_LIMIT;
            }
            c = self.last[i] + step;
            if let Some(f) = fraction {
                let e = f * self.full_error[i];
                let t = c.clamp(s.qpos[i] - e, s.qpos[i] + e);
                if t != c {
                    bits |= EVENT_TORQUE_CLAMP;
                }
                c = t;
            }
            out.push(c);
        }
        self.count(bits);
        self.last.clone_from(&out);
        Output {
            targets: out,
            torque_enable: true,
            bits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::JointSpec;

    fn hello() -> Hello {
        Hello {
            control_dt: 0.02,
            joints: vec![
                JointSpec {
                    lower: -1.0,
                    upper: 1.0,
                    default: 0.0,
                    kp: 5.0,
                    torque_limit: 0.52,
                },
                JointSpec {
                    lower: -0.1,
                    upper: 2.2,
                    default: 0.6,
                    kp: 5.0,
                    torque_limit: 0.52,
                },
            ],
        }
    }

    fn state(step: u64) -> State {
        State {
            step,
            time: step as f64 * 0.02,
            quat: [1.0, 0.0, 0.0, 0.0],
            gyro: [0.0; 3],
            bus_voltage: 7.4,
            max_temperature: 35.0,
            qpos: vec![0.0, 0.6],
            qvel: vec![0.0, 0.0],
            estop: false,
        }
    }

    #[test]
    fn passes_nominal_targets() {
        let mut s = Supervisor::new(&hello(), SafetyLimits::default());
        let o = s.filter(&state(1), &[0.1, 0.65], 0.02);
        assert_eq!(
            o,
            Output {
                targets: vec![0.1, 0.65],
                torque_enable: true,
                bits: 0
            }
        );
    }

    #[test]
    fn slew_and_position_limits() {
        let mut s = Supervisor::new(&hello(), SafetyLimits::default());
        let o = s.filter(&state(1), &[5.0, 0.6], 0.02);
        assert!(o.bits & EVENT_POSITION_CLAMP != 0 && o.bits & EVENT_SLEW_LIMIT != 0);
        assert!((o.targets[0] - 0.2).abs() < 1e-12, "10 rad/s x 0.02 s");
        let mut o = o;
        for k in 2..20 {
            o = s.filter(&state(k), &[5.0, 0.6], 0.02);
        }
        assert!((o.targets[0] - 0.95).abs() < 1e-12, "range 1.0 minus margin 0.05");
    }

    #[test]
    fn nonfinite_target_holds_last() {
        let mut s = Supervisor::new(&hello(), SafetyLimits::default());
        s.filter(&state(1), &[0.1, 0.6], 0.02);
        let o = s.filter(&state(2), &[f64::NAN, 0.6], 0.02);
        assert_eq!(o.targets, vec![0.1, 0.6]);
        assert!(o.torque_enable && o.bits & EVENT_NONFINITE_TARGET != 0);
        assert_eq!(s.latched, 0, "not latching");
    }

    #[test]
    fn tilt_overspeed_and_bad_state_latch_torque_off() {
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let mut tilted = state(3);
        tilted.quat = [h, h, 0.0, 0.0];
        let mut s = Supervisor::new(&hello(), SafetyLimits::default());
        let o = s.filter(&tilted, &[0.0, 0.6], 0.02);
        assert!(!o.torque_enable && o.bits & FAULT_TILT != 0);
        assert_eq!(s.first_fault_step[0], Some(3));
        // Latched: a good state later does not re-enable torque.
        assert!(!s.filter(&state(4), &[0.0, 0.6], 0.02).torque_enable);

        let mut fast = state(1);
        fast.qvel[1] = 25.0;
        let mut s = Supervisor::new(&hello(), SafetyLimits::default());
        assert_eq!(s.filter(&fast, &[0.0, 0.6], 0.02).bits & LATCHING, FAULT_OVERSPEED);

        let mut nan = state(1);
        nan.qpos[0] = f64::NAN;
        let mut s = Supervisor::new(&hello(), SafetyLimits::default());
        assert_eq!(s.filter(&nan, &[0.0, 0.6], 0.02).bits & LATCHING, FAULT_NONFINITE_STATE);
    }

    #[test]
    fn estop_voltage_and_temperature_latch() {
        type Edit = fn(&mut State);
        let cases: [(Edit, u32); 3] = [
            (|s| s.estop = true, FAULT_ESTOP),
            (|s| s.bus_voltage = 6.3, FAULT_UNDERVOLTAGE),
            (|s| s.max_temperature = 69.0, FAULT_OVERTEMPERATURE),
        ];
        for (edit, fault) in cases {
            let mut s = Supervisor::new(&hello(), SafetyLimits::default());
            let mut st = state(5);
            edit(&mut st);
            let o = s.filter(&st, &[0.0, 0.6], 0.02);
            assert!(!o.torque_enable);
            assert_eq!(o.bits & LATCHING, fault);
            assert_eq!(s.first_fault_step[fault.trailing_zeros() as usize], Some(5));
        }
    }

    #[test]
    fn torque_budget_and_derate() {
        // full_error = 0.52 / 5 = 0.104 rad; fraction 0.5 -> 0.052 rad around q.
        let limits = SafetyLimits {
            torque_fraction: Some(0.5),
            ..SafetyLimits::default()
        };
        let mut s = Supervisor::new(&hello(), limits);
        let o = s.filter(&state(1), &[0.15, 0.6], 0.02);
        assert!(o.bits & EVENT_TORQUE_CLAMP != 0);
        assert!((o.targets[0] - 0.052).abs() < 1e-12);
        // No fraction: no torque clamp until the voltage sags below the derate level.
        let mut s = Supervisor::new(&hello(), SafetyLimits::default());
        assert_eq!(s.filter(&state(1), &[0.15, 0.6], 0.02).bits, 0);
        let mut low = state(2);
        low.bus_voltage = 6.6;
        let o = s.filter(&low, &[0.15, 0.6], 0.02);
        assert!(o.torque_enable && o.bits & EVENT_DERATE != 0 && o.bits & EVENT_TORQUE_CLAMP != 0);
        assert!((o.targets[0] - 0.052).abs() < 1e-12, "half of 0.104 around q = 0");
        assert_eq!(s.event_counts[3..], [1, 1]);
    }
}
