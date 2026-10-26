//! Safety supervisor model (S-2 in docs/security/SECURITY_AND_SAFETY.md).
//!
//! Runs every control period between the controller and the actuators. On the
//! real robot this logic belongs on the low-level controller, separate from the
//! high-level computer; here it is the executable model that the sim bridge
//! tests. Not modelled yet (no sensors in sim): temperature and bus voltage.
//!
//! Per period, in order:
//! 1. state checks: non-finite state, tilt, joint overspeed -> latching fault;
//! 2. latched fault -> torque off (the servos go limp; damping remains);
//! 3. non-finite target -> hold the last safe targets this period;
//! 4. clamp targets inside the joint range minus a margin;
//! 5. slew-limit targets to the servo speed.

use crate::protocol::{Hello, State};
use crate::tilt_from_quat;

/// Latching faults: once set, torque stays off until the runtime restarts.
pub const FAULT_TILT: u32 = 1 << 0;
pub const FAULT_OVERSPEED: u32 = 1 << 1;
pub const FAULT_NONFINITE_STATE: u32 = 1 << 2;
pub const FAULT_STATE_TIMEOUT: u32 = 1 << 3;
pub const LATCHING: u32 = FAULT_TILT | FAULT_OVERSPEED | FAULT_NONFINITE_STATE | FAULT_STATE_TIMEOUT;
/// Per-period events (not latching): what the supervisor changed this period.
pub const EVENT_NONFINITE_TARGET: u32 = 1 << 8;
pub const EVENT_POSITION_CLAMP: u32 = 1 << 9;
pub const EVENT_SLEW_LIMIT: u32 = 1 << 10;

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
}

impl Default for SafetyLimits {
    fn default() -> Self {
        SafetyLimits {
            position_margin: 0.05,
            max_target_speed: 10.0, // XL330/XC330 class, WP4-03
            max_joint_speed: 20.0,
            max_tilt: 45f64.to_radians(),
        }
    }
}

#[derive(Debug)]
pub struct Supervisor {
    limits: SafetyLimits,
    lower: Vec<f64>,
    upper: Vec<f64>,
    last: Vec<f64>,
    pub latched: u32,
    /// Control step at which each latching fault first fired (index = bit).
    pub first_fault_step: [Option<u64>; 4],
    pub event_counts: [u64; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    pub targets: Vec<f64>,
    pub torque_enable: bool,
    pub bits: u32,
}

impl Supervisor {
    pub fn new(hello: &Hello, limits: SafetyLimits) -> Self {
        let lower = hello.joints.iter().map(|j| j.lower + limits.position_margin).collect();
        let upper = hello.joints.iter().map(|j| j.upper - limits.position_margin).collect();
        let last = hello.joints.iter().map(|j| j.default).collect();
        Supervisor {
            limits,
            lower,
            upper,
            last,
            latched: 0,
            first_fault_step: [None; 4],
            event_counts: [0; 3],
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
        if !all.clone().all(|v| v.is_finite()) || s.qpos.len() != self.last.len() {
            self.latch(FAULT_NONFINITE_STATE, s.step);
            return;
        }
        if tilt_from_quat(s.quat) > self.limits.max_tilt {
            self.latch(FAULT_TILT, s.step);
        }
        if s.qvel.iter().any(|v| v.abs() > self.limits.max_joint_speed) {
            self.latch(FAULT_OVERSPEED, s.step);
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
            self.event_counts[0] += 1;
            bits |= EVENT_NONFINITE_TARGET;
            return Output {
                targets: self.last.clone(),
                torque_enable: true,
                bits,
            };
        }
        let max_step = self.limits.max_target_speed * dt;
        let mut out = Vec::with_capacity(wanted.len());
        for (i, &w) in wanted.iter().enumerate() {
            let c = w.clamp(self.lower[i], self.upper[i]);
            if c != w {
                bits |= EVENT_POSITION_CLAMP;
            }
            let step = (c - self.last[i]).clamp(-max_step, max_step);
            if step != c - self.last[i] {
                bits |= EVENT_SLEW_LIMIT;
            }
            out.push(self.last[i] + step);
        }
        if bits & EVENT_POSITION_CLAMP != 0 {
            self.event_counts[1] += 1;
        }
        if bits & EVENT_SLEW_LIMIT != 0 {
            self.event_counts[2] += 1;
        }
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
                },
                JointSpec {
                    lower: -0.1,
                    upper: 2.2,
                    default: 0.6,
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
            qpos: vec![0.0, 0.6],
            qvel: vec![0.0, 0.0],
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
}
