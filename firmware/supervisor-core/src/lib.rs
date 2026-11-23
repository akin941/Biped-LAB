//! Core of the RP2040 safety supervisor (ADR-0015, WP5-08).
//!
//! The DS3218MG build drives ten PWM hobby servos. ADR-0006 / S-2 put the
//! safety supervisor on separate hardware from the policy computer, so the Pi
//! sends joint targets over UART to an RP2040, and the RP2040 alone turns them
//! into servo pulses. This crate is everything on that MCU that does not touch
//! a peripheral, so it is unit-tested on the host:
//!
//! - [`frame`]: the UART frames (targets in, status out) with a CRC.
//! - [`servo`]: joint angle (sim frame, rad) to pulse width (us) per joint.
//! - [`Supervisor`]: the checks between a received frame and the pulses: joint
//!   limits, a slew limit, the command watchdog (hold, then pulses off),
//!   battery voltage, the software stop and a latched fault state.
//!
//! What a PWM servo cannot do is stated, not hidden: it has no torque limit,
//! current or temperature readback, and a digital servo may keep holding its
//! last position when its pulses stop. "Pulses off" is therefore a best
//! effort; the hardware e-stop (S-3) that cuts servo power is the real stop.

#![cfg_attr(not(test), no_std)]

pub mod frame;
pub mod joints;
pub mod servo;

use servo::JointPwm;

/// Joint count of the biped (two legs × five joints).
pub const N_JOINTS: usize = 10;

/// Fault and event bits, sent in every status frame.
pub mod bits {
    /// Latched: no valid targets frame for `timeout_off_ms` (pulses off).
    pub const WATCHDOG: u16 = 1 << 0;
    /// Latched: battery below `voltage_stop` (pulses off).
    pub const UNDERVOLTAGE: u16 = 1 << 1;
    /// Latched: the Pi asked for a stop (frame flag) (pulses off).
    pub const SOFT_STOP: u16 = 1 << 2;
    /// Event: no fresh frame for `timeout_hold_ms`, holding the last pulses.
    pub const HOLD: u16 = 1 << 8;
    /// Event: a target was outside its joint range and was clamped.
    pub const POSITION_CLAMP: u16 = 1 << 9;
    /// Event: a target moved faster than `max_speed` and was slowed.
    pub const SLEW_LIMIT: u16 = 1 << 10;
    /// Event: a frame was dropped (bad CRC, length or joint count).
    pub const BAD_FRAME: u16 = 1 << 11;
    /// Event: battery below `voltage_derate` (warning only: a PWM servo has
    /// no torque limit to lower).
    pub const LOW_VOLTAGE: u16 = 1 << 12;
    pub const LATCHED: u16 = WATCHDOG | UNDERVOLTAGE | SOFT_STOP;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
    /// Hold the last pulses after this long without a valid frame.
    pub timeout_hold_ms: u32,
    /// Then stop the pulses and latch WATCHDOG.
    pub timeout_off_ms: u32,
    /// Largest target speed (rad/s), as the runtime's supervisor (10 rad/s).
    pub max_speed: f32,
    /// Battery (2S) warning and stop levels (V).
    pub voltage_derate: f32,
    pub voltage_stop: f32,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            timeout_hold_ms: 60, // 3 periods at 50 Hz
            timeout_off_ms: 500, // as the runtime's state watchdog
            max_speed: 10.0,
            voltage_derate: 7.0,
            voltage_stop: 6.6,
        }
    }
}

/// What the PWM outputs should do now.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Output {
    /// Pulse width per joint in us, or `None` = no pulses (limp attempt).
    pub pulses_us: Option<[u16; N_JOINTS]>,
    pub bits: u16,
}

pub struct Supervisor {
    pub joints: [JointPwm; N_JOINTS],
    pub limits: Limits,
    latched: u16,
    /// Last targets sent to the servos (sim frame, rad); `None` before the
    /// first enabled frame.
    last: Option<[f32; N_JOINTS]>,
    /// Time of the last valid frame (ms, MCU clock).
    last_frame_ms: Option<u32>,
    /// Events seen since the last status frame.
    events: u16,
    torque: bool,
    /// Sequence number of the last valid targets frame (echoed in status).
    pub last_seq: u8,
}

impl Supervisor {
    pub fn new(joints: [JointPwm; N_JOINTS], limits: Limits) -> Self {
        Supervisor {
            joints,
            limits,
            latched: 0,
            last: None,
            last_frame_ms: None,
            events: 0,
            torque: false,
            last_seq: 0,
        }
    }

    pub fn latched(&self) -> u16 {
        self.latched
    }

    /// A frame failed to decode (it never reaches the servos).
    pub fn bad_frame(&mut self) {
        self.events |= bits::BAD_FRAME;
    }

    /// A decoded targets frame arrived at `now_ms`.
    pub fn on_targets(&mut self, f: &frame::Targets, now_ms: u32) {
        if f.stop {
            self.latched |= bits::SOFT_STOP;
        }
        let dt = match self.last_frame_ms {
            Some(t) => now_ms.wrapping_sub(t).max(1) as f32 * 1e-3,
            None => 0.02,
        };
        self.last_frame_ms = Some(now_ms);
        self.last_seq = f.seq;
        self.torque = f.torque_enable;
        if f.clear {
            self.clear(); // refused (no-op) while torque is enabled
        }
        if !f.torque_enable || self.latched != 0 {
            return;
        }
        let max_step = self.limits.max_speed * dt;
        let mut out = [0.0f32; N_JOINTS];
        for (i, (o, j)) in out.iter_mut().zip(&self.joints).enumerate() {
            let mut q = f.targets[i];
            if q < j.lower || q > j.upper {
                q = q.clamp(j.lower, j.upper);
                self.events |= bits::POSITION_CLAMP;
            }
            if let Some(prev) = self.last {
                let step = q - prev[i];
                if step.abs() > max_step {
                    q = prev[i] + step.clamp(-max_step, max_step);
                    self.events |= bits::SLEW_LIMIT;
                }
            }
            *o = q;
        }
        self.last = Some(out);
    }

    /// Called every PWM period (and before each status frame) with the
    /// current time and battery voltage; returns what to output and clears
    /// the reported events.
    pub fn tick(&mut self, now_ms: u32, battery_v: f32) -> Output {
        let l = self.limits;
        if battery_v < l.voltage_stop {
            self.latched |= bits::UNDERVOLTAGE;
        } else if battery_v < l.voltage_derate {
            self.events |= bits::LOW_VOLTAGE;
        }
        let silent = self.last_frame_ms.map(|t| now_ms.wrapping_sub(t));
        match silent {
            Some(ms) if ms >= l.timeout_off_ms => self.latched |= bits::WATCHDOG,
            Some(ms) if ms >= l.timeout_hold_ms => self.events |= bits::HOLD,
            _ => {}
        }
        // Before any frame the watchdog counts from power-up.
        if silent.is_none() && now_ms >= l.timeout_off_ms {
            self.latched |= bits::WATCHDOG;
        }
        let bits = self.latched | self.events;
        self.events = 0;
        let pulses_us = match (self.latched, self.torque, self.last) {
            (0, true, Some(q)) => {
                let mut p = [0u16; N_JOINTS];
                for ((pi, j), qi) in p.iter_mut().zip(&self.joints).zip(q) {
                    *pi = j.pulse_us(qi);
                }
                Some(p)
            }
            _ => None,
        };
        Output { pulses_us, bits }
    }

    /// Clears latched faults; only accepted with torque off (the Pi must
    /// send a frame with torque disabled first), and the next enabled frame
    /// starts without a slew reference.
    pub fn clear(&mut self) -> bool {
        if self.torque {
            return false;
        }
        self.latched = 0;
        self.last = None;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frame::Targets;

    fn sup() -> Supervisor {
        Supervisor::new([JointPwm::ds3218(-1.0, 1.0); N_JOINTS], Limits::default())
    }

    fn targets(q: f32, enable: bool) -> Targets {
        Targets {
            seq: 0,
            torque_enable: enable,
            stop: false,
            clear: false,
            targets: [q; N_JOINTS],
        }
    }

    #[test]
    fn no_pulses_until_an_enabled_frame() {
        let mut s = sup();
        assert_eq!(s.tick(0, 7.4).pulses_us, None);
        s.on_targets(&targets(0.0, false), 10);
        assert_eq!(s.tick(10, 7.4).pulses_us, None);
        s.on_targets(&targets(0.0, true), 30);
        let p = s.tick(30, 7.4).pulses_us.unwrap();
        assert!(p.iter().all(|&v| v == 1500));
    }

    #[test]
    fn clamps_and_slews() {
        let mut s = sup();
        s.on_targets(&targets(0.0, true), 0);
        s.tick(0, 7.4);
        s.on_targets(&targets(5.0, true), 20); // beyond the 1.0 rad limit, and fast
        let o = s.tick(20, 7.4);
        assert_ne!(o.bits & bits::POSITION_CLAMP, 0);
        assert_ne!(o.bits & bits::SLEW_LIMIT, 0);
        // 10 rad/s for 20 ms = 0.2 rad.
        let want = JointPwm::ds3218(-1.0, 1.0).pulse_us(0.2);
        assert!(o.pulses_us.unwrap().iter().all(|&v| v == want));
        // Events are reported once.
        assert_eq!(s.tick(21, 7.4).bits & bits::SLEW_LIMIT, 0);
    }

    #[test]
    fn watchdog_holds_then_latches_off() {
        let mut s = sup();
        s.on_targets(&targets(0.1, true), 0);
        let o = s.tick(70, 7.4);
        assert_ne!(o.bits & bits::HOLD, 0);
        assert!(o.pulses_us.is_some()); // hold the last pulses
        let o = s.tick(500, 7.4);
        assert_ne!(o.bits & bits::WATCHDOG, 0);
        assert_eq!(o.pulses_us, None);
        // A late frame does not restart the servos while latched.
        s.on_targets(&targets(0.1, true), 510);
        assert_eq!(s.tick(510, 7.4).pulses_us, None);
        // Clearing needs torque off first.
        assert!(!s.clear());
        let mut c = targets(0.1, true);
        c.clear = true;
        s.on_targets(&c, 515);
        assert_ne!(s.latched(), 0);
        c.torque_enable = false;
        s.on_targets(&c, 520);
        assert_eq!(s.latched(), 0);
        s.on_targets(&targets(0.1, true), 540);
        assert!(s.tick(540, 7.4).pulses_us.is_some());
    }

    #[test]
    fn watchdog_from_power_up() {
        let mut s = sup();
        assert_eq!(s.tick(499, 7.4).bits & bits::WATCHDOG, 0);
        assert_ne!(s.tick(500, 7.4).bits & bits::WATCHDOG, 0);
    }

    #[test]
    fn voltage_and_soft_stop() {
        let mut s = sup();
        s.on_targets(&targets(0.0, true), 0);
        let o = s.tick(1, 6.9);
        assert_ne!(o.bits & bits::LOW_VOLTAGE, 0);
        assert!(o.pulses_us.is_some());
        let o = s.tick(2, 6.5);
        assert_ne!(o.bits & bits::UNDERVOLTAGE, 0);
        assert_eq!(o.pulses_us, None);
        let mut s = sup();
        let mut t = targets(0.0, true);
        t.stop = true;
        s.on_targets(&t, 0);
        let o = s.tick(1, 7.4);
        assert_ne!(o.bits & bits::SOFT_STOP, 0);
        assert_eq!(o.pulses_us, None);
    }

    #[test]
    fn clock_wraparound_is_not_a_timeout() {
        let mut s = sup();
        s.on_targets(&targets(0.0, true), u32::MAX - 5);
        let o = s.tick(10, 7.4); // 16 ms later across the wrap
        assert_eq!(o.bits & (bits::HOLD | bits::WATCHDOG), 0);
    }
}
