//! Joint angle to servo pulse width (ADR-0015).
//!
//! `pulse = center + sign * us_per_rad * (q + offset)`, clamped to
//! [`min_us`, `max_us`], with `q` the joint angle in the sim frame (rad), the
//! same frame as the runtime's targets. `sign` and `offset` play the roles of
//! the STS joint map's direction and offset (`runtime/src/jointmap.rs`) and are
//! set during calibration; `lower` / `upper` are the joint's range in the sim
//! frame (the model's range), which the supervisor enforces.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointPwm {
    pub center_us: f32,
    pub us_per_rad: f32,
    pub sign: f32,
    pub offset: f32,
    pub min_us: u16,
    pub max_us: u16,
    pub lower: f32,
    pub upper: f32,
}

impl JointPwm {
    /// DSServo DS3218MG (180 deg version): 500-2500 us over 180 deg, centred
    /// at 1500 us (seller datasheet; check the end stops on the bench).
    pub fn ds3218(lower: f32, upper: f32) -> Self {
        JointPwm {
            center_us: 1500.0,
            us_per_rad: 2000.0 / core::f32::consts::PI,
            sign: 1.0,
            offset: 0.0,
            min_us: 500,
            max_us: 2500,
            lower,
            upper,
        }
    }

    pub fn pulse_us(&self, q: f32) -> u16 {
        let us = self.center_us + self.sign * self.us_per_rad * (q + self.offset);
        let us = if us.is_finite() { us } else { self.center_us };
        // Round to the nearest microsecond without std.
        let us = (us + 0.5) as i32;
        us.clamp(self.min_us as i32, self.max_us as i32) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ds3218_mapping() {
        let j = JointPwm::ds3218(-1.5, 1.5);
        assert_eq!(j.pulse_us(0.0), 1500);
        assert_eq!(j.pulse_us(core::f32::consts::FRAC_PI_2), 2500);
        assert_eq!(j.pulse_us(-core::f32::consts::FRAC_PI_2), 500);
        assert_eq!(j.pulse_us(3.0), 2500); // clamped to the end stop
        assert_eq!(j.pulse_us(f32::NAN), 1500);
        let r = JointPwm {
            sign: -1.0,
            offset: 0.1,
            ..j
        };
        assert_eq!(
            r.pulse_us(0.0),
            (1500.0 - 0.1 * 2000.0 / core::f32::consts::PI + 0.5) as u16
        );
    }
}
