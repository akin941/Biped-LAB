//! The DS3218MG build's joint table (G4.5): ranges from the model
//! (`hardware/calibration/biped_ds3218.hello`, checked by
//! `tests/unit/test_firmware_joints.py`) and the pulse mapping per joint.
//!
//! Joint order is the runtime's: left hip yaw, hip roll, hip pitch, knee,
//! ankle pitch, then the same on the right.
//!
//! The DS3218MG turns ±π/2 around its 1500 us centre. The knee's range
//! (−0.1 … 2.2 rad) does not fit around zero, so its servo is mounted with the
//! centre at mid-range: `offset = −KNEE_CENTER`. Hip pitch (±1.6 rad) is 0.03
//! rad wider than the servo and stops at its end stop there. Signs (mirrored
//! sides) and the per-servo trim are measured on the bench (stage 3) and change
//! here, with the owner's approval like any supervisor limit (ADR-0015).

use crate::servo::JointPwm;
use crate::N_JOINTS;

/// Model joint ranges (rad, sim frame): (lower, upper).
pub const RANGES: [(f32, f32); N_JOINTS] = [
    (-0.5, 0.5),
    (-0.4, 0.4),
    (-1.6, 1.6),
    (-0.1, 2.2),
    (-1.2, 1.2),
    (-0.5, 0.5),
    (-0.4, 0.4),
    (-1.6, 1.6),
    (-0.1, 2.2),
    (-1.2, 1.2),
];

/// Knee angle (rad) at the servo's centre pulse.
pub const KNEE_CENTER: f32 = 1.05;

/// Pulse mapping for every joint of the DS3218MG build.
pub fn biped_ds3218() -> [JointPwm; N_JOINTS] {
    let mut joints = [JointPwm::ds3218(0.0, 0.0); N_JOINTS];
    for (k, (j, (lo, hi))) in joints.iter_mut().zip(RANGES).enumerate() {
        *j = JointPwm::ds3218(lo, hi);
        if k % 5 == 3 {
            j.offset = -KNEE_CENTER;
        }
    }
    joints
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_joint_range_reaches_its_pulse_limits_inside_the_servo_travel() {
        let joints = biped_ds3218();
        for (k, j) in joints.iter().enumerate() {
            let (lo, hi) = (j.pulse_us(j.lower), j.pulse_us(j.upper));
            assert!(lo < 1500 && hi > 1500, "joint {k}: {lo}..{hi}");
            assert!((500..=2500).contains(&lo) && (500..=2500).contains(&hi));
        }
        // The knee's whole range maps strictly inside the end stops.
        let knee = joints[3];
        assert!(knee.pulse_us(-0.1) > 500 && knee.pulse_us(2.2) < 2500);
        assert_eq!(knee.pulse_us(KNEE_CENTER), 1500);
        assert_eq!(joints[8], knee);
    }
}
