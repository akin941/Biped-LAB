//! Robot-side runtime for biped_v1 (WP4-04, ADR-0006 / ADR-0007).
//!
//! One control period (50 Hz):
//!
//! ```text
//! state ──► command gate ──► controller ──► safety supervisor ──► action
//! ```
//!
//! In simulation the state comes from, and the action goes to, the MuJoCo sim
//! bridge (`bipedlab.runtime_bridge`) over TCP. On hardware the supervisor part
//! moves to the low-level controller (S-2); this crate is its executable model.

pub mod controller;
pub mod gate;
pub mod protocol;
pub mod safety;

/// Tilt of the body z axis from vertical (rad), from a unit quaternion (w, x, y, z).
pub fn tilt_from_quat(q: [f64; 4]) -> f64 {
    let [_, x, y, _] = q;
    // z component of the body z axis in the world frame.
    let zz = 1.0 - 2.0 * (x * x + y * y);
    zz.clamp(-1.0, 1.0).acos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tilt_of_identity_and_quarter_turn() {
        assert!(tilt_from_quat([1.0, 0.0, 0.0, 0.0]).abs() < 1e-12);
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let t = tilt_from_quat([h, h, 0.0, 0.0]); // 90 deg about x
        assert!((t - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
        // Yaw does not tilt.
        assert!(tilt_from_quat([h, 0.0, 0.0, h]).abs() < 1e-9);
    }
}
