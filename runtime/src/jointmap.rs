//! Joint map and calibration: sim joint angles <-> servo angles (WP5-00).
//!
//! The policy and the safety supervisor work in the sim's joint frame (actuator
//! order of `biped_v1.xml`, 0 rad = the MJCF zero). A real servo has its own id,
//! may turn the other way, and its centre is not the joint zero after assembly.
//! One line per joint, in actuator order:
//!
//! ```text
//! # name               id  sign  offset_rad
//! left_hip_yaw          1   +1    0.000
//! ```
//!
//! `servo = sign * sim + offset`, and back `sim = sign * (servo - offset)`.
//! The offset is measured on the bench (WP5 single-joint calibration): with the
//! joint held at its sim zero, the servo angle read is the offset. Nothing here
//! talks to a device; the servo codecs (`sts`, `dxl`) take the mapped angles.

/// Actuator order of `simulation/custom-biped/biped_v1.xml`.
pub const BIPED_V1_JOINTS: [&str; 10] = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
];

/// Largest accepted calibration offset (rad); a bigger one means a wrong horn
/// spline position or a wrong id, not a calibration.
pub const MAX_OFFSET: f64 = 0.8;

#[derive(Debug, Clone, PartialEq)]
pub struct Joint {
    pub name: String,
    pub id: u8,
    pub sign: f64,
    pub offset: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JointMap {
    pub joints: Vec<Joint>,
}

impl JointMap {
    /// Parses the text format above; `expected` is the actuator order of the model.
    pub fn parse(text: &str, expected: &[&str]) -> Result<Self, String> {
        let mut joints = Vec::new();
        for (k, raw) in text.lines().enumerate() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split_whitespace().collect();
            let at = |what: &str| format!("line {}: {what}", k + 1);
            if f.len() != 4 {
                return Err(at("expected `name id sign offset_rad`"));
            }
            let id: u8 = f[1].parse().map_err(|_| at("id must be 0..=253"))?;
            if id > 253 {
                return Err(at("id must be 0..=253 (254 is broadcast)"));
            }
            let sign = match f[2] {
                "+1" | "1" => 1.0,
                "-1" => -1.0,
                _ => return Err(at("sign must be +1 or -1")),
            };
            let offset: f64 = f[3].parse().map_err(|_| at("offset must be a number"))?;
            if !offset.is_finite() || offset.abs() > MAX_OFFSET {
                return Err(at("offset must be finite and within +-0.8 rad"));
            }
            joints.push(Joint {
                name: f[0].to_string(),
                id,
                sign,
                offset,
            });
        }
        let names: Vec<&str> = joints.iter().map(|j| j.name.as_str()).collect();
        if names != expected {
            return Err(format!("joints must be, in this order: {expected:?}; got {names:?}"));
        }
        for (a, ja) in joints.iter().enumerate() {
            if joints[..a].iter().any(|jb| jb.id == ja.id) {
                return Err(format!("servo id {} is used twice", ja.id));
            }
        }
        Ok(JointMap { joints })
    }

    pub fn joint(&self, name: &str) -> Option<(usize, &Joint)> {
        self.joints.iter().enumerate().find(|(_, j)| j.name == name)
    }

    pub fn ids(&self) -> Vec<u8> {
        self.joints.iter().map(|j| j.id).collect()
    }

    /// Sim joint angles -> servo angles (rad), same order.
    pub fn to_servo(&self, sim: &[f64]) -> Vec<f64> {
        assert_eq!(sim.len(), self.joints.len());
        self.joints
            .iter()
            .zip(sim)
            .map(|(j, q)| j.sign * q + j.offset)
            .collect()
    }

    /// Servo angles (rad) -> sim joint angles.
    pub fn to_sim(&self, servo: &[f64]) -> Vec<f64> {
        assert_eq!(servo.len(), self.joints.len());
        self.joints
            .iter()
            .zip(servo)
            .map(|(j, s)| j.sign * (s - j.offset))
            .collect()
    }

    /// One joint: sim angle -> servo angle.
    pub fn to_servo_one(&self, k: usize, sim: f64) -> f64 {
        let j = &self.joints[k];
        j.sign * sim + j.offset
    }

    /// One joint: servo angle -> sim angle.
    pub fn to_sim_one(&self, k: usize, servo: f64) -> f64 {
        let j = &self.joints[k];
        j.sign * (servo - j.offset)
    }

    /// Servo velocities -> sim joint velocities (the offset drops out).
    pub fn vel_to_sim(&self, servo: &[f64]) -> Vec<f64> {
        assert_eq!(servo.len(), self.joints.len());
        self.joints.iter().zip(servo).map(|(j, w)| j.sign * w).collect()
    }
}

/// Returns `text` with the offset of joint `name` replaced, keeping the other
/// lines and comments; the result is parsed again, so a bad offset is refused.
pub fn set_offset(text: &str, name: &str, offset: f64, expected: &[&str]) -> Result<String, String> {
    let mut found = false;
    let mut out = String::with_capacity(text.len());
    for raw in text.split_inclusive('\n') {
        let (body, comment) = raw.split_once('#').map_or((raw, None), |(b, c)| (b, Some(c)));
        let f: Vec<&str> = body.split_whitespace().collect();
        if f.len() == 4 && f[0] == name {
            found = true;
            out.push_str(&format!("{:<20} {:>3}  {:>3}  {offset:+.4}", f[0], f[1], f[2]));
            match comment {
                Some(c) => out.push_str(&format!("   #{c}")),
                None if raw.ends_with('\n') => out.push('\n'),
                None => {}
            }
        } else {
            out.push_str(raw);
        }
    }
    if !found {
        return Err(format!("no joint {name:?} in the map"));
    }
    JointMap::parse(&out, expected)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sts;

    const NAMES: [&str; 3] = ["left_hip_yaw", "left_knee", "right_knee"];
    const TEXT: &str = "\
# name        id sign offset
left_hip_yaw   1  +1   0.0
left_knee      4  -1   0.05   # mounted mirrored
right_knee     9   1  -0.02
";

    #[test]
    fn parses_and_round_trips() {
        let m = JointMap::parse(TEXT, &NAMES).unwrap();
        assert_eq!(m.ids(), vec![1, 4, 9]);
        let sim = [0.1, -0.7, 0.7];
        let servo = m.to_servo(&sim);
        assert!((servo[0] - 0.1).abs() < 1e-12);
        assert!((servo[1] - (0.7 + 0.05)).abs() < 1e-12);
        assert!((servo[2] - (0.7 - 0.02)).abs() < 1e-12);
        let back = m.to_sim(&servo);
        for (a, b) in back.iter().zip(sim) {
            assert!((a - b).abs() < 1e-12);
        }
        assert_eq!(m.vel_to_sim(&[1.0, 1.0, 1.0]), vec![1.0, -1.0, 1.0]);
    }

    #[test]
    fn mapped_targets_reach_the_sts_codec() {
        let m = JointMap::parse(TEXT, &NAMES).unwrap();
        let servo = m.to_servo(&[0.0, 0.0, 0.0]);
        let pkt = sts::goal_positions(&m.ids(), &servo).unwrap();
        // The knee's offset moves its goal off the 2048 centre; hip yaw stays there.
        let want = sts::rad_to_ticks(0.05).unwrap().to_le_bytes();
        let knee = pkt.windows(3).any(|w| w == [4, want[0], want[1]]);
        assert!(knee, "{pkt:?}");
    }

    #[test]
    fn example_file_matches_the_model() {
        let text = include_str!("../../hardware/calibration/biped_v1_joints.example.txt");
        let m = JointMap::parse(text, &BIPED_V1_JOINTS).unwrap();
        assert_eq!(m.ids(), (1..=10).collect::<Vec<u8>>());
    }

    #[test]
    fn offsets_are_written_back_in_place() {
        let t = set_offset(TEXT, "left_knee", -0.1234, &NAMES).unwrap();
        let m = JointMap::parse(&t, &NAMES).unwrap();
        assert_eq!(m.joint("left_knee").unwrap().1.offset, -0.1234);
        assert_eq!(m.joint("left_knee").unwrap().0, 1);
        assert!(t.contains("# mounted mirrored"), "{t}");
        assert_eq!(m.joints[2], JointMap::parse(TEXT, &NAMES).unwrap().joints[2]);
        assert!(set_offset(TEXT, "left_knee", 0.9, &NAMES)
            .unwrap_err()
            .contains("offset"));
        assert!(set_offset(TEXT, "neck", 0.0, &NAMES).unwrap_err().contains("neck"));
        assert!(m.joint("neck").is_none());
    }

    #[test]
    fn rejects_bad_files() {
        let bad = |t: &str| JointMap::parse(t, &NAMES).unwrap_err();
        assert!(bad("left_hip_yaw 1 +1\n").contains("expected"));
        assert!(bad(&TEXT.replace("+1   0.0", "+2   0.0")).contains("sign"));
        assert!(bad(&TEXT.replace("0.05", "1.2")).contains("offset"));
        assert!(bad(&TEXT.replace("0.05", "nan")).contains("offset"));
        assert!(bad(&TEXT.replace(" 9 ", " 4 ")).contains("twice"));
        assert!(bad(&TEXT.replace(" 9 ", " 254 ")).contains("broadcast"));
        assert!(bad(&TEXT.replace("right_knee", "rite_knee")).contains("order"));
        assert!(bad("").contains("order"));
    }
}
