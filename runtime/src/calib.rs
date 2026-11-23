//! Single-joint calibration steps for Phase 5 stage 3 (WP5-04), used by
//! `biped-bench offset` and `biped-bench move`.
//!
//! - `measure_offset`: with the joint's servo limp and the joint held at its
//!   sim zero by hand (or a jig), the servo angle read is the joint's offset.
//!   Refuses while the servo has torque on.
//! - `move_joint`: one joint to a sim angle inside its model range, with the
//!   servo's torque limit register set to a small fraction first, held for a
//!   while and sampled, then torque off again whatever happened. The readback
//!   goes through the same map, so it shows whether the servo followed the
//!   goal, not whether the sign is right: the sign is checked by eye (a
//!   +0.1 rad target must turn the joint the way it turns in the sim viewer).
//!
//! Nothing here has run on a real servo (Phase 5 stage 3, `env: hw`, owner's
//! approval). The torque limit register address is from the STS memory table
//! and is checked on the first servo.

use crate::jointmap::JointMap;
use crate::protocol::Hello;
use crate::sts::{self, addr};
use crate::stsbus::{Bus, Port};
use std::time::{Duration, Instant};

/// Largest torque limit the bench tool accepts (fraction of stall torque).
pub const MAX_BENCH_TORQUE: f64 = 0.5;

#[derive(Debug, Clone, PartialEq)]
pub struct Offset {
    pub id: u8,
    /// Servo angle read with the joint at its sim zero (rad): the new offset.
    pub offset: f64,
    /// Offset in the map before.
    pub previous: f64,
}

pub fn measure_offset<P: Port>(bus: &mut Bus<P>, map: &JointMap, joint: &str) -> Result<Offset, String> {
    let (_, j) = map.joint(joint).ok_or(format!("no joint {joint:?} in the map"))?;
    let torque = bus.read(j.id, addr::TORQUE_ENABLE, 1).map_err(|e| e.to_string())?;
    if torque[0] != 0 {
        return Err(format!("servo {} has torque on; switch it off before measuring", j.id));
    }
    let st = bus.read_state(&[j.id]).map_err(|e| e.to_string())?;
    Ok(Offset {
        id: j.id,
        offset: st[0].pos_rad,
        previous: j.offset,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub t: f64,
    /// Joint angle in the sim frame (rad).
    pub sim_rad: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Move {
    pub id: u8,
    pub start: f64,
    pub target: f64,
    pub samples: Vec<Sample>,
}

impl Move {
    pub fn end(&self) -> f64 {
        self.samples.last().map_or(self.start, |s| s.sim_rad)
    }

    /// True when the joint moved at least `min_rad` towards the target, i.e.
    /// the servo followed the goal. Not a sign check (see the module doc).
    pub fn moved_towards_target(&self, min_rad: f64) -> bool {
        let want = self.target - self.start;
        let got = self.end() - self.start;
        want.abs() >= min_rad && got * want.signum() >= min_rad
    }
}

pub struct MoveSpec<'a> {
    pub joint: &'a str,
    /// Target in the sim frame (rad); must be inside the model's joint range.
    pub target: f64,
    /// Servo torque limit while moving, fraction of stall torque.
    pub torque: f64,
    pub hold: Duration,
    pub sample_period: Duration,
}

pub fn move_joint<P: Port>(bus: &mut Bus<P>, map: &JointMap, hello: &Hello, spec: &MoveSpec) -> Result<Move, String> {
    let (k, j) = map
        .joint(spec.joint)
        .ok_or(format!("no joint {:?} in the map", spec.joint))?;
    let range = hello.joints.get(k).ok_or("hello file has fewer joints than the map")?;
    if !(spec.target.is_finite() && range.lower <= spec.target && spec.target <= range.upper) {
        return Err(format!(
            "target {} outside the joint range [{}, {}]",
            spec.target, range.lower, range.upper
        ));
    }
    if !(spec.torque > 0.0 && spec.torque <= MAX_BENCH_TORQUE) {
        return Err(format!("torque limit must be in (0, {MAX_BENCH_TORQUE}]"));
    }
    let id = j.id;
    let read_sim = |bus: &mut Bus<P>| -> Result<f64, String> {
        let st = bus.read_state(&[id]).map_err(|e| e.to_string())?;
        Ok(map.to_sim_one(k, st[0].pos_rad))
    };
    let start = read_sim(bus)?;
    let limit = ((spec.torque * 1000.0).round() as u16).max(1);
    bus.write(id, addr::TORQUE_LIMIT, &limit.to_le_bytes())
        .map_err(|e| e.to_string())?;
    let goal = sts::rad_to_ticks(map.to_servo_one(k, spec.target)).ok_or("goal outside one servo turn")?;
    let result = (|| -> Result<Vec<Sample>, String> {
        bus.write(id, addr::GOAL_POSITION, &goal.to_le_bytes())
            .map_err(|e| e.to_string())?;
        bus.write(id, addr::TORQUE_ENABLE, &[1]).map_err(|e| e.to_string())?;
        let t0 = Instant::now();
        let mut samples = Vec::new();
        while t0.elapsed() < spec.hold {
            std::thread::sleep(spec.sample_period);
            samples.push(Sample {
                t: t0.elapsed().as_secs_f64(),
                sim_rad: read_sim(bus)?,
            });
        }
        Ok(samples)
    })();
    // Limp again whatever happened, then report.
    let off = bus.write(id, addr::TORQUE_ENABLE, &[0]).map_err(|e| e.to_string());
    let samples = result?;
    off?;
    Ok(Move {
        id,
        start,
        target: spec.target,
        samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jointmap::BIPED_V1_JOINTS;
    use crate::stsbus::{Fault, SimPort};

    const MAP: &str = include_str!("../../hardware/calibration/biped_v1_joints.example.txt");
    const HELLO: &[u8] = include_bytes!("../../hardware/calibration/biped_v1_sts3215.hello");

    fn setup() -> (Bus<SimPort>, JointMap, Hello) {
        let text = MAP.replace("left_knee             4   +1    0.0", "left_knee 4 -1 0.1");
        let map = JointMap::parse(&text, &BIPED_V1_JOINTS).unwrap();
        let mut sim = sts::SimBus::new(&map.ids());
        for (_, t) in sim.tables.iter_mut() {
            t[56..58].copy_from_slice(&2048u16.to_le_bytes());
        }
        let hello = Hello::read(&mut &HELLO[..]).unwrap();
        (Bus::new(SimPort::new(sim)), map, hello)
    }

    fn table(bus: &Bus<SimPort>, id: u8) -> &[u8; 128] {
        &bus.port.bus.tables.iter().find(|(i, _)| *i == id).unwrap().1
    }

    #[test]
    fn offset_is_the_servo_angle_at_sim_zero_and_needs_a_limp_servo() {
        let (mut bus, map, _) = setup();
        let ticks = sts::rad_to_ticks(0.25).unwrap();
        bus.port.bus.tables[3].1[56..58].copy_from_slice(&ticks.to_le_bytes());
        let o = measure_offset(&mut bus, &map, "left_knee").unwrap();
        assert_eq!((o.id, o.previous), (4, 0.1));
        assert!((o.offset - 0.25).abs() <= std::f64::consts::TAU / 4096.0);
        bus.port.bus.tables[3].1[addr::TORQUE_ENABLE as usize] = 1;
        assert!(measure_offset(&mut bus, &map, "left_knee")
            .unwrap_err()
            .contains("torque on"));
        assert!(measure_offset(&mut bus, &map, "neck").is_err());
    }

    fn spec(joint: &str, target: f64, torque: f64) -> MoveSpec<'_> {
        MoveSpec {
            joint,
            target,
            torque,
            hold: Duration::from_millis(30),
            sample_period: Duration::from_millis(10),
        }
    }

    #[test]
    fn move_sets_the_torque_limit_goal_and_ends_limp() {
        let (mut bus, map, hello) = setup();
        let m = move_joint(&mut bus, &map, &hello, &spec("left_knee", 0.3, 0.2)).unwrap();
        let t = table(&bus, 4);
        assert_eq!(u16::from_le_bytes([t[48], t[49]]), 200);
        // Mirrored knee: servo goal = -0.3 + 0.1.
        let goal = u16::from_le_bytes([t[42], t[43]]);
        assert_eq!(goal, sts::rad_to_ticks(-0.2).unwrap());
        assert_eq!(t[addr::TORQUE_ENABLE as usize], 0, "limp after the move");
        assert!(!m.samples.is_empty() && m.id == 4);
        // Start: 2048 ticks = servo 0 = sim 0.1; the ideal sim servo reaches 0.3.
        assert!((m.start - 0.1).abs() < 1e-9 && m.moved_towards_target(0.1));
        assert!((m.end() - 0.3).abs() <= std::f64::consts::TAU / 4096.0);
    }

    #[test]
    fn move_refuses_targets_outside_the_model_range_and_big_torque() {
        let (mut bus, map, hello) = setup();
        let upper = hello.joints[3].upper;
        assert!(move_joint(&mut bus, &map, &hello, &spec("left_knee", upper + 0.01, 0.2)).is_err());
        assert!(move_joint(&mut bus, &map, &hello, &spec("left_knee", f64::NAN, 0.2)).is_err());
        assert!(move_joint(&mut bus, &map, &hello, &spec("left_knee", 0.1, 0.6)).is_err());
        assert!(move_joint(&mut bus, &map, &hello, &spec("left_knee", 0.1, 0.0)).is_err());
        // Nothing was written by the refused moves.
        assert_eq!(table(&bus, 4)[48], 0);
    }

    #[test]
    fn a_failed_move_still_switches_torque_off() {
        let (mut bus, map, hello) = setup();
        // Start read, torque limit, goal and torque on are answered; every
        // attempt of the first sample read is dropped; torque off goes through.
        bus.port.faults.extend([Fault::Pass; 4]);
        bus.port.faults.extend([Fault::Drop; 3]);
        let e = move_joint(&mut bus, &map, &hello, &spec("left_hip_yaw", 0.1, 0.2)).unwrap_err();
        assert!(e.contains("no reply"), "{e}");
        assert_eq!(table(&bus, 1)[addr::TORQUE_ENABLE as usize], 0);
    }

    #[test]
    fn sign_check() {
        let m = Move {
            id: 1,
            start: 0.0,
            target: 0.1,
            samples: vec![Sample { t: 0.1, sim_rad: 0.09 }],
        };
        assert!(m.moved_towards_target(0.05));
        let wrong = Move {
            samples: vec![Sample { t: 0.1, sim_rad: -0.09 }],
            ..m.clone()
        };
        assert!(!wrong.moved_towards_target(0.05));
    }
}
