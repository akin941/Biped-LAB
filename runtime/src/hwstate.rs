//! Hardware state source and actuator sink (WP5-02): the robot side of the
//! loop that the sim bridge plays in simulation.
//!
//! Each control period `read` fills the same `State` the bridge sends: joint
//! positions and speeds from one STS sync read through the joint map, IMU
//! orientation and gyro, the lowest bus voltage and the hottest servo.
//! Voltage and temperature change slowly, so one servo's are read per period
//! (round robin); all of them are read once at start. `apply` sends an
//! `Action`: goals for all joints in one sync write, torque switched only when
//! the action's torque flag changes.
//!
//! Nothing here decides safety: the supervisor's action is applied as given.
//! Nothing has run on a real servo yet (Phase 5 stage 2+, `env: hw`).

use crate::jointmap::JointMap;
use crate::protocol::{Action, State};
use crate::stsbus::{Bus, BusError, Port};
use std::time::Instant;

/// Body orientation (w, x, y, z) and gyro (rad/s, body frame).
pub trait Imu {
    fn read(&mut self) -> Result<([f64; 4], [f64; 3]), String>;
}

/// Reports upright and still. Only for bench work with the robot on a stand
/// (stages 2-4): the tilt fault cannot fire with it. The real driver depends
/// on the IMU part in the BOM (G4.5).
pub struct UprightImu;

impl Imu for UprightImu {
    fn read(&mut self) -> Result<([f64; 4], [f64; 3]), String> {
        Ok(([1.0, 0.0, 0.0, 0.0], [0.0; 3]))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum HwError {
    Bus(BusError),
    Imu(String),
    /// The action has the wrong number of targets for the joint map.
    Action(String),
}

impl std::fmt::Display for HwError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HwError::Bus(e) => write!(f, "{e}"),
            HwError::Imu(e) => write!(f, "imu: {e}"),
            HwError::Action(e) => write!(f, "action: {e}"),
        }
    }
}

impl From<BusError> for HwError {
    fn from(e: BusError) -> Self {
        HwError::Bus(e)
    }
}

pub struct HwSource<P: Port, I: Imu> {
    pub bus: Bus<P>,
    pub map: JointMap,
    pub imu: I,
    /// Last voltage (V) and temperature (deg C) per joint.
    pub voltage: Vec<f64>,
    pub temperature: Vec<f64>,
    /// Error bits of every servo status in the last read, OR-ed.
    pub servo_errors: u8,
    torque_on: Option<bool>,
    next_health: usize,
    step: u64,
    t0: Instant,
}

impl<P: Port, I: Imu> HwSource<P, I> {
    /// Reads every servo's voltage and temperature once; fails if any is silent.
    pub fn new(bus: Bus<P>, map: JointMap, imu: I) -> Result<Self, HwError> {
        let mut s = HwSource {
            bus,
            imu,
            voltage: vec![0.0; map.joints.len()],
            temperature: vec![0.0; map.joints.len()],
            map,
            servo_errors: 0,
            torque_on: None,
            next_health: 0,
            step: 0,
            t0: Instant::now(),
        };
        for k in 0..s.map.joints.len() {
            s.read_health(k)?;
        }
        Ok(s)
    }

    fn read_health(&mut self, k: usize) -> Result<(), HwError> {
        let (v, t) = self.bus.read_health(self.map.joints[k].id)?;
        self.voltage[k] = v;
        self.temperature[k] = t;
        Ok(())
    }

    /// One control period's state. `estop` is the software stop input.
    pub fn read(&mut self, estop: bool) -> Result<State, HwError> {
        let ids = self.map.ids();
        let servos = self.bus.read_state(&ids)?;
        self.servo_errors = servos.iter().fold(0, |e, s| e | s.error);
        let pos: Vec<f64> = servos.iter().map(|s| s.pos_rad).collect();
        let vel: Vec<f64> = servos.iter().map(|s| s.vel_rad_s).collect();
        let k = self.next_health;
        self.next_health = (k + 1) % ids.len();
        self.read_health(k)?;
        let (quat, gyro) = self.imu.read().map_err(HwError::Imu)?;
        let state = State {
            step: self.step,
            time: self.t0.elapsed().as_secs_f64(),
            quat,
            gyro,
            bus_voltage: self.voltage.iter().copied().fold(f64::INFINITY, f64::min),
            max_temperature: self.temperature.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            qpos: self.map.to_sim(&pos),
            qvel: self.map.vel_to_sim(&vel),
            estop,
        };
        self.step += 1;
        Ok(state)
    }

    /// Sends the supervisor's action. Goals go out before torque is switched
    /// on, so a servo never holds a stale goal with torque.
    pub fn apply(&mut self, action: &Action) -> Result<(), HwError> {
        let ids = self.map.ids();
        if !action.torque_enable {
            if self.torque_on != Some(false) {
                self.bus.set_torque(&ids, false)?;
                self.torque_on = Some(false);
            }
            return Ok(());
        }
        if action.targets.len() != ids.len() {
            return Err(HwError::Action(format!(
                "{} targets for {} joints",
                action.targets.len(),
                ids.len()
            )));
        }
        self.bus.set_goals(&ids, &self.map.to_servo(&action.targets))?;
        if self.torque_on != Some(true) {
            self.bus.set_torque(&ids, true)?;
            self.torque_on = Some(true);
        }
        Ok(())
    }

    pub fn torque_on(&self) -> Option<bool> {
        self.torque_on
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jointmap::BIPED_V1_JOINTS;
    use crate::sts::{self, addr};
    use crate::stsbus::{Fault, SimPort};

    const MAP: &str = include_str!("../../hardware/calibration/biped_v1_joints.example.txt");

    fn source() -> HwSource<SimPort, UprightImu> {
        // Knee mirrored with an offset, to see the joint map applied.
        let text = MAP.replace("left_knee             4   +1    0.0", "left_knee 4 -1 0.1");
        let map = JointMap::parse(&text, &BIPED_V1_JOINTS).unwrap();
        let mut sim = sts::SimBus::new(&map.ids());
        for (k, (_, t)) in sim.tables.iter_mut().enumerate() {
            t[addr::PRESENT_VOLTAGE as usize] = 75 - k as u8; // 7.5 .. 6.6 V
            t[addr::PRESENT_TEMPERATURE as usize] = 30;
        }
        HwSource::new(Bus::new(SimPort::new(sim)), map, UprightImu).unwrap()
    }

    /// Every servo reaches its goal and reports `speed` ticks/s.
    fn settle(hw: &mut HwSource<SimPort, UprightImu>, speed: u16) {
        for (_, t) in hw.bus.port.bus.tables.iter_mut() {
            let g = [t[42], t[43]];
            t[56..58].copy_from_slice(&g);
            t[58..60].copy_from_slice(&speed.to_le_bytes());
        }
    }

    fn action(on: bool, targets: Vec<f64>) -> Action {
        Action {
            step: 0,
            torque_enable: on,
            fault_bits: 0,
            targets,
        }
    }

    fn torque_bytes(hw: &HwSource<SimPort, UprightImu>) -> Vec<u8> {
        hw.bus
            .port
            .bus
            .tables
            .iter()
            .map(|(_, t)| t[addr::TORQUE_ENABLE as usize])
            .collect()
    }

    #[test]
    fn targets_come_back_as_state_in_the_sim_frame() {
        let mut hw = source();
        let targets: Vec<f64> = (0..10).map(|k| 0.05 * k as f64 - 0.2).collect();
        hw.apply(&action(true, targets.clone())).unwrap();
        assert_eq!(torque_bytes(&hw), vec![1; 10]);
        // Left knee: servo = -sim + 0.1.
        let knee = &hw.bus.port.bus.tables[3].1;
        let want = sts::rad_to_ticks(-targets[3] + 0.1).unwrap();
        assert_eq!(u16::from_le_bytes([knee[42], knee[43]]), want);

        settle(&mut hw, 100);
        let s = hw.read(false).unwrap();
        let tick = std::f64::consts::TAU / 4096.0;
        for (k, (q, t)) in s.qpos.iter().zip(&targets).enumerate() {
            assert!((q - t).abs() <= tick, "joint {k}");
        }
        let w = 100.0 * tick;
        assert!((s.qvel[0] - w).abs() < 1e-12 && (s.qvel[3] + w).abs() < 1e-12);
        assert_eq!((s.step, s.quat, s.estop), (0, [1.0, 0.0, 0.0, 0.0], false));
        assert!((s.bus_voltage - 6.6).abs() < 1e-9 && s.max_temperature == 30.0);
        assert_eq!(hw.read(true).unwrap().step, 1);
    }

    #[test]
    fn health_is_polled_round_robin() {
        let mut hw = source();
        hw.bus.port.bus.tables[2].1[addr::PRESENT_TEMPERATURE as usize] = 70;
        hw.bus.port.bus.tables[2].1[addr::PRESENT_VOLTAGE as usize] = 60;
        assert_eq!(hw.read(false).unwrap().max_temperature, 30.0); // servo 1 polled
        hw.read(false).unwrap(); // servo 2
        let s = hw.read(false).unwrap(); // servo 3
        assert_eq!(s.max_temperature, 70.0);
        assert!((s.bus_voltage - 6.0).abs() < 1e-9);
    }

    #[test]
    fn torque_switches_only_on_change_and_off_needs_no_targets() {
        let mut hw = source();
        assert_eq!(hw.torque_on(), None);
        hw.apply(&action(false, vec![])).unwrap();
        assert_eq!((hw.torque_on(), torque_bytes(&hw)), (Some(false), vec![0; 10]));
        hw.apply(&action(true, vec![0.0; 10])).unwrap();
        // A servo that rebooted (torque off) is not re-enabled behind the flag.
        hw.bus.port.bus.tables[0].1[addr::TORQUE_ENABLE as usize] = 0;
        hw.apply(&action(true, vec![0.0; 10])).unwrap();
        assert_eq!(torque_bytes(&hw)[0], 0);
        hw.apply(&action(false, vec![])).unwrap();
        assert_eq!(torque_bytes(&hw), vec![0; 10]);
        assert!(matches!(hw.apply(&action(true, vec![0.0; 9])), Err(HwError::Action(_))));
        assert!(hw.apply(&action(true, vec![f64::NAN; 10])).is_err());
    }

    #[test]
    fn a_silent_servo_is_an_error_not_a_stale_state() {
        let mut hw = source();
        hw.bus.port.faults.extend([Fault::Drop; 3]);
        let e = hw.read(false).unwrap_err();
        assert!(e.to_string().contains("no reply"), "{e}");
        hw.bus.port.bus.tables.retain(|(id, _)| *id != 5);
        let map = hw.map.clone();
        let bus = std::mem::replace(&mut hw.bus, Bus::new(SimPort::new(sts::SimBus::new(&[]))));
        assert!(HwSource::new(bus, map, UprightImu).is_err());
    }
}
