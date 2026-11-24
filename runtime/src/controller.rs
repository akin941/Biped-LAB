//! Controllers turn (state, gated command) into joint targets.
//!
//! `Stand` holds the stand pose; the learned policy (`policy::Policy`, WP4-10)
//! implements the same trait. The fault-injection controllers exist only to
//! exercise the safety supervisor in the sim tests.

use crate::gate::Command;
use crate::protocol::{Hello, State};

pub trait Controller {
    fn targets(&mut self, state: &State, cmd: Command) -> Vec<f64>;

    /// The targets that actually went out this period, after the safety
    /// supervisor. A controller that observes its own commands (PWM servos
    /// without feedback, G4.5) keeps them; the default ignores them.
    fn sent(&mut self, _targets: &[f64]) {}
}

/// Holds the default (stand) pose sent by the sim.
pub struct Stand {
    pose: Vec<f64>,
}

impl Stand {
    pub fn new(hello: &Hello) -> Self {
        Stand {
            pose: hello.joints.iter().map(|j| j.default).collect(),
        }
    }
}

impl Controller for Stand {
    fn targets(&mut self, _state: &State, _cmd: Command) -> Vec<f64> {
        self.pose.clone()
    }
}

/// Sits down from the stand pose (WP5-19): from time `at` the pitch joints
/// follow a smoothstep over `seconds` to a crouch with knee bend `knee`
/// (trunk upright, feet flat: hip pitch and ankle take −knee/2 each), then
/// hold it. In sim (`scripts/sit_stand.py`) knee 2.0 over 1 s stays upright
/// at a third of the servo torque and lowers the trunk before torque-off.
/// Opt-in only: the shutdown and fault paths still cut torque at once.
pub struct Sit {
    stand: Vec<f64>,
    sit: Vec<f64>,
    at: f64,
    seconds: f64,
}

/// Leg joint order: hip yaw, hip roll, hip pitch, knee, ankle pitch.
const PITCH: [(usize, f64); 3] = [(2, -0.5), (3, 1.0), (4, -0.5)];

impl Sit {
    pub fn new(hello: &Hello, at: f64, seconds: f64, knee: f64) -> Result<Self, String> {
        if hello.joints.len() % 5 != 0 || hello.joints.is_empty() {
            return Err(format!("sit needs 5 joints per leg, got {}", hello.joints.len()));
        }
        if !(at >= 0.0 && seconds > 0.0 && knee.is_finite()) {
            return Err("sit: AT >= 0, SECONDS > 0, finite KNEE".into());
        }
        let stand: Vec<f64> = hello.joints.iter().map(|j| j.default).collect();
        let mut sit = stand.clone();
        for leg in 0..hello.joints.len() / 5 {
            for (k, share) in PITCH {
                let j = &hello.joints[5 * leg + k];
                let q = share * knee;
                if q < j.lower || q > j.upper {
                    return Err(format!(
                        "sit: knee {knee} needs joint {} at {q:.3}, outside {}..{}",
                        5 * leg + k,
                        j.lower,
                        j.upper
                    ));
                }
                sit[5 * leg + k] = q;
            }
        }
        Ok(Sit {
            stand,
            sit,
            at,
            seconds,
        })
    }
}

impl Controller for Sit {
    fn targets(&mut self, state: &State, _cmd: Command) -> Vec<f64> {
        let x = ((state.time - self.at) / self.seconds).clamp(0.0, 1.0);
        let s = x * x * (3.0 - 2.0 * x);
        self.stand.iter().zip(&self.sit).map(|(a, b)| a + (b - a) * s).collect()
    }
}

/// Fault injection: from time `at` on, wraps another controller's output.
pub enum Injected {
    /// Every target becomes NaN.
    NonFinite,
    /// Joint `joint` target jumps by `delta` rad.
    Jump { joint: usize, delta: f64 },
}

pub struct FaultInjector<C: Controller> {
    pub inner: C,
    pub at: f64,
    pub fault: Injected,
}

impl<C: Controller> Controller for FaultInjector<C> {
    fn targets(&mut self, state: &State, cmd: Command) -> Vec<f64> {
        let mut t = self.inner.targets(state, cmd);
        if state.time >= self.at {
            match self.fault {
                Injected::NonFinite => t.iter_mut().for_each(|v| *v = f64::NAN),
                Injected::Jump { joint, delta } => {
                    if let Some(v) = t.get_mut(joint) {
                        *v += delta;
                    }
                }
            }
        }
        t
    }

    fn sent(&mut self, targets: &[f64]) {
        self.inner.sent(targets);
    }
}
