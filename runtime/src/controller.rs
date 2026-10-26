//! Controllers turn (state, gated command) into joint targets.
//!
//! The skeleton has no learned policy for our robot yet (that is G4.3), so the
//! nominal controller holds the stand pose. The ONNX policy will implement the
//! same trait. The fault-injection controllers exist only to exercise the
//! safety supervisor in the sim tests.

use crate::gate::Command;
use crate::protocol::{Hello, State};

pub trait Controller {
    fn targets(&mut self, state: &State, cmd: Command) -> Vec<f64>;
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
}
