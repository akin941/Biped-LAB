//! Learned walking policy as a runtime controller (WP4-10, G4.3 preparation).
//!
//! Builds the biped_v1 actor observation exactly as `BipedVelocityEnv` does in
//! training, without its noise, lag and domain randomization:
//!
//! ```text
//! gyro (3) | projected gravity (3) | joint pos - stand pose (n) | joint vel (n) |
//! last action (n) | command vx, vy, wz (3)                        -> 3n + 9 floats
//! ```
//!
//! runs the MLP (ADR-0013), and returns `stand + action_scale * action`. The
//! policy only proposes targets: the safety supervisor still filters every one
//! (ADR-0006). A non-finite action is passed on as NaN targets, which the
//! supervisor replaces with the last safe targets, and is not fed back as the
//! last action.

use crate::controller::Controller;
use crate::gate::Command;
use crate::mlp::Mlp;
use crate::protocol::{Hello, State};

/// `BipedVelocityEnv.ACTION_SCALE` (rad per unit action).
pub const DEFAULT_ACTION_SCALE: f64 = 0.5;

/// Gravity direction (0, 0, -1) expressed in the body frame of the unit
/// quaternion `q` (w, x, y, z); same formula as `velocity_env.rotate_inverse`.
pub fn projected_gravity(q: [f64; 4]) -> [f64; 3] {
    let [w, x, y, z] = q;
    let v = [0.0, 0.0, -1.0];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let u = [x, y, z];
    let c = cross(u, v);
    let t = [2.0 * c[0], 2.0 * c[1], 2.0 * c[2]];
    let ut = cross(u, t);
    [
        v[0] - w * t[0] + ut[0],
        v[1] - w * t[1] + ut[1],
        v[2] - w * t[2] + ut[2],
    ]
}

pub struct Policy {
    mlp: Mlp,
    stand: Vec<f64>,
    action_scale: f64,
    last_action: Vec<f32>,
    obs: Vec<f32>,
    /// Periods whose action had a non-finite entry.
    pub nonfinite_actions: u64,
}

impl Policy {
    pub fn new(mlp: Mlp, hello: &Hello, action_scale: f64) -> Result<Self, String> {
        let n = hello.joints.len();
        let obs_dim = 3 * n + 9;
        if mlp.in_dim != obs_dim || mlp.out_dim != n {
            return Err(format!(
                "policy is {} -> {}, the robot needs {obs_dim} -> {n}",
                mlp.in_dim, mlp.out_dim
            ));
        }
        if !(action_scale.is_finite() && action_scale > 0.0) {
            return Err("action scale must be positive and finite".into());
        }
        Ok(Policy {
            mlp,
            stand: hello.joints.iter().map(|j| j.default).collect(),
            action_scale,
            last_action: vec![0.0; n],
            obs: Vec::with_capacity(obs_dim),
            nonfinite_actions: 0,
        })
    }

    /// The observation of the last call to `targets`.
    pub fn observation(&self) -> &[f32] {
        &self.obs
    }

    fn observe(&mut self, state: &State, cmd: Command) {
        let o = &mut self.obs;
        o.clear();
        o.extend(state.gyro.iter().map(|&v| v as f32));
        o.extend(projected_gravity(state.quat).iter().map(|&v| v as f32));
        o.extend(state.qpos.iter().zip(&self.stand).map(|(q, s)| (q - s) as f32));
        o.extend(state.qvel.iter().map(|&v| v as f32));
        o.extend_from_slice(&self.last_action);
        o.extend([cmd.vx as f32, cmd.vy as f32, cmd.wz as f32]);
    }
}

impl Controller for Policy {
    fn targets(&mut self, state: &State, cmd: Command) -> Vec<f64> {
        self.observe(state, cmd);
        let action = self.mlp.forward(&self.obs);
        if action.iter().any(|a| !a.is_finite()) {
            self.nonfinite_actions += 1;
            return vec![f64::NAN; self.stand.len()];
        }
        let targets = self
            .stand
            .iter()
            .zip(&action)
            .map(|(s, &a)| s + self.action_scale * a as f64)
            .collect();
        self.last_action = action;
        targets
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlp::Op;
    use crate::protocol::JointSpec;

    fn hello(n: usize) -> Hello {
        Hello {
            control_dt: 0.02,
            joints: (0..n)
                .map(|j| JointSpec {
                    lower: -1.0,
                    upper: 1.0,
                    default: 0.1 * j as f64,
                    kp: 2.0,
                    torque_limit: 1.0,
                })
                .collect(),
        }
    }

    fn state(n: usize) -> State {
        State {
            step: 1,
            time: 0.0,
            quat: [1.0, 0.0, 0.0, 0.0],
            gyro: [0.1, -0.2, 0.3],
            bus_voltage: 7.4,
            max_temperature: 35.0,
            qpos: (0..n).map(|j| 0.1 * j as f64 + 0.01).collect(),
            qvel: vec![0.5; n],
            estop: false,
        }
    }

    /// Selects observation entries `pick` (one per output) and scales them by `gain`.
    fn picker(in_dim: usize, pick: &[usize], gain: f32) -> Mlp {
        let cols = pick.len();
        let mut w = vec![0.0; in_dim * cols];
        for (c, &r) in pick.iter().enumerate() {
            w[r * cols + c] = gain;
        }
        Mlp {
            in_dim,
            out_dim: cols,
            ops: vec![Op::Dense {
                rows: in_dim,
                cols,
                w,
                b: vec![0.0; cols],
            }],
        }
    }

    #[test]
    fn gravity_upright_and_tilted() {
        let g = projected_gravity([1.0, 0.0, 0.0, 0.0]);
        assert_eq!(g, [0.0, 0.0, -1.0]);
        // 90 deg roll about x: body +z points along world -y, so world down is body -y.
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let g = projected_gravity([h, h, 0.0, 0.0]);
        assert!((g[0]).abs() < 1e-12 && (g[1] + 1.0).abs() < 1e-12 && g[2].abs() < 1e-12);
    }

    #[test]
    fn observation_layout_and_targets() {
        let n = 2;
        let h = hello(n);
        // Outputs: action0 = joint_pos0 term, action1 = command wz.
        let mlp = picker(3 * n + 9, &[6, 3 * n + 8], 1.0);
        let mut p = Policy::new(mlp, &h, 0.5).unwrap();
        let cmd = Command {
            vx: 0.2,
            vy: 0.0,
            wz: 0.4,
        };
        let t = p.targets(&state(n), cmd);
        let o = p.observation();
        assert_eq!(o.len(), 15);
        assert_eq!(&o[0..3], &[0.1, -0.2, 0.3]);
        assert_eq!(&o[3..6], &[0.0, 0.0, -1.0]);
        assert!((o[6] - 0.01).abs() < 1e-6 && (o[7] - 0.01).abs() < 1e-6);
        assert_eq!(&o[8..10], &[0.5, 0.5]);
        assert_eq!(&o[10..12], &[0.0, 0.0]); // no previous action yet
        assert_eq!(&o[12..15], &[0.2, 0.0, 0.4]);
        assert!((t[0] - (0.0 + 0.5 * 0.01)).abs() < 1e-6);
        assert!((t[1] - (0.1 + 0.5 * 0.4)).abs() < 1e-6);
        // The raw action is fed back on the next period.
        p.targets(&state(n), cmd);
        let o = p.observation();
        assert!((o[10] - 0.01).abs() < 1e-6 && (o[11] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn wrong_size_policy_rejected() {
        let h = hello(3);
        assert!(Policy::new(picker(10, &[0, 1, 2], 1.0), &h, 0.5).is_err());
        assert!(Policy::new(picker(18, &[0, 1], 1.0), &h, 0.5).is_err());
        assert!(Policy::new(picker(18, &[0, 1, 2], 1.0), &h, 0.0).is_err());
    }

    #[test]
    fn nonfinite_action_gives_nan_targets_and_is_not_fed_back() {
        let n = 1;
        let h = hello(n);
        let mut p = Policy::new(picker(12, &[0], 1.0), &h, 0.5).unwrap();
        let mut s = state(n);
        s.gyro[0] = f64::INFINITY;
        let t = p.targets(&s, Command::default());
        assert!(t[0].is_nan());
        assert_eq!(p.nonfinite_actions, 1);
        p.targets(&state(n), Command::default());
        assert_eq!(p.observation()[3 * n + 6], 0.0);
    }
}
