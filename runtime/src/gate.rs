//! Command gate (ADR-0006): the only way a high-level command reaches the
//! controller. Ranges, a slew limit and a freshness timeout; anything else
//! (a planner, a gamepad, later an LLM tool call) goes through here.

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Command {
    pub vx: f64, // m/s, forward
    pub vy: f64, // m/s, left
    pub wz: f64, // rad/s, yaw rate
}

#[derive(Debug, Clone, Copy)]
pub struct GateLimits {
    pub max: Command,  // symmetric range per field
    pub rate: Command, // max change per second per field
    pub timeout_s: f64,
}

impl Default for GateLimits {
    /// Conservative values for a 0.25 m robot; WP4-01 sized the joints for
    /// Microduck's 0.4 m/s gait scaled to our leg length.
    fn default() -> Self {
        GateLimits {
            max: Command {
                vx: 0.3,
                vy: 0.15,
                wz: 1.0,
            },
            rate: Command {
                vx: 0.5,
                vy: 0.5,
                wz: 2.0,
            },
            timeout_s: 0.5,
        }
    }
}

#[derive(Debug)]
pub struct CommandGate {
    limits: GateLimits,
    output: Command,
    last_fresh: Option<(f64, Command)>,
    pub rejected: u64,
}

fn approach(current: f64, target: f64, max_step: f64) -> f64 {
    current + (target - current).clamp(-max_step, max_step)
}

impl CommandGate {
    pub fn new(limits: GateLimits) -> Self {
        CommandGate {
            limits,
            output: Command::default(),
            last_fresh: None,
            rejected: 0,
        }
    }

    /// Offer a command received at time `t` (s). Non-finite commands are dropped.
    pub fn submit(&mut self, t: f64, c: Command) {
        if !(c.vx.is_finite() && c.vy.is_finite() && c.wz.is_finite()) {
            self.rejected += 1;
            return;
        }
        let m = self.limits.max;
        let clamped = Command {
            vx: c.vx.clamp(-m.vx, m.vx),
            vy: c.vy.clamp(-m.vy, m.vy),
            wz: c.wz.clamp(-m.wz, m.wz),
        };
        self.last_fresh = Some((t, clamped));
    }

    /// Command for the control period at time `t`, `dt` after the previous one.
    pub fn output(&mut self, t: f64, dt: f64) -> Command {
        let target = match self.last_fresh {
            Some((t0, c)) if t - t0 <= self.limits.timeout_s => c,
            _ => Command::default(), // stale or never received: stop
        };
        let r = self.limits.rate;
        self.output = Command {
            vx: approach(self.output.vx, target.vx, r.vx * dt),
            vy: approach(self.output.vy, target.vy, r.vy * dt),
            wz: approach(self.output.wz, target.wz, r.wz * dt),
        };
        self.output
    }

    pub fn timed_out(&self, t: f64) -> bool {
        !matches!(self.last_fresh, Some((t0, _)) if t - t0 <= self.limits.timeout_s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(gate: &mut CommandGate, from: f64, to: f64, dt: f64) -> Command {
        let mut t = from;
        let mut c = Command::default();
        while t < to - 1e-9 {
            t += dt;
            c = gate.output(t, dt);
        }
        c
    }

    #[test]
    fn clamps_slews_and_times_out() {
        let mut g = CommandGate::new(GateLimits::default());
        g.submit(
            0.0,
            Command {
                vx: 5.0,
                vy: -5.0,
                wz: f64::NAN,
            },
        );
        assert_eq!(g.rejected, 1);
        g.submit(
            0.0,
            Command {
                vx: 5.0,
                vy: -5.0,
                wz: 0.0,
            },
        );
        let c = g.output(0.02, 0.02);
        assert!((c.vx - 0.01).abs() < 1e-12, "slew 0.5 m/s^2 x 0.02 s");
        let c = run(&mut g, 0.02, 0.4, 0.02); // 20 periods x 0.01, before the timeout
        assert!((c.vx - 0.20).abs() < 1e-9 && (c.vy + 0.15).abs() < 1e-9, "{c:?}");
        // No fresh command after t = 0: past 0.5 s the gate ramps to zero.
        let c = run(&mut g, 0.4, 2.0, 0.02);
        assert!(g.timed_out(2.0));
        assert_eq!(c, Command::default());
    }

    #[test]
    fn fresh_stream_keeps_command() {
        let mut g = CommandGate::new(GateLimits::default());
        let mut t = 0.0;
        let mut c = Command::default();
        for _ in 0..100 {
            t += 0.02;
            g.submit(
                t,
                Command {
                    vx: 0.2,
                    vy: 0.0,
                    wz: 0.5,
                },
            );
            c = g.output(t, 0.02);
        }
        assert!(!g.timed_out(t));
        assert!((c.vx - 0.2).abs() < 1e-9 && (c.wz - 0.5).abs() < 1e-9);
    }
}
