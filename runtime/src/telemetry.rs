//! Per-period telemetry log (WP4-13).
//!
//! `biped-runtime --log FILE` writes one fixed-size record per control period:
//! what the robot measured, what the gate passed, what the controller wanted and
//! what the supervisor sent. `bipedlab.telemetry.runtime_log` reads it and turns
//! it into the WP2-01 telemetry schema, so `bipedlab report` works on it.
//!
//! Logging must never stop the control loop: a failed write disables the log,
//! counts the failure, and the loop goes on.
//!
//! ```text
//! header: magic "BPLG" | version u16 | n u16 | control_dt f64 | n x (kp f64, torque_limit f64)
//! record: step u64 | time f64 | wall f64 (s since the first state) | compute_us u32 |
//!         quat 4 f64 | gyro 3 f64 | bus_voltage f64 | max_temperature f64 | estop u8 |
//!         qpos n f64 | qvel n f64 | command 3 f64 | wanted n f64 | targets n f64 |
//!         torque_enable u8 | bits u32
//! ```
//! All little-endian.

use std::io::{self, Write};

use crate::gate::Command;
use crate::protocol::{Hello, State};

pub const MAGIC: &[u8; 4] = b"BPLG";
pub const VERSION: u16 = 1;

pub fn header_size(n: usize) -> usize {
    4 + 2 + 2 + 8 + 16 * n
}

pub fn record_size(n: usize) -> usize {
    8 + 8 + 8 + 4 + 8 * (4 + 3 + 2) + 1 + 8 * (4 * n + 3) + 1 + 4
}

pub struct Record<'a> {
    pub state: &'a State,
    pub wall: f64,
    pub compute_us: u32,
    pub cmd: Command,
    pub wanted: &'a [f64],
    pub targets: &'a [f64],
    pub torque_enable: bool,
    pub bits: u32,
}

pub struct Log<W: Write> {
    out: Option<W>,
    n: usize,
    buf: Vec<u8>,
    pub records: u64,
    pub write_errors: u64,
}

fn f64s(buf: &mut Vec<u8>, v: &[f64]) {
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}

impl<W: Write> Log<W> {
    pub fn new(mut out: W, hello: &Hello) -> io::Result<Self> {
        let n = hello.joints.len();
        let mut h = Vec::with_capacity(header_size(n));
        h.extend_from_slice(MAGIC);
        h.extend_from_slice(&VERSION.to_le_bytes());
        h.extend_from_slice(&(n as u16).to_le_bytes());
        h.extend_from_slice(&hello.control_dt.to_le_bytes());
        for j in &hello.joints {
            f64s(&mut h, &[j.kp, j.torque_limit]);
        }
        out.write_all(&h)?;
        Ok(Log {
            out: Some(out),
            n,
            buf: Vec::with_capacity(record_size(n)),
            records: 0,
            write_errors: 0,
        })
    }

    pub fn active(&self) -> bool {
        self.out.is_some()
    }

    pub fn record(&mut self, r: &Record) {
        let Some(out) = self.out.as_mut() else { return };
        let s = r.state;
        let n = self.n;
        let wanted_ok = r.wanted.len() == n;
        let b = &mut self.buf;
        b.clear();
        b.extend_from_slice(&s.step.to_le_bytes());
        f64s(b, &[s.time, r.wall]);
        b.extend_from_slice(&r.compute_us.to_le_bytes());
        f64s(b, &s.quat);
        f64s(b, &s.gyro);
        f64s(b, &[s.bus_voltage, s.max_temperature]);
        b.push(s.estop as u8);
        f64s(b, &s.qpos);
        f64s(b, &s.qvel);
        f64s(b, &[r.cmd.vx, r.cmd.vy, r.cmd.wz]);
        if wanted_ok {
            f64s(b, r.wanted);
        } else {
            // A controller that returned the wrong number of targets: log NaN.
            f64s(b, &vec![f64::NAN; n]);
        }
        f64s(b, r.targets);
        b.push(r.torque_enable as u8);
        b.extend_from_slice(&r.bits.to_le_bytes());
        if b.len() != record_size(n) || out.write_all(b).is_err() {
            self.write_errors += 1;
            self.out = None; // stop logging, keep controlling
            return;
        }
        self.records += 1;
    }

    pub fn finish(&mut self) {
        if let Some(mut out) = self.out.take() {
            if out.flush().is_err() {
                self.write_errors += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::JointSpec;

    fn hello(n: usize) -> Hello {
        Hello {
            control_dt: 0.02,
            joints: (0..n)
                .map(|_| JointSpec {
                    lower: -1.0,
                    upper: 1.0,
                    default: 0.0,
                    kp: 2.0,
                    torque_limit: 0.5,
                })
                .collect(),
        }
    }

    fn state(n: usize) -> State {
        State {
            step: 7,
            time: 0.14,
            quat: [1.0, 0.0, 0.0, 0.0],
            gyro: [0.0; 3],
            bus_voltage: 7.4,
            max_temperature: 35.0,
            qpos: vec![0.1; n],
            qvel: vec![0.0; n],
            estop: false,
        }
    }

    #[test]
    fn header_and_records_have_the_documented_size() {
        let n = 3;
        let mut log = Log::new(Vec::new(), &hello(n)).unwrap();
        let s = state(n);
        let t = vec![0.2; n];
        for _ in 0..2 {
            log.record(&Record {
                state: &s,
                wall: 0.0,
                compute_us: 12,
                cmd: Command::default(),
                wanted: &t,
                targets: &t,
                torque_enable: true,
                bits: 1 << 10,
            });
        }
        assert_eq!(log.records, 2);
        let out = log.out.take().unwrap();
        assert_eq!(out.len(), header_size(n) + 2 * record_size(n));
        assert_eq!(&out[0..4], MAGIC);
        let rec = &out[header_size(n)..];
        assert_eq!(u64::from_le_bytes(rec[0..8].try_into().unwrap()), 7);
        let bits_at = record_size(n) - 4;
        assert_eq!(
            u32::from_le_bytes(rec[bits_at..bits_at + 4].try_into().unwrap()),
            1 << 10
        );
    }

    struct Broken;
    impl Write for Broken {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("disk full"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    struct FailAfterHeader(usize);
    impl Write for FailAfterHeader {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            if self.0 == 0 {
                return Err(io::Error::other("disk full"));
            }
            self.0 -= 1;
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_failed_write_disables_the_log_without_panicking() {
        assert!(Log::new(Broken, &hello(1)).is_err());
        let mut log = Log::new(FailAfterHeader(1), &hello(1)).unwrap();
        let s = state(1);
        let r = Record {
            state: &s,
            wall: 0.0,
            compute_us: 0,
            cmd: Command::default(),
            wanted: &[0.0],
            targets: &[0.0],
            torque_enable: true,
            bits: 0,
        };
        log.record(&r);
        log.record(&r);
        assert!(!log.active());
        assert_eq!((log.records, log.write_errors), (0, 1));
    }
}
