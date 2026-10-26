//! Sim bridge wire format: fixed-size little-endian frames over TCP.
//!
//! Must match `bipedlab.runtime_bridge` (Python). The sim is the server and
//! steps in lockstep: it sends `Hello` once, then one `State` per control
//! period and waits for the matching `Action`.
//!
//! ```text
//! Hello  : magic "BPLB" | version u16 | n u16 | control_dt f64 | n x (lower, upper, default) f64
//! State  : tag 1 u8 | step u64 | time f64 | quat 4 f64 | gyro 3 f64 | qpos n f64 | qvel n f64
//! Action : tag 2 u8 | step u64 | torque_enable u8 | fault_bits u32 | targets n f64
//! ```

use std::io::{self, Read, Write};

pub const MAGIC: &[u8; 4] = b"BPLB";
pub const VERSION: u16 = 1;
pub const TAG_STATE: u8 = 1;
pub const TAG_ACTION: u8 = 2;
/// Upper bound on joints, so a corrupt header cannot make us allocate much.
pub const MAX_JOINTS: usize = 64;

#[derive(Debug, Clone, PartialEq)]
pub struct JointSpec {
    pub lower: f64,
    pub upper: f64,
    pub default: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hello {
    pub control_dt: f64,
    pub joints: Vec<JointSpec>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct State {
    pub step: u64,
    pub time: f64,
    pub quat: [f64; 4],
    pub gyro: [f64; 3],
    pub qpos: Vec<f64>,
    pub qvel: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Action {
    pub step: u64,
    pub torque_enable: bool,
    pub fault_bits: u32,
    pub targets: Vec<f64>,
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

fn read_u8(r: &mut impl Read) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u16(r: &mut impl Read) -> io::Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_f64s(r: &mut impl Read, n: usize) -> io::Result<Vec<f64>> {
    let mut buf = vec![0u8; 8 * n];
    r.read_exact(&mut buf)?;
    Ok(buf
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("8-byte chunk")))
        .collect())
}

impl Hello {
    pub fn read(r: &mut impl Read) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(bad("bad magic"));
        }
        if read_u16(r)? != VERSION {
            return Err(bad("unsupported protocol version"));
        }
        let n = read_u16(r)? as usize;
        if n == 0 || n > MAX_JOINTS {
            return Err(bad("joint count out of range"));
        }
        let control_dt = read_f64s(r, 1)?[0];
        if !(control_dt.is_finite() && control_dt > 0.0 && control_dt < 1.0) {
            return Err(bad("control_dt out of range"));
        }
        let v = read_f64s(r, 3 * n)?;
        let joints = v
            .chunks_exact(3)
            .map(|c| JointSpec {
                lower: c[0],
                upper: c[1],
                default: c[2],
            })
            .collect::<Vec<_>>();
        for j in &joints {
            let ok = j.lower.is_finite() && j.upper.is_finite() && j.default.is_finite();
            if !ok || j.lower >= j.upper || j.default < j.lower || j.default > j.upper {
                return Err(bad("invalid joint range"));
            }
        }
        Ok(Hello { control_dt, joints })
    }

    pub fn write(&self, w: &mut impl Write) -> io::Result<()> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(self.joints.len() as u16).to_le_bytes());
        out.extend_from_slice(&self.control_dt.to_le_bytes());
        for j in &self.joints {
            for v in [j.lower, j.upper, j.default] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        w.write_all(&out)
    }
}

impl State {
    pub fn read(r: &mut impl Read, n: usize) -> io::Result<Self> {
        if read_u8(r)? != TAG_STATE {
            return Err(bad("expected a state frame"));
        }
        let step = read_u64(r)?;
        let v = read_f64s(r, 1 + 4 + 3 + 2 * n)?;
        Ok(State {
            step,
            time: v[0],
            quat: [v[1], v[2], v[3], v[4]],
            gyro: [v[5], v[6], v[7]],
            qpos: v[8..8 + n].to_vec(),
            qvel: v[8 + n..].to_vec(),
        })
    }

    pub fn write(&self, w: &mut impl Write) -> io::Result<()> {
        let mut out = vec![TAG_STATE];
        out.extend_from_slice(&self.step.to_le_bytes());
        let scalars = [self.time];
        for v in scalars
            .iter()
            .chain(&self.quat)
            .chain(&self.gyro)
            .chain(&self.qpos)
            .chain(&self.qvel)
        {
            out.extend_from_slice(&v.to_le_bytes());
        }
        w.write_all(&out)
    }
}

impl Action {
    pub fn write(&self, w: &mut impl Write) -> io::Result<()> {
        let mut out = vec![TAG_ACTION];
        out.extend_from_slice(&self.step.to_le_bytes());
        out.push(u8::from(self.torque_enable));
        out.extend_from_slice(&self.fault_bits.to_le_bytes());
        for v in &self.targets {
            out.extend_from_slice(&v.to_le_bytes());
        }
        w.write_all(&out)
    }

    pub fn read(r: &mut impl Read, n: usize) -> io::Result<Self> {
        if read_u8(r)? != TAG_ACTION {
            return Err(bad("expected an action frame"));
        }
        let step = read_u64(r)?;
        let torque_enable = read_u8(r)? != 0;
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        Ok(Action {
            step,
            torque_enable,
            fault_bits: u32::from_le_bytes(b),
            targets: read_f64s(r, n)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hello() -> Hello {
        Hello {
            control_dt: 0.02,
            joints: vec![
                JointSpec {
                    lower: -1.0,
                    upper: 1.0,
                    default: 0.0,
                },
                JointSpec {
                    lower: -0.1,
                    upper: 2.2,
                    default: 0.6,
                },
            ],
        }
    }

    #[test]
    fn frames_round_trip() {
        let mut buf = Vec::new();
        hello().write(&mut buf).unwrap();
        assert_eq!(buf.len(), 4 + 2 + 2 + 8 + 2 * 24);
        assert_eq!(Hello::read(&mut buf.as_slice()).unwrap(), hello());

        let s = State {
            step: 7,
            time: 0.14,
            quat: [1.0, 0.0, 0.0, 0.0],
            gyro: [0.1, -0.2, 0.3],
            qpos: vec![0.5, 0.6],
            qvel: vec![-1.0, 1.0],
        };
        let mut buf = Vec::new();
        s.write(&mut buf).unwrap();
        assert_eq!(State::read(&mut buf.as_slice(), 2).unwrap(), s);

        let a = Action {
            step: 7,
            torque_enable: true,
            fault_bits: 5,
            targets: vec![0.1, 0.2],
        };
        let mut buf = Vec::new();
        a.write(&mut buf).unwrap();
        assert_eq!(buf.len(), 1 + 8 + 1 + 4 + 16);
        assert_eq!(Action::read(&mut buf.as_slice(), 2).unwrap(), a);
    }

    #[test]
    fn rejects_bad_headers() {
        let mut buf = Vec::new();
        hello().write(&mut buf).unwrap();
        let mut wrong = buf.clone();
        wrong[0] = b'X';
        assert!(Hello::read(&mut wrong.as_slice()).is_err());
        let mut many = buf.clone();
        many[6..8].copy_from_slice(&(MAX_JOINTS as u16 + 1).to_le_bytes());
        assert!(Hello::read(&mut many.as_slice()).is_err());
        let mut inverted = Vec::new();
        Hello {
            control_dt: 0.02,
            joints: vec![JointSpec {
                lower: 1.0,
                upper: -1.0,
                default: 0.0,
            }],
        }
        .write(&mut inverted)
        .unwrap();
        assert!(Hello::read(&mut inverted.as_slice()).is_err());
        // Truncated frame.
        assert!(Hello::read(&mut &buf[..buf.len() - 1]).is_err());
    }
}
