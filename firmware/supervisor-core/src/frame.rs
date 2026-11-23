//! UART frames between the Pi (`biped-runtime`) and the supervisor MCU
//! (ADR-0015). Little-endian, CRC-16/CCITT-FALSE over everything after the
//! sync bytes.
//!
//! ```text
//! targets (Pi -> MCU): A5 5A | 01 | seq u8 | flags u8 | n u8 | n x i16 (1e-4 rad) | crc u16
//!   flags: bit0 torque enable, bit1 stop (latches SOFT_STOP), bit2 clear latched faults
//! status (MCU -> Pi):  A5 5A | 02 | seq u8 (last targets) | bits u16 | battery u16 (mV) |
//!                      n u8 | n x u16 (pulse us, 0 = off) | crc u16
//! ```

use crate::N_JOINTS;

pub const SYNC: [u8; 2] = [0xA5, 0x5A];
pub const KIND_TARGETS: u8 = 1;
pub const KIND_STATUS: u8 = 2;
/// Targets are sent in units of 1e-4 rad (±3.2767 rad).
pub const RAD_PER_UNIT: f32 = 1e-4;
pub const TARGETS_LEN: usize = 2 + 4 + 2 * N_JOINTS + 2;
pub const STATUS_LEN: usize = 2 + 7 + 2 * N_JOINTS + 2;

pub fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &b in data {
        crc ^= (b as u16) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x1021
            } else {
                crc << 1
            };
        }
    }
    crc
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Targets {
    pub seq: u8,
    pub torque_enable: bool,
    pub stop: bool,
    pub clear: bool,
    /// Joint targets in the sim frame (rad).
    pub targets: [f32; N_JOINTS],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameError {
    Length,
    Sync,
    Kind,
    Joints,
    Crc,
}

impl Targets {
    pub fn encode(&self) -> [u8; TARGETS_LEN] {
        let mut b = [0u8; TARGETS_LEN];
        b[0..2].copy_from_slice(&SYNC);
        b[2] = KIND_TARGETS;
        b[3] = self.seq;
        b[4] = self.torque_enable as u8 | (self.stop as u8) << 1 | (self.clear as u8) << 2;
        b[5] = N_JOINTS as u8;
        for (i, q) in self.targets.iter().enumerate() {
            let units = (q / RAD_PER_UNIT).clamp(i16::MIN as f32, i16::MAX as f32);
            let units = if units.is_finite() { units as i16 } else { 0 };
            b[6 + 2 * i..8 + 2 * i].copy_from_slice(&units.to_le_bytes());
        }
        let crc = crc16(&b[2..TARGETS_LEN - 2]);
        b[TARGETS_LEN - 2..].copy_from_slice(&crc.to_le_bytes());
        b
    }

    pub fn decode(b: &[u8]) -> Result<Targets, FrameError> {
        if b.len() != TARGETS_LEN {
            return Err(FrameError::Length);
        }
        if b[0..2] != SYNC {
            return Err(FrameError::Sync);
        }
        if b[2] != KIND_TARGETS {
            return Err(FrameError::Kind);
        }
        if b[5] as usize != N_JOINTS {
            return Err(FrameError::Joints);
        }
        let crc = u16::from_le_bytes([b[TARGETS_LEN - 2], b[TARGETS_LEN - 1]]);
        if crc != crc16(&b[2..TARGETS_LEN - 2]) {
            return Err(FrameError::Crc);
        }
        let mut targets = [0.0f32; N_JOINTS];
        for (i, t) in targets.iter_mut().enumerate() {
            *t = i16::from_le_bytes([b[6 + 2 * i], b[7 + 2 * i]]) as f32 * RAD_PER_UNIT;
        }
        Ok(Targets {
            seq: b[3],
            torque_enable: b[4] & 1 != 0,
            stop: b[4] & 2 != 0,
            clear: b[4] & 4 != 0,
            targets,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Status {
    pub seq: u8,
    pub bits: u16,
    pub battery_mv: u16,
    pub pulses_us: [u16; N_JOINTS],
}

impl Status {
    pub fn encode(&self) -> [u8; STATUS_LEN] {
        let mut b = [0u8; STATUS_LEN];
        b[0..2].copy_from_slice(&SYNC);
        b[2] = KIND_STATUS;
        b[3] = self.seq;
        b[4..6].copy_from_slice(&self.bits.to_le_bytes());
        b[6..8].copy_from_slice(&self.battery_mv.to_le_bytes());
        b[8] = N_JOINTS as u8;
        for (i, p) in self.pulses_us.iter().enumerate() {
            b[9 + 2 * i..11 + 2 * i].copy_from_slice(&p.to_le_bytes());
        }
        let crc = crc16(&b[2..STATUS_LEN - 2]);
        b[STATUS_LEN - 2..].copy_from_slice(&crc.to_le_bytes());
        b
    }

    pub fn decode(b: &[u8]) -> Result<Status, FrameError> {
        if b.len() != STATUS_LEN {
            return Err(FrameError::Length);
        }
        if b[0..2] != SYNC {
            return Err(FrameError::Sync);
        }
        if b[2] != KIND_STATUS {
            return Err(FrameError::Kind);
        }
        if b[8] as usize != N_JOINTS {
            return Err(FrameError::Joints);
        }
        let crc = u16::from_le_bytes([b[STATUS_LEN - 2], b[STATUS_LEN - 1]]);
        if crc != crc16(&b[2..STATUS_LEN - 2]) {
            return Err(FrameError::Crc);
        }
        let mut pulses_us = [0u16; N_JOINTS];
        for (i, p) in pulses_us.iter_mut().enumerate() {
            *p = u16::from_le_bytes([b[9 + 2 * i], b[10 + 2 * i]]);
        }
        Ok(Status {
            seq: b[3],
            bits: u16::from_le_bytes([b[4], b[5]]),
            battery_mv: u16::from_le_bytes([b[6], b[7]]),
            pulses_us,
        })
    }
}

/// Finds frames of a fixed length in a byte stream (resyncs on the sync
/// bytes after noise or a dropped byte).
pub struct Reader<const L: usize> {
    buf: [u8; L],
    len: usize,
}

impl<const L: usize> Default for Reader<L> {
    fn default() -> Self {
        Reader {
            buf: [0; L],
            len: 0,
        }
    }
}

impl<const L: usize> Reader<L> {
    /// Feeds one byte; returns a complete candidate frame (still to be
    /// decoded and CRC-checked) when one is ready.
    pub fn push(&mut self, byte: u8) -> Option<[u8; L]> {
        if self.len < 2 && byte != SYNC[self.len] {
            // Not a sync byte where one belongs: restart (A5 A5 5A still syncs).
            self.len = usize::from(byte == SYNC[0]);
            if self.len == 1 {
                self.buf[0] = byte;
            }
            return None;
        }
        self.buf[self.len] = byte;
        self.len += 1;
        if self.len == L {
            self.len = 0;
            return Some(self.buf);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_check_value() {
        assert_eq!(crc16(b"123456789"), 0x29B1); // CRC-16/CCITT-FALSE
    }

    #[test]
    fn targets_round_trip_and_rejects() {
        let mut t = Targets {
            seq: 7,
            torque_enable: true,
            stop: false,
            clear: true,
            targets: [0.0; N_JOINTS],
        };
        t.targets[3] = 1.2345;
        t.targets[9] = -0.5;
        let b = t.encode();
        let d = Targets::decode(&b).unwrap();
        assert_eq!(
            (d.seq, d.torque_enable, d.stop, d.clear),
            (7, true, false, true)
        );
        assert!((d.targets[3] - 1.2345).abs() < 1e-4 && (d.targets[9] + 0.5).abs() < 1e-4);
        let mut bad = b;
        bad[10] ^= 1;
        assert_eq!(Targets::decode(&bad), Err(FrameError::Crc));
        assert_eq!(Targets::decode(&b[1..]), Err(FrameError::Length));
        let mut s = b;
        s[2] = KIND_STATUS;
        assert_eq!(Targets::decode(&s), Err(FrameError::Kind));
        t.targets[0] = f32::NAN;
        t.targets[1] = 100.0;
        let d = Targets::decode(&t.encode()).unwrap();
        assert_eq!(d.targets[0], 0.0);
        assert!((d.targets[1] - 3.2767).abs() < 1e-4);
    }

    #[test]
    fn status_round_trip() {
        let s = Status {
            seq: 3,
            bits: 0x0201,
            battery_mv: 7400,
            pulses_us: [1500; N_JOINTS],
        };
        assert_eq!(Status::decode(&s.encode()), Ok(s));
    }

    #[test]
    fn reader_resyncs_after_noise() {
        let f = Targets {
            seq: 1,
            torque_enable: false,
            stop: false,
            clear: false,
            targets: [0.25; N_JOINTS],
        }
        .encode();
        let mut r = Reader::<TARGETS_LEN>::default();
        let mut got = None;
        for &b in [0x00, 0xA5, 0x13, 0xA5].iter().chain(f.iter()) {
            if let Some(c) = r.push(b) {
                got = Some(c);
            }
        }
        assert_eq!(Targets::decode(&got.unwrap()).unwrap().seq, 1);
    }
}
