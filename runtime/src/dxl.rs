//! Dynamixel Protocol 2.0 packets for the X-series servos of biped_v1 (WP4-15).
//!
//! Only the codec, with no serial port: it builds instruction packets and parses
//! status packets, so the hardware backend (WP5) can be tested here against the
//! protocol's reference packets and a simulated servo bus. Nothing in this crate
//! opens a device.
//!
//! ```text
//! packet: FF FF FD 00 | id | len_l len_h | instruction | params (byte-stuffed) | crc_l crc_h
//! len = instruction + params (after stuffing) + crc = n_params + 3
//! ```
//! CRC-16 (polynomial 0x8005, initial 0, no reflection) over everything before it.

pub const HEADER: [u8; 4] = [0xFF, 0xFF, 0xFD, 0x00];
pub const BROADCAST: u8 = 0xFE;

pub const PING: u8 = 0x01;
pub const READ: u8 = 0x02;
pub const WRITE: u8 = 0x03;
pub const SYNC_READ: u8 = 0x82;
pub const SYNC_WRITE: u8 = 0x83;
pub const STATUS: u8 = 0x55;

/// X-series control table (XL330 / XC330), RAM area.
pub mod addr {
    pub const TORQUE_ENABLE: u16 = 64;
    pub const GOAL_POSITION: u16 = 116; // 4 bytes
    pub const PRESENT_CURRENT: u16 = 126; // 2 bytes
    pub const PRESENT_VELOCITY: u16 = 128; // 4 bytes
    pub const PRESENT_POSITION: u16 = 132; // 4 bytes
    pub const PRESENT_INPUT_VOLTAGE: u16 = 144; // 2 bytes
    pub const PRESENT_TEMPERATURE: u16 = 146; // 1 byte
}

/// Position units: 4096 per revolution, 2048 = 0 rad (servo centre).
pub fn rad_to_ticks(rad: f64) -> i32 {
    2048 + (rad * 4096.0 / std::f64::consts::TAU).round() as i32
}

pub fn ticks_to_rad(ticks: i32) -> f64 {
    (ticks - 2048) as f64 * std::f64::consts::TAU / 4096.0
}

/// Velocity units: 0.229 rev/min per tick.
pub fn ticks_to_rad_s(ticks: i32) -> f64 {
    ticks as f64 * 0.229 * std::f64::consts::TAU / 60.0
}

pub fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0;
    for &b in data {
        crc ^= (b as u16) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x8005
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// Inserts 0xFD after every FF FF FD in the parameters, so the header never
/// appears inside a packet.
fn stuff(params: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(params.len() + params.len() / 3);
    for &b in params {
        out.push(b);
        let n = out.len();
        if n >= 3 && out[n - 3..] == [0xFF, 0xFF, 0xFD] {
            out.push(0xFD);
        }
    }
    out
}

fn unstuff(params: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(params.len());
    let mut i = 0;
    while i < params.len() {
        out.push(params[i]);
        let n = out.len();
        if n >= 3 && out[n - 3..] == [0xFF, 0xFF, 0xFD] && params.get(i + 1) == Some(&0xFD) {
            i += 1; // drop the stuffing byte
        }
        i += 1;
    }
    out
}

pub fn instruction(id: u8, inst: u8, params: &[u8]) -> Vec<u8> {
    let body = stuff(params);
    let len = (body.len() + 3) as u16;
    let mut p = Vec::with_capacity(10 + body.len());
    p.extend_from_slice(&HEADER);
    p.push(id);
    p.extend_from_slice(&len.to_le_bytes());
    p.push(inst);
    p.extend_from_slice(&body);
    let crc = crc16(&p);
    p.extend_from_slice(&crc.to_le_bytes());
    p
}

pub fn ping(id: u8) -> Vec<u8> {
    instruction(id, PING, &[])
}

pub fn read(id: u8, address: u16, size: u16) -> Vec<u8> {
    let mut a = address.to_le_bytes().to_vec();
    a.extend_from_slice(&size.to_le_bytes());
    instruction(id, READ, &a)
}

pub fn write(id: u8, address: u16, data: &[u8]) -> Vec<u8> {
    let mut a = address.to_le_bytes().to_vec();
    a.extend_from_slice(data);
    instruction(id, WRITE, &a)
}

/// One packet that sets `address` on every servo: `(id, data)` with equal sizes.
pub fn sync_write(address: u16, items: &[(u8, &[u8])]) -> Result<Vec<u8>, String> {
    let size = items.first().map_or(0, |(_, d)| d.len());
    if size == 0 || items.iter().any(|(_, d)| d.len() != size) {
        return Err("sync write needs data of one non-zero size per servo".into());
    }
    let mut a = address.to_le_bytes().to_vec();
    a.extend_from_slice(&(size as u16).to_le_bytes());
    for (id, d) in items {
        a.push(*id);
        a.extend_from_slice(d);
    }
    Ok(instruction(BROADCAST, SYNC_WRITE, &a))
}

pub fn sync_read(address: u16, size: u16, ids: &[u8]) -> Vec<u8> {
    let mut a = address.to_le_bytes().to_vec();
    a.extend_from_slice(&size.to_le_bytes());
    a.extend_from_slice(ids);
    instruction(BROADCAST, SYNC_READ, &a)
}

/// Goal positions for all joints in one packet; targets in rad.
pub fn goal_positions(ids: &[u8], targets: &[f64]) -> Result<Vec<u8>, String> {
    if ids.len() != targets.len() || targets.iter().any(|t| !t.is_finite()) {
        return Err("one finite target per servo id".into());
    }
    let data: Vec<[u8; 4]> = targets.iter().map(|&t| rad_to_ticks(t).to_le_bytes()).collect();
    let items: Vec<(u8, &[u8])> = ids.iter().zip(&data).map(|(&i, d)| (i, &d[..])).collect();
    sync_write(addr::GOAL_POSITION, &items)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Status {
    pub id: u8,
    /// Error byte: bit 7 = hardware alert, low 7 bits = result fault.
    pub error: u8,
    pub params: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    /// Fewer bytes than the header or the length field says.
    Incomplete,
    BadHeader,
    BadCrc,
    NotStatus,
}

/// Parses one status packet at the start of `buf`; returns it and the bytes used.
pub fn parse_status(buf: &[u8]) -> Result<(Status, usize), ParseError> {
    if buf.len() < 11 {
        return Err(ParseError::Incomplete);
    }
    if buf[0..4] != HEADER {
        return Err(ParseError::BadHeader);
    }
    let len = u16::from_le_bytes([buf[5], buf[6]]) as usize;
    let total = 7 + len;
    if len < 4 {
        return Err(ParseError::BadHeader);
    }
    if buf.len() < total {
        return Err(ParseError::Incomplete);
    }
    let crc = u16::from_le_bytes([buf[total - 2], buf[total - 1]]);
    if crc16(&buf[..total - 2]) != crc {
        return Err(ParseError::BadCrc);
    }
    if buf[7] != STATUS {
        return Err(ParseError::NotStatus);
    }
    Ok((
        Status {
            id: buf[4],
            error: buf[8],
            params: unstuff(&buf[9..total - 2]),
        },
        total,
    ))
}

/// Little-endian i32 from the first 4 bytes (position and velocity registers).
pub fn le_i32(b: &[u8]) -> Option<i32> {
    Some(i32::from_le_bytes(b.get(0..4)?.try_into().ok()?))
}

/// A simulated bus of X-series servos for tests: answers ping, read, write,
/// sync read and sync write from a byte-level control table, the way the real
/// firmware does (status packets in id order for sync read).
#[derive(Debug, Clone)]
pub struct SimBus {
    pub tables: Vec<(u8, [u8; 256])>,
}

impl SimBus {
    pub fn new(ids: &[u8]) -> Self {
        SimBus {
            tables: ids.iter().map(|&i| (i, [0u8; 256])).collect(),
        }
    }

    fn table(&mut self, id: u8) -> Option<&mut [u8; 256]> {
        self.tables.iter_mut().find(|(i, _)| *i == id).map(|(_, t)| t)
    }

    fn status(id: u8, params: &[u8]) -> Vec<u8> {
        let mut p = vec![0u8];
        p.extend_from_slice(params);
        instruction(id, STATUS, &p)
    }

    /// Bytes the bus answers with, for one instruction packet. A packet that is
    /// corrupt, too short or addresses past the table gets no answer.
    pub fn transact(&mut self, packet: &[u8]) -> Vec<u8> {
        let n = packet.len();
        if n < 10
            || packet[0..4] != HEADER
            || crc16(&packet[..n - 2]) != u16::from_le_bytes([packet[n - 2], packet[n - 1]])
        {
            return Vec::new(); // the firmware stays silent on a corrupt packet
        }
        let p = unstuff(&packet[8..n - 2]);
        self.handle(packet[4], packet[7], &p).unwrap_or_default()
    }

    fn handle(&mut self, id: u8, inst: u8, p: &[u8]) -> Option<Vec<u8>> {
        let u16at = |k: usize| -> Option<usize> { Some(u16::from_le_bytes([*p.get(k)?, *p.get(k + 1)?]) as usize) };
        match inst {
            PING => self.table(id).map(|_| Self::status(id, &[0x2C, 0x04, 0x33])),
            READ => {
                let (a, s) = (u16at(0)?, u16at(2)?);
                let data = self.table(id)?.get(a..a + s)?.to_vec();
                Some(Self::status(id, &data))
            }
            WRITE => {
                let a = u16at(0)?;
                let data = p.get(2..)?;
                self.table(id)?.get_mut(a..a + data.len())?.copy_from_slice(data);
                Some(Self::status(id, &[]))
            }
            SYNC_WRITE => {
                let (a, s) = (u16at(0)?, u16at(2)?);
                for chunk in p.get(4..)?.chunks_exact(s + 1) {
                    if let Some(dst) = self.table(chunk[0]).and_then(|t| t.get_mut(a..a + s)) {
                        dst.copy_from_slice(&chunk[1..]);
                    }
                }
                None // broadcast: no answer
            }
            SYNC_READ => {
                let (a, s) = (u16at(0)?, u16at(2)?);
                let mut out = Vec::new();
                for &i in p.get(4..)? {
                    if let Some(data) = self.table(i).and_then(|t| t.get(a..a + s)).map(<[u8]>::to_vec) {
                        out.extend(Self::status(i, &data));
                    }
                }
                Some(out)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_packets() {
        // Ping ID 1 and Read ID 1 present position (132, 4 bytes), as in the
        // ROBOTIS Protocol 2.0 manual.
        assert_eq!(ping(1), [0xFF, 0xFF, 0xFD, 0x00, 0x01, 0x03, 0x00, 0x01, 0x19, 0x4E]);
        assert_eq!(
            read(1, addr::PRESENT_POSITION, 4),
            [0xFF, 0xFF, 0xFD, 0x00, 0x01, 0x07, 0x00, 0x02, 0x84, 0x00, 0x04, 0x00, 0x1D, 0x15]
        );
    }

    #[test]
    fn byte_stuffing_round_trip() {
        let params = [0x01, 0xFF, 0xFF, 0xFD, 0x02, 0xFF, 0xFF, 0xFD];
        let s = stuff(&params);
        assert_eq!(s, [0x01, 0xFF, 0xFF, 0xFD, 0xFD, 0x02, 0xFF, 0xFF, 0xFD, 0xFD]);
        assert_eq!(unstuff(&s), params);
        let pkt = instruction(3, STATUS, &[0x00, 0xFF, 0xFF, 0xFD, 0x07]);
        let (st, used) = parse_status(&pkt).unwrap();
        assert_eq!((st.id, st.error, used), (3, 0, pkt.len()));
        assert_eq!(st.params, [0xFF, 0xFF, 0xFD, 0x07]);
    }

    #[test]
    fn units() {
        assert_eq!(rad_to_ticks(0.0), 2048);
        assert_eq!(rad_to_ticks(std::f64::consts::FRAC_PI_2), 3072);
        assert!((ticks_to_rad(1024) + std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!((ticks_to_rad_s(100) - 100.0 * 0.229 * std::f64::consts::TAU / 60.0).abs() < 1e-12);
    }

    #[test]
    fn corrupt_and_short_status_rejected() {
        let mut pkt = instruction(1, STATUS, &[0x00, 0x10]);
        assert_eq!(parse_status(&pkt[..pkt.len() - 1]), Err(ParseError::Incomplete));
        let last = pkt.len() - 3;
        pkt[last] ^= 0x01;
        assert_eq!(parse_status(&pkt), Err(ParseError::BadCrc));
        assert_eq!(parse_status(&ping(1)[..]), Err(ParseError::Incomplete));
        let mut bad = instruction(1, STATUS, &[0, 0, 0]);
        bad[0] = 0;
        assert_eq!(parse_status(&bad), Err(ParseError::BadHeader));
        let not = instruction(1, WRITE, &[0, 0, 0]);
        assert_eq!(parse_status(&not), Err(ParseError::NotStatus));
    }

    #[test]
    fn goal_positions_reach_every_servo_through_the_sim_bus() {
        let ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut bus = SimBus::new(&ids);
        let targets: Vec<f64> = (0..10).map(|k| -0.5 + 0.1 * k as f64).collect();
        assert!(bus.transact(&goal_positions(&ids, &targets).unwrap()).is_empty());
        // Copy goal to present, as an ideal servo would, then sync read it back.
        for (_, t) in bus.tables.iter_mut() {
            let g: [u8; 4] = t[116..120].try_into().unwrap();
            t[132..136].copy_from_slice(&g);
        }
        let reply = bus.transact(&sync_read(addr::PRESENT_POSITION, 4, &ids));
        let mut at = 0;
        for (k, &id) in ids.iter().enumerate() {
            let (st, used) = parse_status(&reply[at..]).unwrap();
            at += used;
            assert_eq!(st.id, id);
            let q = ticks_to_rad(le_i32(&st.params).unwrap());
            assert!((q - targets[k]).abs() <= std::f64::consts::TAU / 4096.0);
        }
        assert_eq!(at, reply.len());
    }

    #[test]
    fn bad_inputs_are_refused() {
        assert!(goal_positions(&[1, 2], &[0.0]).is_err());
        assert!(goal_positions(&[1], &[f64::NAN]).is_err());
        assert!(sync_write(116, &[(1, &[0, 0][..]), (2, &[0][..])]).is_err());
        let mut bus = SimBus::new(&[1]);
        let mut pkt = write(1, addr::TORQUE_ENABLE, &[1]);
        let k = pkt.len() - 1;
        pkt[k] ^= 0xFF;
        assert!(bus.transact(&pkt).is_empty()); // corrupt: no reply, no write
        assert_eq!(bus.tables[0].1[64], 0);
        let ok = bus.transact(&write(1, addr::TORQUE_ENABLE, &[1]));
        assert_eq!(parse_status(&ok).unwrap().0.error, 0);
        assert_eq!(bus.tables[0].1[64], 1);
        // Past the end of the control table: no answer, no panic.
        assert!(bus.transact(&read(1, 250, 10)).is_empty());
        assert!(bus.transact(&write(1, 255, &[1, 2])).is_empty());
    }
}
