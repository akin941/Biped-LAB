//! Feetech SCS/STS serial-bus packets for the STS3215 servo (WP4-29).
//!
//! The candidate servo that walks under its real torque-speed curve in MAC-37
//! (mac-m5). This is only the codec: it builds instruction packets and parses
//! status packets, so a hardware backend can be tested here against reference
//! packets and a simulated bus. `stsbus` sends them over a port.
//!
//! ```text
//! instruction: FF FF | id | len | instruction | params | checksum
//! status:      FF FF | id | len | error       | params | checksum
//! len = params + 2; checksum = !(id + len + instruction/error + params) (low byte)
//! ```
//! STS registers are little-endian (the older SCS series is big-endian).

pub const HEADER: [u8; 2] = [0xFF, 0xFF];
pub const BROADCAST: u8 = 0xFE;

pub const PING: u8 = 0x01;
pub const READ: u8 = 0x02;
pub const WRITE: u8 = 0x03;
pub const SYNC_READ: u8 = 0x82;
pub const SYNC_WRITE: u8 = 0x83;

/// STS3215 memory table, RAM area.
pub mod addr {
    pub const TORQUE_ENABLE: u8 = 40; // 1 byte
    pub const GOAL_POSITION: u8 = 42; // 2 bytes
    /// Torque limit, 0..=1000 = 0..100 % of the stall torque (RAM, reset at
    /// power-up from the EEPROM max torque). From the STS memory table; to be
    /// confirmed on the first servo at stage 3.
    pub const TORQUE_LIMIT: u8 = 48; // 2 bytes
    pub const PRESENT_POSITION: u8 = 56; // 2 bytes
    pub const PRESENT_SPEED: u8 = 58; // 2 bytes, bit 15 = sign
    pub const PRESENT_LOAD: u8 = 60; // 2 bytes, bit 10 = sign
    pub const PRESENT_VOLTAGE: u8 = 62; // 1 byte, 0.1 V
    pub const PRESENT_TEMPERATURE: u8 = 63; // 1 byte, deg C
}

/// Position units: 4096 per revolution, 2048 = 0 rad (servo centre), 0..=4095.
pub fn rad_to_ticks(rad: f64) -> Option<u16> {
    let t = 2048.0 + (rad * 4096.0 / std::f64::consts::TAU).round();
    (0.0..=4095.0).contains(&t).then_some(t as u16)
}

pub fn ticks_to_rad(ticks: u16) -> f64 {
    (ticks as f64 - 2048.0) * std::f64::consts::TAU / 4096.0
}

/// Sign-magnitude register (bit `sign_bit` set = negative), as speed and load use.
pub fn sign_magnitude(raw: u16, sign_bit: u32) -> i32 {
    let mag = (raw & ((1u16 << sign_bit) - 1)) as i32;
    if raw & (1u16 << sign_bit) != 0 {
        -mag
    } else {
        mag
    }
}

/// Speed units: position ticks per second.
pub fn speed_to_rad_s(raw: u16) -> f64 {
    sign_magnitude(raw, 15) as f64 * std::f64::consts::TAU / 4096.0
}

pub fn checksum(body: &[u8]) -> u8 {
    !body.iter().fold(0u8, |s, &b| s.wrapping_add(b))
}

/// One packet; `params.len() + 2` must fit the length byte.
pub fn instruction(id: u8, inst: u8, params: &[u8]) -> Result<Vec<u8>, String> {
    if params.len() > 253 {
        return Err("too many parameters for one packet".into());
    }
    let mut p = Vec::with_capacity(6 + params.len());
    p.extend_from_slice(&HEADER);
    p.push(id);
    p.push((params.len() + 2) as u8);
    p.push(inst);
    p.extend_from_slice(params);
    let c = checksum(&p[2..]);
    p.push(c);
    Ok(p)
}

pub fn ping(id: u8) -> Vec<u8> {
    instruction(id, PING, &[]).expect("no params")
}

pub fn read(id: u8, address: u8, size: u8) -> Vec<u8> {
    instruction(id, READ, &[address, size]).expect("two params")
}

pub fn write(id: u8, address: u8, data: &[u8]) -> Result<Vec<u8>, String> {
    let mut a = vec![address];
    a.extend_from_slice(data);
    instruction(id, WRITE, &a)
}

/// One broadcast packet that sets `address` on every servo: `(id, data)` of one size.
pub fn sync_write(address: u8, items: &[(u8, &[u8])]) -> Result<Vec<u8>, String> {
    let size = items.first().map_or(0, |(_, d)| d.len());
    if size == 0 || items.iter().any(|(_, d)| d.len() != size) {
        return Err("sync write needs data of one non-zero size per servo".into());
    }
    let mut a = vec![address, size as u8];
    for (id, d) in items {
        a.push(*id);
        a.extend_from_slice(d);
    }
    instruction(BROADCAST, SYNC_WRITE, &a)
}

pub fn sync_read(address: u8, size: u8, ids: &[u8]) -> Result<Vec<u8>, String> {
    let mut a = vec![address, size];
    a.extend_from_slice(ids);
    instruction(BROADCAST, SYNC_READ, &a)
}

/// Goal positions for all joints in one packet; targets in rad, inside one turn.
pub fn goal_positions(ids: &[u8], targets: &[f64]) -> Result<Vec<u8>, String> {
    if ids.len() != targets.len() {
        return Err("one target per servo id".into());
    }
    let ticks = targets
        .iter()
        .map(|&t| if t.is_finite() { rad_to_ticks(t) } else { None })
        .collect::<Option<Vec<u16>>>()
        .ok_or("targets must be finite and within one turn of the centre")?;
    let data: Vec<[u8; 2]> = ticks.iter().map(|t| t.to_le_bytes()).collect();
    let items: Vec<(u8, &[u8])> = ids.iter().zip(&data).map(|(&i, d)| (i, &d[..])).collect();
    sync_write(addr::GOAL_POSITION, &items)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Status {
    pub id: u8,
    /// Error bits: voltage, sensor, temperature, current, angle, overload.
    pub error: u8,
    pub params: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    Incomplete,
    BadHeader,
    BadChecksum,
}

/// Parses one status packet at the start of `buf`; returns it and the bytes used.
pub fn parse_status(buf: &[u8]) -> Result<(Status, usize), ParseError> {
    if buf.len() < 6 {
        return Err(ParseError::Incomplete);
    }
    if buf[0..2] != HEADER || buf[3] < 2 {
        return Err(ParseError::BadHeader);
    }
    let total = 4 + buf[3] as usize;
    if buf.len() < total {
        return Err(ParseError::Incomplete);
    }
    if checksum(&buf[2..total - 1]) != buf[total - 1] {
        return Err(ParseError::BadChecksum);
    }
    Ok((
        Status {
            id: buf[2],
            error: buf[4],
            params: buf[5..total - 1].to_vec(),
        },
        total,
    ))
}

pub fn le_u16(b: &[u8]) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(0..2)?.try_into().ok()?))
}

/// A simulated STS bus for tests: answers ping, read, write, sync read and sync
/// write from a byte-level memory table. Corrupt packets get no answer. A servo
/// with torque on is at its goal position the next time it is read (an ideal
/// servo); with torque off its position stays where the test put it.
#[derive(Debug, Clone)]
pub struct SimBus {
    pub tables: Vec<(u8, [u8; 128])>,
}

impl SimBus {
    pub fn new(ids: &[u8]) -> Self {
        SimBus {
            tables: ids.iter().map(|&i| (i, [0u8; 128])).collect(),
        }
    }

    fn table(&mut self, id: u8) -> Option<&mut [u8; 128]> {
        self.tables.iter_mut().find(|(i, _)| *i == id).map(|(_, t)| t)
    }

    fn status(id: u8, params: &[u8]) -> Vec<u8> {
        // Same framing as an instruction, with the error byte (0) in its place.
        instruction(id, 0, params).expect("table reads fit one packet")
    }

    pub fn transact(&mut self, packet: &[u8]) -> Vec<u8> {
        for (_, t) in self.tables.iter_mut() {
            if t[addr::TORQUE_ENABLE as usize] == 1 {
                let g = addr::GOAL_POSITION as usize;
                let p = addr::PRESENT_POSITION as usize;
                t.copy_within(g..g + 2, p);
            }
        }
        let n = packet.len();
        if n < 6
            || packet[0..2] != HEADER
            || packet[3] as usize + 4 != n
            || checksum(&packet[2..n - 1]) != packet[n - 1]
        {
            return Vec::new();
        }
        self.handle(packet[2], packet[4], &packet[5..n - 1]).unwrap_or_default()
    }

    fn handle(&mut self, id: u8, inst: u8, p: &[u8]) -> Option<Vec<u8>> {
        match inst {
            PING => self.table(id).map(|_| Self::status(id, &[])),
            READ => {
                let (a, s) = (*p.first()? as usize, *p.get(1)? as usize);
                let data = self.table(id)?.get(a..a + s)?.to_vec();
                Some(Self::status(id, &data))
            }
            WRITE => {
                let a = *p.first()? as usize;
                let data = p.get(1..)?;
                self.table(id)?.get_mut(a..a + data.len())?.copy_from_slice(data);
                Some(Self::status(id, &[]))
            }
            SYNC_WRITE => {
                let (a, s) = (*p.first()? as usize, *p.get(1)? as usize);
                for chunk in p.get(2..)?.chunks_exact(s + 1) {
                    if let Some(dst) = self.table(chunk[0]).and_then(|t| t.get_mut(a..a + s)) {
                        dst.copy_from_slice(&chunk[1..]);
                    }
                }
                None
            }
            SYNC_READ => {
                let (a, s) = (*p.first()? as usize, *p.get(1)? as usize);
                let mut out = Vec::new();
                for &i in p.get(2..)? {
                    if let Some(d) = self.table(i).and_then(|t| t.get(a..a + s)).map(<[u8]>::to_vec) {
                        out.extend(Self::status(i, &d));
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
        // Ping ID 1, and read ID 1 present position (0x38, 2 bytes), as in the
        // Feetech protocol manual's examples.
        assert_eq!(ping(1), [0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB]);
        assert_eq!(
            read(1, addr::PRESENT_POSITION, 2),
            [0xFF, 0xFF, 0x01, 0x04, 0x02, 0x38, 0x02, 0xBE]
        );
    }

    #[test]
    fn units_and_sign_magnitude() {
        assert_eq!(rad_to_ticks(0.0), Some(2048));
        assert_eq!(rad_to_ticks(std::f64::consts::FRAC_PI_2), Some(3072));
        assert_eq!(rad_to_ticks(4.0), None); // past one turn
        assert!((ticks_to_rad(1024) + std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert_eq!(sign_magnitude(0x8000 | 100, 15), -100);
        assert_eq!(sign_magnitude(100, 15), 100);
        assert_eq!(sign_magnitude(0x0400 | 7, 10), -7);
        assert!((speed_to_rad_s(4096) - std::f64::consts::TAU).abs() < 1e-12);
    }

    #[test]
    fn corrupt_and_short_status_rejected() {
        let pkt = SimBus::status(1, &[0x00, 0x08]);
        assert_eq!(parse_status(&pkt).unwrap().0.params, [0x00, 0x08]);
        assert_eq!(parse_status(&pkt[..pkt.len() - 1]), Err(ParseError::Incomplete));
        let mut bad = pkt.clone();
        bad[5] ^= 0x01;
        assert_eq!(parse_status(&bad), Err(ParseError::BadChecksum));
        let mut hdr = pkt.clone();
        hdr[0] = 0;
        assert_eq!(parse_status(&hdr), Err(ParseError::BadHeader));
    }

    #[test]
    fn goal_positions_reach_every_servo_through_the_sim_bus() {
        let ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut bus = SimBus::new(&ids);
        let targets: Vec<f64> = (0..10).map(|k| -0.5 + 0.1 * k as f64).collect();
        assert!(bus.transact(&goal_positions(&ids, &targets).unwrap()).is_empty());
        for (_, t) in bus.tables.iter_mut() {
            let g = [t[42], t[43]];
            t[56..58].copy_from_slice(&g);
        }
        let reply = bus.transact(&sync_read(addr::PRESENT_POSITION, 2, &ids).unwrap());
        let mut at = 0;
        for (k, &id) in ids.iter().enumerate() {
            let (st, used) = parse_status(&reply[at..]).unwrap();
            at += used;
            assert_eq!(st.id, id);
            let q = ticks_to_rad(le_u16(&st.params).unwrap());
            assert!((q - targets[k]).abs() <= std::f64::consts::TAU / 4096.0);
        }
        assert_eq!(at, reply.len());
    }

    #[test]
    fn bad_inputs_are_refused() {
        assert!(goal_positions(&[1, 2], &[0.0]).is_err());
        assert!(goal_positions(&[1], &[f64::NAN]).is_err());
        assert!(goal_positions(&[1], &[4.0]).is_err());
        assert!(sync_write(42, &[(1, &[0, 0][..]), (2, &[0][..])]).is_err());
        assert!(instruction(1, WRITE, &[0u8; 254]).is_err());
        let mut bus = SimBus::new(&[1]);
        let mut pkt = write(1, addr::TORQUE_ENABLE, &[1]).unwrap();
        let k = pkt.len() - 1;
        pkt[k] ^= 0xFF;
        assert!(bus.transact(&pkt).is_empty()); // corrupt: no reply, no write
        assert_eq!(bus.tables[0].1[40], 0);
        let ok = bus.transact(&write(1, addr::TORQUE_ENABLE, &[1]).unwrap());
        assert_eq!(parse_status(&ok).unwrap().0.error, 0);
        assert_eq!(bus.tables[0].1[40], 1);
        assert!(bus.transact(&read(1, 120, 10)).is_empty());
        assert!(bus.transact(&write(1, 127, &[1, 2]).unwrap()).is_empty());
    }
}
