//! STS serial bus backend: sends `sts` packets over a port and reads the
//! replies with a timeout and a retry policy (WP5-01).
//!
//! A `Port` moves bytes. Two exist:
//! - `SimPort` wraps the simulated bus (`sts::SimBus`) and can drop or corrupt
//!   replies on purpose, so the retry and resync paths are tested in CI.
//! - `StreamPort` runs over any byte stream with a reader thread, so reads time
//!   out with std only. `open_device` opens a serial device file with it. The
//!   line settings (1 Mbit/s, 8N1, raw) are set outside this crate, for example
//!   `stty -F /dev/ttyUSB0 1000000 raw -echo` on Linux; nothing here has run
//!   against a real servo yet (Phase 5 stage 2, `env: hw`).
//!
//! `Bus` is what the bench tools and the hardware state source use: ping, scan,
//! register read/write, goal positions for all joints in one sync write, and
//! present position and speed for all joints in one sync read. It only moves
//! servo angles; the joint map (`jointmap`) turns them into sim angles.

use crate::sts;
use std::collections::VecDeque;
use std::io::{self, Read, Write};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Moves bytes to and from the bus.
pub trait Port {
    fn send(&mut self, bytes: &[u8]) -> io::Result<()>;
    /// Appends whatever arrives before `deadline` (at least one chunk if any
    /// arrives); returns false when nothing arrived in time.
    fn recv(&mut self, deadline: Instant, buf: &mut Vec<u8>) -> io::Result<bool>;
    /// Drops bytes still waiting from an earlier exchange.
    fn clear(&mut self);
}

/// What the simulated port does with the reply to the next packet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fault {
    /// The servo does not answer.
    Drop,
    /// One byte of the reply is flipped.
    Corrupt,
    /// Line noise arrives before the reply.
    Garbage,
}

/// The simulated bus as a port, with a script of faults for the next packets.
#[derive(Debug, Clone)]
pub struct SimPort {
    pub bus: sts::SimBus,
    pub faults: VecDeque<Fault>,
    pending: Vec<u8>,
}

impl SimPort {
    pub fn new(bus: sts::SimBus) -> Self {
        SimPort {
            bus,
            faults: VecDeque::new(),
            pending: Vec::new(),
        }
    }
}

impl Port for SimPort {
    fn send(&mut self, bytes: &[u8]) -> io::Result<()> {
        let mut reply = self.bus.transact(bytes);
        match self.faults.pop_front() {
            Some(Fault::Drop) => reply.clear(),
            Some(Fault::Corrupt) if reply.len() > 5 => reply[5] ^= 0x5A,
            Some(Fault::Garbage) => reply = [&[0x00, 0xFF, 0x13][..], &reply].concat(),
            _ => {}
        }
        self.pending.extend(reply);
        Ok(())
    }

    fn recv(&mut self, _deadline: Instant, buf: &mut Vec<u8>) -> io::Result<bool> {
        if self.pending.is_empty() {
            return Ok(false);
        }
        buf.append(&mut self.pending);
        Ok(true)
    }

    fn clear(&mut self) {
        self.pending.clear();
    }
}

/// A port over a byte stream; a reader thread turns blocking reads into reads
/// with a deadline.
pub struct StreamPort<W: Write> {
    writer: W,
    rx: mpsc::Receiver<Vec<u8>>,
    /// Adapters that wire TX to RX (half duplex without direction control)
    /// hear their own packet first; it is skipped when this is set.
    pub echo: bool,
    skip: usize,
}

impl<W: Write> StreamPort<W> {
    pub fn new<R: Read + Send + 'static>(mut reader: R, writer: W) -> Self {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut chunk = [0u8; 256];
            loop {
                match reader.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        if tx.send(chunk[..n].to_vec()).is_err() {
                            break;
                        }
                    }
                }
            }
        });
        StreamPort {
            writer,
            rx,
            echo: false,
            skip: 0,
        }
    }
}

impl<W: Write> Port for StreamPort<W> {
    fn send(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.writer.write_all(bytes)?;
        self.writer.flush()?;
        if self.echo {
            self.skip += bytes.len();
        }
        Ok(())
    }

    fn recv(&mut self, deadline: Instant, buf: &mut Vec<u8>) -> io::Result<bool> {
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let mut chunk = match self.rx.recv_timeout(left) {
                Ok(c) => c,
                Err(mpsc::RecvTimeoutError::Timeout) => return Ok(false),
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(io::Error::new(io::ErrorKind::BrokenPipe, "bus closed"))
                }
            };
            let k = self.skip.min(chunk.len());
            self.skip -= k;
            chunk.drain(..k);
            if !chunk.is_empty() {
                buf.extend(chunk);
                return Ok(true);
            }
        }
    }

    fn clear(&mut self) {
        while self.rx.try_recv().is_ok() {}
        self.skip = 0;
    }
}

/// Opens a serial device file (already set to the bus baud rate, raw mode).
pub fn open_device(path: &str) -> io::Result<StreamPort<std::fs::File>> {
    let file = std::fs::OpenOptions::new().read(true).write(true).open(path)?;
    let reader = file.try_clone()?;
    Ok(StreamPort::new(reader, file))
}

#[derive(Debug, Clone, PartialEq)]
pub enum BusError {
    /// No valid reply from these ids after every retry.
    Timeout {
        missing: Vec<u8>,
    },
    Io(String),
    /// The request itself is invalid (bad target, too many ids).
    Request(String),
}

impl std::fmt::Display for BusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BusError::Timeout { missing } => write!(f, "no reply from servo id(s) {missing:?}"),
            BusError::Io(e) => write!(f, "bus i/o: {e}"),
            BusError::Request(e) => write!(f, "bad request: {e}"),
        }
    }
}

/// Counters for the telemetry and the stage 2 pass criterion (no comm errors).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BusStats {
    pub exchanges: u64,
    pub retries: u64,
    pub timeouts: u64,
    pub bad_checksum: u64,
    /// Bytes dropped while looking for a packet header.
    pub resync_bytes: u64,
    pub unexpected_id: u64,
}

/// One servo's present position and speed (servo frame) and its error bits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ServoState {
    pub id: u8,
    pub pos_rad: f64,
    pub vel_rad_s: f64,
    pub error: u8,
}

pub struct Bus<P: Port> {
    pub port: P,
    /// Time to wait for all replies of one exchange.
    pub timeout: Duration,
    /// Extra attempts after the first one.
    pub retries: u32,
    pub stats: BusStats,
}

impl<P: Port> Bus<P> {
    pub fn new(port: P) -> Self {
        Bus {
            port,
            // 10 status packets of 10 bytes at 1 Mbit/s take ~1 ms; servos
            // answer after a return delay of up to 0.5 ms.
            timeout: Duration::from_millis(5),
            retries: 2,
            stats: BusStats::default(),
        }
    }

    /// Sends `packet` and collects one status from each of `ids`, in order.
    /// Retries the whole exchange; a broken packet is skipped, not trusted.
    pub fn exchange(&mut self, packet: &[u8], ids: &[u8]) -> Result<Vec<sts::Status>, BusError> {
        let mut missing = ids.to_vec();
        for attempt in 0..=self.retries {
            if attempt > 0 {
                self.stats.retries += 1;
            }
            self.stats.exchanges += 1;
            self.port.clear();
            self.port.send(packet).map_err(|e| BusError::Io(e.to_string()))?;
            let got = self.collect(ids)?;
            if got.iter().all(Option::is_some) {
                return Ok(got.into_iter().flatten().collect());
            }
            self.stats.timeouts += 1;
            missing = ids
                .iter()
                .zip(&got)
                .filter(|(_, s)| s.is_none())
                .map(|(&i, _)| i)
                .collect();
        }
        Err(BusError::Timeout { missing })
    }

    fn collect(&mut self, ids: &[u8]) -> Result<Vec<Option<sts::Status>>, BusError> {
        let mut got: Vec<Option<sts::Status>> = vec![None; ids.len()];
        let deadline = Instant::now() + self.timeout;
        let mut buf = Vec::new();
        while got.iter().any(Option::is_none) {
            match sts::parse_status(&buf) {
                Ok((st, used)) => {
                    buf.drain(..used);
                    match ids.iter().position(|&i| i == st.id) {
                        Some(k) if got[k].is_none() => got[k] = Some(st),
                        _ => self.stats.unexpected_id += 1,
                    }
                    continue;
                }
                Err(sts::ParseError::BadHeader) => {
                    buf.remove(0);
                    self.stats.resync_bytes += 1;
                    continue;
                }
                Err(sts::ParseError::BadChecksum) => {
                    // Drop the header and look for the next packet.
                    buf.drain(..2);
                    self.stats.bad_checksum += 1;
                    self.stats.resync_bytes += 2;
                    continue;
                }
                Err(sts::ParseError::Incomplete) => {}
            }
            if !self
                .port
                .recv(deadline, &mut buf)
                .map_err(|e| BusError::Io(e.to_string()))?
            {
                break;
            }
        }
        Ok(got)
    }

    pub fn ping(&mut self, id: u8) -> Result<sts::Status, BusError> {
        Ok(self.exchange(&sts::ping(id), &[id])?.remove(0))
    }

    /// Ids in `candidates` that answer a ping (one attempt each, no retries).
    pub fn scan(&mut self, candidates: impl IntoIterator<Item = u8>) -> Vec<u8> {
        let retries = std::mem::replace(&mut self.retries, 0);
        let found = candidates
            .into_iter()
            .filter(|&id| id != sts::BROADCAST && self.ping(id).is_ok())
            .collect();
        self.retries = retries;
        found
    }

    pub fn read(&mut self, id: u8, address: u8, size: u8) -> Result<Vec<u8>, BusError> {
        let st = self.exchange(&sts::read(id, address, size), &[id])?.remove(0);
        if st.params.len() != size as usize {
            return Err(BusError::Request(format!(
                "servo {id} sent {} bytes, asked {size}",
                st.params.len()
            )));
        }
        Ok(st.params)
    }

    /// Register write with an acknowledgement (retried: writes are idempotent).
    pub fn write(&mut self, id: u8, address: u8, data: &[u8]) -> Result<(), BusError> {
        let pkt = sts::write(id, address, data).map_err(BusError::Request)?;
        self.exchange(&pkt, &[id]).map(drop)
    }

    /// Goal positions (servo frame, rad) for all servos in one broadcast
    /// packet. Broadcast gets no reply, so nothing is retried here; the next
    /// state read shows whether the servos follow.
    pub fn set_goals(&mut self, ids: &[u8], servo_rad: &[f64]) -> Result<(), BusError> {
        let pkt = sts::goal_positions(ids, servo_rad).map_err(BusError::Request)?;
        self.port.clear();
        self.port.send(&pkt).map_err(|e| BusError::Io(e.to_string()))
    }

    /// Torque on or off for all servos in one broadcast packet (no reply).
    pub fn set_torque(&mut self, ids: &[u8], on: bool) -> Result<(), BusError> {
        let v = [on as u8];
        let items: Vec<(u8, &[u8])> = ids.iter().map(|&i| (i, &v[..])).collect();
        let pkt = sts::sync_write(sts::addr::TORQUE_ENABLE, &items).map_err(BusError::Request)?;
        self.port.clear();
        self.port.send(&pkt).map_err(|e| BusError::Io(e.to_string()))
    }

    /// Present position and speed of all servos in one sync read.
    pub fn read_state(&mut self, ids: &[u8]) -> Result<Vec<ServoState>, BusError> {
        let pkt = sts::sync_read(sts::addr::PRESENT_POSITION, 4, ids).map_err(BusError::Request)?;
        self.exchange(&pkt, ids)?
            .into_iter()
            .map(|st| {
                let p = &st.params;
                match (sts::le_u16(p), p.get(2..).and_then(sts::le_u16)) {
                    (Some(pos), Some(vel)) if p.len() == 4 => Ok(ServoState {
                        id: st.id,
                        pos_rad: sts::ticks_to_rad(pos),
                        vel_rad_s: sts::speed_to_rad_s(vel),
                        error: st.error,
                    }),
                    _ => Err(BusError::Request(format!(
                        "servo {} sent {} bytes, asked 4",
                        st.id,
                        p.len()
                    ))),
                }
            })
            .collect()
    }

    /// Bus voltage (V) and temperature (deg C) of one servo.
    pub fn read_health(&mut self, id: u8) -> Result<(f64, f64), BusError> {
        let d = self.read(id, sts::addr::PRESENT_VOLTAGE, 2)?;
        Ok((d[0] as f64 * 0.1, d[1] as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sts::addr;

    const IDS: [u8; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    fn sim_bus() -> Bus<SimPort> {
        let mut sim = sts::SimBus::new(&IDS);
        for (k, (_, t)) in sim.tables.iter_mut().enumerate() {
            t[addr::PRESENT_VOLTAGE as usize] = 74; // 7.4 V
            t[addr::PRESENT_TEMPERATURE as usize] = 30 + k as u8;
        }
        Bus::new(SimPort::new(sim))
    }

    /// The servo "moves" to its goal and reports a speed.
    fn settle(bus: &mut Bus<SimPort>, speed: u16) {
        for (_, t) in bus.port.bus.tables.iter_mut() {
            let g = [t[42], t[43]];
            t[56..58].copy_from_slice(&g);
            t[58..60].copy_from_slice(&speed.to_le_bytes());
        }
    }

    #[test]
    fn ping_scan_read_write() {
        let mut bus = sim_bus();
        assert_eq!(bus.ping(3).unwrap().id, 3);
        assert_eq!(bus.scan(0..=20), IDS.to_vec());
        assert_eq!(bus.retries, 2, "scan restores the retry count");
        bus.write(4, addr::TORQUE_ENABLE, &[1]).unwrap();
        assert_eq!(bus.read(4, addr::TORQUE_ENABLE, 1).unwrap(), [1]);
        let (v, temp) = bus.read_health(2).unwrap();
        assert!((v - 7.4).abs() < 1e-9 && temp == 31.0);
        assert_eq!(bus.stats.retries, 0);
    }

    #[test]
    fn goals_and_state_for_all_joints() {
        let mut bus = sim_bus();
        let goals: Vec<f64> = (0..10).map(|k| -0.45 + 0.1 * k as f64).collect();
        bus.set_goals(&IDS, &goals).unwrap();
        settle(&mut bus, 0x8000 | 512); // -512 ticks/s
        let st = bus.read_state(&IDS).unwrap();
        for (k, s) in st.iter().enumerate() {
            assert_eq!(s.id, IDS[k]);
            assert!((s.pos_rad - goals[k]).abs() <= std::f64::consts::TAU / 4096.0);
            assert!((s.vel_rad_s + std::f64::consts::FRAC_PI_4).abs() < 1e-12);
        }
        assert_eq!(bus.stats.exchanges, 1);
        assert!(bus.set_goals(&IDS, &[0.0; 9]).is_err());
        assert!(bus.set_goals(&[1], &[f64::NAN]).is_err());
    }

    #[test]
    fn dropped_corrupt_and_noisy_replies_are_retried() {
        let mut bus = sim_bus();
        bus.port.faults.extend([Fault::Drop, Fault::Corrupt]);
        assert_eq!(bus.read_state(&IDS).unwrap().len(), 10);
        assert_eq!(bus.stats.retries, 2);
        assert_eq!(bus.stats.timeouts, 2);
        assert_eq!(bus.stats.bad_checksum, 1);

        bus.port.faults.push_back(Fault::Garbage);
        bus.ping(5).unwrap();
        assert_eq!(bus.stats.retries, 2, "noise before a good reply needs no retry");
        assert!(bus.stats.resync_bytes >= 3);
    }

    #[test]
    fn missing_servo_times_out_after_every_retry() {
        let mut bus = sim_bus();
        bus.port.bus.tables.retain(|(id, _)| *id != 7);
        let err = bus.read_state(&IDS).unwrap_err();
        assert_eq!(err, BusError::Timeout { missing: vec![7] });
        assert_eq!(bus.stats.exchanges, 3);
        assert!(err.to_string().contains('7'));
        bus.port.faults.extend([Fault::Drop; 3]);
        assert!(bus.ping(1).is_err());
        assert!(bus.ping(1).is_ok(), "faults are used up");
    }

    /// A servo bus on the far end of a socket, so `StreamPort` (the code that
    /// runs over a device file) is tested with real blocking reads and timeouts.
    #[cfg(unix)]
    fn stream_bus(echo: bool) -> Bus<StreamPort<std::os::unix::net::UnixStream>> {
        use std::os::unix::net::UnixStream;
        let (host, mut dev) = UnixStream::pair().unwrap();
        std::thread::spawn(move || {
            let mut sim = sts::SimBus::new(&[1, 2]);
            sim.tables[1].1[56..60].copy_from_slice(&[0x00, 0x0C, 0x10, 0x00]);
            let mut buf = Vec::new();
            let mut chunk = [0u8; 64];
            while let Ok(n) = dev.read(&mut chunk) {
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
                while buf.len() >= 4 && buf.len() >= 4 + buf[3] as usize {
                    let pkt: Vec<u8> = buf.drain(..4 + buf[3] as usize).collect();
                    let mut out = if echo { pkt.clone() } else { Vec::new() };
                    out.extend(sim.transact(&pkt));
                    // Reply in two pieces, as a USB adapter may deliver it.
                    let (a, b) = out.split_at(out.len() / 2);
                    dev.write_all(a).unwrap();
                    dev.write_all(b).unwrap();
                }
            }
        });
        let mut port = StreamPort::new(host.try_clone().unwrap(), host);
        port.echo = echo;
        let mut bus = Bus::new(port);
        bus.timeout = Duration::from_millis(200);
        bus
    }

    #[cfg(unix)]
    #[test]
    fn stream_port_transacts_and_times_out() {
        for echo in [false, true] {
            let mut bus = stream_bus(echo);
            assert_eq!(bus.scan(0..=3), vec![1, 2]);
            let st = bus.read_state(&[1, 2]).unwrap();
            assert!((st[1].pos_rad - std::f64::consts::FRAC_PI_2).abs() < 1e-12); // 3072 ticks
            assert!((st[1].vel_rad_s - 16.0 * std::f64::consts::TAU / 4096.0).abs() < 1e-12);
            bus.retries = 0;
            let t0 = Instant::now();
            assert_eq!(bus.ping(9), Err(BusError::Timeout { missing: vec![9] }));
            assert!(t0.elapsed() >= Duration::from_millis(200));
        }
    }

    #[test]
    fn opening_a_missing_device_fails_cleanly() {
        assert!(open_device("/nonexistent/biped-bus").is_err());
    }
}
