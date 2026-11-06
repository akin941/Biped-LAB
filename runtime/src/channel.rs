//! Host command channel (WP4-14): velocity commands over UDP from a host
//! program (teleop keyboard, later a gamepad or a planner).
//!
//! Every packet goes through the command gate (ADR-0006), which clamps the
//! range, limits the rate, and stops the robot when no fresh command has arrived
//! for 0.5 s. A sender only has to keep sending; silence means stop.
//!
//! ```text
//! packet (34 bytes, little-endian): magic "BPCM" | version u16 | seq u32 | vx f64 | vy f64 | wz f64
//! ```
//!
//! The channel locks to the first sender's address: packets from any other
//! address are dropped and counted, so a second program on the network cannot
//! take over a running robot. A packet whose `seq` is not newer than the last
//! accepted one (a duplicate or reordered packet) is dropped. The channel carries
//! no stop request: stopping is the gate timeout or the e-stop (S-3), never a
//! packet that could be lost.

use std::io::ErrorKind;
use std::net::{SocketAddr, UdpSocket};

use crate::gate::Command;

pub const MAGIC: &[u8; 4] = b"BPCM";
pub const VERSION: u16 = 1;
pub const PACKET_SIZE: usize = 4 + 2 + 4 + 3 * 8;
/// Datagrams read per control period at most, so a flood cannot stall the loop.
pub const MAX_PER_POLL: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Packet {
    pub seq: u32,
    pub cmd: Command,
}

pub fn encode(p: &Packet) -> [u8; PACKET_SIZE] {
    let mut b = [0u8; PACKET_SIZE];
    b[0..4].copy_from_slice(MAGIC);
    b[4..6].copy_from_slice(&VERSION.to_le_bytes());
    b[6..10].copy_from_slice(&p.seq.to_le_bytes());
    for (k, v) in [p.cmd.vx, p.cmd.vy, p.cmd.wz].iter().enumerate() {
        b[10 + 8 * k..18 + 8 * k].copy_from_slice(&v.to_le_bytes());
    }
    b
}

pub fn decode(b: &[u8]) -> Option<Packet> {
    if b.len() != PACKET_SIZE || &b[0..4] != MAGIC || b[4..6] != VERSION.to_le_bytes() {
        return None;
    }
    let f = |k: usize| f64::from_le_bytes(b[10 + 8 * k..18 + 8 * k].try_into().expect("8 bytes"));
    Some(Packet {
        seq: u32::from_le_bytes(b[6..10].try_into().expect("4 bytes")),
        cmd: Command {
            vx: f(0),
            vy: f(1),
            wz: f(2),
        },
    })
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct ChannelStats {
    pub accepted: u64,
    pub malformed: u64,
    pub stale: u64,
    pub foreign: u64,
}

/// Packet filter, separate from the socket so it can be tested without one.
#[derive(Debug, Default)]
pub struct Filter {
    peer: Option<SocketAddr>,
    last_seq: Option<u32>,
    pub stats: ChannelStats,
}

impl Filter {
    /// The command in `buf` from `from`, if it is accepted.
    pub fn accept(&mut self, buf: &[u8], from: SocketAddr) -> Option<Command> {
        if self.peer.is_some_and(|p| p != from) {
            self.stats.foreign += 1;
            return None;
        }
        let Some(p) = decode(buf) else {
            self.stats.malformed += 1;
            return None;
        };
        if self.last_seq.is_some_and(|s| p.seq <= s) {
            self.stats.stale += 1;
            return None;
        }
        self.peer.get_or_insert(from);
        self.last_seq = Some(p.seq);
        self.stats.accepted += 1;
        Some(p.cmd)
    }

    pub fn peer(&self) -> Option<SocketAddr> {
        self.peer
    }
}

pub struct CommandChannel {
    socket: UdpSocket,
    pub filter: Filter,
}

impl CommandChannel {
    pub fn bind(addr: &str) -> std::io::Result<Self> {
        let socket = UdpSocket::bind(addr)?;
        socket.set_nonblocking(true)?;
        Ok(CommandChannel {
            socket,
            filter: Filter::default(),
        })
    }

    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.socket.local_addr()
    }

    /// Reads the waiting datagrams (at most `MAX_PER_POLL`) and returns the newest
    /// accepted command, if any. Never blocks.
    pub fn poll(&mut self) -> Option<Command> {
        let mut buf = [0u8; 64];
        let mut newest = None;
        for _ in 0..MAX_PER_POLL {
            match self.socket.recv_from(&mut buf) {
                Ok((len, from)) => {
                    if let Some(c) = self.filter.accept(&buf[..len], from) {
                        newest = Some(c);
                    }
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                Err(_) => {
                    // e.g. ICMP port unreachable reported on some platforms: not a packet.
                    self.filter.stats.malformed += 1;
                    break;
                }
            }
        }
        newest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(port: u16) -> SocketAddr {
        SocketAddr::from(([127, 0, 0, 1], port))
    }

    fn pkt(seq: u32, vx: f64) -> [u8; PACKET_SIZE] {
        encode(&Packet {
            seq,
            cmd: Command { vx, vy: 0.0, wz: 0.0 },
        })
    }

    #[test]
    fn round_trip_and_size() {
        let p = Packet {
            seq: 9,
            cmd: Command {
                vx: 0.1,
                vy: -0.2,
                wz: 0.3,
            },
        };
        let b = encode(&p);
        assert_eq!(b.len(), 34);
        assert_eq!(decode(&b), Some(p));
        assert_eq!(decode(&b[..33]), None);
        let mut bad = b;
        bad[4] = 2;
        assert_eq!(decode(&bad), None);
    }

    #[test]
    fn filter_drops_stale_foreign_and_malformed() {
        let mut f = Filter::default();
        assert!(f.accept(&pkt(5, 0.1), addr(1000)).is_some());
        assert!(f.accept(&pkt(5, 0.2), addr(1000)).is_none()); // duplicate
        assert!(f.accept(&pkt(4, 0.2), addr(1000)).is_none()); // reordered
        assert!(f.accept(&pkt(9, 0.2), addr(2000)).is_none()); // another sender
        assert!(f.accept(b"junk", addr(1000)).is_none());
        assert_eq!(f.accept(&pkt(6, 0.25), addr(1000)).map(|c| c.vx), Some(0.25));
        assert_eq!(
            f.stats,
            ChannelStats {
                accepted: 2,
                malformed: 1,
                stale: 2,
                foreign: 1
            }
        );
        assert_eq!(f.peer(), Some(addr(1000)));
    }

    #[test]
    fn malformed_first_packet_does_not_lock_the_peer() {
        let mut f = Filter::default();
        assert!(f.accept(b"junk", addr(3000)).is_none());
        assert!(f.accept(&pkt(1, 0.1), addr(1000)).is_some());
        assert_eq!(f.peer(), Some(addr(1000)));
    }

    #[test]
    fn socket_poll_returns_the_newest_command() {
        let mut ch = CommandChannel::bind("127.0.0.1:0").unwrap();
        assert_eq!(ch.poll(), None);
        let tx = UdpSocket::bind("127.0.0.1:0").unwrap();
        let to = ch.local_addr().unwrap();
        for (s, v) in [(1, 0.1), (3, 0.3), (2, 0.2)] {
            tx.send_to(&pkt(s, v), to).unwrap();
        }
        let mut got = None;
        for _ in 0..100 {
            if let Some(c) = ch.poll() {
                got = Some(c);
            }
            if ch.filter.stats.accepted + ch.filter.stats.stale == 3 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(got.map(|c| c.vx), Some(0.3));
        assert_eq!(ch.filter.stats.stale, 1);
    }
}
