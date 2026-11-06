//! Host command channel (WP4-14, authenticated in WP4-16): velocity commands
//! over UDP from a host program (keyboard teleop, later a gamepad or a planner).
//!
//! Every accepted command goes through the command gate (ADR-0006), which
//! clamps the range, limits the rate and stops the robot when no fresh command
//! has arrived for 0.5 s. A sender only has to keep sending; silence means stop.
//! No stop request travels over the channel: stopping is the gate timeout or the
//! e-stop (S-3), never a packet that could be lost.
//!
//! Authentication (WP4-16, `docs/decisions/ADR-0014-command-channel-auth.md`):
//!
//! ```text
//! hello   host -> robot  14 B: "BPCH" | ver u16 | client_nonce u64
//! welcome robot -> host  38 B: "BPCN" | ver u16 | client_nonce u64 | session u64 | tag[16]
//! command host -> robot  62 B: "BPCM" | ver u16 | session u64 | seq u64 | vx vy wz f64 | tag[16]
//! tag = first 16 bytes of HMAC-SHA256(key, all bytes before the tag); little-endian
//! ```
//!
//! The runtime draws a random `session` at start. A command counts only with the
//! current session, a valid tag and a `seq` above the last accepted one, so
//! packets recorded earlier, including in a previous run, are refused. The first
//! sender with a valid command becomes the only accepted peer. Without a key
//! file the key is all zeros, and the runtime then refuses to bind anywhere but
//! loopback.

use std::io::ErrorKind;
use std::net::{SocketAddr, UdpSocket};

use crate::auth::{ct_eq, hmac_sha256};
use crate::gate::Command;

pub const VERSION: u16 = 2;
pub const HELLO: &[u8; 4] = b"BPCH";
pub const WELCOME: &[u8; 4] = b"BPCN";
pub const COMMAND: &[u8; 4] = b"BPCM";
pub const HELLO_SIZE: usize = 14;
pub const WELCOME_SIZE: usize = 38;
pub const COMMAND_SIZE: usize = 62;
pub const TAG: usize = 16;
pub const MIN_KEY: usize = 16;
/// Datagrams read per control period at most, so a flood cannot stall the loop.
pub const MAX_PER_POLL: usize = 64;
/// Welcome replies per poll at most (no amplification: a welcome is 38 B for a 14 B hello).
pub const MAX_WELCOMES_PER_POLL: usize = 2;

fn tag(key: &[u8], msg: &[u8]) -> [u8; TAG] {
    hmac_sha256(key, msg)[..TAG].try_into().expect("16 bytes")
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes"))
}

pub fn encode_hello(client_nonce: u64) -> [u8; HELLO_SIZE] {
    let mut b = [0u8; HELLO_SIZE];
    b[0..4].copy_from_slice(HELLO);
    b[4..6].copy_from_slice(&VERSION.to_le_bytes());
    b[6..14].copy_from_slice(&client_nonce.to_le_bytes());
    b
}

pub fn encode_welcome(key: &[u8], client_nonce: u64, session: u64) -> [u8; WELCOME_SIZE] {
    let mut b = [0u8; WELCOME_SIZE];
    b[0..4].copy_from_slice(WELCOME);
    b[4..6].copy_from_slice(&VERSION.to_le_bytes());
    b[6..14].copy_from_slice(&client_nonce.to_le_bytes());
    b[14..22].copy_from_slice(&session.to_le_bytes());
    let t = tag(key, &b[..22]);
    b[22..].copy_from_slice(&t);
    b
}

pub fn encode_command(key: &[u8], session: u64, seq: u64, c: Command) -> [u8; COMMAND_SIZE] {
    let mut b = [0u8; COMMAND_SIZE];
    b[0..4].copy_from_slice(COMMAND);
    b[4..6].copy_from_slice(&VERSION.to_le_bytes());
    b[6..14].copy_from_slice(&session.to_le_bytes());
    b[14..22].copy_from_slice(&seq.to_le_bytes());
    for (k, v) in [c.vx, c.vy, c.wz].iter().enumerate() {
        b[22 + 8 * k..30 + 8 * k].copy_from_slice(&v.to_le_bytes());
    }
    let t = tag(key, &b[..46]);
    b[46..].copy_from_slice(&t);
    b
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct ChannelStats {
    pub accepted: u64,
    /// Not a hello or command of this version, or the wrong size.
    pub malformed: u64,
    /// Bad tag or another session: wrong key, tampered, or recorded earlier.
    pub unauthenticated: u64,
    /// Valid but not newer than the last accepted command (duplicate, reordered, replayed).
    pub stale: u64,
    /// Valid, but from another address than the locked peer.
    pub foreign: u64,
    pub hellos: u64,
}

pub enum Verdict {
    Command(Command),
    /// Answer `from` with this welcome.
    Welcome([u8; WELCOME_SIZE]),
    Drop,
}

/// Packet checks, separate from the socket so they can be tested without one.
pub struct Filter {
    key: Vec<u8>,
    session: u64,
    peer: Option<SocketAddr>,
    last_seq: Option<u64>,
    pub stats: ChannelStats,
}

impl Filter {
    pub fn new(key: Vec<u8>, session: u64) -> Self {
        Filter {
            key,
            session,
            peer: None,
            last_seq: None,
            stats: ChannelStats::default(),
        }
    }

    pub fn session(&self) -> u64 {
        self.session
    }

    pub fn peer(&self) -> Option<SocketAddr> {
        self.peer
    }

    pub fn check(&mut self, b: &[u8], from: SocketAddr) -> Verdict {
        let versioned = b.len() >= 6 && b[4..6] == VERSION.to_le_bytes();
        if versioned && b.len() == HELLO_SIZE && &b[0..4] == HELLO {
            self.stats.hellos += 1;
            return Verdict::Welcome(encode_welcome(&self.key, u64_at(b, 6), self.session));
        }
        if !(versioned && b.len() == COMMAND_SIZE && &b[0..4] == COMMAND) {
            self.stats.malformed += 1;
            return Verdict::Drop;
        }
        if !ct_eq(&tag(&self.key, &b[..46]), &b[46..]) || u64_at(b, 6) != self.session {
            self.stats.unauthenticated += 1;
            return Verdict::Drop;
        }
        if self.peer.is_some_and(|p| p != from) {
            self.stats.foreign += 1;
            return Verdict::Drop;
        }
        let seq = u64_at(b, 14);
        if self.last_seq.is_some_and(|s| seq <= s) {
            self.stats.stale += 1;
            return Verdict::Drop;
        }
        let f = |k: usize| f64::from_le_bytes(b[22 + 8 * k..30 + 8 * k].try_into().expect("8 bytes"));
        self.peer.get_or_insert(from);
        self.last_seq = Some(seq);
        self.stats.accepted += 1;
        Verdict::Command(Command {
            vx: f(0),
            vy: f(1),
            wz: f(2),
        })
    }
}

/// Reads a key file: hex, at least `MIN_KEY` bytes.
pub fn load_key(path: &str) -> Result<Vec<u8>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("command key {path:?}: {e}"))?;
    let key = crate::auth::parse_hex(&text).ok_or(format!("command key {path:?}: not hex"))?;
    if key.len() < MIN_KEY {
        return Err(format!(
            "command key {path:?}: at least {MIN_KEY} bytes ({} hex digits)",
            2 * MIN_KEY
        ));
    }
    Ok(key)
}

pub struct CommandChannel {
    socket: UdpSocket,
    pub filter: Filter,
}

impl CommandChannel {
    /// Binds `addr`. Without a key only a loopback address is allowed.
    pub fn bind(addr: &str, key: Option<Vec<u8>>, session: u64) -> Result<Self, String> {
        let socket = UdpSocket::bind(addr).map_err(|e| format!("command channel {addr:?}: {e}"))?;
        let local = socket.local_addr().map_err(|e| e.to_string())?;
        if key.is_none() && !local.ip().is_loopback() {
            return Err(format!(
                "command channel {addr:?}: a non-loopback address needs --command-key (WP4-16)"
            ));
        }
        socket.set_nonblocking(true).map_err(|e| e.to_string())?;
        Ok(CommandChannel {
            socket,
            filter: Filter::new(key.unwrap_or_else(|| vec![0u8; 32]), session),
        })
    }

    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.socket.local_addr()
    }

    /// Reads the waiting datagrams (at most `MAX_PER_POLL`), answers hellos, and
    /// returns the newest accepted command, if any. Never blocks.
    pub fn poll(&mut self) -> Option<Command> {
        let mut buf = [0u8; 128];
        let mut newest = None;
        let mut welcomes = 0;
        for _ in 0..MAX_PER_POLL {
            match self.socket.recv_from(&mut buf) {
                Ok((len, from)) => match self.filter.check(&buf[..len], from) {
                    Verdict::Command(c) => newest = Some(c),
                    Verdict::Welcome(w) if welcomes < MAX_WELCOMES_PER_POLL => {
                        welcomes += 1;
                        let _ = self.socket.send_to(&w, from); // best effort, never blocks the loop
                    }
                    _ => {}
                },
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

    const KEY: &[u8] = b"0123456789abcdef0123456789abcdef";

    fn addr(port: u16) -> SocketAddr {
        SocketAddr::from(([127, 0, 0, 1], port))
    }

    fn cmd(vx: f64) -> Command {
        Command { vx, vy: 0.0, wz: 0.0 }
    }

    fn vx(v: Verdict) -> Option<f64> {
        match v {
            Verdict::Command(c) => Some(c.vx),
            _ => None,
        }
    }

    #[test]
    fn sizes() {
        assert_eq!(encode_hello(1).len(), 14);
        assert_eq!(encode_welcome(KEY, 1, 2).len(), 38);
        assert_eq!(encode_command(KEY, 1, 2, cmd(0.1)).len(), 62);
    }

    #[test]
    fn hello_gets_an_authenticated_welcome() {
        let mut f = Filter::new(KEY.to_vec(), 77);
        let Verdict::Welcome(w) = f.check(&encode_hello(5), addr(1)) else {
            panic!("no welcome")
        };
        assert_eq!(u64_at(&w, 6), 5);
        assert_eq!(u64_at(&w, 14), 77);
        assert!(ct_eq(&tag(KEY, &w[..22]), &w[22..]));
        assert_eq!(f.stats.hellos, 1);
        assert_eq!(f.peer(), None); // a hello does not lock the peer
    }

    #[test]
    fn only_valid_fresh_commands_from_the_first_peer_pass() {
        let mut f = Filter::new(KEY.to_vec(), 77);
        let good = |seq, v| encode_command(KEY, 77, seq, cmd(v));
        assert_eq!(vx(f.check(&good(1, 0.1), addr(1))), Some(0.1));
        assert_eq!(vx(f.check(&good(1, 0.2), addr(1))), None); // replay
        assert_eq!(
            vx(f.check(&encode_command(b"wrong-key-wrong-key!", 77, 2, cmd(0.3)), addr(1))),
            None
        );
        assert_eq!(vx(f.check(&encode_command(KEY, 76, 3, cmd(0.3)), addr(1))), None); // old session
        let mut tampered = good(4, 0.1);
        tampered[22] ^= 0x40; // flip a bit of vx
        assert_eq!(vx(f.check(&tampered, addr(1))), None);
        assert_eq!(vx(f.check(&good(5, 0.3), addr(2))), None); // valid, other sender
        assert_eq!(vx(f.check(b"junk", addr(1))), None);
        assert_eq!(vx(f.check(&good(6, 0.25), addr(1))), Some(0.25));
        assert_eq!(
            f.stats,
            ChannelStats {
                accepted: 2,
                malformed: 1,
                unauthenticated: 3,
                stale: 1,
                foreign: 1,
                hellos: 0
            }
        );
    }

    #[test]
    fn a_wrong_key_cannot_lock_the_peer() {
        let mut f = Filter::new(KEY.to_vec(), 9);
        f.check(&encode_command(&[0u8; 32], 9, 1, cmd(0.3)), addr(3));
        assert_eq!(f.peer(), None);
        assert_eq!(vx(f.check(&encode_command(KEY, 9, 1, cmd(0.1)), addr(1))), Some(0.1));
    }

    #[test]
    fn keyless_channel_refuses_non_loopback() {
        assert!(CommandChannel::bind("0.0.0.0:0", None, 1).is_err());
        assert!(CommandChannel::bind("127.0.0.1:0", None, 1).is_ok());
        assert!(CommandChannel::bind("0.0.0.0:0", Some(KEY.to_vec()), 1).is_ok());
    }

    #[test]
    fn socket_handshake_and_command() {
        let mut ch = CommandChannel::bind("127.0.0.1:0", Some(KEY.to_vec()), 4242).unwrap();
        let tx = UdpSocket::bind("127.0.0.1:0").unwrap();
        tx.set_read_timeout(Some(std::time::Duration::from_millis(20))).unwrap();
        let to = ch.local_addr().unwrap();
        tx.send_to(&encode_hello(11), to).unwrap();
        let mut w = [0u8; 64];
        let mut session = None;
        for _ in 0..50 {
            ch.poll();
            if let Ok((n, _)) = tx.recv_from(&mut w) {
                assert_eq!(n, WELCOME_SIZE);
                session = Some(u64_at(&w, 14));
                break;
            }
        }
        assert_eq!(session, Some(4242));
        tx.send_to(&encode_command(KEY, 4242, 1, cmd(0.2)), to).unwrap();
        let mut got = None;
        for _ in 0..100 {
            if let Some(c) = ch.poll() {
                got = Some(c.vx);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(got, Some(0.2));
    }

    #[test]
    fn load_key_checks_length_and_hex() {
        let dir = std::env::temp_dir();
        let p = dir.join(format!("bpk-{}", std::process::id()));
        std::fs::write(&p, "00112233445566778899aabbccddeeff\n").unwrap();
        assert_eq!(load_key(p.to_str().unwrap()).unwrap().len(), 16);
        std::fs::write(&p, "0011").unwrap();
        assert!(load_key(p.to_str().unwrap()).is_err());
        std::fs::write(&p, "zz").unwrap();
        assert!(load_key(p.to_str().unwrap()).is_err());
        let _ = std::fs::remove_file(&p);
    }
}
