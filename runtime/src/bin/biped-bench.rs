//! `biped-bench`: read-only bench tools for Phase 5 stage 2 (WP5-02).
//!
//! ```text
//! biped-bench scan  (--device DEV | --sim) [--max-id 20]
//! biped-bench read  (--device DEV | --sim) --map FILE
//! biped-bench check (--device DEV | --sim) --map FILE [--seconds 60] [--hz 50]
//! ```
//!
//! `scan` pings ids 0..=max-id and prints the ones that answer. `read` prints
//! each joint's servo angle, sim angle, voltage, temperature and error bits.
//! `check` reads every joint at the control rate for a while and prints the bus
//! counters: stage 2 passes with no timeouts or checksum errors over 1 min.
//! Nothing here writes a register or switches torque on. Set the device first,
//! e.g. `stty -F /dev/ttyUSB0 1000000 raw -echo`. `--sim` runs against the
//! simulated bus (CI / demo). Output is one JSON object per line.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use biped_runtime::jointmap::{JointMap, BIPED_V1_JOINTS};
use biped_runtime::sts::{self, addr};
use biped_runtime::stsbus::{open_device, Bus, BusStats, Port, SimPort};

struct Args {
    cmd: String,
    device: Option<String>,
    sim: bool,
    map: Option<String>,
    max_id: u8,
    seconds: f64,
    hz: f64,
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let mut a = Args {
        cmd: argv.first().cloned().ok_or("missing command: scan, read or check")?,
        device: None,
        sim: false,
        map: None,
        max_id: 20,
        seconds: 60.0,
        hz: 50.0,
    };
    let mut it = argv[1..].iter();
    while let Some(flag) = it.next() {
        let mut value = || it.next().cloned().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--device" => a.device = Some(value()?),
            "--sim" => a.sim = true,
            "--map" => a.map = Some(value()?),
            "--max-id" => a.max_id = value()?.parse().map_err(|_| "--max-id must be 0..=253")?,
            "--seconds" => a.seconds = value()?.parse().map_err(|_| "--seconds must be a number")?,
            "--hz" => a.hz = value()?.parse().map_err(|_| "--hz must be a number")?,
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    if a.sim == a.device.is_some() {
        return Err("give exactly one of --device DEV or --sim".into());
    }
    if a.max_id > 253 || !(a.seconds > 0.0 && a.hz > 0.0 && a.hz <= 1000.0) {
        return Err("--max-id <= 253, --seconds > 0, 0 < --hz <= 1000".into());
    }
    Ok(a)
}

fn load_map(a: &Args) -> Result<JointMap, String> {
    let path = a.map.as_deref().ok_or("--map FILE is needed")?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    JointMap::parse(&text, &BIPED_V1_JOINTS)
}

/// A simulated bus with servos 1..=10 at their centre, 7.4 V, 30 deg C.
fn sim_port() -> SimPort {
    let ids: Vec<u8> = (1..=10).collect();
    let mut bus = sts::SimBus::new(&ids);
    for (_, t) in bus.tables.iter_mut() {
        t[addr::PRESENT_POSITION as usize..][..2].copy_from_slice(&2048u16.to_le_bytes());
        t[addr::PRESENT_VOLTAGE as usize] = 74;
        t[addr::PRESENT_TEMPERATURE as usize] = 30;
    }
    SimPort::new(bus)
}

fn stats_json(s: &BusStats) -> String {
    format!(
        "\"exchanges\":{},\"retries\":{},\"timeouts\":{},\"bad_checksum\":{},\"resync_bytes\":{},\"unexpected_id\":{}",
        s.exchanges, s.retries, s.timeouts, s.bad_checksum, s.resync_bytes, s.unexpected_id
    )
}

fn run<P: Port>(a: &Args, mut bus: Bus<P>) -> Result<bool, String> {
    match a.cmd.as_str() {
        "scan" => {
            let found = bus.scan(0..=a.max_id);
            println!("{{\"found\":{found:?}}}");
            Ok(!found.is_empty())
        }
        "read" => {
            let map = load_map(a)?;
            let servos = bus.read_state(&map.ids()).map_err(|e| e.to_string())?;
            let servo_pos: Vec<f64> = servos.iter().map(|s| s.pos_rad).collect();
            let sim = map.to_sim(&servo_pos);
            for ((j, s), q) in map.joints.iter().zip(&servos).zip(&sim) {
                let (v, t) = bus.read_health(j.id).map_err(|e| e.to_string())?;
                println!(
                    "{{\"joint\":\"{}\",\"id\":{},\"servo_rad\":{:.4},\"sim_rad\":{:.4},\"voltage\":{:.1},\"temperature\":{},\"error\":{}}}",
                    j.name, j.id, s.pos_rad, q, v, t, s.error
                );
            }
            Ok(true)
        }
        "check" => {
            let map = load_map(a)?;
            let ids = map.ids();
            let period = Duration::from_secs_f64(1.0 / a.hz);
            let end = Instant::now() + Duration::from_secs_f64(a.seconds);
            let (mut reads, mut failed, mut servo_errors) = (0u64, 0u64, 0u8);
            let mut next = Instant::now();
            while Instant::now() < end {
                match bus.read_state(&ids) {
                    Ok(s) => servo_errors |= s.iter().fold(0, |e, s| e | s.error),
                    Err(_) => failed += 1,
                }
                reads += 1;
                next += period;
                std::thread::sleep(next.saturating_duration_since(Instant::now()));
            }
            let s = bus.stats;
            let pass = failed == 0 && s.timeouts == 0 && s.bad_checksum == 0 && servo_errors == 0;
            println!(
                "{{\"pass\":{pass},\"reads\":{reads},\"failed_reads\":{failed},\"servo_errors\":{servo_errors},{}}}",
                stats_json(&s)
            );
            Ok(pass)
        }
        other => Err(format!("unknown command {other}: scan, read or check")),
    }
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let result = parse(&argv).and_then(|a| match &a.device {
        Some(dev) => {
            let port = open_device(dev).map_err(|e| format!("{dev}: {e}"))?;
            run(&a, Bus::new(port))
        }
        None => run(&a, Bus::new(sim_port())),
    });
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("biped-bench: {e}");
            ExitCode::from(2)
        }
    }
}
