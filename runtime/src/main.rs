//! `biped-runtime`: runs the control loop against the MuJoCo sim bridge.
//!
//! ```text
//! biped-runtime (--connect 127.0.0.1:PORT
//!                | (--device DEV | --hw-sim) --hello FILE --map FILE --imu upright
//!                  --torque-fraction F [--seconds S])
//!               [--controller stand|policy:MLP_FILE|nan-at:T|jump-at:T:JOINT:DELTA]
//!               [--command START:END:VX:VY:WZ]... [--state-timeout-ms 500] [--realtime]
//!               [--torque-fraction F] [--action-scale 0.5] [--no-joint-vel-lag]
//!               [--joint-feedback measured|commanded] [--log FILE]
//!               [--command-udp 127.0.0.1:PORT [--command-key KEY_FILE]]
//! ```
//!
//! Prints a one-line JSON summary on exit. Exit codes: 0 the sim closed the
//! connection (or `--seconds` ran out on hardware), 2 usage or protocol error,
//! 3 state timeout (watchdog; on hardware also a servo that stops answering).
//!
//! Hardware mode (WP5-03) replaces the sim bridge with the STS bus: the joint
//! contract comes from a `Hello` file (`scripts/export_hello.py`), states from
//! `hwstate::HwSource` through the joint map, actions go out as servo goals. It
//! always paces at the control rate and requires an explicit torque fraction
//! (S-5). `--imu upright` is the only IMU so far: it reports upright, so the
//! tilt fault cannot fire, and it is for a robot on a stand only. `--hw-sim`
//! runs the same path on the simulated bus (CI). Nothing here has run on a
//! real servo; first use is Phase 5 stage 4 with the owner's approval.

use std::fs::File;
use std::io::{BufReader, BufWriter, ErrorKind};
use std::net::TcpStream;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use biped_runtime::auth::random_u64;
use biped_runtime::channel::{load_key, ChannelStats, CommandChannel};
use biped_runtime::controller::{Controller, FaultInjector, Injected, Stand};
use biped_runtime::gate::{Command, CommandGate, GateLimits};
use biped_runtime::hwstate::{HwSource, UprightImu};
use biped_runtime::jointmap::{JointMap, BIPED_V1_JOINTS};
use biped_runtime::mlp::Mlp;
use biped_runtime::pacer::Pacer;
use biped_runtime::policy::{JointFeedback, Policy, DEFAULT_ACTION_SCALE};
use biped_runtime::protocol::{Action, Hello, State};
use biped_runtime::safety::{SafetyLimits, Supervisor, FAULT_STATE_TIMEOUT, N_EVENTS, N_FAULTS};
use biped_runtime::sts::{self, addr};
use biped_runtime::stsbus::{open_device, Bus, Port, SimPort};
use biped_runtime::telemetry::{Log, Record};

struct Scheduled {
    start: f64,
    end: f64,
    cmd: Command,
}

struct Args {
    connect: String,
    controller: String,
    commands: Vec<Scheduled>,
    state_timeout: Duration,
    realtime: bool,
    torque_fraction: Option<f64>,
    action_scale: f64,
    joint_vel_lag: bool,
    joint_feedback: JointFeedback,
    log: Option<String>,
    command_udp: Option<String>,
    command_key: Option<String>,
    device: Option<String>,
    hw_sim: bool,
    hello: Option<String>,
    map: Option<String>,
    imu: Option<String>,
    seconds: Option<f64>,
}

fn floats(s: &str, n: usize) -> Result<Vec<f64>, String> {
    let v: Result<Vec<f64>, _> = s.split(':').map(str::parse::<f64>).collect();
    match v {
        Ok(v) if v.len() == n => Ok(v),
        _ => Err(format!("expected {n} numbers separated by ':' in {s:?}")),
    }
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        connect: String::new(),
        controller: "stand".into(),
        commands: Vec::new(),
        state_timeout: Duration::from_millis(500),
        realtime: false,
        torque_fraction: None,
        action_scale: DEFAULT_ACTION_SCALE,
        joint_vel_lag: true,
        joint_feedback: JointFeedback::Measured,
        log: None,
        command_udp: None,
        command_key: None,
        device: None,
        hw_sim: false,
        hello: None,
        map: None,
        imu: None,
        seconds: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--connect" => a.connect = value()?,
            "--controller" => a.controller = value()?,
            "--command" => {
                let v = floats(&value()?, 5)?;
                a.commands.push(Scheduled {
                    start: v[0],
                    end: v[1],
                    cmd: Command {
                        vx: v[2],
                        vy: v[3],
                        wz: v[4],
                    },
                });
            }
            "--state-timeout-ms" => {
                let ms = value()?.parse::<u64>().map_err(|e| e.to_string())?;
                a.state_timeout = Duration::from_millis(ms.max(1));
            }
            "--realtime" => a.realtime = true,
            "--no-joint-vel-lag" => a.joint_vel_lag = false,
            "--joint-feedback" => a.joint_feedback = value()?.parse()?,
            "--torque-fraction" => {
                let f = value()?.parse::<f64>().map_err(|e| e.to_string())?;
                if !(f > 0.0 && f <= 1.0) {
                    return Err("--torque-fraction must be in (0, 1]".into());
                }
                a.torque_fraction = Some(f);
            }
            "--log" => a.log = Some(value()?),
            "--command-udp" => a.command_udp = Some(value()?),
            "--command-key" => a.command_key = Some(value()?),
            "--device" => a.device = Some(value()?),
            "--hw-sim" => a.hw_sim = true,
            "--hello" => a.hello = Some(value()?),
            "--map" => a.map = Some(value()?),
            "--imu" => a.imu = Some(value()?),
            "--seconds" => {
                let s = value()?.parse::<f64>().map_err(|e| e.to_string())?;
                if !(s > 0.0 && s.is_finite()) {
                    return Err("--seconds must be > 0".into());
                }
                a.seconds = Some(s);
            }
            "--action-scale" => a.action_scale = value()?.parse::<f64>().map_err(|e| e.to_string())?,
            _ => return Err(format!("unknown argument {flag:?}")),
        }
    }
    if a.command_key.is_some() && a.command_udp.is_none() {
        return Err("--command-key needs --command-udp".into());
    }
    let hardware = a.device.is_some() || a.hw_sim;
    if [!a.connect.is_empty(), a.device.is_some(), a.hw_sim]
        .iter()
        .filter(|&&x| x)
        .count()
        != 1
    {
        return Err("give exactly one of --connect HOST:PORT, --device DEV or --hw-sim".into());
    }
    if hardware {
        if a.hello.is_none() || a.map.is_none() {
            return Err("hardware mode needs --hello FILE and --map FILE".into());
        }
        if a.imu.as_deref() != Some("upright") {
            return Err("hardware mode needs --imu upright (robot on a stand; no IMU driver yet)".into());
        }
        if a.torque_fraction.is_none() {
            return Err("hardware mode needs an explicit --torque-fraction (S-5)".into());
        }
        a.realtime = true;
    } else if a.hello.is_some() || a.map.is_some() || a.imu.is_some() || a.seconds.is_some() {
        return Err("--hello, --map, --imu and --seconds are for hardware mode".into());
    }
    Ok(a)
}

fn make_controller(
    spec: &str,
    hello: &Hello,
    action_scale: f64,
    joint_vel_lag: bool,
    joint_feedback: JointFeedback,
) -> Result<Box<dyn Controller>, String> {
    let stand = Stand::new(hello);
    let (kind, rest) = spec.split_once(':').unwrap_or((spec, ""));
    Ok(match kind {
        "stand" => Box::new(stand),
        "policy" => {
            let buf = std::fs::read(rest).map_err(|e| format!("policy {rest:?}: {e}"))?;
            let mlp = Mlp::from_bytes(&buf).map_err(|e| format!("policy {rest:?}: {e}"))?;
            Box::new(
                Policy::new(mlp, hello, action_scale)?
                    .with_joint_vel_lag(joint_vel_lag)
                    .with_joint_feedback(joint_feedback),
            )
        }
        "nan-at" => {
            let v = floats(rest, 1)?;
            Box::new(FaultInjector {
                inner: stand,
                at: v[0],
                fault: Injected::NonFinite,
            })
        }
        "jump-at" => {
            let v = floats(rest, 3)?;
            let joint = v[1] as usize;
            if joint >= hello.joints.len() {
                return Err(format!("joint {joint} out of range"));
            }
            Box::new(FaultInjector {
                inner: stand,
                at: v[0],
                fault: Injected::Jump { joint, delta: v[2] },
            })
        }
        _ => return Err(format!("unknown controller {spec:?}")),
    })
}

#[derive(Default)]
struct Stats {
    periods: u64,
    compute_max_us: u128,
    compute_sum_us: u128,
    period_max_us: u128,
    deadline_misses: u64,
    log_records: u64,
    log_write_errors: u64,
    channel: ChannelStats,
}

const FAULT_NAMES: [&str; N_FAULTS] = [
    "tilt",
    "overspeed",
    "nonfinite_state",
    "state_timeout",
    "estop",
    "undervoltage",
    "overtemperature",
];
const EVENT_NAMES: [&str; N_EVENTS] = [
    "nonfinite_target",
    "position_clamp",
    "slew_limit",
    "torque_clamp",
    "derate",
];

fn summary(exit: &str, s: &Stats, sup: &Supervisor, gate: &CommandGate) -> String {
    let mean = if s.periods > 0 {
        s.compute_sum_us as f64 / s.periods as f64
    } else {
        0.0
    };
    let faults: Vec<String> = FAULT_NAMES
        .iter()
        .zip(sup.first_fault_step)
        .map(|(n, v)| format!("\"{n}\":{}", v.map_or("null".to_string(), |x| x.to_string())))
        .collect();
    let events: Vec<String> = EVENT_NAMES
        .iter()
        .zip(sup.event_counts)
        .map(|(n, c)| format!("\"{n}\":{c}"))
        .collect();
    format!(
        concat!(
            "{{\"exit\":\"{}\",\"periods\":{},\"compute_mean_us\":{:.1},\"compute_max_us\":{},",
            "\"period_max_us\":{},\"deadline_misses\":{},\"latched\":{},",
            "\"first_fault_step\":{{{}}},\"events\":{{{}}},\"gate_rejected\":{},",
            "\"log_records\":{},\"log_write_errors\":{},",
            "\"channel\":{{\"accepted\":{},\"malformed\":{},\"unauthenticated\":{},\"stale\":{},",
            "\"foreign\":{},\"hellos\":{}}}}}"
        ),
        exit,
        s.periods,
        mean,
        s.compute_max_us,
        s.period_max_us,
        s.deadline_misses,
        sup.latched,
        faults.join(","),
        events.join(","),
        gate.rejected,
        s.log_records,
        s.log_write_errors,
        s.channel.accepted,
        s.channel.malformed,
        s.channel.unauthenticated,
        s.channel.stale,
        s.channel.foreign,
        s.channel.hellos,
    )
}

/// What the loop gets from the robot (or the sim playing it) each period.
enum Next {
    State(State),
    /// The sim closed the connection, or the hardware run's time is up.
    Closed,
    /// No fresh state in time (watchdog), with the reason.
    Timeout(String),
}

trait Link {
    fn next(&mut self) -> Result<Next, String>;
    fn send(&mut self, action: &Action) -> Result<(), String>;
    /// Summary `exit` name for `Next::Closed`.
    fn closed_name(&self) -> &'static str;
}

struct SimLink {
    reader: BufReader<TcpStream>,
    writer: TcpStream,
    n: usize,
}

impl Link for SimLink {
    fn next(&mut self) -> Result<Next, String> {
        match State::read(&mut self.reader, self.n) {
            Ok(s) => Ok(Next::State(s)),
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => Ok(Next::Closed),
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                Ok(Next::Timeout("no state from the sim".into()))
            }
            Err(e) => Err(format!("state: {e}")),
        }
    }

    fn send(&mut self, action: &Action) -> Result<(), String> {
        action.write(&mut self.writer).map_err(|e| format!("action: {e}"))
    }

    fn closed_name(&self) -> &'static str {
        "sim_closed"
    }
}

struct HwLink<P: Port> {
    hw: HwSource<P, UprightImu>,
    end: Option<Instant>,
}

impl<P: Port> Link for HwLink<P> {
    fn next(&mut self) -> Result<Next, String> {
        if self.end.is_some_and(|e| Instant::now() >= e) {
            // Time is up: limp, then report a clean end.
            let off = Action {
                step: 0,
                torque_enable: false,
                fault_bits: 0,
                targets: Vec::new(),
            };
            self.hw.apply(&off).map_err(|e| format!("torque off: {e}"))?;
            return Ok(Next::Closed);
        }
        match self.hw.read(false) {
            Ok(s) => Ok(Next::State(s)),
            Err(e) => Ok(Next::Timeout(e.to_string())),
        }
    }

    fn send(&mut self, action: &Action) -> Result<(), String> {
        self.hw.apply(action).map_err(|e| format!("action: {e}"))
    }

    fn closed_name(&self) -> &'static str {
        "duration"
    }
}

/// A simulated bus with every joint's servo at its centre, 7.4 V, 30 deg C.
fn sim_port(ids: &[u8]) -> SimPort {
    let mut bus = sts::SimBus::new(ids);
    for (_, t) in bus.tables.iter_mut() {
        t[addr::PRESENT_POSITION as usize..][..2].copy_from_slice(&2048u16.to_le_bytes());
        t[addr::PRESENT_VOLTAGE as usize] = 74;
        t[addr::PRESENT_TEMPERATURE as usize] = 30;
    }
    SimPort::new(bus)
}

fn hw_link<P: Port + 'static>(args: &Args, port: P, map: JointMap) -> Result<Box<dyn Link>, String> {
    let hw = HwSource::new(Bus::new(port), map, UprightImu).map_err(|e| format!("servo bus: {e}"))?;
    Ok(Box::new(HwLink {
        hw,
        end: args.seconds.map(|s| Instant::now() + Duration::from_secs_f64(s)),
    }))
}

fn open_link(args: &Args) -> Result<(Hello, Box<dyn Link>), String> {
    if !args.connect.is_empty() {
        let stream = TcpStream::connect(&args.connect).map_err(|e| format!("connect: {e}"))?;
        stream.set_nodelay(true).map_err(|e| e.to_string())?;
        let writer = stream.try_clone().map_err(|e| e.to_string())?;
        let mut reader = BufReader::new(stream);
        let hello = Hello::read(&mut reader).map_err(|e| format!("hello: {e}"))?;
        reader
            .get_ref()
            .set_read_timeout(Some(args.state_timeout))
            .map_err(|e| e.to_string())?;
        let n = hello.joints.len();
        return Ok((hello, Box::new(SimLink { reader, writer, n })));
    }
    let path = args.hello.as_deref().expect("checked in parse_args");
    let mut f = File::open(path).map_err(|e| format!("hello {path:?}: {e}"))?;
    let hello = Hello::read(&mut f).map_err(|e| format!("hello {path:?}: {e}"))?;
    let path = args.map.as_deref().expect("checked in parse_args");
    let text = std::fs::read_to_string(path).map_err(|e| format!("map {path:?}: {e}"))?;
    let map = JointMap::parse(&text, &BIPED_V1_JOINTS).map_err(|e| format!("map {path:?}: {e}"))?;
    if map.joints.len() != hello.joints.len() {
        return Err("the hello file and the joint map have different joint counts".into());
    }
    let link = match &args.device {
        Some(dev) => hw_link(args, open_device(dev).map_err(|e| format!("{dev}: {e}"))?, map)?,
        None => {
            let port = sim_port(&map.ids());
            hw_link(args, port, map)?
        }
    };
    Ok((hello, link))
}

fn run(args: &Args) -> Result<ExitCode, String> {
    let (hello, mut link) = open_link(args)?;
    let dt = hello.control_dt;
    let mut controller = make_controller(
        &args.controller,
        &hello,
        args.action_scale,
        args.joint_vel_lag,
        args.joint_feedback,
    )?;
    let mut gate = CommandGate::new(GateLimits::default());
    let mut sup = Supervisor::new(
        &hello,
        SafetyLimits {
            torque_fraction: args.torque_fraction,
            ..SafetyLimits::default()
        },
    );
    let mut log = match &args.log {
        Some(path) => {
            let f = File::create(path).map_err(|e| format!("log {path:?}: {e}"))?;
            Some(Log::new(BufWriter::new(f), &hello).map_err(|e| format!("log {path:?}: {e}"))?)
        }
        None => None,
    };
    let mut channel = match &args.command_udp {
        Some(a) => {
            let key = args.command_key.as_deref().map(load_key).transpose()?;
            let session = random_u64().map_err(|e| format!("session nonce: {e}"))?;
            let ch = CommandChannel::bind(a, key, session)?;
            if let Ok(local) = ch.local_addr() {
                eprintln!("biped-runtime: command channel on udp://{local}");
            }
            Some(ch)
        }
        None => None,
    };
    let mut stats = Stats::default();
    let mut t0: Option<Instant> = None;
    let mut last_step = 0u64;
    let mut prev_start: Option<Instant> = None;
    let mut pacer = Pacer::new(Duration::from_secs_f64(dt));
    loop {
        let state = match link.next()? {
            Next::State(s) => s,
            Next::Closed => {
                close_log(&mut log, &mut stats);
                println!("{}", summary(link.closed_name(), &stats, &sup, &gate));
                return Ok(ExitCode::SUCCESS);
            }
            Next::Timeout(why) => {
                // Watchdog: no fresh state. Latch and send torque off once (best effort).
                eprintln!("biped-runtime: state timeout: {why}");
                sup.latch(FAULT_STATE_TIMEOUT, last_step + 1);
                let off = sup.filter_off();
                let _ = link.send(&Action {
                    step: last_step + 1,
                    torque_enable: false,
                    fault_bits: sup.latched,
                    targets: off,
                });
                close_log(&mut log, &mut stats);
                println!("{}", summary("state_timeout", &stats, &sup, &gate));
                return Ok(ExitCode::from(3));
            }
        };
        let start = Instant::now();
        if let Some(p) = prev_start {
            stats.period_max_us = stats.period_max_us.max(start.duration_since(p).as_micros());
        }
        prev_start = Some(start);
        last_step = state.step;
        for c in &args.commands {
            if c.start <= state.time && state.time < c.end {
                gate.submit(state.time, c.cmd);
            }
        }
        if let Some(ch) = channel.as_mut() {
            if let Some(c) = ch.poll() {
                gate.submit(state.time, c);
            }
            stats.channel = ch.filter.stats;
        }
        let cmd = gate.output(state.time, dt);
        let wanted = controller.targets(&state, cmd);
        let out = sup.filter(&state, &wanted, dt);
        let action = Action {
            step: state.step,
            torque_enable: out.torque_enable,
            fault_bits: out.bits,
            targets: out.targets,
        };
        link.send(&action)?;
        controller.sent(&action.targets);
        let compute = start.elapsed();
        if let Some(l) = log.as_mut() {
            // After the action is sent, so logging never delays it.
            let t0 = *t0.get_or_insert(start);
            l.record(&Record {
                state: &state,
                wall: start.duration_since(t0).as_secs_f64(),
                compute_us: compute.as_micros().min(u32::MAX as u128) as u32,
                cmd,
                wanted: &wanted,
                targets: &action.targets,
                torque_enable: action.torque_enable,
                bits: action.fault_bits,
            });
        }
        stats.periods += 1;
        stats.compute_sum_us += compute.as_micros();
        stats.compute_max_us = stats.compute_max_us.max(compute.as_micros());
        if args.realtime {
            // Absolute schedule: the wait for the next state counts towards the
            // period (relative sleeping added it on top: ~40 Hz on a loaded M5, MAC-09).
            if let Some(rest) = pacer.wait_time(start, Instant::now()) {
                std::thread::sleep(rest);
            }
            stats.deadline_misses = pacer.misses;
        }
    }
}

fn close_log(log: &mut Option<Log<BufWriter<File>>>, stats: &mut Stats) {
    if let Some(l) = log.as_mut() {
        l.finish();
        stats.log_records = l.records;
        stats.log_write_errors = l.write_errors;
    }
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("biped-runtime: {e}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("biped-runtime: {e}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link(ids_on_bus: &[u8]) -> HwLink<SimPort> {
        let text = include_str!("../../hardware/calibration/biped_v1_joints.example.txt");
        let map = JointMap::parse(text, &BIPED_V1_JOINTS).unwrap();
        let mut hw = HwSource::new(Bus::new(sim_port(&map.ids())), map, UprightImu).unwrap();
        hw.bus.port.bus.tables.retain(|(id, _)| ids_on_bus.contains(id));
        HwLink { hw, end: None }
    }

    #[test]
    fn a_silent_servo_is_a_state_timeout() {
        let all: Vec<u8> = (1..=10).collect();
        assert!(matches!(link(&all).next(), Ok(Next::State(_))));
        let mut l = link(&[1, 2, 3]);
        match l.next() {
            Ok(Next::Timeout(why)) => assert!(why.contains('4'), "{why}"),
            _ => panic!("expected a timeout"),
        }
    }

    #[test]
    fn end_of_run_switches_torque_off() {
        let all: Vec<u8> = (1..=10).collect();
        let mut l = link(&all);
        l.send(&Action {
            step: 0,
            torque_enable: true,
            fault_bits: 0,
            targets: vec![0.0; 10],
        })
        .unwrap();
        assert_eq!(l.hw.torque_on(), Some(true));
        l.end = Some(Instant::now());
        assert!(matches!(l.next(), Ok(Next::Closed)));
        assert_eq!(l.hw.torque_on(), Some(false));
        let t = &l.hw.bus.port.bus.tables;
        assert!(t.iter().all(|(_, r)| r[addr::TORQUE_ENABLE as usize] == 0));
    }
}
