//! `biped-runtime`: runs the control loop against the MuJoCo sim bridge.
//!
//! ```text
//! biped-runtime --connect 127.0.0.1:PORT
//!               [--controller stand|policy:MLP_FILE|nan-at:T|jump-at:T:JOINT:DELTA]
//!               [--command START:END:VX:VY:WZ]... [--state-timeout-ms 500] [--realtime]
//!               [--torque-fraction F] [--action-scale 0.5] [--log FILE]
//!               [--command-udp 127.0.0.1:PORT [--command-key KEY_FILE]]
//! ```
//!
//! Prints a one-line JSON summary on exit. Exit codes: 0 the sim closed the
//! connection, 2 usage or protocol error, 3 state timeout (watchdog).

use std::fs::File;
use std::io::{self, BufReader, BufWriter, ErrorKind};
use std::net::TcpStream;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use biped_runtime::auth::random_u64;
use biped_runtime::channel::{load_key, ChannelStats, CommandChannel};
use biped_runtime::controller::{Controller, FaultInjector, Injected, Stand};
use biped_runtime::gate::{Command, CommandGate, GateLimits};
use biped_runtime::mlp::Mlp;
use biped_runtime::pacer::Pacer;
use biped_runtime::policy::{Policy, DEFAULT_ACTION_SCALE};
use biped_runtime::protocol::{Action, Hello, State};
use biped_runtime::safety::{SafetyLimits, Supervisor, FAULT_STATE_TIMEOUT, N_EVENTS, N_FAULTS};
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
    log: Option<String>,
    command_udp: Option<String>,
    command_key: Option<String>,
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
        log: None,
        command_udp: None,
        command_key: None,
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
            "--action-scale" => a.action_scale = value()?.parse::<f64>().map_err(|e| e.to_string())?,
            _ => return Err(format!("unknown argument {flag:?}")),
        }
    }
    if a.command_key.is_some() && a.command_udp.is_none() {
        return Err("--command-key needs --command-udp".into());
    }
    if a.connect.is_empty() {
        return Err("--connect HOST:PORT is required".into());
    }
    Ok(a)
}

fn make_controller(spec: &str, hello: &Hello, action_scale: f64) -> Result<Box<dyn Controller>, String> {
    let stand = Stand::new(hello);
    let (kind, rest) = spec.split_once(':').unwrap_or((spec, ""));
    Ok(match kind {
        "stand" => Box::new(stand),
        "policy" => {
            let buf = std::fs::read(rest).map_err(|e| format!("policy {rest:?}: {e}"))?;
            let mlp = Mlp::from_bytes(&buf).map_err(|e| format!("policy {rest:?}: {e}"))?;
            Box::new(Policy::new(mlp, hello, action_scale)?)
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

fn run(args: &Args) -> Result<ExitCode, String> {
    let stream = TcpStream::connect(&args.connect).map_err(|e| format!("connect: {e}"))?;
    stream.set_nodelay(true).map_err(|e| e.to_string())?;
    let mut writer = stream.try_clone().map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(stream);
    let hello = Hello::read(&mut reader).map_err(|e| format!("hello: {e}"))?;
    reader
        .get_ref()
        .set_read_timeout(Some(args.state_timeout))
        .map_err(|e| e.to_string())?;
    let n = hello.joints.len();
    let dt = hello.control_dt;
    let mut controller = make_controller(&args.controller, &hello, args.action_scale)?;
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
        let state = match State::read(&mut reader, n) {
            Ok(s) => s,
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => {
                close_log(&mut log, &mut stats);
                println!("{}", summary("sim_closed", &stats, &sup, &gate));
                return Ok(ExitCode::SUCCESS);
            }
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                // Watchdog: no fresh state. Latch and send torque off once (best effort).
                sup.latch(FAULT_STATE_TIMEOUT, last_step + 1);
                let off = sup.filter_off();
                let _ = Action {
                    step: last_step + 1,
                    torque_enable: false,
                    fault_bits: sup.latched,
                    targets: off,
                }
                .write(&mut writer);
                close_log(&mut log, &mut stats);
                println!("{}", summary("state_timeout", &stats, &sup, &gate));
                return Ok(ExitCode::from(3));
            }
            Err(e) => return Err(format!("state: {e}")),
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
        action
            .write(&mut writer)
            .map_err(|e: io::Error| format!("action: {e}"))?;
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
