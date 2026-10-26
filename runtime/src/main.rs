//! `biped-runtime`: runs the control loop against the MuJoCo sim bridge.
//!
//! ```text
//! biped-runtime --connect 127.0.0.1:PORT [--controller stand|nan-at:T|jump-at:T:JOINT:DELTA]
//!               [--command START:END:VX:VY:WZ]... [--state-timeout-ms 500] [--realtime]
//! ```
//!
//! Prints a one-line JSON summary on exit. Exit codes: 0 the sim closed the
//! connection, 2 usage or protocol error, 3 state timeout (watchdog).

use std::io::{self, BufReader, ErrorKind};
use std::net::TcpStream;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use biped_runtime::controller::{Controller, FaultInjector, Injected, Stand};
use biped_runtime::gate::{Command, CommandGate, GateLimits};
use biped_runtime::protocol::{Action, Hello, State};
use biped_runtime::safety::{SafetyLimits, Supervisor, FAULT_STATE_TIMEOUT};

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
            _ => return Err(format!("unknown argument {flag:?}")),
        }
    }
    if a.connect.is_empty() {
        return Err("--connect HOST:PORT is required".into());
    }
    Ok(a)
}

fn make_controller(spec: &str, hello: &Hello) -> Result<Box<dyn Controller>, String> {
    let stand = Stand::new(hello);
    let (kind, rest) = spec.split_once(':').unwrap_or((spec, ""));
    Ok(match kind {
        "stand" => Box::new(stand),
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
}

fn summary(exit: &str, s: &Stats, sup: &Supervisor, gate: &CommandGate) -> String {
    let opt = |v: Option<u64>| v.map_or("null".to_string(), |x| x.to_string());
    let mean = if s.periods > 0 {
        s.compute_sum_us as f64 / s.periods as f64
    } else {
        0.0
    };
    format!(
        concat!(
            "{{\"exit\":\"{}\",\"periods\":{},\"compute_mean_us\":{:.1},\"compute_max_us\":{},",
            "\"period_max_us\":{},\"deadline_misses\":{},\"latched\":{},",
            "\"first_fault_step\":{{\"tilt\":{},\"overspeed\":{},\"nonfinite_state\":{},\"state_timeout\":{}}},",
            "\"events\":{{\"nonfinite_target\":{},\"position_clamp\":{},\"slew_limit\":{}}},",
            "\"gate_rejected\":{}}}"
        ),
        exit,
        s.periods,
        mean,
        s.compute_max_us,
        s.period_max_us,
        s.deadline_misses,
        sup.latched,
        opt(sup.first_fault_step[0]),
        opt(sup.first_fault_step[1]),
        opt(sup.first_fault_step[2]),
        opt(sup.first_fault_step[3]),
        sup.event_counts[0],
        sup.event_counts[1],
        sup.event_counts[2],
        gate.rejected,
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
    let mut controller = make_controller(&args.controller, &hello)?;
    let mut gate = CommandGate::new(GateLimits::default());
    let mut sup = Supervisor::new(&hello, SafetyLimits::default());
    let mut stats = Stats::default();
    let mut last_step = 0u64;
    let mut prev_start: Option<Instant> = None;
    loop {
        let state = match State::read(&mut reader, n) {
            Ok(s) => s,
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => {
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
        stats.periods += 1;
        stats.compute_sum_us += compute.as_micros();
        stats.compute_max_us = stats.compute_max_us.max(compute.as_micros());
        if args.realtime {
            let period = Duration::from_secs_f64(dt);
            match period.checked_sub(start.elapsed()) {
                Some(rest) => std::thread::sleep(rest),
                None => stats.deadline_misses += 1,
            }
        }
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
