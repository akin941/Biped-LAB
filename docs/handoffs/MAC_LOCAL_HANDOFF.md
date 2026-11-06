# Mac local handoff (MacBook Pro M5, 32 GB)

For a **local** Claude Code (or any agent / the owner) session on the Mac. The cloud
session cannot run anything here. Run the tasks in order, paste results into the
"Result" blocks, set the status, commit on a branch, open a PR.

Never mark a task `done` from cloud or GitHub-runner results. Label: `mac-m5`.

## Prerequisites (one time)

| Item | Check | Expected |
|---|---|---|
| macOS | `sw_vers -productVersion` | ≥ 14.0 (onnxruntime wheel requires it) |
| Arch | `uname -m` | `arm64` (if `x86_64`, the terminal runs under Rosetta — fix first) |
| Xcode CLT | `xcode-select -p` | a path (else `xcode-select --install`) |
| git | `git --version` | any recent |
| uv | `uv --version` | ≥ 0.9. Install with Homebrew: `brew install uv` (review: Homebrew formula, no curl-pipe script) |

No other system packages are needed. Do **not** run upstream install scripts
(e.g. microduck-lab `scripts/setup.sh`).

## Revision

Run against `main` at or after the revision recorded in the task's result block.
Record `git rev-parse HEAD` in every result.

---

## MAC-01 — Native install and test suite
Status: `done (mac-m5)` · Deps: WP0-04

```bash
git clone https://github.com/akin941/Biped-LAB.git && cd Biped-LAB
uv sync --all-extras
uv run python scripts/fetch_upstream.py microduck_rl
uv run python scripts/fetch_policies.py alpha_walking.onnx alpha_stand.onnx
uv run pytest -rs
uv run python -c "import platform, mujoco, onnxruntime; print(platform.machine(), mujoco.__version__, onnxruntime.__version__)"
```
**Expected:** all tests pass, 0 skipped (policies downloaded); last line
`arm64 3.10.0 1.24.4`.
**Acceptance:** exit code 0 for every command; skipped count 0.

Result:
```
revision: b3165540ffea0f6a007592eb5dab49a396a237f0
date: 2026-11-06, mac-m5 (macOS 27.0.1, Apple M5, 32 GB, uv 0.11.29)
output (tail):
  commands exactly as written: 138 passed, 5 skipped (exit 0), so the skipped-count
  acceptance was NOT met. The 5 skips need policies the commands above do not fetch.
  after `uv run python scripts/fetch_policies.py` (all, sha256-verified):
  143 passed in 9.70s, 0 skipped
  arm64 3.10.0 1.24.4
notes: the command list above should fetch all policies; left as written for the owner to review.
```

## MAC-02 — Interactive viewer
Status: `awaiting-mac` · Deps: MAC-01

On macOS the passive viewer must run under `mjpython` (ships with the mujoco wheel).
```bash
uv run mjpython -m bipedlab.cli view toy                       # our fixture biped
uv run mjpython -m bipedlab.cli view microduck                 # official model, no policy: it falls (expected)
uv run mjpython -m bipedlab.cli view microduck --policy alpha_stand.onnx
uv run mjpython -m bipedlab.cli view microduck --policy alpha_walking.onnx --twist 0.2 0 0
```
**Expected:** a window opens. `toy` stands still. Microduck without policy tips over in
about 1 s. With policies: record what happens (stands? walks? falls?) — note that our
runner uses MuJoCo position actuators, not the BAM model the policies were trained
with (backlog WP1-06), so imperfect behaviour is a valid, useful finding.
**Acceptance:** viewer opens for all four commands; observations written below
(a phone video is welcome but not required).
*This command was written in the cloud and has not been run anywhere yet.*

Result:
```
revision:
toy:
microduck (no policy):
alpha_stand:
alpha_walking 0.2 m/s:
```

## MAC-03 — Apple Silicon CPU benchmark
Status: `done (mac-m5, under load)` · Deps: MAC-01

```bash
uv run bipedlab bench microduck --env-label mac-m5 --policy alpha_walking.onnx --seconds 5
uv run bipedlab eval-policy alpha_walking.onnx --out /tmp/alpha_walking_mac.json
uv run bipedlab record alpha_walking.onnx --twist 0.2 0 0 --env-label mac-m5 --html --out runs/mac_timing
```
**Expected:** JSON with `physics_steps_per_s` and `policy_loop_hz`. For reference only,
the cloud container (Xeon 2.1 GHz) measured ~16–19k physics steps/s — not a target.
**Acceptance:** both JSON outputs pasted below; compare eval scenario metrics with the
`gha-linux` CI artifact `policy-reports` for the same revision (differences between
platforms are expected to be small but non-zero).

Result:
```
revision: b053bc8eaa07d4736f76d1d3f05765f5bc56b277, mac-m5, 1-min load 12.7-13.6 (Docker DGS CI running)
bench json: {"env_label":"mac-m5","physics_steps_per_s":36281,"policy":"alpha_walking.onnx","policy_loop_hz":5395} (mujoco 3.10.0, numpy 2.5.4, cpu_count 10)
eval summary (fell / displacement per scenario), compared with gha-linux policy-reports at 50c7ff1:
  stand_still: mac fell=False dx=0.0016 dy=0.0003; gha-linux fell=False dx=0.0016 dy=0.0003
  walk_forward_0.2: mac fell=False dx=0.0098 dy=0.0048; gha-linux fell=False dx=0.0098 dy=0.0048
  walk_backward_0.15: mac fell=False dx=0.0008 dy=0.0007; gha-linux fell=False dx=0.0008 dy=0.0007
  turn_left_0.5: mac fell=False dx=0.0036 dy=0.0055; gha-linux fell=False dx=0.0036 dy=0.0055
timing (record summary: wall_step_ms_p50 / p99 / max, deadline_miss_ratio): 0.143 / 0.546 / 0.878, 0.0; realtime_factor 96.4
notes: no fall in any scenario, but with these MuJoCo position actuators alpha_walking does not walk at 0.2 m/s:
  record mean_vx_body 0.0009 m/s, tracking error 0.199 m/s. Mac and gha-linux agree on that behaviour.
```

## MAC-04 — Official reference: Hugging Face Space
Status: `awaiting-mac` · Deps: none

Open https://huggingface.co/spaces/pollen-robotics/microduck-simulator in Safari/Chrome.
It runs the same official policies in-browser (MuJoCo WASM). Drive forward and turn.
**Acceptance:** short note of how walking/standing looks there, to compare with MAC-02.

Result:
```
notes:
```

## MAC-07 — PPO library throughput on Apple Silicon (WP3-01)
Status: `done (mac-m5, under load)` · Deps: MAC-01

Same benchmark that chose rsl-rl in ADR-0012 (cloud numbers there). torch and the
libraries are not project dependencies yet, so they go into the project venv by hand
(PyPI wheels only; macOS torch is CPU/MPS, no CUDA download):
```bash
uv pip install torch==2.9.1 tensordict==0.10.0 rsl-rl-lib==5.0.1 torchvision==0.24.1 \
    onnxscript==0.5.7 gitpython==3.1.46 stable-baselines3==2.7.1
uv run python scripts/bench_ppo.py --envs 16 64 256 --iters 20 --out /tmp/wp3-01-mac.json
uv run python scripts/bench_ppo.py --envs 256 --iters 20 --threads 1 --out /tmp/wp3-01-mac-t1.json
uv run python scripts/bench_ppo.py --envs 256 --iters 20 --libraries rsl_rl --workers 4 --out /tmp/wp3-06-mac-w4.json
uv run bipedlab bench microduck --env-label mac-m5 --num-envs 256 --env-workers 0 2 4 8 --seconds 4
```
**Expected:** two JSON reports. Cloud (4 vCPU Xeon 2.1 GHz) measured ~2.2–2.9k raw env
steps/s and ~1.8–1.9k PPO samples/s at 256 envs for both libraries — reference only.
**Acceptance:** both JSONs pasted below (or committed under `docs/experiments/`); note
whether rsl-rl and SB3 stay within ~10 % of each other on the M5. If SB3 is clearly
faster on the Mac, reopen ADR-0012.

Result:
```
revision: 146af37798e1553e4d194864c94080506d42d7a2, mac-m5, 1-min load mean 13.6, max 15.8 (Docker DGS CI running). Reports in docs/experiments/mac07/.
raw_env (num_envs: steps/s): 16: 7321, 64: 7763, 256: 7560
ppo (library, num_envs: samples/s, non_env_fraction):
  rsl_rl 16: 1590, 0.766; 64: 2489, 0.655; 256: 4448, 0.447
  sb3    16: 1751, 0.756; 64: 3472, 0.587; 256: 4519, 0.452
threads=1 run: 256 envs: rsl_rl 4833 (0.406), sb3 5351 (0.392), raw env 7805
workers=4 PPO run, env scaling (bench --env-workers):
  rsl_rl workers=4, 256 envs: 7057 samples/s (0.623); raw env serial 9895
  bench, 256 envs: 0: 9716, 2: 19500 (2.01x), 4: 26613 (2.74x), 8: 30832 (3.17x) env steps/s
notes: at 256 envs, the training setting, rsl-rl and SB3 are within 2 %, so ADR-0012 stays closed on that basis.
  SB3 is ahead by 10 % at 16 envs, 39 % at 64 envs and 11 % at 256 envs with 1 thread; those settings are not the training one.
  Timing varies with the background load, which is why serial raw env reads 7560 and 9716 in different runs.
```

## MAC-06 — First full PPO training run on the M5 (WP3-04, gate G3.3)
Status: `run complete (mac-m5); G3.3 not met: walks on mjcf, not on BAM` · Deps: MAC-01 (MAC-07 helps choose `--workers`)

Trains a walking policy from scratch on the WP3-02 velocity env (upstream's flat
velocity task, CPU MuJoCo). The training stack is the same as MAC-07 (PyPI wheels;
on macOS torch is CPU/MPS only). The run is resumable, so it can be split over
several sessions. Plug the Mac in and keep it awake (`caffeinate -dims` in a second
terminal).

```bash
uv sync --extra policy
uv pip install torch==2.9.1 tensordict==0.10.0 rsl-rl-lib==5.0.1 torchvision==0.24.1 \
    onnxscript==0.5.7 gitpython==3.1.46
uv run python scripts/fetch_upstream.py microduck_rl
# 1) quick timing: 20 iterations, pick the --workers value with the best samples/s
for w in 4 8; do
  uv run python scripts/train_ppo.py --envs 256 --iters 20 --workers $w --out /tmp/mac06-w$w.json
done
# 2) the run itself: 3000 iterations (~18 M samples), checkpoint every 100
uv run python scripts/train_ppo.py --envs 256 --iters 3000 --workers <best> --seed 0 \
    --run-dir runs/mac06-seed0 --save-every 100 --out runs/mac06-seed0/report.json
# resume after an interruption (from the newest model_<n>.pt):
#   uv run python scripts/train_ppo.py --envs 256 --iters <3000-n> --workers <best> \
#       --seed 0 --run-dir runs/mac06-seed0 --save-every 100 --resume runs/mac06-seed0/model_<n>.pt
# 3) evaluate the exported policy in the same runner as the official one (G3.5)
uv run python scripts/eval_velocity_env.py runs/mac06-seed0/policy.onnx --envs 8 \
    --out runs/mac06-seed0/eval.json
uv run bipedlab eval-policy runs/mac06-seed0/policy.onnx --model microduck-walk --actuator bam \
    --repeats 4 --out runs/mac06-seed0/g1-3.json
```
Upstream trained `alpha_walking` with ~4.9 × 10⁹ samples on a GPU. 18 M samples is
about 0.4 % of that, so this first run may only learn to stand or shuffle. That is a
result too: record it honestly. **Do not** commit `runs/` (large binaries). Commit
only `run.json`, `report.json` and the eval JSONs under
`docs/experiments/mac06/`, and record the sha256 of `policy.onnx` from `run.json`.

**Acceptance (G3.3):** the trained policy stands, and walks forward at the G1.3
thresholds (ADR-0011), on the M5. MPS is not tried in this task: the network is
small and the env runs on CPU. If CPU training is clearly too slow, note it and we
add a `--device mps` option.

Result:
```
revision: 0a0c97c51e125f489e34a8e57513f1fb164e0f71 (git_dirty false), mac-m5
timing (workers: samples/s): 4: 7763, 8: 7014 (20 it, 256 envs, load about 10) -> 4
wall time for 3000 iterations: 2368 s (39.5 min), 18.43 M samples, 7784 samples/s; 1-min load mean 12.2, max 19.1 (Docker DGS CI)
reward/step first -> last quarter: 0.0437 -> 0.0975
eval.json (vx at 0.4 cmd, falls), 0 falls in all policy cases:
  mjcf (training actuator): 0.2140 with noise, 0.1999 without; all 8 envs move. 0 cmd: 0.0004 / -0.0001
  bam: 0.0525 with noise (2 of 8 envs move), -0.0001 without. 0 cmd: -0.0001 / -0.0000
  zero-action baseline falls 8 of 8 on both
g1-3.json summary (bam, walk scene, 4 repeats): no falls. walk_forward_0.2: displacement 0.0011 m, mean speed 0.0001 m/s.
  stand_still 0.0004 m, walk_backward_0.15 0.0001 m, turn_left_0.5 0.0019 m; max tilt 1.95 deg
policy.onnx sha256: 3867c12b624e6ed4fa57773e3c8250059b5bd53a8ab764734102bc1c9af0f2d7
notes: G3.3 NOT met. On the mjcf actuator it trained on, the policy walks (about 50 % of a 0.4 cmd). On the BAM
  actuator that G1.3 scores it mostly stands still (0.2 cmd: 0.0001 m/s, threshold +-30 %). Trained with --actuator mjcf,
  the handoff's default, so this is an actuator-model gap on top of the small budget (18 M samples, about 0.4 % of upstream).
  MPS not tried, as the task says.
```

---

## MAC-08 — First full PPO training run for our own biped_v1 (WP4-07, gate G4.3)
Status: `done, first pass (mac-m5); low-friction gap open` · Deps: MAC-01 (MAC-06 can run before or after; they are independent)

Trains a walking policy for **our own robot** (`simulation/custom-biped/biped_v1.xml`,
env `BipedVelocityEnv`, 39 → 10 contract) with domain randomization on, then scores
it and measures the knee torque a learned gait needs. No upstream download is
needed. Same training stack as MAC-06; plug in and keep the Mac awake
(`caffeinate -dims`).

```bash
uv sync --extra policy
uv pip install torch==2.9.1 tensordict==0.10.0 rsl-rl-lib==5.0.1 torchvision==0.24.1 \
    onnxscript==0.5.7 gitpython==3.1.46
# 1) quick timing: pick the --workers value with the best samples/s
for w in 4 8; do
  uv run python scripts/train_ppo.py --robot biped_v1 --dr --envs 256 --iters 20 \
      --workers $w --out /tmp/mac08-w$w.json
done
# 2) the run: 3000 iterations (~18 M samples), checkpoint every 100
uv run python scripts/train_ppo.py --robot biped_v1 --dr --envs 256 --iters 3000 \
    --workers <best> --seed 0 --run-dir runs/mac08-seed0 --save-every 100 \
    --out runs/mac08-seed0/report.json
# resume after an interruption (from the newest model_<n>.pt):
#   uv run python scripts/train_ppo.py --robot biped_v1 --dr --envs 256 --iters <3000-n> \
#       --workers <best> --seed 0 --run-dir runs/mac08-seed0 --save-every 100 \
#       --resume runs/mac08-seed0/model_<n>.pt
# 3) robustness + knee torque report (with and without DR, plus one-factor cases:
#    foot friction 0.5/1.5, forward/sideways push, 20/40/60 ms latency; WP4-12)
uv run python scripts/eval_biped.py runs/mac08-seed0/policy.onnx --dr --robustness --envs 8 \
    --seconds 10 --out runs/mac08-seed0/eval.json
```
Since WP4-11 (2026-11-06) the env applies the runtime's 10 rad/s target slew limit,
penalises steps it cuts (`action_slew`), uses an action-rate weight of −0.5 from the
start, and biped_v1 trains with `--init-std 0.5` by default. Before that change the
smoke policies jittered (slew limit in 488–500 of 500 runtime periods); after it, 0 of
500 at 300 iterations in the cloud. Pull `main` before step 1 so the run uses it; check
that `report.json` says `"init_std": 0.5`.

This is a first attempt; it may only learn to stand or shuffle. Record that honestly.
In `eval.json`, `slew_limited_fraction` should stay near 0 (below 0.05) in every case:
a higher value means the policy jitters again and would be cut by the runtime.
**Do not** commit `runs/`. Commit `run.json`,
`report.json` and `eval.json` under `docs/experiments/mac08/` and the sha256 of
`policy.onnx`.

**Acceptance (G4.3, first pass):** with DR on, the policy walks forward at the
0.15 and 0.25 m/s commands (`vx_mean` ≥ 50 % of the command) with no falls in 10 s,
on the M5. Report the knee `p95` and `max`: a p95 near 0.52 N·m means the
XL330-class knee saturates, which feeds the servo choice in WP4-05.

Result:
```
revision: b3165540ffea0f6a007592eb5dab49a396a237f0 (git_dirty false), mac-m5
timing (workers: samples/s): 4: 12985, 8: 14532 (20 it, 256 envs, load about 5) -> 8
wall time for 3000 iterations: 2189 s (36.5 min), 18.43 M samples, 8419 samples/s
reward/step first -> last quarter: 0.0535 -> 0.1179 (init_std 0.5)
eval.json (vx_mean at 0.15 / 0.25, falls, with and without DR):
  DR on:  0.1453 (97 %) / 0.2228 (89 %), 0 falls in 10 s, 8 envs
  DR off: 0.1506 / 0.2267, 0 falls
  stand (0 cmd) vx: DR off 0.0166, DR on 0.0103; 0 falls. Yaw 0.8 rad/s: 0 falls, yaw rate not scored by this report.
robustness cases (falls / vx_mean each: friction 0.5, 1.5, push fwd, side, latency 20/40/60 ms):
  friction 0.5: 0 / 0.1506  INVALID, see notes (bit-identical to nominal)
  friction 1.5: 0 / 0.1551
  push fwd:     0 / 0.1519
  push side:    0 / 0.1503
  latency 20:   0 / 0.1069 (71 %)
  latency 40:   0 / 0.0655 (44 %, below the 50 % walking threshold)
  latency 60:   0 / 0.0529 (35 %, below the 50 % walking threshold)
slew_limited_fraction (max over cases): 0.05 (latency 60 ms; not below the 0.05 limit). Next highest 0.00275; nominal 0.
knee torque p95 / max (N*m): DR on 0.25 cmd 0.251 / 0.520; DR off 0.236 / 0.520;
  latency 60 ms 0.520 / 0.520 (saturated at p95). Max reaches the 0.52 limit in most cases.
policy.onnx sha256: bf5fff2d1fe60ef3fd91b524de9d9717097b7e4a65b35650629ad2d29a5b58ef
notes:
  G4.3 first-pass criterion (DR on, >= 50 % of command, no falls in 10 s) is met.
  Friction below 1.0 has no effect in this env: MuJoCo uses the element-wise max of
  the two geoms at equal priority and the floor is 1.0. Measured contact mu with foot
  0.5 / 0.7 / 0.99 = 1.0; foot 1.3 / 1.5 = 1.3 / 1.5. So the friction_0.5 case tested
  nothing, and DR FOOT_FRICTION (0.7, 1.3) only ever trained on 1.0-1.3.
  Diagnostic, foot and floor lowered together, DR off (friction_diagnostic.json):
  mu 0.7 / 0.5 / 0.3: vx at 0.15 = 0.1508 / 0.1503 / 0.1512; at 0.25 = 0.2220 / 0.2193 / 0.2136; 0 falls.
  Timing was under load: Docker DGS CI containers on the same Mac, 1-min load mean 14.2,
  max 20.9 during the run. The wall time is not a clean M5 number.
```

---

## MAC-09 — Rust runtime drives the sim on the M5 (WP4-04, gate G4.4)
Status: `failed (mac-m5): real-time pacing test` · Deps: MAC-01

G4.4 needs the runtime-in-the-loop run on the M5 as well as on gha-linux. Rust
from Homebrew (review: Homebrew formula; do **not** use the rustup curl-pipe
script):
```bash
brew install rust
rustc --version && cargo --version
(cd runtime && cargo test --locked && cargo build --locked --release)
BIPEDLAB_REQUIRE_RUNTIME=1 uv run pytest -m runtime -rs
uv run python scripts/runtime_sim.py --seconds 30 --realtime --out /tmp/mac09-realtime.json
```
**Acceptance (G4.4, mac-m5 part):** all `runtime` tests pass; in the 30 s
real-time run, 1500 periods, `deadline_misses` ≤ 15 (1 %), no latched fault.

Result:
```
revision: b053bc8eaa07d4736f76d1d3f05765f5bc56b277 (main 50c7ff1 merged), mac-m5
rustc version: rustc 1.98.1 (48a229cea 2026-09-01), existing rustup toolchain. Deviation:
  `brew install rust` not run (brew not on PATH; toolchain meets rust-version 1.80).
pytest -m runtime: 22 passed, 1 FAILED. cargo test and release build completed.
  test_realtime_pacing_holds_50_hz: wall_s 6.32 (limit 5.5); failed 6 of 6 runs (6.16-6.32).
realtime 30 s (periods, deadline_misses, period_max_us, compute_mean_us):
  1500, 0, 36161, 16.2; latched 0, supervisor events all 0; wall_s 38.86 for 30 s sim
  logged repeat: 1500, 0, 38689, 16.5; latched 0; wall_s 37.39
  period from the log: mean 24.94 ms, p50 24.98, p95 29.60, p99 30.04, max 38.69;
  94.1 % of periods over 20.5 ms. That is about 40 Hz, not 50 Hz.
notes:
  Acceptance is NOT met: not all runtime tests pass. The numeric line (misses <= 15, no
  latch) passes, but it does not detect this: deadline_misses counts only compute > period
  (main.rs), and compute is about 16 us. Sleep overshoot is invisible to it.
  The loop sleeps a relative `period - elapsed` with no absolute deadline, so overshoot
  accumulates. Measured python sleep(19.5 ms) overshoot here: p50 3.81, p99 9.76 ms at load 19.
  Measured with Docker DGS CI load (1-min load 11-21), so the load's share is not separated.
  Second, intermittent failure: tests/runtime/test_command_channel.py::test_second_sender_is_ignored
  failed 3 of 10 runs (command max 0.3, expected 0.1), passing in the -m runtime run above.
  Unproven hypothesis: test race. Both senders start before the runtime binds, so either
  can be the first packet it locks onto. Full suite at this revision: 180 passed, 2 failed.
  Tests and runtime left unchanged. Simulation only; no hardware connected.
```

---

## MAC-10 — Drive the simulated robot from the keyboard (WP4-14, optional)
Status: `done (mac-m5, keystrokes scripted through a pty; owner's look at the chart still open)` · Deps: MAC-09 (Rust installed)

Two terminals. The sim + runtime run 60 s in real time; the teleop sends what you
type. Watch the logged command and the report.
```bash
# terminal 1
uv run python scripts/runtime_sim.py --seconds 60 --realtime --log /tmp/mac10.bplog \
    --env-label mac-m5 --out /tmp/mac10.json -- --command-udp 127.0.0.1:9870
# terminal 2 (start within a few seconds): w/s a/d q/e, space = stop, x = quit
uv run python scripts/teleop.py --port 9870 --keyboard
open /tmp/mac10.html
```
Optional, with a key (WP4-17): run `uv run python scripts/teleop.py --new-key /tmp/mac10.key`
once, add `--command-key /tmp/mac10.key` after `--command-udp 127.0.0.1:9870` in
terminal 1 and `--key-file /tmp/mac10.key` in terminal 2. Same acceptance.

**Acceptance:** the "Dönüş hızı" chart follows your q/e presses, the command
drops to zero within ~1 s after you quit the teleop, `deadline_misses` ≤ 30, and
the `channel` counts in `/tmp/mac10.json` show `foreign` 0.

Result:
```
revision: cea88931f7044b40515a2f800dc38e17bab496c5, mac-m5, 1-min load 3.6-4.3 (Docker DGS CI containers up)
how it was driven: the real `teleop.py --keyboard` (curses) under a pty, keys sent by a script, not typed by a person
  (docs/experiments/mac10/keys_driver.py.txt). Keys: q q, space, e e, space, w, q, x. Two runs: no key, and with --command-key/--key-file (WP4-17).
deadline_misses / periods: no key 0 / 3000; key 0 / 3000. No latched fault, supervisor events 0, gate_rejected 0, wall 59.99 s both.
channel (accepted, malformed, unauthenticated, stale, foreign, hellos): no key 580, 0, 0, 0, 0, 1; key 584, 0, 0, 0, 0, 1
command plateaus from the log, both runs: q q wz +0.5; e e wz -0.5; space 0; w vx 0.075; w+q vx 0.075, wz 0.25.
drop after quit, x pressed at a nonzero command (0.075, 0, 0.25): ramp starts 0.44 / 0.47 s, command zero 0.58 / 0.61 s after x (no key / key).
notes: numeric acceptance met. The "Dönüş hızı" chart's command line follows the presses in the logged data; nobody has looked
  at the chart yet, that stays for the owner: runs/mac10/nokey.html, keyed.html (not committed).
  The actual yaw stays 0 (gyro z 0.0000 throughout): this command list runs the default `stand` controller, which ignores commands.
  So this proves the channel and the gate, not that the robot turns. Summary: docs/experiments/mac10/summary.json.
```

---

## MAC-11 — Re-check after the WP4-16 fixes (runtime pacing, friction, latency)
Status: `failed (mac-m5): runtime part passed; retrained policy stands and does not walk (G4.3 not met)` · Deps: MAC-08, MAC-09 done once; WP4-16 merged

MAC-09 found the runtime at ~40 Hz (relative sleep after the state read) and MAC-08
found that foot friction below 1.0 was never applied. WP4-16 fixes both and adds
0–40 ms action delay to DR. Pull `main`, then:
```bash
(cd runtime && cargo test --locked && cargo build --locked --release)
BIPEDLAB_REQUIRE_RUNTIME=1 uv run pytest -m runtime -rs
uv run python scripts/runtime_sim.py --seconds 30 --realtime --out /tmp/mac11-realtime.json
# retrain biped_v1 (same command as MAC-08 step 2, new run dir) and evaluate
uv run python scripts/train_ppo.py --robot biped_v1 --dr --envs 256 --iters 3000 \
    --workers <best from MAC-08> --seed 0 --run-dir runs/mac11-seed0 --save-every 100 \
    --out runs/mac11-seed0/report.json
uv run python scripts/eval_biped.py runs/mac11-seed0/policy.onnx --dr --robustness --envs 8 \
    --seconds 10 --out runs/mac11-seed0/eval.json
```
**Acceptance:** all `runtime` tests pass; 30 s real-time run: 1500 periods,
`deadline_misses` ≤ 15 (now real overruns), no latched fault, wall time 29.5–31 s
(note the machine load). Retrained policy: G4.3 first pass as in MAC-08 still met,
`friction_0.5` now differs from nominal, and `latency_40ms` reaches ≥ 50 % of the
command. Commit `report.json`, `run.json`, `eval.json` under `docs/experiments/mac11/`.

Result:
```
revision: e5c05586b26476309ca2676d4976a364fe2b0ba8, mac-m5
pytest -m runtime: 27 passed; cargo test 39 passed; release build ok
realtime 30 s (periods, deadline_misses, period_max_us, wall_s, load), three runs, logged:
  1500, 0, 26869, 29.98, load 8.3
  1500, 0, 33647, 29.98, load 8.6
  1500, 1, 45294, 29.98, load 7.7
  no latched fault, supervisor events 0. Period mean 20.00 ms, p50 19.8-19.9, p99 24.7-24.9, min 1.5-12.6 (catch-up after overruns).
retrain: 3000 iterations in two segments, seed 0, 8 workers, 256 envs, 18.43 M samples. 0-1800 at e5c0558 (about 21 min), stopped by the owner;
  1800-3000 at cea8893, resumed from model_1800.pt, 409 s, load about 4-9. The two revisions differ only in docs. report.json/run.json cover the second
  segment only; the first is in history_from_0.json.
  reward/step first -> last quarter of the 3000: 0.0472 -> 0.1026 (MAC-08: 0.0535 -> 0.1179)
eval.json (vx_mean at 0.15 / 0.25, falls, 8 envs, 10 s):
  DR on:  0.0000 / 0.0000, 0 falls (0 %, G4.3 needs >= 50 %). MAC-08 on the same criterion: 0.1453 / 0.2228
  DR off: 0.0000 / 0.0000, 0 falls
  air_time term 0 in every case: no foot leaves the floor. Knee p95 close to the zero-action baseline (below).
robustness (falls / vx_mean at the 0.15 cmd):
  nominal      0 / -0.000006
  friction 0.5 0 / -0.000007, knee p95 0.0384 vs nominal 0.0357: no longer bit-identical, but the robot is standing, so this does not test walking on low friction
  friction 1.5 0 / +0.000000
  push fwd     0 / +0.000020; push side 0 / -0.000080
  latency 20 / 40 / 60 ms: 0 falls each; +0.000006 / +0.000011 / +0.000017 (latency_40ms 0 %, acceptance needs >= 50 %)
knee p95 / max (N*m): DR on 0.15 cmd 0.046 / 0.381; DR on 0.25 cmd 0.040 / 0.520; DR off 0.25 cmd 0.029 / 0.416; highest p95 0.093 (DR on, yaw 0.8).
  Zero-action baseline p95 0.035 (DR off) / 0.047 (DR on). MAC-08 walking: 0.251 / 0.520. This does NOT measure the knee torque of a gait, do not feed it to WP4-05.
slew_limited_fraction: 0.0 in every case. Meaningless here: nothing moves.
policy.onnx sha256: c0e840297f12e06d6f80e79ce7b44c9f388cb5e7f0061f2963a5de565ce4e2ea
notes: policy acceptance NOT met. G4.3 first pass, friction and latency_40ms cannot be claimed.
  Not a resume or export problem: the training history already has air_time 0.0000 from about iteration 600, well before the resume.
  Mean per 300 iterations, MAC-11 vs MAC-08 air_time: 0-300 0.0057 / 0.0049; 300-600 0.0010 / 0.0178; 600-900 0.0000 / 0.0276; then 0.0000 / about 0.029.
  knee_torque_mean falls to 0.074 (MAC-08 0.157). PPO settled on standing still, which already scores 0.102 reward/step against MAC-08's 0.118-0.121.
  reward_rose is true in report.json and does not catch this; neither did the cloud DR smoke (WP4-16), which only checks that reward rises.
  Unproven hypothesis: the WP4-16 DR change (0-40 ms action delay, friction now down to 0.7) makes the gait harder to find on this budget, so
  standing wins. The two changes are not separated, and this is one seed. No further retrain was run.
```

---

## Later Mac tasks (not ready yet)
- MAC-05 live telemetry overlay (after WP2-01)
