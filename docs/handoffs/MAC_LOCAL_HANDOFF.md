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
Status: `awaiting-mac` · Deps: WP0-04

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
revision:
date:
output (tail):
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
Status: `awaiting-mac` · Deps: MAC-01

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
revision:
bench json:
eval summary (fell / displacement per scenario):
timing (record summary: wall_step_ms_p50 / p99 / max, deadline_miss_ratio):
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
Status: `awaiting-mac` · Deps: MAC-01

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
revision:
raw_env (num_envs: steps/s):
ppo (library, num_envs: samples/s, non_env_fraction):
threads=1 run:
workers=4 PPO run, env scaling (bench --env-workers):
```

## MAC-06 — First full PPO training run on the M5 (WP3-04, gate G3.3)
Status: `awaiting-mac` · Deps: MAC-01 (MAC-07 helps choose `--workers`)

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
revision:
timing (workers: samples/s):
wall time for 3000 iterations:
reward/step first -> last quarter:
eval.json (vx at 0.4 cmd, falls):
g1-3.json summary:
policy.onnx sha256:
notes:
```

---

## MAC-08 — First full PPO training run for our own biped_v1 (WP4-07, gate G4.3)
Status: `awaiting-mac` · Deps: MAC-01 (MAC-06 can run before or after; they are independent)

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
revision:
timing (workers: samples/s):
wall time for 3000 iterations:
reward/step first -> last quarter:
eval.json (vx_mean at 0.15 / 0.25, falls, with and without DR):
robustness cases (falls / vx_mean each: friction 0.5, 1.5, push fwd, side, latency 20/40/60 ms):
slew_limited_fraction (max over cases):
knee torque p95 / max (N*m):
policy.onnx sha256:
notes:
```

---

## MAC-09 — Rust runtime drives the sim on the M5 (WP4-04, gate G4.4)
Status: `awaiting-mac` · Deps: MAC-01

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
revision:
rustc version:
pytest -m runtime:
realtime 30 s (periods, deadline_misses, period_max_us, compute_mean_us):
notes:
```

---

## MAC-10 — Drive the simulated robot from the keyboard (WP4-14, optional)
Status: `awaiting-mac` · Deps: MAC-09 (Rust installed)

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
revision:
deadline_misses / periods:
channel (accepted, malformed, unauthenticated, stale, foreign, hellos):
notes:
```

---

## MAC-11 — Re-check after the WP4-16 fixes (runtime pacing, friction, latency)
Status: `awaiting-mac` · Deps: MAC-08, MAC-09 done once; WP4-16 merged

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
revision:
pytest -m runtime:
realtime 30 s (periods, deadline_misses, period_max_us, wall_s, load):
retrain (wall time, reward first -> last quarter):
eval: vx_mean 0.15 / 0.25 (DR on), falls, friction_0.5 / 1.5, latency 20/40/60 ms:
knee p95 / max:
policy.onnx sha256:
notes:
```

---

## MAC-12 — Microduck PPO run with the BAM actuator (WP3-10, gate G3.3)
Status: `done (mac-m5, 2026-11-19): G3.3 not met` · Deps: MAC-01, MAC-06 done once

MAC-06 trained with the default MJCF position actuators: the policy walks there
(~0.20 m/s at the 0.4 command) but barely moves under BAM, which G1.3/G3.3 measure
with (upstream also trains with BAM, ADR-0011). Same run, trained on BAM. BAM steps
are ~4x slower than MJCF (cloud: 5 it 22 s vs 6 s), so expect several hours; the run
is resumable. Cloud check (2026-11-06): a 60-iteration BAM smoke rose
(0.0029 -> 0.0146 reward/step, 46 s on 4 vCPU).
```bash
uv run python scripts/train_ppo.py --actuator bam --envs 256 --iters 20 --workers 8 \
    --out /tmp/mac12-timing.json          # samples/s; pick --workers as in MAC-06
uv run python scripts/train_ppo.py --actuator bam --dr --envs 256 --iters 3000 --workers <best> \
    --seed 0 --run-dir runs/mac12-seed0 --save-every 100 --out runs/mac12-seed0/report.json
uv run python scripts/eval_velocity_env.py runs/mac12-seed0/policy.onnx --actuators bam \
    --envs 8 --out runs/mac12-seed0/eval.json
uv run bipedlab eval-policy runs/mac12-seed0/policy.onnx --model microduck-walk --actuator bam \
    --repeats 4 --out runs/mac12-seed0/g1-3.json
```
**Acceptance (G3.3):** per G1.3 with BAM: stands ≥ 20 s and tracks vx = 0.2 m/s
within ±30 % over 10 s without falling, on the M5. If it only shuffles, record it
honestly with the numbers. Commit the JSONs under `docs/experiments/mac12/`.

Result (env: mac-m5, 2026-11-19): G3.3 NOT met, the policy stands still
```
revision: 75d5eea (same code as main 6164ca5), git_dirty false
timing (workers: samples/s): 4: 3392, 8: 3546 -> 8 workers (load avg 12-18)
wall time for 3000 iterations: 3496 s (58 min; 18.4 M samples)
reward/step first -> last quarter: 0.0427 -> 0.0839 (rose)
g1-3.json (4 repeats x 10 s): stand_still no fall, tilt < 1 deg; walk_forward_0.2
  vx -0.0001 (0 % of the command), no fall; walk_backward_0.15 vx 0.0; turn_left_0.5
  no turn; 0 falls in every scenario
eval.json (BAM, 8 envs, 0.4 command): vx -0.00005 / -0.00007 (noise on / off), 0 falls
policy.onnx sha256: 1216c60502330ed0ac16fec745ff4036218447843d603c38908f51719f905fc7
notes: same standing optimum as biped_v1 MAC-11/13B. air_time 0.0071 (first 250 it)
  -> 0.0009 by it 500 -> 0.0004 at the end; the reward keeps rising from upright,
  pose and head_pose terms while track_linear_velocity stays ~0.008/step.
  JSONs in docs/experiments/mac12/ (report.json history every 10th iteration).
```

---

## MAC-13 — Which WP4-16 DR change stopped the gait? (WP4-18 ablation)
Status: `done (mac-m5, 2026-11-19)` · Deps: MAC-11 done

MAC-08 (old DR) walked; MAC-11 (WP4-16 DR: 0–40 ms action delay and foot friction
that now really goes down to 0.7) only stands: `air_time` hits 0 from ~600 iterations
and standing already earns ~0.10 reward/step. Two runs, each switching one change
back off (same seed and settings as MAC-11 otherwise, fresh run dirs):
```bash
# A: no DR action delay, friction 0.7-1.3 (effective since WP4-16)
uv run python scripts/train_ppo.py --robot biped_v1 --dr --dr-delay-max 0 --envs 256 \
    --iters 3000 --workers <best> --seed 0 --run-dir runs/mac13a --save-every 100 \
    --out runs/mac13a/report.json
# B: DR action delay 0-40 ms, friction 1.0-1.3 (what pre-WP4-16 DR effectively was)
uv run python scripts/train_ppo.py --robot biped_v1 --dr --dr-friction 1.0 1.3 --envs 256 \
    --iters 3000 --workers <best> --seed 0 --run-dir runs/mac13b --save-every 100 \
    --out runs/mac13b/report.json
for r in mac13a mac13b; do
  uv run python scripts/eval_biped.py runs/$r/policy.onnx --dr --robustness --envs 8 \
      --seconds 10 --out runs/$r/eval.json
done
```
Watch `air_time` in the training log: if it is 0 for 500+ iterations, the run has
settled on standing (you may stop it early and record that). Commit `report.json`,
`run.json`, `eval.json` under `docs/experiments/mac13/`.

Result (env: mac-m5, 2026-11-19):
```
revision: 75d5eea (same code as main 6164ca5), git_dirty false; 8 workers, load avg 10-13
A (no delay): WALKS. vx 0.101 / 0.163 at 0.15 / 0.25 (DR on; 67 % / 65 %), 0 falls;
  nominal (DR off) 0.135 / 0.216; latency 20/40/60 ms: 0.090 / 0.063 / 0.046
  (60 / 42 / 31 % of 0.15), 0 falls; friction_0.5 0.134, friction_1.5 0.136;
  pushes 0.3 m/s fwd/side: 0 falls; knee p95 0.195 / 0.216, max 0.52 (DR on);
  air_time (mean per 250 it) 0.005 -> 0.025 by it 500 -> 0.030 at 3000, never 0;
  reward/step 0.022 -> 0.127; wall 2383 s
B (friction >= 1): STANDS. vx 0.000 in every case (DR on/off, all robustness cases),
  0 falls; friction_0.5 0.000; air_time 0.0067 -> 0.0001 by it 500, ~0 from it 629
  to 3000; reward/step plateaus ~0.099 from it 1500; wall 2190 s
policy.onnx sha256: A 9a2823b3fe2e4377e706050ed805f092de42d0cdad2203d628e3ff5ad0b3a7f6
                    B 09b31691ae7027f916c24d443ca6a728b8a3b80ede7e51fbc35eb16e5b24a3b1
notes: the 0-40 ms DR action delay stops the gait, not the lower friction. G4.3
  first pass holds for A; latency_40ms (42 %) misses the MAC-11 >= 50 % target.
  JSONs in docs/experiments/mac13/{a,b}/ (report.json history every 10th iteration).
```

---

## MAC-14 — biped_v1 with the DR delay ramped in (WP4-19, gate G4.3)
Status: `done (mac-m5, 2026-11-19)` · Deps: MAC-13 done, WP4-19 merged

MAC-13 (mac-m5): run A (no DR delay) walks, run B (delay 0–40 ms from the start,
friction ≥ 1) only stands, so the delay is what stopped the gait. Since WP4-19 the DR
delay follows `DR_DELAY_STAGES`: 0 until iteration 1000, 0–20 ms until 1500, then
0–40 ms. One run with the default DR (no ablation flags):
```bash
uv run python scripts/train_ppo.py --robot biped_v1 --dr --envs 256 --iters 3000 \
    --workers <best> --seed 0 --run-dir runs/mac14 --save-every 100 \
    --out runs/mac14/report.json
uv run python scripts/eval_biped.py runs/mac14/policy.onnx --dr --robustness --envs 8 \
    --seconds 10 --out runs/mac14/eval.json
```
Watch `air_time` around iterations 1000 and 1500: if it drops to 0 after a stage
widens the delay, record the iteration. Commit `report.json`, `run.json`, `eval.json`
under `docs/experiments/mac14/`.

Result (env: mac-m5, 2026-11-19): G4.3 first pass met, and the latency target is met too
```
revision: a944b05 (main), git_dirty false; 8 workers, load avg 10-14; wall 1815 s
walks? YES. vx 0.150 / 0.228 at 0.15 / 0.25 (DR on; 100 % / 91 %), DR off 0.179 / 0.264;
  0 falls in every case; latency 20/40/60 ms: 0.148 / 0.129 / 0.114 (99 / 86 / 76 %
  of 0.15); friction_0.5 0.175, friction_1.5 0.185; push fwd / side 0.180 / 0.177,
  0 falls; knee p95 0.188 / 0.223, max 0.41 / 0.51 N.m (DR on, 0.15 / 0.25);
  air_time (100-it mean) 0.0279 at 1000, 0.0277 at 1500, 0.0289 at 2900: no drop
  when a delay stage widens; reward/step 0.053 -> 0.110 (first -> last quarter)
policy.onnx sha256: 83766219233e60a14315ea78e9ee0ff68fcfa680787542f215d0cc8e8be6f78f
notes: the gait forms by it ~500 (delay 0) and survives the 0-20 ms and 0-40 ms
  stages. Against MAC-13 A (no delay): DR-on tracking 67 -> 100 %, latency_40ms
  42 -> 86 % (MAC-11 target >= 50 %). Stand command: vx 0.004 (DR on), no drift.
  JSONs in docs/experiments/mac14/ (report.json history every 10th iteration).
```

---

## MAC-15 — Microduck: fine-tune the MAC-06 gait on BAM (WP3-11, gate G3.3)
Status: `done (mac-m5, 2026-11-19): G3.3 not met` · Deps: MAC-06 and MAC-12 done

MAC-12 (BAM from scratch, 3000 it, mac-m5) only learned to stand: vx −0.0001 at the
0.2 command, reward 0.043 → 0.084 almost all from upright/pose/head terms. MAC-06
(MJCF actuators) did learn a gait with the same budget. This run starts from the
MAC-06 weights and keeps training on BAM with DR, so PPO adapts an existing gait
instead of finding one under the weaker, back-EMF-damped servo model. `--resume`
across actuators works (cloud check: an MJCF checkpoint resumes on BAM; the iteration
count and curricula continue from 3000).
```bash
ls runs/mac06-seed0/model_3000.pt        # the MAC-06 final checkpoint
uv run python scripts/train_ppo.py --actuator bam --dr --envs 256 --iters 3000 \
    --workers <best> --seed 0 --run-dir runs/mac15-seed0 --save-every 100 \
    --resume runs/mac06-seed0/model_3000.pt --out runs/mac15-seed0/report.json
uv run python scripts/eval_velocity_env.py runs/mac15-seed0/policy.onnx --actuators bam \
    --envs 8 --out runs/mac15-seed0/eval.json
uv run bipedlab eval-policy runs/mac15-seed0/policy.onnx --model microduck-walk --actuator bam \
    --repeats 4 --out runs/mac15-seed0/g1-3.json
```
If the MAC-06 checkpoint is gone, write that down and skip the task. Watch the vx
tracking term in the log: if it falls to the MAC-12 level within ~300 iterations,
the gait was lost; record the iteration. Commit the JSONs under `docs/experiments/mac15/`.

**Acceptance (G3.3):** as MAC-12 (G1.3 with BAM: stands ≥ 20 s, vx = 0.2 m/s within
±30 % over 10 s, no falls).

Result (env: mac-m5, 2026-11-19): G3.3 NOT met, walks on BAM only at forward commands >= 0.3
```
revision: 59d9553 (main), git_dirty false; resumed from mac06-seed0/model_3000.pt
  (it 3000 -> 6000), 8 workers; a VM used ~4.4 cores during the run (load avg 9-32)
wall time: 3502 s (58 min)
reward/step first -> last quarter, tracking term trend: 0.0869 -> 0.0989 (rose);
  track_linear_velocity 0.0246 (first 100 it) -> 0.0256 (last), never near the MAC-12
  0.008 level, so the gait was not lost; air_time 0.013-0.015 throughout
eval.json (vx at 0.4 cmd, falls): 0.203 / 0.197 (noise on / off), 0 falls; zero
  command vx 0.000; zero-action baseline 8/8 falls
g1-3.json (4 repeats x 10 s): no falls anywhere, tilt < 2 deg; walk_forward_0.2
  vx 0.0000 (0 %), walk_backward_0.15 0.0000, turn_left_0.5 none
diag (1 repeat, docs/experiments/mac15/diag/): BAM sweep forward 0.1 / 0.3 / 0.4 ->
  vx 0.0001 / 0.120 / 0.149, backward_0.3 / lateral_0.3 / turn_1.0 ~0; BAM walking
  with --init settle / straight: vx at 0.2 -0.0002 / 0.011; MJCF walking (upstream
  init): vx 0.117 at 0.2
policy.onnx sha256: ddda1a28ae7a344b19d67ca49be86e251e33f37dbd7d23c05548e60b6a18c518
notes: fine-tuning kept the forward gait on BAM (MAC-12 from scratch had none), but
  it starts only above a command threshold between 0.2 and 0.3 m/s, and only
  forwards. The same policy on MJCF walks at 0.2, so BAM's weaker response moves the
  threshold. G1.3 measures exactly the 0.2 point, so it reads 0.
```

---

## MAC-16 — The trained biped_v1 policy walks through the Rust runtime (WP4-20, G4.3 + G4.4)
Status: `done (mac-m5, 2026-11-19)` · Deps: MAC-14 done (its `runs/mac14/policy.onnx`), WP4-20 merged

G4.3 was measured in the Python env and G4.4 with the stand controller. This runs
the MAC-14 policy inside the runtime (safety supervisor, 50 Hz wall clock) against
the sim bridge, and reads the walk from the bridge's WP4-20 metrics
(`vx_body_mean`, `distance_xy_m`, `fell`, from 2 s on).
```bash
(cd runtime && cargo build --locked --release)
P=runs/mac14   # MAC-14 run dir (absolute path if you work in another worktree)
uv run python scripts/export_mlp.py $P/policy.onnx --out $P/policy.bpml \
    --rust-bin runtime/target/release/mlp-eval > /tmp/mac16-mlp.json
for v in 0.15 0.25; do
  uv run python scripts/runtime_sim.py --seconds 20 --realtime --env-label mac-m5 \
      --log runs/mac16/v$v.bplog --out runs/mac16/v$v.json --require-clean-exit -- \
      --controller policy:$P/policy.bpml --command 0:20:$v:0:0
done
```
**Acceptance:** for both commands `fell` false, no latched fault, 1000 periods,
`deadline_misses` ≤ 10, and `vx_body_mean` ≥ 50 % of the command (G4.3 first-pass
bar); note how far it is from MAC-14's Python-env vx (0.150 / 0.228). Commit the two
JSONs and `mac16-mlp.json` under `docs/experiments/mac16/` (not the `.bplog`/`.npz`).

Result (env: mac-m5, 2026-11-19): met for both commands
```
revision: e9efa90 (main); policy = MAC-14 policy.onnx (sha256 83766219...e8be6f78f),
  policy.bpml sha256 7a917195ce120ed5965d725ac2195250583da210b6348c9cdef018c7c7ecbba7
  load avg ~9 (no training running; a VM was using several cores)
mlp parity (max abs diff): rust vs onnxruntime 1.43e-6, numpy vs onnxruntime 1.91e-6
v0.15: vx_body_mean 0.194 (129 %), distance 1.714 m, fell false, latched 0,
  deadline_misses 0, period_max_us 29735; 1000 periods, wall 19.98 s, tilt max 2.7 deg,
  wz_mean -0.190 rad/s, slew_limit events 23
v0.25: vx_body_mean 0.285 (114 %), distance 3.286 m, fell false, latched 0,
  deadline_misses 0, period_max_us 29807; 1000 periods, wall 19.99 s, tilt max 2.4 deg,
  wz_mean -0.148 rad/s, slew_limit events 71
notes: faster than MAC-14 in the Python env with DR on (0.150 / 0.228). The runtime
  sim has no DR, and it is closest to MAC-14 with DR off (0.179 / 0.264). The robot
  yaws right by about 0.15-0.19 rad/s at a zero yaw command, so it walks a curve:
  distance_xy (displacement) is less than vx * 18 s. Timing: the runtime counts 0 misses,
  but the bridge-side wall_dt (telemetry recorder) jitters: mean 20.0 ms, p1 / p50 /
  p99 12 / 20 / 29 ms, and ~430 of 1000 steps > 20.5 ms. Long and short steps
  alternate, so the schedule holds (wall 19.98 s). Worth a cloud look at where the
  jitter comes from (bridge vs. runtime) before hardware.
```

---

## MAC-17 — Microduck: tighter velocity tracking on BAM (WP3-12, gate G3.3)
Status: `done (mac-m5, 2026-11-19): G3.3 not met` · Deps: MAC-15 done (its final checkpoint), WP3-12 merged

MAC-15 (mac-m5) keeps the MAC-06 gait on BAM (0.4 → 0.203 m/s, 0 falls) but only
walks above a command threshold: 0.1 → 0.000, 0.2 → 0.000, 0.3 → 0.120. Upstream's
linear tracking kernel (std² 0.1) still pays a standing policy 67 % of the tracking
reward at 0.2 m/s; our biped uses 0.05 and tracks 0.15 fully. This run continues
MAC-15 with `--track-lin-std2 0.05` (standing at 0.2 then earns 45 %).
```bash
ls runs/mac15-seed0/model_6000.pt        # MAC-15 final checkpoint (absolute path if needed)
uv run python scripts/train_ppo.py --actuator bam --dr --envs 256 --iters 2000 \
    --workers <best> --seed 0 --track-lin-std2 0.05 --run-dir runs/mac17-seed0 \
    --save-every 100 --resume runs/mac15-seed0/model_6000.pt --out runs/mac17-seed0/report.json
uv run python scripts/eval_velocity_env.py runs/mac17-seed0/policy.onnx --actuators bam \
    --envs 8 --out runs/mac17-seed0/eval.json
uv run bipedlab eval-policy runs/mac17-seed0/policy.onnx --model microduck-walk --actuator bam \
    --repeats 4 --out runs/mac17-seed0/g1-3.json
```
Also repeat MAC-15's forward sweep (0.1 / 0.2 / 0.3 / 0.4) on the new policy. Commit
the JSONs under `docs/experiments/mac17/`.

**Acceptance (G3.3):** as MAC-12 (G1.3 with BAM). If the threshold only moves, record
the new sweep.

Result (env: mac-m5, 2026-11-19): G3.3 NOT met, the threshold did not move down
```
revision: 441e731 (main), git_dirty false; resumed from mac15-seed0/model_6000.pt
  (it 6000 -> 8000), --track-lin-std2 0.05, 8 workers, load avg 9-16
wall time: 2308 s (38 min)
reward first -> last quarter, tracking term trend: 0.0911 -> 0.0974 (rose);
  track_linear_velocity 0.0198 (first 100 it) -> 0.0210-0.0215 (from it 1200), on the
  tighter kernel (MAC-15's 0.025 was on std2 0.1, so not directly comparable);
  air_time 0.013-0.014 throughout
sweep vx at 0.1 / 0.2 / 0.3 / 0.4 (BAM): 0.0001 / 0.0002 / 0.078 / 0.131
  (0.2 from g1-3.json, others from sweep-bam.json; MAC-15: 0.0001 / 0.0000 / 0.120 /
  0.149); backward_0.3, lateral_0.3, turn_1.0 ~0
eval.json (0.4 cmd): vx 0.199 / 0.193 (noise on / off), 0 falls; zero-action 8/8 falls
g1-3.json (4 repeats x 10 s): no falls, tilt < 3 deg; walk_forward_0.2 vx 0.0002
policy.onnx sha256: e39b7180d6cf072ff6fb19c4d452308e22c935e0fdfea5617e7a049237a7548e
notes: a tighter linear kernel alone does not get the policy out of standing at low
  commands; the 0.3 point got slower. Standing at 0.2 still earns 45 % of the term
  plus the full upright/pose/head terms, so the gradient towards stepping stays weak.
```

---

## MAC-18 — Why does the runtime run faster and turn right? (WP4-21)
Status: `done (mac-m5, 2026-11-19)` · Deps: MAC-16 done, WP4-21 merged

MAC-16 (mac-m5): through the runtime the MAC-14 policy goes 0.194 / 0.285 m/s at
0.15 / 0.25 (129 / 114 %) and yaws −0.15…−0.19 rad/s at a zero yaw command; in the
Python env with DR it went 0.150 / 0.228. The runtime run has no DR and no
observation noise. `eval_biped.py` now also reports `vy_mean` and `wz_mean`. Run the
same policy in the Python env with and without DR (no runtime) to see whether the
drift is the policy's or the runtime path's:
```bash
P=runs/mac14   # absolute path if needed
# --dr runs every command twice: DR off, then DR on (0.15 and 0.25 m/s, zero yaw)
uv run python scripts/eval_biped.py $P/policy.onnx --dr --envs 8 --seconds 20 \
    --out runs/mac18/eval.json
```
If the no-DR Python numbers match MAC-16 (fast, yawing right), the runtime path is
faithful and the policy itself drifts; otherwise compare observations (the
`runtime_policy` parity test covers the layout). Commit `eval.json` under
`docs/experiments/mac18/`.

Result (env: mac-m5, 2026-11-19): the policy drifts; the runtime path is faithful
```
revision: 441e731 (main); policy = MAC-14 policy.onnx (sha256 83766219...e8be6f78f);
  8 envs x 20 s, 0 falls in every case; MAC-17 training was running in parallel
no DR: vx / wz at 0.15: 0.178 / -0.186 rad/s; at 0.25: 0.264 / -0.239 rad/s
DR:    vx / wz at 0.15: 0.150 / -0.128 rad/s; at 0.25: 0.228 / -0.159 rad/s
  (vy <= 0.012 m/s everywhere; zero command: vx 0.017 / 0.004, wz -0.021 / 0.001;
  turn 0.8 command: wz 0.622 / 0.558)
MAC-16 runtime (no DR): vx 0.194 / 0.285, wz -0.190 / -0.148
conclusion (policy or runtime path): policy. Python env without DR already yaws right
  (-0.19 / -0.24 rad/s) and goes fast (119 / 106 %), as in the runtime; DR lowers both
  speed and yaw. The runtime adds ~8-9 % vx (0.194 vs 0.178, 0.285 vs 0.264), and at
  0.25 its yaw is lower than the Python env's (-0.148 vs -0.239).
notes: the drift appears only when walking forward (zero command wz ~0), and grows with
  speed without DR, so it is a gait asymmetry, not a heading bias at rest. The policy
  never trained without DR, which matches both effects being largest with DR off.
```

---

## MAC-19 — biped_v1: fine-tune MAC-14 against the yaw drift (WP4-22)
Status: `done (mac-m5, 2026-11-19): mixed, yaw control regressed` · Deps: MAC-18 done, WP4-22 merged

MAC-18 (mac-m5): the MAC-14 policy itself yaws −0.13…−0.24 rad/s while walking at a
zero yaw command (worse with DR off), the runtime path is faithful. biped_v1 is
mirror-symmetric (cloud check of body/joint/geom data), so this is a learned gait
asymmetry that the 0.5 angular kernel barely penalizes (exp(−0.19²/0.5) = 0.93).
`--track-yaw-std2 0.1` makes the same drift cost 30 % of that term. Continue MAC-14
(its curricula, DR delay at full 0–40 ms) for 1000 iterations from a clean worktree of
origin/main; use absolute paths to the MAC-14 checkpoint. Run it after MAC-17 if the
Mac is busy.
```bash
uv run python scripts/train_ppo.py --robot biped_v1 --dr --envs 256 --iters 1000 \
    --workers <best> --seed 0 --track-yaw-std2 0.1 \
    --resume <mac14>/runs/mac14/model_3000.pt --run-dir runs/mac19 --save-every 100 \
    --out runs/mac19/report.json
uv run python scripts/eval_biped.py runs/mac19/policy.onnx --dr --robustness --envs 8 \
    --seconds 20 --out runs/mac19/eval.json
```
Then MAC-16's runtime check with `runs/mac19/policy.onnx` (export to BPML as in MAC-16),
for 0.15 and 0.25. Commit `report.json`, `run.json`, `eval.json` and the runtime JSON
under `docs/experiments/mac19/`.

Result (env: mac-m5, 2026-11-19): mixed, the forward drift shrinks but yaw control regresses
```
revision: 8f94394 (main), git_dirty false; resumed from mac14/model_3000.pt (it 3000 ->
  4000), --track-yaw-std2 0.1, 8 workers; 658 s; reward 0.101 -> 0.106; air_time ~0.029
  throughout (gait kept); runtime check ran with no training on the Mac
Python env vx / wz at 0.15, 0.25 (DR off): 0.153 / -0.046, 0.242 / +0.054
  (MAC-18: 0.178 / -0.186, 0.264 / -0.239)
Python env vx / wz at 0.15, 0.25 (DR on):  0.136 / -0.169, 0.217 / -0.022
  (MAC-18: 0.150 / -0.128, 0.228 / -0.159)
zero command wz (DR off / on): -0.349 / -0.265 (MAC-18: -0.021 / +0.001) <- new spin in place
turn 0.8 wz (DR off / on): 0.139 / 0.152 (MAC-18: 0.622 / 0.558) <- turning mostly lost
latency 20/40/60 ms vx: 0.139 / 0.125 / 0.114 (93 / 83 / 76 %)
runtime vx / wz at 0.15, 0.25: 0.162 / -0.210, 0.246 / -0.004; fell false, latched 0,
  deadline_misses 0, 1000 periods, period_max 24.9 ms; parity 1.3e-6
falls / knee p95: 0 falls in every case; knee p95 0.19-0.26, max 0.52 (clip) in several
policy.onnx sha256: a793cbb9b579bfdb5218ca4a467b86c9630ed721f7dcd6fe087ca60309213fca
notes: the tighter yaw kernel cut the forward drift at 0.25 (and at 0.15 without DR),
  but the policy now spins at a zero command and barely follows a turn command, and
  at 0.15 the runtime / DR-on drift is still -0.17...-0.21. The track_angular_velocity
  term stayed ~0.0025-0.0029 per step all run. Not an improvement overall; MAC-14
  stays the reference policy.
```

---

## MAC-20 — Microduck: train slow forward commands on BAM (WP3-13, gate G3.3)
Status: `done (mac-m5, 2026-11-19): G3.3 not met` · Deps: MAC-17 done (its `model_8000.pt`), WP3-13 merged

MAC-17 (mac-m5): with std² 0.05 the threshold did not move (0.2 → 0.000, 0.3 → 0.078).
Cloud reading of the command sampler: forward-only draws are max(|U(−0.4, 0.4)|, 0.3),
so 75 % of them are exactly 0.3 m/s and none are below 0.3. A pure 0.1–0.25 m/s
forward command only shows up inside the mixed vx/vy/wz draw. `--forward-min-vx 0.1`
spreads the forward draws over 0.1–0.4. Continue MAC-17 for 2000 iterations from a
clean worktree of origin/main, using the absolute checkpoint path. Run it after MAC-19
if the Mac is busy, under `caffeinate -i nohup`.
```bash
uv run python scripts/train_ppo.py --actuator bam --dr --envs 256 --iters 2000 \
    --workers <best> --seed 0 --track-lin-std2 0.05 --forward-min-vx 0.1 \
    --resume <mac17>/runs/mac17-seed0/model_8000.pt --run-dir runs/mac20-seed0 \
    --save-every 100 --out runs/mac20-seed0/report.json
```
Then run the same evals as MAC-17: eval_velocity_env at 0.4, the G1.3 eval (BAM) and
the forward sweep at 0.1 / 0.3 / 0.4 (0.2 from G1.3). Also record the sweep's lateral
drift (`displacement_y_m`): at 0.3, MAC-17 drifted −1.0 m in 10 s. Commit the JSONs
under `docs/experiments/mac20/`.

**Acceptance (G3.3):** as MAC-12 (G1.3 with BAM). If only the threshold moves, record
the new sweep.

Result (env: mac-m5, 2026-11-19): G3.3 NOT met, the threshold moved UP (only 0.4 walks)
```
revision: 2dc888c (main), git_dirty false; resumed from mac17-seed0/model_8000.pt
  (it 8000 -> 10000), --track-lin-std2 0.05 --forward-min-vx 0.1, 8 workers, load avg 10-24
wall time: 2287 s (38 min)
reward first -> last quarter, air_time trend: 0.0992 -> 0.1031 (rose); air_time
  0.0145 (first 100 it) -> 0.012-0.013 from it 200 on; track_linear_velocity 0.022-0.023
sweep vx at 0.1 / 0.2 / 0.3 / 0.4 (BAM), displacement_y at 0.3 / 0.4:
  vx 0.0002 / 0.0002 / 0.0002 / 0.170; dy -0.001 / +0.739 m in 10 s
  (MAC-17: 0.0001 / 0.0002 / 0.078 / 0.131; dy -0.996 / -1.123); backward, lateral,
  turn ~0
eval.json (0.4 cmd): vx 0.217 / 0.200 (noise on / off), 0 falls; zero-action 8/8 falls
g1-3.json (4 repeats x 10 s): no falls, tilt < 2.2 deg; walk_forward_0.2 vx 0.0002
policy.onnx sha256: b8497e7a74d90e1b938fc82d779a40f78282b9b3715d8ec88671c7dbbe15eac3
notes: spreading the forward draws over 0.1-0.4 made the policy stand at more of them:
  it now stands at 0.3 as well and only steps at 0.4 (faster there, 0.170). The lateral
  drift at 0.4 flipped sign (-1.1 -> +0.7 m). Standing remains the better-paying answer
  for every command the policy cannot track well, so more low-command samples mostly
  teach it to stand at more commands.
```

---

## MAC-21 — biped_v1: a milder yaw kernel (WP4-22)
Status: `done (mac-m5, 2026-11-19): not accepted (DR-off yaw row)` · Deps: MAC-19 done

MAC-19 (mac-m5, std² 0.1): the walking drift at 0.25 is gone, but the policy now spins
in place at a zero command (wz −0.35 / −0.27) and turns only 0.14 at a 0.8 command
(MAC-14: 0.62). A 0.1 kernel gives almost no gradient at large yaw errors
(exp(−0.66²/0.1) ≈ 0.01), so turning stopped paying. MAC-21 repeats MAC-19 from
MAC-14 with std² 0.25. The same drift still costs 13 %, and a 0.4 rad/s turn error
still pays 53 %. MAC-14 stays the reference policy until a run beats it on every
row below.
```bash
uv run python scripts/train_ppo.py --robot biped_v1 --dr --envs 256 --iters 1000 \
    --workers <best> --seed 0 --track-yaw-std2 0.25 \
    --resume <mac14>/runs/mac14/model_3000.pt --run-dir runs/mac21 --save-every 100 \
    --out runs/mac21/report.json
uv run python scripts/eval_biped.py runs/mac21/policy.onnx --dr --robustness --envs 8 \
    --seconds 20 --out runs/mac21/eval.json
```
Then run the MAC-16-style runtime check at 0.15 and 0.25. Run it after MAC-20 if the Mac is busy.

**Acceptance:** at 0.15 and 0.25, |wz| ≤ 0.08 with DR off and on; |wz| ≤ 0.05 at the
zero command; turn 0.8 gives wz ≥ 0.5; 40 ms latency keeps ≥ 80 % of vx; 0 falls.

Result (env: mac-m5, 2026-11-19): 4 of 5 acceptance rows met; |wz| with DR off misses
```
revision: 5633e4d (main), git_dirty false; resumed from mac14/model_3000.pt (it 3000 ->
  4000), --track-yaw-std2 0.25, 8 workers; 444 s; reward 0.100 -> 0.111; air_time ~0.029;
  track_angular_velocity 0.0028 -> 0.0057 per step; runtime runs with no training load
vx / wz at 0.15, 0.25 (DR off / on): off 0.170 / +0.103, 0.260 / +0.097;
  on 0.142 / +0.044, 0.222 / +0.040   (MAC-14: off -0.186 / -0.239, on -0.128 / -0.159)
zero command wz, turn 0.8 wz (DR off / on): zero +0.047 / +0.035; turn 0.625 / 0.536
  (MAC-14: zero -0.021 / +0.001; turn 0.622 / 0.558)
latency 20/40/60 ms vx; runtime vx / wz at 0.15, 0.25: 0.143 / 0.122 / 0.110
  (95 / 81 / 73 % of 0.15); runtime 0.185 / +0.165, 0.276 / +0.080 (fell false,
  latched 0, deadline_misses 0, 1000 periods, period_max 28.1 ms; parity 1.4e-6)
verdict vs MAC-14 (acceptance rows):
  |wz| <= 0.08 at 0.15 / 0.25, DR off: 0.103 / 0.097  FAIL (MAC-14 0.186 / 0.239)
  |wz| <= 0.08 at 0.15 / 0.25, DR on:  0.044 / 0.040  pass (MAC-14 0.128 / 0.159)
  |wz| <= 0.05 at zero command:        0.047 / 0.035  pass (MAC-14 0.021 / 0.001)
  turn 0.8 -> wz >= 0.5:               0.625 / 0.536  pass (MAC-14 0.622 / 0.558)
  40 ms latency >= 80 % of vx (cmd):   81 %           pass (MAC-14 86 %)
  0 falls:                             0 everywhere   pass
  -> not accepted. The drift is about halved and now goes LEFT (sign flip), turning
  is kept, but the DR-off row fails, and the zero-command, turn (DR on) and latency rows are
  slightly worse than MAC-14. MAC-14 stays the reference; MAC-21 is the best yaw
  candidate so far. Runtime 0.15 still yaws +0.165.
```

---

## MAC-22 — Microduck: posture gate on BAM (WP3-14, gate G3.3)
Status: `done (mac-m5, 2026-11-19): G3.3 not met, threshold lowered` · Deps: MAC-17 done (its `model_8000.pt`), MAC-20 done, WP3-14 merged

MAC-20 (mac-m5): spreading the forward commands down to 0.1 m/s raised the threshold.
The policy now also stands at 0.3 and walks only at 0.4. Slow commands taught it to
stand, because standing keeps the full upright, pose and head pose rewards, and those
outweigh the tracking gain from stepping. `--posture-gate` scales those three terms by
the tracking kernels whenever the command is nonzero, so standing at 0.2 m/s
(std² 0.05) keeps only 45 % of them. Continue from MAC-17 (not MAC-20) for 2000
iterations with the gate, std² 0.05 and the forward floor at 0.1, from a clean worktree
of origin/main, using the absolute checkpoint path. Run it after MAC-21, under
`caffeinate -i nohup`.
```bash
uv run python scripts/train_ppo.py --actuator bam --dr --envs 256 --iters 2000 \
    --workers <best> --seed 0 --track-lin-std2 0.05 --forward-min-vx 0.1 --posture-gate \
    --resume <mac17>/runs/mac17-seed0/model_8000.pt --run-dir runs/mac22-seed0 \
    --save-every 100 --out runs/mac22-seed0/report.json
```
Then run the same evals as MAC-20: eval_velocity_env at 0.4, G1.3 (BAM), and the
forward sweep at 0.1 / 0.3 / 0.4 with `displacement_y_m`. Watch `upright` and the falls
in the training history. If falls rise above MAC-17's, stop at that checkpoint and
record it. Commit the JSONs under `docs/experiments/mac22/`.

**Acceptance (G3.3):** as MAC-12 (G1.3 with BAM). If only the threshold moves, record
the new sweep.

Result (env: mac-m5, 2026-11-19): G3.3 NOT met, but the gate breaks the standing optimum
```
revision: 8c701c2 (main), git_dirty false; resumed from mac17-seed0/model_8000.pt
  (it 8000 -> 10000), --track-lin-std2 0.05 --forward-min-vx 0.1 --posture-gate,
  8 workers, load avg 7-17
wall time: 1890 s (32 min)
reward first -> last quarter, air_time and upright trend, training falls:
  reward 0.0409 -> 0.0376 (did not rise; the gated terms shrink the reward scale);
  air_time 0.0149 -> 0.0159; upright 0.0133 -> 0.0137 (gated); track_linear_velocity
  0.0222 -> 0.0232; falls/it (400-it means) 1.97 -> 1.51 -> 1.41 -> 1.18 -> 1.18
  (MAC-17: 1.48 first 100 it, 1.30 last 500), so no early stop
sweep vx at 0.1 / 0.2 / 0.3 / 0.4 (BAM), displacement_y at 0.3 / 0.4:
  vx 0.0005 / 0.061 / 0.219 / 0.120; dy +0.549 / +2.176 m in 10 s
  (MAC-20: 0.0002 / 0.0002 / 0.0002 / 0.170); backward_0.3 vx -0.150 (dy +0.26),
  lateral_0.3 and turn_1.0 ~0
eval.json (0.4 cmd): vx 0.311 / 0.301 (noise on / off), 0 falls; zero-action 8/8 falls
g1-3.json (4 repeats x 10 s): stand_still no fall, vx 0.0003, tilt 1.0 deg;
  walk_forward_0.2 vx 0.061-0.070 (33 % of 0.2, needs 0.14-0.26), dy -1.16 m,
  yaw -0.19 rad/s; walk_backward_0.15 ~0; turn_left_0.5 ~0; 0 falls
policy.onnx sha256: 4b2a2cc1bcf1b2511c1efa78c35f4606a43e90e352e64c4649bd7a8ac317259b
notes: first Microduck BAM policy that steps at 0.2 and 0.3 and walks backward. The
  0.4 point is slower in the sweep (0.120) than in eval_velocity_env (0.30), with a
  2.2 m sideways drift, so the gait curves at speed. 0.1, slow backward, lateral and
  turn still stand.
```

---

## MAC-23 — biped_v1: mirror-symmetry training against the yaw drift (WP4-23)
Status: `awaiting-mac` · Deps: MAC-21 done; run after MAC-22 if the Mac is busy

MAC-19 and MAC-21 show that the yaw kernel alone only trades one error for another:
std² 0.1 kills turning, std² 0.25 halves the drift but flips it to the left (+0.10 rad/s
with DR off). biped_v1 is mirror-symmetric, so the policy should be too. MAC-23 repeats
MAC-21 (std² 0.25, resumed from MAC-14) and adds `--symmetry aug`: rsl-rl trains on a
left/right mirrored copy of every sample (`bipedlab.rl.biped_symmetry`), so a drift in
one direction is paid for by the mirrored drift in the other. The only change versus
MAC-21 is the symmetry flag. The mirror maps were checked in the cloud against a
physically mirrored MuJoCo state (`tests/unit/test_biped_symmetry.py`).
```bash
uv run python scripts/train_ppo.py --robot biped_v1 --dr --envs 256 --iters 1000 \
    --workers <best> --seed 0 --track-yaw-std2 0.25 --symmetry aug \
    --resume <mac14>/runs/mac14/model_3000.pt --run-dir runs/mac23 --save-every 100 \
    --out runs/mac23/report.json
uv run python scripts/eval_biped.py runs/mac23/policy.onnx --dr --robustness --envs 8 \
    --seconds 20 --out runs/mac23/eval.json
```
Then run the MAC-16-style runtime check at 0.15 and 0.25. Augmentation doubles the PPO
batch, so expect roughly 1.5–2× the MAC-21 wall time (444 s).

**Acceptance (same rows as MAC-21):** at 0.15 and 0.25, |wz| ≤ 0.08 with DR off and on;
|wz| ≤ 0.05 at the zero command; turn 0.8 gives wz ≥ 0.5; 40 ms latency keeps ≥ 80 %
of vx; 0 falls. If it passes, it replaces MAC-14 as the reference policy.

Result:
```
revision:
vx / wz at 0.15, 0.25 (DR off / on):
zero command wz, turn 0.8 wz (DR off / on):
latency 20/40/60 ms vx; runtime vx / wz at 0.15, 0.25:
verdict vs MAC-14 and MAC-21:
```

---

## Later Mac tasks (not ready yet)
- MAC-05 live telemetry overlay (after WP2-01)
