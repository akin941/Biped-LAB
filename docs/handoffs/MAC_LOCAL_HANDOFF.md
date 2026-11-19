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
  vx 0.0001 / 0.120 / 0.149, backward_0.3 / lateral_0.3 ~0, turn_1.0 wz 0.941
  (corrected in MAC-24: first recorded as ~0 from vx); BAM walking
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
  lateral_0.3 ~0 (dy 0.008), turn_1.0 wz 0.757 (corrected in MAC-24: first
  recorded as ~0 from vx)
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
Status: `done (mac-m5, 2026-11-19): not accepted, regression` · Deps: MAC-21 done; run after MAC-22 if the Mac is busy

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

Result (env: mac-m5, 2026-11-19): NOT accepted, a regression (training degraded)
```
revision: 4329183 (main), git_dirty false; resumed from mac14/model_3000.pt (it 3000 ->
  4000), --track-yaw-std2 0.25 --symmetry aug, 8 workers; 1221 s (2.75x MAC-21)
training trend (100-it means): reward 0.098 -> 0.102 (it 100) -> 0.068 (it 900),
  falling for 800 it; falls/it 1.14 -> 0.37 (it 200) -> 2.18 (it 800); knee torque
  mean 0.20 -> 0.31 N.m; track_angular_velocity 0.0032 -> 0.0008; air_time ~0.027
vx / wz at 0.15, 0.25 (DR off / on): off 0.178 / +0.164, 0.251 / +0.188;
  on 0.135 / -0.026, 0.203 / -0.037
zero command wz, turn 0.8 wz (DR off / on): zero +0.005 / +0.007; turn 0.195 / 0.196
latency 20/40/60 ms vx; runtime vx / wz at 0.15, 0.25: 0.143 / 0.118 / 0.102
  (95 / 79 / 68 %); runtime 0.196 / +0.446, 0.265 / +0.622 (walks in circles: distance
  0.58 / 0.65 m in 20 s; fell false, latched 0, deadline_misses 0, 1000 periods)
knee torque: p95 at the 0.52 N.m clip in nearly every policy case (MAC-14 ~0.19)
verdict vs MAC-14 and MAC-21:
  |wz| <= 0.08, DR off: 0.164 / 0.188  FAIL (MAC-14 0.186 / 0.239, MAC-21 0.103 / 0.097)
  |wz| <= 0.08, DR on:  0.026 / 0.037  pass (MAC-21 0.044 / 0.040)
  |wz| <= 0.05 zero:    0.005 / 0.007  pass (MAC-21 0.047 / 0.035)
  turn 0.8 >= 0.5:      0.195 / 0.196  FAIL (MAC-14 0.622 / 0.558, MAC-21 0.625 / 0.536)
  40 ms >= 80 %:        79 %           FAIL (MAC-14 86 %, MAC-21 81 %)
  0 falls:              0              pass
  -> worse than MAC-21 and MAC-14. MAC-14 stays the reference. The run was still
  degrading at the end (reward down, falls and knee torque up), which points at the
  augmentation itself (mirror maps for actions or the command/yaw sign, or the doubled
  batch with unchanged PPO settings) rather than too few iterations. Worth a cloud
  check before another symmetry run.
```

---

## MAC-24 — Microduck: continue the posture-gate run on BAM (WP3-14, gate G3.3)
Status: `done (mac-m5, 2026-11-19): G3.3 not met` · Deps: MAC-22 done (its `model_10000.pt`); run after MAC-23

MAC-22 (mac-m5) broke the standing optimum: with the posture gate the policy steps at
0.2 (0.061–0.070 m/s) and 0.3 (0.219) and walks backward (−0.150). It was still
improving when it stopped: tracking and air time were rising, and training falls were
falling (1.97 → 1.18 per iteration). MAC-24 continues the same run with the same
flags for 3000 more iterations, so the only change is training time. Use a clean
worktree of origin/main, the absolute checkpoint path, and `caffeinate -i nohup`.
```bash
uv run python scripts/train_ppo.py --actuator bam --dr --envs 256 --iters 3000 \
    --workers <best> --seed 0 --track-lin-std2 0.05 --forward-min-vx 0.1 --posture-gate \
    --resume <mac22>/runs/mac22-seed0/model_10000.pt --run-dir runs/mac24-seed0 \
    --save-every 100 --out runs/mac24-seed0/report.json
```
Then run the MAC-22 evals: eval_velocity_env at 0.4, G1.3 (BAM), and the forward sweep
at 0.1 / 0.2 / 0.3 / 0.4 plus backward 0.3, with `displacement_y_m` and the yaw rate.
If training falls rise above MAC-22's last 400 iterations (1.18), stop at that
checkpoint and evaluate it. Commit the JSONs under `docs/experiments/mac24/`.

**Acceptance (G3.3):** as MAC-12 (G1.3 with BAM: 0.2 m/s within ±30 %). If it misses,
record whether vx at 0.2 and the drift moved, since that decides between more training
and adding drift pressure (yaw / lateral tracking).

Result (env: mac-m5, 2026-11-19): G3.3 NOT met; better above 0.3, back to standing at 0.2
```
revision: 61caa14 (main), git_dirty false; resumed from mac22-seed0/model_10000.pt
  (it 10000 -> 13000), same flags as MAC-22, 8 workers; a VM used ~4-6 cores
  (load avg 10-33)
wall time: 4292 s (72 min)
reward, air_time, tracking, training falls trend (500-it means): reward 0.0375 ->
  0.0413; air_time 0.0155 -> 0.0144-0.0151; track_linear_velocity 0.0236 -> 0.0247;
  falls/it 1.12 / 1.25 / 0.99 / 1.05 / 0.94 / 0.98 (one bucket above MAC-22's 1.18,
  then below, so no stop)
sweep vx at 0.1 / 0.2 / 0.3 / 0.4, backward 0.3; dy and wz at 0.2 / 0.3 / 0.4:
  vx 0.0002 / 0.0004 / 0.193 / 0.260, backward_0.3 -0.211 (MAC-22: 0.0005 / 0.061 /
  0.219 / 0.120, -0.150); dy -0.002 / +0.572 / +1.458 m, wz -0.001 / +0.085 / +0.070
  (MAC-22: dy -1.16 / +0.55 / +2.18, wz -0.19 / +0.02 / +0.19);
  lateral_0.3 dy +1.06 m (first lateral motion; MAC-22 0.008), turn_1.0 wz 0.965
  (MAC-22 0.757); eval_velocity_env 0.4: vx 0.316 / 0.313, 0 falls
G1.3 (BAM) verdict: not met. walk_forward_0.2 vx 0.0004 (MAC-22: 0.061),
  walk_backward_0.15 and turn_left_0.5 ~0, stand_still fine, 0 falls.
  vx at 0.2 moved DOWN (back to standing) while the drift at 0.4 shrank
  (dy 2.18 -> 1.46 m) and lateral and turning got better, so more training at the same
  sampling does not reach 0.2. The policy still stands at every small command
  (fwd 0.1-0.2, back 0.15, turn 0.5).
policy.onnx sha256: 5c5e9f3566f78a96d73d4f03f611d0ef9fee06af2dcc8853ae3cd6f8c2e64bb7
```

---

## MAC-25 — biped_v1: mirror loss instead of augmentation (WP4-24)
Status: `done (mac-m5, 2026-11-19): accepted, new reference policy` · Deps: MAC-23 done; run after MAC-24

MAC-23 (mac-m5, `--symmetry aug`) degraded from about iteration 100. Reward fell from
0.102 to 0.068, falls rose to about 2.2 per iteration, and turning was lost (0.8 → 0.20).
The cloud check found nothing wrong with the mirror maps: body positions of a mirrored
state match link for link, and actuator order equals joint order. The cause is how
rsl-rl augments. Each mirrored sample gets the *original* sample's old log-probability.
MAC-14 is asymmetric (it drifts), so on the mirrored half the PPO ratio
π(mirror a | mirror o) / π_old(a | o) is far from 1. Where the advantage is negative,
the clipped surrogate keeps the unclipped term and pushes hard. Augmentation suits
training from scratch, not a warm start. `--symmetry loss` keeps PPO on the real samples
only and adds `coeff · ‖π(mirror o) − mirror(π(o))‖²`, which pulls the policy toward
symmetry gradually. The history now also records PPO's losses (`loss_symmetry`,
`loss_surrogate`, …). Repeat MAC-21 (std² 0.25, from MAC-14) with the mirror loss at
0.5. This is MAC-21 plus one change.
```bash
uv run python scripts/train_ppo.py --robot biped_v1 --dr --envs 256 --iters 1000 \
    --workers <best> --seed 0 --track-yaw-std2 0.25 --symmetry loss --mirror-loss-coeff 0.5 \
    --resume <mac14>/runs/mac14/model_3000.pt --run-dir runs/mac25 --save-every 100 \
    --out runs/mac25/report.json
uv run python scripts/eval_biped.py runs/mac25/policy.onnx --dr --robustness --envs 8 \
    --seconds 20 --out runs/mac25/eval.json
```
Then run the MAC-16-style runtime check at 0.15 and 0.25. Watch `loss_symmetry`, which
should fall, and the falls, which should stay near MAC-21's. If the falls pass 1.0 per
iteration for 100 iterations, stop and evaluate the last good checkpoint.

**Acceptance (same rows as MAC-21):** at 0.15 and 0.25, |wz| ≤ 0.08 with DR off and on;
|wz| ≤ 0.05 at the zero command; turn 0.8 gives wz ≥ 0.5; 40 ms latency keeps ≥ 80 %
of vx; 0 falls. If it passes, it replaces MAC-14 as the reference policy.

Result (env: mac-m5, 2026-11-19): ACCEPTED, all six rows pass; replaces MAC-14 as reference
```
revision: f5f71eb (main), git_dirty false; resumed from mac14/model_3000.pt (it 3000 ->
  4000), --track-yaw-std2 0.25 --symmetry loss --mirror-loss-coeff 0.5, 8 workers;
  983 s; reward 0.101 -> 0.111; knee torque mean 0.24 -> 0.17 N.m
loss_symmetry first -> last; falls trend: 0.030 (first 100 it; per-it peak 0.057 at
  it 2) -> 0.0037 (last 100 it); falls/it 2.38 (first 100, resume transient as in
  MAC-21) -> 0.25 -> 0.08-0.14 from it 300 (watchdog from it 200: never above 0.15)
vx / wz at 0.15, 0.25 (DR off / on): off 0.166 / -0.047, 0.252 / +0.006;
  on 0.142 / -0.052, 0.219 / -0.028
zero command wz, turn 0.8 wz (DR off / on): zero -0.034 / -0.020; turn 0.579 / 0.523
latency 20/40/60 ms vx; runtime vx / wz at 0.15, 0.25: 0.141 / 0.125 / 0.110
  (94 / 83 / 73 % of 0.15); runtime 0.177 / -0.133, 0.266 / -0.041 (fell false,
  latched 0; deadline_misses 4 / 0, period_max 53.8 / 32.6 ms, a VM was using several
  cores; distance 2.36 / 4.66 m; parity 1.3e-6)
verdict vs MAC-14 and MAC-21:
  |wz| <= 0.08, DR off: 0.047 / 0.006  pass (MAC-14 0.186 / 0.239, MAC-21 0.103 / 0.097)
  |wz| <= 0.08, DR on:  0.052 / 0.028  pass (MAC-14 0.128 / 0.159, MAC-21 0.044 / 0.040)
  |wz| <= 0.05 zero:    0.034 / 0.020  pass (MAC-14 0.021 / 0.001, MAC-21 0.047 / 0.035)
  turn 0.8 >= 0.5:      0.579 / 0.523  pass (MAC-14 0.622 / 0.558, MAC-21 0.625 / 0.536)
  40 ms >= 80 %:        83 %           pass (MAC-14 86 %, MAC-21 81 %)
  0 falls:              0              pass
  -> accepted: MAC-25 is the new biped_v1 reference policy. Through the runtime the
  0.15 case still yaws -0.133 (MAC-14 -0.190), which is not an acceptance row but is the
  remaining gap; knee p95 0.17-0.24 (max at the clip in some DR-on cases).
policy.onnx sha256: 3789a53b243f32d81492c42d0e8c793c31c10a35ddf26a3f9d15a4872745a52b
quiet re-check (2026-11-19, 20:20Z, recorded in MAC-33): runtime 0.15 / 0.25 again, VM at
  ~1.2-3.8 cores (mean ~2.3) during the run, load avg ~3: deadline_misses 0 / 0,
  period_max 30.0 / 29.9 ms, compute_max 155 / 170 us, 1000 periods, fell false, latched 0,
  vx / wz unchanged (0.177 / -0.133, 0.266 / -0.041). The earlier 4 misses (and 2 in a
  re-run under load) were load. JSONs: docs/experiments/mac25/quiet-runtime-v*.json
```

---

## MAC-26 — Microduck: speed-relative tracking kernel on BAM (WP3-15, gate G3.3)
Status: `done (mac-m5, 2026-11-19): G3.3 not met` · Deps: MAC-24 done (its `model_13000.pt`); run after MAC-25

MAC-24 (mac-m5) improved everything above 0.3 m/s (0.4 → 0.260, backward −0.211,
first lateral motion, turn 1.0 → 0.965), but at 0.2 it went back to standing (vx
0.0004). With one fixed kernel width (std² 0.05), standing at 0.2 keeps
exp(−0.04/0.05) = 45 % of the tracking reward, while standing at 0.4 keeps only 4 %.
So more training mostly pays the faster commands. `--track-lin-ref-speed 0.4` scales the
width by (|v_cmd| / 0.4)², down to 1/16 at the floor of a quarter of ref. Standing then
costs about the same share at every speed (≈ 4 % kept at 0.2 too), and the posture gate
shrinks with it. Continue MAC-24 with the same flags plus this one, from a clean
worktree of origin/main, using the absolute checkpoint path, under `caffeinate -i nohup`.
```bash
uv run python scripts/train_ppo.py --actuator bam --dr --envs 256 --iters 2000 \
    --workers <best> --seed 0 --track-lin-std2 0.05 --forward-min-vx 0.1 --posture-gate \
    --track-lin-ref-speed 0.4 \
    --resume <mac24>/runs/mac24-seed0/model_13000.pt --run-dir runs/mac26-seed0 \
    --save-every 100 --out runs/mac26-seed0/report.json
```
Then run the MAC-24 evals: eval_velocity_env at 0.4, G1.3 (BAM), and the sweep at
0.1 / 0.2 / 0.3 / 0.4, backward 0.3, lateral 0.3 and turn 1.0, with `displacement_y_m`
and the yaw rate. Expect `track_linear_velocity` to drop at first, because the kernel
is tighter. If training falls rise above 1.5 per iteration for 200 iterations, stop and
evaluate the last good checkpoint. Commit the JSONs under `docs/experiments/mac26/`.

**Acceptance (G3.3):** as MAC-12 (G1.3 with BAM: 0.2 m/s within ±30 %). Also record
whether 0.3 / 0.4 stayed at the MAC-24 level.

Result (env: mac-m5, 2026-11-19): G3.3 NOT met; 0.2 and slow backward step, fast commands and turning regress
```
revision: 2ac6883 (main), git_dirty false; resumed from mac24-seed0/model_13000.pt
  (it 13000 -> 15000), MAC-24 flags + --track-lin-ref-speed 0.4, 8 workers; a VM used
  ~5-7 cores (load avg 11-20)
wall time: 3124 s (52 min)
reward, air_time, tracking, training falls trend (250-it means): reward 0.0295 ->
  0.0235 (kernel tighter, scale smaller); air_time 0.0169 -> 0.0181; track_linear_velocity
  0.0174 -> 0.0170 (flat on the tighter kernel); upright 0.012; falls/it 1.18 / 1.39 /
  1.20 / 1.14 / 1.45 / 1.34 / 1.32 / 1.07, 200-it mean never above 1.5 (no stop)
sweep vx at 0.1 / 0.2 / 0.3 / 0.4, backward 0.3; dy and wz at 0.2 / 0.3 / 0.4:
  vx 0.0003 / 0.097 / 0.110 / 0.116, backward_0.3 -0.210 (MAC-24: 0.0002 / 0.0004 /
  0.193 / 0.260, -0.211); dy -1.10 / -1.80 / -2.24 m, wz -0.16 / -0.17 / -0.19
  (MAC-24: dy -0.00 / +0.57 / +1.46, wz -0.00 / +0.09 / +0.07); lateral_0.3 dy +0.64
  (MAC-24 +1.06); turn_1.0 wz 0.002 (MAC-24 0.965, lost); eval_velocity_env 0.4:
  vx 0.319 / 0.312, 0 falls
G1.3 (BAM) verdict: not met. walk_forward_0.2 vx 0.091-0.099 (48 %, needs 0.14-0.26),
  dy -1.10 m, wz -0.16; walk_backward_0.15 vx -0.073 (48 %, new: MAC-24 ~0);
  turn_left_0.5 ~0; stand_still fine; 0 falls. 0.3 / 0.4 did NOT stay at the MAC-24
  level (0.193 / 0.260 -> 0.110 / 0.116), and the gait now curves right at every
  forward speed.
policy.onnx sha256: fd220aaa52439603bdf0bead51187b0dd5b2b6a42813b39a515ecac39a55d35a
```

---

## MAC-27 — Microduck: mirror loss against the curving gait (WP3-16, gate G3.3)
Status: `done (mac-m5, 2026-11-19): G3.3 not met (0.2 -> 63 %)` · Deps: MAC-26 done (its `model_15000.pt`)

MAC-26 (mac-m5): the speed-relative kernel made the policy step at slow commands for
the first time. It reached 0.091–0.099 at 0.2 (48 %) and walked backward at 0.15.
Every forward gait curves right, though: wz −0.16…−0.19 rad/s, dy −1.1…−2.2 m in 10 s.
At 0.3 / 0.4 it fell back to ~0.11 and lost the 1.0 turn. On biped_v1, the same kind
of drift went away with the mirror loss (MAC-25, accepted). `bipedlab.rl.microduck_symmetry`
gives Microduck its own mirror. Its right leg's joint axes are the left's mirrored, so
every leg joint flips sign. The head pitch joints keep their sign, and head yaw and
roll flip. The cloud checked the mirror against a physically mirrored MuJoCo state,
body by body, and the tests catch every wrong sign. Continue MAC-26 with the same
flags plus `--symmetry loss --mirror-loss-coeff 0.5`. Use a clean worktree of
origin/main, the absolute checkpoint path, and `caffeinate -i nohup`.
```bash
uv run python scripts/train_ppo.py --actuator bam --dr --envs 256 --iters 2000 \
    --workers <best> --seed 0 --track-lin-std2 0.05 --forward-min-vx 0.1 --posture-gate \
    --track-lin-ref-speed 0.4 --symmetry loss --mirror-loss-coeff 0.5 \
    --resume <mac26>/runs/mac26-seed0/model_15000.pt --run-dir runs/mac27-seed0 \
    --save-every 100 --out runs/mac27-seed0/report.json
```
Then run the MAC-26 evals: eval_velocity_env at 0.4, G1.3 (BAM), and the sweep at
0.1 / 0.2 / 0.3 / 0.4, backward 0.3, lateral 0.3 and turn 1.0, with `displacement_y_m`
and the yaw rate. Watch `loss_symmetry`, which should fall. If training falls stay
above 1.5 per iteration for 200 iterations after the first 100, stop and evaluate the
last good checkpoint. Commit the JSONs under `docs/experiments/mac27/`.

**Acceptance (G3.3):** as MAC-12 (G1.3 with BAM: 0.2 m/s within ±30 %). Also record
|wz| at 0.2 / 0.3 / 0.4 against MAC-26's 0.16–0.19.

Result (env: mac-m5, 2026-11-19): G3.3 NOT met but close (0.2 -> 62-64 %); the curve is gone
```
revision: 6c04cbe (main), git_dirty false; resumed from mac26-seed0/model_15000.pt
  (it 15000 -> 17000), MAC-26 flags + --symmetry loss --mirror-loss-coeff 0.5,
  8 workers; a VM used ~6-7 cores
wall time; loss_symmetry first -> last; training falls trend: 2746 s (46 min);
  loss_symmetry 0.0203 -> 0.0117 (250-it means); falls/it 1.05-1.45 (250-it means),
  200-it mean never above 1.5 (no stop); air_time ~0.018; tracking ~0.017; reward
  0.026 -> 0.024
sweep vx at 0.1 / 0.2 / 0.3 / 0.4, backward 0.3; dy and wz at 0.2 / 0.3 / 0.4:
  vx 0.0005 / 0.127 / 0.200 / 0.274, backward_0.3 -0.182 (MAC-26: 0.0003 / 0.097 /
  0.110 / 0.116, -0.210; MAC-24: 0.193 / 0.260 at 0.3 / 0.4);
  dy +0.17 / -0.01 / -0.18 m, wz -0.002 / -0.020 / -0.018
  (MAC-26: |wz| 0.16 / 0.17 / 0.19, dy -1.10 / -1.80 / -2.24);
  backward_0.3 dy +0.79, wz -0.14; lateral_0.3 dy +0.49; turn_1.0 wz 0.003 (still lost);
  eval_velocity_env 0.4: vx 0.292 / 0.282, 0 falls
G1.3 (BAM) verdict: not met, but close. walk_forward_0.2 vx 0.124-0.127 (62-64 %, needs
  >= 0.14), |wz| 0.002; walk_backward_0.15 vx -0.123 (82 %); turn_left_0.5 ~0;
  stand_still fine; 0 falls. The mirror loss removed the forward curve and brought
  0.3 / 0.4 back above MAC-24; turning (0.5 and 1.0) is the part still missing.
policy.onnx sha256: f62cbb0b26a603ae3e5f2f2a512d23761a628b75cdc1d8b55f5aaa88c6dd937f
```

---

## MAC-28 — Microduck: speed-relative yaw kernel to bring turning back (WP3-17, gate G3.3)
Status: `done (mac-m5, 2026-11-19): stopped at it 17300 (falls rule); model_17300 meets G3.3, turning still lost` · Deps: MAC-27 done (its `model_17000.pt`)

MAC-27 (mac-m5): the mirror loss removed the curve. 0.2 reached 0.124–0.127 (63 %, and 70 %
is needed), and 0.3 / 0.4 gave 0.200 / 0.274. Turning is still lost: 0.5 and 1.0 give
≈ 0. The yaw kernel has a fixed std² of 0.5, so standing still keeps 61 % of the yaw
reward at a 0.5 rad/s turn and 14 % at 1.0. `--track-ang-ref-speed` (WP3-17) scales the
yaw width the same way WP3-15 scales the linear one, std² × clip(|wz_cmd| / ref, 0.25, 1)².
With ref 1.0, standing costs the same share at every turn rate. The cloud smoke test
checks this. Continue MAC-27 with the same flags plus `--track-ang-ref-speed 1.0`. Use
a clean worktree of origin/main, the absolute checkpoint path, and `caffeinate -i nohup`.
```bash
uv run python scripts/train_ppo.py --actuator bam --dr --envs 256 --iters 2000 \
    --workers <best> --seed 0 --track-lin-std2 0.05 --forward-min-vx 0.1 --posture-gate \
    --track-lin-ref-speed 0.4 --symmetry loss --mirror-loss-coeff 0.5 \
    --track-ang-ref-speed 1.0 \
    --resume <mac27>/runs/mac27-seed0/model_17000.pt --run-dir runs/mac28-seed0 \
    --save-every 100 --out runs/mac28-seed0/report.json
```
Then run the MAC-27 evals: eval_velocity_env at 0.4, G1.3 (BAM), and the sweep at
0.1 / 0.2 / 0.3 / 0.4, backward 0.3, lateral 0.3 and turn 1.0, with dy and the yaw rate.
If training falls stay above 1.5 per iteration for 200 iterations after the first 100,
stop and evaluate the last good checkpoint. Commit the JSONs under `docs/experiments/mac28/`.

**Acceptance (G3.3):** as MAC-12 (G1.3 with BAM: 0.2 m/s within ±30 %). Also record the
turn_left_0.5 and turn_1.0 yaw rates, and check that 0.3 / 0.4 and |wz| at 0.2–0.4 stay
at the MAC-27 level.

Result (env: mac-m5, 2026-11-19): stopped at it 17300 by the falls rule; that checkpoint
MEETS G3.3 (G1.3 with BAM), turning is still lost
```
revision: d357331 (main), git_dirty false; resumed from mac27-seed0/model_17000.pt,
  MAC-27 flags + --track-ang-ref-speed 1.0, 8 workers; a VM used ~6-7 cores
wall time; loss_symmetry; training falls trend: started 17:33Z; falls/it (50-it means)
  1.72 / 1.88 / 2.30 / 2.06 / 1.86 / 1.50 -> mean 1.93 over it 100-300 (> 1.5 for 200 it
  after the first 100), so training was stopped by hand at it 17300 (~9 min).
  track_angular_velocity ~0.007 (flat), tracking ~0.017, air_time 0.017 -> 0.019.
  Checkpoints model_17100 and model_17300 were exported (train_ppo --resume <ckpt>
  --iters 0 with the same flags) and both evaluated.
sweep vx at 0.1 / 0.2 / 0.3 / 0.4, backward 0.3; dy and wz at 0.2 / 0.3 / 0.4; turn 1.0 wz:
  model_17300: vx 0.0000 / 0.160 / 0.255 / 0.342, backward_0.3 -0.235;
  dy -0.15 / -0.35 / +0.05 m, wz -0.011 / -0.031 / +0.010; lateral_0.3 dy +0.65;
  turn_1.0 wz 0.001; eval_velocity_env 0.4: vx 0.355 / 0.351, 0 falls
  model_17100: vx 0.0005 / 0.068 / 0.145 / 0.117, back -0.215; wz -0.25 / -0.23 / -0.27
  (the curve came back in the first 100 it, then went away again); turn_1.0 wz 0.002
  (MAC-27: 0.0005 / 0.127 / 0.200 / 0.274, |wz| 0.002 / 0.020 / 0.018, turn 0.003)
G1.3 (BAM) verdict (incl. turn_left_0.5 wz): model_17300 MEETS G3.3:
  walk_forward_0.2 vx 0.160-0.161 over 4 repeats x 10 s (80 %, inside +-30 %), wz -0.011,
  0 falls; stand_still 10 s fine; extra 4 x 20 s and 4 x 30 s stand_still runs (same
  run() as eval-policy, long-g13.json): 0 falls, tilt <= 1.6 deg; 4 x 20 s at 0.2:
  vx 0.159-0.161, 0 falls. walk_backward_0.15 -0.124 to -0.128 (83-85 %).
  turn_left_0.5 wz 0.002 and turn_1.0 wz 0.001: turning is still lost.
  model_17100: 0.2 -> 0.067-0.068 (fails).
policy.onnx sha256: model_17300 2a55c12e82be6f16e4b1b9bc09f41d74a919fc632b9ad8dc17b8de5489b26904
  (model_17300.pt 26986b6cf963a32bb90e6658a9d39b0dce1b4f6bfad3afd8e07a644a5a3eb090)
notes: this checkpoint is the first Microduck BAM policy to pass G3.3 on the M5, taken
  from a run that was stopped for high training falls; the speed-relative yaw kernel
  did not bring turning back within 300 iterations.
```

---

## MAC-29 — Microduck: confirm the G3.3 pass holds (WP3-17, gate G3.3)
Status: `done (mac-m5, 2026-11-19): not confirmed (falls rule fired, no checkpoint passes)` · Deps: MAC-28 done (its `model_17300.pt`)

MAC-28 (mac-m5): `--track-ang-ref-speed 1.0` raised training falls to 1.93/it, and the
rule stopped the run at 17300. model_17300 meets G1.3 on BAM: 0.2 gives 0.160–0.161 (80 %)
with |wz| 0.011, it stands 30 s, and 0.3 / 0.4 give 0.255 / 0.342. But it comes from a run
that was unstable: at 17100 the curve was back (0.2 → 0.068, wz −0.25). Turning stayed
lost, so the yaw ref did not help with what it was for, and it is dropped here. Continue
model_17300 with the stable MAC-27 flags (no `--track-ang-ref-speed`) for 1000 iterations.
The goal is a G3.3 pass that holds across checkpoints, not a single lucky one. Use a
clean worktree of origin/main, the absolute checkpoint path, and `caffeinate -i nohup`.
```bash
uv run python scripts/train_ppo.py --actuator bam --dr --envs 256 --iters 1000 \
    --workers <best> --seed 0 --track-lin-std2 0.05 --forward-min-vx 0.1 --posture-gate \
    --track-lin-ref-speed 0.4 --symmetry loss --mirror-loss-coeff 0.5 \
    --resume <mac28>/runs/mac28-seed0/model_17300.pt --run-dir runs/mac29-seed0 \
    --save-every 100 --out runs/mac29-seed0/report.json
```
Keep the same falls rule. Export and evaluate model_17500, 17800, 18000 and 18300, or
the last ones saved if the rule stops the run. Use the `--resume <ckpt> --iters 0` export
from MAC-28. For each, run G1.3 (BAM) and the sweep (0.1–0.4, backward 0.3, lateral 0.3,
turn 1.0, with dy and the yaw rate). For the best one, also run eval_velocity_env at 0.4
and the 4 × 20 s / 4 × 30 s stands. Commit the JSONs under `docs/experiments/mac29/`.

**Acceptance (G3.3 confirmed):** at least two consecutive evaluated checkpoints meet G1.3
with BAM (0.2 m/s within ±30 %, 0 falls, |wz| at 0.2 below 0.05). The training falls
rule must not fire. Turning is out of scope here (WP3-18).

Result (env: mac-m5, 2026-11-19): NOT confirmed; the falls rule fired and no checkpoint meets G1.3
```
revision: 3e276f0 (main), git_dirty false; resumed from mac28-seed0/model_17300.pt with the
  MAC-27 flags, 8 workers; a VM used ~4-7 cores
wall time; training falls trend; rule fired?: 15 min until stopped. falls/it (100-it means)
  2.15 / 1.57 / 1.28 / 1.53 / 1.47 / 1.64; the watchdog's 200-it mean read 1.43, 1.41,
  1.50, then 1.55 at it 600 -> YES, the rule fired; training stopped at it 17900 (600
  of 1000). loss_symmetry 0.0148 -> 0.0111; air_time ~0.019; tracking ~0.017.
  Evaluated the last four saved checkpoints: 17600, 17700, 17800, 17900.
per checkpoint: G1.3 0.2 vx / wz, backward 0.15, sweep 0.3 / 0.4, turn 1.0 wz:
  17600: 0.132-0.133 (66 %) / -0.050; -0.154; 0.179 (wz -0.12) / 0.198 (wz -0.18); 0.002
  17700: 0.081-0.092 / -0.132;          -0.084; 0.216 (wz -0.05) / 0.221 (wz +0.12); 0.002
  17800: 0.128-0.135 / -0.097;          -0.119; 0.168 (wz -0.15) / 0.220 (wz -0.12); 0.002
  17900: 0.125-0.129 (63 %) / +0.025;   -0.120; 0.141 (wz +0.14) / 0.239 (wz +0.08); 0.001
  0 falls in every G1.3 case; stand_still fine; turn_left_0.5 wz <= 0.006 everywhere
best checkpoint: 17600. eval_velocity_env 0.4: vx 0.295 / 0.303 (noise on / off), 0 falls;
  stand 4 x 20 s and 4 x 30 s: 0 falls, tilt 1.6 deg; 0.2 for 4 x 20 s: 0.113-0.116, 0 falls
verdict: not confirmed. No checkpoint reaches 0.14 at 0.2 with |wz| < 0.05 (best 0.133 at
  |wz| 0.050), and the training falls rule fired. MAC-28's model_17300 (0.160) was a
  snapshot, not a stable state: from it the gait drifts again (|wz| up to 0.18 at 0.4)
  and 0.2 settles at 63-66 %, the MAC-27 level.
policy.onnx sha256: 17600 e6401b86666adc6eb77afcb260fe8a0ff935f9123f0f2d241583a34a53b347e4
```

---

## MAC-30 — Microduck: why turning was lost, a reward check (WP3-18, diagnostic)
Status: `done (mac-m5, 2026-11-19): verdict (b)` · Deps: MAC-29 done; the MAC-24 `policy.onnx` (run it after MAC-29)

MAC-24 turned at 1.0 rad/s (wz 0.965) on BAM. Every policy since MAC-26 stands still at
turn-only commands (0.5 and 1.0 give ≈ 0). There are two possible causes. (a) The current
reward pays standing more than turning, a reward-design problem. (b) Turning still pays
more, and the policy forgot it, an optimisation problem. `eval_velocity_env.py` now takes
`--twist` (repeatable) and the training reward flags, and it reports `wz_mean`. Score
both policies under the reward MAC-27 / MAC-29 train with. This is evaluation only, a few
minutes, with no training. Run it from a clean worktree of origin/main:
```bash
for p in <mac24>/runs/mac24-seed0/policy.onnx <mac28>/runs/mac28-seed0/policy-17300.onnx \
         <mac29 best>.onnx; do
  uv run python scripts/eval_velocity_env.py "$p" --actuators bam --envs 8 --seconds 10 \
      --track-lin-std2 0.05 --posture-gate --track-lin-ref-speed 0.4 \
      --twist 0 0 1.0 --twist 0 0 0.5 --twist 0.2 0 0 --twist 0 0 0 \
      --out docs/experiments/mac30/$(basename "$(dirname "$p")")-$(basename "$p" .onnx).json
done
```
Use the ONNX names that the MAC-28 / MAC-29 exports actually have. If the MAC-24 ONNX is
gone, export it from MAC-24's last checkpoint with the MAC-24 flags and `--iters 0`.
Commit the JSONs under `docs/experiments/mac30/`.

**Report:** per policy and twist, give `wz_mean`, `vx_mean`, falls, `reward_per_step`,
and the `track_angular_velocity`, `track_linear_velocity`, `upright` and `pose` terms
(noise off). The verdict is (a) if MAC-24's turning earns less `reward_per_step` at
turn 1.0 than the standing policies, and (b) otherwise.

Result (env: mac-m5, 2026-11-19): verdict (b), turning still pays; the policies forgot it
```
revision: 70b875d (main), git_dirty false; eval only, BAM, 8 envs x 10 s, reward flags
  --track-lin-std2 0.05 --posture-gate --track-lin-ref-speed 0.4; noise off below
turn 1.0 / turn 0.5 / fwd 0.2 / zero, per policy (wz, vx, falls, reward/step |
  track_ang, track_lin, upright, pose):
  MAC-24 policy.onnx:
    turn 1.0  wz 0.880 vx 0.011 0  0.1064 | 0.0267 0.0316 0.0214 0.0066
    turn 0.5  wz 0.002 vx 0.000 0  0.1159 | 0.0250 0.0399 0.0247 0.0114
    fwd 0.2   wz -0.001 vx 0.000 0 0.0317 | 0.0398 0.0017 0.0017 0.0008
    zero      wz 0.001 vx 0.000 0  0.1657 | 0.0399 0.0399 0.0399 0.0172
  MAC-28 model_17300:
    turn 1.0  wz 0.003 vx 0.000 0  0.0520 | 0.0070 0.0399 0.0069 0.0022
    turn 0.5  wz 0.004 vx 0.000 0  0.1120 | 0.0249 0.0399 0.0248 0.0089
    fwd 0.2   wz -0.007 vx 0.165 0 0.0651 | 0.0261 0.0116 0.0081 0.0005
    zero      wz -0.001 vx 0.000 0 0.1565 | 0.0398 0.0399 0.0398 0.0106
  MAC-29 model_17600:
    turn 1.0  wz 0.005 vx 0.000 0  0.0475 | 0.0074 0.0399 0.0067 0.0023
    turn 0.5  wz 0.002 vx 0.000 0  0.1142 | 0.0249 0.0399 0.0246 0.0114
    fwd 0.2   wz -0.050 vx 0.140 0 0.0677 | 0.0304 0.0106 0.0075 0.0002
    zero      wz -0.001 vx 0.000 0 0.1626 | 0.0399 0.0399 0.0400 0.0158
verdict (a) or (b): (b). At turn 1.0 MAC-24's turning (wz 0.88) earns 0.106 reward/step under
  the current reward, about 2x the standing policies (0.052 / 0.048); its yaw term is 4x
  theirs and the gated upright term 3x. Turning pays more, so this is an optimisation
  problem (forgotten), not reward design. Caveat: at turn 0.5 all three, MAC-24 too,
  stand, and standing still keeps 62 % of the yaw term (0.025 of 0.040) and about 0.11-0.12
  reward/step, so 0.5 has little pressure under the current yaw kernel.
```

---

## MAC-31 — Microduck: steadier PPO and more turn commands (WP3-19, gate G3.3)
Status: `done (mac-m5, 2026-11-19): not confirmed (1 of 5 checkpoints passes)` · Deps: MAC-28 `model_17300.pt`, MAC-29 / MAC-30 done

MAC-29 (mac-m5) did not confirm G3.3. The falls rule fired at 17900. The best
checkpoints reached 0.125–0.135 at 0.2 (≥ 0.14 needed), and the yaw drift changes sign
from checkpoint to checkpoint. model_17300 was a snapshot, not a steady state. MAC-30
(mac-m5): under the current reward, turning in place at 1.0 pays about 2× standing
(MAC-24 0.106/step against 0.048–0.052), so turning was forgotten, not outscored
(verdict b). Two new knobs, both off by default:
- `--desired-kl` sets the PPO adaptive-lr KL target (default 0.01). 0.005 takes smaller
  steps, aimed at the checkpoint-to-checkpoint swings.
- `--turn-fraction` sets the share of turn-only command draws (default 0.15). 0.30
  rehearses turning twice as often.

Continue model_17300 with the MAC-27 flags plus both knobs for 1500 iterations. Use a
clean worktree of origin/main, the absolute checkpoint path, and `caffeinate -i nohup`.
```bash
uv run python scripts/train_ppo.py --actuator bam --dr --envs 256 --iters 1500 \
    --workers <best> --seed 0 --track-lin-std2 0.05 --forward-min-vx 0.1 --posture-gate \
    --track-lin-ref-speed 0.4 --symmetry loss --mirror-loss-coeff 0.5 \
    --desired-kl 0.005 --turn-fraction 0.3 \
    --resume <mac28>/runs/mac28-seed0/model_17300.pt --run-dir runs/mac31-seed0 \
    --save-every 100 --out runs/mac31-seed0/report.json
```
Keep the MAC-29 falls rule (stop, then evaluate the last 4 checkpoints). Evaluate model_17600,
17900, 18200, 18500 and 18800 (or the last ones saved) as in MAC-29: G1.3 (BAM) and the
sweep, with dy and the yaw rate. Also run the MAC-30 eval (`--twist 0 0 1.0 --twist 0 0 0.5`
with the training reward flags) for each. For the best one, run eval_velocity_env at 0.4
and the 4 × 20 s / 4 × 30 s stands. Commit the JSONs under `docs/experiments/mac31/`.

**Acceptance:** G3.3 confirmed as in MAC-29 (two consecutive checkpoints meet G1.3 with
BAM, |wz| at 0.2 below 0.05, falls rule not fired). WP3-18 also passes if turn 1.0 gives
wz ≥ 0.7 on those checkpoints.

Result (env: mac-m5, 2026-11-19): NOT confirmed (one passing checkpoint, not two); steadier training, turning partly back
```
revision: e4ebe56 (main), git_dirty false; resumed from mac28-seed0/model_17300.pt with the
  MAC-27 flags + --desired-kl 0.005 --turn-fraction 0.3, 8 workers; a VM used ~4-6 cores
wall time; training falls trend; rule fired?: 1756 s (29 min), all 1500 it. falls/it
  (100-it means) 1.98 (first 100) -> 1.10-1.22 -> 0.82-1.01 -> 0.74-0.88 at the end; the
  watchdog's 200-it mean stayed 0.81-1.19 -> NO, the rule did not fire. reward 0.026 ->
  0.034; track_angular_velocity 0.011 -> 0.013; tracking 0.020 -> 0.021; loss_symmetry
  0.013 -> 0.010
per checkpoint: G1.3 0.2 vx / wz, backward 0.15, sweep 0.3 / 0.4, turn 1.0 / 0.5 wz
  (turn wz from the MAC-30 env eval, noise off; sweep turn_1.0 in brackets):
  17600: 0.134-0.136 / -0.064; -0.140; 0.239 / 0.315; 0.004 (0.003) / 0.004
  17900: 0.132-0.134 / -0.061; -0.001; 0.234 / 0.276; 0.003 (0.001) / 0.003
  18200: 0.118-0.120 / -0.052; -0.001; 0.182 / 0.204; 0.575 (-0.183) / 0.003
  18500: 0.151-0.154 / +0.016; -0.132; 0.218 / 0.246; 0.240 (0.021) / 0.006  <- passes G1.3
  18800: 0.144-0.145 / +0.088; -0.147; 0.206 / 0.262; 0.005 (0.004) / 0.005
  0 falls in every G1.3 case; stand_still fine; G1.3 turn_left_0.5 wz <= 0.004 everywhere
  extra (not in the protocol list): 18400 0.131-0.135 / -0.093, 18600 0.147-0.148 / -0.107
best checkpoint: 18500. eval_velocity_env 0.4: vx 0.271 / 0.273 (noise on / off), 0 falls;
  stand 4 x 20 s and 4 x 30 s: 0 falls, tilt <= 1.8 deg; 0.2 for 4 x 20 s: 0.149-0.155,
  0 falls
verdict: G3.3 not confirmed. Only 18500 meets G1.3 (0.2 within +-30 % and |wz| < 0.05);
  its neighbours 18200 / 18800 (and 18400 / 18600) fail on speed or |wz|. The falls rule
  did not fire, and training was steadier than MAC-29 (falls ~0.8-1.0, reward rising).
  The yaw at 0.2 still changes sign between checkpoints (-0.06 ... +0.09).
  WP3-18 not met: turn 1.0 reached 0.575 (18200) and 0.240 (18500), not >= 0.7, and
  turn 0.5 stays ~0.
policy.onnx sha256: 18500 70053902f4fa400e83d523cb2d0d09ff3df8ba73585bd92c1fc4ab7ac429cc76
```

---

## MAC-32 — Microduck: stronger mirror loss against the yaw drift (WP3-19, gate G3.3)
Status: `done (mac-m5, 2026-11-19): G3.3 confirmed; WP3-18 not met` · Deps: MAC-31 done (its `model_18500.pt`)

MAC-31 (mac-m5) trained steadily. The falls rule did not fire, falls ended at ~0.8/it,
and the reward rose from 0.026 to 0.034. G3.3 is still not confirmed. model_18500 passes
G1.3 (0.2 → 0.151–0.154, wz +0.016, stands 30 s), but every checkpoint around it fails on
yaw: |wz| is 0.05–0.11 at 0.2, and the sign changes from one checkpoint to the next.
Turning came partly back, with turn 1.0 reaching 0.575 at 18200. Speed is no longer the
problem; the yaw drift is. The mirror loss is what removed the curve in MAC-27, so double
it to 1.0, and raise the turn share to 0.4. Continue model_18500 with the MAC-31 flags
changed that way, for 1500 iterations. Use a clean worktree of origin/main, the absolute
checkpoint path, and `caffeinate -i nohup`.
```bash
uv run python scripts/train_ppo.py --actuator bam --dr --envs 256 --iters 1500 \
    --workers <best> --seed 0 --track-lin-std2 0.05 --forward-min-vx 0.1 --posture-gate \
    --track-lin-ref-speed 0.4 --symmetry loss --mirror-loss-coeff 1.0 \
    --desired-kl 0.005 --turn-fraction 0.4 \
    --resume <mac31>/runs/mac31-seed0/model_18500.pt --run-dir runs/mac32-seed0 \
    --save-every 100 --out runs/mac32-seed0/report.json
```
Keep the falls rule. Evaluate every checkpoint from 18800 to 20000 (13 checkpoints, 300 it
apart is too coarse to show two in a row) with G1.3 (BAM) only, then run the sweep and
the MAC-30 turn eval on the passing ones and on the last one. For the best one, run
eval_velocity_env at 0.4 and the 4 × 20 s / 4 × 30 s stands. Commit the JSONs under
`docs/experiments/mac32/`, and record `loss_symmetry` from first to last.

**Acceptance:** as MAC-31. G3.3 is confirmed when two consecutive checkpoints meet G1.3
with BAM (0.2 within ±30 %, |wz| at 0.2 below 0.05, 0 falls) and the falls rule does
not fire. WP3-18 passes if turn 1.0 gives wz ≥ 0.7 on those checkpoints.

Result (env: mac-m5, 2026-11-19): G3.3 CONFIRMED (three runs of consecutive passing checkpoints); WP3-18 not met
```
revision: 8e60bf7 (main), git_dirty false; resumed from mac31-seed0/model_18500.pt, MAC-31
  flags with --mirror-loss-coeff 1.0 --turn-fraction 0.4, 8 workers; VM load ~4-6 cores
wall time; loss_symmetry first -> last; training falls trend; rule fired?: 1316 s (22 min),
  all 1500 it; loss_symmetry 0.0109 (first 100 it) -> 0.0070 (last 100); falls/it (100-it
  means) 1.05 -> 0.57-0.91 -> 1.03 at the end, watchdog 200-it mean <= 0.84 -> NO, the rule
  did not fire; reward 0.037 -> 0.042; track_angular_velocity 0.0136 -> 0.0154;
  tracking ~0.023
per checkpoint (G1.3 0.2 vx / wz, backward 0.15; P = meets G1.3, 0 falls everywhere):
  18800 0.159-0.159 / -0.028 / -0.163 P   19500 0.157-0.159 / +0.050 / -0.029
  18900 0.150-0.152 / -0.015 / -0.001 P   19600 0.139-0.142 / +0.011 / -0.097
  19000 0.158-0.159 / +0.025 / -0.001 P   19700 0.168-0.170 / -0.023 / -0.147 P
  19100 0.134-0.142 / -0.097 / -0.001     19800 0.157-0.165 / +0.040 / -0.159 P
  19200 0.158-0.160 / +0.035 / -0.001 P   19900 0.165-0.166 / -0.034 / -0.138 P
  19300 0.141-0.142 / -0.009 / -0.032 P   20000 0.143-0.147 / +0.107 / -0.031
  19400 0.136-0.144 / +0.046 / -0.122
  -> consecutive passes: 18800-18900-19000, 19200-19300, 19700-19800-19900
passing ones: sweep 0.3 / 0.4 (wz), turn 1.0 (env eval / sweep) / 0.5 wz:
  18800 0.192 (+0.11) / 0.236 (+0.11); 0.277 / -0.010; 0.009
  18900 0.220 (-0.07) / 0.288 (-0.06); 0.127 / 0.057;  0.007
  19000 0.234 (+0.06) / 0.230 (+0.12); 0.041 / 0.068;  0.004
  19200 0.242 (+0.00) / 0.304 (-0.00); 0.465 / -0.014; 0.011
  19300 0.216 (-0.05) / 0.272 (-0.06); 0.031 / -0.000; 0.007
  19700 0.265 (-0.01) / 0.309 (+0.04); 0.360 / 0.368;  0.010
  19800 0.253 (-0.00) / 0.259 (-0.13); 0.336 / 0.362;  0.012
  19900 0.237 (-0.07) / 0.292 (+0.06); 0.320 / -0.125; 0.011
  last (20000, fails G1.3 on wz +0.107): 0.185 / 0.226; 0.401 / 0.430; 0.002
best checkpoint: 19700. eval_velocity_env 0.4: vx 0.336 / 0.343 (noise on / off), 0 falls;
  stand 4 x 20 s and 4 x 30 s: 0 falls, tilt 1.3 deg; 0.2 for 4 x 20 s: 0.164-0.166.
  Long stands also for 19800 (0 falls; 0.2 for 20 s 0.136-0.153) and 19900 (0 falls;
  0.2 for 20 s 0.159-0.161).
verdict: G3.3 confirmed on mac-m5: 8 of 13 checkpoints meet G1.3 with BAM (0.2 within
  +-30 %, |wz| < 0.05, 0 falls), in three runs of two or three in a row, and the falls rule
  did not fire. WP3-18 not met: turn 1.0 reaches 0.465 at most (19200), >= 0.7 needed;
  turn 0.5 stays ~0.01.
policy.onnx sha256: 19700 398f399aab64332d13083a81a8b31ff96c06c40b2622b76a79eb30e424bb2846
```

---

## MAC-33 — Microduck: bring turning back on the G3.3 policy (WP3-20)
Status: `done (mac-m5, 2026-11-19): WP3-20 not met; G3.3 holds` · Deps: MAC-32 done (its `model_19700.pt`)

MAC-32 (mac-m5) met G3.3, and model_19700 is the Microduck reference. Turning is still
weak: turn 1.0 reached at most 0.465, and 0.5 stays at ≈ 0.01. In MAC-28,
`--track-ang-ref-speed 1.0` had no effect on turning, because the run became unstable
(1.93 falls/it) and was stopped after 300 iterations. Training is now steady (KL 0.005,
mirror loss 1.0, falls ≤ 0.84/it), so try the yaw ref again on top of it. It makes
standing still cost as much at a 0.5 turn as at 1.0. Continue model_19700 with the
MAC-32 flags plus `--track-ang-ref-speed 1.0`, for 1500 iterations. Use a clean worktree
of origin/main, the absolute checkpoint path, and `caffeinate -i nohup`.
```bash
uv run python scripts/train_ppo.py --actuator bam --dr --envs 256 --iters 1500 \
    --workers <best> --seed 0 --track-lin-std2 0.05 --forward-min-vx 0.1 --posture-gate \
    --track-lin-ref-speed 0.4 --symmetry loss --mirror-loss-coeff 1.0 \
    --desired-kl 0.005 --turn-fraction 0.4 --track-ang-ref-speed 1.0 \
    --resume <mac32>/runs/mac32-seed0/model_19700.pt --run-dir runs/mac33-seed0 \
    --save-every 100 --out runs/mac33-seed0/report.json
```
Keep the falls rule. For every checkpoint from 19800 to 21200, run G1.3 (BAM) and the
MAC-30 turn eval. For the MAC-30 eval, pass `--track-ang-ref-speed 1.0` as well, with
`--twist 0 0 1.0 --twist 0 0 0.5 --twist 0 0 0`. Run the sweep on the best one. Commit
the JSONs under `docs/experiments/mac33/`.

**Acceptance (WP3-20):** on two consecutive checkpoints, turn 1.0 gives wz ≥ 0.7 and
turn 0.5 gives ≥ 0.35. Those same checkpoints must still meet G1.3 with BAM (0.2 within
±30 %, |wz| < 0.05, 0 falls), and the falls rule must not fire. If turning improves but
G3.3 breaks, report both. model_19700 stays the reference until a checkpoint does both.

Result (env: mac-m5, 2026-11-19): WP3-20 NOT met (turn 0.5 stays ~0); G3.3 still holds
```
revision: 54151b4 (main), git_dirty false; resumed from mac32-seed0/model_19700.pt, MAC-32
  flags + --track-ang-ref-speed 1.0, 8 workers; VM ~2-5 cores
wall time; loss_symmetry; training falls trend; rule fired?: 1311 s (22 min), all 1500 it;
  loss_symmetry 0.0074 -> 0.0078 (first / last 100 it); falls/it (100-it means) 0.73-1.26,
  watchdog 200-it mean <= 0.94 -> NO, the rule did not fire. track_angular_velocity
  0.0107 -> 0.0094 (fell); tracking ~0.023; reward 0.031 -> 0.028
per checkpoint (G1.3 0.2 vx / wz; turn 1.0 / 0.5 / 0 wz, env eval with the yaw ref; G = meets G1.3):
  19800 0.132-0.137 / -0.124;  0.691 / 0.008 / -0.001
  19900 0.135-0.147 / -0.097;  1.041 / 0.008 / -0.000   (turn 1.0 >= 0.7, fails G1.3)
  20000 0.181-0.183 / +0.025;  0.694 / 0.004 / -0.000  G
  20100 0.141-0.144 / -0.074;  0.397 / 0.002 / -0.001
  20200 0.162-0.165 / -0.013;  0.335 / 0.003 / -0.001  G
  20300 0.155-0.156 / +0.002;  0.573 / 0.004 / -0.000  G
  20400 0.171-0.172 / -0.004;  0.364 / 0.006 / -0.000  G
  20500 0.148-0.150 / -0.013;  0.472 / 0.003 / -0.000  G
  20600 0.137-0.140 / +0.110;  0.254 / 0.002 / -0.000
  20700 0.158-0.160 / -0.005;  0.398 / 0.002 / +0.000  G
  20800 0.160-0.161 / +0.010;  0.399 / 0.009 / +0.000  G
  20900 0.083-0.088 / +0.214;  0.355 / 0.006 / -0.000
  21000 0.169-0.171 / -0.038;  0.337 / 0.005 / -0.000  G
  21100 0.151-0.153 / +0.043;  0.400 / 0.003 / -0.000  G
  21200 0.155-0.156 / +0.069;  0.604 / 0.011 / +0.000
  0 falls in every G1.3 case; G1.3 turn_left_0.5 wz <= 0.009 everywhere
best: 20000 (passes G1.3, highest turn 1.0 among the passing ones). sweep: fwd 0.1 / 0.3 /
  0.4 -> 0.083 / 0.265 / 0.296 (first steps at 0.1), back 0.3 -0.217, lateral dy +0.76;
  turn_1.0 FELL at 3.96 s (in the sweep's upstream init; the env eval turned 0.694 without
  falls). Long stands: 4 x 20 s / 4 x 30 s 0 falls, tilt <= 1.0 deg; 0.2 for 4 x 20 s
  0.176-0.179, 0 falls
verdict: WP3-20 not met. Turn 0.5 gives <= 0.011 on every checkpoint (>= 0.35 needed), and
  turn 1.0 >= 0.7 appears once (19900), on a checkpoint that fails G1.3. The yaw ref did
  not move turn 0.5 at all, and the yaw term fell during the run. G3.3 still holds:
  10 of 15 meet G1.3, in runs 20200-20500 (4) and 20700-20800 and 21000-21100, with no
  falls-rule stop. model_19700 stays the reference.
policy.onnx sha256: 20000 e87b7134d0a8d823f65eff3b92e5e23be45830012097e060a46195ecde6edbdb
```

---

## MAC-34 — Microduck: DR ablation and ONNX check on the G3.3 policy (WP3-05, gates G3.4, G3.5)
Status: `done (mac-m5, 2026-11-20): G3.4 report done, G3.5 mac-m5 part met` · Deps: MAC-32 done (its `model_19700.pt`)

G3.4 asks for a domain-randomization ablation (with vs without DR), and G3.5 for the
exported ONNX policy to run in the same runner as the official one on the M5. Both use
the reference model_19700 (trained with `--dr`). For the "without" arm, continue
model_19700 with the MAC-32 flags but **without `--dr`**, for 1000 iterations. Use a
clean worktree of origin/main, the absolute checkpoint path, and `caffeinate -i nohup`.
```bash
uv run python scripts/train_ppo.py --actuator bam --envs 256 --iters 1000 \
    --workers <best> --seed 0 --track-lin-std2 0.05 --forward-min-vx 0.1 --posture-gate \
    --track-lin-ref-speed 0.4 --symmetry loss --mirror-loss-coeff 1.0 \
    --desired-kl 0.005 --turn-fraction 0.4 \
    --resume <mac32>/runs/mac32-seed0/model_19700.pt --run-dir runs/mac34-nodr \
    --save-every 100 --out runs/mac34-nodr/report.json
```
Keep the falls rule. Then score two policies, A = model_19700 (DR) and B = the newest
no-DR checkpoint that meets G1.3 (if none does, the last one), each exported to ONNX:
1. `scripts/eval_velocity_env.py <policy.onnx> --actuators bam --envs 8 --twist 0.2 0 0
   --twist 0.3 0 0 --twist 0 0 0`, once without and once with `--dr` (4 runs).
2. `bipedlab eval-policy <policy.onnx> --model microduck-walk --actuator bam --repeats 4`
   (G1.3, the official policies' runner = G3.5 on mac-m5), for A and B.
Commit the JSONs and both `run.json`/`report.json` under `docs/experiments/mac34/`,
with each policy.onnx sha256.

**Acceptance:** G3.4 is a report, so it passes when the table exists: for A and B, vx at
0.2 / 0.3, |wz|, falls, DR off and DR on, plus G1.3. Say which policy degrades less under
DR. G3.5 (mac-m5 part) passes when A's ONNX meets G1.3 through `eval-policy` on the M5.
model_19700 stays the reference whatever B does.

Result (env: mac-m5, 2026-11-20): G3.4 report done (A degrades less under DR); G3.5 mac-m5 part met
```
revision: dd646be (main), git_dirty false; B = model_19700 continued WITHOUT --dr, MAC-32
  flags otherwise, 8 workers; A = model_19700 (its MAC-32 export)
wall time; falls rule fired?: 861 s (14 min), all 1000 it; training falls/it 0.04-0.12
  (200-it mean <= 0.12) -> no; reward 0.050 -> 0.060 (no DR, easier)
B checkpoint chosen: 20400, the newest no-DR checkpoint meeting G1.3 (passes: 19800, 20300,
  20400; 19900 fell once in turn_left_0.5; the others fail on |wz| 0.05-0.10 or speed)
table (A / B; DR off -> DR on; vx 0.2, vx 0.3, |wz| at 0, falls; eval_velocity_env, 8 envs,
  noise on / off):
  A vx 0.2  0.179 / 0.177 -> 0.174 / 0.171   B vx 0.2  0.191 / 0.182 -> 0.206 / 0.195
  A vx 0.3  0.268 / 0.271 -> 0.268 / 0.269   B vx 0.3  0.282 / 0.279 -> 0.299 / 0.295
  A |wz| 0  0.000 / 0.001 -> 0.002 / 0.008   B |wz| 0  0.000 / 0.000 -> 0.016 / 0.010
  A |wz| 0.2 / 0.3 (noise off) 0.027 / 0.009 -> 0.029 / 0.048
  B |wz| 0.2 / 0.3 (noise off) 0.035 / 0.029 -> 0.048 / 0.018 (noise on: 0.087 / 0.097)
  A falls   0 / 0 / 0 -> 0 / 0 / 0 (0.2 / 0.3 / 0)
  B falls   0 / 0 / 0 -> 1 / 2 / 1 (noise on at 0.2 and 0.3, noise off at 0; of 8 envs)
G1.3 A / B: A 0.168-0.170, wz -0.023, 0 falls (meets) / B 0.171-0.173, wz -0.032, 0 falls (meets)
policy.onnx sha256 A / B: 398f399aab64332d13083a81a8b31ff96c06c40b2622b76a79eb30e424bb2846 /
  f7cac8af4a7151aec4c2d195b33ee72e5f1bfbb467590ba59bb6c0f39835e996
verdict: G3.4: A (trained with DR) degrades less under DR. Its speed barely moves (-3 % at
  0.2, 0 % at 0.3) and it never falls, while B (no DR) falls 4 times in 8-env DR runs and
  its yaw with noise doubles (0.087 / 0.097). Without DR, B is a few % faster. G3.5 (mac-m5
  part): met. A's ONNX meets G1.3 through `bipedlab eval-policy`, the official policies'
  runner, on the M5. model_19700 stays the reference.
```

---

## Later Mac tasks (not ready yet)
- MAC-05 live telemetry overlay (after WP2-01)
