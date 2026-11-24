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

## MAC-35 — biped_v1: size the actuators from our own gait (WP4-25, input to G4.1 / WP4-05)
Status: `done (mac-m5, 2026-11-20): 4 of 5 joints above 0.52 N.m` · Deps: MAC-25 done (its `runs/mac25/policy.onnx`)

WP4-01 sized our joints from Microduck's walking dynamic factors. Now `eval_biped.py`
reports, per joint type (both legs pooled), |torque| RMS / p95 / max, |speed| p95 / max
and mechanical power, plus a `sizing` block: 2× the worst walking max torque and p95
speed (G4.1 margin), from the policy's nominal walking cases with DR off and on. Run
it on the MAC-25 reference from a clean worktree of origin/main, with the absolute
policy path. No training; eval only.
```bash
uv run python scripts/eval_biped.py <mac25>/runs/mac25/policy.onnx --dr --robustness \
    --envs 8 --seconds 20 --out runs/mac35/eval.json
uv run python scripts/size_biped.py --out runs/mac35/size-wp4-01.json
```
Commit both JSONs under `docs/experiments/mac35/`, with the policy.onnx sha256.

**Acceptance (WP4-25):** the `sizing` block exists for all five joint types. Put a
table in the result next to WP4-01 (`size-wp4-01.json`): required torque (N·m) and
speed (rad/s) per joint, and RMS torque. Name the joints whose 2× requirement is above
the XL330-class placeholder (0.52 N·m). Also say whether any robustness case
(pushes, friction) has a max torque above the walking max.

Result (env: mac-m5, 2026-11-20): sizing block present for all five joints; four of five need more than 0.52 N.m
```
revision: c94e0d4 (main), git_dirty false; eval only (eval_biped --dr --robustness, 8 envs x
  20 s; sizing from the 6 nominal walking cases with DR off and on); size_biped.py for WP4-01
policy.onnx sha256: 3789a53b243f32d81492c42d0e8c793c31c10a35ddf26a3f9d15a4872745a52b (MAC-25)
table (joint: required torque / speed = 2x walking max torque / 2x p95 speed; RMS; WP4-01 required):
  hip_yaw      0.37 N.m /  2.8 rad/s; RMS 0.054   (WP4-01 0.21 / 2.2)
  hip_roll   >=1.04 N.m /  5.5 rad/s; RMS 0.183   (WP4-01 0.35 / 2.5)  max at the 0.52 clip
  hip_pitch  >=1.04 N.m / 13.7 rad/s; RMS 0.217   (WP4-01 0.37 / 7.4)  max at the clip, p95 = 0.52 at 0.25 m/s
  knee       >=1.04 N.m / 12.2 rad/s; RMS 0.120   (WP4-01 0.68 / 10.0) max at the clip
  ankle_pitch  0.81 N.m /  7.7 rad/s; RMS 0.120   (WP4-01 0.44 / 5.3)
joints above 0.52 N*m: hip_roll, hip_pitch, knee (each >= 1.04, a lower bound because the
  walking max is the 0.52 clip) and ankle_pitch (0.81); only hip_yaw (0.37) fits. Speed:
  hip_pitch (13.7) and knee (12.2) also need more than the placeholder's 10 rad/s.
robustness cases above walking max: none. But for hip_roll / hip_pitch / knee the walking max
  is already the clip, so no case can exceed it: pushes (push_sideways knee and hip_roll at
  0.52) and friction_1.5 (hip_pitch p95 0.41) reach the clip too. Latency cases stay below.
verdict: WP4-25 acceptance met (sizing for all five types, table vs WP4-01). Our own gait
  needs much more than WP4-01 estimated: hip roll / hip pitch about 3x, knee 1.5x and up,
  ankle 1.8x, and hip pitch spends more than 5 % of the 0.25 m/s walk at the clip. The XL330-class
  placeholder (0.52 N.m, 10 rad/s) fails G4.1 at four of five joints in torque and at
  hip pitch / knee in speed. Because three joints are torque-limited in sim, the true
  demand is higher than shown; a re-run with a raised torque limit would measure it.
```

---

## MAC-36 — biped_v1: the torque demand without the 0.52 N·m clip (WP4-26, input to WP4-05)
Status: `done (mac-m5, 2026-11-20): demand measured; step 2 misses the 40 ms row` · Deps: MAC-35 done, MAC-25 done (its newest `runs/mac25/model_<n>.pt` and `policy.onnx`)

MAC-35 (mac-m5) found the MAC-25 gait's walking max torque sitting on the 0.52 N·m limit
at hip roll, hip pitch and knee, so its 2× requirement (≥ 1.04) is only a lower bound.
`BipedVelocityEnv(torque_limit=…)`, `train_ppo.py --torque-limit` and
`eval_biped.py --torque-limit` now override the model's force limit, and the `sizing`
block flags `at_torque_limit` when a walking max reaches it. Use a clean worktree of
origin/main, absolute paths, and `caffeinate -i nohup`. Steps:
1. Same policy, limit raised (no training), to see how far the clipped joints go:
```bash
uv run python scripts/eval_biped.py <mac25>/runs/mac25/policy.onnx --dr --robustness \
    --envs 8 --seconds 20 --torque-limit 1.5 --out runs/mac36/eval-mac25-tl1.5.json
```
2. Fine-tune MAC-25 with the limit at 1.5 N·m (MAC-25 flags plus `--torque-limit 1.5`),
   500 iterations, then score its export with the same limit:
```bash
uv run python scripts/train_ppo.py --robot biped_v1 --dr --envs 256 --iters 500 \
    --workers <best> --seed 0 --track-yaw-std2 0.25 --symmetry loss --mirror-loss-coeff 0.5 \
    --torque-limit 1.5 --resume <mac25>/runs/mac25/model_<newest>.pt --run-dir runs/mac36 \
    --save-every 100 --out runs/mac36/report.json
uv run python scripts/eval_biped.py runs/mac36/policy.onnx --dr --robustness --envs 8 \
    --seconds 20 --torque-limit 1.5 --out runs/mac36/eval.json
```
Keep the falls rule (stop if the falls pass 1.0 per iteration for 100 iterations, and
score the last good checkpoint). Commit `report.json`, `run.json` and both eval JSONs
under `docs/experiments/mac36/`, with each policy.onnx sha256.

**Acceptance (WP4-26):** in step 2, no joint has `at_torque_limit`, and the policy still
meets the MAC-25 rows (0.15 / 0.25 walk with DR on and off, |wz| ≤ 0.08, 40 ms latency
≥ 80 % vx, 0 falls). Report the per-joint table (required torque / speed, RMS) for
steps 1 and 2 next to MAC-35. If a joint still reaches 1.5, say so: its requirement
stays a lower bound. MAC-25 stays the reference policy; this run only measures demand.

Result (env: mac-m5, 2026-11-20): demand measured unclipped (no at_torque_limit); step 2 misses the 40 ms row
```
revision: c0836f2 (main), git_dirty false; eval_biped --dr --robustness, 8 envs x 20 s, --torque-limit 1.5
step 1 table (MAC-25 at 1.5 N*m; required = 2x walking max / 2x p95 speed; RMS; at_torque_limit):
  hip_yaw      0.39 N.m /  3.0 rad/s; RMS 0.055; no    (MAC-35 at 0.52: 0.37 / 2.8)
  hip_roll     1.19 N.m /  5.5 rad/s; RMS 0.180; no    (MAC-35: >=1.04 / 5.5, clipped)
  hip_pitch    1.17 N.m / 14.2 rad/s; RMS 0.172; no    (MAC-35: >=1.04 / 13.7, clipped)
  knee         0.82 N.m / 12.1 rad/s; RMS 0.109; no    (MAC-35: >=1.04 / 12.2, clipped)
  ankle_pitch  0.76 N.m /  8.0 rad/s; RMS 0.119; no    (MAC-35: 0.81 / 7.7)
  walking max torques: hip roll 0.593, hip pitch 0.583, knee 0.410, ankle 0.379, hip yaw 0.197;
  0 falls in all 15 cases; 40 ms latency 0.126 (84 %)
step 2 wall time; falls rule fired?: 576 s (10 min), all 500 it from mac25/model_4000.pt; falls/it
  0.25 -> 0.04-0.14 (100-it mean <= 0.14 after it 100) -> no; reward 0.106 -> 0.117; knee
  torque mean 0.153 -> 0.134 N.m
step 2 table (fine-tune at 1.5 N*m; at_torque_limit):
  hip_yaw      0.40 N.m /  3.3 rad/s; RMS 0.049; no
  hip_roll     1.36 N.m /  5.2 rad/s; RMS 0.187; no    (walking max 0.679, DR on at 0.25)
  hip_pitch    1.14 N.m / 14.1 rad/s; RMS 0.167; no
  knee         0.95 N.m / 13.7 rad/s; RMS 0.113; no
  ankle_pitch  0.82 N.m /  8.0 rad/s; RMS 0.124; no
  robustness maxima: friction_1.5 hip roll 0.61, push_sideways hip roll 0.59 / knee 0.43; all
  below the walking max
step 2 MAC-25 rows: 0.15 / 0.25 DR off vx 0.161 / 0.252, |wz| 0.034 / 0.032 (pass); DR on vx
  0.132 / 0.215, |wz| 0.048 / 0.010 (pass); 40 ms latency 0.110 = 73 % of 0.15 (FAIL, >= 80 %);
  0 falls (pass); also turn 0.8 0.717 / 0.626, zero command |wz| 0.098 / 0.056 (MAC-25 0.034 / 0.020)
policy.onnx sha256: step 2 e992476a52e381b8f36de99a9ef68fae88b6ccbc0c8a7f07a8e1eea393ef7ac3;
  step 1 used MAC-25 3789a53b243f32d81492c42d0e8c793c31c10a35ddf26a3f9d15a4872745a52b
verdict: no joint reaches the 1.5 limit in either step, so the requirements are measured, not lower
  bounds. With the 2x margin: hip roll 1.2-1.4 N.m, hip pitch about 1.15 N.m at about 14 rad/s, knee 0.8-0.95 N.m
  at 12-14 rad/s, ankle about 0.8 N.m, hip yaw about 0.4 N.m. WP4-26 acceptance only partly met: no
  at_torque_limit (met), but the step 2 policy keeps only 73 % at 40 ms latency (MAC-25 row
  fails) and its zero-command yaw got worse. MAC-25 stays the reference; for sizing, step 1
  (the reference gait itself, unclipped) and step 2 agree within about 15 %, except hip roll (+14 %)
  and knee (+16 %) in step 2.
```

---

## MAC-37 — biped_v1: walk with a real servo's torque-speed curve (WP4-28, input to WP4-05)
Status: `done (mac-m5, 2026-11-20): B walks (rows met), 2x speed fails; A fails rows` · Deps: MAC-36 done, MAC-25 done (its `runs/mac25/model_4000.pt`)

MAC-36 (mac-m5) measured the gait's demand. WP4-27 (cloud) then found that no surveyed
servo passes hip pitch or knee: they move at p95 ≈ 7 rad/s, and the policy was trained
with an ideal actuator. `--motor-no-load-speed` (env, `train_ppo.py`, `eval_biped.py`)
adds a DC-motor curve. The force along the motion falls from the stall limit at rest
to 0 at the no-load speed, while braking keeps the full limit. Fine-tune MAC-25 with
two real candidates (datasheet values in `bipedlab.design.actuators`), 1000 iterations
each, one after the other. Use a clean worktree of origin/main, absolute paths, and
`caffeinate -i nohup`.
- A: XC330-M288-T @ 6 V, `--torque-limit 1.10 --motor-no-load-speed 10.2`
- B: STS3215 @ 7.4 V, `--torque-limit 1.91 --motor-no-load-speed 5.45`
```bash
uv run python scripts/train_ppo.py --robot biped_v1 --dr --envs 256 --iters 1000 \
    --workers <best> --seed 0 --track-yaw-std2 0.25 --symmetry loss --mirror-loss-coeff 0.5 \
    --torque-limit <A/B> --motor-no-load-speed <A/B> \
    --resume <mac25>/runs/mac25/model_4000.pt --run-dir runs/mac37-<a|b> \
    --save-every 100 --out runs/mac37-<a|b>/report.json
uv run python scripts/eval_biped.py runs/mac37-<a|b>/policy.onnx --dr --robustness --envs 8 \
    --seconds 20 --torque-limit <A/B> --motor-no-load-speed <A/B> \
    --out runs/mac37-<a|b>/eval.json
```
Keep the falls rule (stop when the falls pass 1.0 per iteration for 100 iterations, then
score the last good checkpoint). Before training, also score MAC-25 itself with each
curve (eval only, same flags) as the "no adaptation" baseline. Commit report/run/eval
JSONs under `docs/experiments/mac37/`, with each policy.onnx sha256.

**Acceptance (WP4-28):** for each candidate, say whether the fine-tune meets the MAC-25
rows under its curve: 0.15 / 0.25 walking with DR off and on, |wz| ≤ 0.08, 40 ms ≥ 80 %
of vx, 0 falls. Report the sizing table and the 2× check. A joint passes when its
torque_max ≤ stall / 2 and its speed_p95 ≤ no-load / 2. MAC-25 stays the reference
unless one of these walks and passes more joints.

Result (env: mac-m5, 2026-11-20): B (STS3215) fine-tune walks and meets every MAC-25 row but passes
the 2x check at 1 of 5 joints (speed); A (XC330) fails the rows; MAC-25 stays the reference
```
revision: dee26b3 (main), git_dirty false; eval_biped --dr --robustness, 8 envs x 20 s, with each curve
2x check: torque_max <= stall/2 and speed_p95 <= no-load/2; A: <= 0.55 N.m and <= 5.10 rad/s,
  B: <= 0.955 N.m and <= 2.725 rad/s (T = torque fails, S = speed fails)
baselines (MAC-25 under A / B curve; walks? falls?): both walk, 0 falls in all 15 cases.
  A: 0.15 / 0.25 DR off vx 0.167 / 0.247, |wz| 0.044 / 0.016; DR on 0.143 / 0.217, |wz| 0.055 / 0.025;
     40 ms 0.126 (84 %) -> meets the MAC-25 rows. 2x: hip_yaw pass, hip_roll T, hip_pitch T+S,
     knee T+S, ankle pass (2 of 5)
  B: DR off 0.158 / 0.238, |wz| 0.061 / 0.077; DR on 0.131 / 0.206, |wz| 0.063 / 0.080; 40 ms 0.114
     (76 %, fails). 2x: hip_yaw pass, hip_roll / hip_pitch / knee / ankle S (1 of 5)
A: wall time; falls rule fired?; MAC-25 rows; sizing table; 2x pass per joint:
  839 s, all 1000 it; falls/it 0.21 -> 0.04 (100-it mean <= 0.09 after it 100) -> no; reward 0.103 -> 0.124
  rows: DR off vx 0.156 / 0.242, |wz| 0.069 / 0.113 (0.25 FAILS); DR on 0.133 / 0.211, |wz| 0.041 /
  0.088 (0.25 FAILS); 40 ms 0.114 (76 %, FAILS); 0 falls; turn 0.8 0.892 / 0.769 -> does NOT meet
  sizing (torque_max / RMS / speed_p95; required 2x): hip_yaw 0.249 / 0.057 / 2.03 (0.50 / 4.1);
  hip_roll 0.655 / 0.190 / 2.88 (1.31 / 5.8); hip_pitch 0.612 / 0.199 / 6.57 (1.22 / 13.1);
  knee 0.456 / 0.114 / 6.19 (0.91 / 12.4); ankle 0.408 / 0.113 / 3.85 (0.82 / 7.7); no at_torque_limit
  2x: hip_yaw pass, hip_roll T, hip_pitch T+S, knee S, ankle pass (2 of 5)
B: wall time; falls rule fired?; MAC-25 rows; sizing table; 2x pass per joint:
  1258 s, all 1000 it; falls/it 0.27 -> 0.03-0.04 -> no; reward 0.106 -> 0.123
  rows: DR off vx 0.163 / 0.244, |wz| 0.017 / 0.005; DR on 0.142 / 0.216, |wz| 0.009 / 0.001;
  40 ms 0.121 (81 %); 0 falls in every case; turn 0.8 0.873 / 0.771; zero |wz| 0.008 / 0.026 -> MEETS
  sizing: hip_yaw 0.259 / 0.061 / 2.24 (0.52 / 4.5); hip_roll 0.700 / 0.189 / 2.84 (1.40 / 5.7);
  hip_pitch 0.642 / 0.193 / 4.76 (1.28 / 9.5); knee 0.464 / 0.122 / 4.79 (0.93 / 9.6); ankle 0.391 /
  0.104 / 3.67 (0.78 / 7.3); no at_torque_limit
  2x: hip_yaw pass; hip_roll, hip_pitch, knee, ankle fail on speed only (1 of 5); torque passes
  at all five
policy.onnx sha256 A / B: d273bd75a02e6d91a709dfd7a3188d0f23d4c43e965397dfd1c747c733888c4b /
  13070c6af571cccaa7363e73ced8a2c0225d0e0e6082e658622af0e85d763bfb
verdict: with its real curve, B (STS3215 @ 7.4 V) can be learned: the fine-tune walks at 0.15 / 0.25
  with DR on and off, yaw <= 0.017, 81 % at 40 ms, 0 falls, and it turns 0.87. The policy also
  slowed hip pitch and knee from ~6.5 to ~4.8 rad/s p95 under B's curve. But B has the torque and
  not the speed for the 2x margin (speed_p95 2.8-4.8 vs 2.7). A (XC330 @ 6 V) is short on torque at
  the hips and on speed at hip pitch / knee, and its fine-tune drifts at 0.25 and misses 40 ms.
  Neither passes more joints than MAC-25 does under curve A (2 of 5), so MAC-25 stays the reference.
  For WP4-05, the gap is speed at about 5 rad/s with about 1 N.m, at hip roll, hip pitch, knee and ankle.
```

---

## MAC-38 — biped_v1: the STS3215 policy through the Rust runtime (WP4-30, gate G4.4)
Status: `done (mac-m5, 2026-11-20): met` · Deps: MAC-37 done (its `runs/mac37-b/policy.onnx`)

MAC-37 (mac-m5) trained a policy that walks under the STS3215 @ 7.4 V torque-speed
curve. `runtime_sim.py --torque-limit / --motor-no-load-speed` (and `BridgeConfig`) now
apply the same servo model in the sim bridge. The runtime gets the 1.91 N·m limit in
the hello, so its torque budget uses it too. Run the MAC-16 check with that model, from
a clean worktree of origin/main and with absolute paths. Do this on a quiet Mac: VM
load gave deadline misses before.
```bash
(cd runtime && cargo build --locked --release)
P=<mac37>/runs/mac37-b
uv run python scripts/export_mlp.py $P/policy.onnx --out $P/policy.bpml \
    --rust-bin runtime/target/release/mlp-eval > runs/mac38/mlp.json
for v in 0.15 0.25; do
  uv run python scripts/runtime_sim.py --seconds 20 --realtime --env-label mac-m5 \
      --torque-limit 1.91 --motor-no-load-speed 5.45 \
      --log runs/mac38/v$v.bplog --out runs/mac38/v$v.json --require-clean-exit -- \
      --controller policy:$P/policy.bpml --command 0:20:$v:0:0
done
```
Commit `mlp.json` and the two result JSONs under `docs/experiments/mac38/`, without the
logs.

**Acceptance (WP4-30):** for both commands, `fell` false, no latched fault, 1000
periods, `deadline_misses` ≤ 10, `vx_body_mean` ≥ 50 % of the command, and
|`wz`| ≤ 0.08 if the bridge reports it. If the supervisor latches a fault, name it.
Nothing touches `safety.rs`.

Result (env: mac-m5, 2026-11-20): met for both commands
```
revision: 397076e (main); policy = MAC-37 B policy.onnx (sha256 13070c6a...63bfb), policy.bpml
  sha256 a7e090fbf4ac2f391fe9386df9f050f7b435ff101a760fa39f541fb0f1912b70; servo model
  --torque-limit 1.91 --motor-no-load-speed 5.45 (STS3215 @ 7.4 V)
  quiet Mac: the run waited for 2 min of VM < 2 cores, started 00:32Z; VM 0.1-3 % (< 0.03 cores)
  during both runs, load avg 2.3 -> 1.7
mlp.json max error: rust vs onnxruntime 1.19e-6, numpy vs onnxruntime 1.91e-6
v0.15: fell false / no fault (latched 0) / 1000 periods / 0 misses / vx_body_mean 0.163 (109 %) /
  wz +0.022; distance 2.91 m, tilt max 3.0 deg, max torque demand 0.78 N.m, no torque or slew
  clamps; period_max 29.9 ms, compute mean / max 105 / 326 us
v0.25: fell false / no fault / 1000 periods / 0 misses / vx_body_mean 0.248 (99 %) / wz +0.043;
  distance 4.38 m, tilt max 3.4 deg, max torque demand 1.02 N.m, slew_limit events 37, no torque
  clamp; period_max 30.0 ms, compute mean / max 103 / 180 us
  bridge wall_dt (both): mean 20.0 ms, p1 / p50 / p99 13 / 20 / 29 ms
verdict: WP4-30 acceptance met. The STS3215 policy walks through the supervised Rust runtime
  with the servo model at 50 Hz real time: vx >= 99 % of the command, |wz| <= 0.043, no
  falls, no faults, 0 deadline misses. Closer to the command and straighter than MAC-16
  (MAC-14 policy, ideal actuator: 0.194 / 0.285, wz -0.19 / -0.15). safety.rs untouched.
```

---

## MAC-39 — biped_v1: walking on an uneven floor (WP4-31)
Status: `done (mac-m5, 2026-11-20): not met (10 mm: 14/16 falls)` · Deps: MAC-37 done (its `runs/mac37-b/policy.onnx`), MAC-25 done (its `runs/mac25/policy.onnx`)

Every biped_v1 policy so far trained and was scored on a flat plane. A real floor is not
flat. `eval_biped.py --terrain` (WP4-31) swaps the floor for a random height field
(heights uniform in [-h, 0] every 10 cm; same field for a given `--seed`) and runs the
0.15 m/s robustness command at h = 5, 10 and 20 mm, DR off. Score the STS3215 policy
under its curve and, for comparison, the MAC-25 reference at the default 0.52 N·m.
Run from a clean worktree of origin/main with absolute paths, under `caffeinate -i nohup`.
No quiet window is needed (no real-time loop).
```bash
uv run python scripts/eval_biped.py <mac37>/runs/mac37-b/policy.onnx --terrain --robustness \
    --envs 16 --seconds 20 --torque-limit 1.91 --motor-no-load-speed 5.45 \
    --out runs/mac39/sts3215.json
uv run python scripts/eval_biped.py <mac25>/runs/mac25/policy.onnx --terrain \
    --envs 16 --seconds 20 --out runs/mac39/mac25.json
```
Commit both JSONs under `docs/experiments/mac39/` and a short report.

**Acceptance (WP4-31):** for the STS3215 policy, `rough_10mm` has 0 falls of 16 and
`vx_mean` ≥ 50 % of 0.15. Also report `rough_5mm` and `rough_20mm` (falls, vx, |wz|) for
both policies; 20 mm is about 8 % of the leg length and is expected to be hard. If
10 mm fails, say whether the policy falls or only slows down.

Result (env: mac-m5, 2026-11-20): WP4-31 NOT met; both policies fall on a 10 mm rough floor
```
revision: b7bc8d6 (main), git_dirty false; eval only, 16 envs x 20 s, DR off, default seed
sts3215 rough_5/10/20mm: falls / vx_mean / wz_mean:
  5 mm   3/16  0.153  -0.018
  10 mm 14/16  0.114  -0.038   (FAILS: needs 0 falls and vx >= 0.075)
  20 mm 16/16 -0.053  -0.117
mac25 rough_5/10/20mm: falls / vx_mean / wz_mean:
  5 mm   4/16  0.156  -0.041
  10 mm 12/16  0.117  -0.061
  20 mm 16/16 -0.050  -0.184
sts3215 robustness rows (falls, vx; flat floor, 16 envs): nominal 0.15 / 0.25 0 / 0.163, 0 / 0.244;
  friction_0.5 0 / 0.162, friction_1.5 0 / 0.171, push_forward 0 / 0.164, push_sideways 0 / 0.163,
  latency 20 / 40 / 60 ms 0 / 0.141, 0 / 0.121, 0 / 0.106; turn 0.8 wz 0.873; all 0 falls
verdict: not met. On the flat floor the STS3215 policy is clean (0 falls in all 11 cases), but
  it falls on uneven ground: 3/16 at 5 mm, 14/16 at 10 mm, 16/16 at 20 mm. It falls, it does not
  only slow down (the vx_mean of 0.114 at 10 mm comes from the 2 survivors and the walking before
  the falls). MAC-25 behaves the same (4 / 12 / 16 falls), so this is not from the STS3215
  fine-tune: neither policy has seen anything but a flat plane. Next step (cloud): train with the
  height field (terrain curriculum from 0 to 10-20 mm), then re-run this check.
```

---

## MAC-40 — biped_v1: learn to walk on an uneven floor (WP4-32)
Status: `done (mac-m5, 2026-11-20, 696d466): not met (10 mm 13/16 falls); MAC-40b next` · Deps: MAC-39 done, MAC-37 done (its `runs/mac37-b/model_*.pt`, last good checkpoint)

MAC-39 (mac-m5): the STS3215 policy falls on a 10 mm floor in 14 of 16 envs, and MAC-25
in 12 of 16. Both only ever trained on a flat plane. `train_ppo.py --rough-floor H
--rough-floor-ramp-it N` (WP4-32) trains on a height field whose height ramps from flat to
H over the first N iterations of this run. The training field is another seed than the
evaluation's, so the MAC-39 score is on a floor the policy never saw. Fine-tune the
MAC-37 B checkpoint under the same STS3215 curve and settings, with H = 20 mm and the ramp
over 600 of 1500 iterations. Use a clean worktree of origin/main, absolute paths, and
`caffeinate -i nohup`.

First run (mac-m5, 2026-11-20, bea6114) is void: the falls rule fired at it 200 (2.11/it,
6.0/it already in the first 25 it at ~0.4 mm), and model_5100 drifted (0.15 -> vx 0.255,
wz +0.31). Cause (cloud): the height field was only 8 m x 8 m, while a 20 s episode at the
largest command reaches ~7.7 m, so walking envs left the floor and fell. Fixed: 24 m x 24 m
field (`ROUGH_HALF_SIZE` 12, with a test that a whole episode fits). Re-run from the same
MAC-37 B checkpoint with the same command, at a revision that has the fix.

Second run (mac-m5, d6b41fb) is void as well: falls 11 / 5.8 / 4.8 / 3.5 per it over the
first 100 it, at ~0.4-3 mm. The Mac then scored MAC-37 B at 0.15 m/s (16 envs x 20 s,
STS3215 curve): plane 0/16 falls, a 10 um height field 4/16 (DR on: 0 / 4), 2 mm 2/16 (DR on
7/16). Cause (cloud): MuJoCo height-field prisms give box feet sideways contact normals at
their edges even when flat (2.7 % of contacts in a zero-action DR stand; the plane: 0).
Fixed: the uneven floor is now a periodic patch of box tiles (`ROUGH_*`, 10 cm tiles,
2 m period, envs wrapped back by one period), above a lowered plane. Tile seams still give
~2 % sideways contacts at any height (a foot corner catching the next tile, which a real
step also does), so expect falls early in the ramp. Before training, re-run that check on
the tiles (plane / `rough_floor=1e-5` / 2 mm, DR off and on) and report it. Then train with
the falls rule **off until iteration 700** (ramp + 100), on from there. Both void runs also turned
model_5100 into a fast spinning gait (0.15 -> vx 0.29, wz +0.49) within 100 it, so the
command now adds `--desired-kl 0.005` (smaller PPO steps; it steadied MAC-32 / MAC-33).
If falls stay above 3/it after it 300 on the tiles, stop and report instead. The tiles step
about 2x slower than the plane. Score MAC-37 B with `--terrain` too, so the terrain rows
compare on the same floor (MAC-39 used the height field).
```bash
uv run python scripts/train_ppo.py --robot biped_v1 --dr --envs 256 --iters 1500 \
    --workers <best> --seed 0 --track-yaw-std2 0.25 --symmetry loss --mirror-loss-coeff 0.5 \
    --torque-limit 1.91 --motor-no-load-speed 5.45 --rough-floor 0.02 --rough-floor-ramp-it 600 \
    --desired-kl 0.005 \
    --resume <mac37>/runs/mac37-b/<last model_*.pt> --run-dir runs/mac40 \
    --save-every 100 --out runs/mac40/report.json
uv run python scripts/eval_biped.py runs/mac40/policy.onnx --terrain --robustness --dr \
    --envs 16 --seconds 20 --torque-limit 1.91 --motor-no-load-speed 5.45 \
    --out runs/mac40/eval.json
```
Keep the falls rule (stop when the falls pass 1.0 per iteration for 100 iterations, then
score the last good checkpoint). If the final checkpoint fails the flat rows, also score
model_1000 and model_1200 and use the best one that meets them. Commit report/eval JSONs
under `docs/experiments/mac40/`, with the policy.onnx sha256.

**Acceptance (WP4-32 = WP4-31's bar):** `rough_10mm` 0 falls of 16 and vx ≥ 50 % of 0.15,
and the flat MAC-25 rows still hold: 0.15 / 0.25 walking with DR off and on, |wz| ≤ 0.08,
40 ms ≥ 80 % of vx, 0 falls. Also report `rough_5mm` and `rough_20mm`.

Result (env: mac-m5, 2026-11-20, revision 696d466, box tiles): WP4-32 NOT met; the ramp is steeper than the policy adapts
```
revision: 696d466 (main), git_dirty false; from mac37-b/model_5000.pt, the handoff command
  (--rough-floor 0.02 --rough-floor-ramp-it 600 --desired-kl 0.005, STS3215 curve), 8 workers
tile check first (MAC-37 B, 0.15 m/s, 16 envs x 20 s): plane / 1e-5 / 2 mm -> 0 / 0 / 0 falls
  with DR off and on, vx 0.163 / 0.163 / 0.167 (DR on 0.145 / 0.145 / 0.145): tiles = plane
wall time; falls rule fired?; checkpoint used: stopped at it 5300 (12 min) by the "> 3/it after
  it 300" rule: falls/it (25-it means; ramp height) 1.28 (0.4 mm), 0.28, 0.24, 0.16, 0.20 (3.7),
  0.44, 0.56 (5.4), 1.24 (6.2), 2.00 (7.1), 3.48 (7.9), 3.64 (8.7), 4.04 (9.6 mm); mean over it
  200-300 3.29. Scored the last good checkpoint, model_5200 (ONNX sha256
  6777dead84c611a79d101b4ba2c6850c04da7e6f2145d45bd952d8db2d9792c2).
rough_5/10/20mm (tiles; falls / vx_mean / wz_mean; vx over envs still standing):
  model_5200:      0/16 0.198 +0.048 | 13/16 0.158 +0.019 | 16/16 -0.038 -0.310
  MAC-37 B (same tiles): 0/16 0.165 -0.009 | 15/16 0.105 -0.080 | 16/16 -0.065 +0.216
flat rows (model_5200; DR off / on; 16 envs): 0.15 vx 0.196 / 0.170, |wz| 0.048 / 0.016;
  0.25 vx 0.272 / 0.247, |wz| 0.130 / 0.116 (FAIL > 0.08); zero |wz| 0.103 / 0.069; 40 ms 0.146
  (97 %); 0 falls in every case. MAC-37 B on the same eval: 0.25 |wz| 0.006 / 0.006, 40 ms 81 %.
verdict: not met. rough_10mm is 13/16 falls (MAC-37 B 15/16), and the fine-tune breaks the flat
  0.25 yaw row. Falls stay low up to ~5 mm and climb from ~6 mm. The ramp (20 mm over 600 it,
  ~3.3 mm per 100 it) is steeper than the policy adapts. Next: MAC-40b, a slower and lower
  ramp (12 mm over 1500 of 2500 it). The two earlier runs (8 m height field, then height-field
  contacts) are void and kept only as notes above.
```

### MAC-40b — slower, lower ramp (note under MAC-40)
Result (env: mac-m5, 2026-11-20, revision 696d466, box tiles): WP4-32 met by model_6400 (one checkpoint)
```
run: from mac37-b/model_5000.pt, MAC-40 flags with --rough-floor 0.012 --rough-floor-ramp-it 1500
  --iters 2500 (--desired-kl 0.005, STS3215 curve, symmetry loss 0.5, yaw std2 0.25, --dr, 256 envs,
  8 workers); stop rule: > 3/it (100-it mean) anytime, falls rule (> 1.0) from it 1600
wall time; rule fired?: 32 min; falls/it (100-it means; ramp height) 0.60, 0.03, 0.06, 0.09 (2.8 mm),
  0.16, 0.25, 0.40, 0.50 (6.0), 0.64, 0.92, 1.16 (8.4), 1.25, 1.30 (10.0), 1.74, 1.70, 2.01 (12 mm);
  the falls rule fired at it 1600, as soon as it switched on (2.01 > 1.0) -> stopped at model_6600;
  last good = model_6500. Scored 6000 and 6500 as asked, plus 6100-6400 and 6600 as extras
  (same eval; extras marked)
per checkpoint (tiles, 16 envs x 20 s; flat |wz| max over 0.15 / 0.25 DR off / on; 40 ms of 0.15;
  rough 5 / 10 / 20 mm falls, vx at 10 mm; 0 falls on the flat floor everywhere):
  6000         flat |wz| 0.187  40 ms 78 %   0 / 4 / 16   0.133
  6100 extra   flat |wz| 0.133  40 ms 82 %   0 / 5 / 14   0.149
  6200 extra   flat |wz| 0.211  40 ms 84 %   0 / 1 / 14   0.137
  6300 extra   flat |wz| 0.256  40 ms 84 %   0 / 0 / 14   0.129
  6400 extra   flat |wz| 0.028  40 ms 84 %   0 / 0 / 7    0.149   <- meets everything
  6500         flat |wz| 0.325  40 ms 95 %   0 / 0 / 10   0.161
  6600 extra   flat |wz| 0.224  40 ms 101 %  0 / 0 / 5    0.173
best checkpoint: model_6400 (ONNX sha256 2a57e5d925b4e1535093aea9f560dbdfe734b9bfbf998dee53e353acf1cc72ec)
  flat: 0.15 DR off / on vx 0.152 / 0.136, wz -0.028 / +0.016; 0.25 vx 0.235 / 0.214, wz -0.026 /
  +0.028; zero command |wz| 0.016 / 0.049; turn 0.8 0.757 / 0.676; 40 ms 0.126 (84 %); friction,
  pushes, latency 20 / 60 ms all 0 falls
  tiles: 5 mm 0/16 vx 0.156 wz +0.005; 10 mm 0/16 vx 0.149 wz +0.014; 20 mm 7/16 vx 0.052 wz -0.054
  (MAC-37 B on the same tiles: 0 / 15 / 16; model_5200 of MAC-40: 0 / 13 / 16)
verdict: WP4-32 met by model_6400: rough_10mm 0/16 falls at 99 % of 0.15, and every flat MAC-25 row
  holds (|wz| <= 0.028, 40 ms 84 %, 0 falls). The slow ramp works for terrain: from 6300 on, 10 mm
  stays at 0/16, and 20 mm improves to 5-10/16. But the flat yaw swings between checkpoints
  (0.13-0.33 around 6400), so 6400 is one good checkpoint, not a stable region, and the falls rule
  fired when the ramp reached 12 mm. The reference stays MAC-25 / MAC-37 B for the flat floor until
  a run holds both; model_6400 is the terrain candidate. Next: hold the ramp at ~10 mm with the
  falls rule on, or add a flat-floor share and mirror loss 1.0 (as on Microduck MAC-32) to steady
  the yaw.
```

### MAC-40c — stabilise at 10 mm (note under MAC-40)
Result (env: mac-m5, 2026-11-20, code = 696d466, box tiles): NOT met; 2 consecutive passes, 3 needed
```
run: from mac40b/model_6400.pt, --rough-floor 0.010 --rough-floor-ramp-it 0 (10 mm from the start)
  --mirror-loss-coeff 1.0 --iters 1000 --save-every 50, otherwise the MAC-40b flags; 8 workers;
  worktree HEAD 7f998c9 (696d466 + MAC-40 docs)
wall time; rule fired?: falls/it (100-it means) 0.92 / 0.94 / 1.08 -> the falls rule (> 1.0 over 100
  it, on from the start) fired at it 300 (4 min) -> stopped at model_6700; last good = model_6650.
  All six saved checkpoints were scored with the MAC-40b eval (--terrain --robustness --dr, 16 x 20 s).
per checkpoint (flat |wz| max over 0.15 / 0.25 DR off / on; 40 ms of 0.15; flat falls;
  rough 5 / 10 / 20 mm falls; vx at 10 mm):
  6450  0.171  78 %  0   0 / 0 / 8    0.140
  6500  0.215  84 %  0   0 / 0 / 10   0.133
  6550  0.272  93 %  0   0 / 0 / 7    0.159
  6600  0.148  90 %  0   0 / 0 / 3    0.152
  6650  0.045  91 %  0   0 / 0 / 9    0.145   PASS (last good)
  6700  0.042  93 %  0   0 / 0 / 7    0.158   PASS (saved when the rule fired)
  ONNX sha256: 6650 3254104cf1fe4b671b686f32bcdd45b5785740faaca5933c88ba58bd4bc1d15f,
  6700 0881422d097af356194116fcd511f972269b5944adae5c79667e7ca851aae50e
verdict: not met. Only 6650 and 6700 pass (two in a row; three needed), and the run stopped at
  300 of 1000 it. Terrain is now steady: 10 mm stays 0/16 falls on every checkpoint (5 mm 0/16,
  20 mm 3-10/16). What swings is the flat yaw: -0.17 to -0.27 at 6450-6550 (the MAC-40b 6400 gait
  drifted at first), then 0.04 by 6650. The falls rule fires on the training floor itself: at a held
  10 mm the policy falls ~0.9-1.1 per it there, while the 10 mm eval floor gives 0/16. So a 1.0/it rule
  is about the floor's own fall rate. Suggestion: rerun from model_6650 with the falls rule at 1.5/it
  (or measured against the 10 mm training baseline), so the run can show three or more checkpoints in a row.
```

### MAC-40d — continue at 10 mm with a falls rule fitted to the floor (note under MAC-40)
Status: `done (mac-m5, 2026-11-20): not met (2 consecutive passes); flat yaw drifts with 10 mm training`

MAC-40c held the floor at 10 mm and passed at 6650 and 6700 (two in a row), but its 1.0/it falls
rule fired at it 300: on the held 10 mm training floor the policy falls ~0.9-1.1 per iteration,
while the 10 mm eval floor gives 0/16. Continue from the last saved checkpoint with the same
flags and a rule at 1.5/it, so three or more consecutive checkpoints can show.
```bash
uv run python scripts/train_ppo.py --robot biped_v1 --dr --envs 256 --iters 800 --workers 8 \
    --seed 0 --track-yaw-std2 0.25 --symmetry loss --mirror-loss-coeff 1.0 \
    --torque-limit 1.91 --motor-no-load-speed 5.45 --rough-floor 0.010 --rough-floor-ramp-it 0 \
    --desired-kl 0.005 --resume <mac40>/runs/mac40c/model_6700.pt --run-dir runs/mac40d \
    --save-every 50 --out runs/mac40d/report.json
```
Falls rule: stop if the 100-it mean is above 1.5/it, active after 100 it. Score every saved checkpoint
with the MAC-40b eval (`eval_biped --terrain --robustness --dr --envs 16 --seconds 20`, STS3215 curve).

**Acceptance (WP4-32, unchanged):** at least 3 consecutive saved checkpoints (50 apart), each with flat
|wz| <= 0.08, 40 ms >= 80 % of vx, 0 flat falls, and 10 mm 0/16 falls. Name the middle one as the
terrain policy.

Result (env: mac-m5, 2026-11-20, code = 696d466, box tiles): NOT met; 2 consecutive passes, and flat yaw drifts with training
```
revision: code 696d466 (worktree HEAD 7f998c9); from mac40c/model_6700.pt, flags as above, 8 workers
wall time; rule fired?: 653 s (11 min), all 800 it; falls/it (100-it means) 1.29, 0.92, 0.70, 0.87,
  0.88, 0.83, 0.78, 0.90 -> the 1.5/it rule did not fire
per checkpoint (flat |wz| max over 0.15 / 0.25 DR off / on; 40 ms of 0.15; flat falls; 5 / 10 / 20 mm
  falls; vx at 10 mm):
  6750  0.085  86 %  0  0/0/5   0.143      7150  0.225  98 %  0  0/0/2   0.131
  6800  0.013  95 %  0  0/0/7   0.120 P    7200  0.268  91 %  0  0/0/3   0.111
  6850  0.072  92 %  0  0/0/9   0.125 P    7250  0.464  95 %  0  0/0/7   0.149
  6900  0.116  93 %  0  0/0/4   0.148      7300  0.475  96 %  0  0/0/6   0.150
  6950  0.240 103 %  0  0/0/5   0.156      7350  0.336  94 %  0  0/0/7   0.161
  7000  0.383 100 %  0  0/0/10  0.158      7400  0.299  94 %  0  0/0/4   0.159
  7050  0.396 105 %  0  0/0/5   0.173      7450  0.293 104 %  0  0/0/3   0.170
  7100  0.211 104 %  0  0/0/6   0.151      7500  0.096  93 %  0  0/0/4   0.153
  (P = meets all four rows; the worst |wz| is almost always the 0.25 command, positive, DR off)
  ONNX sha256: 6800 6e72d75c4c8335ae4c7e40f97ebc96a9d0cf079923f56d1e835cd87f6df5bf8a,
  6850 abddf75d9a68c739eaa6decd7bc729b0364a41bb0c12983c280ffc9124afbb2a
verdict: not met. Only 6800 and 6850 pass in a row (with MAC-40c: 6650 and 6700 pass, 6750 misses at
  0.085, then 6800 and 6850), so there is no run of three. Terrain is fully steady: 10 mm 0/16 falls and 5 mm
  0/16 on all 16 checkpoints, 20 mm 2-10/16, no flat falls, 40 ms 86-105 %. The flat yaw is what
  fails. It grows steadily with 10 mm training to +0.3...+0.48 rad/s at the 0.25 command (DR off) by
  7000-7450, then dips to 0.096 at 7500. Training only on the 10 mm floor pulls the flat 0.25 gait into a
  turn. Mirror loss 1.0 does not stop it. Next (cloud): mix flat and rough episodes (e.g. 30-50 %
  flat), or yaw-track on the flat floor during terrain training, then repeat with every-50 checkpoints.
  No terrain policy is named; model_6400 / 6800 stay single-checkpoint candidates.
```

---

## MAC-41 — biped_v1: terrain training with flat episodes mixed in (WP4-33)
Status: `done (mac-m5, 2026-11-20): not met (no 2 consecutive passes); flat yaw still swings`

MAC-40c/d learned the 10 mm floor (0/16 falls on every checkpoint) but the flat 0.25 m/s gait drifts
into a turn as 10 mm-only training goes on (|wz| up to 0.48). WP4-33 adds
`--rough-floor-flat-fraction p`: a share p of episodes (drawn per env at each reset) runs on flat
tiles, so the flat gait keeps being trained. Continue from the best MAC-40d checkpoint.

Pre-flight (worktree of origin/main at the WP4-33 merge or later):
`uv run pytest -q tests/unit/test_biped_velocity_env.py -k flat_fraction` passes.
```bash
uv run python scripts/train_ppo.py --robot biped_v1 --dr --envs 256 --iters 800 --workers 8 \
    --seed 0 --track-yaw-std2 0.25 --symmetry loss --mirror-loss-coeff 1.0 \
    --torque-limit 1.91 --motor-no-load-speed 5.45 --rough-floor 0.010 --rough-floor-ramp-it 0 \
    --rough-floor-flat-fraction 0.4 --desired-kl 0.005 \
    --resume <mac40>/runs/mac40d/model_6800.pt --run-dir runs/mac41 \
    --save-every 50 --out runs/mac41/report.json
```
Falls rule: stop if the 100-it mean is above 1.5/it, active after 100 it (with 40 % flat episodes the
training fall rate should sit below MAC-40d's 0.7-0.9). Score every saved checkpoint with the MAC-40b
eval (`eval_biped --terrain --robustness --dr --envs 16 --seconds 20`, STS3215 curve).

**Acceptance (WP4-32/33):** at least 3 consecutive saved checkpoints (50 apart), each with flat
|wz| <= 0.08, 40 ms >= 80 % of vx, 0 flat falls, and 10 mm 0/16 falls. Name the middle one as the
terrain policy and record its ONNX sha256.

Result (env: mac-m5, 2026-11-20, revision 7a1fa16, box tiles): NOT met; isolated passes only
```
pre-flight: pytest -k flat_fraction -> 1 passed
run: from mac40d/model_6800.pt, the command above, 8 workers; 799 s (13 min), all 800 it;
  falls/it (100-it means) 0.63-0.84 (MAC-40d 0.70-1.29) -> the 1.5/it rule did not fire
per checkpoint (flat |wz| max over 0.15 / 0.25 DR off / on; 40 ms of 0.15; flat falls; 5 / 10 / 20 mm
  falls; vx at 10 mm):
  6850  0.113  86 %  0  0/0/5   0.141      7250  0.466  88 %  0  0/0/4   0.151
  6900  0.108 103 %  0  0/0/6   0.170      7300  0.201  98 %  0  0/0/8   0.182
  6950  0.100  92 %  0  0/0/5   0.150      7350  0.092  89 %  0  0/0/4   0.162
  7000  0.078  95 %  0  0/0/10  0.152 P    7400  0.078  94 %  0  0/0/5   0.186 P
  7050  0.143  97 %  0  0/0/7   0.174      7450  0.169  94 %  0  0/0/9   0.179
  7100  0.189  97 %  0  0/0/11  0.170      7500  0.110  98 %  0  0/0/12  0.155
  7150  0.084  95 %  0  0/0/7   0.150      7550  0.202 111 %  1  0/0/8   0.191
  7200  0.268  98 %  1  0/0/7   0.159      7600  0.527  93 %  0  0/0/8   0.145
  (flat falls: 7200 and 7550 each fell once at 0.25 with DR on)
  ONNX sha256: 7000 ec2fa1ee406ddbcf67293a2fdce9aa22b4a2f1a3611b78358393527dee528bc0,
  7400 7dbbb54112872eebdd9290e7dca3a2dbdac7efdb297c5078de90a44863a0a43d
verdict: not met; no terrain policy named. Only 7000 and 7400 pass (0.078 each, at the limit), and no two
  in a row. The flat share lowered training falls and changed the drift from a steady positive turn (MAC-40d)
  into swings of both signs (-0.20 ... +0.53 rad/s, worst at 0.25), but did not make it small.
  Terrain stays steady: 10 mm 0/16 and 5 mm 0/16 on all 16 checkpoints. Four more checkpoints sit just
  above the line (6950 0.100, 7150 0.084, 7350 0.092, 7500 0.110). The flat 0.25 yaw is the open
  problem: its variance between checkpoints is the same size as the bar. Ideas (cloud): stronger
  yaw tracking at 0.25 (e.g. --track-yaw-std2 0.15 for flat episodes), mirror loss 2.0, or averaging
  checkpoints (e.g. the mean of 6950-7050 weights).
```

---

## MAC-42 — biped_v1: average neighbouring MAC-41 checkpoints (WP4-34)
Status: `done (mac-m5, 2026-11-20): met by k=3, centre 6900`

MAC-41 checkpoints swing the flat yaw by about the bar itself from one save to the next (7000 and
7400 pass at 0.078, four more miss at 0.084-0.110). No training here: average the weights of
consecutive MAC-41 checkpoints and score the averages. Worktree of origin/main at the WP4-34 merge or later.

For each window of consecutive saved checkpoints in runs/mac41 (k = 3: centres 6900…7550; k = 5:
centres 6950…7500; checkpoints 50 apart):
```bash
uv run python scripts/average_checkpoints.py <mac41>/model_<c-50>.pt <mac41>/model_<c>.pt \
    <mac41>/model_<c+50>.pt --out runs/mac42/k3_<c>/avg.pt          # k = 5: c-100 … c+100
uv run python scripts/train_ppo.py --robot biped_v1 --envs 8 --iters 0 \
    --torque-limit 1.91 --motor-no-load-speed 5.45 \
    --resume runs/mac42/k3_<c>/avg.pt --run-dir runs/mac42/k3_<c> --out runs/mac42/k3_<c>/export.json
uv run python scripts/eval_biped.py runs/mac42/k3_<c>/policy.onnx --terrain --robustness --dr \
    --envs 16 --seconds 20 --torque-limit 1.91 --motor-no-load-speed 5.45 \
    --out runs/mac42/k3_<c>/eval.json                                # same eval as MAC-40b
```
Then rerun the eval with `--seed 1` for every average that passes on seed 0.

**Acceptance (WP4-34):** one averaged policy meets the WP4-32 rows (flat |wz| <= 0.08, 40 ms >= 80 %
of vx, 0 flat falls, 10 mm 0/16) on eval seeds 0 and 1, and the same-k windows centred 50 it either
side also meet them on seed 0. Name it the terrain policy candidate (k, centre, ONNX sha256).
Report a table of all windows (k, centre, flat |wz| max, 40 ms, flat falls, 5/10/20 mm falls).

Result (env: mac-m5, 2026-11-20, revision d7b1ef9): WP4-34 met: terrain policy candidate = k=3 average centred on 6900
```
run: eval only; 14 k=3 and 12 k=5 windows of runs/mac41 checkpoints averaged with average_checkpoints.py,
  exported with train_ppo --iters 0 (the handoff flags were enough), scored with the MAC-40b eval (seed 0);
  passing ones re-scored with --seed 1. Extra: k=3 centred on 6850 (model_6800 of mac40d, the checkpoint
  MAC-41 resumed from, + mac41 6850 + 6900) to give 6900 its -50 neighbour.
table (k, centre; flat |wz| max over 0.15 / 0.25 DR off / on; 40 ms of 0.15; flat falls; 5 / 10 / 20 mm falls;
  P = meets the WP4-32 rows on seed 0):
  k3 6850 (extra)  0.027   95 %  0  0/0/9   P  
  k3 6900  0.063   93 %  0  0/0/8   P  seed 1: 0.068 / 93 % / 0 / 0 -> pass
  k3 6950  0.070   97 %  0  0/0/7   P  seed 1: 0.078 / 97 % / 1 / 0 -> FAIL
  k3 7000  0.097   96 %  0  0/0/6      
  k3 7050  0.135   97 %  0  0/0/8      
  k3 7100  0.122   97 %  0  0/0/7      
  k3 7150  0.140   97 %  0  0/0/8      
  k3 7200  0.199   94 %  0  0/0/8      
  k3 7250  0.133   94 %  0  0/0/4      
  k3 7300  0.110   91 %  0  0/0/9      
  k3 7350  0.110   94 %  0  0/0/7      
  k3 7400  0.096   93 %  0  0/0/9      
  k3 7450  0.068   94 %  0  0/0/8   P  seed 1: 0.078 / 94 % / 0 / 0 -> pass
  k3 7500  0.112  101 %  1  0/0/7      
  k3 7550  0.224  101 %  0  0/0/10     
  k5 6950  0.056   93 %  0  0/0/6   P  seed 1: 0.066 / 93 % / 0 / 0 -> pass
  k5 7000  0.120   97 %  0  0/0/9      
  k5 7050  0.095   96 %  0  0/0/7      
  k5 7100  0.136   97 %  0  0/0/7      
  k5 7150  0.166   96 %  0  0/0/8      
  k5 7200  0.106   96 %  0  0/0/8      
  k5 7250  0.063   93 %  0  0/0/9   P  seed 1: 0.049 / 93 % / 0 / 0 -> pass
  k5 7300  0.108   93 %  0  0/0/5      
  k5 7350  0.090   93 %  0  0/0/7      
  k5 7400  0.109   94 %  0  0/0/10     
  k5 7450  0.085   97 %  1  0/0/2      
  k5 7500  0.177   98 %  0  0/0/9      
verdict: met. k=3 centred on 6900 (models 6850 + 6900 + 6950 of MAC-41) meets the WP4-32 rows on seed 0
  (|wz| 0.063, 40 ms 93 %, 0 flat falls, 10 mm 0/16) and seed 1 (0.068, 93 %, 0, 0/16), and its same-k
  neighbours k=3 6850 (0.027, extra) and k=3 6950 (0.070) pass on seed 0: three passing windows in a row.
  Seed 0 rows: 0.15 / 0.25 DR off vx 0.164 / 0.238, wz -0.063 / -0.005; DR on vx 0.148 / 0.224, wz -0.028 /
  -0.009; zero |wz| 0.051; turn 0.8 0.491; 40 ms 0.139; 10 mm vx 0.149. Terrain policy candidate:
  k=3, centre 6900, ONNX sha256 2e94457133cef85e234858e483a1a2dbe1c9f83d223ac306f032aff023807db3.
  Other seed-0 passes: k=3 6950 (seed 1 fails: one flat fall), k=3 7450, k=5 6950 and k=5 7250 (both seeds
  pass, but a same-k neighbour fails). Averaging narrows the swing (k=3 |wz| 0.063-0.224 against 0.078-0.527
  single), but it does not remove it. Caveat: the -50 neighbour includes the run's start checkpoint.
```

---

## MAC-43 — biped_v1: the terrain policy through the Rust runtime (WP4-35)
Status: `done (mac-m5, 2026-11-20): NOT met (0.25 |wz| 0.132 through the runtime)` · Deps: MAC-42 done (its `runs/mac42/k3_6900/policy.onnx`, sha256 2e944571…)

Repeat the MAC-38 check with the WP4-34 terrain policy, on a quiet Mac, from a clean worktree of
origin/main and with absolute paths. The sim bridge floor is flat; this checks that the averaged
policy runs in real time in the runtime, not the terrain itself.
```bash
(cd runtime && cargo build --locked --release)
P=<mac42>/runs/mac42/k3_6900
shasum -a 256 $P/policy.onnx      # must start 2e944571
uv run python scripts/export_mlp.py $P/policy.onnx --out $P/policy.bpml \
    --rust-bin runtime/target/release/mlp-eval > runs/mac43/mlp.json
for v in 0.15 0.25; do
  uv run python scripts/runtime_sim.py --seconds 20 --realtime --env-label mac-m5 \
      --torque-limit 1.91 --motor-no-load-speed 5.45 \
      --log runs/mac43/v$v.bplog --out runs/mac43/v$v.json --require-clean-exit -- \
      --controller policy:$P/policy.bpml --command 0:20:$v:0:0
done
```
Commit `mlp.json` and the two result JSONs under `docs/experiments/mac43/`, without the logs.

**Acceptance (WP4-35, MAC-38 rows):** for both commands, `fell` false, no latched fault, 1000
periods, `deadline_misses` <= 10, `vx_body_mean` >= 50 % of the command, and |`wz`| <= 0.08 if the
bridge reports it. If the supervisor latches a fault, name it. Nothing touches `safety.rs`.

Result (env: mac-m5, 2026-11-20, revision 5ad58d1): NOT met; the 0.25 m/s run yaws +0.132 rad/s through the runtime
```
policy: runs/mac42/k3_6900/policy.onnx sha256 2e94457133cef85e234858e483a1a2dbe1c9f83d223ac306f032aff023807db3
  (checked), policy.bpml sha256 f91559700e24878b6cc0937adf46a64398e6bc1defef3b186f1042abae0466f0;
  mlp.json: rust vs onnxruntime 1.22e-6, numpy vs onnxruntime 1.67e-6
tries (VM load sampled every 1.5-2 s during the 2 x 20 s runs; % = one core):
  noisy1 (09:32Z)  VM max 668 % / mean 439 %   (not quiet: waited 2 min < 2 cores, load came back)
  try1   (10:27Z)  VM max 243 % / mean 152 %   (not quiet)
  try2   (13:18Z)  VM max  22 % / mean   2 %   FULLY QUIET (best try)
best try (try2):
  v0.15: fell false / latched 0 (no fault) / 1000 periods / 0 misses / vx_body_mean 0.172 (115 %) /
    wz +0.079; period_max 25.0 ms, max torque demand 0.87 N.m, distance 2.86 m
  v0.25: fell false / latched 0 / 1000 periods / 0 misses / vx_body_mean 0.253 (101 %) / wz +0.132
    (FAILS |wz| <= 0.08); period_max 24.9 ms, slew_limit events 61, max torque 1.04 N.m, distance 3.83 m
  noisy tries: identical vx / wz / torque in all three (the sim is deterministic); only timing moved
  (noisy1 0.25: 5 misses, period_max 141 ms; try1: 0 misses, 25 ms)
verdict: not met. Safety and timing are clean on a quiet Mac (no fall, no fault, 0 misses), but at 0.25
  m/s the averaged terrain policy turns +0.132 rad/s through the runtime, and 0.15 is just inside (+0.079).
  Load is not the cause: all three tries give the same wz. The Python env scored this policy at 0.25 with
  wz -0.005 (DR off, mean over 16 envs with reset noise). The runtime runs one env from a fixed start,
  so a single deterministic trajectory picks one side of the yaw spread that MAC-41/42 saw between
  checkpoints. safety.rs untouched. Next (cloud): score the runtime at 0.25 over several start seeds,
  or check the k=3 7450 / k=5 6950 / k=5 7250 averages the same way.
```

---

## MAC-44 — biped_v1: score the MAC-42 averages through the runtime from several starts (WP4-36)
Status: `seed part met by k3_7450 (mac-m5, 2026-11-20); realtime part superseded by MAC-46 (the runtime observation changed, WP4-38)` · Deps: MAC-42 averages in `runs/mac42/`, MAC-43 runtime build

MAC-43 ran the k3_6900 average from the exact keyframe only: one deterministic trajectory, wz
+0.132 at 0.25 m/s, while the Python eval over 16 noisy resets gives -0.005. `runtime_sim.py
--init-seed N` now starts from a perturbed keyframe (the env's reset ranges). No `--realtime` here
(sim results do not depend on load; it runs faster than real time), so the VM does not matter.

Candidates (all passed in Python): k3_6900, k3_7450, k5_6950, k5_7250. For each:
```bash
(cd runtime && cargo build --locked --release)
P=<mac42>/runs/mac42/<cand>
uv run python scripts/export_mlp.py $P/policy.onnx --out $P/policy.bpml \
    --rust-bin runtime/target/release/mlp-eval > runs/mac44/<cand>/mlp.json
for v in 0.15 0.25; do for s in 0 1 2 3 4 5 6 7; do
  uv run python scripts/runtime_sim.py --seconds 20 --env-label mac-m5 --init-seed $s \
      --torque-limit 1.91 --motor-no-load-speed 5.45 \
      --out runs/mac44/<cand>/v${v}_s$s.json --require-clean-exit -- \
      --controller policy:$P/policy.bpml --command 0:20:$v:0:0
done; done
```
Report per candidate and command: falls, latched faults, mean and min/max of `vx_body_mean` and
`wz_mean` over the 8 starts. Commit a summary JSON per candidate under `docs/experiments/mac44/`.

**Acceptance (WP4-36):** one candidate with no fall and no latched fault on any start, |mean `wz_mean`
over the 8 starts| <= 0.08 and mean `vx_body_mean` >= 50 % of the command, at both 0.15 and 0.25.
If several pass, pick the smallest worst-case |wz|. Then, on a quiet Mac (MAC-43 wait script), one
`--realtime` run of it from `--init-seed 0` at both commands: MAC-38 rows (no fall, no fault, 1000
periods, <= 10 misses, vx >= 50 %). Name it (candidate, ONNX sha256).

Result (env: mac-m5, 2026-11-20, revision 32a0ac7): seed part MET by k3_7450; realtime part PENDING
```
run: runtime_sim.py (no --realtime) --seconds 20 --init-seed 0..7 --torque-limit 1.91 --motor-no-load-speed 5.45,
  0.15 and 0.25 m/s, 4 candidates x 2 commands x 8 starts = 64 runs; every run: no fall, no latched fault,
  1000 periods
per candidate (over the 8 starts: mean vx (of cmd) [min..max]; mean wz [min..max]):
  k3_6900  0.15: 0.173 (115 %) [0.173..0.174], wz +0.076 [+0.069..+0.086]
           0.25: 0.252 (101 %) [0.250..0.253], wz +0.088 [+0.072..+0.100]   -> fails (|mean wz| 0.088)
  k3_7450  0.15: 0.192 (128 %) [0.190..0.193], wz +0.021 [-0.022..+0.045]
           0.25: 0.264 (105 %) [0.259..0.267], wz +0.051 [+0.013..+0.078]   -> PASSES, worst |wz| 0.078
  k5_6950  0.15: 0.178 (119 %) [0.178..0.178], wz +0.089 [+0.074..+0.103]
           0.25: 0.261 (104 %) [0.259..0.263], wz +0.106 [+0.078..+0.123]   -> fails
  k5_7250  0.15: 0.185 (124 %) [0.185..0.186], wz +0.084 [+0.059..+0.110]
           0.25: 0.266 (106 %) [0.265..0.268], wz +0.224 [+0.203..+0.252]   -> fails
  rust vs onnxruntime <= 1.3e-6 for all four exports
chosen: k3_7450 (MAC-41 models 7400 + 7450 + 7500), ONNX sha256
  959ab77b081d202e87e767ff12feb73a3547812285c7d8c1e9e682776c36bce0
caveat: k3_7450 is not a stable window like k3_6900. Its own k=3 neighbours failed the Python bar in
  MAC-42 (k3_7400 |wz| 0.096, k3_7500 0.112 with one flat fall), while k3_6900 passed there with both
  neighbours but yaws +0.088 here through the runtime. Every candidate yaws positive through the
  runtime at both commands.
realtime (k3_7450, --init-seed 0, both commands, quiet Mac): pending. The wait script (3 min of VM
  < 2 cores, then the runs with VM sampled; repeat if any sample >= 2 cores) is running.
```

---

## MAC-45 — biped_v1: does the runtime's clean observation explain the +wz? (WP4-37)
Status: `done (mac-m5, 2026-11-23): yes, the observation mismatch (mostly the joint-velocity lag) explains the +wz` · Deps: MAC-42 averages in `runs/mac42/`; can run while the MAC-44 realtime part waits

Every MAC-44 candidate yaws positive through the runtime (+0.02..+0.22), while the Python eval gives
|wz| <= 0.07 (k3_6900 -0.005 over 16 resets). The model is identical (cloud check, 2026-11-20). One
known difference: in training and in `eval_biped.py` the actor sees noise, a gyro/gravity lag and
**one-period-old joint velocities**; the runtime sends clean, current values. `eval_biped.py
--obs-mode` now switches this: `train` (default), `no-vel-lag` (noise on, current joint velocities),
`runtime` (the runtime's clean observation). No `--realtime`, so VM load does not matter.

For each of k3_6900, k3_7450, k5_6950, k5_7250 and each mode:
```bash
P=<mac42>/runs/mac42/<cand>
for mode in train no-vel-lag runtime; do
  uv run python scripts/eval_biped.py $P/policy.onnx --envs 16 --seconds 20 \
      --torque-limit 1.91 --motor-no-load-speed 5.45 --obs-mode $mode \
      --out runs/mac45/<cand>/$mode.json
done
```
Report per candidate, mode and command (0.15, 0.25): falls, `vx_mean`, `wz_mean`. Commit a summary JSON
under `docs/experiments/mac45/`.

**Reading (WP4-37):** if `runtime` mode gives a positive wz near the MAC-44 values (within ~0.04 for
most candidates) while `train` stays near 0, the observation mismatch is the cause; `no-vel-lag` says
whether the joint-velocity lag alone is enough. If all modes stay near 0, the cause is elsewhere
(start pose or something in the runtime path) and the cloud goes on comparing.

Result (env: mac-m5, 2026-11-23, revision aed9352): the runtime's clean observation explains the +wz
```
run: eval_biped.py --obs-mode train / no-vel-lag / runtime, --envs 16 --seconds 20, STS3215 curve,
  nominal rows DR off; 0 falls in every case; no --realtime. "MAC-44" = runtime_sim mean over 8 starts
per candidate and command: vx / wz for train | no-vel-lag | runtime | MAC-44 runtime wz
  k3_6900 0.15  0.164 / -0.063 | 0.172 / +0.057 | 0.173 / +0.065 | +0.076
          0.25  0.238 / -0.005 | 0.251 / +0.080 | 0.252 / +0.092 | +0.088
  k3_7450 0.15  0.182 / -0.068 | 0.191 / +0.022 | 0.192 / +0.006 | +0.021
          0.25  0.257 / +0.031 | 0.261 / +0.052 | 0.263 / +0.053 | +0.051
  k5_6950 0.15  0.169 / -0.027 | 0.177 / +0.071 | 0.179 / +0.065 | +0.089
          0.25  0.248 / +0.019 | 0.260 / +0.091 | 0.261 / +0.104 | +0.106
  k5_7250 0.15  0.176 / +0.009 | 0.187 / +0.056 | 0.186 / +0.076 | +0.084
          0.25  0.258 / +0.063 | 0.264 / +0.245 | 0.265 / +0.235 | +0.224
reading: runtime mode matches the MAC-44 runtime wz within 0.024 in all 8 rows (6 of 8 within 0.015),
  while train mode stays near 0 or negative. no-vel-lag alone gives most of the shift (within 0.02 of
  runtime mode in 7 of 8 rows), so the one-period-old joint velocities in training are the main cause;
  removing noise and the gyro/gravity lag adds the rest. vx rises ~5 % in the same way. The runtime path
  itself is faithful: the Python env with the runtime's observation reproduces it. Next (cloud): make
  training and runtime see the same observation, either feed the runtime the same lagged joint
  velocities (and noise-free equivalents) or train without the lag, then re-score with --obs-mode runtime.
```

---

## MAC-46 — biped_v1: re-score the MAC-42 averages with the runtime's lagged joint velocities (WP4-38)
Status: `done (mac-m5, 2026-11-23): met (seed part + realtime row, realtime not quiet: VM ~7 cores)` · Deps: MAC-42 averages in `runs/mac42/`; replaces the MAC-44 realtime part

MAC-45 showed the runtime's +wz comes from feeding current joint velocities to a policy trained on
one-period-old ones. The runtime now feeds the previous period's joint velocities by default
(`--no-joint-vel-lag` restores the old behaviour). Rebuild the runtime from a clean origin/main worktree;
no `--realtime` until the last step, so VM load does not matter.

For each of k3_6900, k3_7450, k5_6950, k5_7250 (reuse the MAC-44 `policy.bpml` exports):
```bash
(cd runtime && cargo build --locked --release)
P=<mac42>/runs/mac42/<cand>
for v in 0.15 0.25; do for s in 0 1 2 3 4 5 6 7; do
  uv run python scripts/runtime_sim.py --seconds 20 --env-label mac-m5 --init-seed $s \
      --torque-limit 1.91 --motor-no-load-speed 5.45 \
      --out runs/mac46/<cand>/v${v}_s$s.json --require-clean-exit -- \
      --controller policy:$P/policy.bpml --command 0:20:$v:0:0
done; done
uv run python scripts/eval_biped.py $P/policy.onnx --envs 16 --seconds 20 \
    --torque-limit 1.91 --motor-no-load-speed 5.45 --obs-mode runtime --out runs/mac46/<cand>/eval_runtime.json
```
Report per candidate and command: falls, latched faults, mean and min/max of `vx_body_mean` and
`wz_mean` over the 8 starts, next to the MAC-44 numbers and the new `--obs-mode runtime` eval. Commit a
summary JSON under `docs/experiments/mac46/`.

**Acceptance (WP4-38):** as WP4-36, one candidate with no fall and no latched fault on any start,
|mean `wz_mean`| <= 0.08 and mean `vx_body_mean` >= 50 % of the command, at both 0.15 and 0.25; pick
the smallest worst-case |wz|, preferring k3_6900 (its neighbours also passed MAC-42). Then, on a quiet
Mac (MAC-43 wait script, at most 4 h), one `--realtime` run of it from `--init-seed 0` at both commands:
MAC-38 rows (no fall, no fault, 1000 periods, <= 10 misses, vx >= 50 %). Name it (candidate, ONNX sha256).
If the Mac never gets quiet, report the seed part and leave the realtime row pending.

Result (env: mac-m5, 2026-11-23, revision 1ded258 / realtime 3104543): MET, chosen k3_6900
```
run: runtime rebuilt at 1ded258 (lagged joint velocities by default), runtime_sim.py --init-seed 0..7, no
  --realtime, STS3215 curve, MAC-44 policy.bpml exports; plus eval_biped --obs-mode runtime (16 envs x 20 s).
  64 runs: no fall, no latched fault, 1000 periods each.
per candidate and command (8 starts: mean vx (of cmd) [min..max]; mean wz [min..max]; MAC-44 mean wz;
  eval --obs-mode runtime vx / wz):
  k3_6900 0.15  0.163 (109 %) [0.163..0.164]  -0.044 [-0.056..-0.029]  MAC-44 +0.076  eval 0.164 / -0.063
          0.25  0.239 ( 96 %) [0.237..0.240]  +0.009 [-0.009..+0.031]  MAC-44 +0.088  eval 0.238 / -0.002
          -> PASS, worst |wz| 0.056
  k3_7450 0.15  0.181 (121 %) [0.180..0.182]  -0.074 [-0.081..-0.063]  MAC-44 +0.021  eval 0.182 / -0.080
          0.25  0.260 (104 %) [0.258..0.262]  +0.045 [+0.008..+0.093]  MAC-44 +0.051  eval 0.259 / +0.035
          -> PASS, worst |wz| 0.093
  k5_6950 0.15  0.168 (112 %) [0.167..0.170]  -0.013 [-0.023..-0.004]  MAC-44 +0.089  eval 0.169 / -0.028
          0.25  0.247 ( 99 %) [0.245..0.250]  +0.047 [+0.013..+0.073]  MAC-44 +0.106  eval 0.248 / +0.045
          -> PASS, worst |wz| 0.073
  k5_7250 0.15  0.174 (116 %) [0.173..0.176]  +0.037 [+0.011..+0.057]  MAC-44 +0.084  eval 0.174 / +0.033
          0.25  0.257 (103 %) [0.256..0.259]  +0.104 [+0.075..+0.125]  MAC-44 +0.224  eval 0.257 / +0.103
          -> fails (|mean wz| 0.104 at 0.25)
chosen: k3_6900 (smallest worst-case |wz| 0.056, and its neighbours passed MAC-42), ONNX sha256
  2e94457133cef85e234858e483a1a2dbe1c9f83d223ac306f032aff023807db3
reading: with lagged joint velocities the runtime matches the Python env again: runtime mean wz is within
  ~0.02 of eval --obs-mode runtime (and of the MAC-42 train-mode eval) for every candidate and command, and
  the positive bias of MAC-44 is gone (k3_6900 at 0.25: +0.088 -> +0.009). WP4-38 fixes the mismatch.
realtime (k3_6900, --init-seed 0, both commands; NOT QUIET: VM ~7 cores): run at 08:48Z on the owner's
  request, after the quiet-window wait (since 06:35Z) was stopped; runtime rebuilt at 3104543 (main),
  policy.bpml sha256 f91559700e24878b6cc0937adf46a64398e6bc1defef3b186f1042abae0466f0
  v0.15: fell false / no fault (latched 0) / 1000 periods / 0 misses / vx_body_mean 0.164 (109 %) /
    wz -0.041; period_max 28.5 ms, compute mean / max 63 / 1504 us, max torque 0.84 N.m;
    load: not quiet, VM ~7.3 cores (max 796 %), host load1 up to 18.6
  v0.25: fell false / no fault / 1000 periods / 0 misses / vx_body_mean 0.238 (95 %) / wz +0.031;
    period_max 25.0 ms, compute mean / max 64 / 572 us, max torque 1.10 N.m, slew_limit 62;
    load: not quiet, VM ~7.1 cores (max 803 %), host load1 up to 16.4
  -> all MAC-38 rows pass, even under load (no deadline-miss failure to attribute)
verdict: WP4-38 met. Terrain candidate k3_6900 (ONNX sha256 2e94457133cef85e234858e483a1a2dbe1c9f83d223ac306f032aff023807db3)
  walks through the supervised Rust runtime in real time at 0.15 and 0.25 with |wz| <= 0.041, no fall, no fault,
  0 misses; its 8-start seed results agree (mean wz -0.044 / +0.009).
```

---

## MAC-47 — biped_ds3218: learn to walk with the owner's low-budget servos (WP4-40, G4.5)
Status: `done (mac-m5, 2026-11-23): met by run A (fine-tune); run B not needed` · Deps: WP4-39 on main; MAC-37 B checkpoint (`runs/mac37-b/model_<newest>.pt`)

The owner picked ten DSServo DS3218MG PWM hobby servos (G4.5, 2026-11-23). Two things change from
biped_v1: the robot is ~1.04 kg instead of 0.51 kg (`simulation/custom-biped/biped_ds3218.xml`, same
joints and stand pose), and the servos report nothing back, so the actor must use the targets it sent
(`--joint-feedback commanded`) instead of measured joint angles and speeds. Servo at 6 V: 1.96 N·m stall,
7.0 rad/s no-load (seller datasheet; gains and friction in the model are guesses until bench ID).

Two runs, same flags, `--workers <best>` as in MAC-37:
```bash
F="--robot biped_v1 --model biped_ds3218 --joint-feedback commanded --dr --envs 256 --seed 0 \
   --track-yaw-std2 0.25 --symmetry loss --mirror-loss-coeff 0.5 --torque-limit 1.96 --motor-no-load-speed 7.0"
# A: fine-tune the MAC-37 STS3215 policy (warm start; its joint inputs change meaning, the rest carries over)
uv run python scripts/train_ppo.py $F --iters 1500 --workers <best> \
    --resume <mac37>/runs/mac37-b/model_<newest>.pt --run-dir runs/mac47-a --save-every 100 \
    --out runs/mac47-a/report.json
# B: from scratch, only if A does not walk (same 4000 iterations as MAC-25)
uv run python scripts/train_ppo.py $F --iters 4000 --workers <best> --run-dir runs/mac47-b \
    --save-every 100 --out runs/mac47-b/report.json
E="--model biped_ds3218 --joint-feedback commanded --torque-limit 1.96 --motor-no-load-speed 7.0"
uv run python scripts/eval_biped.py runs/mac47-<a|b>/policy.onnx $E --dr --robustness --envs 8 \
    --seconds 20 --out runs/mac47-<a|b>/eval.json
uv run python scripts/eval_biped.py runs/mac47-<a|b>/policy.onnx $E --obs-mode runtime --envs 16 \
    --seconds 20 --out runs/mac47-<a|b>/eval_runtime.json
```
Report per run: the MAC-25 rows (falls, vx at 0.15 / 0.25 against the command, wz at 0.8, 40 ms
latency row), the `sizing` block (max torque and p95 speed per joint against 1.96 N·m / 7.0 rad/s, and
`at_torque_limit`), and the slew-limited fraction. Commit a summary JSON under `docs/experiments/mac47/`
(no ONNX, no runs/).

**Acceptance (WP4-40):** one run with no fall in the nominal and DR cases, vx >= 50 % of the command at
0.15 and 0.25, and no joint whose worst walking max torque is above 1.96 / 1.5 = 1.3 N·m (some margin left
for the heavier real build). If neither run walks, report where it stops (stands only, falls, torque
clip) so the cloud can adjust the model or the gait range; that is a valid result, not a failure of the task.

Result (env: mac-m5, 2026-11-23, revision 53c919f): WP4-40 MET by run A; run B not run
```
run A: train_ppo.py with the flags above, --iters 1500 --workers 8, --resume mac37-b/model_5000.pt;
  3079 s (51 min); reward/step 0.092 -> 0.126 (first -> last quarter); training falls/it (150-it means)
  4.07 (heavier robot, commanded feedback) -> 0.17 -> 0.00-0.03 from it 300 on
MAC-25 rows (eval --dr --robustness, 8 envs x 20 s; 0 falls in every case):
  0.15 / 0.25 DR off: vx 0.169 (113 %) / 0.247 (99 %), wz +0.008 / +0.036
  0.15 / 0.25 DR on:  vx 0.144 (96 %) / 0.216 (86 %), wz -0.010 / +0.006
  zero command: wz +0.048 / +0.028 (vx 0.029 / 0.024); turn 0.8: wz 1.048 / 0.884
  friction 0.5 / 1.5: vx 0.165 / 0.172; pushes fwd / side: 0.169 / 0.168, 0 falls
  latency 20 / 40 / 60 ms: vx 0.142 / 0.122 / 0.105 (95 / 81 / 70 % of 0.15)
  --obs-mode runtime (16 envs): 0.15 / 0.25 vx 0.170 / 0.250, wz +0.006 / +0.050; turn 0.8 wz 1.049; 0 falls
sizing (walking max torque / RMS / p95 speed; servo 1.96 N.m stall, 7.0 rad/s no-load; bar 1.3 N.m):
  hip_yaw      0.435 / 0.114 / 2.13
  hip_roll     1.126 / 0.358 / 2.39   (highest torque, 57 % of stall)
  hip_pitch    1.018 / 0.375 / 5.05   (p95 speed 72 % of no-load)
  knee         0.826 / 0.276 / 5.15   (p95 speed 74 % of no-load)
  ankle_pitch  0.763 / 0.226 / 4.23
  at_torque_limit false everywhere; all five <= 1.3 N.m
slew-limited fraction: 0.000 at 0.15, 0.027 / 0.013 at 0.25 (DR off / on), <= 0.001 elsewhere
policy.onnx sha256: d9a1d6ae1c82b4fc3dab53e4d9f2f5dd07d2231788f3867fc2bc3f100e4bdb5c
verdict: met. The fine-tuned policy walks the heavier DS3218MG build with commanded-only joint feedback:
  no fall in any nominal, DR or robustness case, vx >= 86 % of the command at 0.15 / 0.25, and every
  joint's worst walking torque <= 1.13 N.m (bar 1.3 N.m). The tighter margin is speed: hip pitch and
  knee run at p95 ~5.1 rad/s against 7.0 rad/s no-load, so torque at speed should be checked on the bench
  (the model's gains and friction are guesses until servo ID). No hardware was powered.
```

---

## MAC-48 — biped_ds3218: the DS3218MG policy through the Rust runtime (WP5-06 / WP4-40)
Status: `done (mac-m5, 2026-11-23): met (seed part + realtime row, realtime not quiet: VM ~7 cores)` · Deps: MAC-47 run A (`runs/mac47-a/policy.onnx`), WP5-06 on main (`--joint-feedback commanded` in biped-runtime, `runtime_sim.py --model biped_ds3218`)

Run the MAC-47 run-A policy through the supervised Rust runtime with the DS3218MG build and commanded-only joint
feedback, from a clean worktree of origin/main (bc59934 or later) with absolute paths. Seed part first (no
`--realtime`, so VM load does not matter), then one `--realtime` run from `--init-seed 0`.
```bash
(cd runtime && cargo build --locked --release)
P=<mac47>/runs/mac47-a
uv run python scripts/export_mlp.py $P/policy.onnx --out $P/policy.bpml \
    --rust-bin runtime/target/release/mlp-eval > runs/mac48/mlp.json
for v in 0.15 0.25; do for s in 0 1 2 3 4 5 6 7; do
  uv run python scripts/runtime_sim.py --model biped_ds3218 --seconds 20 --env-label mac-m5 --init-seed $s \
      --torque-limit 1.96 --motor-no-load-speed 7.0 --out runs/mac48/v${v}_s$s.json --require-clean-exit -- \
      --controller policy:$P/policy.bpml --joint-feedback commanded --command 0:20:$v:0:0
done; done
# if it passes: the same command with --realtime --init-seed 0 at both speeds (MAC-38 rows)
```
**Acceptance (as WP4-36/38):** no fall and no latched fault on any start; |mean `wz_mean`| <= 0.08 and mean
`vx_body_mean` >= 50 % of the command at 0.15 and 0.25. Then the realtime run: MAC-38 rows (no fall, no fault,
1000 periods, <= 10 misses, vx >= 50 %), with the VM / host load noted.

Result (env: mac-m5, 2026-11-23, revision bc59934): MET
```
policy: MAC-47 run A, ONNX sha256 d9a1d6ae1c82b4fc3dab53e4d9f2f5dd07d2231788f3867fc2bc3f100e4bdb5c,
  policy.bpml sha256 08be6b7116315e2fe2dcbdfb2cde65c12dd3da92144fd73c50f820abec52dcb1 (39 -> 10);
  rust vs onnxruntime 1.3e-6
seed part (8 starts, no --realtime; 16 runs: no fall, no latched fault, 1000 periods each):
  0.15: vx 0.170 (113 %) [0.170..0.170], wz +0.003 [+0.000..+0.007], max torque demand 1.48 N.m
  0.25: vx 0.250 (100 %) [0.250..0.251], wz +0.044 [+0.041..+0.046], max torque demand 2.03 N.m;
        slew_limit 27-32 events per run, torque_clamp 0, position_clamp 0
realtime (--init-seed 0; NOT QUIET: VM ~7.1-7.2 cores, host load1 up to ~106):
  v0.15: fell false / latched 0 / 1000 periods / 0 misses / vx 0.170 (113 %) / wz +0.000; period_max 30.5 ms
  v0.25: fell false / latched 0 / 1000 periods / 0 misses / vx 0.250 (100 %) / wz +0.045; period_max 30.0 ms,
         slew_limit 27
verdict: met. The DS3218MG policy walks through the runtime with commanded-only feedback, from every start and in
  real time, |mean wz| <= 0.044, matching the Python eval (MAC-47: vx 0.169 / 0.247, runtime-obs wz +0.006 /
  +0.050). Note: at 0.25 the bridge's max_torque_demand peaks at ~2.0 N.m, slightly above the 1.96 N.m stall
  (a demand, capped by the servo curve; the runtime's torque_clamp stayed 0). The Python sizing's walking max was
  1.13 N.m (a different measure: applied actuator force over the walking cases). Worth a look before hardware.
  No hardware powered.
```

---

## MAC-49 — biped_ds3218: does the gait survive cheap-servo imperfections? (WP4-41)
Status: `done (mac-m5, 2026-11-24): met at step 1 (MAC-47 run A survives --servo-dr); step 2 not needed` · Deps: WP4-41 on main (`--servo-dr`), MAC-47 run A (`runs/mac47-a/policy.onnx`, `model_<newest>.pt`)

A PWM hobby servo is not the clean position actuator of the model. `--servo-dr` draws per reset and per joint
a residual centre offset (±0.03 rad), a deadband (0–0.008 rad: targets closer than this to the held one are
ignored) and a stiffness scale (0.7–1.3, since the DS3218MG gain is a guess until the bench). The owner
postponed buying parts (2026-11-23), so this is the next useful check, on the Mac only.

Step 1 (eval only, ~minutes): the MAC-47 run-A policy with and without the imperfections.
```bash
P=<mac47>/runs/mac47-a
E="--model biped_ds3218 --joint-feedback commanded --torque-limit 1.96 --motor-no-load-speed 7.0"
uv run python scripts/eval_biped.py $P/policy.onnx $E --dr --robustness --envs 8 --seconds 20 \
    --servo-dr --out runs/mac49/eval_servo_dr.json
uv run python scripts/eval_biped.py $P/policy.onnx $E --obs-mode runtime --envs 16 --seconds 20 \
    --servo-dr --out runs/mac49/eval_servo_dr_runtime.json
```
If step 1 meets the WP4-40 rows (no fall in nominal / DR cases, vx >= 50 % at 0.15 and 0.25), stop: report it.

Step 2 (only if step 1 misses): fine-tune from run A with the imperfections on, then repeat step 1 on the
new policy and the MAC-48 seed part (8 starts × 0.15 / 0.25 through the runtime, no `--realtime`).
```bash
F="--robot biped_v1 --model biped_ds3218 --joint-feedback commanded --dr --servo-dr --envs 256 --seed 0 \
   --track-yaw-std2 0.25 --symmetry loss --mirror-loss-coeff 0.5 --torque-limit 1.96 --motor-no-load-speed 7.0"
uv run python scripts/train_ppo.py $F --iters 800 --workers <best> --resume $P/model_<newest>.pt \
    --run-dir runs/mac49 --save-every 100 --out runs/mac49/report.json
```
**Acceptance (WP4-41):** a DS3218MG policy that meets the WP4-40 rows with `--servo-dr` on (both eval modes),
and, if step 2 ran, also without it (no regression) and through the runtime (MAC-48 seed rows). Report
the MAC-25 rows, sizing, slew-limited fraction and the ONNX sha256; commit a summary JSON under
`docs/experiments/mac49/` (no ONNX, no runs/). No hardware.

Result (env: mac-m5, 2026-11-24, revision c3c1002): MET at step 1, step 2 not run
```
policy: MAC-47 run A, ONNX sha256 d9a1d6ae1c82b4fc3dab53e4d9f2f5dd07d2231788f3867fc2bc3f100e4bdb5c
step 1, --servo-dr on (offset +-0.03 rad, deadband 0-0.008 rad, stiffness 0.7-1.3):
  eval --dr --robustness (8 envs x 20 s): 0 falls in every case
    0.15 / 0.25 DR off: vx 0.163 (109 %) / 0.245 (98 %), wz -0.000 / +0.017
    0.15 / 0.25 DR on:  vx 0.139 (93 %) / 0.212 (85 %), wz -0.021 / -0.004
    zero command wz +0.079 / +0.033; turn 0.8 wz 1.017 / 0.840
    friction 0.5 / 1.5 vx 0.156 / 0.166; pushes 0.163 / 0.163; latency 20 / 40 / 60 ms 0.136 / 0.116 / 0.100
      (91 / 77 / 67 % of 0.15)
  eval --obs-mode runtime (16 envs x 20 s): 0 falls; 0.15 / 0.25 vx 0.164 / 0.248, wz -0.002 / +0.031; turn 0.8 1.033
  sizing (DR-robustness eval; walking max torque / RMS / p95 speed; servo 1.96 N.m / 7.0 rad/s):
    hip_yaw 0.390 / 0.115 / 1.99; hip_roll 1.101 / 0.356 / 2.15; hip_pitch 1.079 / 0.377 / 5.03;
    knee 0.951 / 0.273 / 5.13; ankle_pitch 0.878 / 0.250 / 4.35; at_torque_limit false everywhere
  slew-limited fraction: 0.000 at 0.15, 0.020 / 0.011 at 0.25 (DR off / on), 0.021 at 60 ms latency
verdict: met. The MAC-47 policy keeps the WP4-40 rows with cheap-servo imperfections on, in both eval modes:
  no fall, vx >= 85 % of the command, worst walking torque 1.10 N.m (bar 1.3). Against MAC-47 without them:
  vx drops ~2-4 %, ankle and knee torque rise (0.76 -> 0.88, 0.83 -> 0.95 N.m), and the 40 ms latency row
  falls from 81 to 77 % (not a WP4-40 row). No fine-tune needed. No hardware.
```

## MAC-50 — biped_ds3218b: walk on the buildable DS3218MG layout (WP5-13)
Status: `done (mac-m5, 2026-11-24): met by the step-1 fine-tune (no from-scratch fallback)` · Deps: WP5-13 on main (`--model biped_ds3218b`), MAC-47 run A (`runs/mac47-a/model_<newest>.pt`)

`scripts/check_clearance.py` (cloud) shows biped_ds3218 cannot be built: it keeps the 18 g servo hip, where
the hip-roll and hip-pitch servo boxes overlap by 20 mm. biped_ds3218b places the same ten DS3218MG as
U-brackets hold them; every same-leg pair clears by >= 5 mm. Changes against biped_ds3218: hip pitch axis
50 mm lower (yaw->roll 30 mm, roll->pitch 50 mm), trunk 10 mm flatter, stand height 0.262 m (top 0.297 m),
same mass (1.04 kg), joints, ranges, stand pose and Hello file (`biped_ds3218.hello` is byte-identical), and
the servo boxes now collide with the other leg (the policy must not cross its legs). Cloud CPU training
runs ~2.9 s/iteration at 256 envs on 4 cores; the checkpoint to start from is only on the Mac.

Step 1: fine-tune MAC-47 run A on the new model with DR and the servo imperfections on.
```bash
P=<mac47>/runs/mac47-a
F="--robot biped_v1 --model biped_ds3218b --joint-feedback commanded --dr --servo-dr --envs 256 --seed 0 \
   --track-yaw-std2 0.25 --symmetry loss --mirror-loss-coeff 0.5 --torque-limit 1.96 --motor-no-load-speed 7.0"
uv run python scripts/train_ppo.py $F --iters 1500 --workers <best> --resume $P/model_<newest>.pt \
    --run-dir runs/mac50 --save-every 100 --out runs/mac50/report.json
```
If the reward stalls far below MAC-47's or the robot falls in most episodes after 1500 iterations, train from
scratch instead with the MAC-37 B settings and `--model biped_ds3218b` (note it in the result).

Step 2: evaluate as MAC-49 step 1 with `--model biped_ds3218b`, with and without `--servo-dr`, plus the MAC-48
seed part through the runtime (8 starts x 0.15 / 0.25, `runtime_sim.py --model biped_ds3218b`, no `--realtime`).

**Acceptance (WP5-13 sim part):** the WP4-40 rows on biped_ds3218b (no fall nominal / DR, vx >= 50 % at
0.15 and 0.25, worst walking torque <= 1.3 N.m) with `--servo-dr` on in both eval modes, and the MAC-48 seed
rows through the runtime. Report the MAC-25 rows, sizing, slew-limited fraction and ONNX sha256; commit a
summary JSON under `docs/experiments/mac50/` (no ONNX, no runs/). No hardware.

Result (env: mac-m5, 2026-11-24, revision 7f46f2e on claude/project-thread-vt5dcm): MET
```
step 1: the fine-tune above from mac47-a/model_6500.pt, --workers 8; 2267 s (38 min); reward/step 0.084 ->
  0.105; training falls/it (150-it means) 37.0 (the new hip geometry at first) -> 1.76 -> 0.3-0.75 from it 150
  on. The from-scratch fallback was not needed.
step 2 (0 falls in every case of every eval; vx at 0.15 / 0.25; wz):
  --servo-dr, --dr --robustness (8 envs): DR off 0.166 (111 %) / 0.248 (99 %), wz +0.006 / +0.023;
    DR on 0.142 (95 %) / 0.219 (88 %), wz -0.008 / +0.014; turn 0.8 0.791 / 0.698; latency 20 / 40 / 60 ms
    0.144 / 0.127 / 0.112 (96 / 85 / 75 %); friction and pushes 0 falls
  --servo-dr, --obs-mode runtime (16 envs): 0.172 / 0.257, wz -0.002 / +0.017; turn 0.8 0.805
  without --servo-dr, --dr --robustness: DR off 0.174 / 0.256, DR on 0.150 / 0.228; 40 ms 0.135 (90 %)
  without --servo-dr, --obs-mode runtime: 0.174 / 0.259, wz +0.004 / +0.013
  sizing (--servo-dr, DR robustness eval; walking max torque / RMS / p95 speed; 1.96 N.m / 7.0 rad/s):
    hip_yaw 0.421 / 0.103 / 1.82; hip_roll 1.246 / 0.367 / 2.14; hip_pitch 1.094 / 0.401 / 5.08;
    knee 0.972 / 0.299 / 5.16; ankle_pitch 0.841 / 0.245 / 4.40; at_torque_limit false everywhere
    (other evals: hip roll 0.89-1.12, the rest within the same range)
  slew-limited fraction: 0.000 at 0.15, 0.006-0.015 at 0.25, <= 0.003 at latency
  runtime seed rows (8 starts, runtime_sim --model biped_ds3218b, no --realtime): 16 runs, no fall, no
    latched fault, 1000 periods; 0.15 vx 0.174 [0.174..0.175], wz +0.005 [-0.000..+0.014]; 0.25 vx 0.259
    [0.259..0.259], wz +0.012 [+0.006..+0.019]; max torque demand 1.56 / 2.03 N.m, torque_clamp 0
policy.onnx sha256: a8fee4a15563889df7733ed4cb2f9ba255b20cebde368572471e882435ef7ea9
  (policy.bpml 6a2d13501c0d8ec2b1126607d6f80c4c0bdae61066fa37d27437848eebb72245, rust vs onnxruntime 1.3e-6)
verdict: met. On the buildable layout, with servo imperfections on, the fine-tuned policy has no fall in any
  case, vx >= 88 % of the command, and a worst walking torque of 1.246 N.m (hip roll, just under the 1.3 bar). Without the
  imperfections it does not regress, and the runtime seed rows match the Python eval. Hip roll has the least margin
  on this layout. No hardware.
```

## MAC-51 — biped_ds3218b_heavy: does the MAC-50 gait carry the real bracket mass? (WP5-15)
Status: `done (mac-m5, 2026-11-24): met at step 1 (MAC-50 policy carries the heavy brackets); step 2 not needed` · Deps: WP5-15 on main (`--model biped_ds3218b_heavy`), MAC-50 (`runs/mac50/policy.onnx`, `model_<newest>.pt`)

Off-the-shelf MG995/MG996 aluminium brackets weigh ~25-30 g per joint (`hardware/bom/frame_ds3218b.md`).
biped_ds3218b_heavy is biped_ds3218b with heavier brackets, links and feet: 1.17 kg instead of 1.04 kg, same
geometry and Hello. MAC-50's worst walking torque was 1.246 N.m (hip roll, bar 1.3), so the margin is thin.

Step 1 (eval only): the MAC-50 policy on the heavy model, as MAC-50 step 2 (with `--servo-dr`, both eval modes).
```bash
P=<mac50>/runs/mac50
E="--model biped_ds3218b_heavy --joint-feedback commanded --torque-limit 1.96 --motor-no-load-speed 7.0"
uv run python scripts/eval_biped.py $P/policy.onnx $E --dr --robustness --envs 8 --seconds 20 \
    --servo-dr --out runs/mac51/eval_heavy.json
uv run python scripts/eval_biped.py $P/policy.onnx $E --obs-mode runtime --envs 16 --seconds 20 \
    --servo-dr --out runs/mac51/eval_heavy_runtime.json
```
Step 2 (only if step 1 misses the rows, or any joint's walking max torque is above 1.3 N.m): fine-tune MAC-50 on
`--model biped_ds3218b_heavy` with `--dr --servo-dr` (MAC-50 flags, 800 iterations, run dir runs/mac51), then
repeat step 1 on it, on biped_ds3218b (no regression), and the MAC-48 seed rows through the runtime.

**Acceptance (WP5-15):** a policy that meets the WP4-40 rows on biped_ds3218b_heavy with `--servo-dr` in both
eval modes. Report the torque per joint against 1.3 / 1.96 N.m: if hip roll stays above 1.3 even after
step 2, say so plainly (the frame then needs lighter parts or printed brackets). Summary JSON under
`docs/experiments/mac51/` (no ONNX, no runs/). No hardware.

Result (env: mac-m5, 2026-11-24, revision d099f27 on claude/project-thread-vt5dcm): MET at step 1
```
policy: MAC-50, ONNX sha256 a8fee4a15563889df7733ed4cb2f9ba255b20cebde368572471e882435ef7ea9 (unchanged)
step 1, biped_ds3218b_heavy (1.17 kg), --servo-dr on; 0 falls in every case of both evals:
  --dr --robustness (8 envs x 20 s): DR off vx 0.170 (113 %) / 0.255 (102 %), wz -0.003 / +0.022;
    DR on 0.144 (96 %) / 0.222 (89 %), wz -0.012 / +0.023; turn 0.8 0.807 / 0.706; friction 0.5 / 1.5 0.165 / 0.174;
    pushes 0.171 / 0.169; latency 20 / 40 / 60 ms 0.147 / 0.129 / 0.112 (98 / 86 / 75 %)
  --obs-mode runtime (16 envs x 20 s): vx 0.176 / 0.264, wz -0.007 / +0.023; turn 0.8 0.814
  torque per joint (walking max, DR eval / runtime eval; bar 1.3, stall 1.96 N.m) and p95 speed (no-load 7.0):
    hip_yaw 0.422 / 0.443 N.m, 1.79 rad/s; hip_roll 1.255 / 1.241, 2.15; hip_pitch 1.123 / 1.084, 5.07;
    knee 0.935 / 0.918, 5.16; ankle_pitch 0.846 / 0.801, 4.42; at_torque_limit false everywhere
  slew-limited fraction: 0.000 at 0.15, 0.006-0.016 at 0.25, <= 0.006 at latency
verdict: met. The MAC-50 policy carries the real bracket mass (+0.13 kg) without retraining: no fall, vx >= 89 %
  of the command, every joint <= 1.3 N.m. Hip roll is the tightest at 1.255 N.m (MAC-50 on biped_ds3218b: 1.246),
  only 0.045 N.m below the bar, so any further mass at the hips or trunk will push it over. No hardware.
```

## MAC-52 — buy torque margin at the hip roll (WP5-16)
Status: `done (mac-m5, 2026-11-24): met (every joint <= 1.01 N.m); flat yaw drift grew, see result` · Deps: WP5-16 on main (`--torque-soft`), MAC-50 (`runs/mac50/model_<newest>.pt`)

MAC-51: the MAC-50 policy carries the heavy frame (1.17 kg) but hip roll peaks at 1.255 N.m, 0.045 under the
1.3 N.m bar (2/3 of the DS3218MG's 1.96 stall). Any extra mass at the hip or trunk (wiring, a bigger
battery) would cross it. `--torque-soft T` adds a reward term (weight -20) on the squared torque above T N.m.

Step 1: fine-tune MAC-50 on the heavy model with the penalty.
```bash
P=<mac50>/runs/mac50
F="--robot biped_v1 --model biped_ds3218b_heavy --joint-feedback commanded --dr --servo-dr --envs 256 --seed 0 \
   --track-yaw-std2 0.25 --symmetry loss --mirror-loss-coeff 0.5 --torque-limit 1.96 --motor-no-load-speed 7.0 \
   --torque-soft 1.0"
uv run python scripts/train_ppo.py $F --iters 800 --workers <best> --resume $P/model_<newest>.pt \
    --run-dir runs/mac52 --save-every 100 --out runs/mac52/report.json
```
Step 2: evaluate as MAC-51 (heavy, `--servo-dr`, both modes), on biped_ds3218b (no regression), and the MAC-48
seed rows through the runtime on `--model biped_ds3218b_heavy`.

**Acceptance (WP5-16):** the WP4-40 rows on biped_ds3218b_heavy with `--servo-dr` in both eval modes, with every
joint's walking max torque <= 1.15 N.m (>= 0.15 N.m under the bar), and the runtime seed rows clean. If vx
drops below 50 % or falls appear, report the trade-off (torque vs speed) and keep MAC-50 as the reference.
Summary JSON under `docs/experiments/mac52/` (no ONNX, no runs/). No hardware.

Result (env: mac-m5, 2026-11-24, revision 1218150 on claude/project-thread-vt5dcm): MET (with a yaw side effect)
```
step 1: fine-tune from mac50/model_8000.pt with the command above, --workers 8; 1268 s (21 min); reward/step
  0.104 -> 0.109; training falls/it (100-it means) 0.17-0.41; torque_over term -0.0015 -> -0.0005
step 2 (--servo-dr on; 0 falls in every case of every eval):
  heavy, --dr --robustness (8 envs): DR off vx 0.165 (110 %) / 0.249 (100 %), wz -0.091 / -0.066;
    DR on 0.143 (95 %) / 0.219 (88 %), wz -0.081 / -0.056; zero command wz -0.150 / -0.105; turn 0.8 0.768 / 0.687;
    latency 20 / 40 / 60 ms 0.146 / 0.131 / 0.117 (97 / 87 / 78 %)
  heavy, --obs-mode runtime (16 envs): 0.172 / 0.257, wz -0.102 / -0.063; zero -0.149; turn 0.8 0.771
  biped_ds3218b, --dr --robustness: DR off 0.162 / 0.244, DR on 0.141 / 0.216; runtime mode 0.168 / 0.252 (no regression in vx)
  torque per joint (walking max; heavy DR eval / heavy runtime eval / ds3218b DR eval; target <= 1.15, bar 1.3, stall 1.96 N.m):
    hip_yaw 0.528 / 0.464 / 0.515; hip_roll 1.008 / 0.975 / 0.931; hip_pitch 0.999 / 0.994 / 0.975;
    knee 0.931 / 0.871 / 0.906; ankle_pitch 0.771 / 0.778 / 0.751; at_torque_limit false everywhere
  p95 speed (heavy DR eval; no-load 7.0 rad/s): hip_yaw 2.33, hip_roll 1.96, hip_pitch 5.07, knee 5.18, ankle 4.26
  slew-limited fraction <= 0.001
  runtime seed rows (heavy, 8 starts, no --realtime): 16 runs, no fall, no latched fault, 1000 periods;
    0.15 vx 0.173, wz -0.058 [-0.063..-0.051]; 0.25 vx 0.259, wz -0.012 [-0.019..-0.005]; max torque demand
    1.49 / 1.87 N.m (MAC-50 runtime: 1.56 / 2.03), torque_clamp 0
policy.onnx sha256: d899479472aa1f9521f3f865a28deeae9a6568b1396e29920bc1ee8a90837c80
  (policy.bpml 17e8b295087ee79654302dea3be78943819d6ad5bd0859e5a0c3b1355dd3dda9, rust vs onnxruntime 1.2e-6)
verdict: met. Every joint's walking max is <= 1.008 N.m on the heavy model (hip roll 1.255 -> 1.008, >= 0.29 N.m
  under the bar), with no fall, vx >= 88 % and clean runtime seed rows. Side effect: the walking yaw moved from
  ~0 to -0.06...-0.10 rad/s at 0.15 / 0.25 and -0.15 at the zero command (MAC-51: -0.003 / +0.022, -0.015).
  The WP4-40 rows do not include yaw, but the MAC-25 |wz| <= 0.08 row would fail at 0.15 (-0.091 DR off,
  -0.102 runtime mode). Keep MAC-50 as the yaw reference; this policy is the torque-margin candidate. A short
  fine-tune with --track-yaw-std2 tighter (or mirror loss 1.0) may recover the yaw. No hardware.
```

## MAC-53 — win back straight walking on the MAC-52 policy (WP5-16)
Status: `done (mac-m5, 2026-11-24): NOT met (yaw worse; MAC-50 stays the yaw reference, MAC-52 the torque candidate)` · Deps: MAC-52 (`runs/mac52/model_<newest>.pt`)

MAC-52 bought the torque margin (every joint <= 1.008 N.m on the heavy model) but the gait now drifts in yaw:
wz -0.091 at 0.15 m/s and -0.150 with a zero command (MAC-51: -0.003 / -0.015). The MAC-25 row |wz| <= 0.08
at 0.15 fails. Short fine-tune with tighter yaw tracking and a stronger mirror loss, torque penalty kept.

```bash
P=<mac52>/runs/mac52
F="--robot biped_v1 --model biped_ds3218b_heavy --joint-feedback commanded --dr --servo-dr --envs 256 --seed 0 \
   --track-yaw-std2 0.1 --symmetry loss --mirror-loss-coeff 1.0 --torque-limit 1.96 --motor-no-load-speed 7.0 \
   --torque-soft 1.0"
uv run python scripts/train_ppo.py $F --iters 500 --workers <best> --resume $P/model_<newest>.pt \
    --run-dir runs/mac53 --save-every 100 --out runs/mac53/report.json
```
Then the MAC-52 evals (heavy both modes with `--servo-dr`, biped_ds3218b, MAC-48 runtime seed rows on heavy).

**Acceptance (WP5-16):** MAC-52's rows (no falls, vx >= 50 %, every joint <= 1.15 N.m) and |wz| <= 0.08 at
0.15 and 0.25 and with a zero command, in both eval modes. If yaw does not come back within 500 iterations,
report it; MAC-50 stays the yaw reference and MAC-52 the torque-margin candidate. Summary JSON under
`docs/experiments/mac53/` (no ONNX, no runs/). No hardware.

Result (env: mac-m5, 2026-11-24, revision 368cb79 on claude/project-thread-vt5dcm): NOT met, yaw got worse
```
run: fine-tune from mac52/model_8800.pt with the command above, --workers 8; 978 s (16 min); reward/step 0.106 ->
  0.107; training falls/it 0.23-0.29; track_angular_velocity term flat at ~0.005
evals (--servo-dr on; 0 falls in every case; vx at 0.15 / 0.25; wz at 0.15 / 0.25 / zero command):
  heavy --dr --robustness, DR off: vx 0.164 (109 %) / 0.241 (96 %); wz -0.111 / -0.038 / -0.290
                           DR on:  vx 0.141 (94 %) / 0.216 (86 %); wz -0.078 / -0.001 / -0.202
  heavy --obs-mode runtime:        vx 0.170 / 0.250;                 wz -0.148 / -0.054 / -0.300
  biped_ds3218b DR off / on / runtime: wz at 0.15 -0.124 / -0.088 / -0.153, zero -0.302 / -0.212 / -0.309
  turn 0.8: 0.656 (heavy DR off), 0.584 (DR on)
  MAC-52 for comparison (heavy DR off / runtime): wz -0.091 / -0.102 at 0.15, -0.150 / -0.149 at zero
  torque per joint (walking max, heavy DR / heavy runtime / ds3218b DR; target 1.15 N.m):
    hip_yaw 0.424 / 0.338 / 0.392; hip_roll 1.158 / 0.945 / 0.979; hip_pitch 1.072 / 1.060 / 1.053;
    knee 0.986 / 0.888 / 0.930; ankle_pitch 0.781 / 0.742 / 0.719 (hip roll 1.158 in the heavy DR eval, just
    over 1.15)
  runtime seed rows (heavy, 8 starts): no fall, no fault, 1000 periods; 0.15 vx 0.168, wz -0.112 (FAILS 0.08);
    0.25 vx 0.248, wz +0.031
policy.onnx sha256: 5acc2cf37ffb68de2ff8fe705aadd24678865d5e84df996b6ff89ed67a946d25
verdict: not met. The yaw drift did not come back. At the zero command it doubled (-0.15 -> -0.29...-0.31)
  and at 0.15 it grew to -0.11...-0.15, while hip roll went back above 1.15 in one eval. Like biped_v1
  in MAC-19, a tight yaw kernel (std2 0.1) gives almost no gradient at large yaw errors, so the policy spins in
  place instead. MAC-50 stays the yaw reference, and MAC-52 the torque-margin candidate. Next (cloud): train
  MAC-52 with a milder kernel (std2 0.25, as MAC-21/25 on biped_v1) plus mirror loss 1.0, or relax the torque
  penalty on hip yaw only. No hardware.
```

## MAC-54 — yaw back on the torque-margin policy, softer kernel (WP5-16)
Status: `done (mac-m5, 2026-11-24): MET (torque margin and straight walking together)` · Deps: MAC-52 (`runs/mac52/model_<newest>.pt`)

MAC-53 (yaw std2 0.1) made the drift worse (zero command -0.29 rad/s), as MAC-19 did on biped_v1: a tight kernel
leaves no gradient for large yaw errors. Use the MAC-21/25 recipe instead: the default kernel (std2 0.25) with
the stronger mirror loss, from MAC-52, torque penalty kept.

```bash
P=<mac52>/runs/mac52
F="--robot biped_v1 --model biped_ds3218b_heavy --joint-feedback commanded --dr --servo-dr --envs 256 --seed 0 \
   --track-yaw-std2 0.25 --symmetry loss --mirror-loss-coeff 1.0 --torque-limit 1.96 --motor-no-load-speed 7.0 \
   --torque-soft 1.0"
uv run python scripts/train_ppo.py $F --iters 800 --workers <best> --resume $P/model_<newest>.pt \
    --run-dir runs/mac54 --save-every 100 --out runs/mac54/report.json
```
Then the MAC-52 evals. **Acceptance:** as MAC-53 (no falls, vx >= 50 %, every joint <= 1.15 N.m, |wz| <= 0.08
at 0.15 / 0.25 / zero command in both modes and the runtime seed rows). If it misses, stop the WP5-16 tuning:
MAC-50 stays the reference policy and the torque margin goes on the hardware list (lighter brackets). Summary
JSON under `docs/experiments/mac54/`. No hardware.

Result (env: mac-m5, 2026-11-24, revision 09de825 on claude/project-thread-vt5dcm): MET
```
run: fine-tune from mac52/model_8800.pt with the command above, --workers 8; 1614 s (27 min); reward/step 0.109 ->
  0.115; training falls/it 0.15-0.22
evals (--servo-dr on; 0 falls in every case; vx at 0.15 / 0.25; wz at 0.15 / 0.25 / zero command):
  heavy --dr --robustness, DR off: vx 0.164 (109 %) / 0.242 (97 %); wz -0.014 / +0.033 / -0.055
                           DR on:  vx 0.143 (95 %) / 0.216 (86 %); wz -0.023 / +0.008 / -0.049
  heavy --obs-mode runtime:        vx 0.169 / 0.249;                 wz -0.030 / +0.025 / -0.057
  biped_ds3218b DR off / DR on:    vx 0.159 / 0.234, 0.141 / 0.212;  wz -0.009 / +0.040 / -0.052, -0.027 / +0.007 / -0.056
  biped_ds3218b runtime mode:      vx 0.165 / 0.243;                 wz -0.022 / +0.030 / -0.059
  turn 0.8: 0.889 (heavy), 0.886 (runtime mode); 40 ms latency 91 % (heavy), 89 % (ds3218b)
  torque per joint (walking max; heavy DR / heavy runtime / ds3218b DR; target 1.15, bar 1.3, stall 1.96 N.m)
  and p95 speed (heavy DR; no-load 7.0 rad/s):
    hip_yaw 0.422 / 0.405 / 0.437, 1.99; hip_roll 1.050 / 1.018 / 1.005, 1.61; hip_pitch 1.079 / 1.034 / 1.038, 5.09;
    knee 1.001 / 0.857 / 0.984, 5.18; ankle_pitch 0.838 / 0.745 / 0.803, 4.34; at_torque_limit false everywhere
  runtime seed rows (heavy, 8 starts): 16 runs, no fall, no latched fault, 1000 periods; 0.15 vx 0.170, wz -0.018
    [-0.022..-0.010]; 0.25 vx 0.252, wz +0.039 [+0.034..+0.046]; max torque demand 1.42 / 2.04 N.m, torque_clamp 0
policy.onnx sha256: e5720fca62fc5750660962384f6ce16c9ec80555818c6ef1b552963d30cffcb1
  (policy.bpml 4a32b010888a47cdaa05ff54d63eb957612e03f43a36c89305f5c22fc3dd3b98)
verdict: met. The softer kernel with mirror loss 1.0 brings straight walking back on the torque-margin policy:
  |wz| <= 0.059 everywhere (MAC-52 zero command -0.150, MAC-53 -0.29), with every joint <= 1.079 N.m
  (>= 0.22 N.m under the 1.3 bar), no fall, vx >= 86 %, turn 0.8 -> 0.89, and clean runtime seed rows. This
  policy is the new heavy-frame reference (supersedes MAC-50 for the 1.17 kg build). No hardware.
```

## MAC-55 — hardening the MAC-54 reference: supply sag, uneven floor, pushes, turning (WP5-17)
Status: `todo` · Deps: WP5-17 on main (`--supply-dr`), MAC-54 (`runs/mac54/policy.onnx`, `model_<newest>.pt`)

Sim-only checks of the DS3218MG reference policy on the 1.17 kg model while parts wait. `--supply-dr` draws a
supply scale U(0.85, 1.0) per env and reset and applies it to stall torque and no-load speed (one UBEC / LiPo
sagging under ten loaded servos).

Step 1 (eval only), everything with `--servo-dr --supply-dr`:
```bash
P=<mac54>/runs/mac54
E="--model biped_ds3218b_heavy --joint-feedback commanded --torque-limit 1.96 --motor-no-load-speed 7.0 \
   --servo-dr --supply-dr"
uv run python scripts/eval_biped.py $P/policy.onnx $E --dr --robustness --terrain --envs 8 --seconds 20 \
    --out runs/mac55/eval_hard.json
uv run python scripts/eval_biped.py $P/policy.onnx $E --obs-mode runtime --envs 16 --seconds 20 \
    --out runs/mac55/eval_hard_runtime.json
```
Rows to report: falls and vx per case (nominal, DR, friction, pushes 0.3 m/s forward / sideways, latency,
terrain 5 / 10 / 20 mm), turning in place (0.8 rad/s command), torque per joint.

Step 2 (only if a nominal, DR, push or 5 / 10 mm terrain case falls, or vx < 50 %): fine-tune MAC-54 with the MAC-54
flags plus `--supply-dr` (and `--rough-floor 0.01 --rough-floor-flat-fraction 0.4` if terrain failed, as
MAC-41), 800 iterations into runs/mac55, then repeat step 1 and the MAC-48 runtime seed rows on heavy.

**Acceptance (WP5-17):** no falls in nominal / DR / pushes / 5 and 10 mm terrain with `--supply-dr`, vx >= 50 %,
every joint <= 1.15 N.m, |wz| <= 0.08 straight; 20 mm terrain is reported, not required. Summary JSON under
`docs/experiments/mac55/` (no ONNX, no runs/). No hardware.

---

## Later Mac tasks (not ready yet)
- MAC-05 live telemetry overlay (after WP2-01)
