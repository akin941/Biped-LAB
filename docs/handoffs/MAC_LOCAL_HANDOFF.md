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

## Later Mac tasks (not ready yet)
- MAC-05 live telemetry overlay (after WP2-01)
