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

---

## Later Mac tasks (not ready yet)
- MAC-05 live telemetry overlay (after WP2-01)
- MAC-06 PPO training on CPU vs MPS (after WP3-03)
