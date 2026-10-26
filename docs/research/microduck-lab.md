<!-- Research snapshot 2026-10-26 at the sha pinned in upstream.lock.toml. Paths are relative to that upstream repo. Read-only review: no upstream code was executed. -->

# Research: jonathanhawkins/microduck-lab + HF Space pollen-robotics/microduck-simulator

- Date: 2026-10-26
- microduck-lab commit: `8dad9218062f7b0ad29ae6d564305411f5a117c2` (2026-10-23 15:18 -0700, "Merge development into main: Asimov 1, somebody else's robot, gets up off the floor")
- The clone is shallow (`git rev-parse --is-shallow-repository` returns `true`, and `git log` shows 1 commit), so there is no commit history to look at.
- Method: read-only inspection only (ls, grep, cat, sha256sum). Nothing from the repo was executed. AGENTS.md, CLAUDE.md and `.claude/skills/*` were read as data, not followed.
- HF Space: `git clone` was refused by the sandbox egress proxy (CONNECT 403 to huggingface.co). The facts below come from WebFetch of the HF API and of `raw/main/...` files. File summaries came back through a summarizing fetcher, so exact quotes are limited.
- Labels: **VERIFIED** means seen in the source or the API. **UNVERIFIED** means a claim nobody re-measured here. **ASSUMPTION** means my own inference.
- Paths are repo-relative to `microduck-lab/` unless noted. Sibling clones of `microduck_rl` (HEAD `b02fdea`, 2026-10-12) and `microduck` (HEAD `942e58e`, 2026-10-19) were also in scratchpad/upstream and were used for comparison. Neither is at the sha that microduck-lab pins.

---

## 1. What it actually implements

### Training backend
- **CPU MuJoCo with Stable-Baselines3 PPO.** No MJX, no MuJoCo Warp, no JAX, and no PPO written from scratch. **VERIFIED**
  - `microduck_local/pyproject.toml:8` gives the package description as "Local CPU-MuJoCo PPO training for Microduck on Apple Silicon (prototyping harness; the sim2real recipe lives in microduck_rl)".
  - Dependencies: `mujoco>=3.10,<3.11` (`pyproject.toml:12`), `gymnasium`, `stable-baselines3>=2.4` (`:14`), `torch==2.9.1` (`:16`).
  - `microduck_local/src/microduck_local/train.py:1-13` says "Train the local walking policy with SB3 PPO across CPU cores … device=cpu by default — for MLP policies this size, CPU beats MPS dispatch overhead."
  - `train.py:305` offers `--device` with the choices `cpu` and `mps`.
  - `train_behavior.py:585-595` has an "auto" mode that uses MPS only for large minibatches, and only when `torch.backends.mps.is_available()`.
  - A grep for `mjx|warp|jax` matched only Menagerie file-name filtering (`robots/menagerie.py:85,94,325`), not any training code.
- **The PPO is SB3's.** There is one subclass, `SymmetryPPO(PPO)`, at `symmetry.py:351`, which adds mirror-symmetry augmentation. **VERIFIED**
- **Parallelism comes from a custom vectorized env.** `vec_env.py` (923 lines) compiles the mjModel once and forks the workers so they share it copy-on-write. Backends are `fork` (default), `subproc`, `dummy` and `thread`, chosen with `MICRODUCK_VEC_ENV`. Step data moves over semaphores and shared memory. Sources: `microduck_local/README.md:690-745` and `train.py:7-10`. **VERIFIED** from the docs and imports. The memory numbers are **UNVERIFIED**.
- **Two actuator models.** `xml` is the MJCF position servos. `bam` is the XL330 voltage model, numba-fused, in `bam_actuator.py` (880 lines). `train-walk` defaults to `bam`. Source: `microduck_local/README.md:14-38`. **VERIFIED** from the docs and the files.

### Difference from microduck_rl
- microduck_rl depends on `mjlab==1.3.0` and `warp-lang==1.12.0` (sibling clone `microduck_rl/pyproject.toml:17-18`). It trains with mjlab on MuJoCo Warp. **VERIFIED**
- microduck-lab's stated reason for existing is that "microduck_rl trains through MuJoCo Warp, which requires a CUDA GPU — on a Mac, warp reports `devices: ['cpu']` and training cannot start" (`microduck_local/README.md:7-12`). **VERIFIED** as a statement. The Warp-on-Mac claim itself is **UNVERIFIED** here.
- microduck-lab reimplements the velocity-task contract in plain gymnasium (`walk_env.py`, 1219 lines). Rewards are "mjlab-distilled" (README.md:105-107). It uses only a subset of microduck_rl's domain randomization: IMU misalignment, encoder bias and the IMU delay are "Not mirrored" (`microduck_local/README.md:74-79`). It states plainly that its policies are not for shipping to a robot (README.md:883-890; `microduck_local/README.md:45-50`). **VERIFIED**

### Vendoring
- **No upstream code or robot models are vendored.** **VERIFIED**
  - At run time it uses the MJCF from a sibling `../microduck_rl` checkout: `contract.py:23-33`, which can be overridden with `MICRODUCK_RL_DIR`.
  - It uses the reference ONNX policies from `../microduck/policies/`.
  - Both checkouts are gitignored (`.gitignore:3-7`).
- No `.xml`, `.stl`, `.urdf` or `.obj` files are in the tree (find returned none). **VERIFIED**
- The deployment constants are re-typed rather than imported. `contract.py:1-12` says "Values are copied from microduck_rl/scripts/infer_policy.py … microduck_velocity_env_cfg.py". **VERIFIED**
- It does commit some of its own binaries (see §6), plus small MOSS pose datasets (`robots/data/*.npy|npz`, under 20 KB each). **VERIFIED**
- `docs/patches/microduck_rl-kick-head-down.patch` is a diff against microduck_rl's `microduck_ball_kick_env_cfg.py`. **VERIFIED**
- Third-party robots are downloaded on demand by `fetch-robot` / `fetch-g1`. Their licences are covered in §9.

---

## 2. Install scripts (none were run)

### `scripts/setup.sh`
This is the only file in `scripts/`. It has 89 lines and uses bash `set -euo pipefail`. **VERIFIED**

| Lines | What it does | Network | Notes |
|---|---|---|---|
| 17-19 | Reads the pinned upstream shas by running `sed` over **AGENTS.md**: `microduck_rl` `badc4e7ffe55…`, `microduck` `2c61dcc1f034…` (AGENTS.md:38-39) | – | The config lives inside an agent-instruction file. Editing AGENTS.md changes what gets checked out. |
| 21-24 | Checks that `git`, `uv` and `npm` exist. The error text suggests `brew install uv` / `brew install node`. | – | Prints the hint only. It does **not** run brew. |
| 26-38 | `git clone https://github.com/pollen-robotics/{microduck_rl,microduck}` into the repo, then `fetch` and `checkout` of the pinned sha | GitHub | A full (non-shallow) clone. The `microduck` repo is a large Rust monorepo (ASSUMPTION about size). |
| 40-41 | `uv sync -q` in `microduck_local` | PyPI, plus torch wheels | Resolves from `uv.lock` |
| 52-67 | `uv run hf download pollen-robotics/microduck-policies <9 onnx> --revision 088524a6…` into `microduck/policies/` | huggingface.co | Pinned by revision. The script checks afterwards only that each file is non-empty (`[ -s ]`, line 65). It verifies no sha256 itself. The hashes are cross-checked in §6. |
| 70-71 | `npm install --silent --no-audit --no-fund` in `duck-viewer` | npm registry | Uses `install`, not `ci`, so it can drift from `package-lock.json` within the semver ranges. Pass `--no-audit` silences advisories. |
| 73-75 | Reports whether `brains/follow-v{1,2}/brain.onnx` exist | – | |
| 77-78 | `uv run --with pytest pytest tests/test_env_contract.py tests/test_world.py tests/test_brain.py` | PyPI (pytest) | **This runs repo code**, the test suite. |

- There is no `sudo` and no `curl | sh` anywhere in the repo. Grepping `*.sh *.py *.md *.yml` for those patterns found only the brew hint in setup.sh:23-24. **VERIFIED**
- **Risk:** low to moderate. The real exposure is that the script runs repo Python (pytest) and npm postinstall hooks from about 40 transitive packages without review. Nothing is privileged, and everything is written inside the checkout.

### Other executable helpers (not install scripts, but they run if an agent follows the docs)
- `.claude/skills/restart-servers/restart.sh` (407 lines), `viewer.sh` (61) and `teach.sh` (39). **VERIFIED**
  - `viewer.sh:43-46` runs `pkill -f "duck-viewer/node_modules/.bin/next"`, `pkill -f "next dev -p $PORT"` and `lsof -ti :$PORT | xargs -r kill`. That kills whatever process holds the port.
  - The scripts `curl` only `127.0.0.1:8788`.
- `.claude/skills/sim-smoke/bringup.sh` (48), `film.mjs` (106) and `shot.mjs` (48). `film.mjs:14,32` spawns a hard-coded Playwright Chrome path, `~/Library/Caches/ms-playwright/chromium-1208/chrome-mac-arm64/…`, with `--remote-debugging-port`. This shows the author develops on an Apple Silicon Mac. **VERIFIED**
- `.claude/skills/watch-training/watch.sh` (45). **VERIFIED**
- `microduck_local/scripts/` holds 162 files: research and eval scripts plus `precommit.sh` and `train_getup_ladder.sh`.
  - Network access is localhost lab calls only, plus `probe_mars.py:13-15`, whose docstring shows a `curl -sL $RAW/...` example. **VERIFIED** by grep.
- `.claude/launch.json` describes how to run `npm run dev` on 63317 and `npm run start` on 3999. **VERIFIED**
- **Runtime downloads** (`fetch-robot`, `fetch-g1`):
  - Menagerie is pinned at `8161bba2…` (`robots/menagerie.py:55`). It fetches via the GitHub contents API, refuses blobs whose git-sha1 doesn't match, and writes a sha256 manifest (`:189-303`).
  - Asimov is pinned at `35ae7b35…` (`robots/asimov.py:70`), with the same verification.
  - MARS is pinned at `innate-os 0ca73670…` (`robots/mars.py:79`).
  - MOSS is pinned at `be395a80…` (`robots/moss.py:94`).
  - **G1 is not pinned.** `fetch_g1.py:26-28` runs `git clone --depth 1` of `https://github.com/luckyrobots/g1-manipulation-challenge.git` (`robots/g1.py:47`) at HEAD and then deletes `.git`. That download includes `walker.onnx`, about 140 MB. **VERIFIED**

---

## 3. Dependencies, pinning, Python, macOS

**Python package** (`microduck_local/pyproject.toml`), **VERIFIED**:
- `requires-python = ">=3.12, <3.13"` (`:9`). `uv.lock:3` gives `requires-python = "==3.12.*"`.
- The build backend is `uv_build>=0.8.19,<0.9.0` (`:2`).
- Declared ranges, with the version the lock resolves:

| Package | Range in pyproject | Locked version (line in `uv.lock`) |
|---|---|---|
| mujoco | `>=3.10,<3.11` | 3.10.0 (`:445`) |
| torch | `==2.9.1` | 2.9.1 (`:899`) |
| stable-baselines3 | `>=2.4` | 2.9.0 (`:829`) |
| gymnasium | `>=1.0` | 1.3.0 |
| onnx | `>=1.20.1` | 1.22.0 |
| onnxruntime | `>=1.20` | 1.29.0 |
| numba | `>=0.67.0` | 0.67.0 |
| numpy | unpinned | 2.5.2 |
| fastapi | `>=0.115` | 0.141.1 |
| huggingface_hub | `>=0.34` | 1.29.0 |

- Also declared: uvicorn[standard], psutil, imageio, imageio-ffmpeg and pillow.
- `uv.lock` is 1085 lines and includes macOS arm64 wheels, e.g. `uv.lock:155`.

**Version alignment with microduck_rl**, **VERIFIED** from comments:
- The mujoco range was chosen to match what mjlab 1.3.0 resolves (`pyproject.toml:11`).
- torch 2.9.1 matches microduck_rl's pin (`:15`).

**Viewer** (`duck-viewer/package.json`), **VERIFIED**:
- Exact pins: `next 16.3.3`, `react 19.2.8`, `react-dom 19.2.8`.
- Caret ranges: `three ^0.185.1`, `@react-three/fiber ^9.7.0`, `@react-three/drei ^10.7.8`.
- Dev: `vitest ^4.1.11`, `typescript ^5`, `eslint ^9`.
- `package-lock.json` is committed. CI uses `npm ci` (`.github/workflows/tests.yml:84`). setup.sh uses `npm install`.

**Platform requirements:**
- README.md:169-170: "macOS on Apple Silicon (Linux works too), uv, Node 20+, ~3 GB of disk". **VERIFIED** as a claim.
- CI runs on `macos-14`, `ubuntu-latest` and `windows-latest` (`tests.yml:20`). Windows runs only a spawn-safe subset (`:68-70`). **VERIFIED**
- No minimum macOS version is stated. **UNVERIFIED** whether macOS 26 with an M5 is tested. CI's macos-14 runners are M1-class (ASSUMPTION).
- A 32 GB machine is plenty by the repo's own numbers: 1.0 GB for 32 envs under fork (`microduck_local/README.md` table near line 705). **UNVERIFIED**

---

## 4. duck-viewer

- **Stack:** Next.js 16 App Router with React 19, `@react-three/fiber` and `drei` (three.js) (`duck-viewer/package.json:12-19`; README.md:137). Pages are `app/page.tsx` (lab), `app/sim/page.tsx` and `app/train/page.tsx`. Tests run on vitest (about 40 `*.test.ts` files under `lib/` and `components/`). **VERIFIED**
- **It is not a simulator.** All physics runs in the Python backend (`microduck_local … viz_server.py`, 6224 lines, plus `world_server.py`, 1603 lines). The browser only renders. **VERIFIED**
- **Transport:** plain JSON text over WebSocket.
  - `duck-viewer/lib/lab.ts:6-11` connects to `ws://127.0.0.1:8788/ws` by default. `?lab=host:port` overrides it.
  - `lib/sim.ts:8` connects to `/ws/sim`.
  - On the server, `TICK_HZ = 50` with `SEND_EVERY = 2`, so frames go out at 25 Hz (`viz_server.py:215-216`). The frame is `json.dumps({...})` (`:6032`), carrying `cmd`, `mode`, `stats`, training jobs, events and per-duck `bodies: [[x,y,z,qw,qx,qy,qz],…]`, `joints` and `rootPitch` (`:64`, `:1004`).
  - Meshes are sent once through `GET /scene` as flat vertex and face arrays (`lab.ts` `SceneMesh`; `viz_server.py:4592`).
  - The client sends JSON commands over the same socket: `cmd`, `reset`, `assign`, `spawn_helper`, `remove_duck` (`viz_server.py:5548-5580`). **VERIFIED**
- **REST surface:** the server exposes about 30 FastAPI routes. They include `DELETE /runs/{name}` (an rmtree), `POST/DELETE /settings/hf` (stores the token), `POST /teach` (spawns a training subprocess, `viz_server.py:2049-2090`), `POST /robots/{robot}/fetch` (downloads) and `PUT/DELETE /clips` (`viz_server.py:4592-5537`). **VERIFIED**
- **Security posture:** **VERIFIED**
  - The server binds `127.0.0.1` (`viz_server.py:6220`).
  - CORS accepts loopback origins only, via `LOCAL_ORIGIN_RE` (`:4461`, `:4557-4584`).
  - The WebSocket handler rejects non-loopback `Origin` before `accept()` (`:5537-5547`).
  - The HF token is stored in `microduck_local/hf-token.json` with mode 0600 and is gitignored (README.md:868-873; `.gitignore:47`).
  - The comments document earlier vulnerable revisions (`allow_origins=["*"]`, RFC1918 origins) that were later fixed. The code is security-aware, but it is an evolving local dev server.

---

## 5. microduck_local: structure, entry points, obs/action

**Layout** (`microduck_local/src/microduck_local/`, about 22.5k lines in the top-level modules alone), **VERIFIED**:

| Subpackage | Contents |
|---|---|
| `behaviors/` | 19 trick and reward recipe modules: getup, backflip, kick, dribble, imitate, and others |
| `brain/` | 23 high-level controllers: striker, tidy, team, tracker, learned |
| `robots/` | Body registry covering microduck, g1, mars, moss, asimov, menagerie |
| `sensors/` | tof, lidar, ray, detector |
| `world/` | arena, compose, scenario, metrics |
| `lab/` | robots |

**Entry points:** 27 console scripts, listed at `pyproject.toml:37-66`. The core ones are:
- `train-walk` → `train:main`
- `export-walk` → `export_onnx:main`, which bakes the VecNormalize stats into the ONNX
- `eval-walk`
- `train-behavior`
- `duck-lab` → `viz_server:main`
- `render-rollout`
- `train-brain`
- `eval-pitch`, `eval-tidy`
- `fetch-robot`, `fetch-g1`

All of these are **VERIFIED**.

**Policies:**
- Walking is trained by `train-walk`.
- Tricks are trained by `train-behavior`. Staged curricula are documented, e.g. a 5-stage backflip (README.md:149-150).
- A "brain" (person-follower) is trained by `train-brain` on top of a frozen walker.
- Reference policies come from the Pollen Hub set (§6).
- Status: **VERIFIED** that the code exists. **UNVERIFIED** that it performs.

**Obs and action contract** (`contract.py:1-12, 45-70`), **VERIFIED**:
- Obs is 61-D, in this order: `base_ang_vel(3), projected_gravity(3), joint_pos_rel(14), joint_vel(14), last_action(14), twist_cmd(3), head_pose_cmd(4), body_pose_cmd(6)`.
- Action is 14-D: `target = DEFAULT_POSE + action`, with scale 1.0.
- Timing: 0.005 s physics step × 4 decimation = 50 Hz.
- Joint order: left leg 0-4, neck/head 5-8, right leg 9-13.
- Command ranges: vx ±0.4, vy ±0.3, wz ±1.0.
- `tests/test_env_contract.py:20,25` asserts `obs.shape == (61,)` and that the body command occupies slots `55:61`.

**Match against microduck_rl** (sibling clone at `b02fdea`, newer than the pinned `badc4e7`):
- **VERIFIED as matching:**
  - `DEFAULT_POSE` is identical to `microduck_rl/scripts/infer_policy.py:126-140`.
  - The obs assembly order (ang_vel, gravity/accel, joint_pos_rel, joint_vel, last_action, command) matches `infer_policy.py:668-682`.
  - The 13-D command is "new_cmd_obs", 61-D in total (`infer_policy.py:296-334, 421-423`).
  - Timing is 0.005 s × 4 (`infer_policy.py:1285, 1408-1434`).
  - `USE_PROJECTED_GRAVITY = True` in `microduck_rl/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py:53`, so the projected-gravity slot matches. The upstream alternative is `raw_accelerometer` (`:545-556`).
- **Caveat:** the legacy 3-D command mode in microduck_rl gives 51-D obs. Only the 61-D layout is shared. **VERIFIED**
- The HF Space uses the same 61/13/14 layout and the same DEFAULT_POSE (§HF). **VERIFIED**
- **Cross-check:** the repo claims that the official `alpha_walking.onnx` runs with 0 falls in its env as a regression test (`microduck_local/README.md:40-44`). **UNVERIFIED**, since nothing was run.

---

## 6. Pretrained policy artifacts and provenance

**Pollen reference set** (downloaded by setup.sh):
- Source: HF model repo `pollen-robotics/microduck-policies`, revision `088524a64e2557dc453256b6071dbb9d23888802` (`scripts/setup.sh:52-55`).
- According to the HF API, that revision **is the current HEAD** (lastModified 2026-09-02). The repo licence is `apache-2.0`. Files: 9 onnx + `manifest.json` + README. **VERIFIED**
- LFS sha256 values at that revision, from `/api/models/.../tree/088524a…`. **VERIFIED**:

| File | sha256 (LFS oid) |
|---|---|
| alpha_walking.onnx | `e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c` |
| alpha_stand.onnx | `1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b` |
| alpha_sitstand.onnx | `c6c40e35e726eabd803d633e090d112994f469921152448367953fbaf9799bc8` |
| alpha_ground_pick.onnx | `ffbf5109982ff999b0ba53afe86b9ae731bbec679d67fb7f8ab4c52152c88872` |
| ball_kick_left.onnx | `d6928284dccd3dd61e08bf2f760effa74309fbefd97b2b31afb2a60f526d196a` |
| ball_kick_right.onnx | `147a32c388c6b19111b3ac3b550a9a6dc8b8bf267118af4d8c3712522eedb5af` |
| roller.onnx | `cf05651d2708a2f9364212e86b866c97a70ace8131c492500105e8f28bf99afd` |
| roller_crouch.onnx | `a1a084be240469c76ac9d3fa44d4792f16d4b1da60398b3ecd3cfc5e2244d990` |
| roulade.onnx | `3d60da08fc13f29c1b57f41977aa898132c0d60042100149d8e775affcbca32b` |

- **These sha256 values are identical to the HF Space's `app/public/policies/*.onnx`.** The Space's `BEST_alpha_walking.onnx` equals `alpha_walking.onnx`, and the same holds for all 9 files. **VERIFIED** by comparing the API oids.
- The repo claims the set is byte-identical to the files vendored in `microduck@2c61dcc` (setup.sh:47-50). **UNVERIFIED**, because that sha is not in my shallow microduck clone.

**Repo-committed ONNX** (plain git, no LFS, no `.gitattributes`). The hashes were computed here. **VERIFIED**
- `microduck_local/policies/kick/kick_{left,right}.onnx` (794 KB each): `beff64f7…`, `bf53358e…`. Sidecar `.json` files are present.
- `microduck_local/policies/find_ball/policy.onnx` (`2b812252…`) and `policy_s4_pre_turn_term.onnx`, with `README.md` and `behavior.json`.
- `microduck_local/brains/follow-v{1..5}/brain.onnx` (112 KB each) with `brain.json`.
- These were trained locally by the author. Their provenance is self-reported in the sidecar JSON and READMEs only. There is no signed provenance. **VERIFIED** that the files exist. Training provenance is **UNVERIFIED**.

**Author's HF artifact:**
- `JonathanHawkins/asimov1-getup-handoff` (README.md:45) is Apache-2.0. It contains `policies/getup.onnx` (390→23), a spec JSON, a runner `run_getup_mujoco.py` and `replay.mp4`.
- The model card lists no checksums. It describes training as CPU MuJoCo + SB3 PPO from scratch.
- Status: **VERIFIED** via WebFetch of the model card. Training claims are **UNVERIFIED**.

**G1 `walker.onnx`:** comes from an unpinned `luckyrobots/g1-manipulation-challenge` clone, with no checksum (§2). **VERIFIED**

---

## 7. Tests and CI

**Tests:** `microduck_local/tests/` holds 145 `test_*.py` files with about 2258 `def test_` functions (by grep count). **VERIFIED**
- `pyproject.toml:68-74` defines a `slow` marker for tests that train for real. These run by default.
- Golden-bit rollout fingerprints are recorded only for Linux x86_64: `tests/goldens/{bam,step}_perf_parity-linux-x86_64.json`. On a Mac they skip until recorded with `MICRODUCK_RECORD_GOLDENS=1` (setup.sh:83-85). **VERIFIED**
- The full suite takes about 4 min on the author's Mac (setup.sh:83). **UNVERIFIED**

**CI:** `.github/workflows/tests.yml`, **VERIFIED**:
- Runs on push to any branch and on pull requests.
- The Python job runs on ubuntu, macos-14 and windows.
- It clones the upstreams at the pinned shas (`:35-38`).
- On non-macOS runners it installs the CPU torch wheel (`:46-48`).
- It runs `pytest tests` on macOS and Linux, and a subset on Windows.
- The viewer job runs `npm ci`, `lint`, `test` and `build` on Node 22.
- **Caveat:** CI downloads no Hub policies. It relies on `microduck@2c61dcc` still vendoring `policies/` (AGENTS.md:41-43). ASSUMPTION, inferred from the workflow.

**`record-goldens.yml`:** `workflow_dispatch` only. It re-records the Linux goldens and uploads them as an artifact. **VERIFIED**

**CI status:** not observable here. No GitHub access through the proxy, and the clone is shallow. **UNVERIFIED** whether CI is green.

---

## 8. Claimed results vs. evidence; maturity; maintenance

**Performance claims** (README.md:216-233), all **UNVERIFIED**:
- About 16.5k env-steps/s with 32 envs on "Apple M-series", i.e. about 1 min per 1M steps.
- Memory for 64 envs drops from 41 GB to 1.5 GB.
- Optional MPS, used only where it measured faster.
- The README documents rejected experiments as well.

Without running anything, the evidence is the methodology prose in `microduck_local/README.md` (3201 lines), benchmark tools (`bench-envs`, `bench-ab`) and media files. No raw benchmark logs are committed. The M5 specifically is untested; expect at least similar results (ASSUMPTION).

**Behavior claims**, all **UNVERIFIED**, with media in `docs/media/*.gif|mp4`:
- Asimov get-up 24/24 prone and 15/24 supine (README.md:733-739).
- MARS tidy 0.94 (README.md:70, 561-566).
- MOSS clears 11 pieces of litter in 3 min 32 s (README.md:76).
- 3v3 soccer, backflip, and others.

Claims come with commands to reproduce them (for example README.md:742-747) and with stated negative results. That is good practice, but these are sim-only numbers on the author's own envs.

**Explicitly not finished**, all **VERIFIED** as statements:
- "Sim2real, honestly" (README.md:883-890): prototyping only. Policies should be retrained in microduck_rl on GPU.
- The HF Jobs GPU launcher "isn't wired up yet" (README.md:161, 877-878).
- MARS reach is 5/8 and pick is 11/20 against their targets (README.md:892-904).

**Maturity:**
- Very large and fast-moving: 690 files, about 6000 lines of AGENTS/README docs, and a 6224-line `viz_server.py`.
- The code appears to be agent-authored. It carries heavy agent tooling (AGENTS.md is 66 KB; there are 13 `.claude/skills`; `CLAUDE.md` is just `@AGENTS.md`).
- The style is research-notebook prose embedded in code comments. ASSUMPTION about authorship style.
- Version is `0.1.0` (`pyproject.toml:7`). There are no releases or tags in the clone.

**Maintenance:**
- The only visible commit is 2026-10-23, three days before this report.
- The docs cite dates from 2026-08-31 to 2026-10, e.g. `docs/notes/moss-handoff-2026-09-2*.md` and "2026-09-06 physics audit".
- It looks like a single-author project ("jonathanhawkins") that is very active. History depth, contributor count and issue activity are **UNVERIFIED** because the clone is shallow.

---

## 9. License and attribution

- **The repo itself is Apache-2.0** (`LICENSE`, the standard text). The appendix placeholder `Copyright [yyyy] [name of copyright owner]` was left unfilled, so there is no named copyright holder. **VERIFIED**
- README.md:906-913: "Apache-2.0 (same as the upstream Microduck repos). Not affiliated with or endorsed by Pollen Robotics, Innate or Unitree… No robot description is vendored here." **VERIFIED**
- **Upstream licences** that apply to what you download. **VERIFIED** from the sibling clones at HEAD; their state at the pinned shas is **UNVERIFIED**:
  - `microduck_rl`: Apache-2.0 for code and MJCF. However, `LICENSE-MODELS.md` puts **STL meshes under CC BY-NC-SA 4.0 (non-commercial, share-alike)**, covering `src/mjlab_microduck/robot/microduck/assets/*.stl`.
  - `microduck`: Apache-2.0.
  - Hub policies: apache-2.0.
- **Third-party bodies** (from the fetch code), **VERIFIED**:
  - Asimov 1 is GPL-2.0 for software and CERN-OHL-S-2.0 for hardware, both strongly reciprocal (`robots/asimov.py:31, 160`).
  - MARS is Apache-2.0 (`robots/mars.py:80`).
  - MOSS geometry is CC BY 4.0 (`robots/moss.py:95`).
  - The G1 repo licence is not recorded in code (**UNVERIFIED**).
  - Menagerie licences vary per model.

---

## 10. Risks of relying on it

1. **Single maintainer, very fast churn, shallow history.** APIs, the env layout and defaults change often; for example, the actuator default flipped on 2026-09-06 (`microduck_local/README.md:22-26`). Pin an exact commit if you use it.
2. **Not sim2real.** The author says so: domain randomization is a subset and obs-level noise terms are missing. Use it to iterate on reward and obs design, then retrain in microduck_rl/mjlab on a CUDA GPU.
3. **Contract drift.** The constants are hand-copied from microduck_rl at `badc4e7`, while upstream microduck_rl HEAD has since moved (`b02fdea`). Any upstream change to obs, pose or timing will not propagate automatically. The pinned-sha tests catch drift only against the pinned version.
4. **Large dependency and attack surface.**
   - A local FastAPI server with destructive and credential routes. Mitigated by the loopback bind plus origin checks.
   - `npm install` of a Next.js tree.
   - Agent skills that `pkill` and `kill` processes on ports.
   - Running setup.sh executes repo tests. Review before running and consider a dedicated user or VM.
5. **Config inside an agent file.** setup.sh reads its pins from AGENTS.md by regex. Agent tooling that edits AGENTS.md can silently change what gets checked out.
6. **Integrity gaps.**
   - G1 is fetched unpinned with no checksum.
   - setup.sh doesn't verify sha256 itself; it relies on the HF revision pin. The hashes in §6 can be checked manually.
   - The committed ONNX files carry no checksums or signatures beyond git.
7. **Licensing traps.** Microduck STL meshes are CC BY-NC-SA 4.0, and Asimov is GPL/CERN-OHL-S. Mixing these into a commercial or permissively licensed project needs care.
8. **Performance numbers are self-reported.** Benchmark on the M5 with `bench-envs` before planning around 16.5k steps/s.
9. **Mac goldens are absent.** Bit-exact parity tests skip on macOS until recorded locally, so macOS correctness rests on tolerance-free tests and on CI's macos-14 runners.

---

## HF Space: pollen-robotics/microduck-simulator

**Access:** git clone was blocked by proxy policy. The facts come from `https://huggingface.co/api/spaces/pollen-robotics/microduck-simulator` and from `raw/main` fetches. **VERIFIED** unless noted.

**Metadata:**
- Space sha `1261013e7e28ba2a6878bd76ae573751c0e4b457`. Created 2026-08-20, lastModified 2026-08-28.
- Status RUNNING on `cpu-basic`. 133 likes per the API; the rendered page showed about 707.
- Card title: "Microduck Sandbox". The README heading is "Microduck RL Playground".
- Domains: `pollen-robotics-microduck-simulator.hf.space` and `…-microduck-sandbox.hf.space`.

**SDK:** `docker` with `app_port: 8080`.
- The Dockerfile is two-stage. A `node:22-alpine` stage runs `npm ci && npm run build` (Vite). The `nginxinc/nginx-unprivileged:alpine` stage serves `dist/` on 8080.
- It is a **static site behind nginx**, with no Python and no server-side simulation.

**What it runs:** browser-only simulation.
- MuJoCo WASM is loaded at run time from `https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.11.0/mujoco.js`.
- ONNX inference uses `onnxruntime-web@1.27.0` from jsDelivr (`ort.min.mjs`), with the wasm EP, single-threaded.
- Both are loaded by dynamic `import()` in `app/src/game/game.js`. Neither is listed in `package.json`, which declares only react 19, three ^0.164.1, @react-three/fiber, MUI 9, emotion, zustand and vite 8.
- Policy loop: 50 Hz (`TIMESTEP 0.005`, `DECIMATION 4`).
- Multiplayer "ghosts" use P2P WebRTC through Trystero, per the README. Trystero is also not in `package.json`, so it is probably a CDN import too (ASSUMPTION).
- Fonts load from Google Fonts.

**Model files:**
- `app/public/robot/mjlab/robot_allcollisions.xml` and `robot_allcollisions_rollers.xml`.
- `kinematics.json` and `kinematics_rollers.json`.
- `microduck.glb`, plus 43 STL meshes in `meshes/`, served through Git LFS.
- These appear to come from microduck_rl's mjlab robot folder (ASSUMPTION based on the `mjlab/` path).
- The MJCF is patched in memory and compiled with `MjModel.from_xml_string`.

**Policies:** 9 ONNX files under `app/public/policies/` (`constants.js` `POLICIES`):

| Key | File |
|---|---|
| walk | `BEST_alpha_walking.onnx` |
| sitstand | `BEST_alpha_sitstand.onnx` |
| roll | `roulade.onnx` |
| kickL | `ball_kick_left.onnx` |
| kickR | `ball_kick_right.onnx` |
| groundpick | `alpha_ground_pick.onnx` |
| stand | `BEST_alpha_stand.onnx` |
| drive | `BEST_roller.onnx` |
| crouch | `BEST_roller_crouch.onnx` |

- All 9 sha256 values equal `pollen-robotics/microduck-policies@088524a` (§6).
- The contract in `constants.js` is `OBS_SIZE 61`, `CMD_SIZE 13`, `ACTION_SCALE 1.0`. DEFAULT_POSE and joint order are identical to microduck-lab and microduck_rl.
- Ground-pick and crouch take a `[cos, sin, 0]` phase in the velocity command slots.

**Licence:** none declared.
- The card has no `license` field and `package.json` has none (`"private": true`). The README does not state one.
- Assets derive from Pollen repos: Apache-2.0 code, with **CC BY-NC-SA 4.0 for STL meshes** per microduck_rl's `LICENSE-MODELS.md`.
- Treat it as "all rights reserved / upstream licences apply" until clarified (ASSUMPTION).
- Source repos linked from the README: `github.com/pollen-robotics/microduck` and `github.com/pollen-robotics/microduck_rl`.
- Not to be confused with `pollen-robotics/microduck-policy-playground`, whose source is in `microduck/spaces/policy-playground/` and which is linked from `microduck/docs/policy-manifest.md:125`.

**Usefulness as a zero-install reference: high.**
- It is the official Pollen demo: browser MuJoCo WASM plus onnxruntime-web, with the exact shipped policies and the 61-D contract.
- It runs on the Mac with nothing installed.
- It is useful as a ground-truth visual reference for how the official policies should behave, and as a reference implementation of the obs, action and phase-command conventions in JS (`constants.js`, `game.js`).
- Limits:
  - MuJoCo 3.11 WASM differs from the 3.10 that microduck-lab/mjlab use, and it uses the XML servo actuator rather than BAM (ASSUMPTION; not inspected).
  - Not trainable.
  - Last updated 2026-08-28.
  - It loads CDN code at run time.
  - The source could not be cloned here; it is mirrored via HF LFS only.
