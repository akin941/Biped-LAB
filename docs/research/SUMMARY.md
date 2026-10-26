# Upstream research summary (Phase 0)

Snapshot date: 2026-10-26. Revisions: see `upstream.lock.toml`. Detailed, line-cited
reports: [microduck](microduck.md) (robot runtime), [microduck_rl](microduck_rl.md)
(training), [microduck-lab](microduck-lab.md) (community Mac harness + HF Space).

Legend: **V** = verified in source/docs at the pinned sha (or executed here, noted
`ran:cloud`). **U** = unverified / assumption. Nothing from upstream was executed except
loading its MJCF files in our own CPU MuJoCo code.

## Availability check

| Reference | Status 2026-10-26 |
|---|---|
| github.com/pollen-robotics/microduck | V available, Apache-2.0, last commit 2026-10-19 |
| github.com/pollen-robotics/microduck_rl | V available, Apache-2.0 code / CC BY-NC-SA 4.0 STL meshes, last commit 2026-10-12 |
| huggingface.co/spaces/pollen-robotics/microduck-simulator | V running (via HF API); static Vite app, MuJoCo WASM 3.11 + onnxruntime-web; no license declared |
| github.com/jonathanhawkins/microduck-lab | V available, Apache-2.0, single maintainer, last commit 2026-10-23 |
| huggingface.co/pollen-robotics/microduck-policies | V rev `088524a` (apache-2.0, 9 ONNX), via HF API; **download blocked from the cloud container** (proxy 403) |
| mujoco.org / rust-lang.org | reference sites; MuJoCo 3.10.0 wheels exist for macOS arm64 (V, `uv.lock`) |

## Findings by topic

| # | Topic | Finding | |
|---|---|---|---|
| 1 | Mechanics / DoF | ~25 cm, MJCF mass 0.737 kg (README "~800 g"). 15 XL330 servos on the real robot; policies drive **14**: per leg hip_yaw, hip_roll, hip_pitch, knee, ankle (5) + neck_pitch, head_pitch, head_yaw, head_roll (4). Mouth excluded. | V, ran:cloud |
| 2 | Actuators | Dynamixel XL330, protocol 2.0, one 1 Mbps UART, position-P only (kP 200 walk / 160 stand, I=D=0). Knee servos have a −45° offset in EEPROM. Training uses the **BAM** XL330 model (voltage, friction, delay DR), not MJCF `<position>` (kp 0.55, ±0.96 N·m). | V |
| 3 | Sensors | Body IMU LSM6DSV16X on an `imu_to_dxl` board on the servo bus (on-chip fusion); head IMU BMI088/LSM6DSV16X; ToF VL53L5CX/L8CX 8×8; camera IMX219/GC2093; NFC CLRC663. **No LiDAR.** | V |
| 4 | MuJoCo model | `microduck_rl/src/mjlab_microduck/robot/microduck/`: walk / groundcontact / rollers / allcollisions variants + ±1° backlash twins; keyframes INIT, STAND, SIT, FOLD; MJCF dt 0.002 but training/inference override to 0.005 × decimation 4. | V, ran:cloud |
| 5 | Obs / action | **61 → 14**, no history: gyro(3), projected gravity(3), joint pos − default(14), joint vel(14), last action(14), twist(3), head(4), body pose(6). Target = DEFAULT_POSE + action. Same contract in microduck_rl, runtime, HF Space, microduck-lab. Normalizer baked into ONNX. | V (4 sources) |
| 6 | Control loop | 50 Hz, single non-RT tokio thread, no SCHED_FIFO/pinning; docs report 3 missed ticks / 15,022; bus read 3–8 ms of 20 ms. | V (doc numbers U) |
| 7 | PPO / rewards | rsl-rl-lib 5.0.1 via mjlab 1.3.0; actor/critic 512-256-128 ELU; 4096 envs; velocity tracking, upright, pose, air time, head pose, action rate, foot slip, self collision. Some base terms inherited from mjlab, not visible. | V / partly U |
| 8 | Domain randomization | CoM, mass/inertia ±5 %, armature/friction ±10 %, foot friction 0.7–1.3, pushes, IMU mount error ≤ 6°, encoder bias, obs noise, 0–1 step IMU delay, BAM voltage/sag/lag, optional backlash. | V |
| 9 | ONNX | Export via mjlab (needs MuJoCo-Warp env → CUDA, U on Mac). Runtime uses Rust `ort` 2.0.0-rc.11 dlopen-ing ONNX Runtime ≥ 1.23; shapes checked at load. Runtime applies action_scale 0.9 walking × 7.4 V/V_batt, low-pass α 0.7 legs — **differs from sim inference (scale 1.0, no filter)**. | V |
| 10 | Rust runtime | 26-crate workspace (edition 2024, Rust ≥ 1.89); daemons robotd, configd, padd, mediad, btd, updaterd…; JSON-RPC 2.0 over unix sockets; minisign-signed updates with rollback; `robotd --fake` / `--sim` host modes. | V |
| 11 | macOS arm64 | mujoco 3.10 + onnxruntime 1.24.4 have macOS arm64 wheels (V). microduck_rl training **requires CUDA** (README, V); MPS not targeted. microduck-lab trains with CPU MuJoCo + SB3 PPO on Mac (V code, performance claims U). Viewer on macOS needs `mjpython` for `launch_passive` (U). | |
| 12 | CPU / MPS / CUDA | CUDA: mujoco-warp, warp-lang, torch+cu129 (Linux only markers). CPU: mujoco, onnxruntime. MPS: only optional in microduck-lab. | V |
| 13 | microduck-lab | Third-party. Reimplements the 61/14 contract in gymnasium, SB3 PPO with symmetry augmentation, fork-based vec env, Next.js/three.js viewer over WebSocket on 127.0.0.1. Says its own policies are not for the real robot. `setup.sh` runs repo code; one unpinned fetch (G1). | V |
| 14 | Licensing / hardware | Code/MJCF Apache-2.0; **STL meshes CC BY-NC-SA 4.0** (non-commercial, share-alike). No CAD, BOM, or schematics published in these repos; robot sold by Pollen. Policies apache-2.0 (HF card). | V |
| 15 | Gaps / risks | No reference obs/action traces to check sim-vs-real contract; runtime safety thin (±π clamp, no per-joint limits, no e-stop); BAM from a git feature branch; training CUDA-only; several doc-vs-code mismatches; strong Pollen hardware lock-in. | V |

## Own measurements (ran:cloud, Linux x86_64 Xeon 2.1 GHz, 4 vCPU, mujoco 3.10.0)

- Official Microduck `scene.xml` loads; 14 actuators, 0.737 kg — matches research.
- Physics alone: ~18.7k steps/s at dt 0.005 (≈ 93× real time, single thread).
  *Not* a Mac M5 prediction.
- Open-loop STAND hold with MJCF actuators: **falls forward in ~0.9 s** (soft servo
  gains, CoM over ankle). Balance needs closed-loop control — a useful lesson, and a
  characterization test now pins it.

## Consequences for this project

1. We can study the official model and run official policies on **CPU MuJoCo on any
   platform**, without CUDA. Behaviour fidelity depends on porting the BAM actuator
   (backlog WP1-06).
2. Reproducing official *training* on the Mac is not possible unchanged (CUDA). The
   Mac training path is CPU MuJoCo + PPO (as microduck-lab shows), owned by us.
3. Our own robot must not reuse CC BY-NC-SA meshes; design is original (ADR-0004).
4. The runtime's post-processing (action scale, voltage compensation, filter) is part of
   the real policy contract and must be modelled before any sim-to-real claim.
