<!-- Research snapshot 2026-10-26 at the sha pinned in upstream.lock.toml. Paths are relative to that upstream repo. Read-only review: no upstream code was executed. -->

# Research: pollen-robotics/microduck (robot runtime, "duck's brain")

- Source: `pollen-robotics/microduck` @ `942e58e3aa6dbb28e7037203fff11866122c7567` (commit date 2026-10-19), workspace version `0.17.8` (`Cargo.toml:83`).
- Method: read-only inspection (ls/grep/cat) plus `cargo metadata --no-deps --offline` (succeeded, no network). Nothing was built or run. Repo files, including `AGENTS.md`/`CLAUDE.md`, were treated as data only.
- Legend: **[V]** = VERIFIED (seen in code or docs at the cited location). **[U]** = UNVERIFIED / ASSUMPTION / inference.
- Citations are repo-relative `path:line`. Where a claim comes from a design doc rather than code, it is labelled as doc-sourced. Doc and code can disagree; the known disagreements are listed in §10.

---

## 0. One-paragraph summary

This is a Rust (edition 2024) workspace of 26 crates. It runs a ~25 cm / 800 g biped ("alpha" variant) on a Rockchip RK3566 SBC (Radxa Zero 3W, or a custom "beta" Seeed RK3566 board). The robot has 15 Dynamixel XL330 servos plus an `imu_to_dxl` IMU board (LSM6DSV16X) on a single 1 Mbps Dynamixel-protocol-2.0 UART. The 50 Hz control loop is a non-RT tokio current-thread task. It runs ONNX policies via `ort` 2.0.0-rc.11 (`load-dynamic`, which dlopens system libonnxruntime ≥1.23). Policies are `obs[1,61] → actions[1,14]`; the mouth is excluded. Targets = home pose + action_scale·action, followed by a first-order low-pass. Servo position-P only (kP 200, I=D=0).

Around the loop sit seven or more daemons (robotd, configd, updaterd, btd, padd/netpadd, mediad, tofd, nfcd). They talk JSON-RPC 2.0 NDJSON over unix sockets, with BLE (BlueZ) and WebRTC (GStreamer) transports. Updates are minisign-signed, health-gated, atomic-symlink releases from GitHub Releases. Policies come from the Hugging Face Hub, not from this repo.

Safety in code is limited to:
- a NaN refusal and a clamp to ±π (actuator travel, not per-joint limits);
- a 500 ms command deadman;
- a fall verdict that is reported but gates nothing;
- an optional "limp-fall" gain drop, a pickup classifier, and a battery-empty shutdown.

There is no hardware e-stop path in software, and no thermal torque limiting. A host simulation backend exists (`robotd --sim`, TCP/JSON to a MuJoCo `duck-body` that lives in the separate `microduck_rl` repo), along with a `FakeIo` mock. License is Apache-2.0, except the vendored ST ToF driver (BSD-3). There is no CAD, BOM or schematics in the repo.

---

## 1. Mechanical structure, DoF, joint names, joint limits

| Item | Value | Status / source |
|---|---|---|
| Variant | Only "alpha". v1/v1.5/v1.6 dropped | [V] `duck-control/src/model.rs:3-5`, `docs/design/robotd-design.md:10-13` |
| Size / mass | ~25 cm, ~800 g | [V] doc-sourced `README.md:25`. [U] not measured in code |
| Actuated DoF | 15 = 5 per leg ×2 + 5 head/neck/mouth | [V] `duck-control/src/model.rs:11-12` |
| Policy-driven DoF | 14 (mouth, index 9, excluded) | [V] `duck-control/src/model.rs:26-29`, `duck-control/src/obs.rs:51-54` |
| Optional wheels | "roller" mode = different policy set; wheels are passive/add-on per README | [V] `README.md:57`, `robotd-params/src/lib.rs:2117-2122`. [U] wheel hardware not described |

**Joint order** (it is the wire contract, so do not reorder). Defined in `duck-ipc-proto/src/lib.rs:572-588`, with IDs at `duck-ipc-proto/src/lib.rs:596-600`:

| idx | name | DXL ID | home pose (rad) `model.rs:37-53` | MJCF range (rad) `kinematics/assets/alpha/robot_walk.xml` |
|---|---|---|---|---|
| 0 | left_hip_yaw | 20 | 0.0 | [-0.4363, 0.5236] (l.16) |
| 1 | left_hip_roll | 21 | -0.0873 | [-0.3840, 0.3840] (l.21) |
| 2 | left_hip_pitch | 22 | -0.4579 | [-1.5708, 1.5708] (l.26) |
| 3 | left_knee | 23 | -0.0049 | [-1.5708, 1.5708] (l.31) |
| 4 | left_ankle | 24 | 0.4530 | [-1.5708, 1.5708] (l.36) |
| 5 | neck_pitch | 30 | 0.3491 | [-1.5708, 1.0472] (l.48) |
| 6 | head_pitch | 31 | 0.3491 | [-1.5708, 1.5708] (l.53) |
| 7 | head_yaw | 32 | 0.0 | [-2.9671, 2.9671] (l.58) |
| 8 | head_roll | 33 | 0.0 | [-0.4363, 0.4363] (l.63) |
| 9 | mouth | 34 | 0.0 | not in MJCF; code range −5°..+30° (`model.rs:85-86`) |
| 10 | right_hip_yaw | 10 | 0.0 | [-0.5236, 0.4363] (l.79) |
| 11 | right_hip_roll | 11 | 0.0873 | [-0.3840, 0.3840] (l.84) |
| 12 | right_hip_pitch | 12 | 0.4579 | [-1.5708, 1.5708] (l.89) |
| 13 | right_knee | 13 | 0.0049 | [-1.5708, 1.5708] (l.94) |
| 14 | right_ankle | 14 | -0.4530 | [-1.5708, 1.5708] (l.99) |

Notes:
- **Left and right legs are sign-mirrored** in the home pose. A test asserts this (`duck-control/src/model.rs:319-333`) [V].
- **`REST_POSITION`** (seated, torque-off pose) is defined at `duck-control/src/model.rs:62-78` [V].
- **Kinematic tree.** `kinematics/assets/alpha/robot_walk.xml` (109 lines) is a *stripped* MJCF: bodies, hinges, inertials and sites only. It has no geoms, actuators or meshes. Its comment says it was taken "from the same `mjlab_microduck` scene the walking policies are trained in" (`kinematics/src/lib.rs:31-35`) [V]. It defines:
  - sites `imu`, `imu_bno`, `head_imu`, `head_camera`, `tof`, `mouth_tip`, `left_foot`/`right_foot` (`robot_walk.xml:10-12,39,66-71,102`) [V];
  - trunk mass 0.264 kg; the head shell body is 0.173 kg (`robot_walk.xml:8,64`) [V].
- **FK correctness** is tested against MuJoCo `mj_kinematics` on 64 random poses using a committed fixture (`kinematics/src/lib.rs:17-19`, `kinematics/tests/fk_against_mujoco.rs`, `kinematics/tests/fixtures/fk_alpha.json`) [V].
- **Joint limits are NOT enforced at runtime.** Safety clamps only to the actuator range [-π, π] (`duck-control/src/safety.rs:40-48,404`) [V]. The design docs say the "real limits are in the alpha MJCF (31 KB), not vendored here" (`docs/design/robotd-design.md:1147-1150`; same in `safety.rs:42-43`). That is now partly stale: the stripped MJCF *with* ranges is vendored, but only FK reads it [V]. [U] Whether `kinematics` exposes the ranges to safety: not seen.
- **Knee calibration.** Knees (IDs 13, 23) carry `homing_offset = -512` counts (−45°) in servo EEPROM. It is asserted and corrected at startup. Every pose and policy assumes it (`duck-control/src/model.rs:131-150`, `docs/design/robotd-design.md:280-283`) [V].
- **Full MJCF / meshes.** The design doc says the 31 KB full model plus 19 MB of meshes live in the prototype runtime's `scripts/alpha_assets/` (`docs/design/robotd-design.md:1151-1152`) [V, doc-sourced]. `robotctl/assets/duck.bin` is a decimated mesh baked from "the app repository's MJCF" by `scripts/bake-duck-mesh.py` (`robotctl/src/duck.rs:10-23`) [V].

## 2. Servos, actuator control, compute and power hardware

### 2.1 Servos and bus
- **Servo model:** Dynamixel **XL330**, driven through `rustypot` `Xl330Controller` (`duck-control/src/bus.rs:24`) [V]. [U] The exact sub-model (XL330-M288 vs M077) is not stated anywhere seen.
- **Bus:** one UART, `/dev/ttyS2` by default, configurable via `[bus] port` or `--port` (`robotd-params/src/lib.rs:2397`, `deploy/robotd.toml:26-30`) [V]. Exclusive open uses `TIOCEXCL` plus an advisory lock `/run/robotd.sock.lock` (`docs/design/robotd-design.md:44-75`) [V]. Armbian's `serial-getty@ttyS2` must be masked (`docs/design/robotd-design.md:56-59`) [V].
- **Protocol / baud:** Dynamixel protocol 2.0 at 1,000,000 baud (`duck-control/src/model.rs:103`; EEPROM `baud_rate=3`, `model.rs:126`) [V].
- **Fast Sync Read:** instruction 0x8A, on by default (`bus.fast_sync_read`, `robotd-params/src/lib.rs:2398`). It needs **XL330 firmware ≥ v46** and support in the `imu_to_dxl` firmware (`docs/design/robotd-design.md:316-343`). It requires `rustypot` ≥ 1.7 (Cargo pins 1.8.0, `duck-control/Cargo.toml:368-373`) [V].
- **EEPROM asserted and corrected at startup** (`duck-control/src/model.rs:124-129`) [V]:
  - `return_delay_time=0` (the factory value 250 would cost about 8 ms per tick);
  - `baud_rate=3`;
  - `pwm_slope=255`;
  - `shutdown=52`: overload, electrical shock and overheating with the input-voltage bit cleared, so a 2S pack above the 7.0 V Max Voltage Limit does not latch torque off (`model.rs:118-123,233-254`).
- **Auto-adoption of a replacement servo:** a factory servo (ID 1 @ 57,600) is re-addressed when exactly one ID is missing, then rebooted (`docs/design/robotd-design.md:283-291`, `duck-control/src/bus.rs:274-287`) [V].
- **Per tick:** one fast `sync_read` of regs 124–136 (pwm, current, velocity, position; 12 bytes) for the IMU plus the 15 servos, and one `sync_write` of goal positions (`duck-control/src/bus.rs:36-37,514-578`) [V].
- **Once a second:** regs 144–146 (voltage and temperature) (`bus.rs:50-51`; `docs/design/robotd-design.md:298-310`) [V]. Read timeout is 30 ms (`bus.rs:65`) [V].
- **Unit conversions** (`duck-control/src/bus.rs:40,566-568`) [V]:
  - position rad = `2π·count/4096 − π`, so count 2048 = 0 rad;
  - velocity rad/s = `count × 0.229 × 2π/60`;
  - current in mA is taken as abs.

  [U] The write side uses `rustypot`'s own rad→count conversion. I assume it is consistent with the hand-written read conversion but did not verify it.

### 2.2 Actuator control (servo-internal position PD)
- **Position control only.** Goal Position is written each tick. The gain write sets **Position P = kp, I = 0, D = 0** on every servo (`duck-control/src/bus.rs:594-612`; rationale at `docs/design/robotd-design.md:292-296`) [V]. A single kp applies to all 15 joints (`duck-control/src/safety.rs:375,430-437`) [V].
- **Default gains** (XL330 register units) [V]:

  | Gain | Value | Source |
  |---|---|---|
  | Running | 200 | `robotd-params/src/lib.rs:2271`; `safety.rs:75`; `robotd/src/control.rs:76` |
  | Standing | 0.8 × 200 = 160 | `robotd-params/src/lib.rs:2269` (`standing_gain_ratio`) |
  | Limp | 50 | `robotd-params/src/lib.rs:2290` |
  | Limp-fall pose ramp | 160 | `robotd-params/src/lib.rs:2301` |

- **Gain history.** The fleet once ran kP 120 while the default was 160 (`deploy/robotd.toml:44-47`), so the default has changed over time [V].
- **Voltage-adaptive action scale** (on by default): `scale_mult = nominal_voltage / clamp(V_ema, 6.0, 9.5)` with nominal 7.4 V (`robotd/src/main.rs:3524-3532`, `robotd-params/src/lib.rs:1413-1418,2278-2279`) [V].
- **Startup and bring-up:**
  - On start, robotd adopts the current pose and does not touch torque (`docs/design/robotd-design.md:773-779`) [V].
  - `robot.enable` then runs `Limp → Homing (torque on, 2 s linear ramp) → Ready` (`docs/design/robotd-design.md:785-787`; `HOME_RAMP = 2 s` at `robotd/src/main.rs:115`) [V].
  - A posture check picks a sit vs stand start (`robotd/src/posture.rs`, `docs/design/robotd-design.md:483-498`) [V].

### 2.3 Compute
- **SBC.** The target is the Radxa Zero 3W (RK3566, 4× Cortex-A55, aarch64) running Armbian 26.2.x / Debian 13 Trixie (`.cargo/config.toml:3,14-15`; `robotd-params/src/board.rs:30-33`) [V].
- **Second board, "beta"** ("the custom board's first spin: Seeed's RK3566 main board", `hw_rev=2`) (`robotd-params/src/board.rs:34-35,59-63`) [V]. Boards are declared via `[board] version`, not auto-detected (`board.rs:3-9`) [V].
- **Thermal throttling** on the Zero 3 at about 95 °C pins the clock at 408 MHz of 1800 (`docs/design/robotd-design.md:355-359`) [V, doc-sourced].
- **NPU (RK3566 RKNN):** used only by the `duck-detect` vision detector. Runtime pinned at `rknn-toolkit2 v2.3.2` (`Cargo.toml:41-50`, `deploy/overlays/rk3568-npu-enable.dts`, `docs/project/npu-bringup.md`) [V].
- **MCU:**
  - The `imu_to_dxl` v2 board is a Dynamixel-bus slave (ID 200) that serves the IMU registers (`duck-control/src/model.rs:99-101`, `duck-control/src/imu.rs:1-17`) [V].
  - Its MCU model and firmware are **not in this repo** [U]. A `imu_to_dxl fw 6` bug is mentioned (`robotd/src/head_imu.rs:16-19`).
  - Peripheral firmware is "flashed at production", not OTA (`docs/design/updater-design.md:1205-1210`) [V].
  - On beta, the body IMU sits on the "power board" (`duck-control/src/imu.rs` `BETA_MOUNT` doc) [V].

### 2.4 Power / battery
- **Pack:** 2S Li-ion, NP-F550 type (`duck-control/src/model.rs:170`) [V]. Servos are fed straight from the pack, with no regulated 5 V rail (`model.rs:164-168`) [V].
- **No fuel gauge and no ADC.** Battery is estimated from the mean servo-reported voltage, filtered by an EMA over about 10 s (`model.rs:157-162`; `docs/design/robotd-design.md:305-308,655-658`) [V].
- **Battery percent:** a linear map from 6.6 V (empty) to 8.2 V (full) (`model.rs:171-191`) [V].
- **Empty pack:** `safety.battery_empty_shutdown` (default on) sits the robot down and runs `systemctl poweroff` at 6.6 V (`docs/design/robotd-design.md:655-658`; `robotd/src/main.rs:1079-1082`) [V].
- Board power circuitry and BMS: not documented [U].

## 3. Sensors and interfaces

| Sensor | Part | Interface | Rate | Status / source |
|---|---|---|---|---|
| Body IMU | LSM6DSV16X on `imu_to_dxl` v2 board (ID 200); on-chip SFLP game-rotation quaternion and gyro-bias estimation | Dynamixel bus (UART 1 Mbps), same sync_read as servos; 12-byte block @ reg 124 (gyro i16 ±500 dps, quat xyz as fp16) | read at loop rate 50 Hz. Chip rate [U]; "~0.25 s at 100 Hz" readiness comment implies ≥100 Hz | [V] `duck-control/src/imu.rs:1-17,47,103-108` |
| Head IMU (zero3) | BMI088 on the HAT (accel 0x19, gyro 0x68), Madgwick fusion on the CPU | I²C (`i2c3`, shared with ToF and audio codec), read by `tofd` | 100 Hz default (`imu_hz`), off-switchable | [V] `tof/src/imu.rs:1-19`, `tof/src/main.rs:133-135`, `docs/project/tof-on-demand.md` |
| Head IMU (beta) | LSM6DSV16X on the face-board I²C, SFLP fusion on chip, FIFO | I²C, read by `robotd` | 60 Hz sample, drained at 15 Hz | [V] `robotd/src/head_imu.rs:1-30,36-40` |
| ToF | ST VL53L5CX or VL53L8CX, 8×8 zones (detected at runtime) | I²C (`i2c3`, addr 0x29 then 0x52) on zero3 HAT; SPI (spidev, L8CX) on beta | 15 Hz default | [V] `tof/src/lib.rs:1-12`, `tof/src/main.rs:109-121` |
| Camera | IMX219 (zero3, mounted 90°), GC2093 (beta, 0°) | [U] MIPI-CSI (V4L2 + rkaiq ISP) | per `[media] quality` | [V] `robotd-params/src/board.rs:66-87`, `scripts/setup-rkaiq.sh`, `mediad/src/pipeline.rs` |
| NFC | NXP CLRC663 reading NTAG215 (driver also has `st25r100.rs` and `spi.rs`) | USB serial (CLRC663); SPI module present | ~5 Hz poll ("five times a second") | [V] `nfc/src/lib.rs:1-3,13-23,38` |
| Microphone / speaker | TLV320AIC3104 codec (`plughw:aic3104`) | I²C control on i2c3 (0x18) + I²S [U] | — | [V] `robotd-params/src/lib.rs:1259`, `deploy/audio/aic3104-i2c3.dts`, `tof/src/imu.rs:3-4` |
| Servo telemetry | position, velocity, current (per tick); voltage, temperature (1 Hz); hardware error reg 70 | Dynamixel bus | 50 Hz / 1 Hz | [V] `duck-control/src/bus.rs:34-61` |
| SoC thermal | max of sysfs thermal zones + cpufreq ceiling and cooling state | sysfs | 1 Hz | [V] `robotd/src/soc.rs`, `docs/design/robotd-design.md:345-366` |
| Gamepad | Xbox Wireless, Switch-Pro clones (optional pad IMU) via `gilrs` | BLE/USB HID; or UDP (`netpadd`, port 4210) | intents ≤50 Hz | [V] `padd/Cargo.toml`, `robotd-params/src/lib.rs:188-230`, `configd/src/bluez.rs:167` |
| LiDAR | none | — | — | [V] no LiDAR code found (grep) |
| Face LEDs (beta) | 12 LEDs on a GPIO expander, `/sys/class/leds/face:*` | sysfs | — | [V] `docs/design/architecture.md:349-366` |
| Wi-Fi radio | AIC8800 (zero3) | SDIO [U] | — | [V] `robotd-params/src/board.rs:30-31` |

IMU processing details (all [V]):
- Gyro and gravity each pass through a 3-sample median for spike rejection (`imu.rs:150-168`). There is no other filter.
- Projected gravity = rotate_inverse(q, [0,0,−1]), normalised *before* the median. During transients the vector can therefore be slightly shorter than unit length (`imu.rs:140-147`).
- Upright gravity reads `[0,0,−1]` (`imu.rs:29-30`).
- The sensor→trunk mount is a +90° rotation about Y on zero3 and identity on beta (`imu.rs:75-91`; selected per board at `robotd/src/main.rs:1423,1448`).
- A stale-sample detector warns after 25 identical blocks (`bus.rs:85,535-543`).
- There is no per-device IMU calibration; it is deferred (`docs/design/robotd-design.md:1135`).

## 4. Control loop: frequency, timing, threads, RT measures

- **Rate:** `[control] hz = 50` (validated between 1 and 1000) (`robotd-params/src/lib.rs:2406,2443`, `deploy/robotd.toml:49-56`) [V]. The rate was inherited from a Pi Zero 2W prototype and "has never been re-derived on the Radxa" (`deploy/robotd.toml:51-54`; `docs/design/robotd-design.md:1142-1143`) [V].
- **Threads** [V]:
  - The loop runs on a dedicated OS thread named `"control"` with its own `tokio` current-thread runtime, so IPC runs elsewhere (`robotd/src/main.rs:1226-1238`).
  - The tick is `tokio::time::interval(period)` with `MissedTickBehavior::Skip` (`robotd/src/main.rs:2145-2160`). The comments record measurements: `Delay` gave 43.1 Hz, and a real bus read costs 3–8 ms.
  - Other threads: voice/sound, head IMU, and IPC on separate runtimes (`main.rs:1325,2788,4243`).
- **No RT measures, by design.** There is no `SCHED_FIFO`, CPU pinning or `mlockall` (`docs/design/robotd-design.md:1089-1095`) [V]. The systemd unit has no `CPUSchedulingPolicy`, `Nice` or `CPUAffinity` (`robotd/systemd/robotd.service`) [V]. The kernel is stock Armbian vendor/non-RT [V] (`docs/project/roadmap.md:106-108`).
- **Measured on hardware:** "the loop holds 50.0 Hz on a non-RT kernel (`missed=3` in 15022 ticks)" (`docs/project/roadmap.md:106-108`) [V, doc-sourced].
- **Tick budget (20 ms):**

  | Cost | Value | Source |
  |---|---|---|
  | Bus read | 3–8 ms | `robotd/src/main.rs:2155` [V] |
  | Policy inference | 0.029 ms (measured natively on x86) | `docs/design/simulation.md:189` [V] |
  | Pickup classifier | "a few tenths of a ms" | `docs/design/robotd-design.md:646-648` [V] |

  [U] I did not find an aarch64 inference timing number for the board. `duck-control/examples/pickup-bench.rs` measures it.
- **Inference threads:** `ort` intra-op threads = 1, graph optimisation Level3 (`duck-control/src/policy.rs:30-36,594-597`) [V].
- **Allocation:** a 61-float Vec is allocated per inference. Pre-binding I/O was explicitly not done (`docs/design/robotd-design.md:535-537`) [V].
- **Tick order:**
  1. `read` (one fast sync read);
  2. `safety.observe` (fall verdict);
  3. deadman gate;
  4. command EMA;
  5. `Observation::build`;
  6. `Policy::infer`;
  7. scale + low-pass;
  8. `safety.apply` (NaN refusal / clamp, then sync_write);
  9. publish atomics and state frame.

  Sources: `docs/design/robotd-design.md:145-226`, `robotd/src/control.rs:7-16` [V].
- **Read failure handling:** a failed read means `sensors = None`, so that tick does not drive and holds the pose (`docs/design/robotd-design.md:205-232`) [V].
- **Update-gate thresholds:** `min_achieved_hz = 45`, `stall_periods = 25`, `max_consecutive_errors = 10` (`robotd-params/src/lib.rs:2415-2422`) [V].
- **Logging:** a five-minute summary at info level, per-tick output at debug (`docs/design/architecture.md:545-552`) [V].

## 5. ONNX policy loading and inference (sim-to-real contract)

### 5.1 Runtime / crate
- **Crate:** `ort = "=2.0.0-rc.11"`, `default-features=false`, feature `load-dynamic` (`duck-control/Cargo.toml:366`) [V]. It is not `tract`.
- **Runtime library:** libonnxruntime is dlopened. The search order is `ORT_DYLIB_PATH`, then `libonnxruntime.so` (`duck-control/src/policy.rs:93-108`) [V].
- **Version pins:** floor 1.23, install target 1.28.0 (`Cargo.toml:27-34`) [V]. onnxruntime is a board prerequisite installed by `scripts/setup-board.sh` / `install.sh`, not shipped in the release (`docs/design/robotd-design.md:530-533`) [V].
- **Panic guard:** `ensure_runtime()` probes the dylib with `libloading` before touching `ort`, because `ort` panics when the library is missing; `ort` panics are also caught (`duck-control/src/policy.rs:110-127`, `docs/design/robotd-design.md:516-522`) [V].
- **Other ONNX users:** `ort` is also used by `pet-detect` (an audio classifier, `pet-detect/models/pet_detect.onnx`) and `duck-detect` (ONNX or RKNN) [V].

### 5.2 Model contract (validated at load, warm-up inference with an all-zero obs)
- **Feed-forward:** input named `obs` `[1,61]` f32 → one output `[1,14]`, preferably named `actions` (`duck-control/src/policy.rs:605-625` region; `docs/recurrent-policies.md:10-15`) [V].
- **LSTM:** inputs `obs`, `h_in`, `c_in` `[layers,1,hidden]` → outputs `actions`, `h_out`, `c_out`. State is at most 1,048,576 elements per tensor. It requires manifest `model_api: 2` (`docs/recurrent-policies.md:10-27`, `duck-control/src/policy.rs:486-491`) [V]. GRU is not supported [V].
- **Batch:** may be symbolic; inference always uses batch 1 [V].
- **Normalisation:** "Observation normalization must be baked into the exported model" (`docs/recurrent-policies.md:5`) [V]. **The runtime applies no obs scaling or normalisation** (`duck-control/src/obs.rs:174-230`) [V].
- **Shape mismatch:** a wrong shape is refused, e.g. "observation width is 51, expected 61" (`docs/design/robotd-design.md:508-514`) [V].
- **Cross-checks:** `OBS_LEN=61` and `ACTION_LEN=14` are const-asserted against `duck_ipc_proto::POLICY_OBS_LEN/POLICY_ACTION_LEN` (`duck-control/src/obs.rs:60-63`, `duck-ipc-proto/src/lib.rs:482,486`) [V].
- **Hub manifest:** the manifest may declare `obs_len`, `action_len`, `model_api`, `robot.model="microduck"`, `robot.servos="xl330"` and `robot.control_hz=50` (`docs/policy-manifest.md:41-50`) [V].

### 5.3 Observation vector (61 × f32), as built by `Observation::build` (`duck-control/src/obs.rs:12-32,174-230`) [V]

| idx | width | content | units / notes |
|---|---|---|---|
| 0..3 | 3 | gyro, trunk frame | rad/s, after mount rotation and 3-sample median. No scaling |
| 3..6 | 3 | projected gravity, trunk frame | unit vector, upright = (0,0,−1); 3-sample median |
| 6..20 | 14 | joint pos − DEFAULT_POSITION (mouth skipped) | rad; positions from the servo encoder (knee offset in EEPROM) |
| 20..34 | 14 | joint velocity (mouth skipped) | rad/s, raw servo `present_velocity` (0.229 rpm/count). Not filtered or scaled |
| 34..48 | 14 | previous action | **raw policy output before action_scale**, in 14-slot order (`obs.rs:172-173,206`) |
| 48..51 | 3 | vx, vy, vyaw | m/s, m/s, rad/s, after EMA (below) |
| 51..55 | 4 | neck_pitch, head_pitch, head_yaw, head_roll targets | rad, after EMA |
| 55..57 | 2 | body x, y | **always 0** ("unbound in training") |
| 57..60 | 3 | body **z, roll, pitch** (this order) | 0 unless body-pose mode (B button) |
| 60 | 1 | body yaw | **always 0** |

Slot mapping: policy slot *s* → joint *s* if *s* < 9, else *s*+1 (`duck-control/src/obs.rs:116-125`) [V]. The 14-slot joint order is therefore L-leg (5), neck/head (4), R-leg (5).

Head targets ride **in the command block only**. They are **not** added to the actions (`obs.rs:40-43`) [V].

### 5.4 Command encoding per network (`robotd/src/control.rs:588-648`) [V]

| Net | Twist slots | Head / body |
|---|---|---|
| walk / stand | (vx, vy, vyaw); stand is selected when \|twist\| ≤ 0.05 and a stand network is loaded (`policy.rs:27`). The default walk-mode set loads **no** stand network, so `velstand.onnx` runs at all speeds (`robotd-params/src/lib.rs:2100-2110`) | live |
| sit↔stand (`alpha_sitstand.onnx`) | vx = posture flag (1 = sit, 0 = stand), vy = vyaw = 0 | live |
| ground pick | `[cos 2πφ, sin 2πφ, 0]`, period 4 s (walk) / 3 s (roller), ends at φ = 0.7 | zeroed |
| one-shot skills (kicks, roulade, Hub skills) | constant twist from config (usually 0) | zeroed ("zero_command_padding") |

Priority: skill > ground pick > sit/rise > stand-by-magnitude > walk (`control.rs:11`) [V].

### 5.5 Action → target pipeline (`robotd/src/control.rs:651-740`) [V]
1. `action = infer(obs)`; `last_action = action` (raw).
2. Choose scale and gain [V]:

   | Case | action_scale | kp |
   |---|---|---|
   | Walk mode (default) | 0.9 | 200 |
   | Roller mode (default) | 0.8 | 200 |
   | Standing-tuned (Stand net, or skill/sitstand with ≈0 command) | 1.0 | 160 (ratio 0.8) |
   | SitStand | 1.0 | — |
   | Ground pick | 1.0 (walk) / 0.8 (roller) | — |
   | Per-skill override | manifest `action_scale` | — |

   Sources: `robotd-params/src/lib.rs:2162-2189`, `control.rs:666-715`.
3. `scale *= scale_mult` (voltage adaptation, nominal 7.4 V / measured EMA) (`robotd/src/main.rs:3528-3532`).
4. `target[j] = DEFAULT_POSITION[j] + scale · action_scattered[j]`. The mouth slot gets 0 and is later overwritten by mouth control (`control.rs:717-721`).
5. **First-order low-pass**, `y = α·x + (1−α)·y_prev`. Head joints 5..8 use α = 0.5; legs use α = 0.7; the mouth is skipped. It is skipped on the first tick after a reset, when `previous = None` (`control.rs:723-739`; defaults `robotd-params/src/lib.rs:2169-2170`). α ≥ 1 disables it. A comment says "The alpha policies are trained with 0.5 / 0.7 — it must match training" (`control.rs:63-67`).
6. Safety: refuse any NaN (hold the pose), then clamp each target to [−π, π] (`duck-control/src/safety.rs:363-428`).

### 5.6 Command smoothing (before the obs)
- Per tick, `ema += α·(target − ema)`. Twist and body use `cmd_alpha = 0.2`; head uses `head_alpha = 0.2` (`robotd/src/main.rs:1715-1719,2186-2187,3125-3140`; `robotd-params/src/lib.rs:2407-2408`) [V].
- Gamepad full-stick ranges: vx ±0.4 m/s, vy ±0.3 m/s, vyaw ±1.0 rad/s (`robotd-params/src/lib.rs:143-157`) [V].

### 5.7 Reset semantics
- `Controller::reset()` zeroes `last_action`, drops the low-pass anchor, and resets LSTM state. It runs on every start of driving (`robotd/src/control.rs:306-315`, `docs/design/robotd-design.md:212-215`) [V].
- From a seat, `last_action` is seeded from the held pose instead of zero (`control.rs:512-523`) [V].

### 5.8 Policy sourcing
- Official set: Hugging Face `pollen-robotics/microduck-policies`, minimum version `v8` (`Cargo.toml:52-67`). It is installed to `/opt/robot/policies/current` (`robotd-params/src/lib.rs:45`) [V].
- Default files:
  - walk mode: `velstand.onnx`, `alpha_sitstand.onnx`, `alpha_ground_pick.onnx`, plus `ball_kick_left/right.onnx` and roulade (`robotd-params/src/lib.rs:1861-1862,2100-2122`);
  - roller mode: `roller.onnx`, `roller_crouch.onnx`.
- **No gait policies are committed in this repo.** The only committed models are `duck-control/models/pickup_detector.onnx` (held-in-hand classifier) and `pet-detect/models/pet_detect.onnx`, plus test fixtures in `duck-control/tests/fixtures/` (generated by `generate.py`) [V].
- Training happens in `pollen-robotics/microduck_rl` (MuJoCo/mjlab + PPO / rsl_rl) (`README.md:29-31`, `docs/recurrent-policies.md:3-4`) [V].
- [U] Whether training used exactly the same IMU median, velocity source and EMA is claimed by comments but not verifiable from this repo. "Golden observation vectors" from training are listed as wanted but **absent** (`docs/design/robotd-design.md:1106-1108,1144-1146`) [V].

## 6. Process architecture, IPC, communication, updates

### 6.1 Daemons (one crate each, systemd units under `<crate>/systemd/`) [V] (`docs/design/architecture.md:18-101`, `CONTRIBUTING.md:76-93`)

| Daemon | Role | Endpoint |
|---|---|---|
| `robotd` | only motor writer: loop, policies, safety, health, voice/chorale/theremin, odometry, head FK, LEDs | `/run/robotd.sock` |
| `configd` | wifi via NetworkManager D-Bus, name/identity from SoC serial, pairing PIN, gamepad bonding (BlueZ), reboot/power | `/run/configd.sock` |
| `updaterd` (`updater/`) | verify, install, swap, health-gate, roll back releases and policy components | `/run/updaterd.sock` |
| `btd` | BLE GATT transport (bluer 0.17.4) for a subset of the API | GATT service (`duck-ble` UUIDs) |
| `padd` / `netpadd` | gamepad → intents (gilrs) / UDP pad on LAN; mutually exclusive via `ExecCondition` | `/run/padd/pad.sock`; UDP 4210 |
| `mediad` | camera/mic, GStreamer `webrtcsink`, `mpph264enc` HW H.264, HTTP console `:8080`, signalling `:8443`, remote relay (HF rendezvous, TURN) | `/run/mediad/media.sock` |
| `tofd` (`tof/`) | ToF (+ zero3 head IMU) publisher | `/run/tofd/tof.sock` |
| `nfcd` (`nfc/`) | NFC tag → gamepad pairing | — |
| CLIs | `robotctl` (on-board, ratatui monitor), `duckctl` (laptop, BLE via btleplug; not in default-members) | — |

- Socket paths: `duck-ipc-proto/src/lib.rs:506-537` [V].
- Recovery-path invariant: `configd`, `updaterd` and `btd` have no systemd dependency on `robotd` (`docs/design/architecture.md:142-152`) [V].

### 6.2 IPC
- **Protocol:** JSON-RPC 2.0, NDJSON (one object per line), over unix sockets. Framing is `tokio_util::codec::LinesCodec`. There is no broker (`docs/design/architecture.md:187-201`) [V].
- **Contract crate:** `duck-ipc-proto` (serde, serde_json and semver only; 6,940 lines), `API_VERSION = 44` (`duck-ipc-proto/src/lib.rs:473`) [V]. `robotd` `MODEL_API = 2` (`robotd/src/main.rs:64`) [V].
- **Message families:** continuous intents are notifications (`robot.move` {vx,vy,vyaw}, `robot.head`). Discrete calls are requests (`robot.stop/enable/init/relax/do/pose/mouth/...`). Telemetry is `robot.subscribe` → `robot.state` notifications, drop-on-lag (`docs/design/robotd-design.md:476-501`; method list `duck-ipc-proto/src/lib.rs:642-939`) [V].
- **Conventions:** radians, trunk frame, right-handed (`docs/design/robotd-design.md:492-496`) [V].
- **Authorisation:** socket mode 0660 plus group, and `SO_PEERCRED` uid/gid allow-lists for mutating calls (`docs/design/architecture.md:219-236`) [V].
- **Mapping telemetry:** `t_ns` is CLOCK_MONOTONIC; `imu{gyro,quat}` and camera/ToF frame poses are also carried (`docs/design/robotd-design.md:1160-1179`) [V].

### 6.3 Communication
- **BLE** via `btd`. It is unencrypted by default; the pairing PIN defaults to `000000` and is public (`docs/project/roadmap.md` "Where we are" and M6) [V].
- **Wi-Fi** via NetworkManager. A board is migrated from netplan by `scripts/migrate-network.sh` (`docs/design/architecture.md:322-329`) [V].
- **WebRTC** uses two data channels: `control` (reliable) and `teleop` (unreliable) (`docs/design/architecture.md:441-452`) [V].
- **Remote access:** a Hugging Face account device-flow plus a rendezvous service (`docs/design/remote-access-design.md`) [V].
- **Wired Ethernet:** none [V]. `duck-ether` is a *simulated* BLE ether for sim ducks (`duck-ether/Cargo.toml` description).

### 6.4 Updates
- **Releases:** whole `.tar.zst` directories under `/opt/robot/daemon/releases/<ver>/`. The `current` symlink is swapped (`docs/design/architecture.md:62-66`) [V].
- **Signing:** minisign, verified on-robot with `minisign-verify`; publishing is done by `cargo xtask package|sign|promote` (`xtask/src/main.rs:1-33`) [V]. Trusted keys are `deploy/trusted_keys/release-{1,2,3}.pub` plus a dev key `deploy/dev-key/team.dev.pub` [V].
- **Hosting:** GitHub Releases (the repo is public so robots need no token), and an HF source for policies (`docs/design/updater-design.md:486-550`) [V].
- **Health gate:** after apply, `robot.health` is polled; unhealthy rolls back and degraded commits. A boot counter covers two boots, there is a golden release, and a boot-time rescue timer runs (`robot-boot-check`, `robot-rescue`) (`docs/design/updater-design.md:678-716,796-804`, `scripts/robot-boot-check`, `scripts/robot-rescue`) [V]. The boot net has "never been exercised on a board" (`docs/project/roadmap.md` "Where we are"; M6) [V].
- **Board gating:** `hw_rev` / `min_hw_rev` (`robotd-params/src/board.rs:55-101`) [V].
- **OS updates** are out of scope for OTA (`docs/design/updater-design.md:1194`). A separate `microduck_yocto` image exists (`CONTRIBUTING.md:11-15`) [V].

## 7. Safety

| Mechanism | Behaviour | Default | Source |
|---|---|---|---|
| Single writer | only `Safety` holds `RobotIo`; enforced by ownership | — | [V] `duck-control/src/safety.rs:1-10`, `docs/design/robotd-design.md:247-254` |
| NaN / Inf refusal | whole tick refused; hold pose | always | [V] `safety.rs:376-392` |
| Range clamp | every target clamped to [−π, π] (actuator travel). **No per-joint, velocity, acceleration or torque limit** | always | [V] `safety.rs:40-48,396-422`; rate limits deferred `docs/design/robotd-design.md:1133-1135` |
| Deadman | intent age > 500 ms zeroes the twist only (head kept); robot stands, does not go limp | 500 ms | [V] `safety.rs:310-336`, `robotd-params/src/lib.rs:2289` |
| Fall verdict | gravity z > −0.5 for 200 ms → `fallen` published; **gates nothing** | on | [V] `safety.rs:22-31,72-73`, `docs/design/robotd-design.md:566-574` |
| Limp-fall | predictor ġz = −(ωx·gy − ωy·gx) extrapolated 300 ms; tilt z > −0.90, predicted > −0.5, 60 ms debounce → gain 50, wait still (<1 rad/s for 200 ms, max 1.5 s), 0.6 s ramp to stand at kp 160, hand back | on | [V] `duck-control/src/fall.rs:1-40`, `robotd-params/src/lib.rs:2292-2301` |
| Pickup pause | ONNX classifier on 1 s window; p > 0.8 for 100 ms pauses to "pause pose"; p < 0.35 for 80 ms resumes | on | [V] `docs/design/robotd-design.md:609-653`, `robotd-params/src/lib.rs:1120-1127` |
| Battery-empty shutdown | at 6.6 V EMA: sit, then poweroff | on | [V] `docs/design/robotd-design.md:655-658` |
| Servo HW protection | `shutdown=52` latches overload, electrical shock and overheating in servo firmware (XL330 overheat ~70 °C) | EEPROM | [V] `duck-control/src/model.rs:124-129`, `duck-ipc-proto/src/vetbook.rs:26-29` |
| Thermal | servo and SoC temperatures are **reported only** (vet book counts time ≥ 60 °C); no derating or torque limit; "thermal limits" deferred | — | [V] `docs/design/robotd-design.md:836-848,1133-1135`, `duck-ipc-proto/src/vetbook.rs:26-29` |
| E-stop | no software e-stop over BLE (`robot.stop` deliberately refused on BLE); "A real e-stop is physical". Pad: held Select cuts torque then powers off | — | [V] `btd/src/route.rs:485-491`, `mediad/src/route.rs:92-94`, `robotd-params/src/lib.rs:253`. [U] Whether a physical e-stop exists on hardware is not documented |
| Watchdog | design calls for systemd `WatchdogSec` on robotd, but **the unit has none and no sd_notify exists**; health is via atomics + updater gate | — | [V] `docs/design/updater-design.md:683-684` vs `robotd/systemd/robotd.service` (grep: no `WatchdogSec`/`sd_notify` in repo) |
| Restart safety | `robot.safeToRestart` is false while moving; restart never moves the robot | — | [V] `docs/design/robotd-design.md:773-779,857-858` |
| Authority arbitration | specified but unfinished; `robot.enable` not classified mutating | — | [V] `docs/project/roadmap.md` M6 bullets |

## 8. Build system, toolchain, targets, simulation and test support, CI

- **Toolchain:**
  - MSRV `rust-version = "1.89"`, edition 2024, resolver 3 (`Cargo.toml:6,84-95`) [V].
  - Ceiling: must build with Rust **1.94.1** (Yocto/OE wrynose) (`Cargo.toml:86-93`, `CONTRIBUTING.md:11-15`) [V].
  - No `rust-toolchain(.toml)` file; CI uses `dtolnay/rust-toolchain@stable` (`.github/workflows/ci.yml:51`) [V].
- **Cross-compiling:**
  - Target `aarch64-unknown-linux-gnu`, with glibc pinned at 2.31 through `cargo zigbuild`. The alias is `cargo board` = `zigbuild --release --target aarch64-unknown-linux-gnu.2.31` (`.cargo/config.toml:1-24`) [V].
  - Release profile: there is no custom `[profile.release]` and binaries are not stripped (`Cargo.toml:111-115`) [V].
  - Other paths: `scripts/cross-sysroot.sh`; `scripts/dev-push.sh --docker` with `scripts/dev-build.Dockerfile` [V].
- **Crates:** `cargo metadata` lists 26 workspace packages. `default-members` excludes `duckctl` (per comment) and also `duck-ether`, and lists `duck-ble` twice (`Cargo.toml:5-21`) [V].
- **xtask:** `package`, `sign`, `keygen`, `promote` (release tooling, minisign) (`xtask/src/main.rs:13-19,59-127`) [V].
- **Host builds:**
  - macOS: `cargo test --workspace` claims 942 tests pass with nothing excluded (`CONTRIBUTING.md:36-41`) [V, doc-sourced].
  - Linux x86_64: needs `libudev-dev` and the GStreamer dev packages (`CONTRIBUTING.md:26-34`; CI `ci.yml:64`) [V].
  - Linux-only code: `configd` NetworkManager, `btd` BlueZ, `tofd` C ULD (`tof/build.rs`), I²C IMU, `mediad` V4L2 (`CONTRIBUTING.md:41-60`; `tof/src/lib.rs:28-33`) [V].
  - Git dependency: `bmi088` from `github.com/pollen-robotics/bmi088-rs` tag v0.1.2 (`tof/Cargo.toml:45`) [V]. Everything else comes from crates.io, including `hf-robot-account` 0.1 (`Cargo.lock:2097-2100`) [V].
- **Mock hardware:** `FakeIo` is always compiled, uses scripted or echo samples, and can freeze or fail on demand (`duck-control/src/io.rs:214-219`). It is reached with `robotd --fake` (`robotd/src/main.rs:248-251`) [V]. Other fakes: `tofd --fake`, `configd --fake-net` (roadmap) [V].
- **Host simulation backend:**
  - `robotd --sim host:port` uses `duck_control::sim::RemoteIo`, which speaks TCP with NDJSON (`hello` / `read` / `write` / `gain` / `torque` / `slow`, protocol 1, 200 ms timeout, TCP_NODELAY, reconnect each tick) (`duck-control/src/sim.rs:1-80`, `robotd/src/main.rs:253-263`, `docs/design/simulation.md:96-123`) [V].
  - The MuJoCo body server `duck-body` (with BAM actuator models fitted to real XL330s) lives in **`microduck_rl`**, not here (`docs/design/simulation.md:3-8,228-229`; `docs/robot/simulation.md:22,34,161`) [V].
  - The sim runs wall-clock, not lockstep; below 1.0× realtime the policies misbehave (`docs/robot/simulation.md:174-179`) [V].
  - Harness: `scripts/duck-sim` (single duck, or N in systemd-nspawn containers); `duck-ether` provides a simulated lossy radio [V].
  - The sim does not exercise the Dynamixel driver, BLE, ISP, NPU, encoder, thermals or battery (`docs/design/simulation.md:231-232`) [V].
- **test-support crate:** it is **updater release fixtures** (signed-release publisher), not hardware mocks (`test-support/src/lib.rs:1-23`) [V]. Other test assets: `duck-control/tests/*` (recurrent policy, pickup), `kinematics/tests/fk_against_mujoco.rs`, `robotd/tests/{updater_gate,single_instance}.rs`, `updater/tests/*` [V].
- **CI** (`.github/workflows/`, 10 files) [V]:
  - `ci.yml`, ubuntu-latest: fmt, `clippy --workspace --all-targets`, `test --workspace`, installer and board-check lint, xtask smoke test.
  - `ci.yml`, `ubuntu-24.04-arm` job: zig, cargo-zigbuild, aarch64 sysroot, then `scripts/board-test.sh` (60 assertions on a Debian 13 userland).
  - `ci.yml`, coverage job: `cargo-llvm-cov` (`ci.yml:47-68,179-237,243-310`).
  - Other workflows: `release.yml` / `_build-release.yml` / `promote.yml` / `_promote-release.yml` (signed releases), `dev.yml` (branch builds), `duckctl-pypi.yml` / `_build-duckctl.yml` (duckctl to PyPI), `prune-dev-releases.yml`, `turn-endpoint.yml`.
  - There is no macOS CI job (`docs/design/simulation.md` §9) [V].

## 9. Licensing, hardware design availability, attribution

- **License:** Apache-2.0 for the workspace (`LICENSE`, `Cargo.toml:95`; every crate has `license.workspace`) [V].
- **Vendored C:** ST VL53L5CX/VL53L8CX Ultra Lite Drivers under BSD-3-Clause (`tof/vendor/LICENSE.txt`, `tof/src/lib.rs:14-26`) [V]. The firmware blobs (~550 KB) inside the ULD are also vendored [V].
- **Ports from Pollen repos:** NFC driver from "the winnie repo" `ntag663` (`nfc/src/lib.rs:3`). Constants, bus numbers and kinematics come from `apirrone/microduck_runtime` and `microduck_kinematics_rs` (`docs/design/robotd-design.md:7-8`, `kinematics/Cargo.toml:1-9`) [V]. [U] Those upstream licenses were not checked.
- **Hardware design files:** **not in the repo.** There is no CAD (STEP/STL), BOM, schematics or PCB files (find/grep found none) [V].
  - Only the stripped kinematic MJCF and a decimated render mesh (`robotctl/assets/duck.bin`) are present [V].
  - A HAT schematic is referenced but not included (`tof/src/imu.rs:3`; `nfc/src/st25r100.rs:5` "schematic sheet 23") [V].
  - Device-tree overlays for audio and NPU are included (`deploy/audio/*.dts`, `deploy/overlays/rk3568-npu-enable.dts`) [V].
  - The product is sold ("Get yours here", `README.md:31-33`). The relation to "Open Duck" CAD is only mentioned in `docs/ideas/autonomous_behavior.md:83` [V].
- **Attribution needs** [U, my recommendation]:
  - Keep the Apache-2.0 LICENSE and NOTICE-equivalent attribution if code is copied.
  - Keep the BSD-3 notice for anything derived from the ST ULD.
  - Do not reuse Pollen release signing keys, the HF repos or the rendezvous endpoints.
  - Policies and the pickup model are trained on Pollen's hardware model. Their Hub license should be checked separately.

## 10. Gaps, missing docs, risks, Pollen-specific items

**Doc vs code discrepancies found** [V]:
1. *WatchdogSec.* The updater design calls for `WatchdogSec` on robotd (`docs/design/updater-design.md:683-684`). Neither the unit nor the code implements it.
2. *Body pose smoothing.* `deploy/robotd.toml:60-61` says `head_alpha` smooths "head targets and the body pose". The code applies `cmd_alpha` to the body pose (`robotd/src/main.rs:3135-3138`).
3. *Joint limits.* The design says joint limits are "not vendored" (`docs/design/robotd-design.md:1147-1152`). A stripped MJCF with ranges *is* vendored (`kinematics/assets/alpha/robot_walk.xml`), but it is unused for safety.
4. *default-members.* The comment says "everything except duckctl", but `duck-ether` is also missing and `duck-ble` is listed twice (`Cargo.toml:8-21`).
5. *Roadmap staleness.* The roadmap table says `duck-ipc-proto` is "at API v14" while the code has `API_VERSION = 44` (`docs/project/roadmap.md` "Where we are"; `duck-ipc-proto/src/lib.rs:473`).
6. *Control-rate rationale.* `[control] hz` is 50 "never re-derived on the Radxa", yet the policies' manifests pin `control_hz: 50`. Changing the rate would break sim-to-real.

**Sim-to-real risks** (relevant if reusing the policies or this contract):
- The observation contract has no runtime normalisation. Normalisation must be inside the ONNX. The previous action is the *unscaled* output [V].
- Voltage adaptation (on by default since 2026-09-14) multiplies action_scale by 7.4/V. At 8.2 V that is about ×0.90 [V]. [U] Unclear whether training modelled this.
- The IMU gravity vector is median-filtered then not renormalised, so it is slightly sub-unit during transients (`duck-control/src/imu.rs:140-147`) [V].
- Joint velocity is the raw servo register with no filtering. Gyro has a 3-tap median [V].
- Training-side parity (low-pass α 0.5/0.7, cmd EMA 0.2, the delays) is asserted only by comments. There are no golden vectors [V].
- Target conversion depends on `rustypot`'s rad↔count mapping and the knee EEPROM homing offset [V/U].
- Read timeout is 30 ms against a 20 ms tick, so one slow bus read costs at least one skipped tick [V]. [U] I did not quantify the worst case.

**Hardware-specific / Pollen-specific** [V]:
- XL330 firmware ≥ v46 and `imu_to_dxl` firmware supporting 0x8A (fast sync read). The `imu_to_dxl` board and firmware are proprietary or external to this repo.
- Board IDs `radxa,zero3*` / `pollen,microduck-beta` / `seeed,microduck` (`robotd-params/src/board.rs:103-120`). Serial `/dev/ttyS2`, Armbian overlay quirks (`deploy/README.md:200`), Rockchip MPP/RGA/rkaiq/RKNN setup scripts.
- Cloud and services: HF Hub policy/detector repos, an HF account device flow, the Pollen rendezvous/TURN (`turn-endpoint.yml`), GitHub Releases on this repo, and Pollen minisign keys.
- Audio is the TLV320AIC3104 on the HAT; ToF and head-IMU wiring is HAT-specific.

**Missing / immature** (doc-sourced unless noted):
- Per-joint limits, rate limits and thermal limits; per-device IMU calibration (`docs/design/robotd-design.md:1132-1135`).
- Authority arbitration; BLE pairing security (default PIN `000000`); the recovery net is unexercised on hardware; the app and SDK do not exist (`docs/project/roadmap.md` "Where we are"; M6).
- No physical e-stop is described.
- No hardware design files.
- No RT kernel or scheduling.
- Inference latency on aarch64 is not recorded in the docs.

**Reusability assessment** [U, opinion]:
- `duck-control` is cleanly separated: no tokio, sockets or systemd, and a `RobotIo` trait seam. It is the most reusable piece; a new bus backend only needs to implement `read / write / set_gain / set_torque / slow_sensors`.
- `kinematics` (MJCF-driven FK) and `duck-ipc-proto` are also portable.
- `updater` and `xtask` are generic but tied to minisign plus GitHub/HF sources.
- The rest is product glue specific to Pollen hardware and services.

## 11. docs/ directory structure (37 .md files, ~18.7k lines) [V] (`docs/README.md`)

```
docs/
  README.md               index; "the page that owns a mechanism wins" rule
  faq.md                  builder-facing Q&A (heavy models, camera access, Spaces)
  policy-manifest.md      manifest.json schema 2 contract for Hub policies (158 l)
  recurrent-policies.md   LSTM ONNX tensor contract and memory lifecycle (86 l)
  design/                 "how and why" (owner per mechanism)
    architecture.md         services, IPC, state, LEDs, API, remote, safety (655 l)
    robotd-design.md        control loop, bus, obs, policy, safety, API (1179 l)
    updater-design.md       signed/health-gated updates (1578 l)
    policy-channel-design.md  policies from the HF Hub (780 l)
    restart-order.md, boot-recovery-net.md, app-path-design.md (BLE + configd),
    mobile-app.md, remote-webrtc.md, webrtc-console.md, remote-access-design.md (1179 l),
    simulation.md           twin: RemoteIo seam, body protocol, containers (354 l)
  robot/                  user/dev operation
    cheatsheet.md (1342 l), cheatsheet-dev.md, dev-push.md, duckctl.md, install-dev.md,
    install-by-hand.md, pair-a-gamepad.md, simulation.md (duck-sim usage), udp-pad.md
  project/                dated records: roadmap.md, ci-setup.md, idle-cpu.md,
    install-path-gap.md, media-bringup.md, npu-bringup.md, pad-minimal-pairing.md,
    slice-2-bringup.md, tof-on-demand.md, update-over-ble.md
  ideas/autonomous_behavior.md
  superpowers/{plans,specs}/2026-10-07-netpad-udp*.md   (AI-agent planning artefacts for netpadd)
```

Other notable top-level items [V]:
- `deploy/` (robotd.toml with all defaults commented, updater.toml, trusted keys, journald, DT overlays);
- `scripts/` (board setup, provisioning, duck-sim, dev-push, board-test);
- `spaces/` (Hugging Face Spaces in Python: hello, policy-playground, vetbook, vision-demo);
- `hooks/`; `AGENTS.md` (with `CLAUDE.md` symlinked to it).
