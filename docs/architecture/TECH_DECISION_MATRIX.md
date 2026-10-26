# Technology decision matrix

Scores 1 (poor) – 5 (good). "Mac" = native Apple Silicon support; "Up" = alignment
with upstream Microduck; "RT" = suitability for real-time/safety roles.

| Need | Candidate | Mac | Up | RT | Cost/complexity | Decision | Why |
|---|---|---|---|---|---|---|---|
| Physics sim | **MuJoCo 3.10 CPU** | 5 | 5 | – | 5 | **adopt** (ADR-0002) | same engine/version as upstream; wheels everywhere |
| | MuJoCo-Warp / mjlab | 1 | 5 | – | 2 | reject for now | CUDA required |
| | MJX (JAX) | 2 | 2 | – | 3 | defer | JAX-Metal experimental; different from upstream |
| | Isaac Lab | 1 | 1 | – | 1 | reject | NVIDIA only |
| Experiments/RL language | **Python 3.12** | 5 | 5 | 1 | 5 | **adopt** | ecosystem; upstream pin < 3.13 |
| Env manager | **uv + uv.lock** | 5 | 5 | – | 5 | **adopt** | upstream uses it; fast, reproducible |
| RL algorithm | **PPO** | 5 | 5 | – | 4 | **adopt** (Phase 3) | upstream recipe; robust for locomotion |
| PPO library | SB3 / rsl-rl (CPU torch) | 4 | 3/5 | – | 4 | evaluate in Phase 3 (WP3-01) | SB3 proven on Mac by microduck-lab; rsl-rl matches upstream |
| Policy inference | **ONNX Runtime CPU** | 5 | 5 | 4 | 5 | **adopt** | upstream export format; same runtime in Python and Rust (`ort`) |
| Robot runtime | Rust (`ort`, tokio) | 5 | 5 | 4 | 3 | **adopt in Phase 4** (ADR-0007) | memory safety, predictable latency; upstream reference design |
| | Python on SBC | 5 | 2 | 2 | 5 | reject for control loop | GC pauses, weaker typing for safety code |
| Low-level controller | ESP32-S3 | – | 1 | 4 | 4 | candidate (Phase 4 study) | cheap, dual core, FreeRTOS; Microduck uses servo-internal control instead |
| | STM32 | – | 1 | 5 | 3 | candidate | better real-time peripherals, CAN |
| High-level compute | Linux SBC (RK3566 / Pi 5 class) | – | 4 | 2 | 4 | candidate | upstream uses RK3566 |
| Middleware | ROS 2 | 2 | 1 | 3 | 2 | **not now** (ADR-0009) | upstream doesn't use it; Mac support weak |
| Containers | Docker | 3 | 2 | – | 3 | optional, Linux CI only | uv suffices; no Metal in containers |
| CI | **GitHub Actions** (+ macOS arm64 runner) | 4 | 5 | – | 5 | **adopt** | free for public repos; pinned actions |
| GPU training | Paid cloud GPU / HF Jobs | – | 5 | – | 1 | **reject** | owner constraint; needs explicit approval |
| Actuators (custom robot) | Dynamixel XL330 class | – | 5 | 3 | 3 | candidate | proven on Microduck, has BAM model; torque may be low for heavier robot |
| | Feetech STS3215 class | – | 2 | 3 | 4 | candidate | cheaper, more torque; needs own actuator identification |
