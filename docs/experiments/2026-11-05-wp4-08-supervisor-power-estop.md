# WP4-08: supervisor e-stop, power, temperature and torque budget (S-2, S-5)

**Environment:** cloud (container, rustc 1.97). Simulation only; the hardware
e-stop (S-3) and every hardware safety item stay open until tested on hardware
(S-6). Stacked on WP4-04 (PR #25).

## Change
- Bridge protocol v2: `Hello` carries each joint's `kp` and torque limit; `State`
  carries bus voltage, hottest-servo temperature and a software e-stop flag.
- New latching faults → torque off: software e-stop, bus voltage < 6.4 V
  (2S, 3.2 V/cell), servo temperature > 68 °C.
- Derate (not latching): below 6.8 V or above 60 °C the torque budget halves.
- Torque budget: `--torque-fraction F` keeps `kp·|target − q| ≤ F·torque_limit`
  per joint (first power-on at a low fraction, S-5). Without it, only a derate
  applies a budget; otherwise the servo's own current limit does.
- The sim has no battery or thermal model: voltage and temperature are scripted
  linear test inputs in `BridgeConfig`.

## Results (cloud, `pytest -m runtime`, 11 tests; 11 Rust unit tests)
| Scenario | Result |
|---|---|
| Software e-stop at 1 s | torque off at step 51, the first state with the flag |
| Battery 7.4 V − 0.5 V/s | derate from step 62 (6.78 V), undervoltage stop at step 102 (6.38 V) |
| Temperature 60.5 °C + 5 °C/s | derate from step 1, overtemperature stop at step 77 |
| Torque fraction 0.3 + knee target +1 rad | max torque demand 0.300 of the limit (clamped in 76 periods); the open-loop stand still tips later (tilt fault at step 127 instead of 88) |
| Stand (no budget) | max torque demand 0.066 of the limit |

After any torque-off fault the limp open-loop robot may tip over, so the tilt
bit can latch as well; the tests check the first fault and its step.

## Not modelled
- Per-servo temperature and current (one hottest value only), the bus current
  limit of the regulator (BOM open point 2), and the hardware e-stop itself.
