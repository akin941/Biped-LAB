# biped_ds3218 bill of materials: low-budget prototype (WP4-05, G4.5)

**Status: servo picked by the owner (2026-11-23): 10 × DSServo DS3218MG.
Nothing is bought.** Purchases need the owner's own decision (AGENTS.md).
**Environment:** cloud web research, 2026-11-23. robotsepeti.com product pages
showed stock in their meta tags but their prices were not readable from the
cloud, so prices marked "est." are estimates to check in the shop cart.
Prices with a shop name were read on that shop's page.

The earlier bus-servo draft (`biped_v1_bom.md`, XL330 / XC330 / STS3215) stays
as the upgrade path; the owner turned it down for now on cost (~570 $ for the
servos alone).

## Architecture this BOM assumes (ADR-0006, S-2, S-3)
```
Pi Zero 2 W (policy, command gate)  --UART-->  RP2040 board (safety supervisor,
                                               10 PWM outputs)  --PWM-->  10 × DS3218MG
2S LiPo -> fuse -> e-stop switch -> 6 V UBEC (>= 10 A) -> servo power
                                 -> 5 V regulator -> Pi + RP2040 + IMU
```
The RP2040 has 16 hardware PWM channels, so it can drive the 10 servos itself
and the supervisor still owns the actuators (S-2). A PCA9685 board is the
alternative if the wiring is easier that way.

**No joint feedback.** PWM hobby servos do not report their angle or speed.
The policy is trained with the targets it sent in place of measured joint
angles (`--joint-feedback commanded`, WP4-39); the robot still measures its
body with the IMU. Reading each servo's potentiometer through an ADC is a later
option if the gait needs it.

## Items
| # | Item | Qty | Unit price | Stock (2026-11-23) | Mass each | Note |
|---|---|---|---|---|---|---|
| 1 | DSServo DS3218MG 20 kg·cm digital servo, 180° | 10 | est. ~600 TL | robotsepeti: in stock | 60 g | 6 V: ~1.96 N·m, 0.15 s/60° (seller datasheet) |
| 2 | RP2040 board, e.g. Raspberry Pi Pico (supervisor + PWM) | 1 | not checked | not checked | ~3 g | |
| 3 | Raspberry Pi Zero 2 W (policy computer) | 1 | not checked in TL | not checked | 10 g | skip if one is already at hand |
| 4 | MPU6050 IMU (GY-521) | 1 | robotistan: 123.00 TL incl. VAT | robotistan: sold out; robotsepeti not checked | ~2 g | BNO085 is the better, dearer option |
| 5 | 6 V UBEC, ≥ 10 A continuous (servo supply) | 1 | not checked (est. ~300 TL) | not checked | ~15 g | DS3218MG max 6.8 V: a full 2S pack (8.4 V) must not reach the servos |
| 6 | 5 V regulator for Pi / RP2040 / IMU (≥ 1.5 A) | 1 | not checked | not checked | ~3 g | |
| 7 | 2S 7.4 V LiPo, 1200 mAh 25C (e.g. Profuse) | 1 | not readable (est. ~600 TL) | robotsepeti: in stock | ~70 g | 450 mAh (JSumo) also in stock there, shorter runtime |
| 8 | 2S balance charger + LiPo safety bag | 1 | not checked | – | – | S-4; skip the charger if one is at hand |
| 9 | Latching e-stop / kill switch ≥ 10 A in the servo supply | 1 | not checked | – | – | S-3 |
| 10 | Blade fuse + holder (≈ 10 A) | 1 | not checked | – | ~2 g | S-4 |
| 11 | Structure: 3D-printed PLA/PETG frame for 40 × 20 mm servos, metal servo horns, M3 screws | – | not checked | – | ~200 g budget | original geometry (ADR-0004) |
| (opt.) | PCA9685 16-channel PWM board | 1 | robotistan: 249.69 TL incl. VAT | robotistan: sold out; robotsepeti: in stock | ~9 g | only if not driving PWM from the RP2040 |

## Cost summary
- Servos: 10 × ~600 TL ≈ **6,000 TL** (estimate, check the cart).
- IMU, UBEC, LiPo: ≈ 1,000 TL (MPU6050 price read; the others estimated).
- **Rough total ≈ 7,500 TL** with a little left for wiring, fuse and switch,
  not counting a Pi, charger or filament the owner may already have.

## What the cheaper servo costs (honest trade-offs)
- **Weight:** 10 × 60 g puts the robot at ~1.04 kg instead of 0.51 kg
  (`biped.BIPED_DS3218`). The joints need about twice the torque of the
  biped_v1 gait; DS3218MG has about 2× margin over that gait at 0.51 kg, so at
  1 kg the margin is thin. MAC-47 measures it.
- **Speed:** ~7 rad/s no-load, below 2× the MAC-36 hip-pitch / knee p95
  (~7 rad/s). The STS3215 fine-tune walked under a 5.45 rad/s curve (MAC-37),
  so a slower gait is expected to work, not proven.
- **No feedback:** the policy cannot see a joint that lags or is blocked; the
  sim-to-real report (WP5-05) loses joint tracking and torque estimates.
- **Bench identification:** stiffness, deadband and friction of the servo are
  guessed in the model; stage 3 must measure them before training is trusted.

## Open points
1. Confirm the cart prices on robotsepeti before deciding the purchase.
2. MAC-47 (sim training on the Mac) should show the robot walks before buying.
3. Runtime work for PWM output, commanded feedback and the MPU6050 (WP5-06).
