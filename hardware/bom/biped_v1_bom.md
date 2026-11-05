# biped_v1 bill of materials: draft (WP4-05, G4.5)

**Status: draft for the owner's decision. Nothing is bought** (AGENTS.md;
Phase 5 entry needs Phase 4 gate + owner purchase approval).
**Environment:** cloud web research, 2026-10-26. Prices are single-unit retail
snapshots in the currency the shop showed; taxes, shipping and tariffs vary.
"not checked" means no price was confirmed.

## Architecture this BOM assumes (ADR-0006, S-2, S-3)
```
Pi Zero 2 W (policy, command gate)  --UART-->  RP2040 board (safety supervisor,
                                               owns the servo bus)  --TTL half-duplex-->  10 servos
2S LiPo -> fuse -> e-stop switch -> 6 V regulator -> servo bus
                                 -> 5 V regulator -> Pi + RP2040 + IMU
```
The supervisor runs on separate hardware from the high-level computer (S-2),
so the servos are driven by the RP2040, not by the Pi. The e-stop cuts servo
power in hardware (S-3), independent of both computers.

## Items
| # | Item | Qty | Unit price (snapshot) | Mass each | Source / note |
|---|---|---|---|---|---|
| 1 | ROBOTIS XL330-M288-T (hip yaw/roll/pitch, ankle) | 8 | €40.20 incl. VAT | 18 g | generationrobots.com; WP4-03 |
| 2 | ROBOTIS XC330-M288-T (knees) | 2 | $103.39 | 23 g | robotshop.com (US, incl. tariffs); EU price not checked; WP4-03 |
| 3 | Raspberry Pi Zero 2 W (policy computer) | 1 | €21.27 incl. VAT ($15 list) | 10 g | raspberrypi.dk; 4×A53 1 GHz, 512 MB |
| 4 | RP2040 board, e.g. Raspberry Pi Pico (safety supervisor) | 1 | not checked (~$4 list) | ~3 g | firmware language: later ADR (ADR-0007) |
| 5 | Half-duplex TTL buffer for the Dynamixel bus (74LVC2G241 or similar) | 1 | not checked | <1 g | replaces a USB adapter |
| 6 | Adafruit BNO085 IMU breakout | 1 | $29.50 | 2.5 g | adafruit.com #4754 |
| 7 | Pololu D36V50F6 6 V 5.5 A step-down (servo bus) | 1 | $39.95 | 7 g | pololu.com #4092; input 6.5–50 V |
| 8 | 5 V step-down for the Pi/RP2040 (≥ 1.5 A) | 1 | not checked | ~3 g | |
| 9 | 2S LiPo ~450–500 mAh, XT30 | 1 (+1 spare) | not checked | ~29 g (450 mAh) | e.g. Tattu 450 mAh 75C |
| 10 | Balance charger for 2S + LiPo safety bag | 1 | not checked | – | S-4 |
| 11 | Latching e-stop / kill switch rated ≥ 10 A in the servo supply | 1 | not checked | – | S-3; on-robot switch or tethered |
| 12 | Blade fuse + holder (≈ 10 A) | 1 | not checked | ~2 g | S-4 |
| 13 | Structure: 3D-printed PETG/PLA frame, M2/M2.5 screws, servo horns/cables | – | not checked | ≤ 110 g (budget) | original geometry (ADR-0004) |
| (opt.) | ROBOTIS U2D2 USB–TTL adapter, for bench ID of servos from a PC | 1 | $36.92 | 9 g | robotis.us; useful for WP4-03 identification plan |

## Cost summary (known prices only)
- Servos: 8 × €40.20 = **€321.60** and 2 × $103.39 = **$206.78**.
- Electronics with confirmed prices: €21.27 + $29.50 + $39.95 (+ $36.92 if the
  U2D2 bench adapter is bought).
- Not yet priced: RP2040 board, buffer, 5 V regulator, battery, charger, e-stop,
  fuse, filament and fasteners.
- Rough total with confirmed prices: about €343 + $276 (servos dominate).

## Checks against the WP4-01 budget (0.512 kg)
- Servos 8 × 18 + 2 × 23 = 190 g (budget 180 g): +10 g.
- Electronics: Pi 10 + RP2040 3 + IMU 2.5 + regulators ~10 + wiring ~10 ≈ 36 g
  (budget 40 g).
- Battery: 450 mAh ≈ 29 g (budget 50 g, 500 mAh); runtime estimate drops from
  ~23 to ~21 min.

## Open points (owner decisions)
1. **Bus voltage.** The 6 V regulator needs ≥ 6.5 V in; a 2S pack sags to
   ~6.6 V near empty, so it works only with a low-voltage cutoff around 6.8 V.
   The alternative is a 5 V bus (WP4-03 alternative A: knee speed margin ×1.7)
   and a simpler supply.
2. **Peak current.** Walking mean is ~1.3 A at 6 V (WP4-01: ~7.7 W). Ten servos at
   stall would draw ~15–20 A, far over 5.5 A: the supervisor must keep current
   limits (servo current-limit registers) and the fuse protects the wiring.
3. **E-stop form:** an on-robot latching switch versus a tethered kill switch
   during first tests (S-5 suspended tests favour the tether).
4. **Purchase timing:** only after G4.3 re-checks the knee requirement with our
   own policy.
