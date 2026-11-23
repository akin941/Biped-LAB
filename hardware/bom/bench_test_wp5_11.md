# WP5-11 bench test kit: supervisor board + one servo (G4.5, DS3218MG build)

**Status: list only. Nothing is bought; purchases are the owner's decision
(AGENTS.md). First power-on also needs the owner's approval.**
**Environment:** cloud web research, 2026-11-23. robotsepeti.com pages showed
stock (in their meta tags) but no readable prices from the cloud, as before.
Prices below were read on the named shop's page; "est." means not read.

Almost everything here is also a part of the full robot
([biped_ds3218_bom.md](biped_ds3218_bom.md)), so the kit is the first slice of
that purchase rather than extra spend. Only the USB-serial adapter is extra.

## Kit
| # | Item | Qty | Price seen | Where / stock (2026-11-23) | Why |
|---|---|---|---|---|---|
| 1 | Raspberry Pi Pico (H or plain) | 1 | 303 TL (robolinkmarket, sold out); 325 TL + KDV (robo90, sold out); 485 TL incl. KDV (meonotomasyon) | robotsepeti: Pico H in stock, price not readable | the safety supervisor (runs `supervisor-rp2040.uf2`) |
| 2 | DSServo DS3218MG 180° | 1 | 1,197 TL (robolinkmarket, stock unclear) | robotsepeti: in stock, price not readable | one servo, no load, for the first pulses |
| 3 | 2S 7.4 V LiPo ~1200 mAh | 1 | est. 600–1,750 TL (robotistan 2200 mAh: 1,751 TL) | robotsepeti: Profuse 1200 mAh in stock | the supervisor reads this pack; it stops below 6.6 V, so AA cells cannot stand in |
| 4 | 6 V UBEC, ≥ 5 A (servo supply) | 1 | Henge 6 A: 800.68 TL, Henge 8 A: 1,109.25 TL incl. KDV (promodelhobby, sold out); est. 300–800 TL elsewhere | not checked at robotsepeti | keeps the 8.4 V pack off the servo (max 6.8 V) |
| 5 | USB-TTL serial adapter, 3.3 V (CP2102 or CH340) | 1 | est. 80–150 TL | not checked | lets the owner's Mac run `biped-runtime --servos pwm` against the Pico without a Pi |
| 6 | Latching kill switch ≥ 10 A + blade fuse and holder | 1 + 1 | est. 100–250 TL | not checked | S-3 / S-4, mandatory before the servo moves |
| (opt.) | MPU6050 (GY-521) on the Pico's GP14/GP15 | 1 | 94.86 TL incl. KDV (f1depo, sold out); 123 TL (robotistan, sold out) | robotsepeti: in stock, price not readable | tests the S-2 tilt stop: tip the board past 45° and the LED must light |
| 7 | Breadboard, jumper wires, 22 kΩ + 10 kΩ resistors, XT60/JST lead | – | est. 100–200 TL | not checked | wiring and the battery divider |
| (if missing) | 2S balance charger + LiPo bag | 1 | est. 400–800 TL | – | S-4; skip if one is at hand |

**Rough total: ≈ 2,700–4,500 TL** (plus a charger if needed). The spread comes
from the battery and the UBEC; check the cart before deciding.

**Price warning for the full robot:** the only DS3218MG price readable today is
1,197 TL, about twice the ~600 TL estimate in the BOM. If that holds, ten servos
cost ~12,000 TL, not ~6,000 TL. Confirm in the robotsepeti cart first.

## Steps (each one needs the owner's go, `env: hw`)
1. **Flash.** Hold BOOTSEL, plug the Pico into the Mac, then copy
   `supervisor-rp2040.uf2` (the `supervisor-firmware` artifact of any green CI
   run on main) onto the drive. No servo or battery is connected yet.
2. **Board alone.** Connect the USB-TTL adapter: TX→GP1, RX←GP0, GND. With
   no battery, the start must fail with "faults stay latched … battery 0.00 V".
   That is the undervoltage stop working.
3. **Battery divider.** Connect the 2S pack through the fuse and switch to the
   22k/10k divider on GP26 (ADC0). Then run `biped-runtime --servos pwm --device
   /dev/cu.usbserial-… --hello hardware/calibration/biped_ds3218.hello --imu
   upright --torque-fraction 0.3 --joint-feedback commanded --seconds 5`. Pass
   if the reported battery voltage is within ±0.1 V of a multimeter reading.
4. **One servo, no horn, no load.** Power the servo from the UBEC behind the
   kill switch, connect its signal to GP2 (joint 0), and keep a common ground.
   Run step 3's command again. Pass if the servo goes to its centre and holds
   it. Then pull the USB-TTL lead: the servo's pulses must stop within 0.5 s
   and the LED must light (watchdog latch).
5. **Kill switch.** Cut servo power with the switch while it holds a position;
   the servo must go limp.

Results go back into `docs/roadmap/BACKLOG.md` (WP5-11) with `env: hw`.
