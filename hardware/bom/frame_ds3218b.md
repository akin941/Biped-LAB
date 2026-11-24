# Frame options for the buildable DS3218MG layout (WP5-14)

**Status: research only, nothing bought** (purchases need the owner, AGENTS.md).
**Environment:** cloud, web research 2026-11-24. Model: `biped_ds3218b` (WP5-13).

The DS3218MG has the MG995 / MG996R body (40 × 20 × 40.5 mm, same mounting tabs), so
the cheap aluminium "MG995/MG996 servo bracket" parts sold for 17-DOF hobby humanoids fit it.
That makes the frame a parts question, not a CAD job, for a first prototype.

## Per-leg layout that `biped_ds3218b` assumes (hip to foot)
| Joint | Servo sits in | Bracket that carries the next link | Model offset |
|---|---|---|---|
| hip yaw | trunk plate, shaft down | multi-function bracket on the horn | yaw → roll 30 mm |
| hip roll | multi-function bracket, shaft along x | short U bracket + multi-function bracket (cross) | roll → pitch 50 mm |
| hip pitch | multi-function bracket, shaft along y | long U bracket (thigh) | thigh 65 mm |
| knee | multi-function bracket at the thigh end | long U bracket (shin) | shin 65 mm |
| ankle pitch | multi-function bracket at the shin end | U bracket on a foot plate | ankle 26 mm |

Per robot: 10 multi-function brackets, ~4 short U, 4 long U, 2 foot plates, 10 metal horns,
10 ball bearings (idler side of each U), M3 screws. The two hip-yaw servos mount on a flat
trunk plate (aluminium, PCB blank or printed).

## Prices seen (2026-11-24, read on the shop pages)
| Part | Where | Price | Stock | Notes |
|---|---|---|---|---|
| Long U servo bracket (MG995/MG996R) | gorsuelektronik.com | 40.85 TL + KDV | out of stock | no dimensions given |
| Bracket kit: 1 multi-function + 1 short U + 2 metal horns + 1 bearing | communica.co.za (ZA) | R 69 incl. VAT (~150 TL) | sold out | 58 × 37 × 25.5 mm, 2 mm plate, 15.5 g |
| Full 17-DOF biped bracket set (16 multi-function, 7 short U, 4 long U, 4 inclined U, 4 L, 2 foot plates, 14 bearings, 17 horns) | kitsguru.com (IN) | ₹ 6,999 | in stock | robot ~40 cm tall; shipping to Turkey not checked |
| Same 17-DOF set | vxb.com (US) | $173.95 | add-to-cart shown | taxes / shipping extra |

Estimate for Turkey: 10 × (multi-function + U) at ~40-80 TL each part ≈ **1,000-1,600 TL**
for the brackets, if a Turkish shop has them in stock (check robotsepeti / robotistan /
direnc.net carts; their prices were not readable from the cloud). A full imported set costs
more after shipping and customs.

## What this changes in the model (honest gaps)
- **Mass:** one multi-function + U pair is ~25-30 g, so ten joints add ~250-300 g of
  brackets. The model budgets 12 g per hip bracket and 20 g per thigh / shin link
  (~0.13 kg of structure in the legs). With real brackets the robot is likely ~1.15-1.2 kg,
  not 1.04 kg. At 1.04 kg the worst walking torque was 1.10 N·m (MAC-49, bar 1.3), so ~15 %
  more mass eats most of that margin; the next
  model update (after the first brackets are weighed) raises the link masses and re-checks it.
- **Offsets:** the hobby kits' hole spacing is not published on these pages. The 30 / 50 /
  65 / 65 / 26 mm offsets are the model's; measure the bought brackets (calipers, axis to
  axis) and set them in `BipedParams`, then run `scripts/check_clearance.py` and retrain.
- **Option B, printed frame:** PLA/PETG parts drawn for these offsets would be lighter
  (~150-200 g total) and match the model exactly, but need a printer or a print service.
  Not drawn yet.
