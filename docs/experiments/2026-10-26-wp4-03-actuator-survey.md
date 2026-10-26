# WP4-03: actuator survey for biped_v1 and identification plan

**Environment:** cloud (container): web research of manufacturer pages on
2026-10-26 and arithmetic in `bipedlab.design.actuators` /
`scripts/actuator_survey.py`. No hardware, no purchase, no measurement. The
requirements are the WP4-01 walking estimates (gha-linux, simulation).

## Method
- Each candidate is one servo at one supply voltage with the manufacturer's stall
  torque and no-load speed (sources in `CANDIDATES`).
- A candidate **passes** a joint type when
  1. stall torque ≥ 2 × estimated walking peak torque (G4.1 margin),
  2. no-load speed ≥ 2 × estimated walking peak speed, and
  3. the speed left at the walking peak torque on a linear torque–speed line,
     `no_load × (1 − peak / stall)`, is still ≥ the walking peak speed.
- `rms/stall` (RMS torque over stall) is reported for the thermal check
  (WP4-01 flag at 0.3).
- Prices are single-unit retail snapshots, informative only.

## Candidates (manufacturer data)
| Servo | Supply | Stall | No-load | Mass | Price snapshot |
|---|---|---|---|---|---|
| XL330-M288-T | 5 V (3.7–6 V) | 0.52 N·m | 103 rpm | 18 g | €40.20 incl. VAT |
| XL330-M288-T | 6 V | 0.60 N·m | 123 rpm | 18 g | – |
| XC330-M181-T | 5 V (3.7–6 V) | 0.60 N·m | 129 rpm | 23 g | not checked |
| XC330-M288-T | 5 V (3.7–6 V) | 0.93 N·m | 81 rpm | 23 g | $103.39 (US, incl. tariffs) |
| XC330-M288-T | 6 V | 1.10 N·m | 97 rpm | 23 g | – |
| XC330-T288-T | 11.1 V (6.5–12 V) | 0.92 N·m | 65 rpm | 23 g | €110.40 incl. VAT |
| XL430-W250-T | 11.1 V (6.5–12 V) | 1.4 N·m | 57 rpm | 57 g | not checked |
| STS3215 | 7.4 V | 1.91 N·m | 0.192 s/60° | 55 g | $33.97 (US, incl. tariffs) |

## Result (`uv run python scripts/actuator_survey.py`, cloud)
| Joint (peak torque, speed) | Passing candidates |
|---|---|
| hip yaw (0.106 N·m, 1.1 rad/s) | all |
| hip roll (0.174 N·m, 1.3 rad/s) | all |
| hip pitch (0.182 N·m, 3.7 rad/s) | XL330, XC330-M181, XC330-M288 (5/6 V); the slow high-voltage parts fail on speed |
| **knee (0.339 N·m, 5.0 rad/s)** | **only XC330-M288-T at 6 V** (torque ×3.2, speed ×2.0, loaded 7.0 rad/s, rms/stall 0.18) |
| ankle (0.218 N·m, 2.6 rad/s) | all (XL330 at 5 V: torque ×2.4) |

At the knee every option is limited by one of the two margins: the XL330 and
XC330-M181 by torque (×1.5–1.8), the 288:1 parts at 5 V or 11.1 V and the larger
servos by speed (×1.1–1.7). XC330-M288-T at 5 V gives torque ×2.7 but speed ×1.7.

## Recommendation (decision for the owner at WP4-05, BOM)
1. **Mixed set on one 6 V regulated bus:** XC330-M288-T at the two knees, XL330-M288-T
   at the other eight joints. Mass +10 g against the WP4-01 budget (190 g servos
   instead of 180 g). Price snapshot: 8 × €40.20 + 2 × $103.39 (EU price of the
   XC330 not checked). Drawback: 6 V is the top of the rated range, so the
   bus needs a regulator from the 2S pack and has no headroom.
2. **Alternative A:** same set at 5 V and accept speed ×1.7 at the knee
   (loaded 5.4 rad/s still above the 5.0 rad/s peak). Simpler electrically.
3. **Alternative B:** shorten thigh and shin (WP4-02 parameters) to cut the knee
   load, then re-run WP4-01; this may let XL330 cover every joint at one price.

The estimates borrow Microduck's gait (WP4-01 caveats). G4.3 (our own policy, Mac)
must re-check the knee before any purchase.

## Identification plan (hardware; owner approval, Phase 5)
Same approach as BAM (the actuator model the Microduck policies use):
1. Pendulum bench: the servo swings a known arm and mass; position, velocity,
   current and voltage are logged over the Dynamixel bus at ≥ 200 Hz.
2. Trajectories: sinusoids and steps at several amplitudes and loads, with the
   firmware's position controller at the gains we will use.
3. Fit the BAM model (friction budget, armature, motor constant, resistance) per
   servo type and compare with the datasheet values used here.
4. Put the fitted model into `bipedlab.sim.bam_xl330`-style code for our robot and
   retrain/evaluate in simulation before the first hardware step.
Nothing in this plan runs before the owner approves the purchase.

## Follow-up
- WP4-02: allow a different servo per joint in `BipedParams` once the owner picks
  an option, then re-run WP4-01 with the chosen ratings.
