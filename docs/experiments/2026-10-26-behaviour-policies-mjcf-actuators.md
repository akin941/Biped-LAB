# Behaviour policies with MJCF actuators (WP1-08)

```
env: gha-linux, CI run 40304688314, job pretrained-policy (PR #9)
actuator: MuJoCo position actuators (not BAM), init: upstream (z = 0.125 m, no settle)
command: as in upstream infer_policy.py (see src/bipedlab/sim/microduck_behaviours.py)
```

| behaviour | duration | upright at end | final / max tilt | trunk height min → final | other | max \|action\| |
|---|---|---|---|---|---|---|
| sitstand (sit 0–4 s, stand 4–8 s) | 8 s | yes | 3.5° / 6.8° | 0.059 → 0.116 m | — | 1.57 rad |
| ground_pick (phase 0 → 0.7) | 2.8 s | yes | 2.9° / 34° | 0.084 → 0.116 m | — | 1.71 rad |
| kick_left | 3 s | yes | 0.9° / 4.5° | 0.110 → 0.114 m | ball moved 0.97 m | 1.67 rad |
| kick_right | 3 s | yes | 0.6° / 8.2° | 0.109 → 0.114 m | ball moved 0.00 m | 1.74 rad |
| roulade | 2 s | **no** | 94° / 165° | 0.041 → 0.048 m | moved 0.35 m forward | 5.50 rad |

## Interpretation

- **Sit/stand works**: the trunk drops to 5.9 cm while the sit flag is on and comes
  back to 11.6 cm after it is cleared. **Ground pick works** in shape: crouch to 8.4 cm,
  34° forward lean, back upright.
- **Left kick works** (ball moved ~1 m). **Right kick misses**: the robot moves but the
  ball does not. Ball placement mirrors upstream exactly (tested), so this is either a
  real policy/actuator effect or a difference in when the ball is placed (upstream places
  it when the kick is triggered, usually after standing a while; we place it at t = 0).
  Not investigated further yet.
- **Roulade does not complete**: the robot rolls forward (tilt reaches 165°) but ends on
  its side (94°). A roll needs fast, high-torque joint motion, which is exactly where the
  MJCF actuators and BAM's voltage-limited motor model differ most.
- **Important contrast with walking**: these policies command large joint offsets
  (1.5–5.5 rad) and clearly act, while alpha_walking stays below 0.24 rad and never lifts
  a foot (see 2026-10-26-official-policies-mjcf-actuators.md). So "the policies cannot act
  in this simulator" is ruled out. The walking failure is specific to the walking policy;
  the BAM actuator is still the leading hypothesis, but a walking-specific input mismatch
  is now a second one to rule out (WP1-11).
