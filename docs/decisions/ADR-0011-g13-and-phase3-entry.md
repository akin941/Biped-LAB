# ADR-0011: G1.3 stays as written; Phase 3 starts on simulator-fidelity evidence

Status: accepted (2026-10-26)

## Context
G1.3 asks that, with the BAM actuator, `alpha_stand` stands ≥ 20 s and the official
`alpha_walking` tracks vx = 0.2 m/s within ±30 %. Phase 3 (training our own policy)
listed G1.3 as its entry gate, on the assumption that a failing G1.3 would mean our
simulator is wrong.

Evidence (gha-linux, PRs #12 and #13):
- alpha_stand with BAM stands and tracks ±10° body-tilt commands at 9.0–9.7°;
  upstream reports 9.2–9.5° for this policy.
- All five behaviour policies (sit/stand, ground pick, both kicks, roulade) do their
  task with BAM and end upright.
- alpha_walking does not step from a still stand at ≤ 0.2 m/s; at 0.4 m/s it reaches
  0.155 m/s. Upstream's own sim evaluation of this policy is 0.4 → 0.17 m/s
  (`microduck_rl` `docs/velstand_policy.md`). The shortfall belongs to the policy.
- The walking policy the robot runtime ships (`velstand.onnx`, v8) is not published
  on Hugging Face (public tags v1–v4, no velstand file), so it cannot be checked.

## Decision
- G1.3 is not changed and is recorded as failing with the official policy.
- Phase 3 entry uses the evidence above instead of G1.3: the simulator reproduces
  upstream's published behaviour for the stand, behaviour and walking policies
  (walking within 10 % of upstream's number). That is what the gate was meant to
  establish.
- G1.3's walking threshold becomes the bar for **our** policy (G3.3), unchanged.
- G3.1 validates the training env by reproducing upstream's numbers (stand body tilt,
  0.4 → 0.17 m/s walk), not by G1.3.
- If the velstand policy is published, WP1-14 evaluates it against G1.3.

## Consequences
- Phase 3 work (PPO library choice, gymnasium env) can start in the cloud now.
- No acceptance criterion is relaxed; one gate's role in sequencing changes, recorded here.
- Revisit if new evidence shows a simulator error behind the walking shortfall.
