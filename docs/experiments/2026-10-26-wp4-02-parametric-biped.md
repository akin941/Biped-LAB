# WP4-02: parametric MJCF for our own biped

**Environment:** cloud (container). Not run on the Mac.

## What was built
- `bipedlab.design.biped`: `BipedParams` (every dimension, mass and joint range) and
  `ServoSpec` (placeholder actuator class) → `to_mjcf()`. Only box primitives; no
  meshes, no third-party geometry (ADR-0004).
- `scripts/gen_biped.py` writes `simulation/custom-biped/biped_v1.xml`; `--check`
  fails if the committed file is stale. `uv run bipedlab stand biped` runs it.
- Layout: 10 joints, per leg hip yaw, hip roll, hip pitch, knee and ankle pitch. Each
  servo sits on the parent side of its joint; the trunk carries electronics and
  battery. Feet are the only geoms that collide with the floor.

## Default design (biped_v1)
| Quantity | Value |
|---|---|
| Height, stand pose (knee 0.6 rad) | 0.25 m (trunk top) |
| Total mass | 0.512 kg (10 × 18 g servos, trunk 0.20 kg incl. 50 g battery) |
| Leg | thigh 65 mm, shin 65 mm, ankle-to-sole 26 mm, hips 80 mm apart |
| Foot | 90 × 50 mm |
| Servo class | XL330-class placeholder: 0.52 N·m limit, kp 5 N·m/rad |

These are budgets, not a measured robot. WP4-01 sizes the actuators against them
and WP4-03 picks real parts.

## Checks (cloud, `tests/unit/test_biped_design.py`, 7 tests)
- Committed XML equals the generator output.
- Box primitives only; 10 hinge joints with the expected names; total mass equals
  the sum of the budget; trunk top within 20–30 cm; centre of mass on the
  sagittal plane.
- STAND keyframe: both soles flat, 2 mm above the floor.
- Open-loop stand for 3 s with thigh = shin ∈ {55, 65, 80} mm: no fall, tilt < 5°.
  At the default design the stand holds 0.39° max tilt over 5 s; the static knee
  torque is 0.035 N·m (7 % of the placeholder limit).

## Not covered yet
Walking needs a policy trained on this model (G4.3, Mac). Torque and speed from
dynamic motions are WP4-01.
