# The MAC-16 yaw drift comes from the policy, not the runtime path (MAC-18, WP4-21)

```
env: mac-m5 (macOS 27.0.1 arm64), 2026-11-19; MAC-17 training ran in parallel (eval is not real time)
revision: 441e731 (main)
policy: MAC-14 policy.onnx (sha256 83766219233e60a14315ea78e9ee0ff68fcfa680787542f215d0cc8e8be6f78f)
run: eval_biped.py --dr --envs 8 --seconds 20 (each command DR off, then DR on)
```

| command (vx, wz) | Python, DR off: vx / wz | Python, DR on: vx / wz | runtime MAC-16 (no DR): vx / wz |
|---|---|---|---|
| 0.15, 0 | 0.178 / −0.186 | 0.150 / −0.128 | 0.194 / −0.190 |
| 0.25, 0 | 0.264 / −0.239 | 0.228 / −0.159 | 0.285 / −0.148 |
| 0, 0 | 0.017 / −0.021 | 0.004 / 0.001 | – |
| 0, 0.8 | 0.016 / 0.622 | 0.003 / 0.558 | – |

0 falls in every case. |vy| ≤ 0.012 m/s. The zero-action baselines stay at 0.

## Conclusion
- **The policy drifts.** Without DR, the Python env already turns right (−0.19 / −0.24
  rad/s) and walks fast (119 / 106 % of the command), just like the runtime. So the
  runtime path is faithful. DR on lowers both the speed and the yaw.
- The drift appears only while walking forward (wz ≈ 0 at the zero command) and grows
  with speed when DR is off, so it comes from an asymmetric gait, not a heading bias.
  The policy was trained only with DR, and both effects are largest with DR off.
- The runtime adds about 8–9 % vx over the Python env with DR off. At 0.25 its yaw is
  smaller (−0.148 vs −0.239). Both differences are small next to the DR on/off effect.
- For the next WP: more weight on yaw tracking at zero yaw commands, or a mirror-symmetry
  augmentation, then check that wz stays near 0 with DR off.

Files: `mac18/eval.json` (policy path shortened to `<mac14 worktree>/…`).
