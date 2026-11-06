# Fixes for the first M5 findings: runtime pacing, foot friction, latency (WP4-16)

```
env: cloud container (Linux, 4 vCPU), 2026-11-06
inputs: MAC-08 and MAC-09 results on mac-m5 (PR #38)
```

## Findings (mac-m5, PR #38)
1. **Runtime ran at ~40 Hz, not 50 Hz** (MAC-09): period p50 24.98 ms, p99 30.04 ms;
   `test_realtime_pacing_holds_50_hz` failed 6/6 (6.16–6.32 s for 5 s), while
   `deadline_misses` stayed 0. The loop slept `period - compute` *after* the blocking
   state read, so the time spent waiting for the sim (≈ 5 ms on a loaded M5) was added
   to every period, and the miss counter only saw compute time (≈ 16 µs).
2. **Low foot friction was never applied** (MAC-08): `friction_0.5` equalled nominal.
   MuJoCo combines two equal-priority geoms with `max()`, and the floor is 1.0, so the
   DR range 0.7–1.3 was effectively 1.0–1.3 and the eval case did nothing.
3. **Latency** (MAC-08): forward speed fell to 71 / 44 / 35 % of the command at
   20 / 40 / 60 ms action delay (no falls); training had no action delay.
4. `test_second_sender_is_ignored` failed 3/10 on the M5.

## Fixes
1. `runtime/src/pacer.rs`: absolute schedule (deadline = previous deadline + period).
   The wait for the state is part of the period; a period ending after its deadline is
   a miss (so `deadline_misses` now counts real overruns); after a miss less than one
   period late the schedule catches up, otherwise it restarts from now (no burst).
   4 Rust unit tests, incl. a model of the MAC-09 case (5 ms outside the runtime per
   period: 250 periods take 5.0 s, 0 misses).
2. Foot geom class gets `priority="1"`, so the foot's friction is the foot–floor
   contact's. All friction values are equal in the committed model, so nominal physics
   is unchanged: a 200-step DR rollout with random actions gave identical observations
   (max |diff| 0.0) when every sampled friction was ≥ 1. Test: contact friction equals
   the foot friction at 0.3, 1.0, 1.5.
3. DR draws an extra 0–2 control periods (0–40 ms) of action delay per env at each
   reset, from its own random stream (the other DR draws are unchanged). Test: per-env
   delays differ and only zero-delay envs move on the first step.
4. The second sender starts 1 s after the first, after the runtime has locked on;
   before, both were live before the runtime bound its port and either could win.

## Checks (cloud)
- `cargo test` 33 passed (4 new), `cargo clippy` clean; `pytest -m runtime` 23 passed;
  10 s real-time run: 500 periods, 0 misses, wall 9.98 s.
- `pytest` 177 passed, 7 skipped (Hugging Face-only policies).
- DR PPO smoke (biped_v1, 60 it, seed 0) still rises: −0.0175 → −0.0046.

The policy from MAC-08 was trained without these fixes; MAC-11 re-runs MAC-09 and
retrains MAC-08 on the M5.
