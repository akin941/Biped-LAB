# WP4-16: command-channel authentication

**Environment:** cloud (container) for every result below. The CI `runtime` job
runs the same tests on gha-linux. Simulation only. Design: ADR-0014.

## Change
- `runtime/src/auth.rs`: SHA-256 and HMAC-SHA256, std-only; constant-time
  compare; session nonce from `/dev/urandom`; hex key parsing.
- `runtime/src/channel.rs` v2: hello (14 B) → tagged welcome (38 B) with the
  runtime's session number → tagged commands (62 B) with session and sequence.
  Checks in order: shape, tag + session (`unauthenticated`), peer lock
  (`foreign`), sequence (`stale`). At most 64 datagrams and 2 welcomes per period.
- `biped-runtime --command-key FILE` (hex, ≥ 16 bytes). Without it the key is
  all zeros and a non-loopback `--command-udp` address exits with code 2.
- `bipedlab.command_channel`: `CommandSender(host, port, key).connect()` does the
  handshake; `load_key`, `new_key_file` (mode 0600, never overwrites).
  `scripts/teleop.py --key-file FILE`, `--new-key FILE`.
- The control loop and its timing are unchanged: the channel is still polled once
  per period before the gate.

## Results (cloud)
- Rust unit tests (35 in the crate): SHA-256 FIPS 180-4 vectors ("", "abc",
  448-bit), HMAC RFC 4231 cases 1, 2 and 6, filter tests for a wrong key, a
  wrong session, replay/reorder, a foreign peer, malformed packets, hello →
  welcome, keyless non-loopback bind refused.
- Python unit tests: packet sizes and HMAC match the Rust layout; a welcome with
  a wrong key, another nonce, a flipped session bit or a wrong size is refused.
- End to end, real time (7 tests, three full `tests/runtime` runs in a row passed):
  - with a matching key file, commands are accepted and gated as before (ramp,
    clamp, timeout, second sender ignored);
  - with a wrong key the sender never gets an authentic welcome, sends nothing,
    and the logged command stays 0;
  - packets recorded in one run and replayed raw into a second run with the
    same key are all refused (`unauthenticated` > 10, `accepted` 0, command 0);
  - `--command-udp 0.0.0.0:PORT` without a key exits with code 2.

## Not covered
- No key rotation, one operator only, a restarted teleop is `foreign` until the
  runtime restarts (ADR-0014 consequences).
- Not yet on the robot or the Mac: MAC-10 has the optional key steps.
