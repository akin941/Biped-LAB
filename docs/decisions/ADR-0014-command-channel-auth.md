# ADR-0014: Command-channel authentication with a pre-shared key

Status: accepted (2026-11-06). Applies to `biped-runtime --command-udp` (WP4-14).
Sim only so far; using it on the robot is part of the WP5 hardware review.

## Context
The WP4-14 channel accepted any UDP packet with the right shape and locked onto
the first sender's address. On a robot that binds to Wi-Fi this is not enough:
anyone on the network can spoof the source address, or record packets and send
them again. The channel only carries velocity commands, which the command gate
(ADR-0006) clamps, rate-limits and times out, and no stop travels over it. So
the risk is an unwanted walk within the gate limits, not a bypass of the safety
layer. It still must not be possible without the key.

## Decision
- **Pre-shared key.** A hex key file of at least 16 bytes (`--command-key FILE`
  on the runtime, `--key-file` on `scripts/teleop.py`, `--new-key` writes 32
  random bytes with mode 0600 and never overwrites). No key exchange, no
  certificates: one operator, one robot.
- **Tag.** HMAC-SHA256 over every byte before the tag, truncated to 16 bytes,
  compared in constant time. SHA-256/HMAC are implemented std-only in
  `runtime/src/auth.rs` (the crate keeps no dependencies, ADR-0007) and checked
  against the FIPS 180-4 and RFC 4231 test vectors. The Python side uses the
  standard library's `hmac`.
- **Replay.** The runtime draws a random 64-bit session number from
  `/dev/urandom` at start. A sender gets it with a hello (random client nonce) and
  a tagged welcome that echoes the nonce. Every command carries the session and
  a sequence number; the runtime accepts only the current session and a sequence
  above the last accepted one. Packets recorded in an earlier run, or earlier in
  this run, are refused.
- **Peer lock** stays: the first sender with a valid command is the only one
  accepted until the runtime restarts.
- **Default is loopback.** Without a key file both sides use an all-zero key and
  the runtime refuses to bind to any non-loopback address (exit code 2).
- **Bounded work.** At most 64 datagrams and 2 welcomes per control period, so a
  flood cannot stall the loop or turn the robot into a reflector.
- **No encryption.** Commands are not secret; only their origin matters.

## Consequences
- A forged, replayed or wrong-key packet is dropped and counted
  (`unauthenticated`, `stale`, `foreign`, `malformed` in the runtime summary).
- A restarted teleop gets a new source port and is `foreign` until the runtime
  restarts. Acceptable for now; a re-lock rule would need its own review.
- Anyone who can read the key file can drive the robot within the gate limits.
  The key stays on the operator's machine and the robot, never in the repo.
- Revisit if more than one operator, a gamepad bridge on another machine, or
  key rotation is needed.
