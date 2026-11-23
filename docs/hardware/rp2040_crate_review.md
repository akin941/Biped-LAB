# Crate review: RP2040 supervisor firmware (WP5-10, ADR-0015)

**Environment:** cloud, 2026-11-23. Versions read from crates.io with
`cargo info`; advisories checked against a fresh shallow clone of
`rustsec/advisory-db` and the versions in `firmware/supervisor-rp2040/Cargo.lock`.
Nothing has been flashed to a chip.

## Direct dependencies (exact versions, lock committed)
| Crate | Version | License | Maintainer | Why |
|---|---|---|---|---|
| `rp2040-hal` | 0.12.0 | MIT OR Apache-2.0 | rp-rs (community HAL for the RP2040) | clocks, GPIO, PWM slices, UART, ADC, timer, hardware watchdog |
| `rp2040-boot2` | 0.3.0 | BSD-3-Clause | rp-rs | second-stage boot loader for the Pico's W25Q080 flash |
| `cortex-m-rt` | 0.7.7 | MIT OR Apache-2.0 | rust-embedded | reset handler, vector table, `link.x` |
| `embedded-hal` | 1.0.0 | MIT OR Apache-2.0 | rust-embedded | `SetDutyCycle`, `OutputPin` traits |
| `embedded-hal-nb` | 1.0.0 | MIT OR Apache-2.0 | rust-embedded | `nb` result of the UART's non-blocking read |
| `panic-halt` | 1.0.0 | MIT OR Apache-2.0 | rust-embedded community | a panic stops the core; the hardware watchdog then resets the chip with outputs low |

All decisions (limits, slew, watchdog, battery, latching) stay in
`supervisor-core`, which has no third-party crates and is host-tested. The
binary only moves bytes and sets duty cycles.

## Whole tree
`Cargo.lock` lists 111 packages. Many are build-time only (proc macros, the
PIO assembler's parser, `syn`, `regex`), and none of those reach the chip. On the
thumbv6m target the licenses are MIT, Apache-2.0, BSD-3-Clause, Unlicense, or
Unicode-3.0 combined with MIT/Apache, all permissive and compatible with
ADR-0004. The release image is ~7.4 KB of code.

## Advisories (RustSec) against the locked versions
- **Fixed in the locked version**, so no action: `cortex-m-rt` (2023-0014,
  fixed ≥ 0.7.3), `smallvec`, `regex`, `rand_core`, `generic-array`,
  `hashbrown`, `lock_api`, `keccak`.
- **Unsound, not fixed in the locked version:** `heapless` 0.8.0 has nine
  2026 advisories, fixed in ≥ 0.9.4, covering `IndexMap`, `Deque`, `Spsc`, the
  treiber stack, and `SortedLinkedList`. It comes in only through `usb-device`,
  which `rp2040-hal` pulls in. The firmware has no USB and never calls these
  types. **Accepted.** Re-check when `rp2040-hal` moves to `heapless` 0.9.
- **Unmaintained (informational):**
  - `bare-metal` 0.2, through `cortex-m` 0.7.
  - `paste` and `proc-macro-error2`. Both are build-time macros only;
    `proc-macro-error2` also gives a future-incompatibility warning.
  - `term`. It is in the lock but not built for the target.
  - **Accepted**, because none of them has a known vulnerability. Re-check
    with each `rp2040-hal` upgrade.

## Rules kept
- Exact versions (`=x.y.z`) and a committed `Cargo.lock`. CI builds with `--locked`.
- No USB stack, no allocator, no `unsafe` in our code.
- `supervisor-core` keeps zero dependencies; the runtime uses it as a path crate.
- Upgrades get a new pass over this file.
