"""WP5-11: turn the RP2040 supervisor firmware (ELF) into a UF2 file.

A Raspberry Pi Pico held in BOOTSEL mode shows up as a USB drive; copying a
``.uf2`` file onto it flashes the chip. This script does the conversion with
the standard library only, so no extra tool is needed (CI uploads the result as
the ``supervisor-firmware`` artifact).

Usage:
  uv run python scripts/elf_to_uf2.py \\
      firmware/supervisor-rp2040/target/thumbv6m-none-eabi/release/supervisor-rp2040 \\
      --out supervisor-rp2040.uf2
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

UF2_MAGIC0 = 0x0A324655
UF2_MAGIC1 = 0x9E5D5157
UF2_MAGIC_END = 0x0AB16F30
UF2_FLAG_FAMILY_ID = 0x00002000
RP2040_FAMILY_ID = 0xE48BFF56
PAGE = 256
FLASH_START = 0x10000000
FLASH_END = FLASH_START + 2 * 1024 * 1024  # Pico: 2 MiB
PT_LOAD = 1


def flash_image(elf: bytes) -> dict[int, bytes]:
    """Loadable ELF32 segments placed by their load (physical) address, as 256-byte pages."""
    if elf[:4] != b"\x7fELF" or elf[4] != 1 or elf[5] != 1:
        raise ValueError("not a 32-bit little-endian ELF file")
    phoff = struct.unpack_from("<I", elf, 0x1C)[0]
    phentsize, phnum = struct.unpack_from("<HH", elf, 0x2A)
    pages: dict[int, bytearray] = {}
    for i in range(phnum):
        p_type, offset, _vaddr, paddr, filesz = struct.unpack_from(
            "<IIIII", elf, phoff + i * phentsize
        )
        if p_type != PT_LOAD or filesz == 0:
            continue
        if not (paddr >= FLASH_START and paddr + filesz <= FLASH_END):
            raise ValueError(f"segment at {paddr:#010x} (+{filesz}) is outside the Pico's flash")
        data = elf[offset : offset + filesz]
        for k, byte in enumerate(data):
            addr = paddr + k
            page = pages.setdefault(addr - addr % PAGE, bytearray(PAGE))
            page[addr % PAGE] = byte
    if FLASH_START not in pages:
        raise ValueError("no boot2 block at the start of flash (0x10000000)")
    return {a: bytes(p) for a, p in sorted(pages.items())}


def uf2(pages: dict[int, bytes]) -> bytes:
    out = bytearray()
    n = len(pages)
    for k, (addr, data) in enumerate(pages.items()):
        head = struct.pack(
            "<IIIIIIII",
            UF2_MAGIC0,
            UF2_MAGIC1,
            UF2_FLAG_FAMILY_ID,
            addr,
            PAGE,
            k,
            n,
            RP2040_FAMILY_ID,
        )
        block = head + data + bytes(476 - PAGE) + struct.pack("<I", UF2_MAGIC_END)
        assert len(block) == 512
        out += block
    return bytes(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("elf", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pages = flash_image(args.elf.read_bytes())
    args.out.write_bytes(uf2(pages))
    print(f"wrote {args.out}: {len(pages)} blocks, {len(pages) * PAGE} bytes of flash")


if __name__ == "__main__":
    main()
