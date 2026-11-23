"""WP5-11: the UF2 converter places ELF load segments in RP2040 flash blocks."""

import importlib.util
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _mod():
    spec = importlib.util.spec_from_file_location("elf_to_uf2", ROOT / "scripts/elf_to_uf2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _elf(segments):
    """A minimal ELF32 LE with PT_LOAD segments [(paddr, data)]."""
    phoff = 52
    data_off = phoff + 32 * len(segments)
    head = bytearray(52)
    head[:6] = b"\x7fELF\x01\x01"
    struct.pack_into("<I", head, 0x1C, phoff)
    struct.pack_into("<HH", head, 0x2A, 32, len(segments))
    ph, body = bytearray(), bytearray()
    for paddr, data in segments:
        ph += struct.pack(
            "<IIIIIIII", 1, data_off + len(body), paddr, paddr, len(data), len(data), 5, 4
        )
        body += data
    return bytes(head + ph + body)


def test_blocks_carry_the_segments_at_their_flash_addresses():
    m = _mod()
    boot2 = bytes(range(256))
    code = b"\xaa" * 300  # spans two pages
    elf = _elf([(0x10000000, boot2), (0x10000100, code)])
    out = m.uf2(m.flash_image(elf))
    assert len(out) == 3 * 512
    for k in range(3):
        b = out[512 * k : 512 * (k + 1)]
        magic0, magic1, flags, addr, size, no, n, fam = struct.unpack_from("<IIIIIIII", b)
        assert (magic0, magic1, flags, size, no, n, fam) == (
            m.UF2_MAGIC0, m.UF2_MAGIC1, m.UF2_FLAG_FAMILY_ID, 256, k, 3, m.RP2040_FAMILY_ID,
        )  # fmt: skip
        assert addr == 0x10000000 + 256 * k
        assert struct.unpack_from("<I", b, 508)[0] == m.UF2_MAGIC_END
    assert out[32 : 32 + 256] == boot2
    assert out[512 + 32 : 512 + 32 + 256] == b"\xaa" * 256
    last = out[1024 + 32 : 1024 + 32 + 256]
    assert last[:44] == b"\xaa" * 44 and last[44:] == bytes(212)


def test_bad_images_are_refused():
    m = _mod()
    with pytest.raises(ValueError, match="ELF"):
        m.flash_image(b"not an elf")
    with pytest.raises(ValueError, match="outside"):
        m.flash_image(_elf([(0x20000000, b"\x00" * 4)]))
    with pytest.raises(ValueError, match="boot2"):
        m.flash_image(_elf([(0x10000100, b"\x00" * 4)]))
