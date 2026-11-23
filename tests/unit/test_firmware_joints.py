"""WP5-10: the supervisor firmware's joint ranges are the DS3218MG model's."""

import re
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _firmware_ranges() -> list[tuple[float, float]]:
    src = (ROOT / "firmware/supervisor-core/src/joints.rs").read_text()
    block = re.search(r"pub const RANGES[^=]*=\s*\[(.*?)\];", src, re.S).group(1)
    return [(float(a), float(b)) for a, b in re.findall(r"\(\s*([-\d.]+),\s*([-\d.]+)\s*\)", block)]


def test_firmware_ranges_match_the_ds3218_hello():
    data = (ROOT / "hardware/calibration/biped_ds3218.hello").read_bytes()
    n = struct.unpack_from("<H", data, 6)[0]
    rows = struct.unpack_from(f"<{5 * n}d", data, 16)
    model = [(rows[5 * k], rows[5 * k + 1]) for k in range(n)]
    fw = _firmware_ranges()
    assert len(fw) == n == 10
    for k, (got, want) in enumerate(zip(fw, model, strict=True)):
        assert got == pytest.approx(want, abs=1e-6), f"joint {k}"
