"""WP5-03: the committed hardware Hello file is what the sim bridge sends."""

import importlib.util
import struct
from pathlib import Path

from bipedlab import runtime_bridge as rb

ROOT = Path(__file__).resolve().parents[2]
HELLO = ROOT / "hardware/calibration/biped_v1_sts3215.hello"


def _script():
    spec = importlib.util.spec_from_file_location("export_hello", ROOT / "scripts/export_hello.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_hello_matches_the_model():
    data = HELLO.read_bytes()
    assert data == _script().hello_bytes(1.91), "regenerate with scripts/export_hello.py"
    n = struct.unpack_from("<H", data, 6)[0]
    assert data[:4] == rb.MAGIC and len(data) == rb.hello_size(n) and n == 10
    rows = struct.unpack_from(f"<{5 * n}d", data, 16)
    assert all(rows[k * 5 + 4] == 1.91 for k in range(n))  # STS3215 torque limit


def test_committed_ds3218_hello_matches_its_model():
    """G4.5 / WP5-09: the DS3218MG build's contract, with the model's own 1.96 N*m limit."""
    data = (ROOT / "hardware/calibration/biped_ds3218.hello").read_bytes()
    want = _script().hello_bytes(None, "biped_ds3218")
    assert data == want, "regenerate with scripts/export_hello.py"
    n = struct.unpack_from("<H", data, 6)[0]
    rows = struct.unpack_from(f"<{5 * n}d", data, 16)
    assert n == 10 and all(rows[k * 5 + 4] == 1.96 for k in range(n))
