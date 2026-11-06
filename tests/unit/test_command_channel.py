"""WP4-14/16: authenticated host command packets and the key teleop state."""

import hashlib
import hmac
import stat
import struct

import pytest

from bipedlab import command_channel as cc

KEY = bytes(range(32))


def test_packet_sizes_match_the_rust_decoder():
    assert len(cc.encode_hello(1)) == 14
    assert len(cc.encode_welcome(KEY, 1, 2)) == 38
    b = cc.encode_command(KEY, 0xABC, 9, 0.1, -0.2, 0.3)
    assert len(b) == 62 and b[:4] == b"BPCM" and b[4:6] == b"\x02\x00"
    assert struct.unpack_from("<QQ3d", b, 6) == (0xABC, 9, 0.1, -0.2, 0.3)
    want = hmac.new(KEY, b[:46], hashlib.sha256).digest()[:16]
    assert b[46:] == want


def test_welcome_is_checked_for_key_nonce_and_shape():
    w = cc.encode_welcome(KEY, 77, 1234)
    assert cc.decode_welcome(KEY, w, 77) == 1234
    with pytest.raises(ValueError, match="authentication"):
        cc.decode_welcome(bytes(32), w, 77)
    with pytest.raises(ValueError, match="another hello"):
        cc.decode_welcome(KEY, w, 78)
    with pytest.raises(ValueError):
        cc.decode_welcome(KEY, w[:-1], 77)
    tampered = w[:14] + bytes([w[14] ^ 1]) + w[15:]  # flip a session bit
    with pytest.raises(ValueError):
        cc.decode_welcome(KEY, tampered, 77)


def test_key_files(tmp_path):
    p = cc.new_key_file(tmp_path / "k.hex")
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert len(cc.load_key(p)) == 32
    with pytest.raises(FileExistsError):
        cc.new_key_file(p)  # never overwrite a key
    short = tmp_path / "short.hex"
    short.write_text("00" * 8)
    with pytest.raises(ValueError):
        cc.load_key(short)


def test_send_needs_connect():
    tx = cc.CommandSender("127.0.0.1", 9)
    with pytest.raises(RuntimeError):
        tx.send(0.1)
    tx.close()


def test_key_teleop_steps_clamps_and_stops():
    t = cc.KeyTeleop(step=0.25)
    assert t.press("w") == (0.075, 0.0, 0.0)
    for _ in range(10):
        t.press("w")
    assert t.command()[0] == 0.3  # clamped at the gate maximum
    t.press("a")
    t.press("e")
    assert t.command() == (0.3, 0.0375, -0.25)
    assert t.press(" ") == (0.0, 0.0, 0.0)
    assert t.press("z") == (0.0, 0.0, 0.0)  # unknown keys do nothing
