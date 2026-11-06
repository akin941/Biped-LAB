"""WP4-14: host command packets and the key teleop state."""

import pytest

from bipedlab import command_channel as cc


def test_packet_layout_matches_the_rust_decoder():
    b = cc.encode(9, 0.1, -0.2, 0.3)
    assert len(b) == 34 and b[:4] == b"BPCM" and b[4:6] == b"\x01\x00"
    assert cc.decode(b) == (9, 0.1, -0.2, 0.3)
    with pytest.raises(ValueError):
        cc.decode(b[:-1])
    with pytest.raises(ValueError):
        cc.decode(b"XXXX" + b[4:])


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
