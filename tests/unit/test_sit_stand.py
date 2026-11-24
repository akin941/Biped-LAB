"""WP5-19: open-loop sit / stand ramps and power-off drops on the 1.17 kg model."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("sit_stand", ROOT / "scripts/sit_stand.py")
assert spec and spec.loader
sit_stand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sit_stand)

MODEL = "biped_ds3218b_heavy"


def test_deep_sit_and_stand_up_stay_upright_inside_the_torque_limit():
    r = sit_stand.ramp(MODEL, knee=2.0, seconds=1.0)
    assert r["upright"] and r["back_to_stand"]
    assert r["lowest_trunk_m"] < 0.21
    assert max(r["peak_torque_nm"].values()) < 0.5 * r["torque_limit_nm"]


def test_power_off_lands_on_the_floor_and_sitting_first_lands_softer():
    stand = sit_stand.drop(MODEL, knee=0.6)
    sit = sit_stand.drop(MODEL, knee=2.0)
    for r in (stand, sit):  # the trunk ends on the floor, not through it
        assert not r["upright_after"] and 0.0 < r["trunk_after_m"] < 0.05
    assert sit["peak_down_speed_mps"] < 0.8 * stand["peak_down_speed_mps"]
    # A gearbox that holds 0.2 N*m unpowered keeps the robot where it was.
    assert sit_stand.drop(MODEL, knee=0.6, friction=0.2)["upright_after"]
