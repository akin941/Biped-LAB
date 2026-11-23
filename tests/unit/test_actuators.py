import math

from bipedlab.design import actuators as act


def _get(name: str, volts: float) -> act.Candidate:
    return next(c for c in act.CANDIDATES if c.name == name and c.volts == volts)


def _req(joint: str) -> act.Requirement:
    return next(r for r in act.WP4_01 if r.joint == joint)


def test_candidates_within_supply_range() -> None:
    for c in act.CANDIDATES:
        lo, hi = c.supply_range
        assert lo <= c.volts <= hi, c
        assert c.stall_torque > 0 and c.no_load_speed > 0 and c.mass > 0


def test_fit_margins_and_loaded_speed() -> None:
    c = act.Candidate("t", 5.0, 1.0, 10.0, 1.0, 0.02, (4.0, 6.0), "-", "-")
    f = act.fit(c, act.Requirement("j", 0.25, 2.0, 0.1))
    assert f["torque_margin"] == 4.0 and f["speed_margin"] == 5.0
    assert math.isclose(f["loaded_speed_rad_s"], 7.5)
    assert f["passes"]
    assert not act.fit(c, act.Requirement("j", 0.6, 2.0, 0.1))["passes"]  # torque x1.67


def test_xl330_fails_the_knee_as_in_wp4_01() -> None:
    assert not act.fit(_get("XL330-M288-T", 5.0), _req("knee"))["passes"]


def test_some_candidate_covers_every_joint() -> None:
    for r in act.WP4_01:
        assert any(act.fit(c, r)["passes"] for c in act.CANDIDATES), r.joint


def test_gait_requirements_cover_the_same_joints():
    # WP4-27: the MAC-36 gait requirements replace WP4-01's joint by joint.
    from bipedlab.design import actuators as act

    assert [r.joint for r in act.GAIT_MAC36] == [r.joint for r in act.WP4_01]
    assert set(act.REQUIREMENTS) == {"wp4-01", "mac36"}
    # Our gait needs more than the WP4-01 estimate at every joint.
    for new, old in zip(act.GAIT_MAC36, act.WP4_01, strict=True):
        assert new.peak_torque > old.peak_torque


def test_low_budget_pwm_servos() -> None:
    """G4.5: DS3218MG covers the MAC-36 hip torques with margin; MG90S cannot."""
    by_name = {c.name: c for c in act.CANDIDATES}
    ds = by_name["DS3218MG (6 V)"]
    mg90 = by_name["MG90S (6 V)"]
    for r in act.GAIT_MAC36:
        assert act.fit(ds, r)["torque_margin"] >= 2.0
        assert act.fit(mg90, r)["torque_margin"] < 2.0
    hip_roll = next(r for r in act.GAIT_MAC36 if r.joint == "hip_roll")
    assert act.fit(mg90, hip_roll)["torque_margin"] < 1.0
