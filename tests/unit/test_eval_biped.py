"""G4.3 robustness cases of scripts/eval_biped.py (WP4-12), with the zero-action baseline."""

import importlib.util

import numpy as np

from bipedlab import REPO_ROOT

spec = importlib.util.spec_from_file_location("eval_biped", REPO_ROOT / "scripts/eval_biped.py")
assert spec is not None and spec.loader is not None
eval_biped = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_biped)


def test_robustness_cases_cover_friction_push_and_latency():
    cases = eval_biped.robustness_cases(None, num_envs=2, seconds=5.0, seed=0)
    names = [c["case"] for c in cases]
    assert names == [
        "friction_0.5",
        "friction_1.5",
        "push_forward",
        "push_sideways",
        "latency_20ms",
        "latency_40ms",
        "latency_60ms",
    ]
    assert all(not c["dr"] and c["twist"] == list(eval_biped.ROBUST_TWIST) for c in cases)
    assert [c["action_delay_ms"] for c in cases[4:]] == [20.0, 40.0, 60.0]
    assert cases[2]["push_dv"] == [eval_biped.PUSH_DV, 0.0]
    for c in cases:
        assert np.isfinite(c["knee_torque"]["p95"])


def test_push_moves_the_trunk():
    still = eval_biped.run_case(None, (0.0, 0.0, 0.0), 1, 4.5, 0, False)
    pushed = eval_biped.run_case(None, (0.0, 0.0, 0.0), 1, 4.5, 0, False, push_dv=(0.3, 0.0))
    assert pushed["falls"] == 0
    # The stance springs back (vx_mean stays ~0), but the kick loads the knees.
    assert pushed["knee_torque"]["mean"] > still["knee_torque"]["mean"]


def test_cases_report_lateral_and_yaw_rates():
    # WP4-21: the zero-action stance neither turns nor slides.
    still = eval_biped.run_case(None, (0.0, 0.0, 0.0), 1, 3.0, 0, False)
    assert abs(still["wz_mean"]) < 0.02 and abs(still["vy_mean"]) < 0.02


def test_joint_stats_and_sizing_summary():
    # WP4-25: per joint type stats; the knee holds the crouched stance against gravity.
    still = eval_biped.run_case(None, (0.15, 0.0, 0.0), 1, 3.0, 0, False)
    joints = still["joints"]
    assert set(joints) == set(eval_biped.JOINT_GROUPS)
    knee = joints["knee"]
    assert 0 < knee["torque_rms"] <= knee["torque_max"] and knee["torque_p95"] <= knee["torque_max"]
    assert knee["speed_p95"] <= knee["speed_max"] and knee["power_mean_w"] >= 0
    # The zero-action baseline is left out of the sizing; a walking case is used, with 2x.
    assert eval_biped.sizing_summary([still])["cases_used"] == 0
    walking = dict(still, policy="onnx")
    sizing = eval_biped.sizing_summary([walking])
    assert sizing["cases_used"] == 1
    assert sizing["joints"]["knee"]["required_torque"] == 2.0 * knee["torque_max"]
    assert sizing["joints"]["knee"]["required_speed"] == 2.0 * knee["speed_p95"]


def test_torque_limit_override_and_clip_flag():
    # WP4-26: the override replaces the model's 0.52 N*m force limit in every case.
    case = eval_biped.run_case(None, (0.15, 0.0, 0.0), 1, 3.0, 0, False, torque_limit=1.5)
    assert case["torque_limit"] == 1.5
    default = eval_biped.run_case(None, (0.15, 0.0, 0.0), 1, 3.0, 0, False)
    assert default["torque_limit"] == 0.52
    # A walking max at the limit is flagged as clipped (the requirement is a lower bound).
    walking = dict(default, policy="onnx")
    walking["joints"] = {k: dict(v) for k, v in default["joints"].items()}
    walking["joints"]["knee"]["torque_max"] = 0.52
    sizing = eval_biped.sizing_summary([walking])["joints"]
    assert sizing["knee"]["at_torque_limit"] and not sizing["hip_yaw"]["at_torque_limit"]


def test_motor_curve_case_records_its_settings():
    case = eval_biped.run_case(
        None, (0.15, 0.0, 0.0), 1, 3.0, 0, False, torque_limit=1.1, motor_no_load_speed=10.2
    )
    assert case["torque_limit"] == 1.1 and case["motor_no_load_speed"] == 10.2
    assert case["falls"] == 0


def test_terrain_cases_run_on_height_fields():
    # WP4-31: one case per height, the robustness command, DR off; zero action stands.
    cases = eval_biped.terrain_cases(None, num_envs=1, seconds=3.0, seed=0)
    assert [c["case"] for c in cases] == ["rough_5mm", "rough_10mm", "rough_20mm"]
    assert [c["rough_floor"] for c in cases] == list(eval_biped.TERRAIN_HEIGHTS)
    assert all(c["twist"] == list(eval_biped.ROBUST_TWIST) and not c["dr"] for c in cases)
    assert all(c["falls"] == 0 for c in cases)
