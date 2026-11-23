"""WP5-00: the example joint map lists the model's actuators in actuator order."""

import mujoco

from bipedlab import REPO_ROOT
from bipedlab.design import biped


def test_example_joint_map_follows_the_actuator_order():
    m = mujoco.MjModel.from_xml_string(biped.to_mjcf())
    order = [m.joint(int(j)).name for j in m.actuator_trnid[:, 0]]
    path = REPO_ROOT / "hardware/calibration/biped_v1_joints.example.txt"
    rows = [ln.split("#")[0].split() for ln in path.read_text().splitlines()]
    names = [r[0] for r in rows if r]
    assert names == order
