from pathlib import Path

import mujoco
import numpy as np
import yaml

from motorhand_teleop.mujoco_sim import build_glove_mjcf
from motorhand_teleop.retargeting import wuji_hand2_joint_names


ROOT = Path(__file__).parents[1]


def test_official_hand2_mjcf_matches_retarget_joint_order():
    config = yaml.safe_load((ROOT / "config/simulation.yaml").read_text(encoding="utf-8"))
    model_path = ROOT / config["model"]["mjcf_path"]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    assert model.nq == 20
    assert [model.joint(index).name for index in range(model.njnt)] == list(wuji_hand2_joint_names("right"))


def test_viewer_mount_uses_canonical_rigid_registration():
    from motorhand_teleop.domain import GLOVE_JOINTS
    from motorhand_teleop.landmarks import Ve33LandmarkSource
    from motorhand_teleop.mujoco_viewer import _HAND_JOINTS, _glove_to_hand_mount
    from scipy.spatial.transform import Rotation

    source = Ve33LandmarkSource("right", asset_root=ROOT / "assets" / "hand_motor_ve33")
    points = source.step({name: 0.0 for name in GLOVE_JOINTS}).points_m
    mount_pos, mount_quat, registration_rms = _glove_to_hand_mount(
        ROOT,
        [0.0] * len(_HAND_JOINTS),
        points,
        source.model_to_canonical_row,
    )
    rotation = Rotation.from_quat(
        [mount_quat[1], mount_quat[2], mount_quat[3], mount_quat[0]]
    ).as_matrix()

    canonical_to_hand = np.diag([1.0, -1.0, -1.0])
    finger_axis_correction = Rotation.from_rotvec([0.0, 0.0, np.pi / 2.0]).as_matrix()
    expected = Rotation.from_matrix(
        finger_axis_correction @ canonical_to_hand @ source.model_to_canonical_row.T
    )
    wrist_angle = (expected.inv() * Rotation.from_matrix(rotation)).magnitude()

    assert np.allclose(mount_pos, np.zeros(3), atol=1e-9)
    assert np.isfinite(mount_pos).all()
    assert np.isfinite(mount_quat).all()
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    assert wrist_angle < np.deg2rad(5.0)
    assert registration_rms < 0.30



def test_hand2_viewer_uses_position_targets_and_moves_the_hand():
    from motorhand_teleop.mujoco_viewer import _HAND_JOINTS, _hand_position_actuators

    model_path = ROOT / "vendor/wuji-retargeting/wuji_retargeting/wuji-description/hand2/hand2_beta1/body/mjcf/right.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    actuator_ids = _hand_position_actuators(model, _HAND_JOINTS)
    target = np.zeros(20)
    target[6] = 0.8
    data.ctrl[actuator_ids] = np.clip(
        target,
        model.actuator_ctrlrange[actuator_ids, 0],
        model.actuator_ctrlrange[actuator_ids, 1],
    )
    for _ in range(500):
        mujoco.mj_step(model, data)

    index_pip = int(model.joint("r_index_finger_pip").qposadr[0])
    assert data.qpos[index_pip] > 0.1

def test_glove_simulation_model_has_twelve_actuated_joints():
    model = mujoco.MjModel.from_xml_string(build_glove_mjcf())
    assert model.nq == 12
    assert model.nu == 12
