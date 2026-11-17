"""只读Sharpa查看器的数据边界，不启动SDK或窗口。"""
from pathlib import Path

import pytest

from motorhand_teleop.sharpa_viewer import JOINT_SUFFIXES, validate_frame

ROOT = Path(__file__).resolve().parents[1]


def frame():
    return {
        "schema_version": 2, "robot": "sharpa", "hand": "right",
        "joint_names": [f"right_{name}" for name in JOINT_SUFFIXES],
        "position_rad": [0.0] * 22,
        "scene_path": str(ROOT / "assets/sharpa-hand/right.xml"),
        "qpos": [0.0] * 22, "load_nm": [0.0] * 22,
        "load_source": "sharpa-mujoco-contact-joint-torque", "simulated": True,
    }


def test_accepts_complete_22_axis_simulation_frame():
    result = validate_frame(frame(), ROOT)
    assert result.scene_path == ROOT / "assets/sharpa-hand/right.xml"
    assert len(result.position_rad) == 22
    assert result.simulated


@pytest.mark.parametrize("field,value", [
    ("schema_version", 1), ("schema_version", True), ("robot", "wuji"),
    ("hand", "invalid"), ("position_rad", [0.0] * 21),
    ("position_rad", [float("nan")] * 22), ("position_rad", [True] * 22),
    ("joint_names", ["right_thumb_CMC_FE"] * 22),
    ("joint_names", [f"left_{name}" for name in JOINT_SUFFIXES]),
    ("qpos", []), ("qpos", [float("inf")]), ("load_nm", [0.0] * 21),
    ("simulated", "false"), ("load_source", "wuji-current"),
])
def test_rejects_invalid_schema_or_axis_data(field, value):
    payload = frame()
    payload[field] = value
    with pytest.raises(ValueError):
        validate_frame(payload, ROOT)


@pytest.mark.parametrize("scene_path", [
    "/tmp/outside-sharpa-viewer.xml", "assets/sharpa-hand/right.xml",
    str(ROOT / "assets/sharpa-hand/right.xml/../../../outside.xml"),
    str(ROOT / "README.md"),
])
def test_rejects_missing_outside_relative_or_non_xml_scene(scene_path):
    payload = frame()
    payload["scene_path"] = scene_path
    with pytest.raises(ValueError):
        validate_frame(payload, ROOT)


def test_rejects_symlink_that_escapes_project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.xml"
    outside.write_text("<mujoco/>")
    (root / "scene.xml").symlink_to(outside)
    payload = frame()
    payload["scene_path"] = str(root / "scene.xml")
    with pytest.raises(ValueError):
        validate_frame(payload, root)


def test_hardware_position_frame_does_not_require_load_or_qpos():
    payload = frame()
    payload["simulated"] = False
    payload["load_nm"] = None
    payload["load_source"] = "sharpa-sdk-reported-joint-torque"
    del payload["qpos"]
    result = validate_frame(payload, ROOT)
    assert result.qpos is None and result.load_nm is None


def test_hardware_display_maps_semantic_joint_names_even_when_reordered():
    import mujoco
    from motorhand_teleop.sharpa_viewer import apply_frame

    payload = frame()
    payload["simulated"] = False
    payload["load_source"] = "sharpa-sdk-reported-joint-torque"
    payload["joint_names"].reverse()
    payload["position_rad"] = [index / 100 for index in range(22)]
    parsed = validate_frame(payload, ROOT)
    model = mujoco.MjModel.from_xml_path(str(parsed.scene_path))
    data = mujoco.MjData(model)
    apply_frame(model, data, parsed)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_thumb_CMC_FE")
    assert data.qpos[model.jnt_qposadr[joint]] == pytest.approx(.21)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_pinky_DIP")
    assert data.qpos[model.jnt_qposadr[joint]] == 0
    assert data.time == 0  # mj_forward只刷新展示，不推进物理模拟。


def test_simulation_display_requires_full_model_qpos_and_preserves_state():
    import mujoco
    from motorhand_teleop.sharpa_viewer import apply_frame

    payload = frame()
    model = mujoco.MjModel.from_xml_path(payload["scene_path"])
    data = mujoco.MjData(model)
    payload["qpos"] = model.qpos0.tolist()
    payload["qpos"][-1] = .2
    apply_frame(model, data, validate_frame(payload, ROOT))
    assert data.qpos[-1] == pytest.approx(.2)
    assert data.time == 0
    payload["qpos"] = [0.0]
    with pytest.raises(ValueError, match="qpos需要"):
        apply_frame(model, data, validate_frame(payload, ROOT))
