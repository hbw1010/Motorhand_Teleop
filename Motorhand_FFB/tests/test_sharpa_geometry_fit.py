"""Sharpa 指腹几何和比例拟合的独立约束。"""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fit_sharpa_retarget as fit
from validate_sharpa_retarget import validate


@pytest.mark.parametrize("side", ["right", "left"])
def test_derived_pad_frames_preserve_source_joints_and_meshes(side):
    source = ET.parse(fit.source_urdf(side)).getroot()
    derived = ET.parse(fit.pad_urdf(side)).getroot()
    original_joints = {j.attrib["name"]: ET.tostring(j).strip() for j in source.findall("joint")}
    derived_joints = {j.attrib["name"]: ET.tostring(j).strip() for j in derived.findall("joint")}
    assert all(derived_joints[name] == xml for name, xml in original_joints.items())
    extra = set(derived_joints) - set(original_joints)
    assert len(extra) == 15
    assert all("_pad_" in name for name in extra)
    assert len([j for j in derived.findall("joint") if j.attrib["type"] == "revolute"]) == 22
    assert [ET.tostring(e) for e in source.iter("mesh")] == [ET.tostring(e) for e in derived.iter("mesh")]


def test_left_right_pad_geometry_mirrors_without_extra_rotation():
    right = fit.SharpaFK("right").points(np.zeros(22))
    left = fit.SharpaFK("left").points(np.zeros(22))
    assert np.max(np.abs(right * [1., -1., 1.] - left)) < 3e-6
    assert np.all(right[fit.TASK_INDICES[1:, 2], 0] > 0)
    assert np.all(right[fit.TASK_INDICES[1:, 2], 2] > .18)
    assert np.allclose(right[1], right[2] / 2)


def test_fixed_task_domain_preserves_physical_points_and_pinch_equality():
    scale = np.array([1.2, .87, 1.1])
    offset = np.array([-.0003, .0098, .0225])
    fk = fit.SharpaFK("right", scale, offset)
    q = np.zeros(22)
    physical, task = fk.physical_points(q), fk.points(q)
    assert np.allclose(physical, fit.SharpaFK("right").physical_points(q))
    assert np.allclose(task[2:] * scale + offset, physical[2:])
    assert np.all(task[0] == 0)
    assert np.allclose(task[1], .5 * task[2])
    for index in (8, 12, 16, 20):
        assert np.allclose((task[4] - task[index]) * scale, physical[4] - physical[index])


@pytest.mark.parametrize("side", ["right", "left"])
def test_palmar_tip_is_on_actual_mesh_surface(side):
    hand = fit.load_visual_hand(side)
    pad = fit.SharpaFK(side).points(np.zeros(22))
    for i, finger in enumerate(fit.FINGERS):
        distal = hand.placement(f"{side}_{finger}_DP").translation
        normal = pad[fit.TASK_INDICES[i, 1]] - distal
        depth = np.linalg.norm(normal)
        normal /= depth
        mesh = np.vstack([hand.mesh[f"{side}_{finger}_DP"], hand.mesh[f"{side}_{finger}_elastomer"]])
        center = pad[fit.TASK_INDICES[i, 2]] - depth * normal
        assert abs(fit.ray_surface_depth(mesh, center, normal) - depth) < .001
        check = (hand.placement(f"{side}_index_fingertip").translation -
                 hand.placement(f"{side}_thumb_fingertip").translation
                 if finger == "thumb" else np.array([1., 0., 0.]))
        assert normal @ check > 0


def test_training_only_least_squares_and_held_out_residual():
    class Reference:
        def reference(self, frame):
            return np.asarray(frame["reference"]), np.zeros(22)

    points = np.arange(63, dtype=float).reshape(21, 3) * .001
    points -= points[0]
    frame = {"points_m": points.tolist(), "reference": (points * 1.7).tolist(), "split": "train"}
    held_out = {"points_m": points.tolist(), "reference": (points * 4).tolist(), "split": "validation"}
    scales, report = fit.fit_scales([frame, frame, held_out], Reference())
    assert np.allclose(scales, 1.7)
    assert report["unique_training_frames"] == 1
    assert report["train"]["reference_residual_rms_mm"] < 1e-10
    assert report["validation"]["reference_residual_rms_mm"] > 10


def test_axis_and_wrist_fit_uses_training_geometry_only():
    fk = fit.SharpaFK("right")
    expected_scale = np.array([1.2, .8, 1.1])
    expected_offset = np.array([.001, .01, .02])
    frames = []
    for amount in (0., .2, .5):
        frame = {"joint_rad": [amount] * 12, "dependent_rad": [amount * .7] * 4, "split": "train"}
        source = (fk.reference(frame)[0] - expected_offset) / expected_scale
        source[0] = 0.
        source[1] = source[2] / 2
        frame["points_m"] = source.tolist()
        frames.append(frame)
    validation = {**frames[-1], "split": "validation", "points_m": (source * 100).tolist()}
    scale, offset, _ = fit.fit_task_geometry(frames + [validation], "right")
    assert np.allclose(scale, expected_scale)
    assert np.allclose(offset, expected_offset)


def test_config_changes_only_geometry_and_four_finger_side_axis_policy(tmp_path):
    config = fit.write_config("right", np.ones((5, 3)), path=tmp_path / "right.yaml")
    value = yaml.safe_load(config.read_text())
    r = value["retarget"]
    assert (r["huber_delta"], r["huber_delta_dir"], r["norm_delta"]) == (2., .5, .04)
    assert (r["w_pos"], r["w_dir"], r["w_full_hand"], r["w_hyper"]) == (1., 2., 1., 1.)
    assert (r["w_couple"], r["couple_ratio"], r["pinch_weight_max"]) == (.1, .7, .7)
    assert not r["thumb_skip_pip"]
    assert len(r["fixed_joints"]) == 4
    assert "right_thumb_MCP_AA" not in r["fixed_joints"]
    assert "right_pinky_CMC" not in r["fixed_joints"]
    assert Path(value["optimizer"]["urdf_path"]).is_absolute()
    fit.write_config("right", np.ones((5, 3)), path=config, norm_delta=.4)
    assert yaml.safe_load(config.read_text())["retarget"]["norm_delta"] == .4


def test_opposition_reports_contact_within_command_interval(tmp_path):
    # 闭链右食指对指保持帧的原生目标。命令几何相交，实际控制从开放零位积分。
    q = [0.69609, -0.06298, 0.933, -0.3, 0.7207, 0.76526, 0., 1.31331, 0.49811,
         0.41811, 0., 0., 0., 0.29884, 0., 0.00308, 0.0948, 0., 0.24684, 0., 0.00001, 0.19785]
    points = fit.SharpaFK("right").physical_points(np.array(q)).tolist()
    fixture = tmp_path / "fixture.json"
    commands = tmp_path / "commands.json"
    frames = [{"sequence": split, "split": split, "phase": "hold", "progress": 1.,
               "target_finger": "index", "points_m": points} for split in ("train", "validation")]
    fixture.write_text(json.dumps({"hand": "right", "frames": frames}))
    commands.write_text(json.dumps({"hand": "right", "frames": [{"q_rad": q, "physical_points_m": points}] * 2}))

    report = validate("right", fixture, commands, 500.)
    opposition = report["splits"]["validation"]["opposition"]["index"]
    first = opposition["first_actual_contact"]
    deepest = opposition["deepest_actual_contact"]
    static = opposition["deepest_command_intersection"]
    assert .0 < first["sequence_time_s"] < .5
    assert first["fixture_frame_index"] == 1
    assert first["geoms"] == ["right_thumb_elastomer", "right_index_DP"]
    assert static["distance_mm"] < deepest["distance_mm"] <= first["distance_mm"] < 0.
    assert static["distance_mm"] < -10.
