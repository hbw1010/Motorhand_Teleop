"""真实CAD装配的电机正解/指节逆解；不依赖DexGlove包或实体设备。"""
import json
from pathlib import Path
import runpy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from motorhand_teleop.ve33_kinematics import Ve33Kinematics

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["right", "left"])
def mechanism(request):
    side = "R" if request.param == "right" else "L"
    directory = ROOT / "assets" / "hand_motor_ve33" / f"hand_motor_ve33_{side}"
    core = Ve33Kinematics(request.param, model_dir=directory)
    namespace = runpy.run_path(str(directory / "scripts/solve_closed_chain.py"))
    reference = namespace["ClosureSolver"](directory / "hand_tree_only.urdf", directory / "config/loop_closures.yaml")
    definitions = json.loads((directory / "config/joint_space.json").read_text())["independent"]
    return core, reference, definitions


def measured_joints(reference, definitions, scene):
    result = {}
    zero = dict.fromkeys(reference.independent + reference.dependent, 0.0)
    for item in definitions:
        parent0 = reference.fk.world(item["parent"], zero)[:3, :3]
        child0 = reference.fk.world(item["child"], zero)[:3, :3]
        axis = child0 @ reference.fk.j[item["axis_joint"]][4]
        axis /= np.linalg.norm(axis)
        if axis @ child0[:, 2] < 0:
            axis = -axis
        parent = reference.fk.world(item["parent"], scene)[:3, :3]
        child = reference.fk.world(item["child"], scene)[:3, :3]
        delta = parent.T @ child @ (parent0.T @ child0).T
        result[item["name"]] = float((parent0.T @ axis) @ Rotation.from_matrix(delta).as_rotvec())
    return result


def test_motor_feedback_is_real_joint_pose_and_roundtrips_without_branch_jump(mechanism):
    core, reference, definitions = mechanism
    motors = {name: 0.04 + 0.007 * index for index, name in enumerate(reference.independent)}
    scene = reference.solve(motors)
    expected = measured_joints(reference, definitions, scene)
    initial = core.current
    trial = core.evaluate_motors(motors)
    assert core.current is initial
    assert trial.joint_positions == pytest.approx(expected, abs=2e-8)
    assert abs(trial.joint_positions["index_pip_joint"] - motors["index_pip_dip_motor_joint"]) > 0.01
    committed = core.solve_motors(motors)
    inverse = core.evaluate(committed.joint_positions)
    assert inverse.motor_positions == pytest.approx(motors, abs=2e-8)
    assert inverse.max_gap_m < 1e-7
    assert core.current is committed
    for invalid in ({"index_pip_joint": 0.1}, {core.motor_names[0]: float("nan")}, {core.motor_names[0]: 6.0}):
        with pytest.raises(ValueError):
            core.solve_motors(invalid)
        assert core.current is committed


def test_full_tree_geometry_is_verified_without_committing_motor_input(mechanism):
    core, reference, definitions = mechanism
    motors = {name: 0.03 + 0.005 * index for index, name in enumerate(reference.independent)}
    scene = reference.solve(motors)
    initial = core.current
    result = core.evaluate_tree(scene)
    assert set(core.tree_names) == set(scene)
    assert result.joint_positions == pytest.approx(measured_joints(reference, definitions, scene), abs=2e-8)
    assert core.current is initial
    broken = dict(scene)
    broken["index_pip_dip_motor_joint"] += 0.2
    for invalid in (broken, {**scene, "unknown_joint": 0.0}, {**scene, core.motor_names[0]: float("inf")}):
        with pytest.raises(ValueError):
            core.evaluate_tree(invalid)
        assert core.current is initial
