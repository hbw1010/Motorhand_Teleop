import importlib.util
import json

from pathlib import Path

import numpy as np
import pytest

from motorhand_teleop.domain import GLOVE_JOINTS
from motorhand_teleop.ve33_adapter import Ve33GloveModel


ROOT = Path(__file__).parents[1]


def test_real_ve33_solver_returns_motor_and_dependent_coordinates():
    model = Ve33GloveModel("right", asset_root=ROOT.parent / "assets/hand_motor_ve33")
    pose, motors = model.solve({name: 0.0 for name in GLOVE_JOINTS})
    assert set(pose.joints) == set(GLOVE_JOINTS)
    assert len(motors) == 12
    assert pose.dependent_jacobian.shape == (4, 12)
    assert np.isfinite(pose.dependent_jacobian).all()
    assert np.isfinite(list(motors.values())).all()


def test_real_ve33_transmission_jacobian_is_finite():
    model = Ve33GloveModel("right", asset_root=ROOT.parent / "assets/hand_motor_ve33")
    pose, _ = model.solve({name: 0.0 for name in GLOVE_JOINTS})
    jacobian = model.transmission().motor_jacobian(pose.vector())
    assert jacobian.shape == (12, 12)
    assert np.isfinite(jacobian).all()


def cad_oracle(hand, motors):
    """Independent original CAD closure plus adjacent-body rotation oracle."""
    side = "L" if hand == "left" else "R"
    root = ROOT.parent / "assets/hand_motor_ve33" / f"hand_motor_ve33_{side}"
    spec = importlib.util.spec_from_file_location(f"oracle_{side}", root / "scripts/solve_closed_chain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    solver = module.ClosureSolver(root / "hand_tree_only.urdf", root / "config/loop_closures.yaml")
    state = solver.solve(motors)
    config = json.loads((root / "config/joint_space.json").read_text())
    angles = {}
    for item in (*config["independent"], *config["dependent"]):
        parent, child = item["parent"], item["child"]
        r0 = solver.fk.world(parent, {})[:3, :3].T @ solver.fk.world(child, {})[:3, :3]
        relative = solver.fk.world(parent, state)[:3, :3].T @ solver.fk.world(child, state)[:3, :3]
        rotation = r0.T @ relative
        _, axis_child, _, _, axis_local = solver.fk.j[item["axis_joint"]]
        axis = solver.fk.world(axis_child, {})[:3, :3] @ axis_local
        child_zero = solver.fk.world(child, {})[:3, :3]
        axis *= 1.0 if axis @ child_zero[:, 2] > 0.0 else -1.0
        axis = child_zero.T @ axis
        axis /= np.linalg.norm(axis)
        skew = np.array([rotation[2, 1] - rotation[1, 2],
                         rotation[0, 2] - rotation[2, 0],
                         rotation[1, 0] - rotation[0, 1]]) / 2.0
        angles[item["name"].removesuffix("_joint")] = np.arctan2(axis @ skew, (np.trace(rotation) - 1.0) / 2.0)
    return angles, state, solver


@pytest.mark.parametrize("hand", ["left", "right"])
def test_motor_forward_and_joint_inverse_match_original_cad_adjacent_rotations(hand):
    model = Ve33GloveModel(hand, asset_root=ROOT.parent / "assets/hand_motor_ve33")
    motors = {name: 0.04 + index * 0.002 for index, name in enumerate(model.motor_names)}
    expected, _, _ = cad_oracle(hand, motors)
    pose, actual_motors = model.evaluate_motors(motors, timestamp_ns=123)
    assert pose.timestamp_ns == 123
    assert actual_motors == pytest.approx(motors, abs=1e-9)
    assert pose.joints == pytest.approx({name: expected[name] for name in GLOVE_JOINTS}, abs=2e-6)
    assert pose.dependent == pytest.approx({name: expected[name] for name in pose.dependent}, abs=2e-6)
    inverse = model.evaluate_raw(dict(pose.joints))
    assert inverse.motor_positions == pytest.approx(motors, abs=2e-5)
    assert not np.allclose(pose.vector(), list(motors.values()), atol=1e-3)
    larger = {name: value * 2 for name, value in motors.items()}
    larger_pose, _ = model.evaluate_motors(larger)
    assert np.max(np.abs(larger_pose.vector() - 2 * pose.vector())) > 1e-5
