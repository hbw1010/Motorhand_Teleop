from types import SimpleNamespace

import numpy as np

from wujihand2_force_glove.retargeting import (
    WujiHand2Retargeter,
    wuji_hand2_joint_names,
)


class FakeRetargeter:
    def __init__(self, joint_names):
        self.calls = []
        robot = SimpleNamespace(dof_joint_names=joint_names)
        self.optimizer = SimpleNamespace(robot=robot, solve=self.solve)
        self.lp_filter = SimpleNamespace(next=lambda values: values)

    def solve(self, points):
        self.calls.append(points.copy())
        return np.arange(20, dtype=np.float64)

    def reset(self):
        pass


def test_retargeter_reorders_optimizer_output_by_hand2_semantic_names():
    expected = wuji_hand2_joint_names("right")
    core = FakeRetargeter(tuple(reversed(expected)))
    adapter = WujiHand2Retargeter(core, hand="right")
    points = np.arange(63, dtype=np.float64).reshape(21, 3) * 0.001
    result = adapter.step(points)

    assert result.shape == (20,)
    assert np.array_equal(result, np.arange(19, -1, -1, dtype=np.float64))
    assert core.calls[0].shape == (21, 3)


def test_retargeter_can_use_fixed_model_and_optimizer_coordinate_rotations():
    expected = wuji_hand2_joint_names("right")
    core = FakeRetargeter(tuple(reversed(expected)))
    adapter = WujiHand2Retargeter(core, hand="right")
    points = np.arange(63, dtype=np.float64).reshape(21, 3) * 0.001
    model_rotation = np.diag([1.0, -1.0, -1.0])
    optimizer_rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    adapter.step(
        points,
        model_to_canonical_row=model_rotation,
        canonical_to_optimizer_row=optimizer_rotation,
    )

    assert np.allclose(
        core.calls[0],
        (points - points[0]) @ model_rotation @ optimizer_rotation,
    )
