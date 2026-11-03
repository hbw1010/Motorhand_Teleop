"""力反馈分片必须保留原完整中心差分的数值语义，不连接硬件。"""
import numpy as np
import pytest

from motorhand_teleop.domain import GLOVE_JOINTS
from motorhand_teleop.real_hand.force_geometry import ForceJacobianColumns
from motorhand_teleop.real_hand.runtime import RetargetPipelineFactory


@pytest.mark.parametrize('positions', [
    [0.0] * 12,
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.20, 0.0, 0.10, 0.0, 0.15, 0.0, 0.12],
    [-0.26636, 0.00021, 0.00098, 0.00006, 0.00555, 1.62523,
     -0.00059, 1.51784, -0.00087, 1.59746, -0.00017, 0.00006],
])
def test_force_columns_preserve_complete_retarget_jacobian_at_open_and_folded_poses(positions):
    q = np.asarray(positions)
    reference_pipeline = RetargetPipelineFactory()()
    reference = reference_pipeline.local_jacobian(dict(zip(GLOVE_JOINTS, q)), step_rad=0.01)
    pieces = [
        ForceJacobianColumns(RetargetPipelineFactory()()).compute(
            q, tuple(range(start, start + 3)), step_rad=0.01,
        )
        for start in range(0, 12, 3)
    ]
    actual = np.concatenate(pieces, axis=1)
    np.testing.assert_allclose(actual, reference, atol=1e-8, rtol=1e-7)
    np.testing.assert_array_equal(actual[[5, 9, 13, 17]], np.zeros((4, 12)))
