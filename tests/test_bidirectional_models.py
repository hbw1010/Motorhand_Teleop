from pathlib import Path

import numpy as np

from wujihand2_force_glove.bidirectional import (
    BidirectionalRetargeter,
    GloveInverseSolver,
    GloveTaskModel,
    Hand2TaskModel,
)
from wujihand2_force_glove.domain import GLOVE_JOINTS
from wujihand2_force_glove.landmarks import Ve33LandmarkSource
from wujihand2_force_glove.retargeting import (
    GloveWujiRetargetPipeline,
    default_retarget_config,
)


ROOT = Path(__file__).parents[1]
ASSET_ROOT = ROOT / "assets" / "hand_motor_ve33"


def _bridge() -> BidirectionalRetargeter:
    source = Ve33LandmarkSource("right", asset_root=ASSET_ROOT)
    pipeline = GloveWujiRetargetPipeline.from_yaml(
        "right",
        asset_root=ASSET_ROOT,
        config_path=default_retarget_config(ROOT, "right"),
        estimate_jacobian=False,
    )
    glove_model = GloveTaskModel(source)
    return BidirectionalRetargeter(
        pipeline,
        inverse=GloveInverseSolver(glove_model, max_nfev=40),
        hand2_model=Hand2TaskModel(pipeline.retargeter, source.canonical_to_optimizer_row),
    )


def test_forward_retarget_locks_four_finger_mcp_abduction():
    bridge = _bridge()
    joints = {name: 0.0 for name in GLOVE_JOINTS}
    pose, task = bridge.glove_to_hand2(joints)

    assert pose.positions_rad.shape == (20,)
    assert np.allclose(pose.positions_rad[[5, 9, 13, 17]], 0.0, atol=1e-12)
    assert task.points_m.shape == (21, 3)
    assert pose.diagnostics["lateral_policy"] == "fixed_zero"


def test_shared_task_space_supports_reverse_glove_solve():
    bridge = _bridge()
    joints = {name: 0.0 for name in GLOVE_JOINTS}
    joints["index_pip"] = 0.2
    joints["middle_pip"] = 0.1
    forward_pose, _ = bridge.glove_to_hand2(joints)
    result, hand_task = bridge.hand2_to_glove(forward_pose.positions_rad)

    assert result.success
    assert result.glove_positions_rad.shape == (12,)
    assert hand_task.points_m.shape == (21, 3)
    assert np.isfinite(result.task_error_rms_mm)
    assert result.task_error_rms_mm < 30.0
