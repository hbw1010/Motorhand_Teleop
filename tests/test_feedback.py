import numpy as np
import pytest

from wujihand2_force_glove.domain import RetargetedPose
from wujihand2_force_glove.feedback import (
    CalibrationRequired,
    CalibratedFeedback,
    CurrentTorqueCalibration,
)


def test_feedback_requires_calibration_and_full_retarget_jacobian():
    pose = RetargetedPose(np.zeros(20), None)
    with pytest.raises(CalibrationRequired):
        CalibratedFeedback().project_effort(pose, np.zeros(20))


def test_calibrated_feedback_projects_through_retarget_jacobian():
    jacobian = np.zeros((20, 12))
    jacobian[0, 0] = 2.0
    pose = RetargetedPose(np.zeros(20), jacobian)
    feedback = CalibratedFeedback(
        CurrentTorqueCalibration(
            torque_per_amp_nm=np.ones(20),
            bias_a=np.zeros(20),
            calibrated=True,
            source_id="test-calibration",
        )
    )
    result = feedback.project_effort(pose, np.r_[3.0, np.zeros(19)])
    assert np.allclose(result.glove_torque_nm, np.array([6.0] + [0.0] * 11))
