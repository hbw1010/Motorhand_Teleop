from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .domain import RetargetedPose


class CalibrationRequired(RuntimeError):
    """未提供可追溯的电流到力矩标定，禁止生成触觉命令。"""


@dataclass(frozen=True)
class CurrentTorqueCalibration:
    """单机标定参数；torque_per_amp_nm 未经实测时不得构造此对象。"""

    torque_per_amp_nm: np.ndarray
    bias_a: np.ndarray
    calibrated: bool
    source_id: str

    def __post_init__(self) -> None:
        for name in ("torque_per_amp_nm", "bias_a"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (20,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name}必须是有限的20维数组")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not self.source_id:
            raise ValueError("source_id不能为空")


@dataclass(frozen=True)
class FeedbackProjection:
    glove_torque_nm: np.ndarray
    calibrated: bool
    source_id: str


class CalibratedFeedback:
    def __init__(self, calibration: CurrentTorqueCalibration | None = None):
        self.calibration = calibration

    def project_effort(self, pose: RetargetedPose, effort_a: np.ndarray) -> FeedbackProjection:
        if self.calibration is None or not self.calibration.calibrated:
            raise CalibrationRequired("Wuji effort/current 仍是 A，完成单机标定前禁止进入触觉闭环")
        if pose.jacobian_wuji_glove is None:
            raise CalibrationRequired("缺少完整重定向雅可比，禁止把反馈力矩投影到手套")
        effort = np.asarray(effort_a, dtype=np.float64)
        if effort.shape != (20,) or not np.all(np.isfinite(effort)):
            raise ValueError("effort_a必须是有限的20维数组")
        torque_nm = (effort - self.calibration.bias_a) * self.calibration.torque_per_amp_nm
        glove_torque = pose.jacobian_wuji_glove.T @ torque_nm
        return FeedbackProjection(glove_torque, True, self.calibration.source_id)
