from __future__ import annotations

from typing import Callable

import numpy as np


class GloveTransmission:
    """把手套独立坐标力矩转换到物理电机轴；闭链求解器由调用方注入。"""

    def __init__(self, motor_position_from_joint: Callable[[np.ndarray], np.ndarray], *, step_rad: float = 1e-5):
        self._solve = motor_position_from_joint
        self._step = float(step_rad)
        if not np.isfinite(self._step) or self._step <= 0:
            raise ValueError("step_rad必须为正有限值")

    def motor_jacobian(self, joint_positions_rad: np.ndarray) -> np.ndarray:
        q = np.asarray(joint_positions_rad, dtype=np.float64)
        if q.shape != (12,) or not np.all(np.isfinite(q)):
            raise ValueError("joint_positions_rad必须是有限的12维数组")
        jacobian = np.empty((12, 12), dtype=np.float64)
        for index in range(12):
            plus = q.copy(); plus[index] += self._step
            minus = q.copy(); minus[index] -= self._step
            motor_plus = np.asarray(self._solve(plus), dtype=np.float64)
            motor_minus = np.asarray(self._solve(minus), dtype=np.float64)
            if motor_plus.shape != (12,) or motor_minus.shape != (12,):
                raise ValueError("闭链求解器必须返回12维电机角")
            jacobian[:, index] = (motor_plus - motor_minus) / (2.0 * self._step)
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("闭链传动雅可比含非有限值")
        return jacobian

    def joint_torque_to_motor_torque(self, joint_positions_rad: np.ndarray, joint_torque_nm: np.ndarray) -> np.ndarray:
        torque = np.asarray(joint_torque_nm, dtype=np.float64)
        jacobian = self.motor_jacobian(joint_positions_rad)
        return self.joint_torque_to_motor_torque_from_jacobian(jacobian, torque)

    @staticmethod
    def joint_torque_to_motor_torque_from_jacobian(
        jacobian: np.ndarray,
        joint_torque_nm: np.ndarray,
    ) -> np.ndarray:
        jacobian = np.asarray(jacobian, dtype=np.float64)
        torque = np.asarray(joint_torque_nm, dtype=np.float64)
        if jacobian.shape != (12, 12) or not np.all(np.isfinite(jacobian)):
            raise ValueError("jacobian必须是有限的12x12数组")
        if torque.shape != (12,) or not np.all(np.isfinite(torque)):
            raise ValueError("joint_torque_nm必须是有限的12维数组")
        return np.linalg.solve(jacobian.T, torque)

    @staticmethod
    def motor_torque_to_current(motor_torque_nm: np.ndarray, torque_constant_nm_a: np.ndarray) -> np.ndarray:
        torque = np.asarray(motor_torque_nm, dtype=np.float64)
        kt = np.asarray(torque_constant_nm_a, dtype=np.float64)
        if torque.shape != (12,) or kt.shape != (12,) or np.any(kt <= 0) or not np.all(np.isfinite(torque)) or not np.all(np.isfinite(kt)):
            raise ValueError("电机力矩和力矩常数必须是12维正有限数组")
        return torque / kt
