from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .domain import GLOVE_JOINTS, GlovePose
from .transmission import GloveTransmission
from .ve33_kinematics import Ve33Kinematics, Ve33Pose


class Ve33GloveModel:
    """把用户提供的 vE.3.3 闭链求解器接到本项目的 12 维领域坐标。"""

    def __init__(self, hand: str = "right", *, asset_root: str | Path | None = None):
        side = "L" if hand == "left" else "R" if hand == "right" else None
        if side is None:
            raise ValueError("hand必须为left或right")
        root = Path(asset_root) if asset_root is not None else Path(__file__).resolve().parents[2] / "assets/hand_motor_ve33"
        model_dir = root / f"hand_motor_ve33_{side}"
        self.kinematics = Ve33Kinematics(hand, model_dir=model_dir)
        self._reference_pose = self.kinematics.current
        self.joint_motor_names = {
            self._public_joint_name(item["name"]): item["motor"]
            for item in self.kinematics.joint_definitions
        }
        self.motor_names = self.kinematics.motor_names

    @staticmethod
    def _public_joint_name(name: str) -> str:
        return name.removesuffix("_joint")

    def to_glove_pose(self, pose: Ve33Pose, *, timestamp_ns: int | None = None) -> GlovePose:
        joints = {self._public_joint_name(name): value for name, value in pose.joint_positions.items()}
        dependent = {self._public_joint_name(name): value for name, value in pose.dependent_positions.items() if name.endswith("_joint")}
        derivative = self.kinematics._tree_derivatives(pose.all_positions)
        dependent_rows = []
        for definition in self.kinematics._dependent_definitions:
            angle_row = self.kinematics._angles[12 + len(dependent_rows)]
            dependent_rows.append(angle_row @ derivative)
        return GlovePose(
            joints=joints,
            dependent=dependent,
            dependent_jacobian=np.asarray(dependent_rows, dtype=np.float64),
            timestamp_ns=timestamp_ns,
        )

    def _result(self, pose: Ve33Pose) -> tuple[GlovePose, dict[str, float]]:
        public = self.to_glove_pose(pose)
        motors = dict(pose.motor_positions)
        return public, motors

    def evaluate_raw(self, joints: dict[str, float]) -> Ve33Pose:
        """评估并返回内部68维闭链姿态，不提交到连续求解状态。"""
        internal = {f"{name}_joint": value for name, value in joints.items()}
        return self.kinematics.evaluate(internal, seed=self._reference_pose)

    def evaluate(self, joints: dict[str, float]) -> tuple[GlovePose, dict[str, float]]:
        """评估目标角对应的闭链姿态，但不改变连续求解的参考状态。"""
        pose = self.evaluate_raw(joints)
        return self._result(pose)

    def solve(self, joints: dict[str, float]) -> tuple[GlovePose, dict[str, float]]:
        internal = {f"{name}_joint": value for name, value in joints.items()}
        pose = self.kinematics.solve(internal)
        self._reference_pose = pose
        return self._result(pose)

    def evaluate_motors(
        self, targets: Mapping[str, float], *, timestamp_ns: int | None = None,
        seed: Ve33Pose | None = None,
    ) -> tuple[GlovePose, dict[str, float]]:
        """完整模型电机角的闭链正解；不提交，不把电机角冒充真实指节。"""
        public, pose = self.evaluate_motor_state(targets, timestamp_ns=timestamp_ns, seed=seed)
        return public, dict(pose.motor_positions)

    def evaluate_motor_state(
        self, targets: Mapping[str, float], *, timestamp_ns: int | None = None,
        seed: Ve33Pose | None = None,
    ) -> tuple[GlovePose, Ve33Pose]:
        """一次正解同时返回领域姿态和完整CAD树快照。"""
        pose = self.kinematics.evaluate_motors(
            targets, seed=self._reference_pose if seed is None else seed
        )
        return self.to_glove_pose(pose, timestamp_ns=timestamp_ns), pose

    def motor_positions_from_joint_vector(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (12,) or not np.all(np.isfinite(values)):
            raise ValueError("手套独立坐标必须是有限的12维数组")
        targets = {f"{name}_joint": value for name, value in zip(GLOVE_JOINTS, values)}
        pose = self.kinematics.evaluate(targets, seed=self._reference_pose)
        return np.asarray([pose.motor_positions[name] for name in self.kinematics.motor_names], dtype=np.float64)

    def transmission(self) -> GloveTransmission:
        return GloveTransmission(self.motor_positions_from_joint_vector)

