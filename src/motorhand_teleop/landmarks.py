from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .domain import GLOVE_JOINTS, GlovePose
from .ve33_adapter import Ve33GloveModel
from .ve33_kinematics import Ve33Pose

_CANONICAL_TO_OPTIMIZER_ROW = np.asarray(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
_CANONICAL_TO_OPTIMIZER_ROW.setflags(write=False)


@dataclass(frozen=True)
class LandmarkFrame:
    """按 MediaPipe 21 点顺序输出的 vE3.3 模型任务点。"""

    points_m: np.ndarray
    jacobian_points_glove: np.ndarray
    glove_pose: GlovePose
    motor_positions: Mapping[str, float]

    def __post_init__(self) -> None:
        points = np.asarray(self.points_m, dtype=np.float64)
        jacobian = np.asarray(self.jacobian_points_glove, dtype=np.float64)
        if points.shape != (21, 3) or not np.all(np.isfinite(points)):
            raise ValueError("points_m必须是有限的21x3数组")
        if jacobian.shape != (21, 3, 12) or not np.all(np.isfinite(jacobian)):
            raise ValueError("jacobian_points_glove必须是有限的21x3x12数组")
        points.setflags(write=False)
        jacobian.setflags(write=False)
        object.__setattr__(self, "points_m", points)
        object.__setattr__(self, "jacobian_points_glove", jacobian)
        object.__setattr__(self, "motor_positions", dict(self.motor_positions))


class Ve33LandmarkSource:
    """将 vE3.3 闭链运动学转换成 WujiRetarget 所需的21个点。"""

    def __init__(self, hand: str, *, asset_root: str | Path):
        self.hand = hand
        self.asset_root = Path(asset_root)
        self.model = Ve33GloveModel(hand, asset_root=self.asset_root)
        side = "L" if hand == "left" else "R" if hand == "right" else None
        if side is None:
            raise ValueError("hand必须为left或right")
        target_path = self.asset_root / f"hand_motor_ve33_{side}" / "config" / "retarget_targets.json"
        with target_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
        sites = config.get("sites")
        if not isinstance(sites, list) or len(sites) != 21:
            raise ValueError("retarget_targets.json必须包含21个sites")
        sites = sorted(sites, key=lambda item: item["index"])
        if [item.get("index") for item in sites] != list(range(21)):
            raise ValueError("retarget_targets.json的site index必须连续覆盖0到20")
        self._sites = tuple(
            (str(item["link"]), np.asarray(item["local_xyz_m"], dtype=np.float64))
            for item in sites
        )
        model_to_canonical = np.asarray(config.get("canonical_to_model"), dtype=np.float64)
        if model_to_canonical.shape != (3, 3):
            raise ValueError("retarget_targets.json必须包含3x3 canonical_to_model")
        if not np.allclose(model_to_canonical.T @ model_to_canonical, np.eye(3), atol=1e-6):
            raise ValueError("canonical_to_model必须是正交旋转矩阵")
        if not np.isclose(np.linalg.det(model_to_canonical), 1.0, atol=1e-6):
            raise ValueError("canonical_to_model必须是proper rotation")
        model_to_canonical.setflags(write=False)
        self._model_to_canonical_row = model_to_canonical

    @property
    def model_to_canonical_row(self) -> np.ndarray:
        """返回 vE3.3 base_link 行向量到 canonical 坐标的固定旋转。"""
        return self._model_to_canonical_row

    @property
    def canonical_to_optimizer_row(self) -> np.ndarray:
        """返回 canonical 到当前 Hand2 优化器模型的固定行旋转。"""
        return _CANONICAL_TO_OPTIMIZER_ROW

    def _frame(self, pose, motors: Mapping[str, float]) -> LandmarkFrame:
        points, jacobian = self.model.kinematics.point_kinematics(pose, self._sites)
        # 与 dex_glove 的 JointRetargetPipeline 保持一致：thumb_cmc 使用腕点与
        # thumb_mcp 的几何中点，避免基座工程锚点直接污染重定向的手掌姿态。
        points = np.array(points, copy=True)
        jacobian = np.array(jacobian, copy=True)
        points[1] = (points[0] + points[2]) * 0.5
        jacobian[1] = (jacobian[0] + jacobian[2]) * 0.5
        public_pose = self.model.to_glove_pose(pose)
        return LandmarkFrame(points, jacobian, public_pose, motors)

    def step(self, joints: Mapping[str, float]) -> LandmarkFrame:
        values = {name: float(joints[name]) for name in GLOVE_JOINTS}
        pose, motors = self.model.solve(values)
        return self._frame(self.model.kinematics.current, motors)

    def evaluate_tree(self, positions: Mapping[str, float]) -> LandmarkFrame:
        """重验上游同一完整CAD解后构建任务点，不再做指节逆解或电机正解。"""
        pose = self.model.kinematics.evaluate_tree(positions)
        return self._frame(pose, pose.motor_positions)

    def evaluate_with_seed(
        self, joints: Mapping[str, float], *, seed: Ve33Pose | None = None
    ) -> tuple[LandmarkFrame, Ve33Pose]:
        """纯试算并返回下次试算的装配种子；不提交连续状态。"""
        values = {name: float(joints[name]) for name in GLOVE_JOINTS}
        if seed is None:
            raw_pose = self.model.evaluate_raw(values)
        else:
            internal = {f"{name}_joint": value for name, value in values.items()}
            kinematics = self.model.kinematics
            distance = max(
                abs(values[name] - seed.joint_positions[f"{name}_joint"])
                for index, name in enumerate(GLOVE_JOINTS)
                if kinematics._continuation_inputs[index]
            )
            # 复用已有隐函数预测的局部闭环校正；最终gap/bounds/rank仍由CAD模型全检。
            raw_pose = kinematics.evaluate(internal, seed=seed, local=distance <= 0.04 + 1e-12)
        # _frame已构建领域姿态，不再经_result重复求其从属角导数。
        frame = self._frame(raw_pose, raw_pose.motor_positions)
        return frame, raw_pose

    def evaluate(self, joints: Mapping[str, float]) -> LandmarkFrame:
        frame, _ = self.evaluate_with_seed(joints)
        return frame
