from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from .domain import GLOVE_JOINTS, RetargetedPose
from .landmarks import Ve33LandmarkSource


_FINGER_NAMES = ("index", "middle", "ring", "pinky")


def wuji_hand2_joint_names(hand: str) -> tuple[str, ...]:
    prefix = "r_" if hand == "right" else "l_" if hand == "left" else None
    if prefix is None:
        raise ValueError("hand必须为left或right")
    names = [
        f"{prefix}thumb_cmc_flex",
        f"{prefix}thumb_cmc_abd",
        f"{prefix}thumb_mcp",
        f"{prefix}thumb_ip",
    ]
    for finger in _FINGER_NAMES:
        joint_prefix = f"{finger}_finger_" if finger != "pinky" else "pinky_"
        names.extend(
            (
                f"{prefix}{joint_prefix}mcp_flex",
                f"{prefix}{joint_prefix}mcp_abd",
                f"{prefix}{joint_prefix}pip",
                f"{prefix}{joint_prefix}dip",
            )
        )
    return tuple(names)


class WujiHand2Retargeter:
    """官方 wuji-retargeting 的 Hand2 适配层，不维护手写关节映射。

    vE3.3 点源是模型世界坐标。流水线优先使用资产声明的固定
    ``model_to_canonical_row``；只有通用外部输入未提供固定模型坐标时，
    才回退到官方 MediaPipe 掌坐标预处理，避免让屈伸姿态重新定义 CAD wrist。
    """

    def __init__(self, core, *, hand: str):
        self.hand = hand
        self._core = core
        self.model_joint_names = tuple(core.optimizer.robot.dof_joint_names)
        self.output_joint_names = wuji_hand2_joint_names(hand)
        missing = [name for name in self.output_joint_names if name not in self.model_joint_names]
        if missing:
            raise ValueError(f"Hand2 URDF缺少预期关节: {missing}")
        self._output_indices = np.asarray(
            [self.model_joint_names.index(name) for name in self.output_joint_names], dtype=np.int64
        )
        raw_limits = getattr(core.optimizer.robot, "joint_limits", None)
        if raw_limits is None:
            self._effective_joint_limits = np.column_stack(
                (np.full(len(self.model_joint_names), -np.inf),
                 np.full(len(self.model_joint_names), np.inf))
            )
        else:
            self._effective_joint_limits = np.asarray(raw_limits, dtype=np.float64).copy()
        self._fixed_joint_values = self._configure_fixed_joints()

    def _configure_fixed_joints(self) -> dict[str, float]:
        """把配置声明的固定关节锁进官方优化器边界，而不是事后清零。"""
        config = getattr(self._core.optimizer, "config", {})
        retarget_config = config.get("retarget", {}) if isinstance(config, dict) else {}
        fixed = retarget_config.get("fixed_joints", {})
        if fixed is None:
            fixed = {}
        if not isinstance(fixed, dict):
            raise ValueError("retarget.fixed_joints必须是关节名到弧度的映射")
        if not fixed:
            return {}
        optimizer = self._core.optimizer
        if not hasattr(optimizer, "opt"):
            raise ValueError("配置了fixed_joints，但官方优化器没有可修改的边界")
        lower = self._effective_joint_limits[:, 0].copy()
        upper = self._effective_joint_limits[:, 1].copy()
        result: dict[str, float] = {}
        for name, value in fixed.items():
            if name not in self.model_joint_names:
                raise ValueError(f"fixed_joints包含未知关节: {name}")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"fixed_joints.{name}必须是有限弧度")
            index = self.model_joint_names.index(name)
            if not lower[index] <= value <= upper[index]:
                raise ValueError(f"fixed_joints.{name}超出URDF关节限位")
            lower[index] = value
            upper[index] = value
            self._effective_joint_limits[index] = (value, value)
            result[name] = value
        optimizer.opt.set_lower_bounds(lower.tolist())
        optimizer.opt.set_upper_bounds(upper.tolist())
        return result

    @classmethod
    def from_yaml(cls, path: str | Path, *, hand: str) -> "WujiHand2Retargeter":
        from wuji_retargeting import Retargeter

        config_path = Path(path).resolve()
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise ValueError(f"重定向配置不是对象: {config_path}")
        config["__yaml_dir"] = str(config_path.parent)
        core = Retargeter.from_config(config, hand_side=hand)
        return cls(core, hand=hand)

    def _canonical_points(
        self,
        points_m: np.ndarray,
        model_to_canonical_row: np.ndarray | None = None,
        canonical_to_optimizer_row: np.ndarray | None = None,
    ) -> np.ndarray:
        points = np.asarray(points_m, dtype=np.float64)
        if points.shape != (21, 3) or not np.all(np.isfinite(points)):
            raise ValueError("WujiRetarget输入必须是有限的21x3数组")
        if model_to_canonical_row is None:
            from wuji_retargeting.mediapipe import apply_mediapipe_transformations

            canonical = np.asarray(
                apply_mediapipe_transformations(points, self.hand),
                dtype=np.float64,
            )
        else:
            model_rotation = np.asarray(model_to_canonical_row, dtype=np.float64)
            if model_rotation.shape != (3, 3) or not np.allclose(
                model_rotation.T @ model_rotation, np.eye(3), atol=1e-6
            ):
                raise ValueError("model_to_canonical_row必须是3x3正交矩阵")
            canonical = (points - points[0]) @ model_rotation
        if canonical_to_optimizer_row is not None:
            optimizer_rotation = np.asarray(canonical_to_optimizer_row, dtype=np.float64)
            if optimizer_rotation.shape != (3, 3) or not np.allclose(
                optimizer_rotation.T @ optimizer_rotation, np.eye(3), atol=1e-6
            ):
                raise ValueError("canonical_to_optimizer_row必须是3x3正交矩阵")
            canonical = canonical @ optimizer_rotation
        if canonical.shape != (21, 3) or not np.all(np.isfinite(canonical)):
            raise ValueError("WujiRetarget掌坐标预处理输出非法")
        return canonical

    def step(
        self,
        points_m: np.ndarray,
        *,
        apply_filter: bool = False,
        model_to_canonical_row: np.ndarray | None = None,
        canonical_to_optimizer_row: np.ndarray | None = None,
    ) -> np.ndarray:
        canonical = self._canonical_points(
            points_m,
            model_to_canonical_row,
            canonical_to_optimizer_row,
        )


        raw = np.asarray(self._core.optimizer.solve(canonical), dtype=np.float64)
        self._check_optimization()
        if apply_filter:
            raw = np.asarray(self._core.lp_filter.next(raw), dtype=np.float64)
        if raw.shape != (len(self.model_joint_names),) or not np.all(np.isfinite(raw)):
            raise ValueError("WujiRetarget输出维度或数值非法")
        return raw[self._output_indices]

    @property
    def joint_limits(self) -> np.ndarray:
        reordered = self._effective_joint_limits[self._output_indices].copy()
        reordered.setflags(write=False)
        return reordered

    @property
    def fixed_joint_values(self) -> dict[str, float]:
        return dict(self._fixed_joint_values)
    def _check_optimization(self) -> None:
        """官方求解器会吞掉RuntimeError；不得把回退初值当成成功目标。"""
        optimization = getattr(self._core.optimizer, "opt", None)
        if optimization is not None:
            status = optimization.last_optimize_result()
            # 5/6是合法预算停止，仍有有限候选；负码/0才是失败或尚未执行。
            if status <= 0:
                raise ValueError(f"WujiRetarget优化失败，禁止使用回退目标：status={status}")


    def diagnose(
        self,
        points_m: np.ndarray,
        model_to_canonical_row: np.ndarray | None = None,
        canonical_to_optimizer_row: np.ndarray | None = None,
    ) -> dict[str, object]:
        """返回优化器真实 FK 与21点任务点之间的误差。"""
        canonical = self._canonical_points(
            points_m,
            model_to_canonical_row,
            canonical_to_optimizer_row,
        )
        optimizer = self._core.optimizer
        raw_q = np.asarray(optimizer.solve(canonical), dtype=np.float64)
        self._check_optimization()
        optimizer.robot.compute_forward_kinematics(raw_q)
        positions_m = np.asarray(
            [
                optimizer.robot.get_link_pose(index)[:3, 3]
                for index in optimizer.computed_link_indices
            ],
            dtype=np.float64,
        )
        positions_cm = positions_m * 100.0
        transformed = canonical
        origin = positions_cm[optimizer.origin_indices]
        task = positions_cm[optimizer.task_indices]
        link3 = positions_cm[optimizer.link3_indices]
        link4 = positions_cm[optimizer.link4_indices]
        wrist = positions_cm[optimizer.origin_indices[0]]
        robot_tip = task - origin
        target_tip = optimizer._compute_tip_vectors(transformed, optimizer.scaling)
        robot_full = np.vstack((link3 - wrist, link4 - wrist, task - wrist))
        target_full = optimizer._compute_full_hand_vectors(transformed, optimizer.segment_scaling)
        tip_error = np.linalg.norm(robot_tip - target_tip, axis=1)
        full_error = np.linalg.norm(robot_full - target_full, axis=1)
        q = raw_q[self._output_indices]
        limits = self.joint_limits
        return {
            "cost": float(optimizer.compute_cost(raw_q, canonical)),
            "tip_error_cm": tip_error.tolist(),
            "tip_error_max_cm": float(np.max(tip_error)),
            "tip_error_mean_cm": float(np.mean(tip_error)),
            "full_hand_error_cm": full_error.tolist(),
            "full_hand_error_max_cm": float(np.max(full_error)),
            "full_hand_error_mean_cm": float(np.mean(full_error)),
            "joint_limit_margin_rad": float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q))),
            "mcp_abd_rad": [
                float(q[self.output_joint_names.index(name)])
                for name in self.output_joint_names
                if "_mcp_abd" in name
            ],
        }

    def reset(self) -> None:
        self._core.reset()


class GloveWujiRetargetPipeline:
    """vE3.3闭链任务点 -> 官方 WujiRetarget -> Hand2 20维目标。"""

    def __init__(
        self,
        landmarks: Ve33LandmarkSource,
        retargeter: WujiHand2Retargeter,
        *,
        retargeter_factory: Callable[[], WujiHand2Retargeter] | None = None,
    ):
        self.landmarks = landmarks
        self.retargeter = retargeter
        self._retargeter_factory = retargeter_factory
        self._jacobian_retargeter: WujiHand2Retargeter | None = None
        self._jacobian_seed = None

    @classmethod
    def from_yaml(
        cls,
        hand: str,
        *,
        asset_root: str | Path,
        config_path: str | Path,
        estimate_jacobian: bool = False,
    ) -> "GloveWujiRetargetPipeline":
        asset_root = Path(asset_root)
        config_path = Path(config_path)
        landmarks = Ve33LandmarkSource(hand, asset_root=asset_root)
        retargeter = WujiHand2Retargeter.from_yaml(config_path, hand=hand)
        factory = None
        if estimate_jacobian:
            factory = lambda: WujiHand2Retargeter.from_yaml(config_path, hand=hand)
        return cls(landmarks, retargeter, retargeter_factory=factory)

    def step(
        self,
        joints: dict[str, float],
        *,
        estimate_jacobian: bool = False,
    ) -> RetargetedPose:
        frame = self.landmarks.step(joints)
        positions = self.retargeter.step(
            frame.points_m,
            apply_filter=False,
            model_to_canonical_row=self.landmarks.model_to_canonical_row,
            canonical_to_optimizer_row=self.landmarks.canonical_to_optimizer_row,
        )
        jacobian = self.local_jacobian(joints) if estimate_jacobian else None
        return RetargetedPose(
            positions,
            jacobian,
            diagnostics={
                "retarget_backend": "wuji_retargeting",
                "lateral_policy": (
                    "fixed_zero"
                    if all(
                        name in self.retargeter.fixed_joint_values
                        for name in self.retargeter.output_joint_names
                        if "_mcp_abd" in name and "thumb" not in name
                    )
                    else "inferred_from_21point_geometry"
                ),
                "jacobian": "central_finite_difference" if jacobian is not None else "not_requested",
            },
        )

    def local_jacobian(self, joints: dict[str, float], *, step_rad: float = 1e-4) -> np.ndarray:
        """低频估计完整重定向链路 J=∂q_wuji/∂q_glove；不伪造侧摆测量量。"""
        if self._retargeter_factory is None:
            raise RuntimeError("估计重定向雅可比需要从工厂创建独立重定向求解器")
        if not np.isfinite(step_rad) or step_rad <= 0:
            raise ValueError("step_rad必须为正有限数")
        # 每个扰动重建URDF/优化器会使矩阵交付时已经过期。独占一个求解器，
        # 每次正/负扰动前重置历史，保留原先从全新状态求导的数值语义。
        if self._jacobian_retargeter is None:
            self._jacobian_retargeter = self._retargeter_factory()
        solver = self._jacobian_retargeter
        # 正常遥操的1e-4代价容差不足以稳定求导；只收紧独占求导器，不改主控制器。
        solver._core.optimizer.opt.set_ftol_abs(1e-11)
        solver._core.optimizer.opt.set_maxeval(200)
        base = np.asarray([float(joints[name]) for name in GLOVE_JOINTS], dtype=np.float64)
        base_frame, seed = self.landmarks.evaluate_with_seed(joints, seed=self._jacobian_seed)
        solver.reset()
        solver._core.optimizer.initial_qpos_hint = None
        solver.step(
            base_frame.points_m, apply_filter=False,
            model_to_canonical_row=self.landmarks.model_to_canonical_row,
            canonical_to_optimizer_row=self.landmarks.canonical_to_optimizer_row,
        )
        # 正负扰动共享中心最优解作初值，但各自仍清空历史正则，目标函数不变。
        solver._core.optimizer.initial_qpos_hint = solver._core.optimizer.last_qpos.copy()
        jacobian = np.empty((20, 12), dtype=np.float64)
        for index, name in enumerate(GLOVE_JOINTS):
            plus = base.copy()
            minus = base.copy()
            plus[index] += step_rad
            minus[index] -= step_rad
            plus_joints = dict(zip(GLOVE_JOINTS, plus))
            minus_joints = dict(zip(GLOVE_JOINTS, minus))
            plus_frame, _ = self.landmarks.evaluate_with_seed(plus_joints, seed=seed)
            minus_frame, _ = self.landmarks.evaluate_with_seed(minus_joints, seed=seed)
            solver.reset()
            plus_q = solver.step(
                plus_frame.points_m,
                apply_filter=False,
                model_to_canonical_row=self.landmarks.model_to_canonical_row,
                canonical_to_optimizer_row=self.landmarks.canonical_to_optimizer_row,
            )
            solver.reset()
            minus_q = solver.step(
                minus_frame.points_m,
                apply_filter=False,
                model_to_canonical_row=self.landmarks.model_to_canonical_row,
                canonical_to_optimizer_row=self.landmarks.canonical_to_optimizer_row,
            )
            jacobian[:, index] = (plus_q - minus_q) / (2.0 * step_rad)
        self._jacobian_seed = seed
        if not np.all(np.isfinite(jacobian)):
            raise ValueError("重定向雅可比含非有限值")
        return jacobian


def default_retarget_config(project_root: str | Path, hand: str) -> Path:
    if hand not in {"left", "right"}:
        raise ValueError("hand必须为left或right")
    return Path(project_root) / "config" / "retargeting" / "official" / f"{hand}.yaml"
