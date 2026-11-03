from __future__ import annotations

import numpy as np

from ..domain import GLOVE_JOINTS
from ..retargeting import GloveWujiRetargetPipeline, WujiHand2Retargeter


class ForceJacobianColumns:
    """进程独占的精确中心差分列求导器，不复用正向控制求解器。

    标准流水线每次只求一个中心和指定列的正负扰动；CAD 装配种子与
    独立官方优化器跨调用保留。自定义流水线仍调用其公开 local_jacobian，
    不绕过封装上的阻塞、故障或其他自定义行为。
    """

    def __init__(self, pipeline):
        self._pipeline = pipeline
        self._solver: WujiHand2Retargeter | None = None
        self._seed = None
        # 仅优化已知的原始实现；包括子类在内的其他对象遵守其公开接口。
        self._standard_pipeline = type(pipeline) is GloveWujiRetargetPipeline

    @staticmethod
    def _finite_output(value, *, shape: tuple[int, ...], name: str) -> np.ndarray:
        if np.iscomplexobj(value):
            raise ValueError(f"{name}必须为实数")
        result = np.asarray(value, dtype=np.float64)
        if result.shape != shape:
            raise ValueError(f"{name}维数必须为{shape}，实际为{result.shape}")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name}含非有限值")
        return result

    def compute(
        self, q: np.ndarray, columns: tuple[int, ...], *, step_rad: float
    ) -> np.ndarray:
        """返回形状为 (20, len(columns)) 的完整链路雅可比列，顺序不变。"""
        base = self._finite_output(q, shape=(12,), name="手套关节q")
        if not isinstance(columns, tuple):
            raise ValueError("columns必须为列索引元组")
        if any(
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            or not 0 <= index < 12
            for index in columns
        ):
            raise ValueError("columns中的列索引必须为0到11的整数")
        if len(set(columns)) != len(columns):
            raise ValueError("columns不能包含重复列索引")
        if np.iscomplexobj(step_rad):
            raise ValueError("step_rad必须为正有限标量")
        step = np.asarray(step_rad, dtype=np.float64)
        if step.shape != () or not np.isfinite(step) or step <= 0:
            raise ValueError("step_rad必须为正有限标量")
        step_rad = float(step)
        joints = dict(zip(GLOVE_JOINTS, base))
        if not self._standard_pipeline:
            full = self._finite_output(
                self._pipeline.local_jacobian(joints, step_rad=step_rad),
                shape=(20, 12),
                name="重定向雅可比",
            )
            return full[:, columns]
        if not columns:
            return np.empty((20, 0), dtype=np.float64)

        pipeline = self._pipeline
        if self._solver is None:
            if pipeline._retargeter_factory is None:
                raise RuntimeError("估计重定向雅可比需要从工厂创建独立重定向求解器")
            solver = pipeline._retargeter_factory()
            if solver is pipeline.retargeter:
                raise RuntimeError("力反馈列求导器不能复用正向控制求解器")
            self._solver = solver
        solver = self._solver
        optimizer = solver._core.optimizer
        optimizer.opt.set_ftol_abs(1e-11)
        optimizer.opt.set_maxeval(200)
        landmarks = pipeline.landmarks
        base_frame, seed = landmarks.evaluate_with_seed(joints, seed=self._seed)
        solver.reset()
        optimizer.initial_qpos_hint = None
        self._finite_output(
            solver.step(
                base_frame.points_m,
                apply_filter=False,
                model_to_canonical_row=landmarks.model_to_canonical_row,
                canonical_to_optimizer_row=landmarks.canonical_to_optimizer_row,
            ),
            shape=(20,),
            name="中心重定向关节",
        )
        # reset 清空 last_qpos 历史正则，但保留共同中心初值。
        optimizer.initial_qpos_hint = optimizer.last_qpos.copy()
        jacobian = np.empty((20, len(columns)), dtype=np.float64)
        for output_index, input_index in enumerate(columns):
            plus = base.copy()
            minus = base.copy()
            plus[input_index] += step_rad
            minus[input_index] -= step_rad
            plus_frame, _ = landmarks.evaluate_with_seed(
                dict(zip(GLOVE_JOINTS, plus)), seed=seed
            )
            minus_frame, _ = landmarks.evaluate_with_seed(
                dict(zip(GLOVE_JOINTS, minus)), seed=seed
            )
            solver.reset()
            plus_q = self._finite_output(
                solver.step(
                    plus_frame.points_m,
                    apply_filter=False,
                    model_to_canonical_row=landmarks.model_to_canonical_row,
                    canonical_to_optimizer_row=landmarks.canonical_to_optimizer_row,
                ),
                shape=(20,),
                name="正扰动重定向关节",
            )
            solver.reset()
            minus_q = self._finite_output(
                solver.step(
                    minus_frame.points_m,
                    apply_filter=False,
                    model_to_canonical_row=landmarks.model_to_canonical_row,
                    canonical_to_optimizer_row=landmarks.canonical_to_optimizer_row,
                ),
                shape=(20,),
                name="负扰动重定向关节",
            )
            jacobian[:, output_index] = (plus_q - minus_q) / (2.0 * step_rad)
        self._finite_output(
            jacobian, shape=(20, len(columns)), name="重定向雅可比列"
        )
        self._seed = seed
        return jacobian
