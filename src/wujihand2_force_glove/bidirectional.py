"""共享任务空间、双向求解和无扰控制权切换。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from threading import Lock
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.optimize import least_squares, minimize

from .domain import GLOVE_JOINTS
from .landmarks import Ve33LandmarkSource
from .retargeting import GloveWujiRetargetPipeline, WujiHand2Retargeter
from .ve33_kinematics import Ve33Pose

# 数值纠偏上限；最终电机目标仍必须严格位于原始软件限位内。
_MOTOR_BOUND_ROUNDOFF_RAD = 1e-7


class ControlMode(str, Enum):
    """当前拥有位置控制权的一侧。"""

    INFERENCE = "inference"
    GLOVE_TAKEOVER = "glove_takeover"


@dataclass(frozen=True)
class HandTaskState:
    """统一 canonical 掌坐标下的21点任务状态。"""

    points_m: np.ndarray
    timestamp_ns: int
    source: str

    def __post_init__(self) -> None:
        points = np.asarray(self.points_m, dtype=np.float64)
        if points.shape != (21, 3) or not np.all(np.isfinite(points)):
            raise ValueError("HandTaskState.points_m必须是有限的21x3数组")
        if not isinstance(self.timestamp_ns, int) or self.timestamp_ns < 0:
            raise ValueError("HandTaskState.timestamp_ns必须是非负整数")
        if not self.source:
            raise ValueError("HandTaskState.source不能为空")
        points = points.copy()
        points.setflags(write=False)
        object.__setattr__(self, "points_m", points)


@dataclass(frozen=True)
class DeviceState:
    """一侧设备的最新关节状态；effort和速度可以暂缺。"""

    timestamp_ns: int
    positions_rad: np.ndarray
    velocities_rad_s: np.ndarray | None = None
    effort: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp_ns, int) or self.timestamp_ns < 0:
            raise ValueError("DeviceState.timestamp_ns必须是非负整数")
        positions = np.asarray(self.positions_rad, dtype=np.float64)
        if positions.ndim != 1 or not np.all(np.isfinite(positions)):
            raise ValueError("DeviceState.positions_rad必须是一维有限数组")
        object.__setattr__(self, "positions_rad", _readonly(positions))
        for name in ("velocities_rad_s", "effort"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=np.float64)
                if array.shape != positions.shape or not np.all(np.isfinite(array)):
                    raise ValueError(f"DeviceState.{name}必须与positions_rad同形且有限")
                object.__setattr__(self, name, _readonly(array))


@dataclass(frozen=True)
class DeviceTaskSnapshot:
    device: DeviceState | None
    task: HandTaskState | None


@dataclass(frozen=True)
class BidirectionalSnapshot:
    glove: DeviceTaskSnapshot
    hand2: DeviceTaskSnapshot


class BidirectionalStateBus:
    """线程安全的双端最新状态总线，不负责发送执行器命令。"""

    def __init__(self) -> None:
        self._lock = Lock()
        self._glove = DeviceTaskSnapshot(None, None)
        self._hand2 = DeviceTaskSnapshot(None, None)

    def publish_glove(self, device: DeviceState, task: HandTaskState) -> None:
        with self._lock:
            _check_monotonic(self._glove.device, device, "手套")
            _check_monotonic(self._glove.task, task, "手套任务")
            self._glove = DeviceTaskSnapshot(device, task)

    def publish_hand2(self, device: DeviceState, task: HandTaskState) -> None:
        with self._lock:
            _check_monotonic(self._hand2.device, device, "Hand2")
            _check_monotonic(self._hand2.task, task, "Hand2任务")
            self._hand2 = DeviceTaskSnapshot(device, task)

    def snapshot(self) -> BidirectionalSnapshot:
        with self._lock:
            return BidirectionalSnapshot(self._glove, self._hand2)


@dataclass(frozen=True)
class _Handover:
    source: ControlMode
    target: ControlMode
    anchor: np.ndarray
    start_ns: int
    duration_ns: int


class HandoverController:
    """控制权切换器：候选目标可以连续变化，输出从切换锚点平滑过渡。"""

    def __init__(self, mode: ControlMode) -> None:
        self._mode = mode
        self._handover: _Handover | None = None

    @property
    def mode(self) -> ControlMode:
        return self._mode

    @property
    def target_mode(self) -> ControlMode:
        """当前候选目标应使用的模式；切换期间指向待接管方。"""
        if self._handover is not None:
            return self._handover.target
        return self._mode

    @property
    def in_handover(self) -> bool:
        return self._handover is not None

    def request(
        self,
        mode: ControlMode,
        current_command: np.ndarray,
        *,
        now_ns: int,
        duration_s: float,
    ) -> None:
        if not isinstance(mode, ControlMode):
            raise ValueError("mode必须是ControlMode")
        if now_ns < 0 or not np.isfinite(duration_s) or duration_s < 0:
            raise ValueError("切换时间参数非法")
        anchor = np.asarray(current_command, dtype=np.float64)
        if anchor.ndim != 1 or not np.all(np.isfinite(anchor)):
            raise ValueError("current_command必须是一维有限数组")
        if mode is self._mode and self._handover is None:
            return
        if duration_s == 0:
            self._mode = mode
            self._handover = None
            return
        self._handover = _Handover(
            source=self._mode,
            target=mode,
            anchor=_readonly(anchor.copy()),
            start_ns=now_ns,
            duration_ns=max(1, int(round(duration_s * 1e9))),
        )

    def defer(self, duration_ns: int) -> None:
        """暂停手动切换过渡，等待期间不消耗插值时长。"""
        if type(duration_ns) is not int or duration_ns < 0:
            raise ValueError("过渡暂停时间必须是非负整数ns")
        if self._handover is not None:
            self._handover = replace(
                self._handover, start_ns=self._handover.start_ns + duration_ns,
            )

    def output(self, candidate: np.ndarray, *, now_ns: int) -> np.ndarray:
        value = np.asarray(candidate, dtype=np.float64)
        if value.ndim != 1 or not np.all(np.isfinite(value)):
            raise ValueError("candidate必须是一维有限数组")
        handover = self._handover
        if handover is None:
            return value.copy()
        if value.shape != handover.anchor.shape:
            raise ValueError("切换候选与锚点维度不一致")
        elapsed = max(0, now_ns - handover.start_ns)
        alpha = min(1.0, elapsed / handover.duration_ns)
        # 五次 smoothstep 保持切换起止速度为零，降低真机冲击。
        smooth = alpha * alpha * alpha * (alpha * (alpha * 6.0 - 15.0) + 10.0)
        output = handover.anchor + smooth * (value - handover.anchor)
        if alpha >= 1.0:
            self._mode = handover.target
            self._handover = None
        return output


class GloveTaskModel:
    """把vE3.3独立12维姿态转成统一canonical任务点。"""

    def __init__(self, landmarks: Ve33LandmarkSource) -> None:
        self.landmarks = landmarks

    def state(self, joints: Mapping[str, float], *, timestamp_ns: int = 0) -> HandTaskState:
        state, _ = self.state_and_jacobian(joints, timestamp_ns=timestamp_ns)
        return state

    def state_from_tree(
        self, positions: Mapping[str, float], *, timestamp_ns: int = 0
    ) -> HandTaskState:
        """让已验证的68维CAD树贯穿周期；只重验闭环并做同一canonical真FK。"""
        frame = self.landmarks.evaluate_tree(positions)
        points = (frame.points_m - frame.points_m[0]) @ self.landmarks.model_to_canonical_row
        return HandTaskState(points, timestamp_ns=timestamp_ns, source="glove")

    def state_and_jacobian(
        self, joints: Mapping[str, float], *, timestamp_ns: int = 0
    ) -> tuple[HandTaskState, np.ndarray]:
        """一次CAD闭环试算，返回canonical任务点及其对GLOVE_JOINTS的导数。"""
        state, jacobian, _ = self._evaluate(joints, timestamp_ns=timestamp_ns)
        return state, jacobian

    def _evaluate(
        self,
        joints: Mapping[str, float],
        *,
        timestamp_ns: int,
        seed: Ve33Pose | None = None,
    ) -> tuple[HandTaskState, np.ndarray, Ve33Pose]:
        values = {name: float(joints[name]) for name in GLOVE_JOINTS}
        frame, pose = self.landmarks.evaluate_with_seed(values, seed=seed)
        rotation = self.landmarks.model_to_canonical_row
        points = (frame.points_m - frame.points_m[0]) @ rotation
        # 与点值完全相同的腕点平移和行向量旋转，最后一轴始终是12真实指节。
        jacobian = np.einsum(
            "paj,ab->pbj", frame.jacobian_points_glove - frame.jacobian_points_glove[0], rotation
        )
        jacobian.setflags(write=False)
        return HandTaskState(points, timestamp_ns=timestamp_ns, source="glove"), jacobian, pose


class Hand2TaskModel:
    """把Hand2语义20维姿态转成与手套相同的canonical任务点。"""

    def __init__(self, retargeter: WujiHand2Retargeter, canonical_to_optimizer_row: np.ndarray) -> None:
        self.retargeter = retargeter
        rotation = np.asarray(canonical_to_optimizer_row, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("canonical_to_optimizer_row必须是3x3正交矩阵")
        self._optimizer_to_canonical_row = rotation.T

    def state(self, positions_rad: np.ndarray, *, timestamp_ns: int = 0) -> HandTaskState:
        positions = np.asarray(positions_rad, dtype=np.float64)
        if positions.shape != (20,) or not np.all(np.isfinite(positions)):
            raise ValueError("Hand2姿态必须是有限的20维数组")
        optimizer = self.retargeter._core.optimizer
        raw = np.zeros(len(self.retargeter.model_joint_names), dtype=np.float64)
        raw[self.retargeter._output_indices] = positions
        # fixed_joint_values仅约束正向优化；实测20维FK不能把真实侧摆反馈抹成零。
        optimizer.robot.compute_forward_kinematics(raw)
        positions_by_link = np.asarray(
            [optimizer.robot.get_link_pose(index)[:3, 3] for index in optimizer.computed_link_indices],
            dtype=np.float64,
        )
        points = np.empty((21, 3), dtype=np.float64)
        points[0] = positions_by_link[optimizer.origin_indices[0]]
        for finger in range(5):
            points[1 + 4 * finger] = positions_by_link[optimizer.link1_indices[finger]]
            points[2 + 4 * finger] = positions_by_link[optimizer.link3_indices[finger]]
            points[3 + 4 * finger] = positions_by_link[optimizer.link4_indices[finger]]
            points[4 + 4 * finger] = positions_by_link[optimizer.task_indices[finger]]
        points -= points[0]
        points = points @ self._optimizer_to_canonical_row
        return HandTaskState(points, timestamp_ns=timestamp_ns, source="wuji_hand2")


@dataclass(frozen=True)
class InverseRetargetResult:
    glove_positions_rad: np.ndarray
    task_error_max_mm: float
    task_error_rms_mm: float
    success: bool
    iterations: int
    tree_positions_rad: Mapping[str, float]
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.glove_positions_rad, dtype=np.float64)
        if values.shape != (12,) or not np.all(np.isfinite(values)):
            raise ValueError("glove_positions_rad必须是有限的12维数组")
        object.__setattr__(self, "glove_positions_rad", _readonly(values))
        tree = dict(self.tree_positions_rad)
        if (self.success or tree) and len(tree) != 68:
            raise ValueError("成功逆解必须携带本次完整68维CAD树")
        if any(not np.isfinite(value) for value in tree.values()):
            raise ValueError("tree_positions_rad必须是有限的CAD树角")
        object.__setattr__(self, "tree_positions_rad", MappingProxyType(tree))


class GloveInverseSolver:
    """统一任务空间到vE3.3 12维独立姿态的受限逆解。"""

    def __init__(
        self,
        glove_model: GloveTaskModel,
        *,
        task_weight: float = 1.0,
        regularization_weight: float = 1e-6,
        joint_bound_rad: float = 1.5,
        max_nfev: int = 80,
    ) -> None:
        if not np.isfinite(task_weight) or task_weight <= 0:
            raise ValueError("task_weight必须为正有限数")
        if not np.isfinite(regularization_weight) or regularization_weight < 0:
            raise ValueError("regularization_weight必须为非负有限数")
        if not np.isfinite(joint_bound_rad) or joint_bound_rad <= 0:
            raise ValueError("joint_bound_rad必须为正有限数")
        if max_nfev <= 0:
            raise ValueError("max_nfev必须为正整数")
        self.glove_model = glove_model
        self.task_weight = float(task_weight)
        self.regularization_weight = float(regularization_weight)
        # 点残差以米计：默认sqrt(lambda)=1mm/rad，避免厘米级任务被角度锚淹没。
        self.joint_bound_rad = float(joint_bound_rad)
        self.max_nfev = int(max_nfev)
        self._motor_names = glove_model.landmarks.model.motor_names
        self._motor_lower = np.full(12, -np.inf)
        self._motor_upper = np.full(12, np.inf)
        self._clear_warm_start()

    def _clear_warm_start(self) -> None:
        self._last_successful_positions: np.ndarray | None = None
        self._last_successful_seed: Ve33Pose | None = None
        self._last_successful_target: np.ndarray | None = None
        self._last_successful_anchor: np.ndarray | None = None
        self._last_successful_jacobian: np.ndarray | None = None
        self._warm_step_scale = 0.04

    def set_motor_limits(self, limits: Mapping[str, tuple[float, float]]) -> None:
        """绑定完整模型电机名的真实范围；保留非线性CAD传动，不伪造q轴限位。"""
        if not isinstance(limits, Mapping):
            raise ValueError("电机限位必须是按完整模型电机名称索引的映射")
        unknown = set(limits).difference(self._motor_names)
        if unknown:
            raise ValueError(f"未知模型电机：{', '.join(map(str, unknown))}")
        lower = np.full(12, -np.inf)
        upper = np.full(12, np.inf)
        for name, bounds in limits.items():
            if np.iscomplexobj(bounds):
                raise ValueError(f"{name}限位必须是有限实数上下界")
            try:
                pair = np.asarray(bounds, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{name}限位必须是有限实数上下界") from error
            if pair.shape != (2,) or not np.all(np.isfinite(pair)) or pair[0] >= pair[1]:
                raise ValueError(f"{name}限位必须满足有限lower < upper")
            index = self._motor_names.index(name)
            lower[index], upper[index] = pair
        if np.array_equal(lower, self._motor_lower) and np.array_equal(upper, self._motor_upper):
            return
        self._motor_lower = lower
        self._motor_upper = upper
        self._clear_warm_start()

    def solve(
        self,
        target: HandTaskState,
        *,
        initial: np.ndarray | None = None,
        timestamp_ns: int | None = None,
    ) -> InverseRetargetResult:
        if not isinstance(target, HandTaskState):
            raise ValueError("target必须是HandTaskState")
        if initial is None:
            start = (
                np.zeros(12, dtype=np.float64)
                if self._last_successful_positions is None
                else self._last_successful_positions.copy()
            )
        else:
            start = np.asarray(initial, dtype=np.float64)
        if start.shape != (12,) or not np.all(np.isfinite(start)):
            raise ValueError("initial必须是有限的12维数组")
        motor_indices = np.flatnonzero(np.isfinite(self._motor_lower))
        constrained = bool(motor_indices.size)
        # 实测角是正规项锚点，不是强制的数值起点；执行器滞后不能让每帧从零重求。
        numerical_start = (
            np.zeros(12, dtype=np.float64) if constrained else start
        ) if self._last_successful_positions is None else self._last_successful_positions
        guess = np.clip(numerical_start, -self.joint_bound_rad, self.joint_bound_rad)
        target_points = target.points_m
        weight = np.sqrt(self.regularization_weight)
        eval_timestamp = target.timestamp_ns if timestamp_ns is None else int(timestamp_ns)
        trust_scale = 1.0
        if self._last_successful_jacobian is not None:
            change = ((target_points - self._last_successful_target) * self.task_weight).reshape(-1)
            if weight != 0.0:
                change = np.concatenate((change, weight * (start - self._last_successful_anchor)))
            predicted_change = np.linalg.lstsq(
                self._last_successful_jacobian, change, rcond=1e-9
            )[0]
            # 继承成功解收缩后的数值尺度；真实任务/正规锚变化仍可主动扩大邻域。
            trust_scale = min(0.04, max(self._warm_step_scale, float(np.max(np.abs(predicted_change)))))
        lower = np.full(12, -self.joint_bound_rad)
        upper = np.full(12, self.joint_bound_rad)
        if self._last_successful_seed is not None:
            active_lower, active_upper = self.glove_model.landmarks.model.kinematics.active_joint_bounds(
                self._last_successful_seed
            )
            lower = np.maximum(lower, active_lower)
            upper = np.minimum(upper, active_upper)
            if np.any(lower >= upper):
                raise ValueError("CAD活跃边界与指节盒约束不相容，保留原姿态")
            if np.any(np.isfinite(active_lower)) or np.any(np.isfinite(active_upper)):
                # 真实活跃轴由有界求解器处理，不能让不可达方向把所有自由轴一并缩步。
                trust_scale = 0.04
        guess = np.clip(guess, lower, upper)

        # 候选种子仅属于本次求解；成功后才能替换上一成功帧的warm start。
        seed = self._last_successful_seed
        last_values = None
        last_state = None
        last_residual = None
        last_jacobian = None
        last_error = None
        last_motor_positions = None
        last_motor_jacobian = None
        regularization_jacobian = weight * np.eye(12)

        def evaluate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            nonlocal seed, last_values, last_state, last_residual, last_jacobian, last_error
            nonlocal last_motor_positions, last_motor_jacobian
            if last_values is not None and np.array_equal(values, last_values):
                return last_residual, last_jacobian
            failure = None
            try:
                state, point_jacobian, candidate_seed = self.glove_model._evaluate(
                    dict(zip(GLOVE_JOINTS, values)), timestamp_ns=eval_timestamp, seed=seed
                )
                task_residual = ((state.points_m - target_points) * self.task_weight).reshape(-1)
                task_jacobian = (point_jacobian * self.task_weight).reshape(63, 12)
                seed = candidate_seed
                if constrained:
                    kinematics = self.glove_model.landmarks.model.kinematics
                    last_motor_positions = np.asarray([
                        candidate_seed.motor_positions[name] for name in self._motor_names
                    ])
                    last_motor_jacobian = kinematics._tree_derivatives(
                        candidate_seed.all_positions
                    )[kinematics._motor_indices]
            except (ValueError, FloatingPointError) as error:
                state = None
                task_residual = np.full(63, 10.0, dtype=np.float64)
                task_jacobian = np.zeros((63, 12), dtype=np.float64)
                failure = error
                last_motor_positions = None
                last_motor_jacobian = None
            last_values = values.copy()
            last_state = state
            last_error = failure
            if weight == 0.0:
                last_residual = task_residual
                last_jacobian = task_jacobian
            else:
                last_residual = np.concatenate((task_residual, weight * (values - start)))
                last_jacobian = np.concatenate((task_jacobian, regularization_jacobian), axis=0)
            return last_residual, last_jacobian

        # 优化增量而非绝对角；actual只作正规锚，不把越硬件限位反馈冒充可行输出。
        if constrained:
            def objective(delta: np.ndarray) -> float:
                residual, _ = evaluate(guess + delta)
                # 同时缩放任务和正规项，仅改善米制目标的数值尺度，不改变最优解。
                return float(500.0 * (residual @ residual))

            def objective_jacobian(delta: np.ndarray) -> np.ndarray:
                residual, jacobian = evaluate(guess + delta)
                return 1000.0 * (jacobian.T @ residual)

            def motor_constraint(delta: np.ndarray) -> np.ndarray:
                evaluate(guess + delta)
                if last_motor_positions is None:
                    return np.full(2 * motor_indices.size, -10.0)
                motors = last_motor_positions[motor_indices]
                return np.concatenate((
                    motors - self._motor_lower[motor_indices],
                    self._motor_upper[motor_indices] - motors,
                ))

            def motor_constraint_jacobian(delta: np.ndarray) -> np.ndarray:
                evaluate(guess + delta)
                if last_motor_jacobian is None:
                    return np.zeros((2 * motor_indices.size, 12))
                jacobian = last_motor_jacobian[motor_indices]
                return np.concatenate((jacobian, -jacobian), axis=0)

            fit = minimize(
                objective, np.zeros(12, dtype=np.float64), jac=objective_jacobian,
                bounds=list(zip(lower - guess, upper - guess)), method="SLSQP",
                constraints={"type": "ineq", "fun": motor_constraint, "jac": motor_constraint_jacobian},
                options={"maxiter": self.max_nfev, "ftol": 1e-8},
            )
        else:
            fit = least_squares(
                lambda delta: evaluate(guess + delta)[0],
                np.zeros(12, dtype=np.float64),
                jac=lambda delta: evaluate(guess + delta)[1],
                bounds=(lower - guess, upper - guess),
                method="dogbox",
                x_scale=trust_scale,
                max_nfev=self.max_nfev,
                xtol=1e-9,
                ftol=1e-5,
                gtol=1e-7,
            )
        positions = guess + fit.x
        evaluate(positions)
        if last_state is None:
            raise ValueError("逆解最终候选不满足CAD闭环约束，保留原姿态") from last_error
        solved = last_state
        success = bool(fit.success)
        failure_reason = None if success else (
            f"优化器未收敛：status={getattr(fit, 'status', 'unknown')}, "
            f"message={getattr(fit, 'message', 'unknown')}, nfev={fit.nfev}"
        )
        if constrained and success:
            bounded = np.clip(last_motor_positions, self._motor_lower, self._motor_upper)
            excess = np.abs(last_motor_positions - bounded)
            worst = int(np.argmax(excess))
            if excess[worst] > _MOTOR_BOUND_ROUNDOFF_RAD:
                success = False
                failure_reason = f"模型电机{self._motor_names[worst]}超出软限位{excess[worst]:.9g} rad"
            elif np.any(excess > 0):
                try:
                    # 仅修复数值级越界，随后以精确限位内的m12重算同一CAD闭链。
                    landmarks = self.glove_model.landmarks
                    projected = landmarks.model.kinematics.evaluate_motors(
                        dict(zip(self._motor_names, bounded)), seed=seed,
                    )
                    frame = landmarks.evaluate_tree(projected.all_positions)
                    actual_m = np.asarray([projected.motor_positions[name] for name in self._motor_names])
                    actual_q = frame.glove_pose.vector()
                    if (np.any(actual_m < self._motor_lower) or np.any(actual_m > self._motor_upper)
                            or np.any(np.abs(actual_q) > self.joint_bound_rad)):
                        raise ValueError("投影后的CAD姿态仍超出电机或指节边界")
                    rotation = landmarks.model_to_canonical_row
                    points = (frame.points_m - frame.points_m[0]) @ rotation
                    point_jacobian = np.einsum(
                        "paj,ab->pbj",
                        frame.jacobian_points_glove - frame.jacobian_points_glove[0], rotation,
                    )
                    task_jacobian = (point_jacobian * self.task_weight).reshape(63, 12)
                    last_jacobian = (
                        task_jacobian if weight == 0.0
                        else np.concatenate((task_jacobian, regularization_jacobian), axis=0)
                    )
                    solved = HandTaskState(points, timestamp_ns=eval_timestamp, source="glove")
                    positions, seed = actual_q, projected
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    success = False
                    failure_reason = f"数值限位投影后的CAD闭链无效：{error}"
        errors_mm = np.linalg.norm(solved.points_m - target_points, axis=1) * 1000.0
        if success:
            self._last_successful_positions = positions.copy()
            self._last_successful_seed = seed
            self._last_successful_target = target_points
            self._last_successful_anchor = start.copy()
            self._last_successful_jacobian = last_jacobian
            self._warm_step_scale = min(0.04, max(1e-8, float(np.max(np.abs(fit.x)))))
        return InverseRetargetResult(
            glove_positions_rad=positions,
            task_error_max_mm=float(np.max(errors_mm)),
            task_error_rms_mm=float(np.sqrt(np.mean(errors_mm**2))),
            success=success,
            iterations=int(fit.nfev),
            tree_positions_rad=seed.all_positions if success else {},
            failure_reason=failure_reason,
        )


class BidirectionalRetargeter:
    """共享任务空间的正向/逆向 retarget 入口。"""

    def __init__(
        self,
        forward: GloveWujiRetargetPipeline,
        *,
        inverse: GloveInverseSolver,
        hand2_model: Hand2TaskModel,
    ) -> None:
        self.forward = forward
        self.inverse = inverse
        self.hand2_model = hand2_model

    def glove_to_hand2(
        self,
        joints: Mapping[str, float],
        *,
        timestamp_ns: int = 0,
        estimate_jacobian: bool = False,
    ):
        pose = self.forward.step(dict(joints), estimate_jacobian=estimate_jacobian)
        task = self.hand2_model.state(pose.positions_rad, timestamp_ns=timestamp_ns)
        return pose, task

    def hand2_to_glove(
        self,
        positions_rad: np.ndarray,
        *,
        initial: np.ndarray | None = None,
        timestamp_ns: int = 0,
    ) -> tuple[InverseRetargetResult, HandTaskState]:
        task = self.hand2_model.state(positions_rad, timestamp_ns=timestamp_ns)
        return self.inverse.solve(task, initial=initial), task


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _check_monotonic(previous, current, label: str) -> None:
    if previous is not None and current.timestamp_ns < previous.timestamp_ns:
        raise ValueError(f"{label}时间戳倒退")
