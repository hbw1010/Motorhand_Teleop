"""只消费原始状态的反作用输出；不连接设备或取得输出权。

真实Kt模式估计含内部负载的执行器反作用；电流比例与接触阻抗模式只产生
明确的名义输出。电流和软件虚拟力矩均不能冒充独立测得的净接触力。
"""
from __future__ import annotations

import json
import multiprocessing
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from ..domain import GLOVE_JOINTS, RetargetedPose, WujiState
from ..feedback import CalibrationRequired, CalibratedFeedback, CurrentTorqueCalibration
from .force_linear import MaxTorqueLinearConfig, project_max_torque_linear
from .contact_feedback import ContactFeedbackConfig, ContactImpedance
from .force_geometry import ForceJacobianColumns
from ..transmission import GloveTransmission

if TYPE_CHECKING:
    from ..bidirectional import BidirectionalRetargeter
    from ..retargeting import GloveWujiRetargetPipeline
    from .motor_hand import EncosGloveState


def _vector(value, size: int, name: str, *, positive: bool = False, nonzero: bool = False):
    # bool 不能成为已实测的物理系数；先于numpy转换检查原始类型。
    raw = np.asarray(value, dtype=object)
    if raw.shape != (size,) or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.number)) or np.iscomplexobj(x) for x in raw):
        raise ValueError(f"{name}必须是{size}维有限实数数组")
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name}必须是{size}维有限实数数组") from error
    if not np.isfinite(result).all() or (positive and np.any(result <= 0)) or (nonzero and np.any(result == 0)):
        raise ValueError(f"{name}物理标定值非法")
    result.setflags(write=False)
    return result


def _positive(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name}必须为正有限实数")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name}必须为正有限实数")
    return result


@dataclass(frozen=True)
class ForceFeedbackCalibration:
    """Wuji取关节侧等效Nm/A（含内部减速），手套Kt对应其SDK原始角坐标轴。"""

    schema_version: int
    calibrated: bool
    source_id: str
    wuji_serial_number: str
    glove_hand: str
    wuji_torque_per_amp_nm: np.ndarray
    wuji_current_bias_a: np.ndarray
    glove_torque_constant_nm_per_a: np.ndarray
    max_current_a: np.ndarray
    max_slew_a_s: np.ndarray
    gain: float
    ramp_duration_s: float = 0.5
    jacobian_max_age_s: float = 0.5
    jacobian_max_pose_delta_rad: float = 0.25
    feedback_max_age_s: float = 0.1
    jacobian_refresh_interval_s: float = 0.1
    jacobian_step_rad: float = 0.01

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("力反馈标定schema_version必须为1")
        if self.calibrated is not True:
            raise CalibrationRequired("力反馈必须使用calibrated=true的真实单机标定")
        for name in ("source_id", "wuji_serial_number"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise CalibrationRequired(f"缺少力反馈标定{name}")
        if self.glove_hand not in ("left", "right"):
            raise CalibrationRequired("力反馈标定必须明确glove_hand为left或right")
        for name, size, positive, nonzero in (
            ("wuji_torque_per_amp_nm", 20, False, True),
            ("wuji_current_bias_a", 20, False, False),
            ("glove_torque_constant_nm_per_a", 12, True, False),
            ("max_current_a", 12, True, False),
            ("max_slew_a_s", 12, True, False),
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), size, name, positive=positive, nonzero=nonzero))
        for name in ("gain", "ramp_duration_s", "jacobian_max_age_s", "jacobian_max_pose_delta_rad",
                     "feedback_max_age_s", "jacobian_refresh_interval_s", "jacobian_step_rad"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.gain > 1:
            raise ValueError("力反馈gain不能超过1")

    @classmethod
    def from_json(cls, path: str | Path) -> ForceFeedbackCalibration:
        try:
            with Path(path).open(encoding="utf-8") as stream:
                data = json.load(stream)
        except (OSError, ValueError, TypeError) as error:
            raise CalibrationRequired("无法读取真实力反馈标定文件") from error
        if not isinstance(data, dict):
            raise CalibrationRequired("力反馈标定必须是JSON对象")
        try:
            return cls(**data)
        except TypeError as error:
            raise CalibrationRequired("力反馈标定字段缺失或未知") from error

    @property
    def glove_torque_constants_by_id(self) -> Mapping[int, float]:
        return MappingProxyType({index + 1: float(value) for index, value in enumerate(self.glove_torque_constant_nm_per_a)})

    def validate_serial(self, actual_serial: str) -> None:
        if not isinstance(actual_serial, str) or actual_serial != self.wuji_serial_number:
            raise CalibrationRequired("Wuji序列号与力反馈单机标定不一致")


@dataclass(frozen=True)
class ForceFeedbackCommand:
    model_current_a: tuple[float, ...] | None
    wuji_actuator_torque_nm: np.ndarray | None
    glove_joint_torque_nm: np.ndarray | None
    source_id: str
    jacobian_age_ns: int
    model_torque_nm: tuple[float, ...] | None = None
    wuji_load_fraction: np.ndarray | None = None
    mapping_mode: str = "calibrated"
    projection_status: str = "active"
    contact_state: str = "not_configured"
    contact_active_mask: np.ndarray | None = None
    virtual_hand_torque_nm: np.ndarray | None = None
    projected_model_torque_nm: tuple[float, ...] | None = None
    contact_reflection_gain: float | None = None

    def __post_init__(self):
        if self.projection_status not in ("active", "refreshing_pose", "refreshing_age"):
            raise ValueError("未知力反射投影状态")
        if self.contact_state not in ("not_configured", "free", "arming", "contact", "releasing", "waiting_geometry"):
            raise ValueError("未知受阻估计状态")
        if self.contact_active_mask is not None:
            mask = np.array(self.contact_active_mask, copy=True)
            if mask.shape != (20,) or mask.dtype != np.bool_:
                raise ValueError("受阻估计掩码必须为20维bool")
            mask.setflags(write=False)
            object.__setattr__(self, "contact_active_mask", mask)
        if (self.model_current_a is None) == (self.model_torque_nm is None):
            raise ValueError("力反馈命令必须明确且仅使用A或名义Nm一种输出单位")
        if self.projected_model_torque_nm is not None and self.model_torque_nm is None:
            raise ValueError("投影名义Nm不能附在A输出命令上")
        for name in ("model_current_a", "model_torque_nm", "projected_model_torque_nm"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(float(x) for x in _vector(value, 12, name)))
        if self.contact_reflection_gain is not None:
            object.__setattr__(self, "contact_reflection_gain",
                               _positive(self.contact_reflection_gain, "contact_reflection_gain"))
        for name, size in (
            ("wuji_actuator_torque_nm", 20), ("glove_joint_torque_nm", 12),
            ("wuji_load_fraction", 20),
            ("virtual_hand_torque_nm", 20),
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _vector(value, size, name))
        if self.mapping_mode in ("max_torque_linear_estimate", "contact_impedance_estimate"):
            if (self.model_torque_nm is None or self.wuji_load_fraction is None
                    or self.wuji_actuator_torque_nm is not None or self.glove_joint_torque_nm is not None):
                raise ValueError("线性近似只能报告负载比例与名义输出，不能伪造Wuji Nm")
        elif self.mapping_mode != "calibrated" or self.model_current_a is None:
            raise ValueError("未知力反馈模式或命令单位不匹配")
        if self.mapping_mode == "contact_impedance_estimate":
            if (self.contact_state == "not_configured" or self.contact_active_mask is None
                    or self.virtual_hand_torque_nm is None):
                raise ValueError("接触阻抗估计必须明确受阻状态及软件虚拟力矩")
        elif (self.contact_state != "not_configured" or self.contact_active_mask is not None
              or self.virtual_hand_torque_nm is not None or self.contact_reflection_gain is not None):
            raise ValueError("非接触阻抗模式不能携带受阻估计输出")


@dataclass(frozen=True)
class _Jacobian:
    value: np.ndarray
    joints: np.ndarray
    source_ns: int
    request_id: int

_JACOBIAN_COLUMNS = tuple(tuple(range(start, start + 3)) for start in range(0, 12, 3))

def _jacobian_process(factory, connection, step_rad, columns):
    """spawn独立解释器，精确求同一完整重定向Jac的指定列，不接触SDK。"""
    try:
        pipeline = factory()
        solver = ForceJacobianColumns(pipeline)
        signature = (
            pipeline.landmarks.model.kinematics.hand,
            tuple(pipeline.landmarks.model.motor_names),
            tuple(pipeline.retargeter.output_joint_names),
            pipeline.retargeter.fixed_joint_values,
        )
        connection.send(("ready", signature))
        while True:
            request = connection.recv()
            if request is None:
                break
            request_id, q, source_ns = request
            jacobian = solver.compute(q, columns, step_rad=step_rad)
            if jacobian.shape != (20, len(columns)) or not np.isfinite(jacobian).all():
                raise ValueError("完整重定向Jac分片必须为有限20x列数矩阵")
            connection.send(("result", request_id, jacobian, q, source_ns))
    except EOFError:
        pass
    except Exception as error:
        try:
            connection.send(("error", f"{type(error).__name__}: {error}"))
        except (EOFError, OSError):
            pass
    finally:
        connection.close()



class ForceFeedbackController:
    """CAD功率共轭、反作用符号及实际发送历史的限幅器。

    四个spawn进程各自独占pipeline/NLopt，每次共同计算同一原始姿态的12列。
    实时compute只提交容量为1的最新输入，不等待差分；prepare可等待首个完整Jac。
    """

    def __init__(self, config: ForceFeedbackCalibration | MaxTorqueLinearConfig,
                 retargeter: BidirectionalRetargeter, glove_driver,
                 pipeline_factory: Callable[[], GloveWujiRetargetPipeline], *,
                 clock_ns: Callable[[], int] = time.monotonic_ns,
                 contact_config: ContactFeedbackConfig | None = None):
        if not isinstance(config, (ForceFeedbackCalibration, MaxTorqueLinearConfig)):
            raise CalibrationRequired("缺少真实标定或显式确认的线性近似配置")
        self._config = config
        self.output_is_torque = isinstance(config, MaxTorqueLinearConfig)
        if contact_config is not None and (
                not isinstance(contact_config, ContactFeedbackConfig) or not self.output_is_torque):
            raise CalibrationRequired("接触阻抗估计需要显式接触参数和独立名义Nm限额配置")
        self.contact_config = contact_config
        self._contact = None if contact_config is None else ContactImpedance(contact_config)
        self._forward = retargeter.forward
        self._model = self._forward.landmarks.model
        self._driver = glove_driver
        if glove_driver.hand != config.glove_hand or self._model.kinematics.hand != config.glove_hand:
            raise CalibrationRequired("手套驱动、CAD与反馈参数手侧不一致")
        self._clock_ns = clock_ns
        self._pipeline_factory = pipeline_factory
        self._feedback = None if self.output_is_torque else CalibratedFeedback(CurrentTorqueCalibration(
            config.wuji_torque_per_amp_nm, config.wuji_current_bias_a, True, config.source_id))
        self._motor_names = tuple(self._model.motor_names)
        if len(set(self._motor_names)) != 12 or set(glove_driver.motors) != set(range(1, 13)):
            raise CalibrationRequired("力反馈需要完整12轴设备绑定")
        self._kt = None if self.output_is_torque else np.empty(12)
        self._max_current = None if self.output_is_torque else np.empty(12)
        self._max_torque = np.empty(12) if self.output_is_torque else None
        self._max_ack_torque = np.empty(12) if self.output_is_torque else None
        self._codec_zero_torque = np.empty(12) if self.output_is_torque else None
        self._max_slew = np.empty(12)
        self._limits = np.empty((12, 2))
        constants = None if self.output_is_torque else glove_driver.model_motor_torque_constants_nm_per_a
        caps = (
            glove_driver.model_force_torque_limits_nm if self.output_is_torque
            else glove_driver.model_current_limits_a
        )
        limits = glove_driver.model_motor_limits_rad
        command_caps = glove_driver.model_force_torque_command_limits_nm if self.output_is_torque else None
        metadata = (caps, command_caps, limits) if constants is None else (constants, caps, limits)
        if any(set(values) != set(self._motor_names) for values in metadata):
            raise CalibrationRequired("手套模型电机元数据不完整")
        indices = {name: index for index, name in enumerate(self._motor_names)}
        bound = set()
        for raw_id, motor in glove_driver.motors.items():
            name = self._model.joint_motor_names[motor.joint]
            if name in bound:
                raise CalibrationRequired("手套raw轴重复绑定到同一模型电机")
            bound.add(name)
            index = indices[name]
            scale = abs(float(motor.sign * motor.scale))
            if not np.isfinite(scale) or scale <= 0:
                raise CalibrationRequired("手套电流／力矩方向比例未标定")
            if self.output_is_torque:
                declared_cap = config.glove_max_output_torque_nm[raw_id - 1] / scale
                if not np.isclose(float(command_caps[name]), declared_cap, rtol=1e-12, atol=1e-12):
                    raise CalibrationRequired("手套名义Nm限额与线性反馈配置不一致")
                self._max_ack_torque[index] = declared_cap
                self._codec_zero_torque[index] = declared_cap - float(caps[name])
                self._max_torque[index] = min(
                    float(caps[name]), config.glove_max_output_torque_nm[raw_id - 1] / scale,
                )
                self._max_slew[index] = config.max_torque_slew_nm_s[raw_id - 1] / scale
            else:
                kt = float(constants[name])
                if not np.isclose(kt, config.glove_torque_constant_nm_per_a[raw_id - 1], rtol=1e-12, atol=0):
                    raise CalibrationRequired("手套驱动Kt与真实力反馈标定不一致")
                self._kt[index] = kt
                self._max_current[index] = min(config.max_current_a[raw_id - 1] / scale, float(caps[name]))
                self._max_slew[index] = config.max_slew_a_s[raw_id - 1] / scale
            self._limits[index] = limits[name]
        maximum = self._max_torque if self.output_is_torque else self._max_current
        if (not np.isfinite(self._limits).all() or np.any(self._limits[:, 0] >= self._limits[:, 1])
                or not np.isfinite(maximum).all() or np.any(maximum < 0)):
            raise CalibrationRequired("手套模型限位或输出限幅非法")
        self._condition = threading.Condition()
        self._pending = None
        self._snapshot = None
        self._worker_error = None
        self._last_submit_ns = None
        self._next_request_id = 0
        self._closed = False
        self._threads = []
        self._processes = []
        self._prepared = False
        self._reset_ns = None
        self._history_ns = None
        self._history = np.zeros(12)
        self._pending_command_ns = None
        self._last_feedback_ns = None
        self._reflection_active = True

    @property
    def config(self) -> ForceFeedbackCalibration | MaxTorqueLinearConfig:
        return self._config

    def _require_open(self):
        if self._closed:
            raise RuntimeError("力反馈计算器已经关闭")

    @staticmethod
    def _stamp(value, now_ns: int, maximum_ns: int, label: str):
        if type(value) is not int or value < 0 or value > now_ns or now_ns - value > maximum_ns:
            raise TimeoutError(f"{label}原始反馈时戳缺失、来自未来或过期")

    def _validate_inputs(self, hand2_state, glove_state, now_ns):
        self._require_open()
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns必须为非负整数")
        maximum = int(self.config.feedback_max_age_s * 1e9)
        for value, label in ((hand2_state.received_ns, "Wuji"), (glove_state.received_ns, "手套"),
                             (glove_state.pose.timestamp_ns, "手套姿态")):
            self._stamp(value, now_ns, maximum, label)
        if hand2_state.effort_a is None:
            raise CalibrationRequired("Wuji缺少实测effort电流，禁止力反馈")
        _vector(hand2_state.effort_a, 20, "Wuji实测effort")
        if hand2_state.online_count != 20 or set(glove_state.online_ids) != set(range(1, 13)):
            raise RuntimeError("力反馈要求两侧所有电机在线")
        serial = hand2_state.diagnostics.get("serial_number")
        if serial is not None:
            self.config.validate_serial(serial)
        stamps = (hand2_state.received_ns, glove_state.received_ns, glove_state.pose.timestamp_ns)
        if self._last_feedback_ns is not None and any(a < b for a, b in zip(stamps, self._last_feedback_ns)):
            raise ValueError("力反馈原始时戳倒退")
        pose = self._model.kinematics.evaluate_tree(glove_state.tree_positions_rad)
        actual = self._model.to_glove_pose(pose)
        q = actual.vector()
        if not np.allclose(q, glove_state.pose.vector(), rtol=0, atol=1e-7):
            raise ValueError("手套姿态与本次完整CAD树不一致")
        if set(glove_state.motor_positions_rad) != set(self._motor_names):
            raise ValueError("缺少本次完整模型电机反馈")
        motors = np.array([glove_state.motor_positions_rad[name] for name in self._motor_names])
        tree_motors = np.array([pose.motor_positions[name] for name in self._motor_names])
        if not np.isfinite(motors).all() or not np.allclose(motors, tree_motors, rtol=0, atol=1e-7):
            raise ValueError("模型电机反馈与本次完整CAD树不一致")
        return pose, q, tree_motors, stamps

    def _submit(self, q, source_ns, now_ns, *, force=False):
        with self._condition:
            if self._closed:
                raise RuntimeError("力反馈计算器已经关闭")
            if not self._threads:
                thread = threading.Thread(
                    target=self._work, name="force-feedback-jacobian", daemon=True,
                )
                self._threads.append(thread)
                thread.start()
            if force or self._last_submit_ns is None or now_ns - self._last_submit_ns >= int(self.config.jacobian_refresh_interval_s * 1e9):
                self._next_request_id += 1
                self._pending = (self._next_request_id, q.copy(), source_ns)
                self._last_submit_ns = now_ns
                self._condition.notify_all()
                return self._next_request_id

    def _work(self):
        context = multiprocessing.get_context("spawn")
        parents, processes = [], []
        expected = (
            self._model.kinematics.hand, self._motor_names,
            tuple(self._forward.retargeter.output_joint_names),
            self._forward.retargeter.fixed_joint_values,
        )

        def receive(connection):
            while not connection.poll(0.05):
                with self._condition:
                    if self._closed:
                        return None
            with self._condition:
                if self._closed:
                    return None
            return connection.recv()

        try:
            for index, columns in enumerate(_JACOBIAN_COLUMNS):
                with self._condition:
                    if self._closed:
                        return
                parent, child = context.Pipe()
                process = context.Process(
                    target=_jacobian_process,
                    args=(self._pipeline_factory, child, self.config.jacobian_step_rad, columns),
                    name=f"force-feedback-jacobian-{index}", daemon=True,
                )
                parents.append(parent)
                processes.append(process)
                with self._condition:
                    self._processes.append(process)
                try:
                    process.start()
                finally:
                    child.close()
            for parent in parents:
                ready = receive(parent)
                if ready is None:
                    return
                if ready[0] != "ready":
                    raise RuntimeError(f"完整Jac子进程初始化失败: {ready[1]}")
                if ready[1] != expected:
                    raise ValueError("后台Jac模型、轴序或固定关节配置与主求解器不同")
            while True:
                with self._condition:
                    while self._pending is None and not self._closed:
                        self._condition.wait()
                    if self._closed:
                        return
                    request, self._pending = self._pending, None
                # 整批使用同一q/原始时戳，不能拼接来自不同姿态的列。
                for parent in parents:
                    parent.send(request)
                request_id, q, source_ns = request
                jacobian = np.empty((20, 12), dtype=float)
                for parent, columns in zip(parents, _JACOBIAN_COLUMNS):
                    response = receive(parent)
                    if response is None:
                        return
                    if response[0] != "result":
                        raise RuntimeError(f"完整Jac子进程计算失败: {response[1]}")
                    _, part_id, part, part_q, part_ns = response
                    if (part_id != request_id or part_ns != source_ns
                            or not np.array_equal(part_q, q)
                            or part.shape != (20, len(columns)) or not np.isfinite(part).all()):
                        raise ValueError("完整Jac分片的姿态、时戳、请求号或维数不一致")
                    jacobian[:, columns] = part
                jacobian.setflags(write=False)
                with self._condition:
                    if self._snapshot is None or request_id > self._snapshot.request_id:
                        self._snapshot = _Jacobian(jacobian, q, source_ns, request_id)
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._worker_error = error
                self._condition.notify_all()
        finally:
            for parent in parents:
                try:
                    parent.send(None)
                except (EOFError, OSError):
                    pass
            finish = time.monotonic() + 1.0
            for process in processes:
                if process.pid is not None:
                    process.join(max(0.0, finish - time.monotonic()))
            for process in processes:
                if process.pid is not None:
                    if process.is_alive():
                        process.terminate()
                    process.join()
            for parent in parents:
                parent.close()

    def validate_inputs(self, hand2_state: WujiState, glove_state: EncosGloveState, *,
                        now_ns: int) -> None:
        """在取得输出权之前检查实测反馈与标定，不等待几何计算。"""
        self._validate_inputs(hand2_state, glove_state, now_ns)

    def prepare(self, hand2_state: WujiState, glove_state: EncosGloveState, *,
                now_ns: int | None = None, timeout_s: float = 2) -> None:
        timeout_s = _positive(timeout_s, "timeout_s")
        now = self._clock_ns() if now_ns is None else now_ns
        _, q, _, _ = self._validate_inputs(hand2_state, glove_state, now)
        source_ns = min(glove_state.received_ns, glove_state.pose.timestamp_ns)
        request_id = self._submit(q, source_ns, now, force=True)
        finish = time.monotonic() + timeout_s
        with self._condition:
            if self._worker_error is not None:
                raise RuntimeError("完整重定向Jac后台计算失败") from self._worker_error
            while self._snapshot is None or self._snapshot.request_id < request_id:
                if self._worker_error is not None:
                    raise RuntimeError("完整重定向Jac后台计算失败") from self._worker_error
                self._require_open()
                remaining = finish - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("完整重定向Jac预热超时")
                self._condition.wait(remaining)
        self._prepared = True

    def reset(self, *, now_ns: int) -> None:
        self._require_open()
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns必须为非负整数")
        self._reset_ns = self._history_ns = now_ns
        self._history.fill(0)
        self._pending_command_ns = None
        self._last_feedback_ns = None
        self._reflection_active = True
        if self._contact is not None:
            self._contact.reset()

    def compute(self, hand2_state: WujiState, glove_state: EncosGloveState, *,
                now_ns: int, deadline_ns: int,
                target_positions_rad: np.ndarray | None = None) -> ForceFeedbackCommand:
        pose, q, motors, stamps = self._validate_inputs(hand2_state, glove_state, now_ns)
        if self._contact is not None:
            if target_positions_rad is None:
                raise ValueError("接触阻抗估计缺少既有控制链路的最终Wuji目标")
            target_positions_rad = _vector(target_positions_rad, 20, "Wuji最终位置目标")
        if type(deadline_ns) is not int or deadline_ns <= now_ns or deadline_ns <= self._clock_ns():
            raise TimeoutError("力反馈命令期限已过期")
        if not self._prepared or self._reset_ns is None:
            raise RuntimeError("力反馈必须先prepare并reset到零输出")
        if now_ns < self._history_ns or now_ns < self._reset_ns:
            raise ValueError("力反馈计算时间倒退")
        with self._condition:
            snapshot, error = self._snapshot, self._worker_error
        if error is not None:
            raise RuntimeError("完整重定向Jac后台计算失败") from error
        if snapshot is None:
            raise RuntimeError("缺少完整重定向Jac")
        age = now_ns - snapshot.source_ns
        if age < 0:
            raise TimeoutError("完整重定向Jac原始姿态来自未来")
        pose_delta = float(np.max(np.abs(q - snapshot.joints)))
        status = (
            "refreshing_age" if age > int(self.config.jacobian_max_age_s * 1e9) else
            "refreshing_pose" if pose_delta > self.config.jacobian_max_pose_delta_rad else
            "active"
        )
        # 正常运动可快于低频差分。立即更新待算姿态，不等常规周期，
        # 也不阻塞实时环或将旧矩阵当成当前姿态的力矩投影。
        self._submit(q, min(stamps[1:]), now_ns, force=status != "active")
        kinematics = self._model.kinematics
        motor_jacobian = kinematics._tree_derivatives(pose.all_positions)[kinematics._motor_indices]
        if (motor_jacobian.shape != (12, 12) or not np.isfinite(motor_jacobian).all()
                or np.linalg.matrix_rank(motor_jacobian) != 12):
            raise ValueError("当前CAD传动Jac奇异或非有限")
        actuator = joint_torque = load_fraction = None
        virtual_torque = contact_mask = None
        contact_state = "not_configured"
        if self._contact is not None:
            maximum = self._max_torque
            load_fraction = self.config.load_fraction(hand2_state.effort_a)
            # 受阻判据消费当周期新鲜状态；几何失效只撤力，不丢失接触或释放历史。
            contact = self._contact.step(
                target_positions_rad, hand2_state, source_ns=min(stamps),
                observable_joints=np.any(snapshot.value != 0, axis=1),
                current_bias_a=self.config.wuji_current_bias_a,
            )
            virtual_torque = contact.virtual_hand_torque_nm
            contact_mask = contact.active_mask
            if status == "active":
                contact_state = contact.phase
                # 仅消费既有导数；软件虚拟力矩不是Wuji实测力矩，也不改变正向目标。
                with np.errstate(over="raise", invalid="raise"):
                    output = (
                        self.config.gain * self.contact_config.reflection_gain
                    ) * GloveTransmission.joint_torque_to_motor_torque_from_jacobian(
                        motor_jacobian, snapshot.value.T @ virtual_torque,
                    )
            else:
                contact_state = "waiting_geometry"
                output = np.zeros(12)
        elif self.output_is_torque:
            maximum = self._max_torque
            if status == "active":
                projection = project_max_torque_linear(
                    snapshot.value, motor_jacobian, hand2_state.effort_a, self.config, maximum,
                )
                output = projection.model_torque_nm.copy()
                load_fraction = projection.wuji_load_fraction
            else:
                load_fraction = self.config.load_fraction(hand2_state.effort_a)
                output = np.zeros(12)
        else:
            maximum = self._max_current
            actuator = (hand2_state.effort_a - self.config.wuji_current_bias_a) * self.config.wuji_torque_per_amp_nm
            if status == "active":
                projected = self._feedback.project_effort(
                    RetargetedPose(hand2_state.positions_rad, snapshot.value), hand2_state.effort_a,
                )
                joint_torque = -self.config.gain * projected.glove_torque_nm
                motor_torque = GloveTransmission.joint_torque_to_motor_torque_from_jacobian(motor_jacobian, joint_torque)
                output = GloveTransmission.motor_torque_to_current(motor_torque, self._kt)
            else:
                output = np.zeros(12)
        # 在output被原地乘渐入前保存模型轴名义Nm；无效几何和A模式不能伪造该量。
        projected_model_torque = (
            tuple(float(value) for value in output)
            if self.output_is_torque and status == "active" else None
        )
        if status != "active":
            self._reflection_active = False
        elif not self._reflection_active:
            # 接触模式仅在真正start/reset时渐入；运行期恢复从零按斜率有界衔接。
            if self._contact is None:
                self._reset_ns = now_ns
            self._reflection_active = True
        x = min(1.0, (now_ns - self._reset_ns) / (self.config.ramp_duration_s * 1e9))
        output *= x * x * x * (10 + x * (-15 + 6 * x))
        output = np.clip(output, -maximum, maximum)
        if status == "active" and (virtual_torque is None or np.any(virtual_torque != 0)):
            delta = self._max_slew * ((now_ns - self._history_ns) / 1e9)
            if self._contact is None:
                output = np.clip(output, self._history - delta, self._history + delta)
            else:
                # 受阻增强按斜率渐入；减力立即生效，换向先归零再按斜率建立。
                # 多指同时接触时不能因另一指仍有力而继续保留已张开手指的旧力。
                reversal = self._history * output < 0.0
                magnitude = np.minimum(np.abs(output), np.abs(self._history) + delta)
                output = np.where(reversal, 0.0, np.copysign(magnitude, output))
        # 无效局部导数期间直接归零，不让斜率历史继续输出旧姿态的反作用力。
        if virtual_torque is not None and not np.any(virtual_torque != 0):
            output.fill(0)
        outward = ((motors <= self._limits[:, 0]) & (output < 0)) | ((motors >= self._limits[:, 1]) & (output > 0))
        output[outward] = 0
        if self._clock_ns() >= deadline_ns:
            raise TimeoutError("力反馈计算超过命令期限")
        self._pending_command_ns = now_ns
        self._last_feedback_ns = stamps
        self._pending_nominal_output = output.copy() if self.output_is_torque else None
        values = tuple(float(x) for x in output)
        if self.output_is_torque:
            mode = "contact_impedance_estimate" if self._contact is not None else "max_torque_linear_estimate"
            source_id = self.config.source_id
            if self.contact_config is not None:
                source_id += ";" + self.contact_config.source_id
            return ForceFeedbackCommand(
                None, None, None, source_id, age,
                model_torque_nm=values, wuji_load_fraction=load_fraction,
                mapping_mode=mode, projection_status=status,
                contact_state=contact_state, contact_active_mask=contact_mask,
                virtual_hand_torque_nm=virtual_torque,
                projected_model_torque_nm=projected_model_torque,
                contact_reflection_gain=(None if self.contact_config is None else
                                         self.config.gain * self.contact_config.reflection_gain),
            )
        return ForceFeedbackCommand(
            values, actuator, joint_torque, self.config.source_id, age, projection_status=status,
        )

    def acknowledge(self, current_a: tuple[float, ...]) -> None:
        if self.output_is_torque:
            raise ValueError("当前输出为名义Nm，不能用A确认")
        self._acknowledge(current_a, self._max_current, "已发送模型电流A")

    def acknowledge_torque(self, torque_nm: tuple[float, ...]) -> None:
        if not self.output_is_torque:
            raise ValueError("当前输出为A，不能用名义Nm确认")
        self._acknowledge(torque_nm, self._max_ack_torque, "已发送模型名义力矩Nm")

    def _acknowledge(self, values, maximum, label) -> None:
        self._require_open()
        if self._pending_command_ns is None:
            raise RuntimeError("没有待确认的力反馈命令")
        output = _vector(values, 12, label)
        if np.any(np.abs(output) > maximum + 1e-12):
            raise ValueError("已发送目标超过当前模式独立限幅")
        if self.output_is_torque:
            # 量化零附近的向内档不是新的连续力矩目标，否则小步斜率会积累偏置。
            pending = self._pending_nominal_output
            zero_bin = np.isclose(np.abs(output), self._codec_zero_torque, rtol=1e-9, atol=1e-12)
            same_small = (pending * output >= 0) & (np.abs(pending) <= self._codec_zero_torque + 1e-12)
            history = np.where(
                zero_bin,
                np.where(same_small, pending, np.where(pending * output < 0, 0.0, output)),
                output,
            )
            self._history[:] = np.clip(history, -self._max_torque, self._max_torque)
        else:
            self._history[:] = output
        self._history_ns = self._pending_command_ns
        self._pending_command_ns = None

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()
        # 管理线程仅做IPC；数值子进程不接设备，异常卡住时可结束而不碰硬件。
        finish = time.monotonic() + 2.0
        for thread in self._threads:
            thread.join(max(0.0, finish - time.monotonic()))
        for process in tuple(self._processes):
            if process.pid is not None and process.is_alive():
                process.terminate()
        for thread in self._threads:
            thread.join()
