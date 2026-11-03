"""由位置跟踪、电流负载及速度推断受阻，再生成单向虚拟阻抗。

20轴顺序与WujiState一致。电流不是独立接触力传感器；输出仅为软件虚拟
手关节名义Nm，不是Wuji实测力矩。本模块没有SDK、几何投影或Retarget调用。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Literal

import numpy as np

from ..domain import WujiState
from ..feedback import CalibrationRequired


ContactPhase = Literal["free", "arming", "contact", "releasing"]


def _real_array(value, name: str) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if raw.shape != (20,) or any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, Real)
        for item in raw.flat
    ):
        raise ValueError(f"{name}必须是20维有限实数数组，不能含bool")
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name}必须是20维有限实数数组") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name}必须是20维有限实数数组")
    result.setflags(write=False)
    return result


def _bool_array(value, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != (20,) or result.dtype != np.dtype(bool):
        raise ValueError(f"{name}必须是20维bool数组")
    result = result.copy()
    result.setflags(write=False)
    return result


def _positive(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name}必须是正有限实数")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name}必须是正有限实数") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name}必须是正有限实数")
    return result


@dataclass(frozen=True)
class ContactFeedbackConfig:
    """显式承认估计的20轴阈值与软件系数；字段后缀给出SI单位。"""

    schema_version: int
    estimate_acknowledged: bool
    source_id: str
    current_on_a: np.ndarray
    current_off_a: np.ndarray
    error_on_rad: np.ndarray
    error_off_rad: np.ndarray
    stalled_speed_rad_s: np.ndarray
    stiffness_nm_per_rad: np.ndarray
    damping_nm_s_per_rad: np.ndarray
    dwell_s: float = 0.15
    release_dwell_s: float = 0.06
    current_filter_hz: float = 5.0
    output_filter_hz: float = 8.0
    max_sample_gap_s: float = 0.1
    load_weight_scale_a: float = 0.5
    opening_speed_deadband_rad_s: float = 0.02
    relative_velocity_limit_rad_s: float = 2.0
    reflection_gain: float = 1.0

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("接触阻抗估计schema_version必须为1")
        if self.estimate_acknowledged is not True:
            raise ValueError("必须显式estimate_acknowledged=true确认虚拟接触阻抗估计")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("必须明确接触阻抗估计source_id")
        for name in ("current_on_a", "current_off_a", "error_on_rad", "error_off_rad",
                     "stalled_speed_rad_s", "stiffness_nm_per_rad", "damping_nm_s_per_rad"):
            object.__setattr__(self, name, _real_array(getattr(self, name), name))
        for prefix in ("current", "error"):
            suffix = "a" if prefix == "current" else "rad"
            on = getattr(self, f"{prefix}_on_{suffix}")
            off = getattr(self, f"{prefix}_off_{suffix}")
            if np.any(on <= 0) or np.any(off < 0) or np.any(off >= on):
                raise ValueError(f"{prefix}每轴必须满足on>off>=0")
        if np.any(self.stalled_speed_rad_s <= 0) or np.any(self.stiffness_nm_per_rad <= 0):
            raise ValueError("堵转速度阈值与虚拟刚度必须每轴严格为正")
        if np.any(self.damping_nm_s_per_rad < 0):
            raise ValueError("软件虚拟阻尼不能为负")
        for name in ("reflection_gain", "dwell_s", "release_dwell_s", "current_filter_hz",
                     "output_filter_hz", "max_sample_gap_s", "load_weight_scale_a",
                     "opening_speed_deadband_rad_s", "relative_velocity_limit_rad_s"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.opening_speed_deadband_rad_s >= self.relative_velocity_limit_rad_s:
            raise ValueError("张开撤力速度死区必须小于相对速度限幅")

    @classmethod
    def from_json(cls, path: str | Path) -> ContactFeedbackConfig:
        try:
            with Path(path).open(encoding="utf-8") as stream:
                data = json.load(stream)
        except (OSError, ValueError, TypeError) as error:
            raise CalibrationRequired("无法读取接触阻抗估计配置文件") from error
        if not isinstance(data, dict):
            raise CalibrationRequired("接触阻抗估计配置必须是JSON对象")
        try:
            return cls(**data)
        except TypeError as error:
            raise CalibrationRequired("接触阻抗估计配置字段缺失或未知") from error


@dataclass(frozen=True)
class ContactFeedbackResult:
    """不可变估计快照；active_mask在释放驻留结束前仍保留接触锁存。

    多轴聚合phase优先级为contact > releasing > arming > free。
    释放候选及不可观测轴的虚拟力矩立即清零，不受输出低通影响。
    """

    virtual_hand_torque_nm: np.ndarray
    active_mask: np.ndarray
    phase: ContactPhase
    tracking_error_rad: np.ndarray

    def __post_init__(self) -> None:
        if self.phase not in ("free", "arming", "contact", "releasing"):
            raise ValueError("未知接触阻抗阶段")
        object.__setattr__(self, "virtual_hand_torque_nm",
                           _real_array(self.virtual_hand_torque_nm, "virtual_hand_torque_nm"))
        object.__setattr__(self, "active_mask", _bool_array(self.active_mask, "active_mask"))
        object.__setattr__(self, "tracking_error_rad",
                           _real_array(self.tracking_error_rad, "tracking_error_rad"))


class ContactImpedance:
    """源帧时间驱动的受阻迟滞与单向弹簧阻尼；reset允许重新建立时间基准。"""

    def __init__(self, config: ContactFeedbackConfig):
        if not isinstance(config, ContactFeedbackConfig):
            raise ValueError("需要显式ContactFeedbackConfig")
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._last_source_ns: int | None = None
        self._last_hand_ns: int | None = None
        self._last_positions_rad: np.ndarray | None = None
        self._last_target_rad: np.ndarray | None = None
        self._filtered_current_a: np.ndarray | None = None
        self._active = np.zeros(20, dtype=bool)
        self._direction = np.zeros(20, dtype=np.int8)
        self._candidate_direction = np.zeros(20, dtype=np.int8)
        self._candidate_since_ns: list[int | None] = [None] * 20
        self._release_since_ns: list[int | None] = [None] * 20
        self._output_nm = np.zeros(20)

    def _clear_axis(self, axis: int) -> None:
        self._active[axis] = False
        self._direction[axis] = 0
        self._candidate_direction[axis] = 0
        self._candidate_since_ns[axis] = None
        self._release_since_ns[axis] = None
        self._output_nm[axis] = 0.0

    def _result(self, error_rad: np.ndarray) -> ContactFeedbackResult:
        contacting = any(self._active[i] and self._release_since_ns[i] is None
                         for i in range(20))
        if contacting:
            phase: ContactPhase = "contact"
        elif self._active.any():
            phase = "releasing"
        elif any(start is not None for start in self._candidate_since_ns):
            phase = "arming"
        else:
            phase = "free"
        return ContactFeedbackResult(self._output_nm, self._active, phase, error_rad)

    def step(
        self,
        target_positions_rad: np.ndarray,
        hand_state: WujiState,
        *,
        source_ns: int,
        observable_joints: np.ndarray,
        current_bias_a: np.ndarray,
    ) -> ContactFeedbackResult:
        """只消费已生成的目标与原始状态，不补缺失电流或虚假零速度。

        source_ns由调用方验证为两侧反馈与手套姿态的最早时间。重复源帧只允许
        安全释放/清零，不推进滤波、驻留或导数。缺原始速度时，要求源时间及
        Wuji收包时间都真正前进，并用Wuji收包间隔求实测位置差分rad/s；
        目标速度按源时间求差分。两者之差为弹簧穿透相对速度rad/s。
        """
        if not isinstance(hand_state, WujiState):
            raise ValueError("hand_state必须是WujiState")
        if (type(source_ns) is not int or source_ns < 0
                or source_ns > hand_state.received_ns):
            raise ValueError("source_ns必须是非负整数且不晚于Wuji原始接收时间")
        if self._last_source_ns is not None and source_ns < self._last_source_ns:
            raise ValueError("source_ns不能倒退")
        target = _real_array(target_positions_rad, "target_positions_rad")
        observable = _bool_array(observable_joints, "observable_joints")
        bias = _real_array(current_bias_a, "current_bias_a")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                error_rad = target - hand_state.positions_rad
                return self._step(target, hand_state, source_ns, observable, bias, error_rad)
        except FloatingPointError as error:
            self.reset()
            self._last_source_ns = source_ns
            raise ValueError("接触阻抗估计数值非有限") from error

    def _step(self, target, hand, source_ns, observable, bias, error_rad):
        config = self.config
        elapsed_ns = None if self._last_source_ns is None else source_ns - self._last_source_ns
        if elapsed_ns is not None and elapsed_ns / 1_000_000_000 > config.max_sample_gap_s:
            self.reset()
            elapsed_ns = None
        if hand.effort_a is None:
            self.reset()
            self._last_source_ns = source_ns
            return self._result(error_rad)

        new_frame = elapsed_ns is None or elapsed_ns > 0
        dt_s = 0.0 if elapsed_ns is None else elapsed_ns / 1_000_000_000
        measured_velocity = hand.velocities_rad_s
        target_velocity = None
        if new_frame:
            if dt_s > 0 and self._last_target_rad is not None:
                target_velocity = (target - self._last_target_rad) / dt_s
            if (measured_velocity is None and self._last_hand_ns is not None
                    and hand.received_ns > self._last_hand_ns
                    and self._last_positions_rad is not None):
                hand_dt_s = (hand.received_ns - self._last_hand_ns) * 1e-9
                measured_velocity = (hand.positions_rad - self._last_positions_rad) / hand_dt_s
            load_a = np.abs(hand.effort_a - bias)
            if self._filtered_current_a is None:
                self._filtered_current_a = load_a
            else:
                alpha = -math.expm1(-(config.current_filter_hz * dt_s) * math.tau)
                self._filtered_current_a += alpha * (load_a - self._filtered_current_a)

        output_alpha = -math.expm1(-(config.output_filter_hz * dt_s) * math.tau)
        for axis in range(20):
            if not observable[axis]:
                self._clear_axis(axis)
                continue
            if self._active[axis]:
                n = self._direction[axis]
                penetration_rad = n * error_rad[axis]
                releasing = (penetration_rad <= config.error_off_rad[axis]
                             or self._filtered_current_a[axis] <= config.current_off_a[axis])
                if releasing:
                    # 先清输出低通历史，再等待状态迟滞；绝不保留残余反作用。
                    self._output_nm[axis] = 0.0
                    if self._release_since_ns[axis] is None:
                        self._release_since_ns[axis] = source_ns
                    if (source_ns - self._release_since_ns[axis]) / 1_000_000_000 >= config.release_dwell_s:
                        self._clear_axis(axis)
                    continue
                self._release_since_ns[axis] = None
            else:
                if not new_frame:
                    # 同源的新目标可否定候选，但不能为相反方向累计新驻留。
                    if (abs(error_rad[axis]) < config.error_on_rad[axis]
                            or int(np.sign(error_rad[axis])) != self._candidate_direction[axis]
                            or (measured_velocity is not None
                                and abs(measured_velocity[axis]) > config.stalled_speed_rad_s[axis])):
                        self._clear_axis(axis)
                    continue
                direction = int(np.sign(error_rad[axis]))
                eligible = (abs(error_rad[axis]) >= config.error_on_rad[axis]
                            and self._filtered_current_a[axis] >= config.current_on_a[axis]
                            and measured_velocity is not None
                            and abs(measured_velocity[axis]) <= config.stalled_speed_rad_s[axis])
                if not eligible:
                    self._clear_axis(axis)
                    continue
                if (self._candidate_since_ns[axis] is None
                        or direction != self._candidate_direction[axis]):
                    self._candidate_since_ns[axis] = source_ns
                    self._candidate_direction[axis] = direction
                if (source_ns - self._candidate_since_ns[axis]) / 1_000_000_000 < config.dwell_s:
                    continue
                self._active[axis] = True
                self._direction[axis] = direction
                self._candidate_since_ns[axis] = None
                self._candidate_direction[axis] = 0
            if not new_frame:
                if (self._last_target_rad is not None
                        and self._direction[axis] * (target[axis] - self._last_target_rad[axis]) < 0):
                    # 同源帧无法估算速度，但目标已经回开时不能保留旧反力。
                    self._output_nm[axis] = 0.0
                continue
            n = self._direction[axis]
            penetration_rad = max(n * error_rad[axis] - config.error_off_rad[axis], 0.0)
            relative_speed_rad_s = 0.0
            if target_velocity is not None and measured_velocity is not None:
                relative_speed_rad_s = min(
                    max(n * (target_velocity[axis] - measured_velocity[axis]),
                        -config.relative_velocity_limit_rad_s),
                    config.relative_velocity_limit_rad_s,
                )
            if relative_speed_rad_s < -config.opening_speed_deadband_rad_s:
                # 张开时立即清掉低通历史，防止旧弹簧力继续把手拉回。
                self._output_nm[axis] = 0.0
                continue
            # Franka参考控制器使用外力矩连续加权；Wuji只有电流，因此这里只用
            # 无量纲负载权重调制虚拟弹簧，绝不将A直接解释为接触Nm。
            load_ratio = min(float(self._filtered_current_a[axis] / config.load_weight_scale_a), 20.0)
            speed_ratio = min(float(abs(relative_speed_rad_s) / config.stalled_speed_rad_s[axis]), 20.0)
            load_weight = -math.expm1(-(load_ratio * load_ratio))
            damping_weight = (0.6 / (1.0 + math.exp(-load_ratio))
                              + 0.4 / (1.0 + math.exp(-speed_ratio)))
            # 张开时相对速度为负，阻尼先抵消弹簧；不能反向变成继续闭合的推力。
            requested_nm = -n * max(
                load_weight * config.stiffness_nm_per_rad[axis] * penetration_rad
                + damping_weight * config.damping_nm_s_per_rad[axis] * relative_speed_rad_s,
                0.0,
            )
            self._output_nm[axis] += output_alpha * (requested_nm - self._output_nm[axis])

        if new_frame:
            self._last_source_ns = source_ns
            self._last_target_rad = target
            # 同一Wuji收包不会覆盖真正的差分基准，也不能产生假零速度。
            if self._last_hand_ns is None or hand.received_ns > self._last_hand_ns:
                self._last_hand_ns = hand.received_ns
                self._last_positions_rad = hand.positions_rad
        return self._result(error_rad)
