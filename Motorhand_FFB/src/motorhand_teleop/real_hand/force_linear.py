"""显式确认的最大负载占比近似；输出是SDK名义Nm，不是实测接触力。

Wuji反馈仅以实测A除显式电流满量程归一化，不生成Wuji Nm或假Kt。
完整CAD/重定向几何经最坏符号组合包络归一化后，使用各模型电机独立的
允许名义Nm上限；这不是定量功率守恒的力矩还原。
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ..feedback import CalibrationRequired


def _real_array(value, shape: tuple[int, ...], name: str, *, positive: bool = False) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if raw.shape != shape or any(
        isinstance(item, (bool, np.bool_))
        or not isinstance(item, (int, float, np.number))
        or np.iscomplexobj(item)
        for item in raw.flat
    ):
        raise ValueError(f"{name}必须是形状{shape}的有限实数数组")
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name}必须是有限实数数组") from error
    if not np.isfinite(result).all() or (positive and np.any(result <= 0)):
        raise ValueError(f"{name}必须有限" + ("且严格为正" if positive else ""))
    result.setflags(write=False)
    return result


def _positive(value, name: str) -> float:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.number)) or np.iscomplexobj(value)):
        raise ValueError(f"{name}必须为正有限实数")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name}必须为正有限实数") from error
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name}必须为正有限实数")
    return result


@dataclass(frozen=True)
class MaxTorqueLinearConfig:
    """独立于真实Nm/A标定的配置；12轴数组按raw SDK motor ID 1..12。"""

    schema_version: int
    mode: str
    estimate_acknowledged: bool
    source_id: str
    wuji_serial_number: str
    glove_hand: str
    wuji_max_current_a: np.ndarray
    wuji_current_bias_a: np.ndarray
    wuji_effort_sign: np.ndarray
    glove_max_output_torque_nm: np.ndarray
    max_torque_slew_nm_s: np.ndarray
    gain: float
    ramp_duration_s: float = 0.5
    jacobian_max_age_s: float = 0.5
    jacobian_max_pose_delta_rad: float = 0.25
    feedback_max_age_s: float = 0.1
    jacobian_refresh_interval_s: float = 0.1
    jacobian_step_rad: float = 0.01

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("最大负载线性估计schema_version必须为1")
        if self.mode != "max_torque_linear_estimate":
            raise ValueError("必须显式选择max_torque_linear_estimate模式")
        if self.estimate_acknowledged is not True:
            raise ValueError("必须显式estimate_acknowledged=true确认名义力矩近似")
        for name in ("source_id", "wuji_serial_number"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"缺少最大负载线性估计{name}")
        if self.glove_hand not in ("left", "right"):
            raise ValueError("必须明确glove_hand为left或right")
        for name, size, positive in (
            ("wuji_max_current_a", 20, True),
            ("wuji_current_bias_a", 20, False),
            ("wuji_effort_sign", 20, False),
            ("glove_max_output_torque_nm", 12, True),
            ("max_torque_slew_nm_s", 12, True),
        ):
            object.__setattr__(self, name, _real_array(getattr(self, name), (size,), name, positive=positive))
        if not np.isin(self.wuji_effort_sign, (-1.0, 1.0)).all():
            raise ValueError("wuji_effort_sign每轴只能为+1或-1")
        for name in ("gain", "ramp_duration_s", "jacobian_max_age_s", "jacobian_max_pose_delta_rad",
                     "feedback_max_age_s", "jacobian_refresh_interval_s", "jacobian_step_rad"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.gain > 1:
            raise ValueError("最大负载线性估计gain不能超过1")

    @classmethod
    def from_json(cls, path: str | Path) -> MaxTorqueLinearConfig:
        try:
            with Path(path).open(encoding="utf-8") as stream:
                data = json.load(stream)
        except (OSError, ValueError, TypeError) as error:
            raise CalibrationRequired("无法读取最大负载线性估计配置文件") from error
        if not isinstance(data, dict):
            raise CalibrationRequired("最大负载线性估计配置必须是JSON对象")
        try:
            return cls(**data)
        except TypeError as error:
            raise CalibrationRequired("最大负载线性估计配置字段缺失或未知") from error

    def validate_serial(self, serial: str) -> None:
        if not isinstance(serial, str) or serial != self.wuji_serial_number:
            raise CalibrationRequired("Wuji序列号与最大负载线性估计配置不一致")

    def load_fraction(self, effort_a: np.ndarray) -> np.ndarray:
        effort = _real_array(effort_a, (20,), "effort_a")
        try:
            # 有限超量程输入允许饱和；这不是Wuji Nm或接触力估计。
            with np.errstate(over="ignore", invalid="raise", divide="raise"):
                return np.clip((effort - self.wuji_current_bias_a) * self.wuji_effort_sign
                               / self.wuji_max_current_a, -1.0, 1.0)
        except FloatingPointError as error:
            raise ValueError("线性负载比例数值非有限") from error

    @property
    def glove_torque_limits_by_id(self) -> Mapping[int, float]:
        return MappingProxyType({index + 1: float(value)
                                 for index, value in enumerate(self.glove_max_output_torque_nm)})


@dataclass(frozen=True)
class LinearTorqueProjection:
    wuji_load_fraction: np.ndarray
    model_torque_nm: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "wuji_load_fraction",
                           _real_array(self.wuji_load_fraction, (20,), "wuji_load_fraction"))
        object.__setattr__(self, "model_torque_nm",
                           _real_array(self.model_torque_nm, (12,), "model_torque_nm"))


def project_max_torque_linear(
    jacobian_wuji_glove: np.ndarray,
    motor_jacobian: np.ndarray,
    effort_a: np.ndarray,
    config: MaxTorqueLinearConfig,
    model_torque_limits_nm: np.ndarray,
) -> LinearTorqueProjection:
    """按完整B=J_motor^-T J_retarget.T和每行L1包络反射负载占比。

    model_torque_limits_nm由驱动从raw允许上限换算到模型电机坐标，不能直接
    用raw配置数组代替。此纯函数不进行渐入、斜率限制或设备输出。
    """
    if not isinstance(config, MaxTorqueLinearConfig):
        raise ValueError("需要显式MaxTorqueLinearConfig")
    retarget = _real_array(jacobian_wuji_glove, (20, 12), "jacobian_wuji_glove")
    motor = _real_array(motor_jacobian, (12, 12), "motor_jacobian")
    limits = _real_array(model_torque_limits_nm, (12,), "model_torque_limits_nm", positive=True)
    if np.linalg.matrix_rank(motor) != 12:
        raise ValueError("CAD电机Jac奇异")
    try:
        load = config.load_fraction(effort_a)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            mapping = np.linalg.solve(motor.T, retarget.T)
            capacity = np.sum(np.abs(mapping), axis=1)
            reflected = mapping @ load
            normalized = np.zeros(12, dtype=np.float64)
            # 真正不可控的零行没有负载容量，保持零；不以epsilon吞掉小的非零行。
            np.divide(reflected, capacity, out=normalized, where=capacity != 0)
            torque = limits * np.clip(-config.gain * normalized, -1.0, 1.0)
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        raise ValueError("线性负载投影Jac奇异或数值非有限") from error
    if not np.isfinite(mapping).all() or not np.isfinite(capacity).all():
        raise ValueError("线性负载投影Jac数值非有限")
    return LinearTorqueProjection(load, torque)
