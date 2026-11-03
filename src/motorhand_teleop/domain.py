from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

GLOVE_JOINTS = (
    "thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip",
    "index_mcp_flex", "index_pip",
    "middle_mcp_flex", "middle_pip",
    "ring_mcp_flex", "ring_pip",
    "pinky_mcp_flex", "pinky_pip",
)
GLOVE_DEPENDENT_JOINTS = ("index_dip", "middle_dip", "ring_dip", "pinky_dip")
WUJI_JOINTS = (
    "thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip",
    "index_mcp_flex", "index_mcp_abd", "index_pip", "index_dip",
    "middle_mcp_flex", "middle_mcp_abd", "middle_pip", "middle_dip",
    "ring_mcp_flex", "ring_mcp_abd", "ring_pip", "ring_dip",
    "pinky_mcp_flex", "pinky_mcp_abd", "pinky_pip", "pinky_dip",
)


def _finite_mapping(values: Mapping[str, float], names: tuple[str, ...], label: str) -> MappingProxyType:
    missing = [name for name in names if name not in values]
    extra = [name for name in values if name not in names]
    if missing or extra:
        raise ValueError(f"{label}名称不匹配: missing={missing}, extra={extra}")
    result = {name: float(values[name]) for name in names}
    if not np.all(np.isfinite(tuple(result.values()))):
        raise ValueError(f"{label}含非有限值")
    return MappingProxyType(result)


@dataclass(frozen=True)
class GlovePose:
    """手套运动学独立角，以及可选的闭链 dependent 角；单位 rad。"""

    joints: Mapping[str, float]
    dependent: Mapping[str, float] = field(default_factory=dict)
    dependent_jacobian: np.ndarray | None = None
    timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "joints", _finite_mapping(self.joints, GLOVE_JOINTS, "手套独立坐标"))
        if self.dependent:
            object.__setattr__(self, "dependent", _finite_mapping(self.dependent, GLOVE_DEPENDENT_JOINTS, "手套 dependent 坐标"))
            if self.dependent_jacobian is None:
                raise ValueError("提供 dependent 坐标时必须同时提供其对12个独立坐标的雅可比")
        else:
            object.__setattr__(self, "dependent", MappingProxyType({}))
            if self.dependent_jacobian is not None:
                raise ValueError("没有 dependent 坐标时不能提供 dependent_jacobian")
        if self.dependent_jacobian is not None:
            jacobian = np.asarray(self.dependent_jacobian, dtype=np.float64)
            if jacobian.shape != (4, 12) or not np.all(np.isfinite(jacobian)):
                raise ValueError("dependent_jacobian必须是有限的4x12数组")
            jacobian.setflags(write=False)
            object.__setattr__(self, "dependent_jacobian", jacobian)

    def vector(self) -> np.ndarray:
        return np.asarray([self.joints[name] for name in GLOVE_JOINTS], dtype=np.float64)


@dataclass(frozen=True)
class RetargetedPose:
    """WujiRetarget 输出的20维语义角与可选的局部雅可比。"""

    positions_rad: np.ndarray
    jacobian_wuji_glove: np.ndarray | None
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_rad, dtype=np.float64)
        if positions.shape != (20,) or not np.all(np.isfinite(positions)):
            raise ValueError("positions_rad必须是有限的20维数组")
        positions.setflags(write=False)
        object.__setattr__(self, "positions_rad", positions)
        if self.jacobian_wuji_glove is not None:
            jacobian = np.asarray(self.jacobian_wuji_glove, dtype=np.float64)
            if jacobian.shape != (20, 12) or not np.all(np.isfinite(jacobian)):
                raise ValueError("jacobian_wuji_glove必须是有限的20x12数组")
            jacobian.setflags(write=False)
            object.__setattr__(self, "jacobian_wuji_glove", jacobian)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@dataclass(frozen=True)
class WujiState:
    """已按项目20关节顺序归一化的真实状态。

    C ABI 6按NID重排同一joint_states帧的位置rad、速度rad/s和effort电流A，
    共用原始接收时间；速度/effort缺失或任一轴非有限时整项为None，不能伪零。
    effort是执行器输出电流，不是外部接触力。经单机偏置与Nm/A标定得到的
    执行器力矩估计仍可含重力、摩擦等内部负载，不能冒充独立力传感器测量。
    """

    positions_rad: np.ndarray
    velocities_rad_s: np.ndarray | None
    effort_a: np.ndarray | None
    received_ns: int
    online_count: int
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_rad, dtype=np.float64)
        if positions.shape != (20,) or not np.all(np.isfinite(positions)):
            raise ValueError("positions_rad必须是有限的20维数组")
        positions.setflags(write=False)
        object.__setattr__(self, "positions_rad", positions)
        for name in ("velocities_rad_s", "effort_a"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=np.float64)
                if array.shape != (20,) or not np.all(np.isfinite(array)):
                    raise ValueError(f"{name}必须是有限的20维数组或None")
                array.setflags(write=False)
                object.__setattr__(self, name, array)
        if not isinstance(self.received_ns, int) or self.received_ns < 0:
            raise ValueError("received_ns必须是非负整数")
        if not 0 <= self.online_count <= 20:
            raise ValueError("online_count必须在0到20之间")
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@dataclass(frozen=True)
class WujiTarget:
    positions_rad: np.ndarray
    velocities_rad_s: np.ndarray
    effort_a: np.ndarray

    def __post_init__(self) -> None:
        for name in ("positions_rad", "velocities_rad_s", "effort_a"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (20,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name}必须是有限的20维数组")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
