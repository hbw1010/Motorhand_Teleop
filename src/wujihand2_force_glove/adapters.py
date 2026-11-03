from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Protocol, runtime_checkable

from .domain import GlovePose, WujiState, WujiTarget

if TYPE_CHECKING:
    from .real_hand.motor_hand import EncosGloveState


@runtime_checkable
class GloveDriver(Protocol):
    """手套 SDK 适配器；connect 不得隐式使能或回零。

    位置始终是 GlovePose 的12个真实指节 rad，不是电机角。
    """

    def connect(self) -> None: ...
    def prepare(self) -> None: ...
    def read_pose(self) -> GlovePose: ...
    def read_state(self) -> EncosGloveState: ...
    @property
    def motor_names(self) -> tuple[str, ...]: ...
    @property
    def model_motor_torque_constants_nm_per_a(self) -> Mapping[str, float]:
        """完整模型motor名→正Kt；缺真实标定必须拒绝。"""
    @property
    def model_current_limits_a(self) -> Mapping[str, float]:
        """完整模型motor名→已按raw轴比例换算的电流上限A。"""
    @property
    def model_force_torque_limits_nm(self) -> Mapping[str, float]:
        """显式名义Nm模式的完整模型motor cap，已保留协议量化裕量；非实测力矩。"""
    @property
    def model_force_torque_command_limits_nm(self) -> Mapping[str, float]:
        """含零附近编码档的SDK参数确认上限；不同于减去裕量后的连续目标上限。"""
    def position_from_tree(
        self, positions: Mapping[str, float], *, timestamp_ns: int | None = None,
    ) -> GlovePose: ...
    def position_from_motors(
        self, positions: Mapping[str, float], *, timestamp_ns: int | None = None,
    ) -> GlovePose: ...
    def home(self) -> Mapping[int, float]: ...
    def enable(self) -> None: ...
    def enable_force_feedback(self) -> None:
        """独立取得纯前馈输出权；显式名义Nm或真实Kt模式，首次沿用servo回零。"""
    def constrain_position(self, pose: GlovePose) -> GlovePose: ...
    def position_tree(self, pose: GlovePose) -> Mapping[str, float]: ...
    def validate_command(
        self,
        *,
        target_pose: GlovePose | None = None,
        current_a: tuple[float, ...] | None = None,
    ) -> None:
        """只读预检与 send 相同的机构/标定/限流约束，不发送或提交状态。

        位置模式必须提供 target_pose；电流模式可省略以保持当前电机位置。
        """
    def send_position(self, pose: GlovePose, *, deadline_ns: int) -> None: ...
    def send_haptic_current(
        self,
        current_a: tuple[float, ...],
        *,
        deadline_ns: int,
        target_pose: GlovePose | None = None,
    ) -> None:
        """同一 MIT 帧发送位置和模型电机轴等效电流 A。

        current_a 按模型 motor_names/MOTOR_IDS 顺序，不是指节力矩；
        上游已处理传动雅可比。适配器只做一次轴坐标换算：
        m = sign * scale * theta_raw，I_raw = sign * scale * I_model，
        tau_raw = Kt * I_raw（Nm）；按 I_raw 限流，无 Kt 不得发送。
        target_pose=None 保持当前原始电机位置，不做 FK→IK 往返；
        否则同帧发送该真实指节目标的闭链逆解和前馈力矩。
        """
    def validate_force_current(self, current_a: tuple[float, ...]) -> None:
        """只读完整电流、标定、在线和真实反馈时戳预检，不使能。"""
    def send_force_current(
        self, current_a: tuple[float, ...], *, deadline_ns: int,
    ) -> tuple[float, ...]:
        """仅力控模式发送MIT kp=kd=0；返回软边界投影后的实际模型电流。

        不持位；实测模型电机边界处向外电流为零，向内允许。
        force模式禁止普通位置／位置加前馈发送，先disable再enable恢复位置。
        """
    def validate_force_torque(self, model_torque_nm: tuple[float, ...]) -> None:
        """只读完整名义Nm、标定、在线、真实时戳及CAD预检，不使能。"""
    def send_force_torque(
        self, model_torque_nm: tuple[float, ...], *, deadline_ns: int,
    ) -> tuple[float, ...]:
        """显式名义Nm模式专用，MIT kp=kd=0；raw tau=sign*scale*model tau。

        返回SDK实际量化前名义参数换算到模型轴，而非实测／解码力矩。
        raw边界向外投影至零附近向内量化档；无实际电流保证。
        """
    def disable(self) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class WujiDriver(Protocol):
    """WujiHand2 SDK 适配器；状态由 nid 归一化为项目 20 维顺序。"""

    def connect(self, *, serial_number: str | None = None) -> None: ...
    def read_state(self) -> WujiState: ...
    def send_target(self, target: WujiTarget, *, deadline_ns: int) -> None: ...
    def validate_enable(self) -> None: ...
    def enable(self) -> None: ...
    def disable(self) -> None: ...
    def emergency_stop(self) -> None: ...
    def close(self) -> None: ...
