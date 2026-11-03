"""真实 Wuji 状态驱动的双向控制周期和无扰切换。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping
import time

import numpy as np

from ..bidirectional import (
    BidirectionalRetargeter,
    BidirectionalStateBus,
    ControlMode,
    HandoverController,
)
from ..domain import GLOVE_JOINTS, GlovePose, WujiState, WujiTarget
from .force_feedback import ForceFeedbackCommand, ForceFeedbackController


@dataclass(frozen=True)
class ControlTick:
    """一次控制周期的真实反馈、候选目标和最终发送目标。"""

    mode: ControlMode
    hand2_state: WujiState
    candidate_positions_rad: np.ndarray
    sent_positions_rad: np.ndarray
    shadow_glove_positions_rad: np.ndarray | None
    timestamp_ns: int
    force_feedback: ForceFeedbackCommand | None = None

    def __post_init__(self) -> None:
        for name in ("candidate_positions_rad", "sent_positions_rad"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (20,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name}必须是有限的20维数组")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.shadow_glove_positions_rad is not None:
            value = np.asarray(self.shadow_glove_positions_rad, dtype=np.float64)
            if value.shape != (12,) or not np.all(np.isfinite(value)):
                raise ValueError("shadow_glove_positions_rad必须是有限的12维数组或None")
            value.setflags(write=False)
            object.__setattr__(self, "shadow_glove_positions_rad", value)


class RealWujiControlLoop:
    """把真实 Wuji 状态、双向求解和位置命令接成一个周期。

    推理主导时手套跟随最终平滑后的 Hand2 目标；接管期间手套位置始终
    由人手主导，反解只作影子诊断。底层看门狗、固件审批和急停由适配器负责。
    """

    def __init__(
        self,
        driver,
        retargeter: BidirectionalRetargeter,
        *,
        glove_driver=None,
        initial_mode: ControlMode = ControlMode.INFERENCE,
        command_timeout_ns: int = 100_000_000,
        force_feedback: ForceFeedbackController | None = None,
    ) -> None:
        if command_timeout_ns <= 0:
            raise ValueError("command_timeout_ns必须为正整数")
        self.driver = driver
        self.glove_driver = glove_driver
        self.retargeter = retargeter
        if force_feedback is not None and glove_driver is None:
            raise ValueError("真实力反馈接管需要MotorHand驱动")
        self.force_feedback = force_feedback
        self._force_output_active = False
        self._enabled = False
        if self.glove_driver is not None:
            limits = getattr(self.glove_driver, "model_motor_limits_rad", None)
            if limits is not None:
                self.retargeter.inverse.set_motor_limits(limits)
        self.state_bus = BidirectionalStateBus()
        self.handover = HandoverController(initial_mode)
        # 两骨架不精确互逆；手套以实测模型m12锚定内部轨迹，外部始终是真实q12。
        self._glove_handover = HandoverController(ControlMode.GLOVE_TAKEOVER)
        self.command_timeout_ns = int(command_timeout_ns)
        self._last_command = np.zeros(20, dtype=np.float64)
        self._has_command = False

    @property
    def mode(self) -> ControlMode:
        return self.handover.mode

    def connect(self, *, serial_number: str | None = None):
        if serial_number is None:
            identity = self.driver.connect()
        else:
            identity = self.driver.connect(serial_number=serial_number)
        if self.glove_driver is not None:
            try:
                if self.force_feedback is not None:
                    actual_serial = (
                        identity["serial_number"] if isinstance(identity, Mapping)
                        else identity.serial_number
                    )
                    self.force_feedback.config.validate_serial(actual_serial)
                self.glove_driver.connect()
                self.glove_driver.prepare()
            except BaseException:
                try:
                    self.glove_driver.close()
                finally:
                    self.driver.close()
                raise
        return identity

    def enable(self, *, now_ns: int | None = None) -> None:
        if now_ns is not None and int(now_ns) < 0:
            raise ValueError("启动时间必须非负")
        self.driver.validate_enable()
        try:
            if self.glove_driver is not None:
                if (self.handover.target_mode is ControlMode.GLOVE_TAKEOVER
                        and self.force_feedback is not None):
                    self._prepare_force_takeover(now_ns=now_ns)
                else:
                    self.glove_driver.home()
                    self.glove_driver.prepare()
                    if self.handover.target_mode is ControlMode.GLOVE_TAKEOVER:
                        self.glove_driver.disable()
            # 回零可能耗时数十秒；完成后才取得Wuji输出，避免看门狗先超时。
            self.driver.enable()
            self._enabled = True
            if (self.glove_driver is not None
                    and self.handover.target_mode is ControlMode.INFERENCE):
                state = self.glove_driver.read_state()
                ready_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
                self._start_glove_follow(
                    state.motor_positions_rad, now_ns=ready_ns, duration_s=0.5,
                )
        except BaseException:
            self.emergency_stop()
            raise

    def request_mode(
        self, mode: ControlMode, *, now_ns: int | None = None,
        duration_s: float = 0.5,
    ) -> None:
        try:
            self._request_mode(mode, now_ns=now_ns, duration_s=duration_s)
        except BaseException:
            if self.force_feedback is not None and self._enabled:
                self.emergency_stop()
            raise

    def _request_mode(
        self,
        mode: ControlMode,
        *,
        now_ns: int | None = None,
        duration_s: float = 0.5,
    ) -> None:
        """以两侧各自实际姿态接入；重复请求当前目标模式不重启过渡。"""
        if not isinstance(mode, ControlMode):
            raise ValueError("mode必须是ControlMode")
        if not np.isfinite(duration_s) or duration_s < 0:
            raise ValueError("切换时间参数非法")
        if now_ns is not None and int(now_ns) < 0:
            raise ValueError("切换时间参数非法")
        if mode is self.handover.target_mode:
            return
        if self.force_feedback is not None and not self._enabled:
            raise RuntimeError("力反馈接管必须先显式enable")
        if self._has_command:
            anchor = self._last_command
        else:
            anchor = self.driver.read_state().positions_rad
        glove_motors = None
        if self.glove_driver is not None:
            if mode is ControlMode.GLOVE_TAKEOVER:
                if self.force_feedback is None:
                    self.glove_driver.disable()
                else:
                    self._prepare_force_takeover(now_ns=now_ns)
                self._glove_handover = HandoverController(ControlMode.GLOVE_TAKEOVER)
            else:
                if self._force_output_active:
                    self.glove_driver.disable()
                    self._force_output_active = False
                self.glove_driver.enable()
                glove_motors = self.glove_driver.read_state().motor_positions_rad
        # enable/反馈准备可能耗时；两侧计时均从准备完毕开始，仿真不混入真实时钟。
        ready_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        self.handover.request(mode, anchor, now_ns=ready_ns, duration_s=duration_s)
        if glove_motors is not None:
            self._start_glove_follow(glove_motors, now_ns=ready_ns, duration_s=duration_s)

    def _prepare_force_takeover(self, *, now_ns: int | None) -> None:
        hand = self.driver.read_state()
        glove = self.glove_driver.read_state()
        now = time.monotonic_ns() if now_ns is None else int(now_ns)
        # 先拒绝缺实测effort/标定，再取得手套输出；初次home之后重新预热。
        self.force_feedback.prepare(
            hand, glove, now_ns=now,
        )
        self._force_output_active = True
        self.glove_driver.enable_force_feedback()
        hand = self.driver.read_state()
        glove = self.glove_driver.read_state()
        now = time.monotonic_ns() if now_ns is None else int(now_ns)
        self.force_feedback.prepare(
            hand, glove, now_ns=now,
        )
        now = time.monotonic_ns() if now_ns is None else int(now_ns)
        self.force_feedback.reset(now_ns=now)

    def _start_glove_follow(
        self, motors: Mapping[str, float], *, now_ns: int, duration_s: float,
    ) -> None:
        anchor = np.asarray([motors[name] for name in self.glove_driver.motor_names])
        self._glove_handover = HandoverController(ControlMode.GLOVE_TAKEOVER)
        self._glove_handover.request(
            ControlMode.INFERENCE, anchor, now_ns=now_ns, duration_s=duration_s,
        )

    def step(
        self,
        *,
        glove_joints: Mapping[str, float] | None = None,
        inference_positions_rad: np.ndarray | None = None,
        now_ns: int | None = None,
    ) -> ControlTick:
        """周期失败时释放已经取得的双侧力反馈输出。"""
        try:
            return self._step(
                glove_joints=glove_joints,
                inference_positions_rad=inference_positions_rad, now_ns=now_ns,
            )
        except BaseException:
            if self._force_output_active:
                self.emergency_stop()
            raise

    def _step(
        self,
        *,
        glove_joints: Mapping[str, float] | None = None,
        inference_positions_rad: np.ndarray | None = None,
        now_ns: int | None = None,
    ) -> ControlTick:
        """执行一个周期；now_ns 显式指定时使用调用方的确定性仿真时域。

        实时调用在昂贵求解后重查原始期限，不刷新输入时戳或续期。
        两设备 SDK 不提供跨设备事务；这里只预检本层可见的输入和求解结果。
        """
        now = time.monotonic_ns() if now_ns is None else int(now_ns)
        deadline_ns = now + self.command_timeout_ns
        hand2_state = self.driver.read_state()
        if now_ns is None:
            observed_ns = time.monotonic_ns()
            if hand2_state.online_count != 20:
                raise RuntimeError("Hand2反馈并非20轴全在线")
            deadline_ns = min(deadline_ns, hand2_state.received_ns + self.command_timeout_ns)
            if not 0 <= observed_ns - hand2_state.received_ns < self.command_timeout_ns:
                raise TimeoutError("Hand2反馈已过期或时钟域错误")
        self.state_bus.publish_hand2(
            _device_state(hand2_state),
            self.retargeter.hand2_model.state(
                hand2_state.positions_rad, timestamp_ns=hand2_state.received_ns
            ),
        )

        shadow_glove = None
        glove_pose = None
        glove_state = None
        if self.handover.target_mode is ControlMode.GLOVE_TAKEOVER:
            if self.force_feedback is not None and glove_joints is not None:
                raise ValueError("真实力反馈接管禁止用外部姿态覆盖手套实测反馈")
            if glove_joints is None:
                if self.glove_driver is None:
                    raise RuntimeError("GLOVE_TAKEOVER模式缺少手套驱动和手套姿态")
                glove_state = self.glove_driver.read_state()
                glove_pose = glove_state.pose
                glove_joints = glove_pose.joints
                if now_ns is None:
                    observed_ns = time.monotonic_ns()
                    stamp = glove_pose.timestamp_ns
                    if stamp is None or not 0 <= observed_ns - stamp < self.command_timeout_ns:
                        raise TimeoutError("MotorHand反馈已过期或时钟域错误")
                    deadline_ns = min(deadline_ns, stamp + self.command_timeout_ns)
            else:
                glove_pose = _make_glove_pose(glove_joints, timestamp_ns=now)
            pose, task = self.retargeter.glove_to_hand2(
                glove_joints, timestamp_ns=glove_pose.timestamp_ns or 0
            )
            candidate = pose.positions_rad
            if self.glove_driver is not None:
                self.state_bus.publish_glove(
                    _glove_device_state(glove_pose),
                    task,
                )
        else:
            if inference_positions_rad is None:
                raise RuntimeError("INFERENCE模式缺少推理Hand2目标")
            candidate = _positions(inference_positions_rad)

        sent = self.handover.output(candidate, now_ns=now)
        glove_target = None
        if self.handover.target_mode is ControlMode.INFERENCE:
            # 必须反解最终发送目标，而不是尚未经过无扰切换的 candidate。
            inverse, _ = self.retargeter.hand2_to_glove(
                sent, initial=None, timestamp_ns=now
            )
            if not inverse.success:
                raise ValueError("最终Hand2目标的手套逆解未收敛")
            shadow_glove = inverse.glove_positions_rad
            if self.glove_driver is not None:
                # 成功逆解已提供同次完整CAD树；禁止把q12再做一次昂贵的闭链逆解。
                goal_pose = self.glove_driver.position_from_tree(
                    inverse.tree_positions_rad, timestamp_ns=now,
                )
                goal_tree = self.glove_driver.position_tree(goal_pose)
                names = self.glove_driver.motor_names
                goal_motors = np.asarray([goal_tree[name] for name in names])
                motors = self._glove_handover.output(goal_motors, now_ns=now)
                glove_target = self.glove_driver.position_from_motors(
                    dict(zip(names, motors)), timestamp_ns=now,
                )
                shadow_glove = glove_target.vector()
        target = WujiTarget(
            positions_rad=sent,
            velocities_rad_s=np.zeros(20),
            effort_a=np.zeros(20),
        )
        self.driver.validate(target)
        feedback = None
        if (self.handover.target_mode is ControlMode.GLOVE_TAKEOVER
                and self.force_feedback is not None):
            feedback = self.force_feedback.compute(
                hand2_state, glove_state,
                now_ns=time.monotonic_ns() if now_ns is None else now,
                deadline_ns=deadline_ns,
                target_positions_rad=target.positions_rad,
            )
            if feedback.model_torque_nm is not None:
                self.glove_driver.validate_force_torque(feedback.model_torque_nm)
            else:
                self.glove_driver.validate_force_current(feedback.model_current_a)
            if not self._force_output_active:
                raise RuntimeError("手套力控未使能")
        elif self.glove_driver is not None and glove_target is not None:
            self.glove_driver.validate_command(target_pose=glove_target)
        # 所有本层准备/验证均在任一侧发送前完成。显式 now_ns 不与真实时钟混用。
        if (time.monotonic_ns() if now_ns is None else now) >= deadline_ns:
            raise TimeoutError("控制周期计算完成时命令已过期")
        self.driver.send_target(target, deadline_ns=deadline_ns)
        # 锚点记录实际已发往 Hand2 的命令，即使另一设备随后发送失败也不回退。
        self._last_command = sent.copy()
        self._has_command = True
        if self.glove_driver is not None:
            if feedback is not None:
                if feedback.model_torque_nm is not None:
                    actual = self.glove_driver.send_force_torque(
                        feedback.model_torque_nm, deadline_ns=deadline_ns,
                    )
                    self.force_feedback.acknowledge_torque(actual)
                    feedback = replace(feedback, model_torque_nm=actual)
                else:
                    actual = self.glove_driver.send_force_current(
                        feedback.model_current_a, deadline_ns=deadline_ns,
                    )
                    self.force_feedback.acknowledge(actual)
                    feedback = replace(feedback, model_current_a=actual)
            elif glove_target is not None:
                self.glove_driver.send_position(glove_target, deadline_ns=deadline_ns)
        return ControlTick(
            mode=self.mode,
            hand2_state=hand2_state,
            candidate_positions_rad=candidate,
            sent_positions_rad=sent,
            shadow_glove_positions_rad=shadow_glove,
            timestamp_ns=now,
            force_feedback=feedback,
        )

    def disable(self):
        self._force_output_active = False
        self._enabled = False
        try:
            return self.driver.disable()
        finally:
            if self.glove_driver is not None:
                self.glove_driver.disable()

    def emergency_stop(self):
        self._force_output_active = False
        self._enabled = False
        try:
            return self.driver.emergency_stop()
        finally:
            if self.glove_driver is not None:
                self.glove_driver.emergency_stop()

    def close(self) -> None:
        self._enabled = False
        try:
            self.driver.close()
        finally:
            try:
                if self.glove_driver is not None:
                    self.glove_driver.close()
            finally:
                self._force_output_active = False
                if self.force_feedback is not None:
                    self.force_feedback.close()


def _positions(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (20,) or not np.all(np.isfinite(result)):
        raise ValueError("Hand2目标必须是有限的20维数组")
    return result.copy()


def _device_state(state: WujiState):
    from ..bidirectional import DeviceState

    return DeviceState(
        timestamp_ns=state.received_ns,
        positions_rad=state.positions_rad,
        velocities_rad_s=state.velocities_rad_s,
        effort=state.effort_a,
    )


def _make_glove_pose(joints: Mapping[str, float], *, timestamp_ns: int) -> GlovePose:
    return GlovePose(
        joints={name: float(joints[name]) for name in GLOVE_JOINTS},
        timestamp_ns=timestamp_ns,
    )


def _glove_device_state(pose: GlovePose):
    from ..bidirectional import DeviceState

    return DeviceState(
        timestamp_ns=pose.timestamp_ns or 0,
        positions_rad=pose.vector(),
    )
