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
from .inverse_worker import InverseWorkerResult


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
    inverse_fresh: bool = True

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

    默认推理时手套跟随最终平滑后的 Hand2 目标；glove_follow_actual=True
    改为跟随本周期实测姿态。startup_transition_s 显式启用首帧实测锚定；
    无手套回放可用 compute_shadow_glove=False 跳过影子逆解。接管期间
    手套位置由人手主导。底层看门狗、固件审批和急停由适配器负责。
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
        glove_follow_actual: bool = False,
        startup_transition_s: float | None = None,
        compute_shadow_glove: bool = True,
        inverse_worker=None,
        forward_worker=None,
    ) -> None:
        if command_timeout_ns <= 0:
            raise ValueError("command_timeout_ns必须为正整数")
        if (startup_transition_s is not None
                and (not np.isfinite(startup_transition_s) or startup_transition_s < 0)):
            raise ValueError("回放启动过渡时间非法")
        if (glove_follow_actual or startup_transition_s is not None) and initial_mode is not ControlMode.INFERENCE:
            raise ValueError("回放必须从INFERENCE模式启动")
        if not compute_shadow_glove and glove_driver is not None:
            raise ValueError("有手套驱动时不能关闭手套逆解")
        if inverse_worker is not None and (glove_driver is None or not glove_follow_actual):
            raise ValueError("异步逆解仅用于手套跟随实测Hand2的回放")
        self.driver = driver
        self.glove_driver = glove_driver
        self.retargeter = retargeter
        if force_feedback is not None and glove_driver is None:
            raise ValueError("真实力反馈接管需要MotorHand驱动")
        self.force_feedback = force_feedback
        self._force_output_active = False
        self.glove_follow_actual = bool(glove_follow_actual)
        self.startup_transition_s = None if startup_transition_s is None else float(startup_transition_s)
        self.compute_shadow_glove = bool(compute_shadow_glove)
        self.inverse_worker = inverse_worker
        self.forward_worker = forward_worker
        self._inverse_result: InverseWorkerResult | None = None
        self._inverse_last_id = 0
        self._inverse_consumed_id = 0
        self._inverse_last_stamp = -1
        self._inverse_wait_started_ns: int | None = None
        self._inverse_last_cycle_ns: int | None = None
        self._inverse_was_ready = False
        self._inverse_needs_reanchor = False
        self._playback_start_pending = self.startup_transition_s is not None
        self._startup_blending = False
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
        self._last_deadline_ns: int | None = None

    def _queue_inverse(self, state: WujiState, *, force: bool = False) -> None:
        if self.inverse_worker is None:
            return
        if not force and state.received_ns <= self._inverse_last_stamp:
            return
        try:
            request_id = self.inverse_worker.submit(state.positions_rad, state.received_ns)
        except BlockingIOError:
            return
        self._inverse_last_id = request_id
        self._inverse_last_stamp = state.received_ns

    def _resolve_inverse(self, state: WujiState, *, now: int):
        """非阻塞使用最近的真实帧逆解，保留该帧的原始期限。"""
        result = self.inverse_worker.latest()
        # 新结果与仍在原始期限内的缓存都可以驱动本周期。
        if result is not None:
            if not result.success:
                raise ValueError(f"后台手套逆解失败：{result.error}")
            if result.received_ns > state.received_ns or result.received_ns > now:
                raise ValueError("后台手套逆解来源时戳异常")
            if (result.request_id > self._inverse_consumed_id
                    and now - result.received_ns < self.command_timeout_ns):
                self._inverse_result = result
                self._inverse_consumed_id = result.request_id
        self._queue_inverse(state)
        cached = self._inverse_result
        # 12轴命令需要逐轴提交；不在源期限只剩几毫秒时开始发送。
        margin_ns = min(30_000_000, self.command_timeout_ns // 2)
        usable = cached is not None and now - cached.received_ns < self.command_timeout_ns - margin_ns
        if not usable:
            self._inverse_needs_reanchor = True
            if self._inverse_wait_started_ns is None:
                self._inverse_wait_started_ns = now
            elif now - self._inverse_wait_started_ns >= 2_000_000_000:
                raise TimeoutError("等待新鲜手套逆解超过2秒，已停止回放")
            return None, False
        self._inverse_wait_started_ns = None
        return cached, True

    def _reset_inverse(self) -> None:
        self._inverse_result = None
        self._inverse_consumed_id = self._inverse_last_id
        self._inverse_last_stamp = -1
        self._inverse_wait_started_ns = None
        self._inverse_last_cycle_ns = None
        self._inverse_was_ready = False
        self._inverse_needs_reanchor = False

    @property
    def mode(self) -> ControlMode:
        return ControlMode.INFERENCE if self._startup_blending else self.handover.mode

    def connect(self, *, serial_number: str | None = None):
        try:
            if self.forward_worker is not None:
                self.forward_worker.start()
            if serial_number is None:
                identity = self.driver.connect()
            else:
                identity = self.driver.connect(serial_number=serial_number)
            if self.glove_driver is not None:
                if self.force_feedback is not None:
                    actual_serial = (
                        identity["serial_number"] if isinstance(identity, Mapping)
                        else identity.serial_number
                    )
                    self.force_feedback.config.validate_serial(actual_serial)
                self.glove_driver.connect()
                self.glove_driver.prepare()
                if self.forward_worker is not None:
                    glove = self.glove_driver.read_state()
                    observed = time.monotonic_ns()
                    stamps = (glove.received_ns, glove.pose.timestamp_ns)
                    if (tuple(sorted(glove.online_ids)) != tuple(range(1, 13))
                            or any(stamp is None or not 0 <= observed - stamp < self.command_timeout_ns
                                   for stamp in stamps)):
                        raise TimeoutError("正向求解预热前MotorHand反馈不完整或已过期")
                    self.forward_worker.solve(
                        glove.pose.joints, source_ns=glove.pose.timestamp_ns, deadline_ns=None,
                    )
            return identity
        except BaseException:
            try:
                if self.glove_driver is not None:
                    self.glove_driver.close()
            finally:
                try:
                    self.driver.close()
                finally:
                    if self.forward_worker is not None:
                        self.forward_worker.close()
            raise

    def enable(self, *, now_ns: int | None = None) -> None:
        if now_ns is not None and int(now_ns) < 0:
            raise ValueError("启动时间必须非负")
        self.driver.validate_enable()
        try:
            if self.glove_driver is not None:
                if self.force_feedback is not None:
                    hand = self.driver.read_state()
                    glove = self.glove_driver.read_state()
                    now = time.monotonic_ns() if now_ns is None else int(now_ns)
                    self.force_feedback.validate_inputs(hand, glove, now_ns=now)
                self.glove_driver.home()
                self.glove_driver.prepare()
                if self.force_feedback is not None:
                    hand = self.driver.read_state()
                    glove = self.glove_driver.read_state()
                    now = time.monotonic_ns() if now_ns is None else int(now_ns)
                    self.force_feedback.prepare(hand, glove, now_ns=now)
                if self.handover.target_mode is ControlMode.GLOVE_TAKEOVER:
                    if self.force_feedback is not None:
                        self._prepare_force_takeover(now_ns=now_ns)
                    else:
                        self.glove_driver.disable()
            if (self.glove_follow_actual and self.glove_driver is not None
                    and self.inverse_worker is None):
                # 首次逆解初始化不得占用Wuji看门狗租约；正式周期仍重新读状态、
                # 求解并检查原始反馈期限，不能把这里的结果或时戳当作新反馈。
                hand = self.driver.read_state()
                if hand.online_count != 20:
                    raise RuntimeError("Hand2反馈并非20轴全在线")
                if now_ns is None:
                    observed_ns = time.monotonic_ns()
                    if not 0 <= observed_ns - hand.received_ns < self.command_timeout_ns:
                        raise TimeoutError("Hand2反馈已过期或时钟域错误")
                inverse, _ = self.retargeter.hand2_to_glove(
                    hand.positions_rad, initial=None, timestamp_ns=hand.received_ns,
                )
                if not inverse.success:
                    raise ValueError(f"实测Hand2姿态的手套逆解预热未收敛：{getattr(inverse, 'failure_reason', None)}")
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
        if mode is ControlMode.GLOVE_TAKEOVER and self.forward_worker is not None:
            self.forward_worker.check_ready()
        if self.inverse_worker is not None and self._enabled and self._has_command:
            # 切换手套模式可能阻塞；先用新鲜Hand2状态刷新同一保持目标。
            state = self.driver.read_state()
            observed_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
            if state.online_count != 20 or not 0 <= observed_ns - state.received_ns < self.command_timeout_ns:
                raise TimeoutError("切换前Hand2反馈不完整或已过期")
            hold = WujiTarget(self._last_command.copy(), np.zeros(20), np.zeros(20))
            self.driver.validate(hold)
            hold_deadline = min(observed_ns, state.received_ns) + self.command_timeout_ns
            self.driver.send_target(hold, deadline_ns=hold_deadline)
            self._last_deadline_ns = hold_deadline
            self._reset_inverse()
            if mode is ControlMode.INFERENCE:
                self._queue_inverse(state, force=True)
        self._playback_start_pending = False
        self._startup_blending = False
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
        # 几何已在Wuji使能前预热。切换只取得零力输出权并重置历史；
        # 若几何已过期，compute会撤力并异步刷新，不在实时链路上等待。
        self._force_output_active = True
        self.glove_driver.enable_force_feedback()
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
        inference_source_ns: int | None = None,
        now_ns: int | None = None,
    ) -> ControlTick:
        """周期失败时释放已经取得的双侧力反馈输出。"""
        try:
            return self._step(
                glove_joints=glove_joints,
                inference_positions_rad=inference_positions_rad,
                inference_source_ns=inference_source_ns, now_ns=now_ns,
            )
        except BaseException:
            if (self._force_output_active or (
                self._enabled and (self.glove_follow_actual or self.startup_transition_s is not None)
            )):
                self.emergency_stop()
            raise

    def _step(
        self,
        *,
        glove_joints: Mapping[str, float] | None = None,
        inference_positions_rad: np.ndarray | None = None,
        inference_source_ns: int | None = None,
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
        if self._playback_start_pending:
            self.handover = HandoverController(ControlMode.GLOVE_TAKEOVER)
            self.handover.request(
                ControlMode.INFERENCE, hand2_state.positions_rad,
                now_ns=now, duration_s=self.startup_transition_s,
            )
            self._playback_start_pending = False
            self._startup_blending = self.handover.in_handover
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
            if self.forward_worker is None:
                pose, task = self.retargeter.glove_to_hand2(
                    glove_joints, timestamp_ns=glove_pose.timestamp_ns or 0
                )
            else:
                pose = self.forward_worker.solve(
                    glove_joints, source_ns=glove_pose.timestamp_ns or 0,
                    deadline_ns=deadline_ns if now_ns is None else None,
                )
                task = self.retargeter.hand2_model.state(
                    pose.positions_rad, timestamp_ns=glove_pose.timestamp_ns or 0,
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
            if inference_source_ns is not None:
                if (isinstance(inference_source_ns, (bool, np.bool_))
                        or not isinstance(inference_source_ns, (int, np.integer))
                        or inference_source_ns < 0):
                    raise ValueError("推理来源时间戳必须是非负整数")
                observed_ns = time.monotonic_ns() if now_ns is None else now
                if not 0 <= observed_ns - inference_source_ns < self.command_timeout_ns:
                    raise TimeoutError("推理目标已过期或时钟域错误")
                deadline_ns = min(deadline_ns, int(inference_source_ns) + self.command_timeout_ns)
            candidate = _positions(inference_positions_rad)

        inverse = None
        inverse_fresh = True
        glove_deadline_ns = deadline_ns
        async_follow = (
            self.inverse_worker is not None
            and self.handover.target_mode is ControlMode.INFERENCE
        )
        if async_follow:
            check_now = time.monotonic_ns() if now_ns is None else now
            inverse, inverse_fresh = self._resolve_inverse(hand2_state, now=check_now)
            if inverse_fresh and self._inverse_needs_reanchor:
                # 等待期间手套可能已被人手移动；重新采样电机位置再平滑接入。
                glove_state = self.glove_driver.read_state()
                reanchor_ns = time.monotonic_ns() if now_ns is None else now
                stamps = (glove_state.received_ns, glove_state.pose.timestamp_ns)
                if (tuple(sorted(glove_state.online_ids)) != tuple(range(1, 13))
                        or any(stamp is None or not 0 <= reanchor_ns - stamp < self.command_timeout_ns
                               for stamp in stamps)):
                    raise TimeoutError("恢复手套跟随时反馈不完整或已过期")
                glove_deadline_ns = min(
                    glove_deadline_ns, *(stamp + self.command_timeout_ns for stamp in stamps),
                )
                self._start_glove_follow(
                    glove_state.motor_positions_rad, now_ns=reanchor_ns, duration_s=0.5,
                )
                self._inverse_needs_reanchor = False
            if (not inverse_fresh or not self._inverse_was_ready) and self._inverse_last_cycle_ns is not None:
                self.handover.defer(max(0, now - self._inverse_last_cycle_ns))
            self._inverse_last_cycle_ns = now
            self._inverse_was_ready = inverse_fresh

        # 没有有效逆解时保持上一条已发目标；首帧使用真实反馈锚点。
        if async_follow and not inverse_fresh:
            sent = self._last_command.copy() if self._has_command else hand2_state.positions_rad.copy()
        else:
            sent = self.handover.output(candidate, now_ns=now)
            if self._startup_blending and not self.handover.in_handover:
                self._startup_blending = False
        glove_target = None
        if (self.handover.target_mode is ControlMode.INFERENCE
                and self.compute_shadow_glove):
            if not async_follow:
                # 普通推理跟随最终目标；回放跟随本周期实测姿态。
                inverse_source = hand2_state.positions_rad if self.glove_follow_actual else sent
                inverse, _ = self.retargeter.hand2_to_glove(
                    inverse_source, initial=None, timestamp_ns=now
                )
                if not inverse.success:
                    raise ValueError(f"最终Hand2目标的手套逆解未收敛：{getattr(inverse, 'failure_reason', None)}")
            if inverse is not None:
                shadow_glove = inverse.glove_positions_rad
                if self.glove_driver is not None:
                    # 后台结果仍使用它的真实来源期限，绝不能用新Hand2帧续期。
                    source_ns = (
                        self._inverse_result.received_ns if async_follow
                        else now
                    )
                    if async_follow:
                        glove_deadline_ns = min(glove_deadline_ns, source_ns + self.command_timeout_ns)
                    goal_pose = self.glove_driver.position_from_tree(
                        inverse.tree_positions_rad, timestamp_ns=source_ns,
                    )
                    goal_tree = self.glove_driver.position_tree(goal_pose)
                    names = self.glove_driver.motor_names
                    goal_motors = np.asarray([goal_tree[name] for name in names])
                    motors = self._glove_handover.output(goal_motors, now_ns=now)
                    glove_target = self.glove_driver.position_from_motors(
                        dict(zip(names, motors)), timestamp_ns=source_ns,
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
        # 两侧在发送前分别检查各自真实来源期限。
        check_ns = time.monotonic_ns() if now_ns is None else now
        if check_ns >= deadline_ns or (glove_target is not None and check_ns >= glove_deadline_ns):
            raise TimeoutError("控制周期计算完成时命令已过期")
        try:
            self.driver.send_target(target, deadline_ns=deadline_ns)
        except RuntimeError as error:
            if "原始命令期限已到" not in str(error):
                raise
            failed_ns = time.monotonic_ns() if now_ns is None else now
            previous = (
                "未知" if self._last_deadline_ns is None
                else f"{(self._last_deadline_ns - check_ns) / 1e6:.1f}ms"
            )
            glove_age = (
                "无" if glove_state is None
                else f"{(check_ns - glove_state.received_ns) / 1e6:.1f}ms"
            )
            raise RuntimeError(
                f"{error}；前一命令剩余={previous}；"
                f"当前命令剩余={(deadline_ns - check_ns) / 1e6:.1f}ms；"
                f"本周期计算={(check_ns - now) / 1e6:.1f}ms；"
                f"原生发送耗时={(failed_ns - check_ns) / 1e6:.1f}ms；"
                f"Wuji反馈年龄={(check_ns - hand2_state.received_ns) / 1e6:.1f}ms；"
                f"手套反馈年龄={glove_age}"
            ) from error
        self._last_deadline_ns = deadline_ns
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
                self.glove_driver.send_position(glove_target, deadline_ns=glove_deadline_ns)
        return ControlTick(
            mode=self.mode,
            hand2_state=hand2_state,
            candidate_positions_rad=candidate,
            sent_positions_rad=sent,
            shadow_glove_positions_rad=shadow_glove,
            timestamp_ns=now,
            force_feedback=feedback,
            inverse_fresh=inverse_fresh,
        )

    def disable(self):
        self._force_output_active = False
        self._has_command = False
        self._last_deadline_ns = None
        self._playback_start_pending = (
            self.startup_transition_s is not None
            and self.handover.target_mode is ControlMode.INFERENCE
        )
        self._startup_blending = False
        self._reset_inverse()
        self._enabled = False
        try:
            return self.driver.disable()
        finally:
            if self.glove_driver is not None:
                self.glove_driver.disable()

    def emergency_stop(self):
        self._force_output_active = False
        self._has_command = False
        self._last_deadline_ns = None
        self._playback_start_pending = (
            self.startup_transition_s is not None
            and self.handover.target_mode is ControlMode.INFERENCE
        )
        self._startup_blending = False
        self._reset_inverse()
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
                try:
                    if self.force_feedback is not None:
                        self.force_feedback.close()
                finally:
                    try:
                        if self.inverse_worker is not None:
                            self.inverse_worker.close()
                    finally:
                        if self.forward_worker is not None:
                            self.forward_worker.close()


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
