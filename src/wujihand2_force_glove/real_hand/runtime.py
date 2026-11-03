"""真实配对会话：位置主导权唯一，标定后接管可同时反射Wuji实测力矩。

这不是外部推理控制器的替代品；外部 q20 同时驱动两端仍使用
``real_hand.control.RealWujiControlLoop``。本模块的 hand2-to-glove 将 Wuji
当作只读输入，不向它发送 hold。没有 Kt/触觉标定时不生成任何前馈力矩。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
import json
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING, Callable

import numpy as np

from ..bidirectional import (
    BidirectionalRetargeter,
    GloveInverseSolver,
    GloveTaskModel,
    Hand2TaskModel,
    HandTaskState,
)
from ..domain import GlovePose, WujiState, WujiTarget
from ..landmarks import Ve33LandmarkSource
from ..retargeting import GloveWujiRetargetPipeline, WujiHand2Retargeter, default_retarget_config
from .force_feedback import (
    ForceFeedbackCalibration, ForceFeedbackCommand, ForceFeedbackController,
)
from .force_linear import MaxTorqueLinearConfig
from .contact_feedback import ContactFeedbackConfig

if TYPE_CHECKING:
    from .motor_hand import EncosGloveState


class FollowMode(str, Enum):
    GLOVE_TO_HAND2 = "glove-to-hand2"
    HAND2_TO_GLOVE = "hand2-to-glove"


@dataclass(frozen=True)
class PairTick:
    mode: FollowMode
    hand2_state: WujiState
    glove_pose: GlovePose
    candidate_positions_rad: np.ndarray
    output_positions_rad: np.ndarray
    output_task: HandTaskState
    deadline_ns: int
    executed: bool
    glove_state: EncosGloveState
    force_feedback: ForceFeedbackCommand | None = None
    cycle_started_ns: int | None = None
    cycle_completed_ns: int | None = None
    wuji_submit_ns: int | None = None
    glove_submit_ns: int | None = None


def create_retargeter(
    *, hand: str = "right", glove_hand: str = "right",
    project_root: str | Path | None = None,
    asset_root: str | Path | None = None, config_path: str | Path | None = None,
    estimate_jacobian: bool = False,
) -> BidirectionalRetargeter:
    """仅用本项目资产；两侧左右手独立配置，通过同一canonical任务空间配对。"""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[3]
    assets = Path(asset_root) if asset_root is not None else root / "assets" / "hand_motor_ve33"
    config = Path(config_path) if config_path is not None else default_retarget_config(root, hand)
    source = Ve33LandmarkSource(glove_hand, asset_root=assets)
    forward = GloveWujiRetargetPipeline(
        source, WujiHand2Retargeter.from_yaml(config, hand=hand),
        retargeter_factory=(
            (lambda: WujiHand2Retargeter.from_yaml(config, hand=hand))
            if estimate_jacobian else None
        ),
    )
    return BidirectionalRetargeter(
        forward,
        inverse=GloveInverseSolver(GloveTaskModel(forward.landmarks)),
        hand2_model=Hand2TaskModel(
            forward.retargeter, forward.landmarks.canonical_to_optimizer_row,
        ),
    )


@dataclass(frozen=True)
class RetargetPipelineFactory:
    """可spawn序列化的同配置求解工厂；不捕获SDK对象或主线程优化器。"""

    hand: str = "right"
    glove_hand: str = "right"
    project_root: Path | None = None
    asset_root: Path | None = None
    config_path: Path | None = None

    def __call__(self):
        return create_retargeter(
            hand=self.hand, glove_hand=self.glove_hand, project_root=self.project_root,
            asset_root=self.asset_root, config_path=self.config_path, estimate_jacobian=True,
        ).forward


def _identity_json(identity) -> dict[str, object]:
    """真实Wuji身份是dataclass，Motor身份是映射；输出前显式归一。"""
    return asdict(identity) if is_dataclass(identity) else dict(identity)


class PairRuntime:
    """位置主导端唯一的配对控制器；执行权必须由调用方显式授予。

    ``connect``/``inspect`` 永不使能。``start`` 或 ``request_mode`` 首次取得
    MotorHand 输出前执行安全 home，随后立即采集实测模型m12作为轨迹锚点。
    同一会话后续切换不重复回零。退出或异常仅 disable 本会话取得的输出，
    不停只读输入端其他进程的控制；原生 Wuji 看门狗由 native 层负责。
    """

    def __init__(
        self, driver, glove_driver, retargeter: BidirectionalRetargeter, *,
        execute: bool = False,
        mode: FollowMode | str = FollowMode.GLOVE_TO_HAND2,
        command_timeout_ns: int = 100_000_000,
        max_state_age_ns: int = 100_000_000,
        transition_s: float = 0.5,
        clock_ns: Callable[[], int] | None = None,
        force_feedback: ForceFeedbackController | None = None,
    ) -> None:
        if command_timeout_ns <= 0 or max_state_age_ns <= 0:
            raise ValueError("命令期限和反馈最大年龄必须为正数")
        if not np.isfinite(transition_s) or transition_s <= 0:
            raise ValueError("五次过渡时长必须为正有限数")
        self.driver = driver
        self.glove_driver = glove_driver
        self.retargeter = retargeter
        self.force_feedback = force_feedback
        self.execute = bool(execute)
        self.mode = FollowMode(mode)
        self.command_timeout_ns = int(command_timeout_ns)
        self.max_state_age_ns = int(max_state_age_ns)
        self.transition_ns = max(1, int(transition_s * 1e9))
        self._clock = clock_ns or time.monotonic_ns
        self._owned: FollowMode | None = None
        self._owns_force_glove = False
        self._motor_homed = False
        self._prepared: set[FollowMode] = set()
        self._running = False
        self._anchor: np.ndarray | None = None
        self._transition_start_ns: int | None = None
        self._closed = False
        self.identities: dict[str, object] = {}

    def connect(self, *, serial_number: str) -> dict[str, object]:
        if self._closed:
            raise RuntimeError("配对会话已经关闭")
        if not isinstance(serial_number, str) or not serial_number.strip():
            raise ValueError("连接必须指定 Wuji 序列号，禁止自动选择首个设备")
        self.identities["wuji_hand2"] = _identity_json(
            self.driver.connect(serial_number=serial_number)
        )
        try:
            if self.force_feedback is not None:
                self.force_feedback.config.validate_serial(
                    self.identities["wuji_hand2"]["serial_number"],
                )
            self.identities["motor_hand"] = _identity_json(self.glove_driver.connect())
            # 首次真实大姿态CAD FK可能慢于反馈年龄上限；只读预热丢弃该帧。
            self.glove_driver.prepare()
            self._bind_motor_limits()
        except BaseException:
            try:
                self.glove_driver.close()
            finally:
                self.driver.close()
            raise
        return dict(self.identities)

    def _check_stamp(self, stamp: int | None, now: int, label: str) -> None:
        if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp <= 0:
            raise ValueError(f"{label}缺少真实接收时刻")
        if stamp > now:
            raise ValueError(f"{label}接收时刻位于未来")
        if now - stamp > self.max_state_age_ns:
            raise TimeoutError(f"{label}真实反馈过期")

    def _check_samples(self, hand, glove, now: int) -> None:
        self._check_stamp(hand.received_ns, now, "Wuji")
        self._check_stamp(glove.received_ns, now, "MotorHand")
        self._check_stamp(glove.pose.timestamp_ns, now, "MotorHand姿态")
        if hand.online_count != 20:
            raise RuntimeError("Wuji 必须20轴全部在线")
        if tuple(sorted(glove.online_ids)) != tuple(range(1, 13)):
            raise RuntimeError("MotorHand 必须12轴全部在线")
        if hand.diagnostics.get("faults"):
            raise RuntimeError(f"Wuji 存在故障: {hand.diagnostics['faults']}")

    def _sample(self):
        glove = self.glove_driver.read_state()
        hand = self.driver.read_state()
        self._check_samples(hand, glove, self._clock())
        return hand, glove

    def inspect(self) -> dict[str, object]:
        """只读身份、真实20/12轴状态和映射标定状态，不获得输出权。"""
        hand, glove = self._sample()
        return {
            "身份": dict(self.identities),
            "MotorHand映射已标定": bool(self.glove_driver.mapping_calibrated),
            "MotorHand映射来源": self.glove_driver.mapping_source,
            "Wuji20语义关节rad": hand.positions_rad.tolist(),
            "Wuji接收时刻ns": hand.received_ns,
            "Wuji20实测速度rad_s": None if hand.velocities_rad_s is None else hand.velocities_rad_s.tolist(),
            "Wuji20实测电流A": None if hand.effort_a is None else hand.effort_a.tolist(),
            "MotorHand12真实指节rad": glove.pose.vector().tolist(),
            "MotorHand接收时刻ns": glove.received_ns,
            "MotorHand原始电机degree": dict(glove.raw_motor_positions_deg),
            "运动": "未使能、未回零、未发控制目标；MotorHand只读轮询含0A协议探针",
        }

    def _bind_motor_limits(self) -> None:
        # 既有原始电机软限位进入同CAD非线性逆解；未知capability不伪造bounds。
        self.retargeter.inverse.set_motor_limits(
            getattr(self.glove_driver, "model_motor_limits_rad", {}),
        )

    def prepare(self, mode: FollowMode | str | None = None) -> None:
        """未取得输出时只读预热真实求解器；预热目标绝不充作后续命令。

        首次CAD/官方求解可比实时周期长，但此时没有输出所有权。预热后
        重新读取真实帧；start/home之后和每个step仍须fresh采样重新求解。
        """
        if self._closed or self._running or self._owned is not None:
            raise RuntimeError("只读预热必须在未运行、未取得输出权时执行")
        self._bind_motor_limits()
        requested = self.mode if mode is None else FollowMode(mode)
        hand, glove = self._sample()
        if not self.glove_driver.mapping_calibrated:
            raise ValueError("MotorHand 映射未标定，禁止预热双向跟随")
        if requested is FollowMode.GLOVE_TO_HAND2:
            pose, _ = self.retargeter.glove_to_hand2(
                glove.pose.joints, timestamp_ns=glove.pose.timestamp_ns,
            )
            self._forward_success(pose)
            positions = self._constrain_hand(pose.positions_rad)
            self.retargeter.hand2_model.state(positions, timestamp_ns=hand.received_ns)
            self.driver.validate(WujiTarget(positions, np.zeros(20), np.zeros(20)))
        else:
            inverse, _ = self.retargeter.hand2_to_glove(
                hand.positions_rad, initial=glove.pose.vector(),
                timestamp_ns=hand.received_ns,
            )
            if not inverse.success:
                raise ValueError("MotorHand 只读预热逆解未收敛")
            actual = self.glove_driver.position_from_tree(
                inverse.tree_positions_rad, timestamp_ns=glove.pose.timestamp_ns,
            )
            self.glove_driver.validate_command(target_pose=actual)
        # 不把计算前帧的接收时刻续期，不把预热候选缓存为可发送命令。
        hand, glove = self._sample()
        if requested is FollowMode.GLOVE_TO_HAND2 and self.force_feedback is not None:
            self.force_feedback.prepare(hand, glove, now_ns=self._clock())
        self._prepared.add(requested)

    def start(self) -> None:
        if self._closed or self._running:
            raise RuntimeError("配对会话已关闭或已经运行")
        self._activate(self.mode)

    def request_mode(self, mode: FollowMode | str) -> None:
        """切换唯一输出源；新跟随端使用实测姿态而不是另一端旧目标锚定。"""
        if not self._running:
            raise RuntimeError("切换前必须 start")
        requested = FollowMode(mode)
        if requested is self.mode:
            return
        self._activate(requested)

    def _activate(self, mode: FollowMode) -> None:
        try:
            # 先检查双方真实反馈和标定，再释放旧输出/获得新输出。
            self._sample()
            if not self.glove_driver.mapping_calibrated:
                raise ValueError("MotorHand 映射未标定，禁止双向跟随")
            self._release_output()
            self._running = False
            if mode not in self._prepared:
                self.prepare(mode)
            if self.execute:
                # 在调用前登记所有权，部分使能/home失败也必须清理。
                self._owned = mode
                if mode is FollowMode.GLOVE_TO_HAND2:
                    if self.force_feedback is not None:
                        self._owns_force_glove = True
                        self.glove_driver.enable_force_feedback()
                        self._motor_homed = True
                        # 回零改变主端姿态；Wuji使能前重取实测树并预热新姿态Jac。
                        hand, glove = self._sample()
                        self.force_feedback.prepare(hand, glove, now_ns=self._clock())
                    self.driver.enable()
                elif not self._motor_homed:
                    self.glove_driver.home()
                    self._motor_homed = True
                else:
                    self.glove_driver.enable()
            # 尤其首次Motor home后必须重新采样，不能沿用回零前的姿态。
            hand, glove = self._sample()
            if mode is FollowMode.GLOVE_TO_HAND2 and self.force_feedback is not None:
                self.force_feedback.reset(now_ns=self._clock())
            self._anchor = (
                hand.positions_rad.copy() if mode is FollowMode.GLOVE_TO_HAND2
                else self._motor_vector(glove.motor_positions_rad)
            )
            self.mode = mode
            self._transition_start_ns = None
            self._running = True
        except BaseException:
            self._running = False
            self._release_output()
            raise

    def _motor_vector(self, positions) -> np.ndarray:
        values = np.asarray(
            [float(positions[name]) for name in self.glove_driver.motor_names], dtype=np.float64,
        )
        if values.shape != (12,) or not np.isfinite(values).all():
            raise ValueError("MotorHand 模型电机位置必须为有限12维rad")
        return values

    def _blend(self, candidate: np.ndarray, now: int) -> np.ndarray:
        if self._anchor is None:
            raise RuntimeError("缺少实测切换锚点")
        if self._transition_start_ns is None:
            self._transition_start_ns = now
        u = float(np.clip((now - self._transition_start_ns) / self.transition_ns, 0.0, 1.0))
        weight = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
        return self._anchor + weight * (candidate - self._anchor)

    def _forward_success(self, pose) -> None:
        # 官方NLopt退出码由WujiHand2Retargeter.step统一检查并提升异常。
        explicit = pose.diagnostics.get("success")
        if explicit is False:
            raise ValueError("Wuji 正向求解失败")

    def _constrain_hand(self, positions: np.ndarray) -> np.ndarray:
        values = np.asarray(positions, dtype=np.float64)
        if values.shape != (20,) or not np.all(np.isfinite(values)):
            raise ValueError("Wuji 目标必须为有限20维语义关节rad")
        limits = self.retargeter.forward.retargeter.joint_limits
        constrained = np.clip(values, limits[:, 0], limits[:, 1])
        constrained[[5, 9, 13, 17]] = 0.0
        return constrained

    def step(self) -> PairTick:
        """所有求解、限位/FK与预检在发送前完成，绝不刷新本周期原始deadline。"""
        if not self._running:
            raise RuntimeError("跟随前必须 start；异常后必须显式重新启动")
        started = self._clock()
        deadline = started + self.command_timeout_ns
        try:
            hand, glove = self._sample()
            if self.mode is FollowMode.GLOVE_TO_HAND2:
                pose, _ = self.retargeter.glove_to_hand2(
                    glove.pose.joints, timestamp_ns=glove.pose.timestamp_ns,
                )
                self._forward_success(pose)
                candidate = self._constrain_hand(pose.positions_rad)
                # 固定侧摆只约束求解候选，不覆盖反馈锚点。凸过渡由真实位置
                # 平滑回到零侧摆；最终硬件validate仍检查真实物理限位。
                output = self._blend(candidate, started)
                # 限位后的最终20维目标重新FK，不能发布未限位候选的任务状态。
                task = self.retargeter.hand2_model.state(output, timestamp_ns=started)
                target = WujiTarget(output, np.zeros(20), np.zeros(20))
                self.driver.validate(target)
            else:
                inverse, _ = self.retargeter.hand2_to_glove(
                    hand.positions_rad, initial=glove.pose.vector(),
                    timestamp_ns=hand.received_ns,
                )
                if not inverse.success:
                    raise ValueError("MotorHand 任务空间逆解未收敛")
                candidate = np.asarray(inverse.glove_positions_rad, dtype=np.float64)
                if candidate.shape != (12,) or not np.all(np.isfinite(candidate)):
                    raise ValueError("MotorHand 逆解必须为有限12真实指节rad")
                goal = self.glove_driver.position_from_tree(
                    inverse.tree_positions_rad, timestamp_ns=started,
                )
                goal_motors = self._motor_vector(self.glove_driver.position_tree(goal))
                # 硬件内部轨迹在m12模型电机轴上五次过渡，不对真实q12做线性混合。
                # 保持外部q12语义；最终实际姿态只由完整motor FK/软限位产生。
                motors = self._blend(goal_motors, started)
                target = self.glove_driver.position_from_motors(
                    dict(zip(self.glove_driver.motor_names, motors)), timestamp_ns=started,
                )
                output = target.vector()
                task = self.retargeter.inverse.glove_model.state_from_tree(
                    self.glove_driver.position_tree(target), timestamp_ns=started,
                )
                # 最终raw限位后姿态已重新FK；完整驱动预检也必须在发令前完成。
                self.glove_driver.validate_command(target_pose=target)
            feedback = None
            if self.mode is FollowMode.GLOVE_TO_HAND2 and self.force_feedback is not None:
                feedback = self.force_feedback.compute(
                    hand, glove, now_ns=self._clock(), deadline_ns=deadline,
                    target_positions_rad=target.positions_rad,
                )
                if feedback.model_torque_nm is not None:
                    self.glove_driver.validate_force_torque(feedback.model_torque_nm)
                else:
                    self.glove_driver.validate_force_current(feedback.model_current_a)
            finished = self._clock()
            self._check_samples(hand, glove, finished)
            if finished >= deadline:
                raise TimeoutError("控制周期求解/预检已超过原始命令期限，拒绝发送")
            wuji_submit_ns = glove_submit_ns = None
            if self.execute:
                if self._owned is not self.mode:
                    raise RuntimeError("当前跟随端没有本会话输出所有权")
                if feedback is not None and not self._owns_force_glove:
                    raise RuntimeError("手套力输出不属于本会话")
                if self.mode is FollowMode.GLOVE_TO_HAND2:
                    self.driver.send_target(target, deadline_ns=deadline)
                    wuji_submit_ns = self._clock()
                    if feedback is not None:
                        if feedback.model_torque_nm is not None:
                            actual = self.glove_driver.send_force_torque(
                                feedback.model_torque_nm, deadline_ns=deadline,
                            )
                            glove_submit_ns = self._clock()
                            self.force_feedback.acknowledge_torque(actual)
                            feedback = replace(feedback, model_torque_nm=actual)
                        else:
                            actual = self.glove_driver.send_force_current(
                                feedback.model_current_a, deadline_ns=deadline,
                            )
                            glove_submit_ns = self._clock()
                            self.force_feedback.acknowledge(actual)
                            feedback = replace(feedback, model_current_a=actual)
                else:
                    self.glove_driver.send_position(target, deadline_ns=deadline)
                    glove_submit_ns = self._clock()
            elif feedback is not None:
                # 只读数值预览也推进限斜率历史，不将其描述为已发/实测电流。
                if feedback.model_torque_nm is not None:
                    self.force_feedback.acknowledge_torque(feedback.model_torque_nm)
                else:
                    self.force_feedback.acknowledge(feedback.model_current_a)
            completed = self._clock()
            return PairTick(
                self.mode, hand, glove.pose, candidate.copy(), output.copy(), task,
                deadline, self.execute, glove, feedback, started, completed,
                wuji_submit_ns, glove_submit_ns,
            )
        except BaseException:
            self._running = False
            self._release_output()
            raise

    def _release_output(self) -> None:
        owned, self._owned = self._owned, None
        force_owned, self._owns_force_glove = self._owns_force_glove, False
        try:
            if owned is FollowMode.GLOVE_TO_HAND2:
                self.driver.disable()
            elif owned is FollowMode.HAND2_TO_GLOVE:
                self.glove_driver.disable()
        finally:
            # 任一侧停止失败仍尝试另一侧；从不停止未取得输出权的只读输入。
            if force_owned:
                self.glove_driver.disable()

    def stop(self) -> None:
        self._running = False
        self._release_output()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.stop()
        finally:
            try:
                self.driver.close()
            finally:
                try:
                    self.glove_driver.close()
                finally:
                    if self.force_feedback is not None:
                        self.force_feedback.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WujiHand2 / MotorHand 独立真机双向配对")
    commands = parser.add_subparsers(dest="command", required=True)
    discover = commands.add_parser("discover", help="只调用官方SDK发现Wuji身份，不连接任何设备/MotorHand")
    discover.add_argument("--timeout-ms", type=int, default=1000, help="官方SDK发现超时毫秒数")
    for name in ("inspect", "follow"):
        command = commands.add_parser(name, help="只读检查" if name == "inspect" else "单跟随端位置控制")
        command.add_argument("--hand", choices=("left", "right"), default="right", help="Wuji侧左右手")
        command.add_argument("--glove-hand", choices=("left", "right"), default="right", help="MotorHand侧CAD左右手，现场默认right")
        command.add_argument("--wuji-serial", help="显式设备SN；inspect未提供时只列discover")
        command.add_argument("--approved-firmware", help="用户明确批准的Wuji固件完整版本字符串")
        command.add_argument("--mapping", type=Path, help="MotorHand轴序/零位/符号/软限位标定JSON")
        command.add_argument("--assets", type=Path, help="本项目 hand_motor_ve33 资产目录")
        if name == "follow":
            command.add_argument("--mode", choices=[mode.value for mode in FollowMode], required=True)
            command.add_argument("--execute", action="store_true", help="允许本模式取得输出权的设备使能／输出；力反馈接管包括手套，默认双方只读")
            command.add_argument("--config", type=Path, help="本项目官方WujiRetarget YAML")
            command.add_argument("--rate", type=float, default=None, help="循环目标频率Hz：手套→Wuji默认50，Wuji→手套默认20")
            command.add_argument("--duration", type=float, default=0.0, help="秒；0表示运行到Ctrl-C")
            command.add_argument("--transition", type=float, default=0.5, help="五次平滑过渡秒数")
            command.add_argument("--curves-port", type=int, default=None,
                                 help="旁路发送曲线到本机UDP端口；不启动UI，不改变控制目标")
            command.add_argument("--contact-feedback-config", type=Path, default=None,
                                 help="受阻估计与软件虚拟弹簧阻尼参数；须同时提供名义Nm限额配置")
            feedback_options = command.add_mutually_exclusive_group()
            feedback_options.add_argument(
                "--feedback-calibration", type=Path,
                help="真实电流—力矩标定JSON；显式启用手套力控接管，不改变默认只读手套遥操",
            )
            feedback_options.add_argument(
                "--feedback-linear-config", type=Path,
                help="显式最大负载占比线性近似配置；直接名义Nm输出，不冒充真实Kt标定",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments or ["inspect"])
    if args.command == "discover":
        if args.timeout_ms <= 0:
            parser.error("--timeout-ms 必须为正整数")
        from .preflight import discover_devices
        print(json.dumps({
            "模式": "只读发现：未连接、未使能、不自动选择设备",
            "Wuji发现设备": discover_devices(timeout_ms=args.timeout_ms),
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "follow":
        if args.rate is None:
            args.rate = 50.0 if args.mode == FollowMode.GLOVE_TO_HAND2.value else 20.0
        if not args.wuji_serial:
            parser.error("follow 必须显式提供 --wuji-serial，禁止自动选择首个设备")
        if args.execute and not args.approved_firmware:
            parser.error("--execute 必须同时提供 --approved-firmware 完整批准版本")
        if not np.isfinite(args.rate) or args.rate <= 0:
            parser.error("--rate 必须为正有限数")
        if not np.isfinite(args.duration) or args.duration < 0:
            parser.error("--duration 必须为非负有限数")
        if not np.isfinite(args.transition) or args.transition <= 0:
            parser.error("--transition 必须为正有限数")
        if args.curves_port is not None and not 1 <= args.curves_port <= 65535:
            parser.error("--curves-port 必须为1至65535的整数")
        if ((args.feedback_calibration is not None or args.feedback_linear_config is not None)
                and args.mode != FollowMode.GLOVE_TO_HAND2.value):
            parser.error("力反馈配置仅用于 glove-to-hand2 人手接管")
        if args.contact_feedback_config is not None and args.feedback_linear_config is None:
            parser.error("--contact-feedback-config 必须同时指定 --feedback-linear-config")
    feedback_config = None
    contact_config = None
    if args.command == "follow":
        if args.feedback_calibration is not None:
            feedback_config = ForceFeedbackCalibration.from_json(args.feedback_calibration)
        elif args.feedback_linear_config is not None:
            feedback_config = MaxTorqueLinearConfig.from_json(args.feedback_linear_config)
        if feedback_config is not None:
            # 在构造/连接硬件之前完成参数、单机和手侧绑定；近似不冒充实测标定。
            feedback_config.validate_serial(args.wuji_serial)
            if feedback_config.glove_hand != args.glove_hand:
                parser.error("力反馈参数的glove_hand与--glove-hand不一致")
    if args.command == "follow" and args.contact_feedback_config is not None:
        contact_config = ContactFeedbackConfig.from_json(args.contact_feedback_config)
    from .motor_hand import EncosGloveHardwareDriver
    from .wuji_hand2 import WujiHand2HardwareDriver

    glove = EncosGloveHardwareDriver(
        hand=args.glove_hand, asset_root=args.assets, config_path=args.mapping,
        haptic_kp=45.0, haptic_kd=0.5,
        torque_constant_nm_per_a=(
            feedback_config.glove_torque_constants_by_id
            if isinstance(feedback_config, ForceFeedbackCalibration) else None
        ),
        force_torque_limits_nm=(
            feedback_config.glove_torque_limits_by_id
            if isinstance(feedback_config, MaxTorqueLinearConfig) else None
        ),
    )
    driver = None
    runtime = None
    curve_publisher = None
    try:
        if args.command == "inspect":
            from .preflight import discover_devices
            discovered = discover_devices()
            report: dict[str, object] = {
                "Wuji发现设备": discovered,
                "Wuji选择": args.wuji_serial or "未指定SN，仅列发现结果，不连接首个设备",
            }
            if args.wuji_serial:
                driver = WujiHand2HardwareDriver(
                    hand=args.hand, serial_number=args.wuji_serial,
                    approved_firmware_version=args.approved_firmware,
                )
                report["Wuji身份"] = _identity_json(driver.connect(serial_number=args.wuji_serial))
            report["MotorHand身份"] = _identity_json(glove.connect())
            glove.prepare()
            state = glove.read_state()
            now = time.monotonic_ns()
            for stamp in (state.received_ns, state.pose.timestamp_ns):
                if stamp is None or stamp <= 0 or stamp > now or now - stamp > 100_000_000:
                    raise TimeoutError("MotorHand真实反馈过期或接收时刻非法")
            if driver is not None:
                hand = driver.read_state()
                now = time.monotonic_ns()
                if hand.received_ns <= 0 or hand.received_ns > now or now - hand.received_ns > 100_000_000:
                    raise TimeoutError("Wuji真实反馈过期或接收时刻非法")
                report["Wuji20语义关节rad"] = hand.positions_rad.tolist()
                report["Wuji在线轴数"] = hand.online_count
            report["MotorHand12真实指节rad"] = state.pose.vector().tolist()
            report["MotorHand原始电机degree"] = dict(state.raw_motor_positions_deg)
            report["MotorHand在线轴"] = state.online_ids
            report["MotorHand接收时刻ns"] = state.received_ns
            report["MotorHand映射已标定"] = glove.mapping_calibrated
            report["MotorHand映射来源"] = glove.mapping_source
            report["模式"] = "只读：不使能、不回零、不发控制目标；MotorHand轮询含0A协议探针"
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        driver = WujiHand2HardwareDriver(
            hand=args.hand, serial_number=args.wuji_serial,
            approved_firmware_version=args.approved_firmware,
        )
        bridge = create_retargeter(
            hand=args.hand, glove_hand=args.glove_hand,
            asset_root=args.assets, config_path=args.config,
        )
        if args.curves_port is not None:
            from .curve_telemetry import CurvePublisher
            curve_publisher = CurvePublisher(
                args.curves_port, glove_model=bridge.forward.landmarks.model,
                motors=glove.motors, wuji_joint_names=bridge.forward.retargeter.output_joint_names,
            )
        feedback = None
        if feedback_config is not None:
            feedback = ForceFeedbackController(
                feedback_config, bridge, glove,
                RetargetPipelineFactory(
                    hand=args.hand, glove_hand=args.glove_hand,
                    asset_root=args.assets, config_path=args.config,
                ),
                contact_config=contact_config,
            )
        runtime = PairRuntime(
            driver, glove, bridge, execute=args.execute, mode=args.mode,
            transition_s=args.transition, force_feedback=feedback,
        )
        runtime.connect(serial_number=args.wuji_serial)
        print(json.dumps(runtime.inspect(), ensure_ascii=False, indent=2))
        print(
            f"模式={args.mode}；" + (
                "已明确 --execute，Wuji位置跟随＋手套力控反射。"
                if args.execute and feedback is not None else
                "已明确 --execute，只有位置跟随侧输出。" if args.execute else
                "未提供 --execute：双方只读，仅计算，不使能/回零/发送。"
            )
        )
        if args.mode == FollowMode.GLOVE_TO_HAND2.value:
            if feedback is None:
                print("控制量：Wuji20语义关节rad位置；MotorHand只读，绝不拉人手。")
            else:
                if feedback.output_is_torque:
                    if feedback.contact_config is not None:
                        print("受阻估计：最终位置跟踪误差＋实测电流＋速度持续判据→单向软件弹簧阻尼→既有几何传动→手套名义Nm；不是独立接触力测量。")
                        print("软件虚拟K/D与SDK参数独立；SDK仍kp=kd=0。阈值与虚拟刚阻尼为初始估计，不是实物标定。")
                    else:
                        print("近似反馈：Wuji实测A/满量程A→完整几何投影→手套允许的名义Nm；不计算假Wuji Nm，不是净接触力。")
                    print("手套名义输出上限raw Nm=" + str(feedback_config.glove_max_output_torque_nm.tolist()))
                    print("输入归一化不修改Wuji电流限值；两侧kp/kd独立，手套接管MIT0/0，保留协议量化裕量。")
                else:
                    print("控制量：手套实测姿态→Wuji位置；Wuji实测电流→标定执行器反作用力矩→手套MIT纯力矩(kp0/kd0)，不追逆解位置。")
                if args.execute:
                    print("首次手套输出先servo回零：5rpm，J1=1.5A/其余1A，容差0.15rad；随后连续名义目标从0渐入（零附近有协议量化偏差）。")
                print("姿态超出局部Jac范围时只暂停力反射并后台刷新，位置跟随继续；恢复时重新渐入，终端反馈状态非active时不代表力反馈生效。")
        else:
            print("控制量：共享任务逆解→模型m12五次过渡→CAD FK→MIT位置(kp45/kd0.5)；外部仍为真实指节q12 rad，无力矩前馈；Wuji只读，不发hold。")
            if args.execute:
                print("首次MotorHand输出将安全回零：servo5rpm，J1=1.5A/其余1A，容差0.15rad。")
        print("只读预热真实任务求解器；预热不使能/不回零/不发送，实时周期仍使用新反馈与原始100ms期限。")
        runtime.prepare()
        runtime.start()
        began = time.monotonic()
        next_report = began
        while args.duration == 0 or time.monotonic() - began < args.duration:
            cycle = time.monotonic()
            tick = runtime.step()
            if curve_publisher is not None:
                curve_publisher.publish(tick)
            if cycle >= next_report:
                print(json.dumps({
                    "模式": tick.mode.value, "真实执行": tick.executed,
                    "跟随端目标rad": tick.output_positions_rad.tolist(),
                    "原始命令期限ns": tick.deadline_ns,
                    **({} if tick.force_feedback is None else {
                        "反馈模式": tick.force_feedback.mapping_mode,
                        "反馈状态": tick.force_feedback.projection_status,
                        "目标已发送": tick.executed,
                        "反馈参数来源": tick.force_feedback.source_id,
                        "反馈Jac年龄ns": tick.force_feedback.jacobian_age_ns,
                        "受阻估计状态": tick.force_feedback.contact_state,
                        **({} if tick.force_feedback.contact_active_mask is None else {
                            "估计受阻Wuji语义轴_1至20": (np.flatnonzero(tick.force_feedback.contact_active_mask) + 1).tolist(),
                            "软件虚拟手关节反作用Nm_非实测": tick.force_feedback.virtual_hand_torque_nm.tolist(),
                        }),
                        **({
                            "手套名义目标Nm": tick.force_feedback.model_torque_nm,
                            "Wuji电流负载占比": tick.force_feedback.wuji_load_fraction.tolist(),
                        } if tick.force_feedback.model_torque_nm is not None else {
                            "力反馈电流A": tick.force_feedback.model_current_a,
                        }),
                    }),
                }, ensure_ascii=False))
                next_report = cycle + 1.0
            time.sleep(max(0.0, 1.0 / args.rate - (time.monotonic() - cycle)))
        return 0
    except KeyboardInterrupt:
        print("已中断：释放本会话输出；只读输入端不失能。")
        return 130
    finally:
        try:
            if runtime is not None:
                runtime.close()
            else:
                try:
                    if driver is not None:
                        driver.close()
                finally:
                    glove.close()
        finally:
            if curve_publisher is not None:
                curve_publisher.close()
