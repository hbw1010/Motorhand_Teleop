"""Wuji实测轨迹回放，以及可选的手套跟随和按键接管。"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty

import numpy as np

from ..bidirectional import ControlMode
from .control import RealWujiControlLoop
from .trajectory import RecordedTrajectory


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class PlaybackClock:
    """按原始轨迹时间推进；启动过渡和人工接管期间暂停。"""

    def __init__(self, *, duration_ns: int, transition_ns: int, start_ns: int,
                 speed: float = 1.0):
        if any(type(value) is not int or value < 0
               for value in (duration_ns, transition_ns, start_ns)) or duration_ns == 0:
            raise ValueError("回放时间必须是非负整数，轨迹时长必须为正")
        try:
            speed = float(speed)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("回放速度必须为大于0且不超过1的有限数") from exc
        if not math.isfinite(speed) or not 0 < speed <= 1:
            raise ValueError("回放速度必须为大于0且不超过1的有限数")
        self._speed_numerator, self._speed_denominator = speed.as_integer_ratio()
        self._remainder = 0
        self.duration_ns = duration_ns
        self.transition_ns = transition_ns
        self._last_ns = start_ns
        self._resume_ns = start_ns + transition_ns
        self._position_ns = 0
        self._playing = True

    @property
    def playing(self) -> bool:
        return self._playing

    def position(self, now_ns: int, *, ready: bool = True) -> int:
        if type(now_ns) is not int or now_ns < self._last_ns:
            raise ValueError("回放时钟倒退或时间类型非法")
        if self._playing and ready:
            elapsed_ns = max(0, now_ns - max(self._last_ns, self._resume_ns))
            advance, self._remainder = divmod(
                elapsed_ns * self._speed_numerator + self._remainder,
                self._speed_denominator,
            )
            self._position_ns = min(self.duration_ns, self._position_ns + advance)
        elif self._playing and self._last_ns < self._resume_ns:
            # 等待期间保留尚未完成的过渡时长。
            self._resume_ns += now_ns - self._last_ns
        self._last_ns = now_ns
        return self._position_ns

    def pause(self, now_ns: int) -> None:
        self.position(now_ns)
        self._playing = False

    def resume(self, now_ns: int) -> None:
        self.position(now_ns)
        self._playing = True
        self._resume_ns = now_ns + self.transition_ns


class TerminalKeys:
    """终端单键输入，保留Ctrl+C信号并在所有正常退出路径恢复终端设置。"""

    def __init__(self, fd: int):
        self.fd = fd
        self._saved = None

    def __enter__(self):
        if not os.isatty(self.fd):
            raise RuntimeError("手套接管需要交互终端，请在终端中启动回放")
        self._saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def read(self) -> tuple[str, ...]:
        if not select.select([self.fd], [], [], 0)[0]:
            return ()
        data = os.read(self.fd, 256)
        if not data:
            raise EOFError("键盘输入已关闭，停止回放")
        return tuple(
            "toggle" if key in (ord("t"), ord("T"), ord(" ")) else "quit"
            for key in data if key in (ord("t"), ord("T"), ord(" "), ord("q"), ord("Q"))
        )

    def __exit__(self, *_):
        if self._saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)
            self._saved = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="回放录制的Wuji实测轨迹，支持手套跟随和按键力反馈接管")
    parser.add_argument("trajectory", type=Path, help="录制的完整JSONL轨迹文件")
    parser.add_argument("--wuji-serial", help="可选SN；默认使用录制时的设备身份")
    parser.add_argument("--wuji-address", help="可选设备地址；默认自动发现")
    parser.add_argument("--approved-firmware", help="批准的真实固件版本；执行时必须与设备及录制身份一致")
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--glove-hand", choices=("left", "right"), default="right")
    parser.add_argument("--with-glove", action="store_true", help="回放中手套跟随Wuji实测姿态，可按T切换接管")
    parser.add_argument("--execute", action="store_true", help="明确允许使能设备并执行轨迹；默认只验证文件")
    parser.add_argument("--rate", type=float, default=50.0, help="目标控制频率Hz")
    parser.add_argument("--speed", type=float, default=1.0, help="轨迹回放速度倍率，范围(0, 1]")
    parser.add_argument("--transition", type=float, default=0.5, help="启动与切换的平滑过渡秒数")
    parser.add_argument("--mapping", type=Path, help="MotorHand标定文件")
    parser.add_argument("--assets", type=Path, help="MotorHand模型资产目录")
    parser.add_argument("--feedback-linear-config", type=Path,
                        default=_PROJECT_ROOT / "config/feedback/max-torque-linear-right.json")
    parser.add_argument("--contact-feedback-config", type=Path,
                        default=_PROJECT_ROOT / "config/feedback/contact-impedance-right.json")
    return parser


def _check_trajectory_limits(trajectory: RecordedTrajectory, driver) -> None:
    """在使能之前核对每个实测帧，回放时不静默裁剪轨迹。"""
    positions = trajectory.positions_rad
    lower, upper = driver.lower_limits, driver.upper_limits
    outside = np.argwhere((positions < lower) | (positions > upper))
    if outside.size:
        frame, axis = (int(value) for value in outside[0])
        raise ValueError(f"轨迹第{frame + 1}帧Wuji第{axis + 1}轴超出设备关节限位，拒绝使能")


def _format_status(tick, position_ns: int, duration_ns: int, bindings) -> str:
    if tick.force_feedback is None:
        waiting = "" if tick.inverse_fresh else "（等待新鲜手套逆解，轨迹暂停）"
        return f"回放 {position_ns / 1e9:.2f}/{duration_ns / 1e9:.2f}s{waiting}"
    feedback = tick.force_feedback
    if feedback.model_torque_nm is None:
        raise RuntimeError("回放接管要求名义Nm力反馈")
    axes = " ".join(
        f"J{mid}={feedback.model_torque_nm[index] * scale:+.3f}"
        for mid, (index, scale) in enumerate(bindings, 1)
    )
    return f"手套前馈[Nm,SDK已提交,{feedback.projection_status}/{feedback.contact_state}] {axes}"


def _run(args, trajectory: RecordedTrajectory, keys: TerminalKeys | None) -> int:
    from .motor_hand import EncosGloveHardwareDriver
    from .wuji_hand2 import WujiHand2HardwareDriver
    from .runtime import RetargetPipelineFactory, create_retargeter
    from .force_feedback import ForceFeedbackController
    from .force_linear import MaxTorqueLinearConfig
    from .contact_feedback import ContactFeedbackConfig

    driver = WujiHand2HardwareDriver(
        hand=args.hand, serial_number=args.wuji_serial,
        address=getattr(args, "wuji_address", "") or "",
        approved_firmware_version=args.approved_firmware,
    )
    glove = feedback = loop = worker = None
    enabled = False
    try:
        bridge = create_retargeter(hand=args.hand, glove_hand=args.glove_hand, asset_root=args.assets)
        bindings = None
        if args.with_glove:
            config = MaxTorqueLinearConfig.from_json(args.feedback_linear_config)
            config.validate_serial(args.wuji_serial)
            if config.glove_hand != args.glove_hand:
                raise ValueError("手套力反馈配置手侧与回放手套不一致")
            contact = ContactFeedbackConfig.from_json(args.contact_feedback_config)
            glove = EncosGloveHardwareDriver(
                hand=args.glove_hand, asset_root=args.assets, config_path=args.mapping,
                force_torque_limits_nm=config.glove_torque_limits_by_id,
                haptic_kp=45.0, haptic_kd=0.5,
            )
            feedback = ForceFeedbackController(
                config, bridge, glove,
                RetargetPipelineFactory(hand=args.hand, glove_hand=args.glove_hand,
                                        asset_root=args.assets),
                contact_config=contact,
            )
            from .curve_telemetry import _motor_axis_bindings
            bindings = _motor_axis_bindings(bridge.forward.landmarks.model, glove.motors)
            from .inverse_worker import InverseWorker
            worker = InverseWorker(
                args.hand, args.glove_hand, args.assets, None, glove.model_motor_limits_rad,
            )
        loop = RealWujiControlLoop(
            driver, bridge, glove_driver=glove, force_feedback=feedback,
            glove_follow_actual=args.with_glove,
            compute_shadow_glove=args.with_glove,
            startup_transition_s=args.transition,
            inverse_worker=worker,
        )
        print("正在只读连接并预检轨迹；尚未使能或回零。", flush=True)
        identity = loop.connect(serial_number=args.wuji_serial)
        if identity.hand != trajectory.hand or identity.firmware_version != trajectory.firmware_version:
            raise ValueError("设备手侧或实测固件与录制身份不一致，拒绝回放")
        _check_trajectory_limits(trajectory, driver)
        if worker is not None:
            state = driver.read_state()
            now_ns = time.monotonic_ns()
            if state.online_count != 20 or not 0 <= now_ns - state.received_ns < 100_000_000:
                raise TimeoutError("逆解预热前Hand2反馈不完整或已过期")
            print("只读启动独立逆解进程；预热结果不作为控制目标。", flush=True)
            worker.start(state.positions_rad, state.received_ns)
        if args.with_glove:
            print("即将伺服回零手套并使能两侧；回放中按 T 或空格接管/恢复，Q退出。", flush=True)
        else:
            print("即将使能Wuji回放；Ctrl+C停止。", flush=True)
        loop.enable()
        enabled = True
        print(f"启动平滑过渡 {args.transition:.2f} 秒；手套逆解未就绪时回放与过渡暂停。", flush=True)
        # 首个周期先从真实Wuji反馈启动平滑过渡；轨迹时间从该周期后起算。
        first = loop.step(inference_positions_rad=trajectory.positions_at(0))
        cursor = PlaybackClock(duration_ns=trajectory.duration_ns,
                               transition_ns=round(args.transition * 1e9),
                               start_ns=time.monotonic_ns(), speed=args.speed)
        print(_format_status(first, 0, trajectory.duration_ns, bindings), flush=True)
        inverse_ready = first.inverse_fresh
        next_report = time.monotonic_ns() + 1_000_000_000
        while True:
            cycle_ns = time.monotonic_ns()
            if keys is not None:
                events = keys.read()
                if "quit" in events:
                    print("收到退出按键，停止本次回放。", flush=True)
                    return 0
                if "toggle" in events:
                    if cursor.playing:
                        switch_ns = time.monotonic_ns()
                        cursor.position(switch_ns, ready=inverse_ready)
                        cursor.pause(switch_ns)
                        loop.request_mode(ControlMode.GLOVE_TAKEOVER, duration_s=args.transition)
                        print("已请求手套接管；回放暂停，手套启用力反馈。", flush=True)
                    else:
                        loop.request_mode(ControlMode.INFERENCE, duration_s=args.transition)
                        cursor.resume(time.monotonic_ns())
                        inverse_ready = False
                        print("已请求恢复回放；从暂停位置平滑衔接。", flush=True)
            position = cursor.position(time.monotonic_ns(), ready=inverse_ready)
            if cursor.playing:
                tick = loop.step(inference_positions_rad=trajectory.positions_at(position))
            else:
                tick = loop.step()
            inverse_ready = tick.inverse_fresh
            if cycle_ns >= next_report:
                print(_format_status(tick, position, trajectory.duration_ns, bindings), flush=True)
                next_report = cycle_ns + 1_000_000_000
            if cursor.playing and inverse_ready and position >= trajectory.duration_ns:
                print("轨迹回放完成。", flush=True)
                return 0
            remaining_ns = cycle_ns + round(1e9 / args.rate) - time.monotonic_ns()
            if remaining_ns > 0:
                time.sleep(remaining_ns / 1e9)
    finally:
        if loop is not None:
            try:
                if enabled:
                    loop.disable()
            finally:
                loop.close()
        else:
            try:
                driver.close()
            finally:
                try:
                    if glove is not None:
                        glove.close()
                finally:
                    try:
                        if feedback is not None:
                            feedback.close()
                    finally:
                        if worker is not None:
                            worker.close()


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not np.isfinite(args.rate) or args.rate <= 0:
        parser.error("--rate必须为正有限数")
    if not np.isfinite(args.transition) or args.transition <= 0:
        parser.error("--transition必须为正有限数")
    if not np.isfinite(args.speed) or not 0 < args.speed <= 1:
        parser.error("--speed必须为大于0且不超过1的有限数")
    trajectory = RecordedTrajectory.load(args.trajectory)
    if trajectory.hand != args.hand:
        parser.error("录制手侧与--hand不一致")
    if args.wuji_serial and trajectory.serial_number != args.wuji_serial:
        parser.error("录制序列号与--wuji-serial不一致")
    if args.approved_firmware and trajectory.firmware_version != args.approved_firmware:
        parser.error("录制固件与--approved-firmware不一致")
    print(f"轨迹已校验：{len(trajectory.positions_rad)}帧，{trajectory.duration_ns / 1e9:.3f}秒，手侧={trajectory.hand}。", flush=True)
    if not args.execute:
        print("仅检查文件；未连接、使能或控制设备。", flush=True)
        return 0
    if not args.approved_firmware:
        parser.error("--execute必须提供--approved-firmware")
    args.wuji_serial = args.wuji_serial or trajectory.serial_number
    if not args.wuji_address:
        from .preflight import select_device
        selected = select_device(serial_number=args.wuji_serial)
        args.wuji_address = selected["address"]
    try:
        if args.with_glove:
            with TerminalKeys(sys.stdin.fileno()) as keys:
                return _run(args, trajectory, keys)
        return _run(args, trajectory, None)
    except KeyboardInterrupt:
        print("已中断：正在释放本次回放的设备输出。", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
