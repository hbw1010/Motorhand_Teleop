"""外部推理与人工力反馈接管的单主导会话。"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import time

from ..bidirectional import ControlMode
from .control import RealWujiControlLoop
from .inverse_worker import InverseWorker


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class InferenceSession:
    """复用真实控制循环；connect只读，start才会回零手套并使能。

    推理方按目标周期调用step；接管期间仍需持续调用step()驱动位置和反馈。
    返回的ControlTick区分实测Wuji状态、发送目标及名义力反馈，不能把名义力矩当实测值。
    不拥有时间调度线程；调用方必须保证及时更新或由底层看门狗停止输出。
    """

    def __init__(self, loop: RealWujiControlLoop, *, serial_number: str,
                 transition_s: float = 0.5) -> None:
        self._loop = loop
        self._serial_number = serial_number
        self._transition_s = transition_s
        self._connected = False
        self._started = False
        self._closed = False

    @property
    def mode(self) -> ControlMode:
        """返回当前请求的主导方；过渡期间输出仍从实测锚点平滑衔接。"""
        return self._loop.handover.target_mode

    def connect(self):
        """只读连接、预热逆解；不回零、不使能。"""
        if self._closed or self._connected:
            raise RuntimeError("会话已关闭或已连接")
        try:
            identity = self._loop.connect(serial_number=self._serial_number)
            worker = self._loop.inverse_worker
            if worker is not None:
                state = self._loop.driver.read_state()
                now = time.monotonic_ns()
                if state.online_count != 20 or not 0 <= now - state.received_ns < self._loop.command_timeout_ns:
                    raise TimeoutError("逆解预热前Wuji反馈不完整或已过期")
                worker.start(state.positions_rad, state.received_ns)
            self._connected = True
            return identity
        except BaseException:
            self.close()
            raise

    def read_wuji_state(self):
        """只读最新Wuji位置、速度、电流与源接收时刻。"""
        if not self._connected or self._closed:
            raise RuntimeError("会话尚未连接或已关闭")
        return self._loop.driver.read_state()

    def start(self) -> None:
        """显式获得输出权：伺服回零手套并使能两侧。"""
        if not self._connected or self._closed or self._started:
            raise RuntimeError("启动要求已连接且尚未启动的会话")
        self._loop.enable()
        self._started = True

    def step(self, positions_rad=None, *, produced_at_ns: int | None = None):
        """执行一个周期。

        推理主导时positions_rad为WUJI_JOINTS顺序的20维弧度目标；produced_at_ns
        可提供目标生成时的time.monotonic_ns()，过期目标不得发送。未提供时调用方
        须保证目标是当前周期实时生成的；本接口无法判断上游缓存的真实年龄。
        手套主导时不使用外部推理目标，调用step()即可。
        """
        if not self._started or self._closed:
            raise RuntimeError("会话尚未启动或已关闭")
        if self.mode is ControlMode.GLOVE_TAKEOVER:
            if positions_rad is not None or produced_at_ns is not None:
                self._reject_input("手套接管期间不接收推理目标")
            try:
                return self._loop.step()
            except BaseException:
                self._started = False
                raise
        if positions_rad is None:
            self._reject_input("推理主导期间需要20维关节目标")
        try:
            return self._loop.step(
                inference_positions_rad=positions_rad, inference_source_ns=produced_at_ns,
            )
        except BaseException:
            self._started = False
            raise

    def _reject_input(self, message: str) -> None:
        self._started = False
        self._loop.emergency_stop()
        raise ValueError(message)

    def take_over(self) -> None:
        """暂停推理主导，手套取得人手遥操作和力反馈输出权。"""
        self._request(ControlMode.GLOVE_TAKEOVER)

    def resume_inference(self) -> None:
        """从当前命令和手套实测姿态平滑切回模型主导。"""
        self._request(ControlMode.INFERENCE)

    def _request(self, mode: ControlMode) -> None:
        if not self._started or self._closed:
            raise RuntimeError("切换主导方要求已启动的会话")
        try:
            self._loop.request_mode(mode, duration_s=self._transition_s)
        except BaseException:
            self._started = False
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._started:
                self._loop.disable()
        finally:
            self._started = False
            self._loop.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()


def create_inference_session(
    *, serial_number: str | None = None, address: str | None = None,
    approved_firmware: str = "2.2.3",
    hand: str = "right", glove_hand: str = "right", transition_s: float = 0.5,
    mapping_path: str | Path | None = None, asset_root: str | Path | None = None,
    feedback_path: str | Path | None = None, contact_path: str | Path | None = None,
) -> InferenceSession:
    """发现Wuji并组装同一任务空间双向链路；未连接、未回零、未使能。"""
    from .contact_feedback import ContactFeedbackConfig
    from .force_feedback import ForceFeedbackController
    from .force_linear import MaxTorqueLinearConfig
    from .motor_hand import EncosGloveHardwareDriver
    from .preflight import select_device
    from .runtime import RetargetPipelineFactory, create_retargeter
    from .wuji_hand2 import WujiHand2HardwareDriver

    feedback_path = feedback_path or _PROJECT_ROOT / "config/feedback/max-torque-linear-right.json"
    contact_path = contact_path or _PROJECT_ROOT / "config/feedback/contact-impedance-right.json"
    config = MaxTorqueLinearConfig.from_json(feedback_path)
    contact = ContactFeedbackConfig.from_json(contact_path)
    if address:
        if not serial_number:
            raise ValueError("手动指定地址时必须同时指定Wuji序列号")
        selected = {"serial_number": serial_number, "address": address}
    else:
        selected = select_device(serial_number=serial_number)
    serial = selected["serial_number"]
    config.validate_serial(serial)
    if config.glove_hand != glove_hand:
        raise ValueError("力反馈配置与手套手侧不一致")

    with ExitStack() as resources:
        driver = WujiHand2HardwareDriver(
            hand=hand, serial_number=serial, address=selected["address"],
            approved_firmware_version=approved_firmware,
        )
        resources.callback(driver.close)
        glove = EncosGloveHardwareDriver(
            hand=glove_hand, asset_root=asset_root, config_path=mapping_path,
            force_torque_limits_nm=config.glove_torque_limits_by_id,
            haptic_kp=45.0, haptic_kd=0.5,
        )
        resources.callback(glove.close)
        bridge = create_retargeter(hand=hand, glove_hand=glove_hand, asset_root=asset_root)
        feedback = ForceFeedbackController(
            config, bridge, glove,
            RetargetPipelineFactory(hand=hand, glove_hand=glove_hand, asset_root=asset_root),
            contact_config=contact,
        )
        resources.callback(feedback.close)
        worker = InverseWorker(hand, glove_hand, asset_root, None, glove.model_motor_limits_rad)
        resources.callback(worker.close)
        loop = RealWujiControlLoop(
            driver, bridge, glove_driver=glove, force_feedback=feedback,
            glove_follow_actual=True, compute_shadow_glove=True,
            startup_transition_s=transition_s, inverse_worker=worker,
        )
        session = InferenceSession(loop, serial_number=serial, transition_s=transition_s)
        resources.pop_all()
        return session
