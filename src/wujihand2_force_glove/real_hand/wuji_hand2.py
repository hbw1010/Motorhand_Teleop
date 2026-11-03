"""独立WujiHand2真机边界：20关节rad，显式固件授权，默认全程只读。

构造、connect、inspect和未授权close均不发运动/失能/清错命令。
真实速度rad/s与effort电流A来自同一joint_states帧；不可用时保持None，不伪造反馈。
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import threading
from typing import Any

import numpy as np

from ..domain import WUJI_JOINTS, WujiState, WujiTarget
from ._wuji_hand2_native import NativeWujiHand2Driver, SafetyOptions, check_deadline
from .preflight import SDK_VERSION


_NATIVE_JOINT_NAMES = tuple(
    f"{finger}_S{joint}"
    for finger in ("thumb", "index", "middle", "ring", "pinky")
    for joint in range(1, 5)
)
# 与固定2026.8.31运输层的float32限位完全一致；连接后再次读取实际原生边界。
_LOWER = np.array([-1.187, -1.484, -1.047, -1.047] + [-1.047, -0.698, -1.047, -1.047] * 4, dtype=np.float32).astype(np.float64)
_UPPER = np.array([1.291, 0.698, 1.570, 1.570] + [1.570, 0.698, 2.094, 1.570] * 4, dtype=np.float32).astype(np.float64)


class WujiHardwareUnavailable(RuntimeError):
    """本项目WujiHand2原生库或固定官方SDK不可用。"""


@dataclass(frozen=True)
class WujiDeviceIdentity:
    model: str
    hand: str
    serial_number: str
    firmware_version: str
    address: str = ""
    sdk_version: str = SDK_VERSION


@dataclass(frozen=True)
class StopResult:
    requested: bool
    confirmed: bool | None
    method: str


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _ordered_positions(values, *, optional=False):
    if values is None and optional:
        return None
    if isinstance(values, Mapping):
        if set(values) == set(WUJI_JOINTS):
            values = [values[name] for name in WUJI_JOINTS]
        elif set(values) == set(_NATIVE_JOINT_NAMES):
            values = [values[name] for name in _NATIVE_JOINT_NAMES]
        else:
            if optional:
                return None
            raise ValueError("Wuji反馈必须包含完整20个语义关节")
    result = np.array(values, dtype=np.float64, copy=True)
    if result.shape != (20,) or not np.all(np.isfinite(result)):
        if optional:
            return None
        raise ValueError("Wuji反馈必须是有限的20维位置")
    return result


class WujiHand2HardwareDriver:
    """公开生命周期适配；backend_factory仅用于显式注入真实接口或离线测试。

    注入backend.connect返回含model/hand/serial_number/firmware_version的身份；
    read_state返回WujiState或含positions_rad/received_ns/online/diagnostics的真实状态。
    backend.send接受20维rad；明确提供send_target的backend可接收完整目标。
    enable失败也视为已尝试接管，必须执行停止收尾；只读连接不得停止他人。
    """

    model = "wuji_hand2"
    joint_names = WUJI_JOINTS

    def __init__(self, *, hand: str = "right", serial_number: str,
                 approved_firmware_version: str | None = None, address: str = "",
                 timeout_ms: int = 1000, backend_factory: Callable[[], Any] | None = None,
                 native_safety: SafetyOptions | Mapping | None = None,
                 effort_limit=None, mit_kp=5.0, mit_kd=0.1):
        if isinstance(native_safety, Mapping):
            native_safety = SafetyOptions(**native_safety)
        self.native_safety = native_safety if native_safety is not None else SafetyOptions()
        approved = "" if approved_firmware_version is None else approved_firmware_version
        # 复用薄绑定的输入校验；构造本身不加载库、不联网、不创建原生句柄。
        native = NativeWujiHand2Driver(
            model=self.model, hand=hand, serial_number=serial_number, address=address,
            approved_firmware_version=approved, timeout_ms=timeout_ms,
            safety=self.native_safety, effort_limit=effort_limit,
            mit_kp=mit_kp, mit_kd=mit_kd, auto_home=False,
        )
        if backend_factory is not None and not callable(backend_factory):
            raise TypeError("backend_factory必须可调用或为None")
        self.hand, self.serial_number = hand, serial_number
        self._address, self._approved = address, approved
        self._factory = (lambda: native) if backend_factory is None else backend_factory
        self._backend = None
        self._identity = None
        self._lock = threading.RLock()
        self._closed = self._connected = self._active = self._owned = False
        self._failure = None
        self._lower, self._upper = _LOWER.copy(), _UPPER.copy()

    @property
    def active(self):
        return self._active and self.connected and self._failure is None

    @property
    def connected(self):
        return self._connected and not self._closed

    @property
    def closed(self):
        return self._closed

    @property
    def lower_limits(self):
        result = self._lower.copy()
        result.setflags(write=False)
        return result

    @property
    def upper_limits(self):
        result = self._upper.copy()
        result.setflags(write=False)
        return result

    def _require_connected(self):
        if not self.connected or self._backend is None:
            raise RuntimeError("WujiHand2未连接或已关闭")
        if self._failure is not None:
            raise RuntimeError("WujiHand2故障已锁存，必须关闭后重新显式授权") from self._failure

    def connect(self, *, serial_number=None):
        with self._lock:
            if self._closed or self._connected:
                raise RuntimeError("WujiHand2已连接或已关闭，不能重复连接")
            if serial_number is not None and serial_number != self.serial_number:
                raise ValueError("serial_number与构造身份不一致")
            backend = self._factory()
            try:
                identity = backend.connect()
                serial = _field(identity, "serial_number", _field(identity, "identifier"))
                model, hand = _field(identity, "model"), _field(identity, "hand")
                firmware = _field(identity, "firmware_version")
                if (serial, model, hand) != (self.serial_number, self.model, self.hand):
                    raise ValueError("WujiHand2实际型号、手侧或SN与显式配置不一致")
                if not isinstance(firmware, str) or not firmware.strip() or "\x00" in firmware:
                    raise ValueError("WujiHand2连接必须返回真实非空固件版本")
                lower = _ordered_positions(getattr(backend, "lower_limits", self._lower))
                upper = _ordered_positions(getattr(backend, "upper_limits", self._upper))
                if np.any(lower >= upper) or np.any(lower > 0) or np.any(upper < 0):
                    raise ValueError("WujiHand2关节限位无效或不含零位")
                self._lower, self._upper = lower, upper
                self._identity = WujiDeviceIdentity(model, hand, serial, firmware, self._address)
            except BaseException as error:
                self._closed = True
                try:
                    backend.close()
                except BaseException as cleanup:
                    raise BaseExceptionGroup("只读连接失败且释放失败", [error, cleanup]) from None
                raise
            self._backend, self._connected = backend, True
            return self._identity

    def _fail(self, error):
        self._failure, self._active = error, False
        if self._owned and self._backend is not None:
            try:
                self._backend.emergency_stop()
            except BaseException as cleanup:
                raise BaseExceptionGroup("WujiHand2操作失败且停止未确认", [error, cleanup]) from None
        raise error

    def validate_enable(self) -> None:
        """仅预检当前会话身份与固件批准，不取得输出权、不访问SDK。"""
        with self._lock:
            self._require_connected()
            if not self._approved or self._identity.firmware_version != self._approved:
                raise ValueError("WujiHand2使能要求实测固件与显式批准版本完全相等")

    def enable(self):
        with self._lock:
            self.validate_enable()
            if self.active:
                return
            self._owned = True
            try:
                self._backend.enable()
                self._active = True
            except BaseException as error:
                self._fail(error)

    def validate_target(self, target: WujiTarget, deadline_ns: int):
        """纯只读目标预检；不发送、停机、读硬件或改变故障/授权状态。"""
        with self._lock:
            self._require_connected()
            self._validate(target)
            check_deadline(deadline_ns)

    def _validate(self, target):
        if not isinstance(target, WujiTarget):
            raise TypeError("target必须为WujiTarget")
        for name in ("positions_rad", "velocities_rad_s", "effort_a"):
            values = np.asarray(getattr(target, name))
            if values.shape != (20,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name}必须为有限20维数组")
        if np.any(target.positions_rad < self._lower) or np.any(target.positions_rad > self._upper):
            raise ValueError("WujiHand2目标超过关节限位；硬件边界禁止静默裁剪")
        if not callable(getattr(self._backend, "send_target", None)):
            if np.any(target.velocities_rad_s != 0):
                raise NotImplementedError("当前WujiHand2原生位置接口不支持非零速度目标")
            if np.any(target.effort_a != 0):
                raise NotImplementedError("当前WujiHand2原生位置接口不支持非零effort目标")

    def validate(self, target):
        """仅预检目标内容，供不持有发布期限的上层约束使用。"""
        with self._lock:
            self._require_connected()
            self._validate(target)

    def send_target(self, target: WujiTarget, *, deadline_ns: int):
        with self._lock:
            self._require_connected()
            if not self.active:
                raise RuntimeError("必须先显式enable WujiHand2")
            try:
                self.validate_target(target, deadline_ns)
                if callable(getattr(self._backend, "send_target", None)):
                    self._backend.send_target(target, deadline_ns=int(deadline_ns))
                else:
                    self._backend.send(target.positions_rad, deadline_ns=int(deadline_ns))
            except BaseException as error:
                self._fail(error)

    def read_state(self):
        with self._lock:
            self._require_connected()
            try:
                state = self._backend.read_state()
                if isinstance(state, WujiState):
                    return state
                diagnostics = dict(_field(state, "diagnostics", {}) or {})
                online = diagnostics.get("joint_online")
                if online is None:
                    online_value = _field(state, "online")
                    if not isinstance(online_value, bool):
                        raise ValueError("WujiHand2反馈缺少真实在线诊断")
                    online_count = 20 if online_value else 0
                else:
                    if len(online) != 20:
                        raise ValueError("WujiHand2在线诊断必须有20项")
                    online_count = sum(bool(value) for value in online)
                enabled = _field(state, "enabled")
                if enabled is not None:
                    diagnostics["enabled"] = enabled
                faults = _field(state, "faults")
                if faults:
                    diagnostics["faults"] = tuple(faults)
                velocity = _ordered_positions(_field(state, "velocities_rad_s"), optional=True)
                effort = _ordered_positions(_field(state, "effort_a"), optional=True)
                diagnostics.update(
                    velocity_feedback_available=velocity is not None,
                    effort_feedback_available=effort is not None,
                )
                received = _field(state, "received_ns")
                if received is None:
                    raise ValueError("WujiHand2反馈缺少真实接收时间")
                return WujiState(
                    _ordered_positions(_field(state, "positions_rad")), velocity, effort,
                    received, online_count, diagnostics,
                )
            except BaseException as error:
                self._fail(error)

    def home(self):
        with self._lock:
            self._require_connected()
            if not self.active:
                raise RuntimeError("必须先显式enable WujiHand2才能home")
            try:
                return self._backend.home()
            except BaseException as error:
                self._fail(error)

    def _stop(self, emergency):
        self._active = False
        if not self._owned or self._backend is None or self.closed:
            return StopResult(False, None, "只读会话不停止设备")
        method = "emergency_stop" if emergency else "disable"
        try:
            result = getattr(self._backend, method)()
            self._owned = False
            # 原生接口成功返回之前已核验新鲜失能反馈；注入接口须保留确认结果。
            return result if result is not None else StopResult(
                True, True if isinstance(self._backend, NativeWujiHand2Driver) else None, method,
            )
        except BaseException as error:
            self._failure = error
            raise

    def disable(self):
        with self._lock:
            return self._stop(False)

    def emergency_stop(self):
        with self._lock:
            return self._stop(True)

    def close(self):
        with self._lock:
            if self._closed:
                return
            errors = []
            if self._owned:
                try:
                    self._stop(self._failure is not None)
                except BaseException as error:
                    errors.append(error)
            backend, self._backend = self._backend, None
            self._closed, self._connected, self._active = True, False, False
            if backend is not None:
                try:
                    backend.close()
                except BaseException as error:
                    errors.append(error)
            if len(errors) > 1:
                raise BaseExceptionGroup("WujiHand2关闭过程中停止与释放失败", errors)
            if errors:
                raise errors[0]

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if exc is not None:
                raise BaseExceptionGroup("控制异常且关闭失败", [exc, cleanup]) from None
            raise
        return False
