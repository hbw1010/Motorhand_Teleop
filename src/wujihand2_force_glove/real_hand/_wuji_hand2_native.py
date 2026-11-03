"""真机 C ABI 薄绑定；构造不加载库、不联网、不连接或使能设备。"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, fields
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path
import threading
import weakref

import numpy as np

from ..domain import WUJI_JOINTS, WujiState
from . import preflight


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LIBRARY = _PROJECT_ROOT / ".cache/native-real-hand/build/libreal_hand.so"


def check_deadline(deadline_ns):
    """仅核验主机单调时钟期限，不访问设备或改变会话状态。"""
    import time

    if (isinstance(deadline_ns, (bool, np.bool_)) or not isinstance(deadline_ns, Integral)
            or not 0 <= deadline_ns <= 2**64 - 1):
        raise ValueError("deadline_ns必须是uint64范围内的整数")
    if deadline_ns <= time.monotonic_ns():
        raise TimeoutError("WujiHand2目标期限已过期")


def _number(value, name, *, allow_zero=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or (value < 0 if allow_zero else value <= 0)):
        raise ValueError(f"{name}必须为{'非负' if allow_zero else '正'}有限实数")
    return float(value)


def _text(value, name, *, required=False):
    if (not isinstance(value, str) or "\x00" in value
            or (required and not value.strip())):
        raise ValueError(f"{name}必须为{'非空' if required else ''}且不含NUL的字符串")
    return value


def _positions(value):
    array = np.asarray(value)
    if array.shape != (20,) or array.dtype.kind not in "iuf":
        raise ValueError("目标必须为20个实数弧度角，顺序为拇指至小指、各S1-S4")
    result = np.array(array, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(result).all():
        raise ValueError("目标角度不得含NaN或Inf")
    return result


@dataclass(frozen=True)
class SafetyOptions:
    """回零渐变及可选主机约束；不改变正常直传，不代表实体安全保证。"""

    max_velocity: float | None = None
    max_acceleration: float | None = None
    realtime_max_velocity: float | None = None
    command_timeout: float = 0.5
    feedback_timeout: float = 0.5
    home_rate_hz: float = 100.0
    home_tolerance: float = 0.15
    home_timeout: float = 15.0
    enable_timeout: float = 5.0
    transition_duration: float = 1.5

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None and field.name in {"max_velocity", "max_acceleration", "realtime_max_velocity"}:
                continue
            object.__setattr__(self, field.name, _number(value, field.name))
        if (self.max_velocity is None) != (self.max_acceleration is None):
            raise ValueError("显式回零轨迹需同时提供max_velocity和max_acceleration")
        if self.home_rate_hz > 1000:
            raise ValueError("home_rate_hz不得超过1000Hz")


class _Options(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_int), ("hand", ctypes.c_int),
        ("serial_number", ctypes.c_char_p), ("address", ctypes.c_char_p),
        ("approved_firmware_version", ctypes.c_char_p),
        ("preflight_firmware_version", ctypes.c_char_p),
        *((field.name, ctypes.c_double) for field in fields(SafetyOptions)),
        ("effort_limit", ctypes.c_double), ("mit_kp", ctypes.c_double),
        ("mit_kd", ctypes.c_double), ("timeout_ms", ctypes.c_uint32),
        ("auto_home", ctypes.c_uint8),
    ]


class _State(ctypes.Structure):
    _fields_ = [
        ("positions", ctypes.c_double * 20),
        ("velocities_rad_s", ctypes.c_double * 20),
        ("effort_a", ctypes.c_double * 20),
        ("has_velocity", ctypes.c_uint8), ("has_effort", ctypes.c_uint8),
        ("lower_limits", ctypes.c_double * 20),
        ("upper_limits", ctypes.c_double * 20),
        ("enabled", ctypes.c_uint8 * 20), ("online", ctypes.c_uint8 * 20),
        ("error_codes", ctypes.c_uint32 * 20),
        ("timestamp_ns", ctypes.c_uint64), ("device_timestamp_us", ctypes.c_uint64),
        ("phase", ctypes.c_int), ("firmware_version", ctypes.c_char * 128),
    ]


@lru_cache(maxsize=1)
def _native_library():
    hint = ("请先执行 cmake -S src/wujihand2_force_glove/real_hand/native -B .cache/native-real-hand/build "
            "&& cmake --build .cache/native-real-hand/build --parallel；"
            "构建需要固定版本官方 C SDK，导入与构造不会自动下载或构建")
    if not _LIBRARY.is_file():
        raise FileNotFoundError(f"真机原生库尚未构建：{_LIBRARY}；{hint}")
    try:
        lib = ctypes.CDLL(str(_LIBRARY))
    except OSError as exc:
        raise RuntimeError(f"真机原生库加载失败：{exc}；{hint}") from exc
    lib.real_hand_abi_version.argtypes = []
    lib.real_hand_abi_version.restype = ctypes.c_int
    abi_version = lib.real_hand_abi_version()
    if abi_version != 6:
        raise RuntimeError(f"真机原生库 ABI 不匹配：需要6，实际{abi_version}，请重新构建；{hint}")
    pointer, error, size = ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t
    lib.real_hand_create.argtypes = [ctypes.POINTER(_Options), error, size]
    lib.real_hand_create.restype = pointer
    for name in ("connect", "enable", "home", "disable", "emergency_stop", "close"):
        method = getattr(lib, f"real_hand_{name}")
        method.argtypes = [pointer, error, size]
        method.restype = ctypes.c_int
    lib.real_hand_send.argtypes = [
        pointer, ctypes.POINTER(ctypes.c_double), size, ctypes.c_uint64, error, size,
    ]
    lib.real_hand_send.restype = ctypes.c_int
    lib.real_hand_read_state.argtypes = [pointer, ctypes.POINTER(_State), error, size]
    lib.real_hand_read_state.restype = ctypes.c_int
    lib.real_hand_destroy.argtypes = [pointer]
    lib.real_hand_destroy.restype = None
    return lib


class NativeWujiHand2Driver:
    """二代私有C ABI绑定；不作为公共硬件接口。

    connect先经官方SDK只读预检固件并断开，再由C++连接核对身份。
    enable核准固件，接管并失能、清错一次、核验恢复后使能。
    auto_home默认False；公共接口仅允许显式home，不在enable/close自动运动。
    析构仅对本实例已接管的设备急停释放；只读会话只释放，with进入仅connect。
    同实例方法串行化；close后不能重连，须显式构造新实例。
    """

    joint_names = WUJI_JOINTS

    def __init__(self, *, model, hand, serial_number, address="",
                 approved_firmware_version="", safety=SafetyOptions(), effort_limit=None,
                 mit_kp=None, mit_kd=None, timeout_ms=1000, auto_home=False):
        if hand not in ("left", "right"):
            raise ValueError("hand必须为left或right")
        if model != "wuji_hand2":
            raise ValueError("本轮真机控制仅支持model='wuji_hand2'")
        if not isinstance(safety, SafetyOptions):
            raise TypeError("safety必须为SafetyOptions")
        if (isinstance(timeout_ms, (bool, np.bool_)) or not isinstance(timeout_ms, Integral)
                or not 1 <= timeout_ms <= 2**32 - 1):
            raise ValueError("timeout_ms必须为1至4294967295的整数")
        self._model, self._hand = model, hand
        self._serial_number = _text(serial_number, "serial_number", required=True)
        if (self.serial_number != self.serial_number.strip()
                or ord(self.serial_number[0]) <= 32 or ord(self.serial_number[-1]) <= 32):
            raise ValueError("serial_number不得包含首尾空白或控制字符")
        if not isinstance(auto_home, bool):
            raise TypeError("auto_home必须为bool")
        if (mit_kp is None) != (mit_kd is None):
            raise ValueError("mit_kp和mit_kd必须同时显式提供或同时省略")
        effort_limit = -1.0 if effort_limit is None else _number(effort_limit, "effort_limit", allow_zero=True)
        mit_kp = -1.0 if mit_kp is None else _number(mit_kp, "mit_kp", allow_zero=True)
        mit_kd = -1.0 if mit_kd is None else _number(mit_kd, "mit_kd", allow_zero=True)
        if effort_limit > float(np.finfo(np.float32).max) or mit_kp > 10 or mit_kd > 1:
            raise ValueError("控制参数越界：effort_limit不得超过float32上限，mit_kp<=10，mit_kd<=1")
        self._options = _Options(
            2, 0 if hand == "left" else 1,
            self._serial_number.encode("utf-8"), _text(address, "address").encode("utf-8"),
            _text(approved_firmware_version, "approved_firmware_version").encode("utf-8"),
            b"",
            *(0.0 if getattr(safety, field.name) is None else getattr(safety, field.name)
              for field in fields(SafetyOptions)),
            effort_limit, mit_kp, mit_kd, int(timeout_ms), int(auto_home),
        )
        self._safety = safety
        self._lock = threading.RLock()
        self._handle = None
        self._lib = None
        self._finalizer = None
        self._closed = False
        self._send_failed = False

    @property
    def model(self):
        return self._model

    @property
    def hand(self):
        return self._hand

    @property
    def serial_number(self):
        return self._serial_number

    @property
    def safety(self):
        return self._safety

    @staticmethod
    def _check(status, error):
        if status:
            message = error.value.decode("utf-8", errors="replace") or f"原生调用失败（{status}）"
            raise (ValueError if status == -1 else RuntimeError)(message)

    def _call(self, name, *args):
        if self._closed or self._handle is None:
            raise RuntimeError("真机会话未连接或已关闭")
        error = ctypes.create_string_buffer(2048)
        status = getattr(self._lib, f"real_hand_{name}")(
            self._handle, *args, error, len(error)
        )
        self._check(status, error)

    def connect(self):
        """显式只读连接；不使能、不回零、不扫描默认设备。"""
        with self._lock:
            if self._closed:
                raise RuntimeError("真机会话已关闭，不能重连")
            if self._handle is not None:
                raise RuntimeError("真机会话已连接，不重复连接")
            self._lib = _native_library()
            try:
                firmware = preflight.read_firmware_version(
                    serial_number=self.serial_number, hand=self.hand,
                    address=self._options.address.decode("utf-8"), timeout_ms=self._options.timeout_ms,
                )
                self._options.preflight_firmware_version = firmware.encode("utf-8")
            except BaseException:
                self._closed = True
                raise
            error = ctypes.create_string_buffer(2048)
            self._handle = self._lib.real_hand_create(ctypes.byref(self._options), error, len(error))
            if not self._handle:
                raise RuntimeError(error.value.decode("utf-8", errors="replace") or "原生会话创建失败")
            self._finalizer = weakref.finalize(self, self._lib.real_hand_destroy, self._handle)
            try:
                self._call("connect")
            except BaseException:
                self._destroy()
                raise
        return self

    def enable(self):
        """显式接管并确认使能；仅auto_home=True时随后回零。"""
        with self._lock:
            self._call("enable")

    def send(self, positions_rad, *, deadline_ns):
        """发送20角目标；拒绝非法输入，不截断、不自动重试或清错。"""
        positions = _positions(positions_rad)
        with self._lock:
            try:
                pointer = positions.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                check_deadline(deadline_ns)
                if deadline_ns > 2**64 - 1:
                    raise ValueError("Wuji2期限必须可表示为uint64")
                self._call("send", pointer, 20, int(deadline_ns))
            except BaseException:
                # 原生发送失败执行急停；close不得再次发起回零运动。
                self._send_failed = True
                raise

    def _read_native(self):
        state = _State()
        self._call("read_state", ctypes.byref(state))
        return state

    def read_state(self):
        """返回独立只读数组；保留真实接收时间，设备未给时间戳时返回None。"""
        with self._lock:
            state = self._read_native()
            return WujiState(
                positions_rad=np.array(state.positions, dtype=np.float64, copy=True),
                velocities_rad_s=(np.array(state.velocities_rad_s, dtype=np.float64, copy=True)
                                  if state.has_velocity else None),
                effort_a=(np.array(state.effort_a, dtype=np.float64, copy=True)
                          if state.has_effort else None),
                received_ns=int(state.timestamp_ns),
                online_count=sum(state.online),
                diagnostics={
                    "joint_enabled": tuple(bool(value) for value in state.enabled),
                    "enabled": all(state.enabled),
                    "joint_online": tuple(bool(value) for value in state.online),
                    "joint_error_codes": tuple(state.error_codes),
                    "device_timestamp_us": state.device_timestamp_us or None,
                    "phase": {0: "disconnected", 1: "connected", 2: "enabled", 3: "fault", 4: "closed"}[state.phase],
                    "firmware_version": state.firmware_version.decode("utf-8"),
                    "firmware_source": preflight.FIRMWARE_SOURCE,
                    "velocity_feedback_available": bool(state.has_velocity),
                    "effort_feedback_available": bool(state.has_effort),
                    "feedback_source": "本项目WujiHand2 C ABI 6 joint_states",
                    "fault_code_policy": "原生核心只放行已知Warning，保留全部原始错误码",
                },
            )

    @property
    def firmware_version(self):
        with self._lock:
            return self._read_native().firmware_version.decode("utf-8", errors="replace")

    @property
    def firmware_version_source(self):
        """固件来自同设备连接前只读预检，而不是C接口或操作人批准值。"""
        return preflight.FIRMWARE_SOURCE

    def _limits(self, name):
        with self._lock:
            result = np.array(getattr(self._read_native(), name), dtype=np.float64, copy=True)
            result.flags.writeable = False
            return result

    @property
    def lower_limits(self):
        return self._limits("lower_limits")

    @property
    def upper_limits(self):
        return self._limits("upper_limits")

    def home(self):
        """通过官方位置发布路径回到关节零坐标，并确认新鲜实测到位。"""
        with self._lock:
            self._call("home")

    def disable(self):
        """直接禁用，不先回零。"""
        with self._lock:
            self._call("disable")

    def emergency_stop(self):
        """直接急停，不回零、不自动清错。"""
        with self._lock:
            self._call("emergency_stop")

    def _destroy(self):
        self._closed = True
        self._handle = None
        if self._finalizer is not None:
            self._finalizer()

    def close(self):
        """按auto_home策略正常关闭；失败仍销毁句柄并保留原生错误。"""
        with self._lock:
            if self._closed:
                return
            try:
                if self._handle is not None and not self._send_failed:
                    self._call("close")
            finally:
                self._destroy()

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if exc is not None:
                raise BaseExceptionGroup("控制异常且关闭清理失败", [exc, cleanup]) from None
            raise
        return False
