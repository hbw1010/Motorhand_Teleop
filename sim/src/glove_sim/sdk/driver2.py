"""驱动 2: STM32G4 LQS-GW02 3-CAN gateway via the vendor C++ SDK.

The vendor SDK (``hardware/encos_g4_gateway_sdk/vendor``, unmodified) is
header-only C++.  ``shim/g4_shim.cpp`` wraps its ``lqs::HandClientG4`` in a C
ABI; this module builds that shim on first use and exposes the subset of the
``encos_hand.EncosHand`` API that ``RealGlove`` uses, so 驱动 1 (3x CANable2)
and 驱动 2 are interchangeable.

SDK behaviour kept as is: one serial port at 921600 baud, 200 Hz IO thread,
fixed wiring CAN1 ID9-12 / CAN2 ID5-8 / CAN3 ID1-4, MIT target in degrees,
vendor 3x zero-gain release on close.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4] / "hardware" / "encos_g4_gateway_sdk"
VENDOR_INCLUDE = ROOT / "vendor" / "include"
SHIM_SOURCE = ROOT / "shim" / "g4_shim.cpp"
LIBRARY = ROOT / "build" / "libg4_shim.so"
BAUD = 921600

ERROR_MAP = {
    0: "No Error", 1: "电机过热", 2: "电机过流", 3: "电压过高",
    4: "电压过低", 5: "编码器错误", 6: "刹车电压高", 7: "DRV错误",
}
VALID_POSITION, VALID_CURRENT, VALID_TEMPERATURE = 1, 2, 4

# USB-serial bridges of the gateway board (README: /dev/ttyUSB0 or ttyACM0).
GATEWAY_USB_IDS = {(0x1A86, 0x7523), (0x1A86, 0x55D3), (0x1A86, 0x55D4), (0x0483, 0x5740),
                   (0x10C4, 0xEA60), (0x0403, 0x6001), (0x0403, 0x6015)}


class Driver2Error(RuntimeError):
    pass


class G4Joint(ctypes.Structure):
    _fields_ = [("bus", ctypes.c_uint32), ("id", ctypes.c_uint32), ("valid", ctypes.c_uint32),
                ("error", ctypes.c_uint32), ("angle_deg", ctypes.c_float),
                ("current_a", ctypes.c_float), ("temp_c", ctypes.c_float),
                ("speed_rpm", ctypes.c_float), ("hz", ctypes.c_float),
                ("samples", ctypes.c_uint64), ("received_ns", ctypes.c_int64),
                ("board_timestamp_us", ctypes.c_uint32), ("max_board_gap_us", ctypes.c_uint32),
                ("max_host_gap_us", ctypes.c_uint32), ("tx_failures", ctypes.c_uint32)]


_lib = None
_lib_lock = threading.Lock()


def build_shim(force: bool = False) -> Path:
    sources = [SHIM_SOURCE, *VENDOR_INCLUDE.glob("*.hpp")]
    if not force and LIBRARY.exists() and LIBRARY.stat().st_mtime >= max(p.stat().st_mtime for p in sources):
        return LIBRARY
    LIBRARY.parent.mkdir(parents=True, exist_ok=True)
    compiler = os.environ.get("CXX", "g++")
    tmp = LIBRARY.with_suffix(f".{os.getpid()}.tmp")
    cmd = [compiler, "-std=c++17", "-O2", "-shared", "-fPIC", "-pthread",
           f"-I{VENDOR_INCLUDE}", str(SHIM_SOURCE), "-o", str(tmp)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise Driver2Error(f"驱动 2 需要 C++ 编译器（{compiler}）来编译厂商 SDK") from exc
    if result.returncode != 0:
        raise Driver2Error(f"编译驱动 2 失败:\n{result.stderr[-2000:]}")
    tmp.replace(LIBRARY)
    return LIBRARY


def load_library():
    global _lib
    with _lib_lock:
        if _lib is not None:
            return _lib
        lib = ctypes.CDLL(str(build_shim()))
        lib.g4_joint_size.restype = ctypes.c_uint32
        if lib.g4_joint_size() != ctypes.sizeof(G4Joint):
            raise Driver2Error("驱动 2 结构体大小不一致，请重新编译 shim")
        lib.g4_now_ns.restype = ctypes.c_int64
        lib.g4_open.restype = ctypes.c_void_p
        lib.g4_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
        lib.g4_set_mit.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32] + [ctypes.c_float] * 5 + [ctypes.c_int64]
        lib.g4_set_servo.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32] + [ctypes.c_float] * 3 + [ctypes.c_int64]
        lib.g4_release.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
        lib.g4_snapshots.restype = ctypes.c_int
        lib.g4_snapshots.argtypes = [ctypes.c_void_p, ctypes.POINTER(G4Joint), ctypes.c_int]
        lib.g4_release_count.restype = ctypes.c_uint64
        lib.g4_release_count.argtypes = [ctypes.c_void_p]
        lib.g4_close.argtypes = [ctypes.c_void_p]
        _lib = lib
        return lib


def discover_gateway_ports() -> list[str]:
    try:
        import serial.tools.list_ports as lp
    except ImportError as exc:  # pragma: no cover
        raise Driver2Error("缺少 pyserial，无法发现网关串口") from exc
    return [info.device for info in lp.comports()
            if (getattr(info, "vid", None), getattr(info, "pid", None)) in GATEWAY_USB_IDS]


class GatewayHand:
    """EncosHand-compatible facade over ``lqs::HandClientG4``.

    Motor id == CAN id (unique 1..12 across the three buses).
    """

    def __init__(self, port: str, *, baud: int = BAUD, hz: int = 200,
                 online_age_s: float = 0.25) -> None:
        self.port = port
        self.baud = int(baud)
        self.hz = int(hz)
        self.online_age_s = float(online_age_s)
        self._lib = load_library()
        self._h = None
        self._routes: dict[int, tuple[int, int]] = {}
        self._buf = (G4Joint * 32)()

    def connect(self) -> None:
        if not os.path.exists(self.port):
            raise Driver2Error(f"网关串口不存在: {self.port}")
        h = self._lib.g4_open(self.port.encode(), self.baud, self.hz)
        if not h:
            raise Driver2Error(f"无法打开网关串口 {self.port}（权限？可执行 sudo chmod 666 {self.port} 或加入 dialout 组）")
        self._h = h

    def _snapshots(self) -> list[G4Joint]:
        if not self._h:
            return []
        n = self._lib.g4_snapshots(self._h, self._buf, len(self._buf))
        return [self._buf[i] for i in range(n)]

    def scan(self, timeout_s: float = 0.6) -> dict[int, list[int]]:
        """Read only: wait for position replies on the SDK's fixed routes."""
        end = time.monotonic() + timeout_s
        while True:
            joints = self._snapshots()
            answered = [j for j in joints if j.samples > 0]
            if len(answered) == len(joints) and joints or time.monotonic() >= end:
                break
            time.sleep(0.02)
        self._routes = {int(j.id): (int(j.bus), int(j.id)) for j in joints}
        result: dict[int, list[int]] = {}
        for j in answered:
            result.setdefault(int(j.bus), []).append(int(j.id))
        return result

    def routes(self) -> dict[int, tuple[int, int]]:
        return dict(self._routes)

    def get_online_ids(self) -> tuple[int, ...]:
        now = self._lib.g4_now_ns()
        limit = int(self.online_age_s * 1e9)
        return tuple(sorted(int(j.id) for j in self._snapshots()
                            if j.samples and j.received_ns and now - j.received_ns <= limit))

    def get_all_angles(self) -> dict[int, float]:
        return {int(j.id): float(j.angle_deg) for j in self._snapshots() if j.valid & VALID_POSITION}

    def get_all_feedbacks(self) -> dict[int, dict]:
        out = {}
        for j in self._snapshots():
            if not j.valid & VALID_POSITION:
                continue
            fb = {"angle_deg": float(j.angle_deg), "err_code": int(j.error),
                  "err_str": ERROR_MAP.get(int(j.error), f"Err:{int(j.error)}"),
                  "current_a": float(j.current_a) if j.valid & VALID_CURRENT else float("nan"),
                  "temp_driver": float(j.temp_c) if j.valid & VALID_TEMPERATURE else float("nan"),
                  "hz": float(j.hz), "samples": int(j.samples), "bus": int(j.bus),
                  "max_host_gap_us": int(j.max_host_gap_us)}
            if j.received_ns:
                fb["received_ns"] = int(j.received_ns)
            out[int(j.id)] = fb
        return out

    def _route(self, motor_id: int) -> tuple[int, int] | None:
        return self._routes.get(int(motor_id))

    def enable(self, motor_id: int, enable: bool = True) -> None:
        """Enable = zero-gain MIT (limp) until the controller commands; disable = release."""
        route = self._route(motor_id)
        if route is None or not self._h:
            return
        if enable:
            self._lib.g4_set_mit(self._h, *route, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        else:
            self._lib.g4_release(self._h, *route)

    def set_mit(self, motor_id: int, target_deg: float, kp: float = 15.0, kd: float = 0.5,
                vel_rad_s: float = 0.0, torque_nm: float = 0.0, *, deadline_ns: int | None = None) -> None:
        route = self._route(motor_id)
        if route is None or not self._h:
            return
        self._lib.g4_set_mit(self._h, *route, float(target_deg), float(kp), float(kd),
                             float(vel_rad_s), float(torque_nm), int(deadline_ns or 0))

    def set_position(self, motor_id: int, target_deg: float, speed_rpm: float = 30.0,
                     current_a: float = 0.8, *, deadline_ns: int | None = None) -> None:
        route = self._route(motor_id)
        if route is None or not self._h:
            return
        self._lib.g4_set_servo(self._h, *route, float(target_deg), float(speed_rpm),
                               float(current_a), int(deadline_ns or 0))

    def diagnostics(self) -> dict:
        return {"release_count": int(self._lib.g4_release_count(self._h)) if self._h else 0,
                "joints": {int(j.id): {"bus": int(j.bus),
                                       "samples": int(j.samples),
                                       "max_host_gap_ms": j.max_host_gap_us / 1000.0}
                           for j in self._snapshots()}}

    def close(self) -> None:
        if self._h:
            h, self._h = self._h, None
            self._lib.g4_close(h)
