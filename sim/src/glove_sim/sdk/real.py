from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Mapping

import numpy as np

from ..mapping import GloveMapping
from .base import GloveState


class RealGloveError(RuntimeError):
    pass


def discover_ports() -> list[str]:
    """CANable2 ports only (VID/PID 16d0:117e)."""
    try:
        import serial.tools.list_ports as lp
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RealGloveError("缺少 pyserial，无法发现手套设备") from exc
    return [info.device for info in lp.comports()
            if (getattr(info, "vid", None), getattr(info, "pid", None)) == (0x16D0, 0x117E)]


class RealGlove:
    """Thin wrapper around the extracted ``encos_hand`` SDK.

    The wrapper is intentionally conservative: it refuses to enable output
    unless all 12 motors are online, the calibration file is valid, and no
    motor reports an error.  Torque commands are MIT feed-forward with kp=kd=0.
    """

    def __init__(self, mapping: GloveMapping, *, ports: Mapping[int, str] | None = None,
                 torque_sign: tuple[float, ...] | None = None) -> None:
        self.mapping = mapping
        self.ports = dict(ports) if ports else None
        self.torque_sign = np.asarray(torque_sign if torque_sign is not None else [1.0] * 12, dtype=np.float64)
        self._hand = None
        self._enabled = False
        self._last_angles_deg: dict[int, float] = {}

    @property
    def connected(self) -> bool:
        return self._hand is not None and bool(self._hand.get_online_ids())

    def open(self) -> GloveState:
        try:
            from encos_hand import EncosHand
        except ImportError as exc:
            raise RealGloveError("缺少 encos_hand SDK；请在 sim 的 Pixi 环境中运行") from exc
        if self.ports is None:
            found = discover_ports()
            if not found:
                raise RealGloveError(
                    "未发现 CANable2（USB 16d0:117e）。检查适配器、udev 规则和用户组；"
                    "离线演示请用默认 sim 命令（假手套）。"
                )
            self.ports = {i + 1: port for i, port in enumerate(found[:3])}
        self._hand = EncosHand(port_map=self.ports, announce_connection=True)
        try:
            self._hand.connect()
            self._hand.scan()
            online = tuple(sorted(self._hand.get_online_ids()))
            if online != tuple(range(1, 13)):
                raise RealGloveError(f"必须 12 个电机在线，当前在线={online}")
            state = self.read()
            bad = {mid: err for mid, err in state.errors.items() if err}
            if bad:
                raise RealGloveError(f"电机报警: {bad}")
            return state
        except Exception:
            self.close()
            raise

    def read(self) -> GloveState:
        if self._hand is None:
            raise RealGloveError("手套未连接")
        angles_deg = {int(k): float(v) for k, v in self._hand.get_all_angles().items()}
        missing = [mid for mid in range(1, 13) if mid not in angles_deg or not math.isfinite(angles_deg[mid])]
        if missing:
            raise RealGloveError(f"反馈缺失或无效: {missing}")
        feedbacks = {int(k): dict(v) for k, v in self._hand.get_all_feedbacks().items()}
        currents, temps, errors = {}, {}, {}
        received_ns: list[int] = []
        for mid in range(1, 13):
            fb = feedbacks.get(mid, {})
            currents[mid] = float(fb.get("current_a", float("nan")))
            temps[mid] = float(fb.get("temp_driver", float("nan")))
            code = int(fb.get("err_code", 0) or 0)
            errors[mid] = str(fb.get("err_str", "")) if code else ""
            stamp = fb.get("received_ns")
            if stamp is not None:
                try:
                    received_ns.append(int(stamp))
                except (TypeError, ValueError):
                    pass
        self._last_angles_deg = angles_deg
        # Use the oldest motor packet timestamp for a conservative sensor age.
        stamp_ns = min(received_ns) if received_ns else time.monotonic_ns()
        return GloveState(
            timestamp_ns=stamp_ns,
            angles_rad=np.asarray(self.mapping.raw_deg_to_model_rad(angles_deg), dtype=np.float64),
            raw_deg=angles_deg,
            current_a=currents,
            temperature_c=temps,
            errors={mid: err for mid, err in errors.items() if err},
            online_ids=tuple(sorted(self._hand.get_online_ids())),
            source="real",
        )

    def enable(self, enabled: bool = True) -> None:
        if self._hand is None:
            raise RealGloveError("手套未连接")
        if enabled:
            state = self.read()
            if state.errors:
                raise RealGloveError(f"存在报警，拒绝使能: {state.errors}")
            for mid in range(1, 13):
                self._hand.enable(mid, True)
            self._enabled = True
        else:
            for mid in range(1, 13):
                try:
                    self._hand.enable(mid, False)
                except Exception:
                    pass
            self._enabled = False

    def command_torque(self, motor_id: int, torque_nm: float, *, deadline_ns: int | None = None) -> None:
        if self._hand is None or not self._enabled:
            return
        angle = self._last_angles_deg.get(motor_id)
        if angle is None or not math.isfinite(angle):
            return
        sign = float(self.torque_sign[motor_id - 1])
        self._hand.set_mit(
            motor_id=motor_id,
            target_deg=angle,
            kp=0.0,
            kd=0.0,
            vel_rad_s=0.0,
            torque_nm=float(torque_nm) * sign,
            deadline_ns=deadline_ns,
        )

    def command_position(self, motor_id: int, target_deg: float, *, speed_rpm: float = 5.0,
                         current_a: float = 0.8, deadline_ns: int | None = None) -> None:
        if self._hand is None:
            raise RealGloveError("手套未连接")
        self._hand.set_position(motor_id, target_deg, speed_rpm=speed_rpm, current_a=current_a,
                                deadline_ns=deadline_ns)

    def close(self) -> None:
        if self._hand is not None:
            try:
                self.enable(False)
            except Exception:
                pass
            try:
                self._hand.close()
            except Exception:
                pass
            self._hand = None
        self._enabled = False
