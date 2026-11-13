# -*- coding: utf-8 -*-
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
import serial.tools.list_ports as lp

from .channel_worker import HeadlessChannelWorker
from .protocol import MotorProtocol


_SUPPORTED_USB_CAN_IDS = {(0x16D0, 0x117E)}  # CANable2

def discover_usb_serial_ports() -> tuple[str, ...]:
    """只返回已知USB-CAN适配器，避免把Rockchip CDC控制台当成CAN通道。"""
    return tuple(
        info.device
        for info in lp.comports()
        if (getattr(info, "vid", None), getattr(info, "pid", None)) in _SUPPORTED_USB_CAN_IDS
    )


class EncosHand:
    def __init__(self, port_map: Optional[Dict[int, str]] = None, config_path: Optional[str] = None,
                 *, announce_connection: bool = True):
        self.port_map = port_map
        self.config_path = Path(config_path) if config_path else Path(__file__).resolve().parents[2] / "configs" / "sim" / "joint_calibration.json"
        self.announce_connection = announce_connection
        self.calibrations = self._load_calibrations()
        
        self.workers: Dict[int, HeadlessChannelWorker] = {}
        for ch_id in (1, 2, 3):
            w = HeadlessChannelWorker(ch_id)
            w.start()
            self.workers[ch_id] = w

    def _load_calibrations(self) -> dict:
        defaults = {str(i): {"zero": 0.0, "min": -360.0, "max": 360.0} for i in range(1, 33)}
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text(encoding="utf-8"))
                for k, v in data.items():
                    defaults[str(k)] = v
            except Exception:
                pass
        return defaults

    def connect(self):
        all_ports = [p.device for p in lp.comports()]
        available = list(discover_usb_serial_ports())

        if not self.port_map:
            self.port_map = {i + 1: port for i, port in enumerate(available[:3])}
            ignored = [port for port in all_ports if port not in available]
            if ignored and not available:
                print(f"⚠️ 未发现带USB VID/PID的串口，已忽略平台串口: {ignored}")

        for ch_id, port in self.port_map.items():
            if ch_id in self.workers:
                if port in available or port in all_ports:
                    self.workers[ch_id].connect(port)
                else:
                    print(f"⚠️ [CAN-{ch_id}] 物理端口 {port} 未识别，跳过挂载")

        for ch_id, w in self.workers.items():
            if ch_id in self.port_map and self.port_map[ch_id] in all_ports:
                w.conn_event.wait(timeout=3.0)
                if w.is_connected:
                    if self.announce_connection:
                        print(f"✅ [CAN-{ch_id}] 成功挂载 {w.port} @ 1Mbps")
                else:
                    print(f"❌ [CAN-{ch_id}] 无法挂载 {w.port}: {w.error_msg}")


    def scan(self) -> Dict[int, List[int]]:
        active_workers = [w for w in self.workers.values() if w.is_connected]
        if not active_workers:
            return {}

        for w in active_workers:
            w.trigger_scan()

        for w in active_workers:
            w.scan_event.wait(timeout=1.5)

        return {w.ch_id: w.active_ids for w in active_workers}

    def _find_worker(self, motor_id: int) -> Optional[HeadlessChannelWorker]:
        for w in self.workers.values():
            if motor_id in w.active_ids:
                return w
        return None

    def enable(self, motor_id: int, enable: bool = True):
        w = self._find_worker(motor_id)
        if w:
            current_target = w.last_angles.get(motor_id, 0.0)
            cfg = w.targets.get(motor_id, {"mode": "servo", "servo_spd": 30.0, "servo_cur": 0.8})
            cfg["enabled"] = enable
            cfg["target_angle"] = current_target
            w.set_joint_cmd(motor_id, cfg)

    def set_position(self, motor_id: int, target_deg: float, speed_rpm: float = 30.0, current_a: float = 0.8, *, deadline_ns: Optional[int] = None):
        w = self._find_worker(motor_id)
        if w:
            w.set_joint_cmd(motor_id, {
                "enabled": True, "mode": "servo",
                "target_angle": float(target_deg),
                "servo_spd": float(speed_rpm),
                "servo_cur": float(current_a), "deadline_ns": deadline_ns
            })

    def set_mit(self, motor_id: int, target_deg: float, kp: float = 15.0, kd: float = 0.5, vel_rad_s: float = 0.0, torque_nm: float = 0.0, *, deadline_ns: Optional[int] = None):
        w = self._find_worker(motor_id)
        if w:
            w.set_joint_cmd(motor_id, {
                "enabled": True, "mode": "mit",
                "target_angle": float(target_deg),
                "mit_kp": float(kp),
                "mit_kd": float(kd),
                "mit_vel": float(vel_rad_s),
                "mit_tor": float(torque_nm), "deadline_ns": deadline_ns
            })

    def get_all_angles(self) -> Dict[int, float]:
        res = {}
        for w in self.workers.values():
            with w.state_lock:
                res.update(w.last_angles)
        return res

    def get_all_feedbacks(self) -> Dict[int, dict]:
        """返回最近一次每个电机的诊断反馈，不触发额外总线请求。"""
        result = {}
        for worker in self.workers.values():
            with worker.state_lock:
                for motor_id, feedback in worker.latest_feedbacks.items():
                    result[motor_id] = dict(feedback)
        return result

    def get_online_ids(self) -> tuple[int, ...]:
        return tuple(sorted({motor_id for worker in self.workers.values() for motor_id in worker.active_ids}))


    def close(self):
        for w in self.workers.values():
            w.stop()
