# -*- coding: utf-8 -*-
import time
import math
import threading
from typing import Callable, Optional
import can
from .protocol import MotorProtocol

class HeadlessChannelWorker(threading.Thread):
    def __init__(self, ch_id: int, on_feedback: Optional[Callable[[int, int, dict], None]] = None):
        super().__init__(daemon=True)
        self.ch_id = ch_id
        self.on_feedback = on_feedback
        self.running = True
        self.bus = None
        self.port = None
        self.is_connected = False
        self.error_msg = ""

        self._req_open = False
        self._target_port = None
        self._req_close = False
        self._req_scan = False
        self._cmd_lock = threading.Lock()
        self.state_lock = threading.Lock()

        # 握手事件
        self.conn_event = threading.Event()
        self.scan_event = threading.Event()

        self.active_ids = []
        self.targets = {}
        self.special_cmds = []
        self.last_angles = {}
        self.latest_feedbacks = {}
        self.slot_time_s = 0.005  # 5ms 时隙

    def connect(self, port: str):
        with self._cmd_lock:
            self._target_port = port
            self.conn_event.clear()
            self._req_open = True

    def disconnect(self):
        with self._cmd_lock:
            self._req_close = True

    def trigger_scan(self):
        with self._cmd_lock:
            self.scan_event.clear()
            self._req_scan = True

    def set_joint_cmd(self, mid: int, cmd: dict):
        with self._cmd_lock:
            self.targets[mid] = cmd

    def post_special_cmd(self, arb_id: int, data: bytes):
        with self._cmd_lock:
            self.special_cmds.append((arb_id, data))

    def _internal_close(self):
        if self.bus:
            try:
                self.bus.shutdown()
            except Exception:
                pass
            self.bus = None
        self.is_connected = False
        self.active_ids.clear()
        # 注意: 严禁在此处调用 conn_event.set()，防止 open 前期假唤醒！

    def _internal_open(self, port: str):
        self._internal_close()
        self.port = port
        try:
            self.bus = can.Bus(interface='slcan', channel=self.port, bitrate=1000000)
            self.is_connected = True
            self.error_msg = ""
        except Exception as e:
            self.is_connected = False
            self.error_msg = str(e)
        finally:
            # 只有在物理总线彻底初始化完毕后，才允许唤醒主线程
            self.conn_event.set()

    def run(self):
        slot_idx = 0
        last_tx = {}

        while self.running:
            if self._req_open:
                with self._cmd_lock:
                    target = self._target_port
                    self._req_open = False
                self._internal_open(target)

            if self._req_close:
                with self._cmd_lock:
                    self._req_close = False
                self._internal_close()

            if not self.is_connected or not self.bus:
                time.sleep(0.02)
                continue

            # ================= 自检探测 =================
            if self._req_scan:
                with self._cmd_lock:
                    self._req_scan = False
                found = []
                try:
                    while self.bus.recv(timeout=0) is not None:
                        pass
                    
                    self.bus.send(can.Message(arbitration_id=0x7FF, data=[0xFF, 0xFF, 0x00, 0x82], is_extended_id=False))
                    t_scan_end = time.time() + 0.15
                    while time.time() < t_scan_end:
                        rx = self.bus.recv(timeout=0.005)
                        if rx and rx.arbitration_id == 0x7FF and len(rx.data) >= 5 and rx.data[:3] == b'\xff\xff\x01':
                            mid = int.from_bytes(rx.data[3:5], "big")
                            if mid not in found:
                                found.append(mid)
                    found.sort()

                    for mid in found:
                        probe_pkt = MotorProtocol.position_command(0.0, 0.0, 0.0, ack=2)
                        self.bus.send(can.Message(arbitration_id=mid, data=probe_pkt, is_extended_id=False))
                        t_wait = time.time() + 0.03
                        while time.time() < t_wait:
                            rx = self.bus.recv(timeout=0.002)
                            if rx and rx.arbitration_id == mid and len(rx.data) >= 8:
                                fb = MotorProtocol.decode_feedback(rx.data)
                                if fb and "angle_deg" in fb:
                                    fb["received_ns"] = time.monotonic_ns()
                                    with self.state_lock:
                                        self.last_angles[mid] = fb["angle_deg"]
                                        self.latest_feedbacks[mid] = fb
                                    if self.on_feedback:
                                        self.on_feedback(self.ch_id, mid, fb)
                                    break
                    self.active_ids = found
                except Exception:
                    self._internal_close()
                finally:
                    self.scan_event.set()
                continue

            # ================= 消费特殊指令 =================
            cmds_to_run = []
            with self._cmd_lock:
                if self.special_cmds:
                    cmds_to_run = self.special_cmds[:]
                    self.special_cmds.clear()

            for arb_id, s_data in cmds_to_run:
                try:
                    self.bus.send(can.Message(arbitration_id=arb_id, data=s_data, is_extended_id=False))
                    time.sleep(0.002)
                except Exception:
                    pass

            if not self.active_ids:
                time.sleep(0.01)
                continue

            # ================= 5ms 严格单播 =================
            current_id = self.active_ids[slot_idx]
            slot_idx = (slot_idx + 1) % len(self.active_ids)

            with self._cmd_lock:
                cfg = self.targets.get(current_id, {}).copy()

            motor_enabled = cfg.get("enabled", False)
            # 期限来自控制器采样周期；Python上层卡住也不能无限重发旧MIT目标。
            deadline_ns = cfg.get("deadline_ns")
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                motor_enabled = False
            mode = cfg.get("mode", "servo")

            if motor_enabled:
                if mode == "mit":
                    payload = MotorProtocol.mit_command(
                        math.radians(float(cfg.get("target_angle", 0.0))),
                        float(cfg.get("mit_vel", 0.0)),
                        float(cfg.get("mit_kp", 15.0)),
                        float(cfg.get("mit_kd", 0.5)),
                        float(cfg.get("mit_tor", 0.0))
                    )
                else:
                    payload = MotorProtocol.position_command(
                        float(cfg.get("target_angle", 0.0)),
                        float(cfg.get("servo_spd", 30.0)),
                        float(cfg.get("servo_cur", 0.8)),
                        ack=2
                    )
            else:
                cur_mech_ang = self.last_angles.get(current_id, 0.0)
                payload = MotorProtocol.position_command(cur_mech_ang, 0.0, 0.0, ack=2)

            last_tx[current_id] = payload
            try:
                self.bus.send(can.Message(arbitration_id=current_id, data=payload, is_extended_id=False))
            except Exception:
                pass

            slot_end = time.monotonic() + self.slot_time_s
            while time.monotonic() < slot_end:
                try:
                    rx = self.bus.recv(timeout=0.001)
                except Exception:
                    break
                if not rx:
                    continue
                if rx.arbitration_id == current_id and len(rx.data) >= 8:
                    if rx.data == last_tx.get(current_id):
                        continue
                    fb = MotorProtocol.decode_feedback(rx.data)
                    if fb and "angle_deg" in fb:
                        fb["received_ns"] = time.monotonic_ns()
                        with self.state_lock:
                            self.last_angles[current_id] = fb["angle_deg"]
                            self.latest_feedbacks[current_id] = fb
                        if self.on_feedback:
                            self.on_feedback(self.ch_id, current_id, fb)
                        break
            # 当前节点已回包就让出GIL，但保留原5ms时隙，不能加速刷CAN。
            # 三条串口线程不应在等待同一时隙结束时持续争抢算法线程。
            remaining = slot_end - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

    def stop(self):
        self.running = False
        self._internal_close()
