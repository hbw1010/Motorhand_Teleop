"""本机只读曲线遥测：不接设备、不等UI、不把缺失反馈补成零。"""
from __future__ import annotations

import json
import math
from numbers import Real
import socket
import time
import uuid

from ..domain import GLOVE_JOINTS

DEFAULT_CURVE_PORT = 8765
_MAX_PAYLOAD_BYTES = 16_384


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _values(values, count):
    return [None] * count if values is None else [_number(value) for value in values]


class CurvePublisher:
    """CLI在周期发送完成后旁路发布；丢曲线包不改变硬件输出或重发命令。"""

    def __init__(self, port, *, glove_model, motors, wuji_joint_names, clock_ns=None):
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("曲线端口必须为1至65535的整数")
        names = tuple(glove_model.motor_names)
        if set(motors) != set(range(1, 13)) or len(wuji_joint_names) != 20:
            raise ValueError("曲线通道必须包含20个Wuji语义轴及12个手套原始轴")
        self._bindings = tuple(
            (names.index(glove_model.joint_motor_names[motors[mid].joint]),
             float(motors[mid].sign * motors[mid].scale))
            for mid in range(1, 13)
        )
        self._wuji_names = list(wuji_joint_names)
        self._glove_names = [motors[mid].joint for mid in range(1, 13)]
        self._session_id = uuid.uuid4().hex
        self._sequence = 0
        self._clock = clock_ns or time.monotonic_ns
        self._destination = ("127.0.0.1", port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self.dropped_packets = 0

    def publish(self, tick):
        """只取本周期原始反馈与确认后的指令，不额外读取硬件。"""
        glove = tick.glove_state
        hand = tick.hand2_state
        feedback = tick.force_feedback
        unit, target, projected = None, None, None
        if feedback is not None:
            if feedback.model_torque_nm is not None:
                unit, target = "Nm", feedback.model_torque_nm
                projected = feedback.projected_model_torque_nm
            else:
                unit, target = "A", feedback.model_current_a
        self._sequence += 1
        published_ns = self._clock()
        sample = {
            "schema_version": 1,
            "session_id": self._session_id,
            "sequence": self._sequence,
            "source": "hardware",
            "timestamp_ns": max(hand.received_ns, glove.received_ns),
            "wuji_received_ns": hand.received_ns,
            "glove_received_ns": glove.received_ns,
            "published_ns": published_ns,
            "cycle_started_ns": tick.cycle_started_ns,
            "cycle_completed_ns": tick.cycle_completed_ns,
            "wuji_submit_ns": tick.wuji_submit_ns,
            "glove_submit_ns": tick.glove_submit_ns,
            "executed": tick.executed,
            "mode": tick.mode.value,
            "mapping_mode": "disabled" if feedback is None else feedback.mapping_mode,
            "feedback_state": "disabled" if feedback is None else feedback.projection_status,
            "jacobian_age_ns": None if feedback is None else feedback.jacobian_age_ns,
            "contact_state": "not_configured" if feedback is None else feedback.contact_state,
            "contact_active_mask": (None if feedback is None or feedback.contact_active_mask is None
                                    else feedback.contact_active_mask.tolist()),
            "virtual_hand_torque_nm": (None if feedback is None or feedback.virtual_hand_torque_nm is None
                                       else feedback.virtual_hand_torque_nm.tolist()),
            "contact_reflection_gain": None if feedback is None else feedback.contact_reflection_gain,
            "glove_projected_torque_nm": ([None] * 12 if projected is None else
                                          [float(projected[index] * scale)
                                           for index, scale in self._bindings]),
            "wuji_joint_names": self._wuji_names,
            "glove_joint_names": self._glove_names,
            "wuji_position_rad": _values(hand.positions_rad, 20),
            "glove_position_rad": _values(tick.glove_pose.vector(), 12),
            "glove_position_names": list(GLOVE_JOINTS),
            "glove_raw_position_deg": [
                _number(getattr(glove, "raw_motor_positions_deg", {}).get(mid))
                for mid in range(1, 13)
            ],
            # q12由异步电机帧组合；整体接收时间是最老帧，不能视为同时采样。
            "glove_motor_received_ns": [
                glove.feedback_by_motor.get(mid, {}).get("received_ns")
                for mid in range(1, 13)
            ],
            "wuji_target_rad": (_values(tick.output_positions_rad, 20)
                                if tick.mode.value == "glove-to-hand2" else None),
            "glove_target_position_rad": (_values(tick.output_positions_rad, 12)
                                          if tick.mode.value == "hand2-to-glove" else None),
            "wuji_current_a": _values(hand.effort_a, 20),
            "wuji_velocity_rad_s": _values(hand.velocities_rad_s, 20),
            "wuji_tracking_error_rad": _values(
                tick.output_positions_rad - hand.positions_rad
                if tick.mode.value == "glove-to-hand2" else None, 20,
            ),
            "wuji_load_fraction": _values(None if feedback is None else feedback.wuji_load_fraction, 20),
            "glove_current_a": [_number(glove.feedback_by_motor.get(mid, {}).get("current_a"))
                                for mid in range(1, 13)],
            "glove_target": ([None] * 12 if target is None else
                             [float(target[index] * scale) for index, scale in self._bindings]),
            "glove_target_unit": unit,
        }
        payload = json.dumps(sample, ensure_ascii=False, allow_nan=False,
                             separators=(",", ":")).encode("utf-8")
        try:
            self._socket.sendto(payload, self._destination)
        except OSError:
            # 非阻塞旁路只允许丢观测包，不等待重试、不影响已完成的控制周期。
            self.dropped_packets += 1

    def close(self):
        self._socket.close()


def _reject_constant(value):
    raise ValueError(f"曲线包不允许非有限数值：{value}")


def decode_sample(payload: bytes) -> dict:
    """UI只接受已知、有界的数据结构；未知或缺失量保持None而不是造测量。"""
    if not isinstance(payload, bytes) or not 0 < len(payload) <= _MAX_PAYLOAD_BYTES:
        raise ValueError("曲线数据包大小无效")
    try:
        sample = json.loads(payload, parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("曲线数据包不是有效JSON") from error
    if not isinstance(sample, dict) or type(sample.get("schema_version")) is not int or sample["schema_version"] != 1:
        raise ValueError("未知曲线数据版本")
    if not isinstance(sample.get("session_id"), str) or not 0 < len(sample["session_id"]) <= 128:
        raise ValueError("曲线会话标识无效")
    for key in ("sequence", "timestamp_ns", "wuji_received_ns", "glove_received_ns"):
        if type(sample.get(key)) is not int or sample[key] < 0:
            raise ValueError(f"曲线{key}必须为非负整数")
    if type(sample.get("executed")) is not bool:
        raise ValueError("曲线必须明确是否发送控制目标")
    for key, allowed in (
        ("source", ("hardware", "demo")),
        ("mode", ("glove-to-hand2", "hand2-to-glove")),
        ("mapping_mode", ("max_torque_linear_estimate", "contact_impedance_estimate", "calibrated", "disabled")),
        ("feedback_state", ("active", "refreshing_pose", "refreshing_age", "disabled")),
        ("glove_target_unit", ("Nm", "A", None)),
    ):
        if key not in sample or sample[key] not in allowed:
            raise ValueError(f"曲线{key}取值无效")
    if "jacobian_age_ns" not in sample or (sample["jacobian_age_ns"] is not None and
            (type(sample["jacobian_age_ns"]) is not int or sample["jacobian_age_ns"] < 0)):
        raise ValueError("曲线Jac年龄无效")
    for key in ("published_ns", "cycle_started_ns", "cycle_completed_ns",
                "wuji_submit_ns", "glove_submit_ns"):
        value = sample.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"曲线{key}必须为非负整数或缺失值")
    motor_stamps = sample.get("glove_motor_received_ns")
    if motor_stamps is not None and (not isinstance(motor_stamps, list) or len(motor_stamps) != 12
            or any(value is not None and (type(value) is not int or value < 0)
                   for value in motor_stamps)):
        raise ValueError("曲线glove_motor_received_ns必须为12路非负整数或缺失值")
    if "glove_position_names" in sample:
        names = sample["glove_position_names"]
        if not isinstance(names, list) or len(names) != 12 or any(
                not isinstance(name, str) or not 0 < len(name) <= 128 for name in names):
            raise ValueError("曲线glove_position_names通道名无效")
    for key, count in (("wuji_position_rad", 20), ("glove_position_rad", 12),
                       ("glove_raw_position_deg", 12), ("wuji_target_rad", 20),
                       ("glove_target_position_rad", 12)):
        values = sample.get(key)
        if values is not None and (not isinstance(values, list) or len(values) != count or any(
                value is not None and (type(value) not in (int, float) or _number(value) is None)
                for value in values)):
            raise ValueError(f"曲线{key}必须为{count}路有限数值或缺失值")
    if (sample["mode"] != "glove-to-hand2" and sample.get("wuji_target_rad") is not None
            or sample["mode"] != "hand2-to-glove" and sample.get("glove_target_position_rad") is not None):
        raise ValueError("曲线位置目标与跟随方向不一致")
    for key, count in (("wuji_joint_names", 20), ("glove_joint_names", 12)):
        names = sample.get(key)
        if not isinstance(names, list) or len(names) != count or any(
                not isinstance(name, str) or not 0 < len(name) <= 128 for name in names):
            raise ValueError(f"曲线{key}通道名无效")
    for key, count in (("wuji_current_a", 20), ("wuji_load_fraction", 20),
                       ("glove_current_a", 12), ("glove_target", 12)):
        values = sample.get(key)
        if not isinstance(values, list) or len(values) != count or any(
                value is not None and (type(value) not in (int, float) or _number(value) is None)
                for value in values):
            raise ValueError(f"曲线{key}必须为{count}路有限数值或缺失值")
    if sample["glove_target_unit"] is None and any(value is not None for value in sample["glove_target"]):
        raise ValueError("曲线有目标值但缺少量纲")
    for key, count in (("glove_projected_torque_nm", 12), ("wuji_tracking_error_rad", 20),
                       ("wuji_velocity_rad_s", 20)):
        if key not in sample:
            continue
        values = sample[key]
        if not isinstance(values, list) or len(values) != count or any(
                value is not None and (type(value) not in (int, float) or _number(value) is None)
                for value in values):
            raise ValueError(f"曲线{key}必须为{count}路有限数值或缺失值")
    if sample["glove_target_unit"] != "Nm" and any(
            value is not None for value in sample.get("glove_projected_torque_nm", [])):
        raise ValueError("曲线投影名义Nm只能伴随Nm输出")
    if sample["mode"] != "glove-to-hand2" and any(
            value is not None for value in sample.get("wuji_tracking_error_rad", [])):
        raise ValueError("反向跟随不能报告Wuji位置目标误差")
    gain = sample.get("contact_reflection_gain")
    if gain is not None and (
            type(gain) not in (int, float) or _number(gain) is None or gain <= 0
            or sample["mapping_mode"] != "contact_impedance_estimate"):
        raise ValueError("曲线接触反射倍率必须为正有限数且仅用于接触模式")
    if sample["mapping_mode"] == "contact_impedance_estimate":
        if sample.get("contact_state") not in ("free", "arming", "contact", "releasing", "waiting_geometry"):
            raise ValueError("接触阻抗曲线缺少有效的受阻估计状态")
        mask = sample.get("contact_active_mask")
        virtual = sample.get("virtual_hand_torque_nm")
        if not isinstance(mask, list) or len(mask) != 20 or any(type(value) is not bool for value in mask):
            raise ValueError("接触阻抗曲线掩码必须为20路bool")
        if (not isinstance(virtual, list) or len(virtual) != 20
                or any(type(value) not in (int, float) or _number(value) is None for value in virtual)):
            raise ValueError("接触阻抗曲线必须明确20路软件虚拟力矩")
        if sample["glove_target_unit"] != "Nm":
            raise ValueError("接触阻抗曲线只允许明确的名义Nm目标")
    elif (sample.get("contact_state", "not_configured") != "not_configured"
          or sample.get("contact_active_mask") is not None
          or sample.get("virtual_hand_torque_nm") is not None):
        raise ValueError("非接触阻抗模式不能报告已启用的受阻估计")
    return sample
