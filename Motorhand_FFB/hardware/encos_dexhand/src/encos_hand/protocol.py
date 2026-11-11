# -*- coding: utf-8 -*-
import math
import struct

# MIT nominal torque encoding range, not a motor rated/peak capability.
MIT_TORQUE_MIN_NM = -10.0
MIT_TORQUE_MAX_NM = 10.0
MIT_TORQUE_BITS = 12
MIT_TORQUE_MAX_RAW = (1 << MIT_TORQUE_BITS) - 1
MIT_TORQUE_STEP_NM = (MIT_TORQUE_MAX_NM - MIT_TORQUE_MIN_NM) / MIT_TORQUE_MAX_RAW
MIT_TORQUE_HALF_LSB_NM = MIT_TORQUE_STEP_NM / 2.0

ERROR_MAP = {
    0: "No Error", 1: "电机过热", 2: "电机过流", 3: "电压过高",
    4: "电压过低", 5: "编码器错误", 6: "刹车电压高", 7: "DRV错误"
}

class MotorProtocol:
    @staticmethod
    def position_command(position_deg: float, speed_rpm: float, current_a: float, ack: int = 2) -> bytes:
        position_bits = int.from_bytes(struct.pack(">f", float(position_deg)), "big")
        speed_raw = round(max(0.0, min(3276.7, float(speed_rpm))) * 10.0) & 0x7FFF
        current_raw = round(max(0.0, min(409.5, float(current_a))) * 10.0) & 0xFFF
        word = (
            (1 << 61)
            | ((position_bits & 0xFFFFFFFF) << 29)
            | (speed_raw << 14)
            | (current_raw << 2)
            | (int(ack) & 0x03)
        )
        return word.to_bytes(8, "big")

    @staticmethod
    def mit_command(position_rad: float, velocity_rad_s: float, kp: float, kd: float, torque_nm: float) -> bytes:
        p_raw = round((max(-5.0, min(5.0, float(position_rad))) + 5.0) / 10.0 * 0xFFFF) & 0xFFFF
        v_raw = round((max(-10.0, min(10.0, float(velocity_rad_s))) + 10.0) / 20.0 * 0xFFF) & 0xFFF
        kp_raw = round(max(0.0, min(500.0, float(kp))) / 500.0 * 0xFFF) & 0xFFF
        kd_raw = round(max(0.0, min(5.0, float(kd))) / 5.0 * 0x1FF) & 0x1FF
        t_raw = round((max(MIT_TORQUE_MIN_NM, min(MIT_TORQUE_MAX_NM, float(torque_nm))) - MIT_TORQUE_MIN_NM) / (MIT_TORQUE_MAX_NM - MIT_TORQUE_MIN_NM) * MIT_TORQUE_MAX_RAW) & MIT_TORQUE_MAX_RAW
        word = (
            (0 << 61)
            | (kp_raw << 49)
            | (kd_raw << 40)
            | (p_raw << 24)
            | (v_raw << 12)
            | t_raw
        )
        return word.to_bytes(8, "big")

    @staticmethod
    def decode_feedback(payload: bytes) -> dict:
        if len(payload) < 8:
            return {}
        rtype = payload[0] >> 5
        err_code = payload[0] & 0x1F
        res = {"err_code": err_code, "err_str": ERROR_MAP.get(err_code, f"Err:{err_code}")}
        if rtype == 2:
            res["angle_deg"] = struct.unpack(">f", payload[1:5])[0]
            res["current_a"] = int.from_bytes(payload[5:7], "big", signed=True) / 100.0
            raw_t = payload[7]
            res["temp_driver"] = (raw_t - 50) / 2.0 if 50 <= raw_t <= 220 else 0.0
            return res
        elif rtype == 1:
            raw_pos = (payload[1] << 8) | payload[2]
            res["angle_deg"] = math.degrees(-5.0 + raw_pos / 65535.0 * 10.0)
            raw_cur = ((payload[4] & 0x0F) << 8) | payload[5]
            res["current_a"] = -2.0 + raw_cur / 4095.0 * 4.0
            raw_mos = payload[6]
            res["temp_driver"] = (raw_mos - 50) / 2.0 if 50 <= raw_mos <= 220 else 0.0
            return res
        return {}
