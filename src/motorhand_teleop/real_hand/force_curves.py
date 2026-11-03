"""独立只读曲线入口；默认仅接收遥测，绝不启动控制或连接设备。"""

from __future__ import annotations

import argparse
from itertools import count
import json
import math
import socket
import sys
import time

from .curve_telemetry import DEFAULT_CURVE_PORT, decode_sample


_DEMO_SESSION_ID = "force-curves-offline-demo"
_DEMO_SEQUENCE = count()
_FINGERS = ("拇指", "食指", "中指", "无名指", "小指")


def make_demo_sample(elapsed_s: float, now_ns: int) -> dict:
    """生成离线合成曲线，不代表物理仿真、实测设备或触觉验证。"""
    phase = elapsed_s % 15.0
    if phase < 9.0:
        feedback_state = "active"
        jacobian_age_ns = 5_000_000
    elif phase < 12.0:
        feedback_state = "refreshing_pose"
        jacobian_age_ns = 80_000_000
    else:
        feedback_state = "refreshing_age"
        jacobian_age_ns = 600_000_000

    wuji_current = [
        0.35 * math.sin(elapsed_s * (0.8 + index * 0.025) + index * 0.43)
        + 0.08 * math.sin(elapsed_s * 2.4 + index)
        for index in range(20)
    ]
    wuji_load = [value / 2.0 for value in wuji_current]
    glove_current = [
        0.22 * math.sin(elapsed_s * (1.0 + index * 0.04) + index * 0.6)
        for index in range(12)
    ]
    glove_target = [
        -0.006 * math.sin(elapsed_s * 1.15 + index * 0.5)
        if feedback_state == "active" else 0.0
        for index in range(12)
    ]
    # 以下各列分别显式合成，仅用于检查显示；并非由设备或物理模型推算。
    wuji_error = [
        0.04 * math.sin(elapsed_s * 0.9 + index * 0.31)
        for index in range(20)
    ]
    wuji_velocity = [
        0.12 * math.cos(elapsed_s * 0.9 + index * 0.31)
        for index in range(20)
    ]
    contact_mask = [feedback_state == "active" and index % 4 >= 1 for index in range(20)]
    virtual_torque = [
        -2.5e-6 * math.sin(elapsed_s * 1.1 + index * 0.27) if active else 0.0
        for index, active in enumerate(contact_mask)
    ]
    projected_torque = [
        -0.012 * math.sin(elapsed_s * 1.15 + index * 0.5)
        if feedback_state == "active" else None
        for index in range(12)
    ]
    reflection_gain = 1500.0  # 演示合成倍率；不是硬件当前配置。
    glove_names = [
        f"ID{raw_id} {finger}{joint}"
        for raw_id, (finger, joint) in enumerate(
            ((finger, joint) for finger_index, finger in enumerate(_FINGERS)
             for joint in range(1, 5 if finger_index == 0 else 3)),
            start=1,
        )
    ]
    return {
        "schema_version": 1,
        "session_id": _DEMO_SESSION_ID,
        "sequence": next(_DEMO_SEQUENCE),
        "source": "demo",
        "timestamp_ns": now_ns,
        "wuji_received_ns": max(0, now_ns - 2_000_000),
        "glove_received_ns": max(0, now_ns - 3_000_000),
        "executed": False,
        "mode": "glove-to-hand2",
        "mapping_mode": "contact_impedance_estimate",
        "feedback_state": feedback_state,
        "jacobian_age_ns": jacobian_age_ns,
        "contact_state": "contact" if feedback_state == "active" else "waiting_geometry",
        "contact_active_mask": contact_mask,
        "virtual_hand_torque_nm": virtual_torque,
        "glove_projected_torque_nm": projected_torque,
        "contact_reflection_gain": reflection_gain,
        "wuji_tracking_error_rad": wuji_error,
        "wuji_velocity_rad_s": wuji_velocity,
        "wuji_joint_names": [
            f"{finger}{joint}" for finger in _FINGERS for joint in range(1, 5)
        ],
        "glove_joint_names": glove_names,
        "wuji_current_a": wuji_current,
        "wuji_load_fraction": wuji_load,
        "glove_current_a": glove_current,
        "glove_target": glove_target,
        "glove_target_unit": "Nm",
    }


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("端口必须是1至65535的整数") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("端口必须是1至65535的整数")
    return port


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="只读实时曲线监视器，不连接硬件、不启动控制。关闭监视器不会停止控制。",
    )
    parser.add_argument(
        "--port", type=_port, default=DEFAULT_CURVE_PORT,
        help=f"本机UDP遥测监听端口（默认：{DEFAULT_CURVE_PORT}）",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="离线合成演示，不打开网络或加载设备SDK；非物理仿真或触觉验证",
    )
    args = parser.parse_args(argv)

    # 延迟加载Qt：生成演示样本与解析命令行不需要创建GUI对象。
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication

    from .force_curves_view import ForceCurvesWindow

    receiver = None
    if not args.demo:
        try:
            receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            receiver.setblocking(False)
            receiver.bind(("127.0.0.1", args.port))
        except OSError as exc:
            if receiver is not None:
                receiver.close()
            print(
                f"无法监听只读遥测端口127.0.0.1:{args.port}：{exc}。"
                "请检查端口是否被其他监视器占用，或使用--port指定其他端口。",
                file=sys.stderr,
            )
            return 2

    timer = None
    try:
        app = QApplication.instance() or QApplication(sys.argv[:1])
        window = ForceCurvesWindow()
        window.setProperty("telemetry_port", receiver.getsockname()[1] if receiver else None)
        timer = QTimer(window)

        def cleanup() -> None:
            timer.stop()
            if receiver is not None:
                receiver.close()

        app.aboutToQuit.connect(cleanup)
        if args.demo:
            window.set_transport_notice(
                "离线合成演示（非物理仿真或触觉验证）；未连接硬件、未打开网络。"
            )
            started_ns = time.monotonic_ns()

            def deliver_demo() -> None:
                now_ns = time.monotonic_ns()
                sample = make_demo_sample((now_ns - started_ns) / 1e9, now_ns)
                payload = json.dumps(sample, ensure_ascii=False, allow_nan=False).encode("utf-8")
                window.ingest(decode_sample(payload))

            timer.timeout.connect(deliver_demo)
            timer.start(20)
            deliver_demo()
        else:
            ignored_packets = 0
            notice_prefix = (
                f"只读监听127.0.0.1:{args.port}；未启动控制。"
            )
            window.set_transport_notice(notice_prefix)

            def receive_pending() -> None:
                nonlocal ignored_packets
                previous_ignored = ignored_packets
                # 有界处理避免高频发送或错误包洪水阻塞GUI事件循环。
                for _ in range(128):
                    try:
                        payload, _address = receiver.recvfrom(65_535)
                    except BlockingIOError:
                        break
                    except OSError as exc:
                        timer.stop()
                        receiver.close()
                        window.set_transport_notice(f"{notice_prefix} 接收已停止：{exc}")
                        return
                    try:
                        sample = decode_sample(payload)
                    except ValueError:
                        ignored_packets += 1
                        continue
                    window.ingest(sample)
                if ignored_packets != previous_ignored:
                    window.set_transport_notice(
                        f"{notice_prefix} 已忽略无效数据包：{ignored_packets}。"
                    )

            timer.timeout.connect(receive_pending)
            timer.start(20)

        window.show()
        return int(app.exec_())
    finally:
        if timer is not None:
            timer.stop()
        if receiver is not None:
            receiver.close()


if __name__ == "__main__":
    raise SystemExit(main())
