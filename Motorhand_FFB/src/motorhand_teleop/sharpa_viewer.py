"""通过本机UDP显示Sharpa状态；只显示，不控制设备或访问SDK。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import socket
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOINT_SUFFIXES = (
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
)


@dataclass(frozen=True)
class ViewerFrame:
    hand: str
    joint_names: tuple[str, ...]
    position_rad: tuple[float, ...]
    scene_path: Path
    qpos: tuple[float, ...] | None
    load_nm: tuple[float, ...] | None
    load_source: str
    simulated: bool
    received_ns: int | None


def _numbers(value, name: str, size: int | None = None) -> tuple[float, ...]:
    if not isinstance(value, list) or not value or (size is not None and len(value) != size):
        raise ValueError(f"{name}维度无效")
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
        raise ValueError(f"{name}必须包含有限数值")
    return tuple(float(item) for item in value)


def validate_frame(payload, project_root: Path = PROJECT_ROOT) -> ViewerFrame:
    """校验数据和场景文件；不导入MuJoCo、不打开网络或设备。"""
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int:
        raise ValueError("需要schema_version=2的Sharpa对象")
    if payload["schema_version"] != 2 or payload.get("robot") != "sharpa":
        raise ValueError("仅接受schema_version=2、robot=sharpa")
    hand = payload.get("hand")
    if hand not in ("left", "right") or type(payload.get("simulated")) is not bool:
        raise ValueError("hand或simulated字段无效")
    names = payload.get("joint_names")
    expected_names = {f"{hand}_{suffix}" for suffix in JOINT_SUFFIXES}
    if not isinstance(names, list) or len(names) != 22 or any(not isinstance(name, str) for name in names):
        raise ValueError("需要22个语义关节名")
    if set(names) != expected_names:
        raise ValueError("22个关节名必须完整、无重复并匹配手侧")
    positions = _numbers(payload.get("position_rad"), "position_rad", 22)
    scene_value = payload.get("scene_path")
    if not isinstance(scene_value, str) or not Path(scene_value).is_absolute():
        raise ValueError("scene_path必须为本项目内XML绝对路径")
    scene = Path(scene_value).resolve()
    root = project_root.resolve()
    if not scene.is_relative_to(root) or scene.suffix.lower() != ".xml" or not scene.is_file():
        raise ValueError("scene_path必须为本项目内现有XML文件")
    simulated = payload["simulated"]
    qpos_value = payload.get("qpos")
    qpos = _numbers(qpos_value, "qpos") if qpos_value is not None else None
    if simulated and qpos is None:
        raise ValueError("仿真显示需要完整qpos")
    load_value = payload.get("load_nm")
    load = _numbers(load_value, "load_nm", 22) if load_value is not None else None
    expected_source = "sharpa-mujoco-contact-joint-torque" if simulated else "sharpa-sdk-reported-joint-torque"
    if payload.get("load_source") != expected_source:
        raise ValueError("load_source必须明确匹配真实SDK或仿真来源")
    received_ns = payload.get("received_ns")
    if received_ns is not None and (type(received_ns) is not int or received_ns <= 0):
        raise ValueError("received_ns必须为正整数原始反馈时间")
    return ViewerFrame(hand, tuple(names), positions, scene, qpos, load, expected_source, simulated, received_ns)


def apply_frame(model, data, frame: ViewerFrame) -> None:
    """只修改展示副本；不积分、不发送控制。"""
    import mujoco

    if frame.simulated:
        if len(frame.qpos) != model.nq:
            raise ValueError(f"qpos需要{model.nq}维，收到{len(frame.qpos)}维")
        data.qpos[:] = frame.qpos
    else:
        addresses = []
        for name in frame.joint_names:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"展示场景缺少语义转动关节：{name}")
            addresses.append(model.jnt_qposadr[joint])
        data.qpos[addresses] = frame.position_rad
    mujoco.mj_forward(model, data)


def _receive(connection: socket.socket) -> ViewerFrame | None:
    try:
        packet, _ = connection.recvfrom(65536)
    except (BlockingIOError, socket.timeout):
        return None
    try:
        return validate_frame(json.loads(packet))
    except (ValueError, UnicodeDecodeError, OSError) as error:
        print(f"忽略无效Sharpa状态：{error}", flush=True)
        return None


def run(port: int) -> None:
    import mujoco
    import mujoco.viewer

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        connection.bind(("127.0.0.1", port))
        connection.settimeout(0.05)
        print(f"等待Sharpa状态：127.0.0.1:{port}；关闭窗口仅退出查看器，不停止控制端。", flush=True)
        frame = None
        while frame is None:
            frame = _receive(connection)
        model = mujoco.MjModel.from_xml_path(str(frame.scene_path))
        data = mujoco.MjData(model)
        apply_frame(model, data, frame)
        print(f"显示{'仿真' if frame.simulated else '真机'}{frame.hand}手；负载来源：{frame.load_source}", flush=True)
        connection.setblocking(False)
        last_update = min(time.monotonic(), frame.received_ns * 1e-9) if frame.received_ns else time.monotonic()
        source_ns = frame.received_ns
        stale = False
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                latest = None
                while True:
                    incoming = _receive(connection)
                    if incoming is None:
                        break
                    if (incoming.scene_path, incoming.hand, incoming.simulated) != (frame.scene_path, frame.hand, frame.simulated):
                        print("忽略场景或设备侧别变化；请重新启动查看器。", flush=True)
                        continue
                    if incoming.received_ns is not None and source_ns is not None and incoming.received_ns <= source_ns:
                        continue
                    if latest is not None and incoming.received_ns is not None and latest.received_ns is not None and incoming.received_ns <= latest.received_ns:
                        continue
                    latest = incoming
                if latest is not None:
                    try:
                        with viewer.lock():
                            apply_frame(model, data, latest)
                    except ValueError as error:
                        print(f"忽略无效展示姿态：{error}", flush=True)
                    else:
                        frame = latest
                        source_ns = latest.received_ns
                        last_update = min(time.monotonic(), source_ns * 1e-9) if source_ns else time.monotonic()
                        if stale:
                            print("Sharpa状态已恢复更新。", flush=True)
                        stale = False
                if not stale and time.monotonic() - last_update > 1:
                    print("Sharpa控制源已超过1秒未更新；窗口保留最后画面，当前状态未知。", flush=True)
                    stale = True
                viewer.sync()
                time.sleep(0.01)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, epilog="关闭查看器只关闭显示窗口，不停止或失能native控制端；请在控制端单独停止。")
    parser.add_argument("--port", type=int, default=8766, help="本机UDP端口（默认8766，仅绑定127.0.0.1）")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port必须在1至65535之间")
    try:
        run(args.port)
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError) as error:
        parser.exit(1, f"查看器错误：{error}\n")


if __name__ == "__main__":
    main()
