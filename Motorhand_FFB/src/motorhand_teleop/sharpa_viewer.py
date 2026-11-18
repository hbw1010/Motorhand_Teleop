"""通过本机UDP并排显示Motorhand与Sharpa；只显示，不控制设备或访问SDK。"""
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
    glove_tree_rad: tuple[float, ...] | None = None


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
    tree = payload.get("glove_tree_rad")
    if tree is not None:
        if payload.get("glove_tree_order") != "ve33_closure_independent_dependent":
            raise ValueError("Motorhand闭链姿态顺序无效")
        tree = _numbers(tree, "glove_tree_rad", 68)
    return ViewerFrame(hand, tuple(names), positions, scene, qpos, load, expected_source, simulated, received_ns, tree)


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


class ComparisonScene:
    """只合并显示副本；68轴直接取控制端闭链结果，不在界面重新求解。"""

    def __init__(self, frame: ViewerFrame):
        import mujoco
        import numpy as np
        import yaml

        side = "L" if frame.hand == "left" else "R"
        glove_root = PROJECT_ROOT.parent / "assets/hand_motor_ve33" / f"hand_motor_ve33_{side}"
        targets = json.loads((glove_root / "config/retarget_targets.json").read_text())
        rotation = np.asarray(targets["canonical_to_model"], dtype=float).T
        wrist = targets["sites"][0]
        if wrist["name"] != "wrist" or wrist["link"] != "base_link":
            raise ValueError("Motorhand腕部坐标定义不匹配")
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, rotation.ravel())
        glove_pos = np.array([0., -.17, 0.]) - rotation @ np.asarray(wrist["local_xyz_m"])

        hand_spec = mujoco.MjSpec.from_file(str(frame.scene_path))
        original = hand_spec.compile()
        original_data = mujoco.MjData(original)
        mujoco.mj_forward(original, original_data)
        hand_pos = np.array([0., .17, 0.]) - original_data.xpos[original.body(f"{frame.hand}_hand_C_MC").id]
        spec = mujoco.MjSpec.from_string('''<mujoco><visual>
          <global offwidth="1280" offheight="720"/>
          <headlight ambient=".35 .35 .35" diffuse=".6 .6 .6" specular=".1 .1 .1"/>
          </visual></mujoco>''')
        glove_spec = mujoco.MjSpec.from_file(str(glove_root / "hand_closed_chain.xml"))
        for child, prefix, pos, orientation in (
            (glove_spec, "glove_", glove_pos, quat),
            (hand_spec, "sharpa_", hand_pos, [1., 0., 0., 0.]),
        ):
            # 查看器只做FK，两份资产的物理步进选项不参与显示。
            child.option.timestep = spec.option.timestep
            child.option.gravity[:] = spec.option.gravity
            child.option.integrator = spec.option.integrator
            for geom in child.geoms:
                geom.contype = geom.conaffinity = 0
            mount = spec.worldbody.add_frame(pos=pos, quat=orientation)
            spec.attach(child, prefix=prefix, frame=mount)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        closure = yaml.safe_load((glove_root / "config/loop_closures.yaml").read_text())
        tree_names = closure["independent"] + closure["dependent"]
        self.glove_indices = np.array([int(self.model.joint("glove_" + name).qposadr[0]) for name in tree_names])
        self.hand_indices = np.empty(original.nq, dtype=int)
        for joint in range(original.njnt):
            if original.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError("Sharpa展示模型需要语义转动关节")
            self.hand_indices[original.jnt_qposadr[joint]] = self.model.joint("sharpa_" + original.joint(joint).name).qposadr[0]

    def apply(self, frame: ViewerFrame) -> None:
        import mujoco

        if frame.glove_tree_rad is None:
            raise ValueError("缺少Motorhand完整姿态，请重启原生控制端")
        if frame.simulated:
            if len(frame.qpos) != len(self.hand_indices):
                raise ValueError("Sharpa仿真qpos维度不匹配")
            self.data.qpos[self.hand_indices] = frame.qpos
        else:
            for name, value in zip(frame.joint_names, frame.position_rad):
                self.data.qpos[self.model.joint("sharpa_" + name).qposadr[0]] = value
        self.data.qpos[self.glove_indices] = frame.glove_tree_rad
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def setup_view(viewer) -> None:
        import mujoco
        import numpy as np

        viewer.cam.lookat[:] = [0., 0., .12]
        viewer.cam.distance = .75
        viewer.cam.azimuth, viewer.cam.elevation = 180., -12.
        for index, (label, y) in enumerate((("Motorhand", -.17), ("Sharpa", .17))):
            geom = viewer.user_scn.geoms[index]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3),
                               np.array([0., y, -.08]), np.eye(3).ravel(), np.ones(4, dtype=np.float32))
            geom.label = label
        viewer.user_scn.ngeom = 2


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


def run(port: int, managed: bool = False) -> None:
    import mujoco
    import mujoco.viewer

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        connection.bind(("127.0.0.1", port))
        connection.settimeout(0.05)
        close_hint = "关闭窗口将请求本次控制停止。" if managed else "关闭窗口仅退出查看器，不停止控制端。"
        print(f"等待Sharpa状态：127.0.0.1:{port}；{close_hint}", flush=True)
        frame = None
        while frame is None:
            frame = _receive(connection)
        comparison = ComparisonScene(frame) if frame.glove_tree_rad is not None else None
        if comparison is not None:
            model, data = comparison.model, comparison.data
            update = comparison.apply
        else:
            print("当前控制端缺少Motorhand完整姿态；重启控制端后可并排显示。", flush=True)
            model = mujoco.MjModel.from_xml_path(str(frame.scene_path))
            data = mujoco.MjData(model)
            update = lambda incoming: apply_frame(model, data, incoming)
        update(frame)
        print(f"显示{'仿真' if frame.simulated else '真机'}{frame.hand}手；负载来源：{frame.load_source}", flush=True)
        connection.setblocking(False)
        last_update = min(time.monotonic(), frame.received_ns * 1e-9) if frame.received_ns else time.monotonic()
        source_ns = frame.received_ns
        stale = False
        with mujoco.viewer.launch_passive(model, data) as viewer:
            if comparison is not None:
                with viewer.lock():
                    comparison.setup_view(viewer)
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
                            update(latest)
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
    parser.add_argument("--managed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port必须在1至65535之间")
    try:
        run(args.port, args.managed)
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError) as error:
        parser.exit(1, f"查看器错误：{error}\n")


if __name__ == "__main__":
    main()
