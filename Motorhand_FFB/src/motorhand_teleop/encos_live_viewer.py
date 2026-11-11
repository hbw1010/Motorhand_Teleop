"""只读Encos反馈驱动的MuJoCo方向观察器。

该工具不发送位置或MIT目标；它把Encos角度转换为项目GlovePose，再经过同一套
vE3.3闭链模型和WujiRetarget输出，供人工判断真实电机方向与仿真模型方向。

* --driver 2（默认）：STM32G4 网关（厂商 C++ SDK，只发位置查询）；--driver 1：3×CANable2。
* --layout side（默认）：手套模型放在 Wuji 手旁边并列显示；overlay：按腕部重合叠放。
* --retarget pad（默认）/ axis：指腹任务点或原手指中心轴线任务点（*_axis.yaml）。
* 球标记 = 当前重定向实际使用的任务点（红：手套，蓝：Wuji；PIP/DIP/指尖）。
* --demo：无硬件，脚本驱动手指开合。
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Mapping

import mujoco
import mujoco.viewer
import numpy as np

from .domain import GLOVE_JOINTS
import pinocchio as pin
import yaml
from scipy.spatial.transform import Rotation

from .real_hand.motor_hand import EncosGloveHardwareDriver
from .mujoco_viewer import (
    _HAND_JOINTS,
    _HAND_MJCF,
    _combined_model,
    _glove_to_hand_mount,
    _set_glove_pose,
)
from .retargeting import GloveWujiRetargetPipeline, default_retarget_config


_DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def _set_hand_qpos(model: mujoco.MjModel, data: mujoco.MjData, values: np.ndarray) -> None:
    if values.shape != (20,):
        raise ValueError("Hand2目标必须是20维")
    for name, value in zip(_HAND_JOINTS, values):
        joint = model.joint(name)
        data.qpos[int(joint.qposadr[0])] = float(value)
        data.qvel[int(joint.dofadr[0])] = 0.0


def _print_axis_table(
    driver: EncosGloveHardwareDriver,
    raw_angles: Mapping[int, float],
    glove_joints: np.ndarray,
    hand_target: np.ndarray,
    previous_glove: np.ndarray | None,
    previous_hand: np.ndarray | None,
) -> None:
    delta_glove = np.zeros(12) if previous_glove is None else glove_joints - previous_glove
    delta_hand = np.zeros(20) if previous_hand is None else hand_target - previous_hand
    print("axis | motor | raw_deg | motor_rad | joint_rad | d_joint | hand_joint | hand_rad | d_hand", flush=True)
    for axis, config in enumerate(driver.motors.values(), start=1):
        glove_index = GLOVE_JOINTS.index(config.joint)
        hand_joint = _hand_joint_for_glove(config.joint)
        hand_index = _HAND_JOINTS.index(hand_joint)
        motor_rad = math.radians(
            (raw_angles[config.motor_id] - config.zero_deg) * config.sign * config.scale
        )
        print(
            f"{axis:4d} | {config.motor_id:5d} | "
            f"{raw_angles[config.motor_id]:7.2f} | {motor_rad:9.4f} | "
            f"{glove_joints[glove_index]:9.4f} | {delta_glove[glove_index]:7.4f} | "
            f"{hand_joint:17s} | {hand_target[hand_index]:8.4f} | {delta_hand[hand_index]:7.4f}",
            flush=True,
        )


def _hand_joint_for_glove(glove_joint: str) -> str:
    if glove_joint == "thumb_cmc_flex":
        return "r_thumb_cmc_flex"
    if glove_joint == "thumb_cmc_abd":
        return "r_thumb_cmc_abd"
    if glove_joint == "thumb_mcp":
        return "r_thumb_mcp"
    if glove_joint == "thumb_ip":
        return "r_thumb_ip"
    finger, joint = glove_joint.split("_", 1)
    prefix = f"{finger}_finger" if finger != "pinky" else "pinky"
    return f"r_{prefix}_{joint}"


def _wait_for_viewer_shutdown(viewer) -> None:
    # MuJoCo close() 只请求退出；弱引用在渲染线程和 GLFW 清理完成前仍有效。
    # 等待清理结束，避免 Python 退出时与渲染线程竞争。
    renderer_ref = getattr(viewer, "_sim", None)
    if not callable(renderer_ref):
        return
    deadline = time.monotonic() + 5.0
    while renderer_ref() is not None:
        if time.monotonic() >= deadline:
            raise TimeoutError("MuJoCo观察器渲染线程未及时退出")
        time.sleep(0.01)


_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
_WUJI_FINGER = {"thumb": "thumb", "index": "index_finger", "middle": "middle_finger",
                "ring": "ring_finger", "pinky": "pinky"}
# MediaPipe 21 点里每指的 PIP/DIP/指尖下标（拇指为 MCP/IP/指尖）。
_TASK_POINTS = tuple((2 + 4 * i, 3 + 4 * i, 4 + 4 * i) for i in range(5))


def _retarget_inputs(root: Path, retarget: str) -> tuple[Path, Path]:
    """(手套 asset_root, Wuji yaml)。axis 用生成脚本留下的轴线资产副本。"""
    config = root / "config/retargeting/official"
    if retarget == "pad":
        return root / "assets" / "hand_motor_ve33", config / "right.yaml"
    assets = root / "build/pad_fit_axis_assets/right"
    if not (assets / "hand_motor_ve33_R/config/retarget_targets.json").exists():
        raise RuntimeError("缺少轴线资产副本；先运行 python scripts/generate_pad_task_points.py")
    return assets, config / "right_axis.yaml"


def _wuji_task_offsets(root: Path, yaml_path: Path) -> list[tuple[str, np.ndarray]]:
    """Wuji 任务点（yaml 的 link_naming）相对其所在活动指节的偏移，用于在 MJCF 上画标记。

    MJCF 与 URDF 的关节位置有几毫米出入，所以挂在 MJCF 指节上而不是用 URDF 世界坐标。
    """
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    urdf = (yaml_path.parent / config["optimizer"]["urdf_path"]).resolve()
    naming = config["optimizer"]["link_naming"]
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    pin.framesForwardKinematics(model, data, pin.neutral(model))
    out = []
    for finger in _FINGERS:
        name = _WUJI_FINGER[finger]
        for key, body in (("pip", "middle"), ("dip", "distal"), ("tip", "distal")):
            frame = "r_" + naming[key].format(finger=name)
            body_name = f"r_{name}_{body}"
            local = data.oMf[model.getFrameId(body_name)].actInv(data.oMf[model.getFrameId(frame)].translation)
            out.append((body_name, np.asarray(local, dtype=np.float64)))
    return out


def _side_offset(root: Path, gap_m: float) -> np.ndarray:
    """Wuji 手小指一侧的横向平移，使手套模型与 Wuji 手并列。"""
    model = mujoco.MjModel.from_xml_path(str(root / _HAND_MJCF))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    index = data.site_xpos[model.site("r_index_finger_tip").id]
    pinky = data.site_xpos[model.site("r_pinky_tip").id]
    lateral = (pinky - index) / np.linalg.norm(pinky - index)
    return lateral * gap_m


def _add_sphere(scene, pos: np.ndarray, rgba, radius: float = 0.0035) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0.0, 0.0]), np.asarray(pos, dtype=np.float64),
                        np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def _demo_joints(elapsed: float) -> dict[str, float]:
    phase = 0.5 * (1.0 - np.cos(2.0 * np.pi * ((elapsed % 8.0) / 8.0)))
    joints = {name: 0.0 for name in GLOVE_JOINTS}
    for finger, (mcp, pip) in {"index": (0.28, 1.0), "middle": (0.22, 0.78),
                               "ring": (0.18, 0.62), "pinky": (0.14, 0.5)}.items():
        joints[f"{finger}_mcp_flex"] = mcp * phase
        joints[f"{finger}_pip"] = pip * phase
    return joints


def _pinch_poses(landmarks) -> list[tuple[str, dict[str, float]]]:
    """手套模型上拇指指尖任务点与各指指尖任务点重合的姿态（拇指 4 关节 + 该指 MCP/PIP 优化）。"""
    from scipy.optimize import minimize

    thumb = ["thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip"]
    poses = []
    for finger, tip in (("index", 8), ("middle", 12), ("ring", 16), ("pinky", 20)):
        names = thumb + [f"{finger}_mcp_flex", f"{finger}_pip"]

        def joints_of(x, names=names):
            joints = {name: 0.0 for name in GLOVE_JOINTS}
            joints.update(zip(names, (float(v) for v in x)))
            return joints

        def cost(x, tip=tip, joints_of=joints_of):
            try:
                points = landmarks.step(joints_of(x)).points_m
            except Exception:
                return 1.0
            return float(np.sum((points[4] - points[tip]) ** 2)) + 1e-4 * float(np.sum(np.square(x)))

        best = None
        for x0 in ([0.3, 0.3, 0.3, 0.3, 0.5, 0.6], [0.6, 0.6, 0.4, 0.4, 0.8, 0.8], [0.0, 0.8, 0.2, 0.2, 0.9, 0.5]):
            result = minimize(cost, x0, method="L-BFGS-B", bounds=[(-0.6, 1.6)] * 6)
            if best is None or result.fun < best.fun:
                best = result
        poses.append((finger, joints_of(best.x)))
    return poses


def _demo_pinch(elapsed: float, poses) -> tuple[str, dict[str, float]]:
    """每 4 s 一指：张开 → 捏合 → 停留 → 张开。"""
    finger, target = poses[int(elapsed // 4.0) % len(poses)]
    t = (elapsed % 4.0) / 4.0
    s = 0.5 * (1.0 - np.cos(np.pi * min(t / 0.35, 1.0))) if t < 0.35 else 1.0 if t < 0.6 else \
        0.5 * (1.0 + np.cos(np.pi * (t - 0.6) / 0.35)) if t < 0.95 else 0.0
    return finger, {name: s * value for name, value in target.items()}


def _pinch_mm(points: np.ndarray) -> list[float]:
    return [float(np.linalg.norm(points[4] - points[4 + 4 * k]) * 1000.0) for k in range(1, 5)]


def run_live_viewer(
    project_root: str | Path = _DEFAULT_ROOT, *, mapping_path: str | Path | None = None,
    driver: str = "2", port: str = "auto", layout: str = "side", gap_m: float = 0.13,
    retarget: str = "pad", demo: str | None = None, markers: bool = True, trace: int | None = None,
) -> None:
    root = Path(project_root)
    asset_root, yaml_path = _retarget_inputs(root, retarget)
    glove = None
    if demo is None:
        hand_factory = None
        if driver == "2":
            from .real_hand.encos_gateway import gateway_hand_factory
            hand_factory = gateway_hand_factory(port)
        glove = EncosGloveHardwareDriver(
            hand="right", asset_root=root / "assets" / "hand_motor_ve33", config_path=mapping_path,
            hand_factory=hand_factory,
        )
    pipeline = GloveWujiRetargetPipeline.from_yaml(
        "right", asset_root=asset_root, config_path=yaml_path, estimate_jacobian=False,
    )
    if tuple(pipeline.retargeter.output_joint_names) != _HAND_JOINTS:
        raise RuntimeError("Hand2仿真关节顺序与重定向输出不一致")
    landmarks = pipeline.landmarks

    def solve(joints: dict[str, float]):
        frame = landmarks.step(joints)
        q = pipeline.retargeter.step(
            frame.points_m, apply_filter=False,
            model_to_canonical_row=landmarks.model_to_canonical_row,
            canonical_to_optimizer_row=landmarks.canonical_to_optimizer_row,
        )
        return frame, q

    zero = {name: 0.0 for name in GLOVE_JOINTS}
    zero_frame, zero_target = solve(zero)
    mount_pos, mount_quat, geometry_rms = _glove_to_hand_mount(
        root, zero_target, zero_frame.points_m, landmarks.model_to_canonical_row,
    )
    if layout == "side":
        mount_pos = mount_pos + _side_offset(root, gap_m)
    mount_rot = Rotation.from_quat([mount_quat[1], mount_quat[2], mount_quat[3], mount_quat[0]]).as_matrix()
    model = _combined_model(root, mount_pos, mount_quat)
    data = mujoco.MjData(model)
    pinch_poses = _pinch_poses(landmarks) if demo == "pinch" else None
    wuji_offsets = _wuji_task_offsets(root, yaml_path)
    wuji_bodies = [(model.body(name).id, local) for name, local in wuji_offsets]
    glove_points_ids = [i for triple in _TASK_POINTS for i in triple]

    viewer = None
    try:
        if glove is not None:
            glove.connect()
            found = glove.scan()
            if glove.online_ids != tuple(range(1, 13)):
                raise RuntimeError(f"要求12轴在线，当前online_ids={glove.online_ids}，scan={found}")
            print(f"实时只读MuJoCo观察器已连接手套（驱动 {driver}）；不会发送位置/MIT命令。", flush=True)
        else:
            print("演示模式（无硬件）：" + ("依次拇-食/中/无名/小对指，每指 4 s。" if demo == "pinch"
                                             else "脚本驱动四指开合。"), flush=True)
        print(f"重定向任务点：{'指腹' if retarget == 'pad' else '手指中心轴线'}（{yaml_path.name}）；"
              f"布局：{'并列' if layout == 'side' else '叠放'}；腕部换基几何残差={geometry_rms * 1000.0:.2f} mm", flush=True)
        if markers:
            print("球标记：红=手套任务点，蓝=Wuji 任务点（每指 PIP/DIP/指尖）。", flush=True)
        last_report = 0.0
        last_trace = 0.0
        trace_base = None
        if trace is not None:
            trace_joint = next(c.joint for c in (glove.motors.values() if glove else []) if c.motor_id == trace) \
                if glove is not None else None
            if trace_joint is None:
                raise RuntimeError("--trace 需要真实手套")
            trace_mj = model.joint("glove_" + landmarks.model.joint_motor_names[trace_joint])
            trace_hand = _hand_joint_for_glove(trace_joint)
            print(f"TRACE 电机 {trace} = {trace_joint}：每 0.1 s 打印相对开始时的变化量（度）："
                  f"电机原始角 | 手套关节角 | MuJoCo手套显示角 | Wuji {trace_hand}", flush=True)
        previous_glove = None
        previous_hand = None
        start = time.monotonic()
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = model.stat.center
            viewer.cam.distance = 0.55 if layout == "overlay" else 0.7
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = -18.0
            while viewer.is_running():
                if glove is not None:
                    state = glove.read_state()
                    joints = {name: float(state.pose.joints[name]) for name in GLOVE_JOINTS}
                    frame, q = solve(joints)
                    # SDK反馈已完成电机正解；显示直接使用同一真实装配，不再反解q。
                    _set_glove_pose(model, data, state.tree_positions_rad)
                else:
                    if pinch_poses is not None:
                        _, joints = _demo_pinch(time.monotonic() - start, pinch_poses)
                    else:
                        joints = _demo_joints(time.monotonic() - start)
                    frame, q = solve(joints)
                    landmarks.model.solve(joints)
                    _set_glove_pose(model, data, landmarks.model.evaluate_raw(joints).all_positions)
                _set_hand_qpos(model, data, q)
                mujoco.mj_forward(model, data)
                glove_world = frame.points_m @ mount_rot.T + mount_pos
                wuji_world = np.array([data.xpos[b] + data.xmat[b].reshape(3, 3) @ local for b, local in wuji_bodies])
                if markers:
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0
                        for i in glove_points_ids:
                            _add_sphere(viewer.user_scn, glove_world[i], (0.9, 0.15, 0.1, 0.9))
                        for p in wuji_world:
                            _add_sphere(viewer.user_scn, p, (0.1, 0.35, 0.95, 0.9))
                viewer.sync()

                now = time.monotonic()
                if trace is not None and now - last_trace >= 0.1:
                    values = np.array([
                        float(state.feedback_by_motor[trace]["angle_deg"]),
                        math.degrees(joints[trace_joint]),
                        math.degrees(data.qpos[int(trace_mj.qposadr[0])]),
                        math.degrees(q[_HAND_JOINTS.index(trace_hand)]),
                    ])
                    if trace_base is None:
                        trace_base = values
                    d = values - trace_base
                    ratio = f"{d[3] / d[0]:+.2f}" if abs(d[0]) > 2.0 else "  -  "
                    print(f"TRACE 原始 {d[0]:+7.2f} | 关节 {d[1]:+7.2f} | 显示 {d[2]:+7.2f} | Wuji {d[3]:+7.2f}"
                          f"  (Wuji/电机={ratio})", flush=True)
                    last_trace = now
                if trace is None and now - last_report >= 1.0:
                    wuji21 = np.zeros((21, 3))
                    for k, triple in enumerate(_TASK_POINTS):
                        wuji21[list(triple)] = wuji_world[3 * k:3 * k + 3]
                    g, w = _pinch_mm(glove_world), _pinch_mm(wuji21)
                    print("拇指-食/中/无名/小 任务点距离 mm  手套: " + " ".join(f"{v:5.1f}" for v in g)
                          + "   Wuji: " + " ".join(f"{v:5.1f}" for v in w), flush=True)
                    if glove is not None:
                        glove_vector = np.asarray([joints[name] for name in GLOVE_JOINTS], dtype=np.float64)
                        raw_angles = {
                            motor_id: float(values["angle_deg"])
                            for motor_id, values in state.feedback_by_motor.items()
                            if "angle_deg" in values
                        }
                        _print_axis_table(glove, raw_angles, glove_vector, q, previous_glove, previous_hand)
                        previous_glove = glove_vector
                        previous_hand = q.copy()
                    last_report = now
                time.sleep(0.01)
    finally:
        try:
            if glove is not None:
                glove.close()
        finally:
            if viewer is not None:
                _wait_for_viewer_shutdown(viewer)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="手套只读反馈驱动的 MuJoCo 观察器（手套模型 + Wuji Hand2）")
    parser.add_argument("--project-root", type=Path, default=_DEFAULT_ROOT)
    parser.add_argument("--mapping", type=Path, default=None)
    parser.add_argument("--driver", choices=("1", "2"), default="2", help="1：3×CANable2；2：G4 网关（默认）")
    parser.add_argument("--port", default="auto", help="驱动 2 网关串口，默认自动发现")
    parser.add_argument("--layout", choices=("side", "overlay"), default="side", help="并列（默认）或腕部重合叠放")
    parser.add_argument("--gap", type=float, default=0.13, help="并列时两模型的横向间距（米）")
    parser.add_argument("--retarget", choices=("pad", "axis"), default="pad", help="指腹（默认）或原中心轴线任务点")
    parser.add_argument("--demo", nargs="?", const="open", choices=("open", "pinch"), default=None,
                        help="无硬件演示：open 四指开合（默认），pinch 依次拇指对四指")
    parser.add_argument("--no-markers", action="store_true", help="不画任务点球标记")
    parser.add_argument("--trace", type=int, default=None, metavar="MOTOR_ID",
                        help="每 0.1 s 打印该电机的原始角/关节角/显示角/Wuji 角变化量（替代每秒总表）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_live_viewer(args.project_root, mapping_path=args.mapping, driver=args.driver, port=args.port,
                        layout=args.layout, gap_m=args.gap, retarget=args.retarget, demo=args.demo,
                        markers=not args.no_markers, trace=args.trace)
    except KeyboardInterrupt:
        print("LIVE_VIEWER_STOPPED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
