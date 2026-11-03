"""只读Encos反馈驱动的MuJoCo方向观察器。

该工具不发送位置或MIT目标；它把Encos角度转换为项目GlovePose，再经过同一套
vE3.3闭链模型和WujiRetarget输出，供人工判断真实电机方向与仿真模型方向。
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
from .real_hand.motor_hand import EncosGloveHardwareDriver
from .mujoco_viewer import (
    _HAND_JOINTS,
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


def run_live_viewer(project_root: str | Path = _DEFAULT_ROOT) -> None:
    root = Path(project_root)
    driver = EncosGloveHardwareDriver(
        hand="right", asset_root=root / "assets" / "hand_motor_ve33",
    )
    pipeline = GloveWujiRetargetPipeline.from_yaml(
        "right",
        asset_root=root / "assets" / "hand_motor_ve33",
        config_path=default_retarget_config(root, "right"),
        estimate_jacobian=False,
    )
    if tuple(pipeline.retargeter.output_joint_names) != _HAND_JOINTS:
        raise RuntimeError("Hand2仿真关节顺序与重定向输出不一致")

    zero = {name: 0.0 for name in GLOVE_JOINTS}
    zero_frame = pipeline.landmarks.step(zero)
    zero_target = pipeline.retargeter.step(
        zero_frame.points_m,
        apply_filter=False,
        model_to_canonical_row=pipeline.landmarks.model_to_canonical_row,
        canonical_to_optimizer_row=pipeline.landmarks.canonical_to_optimizer_row,
    )
    mount_pos, mount_quat, geometry_rms = _glove_to_hand_mount(
        root,
        zero_target,
        zero_frame.points_m,
        pipeline.landmarks.model_to_canonical_row,
    )
    model = _combined_model(root, mount_pos, mount_quat)
    data = mujoco.MjData(model)

    driver.connect()
    found = driver.scan()
    if driver.online_ids != tuple(range(1, 13)):
        driver.close()
        raise RuntimeError(f"要求12轴在线，当前online_ids={driver.online_ids}，scan={found}")

    try:
        print("实时只读MuJoCo观察器已连接Encos；不会发送位置/MIT命令。", flush=True)
        print(f"腕部换基几何残差={geometry_rms * 1000.0:.2f} mm", flush=True)
        print("移动一根真实手指，观察MuJoCo手套和Hand2模型方向；Ctrl+C退出。", flush=True)
        last_report = 0.0
        previous_glove = None
        previous_hand = None
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = model.stat.center
            viewer.cam.distance = 0.55
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = -18.0
            while viewer.is_running():
                state = driver.read_state()
                joints = {name: float(state.pose.joints[name]) for name in GLOVE_JOINTS}
                result = pipeline.step(joints)
                # SDK反馈已完成电机正解；显示直接使用同一真实装配，不再反解q。
                _set_glove_pose(model, data, state.tree_positions_rad)
                _set_hand_qpos(model, data, result.positions_rad)
                mujoco.mj_forward(model, data)
                viewer.sync()

                now = time.monotonic()
                if now - last_report >= 1.0:
                    glove_vector = np.asarray([joints[name] for name in GLOVE_JOINTS], dtype=np.float64)
                    raw_angles = {
                        motor_id: float(values["angle_deg"])
                        for motor_id, values in state.feedback_by_motor.items()
                        if "angle_deg" in values
                    }
                    _print_axis_table(
                        driver,
                        raw_angles,
                        glove_vector,
                        result.positions_rad,
                        previous_glove,
                        previous_hand,
                    )
                    previous_glove = glove_vector
                    previous_hand = result.positions_rad.copy()
                    last_report = now
                time.sleep(0.01)
    finally:
        driver.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encos实时反馈驱动MuJoCo方向观察器")
    parser.add_argument("--project-root", type=Path, default=_DEFAULT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_live_viewer(args.project_root)
    except KeyboardInterrupt:
        print("LIVE_VIEWER_STOPPED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
