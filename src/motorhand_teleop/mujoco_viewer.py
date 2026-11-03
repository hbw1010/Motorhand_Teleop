from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from .domain import GLOVE_JOINTS, RetargetedPose
from .feedback import CalibratedFeedback, CurrentTorqueCalibration
from .retargeting import GloveWujiRetargetPipeline, default_retarget_config
from .safety import CurrentLimits, SafetyLimiter
from .ve33_adapter import Ve33GloveModel


_HAND_MJCF = (
    "vendor/wuji-retargeting/wuji_retargeting/wuji-description/"
    "hand2/hand2_beta1/body/mjcf/right.xml"
)
_GLOVE_MJCF = "assets/hand_motor_ve33/hand_motor_ve33_R/hand_closed_chain.xml"
_HAND_JOINTS = (
    "r_thumb_cmc_flex",
    "r_thumb_cmc_abd",
    "r_thumb_mcp",
    "r_thumb_ip",
    "r_index_finger_mcp_flex",
    "r_index_finger_mcp_abd",
    "r_index_finger_pip",
    "r_index_finger_dip",
    "r_middle_finger_mcp_flex",
    "r_middle_finger_mcp_abd",
    "r_middle_finger_pip",
    "r_middle_finger_dip",
    "r_ring_finger_mcp_flex",
    "r_ring_finger_mcp_abd",
    "r_ring_finger_pip",
    "r_ring_finger_dip",
    "r_pinky_mcp_flex",
    "r_pinky_mcp_abd",
    "r_pinky_pip",
    "r_pinky_dip",
)


def _glove_to_hand_mount(
    root: Path,
    hand_q: np.ndarray,
    glove_points: np.ndarray,
    raw_to_canonical_row: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """把 vE3.3 base_link 按固定模型坐标换到 Hand2 wrist 坐标。
    vE3.3 的 ``canonical_to_model`` 是资产定义的一部分，不能用当前屈伸
    点集重新估计腕部朝向；否则手指屈曲会把整只手套的 wrist 旋转带偏。
    Hand2 MJCF 的 ``r_wrist`` 与当前 Wuji Hand2 重定向 URDF 相差固定的
    Rx(pi) 基座换向（+X 不变、+Y/+Z 取反），当前再按用户指定绕掌心
    指向手指的轴追加 180°，总装配补偿为 +90°（在 +Z 表达下）。
    两项都固定在基座换算中，不用指尖 Kabsch 拟合污染 wrist 方向。
    运动仍完全由 WujiRetarget 输出的 Hand2 关节驱动。
    """
    raw_points = np.asarray(glove_points, dtype=np.float64)
    hand_q = np.asarray(hand_q, dtype=np.float64)
    raw_to_canonical_row = np.asarray(raw_to_canonical_row, dtype=np.float64)
    if (
        raw_points.shape != (21, 3)
        or hand_q.shape != (20,)
        or raw_to_canonical_row.shape != (3, 3)
    ):
        raise ValueError("vE3.3点集必须为21x3，Hand2姿态必须为20维，坐标换基必须为3x3")
    canonical_points = raw_points @ raw_to_canonical_row
    sample = canonical_points[[0, 4, 8, 12, 16, 20]]

    hand_model = mujoco.MjModel.from_xml_path(str(root / _HAND_MJCF))
    hand_data = mujoco.MjData(hand_model)
    for name, value in zip(_HAND_JOINTS, hand_q):
        hand_data.qpos[int(hand_model.joint(name).qposadr[0])] = float(value)
    mujoco.mj_forward(hand_model, hand_data)
    hand_points = np.vstack(
        [
            hand_data.xpos[hand_model.body("r_wrist").id],
            *[
                hand_data.site_xpos[hand_model.site(name).id]
                for name in (
                    "r_thumb_tip",
                    "r_index_finger_tip",
                    "r_middle_finger_tip",
                    "r_ring_finger_tip",
                    "r_pinky_tip",
                )
            ],
        ]
    )
    finger_axis_correction = Rotation.from_rotvec([0.0, 0.0, np.pi / 2.0]).as_matrix()
    canonical_to_hand = finger_axis_correction @ np.diag([1.0, -1.0, -1.0])
    raw_to_hand_column = canonical_to_hand @ raw_to_canonical_row.T
    translation = hand_points[0]
    residual = np.linalg.norm(
        sample @ canonical_to_hand.T + hand_points[0] - hand_points,
        axis=1,
    )
    quat_xyzw = Rotation.from_matrix(raw_to_hand_column).as_quat()
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )
    return translation, quat_wxyz, float(np.sqrt(np.mean(residual**2)))


def _combined_model(
    root: Path,
    mount_pos: np.ndarray,
    mount_quat_wxyz: np.ndarray,
) -> mujoco.MjModel:
    hand_spec = mujoco.MjSpec.from_file(str(root / _HAND_MJCF))
    glove_spec = mujoco.MjSpec.from_file(str(root / _GLOVE_MJCF))
    mount = hand_spec.worldbody.add_frame()
    mount.name = "glove_visual_mount"
    mount.pos = np.asarray(mount_pos, dtype=np.float64).tolist()
    mount.quat = np.asarray(mount_quat_wxyz, dtype=np.float64).tolist()
    hand_spec.attach(glove_spec, prefix="glove_", frame=mount)
    model = hand_spec.compile()
    model.opt.timestep = 0.001
    model.opt.gravity[:] = 0.0
    for geom_id in range(model.ngeom):
        geom_name = model.geom(geom_id).name
        if geom_name and geom_name.startswith("glove_"):
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
    return model

def _joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, names: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [data.qpos[int(model.joint(name).qposadr[0])] for name in names],
        dtype=np.float64,
    )



def _hand_position_actuators(
    model: mujoco.MjModel,
    joint_names: tuple[str, ...],
) -> np.ndarray:
    """按关节语义查找 MuJoCo position actuator，避免把 ctrl 当力矩写入。"""
    actuator_ids: list[int] = []
    for joint_name in joint_names:
        joint_id = int(model.joint(joint_name).id)
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
        if len(matches) != 1:
            raise RuntimeError(f"Hand2关节 {joint_name} 必须恰好对应一个 actuator，实际={matches.tolist()}")
        actuator_ids.append(int(matches[0]))
    return np.asarray(actuator_ids, dtype=np.int32)

def _set_glove_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pose_positions: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    qpos_indices = []
    dof_indices = []
    for name, value in pose_positions.items():
        joint = model.joint(f"glove_{name}")
        qpos_index = int(joint.qposadr[0])
        dof_index = int(joint.dofadr[0])
        data.qpos[qpos_index] = float(value)
        data.qvel[dof_index] = 0.0
        qpos_indices.append(qpos_index)
        dof_indices.append(dof_index)
    return np.asarray(qpos_indices, dtype=np.int32), np.asarray(dof_indices, dtype=np.int32)


def run_viewer(project_root: str | Path) -> None:
    root = Path(project_root)
    pipeline = GloveWujiRetargetPipeline.from_yaml(
        "right",
        asset_root=root / "assets" / "hand_motor_ve33",
        config_path=default_retarget_config(root, "right"),
        estimate_jacobian=True,
    )
    glove_model = Ve33GloveModel("right", asset_root=root / "assets" / "hand_motor_ve33")
    if tuple(pipeline.retargeter.output_joint_names) != _HAND_JOINTS:
        raise RuntimeError("可视化模型关节顺序与官方重定向输出不一致")
    zero = {name: 0.0 for name in GLOVE_JOINTS}
    zero_frame = pipeline.landmarks.step(zero)
    zero_q = pipeline.retargeter.step(
        zero_frame.points_m,
        apply_filter=False,
        model_to_canonical_row=pipeline.landmarks.model_to_canonical_row,
        canonical_to_optimizer_row=pipeline.landmarks.canonical_to_optimizer_row,
    )
    mount_pos, mount_quat, geometry_rms = _glove_to_hand_mount(
        root,
        zero_q,
        zero_frame.points_m,
        pipeline.landmarks.model_to_canonical_row,
    )
    model = _combined_model(root, mount_pos, mount_quat)
    data = mujoco.MjData(model)

    hand_torque_per_amp = 0.50
    glove_kt = 0.035
    feedback_gain = 0.10
    feedback_step = 0.01
    hand_actuators = _hand_position_actuators(model, _HAND_JOINTS)
    hand_position_kp = 10.0
    hand_position_kd = 0.15
    model.actuator_gainprm[hand_actuators, 0] = hand_position_kp
    model.actuator_biasprm[hand_actuators, 1] = -hand_position_kp
    model.actuator_biasprm[hand_actuators, 2] = -hand_position_kd
    index_site = model.site("r_index_finger_tip").id

    calibration = CurrentTorqueCalibration(
        torque_per_amp_nm=np.full(20, hand_torque_per_amp),
        bias_a=np.zeros(20),
        calibrated=True,
        source_id="mujoco-viewer-ideal-effort",
    )
    feedback = CalibratedFeedback(calibration)
    limiter = SafetyLimiter(
        CurrentLimits(
            max_abs_a=np.full(12, 1.5),
            max_slew_a_s=np.full(12, 30.0),
        )
    )

    glove_model.solve(zero)
    glove_pose = glove_model.evaluate_raw(zero)
    _set_glove_pose(model, data, glove_pose.all_positions)
    q_target = zero_q.copy()
    # Jacobian 只在启动时计算一次；循环内重复求有限差分会阻塞 viewer，导致仿真时间几乎不前进。
    retarget_jacobian = pipeline.local_jacobian(zero, step_rad=feedback_step)
    data.qpos[[int(model.joint(name).qposadr[0]) for name in _HAND_JOINTS]] = q_target
    mujoco.mj_forward(model, data)

    print("MuJoCo 双模型可视化已启动：官方 Hand2 + 真实 vE3.3 闭链模型。", flush=True)
    print(f"已应用固定腕部换基，零位几何残差 RMS={geometry_rms * 1000.0:.2f} mm。", flush=True)
    print("两套模型已按腕部基座重合显示；每 4 秒注入一次 1 N 指尖外力。", flush=True)
    start = time.monotonic()
    next_command = start
    physics_steps_per_frame = 100
    last_report = start
    max_feedback_current = 0.0
    qg_target = np.zeros(12, dtype=np.float64)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = model.stat.center
        viewer.cam.distance = 0.55
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -18.0
        while viewer.is_running():
            now = time.monotonic()
            elapsed = now - start
            if now >= next_command:
                phase = 0.5 * (1.0 - np.cos(2.0 * np.pi * ((elapsed % 8.0) / 8.0)))
                joints = {name: 0.0 for name in GLOVE_JOINTS}
                joints["index_mcp_flex"] = 0.28 * phase
                joints["index_pip"] = 1.00 * phase
                joints["middle_mcp_flex"] = 0.22 * phase
                joints["middle_pip"] = 0.78 * phase
                joints["ring_mcp_flex"] = 0.18 * phase
                joints["ring_pip"] = 0.62 * phase
                joints["pinky_mcp_flex"] = 0.14 * phase
                joints["pinky_pip"] = 0.50 * phase
                qg_target = np.asarray([joints[name] for name in GLOVE_JOINTS])
                q_target = pipeline.step(joints).positions_rad
                glove_model.solve(joints)
                next_glove_pose = glove_model.evaluate_raw(joints)
                _set_glove_pose(model, data, next_glove_pose.all_positions)
                next_command = now + 0.1

            data.ctrl[hand_actuators] = np.clip(
                q_target,
                model.actuator_ctrlrange[hand_actuators, 0],
                model.actuator_ctrlrange[hand_actuators, 1],
            )
            data.qfrc_applied[:] = 0.0
            mujoco.mj_forward(model, data)
            force = np.zeros(3)
            if 0.6 <= (elapsed % 4.0) <= 1.2:
                force[2] = 1.0
                point = data.site_xpos[index_site].copy()
                body_id = int(model.site_bodyid[index_site])
                mujoco.mj_applyFT(model, data, force, np.zeros(3), point, body_id, data.qfrc_applied)

            external_tau = data.qfrc_applied[:20].copy()
            effort_a = external_tau / hand_torque_per_amp
            glove_torque = feedback.project_effort(
                RetargetedPose(q_target, retarget_jacobian), effort_a
            ).glove_torque_nm
            glove_torque *= feedback_gain
            transmission = pipeline.landmarks.model.transmission()
            candidate_current = transmission.motor_torque_to_current(
                transmission.joint_torque_to_motor_torque(qg_target, glove_torque),
                np.full(12, glove_kt),
            )
            glove_current = limiter.apply(candidate_current, dt_s=model.opt.timestep)

            visual_joints = dict(
                zip(GLOVE_JOINTS, qg_target + np.clip(glove_torque / 3.0, -0.15, 0.15))
            )
            visual_pose = glove_model.evaluate_raw(visual_joints)
            glove_qpos_indices, glove_dof_indices = _set_glove_pose(
                model, data, visual_pose.all_positions
            )
            glove_qpos_values = np.asarray(
                list(visual_pose.all_positions.values()),
                dtype=np.float64,
            )
            for _ in range(physics_steps_per_frame):
                mujoco.mj_step(model, data)
                # vE3.3 在这里是视觉闭链，不参与 Hand2 动力学；每个子步锁回输入姿态，
                # 避免批量步进把闭链约束积分发散。
                data.qpos[glove_qpos_indices] = glove_qpos_values
                data.qvel[glove_dof_indices] = 0.0
                mujoco.mj_forward(model, data)
            viewer.sync()
            max_feedback_current = max(max_feedback_current, float(np.max(np.abs(glove_current))))
            if now - last_report >= 1.0:
                hand_motion = np.linalg.norm(_joint_qpos(model, data, _HAND_JOINTS) - zero_q)
                target_motion = np.linalg.norm(q_target - zero_q)
                hand_actuator_force = float(np.max(np.abs(data.qfrc_actuator[:20])))
                print(
                    f"t={elapsed:5.1f}s(sim={data.time:5.2f}s)  外力={'1.0N' if force[2] else '0N':>4}  "
                    f"Hand2目标/实际位移={target_motion:5.3f}/{hand_motion:5.3f}rad  "
                    f"控制力峰值={hand_actuator_force:5.3f}Nm  "
                    f"反馈峰值={max_feedback_current:5.3f}A  "
                    f"视觉偏转={np.linalg.norm(glove_torque / 3.0):5.3f}rad",
                    flush=True,
                )
                last_report = now


def main() -> None:
    run_viewer(Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    main()
