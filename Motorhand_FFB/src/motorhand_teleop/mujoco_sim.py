from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import mujoco
import numpy as np
import yaml

from .domain import GLOVE_JOINTS, RetargetedPose
from .feedback import CalibratedFeedback, CurrentTorqueCalibration
from .retargeting import GloveWujiRetargetPipeline, default_retarget_config
from .safety import CurrentLimits, SafetyLimiter


_FINGER_SPECS = (
    ("index", (-0.030, 0.000, 0.020)),
    ("middle", (-0.010, 0.000, 0.022)),
    ("ring", (0.012, 0.000, 0.020)),
    ("pinky", (0.034, 0.000, 0.015)),
)


def _finger_prefix(finger: str) -> str:
    return f"{finger}_finger" if finger != "pinky" else "pinky"


def _hand_joint_names() -> tuple[str, ...]:
    names = ["r_thumb_cmc_flex", "r_thumb_cmc_abd", "r_thumb_mcp", "r_thumb_ip"]
    for finger, _ in _FINGER_SPECS:
        prefix = _finger_prefix(finger)
        names.extend(
            (
                f"r_{prefix}_mcp_flex",
                f"r_{prefix}_mcp_abd",
                f"r_{prefix}_pip",
                f"r_{prefix}_dip",
            )
        )
    return tuple(names)




def build_glove_mjcf() -> str:
    bodies = []
    actuators = []
    for index, name in enumerate(GLOVE_JOINTS):
        body_name = f"glove_{index:02d}"
        x = 0.02 * (index % 6) - 0.05
        y = 0.02 * (index // 6)
        bodies.append(
            f'<body name="{body_name}" pos="{x} {y} 0">'
            f'<joint name="{name}" type="hinge" axis="1 0 0" '
            'range="-2 2" limited="true" damping="0.08"/>'
            '<geom type="capsule" fromto="0 0 0 0 0 -0.025" size="0.006" mass="0.01"/>'
            '</body>'
        )
        actuators.append(f'<motor name="glove_act_{index:02d}" joint="{name}" gear="1"/>')
    body_xml = "".join(bodies)
    actuator_xml = "".join(actuators)
    return f"""<mujoco model="ve33_glove_sim">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.001" integrator="implicitfast" gravity="0 0 0"/>
  <worldbody><body name="glove_base">{body_xml}</body></worldbody>
  <actuator>{actuator_xml}</actuator>
</mujoco>"""


def _joint_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    return np.asarray([data.qpos[model.jnt_qposadr[i]] for i in range(model.njnt)], dtype=np.float64)


def _joint_qvel(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    return np.asarray([data.qvel[model.jnt_dofadr[i]] for i in range(model.njnt)], dtype=np.float64)


def _load_config(path: str | Path) -> Mapping[str, object]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("仿真配置必须是对象")
    return config


def _rms(values: list[float]) -> float:
    return float(np.sqrt(np.mean(values))) if values else 0.0


def run_force_feedback_simulation(project_root: str | Path, config_path: str | Path) -> dict[str, object]:
    root = Path(project_root)
    config = _load_config(config_path)
    control = config["control"]
    disturbance = config["disturbance"]
    sim = config["simulation"]
    dt = float(sim["dt_s"])
    duration = float(sim["duration_s"])
    steps = int(round(duration / dt))
    command_stride = max(1, round(1.0 / (float(control["retarget_hz"]) * dt)))
    feedback_stride = max(1, round(1.0 / (float(control["feedback_jacobian_hz"]) * dt)))

    pipeline = GloveWujiRetargetPipeline.from_yaml(
        "right",
        asset_root=root.parent / "assets" / "hand_motor_ve33",
        config_path=default_retarget_config(root, "right"),
        estimate_jacobian=True,
    )
    hand_mjcf = root / str(config["model"]["mjcf_path"])
    if not hand_mjcf.is_file():
        raise FileNotFoundError(f"官方 Hand2 MJCF 不存在: {hand_mjcf}")
    hand_model = mujoco.MjModel.from_xml_path(str(hand_mjcf))
    hand_model.opt.timestep = dt
    if bool(sim.get("disable_gravity", False)):
        hand_model.opt.gravity[:] = 0.0
    hand_data = mujoco.MjData(hand_model)
    glove_model = mujoco.MjModel.from_xml_string(build_glove_mjcf())
    glove_model.opt.timestep = dt
    glove_data = mujoco.MjData(glove_model)
    hand_names = _hand_joint_names()
    if hand_names != tuple(pipeline.retargeter.output_joint_names):
        raise AssertionError("仿真 Hand2 关节顺序与重定向输出不一致")

    hand_torque_per_amp = float(sim["hand_torque_per_amp_nm"])
    glove_kt = np.full(12, float(sim["glove_torque_constant_nm_a"]), dtype=np.float64)
    feedback = CalibratedFeedback(
        CurrentTorqueCalibration(
            torque_per_amp_nm=np.full(20, hand_torque_per_amp),
            bias_a=np.zeros(20),
            calibrated=True,
            source_id="mujoco_ground_truth_disturbance_proxy",
        )
    )
    limiter = SafetyLimiter(
        CurrentLimits(
            max_abs_a=np.full(12, float(sim["glove_current_limit_a"])),
            max_slew_a_s=np.full(12, float(sim["glove_current_slew_a_s"])),
        )
    )
    transmission = pipeline.landmarks.model.transmission()
    index_site = hand_model.site("r_index_finger_tip").id
    q_target = np.zeros(20, dtype=np.float64)
    retarget_jacobian = np.zeros((20, 12), dtype=np.float64)
    qg_amplitude = np.zeros(12, dtype=np.float64)
    qg_amplitude[GLOVE_JOINTS.index("index_pip")] = 0.15
    qg_amplitude[GLOVE_JOINTS.index("middle_pip")] = 0.10
    initial_joints = {name: 0.0 for name in GLOVE_JOINTS}
    feedback_jacobian_step_rad = float(sim["feedback_jacobian_step_rad"])
    q_target = pipeline.step(initial_joints).positions_rad.copy()
    retarget_jacobian = pipeline.local_jacobian(
        initial_joints,
        step_rad=feedback_jacobian_step_rad,
    )
    transmission_jacobian = transmission.motor_jacobian(np.zeros(12, dtype=np.float64))
    hand_data.qpos[:] = q_target
    mujoco.mj_forward(hand_model, hand_data)
    metrics: dict[str, list[float]] = {
        "tracking_mse": [],
        "feedback_current_norm": [],
        "feedback_current_channel_peak": [],
        "glove_deflection_norm": [],
        "external_tau_norm": [],
        "candidate_current_norm": [],
        "candidate_current_channel_peak": [],
    }
    contact_currents: list[float] = []
    contact_channel_peaks: list[float] = []
    contact_deflections: list[float] = []
    stable = True

    for step in range(steps):
        t = step * dt
        phase = 0.5 * (1.0 - np.cos(2.0 * np.pi * min(t / duration, 1.0)))
        qg_now = qg_amplitude * phase
        joints = dict(zip(GLOVE_JOINTS, qg_now))
        if step % command_stride == 0 and step != 0:
            q_target = pipeline.step(joints).positions_rad.copy()
        if step != 0 and step % feedback_stride == 0:
            retarget_jacobian = pipeline.local_jacobian(
                joints,
                step_rad=feedback_jacobian_step_rad,
            )
            transmission_jacobian = transmission.motor_jacobian(qg_now)

        hand_q = _joint_qpos(hand_model, hand_data)
        hand_dq = _joint_qvel(hand_model, hand_data)
        kp = float(control["kp"])
        kd = float(control["kd"])
        effort_limit_nm = float(control["effort_limit_a"]) * hand_torque_per_amp
        hand_data.ctrl[:] = np.clip(kp * (q_target - hand_q) - kd * hand_dq, -effort_limit_nm, effort_limit_nm)
        hand_data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(hand_model, hand_data)
        force = np.zeros(3, dtype=np.float64)
        if float(disturbance["start_s"]) <= t <= float(disturbance["end_s"]):
            force[2] = float(disturbance["force_n"])
        if np.any(force):
            point = hand_data.site_xpos[index_site].copy()
            body_id = int(hand_model.site_bodyid[index_site])
            mujoco.mj_applyFT(hand_model, hand_data, force, np.zeros(3), point, body_id, hand_data.qfrc_applied)
        external_tau = hand_data.qfrc_applied[:20].copy()
        effort_a = external_tau / hand_torque_per_amp
        pose = RetargetedPose(q_target, retarget_jacobian)
        glove_torque = feedback.project_effort(pose, effort_a).glove_torque_nm
        glove_torque *= float(sim["feedback_gain"])
        motor_torque = transmission.joint_torque_to_motor_torque_from_jacobian(
            transmission_jacobian,
            glove_torque,
        )
        candidate_current = transmission.motor_torque_to_current(motor_torque, glove_kt)
        glove_current = limiter.apply(candidate_current, dt_s=dt)

        mujoco.mj_step(hand_model, hand_data)
        glove_q = _joint_qpos(glove_model, glove_data)
        glove_dq = _joint_qvel(glove_model, glove_data)
        glove_data.ctrl[:] = glove_current * glove_kt
        glove_data.qfrc_applied[:] = (
            -float(sim["human_stiffness_nm_rad"]) * (glove_q - qg_now)
            - float(sim["human_damping_nm_s_rad"]) * glove_dq
        )
        mujoco.mj_step(glove_model, glove_data)

        tracking_error = _joint_qpos(hand_model, hand_data) - q_target
        deflection = _joint_qpos(glove_model, glove_data) - qg_now
        metrics["feedback_current_norm"].append(float(np.linalg.norm(glove_current)))
        metrics["feedback_current_channel_peak"].append(float(np.max(np.abs(glove_current))))
        metrics["glove_deflection_norm"].append(float(np.linalg.norm(deflection)))
        metrics["external_tau_norm"].append(float(np.linalg.norm(external_tau)))
        metrics["candidate_current_norm"].append(float(np.linalg.norm(candidate_current)))
        metrics["candidate_current_channel_peak"].append(float(np.max(np.abs(candidate_current))))
        in_contact = float(disturbance["start_s"]) <= t <= float(disturbance["end_s"])
        if in_contact:
            contact_currents.append(float(np.linalg.norm(glove_current)))
            contact_channel_peaks.append(float(np.max(np.abs(glove_current))))
            contact_deflections.append(float(np.linalg.norm(deflection)))
        stable &= bool(np.all(np.isfinite(hand_data.qpos))) and bool(np.all(np.isfinite(glove_data.qpos)))
        stable &= bool(np.max(np.abs(glove_current)) <= float(sim["glove_current_limit_a"]) + 1e-9)
        stable &= bool(np.max(np.abs(hand_data.qvel)) < 100.0)

    contact_current_peak = max(contact_currents, default=0.0)
    contact_deflection_peak = max(contact_deflections, default=0.0)
    force_feedback_effect_detected = contact_current_peak > 1e-4 and contact_deflection_peak > 1e-7
    result = {
        "stable": bool(stable and force_feedback_effect_detected),
        "force_feedback_effect_detected": force_feedback_effect_detected,
        "steps": steps,
        "dt_s": dt,
        "official_control": {
            "kp": float(control["kp"]),
            "kd": float(control["kd"]),
            "effort_limit_a": float(control["effort_limit_a"]),
            "command_rate_hz": float(control["command_rate_hz"]),
        },
        "simulation_assumptions": {
            "hand_torque_per_amp_nm": hand_torque_per_amp,
            "glove_torque_constant_nm_a": float(sim["glove_torque_constant_nm_a"]),
            "feedback_gain": float(sim["feedback_gain"]),
            "external_force_n": float(disturbance["force_n"]),
            "gravity_disabled": bool(sim.get("disable_gravity", False)),
        },
        "tracking_rmse_rad": _rms(metrics["tracking_mse"]),
        "external_tau_peak_nm": max(metrics["external_tau_norm"], default=0.0),
        "candidate_current_peak_a": max(metrics["candidate_current_channel_peak"], default=0.0),
        "candidate_current_norm_peak_a": max(metrics["candidate_current_norm"], default=0.0),
        "feedback_current_peak_a": max(metrics["feedback_current_channel_peak"], default=0.0),
        "feedback_current_norm_peak_a": max(metrics["feedback_current_norm"], default=0.0),
        "feedback_current_rms_a": _rms([x * x for x in metrics["feedback_current_norm"]]),
        "contact_feedback_current_peak_a": max(contact_channel_peaks, default=0.0),
        "contact_feedback_current_norm_peak_a": contact_current_peak,
        "contact_glove_deflection_rms_rad": _rms([x * x for x in contact_deflections]),
        "contact_glove_deflection_peak_rad": contact_deflection_peak,
    }
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    result = run_force_feedback_simulation(root, root / "config" / "simulation.yaml")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["stable"]:
        raise SystemExit("MuJoCo 力反馈仿真未通过稳定性或安全限幅检查")


if __name__ == "__main__":
    main()
