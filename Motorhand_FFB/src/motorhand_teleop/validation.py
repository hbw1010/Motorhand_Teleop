from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .domain import GLOVE_JOINTS, RetargetedPose
from .feedback import CalibratedFeedback, CurrentTorqueCalibration
from .landmarks import Ve33LandmarkSource
from .retargeting import WujiHand2Retargeter, default_retarget_config, wuji_hand2_joint_names
from .safety import CurrentLimits, SafetyLimiter


def validate_pose_mapping(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root)
    reports: dict[str, object] = {}
    for hand in ("right", "left"):
        source = Ve33LandmarkSource(hand, asset_root=root.parent / "assets" / "hand_motor_ve33")
        retargeter = WujiHand2Retargeter.from_yaml(
            default_retarget_config(root, hand), hand=hand
        )
        zero = {name: 0.0 for name in GLOVE_JOINTS}
        curl = dict(zero)
        curl["index_pip"] = 0.30
        curl["middle_pip"] = 0.20
        zero_frame = source.step(zero)
        curl_frame = source.step(curl)
        zero_q = retargeter.step(
            zero_frame.points_m,
            model_to_canonical_row=source.model_to_canonical_row,
            canonical_to_optimizer_row=source.canonical_to_optimizer_row,
        )
        curl_q = retargeter.step(
            curl_frame.points_m,
            model_to_canonical_row=source.model_to_canonical_row,
            canonical_to_optimizer_row=source.canonical_to_optimizer_row,
        )
        report = retargeter.diagnose(
            curl_frame.points_m,
            model_to_canonical_row=source.model_to_canonical_row,
            canonical_to_optimizer_row=source.canonical_to_optimizer_row,
        )
        limits = retargeter.joint_limits
        if retargeter.output_joint_names != wuji_hand2_joint_names(hand):
            raise AssertionError(f"{hand} Hand2语义关节顺序不一致")
        if report["tip_error_max_cm"] > 5.0 or report["full_hand_error_max_cm"] > 5.0:
            raise AssertionError(f"{hand}姿态任务点误差超过5 cm: {report}")
        if report["joint_limit_margin_rad"] < -1e-5:
            raise AssertionError(f"{hand}输出超出关节限位: {report}")
        if not np.all((curl_q >= limits[:, 0] - 1e-5) & (curl_q <= limits[:, 1] + 1e-5)):
            raise AssertionError(f"{hand}关节限位检查失败")
        if np.linalg.norm(curl_q - zero_q) < 0.02:
            raise AssertionError(f"{hand}屈曲输入没有产生可观测姿态变化")
        if len(report["mcp_abd_rad"]) != 4 or not np.allclose(report["mcp_abd_rad"], 0.0, atol=1e-9):
            raise AssertionError(f"{hand}侧摆未按固定零位约束: {report['mcp_abd_rad']}")
        reports[hand] = {
            "zero_q_rad": zero_q.tolist(),
            "curl_q_rad": curl_q.tolist(),
            "curl_diagnostics": report,
            "lateral_policy": "fixed_zero",
        }

    feedback = CalibratedFeedback(
        CurrentTorqueCalibration(
            torque_per_amp_nm=np.ones(20),
            bias_a=np.zeros(20),
            calibrated=True,
            source_id="strict-validation",
        )
    )
    feedback_jacobian = np.zeros((20, 12))
    feedback_jacobian[0, 0] = 2.0
    feedback_pose = RetargetedPose(np.zeros(20), feedback_jacobian)
    positive = feedback.project_effort(feedback_pose, np.r_[1.0, np.zeros(19)]).glove_torque_nm
    negative = feedback.project_effort(feedback_pose, np.r_[-1.0, np.zeros(19)]).glove_torque_nm
    zero = feedback.project_effort(feedback_pose, np.zeros(20)).glove_torque_nm
    if not np.allclose(positive, -negative) or not np.allclose(zero, 0.0):
        raise AssertionError("力反馈转置投影的零输入或符号检查失败")
    reports["feedback_projection"] = {
        "positive_glove_torque_nm": positive.tolist(),
        "negative_glove_torque_nm": negative.tolist(),
        "zero_input_is_zero": True,
    }
    limiter = SafetyLimiter(
        CurrentLimits(
            max_abs_a=np.full(12, 1.5),
            max_slew_a_s=np.full(12, 30.0),
        )
    )
    first = limiter.apply(np.full(12, 10.0), dt_s=0.001)
    if not np.allclose(first, 0.03):
        raise AssertionError(f"斜率限制未生效: {first}")
    limited = limiter.apply(np.full(12, 10.0), dt_s=1.0)
    if not np.allclose(limited, 1.5):
        raise AssertionError(f"电流限幅未生效: {limited}")
    limiter.trip()
    try:
        limiter.apply(np.zeros(12), dt_s=0.001)
    except RuntimeError:
        pass
    else:
        raise AssertionError("故障锁存后仍允许输出电流")
    reports["safety"] = {
        "slew_first_a": first.tolist(),
        "limited_a": limited.tolist(),
        "trip_lockout": True,
    }
    return reports


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    reports = validate_pose_mapping(root)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
