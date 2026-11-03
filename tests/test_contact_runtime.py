"""真实CAD／官方Retarget／配对运行器的接触阻抗消费者回归；不连接硬件。"""
from dataclasses import replace
from pathlib import Path
import socket

import numpy as np
import pytest

from wujihand2_force_glove.domain import GLOVE_JOINTS
from wujihand2_force_glove.real_hand.contact_feedback import ContactFeedbackConfig
from wujihand2_force_glove.real_hand.curve_telemetry import CurvePublisher, decode_sample
from wujihand2_force_glove.real_hand.force_feedback import ForceFeedbackController
from wujihand2_force_glove.real_hand.force_linear import MaxTorqueLinearConfig
from wujihand2_force_glove.real_hand.runtime import (
    FollowMode, PairRuntime, RetargetPipelineFactory, create_retargeter,
)
from test_real_hand_runtime import Clock, EffortHandPlant, ForceMotorPlant, MotorPlant

ROOT = Path(__file__).resolve().parents[1]


def human_joints(amount):
    joints = dict.fromkeys(GLOVE_JOINTS, 0.0)
    joints["index_pip"] = amount
    joints["middle_pip"] = amount * 0.5
    return joints


class ContactHandPlant(EffortHandPlant):
    """显式数值设备：位置、速度、电流分开观测，发送目标不回显为状态。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.velocity = np.zeros(20)
        self.faults = ()

    def read_state(self):
        return replace(
            super().read_state(), velocities_rad_s=self.velocity.copy(),
            diagnostics={"faults": self.faults},
        )


def contact_config():
    return ContactFeedbackConfig(
        schema_version=1, estimate_acknowledged=True,
        source_id="synthetic-consumer-only__not-calibrated",
        current_on_a=np.full(20, 0.12), current_off_a=np.full(20, 0.06),
        error_on_rad=np.full(20, 0.10), error_off_rad=np.full(20, 0.03),
        stalled_speed_rad_s=np.full(20, 0.2),
        stiffness_nm_per_rad=np.full(20, 0.10),
        damping_nm_s_per_rad=np.full(20, 0.003),
    )


def create_contact_pair(*, clock=None, glove_type=ForceMotorPlant, realtime=False,
                        contact_parameters=None, limit_parameters=None):
    """复用既有真实数值设备模式；烟测可注入有动力学的手套消费者。"""
    clock = Clock() if clock is None else clock
    bridge = create_retargeter(project_root=ROOT)
    config = limit_parameters
    if config is None:
        # 永久回归使用自有工况，不随现场操作者调节限额而改变验收条件。
        config = MaxTorqueLinearConfig(
            schema_version=1, mode="max_torque_linear_estimate", estimate_acknowledged=True,
            source_id="synthetic-consumer-limits-only", wuji_serial_number="explicit-test-SN",
            glove_hand="right", wuji_max_current_a=np.full(20, 2.0),
            wuji_current_bias_a=np.zeros(20), wuji_effort_sign=np.ones(20),
            glove_max_output_torque_nm=np.full(12, 0.02),
            max_torque_slew_nm_s=np.full(12, 0.05), gain=1.0,
        )
    # 虚拟时钟回归冻结几何刷新，仅隔离后台墙钟调度；墙钟烟测保留正式TTL。
    options = {} if realtime else {"jacobian_max_age_s": 10.0, "jacobian_refresh_interval_s": 100.0}
    config = replace(config, wuji_serial_number="explicit-test-SN", **options)
    positions = bridge.glove_to_hand2(human_joints(0.0))[0].positions_rad.copy()
    robot = bridge.forward.retargeter
    hand = ContactHandPlant(
        clock, positions, robot._core.optimizer.robot.joint_limits[robot._output_indices],
    )
    glove = glove_type(MotorPlant(clock, bridge.forward.landmarks.model, human_joints(0.0)), config)
    controller = ForceFeedbackController(
        config, bridge, glove, RetargetPipelineFactory(project_root=ROOT),
        clock_ns=clock, contact_config=contact_config() if contact_parameters is None else contact_parameters,
    )
    runtime = PairRuntime(
        hand, glove, bridge, execute=True, clock_ns=clock,
        force_feedback=controller, transition_s=0.1,
    )
    runtime.connect(serial_number="explicit-test-SN")
    return runtime, hand, glove, clock


@pytest.fixture
def contact_pair():
    pair = create_contact_pair()
    try:
        pair[0].start()
        yield pair
    finally:
        pair[0].close()


def set_human(glove, amount):
    _, motors = glove.model.evaluate(human_joints(amount))
    glove.plant.motors = dict(motors)


def advance_steps(pair, count):
    runtime, _, _, clock = pair
    ticks = []
    for _ in range(count):
        clock.advance(0.02)
        ticks.append(runtime.step())
    return ticks


def assert_zero_feedback(tick):
    feedback = tick.force_feedback
    assert feedback.mapping_mode == "contact_impedance_estimate"
    assert feedback.wuji_actuator_torque_nm is None
    assert feedback.glove_joint_torque_nm is None
    assert feedback.model_current_a is None
    np.testing.assert_array_equal(feedback.virtual_hand_torque_nm, np.zeros(20))
    np.testing.assert_array_equal(feedback.model_torque_nm, np.zeros(12))
    if feedback.projection_status == "active":
        np.testing.assert_array_equal(feedback.projected_model_torque_nm, np.zeros(12))
    else:
        assert feedback.projected_model_torque_nm is None


def test_contact_pair_free_blocked_release_keeps_original_position_master(contact_pair):
    runtime, hand, glove, clock = contact_pair
    first = runtime.step()
    assert_zero_feedback(first)
    held_motor_reference = dict(glove.plant.reference)

    # 大误差但无负载不能变成接触。数值手被显式挡住，未更新到发出的目标。
    set_human(glove, 0.20)
    free = advance_steps(contact_pair, 18)
    assert all(not np.any(t.force_feedback.contact_active_mask) for t in free)
    for tick in free:
        assert_zero_feedback(tick)
    assert np.max(np.abs(free[-1].output_positions_rad - hand.positions)) > 0.10

    # 负载是数值设备的独立观测，不是目标回显；驻留前不得输出，驻留后才反射。
    hand.effort[:] = 0.8
    loaded = advance_steps(contact_pair, 24)
    assert_zero_feedback(loaded[0])
    contact = [t for t in loaded if np.any(t.force_feedback.contact_active_mask)]
    assert contact and any(t.force_feedback.contact_state == "contact" for t in contact)
    assert max(np.max(np.abs(t.force_feedback.model_torque_nm)) for t in contact) > 1e-5
    for tick in loaded:
        np.testing.assert_allclose(tick.output_positions_rad, tick.candidate_positions_rad, atol=1e-12)
        np.testing.assert_allclose(tick.force_feedback.wuji_load_fraction, hand.effort / 2.0)
        np.testing.assert_array_equal(tick.output_positions_rad[[5, 9, 13, 17]], np.zeros(4))
        assert not np.any(tick.force_feedback.contact_active_mask[[5, 9, 13, 17]])
        assert np.max(np.abs(tick.force_feedback.model_torque_nm)) <= 0.02
    assert np.max(np.abs(hand.reference - hand.positions)) > 0.10
    assert glove.commands == [] and glove.force_commands == []
    assert glove.plant.reference == held_motor_reference and glove.homes == 1

    # 操作者主动回开，实际手仍被挡在原处：跨过原接触面立即撤力，不让斜率历史拉人。
    set_human(glove, 0.0)
    released = advance_steps(contact_pair, 8)
    for tick in released:
        assert_zero_feedback(tick)
    assert released[-1].force_feedback.contact_state == "free"
    assert not np.any(released[-1].force_feedback.contact_active_mask)
    assert glove.homes == 1 and glove.commands == []
    assert all(deadline - tick.hand2_state.received_ns == 100_000_000
               for tick, (_, deadline) in zip([first, *free, *loaded, *released], hand.commands))


def test_high_current_alone_never_becomes_contact(contact_pair):
    runtime, hand, glove, _ = contact_pair
    runtime.step()
    hand.effort[:] = 1.5
    ticks = advance_steps(contact_pair, 24)
    assert max(np.max(np.abs(t.force_feedback.wuji_load_fraction)) for t in ticks) > 0.5
    for tick in ticks:
        assert_zero_feedback(tick)
        assert not np.any(tick.force_feedback.contact_active_mask)
    assert glove.commands == [] and glove.homes == 1


def test_pinky_small_displacement_and_sustained_load_reach_glove_motor():
    config = replace(
        contact_config(), error_on_rad=np.full(20, 0.06),
        dwell_s=0.04, reflection_gain=6000.0,
    )
    pair = create_contact_pair(contact_parameters=config)
    runtime, hand, glove, _ = pair
    try:
        runtime.start()
        runtime.step()
        joints = dict.fromkeys(GLOVE_JOINTS, 0.0)
        joints["pinky_pip"] = 0.09
        _, motors = glove.model.evaluate(joints)
        glove.plant.motors = dict(motors)
        hand.effort[18] = 1.2

        tick = advance_steps(pair, 40)[-1]
        error = tick.output_positions_rad[18] - tick.hand2_state.positions_rad[18]
        assert 0.06 < error < 0.10
        assert tick.force_feedback.contact_active_mask[18]
        motor_name = glove.model.joint_motor_names["pinky_pip"]
        motor_index = glove.model.motor_names.index(motor_name)
        assert abs(tick.force_feedback.model_torque_nm[motor_index]) > 1e-5
    finally:
        runtime.close()


def test_transition_uses_sent_target_not_unblended_forward_candidate(contact_pair):
    runtime, hand, glove, _ = contact_pair
    set_human(glove, 0.20)
    hand.effort[:] = 0.8
    first = runtime.step()
    assert np.max(np.abs(first.candidate_positions_rad - hand.positions)) > 0.10
    np.testing.assert_array_equal(first.output_positions_rad, hand.positions)
    assert_zero_feedback(first)
    early = advance_steps(contact_pair, 4)
    for tick in early:
        assert_zero_feedback(tick)
    assert np.max(np.abs(early[-1].output_positions_rad - first.output_positions_rad)) > 0.01
    assert glove.commands == [] and glove.homes == 1


@pytest.mark.parametrize("failure", ["hand_stale", "glove_stale", "pose_stale", "fault", "offline", "preflight"])
def test_contact_mode_rejects_original_stale_fault_or_invalid_frame_before_either_send(contact_pair, failure):
    runtime, hand, glove, _ = contact_pair
    runtime.step()
    if failure == "hand_stale":
        hand.stamp_offset = -100_000_001
    elif failure == "glove_stale":
        glove.plant.stamp_offset = -100_000_001
    elif failure == "pose_stale":
        glove.plant.pose_stamp_offset = -100_000_001
    elif failure == "fault":
        hand.faults = ("synthetic-motor-fault",)
    elif failure == "offline":
        hand.online = 19
    else:
        glove.plant.fail_validation = True
    sent = len(hand.commands), len(glove.force_torque_commands)
    with pytest.raises((TimeoutError, ValueError, RuntimeError)):
        runtime.step()
    assert (len(hand.commands), len(glove.force_torque_commands)) == sent
    assert hand.disabled == glove.disabled == 1
    assert glove.commands == [] and glove.homes == 1


def test_contact_slow_forward_solver_does_not_renew_original_100ms_deadline(contact_pair, monkeypatch):
    runtime, hand, glove, clock = contact_pair
    runtime.step()
    solve = runtime.retargeter.glove_to_hand2

    def slow_original_solver(*args, **kwargs):
        result = solve(*args, **kwargs)
        clock.advance(0.100000001)
        return result

    monkeypatch.setattr(runtime.retargeter, "glove_to_hand2", slow_original_solver)
    sent = len(hand.commands), len(glove.force_torque_commands)
    with pytest.raises(TimeoutError):
        runtime.step()
    assert (len(hand.commands), len(glove.force_torque_commands)) == sent
    assert runtime.command_timeout_ns == 100_000_000
    assert hand.disabled == glove.disabled == 1


def test_contact_pose_refresh_stops_old_reaction_without_stopping_position_master(contact_pair):
    runtime, hand, glove, _ = contact_pair
    runtime.step()
    set_human(glove, 0.20)
    hand.effort[:] = 0.8
    loaded = advance_steps(contact_pair, 28)
    assert any(np.linalg.norm(t.force_feedback.model_torque_nm) > 1e-5 for t in loaded)
    previous_hand_target = hand.commands[-1][0].positions_rad.copy()
    held_reference = dict(glove.plant.reference)
    # 上一有效局部Jac位于零姿态。真正改变CAD输入，令其超出既有姿态半径。
    set_human(glove, 0.45)
    tick = advance_steps(contact_pair, 1)[0]
    assert tick.force_feedback.projection_status == "refreshing_pose"
    assert tick.force_feedback.contact_state == "waiting_geometry"
    np.testing.assert_array_equal(tick.force_feedback.model_torque_nm, np.zeros(12))
    assert tick.force_feedback.projected_model_torque_nm is None
    assert not np.allclose(
        hand.commands[-1][0].positions_rad, previous_hand_target, rtol=0, atol=1e-4,
    )
    assert glove.commands == [] and glove.homes == 1
    assert glove.plant.reference == held_reference
    assert hand.disabled == glove.disabled == 0


@pytest.mark.parametrize("reflection_gain", [1.0, 1000.0])
def test_reflection_strength_reaches_cap_only_after_contact_and_releases(reflection_gain):
    limits = MaxTorqueLinearConfig(
        schema_version=1, mode="max_torque_linear_estimate", estimate_acknowledged=True,
        source_id="synthetic-strength-limits", wuji_serial_number="explicit-test-SN",
        glove_hand="right", wuji_max_current_a=np.full(20, 2.0),
        wuji_current_bias_a=np.zeros(20), wuji_effort_sign=np.ones(20),
        glove_max_output_torque_nm=np.full(12, 5.0),
        max_torque_slew_nm_s=np.full(12, 1.0), gain=1.0,
    )
    pair = create_contact_pair(
        limit_parameters=limits,
        contact_parameters=replace(contact_config(), reflection_gain=reflection_gain),
    )
    runtime, hand, glove, _ = pair
    try:
        runtime.start()
        runtime.step()
        set_human(glove, 0.20)
        for tick in advance_steps(pair, 18):
            assert_zero_feedback(tick)
        hand.effort[:] = 0.8
        loaded = advance_steps(pair, 320)
        output = np.array([t.force_feedback.model_torque_nm for t in loaded])
        peak = float(np.max(np.abs(output)))
        projected = np.array([t.force_feedback.projected_model_torque_nm for t in loaded])
        assert all(t.force_feedback.contact_reflection_gain == limits.gain * reflection_gain
                   for t in loaded)
        assert any(t.force_feedback.contact_state == "contact" for t in loaded)
        assert np.max(np.abs(np.diff(output, axis=0))) <= 0.02 + 1e-10
        if reflection_gain == 1.0:
            assert 1e-5 < peak < 0.02
        else:
            assert 4.9 < peak < 5.0  # 协议半LSB裕量保留；不是固定满幅输出。
            assert np.max(np.abs(projected)) > peak
            assert np.any(np.abs(projected - output) > 0.02)
        state = glove.read_state()
        kinematics = glove.model.kinematics
        pose = kinematics.evaluate_tree(state.tree_positions_rad)
        motor_jacobian = kinematics._tree_derivatives(pose.all_positions)[kinematics._motor_indices]
        reaction = motor_jacobian.T @ output[-1]
        closing_direction = np.array([human_joints(0.20)[name] for name in GLOVE_JOINTS])
        assert np.dot(reaction, closing_direction) < 0  # 反作用不帮助继续闭合。
        set_human(glove, 0.0)
        for tick in advance_steps(pair, 8):
            assert_zero_feedback(tick)
        assert glove.homes == 1 and glove.commands == []
    finally:
        runtime.close()


def test_current_cycle_projection_precedes_ramp_and_udp_keeps_three_torque_stages():
    limits = MaxTorqueLinearConfig(
        schema_version=1, mode="max_torque_linear_estimate", estimate_acknowledged=True,
        source_id="synthetic-stage-limits", wuji_serial_number="explicit-test-SN",
        glove_hand="right", wuji_max_current_a=np.full(20, 2.0),
        wuji_current_bias_a=np.zeros(20), wuji_effort_sign=np.ones(20),
        glove_max_output_torque_nm=np.full(12, 0.02),
        max_torque_slew_nm_s=np.full(12, 0.05), gain=0.5, ramp_duration_s=1.0,
    )
    pair = create_contact_pair(
        limit_parameters=limits,
        contact_parameters=replace(contact_config(), reflection_gain=7.0),
    )
    runtime, hand, glove, _ = pair
    try:
        runtime.start()
        runtime.step()
        set_human(glove, 0.20)
        hand.effort[:] = 0.8
        loaded = advance_steps(pair, 24)
        tick = next(t for t in loaded if np.any(t.force_feedback.virtual_hand_torque_nm != 0))
        feedback = tick.force_feedback
        assert isinstance(feedback.projected_model_torque_nm, tuple)
        projected = np.array(feedback.projected_model_torque_nm)
        assert feedback.contact_reflection_gain == 3.5
        # 独立使用本周期CAD传动和已缓存完整Jac核对真实控制计算，不重跑Retarget。
        kinematics = glove.model.kinematics
        pose = kinematics.evaluate_tree(tick.glove_state.tree_positions_rad)
        motor_jacobian = kinematics._tree_derivatives(pose.all_positions)[kinematics._motor_indices]
        expected = 3.5 * np.linalg.solve(
            motor_jacobian.T,
            runtime.force_feedback._snapshot.value.T @ feedback.virtual_hand_torque_nm,
        )
        np.testing.assert_allclose(projected, expected, rtol=1e-12, atol=1e-12)
        assert np.linalg.norm(projected) > np.linalg.norm(feedback.model_torque_nm) > 0
        saved = feedback.projected_model_torque_nm
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
            receiver.bind(("127.0.0.1", 0))
            receiver.settimeout(1)
            publisher = CurvePublisher(
                receiver.getsockname()[1], glove_model=glove.model, motors=glove.motors,
                wuji_joint_names=runtime.retargeter.forward.retargeter.output_joint_names,
            )
            try:
                publisher.publish(tick)
                sample = decode_sample(receiver.recvfrom(16_384)[0])
            finally:
                publisher.close()
        assert sample["contact_reflection_gain"] == 3.5
        np.testing.assert_allclose(sample["wuji_tracking_error_rad"],
                                   tick.output_positions_rad - tick.hand2_state.positions_rad)
        np.testing.assert_array_equal(sample["wuji_velocity_rad_s"], tick.hand2_state.velocities_rad_s)
        np.testing.assert_array_equal(sample["virtual_hand_torque_nm"], feedback.virtual_hand_torque_nm)
        indices = {name: index for index, name in enumerate(glove.model.motor_names)}
        for mid, motor in glove.motors.items():
            index = indices[glove.model.joint_motor_names[motor.joint]]
            scale = motor.sign * motor.scale
            assert sample["glove_projected_torque_nm"][mid - 1] == projected[index] * scale
            assert sample["glove_target"][mid - 1] == feedback.model_torque_nm[index] * scale
        advance_steps(pair, 4)
        assert feedback.projected_model_torque_nm == saved
    finally:
        runtime.close()
