"""真实CAD／完整官方Jac的消费者回归；虚拟时钟隔离几何后台调度，不连接硬件。"""
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest

from wujihand2_force_glove.domain import GLOVE_JOINTS
from wujihand2_force_glove.real_hand.force_feedback import _Jacobian
from wujihand2_force_glove.real_hand.force_linear import MaxTorqueLinearConfig
from wujihand2_force_glove.real_hand.runtime import create_retargeter
from test_contact_runtime import (
    ROOT, advance_steps, contact_config, create_contact_pair, human_joints, set_human,
)
from test_real_hand_runtime import ForceMotorPlant


DT_S = 0.02
FIXED_SWING = [5, 9, 13, 17]


@pytest.fixture(scope="module")
def complete_geometry():
    # 只缓存静态输入的数学结果，不测/推断真实worker的墙钟性能。
    # 每份矩阵均为完整CAD扰动、官方Retarget中心差分，不是目标回显或手写映射。
    bridge = create_retargeter(project_root=ROOT, estimate_jacobian=True)
    snapshots = []
    for amount in (0.0, 0.20):
        joints = human_joints(amount)
        q = np.array([joints[name] for name in GLOVE_JOINTS])
        value = bridge.forward.local_jacobian(joints, step_rad=0.01)
        assert value.shape == (20, 12) and np.isfinite(value).all()
        np.testing.assert_array_equal(value[FIXED_SWING], np.zeros((4, 12)))
        assert np.any(value != 0)
        snapshots.append((q, value))
    return snapshots


class ControlledGeometry:
    """仅替换异步几何收件箱；状态机、投影、限幅、实际发送均走正式路径。"""

    def __init__(self, controller, snapshots):
        self.controller = controller
        self.snapshots = snapshots
        self.automatic = True
        self.latest_request = None
        self.request_id = 0

    def submit(self, q, source_ns, now_ns, *, force=False):
        self.request_id += 1
        self.latest_request = (q.copy(), source_ns, self.request_id)
        if self.automatic:
            self.publish_latest()
        return self.request_id

    def publish_latest(self):
        q, source_ns, request_id = self.latest_request
        for center, value in self.snapshots:
            # 数值设备m->q的闭链求解误差不改变本回归的静态输入工况。
            if np.allclose(q, center, rtol=0, atol=1e-7):
                snapshot = _Jacobian(value, center, source_ns, request_id)
                with self.controller._condition:
                    self.controller._snapshot = snapshot
                    self.controller._condition.notify_all()
                return
        raise AssertionError("受控源只能投递已真实求解的中心姿态，不能续期其他输入")

    def freeze(self):
        self.automatic = False

    def recover(self):
        # 使用确实采到的新源帧，而不是把旧source_ns改成投递/接收时刻。
        self.publish_latest()
        self.automatic = True

    def replay_old_result(self):
        with self.controller._condition:
            old = self.controller._snapshot
            self.request_id += 1
            self.controller._snapshot = replace(old, request_id=self.request_id)


class Consumer:
    def __init__(self, pair, geometry):
        self.pair = pair
        self.runtime, self.hand, self.glove, self.clock = pair
        self.geometry = geometry
        self.controller = self.runtime.force_feedback
        self.slew = np.empty(12)
        self.cap = np.empty(12)
        names = list(self.glove.model.motor_names)
        for mid, motor in self.glove.motors.items():
            axis = names.index(self.glove.model.joint_motor_names[motor.joint])
            scale = abs(motor.sign * motor.scale)
            self.slew[axis] = self.controller.config.max_torque_slew_nm_s[mid - 1] / scale
            self.cap[axis] = self.glove.model_force_torque_limits_nm[names[axis]]

    def steps(self, count=1):
        ticks = advance_steps(self.pair, count)
        for tick in ticks:
            np.testing.assert_array_equal(tick.output_positions_rad[FIXED_SWING], np.zeros(4))
            assert not np.any(tick.force_feedback.contact_active_mask[FIXED_SWING])
            assert np.max(np.abs(tick.force_feedback.model_torque_nm)) <= 5.0
        assert self.glove.commands == [] and self.glove.force_commands == []
        assert self.glove.homes == 1
        return ticks

    def blocked(self):
        set_human(self.glove, 0.20)
        self.hand.effort[:] = 0.8

    def settled_contact(self):
        self.blocked()
        tick = self.steps(32)[-1]
        assert tick.force_feedback.contact_state == "contact"
        assert np.max(np.abs(tick.force_feedback.model_torque_nm)) > 1e-5
        return tick

    def expire_geometry(self):
        self.geometry.freeze()
        tick = self.steps(26)[-1]
        assert tick.force_feedback.projection_status == "refreshing_age"
        assert tick.force_feedback.jacobian_age_ns > 500_000_000
        assert_waiting_zero(tick)
        return tick


@pytest.fixture
def consumer_factory(monkeypatch, complete_geometry):
    @contextmanager
    def create(*, reflection_gain=6000.0, slew=1.0, glove_type=ForceMotorPlant, dwell_s=0.15):
        limits = MaxTorqueLinearConfig(
            schema_version=1, mode="max_torque_linear_estimate", estimate_acknowledged=True,
            source_id="synthetic-latency-consumer-limits", wuji_serial_number="explicit-test-SN",
            glove_hand="right", wuji_max_current_a=np.full(20, 2.0),
            wuji_current_bias_a=np.zeros(20), wuji_effort_sign=np.ones(20),
            glove_max_output_torque_nm=np.full(12, 5.0),
            max_torque_slew_nm_s=np.full(12, slew), gain=1.0,
        )
        pair = create_contact_pair(
            realtime=True, glove_type=glove_type, limit_parameters=limits,
            contact_parameters=replace(contact_config(), reflection_gain=reflection_gain, dwell_s=dwell_s),
        )
        runtime = pair[0]
        geometry = ControlledGeometry(runtime.force_feedback, complete_geometry)
        monkeypatch.setattr(runtime.force_feedback, "_submit", geometry.submit)
        try:
            runtime.start()
            runtime.step()
            yield Consumer(pair, geometry)
        finally:
            runtime.close()
    return create


def assert_waiting_zero(tick):
    feedback = tick.force_feedback
    assert feedback.projection_status != "active"
    assert feedback.contact_state == "waiting_geometry"
    assert feedback.projected_model_torque_nm is None
    np.testing.assert_array_equal(feedback.model_torque_nm, np.zeros(12))


def assert_first_reaction_is_slew_bounded(consumer, tick):
    feedback = tick.force_feedback
    assert feedback.projection_status == "active"
    assert feedback.contact_state == "contact"
    projected = np.asarray(feedback.projected_model_torque_nm)
    output = np.asarray(feedback.model_torque_nm)
    assert np.any(feedback.contact_active_mask)
    assert np.max(np.abs(output)) > 1e-5
    assert np.all(np.abs(output) <= consumer.slew * DT_S + 1e-10)
    # 会话已成熟，第一帧应受实际斜率/上限约束，而非重新乘500ms启动包络。
    axis = int(np.argmax(np.abs(output)))
    bound = min(abs(projected[axis]), consumer.cap[axis], consumer.slew[axis] * DT_S)
    assert abs(output[axis]) >= bound * 0.99
    assert output[axis] * projected[axis] > 0


def test_blocked_operator_feels_reaction_on_first_fresh_geometry_cycle(consumer_factory):
    with consumer_factory() as consumer:
        consumer.settled_contact()
        waiting = consumer.expire_geometry()
        # 实际输出零不等于新鲜受阻信息被抹掉；恢复前估计仍可继续观察。
        assert np.any(waiting.force_feedback.contact_active_mask)
        assert np.any(waiting.force_feedback.virtual_hand_torque_nm != 0)
        for tick in consumer.steps(3):
            assert_waiting_zero(tick)
            assert np.any(tick.force_feedback.contact_active_mask)
        consumer.geometry.recover()
        first = consumer.steps()[0]
        assert_first_reaction_is_slew_bounded(consumer, first)
        # 第一个20ms周期即有反馈，不再等待150ms确认；后续也不突破斜率。
        following = consumer.steps(4)
        outputs = np.array([first.force_feedback.model_torque_nm,
                            *(t.force_feedback.model_torque_nm for t in following)])
        assert np.all(np.abs(np.diff(outputs, axis=0)) <= consumer.slew * DT_S + 1e-10)
        assert all(deadline - target_tick.hand2_state.received_ns == 100_000_000
                   for target_tick, (_, deadline) in zip(
                       [first, *following], consumer.hand.commands[-5:]))


def test_obstruction_confirmed_during_geometry_gap_is_felt_without_second_dwell(consumer_factory):
    with consumer_factory() as consumer:
        consumer.steps(32)  # 会话已过初始渐入，但尚无负载或受阻。
        consumer.expire_geometry()
        consumer.blocked()
        waiting = consumer.steps(12)
        for tick in waiting:
            assert_waiting_zero(tick)
        assert not np.any(waiting[0].force_feedback.contact_active_mask)
        assert np.any(waiting[-1].force_feedback.contact_active_mask)
        assert np.any(waiting[-1].force_feedback.virtual_hand_torque_nm != 0)
        consumer.geometry.recover()
        assert_first_reaction_is_slew_bounded(consumer, consumer.steps()[0])


def test_operator_opening_during_geometry_gap_cannot_receive_old_reaction_on_recovery(consumer_factory):
    with consumer_factory() as consumer:
        consumer.settled_contact()
        consumer.expire_geometry()
        set_human(consumer.glove, 0.0)
        waiting = consumer.steps(8)
        for tick in waiting:
            assert_waiting_zero(tick)
            np.testing.assert_array_equal(tick.force_feedback.virtual_hand_torque_nm, np.zeros(20))
        assert not np.any(waiting[-1].force_feedback.contact_active_mask)
        consumer.geometry.recover()
        for tick in consumer.steps(4):
            np.testing.assert_array_equal(tick.force_feedback.model_torque_nm, np.zeros(12))
            assert not np.any(tick.force_feedback.contact_active_mask)
        # 再次闭合是新接触，不能复用旧锁存来跳过驻留。
        consumer.blocked()
        for tick in consumer.steps(5):
            np.testing.assert_array_equal(tick.force_feedback.model_torque_nm, np.zeros(12))
            assert not np.any(tick.force_feedback.contact_active_mask)


def test_first_session_contact_still_obeys_original_half_second_startup_envelope(consumer_factory):
    with consumer_factory(reflection_gain=1.0, slew=1000.0) as consumer:
        consumer.blocked()
        ticks = consumer.steps(18)
        assert all(not np.any(t.force_feedback.contact_active_mask) for t in ticks[:5])
        feedback = ticks[-1].force_feedback
        assert feedback.contact_state == "contact"
        projected = np.asarray(feedback.projected_model_torque_nm)
        output = np.asarray(feedback.model_torque_nm)
        axis = int(np.argmax(np.abs(output)))
        assert 0 < abs(output[axis]) < abs(projected[axis]) * 0.95
        mature = consumer.steps(10)[-1].force_feedback
        np.testing.assert_allclose(mature.model_torque_nm[axis],
                                   mature.projected_model_torque_nm[axis], rtol=1e-8, atol=1e-10)


def test_explicit_reset_during_geometry_gap_preserves_startup_not_a_recovery_restart(consumer_factory):
    with consumer_factory(reflection_gain=1.0, slew=1000.0) as consumer:
        consumer.settled_contact()
        consumer.expire_geometry()
        consumer.controller.reset(now_ns=consumer.clock())
        waiting = consumer.steps(12)
        for tick in waiting:
            assert_waiting_zero(tick)
        consumer.geometry.recover()
        recovered = consumer.steps()[0].force_feedback  # reset后260ms，而不是新500ms会话。
        assert recovered.contact_state == "contact"
        projected = np.asarray(recovered.projected_model_torque_nm)
        output = np.asarray(recovered.model_torque_nm)
        axis = int(np.argmax(np.abs(projected)))
        assert 0 < abs(output[axis]) < abs(projected[axis]) * 0.75
        mature = consumer.steps(13)[-1].force_feedback
        np.testing.assert_allclose(mature.model_torque_nm[axis],
                                   mature.projected_model_torque_nm[axis], rtol=1e-8, atol=1e-10)


def test_short_load_pulse_is_rejected_but_sustained_contact_meets_two_period_dwell(consumer_factory):
    with consumer_factory(dwell_s=0.04) as consumer:
        consumer.steps(32)
        consumer.blocked()
        for tick in consumer.steps(2):
            assert not np.any(tick.force_feedback.contact_active_mask)
            np.testing.assert_array_equal(tick.force_feedback.model_torque_nm, np.zeros(12))
        set_human(consumer.glove, 0.0)
        released = consumer.steps()[0]
        assert not np.any(released.force_feedback.contact_active_mask)
        np.testing.assert_array_equal(released.force_feedback.model_torque_nm, np.zeros(12))
        consumer.blocked()
        sustained = consumer.steps(3)
        assert not np.any(sustained[0].force_feedback.contact_active_mask)
        assert sustained[-1].force_feedback.contact_state == "contact"
        assert_first_reaction_is_slew_bounded(consumer, sustained[-1])


def test_repeated_source_frames_cannot_confirm_contact_or_extend_original_freshness(consumer_factory):
    with consumer_factory() as consumer:
        consumer.steps(32)
        consumer.expire_geometry()
        consumer.blocked()
        first = consumer.steps()[0]
        frozen_ns = consumer.clock()
        assert not np.any(first.force_feedback.contact_active_mask)
        for _ in range(4):
            # 三个源时间都不前进，但调度时钟继续走；仍在原始100ms有效期内。
            offset = frozen_ns - (consumer.clock() + int(DT_S * 1e9))
            consumer.hand.stamp_offset = consumer.glove.plant.stamp_offset = offset
            tick = consumer.steps()[0]
            assert tick.hand2_state.received_ns == frozen_ns
            assert tick.glove_state.received_ns == frozen_ns
            assert tick.glove_state.pose.timestamp_ns == frozen_ns
            assert_waiting_zero(tick)
            assert not np.any(tick.force_feedback.contact_active_mask)
        sent = len(consumer.hand.commands), len(consumer.glove.force_torque_commands)
        consumer.clock.advance(0.020000001)
        offset = frozen_ns - consumer.clock()
        consumer.hand.stamp_offset = consumer.glove.plant.stamp_offset = offset
        with pytest.raises(TimeoutError):
            consumer.runtime.step()
        assert (len(consumer.hand.commands), len(consumer.glove.force_torque_commands)) == sent
        assert consumer.hand.disabled == consumer.glove.disabled == 1


def test_replayed_geometry_does_not_renew_age_or_override_pose_radius(consumer_factory):
    with consumer_factory() as consumer:
        consumer.settled_contact()
        expired = consumer.expire_geometry()
        consumer.geometry.replay_old_result()
        replayed = consumer.steps()[0]
        assert_waiting_zero(replayed)
        assert replayed.force_feedback.projection_status == "refreshing_age"
        assert replayed.force_feedback.jacobian_age_ns == expired.force_feedback.jacobian_age_ns + 20_000_000
        consumer.geometry.recover()
        consumer.steps()
        consumer.geometry.freeze()
        set_human(consumer.glove, 0.48)
        outside = consumer.steps()[0]
        assert outside.force_feedback.jacobian_age_ns < 500_000_000
        assert outside.force_feedback.projection_status == "refreshing_pose"
        assert_waiting_zero(outside)


def test_recovered_reaction_still_respects_actual_model_motor_soft_limit(consumer_factory):
    with consumer_factory() as consumer:
        baseline = consumer.settled_contact()
        output = np.asarray(baseline.force_feedback.model_torque_nm)
        axis = int(np.argmax(np.abs(output)))
        assert abs(output[axis]) > 1e-5
        name = consumer.glove.model.motor_names[axis]
        position = baseline.glove_state.motor_positions_rad[name]
        direction = np.sign(output[axis])

    class BoundaryGlove(ForceMotorPlant):
        @property
        def model_motor_limits_rad(self):
            limits = dict(super().model_motor_limits_rad)
            # 只改该数值设备的软件窗，不改CAD反馈、真实配置、零位或正向目标。
            limits[name] = ((position - 0.2, position - 1e-6) if direction > 0
                            else (position + 1e-6, position + 0.2))
            return limits

    with consumer_factory(glove_type=BoundaryGlove) as consumer:
        consumer.blocked()
        consumer.steps(32)
        consumer.expire_geometry()
        consumer.geometry.recover()
        tick = consumer.steps()[0]
        feedback = tick.force_feedback
        assert feedback.contact_state == "contact"
        assert direction * feedback.projected_model_torque_nm[axis] > 1e-5
        assert feedback.model_torque_nm[axis] == 0
        assert consumer.glove.force_torque_commands[-1][axis] == 0
