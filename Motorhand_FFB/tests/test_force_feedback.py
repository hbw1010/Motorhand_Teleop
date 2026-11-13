"""仅离线CAD/真实重定向数学；合成标定只存在于测试临时目录，不可上机。"""
import json
import multiprocessing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from motorhand_teleop.domain import GLOVE_JOINTS, WujiState
from motorhand_teleop.feedback import CalibrationRequired
from motorhand_teleop.real_hand.force_feedback import (
    ForceFeedbackCalibration,
    ForceFeedbackController,
)
from motorhand_teleop.real_hand.motor_hand import EncosGloveHardwareDriver, EncosGloveState
from motorhand_teleop.retargeting import GloveWujiRetargetPipeline, default_retarget_config

ROOT = Path(__file__).parents[1]
ASSETS = ROOT.parent / "assets" / "hand_motor_ve33"
START = 1_000_000_000
NOW = START + 200_000_000


def _calibration_data():
    # 非真实物理标定，仅用于独立数学oracle，禁止作为仓库默认配置。
    return dict(schema_version=1, calibrated=True, source_id="synthetic-offline-oracle-only",
                wuji_serial_number="offline-test-device", glove_hand="right",
                wuji_torque_per_amp_nm=np.linspace(-0.3, -0.1, 20).tolist(),
                wuji_current_bias_a=np.linspace(-0.02, 0.02, 20).tolist(),
                glove_torque_constant_nm_per_a=np.linspace(0.08, 0.14, 12).tolist(),
                max_current_a=[0.2] * 12, max_slew_a_s=[1e6] * 12, gain=0.6,
                ramp_duration_s=0.1, jacobian_max_age_s=10.0,
                jacobian_refresh_interval_s=100.0)


def _pipeline():
    return GloveWujiRetargetPipeline.from_yaml(
        "right", asset_root=ASSETS, config_path=default_retarget_config(ROOT, "right"),
        estimate_jacobian=True)


def _glove(model, *, now=NOW, joints=None):
    if joints is None:
        # 已有CAD回归使用的真实模型m内点；避免零位的软件限位投影污染功率oracle。
        _, pose = model.evaluate_motor_state({name: 0.04 + i * 0.002 for i, name in enumerate(model.motor_names)})
    else:
        pose = model.evaluate_raw(joints)
    return EncosGloveState(pose=model.to_glove_pose(pose, timestamp_ns=now), feedback_by_motor={},
                           online_ids=tuple(range(1, 13)), received_ns=now,
                           motor_positions_rad=pose.motor_positions,
                           tree_positions_rad=pose.all_positions)


def _hand(calibration, *, now=NOW, offset=None):
    delta = np.linspace(-2e-5, 3e-5, 20) if offset is None else np.asarray(offset)
    return WujiState(np.zeros(20), None, calibration.wuji_current_bias_a + delta,
                     now, 20, {"serial_number": calibration.wuji_serial_number})


@pytest.fixture(scope="module")
def rig():
    cal = ForceFeedbackCalibration(**_calibration_data())
    driver = EncosGloveHardwareDriver(torque_constant_nm_per_a=cal.glove_torque_constants_by_id)
    pipeline = _pipeline()
    controller = ForceFeedbackController(cal, SimpleNamespace(forward=pipeline), driver, _pipeline,
                                        clock_ns=lambda: START)
    model = pipeline.landmarks.model
    controller.prepare(_hand(cal, now=START), _glove(model, now=START), now_ns=START, timeout_s=30)
    yield SimpleNamespace(controller=controller, calibration=cal, driver=driver, model=model,
                          pipeline=pipeline)
    controller.close()
    # driver未连接，没有SDK对象或输出权。


@pytest.fixture
def active(rig):
    rig.controller.reset(now_ns=START)
    return rig


def _compute(rig, *, now=NOW, hand=None, glove=None):
    return rig.controller.compute(_hand(rig.calibration, now=now) if hand is None else hand,
                                  _glove(rig.model, now=now) if glove is None else glove,
                                  now_ns=now, deadline_ns=now + 20_000_000)


def test_calibration_requires_real_file_all_fields_and_device_binding(tmp_path):
    with pytest.raises(CalibrationRequired):
        ForceFeedbackCalibration.from_json(tmp_path / "absent.json")
    path = tmp_path / "offline-test-only.json"
    data = _calibration_data()
    path.write_text(json.dumps(data), encoding="utf-8")
    cal = ForceFeedbackCalibration.from_json(path)
    cal.validate_serial("offline-test-device")
    with pytest.raises(CalibrationRequired, match="序列号"):
        cal.validate_serial("another-device")
    with pytest.raises(TypeError):
        cal.glove_torque_constants_by_id[1] = 1.0
    with pytest.raises(ValueError):
        cal.wuji_torque_per_amp_nm[0] = 1
    for key in ("wuji_torque_per_amp_nm", "glove_hand", "wuji_serial_number", "glove_torque_constant_nm_per_a"):
        missing = dict(data)
        del missing[key]
        path.write_text(json.dumps(missing), encoding="utf-8")
        with pytest.raises(CalibrationRequired):
            ForceFeedbackCalibration.from_json(path)


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("calibrated", 1), ("calibrated", False),
    ("glove_hand", "unknown"), ("source_id", " "), ("wuji_serial_number", ""),
    ("gain", 1.1), ("gain", True), ("ramp_duration_s", 0),
    ("jacobian_max_age_s", float("inf")), ("feedback_max_age_s", float("nan")),
    ("max_current_a", [0.1] * 11), ("max_slew_a_s", [-1] * 12),
    ("wuji_torque_per_amp_nm", [0] * 20), ("wuji_current_bias_a", [True] * 20),
    ("glove_torque_constant_nm_per_a", ["0.1"] * 12),
])
def test_calibration_rejects_unphysical_or_non_strict_values(field, value):
    data = _calibration_data()
    data[field] = value
    with pytest.raises((ValueError, CalibrationRequired)):
        ForceFeedbackCalibration(**data)


def test_constructor_rejects_missing_calibration_and_wrong_hand_or_kt(rig):
    bridge = SimpleNamespace(forward=rig.pipeline)
    with pytest.raises(CalibrationRequired):
        ForceFeedbackController(None, bridge, rig.driver, _pipeline)
    with pytest.raises(CalibrationRequired, match="手侧"):
        ForceFeedbackController(replace(rig.calibration, glove_hand="left"), bridge, rig.driver, _pipeline)
    with pytest.raises(CalibrationRequired, match="Kt"):
        ForceFeedbackController(replace(rig.calibration, glove_torque_constant_nm_per_a=np.ones(12)),
                                bridge, rig.driver, _pipeline)


def test_real_cad_reflection_has_reaction_sign_and_power_conjugacy(active):
    rig = active
    state = _hand(rig.calibration)
    result = _compute(rig, hand=state)
    jac = rig.controller._snapshot.value
    actuator = (state.effort_a - rig.calibration.wuji_current_bias_a) * rig.calibration.wuji_torque_per_amp_nm
    expected_joint = -rig.calibration.gain * jac.T @ actuator
    assert np.linalg.norm(expected_joint) > 1e-10
    np.testing.assert_allclose(result.wuji_actuator_torque_nm, actuator, atol=1e-15)
    np.testing.assert_allclose(result.glove_joint_torque_nm, expected_joint, atol=1e-15)

    # 独立中央差分真实CAD q→m，不能复用被测隐函数Jac或其转置求解作oracle。
    q = _glove(rig.model).pose.vector()
    step = 1e-5
    motor_jac = np.column_stack([
        (rig.model.motor_positions_from_joint_vector(q + np.eye(12)[i] * step)
         - rig.model.motor_positions_from_joint_vector(q - np.eye(12)[i] * step)) / (2 * step)
        for i in range(12)
    ])
    constants = np.array([rig.driver.model_motor_torque_constants_nm_per_a[name] for name in rig.model.motor_names])
    motor_torque = np.asarray(result.model_current_a) * constants
    np.testing.assert_allclose(motor_jac.T @ motor_torque, expected_joint, rtol=3e-3, atol=3e-8)
    velocity = np.linspace(-0.2, 0.3, 12)
    glove_power = motor_torque @ (motor_jac @ velocity)
    wuji_power = actuator @ (jac @ velocity)
    assert glove_power == pytest.approx(-rig.calibration.gain * wuji_power, rel=3e-3, abs=2e-8)
    assert expected_joint @ expected_joint > 0
    assert (jac @ expected_joint) @ actuator < 0  # 正向执行器力矩在手套侧是阻力。
    with pytest.raises(ValueError):
        result.glove_joint_torque_nm[0] = 1


def test_actual_effort_change_changes_reflection_but_position_change_does_not(active):
    first = _compute(active)
    active.controller.acknowledge(first.model_current_a)
    second_state = _hand(active.calibration, now=NOW + 20_000_000,
                         offset=2 * np.linspace(-2e-5, 3e-5, 20))
    second = _compute(active, now=NOW + 20_000_000, hand=second_state)
    np.testing.assert_allclose(second.model_current_a, 2 * np.asarray(first.model_current_a), atol=1e-10)
    altered_position = replace(second_state, positions_rad=np.linspace(0, 0.3, 20))
    third = _compute(active, now=NOW + 20_000_000, hand=altered_position)
    np.testing.assert_allclose(third.model_current_a, second.model_current_a, atol=1e-12)


def test_reset_reanchors_zero_and_quintic_torque_ramp(active):
    full = _compute(active)
    active.controller.acknowledge(full.model_current_a)
    active.controller.reset(now_ns=NOW)
    zero = _compute(active, now=NOW)
    np.testing.assert_array_equal(zero.model_current_a, np.zeros(12))
    active.controller.acknowledge(zero.model_current_a)
    half = _compute(active, now=NOW + 50_000_000)
    np.testing.assert_allclose(half.model_current_a, 0.5 * np.asarray(full.model_current_a), atol=1e-10)
    active.controller.acknowledge(half.model_current_a)
    end = _compute(active, now=NOW + 100_000_000)
    np.testing.assert_allclose(end.model_current_a, full.model_current_a, atol=1e-10)


def test_raw_id_binding_not_model_array_order_and_scaled_current_caps(rig):
    motors = {raw_id: replace(rig.driver.motors[13 - raw_id], motor_id=raw_id,
                              sign=(-1.0 if raw_id % 2 else 1.0), scale=1.0 + raw_id / 10)
              for raw_id in range(1, 13)}
    constants = {rig.model.joint_motor_names[config.joint]:
                 rig.calibration.glove_torque_constant_nm_per_a[raw_id - 1]
                 for raw_id, config in motors.items()}
    metadata = SimpleNamespace(hand="right", motors=motors,
                               model_motor_torque_constants_nm_per_a=constants,
                               model_current_limits_a=dict.fromkeys(rig.model.motor_names, 0.15),
                               model_motor_limits_rad=rig.driver.model_motor_limits_rad)
    cal = replace(rig.calibration, max_current_a=np.linspace(0.1, 0.5, 12),
                  max_slew_a_s=np.linspace(0.2, 0.8, 12))
    controller = ForceFeedbackController(cal, SimpleNamespace(forward=rig.pipeline), metadata, _pipeline)
    try:
        for raw_id, config in motors.items():
            name = rig.model.joint_motor_names[config.joint]
            index = rig.model.motor_names.index(name)
            scale = abs(config.sign * config.scale)
            assert controller._kt[index] == cal.glove_torque_constant_nm_per_a[raw_id - 1]
            assert controller._max_current[index] == pytest.approx(min(cal.max_current_a[raw_id - 1] / scale, 0.15))
            assert controller._max_slew[index] == pytest.approx(cal.max_slew_a_s[raw_id - 1] / scale)
    finally:
        controller.close()


def test_amplitude_slew_and_actual_projected_ack_history(active, monkeypatch):
    controller = active.controller
    monkeypatch.setattr(controller, "_max_current", np.full(12, 0.01))
    monkeypatch.setattr(controller, "_max_slew", np.full(12, 0.002))
    huge = _hand(active.calibration, offset=np.linspace(-20, 30, 20))
    first = _compute(active, hand=huge)
    assert np.max(np.abs(first.model_current_a)) <= 0.0004 + 1e-14
    # 发送器可能把边界外向电流投影为零；下一帧必须从实际零输出而非候选积累。
    controller.acknowledge((0.0,) * 12)
    second = _compute(active, now=NOW + 20_000_000,
                      hand=replace(huge, received_ns=NOW + 20_000_000))
    assert np.max(np.abs(second.model_current_a)) <= 0.00004 + 1e-14
    controller.acknowledge(second.model_current_a)
    capped = _compute(active, now=START + 9_000_000_000,
                      hand=replace(huge, received_ns=START + 9_000_000_000))
    assert np.max(np.abs(capped.model_current_a)) <= 0.01
    assert np.max(np.abs(capped.model_current_a)) > 0.009


def test_real_model_motor_boundary_blocks_only_outward_torque(active, monkeypatch):
    baseline = _compute(active)
    currents = np.asarray(baseline.model_current_a)
    axis = int(np.argmax(np.abs(currents)))
    assert abs(currents[axis]) > 1e-10
    limits = active.controller._limits.copy()
    motors = _glove(active.model).motor_positions_rad
    position = motors[active.model.motor_names[axis]]
    if currents[axis] > 0:
        limits[axis] = (position - 0.2, position)
    else:
        limits[axis] = (position, position + 0.2)
    monkeypatch.setattr(active.controller, "_limits", limits)
    blocked = _compute(active)
    assert blocked.model_current_a[axis] == 0
    active.controller.acknowledge(blocked.model_current_a)
    reverse = _hand(active.calibration, now=NOW + 20_000_000,
                    offset=-np.linspace(-2e-5, 3e-5, 20))
    inward = _compute(active, now=NOW + 20_000_000, hand=reverse)
    assert inward.model_current_a[axis] == pytest.approx(-currents[axis])


@pytest.mark.parametrize("invalid", ["missing-effort", "stale-hand", "stale-glove", "stale-pose", "future", "offline", "serial", "tree", "motor-tree"])
def test_invalid_live_inputs_reject_instead_of_fake_zero(active, invalid):
    hand = _hand(active.calibration)
    glove = _glove(active.model)
    if invalid == "missing-effort":
        hand = replace(hand, effort_a=None)
    elif invalid == "stale-hand":
        hand = replace(hand, received_ns=START)
    elif invalid == "stale-glove":
        glove = replace(glove, received_ns=START)
    elif invalid == "stale-pose":
        glove = replace(glove, pose=replace(glove.pose, timestamp_ns=START))
    elif invalid == "future":
        hand = replace(hand, received_ns=NOW + 1)
    elif invalid == "offline":
        hand = replace(hand, online_count=19)
    elif invalid == "serial":
        hand = replace(hand, diagnostics={"serial_number": "wrong-device"})
    elif invalid == "tree":
        glove = replace(glove, tree_positions_rad={})
    else:
        values = dict(glove.motor_positions_rad)
        values[next(iter(values))] += 0.01
        glove = replace(glove, motor_positions_rad=values)
    with pytest.raises((CalibrationRequired, ValueError, RuntimeError, TimeoutError)):
        _compute(active, hand=hand, glove=glove)


def test_expired_matrix_pauses_reflection_without_renewing_its_source_stamp(rig):
    controller = ForceFeedbackController(
        rig.calibration, SimpleNamespace(forward=rig.pipeline), rig.driver, _pipeline,
        clock_ns=lambda: START,
    )
    local = SimpleNamespace(controller=controller, calibration=rig.calibration, model=rig.model)
    try:
        controller.prepare(_hand(rig.calibration, now=START), _glove(rig.model, now=START),
                           now_ns=START, timeout_s=30)
        controller.reset(now_ns=START)
        previous = _compute(local)
        controller.acknowledge(previous.model_current_a)
        expired = START + 10_000_000_001
        waiting = _compute(local, now=expired)
        assert waiting.projection_status == "refreshing_age"
        assert waiting.jacobian_age_ns == expired - START
        np.testing.assert_array_equal(waiting.model_current_a, np.zeros(12))
        assert waiting.glove_joint_torque_nm is None
    finally:
        controller.close()


def test_prepare_cannot_substitute_stale_warmup_feedback(active):
    with pytest.raises(TimeoutError, match="Wuji.*过期"):
        _compute(active, hand=_hand(active.calibration, now=START))
    with pytest.raises(TimeoutError, match="期限"):
        active.controller.compute(_hand(active.calibration), _glove(active.model),
                                  now_ns=NOW, deadline_ns=NOW)


def test_backend_failure_explicitly_rejects_and_never_reuses_old_matrix(active, monkeypatch):
    monkeypatch.setattr(active.controller, "_worker_error", ValueError("offline numeric failure"))
    with pytest.raises(RuntimeError, match="后台计算失败"):
        _compute(active)


def test_realtime_uses_current_tree_implicit_derivative_not_joint_inverse_differences(active, monkeypatch):
    glove = _glove(active.model)

    def forbidden_inverse(*args, **kwargs):
        raise AssertionError("实时力反射不能再次逐轴q→m逆解")

    monkeypatch.setattr(active.model, "motor_positions_from_joint_vector", forbidden_inverse)
    result = _compute(active, glove=glove)
    assert np.isfinite(result.model_current_a).all()


def test_singular_current_cad_transmission_is_not_damped_or_zero_fallback(active, monkeypatch):
    glove = _glove(active.model)
    monkeypatch.setattr(active.model.kinematics, "_tree_derivatives", lambda state: np.zeros((68, 12)))
    with pytest.raises(ValueError, match="奇异"):
        _compute(active, glove=glove)


def test_prepare_rejects_shared_forward_optimizer_instead_of_cross_thread_reuse(rig):
    controller = ForceFeedbackController(rig.calibration, SimpleNamespace(forward=rig.pipeline),
                                        rig.driver, lambda: rig.pipeline, clock_ns=lambda: START)
    try:
        with pytest.raises(RuntimeError, match="后台计算失败"):
            controller.prepare(_hand(rig.calibration, now=START), _glove(rig.model, now=START),
                               now_ns=START, timeout_s=5)
    finally:
        controller.close()


class BlockedPipeline:
    def __init__(self, entered, release, block):
        self.pipeline = _pipeline()
        self.entered, self.release, self.block = entered, release, block

    @property
    def landmarks(self):
        return self.pipeline.landmarks

    @property
    def retargeter(self):
        return self.pipeline.retargeter

    def local_jacobian(self, joints, *, step_rad):
        if self.block.is_set():
            self.entered.set()
            if not self.release.wait(30):
                raise TimeoutError("测试门控未释放")
        return self.pipeline.local_jacobian(joints, step_rad=step_rad)


class BlockedPipelineFactory:
    def __init__(self, entered, release, block):
        self.entered, self.release, self.block = entered, release, block

    def __call__(self):
        return BlockedPipeline(self.entered, self.release, self.block)


def test_realtime_reflection_uses_new_effort_while_matrix_workers_are_blocked(rig):
    context = multiprocessing.get_context("spawn")
    entered, release, block = context.Event(), context.Event(), context.Event()
    factory = BlockedPipelineFactory(entered, release, block)
    cal = replace(rig.calibration, jacobian_refresh_interval_s=0.001)
    controller = ForceFeedbackController(cal, SimpleNamespace(forward=rig.pipeline), rig.driver, factory,
                                        clock_ns=lambda: START)
    local = SimpleNamespace(controller=controller, calibration=cal, model=rig.model)
    try:
        controller.prepare(_hand(cal, now=START), _glove(rig.model, now=START), now_ns=START, timeout_s=30)
        controller.reset(now_ns=START)
        block.set()
        first = _compute(local)
        assert entered.wait(5)
        # 有限差分被阻塞时仍可计算当前真实effort；不可阻塞等worker或返回旧电流。
        second = _compute(local, now=NOW + 20_000_000,
                          hand=_hand(cal, now=NOW + 20_000_000, offset=np.ones(20) * 0.0001))
        assert not np.allclose(first.model_current_a, second.model_current_a, atol=1e-10)
    finally:
        release.set()
        controller.close()
    with pytest.raises(RuntimeError, match="关闭"):
        _compute(local)


def test_reused_jacobian_solver_preserves_fresh_official_solution_derivatives():
    pipeline = _pipeline()
    joints = dict.fromkeys(GLOVE_JOINTS, 0.0)
    joints["index_pip"] = 0.08
    step = 0.01
    initial = pipeline.local_jacobian(joints, step_rad=step)
    moved = dict(joints, index_pip=0.12)
    pipeline.local_jacobian(moved, step_rad=step)
    repeated = pipeline.local_jacobian(joints, step_rad=step)
    np.testing.assert_allclose(repeated, initial, atol=3e-4, rtol=3e-4)
    # 使用全新官方求解器与无种子点源，独立核对主要动作列；
    # 若复用优化历史改变了正负扰动的目标函数/初值，这里的力映射会变化。
    def precise_step(values):
        fresh = _pipeline()
        fresh.retargeter._core.optimizer.opt.set_ftol_abs(1e-12)
        fresh.retargeter._core.optimizer.opt.set_maxeval(200)
        return fresh.step(values).positions_rad

    for index in (0, 4, 5):
        plus, minus = dict(joints), dict(joints)
        plus[GLOVE_JOINTS[index]] += step
        minus[GLOVE_JOINTS[index]] -= step
        expected = (
            precise_step(plus) - precise_step(minus)
        ) / (2 * step)
        np.testing.assert_allclose(repeated[:, index], expected, atol=3e-4, rtol=3e-4)


def test_glove_torque_constant_is_independent_from_wuji_current_to_torque(active):
    baseline = _compute(active)
    calibration = replace(
        active.calibration,
        glove_torque_constant_nm_per_a=2 * active.calibration.glove_torque_constant_nm_per_a,
    )
    driver = EncosGloveHardwareDriver(
        torque_constant_nm_per_a=calibration.glove_torque_constants_by_id,
    )
    controller = ForceFeedbackController(
        calibration, SimpleNamespace(forward=active.pipeline), driver, _pipeline,
        clock_ns=lambda: START,
    )
    try:
        controller.prepare(
            _hand(calibration, now=START), _glove(active.model, now=START), now_ns=START,
        )
        controller.reset(now_ns=START)
        local = SimpleNamespace(
            controller=controller, calibration=calibration, model=active.model,
        )
        changed = _compute(local)
        np.testing.assert_allclose(changed.wuji_actuator_torque_nm, baseline.wuji_actuator_torque_nm)
        np.testing.assert_allclose(changed.glove_joint_torque_nm, baseline.glove_joint_torque_nm)
        np.testing.assert_allclose(
            changed.model_current_a, 0.5 * np.asarray(baseline.model_current_a),
            rtol=1e-9, atol=1e-12,
        )
    finally:
        controller.close()


def test_nominal_zero_quantization_ack_does_not_accumulate_force_or_mix_amp_units(rig):
    from motorhand_teleop.real_hand.force_linear import MaxTorqueLinearConfig

    config = replace(
        MaxTorqueLinearConfig.from_json(ROOT.parent / "config/feedback/max-torque-linear-right.json"),
        wuji_serial_number="offline-test-device",
        glove_max_output_torque_nm=np.full(12, 0.003),
        max_torque_slew_nm_s=np.full(12, 1e-6),
        jacobian_max_age_s=10.0, jacobian_refresh_interval_s=100.0,
    )
    driver = EncosGloveHardwareDriver(force_torque_limits_nm=config.glove_torque_limits_by_id)
    controller = ForceFeedbackController(
        config, SimpleNamespace(forward=rig.pipeline), driver, _pipeline,
        clock_ns=lambda: START,
    )

    def sample(now):
        return WujiState(np.zeros(20), None, np.zeros(20), now, 20), _glove(rig.model, now=now)

    try:
        controller.prepare(*sample(START), now_ns=START)
        controller.reset(now_ns=START)
        before = controller.compute(*sample(NOW), now_ns=NOW, deadline_ns=NOW + 20_000_000)
        np.testing.assert_array_equal(before.model_torque_nm, np.zeros(12))
        assert before.model_current_a is None and before.wuji_actuator_torque_nm is None
        with pytest.raises(ValueError, match="不能用A"):
            controller.acknowledge((0.0,) * 12)
        # SDK软界零附近必须选向内编码档；实际参数可大于留半LSB后的连续目标cap。
        bias = tuple(
            driver.model_force_torque_command_limits_nm[name] - driver.model_force_torque_limits_nm[name]
            for name in driver.motor_names
        )
        controller.acknowledge_torque(bias)
        after = controller.compute(
            *sample(NOW + 20_000_000), now_ns=NOW + 20_000_000,
            deadline_ns=NOW + 40_000_000,
        )
        np.testing.assert_array_equal(after.model_torque_nm, np.zeros(12))
    finally:
        controller.close()


@pytest.mark.parametrize("mode", ["calibrated", "max_torque_linear_estimate"])
def test_pose_change_pauses_only_reflection_then_recovers_from_background_refresh(rig, mode):
    from motorhand_teleop.real_hand.force_linear import MaxTorqueLinearConfig

    context = multiprocessing.get_context("spawn")
    entered, release, block = context.Event(), context.Event(), context.Event()
    factory = BlockedPipelineFactory(entered, release, block)
    if mode == "calibrated":
        config = rig.calibration
        driver = rig.driver
    else:
        config = replace(
            MaxTorqueLinearConfig.from_json(ROOT.parent / "config/feedback/max-torque-linear-right.json"),
            wuji_serial_number="offline-test-device", jacobian_max_age_s=10.0,
            jacobian_refresh_interval_s=100.0,
        )
        driver = EncosGloveHardwareDriver(force_torque_limits_nm=config.glove_torque_limits_by_id)
    controller = ForceFeedbackController(
        config, SimpleNamespace(forward=rig.pipeline), driver, factory, clock_ns=lambda: START,
    )
    local = SimpleNamespace(controller=controller, calibration=config, model=rig.model)

    def output(command):
        return command.model_torque_nm if controller.output_is_torque else command.model_current_a

    def acknowledge(command):
        if controller.output_is_torque:
            controller.acknowledge_torque(output(command))
        else:
            controller.acknowledge(output(command))

    def compute(now, joints=None):
        return _compute(
            local, now=now, glove=_glove(rig.model, now=now, joints=joints),
            hand=_hand(config, now=now, offset=np.linspace(0.3, 1.0, 20)),
        )

    try:
        controller.prepare(_hand(config, now=START), _glove(rig.model, now=START),
                           now_ns=START, timeout_s=30)
        controller.reset(now_ns=START)
        baseline = compute(NOW)
        assert np.max(np.abs(output(baseline))) > 1e-8
        acknowledge(baseline)
        block.set()
        joints = dict(_glove(rig.model).pose.joints)
        joints["index_pip"] += 0.4
        moved = _glove(rig.model, joints=joints)
        assert np.max(np.abs(moved.pose.vector() - controller._snapshot.joints)) > config.jacobian_max_pose_delta_rad

        paused = compute(NOW + 20_000_000, joints)
        assert paused.projection_status == "refreshing_pose"
        np.testing.assert_array_equal(output(paused), np.zeros(12))
        assert paused.glove_joint_torque_nm is None
        acknowledge(paused)
        # 即使常规刷新周期尚未到，也必须立即提交新姿态，且主循环不等待差分。
        assert entered.wait(5)
        waiting = compute(NOW + 40_000_000, joints)
        assert waiting.projection_status == "refreshing_pose"
        np.testing.assert_array_equal(output(waiting), np.zeros(12))
        acknowledge(waiting)

        release.set()
        with controller._condition:
            assert controller._condition.wait_for(
                lambda: np.allclose(controller._snapshot.joints, moved.pose.vector(), atol=1e-7, rtol=0),
                timeout=15,
            )
        resumed = compute(NOW + 60_000_000, joints)
        assert resumed.projection_status == "active"
        np.testing.assert_array_equal(output(resumed), np.zeros(12))
        acknowledge(resumed)
        restored = compute(NOW + 60_000_000 + int(config.ramp_duration_s * 1e9), joints)
        assert restored.projection_status == "active"
        assert np.max(np.abs(output(restored))) > 1e-8
    finally:
        release.set()
        controller.close()
