"""纯离线最大负载占比数学oracle；测试数字不是可上机的物理参数。"""
import itertools
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from motorhand_teleop.domain import GLOVE_JOINTS
from motorhand_teleop.feedback import CalibrationRequired
from motorhand_teleop.real_hand.force_linear import (
    LinearTorqueProjection,
    MaxTorqueLinearConfig,
    project_max_torque_linear,
)
from motorhand_teleop.retargeting import GloveWujiRetargetPipeline, default_retarget_config

ROOT = Path(__file__).parents[1]


def _data():
    return dict(
        schema_version=1, mode="max_torque_linear_estimate", estimate_acknowledged=True,
        source_id="synthetic-offline-oracle-not-hardware-maxima",
        wuji_serial_number="offline-only", glove_hand="right",
        wuji_max_current_a=np.linspace(0.2, 1.1, 20).tolist(),
        wuji_current_bias_a=np.linspace(-0.03, 0.04, 20).tolist(),
        wuji_effort_sign=[(-1) ** i for i in range(20)],
        glove_max_output_torque_nm=np.linspace(0.01, 0.03, 12).tolist(),
        max_torque_slew_nm_s=np.linspace(0.1, 0.4, 12).tolist(), gain=1.0,
    )


def _config():
    return MaxTorqueLinearConfig(**_data())


def _effort(config, load):
    return config.wuji_current_bias_a + np.asarray(load) * config.wuji_max_current_a * config.wuji_effort_sign


def _axis_retarget():
    result = np.zeros((20, 12))
    result[:12] = np.eye(12)
    return result


def test_config_explicit_estimate_binding_defaults_and_independent_arrays(tmp_path):
    data = _data()
    original = np.asarray(data["wuji_max_current_a"])
    data["wuji_max_current_a"] = original
    config = MaxTorqueLinearConfig(**data)
    original[0] = 99
    assert config.wuji_max_current_a[0] == pytest.approx(0.2)
    assert not hasattr(config, "calibrated")
    assert not hasattr(config, "wuji_torque_per_amp_nm")
    assert not hasattr(config, "glove_torque_constant_nm_per_a")
    assert (config.ramp_duration_s, config.jacobian_max_age_s,
            config.jacobian_max_pose_delta_rad, config.feedback_max_age_s,
            config.jacobian_refresh_interval_s, config.jacobian_step_rad) == (0.5, 0.5, 0.25, 0.1, 0.1, 0.01)
    for name in ("wuji_max_current_a", "wuji_current_bias_a", "wuji_effort_sign",
                 "glove_max_output_torque_nm", "max_torque_slew_nm_s"):
        with pytest.raises(ValueError):
            getattr(config, name)[0] = 10
    with pytest.raises(FrozenInstanceError):
        config.gain = 0.5
    assert dict(config.glove_torque_limits_by_id) == dict(enumerate(config.glove_max_output_torque_nm, start=1))
    with pytest.raises(TypeError):
        config.glove_torque_limits_by_id[1] = 1
    path = tmp_path / "offline-only.json"
    path.write_text(json.dumps(_data()), encoding="utf-8")
    loaded = MaxTorqueLinearConfig.from_json(path)
    np.testing.assert_array_equal(loaded.wuji_effort_sign, config.wuji_effort_sign)
    loaded.validate_serial("offline-only")
    for serial in ("other", None, 1):
        with pytest.raises(CalibrationRequired, match="序列号"):
            loaded.validate_serial(serial)


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 2),
    ("mode", "calibrated"), ("estimate_acknowledged", False),
    ("estimate_acknowledged", 1), ("estimate_acknowledged", np.bool_(True)),
    ("source_id", " "), ("wuji_serial_number", ""), ("glove_hand", "both"),
    ("wuji_max_current_a", [0] * 20), ("wuji_max_current_a", [1] * 19),
    ("wuji_max_current_a", [True] * 20), ("wuji_max_current_a", ["1"] * 20),
    ("wuji_max_current_a", [complex(1, 0)] * 20),
    ("wuji_current_bias_a", [float("nan")] * 20),
    ("wuji_effort_sign", [0] * 20), ("wuji_effort_sign", [0.5] * 20),
    ("wuji_effort_sign", [True] * 20),
    ("glove_max_output_torque_nm", [-0.1] * 12),
    ("glove_max_output_torque_nm", [float("inf")] * 12),
    ("max_torque_slew_nm_s", [0] * 12), ("max_torque_slew_nm_s", [1] * 11),
    ("gain", 0), ("gain", 1.01), ("gain", True), ("gain", 1j),
    ("ramp_duration_s", 0), ("jacobian_max_age_s", float("inf")),
    ("jacobian_max_pose_delta_rad", -1), ("feedback_max_age_s", float("nan")),
    ("jacobian_refresh_interval_s", False), ("jacobian_step_rad", "0.01"),
])
def test_config_rejects_invalid_values(field, value):
    data = _data()
    data[field] = value
    with pytest.raises(ValueError):
        MaxTorqueLinearConfig(**data)


def test_config_json_rejects_missing_unknown_nonobject_and_unreadable(tmp_path):
    path = tmp_path / "invalid.json"
    with pytest.raises(CalibrationRequired):
        MaxTorqueLinearConfig.from_json(path)
    for field in _data():
        data = _data()
        del data[field]
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(CalibrationRequired):
            MaxTorqueLinearConfig.from_json(path)
    for data in ({**_data(), "calibrated": True}, {**_data(), "unknown": 1}, [], None):
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(CalibrationRequired):
            MaxTorqueLinearConfig.from_json(path)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(CalibrationRequired):
        MaxTorqueLinearConfig.from_json(path)


def test_single_axis_full_scale_reaction_uses_each_independent_model_limit():
    config = _config()
    # 显式模型坐标上限，不是照搬raw ID配置，也不借用协议编码范围。
    limits = np.linspace(0.003, 0.009, 12)
    for axis in range(12):
        load = np.zeros(20)
        load[axis] = 1
        result = project_max_torque_linear(_axis_retarget(), np.eye(12), _effort(config, load), config, limits)
        expected = np.zeros(12)
        expected[axis] = -limits[axis]
        np.testing.assert_allclose(result.model_torque_nm, expected, atol=1e-17)
        np.testing.assert_allclose(result.wuji_load_fraction, load, atol=1e-15)


def test_joint_worst_sign_envelope_has_full_geometry_not_channelwise_normalization():
    config = _config()
    motor = np.eye(12)
    motor[0, 1] = 0.75
    motor[2, 0] = -0.4
    desired_mapping = np.zeros((12, 20))
    desired_mapping[:, :4] = np.array([
        [1 + axis * 0.1, -0.4, (-1) ** axis * 0.7, 0.2] for axis in range(12)
    ])
    # 构造已知完整B，oracle不重复被测solve。
    retarget = (motor.T @ desired_mapping).T
    limits = np.linspace(0.002, 0.006, 12)
    maxima = np.zeros(12)
    for signs in itertools.product((-1, 1), repeat=4):
        load = np.zeros(20)
        load[:4] = signs
        result = project_max_torque_linear(retarget, motor, _effort(config, load), config, limits)
        expected = np.array([
            -limits[i] * sum(desired_mapping[i, j] * signs[j] for j in range(4))
            / sum(abs(desired_mapping[i, j]) for j in range(4))
            for i in range(12)
        ])
        np.testing.assert_allclose(result.model_torque_nm, expected, atol=1e-17)
        assert np.all(np.abs(result.model_torque_nm) <= limits + 1e-17)
        maxima = np.maximum(maxima, np.abs(result.model_torque_nm))
    np.testing.assert_allclose(maxima, limits, atol=1e-17)


def test_zero_bias_sign_saturation_and_gain_are_explicit():
    config = replace(_config(), gain=0.37)
    limits = np.linspace(0.004, 0.008, 12)
    zero = project_max_torque_linear(_axis_retarget(), np.eye(12), config.wuji_current_bias_a, config, limits)
    np.testing.assert_array_equal(zero.wuji_load_fraction, np.zeros(20))
    np.testing.assert_array_equal(zero.model_torque_nm, np.zeros(12))
    requested = np.linspace(-2.5, 2.5, 20)
    result = project_max_torque_linear(_axis_retarget(), np.eye(12), _effort(config, requested), config, limits)
    clipped = np.maximum(-1, np.minimum(1, requested))
    np.testing.assert_allclose(result.wuji_load_fraction, clipped, atol=1e-15)
    np.testing.assert_allclose(result.model_torque_nm, -0.37 * limits * clipped[:12], atol=1e-17)
    half = project_max_torque_linear(_axis_retarget(), np.eye(12),
                                    _effort(config, clipped * 0.5), config, limits)
    np.testing.assert_allclose(half.model_torque_nm, result.model_torque_nm * 0.5, atol=1e-17)


def test_finite_extreme_overrange_saturates_even_when_intermediate_float_overflows():
    config = replace(_config(), wuji_max_current_a=np.full(20, 1e-100),
                     wuji_current_bias_a=np.full(20, -1e308))
    limits = np.linspace(0.001, 0.002, 12)
    result = project_max_torque_linear(_axis_retarget(), np.eye(12), np.full(20, 1e308), config, limits)
    np.testing.assert_array_equal(result.wuji_load_fraction, config.wuji_effort_sign)
    np.testing.assert_allclose(result.model_torque_nm, -limits * config.wuji_effort_sign[:12], atol=1e-17)


def test_uncontrollable_zero_rows_and_small_nonzero_rows_are_not_fallbacks():
    config = _config()
    retarget = np.zeros((20, 12))
    retarget[3, 2] = 1e-100
    load = np.ones(20)
    result = project_max_torque_linear(retarget, np.eye(12), _effort(config, load), config, np.ones(12) * 0.005)
    expected = np.zeros(12)
    expected[2] = -0.005
    np.testing.assert_allclose(result.model_torque_nm, expected, atol=1e-17)
    empty = project_max_torque_linear(np.zeros((20, 12)), np.eye(12), _effort(config, load), config, np.ones(12))
    np.testing.assert_array_equal(empty.model_torque_nm, np.zeros(12))


def test_projection_is_frozen_copied_and_read_only():
    load, torque = np.zeros(20), np.zeros(12)
    result = LinearTorqueProjection(load, torque)
    load[:] = 1
    torque[:] = 1
    np.testing.assert_array_equal(result.wuji_load_fraction, 0)
    np.testing.assert_array_equal(result.model_torque_nm, 0)
    with pytest.raises(ValueError):
        result.wuji_load_fraction[0] = 1
    with pytest.raises(ValueError):
        result.model_torque_nm[0] = 1
    with pytest.raises(FrozenInstanceError):
        result.model_torque_nm = np.ones(12)


@pytest.mark.parametrize("field,value", [
    ("jacobian_wuji_glove", np.zeros((12, 20))),
    ("jacobian_wuji_glove", np.full((20, 12), np.nan)),
    ("motor_jacobian", np.zeros((12, 12))),
    ("motor_jacobian", np.diag([1] * 11 + [0])),
    ("motor_jacobian", np.full((12, 12), np.inf)),
    ("effort_a", np.zeros(19)), ("effort_a", np.full(20, np.nan)),
    ("effort_a", np.ones(20, dtype=bool)),
    ("model_torque_limits_nm", np.zeros(12)),
    ("model_torque_limits_nm", -np.ones(12)),
    ("model_torque_limits_nm", np.ones(11)),
    ("model_torque_limits_nm", np.full(12, np.inf)),
    ("config", None),
])
def test_projection_rejects_singular_nonfinite_wrong_shape_and_unit_ambiguous_inputs(field, value):
    kwargs = dict(jacobian_wuji_glove=_axis_retarget(), motor_jacobian=np.eye(12),
                  effort_a=np.zeros(20), config=_config(), model_torque_limits_nm=np.ones(12))
    kwargs[field] = value
    with pytest.raises(ValueError):
        project_max_torque_linear(**kwargs)


def test_coordinate_rescaling_cancels_but_each_nominal_model_limit_remains_independent():
    config = _config()
    motor = np.eye(12) + np.diag(np.full(11, 0.2), 1)
    retarget = np.arange(240, dtype=float).reshape(20, 12) / 100 - 0.7
    effort = _effort(config, np.linspace(-0.9, 0.8, 20))
    limits = np.linspace(0.001, 0.005, 12)
    baseline = project_max_torque_linear(retarget, motor, effort, config, limits)
    joint_scale = np.diag(np.linspace(0.2, 3.0, 12))
    transformed = project_max_torque_linear(retarget @ joint_scale, motor @ joint_scale, effort, config, limits)
    np.testing.assert_allclose(transformed.model_torque_nm, baseline.model_torque_nm, atol=1e-17)
    # 改变模型电机角尺度/符号时，上限按逆绝对尺度变换，不能混入raw角度。
    motor_scale = np.diag(np.array([(-1) ** i for i in range(12)]) * np.linspace(0.5, 2.5, 12))
    rescaled = project_max_torque_linear(retarget, motor_scale @ motor, effort, config,
                                         limits / np.abs(np.diag(motor_scale)))
    np.testing.assert_allclose(rescaled.model_torque_nm,
                               baseline.model_torque_nm / np.diag(motor_scale), atol=1e-17)


@pytest.mark.parametrize("hand", ["left", "right"])
def test_real_cad_coordinate_boundary_and_complete_retarget_have_independent_fd_oracle(hand):
    config = replace(_config(), glove_hand=hand, gain=0.6)
    pipeline = GloveWujiRetargetPipeline.from_yaml(
        hand, asset_root=ROOT / "assets" / "hand_motor_ve33",
        config_path=default_retarget_config(ROOT, hand), estimate_jacobian=True)
    model = pipeline.landmarks.model
    glove, pose = model.evaluate_motor_state({name: 0.04 + i * 0.002 for i, name in enumerate(model.motor_names)})
    q = glove.vector()
    retarget = pipeline.local_jacobian(dict(zip(GLOVE_JOINTS, q)), step_rad=0.01)
    kinematics = model.kinematics
    motor = kinematics._tree_derivatives(pose.all_positions)[kinematics._motor_indices]
    # oracle越过CAD q→m接口，中央差分不复用被测隐函数传动Jac。
    step = 1e-5
    fd_motor = np.column_stack([
        (model.motor_positions_from_joint_vector(q + np.eye(12)[i] * step)
         - model.motor_positions_from_joint_vector(q - np.eye(12)[i] * step)) / (2 * step)
        for i in range(12)
    ])
    np.testing.assert_allclose(motor, fd_motor, rtol=3e-3, atol=3e-5)
    # 用逆矩阵和标量逐项和构造独立投影oracle，不调用被测solve或归一化。
    oracle_mapping = np.linalg.inv(fd_motor).T @ retarget.T
    assert np.linalg.norm(oracle_mapping) > 0
    limits = np.linspace(0.002, 0.008, 12)
    load = np.linspace(-0.8, 0.9, 20)
    expected = np.array([
        -config.gain * limits[i] * sum(oracle_mapping[i, j] * load[j] for j in range(20))
        / sum(abs(oracle_mapping[i, j]) for j in range(20))
        if np.any(oracle_mapping[i]) else 0.0 for i in range(12)
    ])
    result = project_max_torque_linear(retarget, motor, _effort(config, load), config, limits)
    np.testing.assert_allclose(result.model_torque_nm, expected, rtol=4e-3, atol=3e-6)
    assert np.all(np.abs(result.model_torque_nm) <= config.gain * limits + 1e-15)
    # 每个真实CAD模型电机的最坏20轴联合符号输入达到它自己的限额。
    for axis in range(12):
        row = np.linalg.inv(motor).T[axis] @ retarget.T
        if not np.any(row):
            continue
        worst_load = np.sign(row)
        worst = project_max_torque_linear(retarget, motor, _effort(config, worst_load), config, limits)
        assert worst.model_torque_nm[axis] == pytest.approx(-config.gain * limits[axis], abs=1e-14)
        assert np.all(np.abs(worst.model_torque_nm) <= config.gain * limits + 1e-15)
