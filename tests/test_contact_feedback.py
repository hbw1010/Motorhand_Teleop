"""只经纯估计器公共接口验证；所有参数是离线合成假设，不是实测接触力。"""
import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from wujihand2_force_glove.domain import WujiState
from wujihand2_force_glove.feedback import CalibrationRequired
from wujihand2_force_glove.real_hand.contact_feedback import (
    ContactFeedbackConfig,
    ContactImpedance,
)

START_NS = 1_000_000_000


def _data(**changes):
    data = dict(
        schema_version=1,
        estimate_acknowledged=True,
        source_id="synthetic-contact-estimate-not-calibrated",
        current_on_a=[0.12] * 20,
        current_off_a=[0.06] * 20,
        error_on_rad=[0.10] * 20,
        error_off_rad=[0.03] * 20,
        stalled_speed_rad_s=[0.2] * 20,
        stiffness_nm_per_rad=[0.1] * 20,
        damping_nm_s_per_rad=[0.003] * 20,
    )
    data.update(changes)
    return data


def _step(model, seconds, *, target=0.2, position=0.0, current=0.3,
          velocity=0.0, observable=None, bias=0.0, received_seconds=None):
    source_ns = START_NS + round(seconds * 1e9)
    received_ns = (source_ns if received_seconds is None
                   else START_NS + round(received_seconds * 1e9))
    state = WujiState(
        np.broadcast_to(position, (20,)).copy(),
        None if velocity is None else np.broadcast_to(velocity, (20,)).copy(),
        None if current is None else np.broadcast_to(current, (20,)).copy(),
        received_ns, 20,
    )
    return model.step(
        np.broadcast_to(target, (20,)).copy(), state, source_ns=source_ns,
        observable_joints=np.ones(20, dtype=bool) if observable is None else observable,
        current_bias_a=np.broadcast_to(bias, (20,)).copy(),
    )


def _contact(model, **kwargs):
    result = None
    for seconds in (0.0, 0.05, 0.10, 0.15):
        result = _step(model, seconds, **kwargs)
    assert result.phase == "contact"
    assert result.active_mask.all()
    return result


@pytest.mark.parametrize("kwargs", [
    {"current": 0.0},
    {"target": 0.0},
    {"velocity": 0.21},
    {"velocity": -0.21},
    {"current": 0.3, "bias": 0.3},
])
def test_load_error_and_stalled_motion_are_all_required(kwargs):
    model = ContactImpedance(ContactFeedbackConfig(**_data()))
    for seconds in np.arange(0.0, 0.55, 0.05):
        result = _step(model, float(seconds), **kwargs)
        assert result.phase == "free"
        assert not result.active_mask.any()
        assert not result.virtual_hand_torque_nm.any()


def test_dwell_uses_source_time_and_requires_continuous_candidate():
    model = ContactImpedance(ContactFeedbackConfig(**_data()))
    assert _step(model, 0.0).phase == "arming"
    assert _step(model, 0.05).phase == "arming"
    assert _step(model, 0.10, velocity=0.3).phase == "free"
    assert _step(model, 0.15).phase == "arming"
    assert _step(model, 0.20).phase == "arming"
    assert _step(model, 0.25).phase == "arming"
    result = _step(model, 0.30)
    assert result.phase == "contact"
    assert np.all(result.virtual_hand_torque_nm < 0)


def test_threshold_equality_and_hysteresis_keep_contact_until_off_dwell():
    model = ContactImpedance(ContactFeedbackConfig(**_data()))
    _contact(model, target=0.10, current=0.12, velocity=0.2)
    result = _step(model, 0.20, target=0.08, current=0.09)
    assert result.phase == "contact"
    assert np.all(result.virtual_hand_torque_nm < 0)
    result = _step(model, 0.25, target=0.03)
    assert result.phase == "releasing"
    assert result.active_mask.all()
    assert not result.virtual_hand_torque_nm.any()
    assert _step(model, 0.30, target=0.03).phase == "releasing"
    result = _step(model, 0.31, target=0.03)
    assert result.phase == "free"
    assert not result.active_mask.any()


def test_current_bias_is_subtracted_before_absolute_load_and_filtering():
    model = ContactImpedance(ContactFeedbackConfig(**_data()))
    result = _contact(model, current=-0.1, bias=0.2)
    assert np.all(result.virtual_hand_torque_nm < 0)
    for seconds in np.arange(0.2, 0.8, 0.05):
        result = _step(model, float(seconds), current=0.2, bias=0.2)
    assert result.phase == "free"
    assert not result.virtual_hand_torque_nm.any()


def test_current_low_pass_rejects_a_single_high_load_frame():
    model = ContactImpedance(ContactFeedbackConfig(**_data(current_filter_hz=1.0)))
    _step(model, 0.0, current=0)
    assert _step(model, 0.02, current=0.3).phase == "free"
    for seconds in np.arange(0.04, 0.4, 0.02):
        result = _step(model, float(seconds), current=0)
    assert result.phase == "free"
    assert not result.virtual_hand_torque_nm.any()


def test_missing_velocity_needs_two_distinct_source_and_hand_frames():
    model = ContactImpedance(ContactFeedbackConfig(**_data()))
    assert _step(model, 0.0, velocity=None).phase == "free"
    for _ in range(10):
        assert _step(model, 0.0, velocity=None).phase == "free"
    assert _step(model, 0.05, velocity=None).phase == "arming"
    assert _step(model, 0.10, velocity=None).phase == "arming"
    assert _step(model, 0.15, velocity=None).phase == "arming"
    assert _step(model, 0.20, velocity=None).phase == "contact"

    model.reset()
    assert _step(model, 0.0, velocity=None, received_seconds=0.1).phase == "free"
    assert _step(model, 0.05, velocity=None, received_seconds=0.1).phase == "free"
    assert _step(model, 0.1, velocity=None, received_seconds=0.1).phase == "free"


def test_position_difference_detects_motion_instead_of_fabricating_zero_speed():
    model = ContactImpedance(ContactFeedbackConfig(**_data()))
    for index in range(8):
        position = index * 0.02
        result = _step(model, index * 0.05, velocity=None,
                       position=position, target=position + 0.2)
        assert result.phase == "free"
        assert not result.virtual_hand_torque_nm.any()


def test_duplicate_source_does_not_advance_dwell_output_filter_or_target_derivative():
    model = ContactImpedance(ContactFeedbackConfig(**_data()))
    first = _step(model, 0.0)
    for _ in range(20):
        result = _step(model, 0.0, received_seconds=0.09)
        assert result.phase == first.phase == "arming"
        assert not result.active_mask.any()
    _step(model, 0.05)
    _step(model, 0.10)
    contact = _step(model, 0.15)
    for _ in range(20):
        result = _step(model, 0.15, received_seconds=0.20)
        np.testing.assert_array_equal(result.virtual_hand_torque_nm,
                                      contact.virtual_hand_torque_nm)
    assert _step(model, 0.15, target=-0.2).phase == "releasing"
    assert not _step(model, 0.15, target=-0.2).virtual_hand_torque_nm.any()


def test_large_gap_clears_contact_and_velocity_history_and_restarts_dwell():
    model = ContactImpedance(ContactFeedbackConfig(**_data()))
    _contact(model)
    result = _step(model, 0.30, velocity=None)
    assert result.phase == "free"
    assert not result.virtual_hand_torque_nm.any()
    assert _step(model, 0.35, velocity=None).phase == "arming"
    assert _step(model, 0.40, velocity=None).phase == "arming"
    assert _step(model, 0.45, velocity=None).phase == "arming"
    assert _step(model, 0.50, velocity=None).phase == "contact"


def test_candidate_direction_change_restarts_dwell():
    model = ContactImpedance(ContactFeedbackConfig(**_data()))
    _step(model, 0)
    _step(model, 0.05)
    assert _step(model, 0.10, target=-0.2).phase == "arming"
    assert _step(model, 0.15, target=-0.2).phase == "arming"
    assert _step(model, 0.20, target=-0.2).phase == "arming"
    result = _step(model, 0.25, target=-0.2)
    assert result.phase == "contact"
    assert np.all(result.virtual_hand_torque_nm > 0)


def test_release_reversal_clears_filter_history_before_opposite_contact():
    model = ContactImpedance(ContactFeedbackConfig(**_data(output_filter_hz=0.5)))
    _contact(model)
    for seconds in (0.20, 0.25):
        result = _step(model, seconds, target=-0.2)
        assert result.phase == "releasing"
        assert not result.virtual_hand_torque_nm.any()
    assert _step(model, 0.26, target=-0.2).phase == "free"
    for seconds in (0.30, 0.35, 0.40):
        result = _step(model, seconds, target=-0.2)
        assert not result.virtual_hand_torque_nm.any()
    result = _step(model, 0.45, target=-0.2)
    assert result.phase == "contact"
    assert np.all(result.virtual_hand_torque_nm > 0)


def test_cancelled_release_resumes_only_from_zero_not_residual_filter_history():
    model = ContactImpedance(ContactFeedbackConfig(**_data(output_filter_hz=0.5)))
    _contact(model)
    for seconds in np.arange(0.20, 0.60, 0.05):
        old = _step(model, float(seconds))
    assert not _step(model, 0.60, target=0.03).virtual_hand_torque_nm.any()
    resumed = _step(model, 0.65)
    assert resumed.phase == "contact"
    assert np.all(np.abs(resumed.virtual_hand_torque_nm) < np.abs(old.virtual_hand_torque_nm))


def test_software_spring_and_damping_are_unilateral_and_use_relative_velocity():
    config = ContactFeedbackConfig(**_data(damping_nm_s_per_rad=[0.1] * 20,
                                           output_filter_hz=1e6))
    model = ContactImpedance(config)
    result = _contact(model)
    # 0.1 Nm/rad * (0.20 - 0.03) rad = 0.017 virtual Nm.
    np.testing.assert_allclose(result.virtual_hand_torque_nm, -0.017, atol=1e-12)
    result = _step(model, 0.20, target=0.21)
    # 目标增长0.01rad/0.05s，增加0.1 Nm*s/rad * 0.2rad/s阻尼。
    np.testing.assert_allclose(result.virtual_hand_torque_nm, -0.038, atol=1e-12)
    result = _step(model, 0.25, target=0.20)
    np.testing.assert_allclose(result.virtual_hand_torque_nm, -0.017, atol=1e-12)
    result = _step(model, 0.30, target=0.20, velocity=0.1)
    np.testing.assert_allclose(result.virtual_hand_torque_nm, -0.017, atol=1e-12)
    result = _step(model, 0.35, target=0.20, velocity=-0.1)
    np.testing.assert_allclose(result.virtual_hand_torque_nm, -0.027, atol=1e-12)


def test_output_low_pass_uses_elapsed_source_time():
    model = ContactImpedance(ContactFeedbackConfig(**_data(output_filter_hz=1.0)))
    first = _contact(model)
    # 一阶1Hz连续极点在50ms的阶跃响应，独立已算数值。
    np.testing.assert_allclose(first.virtual_hand_torque_nm, -0.004583154252,
                               atol=1e-12)
    second = _step(model, 0.20)
    assert np.all(np.abs(second.virtual_hand_torque_nm) > np.abs(first.virtual_hand_torque_nm))
    assert np.all(np.abs(second.virtual_hand_torque_nm) < 0.017)


def test_current_off_clears_output_before_release_dwell_and_does_not_rearm_early():
    model = ContactImpedance(ContactFeedbackConfig(**_data(current_filter_hz=1e6)))
    _contact(model)
    result = _step(model, 0.20, current=0.06)
    assert result.phase == "releasing"
    assert result.active_mask.all()
    assert not result.virtual_hand_torque_nm.any()
    assert _step(model, 0.25, current=0.06).phase == "releasing"
    result = _step(model, 0.26, current=0.06)
    assert result.phase == "free"
    assert not result.active_mask.any()
    assert _step(model, 0.30, current=0.09).phase == "free"


def test_exact_maximum_gap_does_not_restart_an_existing_contact():
    model = ContactImpedance(ContactFeedbackConfig(**_data()))
    _contact(model)
    result = _step(model, 0.25)
    assert result.phase == "contact"
    assert result.active_mask.all()
    assert np.all(result.virtual_hand_torque_nm < 0)


def test_duplicate_target_reversal_invalidates_arming_without_advancing_time():
    model = ContactImpedance(ContactFeedbackConfig(**_data()))
    _step(model, 0.0)
    _step(model, 0.05)
    result = _step(model, 0.05, target=-0.2)
    assert result.phase == "free"
    assert _step(model, 0.10).phase == "arming"
    assert _step(model, 0.15).phase == "arming"
    assert _step(model, 0.20).phase == "arming"
    assert _step(model, 0.25).phase == "contact"


def test_each_axis_requires_its_own_error_and_load_without_cross_axis_contact():
    model = ContactImpedance(ContactFeedbackConfig(**_data()))
    target = np.zeros(20)
    target[0] = 0.2
    current = np.zeros(20)
    current[1] = 0.3
    for seconds in (0.0, 0.05, 0.10, 0.15, 0.20):
        result = _step(model, seconds, target=target, current=current)
        assert result.phase == "free"
        assert not result.active_mask.any()
    current[0] = 0.3
    for seconds in (0.25, 0.30, 0.35, 0.40, 0.45):
        result = _step(model, seconds, target=target, current=current)
    assert result.active_mask[0]
    assert not result.active_mask[1:].any()
    assert result.virtual_hand_torque_nm[0] < 0
    assert not result.virtual_hand_torque_nm[1:].any()


def test_unobservable_axes_clear_immediately_and_rearm_independently():
    model = ContactImpedance(ContactFeedbackConfig(**_data()))
    _contact(model)
    mask = np.ones(20, dtype=bool)
    mask[[0, 5, 9]] = False
    result = _step(model, 0.15, observable=mask)
    assert result.phase == "contact"
    assert not result.active_mask[[0, 5, 9]].any()
    assert not result.virtual_hand_torque_nm[[0, 5, 9]].any()
    assert result.active_mask[1]
    result = _step(model, 0.20)
    assert not result.active_mask[[0, 5, 9]].any()
    for seconds in (0.25, 0.30, 0.35):
        result = _step(model, seconds)
    assert result.active_mask.all()


def test_missing_current_clears_existing_contact_without_fake_load():
    model = ContactImpedance(ContactFeedbackConfig(**_data()))
    _contact(model)
    result = _step(model, 0.20, current=None)
    assert result.phase == "free"
    assert not result.active_mask.any()
    assert not result.virtual_hand_torque_nm.any()
    assert _step(model, 0.25).phase == "arming"


def test_reset_clears_derivative_contact_and_filter_history():
    model = ContactImpedance(ContactFeedbackConfig(**_data()))
    old = _contact(model)
    model.reset()
    result = _step(model, 0.0, velocity=None)
    assert result.phase == "free"
    assert not result.active_mask.any()
    assert not result.virtual_hand_torque_nm.any()
    assert old.active_mask.all()
    assert np.all(old.virtual_hand_torque_nm < 0)


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 2),
    ("estimate_acknowledged", 1), ("estimate_acknowledged", False),
    ("source_id", " "), ("source_id", None),
    ("current_on_a", [0.0] * 20), ("current_off_a", [0.12] * 20),
    ("current_off_a", [-0.1] * 20),
    ("error_on_rad", [-0.1] * 20), ("error_off_rad", [0.10] * 20),
    ("error_off_rad", [-0.1] * 20),
    ("stalled_speed_rad_s", [0] * 20),
    ("stiffness_nm_per_rad", [0] * 20),
    ("damping_nm_s_per_rad", [-0.1] * 20),
    ("current_on_a", [0.1] * 19), ("current_on_a", [True] * 20),
    ("current_on_a", ["0.1"] * 20), ("current_on_a", [complex(1)] * 20),
    ("error_on_rad", [float("nan")] * 20),
    ("dwell_s", True), ("release_dwell_s", 0),
    ("current_filter_hz", float("inf")), ("output_filter_hz", "8"),
    ("max_sample_gap_s", -0.1),
    ("reflection_gain", 0), ("reflection_gain", -1), ("reflection_gain", True),
    ("reflection_gain", float("nan")), ("reflection_gain", float("inf")),
    ("reflection_gain", "1000"),
])
def test_config_rejects_invalid_parameters(field, value):
    with pytest.raises(ValueError):
        ContactFeedbackConfig(**_data(**{field: value}))


def test_configuration_and_results_own_read_only_arrays_and_json_loading(tmp_path):
    data = _data()
    path = tmp_path / "contact.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    config = ContactFeedbackConfig.from_json(path)
    data["current_on_a"][0] = 10
    assert config.current_on_a[0] == 0.12
    with pytest.raises(ValueError):
        config.current_on_a[0] = 10
    with pytest.raises(FrozenInstanceError):
        config.source_id = "different"
    model = ContactImpedance(config)
    result = _contact(model)
    for array in (result.virtual_hand_torque_nm, result.active_mask, result.tracking_error_rad):
        with pytest.raises(ValueError):
            array[0] = 0
    _step(model, 0.20, target=-0.2)
    assert result.active_mask.all()
    assert np.all(result.virtual_hand_torque_nm < 0)
    assert result.active_mask.dtype == np.bool_
    with pytest.raises(CalibrationRequired):
        ContactFeedbackConfig.from_json(tmp_path / "missing.json")
    for invalid in ([], {"schema_version": 1}, dict(data, unknown=1)):
        path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(CalibrationRequired):
            ContactFeedbackConfig.from_json(path)


@pytest.mark.parametrize("source_ns", [-1, True, 1.5, START_NS + 1])
def test_source_timestamp_is_strict_nonnegative_and_not_later_than_hand(source_ns):
    model = ContactImpedance(ContactFeedbackConfig(**_data()))
    state = WujiState(np.zeros(20), np.zeros(20), np.ones(20), START_NS, 20)
    with pytest.raises(ValueError):
        model.step(np.ones(20), state, source_ns=source_ns,
                   observable_joints=np.ones(20, dtype=bool), current_bias_a=np.zeros(20))


def test_source_timestamp_must_not_move_backwards():
    model = ContactImpedance(ContactFeedbackConfig(**_data()))
    _step(model, 0.05)
    with pytest.raises(ValueError):
        _step(model, 0.0)


@pytest.mark.parametrize("name,value", [
    ("target_positions_rad", np.zeros(19)),
    ("target_positions_rad", np.full(20, np.nan)),
    ("current_bias_a", np.ones(20, dtype=bool)),
    ("observable_joints", np.ones(20)),
    ("observable_joints", np.ones(19, dtype=bool)),
])
def test_step_validates_public_input_dimensions_and_types(name, value):
    model = ContactImpedance(ContactFeedbackConfig(**_data()))
    state = WujiState(np.zeros(20), np.zeros(20), np.ones(20), START_NS, 20)
    kwargs = dict(target_positions_rad=np.ones(20), hand_state=state, source_ns=START_NS,
                  observable_joints=np.ones(20, dtype=bool), current_bias_a=np.zeros(20))
    kwargs[name] = value
    with pytest.raises(ValueError):
        model.step(**kwargs)
