"""纯本机UDP曲线消费者回归；没有硬件连接或触觉标定。"""
from dataclasses import replace
import json
import socket
from types import SimpleNamespace
import time

import numpy as np
import pytest

from motorhand_teleop.domain import GLOVE_JOINTS, GlovePose, WujiState
from motorhand_teleop.real_hand.curve_telemetry import CurvePublisher, decode_sample
from motorhand_teleop.real_hand.force_feedback import ForceFeedbackCommand
from motorhand_teleop.real_hand.runtime import FollowMode, PairTick


def _source(unit):
    motor_names = tuple(f"model_{index}" for index in range(12))
    model = SimpleNamespace(motor_names=motor_names,
                            joint_motor_names=dict(zip(GLOVE_JOINTS, reversed(motor_names))))
    motors = {mid: SimpleNamespace(joint=joint, sign=(-1.0 if mid % 2 == 0 else 1.0), scale=1.5)
              for mid, joint in enumerate(GLOVE_JOINTS, 1)}
    model_target = tuple((index + 1) / (10_000 if unit == "Nm" else 100) for index in range(12))
    feedback = ForceFeedbackCommand(
        model_target if unit == "A" else None, None, None, "offline-only", 123,
        model_torque_nm=model_target if unit == "Nm" else None,
        wuji_load_fraction=np.linspace(-0.2, 0.2, 20) if unit == "Nm" else None,
        mapping_mode="max_torque_linear_estimate" if unit == "Nm" else "calibrated",
        projected_model_torque_nm=(tuple(value * 20 for value in model_target) if unit == "Nm" else None),
    )
    # 缺失电流不能被画成0：0、负数、None及非有限值必须区分。
    glove = SimpleNamespace(received_ns=1_040_000_000,
                            feedback_by_motor={1: {"current_a": 0.0, "received_ns": 1_040_000_000},
                                               2: {"current_a": -0.3, "received_ns": 1_040_000_123},
                                               4: {"current_a": float("nan")}})
    glove.raw_motor_positions_deg = {mid: mid * 17.0 - 100 for mid in range(1, 13)}
    hand = WujiState(np.linspace(-0.2, 0.1, 20), np.linspace(-0.3, 0.4, 20),
                     np.linspace(-0.4, 0.6, 20), 1_020_000_000, 20)
    tick = PairTick(FollowMode.GLOVE_TO_HAND2, hand,
                    GlovePose(dict(zip(GLOVE_JOINTS, np.linspace(-1.2, 1.5, 12))), timestamp_ns=glove.received_ns),
                    np.full(20, 0.9), np.linspace(0.05, 0.4, 20), None,
                    1_100_000_000, False, glove, feedback)
    return model, motors, model_target, tick


@pytest.mark.parametrize("unit", ["Nm", "A"])
def test_udp_consumer_receives_raw_axis_sign_scale_without_faking_missing_measurements(unit):
    model, motors, target, tick = _source(unit)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(1)
        publisher = CurvePublisher(receiver.getsockname()[1], glove_model=model, motors=motors,
                                   wuji_joint_names=[f"semantic_{i}" for i in range(20)])
        try:
            publisher.publish(tick)
            sample = decode_sample(receiver.recvfrom(16_384)[0])
        finally:
            publisher.close()
    expected = [target[11 - index] * (-1 if index % 2 else 1) * 1.5 for index in range(12)]
    np.testing.assert_allclose(sample["glove_target"], expected)
    assert sample["glove_target_unit"] == unit
    assert sample["glove_current_a"] == [0.0, -0.3] + [None] * 10
    assert sample["executed"] is False
    assert sample["wuji_received_ns"] != sample["glove_received_ns"]
    np.testing.assert_allclose(sample["wuji_position_rad"], tick.hand2_state.positions_rad)
    np.testing.assert_allclose(sample["wuji_target_rad"], tick.output_positions_rad)
    np.testing.assert_allclose(sample["glove_position_rad"], tick.glove_pose.vector())
    assert sample["glove_position_names"] == list(GLOVE_JOINTS)
    assert sample["glove_raw_position_deg"] == [mid * 17.0 - 100 for mid in range(1, 13)]
    assert sample["glove_motor_received_ns"] == [1_040_000_000, 1_040_000_123] + [None] * 10
    assert sample["glove_target_position_rad"] is None
    assert sample["cycle_started_ns"] is None
    assert sample["wuji_submit_ns"] is None and sample["glove_submit_ns"] is None
    assert abs(time.monotonic_ns() - sample["published_ns"]) < 10_000_000_000
    np.testing.assert_allclose(sample["wuji_tracking_error_rad"],
                               tick.output_positions_rad - tick.hand2_state.positions_rad)
    np.testing.assert_allclose(sample["wuji_velocity_rad_s"], tick.hand2_state.velocities_rad_s)
    assert sample["contact_reflection_gain"] is None
    if unit == "Nm":
        np.testing.assert_allclose(sample["glove_projected_torque_nm"], np.array(expected) * 20)
    if unit == "A":
        assert sample["wuji_load_fraction"] == [None] * 20
        assert sample["glove_projected_torque_nm"] == [None] * 12


def test_console_reports_confirmed_nominal_torque_in_raw_motor_order():
    import re

    from motorhand_teleop.real_hand.curve_telemetry import _motor_axis_bindings
    from motorhand_teleop.real_hand.runtime import _format_torque_line

    model, motors, _, tick = _source("Nm")
    confirmed = tuple((index + 1) / 10 for index in range(12))
    tick = replace(tick, executed=True,
                   force_feedback=replace(tick.force_feedback, model_torque_nm=confirmed))
    bindings = _motor_axis_bindings(model, motors)
    line = _format_torque_line(tick, bindings)
    axes = [(int(mid), float(value)) for mid, value in re.findall(r"J(\d+)=([+-]\d+\.\d+)", line)]
    assert [mid for mid, _ in axes] == list(range(1, 13))
    expected = [confirmed[11 - index] * (-1 if index % 2 else 1) * 1.5 for index in range(12)]
    np.testing.assert_allclose([value for _, value in axes], expected, atol=0.0005)
    assert "SDK已提交" in line
    assert "只读预览" in _format_torque_line(replace(tick, executed=False), bindings)


def test_curve_socket_failure_does_not_raise_into_control_or_retry_output():
    model, motors, _, tick = _source("Nm")
    publisher = CurvePublisher(8765, glove_model=model, motors=motors,
                               wuji_joint_names=[f"semantic_{i}" for i in range(20)])
    publisher.close()
    publisher.publish(tick)
    assert publisher.dropped_packets == 1


@pytest.mark.parametrize("mutate", [
    lambda sample: sample.update(schema_version=2),
    lambda sample: sample.update(source="unknown"),
    lambda sample: sample.update(feedback_state="enabled"),
    lambda sample: sample.update(wuji_current_a=[0.0] * 12),
    lambda sample: sample.update(glove_target=[True] * 12),
    lambda sample: sample.update(glove_current_a=[float("nan")] * 12),
    lambda sample: sample.update(wuji_received_ns=True),
    lambda sample: sample.update(glove_target_unit=None),
])
def test_receiver_rejects_wrong_channel_units_or_unknown_measurements(mutate):
    from motorhand_teleop.real_hand.force_curves import make_demo_sample

    sample = make_demo_sample(1.0, 1_000_000_000)
    mutate(sample)
    with pytest.raises(ValueError):
        decode_sample(json.dumps(sample).encode())


@pytest.mark.parametrize("mutate", [
    lambda sample: sample.update(contact_state="enabled"),
    lambda sample: sample.update(contact_active_mask=[1] * 20),
    lambda sample: sample.update(virtual_hand_torque_nm=[None] * 20),
    lambda sample: sample.update(glove_target_unit="A"),
    lambda sample: sample.update(mapping_mode="max_torque_linear_estimate"),
])
def test_contact_packet_requires_explicit_estimated_state_mask_and_nominal_units(mutate):
    from motorhand_teleop.real_hand.force_curves import make_demo_sample

    sample = make_demo_sample(1.0, 1_000_000_000)
    sample.update(mapping_mode="contact_impedance_estimate", contact_state="free",
                  contact_active_mask=[False] * 20, virtual_hand_torque_nm=[0.0] * 20)
    mutate(sample)
    with pytest.raises(ValueError):
        decode_sample(json.dumps(sample).encode())


@pytest.mark.parametrize("feedback_enabled", [False, True])
def test_reverse_mode_and_missing_velocity_keep_diagnostics_missing(feedback_enabled):
    model, motors, _, tick = _source("A")
    tick = replace(tick, mode=FollowMode.HAND2_TO_GLOVE,
                   hand2_state=replace(tick.hand2_state, velocities_rad_s=None),
                   output_positions_rad=np.linspace(-0.7, 0.8, 12),
                   force_feedback=tick.force_feedback if feedback_enabled else None)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(1)
        publisher = CurvePublisher(receiver.getsockname()[1], glove_model=model, motors=motors,
                                   wuji_joint_names=[f"semantic_{i}" for i in range(20)])
        try:
            publisher.publish(tick)
            sample = decode_sample(receiver.recvfrom(16_384)[0])
        finally:
            publisher.close()
    assert sample["wuji_tracking_error_rad"] == [None] * 20
    assert sample["wuji_velocity_rad_s"] == [None] * 20
    assert sample["glove_projected_torque_nm"] == [None] * 12
    assert sample["contact_reflection_gain"] is None
    assert sample["wuji_target_rad"] is None
    np.testing.assert_allclose(sample["glove_target_position_rad"], np.linspace(-0.7, 0.8, 12))
    np.testing.assert_allclose(sample["wuji_position_rad"], tick.hand2_state.positions_rad)


def test_legacy_schema_without_diagnostics_does_not_create_zero_values():
    from motorhand_teleop.real_hand.force_curves import make_demo_sample

    sample = make_demo_sample(1.0, 1_000_000_000)
    keys = ("glove_projected_torque_nm", "contact_reflection_gain",
            "wuji_tracking_error_rad", "wuji_velocity_rad_s", "glove_motor_received_ns")
    for key in keys:
        sample.pop(key, None)
    decoded = decode_sample(json.dumps(sample).encode())
    assert not any(key in decoded for key in keys)


@pytest.mark.parametrize("key,count", [
    ("glove_projected_torque_nm", 12),
    ("wuji_tracking_error_rad", 20),
    ("wuji_velocity_rad_s", 20),
    ("wuji_position_rad", 20),
    ("glove_position_rad", 12),
    ("glove_raw_position_deg", 12),
    ("wuji_target_rad", 20),
    ("glove_target_position_rad", 12),
])
@pytest.mark.parametrize("invalid", ["short", "bool", "string", "nonfinite", "null"])
def test_optional_diagnostic_vectors_validate_dimension_and_numeric_type(key, count, invalid):
    from motorhand_teleop.real_hand.force_curves import make_demo_sample

    sample = make_demo_sample(1.0, 1_000_000_000)
    values = [None] * count
    if invalid == "short":
        values.pop()
    elif invalid == "null":
        values = None
    else:
        values[0] = {"bool": True, "string": "0.1", "nonfinite": float("inf")}[invalid]
    sample[key] = values
    if invalid == "null" and key in ("wuji_position_rad", "glove_position_rad",
                                     "glove_raw_position_deg", "wuji_target_rad",
                                     "glove_target_position_rad"):
        assert decode_sample(json.dumps(sample).encode())[key] is None
        return
    with pytest.raises(ValueError):
        decode_sample(json.dumps(sample).encode())


@pytest.mark.parametrize("gain", [True, "2000", 0, -1, float("inf")])
def test_optional_contact_gain_requires_positive_finite_numeric_value(gain):
    from motorhand_teleop.real_hand.force_curves import make_demo_sample

    sample = make_demo_sample(1.0, 1_000_000_000)
    sample.update(mapping_mode="contact_impedance_estimate", contact_state="free",
                  contact_active_mask=[False] * 20, virtual_hand_torque_nm=[0.0] * 20,
                  contact_reflection_gain=gain)
    with pytest.raises(ValueError):
        decode_sample(json.dumps(sample).encode())


@pytest.mark.parametrize("mutate", [
    lambda sample: sample.update(glove_target_unit="A", glove_projected_torque_nm=[0.1] * 12),
    lambda sample: sample.update(mode="hand2-to-glove", wuji_tracking_error_rad=[0.1] * 20),
    lambda sample: sample.update(mapping_mode="calibrated", contact_reflection_gain=2000),
])
def test_optional_diagnostics_cannot_claim_wrong_output_units_or_control_mode(mutate):
    from motorhand_teleop.real_hand.force_curves import make_demo_sample

    sample = make_demo_sample(1.0, 1_000_000_000)
    mutate(sample)
    with pytest.raises(ValueError):
        decode_sample(json.dumps(sample).encode())




@pytest.mark.parametrize("projected", [np.zeros(11), [True] * 12, [float("inf")] * 12])
def test_force_command_rejects_invalid_projected_torque(projected):
    _, _, _, tick = _source("Nm")
    with pytest.raises(ValueError):
        replace(tick.force_feedback, projected_model_torque_nm=projected)


def test_force_command_rejects_nm_projection_attached_to_current_output():
    _, _, _, tick = _source("A")
    with pytest.raises(ValueError):
        replace(tick.force_feedback, projected_model_torque_nm=np.zeros(12))


@pytest.mark.parametrize("key", ["published_ns", "cycle_started_ns", "cycle_completed_ns",
                                  "wuji_submit_ns", "glove_submit_ns"])
@pytest.mark.parametrize("value", [True, -1, 1.5, "1"])
def test_optional_timestamps_reject_invalid_values(key, value):
    from motorhand_teleop.real_hand.force_curves import make_demo_sample

    sample = make_demo_sample(1.0, 1_000_000_000)
    sample[key] = value
    with pytest.raises(ValueError):
        decode_sample(json.dumps(sample).encode())


@pytest.mark.parametrize("stamps", [[0] * 11, [0] * 13, [True] + [None] * 11,
                                    [-1] + [None] * 11, [1.0] + [None] * 11,
                                    ["1"] + [None] * 11, "invalid"])
def test_motor_receive_timestamps_require_twelve_optional_nonnegative_integers(stamps):
    from motorhand_teleop.real_hand.force_curves import make_demo_sample

    sample = make_demo_sample(1.0, 1_000_000_000)
    sample["glove_motor_received_ns"] = stamps
    with pytest.raises(ValueError):
        decode_sample(json.dumps(sample).encode())


@pytest.mark.parametrize("mode", list(FollowMode))
@pytest.mark.parametrize("execute", [False, True])
def test_runtime_samples_and_sdk_submission_times_reach_udp_consumer(mode, execute):
    from test_real_hand_runtime import _pair, bridge

    model_bridge = bridge.__wrapped__()
    runtime, hand, glove, clock = _pair(model_bridge, mode=mode, execute=execute)
    original_read = glove.read_state

    def read_glove():
        state = original_read()
        state.feedback_by_motor = {}
        return state

    glove.read_state = read_glove
    original_send = hand.send_target if mode is FollowMode.GLOVE_TO_HAND2 else glove.send_position
    submissions = []

    def send(target, *, deadline_ns):
        original_send(target, deadline_ns=deadline_ns)
        clock.now += 17
        submissions.append((clock.now, deadline_ns))

    if mode is FollowMode.GLOVE_TO_HAND2:
        hand.send_target = send
    else:
        glove.send_position = send
    runtime.start()
    clock.now += 100
    started = clock.now
    tick = runtime.step()
    assert tick.cycle_started_ns == started
    assert tick.deadline_ns == started + runtime.command_timeout_ns
    assert tick.cycle_completed_ns == clock.now
    expected = submissions[0][0] if execute else None
    assert tick.wuji_submit_ns == (expected if mode is FollowMode.GLOVE_TO_HAND2 else None)
    assert tick.glove_submit_ns == (expected if mode is FollowMode.HAND2_TO_GLOVE else None)
    if execute:
        assert submissions[0][1] == tick.deadline_ns
    model, motors, _, _ = _source("A")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(1)
        publisher = CurvePublisher(receiver.getsockname()[1], glove_model=model, motors=motors,
                                   wuji_joint_names=[f"semantic_{i}" for i in range(20)],
                                   clock_ns=clock)
        try:
            publisher.publish(tick)
            sample = decode_sample(receiver.recvfrom(16_384)[0])
        finally:
            publisher.close()
            runtime.close()
    assert sample["published_ns"] == clock.now
    assert sample["cycle_started_ns"] == started
    assert sample["cycle_completed_ns"] == tick.cycle_completed_ns
    assert sample["wuji_submit_ns"] == tick.wuji_submit_ns
    assert sample["glove_submit_ns"] == tick.glove_submit_ns
    assert sample["wuji_received_ns"] == tick.hand2_state.received_ns
    assert sample["glove_received_ns"] == tick.glove_state.received_ns
    np.testing.assert_allclose(sample["wuji_position_rad"], hand.positions)
    np.testing.assert_allclose(sample["glove_position_rad"], tick.glove_pose.vector())
    np.testing.assert_allclose(sample["glove_raw_position_deg"],
                               [tick.glove_state.raw_motor_positions_deg[i] for i in range(1, 13)])
    target_key = "wuji_target_rad" if mode is FollowMode.GLOVE_TO_HAND2 else "glove_target_position_rad"
    np.testing.assert_allclose(sample[target_key], tick.output_positions_rad)


def test_force_feedback_records_each_sdk_submission_after_return():
    from test_real_hand_runtime import bridge, force_pair

    fixture = force_pair.__wrapped__(bridge.__wrapped__())
    runtime, hand, glove, clock = next(fixture)
    original_hand = hand.send_target
    original_glove = glove.send_force_current
    returned = []

    def send_hand(target, *, deadline_ns):
        original_hand(target, deadline_ns=deadline_ns)
        clock.now += 11
        returned.append(("wuji", clock.now))

    def send_glove(target, *, deadline_ns):
        result = original_glove(target, deadline_ns=deadline_ns)
        clock.now += 13
        returned.append(("glove", clock.now))
        return result

    hand.send_target = send_hand
    glove.send_force_current = send_glove
    try:
        runtime.start()
        tick = runtime.step()
        assert returned == [("wuji", tick.wuji_submit_ns), ("glove", tick.glove_submit_ns)]
        assert tick.cycle_started_ns <= tick.wuji_submit_ns < tick.glove_submit_ns <= tick.cycle_completed_ns
    finally:
        fixture.close()
