import time
import json

import numpy as np
import pytest

from motorhand_teleop.domain import GLOVE_JOINTS, GlovePose
from motorhand_teleop.real_hand.motor_hand import (
    EncosCalibrationRequired,
    EncosGloveHardwareDriver,
)
from motorhand_teleop.ve33_adapter import Ve33GloveModel
from test_ve33_adapter import cad_oracle


class FakeEncosHand:
    def __init__(self):
        self.commands = []
        self.enabled = []
        self.closed = False
        self.angles = {motor_id: float(motor_id) * 0.2 for motor_id in range(1, 13)}
        self.offline = set()
        self.errors = {}

    def connect(self):
        return None

    def scan(self):
        return {1: list(range(1, 5)), 2: list(range(5, 9)), 3: list(range(9, 13))}

    def get_online_ids(self):
        return tuple(mid for mid in range(1, 13) if mid not in self.offline)

    def get_all_angles(self):
        return dict(self.angles)

    def get_all_feedbacks(self):
        now = time.monotonic_ns()
        return {motor_id: {"angle_deg": angle, "current_a": 0.1,
                           "err_code": self.errors.get(motor_id, 0), "received_ns": now}
                for motor_id, angle in self.angles.items()}

    def set_position(self, **kwargs):
        self.commands.append(("position", kwargs))
        if kwargs["speed_rpm"] > 0.0:
            self.angles[kwargs["motor_id"]] = kwargs["target_deg"]

    def set_mit(self, **kwargs):
        self.commands.append(("mit", kwargs))

    def enable(self, motor_id, enable=True):
        self.enabled.append((motor_id, enable))

    def close(self):
        self.closed = True


def _connect_for_output(driver, backend):
    driver.connect()
    angles = dict(backend.angles)
    driver.home()
    # 单元场景从已回零、随后由人手移动到采样姿态开始。
    backend.angles = angles
    backend.commands.clear()
    backend.enabled.clear()


def _mapping():
    return {
        motor_id: {
            "joint": joint,
            "zero_deg": 0.0,
            "sign": 1.0,
            "scale": 1.0,
            "min_deg": -180.0,
            "max_deg": 180.0,
            "speed_rpm": 20.0,
            "current_limit_a": 0.5,
        }
        for motor_id, joint in enumerate(GLOVE_JOINTS, start=1)
    }


@pytest.mark.parametrize("hand", ["left", "right"])
def test_encos_reads_real_joints_and_sends_inverse_raw_positions(hand):
    backend = FakeEncosHand()
    mapping = _mapping()
    mapping[5] = {**mapping[5], "zero_deg": 7.0, "sign": -1.0, "scale": 2.0}
    model = Ve33GloveModel(hand)
    motors = {model.joint_motor_names[c["joint"]]:
              np.deg2rad((backend.angles[mid] - c["zero_deg"]) * c["sign"] * c["scale"])
              for mid, c in mapping.items()}
    expected, _, _ = cad_oracle(hand, motors)
    driver = EncosGloveHardwareDriver(hand=hand, hand_factory=lambda: backend, mapping=mapping)
    _connect_for_output(driver, backend)
    before = time.monotonic_ns()
    state = driver.read_state()
    assert state.pose.joints == pytest.approx({name: expected[name] for name in GLOVE_JOINTS}, abs=2e-6)
    assert state.raw_motor_positions_deg == backend.angles
    assert state.motor_positions_rad == pytest.approx(motors)
    assert len(state.tree_positions_rad) == 68
    with pytest.raises(TypeError):
        state.motor_positions_rad["illegal"] = 1.0
    assert before <= state.received_ns == state.pose.timestamp_ns <= time.monotonic_ns()
    driver.send_position(state.pose, deadline_ns=10**18)
    assert len(backend.commands) == 12
    for _, command in backend.commands:
        assert command["target_deg"] == pytest.approx(backend.angles[command["motor_id"]], abs=2e-4)
        assert command["torque_nm"] == 0.0
    driver.close()


@pytest.mark.parametrize("from_tree", [False, True])
def test_motor_trajectory_preview_saturates_then_sends_the_actual_cad_pose(from_tree):
    backend = FakeEncosHand()
    mapping = _mapping()
    mapping[4] = {**mapping[4], "max_deg": 5.0}
    mapping[5] = {**mapping[5], "zero_deg": 7.0, "sign": -1.0, "scale": 2.0}
    model = Ve33GloveModel("right")
    requested = dict(zip(model.motor_names, (0.1, 0.1, 0.2, 0.35, 0.2, 0.3,
                                            0.15, 0.25, 0.2, 0.3, 0.15, 0.2)))
    driver = EncosGloveHardwareDriver(hand_factory=lambda: backend, mapping=mapping)
    _connect_for_output(driver, backend)
    driver.prepare()
    if from_tree:
        tree = model.kinematics.evaluate_motors(requested).all_positions
        actual = driver.position_from_tree(tree, timestamp_ns=123)
    else:
        actual = driver.position_from_motors(requested, timestamp_ns=123)
    expected_motors = dict(requested)
    expected_motors[model.joint_motor_names[GLOVE_JOINTS[3]]] = np.deg2rad(5.0)
    expected, oracle_tree, _ = cad_oracle("right", expected_motors)
    assert actual.timestamp_ns == 123
    assert actual.joints == pytest.approx(
        {name: expected[name] for name in GLOVE_JOINTS}, abs=2e-6)
    assert driver.position_tree(actual) == pytest.approx(oracle_tree, abs=2e-6)
    driver.send_position(actual, deadline_ns=10**18)
    commands = {cmd["motor_id"]: cmd for kind, cmd in backend.commands if kind == "mit"}
    for mid, config in mapping.items():
        motor = expected_motors[model.joint_motor_names[config["joint"]]]
        assert commands[mid]["target_deg"] == pytest.approx(
            config["zero_deg"] + np.rad2deg(motor) / config["sign"] / config["scale"],
            abs=2e-6)
    driver.close()


def test_encos_haptic_holds_raw_exactly_and_transforms_power_conjugate_current():
    backend = FakeEncosHand()
    mapping = _mapping()
    mapping[5] = {**mapping[5], "sign": -1.0, "scale": 2.0}
    driver = EncosGloveHardwareDriver(
        hand_factory=lambda: backend, mapping=mapping, torque_constant_nm_per_a=0.2)
    _connect_for_output(driver, backend)
    currents = np.zeros(12)
    currents[4] = 0.1
    driver.send_haptic_current(tuple(currents), deadline_ns=10**18)
    assert len(backend.commands) == 12
    for _, command in backend.commands:
        assert command["target_deg"] == backend.angles[command["motor_id"]]
    command = backend.commands[4][1]
    assert command["torque_nm"] == pytest.approx(-0.04)
    raw_velocity = 0.3
    model_velocity = -2.0 * raw_velocity
    assert command["torque_nm"] * raw_velocity == pytest.approx(0.1 * 0.2 * model_velocity)
    driver.close()


def test_encos_position_and_current_share_one_frame():
    backend = FakeEncosHand()
    driver = EncosGloveHardwareDriver(
        hand_factory=lambda: backend, mapping=_mapping(), torque_constant_nm_per_a=0.2)
    _connect_for_output(driver, backend)
    target = GlovePose(dict.fromkeys(GLOVE_JOINTS, 0.0))
    driver.send_haptic_current((0.1,) * 12, deadline_ns=10**18, target_pose=target)
    assert len(backend.commands) == 12
    assert all(command["target_deg"] == pytest.approx(0.0, abs=1e-6) and
               command["torque_nm"] == pytest.approx(0.02) for _, command in backend.commands)
    driver.close()


@pytest.mark.parametrize("with_target", [False, True])
def test_encos_soft_limit_saturates_without_stopping(with_target):
    backend = FakeEncosHand()
    mapping = _mapping()
    mapping[4] = {**mapping[4], "max_deg": 0.4}
    driver = EncosGloveHardwareDriver(
        hand_factory=lambda: backend, mapping=mapping, torque_constant_nm_per_a=0.2)
    _connect_for_output(driver, backend)
    target = driver.read_pose() if with_target else None
    driver.send_haptic_current((0.0,) * 12, deadline_ns=10**18, target_pose=target)
    assert len(backend.commands) == 12
    assert backend.commands[3][1]["target_deg"] == 0.4
    assert backend.enabled == []
    driver.close()


@pytest.mark.parametrize("failure", ["missing_kt", "expired", "raw_current", "alarm", "missing_angle", "nonfinite", "offline", "unreachable"])
def test_encos_rejects_invalid_frame_before_any_axis_is_sent(failure):
    backend = FakeEncosHand()
    mapping = _mapping()
    mapping[12] = {**mapping[12], "scale": 4.0}
    driver = EncosGloveHardwareDriver(
        hand_factory=lambda: backend, mapping=mapping,
        torque_constant_nm_per_a=None if failure == "missing_kt" else 0.2)
    _connect_for_output(driver, backend)
    currents = [0.0] * 12
    deadline = 0 if failure == "expired" else 10**18
    if failure == "raw_current":
        currents[-1] = 0.1
    elif failure == "alarm":
        backend.errors[12] = 1
    elif failure == "missing_angle":
        del backend.angles[12]
    elif failure == "nonfinite":
        backend.angles[12] = float("nan")
    elif failure == "offline":
        backend.offline.add(12)
    elif failure == "unreachable":
        backend.angles[12] = 1e6
        mapping[12]["max_deg"] = 1e9
        driver.close()
        backend = FakeEncosHand()
        backend.angles[12] = 1e6
        driver = EncosGloveHardwareDriver(
            hand_factory=lambda: backend, mapping=mapping, torque_constant_nm_per_a=0.2)
        _connect_for_output(driver, backend)
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        driver.send_haptic_current(tuple(currents), deadline_ns=deadline)
    assert backend.commands == []
    driver.close()


def test_encos_uncalibrated_mapping_blocks_real_commands(tmp_path):
    config_path = tmp_path / "uncalibrated.json"
    config_path.write_text(json.dumps({"calibrated": False, "motors": _mapping()}))
    backend = FakeEncosHand()
    driver = EncosGloveHardwareDriver(
        hand_factory=lambda: backend,
        config_path=config_path,
    )
    driver.connect()
    with pytest.raises(EncosCalibrationRequired):
        driver.send_position(
            GlovePose({name: 0.0 for name in GLOVE_JOINTS}),
            deadline_ns=10**18,
        )
    driver.close()


def test_encos_construction_and_close_do_not_load_cad_or_connect_sdk(tmp_path):
    def forbidden():
        raise AssertionError("SDK must remain lazy")
    driver = EncosGloveHardwareDriver(
        hand_factory=forbidden, mapping=_mapping(), asset_root=tmp_path / "missing-cad")
    driver.close()


def test_encos_rechecks_original_deadline_after_sdk_preparation(monkeypatch):
    clock = [1]
    monkeypatch.setattr("motorhand_teleop.real_hand.motor_hand.time.monotonic_ns", lambda: clock[0])
    class SlowHand(FakeEncosHand):
        def get_all_feedbacks(self):
            clock[0] = 101
            return super().get_all_feedbacks()
    backend = SlowHand()
    driver = EncosGloveHardwareDriver(
        hand_factory=lambda: backend, mapping=_mapping(), torque_constant_nm_per_a=0.2)
    _connect_for_output(driver, backend)
    clock[0] = 1
    with pytest.raises(TimeoutError):
        driver.send_haptic_current((0.0,) * 12, deadline_ns=100)
    assert backend.commands == []
    driver.close()


def test_encos_partial_sdk_failure_disables_all_axes():
    class FailingHand(FakeEncosHand):
        def set_mit(self, **kwargs):
            super().set_mit(**kwargs)
            if kwargs["motor_id"] == 5:
                raise OSError("CAN disconnected")
    backend = FailingHand()
    driver = EncosGloveHardwareDriver(hand_factory=lambda: backend, mapping=_mapping())
    _connect_for_output(driver, backend)
    with pytest.raises(OSError, match="CAN disconnected"):
        driver.send_position(GlovePose(dict.fromkeys(GLOVE_JOINTS, 0.0)), deadline_ns=10**18)
    assert len(backend.commands) == 5
    assert backend.enabled == [(mid, False) for mid in range(1, 13)]
    driver.close()


def test_encos_public_preflight_is_read_only_and_rejects_missing_kt():
    backend = FakeEncosHand()
    driver = EncosGloveHardwareDriver(hand_factory=lambda: backend, mapping=_mapping())
    driver.connect()
    target = GlovePose(dict.fromkeys(GLOVE_JOINTS, 0.0))
    driver.validate_command(target_pose=target)
    with pytest.raises(EncosCalibrationRequired):
        driver.validate_command(target_pose=target, current_a=(0.0,) * 12)
    with pytest.raises(ValueError):
        driver.validate_command()
    assert backend.commands == []
    assert backend.enabled == []
    driver.close()


def test_encos_current_order_follows_model_binding_not_hardware_or_joint_spelling():
    backend = FakeEncosHand()
    mapping = _mapping()
    mapping[1], mapping[12] = mapping[12], mapping[1]
    mapping[12] = {**mapping[12], "sign": -1.0, "scale": 2.0}
    driver = EncosGloveHardwareDriver(
        hand_factory=lambda: backend, mapping=mapping, torque_constant_nm_per_a=0.2)
    _connect_for_output(driver, backend)
    driver.send_haptic_current((0.1,) + (0.0,) * 11, deadline_ns=10**18)
    assert backend.commands[0][1]["torque_nm"] == 0.0
    assert backend.commands[11][1]["torque_nm"] == pytest.approx(-0.04)
    driver.close()


def test_encos_current_limit_is_per_motor_raw_current():
    backend = FakeEncosHand()
    mapping = _mapping()
    mapping[12] = {**mapping[12], "current_limit_a": 0.05, "scale": 0.5}
    driver = EncosGloveHardwareDriver(
        hand_factory=lambda: backend, mapping=mapping, torque_constant_nm_per_a=0.2)
    _connect_for_output(driver, backend)
    driver.send_haptic_current((0.0,) * 11 + (0.09,), deadline_ns=10**18)
    assert backend.commands[-1][1]["torque_nm"] == pytest.approx(0.009)
    backend.commands.clear()
    with pytest.raises(ValueError, match="原始轴"):
        driver.send_haptic_current((0.0,) * 11 + (0.11,), deadline_ns=10**18)
    assert backend.commands == []
    driver.close()



def test_motor_home_uses_fixed_servo_policy_and_mit_requires_session_home():
    backend = FakeEncosHand()
    driver = EncosGloveHardwareDriver(hand_factory=lambda: backend, mapping=_mapping())
    driver.connect()
    target = GlovePose(dict.fromkeys(GLOVE_JOINTS, 0.0))
    with pytest.raises(RuntimeError, match="先home"):
        driver.send_position(target, deadline_ns=10**18)
    assert backend.commands == []
    errors = driver.home()
    assert max(errors.values()) <= 0.15
    for mode, command in backend.commands:
        assert mode == "position"
        mid = command["motor_id"]
        assert command["current_a"] == (1.5 if mid == 1 else 1.0)
        assert command["speed_rpm"] in (0.0, 5.0)
    assert {c["motor_id"]: c["target_deg"] for _, c in backend.commands} == dict.fromkeys(range(1, 13), 0.0)
    driver.disable()
    backend.commands.clear()
    driver.enable()
    assert backend.commands == []  # 同会话主导切换不回零，不先发旧目标。
    driver.send_position(target, deadline_ns=10**18)
    assert all(mode == "mit" and command["kp"] == 45.0 and command["kd"] == 0.5
               for mode, command in backend.commands)
    driver.close()


def test_read_only_motor_connection_close_does_not_take_or_disable_output():
    backend = FakeEncosHand()
    driver = EncosGloveHardwareDriver(hand_factory=lambda: backend, mapping=_mapping())
    driver.connect()
    driver.read_state()
    driver.close()
    assert backend.commands == backend.enabled == []
    assert backend.closed


@pytest.mark.parametrize("fault", ["stale", "future", "missing"])
def test_real_receive_timestamp_cannot_be_replaced_by_cache_read_time(fault):
    class FrozenFeedback(FakeEncosHand):
        frozen = None

        def get_all_feedbacks(self):
            frames = super().get_all_feedbacks()
            if self.frozen is not None:
                frames[12] = dict(self.frozen)
            return frames

    backend = FrozenFeedback()
    driver = EncosGloveHardwareDriver(hand_factory=lambda: backend, mapping=_mapping())
    _connect_for_output(driver, backend)
    old = backend.get_all_feedbacks()[12]
    if fault == "missing":
        old.pop("received_ns")
    else:
        old["received_ns"] = time.monotonic_ns() + (-1 if fault == "stale" else 1) * 1_000_000_000
    backend.frozen = old
    with pytest.raises(RuntimeError, match="时戳|过期|时钟"):
        driver.send_position(GlovePose(dict.fromkeys(GLOVE_JOINTS, 0.0)), deadline_ns=10**18)
    assert backend.commands == []
    driver.close()


def test_public_position_constraint_returns_cad_fk_after_motor_saturation():
    backend = FakeEncosHand()
    mapping = _mapping()
    mapping[4] = {**mapping[4], "max_deg": 0.4}
    driver = EncosGloveHardwareDriver(hand_factory=lambda: backend, mapping=mapping)
    driver.connect()
    requested = driver.read_pose()
    actual = driver.constrain_position(requested)
    model = Ve33GloveModel("right")
    motors = {model.joint_motor_names[c["joint"]]: np.deg2rad(
        min(max(backend.angles[mid], c["min_deg"]), c["max_deg"])) for mid, c in mapping.items()}
    expected, _, _ = cad_oracle("right", motors)
    np.testing.assert_allclose(actual.vector(), [expected[name] for name in GLOVE_JOINTS], atol=2e-6)
    tree = driver.position_tree(actual)
    motor = model.joint_motor_names["thumb_ip"]
    assert tree[motor] == pytest.approx(np.deg2rad(0.4), abs=1e-9)
    with pytest.raises(TypeError):
        tree[motor] = 0.0
    assert abs(actual.joints["thumb_ip"] - requested.joints["thumb_ip"]) > 1e-4
    assert backend.commands == []
    driver.close()


def test_outside_limit_axis_enters_home_at_five_rpm_not_clipped_hold():
    backend = FakeEncosHand()
    mapping = _mapping()
    mapping[5] = {**mapping[5], "min_deg": 0.0, "max_deg": 90.0}
    backend.angles[5] = -20.0
    driver = EncosGloveHardwareDriver(hand_factory=lambda: backend, mapping=mapping)
    driver.connect()
    driver.home()
    commands = [command for _, command in backend.commands if command["motor_id"] == 5]
    assert commands[0]["target_deg"] == 0.0
    assert commands[0]["speed_rpm"] == 5.0
    assert commands[0]["current_a"] == 1.0
    assert backend.angles[5] == 0.0
    driver.close()


def test_model_motor_limits_preserve_raw_sign_scale_and_do_not_connect():
    mapping = _mapping()
    mapping[5] = {**mapping[5], "zero_deg": 7.0, "sign": -1.0, "scale": 2.0,
                  "min_deg": -13.0, "max_deg": 17.0}
    def forbidden():
        raise AssertionError("限位元数据不能连接或使能SDK")
    driver = EncosGloveHardwareDriver(hand_factory=forbidden, mapping=mapping)
    model = Ve33GloveModel("right")
    limits = driver.model_motor_limits_rad
    assert set(limits) == set(model.motor_names)
    motor = model.joint_motor_names["index_mcp_flex"]
    np.testing.assert_allclose(limits[motor], np.deg2rad((-20.0, 40.0)), atol=1e-12)
    assert "index_mcp_flex" not in limits
    with pytest.raises(TypeError):
        limits[motor] = (0.0, 1.0)
    driver.close()


def test_force_kt_metadata_is_raw_id_bound_immutable_and_power_conjugate():
    mapping = _mapping()
    mapping[1], mapping[12] = mapping[12], mapping[1]
    mapping[12] = {**mapping[12], "sign": -1.0, "scale": 2.0, "current_limit_a": 0.1}
    constants = {str(mid): 0.1 + mid * 0.01 for mid in range(1, 13)}
    driver = EncosGloveHardwareDriver(mapping=mapping, torque_constant_nm_per_a=constants)
    model = Ve33GloveModel("right")
    constants["12"] = 999.0
    kt = driver.model_motor_torque_constants_nm_per_a
    caps = driver.model_current_limits_a
    assert set(kt) == set(caps) == set(model.motor_names)
    for mid, config in mapping.items():
        name = model.joint_motor_names[config["joint"]]
        assert kt[name] == pytest.approx(0.1 + mid * 0.01)
        assert caps[name] == pytest.approx(min(0.25, config["current_limit_a"])
                                          / abs(config["sign"] * config["scale"]))
    with pytest.raises(TypeError):
        kt[model.motor_names[0]] = 1.0
    with pytest.raises(TypeError):
        caps[model.motor_names[0]] = 1.0
    assert not driver.connected
    driver.close()


@pytest.mark.parametrize("constants", [
    {}, {mid: 0.2 for mid in range(1, 12)},
    {**dict.fromkeys(range(1, 13), 0.2), 13: 0.2},
    {**dict.fromkeys(range(1, 13), 0.2), 12: float("nan")},
    {**dict.fromkeys(range(1, 13), 0.2), 12: 0.0},
    {**dict.fromkeys(range(1, 13), 0.2), 12: -0.2},
    {**dict.fromkeys(range(1, 13), 0.2), "1": 0.3},
    0.0, float("inf"), True,
])
def test_force_rejects_incomplete_or_invalid_kt_before_sdk_creation(constants):
    def forbidden():
        raise AssertionError("标定非法时不能构造SDK连接")
    with pytest.raises(ValueError):
        EncosGloveHardwareDriver(
            mapping=_mapping(), hand_factory=forbidden, torque_constant_nm_per_a=constants,
        )


def test_force_missing_kt_rejects_before_any_output_or_home():
    backend = FakeEncosHand()
    driver = EncosGloveHardwareDriver(hand_factory=lambda: backend, mapping=_mapping())
    driver.connect()
    with pytest.raises(EncosCalibrationRequired):
        _ = driver.model_motor_torque_constants_nm_per_a
    with pytest.raises(EncosCalibrationRequired):
        driver.validate_force_current((0.0,) * 12)
    with pytest.raises(EncosCalibrationRequired):
        driver.enable_force_feedback()
    with pytest.raises(RuntimeError, match="enable_force_feedback"):
        driver.send_force_current((0.0,) * 12, deadline_ns=10**18)
    driver.disable()
    driver.close()
    assert backend.commands == backend.enabled == []


def test_force_enable_homes_once_then_zero_feedforward_and_mode_switch_restores_position():
    backend = FakeEncosHand()
    driver = EncosGloveHardwareDriver(
        hand_factory=lambda: backend, mapping=_mapping(), torque_constant_nm_per_a=0.2,
    )
    driver.connect()
    driver.validate_force_current((0.1,) * 12)
    assert backend.commands == backend.enabled == []
    driver.enable_force_feedback()
    servo = [cmd for kind, cmd in backend.commands if kind == "position"]
    assert servo
    for cmd in servo:
        assert cmd["current_a"] == (1.5 if cmd["motor_id"] == 1 else 1.0)
        assert cmd["speed_rpm"] in (0.0, 5.0)
        assert cmd["target_deg"] == (cmd["motor_id"] * 0.2 if cmd["speed_rpm"] == 0.0 else 0.0)
    frames = [cmd for kind, cmd in backend.commands if kind == "mit"]
    assert frames and all(cmd["kp"] == cmd["kd"] == cmd["torque_nm"] == 0.0 for cmd in frames)
    backend.commands.clear()
    driver.enable_force_feedback()
    assert backend.commands == []
    target = driver.read_pose()
    for operation in (
        lambda: driver.send_position(target, deadline_ns=10**18),
        lambda: driver.send_haptic_current((0.0,) * 12, deadline_ns=10**18),
        driver.enable,
        driver.home,
    ):
        with pytest.raises(RuntimeError, match="力控模式"):
            operation()
    assert backend.commands == []
    driver.disable()
    with pytest.raises(RuntimeError, match="enable_force_feedback"):
        driver.send_force_current((0.0,) * 12, deadline_ns=10**18)
    driver.enable()
    driver.send_position(target, deadline_ns=10**18)
    assert all(kind == "mit" and cmd["kp"] == 45.0 and cmd["kd"] == 0.5
               and cmd["torque_nm"] == 0.0 for kind, cmd in backend.commands)
    driver.close()


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("raw_angle", [-6.0, -5.0, 5.0, 6.0])
def test_force_projects_actual_motor_boundaries_not_position_fields(sign, raw_angle, monkeypatch):
    backend = FakeEncosHand()
    mapping = _mapping()
    mapping[5] = {**mapping[5], "sign": sign, "scale": 2.0, "min_deg": -5.0, "max_deg": 5.0}
    driver = EncosGloveHardwareDriver(
        hand_factory=lambda: backend, mapping=mapping, torque_constant_nm_per_a=0.2,
    )
    driver.connect()
    driver.enable_force_feedback()
    backend.commands.clear()
    backend.angles[5] = raw_angle
    state = driver.read_state()
    assert state.raw_motor_positions_deg[5] == raw_angle
    name = driver._glove_model().joint_motor_names[mapping[5]["joint"]]
    index = driver.motor_names.index(name)
    assert state.motor_positions_rad[name] == pytest.approx(np.deg2rad(raw_angle * sign * 2.0))
    def forbidden(*args, **kwargs):
        raise AssertionError("同次已验证CAD树不应重复求解或走位置IK")
    monkeypatch.setattr(driver, "_position_targets", forbidden)
    monkeypatch.setattr(driver._glove_model().kinematics, "evaluate_motors", forbidden)
    outward = np.zeros(12)
    outward[index] = np.sign(raw_angle) * sign * 0.05
    actual = driver.send_force_current(tuple(outward), deadline_ns=10**18)
    assert actual == (0.0,) * 12
    command = backend.commands[4][1]
    assert command["target_deg"] == np.clip(raw_angle, -5.0, 5.0)
    assert command["kp"] == command["kd"] == command["torque_nm"] == 0.0
    backend.commands.clear()
    inward = tuple(-outward)
    actual = driver.send_force_current(inward, deadline_ns=10**18)
    assert actual == inward
    command = backend.commands[4][1]
    assert command["kp"] == command["kd"] == 0.0
    assert command["torque_nm"] * np.sign(raw_angle) < 0.0
    assert command["torque_nm"] == pytest.approx(inward[index] * sign * 2.0 * 0.2)
    assert backend.angles[5] == raw_angle
    assert backend.enabled == []
    driver.close()


def test_force_resistance_and_power_match_independent_real_cad_at_sdk_boundary():
    backend = FakeEncosHand()
    mapping = _mapping()
    mapping[5] = {**mapping[5], "sign": -1.0, "scale": 2.0}
    mapping[1], mapping[12] = mapping[12], mapping[1]
    constants = {mid: 0.15 + mid * 0.005 for mid in range(1, 13)}
    driver = EncosGloveHardwareDriver(
        hand_factory=lambda: backend, mapping=mapping, torque_constant_nm_per_a=constants,
    )
    _connect_for_output(driver, backend)
    driver.enable_force_feedback()
    state = driver.read_state()
    model = Ve33GloveModel("right")
    motors = dict(state.motor_positions_rad)
    dq_dm = np.empty((12, 12))
    step = 2e-6
    for col, name in enumerate(model.motor_names):
        plus, minus = dict(motors), dict(motors)
        plus[name] += step
        minus[name] -= step
        q_plus, _, _ = cad_oracle("right", plus)
        q_minus, _, _ = cad_oracle("right", minus)
        dq_dm[:, col] = [(q_plus[joint] - q_minus[joint]) / (2.0 * step)
                         for joint in GLOVE_JOINTS]
    dm_dt = np.linspace(-0.2, 0.3, 12)
    dq_dt = dq_dm @ dm_dt
    tau_q = -0.001 * dq_dt
    tau_m = dq_dm.T @ tau_q
    kt = driver.model_motor_torque_constants_nm_per_a
    currents = tau_m / np.array([kt[name] for name in model.motor_names])
    backend.commands.clear()
    actual = driver.send_force_current(tuple(currents), deadline_ns=10**18)
    np.testing.assert_allclose(actual, currents, atol=1e-15)
    commands = {cmd["motor_id"]: cmd for _, cmd in backend.commands}
    raw_power = 0.0
    for mid, config in mapping.items():
        name = model.joint_motor_names[config["joint"]]
        index = model.motor_names.index(name)
        cmd = commands[mid]
        assert cmd["kp"] == cmd["kd"] == 0.0
        assert cmd["target_deg"] == backend.angles[mid]
        assert cmd["torque_nm"] == pytest.approx(
            currents[index] * config["sign"] * config["scale"] * constants[mid],
        )
        raw_power += cmd["torque_nm"] * dm_dt[index] / config["sign"] / config["scale"]
    assert raw_power < 0.0
    assert raw_power == pytest.approx(float(tau_q @ dq_dt), abs=1e-12)
    assert raw_power == pytest.approx(float(tau_m @ dm_dt), abs=1e-12)
    driver.close()


@pytest.mark.parametrize("fault", ["deadline", "current", "stale", "offline", "alarm", "sdk"])
def test_force_fault_stops_owned_output_and_clears_mode_before_position_reentry(fault):
    class FaultHand(FakeEncosHand):
        stale = False
        fail = False
        def get_all_feedbacks(self):
            result = super().get_all_feedbacks()
            if self.stale:
                result[12]["received_ns"] -= 1_000_000_000
            return result
        def set_mit(self, **kwargs):
            if self.fail and kwargs["motor_id"] == 5:
                raise OSError("CAN force write failed")
            return super().set_mit(**kwargs)
    backend = FaultHand()
    driver = EncosGloveHardwareDriver(
        hand_factory=lambda: backend, mapping=_mapping(), torque_constant_nm_per_a=0.2,
    )
    driver.connect()
    driver.enable_force_feedback()
    backend.commands.clear()
    currents = (0.0,) * 12
    deadline = 10**18
    if fault == "deadline":
        deadline = 0
    elif fault == "current":
        currents = (0.0,) * 11 + (0.3,)
    elif fault == "stale":
        backend.stale = True
    elif fault == "offline":
        backend.offline.add(12)
    elif fault == "alarm":
        backend.errors[12] = 1
    else:
        backend.fail = True
    with pytest.raises((RuntimeError, ValueError, TimeoutError, OSError)):
        driver.send_force_current(currents, deadline_ns=deadline)
    if fault != "sdk":
        assert backend.commands == []
    assert backend.enabled
    assert all(not enabled for _, enabled in backend.enabled)
    with pytest.raises(RuntimeError, match="enable_force_feedback"):
        driver.send_force_current((0.0,) * 12, deadline_ns=10**18)
    backend.stale = backend.fail = False
    backend.offline.clear()
    backend.errors.clear()
    backend.commands.clear()
    driver.enable()
    driver.send_position(GlovePose(dict.fromkeys(GLOVE_JOINTS, 0.0)), deadline_ns=10**18)
    assert all(kind == "mit" and cmd["kp"] == 45.0 and cmd["kd"] == 0.5
               for kind, cmd in backend.commands)
    driver.close()


def test_force_rechecks_original_deadline_after_feedback_read(monkeypatch):
    class SlowHand(FakeEncosHand):
        advance = None
        def get_all_feedbacks(self):
            if self.advance is not None:
                self.advance()
            return super().get_all_feedbacks()
    backend = SlowHand()
    driver = EncosGloveHardwareDriver(
        hand_factory=lambda: backend, mapping=_mapping(), torque_constant_nm_per_a=0.2,
    )
    driver.connect()
    driver.enable_force_feedback()
    backend.commands.clear()
    clock = [1]
    monkeypatch.setattr("motorhand_teleop.real_hand.motor_hand.time.monotonic_ns",
                        lambda: clock[0])
    def advance():
        clock[0] = 101
    backend.advance = advance
    with pytest.raises(TimeoutError):
        driver.send_force_current((0.1,) * 12, deadline_ns=100)
    assert backend.commands == []
    assert backend.enabled and all(not enabled for _, enabled in backend.enabled)
    with pytest.raises(RuntimeError, match="enable_force_feedback"):
        driver.send_force_current((0.0,) * 12, deadline_ns=1000)
    driver.close()


@pytest.mark.parametrize("calibrated", [False, True])
def test_force_reads_json_raw_id_kt_but_uncalibrated_mapping_cannot_enable(tmp_path, calibrated):
    config_path = tmp_path / "measured-glove.json"
    config_path.write_text(json.dumps({
        "calibrated": calibrated, "mapping_source": "离线回归标定",
        "motors": _mapping(),
        "haptic": {"torque_constant_nm_per_a": {str(mid): 0.1 + mid * 0.01
                                                for mid in range(1, 13)}},
    }), encoding="utf-8")
    backend = FakeEncosHand()
    driver = EncosGloveHardwareDriver(
        hand_factory=lambda: backend, config_path=config_path, allow_uncalibrated=True,
    )
    driver.connect()
    if calibrated:
        driver.enable_force_feedback()
        driver.send_force_current((0.1,) * 12, deadline_ns=10**18)
        for _, cmd in backend.commands[-12:]:
            assert cmd["kp"] == cmd["kd"] == 0.0
            assert cmd["torque_nm"] == pytest.approx(0.1 * (0.1 + cmd["motor_id"] * 0.01))
    else:
        with pytest.raises(EncosCalibrationRequired, match="真实标定"):
            driver.validate_force_current((0.0,) * 12)
        with pytest.raises(EncosCalibrationRequired, match="真实标定"):
            driver.enable_force_feedback()
        assert backend.commands == backend.enabled == []
    driver.close()


def _decoded_nominal_torque(command):
    """Decode the real CAN word independently, without SDK encoding constants."""
    from encos_hand.protocol import MotorProtocol

    payload = MotorProtocol.mit_command(
        np.deg2rad(command["target_deg"]), command["vel_rad_s"],
        command["kp"], command["kd"], command["torque_nm"],
    )
    word = int.from_bytes(payload, "big")
    assert (word >> 49) & 0xFFF == 0
    assert (word >> 40) & 0x1FF == 0
    return -10.0 + (word & 0xFFF) * 20.0 / 4095.0


def test_nominal_torque_metadata_is_explicit_raw_bound_and_does_not_connect():
    mapping = _mapping()
    mapping[1], mapping[12] = mapping[12], mapping[1]
    mapping[12] = {**mapping[12], "sign": -1.0, "scale": 2.0}
    raw_caps = {str(mid): 0.01 + mid * 0.001 for mid in range(1, 13)}
    def forbidden():
        raise AssertionError("metadata cannot construct SDK or connect")
    driver = EncosGloveHardwareDriver(
        hand_factory=forbidden, mapping=mapping, force_torque_limits_nm=raw_caps,
    )
    caps = driver.model_force_torque_limits_nm
    raw_caps["12"] = 999.0
    assert set(caps) == set(driver.motor_names)
    for mid, config in mapping.items():
        name = driver._glove_model().joint_motor_names[config["joint"]]
        assert caps[name] == pytest.approx(
            (0.01 + mid * 0.001 - 10.0 / 4095.0) / abs(config["sign"] * config["scale"]),
        )
    with pytest.raises(TypeError):
        caps[driver.motor_names[0]] = 1.0
    with pytest.raises(EncosCalibrationRequired):
        _ = driver.model_motor_torque_constants_nm_per_a
    assert not driver.connected
    driver.close()
    current_driver = EncosGloveHardwareDriver(mapping=_mapping())
    with pytest.raises(RuntimeError, match="force_torque_limits_nm"):
        _ = current_driver.model_force_torque_limits_nm
    current_driver.close()


@pytest.mark.parametrize("limits", [
    {}, {mid: 0.01 for mid in range(1, 12)}, 0.01,
    {**dict.fromkeys(range(1, 13), 0.01), "1": 0.01},
    {**dict.fromkeys(range(1, 13), 0.01), 13: 0.01},
    *[{**dict.fromkeys(range(1, 13), 0.01), 12: value}
      for value in [0.0, -0.01, True, float("nan"), float("inf"), 0.001, 10.1, 1j]],
])
def test_nominal_torque_rejects_invalid_limits_before_sdk_creation(limits):
    def forbidden():
        raise AssertionError("invalid configuration must not construct SDK")
    with pytest.raises(ValueError):
        EncosGloveHardwareDriver(
            mapping=_mapping(), hand_factory=forbidden, force_torque_limits_nm=limits,
        )


def test_nominal_torque_enable_preserves_home_and_independent_position_gains():
    backend = FakeEncosHand()
    driver = EncosGloveHardwareDriver(
        mapping=_mapping(), hand_factory=lambda: backend,
        force_torque_limits_nm=dict.fromkeys(range(1, 13), 0.01),
        max_haptic_current_a=0.0,
    )
    driver.connect()
    driver.validate_force_torque((0.0,) * 12)
    assert backend.commands == backend.enabled == []
    with pytest.raises(RuntimeError, match="名义Nm"):
        driver.validate_force_current((0.0,) * 12)
    with pytest.raises(RuntimeError, match="enable_force_feedback"):
        driver.send_force_torque((0.0,) * 12, deadline_ns=10**18)
    driver.enable_force_feedback()
    servo = [cmd for kind, cmd in backend.commands if kind == "position"]
    assert servo and all(cmd["speed_rpm"] in (0.0, 5.0) for cmd in servo)
    assert all(cmd["current_a"] == (1.5 if cmd["motor_id"] == 1 else 1.0) for cmd in servo)
    frames = [cmd for kind, cmd in backend.commands if kind == "mit"]
    assert len(frames) == 12
    assert all(cmd["kp"] == cmd["kd"] == cmd["torque_nm"] == 0.0 for cmd in frames)
    backend.commands.clear()
    driver.enable_force_feedback()
    assert backend.commands == []
    driver.send_force_torque((0.005,) * 12, deadline_ns=10**18)
    assert all(cmd["kp"] == cmd["kd"] == 0.0 for _, cmd in backend.commands)
    # A nominal Nm cap is not a promise to implement current limiting without Kt.
    assert all(cmd["torque_nm"] == 0.005 for _, cmd in backend.commands)
    backend.commands.clear()
    with pytest.raises(RuntimeError, match="名义Nm"):
        driver.send_force_current((0.0,) * 12, deadline_ns=10**18)
    assert backend.commands == []
    assert not driver._force_enabled
    driver.enable()
    driver.send_position(driver.read_pose(), deadline_ns=10**18)
    assert all(cmd["kp"] == 45.0 and cmd["kd"] == 0.5 and cmd["torque_nm"] == 0.0
               for _, cmd in backend.commands)
    driver.close()


@pytest.mark.parametrize("polarity", [-1.0, 1.0])
@pytest.mark.parametrize("full_scale", [False, True])
def test_nominal_torque_can_quantization_stays_within_raw_software_limits(polarity, full_scale):
    backend = FakeEncosHand()
    mapping = _mapping()
    mapping[1], mapping[12] = mapping[12], mapping[1]
    for mid in mapping:
        mapping[mid] = {**mapping[mid], "sign": -1.0 if mid % 2 else 1.0,
                        "scale": -2.0 if mid % 3 else 0.5}
    raw_caps = (dict.fromkeys(range(1, 13), 10.0) if full_scale else
                {mid: 0.01 + mid * 0.0001 for mid in range(1, 13)})
    driver = EncosGloveHardwareDriver(
        mapping=mapping, hand_factory=lambda: backend, force_torque_limits_nm=raw_caps,
    )
    driver.connect()
    driver.enable_force_feedback()
    backend.commands.clear()
    values = tuple(polarity * driver.model_force_torque_limits_nm[name] for name in driver.motor_names)
    driver.validate_force_torque(values)
    assert backend.commands == []
    actual = driver.send_force_torque(values, deadline_ns=10**18)
    np.testing.assert_allclose(actual, values, rtol=1e-15, atol=1e-18)
    for _, cmd in backend.commands:
        mid = cmd["motor_id"]
        config = mapping[mid]
        name = driver._glove_model().joint_motor_names[config["joint"]]
        index = driver.motor_names.index(name)
        assert cmd["torque_nm"] == pytest.approx(values[index] * config["sign"] * config["scale"])
        decoded = _decoded_nominal_torque(cmd)
        assert abs(decoded) <= raw_caps[mid]
        assert abs(decoded - cmd["torque_nm"]) <= 10.0 / 4095.0 + 1e-12
        assert cmd["deadline_ns"] == 10**18
    driver.close()


@pytest.mark.parametrize(("sign", "scale"), [(-1.0, 2.0), (1.0, -2.0), (1.0, 2.0), (-1.0, -2.0)])
def test_nominal_torque_preserves_power_duality_once_at_raw_boundary(sign, scale):
    backend = FakeEncosHand()
    mapping = _mapping()
    mapping[5] = {**mapping[5], "sign": sign, "scale": scale}
    driver = EncosGloveHardwareDriver(
        mapping=mapping, hand_factory=lambda: backend,
        force_torque_limits_nm=dict.fromkeys(range(1, 13), 0.1),
    )
    driver.connect()
    driver.enable_force_feedback()
    backend.commands.clear()
    values = np.linspace(-0.02, 0.02, 12)
    velocity_model = np.linspace(0.2, -0.3, 12)
    actual = driver.send_force_torque(tuple(values), deadline_ns=10**18)
    raw_power = 0.0
    for _, cmd in backend.commands:
        config = mapping[cmd["motor_id"]]
        name = driver._glove_model().joint_motor_names[config["joint"]]
        index = driver.motor_names.index(name)
        axis = config["sign"] * config["scale"]
        assert cmd["torque_nm"] == pytest.approx(values[index] * axis)
        raw_power += cmd["torque_nm"] * velocity_model[index] / axis
    np.testing.assert_allclose(actual, values, rtol=1e-15, atol=1e-18)
    assert raw_power == pytest.approx(values @ velocity_model, abs=1e-15)
    driver.close()


@pytest.mark.parametrize(("sign", "scale"), [(-1.0, 2.0), (1.0, -2.0), (1.0, 2.0), (-1.0, -2.0)])
@pytest.mark.parametrize("raw_angle", [-6.0, -5.0, 5.0, 6.0])
def test_nominal_torque_boundary_zero_and_tiny_inward_encode_inward(sign, scale, raw_angle):
    backend = FakeEncosHand()
    mapping = _mapping()
    mapping[5] = {**mapping[5], "sign": sign, "scale": scale,
                  "min_deg": -5.0, "max_deg": 5.0}
    driver = EncosGloveHardwareDriver(
        mapping=mapping, hand_factory=lambda: backend,
        force_torque_limits_nm=dict.fromkeys(range(1, 13), 0.01),
    )
    driver.connect()
    driver.enable_force_feedback()
    backend.angles[5] = raw_angle
    name = driver._glove_model().joint_motor_names[mapping[5]["joint"]]
    index = driver.motor_names.index(name)
    axis = sign * scale
    outward = np.sign(raw_angle)
    for raw_torque in (0.005 * outward, 0.0, -1e-18 * outward, -0.005 * outward):
        backend.commands.clear()
        target = np.zeros(12)
        target[index] = raw_torque / axis
        actual = driver.send_force_torque(tuple(target), deadline_ns=10**18)
        command = backend.commands[4][1]
        assert command["motor_id"] == 5
        assert command["target_deg"] == np.clip(raw_angle, -5.0, 5.0)
        assert command["torque_nm"] * outward < 0.0
        assert _decoded_nominal_torque(command) * outward < 0.0
        assert abs(_decoded_nominal_torque(command)) <= 0.01
        assert actual[index] == command["torque_nm"] / axis
        if abs(raw_torque) < 0.005 or raw_torque * outward > 0.0:
            assert abs(command["torque_nm"]) == pytest.approx(10.0 / 4095.0)
        else:
            assert command["torque_nm"] == raw_torque
        assert backend.angles[5] == raw_angle
    driver.close()


def test_nominal_torque_slew_acknowledges_continuous_targets_not_decoded_grid():
    backend = FakeEncosHand()
    driver = EncosGloveHardwareDriver(
        mapping=_mapping(), hand_factory=lambda: backend,
        force_torque_limits_nm=dict.fromkeys(range(1, 13), 0.02),
    )
    driver.connect()
    driver.enable_force_feedback()
    previous = np.zeros(12)
    decoded = []
    for _ in range(130):
        backend.commands.clear()
        target = previous + 0.0001
        actual = driver.send_force_torque(tuple(target), deadline_ns=10**18)
        np.testing.assert_allclose(actual, target, atol=1e-18, rtol=1e-15)
        previous = np.asarray(actual)
        decoded.append(_decoded_nominal_torque(backend.commands[0][1]))
    np.testing.assert_allclose(previous, 0.013, atol=1e-15)
    assert len(set(decoded)) >= 3
    assert decoded[0] != pytest.approx(0.0001)
    driver.close()


@pytest.mark.parametrize("fault", ["limit", "nan", "shape", "stale", "missing_stamp", "offline", "alarm", "cad", "deadline", "sdk"])
def test_nominal_torque_fault_rejects_whole_frame_and_releases_owned_output(fault, monkeypatch):
    class FaultHand(FakeEncosHand):
        stale = False
        missing_stamp = False
        fail = False
        def get_all_feedbacks(self):
            result = super().get_all_feedbacks()
            if self.stale:
                result[12]["received_ns"] -= 1_000_000_000
            if self.missing_stamp:
                result[12].pop("received_ns")
            return result
        def set_mit(self, **kwargs):
            if self.fail and kwargs["motor_id"] == 5:
                raise OSError("CAN nominal torque write failed")
            return super().set_mit(**kwargs)
    backend = FaultHand()
    driver = EncosGloveHardwareDriver(
        mapping=_mapping(), hand_factory=lambda: backend,
        force_torque_limits_nm=dict.fromkeys(range(1, 13), 0.01),
    )
    driver.connect()
    driver.enable_force_feedback()
    backend.commands.clear()
    values = (0.0,) * 12
    deadline = 10**18
    if fault == "limit":
        values = (0.0,) * 11 + (0.01,)
    elif fault == "nan":
        values = (0.0,) * 11 + (float("nan"),)
    elif fault == "shape":
        values = (0.0,) * 11
    elif fault == "stale":
        backend.stale = True
    elif fault == "missing_stamp":
        backend.missing_stamp = True
    elif fault == "offline":
        backend.offline.add(12)
    elif fault == "alarm":
        backend.errors[12] = 1
    elif fault == "cad":
        def unreachable(*args, **kwargs):
            raise ValueError("unreachable real CAD")
        monkeypatch.setattr(driver, "_force_targets", unreachable)
    elif fault == "deadline":
        deadline = 0
    else:
        backend.fail = True
    if fault not in ("deadline", "sdk"):
        with pytest.raises((ValueError, RuntimeError)):
            driver.validate_force_torque(values)
        assert backend.commands == []
        assert driver._force_enabled
        assert backend.enabled == []
    with pytest.raises((ValueError, RuntimeError, TimeoutError, OSError)):
        driver.send_force_torque(values, deadline_ns=deadline)
    if fault != "sdk":
        assert backend.commands == []
    assert backend.enabled and all(not enabled for _, enabled in backend.enabled)
    assert not driver._force_enabled
    with pytest.raises(RuntimeError, match="enable_force_feedback"):
        driver.send_force_torque((0.0,) * 12, deadline_ns=10**18)
    driver.close()


def test_nominal_torque_rechecks_original_deadline_after_feedback(monkeypatch):
    class SlowHand(FakeEncosHand):
        advance = None
        def get_all_feedbacks(self):
            if self.advance is not None:
                self.advance()
            return super().get_all_feedbacks()
    backend = SlowHand()
    driver = EncosGloveHardwareDriver(
        mapping=_mapping(), hand_factory=lambda: backend,
        force_torque_limits_nm=dict.fromkeys(range(1, 13), 0.01),
    )
    driver.connect()
    driver.enable_force_feedback()
    backend.commands.clear()
    clock = [1]
    monkeypatch.setattr("motorhand_teleop.real_hand.motor_hand.time.monotonic_ns", lambda: clock[0])
    def advance():
        clock[0] = 101
    backend.advance = advance
    with pytest.raises(TimeoutError):
        driver.send_force_torque((0.005,) * 12, deadline_ns=100)
    assert backend.commands == []
    assert backend.enabled and all(not enabled for _, enabled in backend.enabled)
    driver.close()


def test_nominal_torque_does_not_relax_mapping_calibration(tmp_path):
    path = tmp_path / "uncalibrated.json"
    path.write_text(json.dumps({"calibrated": False, "motors": _mapping()}), encoding="utf-8")
    backend = FakeEncosHand()
    driver = EncosGloveHardwareDriver(
        config_path=path, allow_uncalibrated=True, hand_factory=lambda: backend,
        force_torque_limits_nm=dict.fromkeys(range(1, 13), 0.01),
    )
    driver.connect()
    for operation in (lambda: driver.validate_force_torque((0.0,) * 12), driver.enable_force_feedback):
        with pytest.raises(EncosCalibrationRequired, match="真实标定"):
            operation()
    assert backend.commands == backend.enabled == []
    driver.close()
