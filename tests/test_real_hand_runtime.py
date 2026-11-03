"""真实canonical任务模型与有滞后数值执行器的配对消费者回归。"""

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import json

import numpy as np
import pytest

from motorhand_teleop.domain import GLOVE_JOINTS, GlovePose, WujiState
from motorhand_teleop.real_hand.runtime import (
    FollowMode,
    PairRuntime,
    create_retargeter,
    RetargetPipelineFactory,
    main,
)


ROOT = Path(__file__).parents[1]


@dataclass(frozen=True)
class DeviceIdentity:
    serial_number: str
    firmware_version: str


class Clock:
    def __init__(self):
        self.now = 1_000_000_000

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += int(seconds * 1e9)


class HandPlant:
    """带一阶机械滞后的20轴位置数值模型；写入目标不会回显为测量。"""

    def __init__(self, clock, positions, limits):
        self.clock = clock
        self.positions = positions.copy()
        self.reference = positions.copy()
        self.limits = limits
        self.commands = []
        self.enabled = 0
        self.disabled = 0
        self.closed = False
        self.stamp_offset = 0
        self.online = 20

    def connect(self, *, serial_number):
        assert serial_number == "explicit-test-SN"
        return DeviceIdentity(serial_number, "approved-test")

    def read_state(self):
        return WujiState(
            self.positions.copy(), None, None,
            self.clock() + self.stamp_offset, self.online,
        )

    def enable(self):
        self.enabled += 1

    def disable(self):
        self.disabled += 1

    def validate(self, target):
        assert target.positions_rad.shape == (20,)
        assert np.isfinite(target.positions_rad).all()
        assert (target.positions_rad >= self.limits[:, 0] - 1e-7).all()
        assert (target.positions_rad <= self.limits[:, 1] + 1e-7).all()
        assert np.count_nonzero(target.effort_a) == 0

    def send_target(self, target, *, deadline_ns):
        self.validate(target)
        assert self.clock() < deadline_ns
        self.commands.append((target, deadline_ns))
        self.reference = target.positions_rad.copy()

    def advance_mechanics(self):
        self.positions += 0.3 * (self.reference - self.positions)

    def close(self):
        self.closed = True


class MotorPlant:
    """真实CAD q↔m求解配合一阶电机滞后；绝不把目标q当成反馈。"""

    mapping_calibrated = True
    mapping_source = "数值模型已知标定"

    def __init__(self, clock, model, joints):
        self.clock = clock
        self.model = model
        self.pose, motors = model.evaluate(joints)
        self.motors = dict(motors)
        self.reference = dict(motors)
        self.commands = []
        self.homes = 0
        self.enabled = 0
        self.disabled = 0
        self.stamp_offset = 0
        self.pose_stamp_offset = 0
        self.online_ids = tuple(range(1, 13))
        self.closed = False
        self.fail_validation = False
        self.motor_bounds = {}
        self.prepares = 0
        self.state_reads = 0
        self._prepared_q = None
        self._prepared_motors = None
        self._prepared_tree = None
        self._feedback_seed = None
        self._target_seed = None

    @property
    def model_motor_limits_rad(self):
        return MappingProxyType(self.motor_bounds)

    @property
    def motor_names(self):
        return self.model.motor_names

    def connect(self):
        return {"model": "CAD数值MotorHand", "mapping_calibrated": True}

    def prepare(self):
        self.prepares += 1
        # 实际CAD FK热身，结果丢弃；后续read_state必须另采新帧。
        self.read_state()

    def read_state(self):
        self.state_reads += 1
        # 与真实驱动一致：新反馈开始新周期，不复用上一帧的raw预检结果。
        self._prepared_q = None
        self._prepared_motors = None
        self._prepared_tree = None
        stamp = self.clock() + self.stamp_offset
        solution = self.model.kinematics.evaluate_motors(self.motors, seed=self._feedback_seed)
        pose = self.model.to_glove_pose(solution, timestamp_ns=stamp + self.pose_stamp_offset)
        self._feedback_seed = solution
        self.pose = pose
        return SimpleNamespace(
            pose=pose, received_ns=stamp, online_ids=self.online_ids,
            motor_positions_rad=dict(self.motors),
            raw_motor_positions_deg={index: float(np.rad2deg(value)) for index, value in enumerate(self.motors.values(), 1)},
        )

    def home(self):
        self.homes += 1
        self.enabled += 1
        self.pose, motors = self.model.evaluate(dict.fromkeys(GLOVE_JOINTS, 0.0))
        self.motors = dict(motors)
        self.reference = dict(motors)
        self._feedback_seed = None
        self._target_seed = None

    def enable(self):
        self.enabled += 1
        self._target_seed = None

    def disable(self):
        self.disabled += 1

    def _coupled_names(self, *, motor_space):
        field = "motor" if motor_space else "name"
        return tuple(
            item[field] for item, coupled in zip(
                self.model.kinematics.joint_definitions, self.model.kinematics._continuation_inputs,
            ) if coupled
        )

    def _nearest_seed(self, positions, *, motor_space, known_pose=None):
        candidates = [
            seed for seed in (known_pose, self._feedback_seed, self._target_seed)
            if seed is not None
        ]
        if not candidates:
            return None
        names = self._coupled_names(motor_space=motor_space)

        def distance(seed):
            values = seed.motor_positions if motor_space else seed.joint_positions
            return max((abs(positions[name] - values[name]) for name in names), default=0.0)

        return min(candidates, key=distance)

    def constrain_position(self, pose):
        internal = {f"{name}_joint": value for name, value in pose.joints.items()}
        seed = self._nearest_seed(internal, motor_space=False)
        if seed is None:
            solution = self.model.evaluate_raw(dict(pose.joints))
        else:
            nearby = all(
                abs(internal[name] - seed.joint_positions[name]) <= 0.04 + 1e-12
                for name in self._coupled_names(motor_space=False)
            )
            solution = self.model.kinematics.evaluate(internal, seed=seed, local=nearby)
        return self._position_from_motor_values(solution.motor_positions, pose.timestamp_ns, solution)

    def _position_from_motor_values(self, motors, timestamp_ns, known_pose=None, *, update_target=True):
        if set(motors) != set(self.motor_names) or not np.isfinite(list(motors.values())).all():
            raise ValueError("模型电机位置必须完整覆盖12轴且有限")
        constrained = {
            name: float(np.clip(value, *self.motor_bounds[name]))
            if name in self.motor_bounds else value
            for name, value in motors.items()
        }
        if known_pose is None or any(constrained[name] != motors[name] for name in motors):
            solution = self.model.kinematics.evaluate_motors(
                constrained, seed=self._nearest_seed(
                    constrained, motor_space=True, known_pose=known_pose,
                ),
            )
        else:
            solution = known_pose
        if update_target:
            self._target_seed = solution
        actual = self.model.to_glove_pose(solution, timestamp_ns=timestamp_ns)
        self._prepared_q = actual.vector()
        self._prepared_motors = dict(constrained)
        self._prepared_tree = dict(solution.all_positions)
        return actual

    def position_from_tree(self, tree, *, timestamp_ns):
        solution = self.model.kinematics.evaluate_tree(tree)
        return self._position_from_motor_values(
            solution.motor_positions, timestamp_ns, solution, update_target=False,
        )

    def position_from_motors(self, positions, *, timestamp_ns):
        return self._position_from_motor_values(positions, timestamp_ns)

    def position_tree(self, pose):
        if self._prepared_q is None or not np.array_equal(self._prepared_q, pose.vector()):
            raise ValueError("没有与当前目标匹配的已验证CAD树")
        if self._prepared_tree is None:
            raise ValueError("缺少本周期实际raw限位后的CAD树")
        return MappingProxyType(self._prepared_tree)

    def validate_command(self, *, target_pose):
        if self.fail_validation:
            raise ValueError("数值机构完整预检失败")
        if self._prepared_q is None or not np.array_equal(self._prepared_q, target_pose.vector()):
            # 缓存只来自本周期真实CAD IK/FK，不模拟目标echo。
            _, motors = self.model.evaluate(dict(target_pose.joints))
            _, solution = self.model.evaluate_motor_state(motors)
            self._prepared_q = target_pose.vector()
            self._prepared_motors = dict(motors)
            self._prepared_tree = dict(solution.all_positions)

    def send_position(self, pose, *, deadline_ns):
        assert self.clock() < deadline_ns
        assert self.homes == 1
        self.validate_command(target_pose=pose)
        self.reference = dict(self._prepared_motors)
        self.commands.append((pose, deadline_ns))

    def advance_mechanics(self):
        for name, reference in self.reference.items():
            self.motors[name] += 0.3 * (reference - self.motors[name])

    def close(self):
        self.closed = True


@pytest.fixture
def bridge():
    return create_retargeter(project_root=ROOT)


def _joints(amount):
    joints = dict.fromkeys(GLOVE_JOINTS, 0.0)
    joints["index_pip"] = amount
    joints["middle_pip"] = amount * 0.5
    return joints


def _pair(bridge, *, mode=FollowMode.GLOVE_TO_HAND2, execute=True):
    clock = Clock()
    positions = bridge.glove_to_hand2(_joints(0.1))[0].positions_rad
    retargeter = bridge.forward.retargeter
    physical_limits = retargeter._core.optimizer.robot.joint_limits[retargeter._output_indices]
    hand = HandPlant(clock, positions, physical_limits)
    glove = MotorPlant(clock, bridge.forward.landmarks.model, _joints(0.03))
    runtime = PairRuntime(
        hand, glove, bridge, mode=mode, execute=execute,
        clock_ns=clock, transition_s=0.2,
    )
    runtime.connect(serial_number="explicit-test-SN")
    return runtime, hand, glove, clock


def test_connect_inspect_and_close_never_acquire_output(bridge):
    runtime, hand, glove, _ = _pair(bridge)
    assert glove.prepares == 1 and glove.state_reads == 1
    report = runtime.inspect()
    assert json.loads(json.dumps(report))["身份"]["wuji_hand2"]["serial_number"] == "explicit-test-SN"
    assert len(report["Wuji20语义关节rad"]) == 20
    assert len(report["MotorHand12真实指节rad"]) == 12
    assert report["MotorHand映射已标定"]
    runtime.close()
    assert hand.enabled == hand.disabled == glove.homes == glove.enabled == glove.disabled == 0
    assert hand.commands == glove.commands == []


@pytest.mark.parametrize("mode", list(FollowMode))
def test_preview_computes_real_task_without_enabling_or_writing(bridge, mode):
    runtime, hand, glove, clock = _pair(bridge, mode=mode, execute=False)
    runtime.start()
    first = runtime.step()
    clock.advance(0.2)
    tick = runtime.step()
    assert not first.executed and not tick.executed
    assert tick.output_task.points_m.shape == (21, 3)
    assert np.isfinite(tick.output_task.points_m).all()
    runtime.close()
    assert hand.commands == glove.commands == []
    assert hand.enabled == hand.disabled == glove.homes == glove.enabled == glove.disabled == 0


def test_glove_to_hand_only_writes_wuji_and_mechanics_approach_target(bridge):
    runtime, hand, glove, clock = _pair(bridge)
    runtime.start()
    anchor = hand.read_state().positions_rad.copy()
    first = runtime.step()
    np.testing.assert_allclose(first.output_positions_rad, anchor, atol=1e-7)
    clock.advance(0.2)
    tick = runtime.step()
    np.testing.assert_allclose(tick.output_positions_rad, tick.candidate_positions_rad, atol=1e-12)
    before = np.linalg.norm(hand.positions - tick.output_positions_rad)
    assert before > 1e-6
    for _ in range(8):
        hand.advance_mechanics()
    after = np.linalg.norm(hand.positions - tick.output_positions_rad)
    assert after < before * 0.1
    np.testing.assert_allclose(
        tick.output_task.points_m,
        bridge.hand2_model.state(tick.output_positions_rad).points_m,
    )
    runtime.close()
    assert len(hand.commands) == 2
    assert glove.commands == []
    assert glove.homes == glove.enabled == glove.disabled == 0
    assert hand.disabled == 1


def test_hand_to_glove_uses_actual_wuji_and_never_holds_it(bridge):
    runtime, hand, glove, clock = _pair(bridge, mode=FollowMode.HAND2_TO_GLOVE)
    runtime.start()
    anchor = glove.read_state().pose.vector()
    first = runtime.step()
    np.testing.assert_allclose(first.output_positions_rad, anchor, atol=1e-7)
    clock.advance(0.2)
    tick = runtime.step()
    before = np.linalg.norm(glove.read_state().pose.vector() - tick.output_positions_rad)
    assert before > 1e-6
    for _ in range(8):
        glove.advance_mechanics()
    after = np.linalg.norm(glove.read_state().pose.vector() - tick.output_positions_rad)
    assert after < before * 0.2
    runtime.close()
    assert hand.commands == []
    assert hand.enabled == hand.disabled == 0
    assert glove.homes == 1 and glove.disabled == 1


def test_switch_anchors_to_measured_new_follower_and_homes_only_once(bridge):
    runtime, hand, glove, clock = _pair(bridge)
    runtime.start()
    runtime.step()
    runtime.request_mode(FollowMode.HAND2_TO_GLOVE)
    assert hand.disabled == 1 and glove.homes == 1
    anchor = glove.read_state().pose.vector().copy()
    anchor_motors = np.asarray([glove.motors[name] for name in glove.motor_names])
    first = runtime.step()
    np.testing.assert_allclose(first.output_positions_rad, anchor, atol=1e-7)
    clock.advance(0.1)
    mid = runtime.step()
    goal_tree = bridge.inverse._last_successful_seed.all_positions
    goal_motors = np.asarray([goal_tree[name] for name in glove.motor_names])
    expected_motors = (anchor_motors + goal_motors) * 0.5
    actual_motors = np.asarray([glove._prepared_motors[name] for name in glove.motor_names])
    np.testing.assert_allclose(actual_motors, expected_motors, atol=1e-7)
    oracle, _ = glove.model.evaluate_motors(dict(zip(glove.motor_names, expected_motors)))
    np.testing.assert_allclose(mid.output_positions_rad, oracle.vector(), atol=1e-7)
    # Wuji只读期间被外部推理/人工改变，旧发送目标绝不是新的切换锚点。
    hand.positions = bridge.glove_to_hand2(_joints(0.3))[0].positions_rad.copy()
    assert np.linalg.norm(hand.positions - hand.commands[-1][0].positions_rad) > 1e-5
    runtime.request_mode(FollowMode.GLOVE_TO_HAND2)
    hand_anchor = hand.read_state().positions_rad.copy()
    tick = runtime.step()
    np.testing.assert_allclose(tick.output_positions_rad, hand_anchor, atol=1e-7)
    runtime.request_mode(FollowMode.HAND2_TO_GLOVE)
    assert glove.homes == 1
    assert glove.enabled == 2
    resumed = runtime.step()
    np.testing.assert_allclose(resumed.output_positions_rad, glove.read_state().pose.vector(), atol=1e-7)
    assert np.linalg.norm(resumed.output_positions_rad - mid.output_positions_rad) > 1e-6
    runtime.close()


@pytest.mark.parametrize("which,offset", [("hand", -100_000_001), ("hand", 1), ("glove", -100_000_001), ("glove", 1), ("pose", 1)])
def test_actual_receive_stamp_stale_or_future_prevents_any_output(bridge, which, offset):
    runtime, hand, glove, _ = _pair(bridge)
    if which == "hand":
        hand.stamp_offset = offset
    elif which == "glove":
        glove.stamp_offset = offset
    else:
        glove.pose_stamp_offset = offset
    with pytest.raises((ValueError, TimeoutError)):
        runtime.start()
    assert hand.commands == glove.commands == []
    assert hand.enabled == glove.homes == 0


def test_unconverged_real_inverse_never_sends_and_releases_only_owned_motor(bridge):
    runtime, hand, glove, _ = _pair(bridge, mode=FollowMode.HAND2_TO_GLOVE)
    runtime.start()
    # start已只读预热同一姿态；换成真正不同的任务才会消耗求解预算。
    hand.positions = bridge.glove_to_hand2(_joints(0.6))[0].positions_rad.copy()
    bridge.inverse.max_nfev = 1
    with pytest.raises(ValueError, match="逆解未收敛"):
        runtime.step()
    assert hand.commands == glove.commands == []
    assert hand.disabled == 0
    assert glove.disabled == 1


def test_slow_real_solver_cannot_extend_original_deadline(bridge, monkeypatch):
    runtime, hand, glove, clock = _pair(bridge)
    runtime.start()
    runtime.command_timeout_ns = 10_000_000
    solve = bridge.glove_to_hand2

    def slow_solve(*args, **kwargs):
        result = solve(*args, **kwargs)
        clock.advance(0.011)
        return result

    monkeypatch.setattr(bridge, "glove_to_hand2", slow_solve)
    with pytest.raises(TimeoutError, match="原始命令期限"):
        runtime.step()
    assert hand.commands == glove.commands == []
    assert hand.disabled == 1 and glove.disabled == 0


def test_preflight_failure_never_sends_and_stops_only_owned_output(bridge):
    runtime, hand, glove, _ = _pair(bridge, mode=FollowMode.HAND2_TO_GLOVE)
    runtime.start()
    glove.fail_validation = True
    with pytest.raises(ValueError, match="预检失败"):
        runtime.step()
    assert hand.commands == glove.commands == []
    assert glove.disabled == 1 and hand.disabled == 0


def test_official_solver_failure_code_cannot_send_its_initial_pose(bridge, monkeypatch):
    runtime, hand, glove, _ = _pair(bridge)
    runtime.start()
    solve = bridge.glove_to_hand2

    def failed_result(*args, **kwargs):
        pose, task = solve(*args, **kwargs)
        return replace(pose, diagnostics={**pose.diagnostics, "success": False}), task

    monkeypatch.setattr(bridge, "glove_to_hand2", failed_result)
    with pytest.raises(ValueError, match="正向求解失败"):
        runtime.step()
    assert hand.commands == glove.commands == []


def test_uncalibrated_mapping_blocks_enable_in_both_directions(bridge):
    for mode in FollowMode:
        runtime, hand, glove, _ = _pair(bridge, mode=mode)
        glove.mapping_calibrated = False
        with pytest.raises(ValueError, match="映射未标定"):
            runtime.start()
        assert hand.enabled == glove.homes == 0
        runtime.close()


@pytest.mark.parametrize("args", [
    ["follow", "--mode", "glove-to-hand2", "--execute"],
    ["follow", "--mode", "hand2-to-glove", "--wuji-serial", "explicit-test-SN", "--execute"],
])
def test_cli_rejects_execution_without_explicit_serial_and_firmware_before_hardware_import(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


@pytest.mark.parametrize("code", [-1, 0])
def test_nlopt_fallback_never_becomes_a_command(bridge, monkeypatch, code):
    runtime, hand, glove, _ = _pair(bridge)
    runtime.start()
    optimizer = bridge.forward.retargeter._core.optimizer
    real_opt = optimizer.opt

    class FailedStatus:
        def __getattr__(self, name):
            return getattr(real_opt, name)

        def last_optimize_result(self):
            return code

    monkeypatch.setattr(optimizer, "opt", FailedStatus())
    with pytest.raises((ValueError, RuntimeError)):
        runtime.step()
    assert hand.commands == glove.commands == []
    assert hand.disabled == 1 and glove.disabled == 0


@pytest.mark.parametrize("arguments", [[], ["inspect"]])
def test_inspect_without_serial_only_discovers_wuji_and_never_connects_first(
    bridge, monkeypatch, arguments,
):
    from motorhand_teleop.real_hand import motor_hand, preflight, wuji_hand2
    from motorhand_teleop.real_hand import runtime as runtime_module

    clock = Clock()
    glove = MotorPlant(clock, bridge.forward.landmarks.model, _joints(0.0))
    monkeypatch.setattr(motor_hand, "EncosGloveHardwareDriver", lambda **kwargs: glove)
    monkeypatch.setattr(preflight, "discover_devices", lambda: ({"serial_number": "not-selected"},))
    monkeypatch.setattr(runtime_module.time, "monotonic_ns", clock)

    def forbidden_constructor(**kwargs):
        pytest.fail("inspect没有SN却构造了Wuji连接")

    monkeypatch.setattr(wuji_hand2, "WujiHand2HardwareDriver", forbidden_constructor)
    assert main(arguments) == 0
    assert glove.homes == glove.enabled == glove.disabled == 0
    assert glove.commands == []
    assert glove.closed
    assert glove.prepares == 1 and glove.state_reads == 2


def test_motor_limit_projection_records_actual_cad_fk_not_unreachable_request(bridge):
    runtime, hand, glove, clock = _pair(bridge, mode=FollowMode.HAND2_TO_GLOVE)
    runtime.start()
    # 固定每个电机在实测位置的极窄可达区间，几何模型仍是真实闭链。
    glove.motor_bounds = {
        name: (value - 0.001, value + 0.001) for name, value in glove.motors.items()
    }
    runtime.step()
    clock.advance(0.2)
    tick = runtime.step()
    assert np.linalg.norm(tick.output_positions_rad - tick.candidate_positions_rad) > 1e-4
    sent, _ = glove.commands[-1]
    _, sent_motors = glove.model.evaluate(dict(sent.joints))
    for name, bounds in glove.motor_bounds.items():
        assert bounds[0] - 1e-7 <= sent_motors[name] <= bounds[1] + 1e-7
    np.testing.assert_allclose(
        tick.output_task.points_m,
        bridge.inverse.glove_model.state(sent.joints).points_m,
        atol=1e-10,
    )
    assert hand.commands == []
    runtime.close()


@pytest.mark.parametrize("mode", list(FollowMode))
def test_right_motor_glove_and_left_wuji_share_canonical_model_without_changing_calibration(mode):
    mixed = create_retargeter(hand="left", glove_hand="right", project_root=ROOT)
    assert mixed.forward.landmarks.hand == "right"
    assert mixed.forward.landmarks.model.kinematics.hand == "right"
    assert mixed.forward.retargeter.hand == "left"
    runtime, hand, glove, clock = _pair(mixed, mode=mode)
    runtime.start()
    runtime.step()
    clock.advance(0.2)
    tick = runtime.step()
    assert np.isfinite(tick.output_task.points_m).all()
    if mode is FollowMode.GLOVE_TO_HAND2:
        assert len(hand.commands) == 2 and glove.commands == []
        np.testing.assert_allclose(tick.output_positions_rad[[5, 9, 13, 17]], 0.0)
        expected_task = mixed.hand2_model.state(tick.output_positions_rad)
    else:
        assert len(glove.commands) == 2 and hand.commands == []
        expected_task = mixed.inverse.glove_model.state(dict(zip(GLOVE_JOINTS, tick.output_positions_rad)))
    np.testing.assert_allclose(tick.output_task.points_m, expected_task.points_m, atol=1e-10)
    runtime.close()


def test_measured_nonzero_wuji_side_is_continuous_then_smoothly_returns_to_fixed_zero(bridge):
    runtime, hand, glove, clock = _pair(bridge)
    side = np.array([5, 9, 13, 17])
    hand.positions[side] = 0.206
    runtime.start()
    first = runtime.step()
    np.testing.assert_allclose(first.output_positions_rad, hand.positions, atol=1e-12)
    np.testing.assert_allclose(first.candidate_positions_rad[side], 0.0, atol=1e-12)
    clock.advance(0.1)
    middle = runtime.step()
    np.testing.assert_allclose(middle.output_positions_rad[side], 0.103, atol=1e-12)
    clock.advance(0.1)
    final = runtime.step()
    np.testing.assert_allclose(final.output_positions_rad[side], 0.0, atol=1e-12)
    assert glove.commands == [] and glove.homes == 0
    # FK必须保留实测侧摆，而不是悄悄把输入side当成求解约束覆盖为0。
    fixed_state = first.output_positions_rad.copy()
    fixed_state[side] = 0.0
    fixed_task = bridge.hand2_model.state(fixed_state)
    assert np.linalg.norm(first.output_task.points_m - fixed_task.points_m) > 0.001
    runtime.close()


def test_discover_command_only_calls_sdk_scan_without_constructing_either_driver(monkeypatch, capsys):
    from motorhand_teleop.real_hand import motor_hand, preflight, wuji_hand2

    def forbidden_constructor(**kwargs):
        pytest.fail("discover不应实例化任何硬件驱动")

    seen = []

    def scan(*, timeout_ms):
        seen.append(timeout_ms)
        return ({"serial_number": "not-selected", "firmware_version": "2.2.3"},)

    monkeypatch.setattr(motor_hand, "EncosGloveHardwareDriver", forbidden_constructor)
    monkeypatch.setattr(wuji_hand2, "WujiHand2HardwareDriver", forbidden_constructor)
    monkeypatch.setattr(preflight, "discover_devices", scan)
    assert main(["discover", "--timeout-ms", "321"]) == 0
    assert seen == [321]
    report = json.loads(capsys.readouterr().out)
    assert report["Wuji发现设备"][0]["serial_number"] == "not-selected"


def test_inspect_serial_serializes_actual_dataclass_identity_without_acquiring_output(bridge, monkeypatch, capsys):
    from motorhand_teleop.real_hand import motor_hand, preflight, wuji_hand2
    from motorhand_teleop.real_hand import runtime as runtime_module

    runtime, hand, glove, clock = _pair(bridge)
    monkeypatch.setattr(motor_hand, "EncosGloveHardwareDriver", lambda **kwargs: glove)
    monkeypatch.setattr(wuji_hand2, "WujiHand2HardwareDriver", lambda **kwargs: hand)
    monkeypatch.setattr(preflight, "discover_devices", lambda: ({"serial_number": "explicit-test-SN"},))
    monkeypatch.setattr(runtime_module.time, "monotonic_ns", clock)
    assert main(["inspect", "--wuji-serial", "explicit-test-SN"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["Wuji身份"]["serial_number"] == "explicit-test-SN"
    assert len(report["Wuji20语义关节rad"]) == 20
    assert hand.enabled == hand.disabled == glove.enabled == glove.disabled == glove.homes == 0
    assert hand.commands == glove.commands == []
    # 本地PairRuntime未启动，仅用于构造有真实CAD状态的数值边界。
    runtime.close()


def test_prepare_is_readonly_and_never_reuses_its_target_as_a_fresh_command(bridge):
    runtime, hand, glove, clock = _pair(bridge, mode=FollowMode.HAND2_TO_GLOVE)
    runtime.prepare()
    prepared = bridge.inverse._last_successful_positions.copy()
    assert hand.enabled == hand.disabled == glove.enabled == glove.disabled == glove.homes == 0
    assert hand.commands == glove.commands == []
    # 预热后真实主导端改变；正式周期必须依据新反馈重新求解。
    hand.positions = bridge.glove_to_hand2(_joints(0.4))[0].positions_rad.copy()
    runtime.start()
    runtime.step()
    clock.advance(0.2)
    fresh = runtime.step()
    assert np.linalg.norm(fresh.candidate_positions_rad - prepared) > 0.001
    assert np.linalg.norm(fresh.output_positions_rad - prepared) > 0.001
    assert fresh.hand2_state.received_ns == clock()
    assert len(glove.commands) == 2 and hand.commands == []
    runtime.close()


def test_connection_discards_slow_cold_motor_fk_and_samples_wuji_after_fresh_motor(bridge):
    clock = Clock()
    source = bridge.glove_to_hand2(_joints(0.1))[0].positions_rad
    retargeter = bridge.forward.retargeter
    limits = retargeter._core.optimizer.robot.joint_limits[retargeter._output_indices]
    order = []

    class OrderedHand(HandPlant):
        def read_state(self):
            order.append("wuji")
            return super().read_state()

    class ColdMotor(MotorPlant):
        def read_state(self):
            order.append("motor")
            state = super().read_state()
            if self.state_reads == 1:
                # 首次实际FK跨越100ms，但接收时刻不续期，冷帧只允许被丢弃。
                clock.advance(0.109)
            return state

    hand = OrderedHand(clock, source, limits)
    glove = ColdMotor(clock, bridge.forward.landmarks.model, _joints(0.03))
    runtime = PairRuntime(hand, glove, bridge, clock_ns=clock)
    runtime.connect(serial_number="explicit-test-SN")
    assert glove.prepares == 1 and glove.state_reads == 1
    order.clear()
    report = runtime.inspect()
    assert order == ["motor", "wuji"]
    assert report["MotorHand接收时刻ns"] == report["Wuji接收时刻ns"] == clock()
    assert glove.state_reads == 2
    runtime.close()
    assert hand.enabled == hand.disabled == glove.enabled == glove.disabled == glove.homes == 0
    assert hand.commands == glove.commands == []


def _complex_readonly_pair(bridge):
    """现场复杂指节姿态，使用真实mapping软限位，而非近零无饱和样本。"""
    clock = Clock()
    actual_joints = dict(zip(GLOVE_JOINTS, (
        -0.45557864530144876, 0.40510411934582685, -0.10513885764870402, 0.11835835174349182,
        -0.14851558818714744, 0.15923942108020767, -0.21220637259037775, 0.700980775982606,
        -0.02917190458658229, -0.1002577582454216, 0.02818358328503218, 0.08259613156419458,
    )))
    source = bridge.forward.landmarks
    # 父级实际SDK只读样本；不是forward(glove)生成的较容易目标。
    target = np.asarray([
        0.4735102653503418, -0.5586140155792236, -0.3877113461494446, 0.17075419425964355,
        0.40879833698272705, -0.000769197940826416, -0.0451662540435791, 0.46336185932159424,
        0.21620434522628784, 0.2063676118850708, -0.21451014280319214, 0.11865133047103882,
        0.08054077625274658, 0.04827171564102173, -0.005648255348205566, 0.39351558685302734,
        -0.05427360534667969, -0.07631778717041016, 0.277258038520813, 0.2577986717224121,
    ])
    retargeter = bridge.forward.retargeter
    physical_limits = retargeter._core.optimizer.robot.joint_limits[retargeter._output_indices]
    hand = HandPlant(clock, target, physical_limits)
    glove = MotorPlant(clock, source.model, actual_joints)
    mapping = json.loads((ROOT / "hardware/encos_dexhand/configs/glove_mapping.json").read_text())
    for config in mapping["motors"].values():
        motor = source.model.joint_motor_names[config["joint"]]
        limits = [
            float(np.deg2rad((raw - config["zero_deg"]) * config["sign"] * config["scale"]))
            for raw in (config["min_deg"], config["max_deg"])
        ]
        glove.motor_bounds[motor] = (min(limits), max(limits))
    runtime = PairRuntime(
        hand, glove, bridge, mode=FollowMode.HAND2_TO_GLOVE,
        execute=False, clock_ns=clock, transition_s=0.5,
    )
    runtime.connect(serial_number="explicit-test-SN")
    return runtime, hand, glove, clock, actual_joints


def test_complex_real_pose_uses_limited_cad_tree_without_repeated_output_ik(bridge):
    runtime, hand, glove, clock, actual = _complex_readonly_pair(bridge)
    runtime.prepare()
    runtime.start()
    measured = glove.read_state().pose
    np.testing.assert_allclose(measured.vector(), list(actual.values()), atol=1e-6)
    tick = runtime.step()
    for _ in range(3):
        clock.advance(0.05)
        tick = runtime.step()
    assert glove._prepared_tree is not None
    assert any(value < 0.0 for value in measured.vector()[4:])
    for name, value in glove._prepared_motors.items():
        lower, upper = glove.motor_bounds[name]
        assert lower - 1e-9 <= value <= upper + 1e-9
    # 独立慢q→IK oracle仅用于离线断言，不在被测实时消费者路径中。
    oracle = bridge.inverse.glove_model.state(dict(zip(GLOVE_JOINTS, tick.output_positions_rad)))
    np.testing.assert_allclose(tick.output_task.points_m, oracle.points_m, atol=1e-7)
    assert hand.commands == glove.commands == []
    assert hand.enabled == hand.disabled == glove.enabled == glove.disabled == glove.homes == 0
    runtime.close()


def test_reverse_output_task_consumes_prepared_tree_without_restarting_joint_ik(bridge, monkeypatch):
    runtime, hand, glove, _ = _pair(bridge, mode=FollowMode.HAND2_TO_GLOVE, execute=False)
    slow_oracle = bridge.inverse.glove_model.state

    def forbidden_output_ik(*args, **kwargs):
        pytest.fail("输出任务状态必须读取同周期已验证树，禁止从reference0再次q→m逆解")

    monkeypatch.setattr(bridge.inverse.glove_model, "state", forbidden_output_ik)
    monkeypatch.setattr(glove, "constrain_position", forbidden_output_ik)
    runtime.prepare()
    runtime.start()
    tick = runtime.step()
    oracle = slow_oracle(dict(zip(GLOVE_JOINTS, tick.output_positions_rad)))
    np.testing.assert_allclose(tick.output_task.points_m, oracle.points_m, atol=1e-7)
    assert glove._prepared_tree is not None
    assert hand.commands == glove.commands == []
    runtime.close()


def test_nonlinear_motor_limits_bind_targets_without_rewriting_outside_actual_feedback(bridge):
    runtime, hand, glove, clock, actual = _complex_readonly_pair(bridge)
    # 单独收紧数值边界的直驱CMC软件窗口，明确制造“实际合法但目标越软件界”。
    # 不改真实mapping文件；真实literal回放基准仍保留原始全部软限位。
    motor = glove.model.joint_motor_names["thumb_cmc_flex"]
    glove.motor_bounds[motor] = (-0.1, 0.1)
    limits = glove.model_motor_limits_rad
    assert len(limits) == 12
    assert any(glove.motors[name] < lower or glove.motors[name] > upper
               for name, (lower, upper) in limits.items())
    runtime.prepare()
    runtime.start()
    first = runtime.step()
    np.testing.assert_allclose(first.glove_pose.vector(), list(actual.values()), atol=1e-6)
    clock.advance(0.5)
    planned = runtime.step()
    # 独立真实CAD q→m oracle检查规划本身可发，而不只检查后续raw裁剪结果。
    candidate = glove.model.evaluate_raw(dict(zip(GLOVE_JOINTS, planned.candidate_positions_rad)))
    for name, (lower, upper) in limits.items():
        assert lower - 1e-7 <= candidate.motor_positions[name] <= upper + 1e-7
    # 真正位于目标限位外的输入原样发布，不拿规划边界篡改真人姿态。
    np.testing.assert_allclose(planned.glove_pose.vector(), list(actual.values()), atol=1e-6)
    assert hand.commands == glove.commands == []
    runtime.close()


def test_unknown_motor_bounds_do_not_reuse_a_previous_devices_limits(bridge):
    runtime, hand, glove, clock = _pair(
        bridge, mode=FollowMode.HAND2_TO_GLOVE, execute=False,
    )
    historical = {
        name: (value - 0.0001, value + 0.0001) for name, value in glove.motors.items()
    }
    bridge.inverse.set_motor_limits(historical)

    class InputWithoutLimitCapability:
        def __getattr__(self, name):
            if name == "model_motor_limits_rad":
                raise AttributeError(name)
            return getattr(glove, name)

    runtime.glove_driver = InputWithoutLimitCapability()
    runtime.prepare()
    runtime.start()
    runtime.step()
    clock.advance(0.2)
    tick = runtime.step()
    actual = glove.model.evaluate_raw(dict(zip(GLOVE_JOINTS, tick.output_positions_rad)))
    # 换设备后不能被上一设备的极窄窗口锁住；用真实机构反解检查最终输出。
    assert any(
        value < historical[name][0] - 0.001 or value > historical[name][1] + 0.001
        for name, value in actual.motor_positions.items()
    )
    assert hand.commands == glove.commands == []
    assert hand.enabled == glove.enabled == glove.homes == 0
    runtime.close()


class ForceMotorPlant:
    """真实CAD数值机构；操作者夹持姿态，力输出不篡改位置反馈。"""

    def __init__(self, plant, calibration):
        from motorhand_teleop.real_hand.motor_hand import EncosGloveHardwareDriver
        from motorhand_teleop.real_hand.force_linear import MaxTorqueLinearConfig

        self.plant = plant
        self.model = plant.model
        self.hand = "right"
        self.metadata = EncosGloveHardwareDriver(
            torque_constant_nm_per_a=(
                None if isinstance(calibration, MaxTorqueLinearConfig)
                else calibration.glove_torque_constants_by_id
            ),
            force_torque_limits_nm=(
                calibration.glove_torque_limits_by_id
                if isinstance(calibration, MaxTorqueLinearConfig) else None
            ),
        )
        self.motors = self.metadata.motors
        self.force_mode = False
        self.force_commands = []
        self.force_torque_commands = []

    def __getattr__(self, name):
        return getattr(self.plant, name)

    @property
    def model_motor_limits_rad(self):
        return self.metadata.model_motor_limits_rad

    @property
    def model_motor_torque_constants_nm_per_a(self):
        return self.metadata.model_motor_torque_constants_nm_per_a

    @property
    def model_current_limits_a(self):
        return self.metadata.model_current_limits_a

    @property
    def model_force_torque_limits_nm(self):
        return self.metadata.model_force_torque_limits_nm

    @property
    def model_force_torque_command_limits_nm(self):
        return self.metadata.model_force_torque_command_limits_nm

    def read_state(self):
        from motorhand_teleop.real_hand.motor_hand import EncosGloveState

        state = self.plant.read_state()
        raw = {
            mid: config.zero_deg + np.rad2deg(
                state.motor_positions_rad[self.model.joint_motor_names[config.joint]]
                / (config.sign * config.scale)
            )
            for mid, config in self.motors.items()
        }
        return EncosGloveState(
            state.pose, {}, state.online_ids, state.received_ns,
            raw, state.motor_positions_rad, self.plant._feedback_seed.all_positions,
        )

    def enable_force_feedback(self):
        if not self.plant.homes:
            self.plant.home()
        self.force_mode = True

    def validate_force_current(self, current_a):
        if not np.isfinite(current_a).all():
            raise ValueError("数值力输出必须有限")
        if self.plant.fail_validation:
            raise ValueError("数值力输出完整预检失败")

    def send_force_current(self, current_a, *, deadline_ns):
        assert self.force_mode and self.plant.clock() < deadline_ns
        self.validate_force_current(current_a)
        self.force_commands.append(np.array(current_a))
        return tuple(current_a)

    def validate_force_torque(self, torque_nm):
        if not np.isfinite(torque_nm).all() or self.plant.fail_validation:
            raise ValueError("数值名义力矩完整预检失败")

    def send_force_torque(self, torque_nm, *, deadline_ns):
        assert self.force_mode and self.plant.clock() < deadline_ns
        self.validate_force_torque(torque_nm)
        self.force_torque_commands.append(np.array(torque_nm))
        return tuple(torque_nm)

    def enable(self):
        assert not self.force_mode
        self.plant.enable()

    def disable(self):
        self.force_mode = False
        self.plant.disable()


class EffortHandPlant(HandPlant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.effort = np.zeros(20)

    def read_state(self):
        return replace(super().read_state(), effort_a=self.effort.copy())


@pytest.fixture
def force_pair(bridge):
    from motorhand_teleop.real_hand.force_feedback import (
        ForceFeedbackCalibration, ForceFeedbackController,
    )

    # 仅限数值模型的已知系数，禁止写入真机配置或作为启动默认。
    calibration = ForceFeedbackCalibration(
        schema_version=1, calibrated=True, source_id="offline-pair-plant-only",
        wuji_serial_number="explicit-test-SN", glove_hand="right",
        wuji_torque_per_amp_nm=[-0.2] * 20, wuji_current_bias_a=[0.0] * 20,
        glove_torque_constant_nm_per_a=[0.1] * 12,
        max_current_a=[0.2] * 12, max_slew_a_s=[10.0] * 12, gain=0.2,
        ramp_duration_s=0.1, jacobian_max_age_s=10.0,
        jacobian_refresh_interval_s=100.0,
    )
    clock = Clock()
    positions = bridge.glove_to_hand2(_joints(0.1))[0].positions_rad
    robot = bridge.forward.retargeter
    limits = robot._core.optimizer.robot.joint_limits[robot._output_indices]
    hand = EffortHandPlant(clock, positions, limits)
    glove = ForceMotorPlant(
        MotorPlant(clock, bridge.forward.landmarks.model, _joints(0.03)), calibration,
    )
    controller = ForceFeedbackController(
        calibration, bridge, glove, RetargetPipelineFactory(project_root=ROOT),
        clock_ns=clock,
    )
    runtime = PairRuntime(
        hand, glove, bridge, execute=True, clock_ns=clock,
        force_feedback=controller, transition_s=0.1,
    )
    runtime.connect(serial_number="explicit-test-SN")
    yield runtime, hand, glove, clock
    runtime.close()


def test_takeover_reflects_measured_effort_without_position_pull_and_restarts_from_zero(force_pair):
    runtime, hand, glove, clock = force_pair
    runtime.start()
    first = runtime.step()
    np.testing.assert_array_equal(first.force_feedback.model_current_a, np.zeros(12))
    human_pose = glove.read_state().pose.vector()
    delta = np.linspace(-1e-4, 2e-4, 20)
    hand.effort = delta
    clock.advance(0.1)
    loaded = runtime.step()
    assert np.linalg.norm(loaded.force_feedback.model_current_a) > 1e-7
    np.testing.assert_allclose(loaded.force_feedback.wuji_actuator_torque_nm, -0.2 * delta)
    np.testing.assert_allclose(glove.read_state().pose.vector(), human_pose, atol=1e-10)
    assert glove.commands == []
    assert not np.any(hand.commands[-1][0].effort_a)

    hand.effort = -delta
    clock.advance(0.02)
    reversed_load = runtime.step()
    assert np.max(np.abs(
        np.asarray(reversed_load.force_feedback.model_current_a)
        - np.asarray(loaded.force_feedback.model_current_a)
    )) > 1e-7

    runtime.request_mode(FollowMode.HAND2_TO_GLOVE)
    resumed = runtime.step()
    np.testing.assert_allclose(resumed.output_positions_rad, human_pose, atol=2e-7)
    assert resumed.force_feedback is None and not glove.force_mode
    runtime.request_mode(FollowMode.GLOVE_TO_HAND2)
    restarted = runtime.step()
    np.testing.assert_array_equal(restarted.force_feedback.model_current_a, np.zeros(12))
    assert glove.homes == 1


def test_force_preflight_failure_stops_both_owned_outputs_before_new_position(force_pair):
    runtime, hand, glove, clock = force_pair
    runtime.start()
    runtime.step()
    previous_reference = hand.reference.copy()
    previous_current = glove.force_commands[-1].copy()
    glove.plant.fail_validation = True
    hand.effort = np.full(20, 1e-4)
    clock.advance(0.02)
    with pytest.raises(ValueError, match="完整预检"):
        runtime.step()
    np.testing.assert_array_equal(hand.reference, previous_reference)
    np.testing.assert_array_equal(glove.force_commands[-1], previous_current)
    assert not glove.force_mode and hand.disabled and glove.disabled


def test_force_missing_effort_is_rejected_before_either_side_is_enabled(force_pair):
    runtime, hand, glove, _ = force_pair
    hand.read_state = lambda: WujiState(hand.positions, None, None, hand.clock(), 20)
    with pytest.raises(RuntimeError, match="effort"):
        runtime.start()
    assert hand.enabled == 0 and glove.homes == 0 and not glove.force_mode


@pytest.mark.parametrize("linear_mode", [False, True])
def test_inference_takeover_uses_same_solver_and_force_role_without_pose_jump(force_pair, linear_mode):
    from motorhand_teleop.bidirectional import ControlMode
    from motorhand_teleop.real_hand.control import RealWujiControlLoop

    runtime, hand, glove, clock = force_pair
    controller, runtime.force_feedback = runtime.force_feedback, None
    if linear_mode:
        from motorhand_teleop.real_hand.force_feedback import ForceFeedbackController
        from motorhand_teleop.real_hand.force_linear import MaxTorqueLinearConfig
        from motorhand_teleop.real_hand.motor_hand import EncosGloveHardwareDriver

        controller.close()
        config = replace(
            MaxTorqueLinearConfig.from_json(ROOT / "config/feedback/max-torque-linear-right.json"),
            wuji_serial_number="explicit-test-SN", ramp_duration_s=0.1,
            jacobian_max_age_s=10.0, jacobian_refresh_interval_s=100.0,
        )
        glove.metadata = EncosGloveHardwareDriver(force_torque_limits_nm=config.glove_torque_limits_by_id)
        glove.motors = glove.metadata.motors
        controller = ForceFeedbackController(
            config, runtime.retargeter, glove, RetargetPipelineFactory(project_root=ROOT),
            clock_ns=clock,
        )
    hand.validate_enable = lambda: hand.validate(
        SimpleNamespace(positions_rad=hand.positions, effort_a=np.zeros(20)),
    )
    hand.emergency_stop = hand.disable
    glove.emergency_stop = glove.disable
    loop = RealWujiControlLoop(
        hand, runtime.retargeter, glove_driver=glove, force_feedback=controller,
    )
    try:
        loop.connect(serial_number="explicit-test-SN")
        loop.enable(now_ns=clock())
        inferred = runtime.retargeter.glove_to_hand2(_joints(0.12))[0].positions_rad
        old = loop.step(inference_positions_rad=inferred, now_ns=clock())
        loop.request_mode(ControlMode.GLOVE_TAKEOVER, now_ns=clock())
        first = loop.step(now_ns=clock())
        np.testing.assert_array_equal(first.sent_positions_rad, old.sent_positions_rad)
        zero = first.force_feedback.model_torque_nm if linear_mode else first.force_feedback.model_current_a
        np.testing.assert_array_equal(zero, np.zeros(12))
        human = glove.read_state().pose.vector()
        old_glove_reference = dict(glove.plant.reference)
        hand.effort = np.linspace(-0.5, 1.0, 20) if linear_mode else np.linspace(-1e-4, 2e-4, 20)
        clock.advance(0.1)
        loaded = loop.step(now_ns=clock())
        if linear_mode:
            assert np.linalg.norm(loaded.force_feedback.model_torque_nm) > 1e-7
            assert loaded.force_feedback.wuji_actuator_torque_nm is None
            np.testing.assert_allclose(loaded.force_feedback.wuji_load_fraction, hand.effort / 2.0)
        else:
            assert np.linalg.norm(loaded.force_feedback.model_current_a) > 1e-7
            np.testing.assert_allclose(loaded.force_feedback.wuji_actuator_torque_nm, -0.2 * hand.effort)
        assert glove.plant.reference == old_glove_reference
        loop.request_mode(ControlMode.INFERENCE, now_ns=clock())
        resumed = loop.step(inference_positions_rad=inferred, now_ns=clock())
        np.testing.assert_array_equal(resumed.sent_positions_rad, loaded.sent_positions_rad)
        np.testing.assert_allclose(resumed.shadow_glove_positions_rad, human, atol=2e-7)
        assert resumed.force_feedback is None and glove.homes == 1
    finally:
        loop.disable()
        loop.close()


def test_maximum_linear_takeover_works_without_kt_and_preserves_single_position_master(bridge):
    from motorhand_teleop.real_hand.force_feedback import ForceFeedbackController
    from motorhand_teleop.real_hand.force_linear import MaxTorqueLinearConfig

    config = replace(
        MaxTorqueLinearConfig.from_json(ROOT / "config/feedback/max-torque-linear-right.json"),
        wuji_serial_number="explicit-test-SN", ramp_duration_s=0.1,
        jacobian_max_age_s=10.0, jacobian_refresh_interval_s=100.0,
    )
    clock = Clock()
    positions = bridge.glove_to_hand2(_joints(0.1))[0].positions_rad
    robot = bridge.forward.retargeter
    hand = EffortHandPlant(
        clock, positions, robot._core.optimizer.robot.joint_limits[robot._output_indices],
    )
    glove = ForceMotorPlant(
        MotorPlant(clock, bridge.forward.landmarks.model, _joints(0.03)), config,
    )
    controller = ForceFeedbackController(
        config, bridge, glove, RetargetPipelineFactory(project_root=ROOT), clock_ns=clock,
    )
    runtime = PairRuntime(
        hand, glove, bridge, execute=True, clock_ns=clock,
        force_feedback=controller, transition_s=0.1,
    )
    try:
        runtime.connect(serial_number="explicit-test-SN")
        runtime.start()
        first = runtime.step()
        np.testing.assert_array_equal(first.force_feedback.model_torque_nm, np.zeros(12))
        assert first.force_feedback.model_current_a is None
        human = glove.read_state().pose.vector()
        hand.effort = np.linspace(-0.5, 1.0, 20)
        clock.advance(0.1)
        loaded = runtime.step()
        feedback = loaded.force_feedback
        assert feedback.mapping_mode == "max_torque_linear_estimate"
        assert feedback.wuji_actuator_torque_nm is None and feedback.glove_joint_torque_nm is None
        np.testing.assert_allclose(feedback.wuji_load_fraction, hand.effort / 2.0)
        assert np.linalg.norm(feedback.model_torque_nm) > 1e-7
        caps = np.array([glove.model_force_torque_limits_nm[name] for name in glove.motor_names])
        assert np.all(np.abs(feedback.model_torque_nm) <= caps)
        np.testing.assert_allclose(glove.read_state().pose.vector(), human, atol=1e-10)
        assert glove.commands == [] and glove.force_commands == []
        runtime.request_mode(FollowMode.HAND2_TO_GLOVE)
        returned = runtime.step()
        np.testing.assert_allclose(returned.output_positions_rad, human, atol=2e-7)
        assert returned.force_feedback is None and not glove.force_mode
        runtime.request_mode(FollowMode.GLOVE_TO_HAND2)
        np.testing.assert_array_equal(runtime.step().force_feedback.model_torque_nm, np.zeros(12))
        assert glove.homes == 1
    finally:
        runtime.close()


def test_pose_refresh_preserves_position_following_without_stale_force_or_rehoming(force_pair):
    runtime, hand, glove, clock = force_pair
    runtime.start()
    runtime.step()
    held_reference = dict(glove.plant.reference)
    before = hand.positions.copy()
    human = dict(glove.read_state().pose.joints)
    human["index_pip"] += 0.4
    _, motors = glove.model.evaluate(human)
    glove.plant.motors = dict(motors)
    hand.effort = np.full(20, 1e-4)
    clock.advance(0.2)
    # 实时线程可重入Condition；这里只阻止worker发布新矩阵，确定性覆盖等待期。
    with runtime.force_feedback._condition:
        waiting = runtime.step()
    assert waiting.force_feedback.projection_status == "refreshing_pose"
    np.testing.assert_array_equal(glove.force_commands[-1], np.zeros(12))
    hand.advance_mechanics()
    assert np.linalg.norm(hand.positions - before) > 0.02
    assert runtime._running and glove.force_mode
    assert hand.disabled == glove.disabled == 0
    assert glove.homes == 1 and glove.plant.reference == held_reference
