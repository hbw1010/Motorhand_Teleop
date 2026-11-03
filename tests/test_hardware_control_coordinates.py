"""Offline control-boundary regressions using the shared canonical geometry and IK."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wujihand2_force_glove.bidirectional import (
    BidirectionalRetargeter,
    ControlMode,
    GloveInverseSolver,
    GloveTaskModel,
    Hand2TaskModel,
)
from wujihand2_force_glove.domain import GLOVE_JOINTS, GlovePose, WujiState
from wujihand2_force_glove.real_hand import control as hardware_control
from wujihand2_force_glove.real_hand.control import RealWujiControlLoop
from wujihand2_force_glove.landmarks import Ve33LandmarkSource
from wujihand2_force_glove.retargeting import (
    GloveWujiRetargetPipeline,
    default_retarget_config,
)
from wujihand2_force_glove.ve33_adapter import Ve33GloveModel


ROOT = Path(__file__).parents[1]


@pytest.fixture
def bridge():
    assets = ROOT / "assets" / "hand_motor_ve33"
    source = Ve33LandmarkSource("right", asset_root=assets)
    pipeline = GloveWujiRetargetPipeline.from_yaml(
        "right", asset_root=assets,
        config_path=default_retarget_config(ROOT, "right"),
        estimate_jacobian=False,
    )
    return BidirectionalRetargeter(
        pipeline,
        inverse=GloveInverseSolver(GloveTaskModel(source)),
        hand2_model=Hand2TaskModel(pipeline.retargeter, source.canonical_to_optimizer_row),
    )


class HandBoundary:
    def __init__(self, positions):
        self.state = WujiState(positions, None, None, 7, 20)
        self.commands = []
        self.events = []
        self.validation_error = None

    def read_state(self):
        return self.state

    def validate_enable(self):
        self.events.append("approve_hand")

    def enable(self):
        self.events.append("enable_hand")

    def validate(self, target):
        if self.validation_error is not None:
            raise self.validation_error

    def send_target(self, target, *, deadline_ns):
        self.commands.append((target, deadline_ns))


class GloveBoundary:
    def __init__(self, joints):
        self.model = Ve33GloveModel("right", asset_root=ROOT / "assets" / "hand_motor_ve33")
        self.motor_names = self.model.motor_names
        self._feedback = self.model.evaluate_raw(joints)
        self.pose = self.model.to_glove_pose(self._feedback, timestamp_ns=5)
        self._trees = {}
        self._last_preview = self._feedback
        self.validation_error = None
        self.positions = []
        self.enable_count = 0
        self.disable_count = 0
        self.events = []

    def validate_command(self, *, target_pose=None, current_a=None):
        if self.validation_error is not None:
            raise self.validation_error
        if target_pose is not None:
            self.position_tree(target_pose)

    def read_state(self):
        self.events.append("read_glove")
        return SimpleNamespace(
            pose=self.pose,
            motor_positions_rad=dict(self._feedback.motor_positions),
        )

    def read_pose(self):
        return self.read_state().pose

    def _preview(self, solution, timestamp_ns):
        pose = self.model.to_glove_pose(solution, timestamp_ns=timestamp_ns)
        self._trees[id(pose)] = (pose, dict(solution.all_positions))
        self._last_preview = solution
        return pose

    def position_from_tree(self, tree, *, timestamp_ns):
        return self._preview(self.model.kinematics.evaluate_tree(tree), timestamp_ns)

    def position_tree(self, pose):
        cached_pose, tree = self._trees[id(pose)]
        assert cached_pose is pose
        return tree

    def position_from_motors(self, motors, *, timestamp_ns):
        _, solution = self.model.evaluate_motor_state(
            motors, seed=self._last_preview, timestamp_ns=timestamp_ns,
        )
        return self._preview(solution, timestamp_ns)

    def motor_vector(self, pose):
        tree = self.position_tree(pose)
        return np.asarray([tree[name] for name in self.motor_names])

    def send_position(self, pose, *, deadline_ns):
        self.positions.append((pose, deadline_ns))

    def home(self):
        self.events.append("home_glove")
        self._feedback = self.model.evaluate_raw(_joints(0.0))
        self.pose = self.model.to_glove_pose(self._feedback, timestamp_ns=self.pose.timestamp_ns)
        self._last_preview = self._feedback

    def enable(self):
        self.enable_count += 1
        self.events.append("enable_glove")

    def disable(self):
        self.disable_count += 1
        self.events.append("disable_glove")

    def prepare(self):
        self.events.append("prepare_glove")


def _joints(flexion):
    joints = dict.fromkeys(GLOVE_JOINTS, 0.0)
    joints["index_pip"] = flexion
    joints["middle_pip"] = flexion * 0.5
    return joints


def _switching_loop(bridge):
    joints = _joints(0.05)
    anchor = bridge.glove_to_hand2(joints)[0].positions_rad
    hand = HandBoundary(anchor)
    glove = GloveBoundary(joints)
    loop = RealWujiControlLoop(
        hand, bridge, glove_driver=glove, initial_mode=ControlMode.GLOVE_TAKEOVER,
    )
    first = loop.step(now_ns=100)
    loop.request_mode(ControlMode.INFERENCE, now_ns=200, duration_s=1e-6)
    return loop, hand, glove, first.sent_positions_rad


def _assert_final_task_projection(bridge, glove_q, hand_q):
    expected, task = bridge.hand2_to_glove(hand_q)
    assert expected.success
    actual_points = bridge.inverse.glove_model.state(dict(zip(GLOVE_JOINTS, glove_q))).points_m
    optimum_points = bridge.inverse.glove_model.state(
        dict(zip(GLOVE_JOINTS, expected.glove_positions_rad))
    ).points_m
    actual_rms = np.sqrt(np.mean(np.sum((actual_points - task.points_m) ** 2, axis=1)))
    optimum_rms = np.sqrt(np.mean(np.sum((optimum_points - task.points_m) ** 2, axis=1)))
    # 错用raw candidate在本场景至少多0.7mm；只容许0.05mm迭代误差。
    assert actual_rms <= optimum_rms + 0.00005


def test_inference_follows_final_handover_target_in_one_command(bridge):
    loop, hand, glove, anchor = _switching_loop(bridge)
    glove_anchor = glove.pose.vector()
    motor_anchor = np.asarray([glove._feedback.motor_positions[name] for name in glove.motor_names])
    candidate = bridge.glove_to_hand2(_joints(0.55))[0].positions_rad

    start = loop.step(inference_positions_rad=candidate, now_ns=200)
    np.testing.assert_allclose(start.sent_positions_rad, anchor, atol=1e-12)
    np.testing.assert_allclose(start.shadow_glove_positions_rad, glove_anchor, atol=1e-7, rtol=0)
    assert not np.allclose(start.sent_positions_rad, candidate)
    # 重复请求不能重新enable或改变任一侧过渡的起算时间。
    loop.request_mode(ControlMode.INFERENCE, now_ns=300, duration_s=1.0)
    assert glove.enable_count == 1
    assert glove.events[-2:] == ["enable_glove", "read_glove"]
    mid = loop.step(inference_positions_rad=candidate, now_ns=700)
    np.testing.assert_allclose(mid.sent_positions_rad, (anchor + candidate) * 0.5, atol=1e-12)
    end = loop.step(inference_positions_rad=candidate, now_ns=1200)
    np.testing.assert_allclose(end.sent_positions_rad, candidate, atol=1e-12)

    for index, tick in enumerate((start, mid, end)):
        actual_target, deadline = hand.commands[index + 1]
        pose, glove_deadline = glove.positions[index]
        assert isinstance(pose, GlovePose)
        assert tuple(pose.joints) == GLOVE_JOINTS
        np.testing.assert_array_equal(pose.vector(), tick.shadow_glove_positions_rad)
        if index:
            # 电机轨迹的五次smoothstep中点=1/2，真实q12并不线性。
            inverse_m = 2.0 * glove.motor_vector(pose) - motor_anchor if index == 1 else glove.motor_vector(pose)
            inverse_pose = glove.position_from_motors(
                dict(zip(glove.motor_names, inverse_m)), timestamp_ns=tick.timestamp_ns,
            )
            _assert_final_task_projection(bridge, inverse_pose.vector(), actual_target.positions_rad)
        assert glove_deadline == deadline == tick.timestamp_ns + loop.command_timeout_ns

    snapshot = loop.state_bus.snapshot()
    assert snapshot.hand2.device.timestamp_ns == hand.state.received_ns
    assert snapshot.hand2.task.timestamp_ns == hand.state.received_ns


def test_takeover_never_drives_glove_to_reverse_shadow(bridge):
    hand = HandBoundary(np.zeros(20))
    glove = GloveBoundary(_joints(0.25))
    loop = RealWujiControlLoop(hand, bridge, glove_driver=glove)
    loop.request_mode(ControlMode.GLOVE_TAKEOVER, now_ns=100, duration_s=1e-6)
    loop.step(now_ns=100)
    loop.step(now_ns=1100)
    assert glove.positions == []
    np.testing.assert_allclose(hand.commands[-1][0].positions_rad[[5, 9, 13, 17]], 0.0, atol=1e-12)
    snapshot = loop.state_bus.snapshot()
    assert snapshot.glove.device.timestamp_ns == glove.pose.timestamp_ns
    assert snapshot.glove.task.timestamp_ns == glove.pose.timestamp_ns



def test_expensive_inverse_cannot_renew_original_deadline(bridge, monkeypatch):
    hand = HandBoundary(np.zeros(20))
    glove = GloveBoundary(_joints(0.0))
    hand.state = WujiState(np.zeros(20), None, None, 1_000_000_000, 20)
    loop = RealWujiControlLoop(hand, bridge, glove_driver=glove)
    candidate = bridge.glove_to_hand2(_joints(0.1))[0].positions_rad
    clock = iter((1_000_000_000, 1_000_000_000, 1_100_000_001))
    monkeypatch.setattr(hardware_control.time, "monotonic_ns", lambda: next(clock))
    with pytest.raises(TimeoutError):
        loop.step(inference_positions_rad=candidate)
    assert hand.commands == glove.positions == []
    assert not loop._has_command


def test_nonfinite_inference_does_not_send_either_side(bridge):
    hand = HandBoundary(np.zeros(20))
    glove = GloveBoundary(_joints(0.0))
    loop = RealWujiControlLoop(hand, bridge, glove_driver=glove)
    with pytest.raises(ValueError):
        loop.step(inference_positions_rad=np.full(20, np.nan), now_ns=100)
    assert hand.commands == glove.positions == []


def test_unconverged_real_inverse_does_not_send_either_side(bridge):
    hand = HandBoundary(np.zeros(20))
    glove = GloveBoundary(_joints(0.0))
    loop = RealWujiControlLoop(hand, bridge, glove_driver=glove)
    candidate = bridge.glove_to_hand2(_joints(0.55))[0].positions_rad
    # A real one-evaluation solve cannot converge from the neutral initial pose.
    bridge.inverse.max_nfev = 1
    with pytest.raises(ValueError, match="逆解未收敛"):
        loop.step(inference_positions_rad=candidate, now_ns=100)
    assert hand.commands == glove.positions == []
    assert not loop._has_command


def test_glove_hardware_preflight_failure_prevents_hand2_send(bridge):
    hand = HandBoundary(np.zeros(20))
    glove = GloveBoundary(_joints(0.0))
    glove.validation_error = RuntimeError("手套标定门控拒绝")
    loop = RealWujiControlLoop(hand, bridge, glove_driver=glove)
    candidate = bridge.glove_to_hand2(_joints(0.1))[0].positions_rad
    with pytest.raises(RuntimeError, match="标定门控"):
        loop.step(
            inference_positions_rad=candidate,
            now_ns=100,
        )
    assert hand.commands == glove.positions == []
    assert not loop._has_command


def test_startup_glove_ramp_begins_at_post_home_pose_without_resetting_hand_anchor(bridge):
    hand = HandBoundary(np.zeros(20))
    glove = GloveBoundary(_joints(0.25))
    events = []
    hand.events = glove.events = events
    loop = RealWujiControlLoop(hand, bridge, glove_driver=glove)
    previous_target = bridge.glove_to_hand2(_joints(0.05))[0].positions_rad
    previous = loop.step(inference_positions_rad=previous_target, now_ns=100)
    loop.enable(now_ns=200)
    assert events == ["approve_hand", "home_glove", "prepare_glove", "enable_hand", "read_glove"]
    np.testing.assert_array_equal(loop._last_command, previous.sent_positions_rad)
    assert loop._has_command
    glove_anchor = glove.pose.vector()
    motor_anchor = np.asarray([glove._feedback.motor_positions[name] for name in glove.motor_names])
    np.testing.assert_allclose(glove_anchor, np.zeros(12), atol=1e-7, rtol=0)
    candidate = bridge.glove_to_hand2(_joints(0.55))[0].positions_rad
    first = loop.step(inference_positions_rad=candidate, now_ns=200)
    np.testing.assert_allclose(first.shadow_glove_positions_rad, glove_anchor, atol=1e-7, rtol=0)
    mid = loop.step(
        inference_positions_rad=candidate, now_ns=250_000_200,
    )
    end = loop.step(
        inference_positions_rad=candidate, now_ns=500_000_200,
    )
    mid_pose = glove.positions[-2][0]
    inverse_m = 2.0 * glove.motor_vector(mid_pose) - motor_anchor
    inverse_pose = glove.position_from_motors(
        dict(zip(glove.motor_names, inverse_m)), timestamp_ns=mid.timestamp_ns,
    )
    _assert_final_task_projection(bridge, inverse_pose.vector(), mid.sent_positions_rad)
    _assert_final_task_projection(bridge, end.shadow_glove_positions_rad, end.sent_positions_rad)
    commands = glove.positions[1:]
    assert len(commands) == 3
    for command, tick in zip(commands, (first, mid, end)):
        pose = command[0]
        np.testing.assert_array_equal(pose.vector(), tick.shadow_glove_positions_rad)
    # Home/prepare and Motor's own acquisition must finish before Wuji enable.
    assert events[:4] == ["approve_hand", "home_glove", "prepare_glove", "enable_hand"]


def test_mode_ramps_start_after_glove_enable_and_feedback_preparation(bridge, monkeypatch):
    hand = HandBoundary(np.zeros(20))
    glove = GloveBoundary(_joints(0.25))
    loop = RealWujiControlLoop(
        hand, bridge, glove_driver=glove, initial_mode=ControlMode.GLOVE_TAKEOVER,
    )
    clock = {"now": 100}
    enable = glove.enable
    read_state = glove.read_state

    def delayed_enable():
        enable()
        clock["now"] = 900

    def delayed_feedback():
        state = read_state()
        clock["now"] = 1000
        return state

    monkeypatch.setattr(glove, "enable", delayed_enable)
    monkeypatch.setattr(glove, "read_state", delayed_feedback)
    monkeypatch.setattr(hardware_control.time, "monotonic_ns", lambda: clock["now"])
    loop.request_mode(ControlMode.INFERENCE, duration_s=1e-6)
    candidate = bridge.glove_to_hand2(_joints(0.55))[0].positions_rad
    tick = loop.step(inference_positions_rad=candidate, now_ns=1000)
    np.testing.assert_array_equal(tick.sent_positions_rad, hand.state.positions_rad)
    np.testing.assert_allclose(tick.shadow_glove_positions_rad, glove.pose.vector(), atol=1e-7, rtol=0)


def test_takeover_cancels_active_glove_ramp_immediately(bridge):
    loop, hand, glove, anchor = _switching_loop(bridge)
    candidate = bridge.glove_to_hand2(_joints(0.55))[0].positions_rad
    tick = loop.step(inference_positions_rad=candidate, now_ns=200)
    np.testing.assert_allclose(tick.shadow_glove_positions_rad, glove.pose.vector(), atol=1e-7, rtol=0)
    loop.request_mode(ControlMode.GLOVE_TAKEOVER, now_ns=300, duration_s=1e-6)
    assert glove.disable_count == 1
    loop.request_mode(ControlMode.GLOVE_TAKEOVER, now_ns=400, duration_s=1.0)
    assert glove.disable_count == 1
    start = loop.step(now_ns=300)
    np.testing.assert_array_equal(start.sent_positions_rad, anchor)
    loop.step(now_ns=1300)
    assert len(glove.positions) == 1
    assert not loop._glove_handover.in_handover
    assert loop.mode is ControlMode.GLOVE_TAKEOVER


@pytest.mark.parametrize("mode_request", [
    {"mode": "inference"},
    {"mode": ControlMode.INFERENCE, "duration_s": np.nan},
    {"mode": ControlMode.INFERENCE, "duration_s": -0.1},
    {"mode": ControlMode.INFERENCE, "now_ns": -1},
])
def test_invalid_mode_request_does_not_acquire_hardware(bridge, mode_request):
    hand = HandBoundary(np.zeros(20))
    glove = GloveBoundary(_joints(0.25))
    loop = RealWujiControlLoop(
        hand, bridge, glove_driver=glove, initial_mode=ControlMode.GLOVE_TAKEOVER,
    )
    with pytest.raises(ValueError):
        loop.request_mode(**mode_request)
    assert hand.events == glove.events == []
    assert hand.commands == glove.positions == []


def test_hand2_preflight_failure_prevents_glove_output(bridge):
    hand = HandBoundary(np.zeros(20))
    hand.validation_error = ValueError("Hand2关节限位拒绝")
    glove = GloveBoundary(_joints(0.0))
    loop = RealWujiControlLoop(hand, bridge, glove_driver=glove)
    candidate = bridge.glove_to_hand2(_joints(0.1))[0].positions_rad
    with pytest.raises(ValueError, match="关节限位"):
        loop.step(inference_positions_rad=candidate, now_ns=100)
    assert hand.commands == glove.positions == []
    assert not loop._has_command


def test_model_motor_transition_never_reinverts_glove_joint_targets(bridge, monkeypatch):
    loop, hand, glove, _ = _switching_loop(bridge)
    candidate = bridge.glove_to_hand2(_joints(0.55))[0].positions_rad

    def reject_joint_inverse(*args, **kwargs):
        raise AssertionError("Motor轨迹不能把真实q12重新送入闭链逆解")

    monkeypatch.setattr(glove.model.kinematics, "evaluate", reject_joint_inverse)
    for now_ns in (200, 700, 1200):
        tick = loop.step(inference_positions_rad=candidate, now_ns=now_ns)
        pose, _ = glove.positions[-1]
        np.testing.assert_array_equal(pose.vector(), tick.shadow_glove_positions_rad)
        assert len(glove.position_tree(pose)) == 68
    # 确认本回归实际经过不等于指节角的电机坐标，而不是q/m身份映射。
    assert not np.allclose(glove.motor_vector(pose), pose.vector())
    assert len(hand.commands) == 4
