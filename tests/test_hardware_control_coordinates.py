"""Offline control-boundary regressions using the shared canonical geometry and IK."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from motorhand_teleop.bidirectional import (
    BidirectionalRetargeter,
    ControlMode,
    GloveInverseSolver,
    GloveTaskModel,
    Hand2TaskModel,
)
from motorhand_teleop.domain import GLOVE_JOINTS, GlovePose, WujiState
from motorhand_teleop.real_hand import control as hardware_control
from motorhand_teleop.real_hand.control import RealWujiControlLoop
from motorhand_teleop.landmarks import Ve33LandmarkSource
from motorhand_teleop.retargeting import (
    GloveWujiRetargetPipeline,
    default_retarget_config,
)
from motorhand_teleop.ve33_adapter import Ve33GloveModel


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
            received_ns=self.pose.timestamp_ns,
            online_ids=tuple(range(1, 13)),
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


def test_playback_follows_measured_hand_and_blends_from_actual(bridge, monkeypatch):
    actual = bridge.glove_to_hand2(_joints(0.05))[0].positions_rad
    target = bridge.glove_to_hand2(_joints(0.3))[0].positions_rad
    hand = HandBoundary(actual)
    glove = GloveBoundary(_joints(0.0))
    loop = RealWujiControlLoop(
        hand, bridge, glove_driver=glove, glove_follow_actual=True,
        startup_transition_s=1e-6,
    )
    sources = []
    inverse = bridge.hand2_to_glove

    def capture(q, **kwargs):
        sources.append(q.copy())
        return inverse(q, **kwargs)

    monkeypatch.setattr(bridge, "hand2_to_glove", capture)
    first = loop.step(inference_positions_rad=target, now_ns=100)
    second = loop.step(inference_positions_rad=target, now_ns=1100)
    np.testing.assert_allclose(first.sent_positions_rad, actual)
    np.testing.assert_allclose(second.sent_positions_rad, target)
    np.testing.assert_allclose(sources[0], actual)
    np.testing.assert_allclose(sources[1], actual)
    assert not np.allclose(second.sent_positions_rad, sources[1])


def test_async_playback_uses_fresh_cached_result_without_renewing_deadline(bridge):
    """同一真实帧未到期时持续推进，不能借用较新的反馈时间续期。"""
    first_hand = bridge.glove_to_hand2(_joints(0.05))[0].positions_rad
    next_hand = bridge.glove_to_hand2(_joints(0.12))[0].positions_rad
    recorded = bridge.glove_to_hand2(_joints(0.3))[0].positions_rad
    hand = HandBoundary(first_hand)
    glove = GloveBoundary(_joints(0.0))
    hand.state = WujiState(first_hand, None, None, 1_000_000_000, 20)

    class Worker:
        def __init__(self):
            self.pending = []
            self.result = None

        def submit(self, positions_rad, received_ns):
            self.pending.append((positions_rad.copy(), received_ns))
            return len(self.pending)

        def latest(self):
            return self.result

    worker = Worker()
    loop = RealWujiControlLoop(
        hand, bridge, glove_driver=glove, glove_follow_actual=True,
        startup_transition_s=1e-6, inverse_worker=worker,
    )
    first = loop.step(inference_positions_rad=recorded, now_ns=1_000_000_000)
    assert not first.inverse_fresh and not glove.positions
    inverse, _ = bridge.hand2_to_glove(first_hand, timestamp_ns=1_000_000_000)
    worker.result = SimpleNamespace(
        request_id=1, received_ns=1_000_000_000,
        glove_positions_rad=inverse.glove_positions_rad,
        tree_positions_rad=inverse.tree_positions_rad, success=True, error=None,
    )
    glove.pose = glove.model.to_glove_pose(glove._feedback, timestamp_ns=1_020_000_000)
    hand.state = WujiState(next_hand, None, None, 1_020_000_000, 20)
    resumed = loop.step(inference_positions_rad=recorded, now_ns=1_020_000_000)
    assert resumed.inverse_fresh
    hand.state = WujiState(next_hand, None, None, 1_040_000_000, 20)
    cached = loop.step(inference_positions_rad=recorded, now_ns=1_040_000_000)
    assert cached.inverse_fresh
    np.testing.assert_allclose(cached.sent_positions_rad, recorded)
    assert glove.positions[-1][1] <= 1_100_000_000
    hand.state = WujiState(next_hand, None, None, 1_070_000_000, 20)
    waiting = loop.step(inference_positions_rad=recorded, now_ns=1_070_000_000)
    assert not waiting.inverse_fresh
    np.testing.assert_array_equal(waiting.sent_positions_rad, cached.sent_positions_rad)
    assert len(glove.positions) == 2


def test_async_wait_does_not_skip_startup_handover(bridge):
    actual = bridge.glove_to_hand2(_joints(0.05))[0].positions_rad
    target = bridge.glove_to_hand2(_joints(0.4))[0].positions_rad
    hand, glove = HandBoundary(actual), GloveBoundary(_joints(0.0))
    hand.state = WujiState(actual, None, None, 1_000_000_000, 20)

    class Worker:
        def __init__(self):
            self.sequence = 0
            self.result = None

        def submit(self, _positions, _stamp):
            self.sequence += 1
            return self.sequence

        def latest(self):
            return self.result

    worker = Worker()
    loop = RealWujiControlLoop(
        hand, bridge, glove_driver=glove, glove_follow_actual=True,
        startup_transition_s=1.0, inverse_worker=worker,
    )
    first = loop.step(inference_positions_rad=target, now_ns=1_000_000_000)
    for index in range(1, 25):
        stamp = 1_000_000_000 + index * 20_000_000
        hand.state = WujiState(actual, None, None, stamp, 20)
        tick = loop.step(inference_positions_rad=target, now_ns=stamp)
        np.testing.assert_array_equal(tick.sent_positions_rad, first.sent_positions_rad)
    inverse, _ = bridge.hand2_to_glove(actual, timestamp_ns=1_480_000_000)
    worker.result = SimpleNamespace(
        request_id=worker.sequence, received_ns=1_480_000_000,
        glove_positions_rad=inverse.glove_positions_rad,
        tree_positions_rad=inverse.tree_positions_rad, success=True, error=None,
    )
    # 等待期间人手可能移动：恢复时应从当下真实电机位置重新平滑接入。
    glove._feedback = glove.model.evaluate_raw(_joints(0.1))
    glove.pose = glove.model.to_glove_pose(glove._feedback, timestamp_ns=1_500_000_000)
    hand.state = WujiState(actual, None, None, 1_500_000_000, 20)
    resumed = loop.step(inference_positions_rad=target, now_ns=1_500_000_000)
    assert np.max(np.abs(resumed.sent_positions_rad - first.sent_positions_rad)) < 1e-3
    np.testing.assert_allclose(resumed.shadow_glove_positions_rad, glove.pose.vector(), atol=1e-7)


def test_async_playback_stops_advancing_if_inverse_remains_late(bridge):
    actual = bridge.glove_to_hand2(_joints(0.05))[0].positions_rad
    target = bridge.glove_to_hand2(_joints(0.3))[0].positions_rad
    hand = HandBoundary(actual)
    glove = GloveBoundary(_joints(0.0))
    hand.state = WujiState(actual, None, None, 1_000_000_000, 20)

    class LateWorker:
        def submit(self, _positions, _received_ns):
            return 1

        def latest(self):
            return None

    loop = RealWujiControlLoop(
        hand, bridge, glove_driver=glove, glove_follow_actual=True,
        startup_transition_s=1e-6, inverse_worker=LateWorker(),
    )
    first = loop.step(inference_positions_rad=target, now_ns=1_000_000_000)
    hand.state = WujiState(actual, None, None, 1_080_000_000, 20)
    waiting = loop.step(inference_positions_rad=target, now_ns=1_080_000_000)
    np.testing.assert_array_equal(waiting.sent_positions_rad, first.sent_positions_rad)
    assert not waiting.inverse_fresh and len(glove.positions) == 0
    hand.state = WujiState(actual, None, None, 1_180_000_000, 20)
    still_waiting = loop.step(inference_positions_rad=target, now_ns=1_180_000_000)
    np.testing.assert_array_equal(still_waiting.sent_positions_rad, first.sent_positions_rad)
    assert len(glove.positions) == 0
    hand.state = WujiState(actual, None, None, 3_080_000_000, 20)
    with pytest.raises(TimeoutError, match="等待新鲜手套逆解"):
        loop.step(inference_positions_rad=target, now_ns=3_080_000_000)
    assert len(hand.commands) == 3 and len(glove.positions) == 0


def test_async_playback_rejects_failed_inverse_before_both_outputs(bridge):
    actual = bridge.glove_to_hand2(_joints(0.05))[0].positions_rad
    hand, glove = HandBoundary(actual), GloveBoundary(_joints(0.0))
    hand.state = WujiState(actual, None, None, 1_000_000_000, 20)
    stopped = []
    hand.emergency_stop = lambda: stopped.append("hand")
    glove.emergency_stop = lambda: stopped.append("glove")

    class FailingWorker:
        def submit(self, _positions, _received_ns):
            return 1

        def latest(self):
            return SimpleNamespace(success=False, error="数值求解失败")

    loop = RealWujiControlLoop(
        hand, bridge, glove_driver=glove, glove_follow_actual=True,
        inverse_worker=FailingWorker(),
    )
    loop.enable(now_ns=1_000_000_000)
    with pytest.raises(ValueError, match="数值求解失败"):
        loop.step(inference_positions_rad=actual, now_ns=1_000_000_000)
    assert len(hand.commands) == len(glove.positions) == 0
    assert stopped == ["hand", "glove"]


def test_offline_realtime_playback_rate_with_real_inverse_worker(bridge):
    """显式启用的墙钟验收：真实CAD/优化器/worker，数值边界不连接硬件。"""
    import os
    import time
    from motorhand_teleop.real_hand.inverse_worker import InverseWorker
    from motorhand_teleop.real_hand.motor_hand import EncosGloveHardwareDriver
    from motorhand_teleop.real_hand.playback import PlaybackClock

    if os.environ.get("WUJI_OFFLINE_REALTIME") != "1":
        pytest.skip("设置 WUJI_OFFLINE_REALTIME=1 执行墙钟性能验收")
    actual = bridge.glove_to_hand2(_joints(0.05))[0].positions_rad
    end = bridge.glove_to_hand2(_joints(0.3))[0].positions_rad
    hand, glove = HandBoundary(actual), GloveBoundary(_joints(0.0))
    limits = dict(EncosGloveHardwareDriver().model_motor_limits_rad)
    glove.model_motor_limits_rad = limits
    measured = actual.copy()
    original_read_glove = glove.read_state
    original_send_hand = hand.send_target
    original_send_glove = glove.send_position

    def read_hand():
        return WujiState(measured.copy(), None, None, time.monotonic_ns(), 20)

    def read_glove():
        glove.pose = glove.model.to_glove_pose(glove._feedback, timestamp_ns=time.monotonic_ns())
        return original_read_glove()

    def send_hand(target, *, deadline_ns):
        assert time.monotonic_ns() < deadline_ns
        original_send_hand(target, deadline_ns=deadline_ns)
        measured[:] += 0.3 * (target.positions_rad - measured)

    def send_glove(pose, *, deadline_ns):
        assert time.monotonic_ns() < deadline_ns
        original_send_glove(pose, deadline_ns=deadline_ns)
        target_motors = glove.motor_vector(pose)
        names = glove.motor_names
        current = np.asarray([glove._feedback.motor_positions[name] for name in names])
        glove._feedback = glove.model.kinematics.evaluate_motors(
            dict(zip(names, current + 0.3 * (target_motors - current))),
            seed=glove._feedback,
        )

    hand.read_state, hand.send_target = read_hand, send_hand
    glove.read_state, glove.send_position = read_glove, send_glove
    worker = InverseWorker("right", "right", ROOT / "assets/hand_motor_ve33",
                           ROOT / "config/retargeting/official/right.yaml", limits)
    loop = RealWujiControlLoop(
        hand, bridge, glove_driver=glove, glove_follow_actual=True,
        startup_transition_s=0.5, inverse_worker=worker,
    )
    try:
        worker.start(actual, time.monotonic_ns())
        loop.enable()
        first = loop.step(inference_positions_rad=actual)
        start = time.monotonic_ns()
        clock = PlaybackClock(duration_ns=3_000_000_000, transition_ns=500_000_000,
                              start_ns=start)
        ready = first.inverse_fresh
        waits = cycles = 0
        max_cycle_ns = 0
        while time.monotonic_ns() - start < 4_000_000_000:
            cycle = time.monotonic_ns()
            position = clock.position(cycle, ready=ready)
            target = actual + (end - actual) * (position / clock.duration_ns)
            tick = loop.step(inference_positions_rad=target)
            ready = tick.inverse_fresh
            waits += not ready
            cycles += 1
            max_cycle_ns = max(max_cycle_ns, time.monotonic_ns() - cycle)
            remaining = cycle + 20_000_000 - time.monotonic_ns()
            if remaining > 0:
                time.sleep(remaining / 1e9)
        elapsed = (time.monotonic_ns() - start) / 1e9
        position = clock.position(time.monotonic_ns(), ready=ready)
        print(f"offline elapsed={elapsed:.3f}s trajectory={position / 1e9:.3f}s "
              f"cycles={cycles} inverse_wait={waits} max_cycle={max_cycle_ns / 1e6:.1f}ms")
        assert position >= 2_850_000_000, "墙钟4秒内未接近1倍速（含0.5秒过渡）"
        assert len(glove.positions) > 0
    finally:
        worker.close()


def test_inference_source_deadline_is_not_renewed_and_takeover_ignores_it(bridge):
    actual = bridge.glove_to_hand2(_joints(0.05))[0].positions_rad
    hand = HandBoundary(actual)
    loop = RealWujiControlLoop(hand, bridge, compute_shadow_glove=False)
    loop.step(inference_positions_rad=actual, inference_source_ns=950_000_000,
              now_ns=1_000_000_000)
    assert hand.commands[-1][1] == 1_050_000_000
    loop.request_mode(ControlMode.GLOVE_TAKEOVER, now_ns=1_000_000_000, duration_s=0)
    loop.step(glove_joints=_joints(0.05), inference_source_ns=1,
              now_ns=1_010_000_000)
    assert hand.commands[-1][1] == 1_110_000_000


@pytest.mark.parametrize("source, error", [
    (True, ValueError), (-1, ValueError), (1.0, ValueError),
    (1_000_000_001, TimeoutError), (900_000_000, TimeoutError),
])
def test_inference_source_rejects_invalid_future_and_expired(bridge, source, error):
    actual = bridge.glove_to_hand2(_joints(0.05))[0].positions_rad
    hand = HandBoundary(actual)
    loop = RealWujiControlLoop(hand, bridge, compute_shadow_glove=False)
    with pytest.raises(error):
        loop.step(inference_positions_rad=actual, inference_source_ns=source,
                  now_ns=1_000_000_000)
    assert not hand.commands


def test_standalone_playback_skips_inverse_and_anchors_start(bridge, monkeypatch):
    actual = np.full(20, 0.1)
    target = np.full(20, 0.4)
    hand = HandBoundary(actual)
    loop = RealWujiControlLoop(
        hand, bridge, compute_shadow_glove=False, startup_transition_s=1e-6,
    )

    def reject(*args, **kwargs):
        raise AssertionError("standalone playback must not solve glove inverse")

    monkeypatch.setattr(bridge, "hand2_to_glove", reject)
    first = loop.step(inference_positions_rad=target, now_ns=100)
    last = loop.step(inference_positions_rad=target, now_ns=1100)
    np.testing.assert_allclose(first.sent_positions_rad, actual)
    np.testing.assert_allclose(last.sent_positions_rad, target)
    assert first.shadow_glove_positions_rad is None


def test_playback_force_switch_uses_prewarmed_geometry_and_releases_on_failure(
    bridge, monkeypatch,
):
    from motorhand_teleop.real_hand.force_feedback import ForceFeedbackCommand

    actual = bridge.glove_to_hand2(_joints(0.05))[0].positions_rad
    target = bridge.glove_to_hand2(_joints(0.2))[0].positions_rad
    hand = HandBoundary(actual)
    glove = GloveBoundary(_joints(0.0))
    events = hand.events = glove.events = []
    stopped = []
    monkeypatch.setattr(hand, "emergency_stop", lambda: stopped.append("hand"), raising=False)
    monkeypatch.setattr(glove, "emergency_stop", lambda: stopped.append("glove"), raising=False)
    monkeypatch.setattr(glove, "enable_force_feedback", lambda: events.append("force_enable"), raising=False)
    monkeypatch.setattr(glove, "validate_force_current", lambda current: None, raising=False)
    currents = []
    monkeypatch.setattr(
        glove, "send_force_current",
        lambda current, **kwargs: currents.append(current) or current, raising=False,
    )

    class Force:
        def validate_inputs(self, hand_state, glove_state, *, now_ns):
            events.append("feedback_preflight")

        def prepare(self, hand_state, glove_state, *, now_ns):
            events.append("geometry_prepare")

        def reset(self, *, now_ns):
            events.append("force_reset")

        def compute(self, hand_state, glove_state, **kwargs):
            events.append("compute")
            return ForceFeedbackCommand(
                (0.0,) * 12, None, None, "test", 1,
                projection_status="refreshing_age",
            )

        def acknowledge(self, current):
            pass

    loop = RealWujiControlLoop(
        hand, bridge, glove_driver=glove, force_feedback=Force(),
        glove_follow_actual=True, startup_transition_s=1e-6,
    )
    loop.enable(now_ns=10)
    assert events.index("geometry_prepare") < events.index("enable_hand")
    loop.step(inference_positions_rad=target, now_ns=100)
    prior = loop.step(inference_positions_rad=target, now_ns=1100)
    loop.request_mode(ControlMode.GLOVE_TAKEOVER, now_ns=1200, duration_s=1e-6)
    assert events.count("geometry_prepare") == 1
    before = len(glove.positions)
    takeover = loop.step(now_ns=1200)
    np.testing.assert_allclose(takeover.sent_positions_rad, prior.sent_positions_rad)
    assert len(glove.positions) == before
    assert takeover.force_feedback.projection_status == "refreshing_age"
    assert currents[-1] == (0.0,) * 12
    loop.request_mode(ControlMode.INFERENCE, now_ns=1300, duration_s=1e-6)
    resumed = loop.step(inference_positions_rad=target, now_ns=1300)
    np.testing.assert_allclose(resumed.sent_positions_rad, takeover.sent_positions_rad)
    hand.validation_error = RuntimeError("feedback lost")
    with pytest.raises(RuntimeError, match="feedback lost"):
        loop.step(inference_positions_rad=target, now_ns=1400)
    assert stopped == ["hand", "glove"]


def test_standalone_playback_disable_releases_and_reanchors(bridge, monkeypatch):
    hand = HandBoundary(np.full(20, 0.1))
    stopped = []
    monkeypatch.setattr(hand, "disable", lambda: stopped.append("disable"), raising=False)
    loop = RealWujiControlLoop(
        hand, bridge, compute_shadow_glove=False, startup_transition_s=1e-6,
    )
    loop.enable(now_ns=10)
    loop.step(inference_positions_rad=np.full(20, 0.4), now_ns=100)
    loop.step(inference_positions_rad=np.full(20, 0.4), now_ns=1100)
    loop.disable()
    assert stopped == ["disable"]
    hand.state = WujiState(np.full(20, 0.2), None, None, 7, 20)
    loop.enable(now_ns=1200)
    restarted = loop.step(inference_positions_rad=np.full(20, 0.4), now_ns=1300)
    np.testing.assert_allclose(restarted.sent_positions_rad, 0.2)


def test_playback_prewarms_inverse_before_enabling_and_resamples(bridge, monkeypatch):
    first = bridge.glove_to_hand2(_joints(0.05))[0].positions_rad
    measured = bridge.glove_to_hand2(_joints(0.12))[0].positions_rad
    hand = HandBoundary(first)
    glove = GloveBoundary(_joints(0.0))
    events = hand.events = glove.events = []
    inputs = []
    solve = bridge.hand2_to_glove

    def inverse(q, **kwargs):
        events.append("inverse")
        inputs.append((q.copy(), kwargs["timestamp_ns"]))
        return solve(q, **kwargs)

    monkeypatch.setattr(bridge, "hand2_to_glove", inverse)
    loop = RealWujiControlLoop(
        hand, bridge, glove_driver=glove, glove_follow_actual=True,
        startup_transition_s=1e-6,
    )
    loop.enable(now_ns=10)
    assert events.index("inverse") < events.index("enable_hand")
    np.testing.assert_array_equal(inputs[0][0], first)
    assert inputs[0][1] == 7
    hand.state = WujiState(measured, None, None, 20, 20)
    loop.step(inference_positions_rad=first, now_ns=20)
    assert len(inputs) == 2
    np.testing.assert_array_equal(inputs[1][0], measured)


def test_failed_playback_inverse_prewarm_never_enables_hand(bridge, monkeypatch):
    hand = HandBoundary(np.zeros(20))
    glove = GloveBoundary(_joints(0.0))
    stopped = []
    monkeypatch.setattr(hand, "emergency_stop", lambda: stopped.append("hand"), raising=False)
    monkeypatch.setattr(glove, "emergency_stop", lambda: stopped.append("glove"), raising=False)
    monkeypatch.setattr(
        bridge, "hand2_to_glove",
        lambda *args, **kwargs: (SimpleNamespace(success=False), None),
    )
    loop = RealWujiControlLoop(hand, bridge, glove_driver=glove, glove_follow_actual=True)
    with pytest.raises(ValueError, match="预热未收敛"):
        loop.enable(now_ns=10)
    assert "enable_hand" not in hand.events
    assert stopped == ["hand", "glove"]


def _inference_session_with_simulated_devices(bridge, monkeypatch):
    from motorhand_teleop.real_hand import inference

    clock = [1_000_000_000]
    actual = bridge.glove_to_hand2(_joints(0.05))[0].positions_rad
    hand = HandBoundary(actual)
    glove = GloveBoundary(_joints(0.0))
    stopped = []
    hand.connect = lambda **_: {"serial_number": "offline"}
    hand.close = lambda: None
    hand.disable = lambda: hand.events.append("disable_hand")
    hand.emergency_stop = lambda: stopped.append("hand")
    hand.read_state = lambda: WujiState(actual, None, None, clock[0], 20)
    glove.connect = lambda: {"model": "offline"}
    glove.close = lambda: None
    glove.emergency_stop = lambda: stopped.append("glove")
    read_glove = glove.read_state

    def fresh_glove():
        glove.pose = glove.model.to_glove_pose(glove._feedback, timestamp_ns=clock[0])
        return read_glove()

    glove.read_state = fresh_glove
    monkeypatch.setattr(hardware_control, "time", SimpleNamespace(monotonic_ns=lambda: clock[0]))
    monkeypatch.setattr(inference, "time", SimpleNamespace(monotonic_ns=lambda: clock[0]))
    loop = RealWujiControlLoop(
        hand, bridge, glove_driver=glove, startup_transition_s=0.5,
    )
    session = inference.InferenceSession(loop, serial_number="offline")
    return session, hand, glove, clock, stopped, actual


def test_inference_session_switches_single_owner_and_releases(bridge, monkeypatch):
    from motorhand_teleop.bidirectional import ControlMode

    session, hand, glove, clock, _, actual = _inference_session_with_simulated_devices(
        bridge, monkeypatch,
    )
    session.connect()
    assert "enable_hand" not in hand.events and "home_glove" not in glove.events
    session.start()
    first = session.step(actual, produced_at_ns=clock[0])
    np.testing.assert_allclose(first.sent_positions_rad, actual, atol=1e-10)
    clock[0] += 600_000_000
    session.step(actual, produced_at_ns=clock[0])
    before = len(glove.positions)
    session.take_over()
    assert session.mode is ControlMode.GLOVE_TAKEOVER
    session.step()
    assert len(glove.positions) == before
    session.resume_inference()
    assert session.mode is ControlMode.INFERENCE
    clock[0] += 600_000_000
    session.step(actual, produced_at_ns=clock[0])
    assert len(glove.positions) > before
    session.close()
    assert "disable_hand" in hand.events and glove.disable_count > 0


def test_inference_session_stops_both_sides_on_expired_model_output(bridge, monkeypatch):
    session, hand, _, clock, stopped, actual = _inference_session_with_simulated_devices(
        bridge, monkeypatch,
    )
    session.connect()
    session.start()
    with pytest.raises(TimeoutError):
        session.step(actual, produced_at_ns=clock[0] - 100_000_000)
    assert not hand.commands and stopped == ["hand", "glove"]
    session.close()


def test_public_inference_factory_builds_read_only_session():
    from motorhand_teleop.real_hand import create_inference_session
    from motorhand_teleop.real_hand.force_linear import MaxTorqueLinearConfig

    config = MaxTorqueLinearConfig.from_json(ROOT / "config/feedback/max-torque-linear-right.json")
    session = create_inference_session(
        serial_number=config.wuji_serial_number, address="127.0.0.1:7447",
    )
    try:
        assert not session._loop.driver.connected
        with pytest.raises(RuntimeError, match="尚未启动"):
            session.step(np.zeros(20))
    finally:
        session.close()
