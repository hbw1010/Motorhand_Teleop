"""回放时间与终端接管行为；不连接或使能设备。"""
import os
import termios

import pytest

from wujihand2_force_glove.real_hand.playback import PlaybackClock, TerminalKeys


def test_playback_clock_pauses_for_startup_and_manual_takeover():
    clock = PlaybackClock(duration_ns=2_000_000_000, transition_ns=500_000_000,
                          start_ns=1_000_000_000)
    assert clock.position(1_400_000_000) == 0
    assert clock.position(1_750_000_000) == 250_000_000
    clock.pause(2_000_000_000)
    assert clock.position(4_000_000_000) == 500_000_000
    clock.resume(4_000_000_000)
    assert clock.position(4_400_000_000) == 500_000_000
    assert clock.position(4_750_000_000) == 750_000_000
    assert clock.position(6_000_000_000) == 2_000_000_000


def test_playback_clock_rejects_backwards_time_without_advancing():
    clock = PlaybackClock(duration_ns=1_000_000_000, transition_ns=0, start_ns=10)
    assert clock.position(20) == 10
    with pytest.raises(ValueError):
        clock.position(19)
    assert clock.position(30) == 20


def test_playback_clock_does_not_jump_after_waiting_for_inverse():
    clock = PlaybackClock(duration_ns=10_000_000_000, transition_ns=0, start_ns=0, speed=0.5)
    assert clock.position(1_000_000_000, ready=False) == 0
    assert clock.position(2_000_000_000, ready=False) == 0
    assert clock.position(3_000_000_000, ready=True) == 500_000_000


def test_slow_playback_pauses_and_resumes_with_wall_clock_transitions():
    clock = PlaybackClock(duration_ns=1_000_000_000, transition_ns=500_000_000,
                          start_ns=1_000_000_000, speed=0.05)
    assert clock.position(1_400_000_000) == 0
    assert clock.position(2_500_000_000) == 50_000_000
    clock.pause(3_500_000_000)
    assert clock.position(10_000_000_000) == 100_000_000
    clock.resume(10_000_000_000)
    assert clock.position(10_400_000_000) == 100_000_000
    assert clock.position(11_500_000_000) == 150_000_000


def test_slow_playback_accumulates_subnanosecond_steps_across_pause():
    clock = PlaybackClock(duration_ns=100, transition_ns=0, start_ns=0, speed=0.05)
    for now in range(1, 11):
        assert clock.position(now) == 0
    clock.pause(10)
    clock.resume(100)
    for now in range(101, 111):
        clock.position(now)
    assert clock.position(110) == 1
    for now in range(111, 291):
        clock.position(now)
    assert clock.position(290) == 10


@pytest.mark.parametrize("speed", [0, -0.05, 1.01, float("nan"), float("inf"), -float("inf")])
def test_playback_clock_rejects_invalid_speed(speed):
    with pytest.raises(ValueError, match="回放速度"):
        PlaybackClock(duration_ns=1, transition_ns=0, start_ns=0, speed=speed)


@pytest.mark.parametrize("speed", ["0", "-0.05", "1.01", "nan", "inf", "-inf"])
def test_playback_cli_rejects_invalid_speed_before_loading(speed, capsys):
    from wujihand2_force_glove.real_hand.playback import main

    with pytest.raises(SystemExit) as exc:
        main(["unused.jsonl", f"--speed={speed}"])
    assert exc.value.code == 2
    assert "--speed" in capsys.readouterr().err


def test_playback_cli_speed_defaults_to_one_and_accepts_slow_speed():
    from wujihand2_force_glove.real_hand.playback import _parser

    assert _parser().parse_args(["unused.jsonl"]).speed == 1.0
    assert _parser().parse_args(["unused.jsonl", "--speed", "0.05"]).speed == 0.05


def test_terminal_keys_toggle_and_quit_and_restore_settings():
    master, slave = os.openpty()
    original = termios.tcgetattr(slave)
    try:
        with TerminalKeys(slave) as keys:
            os.write(master, b"tq")
            assert keys.read() == ("toggle", "quit")
        assert termios.tcgetattr(slave) == original
    finally:
        os.close(master)
        os.close(slave)


def test_recorded_motion_replays_and_keyboard_switches_to_contact_feedback(tmp_path, monkeypatch):
    """真实共享求解器＋带滞后数值机构：回放、受阻接管、恢复，不连接实机。"""
    from types import SimpleNamespace

    import numpy as np
    from test_contact_runtime import create_contact_pair, human_joints, set_human
    from wujihand2_force_glove.domain import WujiState
    from wujihand2_force_glove.real_hand import (
        contact_feedback, control, force_feedback, force_linear, inverse_worker,
        motor_hand, playback, runtime, wuji_hand2,
    )
    from wujihand2_force_glove.real_hand.trajectory import RecordedTrajectory, TrajectoryRecorder
    make_bridge = runtime.create_retargeter

    class DeterministicWorker:
        """同一真实逆解的确定性消费者；独立进程另由worker专项测试。"""
        def __init__(self, hand, glove_hand, assets, config, limits):
            self.bridge = make_bridge(hand=hand, glove_hand=glove_hand, asset_root=assets, config_path=config)
            self.bridge.inverse.set_motor_limits(limits)
            self.result = None
            self.sequence = 0

        def start(self, positions, stamp):
            result, _ = self.bridge.hand2_to_glove(positions, timestamp_ns=stamp)
            assert result.success

        def submit(self, positions, stamp):
            self.sequence += 1
            result, _ = self.bridge.hand2_to_glove(positions, timestamp_ns=stamp)
            self.result = inverse_worker.InverseWorkerResult(
                self.sequence, stamp, result.glove_positions_rad,
                dict(result.tree_positions_rad), result.success,
                None if result.success else "真实逆解未收敛",
            )
            return self.sequence

        def latest(self):
            return self.result

        def close(self):
            pass

    pair, hand, glove, clock = create_contact_pair()
    bridge, feedback = pair.retargeter, pair.force_feedback
    path = tmp_path / "recording.jsonl"
    poses = (
        hand.positions.copy(),
        bridge.glove_to_hand2(human_joints(0.1))[0].positions_rad.copy(),
        hand.positions.copy(),
    )
    writer = TrajectoryRecorder(
        path, hand="right", serial_number="explicit-test-SN", firmware_version="approved-test",
    )
    for index, pose in enumerate(poses):
        state = WujiState(pose, np.zeros(20), np.zeros(20), 1_000_000_000 + index * 1_000_000_000, 20)
        writer.record(state, pose)
    writer.close()
    trajectory = RecordedTrajectory.load(path)

    hand.connect = lambda **_: SimpleNamespace(
        hand="right", serial_number="explicit-test-SN", firmware_version="approved-test",
    )
    hand.validate_enable = lambda: None
    hand.emergency_stop = hand.disable
    hand.lower_limits, hand.upper_limits = hand.limits[:, 0], hand.limits[:, 1]
    glove.emergency_stop = glove.disable
    monkeypatch.setattr(inverse_worker, "InverseWorker", DeterministicWorker)
    monkeypatch.setattr(wuji_hand2, "WujiHand2HardwareDriver", lambda **_: hand)
    monkeypatch.setattr(motor_hand, "EncosGloveHardwareDriver", lambda **_: glove)
    monkeypatch.setattr(runtime, "create_retargeter", lambda **_: bridge)
    monkeypatch.setattr(force_feedback, "ForceFeedbackController", lambda *_, **__: feedback)
    monkeypatch.setattr(force_linear.MaxTorqueLinearConfig, "from_json", lambda _: feedback.config)
    monkeypatch.setattr(contact_feedback.ContactFeedbackConfig, "from_json", lambda _: feedback.contact_config)
    monkeypatch.setattr(control, "time", SimpleNamespace(monotonic_ns=clock))
    monkeypatch.setattr(playback, "time", SimpleNamespace(
        monotonic_ns=clock, sleep=lambda seconds: clock.advance(seconds),
    ))

    master, slave = os.openpty()
    original_send = hand.send_target

    def send_and_advance(target, *, deadline_ns):
        original_send(target, deadline_ns=deadline_ns)
        frame = len(hand.commands)
        if frame < 50 or frame >= 100:
            hand.advance_mechanics()
            glove.plant.advance_mechanics()
        if frame == 50:
            # 手套继续屈指，数值Wuji被挡住并出现独立电流反馈。
            set_human(glove, 0.24)
            hand.effort[:] = 0.8
            os.write(master, b"t")
        elif frame == 100:
            os.write(master, b"t")
        elif frame == 220:
            os.write(master, b"q")  # 防止回放时钟故障导致测试无限运行。

    hand.send_target = send_and_advance
    args = playback._parser().parse_args([
        str(path), "--wuji-serial", "explicit-test-SN", "--approved-firmware", "approved-test",
        "--with-glove", "--execute",
    ])
    try:
        with TerminalKeys(slave) as keys:
            assert playback._run(args, trajectory, keys) == 0
    finally:
        os.close(master)
        os.close(slave)

    assert len(hand.commands) > 120
    replayed = np.asarray([command.positions_rad for command, _ in hand.commands[:50]])
    assert np.max(np.ptp(replayed, axis=0)) > 0.005
    assert len(glove.commands) > 50
    assert len(glove.force_torque_commands) > 30
    assert max(np.max(np.abs(torque)) for torque in glove.force_torque_commands) > 0
    assert hand.disabled and glove.disabled and hand.closed and glove.plant.closed
