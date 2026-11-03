import json
import hashlib
import queue
import threading

import numpy as np
import pytest

from motorhand_teleop.domain import WUJI_JOINTS, WujiState
from motorhand_teleop.real_hand.trajectory import RecordedTrajectory, TrajectoryRecorder


def state(stamp, position, velocity=None, effort=None):
    return WujiState(np.full(20, position), velocity, effort, stamp, 20)


def recorder(path, **kwargs):
    return TrajectoryRecorder(path, hand="right", serial_number="serial-1",
                              firmware_version="firmware-1", **kwargs)


def test_measured_frames_interpolation_and_metadata(tmp_path):
    path = tmp_path / "nested" / "recording.jsonl"
    writer = recorder(path)
    writer.record(state(100, 1, np.full(20, 3), None), np.full(20, 8))
    writer.record(state(100, 99), np.full(20, 77))  # same source frame, not a new time
    writer.record(state(300, 5, None, np.full(20, 2)), np.full(20, 9))
    writer.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["joint_names"] == list(WUJI_JOINTS)
    assert len(rows) == 4
    assert rows[1]["source_ns"] == 100 and rows[1]["elapsed_ns"] == 0
    assert rows[2]["source_ns"] == 300 and rows[2]["elapsed_ns"] == 200
    assert rows[1]["positions_rad"] == [1] * 20
    assert rows[1]["target_positions_rad"] == [8] * 20
    assert rows[1]["velocities_rad_s"] == [3] * 20 and rows[1]["effort_a"] is None
    assert rows[2]["velocities_rad_s"] is None and rows[2]["effort_a"] == [2] * 20
    trajectory = RecordedTrajectory.load(path)
    assert (trajectory.hand, trajectory.serial_number, trajectory.firmware_version,
            trajectory.duration_ns) == ("right", "serial-1", "firmware-1", 200)
    frames = trajectory.positions_rad
    np.testing.assert_allclose(frames, np.array([[1] * 20, [5] * 20]))
    assert not frames.flags.writeable
    with pytest.raises(ValueError):
        frames[0, 0] = 0
    frames.setflags(write=True)
    frames[0, 0] = 0
    assert trajectory.positions_at(0)[0] == 1
    np.testing.assert_allclose(trajectory.positions_at(50), np.full(20, 2))
    np.testing.assert_allclose(trajectory.positions_at(-1), np.full(20, 1))
    np.testing.assert_allclose(trajectory.positions_at(500), np.full(20, 5))
    endpoint = trajectory.positions_at(-1)
    endpoint[:] = 123
    np.testing.assert_allclose(trajectory.positions_at(0), np.full(20, 1))
    endpoint = trajectory.positions_at(500)
    endpoint[:] = 123
    np.testing.assert_allclose(trajectory.positions_at(200), np.full(20, 5))


def test_existing_file_and_aborted_recording(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    writer = recorder(path)
    with pytest.raises(FileExistsError):
        recorder(path)
    writer.record(state(100, 1), np.zeros(20))
    writer.abort()
    with pytest.raises(ValueError):
        RecordedTrajectory.load(path)


def test_out_of_order_invalidates_recording(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    writer = recorder(path)
    writer.record(state(100, 1), np.zeros(20))
    with pytest.raises(ValueError):
        writer.record(state(99, 2), np.zeros(20))
    with pytest.raises(RuntimeError):
        writer.close()
    with pytest.raises(ValueError):
        RecordedTrajectory.load(path)


def test_truncation_tampering_and_invalid_json_rejected(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    writer = recorder(path)
    writer.record(state(100, 1), np.zeros(20))
    writer.record(state(200, 2), np.zeros(20))
    writer.close()
    original = path.read_bytes()
    for damaged in (original[:-1], original.splitlines(keepends=True)[0],
                    original.replace(b'"positions_rad":[1.0', b'"positions_rad":[2.0'),
                    original + b'{}\n', original.replace(b'"source_ns":100', b'"source_ns":NaN')):
        path.write_bytes(damaged)
        with pytest.raises(ValueError):
            RecordedTrajectory.load(path)


@pytest.mark.parametrize("count", [0, 1])
def test_short_recording_cannot_commit(tmp_path, count):
    path = tmp_path / "short.jsonl"
    writer = recorder(path)
    if count:
        writer.record(state(100, 1), np.zeros(20))
    with pytest.raises(RuntimeError):
        writer.close()
    with pytest.raises(ValueError):
        RecordedTrajectory.load(path)


def test_loader_rejects_checksummed_single_frame(tmp_path):
    path = tmp_path / "single.jsonl"
    writer = recorder(path)
    writer.record(state(100, 1), np.zeros(20))
    writer.abort()
    prefix = path.read_bytes()
    footer = {"type": "end", "frame_count": 1, "duration_ns": 0,
              "sha256": hashlib.sha256(prefix).hexdigest()}
    path.write_bytes(prefix + (json.dumps(footer) + "\n").encode())
    with pytest.raises(ValueError):
        RecordedTrajectory.load(path)




def test_queue_overflow_fails_and_never_commits(tmp_path, monkeypatch):
    from pathlib import Path
    opened = threading.Event()
    release = threading.Event()
    original_open = Path.open

    class BlockedFile:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def write(self, data):
            opened.set()
            assert release.wait(5)
            return self.wrapped.write(data)

        def flush(self):
            return self.wrapped.flush()

        def close(self):
            return self.wrapped.close()

    def open_blocked(path, mode="r", *args, **kwargs):
        file = original_open(path, mode, *args, **kwargs)
        return BlockedFile(file) if mode == "xb" else file

    monkeypatch.setattr(Path, "open", open_blocked)
    path = tmp_path / "overflow.jsonl"
    writer = recorder(path, queue_size=1)
    try:
        assert opened.wait(5)
        writer.record(state(100, 1), np.zeros(20))
        with pytest.raises(queue.Full):
            writer.record(state(101, 2), np.zeros(20))
    finally:
        release.set()
        writer.abort()
    with pytest.raises(ValueError):
        RecordedTrajectory.load(path)


@pytest.mark.parametrize("ending,playable", [(KeyboardInterrupt, True), (RuntimeError, False)])
def test_follow_cli_commits_only_on_clean_interrupt(tmp_path, monkeypatch, ending, playable):
    from types import SimpleNamespace
    from motorhand_teleop.real_hand import runtime, motor_hand, preflight, wuji_hand2

    class FakeGlove:
        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

    class FakeHand:
        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

    class FakeRuntime:
        def __init__(self, *args, **kwargs):
            self.identities = {}
            self.calls = 0

        def connect(self, *, serial_number):
            self.identities["wuji_hand2"] = {
                "hand": "right", "serial_number": serial_number, "firmware_version": "fw",
            }

        def inspect(self):
            return {}

        def prepare(self):
            pass

        def start(self):
            pass

        def step(self):
            self.calls += 1
            if self.calls > 2:
                raise ending()
            return SimpleNamespace(
                hand2_state=state(100 * self.calls, 1), output_positions_rad=np.full(20, 7),
                mode=runtime.FollowMode.GLOVE_TO_HAND2, executed=False,
                deadline_ns=200, force_feedback=None,
            )

        def close(self):
            pass

    monkeypatch.setattr(preflight, "select_device", lambda **_: {
        "serial_number": "serial", "address": "",
    })
    monkeypatch.setattr(motor_hand, "EncosGloveHardwareDriver", FakeGlove)
    monkeypatch.setattr(wuji_hand2, "WujiHand2HardwareDriver", FakeHand)
    monkeypatch.setattr(runtime, "PairRuntime", FakeRuntime)
    monkeypatch.setattr(runtime, "create_retargeter", lambda **kwargs: object())
    monkeypatch.setattr(runtime.time, "sleep", lambda _: None)
    path = tmp_path / "cli.jsonl"
    args = ["follow", "--mode", "glove-to-hand2", "--wuji-serial", "serial", "--record", str(path)]
    if playable:
        assert runtime.main(args) == 130
        np.testing.assert_allclose(RecordedTrajectory.load(path).positions_at(0), np.ones(20))
    else:
        with pytest.raises(RuntimeError):
            runtime.main(args)
        with pytest.raises(ValueError):
            RecordedTrajectory.load(path)
