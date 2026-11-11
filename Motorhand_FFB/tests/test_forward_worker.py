"""独立正向求解保留官方重定向结果与原始期限。"""

import time

import numpy as np
import pytest

from motorhand_teleop.domain import GLOVE_JOINTS
from motorhand_teleop.real_hand import forward_worker
from motorhand_teleop.real_hand.forward_worker import ForwardRetargetWorker
from motorhand_teleop.real_hand.runtime import create_retargeter


def _delayed_worker(connection, settings):
    connection.send(("ready",))
    try:
        if connection.recv() is not None:
            time.sleep(0.5)
    finally:
        connection.close()


def test_worker_matches_shared_retarget_framework_across_motion():
    reference = create_retargeter()
    worker = ForwardRetargetWorker(hand="right", glove_hand="right")
    worker.start()
    try:
        for bend in (0.0, 0.04, 0.1, 0.16, 0.08, 0.0):
            joints = dict.fromkeys(GLOVE_JOINTS, 0.0)
            joints["index_pip"] = bend
            joints["middle_pip"] = bend * 0.5
            stamp = time.monotonic_ns()
            expected, _ = reference.glove_to_hand2(joints, timestamp_ns=stamp)
            actual = worker.solve(joints, source_ns=stamp, deadline_ns=time.monotonic_ns() + 2_000_000_000)
            np.testing.assert_allclose(actual.positions_rad, expected.positions_rad, atol=2e-6, rtol=0)
            assert actual.diagnostics["retarget_backend"] == "wuji_retargeting"
    finally:
        worker.close()


def test_expired_request_is_rejected_without_changing_next_solution():
    reference = create_retargeter()
    worker = ForwardRetargetWorker(hand="right", glove_hand="right")
    worker.start()
    try:
        joints = dict.fromkeys(GLOVE_JOINTS, 0.0)
        stamp = time.monotonic_ns()
        with pytest.raises(TimeoutError):
            worker.solve(joints, source_ns=stamp, deadline_ns=stamp - 1)
        expected, _ = reference.glove_to_hand2(joints, timestamp_ns=stamp)
        actual = worker.solve(joints, source_ns=stamp, deadline_ns=time.monotonic_ns() + 2_000_000_000)
        np.testing.assert_allclose(actual.positions_rad, expected.positions_rad, atol=2e-6, rtol=0)
    finally:
        worker.close()


def test_inflight_timeout_latches_worker_without_reusing_old_result(monkeypatch):
    monkeypatch.setattr(forward_worker, "_run_worker", _delayed_worker)
    worker = ForwardRetargetWorker(hand="right", glove_hand="right")
    worker.start()
    try:
        stamp = time.monotonic_ns()
        joints = dict.fromkeys(GLOVE_JOINTS, 0.0)
        with pytest.raises(TimeoutError):
            worker.solve(joints, source_ns=stamp, deadline_ns=stamp + 30_000_000)
        with pytest.raises(RuntimeError, match="故障锁存"):
            worker.check_ready()
    finally:
        worker.close()
