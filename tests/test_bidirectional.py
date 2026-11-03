import numpy as np
import pytest

from wujihand2_force_glove.bidirectional import (
    BidirectionalStateBus,
    ControlMode,
    DeviceState,
    HandoverController,
    HandTaskState,
)


def _task(timestamp_ns: int = 1) -> HandTaskState:
    return HandTaskState(np.zeros((21, 3)), timestamp_ns=timestamp_ns, source="test")


def test_state_bus_keeps_latest_state_per_device():
    bus = BidirectionalStateBus()
    glove = DeviceState(timestamp_ns=1, positions_rad=np.zeros(12))
    hand2 = DeviceState(timestamp_ns=2, positions_rad=np.zeros(20))

    bus.publish_glove(glove, _task(1))
    bus.publish_hand2(hand2, _task(2))
    snapshot = bus.snapshot()

    assert snapshot.glove.device.timestamp_ns == glove.timestamp_ns
    assert np.array_equal(snapshot.glove.device.positions_rad, glove.positions_rad)
    assert snapshot.glove.task.timestamp_ns == 1
    assert snapshot.hand2.device.timestamp_ns == hand2.timestamp_ns
    assert np.array_equal(snapshot.hand2.device.positions_rad, hand2.positions_rad)
    assert snapshot.hand2.task.timestamp_ns == 2


def test_state_bus_rejects_out_of_order_device_state():
    bus = BidirectionalStateBus()
    bus.publish_glove(DeviceState(timestamp_ns=2, positions_rad=np.zeros(12)), _task(2))

    with pytest.raises(ValueError, match="时间戳倒退"):
        bus.publish_glove(DeviceState(timestamp_ns=1, positions_rad=np.zeros(12)), _task(1))


def test_handover_blends_from_current_command_without_jump():
    controller = HandoverController(ControlMode.INFERENCE)
    current = np.array([1.0, 2.0])
    target = np.array([3.0, 4.0])

    controller.request(ControlMode.GLOVE_TAKEOVER, current, now_ns=100, duration_s=10e-9)
    assert np.allclose(controller.output(target, now_ns=100), current)
    assert np.allclose(controller.output(target, now_ns=105), [2.0, 3.0])
    assert np.allclose(controller.output(target, now_ns=110), target)
    assert controller.mode is ControlMode.GLOVE_TAKEOVER


def test_handover_rejects_shape_mismatch():
    controller = HandoverController(ControlMode.INFERENCE)
    controller.request(ControlMode.GLOVE_TAKEOVER, np.zeros(2), now_ns=0, duration_s=1)

    with pytest.raises(ValueError, match="维度不一致"):
        controller.output(np.zeros(3), now_ns=1)


def test_task_state_requires_canonical_21_points():
    with pytest.raises(ValueError, match="21x3"):
        HandTaskState(np.zeros((20, 3)), timestamp_ns=1, source="test")
