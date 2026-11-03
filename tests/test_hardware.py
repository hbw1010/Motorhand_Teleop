from types import SimpleNamespace
import time

import numpy as np
import pytest

from wujihand2_force_glove import HandTaskState
from wujihand2_force_glove.bidirectional import ControlMode
from wujihand2_force_glove.domain import GLOVE_JOINTS, WujiState, WujiTarget
from wujihand2_force_glove.real_hand.wuji_hand2 import WujiHand2HardwareDriver
from wujihand2_force_glove.real_hand.control import RealWujiControlLoop


NATIVE_NAMES = tuple(
    f"{finger}_S{joint}"
    for finger in ("thumb", "index", "middle", "ring", "pinky")
    for joint in range(1, 5)
)


class FakeBackend:
    def __init__(self):
        self.connected = False
        self.enabled = False
        self.sent = []
        self.closed = False

    def connect(self):
        self.connected = True
        return SimpleNamespace(
            model="wuji_hand2", hand="right", identifier="SN1",
            serial_number="SN1", firmware_version="test-fw",
        )

    def enable(self):
        self.enabled = True

    def send(self, pose, *, deadline_ns):
        self.sent.append((pose, deadline_ns))

    def read_state(self):
        return SimpleNamespace(
            positions_rad={name: index * 0.01 for index, name in enumerate(NATIVE_NAMES)},
            received_ns=time.monotonic_ns(),
            online=True,
            faults=(),
            diagnostics={"joint_online": (True,) * 20},
            model="wuji_hand2",
            hand="right",
            phase="enabled",
        )

    def disable(self):
        self.enabled = False
        return SimpleNamespace(requested=True, confirmed=True, method="fake")

    def emergency_stop(self):
        self.enabled = False

    def close(self):
        self.closed = True





def test_wuji_adapter_normalizes_native_joint_state_and_position_target():
    backend = FakeBackend()
    driver = WujiHand2HardwareDriver(
        hand="right", serial_number="SN1", approved_firmware_version="test-fw",
        backend_factory=lambda: backend,
    )
    driver.connect()
    driver.enable()
    state = driver.read_state()
    assert np.allclose(state.positions_rad, np.arange(20) * 0.01)
    assert state.velocities_rad_s is None
    assert state.effort_a is None

    target = WujiTarget(np.zeros(20), np.zeros(20), np.zeros(20))
    driver.send_target(target, deadline_ns=10**18)
    assert len(backend.sent) == 1
    driver.close()
    assert backend.closed


def test_wuji_adapter_rejects_unavailable_per_frame_effort():
    backend = FakeBackend()
    driver = WujiHand2HardwareDriver(
        hand="right", serial_number="SN1", approved_firmware_version="test-fw",
        backend_factory=lambda: backend,
    )
    driver.connect()
    driver.enable()
    with pytest.raises(NotImplementedError, match="effort"):
        driver.send_target(WujiTarget(np.zeros(20), np.zeros(20), np.ones(20)), deadline_ns=10**18)
    driver.close()




@pytest.mark.parametrize("deadline,error", [(0, TimeoutError), (True, ValueError)])
def test_wuji_adapter_rejects_deadline_and_latches_failure(deadline, error):
    backend = FakeBackend()
    driver = WujiHand2HardwareDriver(
        hand="right", serial_number="SN1", approved_firmware_version="test-fw",
        backend_factory=lambda: backend,
    )
    driver.connect()
    driver.enable()
    target = WujiTarget(np.zeros(20), np.zeros(20), np.zeros(20))
    with pytest.raises(error):
        driver.send_target(target, deadline_ns=deadline)
    assert not driver.active and backend.sent == []
    with pytest.raises(RuntimeError, match="故障已锁存"):
        driver.send_target(target, deadline_ns=10**18)
    driver.close()




class FakeTaskModel:
    def state(self, positions, *, timestamp_ns=0):
        return HandTaskState(np.zeros((21, 3)), timestamp_ns=timestamp_ns, source="fake")


class FakeBridge:
    def __init__(self):
        self.hand2_model = FakeTaskModel()

    def glove_to_hand2(self, joints, *, timestamp_ns=0):
        values = np.asarray([joints[name] for name in GLOVE_JOINTS])
        return SimpleNamespace(positions_rad=np.r_[values[:4], np.zeros(16)]), None

    def hand2_to_glove(self, positions, *, initial=None, timestamp_ns=0):
        return SimpleNamespace(glove_positions_rad=np.zeros(12), success=True), None


class FakeLoopDriver:
    def __init__(self):
        self.commands = []
        self.state = WujiState(np.zeros(20), None, None, 1, 20)

    def read_state(self):
        return self.state

    def validate(self, target):
        pass

    def send_target(self, target, *, deadline_ns):
        self.commands.append((target, deadline_ns))

    def enable(self):
        pass

    def disable(self):
        pass

    def emergency_stop(self):
        pass

    def close(self):
        pass


def test_real_control_loop_uses_real_state_and_smooth_mode_switch():
    driver = FakeLoopDriver()
    loop = RealWujiControlLoop(driver, FakeBridge(), initial_mode=ControlMode.INFERENCE)
    first = loop.step(inference_positions_rad=np.zeros(20), now_ns=100)
    assert np.allclose(first.sent_positions_rad, 0.0)

    loop.request_mode(ControlMode.GLOVE_TAKEOVER, now_ns=200, duration_s=1e-6)
    joints = {name: 0.1 for name in GLOVE_JOINTS}
    mid = loop.step(glove_joints=joints, now_ns=201)
    assert mid.mode is ControlMode.INFERENCE
    assert 0.0 < mid.sent_positions_rad[0] < 0.1
    switched = loop.step(glove_joints=joints, now_ns=1201)
    assert loop.mode is ControlMode.GLOVE_TAKEOVER
    assert np.allclose(switched.sent_positions_rad[:4], 0.1)
    assert len(driver.commands) == 3
