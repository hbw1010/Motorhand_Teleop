from types import SimpleNamespace

from encos_hand import controller


def test_auto_discovery_accepts_canable_and_ignores_other_usb_serials(monkeypatch):
    ports = [
        SimpleNamespace(device="/dev/ttyS31", vid=None, pid=None),
        SimpleNamespace(device="/dev/ttyACM3", vid=0x2207, pid=0x0010),
        SimpleNamespace(device="/dev/ttyACM0", vid=0x16D0, pid=0x117E),
        SimpleNamespace(device="/dev/ttyACM1", vid=0x16D0, pid=0x117E),
    ]
    monkeypatch.setattr(controller.lp, "comports", lambda: ports)
    assert controller.discover_usb_serial_ports() == ("/dev/ttyACM0", "/dev/ttyACM1")


def test_worker_expires_mit_without_further_controller_calls(monkeypatch):
    import math
    import threading

    from encos_hand import channel_worker
    from encos_hand.protocol import MotorProtocol

    clock = [50]
    monkeypatch.setattr(channel_worker.time, "monotonic_ns", lambda: clock[0])
    expected_mit = MotorProtocol.mit_command(math.radians(12.0), 0.0, 45.0, 0.5, 0.0)
    expected_safe = MotorProtocol.position_command(12.0, 0.0, 0.0, ack=2)
    expired = threading.Event()

    class Bus:
        def __init__(self):
            self.messages = []

        def send(self, message):
            self.messages.append(bytes(message.data))
            if bytes(message.data) == expected_mit:
                clock[0] = 101
            elif bytes(message.data) == expected_safe:
                expired.set()

        def recv(self, timeout):
            return None

        def shutdown(self):
            pass

    bus = Bus()
    worker = channel_worker.HeadlessChannelWorker(1)
    worker.bus, worker.is_connected = bus, True
    worker.active_ids = [1]
    worker.last_angles = {1: 12.0}
    worker.set_joint_cmd(1, {"enabled": True, "mode": "mit", "target_angle": 12.0,
                             "mit_kp": 45.0, "mit_kd": 0.5, "deadline_ns": 100})
    worker.start()
    try:
        assert expired.wait(timeout=1.0)
    finally:
        worker.stop()
        worker.join(timeout=1.0)
    assert bus.messages[0] == expected_mit
    assert all(frame == expected_safe for frame in bus.messages[1:])
