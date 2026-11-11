"""独立Wuji接口离线契约；全部显式注入边界，不连接或使能真实设备。"""
import ctypes
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from motorhand_teleop.domain import WUJI_JOINTS, WujiState, WujiTarget
from motorhand_teleop.real_hand import _wuji_hand2_native as native
from motorhand_teleop.real_hand import preflight
from motorhand_teleop.real_hand.wuji_hand2 import WujiHand2HardwareDriver


class Backend:
    def __init__(self):
        self.calls = []
        self.firmware = "实测固件"
        self.positions = np.linspace(0, 0.19, 20)
        self.send_failure = False
        self.enable_failure = False

    def connect(self):
        self.calls.append("connect")
        return SimpleNamespace(
            model="wuji_hand2", hand="right", serial_number="SN1",
            firmware_version=self.firmware,
        )

    def read_state(self):
        self.calls.append("read_state")
        return WujiState(
            self.positions.copy(), None, None, 12345, 20,
            {"joint_enabled": (True,) * 20, "joint_online": (True,) * 20,
             "joint_error_codes": (0,) * 20},
        )

    def enable(self):
        self.calls.append("enable")
        if self.enable_failure:
            raise RuntimeError("使能失败")

    def send(self, positions, *, deadline_ns):
        self.calls.append(("send", np.array(positions), deadline_ns))
        if self.send_failure:
            raise RuntimeError("发送失败")

    def home(self):
        self.calls.append("home")

    def disable(self):
        self.calls.append("disable")

    def emergency_stop(self):
        self.calls.append("emergency_stop")

    def close(self):
        self.calls.append("close")


def make_driver(backend, **options):
    return WujiHand2HardwareDriver(
        serial_number="SN1", backend_factory=lambda: backend, **options,
    )


def target(*, velocity=0.0, effort=0.0, positions=None):
    return WujiTarget(
        np.zeros(20) if positions is None else positions,
        np.full(20, velocity), np.full(20, effort),
    )


def deadline():
    return time.monotonic_ns() + 10_000_000_000


def test_default_constructor_is_lazy_and_disables_automatic_home(monkeypatch):
    monkeypatch.setattr(native, "_native_library", lambda: pytest.fail("构造不得加载原生库"))
    driver = WujiHand2HardwareDriver(serial_number="SN1")
    backend = driver._factory()
    assert not driver.connected
    assert backend._options.auto_home == 0
    driver.close()




@pytest.fixture
def sdk_callback_library(monkeypatch):
    """显式加载仅替换SDK符号的离线库；绝不回退到真实SDK或真实设备。"""
    library = (Path(__file__).resolve().parents[1]
               / ".cache/native-real-hand/build/libreal_hand_sdk_offline.so")
    if not library.is_file():
        pytest.skip("需要父级先构建REAL_HAND_BUILD_TESTS离线SDK库")
    monkeypatch.setattr(native, "_LIBRARY", library)
    native._native_library.cache_clear()
    monkeypatch.setattr(preflight, "read_firmware_version", lambda **kwargs: "offline-fw")
    lib = native._native_library()
    pointer = ctypes.POINTER(ctypes.c_float)
    lib.real_hand_test_set_default_feedback.argtypes = [pointer, pointer]
    lib.real_hand_test_set_default_feedback.restype = None
    lib.real_hand_test_set_default_hand.argtypes = [ctypes.c_int]
    lib.real_hand_test_set_default_hand.restype = None
    lib.real_hand_test_set_default_hand(1)
    try:
        yield lib
    finally:
        zeros = (ctypes.c_float * 20)()
        lib.real_hand_test_set_default_feedback(zeros, zeros)
        native._native_library.cache_clear()


@pytest.mark.parametrize("unavailable", [None, "velocities_rad_s", "effort_a"])
def test_sdk_callback_through_abi6_and_public_read_state(sdk_callback_library, unavailable):
    # 经过生产callback -> NID重排 -> 核心C ABI -> ctypes -> 公开read_state，
    # 不是向_State填值后只核对字段拷贝。每轴量可区分，且不同于位置/诊断零值。
    velocity = -.5 - np.arange(20) * .125
    effort = .125 + np.arange(20) * .03125
    if unavailable == "velocities_rad_s":
        velocity[17] = np.nan
    elif unavailable == "effort_a":
        effort[17] = np.inf
    sdk_callback_library.real_hand_test_set_default_feedback(
        (ctypes.c_float * 20)(*velocity), (ctypes.c_float * 20)(*effort),
    )
    driver = WujiHand2HardwareDriver(serial_number="OFFLINE-ABI6")
    try:
        driver.connect()
        state = driver.read_state()
        assert state.received_ns > 0 and state.diagnostics["device_timestamp_us"] > 0
        assert state.online_count == 20 and state.diagnostics["enabled"] is False
        np.testing.assert_array_equal(state.positions_rad, np.zeros(20))
        for name, expected in (("velocities_rad_s", velocity), ("effort_a", effort)):
            actual = getattr(state, name)
            if name == unavailable:
                assert actual is None
            else:
                np.testing.assert_array_equal(actual, expected)
                assert not actual.flags.writeable
        assert state.diagnostics["velocity_feedback_available"] == (unavailable != "velocities_rad_s")
        assert state.diagnostics["effort_feedback_available"] == (unavailable != "effort_a")
        assert "ABI 6 joint_states" in state.diagnostics["feedback_source"]
        assert driver.active is False
    finally:
        driver.close()


@pytest.mark.parametrize("native_names", [False, True])
def test_public_read_state_reorders_optional_semantic_fields_and_does_not_keep_missing_cache(native_names):
    names = (tuple(f"{finger}_S{joint}" for finger in ("thumb", "index", "middle", "ring", "pinky")
                   for joint in range(1, 5)) if native_names else WUJI_JOINTS)
    positions = np.arange(20) * .01
    velocity = -.5 - np.arange(20) * .125
    effort = .125 + np.arange(20) * .03125
    payload = {
        "positions_rad": dict(reversed(tuple(zip(names, positions)))),
        "velocities_rad_s": dict(reversed(tuple(zip(names, velocity)))),
        "effort_a": dict(reversed(tuple(zip(names, effort)))),
        "received_ns": 12345, "online": True,
    }
    backend = Backend()
    backend.read_state = lambda: payload
    driver = make_driver(backend)
    try:
        driver.connect()
        first = driver.read_state()
        np.testing.assert_array_equal(first.positions_rad, positions)
        np.testing.assert_array_equal(first.velocities_rad_s, velocity)
        np.testing.assert_array_equal(first.effort_a, effort)
        payload["effort_a"][names[17]] = np.nan
        second = driver.read_state()
        assert second.effort_a is None and second.velocities_rad_s is not None
        assert second.diagnostics["effort_feedback_available"] is False
        # 缺单轴也不能返回旧完整帧或以零补齐。
        del payload["velocities_rad_s"][names[17]]
        del payload["effort_a"]
        third = driver.read_state()
        assert third.velocities_rad_s is None and third.effort_a is None
        assert first.received_ns == second.received_ns == third.received_ns == 12345
        np.testing.assert_array_equal(first.effort_a, effort)
        assert not first.effort_a.flags.writeable and not first.velocities_rad_s.flags.writeable
    finally:
        driver.close()


def test_readonly_connect_state_and_close_never_take_ownership():
    backend = Backend()
    driver = make_driver(backend)
    identity = driver.connect()
    assert identity.serial_number == "SN1"
    assert identity.firmware_version == backend.firmware
    assert identity.hand == "right"
    state = driver.read_state()
    assert np.array_equal(state.positions_rad, backend.positions)
    assert state.received_ns == 12345
    assert state.velocities_rad_s is None and state.effort_a is None
    assert state.online_count == 20
    driver.disable()
    driver.emergency_stop()
    driver.close()
    assert backend.calls == ["connect", "read_state", "close"]


@pytest.mark.parametrize("approved", [None, "错误固件"])
def test_enable_requires_exact_explicit_firmware_approval(approved):
    backend = Backend()
    driver = make_driver(backend, approved_firmware_version=approved)
    driver.connect()
    with pytest.raises(ValueError, match="批准"):
        driver.enable()
    driver.close()
    assert backend.calls == ["connect", "close"]


def test_explicit_lifecycle_sends_twenty_radians_without_implicit_home():
    backend = Backend()
    driver = make_driver(backend, approved_firmware_version=backend.firmware)
    driver.connect()
    driver.enable()
    value = target(positions=np.linspace(0, 0.19, 20))
    expires = deadline()
    driver.validate_target(value, expires)
    assert backend.calls == ["connect", "enable"]
    driver.send_target(value, deadline_ns=expires)
    sent = backend.calls[-1]
    assert sent[0] == "send" and np.array_equal(sent[1], value.positions_rad)
    assert sent[2] == expires
    driver.home()
    driver.close()
    assert backend.calls[-3:] == ["home", "disable", "close"]


@pytest.mark.parametrize("options", [{"velocity": 1e-15}, {"effort": 1e-15}])
def test_position_only_preflight_refuses_any_nonzero_channel_without_side_effects(options):
    backend = Backend()
    driver = make_driver(backend, approved_firmware_version=backend.firmware)
    driver.connect()
    driver.enable()
    with pytest.raises(NotImplementedError):
        driver.validate_target(target(**options), deadline())
    assert driver.active
    assert backend.calls == ["connect", "enable"]
    driver.close()




@pytest.mark.parametrize("expires, error", [(0, TimeoutError), (True, ValueError), (2**64, ValueError)])
def test_deadline_preflight_is_readonly(expires, error):
    backend = Backend()
    driver = make_driver(backend)
    driver.connect()
    with pytest.raises(error):
        driver.validate_target(target(), expires)
    assert backend.calls == ["connect"]
    driver.close()


@pytest.mark.parametrize("expires,error", [(0, TimeoutError), (True, ValueError)])
def test_expired_send_stops_and_latches_owned_device(expires, error):
    backend = Backend()
    driver = make_driver(backend, approved_firmware_version=backend.firmware)
    driver.connect()
    driver.enable()
    with pytest.raises(error):
        driver.send_target(target(), deadline_ns=expires)
    assert not driver.active and "emergency_stop" in backend.calls
    with pytest.raises(RuntimeError):
        driver.send_target(target(), deadline_ns=deadline())
    driver.close()


def test_joint_limit_preflight_does_not_clamp_or_stop():
    backend = Backend()
    driver = make_driver(backend)
    driver.connect()
    with pytest.raises(ValueError, match="限位"):
        driver.validate_target(target(positions=np.full(20, 100.0)), deadline())
    assert backend.calls == ["connect"]
    driver.close()


@pytest.mark.parametrize("failure", ["enable_failure", "send_failure"])
def test_owned_operation_failure_stops_without_automatic_home(failure):
    backend = Backend()
    setattr(backend, failure, True)
    driver = make_driver(backend, approved_firmware_version=backend.firmware)
    driver.connect()
    with pytest.raises(RuntimeError, match="失败"):
        driver.enable()
        driver.send_target(target(), deadline_ns=deadline())
    assert not driver.active
    assert "emergency_stop" in backend.calls
    assert "home" not in backend.calls
    driver.close()


def test_connect_rejects_wrong_identity_without_stopping_anyone():
    backend = Backend()
    backend.connect = lambda: SimpleNamespace(
        model="wuji_hand2", hand="left", serial_number="SN1", firmware_version="实测固件",
    )
    driver = make_driver(backend)
    with pytest.raises(ValueError, match="实际型号"):
        driver.connect()
    assert backend.calls == ["close"]


def test_official_discovery_lists_every_identity_without_connect(monkeypatch):
    class Manager:
        def scan(self):
            return [SimpleNamespace(sn=sn, address=f"usb://{sn}", transport_type="USB", device_type="WujiHand2")
                    for sn in ("SN2", "SN1")]

        def connect(self, **options):
            pytest.fail("扫描不得自动连接设备")

    sdk = SimpleNamespace(SdkManager=SimpleNamespace(instance=lambda: Manager()))
    monkeypatch.setattr(preflight, "_load_sdk", lambda: sdk)
    devices = preflight._discover_devices()
    assert [device["serial_number"] for device in devices] == ["SN2", "SN1"]
    assert all(device["hand"] is None and device["firmware_version"] is None for device in devices)


def test_discovery_normalizes_official_enums_without_connecting(monkeypatch):
    hand2, unknown, udp = object(), object(), object()
    class Manager:
        def scan(self):
            return [
                SimpleNamespace(sn="SN1", address="192.0.2.1:1234", transport_type=udp, device_type=hand2),
                SimpleNamespace(sn="SN2", address="", transport_type=udp, device_type=unknown),
            ]
    sdk = SimpleNamespace(
        SdkManager=SimpleNamespace(instance=lambda: Manager()),
        DeviceType=SimpleNamespace(WujiHand2=hand2),
        TransportType=SimpleNamespace(Udp=udp),
    )
    monkeypatch.setattr(preflight, "_load_sdk", lambda: sdk)
    devices = preflight._discover_devices()
    assert devices[0]["device_type"] == "WujiHand2"
    assert devices[0]["transport_type"] == "UDP"
    assert preflight.select_device(devices=devices)["serial_number"] == "SN1"


def test_selection_reuses_scan_and_prefers_discovered_udp(monkeypatch):
    devices = [
        {"serial_number": "SN1", "device_type": "WujiHand2", "transport_type": "USB", "address": "usb:SN1"},
        {"serial_number": "SN1", "device_type": "WujiHand2", "transport_type": "UDP", "address": "192.0.2.1:1234"},
        {"serial_number": "other", "device_type": "Unknown", "transport_type": "UDP", "address": "192.0.2.2:1234"},
    ]
    monkeypatch.setattr(preflight, "discover_devices", lambda **kwargs: pytest.fail("不得重复扫描"))
    assert preflight.select_device(devices=devices)["address"] == "192.0.2.1:1234"
    assert preflight.select_device(serial_number="SN1", devices=devices)["serial_number"] == "SN1"
    with pytest.raises(RuntimeError, match="未发现"):
        preflight.select_device(serial_number="SN2", devices=devices)
    with pytest.raises(ValueError, match="serial_number"):
        preflight.select_device(serial_number=" SN1", devices=devices)


def test_selection_rejects_ambiguous_devices_and_addresses():
    def entry(sn, address):
        return {"serial_number": sn, "device_type": "WujiHand2",
                "transport_type": "UDP", "address": address}
    with pytest.raises(RuntimeError, match="未发现"):
        preflight.select_device(devices=[])
    with pytest.raises(RuntimeError, match="多个可识别"):
        preflight.select_device(devices=[entry("SN1", "a"), entry("SN2", "b")])
    with pytest.raises(RuntimeError, match="多个连接地址"):
        preflight.select_device(serial_number="SN1", devices=[entry("SN1", "a"), entry("SN1", "b")])
    assert preflight.select_device(devices=[entry("SN1", "")])["address"] == ""


def test_firmware_preflight_address_still_checks_identity_and_disconnects(monkeypatch):
    calls = []
    class Device:
        serial_number = "WRONG"
        def disconnect(self):
            calls.append("disconnect")
    class Manager:
        def connect(self, **options):
            calls.append(options)
            return Device()
    sdk = SimpleNamespace(
        SdkManager=SimpleNamespace(instance=lambda: Manager()), WujiHand2=Device,
        ConnectOptions=lambda **options: options,
    )
    monkeypatch.setattr(preflight, "_load_sdk", lambda: sdk)
    with pytest.raises(RuntimeError, match="序列号"):
        preflight._read_firmware_version(serial_number="SN1", hand="right", address="192.0.2.1:1234")
    assert calls[0]["address"] == "192.0.2.1:1234"
    assert "sn" not in calls[0]
    assert calls[-1] == "disconnect"


def test_firmware_preflight_uses_explicit_selector_and_only_reads(monkeypatch):
    calls = []

    class Device:
        serial_number = "SN1"
        info = SimpleNamespace(serial_number="SN1", firmware_version="实测固件")

        def handedness(self):
            return SimpleNamespace(get=lambda: "right")

        def disconnect(self):
            calls.append("disconnect")

    class Manager:
        def connect(self, **options):
            calls.append(options)
            return Device()

    sdk = SimpleNamespace(
        SdkManager=SimpleNamespace(instance=lambda: Manager()), WujiHand2=Device,
        ConnectOptions=lambda **options: options,
    )
    monkeypatch.setattr(preflight, "_load_sdk", lambda: sdk)
    assert preflight._read_firmware_version(serial_number="SN1", hand="right") == "实测固件"
    assert calls[0]["sn"] == "SN1"
    assert calls[0]["options"]["retry_count"] == 0
    assert calls[0]["options"]["enable_bridge"] is False
    assert calls[-1] == "disconnect"


def test_sdk_version_must_match_official_pin(monkeypatch):
    monkeypatch.setattr(preflight, "version", lambda name: "2026.10.12")
    with pytest.raises(RuntimeError, match="2026.8.31"):
        preflight._load_sdk()


def test_discovery_prefers_unique_physical_wired_route(monkeypatch):
    candidates = [
        {"serial_number": "SN1", "device_type": "WujiHand2", "transport_type": "UDP", "address": address}
        for address in ("192.0.2.1:7447", "192.0.2.2:7447")
    ]
    monkeypatch.setattr(preflight, "_has_wired_route", lambda address: address == "192.0.2.2:7447")
    assert preflight.select_device(devices=candidates)["address"] == "192.0.2.2:7447"


def test_route_detection_distinguishes_wired_from_wireless_without_changing_network(tmp_path, monkeypatch):
    import json

    root = tmp_path / "net"
    for name in ("eth0", "wifi0"):
        interface = root / name
        (interface / "device").mkdir(parents=True)
        (interface / "type").write_text("1")
    (root / "wifi0" / "wireless").mkdir()
    monkeypatch.setattr(preflight, "Path", lambda _: root)
    monkeypatch.setattr(
        preflight.subprocess, "run",
        lambda command, **_: SimpleNamespace(stdout=json.dumps([
            {"dev": "eth0" if command[-1] == "192.0.2.1" else "wifi0"}
        ])),
    )
    assert preflight._has_wired_route("192.0.2.1:7447")
    assert not preflight._has_wired_route("192.0.2.2:7447")
