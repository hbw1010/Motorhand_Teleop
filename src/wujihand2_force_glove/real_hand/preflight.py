"""官方Python SDK只读身份/固件预检；连接释放后才允许C++接管控制。"""

from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from multiprocessing import get_context
from numbers import Integral
from uuid import uuid4


SDK_VERSION = "2026.8.31"
FIRMWARE_SOURCE = f"wuji-sdk=={SDK_VERSION} device.info.firmware_version（连接前只读预检）"


def _load_sdk():
    try:
        installed = version("wuji-sdk")
    except PackageNotFoundError as exc:
        raise RuntimeError(f"固件只读预检需要可选依赖wuji-sdk=={SDK_VERSION}") from exc
    if installed != SDK_VERSION:
        raise RuntimeError(f"固件只读预检需要wuji-sdk=={SDK_VERSION}，当前为{installed}")
    try:
        import wuji_sdk
    except ImportError as exc:
        raise RuntimeError(f"无法加载固件预检SDK wuji-sdk=={SDK_VERSION}：{exc}") from exc
    return wuji_sdk


def read_firmware_version(*, serial_number, hand, address="", timeout_ms=1000):
    """精确连接二代同侧设备，只读身份与固件；不使能、清错、设目标或回零。

    地址存在时按地址连接并再次核对序列号，否则使用显式SN选择器；不调用
    scan或auto_connect。SDK内部可能按SN发现设备，但绝不选择首个设备。
    仅断开此次独立别名对应的device，不shutdown全局manager或影响另一手。
    版本不是批准值；预检失败或断开失败均抛错，不交还后续控制权限。

    官方Python SDK连接等待会持有GIL；使用spawn隔离，不能饿死本地USB采集。
    子进程结束后才返回固件版本，保证其连接已释放；不fork正在采集的线程。
    """
    if hand not in ("left", "right"):
        raise ValueError("预检手侧必须为left或right")
    if not isinstance(serial_number, str) or not serial_number.strip() or "\x00" in serial_number:
        raise ValueError("预检必须显式指定非空且无NUL的设备序列号")
    if (serial_number != serial_number.strip()
            or ord(serial_number[0]) <= 32 or ord(serial_number[-1]) <= 32):
        raise ValueError("预检序列号不得包含首尾空白或控制字符")
    if not isinstance(address, str) or "\x00" in address:
        raise ValueError("预检地址必须为不含NUL的字符串")
    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, Integral) or not 1 <= timeout_ms <= 2**32 - 1:
        raise ValueError("预检timeout_ms必须为正uint32整数")
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as worker:
        return worker.submit(
            _read_firmware_version, serial_number=serial_number, hand=hand,
            address=address, timeout_ms=int(timeout_ms),
        ).result()


def _read_firmware_version(*, serial_number, hand, address="", timeout_ms=1000):
    """子进程内完成只读SDK操作；不接触主进程的采集和控制会话。"""
    sdk = _load_sdk()
    manager = sdk.SdkManager.instance()
    options = sdk.ConnectOptions(
        timeout_ms=int(timeout_ms), retry_count=0, enable_bridge=False,
        auto_time_sync_interval_ms=None,
    )
    selector = {"address": address} if address else {"sn": serial_number}
    device = manager.connect(
        **selector, device_name=f"real_hand_preflight_{uuid4().hex}", options=options,
    )
    failure = None
    try:
        if not isinstance(device, sdk.WujiHand2):
            raise RuntimeError("预检连接的设备不是WujiHand2")
        if device.serial_number != serial_number:
            raise RuntimeError("预检设备序列号缺失或与显式配置不符")
        # 固定版官方HandednessResource声明此GET返回left/right字符串。
        side = device.handedness().get()
        if not isinstance(side, str) or side.lower() != hand:
            raise RuntimeError("预检设备手侧与显式配置不符")
        info = device.info
        if info is None or info.serial_number != serial_number:
            raise RuntimeError("预检设备信息中的序列号缺失或不一致")
        firmware = info.firmware_version
        if (not isinstance(firmware, str) or not firmware.strip() or "\x00" in firmware
                or len(firmware.encode("utf-8")) > 127):
            raise RuntimeError("预检未获得真实非空且可完整表示的固件版本字符串")
        return firmware
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            device.disconnect()
        except BaseException as cleanup:
            if failure is not None:
                raise BaseExceptionGroup("只读预检失败且断开连接失败", [failure, cleanup]) from None
            raise


def _discover_devices():
    """复用官方SdkManager.scan；扫描不连接、清错、使能或选择首个设备。"""
    sdk = _load_sdk()
    return tuple(
        {
            "serial_number": device.sn,
            "address": device.address,
            "transport_type": str(device.transport_type),
            "device_type": str(device.device_type),
            # 固定SDK扫描结果没有手侧和固件，不能把未知身份猜成右手。
            "hand": None,
            "firmware_version": None,
        }
        for device in sdk.SdkManager.instance().scan()
    )


def discover_devices(*, timeout_ms=1000):
    """只读列出官方SDK发现的全部身份，不建立控制会话。

    SDK的scan没有可配置超时参数；timeout_ms仅作调用配置一致性校验，
    不冒充底层扫描超时保证。扫描放入spawn子进程避免阻塞采集线程。
    """
    if (isinstance(timeout_ms, bool) or not isinstance(timeout_ms, Integral)
            or not 1 <= timeout_ms <= 2**32 - 1):
        raise ValueError("timeout_ms必须为正uint32整数")
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as worker:
        return worker.submit(_discover_devices).result()


