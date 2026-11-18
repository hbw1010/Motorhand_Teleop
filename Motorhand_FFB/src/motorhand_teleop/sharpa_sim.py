"""一起启动Sharpa控制和Motorhand／Sharpa并排查看器。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SIMULATION_MODES = ("sim", "sim-teleop")
LIVE_MODES = ("teleop", "force-contact", "force-linear", "takeover")


def _stop(process: subprocess.Popen | None, *, controller: bool) -> int | None:
    if process is None:
        return None
    if process.poll() is not None:
        return process.returncode
    process.send_signal(signal.SIGINT if controller else signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if controller:
            # 不强杀仍在卸力/回零清理的驱动，也不把未确认退出报成成功。
            raise RuntimeError(f"控制进程 {process.pid} 尚未确认停止，请查看终端状态")
        process.kill()
        process.wait()
    return process.returncode


def run_pair(controller_command: list[str], viewer_command: list[str]) -> int:
    controller = viewer = None
    result = 1
    try:
        viewer = subprocess.Popen(viewer_command, stdin=subprocess.DEVNULL)
        controller = subprocess.Popen(controller_command)
        while True:
            code = controller.poll()
            if code is not None:
                result = code
                break
            code = viewer.poll()
            if code is not None:
                result = code
                break
            time.sleep(.05)
    except KeyboardInterrupt:
        result = 0
    finally:
        try:
            stopped_code = _stop(controller, controller=True)
            if result == 0 and stopped_code not in (None, 0):
                result = stopped_code
        finally:
            _stop(viewer, controller=False)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=SIMULATION_MODES + LIVE_MODES)
    parser.add_argument("--headless", action="store_true", help="仅运行控制进程，不打开查看窗口")
    parser.add_argument("--curves-port", type=int, default=0, help="默认自动选择本机空闲端口")
    args, native_args = parser.parse_known_args(argv)
    if not 0 <= args.curves_port <= 65535:
        parser.error("--curves-port必须为0至65535")
    simulation = args.mode in SIMULATION_MODES
    build_task = "build-sharpa-sim" if simulation else "build-sharpa"
    build_directory = "sharpa-sim-control" if simulation else "sharpa-control"
    binary = ROOT / ".cache" / build_directory / "motorhand_sharpa"
    if not binary.is_file():
        parser.error(f"请先运行 pixi run {build_task}")
    command = [str(binary), args.mode, *native_args]
    if args.headless:
        os.execv(command[0], command + ["--curves-port", str(args.curves_port or 8766)])
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        parser.error("未检测到桌面显示；无窗口运行请加 --headless")
    port = args.curves_port
    if not port:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
    try:
        return run_pair(command + ["--curves-port", str(port)],
                        [sys.executable, "-m", "motorhand_teleop.sharpa_viewer", "--port", str(port), "--managed"])
    except (OSError, RuntimeError) as error:
        print(f"Sharpa启动/停止失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
