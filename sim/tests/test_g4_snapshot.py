"""共享 G4 C ABI 的离线回归测试，不访问真实串口。"""
import os
from pathlib import Path
import subprocess

from glove_sim.sdk.driver2 import SHIM_SOURCE, VENDOR_INCLUDE


def test_position_timestamp_is_available_before_watchdog_starts(tmp_path):
    source = Path(__file__).with_name("native") / "test_g4_snapshot.cpp"
    binary = tmp_path / "test_g4_snapshot"
    subprocess.run(
        [os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-pthread",
         f"-I{SHIM_SOURCE.parent}", f"-I{VENDOR_INCLUDE}", str(source),
         "-lutil", "-o", str(binary)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
