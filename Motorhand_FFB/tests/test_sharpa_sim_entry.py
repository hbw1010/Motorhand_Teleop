"""真实原生仿真入口验证；仅假手套，不访问硬件。"""
import json
from pathlib import Path
import socket
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / ".cache/sharpa-sim-control/motorhand_sharpa"


@pytest.mark.skipif(not BINARY.exists(), reason="需要先构建Sharpa仿真")
@pytest.mark.parametrize("feedback", [False, True])
def test_calibrated_absolute_pose_survives_startup_and_feedback(feedback):
    config = yaml.safe_load((ROOT / "config/retargeting/sharpa/right.yaml").read_text())
    thumb_axis = config["retarget"]["fixed_joints"]["right_thumb_MCP_AA"]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        connection.bind(("127.0.0.1", 0))
        connection.settimeout(.2)
        command = [str(BINARY), "sim", "--duration", "1.4", "--curves-port", str(connection.getsockname()[1])]
        if feedback:
            command.append("--feedback")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        frames = []
        try:
            while process.poll() is None:
                try:
                    frames.append(json.loads(connection.recv(65535)))
                except socket.timeout:
                    continue
            stdout, stderr = process.communicate(timeout=3)
            assert process.returncode == 0, (stdout, stderr)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        assert frames
        start = frames[0]["received_ns"]
        steady = [f for f in frames if f["received_ns"] - start > 800_000_000]
        assert steady
        for frame in steady:
            assert frame["hand_target_rad"][3] == pytest.approx(thumb_axis, abs=1e-6)
        if feedback:
            beginning = [f for f in frames if f["received_ns"] - start < 300_000_000]
            assert beginning
            for frame in beginning:
                assert not frame["force_geometry_valid"]
                assert all(abs(v) < 1e-12 for v in (frame["glove_nominal_nm"] or []))
