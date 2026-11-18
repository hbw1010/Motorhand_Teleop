"""单命令入口的子进程生命周期；不连接硬件。"""
import sys

import pytest

from motorhand_teleop import sharpa_sim
from motorhand_teleop.sharpa_sim import run_pair


def test_window_exit_stops_only_the_owned_backend(tmp_path):
    stopped = tmp_path / "stopped"
    script = ("import signal,time,pathlib; "
              f"signal.signal(signal.SIGINT,lambda *_:(pathlib.Path({str(stopped)!r}).write_text('ok'),exit(0))); "
              "time.sleep(30)")
    assert run_pair([sys.executable, "-c", script],
                    [sys.executable, "-c", "import time;time.sleep(.3)"]) == 0
    assert stopped.read_text() == "ok"


def test_backend_failure_closes_owned_viewer():
    assert run_pair([sys.executable, "-c", "raise SystemExit(7)"],
                    [sys.executable, "-c", "import time;time.sleep(30)"]) == 7


def test_stop_failure_is_not_hidden_by_normal_window_exit():
    script = "import signal,time;signal.signal(signal.SIGINT,lambda *_:exit(7));time.sleep(30)"
    assert run_pair([sys.executable, "-c", script],
                    [sys.executable, "-c", "import time;time.sleep(.3)"]) == 7


@pytest.mark.parametrize("mode", ["teleop", "force-contact", "force-linear", "takeover"])
def test_live_mode_uses_hardware_binary_and_managed_viewer(tmp_path, monkeypatch, mode):
    binary = tmp_path / ".cache/sharpa-control/motorhand_sharpa"
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setattr(sharpa_sim, "ROOT", tmp_path)
    monkeypatch.setenv("DISPLAY", ":1")
    native = ["--serial", "OFFLINE-TEST", "--hand", "left", "--execute"]
    if mode == "takeover":
        native.extend(["--trajectory", "/tmp/offline-trajectory.jsonl"])

    def verify(controller, viewer):
        assert controller == [str(binary), mode, *native, "--curves-port", "18766"]
        assert viewer == [sys.executable, "-m", "motorhand_teleop.sharpa_viewer",
                          "--port", "18766", "--managed"]
        return 0

    monkeypatch.setattr(sharpa_sim, "run_pair", verify)
    assert sharpa_sim.main([mode, *native, "--curves-port", "18766"]) == 0
