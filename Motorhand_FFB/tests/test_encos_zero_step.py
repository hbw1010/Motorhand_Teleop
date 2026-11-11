import copy
import io
import json

import pytest

from motorhand_teleop import encos_zero_step as zero
from motorhand_teleop.domain import GLOVE_JOINTS


class ReadOnlyHand:
    def __init__(self):
        self.read_count = 0
        self.closed = False
        self.viewed = []

    def connect(self):
        pass

    def scan(self):
        return {1: list(range(1, 13))}

    def get_online_ids(self):
        return tuple(range(1, 13))

    def get_all_feedbacks(self):
        return {mid: {"err_code": 0} for mid in range(1, 13)}

    def get_all_angles(self):
        self.read_count += 1
        return {mid: mid * 10.0 + self.read_count * 0.1 for mid in range(1, 13)}

    def enable(self, motor_id, enabled=True):
        assert enabled is False, "标零不得使能电机"

    def set_position(self, **kwargs):
        raise AssertionError("标零不得发送位置目标")

    def set_mit(self, **kwargs):
        raise AssertionError("标零不得发送MIT目标")

    def close(self):
        self.closed = True


def make_mapping():
    motors = {}
    for mid, joint in enumerate(GLOVE_JOINTS, 1):
        z = mid * 0.123
        sign = -1.0 if mid in (4, 6, 8, 10, 12) else 1.0
        relative = (-90.0, 90.0) if mid <= 2 else (0.0, 90.0)
        raw = sorted(z + value / sign for value in relative)
        motors[str(mid)] = dict(joint=joint, zero_deg=z, sign=sign, scale=1.0,
                                min_deg=raw[0], max_deg=raw[1])
    return {"calibrated": True, "motors": motors, "haptic": {"kp": 45.0, "kd": 0.5}}


def prepare(tmp_path, monkeypatch, line="\n"):
    from motorhand_teleop import encos_live_viewer

    paths = [tmp_path / name / "glove_mapping.json" for name in ("source", "target")]

    before = []
    for index, path in enumerate(paths):
        path.parent.mkdir()
        data = make_mapping()
        data["project_value"] = index
        path.write_text(json.dumps(data))
        path.with_name("joint_calibration.json").write_text(json.dumps({"1": {"zero": 7.0}}))
        before.append(copy.deepcopy(data))
    hand = ReadOnlyHand()
    monkeypatch.setattr(zero, "EncosHand", lambda: hand)
    monkeypatch.setattr(zero.sys, "stdin", io.StringIO(line))

    def view(*, mapping_path):
        assert hand.closed
        hand.viewed.append(mapping_path)

    monkeypatch.setattr(encos_live_viewer, "run_live_viewer", view)
    monkeypatch.setattr(zero.select, "select", lambda *args: ([zero.sys.stdin], [], []))
    return paths, before, hand


@pytest.mark.parametrize("finger,ids", [("index", (5, 6)), ("middle", (7, 8)),
                                      ("ring", (9, 10)), ("pinky", (11, 12))])
def test_single_finger_confirmation_preserves_other_axes_and_relative_limits(tmp_path, monkeypatch, finger, ids):
    paths, before, hand = prepare(tmp_path, monkeypatch)
    assert zero.main(["--mapping", str(paths[0]), "--sync-mapping", str(paths[1]),
                      "--finger", finger]) == 0
    for path, original in zip(paths, before):
        updated = json.loads(path.read_text())
        calibration = json.loads(path.with_name("joint_calibration.json").read_text())
        assert updated["project_value"] == original["project_value"]
        assert updated["haptic"] == original["haptic"]
        assert updated["calibrated"] is True
        assert calibration["1"] == {"zero": 7.0}
        for mid in range(1, 13):
            key = str(mid)
            old, new = original["motors"][key], updated["motors"][key]
            if mid not in ids:
                assert new == old
                continue
            # 第二次读取发生于Enter确认，不能保存确认前用于显示的角度。
            assert new["zero_deg"] == pytest.approx(mid * 10.0 + 0.2)
            for bound in ("min_deg", "max_deg"):
                assert new[bound] - new["zero_deg"] == pytest.approx(old[bound] - old["zero_deg"])
            zero_bound = "min_deg" if new["sign"] > 0 else "max_deg"
            assert new[zero_bound] == new["zero_deg"]
            assert calibration[key] == {"zero": new["zero_deg"], "min": new["min_deg"], "max": new["max_deg"]}
    assert hand.closed
    assert hand.viewed == [paths[0].resolve()]


def test_eof_is_not_confirmation_and_does_not_save(tmp_path, monkeypatch):
    paths, _, hand = prepare(tmp_path, monkeypatch, line="")
    files = paths + [path.with_name("joint_calibration.json") for path in paths]
    before = {path: path.read_bytes() for path in files}
    with pytest.raises(EOFError, match="未确认零位"):
        zero.main(["--mapping", str(paths[0]), "--sync-mapping", str(paths[1]), "--finger", "index"])
    assert {path: path.read_bytes() for path in files} == before
    assert hand.closed
    assert hand.viewed == []


def test_keyboard_interrupt_keeps_calibration_and_skips_viewer(tmp_path, monkeypatch):
    paths, _, hand = prepare(tmp_path, monkeypatch)
    before = paths[0].read_bytes()

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(zero.select, "select", interrupt)
    assert zero.main(["--mapping", str(paths[0]), "--finger", "index"]) == 130
    assert paths[0].read_bytes() == before
    assert hand.closed and not hand.viewed


def test_one_command_calibrates_all_axes_then_opens_viewer(tmp_path, monkeypatch):
    from motorhand_teleop import encos_live_viewer

    paths, before, hand = prepare(tmp_path, monkeypatch, line="")
    viewed = []

    def view(*, mapping_path):
        assert hand.closed
        viewed.append(json.loads(mapping_path.read_text()))

    monkeypatch.setattr(encos_live_viewer, "run_live_viewer", view)
    assert zero.main(["--mapping", str(paths[0]), "--all"]) == 0

    assert len(viewed) == 1
    assert viewed[0]["zero_calibrated_fingers"] == [
        "thumb", "index", "middle", "ring", "pinky",
    ]
    for mid in range(1, 13):
        old = before[0]["motors"][str(mid)]
        new = viewed[0]["motors"][str(mid)]
        assert new["zero_deg"] == pytest.approx(mid * 10.0 + 0.1)
        for bound in ("min_deg", "max_deg"):
            assert new[bound] - new["zero_deg"] == pytest.approx(old[bound] - old["zero_deg"])


def test_sequential_fingers_open_viewer_only_after_all_confirmations(tmp_path, monkeypatch):
    paths, _, hand = prepare(tmp_path, monkeypatch, line="\n" * 5)
    assert zero.main(["--mapping", str(paths[0])]) == 0
    mapping = json.loads(paths[0].read_text())
    for index, (_, _, ids) in enumerate(zero.FINGER_GROUPS, start=1):
        for mid in ids:
            assert mapping["motors"][str(mid)]["zero_deg"] == pytest.approx(mid * 10.0 + index * 0.2)
    assert hand.viewed == [paths[0].resolve()]


@pytest.mark.parametrize("invalid", [None, float("nan")])
def test_invalid_full_hand_feedback_preserves_calibration_and_skips_viewer(tmp_path, monkeypatch, invalid):
    paths, _, hand = prepare(tmp_path, monkeypatch, line="")
    files = [paths[0], paths[0].with_name("joint_calibration.json")]
    before = {path: path.read_bytes() for path in files}
    angles = {mid: 1.0 for mid in range(1, 13)}
    if invalid is None:
        del angles[12]
    else:
        angles[12] = invalid
    monkeypatch.setattr(hand, "get_all_angles", lambda: angles)
    with pytest.raises(RuntimeError, match="ID 12"):
        zero.main(["--mapping", str(paths[0]), "--all"])
    assert {path: path.read_bytes() for path in files} == before
    assert hand.closed and not hand.viewed


def test_viewer_failure_reports_saved_calibration(tmp_path, monkeypatch, capsys):
    from motorhand_teleop import encos_live_viewer

    paths, _, hand = prepare(tmp_path, monkeypatch, line="")

    def fail_viewer(*, mapping_path):
        assert hand.closed
        raise RuntimeError("显示环境不可用")

    monkeypatch.setattr(encos_live_viewer, "run_live_viewer", fail_viewer)
    assert zero.main(["--mapping", str(paths[0]), "--all"]) == 1
    assert json.loads(paths[0].read_text())["motors"]["12"]["zero_deg"] == pytest.approx(120.1)
    assert "零位已保存" in capsys.readouterr().err
