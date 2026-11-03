import copy
import io
import json

import pytest

from wujihand2_force_glove import encos_zero_step as zero
from wujihand2_force_glove.domain import GLOVE_JOINTS


class ReadOnlyHand:
    def __init__(self):
        self.read_count = 0
        self.closed = False

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


def test_eof_is_not_confirmation_and_does_not_save(tmp_path, monkeypatch):
    paths, _, hand = prepare(tmp_path, monkeypatch, line="")
    files = paths + [path.with_name("joint_calibration.json") for path in paths]
    before = {path: path.read_bytes() for path in files}
    with pytest.raises(EOFError, match="未确认零位"):
        zero.main(["--mapping", str(paths[0]), "--sync-mapping", str(paths[1]), "--finger", "index"])
    assert {path: path.read_bytes() for path in files} == before
    assert hand.closed
