from __future__ import annotations

import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..mapping import GloveMapping

FINGER_GROUPS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("thumb", "拇指", (1, 2, 3, 4)),
    ("index", "食指", (5, 6)),
    ("middle", "中指", (7, 8)),
    ("ring", "无名指", (9, 10)),
    ("pinky", "小拇指", (11, 12)),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def backup_mapping(mapping_path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = backup_dir / f"glove_mapping-{stamp}.json"
    shutil.copy2(mapping_path, target)
    return target


def write_zero(mapping_path: Path, values: dict[int, float], finger_keys: tuple[str, ...]) -> None:
    """Write zero_deg while preserving each motor's relative soft limits.

    Mirrors the extracted ``encos_zero_step.py`` behaviour: no position or MIT
    command is sent; only the software calibration is updated.
    """
    data = _read_json(mapping_path)
    motors = data.get("motors")
    if not isinstance(motors, dict):
        raise ValueError("glove_mapping.json 缺少 motors")
    for mid, zero in values.items():
        entry = motors.get(str(mid))
        if not isinstance(entry, dict):
            raise ValueError(f"缺少电机 {mid}")
        old = float(entry["zero_deg"])
        entry["min_deg"] = float(zero) + (float(entry["min_deg"]) - old)
        entry["max_deg"] = float(zero) + (float(entry["max_deg"]) - old)
        entry["zero_deg"] = float(zero)
    completed = set(data.get("zero_calibrated_fingers", []))
    completed.update(finger_keys)
    data["zero_calibrated_fingers"] = [name for name, _, _ in FINGER_GROUPS if name in completed]
    mapping_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    calibration_path = mapping_path.with_name("joint_calibration.json")
    calibration = _read_json(calibration_path) if calibration_path.exists() else {}
    if not isinstance(calibration, dict):
        calibration = {}
    for mid, zero in values.items():
        entry = calibration.get(str(mid), {})
        if not isinstance(entry, dict):
            entry = {}
        motor = motors[str(mid)]
        entry.update(zero=float(zero), min=float(motor["min_deg"]), max=float(motor["max_deg"]))
        calibration[str(mid)] = entry
    calibration_path.write_text(json.dumps(calibration, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def zero_online_glove(mapping_path: Path, *, finger: str | None, all_motors: bool,
                      backup_dir: Path, assume_yes: bool = False, glove_kwargs: dict | None = None) -> int:
    from .real import RealGlove, RealGloveError

    mapping = GloveMapping.load(mapping_path)
    glove = RealGlove(mapping, **(glove_kwargs or {}))
    try:
        state = glove.open()
        online = state.online_ids
        if online != tuple(range(1, 13)):
            raise RealGloveError(f"必须 12 轴在线，当前={online}")
        if state.errors:
            raise RealGloveError(f"标零前存在报警: {state.errors}")
        backup = backup_mapping(mapping_path, backup_dir)
        print(f"已备份标定: {backup}")
        groups = [g for g in FINGER_GROUPS if finger is None or g[0] == finger]
        if not all_motors:
            print("逐指置零：摆好当前手指零位后按 Enter；输入 q 取消。")
        captures: list[tuple[tuple[str, ...], tuple[int, ...], dict[int, float]]] = []
        for key, label, ids in groups:
            angles = glove.read().raw_deg
            if all_motors:
                captures.append(((key,), ids, {mid: angles[mid] for mid in ids}))
            else:
                print(f"\n[{label}] 当前角度:")
                while True:
                    angles = glove.read().raw_deg
                    print("  " + " | ".join(f"ID {mid}: {angles[mid]: .3f}°" for mid in ids), end="\r")
                    answer = input().strip().lower()
                    if answer == "q":
                        print("\n已取消，未写入新零位。")
                        return 1
                    captures.append(((key,), ids, {mid: float(angles[mid]) for mid in ids}))
                    break
        for keys, ids, values in captures:
            write_zero(mapping_path, values, keys)
            print(f"已保存 {keys}: {values}")
        print("ZERO_OK")
        print("提醒：置零只写软件 zero_deg，不固化电机硬件零位；请随后做只读方向/限位检查。")
        return 0
    finally:
        glove.close()
