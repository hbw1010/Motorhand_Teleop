"""按手指分组、回车确认的Encos软件零位标定工具。

该工具只读取反馈并记录 zero_deg，不发送位置/MIT目标，也不固化电机硬件零位。
默认依次标五指；--finger可只标一指，--sync-mapping同步另一项目的同一指。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import select
import sys
from typing import Any, Mapping

from encos_hand import EncosHand


MOTOR_IDS = tuple(range(1, 13))
FINGER_GROUPS = (
    ("thumb", "拇指", (1, 2, 3, 4)),
    ("index", "食指", (5, 6)),
    ("middle", "中指", (7, 8)),
    ("ring", "无名指", (9, 10)),
    ("pinky", "小拇指", (11, 12)),
)
_DEFAULT_MAPPING = (
    Path(__file__).resolve().parents[2]
    / "hardware"
    / "encos_dexhand"
    / "configs"
    / "glove_mapping.json"
)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_mapping(data: Mapping[str, Any]) -> None:
    motors = data.get("motors")
    if not isinstance(motors, Mapping):
        raise ValueError("glove_mapping.json缺少motors对象")
    for motor_id in MOTOR_IDS:
        config = motors.get(str(motor_id))
        if not isinstance(config, Mapping) or "joint" not in config:
            raise ValueError(f"缺少电机{motor_id}的映射配置")


def _write_calibration(
    mapping_path: Path,
    mapping: dict[str, Any],
    finger_key: str,
    values: Mapping[int, float],
) -> None:
    for motor_id, zero_deg in values.items():
        config = mapping["motors"][str(motor_id)]
        old_zero = float(config["zero_deg"])
        config["min_deg"] = float(zero_deg) + (float(config["min_deg"]) - old_zero)
        config["max_deg"] = float(zero_deg) + (float(config["max_deg"]) - old_zero)
        config["zero_deg"] = float(zero_deg)
    completed = set(mapping.get("zero_calibrated_fingers", []))
    completed.add(finger_key)
    mapping["zero_calibrated_fingers"] = [
        name for name, _, _ in FINGER_GROUPS if name in completed
    ]
    mapping_path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    calibration_path = mapping_path.with_name("joint_calibration.json")
    calibration = _load_json(calibration_path, {})
    if not isinstance(calibration, dict):
        calibration = {}
    for motor_id, zero_deg in values.items():
        entry = calibration.get(str(motor_id), {})
        if not isinstance(entry, dict):
            entry = {}
        config = mapping["motors"][str(motor_id)]
        entry.update(zero=float(zero_deg), min=config["min_deg"], max=config["max_deg"])
        calibration[str(motor_id)] = entry
    calibration_path.write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encos按手指逐组回车确认软件零位")
    parser.add_argument("--mapping", type=Path, default=_DEFAULT_MAPPING)
    parser.add_argument("--finger", choices=[name for name, _, _ in FINGER_GROUPS])
    parser.add_argument("--sync-mapping", type=Path, action="append", default=[])
    return parser


def _wait_for_enter(
    hand: EncosHand,
    finger_name: str,
    motor_ids: tuple[int, ...],
) -> dict[int, float]:
    print(f"请摆好{finger_name}零位；当前角度会持续刷新，确认后按Enter。")
    while True:
        angles = hand.get_all_angles()
        if any(
            motor_id not in angles or not math.isfinite(float(angles[motor_id]))
            for motor_id in motor_ids
        ):
            raise RuntimeError(f"{finger_name}反馈角度不完整或无效")
        display = " | ".join(
            f"ID {motor_id}: {float(angles[motor_id]): .3f}°"
            for motor_id in motor_ids
            if motor_id in angles
        )
        print(f"\r当前角度: {display}    ", end="", flush=True)
        ready, _, _ = select.select([sys.stdin], [], [], 0.2)
        if ready:
            if not sys.stdin.readline():
                raise EOFError("输入已关闭，未确认零位，不保存当前手指")
            print()
            return dict(hand.get_all_angles())


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mappings = []
    for path in dict.fromkeys(path.resolve() for path in [args.mapping, *args.sync_mapping]):
        mapping = _load_json(path, {})
        if not isinstance(mapping, dict):
            raise ValueError("映射配置必须是JSON对象")
        _validate_mapping(mapping)
        mappings.append((path, mapping))
    groups = [
        group for group in FINGER_GROUPS if args.finger is None or group[0] == args.finger
    ]
    for _, _, motor_ids in groups:
        for motor_id in motor_ids:
            configs = [mapping["motors"][str(motor_id)] for _, mapping in mappings]
            for config in configs:
                numbers = [float(config[key]) for key in ("zero_deg", "min_deg", "max_deg")]
                if not all(math.isfinite(value) for value in numbers) or numbers[1] >= numbers[2]:
                    raise ValueError(f"ID {motor_id}原始零位或限位无效")
            if any(
                tuple(config[key] for key in ("joint", "sign", "scale"))
                != tuple(configs[0][key] for key in ("joint", "sign", "scale"))
                for config in configs[1:]
            ):
                raise ValueError(f"ID {motor_id}两项目轴名、方向或比例不一致，不能同步标零")

    hand = EncosHand()
    try:
        hand.connect()
        found = hand.scan()
        online = tuple(sorted(hand.get_online_ids()))
        print(f"online_ids={online}")
        if online != MOTOR_IDS:
            raise RuntimeError(f"必须12轴全在线才能逐组标零，当前在线={online}，扫描={found}")

        feedbacks = hand.get_all_feedbacks()
        errors = {
            motor_id: values.get("err_str", values.get("err_code"))
            for motor_id, values in feedbacks.items()
            if int(values.get("err_code", 0)) != 0
        }
        if errors:
            raise RuntimeError(f"标零前存在电机报警: {errors}")

        print("只记录软件zero_deg；不会发送位置/MIT命令，也不会固化硬件零位。")
        print("请保持手套电机失能；摆好当前手指零位后直接按Enter确认。")
        for finger_key, finger_name, motor_ids in groups:
            print(f"\n[{finger_name}] ID {motor_ids}")
            captured = _wait_for_enter(hand, finger_name, motor_ids)
            values: dict[int, float] = {}
            for motor_id in motor_ids:
                value = captured.get(motor_id)
                if value is None or not math.isfinite(float(value)):
                    raise RuntimeError(f"ID {motor_id}确认后缺少有效角度反馈")
                values[motor_id] = float(value)
            for path, mapping in mappings:
                _write_calibration(path, mapping, finger_key, values)
            print(f"已保存{finger_name} zero_deg: {values}")

        print("ZERO_STEP_OK")
        return 0
    except KeyboardInterrupt:
        print("ZERO_STEP_ABORTED_BY_USER")
        return 130
    finally:
        if hand.get_online_ids():
            for motor_id in hand.get_online_ids():
                try:
                    hand.enable(motor_id, False)
                except BaseException:
                    pass
        hand.close()


if __name__ == "__main__":
    raise SystemExit(main())
