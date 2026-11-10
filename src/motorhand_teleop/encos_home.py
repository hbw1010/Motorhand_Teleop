"""Encos 12轴统一使能回零工具。

回零目标取自 glove_mapping.json 的 zero_deg；判定使用弧度误差。
回零目标电流固定为J1=1.5A、其余轴1.0A；超时、报警、节点缺失都会先失能再退出。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

from encos_hand import EncosHand


MOTOR_IDS = tuple(range(1, 13))
HOME_CURRENT_RULE = {1: 1.5, **{motor_id: 1.0 for motor_id in range(2, 13)}}
_DEFAULT_MAPPING = (
    Path(__file__).resolve().parents[2]
    / "hardware"
    / "encos_dexhand"
    / "configs"
    / "glove_mapping.json"
)


def _load_targets(path: Path) -> dict[int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    motors = data.get("motors")
    if not isinstance(motors, Mapping):
        raise ValueError("glove_mapping.json缺少motors对象")
    targets: dict[int, float] = {}
    for motor_id in MOTOR_IDS:
        config = motors.get(str(motor_id))
        if not isinstance(config, Mapping):
            raise ValueError(f"缺少电机{motor_id}的映射配置")
        target = float(config["zero_deg"])
        if not math.isfinite(target):
            raise ValueError(f"电机{motor_id}的zero_deg不是有限数")
        targets[motor_id] = target
    return targets


def _disable_all(hand: EncosHand, motor_ids: tuple[int, ...]) -> None:
    for motor_id in motor_ids:
        try:
            hand.enable(motor_id, False)
        except BaseException:
            pass
    # 给工作线程一个发送失能保持帧的时间，随后再关闭CAN线程。
    time.sleep(0.1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encos 12轴统一使能回零")
    parser.add_argument("--mapping", type=Path, default=_DEFAULT_MAPPING)
    parser.add_argument("--tolerance-rad", type=float, default=0.15)
    parser.add_argument("--speed-rpm", type=float, default=5.0)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--poll-s", type=float, default=0.05)
    return parser


def _load_home_currents(path: Path) -> dict[int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    configured_default = float(data.get("home_current_a", 1.0))
    configured_overrides = data.get("home_current_a_by_motor", {})
    if not isinstance(configured_overrides, Mapping):
        raise ValueError("home_current_a_by_motor必须是对象")
    configured = {
        motor_id: float(configured_overrides.get(str(motor_id), configured_default))
        for motor_id in MOTOR_IDS
    }
    for motor_id, expected in HOME_CURRENT_RULE.items():
        if not math.isclose(configured[motor_id], expected, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"J{motor_id}回零电流规则固定为{expected:.1f}A，配置为{configured[motor_id]:.3f}A"
            )
    return dict(HOME_CURRENT_RULE)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    home_currents = _load_home_currents(args.mapping)
    if (
        args.tolerance_rad <= 0.0
        or args.speed_rpm <= 0.0
        or args.timeout_s <= 0.0
        or args.poll_s <= 0.0
    ):
        raise ValueError("容差、速度、超时和轮询周期必须为正数")
    targets = _load_targets(args.mapping)
    tolerance_deg = math.degrees(args.tolerance_rad)
    hand = EncosHand()
    success = False
    try:
        hand.connect()
        found = hand.scan()
        online = tuple(sorted(hand.get_online_ids()))
        print(f"online_ids={online}")
        if online != MOTOR_IDS:
            raise RuntimeError(f"必须12轴全在线才能统一回零，当前在线={online}，扫描={found}")

        feedbacks = hand.get_all_feedbacks()
        errors = {
            motor_id: values.get("err_str", values.get("err_code"))
            for motor_id, values in feedbacks.items()
            if int(values.get("err_code", 0)) != 0
        }
        if errors:
            raise RuntimeError(f"回零前存在电机报警: {errors}")

        print(f"targets_deg={targets}")
        print(
            f"enable=True, speed_rpm={args.speed_rpm}, home_current_a={home_currents}, "
            f"tolerance={args.tolerance_rad:.6f} rad ({tolerance_deg:.3f} deg)"
        )
        current_angles = hand.get_all_angles()
        if not all(motor_id in current_angles for motor_id in MOTOR_IDS):
            raise RuntimeError(f"使能前反馈不完整: {sorted(current_angles)}")
        for motor_id in MOTOR_IDS:
            hand.set_position(
                motor_id=motor_id,
                target_deg=float(current_angles[motor_id]),
                speed_rpm=0.0,
                current_a=home_currents[motor_id],
            )
        print("enable_stage=hold_current")
        time.sleep(max(0.2, args.poll_s * 4.0))
        for motor_id, target_deg in targets.items():
            hand.set_position(
                motor_id=motor_id,
                target_deg=target_deg,
                speed_rpm=args.speed_rpm,
                current_a=home_currents[motor_id],
            )

        deadline = time.monotonic() + args.timeout_s
        last_print = 0.0
        while True:
            angles = hand.get_all_angles()
            feedbacks = hand.get_all_feedbacks()
            errors = {
                motor_id: values.get("err_str", values.get("err_code"))
                for motor_id, values in feedbacks.items()
                if int(values.get("err_code", 0)) != 0
            }
            if errors:
                raise RuntimeError(f"回零过程中出现电机报警: {errors}")
            if not all(motor_id in angles for motor_id in MOTOR_IDS):
                raise RuntimeError(f"回零过程中反馈不完整: {sorted(angles)}")
            errors_rad = {
                motor_id: abs(math.radians(float(angles[motor_id]) - targets[motor_id]))
                for motor_id in MOTOR_IDS
            }
            max_error = max(errors_rad.values())
            now = time.monotonic()
            if now - last_print >= 0.25:
                print(
                    "errors_rad="
                    + str({motor_id: round(error, 5) for motor_id, error in errors_rad.items()})
                    + f", max={max_error:.5f}"
                )
                last_print = now
            if max_error <= args.tolerance_rad:
                success = True
                print("HOME_OK")
                break
            if now >= deadline:
                raise TimeoutError(f"统一回零超时，max_error_rad={max_error:.6f}")
            time.sleep(args.poll_s)
    except KeyboardInterrupt:
        print("HOME_ABORTED_BY_USER")
        return 130
    finally:
        if hand.get_online_ids():
            _disable_all(hand, tuple(sorted(hand.get_online_ids())))
        hand.close()

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
