"""Glove state sources and opt-in Encos nominal torque output.

No device is opened by importing this module. The hardware driver's torque command
is a nominal SDK value, not measured force or measured motor torque.
"""
from __future__ import annotations

import math
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from motorhand_teleop.domain import GLOVE_JOINTS
from motorhand_teleop.real_hand.motor_hand import EncosGloveHardwareDriver
from motorhand_teleop.ve33_adapter import Ve33GloveModel

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "assets" / "hand_motor_ve33"
MAX_FEEDBACK_AGE_NS = 100_000_000


@dataclass(frozen=True)
class GloveSample:
    received_ns: int
    motor_positions_rad: Mapping[str, float]
    tree_positions_rad: Mapping[str, float]
    joint_positions_rad: Mapping[str, float]
    motor_current_a: list[float | None]


class DemoGlove:
    """Repeatable, non-hardware source using the real vE3.3 closure solver."""

    def __init__(self, side: str = "right", scenario: str = "press"):
        self.model = Ve33GloveModel(side, asset_root=ASSET_ROOT)
        if scenario not in ("press", "grasp"):
            raise ValueError("scenario must be press or grasp")
        self.scenario = scenario
        self._start_ns = time.monotonic_ns()

    def reset_clock(self) -> None:
        """Start the scripted motion after model and display initialization."""
        self._start_ns = time.monotonic_ns()

    def read(self) -> GloveSample:
        now = time.monotonic_ns()
        elapsed = (now - self._start_ns) / 1e9
        # Slow flexion and release. The closure solver rejects unreachable poses.
        amplitude = 0.3 if self.scenario == "grasp" else 0.25
        flex = amplitude * (1.0 - math.cos(2.0 * math.pi * elapsed / 6.0))
        values = {name: 0.0 for name in GLOVE_JOINTS}
        for name in ("index_mcp_flex", "index_pip", "middle_mcp_flex", "middle_pip",
                     "ring_mcp_flex", "ring_pip", "pinky_mcp_flex", "pinky_pip"):
            values[name] = flex
        if self.scenario == "grasp":
            values["thumb_cmc_flex"] = flex
            values["thumb_mcp"] = values["thumb_ip"] = flex
        else:
            values["thumb_mcp"] = values["thumb_ip"] = flex * 0.4
        pose = self.model.evaluate_raw(values)
        return GloveSample(now, pose.motor_positions, pose.all_positions,
                           {name: pose.joint_positions[f"{name}_joint"] for name in GLOVE_JOINTS},
                           [None] * 12)

    def close(self) -> None:
        pass


class HardwareGlove:
    """Read-only by default; explicit output opt-in may physically home the glove."""

    def __init__(self, side: str, mapping: str | None, execute: bool,
                 acknowledge_homing: bool, raw_torque_limit_nm: float | None,
                 ports: tuple[str, ...] | list[str] | None = None):
        if execute and not acknowledge_homing:
            raise ValueError("真实出力需要 --acknowledge-homing：使能时手套会实际回零")
        if execute and (raw_torque_limit_nm is None or not math.isfinite(raw_torque_limit_nm)
                        or not 0.01 <= raw_torque_limit_nm <= 0.2):
            raise ValueError("真实出力须显式指定 --raw-torque-limit-nm，范围 0.01–0.2 名义 Nm")
        if side == "left":
            if mapping is None:
                raise ValueError("左手必须显式提供经过左手标定的 --mapping")
            left_mapping = json.loads(Path(mapping).read_text(encoding="utf-8"))
            if not isinstance(left_mapping, dict) or left_mapping.get("hand") != "left":
                raise ValueError("左手映射必须包含 hand='left'，禁止复用默认右手映射")
        if execute and ports is None:
            raise ValueError("真实出力必须使用 --ports 显式指定本手套的三个 CAN 适配器")
        hand_factory = None
        if ports is not None:
            if len(ports) != 3 or len({str(Path(p).resolve()) for p in ports}) != 3:
                raise ValueError("--ports 必须指定三个不同的 CAN 适配器设备")
            port_map = {i + 1: str(Path(port).resolve()) for i, port in enumerate(ports)}
            def hand_factory():
                from encos_hand import EncosHand
                return EncosHand(port_map=port_map)
        self.execute = execute
        self.model = Ve33GloveModel(side, asset_root=ASSET_ROOT)
        limits = None if not execute else {mid: raw_torque_limit_nm for mid in range(1, 13)}
        self.driver = EncosGloveHardwareDriver(
            hand=side, asset_root=ASSET_ROOT, config_path=mapping,
            force_torque_limits_nm=limits, hand_factory=hand_factory,
        )
        self._started = False
        self._last_confirmed = None
        self._output_started_once = False

    def connect(self) -> None:
        try:
            self.driver.connect()
            if self.driver.online_ids != tuple(range(1, 13)):
                raise RuntimeError(f"需要全部 12 轴在线，当前 {self.driver.online_ids}")
            self._started = True
        except BaseException:
            self.driver.close()
            raise

    def enable_output(self) -> None:
        if not self.execute or not self._started:
            raise RuntimeError("真实电机输出未授权或设备未连接")
        if self._output_started_once:
            self.driver.resume_force_feedback()
        else:
            # The first enable may physically home the glove.
            self.driver.enable_force_feedback()
            self._output_started_once = True

    def pause_output(self) -> None:
        """Request zero-current mode before moving an object in the simulation."""
        if not self.execute or not self._started or not self._output_started_once:
            raise RuntimeError("真实电机输出尚未启用")
        self.driver.disable()

    @property
    def torque_limits_nm(self) -> np.ndarray:
        limits = self.driver.model_force_torque_limits_nm
        return np.asarray([limits[name] for name in self.model.motor_names], dtype=float)

    def read(self) -> GloveSample:
        state = self.driver.read_state()
        current_by_name = {}
        for mid, config in self.driver.motors.items():
            value = state.feedback_by_motor.get(mid, {}).get("current_a")
            current_by_name[self.model.joint_motor_names[config.joint]] = (
                None if value is None else float(value)
            )
        return GloveSample(state.received_ns, state.motor_positions_rad, state.tree_positions_rad,
                           state.pose.joints,
                           [current_by_name[name] for name in self.model.motor_names])

    def send(self, torque_nm: np.ndarray, input_time_ns: int) -> tuple[float, ...]:
        if not self.execute or not self._started:
            raise RuntimeError("真实电机输出未启用")
        now = time.monotonic_ns()
        age = now - input_time_ns
        if not 0 <= age < MAX_FEEDBACK_AGE_NS:
            raise TimeoutError(f"手套输入数据已过期或时钟域错误：{age / 1e6:.1f} ms")
        # Lease is anchored to the oldest source frame, never extended after computation.
        result = self.driver.send_force_torque(tuple(float(x) for x in torque_nm),
                                               deadline_ns=input_time_ns + MAX_FEEDBACK_AGE_NS)
        self._last_confirmed = result
        return result

    def close(self) -> None:
        self.driver.close()  # disables if this process acquired the output
        self._started = False
