from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

# Motor ID -> MuJoCo joint in the vE3.3 closed-chain model.  The joint names
# in glove_mapping.json are the semantic names; this table fixes the linkage
# between the physical motor and the simulated motor joint.
MODEL_MOTOR_JOINTS: tuple[str, ...] = (
    "thumb_cmc_flex_motor_joint",
    "thumb_cmc_abd_motor_joint",
    "thumb_mcp_motor_joint",
    "thumb_ip_motor_joint",
    "index_mcp_motor_joint",
    "index_pip_dip_motor_joint",
    "middle_mcp_motor_joint",
    "middle_pip_dip_motor_joint",
    "ring_mcp_motor_joint",
    "ring_pip_dip_motor_joint",
    "pinky_mcp_motor_joint",
    "pinky_pip_dip_motor_joint",
)

# Motor ID -> finger used for per-finger contact summaries.
MOTOR_FINGER: tuple[str, ...] = (
    "thumb", "thumb", "thumb", "thumb",
    "index", "index",
    "middle", "middle",
    "ring", "ring",
    "pinky", "pinky",
)

FINGERS: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class MotorCalibration:
    motor_id: int
    joint: str
    model_joint: str
    zero_deg: float
    sign: float
    scale: float
    min_deg: float
    max_deg: float
    current_limit_a: float

    @property
    def min_rad(self) -> float:
        return math.radians(self.min_deg * self.sign * self.scale) if self.sign >= 0 else math.radians(self.max_deg * self.sign * self.scale)

    def raw_to_model_rad(self, raw_deg: float) -> float:
        """Encoder degrees (raw frame) -> simulated motor-joint radians.

        The mapping follows the extracted glove_mapping.json convention:
        (raw - zero) * sign * scale.  Direction and zero are displayable and
        are recorded in every telemetry row.
        """
        return math.radians((float(raw_deg) - self.zero_deg) * self.sign * self.scale)

    def model_rad_limits(self) -> tuple[float, float]:
        a = math.radians((self.min_deg - self.zero_deg) * self.sign * self.scale)
        b = math.radians((self.max_deg - self.zero_deg) * self.sign * self.scale)
        return (min(a, b), max(a, b))


class GloveMapping:
    def __init__(self, data: dict, source: str):
        motors = data.get("motors")
        if not isinstance(motors, dict):
            raise ValueError("glove_mapping.json 缺少 motors")
        self.source = source
        self.raw = data
        self.motors: list[MotorCalibration] = []
        for motor_id in range(1, 13):
            entry = motors.get(str(motor_id))
            if not isinstance(entry, dict):
                raise ValueError(f"缺少电机 {motor_id} 的映射")
            self.motors.append(
                MotorCalibration(
                    motor_id=motor_id,
                    joint=str(entry["joint"]),
                    model_joint=MODEL_MOTOR_JOINTS[motor_id - 1],
                    zero_deg=float(entry["zero_deg"]),
                    sign=float(entry.get("sign", 1.0)),
                    scale=float(entry.get("scale", 1.0)),
                    min_deg=float(entry["min_deg"]),
                    max_deg=float(entry["max_deg"]),
                    current_limit_a=float(entry.get("current_limit_a", 0.8)),
                )
            )

    @classmethod
    def load(cls, path: str | Path) -> "GloveMapping":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(data, str(p))

    @property
    def model_joint_names(self) -> tuple[str, ...]:
        return tuple(m.model_joint for m in self.motors)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(m.joint for m in self.motors)

    def raw_deg_to_model_rad(self, raw_deg: dict[int, float]) -> list[float]:
        out: list[float] = []
        for motor in self.motors:
            value = motor.raw_to_model_rad(raw_deg[motor.motor_id])
            lo, hi = motor.model_rad_limits()
            out.append(max(lo, min(hi, value)))
        return out

    def model_rad_to_raw_deg(self, q_rad: list[float]) -> dict[int, float]:
        out: dict[int, float] = {}
        for motor, q in zip(self.motors, q_rad):
            out[motor.motor_id] = motor.zero_deg + math.degrees(q) / (motor.sign * motor.scale)
        return out
