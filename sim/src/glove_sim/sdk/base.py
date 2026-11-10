from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class GloveState:
    """One glove sample.

    ``angles_rad`` is the simulated motor-joint angle after applying the
    encoder zero/sign/scale mapping.  ``raw_deg`` and ``current_a`` are the
    original SDK readings and are recorded as measured values.
    """

    timestamp_ns: int
    angles_rad: np.ndarray
    raw_deg: dict[int, float]
    current_a: dict[int, float] = field(default_factory=dict)
    temperature_c: dict[int, float] = field(default_factory=dict)
    errors: dict[int, str] = field(default_factory=dict)
    online_ids: tuple[int, ...] = ()
    source: str = "fake"


class GloveBackend(Protocol):
    def open(self) -> GloveState: ...
    def read(self) -> GloveState: ...
    def enable(self, enabled: bool = True) -> None: ...
    def command_torque(self, motor_id: int, torque_nm: float, *,
                       deadline_ns: int | None = None) -> float | None:
        """Send a torque; return the value put on the motor (None = nothing sent)."""
        ...
    # Optional: def command_impedance(motor_id, cmd, q_model, *, deadline_ns) -> float | None
    def command_position(self, motor_id: int, target_deg: float, *, speed_rpm: float = 5.0,
                         current_a: float = 0.8, deadline_ns: int | None = None) -> None: ...
    def close(self) -> None: ...
