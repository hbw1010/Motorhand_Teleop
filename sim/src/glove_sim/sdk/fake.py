from __future__ import annotations

import math
import time

import numpy as np

from ..mapping import GloveMapping
from .base import GloveState

# A slow open/close demonstration used only in offline mode.  Values are
# simulated joint angles (rad); they are not calibrated real-glove commands.
_GRASP = np.array(
    [
        0.20, 0.15, 0.25, 0.25,      # thumb
        0.25, 0.80,                   # index
        0.25, 0.80,                   # middle
        0.25, 0.80,                   # ring
        0.25, 0.80,                   # pinky
    ],
    dtype=np.float64,
)


class FakeGlove:
    """Deterministic offline backend for smoke tests and demos.

    It is intentionally not a physics device: ``command_torque`` records the
    requested nominal torque and the reported current is a simulated proxy.
    """

    def __init__(self, mapping: GloveMapping, *, demo: bool = True, period_s: float = 6.0,
                 amplitude: float = 1.0, seed: int = 7) -> None:
        self.mapping = mapping
        self.demo = demo
        self.period_s = float(period_s)
        self.amplitude = float(amplitude)
        self._rng = np.random.default_rng(seed)
        self._q = np.zeros(12, dtype=np.float64)
        self._start = time.monotonic()
        self._torque = np.zeros(12, dtype=np.float64)
        self._enabled = False
        self._noise = 0.0
        self._hold: np.ndarray | None = None

    def open(self) -> GloveState:
        self._start = time.monotonic()
        self._enabled = False
        return self.read()

    def set_demo(self, enabled: bool) -> None:
        """Pause the demo at the current pose so objects are easy to place."""
        enabled = bool(enabled)
        if not enabled and self.demo:
            self._hold = np.asarray(self._q, dtype=np.float64).copy()
        if enabled:
            self._hold = None
            self._start = time.monotonic()
        self.demo = enabled

    def _demo_target(self) -> np.ndarray:
        if not self.demo:
            if self._hold is not None:
                return self._hold.copy()
            return np.zeros(12, dtype=np.float64)
        t = (time.monotonic() - self._start) % self.period_s
        # 0..1 s open, 1..3 s close, 3..4.5 s hold, 4.5..6 s open.
        if t < 1.0:
            p = 0.0
        elif t < 3.0:
            p = (t - 1.0) / 2.0
        elif t < 4.5:
            p = 1.0
        else:
            p = max(0.0, 1.0 - (t - 4.5) / 1.5)
        # smoothstep
        p = p * p * (3.0 - 2.0 * p)
        return _GRASP * (p * self.amplitude)

    def read(self) -> GloveState:
        target = self._demo_target()
        self._q += 0.18 * (target - self._q)
        if self._noise:
            self._q += self._rng.normal(0.0, self._noise, size=12)
        raw = self.mapping.model_rad_to_raw_deg([float(v) for v in self._q])
        current = {
            motor_id: float(self._torque[motor_id - 1] / 0.035) if motor_id - 1 < len(self._torque) else 0.0
            for motor_id in range(1, 13)
        }
        return GloveState(
            timestamp_ns=time.monotonic_ns(),
            angles_rad=self._q.copy(),
            raw_deg=raw,
            current_a=current,
            temperature_c={motor_id: 30.0 for motor_id in range(1, 13)},
            errors={},
            online_ids=tuple(range(1, 13)),
            source="fake",
        )

    def enable(self, enabled: bool = True) -> None:
        self._enabled = bool(enabled)
        if not enabled:
            self._torque[:] = 0.0

    def command_torque(self, motor_id: int, torque_nm: float, *,
                       deadline_ns: int | None = None) -> float | None:
        if 1 <= motor_id <= 12 and self._enabled:
            self._torque[motor_id - 1] = float(torque_nm)
            return float(torque_nm)
        return None

    def command_impedance(self, motor_id: int, cmd, q_model: float, *,
                          deadline_ns: int | None = None) -> float | None:
        """Record the torque the driver PD would produce now (model frame)."""
        if not (1 <= motor_id <= 12 and self._enabled):
            return None
        tau = float(cmd.k) * (float(cmd.target_rad) - float(q_model)) + float(cmd.tau_ff)
        self._torque[motor_id - 1] = tau
        return tau

    def command_position(self, motor_id: int, target_deg: float, *, speed_rpm: float = 5.0,
                         current_a: float = 0.8, deadline_ns: int | None = None) -> None:
        if not (1 <= motor_id <= 12):
            return
        idx = motor_id - 1
        motor = self.mapping.motors[idx]
        target = math.radians((float(target_deg) - motor.zero_deg) * motor.sign * motor.scale)
        self._q[idx] = target

    def close(self) -> None:
        self.enable(False)
