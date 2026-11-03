from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CurrentLimits:
    max_abs_a: np.ndarray
    max_slew_a_s: np.ndarray

    def __post_init__(self) -> None:
        for name in ("max_abs_a", "max_slew_a_s"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (12,) or not np.all(np.isfinite(value)) or np.any(value <= 0):
                raise ValueError(f"{name}必须是12维正有限数组")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


class SafetyLimiter:
    def __init__(self, limits: CurrentLimits):
        self.limits = limits
        self._previous = np.zeros(12, dtype=np.float64)
        self._faulted = False

    def trip(self) -> None:
        self._faulted = True
        self._previous.fill(0.0)

    def reset(self) -> None:
        self._faulted = False
        self._previous.fill(0.0)

    def apply(self, candidate_a: np.ndarray, *, dt_s: float) -> np.ndarray:
        if self._faulted:
            raise RuntimeError("触觉限幅器已锁存故障，需要显式 reset")
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s必须为正有限值")
        candidate = np.asarray(candidate_a, dtype=np.float64)
        if candidate.shape != (12,) or not np.all(np.isfinite(candidate)):
            raise ValueError("candidate_a必须是有限的12维数组")
        clipped = np.clip(candidate, -self.limits.max_abs_a, self.limits.max_abs_a)
        delta = self.limits.max_slew_a_s * dt_s
        output = self._previous + np.clip(clipped - self._previous, -delta, delta)
        self._previous = output
        return output.copy()
