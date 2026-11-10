"""Driver-side impedance feedback and friction compensation.

The proxy ("god-object") spring used to be evaluated here in Python and sent
as MIT feed-forward torque (kp=kd=0).  Each motor only receives a new frame
every ~20 ms (4 motors x 5 ms CAN slots per bus), so a stiff spring computed
off-board goes unstable long before it feels like a wall.  Following the ARX
master-arm examples, the spring and damper now run *inside the Encos driver*
(MIT mode: tau = kp*(p_t - p) + kd*(v_t - v) + tau_ff) and this module only
plans the per-motor MIT parameters at the control rate:

* free space  : kp = kd = 0, tau_ff = friction compensation (transparency);
* in contact  : target = surface angle (proxy), kp = k, kd = b while closing.

The driver PD is two-sided and has no torque cap, so the plan

* clamps the target so k*|target - q| never exceeds the torque limit (the
  spring "slides" with the finger beyond that, like a saturated god-object);
* releases (kp = 0) one command period *before* the finger would cross back
  out of the surface, so the PD never pulls a retreating finger back in;
* applies damping only while the finger is still pressing in.

All quantities are in model-joint coordinates (closing = +); the SDK converts
to motor axes.  Hardware facts behind the defaults (measured 2026-11-10, glove
not worn): MIT position frame equals the reported angle on all 12 motors,
positive MIT torque increases the raw angle on all 12 motors, kinetic
friction 0.6-1.0 Nm at 5-10 deg/s, static >= 0.4-1 Nm, no PD oscillation for
kp <= 60 Nm/rad, kd <= 0.3.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ImpedanceCommand:
    """One motor's MIT command in model-joint coordinates."""

    target_rad: float
    k: float = 0.0       # Nm/rad
    b: float = 0.0       # Nm*s/rad
    tau_ff: float = 0.0  # Nm

    def torque(self, q: float, dq: float = 0.0) -> float:
        """Torque the driver PD produces at joint angle q / velocity dq."""
        return self.k * (self.target_rad - q) - self.b * dq + self.tau_ff


def _vec(value: Any, n: int = 12) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.full(n, float(arr[0]))
    if arr.size != n:
        raise ValueError(f"expected {n} values, got {arr.size}")
    return arr


class FrictionCompensator:
    """Coulomb + viscous friction feed-forward: tau = gain*(Fc*tanh(dq/vs) + Fv*dq).

    Only kinetic friction can be compensated (static friction would need a
    force sensor to know the intended direction).  Defaults are deliberately
    below the measured kinetic friction: over-compensation makes a joint run
    away on its own.
    """

    def __init__(self, cfg: Mapping[str, Any] | None) -> None:
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.coulomb = _vec(cfg.get("coulomb_nm", 0.0))
        self.viscous = _vec(cfg.get("viscous_nm_s_per_rad", 0.0))
        self.v_scale = max(1e-4, float(cfg.get("velocity_scale_rad_s", 0.08)))
        self.v_dead = max(0.0, float(cfg.get("velocity_deadband_rad_s", 0.02)))
        self.gain = float(cfg.get("gain", 1.0))
        self.limit = _vec(cfg.get("max_nm", 0.6))

    def torque(self, dq: np.ndarray) -> np.ndarray:
        if not self.enabled or self.gain <= 0.0:
            return np.zeros(12)
        dq = np.asarray(dq, dtype=float)
        # Deadband: no assist while (nearly) at rest, so estimator noise or a
        # settling link can never start a self-sustained drift.
        moving = np.sign(dq) * np.maximum(0.0, np.abs(dq) - self.v_dead)
        tau = self.gain * (self.coulomb * np.tanh(moving / self.v_scale) + self.viscous * moving)
        return np.clip(tau, -self.limit, self.limit)


class DriverImpedancePlanner:
    """Plans MIT kp/kd/target/feed-forward per motor from the proxy blocks."""

    def __init__(self, cfg: Mapping[str, Any] | None, torque_limit_nm: Any) -> None:
        # Per-motor damping state (hysteresis), kept across ticks.
        self._damping_on = np.zeros(12, dtype=bool)
        self.configure(cfg, torque_limit_nm)

    def configure(self, cfg: Mapping[str, Any] | None, torque_limit_nm: Any) -> None:
        """(Re)read tunables; keeps the per-motor state (live UI sliders)."""
        cfg = dict(cfg or {})
        self.gain = float(cfg.get("gain", 1.0))
        self.damping_s = float(cfg.get("damping_s", 0.008))
        self.damping_max = float(cfg.get("damping_max", 0.5))
        self.k_max = float(cfg.get("k_max_nm_per_rad", 80.0))
        self.lookahead_s = float(cfg.get("release_lookahead_s", 0.02))
        self.margin = float(cfg.get("torque_margin", 0.85))
        # "soften" (default): on rebound the push is reduced by k*|v|*lookahead
        # (continuous, never pulls, dissipates rebound energy).  "release":
        # legacy hard kp=0 before the predicted crossing -- it is triggered by
        # the spring's own rebound and caused a push/release limit cycle at
        # contact (reproduced on J4: 53 toggles/3 s at kp 60).
        self.rebound = str(cfg.get("rebound_mode", "soften"))
        # Damping stays on in contact (both directions) and switches off only
        # for a clear withdrawal, with hysteresis so it cannot chatter.
        self.damp_off_vel = float(cfg.get("damping_off_velocity_rad_s", 0.15))
        self.damp_on_vel = float(cfg.get("damping_on_velocity_rad_s", 0.05))
        self.limit = _vec(torque_limit_nm)

    def plan(self, q: np.ndarray, dq: np.ndarray,
             blocked: Mapping[int, tuple[int, float, float]],
             friction: np.ndarray | None = None, *,
             ramp: float = 1.0) -> list[ImpedanceCommand]:
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)
        friction = np.zeros(12) if friction is None else np.asarray(friction, dtype=float)
        ramp = float(min(1.0, max(0.0, ramp)))
        out: list[ImpedanceCommand] = []
        for i in range(12):
            free = ImpedanceCommand(target_rad=float(q[i]), tau_ff=float(friction[i]) * ramp)
            block = blocked.get(i)
            if block is None:
                out.append(free)
                continue
            sign, surface, k = int(block[0]), float(block[1]), float(block[2])
            depth = sign * (q[i] - surface)       # > 0: finger inside the object
            press_vel = sign * dq[i]              # > 0: still pressing in
            k = min(self.k_max, max(0.0, k * self.gain)) * ramp
            if k <= 0.0 or depth <= 0.0:
                self._damping_on[i] = False
                out.append(free)
                continue
            if self.rebound == "release" and press_vel < 0.0 and depth < -press_vel * self.lookahead_s:
                self._damping_on[i] = False
                out.append(free)
                continue
            cap = self.margin * float(self.limit[i])
            # Saturated spring: the PD target never sits more than cap/k away.
            reach = min(depth, cap / k)
            if self.rebound != "release" and press_vel < 0.0:
                # Moving out: the next frame lands up to ~lookahead later, so
                # aim the spring that much further out.  The push shrinks
                # continuously to 0 at the predicted crossing -- no pull-back,
                # no on/off switching, and less push on the way out than on
                # the way in (rebound energy is dissipated).
                reach = reach + press_vel * self.lookahead_s
                if reach <= 0.0:
                    # Push already 0: drop kp now, otherwise a PD pinned at the
                    # current angle would pull the withdrawing finger back.
                    self._damping_on[i] = False
                    out.append(free)
                    continue
            target = float(q[i] - sign * reach)
            if self._damping_on[i]:
                if press_vel < -self.damp_off_vel:
                    self._damping_on[i] = False
            elif press_vel > -self.damp_on_vel:
                self._damping_on[i] = True
            b = min(self.damping_max, self.damping_s * k) if self._damping_on[i] else 0.0
            out.append(ImpedanceCommand(target_rad=target, k=k, b=b))
        return out


class VelocityEstimator:
    """Low-pass finite-difference joint velocity (rad/s).

    Each motor's angle only refreshes every ~20 ms while the loop runs at
    100 Hz, so raw differences alternate between 0 and 2x; the filter
    averages that out.
    """

    def __init__(self, cutoff_hz: float = 12.0) -> None:
        self.cutoff_hz = float(cutoff_hz)
        self._q: np.ndarray | None = None
        self.dq = np.zeros(12)

    def update(self, q: np.ndarray, dt_s: float) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if self._q is None or dt_s <= 0.0 or dt_s > 0.2:
            self._q = q.copy()
            self.dq = np.zeros(12)
            return self.dq
        raw = (q - self._q) / dt_s
        alpha = 1.0 - math.exp(-2.0 * math.pi * self.cutoff_hz * dt_s)
        self.dq = self.dq + alpha * (raw - self.dq)
        self._q = q.copy()
        return self.dq

    def reset(self) -> None:
        self._q = None
        self.dq = np.zeros(12)
