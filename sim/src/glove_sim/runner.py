from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .mapping import GloveMapping
from .objects import ObjectSpec
from .physics import Simulation, StepResult
from .sdk.base import GloveBackend, GloveState
from .telemetry import Telemetry


@dataclass
class RunnerStats:
    loop_hz: float = 0.0
    physics_steps: int = 0
    last_result: StepResult | None = None
    last_state: GloveState | None = None
    error: str = ""


class Runner:
    """Non-blocking control loop shared by the UI and headless smoke tests."""

    def __init__(self, config: dict[str, Any], mapping: GloveMapping, glove: GloveBackend,
                 specs: list[ObjectSpec], *, telemetry: Telemetry | None = None,
                 feedback: bool = False) -> None:
        self.config = config
        self.mapping = mapping
        self.glove = glove
        self.specs = specs
        self.sim = Simulation(config, mapping, specs)
        self.telemetry = telemetry or Telemetry(config, source="fake")
        self.feedback = bool(feedback)
        self.state: GloveState | None = None
        self.result: StepResult | None = None
        self.stats = RunnerStats()
        self._last_control = 0.0
        self._last_physics = time.monotonic()
        self._control_period = 1.0 / float(config["simulation"].get("control_hz", 200.0))
        self._physics_dt = float(config["simulation"].get("timestep_s", 0.001))
        self._enabled = False
        self._started = False
        self._last_loop = time.monotonic()
        self._loop_count = 0
        self.state_source = "fake"
        self.notes = ""
        self.kinematics_error = ""

    # ------------------------------------------------------------------
    def start(self) -> None:
        self.state = self.glove.open()
        if self.state.source == "real" and not self.feedback:
            self.glove.enable(False)
        self.sim.update_glove(self.state.angles_rad)
        if self.feedback:
            self.glove.enable(True)
            self._enabled = True
        self._started = True
        self._last_loop = time.monotonic()
        self._loop_count = 0

    def set_feedback(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self.feedback:
            return
        if self.state is not None and self.state.source == "real":
            if enabled:
                if self.state.errors:
                    raise RuntimeError(f"电机报警，拒绝使能: {self.state.errors}")
                self.glove.enable(True)
                self._enabled = True
            else:
                for mid in range(1, 13):
                    try:
                        self.glove.command_torque(mid, 0.0, deadline_ns=time.monotonic_ns())
                    except Exception:
                        pass
                self.glove.enable(False)
                self._enabled = False
        else:
            # The fake backend also records the nominal torque/current proxy.
            try:
                self.glove.enable(enabled)
            except Exception:
                pass
        self.feedback = enabled
        self.sim._enabled_since_ns = None

    def emergency_stop(self) -> None:
        """Zero every motor and disable feedback immediately (safe stop)."""
        self.feedback = False
        self._enabled = False
        self.sim._enabled_since_ns = None
        self.stats.error = "EMERGENCY STOP: 反馈已关闭，力矩已清零"
        if self.state is not None and self.state.source == "real":
            try:
                for mid in range(1, 13):
                    try:
                        self.glove.command_torque(mid, 0.0, deadline_ns=time.monotonic_ns())
                    except Exception:
                        pass
                self.glove.enable(False)
            except Exception as error:
                self.stats.error = f"EMERGENCY STOP 时关闭电机失败: {error}"
        try:
            self.glove.enable(False)
        except Exception as error:
            if not self.stats.error:
                self.stats.error = f"EMERGENCY STOP 时失能失败: {error}"

    def set_demo(self, enabled: bool) -> None:
        """Pause/resume the fake glove demo (no effect on a real glove)."""
        setter = getattr(self.glove, "set_demo", None)
        if setter is None:
            raise RuntimeError("当前后端不是假手套，无法暂停演示")
        setter(bool(enabled))

    def tick(self, *, max_steps: int = 8, wall_clock: bool = True) -> StepResult | None:
        """Run one control update and the physics steps due since the last call."""
        if not self._started:
            self.start()
        if self.state is None:
            return None
        now = time.monotonic()
        if wall_clock and now - self._last_control < self._control_period * 0.8:
            return self.result
        # Read the real/fake glove at the control rate.
        try:
            self.state = self.glove.read()
            self.state_source = self.state.source
        except Exception as error:
            self.kinematics_error = str(error)
            self.stats.error = f"读取失败: {error}"
            if self.feedback:
                self.emergency_stop()
            return self.result
        if self.feedback and self.state.source == "real" and self.state.errors:
            alarm = dict(self.state.errors)
            self.emergency_stop()
            self.stats.error = f"电机报警，已急停: {alarm}"
            return self.result
        self.sim.update_glove(self.state.angles_rad)
        # Physics steps: keep up with wall time, capped so a UI hiccup does not
        # stall the whole session.
        if wall_clock:
            elapsed = max(0.0, min(0.05, now - self._last_physics))
            due = int(elapsed / self._physics_dt)
        else:
            due = max_steps
        due = max(1, min(int(max_steps), due))
        result = None
        for _ in range(due):
            result = self.sim.step(feedback_enabled=self.feedback, dt_s=self._physics_dt)
        self._last_physics = now
        self._last_control = now
        self.result = result
        self.stats.last_result = result
        self.stats.last_state = self.state
        self.stats.physics_steps += due
        self._loop_count += 1
        dt = max(1e-6, now - self._last_loop)
        if dt >= 0.5:
            self.stats.loop_hz = self._loop_count / dt
            self._loop_count = 0
            self._last_loop = now
        if result is not None and self.feedback:
            deadline = time.monotonic_ns() + int(float(self.config["simulation"].get("command_ttl_s", 0.1)) * 1e9)
            for i, torque in enumerate(result.torque_cmd_nm):
                try:
                    self.glove.command_torque(i + 1, float(torque), deadline_ns=deadline)
                except Exception as error:
                    self.stats.error = f"发送电机 {i + 1} 失败，已急停: {error}"
                    self.emergency_stop()
                    break
        if result is not None:
            hz = self.stats.loop_hz or float(self.config["simulation"].get("control_hz", 200.0))
            row = self.telemetry.make_row(result, self.state, loop_hz=hz, notes=self.notes)
            self.telemetry.ring.push(row)
            if self.telemetry.recording:
                self.telemetry.write(row)
        return result

    def close(self) -> None:
        try:
            if self._enabled:
                self.glove.enable(False)
        finally:
            try:
                self.glove.close()
            except Exception:
                pass
            self.telemetry.stop()

    def start_recording(self, directory: str) -> str:
        return str(self.telemetry.start(directory))

    def stop_recording(self) -> None:
        self.telemetry.stop()
