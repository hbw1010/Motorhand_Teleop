from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .impedance import DriverImpedancePlanner, FrictionCompensator, VelocityEstimator
from .mapping import FINGERS, GloveMapping
from .objects import ObjectSpec
from .physics import Simulation, StepResult
from .sdk.base import GloveBackend, GloveState
from .recording import SessionRecorder
from .telemetry import Telemetry


@dataclass
class PlotSample:
    """Everything the UI curves show, captured once per control tick.

    Torques are all in the model-joint frame (closing = +) so they overlay.
    """

    t_ns: int
    finger_force_n: list[float]       # thumb..pinky, palm (simulated contact)
    torque_driver_nm: np.ndarray      # torque_cmd_nm: what the drivers produce now (predicted)
    sim_q: np.ndarray                 # visible simulated hand
    penetration_m: list[float]
    object_disp_m: float
    contact_count: int
    sensor_age_ms: float
    current_a: np.ndarray             # measured (real) / simulated (fake)
    sim_dq: np.ndarray                # visible simulated hand velocity (differenced)
    torque_spring_nm: np.ndarray      # torque_raw_nm: virtual spring, uncapped
    torque_sent_nm: np.ndarray        # computed at send time, model frame; 0 = nothing sent
    real_q: np.ndarray                # encoder, NOT clamped to the calibrated range
    real_dq: np.ndarray               # encoder velocity estimate (used by damping/friction)
    torque_measured_nm: np.ndarray    # current x Kt, model frame
    encoder_clamped: np.ndarray       # bool: encoder outside the calibrated range
    source: str


@dataclass
class RunnerStats:
    loop_hz: float = 0.0
    physics_steps: int = 0
    last_result: StepResult | None = None
    last_state: GloveState | None = None
    error: str = ""
    tick_ms: float = 0.0          # wall time of the last control tick
    steps_last_tick: int = 0      # physics steps executed in the last tick


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
        # Torque actually queued to each motor on the last tick (see _tick_locked).
        self.sent_torque_nm = np.zeros(12)
        # Real-finger joint velocity (model frame) for damping/friction.
        self.velocity = VelocityEstimator(float(config["glove"].get("velocity_filter_hz", 12.0)))
        # The glove is kinematic (MuJoCo qvel is zeroed every step), so the
        # visible sim-hand velocity is differenced from its pose per tick.
        self.sim_velocity = VelocityEstimator(float(config["glove"].get("velocity_filter_hz", 12.0)))
        self._last_sim_t_ns: int | None = None
        # Per-tick samples for the UI curves (drained by the UI thread).
        self.plot_samples: deque[PlotSample] = deque(maxlen=4000)
        # Full per-tick session recording (curves + scene state + real glove),
        # for offline review: python -m glove_sim.cli review.
        self.recorder: SessionRecorder | None = None
        self.torque_constant = float(config["glove"].get("torque_constant_nm_per_a", 5.1))
        self._torque_axis = np.asarray(
            [m.sign * m.scale for m in mapping.motors], dtype=float) * np.asarray(
            config["glove"].get("torque_sign", [1.0] * 12), dtype=float)
        self._last_read_t: float | None = None
        self.last_impedance: list | None = None
        self.impedance_planner = DriverImpedancePlanner(config["glove"].get("driver_impedance", {}),
                                                        config["glove"].get("torque_limit_nm", [4.0] * 12))
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
        # One lock guards glove/sim/telemetry state.  The control thread holds
        # it for one tick; the UI holds it only around sim access.
        self.lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        budget = config["simulation"].get("physics_budget_ms")
        self._physics_budget_s = None if budget is None else float(budget) / 1e3
        self._step_cost_s = self._physics_dt

    # ------------------------------------------------------------------
    def start(self) -> None:
        with self.lock:
            self._start_locked()

    def _start_locked(self) -> None:
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
        with self.lock:
            self._set_feedback_locked(enabled)

    def _set_feedback_locked(self, enabled: bool) -> None:
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
        if not enabled:
            self.sent_torque_nm = np.zeros(12)

    def emergency_stop(self) -> None:
        """Zero every motor and disable feedback immediately (safe stop)."""
        # Drop the feedback flag before waiting for the lock so an in-flight
        # tick of the control thread stops commanding torque as early as possible.
        self.feedback = False
        with self.lock:
            self._emergency_stop_locked()

    def _emergency_stop_locked(self) -> None:
        self.feedback = False
        self._enabled = False
        self.sent_torque_nm = np.zeros(12)
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

    def tick(self, *, max_steps: int = 8, wall_clock: bool = True,
             rate_gate: bool = True) -> StepResult | None:
        """Run one control update and the physics steps due since the last call.

        ``rate_gate`` skips calls that arrive faster than control_hz (for
        callers that poll).  The fixed-rate control thread schedules ticks
        itself and disables it, so a late tick never causes a skipped one.
        """
        with self.lock:
            return self._tick_locked(max_steps=max_steps, wall_clock=wall_clock,
                                     rate_gate=rate_gate)

    def _tick_locked(self, *, max_steps: int, wall_clock: bool,
                     rate_gate: bool = True) -> StepResult | None:
        if not self._started:
            self._start_locked()
        if self.state is None:
            return None
        now = time.monotonic()
        if rate_gate and wall_clock and now - self._last_control < self._control_period * 0.8:
            return self.result
        t_tick = time.perf_counter()
        # Read the real/fake glove at the control rate.
        try:
            self.state = self.glove.read()
            self.state_source = self.state.source
        except Exception as error:
            self.kinematics_error = str(error)
            self.stats.error = f"读取失败: {error}"
            if self.feedback:
                self._emergency_stop_locked()
            return self.result
        if self.feedback and self.state.source == "real" and self.state.errors:
            alarm = dict(self.state.errors)
            self._emergency_stop_locked()
            self.stats.error = f"电机报警，已急停: {alarm}"
            return self.result
        t_read = time.monotonic()
        dq = self.velocity.update(self.state.angles_rad,
                                  0.0 if self._last_read_t is None else t_read - self._last_read_t)
        self._last_read_t = t_read
        self.sim.update_glove(self.state.angles_rad)
        # Physics steps: keep up with wall time, capped so a UI hiccup does not
        # stall the whole session.
        if wall_clock:
            elapsed = max(0.0, min(0.05, now - self._last_physics))
            due = int(elapsed / self._physics_dt)
        else:
            due = max_steps
        due = max(1, min(int(max_steps), due))
        # Optional per-tick physics budget keeps the control rate fixed under
        # heavy contact: physics then runs slower than wall time (visual only;
        # in the default locked mode feedback comes from the ghost forward pass
        # evaluated in update_glove).  The feedback filter/slew still integrate
        # the full wall-clock interval, so their Hz and Nm/s keep their meaning.
        steps = due
        if wall_clock and self._physics_budget_s is not None:
            steps = max(1, min(due, int(self._physics_budget_s / max(self._step_cost_s, 1e-5))))
        feedback_dt = (elapsed / steps) if (wall_clock and elapsed > 0) else self._physics_dt
        result = None
        t_phys = time.perf_counter()
        for _ in range(steps):
            result = self.sim.step(feedback_enabled=self.feedback, dt_s=feedback_dt)
        cost = (time.perf_counter() - t_phys) / steps
        self._step_cost_s += 0.3 * (cost - self._step_cost_s)
        self.stats.steps_last_tick = steps
        due = steps
        self._last_physics = now
        self._last_control = now
        self.stats.tick_ms = (time.perf_counter() - t_tick) * 1e3
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
        sent = np.zeros(12)
        self.last_impedance = None
        driver_mode = (str(self.config["glove"].get("feedback_mode", "proxy_spring")) == "driver_impedance"
                       and hasattr(self.glove, "command_impedance"))
        if result is not None and self.feedback and driver_mode:
            deadline = time.monotonic_ns() + int(float(self.config["simulation"].get("command_ttl_s", 0.1)) * 1e9)
            cmds = self._plan_impedance(dq)
            q = self.state.angles_rad
            self.last_impedance = cmds
            # What the drivers produce now, in model frame (plots/telemetry).
            result.torque_cmd_nm = np.asarray([c.torque(q[i], dq[i]) for i, c in enumerate(cmds)])
            for i, cmd in enumerate(cmds):
                try:
                    value = self.glove.command_impedance(i + 1, cmd, float(q[i]), deadline_ns=deadline)
                    if value is not None:
                        sent[i] = float(value)
                except Exception as error:
                    self.stats.error = f"发送电机 {i + 1} 失败，已急停: {error}"
                    self._emergency_stop_locked()
                    sent[:] = 0.0
                    break
        elif result is not None and self.feedback:
            deadline = time.monotonic_ns() + int(float(self.config["simulation"].get("command_ttl_s", 0.1)) * 1e9)
            for i, torque in enumerate(result.torque_cmd_nm):
                try:
                    value = self.glove.command_torque(i + 1, float(torque), deadline_ns=deadline)
                    if value is not None:
                        sent[i] = float(value)
                except Exception as error:
                    self.stats.error = f"发送电机 {i + 1} 失败，已急停: {error}"
                    self._emergency_stop_locked()
                    sent[:] = 0.0
                    break
        # Exactly what was queued to the glove this tick (motor-axis Nm;
        # zero when feedback is off or a motor is skipped).
        self.sent_torque_nm = sent
        if result is not None:
            self._update_sim_velocity(result)
            sample = self._plot_sample(result, sent)
            self.plot_samples.append(sample)
            if self.recorder is not None:
                self.recorder.record(self, result, sample)
            hz = self.stats.loop_hz or float(self.config["simulation"].get("control_hz", 200.0))
            row = self.telemetry.make_row(result, self.state, loop_hz=hz, notes=self.notes)
            self.telemetry.ring.push(row)
            if self.telemetry.recording:
                self.telemetry.write(row)
        return result

    def _update_sim_velocity(self, result: StepResult) -> None:
        t = int(result.timestamp_ns)
        dt = 0.0 if self._last_sim_t_ns is None else (t - self._last_sim_t_ns) / 1e9
        self._last_sim_t_ns = t
        result.sim_dq = self.sim_velocity.update(result.sim_q, dt).copy()

    def to_model_torque(self, motor_axis_nm: np.ndarray) -> np.ndarray:
        """Motor-axis torque -> model-joint torque (inverse of RealGlove.motor_axis_torque)."""
        axis = np.where(self._torque_axis == 0.0, 1.0, self._torque_axis)
        return np.asarray(motor_axis_nm, dtype=float) / axis

    def _plot_sample(self, result: StepResult, sent_motor_nm: np.ndarray) -> PlotSample:
        state = self.state
        nan12 = np.full(12, np.nan)
        if state is not None:
            current = np.asarray([float(state.current_a.get(i + 1, np.nan)) for i in range(12)])
            real_q = np.asarray(state.angles_rad, dtype=float).copy()
            clamped = np.zeros(12, dtype=bool)
            for i, motor in enumerate(self.mapping.motors):
                raw = (state.raw_deg or {}).get(motor.motor_id)
                if raw is None or not np.isfinite(raw):
                    continue
                q = motor.raw_to_model_rad(raw)
                lo, hi = motor.model_rad_limits()
                real_q[i] = q
                clamped[i] = q < lo - 1e-9 or q > hi + 1e-9
            age = max(0.0, (result.timestamp_ns - state.timestamp_ns) / 1e6)
            source = state.source
        else:
            current, real_q, clamped, age, source = nan12, nan12, np.zeros(12, bool), float("nan"), ""
        return PlotSample(
            t_ns=int(result.timestamp_ns),
            finger_force_n=[float(result.finger_force_n.get(f, 0.0)) for f in FINGERS + ("palm",)],
            torque_driver_nm=np.asarray(result.torque_cmd_nm, dtype=float).copy(),
            sim_q=np.asarray(result.sim_q, dtype=float).copy(),
            penetration_m=[float(result.finger_penetration_m.get(f, 0.0)) for f in FINGERS],
            object_disp_m=float(result.max_object_disp_m),
            contact_count=int(result.contact_count),
            sensor_age_ms=age,
            current_a=current,
            sim_dq=np.asarray(result.sim_dq, dtype=float).copy(),
            torque_spring_nm=np.asarray(result.torque_raw_nm, dtype=float).copy(),
            torque_sent_nm=self.to_model_torque(sent_motor_nm),
            real_q=real_q,
            real_dq=self.velocity.dq.copy(),
            torque_measured_nm=self.to_model_torque(current * self.torque_constant),
            encoder_clamped=clamped,
            source=source,
        )

    def _plan_impedance(self, dq: np.ndarray) -> list:
        """MIT kp/kd/target/feed-forward per motor (model frame).

        The planner keeps per-motor state across ticks and re-reads its
        tunables from the live config every tick (UI sliders apply at once).
        """
        glove = self.config["glove"]
        planner = self.impedance_planner
        planner.configure(glove.get("driver_impedance", {}), glove.get("torque_limit_nm", [4.0] * 12))
        friction = FrictionCompensator(glove.get("friction_comp", {}))
        blocks = self.sim.contact_blocks()
        tau_f = friction.torque(dq)
        since = self.sim._enabled_since_ns
        ramp_s = max(1e-6, float(glove.get("feedback_ramp_s", 0.15)))
        ramp = 1.0 if since is None else min(1.0, (time.monotonic_ns() - since) / 1e9 / ramp_s)
        return planner.plan(self.state.angles_rad, dq, blocks, tau_f, ramp=ramp)

    # ------------------------------------------------------------------
    def start_thread(self) -> None:
        """Run tick() at simulation.control_hz on a dedicated thread."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="glove-control", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        period = self._control_period
        next_t = time.monotonic()
        while not self._stop.is_set():
            try:
                self.tick(max_steps=30, wall_clock=True, rate_gate=False)
            except Exception as error:  # keep the loop alive; UI shows the error
                self.stats.error = str(error)
                if self.feedback:
                    try:
                        self.emergency_stop()
                    except Exception:
                        pass
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_t = time.monotonic()  # overrun: do not try to catch up

    def stop_thread(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def close(self) -> None:
        self.stop_thread()
        try:
            if self._enabled:
                self.glove.enable(False)
        finally:
            try:
                self.glove.close()
            except Exception:
                pass
            self.stop_recording()

    def start_recording(self, directory: str) -> str:
        """Start the CSV/JSONL telemetry and the full session recording.

        Returns the session directory (see glove_sim.recording).
        """
        with self.lock:
            self.telemetry.start(directory)
            if self.recorder is None or not self.recorder.active:
                self.recorder = SessionRecorder(directory)
                self.recorder.start(self)
            return str(self.recorder.path)

    def stop_recording(self) -> str | None:
        with self.lock:
            self.telemetry.stop()
            recorder, self.recorder = self.recorder, None
        path = recorder.stop() if recorder is not None else None
        return None if path is None else str(path)

    def record_event(self, kind: str, **data) -> None:
        """Note a user action (object change, slider, ...) in the session recording."""
        recorder = self.recorder
        if recorder is not None:
            recorder.event(kind, **data)
