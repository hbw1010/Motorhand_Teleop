"""Bounded, asynchronous JSONL telemetry and offline plotting.

Times ending in ``_ns`` are timestamps, not durations. Latency plots subtract
only timestamps explicitly declared to share a clock in metadata["clock_domains"].
The control loop never waits for disk I/O: a full queue rejects the new sample
with an exception, and the final summary counts rejections. Call close() outside
the loop to flush and surface any writer errors.
"""

from __future__ import annotations

import json
import math
import queue
import threading
from pathlib import Path


FIELDS = (
    "time_ns", "sim_time_s", "input_time_ns", "contact_time_ns", "command_time_ns",
    "control_interval_s", "simulation_step_s",
    "contact_force_n", "contact_penetration_m", "requested_torque_nm", "computed_torque_nm",
    "command_torque_nm", "torque_limited", "slew_limited", "motor_position_rad", "joint_position_rad",
    "sim_motor_position_rad", "sim_joint_position_rad",
    "virtual_servo_mode", "virtual_servo_limit_nm", "input_alignment_error_rad",
    "motor_current_a", "object_position_m", "object_velocity_m_s", "object_deformation_m",
    "object_orientation_wxyz", "object_dimensions_m",
    "grasp_state", "slip_speed_m_s", "displacement_m", "object_reset", "object_edited", "object_edit_blocked",
    "feedback_pause_reason", "feedback_transition",
)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
UNITS = {
    "time_ns": "ns", "sim_time_s": "s", "input_time_ns": "ns",
    "control_interval_s": "s", "simulation_step_s": "s",
    "contact_time_ns": "ns", "command_time_ns": "ns", "contact_force_n": "N",
    "command_torque_nm": "N m", "motor_position_rad": "rad", "joint_position_rad": "rad",
    "sim_motor_position_rad": "rad", "sim_joint_position_rad": "rad",
    "virtual_servo_mode": "state", "virtual_servo_limit_nm": "N m",
    "input_alignment_error_rad": "rad",
    "motor_current_a": "A", "object_position_m": "m", "object_velocity_m_s": "m/s",
    "object_deformation_m": "m",
    "object_orientation_wxyz": "unit quaternion wxyz",
    "object_dimensions_m": "m", "object_edited": "bool", "object_edit_blocked": "bool",
    "feedback_pause_reason": "state", "feedback_transition": "state",
    "grasp_state": "state", "slip_speed_m_s": "m/s", "displacement_m": "m",
    "computed_torque_nm": "N m", "contact_penetration_m": "m",
    "object_reset": "bool",
    "requested_torque_nm": "N m", "torque_limited": "bool[12]", "slew_limited": "bool[12]",
}
# The caller must override these when its input or estimator has another source.
SOURCES = {
    "time_ns": "measured", "sim_time_s": "simulated", "input_time_ns": "measured",
    "control_interval_s": "measured", "simulation_step_s": "simulation_schedule",
    "contact_time_ns": "measured", "command_time_ns": "measured",
    "contact_force_n": "simulated", "command_torque_nm": "command",
    "motor_position_rad": "unknown", "joint_position_rad": "estimated",
    "sim_motor_position_rad": "simulated", "sim_joint_position_rad": "simulated",
    "virtual_servo_mode": "simulation_control_state", "virtual_servo_limit_nm": "simulation_control_constraint",
    "input_alignment_error_rad": "derived_from_input_and_simulation",
    "motor_current_a": "measured", "object_position_m": "simulated",
    "object_velocity_m_s": "simulated", "object_deformation_m": "simulated",
    "object_orientation_wxyz": "simulated_or_operator_set",
    "object_dimensions_m": "configured_or_operator_set",
    "object_edited": "operator_action",
    "object_edit_blocked": "operator_action_rejected",
    "feedback_pause_reason": "control_state", "feedback_transition": "control_state",
    "grasp_state": "simulation_heuristic",
    "slip_speed_m_s": "simulated", "displacement_m": "simulated",
    "computed_torque_nm": "derived_from_simulated_contact", "contact_penetration_m": "simulated",
    "object_reset": "operator_action",
    "requested_torque_nm": "derived_from_simulated_contact",
    "torque_limited": "control_constraint", "slew_limited": "control_constraint",
}


def _json_value(value):
    """Make an independent JSON-compatible snapshot, including numpy values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if hasattr(value, "item"):
        return _json_value(value.item())
    raise TypeError(f"Unsupported telemetry value: {type(value).__name__}")


class Recorder:
    """Write JSONL on a worker thread with a bounded queue.

    ``metadata`` may provide ``units``, ``sources``, ``clock_domains``,
    ``motor_names`` and ``joint_names``. The former two override field defaults.
    Queue saturation raises immediately; a summary records rejected samples.
    Writer errors are raised by append() and close(), which flushes on shutdown.
    """

    def __init__(self, path, metadata):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        supplied = _json_value(dict(metadata))
        supplied["units"] = {**UNITS, **(supplied.get("units") or {})}
        supplied["sources"] = {**SOURCES, **(supplied.get("sources") or {})}
        self._file = self.path.open("x", encoding="utf-8")
        self._queue = queue.Queue(maxsize=1024)
        self._lock = threading.Lock()
        self._dropped = 0
        self._closed = False
        self._error = None
        self._metadata = supplied
        self._thread = threading.Thread(target=self._run, name="telemetry-writer", daemon=True)
        self._thread.start()

    def _run(self):
        try:
            self._write({"type": "metadata", "metadata": self._metadata})
            while True:
                sample = self._queue.get()
                if sample is None:
                    break
                self._write({"type": "sample", **sample})
            self._write({"type": "summary", "dropped_samples": self._dropped})
            self._file.flush()
        except Exception as exc:
            self._error = exc
        finally:
            try:
                self._file.close()
            except Exception as exc:
                if self._error is None:
                    self._error = exc

    def _write(self, record):
        self._file.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")

    def _raise_error(self):
        if self._error is not None:
            raise RuntimeError("Telemetry writer failed") from self._error

    def append(self, sample):
        self._raise_error()
        snapshot = _json_value(dict(sample))
        snapshot.pop("type", None)
        for field in FIELDS:
            snapshot.setdefault(field, None)
        with self._lock:
            if self._closed:
                raise RuntimeError("Telemetry recorder is closed")
            try:
                self._queue.put_nowait(snapshot)
            except queue.Full as exc:
                self._dropped += 1
                raise RuntimeError("Telemetry queue is full; sample was not recorded") from exc

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                # Only shutdown may wait for the writer; never block a producer.
                while self._thread.is_alive():
                    try:
                        self._queue.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        pass
        self._thread.join()
        self._raise_error()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _series(samples, field, names=None):
    """Return named columns; absent values remain NaN, never zero."""
    values = [sample.get(field) for sample in samples]
    keys = []
    for value in values:
        if isinstance(value, dict):
            for key in value:
                if key not in keys:
                    keys.append(key)
        elif isinstance(value, (list, tuple)):
            for i in range(len(value)):
                key = names[i] if names and i < len(names) else str(i + 1)
                if key not in keys:
                    keys.append(key)
        elif value is not None and "value" not in keys:
            keys.append("value")
    result = {}
    for key in keys:
        column = []
        for value in values:
            if isinstance(value, dict):
                v = value.get(key)
            elif isinstance(value, (list, tuple)):
                try:
                    index = names.index(key) if names and key in names else int(key) - 1
                    v = value[index] if 0 <= index < len(value) else None
                except (ValueError, IndexError):
                    v = None
            else:
                v = value if key == "value" else None
            column.append(_number(v))
        result[str(key)] = column
    return result


def plot_log(path, output_dir):
    """Plot available telemetry to PNG files and return their paths.

    Matplotlib is imported only here, so recording has no plotting dependency.
    Metadata clock_domains must explicitly identify the same clock for both
    ends of a latency; unknown, mismatched or negative intervals are omitted.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib; install it in the active environment") from exc

    metadata, samples = {}, []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "metadata":
                metadata = record.get("metadata", {})
            elif record.get("type") == "sample":
                samples.append(record)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sources = {**SOURCES, **metadata.get("sources", {})}
    domains = metadata.get("clock_domains", {})
    sim_times = [_number(s.get("sim_time_s")) for s in samples]
    if any(math.isfinite(t) for t in sim_times):
        times, time_label = sim_times, "Simulation time [s] (simulated)"
    else:
        raw_times = [_number(s.get("time_ns")) for s in samples]
        origin = next((t for t in raw_times if math.isfinite(t)), 0)
        times = [(t - origin) * 1e-9 for t in raw_times]
        time_label = f"Elapsed time [s] ({sources.get('time_ns', 'unknown')})"
    paths = []

    def label(field):
        return f"{field} [{UNITS[field]}] ({sources.get(field, 'unknown')})"

    def draw(ax, field, names=None, x=None, prefix=""):
        columns = _series(samples, field, names)
        count = 0
        for name, values in columns.items():
            if any(math.isfinite(v) for v in values):
                ax.plot(times if x is None else x, values, label=f"{prefix}{name}")
                count += 1
        if not count:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.legend(fontsize="x-small", ncol=2)
        ax.set_ylabel(label(field))
        ax.grid(True, alpha=0.3)

    def save(fig, filename):
        fig.tight_layout()
        target = output / filename
        fig.savefig(target, dpi=140)
        plt.close(fig)
        paths.append(target)

    fig, ax = plt.subplots()
    draw(ax, "contact_force_n", list(FINGERS))
    ax.set_xlabel(time_label)
    save(fig, "contact_force.png")

    fig, axes = plt.subplots(3, 1, sharex=True)
    draw(axes[0], "requested_torque_nm", metadata.get("motor_names"))
    draw(axes[1], "computed_torque_nm", metadata.get("motor_names"))
    draw(axes[2], "command_torque_nm", metadata.get("motor_names"))
    axes[2].set_xlabel(time_label)
    save(fig, "command_torque.png")

    fig, ax = plt.subplots()
    for field, name in (("torque_limited", "torque cap"), ("slew_limited", "slew cap")):
        counts = [sum(v) if isinstance(v, list) and all(isinstance(x, bool) for x in v)
                  else float("nan") for v in (s.get(field) for s in samples)]
        ax.plot(times, counts, label=name)
    ax.set(xlabel=time_label, ylabel="Limited motor axes [count] (control)")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    save(fig, "torque_limiting.png")

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 12))
    draw(axes[0], "motor_position_rad", metadata.get("motor_names"))
    draw(axes[1], "sim_motor_position_rad", metadata.get("motor_names"))
    draw(axes[2], "joint_position_rad", metadata.get("joint_names"))
    draw(axes[3], "sim_joint_position_rad", metadata.get("joint_names"))
    axes[3].set_xlabel(time_label)
    save(fig, "positions.png")

    fig, ax = plt.subplots()
    draw(ax, "motor_current_a", metadata.get("motor_names"))
    ax.set_xlabel(time_label)
    save(fig, "motor_current.png")

    fig, ax = plt.subplots()
    depths = _series(samples, "contact_penetration_m", list(FINGERS))
    force = _series(samples, "contact_force_n", list(FINGERS))
    count = 0
    for finger in FINGERS:
        if finger in depths and finger in force:
            ax.plot(depths[finger], force[finger], label=finger)
            count += 1
    if count:
        ax.legend(fontsize="x-small")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(label("contact_penetration_m"))
    ax.set_ylabel(label("contact_force_n"))
    ax.grid(True, alpha=0.3)
    save(fig, "force_displacement.png")

    fig, axes = plt.subplots(2, 1)
    draw(axes[0], "object_deformation_m")
    axes[0].set_xlabel(time_label)
    deformation = [_number(s.get("object_deformation_m")) for s in samples]
    drawn = False
    if any(math.isfinite(value) for value in deformation):
        for finger, values in force.items():
            if any(math.isfinite(value) for value in values):
                axes[1].plot(deformation, values, label=finger)
                drawn = True
    if drawn:
        axes[1].legend(fontsize="x-small")
    else:
        axes[1].text(0.5, 0.5, "No deformable-object data",
                     ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_xlabel(label("object_deformation_m"))
    axes[1].set_ylabel(label("contact_force_n"))
    axes[1].grid(True, alpha=0.3)
    save(fig, "object_deformation.png")

    fig, ax = plt.subplots()
    count = 0
    for start, end in (("input_time_ns", "contact_time_ns"),
                       ("contact_time_ns", "command_time_ns"),
                       ("input_time_ns", "command_time_ns")):
        if not domains.get(start) or domains.get(start) != domains.get(end):
            continue
        values = []
        for sample in samples:
            a, b = _number(sample.get(start)), _number(sample.get(end))
            values.append((b - a) * 1e-6 if math.isfinite(a) and math.isfinite(b) and b >= a else float("nan"))
        if any(math.isfinite(v) for v in values):
            ax.plot(times, values, label=f"{start} → {end} ({sources.get(start, 'unknown')} → {sources.get(end, 'unknown')})")
            count += 1
    if count:
        ax.legend(fontsize="x-small")
    else:
        ax.text(0.5, 0.5, "No comparable timestamps", ha="center", va="center", transform=ax.transAxes)
    ax.set(xlabel=time_label, ylabel="Stage latency [ms]")
    ax.grid(True, alpha=0.3)
    save(fig, "latency.png")

    fig, ax = plt.subplots()
    states = [s.get("grasp_state") for s in samples]
    categories = []
    for state in states:
        if state is not None and isinstance(state, (str, bool, int, float)) and state not in categories:
            categories.append(state)
    if categories:
        ax.step(times, [categories.index(s) if s in categories else float("nan") for s in states], where="post")
        ax.set_yticks(range(len(categories)), [str(s) for s in categories])
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set(xlabel=time_label, ylabel=label("grasp_state"))
    ax.grid(True, alpha=0.3)
    save(fig, "grasp_state.png")

    fig, axes = plt.subplots(3, 1, sharex=True)
    draw(axes[0], "slip_speed_m_s")
    draw(axes[1], "object_position_m", ["x", "y", "z"])
    draw(axes[2], "object_velocity_m_s", ["x", "y", "z"])
    axes[2].set_xlabel(time_label)
    save(fig, "object_motion.png")
    return paths
