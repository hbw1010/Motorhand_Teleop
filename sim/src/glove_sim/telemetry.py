from __future__ import annotations

import csv
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .mapping import FINGERS
from .physics import CONTACT_GROUPS, StepResult


def _columns() -> list[str]:
    cols = [
        "timestamp_mono_ns", "timestamp_iso", "source", "data_validity",
        "sensor_timestamp_ns", "sensor_age_ms", "loop_hz",
        "contact_count", "kinematics_ms", "kinematics_gap_m", "physics_step_ms",
        "max_object_disp_m",
    ]
    for finger in FINGERS:
        cols += [f"contact_force_{finger}_n", f"penetration_{finger}_m"]
    cols += ["contact_force_palm_n", "penetration_palm_m"]
    for i in range(1, 13):
        cols += [
            f"real_joint_{i}_rad",
            f"sim_joint_{i}_rad",
            f"sim_vel_{i}_rad_s",
            f"torque_raw_{i}_nm",
            f"torque_cmd_{i}_nm",
            f"measured_current_{i}_a",
            f"estimated_force_{i}_n",
        ]
    cols += ["grasp_hold", "slip_validity", "slip_m", "force_estimate_validity", "notes"]
    return cols


COLUMNS = _columns()


class RingBuffer:
    def __init__(self, maxlen: int = 6000) -> None:
        self.t = deque(maxlen=maxlen)
        self.force = deque(maxlen=maxlen)
        self.torque = deque(maxlen=maxlen)
        self.q = deque(maxlen=maxlen)
        self.latency = deque(maxlen=maxlen)

    def push(self, row: dict[str, Any]) -> None:
        self.t.append(row["timestamp_mono_ns"] / 1e9)
        self.force.append([row[f"contact_force_{f}_n"] for f in FINGERS])
        self.torque.append([row[f"torque_cmd_{i}_nm"] for i in range(1, 13)])
        self.q.append([row[f"sim_joint_{i}_rad"] for i in range(1, 13)])
        self.latency.append(row.get("sensor_age_ms", float("nan")))


class Telemetry:
    def __init__(self, config: dict[str, Any], *, source: str = "fake",
                 force_estimate_available: bool = False) -> None:
        self.config = config
        self.source = source
        self.force_estimate_available = force_estimate_available
        self.ring = RingBuffer(int(float(config.get("telemetry", {}).get("ring_seconds", 30.0)) * 200))
        self._csv = None
        self._jsonl = None
        self._writer = None
        self._path_prefix: Path | None = None
        self.rows_written = 0

    @property
    def recording(self) -> bool:
        return self._writer is not None

    def start(self, directory: str | Path, *, prefix: str = "glove_session") -> Path:
        if self.recording:
            return self._path_prefix  # type: ignore[return-value]
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._path_prefix = directory / f"{prefix}-{stamp}"
        csv_path = self._path_prefix.with_suffix(".csv")
        jsonl_path = self._path_prefix.with_suffix(".jsonl")
        self._csv = csv_path.open("w", newline="", encoding="utf-8")
        self._csv.write(
            "# Motorhand glove simulation telemetry. source=%s validity=%s\n"
            "# forces N (MuJoCo contact force); torques Nm (nominal command); currents A (SDK measured on real, fake proxy otherwise); "
            "positions rad; timestamps monotonic ns; object disp m (body-local flex strain); "
            "slip unavailable because no object pose sensor exists; estimated force is uncalibrated unless stated.\n"
            % (self.source, "simulated" if self.source == "fake" else "measured+simulated")
        )
        self._writer = csv.DictWriter(self._csv, fieldnames=COLUMNS)
        self._writer.writeheader()
        self._jsonl = jsonl_path.open("w", encoding="utf-8")
        self.rows_written = 0
        return self._path_prefix

    def write(self, row: dict[str, Any]) -> None:
        if self._writer is not None:
            self._writer.writerow({key: row.get(key, "") for key in COLUMNS})
            self._csv.flush()
        if self._jsonl is not None:
            self._jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._jsonl.flush()
        self.rows_written += 1

    def stop(self) -> None:
        if self._csv is not None:
            self._csv.close()
        if self._jsonl is not None:
            self._jsonl.close()
        self._csv = self._jsonl = self._writer = None

    def make_row(self, result: StepResult, state: Any, *, loop_hz: float,
                 notes: str = "") -> dict[str, Any]:
        now = result.timestamp_ns
        row: dict[str, Any] = {
            "timestamp_mono_ns": now,
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "source": state.source,
            "data_validity": "simulated" if state.source == "fake" else "mixed",
            "sensor_timestamp_ns": state.timestamp_ns,
            "sensor_age_ms": max(0.0, (now - state.timestamp_ns) / 1e6),
            "loop_hz": round(float(loop_hz), 3),
            "contact_count": result.contact_count,
            "kinematics_ms": round(float(result.kinematics_ms), 4),
            "kinematics_gap_m": float(result.kinematics_gap_m),
            "physics_step_ms": round(float(result.step_ms), 4),
            "max_object_disp_m": float(result.max_object_disp_m),
            "grasp_hold": bool(any(v > 0.2 for v in result.finger_force_n.values())),
            # The objects are pose-locked by default, so object slip cannot be
            # measured.  None of the motors has a force sensor or calibration.
            "slip_validity": "unavailable_no_pose_sensor",
            "slip_m": "",
            "force_estimate_validity": "uncalibrated" if not self.force_estimate_available else "calibrated",
            "notes": notes,
        }
        for finger in FINGERS:
            row[f"contact_force_{finger}_n"] = float(result.finger_force_n.get(finger, 0.0))
            row[f"penetration_{finger}_m"] = float(result.finger_penetration_m.get(finger, 0.0))
        row["contact_force_palm_n"] = float(result.finger_force_n.get("palm", 0.0))
        row["penetration_palm_m"] = float(result.finger_penetration_m.get("palm", 0.0))
        for i in range(12):
            mid = i + 1
            row[f"real_joint_{mid}_rad"] = float(state.angles_rad[i])
            row[f"sim_joint_{mid}_rad"] = float(result.sim_q[i])
            row[f"sim_vel_{mid}_rad_s"] = float(result.sim_dq[i])
            row[f"torque_raw_{mid}_nm"] = float(result.torque_raw_nm[i])
            row[f"torque_cmd_{mid}_nm"] = float(result.torque_cmd_nm[i])
            row[f"measured_current_{mid}_a"] = float(state.current_a.get(mid, float("nan")))
            # There is no force sensor and no per-motor force calibration.
            row[f"estimated_force_{mid}_n"] = "" if not self.force_estimate_available else float("nan")
        return row

    def latency_stats(self) -> dict[str, float]:
        values = [v for v in self.ring.latency if np.isfinite(v)]
        if not values:
            return {"p50_ms": float("nan"), "p99_ms": float("nan"), "max_ms": float("nan")}
        arr = np.asarray(values)
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(arr.max()),
        }
