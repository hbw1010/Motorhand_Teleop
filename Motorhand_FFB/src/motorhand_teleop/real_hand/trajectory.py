"""严格校验的Wuji实测轨迹与异步JSONL录制。"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
from pathlib import Path
import queue
import threading

import numpy as np

from ..domain import WUJI_JOINTS, WujiState


def _identity(hand, serial_number, firmware_version):
    if hand not in ("left", "right"):
        raise ValueError("invalid Wuji hand")
    for value in (serial_number, firmware_version):
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError("invalid Wuji device identity")


def _vector(value, label, *, optional=False):
    if value is None and optional:
        return None
    # JSON numeric values only; reject strings, booleans and nested arrays.
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 20:
        raise ValueError(f"{label} must contain 20 numbers")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
        raise ValueError(f"{label} must contain 20 numbers")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains non-finite numbers")
    return result.tolist()


def _line(obj):
    return (json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


class TrajectoryRecorder:
    """Single-producer bounded recorder. Disk I/O is confined to its writer thread.

    Duplicate source frames are ignored; out-of-order frames and queue/write errors
    invalidate the recording. Existing files are overwritten; concurrent writers
    are rejected. close() commits a footer; abort() never does.
    """
    def __init__(self, path, *, hand, serial_number, firmware_version, queue_size=1024):
        _identity(hand, serial_number, firmware_version)
        if type(queue_size) is not int or queue_size <= 0:
            raise ValueError("queue_size must be positive")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a+b")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._file.seek(0)
            self._file.truncate()
        except BaseException:
            self._file.close()
            raise
        self._queue = queue.Queue(maxsize=queue_size)
        self._error = None
        self._closed = False
        self._failed = False
        self._first_ns = None
        self._last_ns = None
        self._count = 0
        self._header = _line({"type": "header", "version": 1, "hand": hand,
            "serial_number": serial_number, "firmware_version": firmware_version,
            "joint_names": list(WUJI_JOINTS), "time_unit": "ns",
            "position_unit": "rad", "velocity_unit": "rad/s", "effort_unit": "A"})
        self._thread = threading.Thread(target=self._write, name="wuji-trajectory-writer", daemon=True)
        self._thread.start()

    def _write(self):
        try:
            digest = hashlib.sha256()
            self._file.write(self._header)
            digest.update(self._header)
            while True:
                item = self._queue.get()
                if item is None:
                    break
                self._file.write(item)
                digest.update(item)
            if not self._failed and self._count >= 2:
                self._file.write(_line({"type": "end", "frame_count": self._count,
                    "duration_ns": self._last_ns - self._first_ns, "sha256": digest.hexdigest()}))
                self._file.flush()
        except BaseException as exc:
            self._error = exc
        finally:
            try:
                self._file.close()
            except BaseException as exc:
                self._error = self._error or exc

    def record(self, state: WujiState, target_positions_rad):
        if self._closed or self._failed:
            raise RuntimeError("trajectory recorder is closed or failed")
        if self._error is not None:
            self._failed = True
            raise RuntimeError("trajectory writer failed") from self._error
        try:
            stamp = state.received_ns
            if type(stamp) is not int or stamp <= 0:
                raise ValueError("invalid source timestamp")
            if self._last_ns is not None:
                if stamp == self._last_ns:
                    return
                if stamp < self._last_ns:
                    raise ValueError("source timestamp went backwards")
            first = stamp if self._first_ns is None else self._first_ns
            frame = _line({"type": "frame", "source_ns": stamp, "elapsed_ns": stamp - first,
                "positions_rad": _vector(state.positions_rad, "positions_rad"),
                "velocities_rad_s": _vector(state.velocities_rad_s, "velocities_rad_s", optional=True),
                "effort_a": _vector(state.effort_a, "effort_a", optional=True),
                "target_positions_rad": _vector(target_positions_rad, "target_positions_rad")})
            self._queue.put_nowait(frame)
            self._first_ns = first
            self._last_ns = stamp
            self._count += 1
        except BaseException:
            self._failed = True
            raise

    def _stop(self, commit):
        if self._closed:
            return
        self._closed = True
        if not commit or self._count < 2:
            self._failed = True
        # Blocking is allowed at shutdown, never on the 50 Hz recording path.
        while self._thread.is_alive():
            try:
                self._queue.put(None, timeout=0.1)
                break
            except queue.Full:
                pass
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("trajectory writer failed") from self._error
        if commit and self._failed:
            raise RuntimeError("trajectory recording failed or has fewer than two distinct frames")

    def close(self):
        self._stop(True)

    def abort(self):
        self._stop(False)


@dataclass(frozen=True)
class RecordedTrajectory:
    hand: str
    serial_number: str
    firmware_version: str
    duration_ns: int
    _times: np.ndarray
    _positions: np.ndarray

    @property
    def positions_rad(self):
        """Measured frames in source order, as an independent read-only N×20 array."""
        values = self._positions.copy()
        values.setflags(write=False)
        return values

    @classmethod
    def load(cls, path):
        digest = hashlib.sha256()
        times, positions = [], []
        header = None
        footer = None
        first = previous = None
        with Path(path).open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    raise ValueError("truncated trajectory line")
                try:
                    obj = json.loads(line, object_pairs_hook=_pairs,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")))
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise ValueError("invalid trajectory JSON") from exc
                if not isinstance(obj, dict) or footer is not None:
                    raise ValueError("invalid trajectory record order")
                kind = obj.get("type")
                if header is None:
                    if kind != "header" or obj.get("version") != 1 or type(obj.get("version")) is not int:
                        raise ValueError("invalid trajectory header/version")
                    _identity(obj.get("hand"), obj.get("serial_number"), obj.get("firmware_version"))
                    if (obj.get("joint_names") != list(WUJI_JOINTS) or obj.get("time_unit") != "ns"
                            or obj.get("position_unit") != "rad" or obj.get("velocity_unit") != "rad/s"
                            or obj.get("effort_unit") != "A"):
                        raise ValueError("trajectory joint order or units mismatch")
                    header = obj
                elif kind == "frame":
                    stamp, elapsed = obj.get("source_ns"), obj.get("elapsed_ns")
                    if type(stamp) is not int or stamp <= 0 or type(elapsed) is not int:
                        raise ValueError("invalid trajectory timestamp")
                    if first is None:
                        first = stamp
                    if (previous is not None and stamp <= previous) or elapsed != stamp - first:
                        raise ValueError("non-increasing or inconsistent trajectory time")
                    previous = stamp
                    positions.append(_vector(obj.get("positions_rad"), "positions_rad"))
                    for key in ("velocities_rad_s", "effort_a"):
                        if key not in obj:
                            raise ValueError(f"missing {key}")
                        _vector(obj[key], key, optional=True)
                    _vector(obj.get("target_positions_rad"), "target_positions_rad")
                    times.append(elapsed)
                elif kind == "end":
                    footer = obj
                    if (len(times) < 2 or times[-1] <= 0 or type(obj.get("frame_count")) is not int
                            or obj["frame_count"] != len(times) or type(obj.get("duration_ns")) is not int
                            or obj["duration_ns"] != times[-1] or obj.get("sha256") != digest.hexdigest()):
                        raise ValueError("invalid trajectory completion or checksum")
                    continue
                else:
                    raise ValueError("invalid trajectory record")
                digest.update(line)
        if header is None or footer is None:
            raise ValueError("incomplete trajectory")
        return cls(header["hand"], header["serial_number"], header["firmware_version"],
                   times[-1], np.asarray(times, dtype=np.int64), np.asarray(positions, dtype=np.float64))

    def positions_at(self, elapsed_ns):
        if type(elapsed_ns) is not int:
            raise ValueError("elapsed_ns must be an integer")
        if elapsed_ns <= 0:
            return self._positions[0].copy()
        if elapsed_ns >= self.duration_ns:
            return self._positions[-1].copy()
        index = int(np.searchsorted(self._times, elapsed_ns, side="right"))
        start, end = int(self._times[index - 1]), int(self._times[index])
        weight = (elapsed_ns - start) / (end - start)
        return self._positions[index - 1] * (1.0 - weight) + self._positions[index] * weight
