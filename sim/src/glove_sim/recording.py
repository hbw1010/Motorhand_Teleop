"""Session recording: every control tick, enough to rebuild the whole scene.

A session is a directory::

    session-YYYYmmdd-HHMMSS/
        meta.json          config snapshot, mapping, field descriptions,
                           scene segments, events (param changes, toggles)
        scene_000.xml      compiled MuJoCo scene of segment 0 (self-contained)
        chunk_00000.npz    ~1 s of ticks (one segment per chunk)
        ...

Per tick it stores every curve the live UI shows (PlotSample), the real glove
readings (raw encoder, current, temperature, alarms), the MIT impedance
command actually planned for each motor, the contact-proxy state, and the
full MuJoCo ``qpos`` of the visible scene plus the unclamped target ("real
hand") pose.  With the scene XML that reproduces the simulated hand, the
object (incl. the soft ball's flex vertices) and both contact sets exactly:
``Session.mjdata_at(i)`` / ``Session.contacts_at(i)``.

Writing happens on a background thread so the control loop only appends
small arrays.  Each chunk is a complete file, so a crash loses at most ~1 s.

Load for analysis::

    from glove_sim.recording import load_session
    s = load_session("data/session-20261110-153000")
    s.t                    # (N,) seconds since start
    s["real_q"]            # (N, 12) rad, model frame
    s.fields               # names + units + meaning
    i = s.index_at(12.5)   # tick nearest t = 12.5 s
    s.contacts_at(i, pose="target")
"""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .mapping import FINGERS

CONTACT_FINGERS = FINGERS + ("palm",)

# name -> (unit, meaning).  Shapes: scalar per tick, or (12,) per motor
# (motor-ID order J1..J12, model-joint frame, closing = +), or (6,) per
# contact group thumb..pinky, palm.  Saved into meta.json for reference.
FIELDS: dict[str, tuple[str, str]] = {
    "t_ns": ("ns", "monotonic time of the control tick (end of physics)"),
    "segment": ("-", "scene segment index (changes when the object/size changes)"),
    "source": ("-", "0 fake glove, 1 real glove"),
    "feedback_on": ("bool", "force feedback enabled this tick"),
    "loop_hz": ("Hz", "measured control-loop rate"),
    "tick_ms": ("ms", "wall time of this control tick"),
    "physics_steps": ("-", "mj_step calls in this tick"),
    "sensor_age_ms": ("ms", "glove sample age at the end of the tick"),
    # real glove
    "real_q": ("rad", "encoder angle, model frame, NOT clamped to the calibration range"),
    "real_q_used": ("rad", "encoder angle as fed to the simulation (clamped to range)"),
    "real_dq": ("rad/s", "encoder velocity estimate (used by damping / friction comp)"),
    "raw_deg": ("deg", "raw SDK encoder reading per motor"),
    "encoder_clamped": ("bool", "encoder outside the calibrated range"),
    "current_a": ("A", "motor current (measured on the real glove, simulated on the fake one)"),
    "temperature_c": ("degC", "motor temperature (NaN if not reported)"),
    "motor_error": ("bool", "motor reported an alarm"),
    "torque_measured_nm": ("Nm", "current x Kt, model frame (measured estimate)"),
    # simulated hand
    "sim_q": ("rad", "visible simulated motor joint (stops at the object surface)"),
    "sim_dq": ("rad/s", "visible simulated joint velocity (differenced)"),
    "motor_target": ("rad", "motor angle the target/real-hand pose was solved from"),
    "blocked": ("bool", "finger held at the object surface (contact proxy engaged)"),
    "surface_q": ("rad", "surface angle where the proxy engaged (NaN if not blocked)"),
    "proxy_k": ("Nm/rad", "proxy stiffness from the object hardness (NaN if not blocked)"),
    # torques (model frame)
    "torque_spring_nm": ("Nm", "virtual spring torque, computed, uncapped"),
    "torque_driver_nm": ("Nm", "torque the driver PD produces now (predicted from the command)"),
    "torque_sent_nm": ("Nm", "torque computed at send time; 0 = nothing sent"),
    "imp_k": ("Nm/rad", "MIT kp sent (model frame; NaN when no impedance command)"),
    "imp_b": ("Nm s/rad", "MIT kd sent"),
    "imp_target": ("rad", "MIT position target (model frame)"),
    "imp_tau_ff": ("Nm", "MIT feed-forward torque (friction comp etc.)"),
    # contacts / object
    "finger_force_n": ("N", "simulated contact normal force per group thumb..pinky, palm"),
    "penetration_m": ("m", "max penetration per group thumb..pinky, palm"),
    "contact_count": ("-", "contacts used for feedback (target pose)"),
    "object_disp_m": ("m", "max soft-object vertex displacement from rest"),
    # full scene state
    "qpos": ("-", "MuJoCo qpos of the visible scene (per segment; see Session.qpos_at)"),
    "qvel": ("-", "MuJoCo qvel of the visible scene (per segment)"),
    "tree_target": ("rad", "glove tree joints of the unclamped target pose (per segment)"),
}

PER_SEGMENT = ("qpos", "qvel", "tree_target")


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class SessionRecorder:
    """Records ticks of a Runner.  ``record`` runs on the control thread."""

    def __init__(self, directory: str | Path, *, chunk_ticks: int = 100) -> None:
        self.root = Path(directory)
        self.chunk_ticks = int(chunk_ticks)
        self.path: Path | None = None
        self.ticks = 0
        self._rows: list[dict[str, Any]] = []
        self._chunk_index = 0
        self._segment = -1
        self._segment_xml_ref: str | None = None
        self._meta: dict[str, Any] = {}
        self._last_glove_cfg = ""
        self._last_feedback: bool | None = None
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self.error = ""

    @property
    def active(self) -> bool:
        return self.path is not None

    # ------------------------------------------------------------------
    def start(self, runner: Any) -> Path:
        if self.path is not None:
            return self.path
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"session-{_now_stamp()}"
        n = 1
        while path.exists():
            path = self.root / f"session-{_now_stamp()}-{n}"
            n += 1
        path.mkdir(parents=True)
        mapping = runner.mapping
        config = {k: v for k, v in runner.config.items() if not str(k).startswith("_")}
        self._meta = {
            "format": "glove_sim.session/1",
            "created": datetime.now().isoformat(timespec="seconds"),
            "t0_ns": time.monotonic_ns(),
            "source": runner.state.source if runner.state is not None else "",
            "joint_names": list(mapping.model_joint_names),
            "motor_ids": [m.motor_id for m in mapping.motors],
            "motor_limits_rad": [list(m.model_rad_limits()) for m in mapping.motors],
            "contact_groups": list(CONTACT_FINGERS),
            "torque_constant_nm_per_a": runner.torque_constant,
            "config": _jsonable(config),
            "fields": {k: {"unit": u, "meaning": d} for k, (u, d) in FIELDS.items()},
            "segments": [],
            "events": [],
        }
        self.path = path
        self.ticks = 0
        self._rows = []
        self._chunk_index = 0
        self._segment = -1
        self._segment_xml_ref = None
        self._last_glove_cfg = ""
        self._last_feedback = None
        self.error = ""
        self._thread = threading.Thread(target=self._writer, name="session-writer", daemon=True)
        self._thread.start()
        self._write_meta()
        return path

    def stop(self) -> Path | None:
        path = self.path
        if path is None:
            return None
        self._flush()
        self._meta["ticks"] = self.ticks
        self._meta["stopped"] = datetime.now().isoformat(timespec="seconds")
        self._write_meta()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._thread = None
        self.path = None
        return path

    def event(self, kind: str, t_ns: int | None = None, **data: Any) -> None:
        if self.path is None:
            return
        self._meta["events"].append({"t_ns": int(t_ns or time.monotonic_ns()), "kind": kind,
                                     **_jsonable(data)})

    # ------------------------------------------------------------------
    def record(self, runner: Any, result: Any, sample: Any) -> None:
        """Append one tick.  Called by the control thread with the runner lock held."""
        if self.path is None:
            return
        try:
            self._record(runner, result, sample)
        except Exception as error:  # never break the control loop
            self.error = f"记录失败: {error}"

    def _record(self, runner: Any, result: Any, sample: Any) -> None:
        sim = runner.sim
        if not self._meta.get("source"):
            self._meta["source"] = sample.source
        if sim.xml is not self._segment_xml_ref:
            self._new_segment(runner, sample.t_ns)
        glove_cfg = json.dumps(_jsonable(runner.config.get("glove", {})), sort_keys=True)
        if glove_cfg != self._last_glove_cfg:
            if self._last_glove_cfg:
                self.event("glove_config", sample.t_ns, glove=json.loads(glove_cfg))
            self._last_glove_cfg = glove_cfg
        if runner.feedback != self._last_feedback:
            if self._last_feedback is not None:
                self.event("feedback", sample.t_ns, enabled=bool(runner.feedback))
            self._last_feedback = bool(runner.feedback)

        state = runner.state
        nan12 = np.full(12, np.nan)
        raw = nan12.copy()
        temp = nan12.copy()
        err = np.zeros(12, dtype=bool)
        q_used = nan12.copy()
        if state is not None:
            q_used = np.asarray(state.angles_rad, dtype=float).copy()
            for i, motor in enumerate(runner.mapping.motors):
                mid = motor.motor_id
                v = (state.raw_deg or {}).get(mid)
                raw[i] = np.nan if v is None else float(v)
                tv = (state.temperature_c or {}).get(mid)
                temp[i] = np.nan if tv is None else float(tv)
                err[i] = mid in (state.errors or {})
        blocked = np.zeros(12, dtype=bool)
        surface = nan12.copy()
        proxy_k = nan12.copy()
        for idx, (_sign, frozen, k) in getattr(sim, "_blocked", {}).items():
            blocked[idx] = True
            surface[idx] = float(frozen)
            proxy_k[idx] = float(k)
        imp = runner.last_impedance
        if imp:
            imp_k = np.asarray([c.k for c in imp], dtype=float)
            imp_b = np.asarray([c.b for c in imp], dtype=float)
            imp_t = np.asarray([c.target_rad for c in imp], dtype=float)
            imp_ff = np.asarray([c.tau_ff for c in imp], dtype=float)
        else:
            imp_k = imp_b = imp_t = imp_ff = nan12
        self._rows.append({
            "t_ns": np.int64(sample.t_ns),
            "segment": np.int32(self._segment),
            "source": np.int8(1 if sample.source == "real" else 0),
            "feedback_on": bool(runner.feedback),
            "loop_hz": float(runner.stats.loop_hz),
            "tick_ms": float(runner.stats.tick_ms),
            "physics_steps": np.int32(runner.stats.steps_last_tick),
            "sensor_age_ms": float(sample.sensor_age_ms),
            "real_q": sample.real_q,
            "real_q_used": q_used,
            "real_dq": sample.real_dq,
            "raw_deg": raw,
            "encoder_clamped": sample.encoder_clamped,
            "current_a": sample.current_a,
            "temperature_c": temp,
            "motor_error": err,
            "torque_measured_nm": sample.torque_measured_nm,
            "sim_q": sample.sim_q,
            "sim_dq": sample.sim_dq,
            "motor_target": np.asarray(sim._motor_target, dtype=float).copy(),
            "blocked": blocked,
            "surface_q": surface,
            "proxy_k": proxy_k,
            "torque_spring_nm": sample.torque_spring_nm,
            "torque_driver_nm": sample.torque_driver_nm,
            "torque_sent_nm": sample.torque_sent_nm,
            "imp_k": imp_k,
            "imp_b": imp_b,
            "imp_target": imp_t,
            "imp_tau_ff": imp_ff,
            "finger_force_n": np.asarray(sample.finger_force_n, dtype=float),
            "penetration_m": np.asarray(
                [float(result.finger_penetration_m.get(f, 0.0)) for f in CONTACT_FINGERS]),
            "contact_count": np.int32(sample.contact_count),
            "object_disp_m": float(sample.object_disp_m),
            "qpos": np.asarray(sim.data.qpos, dtype=float).copy(),
            "qvel": np.asarray(sim.data.qvel, dtype=float).copy(),
            "tree_target": np.asarray(sim._q_tree_target, dtype=float).copy(),
        })
        self.ticks += 1
        if len(self._rows) >= self.chunk_ticks:
            self._flush()

    def _new_segment(self, runner: Any, t_ns: int) -> None:
        self._flush()
        sim = runner.sim
        self._segment += 1
        self._segment_xml_ref = sim.xml
        name = f"scene_{self._segment:03d}.xml"
        self._queue.put(("text", self.path / name, sim.xml))
        objects = []
        for obj in sim.object_runtime:
            spec = obj.spec
            objects.append({"name": spec.name, "label": spec.display_name, "shape": spec.shape,
                            "size_m": spec.size_m, "stiffness": spec.stiffness,
                            "deformable": spec.deformable, "qposadr": obj.qposadr})
        self._meta["segments"].append({
            "index": self._segment,
            "xml": name,
            "t_start_ns": int(t_ns),
            "nq": int(sim.model.nq),
            "tree_qposadr": [int(a) for a in sim._tree_qposadr],
            "motor_qposadr": [int(a) for a in sim.motor_qposadr],
            "objects": objects,
            "objects_locked": bool(sim.objects_locked),
        })
        self._write_meta()

    def _flush(self) -> None:
        if not self._rows or self.path is None:
            return
        rows, self._rows = self._rows, []
        path = self.path / f"chunk_{self._chunk_index:05d}.npz"
        self._chunk_index += 1
        self._queue.put(("chunk", path, rows))
        self._meta["ticks"] = self.ticks
        self._write_meta()

    def _write_meta(self) -> None:
        if self.path is not None:
            self._queue.put(("text", self.path / "meta.json",
                             json.dumps(self._meta, ensure_ascii=False, indent=1)))

    def _writer(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            kind, path, payload = item
            try:
                if kind == "text":
                    tmp = path.with_suffix(path.suffix + ".tmp")
                    tmp.write_text(payload, encoding="utf-8")
                    tmp.replace(path)
                else:
                    arrays = {key: np.stack([np.asarray(r[key]) for r in payload])
                              for key in payload[0]}
                    tmp = path.with_name(path.stem + ".tmp.npz")
                    np.savez(tmp, **arrays)
                    tmp.replace(path)
            except Exception as error:
                self.error = f"写入记录失败 {path.name}: {error}"


# ----------------------------------------------------------------------
@dataclass
class Contact:
    group: str               # thumb..pinky / palm
    geom: str
    object_name: str
    position_m: np.ndarray   # world
    force_world_n: np.ndarray
    normal_n: float
    penetration_m: float


class Session:
    """A recorded session loaded for offline analysis."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.meta: dict[str, Any] = json.loads((self.path / "meta.json").read_text(encoding="utf-8"))
        chunks = sorted(self.path.glob("chunk_*.npz"))
        chunks = [c for c in chunks if ".tmp" not in c.name]
        if not chunks:
            raise FileNotFoundError(f"{self.path} 里没有数据块（记录太短？）")
        parts: dict[str, list[np.ndarray]] = {}
        seg_parts: dict[int, dict[str, list[np.ndarray]]] = {}
        for chunk in chunks:
            with np.load(chunk) as z:
                seg = int(z["segment"][0])
                for key in z.files:
                    if key in PER_SEGMENT:
                        seg_parts.setdefault(seg, {}).setdefault(key, []).append(z[key])
                    else:
                        parts.setdefault(key, []).append(z[key])
        self.data: dict[str, np.ndarray] = {k: np.concatenate(v) for k, v in parts.items()}
        self._seg_data = {seg: {k: np.concatenate(v) for k, v in d.items()}
                          for seg, d in seg_parts.items()}
        order = np.argsort(self.data["t_ns"], kind="stable")
        if np.any(order != np.arange(order.size)):
            self.data = {k: v[order] for k, v in self.data.items()}
        t0 = int(self.meta.get("t0_ns", self.data["t_ns"][0]))
        self.t = (self.data["t_ns"] - t0) / 1e9
        # Row of each tick inside its segment's qpos array.
        seg = self.data["segment"]
        self._seg_row = np.zeros(seg.size, dtype=int)
        for s in np.unique(seg):
            idx = np.nonzero(seg == s)[0]
            self._seg_row[idx] = np.arange(idx.size)
        self._models: dict[int, Any] = {}

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.t.size)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.data[key]

    @property
    def fields(self) -> dict[str, dict[str, str]]:
        return self.meta.get("fields", {})

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.meta.get("events", [])

    def event_times(self) -> list[tuple[float, dict[str, Any]]]:
        t0 = int(self.meta.get("t0_ns", self.data["t_ns"][0]))
        return [((e["t_ns"] - t0) / 1e9, e) for e in self.events]

    def segment(self, index: int) -> dict[str, Any]:
        return self.meta["segments"][int(index)]

    def segment_of(self, i: int) -> dict[str, Any]:
        return self.segment(int(self.data["segment"][i]))

    def index_at(self, t_s: float) -> int:
        i = int(np.searchsorted(self.t, t_s))
        if i <= 0:
            return 0
        if i >= self.t.size:
            return int(self.t.size - 1)
        return i if self.t[i] - t_s < t_s - self.t[i - 1] else i - 1

    def scene_xml(self, segment: int) -> str:
        return (self.path / self.segment(segment)["xml"]).read_text(encoding="utf-8")

    def model(self, segment: int):
        import mujoco
        if segment not in self._models:
            self._models[segment] = mujoco.MjModel.from_xml_string(self.scene_xml(segment))
        return self._models[segment]

    def qpos_at(self, i: int, pose: str = "visible") -> np.ndarray:
        """Full scene qpos at tick i.  pose='target': glove at the unclamped real-hand pose."""
        seg = int(self.data["segment"][i])
        row = int(self._seg_row[i])
        qpos = self._seg_data[seg]["qpos"][row].copy()
        if pose == "target":
            adr = np.asarray(self.segment(seg)["tree_qposadr"], dtype=int)
            qpos[adr] = self._seg_data[seg]["tree_target"][row]
        elif pose != "visible":
            raise ValueError("pose must be 'visible' or 'target'")
        return qpos

    def mjdata_at(self, i: int, pose: str = "visible"):
        """MjData with the recorded state at tick i (mj_forward already run)."""
        import mujoco
        seg = int(self.data["segment"][i])
        model = self.model(seg)
        data = mujoco.MjData(model)
        data.qpos[:] = self.qpos_at(i, pose)
        qvel = self._seg_data[seg].get("qvel")
        if qvel is not None:
            data.qvel[:] = qvel[int(self._seg_row[i])]
            # The glove is kinematic: the live sim zeroes its qvel every step.
            dof = self._glove_dofs(seg)
            data.qvel[dof] = 0.0
        mujoco.mj_forward(model, data)
        return data

    def _glove_dofs(self, seg: int) -> np.ndarray:
        import mujoco
        model = self.model(seg)
        adr = set(int(a) for a in self.segment(seg)["tree_qposadr"])
        return np.asarray([model.jnt_dofadr[j] for j in range(model.njnt)
                           if int(model.jnt_qposadr[j]) in adr], dtype=int)

    def _classify(self, seg: int) -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
        """Same geom classification as Simulation._compile / compute_contacts."""
        import mujoco
        key = ("cls", seg)
        if key in self._models:
            return self._models[key]
        model = self.model(seg)
        glove: dict[int, str] = {}
        for gid in range(model.ngeom):
            if model.geom_contype[gid] != 1:
                continue
            name = model.geom(gid).name
            if name.startswith("col_"):
                parts = name.split("_", 2)
                if len(parts) >= 2:
                    glove[gid] = parts[1]
            elif model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                body = model.body(model.geom_bodyid[gid]).name
                glove[gid] = next((f for f in FINGERS if body.startswith(f)), "palm")
        objects = {o["name"] for o in self.segment(seg)["objects"]}
        obj_geom = {}
        for name in objects:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"obj_{name}_geom")
            if gid >= 0:
                obj_geom[gid] = name
        obj_flex = {}
        for name in objects:
            fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, f"flex_{name}")
            if fid >= 0:
                obj_flex[fid] = name
        self._models[key] = (glove, obj_geom, obj_flex)
        return self._models[key]

    def contacts_at(self, i: int, pose: str = "target") -> list[Contact]:
        """Hand-object contacts at tick i, recomputed from the recorded state.

        pose='target' (default) is what the force feedback used; 'visible'
        is the clamped hand drawn on screen.  Forces are force on the glove,
        world frame (same convention as Simulation.compute_contacts).  They are
        a reconstruction; the live per-group values are in 'finger_force_n'.
        """
        import mujoco
        seg = int(self.data["segment"][i])
        model = self.model(seg)
        glove, obj_geom, obj_flex = self._classify(seg)
        data = self.mjdata_at(i, pose)
        out: list[Contact] = []
        result = np.zeros(6)
        for k in range(data.ncon):
            con = data.contact[k]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 in glove:
                glove_gid, object_gid = g1, g2
            elif g2 in glove:
                glove_gid, object_gid = g2, g1
            else:
                continue
            flex_id = int(np.asarray(con.flex).reshape(-1)[1]) if np.ndim(con.flex) else -1
            obj = obj_flex.get(flex_id) or obj_geom.get(object_gid)
            if obj is None:
                continue
            mujoco.mj_contactForce(model, data, k, result)
            frame = np.asarray(con.frame, dtype=float).reshape(3, 3)
            world = frame.T @ result[:3]
            if glove_gid == g1:
                world = -world
            out.append(Contact(group=glove[glove_gid], geom=model.geom(glove_gid).name,
                               object_name=obj,
                               position_m=np.asarray(con.pos, dtype=float).copy(),
                               force_world_n=world, normal_n=float(result[0]),
                               penetration_m=max(0.0, -float(con.dist))))
        return out

    # ------------------------------------------------------------------
    def to_csv(self, path: str | Path | None = None) -> Path:
        """Flat CSV, one row per tick (qpos omitted; use the npz/Session for it)."""
        path = Path(path) if path is not None else self.path / "session.csv"
        joints = [f"J{i + 1}" for i in range(12)]
        cols: list[str] = ["t_s"]
        mats: list[np.ndarray] = [self.t[:, None]]
        for key, arr in self.data.items():
            if key in PER_SEGMENT:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 1:
                cols.append(key)
                mats.append(arr[:, None].astype(float))
            else:
                names = joints if arr.shape[1] == 12 else (
                    list(CONTACT_FINGERS) if arr.shape[1] == len(CONTACT_FINGERS)
                    else [str(j) for j in range(arr.shape[1])])
                cols += [f"{key}_{n}" for n in names]
                mats.append(arr.astype(float))
        table = np.hstack(mats)
        with path.open("w", encoding="utf-8") as stream:
            stream.write("# glove_sim session export; units/meanings in meta.json 'fields'. "
                         "Per-motor columns are model frame, motor-ID order J1..J12.\n")
            stream.write(",".join(cols) + "\n")
            np.savetxt(stream, table, delimiter=",", fmt="%.9g")
        return path


def load_session(path: str | Path) -> Session:
    return Session(path)


def latest_session(directory: str | Path) -> Path | None:
    sessions = sorted(Path(directory).glob("session-*/meta.json"), key=lambda p: p.stat().st_mtime)
    return sessions[-1].parent if sessions else None
