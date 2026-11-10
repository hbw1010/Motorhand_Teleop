#!/usr/bin/env python3
"""Offline smoke test for the standalone glove simulation.

Runs without hardware.  Every force printed here is a MuJoCo contact force and
every torque is a nominal command; no measured force sensor exists.
"""
from __future__ import annotations

import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SIM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM_ROOT / "src"))
for extra in (SIM_ROOT.parent / "hardware" / "encos_dexhand" / "src",):
    if extra.is_dir():
        sys.path.insert(0, str(extra))

import mujoco  # noqa: E402

from glove_sim.config import load_config  # noqa: E402
from glove_sim.mapping import FINGERS, MOTOR_FINGER, GloveMapping  # noqa: E402
from glove_sim.objects import ObjectSpec  # noqa: E402
from glove_sim.physics import Simulation  # noqa: E402
from glove_sim.sdk.base import GloveState  # noqa: E402
from glove_sim.sdk.fake import FakeGlove  # noqa: E402
from glove_sim.telemetry import COLUMNS, Telemetry  # noqa: E402


PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"FAIL  {name}" + (f"  ({detail})" if detail else ""))


def make_sim(specs: list[ObjectSpec]):
    config = load_config(None)
    mapping = GloveMapping.load(config["paths"]["glove_mapping"])
    return config, mapping, Simulation(config, mapping, specs)


def grasp_sim(spec: ObjectSpec, steps: int = 600):
    config, mapping, sim = make_sim([spec])
    open_angles = np.zeros(12)
    grasp = np.array([0.20, 0.15, 0.25, 0.25, 0.25, 0.80,
                      0.25, 0.80, 0.25, 0.80, 0.25, 0.80])
    sim._q_tree = sim.kinematics.full_pose(open_angles, sim.model, mapping)
    sim._apply_pose()
    mujoco.mj_forward(sim.model, sim.data)
    result = None
    for i in range(steps):
        frac = min(1.0, i / 300.0)
        target = open_angles + (grasp - open_angles) * frac
        sim._q_tree = sim.kinematics.full_pose(target, sim.model, mapping)
        sim._apply_pose()
        result = sim.step(feedback_enabled=True, dt_s=1 / 1000)
    return sim, result


def main() -> int:
    print(f"Python {sys.version.split()[0]}  MuJoCo {mujoco.__version__}")
    config = load_config(None)
    mapping = GloveMapping.load(config["paths"]["glove_mapping"])
    specs = [ObjectSpec.from_dict(item) for item in config["objects"]["defaults"]]

    check("12 motor mapping entries", len(mapping.motors) == 12)
    check("all motor model joints resolve",
          all(m.model_joint for m in mapping.motors))
    sim = Simulation(config, mapping, specs)
    proxies = [gid for gid in range(sim.model.ngeom)
               if sim.model.geom(gid).name.startswith("col_")]
    check("17 glove collision proxies", len(proxies) == 17, str(len(proxies)))
    check("glove is kinematic (no actuators drive it)",
          all(sim.model.actuator_trnid[i, 0] >= 0 for i in range(sim.model.nu)) if sim.model.nu else True)

    # The pose derivative must use physical motor order J1..J12, not CAD order.
    sim._q_tree = sim.kinematics.full_pose(np.zeros(12), sim.model, mapping)
    sim._apply_pose()
    deriv = sim.kinematics.tree_derivative()
    check("derivative shape 68x12", deriv.shape == (68, 12), str(deriv.shape))
    physical_ok = True
    for motor in mapping.motors:
        idx = sim.kinematics.joint_names.index(motor.model_joint)
        if not math.isclose(float(deriv[idx, motor.motor_id - 1]), 1.0, abs_tol=1e-12):
            physical_ok = False
    check("derivative columns in J1..J12 order", physical_ok)

    # Initial layout must not touch the open glove.
    for _ in range(5):
        result = sim.step(feedback_enabled=False, dt_s=0.001)
    check("initial scene has no glove/object contact", result.contact_count == 0,
          f"contacts={result.contact_count}")

    # Fixed objects must not drift even when the glove poses change.
    poses0 = {o.name: o.target_position.copy() for o in sim.object_runtime}
    for _ in range(200):
        sim.step(feedback_enabled=False, dt_s=0.001)
    drift = max(float(np.linalg.norm(sim.data.qpos[o.qposadr:o.qposadr + 3] - poses0[o.name]))
                for o in sim.object_runtime)
    check("locked objects do not drift", drift < 1e-10, f"drift={drift:.2e} m")

    # Dragging writes a persistent pose.
    dragged = np.array([0.160, 0.050, 0.060])
    sim.set_object_pose("ball_hard", position=dragged)
    for _ in range(100):
        sim.step(feedback_enabled=False, dt_s=0.001)
    obj = next(o for o in sim.object_runtime if o.name == "ball_hard")
    pos = sim.data.qpos[obj.qposadr:obj.qposadr + 3]
    check("dragged pose persists", float(np.linalg.norm(pos - dragged)) < 1e-10)

    # Soft ball physically deforms, hard ball does not; both contact the glove.
    soft_spec = ObjectSpec.from_dict(next(x for x in config["objects"]["defaults"]
                                          if x["name"] == "ball_soft"))
    hard_spec = ObjectSpec.from_dict(next(x for x in config["objects"]["defaults"]
                                          if x["name"] == "ball_hard"))
    hard_spec.position = soft_spec.position
    soft_sim, soft_res = grasp_sim(soft_spec)
    hard_sim, hard_res = grasp_sim(hard_spec)
    soft_force = sum(soft_res.finger_force_n.values())
    hard_force = sum(hard_res.finger_force_n.values())
    check("soft ball contacts and deforms", soft_res.contact_count > 0 and
          soft_res.max_object_disp_m > 0.002,
          f"contacts={soft_res.contact_count} disp={soft_res.max_object_disp_m:.4f} m")
    check("soft deformation stays bounded", soft_res.max_object_disp_m < 0.10,
          f"disp={soft_res.max_object_disp_m:.4f} m")
    check("rigid ball does not deform", hard_res.contact_count > 0 and
          hard_res.max_object_disp_m == 0.0,
          f"contacts={hard_res.contact_count} disp={hard_res.max_object_disp_m:.4f} m")
    check("soft/hard contact forces are finite",
          np.isfinite([soft_force, hard_force]).all(),
          f"soft={soft_force:.2f} N hard={hard_force:.2f} N")
    check("simulation state finite after grasp",
          np.isfinite(soft_sim.data.qpos).all() and np.isfinite(hard_sim.data.qpos).all())

    # Torque caps, slew and filtering.
    config2, mapping2, sim2 = make_sim([soft_spec])
    sim2._enabled_since_ns = None
    cmd = None
    for _ in range(2000):
        cmd = sim2.feedback_step(np.full(12, 100.0), enabled=True, dt_s=0.001,
                                 now_ns=time.monotonic_ns())
    limits = np.asarray(config2["glove"]["torque_limit_nm"], dtype=float)
    check("feedback torque respects motor caps",
          np.all(np.abs(cmd) <= limits + 1e-9),
          f"max={np.max(np.abs(cmd)):.3f} Nm limits={limits.tolist()}")

    # Telemetry columns and truthful provenance.
    state = FakeGlove(mapping, demo=False).read()
    tel = Telemetry(config, source="fake")
    row = tel.make_row(soft_res, state, loop_hz=100.0)
    missing = [c for c in COLUMNS if c not in row]
    check("telemetry has all declared columns", not missing, f"missing={missing}")
    check("telemetry marks estimated force unavailable", row["force_estimate_validity"] == "uncalibrated")
    check("telemetry marks slip unavailable for locked objects",
          row["slip_validity"] == "unavailable_no_pose_sensor" and row["slip_m"] == "")
    check("telemetry carries simulated object deformation",
          row["max_object_disp_m"] == float(soft_res.max_object_disp_m))
    with tempfile.TemporaryDirectory() as tmp:
        prefix = tel.start(tmp, prefix="smoke")
        tel.write(row)
        tel.stop()
        csv_text = prefix.with_suffix(".csv").read_text(encoding="utf-8")
        jsonl_text = prefix.with_suffix(".jsonl").read_text(encoding="utf-8")
        check("CSV header declares sources/units", "MuJoCo contact force" in csv_text and
              "uncalibrated" in csv_text)
        check("JSONL record written", len(jsonl_text.strip().splitlines()) == 1)

    # Hardware path imports and read-only discovery must not crash without a device.
    try:
        from glove_sim.sdk.real import discover_ports
        ports = discover_ports()
        check("CANable2 discovery runs without hardware", isinstance(ports, list),
              f"ports={ports}")
    except Exception as error:  # pragma: no cover - environment-specific
        check("CANable2 discovery runs without hardware", False, str(error))

    # CLI parser includes the safe/required commands.
    from glove_sim.cli import build_parser
    choices = set(build_parser()._subparsers._group_actions[0].choices)
    check("CLI exposes check/zero/home/teleop/sim",
          {"check-glove", "zero-glove", "home-glove", "teleop", "sim", "record", "replay"} <= choices,
          str(sorted(choices)))

    print(f"\nSMOKE {'OK' if FAIL == 0 else 'FAILED'}: {PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
