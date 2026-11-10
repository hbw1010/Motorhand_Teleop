from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import pytest

from glove_sim.config import load_config
from glove_sim.mapping import GloveMapping
from glove_sim.objects import ObjectSpec
from glove_sim.physics import Simulation
from glove_sim.sdk.fake import FakeGlove
from glove_sim.telemetry import COLUMNS, Telemetry
from glove_sim.sdk.zero import write_zero


@pytest.fixture(scope="module")
def setup():
    config = load_config(None)
    mapping = GloveMapping.load(config["paths"]["glove_mapping"])
    specs = [ObjectSpec.from_dict(item) for item in config["objects"]["defaults"]]
    return config, mapping, specs


def _grasp(spec: ObjectSpec, config, mapping, steps: int = 550):
    sim = Simulation(config, mapping, [spec])
    open_q = np.zeros(12)
    grasp = np.array([0.20, 0.15, 0.25, 0.25, 0.25, 0.80,
                      0.25, 0.80, 0.25, 0.80, 0.25, 0.80])
    sim._q_tree = sim.kinematics.full_pose(open_q, sim.model, mapping)
    sim._apply_pose()
    mujoco.mj_forward(sim.model, sim.data)
    result = None
    for i in range(steps):
        p = min(1.0, i / 200.0)
        target = open_q + (grasp - open_q) * p
        sim._q_tree = sim.kinematics.full_pose(target, sim.model, mapping)
        sim._apply_pose()
        result = sim.step(feedback_enabled=True, dt_s=0.001)
    return sim, result


def test_mapping_and_collisions(setup):
    config, mapping, specs = setup
    assert len(mapping.motors) == 12
    sim = Simulation(config, mapping, specs)
    proxies = [gid for gid in range(sim.model.ngeom)
               if sim.model.geom(gid).name.startswith("col_")]
    assert len(proxies) == 17
    # Every declared motor joint must exist in the compiled scene.
    for motor in mapping.motors:
        assert mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, motor.model_joint) >= 0


def test_derivative_column_order(setup):
    config, mapping, _ = setup
    sim = Simulation(config, mapping, [])
    sim._q_tree = sim.kinematics.full_pose(np.zeros(12), sim.model, mapping)
    sim._apply_pose()
    deriv = sim.kinematics.tree_derivative()
    assert deriv.shape == (68, 12)
    for motor in mapping.motors:
        idx = sim.kinematics.joint_names.index(motor.model_joint)
        assert math.isclose(float(deriv[idx, motor.motor_id - 1]), 1.0, abs_tol=1e-12)


def test_locked_objects_and_drag(setup):
    config, mapping, specs = setup
    sim = Simulation(config, mapping, specs)
    result = sim.step(feedback_enabled=False, dt_s=0.001)
    assert result.contact_count == 0
    start = {o.name: o.target_position.copy() for o in sim.object_runtime}
    for _ in range(100):
        sim.step(feedback_enabled=False, dt_s=0.001)
    for obj in sim.object_runtime:
        got = sim.data.qpos[obj.qposadr:obj.qposadr + 3]
        assert np.linalg.norm(got - start[obj.name]) < 1e-10
    target = np.array([0.16, 0.05, 0.06])
    sim.set_object_pose("ball_hard", position=target)
    for _ in range(50):
        sim.step(feedback_enabled=False, dt_s=0.001)
    obj = next(o for o in sim.object_runtime if o.name == "ball_hard")
    got = sim.data.qpos[obj.qposadr:obj.qposadr + 3]
    assert np.linalg.norm(got - target) < 1e-10


def test_soft_deforms_hard_does_not(setup):
    config, mapping, specs = setup
    soft = next(s for s in specs if s.name == "ball_soft")
    hard = ObjectSpec(**{**next(s for s in specs if s.name == "ball_hard").__dict__,
                         "position": soft.position})
    soft_sim, soft_result = _grasp(soft, config, mapping)
    hard_sim, hard_result = _grasp(hard, config, mapping)
    assert soft_result.contact_count > 0
    assert hard_result.contact_count > 0
    assert soft_result.max_object_disp_m > 0.002
    assert soft_result.max_object_disp_m < 0.10
    assert hard_result.max_object_disp_m == 0.0
    assert np.isfinite(soft_sim.data.qpos).all()
    assert np.isfinite(hard_sim.data.qpos).all()


def test_kinematic_clamp_prevents_locked_object_penetration(setup):
    """A commanded finger must stop at a locked rigid object, while the ghost
    target pose still produces a meaningful feedback force."""
    config, mapping, specs = setup
    soft = next(s for s in specs if s.name == "ball_soft")
    hard = ObjectSpec(**{**next(s for s in specs if s.name == "ball_hard").__dict__,
                         "position": soft.position})
    sim = Simulation(config, mapping, [hard])
    sim.set_objects_locked(True)
    grasp = np.array([0.20, 0.15, 0.25, 0.25, 0.25, 0.80,
                      0.25, 0.80, 0.25, 0.80, 0.25, 0.80])
    max_visual_pen = 0.0
    result = None
    for i in range(260):
        sim.update_glove(grasp * min(1.0, i / 150.0))
        for _ in range(2):
            result = sim.step(feedback_enabled=True, dt_s=0.001)
            pen = max([-float(sim.data.contact[j].dist)
                       for j in range(sim.data.ncon)] + [0.0])
            max_visual_pen = max(max_visual_pen, pen)
    assert result is not None
    assert max_visual_pen < 1e-3
    assert result.contact_count > 0
    assert sum(result.finger_force_n.values()) > 0.05
    assert np.max(np.abs(result.torque_cmd_nm)) > 0.05
    assert np.max(np.abs(sim._motor_target - sim._motor_rad)) > 0.02
    assert np.isfinite(sim.data.qpos).all()

    # Opening the hand must release the temporary per-finger blocks.
    for i in range(180):
        sim.update_glove(grasp * (1.0 - min(1.0, i / 100.0)))
        for _ in range(2):
            sim.step(feedback_enabled=True, dt_s=0.001)
    assert not sim._blocked
    assert np.max(np.abs(sim._motor_target - sim._motor_rad)) < 1e-9


def test_feedback_caps(setup):
    config, mapping, specs = setup
    sim = Simulation(config, mapping, specs)
    sim._enabled_since_ns = None
    command = np.zeros(12)
    for i in range(1500):
        command = sim.feedback_step(np.full(12, 100.0), enabled=True,
                                    dt_s=0.001, now_ns=1_000_000_000 + i * 1_000_000)
    limits = np.asarray(config["glove"]["torque_limit_nm"], dtype=float)
    assert np.all(np.abs(command) <= limits + 1e-9)
    assert np.isfinite(command).all()


def test_telemetry_provenance(setup):
    config, mapping, specs = setup
    sim = Simulation(config, mapping, specs)
    result = sim.step(feedback_enabled=False, dt_s=0.001)
    state = FakeGlove(mapping, demo=False).read()
    telemetry = Telemetry(config, source="fake")
    row = telemetry.make_row(result, state, loop_hz=100.0)
    assert set(COLUMNS) <= set(row)
    assert row["data_validity"] == "simulated"
    assert row["force_estimate_validity"] == "uncalibrated"
    assert row["slip_validity"] == "unavailable_no_pose_sensor"
    assert row["estimated_force_1_n"] == ""
    with tempfile.TemporaryDirectory() as tmp:
        prefix = telemetry.start(tmp, prefix="test")
        telemetry.write(row)
        telemetry.stop()
        text = prefix.with_suffix(".csv").read_text(encoding="utf-8")
        assert "MuJoCo contact force" in text
        assert "uncalibrated" in text


def test_ui_module_imports():
    from glove_sim.ui import app
    assert callable(app.run_ui)


def test_zeroing_preserves_relative_limits(setup):
    config, _, _ = setup
    source = Path(config["paths"]["glove_mapping"])
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "glove_mapping.json"
        shutil.copy2(source, target)
        before = json.loads(target.read_text(encoding="utf-8"))
        motor = before["motors"]["5"]
        old_zero = float(motor["zero_deg"])
        old_span = (float(motor["max_deg"]) - old_zero,
                    old_zero - float(motor["min_deg"]))
        new_zero = old_zero + 3.25
        write_zero(target, {5: new_zero}, ("index",))
        after = json.loads(target.read_text(encoding="utf-8"))
        motor = after["motors"]["5"]
        assert math.isclose(float(motor["zero_deg"]), new_zero, abs_tol=1e-9)
        assert math.isclose(float(motor["max_deg"]) - new_zero, old_span[0], abs_tol=1e-9)
        assert math.isclose(new_zero - float(motor["min_deg"]), old_span[1], abs_tol=1e-9)
        assert "index" in after["zero_calibrated_fingers"]
        assert (Path(tmp) / "joint_calibration.json").is_file()
