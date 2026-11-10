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


def test_fast_kinematics_matches_reference_solver(setup):
    """Vectorized closure solver must agree with the reference CAD solver."""
    from glove_sim.kinematics import GloveKinematics

    config, mapping, _specs = setup
    model_dir = Path(config["paths"]["glove_model"]).parent
    fast = GloveKinematics(model_dir, mapping, fast=True)
    ref = GloveKinematics(model_dir, mapping, fast=False)
    grasp = np.array([0.20, 0.15, 0.25, 0.25, 0.25, 0.80,
                      0.25, 0.80, 0.25, 0.80, 0.25, 0.80])
    for p in (0.0, 0.3, 0.7, 1.0, 0.5):
        a = fast.full_pose(grasp * p, None, mapping)
        b = ref.full_pose(grasp * p, None, mapping)
        assert np.abs(a - b).max() < 1e-5
        assert fast.last_gap_m <= 1e-7
        da, db = fast.tree_derivative(), ref.tree_derivative()
        assert np.abs(da - db).max() < 1e-4
    assert fast.fallback_count == 0


def test_control_thread_runs_at_fixed_rate(setup):
    import time as _time

    from glove_sim.runner import Runner

    config, mapping, specs = setup
    runner = Runner(config, mapping, FakeGlove(mapping), specs, feedback=False)
    runner.start()
    runner.start_thread()
    try:
        _time.sleep(1.5)
        hz = runner.stats.loop_hz
    finally:
        runner.close()
    assert runner._thread is None
    target = float(config["simulation"]["control_hz"])
    assert hz > 0.6 * target


def test_ui_locked_callbacks_receive_dearpygui_arguments():
    """DearPyGui passes args by the callable's co_argcount; *args gets none."""
    import threading
    from types import SimpleNamespace

    from glove_sim.ui.app import App

    calls = []

    class Fake:
        runner = SimpleNamespace(lock=threading.RLock())

        def two(self, sender, app_data):
            calls.append(("two", sender, app_data))

        def zero(self):
            calls.append(("zero",))

    fake = Fake()
    for fn in (fake.two, fake.zero):
        wrapped = App._locked(fake, fn)
        assert wrapped.__code__.co_argcount == 3
        wrapped("item", True, None)
    assert calls == [("two", "item", True), ("zero",)]


def _place_at_fingertips(sim, spec):
    mujoco.mj_forward(sim.model, sim.data)
    tips = [sim.data.geom_xpos[g].copy() for g in range(sim.model.ngeom)
            if (sim.model.geom(g).name or "").startswith(("col_index_index_dip", "col_middle_middle_dip"))]
    tip = np.mean(tips, axis=0)
    sim.set_object_pose(spec.name, position=np.array([tip[0], tip[1], tip[2] + 0.5 * spec.size_m + 0.054]))


def test_proxy_spring_grows_with_depth_and_holds_through_object(setup):
    config, mapping, specs = setup
    spec = next(s for s in specs if s.name == "box_hard")
    config = json.loads(json.dumps(config))
    config["glove"]["feedback_mode"] = "proxy_spring"
    sim = Simulation(config, mapping, [spec])
    assert sim._feedback_mode() == "proxy_spring"
    _place_at_fingertips(sim, spec)
    grasp = np.array([0.20, 0.15, 0.25, 0.25, 0.25, 0.80,
                      0.25, 0.80, 0.25, 0.80, 0.25, 0.80]) * 1.6
    raw_by_p = []
    for i in range(1300):
        p = min(1.0, i / 1200)
        sim.update_glove(grasp * p)
        r = sim.step(feedback_enabled=True, dt_s=0.001)
        raw_by_p.append(float(np.abs(r.torque_raw_nm).max()))
    # Contact starts mid-way; afterwards the spring torque never decreases
    # while squeezing further, even after the target passes through the box.
    first = next(i for i, v in enumerate(raw_by_p) if v > 0)
    tail = np.array(raw_by_p[first:])
    assert np.all(np.diff(tail) >= -1e-9)
    assert tail[-1] > 1.0
    assert sim._blocked, "proxy must stay engaged at full closure"
    # Push-back opposes closing (closing = + in model joints).
    for idx in sim._blocked:
        assert r.torque_raw_nm[idx] < 0
    # Visible hand stays outside the locked box.
    pen = max([-float(sim.data.contact[j].dist) for j in range(sim.data.ncon)] + [0.0])
    assert pen < 1e-3
    # Opening releases the proxy and the torque.
    for i in range(400):
        sim.update_glove(grasp * (1.0 - min(1.0, i / 300)))
        r = sim.step(feedback_enabled=True, dt_s=0.001)
    assert not sim._blocked
    assert np.abs(r.torque_raw_nm).max() == 0.0


def test_motor_axis_torque_follows_mapping_sign(setup):
    from glove_sim.sdk.real import RealGlove

    _config, mapping, _specs = setup
    glove = RealGlove(mapping)
    for motor in mapping.motors:
        out = glove.motor_axis_torque(motor.motor_id, 1.0)
        assert out == pytest.approx(motor.sign * motor.scale)


def test_contact_force_is_force_on_glove(setup):
    config, mapping, specs = setup
    spec = next(s for s in specs if s.name == "box_hard")
    sim = Simulation(config, mapping, [spec])
    sim.set_objects_locked(False)
    _place_at_fingertips(sim, spec)
    grasp = np.array([0.20, 0.15, 0.25, 0.25, 0.25, 0.80,
                      0.25, 0.80, 0.25, 0.80, 0.25, 0.80]) * 1.6
    checked = 0
    for i in range(900):
        sim.update_glove(grasp * min(1.0, i / 800))
        sim.step(feedback_enabled=False, dt_s=0.001)
        contacts, _f, _p = sim.compute_contacts()
        for c in contacts:
            gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom_name)
            away = sim.data.geom_xpos[gid] - c.position_m  # contact point -> glove geom centre
            if np.linalg.norm(c.force_world_n) > 1e-3 and np.linalg.norm(away) > 1e-6:
                assert np.dot(c.force_world_n, away) > 0
                checked += 1
    assert checked > 0


def test_large_ball_allowed_and_nudged_clear_of_neighbours(setup):
    config, mapping, _specs = setup
    lo, hi = config["objects"]["size_range_m_by_shape"]["sphere"]
    assert hi == pytest.approx(0.200)
    assert config["objects"]["size_range_m"][1] == pytest.approx(0.070)
    specs = [ObjectSpec.from_dict(item) for item in config["objects"]["defaults"]]
    sim = Simulation(config, mapping, specs)
    sim.set_object_shape("ball_soft", size_m=hi)
    obj = next(o for o in sim.object_runtime if o.name == "ball_soft")
    sim.set_object_pose("ball_soft", position=sim.free_position("ball_soft", obj.target_position))
    big = next(s for s in sim.specs if s.name == "ball_soft")
    for other in sim.object_runtime:
        if other.name == "ball_soft":
            continue
        spec = next(s for s in sim.specs if s.name == other.name)
        gap = np.linalg.norm(obj.target_position - other.target_position) \
            - sim.bounding_radius(big) - sim.bounding_radius(spec)
        assert gap > 0
    for _ in range(20):
        sim.update_glove(np.zeros(12))
        sim.step(feedback_enabled=False, dt_s=0.001)
    assert sim.data.ncon == 0
    assert np.isfinite(sim.data.qpos).all()


def test_runner_records_torque_actually_sent(setup):
    """The 仿真 vs 真机 plot shows runner.sent_torque_nm: what was queued per motor."""
    from glove_sim.runner import Runner

    config, mapping, specs = setup
    runner = Runner(config, mapping, FakeGlove(mapping), specs, feedback=True)
    try:
        runner.start()
        runner.sim.feedback_step = lambda raw, **_: np.linspace(-0.3, 0.3, 12)
        result = runner.tick(max_steps=1, wall_clock=False)
        assert np.allclose(runner.sent_torque_nm, result.torque_cmd_nm)
        runner.set_feedback(False)
        assert not np.any(runner.sent_torque_nm)
        runner.tick(max_steps=1, wall_clock=False)
        assert not np.any(runner.sent_torque_nm)
    finally:
        runner.close()


def _font_cmap(path: Path) -> set[int]:
    """Code points of a TrueType/OpenType font (cmap formats 4 and 12)."""
    import struct

    data = path.read_bytes()
    num_tables = struct.unpack(">H", data[4:6])[0]
    cmap_off = None
    for i in range(num_tables):
        tag, _, off, _ = struct.unpack(">4sIII", data[12 + 16 * i:28 + 16 * i])
        if tag == b"cmap":
            cmap_off = off
    assert cmap_off is not None
    codes: set[int] = set()
    n_sub = struct.unpack(">H", data[cmap_off + 2:cmap_off + 4])[0]
    for i in range(n_sub):
        _, _, off = struct.unpack(">HHI", data[cmap_off + 4 + 8 * i:cmap_off + 12 + 8 * i])
        sub = cmap_off + off
        fmt = struct.unpack(">H", data[sub:sub + 2])[0]
        if fmt == 4:
            seg2 = struct.unpack(">H", data[sub + 6:sub + 8])[0]
            ends = struct.unpack(f">{seg2 // 2}H", data[sub + 14:sub + 14 + seg2])
            starts = struct.unpack(f">{seg2 // 2}H", data[sub + 16 + seg2:sub + 16 + 2 * seg2])
            for a, b in zip(starts, ends):
                codes.update(range(a, b + 1))
        elif fmt == 12:
            n = struct.unpack(">I", data[sub + 12:sub + 16])[0]
            for k in range(n):
                a, b, _ = struct.unpack(">III", data[sub + 16 + 12 * k:sub + 28 + 12 * k])
                codes.update(range(a, b + 1))
    return codes


def test_ui_font_has_every_character_the_sim_displays():
    """A glyph missing from the bundled subset font renders as '?'."""
    import glove_sim.ui.app as app

    font = Path(app.FONT_CANDIDATES[0])
    cmap = _font_cmap(font)
    root = Path(app.__file__).resolve().parents[1]
    missing = set()
    for path in list(root.rglob("*.py")) + list((root.parents[1] / "config").rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        missing |= {c for c in text if ord(c) > 0x7F and ord(c) not in cmap}
    assert not missing, f"{font.name} 缺字: {''.join(sorted(missing))}（重新子集化字体）"


def test_paused_plots_keep_the_frozen_snapshot():
    """Pause freezes the curves; switching motor redraws the frozen data, not live data."""
    import glove_sim.ui.app as app

    calls = []
    fake = type("Dpg", (), {
        "set_value": staticmethod(lambda tag, v: calls.append((tag, v))),
        "set_axis_limits": staticmethod(lambda *a: None),
        "configure_item": staticmethod(lambda *a, **k: None),
        "bind_item_theme": staticmethod(lambda *a: None),
    })
    original = app.dpg
    app.dpg = fake
    try:
        ui = app.App.__new__(app.App)
        ui._history = app.deque(maxlen=10)
        ui._zoom, ui._auto_lims, ui._plot_data = {}, {}, {}
        ui._time_span, ui._cmp_motor = 10.0, 0
        ui._plots_paused, ui._frozen_hist, ui._frozen_t = False, [], np.zeros(0)
        ui._danger_theme = 0

        def row(t, value):
            z = [0.0] * 12
            col = [value] * 12
            return (t, [0.0] * 6, col, z, [0.0] * 5, 0.0, 0, 0.0, z, z, col, col, z)

        ui._history.extend([row(0.0, 1.0), row(0.1, 1.0)])
        ui._on_pause_plots()
        ui._history.append(row(0.2, 9.0))  # live data keeps arriving
        ui._on_cmp_motor(None, "J8 中指")
        frozen = dict(calls)["cmp_tau_sent"]
        assert ui._cmp_motor == 7 and frozen[0] == [0.0, 0.1] and max(frozen[1]) == 1.0
        ui._on_pause_plots()  # resume -> live
        calls.clear()
        ui._update_plots()
        assert dict(calls)["cmp_tau_sent"][0][-1] == 0.2
    finally:
        app.dpg = original


# ---------------------------------------------------------------- impedance
def _planner(**over):
    from glove_sim.impedance import DriverImpedancePlanner

    cfg = {"gain": 1.0, "damping_s": 0.004, "damping_max": 0.3, "k_max_nm_per_rad": 80.0,
           "release_lookahead_s": 0.04, "torque_margin": 0.85}
    cfg.update(over)
    return DriverImpedancePlanner(cfg, [4.0] * 12)


def test_driver_impedance_spring_saturates_and_is_one_sided():
    planner = _planner()
    q = np.zeros(12)
    dq = np.zeros(12)
    # Motor 5 closes with sign +1 onto a surface at 0.30 rad, k = 40 Nm/rad.
    for depth, expected in ((0.0, 0.0), (0.02, -0.8), (0.05, -2.0), (0.5, -3.4)):
        q[5] = 0.30 + depth
        cmd = planner.plan(q, dq, {5: (1, 0.30, 40.0)})[5]
        assert cmd.torque(q[5]) == pytest.approx(expected, abs=1e-9)  # pushes open, capped at 85% of 4 Nm
    # Free motors carry only friction feed-forward and no stiffness.
    cmds = planner.plan(q, dq, {5: (1, 0.30, 40.0)}, friction=np.full(12, 0.2))
    assert cmds[0].k == 0.0 and cmds[0].tau_ff == pytest.approx(0.2)
    assert cmds[5].tau_ff == 0.0  # no friction assist into the object


def test_driver_impedance_rebound_is_smooth_damped_and_never_pulls():
    planner = _planner(damping_s=0.008, damping_max=0.5, release_lookahead_s=0.02)
    q = np.zeros(12)
    dq = np.zeros(12)
    q[5] = 0.31  # 10 mrad inside
    dq[5] = +0.5  # pressing in -> spring + damping
    cmd = planner.plan(q, dq, {5: (1, 0.30, 40.0)})[5]
    assert cmd.k == pytest.approx(40.0) and cmd.b == pytest.approx(0.32)
    assert cmd.torque(q[5]) == pytest.approx(-0.4)
    # Rebound at 0.1 rad/s: push shrinks smoothly (10 -> 8 mrad), damping stays on.
    dq[5] = -0.1
    cmd = planner.plan(q, dq, {5: (1, 0.30, 40.0)})[5]
    assert cmd.torque(q[5]) == pytest.approx(-0.32) and cmd.b > 0.0
    # Fast withdrawal: damping switches off (hysteresis) and the push reaches
    # 0 -> kp dropped, so the PD can never pull the finger back in.
    dq[5] = -0.6
    cmd = planner.plan(q, dq, {5: (1, 0.30, 40.0)})[5]
    assert cmd.k == 0.0 and cmd.b == 0.0
    # Damping only re-arms once the finger stops withdrawing (no chatter at 0).
    dq[5] = -0.1
    q[5] = 0.33
    assert planner.plan(q, dq, {5: (1, 0.30, 40.0)})[5].b == 0.0
    dq[5] = -0.01
    assert planner.plan(q, dq, {5: (1, 0.30, 40.0)})[5].b > 0.0
    # Torque is continuous in the rebound velocity (no on/off switching).
    q[5] = 0.31
    taus = []
    for v in np.linspace(0.0, -0.4, 41):
        dq[5] = v
        taus.append(planner.plan(q, dq, {5: (1, 0.30, 40.0)})[5].torque(q[5]))
    assert max(abs(a - b) for a, b in zip(taus, taus[1:])) < 0.05
    assert all(t <= 0.0 for t in taus)  # never pulls inward
    # Opposite closing sign (sign -1 motor): pushing toward larger angles.
    q[5], dq[5] = 0.28, 0.0
    cmd = planner.plan(q, dq, {5: (-1, 0.30, 40.0)})[5]
    assert cmd.torque(q[5]) == pytest.approx(+0.8)
    cmd = planner.plan(q, dq, {5: (-1, 0.30, 40.0)}, ramp=0.25)[5]
    assert cmd.k == pytest.approx(10.0)


def test_contact_release_has_hysteresis(setup):
    """Just outside the surface the wall is kept (spring 0, visible finger free)."""
    config, mapping, specs = setup
    config = json.loads(json.dumps(config))
    config["glove"]["release_margin_rad"] = 0.01
    sim = Simulation(config, mapping, specs[:1])
    q = np.zeros(12)
    sim.update_glove(q)
    sim._blocked[5] = (1, 0.20, 30.0)
    q[5] = 0.195  # 5 mrad outside: kept, not clamped
    sim.update_glove(q)
    assert 5 in sim._blocked and sim._motor_rad[5] == pytest.approx(0.195)
    q[5] = 0.25  # back inside: same wall, visible finger held at the surface
    sim.update_glove(q)
    assert sim._blocked[5][1] == 0.20 and sim._motor_rad[5] == pytest.approx(0.20)
    q[5] = 0.185  # 15 mrad outside: released
    sim.update_glove(q)
    assert 5 not in sim._blocked


def test_friction_compensation_is_bounded_and_follows_velocity():
    from glove_sim.impedance import FrictionCompensator

    fc = FrictionCompensator({"enabled": True, "coulomb_nm": 0.3, "velocity_scale_rad_s": 0.08,
                              "gain": 1.0, "max_nm": 0.25, "velocity_deadband_rad_s": 0.0})
    tau = fc.torque(np.array([0.0, 0.01, -0.01, 1.0, -1.0] + [0.0] * 7))
    assert tau[0] == 0.0
    assert 0.0 < tau[1] < 0.05 and tau[2] == pytest.approx(-tau[1])
    assert tau[3] == pytest.approx(0.25) and tau[4] == pytest.approx(-0.25)  # clipped to max_nm
    assert not np.any(FrictionCompensator({"enabled": False, "coulomb_nm": 0.3}).torque(np.ones(12)))


def test_real_glove_impedance_uses_raw_offset_and_mit_range(setup):
    """Target = latest raw angle + offset, so a clamped model angle cannot become a pull."""
    import math as _m

    from glove_sim.impedance import ImpedanceCommand
    from glove_sim.sdk.real import RealGlove

    _config, mapping, _specs = setup
    sent = []

    class Hand:
        def set_mit(self, **kw):
            sent.append(kw)

    glove = RealGlove(mapping)
    glove._hand, glove._enabled = Hand(), True
    motor = mapping.motors[7]  # J8, sign -1
    # Raw angle far outside the calibrated range: the model angle is clamped,
    # the command must still be relative to the true raw angle.
    glove._last_angles_deg = {8: -44.0}
    q_model = 1.0
    cmd = ImpedanceCommand(target_rad=q_model - (-1) * 0.05, k=40.0, b=0.1)  # 0.05 rad spring stretch
    tau = glove.command_impedance(8, cmd, q_model)
    kw = sent[-1]
    ratio = motor.sign * motor.scale
    assert kw["target_deg"] == pytest.approx(-44.0 + _m.degrees(0.05) / ratio)
    assert kw["kp"] == pytest.approx(40.0 * motor.scale ** 2) and kw["kd"] == pytest.approx(0.1)
    assert abs(tau) == pytest.approx(40.0 * 0.05, rel=1e-6)
    # Outside the +-5 rad MIT position field: degrade to an equivalent feed-forward torque.
    glove._last_angles_deg = {8: 300.0}
    tau = glove.command_impedance(8, cmd, q_model)
    assert sent[-1]["kp"] == 0.0 and sent[-1]["target_deg"] == 300.0
    assert sent[-1]["torque_nm"] == pytest.approx(tau) and abs(tau) == pytest.approx(2.0, rel=1e-6)


def test_runner_drives_impedance_in_contact(setup):
    from glove_sim.runner import Runner

    config, mapping, specs = setup
    config = json.loads(json.dumps(config))
    config["glove"]["feedback_mode"] = "driver_impedance"
    config["glove"]["stale_release_clearance_m"] = 1e9  # synthetic block with no object nearby
    runner = Runner(config, mapping, FakeGlove(mapping), specs, feedback=True)
    try:
        runner.start()
        runner.tick(max_steps=1, wall_clock=False)
        assert runner.last_impedance is not None and all(c.k == 0.0 for c in runner.last_impedance)
        q = runner.state.angles_rad
        runner.sim._blocked[5] = (1, float(q[5]) - 0.05, 30.0)  # finger 0.05 rad inside an object
        runner.sim._enabled_since_ns = 0  # ramp complete
        runner.sim.update_glove = lambda _q: None  # keep the block for this tick
        runner.tick(max_steps=1, wall_clock=False)
        cmd = runner.last_impedance[5]
        assert cmd.k == pytest.approx(30.0)
        assert runner.sent_torque_nm[5] == pytest.approx(-1.5, abs=0.05)
        assert runner.result.torque_cmd_nm[5] == pytest.approx(runner.sent_torque_nm[5], abs=0.05)
    finally:
        runner.close()


def test_friction_compensation_has_rest_deadband():
    from glove_sim.impedance import FrictionCompensator

    fc = FrictionCompensator({"enabled": True, "coulomb_nm": 0.3, "velocity_deadband_rad_s": 0.02})
    tau = fc.torque(np.array([0.019, -0.019, 0.2, -0.2] + [0.0] * 8))
    assert tau[0] == 0.0 and tau[1] == 0.0
    assert tau[2] > 0.25 and tau[3] < -0.25


def _palm_side_probe(sim, finger, spec):
    """Place a small ball 2 mm into the most palm-side visible point of a finger."""
    model, data = sim.model, sim.data
    mujoco.mj_forward(model, data)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    rb, pb = data.xmat[base].reshape(3, 3), data.xpos[base]
    best = None
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if not model.body(model.geom_bodyid[g]).name.startswith(finger):
            continue
        mid = model.geom_dataid[g]
        a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        pts = model.mesh_vert[a:a + n] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        z = (pts - pb) @ rb[:, 2]  # +z of the hand frame = palm side (fingers curl toward +z)
        i = int(np.argmax(z))
        if best is None or z[i] > best[0]:
            best = (z[i], pts[i])
    radius = 0.5 * spec.size_m
    centre = best[1] + rb[:, 2] * (radius - 0.002)
    sim.set_object_pose(spec.name, position=centre)
    sim._lock_objects()
    mujoco.mj_forward(model, data)
    contacts, _force, _pen = sim.compute_contacts()
    return {c.finger for c in contacts}


def test_mesh_collision_covers_the_visible_palm_side(setup):
    """The glove collides with what is drawn, including the palm-side finger rings."""
    config, mapping, specs = setup
    for geometry, expect_hit in (("mesh", True), ("spheres", False)):
        cfg = json.loads(json.dumps(config))
        cfg["glove"]["collision_geometry"] = geometry
        ball = next(d for d in cfg["objects"]["defaults"] if d["name"] == "ball_hard")
        sim = Simulation(cfg, mapping, [ObjectSpec.from_dict(ball)])
        sim.set_object_shape("ball_hard", size_m=0.02)
        for finger in ("index", "middle", "ring", "pinky"):
            hit = finger in _palm_side_probe(sim, finger, sim.specs[0])
            # Legacy spheres sat on the dorsal housings: the palm-side rings were not covered.
            assert hit == expect_hit, (geometry, finger)


def test_stale_block_released_when_object_is_gone(setup):
    """A finger held by a contact is freed once the object has moved away."""
    config, mapping, specs = setup
    cfg = json.loads(json.dumps(config))
    ball = next(d for d in cfg["objects"]["defaults"] if d["name"] == "ball_hard")
    sim = Simulation(cfg, mapping, [ObjectSpec.from_dict(ball)])
    sim.set_object_shape("ball_hard", size_m=0.02)
    sim.update_glove(np.zeros(12))
    _palm_side_probe(sim, "ring", sim.specs[0])  # ball touching the ring finger
    ring = sim._finger_motors["ring"]
    for i in ring:
        sim._blocked[i] = (1, 0.0, 30.0)
    sim.step(feedback_enabled=False, dt_s=0.001)
    assert all(i in sim._blocked for i in ring)  # still touching: kept
    sim.set_object_pose("ball_hard", position=np.array([0.5, 0.5, 0.5]))
    sim.step(feedback_enabled=False, dt_s=0.001)
    assert not any(i in sim._blocked for i in ring)  # object gone: released
