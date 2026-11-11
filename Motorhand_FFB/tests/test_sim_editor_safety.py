"""Physical-output safety during interactive object editing."""
import json
import time

import mujoco
import numpy as np
import pytest

from sim.__main__ import _apply_object_edit
import sim.__main__ as app
from motorhand_teleop.domain import GLOVE_JOINTS
from sim.device import DemoGlove, GloveSample
from sim.scene import GloveScene


def test_physical_output_blocks_even_a_forged_editor_request():
    scene = GloveScene(object_shape="box", object_material="wood")
    position, quaternion = scene.object_pose()
    before_qpos = scene.data.qpos.copy()
    request = {
        "position_m": position + [0.04, 0.02, 0.01],
        "quat_wxyz": [0.9238795325, 0.0, 0.0, 0.3826834324],
        "dimensions_m": [0.08, 0.06, 0.04],
    }

    result, edited, rebuilt, done, blocked = _apply_object_edit(scene, request, editing_enabled=False)

    assert result is scene
    assert not edited and not rebuilt and done and not blocked
    actual_position, actual_quaternion = scene.object_pose()
    assert actual_position == pytest.approx(position)
    assert actual_quaternion == pytest.approx(quaternion)
    assert scene.object_dimensions_m == pytest.approx([0.05, 0.05, 0.05])
    assert np.array_equal(scene.data.qpos, before_qpos)


def test_overloaded_dry_run_uses_fixed_simulation_step_but_output_times_out():
    assert app._simulation_interval(0.14, 0.02, output_enabled=False) == 0.02
    with pytest.raises(TimeoutError):
        app._simulation_interval(0.14, 0.02, output_enabled=True)
    assert app._simulation_interval(0.06, 0.02, output_enabled=True) == 0.06


def test_authorized_edit_applies_dimensions_and_pose_to_the_physical_scene():
    scene = GloveScene(object_shape="box", object_material="wood",
                       object_position_m=[1.0, 1.0, 1.0])
    position, _ = scene.object_pose()
    request = {
        "position_m": position + [0.03, -0.02, 0.01],
        "quat_wxyz": [0.9238795325, 0.0, 0.0, 0.3826834324],
        "dimensions_m": [0.08, 0.06, 0.04],
    }

    for _ in range(200):
        before = scene.object_pose()[0]
        scene, edited, rebuilt, done, blocked = _apply_object_edit(scene, request, editing_enabled=True)
        assert edited and not blocked
        assert np.linalg.norm(scene.object_pose()[0] - before) <= 0.001 + 1e-9
        if done:
            break
    else:
        pytest.fail("bounded editor did not reach its target")
    result = scene
    assert result.object_dimensions_m == pytest.approx(request["dimensions_m"])
    actual_position, actual_quaternion = result.object_pose()
    assert actual_position == pytest.approx(request["position_m"])
    assert actual_quaternion == pytest.approx(request["quat_wxyz"], abs=1e-9)
    assert result.model.geom_size[result.object_geom_id] == pytest.approx([0.04, 0.03, 0.02])


@pytest.mark.parametrize("material", ["rubber", "wood"])
def test_editor_stops_before_pushing_object_deep_into_glove(material):
    scene = GloveScene(object_shape="sphere", object_material=material)
    geom = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "index_dip_link_visual")
    target = scene.data.geom_xpos[geom].copy()
    scene.set_object_pose(target + [0, 0, 0.08], [1, 0, 0, 0])
    request = {"position_m": target, "quat_wxyz": [1, 0, 0, 0],
               "dimensions_m": [0.05, 0.05, 0.05]}
    for _ in range(100):
        scene, _, _, done, blocked = _apply_object_edit(scene, request, editing_enabled=True)
        if done:
            break
    assert blocked
    assert scene.object_max_penetration() <= 0.001 + 1e-8
    assert np.linalg.norm(scene.object_pose()[0] - target) > 0.001


@pytest.mark.parametrize("release_frame", [0, 3], ids=["quick-release", "held-drag"])
def test_run_pauses_before_drag_and_resumes_only_after_target_is_applied(
        monkeypatch, tmp_path, release_frame):
    hardware, viewer, events, samples = _run_drag(
        monkeypatch, tmp_path, release_frame=release_frame, close_frame=25)

    assert viewer.editing_enabled is False
    assert hardware.homes == 1
    assert [event[0] for event in events].count("enable") == 2
    pause_index = next(i for i, event in enumerate(events) if event[0] == "pause")
    enable_index = next(i for i, event in enumerate(events)
                        if event[0] == "enable" and i > pause_index)
    moved = [(i, event) for i, event in enumerate(events) if event[0] == "move"]
    assert moved and all(pause_index < i < enable_index for i, _ in moved)
    assert events[pause_index][1] == pytest.approx(viewer.initial_position)
    assert events[enable_index][1] == pytest.approx(viewer.target_position)
    assert not any(event[0] == "send" for event in events[pause_index + 1:enable_index])
    assert any(event[0] == "send" for event in events[enable_index + 1:])
    assert samples
    paused = [row for row in samples if row["feedback_pause_reason"] is not None]
    assert paused and all(row["command_torque_nm"] is None for row in paused)
    if release_frame:
        assert any(row["feedback_pause_reason"] == "dragging" for row in paused)
    assert any(row["feedback_pause_reason"] == "waiting_clear" for row in paused)
    assert any(row["feedback_pause_reason"] is None and row["command_torque_nm"] is not None
               for row in samples)
    assert all(row["object_dimensions_m"] == pytest.approx([0.05, 0.05, 0.05])
               for row in samples)
    assert all(row["object_orientation_wxyz"] == pytest.approx(viewer.initial_quaternion)
               for row in samples)
    assert samples[-1]["object_position_m"] == pytest.approx(viewer.target_position)
    if release_frame:
        assert "dragging" in viewer.feedback_states
    assert "waiting_clear" in viewer.feedback_states
    assert "active" in viewer.feedback_states


def test_run_keeps_feedback_paused_until_simulated_contact_clears(monkeypatch, tmp_path):
    _, _, _, samples = _run_drag(
        monkeypatch, tmp_path, release_frame=0, close_frame=25, contact_until_frame=18)
    blocked = [row for row in samples if row["feedback_pause_reason"] == "waiting_clear"
               and row["contact_force_n"]["index"] > 0.01]
    assert blocked and all(row["command_torque_nm"] is None for row in blocked)
    resumes = [row for row in samples if row["feedback_transition"] == "resume_requested"]
    assert resumes and resumes[0]["contact_force_n"]["index"] == 0
    assert resumes[0]["time_ns"] > blocked[-1]["time_ns"]


def test_run_waits_for_real_virtual_alignment_before_resuming(monkeypatch, tmp_path):
    _, _, _, samples = _run_drag(
        monkeypatch, tmp_path, release_frame=0, close_frame=80, input_jump=True)
    waiting = [row for row in samples if row["feedback_pause_reason"] == "waiting_clear"
               and row["input_alignment_error_rad"] > app._RESUME_ALIGNMENT_RAD]
    assert waiting and all(row["command_torque_nm"] is None for row in waiting)
    resumes = [row for row in samples if row["feedback_transition"] == "resume_requested"]
    assert resumes and resumes[0]["input_alignment_error_rad"] <= app._RESUME_ALIGNMENT_RAD
    assert resumes[0]["time_ns"] > waiting[-1]["time_ns"]
    assert resumes[0]["virtual_servo_mode"] == "edit_clutch"


def test_failed_pause_never_moves_object_and_closes_device(monkeypatch, tmp_path):
    hardware, _, events, _ = _run_drag(
        monkeypatch, tmp_path, release_frame=None, close_frame=3, pause_fails=True)
    assert any(event[0] == "pause" for event in events)
    assert not any(event[0] == "move" for event in events)
    assert events[-1][0] == "close"
    assert not hardware.enabled


def test_queued_release_then_repress_remains_paused(monkeypatch, tmp_path):
    _, _, events, samples = _run_drag(
        monkeypatch, tmp_path, release_frame=3, close_frame=25, queued_repress=True)
    pause = next(i for i, event in enumerate(events) if event[0] == "pause")
    resume = next(i for i, event in enumerate(events) if event[0] == "enable" and i > pause)
    assert not any(event[0] == "send" for event in events[pause + 1:resume])
    assert [row["feedback_pause_reason"] for row in samples[:3]] == ["dragging"] * 3
    assert all(row["command_torque_nm"] is None for row in samples[:3])


def test_run_closing_window_during_drag_disables_output(monkeypatch, tmp_path):
    hardware, _, events, samples = _run_drag(
        monkeypatch, tmp_path, release_frame=None, close_frame=3)

    pause_index = next(i for i, event in enumerate(events) if event[0] == "pause")
    assert any(event[0] == "move" for event in events[pause_index + 1:])
    assert not any(event[0] in ("send", "enable") for event in events[pause_index + 1:])
    assert events[-1][0] == "close"
    assert not hardware.enabled
    assert samples and samples[-1]["feedback_pause_reason"] == "dragging"
    assert samples[-1]["command_torque_nm"] is None


def _run_drag(monkeypatch, tmp_path, *, release_frame, close_frame, contact_until_frame=0,
              pause_fails=False, queued_repress=False, input_jump=False):
    """Exercise the real CLI loop and MuJoCo scene with an instrumented device/UI."""
    import sim.viewer

    events = []

    class SimulatedHardware(DemoGlove):
        instance = None

        def __init__(self, side, *_args):
            super().__init__(side, "press")
            type(self).instance = self
            self.torque_limits_nm = np.full(12, 0.1)
            self.enabled = False
            self.homes = 0

        def connect(self):
            pass

        def enable_output(self):
            if not self.homes:
                self.homes += 1
            self.enabled = True
            events.append(("enable", current_position()))

        def pause_output(self):
            events.append(("pause", current_position()))
            if pause_fails:
                raise RuntimeError("offline pause failure")
            self.enabled = False

        def read(self):
            if input_jump and DragViewer.instance is not None and DragViewer.instance.frame > 0:
                values = {name: 0.0 for name in GLOVE_JOINTS}
                values["index_mcp_flex"] = values["index_pip"] = 0.3
                pose = self.model.evaluate_raw(values)
                return GloveSample(time.monotonic_ns(), pose.motor_positions, pose.all_positions,
                                   {name: pose.joint_positions[f"{name}_joint"] for name in GLOVE_JOINTS},
                                   [None] * 12)
            self.reset_clock()
            return super().read()

        def send(self, torque, _received_ns):
            assert self.enabled, "force command sent while output is paused"
            events.append(("send", current_position()))
            return tuple(torque)

        def close(self):
            self.enabled = False
            events.append(("close", current_position()))

    class DragViewer:
        instance = None

        def __init__(self, scene, *, editing_enabled):
            type(self).instance = self
            self.scene = scene
            self.editing_enabled = editing_enabled
            self.initial_position, self.initial_quaternion = scene.object_pose()
            self.initial_position = self.initial_position.copy()
            self.initial_quaternion = self.initial_quaternion.copy()
            self.target_position = self.initial_position + [0.012, 0, 0]
            self.frame = 0
            self.requests = []
            self.feedback_states = []

        def consume_edit(self):
            if not self.requests:
                return None
            return {
                "position_m": self.target_position.copy(),
                "quat_wxyz": [0.7071067811865476, 0, 0, 0.7071067811865476],
                "dimensions_m": [0.08, 0.06, 0.04],
                "dragging": self.requests.pop(0),
            }

        def consume_reset(self):
            return False

        def is_running(self):
            self.frame += 1
            if self.frame == 1:
                self.requests.append(True)
                if queued_repress:
                    self.requests.extend([False, True])
                if release_frame == 0:
                    self.requests.append(False)
            elif release_frame is not None and self.frame == release_frame + 1:
                self.requests.append(False)
            return self.frame <= close_frame

        def set_feedback_state(self, state):
            self.feedback_states.append(state)

        def submit(self, scene):
            self.scene = scene

        def notify_blocked(self):
            pytest.fail("free-space object drag unexpectedly blocked")

        def close(self):
            pass

    def current_position():
        viewer = DragViewer.instance
        return None if viewer is None else viewer.scene.object_pose()[0].copy()

    set_object_pose = GloveScene.set_object_pose

    def record_object_pose(scene, position, quaternion):
        hardware = SimulatedHardware.instance
        if DragViewer.instance is not None:
            assert not hardware.enabled, "object moved before output was paused"
            events.append(("move", np.asarray(position).copy()))
        return set_object_pose(scene, position, quaternion)

    monkeypatch.setattr(GloveScene, "set_object_pose", record_object_pose)
    if contact_until_frame:
        original_step = GloveScene.step

        def step_with_contact(scene, tree, dt, **kwargs):
            frame = original_step(scene, tree, dt, **kwargs)
            viewer = DragViewer.instance
            if viewer is not None and 1 <= viewer.frame <= contact_until_frame:
                geom = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM,
                                        "index_dip_link_visual")
                frame["contacts"] = [{
                    "finger": "index", "body_id": int(scene.model.geom_bodyid[geom]),
                    "point_world": scene.data.geom_xpos[geom].copy(),
                    "force_world_on_glove": np.array([0.02, 0, 0]),
                    "normal_force_n": 0.02, "tangential_force_n": 0.0,
                    "relative_tangent_speed_m_s": 0.0, "penetration_m": 0.0,
                }]
            return frame

        monkeypatch.setattr(GloveScene, "step", step_with_contact)
    monkeypatch.setattr(app, "HardwareGlove", SimulatedHardware)
    monkeypatch.setattr(sim.viewer, "LiveViewer", DragViewer)
    log = tmp_path / "drag.jsonl"
    args = app._parser().parse_args([
        "run", "--input", "glove", "--execute", "--acknowledge-homing",
        "--raw-torque-limit-nm", "0.1", "--ports", "fake1", "fake2", "fake3",
        "--object", "box", "--material", "wood", "--object-motion", "fixed",
        "--object-position", "1", "1", "1", "--rate", "20", "--no-live-plot",
        "--log", str(log),
    ])
    if pause_fails:
        with pytest.raises(RuntimeError, match="offline pause failure"):
            app.run(args)
    else:
        app.run(args)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    samples = [row for row in rows if row["type"] == "sample"]
    return SimulatedHardware.instance, DragViewer.instance, events, samples
