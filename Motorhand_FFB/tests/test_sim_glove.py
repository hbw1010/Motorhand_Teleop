"""Behavioral checks for the glove-only MuJoCo contact and safety boundaries."""
import json
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from motorhand_teleop.domain import GLOVE_JOINTS
from motorhand_teleop.ve33_adapter import Ve33GloveModel
from sim.device import HardwareGlove, MAX_FEEDBACK_AGE_NS
from sim.feedback import FeedbackLimiter, ForceProjector
from sim.scene import GloveScene
from sim.telemetry import Recorder


@pytest.fixture(scope="module")
def glove():
    return Ve33GloveModel("right")


def pose_at(glove, angle):
    joints = {name: 0.0 for name in GLOVE_JOINTS}
    for finger in ("index", "middle", "ring", "pinky"):
        joints[f"{finger}_mcp_flex"] = angle
        joints[f"{finger}_pip"] = angle
    return glove.evaluate_raw(joints)


def drive_to_press(scene, glove, angle=0.35):
    """Approach the object gradually, then let the servo settle against it."""
    for value in np.linspace(0.0, angle, 21)[1:]:
        frame = scene.step(pose_at(glove, value).all_positions, 0.05)
    target = pose_at(glove, angle)
    for _ in range(20):
        frame = scene.step(target.all_positions, 0.05)
    return target, frame


def max_penetration(frame):
    return max((c["penetration_m"] for c in frame["contacts"]), default=0.0)


def best_fit_rotation_degrees(initial, current):
    reference = initial - initial.mean(axis=0)
    displaced = current - current.mean(axis=0)
    u, _, vt = np.linalg.svd(reference.T @ displaced)
    rotation = vt.T @ np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)]) @ u.T
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def assert_virtual_pose_matches_data(scene, glove, frame):
    tree = frame["virtual_tree_positions_rad"]
    motors = frame["virtual_motor_positions_rad"]
    assert set(tree) == set(glove.kinematics.tree_names)
    assert set(motors) == set(glove.motor_names)
    for name, value in tree.items():
        address = int(scene.model.joint(name).qposadr[0])
        assert value == pytest.approx(scene.data.qpos[address], abs=1e-10)
    for name, value in motors.items():
        assert value == pytest.approx(tree[name], abs=1e-10)


def test_press_scene_starts_clear_then_constrains_and_releases_glove(glove):
    zero = pose_at(glove, 0)
    scene = GloveScene("right", "press", "hard", object_material="wood",
                       initial_tree_positions_rad=zero.all_positions)
    first = scene.step(zero.all_positions, 0.05)
    assert not first["contacts"]
    initial_position = first["object_position_m"].copy()
    pressed, frame = drive_to_press(scene, glove)
    assert_virtual_pose_matches_data(scene, glove, frame)
    assert any(c["normal_force_n"] > 0.01 for c in frame["contacts"])
    assert all(c["finger"] in ("thumb", "index", "middle", "ring", "pinky") for c in frame["contacts"])
    assert frame["object_position_m"] == pytest.approx(initial_position)
    assert frame["object_velocity_m_s"] == pytest.approx(np.zeros(3))
    assert 0 < max_penetration(frame) < 0.005
    motor_error = max(abs(frame["virtual_motor_positions_rad"][name] - value)
                      for name, value in pressed.motor_positions.items())
    tree_error = max(abs(frame["virtual_tree_positions_rad"][name] - value)
                     for name, value in pressed.all_positions.items())
    assert motor_error > 0.01
    assert tree_error > 0.01
    for contact in frame["contacts"]:
        if contact["normal_force_n"] > 1e-6:
            outward = contact["point_world"] - initial_position
            assert np.dot(contact["force_world_on_glove"], outward) > 0

    for value in np.linspace(0.35, 0.0, 21)[1:]:
        released = scene.step(pose_at(glove, value).all_positions, 0.05)
    for _ in range(20):
        released = scene.step(zero.all_positions, 0.05)
    assert_virtual_pose_matches_data(scene, glove, released)
    assert sum(c["normal_force_n"] for c in released["contacts"]) < 0.01
    assert max(abs(released["virtual_motor_positions_rad"][name] - value)
               for name, value in zero.motor_positions.items()) < motor_error
    assert max(abs(released["virtual_tree_positions_rad"][name] - value)
               for name, value in zero.all_positions.items()) < tree_error
    assert released["object_position_m"] == pytest.approx(initial_position)


@pytest.mark.parametrize("shape,kind", [
    ("sphere", mujoco.mjtGeom.mjGEOM_SPHERE),
    ("box", mujoco.mjtGeom.mjGEOM_BOX),
    ("cylinder", mujoco.mjtGeom.mjGEOM_CYLINDER),
])
def test_larger_objects_are_fixed_and_contactable_by_default(glove, shape, kind):
    zero = pose_at(glove, 0)
    scene = GloveScene("right", "press", object_shape=shape, object_material="wood",
                       initial_tree_positions_rad=zero.all_positions)
    assert scene.metadata["object_shape"] == shape
    assert scene.metadata["object_size_m"] == pytest.approx(0.05)
    assert scene.model.geom_type[scene.object_geom_id] == kind
    first = scene.step(zero.all_positions, 0.02)
    assert not first["contacts"]
    position = first["object_position_m"].copy()
    _, second = drive_to_press(scene, glove)
    assert any(c["normal_force_n"] > 0 for c in second["contacts"])
    assert second["object_position_m"] == pytest.approx(position)
    assert second["object_velocity_m_s"] == pytest.approx(np.zeros(3))


def test_grasp_object_is_fixed_unless_free_is_requested(glove):
    zero = pose_at(glove, 0)
    scene = GloveScene("right", "grasp", object_shape="box", object_size_m=0.06,
                       initial_tree_positions_rad=zero.all_positions)
    assert scene.object_fixed
    assert scene.model.geom_size[scene.object_geom_id][:3] == pytest.approx([0.03] * 3)
    initial = scene.step(zero.all_positions, 0.02)["object_position_m"].copy()
    _, frame = drive_to_press(scene, glove)
    assert frame["object_position_m"] == pytest.approx(initial)
    assert frame["object_velocity_m_s"] == pytest.approx(np.zeros(3))


def test_material_changes_contact_response_and_visual_appearance(glove):
    zero = pose_at(glove, 0)
    results = {}
    for material in ("foam", "wood"):
        scene = GloveScene("right", "press", object_shape="box", object_material=material,
                           initial_tree_positions_rad=zero.all_positions)
        _, frame = drive_to_press(scene, glove)
        geom = scene.object_geom_id
        material_id = scene.model.geom_matid[geom]
        results[material] = (max_penetration(frame),
                             scene.model.geom_friction[geom, 0],
                             scene.model.mat_rgba[material_id].copy(),
                             sum(c["normal_force_n"] for c in frame["contacts"]))
    assert results["foam"][0] > results["wood"][0] > 0
    assert results["wood"][3] > 0 and results["foam"][3] > 0
    assert results["wood"][1] != results["foam"][1]
    assert not np.allclose(results["wood"][2], results["foam"][2])


def test_stiffness_override_preserves_material_friction(glove):
    zero = pose_at(glove, 0)
    results = []
    for stiffness in (None, "hard"):
        scene = GloveScene("right", "press", stiffness, object_shape="box",
                           object_material="foam", initial_tree_positions_rad=zero.all_positions)
        _, frame = drive_to_press(scene, glove)
        results.append((max_penetration(frame),
                        scene.model.geom_friction[scene.object_geom_id, 0],
                        sum(c["normal_force_n"] for c in frame["contacts"])))
    assert results[0][0] > results[1][0] > 0
    assert results[0][2] > 0 and results[1][2] > 0
    assert results[0][1] == pytest.approx(results[1][1])


def test_contact_materials_change_penetration_at_same_target(glove):
    zero = pose_at(glove, 0)
    results = {}
    for stiffness in ("soft", "hard"):
        scene = GloveScene("right", "press", stiffness, object_material="wood",
                           initial_tree_positions_rad=zero.all_positions)
        _, frame = drive_to_press(scene, glove)
        assert sum(c["normal_force_n"] for c in frame["contacts"]) > 0
        results[stiffness] = max_penetration(frame)
    assert results["soft"] > results["hard"] > 0


@pytest.mark.parametrize("size", [0.03, 0.07])
def test_deformable_ball_keeps_requested_size(glove, size):
    zero = pose_at(glove, 0)
    scene = GloveScene("right", "press", object_size_m=size,
                       initial_tree_positions_rad=zero.all_positions)
    vertices = scene.data.flexvert_xpos
    diameter = np.ptp(vertices, axis=0) + 2 * scene.model.flex_radius[0]
    assert diameter == pytest.approx([size] * 3, abs=2e-7)


def test_fixed_deformable_ball_resists_torque_but_still_deforms(glove):
    zero = pose_at(glove, 0)
    scene = GloveScene("right", "grasp", initial_tree_positions_rad=zero.all_positions)
    initial = scene.data.flexvert_xpos.copy()
    center = scene.model.body("scene_soft_1_1_1").id
    center_position = scene.data.xpos[center].copy()
    loaded_node = scene.model.body("scene_soft_2_0_2").id
    peak_rotation = 0.0
    peak_deformation = 0.0
    for index in range(80):
        scene.data.xfrc_applied[loaded_node, :3] = (0, 0.08 if index < 30 else 0, 0)
        frame = scene.step(zero.all_positions, 0.02)
        peak_rotation = max(peak_rotation, best_fit_rotation_degrees(initial, scene.data.flexvert_xpos))
        peak_deformation = max(peak_deformation, frame["object_deformation_m"])
    assert scene.data.xpos[center] == pytest.approx(center_position, abs=1e-10)
    assert peak_rotation < 0.1
    assert peak_deformation > 0.0005
    assert frame["object_deformation_m"] < peak_deformation * 0.25


def test_free_deformable_ball_can_rotate_under_torque(glove):
    zero = pose_at(glove, 0)
    scene = GloveScene("right", "grasp", initial_tree_positions_rad=zero.all_positions,
                       object_fixed=False, object_position_m=(0.4, 0.4, 0.4))
    scene.model.opt.gravity[:] = 0
    floor = scene.model.geom("scene_floor").id
    scene.model.geom_pos[floor, 2] = -1
    initial = scene.data.flexvert_xpos.copy()
    plus = scene.model.body("scene_soft_2_1_1").id
    minus = scene.model.body("scene_soft_0_1_1").id
    scene.data.xfrc_applied[plus, :3] = (0, 0.001, 0)
    scene.data.xfrc_applied[minus, :3] = (0, -0.001, 0)
    for _ in range(50):
        scene.step(zero.all_positions, 0.01)
    assert best_fit_rotation_degrees(initial, scene.data.flexvert_xpos) > 2.0


@pytest.mark.parametrize("material", ["rubber", "foam"])
def test_deformable_ball_is_compressed_by_contact_and_recovers(glove, material):
    zero = pose_at(glove, 0)
    scene = GloveScene("right", "press", object_shape="sphere", object_material=material,
                       initial_tree_positions_rad=zero.all_positions)
    assert scene.model.nflex > 0
    initial = scene.data.flexvert_xpos.copy()
    assert len(initial) > 4
    initial_center = initial.mean(axis=0)
    initial_radius = np.linalg.norm(initial - initial_center, axis=1)

    target, frame = drive_to_press(scene, glove)
    compressed = scene.data.flexvert_xpos.copy()
    compressed_radius = np.linalg.norm(compressed - compressed.mean(axis=0), axis=1)
    actual_deformation = np.max(np.abs(compressed_radius - initial_radius))
    assert actual_deformation > 1e-4
    assert best_fit_rotation_degrees(initial, compressed) < 0.1
    assert frame["object_deformation_m"] == pytest.approx(actual_deformation, abs=1e-8)
    assert any(c["normal_force_n"] > 0.01 for c in frame["contacts"])
    assert max_penetration(frame) < 0.005
    assert np.linalg.norm(compressed.mean(axis=0) - initial_center) < 0.025

    projector = ForceProjector(glove, scene.model, max_compute_s=3)
    torque = projector.project(scene.data, frame["contacts"], frame["virtual_motor_positions_rad"],
                               frame["virtual_tree_positions_rad"])
    assert np.max(np.abs(torque)) > 1e-6
    error = np.array([target.motor_positions[name] - frame["virtual_motor_positions_rad"][name]
                      for name in glove.motor_names])
    assert np.dot(torque, error) < 0

    for value in np.linspace(0.35, 0.0, 21)[1:]:
        frame = scene.step(pose_at(glove, value).all_positions, 0.05)
    for _ in range(30):
        frame = scene.step(zero.all_positions, 0.05)
    recovered = scene.data.flexvert_xpos.copy()
    recovered_radius = np.linalg.norm(recovered - recovered.mean(axis=0), axis=1)
    residual = np.max(np.abs(recovered_radius - initial_radius))
    assert residual < actual_deformation * 0.5
    assert sum(c["normal_force_n"] for c in frame["contacts"]) < 0.01

def test_edit_clutch_releases_a_sustained_deformable_ball_press(glove):
    zero = pose_at(glove, 0)
    scene = GloveScene("right", "press", object_shape="sphere", object_material="rubber",
                       initial_tree_positions_rad=zero.all_positions)
    _, pressed = drive_to_press(scene, glove)
    initial_force = sum(c["normal_force_n"] for c in pressed["contacts"])
    assert initial_force > 0.01
    assert pressed["object_deformation_m"] > 1e-4

    # Keep increasing the physical input after contact, then hold it. The
    # paused virtual servo must not accumulate an unreachable absolute target.
    for value in np.linspace(0.35, 0.55, 21)[1:]:
        frame = scene.step(pose_at(glove, value).all_positions, 0.05, edit_clutch=True)
    closed = pose_at(glove, 0.55)
    for _ in range(100):
        frame = scene.step(closed.all_positions, 0.05, edit_clutch=True)
    held_force = sum(c["normal_force_n"] for c in frame["contacts"])
    assert held_force < initial_force
    assert frame["object_deformation_m"] < pressed["object_deformation_m"]
    assert frame["input_alignment_error_rad"] > 0.05
    assert np.max(np.abs(scene.data.qfrc_applied[scene._motor_dofs])) <= 0.05

    for value in np.linspace(0.55, 0.0, 41)[1:]:
        frame = scene.step(pose_at(glove, value).all_positions, 0.05, edit_clutch=True)
    for _ in range(60):
        frame = scene.step(zero.all_positions, 0.05, edit_clutch=True)
    assert sum(c["normal_force_n"] for c in frame["contacts"]) < 0.01
    assert frame["input_alignment_error_rad"] < 0.05
    assert frame["object_deformation_m"] < pressed["object_deformation_m"] * 0.5
    normal = scene.step(zero.all_positions, 0.05)
    assert normal["virtual_servo_mode"] == "tracking"
    assert sum(c["normal_force_n"] for c in normal["contacts"]) < 0.01



def test_inverted_deformable_element_is_rejected_before_feedback(glove):
    zero = pose_at(glove, 0)
    scene = GloveScene("right", "press", initial_tree_positions_rad=zero.all_positions)
    # Move one interpolated control node far enough to invert a tetrahedron.
    scene.data.qpos[scene._flex_qpos[0]] += 0.3
    mujoco.mj_forward(scene.model, scene.data)
    with pytest.raises(RuntimeError, match="inverted or nearly collapsed"):
        scene.step(zero.all_positions, 0.02)


def test_projection_matches_closed_chain_contact_virtual_work(glove):
    scene = GloveScene("right", "press", initial_tree_positions_rad=pose_at(glove, 0).all_positions)
    _, frame = drive_to_press(scene, glove)
    assert_virtual_pose_matches_data(scene, glove, frame)
    pose = glove.kinematics.evaluate_tree(frame["virtual_tree_positions_rad"])
    contact = next(c for c in frame["contacts"] if c["normal_force_n"] > 0.01)
    projector = ForceProjector(glove, scene.model, max_compute_s=3)
    torque = projector.project(scene.data, [contact], frame["virtual_motor_positions_rad"],
                               frame["virtual_tree_positions_rad"])
    axis = int(np.argmax(np.abs(torque)))
    assert abs(torque[axis]) > 1e-6
    body = contact["body_id"]
    point = contact["point_world"]
    local = scene.data.xmat[body].reshape(3, 3).T @ (point - scene.data.xpos[body])
    motor = glove.motor_names[axis]
    point_positions = []
    for delta in (-1e-4, 1e-4):
        motors = dict(pose.motor_positions)
        motors[motor] += delta
        shifted = glove.kinematics.evaluate_motors(motors, seed=pose)
        scratch = mujoco.MjData(scene.model)
        scratch.qpos[:] = scene.data.qpos
        for name, value in shifted.all_positions.items():
            scratch.qpos[int(scene.model.joint(name).qposadr[0])] = value
        mujoco.mj_forward(scene.model, scratch)
        point_positions.append(scratch.xpos[body] + scratch.xmat[body].reshape(3, 3) @ local)
    velocity = (point_positions[1] - point_positions[0]) / (2e-4)
    expected = float(np.dot(contact["force_world_on_glove"], velocity))
    assert torque[axis] == pytest.approx(expected, rel=0.02, abs=1e-5)


def test_free_object_moves_under_contact(glove):
    zero = pose_at(glove, 0)
    scene = GloveScene("right", "grasp", object_position_m=(0.14, 0.012, -0.06),
                       initial_tree_positions_rad=zero.all_positions, object_material="wood",
                       object_fixed=False)
    before = np.array(scene.metadata["object_initial_position_m"])
    contacted = False
    for value in np.linspace(0.0, 0.35, 21)[1:]:
        frame = scene.step(pose_at(glove, value).all_positions, 0.05)
        contacted |= any(c["normal_force_n"] > 0 for c in frame["contacts"])
    assert contacted
    assert np.linalg.norm(frame["object_position_m"] - before) > 1e-6
    scene.reset_object()
    assert scene.data.xpos[scene.object_body_id] == pytest.approx(before)
    dof = int(scene.model.jnt_dofadr[scene._object_joint_id])
    assert scene.data.qvel[dof:dof + 6] == pytest.approx(np.zeros(6))


def test_free_deformable_ball_moves_and_resets(glove):
    zero = pose_at(glove, 0)
    scene = GloveScene("right", "grasp", object_position_m=(0.14, 0.012, -0.06),
                       initial_tree_positions_rad=zero.all_positions, object_material="rubber",
                       object_fixed=False)
    initial_vertices = scene.data.flexvert_xpos.copy()
    initial_center = initial_vertices.mean(axis=0)
    for value in np.linspace(0.0, 0.35, 21)[1:]:
        frame = scene.step(pose_at(glove, value).all_positions, 0.05)
    assert np.linalg.norm(frame["object_position_m"] - initial_center) > 1e-5
    scratch = mujoco.MjData(scene.model)
    scratch.qpos[:] = scene.data.qpos
    mujoco.mj_integratePos(scene.model, scratch.qpos, scene.data.qvel, 1e-5)
    mujoco.mj_kinematics(scene.model, scratch)
    mujoco.mj_flex(scene.model, scratch)
    numerical_velocity = (scratch.flexvert_xpos.mean(axis=0) - frame["object_position_m"]) / 1e-5
    assert frame["object_velocity_m_s"] == pytest.approx(numerical_velocity, abs=1e-6)
    scene.reset_object()
    assert scene.data.flexvert_xpos == pytest.approx(initial_vertices, abs=1e-8)


def test_default_grasp_path_reaches_sustained_thumb_index_contact(glove):
    zero = pose_at(glove, 0)
    scene = GloveScene("right", "grasp", initial_tree_positions_rad=zero.all_positions)
    longest = current = 0
    for t in np.linspace(0.0, 6.0, 121):
        flex = 0.3 * (1.0 - np.cos(2.0 * np.pi * t / 6.0))
        joints = {name: 0.0 for name in GLOVE_JOINTS}
        for finger in ("index", "middle", "ring", "pinky"):
            joints[f"{finger}_mcp_flex"] = joints[f"{finger}_pip"] = flex
        joints["thumb_cmc_flex"] = joints["thumb_mcp"] = joints["thumb_ip"] = flex
        frame = scene.step(glove.evaluate_raw(joints).all_positions, 0.05)
        active = {c["finger"] for c in frame["contacts"] if c["normal_force_n"] > 0.01}
        current = current + 1 if {"thumb", "index"} <= active else 0
        longest = max(longest, current)
    assert longest >= 4  # At least 0.2 s of thumb/index contact.


def test_invalid_simulator_state_is_rejected_before_contact_feedback(glove):
    zero = pose_at(glove, 0)
    scene = GloveScene("right", "press", initial_tree_positions_rad=zero.all_positions)
    scene.data.qacc[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        scene.step(zero.all_positions, 0.02)


def test_simulator_warning_is_not_ignored(glove):
    zero = pose_at(glove, 0)
    scene = GloveScene("right", "press", initial_tree_positions_rad=zero.all_positions)
    scene.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = 1
    with pytest.raises(RuntimeError, match="mjWARN_BADQACC"):
        scene.step(zero.all_positions, 0.02)


def test_force_limiter_clamps_and_fails_safe_on_stale_data():
    limiter = FeedbackLimiter(np.full(12, 0.2), np.full(12, 1.0))
    assert np.allclose(limiter.apply(np.full(12, 5.0), 0.02), 0.02)
    assert np.allclose(limiter.apply(np.full(12, 0.1), 0.2), 0.0)
    assert limiter.last_error is not None
    assert np.allclose(limiter.apply(np.full(12, np.nan), 0.02), 0.0)


def test_hardware_output_requires_explicit_homing_acknowledgement():
    with pytest.raises(ValueError, match="acknowledge-homing"):
        HardwareGlove("right", None, True, False, 0.1)
    with pytest.raises(ValueError, match="--ports"):
        HardwareGlove("right", None, True, True, 0.1)
    with pytest.raises(ValueError, match="左手"):
        HardwareGlove("left", None, False, False, None)


def test_stale_source_never_sends_hardware_command():
    device = object.__new__(HardwareGlove)
    device.execute = True
    device._started = True
    sent = []
    device.driver = SimpleNamespace(send_force_torque=lambda *a, **kw: sent.append((a, kw)))
    with pytest.raises(TimeoutError):
        device.send(np.ones(12), time.monotonic_ns() - MAX_FEEDBACK_AGE_NS - 1)
    assert not sent


def test_recording_preserves_unavailable_measured_values_and_source(tmp_path):
    path = tmp_path / "session.jsonl"
    recorder = Recorder(path, {"sources": {"motor_current_a": "unavailable"}})
    recorder.append({"sim_time_s": 0.1, "motor_current_a": [None] * 12,
                     "contact_force_n": {"index": 1.2}, "command_torque_nm": None})
    recorder.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["metadata"]["sources"]["contact_force_n"] == "simulated"
    assert records[0]["metadata"]["sources"]["motor_current_a"] == "unavailable"
    assert records[1]["motor_current_a"] == [None] * 12
    assert records[1]["command_torque_nm"] is None
    assert records[2]["dropped_samples"] == 0


def test_sim_contact_reaches_nominal_sdk_output_and_disables_on_close(glove):
    """Exercise the actual driver against its offline SDK test device, not hardware."""
    from motorhand_teleop.real_hand.motor_hand import EncosGloveHardwareDriver
    from test_encos_hardware import FakeEncosHand, _mapping

    backend = FakeEncosHand()
    device = HardwareGlove("right", None, True, True, 0.2,
                           ("/dev/test-can1", "/dev/test-can2", "/dev/test-can3"))
    device.driver = EncosGloveHardwareDriver(
        hand_factory=lambda: backend, mapping=_mapping(),
        force_torque_limits_nm={mid: 0.2 for mid in range(1, 13)},
    )
    try:
        device.connect()
        device.enable_output()
        zero, pressed = pose_at(glove, 0), pose_at(glove, 0.35)
        for mid, config in device.driver.motors.items():
            name = glove.joint_motor_names[config.joint]
            backend.angles[mid] = float(np.rad2deg(pressed.motor_positions[name]))
        scene = GloveScene("right", "press", initial_tree_positions_rad=zero.all_positions)
        projector = ForceProjector(glove, scene.model, max_compute_s=1)
        sample = device.read()
        for _ in range(40):
            frame = scene.step(sample.tree_positions_rad, 0.05)
        assert_virtual_pose_matches_data(scene, glove, frame)
        requested = projector.project(scene.data, frame["contacts"],
                                      frame["virtual_motor_positions_rad"],
                                      frame["virtual_tree_positions_rad"])
        limiter = FeedbackLimiter(device.torque_limits_nm, np.full(12, 1.0))
        command = limiter.apply(requested * 0.1, 0.02)
        assert np.max(np.abs(command)) > 1e-6
        backend.commands.clear()
        sample = device.read()
        device.send(command, sample.received_ns)
        sent = [data for kind, data in backend.commands if kind == "mit"]
        assert len(sent) == 12
        assert any(abs(data["torque_nm"]) > 1e-6 for data in sent)
        assert all(abs(data["torque_nm"]) <= 0.2 and data["kp"] == data["kd"] == 0 for data in sent)
    finally:
        device.close()
    assert backend.closed
    assert all((mid, False) in backend.enabled for mid in range(1, 13))


@pytest.mark.parametrize("shape,material,kind", [
    ("sphere", "wood", mujoco.mjtGeom.mjGEOM_ELLIPSOID),
    ("box", "wood", mujoco.mjtGeom.mjGEOM_BOX),
    ("cylinder", "wood", mujoco.mjtGeom.mjGEOM_MESH),
])
@pytest.mark.parametrize("fixed", [True, False])
def test_resize_rigid_geometry_and_world_pose(glove, shape, material, kind, fixed):
    scene = GloveScene(object_shape=shape, object_material=material, object_fixed=fixed,
                       glove_model=glove)
    dims = np.array([0.06, 0.04, 0.08])
    position = np.array([0.13, -0.08, 0.17])
    quat = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    scene.set_object_pose(position, quat * 2)
    floor = None
    if not fixed:
        floor_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "scene_floor")
        floor = scene.model.geom_pos[floor_id].copy()
    scene.data.qpos[scene._tree_qpos] += 0.001
    scene.data.qvel[scene._tree_dofs] = 0.02
    scene.data.time = 1.25
    changed = scene.resized(dims)
    assert changed.model.geom_type[changed.object_geom_id] == kind
    assert changed.object_dimensions_m == pytest.approx(dims)
    assert changed.data.qpos[changed._tree_qpos] == pytest.approx(scene.data.qpos[scene._tree_qpos])
    assert changed.data.qvel[changed._tree_dofs] == pytest.approx(scene.data.qvel[scene._tree_dofs])
    assert changed.data.time == 1.25
    actual_position, actual_quat = changed.object_pose()
    assert actual_position == pytest.approx(position)
    assert abs(np.dot(actual_quat, quat)) == pytest.approx(1)
    if shape != "cylinder":
        assert changed.model.geom_size[changed.object_geom_id, :3] == pytest.approx(dims / 2)
    else:
        mesh_id = changed.model.geom_dataid[changed.object_geom_id]
        start = changed.model.mesh_vertadr[mesh_id]
        count = changed.model.mesh_vertnum[mesh_id]
        vertices = changed.model.mesh_vert[start:start + count]
        assert np.sort(np.ptp(vertices, axis=0)) == pytest.approx(np.sort(dims), abs=1e-6)
    if not fixed:
        floor_id = mujoco.mj_name2id(changed.model, mujoco.mjtObj.mjOBJ_GEOM, "scene_floor")
        assert changed.model.geom_pos[floor_id] == pytest.approx(floor)
        dof = changed.model.jnt_dofadr[changed._object_joint_id]
        assert changed.data.qvel[dof:dof + 6] == pytest.approx(np.zeros(6))


@pytest.mark.parametrize("fixed", [True, False])
def test_flex_resize_and_pose_rotate_real_vertices_and_velocity_jacobian(glove, fixed):
    scene = GloveScene(object_fixed=fixed, glove_model=glove)
    scene = scene.resized([0.06, 0.04, 0.08])
    vertices = scene.data.flexvert_xpos[scene._flex_vertices].copy()
    diameter = np.ptp(vertices, axis=0) + 2 * scene.model.flex_radius[scene._flex_id]
    assert diameter == pytest.approx([0.06, 0.04, 0.08], abs=2e-7)
    initial_position, _ = scene.object_pose()
    target = np.array([0.1, -0.2, 0.3])
    quat = np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    old_jacobian = scene._centroid_jacobian.copy()
    scene.set_object_pose(target, quat)
    actual, actual_quat = scene.object_pose()
    assert actual == pytest.approx(target, abs=1e-9)
    assert abs(np.dot(actual_quat, quat)) == pytest.approx(1, abs=1e-8)
    assert scene.data.flexvert_xpos[scene._flex_vertices] == pytest.approx(
        (vertices - initial_position) @ rotation.T + target, abs=1e-8)
    assert scene._centroid_jacobian == pytest.approx(rotation @ old_jacobian, abs=1e-8)
    scene._check_dynamics()
    resized = scene.resized([0.05, 0.07, 0.09])
    assert resized.object_pose()[0] == pytest.approx(target, abs=1e-9)
    assert abs(np.dot(resized.object_pose()[1], quat)) == pytest.approx(1, abs=1e-8)
    assert resized.data.qvel[resized._flex_dofs] == pytest.approx(np.zeros(len(resized._flex_dofs)))


@pytest.mark.parametrize("bad", [[0, 1, 1], [1, -1, 1], [1, np.inf, 1], [1, 2]])
def test_resize_rejects_invalid_dimensions(glove, bad):
    scene = GloveScene(glove_model=glove)
    with pytest.raises(ValueError):
        scene.resized(bad)


def test_pose_edit_moves_contact_geometry(glove):
    scene = GloveScene(object_shape="box", object_material="wood", glove_model=glove)
    geom = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "index_dip_link_visual")
    scene.set_object_pose(scene.data.geom_xpos[geom].copy(), [1, 0, 0, 0])
    assert any(scene.object_geom_id in contact.geom for contact in scene.data.contact[:scene.data.ncon])
    scene.set_object_pose([10, 10, 10], [1, 0, 0, 0])
    assert not any(scene.object_geom_id in contact.geom for contact in scene.data.contact[:scene.data.ncon])


@pytest.mark.parametrize("fixed", [True, False])
def test_flex_pose_keeps_deformation_and_resize_clears_it(glove, fixed):
    scene = GloveScene(object_fixed=fixed, glove_model=glove)
    scene.data.qpos[scene._flex_qpos[0]] += 0.001
    scene.data.qvel[scene._flex_dofs[0]] = 0.02
    mujoco.mj_forward(scene.model, scene.data)
    vertices = scene.data.flexvert_xpos[scene._flex_vertices].copy()
    radii = np.linalg.norm(vertices - vertices.mean(axis=0), axis=1)
    original_radii = scene._initial_radii.copy()
    assert np.max(np.abs(radii - original_radii)) > 1e-7
    floor = None
    if not fixed:
        floor_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "scene_floor")
        floor = scene.model.geom_pos[floor_id].copy()
    scene.set_object_pose([0.12, 0.03, 0.18], [0.5, 0.5, 0.5, 0.5])
    moved = scene.data.flexvert_xpos[scene._flex_vertices]
    assert np.linalg.norm(moved - moved.mean(axis=0), axis=1) == pytest.approx(radii, abs=1e-8)
    scene.data.time = 3.5
    resized = scene.resized([0.04, 0.06, 0.08])
    fresh_vertices = resized.data.flexvert_xpos[resized._flex_vertices]
    fresh_radii = np.linalg.norm(fresh_vertices - fresh_vertices.mean(axis=0), axis=1)
    assert fresh_radii == pytest.approx(resized._initial_radii, abs=1e-8)
    assert resized.data.qvel[resized._flex_dofs] == pytest.approx(np.zeros(len(resized._flex_dofs)))
    assert resized.data.time == 3.5
    if not fixed:
        floor_id = mujoco.mj_name2id(resized.model, mujoco.mjtObj.mjOBJ_GEOM, "scene_floor")
        assert resized.model.geom_pos[floor_id] == pytest.approx(floor)


def test_resized_flex_contacts_glove_at_edited_location(glove):
    scene = GloveScene(glove_model=glove).resized([0.08, 0.06, 0.07])
    geom = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "index_dip_link_visual")
    scene.set_object_pose(scene.data.geom_xpos[geom].copy(), [0.5, 0.5, 0.5, 0.5])
    assert any(scene._flex_id in contact.flex for contact in scene.data.contact[:scene.data.ncon])
    scene.set_object_pose([10, 10, 10], [1, 0, 0, 0])
    assert not any(scene._flex_id in contact.flex for contact in scene.data.contact[:scene.data.ncon])


@pytest.mark.parametrize("shape,material", [("sphere", "rubber"), ("box", "wood")])
def test_free_object_reset_keeps_original_pose_after_move_and_resize(glove, shape, material):
    scene = GloveScene(object_fixed=False, object_shape=shape, object_material=material,
                       glove_model=glove)
    original_position, original_quat = scene.object_pose()
    scene.set_object_pose(original_position + [0.05, -0.02, 0.06],
                          [np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
    scene = scene.resized([0.06, 0.04, 0.08])
    scene.reset_object()
    position, quat = scene.object_pose()
    assert position == pytest.approx(original_position, abs=1e-9)
    assert abs(np.dot(quat, original_quat)) == pytest.approx(1, abs=1e-9)
    assert scene.object_dimensions_m == pytest.approx([0.06, 0.04, 0.08])
