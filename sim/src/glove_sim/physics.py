from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .kinematics import GloveKinematics, KinematicsError
from .mapping import FINGERS, MOTOR_FINGER, GloveMapping
from .model_builder import compile_scene
from .objects import ObjectSpec, ObjectRuntime

CONTACT_GROUPS = FINGERS + ("palm",)


@dataclass
class ContactInfo:
    finger: str
    object_name: str
    force_world_n: np.ndarray
    normal_n: float
    position_m: np.ndarray
    penetration_m: float
    geom_name: str


@dataclass
class StepResult:
    timestamp_ns: int = 0
    sim_q: np.ndarray = field(default_factory=lambda: np.zeros(12))
    sim_dq: np.ndarray = field(default_factory=lambda: np.zeros(12))
    contacts: list[ContactInfo] = field(default_factory=list)
    finger_force_n: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in CONTACT_GROUPS})
    finger_penetration_m: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in CONTACT_GROUPS})
    torque_cmd_nm: np.ndarray = field(default_factory=lambda: np.zeros(12))
    torque_raw_nm: np.ndarray = field(default_factory=lambda: np.zeros(12))
    contact_count: int = 0
    step_ms: float = 0.0
    feedback_active: bool = False
    kinematics_ms: float = 0.0
    kinematics_gap_m: float = 0.0
    max_object_disp_m: float = 0.0
    object_disp_m: dict[str, float] = field(default_factory=dict)


class Simulation:
    """MuJoCo scene with a kinematic vE3.3 glove and deformable objects.

    The glove pose is solved from the 12 physical motor angles through the
    extracted CAD closed-chain solver.  Objects are dynamic flex lattices whose
    root body is pose-locked (fixed by default, draggable).  MuJoCo solves the
    contact between the kinematic glove collision proxies and the objects.
    """

    def __init__(self, config: dict[str, Any], mapping: GloveMapping,
                 specs: list[ObjectSpec]) -> None:
        self.config = config
        self.mapping = mapping
        # Every selectable object.  In palm mode only one of them is in the
        # scene at a time (``self.specs``), resting on the palm.
        self.catalog = list(specs)
        self.specs = list(specs)
        self._palm_tree_poses: tuple[np.ndarray, np.ndarray] | None = None
        # Palm mode: object rotation about the palm normal (rad), per object.
        self._palm_yaw: dict[str, float] = {}
        # Contact surfaces being moved after the object changed size/pose:
        # motor idx -> goal surface angle.  The held surface walks there at
        # glove.surface_rate_rad_s, so the hand (sim and real) is pushed open
        # gradually instead of jumping.
        self._surface_goal: dict[int, float] = {}
        self._surface_goal_t = time.monotonic()
        self._placement_data: mujoco.MjData | None = None
        if self.palm_mode and self.catalog:
            wanted = str(self._palm_cfg().get("default_object", self.catalog[0].name))
            first = next((s for s in self.catalog if s.name == wanted), self.catalog[0])
            self.specs = [first]
        # Objects are pose-locked by default (no drift).  Unlocking lets the
        # kinematic glove push/grasp them; the UI exposes this as a checkbox.
        self.objects_locked = bool(config.get("objects", {}).get("locked_default", True))
        self._drag_name: str | None = None
        self._drag_target_pos: np.ndarray | None = None
        self._drag_target_quat: np.ndarray | None = None
        self.model: mujoco.MjModel
        self.data: mujoco.MjData
        model_dir = Path(config["paths"]["glove_model"]).parent
        self.kinematics = GloveKinematics(
            model_dir, mapping, fast=bool(config.get("glove", {}).get("fast_kinematics", True)))
        self._motor_rad = np.zeros(12, dtype=float)
        self._visual_motor_rad = np.zeros(12, dtype=float)
        self._q_tree = np.zeros(len(self.kinematics.joint_names), dtype=float)
        self._last_kinematics_s = 0.0
        self._compile()
        self._solve_and_apply(force=True)
        self._filtered = np.zeros(12)
        self._last_cmd = np.zeros(12)
        self._enabled_since_ns: int | None = None
        self._step_count = 0
        self._flex_info: dict[str, tuple[int, int, int, int, np.ndarray]] = {}
        mujoco.mj_forward(self.model, self.data)
        self._cache_flex_rest()

        # Collision-aware kinematic hand:
        # A purely kinematic glove can be commanded through a locked object
        # (infinite mass vs infinite mass), which shows up as visible "穿模".
        # Keep the real glove angles as the target, but clamp the visual/sim
        # joint motion per finger once the target pose would penetrate a
        # locked object.  A private MjData evaluates the unclamped target pose
        # for force feedback, so the clamp does not silently kill the haptics.
        glove_cfg = config.get("glove", {})
        self.collision_clamp = bool(glove_cfg.get("collision_clamp", True))
        self._motor_target = np.zeros(12, dtype=float)
        self._prev_motor_target = np.zeros(12, dtype=float)
        self._q_tree_target = self._q_tree.copy()
        # Joint-space proxy ("god object"): motor idx -> (closing sign,
        # surface angle where the target first touched a locked object,
        # virtual stiffness Nm/rad of that object).
        self._blocked: dict[int, tuple[int, float, float]] = {}
        self._ghost_data = mujoco.MjData(self.model)
        self._ghost_contacts: list[ContactInfo] | None = None
        self._ghost_finger_force = {k: 0.0 for k in CONTACT_GROUPS}
        self._ghost_finger_pen = {k: 0.0 for k in CONTACT_GROUPS}
        self._ghost_raw = np.zeros(12, dtype=float)
        self._ghost_ready = False
        self._ghost_dirty = False
        if self.palm_mode:
            self.place_on_palm()

    # ------------------------------------------------------------------
    def _cache_flex_rest(self) -> None:
        """Cache static (body-local) flex rest vertices for strain display."""
        self._flex_info = {}
        for obj in self.object_runtime:
            if not obj.spec.deformable:
                continue
            fid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_FLEX, obj.flex_name)
            if fid < 0:
                continue
            adr = int(self.model.flex_vertadr[fid])
            num = int(self.model.flex_vertnum[fid])
            body_id = int(self.model.flex_vertbodyid[adr])
            # model.flex_vert has zero entries for some generated mesh vertices;
            # use the current world positions expressed in the locked root frame
            # so the initial strain is exactly zero.
            pos = np.asarray(self.data.xpos[body_id], dtype=float)
            quat = np.asarray(self.data.xquat[body_id], dtype=float)
            w, x, y, z = quat
            rot = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ])
            verts = np.asarray(self.data.flexvert_xpos[adr:adr + num], dtype=float)
            rest = (verts - pos) @ rot
            self._flex_info[obj.name] = (fid, adr, num, body_id, rest)

    def object_deformation(self) -> dict[str, float]:
        """Max body-local vertex displacement (m) for each deformable object."""
        result: dict[str, float] = {}
        for name, (fid, adr, num, body_id, rest) in self._flex_info.items():
            pos = np.asarray(self.data.xpos[body_id], dtype=float)
            quat = np.asarray(self.data.xquat[body_id], dtype=float)
            w, x, y, z = quat
            rot = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ])
            verts = np.asarray(self.data.flexvert_xpos[adr:adr + num], dtype=float)
            local = (verts - pos) @ rot
            if len(local) == len(rest):
                result[name] = float(np.linalg.norm(local - rest, axis=1).max())
        return result

    # ------------------------------------------------------------------
    def compile_specs(self, specs: list[ObjectSpec]):
        """Compile a scene for ``specs`` without touching the live state.

        Safe to call without the runner lock (MuJoCo releases the GIL while
        compiling), so the ~0.1 s compile does not stall the control loop.
        """
        return compile_scene(self.config, self.mapping, specs)

    def _compile(self, prebuilt=None) -> None:
        self.model, self.data, self.xml = prebuilt or compile_scene(self.config, self.mapping, self.specs)
        self.motor_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in self.mapping.model_joint_names]
        if any(i < 0 for i in self.motor_ids):
            raise RuntimeError(f"模型缺少电机关节: {self.mapping.model_joint_names}")
        self.motor_qposadr = np.asarray([self.model.jnt_qposadr[i] for i in self.motor_ids], dtype=int)
        self.motor_dofadr = np.asarray([self.model.jnt_dofadr[i] for i in self.motor_ids], dtype=int)
        self.glove_geoms: dict[int, str] = {}
        for gid in range(self.model.ngeom):
            if self.model.geom_contype[gid] != 1:
                continue  # only geoms that actually collide with objects
            name = self.model.geom(gid).name
            if name.startswith("col_"):
                parts = name.split("_", 2)
                if len(parts) >= 2:
                    self.glove_geoms[gid] = parts[1]
            elif self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                body = self.model.body(self.model.geom_bodyid[gid]).name
                self.glove_geoms[gid] = next((f for f in FINGERS if body.startswith(f)), "palm")
        self.object_runtime: list[ObjectRuntime] = []
        for spec in self.specs:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"obj_{spec.name}_free")
            if jid < 0:
                raise RuntimeError(f"缺少对象根关节: {spec.name}")
            self.object_runtime.append(
                ObjectRuntime(
                    spec=spec,
                    body_name=f"obj_{spec.name}",
                    free_joint=f"obj_{spec.name}_free",
                    qposadr=int(self.model.jnt_qposadr[jid]),
                    flex_name=f"flex_{spec.name}",
                    target_position=np.asarray(spec.position, dtype=float).copy(),
                    target_quat=np.asarray(spec.quat, dtype=float).copy(),
                )
            )
        self.object_geom_ids: dict[int, ObjectRuntime] = {}
        for obj in self.object_runtime:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obj_{obj.name}_geom")
            if gid >= 0:
                self.object_geom_ids[gid] = obj
        # Damp free-object motion; pose locking still zeroes it when enabled.
        free_damping = float(self.config.get("objects", {}).get("dynamic_damping", 0.0))
        if free_damping > 0:
            for obj in self.object_runtime:
                d = self._object_dof(obj)
                self.model.dof_damping[d:d + 6] = free_damping
        self._finger_motors = {finger: [i for i, f in enumerate(MOTOR_FINGER) if f == finger]
                               for finger in FINGERS}
        self._tree_qposadr = np.asarray([
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.kinematics.joint_names
        ], dtype=int)
        self._tree_dofadr = np.asarray([
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.kinematics.joint_names
        ], dtype=int)
        self._glove_dofs = self._tree_dofadr.astype(int, copy=True)
        self._lock_objects()

    def _object_dof(self, obj: ObjectRuntime) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, obj.free_joint)
        return int(self.model.jnt_dofadr[jid])

    def _lock_objects(self) -> None:
        for obj in self.object_runtime:
            if self._drag_name is not None and obj.name == self._drag_name:
                continue
            a = obj.qposadr
            self.data.qpos[a:a + 3] = obj.target_position
            self.data.qpos[a + 3:a + 7] = obj.target_quat / np.linalg.norm(obj.target_quat)
            d = self._object_dof(obj)
            self.data.qvel[d:d + 6] = 0.0

    # ------------------------------------------------------------------
    def _apply_pose(self) -> None:
        # The vE3.3 glove is kinematic.  Writing the cached qpos addresses and
        # zeroing the whole glove qvel avoids 68 name lookups twice per step.
        self.data.qpos[self._tree_qposadr] = self._q_tree
        self.data.qvel[self._tree_dofadr] = 0.0

    def _solve_and_apply(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_kinematics_s < 0.02:
            return
        t0 = time.perf_counter()
        try:
            self._q_tree = self.kinematics.full_pose(self._motor_rad, self.model, self.mapping)
            self.kinematics.copy_slot("target", "visual")
        except KinematicsError:
            # Keep the last valid pose; the telemetry/UI reports the error.
            return
        self._last_kinematics_s = now
        self.kinematics.last_solve_ms = (time.perf_counter() - t0) * 1e3
        self._apply_pose()

    def update_glove(self, motor_rad: np.ndarray) -> None:
        target = np.asarray(motor_rad, dtype=float).copy()
        previous = self._motor_target
        delta = target - previous
        cmd = target.copy()
        if not self.collision_clamp or not self.objects_locked:
            self._blocked.clear()
        else:
            self._advance_surfaces()
            margin = float(self.config.get("glove", {}).get("release_margin_rad", 0.01))
            for idx, (sign, frozen, _k) in list(self._blocked.items()):
                # Hold the visible finger at the surface while the real finger
                # is beyond it; release only once the real finger has backed
                # out past the surface.  (The old rule used the per-frame
                # direction, so encoder noise made the visible finger jump
                # between the surface and the inside of the object.)
                inside = sign * (target[idx] - frozen)
                if inside > 0.0:
                    cmd[idx] = frozen
                elif inside < -margin:
                    del self._blocked[idx]
                    self._surface_goal.pop(idx, None)
                # else: just outside -- the visible finger follows the real one
                # and the spring is 0, but the surface is kept, so a re-press
                # meets the same wall instead of re-freezing a new one (the
                # release/re-engage switching made contact chatter).
        target_changed = not np.array_equal(target, previous)
        self._prev_motor_target = previous.copy()
        self._motor_target = target.copy()
        self._motor_rad = cmd
        t0 = time.perf_counter()
        try:
            if target_changed:
                target_pose = self.kinematics.full_pose(target, self.model, self.mapping)
            else:
                target_pose = self._q_tree_target
            if np.array_equal(cmd, target):
                pose = target_pose
                self.kinematics.copy_slot("target", "visual")
            elif np.array_equal(cmd, self._visual_motor_rad):
                # Contact clamp held the visible pose; no second CAD solve.
                pose = self._q_tree
            else:
                pose = self.kinematics.full_pose(cmd, self.model, self.mapping, slot="visual")
        except KinematicsError:
            # Keep the last valid pose; the telemetry/UI reports the error.
            return
        self._q_tree_target = target_pose
        self._q_tree = pose
        self._visual_motor_rad = cmd.copy()
        self.kinematics.last_solve_ms = (time.perf_counter() - t0) * 1e3
        self._apply_pose()
        self._ghost_ready = True
        self._ghost_dirty = True

    def _refresh_visual_pose(self) -> None:
        """Recompute the displayed hand pose after contact blocks changed."""
        cmd = self._motor_target.copy()
        for idx, (sign, frozen, _k) in self._blocked.items():
            if sign * (cmd[idx] - frozen) > 0.0:
                cmd[idx] = frozen
        if np.array_equal(cmd, self._visual_motor_rad):
            return
        try:
            pose = self.kinematics.full_pose(cmd, self.model, self.mapping, slot="visual")
        except KinematicsError:
            return
        self._motor_rad = cmd
        self._q_tree = pose
        self._visual_motor_rad = cmd.copy()
        self._apply_pose()

    def _update_contact_blocks(self, contacts: list[ContactInfo]) -> None:
        """Engage the proxy for fingers whose *target* pose touches a locked object.

        Blocks are only released in update_glove(), when the real finger backs
        out past the surface.  They are deliberately *not* released when the
        ghost contact disappears: a target that has been pushed through a
        locked object loses its contacts, which previously released the clamp
        and dropped the feedback to zero exactly when squeezing hardest.
        """
        if not self.collision_clamp or not self.objects_locked:
            self._blocked.clear()
            return
        soft_allow = float(self.config.get("glove", {}).get("clamp_soft_allow_m", 0.004))
        active: dict[str, float] = {}
        for contact in contacts:
            if contact.object_name == self._drag_name:
                continue
            spec = next((s for s in self.specs if s.name == contact.object_name), None)
            allowed = soft_allow if (spec is not None and spec.deformable) else 0.0
            if contact.penetration_m > allowed:
                k = self._proxy_stiffness(spec)
                active[contact.finger] = max(active.get(contact.finger, 0.0), k)
        for finger, k in active.items():
            for idx in self._finger_motors.get(finger, ()):
                direction = self._motor_target[idx] - self._prev_motor_target[idx]
                # Only block the closing direction.  A contact that appears
                # while the wearer is opening/backing off must not freeze the
                # retreat.
                if direction <= 1e-12 or idx in self._blocked:
                    continue
                self._blocked[idx] = (
                    int(np.sign(direction)),
                    float(self._prev_motor_target[idx]),
                    float(k),
                )

    def _release_stale_blocks(self) -> None:
        """Drop blocks of fingers whose visible pose is clear of every object.

        Blocks deliberately survive the loss of the ghost contact (squeezing
        through a locked object).  But when the object itself has gone --
        dragged away, unlocked, resized or nudged -- the finger stayed locked
        with no visible contact.  The visible (clamped) finger sits at the
        surface while legitimately blocked, so a clearance of ~1 cm between
        conservative bounding spheres only triggers when nothing is there.
        """
        if not self._blocked:
            return
        clear = float(self.config.get("glove", {}).get("stale_release_clearance_m", 0.01))
        centres, radii = [], []
        for obj in self.object_runtime:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj.body_name)
            if bid < 0:
                continue
            centres.append(self.data.xpos[bid])
            radii.append(self.bounding_radius(obj.spec) + self.object_deformation_margin(obj))
        if not centres:
            self._blocked.clear()
            return
        centres = np.asarray(centres)
        radii = np.asarray(radii)
        for finger, motors in self._finger_motors.items():
            held = [i for i in motors if i in self._blocked]
            if not held:
                continue
            geoms = [g for g, f in self.glove_geoms.items() if f == finger]
            if geoms:
                pos = self.data.geom_xpos[geoms]
                gap = (np.linalg.norm(pos[:, None, :] - centres[None, :, :], axis=2)
                       - self.model.geom_rbound[geoms][:, None] - radii[None, :])
                if float(gap.min()) < clear:
                    continue
            for i in held:
                del self._blocked[i]

    def object_deformation_margin(self, obj: ObjectRuntime) -> float:
        """Extra radius for soft objects whose surface may bulge (m)."""
        return 0.005 if obj.spec.deformable else 0.0

    def _proxy_stiffness(self, spec: ObjectSpec | None) -> float:
        """Virtual joint stiffness (Nm/rad) from the object's 0..1 hardness.

        The driver-side spring runs at the driver rate, so it uses its own
        (much stiffer) range; the off-board spring keeps the soft range.
        """
        glove = self.config.get("glove", {})
        if self._feedback_mode() == "driver_impedance":
            lo, hi = glove.get("driver_impedance", {}).get("stiffness_nm_per_rad", [5.0, 60.0])
        else:
            lo, hi = glove.get("proxy_stiffness_nm_per_rad", [2.0, 20.0])
        hardness = 0.5 if spec is None else max(0.0, min(1.0, float(spec.stiffness)))
        return float(lo) + (float(hi) - float(lo)) * hardness

    def _feedback_mode(self) -> str:
        return str(self.config.get("glove", {}).get("feedback_mode", "proxy_spring"))

    def contact_blocks(self) -> dict[int, tuple[int, float, float]]:
        """motor index -> (closing sign, surface angle rad, stiffness Nm/rad)."""
        return dict(self._blocked)

    def proxy_feedback_torque(self) -> np.ndarray:
        """Joint-space virtual spring between the real finger and its proxy.

        tau_i = -sign_i * k_i * depth_i * proxy_gain, depth_i = how far the
        real finger angle is past the surface angle (rad).  Model-joint
        coordinates (closing = +); converted to motor axes by the SDK wrapper.
        Nominal torque before caps/slew/filter, not a measured force.
        """
        tau = np.zeros(12, dtype=float)
        glove = self.config.get("glove", {})
        joint_gain = np.ones(12)
        if self._feedback_mode() == "driver_impedance":
            cfg = glove.get("driver_impedance", {})
            gain = float(cfg.get("gain", 1.0))
            jg = np.asarray(cfg.get("joint_gain", 1.0), dtype=float).reshape(-1)
            joint_gain = np.full(12, float(jg[0])) if jg.size == 1 else jg
        else:
            gain = float(glove.get("proxy_gain", 1.0))
        for idx, (sign, frozen, k) in self._blocked.items():
            depth = max(0.0, sign * (float(self._motor_target[idx]) - frozen))
            tau[idx] = -sign * k * depth * gain * float(joint_gain[idx])
        return tau

    def _refresh_ghost_feedback(self) -> None:
        """Evaluate the unclamped target pose in a private MjData.

        The main state stays penetration-free; this ghost state keeps the
        original contact force feel while the visible hand is clamped.
        """
        if not self.collision_clamp or not self.objects_locked or not self._ghost_ready:
            self._ghost_contacts = None
            self._ghost_raw = np.zeros(12)
            self._ghost_dirty = False
            self._blocked.clear()
            return
        ghost = self._ghost_data
        ghost.qpos[:] = self.data.qpos
        ghost.qvel[:] = self.data.qvel
        ghost.qpos[self._tree_qposadr] = self._q_tree_target
        ghost.qvel[self._tree_dofadr] = 0.0
        mujoco.mj_forward(self.model, ghost)
        contacts, finger_force, finger_pen = self.compute_contacts(ghost)
        self._ghost_contacts = contacts
        self._ghost_finger_force = finger_force
        self._ghost_finger_pen = finger_pen
        self._ghost_raw = self.raw_feedback_torque(contacts, data=ghost)
        self._ghost_dirty = False
        self._update_contact_blocks(contacts)
        self._refresh_visual_pose()

    def _invalidate_ghost(self, *, clear_blocks: bool = True) -> None:
        self._ghost_dirty = True
        self._ghost_contacts = None
        self._ghost_raw = np.zeros(12)
        if clear_blocks:
            self._blocked.clear()

    def _use_ghost_feedback(self) -> bool:
        return bool(self.collision_clamp and self.objects_locked and self._ghost_ready
                    and self._ghost_contacts is not None)

    def sim_q(self) -> np.ndarray:
        return np.asarray([self.data.qpos[a] for a in self.motor_qposadr], dtype=float)

    def sim_dq(self) -> np.ndarray:
        return np.asarray([self.data.qvel[a] for a in self.motor_dofadr], dtype=float)

    # ------------------------------------------------------------------
    def _contact_force(self, index: int,
                       data: mujoco.MjData | None = None) -> tuple[np.ndarray, float]:
        data = self.data if data is None else data
        result = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(self.model, data, index, result)
        frame = np.asarray(data.contact[index].frame, dtype=float).reshape(3, 3)
        world = frame.T @ result[:3]
        return world, float(result[0])

    def compute_contacts(self, data: mujoco.MjData | None = None
                         ) -> tuple[list[ContactInfo], dict[str, float], dict[str, float]]:
        data = self.data if data is None else data
        contacts: list[ContactInfo] = []
        finger_force = {k: 0.0 for k in CONTACT_GROUPS}
        finger_pen = {k: 0.0 for k in CONTACT_GROUPS}
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self.glove_geoms:
                glove_gid, object_gid = g1, g2
            elif g2 in self.glove_geoms:
                glove_gid, object_gid = g2, g1
            else:
                continue
            flex_id = int(np.asarray(c.flex).reshape(-1)[1]) if np.ndim(c.flex) else -1
            obj = self.object_by_flex(flex_id) or self.object_geom_ids.get(object_gid)
            if obj is None:
                continue
            world, normal = self._contact_force(i, data)
            # mj_contactForce gives the force geom1 exerts on geom2 (normal
            # points geom1 -> geom2).  Express it as the force on the glove.
            if glove_gid == g1:
                world = -world
            finger = self.glove_geoms[glove_gid]
            pen = max(0.0, -float(c.dist))
            contacts.append(
                ContactInfo(
                    finger=finger,
                    object_name=obj.name,
                    force_world_n=world,
                    normal_n=normal,
                    position_m=np.asarray(c.pos, dtype=float).copy(),
                    penetration_m=pen,
                    geom_name=self.model.geom(glove_gid).name,
                )
            )
            finger_force[finger] += float(np.linalg.norm(world))
            finger_pen[finger] = max(finger_pen[finger], pen)
        return contacts, finger_force, finger_pen

    def visual_contact_stats(self) -> tuple[int, float]:
        """Actual main-data glove/object contacts (what the rendered hand does).

        This is deliberately separate from the ghost target contacts used for
        force feedback; the UI shows both so "feedback touch" and visible
        penetration cannot be confused.
        """
        count = 0
        max_pen = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 not in self.glove_geoms and g2 not in self.glove_geoms:
                continue
            penetration = max(0.0, -float(contact.dist))
            # Touching with zero depth is not a visible collision.
            if penetration > 5e-5:
                count += 1
                max_pen = max(max_pen, penetration)
        return count, max_pen

    def object_by_flex(self, flex_id: int) -> ObjectRuntime | None:
        if flex_id < 0:
            return None
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_FLEX, int(flex_id))
        for obj in self.object_runtime:
            if obj.flex_name == name:
                return obj
        return None

    def raw_feedback_torque(self, contacts: list[ContactInfo],
                            data: mujoco.MjData | None = None) -> np.ndarray:
        """Project simulated contact forces onto the glove's motor DOFs.

        This is a simulated contact force, not a measured force.  The result is
        a nominal torque command before caps, slew and filtering.
        """
        data = self.data if data is None else data
        tau = np.zeros(12, dtype=float)
        if not contacts:
            return tau
        dq_du = self.kinematics.tree_derivative()
        glove_dofs = self._glove_dofs
        for contact in contacts:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom_name)
            if gid < 0:
                continue
            body_id = int(self.model.geom_bodyid[gid])
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jac(self.model, data, jacp, jacr, contact.position_m, body_id)
            # mj_jac sees the open tree; chain-rule through the closed-chain
            # derivative to obtain d(point)/d(motor angle).
            jac_motor = jacp[:, glove_dofs] @ dq_du
            force = np.asarray(contact.force_world_n, dtype=float)
            tau += jac_motor.T @ force
        gain = float(self.config["glove"].get("feedback_gain", 0.08))
        return tau * gain

    def feedback_step(self, raw: np.ndarray, *, enabled: bool, dt_s: float, now_ns: int) -> np.ndarray:
        cfg = self.config["glove"]
        limit = np.asarray(cfg.get("torque_limit_nm", [4.0] * 12), dtype=float)
        slew = np.asarray(cfg.get("slew_nm_s", [60.0] * 12), dtype=float)
        hz = float(cfg.get("feedback_filter_hz", 25.0))
        alpha = 1.0 - math.exp(-2.0 * math.pi * hz * max(dt_s, 1e-6))
        target = np.clip(raw, -limit, limit) if enabled else np.zeros(12)
        if enabled and self._enabled_since_ns is None:
            self._enabled_since_ns = now_ns
        if not enabled:
            self._enabled_since_ns = None
        if enabled and self._enabled_since_ns is not None:
            ramp = min(1.0, (now_ns - self._enabled_since_ns) / 1e9
                       / max(1e-6, float(cfg.get("feedback_ramp_s", 0.15))))
            target = target * ramp
        delta = np.clip(target - self._last_cmd, -slew * max(dt_s, 1e-6), slew * max(dt_s, 1e-6))
        self._last_cmd = self._last_cmd + delta
        self._filtered += alpha * (self._last_cmd - self._filtered)
        deadband = float(cfg.get("feedback_deadband_nm", 0.01))
        out = np.where(np.abs(self._filtered) < deadband, 0.0, self._filtered)
        return np.asarray(out, dtype=float)

    # ------------------------------------------------------------------
    def step(self, *, feedback_enabled: bool, dt_s: float) -> StepResult:
        start = time.perf_counter()
        self._apply_pose()
        self._apply_drag_servo()
        if self.objects_locked:
            self._lock_objects()
        if self._ghost_dirty and self.objects_locked:
            self._refresh_ghost_feedback()
        mujoco.mj_step(self.model, self.data)
        self._release_stale_blocks()
        if self._use_ghost_feedback():
            contacts = self._ghost_contacts or []
            finger_force = dict(self._ghost_finger_force)
            finger_pen = dict(self._ghost_finger_pen)
            if self._feedback_mode() in ("proxy_spring", "driver_impedance"):
                # driver_impedance: the spring runs in the motor driver; this
                # is its nominal (uncapped) torque, used for display/telemetry.
                raw = self.proxy_feedback_torque()
            else:
                raw = self._ghost_raw.copy()
        else:
            contacts, finger_force, finger_pen = self.compute_contacts()
            raw = self.raw_feedback_torque(contacts)
        self._apply_pose()
        # Keep the UI/picking target in sync with any physical object.
        for obj in self.object_runtime:
            if not self.objects_locked or obj.name == self._drag_name:
                a = obj.qposadr
                obj.target_position = np.asarray(self.data.qpos[a:a + 3], dtype=float).copy()
                q = np.asarray(self.data.qpos[a + 3:a + 7], dtype=float)
                obj.target_quat = q / max(1e-12, float(np.linalg.norm(q)))
        object_disp = self.object_deformation()
        now = time.monotonic_ns()
        cmd = self.feedback_step(raw, enabled=feedback_enabled, dt_s=dt_s, now_ns=now)
        self._step_count += 1
        return StepResult(
            timestamp_ns=now,
            sim_q=self.sim_q().copy(),
            sim_dq=self.sim_dq().copy(),
            contacts=contacts,
            finger_force_n=finger_force,
            finger_penetration_m=finger_pen,
            torque_cmd_nm=cmd.copy(),
            torque_raw_nm=raw.copy(),
            contact_count=len(contacts),
            step_ms=(time.perf_counter() - start) * 1e3,
            feedback_active=feedback_enabled,
            kinematics_ms=self.kinematics.last_solve_ms,
            kinematics_gap_m=self.kinematics.last_gap_m,
            max_object_disp_m=max(object_disp.values(), default=0.0),
            object_disp_m=object_disp,
        )

    def reset_object_poses(self) -> None:
        """Restore every object to its configured start pose (palm mode: re-seat)."""
        if self.palm_mode:
            previous = dict(self._blocked)
            self._palm_yaw.clear()
            self.place_on_palm()
            self.refit_contact_blocks(previous)
            return
        if self._drag_name is not None:
            self.end_drag()
        for obj in self.object_runtime:
            spec = next((s for s in self.specs if s.name == obj.name), None)
            if spec is None:
                continue
            obj.target_position = np.asarray(spec.position, dtype=float).copy()
            obj.target_quat = np.asarray(spec.quat, dtype=float).copy()
        if self.objects_locked:
            self._lock_objects()
        self._invalidate_ghost()
        mujoco.mj_forward(self.model, self.data)

    def set_objects_locked(self, locked: bool) -> None:
        """Toggle fixed root poses.

        Re-locking snapshots the current physical pose so a free object does
        not snap back to its pre-unlock target.
        """
        locked = bool(locked)
        if locked and not self.objects_locked:
            for obj in self.object_runtime:
                a = obj.qposadr
                obj.target_position = np.asarray(self.data.qpos[a:a + 3], dtype=float).copy()
                q = np.asarray(self.data.qpos[a + 3:a + 7], dtype=float)
                obj.target_quat = q / max(1e-12, float(np.linalg.norm(q)))
        self.objects_locked = locked
        self._invalidate_ghost()
        if locked:
            self._lock_objects()
            mujoco.mj_forward(self.model, self.data)

    def begin_drag(self, name: str) -> None:
        """Attach a PD servo to one free object so it follows the cursor.

        Unlike writing qpos directly, the servo goes through MuJoCo, so the
        dragged object collides with the glove and the other objects instead of
        passing through them.
        """
        for obj in self.object_runtime:
            if obj.name != name:
                continue
            a = obj.qposadr
            self._drag_name = name
            self._drag_target_pos = np.asarray(self.data.qpos[a:a + 3], dtype=float).copy()
            q = np.asarray(self.data.qpos[a + 3:a + 7], dtype=float)
            self._drag_target_quat = q / max(1e-12, float(np.linalg.norm(q)))
            self._invalidate_ghost()
            return
        raise KeyError(name)

    def update_drag(self, position: np.ndarray | None = None,
                    quat: np.ndarray | None = None) -> None:
        if position is not None:
            pos = np.asarray(position, dtype=float).copy()
            # Limit the cursor target speed per control update.  Without this,
            # a fast mouse jump can carry a small object through a thin glove
            # proxy between two solver steps (visual tunnelling).
            if self._drag_target_pos is not None:
                limit = float(self.config.get("objects", {}).get("drag_max_step_m", 0.015))
                delta = pos - self._drag_target_pos
                dist = float(np.linalg.norm(delta))
                if dist > limit > 0:
                    pos = self._drag_target_pos + delta * (limit / dist)
            self._drag_target_pos = pos
            self._invalidate_ghost(clear_blocks=False)
        if quat is not None:
            q = np.asarray(quat, dtype=float)
            self._drag_target_quat = q / max(1e-12, float(np.linalg.norm(q)))
            self._invalidate_ghost(clear_blocks=False)

    def end_drag(self) -> None:
        if self._drag_name is None:
            return
        obj = next((o for o in self.object_runtime if o.name == self._drag_name), None)
        if obj is not None:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj.body_name)
            if bid >= 0:
                self.data.xfrc_applied[bid, :] = 0.0
            a = obj.qposadr
            obj.target_position = np.asarray(self.data.qpos[a:a + 3], dtype=float).copy()
            q = np.asarray(self.data.qpos[a + 3:a + 7], dtype=float)
            obj.target_quat = q / max(1e-12, float(np.linalg.norm(q)))
        self._drag_name = None
        self._drag_target_pos = None
        self._drag_target_quat = None
        self._invalidate_ghost()
        if self.objects_locked:
            self._lock_objects()
            mujoco.mj_forward(self.model, self.data)

    def _apply_drag_servo(self) -> None:
        if self._drag_name is None or self._drag_target_pos is None:
            return
        obj = next((o for o in self.object_runtime if o.name == self._drag_name), None)
        if obj is None:
            return
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj.body_name)
        if bid < 0:
            return
        pos = np.asarray(self.data.xpos[bid], dtype=float)
        quat = np.asarray(self.data.xquat[bid], dtype=float)
        d = self._object_dof(obj)
        lin_vel = np.asarray(self.data.qvel[d:d + 3], dtype=float)
        ang_vel = np.asarray(self.data.qvel[d + 3:d + 6], dtype=float)
        mass = max(1e-6, float(self.model.body_mass[bid]))
        inertia = np.maximum(1e-8, np.asarray(self.model.body_inertia[bid], dtype=float))
        i_avg = float(np.mean(inertia))
        kp, kd = 600.0, 45.0
        force = mass * (kp * (self._drag_target_pos - pos) - kd * lin_vel)
        force = np.clip(force, -10.0, 10.0)
        # Quaternion error in world frame: q_target * conj(q_current).
        target_q = self._drag_target_quat
        if target_q is None:
            target_q = quat
        w1, x1, y1, z1 = target_q
        w2, x2, y2, z2 = -quat[0], quat[1], quat[2], quat[3]
        ew = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        ex = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        ey = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        ez = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        if ew < 0:
            ew, ex, ey, ez = -ew, -ex, -ey, -ez
        vnorm = math.sqrt(ex * ex + ey * ey + ez * ez)
        if vnorm > 1e-9:
            angle = 2.0 * math.atan2(vnorm, max(-1.0, min(1.0, ew)))
            rotvec = np.asarray([ex, ey, ez]) * (angle / vnorm)
        else:
            rotvec = np.zeros(3)
        torque = i_avg * (kp * rotvec - kd * ang_vel)
        torque = np.clip(torque, -1.5, 1.5)
        self.data.xfrc_applied[bid, :3] = force
        self.data.xfrc_applied[bid, 3:] = torque

    def set_object_pose(self, name: str, position: np.ndarray | None = None,
                        quat: np.ndarray | None = None) -> None:
        for obj in self.object_runtime:
            if obj.name != name:
                continue
            if position is not None:
                obj.target_position = np.asarray(position, dtype=float)
            if quat is not None:
                obj.target_quat = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
            self._lock_objects()
            self._invalidate_ghost()
            mujoco.mj_forward(self.model, self.data)
            return
        raise KeyError(name)

    @staticmethod
    def bounding_radius(spec: ObjectSpec) -> float:
        """Radius of a sphere that encloses the object (size = diameter/edge)."""
        factor = {"sphere": 0.5, "box": 0.5 * math.sqrt(3.0), "cylinder": 0.5 * math.sqrt(2.0)}
        return float(spec.size_m) * factor.get(spec.shape, 0.5 * math.sqrt(3.0))

    def free_position(self, name: str, position: np.ndarray, *, margin_m: float = 0.008,
                      max_iter: int = 50) -> np.ndarray:
        """Shift ``position`` sideways (+y) until object ``name`` overlaps no other object.

        Overlapping objects (especially the soft flex ball) create hundreds of
        object-object contacts and can make one physics step take >50 ms.
        """
        spec = next(s for s in self.specs if s.name == name)
        pos = np.asarray(position, dtype=float).copy()
        others = [(o, next(s for s in self.specs if s.name == o.name))
                  for o in self.object_runtime if o.name != name]
        for _ in range(max_iter):
            moved = False
            for obj, other in others:
                clearance = self.bounding_radius(spec) + self.bounding_radius(other) + margin_m
                if float(np.linalg.norm(pos - obj.target_position)) < clearance:
                    pos[1] = float(obj.target_position[1]) + clearance + 1e-4
                    moved = True
            if not moved:
                break
        return pos

    def specs_with(self, name: str, *, size_m: float | None = None,
                   stiffness: float | None = None) -> list[ObjectSpec] | None:
        """Copies of the scene specs with one object changed (None: not in scene)."""
        import copy
        if not any(s.name == name for s in self.specs):
            return None
        out = []
        for spec in self.specs:
            spec = copy.copy(spec)
            if spec.name == name:
                if size_m is not None:
                    spec.size_m = float(size_m)
                if stiffness is not None:
                    spec.stiffness = float(stiffness)
            out.append(spec)
        return out

    def set_object_shape(self, name: str, *, size_m: float | None = None,
                         stiffness: float | None = None, prebuilt=None,
                         refit: bool = True) -> dict[str, Any] | None:
        """Rebuild the scene with an updated object parameter.

        The generated scene compiles in ~0.1 s on the target machine; the UI
        throttles this to ``objects.rebuild_hz`` (default 10 Hz).  In palm
        mode a resized object is re-seated on the palm and the holding hand
        refitted; ``refit=False`` returns the refit request instead, for the
        caller to run refit_compute/refit_apply outside the runner lock.
        """
        for spec in self.catalog:
            if spec.name == name:
                if size_m is not None:
                    spec.size_m = float(size_m)
                if stiffness is not None:
                    spec.stiffness = float(stiffness)
                break
        else:
            raise KeyError(name)
        if not any(s.name == name for s in self.specs):
            return None  # not in the scene (palm mode); used when it is selected
        previous = dict(self._blocked)
        self._rebuild(prebuilt=prebuilt)
        if self.palm_mode:
            if size_m is not None:
                self.place_on_palm()
            # The hand that was holding the object now meets its new surface:
            # a grown ball pushes the fingers (sim and real) open.
            if not refit:
                # Caller runs compute/apply; previous surfaces hold meanwhile.
                return self.refit_request(previous)
            self.refit_contact_blocks(previous)
        return None

    def _rebuild(self, *, keep_object_poses: bool = True, prebuilt=None) -> None:
        """Recompile the scene from ``self.specs``, keeping the glove state."""
        if self._drag_name is not None:
            self.end_drag()
        old_q = {self.model.joint(i).name: float(self.data.qpos[self.model.jnt_qposadr[i]])
                 for i in range(self.model.njnt)}
        old_dq = {self.model.joint(i).name: float(self.data.qvel[self.model.jnt_dofadr[i]])
                  for i in range(self.model.njnt)}
        old_poses = {o.name: (o.target_position.copy(), o.target_quat.copy()) for o in self.object_runtime}
        self._compile(prebuilt)
        for i in range(self.model.njnt):
            name = self.model.joint(i).name
            if name in old_q:
                self.data.qpos[self.model.jnt_qposadr[i]] = old_q[name]
                self.data.qvel[self.model.jnt_dofadr[i]] = old_dq[name]
        if keep_object_poses:
            for obj in self.object_runtime:
                if obj.name in old_poses:
                    obj.target_position, obj.target_quat = old_poses[obj.name]
        self._lock_objects()
        self._apply_pose()
        mujoco.mj_forward(self.model, self.data)
        self._cache_flex_rest()
        # The compiled model/data changed; the ghost evaluator must follow it.
        self._ghost_data = mujoco.MjData(self.model)
        self._placement_data = None
        self._q_tree_target = self._q_tree.copy()
        self._ghost_ready = False
        self._invalidate_ghost()

    # ------------------------------------------------------------------
    # Palm-mounted object
    @property
    def palm_mode(self) -> bool:
        return str(self.config.get("objects", {}).get("placement", "scene")) == "palm"

    def _palm_cfg(self) -> dict[str, Any]:
        return dict(self.config.get("objects", {}).get("palm") or {})

    @property
    def active_object(self) -> str:
        return self.specs[0].name if self.specs else ""

    def select_object(self, name: str, prebuilt=None, *, refit: bool = True) -> dict[str, Any] | None:
        """Palm mode: replace the object on the palm with catalog entry ``name``.

        ``prebuilt`` = ``compile_specs([spec])`` done outside the runner lock.
        """
        spec = next((s for s in self.catalog if s.name == name), None)
        if spec is None:
            raise KeyError(name)
        if not self.palm_mode or (self.specs and self.specs[0].name == name):
            return None
        previous = dict(self._blocked)
        self.specs = [spec]
        self._rebuild(keep_object_poses=False, prebuilt=prebuilt)
        self.place_on_palm()
        if not refit:
            return self.refit_request(previous)
        self.refit_contact_blocks(previous)
        return None

    def _palm_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(origin, rotation, forward, palm normal, lateral) of the hand base."""
        cfg = self._palm_cfg()
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, str(cfg.get("body", "base_link")))
        if bid < 0:
            raise RuntimeError("模型缺少手掌基座 body")
        origin = np.asarray(self.data.xpos[bid], dtype=float).copy()
        rot = np.asarray(self.data.xmat[bid], dtype=float).reshape(3, 3).copy()
        forward = rot[:, 0].copy()                                   # towards the fingertips
        normal = -rot[:, 2] * float(cfg.get("normal_sign", 1.0))     # out of the palm
        lateral = np.cross(forward, normal)
        return origin, rot, forward, normal, lateral

    def _placement_tree_poses(self) -> tuple[np.ndarray, np.ndarray]:
        """Glove tree poses for the open hand and a slightly closed "clear" hand."""
        if self._palm_tree_poses is None:
            limits = np.asarray([m.model_rad_limits() for m in self.mapping.motors], dtype=float)
            open_q = np.clip(np.zeros(12), limits[:, 0], limits[:, 1])
            frac = float(self._palm_cfg().get("clear_close_fraction", 0.15))
            clear_q = np.clip(frac * limits[:, 1], limits[:, 0], limits[:, 1])
            open_tree = self.kinematics.full_pose(open_q, self.model, self.mapping, slot="placement")
            clear_tree = self.kinematics.full_pose(clear_q, self.model, self.mapping, slot="placement")
            self._palm_tree_poses = (np.asarray(open_tree, dtype=float).copy(),
                                     np.asarray(clear_tree, dtype=float).copy())
        return self._palm_tree_poses

    def _placement_touches(self, obj: ObjectRuntime, centre: np.ndarray, quat: np.ndarray,
                           tree: np.ndarray) -> bool:
        """True if the object at ``centre`` touches the glove in pose ``tree``."""
        if self._placement_data is None:
            self._placement_data = mujoco.MjData(self.model)
        aux = self._placement_data
        aux.qpos[:] = self.model.qpos0
        aux.qpos[self._tree_qposadr] = tree
        a = obj.qposadr
        aux.qpos[a:a + 3] = centre
        aux.qpos[a + 3:a + 7] = quat
        mujoco.mj_fwdPosition(self.model, aux)
        for i in range(aux.ncon):
            c = aux.contact[i]
            if (int(c.geom1) in self.glove_geoms) != (int(c.geom2) in self.glove_geoms):
                return True
        return False

    @staticmethod
    def _mat_to_quat(rot: np.ndarray) -> np.ndarray:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rot, dtype=float).reshape(-1))
        return quat / max(1e-12, float(np.linalg.norm(quat)))

    def palm_position(self, obj: ObjectRuntime) -> tuple[np.ndarray, np.ndarray]:
        """Seat the object on the palm.

        The object's centre stays over the configured palm point
        (``anchor_local_m``, base_link frame) for every size: it is only ever
        moved along the palm normal, away from the palm -- first to rest on
        the palm surface, then further out until the open hand is clear.  (The
        old seat kept the front edge at the finger roots, so a bigger object
        slid towards the wrist.)  ``palm_yaw`` rotates it about the palm normal
        through its centre.
        """
        cfg = self._palm_cfg()
        spec = obj.spec
        origin, rot, forward, normal, lateral = self._palm_frame()
        if spec.shape == "cylinder":
            # Axis across the palm, like gripping a bar.
            orient = np.column_stack([forward, normal, lateral])
        else:
            orient = np.column_stack([forward, np.cross(normal, forward), normal])
        yaw = float(self._palm_yaw.get(obj.name, 0.0))
        if yaw:
            orient = self._axis_rotation(normal, yaw) @ orient
        quat = self._mat_to_quat(orient)
        half = 0.5 * float(spec.size_m)    # extent along the normal (all shapes)
        anchor = origin + rot @ np.asarray(cfg.get("anchor_local_m", (0.072, -0.005, -0.001)),
                                           dtype=float)
        gap = float(cfg.get("gap_m", 0.002))
        lift_max = float(cfg.get("max_lift_m", 0.05))
        base = anchor + normal * (half + gap)
        # Clear of the open hand and of a relaxed, slightly closed hand
        # (objects.palm.clear_close_fraction), so resting fingers do not
        # start in contact -- reached only by lifting away from the palm.
        trees = self._placement_tree_poses()
        lift = 0.0
        while lift <= lift_max + 1e-9:
            centre = base + normal * lift
            if not any(self._placement_touches(obj, centre, quat, tree) for tree in trees):
                return centre, quat
            lift += 0.002
        return base + normal * lift_max, quat

    @staticmethod
    def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
        a = np.asarray(axis, dtype=float)
        a = a / max(1e-12, float(np.linalg.norm(a)))
        k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
        return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)

    def palm_yaw(self, name: str | None = None) -> float:
        return float(self._palm_yaw.get(name or self.active_object, 0.0))

    def set_palm_yaw(self, name: str, yaw_rad: float, *, refit: bool = True) -> dict[str, Any] | None:
        """Rotate the palm object about the palm normal through its centre.

        The fingers holding it meet the rotated surface: blocks are refitted,
        so the sim hand and (with feedback on) the real glove follow.
        """
        obj = next((o for o in self.object_runtime if o.name == name), None)
        if obj is None or not self.palm_mode:
            return None
        yaw = math.remainder(float(yaw_rad), 2.0 * math.pi)
        if abs(yaw - self._palm_yaw.get(name, 0.0)) < 1e-9:
            return None
        previous = dict(self._blocked)
        self._palm_yaw[name] = yaw
        mujoco.mj_forward(self.model, self.data)
        pos, quat = self.palm_position(obj)
        obj.target_position = pos
        obj.target_quat = quat
        obj.spec.position = tuple(float(v) for v in pos)
        obj.spec.quat = tuple(float(v) for v in quat)
        self._lock_objects()
        mujoco.mj_forward(self.model, self.data)
        if not refit:
            return self.refit_request(previous)
        self.refit_contact_blocks(previous)
        return None

    # ------------------------------------------------------------------
    # Contact refit after the object changed (size / rotation / kind).
    #
    # The search needs ~10-20 closed-chain solves (~4 ms each).  It is split
    # so that only a snapshot and the final swap need the runner lock:
    #   req = sim.refit_request(previous)        # lock held, ~1 ms
    #   res = sim.refit_compute(req)             # NO lock: private solver/MjData
    #   sim.refit_apply(res)                     # lock held, ~1 ms
    # While it computes, the previous surfaces stay engaged, so the real
    # glove keeps its feedback instead of dropping to zero.
    def _fingers_touching(self, tree: np.ndarray, req: dict[str, Any] | None = None) -> dict[str, str]:
        """finger -> object it penetrates, with the glove at tree pose ``tree``."""
        if req is None:
            if self._placement_data is None:
                self._placement_data = mujoco.MjData(self.model)
            model, aux, qpos = self.model, self._placement_data, self.data.qpos
            tree_adr, glove = self._tree_qposadr, self.glove_geoms
            by_flex = {fid: o.name for fid, o in self._flex_objects().items()}
            by_geom = {g: o.name for g, o in self.object_geom_ids.items()}
        else:
            model, aux, qpos = req["model"], req["aux"], req["qpos"]
            tree_adr, glove = req["tree_qposadr"], req["glove_geoms"]
            by_flex, by_geom = req["by_flex"], req["by_geom"]
        aux.qpos[:] = qpos
        aux.qpos[tree_adr] = tree
        mujoco.mj_fwdPosition(model, aux)
        out: dict[str, str] = {}
        for i in range(aux.ncon):
            c = aux.contact[i]
            if float(c.dist) >= 0.0:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in glove:
                glove_gid, object_gid = g1, g2
            elif g2 in glove:
                glove_gid, object_gid = g2, g1
            else:
                continue
            flex_id = int(np.asarray(c.flex).reshape(-1)[1]) if np.ndim(c.flex) else -1
            name = by_flex.get(flex_id) or by_geom.get(object_gid)
            if name is not None:
                out.setdefault(glove[glove_gid], name)
        return out

    def _flex_objects(self) -> dict[int, ObjectRuntime]:
        out = {}
        for obj in self.object_runtime:
            fid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_FLEX, obj.flex_name)
            if fid >= 0:
                out[fid] = obj
        return out

    def refit_request(self, previous: dict[int, tuple[int, float, float]] | None = None
                      ) -> dict[str, Any] | None:
        """Snapshot for refit_compute (call with the runner lock held).

        Re-engages ``previous`` surfaces meanwhile (a rebuild cleared them).
        """
        self._surface_goal.clear()
        if not (self.collision_clamp and self.objects_locked) or not self.object_runtime:
            self._blocked.clear()
            self._invalidate_ghost()
            return None
        previous = dict(previous if previous is not None else self._blocked)
        if previous:
            self._blocked = dict(previous)
            self._refresh_visual_pose()
            self._invalidate_ghost(clear_blocks=False)
        if getattr(self, "_refit_kin", None) is None:
            self._refit_kin = GloveKinematics(
                self.kinematics.model_dir, self.mapping,
                fast=bool(self.config.get("glove", {}).get("fast_kinematics", True)))
        # Warm-start the private solver from the real-hand pose.
        src = self.kinematics._slots["target"]
        self._refit_kin._slots["placement"] = {
            "targets": dict(src["targets"]), "state": dict(src["state"]),
            "q": None if src["q"] is None else src["q"].copy()}
        if getattr(self, "_refit_model", None) is not self.model:
            self._refit_model = self.model
            self._refit_aux = mujoco.MjData(self.model)
        stiffness = {sp.name: self._proxy_stiffness(sp) for sp in self.specs}
        return {
            "model": self.model, "aux": self._refit_aux, "qpos": self.data.qpos.copy(),
            "target": np.asarray(self._motor_target, dtype=float).copy(),
            "tree_qposadr": self._tree_qposadr.copy(), "glove_geoms": dict(self.glove_geoms),
            "by_flex": {fid: o.name for fid, o in self._flex_objects().items()},
            "by_geom": {g: o.name for g, o in self.object_geom_ids.items()},
            "finger_motors": {f: list(v) for f, v in self._finger_motors.items()},
            "stiffness": stiffness,
            "step": float(self.config.get("glove", {}).get("refit_step_fraction", 0.05)),
        }

    def refit_compute(self, req: dict[str, Any] | None, *, iterations: int = 3) -> dict[str, Any] | None:
        """The expensive search; touches no live state (no lock needed)."""
        if req is None:
            return None
        kin = self._refit_kin
        model = req["model"]
        target = req["target"]
        motors = req["finger_motors"]

        def pose(q: np.ndarray) -> np.ndarray:
            return np.asarray(kin.full_pose(q, model, self.mapping, slot="placement"), dtype=float)
        try:
            tree = pose(target).copy()
            # Only fingers can be pushed open; the palm itself is not a motor.
            touching = {f: o for f, o in self._fingers_touching(tree, req).items() if f in motors}
            # Walk each touching finger from the real pose towards open in
            # small steps until it clears, then bisect inside the last step.
            lo: dict[str, float] = {}
            hi = {f: 1.0 for f in touching}
            remaining = set(touching)
            frac = 1.0
            while remaining and frac > 0.0:
                frac = max(0.0, frac - req["step"])
                q = target.copy()
                for f in touching:
                    q[motors[f]] = (frac if f in remaining else lo[f]) * target[motors[f]]
                hits = self._fingers_touching(pose(q), req)
                for f in list(remaining):
                    if f in hits:
                        hi[f] = frac
                    else:
                        lo[f] = frac
                        remaining.discard(f)
            # Still touching fully open (palm-fixed finger parts): it cannot be
            # pushed out of the object, so no spring for this finger.
            for f in remaining:
                touching.pop(f, None)
            for _ in range(iterations if touching else 0):
                q = target.copy()
                mid = {f: 0.5 * (lo[f] + hi[f]) for f in touching}
                for f in touching:
                    q[motors[f]] = mid[f] * target[motors[f]]
                hits = self._fingers_touching(pose(q), req)
                for f in touching:
                    if f in hits:
                        hi[f] = mid[f]
                    else:
                        lo[f] = mid[f]
        except KinematicsError:
            return None
        goals: dict[int, tuple[int, float, float]] = {}
        for f, obj_name in touching.items():
            k = float(req["stiffness"].get(obj_name, 0.0))
            for idx in motors[f]:
                tq = float(target[idx])
                if abs(tq) < 1e-6:
                    continue
                goals[idx] = (1 if tq > 0 else -1, lo[f] * tq, k)
        return {"model": model, "target": target, "tree": tree, "goals": goals}

    def refit_apply(self, res: dict[str, Any] | None) -> bool:
        """Install the refitted surfaces (runner lock held).  False if stale."""
        if res is None or res["model"] is not self.model:
            return False
        current = self._blocked
        target = np.asarray(self._motor_target, dtype=float)
        if np.array_equal(target, res["target"]):
            self._q_tree_target = np.asarray(res["tree"], dtype=float).copy()
        blocks: dict[int, tuple[int, float, float]] = {}
        self._surface_goal.clear()
        for idx, (sign, goal, k) in res["goals"].items():
            tq = float(target[idx])
            old = current.get(idx)
            start = old[1] if (old is not None and old[0] == sign) else tq
            # Never start "inside" the real finger (no instant push).
            if sign * (tq - start) < 0.0:
                start = tq
            blocks[idx] = (sign, float(start), float(k))
            if abs(goal - start) > 1e-9:
                self._surface_goal[idx] = float(goal)
        self._blocked = blocks
        self._surface_goal_t = time.monotonic()
        self._refresh_visual_pose()
        self._invalidate_ghost(clear_blocks=False)
        return True

    def refit_contact_blocks(self, previous: dict[int, tuple[int, float, float]] | None = None,
                             *, iterations: int = 3) -> None:
        """Re-derive the contact surfaces after the object changed size/pose/kind.

        For every finger whose real-hand (target) pose penetrates the object,
        walk from the real pose towards the open pose for the last
        non-penetrating angle: that is the new surface.  The held surface
        walks from where it was (``previous``; or the real finger angle for a
        new contact) to the new one at ``glove.surface_rate_rad_s``, so a
        growing ball first touches and then gradually pushes the hand open;
        with feedback on the driver spring pushes the real fingers the same
        way.  Fingers no longer touching are released.  Synchronous version
        (tests/headless); the UI uses request/compute/apply around the lock.
        """
        self.refit_apply(self.refit_compute(self.refit_request(previous), iterations=iterations))

    def _advance_surfaces(self) -> None:
        """Walk refitted contact surfaces towards their goal at a bounded rate."""
        now = time.monotonic()
        dt = min(0.1, max(0.0, now - self._surface_goal_t))
        self._surface_goal_t = now
        if not self._surface_goal:
            return
        rate = float(self.config.get("glove", {}).get("surface_rate_rad_s", 1.0))
        step = rate * dt
        for idx, goal in list(self._surface_goal.items()):
            block = self._blocked.get(idx)
            if block is None:
                del self._surface_goal[idx]
                continue
            sign, frozen, k = block
            new = frozen + float(np.clip(goal - frozen, -step, step))
            self._blocked[idx] = (sign, new, k)
            if abs(goal - new) < 1e-9:
                del self._surface_goal[idx]

    def place_on_palm(self) -> None:
        """Seat every scene object on the palm and lock it there."""
        if self._drag_name is not None:
            self.end_drag()
        mujoco.mj_forward(self.model, self.data)
        for obj in self.object_runtime:
            pos, quat = self.palm_position(obj)
            obj.target_position = pos
            obj.target_quat = quat
            obj.spec.position = tuple(float(v) for v in pos)
            obj.spec.quat = tuple(float(v) for v in quat)
        self.objects_locked = True
        self._lock_objects()
        self._invalidate_ghost()
        mujoco.mj_forward(self.model, self.data)
        self._cache_flex_rest()
