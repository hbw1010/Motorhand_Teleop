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
        self.specs = list(specs)
        # Objects are pose-locked by default (no drift).  Unlocking lets the
        # kinematic glove push/grasp them; the UI exposes this as a checkbox.
        self.objects_locked = bool(config.get("objects", {}).get("locked_default", True))
        self._drag_name: str | None = None
        self._drag_target_pos: np.ndarray | None = None
        self._drag_target_quat: np.ndarray | None = None
        self.model: mujoco.MjModel
        self.data: mujoco.MjData
        model_dir = Path(config["paths"]["glove_model"]).parent
        self.kinematics = GloveKinematics(model_dir, mapping)
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
        self._blocked: dict[int, tuple[int, float]] = {}
        self._ghost_data = mujoco.MjData(self.model)
        self._ghost_contacts: list[ContactInfo] | None = None
        self._ghost_finger_force = {k: 0.0 for k in CONTACT_GROUPS}
        self._ghost_finger_pen = {k: 0.0 for k in CONTACT_GROUPS}
        self._ghost_raw = np.zeros(12, dtype=float)
        self._ghost_ready = False
        self._ghost_dirty = False

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
    def _compile(self) -> None:
        self.model, self.data, self.xml = compile_scene(self.config, self.mapping, self.specs)
        self.motor_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in self.mapping.model_joint_names]
        if any(i < 0 for i in self.motor_ids):
            raise RuntimeError(f"模型缺少电机关节: {self.mapping.model_joint_names}")
        self.motor_qposadr = np.asarray([self.model.jnt_qposadr[i] for i in self.motor_ids], dtype=int)
        self.motor_dofadr = np.asarray([self.model.jnt_dofadr[i] for i in self.motor_ids], dtype=int)
        self.glove_geoms: dict[int, str] = {}
        for gid in range(self.model.ngeom):
            name = self.model.geom(gid).name
            if name.startswith("col_"):
                parts = name.split("_", 2)
                if len(parts) >= 2:
                    self.glove_geoms[gid] = parts[1]
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
            for idx, (sign, frozen) in list(self._blocked.items()):
                if abs(delta[idx]) < 1e-12:
                    cmd[idx] = frozen
                elif int(np.sign(delta[idx])) == sign:
                    cmd[idx] = frozen
                else:
                    # The wearer is opening/backing off this axis.  Allow the
                    # retreat but keep the block until the contact disappears,
                    # otherwise a still-touching finger gets re-blocked while
                    # opening and becomes stuck.
                    cmd[idx] = target[idx]
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
            elif np.array_equal(cmd, self._visual_motor_rad):
                # Contact clamp held the visible pose; no second CAD solve.
                pose = self._q_tree
            else:
                pose = self.kinematics.full_pose(cmd, self.model, self.mapping)
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
        for idx, (sign, frozen) in self._blocked.items():
            delta = self._motor_target[idx] - self._prev_motor_target[idx]
            if abs(delta) < 1e-12 or int(np.sign(delta)) == sign:
                cmd[idx] = frozen
        if np.array_equal(cmd, self._visual_motor_rad):
            return
        try:
            pose = self.kinematics.full_pose(cmd, self.model, self.mapping)
        except KinematicsError:
            return
        self._motor_rad = cmd
        self._q_tree = pose
        self._visual_motor_rad = cmd.copy()
        self._apply_pose()

    def _update_contact_blocks(self, contacts: list[ContactInfo]) -> None:
        """Clamp only fingers whose *target* pose touches a locked object."""
        if not self.collision_clamp or not self.objects_locked:
            self._blocked.clear()
            return
        soft_allow = float(self.config.get("glove", {}).get("clamp_soft_allow_m", 0.004))
        active: set[str] = set()
        for contact in contacts:
            if contact.object_name == self._drag_name:
                continue
            spec = next((s for s in self.specs if s.name == contact.object_name), None)
            allowed = soft_allow if (spec is not None and spec.deformable) else 0.0
            if contact.penetration_m > allowed:
                active.add(contact.finger)
        for idx in list(self._blocked):
            if MOTOR_FINGER[idx] not in active:
                del self._blocked[idx]
        for finger in active:
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
                )

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
        if self._use_ghost_feedback():
            contacts = self._ghost_contacts or []
            finger_force = dict(self._ghost_finger_force)
            finger_pen = dict(self._ghost_finger_pen)
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
        """Restore every object to its configured start pose."""
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

    def set_object_shape(self, name: str, *, size_m: float | None = None,
                         stiffness: float | None = None) -> None:
        """Rebuild the scene with an updated object parameter.

        The generated scene compiles in ~0.1 s on the target machine; the UI
        throttles this to ``objects.rebuild_hz`` (default 10 Hz).
        """
        for spec in self.specs:
            if spec.name == name:
                if size_m is not None:
                    spec.size_m = float(size_m)
                if stiffness is not None:
                    spec.stiffness = float(stiffness)
                break
        else:
            raise KeyError(name)
        old_q = {self.model.joint(i).name: float(self.data.qpos[self.model.jnt_qposadr[i]])
                 for i in range(self.model.njnt)}
        old_dq = {self.model.joint(i).name: float(self.data.qvel[self.model.jnt_dofadr[i]])
                  for i in range(self.model.njnt)}
        old_poses = {o.name: (o.target_position.copy(), o.target_quat.copy()) for o in self.object_runtime}
        self._compile()
        for i in range(self.model.njnt):
            name = self.model.joint(i).name
            if name in old_q:
                self.data.qpos[self.model.jnt_qposadr[i]] = old_q[name]
                self.data.qvel[self.model.jnt_dofadr[i]] = old_dq[name]
        for obj in self.object_runtime:
            if obj.name in old_poses:
                obj.target_position, obj.target_quat = old_poses[obj.name]
        self._lock_objects()
        self._apply_pose()
        mujoco.mj_forward(self.model, self.data)
        self._cache_flex_rest()
        # The compiled model/data changed; the ghost evaluator must follow it.
        self._ghost_data = mujoco.MjData(self.model)
        self._q_tree_target = self._q_tree.copy()
        self._ghost_ready = False
        self._invalidate_ghost()
