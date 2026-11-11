"""Contact virtual work and fail-safe limits in model motor coordinates.

The projector returns nominal motor torques in N m, not current or a hardware
command.  Positive torque follows the positive model motor-angle convention.
Hardware calibration, output authorization and command deadlines are separate.
"""
from __future__ import annotations

from collections.abc import Mapping
import time

import mujoco
import numpy as np


def _vector(value, size, label):
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError(label)
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain finite real values") from exc
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite {size}-vector")
    return result


def _named_vector(values, names, label):
    if not isinstance(values, Mapping) or set(values) != set(names):
        raise ValueError(f"{label} must contain exactly the model names")
    return _vector([values[name] for name in names], len(names), label)


class ForceProjector:
    """Map forces *on the glove* at world contact points to motor N m.

    For virtual motor displacement dm, dq_tree = D dm and dx = J dq_tree.
    Thus F_on_glove . dx = (D.T J.T F_on_glove) . dm.  This is the
    environment's torque on the glove, with no extra sign inversion.  D is
    obtained from the vE3.3 implicit closure derivative, including passive
    joints; singular or inconsistent closures are rejected, never regularized.

    ``max_compute_s`` bounds computation age, not input age.  The caller must
    independently enforce a source timestamp lease immediately before output.
    """

    def __init__(self, glove_model, mj_model, *, max_compute_s=0.05):
        self.kinematics = glove_model.kinematics
        self.motor_names = tuple(glove_model.motor_names)
        self.tree_names = tuple(self.kinematics.tree_names)
        if (len(self.motor_names) != 12 or len(set(self.motor_names)) != 12
                or len(set(self.tree_names)) != len(self.tree_names)
                or set(self.motor_names) != set(self.kinematics.motor_names)):
            raise ValueError("invalid glove motor or tree names")
        self.model = mj_model
        self.max_compute_s = float(max_compute_s)
        if not np.isfinite(self.max_compute_s) or self.max_compute_s <= 0:
            raise ValueError("max_compute_s must be finite and positive")
        self._motor_rows = np.array([self.tree_names.index(n) for n in self.motor_names])
        joint_ids = []
        for name in self.tree_names:
            joint = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0 or mj_model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"missing scalar hinge for glove tree joint {name}")
            joint_ids.append(joint)
        self._qpos = np.asarray(mj_model.jnt_qposadr[joint_ids], dtype=int)
        self._dofs = np.asarray(mj_model.jnt_dofadr[joint_ids], dtype=int)

    def project(self, data, contacts, motor_positions, tree_positions):
        started = time.monotonic()
        motors = _named_vector(motor_positions, self.motor_names, "motor positions")
        tree = _named_vector(tree_positions, self.tree_names, "tree positions")
        if not np.allclose(tree[self._motor_rows], motors, rtol=0, atol=1e-7):
            raise ValueError("motor positions do not match the tree")
        if not np.isfinite(data.qpos[self._qpos]).all() or not np.allclose(
                data.qpos[self._qpos], tree, rtol=0, atol=1e-7):
            raise ValueError("MuJoCo qpos does not match the glove tree")
        if not contacts:
            return np.zeros(12, dtype=np.float64)

        # evaluate_tree checks closure residuals, branch bounds and differential
        # rank. Its cached derivative is d(tree)/d(independent joint angle).
        pose = self.kinematics.evaluate_tree(tree_positions)
        derivative = self.kinematics._tree_derivatives(pose.all_positions)
        if derivative.shape != (len(self.tree_names), 12) or not np.isfinite(derivative).all():
            raise ValueError("invalid closed-chain derivative")
        motor_derivative = derivative[self._motor_rows, :]
        singular = np.linalg.svd(motor_derivative, compute_uv=False)
        if singular[-1] <= singular[0] * 1e-9 or singular[0] == 0:
            raise ValueError("motor-space closed-chain derivative is singular")
        # D = derivative @ inverse(motor_derivative), without explicit inversion.
        try:
            tree_per_motor = np.linalg.solve(motor_derivative.T, derivative.T).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("motor-space closed-chain derivative unavailable") from exc
        tree_force = np.zeros(len(self.tree_names), dtype=np.float64)
        jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        for contact in contacts:
            if not isinstance(contact, Mapping):
                raise ValueError("contact must be a mapping")
            body = contact.get("body_id")
            if isinstance(body, (bool, np.bool_)) or not isinstance(body, (int, np.integer)) or not 0 < body < self.model.nbody:
                raise ValueError("contact body_id must identify a non-world body")
            point = _vector(contact.get("point_world"), 3, "contact point")
            force = _vector(contact.get("force_world_on_glove"), 3, "contact force")
            mujoco.mj_jac(self.model, data, jacobian, None, point, int(body))
            if not np.isfinite(jacobian).all():
                raise ValueError("non-finite contact Jacobian")
            tree_force += jacobian[:, self._dofs].T @ force
        torque = tree_per_motor.T @ tree_force
        if not np.isfinite(torque).all():
            raise ValueError("non-finite projected torque")
        if time.monotonic() - started > self.max_compute_s:
            raise TimeoutError("contact projection exceeded its computation deadline")
        return torque


class FeedbackLimiter:
    """Per-axis N m clamp and N m/s slew limit, with immediate fail-safe zero.

    Limits are explicit model-coordinate values; they are not hardware ratings.
    No torque is allowed on invalid data or a stale/invalid time interval.
    ``last_error`` reports why the most recent call returned a fail-safe zero.
    """

    def __init__(self, max_torque_nm, max_slew_nm_per_s, *, max_dt_s=0.1):
        self.max_torque_nm = _vector(max_torque_nm, 12, "torque limits").copy()
        self.max_slew_nm_per_s = _vector(max_slew_nm_per_s, 12, "slew limits").copy()
        self.max_dt_s = float(max_dt_s)
        if (np.any(self.max_torque_nm < 0) or np.any(self.max_slew_nm_per_s < 0)
                or not np.isfinite(self.max_dt_s) or self.max_dt_s <= 0):
            raise ValueError("limits must be nonnegative and max_dt_s positive")
        self._previous = np.zeros(12, dtype=np.float64)
        self.last_error = None

    def reset(self):
        """Immediately discard the previous output, e.g. on a watchdog trip."""
        self._previous.fill(0.0)
        self.last_error = None

    def apply(self, torque_nm, dt_s):
        try:
            requested = _vector(torque_nm, 12, "torque")
            if np.iscomplexobj(dt_s) or np.ndim(dt_s) != 0:
                raise ValueError("dt_s must be a finite positive scalar")
            dt = float(dt_s)
            if not np.isfinite(dt) or dt <= 0 or dt > self.max_dt_s:
                raise ValueError("dt_s is invalid or exceeds the freshness limit")
            bounded = np.clip(requested, -self.max_torque_nm, self.max_torque_nm)
            change = self.max_slew_nm_per_s * dt
            output = np.clip(bounded, self._previous - change, self._previous + change)
            if not np.isfinite(output).all():
                raise ValueError("limited torque is non-finite")
        except (TypeError, ValueError, OverflowError) as exc:
            self._previous.fill(0.0)
            self.last_error = str(exc)
            return self._previous.copy()
        self._previous = output.copy()
        self.last_error = None
        return output
