from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np

from .mapping import GloveMapping


class KinematicsError(RuntimeError):
    pass


class GloveKinematics:
    """Map 12 physical motor angles to the full 68-joint glove tree.

    The vE3.3 glove is a closed linkage; the extracted CAD solver is used for
    the kinematic pose instead of applying motor torques to a 68-DOF closed
    chain, which is numerically unstable in MuJoCo.  The result is a kinematic
    glove: contact forces are still solved by MuJoCo, but the glove pose comes
    from the real glove / demo input, exactly like the real exoskeleton.
    """

    def __init__(self, model_dir: str | Path, mapping: GloveMapping) -> None:
        self.model_dir = Path(model_dir)
        script = self.model_dir / "scripts" / "solve_closed_chain.py"
        if not script.is_file():
            raise KinematicsError(f"缺少闭环求解脚本: {script}")
        spec = importlib.util.spec_from_file_location("_ve33_closure_solver", script)
        if spec is None or spec.loader is None:
            raise KinematicsError("无法加载闭环求解脚本")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._solver = module.ClosureSolver(
            self.model_dir / "hand_tree_only.urdf",
            self.model_dir / "config" / "loop_closures.yaml",
        )
        self.independent: tuple[str, ...] = tuple(self._solver.independent)
        self.joint_names: tuple[str, ...] = tuple(self._solver.independent) + tuple(self._solver.dependent)
        missing = [m.model_joint for m in mapping.motors if m.model_joint not in self.independent]
        if missing:
            raise KinematicsError(f"映射中的电机关节不在闭环独立关节中: {missing}")
        self._targets = {name: 0.0 for name in self.independent}
        self._q = np.zeros(len(self.joint_names), dtype=float)
        self._state: dict[str, float] = {name: 0.0 for name in self.joint_names}
        self._derivative: np.ndarray | None = None
        self._derivative_key: tuple[float, ...] | None = None
        self.last_solve_ms = 0.0
        self.last_gap_m = 0.0
        self._last_time = 0.0
        # Solver columns are independent joints in the CAD order; expose the
        # mapping to physical motor IDs J1..J12.
        self.motor_columns = np.asarray(
            [self.independent.index(m.model_joint) for m in mapping.motors], dtype=int
        )
        self.dependent_groups = {key: tuple(value) for key, value in self._solver.groups.items()}
        self._motor_by_group: dict[str, list[str]] = {}
        for key in self.dependent_groups:
            self._motor_by_group[key] = [m.model_joint for m in mapping.motors
                                         if m.model_joint.split("_")[0] == key]
        self._motor_model_joints = {m.motor_id: m.model_joint for m in mapping.motors}
        # Derivative columns are physical motor IDs J1..J12, not CAD solver order.
        self._motor_column = {m.model_joint: m.motor_id - 1 for m in mapping.motors}

    def full_pose(self, motor_rad: np.ndarray, model, mapping: GloveMapping) -> np.ndarray:
        new_targets = dict(self._targets)
        for motor in mapping.motors:
            new_targets[motor.model_joint] = float(motor_rad[motor.motor_id - 1])
        t0 = time.perf_counter()
        pose = self._solver.solve(new_targets)
        self.last_solve_ms = (time.perf_counter() - t0) * 1e3
        self.last_gap_m = float(self._solver.max_gap(pose))
        self._targets = new_targets
        self._state = {name: float(pose[name]) for name in self.joint_names}
        self._q = np.asarray([self._state[name] for name in self.joint_names], dtype=float)
        self._derivative_key = None
        return self._q

    def tree_derivative(self) -> np.ndarray:
        """d(full 68-joint tree pose) / d(12 physical motor angles).

        The vE3.3 glove is a closed linkage.  MuJoCo's mj_jac sees only the
        tree, so forces on distal links would otherwise project to zero motor
        torque.  This derivative is obtained from the CAD closure constraints
        with the implicit function theorem.  Rows are ``joint_names``; columns
        are physical motor IDs J1..J12.
        """
        key = tuple(round(float(self._state[m]), 9) for m in self._solver.independent)
        if self._derivative is not None and self._derivative_key == key:
            return self._derivative
        derivative = np.zeros((len(self.joint_names), 12), dtype=float)
        motor_col = dict(self._motor_column)

        def residual_and_jac(state: dict[str, float], group: str):
            return self._solver._group_residual_jac(state, group)

        for group, deps in self.dependent_groups.items():
            state = dict(self._state)
            residual0, jac = residual_and_jac(state, group)
            if jac.size == 0:
                continue
            motors = [m for m in self._motor_by_group.get(group, []) if m in motor_col]
            if not motors:
                continue
            cols = np.array([motor_col[m] for m in motors], dtype=int)
            base_q = self._state
            # The residual is a function of the full 68-joint tree.  Perturb
            # one motor joint at a time and difference only its finger block.
            b = np.zeros((residual0.size, len(motors)), dtype=float)
            eps = 1e-6
            for k, motor in enumerate(motors):
                perturbed = dict(base_q)
                perturbed[motor] = float(perturbed.get(motor, 0.0) + eps)
                r, _ = residual_and_jac(perturbed, group)
                b[:, k] = (r - residual0) / eps
            # Solve A * dq_dep/du + B = 0.
            try:
                dq_dep = np.linalg.lstsq(jac, -b, rcond=1e-9)[0]
            except np.linalg.LinAlgError:
                continue
            dep_indices = [self.joint_names.index(name) for name in deps]
            derivative[np.ix_(dep_indices, cols)] = dq_dep
            for k, motor in enumerate(motors):
                derivative[self.joint_names.index(motor), cols[k]] = 1.0
        derivative.setflags(write=False)
        self._derivative = derivative
        self._derivative_key = key
        return derivative

    def apply(self, model, data, q_tree: np.ndarray) -> None:
        import mujoco

        for name, value in zip(self.joint_names, q_tree):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                continue
            data.qpos[model.jnt_qposadr[jid]] = value
            data.qvel[model.jnt_dofadr[jid]] = 0.0
