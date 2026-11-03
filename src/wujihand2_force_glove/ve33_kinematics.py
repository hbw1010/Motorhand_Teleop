"""vE.3.3 真实相邻指节坐标及闭环逆运动学（rad / m）。

保留模型原有树和闭环残差：指节角是树路径上转角的有向和，不是
output_joint 的别名。这里只提供运动学结果，未标定实物电机限位。
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path

import numpy as np

from .ve33_geometry import Ve33ClosureGeometry


@dataclass(frozen=True)
class Ve33Pose:
    """一次完整、闭环有效的姿态快照；字典单位均为弧度。"""

    joint_positions: dict[str, float]
    motor_positions: dict[str, float]
    all_positions: dict[str, float]
    dependent_positions: dict[str, float]
    max_gap_m: float


@dataclass(frozen=True)
class _Group:
    name: str
    names: tuple[str, ...]
    indices: np.ndarray
    input_indices: np.ndarray
    passive_count: int
    passive_map: np.ndarray
    input_map: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


class Ve33Kinematics:
    """以12个真实指节角驱动原CAD装配分支，求出12电机和56从属树角。

    evaluate 不提交，solve 仅在全部连续步进成功后提交。模型的装配边界
    不等于实物限位；越界、闭环不收敛或隐函数奇异均抛出 ValueError。
    """

    def __init__(self, hand: str, model_dir: str | Path | None = None):
        if hand not in ("left", "right"):
            raise ValueError("hand必须为left或right")
        self.hand = hand
        side = "L" if hand == "left" else "R"
        self.model_dir = (
            Path(model_dir).expanduser().resolve()
            if model_dir is not None
            else Path(__file__).resolve().parents[2]
            / "assets" / "hand_motor_ve33" / f"hand_motor_ve33_{side}"
        )
        with (self.model_dir / "config/joint_space.json").open(encoding="utf-8") as stream:
            config = json.load(stream)
        if config.get("side") != side:
            raise ValueError("指节配置侧别与hand不一致")
        definitions = config["independent"]
        self._dependent_definitions = tuple(config["dependent"])
        self.joint_names = tuple(item["name"] for item in definitions)
        self.motor_names = tuple(item["motor"] for item in definitions)
        if len(set(self.joint_names)) != 12 or len(set(self.motor_names)) != 12:
            raise ValueError("模型必须定义12个独立指节与12个电机")
        spec = importlib.util.spec_from_file_location(
            f"_ve33_closure_{side}", self.model_dir / "scripts/solve_closed_chain.py"
        )
        if spec is None or spec.loader is None:
            raise ValueError("无法加载模型闭环求解脚本")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._solver = module.ClosureSolver(
            self.model_dir / "hand_tree_only.urdf",
            self.model_dir / "config/loop_closures.yaml",
        )
        self._fk = self._solver.fk
        self._tree_names = tuple(self._solver.independent + self._solver.dependent)
        if len(set(self._tree_names)) != 68 or set(self.motor_names) != set(self._solver.independent):
            raise ValueError("模型树坐标与12电机、56从属角协议不一致")
        self._tree_index = {name: index for index, name in enumerate(self._tree_names)}
        self._motor_indices = np.array([self._tree_index[name] for name in self.motor_names])
        self._links = tuple(dict.fromkeys(
            link for joint in self._fk.j.values() for link in (joint[2], joint[1])
        ))
        self._ancestors = {link: self._ancestor_joints(link) for link in self._links}
        self._geometry = Ve33ClosureGeometry(self._solver, self._tree_names)
        zero_frames = self._frames({})
        rows, public_definitions = [], []
        for item in (*definitions, *self._dependent_definitions):
            row, axis_local = self._angle_row(item, zero_frames)
            rows.append(row)
            public_definitions.append({**item, "axis_local": axis_local.tolist()})
        self.joint_definitions = tuple(public_definitions[:12])
        self._angles = np.stack(rows)
        self._groups = self._prepare_groups()
        self._continuation_inputs = np.ones(12, dtype=bool)
        self._branch_tree_mask = np.ones(68, dtype=bool)
        for group in self._groups:
            witnesses = self._solver._blocks[group.name][0]
            for column, input_index in enumerate(group.input_indices):
                driven = np.flatnonzero(np.abs(group.input_map[:, column]) > 1e-12)
                if len(driven) != 1 or np.any(np.abs(group.passive_map[driven[0]]) > 1e-12):
                    continue
                name = group.names[driven[0]]
                # 两端见证点共享的直驱祖先只刚性旋转整个闭环，不改变装配分支。
                if all(
                    (name in self._ancestors[item[0]]) == (name in self._ancestors[item[2]])
                    for item in witnesses
                ):
                    self._continuation_inputs[input_index] = False
                    self._branch_tree_mask[group.indices[driven[0]]] = False
        self._derivative_cache = None
        self._tree_children = tuple(self._fk.j[name][1] for name in self._tree_names)
        self._tree_axes = np.array([self._fk.j[name][4] for name in self._tree_names], dtype=float)
        self._tree_axes /= np.linalg.norm(self._tree_axes, axis=1)[:, None]
        self._tree_prismatic = np.array([self._fk.j[name][0] == "prismatic" for name in self._tree_names])
        self._has_prismatic = bool(self._tree_prismatic.any())
        self._point_indices = {
            link: np.array([self._tree_index[name] for name in ancestors if name in self._tree_index], dtype=int)
            for link, ancestors in self._ancestors.items()
        }
        self.current = self._pose(np.zeros(68))
        self._tree_derivatives(self.current.all_positions)

    def _ancestor_joints(self, link: str) -> tuple[str, ...]:
        result = []
        while link in self._fk.cj:
            name = self._fk.cj[link]
            result.append(name)
            link = self._fk.j[name][2]
        return tuple(result)

    def _frames(self, state: Mapping[str, float]) -> dict[str, np.ndarray]:
        return self._geometry.world_frames(state)

    def _angle_row(self, definition, frames):
        parent, child = definition["parent"], definition["child"]
        axis_joint = self._fk.j[definition["axis_joint"]]
        axis = frames[axis_joint[1]][:3, :3] @ axis_joint[4]
        axis /= np.linalg.norm(axis)
        alignment = float(axis @ frames[child][:3, 2])
        if abs(alignment) < 1e-8:
            raise ValueError(f"{definition['name']}的物理轴无法由child参考+Z定向")
        axis *= 1.0 if alignment > 0 else -1.0
        row = np.zeros(len(self._tree_names))
        parent_path, child_path = set(self._ancestors[parent]), set(self._ancestors[child])
        # 共同祖先运动在相邻刚体的相对旋转中消去；剩余轴必须共线。
        for name in parent_path ^ child_path:
            kind, link, _, _, local_axis = self._fk.j[name]
            if kind == "fixed":
                continue
            if kind not in ("revolute", "continuous"):
                raise ValueError("真实指节路径只能包含旋转关节")
            world_axis = frames[link][:3, :3] @ local_axis
            world_axis /= np.linalg.norm(world_axis)
            dot = float(axis @ world_axis)
            if abs(abs(dot) - 1.0) > 1e-8:
                raise ValueError(f"{definition['name']}的树路径轴不共线，不能线性消元")
            sign = 1.0 if dot > 0 else -1.0
            row[self._tree_index[name]] = sign if name in child_path else -sign
        return row, frames[child][:3, :3].T @ axis

    def _prepare_groups(self) -> tuple[_Group, ...]:
        groups = []
        original_groups = self._solver.groups
        expanded_groups = {}
        for group, passive in original_groups.items():
            motors = tuple(name for name in self.motor_names if name.split("_")[0] == group)
            names = tuple(passive) + motors
            indices = np.array([self._tree_index[name] for name in names])
            inputs = np.array([i for i, name in enumerate(self.motor_names) if name in motors])
            matrix = self._angles[inputs]
            outside = np.ones(68, dtype=bool)
            outside[indices] = False
            if np.any(matrix[:, outside]):
                raise ValueError(f"{group}指节输入跨越独立闭环分组")
            matrix = matrix[:, indices]
            count = len(passive)
            try:
                inverse = np.linalg.solve(matrix[:, count:], np.eye(len(motors)))
            except np.linalg.LinAlgError as error:
                raise ValueError(f"{group}指节到电机的线性约束不可逆") from error
            # q_group = passive_map @ 从属角 + input_map @ 真实指节角。
            passive_map = np.vstack((np.eye(count), -inverse @ matrix[:, :count]))
            input_map = np.vstack((np.zeros((count, len(inputs))), inverse))
            lower, upper = np.array([
                self._solver.bounds.get(name, (-np.inf, np.inf)) for name in passive
            ]).T
            groups.append(_Group(
                group, names, indices, inputs, count, passive_map, input_map, lower, upper
            ))
            # 扩展此实例的解析雅可比列，不改旧ClosureSolver实现或全局模块。
            expanded_groups[group] = list(names)
            witnesses = self._solver._blocks[group][0]
            masks = [np.array([
                [name in self._ancestors[witness[side]] for name in names]
                for witness in witnesses
            ]) for side in (0, 2)]
            self._solver._blocks[group] = (witnesses, *masks)
        self._solver.groups = expanded_groups
        if {name for group in groups for name in group.names} != set(self._tree_names):
            raise ValueError("闭环分组没有覆盖全部树关节")
        return tuple(groups)

    def _pose(self, values: np.ndarray) -> Ve33Pose:
        state = dict(zip(self._tree_names, map(float, values)))
        gap = self._geometry.max_gap(state)
        if not np.isfinite(gap) or gap > self._solver.tolerance:
            raise ValueError(f"闭环残差未收敛（最大间隙{gap:.6g}米），保留原姿态")
        angles = self._angles @ values
        return Ve33Pose(
            dict(zip(self.joint_names, map(float, angles[:12]))),
            {name: state[name] for name in self.motor_names},
            state,
            {item["name"]: float(value) for item, value in zip(self._dependent_definitions, angles[12:])},
            gap,
        )

    def _state_vector(self, pose: Ve33Pose) -> np.ndarray:
        if not isinstance(pose, Ve33Pose) or set(pose.all_positions) != set(self._tree_names):
            raise ValueError("姿态必须包含本模型全部68个树关节")
        try:
            values = np.array([pose.all_positions[name] for name in self._tree_names], dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("姿态树关节必须是有限实数") from error
        if values.shape != (68,) or not np.isfinite(values).all():
            raise ValueError("姿态树关节必须是有限实数")
        for name, (lower, upper) in self._solver.bounds.items():
            if name in self._tree_index and not lower <= values[self._tree_index[name]] <= upper:
                raise ValueError(f"姿态超出{name}的原装配分支边界")
        return values

    @property
    def tree_names(self) -> tuple[str, ...]:
        return self._tree_names

    def evaluate_tree(self, positions: Mapping[str, float]) -> Ve33Pose:
        """重验上游完整树姿态并派生真实指节/电机，不重复逆解、不提交。"""
        if not isinstance(positions, Mapping) or set(positions) != set(self._tree_names):
            raise ValueError("完整姿态必须覆盖本模型全部68个树关节")
        if any(np.iscomplexobj(value) or np.ndim(value) != 0 for value in positions.values()):
            raise ValueError("姿态树关节必须是有限实数")
        try:
            values = np.array([positions[name] for name in self._tree_names], dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("姿态树关节必须是有限实数") from error
        if not np.isfinite(values).all():
            raise ValueError("姿态树关节必须是有限实数")
        result = self._pose(values)
        self._state_vector(result)
        self._tree_derivatives(result.all_positions)
        return result

    def evaluate(
        self, targets: Mapping[str, float], *, seed: Ve33Pose | None = None, local: bool = False
    ) -> Ve33Pose:
        """纯逆解；隐函数预测和原0.2rad树分支约束自适应细分，local仅校正邻域。"""
        if not isinstance(targets, Mapping):
            raise ValueError("指节目标必须为名称到弧度的映射")
        unknown = set(targets) - set(self.joint_names)
        if unknown:
            raise ValueError(f"未知独立指节：{', '.join(map(str, unknown))}")
        previous = self._state_vector(self.current if seed is None else seed)
        start = self._angles[:12] @ previous
        target = start.copy()
        for index, name in enumerate(self.joint_names):
            if name not in targets:
                continue
            value = targets[name]
            if np.iscomplexobj(value) or np.ndim(value) != 0:
                raise ValueError(f"{name}目标必须为有限实数弧度")
            try:
                target[index] = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{name}目标必须为有限实数弧度") from error
        if not np.isfinite(target).all():
            raise ValueError("指节目标必须为有限实数弧度")
        distance = float(np.max(np.abs(target - start)))
        if distance > 4.8:
            raise ValueError("单次指节变化过大，请分段输入；保留原姿态")
        coupled_distance = float(np.max(np.abs(target - start)[self._continuation_inputs], initial=0.0))
        if local and coupled_distance > 0.04 + 1e-12:
            raise ValueError("局部试算的闭链耦合指节须在种子姿态0.04rad邻域内")
        changed = np.flatnonzero(target != start)
        active = tuple(group for group in self._groups if np.intersect1d(group.input_indices, changed).size)
        previous = self._advance(previous, target, active, local=local)
        result = self._pose(previous)
        self._tree_derivatives(result.all_positions)
        return result

    def _advance(self, previous, requested, groups, depth=0, *, local=False, motor_space=False):
        candidate = previous.copy()
        jump = 0.0
        converged = True
        predicted = previous
        if groups:
            derivative = self._tree_derivatives(dict(zip(self._tree_names, previous)))
            if motor_space:
                old_inputs = previous[self._motor_indices]
                try:
                    derivative = np.linalg.solve(derivative[self._motor_indices].T, derivative.T).T
                except np.linalg.LinAlgError as error:
                    raise ValueError("电机正解处于奇异位置，无法确定连续装配分支") from error
            else:
                old_inputs = self._angles[:12] @ previous
            predicted = previous + derivative @ (requested - old_inputs)
            predicted_jump = np.max(np.abs(predicted - previous)[self._branch_tree_mask], initial=0.0)
            # 先按原分支跳变量自适应缩步，避免固定0.02rad过采样或无效的大步Newton。
            if not local and predicted_jump > 0.2 and depth < 8:
                middle = self._advance(
                    previous, (old_inputs + requested) / 2, groups, depth + 1, motor_space=motor_space
                )
                return self._advance(middle, requested, groups, depth + 1, motor_space=motor_space)
        for group in groups:
            initial = np.clip(predicted[group.indices[:group.passive_count]], group.lower, group.upper)
            driven = (
                requested[group.input_indices]
                if motor_space else group.input_map @ requested[group.input_indices]
            )
            last_x = last_result = None

            def evaluate(x):
                nonlocal last_x, last_result
                if last_x is None or not np.array_equal(x, last_x):
                    values = np.concatenate((x, driven)) if motor_space else group.passive_map @ x + driven
                    state = dict(zip(group.names, values))
                    residual, jacobian = self._geometry.group_residual_jac(state, group.name)
                    last_x = x.copy()
                    last_result = (
                        residual,
                        jacobian[:, :group.passive_count] if motor_space else jacobian @ group.passive_map,
                    )
                return last_result

            # 每段已有隐函数预测；用满列秩Newton校正，不逐指创建通用优化器。
            corrected = initial
            residual_tolerance = min(1e-11, self._solver.tolerance / np.sqrt(3.0))
            for _ in range(6):
                residual, jacobian = evaluate(corrected)
                if np.max(np.abs(residual), initial=0.0) <= residual_tolerance:
                    break
                try:
                    step, _, rank, singular = np.linalg.lstsq(jacobian, -residual, rcond=1e-9)
                except np.linalg.LinAlgError:
                    break
                if rank != group.passive_count or singular[-1] <= singular[0] * 1e-9:
                    break
                positive = step > 0.0
                negative = step < 0.0
                alpha = min(
                    1.0,
                    float(np.min(
                        (group.upper[positive] - corrected[positive]) / step[positive], initial=1.0
                    )),
                    float(np.min(
                        (group.lower[negative] - corrected[negative]) / step[negative], initial=1.0
                    )),
                )
                if alpha <= 0.0:
                    break
                cost = residual @ residual
                for _ in range(8):
                    trial = np.clip(corrected + alpha * step, group.lower, group.upper)
                    trial_residual, _ = evaluate(trial)
                    if trial_residual @ trial_residual < cost:
                        corrected = trial
                        break
                    alpha *= 0.5
                else:
                    break
            if np.max(np.abs(evaluate(corrected)[0]), initial=0.0) > residual_tolerance:
                if local:
                    raise ValueError("局部闭环校正不收敛或超出原装配边界，保留原姿态")
                # 全局大步未校正成功，沿既有装配路径二分；不把残差当有效姿态。
                converged = False
                break
            values = np.concatenate((corrected, driven)) if motor_space else group.passive_map @ corrected + driven
            candidate[group.indices] = values
            branch_delta = np.abs(values - previous[group.indices])[self._branch_tree_mask[group.indices]]
            jump = max(jump, float(np.max(branch_delta, initial=0.0)))
        if converged:
            state = dict(zip(self._tree_names, candidate))
            gap = self._geometry.max_gap(state)
            bounded = all(
                lower <= state[name] <= upper for name, (lower, upper) in self._solver.bounds.items()
                if name in state
            )
            if bounded and np.isfinite(gap) and gap <= self._solver.tolerance and jump <= 0.2:
                self._tree_derivatives(state)
                return candidate
        old_inputs = previous[self._motor_indices] if motor_space else self._angles[:12] @ previous
        if local or depth >= 8 or np.max(np.abs(requested - old_inputs)) < 1e-5:
            raise ValueError("目标超出当前闭环装配分支或残差未收敛，保留原姿态")
        midpoint = (old_inputs + requested) / 2
        middle = self._advance(previous, midpoint, groups, depth + 1, motor_space=motor_space)
        return self._advance(middle, requested, groups, depth + 1, motor_space=motor_space)

    def _tree_derivatives(self, state: Mapping[str, float]) -> np.ndarray:
        """隐函数求 dq/du；过定约束用满列秩最小二乘，不做阻尼伪回退。"""
        key = tuple(state[name] for name in self._tree_names)
        cached = self._derivative_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        derivative = np.zeros((68, 12))
        for group in self._groups:
            _, jacobian = self._geometry.group_residual_jac(state, group.name)
            passive_jac = jacobian @ group.passive_map
            input_jac = jacobian @ group.input_map
            try:
                response, _, rank, singular = np.linalg.lstsq(passive_jac, -input_jac, rcond=1e-9)
            except np.linalg.LinAlgError as error:
                raise ValueError(f"{group.name}闭环微分求解失败，无法确定真实指节导数") from error
            if rank != group.passive_count or singular[-1] <= singular[0] * 1e-9:
                raise ValueError(f"{group.name}闭环处于奇异位置，无法确定真实指节导数")
            if np.max(np.abs(passive_jac @ response + input_jac)) > 1e-7:
                raise ValueError(f"{group.name}闭环微分约束不相容")
            derivative[np.ix_(group.indices, group.input_indices)] = (
                group.passive_map @ response + group.input_map
            )
        derivative.setflags(write=False)
        self._derivative_cache = (key, derivative)
        return derivative

    def active_joint_bounds(self, pose: Ve33Pose) -> tuple[np.ndarray, np.ndarray]:
        """把当前单轴活跃树限位拉回真实指节；耦合行仍由完整CAD约束处理。"""
        values = self._state_vector(pose)
        joints = self._angles[:12] @ values
        derivative = self._tree_derivatives(pose.all_positions)
        lower = np.full(12, -np.inf)
        upper = np.full(12, np.inf)
        for name, bounds in self._solver.bounds.items():
            if name not in self._tree_index:
                continue
            index = self._tree_index[name]
            row = derivative[index]
            magnitude = np.max(np.abs(row), initial=0.0)
            if magnitude == 0.0:
                continue
            support = np.flatnonzero(np.abs(row) > magnitude * 1e-6)
            if len(support) != 1:
                continue
            joint = support[0]
            slope = row[joint]
            for bound, tree_lower in ((bounds[0], True), (bounds[1], False)):
                # 只识别数值上已活跃的真实约束，不为远处曲线构造虚假的全局q限位。
                if abs(bound - values[index]) > 1e-7:
                    continue
                limit = joints[joint] + (bound - values[index]) / slope
                if (slope > 0.0) == tree_lower:
                    lower[joint] = max(lower[joint], limit)
                else:
                    upper[joint] = min(upper[joint], limit)
        return lower, upper

    def evaluate_motors(
        self, targets: Mapping[str, float], *, seed: Ve33Pose | None = None
    ) -> Ve33Pose:
        """电机相对角正解；使用同一装配分支，试算不提交任何连续状态。"""
        if not isinstance(targets, Mapping):
            raise ValueError("电机目标必须为名称到弧度的映射")
        unknown = set(targets) - set(self.motor_names)
        if unknown:
            raise ValueError(f"未知电机坐标：{', '.join(map(str, unknown))}")
        driven = {}
        for name, value in targets.items():
            if np.iscomplexobj(value) or np.ndim(value) != 0:
                raise ValueError(f"{name}目标必须为有限实数弧度")
            try:
                driven[name] = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{name}目标必须为有限实数弧度") from error
            if not np.isfinite(driven[name]):
                raise ValueError(f"{name}目标必须为有限实数弧度")
        previous = self._state_vector(self.current if seed is None else seed)
        start = previous[self._motor_indices]
        target = start.copy()
        for index, name in enumerate(self.motor_names):
            if name in driven:
                target[index] = driven[name]
        if np.max(np.abs(target - start), initial=0.0) > 4.8:
            raise ValueError("单次电机变化过大，请分段输入；保留原姿态")
        changed = np.flatnonzero(target != start)
        active = tuple(group for group in self._groups if np.intersect1d(group.input_indices, changed).size)
        values = self._advance(previous, target, active, motor_space=True)
        result = self._pose(values)
        self._state_vector(result)
        self._tree_derivatives(result.all_positions)
        return result

    def solve_motors(
        self, targets: Mapping[str, float], *, seed: Ve33Pose | None = None
    ) -> Ve33Pose:
        """电机反馈经正解得到真实指节；全部几何检查成功后才提交。"""
        result = self.evaluate_motors(targets, seed=seed)
        self.current = result
        return result

    def solve(
        self, targets: Mapping[str, float], *, seed: Ve33Pose | None = None, local: bool = False
    ) -> Ve33Pose:
        """全部闭环校验成功才提交；seed可沿用evaluate已连续跟踪的分支姿态。"""
        result = self.evaluate(targets, seed=seed, local=local)
        self.current = result
        return result

    def reset(self) -> Ve33Pose:
        """回到原始CAD零位，不修改模型几何或旧求解器状态。"""
        self.current = self._pose(np.zeros(68))
        return self.current

    def world_frames(self, pose: Ve33Pose | None = None) -> dict[str, np.ndarray]:
        selected = self.current if pose is None else pose
        values = self._state_vector(selected)
        return self._frames(dict(zip(self._tree_names, values)))

    def point_kinematics(
        self, pose: Ve33Pose, sites: Sequence[tuple[str, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回link局部任务点的基坐标及对12个真实指节角的解析雅可比。"""
        values = self._state_vector(pose)
        state = dict(zip(self._tree_names, values))
        frames = self._frames(state)
        derivative = self._tree_derivatives(state)
        points = np.empty((len(sites), 3))
        jacobian = np.zeros((len(sites), 3, 12))
        worlds = np.array([frames[child] for child in self._tree_children])
        axes = np.einsum("nij,nj->ni", worlds[:, :3, :3], self._tree_axes)
        for index, (link, local_point) in enumerate(sites):
            if link not in frames:
                raise ValueError(f"未知任务点link：{link}")
            if np.iscomplexobj(local_point):
                raise ValueError("任务点必须为有限的局部三维实数米制坐标")
            try:
                local = np.asarray(local_point, dtype=float)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("任务点必须为有限的局部三维实数米制坐标") from error
            if local.shape != (3,) or not np.isfinite(local).all():
                raise ValueError("任务点必须为有限的局部三维实数米制坐标")
            point = frames[link][:3, :3] @ local + frames[link][:3, 3]
            points[index] = point
            indices = self._point_indices[link]
            tangents = np.cross(axes[indices], point - worlds[indices, :3, 3])
            if self._has_prismatic:
                prismatic = self._tree_prismatic[indices]
                tangents[prismatic] = axes[indices[prismatic]]
            jacobian[index] = tangents.T @ derivative[indices]
        return points, jacobian
