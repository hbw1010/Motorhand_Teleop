"""原CAD几何的C++计算边界；不承担求解、标定或控制策略。"""

from collections.abc import Mapping, Sequence

import numpy as np

from . import _ve33_geometry as native


class Ve33ClosureGeometry:
    """缓存不可变URDF元数据，计算与原ClosureSolver完全相同的点约束。"""

    def __init__(self, solver, tree_names: Sequence[str]) -> None:
        self._solver = solver
        self._tree_names = tuple(tree_names)
        tree_index = {name: index for index, name in enumerate(self._tree_names)}
        if len(tree_index) != len(self._tree_names):
            raise ValueError("树坐标名必须唯一")
        joints = solver.fk.j
        order, visited, visiting = [], set(), set()

        def visit(name):
            if name in visited:
                return
            if name in visiting:
                raise ValueError("URDF父子关系存在环")
            visiting.add(name)
            parent_joint = solver.fk.cj.get(joints[name][2])
            if parent_joint is not None:
                visit(parent_joint)
            visiting.remove(name)
            visited.add(name)
            order.append(name)

        for name in joints:
            visit(name)
        joint_nodes = {name: index for index, name in enumerate(order)}
        self._child_nodes = {joints[name][1]: joint_nodes[name] for name in order}
        self._root_links = tuple(dict.fromkeys(
            joint[2] for joint in joints.values() if joint[2] not in solver.fk.cj
        ))
        parents, kinds, indices, origins, axes = [], [], [], [], []
        kind_codes = {"fixed": 0, "revolute": 1, "continuous": 1, "prismatic": 2}
        for name in order:
            kind, child, parent, origin, axis = joints[name]
            if kind not in kind_codes:
                raise ValueError(f"不支持URDF关节类型：{kind}")
            axis = np.asarray(axis, dtype=np.float64)
            length = np.linalg.norm(axis)
            if not np.isfinite(length) or length == 0.0:
                raise ValueError(f"URDF关节{name}的轴必须有限且非零")
            parents.append(self._child_nodes.get(parent, -1))
            kinds.append(kind_codes[kind])
            indices.append(tree_index.get(name, -1))
            origins.append(origin)
            axes.append(axis / length)
        self._joint_nodes = joint_nodes
        self._kernel = native.create(
            np.asarray(parents, dtype=np.int64), np.asarray(kinds, dtype=np.uint8),
            np.asarray(indices, dtype=np.int64), np.asarray(origins, dtype=np.float64),
            np.asarray(axes, dtype=np.float64), len(self._tree_names),
        )
        self._all_constraints = self._block(
            solver.witnesses, (), np.empty((len(solver.witnesses), 0), dtype=np.uint8),
            np.empty((len(solver.witnesses), 0), dtype=np.uint8),
        )
        # Kinematics初始化时仍会添加motor-space组；每组首次使用后元数据不再变。
        self._groups = {}

    def _link_node(self, link: str) -> int:
        if link in self._child_nodes:
            return self._child_nodes[link]
        if link in self._root_links:
            return -1
        raise ValueError(f"未知URDF link：{link}")

    def _block(self, witnesses, deps, mask_a, mask_b):
        count, columns = len(witnesses), len(deps)
        return (
            np.array([self._link_node(item[0]) for item in witnesses], dtype=np.int64),
            np.array([item[1] for item in witnesses], dtype=np.float64).reshape(count, 4),
            np.array([self._link_node(item[2]) for item in witnesses], dtype=np.int64),
            np.array([item[3] for item in witnesses], dtype=np.float64).reshape(count, 4),
            np.array([self._joint_nodes[name] for name in deps], dtype=np.int64),
            np.array(mask_a, dtype=np.uint8, order="C").reshape(count, columns),
            np.array(mask_b, dtype=np.uint8, order="C").reshape(count, columns),
        )

    def _state(self, state: Mapping[str, float]) -> np.ndarray:
        return np.fromiter((state.get(name, 0.0) for name in self._tree_names),
                           dtype=np.float64, count=len(self._tree_names))

    def _evaluate(self, state: Mapping[str, float], block):
        count, columns = block[0].size, block[4].size
        residual = np.empty(3 * count, dtype=np.float64)
        jacobian = np.empty((3 * count, columns), dtype=np.float64)
        native.evaluate(self._kernel, self._state(state), *block, residual, jacobian)
        return residual, jacobian

    def group_residual_jac(self, state: Mapping[str, float], group: str):
        if group not in self._groups:
            witnesses, mask_a, mask_b = self._solver._blocks[group]
            self._groups[group] = self._block(
                witnesses, self._solver.groups[group], mask_a, mask_b,
            )
        return self._evaluate(state, self._groups[group])

    def residual(self, state: Mapping[str, float]) -> np.ndarray:
        return self._evaluate(state, self._all_constraints)[0]

    def max_gap(self, state: Mapping[str, float]) -> float:
        return float(np.linalg.norm(self.residual(state).reshape(-1, 3), axis=1).max())

    def world_frames(self, state: Mapping[str, float]) -> dict[str, np.ndarray]:
        frames = np.empty((len(self._joint_nodes), 4, 4), dtype=np.float64)
        native.frames(self._kernel, self._state(state), frames)
        result = {link: frames[index] for link, index in self._child_nodes.items()}
        result.update({link: np.eye(4) for link in self._root_links})
        return result
