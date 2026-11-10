"""Vectorized closed-chain solver for the vE3.3 glove.

The CAD solver in ``assets/.../scripts/solve_closed_chain.py`` is the
reference implementation.  It evaluates forward kinematics with a recursive
per-link Python walk and calls ``scipy.optimize.least_squares`` once per finger
per continuation step, which costs ~7 ms per pose update and limits the
control loop.

This module keeps the reference *semantics* and replaces only the numerics:

* forward kinematics of the whole tree is evaluated level by level with
  batched numpy (one pass for every witness point of every finger);
* each changed finger block is solved with a projected Gauss-Newton iteration
  (exact point-constraint Jacobian, joint bounds by clipping);
* the same continuation (<=0.02 rad per step), closure tolerance,
  per-step branch-jump limit (0.2 rad) and bisection retry are applied, and the
  same error is raised when the target leaves the assembly branch.

``GloveKinematics`` falls back to the reference solver whenever this solver
raises, so a failure here can only cost time, never accept a worse pose.
"""
from __future__ import annotations

import numpy as np


class FastClosureError(ValueError):
    pass


class FastClosure:
    STEP_RAD = 0.02
    MAX_JUMP_RAD = 0.2
    MAX_DEPTH = 8
    MAX_ITERS = 25

    def __init__(self, solver) -> None:
        fk = solver.fk
        self.tolerance = float(solver.tolerance)
        self.independent = list(solver.independent)
        self.dependent = list(solver.dependent)
        self.groups = {k: list(v) for k, v in solver.groups.items()}
        self.bounds = dict(solver.bounds)

        # ---- tree topology, ordered by depth --------------------------
        names = list(fk.j.keys())
        depth: dict[str, int] = {}

        def joint_depth(name: str) -> int:
            if name not in depth:
                parent_link = fk.j[name][2]
                parent_joint = fk.cj.get(parent_link)
                depth[name] = 0 if parent_joint is None else joint_depth(parent_joint) + 1
            return depth[name]

        for name in names:
            joint_depth(name)
        order = sorted(names, key=lambda n: depth[n])
        self.joint_names = order
        self.index = {n: i for i, n in enumerate(order)}
        n = len(order)
        self.n = n
        root = n  # extra identity slot for links without a parent joint
        self.parent = np.array(
            [self.index[fk.cj[fk.j[name][2]]] if fk.j[name][2] in fk.cj else root for name in order],
            dtype=int,
        )
        self.levels = []
        for d in range(max(depth.values()) + 1):
            idx = np.array([i for i, name in enumerate(order) if depth[name] == d], dtype=int)
            if idx.size:
                self.levels.append((idx, self.parent[idx]))
        types = [fk.j[name][0] for name in order]
        self.revolute = np.array([t in ("revolute", "continuous") for t in types])
        self.prismatic = np.array([t == "prismatic" for t in types])
        self.t_rot = np.stack([fk.j[name][3][:3, :3] for name in order])
        self.t_pos = np.stack([fk.j[name][3][:3, 3] for name in order])
        axes = np.stack([np.asarray(fk.j[name][4], dtype=float) for name in order])
        norms = np.linalg.norm(axes, axis=1)
        norms[norms == 0] = 1.0
        self.axis = axes / norms[:, None]
        a = self.axis
        self._aat = a[:, :, None] * a[:, None, :]
        k = np.zeros((n, 3, 3))
        k[:, 0, 1], k[:, 0, 2] = -a[:, 2], a[:, 1]
        k[:, 1, 0], k[:, 1, 2] = a[:, 2], -a[:, 0]
        k[:, 2, 0], k[:, 2, 1] = -a[:, 1], a[:, 0]
        self._skew = k
        self._eye = np.broadcast_to(np.eye(3), (n, 3, 3))
        self.child_slot = {fk.j[name][1]: self.index[name] for name in order}

        # ---- witnesses (same points as the reference solver) ----------
        wit = solver.witnesses
        self.w_la = np.array([self.child_slot.get(w[0], root) for w in wit], dtype=int)
        self.w_lb = np.array([self.child_slot.get(w[2], root) for w in wit], dtype=int)
        self.w_pa = np.stack([np.asarray(w[1], dtype=float)[:3] for w in wit])
        self.w_pb = np.stack([np.asarray(w[3], dtype=float)[:3] for w in wit])
        wid = {id(w): i for i, w in enumerate(wit)}
        self.blocks = {}
        for group, deps in self.groups.items():
            witnesses, mask_a, mask_b = solver._blocks[group]
            rows = np.array([wid[id(w)] for w in witnesses], dtype=int)
            dep_idx = np.array([self.index[d] for d in deps], dtype=int)
            lower = np.array([self.bounds.get(d, (-np.inf, np.inf))[0] for d in deps], dtype=float)
            upper = np.array([self.bounds.get(d, (-np.inf, np.inf))[1] for d in deps], dtype=float)
            self.blocks[group] = (rows, np.asarray(mask_a, float), np.asarray(mask_b, float),
                                  dep_idx, lower, upper)
        # Ancestor masks for independent (motor) joints, used by derivative().
        anc = np.zeros((n + 1, n), dtype=bool)
        for i in range(n):  # order is depth-sorted, parents first
            p = self.parent[i]
            if p != root:
                anc[i] = anc[p]
            anc[i, i] = True
        self._anc = anc
        self.state = np.zeros(n)

    # ------------------------------------------------------------------
    def vector(self, state: dict[str, float]) -> np.ndarray:
        q = np.zeros(self.n)
        for name, value in state.items():
            i = self.index.get(name)
            if i is not None:
                q[i] = value
        return q

    def as_dict(self, q: np.ndarray) -> dict[str, float]:
        return {name: float(q[self.index[name]]) for name in self.independent + self.dependent}

    def _fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = self.n
        c, s = np.cos(q), np.sin(q)
        rot = c[:, None, None] * self._eye + s[:, None, None] * self._skew + (1.0 - c)[:, None, None] * self._aat
        rot[~self.revolute] = np.eye(3)
        local_r = self.t_rot @ rot
        local_p = self.t_pos.copy()
        if self.prismatic.any():
            pm = self.prismatic
            local_p[pm] += np.einsum("nij,nj->ni", self.t_rot[pm], self.axis[pm] * q[pm, None])
        wr = np.empty((n + 1, 3, 3))
        wp = np.empty((n + 1, 3))
        wr[n] = np.eye(3)
        wp[n] = 0.0
        for idx, par in self.levels:
            pr = wr[par]
            wr[idx] = pr @ local_r[idx]
            wp[idx] = np.einsum("nij,nj->ni", pr, local_p[idx]) + wp[par]
        return wr, wp

    def _points(self, wr, wp, rows=None):
        la, lb, pa, pb = self.w_la, self.w_lb, self.w_pa, self.w_pb
        if rows is not None:
            la, lb, pa, pb = la[rows], lb[rows], pa[rows], pb[rows]
        a = np.einsum("nij,nj->ni", wr[la], pa) + wp[la]
        b = np.einsum("nij,nj->ni", wr[lb], pb) + wp[lb]
        return a, b

    def gap(self, q: np.ndarray) -> float:
        wr, wp = self._fk(q)
        a, b = self._points(wr, wp)
        return float(np.linalg.norm(a - b, axis=1).max())

    def _block_jac(self, wr, wp, a, b, cols, mask_a, mask_b):
        origins = wp[cols]
        axes = np.einsum("nij,nj->ni", wr[cols], self.axis[cols])
        da = np.cross(axes[None, :, :], a[:, None, :] - origins[None, :, :])
        db = np.cross(axes[None, :, :], b[:, None, :] - origins[None, :, :])
        pm = self.prismatic[cols]
        if pm.any():
            da[:, pm] = axes[pm]
            db[:, pm] = axes[pm]
        jac = da * mask_a[:, :, None] - db * mask_b[:, :, None]
        return jac.transpose(0, 2, 1).reshape(-1, cols.size)

    def _solve_groups(self, q: np.ndarray, groups: list[str]) -> tuple[np.ndarray, float]:
        """Gauss-Newton on the changed finger blocks (blocks are independent)."""
        q = q.copy()
        initial = q.copy()
        tol = self.tolerance * 0.05
        active = list(groups)
        for _ in range(self.MAX_ITERS):
            wr, wp = self._fk(q)
            still = []
            for group in active:
                rows, mask_a, mask_b, cols, lower, upper = self.blocks[group]
                a, b = self._points(wr, wp, rows)
                res = a - b
                if np.sqrt((res * res).sum(axis=1)).max() <= tol:
                    continue
                jac = self._block_jac(wr, wp, a, b, cols, mask_a, mask_b)
                step = np.linalg.lstsq(jac, -res.ravel(), rcond=1e-10)[0]
                q[cols] = np.clip(q[cols] + step, lower, upper)
                still.append(group)
            if not still:
                break
            active = still
        jump = 0.0
        for group in groups:
            cols = self.blocks[group][3]
            jump = max(jump, float(np.max(np.abs(q[cols] - initial[cols]), initial=0.0)))
        return q, jump

    def solve(self, state: np.ndarray, driven: dict[str, float]) -> np.ndarray:
        """Continuation solve from a valid ``state`` (vector) to ``driven`` motor targets."""
        idx = {name: self.index[name] for name in driven}
        changed = {n: float(v) for n, v in driven.items() if abs(float(v) - state[idx[n]]) > 1e-12}
        if not changed:
            return state.copy()
        count = max(1, int(np.ceil(max(abs(v - state[idx[n]]) / self.STEP_RAD for n, v in changed.items()))))
        if count > 240:
            raise FastClosureError("Move in smaller increments")
        groups = sorted({n.split("_")[0] for n in changed})

        def advance(previous: np.ndarray, targets: dict[str, float], depth: int = 0) -> np.ndarray:
            candidate = previous.copy()
            for name, value in targets.items():
                candidate[idx[name]] = value
            candidate, jump = self._solve_groups(candidate, groups)
            if self.gap(candidate) <= self.tolerance and jump <= self.MAX_JUMP_RAD:
                return candidate
            if depth >= self.MAX_DEPTH or max(abs(v - previous[idx[n]]) for n, v in targets.items()) < 1e-5:
                raise FastClosureError("Target exceeds this closed-chain assembly branch")
            midpoint = {n: (previous[idx[n]] + v) / 2 for n, v in targets.items()}
            middle = advance(previous, midpoint, depth + 1)
            return advance(middle, targets, depth + 1)

        current = state.copy()
        start = {n: state[idx[n]] for n in changed}
        for step in range(1, count + 1):
            targets = {n: start[n] + (v - start[n]) * step / count for n, v in changed.items()}
            current = advance(current, targets)
        return current

    def derivative(self, q: np.ndarray, motor_names: list[str]) -> dict[str, tuple[list[str], list[str], np.ndarray]]:
        """Implicit-function derivative d(dependent)/d(motor) per finger block.

        Returns {group: (deps, motors, dq_dep)} with dq_dep of shape
        (len(deps), len(motors)).  B is the exact point-constraint Jacobian with
        respect to the motor joints (the reference uses finite differences of
        the same residual).
        """
        wr, wp = self._fk(q)
        out = {}
        for group, deps in self.groups.items():
            motors = [m for m in motor_names if m.split("_")[0] == group]
            if not motors:
                continue
            rows, mask_a, mask_b, cols, _lo, _hi = self.blocks[group]
            a, b = self._points(wr, wp, rows)
            jac = self._block_jac(wr, wp, a, b, cols, mask_a, mask_b)
            mcols = np.array([self.index[m] for m in motors], dtype=int)
            ma = self._anc[self.w_la[rows]][:, mcols].astype(float)
            mb = self._anc[self.w_lb[rows]][:, mcols].astype(float)
            bmat = self._block_jac(wr, wp, a, b, mcols, ma, mb)
            try:
                dq = np.linalg.lstsq(jac, -bmat, rcond=1e-9)[0]
            except np.linalg.LinAlgError:
                continue
            out[group] = (deps, motors, dq)
        return out
