"""Offline differential checks against the unmodified CAD closure solver."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from motorhand_teleop import _ve33_geometry as native
from motorhand_teleop.ve33_geometry import Ve33ClosureGeometry


ROOT = Path(__file__).parents[1]


def _original_module(side):
    root = ROOT / "assets/hand_motor_ve33" / f"hand_motor_ve33_{side}"
    spec = importlib.util.spec_from_file_location(
        f"native_geometry_oracle_{side}", root / "scripts/solve_closed_chain.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, root


def _ancestors(solver, link):
    result = set()
    while link in solver.fk.cj:
        name = solver.fk.cj[link]
        result.add(name)
        link = solver.fk.j[name][2]
    return result


@pytest.fixture(scope="module", params=("L", "R"))
def cad(request):
    module, root = _original_module(request.param)
    solver = module.ClosureSolver(
        root / "hand_tree_only.urdf", root / "config/loop_closures.yaml"
    )
    names = tuple(solver.independent + solver.dependent)
    # Generate the actual assembly branch with the ORIGINAL solver before any
    # native object exists. Reverse direction as well as changing each motor.
    closed = []
    for scale in (0.008, -0.006, 0.012):
        motors = {
            name: scale * (1.0 + index / 20.0)
            for index, name in enumerate(solver.independent)
        }
        state = solver.solve(motors)
        assert solver.max_gap(state) <= solver.tolerance
        assert max(abs(state[name]) for name in solver.dependent) > 1e-5
        closed.append(state)
    # Retain every original passive-column block, and additionally exercise
    # motor-space columns, including ancestors shared by BOTH witness ends.
    for group, deps in tuple(solver.groups.items()):
        expanded = list(deps) + [
            name for name in solver.independent if name.split("_")[0] == group
        ]
        witnesses = solver._blocks[group][0]
        key = f"{group}_with_motors"
        solver.groups[key] = expanded
        masks = [np.array([
            [name in _ancestors(solver, witness[side]) for name in expanded]
            for witness in witnesses
        ]) for side in (0, 2)]
        solver._blocks[key] = (witnesses, *masks)
    geometry = Ve33ClosureGeometry(solver, names)
    # A deliberately nonclosed pose prevents zero residuals from hiding errors.
    mixed = {name: 0.031 * np.sin(0.7 + index * 1.37) for index, name in enumerate(names)}
    return solver, geometry, names, [*closed, mixed]


def test_all_cad_groups_residual_gap_and_world_frames_match_original(cad):
    solver, geometry, _, states = cad
    links = {link for joint in solver.fk.j.values() for link in joint[1:3]}
    for state in states:
        expected = solver.residual(state)
        np.testing.assert_allclose(geometry.residual(state), expected, rtol=1e-11, atol=2e-13)
        assert geometry.max_gap(state) == pytest.approx(solver.max_gap(state), rel=1e-10, abs=2e-13)
        frames = geometry.world_frames(state)
        assert set(frames) == links
        for link in links:
            np.testing.assert_allclose(frames[link], solver.fk.world(link, state), rtol=1e-12, atol=2e-13)
        for group in solver.groups:
            residual, jacobian = geometry.group_residual_jac(state, group)
            expected_residual, expected_jacobian = solver._group_residual_jac(state, group)
            np.testing.assert_allclose(residual, expected_residual, rtol=1e-11, atol=2e-13)
            np.testing.assert_allclose(jacobian, expected_jacobian, rtol=1e-11, atol=3e-13)


def test_cad_jacobians_match_independent_centered_differences(cad):
    solver, geometry, _, states = cad
    state = states[-1]
    step = 1e-6
    for group, deps in solver.groups.items():
        _, actual = geometry.group_residual_jac(state, group)
        columns = []
        for name in deps:
            plus, minus = dict(state), dict(state)
            plus[name] += step
            minus[name] -= step
            # Numerical derivative uses only the original Python geometry.
            rp = solver._group_residual_jac(plus, group)[0]
            rm = solver._group_residual_jac(minus, group)[0]
            columns.append((rp - rm) / (2 * step))
        np.testing.assert_allclose(actual, np.column_stack(columns), rtol=2e-6, atol=1e-9)


def test_cad_snapshots_and_cache_survive_mutated_mapping_and_revisits(cad):
    solver, geometry, names, states = cad
    state = dict(states[0])
    group = next(iter(solver.groups))
    residual, jacobian = geometry.group_residual_jac(state, group)
    global_residual = geometry.residual(state)
    frames = geometry.world_frames(state)
    saved = (residual.copy(), jacobian.copy(), global_residual.copy(),
             {name: frame.copy() for name, frame in frames.items()})
    # Reuse the same mapping object and change a late coordinate: neither its
    # identity nor a prefix of the flattened state is a valid FK cache key.
    state[names[-1]] += 0.073
    for current in (state, states[1], states[-1], states[0], {}):
        np.testing.assert_allclose(geometry.residual(current), solver.residual(current), atol=2e-13, rtol=1e-11)
        geometry.group_residual_jac(current, group)
        fresh = geometry.world_frames(current)
        for link in fresh:
            np.testing.assert_allclose(fresh[link], solver.fk.world(link, current), atol=2e-13, rtol=1e-12)
    np.testing.assert_array_equal(residual, saved[0])
    np.testing.assert_array_equal(jacobian, saved[1])
    np.testing.assert_array_equal(global_residual, saved[2])
    for link in frames:
        np.testing.assert_array_equal(frames[link], saved[3][link])
    # Consumer writes must not poison future native evaluations either.
    residual[:] = 123.0
    jacobian[:] = 456.0
    for frame in frames.values():
        frame[:] = 789.0
    actual_r, actual_j = geometry.group_residual_jac(states[0], group)
    np.testing.assert_allclose(actual_r, saved[0], atol=2e-13)
    np.testing.assert_allclose(actual_j, saved[1], atol=3e-13)
    for link, frame in geometry.world_frames(states[0]).items():
        np.testing.assert_allclose(frame, saved[3][link], atol=2e-13)


def test_fixed_rotated_prismatic_chain_has_exact_physical_geometry(tmp_path):
    module, _ = _original_module("R")
    urdf = tmp_path / "chain.urdf"
    urdf.write_text('''<robot name="exact_chain">
<link name="root"/><link name="mount"/><link name="arm"/><link name="tip"/>
<joint name="mount_joint" type="fixed"><parent link="root"/><child link="mount"/>
<origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/></joint>
<joint name="turn" type="revolute"><parent link="mount"/><child link="arm"/>
<origin xyz="2 0 0"/><axis xyz="0 0 5"/></joint>
<joint name="slide" type="prismatic"><parent link="arm"/><child link="tip"/>
<origin xyz="1 0 0"/><axis xyz="3 0 0"/></joint></robot>''', encoding="utf-8")
    config = tmp_path / "chain.yaml"
    config.write_text(yaml.safe_dump({
        "independent": ["turn", "slide"], "dependent": [],
        "closure_tolerance_m": 1e-7, "groups": {"chain": ["turn", "slide"]},
        "closures": [],
    }), encoding="utf-8")
    solver = module.ClosureSolver(urdf, config)
    witness = ("tip", np.array([0., 2., 0., 1.]), "root", np.array([0., 0., 0., 1.]))
    solver.witnesses = [witness]
    solver._blocks["chain"] = ([witness], np.ones((1, 2), dtype=bool), np.zeros((1, 2), dtype=bool))
    geometry = Ve33ClosureGeometry(solver, ("slide", "mount_joint", "turn"))
    state = {"turn": np.pi / 2, "slide": 0.25, "mount_joint": 17.0}
    residual, jacobian = geometry.group_residual_jac(state, "chain")
    np.testing.assert_allclose(residual, [-0.25, 2., 3.], atol=2e-14)
    np.testing.assert_allclose(jacobian, [[2., -1.], [-1.25, 0.], [0., 0.]], atol=2e-14)
    np.testing.assert_allclose(geometry.residual(state), residual, atol=2e-14)
    assert geometry.max_gap(state) == pytest.approx(np.sqrt(13.0625), abs=2e-14)
    frames = geometry.world_frames(state)
    expected_tip = np.diag([-1., -1., 1., 1.])
    expected_tip[:3, 3] = [-0.25, 4., 3.]
    np.testing.assert_allclose(frames["tip"], expected_tip, atol=2e-14)
    np.testing.assert_array_equal(frames["root"], np.eye(4))
    for link in frames:
        np.testing.assert_allclose(frames[link], solver.fk.world(link, state), atol=2e-14)
    reverse = {"turn": -np.pi / 2, "slide": -0.5}
    reverse_r, reverse_j = geometry.group_residual_jac(reverse, "chain")
    np.testing.assert_allclose(reverse_r, [1.5, 6., 3.], atol=2e-14)
    np.testing.assert_allclose(reverse_j, [[-2., 1.], [0.5, 0.], [0., 0.]], atol=2e-14)


def _native_model():
    return [np.array([-1], dtype=np.int64), np.array([1], dtype=np.uint8),
            np.array([0], dtype=np.int64), np.eye(4)[None].copy(),
            np.array([[0., 0., 1.]]), 2]


def _native_evaluation():
    return [np.array([0.3, -0.2]), np.array([0], dtype=np.int64),
            np.array([[1., 0., 0., 1.]]), np.array([-1], dtype=np.int64),
            np.array([[0., 0., 0., 1.]]), np.array([0], dtype=np.int64),
            np.ones((1, 1), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8),
            np.full(3, 71.), np.full((3, 1), 83.)]


@pytest.mark.parametrize("case", ["parent", "kind", "state_index", "dtype", "axis_nan", "origin_shape"])
def test_native_create_rejects_invalid_metadata(case):
    args = _native_model()
    if case == "parent":
        args[0][0] = 0
    elif case == "kind":
        args[1][0] = 3
    elif case == "state_index":
        args[2][0] = 2
    elif case == "dtype":
        args[0] = args[0].astype(np.int32)
    elif case == "axis_nan":
        args[4][0, 1] = np.nan
    elif case == "origin_shape":
        args[3] = np.zeros((1, 3, 4))
    with pytest.raises((TypeError, ValueError)):
        native.create(*args)


@pytest.mark.parametrize("case", [
    "state_dtype", "state_shape", "state_noncontiguous", "state_nan",
    "link", "dependency", "mask_shape", "point_inf", "output_dtype", "readonly",
])
def test_native_evaluate_rejects_before_writing_outputs(case):
    model = native.create(*_native_model())
    args = _native_evaluation()
    if case == "state_dtype":
        args[0] = args[0].astype(np.float32)
    elif case == "state_shape":
        args[0] = args[0].reshape(1, 2)
    elif case == "state_noncontiguous":
        args[0] = np.zeros(4)[::2]
    elif case == "state_nan":
        args[0][1] = np.nan
    elif case == "link":
        args[3][0] = -2
    elif case == "dependency":
        args[5][0] = 1
    elif case == "mask_shape":
        args[7] = np.zeros((1, 2), dtype=np.uint8)
    elif case == "point_inf":
        args[4][0, 0] = np.inf
    elif case == "output_dtype":
        args[9] = args[9].astype(np.float32)
    elif case == "readonly":
        args[9].flags.writeable = False
    before = [args[8].copy(), args[9].copy()]
    with pytest.raises((TypeError, ValueError)):
        native.evaluate(model, *args)
    np.testing.assert_array_equal(args[8], before[0])
    np.testing.assert_array_equal(args[9], before[1])
    # A rejected request must not corrupt the cached state or next valid result.
    valid = _native_evaluation()
    native.evaluate(model, *valid)
    c, s = np.cos(0.3), np.sin(0.3)
    np.testing.assert_allclose(valid[8], [c, s, 0.], atol=2e-15)
    np.testing.assert_allclose(valid[9], [[-s], [c], [0.]], atol=2e-15)


def test_native_frames_rejects_shape_before_overwrite():
    model = native.create(*_native_model())
    output = np.full((1, 3, 4), 97.)
    with pytest.raises((TypeError, ValueError)):
        native.frames(model, np.zeros(2), output)
    np.testing.assert_array_equal(output, np.full((1, 3, 4), 97.))
    frames = np.empty((1, 4, 4))
    state = np.array([np.pi / 2, 0.])
    native.frames(model, state, frames)
    np.testing.assert_allclose(frames[0], [[0., -1., 0., 0.], [1., 0., 0., 0.],
                                          [0., 0., 1., 0.], [0., 0., 0., 1.]], atol=2e-15)
    saved = frames.copy()
    state[0] = -np.pi / 2
    fresh = np.empty_like(frames)
    native.frames(model, state, fresh)
    np.testing.assert_array_equal(frames, saved)
    np.testing.assert_allclose(fresh[0, :2, :2], [[0., 1.], [-1., 0.]], atol=2e-15)


@pytest.mark.parametrize("count", [0, 1, 7])
def test_point_jacobian_matches_independent_numpy_geometry(count):
    rng = np.random.default_rng(42)
    points = rng.normal(size=(count, 3))
    worlds = rng.normal(size=(68, 4, 4))
    axes = rng.normal(size=(68, 3))
    derivative = rng.normal(size=(68, 12))
    indices = np.full((count, 68), -1, dtype=np.int64)
    for row in indices:
        row[:6] = [0, 3, 11, 20, 51, 67]
    prismatic = np.zeros(68, dtype=np.uint8)
    prismatic[[3, 51]] = 1
    actual = np.empty((count, 3, 12))
    native.point_jacobian(points, worlds, axes, derivative, indices, prismatic, actual)
    expected = np.zeros_like(actual)
    for index, point in enumerate(points):
        joints = indices[index][indices[index] >= 0]
        tangents = np.cross(axes[joints], point - worlds[joints, :3, 3])
        tangents[prismatic[joints] != 0] = axes[joints[prismatic[joints] != 0]]
        expected[index] = tangents.T @ derivative[joints]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("bad", ["index", "dtype", "nonfinite", "overlap"])
def test_point_jacobian_rejects_invalid_input_without_partial_output(bad):
    shared = np.full(36, 17.0)
    points = np.ones((1, 3))
    worlds = np.zeros((68, 4, 4))
    axes = np.ones((68, 3))
    derivative = np.ones((68, 12))
    indices = np.full((1, 68), -1, dtype=np.int64)
    prismatic = np.zeros(68, dtype=np.uint8)
    out = shared.reshape(1, 3, 12)
    if bad == "index":
        indices[0, 0] = 68
    elif bad == "dtype":
        indices = indices.astype(np.int32)
    elif bad == "nonfinite":
        axes[0, 0] = np.nan
    else:
        points = shared[:3].reshape(1, 3)
    with pytest.raises((TypeError, ValueError)):
        native.point_jacobian(points, worlds, axes, derivative, indices, prismatic, out)
    np.testing.assert_array_equal(out, np.full((1, 3, 12), 17.0))


def test_state_vector_preserves_name_order_missing_and_unknown_coordinates():
    state = {"b": np.float64(-2.25), "a": "1.5", "unrelated": object()}
    out = np.full(4, 99.0)
    native.state_vector(state, ("c", "b", "a", "missing"), out)
    np.testing.assert_array_equal(out, [0.0, -2.25, 1.5, 0.0])


def test_state_vector_rejects_non_numeric_coordinates():
    with pytest.raises((TypeError, ValueError)):
        native.state_vector({"a": object()}, ("a",), np.empty(1))
