"""实CAD解析导数、受限逆解与纯试算回归；直接执行本文件可打印实时烟测。"""

from pathlib import Path
from time import monotonic_ns, perf_counter_ns

import numpy as np
import pytest
from scipy.optimize import least_squares, minimize

from wujihand2_force_glove.bidirectional import GloveInverseSolver, GloveTaskModel
from wujihand2_force_glove.domain import GLOVE_JOINTS
from wujihand2_force_glove.landmarks import Ve33LandmarkSource


ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "hand_motor_ve33"


def _joints(values):
    return dict(zip(GLOVE_JOINTS, values))


def _model(hand="right"):
    return GloveTaskModel(Ve33LandmarkSource(hand, asset_root=ASSET_ROOT))


def _canonical_points_oracle(model, values):
    """用原CAD逐link FK算点值，不走point_kinematics或其解析雅可比。"""
    source = model.landmarks
    pose = source.model.evaluate_raw(_joints(values))
    return _canonical_tree_points_oracle(model, pose.all_positions)


def _canonical_tree_points_oracle(model, positions):
    """直接用原CAD树的刚体FK读点，独立于任务源的point_kinematics实现。"""
    source = model.landmarks
    fk = source.model.kinematics._solver.fk
    points = []
    for link, local in source._sites:
        world = fk.world(link, positions)
        points.append(world[:3, :3] @ local + world[:3, 3])
    points = np.asarray(points)
    points[1] = (points[0] + points[2]) * 0.5
    return (points - points[0]) @ source.model_to_canonical_row


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize("case", ["positive", "negative", "upper_bound", "lower_bound"])
def test_canonical_jacobian_matches_independent_cad_central_difference(hand, case):
    model = _model(hand)
    values = np.linspace(0.02, 0.08, 12)
    if case in ("negative", "lower_bound"):
        values *= -1
    if case in ("upper_bound", "lower_bound"):
        # CMC直接轴可安全靠近逆解的±1.5rad边界，不假设闭链可达整个盒限位。
        values[0] = 1.499 if case == "upper_bound" else -1.499
    model.landmarks.model.solve(_joints(values))
    committed = model.landmarks.model.kinematics.current
    state, jacobian = model.state_and_jacobian(_joints(values), timestamp_ns=123)
    np.testing.assert_allclose(state.points_m, _canonical_points_oracle(model, values), atol=1e-11)
    assert state.timestamp_ns == 123
    assert jacobian.shape == (21, 3, 12)
    assert not jacobian.flags.writeable
    numeric = np.empty_like(jacobian)
    step = 2e-5
    for index in range(12):
        delta = np.zeros(12)
        delta[index] = step
        numeric[:, :, index] = (
            _canonical_points_oracle(model, values + delta)
            - _canonical_points_oracle(model, values - delta)
        ) / (2 * step)
    np.testing.assert_allclose(jacobian, numeric, atol=2e-6, rtol=2e-4)
    np.testing.assert_allclose(jacobian[0], 0.0, atol=1e-14)
    np.testing.assert_allclose(jacobian[1], jacobian[2] * 0.5, atol=1e-14)
    assert model.landmarks.model.kinematics.current is committed
    assert model.landmarks.model._reference_pose is committed


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_real_inverse_recovers_task_and_reuses_each_candidate_evaluation(monkeypatch, sign):
    model = _model()
    values = sign * np.linspace(0.025, 0.07, 12)
    target = model.state(_joints(values), timestamp_ns=456)
    committed = model.landmarks.model.kinematics.current
    evaluate = model.landmarks.evaluate_with_seed
    candidates = []
    seeds = []

    def counted(joints, *, seed=None):
        candidates.append(np.asarray([joints[name] for name in GLOVE_JOINTS]))
        seeds.append(seed)
        return evaluate(joints, seed=seed)

    monkeypatch.setattr(model.landmarks, "evaluate_with_seed", counted)
    solver = GloveInverseSolver(model, regularization_weight=0.0, max_nfev=30)
    result = solver.solve(target)
    assert result.success
    core = model.landmarks.model.kinematics
    assert set(result.tree_positions_rad) == set(core.tree_names)
    tree_vector = np.asarray([result.tree_positions_rad[name] for name in core.tree_names])
    np.testing.assert_allclose(core._angles[:12] @ tree_vector, result.glove_positions_rad, atol=1e-10)
    with pytest.raises(TypeError):
        result.tree_positions_rad[core.motor_names[0]] = 0.0
    assert result.task_error_max_mm < 0.02
    assert result.task_error_rms_mm < 0.01
    assert len(candidates) <= result.iterations + 1
    assert seeds[0] is None
    assert all(seed is not None for seed in seeds[1:])
    assert all(not np.array_equal(a, b) for a, b in zip(candidates, candidates[1:]))
    np.testing.assert_allclose(result.glove_positions_rad, values, atol=1e-3)
    assert model.landmarks.model.kinematics.current is committed
    assert model.landmarks.model._reference_pose is committed


@pytest.mark.parametrize("regularization", [0.0, 0.02])
def test_analytic_inverse_preserves_numeric_objective_and_boundaries(regularization):
    model = _model()
    initial = np.linspace(-0.025, 0.025, 12)
    initial[0] = 1.48
    expected = initial + np.linspace(-0.01, 0.01, 12)
    expected[0] = 1.499
    model.landmarks.model.solve(_joints(initial))
    target = model.state(_joints(expected), timestamp_ns=789)
    weight = np.sqrt(regularization)
    task_weight = 1.7

    def numeric_residual(values):
        task = ((_canonical_points_oracle(model, values) - target.points_m) * task_weight).reshape(-1)
        return np.concatenate((task, weight * (values - initial)))

    oracle = least_squares(
        numeric_residual, initial, bounds=(-1.5, 1.5), max_nfev=30,
        xtol=1e-9, ftol=1e-9, gtol=1e-9,
    )
    solver = GloveInverseSolver(
        model, task_weight=task_weight, regularization_weight=regularization, max_nfev=30
    )
    actual = solver.solve(target, initial=initial)
    assert oracle.success and actual.success
    np.testing.assert_allclose(actual.glove_positions_rad, oracle.x, atol=2e-4, rtol=1e-4)
    assert np.max(np.abs(actual.glove_positions_rad)) <= 1.5
    np.testing.assert_allclose(
        numeric_residual(actual.glove_positions_rad), numeric_residual(oracle.x), atol=2e-6
    )


def test_motor_limit_changes_clear_all_warm_state_and_equal_mapping_is_idempotent():
    model = _model()
    solver = GloveInverseSolver(model)
    motor = model.landmarks.model.motor_names[0]
    zero = model.state(_joints(np.zeros(12)), timestamp_ns=800)
    assert solver.solve(zero).success
    solver.set_motor_limits({motor: (-0.2, 0.2)})
    assert solver._last_successful_positions is None
    assert solver._last_successful_seed is None
    assert solver._last_successful_target is None
    assert solver._last_successful_anchor is None
    assert solver._last_successful_jacobian is None
    assert solver._warm_step_scale == 0.04
    assert solver.solve(zero).success
    cached = (
        solver._last_successful_positions, solver._last_successful_seed,
        solver._last_successful_target, solver._last_successful_anchor,
        solver._last_successful_jacobian, solver._warm_step_scale,
    )
    solver.set_motor_limits({motor: (-0.2, 0.2)})
    assert solver._last_successful_positions is cached[0]
    assert solver._last_successful_seed is cached[1]
    assert solver._last_successful_target is cached[2]
    assert solver._last_successful_anchor is cached[3]
    assert solver._last_successful_jacobian is cached[4]
    assert solver._warm_step_scale == cached[5]
    solver.set_motor_limits({})
    assert solver._last_successful_positions is None
    assert np.all(np.isneginf(solver._motor_lower))
    assert np.all(np.isposinf(solver._motor_upper))


@pytest.mark.parametrize("limits", [
    None, {"unknown_motor": (0.0, 1.0)}, {"thumb_mcp": (0.0, 1.0)},
    {"thumb_mcp_motor_joint": (0.0, 0.0)}, {"thumb_mcp_motor_joint": (0.2, 0.1)},
    {"thumb_mcp_motor_joint": (0.0, np.nan)}, {"thumb_mcp_motor_joint": (0.0, np.inf)},
    {"thumb_mcp_motor_joint": (0.0, 1j)}, {"thumb_mcp_motor_joint": (0.0,)},
    {"thumb_mcp_motor_joint": 0.1},
])
def test_invalid_motor_limits_do_not_replace_limits_or_successful_cache(limits):
    model = _model()
    solver = GloveInverseSolver(model)
    solver.set_motor_limits({model.landmarks.model.motor_names[0]: (-0.2, 0.2)})
    assert solver.solve(model.state(_joints(np.zeros(12)))).success
    lower, upper = solver._motor_lower.copy(), solver._motor_upper.copy()
    positions, seed = solver._last_successful_positions, solver._last_successful_seed
    with pytest.raises(ValueError):
        solver.set_motor_limits(limits)
    np.testing.assert_array_equal(solver._motor_lower, lower)
    np.testing.assert_array_equal(solver._motor_upper, upper)
    assert solver._last_successful_positions is positions
    assert solver._last_successful_seed is seed


def test_nonlinear_motor_limits_match_numeric_oracle_without_changing_sensor_or_actual_anchor():
    model = _model()
    mechanism = model.landmarks.model
    motor = mechanism.joint_motor_names["index_pip"]
    measured_motors = dict.fromkeys(mechanism.motor_names, 0.0)
    measured_motors[motor] = -0.05
    measured, _ = mechanism.evaluate_motors(measured_motors)
    actual = measured.vector()
    target = model.state(_joints(actual), timestamp_ns=801)
    solver = GloveInverseSolver(model, regularization_weight=0.02)
    original_cad_bounds = dict(mechanism.kinematics._solver.bounds)
    solver.set_motor_limits({motor: (0.0, 0.08)})
    # 软件发令限位不能污染传感正解，实际反馈仍允许处在软件范围之外。
    reread, _ = mechanism.evaluate_motors(measured_motors)
    np.testing.assert_allclose(reread.vector(), actual, atol=1e-10)
    assert mechanism.kinematics._solver.bounds == original_cad_bounds

    def numeric_residual(values):
        task = (_canonical_points_oracle(model, values) - target.points_m).reshape(-1)
        return np.concatenate((task, np.sqrt(0.02) * (values - actual)))

    def numeric_constraint(values):
        position = mechanism.evaluate_raw(_joints(values)).motor_positions[motor]
        return np.asarray([position, 0.08 - position])

    def numeric_objective(values):
        residual = numeric_residual(values)
        return float(500.0 * (residual @ residual))

    # 独立差分目标/非线性约束Jacobian，不复用被测解析motor Jacobian或优化缓存。
    oracle = minimize(
        numeric_objective, np.zeros(12), method="SLSQP", bounds=[(-1.5, 1.5)] * 12,
        constraints={"type": "ineq", "fun": numeric_constraint},
        options={"maxiter": 60, "ftol": 1e-10},
    )
    result = solver.solve(target, initial=actual)
    assert oracle.success and result.success
    assert len(result.tree_positions_rad) == 68
    assert -1e-9 <= result.tree_positions_rad[motor] <= 0.08 + 1e-9
    assert np.min(numeric_constraint(result.glove_positions_rad)) >= -1e-9
    np.testing.assert_allclose(result.glove_positions_rad, oracle.x, atol=2e-4)
    np.testing.assert_array_equal(solver._last_successful_anchor, actual)
    assert mechanism.evaluate_raw(_joints(actual)).motor_positions[motor] < -0.049


def test_motor_constrained_solver_preserves_global_q_box_and_outside_actual_feedback():
    model = _model()
    solver = GloveInverseSolver(model, regularization_weight=0.02)
    solver.set_motor_limits({model.landmarks.model.motor_names[0]: (-3.0, 3.0)})
    actual = np.zeros(12)
    actual[0] = 1.8
    target_values = actual.copy()
    target_values[0] = 1.6
    result = solver.solve(model.state(_joints(target_values)), initial=actual)
    assert result.success
    assert 1.49 < result.glove_positions_rad[0] <= 1.5
    np.testing.assert_array_equal(solver._last_successful_anchor, actual)


def test_infeasible_motor_limits_do_not_commit_a_falsely_successful_solution():
    model = _model()
    solver = GloveInverseSolver(model, max_nfev=1)
    # 电机范围合法，但与q盒约束下的真实CAD可达集无交集，不能靠setter猜测映射。
    solver.set_motor_limits({model.landmarks.model.motor_names[0]: (2.0, 3.0)})
    result = solver.solve(model.state(_joints(np.zeros(12))))
    assert not result.success
    assert not result.tree_positions_rad
    assert solver._last_successful_positions is None
    assert solver._last_successful_seed is None


def test_successful_optimizer_with_gross_motor_violation_is_rejected(monkeypatch):
    """优化器误报成功时，大幅越过电机限位仍不得生成可发送姿态。"""
    from types import SimpleNamespace
    import wujihand2_force_glove.bidirectional as bidirectional

    model = _model()
    solver = GloveInverseSolver(model)
    motor = model.landmarks.model.motor_names[0]
    solver.set_motor_limits({motor: (-0.1, 0.1)})
    delta = np.zeros(12)
    delta[0] = 0.4
    monkeypatch.setattr(
        bidirectional, "minimize",
        lambda *_args, **_kwargs: SimpleNamespace(success=True, x=delta, nfev=1),
    )
    result = solver.solve(model.state(_joints(np.zeros(12))))
    assert not result.success and not result.tree_positions_rad
    assert motor in result.failure_reason
    assert solver._last_successful_seed is None


def test_recorded_hand_pose_at_motor_boundary_produces_feasible_cad_tree():
    """相邻实测姿态在SLSQP微小电机边界误差下仍须得到真实可发的闭链解。"""
    from wujihand2_force_glove.real_hand.runtime import create_retargeter

    frames = np.array([
        [0.1141602441508376, -0.518103999115052, 0.19393482140009408, 0.08768142988872857,
         0.14472091306151072, -0.004362273931419676, 0.03849080231264553, 0.05031555657732201,
         0.2315718108026555, -0.008484213337287634, -0.05176778781632943, 0.001098385993483624,
         0.14815424762755222, 0.0005322718918614999, 0.03910854840737881, 0.04530845310282869,
         0.14527239666949066, -0.007641107249994229, 0.011731388468491533, -0.0022021105329943095],
        [0.10421795661231588, -0.5180054903030396, 0.19264700447556152, 0.07663700402186077,
         0.14509865319926848, -0.0043634544701540105, 0.013519809201406125, 0.03238005485588303,
         0.23261307855622815, -0.008499541564058875, -0.06875705339700657, -0.01213941902942094,
         0.1383466911248636, 0.0005291938333290266, 0.0002415374139070228, 0.026558058934444344,
         0.14687765364893207, -0.007858785354136217, -0.009259297250261206, -0.005534207261432143],
    ])
    bridge = create_retargeter(project_root=Path(__file__).resolve().parents[1])
    model = bridge.inverse.glove_model.landmarks.model
    limits = {
        name: (-np.pi / 2, np.pi / 2) if index < 2 else (0.0, np.pi / 2)
        for index, name in enumerate(model.motor_names)
    }
    bridge.inverse.set_motor_limits(limits)
    results = [bridge.hand2_to_glove(frame, timestamp_ns=stamp)
               for stamp, frame in enumerate(frames, 1)]
    assert results[0][0].success and results[1][0].success
    for result, target in results:
        actual = model.kinematics.evaluate_tree(result.tree_positions_rad)
        for name, (lower, upper) in limits.items():
            assert lower <= actual.motor_positions[name] <= upper
        actual_q = [actual.joint_positions[f"{name}_joint"] for name in GLOVE_JOINTS]
        np.testing.assert_allclose(result.glove_positions_rad, actual_q, rtol=0, atol=1e-10)
        points = bridge.inverse.glove_model.state_from_tree(result.tree_positions_rad)
        error_mm = np.linalg.norm(points.points_m - target.points_m, axis=1) * 1000
        assert result.task_error_max_mm == pytest.approx(np.max(error_mm), abs=1e-8)


def test_failed_or_unconverged_inverse_does_not_commit_successful_seed():
    model = _model()
    solver = GloveInverseSolver(model, regularization_weight=0.0)
    zero = model.state(_joints(np.zeros(12)), timestamp_ns=100)
    assert solver.solve(zero).success
    successful_positions = solver._last_successful_positions.copy()
    successful_seed = solver._last_successful_seed
    committed = model.landmarks.model.kinematics.current
    target = model.state(_joints(np.linspace(0.02, 0.08, 12)), timestamp_ns=101)
    solver.max_nfev = 1
    result = solver.solve(target)
    assert not result.success
    np.testing.assert_array_equal(solver._last_successful_positions, successful_positions)
    assert solver._last_successful_seed is successful_seed
    cold_solver = GloveInverseSolver(model, joint_bound_rad=6.0, max_nfev=1)
    with pytest.raises(ValueError, match="CAD闭环"):
        cold_solver.solve(target, initial=np.full(12, 6.0))
    np.testing.assert_array_equal(solver._last_successful_positions, successful_positions)
    assert solver._last_successful_seed is successful_seed
    assert model.landmarks.model.kinematics.current is committed
    assert model.landmarks.model._reference_pose is committed


def test_next_frame_uses_only_previous_successful_solution_and_pose(monkeypatch):
    model = _model()
    solver = GloveInverseSolver(model, regularization_weight=0.0)
    values = np.linspace(0.02, 0.06, 12)
    first = solver.solve(model.state(_joints(values), timestamp_ns=200))
    assert first.success
    previous_tree = dict(first.tree_positions_rad)
    previous_seed = solver._last_successful_seed
    evaluate = model.landmarks.evaluate_with_seed
    candidates = []
    seeds = []

    def counted(joints, *, seed=None):
        candidates.append(np.asarray([joints[name] for name in GLOVE_JOINTS]))
        seeds.append(seed)
        return evaluate(joints, seed=seed)

    target = model.state(_joints(values + 0.002), timestamp_ns=201)
    monkeypatch.setattr(model.landmarks, "evaluate_with_seed", counted)
    # 反馈仍停在回零噪声附近，数值初值应沿用上一成功候选，而非每帧从反馈重启。
    second = solver.solve(target, initial=np.full(12, 1e-11))
    assert second.success
    assert second.task_error_max_mm < 0.02
    np.testing.assert_array_equal(candidates[0], first.glove_positions_rad)
    assert seeds[0] is previous_seed
    assert dict(first.tree_positions_rad) == previous_tree
    assert dict(second.tree_positions_rad) != previous_tree


def test_near_zero_measured_initial_matches_exact_zero_without_trust_region_stall():
    target_model = _model()
    values = np.linspace(0.02, 0.07, 12)
    target = target_model.state(_joints(values), timestamp_ns=300)
    zero = GloveInverseSolver(_model()).solve(target, initial=np.zeros(12))
    measured = GloveInverseSolver(_model()).solve(target, initial=np.full(12, 1e-11))
    assert zero.success and measured.success
    assert measured.task_error_max_mm < 1.0
    assert np.linalg.norm(measured.glove_positions_rad) > 0.05
    np.testing.assert_allclose(measured.glove_positions_rad, zero.glove_positions_rad, atol=1e-7)


def test_default_regularization_does_not_overpower_metre_scale_large_reachable_task():
    target_model = _model()
    values = np.linspace(0.15, 0.35, 12)
    values[0] = 1.2
    target = target_model.state(_joints(values), timestamp_ns=400)
    result = GloveInverseSolver(_model()).solve(target, initial=np.full(12, 1e-11))
    assert result.success
    assert result.task_error_max_mm < 1.0


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_large_motor_forward_roundtrip_keeps_original_closure_and_committed_state(hand, sign):
    source = _model(hand).landmarks
    values = sign * np.linspace(0.15, 0.35, 12)
    values[0] = sign * 1.2
    pose = source.model.evaluate_raw(_joints(values))
    core = _model(hand).landmarks.model.kinematics
    committed = core.current
    forward = core.evaluate_motors(pose.motor_positions)
    actual = np.asarray([forward.joint_positions[f"{name}_joint"] for name in GLOVE_JOINTS])
    np.testing.assert_allclose(actual, values, atol=2e-7)
    assert forward.max_gap_m < 1e-7
    assert core.current is committed
    with pytest.raises(ValueError):
        core.evaluate_motors(dict.fromkeys(core.motor_names, 6.0), seed=forward)
    assert core.current is committed


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_local_newton_matches_full_cad_without_general_inner_optimizer(monkeypatch, hand, sign):
    import scipy.optimize as module

    source = _model(hand).landmarks
    core = source.model.kinematics
    values = sign * np.linspace(0.01, 0.04, 12)
    _, seed = source.evaluate_with_seed(_joints(values))
    targets = {f"{name}_joint": value for name, value in _joints(values + sign * 0.003).items()}
    expected = core.evaluate(targets, seed=seed)
    committed = core.current

    def forbid_optimizer(*args, **kwargs):
        raise AssertionError("局部闭链校正不得逐指重新创建通用least_squares")

    monkeypatch.setattr(module, "least_squares", forbid_optimizer)
    actual = core.evaluate(targets, seed=seed, local=True)
    global_result = core.evaluate(targets, seed=seed)
    np.testing.assert_allclose(
        [global_result.all_positions[name] for name in core.tree_names],
        [expected.all_positions[name] for name in core.tree_names],
        atol=2e-8,
    )
    np.testing.assert_allclose(
        [actual.all_positions[name] for name in core.tree_names],
        [expected.all_positions[name] for name in core.tree_names],
        atol=2e-8,
    )
    assert actual.max_gap_m < 1e-7
    assert core.current is committed


def test_active_cad_axis_bound_is_pulled_back_without_loosening_geometry():
    core = _model().landmarks.model.kinematics
    feasible, rejected = -0.69, -0.73
    seed = core.evaluate({"thumb_mcp_joint": feasible})
    committed = core.current
    # 用真实闭环二分找边界，而非把URDF树角或电机角假当真实指节角。
    for _ in range(25):
        midpoint = (feasible + rejected) * 0.5
        try:
            candidate = core.evaluate({"thumb_mcp_joint": midpoint}, seed=seed, local=True)
        except ValueError:
            rejected = midpoint
        else:
            feasible, seed = midpoint, candidate
    lower, upper = core.active_joint_bounds(seed)
    assert np.isfinite(lower[2])
    assert lower[2] <= seed.joint_positions["thumb_mcp_joint"] + 2e-7
    assert lower[2] < upper[2]
    with pytest.raises(ValueError):
        core.evaluate({"thumb_mcp_joint": lower[2] - 1e-4}, seed=seed, local=True)
    inward = core.evaluate({"thumb_mcp_joint": lower[2] + 1e-4}, seed=seed, local=True)
    assert inward.max_gap_m < 1e-7
    assert core.current is committed


def test_float_roundoff_at_local_radius_does_not_enter_full_recursive_solver(monkeypatch):
    source = _model().landmarks
    values = np.zeros(12)
    values[2] = -0.69
    _, seed = source.evaluate_with_seed(_joints(values))
    # 原始thumb装配边界附近，浮点0.04rad步略大于十进制0.04但仍在模型容差内。
    values[2] = np.nextafter(seed.joint_positions["thumb_mcp_joint"] - 0.04, -np.inf)
    distance = abs(values[2] - seed.joint_positions["thumb_mcp_joint"])
    assert 0.04 < distance < 0.04 + 1e-12
    evaluate = source.model.kinematics.evaluate
    modes = []

    def counted(targets, *, seed=None, local=False):
        modes.append(local)
        return evaluate(targets, seed=seed, local=local)

    monkeypatch.setattr(source.model.kinematics, "evaluate", counted)
    # 必须由真实机构拒绝越界；不能为提速放宽几何条件或返回伪造姿态。
    with pytest.raises(ValueError):
        source.evaluate_with_seed(_joints(values), seed=seed)
    assert modes == [True]


@pytest.mark.parametrize("hand", ["left", "right"])
def test_task_from_complete_tree_matches_independent_cad_fk_for_distinct_inputs(monkeypatch, hand):
    source = _model(hand)
    first = np.linspace(-0.025, 0.025, 12)
    second = np.linspace(0.08, 0.18, 12)
    second[0] = 1.2
    poses = [source.landmarks.model.evaluate_raw(_joints(values)) for values in (first, second)]
    expected = [_canonical_tree_points_oracle(source, pose.all_positions) for pose in poses]
    model = _model(hand)
    core = model.landmarks.model.kinematics
    committed = core.current

    def forbid_solve(*args, **kwargs):
        raise AssertionError("完整CAD树任务点不得重复调用指节逆解或电机正解")

    monkeypatch.setattr(core, "evaluate", forbid_solve)
    monkeypatch.setattr(core, "evaluate_motors", forbid_solve)
    states = []
    for index, (pose, points) in enumerate(zip(poses, expected)):
        positions = dict(pose.all_positions)
        original = dict(positions)
        frame = model.landmarks.evaluate_tree(positions)
        state = model.state_from_tree(positions, timestamp_ns=600 + index)
        np.testing.assert_allclose(state.points_m, points, atol=1e-11)
        np.testing.assert_allclose(
            frame.glove_pose.vector(),
            np.asarray([pose.joint_positions[f"{name}_joint"] for name in GLOVE_JOINTS]),
            atol=1e-12,
        )
        assert state.timestamp_ns == 600 + index
        assert positions == original
        states.append(state)
    assert np.max(np.abs(states[0].points_m - states[1].points_m)) > 0.01
    assert core.current is committed
    assert model.landmarks.model._reference_pose is committed


@pytest.mark.parametrize("invalid", ["missing", "unknown", "nan", "inf", "complex", "array", "bounds", "closure"])
def test_task_from_complete_tree_rejects_invalid_or_broken_cad_without_commit(invalid):
    model = _model()
    core = model.landmarks.model.kinematics
    committed = core.current
    positions = dict(committed.all_positions)
    name = core.motor_names[4]
    if invalid == "missing":
        positions.pop(name)
    elif invalid == "unknown":
        positions["unknown_joint"] = 0.0
    elif invalid in ("nan", "inf"):
        positions[name] = float(invalid)
    elif invalid == "complex":
        positions[name] = 1j
    elif invalid == "array":
        positions[name] = np.zeros(2)
    elif invalid == "bounds":
        bounded = next(name for name in core._solver.bounds if name in positions)
        positions[bounded] = core._solver.bounds[bounded][1] + 0.1
    else:
        positions[name] += 0.2
    with pytest.raises(ValueError):
        model.state_from_tree(positions, timestamp_ns=700)
    assert core.current is committed
    assert model.landmarks.model._reference_pose is committed


def test_hand2_feedback_fk_preserves_measured_abduction_without_unlocking_forward():
    from wujihand2_force_glove.real_hand.runtime import create_retargeter

    bridge = create_retargeter()
    zero = np.zeros(20)
    measured = zero.copy()
    measured[5] = 0.2
    baseline = bridge.hand2_model.state(zero, timestamp_ns=500)
    actual = bridge.hand2_model.state(measured, timestamp_ns=501)
    assert np.max(np.linalg.norm(actual.points_m[5:9] - baseline.points_m[5:9], axis=1)) > 1e-3
    np.testing.assert_allclose(actual.points_m[9:], baseline.points_m[9:], atol=1e-12)
    forward, _ = bridge.glove_to_hand2(_joints(np.zeros(12)), timestamp_ns=502)
    np.testing.assert_allclose(forward.positions_rad[[5, 9, 13, 17]], 0.0, atol=1e-12)


def _benchmark_inverse():
    """实CAD冷启动/滞后反馈/相邻帧；100ms或可达任务1mm不达标即非零退出。"""
    target_model = _model()
    solver = GloveInverseSolver(_model())
    motor_core = _model().landmarks.model.kinematics
    motor_seed = None
    times_ms = []
    failures = []
    for index in range(21):
        values = np.linspace(0.15, 0.35, 12)
        values[0] = 1.2
        values += 0.002 * index
        sampled_ns = monotonic_ns()
        # 仿真输入产生时记录时间戳，后续FK/IK不续期。
        target_model.landmarks.model.solve(_joints(values))
        target = target_model.state(_joints(values), timestamp_ns=sampled_ns)
        motor_start = perf_counter_ns()
        motor_seed = motor_core.evaluate_motors(
            target_model.landmarks.model.kinematics.current.motor_positions, seed=motor_seed
        )
        motor_ms = (perf_counter_ns() - motor_start) / 1e6
        motor_error = np.max(np.abs(
            np.asarray([motor_seed.joint_positions[f"{name}_joint"] for name in GLOVE_JOINTS]) - values
        ))
        start = perf_counter_ns()
        # 模拟人手阻抗/执行器滞后：反馈一直近零，不能靠initial=None掩盖反复重启。
        result = solver.solve(target, initial=np.full(12, 1e-11))
        elapsed_ms = (perf_counter_ns() - start) / 1e6
        times_ms.append(elapsed_ms)
        print(
            f"frame={index:02d} elapsed_ms={elapsed_ms:.3f} success={result.success} "
            f"nfev={result.iterations} max_mm={result.task_error_max_mm:.3f} "
            f"motor_fk_ms={motor_ms:.3f} motor_q_error_rad={motor_error:.3g} "
            f"rms_mm={result.task_error_rms_mm:.3f}"
        )
        if (
            not result.success or elapsed_ms > 100.0 or result.task_error_max_mm >= 1.0
            or motor_ms > 100.0 or motor_error > 2e-7
        ):
            failures.append(index)
    warm = np.asarray(times_ms[1:])
    print(
        f"cold_ms={times_ms[0]:.3f} warm_p50_ms={np.median(warm):.3f} "
        f"warm_p95_ms={np.percentile(warm, 95):.3f} warm_max_ms={warm.max():.3f} "
        f"deadline_failures={failures}"
    )
    return int(bool(failures))


def _benchmark_hand2_inverse():
    """官方正向骨架目标的冷逆解与真实反馈滞后；不把不同骨架误差伪称为可达。"""
    from wujihand2_force_glove.real_hand.runtime import create_retargeter

    bridge = create_retargeter()
    failures = []
    for index in range(6):
        sampled_ns = monotonic_ns()
        joints = dict.fromkeys(GLOVE_JOINTS, 0.0)
        joints["index_pip"] = 0.1 + index * 0.002
        joints["middle_pip"] = 0.05 + index * 0.001
        hand2, _ = bridge.glove_to_hand2(joints, timestamp_ns=sampled_ns)
        start = perf_counter_ns()
        result, _ = bridge.hand2_to_glove(
            hand2.positions_rad, initial=np.full(12, 1e-11), timestamp_ns=sampled_ns
        )
        elapsed_ms = (perf_counter_ns() - start) / 1e6
        print(
            f"hand2_frame={index:02d} elapsed_ms={elapsed_ms:.3f} success={result.success} "
            f"nfev={result.iterations} max_mm={result.task_error_max_mm:.3f} "
            f"q_norm_rad={np.linalg.norm(result.glove_positions_rad):.6f}"
        )
        if (
            not result.success or elapsed_ms > 100.0
            or np.linalg.norm(result.glove_positions_rad) < 1e-4
        ):
            failures.append(index)
    print(f"hand2_deadline_or_stall_failures={failures}")
    return int(bool(failures))


def _benchmark_complex_runtime():
    """现场literal Wuji姿态/真实限位/只读滞后的24个真实计时周期。"""
    import runpy
    from time import sleep

    from wujihand2_force_glove.real_hand.runtime import create_retargeter

    fixtures = runpy.run_path(str(Path(__file__).with_name("test_real_hand_runtime.py")))
    runtime, hand, glove, _, _ = fixtures["_complex_readonly_pair"](create_retargeter())
    # fixture使用完整精度真实双端样本，不以forward(glove)目标或四舍五入值覆盖。
    runtime._clock = hand.clock = glove.clock = monotonic_ns
    durations = []
    try:
        runtime.prepare()
        runtime.start()
        for index in range(24):
            started = perf_counter_ns()
            runtime.step()
            elapsed_s = (perf_counter_ns() - started) / 1e9
            durations.append(elapsed_s * 1000.0)
            print(f"complex_runtime_frame={index:02d} elapsed_ms={elapsed_s * 1000.0:.3f}")
            if elapsed_s >= 0.1:
                return 1
            sleep(max(0.0, 0.05 - elapsed_s))
        print(
            f"complex_runtime_p95_ms={np.percentile(durations, 95):.3f} "
            f"complex_runtime_max_ms={max(durations):.3f}"
        )
        return 0
    except (ValueError, TimeoutError, RuntimeError) as error:
        print(f"complex_runtime_failed_frame={len(durations)} error={error}")
        return 1
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(max(_benchmark_inverse(), _benchmark_hand2_inverse(), _benchmark_complex_runtime()))
