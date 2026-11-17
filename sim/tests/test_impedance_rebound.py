"""回撤时阻抗连续性、总输出方向及真正离开接触面的回归。"""
import math

import numpy as np
import pytest

from glove_sim.config import load_config
from glove_sim.impedance import DriverImpedancePlanner, VelocityEstimator


def planner():
    glove = load_config()["glove"]
    return DriverImpedancePlanner(glove["driver_impedance"], glove["torque_limit_nm"])


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("frequency", [5, 8])
def test_small_rebound_inside_contact_keeps_resistance(sign, frequency):
    controller = planner()
    velocity = VelocityEstimator(12.0)
    q = np.zeros(12)
    stiffness = []
    for tick in range(400):
        depth = 0.005 + 0.004 * math.sin(2 * math.pi * frequency * tick * 0.01)
        q[4] = 0.3 + sign * depth
        dq = velocity.update(q, 0.01)
        command = controller.plan(q, dq, {4: (sign, 0.3, 55.6)}, ramp=1.0)[4]
        if tick >= 100:
            stiffness.append(command.k)
            assert sign * command.torque(q[4], dq[4]) < 0.0
            predicted_q = q[4] + (dq[4] * 0.02 if sign * dq[4] < 0 else 0.0)
            assert sign * command.torque(predicted_q, dq[4]) <= 1e-12
    assert min(stiffness) > 0.0, "仍在接触面内的小幅回弹不应反复清零刚度"


@pytest.mark.parametrize("sign", [1, -1])
def test_rebound_damping_cannot_reverse_total_torque(sign):
    controller = planner()
    q = np.zeros(12)
    dq = np.zeros(12)
    q[4], dq[4] = 0.3 + sign * 0.01, sign * 0.1
    controller.plan(q, dq, {4: (sign, 0.3, 55.6)})  # 按入，激活阻尼。
    q[4], dq[4] = 0.3 + sign * 0.0025, -sign * 0.1
    command = controller.plan(q, dq, {4: (sign, 0.3, 55.6)})[4]
    assert command.k > 0.0
    assert sign * command.torque(q[4], dq[4]) < 0.0
    assert sign * command.torque(q[4] + dq[4] * 0.02, dq[4]) <= 1e-12


@pytest.mark.parametrize("sign", [1, -1])
def test_fast_withdrawal_softens_then_releases_at_surface(sign):
    controller = planner()
    q = np.zeros(12)
    dq = np.zeros(12)
    q[4], dq[4] = 0.3 + sign * 0.01, -sign * 0.6
    command = controller.plan(q, dq, {4: (sign, 0.3, 40.0)})[4]
    assert 0.0 < command.k < 40.0
    assert -0.4 < sign * command.torque(q[4], dq[4]) < 0.0
    assert sign * command.torque(q[4] + dq[4] * 0.02, dq[4]) <= 1e-12
    for depth in (0.0, -0.001):
        q[4] = 0.3 + sign * depth
        command = controller.plan(q, dq, {4: (sign, 0.3, 40.0)})[4]
        assert command.k == command.b == command.tau_ff == 0.0


def test_softened_contact_still_respects_spring_budget():
    controller = planner()
    q = np.zeros(12)
    dq = np.zeros(12)
    q[4] = 1.5
    dq[4] = -0.1
    command = controller.plan(q, dq, {4: (1, 0.0, 80.0)})[4]
    assert 0.0 < command.k <= 80.0
    assert -8.5 <= command.k * (command.target_rad - q[4]) < 0.0


def test_legacy_release_mode_still_releases_before_crossing():
    glove = load_config()["glove"]
    settings = dict(glove["driver_impedance"], rebound_mode="release")
    controller = DriverImpedancePlanner(settings, glove["torque_limit_nm"])
    q = np.zeros(12)
    dq = np.zeros(12)
    q[4], dq[4] = 0.31, -0.6
    command = controller.plan(q, dq, {4: (1, 0.3, 40.0)})[4]
    assert command.k == command.b == command.tau_ff == 0.0
