from pathlib import Path
import time

import numpy as np
import pytest

from wujihand2_force_glove.domain import GLOVE_JOINTS
from wujihand2_force_glove.real_hand.inverse_worker import InverseWorker
from wujihand2_force_glove.real_hand.runtime import create_retargeter


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def numerical_inputs():
    bridge = create_retargeter(project_root=ROOT)
    joints = dict.fromkeys(GLOVE_JOINTS, 0.0)
    first = bridge.glove_to_hand2(joints)[0].positions_rad.copy()
    joints["index_pip"] = 0.4
    joints["middle_pip"] = 0.2
    second = bridge.glove_to_hand2(joints)[0].positions_rad.copy()
    names = bridge.inverse.glove_model.landmarks.model.motor_names
    return first, second, names


def worker(limits):
    return InverseWorker("right", "right", ROOT / "assets/hand_motor_ve33",
                         ROOT / "config/retargeting/official/right.yaml", limits)


def submit(instance, positions, stamp):
    deadline = time.monotonic() + 10
    while True:
        try:
            return instance.submit(positions, stamp)
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.001)


def result_for(instance, request_id):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        result = instance.latest()
        if result is not None and (result.request_id >= request_id or not result.success):
            return result
        time.sleep(0.005)
    pytest.fail("逆解worker未返回结果")


def test_real_warmup_latest_frames_and_close(numerical_inputs):
    first, second, names = numerical_inputs
    instance = worker({name: (-3.0, 3.0) for name in names})
    try:
        instance.start(first, 100)
        assert instance.latest() is None  # 预热不构成命令
        first_id = submit(instance, first, 123456789)
        a = result_for(instance, first_id)
        assert a.success, a.error
        assert a.request_id == first_id and a.received_ns == 123456789
        assert a.glove_positions_rad.shape == (12,)
        assert len(a.tree_positions_rad) == 68
        # 多个待算帧只保留最新值；无论中间帧是否已开始，最终结果必须对应最后一帧。
        submit(instance, first, 123456790)
        last_id = submit(instance, second, 123456791)
        b = result_for(instance, last_id)
        assert b.success, b.error
        assert (b.request_id, b.received_ns) == (last_id, 123456791)
        assert len(b.tree_positions_rad) == 68
        assert not np.allclose(a.glove_positions_rad, b.glove_positions_rad, atol=1e-4)
    finally:
        instance.close()
        instance.close()
    with pytest.raises(RuntimeError):
        instance.latest()


def test_real_solver_failure_is_sticky(numerical_inputs):
    first, second, names = numerical_inputs
    # 有效但不可达的真实电机范围使真实逆解失败；不得用结果伪装成功。
    instance = worker({name: (100.0, 101.0) for name in names})
    try:
        instance.start()
        request_id = submit(instance, first, 998)
        failure = result_for(instance, request_id)
        assert not failure.success and failure.error
        assert (failure.request_id, failure.received_ns) == (request_id, 998)
        assert failure.glove_positions_rad is None and failure.tree_positions_rad is None
        with pytest.raises(RuntimeError, match="已失败"):
            submit(instance, second, 999)
        assert instance.latest().error == failure.error
    finally:
        instance.close()


def test_spawn_start_failure_and_unexpected_exit_are_visible(numerical_inputs):
    first, _, names = numerical_inputs
    broken = worker({"not_a_real_motor": (-1.0, 1.0)})
    with pytest.raises(RuntimeError, match="启动失败"):
        broken.start(first, 12)
    broken.close()

    instance = worker({name: (-3.0, 3.0) for name in names})
    try:
        instance.start()
        instance._process.terminate()
        instance._process.join(10)
        with pytest.raises(RuntimeError, match="异常退出"):
            instance.latest()
        with pytest.raises(RuntimeError, match="异常退出"):
            instance.submit(first, 12)
    finally:
        instance.close()
