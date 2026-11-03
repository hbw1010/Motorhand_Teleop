from __future__ import annotations

import multiprocessing as mp
import pickle
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class InverseWorkerResult:
    request_id: int
    received_ns: int
    glove_positions_rad: np.ndarray | None
    tree_positions_rad: dict[str, float] | None
    success: bool
    error: str | None = None


def _frame(positions_rad, received_ns):
    positions = np.asarray(positions_rad, dtype=np.float64)
    if positions.shape != (20,) or not np.isfinite(positions).all():
        raise ValueError("Wuji反馈必须是有限20维rad")
    if isinstance(received_ns, (bool, np.bool_)) or not isinstance(received_ns, (int, np.integer)):
        raise ValueError("received_ns必须是整数")
    stamp = int(received_ns)
    if not -(1 << 63) <= stamp < (1 << 63):
        raise ValueError("received_ns超出int64范围")
    return positions.copy(), stamp


def _run_worker(settings, warmup, ready, input_lock, input_values, input_id,
                input_stamp, output_lock, output_bytes, output_size, failed):
    try:
        from .runtime import create_retargeter

        hand, glove_hand, assets, config, limits = settings
        bridge = create_retargeter(hand=hand, glove_hand=glove_hand,
                                   asset_root=assets, config_path=config)
        bridge.inverse.set_motor_limits(limits)
        if warmup is not None:
            result, _ = bridge.hand2_to_glove(warmup[0], timestamp_ns=warmup[1])
            if not result.success:
                raise ValueError(f"实测Hand2姿态的手套逆解预热未收敛：{result.failure_reason}")
        ready.send(None)
    except BaseException:
        try:
            ready.send(traceback.format_exc())
        finally:
            ready.close()
        return
    ready.close()

    processed = 0
    while True:
        with input_lock:
            request_id = input_id.value
            new_request = request_id != processed
            if new_request:
                positions = np.frombuffer(input_values, dtype=np.float64).copy()
                stamp = input_stamp.value
                processed = request_id
        if not new_request:
            time.sleep(0.002)
            continue
        try:
            result, _ = bridge.hand2_to_glove(positions, timestamp_ns=stamp)
            if result.success:
                item = InverseWorkerResult(request_id, stamp, result.glove_positions_rad.copy(),
                                           dict(result.tree_positions_rad), True)
            else:
                item = InverseWorkerResult(request_id, stamp, None, None, False,
                                           f"实测Hand2姿态的手套逆解未收敛：{result.failure_reason}")
        except BaseException:
            item = InverseWorkerResult(request_id, stamp, None, None, False, traceback.format_exc())
        payload = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
        with output_lock:
            if len(payload) > len(output_bytes):
                item = InverseWorkerResult(request_id, stamp, None, None, False, "逆解结果超过共享缓冲区")
                payload = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
            output_bytes[:len(payload)] = payload
            output_size.value = len(payload)
            if not item.success:
                failed.value = True
        # 失败必须保持可见；停止处理新帧，但保持进程存活直到关闭。
        if not item.success:
            while True:
                time.sleep(60)


class InverseWorker:
    """单spawn进程的最新帧逆解；不访问硬件，不更新源时间戳。

    start可同步预热，预热帧不会出现在latest。submit和latest不等待求解：
    输入槽忙时submit抛BlockingIOError，latest在输出槽忙时返回None。
    求解失败作为粘性结果返回，调用方必须停止使用旧成功结果。
    latest可重复返回同一结果；消费者负责按源时间戳检查时效。
    """

    def __init__(self, hand: str, glove_hand: str, asset_root: str | Path | None,
                 config_path: str | Path | None, motor_limits: Mapping[str, tuple[float, float]]):
        self._settings = (hand, glove_hand, asset_root, config_path, dict(motor_limits))
        self._ctx = mp.get_context("spawn")
        self._process = None
        self._closed = False
        self._next_id = 0
        self._input_lock = self._ctx.Lock()
        self._input_values = self._ctx.RawArray("d", 20)
        self._input_id = self._ctx.RawValue("q", 0)
        self._input_stamp = self._ctx.RawValue("q", 0)
        self._output_lock = self._ctx.Lock()
        self._output_bytes = self._ctx.RawArray("B", 65536)
        self._output_size = self._ctx.RawValue("i", 0)
        self._failed = self._ctx.RawValue("b", False)

    def start(self, positions_rad=None, received_ns=None):
        if self._closed or self._process is not None:
            raise RuntimeError("逆解worker已关闭或已启动")
        if (positions_rad is None) != (received_ns is None):
            raise ValueError("预热必须同时提供反馈姿态和接收时间")
        warmup = None if positions_rad is None else _frame(positions_rad, received_ns)
        reader, writer = self._ctx.Pipe(duplex=False)
        process = self._ctx.Process(target=_run_worker, args=(
            self._settings, warmup, writer, self._input_lock, self._input_values,
            self._input_id, self._input_stamp, self._output_lock,
            self._output_bytes, self._output_size, self._failed,
        ), daemon=True)
        self._process = process
        try:
            process.start()
            writer.close()
            if not reader.poll(120):
                raise RuntimeError("逆解worker启动或预热超时")
            try:
                error = reader.recv()
            except EOFError as exc:
                raise RuntimeError("逆解worker启动时异常退出") from exc
            if error is not None:
                raise RuntimeError(f"逆解worker启动失败：\n{error}")
        except BaseException:
            self.close()
            raise
        finally:
            reader.close()
            writer.close()

    def _check_running(self):
        if self._closed or self._process is None:
            raise RuntimeError("逆解worker未启动或已关闭")
        if not self._process.is_alive():
            raise RuntimeError(f"逆解worker异常退出：exitcode={self._process.exitcode}")

    def submit(self, positions_rad, received_ns) -> int:
        self._check_running()
        positions, stamp = _frame(positions_rad, received_ns)
        if self._failed.value:
            raise RuntimeError("逆解worker已失败，请读取latest错误")
        if not self._input_lock.acquire(False):
            raise BlockingIOError("逆解worker输入槽正忙")
        try:
            self._next_id += 1
            np.frombuffer(self._input_values, dtype=np.float64)[:] = positions
            self._input_id.value = self._next_id
            self._input_stamp.value = stamp
            return self._next_id
        finally:
            self._input_lock.release()

    def latest(self) -> InverseWorkerResult | None:
        self._check_running()
        if not self._output_lock.acquire(False):
            return None
        try:
            size = self._output_size.value
            return pickle.loads(bytes(self._output_bytes[:size])) if size else None
        finally:
            self._output_lock.release()

    def close(self):
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is None or process.pid is None:
            return
        # 子进程可能已在持有共享锁时异常退出；不要再触碰其事件锁。
        # worker仅做纯计算、不持有硬件，终止后回收进程不会改变设备输出。
        if process.is_alive():
            process.terminate()
        process.join(1.0)
        if process.is_alive():
            process.kill()
            process.join()
        process.close()
