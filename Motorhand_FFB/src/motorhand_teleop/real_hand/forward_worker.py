"""独立进程执行原有正向重定向，避免求解器占用串口线程的 GIL。"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import time
from typing import Mapping

import numpy as np

from ..domain import GLOVE_JOINTS, RetargetedPose


def _run_worker(connection, settings) -> None:
    try:
        from .runtime import create_retargeter

        hand, glove_hand, asset_root, config_path = settings
        bridge = create_retargeter(
            hand=hand, glove_hand=glove_hand,
            asset_root=asset_root, config_path=config_path,
        )
    except BaseException as error:
        try:
            connection.send(("startup_error", f"{type(error).__name__}: {error}"))
        finally:
            connection.close()
        return

    try:
        connection.send(("ready",))
        while True:
            try:
                request = connection.recv()
            except EOFError:
                break
            if request is None:
                break
            request_id, source_ns, values = request
            try:
                joints = dict(zip(GLOVE_JOINTS, values))
                pose, _ = bridge.glove_to_hand2(joints, timestamp_ns=source_ns)
                connection.send((
                    "ok", request_id, source_ns, pose.positions_rad,
                    dict(pose.diagnostics),
                ))
            except BaseException as error:
                connection.send((
                    "error", request_id, source_ns,
                    f"{type(error).__name__}: {error}",
                ))
                break
    finally:
        connection.close()


class ForwardRetargetWorker:
    """单进程同步正向求解；每次结果对应唯一源帧和不可续期的原始期限。"""

    def __init__(
        self, *, hand: str, glove_hand: str,
        asset_root: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self._settings = (hand, glove_hand, asset_root, config_path)
        self._ctx = mp.get_context("spawn")
        self._process = None
        self._connection = None
        self._closed = False
        self._failed = False
        self._request_id = 0

    def start(self) -> None:
        if self._closed or self._process is not None:
            raise RuntimeError("正向求解进程已关闭或已启动")
        parent, child = self._ctx.Pipe()
        process = self._ctx.Process(
            target=_run_worker, args=(child, self._settings), daemon=True,
        )
        self._process, self._connection = process, parent
        try:
            process.start()
            child.close()
            if not parent.poll(30.0):
                raise TimeoutError("正向求解进程启动超时")
            message = parent.recv()
            if message != ("ready",):
                raise RuntimeError(f"正向求解进程启动失败：{message[1]}")
        except BaseException:
            child.close()
            self.close()
            raise

    def check_ready(self) -> None:
        if self._closed or self._process is None or self._connection is None:
            raise RuntimeError("正向求解进程未启动或已关闭")
        if self._failed:
            raise RuntimeError("正向求解进程已故障锁存，必须重新建立会话")
        if not self._process.is_alive():
            self._failed = True
            raise RuntimeError(f"正向求解进程异常退出：exitcode={self._process.exitcode}")

    def solve(
        self, joints: Mapping[str, float], *, source_ns: int,
        deadline_ns: int | None,
    ) -> RetargetedPose:
        self.check_ready()
        if not isinstance(joints, Mapping) or set(joints) != set(GLOVE_JOINTS):
            raise ValueError("手套姿态必须覆盖12个真实指节")
        if isinstance(source_ns, bool) or not isinstance(source_ns, int) or source_ns <= 0:
            raise ValueError("正向求解必须保留手套真实接收时间")
        if deadline_ns is not None and (isinstance(deadline_ns, bool) or not isinstance(deadline_ns, int)):
            raise ValueError("正向求解期限必须为单调时钟整数纳秒")
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            raise TimeoutError("正向求解前原始命令或反馈期限已过")
        values = np.asarray([joints[name] for name in GLOVE_JOINTS], dtype=np.float64)
        if values.shape != (12,) or not np.isfinite(values).all():
            raise ValueError("手套姿态必须是有限的12维真实指节")
        self._request_id += 1
        request_id = self._request_id
        try:
            self._connection.send((request_id, source_ns, values))
            timeout = 30.0 if deadline_ns is None else max(
                0.0, (deadline_ns - time.monotonic_ns()) * 1e-9,
            )
            if not self._connection.poll(timeout):
                self._failed = True
                raise TimeoutError("正向求解超过原始命令或反馈期限")
            message = self._connection.recv()
        except TimeoutError:
            self._failed = True
            raise
        except (BrokenPipeError, EOFError, OSError) as error:
            self._failed = True
            raise RuntimeError("正向求解进程通信中断") from error
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            self._failed = True
            raise TimeoutError("正向求解超过原始命令或反馈期限")
        if len(message) < 4 or message[1:3] != (request_id, source_ns):
            self._failed = True
            raise RuntimeError("正向求解结果与原始帧不匹配")
        if message[0] != "ok":
            self._failed = True
            raise RuntimeError(f"正向求解进程失败：{message[3]}")
        if len(message) != 5:
            self._failed = True
            raise RuntimeError("正向求解结果格式非法")
        return RetargetedPose(message[3], None, diagnostics=message[4])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process, connection = self._process, self._connection
        if process is None:
            return
        if process.pid is not None and process.is_alive():
            if not self._failed and connection is not None:
                try:
                    connection.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
                process.join(1.0)
            if process.is_alive():
                process.terminate()
                process.join(1.0)
            if process.is_alive():
                process.kill()
                process.join()
        if connection is not None:
            connection.close()
        if process.pid is not None:
            process.close()
