"""保留全部样本的数值环缓存，以及按实际事件计数的滚动帧率。"""
from collections import deque
import math
import time

import numpy as np


class SeriesBuffer:
    """将字段提取移到接收阶段；返回的连续快照不受后续环写入影响。"""

    def __init__(self, trace, capacity):
        if capacity < 0:
            raise ValueError('曲线容量不能为负数')
        self.trace = trace
        self.capacity = capacity
        self._stamps = np.empty(capacity, dtype=np.uint64)
        self._values = np.empty(capacity, dtype=np.float64)
        self._breaks = np.empty(capacity, dtype=np.uint8)
        self._next = 0
        self._count = 0
        self._last_session = None
        self._last_sequence = None
        self._snapshot = None

    def append(self, frame):
        if self.capacity == 0:
            return
        session = frame.get('session_id')
        sequence = frame.get('sequence')
        discontinuous = (self._count == 0 or session != self._last_session
                         or (sequence is not None and self._last_sequence is not None
                             and sequence != self._last_sequence + 1))
        value = self.trace.value(frame)
        self._stamps[self._next] = self.trace.stamp(frame)
        self._values[self._next] = np.nan if value is None else value
        self._breaks[self._next] = discontinuous
        self._next = (self._next + 1) % self.capacity
        self._count = min(self._count + 1, self.capacity)
        self._last_session = session
        self._last_sequence = sequence
        self._snapshot = None

    def _arrays(self):
        if self._snapshot is None:
            start = self._next if self._count == self.capacity else 0
            arrays = []
            for ring in (self._stamps, self._values, self._breaks):
                if start:
                    data = np.concatenate((ring[start:], ring[:start]))
                else:
                    data = ring[:self._count].copy()
                arrays.append(data)
            if self._count:
                arrays[2][0] = 1
            for data in arrays:
                data.flags.writeable = False
            self._snapshot = tuple(arrays)
        return self._snapshot

    def arrays(self):
        """按接收顺序返回 uint64 时间戳与 float64 数值；缺失值保留为 NaN。"""
        stamps, values, _ = self._arrays()
        return stamps, values

    def breaks(self):
        """标记需要从前一样本断开的位置，首样本总是新线段。"""
        return self._arrays()[2]


class FrameRate:
    """统计最近 window_s 秒的实际事件；重复帧键不构成新事件。"""

    def __init__(self, window_s=1.0):
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError('帧率窗口必须为正有限秒数')
        self.window_s = window_s
        self._events = deque()
        self._keys = set()
        self._last_key = None

    def _expire(self, now):
        cutoff = now - self.window_s
        while self._events and self._events[0][0] <= cutoff:
            _, key = self._events.popleft()
            if key is not None:
                self._keys.remove(key)

    def mark(self, now=None, key=None):
        now = time.monotonic() if now is None else now
        self._expire(now)
        if key is not None:
            # 窗口内重复事件去重；窗口过期后也不能把仍在显示的末帧重新计数。
            if key == self._last_key or key in self._keys:
                return
            self._keys.add(key)
            self._last_key = key
        self._events.append((now, key))

    def value(self, now=None):
        now = time.monotonic() if now is None else now
        self._expire(now)
        return len(self._events) / self.window_s
