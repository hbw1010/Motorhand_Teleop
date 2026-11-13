"""有界遥测历史；接收时效与暂停显示相互独立。"""
from collections import deque
import json
import math
from pathlib import Path
from ..real_hand.curve_telemetry import decode_sample
from .quantities import TorqueConversion, enrich
from .series import SeriesBuffer
from .validation_signals import validate_extensions, prepare_signals


def finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def decode_packet(payload: bytes) -> dict:
    if not 0 < len(payload) <= 32768:
        raise ValueError('遥测包长度无效')
    p = json.loads(payload)
    if not isinstance(p, dict):
        raise ValueError('遥测必须为对象')
    for key in ('sequence','timestamp_ns','published_ns','wuji_received_ns','glove_received_ns',
                'cycle_started_ns','cycle_completed_ns','wuji_submit_ns','glove_submit_ns',
                'glove_output_source_ns','jacobian_age_ns','wuji_projection_received_ns',
                'wuji_feedback_received_ns','feedback_request_id','contact_evaluated_ns','impedance_planned_ns'):
        value=p.get(key)
        if value is not None and (type(value) is not int or not 0 <= value < 2**64):
            raise ValueError(f'{key}超出有效整数范围')
    if p.get('type') == 'teleop_status':
        if (type(p.get('schema_version')) is not int or p['schema_version'] != 1 or p.get('state') not in ('fault', 'stopped')
                or not isinstance(p.get('session_id'), str) or not 0 < len(p['session_id']) <= 128
                or type(p.get('timestamp_ns')) is not int or p['timestamp_ns'] < 0
                or not isinstance(p.get('message'), str) or len(p['message']) > 2048):
            raise ValueError('状态事件无效')
        return p
    p = decode_sample(payload)
    validate_extensions(p)
    for key,count in (('glove_requested_torque_nm',12),('wuji_filtered_current_a',20),('wuji_to_glove_torque_matrix',240),
                      ('glove_friction_torque_nm',12),('wuji_projection_current_a',20),
                      ('glove_driver_pd_torque_nm',12),('glove_driver_target_rad',12),
                      ('glove_driver_kp',12),('glove_driver_kd',12),('glove_feedforward_torque_nm',12),
                      ('wuji_feedback_current_a',20),('wuji_feedback_position_rad',20),
                      ('wuji_feedback_velocity_rad_s',20),('wuji_feedback_error_rad',20)):
        values=p.get(key)
        if values is not None and (not isinstance(values,list) or len(values)!=count
                or any(v is not None and not finite_number(v) for v in values)):
            raise ValueError(f'{key}维数或数值无效')
    for key in ('glove_driver_estimate_received_ns','glove_driver_velocity_received_ns','glove_current_received_ns','glove_proxy_source_ns'):
        stamps=p.get(key)
        if stamps is not None and (not isinstance(stamps,list) or len(stamps)!=12
                or any(t is not None and (type(t) is not int or not 0<=t<2**64) for t in stamps)):
            raise ValueError(f'{key}必须为12路真实接收时刻或缺失值')
    if p.get('feedback_control_mode','torque') not in ('torque','driver_impedance'):
        raise ValueError('未知反馈控制方式')
    if p.get('glove_target_semantics') not in (None,'quantized_feedforward','nominal_total_pd_at_submission'):
        raise ValueError('未知手套名义输出含义')
    if p.get('hand', 'right') != 'right':
        raise ValueError('首版仅支持右手模型')
    tree = p.get('glove_tree_rad')
    if tree is not None:
        if (p.get('glove_tree_order') != 've33_closure_independent_dependent'
                or not isinstance(tree, list) or len(tree) != 68
                or any(not finite_number(x) for x in tree)):
            raise ValueError('手套完整树必须是规定顺序的68维有限弧度角')
    gap = p.get('glove_closure_gap_m')
    if gap is not None and (not finite_number(gap) or gap < 0):
        raise ValueError('闭环间隙无效')
    return p


class MonitorState:
    def __init__(self, limit=3000, torque_config=None):
        self.torque_config=torque_config or TorqueConversion.unavailable()
        self.history = deque(maxlen=limit)
        self.latest = None
        self.terminal = None
        self.session = None
        self.retired = deque(maxlen=128)
        self.received_ns = None
        self.paused = False
        self.snapshot = []
        self._snapshot_terminal = None
        self._live_series = {}
        self._snapshot_series = {}
        self._view_revision = 0
        self.newest_ns = -1

    def ingest(self, p: dict, now_ns: int) -> bool:
        session = p['session_id']
        if session in self.retired:
            return False
        if self.session != session:
            if p['timestamp_ns'] <= self.newest_ns:
                return False
            if self.session is not None:
                self.retired.append(self.session)
            self.history.clear()
            self._live_series.clear()
            self.latest = self.terminal = None
            self.session = session
        if p.get('type') == 'teleop_status':
            if p['timestamp_ns'] < self.newest_ns:
                return False
            changed = self.terminal != p
            self.terminal = p
            self.newest_ns = p['timestamp_ns']
            if changed and not self.paused:
                self._view_revision += 1
            return True
        if self.terminal is not None:
            return False
        if self.latest is not None and (p['sequence'] <= self.latest['sequence']
                                       or p['timestamp_ns'] < self.latest['timestamp_ns']):
            return False
        p=prepare_signals(enrich(p,self.torque_config))
        self.latest = p
        self.history.append(p)
        for series in self._live_series.values():
            series.append(p)
        self.received_ns = now_ns
        self.newest_ns = p['timestamp_ns']
        if not self.paused:
            self._view_revision += 1
        return True

    def set_paused(self, paused: bool):
        if self.paused == paused:
            return
        self.paused = paused
        self.snapshot = list(self.history) if paused else []
        self._snapshot_terminal = self.terminal if paused else None
        self._snapshot_series.clear()
        self._view_revision += 1

    def series_for(self, trace):
        key = (trace.field, trace.index, trace.factor, trace.time_field, trace.strict_time)
        cache = self._snapshot_series if self.paused else self._live_series
        if key not in cache:
            series = SeriesBuffer(trace, self.history.maxlen)
            for frame in self.snapshot if self.paused else self.history:
                series.append(frame)
            cache[key] = series
        return cache[key]

    @property
    def view_key(self):
        return self._view_revision

    @property
    def frames(self):
        return self.snapshot if self.paused else list(self.history)

    @property
    def display_frame(self):
        return (self.snapshot[-1] if self.snapshot else None) if self.paused else self.latest

    @property
    def display_terminal(self):
        return self._snapshot_terminal if self.paused else self.terminal

    def stale(self, now_ns):
        if self.latest is None or self.received_ns is None or self.terminal is not None:
            return True
        stamps = [self.received_ns, self.latest['wuji_received_ns'], self.latest['glove_received_ns']]
        return any(t <= 0 or t > now_ns or now_ns - t > 300_000_000 for t in stamps)

    def save(self, path):
        with Path(path).open('w', encoding='utf-8') as f:
            for p in self.frames:
                f.write(json.dumps(p, ensure_ascii=False, allow_nan=False)+'\n')
            if self.display_terminal is not None:
                f.write(json.dumps(self.display_terminal, ensure_ascii=False)+'\n')
