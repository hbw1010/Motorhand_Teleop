"""六组视图从解码、冻结快照到异步分析和实际Qt绘制的集成边界。"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from motorhand_teleop.teleop_monitor.demo import validation_sample
from motorhand_teleop.teleop_monitor.plots import Trace, build_pages
from motorhand_teleop.teleop_monitor.quantities import TorqueConversion
from motorhand_teleop.teleop_monitor.state import MonitorState, decode_packet
from motorhand_teleop.teleop_monitor.validation_analysis import analyze_axis
from motorhand_teleop.teleop_monitor.validation_view import AnalysisBroker


BASE_NS = 1_000_000_000_000


def packet(sequence, *, session='validation-a'):
    sample = validation_sample(4. + sequence * .02, BASE_NS + sequence * 20_000_000)
    sample.update(sequence=sequence, session_id=session)
    return decode_packet(json.dumps(sample).encode())


def test_strict_time_trace_does_not_reuse_fallback_cache():
    state = MonitorState()
    sample = packet(0)
    sample['glove_current_received_ns'] = None
    state.ingest(sample, BASE_NS)
    fallback = Trace('glove_current_a', 0, '兼容回退', '#fff', time_field='glove_current_received_ns')
    strict = Trace('glove_current_a', 0, '真实电流时刻', '#fff',
                   time_field='glove_current_received_ns', strict_time=True)
    assert np.isfinite(state.series_for(fallback).arrays()[1][0])
    assert np.isnan(state.series_for(strict).arrays()[1][0])
    assert state.series_for(strict) is not state.series_for(fallback)


@pytest.mark.parametrize(('index', 'actual', 'proxy', 'direction'), [
    (0, .04, .1, -1), (1, .21, .15, 1),
])
def test_decoded_raw_coordinates_and_nominal_units_reach_analysis(index, actual, proxy, direction):
    state = MonitorState(torque_config=TorqueConversion.load())
    sample = packet(0)
    state.ingest(sample, BASE_NS)
    result = analyze_axis(state.frames, index)
    point = result['points'][0]
    assert point['actual_angle_rad'] == pytest.approx(actual)
    assert point['proxy_rad'] == pytest.approx(proxy)
    assert point['angle_excess_rad'] == pytest.approx(.06)
    assert point['blocking_direction'] == direction
    assert point['response'] > 0
    assert point['phase'] == 'press'
    assert point['time_ns'] == BASE_NS - 1_000_000
    assert point['alignment_ms'] == 1.
    assert result['response_unit'] == 'A'
    assert result['torque_error'] is None
    assert state.display_frame['glove_validated_torque_nm'] == [None] * 12
    assert state.display_frame['validation_timing_ms'] == pytest.approx([.2, .3, .5, 1.])


def test_all_validation_traces_require_their_real_source_timestamp():
    sample = packet(0)
    state = MonitorState()
    state.ingest(sample, BASE_NS)
    prepared = state.display_frame
    tested = 0
    for page in build_pages(1).values():
        for spec in page:
            for trace in spec.traces + spec.states:
                if trace.value(prepared) is None:
                    continue
                missing = dict(prepared)
                missing[trace.time_field] = None
                assert trace.value(missing) is None, (trace.field, trace.time_field)
                tested += 1
    assert tested > 30


def test_frozen_analysis_and_export_use_same_frames_after_new_live_session(tmp_path):
    state = MonitorState(torque_config=TorqueConversion.load())
    for sequence in range(3):
        sample = packet(sequence)
        state.ingest(sample, sample['timestamp_ns'])
    state.set_paused(True)
    revision = state.view_key
    frozen = json.dumps(state.frames, sort_keys=True)
    bounds = (BASE_NS + 18_000_000, BASE_NS + 20_000_000)
    newer = packet(10, session='validation-b')
    state.ingest(newer, newer['timestamp_ns'])
    broker = AnalysisBroker()
    try:
        assert broker.get(state.frames, 0, bounds, 'frozen-a', revision, force=True) is None
        broker.future.result(timeout=3)
        result = broker.get(state.frames, 0, bounds, 'frozen-a', revision, force=True)
        assert len(result['points']) == 1
        assert result['points'][0]['time_ns'] == BASE_NS + 19_000_000
        assert state.view_key == revision
        assert json.dumps(state.frames, sort_keys=True) == frozen
        destination = tmp_path / 'frozen.jsonl'
        state.save(destination)
        exported = [json.loads(line) for line in destination.read_text().splitlines()]
        assert analyze_axis(exported, 0, bounds) == result
        assert state.latest['session_id'] == 'validation-b'
        assert state.display_frame['session_id'] == 'validation-a'
    finally:
        broker.close()


def test_async_broker_cannot_return_previous_axis_result_for_new_scope():
    frames = [packet(0)]
    bounds = (BASE_NS - 10_000_000, BASE_NS)
    broker = AnalysisBroker()
    try:
        broker.get(frames, 0, bounds, ('validation-a', 0), 1, force=True)
        broker.future.result(timeout=3)
        first = broker.get(frames, 0, bounds, ('validation-a', 0), 1, force=True)
        assert first['points'][0]['actual_angle_rad'] == pytest.approx(.04)
        assert broker.get(frames, 1, bounds, ('validation-a', 1), 1, force=True) is None
        broker.future.result(timeout=3)
        second = broker.get(frames, 1, bounds, ('validation-a', 1), 1, force=True)
        assert second['points'][0]['actual_angle_rad'] == pytest.approx(.21)
    finally:
        broker.close()


def test_window_latches_analysis_for_summary_and_real_scatter_paint():
    script = r'''
import time
from PyQt5.QtWidgets import QApplication
from motorhand_teleop.teleop_monitor.demo import validation_sample
from motorhand_teleop.teleop_monitor.window import MonitorWindow
from motorhand_teleop.teleop_monitor.validation_view import ScatterCurve
from motorhand_teleop.teleop_monitor.plots import build_pages
app=QApplication([])
MonitorWindow.start_renderer=lambda self:None
w=MonitorWindow(backend='python');w.timer.stop()
base=time.monotonic_ns()
def ingest(sequence):
    frame=validation_sample(4.+sequence*.02,base+sequence*20_000_000)
    frame.update(session_id='frozen-render',sequence=sequence)
    w.state.ingest(frame,frame['timestamp_ns'])
ingest(0)
assert w.current_analysis(refresh=True) is None
w.analysis_broker.future.result(timeout=3)
assert w.current_analysis() is None
first=w.current_analysis(refresh=True)
assert len(first['points'])==1
ingest(1)
time.sleep(.21)
assert w.current_analysis(refresh=True) is first
w.analysis_broker.future.result(timeout=3)
assert w.current_analysis() is first
second=w.current_analysis(refresh=True)
assert len(second['points'])==2
assert w.current_analysis() is second
# J5为反向raw轴；名义raw力矩为正，散点需按阻挡方向转换为正抵抗响应。
spec=build_pages(1,selected_axis=4)['④ 刚度关系'][1]
curve=ScatterCurve(w,spec);curve.resize(650,500);curve.show();app.processEvents()
assert curve.bounds is not None
xl,xh,xs,yl,yh,ys=curve.bounds
assert xl<0<xh and xh>1 and xl>-.5
assert yl<0<yh and yh>1 and yl>-.5
assert ys>0
assert w.current_analysis() is second
# 切轴会立刻使旧结果不可见，避免把J5数据显示成J6。
w.feedback_signal.setCurrentIndex(1)
assert w.current_analysis() is None
curve.close();w.close();app.processEvents()
'''
    result = subprocess.run([sys.executable, '-c', script],
                            env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
