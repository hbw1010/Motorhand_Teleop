"""曲线缓存、暂停快照与实际事件帧率的行为验证。"""
import json

import numpy as np
import pytest

from motorhand_teleop.real_hand.force_curves import make_demo_sample
from motorhand_teleop.teleop_monitor.plots import Trace
from motorhand_teleop.teleop_monitor.state import MonitorState


def packet(sequence, session='one', stamp=None):
    stamp = sequence * 1_000_000 if stamp is None else stamp
    frame = make_demo_sample(1, stamp)
    frame.update(session_id=session, sequence=sequence,
                 glove_current_a=[float(sequence)] * 12)
    return frame


def terminal(session='one', stamp=3_000_000, message='已停止'):
    return dict(schema_version=1, type='teleop_status', session_id=session,
                state='stopped', message=message, timestamp_ns=stamp)


def test_series_ring_keeps_every_sample_missing_value_and_source_stamp():
    from motorhand_teleop.teleop_monitor.series import SeriesBuffer
    trace = Trace('values', 0, '电流', '#fff', 2., time_field='sample_ns')
    series = SeriesBuffer(trace, 3)
    for sequence, value in enumerate([1., None, -3., 4.], 1):
        series.append(dict(session_id='one', sequence=sequence,
                           timestamp_ns=100 + sequence, sample_ns=200 + sequence,
                           values=[value]))
    stamps, values = series.arrays()
    np.testing.assert_array_equal(stamps, [202, 203, 204])
    np.testing.assert_allclose(values, [np.nan, -6., 8.], equal_nan=True)
    assert stamps.dtype == np.uint64 and values.dtype == np.float64
    assert stamps.flags.c_contiguous and values.flags.c_contiguous
    assert not stamps.flags.writeable and not values.flags.writeable


def test_series_snapshots_stay_unchanged_after_ring_overwrite():
    from motorhand_teleop.teleop_monitor.series import SeriesBuffer
    series = SeriesBuffer(Trace('values', 0, 'x', '#fff'), 2)
    series.append(dict(timestamp_ns=1, values=[11.]))
    stamps, values = series.arrays()
    breaks = series.breaks()
    assert series.arrays()[0] is stamps
    assert series.arrays()[1] is values
    assert series.breaks() is breaks
    for stamp in (2, 3, 4):
        series.append(dict(timestamp_ns=stamp, values=[stamp * 11.]))
    np.testing.assert_array_equal(stamps, [1])
    np.testing.assert_array_equal(values, [11.])
    np.testing.assert_array_equal(breaks, [1])
    np.testing.assert_array_equal(series.arrays()[0], [3, 4])
    assert not breaks.flags.writeable and breaks.dtype == np.uint8


def test_series_marks_sequence_loss_and_session_boundary_without_filling():
    from motorhand_teleop.teleop_monitor.series import SeriesBuffer
    series = SeriesBuffer(Trace('values', 0, 'x', '#fff'), 5)
    for sequence, session in [(1, 'one'), (2, 'one'), (4, 'one'), (5, 'one'), (6, 'two')]:
        series.append(dict(timestamp_ns=sequence, values=[sequence],
                           sequence=sequence, session_id=session))
    np.testing.assert_array_equal(series.breaks(), [1, 0, 1, 0, 1])
    np.testing.assert_array_equal(series.arrays()[1], [1, 2, 4, 5, 6])


def test_state_shares_equivalent_trace_cache_and_only_appends_accepted_samples():
    state = MonitorState(limit=2)
    trace = Trace('glove_current_a', 0, '电流', '#fff')
    assert state.ingest(packet(1), 1_000_000)
    series = state.series_for(trace)
    equivalent = Trace('glove_current_a', 0, '另一图例', '#000', style=2)
    assert state.series_for(equivalent) is series
    scaled = state.series_for(Trace('glove_current_a', 0, '换算', '#fff', 2.))
    assert not state.ingest(packet(1), 1_100_000)
    assert state.ingest(packet(2), 2_000_000)
    assert state.ingest(packet(4), 4_000_000)
    np.testing.assert_array_equal(series.arrays()[1], [2., 4.])
    np.testing.assert_array_equal(scaled.arrays()[1], [4., 8.])
    np.testing.assert_array_equal(series.breaks(), [1, 1])
    assert [p['sequence'] for p in state.history] == [2, 4]


def test_pause_freezes_existing_and_lazily_created_series_during_live_updates():
    state = MonitorState(limit=2)
    first = Trace('glove_current_a', 0, '第一轴', '#fff')
    second = Trace('glove_current_a', 1, '第二轴', '#fff')
    state.ingest(packet(1), 1_000_000)
    live_series = state.series_for(first)
    state.set_paused(True)
    frozen_series = state.series_for(first)
    state.ingest(packet(2), 2_000_000)
    state.ingest(packet(3), 3_000_000)
    state.set_paused(True)
    assert state.series_for(first) is frozen_series
    np.testing.assert_array_equal(frozen_series.arrays()[1], [1.])
    np.testing.assert_array_equal(state.series_for(second).arrays()[1], [1.])
    assert [p['sequence'] for p in state.frames] == [1]
    assert state.display_frame['sequence'] == 1
    assert state.latest['sequence'] == 3
    state.set_paused(False)
    assert state.series_for(first) is live_series
    np.testing.assert_array_equal(live_series.arrays()[1], [2., 3.])
    assert state.display_frame['sequence'] == 3


def test_pause_survives_new_session_and_exports_frozen_terminal(tmp_path):
    state = MonitorState()
    trace = Trace('glove_current_a', 0, '电流', '#fff')
    state.ingest(packet(1), 1_000_000)
    state.ingest(terminal(), 3_000_000)
    state.set_paused(True)
    frozen_key = state.view_key
    assert state.ingest(packet(1, 'two', 4_000_000), 4_000_000)
    assert state.ingest(terminal('two', 5_000_000, '新会话结束'), 5_000_000)
    state.set_paused(True)
    assert state.view_key == frozen_key
    assert state.display_frame['session_id'] == 'one'
    assert state.display_terminal['message'] == '已停止'
    assert state.terminal['message'] == '新会话结束'
    np.testing.assert_array_equal(state.series_for(trace).arrays()[0], [1_000_000])
    path = tmp_path / 'frozen.jsonl'
    state.save(path)
    exported = [json.loads(line) for line in path.read_text().splitlines()]
    assert [p['session_id'] for p in exported] == ['one', 'one']
    assert exported[-1]['message'] == '已停止'
    state.set_paused(False)
    assert state.view_key != frozen_key
    assert state.display_frame['session_id'] == 'two'
    np.testing.assert_array_equal(state.series_for(trace).arrays()[0], [4_000_000])


def test_paused_export_excludes_terminal_arriving_after_freeze(tmp_path):
    state = MonitorState()
    state.ingest(packet(1), 1_000_000)
    state.set_paused(True)
    state.ingest(terminal(), 3_000_000)
    path = tmp_path / 'frozen.jsonl'
    state.save(path)
    exported = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(exported) == 1 and exported[0]['sequence'] == 1
    assert state.display_terminal is None


def test_view_key_changes_only_with_display_content_or_pause_mode():
    state = MonitorState()
    initial = state.view_key
    state.set_paused(False)
    assert state.view_key == initial
    state.ingest(packet(1), 1_000_000)
    first = state.view_key
    assert first != initial
    assert not state.ingest(packet(1), 2_000_000)
    assert state.view_key == first
    state.set_paused(True)
    paused = state.view_key
    assert paused != first
    state.ingest(packet(2), 2_000_000)
    state.ingest(terminal(), 3_000_000)
    assert state.view_key == paused
    state.set_paused(False)
    assert state.view_key != paused


def test_repeated_terminal_does_not_invalidate_an_unchanged_view():
    state = MonitorState()
    state.ingest(packet(1), 1_000_000)
    state.ingest(terminal(), 3_000_000)
    key = state.view_key
    state.ingest(terminal(), 4_000_000)
    assert state.view_key == key


def test_frame_rate_counts_actual_events_and_expires_when_idle():
    from motorhand_teleop.teleop_monitor.series import FrameRate
    rate = FrameRate()
    assert rate.value(0.) == 0.
    for index in range(61):
        rate.mark(index / 60.)
    assert rate.value(1.) == pytest.approx(60.)
    assert rate.value(2.) == 0.


def test_frame_rate_deduplicates_reused_frame_keys_even_after_idle():
    from motorhand_teleop.teleop_monitor.series import FrameRate
    rate = FrameRate(window_s=2.)
    rate.mark(0., key='frame-1')
    rate.mark(.1, key='frame-1')
    rate.mark(.2, key='frame-2')
    rate.mark(.3, key='frame-1')
    assert rate.value(.3) == 1.
    rate.mark(3., key='frame-2')
    assert rate.value(3.) == 0.
    rate.mark(3.1, key='frame-3')
    assert rate.value(3.1) == .5
