import json
import pytest
from motorhand_teleop.real_hand.force_curves import make_demo_sample
from motorhand_teleop.teleop_monitor.state import MonitorState, decode_packet


def packet(seq=1, session='one', stamp=1_000_000_000):
    p = make_demo_sample(1, stamp)
    p.update(session_id=session, sequence=seq, glove_tree_rad=[0.] * 68,
             glove_tree_order='ve33_closure_independent_dependent', hand='right',
             wuji_position_rad=[0.] * 20, wuji_target_rad=[.1] * 20,
             glove_position_rad=[0.] * 12, glove_closure_gap_m=0.)
    return p


def test_bad_tree_and_left_hand_rejected():
    for tree in ([0.] * 67, [float('nan')] * 68, [True] * 68):
        p=packet();p['glove_tree_rad']=tree
        with pytest.raises(ValueError): decode_packet(json.dumps(p).encode())
    p=packet();p['hand']='left'
    with pytest.raises(ValueError):decode_packet(json.dumps(p).encode())
    assert decode_packet(json.dumps(packet()).encode())['glove_tree_rad']==[0.]*68


def test_history_order_session_retirement_and_missing_data():
    s=MonitorState(limit=3)
    assert s.ingest(packet(), 1_000_000_000)
    assert not s.ingest(packet(), 1_100_000_000)
    s.ingest(packet(2), 1_200_000_000)
    s.ingest(packet(1,'two',stamp=1_300_000_000), 1_300_000_000)
    assert len(s.history)==1
    assert not s.ingest(packet(3,'one'), 1_400_000_000)
    p=packet(2,'two',stamp=1_500_000_000);p.pop('glove_tree_rad');p['glove_current_a']=[None]*12
    s.ingest(p,1_500_000_000)
    assert s.latest['glove_current_a']==[None]*12
    assert s.latest.get('glove_tree_rad') is None


def test_pause_freezes_display_but_not_liveness():
    s=MonitorState();s.ingest(packet(),1_000_000_000);s.set_paused(True)
    s.ingest(packet(2,stamp=1_100_000_000),1_100_000_000)
    assert s.display_frame['sequence']==1
    assert s.stale(1_500_000_001)
    s.set_paused(False)
    assert s.display_frame['sequence']==2


def test_terminal_event_preserves_history_and_latches():
    s=MonitorState();s.ingest(packet(),1_000_000_000)
    e={'schema_version':1,'type':'teleop_status','session_id':'one',
       'state':'fault','message':'ve33: closure failed','timestamp_ns':1_010_000_000}
    s.ingest(decode_packet(json.dumps(e).encode()),1_010_000_000)
    assert s.terminal['state']=='fault' and len(s.history)==1
    assert not s.ingest(packet(2),1_020_000_000)
    s.ingest(packet(1,'two',stamp=1_030_000_000),1_030_000_000)
    assert s.terminal is None


def test_export_preserves_missing_values_and_history(tmp_path):
    s=MonitorState();s.ingest(packet(),1_000_000_000)
    path=tmp_path/'capture.jsonl';s.save(path)
    data=[json.loads(line) for line in path.read_text().splitlines()]
    assert data[0]['schema_version']==1 and data[0]['glove_tree_rad']==[0.]*68


def test_scene_applies_actual_tree_and_target_without_solving():
    import numpy as np
    from motorhand_teleop.teleop_monitor.scene import MonitorScene
    scene=MonitorScene()
    p=packet();p['glove_tree_rad']=[.001*i for i in range(68)]
    q, ghost = scene.pose(p)
    np.testing.assert_allclose(q[scene.glove_indices],p['glove_tree_rad'])
    np.testing.assert_allclose(q[scene.wuji_indices],p['wuji_position_rad'])
    np.testing.assert_allclose(ghost[scene.wuji_indices],p['wuji_target_rad'])
    np.testing.assert_allclose(ghost[scene.glove_indices],q[scene.glove_indices])
    p['glove_tree_rad']=None
    assert '手套姿态缺失' in scene.availability(p)


def test_native_packet_remains_compatible_with_existing_decoder():
    import subprocess
    from pathlib import Path
    from motorhand_teleop.real_hand.curve_telemetry import decode_sample
    root=Path(__file__).resolve().parents[1]
    executable=root/'.cache/native-control/build/monitor_telemetry_native_test'
    if not executable.exists():pytest.skip('先构建原生遥测测试')
    output=subprocess.run([str(executable)],cwd=root,check=True,capture_output=True).stdout
    original=decode_sample(output.splitlines()[0])
    extended=decode_packet(output.splitlines()[0])
    assert original['wuji_position_rad']==extended['wuji_position_rad']
    assert len(extended['glove_tree_rad'])==68


def test_delayed_unseen_session_cannot_replace_current_session():
    s=MonitorState();s.ingest(packet(session='current',stamp=3_000_000_000),3_000_000_000)
    event={'schema_version':1,'type':'teleop_status','session_id':'never_seen_old',
           'timestamp_ns':2_000_000_000,'state':'fault','message':'旧会话故障'}
    assert not s.ingest(event,3_010_000_000)
    assert s.session=='current' and s.terminal is None
    assert not s.ingest(packet(session='another_old',stamp=2_000_000_000),3_020_000_000)


def test_unrepresentable_numbers_are_invalid_packets():
    for field,value in (('glove_tree_rad',[10**400]+[0.]*67),
                        ('glove_closure_gap_m',10**400),('timestamp_ns',10**400)):
        p=packet();p[field]=value
        with pytest.raises(ValueError):decode_packet(json.dumps(p).encode())


def test_window_pause_group_selection_demo_label_and_fault(tmp_path):
    import os
    import subprocess
    import sys
    script=r'''
import time
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt
from motorhand_teleop.teleop_monitor.window import MonitorWindow
from motorhand_teleop.real_hand.force_curves import make_demo_sample
app=QApplication([])
MonitorWindow.start_renderer=lambda self:None
w=MonitorWindow();w.show();app.processEvents()
now=time.monotonic_ns();p=make_demo_sample(1,now)
p.update(wuji_feedback_position_rad=[.2]*20,wuji_feedback_error_rad=p['wuji_tracking_error_rad'],wuji_feedback_current_a=p['wuji_current_a'],wuji_feedback_received_ns=now,feedback_request_id=1)
w.state.ingest(p,now);w.tick()
assert '演示' in w.badge.text()
QTest.mouseClick(w.pause,Qt.LeftButton)
assert w.state.paused
old=w.state.display_frame['sequence']
p=make_demo_sample(2,time.monotonic_ns());w.state.ingest(p,time.monotonic_ns());w.tick()
assert w.state.display_frame['sequence']==old
app.processEvents()
curve=w.curves[0]
from PyQt5.QtCore import QPoint
area=curve.plot_area
assert area is not None
QTest.mouseClick(curve,Qt.LeftButton,Qt.NoModifier,QPoint(int(area.left()+area.width()*.4),int(area.center().y())))
QTest.mouseClick(curve,Qt.LeftButton,Qt.ShiftModifier,QPoint(int(area.left()+area.width()*.6),int(area.center().y())))
assert w.cursor_b_ns >= w.cursor_ns
assert w.cursor_ns in {tr.stamp(w.state.display_frame) for tr in curve.spec.traces}
w.tick()
assert 'Δt' in w.metrics.text()
w.finger.setCurrentIndex(5);w.seconds.setCurrentIndex(3);app.processEvents()
assert not w.grab().isNull()
e=dict(type='teleop_status',session_id=p['session_id'],state='fault',message='闭链求解失败',timestamp_ns=time.monotonic_ns())
w.state.ingest(e,time.monotonic_ns());w.tick()
assert '闭链求解失败' in w.status.text()
w.close();app.processEvents()
'''
    result=subprocess.run([sys.executable,'-c',script],env={**os.environ,'QT_QPA_PLATFORM':'offscreen'},
                          capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr


def test_scene_numeric_validation_matches_decoder():
    from motorhand_teleop.teleop_monitor.scene import complete
    p=packet();p['glove_tree_rad']=[10**30]+[0.]*67
    decoded=decode_packet(json.dumps(p).encode())
    assert complete(decoded['glove_tree_rad'],68)
    assert not complete([10**400]+[0.]*67,68)
