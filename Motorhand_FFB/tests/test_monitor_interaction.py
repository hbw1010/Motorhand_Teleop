import os
import subprocess
import sys


def test_box_zoom_and_pause_keep_original_snapshot(tmp_path):
    script=r'''
import json,time
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt,QPoint
from PyQt5.QtTest import QTest
from motorhand_teleop.teleop_monitor.window import MonitorWindow
from motorhand_teleop.real_hand.force_curves import make_demo_sample
from motorhand_teleop.teleop_monitor.metric_info import METRICS
app=QApplication([]);MonitorWindow.start_renderer=lambda self:None
w=MonitorWindow();w.show();app.processEvents()
now=time.monotonic_ns()
for i in range(250):
 t=now-(249-i)*20_000_000;p=make_demo_sample(i*.02,t);p.update(session_id='original',sequence=i,wuji_feedback_position_rad=[.2]*20,wuji_feedback_error_rad=p['wuji_tracking_error_rad'],wuji_feedback_current_a=p['wuji_current_a'],wuji_feedback_received_ns=t,feedback_request_id=i+1)
 w.state.ingest(p,t)
w.tick();app.processEvents()
c=w.curves[0];area=c.plot_area
before=json.dumps(w.state.frames,sort_keys=True)
a=QPoint(int(area.left()+area.width()*.25),int(area.top()+area.height()*.2))
b=QPoint(int(area.left()+area.width()*.7),int(area.top()+area.height()*.8))
QTest.mousePress(c,Qt.LeftButton,Qt.NoModifier,a)
QTest.mouseMove(c,b)
QTest.mouseRelease(c,Qt.LeftButton,Qt.NoModifier,b)
assert w.state.paused and w.view_range_ns is not None and c.y_view is not None
assert w.view_range_ns[1]-w.view_range_ns[0]<3_000_000_000
assert json.dumps(w.state.frames,sort_keys=True)==before
w.tick();app.processEvents()
area=c.plot_area
QTest.mouseClick(c,Qt.LeftButton,Qt.NoModifier,QPoint(int(area.left()+area.width()*.2),int(area.center().y())))
QTest.mouseClick(c,Qt.LeftButton,Qt.ShiftModifier,QPoint(int(area.left()+area.width()*.8),int(area.center().y())))
assert w.cursor_ns is not None and w.cursor_b_ns>w.cursor_ns
stamps={tr.stamp(f) for f in w.state.frames for tr in c.spec.traces}
assert w.cursor_ns in stamps and w.cursor_b_ns in stamps
# 新会话不覆盖正在查看的旧快照，放大和游标也不能改变导出内容。
p=make_demo_sample(1,time.monotonic_ns());p.update(session_id='new',sequence=1)
w.state.ingest(p,time.monotonic_ns());w.tick()
assert json.dumps(w.state.frames,sort_keys=True)==before
path=Path(sys.argv[1]);selection=w.export_snapshot(path)
exported=[json.loads(line) for line in path.read_text().splitlines()]
assert exported==w.state.frames
context=json.loads(selection.read_text())
assert context['paused'] is True and context['raw_axis_index']==4
assert context['time_range_ns']==sorted([w.cursor_ns,w.cursor_b_ns])
assert context['session_id']=='original' and context['frame_count']==250
from motorhand_teleop.teleop_monitor.validation_analysis import analyze_axis,ValidationOptions
options=ValidationOptions(**context['options'])
assert analyze_axis(exported,context['raw_axis_index'],context['time_range_ns'],options)==analyze_axis(w.state.frames,4,context['time_range_ns'],options)
w.reset_zoom();assert w.view_range_ns is None and c.y_view is None
w.pause.setChecked(False);assert w.state.display_frame['session_id']=='new'
assert all(set(('name','unit','kind','meaning','formula','coordinates','time_source','limitations'))<=set(item) for item in METRICS.values())
w.close();app.processEvents()
'''
    # 代码字符串仅用于独立Qt测试进程，sys导入不依赖任何硬件。
    result=subprocess.run([sys.executable,'-c','import sys\n'+script,str(tmp_path/'snapshot.jsonl')],
                          env={**os.environ,'QT_QPA_PLATFORM':'offscreen'},capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr


def test_space_toggles_snapshot_from_window_controls_without_auto_repeat():
    script=r'''
import time
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt,QEvent
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtTest import QTest
from motorhand_teleop.teleop_monitor.window import MonitorWindow
from motorhand_teleop.teleop_monitor.demo import validation_sample
app=QApplication([]);MonitorWindow.start_renderer=lambda self:None
w=MonitorWindow();w.show();w.activateWindow();app.processEvents()
now=time.monotonic_ns();p=validation_sample(4,now);w.state.ingest(p,now)
w.tabs.setFocus();app.processEvents()
QTest.keyClick(w.tabs,Qt.Key_Space);assert w.state.paused
before=w.state.display_frame
later=validation_sample(4.02,now+20_000_000);w.state.ingest(later,now+20_000_000)
assert w.state.display_frame is before and w.state.latest is not before
# 长按产生的自动重复不能把冻结数据来回切换。
app.sendEvent(w.tabs,QKeyEvent(QEvent.KeyPress,Qt.Key_Space,Qt.NoModifier,' ',True))
assert w.state.paused
w.finger.setFocus();app.processEvents()
QTest.keyClick(w.finger,Qt.Key_Space)
assert not w.state.paused and w.state.display_frame is w.state.latest
# 暂停按钮自己获得焦点时也只切换一次。
w.pause.setFocus();app.processEvents()
QTest.keyClick(w.pause,Qt.Key_Space);assert w.state.paused
QTest.keyClick(w.pause,Qt.Key_Space);assert not w.state.paused
w.close();app.processEvents()
'''
    result=subprocess.run([sys.executable,'-c',script],
        env={**os.environ,'QT_QPA_PLATFORM':'offscreen'},capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
