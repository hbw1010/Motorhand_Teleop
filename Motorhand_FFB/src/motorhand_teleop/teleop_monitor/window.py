"""独立Qt监视窗口。绘图仅使用遥测，关闭窗口不停止控制程序。"""
import math
import json
from pathlib import Path
from dataclasses import asdict
import time
from datetime import datetime
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QKeySequence
from PyQt5.QtWidgets import (QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QCheckBox, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter, QFileDialog, QTabWidget, QScrollArea, QShortcut)
from .scene import MonitorScene
from .state import MonitorState
from .quantities import TorqueConversion
from .plots import build_pages
from .plot_widget import Curve
from .curve_backend import CurveBackend
from .series import FrameRate
from .validation_view import AnalysisBroker, ScatterCurve, analysis_summary
from .validation_signals import reason_text
from .validation_analysis import ValidationOptions
from .quantities import joint_label

FINGERS=['拇指','食指','中指','无名指','小指','全部']
PHASES={'free':'自由','arming':'判据确认中','contact':'接触估计','releasing':'释放中',
        'waiting_geometry':'等待几何','not_configured':'等待反馈'}


def finite(v):
    return isinstance(v,(int,float)) and math.isfinite(v)


class Viewport(QLabel):
    def __init__(self, owner):
        super().__init__('正在加载三维场景…')
        self.owner=owner;self.anchor=None
        self.setAlignment(Qt.AlignCenter);self.setMinimumSize(480,180)
        self.setStyleSheet('background:#111925;border:1px solid #26364b;border-radius:10px;')
    def mousePressEvent(self,e):self.anchor=e.pos()
    def mouseMoveEvent(self,e):
        if self.anchor is not None:
            d=e.pos()-self.anchor;self.anchor=e.pos()
            cam=self.owner.scene.camera;cam.azimuth+=d.x()*.4
            cam.elevation=float(np.clip(cam.elevation+d.y()*.35,-85,85))
    def mouseReleaseEvent(self,e):self.anchor=None
    def paintEvent(self,e):
        super().paintEvent(e)
        if self.owner.image_generation:self.owner.render_rate.mark(key=self.owner.image_generation)
    def wheelEvent(self,e):
        cam=self.owner.scene.camera;cam.distance=float(np.clip(cam.distance*math.exp(-e.angleDelta().y()/1200),.2,2.))


class MonitorWindow(QMainWindow):
    def __init__(self,port=8765,demo=False,torque_config=None,*,fps=60,backend='auto',smooth=False):
        super().__init__();self.state=MonitorState(torque_config=torque_config or TorqueConversion.load());self.port=port;self.demo=demo
        self.scene=MonitorScene();self.renderer=None;self.render_error='';self.rendered=False;self.last_session=None
        self.cursor_ns=None;self.cursor_b_ns=None
        self.view_range_ns=None;self.view_generation=0;self.image_generation=0
        self.render_rate=FrameRate();self.curve_rate=FrameRate();self.curve_backend=CurveBackend(backend)
        self._last_plot_key=None;self._status_at=0
        self.analysis_broker=AnalysisBroker();self._analysis_result=None;self._analysis_base=None
        self.setWindowTitle(f'Motorhand × Wuji Hand 2 · 遥操作监视 · UDP {port}');self.resize(1700,1180)
        self.setMinimumSize(1200,1000)
        font=QFont('Noto Sans CJK SC');font.setPixelSize(20);self.setFont(font)
        self.setStyleSheet('''QMainWindow,QWidget{background:#0d1520;color:#dce7f6;font-size:20px;}
            QLabel{background:transparent;} QPushButton,QComboBox{background:#1b2b40;border:1px solid #304860;border-radius:5px;padding:6px 12px;}
            QPushButton:hover{background:#263e58;} QCheckBox{spacing:6px;} QSplitter::handle{background:#263448;}''')
        central=QWidget();outer=QVBoxLayout(central);outer.setContentsMargins(18,12,18,12);outer.setSpacing(10);self.setCentralWidget(central)
        top=QHBoxLayout();title=QLabel('MOTORHAND  ×  WUJI HAND 2');title.setStyleSheet('font-size:26px;font-weight:700;')
        top.addWidget(title);top.addStretch();self.fps_label=QLabel();self.fps_label.setStyleSheet('font-size:16px;color:#b5c5d8;');top.addWidget(self.fps_label);self.badge=QLabel('演示数据' if demo else '等待遥测');top.addWidget(self.badge);outer.addLayout(top)
        self.status=QLabel();self.status.setWordWrap(True);outer.addWidget(self.status)
        chain_row=QHBoxLayout();self.chain_label=QLabel('控制链：等待同一次请求的输入/输出记录');self.chain_label.setWordWrap(True)
        chain_row.addWidget(self.chain_label,1);chain_row.addWidget(QLabel('分析电机'))
        self.feedback_signal=QComboBox();self.feedback_signal.addItem('J5 食指·指根屈伸',4);self.feedback_signal.addItem('J6 食指·中末节驱动',5)
        chain_row.addWidget(self.feedback_signal);outer.addLayout(chain_row)
        splitter=QSplitter(Qt.Vertical);outer.addWidget(splitter,1)
        scene_box=QWidget();scene_layout=QVBoxLayout(scene_box);scene_layout.setContentsMargins(0,0,0,0)
        row=QHBoxLayout();row.addWidget(QLabel('左：Motorhand  /  右：Wuji Hand 2    ·    灰色：实测    红色：目标'))
        row.addStretch();self.overlay=QCheckBox('显示目标叠影');self.overlay.setChecked(True);row.addWidget(self.overlay)
        reset=QPushButton('复位视角');reset.clicked.connect(self.reset_camera);row.addWidget(reset);scene_layout.addLayout(row)
        self.viewport=Viewport(self);scene_layout.addWidget(self.viewport,1)
        self.pose_label=QLabel('模型预览 · 等待实测数据（拖动旋转，滚轮缩放）');scene_layout.addWidget(self.pose_label)
        splitter.addWidget(scene_box)
        bottom=QWidget();bottom_layout=QVBoxLayout(bottom);bottom_layout.setContentsMargins(0,0,0,0)
        controls=QHBoxLayout();controls.addWidget(QLabel('曲线 / 手指'))
        self.finger=QComboBox();self.finger.addItems(FINGERS);self.finger.setCurrentIndex(1);controls.addWidget(self.finger)
        self.seconds=QComboBox()
        for sec in (1,2,5,10,30,60):self.seconds.addItem(f'{sec} 秒',sec)
        self.seconds.setCurrentIndex(2);controls.addWidget(self.seconds)
        self.pause=QPushButton('暂停数据');self.pause.setCheckable(True);self.pause.toggled.connect(self.toggle_pause);controls.addWidget(self.pause)
        self.pause.setToolTip('空格：暂停/继续查看数据；后台仍接收，真机仍运行。')
        self.pause_shortcut=QShortcut(QKeySequence(Qt.Key_Space),self)
        self.pause_shortcut.setContext(Qt.WindowShortcut);self.pause_shortcut.setAutoRepeat(False)
        self.pause_shortcut.activated.connect(self.pause.toggle)
        reset_view=QPushButton('重置缩放');reset_view.clicked.connect(self.reset_zoom);controls.addWidget(reset_view)
        save=QPushButton('保存快照');save.clicked.connect(self.save);controls.addWidget(save)
        explain=QPushButton('指标说明');explain.clicked.connect(lambda:self.show_metric_info());controls.addWidget(explain)
        self.smooth=QCheckBox('线条抗锯齿');self.smooth.setChecked(smooth);self.smooth.toggled.connect(self.invalidate_view);controls.addWidget(self.smooth);controls.addStretch()
        self.metrics=QLabel();controls.addWidget(self.metrics);bottom_layout.addLayout(controls)
        self.tabs=QTabWidget();self.curves=[];bottom_layout.addWidget(self.tabs,1)
        self.rebuild_plots();self.finger.currentIndexChanged.connect(self.change_finger)
        self.feedback_signal.currentIndexChanged.connect(self.rebuild_plots)
        help_text=QLabel('空格：暂停/继续数据，真机仍运行。拖框放大；滚轮缩放；双击重置。点击选A，Shift+点击选B；暂停后后台继续接收。')
        help_text.setWordWrap(True);help_text.setStyleSheet('font-size:16px;color:#b5c5d8;');bottom_layout.addWidget(help_text)
        splitter.addWidget(bottom);splitter.setSizes([250,750])
        self.notice=QLabel('独立只读监视 · 关闭此窗口不会停止硬件；在控制终端按 Ctrl+C 停止。');self.notice.setWordWrap(True);outer.addWidget(self.notice)
        self.timer=QTimer(self);self.timer.setTimerType(Qt.PreciseTimer);self.timer.timeout.connect(self.tick);self.timer.start(max(1,round(1000/fps)))
        QTimer.singleShot(0,self.start_renderer)

    def rebuild_plots(self,*_):
        selected=self.tabs.currentIndex()
        while self.tabs.count():
            page=self.tabs.widget(0);self.tabs.removeTab(0);page.deleteLater()
        self.curves=[];self.analysis_labels=[]
        frame=self.state.display_frame
        self._plotted_mode=frame.get('feedback_control_mode','torque') if frame else 'torque'
        for title,specs in build_pages(self.finger.currentIndex(),self._plotted_mode,self.feedback_signal.currentData()).items():
            scroll=QScrollArea();scroll.setWidgetResizable(True)
            content=QWidget();grid=QGridLayout(content);grid.setContentsMargins(4,4,4,4);grid.setSpacing(10)
            if title in ('④ 刚度关系','⑥ 保持/往返'):
                summary=QLabel('选区分析准备中');summary.setWordWrap(True);summary.setStyleSheet('font-size:16px;color:#c2d2e6;padding:8px;')
                grid.addWidget(summary,0,0,1,2);self.analysis_labels.append(summary);offset=1
            else:offset=0
            for i,spec in enumerate(specs):
                curve=ScatterCurve(self,spec) if spec.kind=='scatter' else Curve(self,spec)
                self.curves.append(curve);grid.addWidget(curve,i//2+offset,i%2)
            scroll.setWidget(content);self.tabs.addTab(scroll,title)
        if selected>=0:self.tabs.setCurrentIndex(min(selected,self.tabs.count()-1))

    def change_finger(self,*_):
        from .plots import GLOVE_GROUPS
        selected=self.feedback_signal.currentData();self.feedback_signal.blockSignals(True);self.feedback_signal.clear()
        for index in GLOVE_GROUPS[self.finger.currentIndex()]:self.feedback_signal.addItem(joint_label('glove',index),index)
        at=self.feedback_signal.findData(selected)
        if at>=0:self.feedback_signal.setCurrentIndex(at)
        self.feedback_signal.blockSignals(False);self.rebuild_plots()

    def current_analysis(self,refresh=False):
        axis=self.feedback_signal.currentData()
        if axis is None:return None
        frame=self.state.display_frame;session=frame.get('session_id') if frame else None
        bounds=self.analysis_bounds()
        base=(session,axis,self.state.paused,self.state.view_key if self.state.paused else None,
              self.view_range_ns,self.seconds.currentData(),self.cursor_ns,self.cursor_b_ns)
        if refresh:
            self._analysis_result=self.analysis_broker.get(self.state.frames,axis,bounds,base,self.state.view_key,force=self.state.paused)
            self._analysis_base=base
        return self._analysis_result if self._analysis_base==base else None

    def analysis_bounds(self):
        bounds=tuple(sorted((self.cursor_ns,self.cursor_b_ns))) if self.cursor_ns is not None and self.cursor_b_ns is not None and self.cursor_ns!=self.cursor_b_ns else self.time_range()
        return max(0,bounds[0]),max(0,bounds[1])

    def export_snapshot(self,path):
        """导出原始快照与选区条件；不保存可能落后于实时帧的后台分析结果。"""
        path=Path(path);self.state.save(path)
        frame=self.state.display_frame;axis=self.feedback_signal.currentData()
        context={'schema_version':1,'type':'teleop_validation_selection','data_file':path.name,
            'session_id':frame.get('session_id') if frame else None,'raw_axis_index':axis,
            'axis_label':joint_label('glove',axis),'time_range_ns':self.analysis_bounds(),
            'cursor_a_ns':self.cursor_ns,'cursor_b_ns':self.cursor_b_ns,'paused':self.state.paused,
            'frame_count':len(self.state.frames),'options':asdict(ValidationOptions()),
            'note':'使用同名JSONL中的原始快照及本选区参数重算；无独立标定时不评价Nm准确性。'}
        selection=path.with_suffix('.selection.json')
        selection.write_text(json.dumps(context,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        return selection

    def show_metric_info(self,field=None):
        from .metric_dialog import MetricDialog
        MetricDialog(self,field,self.state.display_frame).exec_()

    def time_range(self):
        if self.view_range_ns is not None:return self.view_range_ns
        frame=self.state.display_frame
        end=(frame.get('published_ns') or frame['timestamp_ns']) if frame else 0
        return end-self.seconds.currentData()*1_000_000_000,end

    def set_time_range(self,start,end):
        if end-start<1_000_000:return
        if end-start>120_000_000_000:
            center=(start+end)//2;start=center-60_000_000_000;end=center+60_000_000_000
        self.view_range_ns=(int(start),int(end))

    def invalidate_view(self,*_):
        self._last_plot_key=None;self.view_generation+=1;self._status_at=0
        for curve in self.curves:
            if curve.isVisible():curve.update()

    def reset_zoom(self):
        self.view_range_ns=None;self.cursor_ns=self.cursor_b_ns=None
        for curve in self.curves:
            curve.y_view=None
            if hasattr(curve,'x_view'):curve.x_view=None
        self.invalidate_view()

    def start_renderer(self):
        try:
            from glove_sim.ui.render_process import RenderProcess
            self.renderer=RenderProcess()
        except Exception as e:self.render_error=str(e)
    def reset_camera(self):
        self.scene.camera.azimuth=90.;self.scene.camera.elevation=-12.;self.scene.camera.distance=.43
    def toggle_pause(self,on):
        self.state.set_paused(on);self.pause.setText('恢复实时' if on else '暂停数据')
        if not on:self.reset_zoom()
        self.invalidate_view()
    def save(self):
        path,_=QFileDialog.getSaveFileName(self,'保存当前历史',f'monitor-{datetime.now():%Y%m%d-%H%M%S}.jsonl','JSONL (*.jsonl)')
        if path:
            try:
                selection=self.export_snapshot(path)
                self.notice.setText(f'快照已保存：{path}；选区条件：{selection.name}')
            except OSError as e:self.notice.setText(f'保存未完整完成（数据或选区条件写入失败）：{e}')
    def tick(self):
        now=time.monotonic_ns();latest=self.state.latest;frame=self.state.display_frame
        if frame and frame.get('feedback_control_mode','torque')!=self._plotted_mode:self.rebuild_plots()
        stale=self.state.stale(now)
        status_key=(self.state.session,self.state.terminal.get('timestamp_ns') if self.state.terminal else None,self.state.paused)
        status_due=now-self._status_at>=100_000_000 or status_key!=getattr(self,'_last_status_key',None)
        if status_due and self.tabs.currentIndex() in (3,5):
            result=self.current_analysis(refresh=True)
            text=analysis_summary(result,joint_label('glove',self.feedback_signal.currentData()))
            if result is None and self.analysis_broker.error:text='分析不可用：'+self.analysis_broker.error
            text+='\n分析依据：'+('A/B选区' if self.cursor_ns is not None and self.cursor_b_ns is not None else '当前时间窗口')+'；实时分析约5Hz刷新，暂停后固定快照。'
            for label in self.analysis_labels:
                label.setText(text)
                label.setToolTip('\n'.join(result['notes']) if result else '')
            for curve in self.curves:
                if isinstance(curve,ScatterCurve) and curve.isVisible():curve.update()
        if status_due:
            self._status_at=now;self._last_status_key=status_key
            is_demo=self.demo or (latest is not None and latest.get('source')=='demo')
            terminal=self.state.terminal
            if terminal:
                label='控制异常' if terminal['state']=='fault' else '会话结束'
                self.status.setText(label+' · '+terminal['message'])
            elif latest:
                phase=PHASES.get(latest.get('contact_state'),'未提供')
                age=lambda key:max(0,(now-latest[key])/1e6)
                self.status.setText(f'接触状态：{phase}    |    数据帧龄 Wuji {age("wuji_received_ns"):.0f} ms / Motorhand {age("glove_received_ns"):.0f} ms'
                    + ('    |    查看冻结快照，后台继续接收' if self.state.paused else ''))
            else:self.status.setText(f'等待控制程序遥测 · 本机 UDP {self.port} · 此窗口不会启动设备')
            stale=self.state.stale(now)
            self.badge.setText('离线合成演示' if is_demo else '故障 / 已结束' if terminal else '数据过期 / 断流' if stale and latest else '实时遥测' if latest else '等待连接')
            badge_style=('background:#3d2930;color:#ffb6b6;padding:7px;border-radius:5px;' if stale else 'background:#123a39;color:#76e2d6;padding:7px;border-radius:5px;')
            if self.badge.styleSheet()!=badge_style:self.badge.setStyleSheet(badge_style)
            history=list(self.state.history)
            hz=0.
            if len(history)>2:
                a=history[-min(100,len(history))];delta=history[-1]['timestamp_ns']-a['timestamp_ns']
                if delta>0:hz=(min(100,len(history))-1)*1e9/delta
            self.metrics.setText(f'接收 {hz:.1f} Hz · '+(f'快照 {len(self.state.snapshot)} 帧' if self.state.paused else f'{len(history)} 帧')+(' · 断流/已结束' if stale else ''))
            if self.cursor_ns is not None and self.cursor_b_ns is not None:
                self.metrics.setText(self.metrics.text()+f' · Δt(B−A)={(self.cursor_b_ns-self.cursor_ns)/1e6:.1f} ms')
            self.fps_label.setText(f'3D {self.render_rate.value():.0f} FPS · 曲线 {self.curve_rate.value():.0f} FPS · {self.curve_backend.name}')
            delay=frame.get('feedback_chain_delay_ms',[None])[0] if frame else None
            request=frame.get('feedback_request_id') if frame else None
            self.chain_label.setText(
                f'请求 #{request}：Wuji收包→手套SDK提交 {delay:.2f} ms（非执行延迟）'
                if finite(delay) else '控制链：等待同一次请求的输入/输出记录；旧控制程序需重启以提供这些字段')
            reasons=frame.get('glove_control_reason') if frame else None
            axis=self.feedback_signal.currentData()
            if isinstance(reasons,list) and axis is not None and axis<len(reasons):
                self.chain_label.setText(joint_label('glove',axis)+'：'+reason_text(reasons[axis])+' | '+self.chain_label.text())
            source=frame.get('glove_proxy_source_ns') if frame else None
            source=source[axis] if isinstance(source,list) and axis is not None and axis<len(source) else None
            submitted=frame.get('glove_submit_ns') if frame else None
            if type(source) is int and type(submitted) is int and 0<source<=submitted:
                self.chain_label.setText(self.chain_label.text()+f' | 本轴参考源→提交 {(submitted-source)/1e6:.2f} ms')
                self.chain_label.setToolTip(f'原始单调时间：本轴参考输入源 {source} ns；本次SDK提交 {submitted} ns。参考有效性见②状态条。')
            else:self.chain_label.setToolTip('参考来源时刻未提供；不会用发布时间补齐。')
        view_session=frame['session_id'] if frame else None
        if view_session!=self.last_session:
            self.scene.last_q=self.scene.model.qpos0.copy();self.last_session=view_session
            self.view_range_ns=None
            self.cursor_ns=None;self.cursor_b_ns=None
        if frame:
            q,ghost=self.scene.pose(frame);missing=self.scene.availability(frame)
            gap=frame.get('glove_closure_gap_m')
            if status_due:self.pose_label.setText(('演示合成姿态' if frame.get('source')=='demo' else '实测姿态' if not missing else missing+' · 保留上次姿态/模型预览')
                + (' · 暂停快照' if self.state.paused else ' · 数据已过期' if stale else '')
                + (f' · 闭环间隙 {gap*1e3:.4g} mm' if finite(gap) else '')+' · 拖动旋转 / 滚轮缩放')
        else:
            q=self.scene.last_q;ghost=None
            if status_due:self.pose_label.setText('模型预览 · 暂无当前会话的显示帧（拖动旋转，滚轮缩放）')
        if self.renderer and not self.render_error:
            try:
                if not self.renderer.alive:raise RuntimeError('渲染子进程已退出')
                rgb=self.renderer.poll()
                if rgb is not None:
                    rgb=np.ascontiguousarray(np.clip(rgb*255,0,255),dtype=np.uint8)
                    h,w,_=rgb.shape;image=QImage(rgb.data,w,h,w*3,QImage.Format_RGB888).copy()
                    self.viewport.setPixmap(QPixmap.fromImage(image).scaled(self.viewport.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation));self.rendered=True;self.image_generation+=1
                self.renderer.request(self.scene.xml,q,np.zeros(self.scene.model.nv),self.scene.camera,1280,max(200,min(720,round(1280*self.viewport.height()/max(1,self.viewport.width())))),
                    ghost_qpos=ghost if self.overlay.isChecked() else None)
            except Exception as e:self.render_error=str(e)
        if self.render_error:self.pose_label.setText('三维不可用（曲线继续）：'+self.render_error)
        key=(self.state.view_key,self.seconds.currentData(),self.view_range_ns,self.cursor_ns,self.cursor_b_ns,self.smooth.isChecked())
        if key!=self._last_plot_key:
            self._last_plot_key=key;self.view_generation+=1
            for curve in self.curves:
                if curve.isVisible():curve.update()
    def closeEvent(self,event):
        self.timer.stop();self.analysis_broker.close()
        if self.renderer:self.renderer.close();self.renderer=None
        event.accept()
