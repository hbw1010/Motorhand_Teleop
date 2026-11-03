# -*- coding: utf-8 -*-
import os
import sys
import time
import math
import json
from pathlib import Path

# 添加 src 到搜索路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import serial.tools.list_ports as lp
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QSlider, QDoubleSpinBox,
    QPushButton, QScrollArea, QFrame, QTextEdit, QStatusBar,
    QComboBox, QTabWidget, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 从刚刚封装好的 SDK 中直接导入底层协议与调度器
from encos_hand.protocol import MotorProtocol, ERROR_MAP
from encos_hand.channel_worker import HeadlessChannelWorker

CALIBRATION_FILE = PROJECT_ROOT / "configs" / "joint_calibration.json"

class QtChannelWorker(QThread):
    sig_scan_finished = pyqtSignal(int, list, dict)
    sig_feedback = pyqtSignal(int, int, dict)
    sig_status = pyqtSignal(int, str)

    def __init__(self, ch_id: int):
        super().__init__()
        self.ch_id = ch_id
        self.core = HeadlessChannelWorker(ch_id, on_feedback=self._on_core_feedback)

    def _on_core_feedback(self, ch_id: int, mid: int, fb: dict):
        self.sig_feedback.emit(ch_id, mid, fb)

    @property
    def is_enabled(self):
        return self.core.is_connected

    def request_connect(self, port: str):
        self.core.connect(port)
        time.sleep(0.1)
        if self.core.is_connected:
            self.sig_status.emit(self.ch_id, f"CAN-{self.ch_id} 在线 ({port})")
        else:
            self.sig_status.emit(self.ch_id, f"CAN-{self.ch_id} 连接失败")

    def request_disconnect(self):
        self.core.disconnect()
        self.sig_status.emit(self.ch_id, f"CAN-{self.ch_id} 已断开")

    def request_scan(self):
        self.core.trigger_scan()

    def update_motor_cmd(self, mid: int, cmd: dict):
        self.core.set_joint_cmd(mid, cmd)

    def post_special_cmd(self, arb_id: int, data: bytes):
        self.core.post_special_cmd(arb_id, data)

    def run(self):
        if not self.core.is_alive():
            self.core.start()
        last_found = []
        while self.core.running:
            if self.core.active_ids != last_found:
                last_found = list(self.core.active_ids)
                self.sig_scan_finished.emit(self.ch_id, last_found, self.core.latest_feedbacks)
            time.sleep(0.05)

    def stop(self):
        self.core.stop()
        self.wait(500)

class MotorCardWidget(QFrame):
    def __init__(self, ch_id: int, motor_id: int, worker: QtChannelWorker, parent_window, calibration: dict, init_fb: dict = None):
        super().__init__()
        self.ch_id = ch_id
        self.motor_id = motor_id
        self.worker = worker
        self.parent_window = parent_window

        self.calib_zero = float(calibration.get("zero", 0.0))
        self.calib_min = float(calibration.get("min", -360.0))
        self.calib_max = float(calibration.get("max", 360.0))

        self.is_enabled = False
        self.current_mode = "servo"
        self.latest_err_code = 0
        
        self.latest_angle = float(init_fb["angle_deg"]) if init_fb and "angle_deg" in init_fb else 0.0
        self.target_angle = self.latest_angle
        self._ui_initialized = False

        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("MotorCardWidget { background-color: #fdfdfd; border: 1px solid #cfd8dc; border-radius: 6px; margin-bottom: 5px; }")
        self.init_ui()
        self.apply_calibration_limits()
        
        if init_fb:
            self.update_feedback(init_fb)
            self.set_target_angle(self.latest_angle, push=False)

        self._ui_initialized = True
        self.push_command()

    def init_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(10)

        # 左侧状态栏 (加宽至 245px 防截断)
        gb_s = QGroupBox(f"[CAN-{self.ch_id}] 关节 ID: {self.motor_id:02d}")
        gb_s.setFixedWidth(245)
        gb_s.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; }")
        grid_s = QGridLayout(gb_s)
        grid_s.setContentsMargins(10, 8, 10, 8)
        grid_s.setHorizontalSpacing(10)

        self.lbl_angle = QLabel(f"{self.latest_angle:.2f}°")
        self.lbl_angle.setStyleSheet("font-weight: bold; color: #16a085; font-size: 14px;")
        self.lbl_cur = QLabel("0.00A")
        self.lbl_temp = QLabel("--")
        self.lbl_err = QLabel("No Error")
        self.lbl_err.setStyleSheet("color: green; font-weight: bold;")
        self.lbl_limits = QLabel(f"[{self.calib_min:.0f}°, {self.calib_max:.0f}°]")
        self.lbl_limits.setStyleSheet("color: #7f8c8d; font-size: 11px;")

        grid_s.addWidget(QLabel("物理角度:"), 0, 0)
        grid_s.addWidget(self.lbl_angle, 0, 1)
        grid_s.addWidget(QLabel("相电流:"), 1, 0)
        grid_s.addWidget(self.lbl_cur, 1, 1)
        grid_s.addWidget(QLabel("驱动温度:"), 2, 0)
        grid_s.addWidget(self.lbl_temp, 2, 1)
        grid_s.addWidget(QLabel("软限位:"), 3, 0)
        grid_s.addWidget(self.lbl_limits, 3, 1)
        grid_s.addWidget(QLabel("报警状态:"), 4, 0)
        grid_s.addWidget(self.lbl_err, 4, 1)
        root.addWidget(gb_s)

        # 中间：解耦控制甲板
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabBar::tab { min-width: 140px; padding: 5px 12px; font-size: 12px; }")

        # Tab A: 伺服模式
        tab_sv = QWidget()
        v_sv = QVBoxLayout(tab_sv)
        h_pos_sv = QHBoxLayout()
        h_pos_sv.addWidget(QLabel("目标位置:"))
        self.spin_pos = QDoubleSpinBox()
        self.spin_pos.setDecimals(2)
        self.spin_pos.setSuffix("°")
        self.spin_pos.setFixedWidth(85)
        self.spin_pos.setValue(self.target_angle)
        h_pos_sv.addWidget(self.spin_pos)

        self.slider_pos = QSlider(Qt.Horizontal)
        self.slider_pos.setValue(int(self.target_angle * 10))
        self.slider_pos.valueChanged.connect(lambda v: self.set_target_angle(v / 10.0))
        self.spin_pos.valueChanged.connect(lambda v: self.set_target_angle(v))
        h_pos_sv.addWidget(self.slider_pos)
        v_sv.addLayout(h_pos_sv)

        h_param_sv = QHBoxLayout()
        h_param_sv.addWidget(QLabel("期望速度:"))
        self.spin_spd = QDoubleSpinBox()
        self.spin_spd.setRange(0.0, 200.0)
        self.spin_spd.setValue(30.0)
        self.spin_spd.setSuffix("rpm")
        self.spin_spd.setFixedWidth(85)
        self.spin_spd.valueChanged.connect(self.push_command)
        h_param_sv.addWidget(self.spin_spd)

        h_param_sv.addSpacing(15)
        h_param_sv.addWidget(QLabel("限制电流:"))
        self.spin_cur = QDoubleSpinBox()
        self.spin_cur.setRange(0.0, 5.0)
        self.spin_cur.setValue(0.8)
        self.spin_cur.setSuffix("A")
        self.spin_cur.setFixedWidth(80)
        self.spin_cur.valueChanged.connect(self.push_command)
        h_param_sv.addWidget(self.spin_cur)
        h_param_sv.addStretch()
        v_sv.addLayout(h_param_sv)
        self.tabs.addTab(tab_sv, "🎯 伺服位置模式")

        # Tab B: 力位混控 (MIT)
        tab_mit = QWidget()
        v_mit = QVBoxLayout(tab_mit)
        h_pos_mit = QHBoxLayout()
        h_pos_mit.addWidget(QLabel("目标位置:"))
        self.spin_mit_pos = QDoubleSpinBox()
        self.spin_mit_pos.setDecimals(2)
        self.spin_mit_pos.setSuffix("°")
        self.spin_mit_pos.setFixedWidth(85)
        self.spin_mit_pos.setValue(self.target_angle)
        h_pos_mit.addWidget(self.spin_mit_pos)

        self.slider_mit_pos = QSlider(Qt.Horizontal)
        self.slider_mit_pos.setValue(int(self.target_angle * 10))
        self.slider_mit_pos.valueChanged.connect(lambda v: self.set_target_angle(v / 10.0))
        self.spin_mit_pos.valueChanged.connect(lambda v: self.set_target_angle(v))
        h_pos_mit.addWidget(self.slider_mit_pos)
        v_mit.addLayout(h_pos_mit)

        h_param_mit = QHBoxLayout()
        h_param_mit.addWidget(QLabel("刚度(Kp):"))
        self.spin_kp = QDoubleSpinBox()
        self.spin_kp.setRange(0.0, 500.0)
        self.spin_kp.setValue(15.0)  # 经典 15.0
        self.spin_kp.setFixedWidth(70)
        self.spin_kp.valueChanged.connect(self.push_command)
        h_param_mit.addWidget(self.spin_kp)

        h_param_mit.addSpacing(10)
        h_param_mit.addWidget(QLabel("阻尼(Kd):"))
        self.spin_kd = QDoubleSpinBox()
        self.spin_kd.setRange(0.0, 5.0)
        self.spin_kd.setValue(0.5)
        self.spin_kd.setFixedWidth(65)
        self.spin_kd.valueChanged.connect(self.push_command)
        h_param_mit.addWidget(self.spin_kd)

        h_param_mit.addSpacing(10)
        h_param_mit.addWidget(QLabel("期望速度:"))
        self.spin_mit_vel = QDoubleSpinBox()
        self.spin_mit_vel.setRange(-10.0, 10.0)
        self.spin_mit_vel.setValue(0.0)
        self.spin_mit_vel.setSuffix("r/s")
        self.spin_mit_vel.setFixedWidth(75)
        self.spin_mit_vel.valueChanged.connect(self.push_command)
        h_param_mit.addWidget(self.spin_mit_vel)

        h_param_mit.addSpacing(10)
        h_param_mit.addWidget(QLabel("前馈力矩:"))
        self.spin_mit_tor = QDoubleSpinBox()
        self.spin_mit_tor.setRange(-1.0, 1.0)
        self.spin_mit_tor.setValue(0.0)
        self.spin_mit_tor.setSuffix("N·m")
        self.spin_mit_tor.setFixedWidth(75)
        self.spin_mit_tor.valueChanged.connect(self.push_command)
        h_param_mit.addWidget(self.spin_mit_tor)
        h_param_mit.addStretch()
        v_mit.addLayout(h_param_mit)
        self.tabs.addTab(tab_mit, "⚡ 力位混控 (MIT 柔顺)")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        root.addWidget(self.tabs, stretch=1)

        # 右侧操作区 (安全互锁)
        v_act = QVBoxLayout()
        self.btn_en = QPushButton("电机使能 [OFF]")
        self.btn_en.setCheckable(True)
        self.btn_en.setStyleSheet("background-color: #bdc3c7; color: white; font-weight: bold; padding: 5px;")
        self.btn_en.clicked.connect(self.toggle_enable)
        v_act.addWidget(self.btn_en)

        grid_cal = QGridLayout()
        self.btn_sync = QPushButton("读角同步")
        self.btn_sync.setStyleSheet("background-color: #3498db; color: white; font-weight: bold;")
        self.btn_sync.clicked.connect(lambda: self.set_target_angle(self.latest_angle, push=True))

        self.btn_home = QPushButton("归零")
        self.btn_home.clicked.connect(lambda: self.set_target_angle(self.calib_zero, push=True))
        self.btn_zero = QPushButton("记录零点")
        self.btn_zero.clicked.connect(self.action_record_zero)
        self.btn_min = QPushButton("设Min")
        self.btn_min.clicked.connect(self.action_set_min)
        self.btn_max = QPushButton("设Max")
        self.btn_max.clicked.connect(self.action_set_max)
        self.btn_hw_zero = QPushButton("固化硬件0°")
        self.btn_hw_zero.setStyleSheet("color: #d35400;")
        self.btn_hw_zero.clicked.connect(self.action_hardware_zero)

        grid_cal.addWidget(self.btn_sync, 0, 0, 1, 2)
        grid_cal.addWidget(self.btn_home, 1, 0)
        grid_cal.addWidget(self.btn_zero, 1, 1)
        grid_cal.addWidget(self.btn_min, 2, 0)
        grid_cal.addWidget(self.btn_max, 2, 1)
        grid_cal.addWidget(self.btn_hw_zero, 3, 0, 1, 2)
        v_act.addLayout(grid_cal)
        root.addLayout(v_act)

        self.btn_hw_zero.setEnabled(not self.is_enabled)

    def apply_calibration_limits(self):
        self.lbl_limits.setText(f"[{self.calib_min:.0f}°, {self.calib_max:.0f}°]")
        for spin, slider in ((self.spin_pos, self.slider_pos), (self.spin_mit_pos, self.slider_mit_pos)):
            spin.blockSignals(True)
            slider.blockSignals(True)
            spin.setRange(self.calib_min, self.calib_max)
            slider.setRange(int(self.calib_min * 10), int(self.calib_max * 10))
            clamped = max(self.calib_min, min(self.calib_max, spin.value()))
            spin.setValue(clamped)
            slider.setValue(int(clamped * 10))
            spin.blockSignals(False)
            slider.blockSignals(False)

    def clamp(self, val: float) -> float:
        return max(self.calib_min, min(self.calib_max, float(val)))

    def set_target_angle(self, val: float, push: bool = True):
        clamped = self.clamp(val)
        self.target_angle = clamped
        for s_box, s_bar in ((self.spin_pos, self.slider_pos), (self.spin_mit_pos, self.slider_mit_pos)):
            s_box.blockSignals(True)
            s_bar.blockSignals(True)
            s_box.setValue(clamped)
            s_bar.setValue(int(clamped * 10))
            s_box.blockSignals(False)
            s_bar.blockSignals(False)
        if push and self._ui_initialized:
            self.push_command()

    def push_command(self):
        if not self._ui_initialized:
            return
        cmd = {
            "enabled": self.is_enabled,
            "mode": self.current_mode,
            "target_angle": self.target_angle,
            "servo_spd": self.spin_spd.value(),
            "servo_cur": self.spin_cur.value(),
            "mit_vel": self.spin_mit_vel.value(),
            "mit_kp": self.spin_kp.value(),
            "mit_kd": self.spin_kd.value(),
            "mit_tor": self.spin_mit_tor.value(),
        }
        self.worker.update_motor_cmd(self.motor_id, cmd)

    def on_tab_changed(self, idx: int):
        if not self._ui_initialized:
            return
        self.current_mode = "mit" if idx == 1 else "servo"
        self.spin_mit_vel.setValue(0.0)
        self.spin_mit_tor.setValue(0.0)
        self.push_command()

    def toggle_enable(self, checked: bool):
        if checked and self.latest_err_code != 0:
            self.btn_en.setChecked(False)
            QMessageBox.critical(self, "使能拦截", f"关节 {self.motor_id:02d} 当前存在报警 ({self.lbl_err.text()})，禁止使能！")
            return
        self.is_enabled = checked
        self.btn_hw_zero.setEnabled(not checked)
        self.set_target_angle(self.latest_angle, push=True)
        if checked:
            self.btn_en.setText("电机使能 [ON]")
            self.btn_en.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 5px;")
        else:
            self.btn_en.setText("电机使能 [OFF]")
            self.btn_en.setStyleSheet("background-color: #bdc3c7; color: white; font-weight: bold; padding: 5px;")
        self.parent_window.check_mutual_exclusion()

    def action_record_zero(self):
        self.calib_zero = self.latest_angle
        self.parent_window.save_calibration_to_file()
        QMessageBox.information(self, "标定更新", f"[CAN-{self.ch_id}] 关节 {self.motor_id:02d} 工作零点更新为: {self.calib_zero:.2f}°")

    def action_set_min(self):
        if self.latest_angle >= self.calib_max:
            QMessageBox.warning(self, "错误", "最小值不能大于等于最大值！")
            return
        self.calib_min = self.latest_angle
        self.apply_calibration_limits()
        self.parent_window.save_calibration_to_file()

    def action_set_max(self):
        if self.latest_angle <= self.calib_min:
            QMessageBox.warning(self, "错误", "最大值不能小于等于最小值！")
            return
        self.calib_max = self.latest_angle
        self.apply_calibration_limits()
        self.parent_window.save_calibration_to_file()

    def action_hardware_zero(self):
        if self.is_enabled:
            QMessageBox.warning(self, "高危拦截", f"关节 {self.motor_id:02d} 当前处于发力使能状态，严禁固化硬件零位！请先切换为 [OFF] 并在机械完全静止下操作。")
            return
        if self.latest_err_code != 0:
            QMessageBox.warning(self, "操作受阻", f"关节 {self.motor_id:02d} 当前存在报警，固件拒绝写入零位！")
            return
        cmd = bytes([(self.motor_id >> 8) & 0xFF, self.motor_id & 0xFF, 0x00, 0x03])
        self.worker.post_special_cmd(0x7FF, cmd)
        self.calib_zero = 0.0
        self.set_target_angle(0.0, push=False)
        self.parent_window.save_calibration_to_file()
        QMessageBox.information(self, "硬件零位固化", f"已向关节 {self.motor_id:02d} 写入硬件绝对零位！\n软件工作零点已重置为 0.00°。")

    def update_feedback(self, fb: dict):
        self.latest_angle = fb.get("angle_deg", 0.0)
        self.lbl_angle.setText(f"{self.latest_angle:.2f}°")
        self.lbl_cur.setText(f"{fb.get('current_a', 0.0):.2f}A")
        t_drv = fb.get("temp_driver", 0.0)
        self.lbl_temp.setText(f"{t_drv:.1f}℃" if t_drv > 0.0 else "--")
        err_code = fb.get("err_code", 0)
        self.latest_err_code = err_code
        err_str = fb.get("err_str", "No Error")
        self.lbl_err.setText(err_str)
        if err_code != 0:
            self.lbl_err.setStyleSheet("color: red; font-weight: bold;")
            if self.is_enabled and err_code in (1, 2, 3, 4, 7):
                self.btn_en.setChecked(False)
                self.toggle_enable(False)
        else:
            self.lbl_err.setStyleSheet("color: green; font-weight: bold;")
        if not self.is_enabled:
            self.set_target_angle(self.latest_angle, push=False)

class MotorMasterMultiCanWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Encos MotorMaster Multi-CAN v0.3 - 架构重构版 (依赖独立 SDK)")
        self.resize(1240, 850)
        self.cards = {}
        self.calibrations = self.load_calibrations()
        self.scan_results = {1: [], 2: [], 3: []}
        self.scan_feedbacks = {1: {}, 2: {}, 3: {}}

        self.workers = {}
        for ch_id in (1, 2, 3):
            w = QtChannelWorker(ch_id)
            w.sig_scan_finished.connect(self.on_channel_scan_finished)
            w.sig_feedback.connect(self.on_channel_feedback)
            w.sig_status.connect(self.on_channel_status)
            w.start()
            self.workers[ch_id] = w

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        port_bar = QHBoxLayout()
        self.ch_widgets = {}
        available_ports = [p.device for p in lp.comports()]

        for ch_id in (1, 2, 3):
            gb = QGroupBox(f"CAN 通道 {ch_id}")
            h = QHBoxLayout(gb)
            cb_en = QCheckBox("启用")
            combo = QComboBox()
            combo.setMinimumWidth(95)
            combo.addItems(available_ports)
            if ch_id == 1 and "COM15" in available_ports:
                combo.setCurrentText("COM15")
                cb_en.setChecked(True)
            elif ch_id == 2 and "COM16" in available_ports:
                combo.setCurrentText("COM16")
                cb_en.setChecked(True)
            elif len(available_ports) >= ch_id:
                combo.setCurrentIndex(ch_id - 1)

            btn_toggle = QPushButton("连接")
            btn_toggle.clicked.connect(lambda _, c=ch_id: self.on_toggle_channel(c))
            h.addWidget(cb_en)
            h.addWidget(combo)
            h.addWidget(btn_toggle)
            port_bar.addWidget(gb)
            self.ch_widgets[ch_id] = {"check": cb_en, "combo": combo, "btn": btn_toggle}

        btn_refresh_ports = QPushButton("🔄 刷新串口")
        btn_refresh_ports.clicked.connect(self.refresh_ports)
        port_bar.addWidget(btn_refresh_ports)
        port_bar.addStretch()
        main_layout.addLayout(port_bar)

        cmd_bar = QHBoxLayout()
        self.btn_scan = QPushButton("🔍 一键全总线自检重排")
        self.btn_scan.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; padding: 7px 14px;")
        self.btn_scan.clicked.connect(self.on_scan_all_clicked)
        cmd_bar.addWidget(self.btn_scan)

        btn_read_all = QPushButton("🔄 一键读角同步")
        btn_read_all.setStyleSheet("background-color: #16a085; color: white; font-weight: bold; padding: 7px 14px;")
        btn_read_all.clicked.connect(lambda: [c.set_target_angle(c.latest_angle, push=True) for c in self.cards.values()])
        cmd_bar.addWidget(btn_read_all)

        btn_all_en = QPushButton("全局使能")
        btn_all_en.clicked.connect(lambda: [c.toggle_enable(True) for c in self.cards.values()])
        cmd_bar.addWidget(btn_all_en)

        self.btn_all_off = QPushButton("急停 (全失能)")
        self.btn_all_off.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; padding: 7px 14px;")
        self.btn_all_off.clicked.connect(lambda: [c.toggle_enable(False) for c in self.cards.values()])
        cmd_bar.addWidget(self.btn_all_off)

        btn_all_zero = QPushButton("一键全归零")
        btn_all_zero.setStyleSheet("background-color: #f39c12; color: white; font-weight: bold; padding: 7px 14px;")
        btn_all_zero.clicked.connect(lambda: [c.set_target_angle(c.calib_zero, push=True) for c in self.cards.values()])
        cmd_bar.addWidget(btn_all_zero)

        btn_all_servo = QPushButton("一键全伺服")
        btn_all_servo.clicked.connect(lambda: [c.tabs.setCurrentIndex(0) for c in self.cards.values()])
        cmd_bar.addWidget(btn_all_servo)

        btn_all_mit = QPushButton("一键全力控")
        btn_all_mit.clicked.connect(lambda: [c.tabs.setCurrentIndex(1) for c in self.cards.values()])
        cmd_bar.addWidget(btn_all_mit)

        btn_save_cal = QPushButton("💾 保存全部标定")
        btn_save_cal.clicked.connect(self.save_calibration_to_file)
        cmd_bar.addWidget(btn_save_cal)
        cmd_bar.addStretch()
        main_layout.addLayout(cmd_bar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.v_cards_layout = QVBoxLayout(self.scroll_content)
        self.scroll.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(110)
        main_layout.addWidget(self.txt_log)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def refresh_ports(self):
        ports = [p.device for p in lp.comports()]
        for ch_id in (1, 2, 3):
            combo = self.ch_widgets[ch_id]["combo"]
            curr = combo.currentText()
            combo.clear()
            combo.addItems(ports)
            if curr in ports:
                combo.setCurrentText(curr)

    def on_toggle_channel(self, ch_id: int):
        worker = self.workers[ch_id]
        btn = self.ch_widgets[ch_id]["btn"]
        cb_en = self.ch_widgets[ch_id]["check"]
        combo = self.ch_widgets[ch_id]["combo"]

        if worker.is_enabled:
            for (cid, mid), card in self.cards.items():
                if cid == ch_id and card.is_enabled:
                    card.toggle_enable(False)
            worker.request_disconnect()
            btn.setText("连接")
            btn.setStyleSheet("")
            cb_en.setChecked(False)
            return

        if not cb_en.isChecked():
            cb_en.setChecked(True)

        target_port = combo.currentText()
        if not target_port:
            QMessageBox.warning(self, "串口错误", f"CAN 通道 {ch_id} 未选择串口！")
            return

        btn.setText("断开")
        btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        worker.request_connect(target_port)

    def load_calibrations(self) -> dict:
        defaults = {str(i): {"zero": 0.0, "min": -360.0, "max": 360.0} for i in range(1, 33)}
        if CALIBRATION_FILE.exists():
            try:
                data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
                for k, v in data.items():
                    defaults[str(k)] = v
            except Exception:
                pass
        return defaults

    def save_calibration_to_file(self):
        for (ch_id, mid), card in self.cards.items():
            self.calibrations[str(mid)] = {"zero": card.calib_zero, "min": card.calib_min, "max": card.calib_max}
        try:
            CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
            CALIBRATION_FILE.write_text(json.dumps(self.calibrations, indent=2, ensure_ascii=False), encoding="utf-8")
            self.txt_log.append(">> [标定保存] 参数已持久化至 configs/joint_calibration.json")
        except Exception as e:
            self.txt_log.append(f"❌ 标定写入失败: {e}")

    def check_mutual_exclusion(self):
        any_enabled = any(card.is_enabled for card in self.cards.values())
        self.btn_scan.setEnabled(not any_enabled)
        self.btn_scan.setStyleSheet("background-color: #bdc3c7; color: white; font-weight: bold;" if any_enabled else "background-color: #2980b9; color: white; font-weight: bold;")

    def on_scan_all_clicked(self):
        self.btn_scan.setEnabled(False)
        self.btn_scan.setText("⏳ 各总线并行自检中...")
        for card in self.cards.values():
            card.setEnabled(False)
        self.scan_results = {1: [], 2: [], 3: []}
        self.scan_feedbacks = {1: {}, 2: {}, 3: {}}
        for ch_id, w in self.workers.items():
            if w.is_enabled:
                w.request_scan()

    def on_channel_scan_finished(self, ch_id: int, found_ids: list, init_fb: dict):
        self.scan_results[ch_id] = found_ids
        self.scan_feedbacks[ch_id] = init_fb
        self.txt_log.append(f">> [CAN-{ch_id}] 自检完成，发现节点: {found_ids}")
        self.rebuild_cards()
        self.btn_scan.setText("🔍 一键全总线自检重排")
        self.check_mutual_exclusion()

    def rebuild_cards(self):
        for card in self.cards.values():
            card.setParent(None)
        self.cards.clear()
        total_axes = 0
        for ch_id in (1, 2, 3):
            ids = self.scan_results[ch_id]
            total_axes += len(ids)
            fbs = self.scan_feedbacks[ch_id]
            for mid in ids:
                calib = self.calibrations.get(str(mid), {"zero": 0.0, "min": -360.0, "max": 360.0})
                card = MotorCardWidget(ch_id, mid, self.workers[ch_id], self, calib, fbs.get(mid))
                self.cards[(ch_id, mid)] = card
                self.v_cards_layout.addWidget(card)
        active_channels = sum(1 for w in self.workers.values() if w.is_enabled)
        self.status_bar.showMessage(f"当前启用: {active_channels} 个CAN通道 | 在线总轴数: {total_axes} 轴 | SDK 后端内核在线")

    def on_channel_feedback(self, ch_id: int, mid: int, fb: dict):
        key = (ch_id, mid)
        if key in self.cards:
            self.cards[key].update_feedback(fb)

    def on_channel_status(self, ch_id: int, text: str):
        self.txt_log.append(f">> {text}")

    def closeEvent(self, e):
        for w in self.workers.values():
            w.stop()
        e.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MotorMasterMultiCanWindow()
    win.show()
    sys.exit(app.exec_())
