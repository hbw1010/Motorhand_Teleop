"""只读Qt曲线与分阶段τ数值；默认数值页，全部轴可滚动，不加载或调用设备SDK。"""

from collections import deque
import math
import time

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


_HISTORY_LIMIT = 6000
_GAP_NS = 250_000_000
_STALE_NS = 300_000_000
_SOURCE_STALE_NS = 100_000_000
_FINGERS = ("拇指", "食指", "中指", "无名指", "小指", "全部")
_GLOVE_GROUPS = ((0, 4), (4, 6), (6, 8), (8, 10), (10, 12))
_COLORS = (
    "#2376c7", "#d76b13", "#269655", "#bd4291", "#7956bb",
    "#008f99", "#bb4848", "#727d19", "#a16c36", "#52728c",
    "#467acf", "#ce8160", "#5d936e", "#a15e9a", "#736fa8",
    "#348b82", "#ab6666", "#8a8b48", "#97764c", "#667e99",
)


def _number(value):
    return value is not None and math.isfinite(value)


def _display_number(value):
    """有效位格式不会把非零小量舍入成伪零。"""
    return f"{value:.6g}" if _number(value) else "未提供"


def _frame_value(frame, field, index):
    values = frame.get(field)
    return values[index] if values is not None and index < len(values) else None


class _CurvePanel(QWidget):
    """包含通道与末值图例的QPainter时间曲线。"""

    def __init__(self, owner, field, title, parent=None):
        super().__init__(parent)
        self.owner = owner
        self.field = field
        self.title = title
        self.setMinimumHeight(125)
        self.setAutoFillBackground(True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        painter.setFont(QFont("Sans Serif", 9))
        frames = self.owner._visible_frames
        latest = frames[-1] if frames else None
        unit = latest.get("glove_target_unit") if latest else None
        target = self.field == "glove_target"
        scale = 1000.0 if target and unit == "Nm" else 1.0
        if self.field == "wuji_load_fraction":
            scale = 100.0
        if target:
            if unit == "Nm":
                title = "手套名义力矩指令 · mN·m（SDK 量化前 Nm × 1000；非实测力矩）"
            elif unit == "A":
                title = "手套电流目标 · A（原始电机轴等效目标；非实测电流）"
            else:
                title = "手套反馈目标 · 未提供（无有效单位/反馈）"
        else:
            title = self.title
        painter.setPen(QColor("#202c3b"))
        painter.drawText(QRectF(12, 5, self.width() - 24, 23), Qt.AlignVCenter, title)

        indices = self.owner._channel_indices(self.field)
        painter.setFont(QFont("Sans Serif", 8))
        legend_step = painter.fontMetrics().height() + 2
        max_rows = max(1, int((self.height() - 44) / legend_step))
        columns = (len(indices) + max_rows - 1) // max_rows
        legend_width = min(self.width() * 0.55, max(238, columns * 185))
        plot = QRectF(80, 35, max(100, self.width() - legend_width - 99), self.height() - 66)
        legend_x = plot.right() + 16
        end_ns = self.owner._view_end_ns
        start_ns = end_ns - self.owner.window_seconds * 1_000_000_000
        eligible = []
        low = high = 0.0
        for sample in frames:
            valid_unit = not target or (unit is not None and sample["glove_target_unit"] == unit)
            values = sample[self.field] if valid_unit else ()
            eligible.append((sample, values))
            for index in indices:
                if index < len(values) and _number(values[index]):
                    value = values[index] * scale
                    low = min(low, value)
                    high = max(high, value)
        span = high - low
        margin = span * 0.12 if span else 0.1
        low -= margin
        high += margin
        painter.setFont(QFont("Sans Serif", 8))
        tick_labels = [f"{low + tick / 4 * (high - low):.3g}" for tick in range(5)]
        axis_width = max(painter.fontMetrics().horizontalAdvance(label) for label in tick_labels) + 8
        plot.setLeft(axis_width + 4)
        for tick in range(5):
            fraction = tick / 4
            y = plot.bottom() - fraction * plot.height()
            painter.setPen(QPen(QColor("#e4e9ef"), 1))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(QColor("#536274"))
            painter.drawText(QRectF(1, y - 9, axis_width, 18), Qt.AlignRight | Qt.AlignVCenter, tick_labels[tick])
            x = plot.left() + fraction * plot.width()
            painter.setPen(QPen(QColor("#edf0f4"), 1))
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.setPen(QColor("#536274"))
            painter.drawText(QRectF(x - 29, plot.bottom() + 4, 58, 18), Qt.AlignCenter,
                             f"{-(1 - fraction) * self.owner.window_seconds:g}s")
        painter.setPen(QPen(QColor("#a8b4c3"), 1))
        painter.drawRect(plot)

        painter.save()
        painter.setClipRect(plot.adjusted(-1, -1, 1, 1))
        for index in indices:
            path = QPainterPath()
            previous_ns = None
            previous_session = None
            connected = False
            point_count = 0
            only_point = None
            for sample, values in eligible:
                timestamp = sample["timestamp_ns"]
                value = values[index] if index < len(values) else None
                if not _number(value):
                    connected = False
                    continue
                x = plot.left() + (timestamp - start_ns) / (self.owner.window_seconds * 1_000_000_000) * plot.width()
                y = plot.bottom() - (value * scale - low) / (high - low) * plot.height()
                point = QPointF(x, y)
                if (not connected or previous_session != sample["session_id"]
                        or timestamp <= previous_ns or timestamp - previous_ns > _GAP_NS):
                    path.moveTo(point)
                    # A single isolated sample must remain visible without inventing a line.
                    painter.setPen(QPen(QColor(_COLORS[index]), 2.5))
                    painter.drawPoint(point)
                else:
                    path.lineTo(point)
                previous_ns = timestamp
                previous_session = sample["session_id"]
                connected = True
                point_count += 1
                only_point = point
            painter.setPen(QPen(QColor(_COLORS[index]), 1.5))
            painter.drawPath(path)
            if point_count == 1:
                painter.drawPoint(only_point)
        painter.restore()

        names_key = "wuji_joint_names" if self.field.startswith("wuji_") else "glove_joint_names"
        names = latest[names_key] if latest else ()
        values = latest[self.field] if latest else ()
        rows = (len(indices) + columns - 1) // columns
        cell_width = (legend_width - 12) / columns
        painter.setFont(QFont("Sans Serif", 8))
        for position, index in enumerate(indices):
            column, row = divmod(position, rows)
            x = legend_x + column * cell_width
            y = 36 + row * legend_step
            painter.setPen(QPen(QColor(_COLORS[index]), 2))
            painter.drawLine(QPointF(x, y + 7), QPointF(x + 13, y + 7))
            name = names[index] if index < len(names) else f"通道 {index + 1}"
            if not self.field.startswith("wuji_"):
                name = f"J{index + 1} {name}"
            value = values[index] if index < len(values) else None
            display = f"{value * scale:.3g}" if _number(value) and (not target or unit is not None) else "未提供"
            text = f"{name}: {display}"
            painter.setPen(QColor("#344254"))
            available = int(cell_width - 23)
            elided = painter.fontMetrics().elidedText(text, Qt.ElideMiddle, available)
            painter.drawText(QRectF(x + 19, y - 1, available, legend_step), Qt.AlignVCenter, elided)
        if not latest:
            painter.setPen(QColor("#687789"))
            painter.drawText(plot, Qt.AlignCenter, "等待数据 · 不生成替代曲线")
        elif not any(any(i < len(values) and _number(values[i]) for i in indices)
                     for _, values in eligible):
            painter.setPen(QColor("#687789"))
            painter.drawText(plot, Qt.AlignCenter, "所选通道未提供数据")


class ForceCurvesWindow(QMainWindow):
    """同帧曲线/数值共享有界历史与暂停快照；ingest仅在Qt主线程调用，不改变硬件状态。"""

    def __init__(self):
        super().__init__()
        font = QFont(self.font())
        font.setPixelSize(14)
        self.setFont(font)
        self.setWindowTitle("双侧力反馈 · 只读曲线与τ计算数值")
        self.resize(1400, 880)
        self.setMinimumSize(950, 720)
        self._history = deque(maxlen=_HISTORY_LIMIT)
        self._paused_frames = None
        self._visible_frames = []
        self._latest = None
        self._received_at_ns = None
        self._view_end_ns = time.monotonic_ns()
        self._paused_end_ns = None
        self._paused = False
        self._dirty = True
        self._numeric_key = None
        self._transport_notice = "等待接收；本窗口不连接或控制硬件"
        self.window_seconds = 10

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(5)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("手指分组"))
        self.finger_combo = QComboBox()
        self.finger_combo.setObjectName("finger_combo")
        self.finger_combo.addItems(_FINGERS)
        self.finger_combo.setCurrentIndex(1)
        self.finger_combo.currentIndexChanged.connect(self._selection_changed)
        controls.addWidget(self.finger_combo)
        controls.addWidget(QLabel("时间窗"))
        self.window_combo = QComboBox()
        self.window_combo.setObjectName("window_combo")
        for seconds in (5, 10, 30):
            self.window_combo.addItem(f"{seconds} 秒", seconds)
        self.window_combo.setCurrentIndex(1)
        self.window_combo.currentIndexChanged.connect(self._selection_changed)
        controls.addWidget(self.window_combo)
        self.wuji_quantity_combo = QComboBox()
        self.wuji_quantity_combo.setObjectName("wuji_quantity_combo")
        self.wuji_quantity_combo.addItem("Wuji 电流 A", "wuji_current_a")
        self.wuji_quantity_combo.addItem("Wuji 负载占比 %", "wuji_load_fraction")
        self.wuji_quantity_combo.currentIndexChanged.connect(self._selection_changed)
        controls.addWidget(self.wuji_quantity_combo)
        self.pause_button = QPushButton("暂停显示")
        self.pause_button.setObjectName("pause_button")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self._toggle_pause)
        controls.addWidget(self.pause_button)
        self.clear_button = QPushButton("清空显示历史")
        self.clear_button.setObjectName("clear_button")
        self.clear_button.clicked.connect(self._clear_history)
        controls.addWidget(self.clear_button)
        controls.addStretch()
        controls.addWidget(QLabel("只读 · 非一一映射"))
        layout.addLayout(controls)
        self.status_label = QLabel()
        self.status_label.setObjectName("status_label")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(51)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status_label.setTextFormat(Qt.PlainText)
        layout.addWidget(self.status_label)
        self.transport_label = QLabel(self._transport_notice)
        self.transport_label.setObjectName("transport_label")
        self.transport_label.setWordWrap(True)
        self.transport_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.transport_label.setTextFormat(Qt.PlainText)
        layout.addWidget(self.transport_label)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("force_view_tabs")
        curve_page = QWidget()
        curve_page.setObjectName("curves_page")
        curve_layout = QVBoxLayout(curve_page)
        curve_layout.setContentsMargins(0, 0, 0, 0)
        self.panels = (
            _CurvePanel(self, "wuji_current_a", "Wuji 实测关节电流 · A"),
            _CurvePanel(self, "glove_target", "手套反馈目标"),
            _CurvePanel(self, "glove_current_a", "手套 SDK解码电流 · A（非力矩测量）"),
        )
        for name, panel in zip(("wuji_curve", "glove_target_curve", "glove_current_curve"), self.panels):
            panel.setObjectName(name)
            curve_layout.addWidget(panel, 1)
        self.tabs.addTab(curve_page, "曲线")
        numeric_page = QWidget()
        numeric_page.setObjectName("tau_numeric_page")
        numeric_layout = QVBoxLayout(numeric_page)
        numeric_layout.setContentsMargins(4, 4, 4, 4)
        numeric_layout.setSpacing(4)
        self.tau_explanation_label = QLabel(
            "τ_v为Wuji语义关节虚拟量；投影τ和SDK前馈τ为手套raw轴名义量。\n"
            "不同坐标不可逐项相减；均非实测力矩，前馈τ不含MIT位置/速度PD输出。"
        )
        self.tau_explanation_label.setObjectName("tau_explanation_label")
        self.tau_explanation_label.setWordWrap(True)
        self.tau_explanation_label.setToolTip(
            "τ_v：Wuji20语义关节虚拟τ，低通后、未乘反射倍率；"
            "投影τ：手套raw12轴，已放大、尚未渐入/限幅/限斜率；"
            "最终τ：SDK量化前的名义前馈指令。分组不是一一关节映射，也不是接触力测量。"
        )
        numeric_layout.addWidget(self.tau_explanation_label)
        self.tau_gain_label = QLabel("控制端当前接触放大系数：未提供")
        self.tau_gain_label.setObjectName("tau_gain_label")
        self.tau_gain_label.setWordWrap(True)
        numeric_layout.addWidget(self.tau_gain_label)
        self.tau_frame_label = QLabel("显示帧：等待数据")
        self.tau_frame_label.setObjectName("tau_frame_label")
        self.tau_frame_label.setWordWrap(True)
        self.tau_frame_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.tau_frame_label.setTextFormat(Qt.PlainText)
        numeric_layout.addWidget(self.tau_frame_label)
        splitter = QSplitter(Qt.Vertical)
        splitter.setObjectName("tau_tables_splitter")
        wuji_section = QWidget()
        wuji_layout = QVBoxLayout(wuji_section)
        wuji_layout.setContentsMargins(0, 0, 0, 0)
        wuji_layout.setSpacing(2)
        wuji_layout.addWidget(QLabel("Wuji20 · 语义关节（当前分组；滚动查看全部行）"))
        self.wuji_tau_table = self._make_numeric_table(
            "wuji_tau_table",
            ("轴号", "语义关节名", "跟踪误差\nrad", "实际速度\nrad/s", "SDK电流\nA", "虚拟τ_v\n名义Nm"),
        )
        wuji_layout.addWidget(self.wuji_tau_table)
        splitter.addWidget(wuji_section)
        glove_section = QWidget()
        glove_layout = QVBoxLayout(glove_section)
        glove_layout.setContentsMargins(0, 0, 0, 0)
        glove_layout.setSpacing(2)
        glove_layout.addWidget(QLabel("手套raw12 · 原始电机轴（按原有手指分组；非Wuji逐轴对应）"))
        self.glove_tau_table = self._make_numeric_table(
            "glove_tau_table",
            ("raw轴", "电机关节名", "投影并放大τ\n限幅前 · 名义Nm", "最终SDK前馈τ\n量化前 · 名义Nm", "SDK解码电流\nA"),
        )
        glove_layout.addWidget(self.glove_tau_table)
        splitter.addWidget(glove_section)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        numeric_layout.addWidget(splitter, 1)
        self.tabs.addTab(numeric_page, "τ计算数值")
        self.tabs.setCurrentWidget(numeric_page)
        layout.addWidget(self.tabs, 1)
        self.safety_label = QLabel(
            "关闭此窗口不会停止真机！请回到控制终端按 Ctrl+C 停止控制。暂停/清空仅影响显示。\n"
            "active = 投影有效，不代表实测力已验收；refreshing = 力反射暂停。名义指令不是实测力矩。"
        )
        self.safety_label.setObjectName("safety_label")
        self.safety_label.setWordWrap(True)
        self.safety_label.setStyleSheet("font-size: 14px; color: #8a3d0b; background: #fff1dc; padding: 6px;")
        layout.addWidget(self.safety_label)
        self.setCentralWidget(central)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._update_status()
        self._update_numeric_values(None)

    @staticmethod
    def _make_numeric_table(name, headers):
        table = QTableWidget(0, len(headers))
        table.setObjectName(name)
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setMinimumHeight(125)
        table.verticalHeader().hide()
        table.verticalHeader().setMinimumSectionSize(22)
        table.verticalHeader().setDefaultSectionSize(25)
        header = table.horizontalHeader()
        header.setFixedHeight(table.fontMetrics().height() * 2 + 10)
        header.setMinimumSectionSize(58)
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.resizeSection(0, 58)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        return table

    def _update_numeric_values(self, frame):
        key = ((frame["session_id"], frame["sequence"]) if frame else None,
               self.finger_combo.currentIndex())
        if key == self._numeric_key:
            return
        self._numeric_key = key
        gain = frame.get("contact_reflection_gain") if frame else None
        self.tau_gain_label.setText(
            f"控制端当前接触放大系数（base gain × reflection_gain）：{_display_number(gain)}"
            " · 无量纲；非UI本地配置"
        )
        for table, field, names_field, columns in (
            (self.wuji_tau_table, "wuji_current_a", "wuji_joint_names",
             ("wuji_tracking_error_rad", "wuji_velocity_rad_s", "wuji_current_a", "virtual_hand_torque_nm")),
            (self.glove_tau_table, "glove_current_a", "glove_joint_names",
             ("glove_projected_torque_nm", "glove_target", "glove_current_a")),
        ):
            indices = self._channel_indices(field)
            table.setUpdatesEnabled(False)
            try:
                table.setRowCount(len(indices))
                for row, index in enumerate(indices):
                    name = _frame_value(frame, names_field, index) if frame else None
                    values = [f"{'W' if field.startswith('wuji_') else 'J'}{index + 1}",
                              name or "未提供"]
                    for column in columns:
                        value = _frame_value(frame, column, index) if frame else None
                        if column == "glove_target" and frame and frame.get("glove_target_unit") != "Nm":
                            values.append("未提供（A模式）" if frame.get("glove_target_unit") == "A" else "未提供")
                        else:
                            values.append(_display_number(value))
                    for column_index, text in enumerate(values):
                        item = QTableWidgetItem(text)
                        item.setToolTip(text)
                        if column_index != 1:
                            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        table.setItem(row, column_index, item)
            finally:
                table.setUpdatesEnabled(True)

    def _update_numeric_frame_status(self, frame):
        if frame is None:
            self.tau_frame_label.setText("显示帧：无数据（等待接收、已清空或已超出时间窗；不填零）")
            self.tau_frame_label.setStyleSheet("font-size: 14px; color: #735313; background: #fff5d6; padding: 3px;")
            return
        now = time.monotonic_ns()
        wuji_age = now - frame["wuji_received_ns"]
        glove_age = now - frame["glove_received_ns"]
        stale = min(wuji_age, glove_age) < 0 or max(wuji_age, glove_age) > _SOURCE_STALE_NS
        source = "DEMO 合成，非物理仿真" if frame["source"] == "demo" else "硬件遥测"
        state = "已暂停 · 冻结帧，非当前输出" if self._paused else "同周期显示帧"
        freshness = " · 数据过期／时钟异常，请勿视为当前输出" if stale else ""
        self.tau_frame_label.setText(
            f"{source} | {state} | 序号 {frame['sequence']} | 会话 {frame['session_id']}\n"
            f"源帧龄 Wuji {wuji_age / 1e6:.0f}ms / 手套 {glove_age / 1e6:.0f}ms"
            f" | 状态 {frame['feedback_state']}{freshness}"
        )
        self.tau_frame_label.setStyleSheet(
            "font-size: 14px; color: #a21c20; background: #ffe1e1; padding: 3px; font-weight: bold;" if stale else
            "font-size: 14px; color: #735313; background: #fff5d6; padding: 3px;" if self._paused or frame["source"] == "demo" else
            "font-size: 14px; color: #225f41; background: #e9f5ee; padding: 3px;"
        )

    def ingest(self, sample):
        """接收已验证的遥测包；不读取或写入硬件。"""
        frame = dict(sample)
        for key in ("wuji_joint_names", "glove_joint_names", "wuji_current_a",
                    "wuji_load_fraction", "glove_current_a", "glove_target",
                    "contact_active_mask", "virtual_hand_torque_nm",
                    "glove_projected_torque_nm", "wuji_tracking_error_rad", "wuji_velocity_rad_s"):
            if frame.get(key) is not None:
                frame[key] = tuple(frame[key])
        if self._latest is not None:
            if frame["session_id"] != self._latest["session_id"]:
                self._history.clear()
                self._visible_frames = []
                if self._paused:
                    self._paused_frames = []
                self._update_numeric_values(None)
                self._update_numeric_frame_status(None)
            elif (frame["sequence"] <= self._latest["sequence"]
                  or frame["timestamp_ns"] < self._latest["timestamp_ns"]):
                # 乱序或重放包既不倒退时间轴，也不能使旧数据看起来新鲜。
                return
        self._history.append(frame)
        self._latest = frame
        self._received_at_ns = time.monotonic_ns()
        self._dirty = True

    def set_transport_notice(self, message):
        self._transport_notice = str(message)
        self.transport_label.setText(self._transport_notice)

    def _channel_indices(self, field):
        finger = self.finger_combo.currentIndex()
        if field.startswith("wuji_"):
            return range(20) if finger == 5 else range(finger * 4, finger * 4 + 4)
        return range(12) if finger == 5 else range(*_GLOVE_GROUPS[finger])

    def _selection_changed(self, *_):
        self.window_seconds = self.window_combo.currentData()
        self.panels[0].field = self.wuji_quantity_combo.currentData()
        self.panels[0].title = (
            "Wuji 电流负载占比 · %（电流归一化代理；不是接触力）"
            if self.panels[0].field == "wuji_load_fraction" else "Wuji 实测关节电流 · A"
        )
        self._dirty = True

    def _toggle_pause(self, paused):
        self._paused = paused
        self.pause_button.setText("恢复显示" if paused else "暂停显示")
        if paused:
            self._paused_frames = [frame for frame in self._history
                                   if frame["timestamp_ns"] <= self._view_end_ns]
            self._paused_end_ns = self._view_end_ns
        else:
            self._paused_frames = None
            self._paused_end_ns = None
        self._dirty = True
        self._update_status()

    def _clear_history(self):
        self._history.clear()
        if self._paused:
            self._paused_frames = []
        self._visible_frames = []
        self._update_numeric_values(None)
        self._update_numeric_frame_status(None)
        self._dirty = True
        for panel in self.panels:
            panel.update()

    def _tick(self):
        self._update_status()
        # Advance the live axis even without packets: old data visibly ages out.
        end_ns = self._paused_end_ns if self._paused else time.monotonic_ns()
        if not self._paused or self._dirty:
            self._view_end_ns = end_ns
            source = self._paused_frames if self._paused else self._history
            start_ns = end_ns - self.window_seconds * 1_000_000_000
            self._visible_frames = [frame for frame in source
                                    if start_ns <= frame["timestamp_ns"] <= end_ns]
            for panel in self.panels:
                panel.update()
            self._update_numeric_values(self._visible_frames[-1] if self._visible_frames else None)
        self._update_numeric_frame_status(self._visible_frames[-1] if self._visible_frames else None)
        self._dirty = False

    def _update_status(self):
        paused = "已暂停显示（仍持续接收；不是实时曲线）" if self._paused else "实时显示"
        if self._latest is None:
            self.status_label.setText(f"{paused} | 尚未收到数据 | 无硬件控制能力")
            self.status_label.setStyleSheet("font-size: 14px; color: #735313; background: #fff5d6; padding: 4px;")
            return
        now = time.monotonic_ns()
        frame = self._latest
        received_age = now - self._received_at_ns
        wuji_age = now - frame["wuji_received_ns"]
        glove_age = now - frame["glove_received_ns"]
        clock_error = min(wuji_age, glove_age) < 0
        stale = (clock_error or received_age > _STALE_NS
                 or max(wuji_age, glove_age) > _SOURCE_STALE_NS)
        source = "DEMO 合成离线数据" if frame["source"] == "demo" else "硬件遥测"
        executed = "最近周期已发送（不代表当前使能已确认）" if frame["executed"] else "只读计算，未发送"
        state = frame["feedback_state"]
        state_text = {
            "active": "active：投影有效 ≠ 实测力已验收",
            "refreshing_pose": "refreshing_pose：力反射暂停（刷新姿态）",
            "refreshing_age": "refreshing_age：力反射暂停（雅可比过期）",
            "disabled": "disabled：力反射关闭",
        }[state]
        mapping = {
            "max_torque_linear_estimate": "最大力矩线性估计",
            "contact_impedance_estimate": "接触阻抗估计",
            "calibrated": "已标定模型",
            "disabled": "关闭",
        }[frame["mapping_mode"]]
        age = frame["jacobian_age_ns"]
        jacobian = "未提供" if age is None else f"{age / 1e6:.0f} ms"
        contact_state = frame.get("contact_state", "not_configured")
        contact_text = {
            "not_configured": "受阻估计未启用", "free": "未受阻", "arming": "受阻候选",
            "contact": "受阻估计成立（非实测接触）", "releasing": "释放中",
            "waiting_geometry": "受阻估计等待几何刷新",
        }[contact_state]
        freshness = "时钟异常！" if clock_error else "数据过期／请勿视为当前状态！" if stale else "接收正常"
        self.status_label.setText(
            f"{source} | {paused} | {freshness} | 接收距今 {received_age / 1e6:.0f}ms"
            f" / Wuji帧龄 {wuji_age / 1e6:.0f}ms / 手套帧龄 {glove_age / 1e6:.0f}ms"
            f" | 序号 {frame['sequence']} | {executed}\n"
            f"{state_text} | {contact_text} | 映射：{mapping} | 雅可比龄 {jacobian}"
        )
        if stale:
            style = "color: #a21c20; background: #ffe1e1; padding: 4px; font-weight: bold;"
        elif self._paused or frame["source"] == "demo":
            style = "color: #735313; background: #fff5d6; padding: 4px;"
        else:
            style = "color: #225f41; background: #e9f5ee; padding: 4px;"
        self.status_label.setStyleSheet("font-size: 14px; " + style)
