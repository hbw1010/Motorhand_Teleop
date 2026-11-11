from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

try:
    import dearpygui.dearpygui as dpg
except Exception as exc:  # pragma: no cover - only hit on machines without DPG
    dpg = None
    _DPG_IMPORT_ERROR = exc
else:
    _DPG_IMPORT_ERROR = None

from ..mapping import FINGERS, MOTOR_FINGER
from ..physics import Simulation, StepResult
from ..runner import Runner
from .render_process import RenderProcess


RENDER_W = 1280
RENDER_H = 720
IMAGE_W = 760
IMAGE_H = 428
# Curves get one sample per control tick (runner.plot_samples, ~100 Hz);
# PLOT_HZ is only how often the plot widgets are redrawn.
PLOT_HISTORY = 6000
PLOT_HZ = 10.0
SAMPLE_HZ = 100.0
# Max points per series sent to a plot; denser windows are reduced to a
# per-bucket min/max envelope so peaks and oscillations are never dropped.
PLOT_MAX_POINTS = 1600
DEFAULT_TIME_SPAN_S = 10.0
FEEDBACK_MODES = {"driver_impedance": "驱动器阻抗（推荐）", "proxy_spring": "外置弹簧（旧）"}
FINGER_CN = {"thumb": "拇指", "index": "食指", "middle": "中指", "ring": "无名指", "pinky": "小指"}
# plot tag -> (x axis tag, y axis tag, x is time).  Wheel over a plot zooms its
# y axis about the cursor; Ctrl+wheel changes the shared time window (or zooms
# x for the non-time plot); double-click returns the plot to auto scale.
PLOT_AXES: dict[str, tuple[str, str, bool]] = {
    "cmp_torque_plot": ("cmp_torque_x", "cmp_torque_y", True),
    "cmp_pos_plot": ("cmp_pos_x", "cmp_pos_y", True),
    "cmp_cur_plot": ("cmp_cur_x", "cmp_cur_y", True),
    "force_plot": ("force_x", "force_y", True),
    "torque_plot": ("torque_x", "torque_y", True),
    "q_plot": ("q_x", "q_y", True),
    "current_plot": ("current_x", "current_y", True),
    "disp_plot": ("disp_x", "disp_y", False),
    "lat_plot": ("lat_x", "lat_y", True),
    "qd_plot": ("qd_x", "qd_y", True),
}
FONT_CANDIDATES = (
    str(Path(__file__).resolve().parents[3] / "assets" / "fonts" / "NotoSansCJKsc-UI.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
COLORS = (
    (230, 80, 80), (80, 180, 230), (120, 220, 120), (230, 180, 70),
    (200, 120, 230), (90, 230, 200), (240, 140, 160), (170, 170, 170),
)
C_BG = (10, 14, 20, 255)
C_PANEL = (16, 22, 30, 255)
C_CARD = (21, 29, 39, 255)
C_BORDER = (39, 52, 68, 255)
C_TEXT = (226, 235, 244, 255)
C_MUTED = (139, 153, 169, 255)
C_ACCENT = (78, 201, 255, 255)
C_GOLD = (255, 192, 92, 255)
C_GREEN = (105, 225, 167, 255)
C_RED = (255, 107, 107, 255)


def _quat_from_mujoco(q: np.ndarray) -> np.ndarray:
    """MuJoCo wxyz -> scipy xyzw."""
    return np.asarray([q[1], q[2], q[3], q[0]], dtype=float)


def _quat_to_mujoco(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.asarray([q[3], q[0], q[1], q[2]], dtype=float)


class App:
    def __init__(self, runner: Runner) -> None:
        self.runner = runner
        self.sim: Simulation = runner.sim
        self.config = runner.config
        self._proxy_mode = (str(self.config.get("glove", {}).get("feedback_mode", "proxy_spring"))
                            in ("proxy_spring", "driver_impedance"))
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        scene = self.config.get("scene", {})
        self._cam_home = (
            np.asarray(scene.get("camera_lookat", [0.145, 0.005, 0.120]), dtype=float),
            float(scene.get("camera_distance", 0.46)),
            float(scene.get("camera_azimuth", 122.0)),
            float(scene.get("camera_elevation", -24.0)),
        )
        self.cam.lookat[:] = self._cam_home[0]
        self.cam.distance = self._cam_home[1]
        self.cam.azimuth = self._cam_home[2]
        self.cam.elevation = self._cam_home[3]
        self._view_size: tuple[int, ...] = (0, 0)
        self.renderer: RenderProcess | None = None
        self._scene: mujoco.MjvScene | None = None
        self._model_ref = None
        self._texture = None
        self._last_tex = 0.0
        self._last_plot = 0.0
        self._history: deque = deque(maxlen=PLOT_HISTORY)
        self._t0 = time.monotonic()
        self._last_mouse: tuple[float, float] | None = None
        self._mouse_down = False
        self._right_down = False
        self._right_last: tuple[float, float] | None = None
        self._drag_name: str | None = None
        self._drag_mode: str | None = None
        self._wheel = 0.0
        self._cmp_motor = 0  # 0-based motor index shown in the 仿真 vs 真机 tab
        self._time_span = DEFAULT_TIME_SPAN_S
        self._zoom: dict[str, tuple[float, float]] = {}   # axis tag -> user limits
        self._auto_lims: dict[str, tuple[float, float]] = {}
        self._plot_data: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        # Paused: the curves freeze (zoom still works) while history keeps
        # recording in the background; resuming jumps to the live data.
        self._plots_paused = False
        self._frozen_t = np.zeros(0)
        self._frozen_hist: list = []
        self._fullscreen = False
        self._windowed_client = (1400, 980)
        self._pending_window_size: tuple[int, int] | None = None
        self.selected = (self.sim.active_object if self.sim.palm_mode
                         else (self.runner.specs[0].name if self.runner.specs else ""))
        self.pending: dict[str, dict[str, float]] = {}
        self._pending_select: str | None = None
        self._show_collision = False
        self._ui_fps = 0.0
        self._ui_last_t: float | None = None
        self._last_rebuild = 0.0
        # Object edits (select / size / hardness / rotation) run on a worker
        # thread: compiling the scene (~0.1 s) and refitting the holding hand
        # (~50 ms) happen without the runner lock and without blocking the UI;
        # only the state swaps take the lock for ~1 ms.
        self._edit_cv = threading.Condition()
        self._edit_queue: dict[str, object] = {}
        self._edit_stop = False
        self._edit_busy = False
        self._rotating: str | None = None
        self._rotate_last_x = 0.0
        self._yaw_target: float | None = None
        self._edit_thread = threading.Thread(target=self._edit_worker, name="object-edit", daemon=True)
        self._edit_thread.start()
        self._last_status = 0.0
        self._syncing = False
        self._recording_path = ""
        self._last_session: str | None = None
        self._last_error = ""
        self._build_ui()

    # ------------------------------------------------------------------
    def _install_theme(self) -> None:
        """A compact dark control-room theme instead of DearPyGui defaults."""
        assert dpg is not None

        def color(name: str, rgba: tuple[int, int, int, int]) -> None:
            tag = getattr(dpg, name, None)
            if tag is not None:
                dpg.add_theme_color(tag, rgba)

        def style(name: str, x: float, y: float = -1.0) -> None:
            tag = getattr(dpg, name, None)
            if tag is not None:
                dpg.add_theme_style(tag, x, y)

        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                color("mvThemeCol_WindowBg", C_BG)
                color("mvThemeCol_ChildBg", C_PANEL)
                color("mvThemeCol_PopupBg", C_CARD)
                color("mvThemeCol_Border", C_BORDER)
                color("mvThemeCol_Text", C_TEXT)
                color("mvThemeCol_TextDisabled", C_MUTED)
                color("mvThemeCol_FrameBg", C_CARD)
                color("mvThemeCol_FrameBgHovered", (29, 40, 54, 255))
                color("mvThemeCol_FrameBgActive", (35, 49, 66, 255))
                color("mvThemeCol_TitleBg", C_PANEL)
                color("mvThemeCol_TitleBgActive", C_CARD)
                color("mvThemeCol_MenuBarBg", C_PANEL)
                color("mvThemeCol_ScrollbarBg", C_BG)
                color("mvThemeCol_ScrollbarGrab", (55, 71, 90, 255))
                color("mvThemeCol_ScrollbarGrabHovered", (75, 96, 120, 255))
                color("mvThemeCol_ScrollbarGrabActive", C_ACCENT)
                color("mvThemeCol_CheckMark", C_ACCENT)
                color("mvThemeCol_SliderGrab", C_ACCENT)
                color("mvThemeCol_SliderGrabActive", (150, 226, 255, 255))
                color("mvThemeCol_Button", (27, 39, 53, 255))
                color("mvThemeCol_ButtonHovered", (40, 58, 78, 255))
                color("mvThemeCol_ButtonActive", C_ACCENT)
                color("mvThemeCol_Header", (25, 36, 49, 255))
                color("mvThemeCol_HeaderHovered", (34, 49, 67, 255))
                color("mvThemeCol_HeaderActive", (42, 61, 82, 255))
                color("mvThemeCol_Separator", C_BORDER)
                color("mvThemeCol_SeparatorHovered", C_ACCENT)
                color("mvThemeCol_SeparatorActive", C_ACCENT)
                color("mvThemeCol_ResizeGrip", C_BORDER)
                color("mvThemeCol_Tab", C_PANEL)
                color("mvThemeCol_TabHovered", (30, 44, 60, 255))
                color("mvThemeCol_TabActive", (26, 44, 62, 255))
                color("mvThemeCol_TabSelected", (26, 44, 62, 255))
                color("mvThemeCol_TabDimmed", C_PANEL)
                color("mvThemeCol_TabDimmedSelected", (22, 36, 51, 255))
                color("mvThemeCol_TabUnfocused", C_PANEL)
                color("mvThemeCol_TabUnfocusedActive", (22, 36, 51, 255))
                color("mvThemeCol_TableHeaderBg", C_CARD)
                color("mvThemeCol_TableRowBg", C_PANEL)
                color("mvThemeCol_TableRowBgAlt", (19, 26, 36, 255))
                color("mvThemeCol_TableBorderStrong", C_BORDER)
                color("mvThemeCol_TableBorderLight", (28, 38, 51, 255))
                style("mvStyleVar_WindowRounding", 12)
                style("mvStyleVar_ChildRounding", 10)
                style("mvStyleVar_PopupRounding", 10)
                style("mvStyleVar_FrameRounding", 7)
                style("mvStyleVar_GrabRounding", 7)
                style("mvStyleVar_TabRounding", 6)
                style("mvStyleVar_ScrollbarRounding", 8)
                style("mvStyleVar_WindowPadding", 12, 8)
                style("mvStyleVar_FramePadding", 8, 4)
                style("mvStyleVar_ItemSpacing", 8, 5)
                style("mvStyleVar_ItemInnerSpacing", 5, 3)
                style("mvStyleVar_WindowBorderSize", 1)
                style("mvStyleVar_ChildBorderSize", 1)
                style("mvStyleVar_FrameBorderSize", 0)
                style("mvStyleVar_TabBarBorderSize", 1)
                style("mvStyleVar_ImageBorderSize", 0)
            with dpg.theme_component(dpg.mvPlot):
                plot_colors = {
                    "mvPlotCol_PlotBg": (8, 12, 17, 255),
                    "mvPlotCol_FrameBg": (11, 16, 23, 255),
                    "mvPlotCol_PlotBorder": C_BORDER,
                    "mvPlotCol_LegendBg": (12, 18, 25, 235),
                    "mvPlotCol_LegendBorder": C_BORDER,
                    "mvPlotCol_LegendText": C_TEXT,
                    "mvPlotCol_TitleText": C_TEXT,
                    "mvPlotCol_InlayText": C_TEXT,
                    "mvPlotCol_AxisText": C_MUTED,
                    "mvPlotCol_AxisGrid": (31, 43, 58, 255),
                    "mvPlotCol_AxisTick": (53, 68, 87, 255),
                    # No global mvPlotCol_Line: it painted every series the
                    # same colour.  Series get explicit per-joint/finger themes.
                }
                for name, rgba in plot_colors.items():
                    tag = getattr(dpg, name, None)
                    if tag is not None:
                        dpg.add_theme_color(tag, rgba, category=dpg.mvThemeCat_Plots)
        dpg.bind_theme(theme)

        def button_theme(rgba, hovered, active):
            with dpg.theme() as item_theme:
                with dpg.theme_component(dpg.mvButton):
                    color("mvThemeCol_Button", rgba)
                    color("mvThemeCol_ButtonHovered", hovered)
                    color("mvThemeCol_ButtonActive", active)
            return item_theme

        self._primary_theme = button_theme((24, 86, 120, 255), (31, 110, 150, 255),
                                           (23, 132, 184, 255))
        self._danger_theme = button_theme((118, 35, 43, 255), (155, 45, 55, 255),
                                          (190, 52, 62, 255))

        def line_theme(rgba, weight=2.0):
            with dpg.theme() as item_theme:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, rgba, category=dpg.mvThemeCat_Plots)
                    weight_tag = getattr(dpg, "mvPlotStyleVar_LineWeight", None)
                    if weight_tag is not None:
                        dpg.add_theme_style(weight_tag, weight, category=dpg.mvThemeCat_Plots)
            return item_theme

        self._line_sim = line_theme(C_ACCENT)
        self._line_cmd = line_theme(C_GREEN, 1.5)
        self._line_real = line_theme(C_GOLD)
        self._line_sent = line_theme((190, 140, 255, 255), 1.5)
        palette = [(78, 201, 255), (255, 192, 92), (105, 225, 167), (255, 107, 107),
                   (190, 140, 255), (255, 140, 200), (120, 230, 230), (230, 230, 110),
                   (255, 160, 60), (140, 170, 255), (170, 255, 120), (255, 215, 180)]
        self._joint_themes = [line_theme((*c, 255), 1.8) for c in palette]
        # Same hue, thin and translucent: the sim-hand twin of a joint curve.
        self._joint_sim_themes = [line_theme((*c, 120), 1.0) for c in palette]

    def _current_label(self, text: str) -> str:
        """Fake-glove currents are synthesised (torque / Kt), say so."""
        if type(self.runner.glove).__name__ == "FakeGlove":
            return f"{text}（假手套模拟）"
        return text

    def _build_ui(self) -> None:
        assert dpg is not None
        self._install_theme()
        with dpg.texture_registry(show=False):
            # Raw float RGB texture: DearPyGui references the array instead
            # of copying it, and the render process already writes float32
            # RGB into shared memory -- no per-frame conversion or copy here.
            self._blank_texture = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.float32)
            self._texture = dpg.add_raw_texture(
                RENDER_W, RENDER_H, self._blank_texture, format=dpg.mvFormat_Float_rgb,
                tag="scene_texture",
            )
        with dpg.window(tag="main", no_title_bar=True, no_resize=False):
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=0, tag="top_spacer_left")
                with dpg.child_window(tag="view_panel", width=800, height=620, border=True):
                    with dpg.group(horizontal=True):
                        dpg.add_text("MOTORHAND", color=C_ACCENT)
                        dpg.add_text("手套力反馈仿真", color=C_TEXT)
                        hint = ("  |  物体固定在手心，握拳即可接触  ·  物体上右键拖动=旋转物体  ·  "
                                "空白处右键=旋转视角  ·  "
                                "滚轮缩放  ·  F11 全屏") if self.sim.palm_mode else (
                                "  |  左键拖动物体  ·  Shift+左键旋转  ·  右键旋转视角  ·  "
                                "滚轮缩放  ·  F11 全屏")
                        dpg.add_text(hint, wrap=650, color=C_MUTED)
                    dpg.add_spacer(height=4)
                    dpg.add_image("scene_texture", width=IMAGE_W, height=IMAGE_H, tag="scene_image")
                with dpg.child_window(tag="control_panel", width=430, height=620, border=True):
                    dpg.add_text("数据源与安全", color=C_GOLD, tag="control_start")
                    dpg.add_text("", tag="source_text", color=C_ACCENT)
                    dpg.add_text("", tag="status_text", color=C_TEXT)
                    dpg.add_separator()
                    dpg.add_text("手心物体  ·  实时重建 10 Hz" if self.sim.palm_mode
                                 else "物体参数  ·  实时重建 10 Hz", color=C_GOLD)
                    names = [s.display_name for s in self.runner.specs]
                    selected_spec = self._selected_spec()
                    dpg.add_combo(names, default_value=selected_spec.display_name if selected_spec else "",
                                  label="物体", tag="object_combo", callback=self._locked(self._on_select))
                    size_lo, size_hi = self._size_range(self._selected_spec())
                    dpg.add_slider_float(label="大小 mm", default_value=float(
                                             selected_spec.size_m * 1000.0 if selected_spec else 30.0),
                                         min_value=size_lo * 1000, max_value=size_hi * 1000,
                                         tag="size_slider", callback=self._locked(self._on_size))
                    dpg.add_slider_float(label="硬度 0..1", default_value=float(
                                             selected_spec.stiffness if selected_spec else 0.25),
                                         min_value=float(self.config["objects"]["stiffness_range"][0]),
                                         max_value=float(self.config["objects"]["stiffness_range"][1]),
                                         tag="stiff_slider", callback=self._locked(self._on_stiff))
                    dpg.add_text("大小 = 直径/边长；改大小后自动重新放到手心。硬度越高，接触越硬、"
                                 "形变越小；只有软球是柔性晶格。" if self.sim.palm_mode else
                                 "硬度越高，接触越硬、形变越小；只有软球是柔性晶格。",
                                 wrap=400, color=C_MUTED)
                    if self._proxy_mode:
                        dpg.add_combo(list(FEEDBACK_MODES.values()),
                                      default_value=FEEDBACK_MODES[self._feedback_mode()],
                                      label="反馈方式", width=200, tag="mode_combo",
                                      callback=self._locked(self._on_mode))
                        dpg.add_slider_float(label="反馈强度 ×", default_value=self._gain_value(),
                            min_value=0.0, max_value=3.0, tag="gain_slider",
                            callback=self._locked(self._on_gain))
                        glove = self.config["glove"]
                        dpg.add_slider_float(label="摩擦补偿 ×", default_value=float(
                            glove.get("friction_comp", {}).get("gain", 1.0)),
                            min_value=0.0, max_value=2.0, tag="friction_slider",
                            enabled=self._feedback_mode() == "driver_impedance",
                            callback=self._locked(self._on_friction))
                        dpg.add_slider_float(label="J1 拇指根力度 ×", default_value=self._j1_gain_value(),
                            min_value=0.5, max_value=3.0, tag="j1_gain_slider",
                            enabled=self._feedback_mode() == "driver_impedance",
                            callback=self._locked(self._on_j1_gain))
                        dpg.add_text("按进越深力越大，硬度决定刚度；上限与急停始终生效。",
                                     wrap=400, color=C_MUTED)
                    else:
                        dpg.add_slider_float(label="反馈增益 Nm/N", default_value=float(
                            self.config["glove"].get("feedback_gain", 10.0)),
                            min_value=0.0, max_value=50.0, tag="gain_slider",
                            callback=self._locked(self._on_gain))
                        dpg.add_text("增益缩放名义力矩；每轴上限与急停仍生效。",
                                     wrap=400, color=C_MUTED)
                    dpg.add_separator()
                    dpg.add_text("控制", color=C_GOLD)
                    dpg.add_checkbox(label="启用力反馈（真机需设备在线）",
                                     default_value=bool(self.runner.feedback),
                                     tag="feedback_check", callback=self._locked(self._on_feedback))
                    dpg.add_checkbox(label="锁定物体（关闭后手套可推动/带动）",
                                     default_value=bool(self.sim.objects_locked),
                                     show=not self.sim.palm_mode,
                                     tag="lock_check", callback=self._locked(self._on_lock))
                    dpg.add_checkbox(label="显示碰撞体 / 接触点 / 接触力",
                                     default_value=False, tag="collision_check",
                                     callback=self._locked(self._on_show_collision))
                    dpg.add_checkbox(label="暂停假手套演示（方便摆放物体）",
                                     default_value=False,
                                     tag="demo_check", callback=self._locked(self._on_demo))
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="开始记录", tag="record_button", width=180,
                                       callback=self._locked(self._on_record))
                        dpg.add_button(label="急停 / 清零", tag="stop_button", width=180,
                                       callback=self._locked(self._on_stop))
                    dpg.add_button(label="回放分析最近的记录", tag="review_button", width=368,
                                   callback=self._on_review)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="默认视角", width=180, callback=self._locked(self._reset_camera))
                        dpg.add_button(label="全屏 F11", width=180, callback=self._locked(self._toggle_fullscreen))
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="聚焦", width=115, callback=self._locked(self._on_focus))
                        dpg.add_button(label="放到指尖", width=115, show=not self.sim.palm_mode,
                                       callback=self._locked(self._on_place))
                        dpg.add_button(label="放回手心" if self.sim.palm_mode else "重置物体", width=115,
                                       callback=self._locked(self._on_reset_objects))
                    with dpg.collapsing_header(label="来源与限制", default_open=False):
                        dpg.add_text(
                            "接触力 = MuJoCo 仿真接触力；力矩 = 未标定名义指令。\n"
                            "锁定物体时可见手型按手指限位，接触力按未限位目标计算。\n"
                            "滑移不可用（无物体位姿传感器）；穿透量不等于物体应变。",
                            wrap=400, color=C_MUTED,
                        )
                    # Marks the end of the control content; _apply_layout sizes
                    # the bottom panel so this stays visible without scrolling.
                    dpg.add_text("", tag="control_end")  # (spacers report no rect)
                    dpg.bind_item_theme("record_button", self._primary_theme)
                    dpg.bind_item_theme("stop_button", self._danger_theme)
                dpg.add_spacer(width=0, tag="top_spacer_right")
            dpg.add_separator()
            with dpg.child_window(tag="bottom_panel", height=300, width=1400, border=True):
                with dpg.tab_bar():
                    with dpg.tab(label="仿真 vs 真机"):
                        with dpg.group(horizontal=True):
                            dpg.add_text("电机", color=C_MUTED)
                            dpg.add_combo(
                                [f"J{i + 1} {FINGER_CN.get(f, f)}" for i, f in enumerate(MOTOR_FINGER)],
                                default_value=f"J1 {FINGER_CN.get(MOTOR_FINGER[0], MOTOR_FINGER[0])}",
                                width=110, tag="cmp_motor",
                                callback=self._locked(self._on_cmp_motor))
                            dpg.add_button(label="暂停曲线", width=96, tag="pause_plots_button",
                                           callback=self._locked(self._on_pause_plots))
                            dpg.add_text("滚轮：纵轴缩放（以光标为中心）｜Ctrl+滚轮：时间窗｜双击：恢复自动",
                                         color=C_MUTED)
                        with dpg.group(horizontal=True):
                            with dpg.plot(label="力矩 (Nm)", height=220, width=450, no_title=True,
                                          tag="cmp_torque_plot"):
                                dpg.add_plot_legend(outside=True, location=dpg.mvPlot_Location_North, horizontal=True)
                                dpg.add_plot_axis(dpg.mvXAxis, tag="cmp_torque_x")
                                with dpg.plot_axis(dpg.mvYAxis, label="力矩 Nm", tag="cmp_torque_y"):
                                    dpg.add_line_series([], [], label="虚拟弹簧(计算)",
                                                        tag="cmp_tau_sim")
                                    dpg.add_line_series([], [], label="驱动器输出(推算)",
                                                        tag="cmp_tau_cmd")
                                    dpg.add_line_series([], [], label="下发指令(计算)",
                                                        tag="cmp_tau_sent")
                                    dpg.add_line_series([], [], label="电流×Kt(实测估计)",
                                                        tag="cmp_tau_meas")
                            with dpg.plot(label="位置 (rad)", height=220, width=450, no_title=True,
                                          tag="cmp_pos_plot"):
                                dpg.add_plot_legend(outside=True, location=dpg.mvPlot_Location_North, horizontal=True)
                                dpg.add_plot_axis(dpg.mvXAxis, tag="cmp_pos_x")
                                with dpg.plot_axis(dpg.mvYAxis, label="位置 rad", tag="cmp_pos_y"):
                                    dpg.add_line_series([], [], label="仿真手",
                                                        tag="cmp_q_sim")
                                    dpg.add_line_series([], [], label="真机编码器",
                                                        tag="cmp_q_real")
                                    dpg.add_scatter_series([], [], label="超出标定范围",
                                                           tag="cmp_q_clamp")
                            with dpg.plot(label="电流 (A)", height=220, width=450, no_title=True,
                                          tag="cmp_cur_plot"):
                                dpg.add_plot_legend(outside=True, location=dpg.mvPlot_Location_North, horizontal=True)
                                dpg.add_plot_axis(dpg.mvXAxis, tag="cmp_cur_x")
                                with dpg.plot_axis(dpg.mvYAxis, label="电流 A", tag="cmp_cur_y"):
                                    dpg.add_line_series([], [], label=self._current_label("实测电流"),
                                                        tag="cmp_cur_real")
                        for tag, theme in (("cmp_tau_sim", self._line_sim), ("cmp_tau_cmd", self._line_cmd),
                                           ("cmp_tau_sent", self._line_sent), ("cmp_tau_meas", self._line_real),
                                           ("cmp_q_sim", self._line_sim),
                                           ("cmp_q_real", self._line_real), ("cmp_cur_real", self._line_real)):
                            dpg.bind_item_theme(tag, theme)
                    with dpg.tab(label="力 / 力矩 / 关节"):
                        with dpg.group(horizontal=True):
                            with dpg.plot(label="每指仿真接触力 (N)", height=250, width=450,
                                          tag="force_plot"):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label="t (s)", tag="force_x")
                                with dpg.plot_axis(dpg.mvYAxis, label="N", tag="force_y"):
                                    for finger in FINGERS + ("palm",):
                                        dpg.add_line_series([], [], label=finger,
                                                            tag=f"force_line_{finger}")
                            with dpg.plot(label="名义反馈力矩指令 (Nm, 每指最大值)", height=250,
                                          width=450, tag="torque_plot"):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label="t (s)", tag="torque_x")
                                with dpg.plot_axis(dpg.mvYAxis, label="Nm", tag="torque_y"):
                                    for finger in FINGERS:
                                        dpg.add_line_series([], [], label=finger,
                                                            tag=f"torque_line_{finger}")
                            with dpg.plot(label="仿真关节位置 (rad)", height=250, width=450,
                                          tag="q_plot"):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label="t (s)", tag="q_x")
                                with dpg.plot_axis(dpg.mvYAxis, label="rad", tag="q_y"):
                                    for i in range(12):
                                        dpg.add_line_series([], [], label=f"J{i+1}",
                                                            tag=f"q_line_{i+1}")
                    with dpg.tab(label="电流 / 力-位移 / 时延"):
                        with dpg.group(horizontal=True):
                            with dpg.plot(label=self._current_label("电流 (A, 每指最大)"), height=250, width=450,
                                          tag="current_plot"):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label="t (s)", tag="current_x")
                                with dpg.plot_axis(dpg.mvYAxis, label="A", tag="current_y"):
                                    for finger in FINGERS:
                                        dpg.add_line_series([], [], label=finger,
                                                            tag=f"current_line_{finger}")
                            with dpg.plot(label="力—接触穿透/柔体形变 (仿真)", height=250, width=450,
                                          tag="disp_plot"):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label="穿透 / 柔体形变 (m)", tag="disp_x")
                                with dpg.plot_axis(dpg.mvYAxis, label="N", tag="disp_y"):
                                    dpg.add_scatter_series([], [], label="总力/最大穿透",
                                                           tag="disp_line")
                                    dpg.add_scatter_series([], [], label="总力/柔体形变",
                                                           tag="deform_line")
                            with dpg.plot(label="传感器时延 (ms)", height=250, width=450,
                                          tag="lat_plot"):
                                dpg.add_plot_axis(dpg.mvXAxis, label="t (s)", tag="lat_x")
                                with dpg.plot_axis(dpg.mvYAxis, label="ms", tag="lat_y"):
                                    dpg.add_line_series([], [], label="sensor age", tag="lat_line")
                    with dpg.tab(label="速度 / 保持"):
                        with dpg.plot(label="关节速度 (rad/s)：粗线 Jn = 真机编码器估计（阻尼/摩擦补偿所用），"
                                            "细线 Jn 仿真 = 仿真手；点图例可隐藏", height=250, width=1380, tag="qd_plot"):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="t (s)", tag="qd_x")
                            with dpg.plot_axis(dpg.mvYAxis, label="rad/s", tag="qd_y"):
                                for i in range(12):
                                    dpg.add_line_series([], [], label=f"J{i+1}",
                                                        tag=f"qd_line_{i+1}")
                                for i in range(12):
                                    dpg.add_line_series([], [], label=f"J{i+1} 仿真",
                                                        tag=f"qd_sim_line_{i+1}")
                        dpg.add_text("保持/滑移状态见右侧状态栏；滑移在物体锁定模式下不可用。",
                                     color=C_MUTED)
        for i, finger in enumerate(FINGERS + ("palm",)):
            for prefix in ("force_line_", "torque_line_", "current_line_"):
                if dpg.does_item_exist(prefix + finger):
                    dpg.bind_item_theme(prefix + finger, self._joint_themes[i])
        for i in range(12):
            dpg.bind_item_theme(f"q_line_{i+1}", self._joint_themes[i])
            dpg.bind_item_theme(f"qd_line_{i+1}", self._joint_themes[i])
            dpg.bind_item_theme(f"qd_sim_line_{i+1}", self._joint_sim_themes[i])
        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=self._locked(self._on_wheel))
            dpg.add_mouse_double_click_handler(button=dpg.mvMouseButton_Left,
                                               callback=self._locked(self._on_double_click))
            key_f11 = getattr(dpg, "mvKey_F11", None)
            if key_f11 is not None:
                dpg.add_key_press_handler(key=key_f11,
                                          callback=lambda *args: self._toggle_fullscreen())

    # ------------------------------------------------------------------
    def _locked(self, fn):
        """Run a DearPyGui callback under the runner lock (control thread)."""
        import inspect

        lock = self.runner.lock
        # DearPyGui decides how many arguments to pass from the callable's
        # positional parameter count, so the wrapper must declare them
        # explicitly (a *args wrapper receives none).
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        n = len(params)

        def wrapper(sender=None, app_data=None, user_data=None):
            with lock:
                return fn(*(sender, app_data, user_data)[:n])

        return wrapper

    def _rebuild_renderer(self) -> None:
        # MuJoCo GL runs in a separate process; never create an offscreen
        # Renderer in this process or DearPyGui's GL context turns black.
        if self.renderer is None or not self.renderer.alive:
            self.renderer = RenderProcess()
        self._scene = mujoco.MjvScene(self.sim.model, maxgeom=2000)
        self._model_ref = self.sim.model

    def _camera_basis(self):
        if self._scene is None or self.sim.model is not self._model_ref:
            self._rebuild_renderer()
        assert self._scene is not None
        mujoco.mjv_updateScene(self.sim.model, self.sim.data, mujoco.MjvOption(), None,
                               self.cam, mujoco.mjtCatBit.mjCAT_ALL, self._scene)
        glcam = self._scene.camera[0]
        forward = np.asarray(glcam.forward, dtype=float)
        up = np.asarray(glcam.up, dtype=float)
        right = np.cross(forward, up)
        right /= max(1e-12, np.linalg.norm(right))
        tan_v = float(glcam.frustum_top / glcam.frustum_near)
        return np.asarray(glcam.pos, dtype=float), forward, right, up, tan_v

    def _project(self, point: np.ndarray):
        pos, forward, right, up, tan_v = self._camera_basis()
        rel = np.asarray(point, dtype=float) - pos
        depth = float(rel @ forward)
        if depth <= 1e-6:
            return None
        tan_h = tan_v * RENDER_W / RENDER_H
        px = (float(rel @ right) / (depth * tan_h) + 1.0) * 0.5 * RENDER_W
        py = (1.0 - float(rel @ up) / (depth * tan_v)) * 0.5 * RENDER_H
        return px, py, depth

    def _hit_object(self, image_x: float, image_y: float) -> bool:
        """Is the cursor over the (palm) object's projected silhouette?"""
        for obj in self.sim.object_runtime:
            proj = self._project(obj.target_position)
            if proj is None:
                continue
            px, py, depth = proj
            _pos, _f, _r, _u, tan_v = self._camera_basis()
            radius_px = self.sim.bounding_radius(obj.spec) / (depth * tan_v) * 0.5 * RENDER_H
            if math.hypot(image_x - px, image_y - py) <= max(25.0, 1.1 * radius_px):
                return True
        return False

    def _pick(self, image_x: float, image_y: float) -> str | None:
        if self.sim.palm_mode:
            return None  # the palm object is fixed; no mouse dragging
        best = None
        # The offscreen camera's horizontal principal point differs slightly
        # from the analytic projection, so use a generous radius and pick the
        # nearest projected object.
        best_dist = 95.0
        for obj in self.sim.object_runtime:
            proj = self._project(obj.target_position)
            if proj is None:
                continue
            px, py, _ = proj
            dist = math.hypot(image_x - px, image_y - py)
            if dist < best_dist:
                best_dist = dist
                best = obj.name
        return best

    # ------------------------------------------------------------------
    def _size_range(self, spec) -> tuple[float, float]:
        """Allowed size (m) for an object; spheres may be larger than the rest."""
        objects = self.config["objects"]
        if self.sim.palm_mode and (objects.get("palm") or {}).get("size_range_m"):
            lo, hi = objects["palm"]["size_range_m"]
            return float(lo), float(hi)
        default = objects["size_range_m"]
        by_shape = objects.get("size_range_m_by_shape") or {}
        lo, hi = by_shape.get(getattr(spec, "shape", ""), default)
        return float(lo), float(hi)

    def _on_select(self, sender, app_data) -> None:
        label = str(app_data)
        spec = next((s for s in self.runner.specs if s.display_name == label or s.name == label), None)
        if spec is None:
            return
        self.selected = spec.name
        if self.sim.palm_mode:
            # Swapping the object recompiles the scene; do it in the UI loop
            # together with the renderer rebuild.
            self._pending_select = spec.name
        for spec in self.runner.specs:
            if spec.name == self.selected:
                self._syncing = True
                lo, hi = self._size_range(spec)
                dpg.configure_item("size_slider", min_value=lo * 1000.0, max_value=hi * 1000.0)
                dpg.set_value("size_slider", float(spec.size_m) * 1000.0)
                dpg.set_value("stiff_slider", float(spec.stiffness))
                self._syncing = False
                break

    def _on_size(self, sender, app_data) -> None:
        if self._syncing:
            return
        self.pending.setdefault(self.selected, {})["size_m"] = float(app_data) / 1000.0

    def _on_stiff(self, sender, app_data) -> None:
        if self._syncing:
            return
        self.pending.setdefault(self.selected, {})["stiffness"] = float(app_data)

    def _feedback_mode(self) -> str:
        return str(self.config["glove"].get("feedback_mode", "proxy_spring"))

    def _gain_value(self) -> float:
        glove = self.config["glove"]
        if self._feedback_mode() == "driver_impedance":
            return float(glove.get("driver_impedance", {}).get("gain", 1.0))
        return float(glove.get("proxy_gain", 1.0))

    def _on_gain(self, sender, app_data) -> None:
        if self._syncing:
            return
        glove = self.config["glove"]
        if self._feedback_mode() == "driver_impedance":
            glove.setdefault("driver_impedance", {})["gain"] = float(app_data)
        elif self._proxy_mode:
            glove["proxy_gain"] = float(app_data)
        else:
            glove["feedback_gain"] = float(app_data)

    def _joint_gains(self) -> list[float]:
        cfg = self.config["glove"].setdefault("driver_impedance", {})
        jg = np.asarray(cfg.get("joint_gain", 1.0), dtype=float).reshape(-1)
        gains = [float(jg[0])] * 12 if jg.size == 1 else [float(v) for v in jg]
        cfg["joint_gain"] = gains
        return gains

    def _j1_gain_value(self) -> float:
        return self._joint_gains()[0]

    def _on_j1_gain(self, sender, app_data) -> None:
        gains = self._joint_gains()
        gains[0] = float(app_data)
        # New list object so the session recorder logs the change as an event.
        self.config["glove"]["driver_impedance"]["joint_gain"] = list(gains)

    def _on_friction(self, sender, app_data) -> None:
        self.config["glove"].setdefault("friction_comp", {})["gain"] = float(app_data)

    def _on_mode(self, sender, app_data) -> None:
        mode = next((k for k, v in FEEDBACK_MODES.items() if v == app_data), "driver_impedance")
        self.config["glove"]["feedback_mode"] = mode
        # Re-engage contacts with the new mode's stiffness range.
        self.sim._blocked.clear()
        dpg.set_value("gain_slider", self._gain_value())
        dpg.configure_item("friction_slider", enabled=mode == "driver_impedance")
        dpg.configure_item("j1_gain_slider", enabled=mode == "driver_impedance")

    def _on_feedback(self, sender, app_data) -> None:
        try:
            self.runner.set_feedback(bool(app_data))
            self._last_error = "" if self.runner.feedback else "反馈已关闭"
        except Exception as error:
            self._last_error = f"无法启用力反馈: {error}"
            self.runner.feedback = False
            dpg.set_value("feedback_check", False)

    def _on_lock(self, sender, app_data) -> None:
        try:
            self.sim.set_objects_locked(bool(app_data))
            self._last_error = "" if app_data else "物体已解锁：可被手套推动/带动"
        except Exception as error:
            self._last_error = f"切换物体锁定失败: {error}"

    def _on_show_collision(self, sender, app_data) -> None:
        self._show_collision = bool(app_data)

    def _on_demo(self, sender, app_data) -> None:
        # The checkbox is labelled "暂停假手套演示", so checked means paused;
        # FakeGlove.set_demo(True) means the animated demo is running.
        paused = bool(app_data)
        try:
            self.runner.set_demo(not paused)
            self._last_error = "假手套演示已暂停" if paused else ""
        except Exception as error:
            self._last_error = f"切换演示失败: {error}"
            dpg.set_value("demo_check", False)

    def _selected_spec(self):
        return next((s for s in self.runner.specs if s.name == self.selected), None)

    def _on_focus(self, sender, app_data) -> None:
        obj = next((o for o in self.sim.object_runtime if o.name == self.selected), None)
        if obj is None:
            return
        self.cam.lookat[:] = obj.target_position
        self.cam.distance = 0.30

    def _on_place(self, sender, app_data) -> None:
        obj = next((o for o in self.sim.object_runtime if o.name == self.selected), None)
        spec = self._selected_spec()
        if obj is None or spec is None:
            return
        self.sim.end_drag()
        tips = []
        for gid in range(self.sim.model.ngeom):
            name = self.sim.model.geom(gid).name or ""
            if name.startswith("col_index_index_dip") or name.startswith("col_middle_middle_dip"):
                tips.append(np.asarray(self.sim.data.geom_xpos[gid], dtype=float))
        if not tips:
            return
        tip = np.mean(tips, axis=0)
        pos = np.array([tip[0], tip[1], tip[2] + 0.5 * spec.size_m + 0.054])
        # Nudge sideways if the slot is occupied by another object.
        pos = self.sim.free_position(spec.name, pos)
        try:
            self.sim.set_object_pose(spec.name, position=pos)
        except Exception as error:
            self._last_error = f"放置物体失败: {error}"

    def _on_reset_objects(self, sender, app_data) -> None:
        try:
            self.sim.reset_object_poses()
            self._last_error = ""
        except Exception as error:
            self._last_error = f"重置物体失败: {error}"

    def _toggle_fullscreen(self) -> None:
        try:
            if not self._fullscreen:
                self._windowed_client = (
                    int(dpg.get_viewport_client_width()),
                    int(dpg.get_viewport_client_height()),
                )
                dpg.toggle_viewport_fullscreen()
                self._fullscreen = True
            else:
                dpg.toggle_viewport_fullscreen()
                self._fullscreen = False
                # DearPyGui restores a slightly shorter client area (the WM
                # title-bar height is lost), which clipped the lower buttons.
                # Apply the saved client size on the next UI frame, after the
                # fullscreen-exit resize has settled.
                width, height = self._windowed_client
                if width > 0 and height > 0:
                    self._pending_window_size = (width, height)
            self._view_size = (0, 0)
        except Exception as error:
            self._last_error = f"切换全屏失败: {error}"

    def _on_stop(self, sender, app_data) -> None:
        self.runner.emergency_stop()
        dpg.set_value("feedback_check", False)
        self._last_error = self.runner.stats.error

    def _record_dir(self) -> Path:
        directory = Path(self.config.get("telemetry", {}).get("default_dir", "data"))
        if not directory.is_absolute():
            directory = Path(self.config.get("_sim_root", ".")) / directory
        return directory

    def _on_record(self, sender, app_data) -> None:
        if self.runner.recorder is None:
            try:
                path = self.runner.start_recording(str(self._record_dir()))
                self._recording_path = path
                self._last_session = path
                dpg.configure_item("record_button", label="停止记录")
            except Exception as error:
                self._last_error = f"记录失败: {error}"
        else:
            path = self.runner.stop_recording()
            if path:
                self._last_session = path
                self._recording_path = f"{path}（已保存）"
            dpg.configure_item("record_button", label="开始记录")

    def _on_review(self, sender=None, app_data=None) -> None:
        """Open the review window for the newest (or current) session in a new process."""
        import subprocess
        import sys
        from ..recording import latest_session
        path = self._last_session or latest_session(self._record_dir())
        if not path:
            self._last_error = "还没有记录：先点“开始记录”，操作一会儿再停止"
            return
        if self.runner.recorder is not None and str(self.runner.recorder.path) == str(path):
            # Reviewing a session that is still being written: flush what we have.
            self._last_error = "正在记录中：回放只包含已写入的部分（约 1 秒延迟）"
        subprocess.Popen([sys.executable, "-m", "glove_sim.cli", "review", str(path)],
                         start_new_session=True)

    def _on_wheel(self, sender, app_data) -> None:
        plot = self._hovered_plot()
        if plot is not None:
            self._zoom_plot(plot, float(app_data))
        elif dpg.is_item_hovered("scene_image"):
            self._wheel += float(app_data)

    @staticmethod
    def _hovered_plot() -> str | None:
        for tag in PLOT_AXES:
            try:
                if dpg.is_item_hovered(tag):
                    return tag
            except Exception:
                continue
        return None

    @staticmethod
    def _ctrl_down() -> bool:
        return any(bool(dpg.is_key_down(getattr(dpg, k)))
                   for k in ("mvKey_LControl", "mvKey_RControl") if hasattr(dpg, k))

    def _zoom_plot(self, plot: str, wheel: float) -> None:
        x_tag, y_tag, is_time = PLOT_AXES[plot]
        factor = math.exp(-0.15 * wheel)  # wheel up = zoom in
        if self._ctrl_down() and is_time:
            max_span = PLOT_HISTORY / SAMPLE_HZ
            self._time_span = float(min(max_span, max(0.5, self._time_span * factor)))
            return
        tag, coord = (x_tag, 0) if self._ctrl_down() else (y_tag, 1)
        lo, hi = self._zoom.get(tag) or self._auto_lims.get(tag) or (-1.0, 1.0)
        try:
            centre = float(dpg.get_plot_mouse_pos()[coord])
        except Exception:
            centre = 0.5 * (lo + hi)
        if not (lo <= centre <= hi) or not math.isfinite(centre):
            centre = 0.5 * (lo + hi)
        new = (centre - (centre - lo) * factor, centre + (hi - centre) * factor)
        if new[1] - new[0] < 1e-6:
            return
        self._zoom[tag] = new
        dpg.set_axis_limits(tag, *new)

    def _on_double_click(self, sender=None, app_data=None) -> None:
        plot = self._hovered_plot()
        if plot is None:
            return
        x_tag, y_tag, is_time = PLOT_AXES[plot]
        self._zoom.pop(y_tag, None)
        self._zoom.pop(x_tag, None)
        if is_time:
            self._time_span = DEFAULT_TIME_SPAN_S

    def _on_pause_plots(self, sender=None, app_data=None) -> None:
        self._plots_paused = not self._plots_paused
        self._frozen_hist = list(self._history) if self._plots_paused else []
        dpg.configure_item("pause_plots_button",
                           label="继续曲线" if self._plots_paused else "暂停曲线")
        dpg.bind_item_theme("pause_plots_button",
                            self._danger_theme if self._plots_paused else 0)

    def _on_cmp_motor(self, sender, app_data) -> None:
        try:
            self._cmp_motor = int(str(app_data).split()[0][1:]) - 1
        except (ValueError, IndexError):
            self._cmp_motor = 0
        self._zoom.pop("cmp_torque_y", None)
        self._zoom.pop("cmp_pos_y", None)
        self._zoom.pop("cmp_cur_y", None)
        if self._plots_paused:
            self._update_plots()  # redraw the frozen snapshot for the new motor

    def _reset_camera(self) -> None:
        self.cam.lookat[:] = self._cam_home[0]
        self.cam.distance = self._cam_home[1]
        self.cam.azimuth = self._cam_home[2]
        self.cam.elevation = self._cam_home[3]

    # ------------------------------------------------------------------
    def _apply_pending(self) -> None:
        """Hand the latest object edits to the worker (never blocks the UI)."""
        edits: dict[str, object] = {}
        if self._pending_select is not None:
            edits["select"], self._pending_select = self._pending_select, None
        if self.pending:
            now = time.monotonic()
            hz = float(self.config.get("objects", {}).get("rebuild_hz", 10.0))
            if now - self._last_rebuild >= 1.0 / max(1.0, hz):
                self._last_rebuild = now
                edits["shape"] = {k: dict(v) for k, v in self.pending.items()}
                self.pending.clear()
        if self._yaw_target is not None:
            edits["yaw"] = (self.selected, self._yaw_target)
            self._yaw_target = None
        if not edits:
            return
        with self._edit_cv:
            for key, value in edits.items():
                if key == "shape":
                    merged = dict(self._edit_queue.get("shape") or {})
                    for name, changes in value.items():
                        merged.setdefault(name, {}).update(changes)
                    self._edit_queue["shape"] = merged
                else:
                    self._edit_queue[key] = value   # only the latest matters
            self._edit_cv.notify()

    def _edit_worker(self) -> None:
        lock = self.runner.lock
        sim = self.sim
        while True:
            with self._edit_cv:
                while not self._edit_queue and not self._edit_stop:
                    self._edit_cv.wait(0.5)
                if self._edit_stop:
                    return
                queue, self._edit_queue = self._edit_queue, {}
                self._edit_busy = True
            try:
                reqs = []
                name = queue.get("select")
                if name is not None:
                    spec = next((sp for sp in sim.catalog if sp.name == name), None)
                    if spec is not None:
                        prebuilt = sim.compile_specs([spec])          # no lock
                        with lock:
                            reqs.append(sim.select_object(name, prebuilt=prebuilt, refit=False))
                for name, changes in (queue.get("shape") or {}).items():
                    specs = sim.specs_with(name, size_m=changes.get("size_m"),
                                           stiffness=changes.get("stiffness"))
                    prebuilt = sim.compile_specs(specs) if specs is not None else None
                    with lock:
                        reqs.append(sim.set_object_shape(
                            name, size_m=changes.get("size_m"), stiffness=changes.get("stiffness"),
                            prebuilt=prebuilt, refit=False))
                        if changes.get("size_m") is not None and not sim.palm_mode:
                            # A grown object must not overlap its neighbours.
                            obj = next(o for o in sim.object_runtime if o.name == name)
                            pos = sim.free_position(name, obj.target_position)
                            if not np.allclose(pos, obj.target_position):
                                sim.set_object_pose(name, position=pos)
                if queue.get("yaw") is not None:
                    name, yaw = queue["yaw"]
                    with lock:
                        reqs.append(sim.set_palm_yaw(name, yaw, refit=False))
                req = next((r for r in reversed(reqs) if r is not None), None)
                if req is not None:
                    res = sim.refit_compute(req)                       # no lock
                    with lock:
                        sim.refit_apply(res)
                self._last_error = ""
            except Exception as error:
                self._last_error = f"修改物体失败: {error}"
            finally:
                self._edit_busy = False

    def _mouse(self) -> None:
        left = bool(dpg.is_mouse_button_down(dpg.mvMouseButton_Left))
        right = bool(dpg.is_mouse_button_down(dpg.mvMouseButton_Right))
        if not dpg.is_item_hovered("scene_image") and not (self._mouse_down or self._right_down):
            return
        mx, my = dpg.get_mouse_pos(local=False)
        rect_min = dpg.get_item_rect_min("scene_image")
        rect_max = dpg.get_item_rect_max("scene_image")
        rect_w = max(1.0, rect_max[0] - rect_min[0])
        rect_h = max(1.0, rect_max[1] - rect_min[1])
        inside = rect_min[0] <= mx <= rect_max[0] and rect_min[1] <= my <= rect_max[1]
        # The render texture is fixed at RENDER_W x RENDER_H but the image can
        # be displayed larger (or smaller), so all picking/drag maths must use
        # the actual widget rectangle.
        local_x = (mx - rect_min[0]) * RENDER_W / rect_w
        local_y = (my - rect_min[1]) * RENDER_H / rect_h

        # Right button: pressed on the palm object -> rotate it about the palm
        # normal (horizontal drag, 0.5 deg/px); anywhere else -> orbit camera.
        if right:
            if not self._right_down:
                self._right_down = True
                self._right_last = (mx, my)
                self._rotating = None
                if self.sim.palm_mode and inside and self._hit_object(local_x, local_y):
                    self._rotating = self.sim.active_object
                    self._rotate_last_x = mx
                # Cancel a left-button object drag so both gestures cannot
                # fight over the same mouse delta.
                if self._mouse_down:
                    if self._drag_name is not None:
                        self.sim.end_drag()
                    self._mouse_down = False
                    self._drag_name = None
                    self._drag_mode = None
                    self._last_mouse = None
            elif self._rotating is not None:
                dx = mx - self._rotate_last_x
                if abs(dx) >= 1.0:
                    base = self._yaw_target if self._yaw_target is not None else self.sim.palm_yaw(self._rotating)
                    self._yaw_target = base + math.radians(0.5 * dx)
                    self._rotate_last_x = mx
            elif self._right_last is not None:
                dx = mx - self._right_last[0]
                dy = my - self._right_last[1]
                self.cam.azimuth -= dx * 0.35
                self.cam.elevation = max(-89.0, min(20.0, self.cam.elevation + dy * 0.25))
                self._right_last = (mx, my)
        else:
            self._right_down = False
            self._right_last = None
            self._rotating = None

        # Left button only moves/rotates a picked object.  A left drag on empty
        # space does nothing (orbit is right-button only now).
        if left and not right:
            if not self._mouse_down:
                if inside:
                    self._drag_name = self._pick(local_x, local_y)
                    if os.environ.get("GLOVE_SIM_DEBUG_UI"):
                        proj = {o.name: self._project(o.target_position) for o in self.sim.object_runtime}
                        print(f"[UI] mouse=({mx:.1f},{my:.1f}) rect=({rect_min[0]:.1f},"
                              f"{rect_min[1]:.1f})-({rect_max[0]:.1f},{rect_max[1]:.1f}) "
                              f"local=({local_x:.1f},{local_y:.1f}) pick={self._drag_name} "
                              f"proj={proj}", flush=True)
                    if self._drag_name is not None:
                        shift = bool(dpg.is_key_down(dpg.mvKey_Shift)
                                     if hasattr(dpg, "mvKey_Shift") else False)
                        self._drag_mode = "rotate" if shift else "move"
                        try:
                            self.sim.begin_drag(self._drag_name)
                        except Exception as error:
                            self._last_error = f"开始拖动失败: {error}"
                            self._drag_name = None
                        for spec in self.runner.specs:
                            if spec.name == self._drag_name:
                                self._syncing = True
                                dpg.set_value("object_combo", spec.name)
                                lo, hi = self._size_range(spec)
                                dpg.configure_item("size_slider", min_value=lo * 1000.0,
                                                   max_value=hi * 1000.0)
                                dpg.set_value("size_slider", spec.size_m * 1000.0)
                                dpg.set_value("stiff_slider", spec.stiffness)
                                self._syncing = False
                                self.selected = spec.name
                                break
                    else:
                        self._drag_mode = None
                    self._last_mouse = (mx, my)
                self._mouse_down = True
            elif self._last_mouse is not None and self._drag_name is not None:
                dx = mx - self._last_mouse[0]
                dy = my - self._last_mouse[1]
                obj = next((o for o in self.sim.object_runtime if o.name == self._drag_name), None)
                if obj is not None:
                    if self._drag_mode == "move":
                        cam_pos, forward, right_v, up, tan_v = self._camera_basis()
                        # Use the object's actual depth and the displayed image
                        # height so one mouse pixel moves one screen pixel.
                        depth = float((np.asarray(obj.target_position) - cam_pos) @ forward)
                        scale = 2.0 * max(depth, 1e-3) * tan_v / rect_h
                        delta = right_v * (dx * scale) - up * (dy * scale)
                        new_pos = np.asarray(obj.target_position, dtype=float) + delta
                        new_pos[0] = np.clip(new_pos[0], -0.05, 0.42)
                        new_pos[1] = np.clip(new_pos[1], -0.20, 0.20)
                        new_pos[2] = np.clip(new_pos[2], 0.010, 0.25)
                        self.sim.update_drag(position=new_pos)
                    else:
                        from scipy.spatial.transform import Rotation
                        yaw = -dx * 0.008
                        pitch = -dy * 0.008
                        current = Rotation.from_quat(_quat_from_mujoco(obj.target_quat))
                        axis = np.array([0.0, 0.0, 1.0])
                        _, _, right_v, _, _ = self._camera_basis()
                        rotate = (Rotation.from_rotvec(axis * yaw) *
                                  Rotation.from_rotvec(right_v * pitch))
                        new_quat = _quat_to_mujoco((rotate * current).as_quat())
                        self.sim.update_drag(quat=new_quat)
                self._last_mouse = (mx, my)
        elif not left:
            if self._mouse_down:
                if self._drag_name is not None:
                    self.sim.end_drag()
                self._mouse_down = False
                self._drag_name = None
                self._drag_mode = None
                self._last_mouse = None

        if self._wheel:
            self.cam.distance = max(0.15, min(1.6, self.cam.distance * math.exp(-self._wheel * 0.12)))
            self._wheel = 0.0

    # ------------------------------------------------------------------
    def _highlight_payload(self) -> dict | None:
        if not self.selected:
            return None
        obj = next((o for o in self.sim.object_runtime if o.name == self.selected), None)
        spec = next((s for s in self.runner.specs if s.name == self.selected), None)
        if obj is None or spec is None:
            return None
        return {
            "name": str(self.selected),
            "pos": [float(v) for v in obj.target_position],
            "quat": [float(v) for v in obj.target_quat],
            "size": float(spec.size_m),
            "shape": str(spec.shape),
        }

    def _update_scene_texture(self) -> None:
        if self.renderer is None or self.sim.model is not self._model_ref:
            self._rebuild_renderer()
        assert self.renderer is not None
        # Pipelined: every UI frame collects a finished image (if any) and
        # immediately queues the next one, so the renderer is never idle.
        # (The old 30 Hz gate requested on one frame and collected on a later
        # one, which capped the scene at ~15 Hz with 30-60 ms extra age.)
        # Only the state snapshot needs the runner lock; the caller holds it
        # just for this call, and the texture swap below is lock-free.
        try:
            img = self.renderer.poll()
            self.renderer.request(self.sim.xml, self.sim.data.qpos, self.sim.data.qvel,
                                  self.cam, RENDER_W, RENDER_H,
                                  highlight=self._highlight_payload(),
                                  show_collision=self._show_collision)
        except Exception as error:
            self._last_error = f"图像渲染失败: {error}"
            return
        if img is not None and img.shape[:2] == (RENDER_H, RENDER_W):
            # MuJoCo's Renderer already returns a top-down image; do not flip.
            dpg.set_value("scene_texture", img)

    def _update_history(self) -> None:
        """Move the per-tick samples recorded by the control loop into the plot history."""
        samples = self.runner.plot_samples
        while samples:
            self._history.append(samples.popleft())

    @staticmethod
    def _envelope(xs: np.ndarray, ys: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
        """Reduce a dense series to per-bucket (min, max) pairs, keeping every peak."""
        n = xs.size
        if n <= max_points:
            return xs, ys
        buckets = max(1, max_points // 2)
        edges = np.linspace(0, n, buckets + 1).astype(int)
        out_x = np.empty(2 * buckets)
        out_y = np.empty(2 * buckets)
        for b in range(buckets):
            a, e = edges[b], max(edges[b] + 1, edges[b + 1])
            seg = ys[a:e]
            finite = np.isfinite(seg)
            if not finite.any():
                out_x[2 * b:2 * b + 2] = xs[a]
                out_y[2 * b:2 * b + 2] = np.nan
                continue
            i_lo = a + int(np.nanargmin(seg))
            i_hi = a + int(np.nanargmax(seg))
            first, second = sorted((i_lo, i_hi))
            out_x[2 * b], out_y[2 * b] = xs[first], ys[first]
            out_x[2 * b + 1], out_y[2 * b + 1] = xs[second], ys[second]
        return out_x, out_y

    def _update_plots(self) -> None:
        hist = self._frozen_hist if self._plots_paused else list(self._history)
        if not hist:
            return
        t_all = np.asarray([row.t_ns for row in hist], dtype=float) / 1e9 - self._t0
        # Only the visible time window is sent to the widgets.
        first = int(np.searchsorted(t_all, t_all[-1] - self._time_span * 1.02))
        hist = hist[first:]
        t = t_all[first:]
        data: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

        def put(series: str, y_tag: str, xs: np.ndarray, ys, *, envelope: bool = True) -> None:
            ys = np.asarray(ys, dtype=float)
            if envelope:
                xs, ys = self._envelope(xs, ys, PLOT_MAX_POINTS)
            dpg.set_value(series, [xs.tolist(), ys.tolist()])
            data.setdefault(y_tag, []).append((xs, ys))

        def col(field: str, i: int | None = None) -> np.ndarray:
            if i is None:
                return np.asarray([getattr(row, field) for row in hist], dtype=float)
            return np.asarray([getattr(row, field)[i] for row in hist], dtype=float)

        # 仿真 vs 真机 (one motor); all torques in the model-joint frame.
        m = self._cmp_motor
        put("cmp_tau_sim", "cmp_torque_y", t, col("torque_spring_nm", m))
        put("cmp_tau_cmd", "cmp_torque_y", t, col("torque_driver_nm", m))
        put("cmp_tau_sent", "cmp_torque_y", t, col("torque_sent_nm", m))
        put("cmp_tau_meas", "cmp_torque_y", t, col("torque_measured_nm", m))
        put("cmp_q_sim", "cmp_pos_y", t, col("sim_q", m))
        real_q = col("real_q", m)
        put("cmp_q_real", "cmp_pos_y", t, real_q)
        clamped = col("encoder_clamped", m) > 0.5
        dpg.set_value("cmp_q_clamp", [t[clamped].tolist(), real_q[clamped].tolist()])
        put("cmp_cur_real", "cmp_cur_y", t, col("current_a", m))

        forces = np.asarray([row.finger_force_n for row in hist], dtype=float)
        for i, finger in enumerate(FINGERS + ("palm",)):
            put(f"force_line_{finger}", "force_y", t, forces[:, i])
        torques = np.abs(np.asarray([row.torque_driver_nm for row in hist], dtype=float))
        currents = np.abs(np.asarray([row.current_a for row in hist], dtype=float))
        finger_motor = {f: [i for i, mf in enumerate(MOTOR_FINGER) if mf == f] for f in FINGERS}
        for finger in FINGERS:
            idx = finger_motor[finger]
            put(f"torque_line_{finger}", "torque_y", t, torques[:, idx].max(axis=1))
            cur = currents[:, idx]
            cur = np.where(np.isfinite(cur).any(axis=1), np.nanmax(np.where(np.isfinite(cur), cur, -np.inf), axis=1), np.nan)
            put(f"current_line_{finger}", "current_y", t, cur)
        sim_q = np.asarray([row.sim_q for row in hist], dtype=float)
        sim_dq = np.asarray([row.sim_dq for row in hist], dtype=float)
        real_dq = np.asarray([row.real_dq for row in hist], dtype=float)
        for i in range(12):
            put(f"q_line_{i+1}", "q_y", t, sim_q[:, i])
            put(f"qd_line_{i+1}", "qd_y", t, real_dq[:, i])
            put(f"qd_sim_line_{i+1}", "qd_y", t, sim_dq[:, i])
        pens = np.asarray([max(row.penetration_m, default=0.0) for row in hist], dtype=float)
        total_force = forces[:, :5].sum(axis=1)
        put("disp_line", "disp_y", pens, total_force, envelope=False)
        put("deform_line", "disp_y", col("object_disp_m"), total_force, envelope=False)
        put("lat_line", "lat_y", t, col("sensor_age_ms"))
        self._plot_data = data
        self._frozen_t = t
        self._apply_plot_limits(t)

    @staticmethod
    def _auto_range(values: list[np.ndarray]) -> tuple[float, float]:
        finite = [v[np.isfinite(v)] for v in values]
        finite = [v for v in finite if v.size]
        if not finite:
            return (-1.0, 1.0)
        lo = float(min(v.min() for v in finite))
        hi = float(max(v.max() for v in finite))
        span = hi - lo
        pad = max(0.08 * span, 1e-3 if span == 0.0 else 0.0)
        if span < 1e-6:
            pad = max(1e-3, 0.1 * abs(hi))
        return (lo - pad, hi + pad)

    def _apply_plot_limits(self, t: np.ndarray) -> None:
        if t.size < 2:
            return
        t_hi = float(t[-1]) + 0.02 * self._time_span
        t_lo = max(float(t[0]), float(t[-1]) - self._time_span)
        for plot, (x_tag, y_tag, is_time) in PLOT_AXES.items():
            try:
                series = self._plot_data.get(y_tag, [])
                if is_time:
                    dpg.set_axis_limits(x_tag, t_lo, t_hi)
                    visible = [ys[xs >= t_lo] for xs, ys in series]
                else:
                    auto_x = self._auto_range([xs for xs, _ in series])
                    self._auto_lims[x_tag] = auto_x
                    dpg.set_axis_limits(x_tag, *self._zoom.get(x_tag, auto_x))
                    visible = [ys for _, ys in series]
                auto_y = self._auto_range(visible)
                self._auto_lims[y_tag] = auto_y
                dpg.set_axis_limits(y_tag, *self._zoom.get(y_tag, auto_y))
            except Exception:
                continue

    def _update_status(self) -> None:
        r = self.runner.result
        state = self.runner.state
        status = []
        if r is not None:
            total_force = sum(r.finger_force_n.get(f, 0.0) for f in FINGERS)
            visual_contacts, visual_pen = self.sim.visual_contact_stats()
            status.append(f"反馈接触 {r.contact_count:3d} | 可视穿透 {visual_contacts:2d}点/"
                          f"{visual_pen * 1000.0:4.1f} mm | 总力 {total_force:5.2f} N | "
                          f"τmax {np.max(np.abs(r.torque_cmd_nm)):.2f} Nm")
            guard = "开" if (self.sim.collision_clamp and self.sim.objects_locked) else "关"
            status.append(f"保持: {'HOLD' if any(v > 0.2 for v in r.finger_force_n.values()) else 'OPEN'}"
                          f" | 物体: {'锁定' if self.sim.objects_locked else '可被手套推动'}"
                          f" | 防穿模: {guard}"
                          f" | 滑移: 不可用(无物体位姿传感器)")
            status.append(f"柔性形变 {r.max_object_disp_m * 1000.0:5.1f} mm | "
                          f"运动学 {r.kinematics_ms:4.1f} ms | 闭合 gap {r.kinematics_gap_m * 1000:4.2f} mm")
        mode = self._feedback_mode()
        mode_text = {"driver_impedance": "驱动器阻抗", "proxy_spring": "外置弹簧"}.get(mode, mode)
        imp = self.runner.last_impedance
        if mode == "driver_impedance" and imp:
            active = [c.k for c in imp if c.k > 0.0]
            if active:
                mode_text += f" {len(active)} 轴接触 kp≤{max(active):.0f}"
        status.append(f"循环 {self.runner.stats.loop_hz:5.1f} Hz（tick {self.runner.stats.tick_ms:4.1f} ms）"
                      f" | 反馈: {mode_text}")
        if self.renderer is not None:
            status.append(f"画面 {self.renderer.frame_hz:5.1f} Hz | 渲染延迟 "
                          f"{self.renderer.last_latency_s * 1e3:4.1f} ms | 界面 {self._ui_fps:5.1f} fps")
        try:
            lat = self.runner.telemetry.latency_stats()
            status.append(f"传感器时延 p50/p99/max = {lat['p50_ms']:.1f}/{lat['p99_ms']:.1f}/"
                          f"{lat['max_ms']:.1f} ms")
        except Exception:
            pass
        if self.runner.stats.error:
            status.append(f"错误: {self.runner.stats.error}")
        if self._history:
            clamped = [f"J{i + 1}" for i, c in enumerate(self._history[-1].encoder_clamped) if c]
            if clamped:
                status.append(f"注意: {', '.join(clamped)} 编码器超出标定范围"
                              "（仿真手被限位，曲线画的是未限位真值）")
        if self._last_error:
            status.append(self._last_error)
        if self.sim.palm_mode:
            yaw = math.degrees(self.sim.palm_yaw())
            busy = "（正在应用修改…）" if self._edit_busy else ""
            status.append(f"物体朝向 {yaw:+.0f}°（物体上右键拖动旋转，“放回手心”复位）{busy}")
        recorder = self.runner.recorder
        if recorder is not None:
            status.append(f"记录中: {recorder.path}  {recorder.ticks} 帧"
                          f"（{recorder.ticks / max(1.0, self.runner.stats.loop_hz or 100.0):.0f} s）"
                          + (f"  {recorder.error}" if recorder.error else ""))
        elif self._recording_path:
            status.append(f"记录: {self._recording_path}")
        dpg.set_value("status_text", "\n".join(status))
        if state is not None:
            dpg.set_value("source_text",
                          f"源: {state.source} | 数据有效性: "
                          f"{'全仿真' if state.source == 'fake' else '实测+仿真混合'}")
        else:
            dpg.set_value("source_text", "源: 未连接")

    # ------------------------------------------------------------------
    @staticmethod
    def _control_content_height() -> float:
        """Height the control column needs (0 until it has been rendered once)."""
        try:
            # Both markers scroll together, so their distance is the content
            # height regardless of the current scroll position.
            top = float(dpg.get_item_rect_min("control_start")[1])
            end = float(dpg.get_item_rect_max("control_end")[1])
        except Exception:
            return 0.0
        if end <= top:
            return 0.0
        return end - top + 24.0  # + child-window padding and border

    def _apply_layout(self) -> None:
        """Keep the panels usable when the viewport is maximized/fullscreen."""
        try:
            vw = int(dpg.get_viewport_client_width())
            vh = int(dpg.get_viewport_client_height())
        except Exception:
            return
        need = self._control_content_height()
        key = (vw, vh, int(round(need / 4.0)))
        if vw < 320 or vh < 320 or key == self._view_size:
            return
        self._view_size = key
        scale = max(1.0, min(1.8, vh / 1100.0))
        try:
            dpg.set_global_font_scale(scale)
        except Exception:
            pass
        margin = 12
        # Window padding + separator between the two panels.
        chrome = 36
        bottom_h = int(max(260, min(420, vh * 0.28)))
        if need > 0:
            # Give the bottom panel only what the control column can spare, so
            # the controls fit without a scrollbar (down to a 230 px floor).
            bottom_h = int(max(230, min(bottom_h, vh - chrome - need)))
        top_h = int(max(320, vh - bottom_h - chrome))
        control_w = int(min(1100, min(760, max(420, vw * 0.30)) * (0.75 + 0.25 * scale)))
        avail_w = max(320, vw - control_w - 5 * margin)
        avail_h = max(240, top_h - 60)
        img_w = min(avail_w, int(avail_h * 16 / 9), 3600)
        img_h = int(img_w * 9 / 16)
        view_w = img_w + 2 * margin
        dpg.configure_item("view_panel", width=view_w, height=top_h)
        dpg.configure_item("control_panel", width=control_w, height=top_h)
        dpg.configure_item("scene_image", width=img_w, height=img_h)
        spare = max(0, int((vw - view_w - control_w - 4 * margin) / 2))
        dpg.configure_item("top_spacer_left", width=spare)
        dpg.configure_item("top_spacer_right", width=spare)
        dpg.configure_item("bottom_panel", width=max(620, vw - 2 * margin),
                           height=bottom_h)
        plot_w = int(max(280, (vw - 4 * margin) / 3) - 14)
        plot_h = int(max(150, bottom_h - 58))
        for tag in ("force_plot", "torque_plot", "q_plot",
                    "current_plot", "disp_plot", "lat_plot"):
            dpg.configure_item(tag, width=plot_w, height=plot_h)
        for tag in ("cmp_torque_plot", "cmp_pos_plot", "cmp_cur_plot"):
            dpg.configure_item(tag, width=plot_w, height=max(130, plot_h - 28))
        dpg.configure_item("qd_plot", width=max(620, vw - 3 * margin), height=plot_h)

    # ------------------------------------------------------------------
    def update(self) -> None:
        t = time.perf_counter()
        if self._ui_last_t is not None and t > self._ui_last_t:
            fps = 1.0 / (t - self._ui_last_t)
            self._ui_fps = fps if self._ui_fps <= 0 else self._ui_fps + 0.05 * (fps - self._ui_fps)
        self._ui_last_t = t
        if self._pending_window_size is not None and not self._fullscreen:
            width, height = self._pending_window_size
            self._pending_window_size = None
            try:
                dpg.set_viewport_width(width)
                dpg.set_viewport_height(height)
                self._view_size = (0, 0)
            except Exception:
                pass
        lock = self.runner.lock
        self._apply_pending()   # queues edits for the worker; takes no lock
        self._apply_layout()
        with lock:
            self._mouse()
        now = time.monotonic()
        with lock:
            self._update_scene_texture()
        if now - self._last_plot >= 1.0 / PLOT_HZ:
            self._last_plot = now
            with lock:
                self._update_history()
            if self._plots_paused:
                self._apply_plot_limits(self._frozen_t)  # zoom on the frozen data
            else:
                self._update_plots()
            with lock:
                self._update_status()

    def close(self) -> None:
        with self._edit_cv:
            self._edit_stop = True
            self._edit_cv.notify()
        self._edit_thread.join(timeout=3.0)
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None



def run_ui(runner: Runner) -> int:
    if dpg is None:
        print(f"UI 不可用（DearPyGui 导入失败: {_DPG_IMPORT_ERROR}）；请使用 --headless。")
        return 2
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("没有可用的图形显示（DISPLAY/WAYLAND_DISPLAY 未设置）；请加 --headless，或在桌面会话中运行。")
        return 2
    try:
        runner.start()
    except Exception as error:
        print(f"启动失败: {error}")
        return 2
    try:
        dpg.create_context()
        font = None
        try:
            for path in FONT_CANDIDATES:
                if Path(path).is_file():
                    with dpg.font_registry():
                        with dpg.font(path, 18) as font:
                            # Glyph ranges are automatic in DearPyGui 2.x; the
                            # bundled subset must contain every UI character
                            # (tests/test_core.py checks this).
                            pass
                    break
        except Exception:
            font = None
        if font is not None:
            dpg.bind_font(font)
        dpg.create_viewport(title="Motorhand Glove Force-Feedback Simulation",
                            width=1400, height=980, x_pos=40, y_pos=20,
                            resizable=True, min_width=1000, min_height=700)
        dpg.setup_dearpygui()
        # vsync off: the UI loop runs at simulation.ui_hz instead of the
        # monitor refresh, so each shown frame carries a fresher sim state.
        try:
            dpg.set_viewport_vsync(bool(runner.config["simulation"].get("ui_vsync", False)))
        except Exception:
            pass
        dpg.show_viewport()
        app = App(runner)
        dpg.set_primary_window("main", True)
    except Exception as error:
        print(f"UI 启动失败: {error}")
        print("无图形显示时请加 --headless；DearPyGui 需要可用的 OpenGL 显示。")
        try:
            dpg.destroy_context()
        except Exception:
            pass
        runner.close()
        return 2
    try:
        # The control loop (glove read -> kinematics -> contacts -> torque)
        # runs on its own fixed-rate thread; the UI only renders.
        runner.start_thread()
        ui_hz = max(10.0, float(runner.config["simulation"].get("ui_hz", 60.0)))
        while dpg.is_dearpygui_running():
            frame_start = time.perf_counter()
            try:
                app.update()
                dpg.render_dearpygui_frame()
            except Exception as error:
                print(f"UI 渲染失败: {error}")
                runner.stats.error = f"UI 渲染失败: {error}"
                break
            # Cap at simulation.ui_hz (the display's vsync may cap lower), but
            # never sleep when a frame already took longer than that.
            remaining = 1.0 / ui_hz - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        app.close()
        runner.close()
        dpg.destroy_context()
    return 0
