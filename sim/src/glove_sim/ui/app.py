from __future__ import annotations

import math
import os
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
PLOT_HISTORY = 1400
PLOT_HZ = 10.0
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
        self._view_size: tuple[int, int] = (0, 0)
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
        self._fullscreen = False
        self._windowed_client = (1400, 980)
        self._pending_window_size: tuple[int, int] | None = None
        self.selected = self.runner.specs[0].name if self.runner.specs else ""
        self.pending: dict[str, dict[str, float]] = {}
        self._last_rebuild = 0.0
        self._last_status = 0.0
        self._syncing = False
        self._recording_path = ""
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
                    "mvPlotCol_Line": C_ACCENT,
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

    def _build_ui(self) -> None:
        assert dpg is not None
        self._install_theme()
        with dpg.texture_registry(show=False):
            blank = np.zeros((RENDER_H, RENDER_W, 4), dtype=np.float32)
            blank[:, :, 3] = 1.0
            # Reuse one RGBA buffer for every upload instead of allocating and
            # filling ~15 MB per frame; at 1280x720 that allocation was a
            # visible share of the UI frame time.
            self._texture_buf = blank
            self._texture = dpg.add_dynamic_texture(
                RENDER_W, RENDER_H, blank.ravel().tolist(), tag="scene_texture",
            )
        with dpg.window(tag="main", no_title_bar=True, no_resize=False):
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=0, tag="top_spacer_left")
                with dpg.child_window(tag="view_panel", width=800, height=620, border=True):
                    with dpg.group(horizontal=True):
                        dpg.add_text("MOTORHAND", color=C_ACCENT)
                        dpg.add_text("手套力反馈仿真", color=C_TEXT)
                        dpg.add_text("  |  左键拖动物体  ·  Shift+左键旋转  ·  右键旋转视角  ·  "
                                     "滚轮缩放  ·  F11 全屏", wrap=650, color=C_MUTED)
                    dpg.add_spacer(height=4)
                    dpg.add_image("scene_texture", width=IMAGE_W, height=IMAGE_H, tag="scene_image")
                with dpg.child_window(tag="control_panel", width=430, height=620, border=True):
                    dpg.add_text("数据源与安全", color=C_GOLD)
                    dpg.add_text("", tag="source_text", color=C_ACCENT)
                    dpg.add_text("", tag="status_text", color=C_TEXT)
                    dpg.add_separator()
                    dpg.add_text("物体参数  ·  实时重建 10 Hz", color=C_GOLD)
                    names = [s.name for s in self.runner.specs]
                    dpg.add_combo(names, default_value=self.selected, label="物体",
                                  tag="object_combo", callback=self._on_select)
                    dpg.add_slider_float(label="大小 mm", default_value=30.0,
                                         min_value=float(self.config["objects"]["size_range_m"][0]) * 1000,
                                         max_value=float(self.config["objects"]["size_range_m"][1]) * 1000,
                                         tag="size_slider", callback=self._on_size)
                    dpg.add_slider_float(label="硬度 0..1", default_value=0.25,
                                         min_value=float(self.config["objects"]["stiffness_range"][0]),
                                         max_value=float(self.config["objects"]["stiffness_range"][1]),
                                         tag="stiff_slider", callback=self._on_stiff)
                    dpg.add_text("硬度越高，接触越硬、形变越小；只有 ball_soft 是柔性晶格。",
                                 wrap=400, color=C_MUTED)
                    dpg.add_slider_float(label="反馈增益 Nm/N", default_value=float(
                        self.config["glove"].get("feedback_gain", 10.0)),
                        min_value=0.0, max_value=50.0, tag="gain_slider",
                        callback=self._on_gain)
                    dpg.add_text("增益缩放名义力矩；每轴上限与急停仍生效。",
                                 wrap=400, color=C_MUTED)
                    dpg.add_separator()
                    dpg.add_text("控制", color=C_GOLD)
                    dpg.add_checkbox(label="启用力反馈（真机需设备在线）",
                                     default_value=bool(self.runner.feedback),
                                     tag="feedback_check", callback=self._on_feedback)
                    dpg.add_checkbox(label="锁定物体（关闭后手套可推动/带动）",
                                     default_value=bool(self.sim.objects_locked),
                                     tag="lock_check", callback=self._on_lock)
                    dpg.add_checkbox(label="暂停假手套演示（方便摆放物体）",
                                     default_value=False,
                                     tag="demo_check", callback=self._on_demo)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="开始记录", tag="record_button", width=180,
                                       callback=self._on_record)
                        dpg.add_button(label="急停 / 清零", tag="stop_button", width=180,
                                       callback=self._on_stop)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="默认视角", width=180, callback=self._reset_camera)
                        dpg.add_button(label="全屏 F11", width=180, callback=self._toggle_fullscreen)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="聚焦", width=115, callback=self._on_focus)
                        dpg.add_button(label="放到指尖", width=115, callback=self._on_place)
                        dpg.add_button(label="重置物体", width=115,
                                       callback=self._on_reset_objects)
                    with dpg.collapsing_header(label="来源与限制", default_open=False):
                        dpg.add_text(
                            "接触力 = MuJoCo 仿真接触力；力矩 = 未标定名义指令。\n"
                            "锁定物体时可见手型按手指限位，接触力按未限位目标计算。\n"
                            "滑移不可用（无物体位姿传感器）；穿透量不等于物体应变。",
                            wrap=400, color=C_MUTED,
                        )
                    dpg.bind_item_theme("record_button", self._primary_theme)
                    dpg.bind_item_theme("stop_button", self._danger_theme)
                dpg.add_spacer(width=0, tag="top_spacer_right")
            dpg.add_separator()
            with dpg.child_window(tag="bottom_panel", height=300, width=1400, border=True):
                with dpg.tab_bar():
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
                            with dpg.plot(label="电流 (A, 真机实测/假手套代理)", height=250, width=450,
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
                        with dpg.plot(label="仿真关节速度 (rad/s)", height=250, width=1380,
                                      tag="qd_plot"):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="t (s)", tag="qd_x")
                            with dpg.plot_axis(dpg.mvYAxis, label="rad/s", tag="qd_y"):
                                for i in range(12):
                                    dpg.add_line_series([], [], label=f"J{i+1}",
                                                        tag=f"qd_line_{i+1}")
                        dpg.add_text("保持/滑移状态见右侧状态栏；滑移在物体锁定模式下不可用。",
                                     color=C_MUTED)
        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=self._on_wheel)
            key_f11 = getattr(dpg, "mvKey_F11", None)
            if key_f11 is not None:
                dpg.add_key_press_handler(key=key_f11,
                                          callback=lambda *args: self._toggle_fullscreen())

    # ------------------------------------------------------------------
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

    def _pick(self, image_x: float, image_y: float) -> str | None:
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
    def _on_select(self, sender, app_data) -> None:
        self.selected = str(app_data)
        for spec in self.runner.specs:
            if spec.name == self.selected:
                self._syncing = True
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

    def _on_gain(self, sender, app_data) -> None:
        if self._syncing:
            return
        self.config["glove"]["feedback_gain"] = float(app_data)

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
        for other in self.runner.specs:
            if other.name == spec.name:
                continue
            other_obj = next((o for o in self.sim.object_runtime if o.name == other.name), None)
            if other_obj is None:
                continue
            clearance = 0.5 * (spec.size_m + other.size_m) + 0.008
            while float(np.linalg.norm(pos - other_obj.target_position)) < clearance:
                pos[1] += clearance
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

    def _on_record(self, sender, app_data) -> None:
        if not self.runner.telemetry.recording:
            directory = self.config.get("telemetry", {}).get("default_dir", "data")
            try:
                path = self.runner.start_recording(directory)
                self._recording_path = path
                dpg.configure_item("record_button", label="停止记录")
            except Exception as error:
                self._last_error = f"记录失败: {error}"
        else:
            self.runner.stop_recording()
            dpg.configure_item("record_button", label="开始记录")

    def _on_wheel(self, sender, app_data) -> None:
        self._wheel += float(app_data)

    def _reset_camera(self) -> None:
        self.cam.lookat[:] = self._cam_home[0]
        self.cam.distance = self._cam_home[1]
        self.cam.azimuth = self._cam_home[2]
        self.cam.elevation = self._cam_home[3]

    # ------------------------------------------------------------------
    def _apply_pending(self) -> None:
        if not self.pending:
            return
        now = time.monotonic()
        hz = float(self.config.get("objects", {}).get("rebuild_hz", 10.0))
        if now - self._last_rebuild < 1.0 / max(1.0, hz):
            return
        self._last_rebuild = now
        for name, changes in list(self.pending.items()):
            try:
                self.sim.set_object_shape(
                    name,
                    size_m=changes.get("size_m"),
                    stiffness=changes.get("stiffness"),
                )
            except Exception as error:
                self._last_error = f"重建 {name} 失败: {error}"
        self.pending.clear()
        if self.sim.model is not self._model_ref:
            self._rebuild_renderer()

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

        # Right button orbits the camera.  It never picks or moves objects.
        if right:
            if not self._right_down:
                self._right_down = True
                self._right_last = (mx, my)
                # Cancel a left-button object drag so both gestures cannot
                # fight over the same mouse delta.
                if self._mouse_down:
                    if self._drag_name is not None:
                        self.sim.end_drag()
                    self._mouse_down = False
                    self._drag_name = None
                    self._drag_mode = None
                    self._last_mouse = None
            elif self._right_last is not None:
                dx = mx - self._right_last[0]
                dy = my - self._right_last[1]
                self.cam.azimuth -= dx * 0.35
                self.cam.elevation = max(-89.0, min(20.0, self.cam.elevation + dy * 0.25))
                self._right_last = (mx, my)
        else:
            self._right_down = False
            self._right_last = None

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
        try:
            img = self.renderer.poll()
            if img is None:
                self.renderer.request(self.sim.xml, self.sim.data.qpos, self.sim.data.qvel,
                                      self.cam, RENDER_W, RENDER_H,
                                      highlight=self._highlight_payload())
                return
        except Exception as error:
            self._last_error = f"图像渲染失败: {error}"
            return
        # MuJoCo's Renderer already returns a top-down image (it flips the
        # OpenGL readback internally).  Do not flip again: the old extra
        # `img[::-1]` displayed the scene upside down and broke picking.
        rgba = self._texture_buf
        np.divide(img, 255.0, out=rgba[:, :, :3], casting="unsafe")
        flat = rgba.reshape(-1)
        try:
            # Newer DPG builds accept a contiguous float32 buffer directly;
            # this avoids a ~30 ms Python list conversion at 960x540.
            dpg.set_value("scene_texture", flat)
        except (TypeError, ValueError):
            dpg.set_value("scene_texture", flat.tolist())

    def _update_history(self) -> None:
        r = self.runner.result
        if r is None:
            return
        t = (r.timestamp_ns / 1e9) - self._t0
        forces = [float(r.finger_force_n.get(f, 0.0)) for f in FINGERS + ("palm",)]
        torques = [float(v) for v in r.torque_cmd_nm]
        q = [float(v) for v in r.sim_q]
        qd = [float(v) for v in r.sim_dq]
        pens = [float(r.finger_penetration_m.get(f, 0.0)) for f in FINGERS]
        age = 0.0
        try:
            age = float(self.runner.telemetry.ring.latency[-1])
        except Exception:
            pass
        currents = [float(self.runner.state.current_a.get(i + 1, float("nan")))
                    if self.runner.state is not None else float("nan") for i in range(12)]
        self._history.append((t, forces, torques, q, pens, float(r.max_object_disp_m),
                              r.contact_count, age, currents, qd))

    def _update_plots(self) -> None:
        if not self._history:
            return
        hist = list(self._history)
        t = [row[0] for row in hist]
        for i, finger in enumerate(FINGERS + ("palm",)):
            ys = [row[1][i] for row in hist]
            dpg.set_value(f"force_line_{finger}", [t, ys])
        finger_motor = {f: [i for i, mf in enumerate(MOTOR_FINGER) if mf == f] for f in FINGERS}
        for finger in FINGERS:
            idx = finger_motor[finger]
            ys = [max((abs(row[2][i]) for i in idx), default=0.0) for row in hist]
            dpg.set_value(f"torque_line_{finger}", [t, ys])
            cur = [max((abs(row[8][i]) for i in idx), default=0.0) for row in hist]
            dpg.set_value(f"current_line_{finger}", [t, cur])
        for i in range(12):
            ys = [row[3][i] for row in hist]
            dpg.set_value(f"q_line_{i+1}", [t, ys])
            qds = [row[9][i] for row in hist]
            dpg.set_value(f"qd_line_{i+1}", [t, qds])
        pens = [max(row[4], default=0.0) for row in hist]
        total_force = [sum(row[1][:5]) for row in hist]
        dpg.set_value("disp_line", [pens, total_force])
        deform = [row[5] for row in hist]
        dpg.set_value("deform_line", [deform, total_force])
        lat = [row[7] for row in hist]
        dpg.set_value("lat_line", [t, lat])
        if len(t) > 1:
            dpg.set_axis_limits("force_x", max(0.0, t[-1] - 10.0), t[-1] + 0.2)
            dpg.set_axis_limits("torque_x", max(0.0, t[-1] - 10.0), t[-1] + 0.2)
            dpg.set_axis_limits("q_x", max(0.0, t[-1] - 10.0), t[-1] + 0.2)
            dpg.set_axis_limits("lat_x", max(0.0, t[-1] - 10.0), t[-1] + 0.2)
            dpg.set_axis_limits("current_x", max(0.0, t[-1] - 10.0), t[-1] + 0.2)
            dpg.set_axis_limits("qd_x", max(0.0, t[-1] - 10.0), t[-1] + 0.2)
        try:
            dpg.fit_axis_data("force_y")
            dpg.fit_axis_data("torque_y")
            dpg.fit_axis_data("disp_x")
            dpg.fit_axis_data("disp_y")
            dpg.fit_axis_data("lat_y")
            dpg.fit_axis_data("current_y")
            dpg.fit_axis_data("qd_y")
        except Exception:
            pass

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
        status.append(f"循环 {self.runner.stats.loop_hz:5.1f} Hz")
        try:
            lat = self.runner.telemetry.latency_stats()
            status.append(f"传感器时延 p50/p99/max = {lat['p50_ms']:.1f}/{lat['p99_ms']:.1f}/"
                          f"{lat['max_ms']:.1f} ms")
        except Exception:
            pass
        if self.runner.stats.error:
            status.append(f"错误: {self.runner.stats.error}")
        if self._last_error:
            status.append(self._last_error)
        if self._recording_path:
            status.append(f"记录: {self._recording_path}")
        dpg.set_value("status_text", "\n".join(status))
        if state is not None:
            dpg.set_value("source_text",
                          f"源: {state.source} | 数据有效性: "
                          f"{'全仿真' if state.source == 'fake' else '实测+仿真混合'}")
        else:
            dpg.set_value("source_text", "源: 未连接")

    # ------------------------------------------------------------------
    def _apply_layout(self) -> None:
        """Keep the panels usable when the viewport is maximized/fullscreen."""
        try:
            vw = int(dpg.get_viewport_client_width())
            vh = int(dpg.get_viewport_client_height())
        except Exception:
            return
        if vw < 320 or vh < 320 or (vw, vh) == self._view_size:
            return
        self._view_size = (vw, vh)
        scale = max(1.0, min(1.8, vh / 1100.0))
        try:
            dpg.set_global_font_scale(scale)
        except Exception:
            pass
        margin = 12
        bottom_h = int(max(230, min(380, vh * 0.25)))
        top_h = int(max(320, vh - bottom_h - 70))
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
        plot_h = int(max(180, bottom_h - 58))
        for tag in ("force_plot", "torque_plot", "q_plot",
                    "current_plot", "disp_plot", "lat_plot"):
            dpg.configure_item(tag, width=plot_w, height=plot_h)
        dpg.configure_item("qd_plot", width=max(620, vw - 3 * margin), height=plot_h)

    # ------------------------------------------------------------------
    def update(self) -> None:
        if self._pending_window_size is not None and not self._fullscreen:
            width, height = self._pending_window_size
            self._pending_window_size = None
            try:
                dpg.set_viewport_width(width)
                dpg.set_viewport_height(height)
                self._view_size = (0, 0)
            except Exception:
                pass
        self._apply_pending()
        self._apply_layout()
        self._mouse()
        now = time.monotonic()
        if now - self._last_tex >= 1.0 / 30.0:
            self._last_tex = now
            self._update_scene_texture()
        if now - self._last_plot >= 1.0 / PLOT_HZ:
            self._last_plot = now
            self._update_history()
            self._update_plots()
            self._update_status()

    def close(self) -> None:
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
        while dpg.is_dearpygui_running():
            frame_start = time.perf_counter()
            try:
                runner.tick(max_steps=30)
            except Exception as error:
                runner.stats.error = str(error)
            try:
                app.update()
                dpg.render_dearpygui_frame()
            except Exception as error:
                print(f"UI 渲染失败: {error}")
                runner.stats.error = f"UI 渲染失败: {error}"
                break
            # Cap at ~60 UI frames/s, but never sleep when a frame already
            # took longer than that (at 1280x720 it usually does).
            remaining = 1.0 / 60.0 - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        app.close()
        runner.close()
        dpg.destroy_context()
    return 0
