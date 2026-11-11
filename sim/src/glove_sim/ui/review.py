"""Offline review of a recorded session (glove_sim.recording).

One time cursor drives everything: the linked curves, the per-joint value
table and the 3D scene rebuilt from the recorded MuJoCo state at that tick
(visible simulated hand + translucent unclamped real-hand pose + the
feedback contacts with force arrows).

Controls: drag the yellow cursor line or click a plot (without dragging) to
move the cursor; space = play/pause; left/right = one tick (shift: 10);
up/down = previous/next joint; drag the image to orbit, wheel to zoom.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import mujoco
import numpy as np

try:
    import dearpygui.dearpygui as dpg
except Exception as exc:  # pragma: no cover
    dpg = None
    _DPG_IMPORT_ERROR = exc
else:
    _DPG_IMPORT_ERROR = None

from ..mapping import MOTOR_FINGER
from ..recording import CONTACT_FINGERS, Session, load_session
from .app import (C_ACCENT, C_BG, C_BORDER, C_CARD, C_GOLD, C_GREEN, C_MUTED, C_PANEL, C_RED,
                  C_TEXT, COLORS, FINGER_CN, FONT_CANDIDATES)
from .render_process import RenderProcess

RENDER_W, RENDER_H = 1280, 720
MAX_POINTS = 2400
DEG = 180.0 / math.pi
SPEEDS = ("0.1x", "0.25x", "0.5x", "1x", "2x", "4x")
GROUP_CN = dict(FINGER_CN, palm="手掌")
GHOST_RGBA = (1.0, 0.35, 0.30, 0.35)
CONTACT_RGBA = (1.0, 0.25, 0.20, 1.0)


def envelope(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-bucket min/max reduction so every peak survives decimation."""
    n = x.size
    if n <= max_points:
        return x, y
    buckets = max(1, max_points // 2)
    edges = np.linspace(0, n, buckets + 1).astype(int)
    starts = edges[:-1]
    finite = np.isfinite(y)
    lo = np.minimum.reduceat(np.where(finite, y, np.inf), starts)
    hi = np.maximum.reduceat(np.where(finite, y, -np.inf), starts)
    lo[~np.isfinite(lo)] = np.nan
    hi[~np.isfinite(hi)] = np.nan
    ends = np.maximum(edges[1:] - 1, starts)
    out_x = np.empty(2 * buckets)
    out_y = np.empty(2 * buckets)
    out_x[0::2], out_x[1::2] = x[starts], x[ends]
    out_y[0::2], out_y[1::2] = lo, hi
    return out_x, out_y


class ReviewApp:
    def __init__(self, session: Session) -> None:
        self.s = session
        self.n = len(session)
        self.t = session.t
        self.i = 0
        self.joint = 0
        self.playing = False
        self.speed = 1.0
        self._play_t = 0.0
        self._last_frame = time.perf_counter()
        self._dirty_scene = True
        self._show_ghost = True
        self._show_contacts = True
        self._show_collision = False
        self._xlim: tuple[float, float] | None = None
        self._fit_frames = 2
        self._mouse_press: tuple[float, float] | None = None
        self._orbit_last: tuple[float, float] | None = None
        self._view_size = (0, 0)
        self._xml_cache: dict[int, str] = {}
        self._msg = ""
        self.joint_names = list(session.meta.get("joint_names", [f"J{k + 1}" for k in range(12)]))
        self.motor_ids = list(session.meta.get("motor_ids", range(1, 13)))
        scene = session.meta.get("config", {}).get("scene", {})
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self._cam_home = (np.asarray(scene.get("camera_lookat", [0.01, -0.015, 0.135]), dtype=float),
                          float(scene.get("camera_distance", 0.40)),
                          float(scene.get("camera_azimuth", 110.0)),
                          float(scene.get("camera_elevation", -12.0)))
        self._reset_camera()
        self.renderer = RenderProcess()
        # plot tag -> list of (series tag, key, transform)
        self._series: list[tuple[str, np.ndarray]] = []
        self._build()
        self._set_joint(0)
        self._set_cursor(0)

    # ------------------------------------------------------------------
    def joint_label(self, k: int) -> str:
        finger = GROUP_CN.get(MOTOR_FINGER[k], MOTOR_FINGER[k]) if k < len(MOTOR_FINGER) else ""
        return f"J{self.motor_ids[k]} {finger} {self.joint_names[k]}"

    def _reset_camera(self, *_args) -> None:
        self.cam.lookat[:] = self._cam_home[0]
        self.cam.distance = self._cam_home[1]
        self.cam.azimuth = self._cam_home[2]
        self.cam.elevation = self._cam_home[3]
        self._dirty_scene = True

    # ------------------------------------------------------------------
    def _theme(self) -> None:
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, C_BG)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, C_PANEL)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, C_CARD)
                dpg.add_theme_color(dpg.mvThemeCol_Border, C_BORDER)
                dpg.add_theme_color(dpg.mvThemeCol_Text, C_TEXT)
                dpg.add_theme_color(dpg.mvThemeCol_Button, C_CARD)
                dpg.add_theme_color(dpg.mvThemeCol_Header, (35, 60, 85, 255))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 4, 1)
        dpg.bind_theme(theme)

    def _line_theme(self, color, weight: float = 1.6):
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, weight, category=dpg.mvThemeCat_Plots)
        return theme

    def _add_line(self, axis: str, tag: str, label: str, color, data_fn, weight: float = 1.6) -> None:
        dpg.add_line_series([], [], label=label, parent=axis, tag=tag)
        dpg.bind_item_theme(tag, self._line_theme(color, weight))
        self._series.append((tag, data_fn))

    def _build(self) -> None:
        s = self.s
        dpg.add_texture_registry(tag="review_textures")
        self._blank = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.float32)
        dpg.add_raw_texture(RENDER_W, RENDER_H, self._blank, format=dpg.mvFormat_Float_rgb,
                            tag="review_tex", parent="review_textures")
        self._theme()
        duration = float(self.t[-1] - self.t[0]) if self.n else 0.0
        objects = " / ".join(
            f"{o['label']} {o['size_m'] * 1000:.0f}mm 硬度 {o['stiffness']:.2f}"
            for seg in s.meta.get("segments", []) for o in seg.get("objects", [])[:1])
        with dpg.window(tag="review_main", no_scrollbar=True):
            dpg.add_text(f"{s.path}   源: {s.meta.get('source') or '?'}   {self.n} 帧 / {duration:.1f} s"
                         f"   物体: {objects}", color=C_MUTED)
            with dpg.group(horizontal=True):
                dpg.add_button(label="|<", width=36, callback=lambda: self._set_cursor(0))
                dpg.add_button(label="<", width=36, callback=lambda: self._step(-1))
                dpg.add_button(label="播放", tag="play_button", width=70, callback=self._toggle_play)
                dpg.add_button(label=">", width=36, callback=lambda: self._step(1))
                dpg.add_button(label=">|", width=36, callback=lambda: self._set_cursor(self.n - 1))
                dpg.add_combo(SPEEDS, default_value="1x", width=80, callback=self._on_speed)
                dpg.add_slider_double(tag="time_slider", min_value=float(self.t[0]),
                                      max_value=float(self.t[-1]), default_value=float(self.t[0]),
                                      width=-430, format="%.3f s", callback=self._on_slider)
                dpg.add_button(label="看全程", width=80, callback=self._fit_all)
                dpg.add_button(label="导出 CSV", width=90, callback=self._export)
                dpg.add_button(label="重新加载", width=90, callback=self._reload)
                dpg.add_button(label="默认视角", width=90, callback=self._reset_camera)
            with dpg.group(horizontal=True):
                with dpg.child_window(tag="left_panel", width=700, border=False):
                    dpg.add_image("review_tex", tag="review_image", width=684, height=385)
                    with dpg.group(horizontal=True):
                        dpg.add_checkbox(label="真手姿态（红色半透明）", default_value=True,
                                         callback=self._toggle("_show_ghost"))
                        dpg.add_checkbox(label="反馈接触点 / 力", default_value=True,
                                         callback=self._toggle("_show_contacts"))
                        dpg.add_checkbox(label="碰撞体", default_value=False,
                                         callback=self._toggle("_show_collision"))
                    dpg.add_text("", tag="cursor_info", wrap=690)
                    self._build_table()
                    dpg.add_text("", tag="contact_info", wrap=690, color=C_MUTED)
                    dpg.add_text("", tag="event_info", wrap=690, color=C_MUTED)
                with dpg.child_window(tag="right_panel", border=False):
                    with dpg.group(horizontal=True):
                        dpg.add_text("曲线关节:")
                        dpg.add_combo([self.joint_label(k) for k in range(12)], tag="joint_combo",
                                      default_value=self.joint_label(0), width=320,
                                      callback=self._on_joint_combo)
                        dpg.add_text("  点曲线=移游标，拖动=平移，滚轮=缩放，双击=自适应；"
                                     "空格播放，←/→ 单帧，↑/↓ 换关节", color=C_MUTED)
                    self._build_plots()
            dpg.add_text("", tag="review_status", color=C_GOLD)
        with dpg.handler_registry():
            dpg.add_key_press_handler(callback=self._on_key)
            dpg.add_mouse_wheel_handler(callback=self._on_wheel)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=self._on_click)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=self._on_release)

    def _toggle(self, attr: str):
        def callback(sender, value):
            setattr(self, attr, bool(value))
            self._dirty_scene = True
        return callback

    def _build_table(self) -> None:
        cols = ("关节", "实测角°", "仿真角°", "差°", "接触面°", "速度°/s", "弹簧τ", "驱动τ",
                "下发τ", "电流×Kt", "电流A", "kp", "kd", "温度", "状态")
        with dpg.table(tag="value_table", header_row=True, borders_innerV=True, borders_outerH=True,
                       row_background=True, policy=dpg.mvTable_SizingFixedFit,
                       scrollX=True, height=330, width=690):
            for c in cols:
                dpg.add_table_column(label=c)
            for k in range(12):
                with dpg.table_row():
                    dpg.add_selectable(label=f"J{self.motor_ids[k]} {GROUP_CN.get(MOTOR_FINGER[k], '')}",
                                       tag=f"row_sel_{k}", span_columns=True,
                                       callback=lambda sender, v, k=k: self._set_joint(k))
                    for c in range(1, len(cols)):
                        dpg.add_text("", tag=f"cell_{k}_{c}")

    def _build_plots(self) -> None:
        s = self.s
        j = lambda key, scale=1.0: (lambda: s[key][:, self.joint] * scale)  # noqa: E731
        with dpg.subplots(6, 1, tag="plots", link_all_x=True, no_menus=False, width=-1, height=-1,
                          row_ratios=[1.3, 1.3, 1.0, 0.9, 1.0, 0.8]):
            specs = [
                ("p_q", "关节角 (°)", "y_q", [
                    ("s_real_q", "实测编码器（未限位）", COLORS[1], j("real_q", DEG), 2.0),
                    ("s_sim_q", "仿真手（可见）", COLORS[2], j("sim_q", DEG), 1.6),
                    ("s_surface", "接触面（代理弹簧零点）", C_GOLD, j("surface_q", DEG), 1.4),
                    ("s_imp_target", "阻抗目标", COLORS[4], j("imp_target", DEG), 1.0),
                ]),
                ("p_tau", "力矩 (Nm，模型坐标，闭合为+)", "y_tau", [
                    ("s_spring", "虚拟弹簧(计算)", COLORS[3], j("torque_spring_nm"), 1.4),
                    ("s_driver", "驱动器输出(推算)", COLORS[0], j("torque_driver_nm"), 1.8),
                    ("s_sent", "下发指令(计算)", COLORS[5], j("torque_sent_nm"), 1.2),
                    ("s_meas", "电流×Kt(实测估计)", COLORS[6], j("torque_measured_nm"), 1.6),
                    ("s_ff", "前馈(摩擦补偿)", COLORS[7], j("imp_tau_ff"), 1.0),
                ]),
                ("p_dq", "速度 (°/s)", "y_dq", [
                    ("s_real_dq", "实测(编码器估计)", COLORS[1], j("real_dq", DEG), 1.6),
                    ("s_sim_dq", "仿真手", COLORS[2], j("sim_dq", DEG), 1.4),
                ]),
                ("p_imp", "阻抗 kp (Nm/rad) / kd×100", "y_imp", [
                    ("s_kp", "kp", COLORS[0], j("imp_k"), 1.6),
                    ("s_kd", "kd×100", COLORS[4], j("imp_b", 100.0), 1.4),
                ]),
                ("p_force", "仿真接触力 (N)", "y_force", [
                    (f"s_f_{g}", GROUP_CN[g], COLORS[n % len(COLORS)],
                     (lambda n=n: s["finger_force_n"][:, n]), 1.5)
                    for n, g in enumerate(CONTACT_FINGERS)
                ]),
                ("p_timing", "时序 (ms)", "y_timing", [
                    ("s_tick", "控制周期耗时", COLORS[3], lambda: s["tick_ms"], 1.2),
                    ("s_age", "传感器数据龄", COLORS[1], lambda: s["sensor_age_ms"], 1.2),
                    ("s_period", "控制周期(1000/Hz)", COLORS[7],
                     lambda: np.where(s["loop_hz"] > 1.0, 1000.0 / np.maximum(1.0, s["loop_hz"]),
                                      np.nan), 1.0),
                ]),
            ]
            for plot, title, yaxis, series in specs:
                with dpg.plot(label=title, tag=plot, anti_aliased=True):
                    dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthWest, outside=False)
                    dpg.add_plot_axis(dpg.mvXAxis, tag=f"x_{plot}", no_label=True)
                    dpg.add_plot_axis(dpg.mvYAxis, tag=yaxis, auto_fit=True)
                    for tag, label, color, fn, weight in series:
                        self._add_line(yaxis, tag, label, color, fn, weight)
                    if plot == "p_force":
                        marks = sorted({t for t, _e in self.s.event_times()}
                                       | {(seg["t_start_ns"] - self.s.meta.get("t0_ns", 0)) / 1e9
                                          for seg in self.s.meta.get("segments", [])[1:]})
                        if marks:
                            dpg.add_inf_line_series(marks, label="事件/换物体", parent=yaxis,
                                                    tag="s_events")
                            dpg.bind_item_theme("s_events", self._line_theme(C_MUTED, 1.0))
                    dpg.add_drag_line(tag=f"cur_{plot}", color=(255, 210, 60, 255), thickness=1.5,
                                      default_value=float(self.t[0]), callback=self._on_drag_line)
        self._plot_tags = [spec[0] for spec in specs]
        # Feedback-on spans would also be useful; mark them on the torque plot.
        fb = self.s["feedback_on"].astype(bool)
        if fb.any():
            edges = np.flatnonzero(np.diff(np.r_[0, fb.astype(int), 0]))
            xs = [float(self.t[min(e, self.n - 1)]) for e in edges]
            dpg.add_inf_line_series(xs, label="反馈开/关", parent="y_tau", tag="s_fb")
            dpg.bind_item_theme("s_fb", self._line_theme(C_GREEN, 1.0))

    # ------------------------------------------------------------------
    def _refresh_series(self, force: bool = False) -> None:
        lim = tuple(dpg.get_axis_limits("x_p_q"))
        if not force and self._xlim is not None and np.allclose(lim, self._xlim, rtol=0, atol=1e-6):
            return
        self._xlim = lim
        lo, hi = lim
        if hi <= lo:
            lo, hi = float(self.t[0]), float(self.t[-1])
        a = max(0, int(np.searchsorted(self.t, lo)) - 1)
        b = min(self.n, int(np.searchsorted(self.t, hi)) + 2)
        x = self.t[a:b]
        for tag, fn in self._series:
            y = np.asarray(fn(), dtype=float)[a:b]
            xs, ys = envelope(x, y, MAX_POINTS)
            dpg.set_value(tag, [xs.tolist(), ys.tolist()])

    def _fit_all(self, *_args) -> None:
        for plot in self._plot_tags:
            dpg.set_axis_limits(f"x_{plot}", float(self.t[0]), float(self.t[-1]))
        self._fit_frames = 2

    # ------------------------------------------------------------------
    def _set_joint(self, k: int) -> None:
        self.joint = int(k) % 12
        dpg.set_value("joint_combo", self.joint_label(self.joint))
        for r in range(12):
            dpg.set_value(f"row_sel_{r}", r == self.joint)
        self._refresh_series(force=True)

    def _on_joint_combo(self, sender, value) -> None:
        for k in range(12):
            if self.joint_label(k) == value:
                self._set_joint(k)
                return

    def _set_cursor(self, i: int, *, from_play: bool = False) -> None:
        self.i = int(max(0, min(self.n - 1, i)))
        t = float(self.t[self.i])
        if not from_play:
            self._play_t = t
        for plot in self._plot_tags:
            dpg.set_value(f"cur_{plot}", t)
        dpg.set_value("time_slider", t)
        self._update_info()
        self._dirty_scene = True

    def _step(self, n: int) -> None:
        self.playing = False
        dpg.configure_item("play_button", label="播放")
        self._set_cursor(self.i + n)

    def _toggle_play(self, *_args) -> None:
        self.playing = not self.playing
        if self.playing and self.i >= self.n - 1:
            self._set_cursor(0)
        self._play_t = float(self.t[self.i])
        dpg.configure_item("play_button", label="暂停" if self.playing else "播放")

    def _on_speed(self, sender, value) -> None:
        self.speed = float(str(value).rstrip("x"))

    def _on_slider(self, sender, value) -> None:
        self._set_cursor(self.s.index_at(float(value)))

    def _on_drag_line(self, sender, value) -> None:
        self._set_cursor(self.s.index_at(float(dpg.get_value(sender))))

    def _hovered_plot(self) -> str | None:
        for plot in self._plot_tags:
            if dpg.is_item_hovered(plot):
                return plot
        return None

    def _on_click(self, sender, value) -> None:
        self._mouse_press = tuple(dpg.get_mouse_pos(local=False))
        self._orbit_last = self._mouse_press if dpg.is_item_hovered("review_image") else None

    def _on_release(self, sender, value) -> None:
        press, self._mouse_press = self._mouse_press, None
        self._orbit_last = None
        if press is None:
            return
        pos = dpg.get_mouse_pos(local=False)
        if math.hypot(pos[0] - press[0], pos[1] - press[1]) > 4:
            return  # a drag (pan / cursor line), not a click
        plot = self._hovered_plot()
        if plot is not None:
            x = float(dpg.get_plot_mouse_pos()[0])
            self._set_cursor(self.s.index_at(x))

    def _on_wheel(self, sender, value) -> None:
        if dpg.is_item_hovered("review_image"):
            self.cam.distance = float(np.clip(self.cam.distance * (0.9 ** float(value)), 0.05, 3.0))
            self._dirty_scene = True

    def _on_key(self, sender, key) -> None:
        if dpg.is_item_active("time_slider"):
            return
        shift = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
        step = 10 if shift else 1
        if key == dpg.mvKey_Spacebar:
            self._toggle_play()
        elif key == dpg.mvKey_Left:
            self._step(-step)
        elif key == dpg.mvKey_Right:
            self._step(step)
        elif key == dpg.mvKey_Up:
            self._set_joint(self.joint - 1)
        elif key == dpg.mvKey_Down:
            self._set_joint(self.joint + 1)

    def _export(self, *_args) -> None:
        try:
            path = self.s.to_csv()
            self._msg = f"已导出: {path}"
        except Exception as error:
            self._msg = f"导出失败: {error}"

    def _reload(self, *_args) -> None:
        try:
            t = float(self.t[self.i])
            self.s = load_session(self.s.path)
            self.n = len(self.s)
            self.t = self.s.t
            dpg.configure_item("time_slider", min_value=float(self.t[0]), max_value=float(self.t[-1]))
            self._refresh_series(force=True)
            self._set_cursor(self.s.index_at(t))
            self._msg = f"已重新加载: {self.n} 帧"
        except Exception as error:
            self._msg = f"重新加载失败: {error}"

    # ------------------------------------------------------------------
    def _update_info(self) -> None:
        s, i = self.s, self.i
        seg = s.segment_of(i)
        obj = (seg.get("objects") or [{}])[0]
        forces = s["finger_force_n"][i]
        force_txt = "  ".join(f"{GROUP_CN[g]} {forces[n]:.2f}" for n, g in enumerate(CONTACT_FINGERS)
                              if forces[n] > 1e-4) or "无"
        dpg.set_value("cursor_info", (
            f"t = {s.t[i]:.3f} s   帧 {i + 1}/{self.n}   源: {'真机' if s['source'][i] else '假手套'}"
            f"   力反馈: {'开' if s['feedback_on'][i] else '关'}\n"
            f"物体: {obj.get('label', '?')} {obj.get('size_m', 0) * 1000:.0f} mm  硬度 {obj.get('stiffness', 0):.2f}"
            f"   接触数 {int(s['contact_count'][i])}   物体最大变形 {s['object_disp_m'][i] * 1000:.2f} mm\n"
            f"仿真接触力 (N): {force_txt}\n"
            f"控制环 {s['loop_hz'][i]:.0f} Hz   周期耗时 {s['tick_ms'][i]:.1f} ms   "
            f"物理步 {int(s['physics_steps'][i])}   传感器数据龄 {s['sensor_age_ms'][i]:.1f} ms"))
        rows = {
            1: s["real_q"][i] * DEG, 2: s["sim_q"][i] * DEG,
            3: (s["real_q"][i] - s["sim_q"][i]) * DEG, 4: s["surface_q"][i] * DEG,
            5: s["real_dq"][i] * DEG, 6: s["torque_spring_nm"][i], 7: s["torque_driver_nm"][i],
            8: s["torque_sent_nm"][i], 9: s["torque_measured_nm"][i], 10: s["current_a"][i],
            11: s["imp_k"][i], 12: s["imp_b"][i], 13: s["temperature_c"][i],
        }
        fmt = {1: "{:.1f}", 2: "{:.1f}", 3: "{:+.1f}", 4: "{:.1f}", 5: "{:+.0f}", 6: "{:+.3f}",
               7: "{:+.3f}", 8: "{:+.3f}", 9: "{:+.3f}", 10: "{:+.3f}", 11: "{:.1f}", 12: "{:.3f}",
               13: "{:.0f}"}
        for k in range(12):
            for c, arr in rows.items():
                v = float(arr[k])
                dpg.set_value(f"cell_{k}_{c}", "-" if not np.isfinite(v) else fmt[c].format(v))
            flags = []
            if s["blocked"][i][k]:
                flags.append("接触")
            if s["encoder_clamped"][i][k]:
                flags.append("超限")
            if s["motor_error"][i][k]:
                flags.append("报警")
            dpg.set_value(f"cell_{k}_14", " ".join(flags))
        # Contacts the feedback used (recomputed from the recorded state).
        try:
            contacts = s.contacts_at(i, pose="target")
        except Exception as error:
            contacts = []
            self._msg = f"接触重算失败: {error}"
        self._contacts = contacts
        if contacts:
            groups: dict[str, list] = {}
            for c in contacts:
                groups.setdefault(c.group, []).append(c)
            lines = []
            for g in CONTACT_FINGERS:
                cs = groups.get(g)
                if not cs:
                    continue
                total = sum(float(np.linalg.norm(c.force_world_n)) for c in cs)
                pen = max(c.penetration_m for c in cs)
                parts = sorted({c.geom.replace("_visual", "") for c in cs})
                lines.append(f"{GROUP_CN.get(g, g)} - {cs[0].object_name}: {len(cs)} 点  合力 {total:.2f} N"
                             f"  最大穿透 {pen * 1000:.2f} mm  部件 {', '.join(parts)}")
            dpg.set_value("contact_info", "反馈接触（真手姿态，按记录状态重算）:\n" + "\n".join(lines))
        else:
            dpg.set_value("contact_info", "反馈接触: 无")
        past = [(t, e) for t, e in s.event_times() if t <= s.t[i] + 1e-9][-3:]
        if past:
            def describe(e):
                if e["kind"] == "feedback":
                    return "力反馈" + ("开" if e.get("enabled") else "关")
                if e["kind"] == "glove_config":
                    return "参数修改（滑块）"
                return str(e["kind"])
            dpg.set_value("event_info", "此前事件: " + "；".join(
                f"{t:.2f}s {describe(e)}" for t, e in past))
        else:
            dpg.set_value("event_info", "")

    def _xml(self, seg: int) -> str:
        if seg not in self._xml_cache:
            self._xml_cache[seg] = self.s.scene_xml(seg)
        return self._xml_cache[seg]

    def _update_scene(self) -> None:
        img = self.renderer.poll()
        if img is not None and img.shape[:2] == (RENDER_H, RENDER_W):
            dpg.set_value("review_tex", img)
        if not self._dirty_scene or self.renderer._in_flight:
            return
        i = self.i
        seg = int(self.s["segment"][i])
        markers = None
        if self._show_contacts:
            markers = [(c.position_m.tolist(), c.force_world_n.tolist(), CONTACT_RGBA)
                       for c in getattr(self, "_contacts", [])]
        self.renderer.request(self._xml(seg), self.s.qpos_at(i, "visible"),
                              np.zeros(self.s.model(seg).nv), self.cam, RENDER_W, RENDER_H,
                              show_collision=self._show_collision,
                              ghost_qpos=self.s.qpos_at(i, "target") if self._show_ghost else None,
                              markers=markers, arrow_m_per_n=0.01)
        self._dirty_scene = False

    def _orbit(self) -> None:
        if self._orbit_last is None or not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            return
        pos = dpg.get_mouse_pos(local=False)
        dx, dy = pos[0] - self._orbit_last[0], pos[1] - self._orbit_last[1]
        if dx or dy:
            self.cam.azimuth -= dx * 0.4
            self.cam.elevation = float(np.clip(self.cam.elevation - dy * 0.4, -89, 89))
            self._orbit_last = tuple(pos)
            self._dirty_scene = True

    def _layout(self) -> None:
        w, h = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
        if (w, h) == self._view_size:
            return
        self._view_size = (w, h)
        left = int(max(520, min(760, w * 0.42)))
        img_w = left - 16
        dpg.configure_item("left_panel", width=left, height=h - 90)
        dpg.configure_item("review_image", width=img_w, height=int(img_w * RENDER_H / RENDER_W))
        dpg.configure_item("value_table", width=img_w)
        dpg.configure_item("right_panel", width=w - left - 30, height=h - 90)

    def update(self) -> None:
        now = time.perf_counter()
        dt, self._last_frame = now - self._last_frame, now
        self._layout()
        if self._fit_frames > 0:
            self._fit_frames -= 1
            if self._fit_frames == 0:
                for plot in self._plot_tags:
                    dpg.set_axis_limits_auto(f"x_{plot}")
            else:
                for plot in self._plot_tags:
                    dpg.set_axis_limits(f"x_{plot}", float(self.t[0]), float(self.t[-1]))
        self._refresh_series()
        if self.playing:
            self._play_t += dt * self.speed
            if self._play_t >= self.t[-1]:
                self._play_t = float(self.t[-1])
                self.playing = False
                dpg.configure_item("play_button", label="播放")
            j = self.s.index_at(self._play_t)
            if j != self.i:
                self._set_cursor(j, from_play=True)
        self._orbit()
        try:
            self._update_scene()
        except Exception as error:
            self._msg = f"渲染失败: {error}"
        dpg.set_value("review_status", self._msg)

    def close(self) -> None:
        self.renderer.close()


def run_review(path: str | Path) -> int:
    if dpg is None:
        print(f"UI 不可用（DearPyGui 导入失败: {_DPG_IMPORT_ERROR}）")
        return 2
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("没有可用的图形显示；可以用 Python 直接分析: from glove_sim.recording import load_session")
        return 2
    try:
        session = load_session(path)
    except Exception as error:
        print(f"REVIEW_FAIL: 无法加载 {path}: {error}")
        return 2
    dpg.create_context()
    for font_path in FONT_CANDIDATES:
        if Path(font_path).is_file():
            with dpg.font_registry():
                font = dpg.add_font(font_path, 17)
            dpg.bind_font(font)
            break
    dpg.create_viewport(title=f"记录回放分析 - {Path(path).name}", width=1700, height=1000,
                        x_pos=60, y_pos=30, resizable=True, min_width=1100, min_height=700)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    app = ReviewApp(session)
    dpg.set_primary_window("review_main", True)
    try:
        while dpg.is_dearpygui_running():
            app.update()
            dpg.render_dearpygui_frame()
    finally:
        app.close()
        dpg.destroy_context()
    return 0
