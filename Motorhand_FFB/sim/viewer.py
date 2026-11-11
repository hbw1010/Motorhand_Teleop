"""Interactive MuJoCo display operating exclusively on snapshots of the simulation."""
from __future__ import annotations

import copy
from collections import deque
import math
import threading
import time

import glfw
import mujoco
import numpy as np


class LiveViewer:
    """Render on a separate thread; edits are absolute requests to the owner thread."""

    _MIN_SIZE = 0.01
    _PANEL = 300

    def __init__(self, scene, *, editing_enabled=True):
        self._editing_enabled = editing_enabled
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._running = threading.Event()
        self._running.set()
        self._reset = threading.Event()
        self._error = None
        self._snapshot = None
        self._edit = None
        self._edit_queue = deque()
        self._feedback_state = "active"
        self._blocked_until = 0.0
        self._source_model = None
        self._render_model = None
        self._reference = tuple(scene.metadata["reference_point_world_m"])
        self._deformable = scene.metadata["object_model"] == "deformable"
        if self._deformable:
            position = scene.metadata["object_initial_position_m"]
            self._reference = tuple((a + b) / 2 for a, b in zip(self._reference, position))
        self.submit(scene)
        self._thread = threading.Thread(target=self._run, name="glove-sim-viewer", daemon=True)
        self._thread.start()

    def submit(self, scene):
        # Copy model only on generation change. The render thread owns this copy.
        if scene.model is not self._source_model:
            self._source_model = scene.model
            self._render_model = copy.copy(scene.model)
        position, quat = scene.object_pose()
        body = scene.object_body_id
        snapshot = (self._render_model, scene.data.qpos.copy(), scene.data.qvel.copy(),
                    float(scene.data.time), scene.model.body_pos[body].copy(),
                    scene.model.body_quat[body].copy(), np.asarray(position).copy(),
                    np.asarray(quat).copy(), np.asarray(scene.object_dimensions_m).copy(), body)
        with self._lock:
            self._snapshot = snapshot

    def consume_edit(self):
        with self._lock:
            if self._editing_enabled:
                edit, self._edit = self._edit, None
            else:
                edit = self._edit_queue.popleft() if self._edit_queue else None
        return edit

    def set_feedback_state(self, state):
        if state not in ("active", "dragging", "waiting_clear"):
            raise ValueError(f"Unknown feedback state: {state}")
        with self._lock:
            self._feedback_state = state

    def notify_blocked(self):
        with self._lock:
            self._blocked_until = time.monotonic() + 2.0

    def consume_reset(self):
        requested = self._reset.is_set()
        self._reset.clear()
        return requested

    def is_running(self):
        self._raise_error()
        return self._running.is_set()

    def _raise_error(self):
        if self._error is not None:
            raise RuntimeError("MuJoCo 渲染线程失败") from self._error

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError("MuJoCo 渲染线程未能退出")
        self._raise_error()

    def _request_edit(self, dragging=None):
        request = {"position_m": self._position.tolist(), "quat_wxyz": self._quat.tolist(),
                   "dimensions_m": self._dimensions.tolist()}
        if not self._editing_enabled:
            request["dragging"] = dragging
        with self._lock:
            if self._editing_enabled:
                self._edit = request
            elif dragging and self._edit_queue and self._edit_queue[-1]["dragging"]:
                self._edit_queue[-1] = request
            else:
                self._edit_queue.append(request)

    def _callback(self, function):
        # Exceptions cannot propagate through a GLFW C callback.
        def wrapped(*args):
            try:
                function(*args)
            except BaseException as exc:
                self._error = exc
                self._stop.set()
        return wrapped

    def _run(self):
        window = None
        context = None
        initialized = False
        try:
            if not glfw.init():
                raise RuntimeError("GLFW 初始化失败")
            initialized = True
            window = glfw.create_window(1200, 800, "Motorhand simulation", None, None)
            if not window:
                raise RuntimeError("无法创建 MuJoCo 窗口")
            glfw.make_context_current(window)
            glfw.swap_interval(1)
            self._window = window
            self._camera = mujoco.MjvCamera()
            self._camera.lookat[:] = self._reference
            self._camera.azimuth = 200 if self._deformable else 110
            self._camera.elevation = -50 if self._deformable else -25
            self._camera.distance = 0.32
            self._option = mujoco.MjvOption()
            self._model = None
            self._drag = None
            self._last = (0.0, 0.0)
            self._slider_max = 0.2
            glfw.set_key_callback(window, self._callback(self._key))
            glfw.set_mouse_button_callback(window, self._callback(self._button))
            glfw.set_cursor_pos_callback(window, self._callback(self._cursor))
            glfw.set_scroll_callback(window, self._callback(self._scroll))
            while not self._stop.is_set() and not glfw.window_should_close(window):
                render_started = time.monotonic()
                with self._lock:
                    snapshot, self._snapshot = self._snapshot, None
                if snapshot is not None:
                    model, qpos, qvel, sim_time, body_pos, body_quat, pos, quat, dims, body = snapshot
                    if model is not self._model:
                        if context is not None:
                            context.free()
                            context = None
                        self._model = model
                        self._data = mujoco.MjData(model)
                        self._scene = mujoco.MjvScene(model, maxgeom=10000)
                        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
                    self._body = body
                    model.body_pos[body] = body_pos
                    model.body_quat[body] = body_quat
                    self._data.qpos[:] = qpos
                    self._data.qvel[:] = qvel
                    self._data.time = sim_time
                    mujoco.mj_forward(model, self._data)
                    self._actual_position, self._actual_quat, self._actual_dimensions = pos, quat, dims
                    if self._drag is None:
                        self._position, self._quat, self._dimensions = pos.copy(), quat.copy(), dims.copy()
                        self._slider_max = max(0.2, float(max(dims)) * 1.2)
                if self._model is not None:
                    self._draw(context)
                glfw.swap_buffers(window)
                glfw.poll_events()
                # Some X11/virtual displays ignore swap_interval. Limit render
                # load explicitly so the diagnostic window cannot starve control.
                self._stop.wait(max(0.0, 1.0 / 30.0 - (time.monotonic() - render_started)))
        except BaseException as exc:
            self._error = exc
        finally:
            try:
                if context is not None:
                    context.free()
                if window is not None:
                    glfw.destroy_window(window)
                if initialized:
                    glfw.terminate()
            except BaseException as exc:
                if self._error is None:
                    self._error = exc
            self._running.clear()

    def _draw(self, context):
        width, height = glfw.get_framebuffer_size(self._window)
        if width <= 0 or height <= 0:
            return
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(self._model, self._data, self._option, None, self._camera,
                               mujoco.mjtCatBit.mjCAT_ALL, self._scene)
        # Material-frame arrows remain visible even on a rotationally symmetric sphere.
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, self._actual_quat)
        matrix = matrix.reshape(3, 3)
        length = float(max(self._actual_dimensions)) * 0.7
        for axis, color in enumerate(((1, .1, .1, 1), (.1, 1, .1, 1), (.2, .4, 1, 1))):
            if self._scene.ngeom >= self._scene.maxgeom:
                break
            geom = self._scene.geoms[self._scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                                np.eye(3).ravel(), np.array(color, dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.0015,
                                 self._actual_position - length * matrix[:, axis],
                                 self._actual_position + length * matrix[:, axis])
            self._scene.ngeom += 1
        mujoco.mjr_render(viewport, self._scene, context)
        self._panel(context, width, height)

    def _panel(self, context, width, height):
        panel = min(self._PANEL, width)
        x0 = width - panel
        mujoco.mjr_rectangle(mujoco.MjrRect(x0, 0, panel, height), .08, .10, .13, 1)
        font = mujoco.mjtFont.mjFONT_NORMAL
        def label(y, text):
            mujoco.mjr_label(mujoco.MjrRect(x0 + 5, height - y - 22, panel - 10, 22),
                             font, text, .08, .10, .13, 1, .9, .9, .9, context)
        label(12, "OBJECT EDITOR" if self._editing_enabled else "EXECUTE: POSITION ONLY")
        label(44, "Object: left drag = move")
        label(68, "Shift + left drag = rotate" if self._editing_enabled else "Rotation and size disabled")
        label(92, "Camera: left orbit, right pan")
        label(116, "Scroll = zoom     R = reset" if self._editing_enabled else "Scroll = zoom     R disabled")
        label(145, "Size in meters (release to apply)" if self._editing_enabled else "Size in meters (read only)")
        label(170, "Axes: X red, Y green, Z blue")
        for i, axis in enumerate("XYZ"):
            y = 215 + i * 70
            label(y - 25, f"{axis}: {self._dimensions[i]:.4f} m")
            left, right = x0 + 25, width - 25
            track_y = height - y - 10
            shade = .6 if self._editing_enabled else .25
            mujoco.mjr_rectangle(mujoco.MjrRect(left, track_y, max(1, right-left), 4), shade, shade, shade, 1)
            fraction = np.clip((self._dimensions[i] - self._MIN_SIZE) /
                               (self._slider_max - self._MIN_SIZE), 0, 1)
            knob = int(left + fraction * (right-left))
            mujoco.mjr_rectangle(mujoco.MjrRect(knob-5, track_y-5, 10, 14),
                                 .2 if self._editing_enabled else .35, .7 if self._editing_enabled else .35, .9 if self._editing_enabled else .35, 1)
        with self._lock:
            blocked = time.monotonic() < self._blocked_until
            feedback_state = self._feedback_state
        if blocked:
            label(410, "Blocked: move away from glove")
        if not self._editing_enabled:
            label(440, "Drag requests motor output pause")
            label(464, "Release and clear contact to resume")
            if feedback_state == "dragging":
                label(494, "OUTPUT PAUSE REQUESTED")
            elif feedback_state == "waiting_clear":
                label(494, "OUTPUT PAUSE REQUESTED")
                label(518, "RELEASE OBJECT / OPEN GLOVE")
                label(542, "WAIT FOR ALIGNMENT / LOW CONTACT")

    def _coords(self, x, y):
        ww, wh = glfw.get_window_size(self._window)
        fw, fh = glfw.get_framebuffer_size(self._window)
        return x * fw / max(ww, 1), y * fh / max(wh, 1), fw, fh

    def _key(self, window, key, scancode, action, mods):
        if self._editing_enabled and action == glfw.PRESS and key == glfw.KEY_R:
            self._reset.set()

    def _button(self, window, button, action, mods):
        x, y = glfw.get_cursor_pos(window)
        x, y, width, height = self._coords(x, y)
        self._last = (x, y)
        if action == glfw.RELEASE:
            if button != glfw.MOUSE_BUTTON_LEFT:
                if self._drag and self._drag[0] == "pan" and button == glfw.MOUSE_BUTTON_RIGHT:
                    self._drag = None
                return
            if self._drag and self._drag[0] == "slider":
                self._request_edit()
            elif self._drag and self._drag[0] == "move" and not self._editing_enabled:
                self._request_edit(dragging=False)
            self._drag = None
            return
        if action != glfw.PRESS or self._model is None:
            return
        if self._drag is not None:
            return
        if (not self._editing_enabled and button == glfw.MOUSE_BUTTON_LEFT
                and mods & glfw.MOD_SHIFT):
            return
        if x >= width - self._PANEL:
            if self._editing_enabled and button == glfw.MOUSE_BUTTON_LEFT:
                for i in range(3):
                    if abs(y - (225 + i * 70)) <= 20:
                        self._drag = ("slider", i)
                        self._set_slider(x, width, i)
                        return
            return
        if button == glfw.MOUSE_BUTTON_LEFT:
            point = np.zeros(3)
            geom, flex, skin = (np.full(1, -1, dtype=np.int32) for _ in range(3))
            body = mujoco.mjv_select(self._model, self._data, self._option, width / max(height, 1),
                                     x / max(width, 1), 1 - y / max(height, 1),
                                     self._scene, point, geom, flex, skin)
            object_geom = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "scene_object_geom")
            object_flex = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_FLEX, "scene_soft")
            if (body == self._body or (geom[0] >= 0 and geom[0] == object_geom) or
                    (flex[0] >= 0 and flex[0] == object_flex)):
                self._drag = ("rotate" if mods & glfw.MOD_SHIFT else "move",)
                if not self._editing_enabled:
                    self._request_edit(dragging=True)
                return
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._drag = ("orbit",)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._drag = ("pan",)

    def _set_slider(self, x, width, axis):
        left, right = width - min(self._PANEL, width) + 25, width - 25
        fraction = np.clip((x - left) / max(right-left, 1), 0, 1)
        self._dimensions[axis] = self._MIN_SIZE + fraction * (self._slider_max - self._MIN_SIZE)

    def _cursor(self, window, x, y):
        x, y, width, height = self._coords(x, y)
        dx, dy = x - self._last[0], y - self._last[1]
        self._last = (x, y)
        if not self._drag or not height:
            return
        kind = self._drag[0]
        if kind == "slider":
            self._set_slider(x, width, self._drag[1])
        elif kind in ("orbit", "pan"):
            action = (mujoco.mjtMouse.mjMOUSE_ROTATE_H if kind == "orbit"
                      else mujoco.mjtMouse.mjMOUSE_MOVE_H)
            mujoco.mjv_moveCamera(self._model, action, dx / height, dy / height, self._camera)
        else:
            camera = self._scene.camera[0]
            forward = np.asarray(camera.forward, dtype=float)
            up = np.asarray(camera.up, dtype=float)
            right = np.cross(forward, up)
            right /= np.linalg.norm(right)
            up /= np.linalg.norm(up)
            if kind == "move":
                scale = 2 * self._camera.distance * math.tan(math.radians(45) / 2) / height
                self._position = self._position + scale * (dx * right - dy * up)
            else:
                vector = (dx * up + dy * right) * (math.pi / height)
                angle = np.linalg.norm(vector)
                if angle:
                    delta = np.r_[math.cos(angle / 2), vector / angle * math.sin(angle / 2)]
                    result = np.empty(4)
                    mujoco.mju_mulQuat(result, delta, self._quat)
                    self._quat = result / np.linalg.norm(result)
            self._request_edit(dragging=True)

    def _scroll(self, window, xoffset, yoffset):
        if self._model is not None:
            mujoco.mjv_moveCamera(self._model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                                 0, -0.05 * yoffset, self._camera)
