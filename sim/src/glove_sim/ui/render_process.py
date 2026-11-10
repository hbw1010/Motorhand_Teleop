from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any

import numpy as np

# The worker deliberately owns the only MuJoCo OpenGL context.  DearPyGui has
# its own GLFW/GL context in the parent process; creating a MuJoCo offscreen
# renderer there makes DPG render black because the two contexts fight over the
# current context.  A separate process is the robust boundary.
def _render_worker(conn) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import numpy as np

    model = None
    data = None
    renderer = None
    xml: str | None = None
    width = height = 0
    base_geom_colors: dict[int, np.ndarray] = {}
    base_flex_colors: dict[int, np.ndarray] = {}
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg is None:
            break
        try:
            new_xml = msg.get("xml")
            new_width = int(msg.get("width", width or 1280))
            new_height = int(msg.get("height", height or 720))
            if (new_xml is not None and new_xml != xml) or renderer is None \
                    or new_width != width or new_height != height:
                if new_xml is not None and new_xml != xml:
                    xml = new_xml
                    model = mujoco.MjModel.from_xml_string(xml)
                    data = mujoco.MjData(model)
                    base_geom_colors = {
                        gid: model.geom_rgba[gid].copy()
                        for gid in range(model.ngeom)
                        if (model.geom(gid).name or "").startswith("obj_")
                    }
                    base_flex_colors = {
                        fid: model.flex_rgba[fid].copy()
                        for fid in range(model.nflex)
                        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, fid) or "")
                        .startswith("flex_")
                    }
                if model is None or data is None:
                    raise RuntimeError("render worker has no model")
                width, height = new_width, new_height
                if renderer is not None:
                    renderer.close()
                renderer = mujoco.Renderer(model, height=height, width=width)
            if model is None or data is None or renderer is None:
                raise RuntimeError("render worker has no model")
            data.qpos[:] = np.asarray(msg["qpos"], dtype=np.float64)
            data.qvel[:] = np.asarray(msg["qvel"], dtype=np.float64)
            mujoco.mj_forward(model, data)
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.lookat[:] = msg["lookat"]
            cam.azimuth = float(msg["azimuth"])
            cam.elevation = float(msg["elevation"])
            cam.distance = float(msg["distance"])
            highlight = msg.get("highlight")
            # Restore the base palette, then tint the selected object.  Doing
            # this every frame keeps selection visible even after a rebuild.
            for gid, rgba in base_geom_colors.items():
                model.geom_rgba[gid] = rgba
            for fid, rgba in base_flex_colors.items():
                model.flex_rgba[fid] = rgba
            if highlight and highlight.get("name"):
                selected = str(highlight["name"])
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                        f"obj_{selected}_geom")
                if gid >= 0:
                    model.geom_rgba[gid] = (1.0, 0.82, 0.10, 1.0)
                fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX,
                                        f"flex_{selected}")
                if fid >= 0:
                    model.flex_rgba[fid] = (1.0, 0.82, 0.10, 1.0)
            renderer.update_scene(data, camera=cam)
            # The offscreen shadow map is coarse enough to paint large jagged
            # blobs on the floor.  A clean, evenly-lit scene reads better here.
            try:
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            except Exception:
                pass
            if highlight:
                scene = renderer.scene
                if scene.ngeom < scene.maxgeom:
                    shape = highlight.get("shape", "sphere")
                    size_m = float(highlight.get("size", 0.03))
                    inflate = 1.12
                    if shape == "box":
                        gtype = mujoco.mjtGeom.mjGEOM_BOX
                        h = size_m * 0.5 * inflate
                        gsize = np.asarray([h, h, h], dtype=np.float64)
                    elif shape == "cylinder":
                        gtype = mujoco.mjtGeom.mjGEOM_CYLINDER
                        gsize = np.asarray([size_m * 0.5 * inflate,
                                            size_m * 0.4 * inflate, 0.0], dtype=np.float64)
                    else:
                        gtype = mujoco.mjtGeom.mjGEOM_SPHERE
                        gsize = np.asarray([size_m * 0.5 * inflate, 0.0, 0.0], dtype=np.float64)
                    q = np.asarray(highlight.get("quat", [1, 0, 0, 0]), dtype=np.float64)
                    w, x, y, z = q / max(1e-12, float(np.linalg.norm(q)))
                    mat = np.asarray([
                        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    ], dtype=np.float64)
                    pos = np.asarray(highlight.get("pos", [0, 0, 0]), dtype=np.float64)
                    rgba = np.asarray([1.0, 0.78, 0.12, 0.38], dtype=np.float32)
                    geom = scene.geoms[scene.ngeom]
                    # MuJoCo 4.0 takes a flat 9-vector for the rotation matrix.
                    mujoco.mjv_initGeom(geom, gtype, gsize.reshape(3), pos.reshape(3),
                                        mat.reshape(9), rgba.reshape(4))
                    scene.ngeom += 1
            image = np.asarray(renderer.render(), dtype=np.uint8)
            conn.send({"shape": image.shape, "dtype": image.dtype.str,
                       "data": image.tobytes()})
        except Exception as error:  # keep the parent informed and alive
            import traceback
            traceback.print_exc()
            try:
                conn.send({"error": f"{type(error).__name__}: {error}"})
            except Exception:
                break


class RenderProcess:
    """Synchronous renderer IPC; returns an RGB uint8 image per request."""

    def __init__(self) -> None:
        context = mp.get_context("spawn")
        self._conn, child = context.Pipe()
        self._process = context.Process(target=_render_worker, args=(child,), daemon=True)
        self._process.start()
        child.close()
        self._xml: str | None = None
        self._in_flight = False
        self._last_image: np.ndarray | None = None

    @property
    def alive(self) -> bool:
        return self._process.is_alive()

    def request(self, xml: str, qpos: np.ndarray, qvel: np.ndarray,
                camera: Any, width: int, height: int,
                highlight: dict[str, Any] | None = None) -> None:
        """Queue one frame without waiting for the renderer process."""
        if self._in_flight:
            return
        msg: dict[str, Any] = {
            "qpos": np.asarray(qpos, dtype=np.float64).copy(),
            "qvel": np.asarray(qvel, dtype=np.float64).copy(),
            "lookat": list(map(float, camera.lookat)),
            "azimuth": float(camera.azimuth),
            "elevation": float(camera.elevation),
            "distance": float(camera.distance),
            "width": int(width),
            "height": int(height),
        }
        if highlight is not None:
            msg["highlight"] = highlight
        if xml != self._xml:
            msg["xml"] = xml
            self._xml = xml
        try:
            self._conn.send(msg)
            self._in_flight = True
        except (EOFError, OSError) as error:
            raise RuntimeError(f"渲染子进程退出: {error}") from error

    def poll(self) -> np.ndarray | None:
        """Return the newest completed image, or None if still rendering."""
        if not self._in_flight:
            return None
        if not self._conn.poll(0.0):
            return None
        try:
            answer = self._conn.recv()
        except (EOFError, OSError) as error:
            self._in_flight = False
            raise RuntimeError(f"渲染子进程退出: {error}") from error
        self._in_flight = False
        if "error" in answer:
            raise RuntimeError(answer["error"])
        image = np.frombuffer(answer["data"], dtype=np.dtype(answer["dtype"]))
        self._last_image = image.reshape(answer["shape"]).copy()
        return self._last_image

    @property
    def last_image(self) -> np.ndarray | None:
        return self._last_image

    def close(self) -> None:
        try:
            if self._process.is_alive() and not self._in_flight:
                self._conn.send(None)
        except Exception:
            pass
        try:
            self._process.join(timeout=2.0)
        except Exception:
            pass
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        try:
            self._conn.close()
        except Exception:
            pass
