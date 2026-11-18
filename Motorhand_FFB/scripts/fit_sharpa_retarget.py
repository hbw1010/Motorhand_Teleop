"""从现有 CAD 指腹规则和闭链仿真姿态生成 Sharpa 任务帧与几何比例。

只写 Sharpa 派生资产和 retargeting/sharpa 配置；不修改手套、Wuji 或硬件标定。
输入 points_m 必须已经是腕相对 canonical 米，Sharpa 不再旋转 90 度。
几何拟合以同语义关节的 Sharpa FK 为参考；native probe 独立检验采用 Wuji
目标函数的 Sharpa 私有优化器输出。两者均不等同于 MuJoCo 动态验证或真实硬件标定。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml

from generate_pad_task_points import FINGERS, TIP_FRACTION, _Hand, _finger

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config/retargeting/sharpa"
SHARPA = ROOT / "assets/sharpa-hand"
TASK_INDICES = np.array([[4 * f + 2 + j for j in range(3)] for f in range(5)])
JOINT_SUFFIXES = (
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
)


def source_urdf(side: str) -> Path:
    return SHARPA / "wave_01" / f"{side}_sharpa_wave" / f"{side}_sharpa_wave.urdf"


def pad_urdf(side: str) -> Path:
    return source_urdf(side).with_name(f"{side}_sharpa_wave_pad.urdf")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_visual_hand(side: str) -> _Hand:
    """仅临时解析 package URI；网格加载和 CAD 规则复用 Wuji 生成器。"""
    source = source_urdf(side)
    resolved = source.read_text().replace(
        f"package://{side}_sharpa_wave/", str(source.parent.resolve()) + "/"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf") as stream:
        stream.write(resolved)
        stream.flush()
        return _Hand(Path(stream.name))


def ray_surface_depth(vertices: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> float:
    """沿掌侧法线与真实 STL 三角面求交，检验截面近似任务点的表面距离。"""
    triangles = vertices.reshape(-1, 3, 3)
    a = triangles[:, 0]
    e1, e2 = triangles[:, 1] - a, triangles[:, 2] - a
    h = np.cross(np.broadcast_to(normal, e2.shape), e2)
    det = np.einsum("ij,ij->i", e1, h)
    valid = np.abs(det) > 1e-14
    inv = np.zeros_like(det)
    inv[valid] = 1 / det[valid]
    offset = origin - a
    u = inv * np.einsum("ij,ij->i", offset, h)
    cross = np.cross(offset, e1)
    v = inv * (cross @ normal)
    depth = inv * np.einsum("ij,ij->i", e2, cross)
    valid &= (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8) & (depth >= 0)
    if not np.any(valid):
        raise ValueError("指腹中心掌侧射线没有与真实网格相交")
    return float(depth[valid].max())


def generate_geometry(side: str) -> dict:
    hand = load_visual_hand(side)
    extra, report = [], {}
    xml = ET.parse(source_urdf(side)).getroot()
    source_links = {link.attrib["name"]: link for link in xml.findall("link")}
    mesh_hashes = {}
    for finger in FINGERS:
        suffix = ("PP", "DP", "fingertip") if finger == "thumb" else ("MP", "DP", "fingertip")
        parents = [f"{side}_{finger}_{part}" for part in suffix]
        axis = [hand.placement(frame).translation.copy() for frame in parents]
        mesh_names = [f"{side}_{finger}_DP", f"{side}_{finger}_elastomer"]
        for name in mesh_names:
            for mesh_node in source_links[name].findall("visual/geometry/mesh"):
                mesh_path = source_urdf(side).parent / mesh_node.attrib["filename"].split(f"{side}_sharpa_wave/")[1]
                mesh_hashes[str(mesh_path.relative_to(SHARPA))] = sha256(mesh_path)
        mesh = np.vstack([hand.mesh[name] for name in mesh_names])
        check = (hand.placement(f"{side}_index_fingertip").translation - axis[2]
                 if finger == "thumb" else np.array([1., 0., 0.]))
        pads, info = _finger(hand, axis, mesh, parents[1], parents[2], np.zeros(3), check, 0.)
        normal = pads[1] - axis[1]
        depth = float(np.linalg.norm(normal))
        normal /= depth
        surface = ray_surface_depth(mesh, pads[2] - normal * depth, normal)
        # 沿用 Wuji 的截面顶点规则，记录其与精确三角面的偏差，禁止默默改规则。
        surface_error = abs(surface - depth)
        if surface_error > .001:
            raise ValueError(f"{side}/{finger} 指腹任务点偏离真实网格 {surface_error * 1000:.3f} mm")
        frames = []
        for key, parent, point in zip(("pip", "dip", "tip"), parents, pads):
            name = f"{side}_{finger}_pad_{key}"
            local = hand.placement(parent).actInv(point)
            xyz = " ".join(f"{v:.12g}" for v in local)
            extra.append(f'  <link name="{name}" />\n'
                         f'  <joint name="{name}_fixed" type="fixed">\n'
                         f'    <parent link="{parent}" />\n    <child link="{name}" />\n'
                         f'    <origin xyz="{xyz}" rpy="0 0 0" />\n  </joint>\n')
            frames.append({"name": name, "parent": parent, "local_xyz_m": local.tolist(),
                           "zero_world_xyz_m": point.tolist()})
        report[finger] = {**info, "mesh_links": mesh_names, "palmar_normal": normal.tolist(),
                          "palmar_dot_check": float(normal @ check),
                          "tip_surface_error_mm": surface_error * 1000, "frames": frames}
    original = source_urdf(side).read_text()
    marker = "<!-- generated pad task frames: scripts/fit_sharpa_retarget.py -->\n"
    pad_urdf(side).write_text(original[:original.rindex("</robot>")] + marker + "".join(extra) + "</robot>\n")
    return {"source_urdf_sha256": sha256(source_urdf(side)), "derived_urdf_sha256": sha256(pad_urdf(side)),
            "source_mesh_sha256": mesh_hashes, "tip_fraction_from_distal_end": TIP_FRACTION, "fingers": report}


class SharpaFK:
    def __init__(self, side: str, axis_scale=None, wrist_offset=None):
        self.side = side
        self.axis_scale = np.ones(3) if axis_scale is None else np.asarray(axis_scale, dtype=float)
        self.wrist_offset = np.zeros(3) if wrist_offset is None else np.asarray(wrist_offset, dtype=float)
        self.model = pin.buildModelFromUrdf(str(pad_urdf(side)))
        self.data = self.model.createData()
        self.indices = np.array([self.model.joints[self.model.getJointId(f"{side}_{s}")].idx_q
                                 for s in JOINT_SUFFIXES])
        names = [f"{side}_hand_C_MC"]
        for finger in FINGERS:
            names += [f"{side}_{finger}_MCP_VL"] + [f"{side}_{finger}_pad_{k}" for k in ("pip", "dip", "tip")]
        self.frames = [self.model.getFrameId(name) for name in names]
        if self.model.nq != 22 or any(i >= self.model.nframes for i in self.frames):
            raise ValueError("Sharpa 22 轴/21任务帧不完整")

    def physical_points(self, q_rad: np.ndarray) -> np.ndarray:
        q = np.zeros(22)
        q[self.indices] = q_rad
        pin.framesForwardKinematics(self.model, self.data, q)
        points = np.array([self.data.oMf[f].translation for f in self.frames])
        points -= points[0]
        points[1] = .5 * points[2]
        return points

    def points(self, q_rad: np.ndarray) -> np.ndarray:
        points = (self.physical_points(q_rad) - self.wrist_offset) / self.axis_scale
        points[0] = 0.
        points[1] = .5 * points[2]
        return points

    def reference(self, frame: dict) -> tuple[np.ndarray, np.ndarray]:
        """显式同语义姿态只用于离线比例拟合，不用于运行时重定向。"""
        glove = np.asarray(frame["joint_rad"], dtype=float)
        dependent = np.asarray(frame["dependent_rad"], dtype=float)
        if glove.shape != (12,) or dependent.shape != (4,):
            raise ValueError("fixture 需要12个独立角和4个真实闭链DIP角")
        q = np.zeros(22)
        q[[0, 1, 2, 4]] = glove[:4]
        # 左手 Motorhand 的 CMC 侧摆轴镜像后角度变号；Sharpa 左右 URDF 的
        # 同名 CMC_AA 则同角度得到镜像姿态。此处仅对离线参考角作设备映射。
        if self.side == "left":
            q[1] = -q[1]
        for f, start in enumerate((5, 9, 13, 18)):
            q[start] = glove[4 + 2 * f]
            q[start + 2] = glove[5 + 2 * f]
            q[start + 3] = dependent[f]
        lower = self.model.lowerPositionLimit[self.indices]
        upper = self.model.upperPositionLimit[self.indices]
        q = np.clip(q, lower, upper)
        return self.points(q), q


def write_config(side: str, scales: np.ndarray, *, path: Path | None = None,
                 lock_extra: bool = False, pending: bool = False, pinch_scale: float = 1.,
                 axis_scale=None, wrist_offset=None, thumb_aa_fixed: float | None = None,
                 norm_delta: float = .04, w_hyper: float = 1., w_couple: float = .1) -> Path:
    if not np.isfinite(norm_delta) or norm_delta < 0:
        raise ValueError("norm_delta必须为有限非负数")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = path or CONFIG_DIR / f"{side}.yaml"
    names = [f"{side}_{finger}_MCP_AA" for finger in FINGERS[1:]]
    if lock_extra:
        names += [f"{side}_thumb_MCP_AA", f"{side}_pinky_CMC"]
    fixed_joints = dict.fromkeys(names, 0.)
    if thumb_aa_fixed is not None:
        fixed_joints[f"{side}_thumb_MCP_AA"] = float(thumb_aa_fixed)
    config = {
        "geometry": {"task_axis_scale": list(map(float, np.ones(3) if axis_scale is None else axis_scale)),
                     "wrist_offset_m": list(map(float, np.zeros(3) if wrist_offset is None else wrist_offset))},
        "optimizer": {"type": "AdaptiveOptimizerAnalytical",
                      "urdf_path": (str(pad_urdf(side)) if path.parent != CONFIG_DIR else
                                    f"../../../assets/sharpa-hand/wave_01/{side}_sharpa_wave/{side}_sharpa_wave_pad.urdf"),
                      "link_naming": {"prefix": f"{side}_", "palm": "hand_C_MC", "fingers": list(FINGERS),
                                      "pip": "{finger}_pad_pip", "dip": "{finger}_pad_dip",
                                      "tip": "{finger}_pad_tip", "link1": "{finger}_MCP_VL"}},
        "retarget": {"huber_delta": 2., "huber_delta_dir": .5, "norm_delta": float(norm_delta),
                     "w_pos": 1., "w_dir": 2., "w_full_hand": 1., "scaling": round(float(pinch_scale), 6),
                     "fixed_joints": fixed_joints, "thumb_skip_pip": False,
                     "w_hyper": float(w_hyper), "soft_min": 0., "w_couple": float(w_couple), "couple_ratio": .7,
                     "lp_alpha": .2,
                     "segment_scaling": {finger: [round(float(v), 6) for v in scales[f]]
                                         for f, finger in enumerate(FINGERS)},
                     "pinch_thresholds": {finger: {"d1": 2., "d2": 4.} for finger in FINGERS[1:]},
                     "pinch_weight_max": .7, "mediapipe_rotation": dict.fromkeys(("x", "y", "z"), 0.),
                     "wrist_offset_cm": [0., 0., 0.], "thumb_offset_cm": [0., 0., 0.]}}
    note = ("待多姿态拟合：当前仅为CAD初始化值。" if pending else
            "比例由闭链模型多姿态训练集拟合，见fit_report.json；不是实物标定。")
    path.write_text("# scripts/fit_sharpa_retarget.py 生成；Sharpa私有目标函数及损失权重遵循Wuji。\n"
                    "# canonical方向与Sharpa URDF一致；geometry拟合掌尺度/腕参考，无额外90度旋转。\n"
                    f"# {note}\n" + yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    return path


def fixture_frames(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if data.get("input_frame") != "canonical" or data.get("input_units") != "m":
        raise ValueError("fixture必须显式声明input_frame=canonical、input_units=m")
    frames = data["frames"]
    for frame in frames:
        points = np.asarray(frame["points_m"])
        if points.shape != (21, 3) or not np.isfinite(points).all():
            raise ValueError("fixture points_m须为有限21x3 canonical米")
        if "split" not in frame:
            raise ValueError("fixture必须显式标注train/validation，禁止偷偷在验证集上拟合")
    if not {"train", "validation"}.issubset({f["split"] for f in frames}):
        raise ValueError("fixture缺少训练或验证分组")
    return frames


def fit_task_geometry(frames: list[dict], side: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """在同canonical方向下拟合物理掌宽/掌长/屈曲跨度及腕参考位置。"""
    train = [frame for frame in frames if frame["split"] == "train"]
    fk = SharpaFK(side)
    g = np.array([frame["points_m"] for frame in train])
    r = np.array([fk.reference(frame)[0] for frame in train])
    zero = min(range(len(train)), key=lambda i: np.linalg.norm(train[i]["joint_rad"]))
    palm = [5, 9, 13, 17]
    axis, offset = np.ones(3), np.zeros(3)
    for dimension in (1, 2):
        axis[dimension], offset[dimension] = np.linalg.lstsq(
            np.c_[g[zero, palm, dimension], np.ones(4)], r[zero, palm, dimension], rcond=None)[0]
    indices = TASK_INDICES[1:].ravel()
    glove_motion = (g - g[zero])[:, indices, 0]
    robot_motion = (r - r[zero])[:, indices, 0]
    axis[0] = np.sum(glove_motion * robot_motion) / np.sum(glove_motion ** 2)
    offset[0] = np.mean(r[zero, palm, 0] - axis[0] * g[zero, palm, 0])
    if not np.all(axis > 0) or not np.isfinite(axis).all() or not np.isfinite(offset).all():
        raise ValueError("掌几何拟合不合法")
    residual = r[zero, palm] - (g[zero, palm] * axis + offset)
    return axis, offset, {"method": "y/z以四MCP零位仿射LS拟合；x以训练四指屈曲位移LS拟合，腕偏移取MCP残差均值",
                          "task_axis_scale": axis.tolist(), "wrist_offset_m": offset.tolist(),
                          "palm_residual_mm": (residual * 1000).tolist(),
                          "task_domain": "(physical_pad - wrist_offset_m) / task_axis_scale；point0=0，point1=point2/2",
                          "rotation": "canonical方向不变；没有Wuji的90度旋转"}


def fit_scales(frames: list[dict], fk: SharpaFK) -> tuple[np.ndarray, dict]:
    train = [frame for frame in frames if frame["split"] == "train"]
    # 不让保持和回开重复帧改变统计权重。
    unique = {np.asarray(frame["points_m"]).round(10).tobytes(): frame for frame in train}
    source = np.array([frame["points_m"] for frame in unique.values()])
    source -= source[:, :1]
    reference = np.array([fk.reference(frame)[0] for frame in unique.values()])
    g, r = source[:, TASK_INDICES], reference[:, TASK_INDICES]
    scales = np.einsum("nfjc,nfjc->fj", g, r) / np.einsum("nfjc,nfjc->fj", g, g)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("几何拟合产生无效比例")
    report = {"method": "训练集同语义Sharpa FK与闭链指腹腕相对向量逐任务点一维最小二乘",
              "training_frames": len(train), "unique_training_frames": len(unique),
              "reference_extra_axes_rad": {"thumb_MCP_AA": 0., "pinky_CMC": 0.},
              "reference_thumb_cmc_aa_sign": -1 if getattr(fk, "side", "right") == "left" else 1,
              "reference_note": "参考姿态只用于几何拟合；额外轴运行时是否锁定由Sharpa优化器对照决定",
              "segment_scaling": {f: scales[i].tolist() for i, f in enumerate(FINGERS)}}
    for split in ("train", "validation"):
        residual = []
        for frame in frames:
            if frame["split"] != split:
                continue
            p = np.asarray(frame["points_m"])
            residual.append(np.linalg.norm((p - p[0])[TASK_INDICES] * scales[..., None] -
                                           fk.reference(frame)[0][TASK_INDICES], axis=-1) * 1000)
        errors = np.asarray(residual)
        report[split] = {"frames": len(errors), "reference_residual_rms_mm": float(np.sqrt(np.mean(errors ** 2))),
                         "reference_residual_per_task_rms_mm": np.sqrt(np.mean(errors ** 2, axis=0)).tolist()}
    return scales, report


def run_probe(probe: Path, side: str, config: Path, frames: list[dict]) -> list[dict]:
    requests, previous_sequence = [], None
    for frame in frames:
        request = {"points_m": frame["points_m"]}
        if frame["sequence"] != previous_sequence:
            request.update({"reset": True, "seed_rad": [0.] * 22})
        requests.append(json.dumps(request) + "\n")
        previous_sequence = frame["sequence"]
    payload = "".join(requests)
    result = subprocess.run([str(probe), "--hand", side, "--config", str(config)],
                            input=payload, capture_output=True, text=True, check=True)
    output = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if len(output) != len(frames):
        raise ValueError(f"native probe输出帧数{len(output)}与输入{len(frames)}不一致：{result.stderr[-300:]}")
    return output


def open_fist_cycles(side: str, fixture_path: Path) -> tuple[list[dict], dict]:
    """只读现有闭链模型和限位，补完整伸直/半握/握拳/回伸离线验收数据。"""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from generate_retarget_fixtures import FixtureGenerator

    source = json.loads(fixture_path.read_text())
    generator = FixtureGenerator(side, Path(source["mapping_path"]))
    frames = []
    for split in ("train", "validation"):
        full = .9 * generator.bounds[:, 1]
        full[0], full[1] = .9, 0.
        if split == "validation":
            full[0] = .95
            full[2:4] *= [.96, 1.02]
            full[4::2] = generator.bounds[4::2, 1] * [.88, .92, .86, .90]
            full[5::2] = generator.bounds[5::2, 1] * [.92, .88, .90, .86]
        phases = [("open_hold", np.zeros(60)), ("close_to_half", np.linspace(0., .5, 41)[1:]),
                  ("half_hold", np.full(20, .5)), ("close_to_fist", np.linspace(.5, 1., 41)[1:]),
                  ("fist_hold", np.ones(60)), ("open_to_half", np.linspace(1., .5, 41)[1:]),
                  ("return_half_hold", np.full(20, .5)), ("open_to_zero", np.linspace(.5, 0., 41)[1:]),
                  ("return_open_hold", np.zeros(60))]
        seed = None
        for phase, progress_values in phases:
            for progress in progress_values:
                motors = full * progress
                if split == "validation":
                    motors[1] = (.04 if side == "right" else -.04) * np.sin(np.pi * progress)
                frame, seed = generator.frame(motors, f"{split}_open_fist_cycle", phase, progress,
                                               split=split, seed=seed)
                frames.append(frame)
    return frames, {"method": "原闭链FK、原电机软限位只读生成；训练与验证采用不同握拳终点及拇指侧摆路径",
                    "source_fixture_sha256": sha256(fixture_path), "frames": len(frames),
                    "mapping_sha256": sha256(generator.mapping_path),
                    "targets_sha256": sha256(generator.targets_path),
                    "maximum_closure_gap_m": max(f["max_closure_gap_m"] for f in frames),
                    "shared_zero_note": "两组共享已知CAD零位作为硬验收；非零验证轨迹不参与拟合"}


def open_fist_metrics(side: str, frames: list[dict], outputs: list[dict]) -> dict:
    fk = SharpaFK(side)
    result = {}
    # 拇指CMC定义整个拇指朝向，不把不同机械手的基座对准角误当成指骨卷曲。
    finger_axes = ([2, 4], [5, 7, 8], [9, 11, 12], [13, 15, 16], [18, 20, 21])
    straightness_axes = list(range(2, 22))
    for split in ("train", "validation"):
        indices = [i for i, frame in enumerate(frames) if frame["split"] == split]
        if not indices:
            continue
        q = np.array([outputs[i]["q_rad"] for i in indices])
        group = [frames[i] for i in indices]
        phase_metrics = {}
        for phase in ("open_hold", "half_hold", "fist_hold", "return_open_hold"):
            positions = [i for i, f in enumerate(group) if f["phase"] == phase][-20:]
            values = q[positions]
            reference = np.array([fk.reference(group[i])[1] for i in positions])
            phase_metrics[phase] = {"max_abs_joint_deg": float(np.max(np.abs(values)) * 180 / np.pi),
                "max_abs_straightness_joint_deg": float(np.max(np.abs(values[:, straightness_axes])) * 180 / np.pi),
                "mean_joint_rad": values.mean(axis=0).tolist(),
                "reference_mean_joint_rad": reference.mean(axis=0).tolist(),
                "finger_curl_sum_rad": {finger: float(values[:, axes].sum(axis=1).mean())
                                        for finger, axes in zip(FINGERS, finger_axes)},
                "reference_finger_curl_sum_rad": {finger: float(reference[:, axes].sum(axis=1).mean())
                                                  for finger, axes in zip(FINGERS, finger_axes)}}
        differences = np.abs(np.diff(q, axis=0))
        worst_frame, worst_axis = np.unravel_index(np.argmax(differences), differences.shape)
        open_error = max(phase_metrics[p]["max_abs_straightness_joint_deg"] for p in ("open_hold", "return_open_hold"))
        result[split] = {"frames": len(indices), "phases": phase_metrics,
            "straightness_joint_names": [JOINT_SUFFIXES[i] for i in straightness_axes],
            "open_max_straightness_deg": open_error, "open_within_2deg": open_error <= 2.,
            "max_adjacent_joint_step_rad": float(differences.max()),
            "worst_step": {"joint": JOINT_SUFFIXES[worst_axis], "frame": int(worst_frame + 1),
                           "phase": group[worst_frame + 1]["phase"]},
            "continuity_below_015rad": bool(differences.max() < .15)}
    return result


def measure_static_pad_depth(side: str, config: dict, zero_points: np.ndarray) -> tuple[dict, dict]:
    """只重测已有X尺度和15比例；Y/Z及腕参考保持原值，不添加运行字段。"""
    result = json.loads(json.dumps(config))
    axis = np.asarray(result["geometry"]["task_axis_scale"], dtype=float)
    offset = np.asarray(result["geometry"]["wrist_offset_m"], dtype=float)
    physical = SharpaFK(side).physical_points(np.zeros(22))[TASK_INDICES]
    glove = np.asarray(zero_points)[TASK_INDICES]
    trace = []
    for _ in range(4):
        robot = (physical - offset) / axis
        scales = np.sum(glove * robot, axis=-1) / np.sum(glove * glove, axis=-1)
        gx = glove[1:, :, 0] * scales[1:]
        rx = physical[1:, :, 0] - offset[0]
        axis[0] = np.sum(gx * rx) / np.sum(gx * gx)
        trace.append(float(axis[0]))
    robot = (physical - offset) / axis
    scales = np.sum(glove * robot, axis=-1) / np.sum(glove * glove, axis=-1)
    geometric_scales = scales.copy()
    scales[[0, 4]] *= 1.01
    result["geometry"]["task_axis_scale"] = axis.tolist()
    result["retarget"]["segment_scaling"] = {finger: scales[i].tolist() for i, finger in enumerate(FINGERS)}
    result["retarget"]["fixed_joints"][f"{side}_thumb_MCP_AA"] = 0.
    result["retarget"]["fixed_joints"][f"{side}_pinky_CMC"] = 0.
    return result, {"method": "真实CAD零位15个腕相对任务向量LS，与四指实际pad_x-offset对s*glove_x的LS交替4轮",
        "x_scale_iterations": trace, "unchanged_geometry": {"y_z_scale": axis[1:].tolist(), "wrist_offset_m": offset.tolist()},
        "geometric_segment_scaling": {finger: geometric_scales[i].tolist() for i, finger in enumerate(FINGERS)},
        "validated_increment": {"thumb": 1.01, "pinky": 1.01, "reason": "最小已试验1%增量，使伸直内部关节<2度；不改变基准或目标公式"},
        "physical_zero_pad_xyz_m": physical.tolist(), "glove_zero_pad_xyz_m": glove.tolist(),
        "open_target_radial_ratio": (np.linalg.norm(glove * scales[:, :, None], axis=-1) /
                                      np.linalg.norm(robot, axis=-1)).tolist()}


def cycle_target_geometry(side: str, config: dict, frames: list[dict]) -> dict:
    source = np.asarray([frame["points_m"] for frame in frames])[:, TASK_INDICES]
    scales = np.asarray([config["retarget"]["segment_scaling"][finger] for finger in FINGERS])
    edges = np.diff(source, axis=2)
    target_edges = np.diff(source * scales[None, :, :, None], axis=2)
    products = np.sum(edges * target_edges, axis=-1)
    fk = SharpaFK(side, config["geometry"]["task_axis_scale"], config["geometry"]["wrist_offset_m"])
    reference = np.asarray([fk.reference(frame)[0] for frame in frames])[:, TASK_INDICES]
    reference_lengths = np.linalg.norm(np.diff(reference, axis=2), axis=-1)
    return {"minimum_direction_dot_m2": float(products.min()), "all_segments_preserve_direction": bool(np.all(products > 0)),
            "minimum_target_to_source_segment_length_ratio": float(np.min(np.linalg.norm(target_edges, axis=-1) / np.linalg.norm(edges, axis=-1))),
            "maximum_target_to_source_segment_length_ratio": float(np.max(np.linalg.norm(target_edges, axis=-1) / np.linalg.norm(edges, axis=-1))),
            "minimum_target_to_robot_segment_length_ratio": float(np.min(np.linalg.norm(target_edges, axis=-1) / reference_lengths)),
            "maximum_target_to_robot_segment_length_ratio": float(np.max(np.linalg.norm(target_edges, axis=-1) / reference_lengths))}


def run_static_depth_calibration(args) -> None:
    report_path = CONFIG_DIR / "fit_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for side in (("right", "left") if args.hand == "both" else (args.hand,)):
        source_path = (Path(str(args.configuration_source).replace("{side}", side))
                       if args.configuration_source else CONFIG_DIR / f"{side}.yaml")
        before = yaml.safe_load(source_path.read_text())
        source_configuration_sha256 = sha256(source_path)
        fixture_path = Path(str(args.fixtures).replace("{side}", side))
        fixtures = fixture_frames(fixture_path)
        zero = min((f for f in fixtures if f["split"] == "train"), key=lambda f: np.linalg.norm(f["joint_rad"]))
        if np.max(np.abs(zero["joint_rad"])) > 1e-10:
            raise ValueError("静态几何标定必须有真实CAD零位")
        config, calibration = measure_static_pad_depth(side, before, np.asarray(zero["points_m"]))
        config["retarget"]["w_hyper"] = 20. if args.w_hyper is None else args.w_hyper
        config["retarget"]["w_couple"] = 1. if args.w_couple is None else args.w_couple
        if args.scaling is not None:
            config["retarget"]["scaling"] = args.scaling
        config["optimizer"]["urdf_path"] = str(pad_urdf(side))
        frames, generation = open_fist_cycles(side, fixture_path)
        with tempfile.TemporaryDirectory(prefix="sharpa-static-depth-") as directory:
            trial = Path(directory) / f"{side}.yaml"
            trial.write_text(yaml.safe_dump(config, sort_keys=False))
            output = run_probe(args.probe, side, trial, frames)
        metrics = open_fist_metrics(side, frames, output)
        target_geometry = cycle_target_geometry(side, config, frames)
        for split, values in metrics.items():
            if not values["open_within_2deg"] or not values["continuity_below_015rad"]:
                raise ValueError(f"{side}/{split}伸直或连续性未通过，禁止写生产配置：{values['open_max_straightness_deg']:.3f}度/{values['max_adjacent_joint_step_rad']:.4f}rad")
            phase = values["phases"]["fist_hold"]
            for finger in FINGERS:
                if phase["finger_curl_sum_rad"][finger] < .85 * phase["reference_finger_curl_sum_rad"][finger]:
                    raise ValueError(f"{side}/{split}/{finger}握拳幅度不足85%")
        if not target_geometry["all_segments_preserve_direction"]:
            raise ValueError("目标骨段发生倒挂，禁止写生产配置")
        final = CONFIG_DIR / f"{side}.yaml"
        config["optimizer"]["urdf_path"] = f"../../../assets/sharpa-hand/wave_01/{side}_sharpa_wave/{side}_sharpa_wave_pad.urdf"
        comments = ("# scripts/fit_sharpa_retarget.py --measure-pad-depth 生成；只使用现有Wuji式参数。\n"
                    "# X尺度来自实际指腹几何LS；Y/Z及腕参考不变；thumb/pinky比例仅1%微调。\n"
                    f"# w_hyper={config['retarget']['w_hyper']:g}、w_couple={config['retarget']['w_couple']:g}；完整伸直/握拳往返验证见fit_report.json。\n")
        final.write_text(comments + yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        output = run_probe(args.probe, side, final, frames)
        output_path = Path(str(args.native_output).replace("{side}", side))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        input_path = output_path.with_name(f"open_cycle_inputs_{side}.json")
        input_path.write_text(json.dumps({"schema_version": 1, "hand": side, "input_frame": "canonical", "input_units": "m",
                                         "generation": generation, "frames": frames}, ensure_ascii=False) + "\n")
        output_path.write_text(json.dumps({"hand": side, "frames": output, "config_sha256": sha256(final),
            "fixture_sha256": sha256(input_path), "pad_urdf_sha256": sha256(pad_urdf(side)),
            "probe_binary_sha256": sha256(args.probe)}, ensure_ascii=False) + "\n")
        old = report.get(side, {})
        old_hash = old.get("final_configuration", {}).get("sha256", "unknown")
        report.setdefault("superseded_profiles", {}).setdefault(side, {})[old_hash] = {
            "status": "旧配置已过期，旧对指及动态报告不能作为当前配置验证", "profile": old}
        pinch_output = run_probe(args.probe, side, final, fixtures)
        scales = np.asarray([config["retarget"]["segment_scaling"][f] for f in FINGERS])
        report[side] = {"status": "当前伸直/握拳几何配置", "static_pad_depth_fit": calibration,
            "source_configuration_sha256": source_configuration_sha256, "fit": {"fixture_sha256": sha256(fixture_path)},
            "final_configuration": {"sha256": sha256(final), "geometry": config["geometry"], "retarget": config["retarget"]},
            "open_cycle": open_fist_metrics(side, frames, output), "target_geometry": target_geometry,
            "pinch_regression": output_metrics(fixtures, pinch_output, scales),
            "provenance": {"input_path": str(input_path), "input_sha256": sha256(input_path),
                           "native_output_path": str(output_path), "native_output_sha256": sha256(output_path),
                           "probe_binary_sha256": sha256(args.probe), "pad_urdf_sha256": sha256(pad_urdf(side))},
            "limitations": "本轮优先伸直及完整握拳连续性；固定额外关节和几何比例改变会影响对指，小指对指残差须单独披露"}
        report["generator_sha256"] = sha256(Path(__file__))
        report["verification_scope"] = {"cad_mesh_geometry": True, "closed_chain_geometric_fit": True,
            "sharpa_native_solver": True, "mujoco_dynamic_validation": False, "real_hardware_calibration": False}
        report.pop("external_dynamic_validation", None)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"hand": side, "config_sha256": sha256(final), "output": str(output_path)}, ensure_ascii=False))


def output_metrics(frames: list[dict], outputs: list[dict], scales: np.ndarray) -> dict:
    result = {}
    for split in ("train", "validation"):
        grouped = {}
        for frame, output in zip(frames, outputs):
            if frame["split"] != split:
                continue
            p, robot = np.asarray(frame["points_m"]), np.asarray(output["points_m"])
            physical = np.asarray(output.get("physical_points_m", output["points_m"]))
            error = np.linalg.norm(robot[TASK_INDICES] - (p - p[0])[TASK_INDICES] * scales[..., None], axis=-1)
            group = grouped.setdefault(frame["sequence"], {"errors": [], "gaps": [], "source_gaps": [], "q": []})
            group["errors"].append(error)
            group["q"].append(output["q_rad"])
            target = frame.get("target_finger")
            if target in FINGERS[1:]:
                index = 4 + 4 * FINGERS.index(target)
                group["gaps"].append(float(np.linalg.norm(physical[4] - physical[index])))
                group["source_gaps"].append(float(np.linalg.norm(p[4] - p[index])))
        result[split] = {}
        for sequence, data in grouped.items():
            entry = {"frames": len(data["errors"]),
                     "scaled_task_rms_mm": float(np.sqrt(np.mean(np.array(data["errors"]) ** 2)) * 1000),
                     "extra_axes_max_abs_rad": np.max(np.abs(np.array(data["q"])[:, [3, 17]]), axis=0).tolist(),
                     "max_adjacent_q_step_rad": (float(np.max(np.abs(np.diff(data["q"], axis=0))))
                                                 if len(data["q"]) > 1 else 0.)}
            if data["gaps"]:
                nearest = int(np.argmin(data["source_gaps"]))
                entry.update({"source_min_pinch_mm": 1000 * min(data["source_gaps"]),
                              "robot_min_pinch_mm": 1000 * min(data["gaps"]),
                              "robot_at_source_closest_mm": 1000 * data["gaps"][nearest]})
            result[split][sequence] = entry
    return result


def refine_native(probe: Path, side: str, frames: list[dict], scales: np.ndarray,
                  axis_scale: np.ndarray, wrist_offset: np.ndarray,
                  thumb_aa_fixed: float | None = None, norm_delta: float = .04,
                  w_hyper: float = 1., w_couple: float = .1) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, dict]:
    """有限坐标搜索：只用训练集、只变几何比例，不改任何运行时损失权重。"""
    train = [frame for frame in frames if frame["split"] == "train"]
    selected, targets = [], []
    for finger in FINGERS[1:]:
        index = 4 + 4 * FINGERS.index(finger)
        group = [frame for frame in train if frame.get("target_finger") == finger]
        for distance in (0., .01, .03):
            frame = min(group, key=lambda frame: abs(np.linalg.norm(np.asarray(frame["points_m"])[4] -
                                                                    np.asarray(frame["points_m"])[index]) - distance))
            if not any(frame is old for old in selected):
                selected.append(frame)
                targets.append(index)
    anchors = [train[0]]
    grasp = [frame for frame in train if frame["sequence"] == "grasp"]
    if grasp:
        anchors.extend([grasp[len(grasp) // 4], grasp[len(grasp) // 2]])
    # 保留每条真实接近/保持/回开序列，按ControlSession启动方式从零角seed热启动。
    fixture = train
    selected_indices = [next(i for i, frame in enumerate(train) if frame is chosen) for chosen in selected]
    anchor_indices = [next(i for i, frame in enumerate(train) if frame is chosen) for chosen in anchors]
    # 腕偏移以厘米参数化，使有限搜索步长对应毫米而非数十厘米。
    initial = np.r_[1., scales.ravel(), axis_scale, wrist_offset * 100]
    lower = np.r_[.65, np.maximum(.55, scales.ravel() - .5), np.maximum(.4, axis_scale - .3), wrist_offset * 100 - 1.5]
    upper = np.r_[1.7, scales.ravel() + .5, axis_scale + .3, wrist_offset * 100 + 1.5]
    zero = min(train, key=lambda frame: np.linalg.norm(frame["joint_rad"]))
    palm_source = np.asarray(zero["points_m"])[[5, 9, 13, 17]]
    palm_robot = SharpaFK(side).reference(zero)[0][[5, 9, 13, 17]]
    initial_palm_errors = np.linalg.norm(palm_source * axis_scale + wrist_offset - palm_robot, axis=1) * 1000
    palm_max_limit = float(initial_palm_errors.max() + 3.)
    palm_rms_limit = float(np.sqrt(np.mean(initial_palm_errors ** 2)) + 2.5)
    evaluations = 0
    with tempfile.TemporaryDirectory(prefix="sharpa-native-fit-") as directory:
        config = Path(directory) / f"{side}.yaml"

        def predict(values):
            nonlocal evaluations
            evaluations += 1
            write_config(side, values[1:16].reshape(5, 3), path=config, pinch_scale=values[0],
                         axis_scale=values[16:19], wrist_offset=values[19:22] / 100,
                         thumb_aa_fixed=thumb_aa_fixed, norm_delta=norm_delta, w_hyper=w_hyper, w_couple=w_couple)
            output = run_probe(probe, side, config, fixture)
            if not all("physical_points_m" in frame for frame in output):
                raise ValueError("probe需要支持physical_points_m，禁止把任务域距离冒充物理指腹间距")
            return np.array([o["physical_points_m"] for o in output])

        initial_output = predict(initial)
        source_distance = np.array([np.linalg.norm(axis_scale * (np.asarray(frame["points_m"])[4] -
                                                   np.asarray(frame["points_m"])[index]))
                                    for frame, index in zip(selected, targets)])

        def score(values):
            palm_errors = 1000 * np.linalg.norm(palm_source * values[16:19] + values[19:22] / 100 - palm_robot, axis=1)
            palm_rms = float(np.sqrt(np.mean(palm_errors ** 2)))
            if palm_errors.max() > palm_max_limit or palm_rms > palm_rms_limit:
                return 1e6 + float(np.sum(palm_errors ** 2))
            points = predict(values)
            distance = np.array([np.linalg.norm(points[i, 4] - points[i, index])
                                 for i, index in zip(selected_indices, targets)])
            # 对指拟合同时约束接近过程；展开/抓握输出作为小权重几何锚点。
            contact = np.mean((1000 * (distance - values[0] * source_distance)) ** 2)
            anchor = np.mean((1000 * (points[anchor_indices][:, TASK_INDICES] -
                                      initial_output[anchor_indices][:, TASK_INDICES])) ** 2)
            regularization = np.mean(((values[1:16] - initial[1:16]) / initial[1:16]) ** 2)
            geometry_prior = np.mean(((values[16:] - initial[16:]) / .3) ** 2)
            return float(contact + .05 * anchor + 10. * regularization + .5 * geometry_prior + .2 * palm_rms ** 2)

        values = initial.copy()
        initial_score = best = score(values)
        trace = []
        for step in (.2, .1, .05, .025, .0125):
            for _ in range(2):
                improved = False
                for index in range(22):
                    candidate_best, candidate_score = values, best
                    for direction in (-1., 1.):
                        candidate = values.copy()
                        candidate[index] = np.clip(candidate[index] + direction * step, lower[index], upper[index])
                        if 16 <= index < 19:
                            dimension = index - 16
                            candidate[19 + dimension] = np.clip(
                                candidate[19 + dimension] - 100 * np.mean(palm_source[:, dimension]) *
                                (candidate[index] - values[index]), lower[19 + dimension], upper[19 + dimension])
                        if candidate[index] == values[index]:
                            continue
                        loss = score(candidate)
                        if loss + 1e-8 < candidate_score:
                            candidate_best, candidate_score = candidate, loss
                    if candidate_score < best:
                        values, best, improved = candidate_best, candidate_score, True
                trace.append({"step": step, "loss": best})
                if not improved:
                    break
    return values[1:16].reshape(5, 3), float(values[0]), values[16:19], values[19:22] / 100, {
        "method": "只用训练集的native有限坐标搜索：16个几何比例+3掌尺度+3腕偏移，禁止修改运行时损失权重",
        "training_contact_frames": len(selected), "training_anchor_frames": len(anchors),
        "initialization": "每sequence第一帧reset+seed22轴零位，后续连续热启动，与ControlSession一致",
        "native_evaluations": evaluations, "initial_score": initial_score, "final_score": best,
        "objective": "接近/闭合间距误差mm² + .05×展开/抓握几何偏离mm² + 10×比例相对变化² + .5×掌几何CAD先验偏离²",
        "palm_constraint": {"initial_per_mcp_mm": initial_palm_errors.tolist(),
                            "max_per_mcp_mm": palm_max_limit, "max_mcp_rms_mm": palm_rms_limit,
                            "final_per_mcp_mm": (1000 * np.linalg.norm(palm_source * values[16:19] +
                                                                      values[19:22] / 100 - palm_robot, axis=1)).tolist(),
                            "coupled_update": "改变轴尺度时联动腕偏移，保持四MCP质心，特别避免Dz/Oz共线退化"},
        "parameter_lower": lower.tolist(), "parameter_upper": upper.tolist(), "trace": trace,
        "pinch_target_scaling": float(values[0]),
        "segment_scaling": {finger: values[1:16].reshape(5, 3)[i].tolist() for i, finger in enumerate(FINGERS)},
        "task_axis_scale": values[16:19].tolist(), "wrist_offset_m": (values[19:22] / 100).tolist(),
    }


def geometric_reachability(side: str, frames: list[dict], outputs: list[dict]) -> dict:
    """仅诊断本体可达性：四指AA仍锁零，忽略碰撞，绝不用于运行时重定向。"""
    from scipy.optimize import least_squares

    fk = SharpaFK(side)
    lower = fk.model.lowerPositionLimit[fk.indices].copy()
    upper = fk.model.upperPositionLimit[fk.indices].copy()
    scene = ET.parse(SHARPA / f"{side}.xml").getroot()
    joints = {e.attrib["name"]: e for e in scene.find("worldbody").iter("joint")}
    actuators = {e.attrib["joint"]: e for e in scene.find("actuator")}
    for i, suffix in enumerate(JOINT_SUFFIXES):
        name = f"{side}_{suffix}"
        interval = np.fromstring(joints[name].attrib["range"], sep=" ")
        actuator = actuators[name]
        if "ctrlrange" in actuator.attrib:
            ctrl = np.fromstring(actuator.attrib["ctrlrange"], sep=" ")
        else:
            half = .5 * np.diff(interval)[0] * float(actuator.attrib["inheritrange"])
            ctrl = np.mean(interval) + np.array([-half, half])
        lower[i], upper[i] = max(lower[i], interval[0], ctrl[0]), min(upper[i], interval[1], ctrl[1])
    result = {"scope": "URDF/MJCF/执行器限位交集，四指MCP_AA锁零，额外两轴自由；未加碰撞约束",
              "interpretation": "只证实存在的接触构型；不是遥操作输出，不是动态或硬件验证", "fingers": {}}
    for fi, finger in enumerate(FINGERS[1:], 1):
        target = 4 + 4 * fi
        candidates = [i for i, f in enumerate(frames) if f["split"] == "train" and f.get("target_finger") == finger]
        index = min(candidates, key=lambda i: np.linalg.norm(np.array(frames[i]["points_m"])[4] -
                                                            np.array(frames[i]["points_m"])[target]))
        base = np.asarray(outputs[index]["q_rad"]).copy()
        base[[6, 10, 14, 19]] = 0.
        active = list(range(5)) + ([17, 18, 20, 21] if fi == 4 else [4 * fi + 1, 4 * fi + 3, 4 * fi + 4])

        def residual(x):
            q = base.copy()
            q[active] = x
            points = fk.points(q)
            return np.r_[1000 * (points[4] - points[target]), 1e-5 * (x - base[active])]

        best = None
        for seed in (base[active], (lower[active] + upper[active]) / 2):
            fit = least_squares(residual, np.clip(seed, lower[active], upper[active]),
                                bounds=(lower[active], upper[active]), max_nfev=300,
                                ftol=1e-12, xtol=1e-12, gtol=1e-12)
            gap = float(np.linalg.norm(residual(fit.x)[:3]))
            if best is None or gap < best["pad_gap_mm"]:
                q = base.copy()
                q[active] = fit.x
                best = {"pad_gap_mm": gap, "q_rad": q.tolist(), "evaluations": fit.nfev}
        result["fingers"][finger] = best
    return result


def select_thumb_axis(probe: Path, side: str, frames: list[dict], scales, pinch_scale, axis_scale, wrist_offset,
                      norm_delta: float = .04, w_hyper: float = 1., w_couple: float = .1):
    """用训练综合误差选固定冗余轴目标，避免自由轴在两端局部解间跳变。"""
    candidates = {}
    with tempfile.TemporaryDirectory(prefix="sharpa-thumb-axis-") as directory:
        path = Path(directory) / f"{side}.yaml"
        write_config(side, scales, path=path, pinch_scale=pinch_scale,
                     axis_scale=axis_scale, wrist_offset=wrist_offset, norm_delta=norm_delta,
                     w_hyper=w_hyper, w_couple=w_couple)
        free = run_probe(probe, side, path, frames)
        open_index = next(i for i, f in enumerate(frames) if f["split"] == "train" and f["sequence"] == "open")
        for value in (-.30, -.25):
            write_config(side, scales, path=path, pinch_scale=pinch_scale, axis_scale=axis_scale,
                         wrist_offset=wrist_offset, thumb_aa_fixed=value, norm_delta=norm_delta,
                         w_hyper=w_hyper, w_couple=w_couple)
            output = run_probe(probe, side, path, frames)
            metrics = output_metrics(frames, output, scales)
            gap_squared = [v["robot_at_source_closest_mm"] ** 2 for v in metrics["train"].values()
                           if "robot_at_source_closest_mm" in v]
            anchor = 1000 * (np.array(output[open_index]["physical_points_m"])[TASK_INDICES] -
                             np.array(free[open_index]["physical_points_m"])[TASK_INDICES])
            candidates[str(value)] = {"training_pinch_rms_mm": float(np.sqrt(np.mean(gap_squared))),
                "open_anchor_deviation_rms_mm": float(np.sqrt(np.mean(anchor ** 2))),
                "training_score": float(np.mean(gap_squared) + .05 * np.mean(anchor ** 2)),
                "physical_pinch_mm": {split: {name: item["robot_at_source_closest_mm"]
                    for name, item in groups.items() if "robot_at_source_closest_mm" in item}
                    for split, groups in metrics.items()}}
    selected = min(candidates, key=lambda key: candidates[key]["training_score"])
    return float(selected), {"method": "只用train的四组对指RMS²+.05×open偏离RMS²；validation不参与选择",
        "candidates": candidates, "selected_rad": float(selected), "urdf_limits_rad": [-.3491, .3491],
        "lower_limit_margin_rad": float(selected) + .3491,
        "interpretation": "模型固定冗余关节目标；不是实物标定或安装角测量；pinky_CMC保留自由"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hand", choices=("left", "right", "both"), default="both")
    parser.add_argument("--fixtures", type=Path, help="JSON路径，可含{side}；需要train/validation及真实dependent_rad")
    parser.add_argument("--probe", type=Path, help="Sharpa私有native批量probe；不连接硬件")
    parser.add_argument("--baseline", type=Path, help="旧Sharpa输出JSON；使用相同新指腹FK公平对比")
    parser.add_argument("--native-output", type=Path, help="保存native输出明细供仿真复核，文件名可含{side}")
    parser.add_argument("--refine", action="store_true", help="在训练集上通过native有限搜索修正几何比例和腕参考")
    parser.add_argument("--thumb-aa-fixed", type=float, help="按训练几何/连续性对照选定的拇指额外轴固定角；小指CMC仍自由")
    parser.add_argument("--norm-delta", type=float, default=.04, help="Sharpa时序先验系数；默认.04，最终连续性验证选择.4")
    parser.add_argument("--w-hyper", type=float, help="Wuji既有反伸惩罚参数；keep-fitted默认保留当前配置")
    parser.add_argument("--w-couple", type=float, help="Wuji既有PIP/DIP耦合参数；keep-fitted默认保留当前配置")
    parser.add_argument("--keep-fitted-geometry", action="store_true", help="复用现有YAML几何/比例，仅重验轴策略，不重新拟合")
    parser.add_argument("--select-thumb-aa", action="store_true", help="只用train比较-.30/-.25rad固定冗余轴目标")
    parser.add_argument("--geometry-only", action="store_true", help="只生成指腹URDF和几何报告，保留已拟合比例")
    parser.add_argument("--evaluate-open-cycle", action="store_true", help="只读当前配置，离线验收完整伸直/半握/握拳/回伸")
    parser.add_argument("--measure-pad-depth", action="store_true", help="实际零位指腹几何标定已有X尺度和15比例，并硬验收完整往返")
    parser.add_argument("--configuration-source", type=Path, help="几何标定来源配置，可含{side}；缺省读取当前配置")
    parser.add_argument("--scaling", type=float, help="已有retarget.scaling参数；静态标定模式用于记录训练验证选出的值")
    args = parser.parse_args()
    if args.keep_fitted_geometry and args.refine:
        parser.error("--keep-fitted-geometry与--refine互斥")
    if args.select_thumb_aa and (not args.probe or args.thumb_aa_fixed is not None):
        parser.error("--select-thumb-aa需要--probe且不同时指定--thumb-aa-fixed")
    if args.measure_pad_depth:
        if not args.probe or not args.fixtures or not args.native_output:
            parser.error("--measure-pad-depth需要--probe、--fixtures和--native-output")
        run_static_depth_calibration(args)
        return
    if args.evaluate_open_cycle:
        if not args.probe or not args.fixtures or not args.native_output:
            parser.error("--evaluate-open-cycle需要--probe、--fixtures和--native-output")
        for side in (("right", "left") if args.hand == "both" else (args.hand,)):
            fixture_path = Path(str(args.fixtures).replace("{side}", side))
            frames, generation = open_fist_cycles(side, fixture_path)
            config = CONFIG_DIR / f"{side}.yaml"
            outputs = run_probe(args.probe, side, config, frames)
            path = Path(str(args.native_output).replace("{side}", side))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"hand": side, "config_sha256": sha256(config),
                "probe_binary_sha256": sha256(args.probe), "generation": generation,
                "metrics": open_fist_metrics(side, frames, outputs), "input_frames": frames,
                "frames": outputs}, ensure_ascii=False) + "\n")
            print(json.dumps({"hand": side, "output": str(path)}, ensure_ascii=False))
        return
    report_path = CONFIG_DIR / "fit_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report.update({"schema_version": 1, "units": "m_rad", "generator_sha256": sha256(Path(__file__)),
                   "input_coordinates": "Motorhand闭链21点canonical米",
                   "sharpa_canonical_rotation": np.eye(3).tolist()})
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for side in (("right", "left") if args.hand == "both" else (args.hand,)):
        side_report = report.setdefault(side, {})
        side_report.pop("diagnostic_pinch_weight_max", None)
        side_report["geometry"] = generate_geometry(side)
        if args.geometry_only:
            continue
        if not args.fixtures:
            parser.error("多姿态比例拟合需要--fixtures；单独生成几何请用--geometry-only")
        fixture_path = Path(str(args.fixtures).replace("{side}", side))
        frames = fixture_frames(fixture_path)
        pinch_scale = 1.
        w_hyper = 1. if args.w_hyper is None else args.w_hyper
        w_couple = .1 if args.w_couple is None else args.w_couple
        if args.keep_fitted_geometry:
            old = yaml.safe_load((CONFIG_DIR / f"{side}.yaml").read_text())
            axis_scale = np.asarray(old["geometry"]["task_axis_scale"])
            wrist_offset = np.asarray(old["geometry"]["wrist_offset_m"])
            scales = np.asarray([old["retarget"]["segment_scaling"][f] for f in FINGERS])
            pinch_scale = float(old["retarget"]["scaling"])
            if args.w_hyper is None:
                w_hyper = float(old["retarget"]["w_hyper"])
            if args.w_couple is None:
                w_couple = float(old["retarget"]["w_couple"])
            if side_report["fit"]["fixture_sha256"] != sha256(fixture_path):
                raise ValueError("复用几何配置时fixture必须与原拟合数据一致")
        else:
            axis_scale, wrist_offset, geometry_fit = fit_task_geometry(frames, side)
            side_report["task_geometry_fit"] = geometry_fit
            scales, fit = fit_scales(frames, SharpaFK(side, axis_scale, wrist_offset))
            side_report["fit"] = {"fixture_sha256": sha256(fixture_path), **fit}
        for key in ("native_free_extra_axes", "native_locked_extra_axes", "native_final", "extra_axes_decision",
                    "probe_binary_sha256", "native_initialization"):
            side_report.pop(key, None)
        if not args.keep_fitted_geometry:
            side_report.pop("native_geometry_refinement", None)
            side_report.pop("baseline_same_pad_geometry", None)
        if args.refine:
            if not args.probe:
                parser.error("--refine需要--probe")
            scales, pinch_scale, axis_scale, wrist_offset, refinement = refine_native(
                args.probe, side, frames, scales, axis_scale, wrist_offset, args.thumb_aa_fixed, args.norm_delta,
                w_hyper, w_couple)
            side_report["native_geometry_refinement"] = refinement
        thumb_aa_fixed = args.thumb_aa_fixed
        if args.select_thumb_aa:
            thumb_aa_fixed, selection = select_thumb_axis(args.probe, side, frames, scales, pinch_scale, axis_scale,
                                                        wrist_offset, args.norm_delta, w_hyper, w_couple)
            side_report["thumb_axis_selection"] = selection
        config = write_config(side, scales, pinch_scale=pinch_scale, axis_scale=axis_scale, wrist_offset=wrist_offset,
                              thumb_aa_fixed=thumb_aa_fixed, norm_delta=args.norm_delta, w_hyper=w_hyper, w_couple=w_couple)
        side_report["final_configuration"] = {"sha256": sha256(config), "pinch_target_scaling": pinch_scale,
                                               "task_axis_scale": axis_scale.tolist(), "wrist_offset_m": wrist_offset.tolist(),
                                               "thumb_mcp_aa_fixed_rad": thumb_aa_fixed,
                                               "norm_delta": args.norm_delta,
                                               "w_hyper": w_hyper, "w_couple": w_couple,
                                               "segment_scaling": {f: scales[i].tolist() for i, f in enumerate(FINGERS)}}
        if args.baseline:
            baseline = json.loads(Path(str(args.baseline).replace("{side}", side)).read_text())
            if baseline.get("fixture_sha256") != sha256(fixture_path):
                raise ValueError("旧Sharpa输出的fixture哈希不匹配，禁止比较不同输入")
            old = baseline["frames"]
            if len(old) != len(frames):
                raise ValueError("旧Sharpa输出与闭链fixture帧数不一致")
            fk = SharpaFK(side, axis_scale, wrist_offset)
            comparable = [{"q_rad": f["q_rad"], "points_m": fk.points(f["q_rad"]).tolist(),
                           "physical_points_m": fk.physical_points(f["q_rad"]).tolist()} for f in old]
            side_report["baseline_same_pad_geometry"] = output_metrics(frames, comparable, scales)
        if args.probe:
            side_report["probe_binary_sha256"] = sha256(args.probe)
            side_report["native_initialization"] = "每sequence第一帧reset+seed22轴零位，后续连续热启动"
            outputs = run_probe(args.probe, side, config, frames)
            if args.native_output:
                output_path = Path(str(args.native_output).replace("{side}", side))
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps({"hand": side, "frames": outputs,
                    "config_sha256": sha256(config), "fixture_sha256": sha256(fixture_path),
                    "pad_urdf_sha256": sha256(pad_urdf(side)), "probe_binary_sha256": sha256(args.probe)},
                    ensure_ascii=False) + "\n")
            side_report["native_final"] = output_metrics(frames, outputs, scales)
            side_report["geometric_reachability"] = geometric_reachability(side, frames, outputs)
            with tempfile.TemporaryDirectory(prefix="sharpa-fit-") as directory:
                locked = write_config(side, scales, path=Path(directory) / f"{side}.yaml", lock_extra=True,
                                      pinch_scale=pinch_scale, axis_scale=axis_scale, wrist_offset=wrist_offset,
                                      norm_delta=args.norm_delta, w_hyper=w_hyper, w_couple=w_couple)
                locked_output = run_probe(args.probe, side, locked, frames)
                if thumb_aa_fixed is not None:
                    free = write_config(side, scales, path=Path(directory) / f"{side}.yaml", pinch_scale=pinch_scale,
                                        axis_scale=axis_scale, wrist_offset=wrist_offset, norm_delta=args.norm_delta,
                                        w_hyper=w_hyper, w_couple=w_couple)
                    free_output = run_probe(args.probe, side, free, frames)
                else:
                    free_output = outputs
            side_report["native_free_extra_axes"] = output_metrics(frames, free_output, scales)
            side_report["native_locked_extra_axes"] = output_metrics(frames, locked_output, scales)
            rms = {}
            for variant in ("native_final", "native_free_extra_axes", "native_locked_extra_axes"):
                gaps = [entry["robot_at_source_closest_mm"] for entry in side_report[variant]["train"].values()
                        if "robot_at_source_closest_mm" in entry]
                rms[variant] = float(np.sqrt(np.mean(np.square(gaps))))
            side_report["extra_axes_decision"] = {"training_pinch_rms_mm": rms,
                "policy": ("四指MCP_AA固定0；pinky_CMC自由；thumb_MCP_AA=" +
                           ("自由" if thumb_aa_fixed is None else str(thumb_aa_fixed))),
                "reason": "同一训练几何/连续性对照选择；拇指固定目标消除冗余轴换支，小指CMC保持对指所需自由度"}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    report["verification_scope"] = {"cad_mesh_geometry": True,
        "closed_chain_geometric_fit": any("fit" in report.get(s, {}) for s in ("right", "left")),
        "sharpa_native_solver": any("native_free_extra_axes" in report.get(s, {}) for s in ("right", "left")),
        "mujoco_dynamic_validation": False, "real_hardware_calibration": False}
    simulation_path = CONFIG_DIR / "simulation_validation.json"
    report.pop("external_dynamic_validation", None)
    if simulation_path.exists():
        dynamic = json.loads(simulation_path.read_text())
        matched = all(side in dynamic.get("hands", {}) and
                      dynamic["hands"][side]["provenance"]["retarget_config"]["sha256"] ==
                      report.get(side, {}).get("final_configuration", {}).get("sha256") and
                      dynamic["hands"][side]["provenance"]["pad_urdf"]["sha256"] == sha256(pad_urdf(side)) and
                      Path(dynamic["hands"][side]["provenance"]["native_commands"]["path"]).exists() and
                      dynamic["hands"][side]["provenance"]["native_commands"]["sha256"] ==
                      sha256(Path(dynamic["hands"][side]["provenance"]["native_commands"]["path"]))
                      for side in ("right", "left"))
        if matched:
            report["verification_scope"]["mujoco_dynamic_validation"] = True
            report["external_dynamic_validation"] = {"path": simulation_path.name, "sha256": sha256(simulation_path),
                "scope": "独立脚本以真实mj_step验证；接触位置及限位超调见该报告，不等于硬件标定或全部对指达标"}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(report_path), "hands": args.hand}, ensure_ascii=False))


if __name__ == "__main__":
    main()
