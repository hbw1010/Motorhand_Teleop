"""以指腹（而非手指中心轴线）作为 Glove→Wuji 重定向任务点。

手套与 Wuji Hand2 的尺寸、指节厚度不同；按轴线对齐时，两边指腹接触（捏合）对不上。
本脚本在两侧 CAD 零位上，用同一规则从网格求每根手指的指腹点：

* 掌侧方向 n：远节关节正向弯曲时指尖的运动方向，去掉沿指骨的分量（四指须朝掌侧，
  拇指须朝食指指尖，否则报错）。
* 指腹中心：远节网格沿 DIP→指尖方向最远端往回 TIP_FRACTION 处的截面上，掌侧最外点。
* 指腹深度 d（该截面掌侧最外点到轴线的距离）同样用于该指的 PIP、DIP 点：
  task = axis + n·d。手套 PIP/DIP 处只有背侧连杆，没有掌侧外壳，所以不用局部网格。

输出（重复运行幂等，轴线原值始终取 axis_local_xyz_m / 原 URDF）：
* assets/wuji-hand2/body/urdf/{side}_pad.urdf：原 URDF + 每指 3 个固定指腹坐标系
* config/retargeting/official/{side}.yaml：urdf_path 与 link_naming 指向指腹坐标系；
  原轴线配置保存为 {side}_axis.yaml
* segment_scaling 重新标定：以轴线配置在手套零位输出的 Wuji 姿态为参照，逐点最小二乘
* assets/hand_motor_ve33/hand_motor_ve33_{L,R}/config/retarget_targets.json：
  local_xyz_m 换成指腹点，原值保存在 axis_local_xyz_m
"""
from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
TIP_FRACTION = 0.35
SLICE_M, LATERAL_M = 0.0025, 0.004
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
WUJI_FINGER = {"thumb": "thumb", "index": "index_finger", "middle": "middle_finger",
               "ring": "ring_finger", "pinky": "pinky"}


def _stl(path: str) -> np.ndarray:
    raw = Path(path).read_bytes()
    if raw[:5] == b"solid" and b"facet" in raw[:400]:
        return np.array([list(map(float, line.split()[1:4])) for line in raw.decode(errors="ignore").splitlines()
                         if line.strip().startswith("vertex")])
    n = struct.unpack("<I", raw[80:84])[0]
    rec = np.frombuffer(raw[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rec["v"].reshape(-1, 3).astype(float)


class _Hand:
    """Pinocchio model at its neutral (CAD zero) pose with world-space visual meshes per body."""

    def __init__(self, urdf: Path):
        self.model = pin.buildModelFromUrdf(str(urdf))
        geom = pin.buildGeomFromUrdf(self.model, str(urdf), pin.GeometryType.VISUAL, package_dirs=[str(urdf.parent)])
        self.data = self.model.createData()
        q = pin.neutral(self.model)
        pin.framesForwardKinematics(self.model, self.data, q)
        gdata = pin.GeometryData(geom)
        pin.updateGeometryPlacements(self.model, self.data, geom, gdata, q)
        meshes: dict[str, list[np.ndarray]] = {}
        for i, obj in enumerate(geom.geometryObjects):
            body = self.model.frames[obj.parentFrame].name
            m = gdata.oMg[i]
            meshes.setdefault(body, []).append(_stl(obj.meshPath) * np.asarray(obj.meshScale) @ m.rotation.T + m.translation)
        self.mesh = {k: np.vstack(v) for k, v in meshes.items()}

    def placement(self, frame: str) -> pin.SE3:
        fid = self.model.getFrameId(frame)
        if fid >= self.model.nframes:
            raise KeyError(frame)
        return self.data.oMf[fid]

    def flex_motion(self, link: str, frame: str, local: np.ndarray) -> np.ndarray:
        joint = self.model.frames[self.model.getFrameId(link)].parentJoint
        data = self.model.createData()
        points = []
        for dq in (0.0, 0.05):
            q = pin.neutral(self.model)
            q[self.model.joints[joint].idx_q] += dq
            pin.framesForwardKinematics(self.model, data, q)
            points.append(data.oMf[self.model.getFrameId(frame)].act(local))
        return points[1] - points[0]


def _pad_depth(verts, origin, axis, normal, s_centre) -> float:
    lateral_axis = np.cross(axis, normal)
    rel = verts - origin
    s, lateral, depth = rel @ axis, rel @ lateral_axis, rel @ normal
    for k in (1, 2, 3, 4):
        sel = (np.abs(s - s_centre) < SLICE_M * k) & (np.abs(lateral) < LATERAL_M * k)
        if sel.sum() >= 3:
            return float(depth[sel].max())
    raise RuntimeError("指腹截面没有网格顶点")


def _finger(hand: _Hand, axis_points, distal_mesh, flex_link, flex_frame, flex_local, palmar_check, inset_m):
    pip, dip, tip = axis_points
    a_tip = (tip - dip) / np.linalg.norm(tip - dip)
    a_pip = (dip - pip) / np.linalg.norm(dip - pip)
    motion = hand.flex_motion(flex_link, flex_frame, flex_local)
    n_tip = motion - (motion @ a_tip) * a_tip
    n_tip /= np.linalg.norm(n_tip)
    if n_tip @ palmar_check <= 0:
        raise RuntimeError(f"{flex_link}: 掌侧方向检查失败")
    n_pip = n_tip - (n_tip @ a_pip) * a_pip
    n_pip /= np.linalg.norm(n_pip)
    s_end = float(((distal_mesh - dip) @ a_tip).max())
    s_centre = s_end * (1.0 - TIP_FRACTION)
    depth = _pad_depth(distal_mesh, dip, a_tip, n_tip, s_centre) - inset_m
    pads = (pip + n_pip * depth, dip + n_tip * depth, dip + a_tip * s_centre + n_tip * depth)
    return pads, {"depth_mm": round(depth * 1e3, 2),
                  "back_from_tip_mm": float(round((np.linalg.norm(tip - dip) - s_centre) * 1e3, 2))}


def glove(side: str, inset_m: float) -> dict:
    s = "L" if side == "left" else "R"
    base = ROOT / "assets/hand_motor_ve33" / f"hand_motor_ve33_{s}"
    hand = _Hand(base / "hand_tree_only.urdf")
    path = base / "config/retarget_targets.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    sites = {item["name"]: item for item in cfg["sites"]}
    axis_local = lambda item: np.asarray(item.get("axis_local_xyz_m", item["local_xyz_m"]), dtype=float)
    world = lambda item: hand.placement(item["link"]).act(axis_local(item))
    report = {}
    for f in FINGERS:
        names = ("thumb_mcp", "thumb_ip", "thumb_tip") if f == "thumb" else (f"{f}_pip", f"{f}_dip", f"{f}_tip")
        pts = [world(sites[n]) for n in names]
        tip_site = sites[names[2]]
        check = world(sites["index_tip"]) - pts[2] if f == "thumb" else np.array([0.0, 0.0, -1.0])
        distal = hand.mesh["thumb_ip_link" if f == "thumb" else f"{f}_dip_link"]
        pads, info = _finger(hand, pts, distal, tip_site["link"], tip_site["link"], axis_local(tip_site), check, inset_m)
        report[f] = info
        for name, pad in zip(names, pads):
            item = sites[name]
            item.setdefault("axis_local_xyz_m", item["local_xyz_m"])
            item.setdefault("axis_zero_world_xyz_m", item["zero_world_xyz_m"])
            item["local_xyz_m"] = hand.placement(item["link"]).actInv(pad).tolist()
            item["zero_world_xyz_m"] = pad.tolist()
            item["task_point"] = "pad"
    cfg["task_point_rule"] = ("指腹任务点：scripts/generate_pad_task_points.py；各指 PIP/DIP/指尖沿掌侧法向偏移远节指腹深度，"
                              f"指尖取远节网格最远端往回 {TIP_FRACTION:.0%}；原轴线点见 axis_local_xyz_m")
    cfg["pad_report"] = report
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def wuji(side: str, inset_m: float) -> dict:
    p = "l" if side == "left" else "r"
    urdf_dir = ROOT / "assets/wuji-hand2/body/urdf"
    source = urdf_dir / f"{side}.urdf"
    hand = _Hand(source)
    extra, report = [], {}
    for f in FINGERS:
        fn = WUJI_FINGER[f]
        frames = (f"{p}_{fn}_middle", f"{p}_{fn}_distal", f"{p}_{fn}_tip")
        pts = [hand.placement(x).translation.copy() for x in frames]
        check = hand.placement(f"{p}_index_finger_tip").translation - pts[2] if f == "thumb" else np.array([0.0, -1.0, 0.0])
        pads, info = _finger(hand, pts, hand.mesh[frames[1]], frames[1], frames[2], np.zeros(3), check, inset_m)
        report[f] = info
        for key, frame, pad in zip(("pip", "dip", "tip"), frames, pads):
            local = hand.placement(frame).actInv(pad)
            child = f"{p}_{fn}_pad_{key}"
            xyz = " ".join(f"{v:.9g}" for v in local)
            extra.append(f'  <link name="{child}" />\n'
                         f'  <joint name="{child}_fixed" type="fixed">\n'
                         f'    <origin xyz="{xyz}" rpy="0 0 0" />\n'
                         f'    <parent link="{frame}" />\n    <child link="{child}" />\n  </joint>\n')
    text = source.read_text(encoding="utf-8")
    marker = "<!-- generated pad task frames: scripts/generate_pad_task_points.py -->\n"
    out = text[:text.rindex("</robot>")] + marker + "".join(extra) + "</robot>\n"
    (urdf_dir / f"{side}_pad.urdf").write_text(out, encoding="utf-8")

    cfg_dir = ROOT / "config/retargeting/official"
    yaml_path, axis_path = cfg_dir / f"{side}.yaml", cfg_dir / f"{side}_axis.yaml"
    if not axis_path.exists():
        axis_path.write_text(yaml_path.read_text(encoding="utf-8"), encoding="utf-8")
    y = axis_path.read_text(encoding="utf-8")
    y = y.replace(f'urdf/{side}.urdf"', f'urdf/{side}_pad.urdf"')
    y = re.sub(r'pip:\s+"\{finger\}_middle"', 'pip:     "{finger}_pad_pip"   ', y)
    y = re.sub(r'dip:\s+"\{finger\}_distal"', 'dip:     "{finger}_pad_dip"   ', y)
    y = re.sub(r'tip:\s+"\{finger\}_tip"', 'tip:     "{finger}_pad_tip"', y)
    if "_pad_tip" not in y or f"{side}_pad.urdf" not in y:
        raise RuntimeError(f"{axis_path} 的 urdf_path / link_naming 格式不符，未能改成指腹坐标系")
    header = ("# 任务点：指腹（scripts/generate_pad_task_points.py 生成 *_pad.urdf 的固定坐标系）。\n"
              "# segment_scaling 由同一脚本重新标定（见 fit_pad_scaling），不是手调值。\n"
              f"# 原手指中心轴线版本：{axis_path.name}\n")
    yaml_path.write_text(header + y, encoding="utf-8")
    return report


def fit_pad_scaling(side: str) -> dict:
    """Re-fit segment_scaling for the pad points.

    The axis scalings were tuned so that the glove's CAD zero maps to an open Wuji
    hand; pad points sit elsewhere, so reusing them curls the DIPs (~0.4 rad on the
    middle/pinky).  Reference: the Wuji pose the *axis* configuration produces at
    glove zero.  For every task point the least-squares scalar s minimises
    |s * glove_pad_vector - wuji_pad_vector| (wrist-relative, optimizer frame).
    """
    from motorhand_teleop.domain import GLOVE_JOINTS
    from motorhand_teleop.landmarks import Ve33LandmarkSource
    from motorhand_teleop.retargeting import GloveWujiRetargetPipeline, wuji_hand2_joint_names

    p = "l" if side == "left" else "r"
    s = "L" if side == "left" else "R"
    cfg_dir = ROOT / "config/retargeting/official"
    axis_assets = ROOT / "build/pad_fit_axis_assets" / side
    base = ROOT / "assets/hand_motor_ve33" / f"hand_motor_ve33_{s}"
    # Axis-mode glove targets: the same asset tree with the axis site offsets.
    target = axis_assets / f"hand_motor_ve33_{s}"
    if target.exists():
        import shutil
        shutil.rmtree(axis_assets)
    for item in base.rglob("*"):
        dest = target / item.relative_to(base)
        if item.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        elif item.relative_to(base).as_posix() != "config/retarget_targets.json":
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(item)
    cfg = json.loads((base / "config/retarget_targets.json").read_text(encoding="utf-8"))
    for item in cfg["sites"]:
        item["local_xyz_m"] = item.get("axis_local_xyz_m", item["local_xyz_m"])
    (target / "config").mkdir(parents=True, exist_ok=True)
    (target / "config/retarget_targets.json").write_text(json.dumps(cfg), encoding="utf-8")

    zero = dict.fromkeys(GLOVE_JOINTS, 0.0)
    axis = GloveWujiRetargetPipeline.from_yaml(side, asset_root=axis_assets, config_path=cfg_dir / f"{side}_axis.yaml")
    for _ in range(30):  # let the low-pass filter settle
        q_ref = axis.step(zero).positions_rad
    model = pin.buildModelFromUrdf(str(ROOT / "assets/wuji-hand2/body/urdf" / f"{side}_pad.urdf"))
    data = model.createData()
    q = pin.neutral(model)
    for name, value in zip(wuji_hand2_joint_names(side), q_ref):
        q[model.joints[model.getJointId(name)].idx_q] = value
    pin.framesForwardKinematics(model, data, q)
    wrist = data.oMf[model.getFrameId(f"{p}_wrist")].translation
    source = Ve33LandmarkSource(side, asset_root=ROOT / "assets/hand_motor_ve33")
    frame = source.step(zero)
    glove_vectors = (frame.points_m - frame.points_m[0]) @ source.model_to_canonical_row @ source.canonical_to_optimizer_row
    fit, residual = {}, {}
    for i, f in enumerate(FINGERS):
        for j, key in enumerate(("pip", "dip", "tip")):
            w = data.oMf[model.getFrameId(f"{p}_{WUJI_FINGER[f]}_pad_{key}")].translation - wrist
            g = glove_vectors[4 * i + 2 + j]
            scale = float(g @ w / (g @ g))
            fit.setdefault(f, []).append(round(scale, 3))
            residual.setdefault(f, []).append(round(float(np.linalg.norm(scale * g - w)) * 1e3, 1))
    yaml_path = cfg_dir / f"{side}.yaml"
    text = yaml_path.read_text(encoding="utf-8")
    for f, values in fit.items():
        text, n = re.subn(rf"(?m)^(    {f}:\s+)\[[^\]]*\]", lambda m: m.group(1) + "[" + ", ".join(map(str, values)) + "]", text)
        if n != 1:
            raise RuntimeError(f"{yaml_path}: segment_scaling.{f} 未找到")
    yaml_path.write_text(text, encoding="utf-8")
    return {"segment_scaling": fit, "residual_mm": residual}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--side", choices=("right", "left", "both"), default="both")
    parser.add_argument("--glove-inset-mm", type=float, default=0.0, help="手套指套壳厚：人指腹在外壳内侧的距离")
    parser.add_argument("--wuji-inset-mm", type=float, default=0.0)
    args = parser.parse_args()
    for side in (("right", "left") if args.side == "both" else (args.side,)):
        print(side, "glove", glove(side, args.glove_inset_mm / 1e3))
        print(side, "wuji ", wuji(side, args.wuji_inset_mm / 1e3))
        print(side, "scale", fit_pad_scaling(side))


if __name__ == "__main__":
    main()
