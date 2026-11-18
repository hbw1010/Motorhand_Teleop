"""离线验收Sharpa位置执行器动态跟踪；保留原MJCF自碰撞、惯量和限位。

只把native重定向q写入ctrl，再调用mj_step；不覆盖qpos。
实际pad点由mj_step产生的关节反馈经派生pad URDF正解计算。
每个物理步记录目标指的首次接触和最深接触，另用独立状态测量命令几何相交。
指腹任务点距离、MuJoCo接触和真实硬件触觉是三个不同指标。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from fit_sharpa_retarget import JOINT_SUFFIXES, ROOT, SharpaFK, pad_urdf


FINGERS = ("index", "middle", "ring", "pinky")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def active_opposition_contacts(model, data, side, target):
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.efc_address < 0:
            continue
        bodies = [model.body(int(model.geom_bodyid[geom])).name for geom in contact.geom]
        if any(name.startswith(f"{side}_thumb_") for name in bodies) and any(
            name.startswith(f"{side}_{target}_") for name in bodies
        ):
            yield index, contact, bodies


def describe_contact(model, contact, bodies):
    return {
        "bodies": bodies,
        "geoms": [model.geom(geom).name or f"geom_{geom}" for geom in contact.geom],
        "distance_mm": float(contact.dist * 1000),
    }


def normal_force(model, data, index):
    force = np.zeros(6)
    mujoco.mj_contactForce(model, data, index, force)
    return float(force[0])


def contact_summary(model, data, side, target, physical_points):
    contacts = []
    wrist = model.body(f"{side}_hand_C_MC").id
    wrist_rotation = data.xmat[wrist].reshape(3, 3)
    if target:
        tip = (FINGERS.index(target) + 2) * 4
        for index, contact, bodies in active_opposition_contacts(model, data, side, target):
            point = (contact.pos - data.xpos[wrist]) @ wrist_rotation
            contacts.append({
                **describe_contact(model, contact, bodies),
                "position_canonical_m": point.tolist(), "normal_force_n": normal_force(model, data, index),
                "distance_to_thumb_pad_point_mm": float(np.linalg.norm(point - physical_points[4]) * 1000.),
                "distance_to_target_pad_point_mm": float(np.linalg.norm(point - physical_points[tip]) * 1000.),
            })
    return {"all_contact_count": int(data.ncon), "thumb_target_contacts": contacts}


def collect_contacts(events, model, data, side, target, frame, source_distance, *, actual):
    first_key, deepest_key = (("first_actual_contact", "deepest_actual_contact") if actual else
                              ("first_command_intersection", "deepest_command_intersection"))
    for index, contact, bodies in active_opposition_contacts(model, data, side, target):
        distance = contact.dist * 1000
        if not actual and distance >= 0:
            continue
        first = events[first_key] is None
        worst = events[deepest_key]
        deepest = distance < 0 and (worst is None or distance < worst["distance_mm"])
        if first or deepest:
            event = {
                **describe_contact(model, contact, bodies),
                "fixture_frame_index": frame["fixture_frame_index"],
                "phase": frame["phase"], "progress": frame["progress"],
                "source_pad_distance_mm": source_distance,
            }
            if actual:
                event.update({"sequence_time_s": float(data.time), "normal_force_n": normal_force(model, data, index)})
            if first:
                events[first_key] = event
            if deepest:
                events[deepest_key] = event


def compact_frame(frame):
    return {key: value for key, value in frame.items() if key not in ("command_q_rad", "actual_q_rad")}


def validate(side: str, fixture_path: Path, commands_path: Path, interval_ms: float, require_provenance=False):
    fixture_bytes, command_bytes = fixture_path.read_bytes(), commands_path.read_bytes()
    fixture, commands = json.loads(fixture_bytes), json.loads(command_bytes)
    if fixture["hand"] != side or commands["hand"] != side:
        raise ValueError("输入姿态或native输出侧别错误")
    source_frames, command_frames = fixture["frames"], commands["frames"]
    if len(source_frames) != len(command_frames):
        raise ValueError("输入姿态与native输出帧数不一致")
    physical_fields = {"physical_points_m" in frame for frame in command_frames}
    if len(physical_fields) != 1:
        raise ValueError("同一native输出混用了有/无physical_points_m的版本")
    native_physical_field = "physical_points_m" if True in physical_fields else "points_m"
    mjcf = ROOT / "assets/sharpa-hand" / f"{side}.xml"
    files = {"fixture": fixture_path, "native_commands": commands_path, "mjcf": mjcf,
             "pad_urdf": pad_urdf(side), "retarget_config": ROOT / "config/retargeting/sharpa" / f"{side}.yaml"}
    fingerprints = {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in files.items()}
    fingerprints["fixture"]["sha256"] = hashlib.sha256(fixture_bytes).hexdigest()
    fingerprints["native_commands"]["sha256"] = hashlib.sha256(command_bytes).hexdigest()
    native_hashes = {"config_sha256": "retarget_config", "fixture_sha256": "fixture", "pad_urdf_sha256": "pad_urdf"}
    for field, name in native_hashes.items():
        if field in commands and commands[field] != fingerprints[name]["sha256"]:
            raise ValueError(f"native输出的{field}与当前文件不同，禁止混用拟合版本")
    provenance_verified = all(field in commands for field in native_hashes)
    if require_provenance and not provenance_verified:
        raise ValueError("正式验收要求native输出携带配置、fixture和pad URDF三个SHA256")
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    command_geometry = mujoco.MjData(model)
    fk = SharpaFK(side)
    names = [f"{side}_{suffix}" for suffix in JOINT_SUFFIXES]
    joints = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names])
    if model.nq != 22 or model.nu != 22 or np.any(joints < 0):
        raise ValueError("MJCF缺少22个语义关节或执行器")
    if np.any(model.jnt_type[joints] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("Sharpa语义轴必须全部为单自由度铰链")
    addresses = model.jnt_qposadr[joints]
    actuator_ids = []
    for joint in joints:
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
        if len(matches) != 1:
            raise ValueError("每个语义关节必须唯一对应原MJCF位置执行器")
        actuator_ids.append(int(matches[0]))
    actuators = np.array(actuator_ids)
    if np.any(model.actuator_trntype[actuators] != mujoco.mjtTrn.mjTRN_JOINT):
        raise ValueError("执行器必须直接驱动对应关节")
    lower, upper = model.jnt_range[joints].T
    if not np.all(model.jnt_limited[joints]):
        raise ValueError("验收要求原MJCF全部关节限位开启")
    collision_disabled = bool(int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CONTACT))
    if collision_disabled:
        raise ValueError("原MJCF关闭了接触，不能进行保留自碰撞的动态验收")
    steps = round(interval_ms / 1000. / model.opt.timestep)
    if steps < 1 or not np.isclose(steps * model.opt.timestep * 1000., interval_ms, atol=1e-9):
        raise ValueError("command interval必须是原MJCF timestep的正整数倍")

    records, sequence_records, sequence_contacts, previous = [], {}, {}, None
    largest_native_fk_error = 0.
    for index, (source, command_frame) in enumerate(zip(source_frames, command_frames)):
        command = np.asarray(command_frame["q_rad"], dtype=float)
        if command.shape != (22,) or not np.isfinite(command).all():
            raise ValueError(f"第{index}帧native命令不是22个有限弧度值")
        if np.any(command < lower - 1e-6) or np.any(command > upper + 1e-6):
            raise ValueError(f"第{index}帧native命令超出MJCF关节限位")
        if np.any(command < model.actuator_ctrlrange[actuators, 0] - 1e-6) or np.any(
            command > model.actuator_ctrlrange[actuators, 1] + 1e-6
        ):
            raise ValueError(f"第{index}帧native命令超出MJCF执行器ctrlrange")
        if source["sequence"] != previous:
            mujoco.mj_resetData(model, data)
            if np.max(np.abs(data.qpos[addresses])) > 1e-12:
                raise ValueError("MJCF默认姿态不是零位，不能声称每序列零位初始化")
            mujoco.mj_forward(model, data)
        previous = source["sequence"]
        command_points = fk.physical_points(command)
        # 新版points_m属于归一化task域，绝不能用它当实际毫米坐标。
        # 旧输出仅在与派生URDF逐帧独立物理FK一致时才可兼容。
        native_points = np.asarray(command_frame[native_physical_field])
        native_error = float(np.max(np.linalg.norm(command_points - native_points, axis=1)))
        largest_native_fk_error = max(largest_native_fk_error, native_error)
        if native_error > 1e-6:
            raise ValueError(f"第{index}帧native物理点与pad URDF不一致；归一化task域需要physical_points_m")
        target = source["target_finger"]
        record = {
            "fixture_frame_index": index, "sequence": source["sequence"], "split": source["split"],
            "phase": source["phase"], "progress": source["progress"], "target_finger": target,
        }
        if target:
            tip = (FINGERS.index(target) + 2) * 4
            points = np.asarray(source["points_m"])
            source_distance = float(np.linalg.norm(points[4] - points[tip]) * 1000.)
            events = sequence_contacts.setdefault(source["sequence"], {
                "first_actual_contact": None, "deepest_actual_contact": None,
                "first_command_intersection": None, "deepest_command_intersection": None,
            })
            # 命令的接触几何只在独立状态中计算，跟踪状态由执行器积分。
            command_geometry.qpos[addresses] = command
            mujoco.mj_fwdPosition(model, command_geometry)
            collect_contacts(events, model, command_geometry, side, target, record, source_distance, actual=False)
        data.ctrl[actuators] = command
        overshoot = 0.
        overshoot_joint = None
        for _ in range(steps):
            mujoco.mj_step(model, data)
            # 与原生控制一致，同步到积分后的qpos。
            mujoco.mj_forward(model, data)
            if target:
                collect_contacts(events, model, data, side, target, record, source_distance, actual=True)
            if not all(np.isfinite(v).all() for v in (data.qpos, data.qvel, data.qacc)):
                raise ValueError(f"第{index}帧MuJoCo状态出现非有限值")
            q = data.qpos[addresses]
            violations = np.maximum(lower - q, q - upper)
            worst = int(np.argmax(violations))
            if violations[worst] > overshoot:
                overshoot = float(violations[worst])
                overshoot_joint = names[worst]
            if any(int(w.number) for w in data.warning):
                raise ValueError(f"第{index}帧MuJoCo产生数值或容量警告，拒绝将自动恢复当作正常跟踪")
        actual = data.qpos[addresses].copy()
        actual_points = fk.physical_points(actual)
        error = actual - command
        record.update({
            "sequence_time_s": float(data.time), "command_q_rad": command.tolist(), "actual_q_rad": actual.tolist(),
            "joint_tracking_rmse_rad": float(np.sqrt(np.mean(error ** 2))),
            "joint_tracking_max_abs_rad": float(np.max(np.abs(error))),
            "joint_tracking_max_error_joint": names[int(np.argmax(np.abs(error)))],
            "max_limit_overshoot_during_interval_rad": overshoot,
            "limit_overshoot_joint": overshoot_joint,
            "actual_vs_command_pad_rms_mm": float(np.sqrt(np.mean(np.sum((actual_points - command_points) ** 2, axis=1))) * 1000.),
            **contact_summary(model, data, side, target, actual_points),
        })
        if target:
            record.update({"source_pad_distance_mm": source_distance,
                           "command_pad_distance_mm": float(np.linalg.norm(command_points[4] - command_points[tip]) * 1000.),
                           "actual_pad_distance_mm": float(np.linalg.norm(actual_points[4] - actual_points[tip]) * 1000.)})
        records.append(record)
        sequence_records.setdefault(source["sequence"], []).append(record)

    splits = {}
    for split in ("train", "validation"):
        selected = [f for f in records if f["split"] == split]
        opposition = {}
        for sequence, frames in sequence_records.items():
            if frames[0]["split"] != split or not frames[0]["target_finger"]:
                continue
            held = [f for f in frames if f["phase"] == "hold"]
            opposition[frames[0]["target_finger"]] = {
                "sequence": sequence,
                "closest_source_frame": compact_frame(min(frames, key=lambda f: f["source_pad_distance_mm"])),
                "last_hold_frame": compact_frame(held[-1]),
                "minimum_actual_hold_pad_distance_mm": min(f["actual_pad_distance_mm"] for f in held),
                **sequence_contacts[sequence],
            }
        splits[split] = {
            "frames": len(selected),
            "joint_tracking_rmse_rad": float(np.sqrt(np.mean([f["joint_tracking_rmse_rad"] ** 2 for f in selected]))),
            "maximum_joint_tracking_error_rad": max(f["joint_tracking_max_abs_rad"] for f in selected),
            "maximum_limit_overshoot_rad": max(f["max_limit_overshoot_during_interval_rad"] for f in selected),
            "maximum_limit_overshoot_frame": compact_frame(max(selected, key=lambda f: f["max_limit_overshoot_during_interval_rad"])),
            "opposition": opposition,
        }
    if any(sha256(path) != fingerprints[name]["sha256"] for name, path in files.items()):
        raise ValueError("验收期间输入或配置被更新，请待拟合完成后重新运行")
    return {
        "hand": side, "provenance": fingerprints, "mujoco_version": mujoco.__version__,
        "command_interval_ms": interval_ms, "timestep_ms": model.opt.timestep * 1000., "steps_per_command": steps,
        "initialization": "每个sequence独立mj_resetData到原MJCF零位、零速度；首帧直接执行50ms（或指定interval），无额外预热。",
        "dynamics": "保留原MJCF位置执行器、增益、惯量、重力、碰撞和限位；仅修改ctrl，使用mj_step积分，不覆盖qpos。",
        "contact_method": "每个物理步对积分后的实际qpos调用mj_forward，同步接触和法向力；离线命令负距离使用独立MjData。",
        "actual_point_method": "从mj_step后qpos按语义22轴读取actual_q_rad，再经当前派生pad URDF独立FK。",
        "finite_states": True, "mujoco_warnings": 0, "joint_names": names,
        "contact_enabled": not collision_disabled, "original_collision_exclusion_count": int(model.nexclude),
        "native_physical_comparison_field": native_physical_field,
        "native_configuration_provenance_verified": provenance_verified,
        "native_probe_binary_sha256": commands.get("probe_binary_sha256"),
        "native_physical_fk_max_difference_m": largest_native_fk_error, "splits": splits, "frames": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=("right", "left", "both"), default="both")
    parser.add_argument("--fixture-dir", type=Path, default=ROOT / ".cache/retarget-fit")
    parser.add_argument("--retarget-dir", type=Path, default=ROOT / ".cache/retarget-fit")
    parser.add_argument("--command-interval-ms", type=float, default=50.)
    parser.add_argument("--include-frames", action="store_true", help="附带每帧command/实际关节诊断；默认只保存汇总与关键帧")
    parser.add_argument("--require-provenance", action="store_true", help="正式验收强制native输出携带匹配配置、fixture与pad URDF的SHA256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sides = ("right", "left") if args.hand == "both" else (args.hand,)
    report = {
        "schema_version": 1, "hardware_connected": False,
        "scope": "仅MuJoCo仿真动态验收，不是硬件试验；指腹任务点距离不等于实体接触，更不代表真实触觉。",
        "hands": {side: validate(side, args.fixture_dir / f"motorhand_reachable_{side}.json",
                                 args.retarget_dir / f"sharpa_after_{side}.json", args.command_interval_ms,
                                 args.require_provenance) for side in sides},
    }
    if not args.include_frames:
        for result in report["hands"].values():
            del result["frames"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for side, result in report["hands"].items():
        for split, summary in result["splits"].items():
            print(json.dumps({"hand": side, "split": split, "joint_rmse_rad": summary["joint_tracking_rmse_rad"],
                              "max_limit_overshoot_rad": summary["maximum_limit_overshoot_rad"],
                              "last_hold_command_actual_pad_mm": {finger: [v["last_hold_frame"]["command_pad_distance_mm"],
                                                                          v["last_hold_frame"]["actual_pad_distance_mm"]]
                                                                   for finger, v in summary["opposition"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
