"""只读实际CAD及电机软限位，生成可复现的闭链重定向训练/验证姿态。

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=Motorhand_FFB/src \
  .pixi/envs/default/bin/python Motorhand_FFB/scripts/generate_retarget_fixtures.py \
  --hand right --output Motorhand_FFB/.cache/retarget-fit/motorhand_reachable_right.json

左侧可加 --reuse-motors 右侧JSON：按左侧资产的mirror_joint_signs转换电机输入，
随后用左侧真实闭链重新计算全部树关节和任务点，不镜像或复制右侧点数据。
本脚本不连接硬件，不修改资产、标定或生产重定向配置，不做碰撞判定。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from motorhand_teleop.ve33_kinematics import Ve33Kinematics


ROOT = Path(__file__).resolve().parents[2]
FINGERS = ("index", "middle", "ring", "pinky")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FixtureGenerator:
    def __init__(self, hand: str, mapping_path: Path):
        self.hand = hand
        self.model_dir = ROOT / "assets/hand_motor_ve33" / f"hand_motor_ve33_{'R' if hand == 'right' else 'L'}"
        self.kin = Ve33Kinematics(hand, self.model_dir)
        self.targets_path = self.model_dir / "config/retarget_targets.json"
        targets = read_json(self.targets_path)
        self.rotation = np.asarray(targets["canonical_to_model"])
        self.sites = [(s["link"], np.asarray(s["local_xyz_m"])) for s in sorted(targets["sites"], key=lambda s: s["index"])]
        self.mapping_path = mapping_path.resolve()
        mapping = read_json(self.mapping_path)
        definitions = read_json(self.model_dir / "config/joint_space.json")["independent"]
        joint_to_motor = {v["name"].removesuffix("_joint"): v["motor"] for v in definitions}
        limits = {}
        for motor in mapping["motors"].values():
            edges = np.deg2rad(np.array([motor["min_deg"], motor["max_deg"]]) - motor["zero_deg"])
            edges *= motor["sign"] * motor["scale"]
            limits[joint_to_motor[motor["joint"]]] = [float(min(edges)), float(max(edges))]
        self.bounds = np.array([limits[n] for n in self.kin.motor_names])
        self.dependent_names = [f"{f}_dip_joint" for f in FINGERS]

    def evaluate(self, motors, seed=None):
        motors = np.asarray(motors, dtype=float)
        if motors.shape != (12,) or not np.isfinite(motors).all():
            raise ValueError("电机姿态必须是12个有限弧度值")
        pose = self.kin.evaluate_motors(dict(zip(self.kin.motor_names, motors)), seed=seed)
        actual = np.array([pose.motor_positions[n] for n in self.kin.motor_names])
        if np.any(actual < self.bounds[:, 0] - 1e-10) or np.any(actual > self.bounds[:, 1] + 1e-10):
            raise ValueError("姿态超出输入配置的模型电机软件边界")
        points, _ = self.kin.point_kinematics(pose, self.sites)
        points[1] = (points[0] + points[2]) * .5
        return pose, (points - points[0]) @ self.rotation

    def frame(self, motors, sequence, phase, progress, split="train", target_finger=None, seed=None):
        pose, points = self.evaluate(motors, seed)
        record = {
            "sequence": sequence, "phase": phase, "progress": float(progress),
            "target_finger": target_finger, "split": split,
            "joint_rad": [pose.joint_positions[n] for n in self.kin.joint_names],
            "dependent_rad": [pose.dependent_positions[n] for n in self.dependent_names],
            "motor_rad": [pose.motor_positions[n] for n in self.kin.motor_names],
            "tree_rad": [pose.all_positions[n] for n in self.kin.tree_names],
            "points_m": points.tolist(), "max_closure_gap_m": pose.max_gap_m,
            "thumb_pad_distance_m": {f: float(np.linalg.norm(points[4] - points[(i + 2) * 4])) for i, f in enumerate(FINGERS)},
        }
        return record, pose

    def search_opposition(self, finger_index, random):
        active = np.array([0, 1, 2, 3, 4 + finger_index * 2, 5 + finger_index * 2])
        tip = (finger_index + 2) * 4
        lo, hi = self.bounds[active].T
        invalid = 0

        def residual(x):
            nonlocal invalid
            motors = np.zeros(12)
            motors[active] = x
            try:
                _, points = self.evaluate(motors)
                return (points[4] - points[tip]) * 1000.
            except ValueError:
                invalid += 1
                return np.full(3, 1e4) + np.sum(x * x)

        candidates = []
        seeds = [np.zeros(6), np.array([.8, -.2, .5, .3, .7, .5])]
        seeds.extend(random.uniform(lo + .02, hi - .02) for _ in range(6))
        for seed in seeds:
            solution = least_squares(residual, np.clip(seed, lo + 1e-7, hi - 1e-7), bounds=(lo, hi),
                                     ftol=1e-10, xtol=1e-10, gtol=1e-9, max_nfev=150)
            candidates.append((float(np.linalg.norm(residual(solution.x))), solution.x))
        distance, best = min(candidates, key=lambda item: item[0])
        motors = np.zeros(12)
        motors[active] = best
        self.evaluate(motors)
        return {
            "minimum_found_pad_distance_mm": distance, "closed_under_1mm": distance < 1.,
            "active_motor_indices": active.tolist(), "motor_rad": motors.tolist(),
            "optimizer_attempts": len(candidates), "all_attempt_distance_mm": [v[0] for v in candidates],
            "invalid_trials_rejected": invalid,
            "qualification": "有界多初值局部搜索的最小发现值；不是不可达性证明。",
        }

    def generate(self):
        frames = [self.frame(np.zeros(12), "open", "open", 0.)[0]]
        extrema = {}
        random = np.random.default_rng(20261117)
        for index, finger in enumerate(FINGERS):
            extrema[finger] = self.search_opposition(index, random)
            target = np.array(extrema[finger]["motor_rad"])
            seed = self.kin.current
            for phase, fractions in (("approach", np.linspace(0, 1, 21)), ("hold", [1.] * 3), ("reopen", np.linspace(1, 0, 21))):
                for fraction in fractions:
                    frame, seed = self.frame(target * fraction, "opposition_" + finger, phase, fraction,
                                             target_finger=finger, seed=seed)
                    frames.append(frame)
            print(f"{finger}: {extrema[finger]['minimum_found_pad_distance_mm']:.6g} mm", flush=True)
        grasp = np.array([.45, -.15, .55, .8, .8, .9, .8, .9, .8, .9, .8, .7])
        seed = self.kin.current
        for phase, fractions in (("close", np.linspace(0, 1, 16)), ("reopen", np.linspace(1, 0, 16))):
            for fraction in fractions:
                frame, seed = self.frame(grasp * fraction, "grasp", phase, fraction, seed=seed)
                frames.append(frame)
        for finger in FINGERS:
            active = np.array(extrema[finger]["active_motor_indices"])
            other = np.array([i for i in range(12) if i not in active])
            target = np.array(extrema[finger]["motor_rad"])
            offset = np.array([.11, -.07, .08, -.05, .06, -.09])
            seed = self.kin.current
            for phase, fractions in (("approach", np.linspace(0, 1, 17)), ("hold", [1.] * 2), ("reopen", np.linspace(1, 0, 17))):
                for fraction in fractions:
                    motors = target * fraction
                    motors[active] += np.sin(np.pi * fraction) * offset
                    motors[other] = .18 + .08 * np.sin(.4 + np.arange(len(other)) * .7 + fraction)
                    motors = np.clip(motors, self.bounds[:, 0], self.bounds[:, 1])
                    frame, seed = self.frame(motors, "validation_opposition_" + finger, phase, fraction,
                                             "validation", finger, seed)
                    frames.append(frame)
        return frames, extrema

    def reuse_motors(self, source_path):
        source = read_json(source_path)
        if source.get("hand") not in ("right", "left"):
            raise ValueError("复用轨迹必须声明right或left侧别")
        if tuple(source["motor_names"]) != self.kin.motor_names:
            raise ValueError("复用轨迹电机轴顺序与模型不同")
        signs = np.ones(12)
        if source["hand"] != self.hand:
            mirror = read_json(ROOT / "assets/hand_motor_ve33/hand_motor_ve33_L/config/mirror_joint_signs.json")
            signs = np.array([mirror[n] for n in self.kin.motor_names])
        frames, previous, seed = [], None, self.kin.current
        for old in source["frames"]:
            if old["sequence"] != previous:
                seed = self.kin.current
            frame, seed = self.frame(np.array(old["motor_rad"]) * signs, old["sequence"], old["phase"],
                                     old["progress"], old["split"], old["target_finger"], seed)
            frames.append(frame)
            previous = old["sequence"]
        extrema = {}
        for index, finger in enumerate(FINGERS):
            motors = np.array(source["opposition_extrema"][finger]["motor_rad"]) * signs
            _, points = self.evaluate(motors)
            distance = float(np.linalg.norm(points[4] - points[(index + 2) * 4]) * 1000.)
            extrema[finger] = {
                "minimum_found_pad_distance_mm": distance, "closed_under_1mm": distance < 1.,
                "active_motor_indices": source["opposition_extrema"][finger]["active_motor_indices"],
                "motor_rad": motors.tolist(), "qualification": "复用电机轨迹后由本侧实际CAD闭链独立验证；不是点位镜像。",
            }
        return frames, extrema

    def save(self, output, frames, extrema, reused=None):
        train = np.array([f["motor_rad"] for f in frames if f["split"] == "train"])
        validation = np.array([f["motor_rad"] for f in frames if f["split"] == "validation"])
        separation = float(np.min(np.linalg.norm(train[:, None] - validation[None, :], axis=2)))
        if separation <= 1e-8:
            raise ValueError("训练和验证集存在重复电机姿态")
        data = {
            "schema_version": 1, "hand": self.hand, "input_frame": "canonical", "input_units": "m",
            "point_definition": "当前retarget_targets.json指腹点；真实闭链FK；point1=(point0+point2)/2；(points-points[0])@canonical_to_model，与Ve33Model一致。",
            "source": "模型合成、真实CAD闭链可达；不是硬件录制，不包含碰撞判定。",
            "targets_path": str(self.targets_path.relative_to(ROOT)), "targets_sha256": sha256(self.targets_path),
            "mapping_path": str(self.mapping_path), "mapping_sha256": sha256(self.mapping_path),
            "limits_source": "指定配置中的软件电机边界经zero/sign/scale转为模型rad；每帧同时检查全部CAD树分支界限和闭链残差；不额外声称本侧实物已验证。",
            "motor_names": self.kin.motor_names, "joint_names": self.kin.joint_names,
            "dependent_joint_names": self.dependent_names, "tree_names": self.kin.tree_names,
            "motor_bounds_rad": self.bounds.tolist(), "opposition_extrema": extrema,
            "split_rule": "213帧训练；144帧验证改变非对指手指曲度并平滑偏移主动电机路径；同一序列往返/保持只在同一split。",
            "minimum_cross_split_motor_distance_rad": separation, "frames": frames,
        }
        if reused:
            data["reused_motor_fixture"] = {"path": str(reused.resolve()), "sha256": sha256(reused)}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(output.with_suffix(".npz"), **{key: np.array([f[key] for f in frames]) for key in
                           ("points_m", "motor_rad", "joint_rad", "dependent_rad", "sequence", "phase", "split")})
        print(json.dumps({"path": str(output), "frames": len(frames), "train": len(train), "validation": len(validation),
                          "max_gap_m": max(f["max_closure_gap_m"] for f in frames),
                          "closed_distance_mm": {f: v["minimum_found_pad_distance_mm"] for f, v in extrema.items()}}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=("right", "left"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-motors", type=Path, help="复用输入电机轨迹，但用本侧CAD重算全部点和关节")
    parser.add_argument("--motor-mapping", type=Path, default=ROOT / "hardware/encos_dexhand/configs/sim/glove_mapping.json")
    args = parser.parse_args()
    generator = FixtureGenerator(args.hand, args.motor_mapping)
    frames, extrema = generator.reuse_motors(args.reuse_motors) if args.reuse_motors else generator.generate()
    generator.save(args.output, frames, extrema, args.reuse_motors)


if __name__ == "__main__":
    main()
