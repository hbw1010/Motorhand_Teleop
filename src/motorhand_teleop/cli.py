from __future__ import annotations

import argparse
import json
from pathlib import Path

from .domain import GLOVE_JOINTS
from .retargeting import GloveWujiRetargetPipeline, default_retarget_config

ROOT = Path(__file__).resolve().parents[2]


def _demo() -> None:
    pipeline = GloveWujiRetargetPipeline.from_yaml(
        "right",
        asset_root=ROOT / "assets" / "hand_motor_ve33",
        config_path=default_retarget_config(ROOT, "right"),
    )
    result = pipeline.step({name: 0.0 for name in GLOVE_JOINTS})
    print(json.dumps({
        "glove_dimensions": len(GLOVE_JOINTS),
        "motor_dimensions": len(pipeline.landmarks.model.kinematics.motor_names),
        "landmark_dimensions": [21, 3],
        "wuji_dimensions": len(result.positions_rad),
        "lateral_policy": result.diagnostics["lateral_policy"],
        "positions_rad": result.positions_rad.tolist(),
        "jacobian": result.diagnostics["jacobian"],
        "hardware_connected": False,
    }, ensure_ascii=False, indent=2))


def _validate_assets() -> None:
    paths = {
        "glove_right_urdf": ROOT / "assets/hand_motor_ve33/hand_motor_ve33_R/hand_tree_only.urdf",
        "glove_right_closed_chain": ROOT / "assets/hand_motor_ve33/hand_motor_ve33_R/hand_closed_chain.xml",
        "glove_right_targets": ROOT / "assets/hand_motor_ve33/hand_motor_ve33_R/config/retarget_targets.json",
        "glove_left_urdf": ROOT / "assets/hand_motor_ve33/hand_motor_ve33_L/hand_tree_only.urdf",
        "glove_left_closed_chain": ROOT / "assets/hand_motor_ve33/hand_motor_ve33_L/hand_closed_chain.xml",
        "glove_left_targets": ROOT / "assets/hand_motor_ve33/hand_motor_ve33_L/config/retarget_targets.json",
        "wuji_right_urdf": ROOT / "assets/wuji-hand2/body/urdf/right.urdf",
        "wuji_left_urdf": ROOT / "assets/wuji-hand2/body/urdf/left.urdf",
        "retarget_right": ROOT / "config/retargeting/official/right.yaml",
        "retarget_left": ROOT / "config/retargeting/official/left.yaml",
        "official_hand2_right_mjcf": ROOT / "vendor/wuji-retargeting/wuji_retargeting/wuji-description/hand2/hand2_beta1/body/mjcf/right.xml",
    }
    missing = [str(path.relative_to(ROOT)) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit("缺少模型资产: " + ", ".join(missing))
    print(json.dumps({"valid": True, "files": {key: str(path.relative_to(ROOT)) for key, path in paths.items()}}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="WujiHand2 力反馈手套离线工具")
    parser.add_argument("command", choices=("demo", "validate-assets"))
    args = parser.parse_args()
    if args.command == "demo":
        _demo()
    else:
        _validate_assets()

if __name__ == "__main__":
    main()
