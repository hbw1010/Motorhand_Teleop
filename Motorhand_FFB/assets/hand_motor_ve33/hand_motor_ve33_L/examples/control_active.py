"""离线指节输入示例：输出闭环逆解电机角与完整树姿态，不连接硬件。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[2]))

import argparse
import json
from hand_controller.ve33_kinematics import Ve33Kinematics
from hand_controller.ve33_preview import model_hand


def main():
    hand = Ve33Kinematics(model_hand(ROOT), model_dir=ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joint', choices=hand.joint_names, help='12 个真实指节坐标之一；省略则列出名称')
    parser.add_argument('--angle', type=float, default=.1, help='真实指节角度，单位 rad；不是电机角度')
    args = parser.parse_args()
    if args.joint is None:
        print(json.dumps(dict(joint_names=hand.joint_names, angle_unit='rad'), ensure_ascii=False, indent=2))
        return
    try:
        pose = hand.solve({args.joint: args.angle})
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(dict(joint_positions=pose.joint_positions, motor_positions=pose.motor_positions,
                         tree_positions=pose.all_positions, dependent_positions=pose.dependent_positions,
                         max_closure_gap_m=pose.max_gap_m,
                         warning='未标定硬件限位；逆解输出仅供离线预览，不可直接发送硬件'),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
