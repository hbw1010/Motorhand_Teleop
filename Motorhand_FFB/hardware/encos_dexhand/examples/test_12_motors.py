# -*- coding: utf-8 -*-
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from encos_hand import EncosHand

def main():
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 70)
    print("🤖 Encos 12-Axis Dexterous Hand SDK 终端自检测试")
    print("=" * 70)

    hand = EncosHand()
    print(">> 正在连接 USB-CAN 通道...")
    hand.connect()

    print(">> 正在全总线自检重排发现节点...")
    found_map = hand.scan()
    total_axes = sum(len(v) for v in found_map.values())
    print(f">> 自检完成！通道分布: {found_map} | 在线总轴数: {total_axes} 轴\n")

    if total_axes == 0:
        print("⚠️ 未发现任何在线电机节点，请检查供电与接线！")
        hand.close()
        return

    print(">> 12 轴实时角度监控 (手掰关节即可观察读数，按 Ctrl+C 退出):")
    print("-" * 70)
    
    try:
        while True:
            angles = hand.get_all_angles()
            # 分 3 个 CAN 通道整齐排版，杜绝行溢出折叠
            line_ch1 = " | ".join([f"ID {mid:02d}: {angles.get(mid, 0.0):6.2f}°" for mid in [1, 2, 3, 4] if mid in angles])
            line_ch2 = " | ".join([f"ID {mid:02d}: {angles.get(mid, 0.0):6.2f}°" for mid in [5, 6, 7, 8] if mid in angles])
            line_ch3 = " | ".join([f"ID {mid:02d}: {angles.get(mid, 0.0):6.2f}°" for mid in [9, 10, 11, 12] if mid in angles])

            output = (
                f"\r[CAN-1] {line_ch1}\n"
                f"[CAN-2] {line_ch2}\n"
                f"[CAN-3] {line_ch3}\033[2A"  # ANSI 向上回退两行平滑刷新
            )
            sys.stdout.write(output)
            sys.stdout.flush()
            time.sleep(0.04)
    except KeyboardInterrupt:
        print("\n\n\n>> 测试完毕，断开通信连接。")
        hand.close()

if __name__ == "__main__":
    main()
