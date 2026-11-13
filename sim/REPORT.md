# 验收报告 / 交付说明

日期：2026-11-10。所有“已验证”均为本机**离线**结果；本机当前没有 CANable2/真机，真机项均未宣称已验证。

## 1. 功能完成情况

| 需求 | 状态 | 说明 |
|---|---|---|
| 从原仓库抽取手部代码/SDK/模型/映射 | 完成 | `sim/` 独立于 Wuji、ROS2、PICO、DAgger；天机适配代码已移除，历史配置保留 |
| 使用手套本体模型实时仿真 | 完成 | `hand_motor_ve33_R` 68 关节闭环模型，CAD 闭环解算驱动 |
| 真机状态 → 仿真 → 接触力 → 真机力矩 | 代码完成，真机未验证 | `RealGlove` 只读检查、使能、MIT 力矩前馈；读取失败/电机报警自动急停；假手套演示可完整跑通 |
| 碰撞/接触/反馈 | 完成 | 17 个手套碰撞代理；物体—物体、物体—手套碰撞；锁定物体时按手指防穿模限位；反馈用未限位目标接触经闭环雅可比投影到 J1..J12 |
| 触感软硬差异 | 完成（离线） | 软/硬球的接触力与形变不同；见第 3 节数值 |
| 可调大小/软硬，实时更新 | 完成 | UI 滑条，限频 10 Hz 重建场景；拖动/反馈期间仍可用 |
| 物体默认固定、不滚动漂移 | 完成 | 默认锁定自由根体位姿；位移测试为 0；“锁定物体”可关闭以让手套推动 |
| 鼠标拖动位置/旋转，姿态保持 | 完成 | 左键 PD 伺服拖动、黄色选中高亮、与其他物体/手套碰撞；Shift+左键旋转；右键拖动旋转视角；重编译后保留 |
| 至少一个实际形变物体 | 完成 | `ball_soft` 四面体 flex 晶格真实形变；初始应变 0 |
| 球/方块/圆柱，软硬各一 | 完成 | 6 个物体；软方块/圆柱为柔顺接触刚体，无网格形变 |
| 设备识别/状态读取 | 代码完成 | VID/PID `16d0:117e`；只读 `check-glove`；无设备时明确退出 |
| 置零/标零 | 代码完成，真机未验证 | `zero-glove` 只写 mapping/calibration，备份；保留相对限位 |
| 回零 | 代码完成，真机未验证 | `home-glove` 调用原仓库 `encos_home.py`（会使能/运动） |
| 遥测记录/曲线/来源标注 | 完成 | CSV+JSONL；UI 页签显示力/力矩/关节位置/关节速度/电流/力—穿透或形变/时延 |
| 力—位移、延迟、抓取保持、滑移 | 部分完成 | 力—接触穿透、时延、保持已有；无物体位姿传感器，滑移明确标为不可用 |
| 数据来源严格区分 | 完成 | 仿真力、名义力矩、实测电流、未标定估计力分别标注；估计力留空 |
| 环境安装/自测 | 完成 | Pixi frozen 安装、23 项冒烟、8 项 pytest |

## 2. 修改/新增文件

| 文件 | 内容 |
|---|---|
| `sim/src/glove_sim/physics.py` | 运动学手套、物体锁定、接触力、闭环导数力矩投影、柔度上限/滤波/爬升、柔体应变 |
| `sim/src/glove_sim/kinematics.py` | CAD 闭环求解、修正导数列为 J1..J12 物理电机顺序 |
| `sim/src/glove_sim/model_builder.py` | 本体模型补 17 碰撞代理、光照/地面、离线帧缓冲、时间步配置 |
| `sim/src/glove_sim/objects.py` | 6 物体生成；软球 flex 晶格；软/硬接触参数；细分表面网格；固定根体 |
| `sim/src/glove_sim/runner.py` | 控制循环、真机/假手套后端、记录、急停 |
| `sim/src/glove_sim/telemetry.py` | 列定义、来源/单位、形变/保持/滑移不可用标记、墙钟+单调时间戳 |
| `sim/src/glove_sim/sdk/real.py` | CANable2 发现、12 轴检查、状态读取、MIT 力矩、received_ns 时延 |
| `sim/src/glove_sim/sdk/zero.py` | 软件零位写入、相对限位保持、备份 |
| `sim/src/glove_sim/ui/app.py` | DearPyGui 单窗口、自定义深色控制台主题、1280×720 离屏图像、左键移物/右键转视角、防穿模状态、全屏自适应、选中高亮、碰撞伺服拖动、暂停演示/聚焦/放到指尖/重置、滑条、反馈/急停/记录、曲线 |
| `sim/src/glove_sim/cli.py` | `sim/teleop/check-glove/zero-glove/home-glove/record/replay`；无设备/无显示时明确报错 |
| `sim/config/default.yaml` | 时间步 1 ms、6 物体布局、软硬映射、反馈限幅、防穿模限位、拖动速度限幅、torque_sign |
| `sim/scripts/smoke.py` | 23 项离线冒烟 |
| `sim/tests/test_core.py` | 8 项 pytest（含置零相对限位测试） |
| `sim/pixi.toml` / `pyproject.toml` / `pixi.lock` | 环境、任务、依赖锁定 |
| `sim/README.md` | 操作手册、数据字典、限制与排查 |

## 3. 已验证内容（本机离线）

```text
pixi install --frozen       成功
pixi run smoke              23 passed, 0 failed
pixi run test               9 passed
```

- 模型编译成功；17 个碰撞代理；初始布局 `contact_count=0`；200 步位姿漂移 `0.00e+00 m`；拖动后位姿保持。
- 软球脚本挤压：8–17 个接触、约 6–9 N、最大体局部形变约 9.5 mm；硬球：约 1 个接触、形变 0。
- 初始柔体应变为 0（修复了此前把半径偏移误当形变的问题）。
- 反馈路径：闭环导数 `68x12` 且列顺序 J1..J12；力矩上限测试通过；6 s 假手套运行出现接触力约 3.4 N、最大指令约 0.31 Nm。
- 记录：CSV+JSONL 生成；`timestamp_iso` 为真实 UTC 墙钟；`force_estimate_validity=uncalibrated`、`slip_validity=unavailable_no_pose_sensor`、`slip_m` 为空。
- 回放：JSONL 295 行成功回放。
- UI：本机 X11 下连续运行，自定义深色主题、1280×720 渲染、无阴影 + 参考网格；曝光已调低（headlight ambient 0.16 / diffuse 0.24，key/fill 环境光同步降低），实拍不再过曝。左键只移动/旋转物体，右键拖动旋转视角。状态栏区分“反馈接触”（幽灵目标）和“可视穿透点/深度”（真实主数据）；锁定模式实测状态为 `可视穿透 0点/0.0mm`。实测抓握/软球接触时 UI 循环约 12–20 Hz，暂停演示且无接触时约 27–43 Hz（物理仍按 1 kHz 步进，UI 每帧最多补 30 步）。拖动、碰撞、选中高亮、滑条重建、反馈开关、急停、记录按钮、F11 全屏均实测无崩溃；画面上下翻转导致手/拾取错位的问题已修复；暂停演示复选框语义已修正（勾选=暂停）。
- 无硬件时 `check-glove`、`teleop`、`zero-glove` 均明确失败退出，不误发命令。

## 4. 仍需真机验证（不得当作已通过）

1. 3 个 CANable2 的端口顺序、12 轴在线、udev/plugdev 权限。
2. 现有 `zero_deg` 方向/限位；`zero-glove` 写入后读数一致性。
3. `home-glove` 的使能/运动/回零方向/报警/超时/急停。
4. 每轴 `torque_sign`（默认全 +1）和 MIT 力矩方向：反馈应“顶手”，不能“吸手”。
5. 反馈增益/上限/斜率/滤波在佩戴状态下的力度、稳定性和舒适性。
6. CAN/编码器实际时延、丢帧、断链；命令 TTL 0.1 s 的真实保护效果。
7. 软球在解锁后挤压时的观感（锁定模式为保证零穿透不显示可见形变）；软方块/圆柱当前只有接触柔顺，没有网格形变。
8. 无真实力传感器、无 Kt 标定，不能验证实测接触力/估计力。

已知限制：
- 手套仍是**运动学**驱动。锁定物体时可见手型按手指限位，不再穿入硬物体；代价是可见手型会落后于真机手套角度，反馈力矩按未限位目标计算。取消锁定后手套可推动物体，但仍可能出现运动学接触穿透。
- 锁定模式 `glove.clamp_soft_allow_m=0`：软/硬物体可见穿透均为 0；软球可见形变只在取消锁定后出现。软弱方块/圆柱只有接触柔顺，没有网格形变。
- 大幅挤压或 40–70 mm 大球会产生较大接触力和较大的幽灵穿透量（用于反馈，不等于可见穿透）；名义力矩有硬上限。
- UI 接触时约 12–20 Hz，不是 200 Hz 控制回路；软球 flex 接触是主要计算开销。物体默认锁定；无物体位姿传感器，因此不测物体滑移。

## 5. 可直接复制执行命令

```bash
cd /home/fcl/hbw/Motorhand_Teleop/sim

# 安装与自测
pixi install --frozen
pixi run smoke
pixi run test

# 设备检查（只读）
pixi run check-glove

# 软件置零（不驱动电机；先摆好零位）
pixi run zero-glove --all
# 或逐指：
pixi run zero-glove --finger thumb

# 统一使能回零（会运动，先清场）
pixi run home-glove --tolerance-rad 0.15 --speed-rpm 5.0

# 离线仿真（默认包含可拖动的 ball_soft 形变球）
pixi run sim
pixi run sim --headless --feedback --duration 6

# 真机遥操作（UI 默认反馈关闭，检查后再勾选）
pixi run teleop

# 真机无窗口（确认安全后）
pixi run teleop --headless --feedback --duration 10

# 记录/回放
pixi run record --record data
pixi run record --headless --feedback --duration 30 --record data
pixi run replay data/glove_session-YYYYMMDD-HHMMSS.jsonl
```

UI 加载/交互形变球：启动 `pixi run sim` 后默认场景已有 `ball_soft`；左键拖动把它放到手指间，调大小/硬度，佩戴/假手套闭合即可看到形变与接触力。反馈开启时这些操作仍可用。

## 6. 验收结论

离线功能、场景交互、软硬差异、接触力到名义力矩链路、记录/曲线、设备/标零命令均已实现并通过本机自测。真机部分只完成代码与安全边界，**明天连接真机后仍需按第 4 节逐项验证**，尤其力矩符号、回零方向和反馈增益；未验证前不要直接佩戴大力矩。
