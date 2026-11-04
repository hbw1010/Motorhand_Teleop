# Motorhand_Teleop

12 电机外骨骼手套与 Wuji Hand 2 的双向遥操作、轨迹录制回放和人工力反馈接管。Linux x86-64，Pixi 管理环境；项目同时保留真实设备接口与不连接设备的 MuJoCo/数值验证入口。

## 控制约定

- 正向：编码器原始角度 → 电机零位/方向/比例 → vE.3.3 闭链 → 21 个任务点 → 官方 WujiRetarget → Wuji 20 语义关节目标。手套没有四指侧摆输入，正向对应侧摆固定为零。
- 反向：Wuji 实测姿态 → 同一任务空间的受约束逆解 → 手套模型电机目标 → 闭链 FK → 手套真实指节。推理与手套接管只能有一个位置主导方，切换使用实测锚点平滑过渡。
- 力反馈：Wuji 电流、位置跟踪误差和速度用于估计受阻，经虚拟阻抗、几何投影、滤波与限幅后生成手套 MIT 力矩前馈。**电流不是直接测得的接触力，SDK 名义 Nm 也不是实测输出力矩。**当前效果仍需实机标定和验收。
- 软件配置的每轴 10 Nm 是用户选定的**名义命令上限，不是电机额定/安全扭矩**。接触反馈增益未完成实物标定；穿戴前核验设备、电机方向、限位和急停，先低负载测试，出现异常立即停止或断电。
- Wuji MIT `kp=5, kd=0.1`；手套位置跟随 MIT `kp=45, kd=0.5`；手套人手接管为 MIT `kp=0, kd=0` 加力矩前馈。两侧参数及单位不可混用。

## 安装与连接

```bash
cd /home/boen/hand/Motorhand_Teleop
pixi install -e hardware
pixi run -e hardware build-real-hand
pixi run -e hardware wuji-discover
```

`build-real-hand` 首次需要联网获取固定版本官方 C SDK；宿主需 `libudev`。Wuji SDK 固定为 `2026.8.31`，默认批准固件 `2.2.3`。发现过程只读、不使能。无显式序列号时，仅在发现**唯一**二代手时自动选择；优先使用 SDK 发现的 UDP 地址。若同一设备出现多个地址，仅在系统路由明确指向唯一物理有线网卡时自动选该地址，否则要求人工指定；连接后核对 SN、手侧和实测固件。项目不会修改网卡或 IP；单一 UDP 地址不单独证明流量走网线。

多台设备请使用通用入口的 `--wuji-serial`；自动发现不可用时还可指定 `--wuji-address`，地址连接仍会核验 SN。不要同时用 GUI、其他控制进程占用设备或使用同一曲线端口。

## 常用命令

除标明“只读/仿真”的命令外，下面的真机命令会产生运动。手套输出首次取得时先伺服回零：5 rpm、J1 1.5 A、J2–J12 1.0 A，到位容差 0.15 rad。中断或退出释放本会话输出，不额外执行退出回零。

| 用途 | 命令 | 设备行为 |
| --- | --- | --- |
| 检查模型与资产 | `pixi run validate-assets` | 离线，只读 |
| 模拟双手演示 | `pixi run mujoco-viewer` | 不连接真机；自动演示轨迹 |
| 手套标零及观察 | 见下方「手套标零与实时可视化」 | 先标零，再只读观察 |
| 一键回零 | `pixi run -e hardware encos-home` | 使能并伺服回零 |
| 手套遥操作 Wuji | `pixi run -e hardware wuji-teleop` | Wuji 运动；手套只读 |
| 接触估计力反馈 | `pixi run -e hardware wuji-force-contact` | 两侧使能，手套先回零；终端显示名义前馈 |
| 力反馈曲线窗口 | `pixi run -e hardware force-ui` | 只读本机 UDP，不控制设备 |

### 手套标零与实时可视化

连接手套并保持电机失能，确认 12 轴全部在线，再依次运行：

```bash
# 五指逐组标零
pixi run -e hardware encos-zero-step

# 标零完成后，用手套实测输入驱动 MuJoCo 模型
pixi run -e hardware encos-live-viewer
```

标零时按提示依次摆好拇指（J1–J4）、食指（J5–J6）、中指（J7–J8）、无名指（J9–J10）和小拇指（J11–J12），每摆好一指按回车确认。每次确认立即保存该指零位，全部完成显示 `ZERO_STEP_OK`；中断不会撤销已保存的手指。标定更新 `hardware/encos_dexhand/configs/` 下的 `glove_mapping.json` 和 `joint_calibration.json`；软限位随零位平移，相对范围不变。标零不发送位置或 MIT 目标，不执行回零。

可视化窗口同时显示手套和 Wuji Hand 2 仿真模型；逐指移动手套，检查零位、运动方向和对应关系。关闭窗口或按 `Ctrl+C` 退出。它不控制 Wuji 真机，也不发送手套位置或 MIT 目标；只读轮询含协议 0 A 探针，并非总线完全不发报文。

单指重标可用 `encos-zero-index`、`encos-zero-middle`、`encos-zero-ring`、`encos-zero-pinky`；如需拇指，用 `pixi run -e hardware encos-zero-step -- --finger thumb`。默认只修改本项目的标定，不同步其他工程。

### 录制、回放和按键接管

```bash
# 手套遥操作 Wuji 并录制实测状态；Ctrl+C 完成文件，已有文件不会覆盖
pixi run -e hardware wuji-record -- recordings/demo.jsonl

# 仅校验轨迹文件；不连接设备
pixi run -e hardware wuji-replay -- recordings/demo.jsonl

# 只回放 Wuji；会使能真机
pixi run -e hardware wuji-replay -- recordings/demo.jsonl --execute

# Wuji 回放，手套跟随实际姿态并可接管；会先回零手套、使能两侧
pixi run -e hardware wuji-takeover -- recordings/demo.jsonl
```

默认 `--speed 1.0`、50 Hz、启动及切换过渡 0.5 s。回放身份、固件和关节限位在使能前核验；轨迹保存原始采样时间、位置、速度、电流和发送目标，带完整性校验。手套逆解的数据超期时暂停轨迹并等待新鲜结果，不延长原始 100 ms 期限；持续无解会停止。回放过程按 **T 或空格**切换手套接管/恢复，按 **Q** 退出。人工接管暂停轨迹，手套跟随停止，力反馈接管开始；恢复时从暂停位置平滑衔接。

### 查看实时反馈曲线

```bash
# 终端 1：会使能真机和手套
pixi run -e hardware wuji-force-contact -- --curves-port 8765
# 终端 2：仅打开曲线窗口
pixi run -e hardware force-ui
```

窗口分开显示 Wuji 实测电流、目标/实际位置、手套名义前馈及可用的手套实测电流；关闭窗口不会停止控制，需在控制终端按 Ctrl+C。PlotJuggler 需系统 ROS Humble：用 `pixi run plotjuggler-bridge -- --forward-port 8765` 接收 UDP 8766 并转发至 UI，再运行 `pixi run plotjuggler-pinky`；此时控制命令使用 `--curves-port 8766`。

## 外部模型推理接口

从 `motorhand_teleop.real_hand` 导入 `create_inference_session`；工厂发现设备但不使能，上下文进入时只读连接和预热，`start()` 才执行回零和使能。下面示例只保持当前实测关节姿态，用于说明调用顺序；真实模型应替换 `target`。

```python
import time
from motorhand_teleop.real_hand import create_inference_session

with create_inference_session() as session:
    session.start()  # 真机：手套回零并使能两侧
    state = session.read_wuji_state()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        cycle_start = time.monotonic()
        target = state.positions_rad  # 示例：保持实测姿态；模型输出应为 20 维 rad
        tick = session.step(target, produced_at_ns=state.received_ns)
        state = tick.hand2_state
        time.sleep(max(0.0, 0.02 - (time.monotonic() - cycle_start)))
```

`session.take_over()` 切换到手套主导；期间持续调用 `session.step()`。`session.resume_inference()` 切回推理，之后继续传 20 维目标。`produced_at_ns` 使用与 `time.monotonic_ns()` 同一时钟域的输入/目标最早生成时间；超期或未来时刻拒发。省略时间戳时，调用方必须保证目标新鲜，接口无法检测模型上游的旧缓存。控制周期和模型推理调度由调用方负责。

`tick.hand2_state` 为实际设备反馈，`tick.sent_positions_rad` 为已提交 Wuji 的最终目标，`tick.shadow_glove_positions_rad` 为手套目标；`tick.force_feedback` 包含反馈状态和名义指令，不等于实测接触力。关节顺序见 `motorhand_teleop.domain.WUJI_JOINTS`。键盘接管已由回放命令实现，模型应用可把自身按键事件接到上述两个方法。

## 配置与资产

| 路径 | 内容 |
| --- | --- |
| `hardware/encos_dexhand/configs/glove_mapping.json` | 现场电机轴序、零位、方向、比例、软限位及回零电流；重新标零后复核 |
| `config/retargeting/official/` | WujiRetarget 左右手配置和侧摆约束 |
| `config/feedback/max-torque-linear-right.json` | Wuji 设备身份、手套名义输出上限、斜率及几何有效期 |
| `config/feedback/contact-impedance-right.json` | 受阻阈值、虚拟阻抗、滤波和反射增益；不是 SDK kp/kd |
| `assets/training_models.json` | Hand2 与 vE.3.3 左右手 URDF/MJCF 清单及入口哈希 |
| `src/motorhand_teleop/real_hand/` | 两侧驱动、控制循环、反馈、录制/回放、推理会话和原生 SDK 边界 |
| `vendor/wuji-retargeting/` | 固定版本官方优化器和 Hand2 Beta1 仿真资产 |

训练可从资产清单选择 URDF/MJCF。vE.3.3 树形 URDF 不表达闭链；其 MJCF 动力学与接触参数未做实测标定，不能直接视为物理可信的接触训练模型。来源、许可和加载限制见 `THIRD_PARTY_NOTICES.md`。

## 验证

```bash
pixi run -e hardware test-glove-chain
pixi run -e hardware test-record-replay
pixi run -e hardware test-real-hand-native
pixi run mujoco-force-feedback
```

这些是离线/数值验证，不等于现场运动或人体力感验收。模拟力反馈入口使用假设参数，且不覆盖当前接触阻抗控制器；真实硬件需要单独验收。
