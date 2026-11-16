# Motorhand_Teleop — sim2.0

当前仿真与真机力反馈工作区，原分支名 `sim3.0`。包含手套物体交互仿真、Motorhand → Wuji Hand 2 原生遥操作、受阻驱动阻抗和只读监视工具；全部使用仓库根目录的同一个 Pixi 环境。

## 分支区别与选择

以下名称以本次整理后的远程分支为准；切换分支时，以该分支自己的配置和命令为准。

| 分支 | 定位与目录 | 力反馈和集成方式 | 环境与适用场景 |
|---|---|---|---|
| [main](https://github.com/hbw1010/Motorhand_Teleop/tree/main) | 原 `cpp`；核心源码直接位于根目录 `src/motorhand_teleop/` | C++ Wuji 直连、双向跟随、录制回放及接管；接触估计投影为 MIT 纯力矩前馈 | 根目录 Pixi，真机任务用 `-e hardware`；用于查看原生控制基线 |
| [sim](https://github.com/hbw1010/Motorhand_Teleop/tree/sim) | 原 `sim2.0`；外层 `sim/` 与完整 `Motorhand_FFB/` 并列，各有环境和资产 | 外层为手套与虚拟物体交互；FFB 保留 Wuji 控制、手部 ROS2 适配及其独立仿真 | 分别进入 `sim/` 或 `Motorhand_FFB/`；用于复现整合前的双工程版本 |
| [sim2.0](https://github.com/hbw1010/Motorhand_Teleop/tree/sim2.0) | 原 `sim3.0`；根目录统一 Pixi，共享驱动、模型，sim/ffb 标定分开 | Wuji 直连默认受阻代理＋MIT 驱动阻抗＋自由空间摩擦补偿；含 G4 支持及遥操监视器 | 从仓库根目录运行任务；用于当前仿真与真机反馈开发 |
| [tianji_teleop](https://github.com/hbw1010/Motorhand_Teleop/tree/tianji_teleop) | 在原 C++ 基线上增加 `tianji_teleop/` | 天机 ROS2 JointState 话题＋C++ worker；手状态来自天机，配置包含 200 Hz 目标输出 | 需要天机工作区依赖；用于天机系统中的手部接入 |

`sim2.0` 现在指原 `sim3.0` 的代码；整理前叫 `sim2.0` 的版本现在是 `sim`。旧 `main`、`basic_code` 已删除，原 `cpp` 改名为当前默认分支 `main`。提交历史中的旧分支名保留原文。

分支名不代表硬件验收等级。Wuji 电流、软件阻抗和 SDK 名义 Nm 都不能直接解释为实测接触力；设备身份、零位、方向和限额应以当前分支配置为准。

## 本分支相对 sim 的变化

- 合并 Pixi 工作区：`sim/` 与 `Motorhand_FFB/` 任务统一在根目录执行，测试拆成 `test-sim` / `test-ffb`。
- 驱动与模型移到外层共享；仿真和 FFB 的现场标定分别存放在 `hardware/encos_dexhand/configs/sim/`、`configs/ffb/`。
- FFB 移除天机 ROS2 运行器和重复的手套仿真目录；保留历史天机配置作为来源，当前真机读取 `config/feedback/wuji-*.json`。
- `wuji-force-contact` 默认使用受阻代理＋MIT 驱动阻抗，并加入自由空间摩擦补偿；旧纯力矩模式和线性负载反射仍保留。
- 独立力反馈入口增加固定零位参考：手套模型零位对应 Wuji 全零目标；增加 `glove-wuji-viewer` 和 `teleop-ui` 及反馈链路遥测。

## 文档分工

| 文档 | 负责内容 |
|---|---|
| [Motorhand_FFB/README.md](Motorhand_FFB/README.md) | Wuji 真机遥操、回零、反馈模式、只读观察与监视 |
| [sim/README.md](sim/README.md) | 手套与虚拟物体交互、仿真 UI、记录、回放及仿真反馈 |
| [config/feedback/README.md](config/feedback/README.md) | 当前真机反馈参数来源、生效范围和调整记录 |
| [driver-impedance-native.md](config/feedback/driver-impedance-native.md) | 原生代理逆映射、MIT 阻抗、摩擦和提交保护 |
| [hardware/encos_dexhand/README.md](hardware/encos_dexhand/README.md) | Encos SDK、原始单位、轴映射与双份标定 |
| [Motorhand_FFB/scripts/README.md](Motorhand_FFB/scripts/README.md) | PlotJuggler 只读旁路，与原生监视 UI 的区别 |
| [历史反馈资产](Motorhand_FFB/config/motorhand_feedback/README.md) | 保留参数与左手模板，说明与当前生效配置的区别 |
| [模型资产](assets/hand_motor_ve33/hand_motor_ve33_R/README.md) | CAD/URDF/MJCF 坐标、闭链和来源，不是现场运行手册 |

## 环境和离线验证

Linux x86_64，Python 3.12，PyPI `mujoco==4.0.0`；图形界面需要可用的显示环境。

```bash
# 在本仓库根目录执行
pixi install --frozen
pixi run validate-assets
pixi run smoke
pixi run test-sim
pixi run test-ffb
pixi run test-control-native
```

本分支不需要 `-e hardware`。`test-control-native` 自动构建；单独执行 `test-real-hand-native` 前先运行 `build-real-hand`。离线测试不连接真实设备，不等于实际接触力或穿戴手感验收。

## 常用入口与设备行为

| 命令（根目录） | 用途与设备行为 |
|---|---|
| `pixi run sim` | 假手套＋虚拟物体交互，无真机输出 |
| `pixi run teleop` | 真实手套驱动虚拟场景；反馈默认关闭，开启后手套会出力 |
| `pixi run glove-wuji-viewer` | 读手套，显示手套与 Wuji 模型，不控制 Wuji 真机 |
| `pixi run wuji-discover` | 只读发现 Wuji |
| `pixi run wuji-teleop` | 手套控制真实 Wuji；手套只读 |
| `pixi run wuji-force-contact` | 两侧回零后跟随，默认驱动阻抗；会使能两侧 |
| `pixi run wuji-force-linear` | 两侧回零后，按 Wuji 电流负载占比产生手套前馈 |
| `pixi run teleop-ui` | 只读 UDP 遥测，观察接触、代理位置、输出限制与曲线 |
| `pixi run force-ui` | 旧版只读反馈曲线 |

真机命令须匹配设备身份和固件；当前右手完整示例见 [FFB README](Motorhand_FFB/README.md)。控制端通过 `--curves-port 8765` 发布遥测；关闭查看器不会停止控制。

## 两条力反馈链路

- **仿真**：手套姿态 → MuJoCo 虚拟物体接触/代理表面 → 手套反馈。参数在 `sim/config/default.yaml`，硬度来自虚拟物体设置。
- **Wuji 真机**：手套闭链任务点 → Wuji 目标；Wuji 电流、速度和跟踪误差 → 受阻检测 → 局部手套代理位置 → MIT PD。参数在 `config/feedback/`，默认硬度 0.3 是人工设置。

`wuji-force-contact --feedback-mode torque` 使用旧虚拟阻抗投影前馈；`wuji-force-linear` 使用线性电流负载代理。它们与默认驱动阻抗的增益、斜率和输出字段语义不同。名义力矩、实测电流、仿真接触力在记录中分别表达。

重新标零时，`zero-glove` 修改 sim 标定，FFB 的 `encos-zero-*` 修改 FFB 标定；两套文件不会自动同步。`glove-wuji-viewer` 默认观察 sim 标定，真机 FFB 控制默认读取 ffb 标定。
