# Motorhand_Teleop — sim

本分支保留原 `sim2.0` 的整合版本：外层手套交互仿真与完整 Motorhand_FFB 工程并列。两者各自带 Pixi 环境、配置和部分重复资产，根目录没有统一 `pixi.toml`。

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

## 本分支的三个运行入口

| 目录 | 功能 | 文档 |
|---|---|---|
| `sim/` | `glove_sim`：手套与虚拟物体交互、驱动阻抗、摩擦补偿、会话记录与复盘；支持 CANable2/G4 手套 | [手套交互仿真](sim/README.md) |
| `Motorhand_FFB/` | Wuji 直连控制、跟随、录制回放与力反馈；还保留自己的 `sim/` 动力学演示 | [FFB 工程](Motorhand_FFB/README.md) |
| `Motorhand_FFB/tianji_teleop/` | 已解耦天机 Python 包依赖的手部 ROS2 话题适配，仍需 ROS2 和下游手驱动 | [独立手部适配](Motorhand_FFB/tianji_teleop/README.md) |

根目录的 [tianji_teleop/](tianji_teleop/README.md) 是原天机接入代码的保留副本，仍依赖天机工作区；不要和 FFB 子项目里的独立适配器混用。

## 安装与离线检查

以下两段均从仓库根目录开始，分别进入对应工程：

```bash
cd sim
pixi install --frozen
pixi run smoke
pixi run test
pixi run sim --headless --duration 2
```

```bash
cd Motorhand_FFB
pixi install -e hardware
pixi run validate-assets
pixi run -e hardware build-control
pixi run -e hardware test-control-native
```

`Motorhand_FFB/sim/` 通过 FFB 环境的 `pixi run python -m sim ...` 进入；外层 `sim/` 通过自己的 `pixi run sim` 进入。二者的场景、控制方法、记录格式和验证范围不同，不能互换命令或把一套的验收结果用于另一套。

## 与 sim2.0 的主要区别

- 本分支环境和驱动/资产副本尚未合并；`sim2.0` 已改成根目录统一环境与外层共享资产。
- 本分支 FFB 保留 ROS2 适配代码和嵌入式仿真；`sim2.0` 的 FFB 以 Wuji SDK 直连为主，外层 `sim/` 负责手套物体交互。
- 本分支外层仿真已有驱动阻抗，但 Wuji 直连反馈仍走虚拟阻抗投影的纯力矩路径；不能把这两者都称为新版真机驱动阻抗。
- 本分支标定分散在外层及 `Motorhand_FFB/hardware/`，没有新版统一的 `configs/sim/`、`configs/ffb/` 分工。切分支后重新核对所用标定路径。
- 新版 `teleop-ui` 和 `wuji-driver-impedance-right.json` 属于 `sim2.0`，本分支不提供这些入口。

历史过程记录在 [chat_history/README.md](chat_history/README.md)，不作为当前配置或验收依据。真实设备操作请继续阅读各子项目的使能、回零及停止说明。
