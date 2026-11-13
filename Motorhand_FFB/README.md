# Motorhand_FFB

Motorhand 外骨骼手套到 Wuji Hand 2 的手部遥操作项目，使用工作区外层共享驱动、模型和配置：

- 12 电机手套读取、vE3.3 闭链和 WujiRetarget；
- Wuji Hand 2 目标发布、手套回零、跟随和力反馈；
- C++ 实时控制、MuJoCo 离线仿真及只读可视化。

天机 ROS2 适配代码已移除。当前 Wuji 反馈参数按用户要求迁移自保留的天机反馈 JSON，并以 `wuji-` 命名；设备序列号使用当前 Wuji，原天机文件保留作为来源。

## 目录

| 路径 | 用途 |
|---|---|
| `src/motorhand_teleop/` | 手套运动学、重定向、真机控制、反馈和可视化 |
| `../hardware/encos_dexhand/configs/ffb/` | FFB 手套轴映射、零位和软限位 |
| `../hardware/encos_dexhand/configs/sim/` | 外层仿真使用的独立手套标定 |
| `../assets/hand_motor_ve33/` | 共用的左右手套模型 |
| `../config/feedback/` | 共用的右手反馈配置 |
| `config/retargeting/official/` | WujiRetarget 左右手配置 |
| `vendor/wuji-retargeting/` | 固定版本 Wuji Hand 2 重定向与模型资产 |

## 安装与构建

在工作区根目录运行；Pixi 环境固定使用 PyPI 的 `mujoco==4.0.0`。

```bash
pixi install
pixi run validate-assets
pixi run build-control
```

## 只读可视化

```bash
pixi run glove-wuji-viewer
```

查看器默认读取外层 `zero-glove` 使用的 `configs/sim/glove_mapping.json`，复用力控仿真的 12 轴零位、方向、限位和闭链求解；Wuji 任务点使用同一手套姿态。可通过 `--mapping` 显式选择其他标定。查看器不连接 Wuji 真机或使能手套出力；G4 网关断开时 SDK 会发送零增益卸力报文。真实设备需具备访问权限及图形显示，参数可通过 `pixi run glove-wuji-viewer --help` 查看。

手套按相同输入对齐的是仿真未碰撞冻结的目标姿态；力控仿真中的物体接触、力反馈和碰撞冻结不会在此查看器复现。Wuji 姿态经过重定向，手套电机零位不意味着 Wuji 各轴均为零。

## 右手真机力反馈遥操

在工作区根目录运行；以下命令使用 G4 网关及当前右手设备：

```bash
MOTORHAND_GLOVE_DRIVER=2 pixi run wuji-force-contact \
  --hand right \
  --wuji-serial WH2KA01260813020 \
  --approved-firmware 2.13.1 \
  --rate 50
```

程序先检查双方反馈，再依次将手套回到 FFB 标定零位、Wuji 回到 SDK 全零位（实测容差 0.03 rad），完成后才开始跟随；按 `Ctrl+C` 停止。Wuji 可被发现不代表手套可用，网关必须收到全部 12 轴反馈。此入口默认使用接触估计驱动的MIT阻抗和自由空间摩擦补偿；`--feedback-mode torque` 保留旧纯前馈方式。人工硬度默认0.3（`--feedback-hardness`），不是物体硬度测量。控制细节见 `../config/feedback/driver-impedance-native.md`。

独立 `force-contact` / `force-linear` 使用固定零位参考：目标为当前重定向结果减去手套模型零位的重定向结果，再按实际关节限位裁剪并平滑接入。手套归零时目标回到 Wuji 全零，不再随累计运动衰减启动偏置；这属于零位对齐的相对姿态映射。推理/回放接管的保持抓握语义不变。

设备绑定及名义输出参数在 `../config/feedback/wuji-max-torque-linear-right.json`，接触估计参数在 `../config/feedback/wuji-contact-impedance-right.json`。2026-11-12 经用户确认，将此次新标的 sim 零位和配套限位同步到 `configs/ffb/` 的两份标定，方向、比例和力控参数不变。两套文件仍独立，后续 `zero-glove` 不会自动同步 FFB；更换设备或重新标零后需再次核对。

## 反馈与安全

真机控制使用工作区的 `config/feedback/` 配置。电流、SDK 名义力矩和软件估计不是实测接触力；首次上手必须低负载，确认方向和独立急停。两套手套标定分别保存在 `hardware/encos_dexhand/configs/sim/` 和 `hardware/encos_dexhand/configs/ffb/`，不可未经核对互换。

历史天机反馈参数及原配置保留在各自的 `tianji_teleop/config/` 和 `config/motorhand_feedback/`。左手模板不是已完成的硬件标定。更多安全约束见 `CONTEXT.md`。

## 离线验证

```bash
pixi run validate-assets
pixi run test-real-hand-native
pixi run test-control-native
```

这些检查不等于人体穿戴和真实接触力验收。真机控制可能使设备运动，需核实设备身份、标定、限位及独立急停。

## 手套本体 MuJoCo 交互仿真

手套抓握虚拟物体的仿真位于工作区外层的 `sim/`，运行方式和真机安全说明见 `../sim/README.md`。Wuji Hand 2 离线仿真仍保留在 `src/motorhand_teleop/mujoco_sim.py`。
