# Motorhand 手部力反馈遥操作

这个目录是 `Motorhand_FFB` 的手部适配层：手套读取 → vE3.3 → WujiRetarget → Wuji Hand 2 目标，以及手套回零、手套跟随和接触力反馈。

不包含机械臂、PICO 或 DAgger。项目本身包含 Motorhand 核心、native_control、WujiRetarget 资产和配置，不依赖 `/home/boen/hand/Motorhand_Teleop`。ROS2 节点可直接接入天机的标准 Wuji 手话题，但不依赖天机工作区 Python 包。

## 文件布局

| 路径 | 内容 |
|---|---|
| `native/` | C++ worker 和力反馈控制实现 |
| `motorhand_glove/node.py` | ROS2 话题适配器与 worker supervisor |
| `motorhand_glove/worker.py` | Python 只读备用 worker；不出力 |
| `motorhand_glove/feedback.py`、`viewer.py` | 早期 Python 反馈逻辑与 MuJoCo 只读可视化 |
| `config/motorhand_glove.yaml` | 默认右手运行参数 |
| `config/motorhand_feedback/` | 右手参数和左手模板 |
| `scripts/build_motorhand_glove.sh` | 构建 C++ worker |
| `scripts/setup_motorhand_glove.sh` | udev 安装和只读手套检查 |
| `scripts/view_motorhand_glove.sh` | 标零与只读可视化 |
| `udev/` | CANable2 访问规则 |

## 三种 worker 状态

- `force`：Wuji 手跟随本手套目标且受到阻碍时，手套提供接触阻抗力矩；只有 C++ worker 支持真实出力。
- `follow`：其他控制器主导 Wuji 手姿态时，手套跟随手部姿态，输出受力反馈限额约束。
- `idle`：没有新鲜手状态或目标时，释放手套输出。

worker 的 stdin/stdout 使用 JSON 行协议。ROS2 节点发布 `/<hand>/joint_commands`，并在启用出力时把 `/<side>_hand/joint_states` 与最终 `/<side>_hand/joint_commands` 转给 worker。

## 构建和运行

Pixi 环境固定使用 PyPI 的 `mujoco==4.0.0`，避免 conda MuJoCo 版本与 Python API 混用。

在项目根目录执行：

```bash
pixi install
bash tianji_teleop/scripts/build_motorhand_glove.sh
bash tianji_teleop/scripts/setup_motorhand_glove.sh --check
```

ROS2 适配器：

```bash
colcon build --base-paths tianji_teleop --packages-select motorhand_glove
source install/setup.bash
ros2 run motorhand_glove motorhand_glove --ros-args \
  --params-file tianji_teleop/config/motorhand_glove.yaml
```

默认 C++ worker 是：

```text
.cache/tianji-glove-native/motorhand_glove_native
```

参数文件中的 `root` 为空时使用当前项目根目录；也可设置 `MOTORHAND_FFB_ROOT` 或显式传入 `root`。

## 参数边界

右手的 `max-torque-linear-right.json` 和 `contact-impedance-right.json` 是当前天机右手的名义估计/命令上限，不是实测接触力标定。C++ worker 会核对限额文件中的 `wuji_serial_number`。

左手文件是模板，使用前必须替换序列号并重新做低负载硬件验证。禁止绕过序列号检查或把右手参数直接用于左手。
